from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Annotated, Any

import cv2
import numpy as np
import onnxruntime as ort
import pyrealsense2 as rs
import tyro
from tyro.conf import arg
from datetime import datetime

# ==============================================================================
# DATA MODELS
# ==============================================================================


@dataclass(slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = ""
    coeffs: list[float] = field(default_factory=list)


@dataclass(slots=True)
class RGBDFrame:
    frame_id: int
    timestamp: float
    color_bgr: np.ndarray
    depth_raw: np.ndarray
    depth_aligned_to_color: np.ndarray
    depth_scale_m: float
    color_intrinsics: CameraIntrinsics
    depth_intrinsics: CameraIntrinsics
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# REALSENSE SOURCE (LIVE & BAG PLAYBACK)
# ==============================================================================


def _extract_intrinsics(profile) -> CameraIntrinsics:
    intr = profile.as_video_stream_profile().get_intrinsics()
    return CameraIntrinsics(
        width=int(intr.width),
        height=int(intr.height),
        fx=float(intr.fx),
        fy=float(intr.fy),
        cx=float(intr.ppx),
        cy=float(intr.ppy),
        model=str(intr.model),
        coeffs=[float(v) for v in intr.coeffs],
    )


class RealSenseSource:
    def __init__(
        self,
        bag_path: Path | None = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 15,
        serial: str | None = None,
    ):
        self.bag_path = bag_path
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial

        self.pipeline: rs.pipeline | None = None
        self.profile: rs.pipeline_profile | None = None
        self.align: rs.align | None = None
        self.depth_scale: float = 0.001
        self.frame_id: int = 0

    def start(self) -> None:
        config = rs.config()

        if self.bag_path:
            if not self.bag_path.is_file():
                raise FileNotFoundError(f"Bag file not found: {self.bag_path}")
            config.enable_device_from_file(str(self.bag_path), repeat_playback=False)
        else:
            if self.serial:
                config.enable_device(self.serial)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)

        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(config)

        if self.bag_path:
            playback = self.profile.get_device().as_playback()
            playback.set_real_time(False)

        # align depth to color optical frame
        self.align = rs.align(rs.stream.color)
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())

        if not self.bag_path:
            for _ in range(15):
                self.pipeline.wait_for_frames()

    def read(self) -> RGBDFrame | None:
        if self.pipeline is None or self.align is None:
            raise RuntimeError("RealSense source is not started")

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            return None

        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        aligned_depth_frame = aligned.get_depth_frame()
        raw_depth_frame = frames.get_depth_frame()

        if not color_frame or not aligned_depth_frame or not raw_depth_frame:
            return None

        self.frame_id += 1
        color_rgb = np.asanyarray(color_frame.get_data())
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        depth_raw = np.asanyarray(raw_depth_frame.get_data()).copy()
        depth_aligned = np.asanyarray(aligned_depth_frame.get_data()).copy()
        device = self.profile.get_device()

        return RGBDFrame(
            frame_id=self.frame_id,
            timestamp=time.time(),
            color_bgr=color_bgr,
            depth_raw=depth_raw,
            depth_aligned_to_color=depth_aligned,
            depth_scale_m=self.depth_scale,
            color_intrinsics=_extract_intrinsics(color_frame.profile),
            depth_intrinsics=_extract_intrinsics(raw_depth_frame.profile),
            metadata={
                "source": "bag" if self.bag_path else "realsense",
                "device_name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number) if not self.bag_path else "bag",
            },
        )

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


# ==============================================================================
# ONNX SEGMENTATION MODEL
# ==============================================================================


class ONNXModel:
    def __init__(self, path: Path):
        # providers: prioritize OpenVINO if installed, fallback to CPU
        available_providers = ort.get_available_providers()
        providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        selected_providers = [p for p in providers if p in available_providers]

        self.session = ort.InferenceSession(str(path), providers=selected_providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        self.h = int(input_shape[2]) if isinstance(input_shape[2], int) else 640
        self.w = int(input_shape[3]) if isinstance(input_shape[3], int) else 640

        self.output_names = [out.name for out in self.session.get_outputs()]
        if "labels" not in self.output_names or "masks" not in self.output_names:
            raise RuntimeError(f"Expected model outputs 'labels' and 'masks', got {self.output_names}")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def infer(self, bgr: np.ndarray, threshold: float) -> tuple[np.ndarray | None, float, float]:
        t0 = time.perf_counter()

        # preprocessing
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.w, self.h)), cv2.COLOR_BGR2RGB)
        x = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        x = np.ascontiguousarray((x - self.mean) / self.std)[None, ...].astype(np.float32)

        # inference
        outputs = self.session.run(None, {self.input_name: x})
        out_dict = dict(zip(self.output_names, outputs))

        # logits -> probs
        logits = out_dict["labels"][0, :, :-1].astype(np.float32, copy=False)
        probs = 1.0 / (1.0 + np.exp(-logits))

        index = int(np.argmax(probs))
        score = float(probs.reshape(-1)[index])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if score < threshold:
            return None, score, elapsed_ms

        query = int(index // logits.shape[1])
        mask_logits = out_dict["masks"][0, query].astype(np.float32, copy=False)

        # resize mask back to source frame resolution
        mask = cv2.resize(mask_logits, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR) > 0
        return mask, score, elapsed_ms


# ==============================================================================
# DUMMY SEGMENTATION MODEL (FOR DEBUGGING)
# ==============================================================================


class DummyModel:
    def __init__(self, *args, **kwargs):
        print(">>> Using DummyModel (centered rectangle 50% of resolution)")

    def infer(self, bgr: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray | None, float, float]:
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]

        mask = np.zeros((h, w), dtype=bool)

        # Прямоугольник по центру размером W/2 на H/2
        box_w = w // 2
        box_h = h // 2
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        mask[y1:y2, x1:x2] = True

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return mask, 1.0, elapsed_ms

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================


def main(
    model_path: Annotated[Path, arg(aliases=["-m"]),] = Path("results/segmentation/rfdetr-seg-nano.onnx"),
    output_dir: Annotated[Path | None, arg(aliases=["-o"])] = None,
    bag_path: Annotated[Path | None, arg(aliases=["-b"]),] = None,
    threshold: Annotated[float, arg(aliases=["-t"]),] = 0.5,
    max_frames: Annotated[int, arg(aliases=["-n"]),] = 20,
):
    """Capture frames from RealSense or BAG file, run ONNX segmentation, and save RGB-D dataset.

    Args:
        model_path: Path to ONNX model.
        output_dir: Root directory for saving 'rgb', 'depth_raw', and 'mask' subfolders.
        bag_path: Path to .bag file (if not set, live camera is used).
        threshold: Confidence threshold for mask detection.
        max_frames: Max frames to process (0 = infinite / until bag ends).
    """
    if output_dir is None:
        output_dir = Path(f"new_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if not model_path.is_file():
        print(f"Error: Model file '{model_path}' not found.", file=sys.stderr)
        sys.exit(1)

    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth_raw"
    mask_dir = output_dir / "mask"

    for d in (rgb_dir, depth_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(">>> Loading ONNX model...")
    # model = ONNXModel(model_path)
    model = DummyModel()

    camera = RealSenseSource(bag_path=bag_path)
    print(f">>> Starting camera source ({'BAG: ' + str(bag_path) if bag_path else 'Live USB'})...")
    camera.start()

    saved_count = 0
    detected_depth_scale = camera.depth_scale

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print(">>> End of stream / camera timeout.")
                break

            mask, score, ms = model.infer(frame.color_bgr, threshold=threshold)

            if mask is None:
                print(f"[SKIP] Frame {frame.frame_id}: low confidence ({score:.3f})")
                continue

            stem = f"frame_{saved_count:05d}"

            # save rgb
            cv2.imwrite(str(rgb_dir / f"{stem}.png"), frame.color_bgr)
            # save aligned depth (uint16 raw values)
            cv2.imwrite(str(depth_dir / f"{stem}.png"), frame.depth_aligned_to_color)
            # save binary mask (0 or 255)
            cv2.imwrite(str(mask_dir / f"{stem}.png"), (mask.astype(np.uint8) * 255))

            saved_count += 1
            print(f"[{saved_count}] Saved {stem} | Score: {score:.3f} | Infer: {ms:.1f} ms")

            if max_frames > 0 and saved_count >= max_frames:
                print(f">>> Reached target frame count: {max_frames}")
                break

    finally:
        camera.stop()

    print(f"\n>>> Success! Saved {saved_count} triplets to '{output_dir}'.")
    print(f">>> Detected depth scale: {detected_depth_scale:.6f} m")
    print("\nNext step: Run your rendering script:")
    print(f"python your_render_script.py -d {output_dir} -o render_vis --depth-scale-m {detected_depth_scale:.6f} --debug")


if __name__ == "__main__":
    tyro.cli(main)