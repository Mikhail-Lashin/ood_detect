#!/usr/bin/env python3
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
import threading
import time
from typing import Annotated, Any

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pyrealsense2 as rs
import tyro
from tyro.conf import arg
import uvicorn

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
    model: str = "BrownConrady"
    coeffs: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])


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
# 3D POINTCLOUD PROCESSING
# ==============================================================================

_NEIGHBOR_3D = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if dx or dy or dz
)


def resize_depth_context(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, CameraIntrinsics]:
    target_h, target_w = target_shape
    if depth.shape == target_shape:
        return depth, intrinsics
    scale_x = target_w / depth.shape[1]
    scale_y = target_h / depth.shape[0]
    resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    adjusted = CameraIntrinsics(
        width=target_w,
        height=target_h,
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        cx=intrinsics.cx * scale_x,
        cy=intrinsics.cy * scale_y,
        model=intrinsics.model,
        coeffs=list(intrinsics.coeffs),
    )
    return resized, adjusted


def unproject_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
    max_valid_depth_mm: float = 10000.0,
) -> np.ndarray:
    depth, intrinsics = resize_depth_context(depth, intrinsics, mask.shape)
    valid = mask & (depth > 0)
    if np.issubdtype(depth.dtype, np.integer):
        valid &= depth < np.iinfo(depth.dtype).max
    ys, xs = np.where(valid)
    if not len(xs):
        return np.empty((0, 3), dtype=np.float64)
    z_mm = depth[ys, xs].astype(np.float64) * depth_scale_m * 1000.0
    finite = np.isfinite(z_mm) & (z_mm > 0)
    if max_valid_depth_mm > 0:
        finite &= z_mm <= max_valid_depth_mm
    xs = xs[finite].astype(np.float64)
    ys = ys[finite].astype(np.float64)
    z_mm = z_mm[finite]
    x_mm = (xs - intrinsics.cx) * z_mm / intrinsics.fx
    y_mm = (ys - intrinsics.cy) * z_mm / intrinsics.fy
    return np.column_stack((x_mm, y_mm, z_mm))


def filter_pointcloud_outliers(
    points: np.ndarray,
    voxel_mm: float = 25.0,
    min_component_ratio: float = 0.03,
) -> tuple[np.ndarray, dict]:
    raw_count = int(len(points))
    if raw_count < 300 or voxel_mm <= 0:
        return points, {"point_outlier_removed_px": 0}

    z_bins = np.floor(points[:, 2] / float(voxel_mm)).astype(np.int32)
    z_unique, z_inverse, z_counts = np.unique(z_bins, return_inverse=True, return_counts=True)
    z_keep_bins = np.zeros(len(z_unique), dtype=bool)
    start = 0
    best_start, best_stop, best_count = 0, 0, 0
    for stop in range(1, len(z_unique) + 1):
        if stop < len(z_unique) and z_unique[stop] == z_unique[stop - 1] + 1:
            continue
        total = int(z_counts[start:stop].sum())
        if total > best_count:
            best_start, best_stop, best_count = start, stop, total
        start = stop
    z_keep_bins[best_start:best_stop] = True
    depth_keep = z_keep_bins[z_inverse]

    if int(depth_keep.sum()) >= max(120, int(0.50 * raw_count)):
        points = points[depth_keep]

    voxels = np.floor((points - np.min(points, axis=0)) / float(voxel_mm)).astype(np.int32)
    unique, inverse, voxel_counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    occupied = {tuple(v): i for i, v in enumerate(unique.tolist())}
    visited = np.zeros(len(unique), dtype=bool)
    keep_voxels = np.zeros(len(unique), dtype=bool)
    point_count = int(len(points))
    min_points = max(40, int(point_count * float(min_component_ratio)))
    best_component: list[int] = []
    best_count = 0
    for start in range(len(unique)):
        if visited[start]:
            continue
        stack = [start]
        component = []
        points_in_component = 0
        visited[start] = True
        while stack:
            current = stack.pop()
            component.append(current)
            points_in_component += int(voxel_counts[current])
            vx, vy, vz = unique[current]
            for dx, dy, dz in _NEIGHBOR_3D:
                nxt = occupied.get((int(vx + dx), int(vy + dy), int(vz + dz)))
                if nxt is not None and not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        if points_in_component > best_count:
            best_component = component
            best_count = points_in_component
        if points_in_component >= min_points:
            keep_voxels[component] = True
    if best_component:
        keep_voxels[best_component] = True
    keep = keep_voxels[inverse]
    if int(keep.sum()) < max(120, int(0.50 * point_count)):
        keep = np.ones(point_count, dtype=bool)
    filtered = points[keep]
    return filtered, {"point_outlier_removed_px": raw_count - int(len(filtered))}


def clean_mask_with_depth_filtering(
    bin_mask: np.ndarray,
    depth: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
    voxel_mm: float = 25.0,
) -> tuple[np.ndarray, dict]:
    points = unproject_mask(bin_mask, depth, depth_scale_m, intrinsics)
    if len(points) < 300:
        return bin_mask, {"point_outlier_removed_px": 0}

    filtered_points, stats = filter_pointcloud_outliers(points, voxel_mm=voxel_mm)
    clean_mask = np.zeros_like(bin_mask, dtype=bool)
    if len(filtered_points) > 0:
        u = np.round(filtered_points[:, 0] * intrinsics.fx / filtered_points[:, 2] + intrinsics.cx).astype(int)
        v = np.round(filtered_points[:, 1] * intrinsics.fy / filtered_points[:, 2] + intrinsics.cy).astype(int)
        valid_uv = (u >= 0) & (u < clean_mask.shape[1]) & (v >= 0) & (v < clean_mask.shape[0])
        clean_mask[v[valid_uv], u[valid_uv]] = True
    else:
        clean_mask = bin_mask
    return clean_mask, stats


def crop_to_square(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_size: int = 512,
    pad_ratio: float = 0.05,
) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min + 1
    h = y_max - y_min + 1

    side = int(max(w, h) * (1.0 + 2 * pad_ratio))
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side

    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - image.shape[0])
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - image.shape[1])

    if any((pad_top, pad_bottom, pad_left, pad_right)):
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top
    else:
        padded = image

    crop = padded[y1:y2, x1:x2]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)


# ==============================================================================
# REALSENSE SOURCE
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
    ):
        self.bag_path = bag_path
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline: rs.pipeline | None = None
        self.profile: rs.pipeline_profile | None = None
        self.align: rs.align | None = None
        self.depth_scale: float = 0.001
        self.frame_id: int = 0

    def start(self) -> None:
        config = rs.config()
        if self.bag_path:
            config.enable_device_from_file(str(self.bag_path), repeat_playback=True)
        else:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)

        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(config)

        if self.bag_path:
            playback = self.profile.get_device().as_playback()
            playback.set_real_time(False)
        else:
            # set high density preset for live hardware
            depth_sensor = self.profile.get_device().first_depth_sensor()
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(
                    rs.option.visual_preset,
                    float(rs.rs400_visual_preset.high_density),
                )

        self.align = rs.align(rs.stream.color)
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())

        if not self.bag_path:
            for _ in range(15):
                self.pipeline.wait_for_frames()

    def read(self) -> RGBDFrame | None:
        if self.pipeline is None or self.align is None:
            return None
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
        color_bgr = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)
        depth_raw = np.asanyarray(raw_depth_frame.get_data()).copy()
        depth_aligned = np.asanyarray(aligned_depth_frame.get_data()).copy()

        return RGBDFrame(
            frame_id=self.frame_id,
            timestamp=time.time(),
            color_bgr=color_bgr,
            depth_raw=depth_raw,
            depth_aligned_to_color=depth_aligned,
            depth_scale_m=self.depth_scale,
            color_intrinsics=_extract_intrinsics(color_frame.profile),
            depth_intrinsics=_extract_intrinsics(raw_depth_frame.profile),
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
        providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        selected = [p for p in providers if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(path), providers=selected or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        self.h = int(input_shape[2]) if isinstance(input_shape[2], int) else 640
        self.w = int(input_shape[3]) if isinstance(input_shape[3], int) else 640

        self.output_names = [out.name for out in self.session.get_outputs()]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def infer(self, bgr: np.ndarray, threshold: float) -> tuple[np.ndarray | None, float]:
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.w, self.h)), cv2.COLOR_BGR2RGB)
        x = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        x = np.ascontiguousarray((x - self.mean) / self.std)[None, ...].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: x})
        out_dict = dict(zip(self.output_names, outputs))

        logits = out_dict["labels"][0, :, :-1].astype(np.float32, copy=False)
        probs = 1.0 / (1.0 + np.exp(-logits))
        index = int(np.argmax(probs))
        score = float(probs.reshape(-1)[index])

        if score < threshold:
            return None, score

        query = int(index // logits.shape[1])
        mask_logits = out_dict["masks"][0, query].astype(np.float32, copy=False)
        mask = cv2.resize(mask_logits, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR) > 0
        return mask, score


# ==============================================================================
# INTERACTIVE CAPTURE SERVICE (SPLIT: NORMAL / ANOMALOUS)
# ==============================================================================


class CaptureService:
    def __init__(
        self,
        camera: RealSenseSource,
        model: ONNXModel,
        output_dir: Path,
        threshold: float = 0.5,
        target_size: int = 512,
        voxel_mm: float = 25.0,
        debug: bool = True,
    ):
        self.camera = camera
        self.model = model
        self.output_dir = output_dir
        self.threshold = threshold
        self.target_size = target_size
        self.voxel_mm = voxel_mm
        self.debug = debug

        # prepare directory tree for both classes
        self.paths: dict[str, dict[str, Path]] = {}
        for category in ("normal", "anomalous"):
            cat_root = output_dir / category
            self.paths[category] = {
                "rgb": cat_root / "rgb",
                "depth": cat_root / "depth_raw",
                "mask": cat_root / "mask",
                "rgb_seg": cat_root / "rgb_seg",
                "depth_seg": cat_root / "depth_seg",
                "vis": cat_root / "vis",
            }
            for folder in self.paths[category].values():
                if folder.name == "vis" and not self.debug:
                    continue
                folder.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.latest_frame: RGBDFrame | None = None
        self.latest_mask: np.ndarray | None = None
        self.preview_jpeg: bytes = b""
        self.counts = {"normal": 0, "anomalous": 0}
        self.running: bool = False

    def loop(self):
        """Background thread: reads stream, runs inference, creates combined preview."""
        self.running = True
        self.camera.start()

        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            mask, _ = self.model.infer(frame.color_bgr, threshold=self.threshold)

            # compose low-latency side-by-side preview
            h, w = 240, 320
            p_rgb = cv2.resize(frame.color_bgr, (w, h))

            p_seg = np.full_like(frame.color_bgr, 255)
            if mask is not None and np.any(mask):
                p_seg[mask] = frame.color_bgr[mask]
            p_seg = cv2.resize(p_seg, (w, h))

            p_depth = np.full_like(frame.color_bgr, 255)
            depth_valid = frame.depth_aligned_to_color > 0
            if mask is not None and np.any(mask):
                depth_valid &= mask

            if np.any(depth_valid):
                z_vals = frame.depth_aligned_to_color[depth_valid].astype(np.float32)
                z_min, z_max = z_vals.min(), z_vals.max()
                z_norm = (z_vals - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z_vals)
                colors_rgba = plt.cm.Blues(0.25 + 0.75 * z_norm)
                colors_bgr = (colors_rgba[:, :3] * 255)[:, ::-1].astype(np.uint8)
                p_depth[depth_valid] = colors_bgr
            p_depth = cv2.resize(p_depth, (w, h))

            combined = np.hstack([p_rgb, p_seg, p_depth])
            _, jpeg = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            with self.lock:
                self.latest_frame = frame
                self.latest_mask = mask
                self.preview_jpeg = jpeg.tobytes()

    def snapshot(self, category: str) -> dict[str, Any]:
        """Trigger snapshot into specified category (normal / anomalous)."""
        if category not in ("normal", "anomalous"):
            return {"status": "error", "message": f"Unknown category: {category}"}

        with self.lock:
            frame = self.latest_frame
            mask = self.latest_mask

        if frame is None or mask is None or not np.any(mask):
            return {"status": "error", "message": "No valid object in frame"}

        clean_mask, _ = clean_mask_with_depth_filtering(
            bin_mask=mask,
            depth=frame.depth_aligned_to_color,
            depth_scale_m=frame.depth_scale_m,
            intrinsics=frame.color_intrinsics,
            voxel_mm=self.voxel_mm,
        )

        ys, xs = np.where(clean_mask)
        if len(xs) == 0:
            return {"status": "error", "message": "3D filtering eliminated all object points"}

        cat_paths = self.paths[category]
        stem = f"frame_{self.counts[category]:05d}"
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        # segmented rgb on white
        rgb_white = np.full_like(frame.color_bgr, 255)
        rgb_white[clean_mask] = frame.color_bgr[clean_mask]

        # segmented depth colormap
        depth_valid = clean_mask & (frame.depth_aligned_to_color > 0)
        depth_white = np.full_like(frame.color_bgr, 255)
        if np.any(depth_valid):
            z_vals = frame.depth_aligned_to_color[depth_valid].astype(np.float32)
            z_min, z_max = z_vals.min(), z_vals.max()
            z_norm = (z_vals - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z_vals)
            colors_rgba = plt.cm.Blues(0.25 + 0.75 * z_norm)
            depth_white[depth_valid] = (colors_rgba[:, :3] * 255)[:, ::-1].astype(np.uint8)

        # crop squares
        rgb_square = crop_to_square(rgb_white, bbox, target_size=self.target_size)
        depth_square = crop_to_square(depth_white, bbox, target_size=self.target_size)

        # save artifacts
        cv2.imwrite(str(cat_paths["rgb"] / f"{stem}.png"), frame.color_bgr)
        cv2.imwrite(str(cat_paths["depth"] / f"{stem}.png"), frame.depth_aligned_to_color)
        cv2.imwrite(str(cat_paths["mask"] / f"{stem}.png"), (clean_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(cat_paths["rgb_seg"] / f"{stem}.png"), rgb_square)
        cv2.imwrite(str(cat_paths["depth_seg"] / f"{stem}.png"), depth_square)

        if self.debug:
            combined = np.hstack([rgb_square, depth_square])
            cv2.imwrite(str(cat_paths["vis"] / f"{stem}_vis.png"), combined)

        self.counts[category] += 1
        return {
            "status": "ok",
            "category": category,
            "normal_count": self.counts["normal"],
            "anomalous_count": self.counts["anomalous"],
            "stem": stem,
        }


# ==============================================================================
# FASTAPI WEB INTERFACE
# ==============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RGB-D Interactive Capture</title>
    <style>
        body {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 960px;
            margin-bottom: 12px;
        }
        .title { font-size: 18px; font-weight: 600; }
        .counters {
            display: flex;
            gap: 10px;
        }
        .counter-badge {
            font-size: 14px;
            padding: 6px 14px;
            border-radius: 4px;
            border: 1px solid #3c3c3c;
            background: #252526;
        }
        .counter-badge.normal { border-left: 4px solid #388a34; }
        .counter-badge.anomalous { border-left: 4px solid #c93838; }
        .stream-container {
            border: 1px solid #3c3c3c;
            border-radius: 6px;
            overflow: hidden;
            background: #000;
        }
        img { display: block; width: 960px; height: 240px; }
        .labels {
            display: flex;
            width: 960px;
            justify-content: space-around;
            font-size: 13px;
            color: #858585;
            margin-top: 6px;
        }
        .controls { margin-top: 18px; display: flex; gap: 16px; }
        button {
            border: none;
            padding: 12px 28px;
            font-size: 15px;
            font-weight: 600;
            border-radius: 4px;
            cursor: pointer;
            transition: opacity 0.15s, transform 0.1s;
        }
        button:active { transform: scale(0.98); }
        .btn-normal { background: #388a34; color: #fff; }
        .btn-normal:hover { background: #43a047; }
        .btn-anomalous { background: #c93838; color: #fff; }
        .btn-anomalous:hover { background: #e53935; }
        .hint { color: #6e6e6e; font-size: 12px; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">Live RGB-D Anomaly Capture</div>
        <div class="counters">
            <div class="counter-badge normal" id="cnt-norm">Normal: 0</div>
            <div class="counter-badge anomalous" id="cnt-anom">Anomalous: 0</div>
        </div>
    </div>
    <div class="stream-container">
        <img src="/stream" alt="Live Stream" />
    </div>
    <div class="labels">
        <span>Raw RGB</span>
        <span>Segmented RGB</span>
        <span>Depth Map</span>
    </div>
    <div class="controls">
        <button class="btn-normal" onclick="captureSnapshot('normal')">SNAP NORMAL [N / SPACE]</button>
        <button class="btn-anomalous" onclick="captureSnapshot('anomalous')">SNAP ANOMALOUS [A]</button>
    </div>
    <div class="hint">Use <b>N / Space</b> to capture normal sample, or <b>A</b> for anomaly.</div>

    <script>
        async function captureSnapshot(category) {
            try {
                const res = await fetch(`/snapshot/${category}`, { method: 'POST' });
                const data = await res.json();
                if (data.status === 'ok') {
                    document.getElementById('cnt-norm').innerText = `Normal: ${data.normal_count}`;
                    document.getElementById('cnt-anom').innerText = `Anomalous: ${data.anomalous_count}`;
                } else {
                    alert(data.message || 'Capture failed');
                }
            } catch (err) {
                console.error(err);
            }
        }

        window.addEventListener('keydown', (e) => {
            if (e.code === 'KeyN' || e.code === 'Space') {
                e.preventDefault();
                captureSnapshot('normal');
            } else if (e.code === 'KeyA') {
                e.preventDefault();
                captureSnapshot('anomalous');
            }
        });
    </script>
</body>
</html>
"""


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================


def main(
    model_path: Annotated[
        Path,
        arg(aliases=["-m"]),
    ] = Path("results/segmentation/rfdetr-seg-nano.onnx"),
    output_dir: Annotated[
        Path | None,
        arg(aliases=["-o"]),
    ] = None,
    bag_path: Annotated[
        Path | None,
        arg(aliases=["-b"]),
    ] = None,
    port: int = 8000,
    threshold: float = 0.5,
    target_size: int = 512,
    voxel_mm: float = 25.0,
    debug: bool = True,
):
    """Run interactive web-based capture station with live camera preview.

    Args:
        model_path: Path to ONNX model.
        output_dir: Output dataset root (default: new_data_YYYYMMDD_HHMMSS).
        bag_path: Path to RealSense .bag file (if not set, live camera is used).
        port: Local web server port for VS Code preview.
        threshold: Confidence threshold for detection.
        target_size: Square image resolution (default: 512).
        voxel_mm: 3D outlier filter voxel size.
        debug: If True, saves combined visualizations in 'vis/'.
    """
    if not model_path.is_file():
        print(f"Error: ONNX model '{model_path}' not found.", file=sys.stderr)
        sys.exit(1)

    if output_dir is None:
        output_dir = Path(f"new_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print(f">>> Target dataset output: {output_dir.resolve()}")
    camera = RealSenseSource(bag_path=bag_path)
    model = ONNXModel(model_path)

    service = CaptureService(
        camera=camera,
        model=model,
        output_dir=output_dir,
        threshold=threshold,
        target_size=target_size,
        voxel_mm=voxel_mm,
        debug=debug,
    )

    # run camera pipeline in background worker thread
    thread = threading.Thread(target=service.loop, daemon=True)
    thread.start()

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_TEMPLATE

    @app.get("/stream")
    async def stream():
        async def frame_generator():
            while True:
                with service.lock:
                    data = service.preview_jpeg
                if data:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                    )
                await asyncio.sleep(0.033)

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/snapshot/{category}")
    def snapshot(category: str):
        return JSONResponse(service.snapshot(category))

    print("\n" + "=" * 60)
    print(">>> Interactive capture is ready!")
    print(">>> Open in VS Code: Ctrl+Shift+P -> 'Simple Browser: Show'")
    print(f">>> Address: http://127.0.0.1:{port}")
    print("=" * 60 + "\n")

    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        service.running = False
        camera.stop()


if __name__ == "__main__":
    tyro.cli(main)