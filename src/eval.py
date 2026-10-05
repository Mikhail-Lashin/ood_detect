import logging
import os
import warnings
import json
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Annotated
import tyro
from tyro.conf import arg

from anomalib.deploy import OpenVINOInferencer
from metrics import EvaluationMetrics
from visualizer import save_prediction_image


def run_folder_inference(
    folder_path: Path,
    expected_anomalous: bool,
    inferencer: OpenVINOInferencer,
    metrics: EvaluationMetrics,
    threshold: float | None,
    save_vis: bool,
    output_dir: Path,
    records: list[dict],
):
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    images = []
    for ext in valid_exts:
        images.extend(folder_path.glob(ext))

    print(f">>> Processing {folder_path.name} ({len(images)} images)...")

    for img_p in images:
        start_t = time.perf_counter()
        predictions = inferencer.predict(image=img_p)
        inf_time = time.perf_counter() - start_t

        pred_score = float(predictions.pred_score)

        if threshold is not None:
            pred_label = pred_score >= threshold
        else:
            pred_label = bool(predictions.pred_label)

        res_type = metrics.update(
            is_actually_anomalous=expected_anomalous,
            pred_label=pred_label,
            score=pred_score,
            inf_time=inf_time,
        )

        out_img_path = None
        if save_vis:
            out_img_path = output_dir / res_type / f"{img_p.stem}_res.png"
            save_prediction_image(predictions, out_img_path)
            
        records.append({
            "filename": img_p.name,
            "gt": "Anomalous" if expected_anomalous else "Normal",
            "predict": "Anomalous" if pred_label else "Normal",
            "result_type": res_type,  # "TP", "TN", "FP", "FN"
            "score": round(pred_score, 4) if not np.isnan(pred_score) else None,
            "photo_path": str(img_p.resolve()),
            "vis_path": str(out_img_path.resolve()) if out_img_path else None,
        })


def main(
    data: Annotated[Path, arg(aliases=["-d"])],
    model_path: Annotated[Path, arg(aliases=["-m"])],
    anomaly_threshold: Annotated[float | None, arg(aliases=["-t"])] = None,
    visualizations: Annotated[bool, arg(aliases=["-v"])] = True,
    results: Annotated[bool, arg(aliases=["-r"])] = True,
    device: str = "CPU",
):
    """Evaluate model on labeled test data.

    Args:
        data: Path to test directory containing 'normal' and 'anomalous' subfolders.
        model_path: Path to OpenVINO model.bin.
        anomaly_threshold: Override model default threshold. If None, uses model threshold.
        visualizations: Whether to save prediction visualizations (Original, Heatmap, Mask).
        results: Whether to save results for report table.
        device: Inference device (e.g. 'CPU', 'GPU').
    """
    # check test dir structure
    normal_dir = data / "normal"
    anomalous_dir = data / "anomalous"

    if not (normal_dir.is_dir() and anomalous_dir.is_dir()):
        print(f"Error: Directory '{data}' must contain both 'normal' and 'anomalous' subfolders.", file=sys.stderr)
        sys.exit(1)

    if not model_path.is_file():
        print(f"Error: Model file '{model_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # results dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = data / f"eval_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if visualizations:
        for subdir in ["TP", "TN", "FP", "FN"]:
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    # inference
    print(f">>> Loading OpenVINO model: {model_path} (Device: {device})")
    inferencer = OpenVINOInferencer(path=model_path, device=device)

    records = [] 
    metrics = EvaluationMetrics()
    run_folder_inference(normal_dir, False, inferencer, metrics, anomaly_threshold, visualizations, run_dir, records)
    run_folder_inference(anomalous_dir, True, inferencer, metrics, anomaly_threshold, visualizations, run_dir, records)
    metrics.print_report(model_path=model_path, output_dir=run_dir)

    # save metadata
    meta = {
        "timestamp": timestamp,
        "model_path": str(model_path.resolve()),
        "test_dataset": str(data.resolve()),
        "threshold_used": anomaly_threshold if anomaly_threshold is not None else "model_default",
        "device": device,
        "visualizations_saved": visualizations,
    }
    metrics.save_metadata(run_dir / "metadata.json", meta)
    
    # save results
    if results:
        results_file = run_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f">>> Detailed results saved to: {results_file}")
    


if __name__ == "__main__":
    tyro.cli(main)