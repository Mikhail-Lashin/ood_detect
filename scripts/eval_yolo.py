from pathlib import Path
import sys
import time
from typing import Annotated
import json
from datetime import datetime

import numpy as np
import tyro
from tyro.conf import arg
from ultralytics import YOLO


def main(
    data: Annotated[Path, arg(aliases=["-d"])],
    model_path: Annotated[Path, arg(aliases=["-m"])],
    device: str = "cpu",
):
    """Evaluate YOLO classifier on test set and print metrics to console.

    Args:
        data: Path to test dataset directory (or dataset root containing 'test' subfolder).
        model_path: Path to PyTorch model (.pt) or OpenVINO model directory.
        device: Device to run inference on ('cpu' or '0'). Default: 'cpu'.
    """
    # search test folder
    if (data / "test").is_dir():
        test_dir = data / "test"
    elif data.is_dir():
        test_dir = data
    else:
        print(f"Error: Data directory '{data}' not found.", file=sys.stderr)
        sys.exit(1)

    if not model_path.exists():
        print(f"Error: Model '{model_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # load model
    print("=" * 80)
    print(f"EVALUATING MODEL: {model_path.name}")
    print(f"Test data source: {test_dir.resolve()}")
    print(f"Device: {device}")
    print("=" * 80)

    model = YOLO(str(model_path.resolve()), task="classify")

    classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    if not classes:
        print(f"Error: No class subfolders found in {test_dir}", file=sys.stderr)
        sys.exit(1)

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    num_classes = len(classes)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # inference
    valid_exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    inference_times = []
    total_samples = 0
    records = []

    print("\n>>> Running inference...")
    for true_cls in classes:
        cls_dir = test_dir / true_cls
        images = [f for f in cls_dir.iterdir() if f.suffix in valid_exts]
        true_idx = class_to_idx[true_cls]

        for img_p in images:
            start_t = time.perf_counter()
            res = model(str(img_p), device=device, verbose=False)[0]
            inf_time = time.perf_counter() - start_t
            inference_times.append(inf_time)

            pred_idx = int(res.probs.top1)
            pred_cls = classes[pred_idx] if pred_idx < len(classes) else res.names[pred_idx]
            conf = float(res.probs.top1conf)
            is_correct = (true_cls == pred_cls)

            conf_matrix[true_idx, pred_idx] += 1
            total_samples += 1

            records.append({
                "filename": img_p.name,
                "photo_path": str(img_p.resolve()),
                "gt": true_cls,
                "predict": pred_cls,
                "confidence": round(conf, 4),
                "result_type": "CORRECT" if is_correct else "ERROR",
            })


    # metrics
    total_correct = np.trace(conf_matrix)
    overall_accuracy = total_correct / total_samples
    avg_time = float(np.mean(inference_times))
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    # ASCII confusion matrix
    col_w = max(12, max(len(c) for c in classes))
    header_cm = f"{'True / Pred':<20} | " + " | ".join(f"{c:>{col_w}}" for c in classes)
    sep_cm = "-" * len(header_cm)

    print("\n" + "=" * len(header_cm))
    print("CONFUSION MATRIX (Rows = Ground Truth, Cols = Prediction)")
    print("=" * len(header_cm))
    print(header_cm)
    print(sep_cm)

    for i, cls_name in enumerate(classes):
        row_vals = " | ".join(f"{conf_matrix[i, j]:>{col_w}}" for j in range(num_classes))
        print(f"{cls_name:<20} | {row_vals}")
    print("\n" + "=" * len(header_cm))
    print("\n")

    # per-class metrics
    header_metrics = f"{'Class':<20} | {'Recall':>12} | {'Precision':>15} | {'F1-Score':>12} | {'Samples':>9}"
    sep_metrics = "-" * len(header_metrics)

    print("\n" + "=" * len(header_metrics))
    print("PER-CLASS METRICS")
    print("=" * len(header_metrics))
    print(header_metrics)
    print(sep_metrics)

    for i, cls_name in enumerate(classes):
        tp = conf_matrix[i, i]
        fn = np.sum(conf_matrix[i, :]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = tp + fn

        print(f"{cls_name:<20} | {recall:>12.2%} | {precision:>15.2%} | {f1:>12.2%} | {support:>9}")
    print("\n" + "=" * len(header_metrics))
    print("\n")

    # summary
    sep_final = "=" * max(len(header_cm), len(header_metrics))
    print("\n" + sep_final)
    print(f"Overall Accuracy   : {overall_accuracy:.2%}")
    print(f"Total Test Samples : {total_samples}")
    print(f"Avg Inference Time : {avg_time * 1000:.2f} ms ({fps:.1f} FPS on {device.upper()})")
    print(sep_final + "\n")
    
    # save json dumps
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs/eval_yolo") / f"eval_{eval_timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_dump = {
        "timestamp": eval_timestamp,
        "model_path": str(model_path.resolve()),
        "test_dataset": str(test_dir.resolve()),
        "device": device,
        "total_processed": total_samples,
        "accuracy": round(overall_accuracy, 4),
        "performance": {
            "avg_inference_time_sec": round(avg_time, 4),
            "fps": round(fps, 1),
        },
        "classes": classes,
        "confusion_matrix": conf_matrix.tolist(),
        "per_class_metrics": {
            cls_name: {
                "recall": round(float(conf_matrix[i, i] / np.sum(conf_matrix[i, :])) if np.sum(conf_matrix[i, :]) > 0 else 0.0, 4),
                "precision": round(float(conf_matrix[i, i] / np.sum(conf_matrix[:, i])) if np.sum(conf_matrix[:, i]) > 0 else 0.0, 4),
                "f1": round(float(2 * (conf_matrix[i, i] / np.sum(conf_matrix[:, i])) * (conf_matrix[i, i] / np.sum(conf_matrix[i, :])) / ((conf_matrix[i, i] / np.sum(conf_matrix[:, i])) + (conf_matrix[i, i] / np.sum(conf_matrix[i, :])))) if (conf_matrix[i, i] > 0) else 0.0, 4),
                "samples": int(np.sum(conf_matrix[i, :])),
            }
            for i, cls_name in enumerate(classes)
        },
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_dump, f, indent=2, ensure_ascii=False)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f">>> Run artifacts saved to: {out_dir}")


if __name__ == "__main__":
    tyro.cli(main)