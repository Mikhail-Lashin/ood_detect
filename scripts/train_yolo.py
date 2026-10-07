from pathlib import Path
import sys
from typing import Annotated

import tyro
from tyro.conf import arg
from ultralytics import YOLO


def main(
    data: Annotated[Path, arg(aliases=["-d"])],
    epochs: Annotated[int, arg(aliases=["-e"])] = 50,
    batch_size: Annotated[int, arg(aliases=["-b"])] = 32,
    export_openvino: bool = False,
):
    """Train YOLO-cls on package dataset and export to OpenVINO.

    Args:
        data: Path to dataset folder containing 'train' and 'val' subfolders.
        epochs: Number of training epochs (default: 50).
        batch_size: Batch size (default: 32).
        export_openvino: Whether to export best model to OpenVINO FP16 after training.
    """
    # check dataset structure
    train_dir = data / "train"
    val_dir = data / "val"

    if not (train_dir.is_dir() and val_dir.is_dir()):
        print(f"Error: Dataset directory '{data}' must contain 'train' and 'val' subfolders.", file=sys.stderr)
        sys.exit(1)

    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])

    if train_classes != val_classes:
        print(f"Error: Class mismatch between train ({train_classes}) and val ({val_classes}).", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(f"STARTING YOLO CLASSIFICATION TRAINING ({len(train_classes)} CLASSES)")
    print(f"Classes : {', '.join(train_classes)}")
    print(f"Dataset : {data.resolve()}")
    print("=" * 70)

    # init model & pipeline
    base_weights = Path("results/yolo/yolo11n-cls.pt")
    base_weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base_weights))

    results = model.train(
        data=str(data.resolve()),
        epochs=epochs,
        batch=batch_size,
        imgsz=512,
        device=0,                   
        project=str(Path("results").resolve()),
        name="yolo",
        exist_ok=True,
        fliplr=0.5,
        flipud=0.5,                 
        degrees=15.0,              
        hsv_v=0.4,
        patience=15,
    )

    best_pt_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n>>> Training complete! Best PyTorch weights saved to: {best_pt_path}")

    # export to OpenVINO
    if export_openvino:
        print("\n>>> Exporting best model to OpenVINO FP16 format (imgsz=512)...")
        best_model = YOLO(str(best_pt_path))
        exported_path = best_model.export(format="openvino", imgsz=512, half=True)
        print(f">>> OpenVINO export successful! Artifacts located in: {exported_path}")
        print(">>> Ready for inference with OpenVINO runtime!")


if __name__ == "__main__":
    tyro.cli(main)