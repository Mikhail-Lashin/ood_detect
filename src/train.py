import logging
import os
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import sys
from typing import Annotated
import tyro
from tyro.conf import arg

from anomalib.data import Folder
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Dinomaly


def main(dataset: Annotated[Path, arg(aliases=["-d"])],
         epochs: Annotated[int, arg(aliases=["-e"])],
         batch_size: Annotated[int, arg(aliases=["-b"])] = 16,
         ckpt_path: Annotated[Path | None, arg(aliases=["-c"])] = None,
         num_workers: int = 0,
         model_name: str = "Dinomaly"
):
    """Train model & export to OpenVINO format.

    Args:
        dataset: Path to dataset folder containing 'train/normal', 'test/normal', and 'test/anomalous'.
        epochs: Total number of epochs to train.
        batch_size: Training batch size.
        ckpt_path: Optional path to a PyTorch Lightning checkpoint (.ckpt) to resume training.
        num_workers: Number of DataLoader worker processes.
        model_name: Architecture name (default: Dinomaly).
    """
    
    
    # check dataset structure
    if not dataset.exists():
        print(f"Error: Dataset path '{dataset}' does not exist.", file=sys.stderr)
        sys.exit(1)

    train_normal = dataset / "train/normal"
    test_normal = dataset / "test/normal"
    test_anom = dataset / "test/anomalous"

    if not (train_normal.exists() and test_normal.exists() and test_anom.exists()):
        print(f"Error: Dataset '{dataset}' must contain 'train/normal', 'test/normal' and 'test/anomalous'.", file=sys.stderr)
        sys.exit(1)

    # datamodule
    experiment_name = f"Model_{model_name}_{dataset.name}"
    datamodule = Folder(
        name=experiment_name,
        root=dataset,
        normal_dir="train/normal",
        abnormal_dir="test/anomalous",
        normal_test_dir="test/normal",
        mask_dir=None,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
        train_batch_size=batch_size,
        num_workers=num_workers,
    )

    # model & engine
    model = Dinomaly()
    engine = Engine(max_epochs=epochs)

    # train
    if ckpt_path:
        if not ckpt_path.exists():
            print(f"Error: Checkpoint file '{ckpt_path}' not found.", file=sys.stderr)
            sys.exit(1)
        print(f">>> Resuming training from checkpoint: {ckpt_path} to target {epochs} epochs")
        engine.fit(model=model, datamodule=datamodule, ckpt_path=str(ckpt_path))
    else:
        print(f">>> Starting new training run for {epochs} epochs")
        engine.fit(model=model, datamodule=datamodule)

    # export
    print(">>> Exporting model to OpenVINO format...")
    exported_path = engine.export(model=model, export_type=ExportType.OPENVINO)
    print(f">>> Success! Model exported. Check weights in: {exported_path or 'results/ directory'}")


if __name__ == "__main__":
    tyro.cli(main)