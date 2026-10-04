# src/visualizer.py
from pathlib import Path
import matplotlib.pyplot as plt
from anomalib.utils.visualization import ImageResult


def save_prediction_image(predictions, output_path: Path) -> None:
    res = ImageResult.from_dataset_item(predictions)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(res.image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(res.heat_map)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(res.pred_mask)
    axes[2].set_title("Mask")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)