from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np


@dataclass
class EvaluationMetrics:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    inference_times: list[float] = field(default_factory=list)
    normal_scores: list[float] = field(default_factory=list)
    anomalous_scores: list[float] = field(default_factory=list)

    def update(self, is_actually_anomalous: bool, pred_label: bool, score: float, inf_time: float) -> str:
        self.inference_times.append(inf_time)

        if is_actually_anomalous:
            self.anomalous_scores.append(score)
            res_type = "TP" if pred_label else "FN"
        else:
            self.normal_scores.append(score)
            res_type = "TN" if not pred_label else "FP"

        setattr(self, res_type.lower(), getattr(self, res_type.lower()) + 1)
        return res_type

    def get_summary(self) -> dict:
        total = self.tp + self.tn + self.fp + self.fn
        avg_time = float(np.mean(self.inference_times)) if self.inference_times else 0.0
        fps = (1.0 / avg_time) if avg_time > 0 else 0.0

        accuracy = (self.tp + self.tn) / total if total > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "total_processed": total,
            "confusion_matrix": {"TP": self.tp, "TN": self.tn, "FP": self.fp, "FN": self.fn},
            "metrics": {
                "accuracy": round(accuracy, 4),
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f1": round(f1, 4),
            },
            "performance": {
                "avg_inference_time_sec": round(avg_time, 4),
                "fps": round(fps, 2),
            },
            "score_stats": {
                "normal": {
                    "min": round(float(np.min(self.normal_scores)), 4) if self.normal_scores else None,
                    "max": round(float(np.max(self.normal_scores)), 4) if self.normal_scores else None,
                    "mean": round(float(np.mean(self.normal_scores)), 4) if self.normal_scores else None,
                },
                "anomalous": {
                    "min": round(float(np.min(self.anomalous_scores)), 4) if self.anomalous_scores else None,
                    "max": round(float(np.max(self.anomalous_scores)), 4) if self.anomalous_scores else None,
                    "mean": round(float(np.mean(self.anomalous_scores)), 4) if self.anomalous_scores else None,
                },
            },
        }

    def print_report(self, model_path: Path, output_dir: Path):
        summary = self.get_summary()
        cm = summary["confusion_matrix"]
        m = summary["metrics"]
        perf = summary["performance"]
        sc = summary["score_stats"]

        print("\n" + "=" * 80)
        print(f"EVALUATION REPORT: {model_path.name}".center(80))
        print("=" * 80)
        print(f"Processed images: {summary['total_processed']}")
        print(f"TP: {cm['TP']} | TN: {cm['TN']} | FP: {cm['FP']} | FN: {cm['FN']}\n")

        print(f"Accuracy : {m['accuracy']:.2%}")
        print(f"Recall   : {m['recall']:.2%}")
        print(f"Precision: {m['precision']:.2%}")
        print(f"F1-Score : {m['f1']:.2%}\n")

        print(f"Avg Inference Time: {perf['avg_inference_time_sec']:.4f}s ({perf['fps']:.1f} FPS)\n")

        if sc["normal"]["mean"] is not None:
            print("SCORE STATS:")
            print(f"  Normal   : min={sc['normal']['min']}, max={sc['normal']['max']}, mean={sc['normal']['mean']}")
            print(f"  Anomalous: min={sc['anomalous']['min']}, max={sc['anomalous']['max']}, mean={sc['anomalous']['mean']}")

        print(f"\nResults saved to: {output_dir}")
        print("=" * 80 + "\n")

    def save_metadata(self, save_path: Path, extra_info: dict):
        data = extra_info.copy()
        data.update(self.get_summary())
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)