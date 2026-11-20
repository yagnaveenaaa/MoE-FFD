
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ViT_MoE import vit_base_patch16_224_in21k
from dataset import TestDataset


def collect_predictions(model_path, data_root, dataset_name, frames, device):
    model = vit_base_patch16_224_in21k(
        pretrained=False, num_classes=2
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    weights = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(weights)
    model.eval()

    dataset = TestDataset(data_root, dataset=dataset_name, frame=frames)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    frame_rows = []
    video_rows = []

    with torch.inference_mode():
        for video_index, (images, labels) in enumerate(
            tqdm(loader, desc=os.path.basename(model_path))
        ):
            images = images.reshape(-1, *images.shape[-3:]).to(device)
            probabilities = (
                F.softmax(model(images)[0], dim=-1)[:, 1]
                .cpu()
                .tolist()
            )

            label = int(labels.reshape(-1)[0])
            video_score = float(np.mean(probabilities))

            for frame_index, score in enumerate(probabilities):
                frame_rows.append({
                    "video_index": video_index,
                    "frame_index": frame_index,
                    "label": label,
                    "fake_score": score,
                })

            video_rows.append({
                "video_index": video_index,
                "label": label,
                "fake_score": video_score,
            })

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return frame_rows, video_rows


def calculate_metrics(rows, threshold):
    y_true = [row["label"] for row in rows]
    scores = [row["fake_score"] for row in rows]
    y_pred = [int(score > threshold) for score in scores]

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    fake_precision = precision_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )
    fake_recall = recall_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )
    fake_f1 = f1_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    fake_error = fn / (fn + tp) if (fn + tp) else 0.0
    acer = (fake_error + (1 - real_recall)) / 2

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,
        "ACER": acer,
        "AUC": roc_auc_score(y_true, scores),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "samples": len(y_true),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="smoke_test/frames/test")
    parser.add_argument("--dataset", default="CDF")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output_dir", default="smoke_test/threshold_analysis"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    models = {
        "original": "checkpoints/models_params_4.tar",
        "modified": "checkpoints/models_params_4.3.tar",
    }

    thresholds = [
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70,
    ]

    all_results = []

    for model_name, model_path in models.items():
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        frame_rows, video_rows = collect_predictions(
            model_path, args.data_root, args.dataset,
            args.frames, device
        )

        for level, rows in [
            ("frame", frame_rows),
            ("video", video_rows),
        ]:
            prediction_file = os.path.join(
                args.output_dir,
                f"{model_name}_{level}_predictions.csv",
            )

            with open(prediction_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            for threshold in thresholds:
                result = calculate_metrics(rows, threshold)
                result["model"] = model_name
                result["level"] = level
                all_results.append(result)

    output_file = os.path.join(
        args.output_dir, "threshold_metrics.csv"
    )

    fieldnames = [
        "model", "level", "threshold", "accuracy",
        "fake_precision", "fake_recall", "fake_f1",
        "ACER", "AUC", "TN", "FP", "FN", "TP", "samples",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n========== THRESHOLD ANALYSIS ==========")
    print(
        f"{'MODEL':<10} {'LEVEL':<8} {'THR':>5} "
        f"{'ACC':>8} {'FAKE-P':>8} {'FAKE-R':>8} "
        f"{'FAKE-F1':>8} {'ACER':>8}"
    )

    for result in all_results:
        print(
            f"{result['model']:<10} {result['level']:<8} "
            f"{result['threshold']:>5.2f} "
            f"{result['accuracy']:>8.4f} "
            f"{result['fake_precision']:>8.4f} "
            f"{result['fake_recall']:>8.4f} "
            f"{result['fake_f1']:>8.4f} "
            f"{result['ACER']:>8.4f}"
        )

    print("\nSaved results to:", output_file)


if __name__ == "__main__":
    main()
