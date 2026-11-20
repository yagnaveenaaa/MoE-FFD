# MoE-FFD Smoke Test Results

## 1. Purpose

This experiment compares the original baseline MoE-FFD model with the modified MoE-FFD model using the same sample dataset.

It includes evaluation results, prediction scores, and threshold analysis.

These are preliminary smoke-test results, not full benchmark results.

## 2. Folder Structure

- dataset/: Downloaded sample video dataset.
- data/raw/: Original evaluation videos.
- frames/test/: Extracted frames used for evaluation.
- results/: Baseline and modified model evaluation results.
- threshold_analysis/: Prediction CSVs and threshold metrics.
- scripts/: Threshold analysis Python script.

## 3. Dataset

- Total evaluation videos: 20
- Real videos: 10
- Fake videos: 10
- Dataset: CDF
- Frames per video: 20
- Total frame predictions: 400
- Total video predictions: 20

## 4. Models

| Model | Checkpoint |
|---|---|
| Original baseline | models_params_4.tar |
| Modified model | models_params_4.3.tar |

## 5. Evaluation Configuration

| Setting | Value |
|---|---|
| Dataset | CDF |
| Frames per video | 20 |
| Batch size | 1 |
| Workers | 0 |
| Device | CPU |

## 6. Baseline vs Modified Results

### Frame-level

| Metric | Original | Modified |
|---|---:|---:|
| AUC | 0.7488 | 0.8060 |
| EER | 0.2950 | 0.2550 |
| Accuracy | 0.6475 | 0.6475 |
| Samples | 400 | 400 |

### Video-level

| Metric | Original | Modified |
|---|---:|---:|
| AUC | 0.7500 | 0.8100 |
| EER | 0.3000 | 0.2000 |
| Accuracy | 0.7500 | 0.7000 |
| Samples | 20 | 20 |

## 7. Threshold Analysis

Thresholds tested:

0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70.

Metrics calculated at every threshold:

- Accuracy
- Fake precision
- Fake recall
- Fake F1-score
- ACER
- AUC
- TN, FP, FN, TP

### Frame-level at threshold 0.50

| Metric | Original | Modified |
|---|---:|---:|
| Accuracy | 0.6475 | 0.6475 |
| Fake Precision | 0.6148 | 0.6000 |
| Fake Recall | 0.7900 | 0.8850 |
| Fake F1 | 0.6915 | 0.7152 |
| ACER | 0.3525 | 0.3525 |

### Video-level at threshold 0.50

| Metric | Original | Modified |
|---|---:|---:|
| Accuracy | 0.7500 | 0.7000 |
| Fake Precision | 0.6667 | 0.6250 |
| Fake Recall | 1.0000 | 1.0000 |
| Fake F1 | 0.8000 | 0.7692 |
| ACER | 0.2500 | 0.3000 |

### Video-level at threshold 0.60

| Metric | Original | Modified |
|---|---:|---:|
| Accuracy | 0.7500 | 0.8000 |
| Fake Precision | 0.6923 | 0.7143 |
| Fake Recall | 0.9000 | 1.0000 |
| Fake F1 | 0.7826 | 0.8333 |
| ACER | 0.2500 | 0.2000 |

### Observations

- Modified model has higher AUC at both evaluation levels.
- Modified model has higher frame-level fake recall at threshold 0.50.
- Modified frame-level fake F1 is higher at every tested threshold.
- At video level, threshold 0.60 gives the modified model 0.80 accuracy and 0.8333 fake F1.
- AUC is based on raw scores and does not change with the classification threshold.

The complete threshold results are in threshold_analysis/threshold_metrics.csv.

## 8. Running Threshold Analysis

Run from the repository root:

python3 smoke_test/scripts/threshold_analysis.py

Input frames: smoke_test/frames/test

Output directory: smoke_test/threshold_analysis/

The script requires both model checkpoints in checkpoints/.

## 9. Output Files

- results/original_baseline_smoke.csv
- results/modified_model_smoke.csv
- threshold_analysis/threshold_metrics.csv
- threshold_analysis/original_frame_predictions.csv
- threshold_analysis/original_video_predictions.csv
- threshold_analysis/modified_frame_predictions.csv
- threshold_analysis/modified_video_predictions.csv

## 10. Limitations

- Only 20 videos were used.
- Results are preliminary and may not generalize to the full dataset.
- Threshold selection using this small sample may cause overfitting.
- Threshold 0.60 is an exploratory observation, not a validated final threshold.
- A separate validation set should be used to select the final threshold.
- Full benchmark evaluation is still required.

## 11. Data Preservation

Datasets, extracted frames, and model checkpoints are retained locally.

Large datasets and model weights are excluded from GitHub. Evaluation CSVs, analysis scripts, and this README can be version-controlled.

---

Status: Smoke-test evaluation and threshold analysis completed.
