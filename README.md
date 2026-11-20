# Celeb-DF smoke test 2: baseline versus new checkpoint

This folder contains one authoritative metrics table and one paired prediction
file for each evaluation level. Duplicate HTML/Markdown reports, repeated JSON
metrics, and separate per-model prediction CSVs were removed during packaging.
The original evaluation output remains unchanged outside this folder.

## File guide

| File | Contents |
|---|---|
| all_metrics_baseline_vs_new.csv | All frame/video metrics and measured inference timing, with baseline, new, and new-minus-baseline columns |
| frame_predictions_baseline_vs_new.csv | Both models' probabilities, predictions and correctness for each evaluated crop |
| video_predictions_baseline_vs_new.csv | Both models' averaged probabilities, predictions and correctness for each video |
| evaluation_protocol_and_checkpoint_details.json | Evaluation protocol, environment, checkpoint names/hashes, parameter counts and strict-loading status; no repeated metrics |
| charts/*confusion_matrix.png | One confusion matrix per model and evaluation level |
| charts/*roc_baseline_vs_new.png | Both models' ROC curves together, one chart per evaluation level |
| sample_videos.csv | Original video identities, labels, portable file paths and SHA256 hashes |
| sample_frames.csv | Evaluated crop identities, frame indices, labels, portable paths and SHA256 hashes |
| samples/videos/ | Original test videos |
| samples/faces/ | Exact evaluated face crops |
| preprocessing.json | Recorded face-extraction settings and coverage |
| compare_20_models.py | Evaluation and original report-generation script |
| evaluation_source/ViT_MoE.py | Model implementation used for evaluation |
| requirements-lock.txt | Installed package versions |

## Interpretation

The test uses 20 videos (10 real, 10 fake), with five crops per video.
Fake is class 1; real is class 0. The decision threshold is 0.5.
Video probability is the mean of its five crop probabilities.
Metrics are fractions, not percentages. Counts and timing use the units named
in the metric column. EER is interpolated at FPR = 1 - TPR.
TN/FP/FN/TP use fake as positive. Higher is better for accuracy, AUC,
precision, recall, F1 and specificity; lower is better for EER, log loss and
Brier score. Timing is one CPU run, excluding warm-up, loading and face detection.
These results describe this small sample, not full Celeb-DF performance.
Charts visualize the results; they are not additional metric data copies.
Byte-identical sample payloads, if any, are stored once and referenced by manifests.

## Reproduce

With compatible Python and the packages in requirements-lock.txt, run from here:

```powershell
python compare_20_models.py --data . --source evaluation_source --baseline "PATH_TO_models_params_4.tar" --new "PATH_TO_models_params_4.3.tar" --output ../smoketest_rerun
```

The evaluator checks sample hashes before running. Its output uses the original
report layout; this folder is the consolidated publication layout.
Checkpoints are not included. No GitHub upload has been performed.
Dataset redistribution permission is not established by this evaluation.
