<<<<<<< HEAD
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
=======
﻿# MoE-FFD team repository bootstrap

This is the DA6021 team working repository for the five-epoch MoE-FFD baseline and later controlled improvements. The upstream implementation is by Kong et al.; preserve its attribution and applicable license. Check dataset and pretrained weight terms before public redistribution.

## Baseline status

Five epochs completed on the prepared FaceForensics++ C23 split. The checkpoint is `models_params_4.tar` (zero-based index 4 means epoch 5). SHA256: `49432ca97fe6a07d72c9f02afcf1aa3bac8b386f91c6302ea152711ac399d6ad`.

Reported epoch-5 validation: video accuracy 93.29%, AUC 0.9873, EER 5.54%; frame accuracy 90.41%, AUC 0.9660, EER 8.88%.

Reported held-out FF++ test: 700 videos, 20 frames per video; video accuracy 95.00%, AUC 0.985242, EER 4.11%; frame accuracy 90.28%, AUC 0.963987, EER 9.38%. This is a five-epoch result for the supplied split, not a claim of perfect training or full paper reproduction.

## Checkpoint distribution

Download the model from the GitHub Release asset tagged `baseline-epoch5-v1`. The TAR is excluded from Git commits because of its size. Verify it after downloading:

```powershell
Get-FileHash .\models_params_4.tar -Algorithm SHA256
```

## Evaluation

The successful Kaggle evaluation used:

```bash
python eval.py --model_path /kaggle/working/models/baseline/models_params_4.tar --test_root /kaggle/working/data/c23/test --dataset FFPP --frames 20 --num_workers 2 --device 0 --out_json /kaggle/working/models/baseline/ffpp_test_epoch5.json
```

Use the exact patched Kaggle source snapshot in the handover archive and notebook to reproduce the reported JSON. The local clone may have a different evaluation CLI. A teammate should reproduce baseline evaluation before training improvements.

## Team contributions

Create a branch per change. Keep the epoch-5 checkpoint immutable and write experiment outputs separately. Select changes on validation; use the held-out test protocol for final comparison. Record code version, seed, data and preprocessing, initialization hash, parameters, metrics, and limitations.

## Handover files

See `handover/WhatToDoNow.md` and `handover/whatYVhasDone.pdf`. Dataset files are not included. Follow approved provider terms in `DATASETS.md`.

## Citation

Kong et al., "MoE-FFD: Mixture of Experts for Generalized and Parameter-Efficient Face Forgery Detection," arXiv:2404.08452. Upstream: https://github.com/LoveSiameseCat/MoE-FFD.
>>>>>>> 99312b0d552d8087854d75b024fcf40562f99bf7
