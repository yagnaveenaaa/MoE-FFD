# MoE-FFD team repository bootstrap

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
