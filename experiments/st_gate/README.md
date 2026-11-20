# ST_GATE experiment

This folder documents the ST_GATE MoE-FFD experiment. The baseline files at the repository root and the existing `modified_model/` folder are kept unchanged. This is the ST_GATE run, not the earlier CLS_avg initialization work.

## Project and modification

The experiment uses the MoE-FFD ViT-B/16 model. ST_GATE is enabled by setting `ST_GATE=1`. The selected top-1 gate uses a straight-through value during training, and the expert output is multiplied by its gate.

The training log records initialization from the ViT-B/16 ImageNet-21k pretrained weights URL. It does not record loading `clsavg_initialized.tar`.

## Environment

Recorded Kaggle environment:

- Python 3.13.16
- PyTorch 2.11.0+cu130
- CUDA 13.0
- Tesla T4 GPU

See `requirements.txt` for the additional Python packages. Kaggle provided the PyTorch/CUDA installation.

## Dataset and preprocessing

The experiment used FaceForensics++ C23 with the official train, validation, and test splits. Videos were processed into 20 sampled frames per video. Face crops used the largest detected face with a 0.3 bounding-box margin.

Expected prepared data layout:

```text
data/c23/
├── train/
├── valid/
└── test/
