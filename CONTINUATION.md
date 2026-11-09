# Continuation note

## Project
- Paper: MoE-FFD, arXiv:2404.08452.
- Official implementation: https://github.com/LoveSiameseCat/MoE-FFD
- Local working copy: `C:\Users\sreey\OneDrive\Desktop\DA6021\moe-ffd-baseline`.
- User has a Kaggle notebook at `C:\Users\sreey\OneDrive\Desktop\DA6021\notebook362305e1fa.ipynb`.
- The notebook has duplicate clone cells and currently clones another project (`DL-for-CV`); official dataset downloader was separately pasted and saved as `/kaggle/working/download-FaceForensics.py`.

## What was clarified
- The assigned model is MoE-FFD: frozen ImageNet-21K ViT-B/16 plus trainable LoRA rank experts and Convpass local experts, top-1 gating, CE + load balancing loss.
- Train baseline on FF++ C23 using official train/val/test splits. Evaluate on FF++ test plus Celeb-DF v2; WDF, DFDC-P, DFD, and DFR are additional cross-dataset tests.
- Local repo has authors' model source and path-adapted train/eval, plus `extract_frames.py`, dataset docs, requirements. No local training/test was run because local environment lacks torch and datasets are run from Kaggle.

## Kaggle dataset state
- User reports downloading FF++ training data and Celeb-DF test data.
- Their last issue: the video-count checking script is taking too long. I advised interrupting it and checking only the first 5 matching `.mp4` files per category under `/kaggle/temp/ffpp`, or just `ls ... | head` for originals.
- Next: ask them for the quick-check output and whether the downloaded files are full or just the earlier `-n 5` sample. Confirm paths, then arrange official splits, extract/crop frames into the expected folder layout, run a small baseline smoke run, train, then evaluate on FF++ test and CDF.
- Do not assume `/kaggle/working/DL-for-CV/extract_faces.py` exists; check it. Local `extract_frames.py` is not automatically in Kaggle. The published FaceForensics repo does not include the gated downloader; official access approval supplies a downloader script.

## Expected frame layout
`c23/{train,valid,test}/{original,Deepfakes,Face2Face,FaceSwap,NueralTextures}/<video_id>/<frame>.jpg`

CDF should retain real/fake video folders and be evaluation-only. Keep all data processing and model training on Kaggle.
