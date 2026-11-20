# Paper datasets and access status

The paper uses six datasets. FF++ C23 is the training dataset; CDF-v2, WildDeepfake, DFDC Preview, Google DFD, and DeeperForensics-1.0 are cross-dataset evaluations. This repository does not bundle or redistribute video/face data. Several providers require academic approval or a signed terms-of-use agreement, and the datasets are too large for this Windows workspace. Put approved data in Kaggle storage and mount it as a notebook input.

| Dataset | Role | Official access |
|---|---|---|
| FaceForensics++ (C23) | Train, validation, in-domain test | [FaceForensics++ access instructions](https://github.com/ondyari/FaceForensics#access). Request access and use the supplied official downloader. |
| Celeb-DF v2 (CDF) | Cross-dataset test | [Dataset project page](https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics.html); follow its request and terms. |
| WildDeepfake (WDF) | Cross-dataset test | [Authors' repository](https://github.com/xingjunm/wild-deepfake); access is research-use restricted. |
| DFDC Preview (DFDC-P) | Cross-dataset test | [Meta DFDC page](https://ai.meta.com/datasets/dfdc/); follow its current access prerequisites. |
| Google DeepFake Detection (DFD) | Cross-dataset test | Included with FaceForensics access; see [benchmark documentation](https://kaldir.vc.in.tum.de/faceforensics_benchmark/documentation). |
| DeeperForensics-1.0 (DFR) | Cross-dataset test | [Authors' dataset instructions](https://github.com/189569400/DeeperForensics-1.0/tree/master/dataset); request approval and accept terms. |

## Required preprocessed tree

The authors' training code consumes face-cropped frame images, not raw videos. The accepted layout for FF++ is:

```text
FF++/c23/{train,valid,test}/{original,Deepfakes,Face2Face,FaceSwap,NueralTextures}/<video_id>/<frame_number>.jpg
```

For CDF and DFDC-P evaluation, provide video directories below their real/fake class directories as expected by `dataset.py`. All data preparation should happen in Kaggle after dataset access has been granted. Use the official split lists and preserve source licenses; do not upload restricted source data to a public Kaggle dataset. Kaggle's private datasets are appropriate only if the applicable dataset terms permit that sharing.

## Kaggle notebook

1. Add approved dataset assets using **Add Input**. Kaggle mounts them under `/kaggle/input/<dataset-slug>/`.
2. Clone/download this code into `/kaggle/working/moe-ffd-baseline` or upload the repository as a notebook input.
3. Point `--data_root` at the folder containing `train/` and `valid/`. Run a short smoke run first (`--epochs 1 --num_workers 2`).
4. Save checkpoints and logs under `/kaggle/working`; copy the outputs to a persistent Kaggle output dataset before the session ends.

Example:

```bash
python train.py --data_root /kaggle/input/approved-ffpp/FF++/c23 \
  --model_dir /kaggle/working/models/baseline --epochs 1 --batch_size 8
python eval.py --model_path /kaggle/working/models/baseline/model_params_best_*.pkl \
  --data_root /kaggle/input/approved-ffpp/FF++/c23/test --dataset FFPP \
  --output_csv /kaggle/working/baseline_ffpp_test.csv
```

The wildcard in `--model_path` is illustrative; replace it with the actual checkpoint filename printed by the training run. Kaggle account access is not available to this local workspace, so no dataset has been downloaded or mounted here yet.
