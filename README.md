# ST_GATE: Straight-Through Gate Scaling for MoE-FFD

**Course:** DA 6021, Deep Learning for Computer Vision (Jul-Nov 2026)
**Assigned paper:** *MoE-FFD: Mixture of Experts for Generalized and Parameter-Efficient Face Forgery Detection*, Kong et al., IEEE TDSC 2025 ([arXiv 2404.08452](https://arxiv.org/abs/2404.08452)). Original code: [LoveSiameseCat/MoE-FFD](https://github.com/LoveSiameseCat/MoE-FFD).

---

## 1. Project Overview

MoE-FFD detects fake faces with a **frozen ViT-B/16** (ImageNet-21K) and a few small trainable modules: LoRA experts on the attention `qkv` layer and Convpass adapter experts beside the MLP. A router picks **one expert (top-1)** per input. Only 15.51M of 101.29M parameters are trained.

**Our novelty (ST_GATE).** With top-1 routing, the gate is a softmax over one value, so it is always 1 with zero derivative, and the code never multiplies it into the output. The classification loss therefore never reaches the routers; they learn only from the load-balancing loss. ST_GATE keeps the forward pass identical but lets the router's softmax probability carry a gradient in the backward pass:

```
p = Softmax(H),  i* = argmax_j H_j
g = p[i*] / stopgrad(p[i*])      # forward value is exactly 1
y = g * Expert_i*(x)
dL/dH_j = c * (delta(i*, j) - p_j),   c = <dL/dy, Expert_i*(x)>
```

It is applied to both the Adapter-MoE (5 experts) and LoRA-MoE (7 experts). **0 extra parameters**, one softmax and one multiply per expert call. One switch controls it: `ST_GATE=1` is the modified model; unset or `0` is the original baseline (same code, same config).

**Main result (FF++ C23 test, 700 videos, 5 epochs, one seed)**

| Model | Video ACC | Video AUC | Video EER | Frame ACC | Frame AUC | Frame EER |
|---|---|---|---|---|---|---|
| Baseline | 95.00% | 0.9852 | **4.11%** | 90.28% | 0.9640 | 9.38% |
| + ST_GATE | **96.14%** | **0.9895** | 4.29% | **92.10%** | **0.9704** | **8.75%** |

Video EER is slightly worse; everything else improves. Details and caveats are in section 6 and in the one-page report.

---

## 2. Environment and Dependencies

Training and evaluation were run on **Kaggle, Tesla T4 GPU, Internet On**.

| Item | Version used |
|---|---|
| Python | 3.13.16 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| timm | 1.0.30 |
| albumentations | 2.0.8 |
| scikit-learn / scipy | 1.6.1 / 1.16.3 |
| opencv-python-headless | 4.14.0.94 |
| easydict / tqdm | 1.13 / 4.67.3 |
| facenet-pytorch (face cropping) | 2.6.0 |

```bash
git clone https://github.com/yagnaveenaaa/MoE-FFD.git
cd MoE-FFD
pip install -r requirements.txt
# if you have no requirements.txt, install the versions above, e.g.:
pip install timm==1.0.30 albumentations==2.0.8 easydict==1.13 scikit-learn==1.6.1 \
            scipy==1.16.3 opencv-python-headless==4.14.0.94 tqdm==4.67.3 facenet-pytorch==2.6.0
```

The original repo targets old libraries (Python 3.8, PyTorch 1.8, timm 0.5.4). To run on a current Kaggle image we made two small compatibility edits: `hf_hub` was changed to `hf_hub_id`, and the model builder now uses `pretrained_cfg`. The pretrained backbone is `B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz` (downloaded by timm).

---

## 3. Dataset Preparation

**FaceForensics++ (C23)** from the official download script (gated access; we do not redistribute it). Celeb-DF was used only for the small smoke test.

- **Split:** official pair lists, 360 train / 70 validation / 70 test pairs. Paired video IDs always stay in the same split.
- **Frames:** 20 evenly spaced frames per video. **Face crop:** facenet-pytorch MTCNN, largest face, margin 0.3 x the longer box side, saved as RGB JPEG (quality 95).
- **Classes:** `original`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NueralTextures` (the folder spelling is intentional; the loader expects it).
- **Model input:** 224 x 224, mean = std = 0.5. Real-video frames are repeated 4x to balance classes.

Expected layout:

```
data/c23/
  train/{original,Deepfakes,Face2Face,FaceSwap,NueralTextures}/<video_id>/<frames>.jpg
  valid/{same five folders}/...
  test/{same five folders}/...
```

| Split | Videos | Frames |
|---|---|---|
| train | 3,600 | 71,993 |
| valid | 700 | 13,994 |
| test | 700 | 14,000 |

Cropping script: see `preprocess/` (run it on a few videos first and look at the crops).

---

## 4. Training Instructions

Baseline and modified runs use **identical** settings; only `ST_GATE` differs.

```bash
# Baseline (original MoE-FFD)
python train.py --device 0 --model_dir baseline_ffpp_5epochs --data_root data/c23 \
  --epochs 5 --record_step 100 --batch_size 32 --learning_rate 3e-5 \
  --num_classes 2 --num_frames 20 --num_workers 2 --seed 2024

# Modified (ST_GATE), fresh run from ImageNet-21K weights
ST_GATE=1 python train.py --device 0 --model_dir stgate_ffpp_fresh_5epochs --data_root data/c23 \
  --resume -1 --epochs 5 --record_step 100 --batch_size 32 --learning_rate 3e-5 \
  --num_classes 2 --num_frames 20 --num_workers 2 --seed 2024
```

| Setting | Value |
|---|---|
| Optimiser | Adam, betas (0.9, 0.999), weight decay 1e-5 |
| Learning rate | 3e-5 for all weights, 1e-4 for router weights (`w_gate`, `w_noise`) |
| Schedule | StepLR, step 5, gamma 0.5 (so constant over 5 epochs) |
| Loss | CE + moe_loss (200 x LoRA balance + 1 x adapter balance) |
| Seed / batch | 2024 / 32 (3,600 iterations per epoch) |
| Time (T4) | about 74 min per epoch; 6 h 34 min for 5 epochs with validation |

Note: the baseline was trained in staged Kaggle sessions that resumed from saved epoch checkpoints (a kernel interruption forced this). The paper trains for 20 epochs; ours is a **5-epoch** comparison, not a full reproduction.

---

## 5. Inference and Evaluation

```bash
# ST_GATE model on FF++ test
ST_GATE=1 python eval.py --model_path <stgate_checkpoint>.tar \
  --data_root <path>/c23/test --dataset FFPP --output_csv stgate_ffpp_test_epoch5.csv \
  --frames 20 --batch_size 1 --num_workers 2 --device cuda:0

# Baseline on FF++ test (ST_GATE unset)
python eval.py --model_path <baseline_checkpoint>.tar \
  --data_root <path>/c23/test --dataset FFPP --output_csv ffpp_test_baseline.csv \
  --frames 20 --batch_size 1 --num_workers 2 --device cuda:0
```

Video score = mean of frame fake-probabilities; threshold 0.5; metrics are ACC, AUC, EER at frame and video level. Evaluating 700 videos takes about 4 minutes on a T4. **Always set `ST_GATE=1` when loading the ST_GATE checkpoint**, otherwise the gate scaling is off.

**Smoke-test threshold sweep (20 Celeb-DF videos, 0.30 to 0.70):**

```bash
python3 smoke_test/scripts/threshold_analysis.py
```

Outputs go to `smoke_test/threshold_analysis/` (metrics CSV plus frame/video prediction CSVs). Check the checkpoint and input paths inside the script before running on another machine.

---

## 6. Checkpoints and Results

| Model | File | Epoch | SHA-256 |
|---|---|---|---|
| Baseline | `baseline_models_params_4.tar` | 5 (index 4) | `49432ca97fe6a07d72c9f02afcf1aa3bac8b386f91c6302ea152711ac399d6ad` |
| ST_GATE | `stgate_models_params_4.tar` | 5 (index 4) | `2810239d31e355be916c1b10170ceb20535a0fbcfe4f2cf8038189639e01d4ae` |

Both files were originally named `models_params_4.tar` (the index is zero-based, so 4 means epoch 5). **Rename before use** so they do not overwrite each other.

- **Download links:** Baseline: `<ADD LINK>` | ST_GATE: `<ADD LINK>` | Training logs and result JSON/CSV: `<ADD LINK>`

**Test A: FF++ C23 test (700 videos / 14,000 frames), the main result**

| Metric | Baseline | ST_GATE | Change |
|---|---|---|---|
| Video accuracy | 95.00% | 96.14% | +1.14 pts |
| Video AUC | 0.985242 | 0.989490 | +0.004248 |
| Video EER | 4.11% | 4.29% | +0.18 pts (slightly worse) |
| Frame accuracy | 90.28% | 92.10% | +1.82 pts |
| Frame AUC | 0.963987 | 0.970353 | +0.006366 |
| Frame EER | 9.38% | 8.75% | -0.63 pts |

ST_GATE makes 8 fewer video mistakes (35 to 27 of 700). On validation it is ahead on all six metrics at epochs 4 and 5.

**Test B: smoke test on 20 Celeb-DF videos (10 real, 10 fake; epoch-4 checkpoints, CPU)**

| Level | Metric | Baseline | Modified |
|---|---|---|---|
| Video | AUC / EER | 0.7500 / 0.3000 | 0.8100 / 0.2000 |
| Video | ACC at 0.5 / 0.6 | 0.75 / 0.75 | 0.70 / 0.80 |
| Frame | AUC / EER | 0.7488 / 0.2950 | 0.8060 / 0.2550 |
| Frame | ACC at 0.5 | 0.6475 | 0.6475 |

The 0.6 threshold was chosen after seeing these 20 videos, so treat it as exploratory. This test is only a pipeline check, not a benchmark.

**Honest limits.** Single seed, in-domain (FF++ only), 5 epochs, no expert-usage analysis, no adapter-vs-LoRA ablation. A gradient check shows the router now receives gradient from the classification loss (24/24 adapter and 24/24 LoRA gate tensors non-zero); with the baseline code the adapter-router gradients under cross-entropy alone are `None`.

---

## 7. Code Organization and Reproducibility

```
MoE-FFD/
├── ViT_MoE.py        # model; ST_GATE changes live here (4 small edits, behind the ST_GATE switch)
├── train.py          # training (paths now passed as arguments)
├── eval.py           # evaluation, writes CSV
├── dataset.py        # loaders (expects the folder layout in section 3)
├── utils.py          # AUC / EER / ACC helpers
├── demo.py           # single-image inference
├── preprocess/       # frame extraction + MTCNN face cropping
├── smoke_test/
│   ├── README.md
│   ├── results/                 # original_baseline_smoke.csv, modified_model_smoke.csv
│   ├── threshold_analysis/      # threshold_metrics.csv + 4 prediction CSVs
│   └── scripts/threshold_analysis.py
├── requirements.txt
└── README.md
```

**Where the novelty is.** In `ViT_MoE.py`: (1) `ST_GATE = os.environ.get('ST_GATE','0') == '1'` at the top; (2) in both `noisy_top_k_gating()` methods, `top_k_gates = p_selected / p_selected.detach()` when `ST_GATE and k == 1`; (3) `expert_output * gates[i]` in `Adapter_MoElayer.forward`; (4) `qkv_delta * gates[i]` in `LoRA_MoElayer.forward`.

**To reproduce the results with the fewest steps:**
1. Set up the environment (section 2) and prepare FF++ C23 test crops (section 3).
2. Download both checkpoints, check the SHA-256 values, and rename them (section 6).
3. Run the two evaluation commands in section 5 and compare with the Test A table.

**Not changed by ST_GATE:** frozen backbone, expert definitions, hard top-1 rule, balance losses and weights, data split, cropping, optimiser, learning rates, schedule. Do not enable stochastic depth (`drop_path` must stay 0, as in the original code).

**Not included in the repo:** the FF++ and Celeb-DF data (gated access) and the large checkpoints (shared through the links above).

**Credits.** Base model and code: Kong et al. (MoE-FFD). ST_GATE modification, training runs, smoke test and report: this team.
