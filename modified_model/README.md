# Proposed cls_avg Mixture of Experts (MoE) Model

This folder contains the modified architecture and initialization scripts for the proposed `cls_avg` routing mechanism.

## Files Included:
- `ViT_MoE.py`: Modified Vision Transformer architecture featuring the `cls_avg` gating mechanism (`2 * dim` routing dimension combining the `CLS` token and patch mean).
- `apply_gate_patch.py`: Automation script used to patch the baseline gating logic.
- `clsavg_initialized.tar`: Pre-initialized model weights adapted from baseline epoch 5 (`models_params_4.tar`) to match the expanded 1536-dim gating layer.
