# What to Do Now: MoE-FFD Baseline Handover and Improvement Plan

## Executive summary

The first project stage was to establish the authors' MoE-FFD model as a baseline before proposing a change. Kaggle training has now completed five epochs on the prepared FaceForensics++ C23 data. The saved file `models_params_4.tar` is the epoch-5 checkpoint: this training script numbers checkpoints from zero, so index 4 is epoch 5. The checkpoint was downloaded to the Windows Downloads folder and its SHA256 was verified as:

`49432ca97fe6a07d72c9f02afcf1aa3bac8b386f91c6302ea152711ac399d6ad`

Final validation metrics were video accuracy 93.29%, AUC 0.9873, EER 5.54%; frame accuracy 90.41%, AUC 0.9660, EER 8.88%. Held-out FF++ test evaluation processed 700 videos with 20 sampled frames per video: video accuracy 95.00%, AUC 0.985242, EER 4.11%; frame accuracy 90.28%, AUC 0.963987, EER 9.38%. These results apply to this five-epoch run and this exact evaluation protocol. They do not establish perfect training, full reproduction of the paper, or generalization to other datasets.

The next person should first reproduce the test evaluation from the checkpoint and the matching patched code. Only then should they begin a controlled improvement experiment. Preserve the checkpoint and baseline results unchanged.

## 1. Current status and artifacts

The trained model uses a ViT-B/16 backbone with LoRA experts, Convpass adapters, gating, and classification plus MoE balancing losses. Training was resumed across Kaggle sessions, with epoch 5 completing successfully. The final training log reported a total run time of 2 hours 9 minutes.

The checkpoint was saved as `/kaggle/working/models/baseline/models_params_4.tar`, then copied to `/kaggle/working/models_params_4.tar` for download. Splitting the file into `.part001` through `.part006` created additional copies; it did not alter or break the original TAR. The full downloaded TAR has the matching SHA256 above, so no merging is needed. The split parts are unnecessary after the full file is verified.

The evaluation used `/kaggle/working/data/c23/test`, dataset type `FFPP`, 20 frames per video, two data-loader workers, and GPU 0. The script averages frame fake probabilities into a video prediction and reports AUC, EER, and accuracy at threshold 0.5. It processed 700 videos.

The handover archive `baseline_epoch5_handover.zip` was reported as approximately 0.12 MB and should contain code, documentation, environment information, and results, but not the model or dataset. Inspect the archive before relying on it. Also export the Kaggle notebook as `.ipynb`; retain the JSON test metrics, training and evaluation logs, and a text file containing the checkpoint hash. Kaggle's `/kaggle/working` is temporary, so download everything needed before closing the session. The dataset is not included in the archive; confirm it remains accessible to teammates from an approved source.

## 2. Immediate handover checklist

1. Create a local folder such as `DA6021/handover_epoch5/`. Put the checkpoint, downloaded handover ZIP, exported notebook, JSON metrics, logs, environment snapshot, and SHA256 text file there. Keep one untouched copy of the checkpoint as the baseline reference.
2. Extract and inspect the ZIP. Confirm it contains the exact source that ran successfully, including `ViT_MoE.py`, `train.py`, `eval.py`, `dataset.py`, `utils.py`, license/notice files, and the compatibility changes. The Kaggle run needed `hf_hub` changed to `hf_hub_id`, updated timm builder arguments, explicit optimizer learning rates, and working resume handling. Confirm those changes are included.
3. Put source code, documentation, environment details, notebook, and small results files in the team GitHub repository. Do not commit the 530 MB checkpoint into ordinary Git history. Attach it to a GitHub Release or place it in an approved shared Kaggle Dataset/storage location. Include the SHA256 and access instructions. Do not publish the FaceForensics++ dataset unless its terms allow redistribution.
4. Share the repository, release/artifact, notebook, dataset instructions, and this report with the next team. Verify private repository and dataset access from a teammate's account.
5. In a fresh Kaggle copy, the receiver attaches the same approved data and checkpoint, enables GPU, runs setup/path cells, then runs evaluation only. They should skip training when reproducing the baseline. Compare generated JSON and console output against the reported metrics.
6. Only after baseline reproduction should the team change the method. Save every experiment in its own output directory and preserve the baseline checkpoint.

## 3. Reproduction procedure

Use the exact patched source snapshot that produced the checkpoint. Restore dependencies from the saved environment snapshot while keeping a compatible Kaggle PyTorch/CUDA stack. The evaluation script constructs `vit_base_patch16_224_in21k(pretrained=True, num_classes=2)` and loads `model_state_dict` from the TAR. Its pretrained initialization can fetch ImageNet-21K weights when needed; the trained checkpoint is then loaded over the model state. Keep the original architecture and compatibility patches consistent.

The successful evaluation command was equivalent to:

```bash
python eval.py \
  --model_path /kaggle/working/models/baseline/models_params_4.tar \
  --test_root /kaggle/working/data/c23/test \
  --dataset FFPP \
  --frames 20 \
  --num_workers 2 \
  --device 0 \
  --out_json /kaggle/working/models/baseline/ffpp_test_epoch5.json
```

The test directory used category folders `original`, `Deepfakes`, `Face2Face`, `FaceSwap`, and `NueralTextures`. Preserve the exact split IDs and preprocessing. The JSON result is the machine-readable evidence. Record code version, data version, environment, checkpoint hash, sampling rule, video count, frames per video, and command alongside every result.

Do not use the test split to tune thresholds, select epochs, or compare many experimental variants. Use validation for selection and reserve test for the final evaluation. The provided results do not independently prove split independence or absence of source-video overlap. Audit source identities and manipulated-video pairs across train, validation, and test before making strong generalization claims.

## 4. What the checkpoint can and cannot do

The TAR is a model/training checkpoint, not a standalone application. The evaluation script confirms it contains `model_state_dict`. It can be used to reproduce evaluation, run inference on correctly face-cropped and transformed frames, and potentially continue training with compatible code and optimizer/resume state. For architecture changes, some checkpoint parameters may not match; load matching weights deliberately and report missing or unexpected keys.

The checkpoint does not include the dataset, video decoder, face detector, preprocessing pipeline, or a user-facing app. To use it on new videos, the team must define face detection/cropping, frame sampling, frame-to-video aggregation, and threshold policy, then evaluate those choices. Do not promise identical metrics if data, preprocessing, code, frame count, or threshold differs.

## 5. Possible improvements

Make one change at a time and state the question it tests. Record each run's code version, change, seed, dataset version, initialization checkpoint, hyperparameters, validation results, runtime, and checkpoint hash. Use a separate output directory for each experiment.

### Foundation and data checks

- Audit train, validation, and test video IDs for overlap. Ensure manipulated videos and their source originals do not cross splits; check duplicate and near-duplicate frames.
- Verify category counts, videos per category, frames per video, and extraction failures. Keep sampling deterministic.
- Inspect representative crops and confirm RGB ordering, resize, normalization, and face alignment match the baseline.
- Reproduce the current epoch-5 evaluation in a clean Kaggle copy before changing anything.

### Training improvements

- Run a separate longer-training experiment toward the paper's reported 20 epochs. Validate checkpoint resume for optimizer, scheduler, scaler, and epoch state. More epochs may overfit, so compare validation curves and use a predeclared early-stopping rule.
- Try three fixed seeds and report mean and standard deviation, with the same split and sampling.
- Test modest learning-rate or schedule changes while keeping gate and other parameter groups distinct. Change one setting per experiment.
- Save both best-validation and last-epoch checkpoints. Confirm the current checkpoint-selection code before relying on it.
- Study calibration and class balance. Fit class weights or thresholds on training/validation only; report the fixed 0.5 result separately from any validation-selected threshold.

### Small architecture and loss changes

- Test one lower and one higher LoRA rank or Convpass bottleneck width. Compare AUC, accuracy, trainable parameter count, GPU memory, and inference speed.
- Change expert count or top-k routing one setting at a time. Log expert usage, gate entropy, and load balance to detect expert collapse.
- Sweep the MoE balancing-loss coefficient around the baseline while monitoring task metrics and routing statistics.
- Compare the frozen backbone with selective unfreezing of the final ViT block or normalization layers. Track compute cost and overfitting.
- Test modest compression, blur, color, and resize augmentation that reflects realistic video processing. Avoid changes that erase forensic cues.
- Compare mean frame probability to alternatives such as max or trimmed mean using validation data only. Lock the aggregation choice before final test evaluation.

### Robustness and generalization

- Evaluate on approved external datasets such as Celeb-DF-v2 or DFDC Preview when available, keeping each dataset distinct and documenting preprocessing and labels. WildDeepfake, Google DFD, and DeeperForensics are further possibilities if access and terms permit.
- Test compression levels, resolution changes, unseen manipulation methods, and temporal sampling strategies.
- Examine false positives and false negatives. Where labels and permissions permit, assess performance across capture conditions and demographic subgroups.
- Measure inference latency, GPU memory, throughput, and parameter count. A practical improvement may reduce cost without lowering accuracy.

### Reporting practice

Report frame and video metrics, data split, frames per video, seeds, checkpoint-selection rule, and confidence intervals where practical. Keep all runs, including unsuccessful ones, and do not cherry-pick a seed. Select the candidate using validation; run the held-out test for the final candidate under the same protocol. Describe this result as a five-epoch baseline, not a full paper reproduction.

## 6. Recommended sequence

1. Back up the notebook, exact code snapshot, environment, logs, JSON metrics, and checkpoint; verify the hash.
2. Publish code and small artifacts to team GitHub; publish the checkpoint as a Release asset or approved private artifact.
3. Have another team member reproduce test evaluation in a fresh Kaggle session without training.
4. Audit data counts, split IDs, and source-video independence.
5. Agree on one improvement hypothesis and validation metric before running experiments.
6. Run controlled experiments one at a time, in separate output folders.
7. Select using validation, then evaluate the selected method once on the held-out test set.
8. Deliver code, checkpoint, dataset instructions, environment, metrics, and limitations to the project supervisor/team.

## 7. Kaggle images and proof to retain

Screenshots are useful supporting evidence, but raw logs, JSON, notebook export, and the checkpoint hash are stronger and easier to verify. Before ending the Kaggle session, save screenshots of:

1. The final epoch-5 validation line, showing `Epoch [005/005]`, the metrics, and completion message.
2. The test evaluation completion, showing 700 videos and the final video/frame results.
3. Kaggle's file/output panel with the checkpoint, JSON, and logs, if visible.
4. The notebook title and relevant executed training/evaluation cells.

Use clear captions. Avoid exposing tokens, private account details, gated dataset URLs, or unnecessary face imagery. Retain text logs and JSON as the primary evidence; screenshots are supplementary and should not replace the artifacts.

## 8. Suggested teammate handover message

"The MoE-FFD baseline completed five epochs on the prepared FF++ C23 split. The epoch-5 checkpoint is `models_params_4.tar` (zero-based index 4); SHA256: `49432ca97fe6a07d72c9f02afcf1aa3bac8b386f91c6302ea152711ac399d6ad`. On the held-out 700-video FF++ test split with 20 sampled frames per video, video AUC was 0.9852, accuracy 95.00%, EER 4.11%; frame AUC was 0.9640, accuracy 90.28%, EER 9.38%. The patched source, notebook, environment, logs, and JSON results accompany the model. First reproduce evaluation in a clean Kaggle copy; then select one improvement using validation and preserve test for the final comparison. This is a five-epoch baseline, not a full paper-reproduction claim."

## 9. Limitations and open checks

- Inspect the generated handover ZIP to verify its source and results match the successful Kaggle run. The local DA6021 checkout may differ from the patched Kaggle snapshot.
- Confirm the test JSON is present and matches the output shown in Kaggle.
- Verify official split provenance and no source-video identity overlap across splits.
- The checkpoint hash proves file integrity, not provenance or model quality.
- Five epochs is shorter than the paper's reported 20-epoch setting.
- Confirm applicable upstream code/license and weight redistribution permissions before public release. Do not redistribute gated dataset content.
- The report's evidence recommendations are based on the user-provided logs; screenshots and Kaggle files were not independently inspected here.

## References

- Kong et al., "MoE-FFD: Mixture of Experts for Generalized and Parameter-Efficient Face Forgery Detection," arXiv:2404.08452.
- Official implementation: https://github.com/LoveSiameseCat/MoE-FFD
- Dataset: consult the approved FaceForensics++ provider terms and the project's `DATASETS.md`.
