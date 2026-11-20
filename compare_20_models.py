"""Compare two checkpoints on the saved 20-video Celeb-DF sample."""
import argparse
import csv
import hashlib
import html
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=here / 'celebdf_local_eval')
    p.add_argument('--source', type=Path, default=here / 'moe-ffd-baseline')
    p.add_argument('--baseline', type=Path, default=Path.home() / 'Downloads/models_params_4.tar')
    p.add_argument('--new', type=Path, default=Path.home() / 'Downloads/models_params_4.3.tar')
    p.add_argument('--output', type=Path)
    p.add_argument('--check-only', action='store_true', help='Validate data and strict checkpoint loading without evaluation')
    a = p.parse_args()
    print('Loading evaluation libraries...', flush=True)
    import cv2
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn import metrics as sk
    sys.path.insert(0, str(a.source))
    from ViT_MoE import VisionTransformer
    def transform(img):
        # Equivalent to dataset.base_transform: OpenCV linear resize,
        # Albumentations standard normalization with mean=std=0.5.
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) - np.float32(127.5)) * np.float32(1.0 / 127.5)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    if (a.data / 'smoke_manifest.csv').exists():
        videos = read_csv(a.data / 'smoke_manifest.csv')
        frames = read_csv(a.data / 'smoke_frames.csv')
    else:
        videos = read_csv(a.data / 'sample_videos.csv')
        frames = read_csv(a.data / 'sample_frames.csv')
        for v in videos:
            v['local_path'] = str(a.data / v['path'])
            assert sha(v['local_path']) == v['sha256']
        for f in frames:
            f['crop_path'] = str(a.data / f['path'])
            f['status'] = 'ok'
            assert sha(f['crop_path']) == f['sha256']
    assert len(videos) == 20 and len({v['video'] for v in videos}) == 20
    assert len(frames) == 100 and all(f['status'] == 'ok' for f in frames)
    assert sum(int(v['label']) for v in videos) == 10, 'Expected 10 real and 10 fake videos'
    for v in videos:
        group = [f for f in frames if f['video'] == v['video']]
        assert len(group) == 5 and all(f['label'] == v['label'] for f in group)
        assert Path(v['local_path']).is_file(), v['local_path']
    assert len({f['crop_path'] for f in frames}) == 100
    for f in frames:
        assert Path(f['crop_path']).is_file(), f['crop_path']
    # Same constructor arguments as vit_base_patch16_224_in21k(num_classes=2).
    # Direct construction avoids version-dependent timm pretrained-config APIs.
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=2)
    model.eval()
    out = a.output or a.data / ('comparison_20_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not a.check_only:
        out.mkdir(parents=True, exist_ok=False)
    results, predictions, summary = {}, {}, []

    def measure(labels, scores):
        y, s = np.asarray(labels), np.asarray(scores)
        pred = (s >= 0.5).astype(int)
        tn, fp, fn, tp = sk.confusion_matrix(y, pred, labels=[0, 1]).ravel()
        fpr, tpr, _ = sk.roc_curve(y, s, drop_intermediate=False)
        d = fpr + tpr - 1
        i = int(np.flatnonzero(d >= 0)[0])
        eer = fpr[i] if i == 0 or d[i] == 0 else fpr[i-1] + (-d[i-1] / (d[i]-d[i-1])) * (fpr[i]-fpr[i-1])
        m = dict(samples=len(y), real_samples=int(sum(y == 0)), fake_samples=int(sum(y == 1)),
                 accuracy=sk.accuracy_score(y, pred), balanced_accuracy=sk.balanced_accuracy_score(y, pred),
                 roc_auc=sk.roc_auc_score(y, s), eer_interpolated=float(eer),
                 precision_fake=sk.precision_score(y, pred, zero_division=0),
                 recall_fake=sk.recall_score(y, pred, zero_division=0),
                 f1_fake=sk.f1_score(y, pred, zero_division=0), specificity=float(tn/(tn+fp)),
                 log_loss=sk.log_loss(y, s, labels=[0, 1]), brier_score=sk.brier_score_loss(y, s),
                 TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
        return m

    for name, checkpoint in [('baseline', a.baseline), ('new', a.new)]:
        print(f'Loading {name}: {checkpoint}', flush=True)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if isinstance(state, dict):
            for key in ('model_state_dict', 'state_dict', 'model'):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if all(k.startswith('module.') for k in state):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        del state
        print(f'{name}: strict checkpoint loading passed', flush=True)
        if a.check_only:
            continue
        frame_rows, video_rows, elapsed = [], [], 0.0
        with torch.inference_mode():
            model(torch.zeros(5, 3, 224, 224))  # excluded warm-up
            for idx, v in enumerate(videos, 1):
                group = [f for f in frames if f['video'] == v['video']]
                images = []
                for f in group:
                    img = cv2.imread(f['crop_path'])
                    if img is None:
                        raise RuntimeError('Unreadable image: ' + f['crop_path'])
                    images.append(transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                batch = torch.stack(images)
                start = time.perf_counter()
                scores = torch.softmax(model(batch)[0], dim=-1)[:, 1].tolist()
                elapsed += time.perf_counter() - start
                assert np.isfinite(scores).all()
                for f, s in zip(group, scores):
                    frame_rows.append(dict(video=v['video'], frame_index=int(f['frame_index']), label=int(v['label']), fake_probability=s, prediction=int(s >= .5)))
                s = float(np.mean(scores))
                video_rows.append(dict(video=v['video'], label=int(v['label']), fake_probability=s, prediction=int(s >= .5)))
                print(f'{name}: {idx}/20 {v["video"]} fake_probability={s:.6f}', flush=True)
        result = dict(checkpoint=checkpoint.name, sha256=sha(checkpoint), strict_loading=True,
                      total_parameters=sum(x.numel() for x in model.parameters()),
                      checkpoint_bytes=checkpoint.stat().st_size, inference_seconds=elapsed,
                      frames_per_second=100/elapsed, milliseconds_per_frame=elapsed*10)
        for level, rows in [('frame', frame_rows), ('video', video_rows)]:
            write_csv(out / f'{name}_{level}_predictions.csv', rows)
            y = [r['label'] for r in rows]
            s = [r['fake_probability'] for r in rows]
            result[level] = measure(y, s)
            summary.append(dict(model=name, level=level, **result[level]))
            fig, ax = plt.subplots(figsize=(5, 4))
            sk.ConfusionMatrixDisplay.from_predictions(y, [r['prediction'] for r in rows], labels=[0, 1], display_labels=['Real', 'Fake'], ax=ax, colorbar=False)
            ax.set_title(f'{name}: {level}')
            fig.tight_layout()
            fig.savefig(out / f'{name}_{level}_confusion.png', dpi=160)
            plt.close(fig)
        results[name] = result
        predictions[name] = dict(frame=frame_rows, video=video_rows)
    if a.check_only:
        print('CHECK PASSED: 20 videos, 100 crops, both checkpoints load strictly.')
        return
    write_csv(out / 'comparison_metrics.csv', summary)
    paired = []
    for b, n in zip(predictions['baseline']['video'], predictions['new']['video']):
        assert b['video'] == n['video'] and b['label'] == n['label']
        paired.append(dict(video=b['video'], label=b['label'], baseline_probability=b['fake_probability'], new_probability=n['fake_probability'], baseline_prediction=b['prediction'], new_prediction=n['prediction'], baseline_correct=b['prediction']==b['label'], new_correct=n['prediction']==n['label']))
    write_csv(out / 'paired_video_comparison.csv', paired)
    for level in ('frame', 'video'):
        fig, ax = plt.subplots(figsize=(5, 4))
        for name in results:
            rows = predictions[name][level]
            sk.RocCurveDisplay.from_predictions([r['label'] for r in rows], [r['fake_probability'] for r in rows], name=name, ax=ax)
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_title(f'{level.capitalize()} ROC: same 20 videos')
        fig.tight_layout()
        fig.savefig(out / f'{level}_roc.png', dpi=160)
        plt.close(fig)
    # Copy exact evaluated inputs; manifests use portable paths relative to the bundle.
    portable_videos, portable_frames = [], []
    for v in videos:
        target = out / 'samples/videos' / v['video']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v['local_path'], target)
        portable_videos.append(dict(video=v['video'], label=v['label'], path=target.relative_to(out).as_posix(), sha256=sha(target)))
    for f in frames:
        target = out / 'samples/faces' / Path(f['video']).with_suffix('') / Path(f['crop_path']).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f['crop_path'], target)
        portable_frames.append(dict(video=f['video'], label=f['label'], frame_index=f['frame_index'], path=target.relative_to(out).as_posix(), sha256=sha(target)))
    write_csv(out / 'sample_videos.csv', portable_videos)
    write_csv(out / 'sample_frames.csv', portable_frames)
    preprocessing = a.data / 'smoke_preprocessing.json'
    if not preprocessing.exists():
        preprocessing = a.data / 'preprocessing.json'
    shutil.copy2(preprocessing, out / 'preprocessing.json')
    source = out / 'evaluation_source'
    source.mkdir()
    for filename in ('ViT_MoE.py', 'dataset.py'):
        shutil.copy2(a.source / filename, source / filename)
    shutil.copy2(__file__, out / Path(__file__).name)
    (out / 'requirements-lock.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    metadata = dict(evaluation='20-video Celeb-DF sample; not full-test performance', threshold=.5,
                    positive_class='fake (1); real (0)', aggregation='mean of five frame fake probabilities',
                    preprocessing='existing face crops; RGB; repository base_transform: 224x224, mean/std 0.5',
                    timing='CPU, four threads; forward plus softmax; warm-up, I/O and face detection excluded; one timed run',
                    python=sys.version, torch=torch.__version__, platform=platform.platform(), models=results)
    (out / 'results.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Baseline versus new checkpoint: 20-video comparison', '',
             'Same 10 real and 10 fake videos, with five saved face crops each (100 frames).',
             'Fake is the positive class. Prediction is fake when probability >= 0.5. Video probability is the mean of five frame probabilities.',
             'This is a small sample evaluation, not full Celeb-DF test performance. Frames from a video are correlated. No threshold tuning was performed.', '',
             '| Level | Metric | Baseline | New | New minus baseline |', '|---|---|---:|---:|---:|']
    for level in ('video', 'frame'):
        for key in results['baseline'][level]:
            b, n = results['baseline'][level][key], results['new'][level][key]
            lines.append(f'| {level} | {key} | {b:.8f} | {n:.8f} | {n-b:+.8f} |')
    lines += ['', 'Metrics are fractions, not percentages. CSV/JSON preserve full numerical precision. EER is the linear intersection of the empirical ROC with FPR = 1 - TPR. Confusion matrices use rows=true and columns=predicted, ordered real/fake.', '',
              f'Videos corrected by new model: {sum(not r["baseline_correct"] and r["new_correct"] for r in paired)}.',
              f'Videos regressed by new model: {sum(r["baseline_correct"] and not r["new_correct"] for r in paired)}.', '',
              '## Checkpoints and timing', '']
    for name, r in results.items():
        lines += [f'- {name}: {r["checkpoint"]}; SHA256 `{r["sha256"]}`; strict loading passed; {r["total_parameters"]} parameters.',
                  f'  Inference: {r["inference_seconds"]:.6f} s; {r["frames_per_second"]:.6f} frames/s; {r["milliseconds_per_frame"]:.6f} ms/frame.']
    lines += ['', metadata['timing'], '', '## Sample-level predictions', '',
              '| Video | Label | Baseline P(fake) | New P(fake) | Baseline correct | New correct |', '|---|---:|---:|---:|---|---|']
    for r in paired:
        lines.append(f'| {r["video"]} | {r["label"]} | {r["baseline_probability"]:.8f} | {r["new_probability"]:.8f} | {r["baseline_correct"]} | {r["new_correct"]} |')
    lines += ['', '## Bundle contents', '', 'The samples folder contains the original 20 videos and all 100 evaluated crops. Manifests include SHA256 hashes. Checkpoints are excluded. Source and installed package versions are included. No GitHub upload was performed. Dataset redistribution permission is not established by this evaluation.', '',
              'To repeat from this bundle: python compare_20_models.py --data . --source evaluation_source --baseline PATH_TO_BASELINE --new PATH_TO_NEW --output ../rerun_results. Install requirements-lock.txt in a compatible Python environment first. Portable sample hashes are verified before evaluation.', '']
    report = '\n'.join(lines)
    (out / 'REPORT.md').write_text(report, encoding='utf-8')
    (out / 'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>Model comparison</title><style>body{font:14px monospace;margin:30px}pre{white-space:pre-wrap}img{max-width:600px}</style><pre>' + html.escape(report) + '</pre>' + ''.join(f'<img src="{p.name}">' for p in sorted(out.glob('*.png'))), encoding='utf-8')
    (out / '.gitignore').write_text('*.tar\n__pycache__/\n', encoding='utf-8')
    archive = shutil.make_archive(str(out), 'zip', out)
    print('\nCOMPLETED\nReport: ' + str(out / 'REPORT.html') + '\nBundle: ' + archive, flush=True)
    print(json.dumps({name: r['video'] for name, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
