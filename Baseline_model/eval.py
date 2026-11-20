"""Evaluate an MoE-FFD checkpoint on an extracted frame dataset."""
import argparse
import csv
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ViT_MoE import vit_base_patch16_224_in21k
from dataset import TestDataset
from utils import cal_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_root', required=True, help='Dataset split root, e.g. .../c23/test')
    parser.add_argument('--dataset', choices=['FFPP', 'CDF', 'DFDC_P', 'WDF', 'DFD', 'DFR'], default='FFPP')
    parser.add_argument('--output_csv', default='evaluation.csv')
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1, help='Keep at 1 for video-level frame aggregation')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error('--batch_size must be 1 so frame scores stay grouped by video')

    device = torch.device(args.device)
    model = vit_base_patch16_224_in21k(pretrained=False, num_classes=2).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    weights = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(weights)
    model.eval()

    dataset = TestDataset(args.data_root, dataset=args.dataset, frame=args.frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    frame_scores, frame_labels, video_scores, video_labels = [], [], [], []
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc='Evaluating'):
            # TestDataset returns [frames, C, H, W]; default batch size 1 adds a leading dimension.
            images = images.reshape(-1, *images.shape[-3:]).to(device)
            probs = F.softmax(model(images)[0], dim=-1)[:, 1].cpu().tolist()
            label = int(labels.reshape(-1)[0])
            frame_scores.extend(probs)
            frame_labels.extend([label] * len(probs))
            video_scores.append(sum(probs) / len(probs))
            video_labels.append(label)

    rows = []
    for level, labels, scores in [('frame', frame_labels, frame_scores),
                                  ('video', video_labels, video_scores)]:
        result = cal_metrics(labels, scores, threshold=0.5)
        rows.append({'model': os.path.basename(args.model_path), 'dataset': args.dataset,
                     'level': level, 'AUC': result.AUC, 'EER': result.EER,
                     'ACC': result.ACC, 'samples': len(labels)})
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print('Wrote', args.output_csv)


if __name__ == '__main__':
    main()
