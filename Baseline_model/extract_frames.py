"""Sample and MTCNN-crop faces from videos arranged as split/class/video files."""
import argparse
from pathlib import Path

import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', required=True,
                        help='Videos arranged as <split>/<class>/<video file>')
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--frames_per_video', type=int, default=16)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--margin', type=float, default=0.15,
                        help='Extra fraction around the detected face box')
    args = parser.parse_args()

    root, output = Path(args.input_root), Path(args.output_root)
    if args.frames_per_video < 1:
        parser.error('--frames_per_video must be positive')
    if not root.is_dir():
        parser.error('input_root does not exist: ' + str(root))
    detector = MTCNN(keep_all=True, device=torch.device(args.device), post_process=False)
    videos = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        parser.error('No supported videos found under input_root')

    for video_path in videos:
        relative = video_path.relative_to(root)
        target_dir = output / relative.parent / video_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            print('SKIP unreadable video:', video_path)
            capture.release()
            continue
        indices = sorted(set(round(x) for x in
                             __import__('numpy').linspace(0, count - 1,
                                                        min(args.frames_per_video, count))))
        saved = 0
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = detector.detect(Image.fromarray(rgb))
            if boxes is None or probs is None:
                continue
            best = int(probs.argmax())
            if float(probs[best]) < 0.90:
                continue
            x1, y1, x2, y2 = boxes[best]
            pad_x, pad_y = (x2 - x1) * args.margin, (y2 - y1) * args.margin
            height, width = rgb.shape[:2]
            x1 = max(0, int(x1 - pad_x)); y1 = max(0, int(y1 - pad_y))
            x2 = min(width, int(x2 + pad_x)); y2 = min(height, int(y2 + pad_y))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = Image.fromarray(rgb[y1:y2, x1:x2]).resize((224, 224), Image.Resampling.BILINEAR)
            crop.save(target_dir / f'{frame_index:06d}.jpg', quality=95)
            saved += 1
        capture.release()
        print(f'{relative}: saved {saved}/{len(indices)} face frames')


if __name__ == '__main__':
    main()
