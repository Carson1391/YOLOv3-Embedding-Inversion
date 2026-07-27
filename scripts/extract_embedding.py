from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import torch
from torch import Tensor

from data_utils import load_rgb_tensor, rgb_uint8_to_tensor, webcam_frame_to_rgb_uint8
from embedding_io import save_embedding_package
from embedding_ops import heads_to_embedding
from model_setup import load_original_yolov3, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a portable L2-normalized final-vector package from an image or video."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--video")
    parser.add_argument("--output", required=True, help="Output .npz file.")
    project_dir = Path(__file__).resolve().parent
    parser.add_argument("--cfg", default=str(project_dir / "yolov3.cfg"))
    parser.add_argument("--weights", default=str(project_dir / "yolov3.weights"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--every", type=int, default=1, help="For video, keep every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 keeps the full video.")
    return parser.parse_args()


@torch.inference_mode()
def encode_batch(yolo, images: Tensor) -> Tensor:
    return heads_to_embedding(yolo(images))


def extract_video(args: argparse.Namespace, yolo, device: torch.device) -> tuple[Tensor, float]:
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    output_fps = source_fps / max(1, args.every) if source_fps > 0 else 30.0 / max(1, args.every)
    embeddings: List[Tensor] = []
    batch: List[Tensor] = []
    frame_index = 0
    kept = 0

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % max(1, args.every) != 0:
                continue
            frame_rgb = webcam_frame_to_rgb_uint8(frame_bgr)
            batch.append(rgb_uint8_to_tensor(frame_rgb))
            kept += 1

            if len(batch) >= args.batch_size:
                images = torch.stack(batch).to(device, non_blocking=True)
                embeddings.append(encode_batch(yolo, images).cpu())
                batch.clear()

            if args.max_frames > 0 and kept >= args.max_frames:
                break
    finally:
        capture.release()

    if batch:
        images = torch.stack(batch).to(device, non_blocking=True)
        embeddings.append(encode_batch(yolo, images).cpu())
    if not embeddings:
        raise ValueError("The video produced zero selected frames.")
    return torch.cat(embeddings, dim=0), output_fps


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    yolo, weights_hash = load_original_yolov3(args.cfg, args.weights, device)

    if args.image:
        image = load_rgb_tensor(args.image).unsqueeze(0).to(device)
        embedding = encode_batch(yolo, image).cpu()
        source_name = Path(args.image).name
        source_type = "image"
        fps = None
    else:
        embedding, fps = extract_video(args, yolo, device)
        source_name = Path(args.video).name
        source_type = "video"

    save_embedding_package(
        args.output,
        embedding,
        yolo_weights_sha256=weights_hash,
        source_name=source_name,
        source_type=source_type,
        fps=fps,
    )
    print(f"Saved {args.output}")
    print(f"shape={tuple(embedding.shape)}")
    print(f"L2 norms: min={embedding.norm(dim=1).min().item():.8f} max={embedding.norm(dim=1).max().item():.8f}")


if __name__ == "__main__":
    main()
