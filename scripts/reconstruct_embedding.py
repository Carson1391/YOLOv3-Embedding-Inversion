from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import cv2
from PIL import Image
import torch
from torch import Tensor

from checkpointing import load_decoder_checkpoint
from data_utils import tensor_to_rgb_uint8, ycbcr_to_rgb
from embedding_io import load_embedding_package
from model_setup import resolve_device


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate image frames from one L2-normalized YOLOv3 vector or a stack."
    )
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Image file, video file, or output directory.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fps", type=float, default=0.0, help="Video FPS; 0 uses package metadata.")
    return parser.parse_args()


@torch.inference_mode()
def decode_batches(decoder, embedding: Tensor, batch_size: int, device: torch.device) -> Iterator[Tensor]:
    for start in range(0, embedding.shape[0], batch_size):
        batch = embedding[start : start + batch_size]
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = decoder(batch)
        else:
            generated = decoder(batch)
        yield ycbcr_to_rgb(generated.float()).cpu()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    decoder, checkpoint = load_decoder_checkpoint(args.checkpoint, device)
    embedding, metadata = load_embedding_package(args.embedding, device)

    package_hash = metadata.get("weights_sha256", "")
    checkpoint_hash = checkpoint.get("yolo_weights_sha256", "")
    if package_hash and checkpoint_hash and package_hash != checkpoint_hash:
        raise ValueError(
            "Embedding and decoder use different yolov3.weights hashes. "
            f"embedding={package_hash}, checkpoint={checkpoint_hash}"
        )

    decoder.eval()
    output = Path(args.output)
    suffix = output.suffix.lower()
    frame_count = embedding.shape[0]

    if frame_count == 1:
        generated = next(decode_batches(decoder, embedding, 1, device))[0]
        if suffix not in IMAGE_EXTENSIONS:
            output.mkdir(parents=True, exist_ok=True)
            output = output / "reconstruction.png"
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(tensor_to_rgb_uint8(generated), mode="RGB").save(output)
        print(f"Saved {output}")
        return

    if suffix in VIDEO_EXTENSIONS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fps = args.fps if args.fps > 0 else float(metadata.get("fps", 30.0))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, fps, (416, 416))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create video writer: {output}")
        try:
            for generated_batch in decode_batches(decoder, embedding, args.batch_size, device):
                for frame in generated_batch:
                    rgb = tensor_to_rgb_uint8(frame)
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        print(f"Saved {frame_count} decoded frames to {output}")
        return

    output.mkdir(parents=True, exist_ok=True)
    index = 0
    for generated_batch in decode_batches(decoder, embedding, args.batch_size, device):
        for frame in generated_batch:
            frame_path = output / f"frame_{index:06d}.png"
            Image.fromarray(tensor_to_rgb_uint8(frame), mode="RGB").save(frame_path)
            index += 1
    print(f"Saved {index} decoded frames under {output}")


if __name__ == "__main__":
    main()
