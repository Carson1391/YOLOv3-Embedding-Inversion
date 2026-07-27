from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root: str | Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No supported images were found under {root}")
    return paths


def split_paths(paths: Sequence[Path], validation_fraction: float, seed: int) -> tuple[List[Path], List[Path]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * validation_fraction)))
    return shuffled[val_count:], shuffled[:val_count]


def make_transform(image_size: int = 416, training: bool = True) -> transforms.Compose:
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.65, 1.0),
                    ratio=(0.80, 1.25),
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-15, 15), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.04),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ]
    )


class ImagePathDataset(Dataset[Tensor]):
    def __init__(self, paths: Sequence[str | Path], image_size: int = 416, training: bool = True) -> None:
        self.paths = [Path(path) for path in paths]
        self.transform = make_transform(image_size=image_size, training=training)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                return self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {path}") from exc


def webcam_frame_to_rgb_uint8(frame_bgr: np.ndarray, image_size: int = 416) -> np.ndarray:
    if frame_bgr is None or frame_bgr.ndim != 3:
        raise ValueError("Expected a BGR webcam frame with shape [H,W,3].")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


def rgb_uint8_to_tensor(image: np.ndarray) -> Tensor:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected uint8 RGB image [H,W,3].")
    return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)


def load_rgb_tensor(path: str | Path, image_size: int = 416) -> Tensor:
    with Image.open(path) as image:
        transform = make_transform(image_size=image_size, training=False)
        return transform(image.convert("RGB"))


def rgb_to_ycbcr(images: Tensor) -> Tensor:
    """Convert RGB tensor in [0,1] to YCbCr. Supports [3,H,W] and [B,3,H,W]."""
    if images.ndim == 3:
        r, g, b = images[0:1], images[1:2], images[2:3]
        cat_dim = 0
    else:
        r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
        cat_dim = 1
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=cat_dim)


def ycbcr_to_rgb(images: Tensor) -> Tensor:
    """Convert YCbCr tensor to RGB [0,1]. Supports [3,H,W] and [B,3,H,W]."""
    if images.ndim == 3:
        y, cb, cr = images[0:1], images[1:2], images[2:3]
        cat_dim = 0
    else:
        y, cb, cr = images[:, 0:1], images[:, 1:2], images[:, 2:3]
        cat_dim = 1
    cb_s = cb - 0.5
    cr_s = cr - 0.5
    r = y + 1.402 * cr_s
    g = y - 0.344136 * cb_s - 0.714136 * cr_s
    b = y + 1.772 * cb_s
    return torch.cat([r, g, b], dim=cat_dim).clamp(0, 1)


def tensor_to_rgb_uint8(image: Tensor) -> np.ndarray:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("A batched image must contain exactly one item.")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected image tensor [3,H,W].")
    array = image.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return np.rint(array * 255.0).astype(np.uint8)
