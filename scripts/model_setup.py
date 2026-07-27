from __future__ import annotations

from pathlib import Path

import torch

from darknet_v3 import DarknetYOLOv3, file_sha256, freeze_model


def resolve_device(requested: str) -> torch.device:
    """Resolve a general runtime device for extraction/reconstruction/export."""
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def require_cuda_device(requested: str = "cuda") -> torch.device:
    """Resolve a CUDA device and reject CPU training."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This trainer requires a CUDA-enabled PyTorch build and an NVIDIA GPU. "
            "Install the appropriate CUDA PyTorch build before running it."
        )

    if requested == "auto":
        requested = "cuda:0"
    device = torch.device(requested)
    if device.type != "cuda":
        raise RuntimeError(f"Training requires CUDA; received --device {requested!r}.")
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {index} is unavailable; visible device count is {torch.cuda.device_count()}."
        )

    torch.cuda.set_device(index)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "This project is configured for CUDA BF16 training, but the selected GPU/PyTorch "
            "combination reports BF16 as unsupported."
        )

    # Convolutional workloads benefit from cuDNN autotuning because all training images are 416x416.
    torch.backends.cudnn.benchmark = True
    return device


def bf16_autocast():
    """CUDA BF16 mixed precision for the trainable inverse decoder."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def load_original_yolov3(
    cfg_path: str | Path,
    weights_path: str | Path,
    device: torch.device,
) -> tuple[DarknetYOLOv3, str]:
    cfg_path = Path(cfg_path)
    weights_path = Path(weights_path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    model = DarknetYOLOv3(cfg_path)
    model.load_darknet_weights(weights_path)
    freeze_model(model)
    model.to(device)
    return model, file_sha256(weights_path)
