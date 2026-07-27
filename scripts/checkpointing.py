from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from inverse_decoder import DecoderConfig, InverseYOLOv3Transformer


def save_checkpoint(
    path: str | Path,
    decoder: InverseYOLOv3Transformer,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    global_step: int,
    best_validation: float,
    yolo_weights_sha256: str,
    extra: Dict[str, Any] | None = None,
    scheduler: Any | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "decoder_state": decoder.state_dict(),
        "decoder_config": decoder.config.to_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation": float(best_validation),
        "yolo_weights_sha256": yolo_weights_sha256,
        "format": "original-darknet-yolov3-l2-final-vector-v2",
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra:
        payload["extra"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _migrate_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remap legacy key names to current model structure.

    v3 and earlier used `geneo.*` for a single GENEO on head13.
    Current model uses `geneo13.*`, `geneo26.*`, `geneo52.*`.
    """
    migrated = {}
    for key, value in state_dict.items():
        if key.startswith("geneo."):
            migrated["geneo13." + key[len("geneo."):]] = value
        else:
            migrated[key] = value
    return migrated


def load_decoder_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[InverseYOLOv3Transformer, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = DecoderConfig(**payload["decoder_config"])
    decoder = InverseYOLOv3Transformer(config).to(device)
    state_dict = _migrate_state_dict_keys(payload["decoder_state"])
    missing, unexpected = decoder.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[checkpoint] Missing keys (randomly initialized): {missing}")
    if unexpected:
        print(f"[checkpoint] Unexpected keys (ignored): {unexpected}")
    return decoder, payload
