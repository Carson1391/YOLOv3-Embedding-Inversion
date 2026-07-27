from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import Tensor

from embedding_ops import EMBEDDING_DIM, HEAD_SHAPES, normalize_embedding


FORMAT_NAME = "original-darknet-yolov3-l2-final-vector-v2"


def save_embedding_package(
    path: str | Path,
    embedding: Tensor,
    yolo_weights_sha256: str,
    source_name: str = "",
    source_type: str = "image",
    fps: float | None = None,
) -> None:
    embedding = normalize_embedding(embedding).detach().cpu()
    metadata: Dict[str, Any] = {
        "format": FORMAT_NAME,
        "architecture": "Joseph Redmon original Darknet YOLOv3",
        "cfg": "yolov3.cfg",
        "weights": "yolov3.weights",
        "weights_sha256": yolo_weights_sha256,
        "input_contract": "RGB float32 [0,1], stretched to 416x416",
        "vector_contract": "flatten head13, then head26, then head52; one global L2 normalization",
        "head_shapes": [list(shape) for shape in HEAD_SHAPES],
        "embedding_dim": EMBEDDING_DIM,
        "stored_dtype": "float16",
        "source_name": source_name,
        "source_type": source_type,
        "frames": int(embedding.shape[0]),
    }
    if fps is not None:
        metadata["fps"] = float(fps)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embedding=embedding.to(torch.float16).numpy(),
        metadata=np.array(json.dumps(metadata)),
    )


def load_embedding_package(
    path: str | Path,
    device: torch.device,
) -> tuple[Tensor, Dict[str, Any]]:
    path = Path(path)
    metadata: Dict[str, Any] = {}

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as package:
            if "embedding" not in package:
                raise ValueError(f"{path} has no 'embedding' array")
            array = package["embedding"]
            if "metadata" in package:
                metadata = json.loads(str(package["metadata"].item()))
                package_format = metadata.get("format")
                if package_format and package_format != FORMAT_NAME:
                    raise ValueError(f"Unsupported embedding format: {package_format}")
    elif path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, dict):
            raw = value.get("embedding", value.get("embeddings"))
            if raw is None:
                raise ValueError("PyTorch dictionary must contain 'embedding' or 'embeddings'")
            metadata = dict(value.get("metadata", {}))
            tensor = torch.as_tensor(raw)
        else:
            tensor = torch.as_tensor(value)
        tensor = normalize_embedding(tensor).to(device)
        return tensor, metadata
    else:
        raise ValueError("Supported embedding files are .npz, .npy, .pt, and .pth")

    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    tensor = normalize_embedding(tensor).to(device)
    return tensor, metadata
