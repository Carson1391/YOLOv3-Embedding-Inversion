from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


HEAD_SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (255, 13, 13),
    (255, 26, 26),
    (255, 52, 52),
)
HEAD_LENGTHS: Tuple[int, ...] = tuple(c * h * w for c, h, w in HEAD_SHAPES)
EMBEDDING_DIM = sum(HEAD_LENGTHS)  # 904,995 for original YOLOv3 at 416x416


def _validate_heads(heads: Tuple[Tensor, Tensor, Tensor]) -> None:
    if len(heads) != 3:
        raise ValueError(f"Expected 3 YOLO heads, received {len(heads)}")
    batch = heads[0].shape[0]
    for index, (head, shape) in enumerate(zip(heads, HEAD_SHAPES)):
        if head.ndim != 4 or head.shape[0] != batch or tuple(head.shape[1:]) != shape:
            raise ValueError(
                f"Head {index} must have [B,{shape[0]},{shape[1]},{shape[2]}], "
                f"received {tuple(head.shape)}"
            )


def heads_to_embedding(
    heads: Tuple[Tensor, Tensor, Tensor],
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Concatenate the three final dense YOLOv3 heads and L2-normalize once globally."""
    _validate_heads(heads)
    vector = torch.cat([head.flatten(1) for head in heads], dim=1).float()
    return F.normalize(vector, p=2.0, dim=1, eps=eps)


def normalize_embedding(embedding: Tensor, *, eps: float = 1e-12) -> Tensor:
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    if embedding.ndim != 2 or embedding.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding must have [B,{EMBEDDING_DIM}] or [{EMBEDDING_DIM}], "
            f"received {tuple(embedding.shape)}"
        )
    return F.normalize(embedding.float(), p=2.0, dim=1, eps=eps)


def embedding_to_heads(embedding: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Reshape the portable final vector back into its three known grid segments."""
    embedding = normalize_embedding(embedding)
    segments = torch.split(embedding, HEAD_LENGTHS, dim=1)
    heads = tuple(
        segment.reshape(embedding.shape[0], channels, height, width)
        for segment, (channels, height, width) in zip(segments, HEAD_SHAPES)
    )
    return heads  # type: ignore[return-value]
