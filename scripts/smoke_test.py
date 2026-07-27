from __future__ import annotations

from pathlib import Path

import torch

from darknet_v3 import parse_cfg
from embedding_ops import EMBEDDING_DIM, heads_to_embedding
from inverse_decoder import InverseYOLOv3Transformer


def main() -> None:
    blocks = parse_cfg(Path(__file__).with_name("yolov3.cfg"))
    assert sum(block.get("type") == "yolo" for block in blocks) == 3

    heads = (
        torch.randn(1, 255, 13, 13),
        torch.randn(1, 255, 26, 26),
        torch.randn(1, 255, 52, 52),
    )
    embedding = heads_to_embedding(heads)
    assert tuple(embedding.shape) == (1, EMBEDDING_DIM)
    assert torch.allclose(embedding.norm(dim=1), torch.ones(1), atol=1e-5)

    decoder = InverseYOLOv3Transformer().to("meta")
    output = decoder(torch.empty(1, EMBEDDING_DIM, device="meta"))
    assert tuple(output.shape) == (1, 3, 416, 416)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
