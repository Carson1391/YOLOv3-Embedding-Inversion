from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn


class EmptyLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


def parse_cfg(cfg_path: str | Path) -> List[Dict[str, str]]:
    """Parse a Darknet .cfg file into ordered blocks."""
    blocks: List[Dict[str, str]] = []
    current: Dict[str, str] | None = None

    with Path(cfg_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                if current is not None:
                    blocks.append(current)
                current = {"type": line[1:-1].strip()}
                continue
            if current is None or "=" not in line:
                raise ValueError(f"Malformed Darknet cfg line: {raw_line.rstrip()}")
            key, value = line.split("=", 1)
            current[key.strip()] = value.strip()

    if current is not None:
        blocks.append(current)
    if not blocks or blocks[0].get("type") != "net":
        raise ValueError("The first cfg block must be [net].")
    return blocks


def _activation(name: str) -> nn.Module | None:
    name = name.lower()
    if name == "linear":
        return None
    if name == "leaky":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported Darknet activation: {name}")


def create_modules(blocks: Sequence[Dict[str, str]]) -> Tuple[Dict[str, str], nn.ModuleList]:
    hyperparams = dict(blocks[0])
    module_defs = blocks[1:]
    module_list = nn.ModuleList()

    input_channels = int(hyperparams.get("channels", 3))
    output_filters: List[int] = [input_channels]

    def layer_filters(current_index: int, reference: int) -> int:
        resolved = reference if reference >= 0 else current_index + reference
        if resolved < 0 or resolved >= current_index:
            raise IndexError(
                f"Route reference {reference} resolves to {resolved} at layer {current_index}."
            )
        return output_filters[resolved + 1]

    for index, block in enumerate(module_defs):
        module_type = block["type"]
        modules = nn.Sequential()
        filters = output_filters[-1]

        if module_type == "convolutional":
            batch_normalize = int(block.get("batch_normalize", "0"))
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block.get("stride", "1"))
            dilation = int(block.get("dilation", "1"))
            padding = ((kernel_size - 1) // 2) * dilation if int(block.get("pad", "0")) else 0

            conv = nn.Conv2d(
                in_channels=output_filters[-1],
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not bool(batch_normalize),
            )
            modules.add_module(f"conv_{index}", conv)
            if batch_normalize:
                modules.add_module(f"bn_{index}", nn.BatchNorm2d(filters, eps=1e-5, momentum=0.1))
            act = _activation(block.get("activation", "linear"))
            if act is not None:
                modules.add_module(f"act_{index}", act)

        elif module_type == "upsample":
            stride = int(block.get("stride", "2"))
            modules.add_module(
                f"upsample_{index}",
                nn.Upsample(scale_factor=stride, mode="nearest"),
            )

        elif module_type == "route":
            layers = [int(value.strip()) for value in block["layers"].split(",")]
            groups = int(block.get("groups", "1"))
            if len(layers) == 1:
                filters = layer_filters(index, layers[0]) // groups
            else:
                filters = sum(layer_filters(index, reference) for reference in layers)
            modules.add_module(f"route_{index}", EmptyLayer())

        elif module_type == "shortcut":
            filters = output_filters[-1]
            modules.add_module(f"shortcut_{index}", EmptyLayer())

        elif module_type == "yolo":
            filters = output_filters[-1]
            modules.add_module(f"yolo_{index}", EmptyLayer())

        else:
            raise ValueError(f"Unsupported Darknet block type: {module_type}")

        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class DarknetYOLOv3(nn.Module):
    """
    Original Darknet YOLOv3 graph loaded from yolov3.cfg.

    forward() returns the three raw convolutional tensors immediately before
    the [yolo] decode layers. For 416x416 input these are:
      [B, 255, 13, 13], [B, 255, 26, 26], [B, 255, 52, 52].
    """

    def __init__(self, cfg_path: str | Path) -> None:
        super().__init__()
        self.cfg_path = str(cfg_path)
        self.blocks = parse_cfg(cfg_path)
        self.hyperparams, self.module_list = create_modules(self.blocks)
        self.module_defs = self.blocks[1:]
        self.register_buffer("darknet_header", torch.zeros(5, dtype=torch.int32), persistent=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        layer_outputs: List[Tensor] = []
        raw_heads: List[Tensor] = []

        for index, (block, module) in enumerate(zip(self.module_defs, self.module_list)):
            module_type = block["type"]

            if module_type in {"convolutional", "upsample"}:
                x = module(x)

            elif module_type == "route":
                references = [int(value.strip()) for value in block["layers"].split(",")]
                tensors: List[Tensor] = []
                for reference in references:
                    resolved = reference if reference >= 0 else index + reference
                    tensors.append(layer_outputs[resolved])
                x = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=1)

                groups = int(block.get("groups", "1"))
                if groups > 1:
                    group_id = int(block.get("group_id", "0"))
                    channels_per_group = x.shape[1] // groups
                    start = group_id * channels_per_group
                    x = x[:, start : start + channels_per_group]

            elif module_type == "shortcut":
                reference = int(block["from"])
                resolved = reference if reference >= 0 else index + reference
                x = layer_outputs[index - 1] + layer_outputs[resolved]

            elif module_type == "yolo":
                raw_heads.append(x)

            layer_outputs.append(x)

        if len(raw_heads) != 3:
            raise RuntimeError(f"Expected 3 YOLO heads, received {len(raw_heads)}.")

        raw_heads.sort(key=lambda tensor: tensor.shape[-1])
        return raw_heads[0], raw_heads[1], raw_heads[2]

    def load_darknet_weights(self, weights_path: str | Path) -> None:
        """Load Joseph Redmon Darknet .weights into the cfg-defined graph."""
        weights_path = Path(weights_path)
        with weights_path.open("rb") as handle:
            header = np.fromfile(handle, dtype=np.int32, count=5)
            weights = np.fromfile(handle, dtype=np.float32)

        if header.size != 5:
            raise ValueError(f"Invalid Darknet header in {weights_path}")
        self.darknet_header.copy_(torch.from_numpy(header.copy()))

        pointer = 0
        for block, module in zip(self.module_defs, self.module_list):
            if block["type"] != "convolutional":
                continue

            conv = next(child for child in module.children() if isinstance(child, nn.Conv2d))
            has_bn = int(block.get("batch_normalize", "0")) == 1

            if has_bn:
                bn = next(child for child in module.children() if isinstance(child, nn.BatchNorm2d))
                count = bn.bias.numel()
                for destination in (bn.bias, bn.weight, bn.running_mean, bn.running_var):
                    chunk = weights[pointer : pointer + count]
                    if chunk.size != count:
                        raise ValueError("Darknet weights ended while reading batch normalization values.")
                    destination.data.copy_(torch.from_numpy(chunk).view_as(destination))
                    pointer += count
            else:
                if conv.bias is None:
                    raise RuntimeError("A non-BN Darknet convolution must contain a bias tensor.")
                count = conv.bias.numel()
                chunk = weights[pointer : pointer + count]
                if chunk.size != count:
                    raise ValueError("Darknet weights ended while reading convolution bias values.")
                conv.bias.data.copy_(torch.from_numpy(chunk).view_as(conv.bias))
                pointer += count

            count = conv.weight.numel()
            chunk = weights[pointer : pointer + count]
            if chunk.size != count:
                raise ValueError("Darknet weights ended while reading convolution weights.")
            conv.weight.data.copy_(torch.from_numpy(chunk).view_as(conv.weight))
            pointer += count

        if pointer != weights.size:
            remaining = weights.size - pointer
            raise ValueError(
                f"Loaded graph consumed {pointer:,} values but {remaining:,} remain. "
                "Confirm that yolov3.cfg and yolov3.weights are the matching original files."
            )


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
