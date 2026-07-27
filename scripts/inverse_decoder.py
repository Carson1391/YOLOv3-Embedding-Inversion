from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from embedding_ops import EMBEDDING_DIM, embedding_to_heads
from geneo_layer import GENEOLayer, TiledGENEOLayer


@dataclass(frozen=True)
class DecoderConfig:
    head_channels: int = 255
    embedding_dim: int = EMBEDDING_DIM
    stem_channels: int = 64
    d_model: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_dropout: float = 0.10
    output_size: int = 416
    use_geneo: bool = True

    def to_dict(self) -> Dict[str, int | float]:
        return asdict(self)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 8,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        valid_groups = min(groups, out_channels)
        while out_channels % valid_groups != 0:
            valid_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(valid_groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels, 3)
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv2 = nn.Sequential(*layers)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class HeadStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_channels, out_channels, 1),
            ResidualBlock(out_channels, dropout=0.05),
            ResidualBlock(out_channels, dropout=0.05),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UpFuseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0) -> None:
        super().__init__()
        self.project = ConvGNAct(in_channels, out_channels, 3)
        merged_channels = out_channels + skip_channels
        self.fuse = nn.Sequential(
            ConvGNAct(merged_channels, out_channels, 3),
            ResidualBlock(out_channels, dropout=0.03),
        )

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.project(x)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                raise ValueError(f"Skip shape {skip.shape[-2:]} does not match {x.shape[-2:]}")
            x = torch.cat((x, skip), dim=1)
        return self.fuse(x)


class InverseYOLOv3Transformer(nn.Module):
    """
    Converts one L2-normalized final YOLOv3 vector into a 416x416 RGB image.

    The vector is split internally into the known 13x13, 26x26, and 52x52
    final dense-grid segments. The transformer works at the 13x13 global
    bottleneck; convolutional lateral paths retain the finer grid structure.
    """

    def __init__(self, config: DecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or DecoderConfig()
        c = self.config.stem_channels
        d = self.config.d_model

        self.stem13 = HeadStem(self.config.head_channels, c)
        self.stem26 = HeadStem(self.config.head_channels, c)
        self.stem52 = HeadStem(self.config.head_channels, c)

        self.geneo13 = GENEOLayer(c, p=13) if self.config.use_geneo else None
        self.geneo26 = TiledGENEOLayer(c, p=13, grid_size=26) if self.config.use_geneo else None
        self.geneo52 = TiledGENEOLayer(c, p=13, grid_size=52) if self.config.use_geneo else None

        self.down26 = nn.Sequential(
            ConvGNAct(c, c, 3, stride=2),
            ResidualBlock(c),
        )
        self.down52 = nn.Sequential(
            ConvGNAct(c, c, 3, stride=2),
            ConvGNAct(c, c, 3, stride=2),
            ResidualBlock(c),
        )
        self.global_fuse = nn.Sequential(
            ConvGNAct(c * 3, d, 1),
            ResidualBlock(d, dropout=0.05),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.transformer_heads,
            dim_feedforward=d * 4,
            dropout=self.config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.transformer_layers,
            norm=nn.LayerNorm(d),
        )
        self.position = nn.Parameter(torch.zeros(1, 13 * 13, d))

        self.up26 = UpFuseBlock(d, 160, skip_channels=c)
        self.up52 = UpFuseBlock(160, 112, skip_channels=c)
        self.up104 = UpFuseBlock(112, 80)
        self.up208 = UpFuseBlock(80, 48)
        self.up416 = UpFuseBlock(48, 32)
        self.output = nn.Sequential(
            ResidualBlock(32),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.position, std=0.02)

        # Re-init GENEO combine layers as identity pass-through after _init_weights
        # overwrites them with Xavier. This makes new GENEO layers no-ops at start,
        # preserving pretrained features when warm-starting.
        for geneo in [self.geneo13, self.geneo26, self.geneo52]:
            if geneo is not None:
                geneo._init_identity_combine()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, embedding: Tensor) -> Tensor:
        if embedding.shape[-1] != self.config.embedding_dim:
            raise ValueError(
                f"Expected final embedding dimension {self.config.embedding_dim}, "
                f"received {embedding.shape[-1]}"
            )
        head13, head26, head52 = embedding_to_heads(embedding)

        p13 = self.stem13(head13)
        if self.geneo13 is not None:
            p13 = self.geneo13(p13)
        p26 = self.stem26(head26)
        if self.geneo26 is not None:
            p26 = self.geneo26(p26)
        p52 = self.stem52(head52)
        if self.geneo52 is not None:
            p52 = self.geneo52(p52)

        global_map = self.global_fuse(torch.cat((p13, self.down26(p26), self.down52(p52)), dim=1))
        batch, channels, height, width = global_map.shape
        tokens = global_map.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens + self.position)
        global_map = tokens.transpose(1, 2).reshape(batch, channels, height, width)

        x = self.up26(global_map, p26)
        x = self.up52(x, p52)
        x = self.up104(x)
        x = self.up208(x)
        x = self.up416(x)
        image = self.output(x)

        if image.shape[-2:] != (self.config.output_size, self.config.output_size):
            image = F.interpolate(
                image,
                size=(self.config.output_size, self.config.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return image
