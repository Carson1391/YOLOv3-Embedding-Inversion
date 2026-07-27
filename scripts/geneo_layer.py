from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn


def _unit_vectors(p: int) -> List[Tuple[int, int]]:
    """Find unit vectors w=(a,b) in Z_p x Z_p with a^2+b^2 = 1 (mod p), identifying w and -w."""
    directions: List[Tuple[int, int]] = []
    seen: set = set()
    for a in range(p):
        for b in range(p):
            if (a * a + b * b) % p == 1:
                neg = ((p - a) % p, (p - b) % p)
                key = tuple(sorted([(a, b), neg]))
                if key not in seen:
                    seen.add(key)
                    directions.append((a, b))
    return directions


def _build_operator(p: int, w: Tuple[int, int]) -> Tensor:
    """Build the p^2 x p^2 matrix for the GENEO projection-adjoint composition.

    For direction w_bar=(w1,w2), w_bar_perp=(-w2,w1):
      Projection  F(phi)(y) = (1/p) * sum_t phi(y*w_bar + t*w_bar_perp)
      Adjoint     F*(psi)(z) = psi(z . w_bar) / p
      Composition F* o F  averages along lines perpendicular to w_bar.

    Each row has exactly p nonzero entries, each equal to 1/p^2.
    The operator is translation-equivariant and non-expansive (Lipschitz <= 1).
    """
    w1, w2 = w
    wp1, wp2 = (-w2 % p, w1 % p)
    op = torch.zeros(p * p, p * p)
    for z1 in range(p):
        for z2 in range(p):
            out_idx = z1 * p + z2
            s = (z1 * w1 + z2 * w2) % p
            for t in range(p):
                in_z1 = (s * w1 + t * wp1) % p
                in_z2 = (s * w2 + t * wp2) % p
                in_idx = in_z1 * p + in_z2
                op[out_idx, in_idx] += 1.0 / (p * p)
    return op


class GENEOLayer(nn.Module):
    """
    Group Equivariant Non-Expansive Operator layer on Z_p x Z_p.

    Implements the linear GENEO construction from Theorem 5.2 of
    "An Algebraic Representation Theorem for Linear GENEOs" using
    the translation group on Z_p x Z_p with p prime.

    Each unit vector w_bar defines a directional averaging operator
    (projection + adjoint) that is translation-equivariant and
    non-expansive (Lipschitz <= 1).  The layer applies all such
    operators and combines their outputs with a learnable 1x1
    convolution constrained by spectral normalization to preserve
    approximate non-expansivity.

    For p=13 there are 6 distinct unit-vector directions, giving
    6 fixed equivariant operators.  The input and output shapes
    are both [B, C, p, p].
    """

    def __init__(self, in_channels: int, p: int = 13) -> None:
        super().__init__()
        self.p = p
        self.in_channels = in_channels

        directions = _unit_vectors(p)
        self.n_directions = len(directions)

        operators = [_build_operator(p, w) for w in directions]
        self.register_buffer("operators", torch.stack(operators))

        self.combine = nn.utils.spectral_norm(
            nn.Conv2d(in_channels * (self.n_directions + 1), in_channels, 1, bias=False)
        )
        self._init_identity_combine()

    def _init_identity_combine(self) -> None:
        """Init combine conv as identity pass-through on the first (identity) branch,
        zeroing all directional operator branches. Makes the layer a no-op at init."""
        with torch.no_grad():
            w = self.combine.weight_orig  # [C, (n_dir+1)*C, 1, 1]
            C = self.in_channels
            w.zero_()
            # Identity: first C input channels -> output, rest zeroed
            for c in range(C):
                w[c, c, 0, 0] = 1.0

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert H == W == self.p, f"Expected {self.p}x{self.p} input, got {H}x{W}"

        x_flat = x.flatten(2)  # [B, C, p^2]

        outputs: List[Tensor] = [x_flat]
        for i in range(self.n_directions):
            outputs.append(torch.matmul(x_flat, self.operators[i].t()))

        combined = torch.cat(outputs, dim=1)  # [B, (n_dir+1)*C, p^2]
        combined = combined.reshape(B, (self.n_directions + 1) * C, self.p, self.p)
        return self.combine(combined)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, p={self.p}, directions={self.n_directions}"


class TiledGENEOLayer(nn.Module):
    """
    Tiled GENEO layer that applies the p x p GENEO operators to arbitrary
    grid sizes by tiling into non-overlapping p x p patches.

    For a grid of size H x W where H and W are multiples of p:
      - The grid is split into (H/p) x (W/p) patches of size p x p
      - The same fixed GENEO operators are applied to each patch
      - Outputs are reassembled back to H x W
      - A learnable spectrally-normalized 1x1 conv combines directions

    This preserves translation-equivariance within each patch and
    non-expansivity (Lipschitz <= 1) per patch.  Cross-patch boundaries
    are handled by the subsequent convolutional layers.

    For p=13:
      - 13x13 grid: 1 patch (identical to GENEOLayer)
      - 26x26 grid: 4 patches (2x2 tiling)
      - 52x52 grid: 16 patches (4x4 tiling)
    """

    def __init__(self, in_channels: int, p: int = 13, grid_size: int = 13) -> None:
        super().__init__()
        assert grid_size % p == 0, f"grid_size {grid_size} must be divisible by p {p}"
        self.p = p
        self.grid_size = grid_size
        self.in_channels = in_channels
        self.n_tiles = (grid_size // p) ** 2

        directions = _unit_vectors(p)
        self.n_directions = len(directions)

        # Shared operators across all tiles -- same equivariant structure
        operators = [_build_operator(p, w) for w in directions]
        self.register_buffer("operators", torch.stack(operators))

        # Combine directions with spectral norm to preserve non-expansivity
        self.combine = nn.utils.spectral_norm(
            nn.Conv2d(in_channels * (self.n_directions + 1), in_channels, 1, bias=False)
        )
        self._init_identity_combine()

    def _init_identity_combine(self) -> None:
        """Init combine conv as identity pass-through on the first (identity) branch,
        zeroing all directional operator branches. Makes the layer a no-op at init."""
        with torch.no_grad():
            w = self.combine.weight_orig  # [C, (n_dir+1)*C, 1, 1]
            C = self.in_channels
            w.zero_()
            # Identity: first C input channels -> output, rest zeroed
            for c in range(C):
                w[c, c, 0, 0] = 1.0

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert H == W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"

        p = self.p
        n = self.grid_size // p  # number of patches per side

        # Reshape into tiles: [B, C, n, p, n, p] -> [B*n*n, C, p, p]
        x_tiled = x.reshape(B, C, n, p, n, p)
        x_tiled = x_tiled.permute(0, 2, 4, 1, 3, 5)  # [B, n, n, C, p, p]
        x_tiled = x_tiled.reshape(B * self.n_tiles, C, p, p)

        # Apply GENEO operators to each tile
        x_flat = x_tiled.flatten(2)  # [B*n_tiles, C, p^2]

        outputs: List[Tensor] = [x_flat]
        for i in range(self.n_directions):
            outputs.append(torch.matmul(x_flat, self.operators[i].t()))

        combined = torch.cat(outputs, dim=1)  # [B*n_tiles, (n_dir+1)*C, p^2]
        combined = combined.reshape(B * self.n_tiles, (self.n_directions + 1) * C, p, p)
        combined = self.combine(combined)  # [B*n_tiles, C, p, p]

        # Reassemble tiles back to full grid: [B, n, n, C, p, p] -> [B, C, n*p, n*p]
        combined = combined.reshape(B, n, n, C, p, p)
        combined = combined.permute(0, 3, 1, 4, 2, 5)  # [B, C, n, p, n, p]
        combined = combined.reshape(B, C, self.grid_size, self.grid_size)

        return combined

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, p={self.p}, "
                f"grid_size={self.grid_size}, tiles={self.n_tiles}, "
                f"directions={self.n_directions}")
