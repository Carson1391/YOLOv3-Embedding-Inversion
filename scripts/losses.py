from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SobelEdges(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        kernel_y = kernel_x.t()
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3), persistent=False)

    def forward(self, image: Tensor) -> Tensor:
        if image.shape[1] == 1:
            gray = image
        else:
            gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
        gx = F.conv2d(gray, self.kernel_x, padding=1)
        gy = F.conv2d(gray, self.kernel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)


@dataclass(frozen=True)
class LossWeights:
    pixel: float = 1.0
    ssim: float = 0.20
    edge: float = 0.10
    color: float = 0.05


def ssim_index(prediction: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction * prediction, window_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, window_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window_size, 1, padding) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-8)).mean()


class ReconstructionObjective(nn.Module):
    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.edges = SobelEdges()

    @staticmethod
    def charbonnier(prediction: Tensor, target: Tensor, epsilon: float = 1e-3) -> Tensor:
        return torch.sqrt((prediction - target).square() + epsilon * epsilon).mean()

    @staticmethod
    def color_moment_loss(prediction: Tensor, target: Tensor) -> Tensor:
        pred_mean = prediction.mean(dim=(-2, -1))
        target_mean = target.mean(dim=(-2, -1))
        pred_std = prediction.std(dim=(-2, -1), unbiased=False)
        target_std = target.std(dim=(-2, -1), unbiased=False)
        return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)

    def forward(self, prediction: Tensor, target: Tensor) -> tuple[Tensor, Dict[str, float]]:
        pixel = self.charbonnier(prediction, target)
        structural = 1.0 - ssim_index(prediction.float(), target.float())
        edge = F.l1_loss(self.edges(prediction), self.edges(target))
        color = self.color_moment_loss(prediction, target)

        total = (
            self.weights.pixel * pixel
            + self.weights.ssim * structural
            + self.weights.edge * edge
            + self.weights.color * color
        )
        metrics = {
            "pixel": float(pixel.detach()),
            "ssim_loss": float(structural.detach()),
            "edge": float(edge.detach()),
            "color": float(color.detach()),
            "reconstruction": float(total.detach()),
        }
        return total, metrics


class YCbCrReconstructionObjective(nn.Module):
    """
    Reconstruction objective for YCbCr images.

    Y (luminance) receives heavier weighting for pixel, SSIM, and edge losses
    since spatial structure is luminance-dominant.  Cb and Cr receive lighter
    pixel weighting.  Color-moment loss is computed across all three channels.
    """

    def __init__(self, y_pixel: float = 1.5, cc_pixel: float = 0.5,
                 y_ssim: float = 0.20, y_edge: float = 0.10,
                 color: float = 0.05) -> None:
        super().__init__()
        self.y_pixel = y_pixel
        self.cc_pixel = cc_pixel
        self.y_ssim = y_ssim
        self.y_edge = y_edge
        self.color = color
        self.edges = SobelEdges()

    @staticmethod
    def charbonnier(prediction: Tensor, target: Tensor, epsilon: float = 1e-3) -> Tensor:
        return torch.sqrt((prediction - target).square() + epsilon * epsilon).mean()

    def forward(self, prediction: Tensor, target: Tensor) -> tuple[Tensor, Dict[str, float]]:
        y_pred, cb_pred, cr_pred = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]
        y_tgt, cb_tgt, cr_tgt = target[:, 0:1], target[:, 1:2], target[:, 2:3]

        pixel_y = self.charbonnier(y_pred, y_tgt)
        pixel_cb = self.charbonnier(cb_pred, cb_tgt)
        pixel_cr = self.charbonnier(cr_pred, cr_tgt)

        y_pred_f, y_tgt_f = y_pred.float(), y_tgt.float()
        ssim_y = 1.0 - ssim_index(y_pred_f, y_tgt_f)
        edge_y = F.l1_loss(self.edges(y_pred_f), self.edges(y_tgt_f))
        color = self.color_moment_loss(prediction, target)

        total = (
            self.y_pixel * pixel_y
            + self.cc_pixel * pixel_cb
            + self.cc_pixel * pixel_cr
            + self.y_ssim * ssim_y
            + self.y_edge * edge_y
            + self.color * color
        )
        metrics = {
            "pixel_y": float(pixel_y.detach()),
            "pixel_cb": float(pixel_cb.detach()),
            "pixel_cr": float(pixel_cr.detach()),
            "ssim_y": float(ssim_y.detach()),
            "edge_y": float(edge_y.detach()),
            "color": float(color.detach()),
            "reconstruction": float(total.detach()),
        }
        return total, metrics

    @staticmethod
    def color_moment_loss(prediction: Tensor, target: Tensor) -> Tensor:
        pred_mean = prediction.mean(dim=(-2, -1))
        target_mean = target.mean(dim=(-2, -1))
        pred_std = prediction.std(dim=(-2, -1), unbiased=False)
        target_std = target.std(dim=(-2, -1), unbiased=False)
        return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


def tv_beta_regularizer(image: Tensor, beta: float = 2.0) -> Tensor:
    """
    V-beta total-variation regularizer from Mahendran & Vedaldi (CVPR 2015).

    RV_beta(x) = sum_{i,j} [ (x_{i,j+1} - x_{i,j})^2 + (x_{i+1,j} - x_{i,j})^2 ]^{beta/2}

    With beta > 1 (e.g. beta=2), large gradients are penalized more than with
    standard TV (beta=1), which distributes intensity changes across regions
    instead of concentrating them at points/curves.  This removes the "spike"
    artifacts that arise when inverting representations with subsampling
    (e.g. max-pooling in CNNs, stride in YOLOv3 detection heads).

    Applied per-channel and averaged.  Input shape: (N, C, H, W).
    """
    # Horizontal and vertical finite differences
    dh = image[:, :, :, 1:] - image[:, :, :, :-1]   # (N, C, H, W-1)
    dv = image[:, :, 1:, :] - image[:, :, :-1, :]   # (N, C, H-1, W)
    # Pad to same spatial size for clean summation
    dh = F.pad(dh, (0, 1, 0, 0))   # pad last dim on the right
    dv = F.pad(dv, (0, 0, 0, 1))   # pad second-to-last dim on the bottom
    # Element-wise gradient magnitude raised to beta
    grad_mag = (dh.square() + dv.square()).clamp_min(1e-10).pow(beta / 2.0)
    # Mean over spatial dims and channels, then average over batch
    return grad_mag.mean()


def embedding_cycle_loss(reconstructed_embedding: Tensor, target_embedding: Tensor) -> Tensor:
    """Cosine distance between two globally L2-normalized final vectors."""
    target = target_embedding.detach().float()
    reconstructed = reconstructed_embedding.float()
    return (1.0 - F.cosine_similarity(reconstructed, target, dim=1)).mean()
