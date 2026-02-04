"""图像/深度重建指标（PSNR/SSIM/Depth RMSE）。

该模块主要用于阶段6评测：对渲染结果与真实观测进行量化对比。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from flux4d.losses.flux4d_losses import ssim_loss


def _require_torch() -> None:
    """确保 torch 可用。"""
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("缺少 torch：请在训练/评测环境中安装 PyTorch")


def compute_psnr_torch(
    pred_rgb: "torch.Tensor",
    target_rgb: "torch.Tensor",
    *,
    eps: float = 1e-12,
    valid_mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """计算 PSNR（dB，MAX=1）。

    Args:
        pred_rgb: 预测 RGB，形状为 (H, W, 3)，范围建议为 [0,1]。
        target_rgb: GT RGB，形状为 (H, W, 3)，范围为 [0,1]。
        eps: 数值稳定项。
        valid_mask: 可选有效掩码 (H, W)，dtype=bool；仅在 mask=True 的像素上计算。

    Returns:
        标量 PSNR（torch.Tensor）。
    """
    _require_torch()
    if pred_rgb.shape != target_rgb.shape:
        raise ValueError("pred_rgb 与 target_rgb 形状必须一致")
    if pred_rgb.ndim != 3 or int(pred_rgb.shape[2]) != 3:
        raise ValueError("pred_rgb 形状必须为 (H, W, 3)")
    diff2 = (pred_rgb - target_rgb) ** 2
    if valid_mask is None:
        mse = diff2.mean().clamp_min(float(eps))
    else:
        if valid_mask.shape != pred_rgb.shape[:2]:
            raise ValueError("valid_mask 形状必须为 (H, W)")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask dtype 必须为 torch.bool")
        if torch.count_nonzero(valid_mask) == 0:
            raise ValueError("valid_mask 至少需要包含一个 True 像素")
        # 仅在 mask=True 的像素上计算，并对 RGB 三通道一起做均值，确保与 unmasked 情况口径一致。
        mse = diff2[valid_mask].mean().clamp_min(float(eps))
    return -10.0 * torch.log10(mse)


def compute_ssim_value_torch(
    pred_rgb: "torch.Tensor",
    target_rgb: "torch.Tensor",
    *,
    window_size: int = 11,
    weight_map: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """计算 SSIM（值越大越好）。

    Args:
        pred_rgb: 预测 RGB (H, W, 3)。
        target_rgb: GT RGB (H, W, 3)。
        window_size: SSIM 窗口大小（奇数）。
        weight_map: 可选权重图 (H, W)，用于 dynamic-only 的加权 SSIM。

    Returns:
        标量 SSIM 值（torch.Tensor）。
    """
    _require_torch()
    return 1.0 - ssim_loss(pred_rgb, target_rgb, window_size=window_size, weight_map=weight_map)


def compute_depth_rmse_torch(
    pred_depth: "torch.Tensor",
    target_depth: "torch.Tensor",
    *,
    valid_mask: "torch.Tensor",
    eps: float = 1e-12,
) -> "torch.Tensor":
    """计算深度 RMSE（仅在有效像素上）。

    Args:
        pred_depth: 预测深度 (H, W)。
        target_depth: GT 深度 (H, W)。
        valid_mask: 有效像素掩码 (H, W)，dtype=bool。
        eps: 数值稳定项。

    Returns:
        标量 RMSE（torch.Tensor）。
    """
    _require_torch()
    if pred_depth.shape != target_depth.shape:
        raise ValueError("pred_depth 与 target_depth 形状必须一致")
    if pred_depth.ndim != 2:
        raise ValueError("pred_depth 形状必须为 (H, W)")
    if valid_mask.shape != pred_depth.shape:
        raise ValueError("valid_mask 形状必须为 (H, W)")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask dtype 必须为 torch.bool")
    if torch.count_nonzero(valid_mask) == 0:
        return torch.zeros((), device=pred_depth.device, dtype=pred_depth.dtype)
    diff2 = (pred_depth - target_depth) ** 2
    mse = diff2[valid_mask].mean().clamp_min(float(eps))
    return torch.sqrt(mse)


@dataclass(frozen=True)
class ImageEvalMetrics:
    """单帧图像评测指标。"""

    psnr: float
    ssim: float
    depth_rmse: float
