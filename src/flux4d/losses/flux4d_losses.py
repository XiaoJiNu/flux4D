"""Flux4D-base 的核心损失实现（RGB/SSIM/Depth/Velocity）。

本模块实现阶段3所需的最小损失集合：
- RGB photometric L1
- SSIM（窗口可配）
- 深度 L1（基于投影 LiDAR 深度的稀疏监督）
- 速度正则（L2 范数均值）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
    import torch.nn.functional as torch_f
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    torch_f = None  # type: ignore[assignment]


def _require_torch() -> None:
    """确保 torch 可用。"""
    if torch is None or torch_f is None:  # pragma: no cover
        raise ModuleNotFoundError("缺少 torch：请在训练环境（如 gaussianstorm）中安装 PyTorch")


def _to_nchw(image_hwc: "torch.Tensor") -> "torch.Tensor":
    """将 (H,W,C) 转为 (1,C,H,W)。"""
    if image_hwc.ndim != 3 or image_hwc.shape[2] not in (1, 3):
        raise ValueError("image_hwc 形状必须为 (H, W, 1/3)")
    return image_hwc.permute(2, 0, 1).unsqueeze(0)


def photometric_l1_loss(
    pred_rgb: "torch.Tensor",
    target_rgb: "torch.Tensor",
    *,
    weight_map: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """RGB 的 L1 光度损失。

    Args:
        pred_rgb: 预测 RGB，形状为 (H, W, 3)，范围建议为 [0,1]。
        target_rgb: GT RGB，形状为 (H, W, 3)，范围为 [0,1]。
        weight_map: 可选权重图，形状为 (H, W)，用于动态重加权。

    Returns:
        标量损失。
    """
    _require_torch()
    if pred_rgb.shape != target_rgb.shape:
        raise ValueError("pred_rgb 与 target_rgb 形状必须一致")
    if pred_rgb.ndim != 3 or pred_rgb.shape[2] != 3:
        raise ValueError("pred_rgb 形状必须为 (H, W, 3)")
    diff = (pred_rgb - target_rgb).abs().mean(dim=-1)  # (H,W)
    if weight_map is None:
        return diff.mean()
    if weight_map.shape != diff.shape:
        raise ValueError("weight_map 形状必须为 (H, W)")
    valid = weight_map > 0
    denom = torch.count_nonzero(valid).clamp_min(1)
    return (diff * weight_map).sum() / denom


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    *,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    """构建用于 SSIM 的 2D Gaussian window（conv2d groups 形式）。"""
    _require_torch()
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size 必须为正奇数")
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size // 2)
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim_loss(
    pred_rgb: "torch.Tensor",
    target_rgb: "torch.Tensor",
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    weight_map: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """SSIM 损失（1-SSIM）。

    Args:
        pred_rgb: 预测 RGB，形状为 (H, W, 3)，范围建议为 [0,1]。
        target_rgb: GT RGB，形状为 (H, W, 3)，范围为 [0,1]。
        window_size: SSIM 窗口大小（奇数）。
        sigma: Gaussian sigma。
        weight_map: 可选权重图，形状为 (H, W)。

    Returns:
        标量损失（1-SSIM）。
    """
    _require_torch()
    if pred_rgb.shape != target_rgb.shape:
        raise ValueError("pred_rgb 与 target_rgb 形状必须一致")
    x = _to_nchw(pred_rgb)
    y = _to_nchw(target_rgb)
    channels = int(x.shape[1])
    window = _gaussian_window(
        window_size,
        sigma,
        channels,
        device=x.device,
        dtype=x.dtype,
    )
    padding = window_size // 2
    mu_x = torch_f.conv2d(x, window, padding=padding, groups=channels)
    mu_y = torch_f.conv2d(y, window, padding=padding, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = torch_f.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = torch_f.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = torch_f.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    c1 = 0.01 * 0.01
    c2 = 0.03 * 0.03
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / denominator.clamp_min(1e-12)  # (1,C,H,W)
    ssim_map = ssim_map.mean(dim=1, keepdim=True)  # (1,1,H,W)

    if weight_map is None:
        ssim_value = ssim_map.mean()
    else:
        if weight_map.ndim != 2:
            raise ValueError("weight_map 形状必须为 (H, W)")
        w = weight_map.to(device=ssim_map.device, dtype=ssim_map.dtype)[None, None, ...]
        denom = w.sum().clamp_min(1e-8)
        ssim_value = (ssim_map * w).sum() / denom
    return 1.0 - ssim_value


def depth_l1_loss(
    pred_depth: "torch.Tensor",
    target_depth: "torch.Tensor",
    *,
    valid_mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """深度 L1 损失（仅在有效像素上计算）。

    Args:
        pred_depth: 预测深度，形状为 (H, W)。
        target_depth: GT 深度，形状为 (H, W)。
        valid_mask: 可选有效掩码，形状为 (H, W) 且 dtype=bool。

    Returns:
        标量损失。
    """
    _require_torch()
    if pred_depth.shape != target_depth.shape:
        raise ValueError("pred_depth 与 target_depth 形状必须一致")
    if pred_depth.ndim != 2:
        raise ValueError("pred_depth 形状必须为 (H, W)")
    diff = (pred_depth - target_depth).abs()
    if valid_mask is None:
        return diff.mean()
    if valid_mask.shape != diff.shape:
        raise ValueError("valid_mask 形状必须为 (H, W)")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask dtype 必须为 torch.bool")
    num = torch.count_nonzero(valid_mask).clamp_min(1)
    return diff[valid_mask].sum() / num


def velocity_regularization(velocities_world: "torch.Tensor") -> "torch.Tensor":
    """速度正则项：mean(||v||_2)。

    Args:
        velocities_world: 速度张量，形状为 (N, vdim)，vdim 必须为 3 的倍数。

    Returns:
        标量损失。
    """
    _require_torch()
    if velocities_world.ndim != 2:
        raise ValueError("velocities_world 形状必须为 (N, vdim)")
    vdim = int(velocities_world.shape[1])
    if vdim % 3 != 0:
        raise ValueError("velocities_world 的 vdim 必须为 3 的倍数")
    chunks = velocities_world.view(velocities_world.shape[0], vdim // 3, 3)
    norms = torch.linalg.norm(chunks, dim=-1)  # (N, poly_terms)
    return norms.mean()


@dataclass(frozen=True)
class Flux4DBaseLosses:
    """Flux4D-base 的损失分解。"""

    total: "torch.Tensor"
    rgb_l1: "torch.Tensor"
    ssim: "torch.Tensor"
    depth_l1: "torch.Tensor"
    vel: "torch.Tensor"


def compute_flux4d_base_losses(
    pred_rgb: "torch.Tensor",
    target_rgb: "torch.Tensor",
    rgb_weight_map: Optional["torch.Tensor"],
    pred_depth: Optional["torch.Tensor"],
    target_depth: Optional["torch.Tensor"],
    target_depth_valid: Optional["torch.Tensor"],
    velocities_world: "torch.Tensor",
    *,
    lambda_rgb: float,
    lambda_ssim: float,
    lambda_depth: float,
    lambda_vel: float,
    ssim_window: int = 11,
) -> Flux4DBaseLosses:
    """计算 Flux4D-base 的总损失与分项。

    Args:
        pred_rgb: 预测 RGB (H, W, 3)。
        target_rgb: GT RGB (H, W, 3)。
        rgb_weight_map: 可选的 RGB 像素权重图 (H, W)，用于阶段5的速度重加权。
        pred_depth: 预测深度 (H, W)，可为 None（当 render_mode 不输出深度时）。
        target_depth: GT 深度 (H, W)，可为 None（当不使用深度监督时）。
        target_depth_valid: GT 深度有效掩码 (H, W)，可为 None。
        velocities_world: 速度张量 (N, vdim)。
        lambda_rgb: RGB L1 权重。
        lambda_ssim: SSIM 权重。
        lambda_depth: Depth L1 权重。
        lambda_vel: 速度正则权重。
        ssim_window: SSIM 窗口大小。

    Returns:
        Flux4DBaseLosses。
    """
    _require_torch()
    rgb_l1 = photometric_l1_loss(pred_rgb, target_rgb, weight_map=rgb_weight_map)
    ssim = ssim_loss(pred_rgb, target_rgb, window_size=ssim_window)
    if pred_depth is None or target_depth is None:
        depth_l1 = torch.zeros((), device=pred_rgb.device, dtype=pred_rgb.dtype)
    else:
        depth_l1 = depth_l1_loss(pred_depth, target_depth, valid_mask=target_depth_valid)
    vel = velocity_regularization(velocities_world)
    total = (
        float(lambda_rgb) * rgb_l1
        + float(lambda_ssim) * ssim
        + float(lambda_depth) * depth_l1
        + float(lambda_vel) * vel
    )
    return Flux4DBaseLosses(total=total, rgb_l1=rgb_l1, ssim=ssim, depth_l1=depth_l1, vel=vel)
