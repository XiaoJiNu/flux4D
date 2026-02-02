"""高斯参数的体素化与稀疏张量构建工具。

阶段3（Flux4D-base）需要将每个高斯的参数拼接成特征，并在自车坐标系（ego0）下进行体素化，
再送入稀疏 3D U-Net。该模块提供：

- NumPy 版本：用于离线检查与最小单测，不依赖 PyTorch/spconv。
- PyTorch 版本：用于训练/推理，支持返回 spconv 的 SparseConvTensor（若安装）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class VoxelizationResultNumpy:
    """NumPy 体素化输出。

    Attributes:
        voxel_coords_xyz: 体素坐标 (M, 3)，顺序为 (x, y, z)。
        voxel_features: 体素特征 (M, C)，为同体素内点特征的 mean pooling。
        point2voxel: 点到体素的映射 (N,)，无效点为 -1。
        valid_mask: 有效点掩码 (N,)。
        voxel_shape_xyz: 体素网格尺寸 (nx, ny, nz)。
    """

    voxel_coords_xyz: np.ndarray
    voxel_features: np.ndarray
    point2voxel: np.ndarray
    valid_mask: np.ndarray
    voxel_shape_xyz: Tuple[int, int, int]


def _compute_voxel_shape_xyz(
    point_cloud_range: Sequence[float],
    voxel_size: Sequence[float],
) -> Tuple[int, int, int]:
    """根据点云范围与体素大小计算体素网格尺寸（xyz 顺序）。

    Args:
        point_cloud_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]。
        voxel_size: 体素大小 [vx, vy, vz]。

    Returns:
        体素网格尺寸 (nx, ny, nz)。

    Raises:
        ValueError: 输入形状非法或体素大小非正。
    """
    if len(point_cloud_range) != 6:
        raise ValueError("point_cloud_range 必须为长度 6 的序列")
    if len(voxel_size) != 3:
        raise ValueError("voxel_size 必须为长度 3 的序列")
    if any(size <= 0 for size in voxel_size):
        raise ValueError("voxel_size 必须全部为正数")
    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in point_cloud_range]
    vx, vy, vz = [float(v) for v in voxel_size]
    nx = int(np.floor((x_max - x_min) / vx))
    ny = int(np.floor((y_max - y_min) / vy))
    nz = int(np.floor((z_max - z_min) / vz))
    return nx, ny, nz


def voxelize_points_numpy(
    points_xyz: np.ndarray,
    features: np.ndarray,
    point_cloud_range: Sequence[float],
    voxel_size: Sequence[float],
) -> VoxelizationResultNumpy:
    """对点与特征执行体素化，并在体素内做 mean pooling。

    Args:
        points_xyz: 点坐标 (N, 3)，单位米，坐标系建议为 ego0。
        features: 点特征 (N, C)。
        point_cloud_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]。
        voxel_size: 体素大小 [vx, vy, vz]。

    Returns:
        NumPy 体素化结果。

    Raises:
        ValueError: 输入形状非法。
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz 形状必须为 (N, 3)")
    if features.ndim != 2 or features.shape[0] != points_xyz.shape[0]:
        raise ValueError("features 形状必须为 (N, C) 且与 points_xyz 对齐")

    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in point_cloud_range]
    vx, vy, vz = [float(v) for v in voxel_size]
    nx, ny, nz = _compute_voxel_shape_xyz(point_cloud_range, voxel_size)

    coords = np.floor((points_xyz - np.array([x_min, y_min, z_min], dtype=np.float32)) / np.array([vx, vy, vz], dtype=np.float32)).astype(np.int64)
    valid_mask = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < nx)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < ny)
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < nz)
    )

    point2voxel = np.full((points_xyz.shape[0],), -1, dtype=np.int64)
    if not np.any(valid_mask):
        return VoxelizationResultNumpy(
            voxel_coords_xyz=np.zeros((0, 3), dtype=np.int64),
            voxel_features=np.zeros((0, features.shape[1]), dtype=features.dtype),
            point2voxel=point2voxel,
            valid_mask=valid_mask,
            voxel_shape_xyz=(nx, ny, nz),
        )

    coords_valid = coords[valid_mask]
    features_valid = features[valid_mask]
    voxel_coords, inverse = np.unique(coords_valid, axis=0, return_inverse=True)
    point2voxel[valid_mask] = inverse.astype(np.int64)

    num_voxels = voxel_coords.shape[0]
    pooled = np.zeros((num_voxels, features_valid.shape[1]), dtype=np.float32)
    counts = np.zeros((num_voxels,), dtype=np.float32)
    np.add.at(pooled, inverse, features_valid.astype(np.float32))
    np.add.at(counts, inverse, 1.0)
    pooled = pooled / np.maximum(counts[:, None], 1.0)
    return VoxelizationResultNumpy(
        voxel_coords_xyz=voxel_coords.astype(np.int64),
        voxel_features=pooled.astype(features.dtype, copy=False),
        point2voxel=point2voxel,
        valid_mask=valid_mask,
        voxel_shape_xyz=(nx, ny, nz),
    )


@dataclass(frozen=True)
class VoxelizationResultTorch:
    """PyTorch 体素化输出。

    Attributes:
        voxel_coords_xyz: 体素坐标 (M, 3)，顺序为 (x, y, z)。
        voxel_features: 体素特征 (M, C)，为同体素内点特征的 mean pooling。
        point2voxel: 点到体素的映射 (N,)，无效点为 -1。
        valid_mask: 有效点掩码 (N,)。
        voxel_shape_xyz: 体素网格尺寸 (nx, ny, nz)。
        spconv_tensor: 可选的 spconv SparseConvTensor（若安装且 build_spconv=True）。
    """

    voxel_coords_xyz: "torch.Tensor"
    voxel_features: "torch.Tensor"
    point2voxel: "torch.Tensor"
    valid_mask: "torch.Tensor"
    voxel_shape_xyz: Tuple[int, int, int]
    spconv_tensor: Optional[object] = None


def _require_torch() -> None:
    """确保 torch 可用。"""
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("缺少 torch：请在训练环境（如 gaussianstorm）中安装 PyTorch")


def _try_build_spconv_tensor(
    voxel_features: "torch.Tensor",
    voxel_coords_xyz: "torch.Tensor",
    voxel_shape_xyz: Tuple[int, int, int],
    batch_size: int,
) -> Optional[object]:
    """尝试构建 spconv 的 SparseConvTensor（若 spconv 已安装）。"""
    try:
        import spconv.pytorch as spconv  # type: ignore
    except ModuleNotFoundError:
        return None

    nx, ny, nz = voxel_shape_xyz
    coords = voxel_coords_xyz.to(dtype=torch.int32)
    coords_bzyx = torch.empty((coords.shape[0], 4), dtype=torch.int32, device=coords.device)
    coords_bzyx[:, 0] = 0
    coords_bzyx[:, 1] = coords[:, 2]
    coords_bzyx[:, 2] = coords[:, 1]
    coords_bzyx[:, 3] = coords[:, 0]
    spatial_shape_zyx = (nz, ny, nx)
    return spconv.SparseConvTensor(voxel_features, coords_bzyx, spatial_shape_zyx, batch_size)


def voxelize_points_torch(
    points_xyz: "torch.Tensor",
    features: "torch.Tensor",
    point_cloud_range: Sequence[float],
    voxel_size: Sequence[float],
    *,
    build_spconv: bool = True,
) -> VoxelizationResultTorch:
    """对点与特征执行体素化，并在体素内做 mean pooling（PyTorch 版）。

    Args:
        points_xyz: 点坐标 (N, 3)，单位米，坐标系建议为 ego0。
        features: 点特征 (N, C)。
        point_cloud_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]。
        voxel_size: 体素大小 [vx, vy, vz]。
        build_spconv: 若为 True 且安装了 spconv，则额外返回 SparseConvTensor。

    Returns:
        PyTorch 体素化结果。

    Raises:
        ValueError: 输入形状非法。
        ModuleNotFoundError: torch 缺失。
    """
    _require_torch()
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz 形状必须为 (N, 3)")
    if features.ndim != 2 or features.shape[0] != points_xyz.shape[0]:
        raise ValueError("features 形状必须为 (N, C) 且与 points_xyz 对齐")

    device = points_xyz.device
    dtype = points_xyz.dtype
    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in point_cloud_range]
    vx, vy, vz = [float(v) for v in voxel_size]
    nx, ny, nz = _compute_voxel_shape_xyz(point_cloud_range, voxel_size)

    pc_min = torch.tensor([x_min, y_min, z_min], device=device, dtype=dtype)
    vs = torch.tensor([vx, vy, vz], device=device, dtype=dtype)
    coords = torch.floor((points_xyz - pc_min) / vs).to(dtype=torch.int64)
    valid_mask = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < nx)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < ny)
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < nz)
    )
    point2voxel = torch.full((points_xyz.shape[0],), -1, device=device, dtype=torch.int64)
    if bool(torch.count_nonzero(valid_mask) == 0):
        empty_feat = torch.zeros((0, features.shape[1]), device=device, dtype=features.dtype)
        empty_coord = torch.zeros((0, 3), device=device, dtype=torch.int64)
        return VoxelizationResultTorch(
            voxel_coords_xyz=empty_coord,
            voxel_features=empty_feat,
            point2voxel=point2voxel,
            valid_mask=valid_mask,
            voxel_shape_xyz=(nx, ny, nz),
            spconv_tensor=None,
        )

    coords_valid = coords[valid_mask]
    features_valid = features[valid_mask]
    voxel_coords, inverse = torch.unique(coords_valid, dim=0, return_inverse=True)
    point2voxel[valid_mask] = inverse.to(dtype=torch.int64)
    num_voxels = int(voxel_coords.shape[0])

    pooled = torch.zeros((num_voxels, features_valid.shape[1]), device=device, dtype=features_valid.dtype)
    pooled.index_add_(0, inverse, features_valid)
    counts = torch.zeros((num_voxels,), device=device, dtype=features_valid.dtype)
    counts.index_add_(0, inverse, torch.ones((features_valid.shape[0],), device=device, dtype=features_valid.dtype))
    pooled = pooled / counts.clamp_min(1.0).unsqueeze(1)

    spconv_tensor = None
    if build_spconv:
        spconv_tensor = _try_build_spconv_tensor(pooled, voxel_coords, (nx, ny, nz), batch_size=1)
    return VoxelizationResultTorch(
        voxel_coords_xyz=voxel_coords.to(dtype=torch.int64),
        voxel_features=pooled,
        point2voxel=point2voxel,
        valid_mask=valid_mask,
        voxel_shape_xyz=(nx, ny, nz),
        spconv_tensor=spconv_tensor,
    )

