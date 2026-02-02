"""坐标系与 SE(3) 变换工具（NumPy/Torch）。

本模块用于在不同坐标系之间做一致的点/向量变换，避免在 Lift/Models 等子模块中重复实现。

约定：
- `pose` 使用 PandaSet 的结构：包含 `position{x,y,z}` 与 `heading{w,x,y,z}`。
- `pose_to_matrix_*` 生成的矩阵表示 `sensor->world`：`p_world = R * p_sensor + t`。
- 点使用 SE(3) 变换（旋转+平移），向量/速度只使用旋转（不加平移）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _normalize_quaternion_wxyz_numpy(quat_wxyz: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """对四元数做归一化（NumPy）。

    Args:
        quat_wxyz: 四元数数组，形状为 (4,)，顺序为 (w, x, y, z)。
        eps: 数值稳定项。

    Returns:
        归一化后的四元数数组，形状为 (4,)。

    Raises:
        ValueError: 输入形状非法或范数过小。
    """
    if quat_wxyz.shape != (4,):
        raise ValueError("quat_wxyz 形状必须为 (4,)")
    norm = float(np.linalg.norm(quat_wxyz))
    if norm < eps:
        raise ValueError("四元数范数过小，无法归一化")
    return (quat_wxyz / norm).astype(np.float32)


def quat_wxyz_to_rot_matrix_numpy(quat_wxyz: np.ndarray) -> np.ndarray:
    """将四元数（w,x,y,z）转换为旋转矩阵（NumPy）。

    Args:
        quat_wxyz: 四元数数组，形状为 (4,)。

    Returns:
        旋转矩阵，形状为 (3, 3)，dtype=float32。
    """
    q = _normalize_quaternion_wxyz_numpy(quat_wxyz)
    qw, qx, qy, qz = [float(v) for v in q]
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def pose_to_matrix_numpy(pose: Mapping[str, object]) -> np.ndarray:
    """将 PandaSet pose 字典转换为 4x4 变换矩阵（NumPy）。

    Args:
        pose: 位姿字典，需包含 `position` 与 `heading` 字段。

    Returns:
        4x4 变换矩阵，表示从传感器坐标到世界坐标（sensor->world）。

    Raises:
        KeyError: pose 缺少关键字段。
        ValueError: 字段格式非法。
    """
    position = pose["position"]
    heading = pose["heading"]
    if not isinstance(position, Mapping) or not isinstance(heading, Mapping):
        raise ValueError("pose.position/pose.heading 格式非法")
    trans = np.array(
        [float(position["x"]), float(position["y"]), float(position["z"])],
        dtype=np.float32,
    )
    quat = np.array(
        [float(heading["w"]), float(heading["x"]), float(heading["y"]), float(heading["z"])],
        dtype=np.float32,
    )
    rot = quat_wxyz_to_rot_matrix_numpy(quat)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rot
    matrix[:3, 3] = trans
    return matrix


def invert_se3_numpy(transform: np.ndarray) -> np.ndarray:
    """求 SE(3) 变换的逆（NumPy）。

    Args:
        transform: 4x4 变换矩阵。

    Returns:
        4x4 的逆变换矩阵。

    Raises:
        ValueError: 输入形状非法。
    """
    if transform.shape != (4, 4):
        raise ValueError("transform 形状必须为 (4, 4)")
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    rot_inv = rot.T
    trans_inv = -rot_inv @ trans
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = rot_inv.astype(np.float32)
    out[:3, 3] = trans_inv.astype(np.float32)
    return out


def transform_points_numpy(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """使用 4x4 SE(3) 变换矩阵变换点（NumPy）。

    Args:
        points_xyz: 点坐标数组，形状为 (N, 3)。
        transform: 4x4 变换矩阵。

    Returns:
        变换后的点坐标数组，形状为 (N, 3)。

    Raises:
        ValueError: 输入形状非法。
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz 形状必须为 (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("transform 形状必须为 (4, 4)")
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    homo = np.concatenate([points_xyz, ones], axis=1)
    transformed = (transform @ homo.T).T
    return transformed[:, :3]


def transform_vectors_numpy(vectors_xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """使用旋转矩阵变换向量（NumPy）。

    Args:
        vectors_xyz: 向量数组，形状为 (N, 3)。
        rotation: 旋转矩阵，形状为 (3, 3)。

    Returns:
        变换后的向量数组，形状为 (N, 3)。

    Raises:
        ValueError: 输入形状非法。
    """
    if vectors_xyz.ndim != 2 or vectors_xyz.shape[1] != 3:
        raise ValueError("vectors_xyz 形状必须为 (N, 3)")
    if rotation.shape != (3, 3):
        raise ValueError("rotation 形状必须为 (3, 3)")
    return (rotation @ vectors_xyz.T).T.astype(np.float32, copy=False)


def _require_torch() -> None:
    """确保 torch 可用。"""
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("缺少 torch：请在训练环境中安装 PyTorch")


def quat_wxyz_to_rot_matrix_torch(quat_wxyz: "torch.Tensor", eps: float = 1e-8) -> "torch.Tensor":
    """将四元数（w,x,y,z）转换为旋转矩阵（Torch）。

    Args:
        quat_wxyz: 四元数张量，形状为 (..., 4)。
        eps: 数值稳定项。

    Returns:
        旋转矩阵张量，形状为 (..., 3, 3)。
    """
    _require_torch()
    if quat_wxyz.shape[-1] != 4:
        raise ValueError("quat_wxyz 最后一维必须为 4")
    norm = torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(eps)
    q = quat_wxyz / norm
    qw, qx, qy, qz = q.unbind(dim=-1)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def pose_to_matrix_torch(
    pose: Mapping[str, object],
    *,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    """将 PandaSet pose 字典转换为 4x4 变换矩阵（Torch）。

    Args:
        pose: 位姿字典，需包含 `position` 与 `heading` 字段。
        device: 输出张量所在 device。
        dtype: 输出张量 dtype。

    Returns:
        4x4 变换矩阵张量（sensor->world）。
    """
    _require_torch()
    position = pose["position"]
    heading = pose["heading"]
    if not isinstance(position, Mapping) or not isinstance(heading, Mapping):
        raise ValueError("pose.position/pose.heading 格式非法")
    trans = torch.tensor(
        [float(position["x"]), float(position["y"]), float(position["z"])],
        device=device,
        dtype=dtype,
    )
    quat = torch.tensor(
        [float(heading["w"]), float(heading["x"]), float(heading["y"]), float(heading["z"])],
        device=device,
        dtype=dtype,
    )
    rot = quat_wxyz_to_rot_matrix_torch(quat)
    mat = torch.eye(4, device=device, dtype=dtype)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return mat


def invert_se3_torch(transform: "torch.Tensor") -> "torch.Tensor":
    """求 SE(3) 变换的逆（Torch）。

    Args:
        transform: 4x4 变换矩阵张量。

    Returns:
        4x4 的逆变换矩阵张量。
    """
    _require_torch()
    if transform.shape != (4, 4):
        raise ValueError("transform 形状必须为 (4, 4)")
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    rot_inv = rot.transpose(0, 1)
    trans_inv = -rot_inv @ trans
    out = torch.eye(4, device=transform.device, dtype=transform.dtype)
    out[:3, :3] = rot_inv
    out[:3, 3] = trans_inv
    return out


def transform_points_torch(points_xyz: "torch.Tensor", transform: "torch.Tensor") -> "torch.Tensor":
    """使用 4x4 SE(3) 变换矩阵变换点（Torch）。

    Args:
        points_xyz: 点坐标张量，形状为 (N, 3)。
        transform: 4x4 变换矩阵张量。

    Returns:
        变换后的点坐标张量，形状为 (N, 3)。
    """
    _require_torch()
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz 形状必须为 (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("transform 形状必须为 (4, 4)")
    ones = torch.ones((points_xyz.shape[0], 1), device=points_xyz.device, dtype=points_xyz.dtype)
    homo = torch.cat([points_xyz, ones], dim=1)
    transformed = (transform @ homo.t()).t()
    return transformed[:, :3]


def transform_vectors_torch(vectors_xyz: "torch.Tensor", rotation: "torch.Tensor") -> "torch.Tensor":
    """使用旋转矩阵变换向量（Torch）。

    Args:
        vectors_xyz: 向量张量，形状为 (N, 3)。
        rotation: 旋转矩阵，形状为 (3, 3)。

    Returns:
        变换后的向量张量，形状为 (N, 3)。
    """
    _require_torch()
    if vectors_xyz.ndim != 2 or vectors_xyz.shape[1] != 3:
        raise ValueError("vectors_xyz 形状必须为 (N, 3)")
    if rotation.shape != (3, 3):
        raise ValueError("rotation 形状必须为 (3, 3)")
    return (rotation @ vectors_xyz.t()).t()


@dataclass(frozen=True)
class FrameTransformNumpy:
    """一组常用坐标变换（NumPy）。"""

    T_world_ego0: np.ndarray  # (4, 4)
    T_ego0_world: np.ndarray  # (4, 4)
    R_world_ego0: np.ndarray  # (3, 3)
    R_ego0_world: np.ndarray  # (3, 3)


def build_frame_transform_numpy(ego0_pose: Mapping[str, object]) -> FrameTransformNumpy:
    """从 ego0 pose 构建常用的 world/ego0 变换（NumPy）。

    Args:
        ego0_pose: ego0 帧的 LiDAR/ego pose（sensor->world）。

    Returns:
        FrameTransformNumpy，包含 SE(3) 与旋转矩阵的双向变换。
    """
    t_world_ego0 = pose_to_matrix_numpy(ego0_pose)
    t_ego0_world = invert_se3_numpy(t_world_ego0)
    r_world_ego0 = t_world_ego0[:3, :3].astype(np.float32)
    r_ego0_world = r_world_ego0.T.astype(np.float32)
    return FrameTransformNumpy(
        T_world_ego0=t_world_ego0,
        T_ego0_world=t_ego0_world,
        R_world_ego0=r_world_ego0,
        R_ego0_world=r_ego0_world,
    )


def build_frame_transform_torch(
    ego0_pose: Mapping[str, object],
    *,
    device: "torch.device",
    dtype: "torch.dtype",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """从 ego0 pose 构建 world/ego0 的 SE(3)/R 变换（Torch）。

    Args:
        ego0_pose: ego0 帧的 LiDAR/ego pose（sensor->world）。
        device: 输出 device。
        dtype: 输出 dtype。

    Returns:
        `(T_world_ego0, T_ego0_world, R_world_ego0, R_ego0_world)`。
    """
    _require_torch()
    t_world_ego0 = pose_to_matrix_torch(ego0_pose, device=device, dtype=dtype)
    t_ego0_world = invert_se3_torch(t_world_ego0)
    r_world_ego0 = t_world_ego0[:3, :3]
    r_ego0_world = r_world_ego0.transpose(0, 1)
    return t_world_ego0, t_ego0_world, r_world_ego0, r_ego0_world
