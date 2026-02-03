"""阶段3：Flux4D-base 预测器（G_init -> ΔG, V）。

该模块聚合体素化与 Sparse 3D U-Net，并在点级输出：

- `ΔG ∈ R^{N×14}`：逐高斯残差更新。
- `V ∈ R^{N×vdim}`：逐高斯速度（Flux4D-base: vdim=3）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from flux4d.models.gaussian_voxelizer import VoxelizationResultTorch, voxelize_points_torch
from flux4d.models.flux4d_unet import build_sparse_unet
from flux4d.utils.frames import (
    build_frame_transform_torch,
    transform_points_torch,
)


def _require_torch_tensor() -> None:
    """确保 torch.Tensor 类型可用。"""
    _require_torch()


def _require_torch() -> None:
    """确保 torch 可用。"""
    if torch is None or nn is None:  # pragma: no cover
        raise ModuleNotFoundError("缺少 torch：请在训练环境中安装 PyTorch")


def _normalize_quaternions(quat_wxyz: "torch.Tensor", eps: float = 1e-8) -> "torch.Tensor":
    """对四元数做归一化。

    Args:
        quat_wxyz: 四元数张量 (N, 4)，顺序为 (w, x, y, z)。
        eps: 数值稳定项。

    Returns:
        归一化后的四元数张量 (N, 4)。
    """
    norm = torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(eps)
    return quat_wxyz / norm


def _quat_conjugate_wxyz(quat_wxyz: "torch.Tensor") -> "torch.Tensor":
    """对四元数取共轭（w, -x, -y, -z）。

    Args:
        quat_wxyz: 四元数张量 (..., 4)。

    Returns:
        共轭四元数张量 (..., 4)。
    """
    _require_torch_tensor()
    if quat_wxyz.shape[-1] != 4:
        raise ValueError("quat_wxyz 最后一维必须为 4")
    w = quat_wxyz[..., 0:1]
    xyz = -quat_wxyz[..., 1:4]
    return torch.cat([w, xyz], dim=-1)


def _quat_mul_wxyz(left: "torch.Tensor", right: "torch.Tensor") -> "torch.Tensor":
    """四元数乘法（Hamilton product），顺序为 wxyz。

    Args:
        left: 左四元数 (..., 4)。
        right: 右四元数 (..., 4)。

    Returns:
        乘积四元数 (..., 4)。
    """
    _require_torch_tensor()
    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ValueError("输入四元数最后一维必须为 4")
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    w = lw * rw - lx * rx - ly * ry - lz * rz
    x = lw * rx + lx * rw + ly * rz - lz * ry
    y = lw * ry - lx * rz + ly * rw + lz * rx
    z = lw * rz + lx * ry - ly * rx + lz * rw
    return torch.stack([w, x, y, z], dim=-1)


def _rotate_motion_params_world_from_ego0(
    velocities_ego0: "torch.Tensor",
    r_world_ego0: "torch.Tensor",
) -> "torch.Tensor":
    """将 `ego0` 坐标系下的运动参数旋转到 `world`（支持多项式 vdim）。

    Args:
        velocities_ego0: 速度张量 (N, vdim)，vdim 必须是 3 的倍数。
        r_world_ego0: 旋转矩阵 (3, 3)。

    Returns:
        `world` 坐标系下的速度张量 (N, vdim)。

    Raises:
        ValueError: vdim 不是 3 的倍数或矩阵形状非法。
    """
    _require_torch_tensor()
    if r_world_ego0.shape != (3, 3):
        raise ValueError("r_world_ego0 形状必须为 (3, 3)")
    if velocities_ego0.ndim != 2:
        raise ValueError("velocities_ego0 形状必须为 (N, vdim)")
    vdim = int(velocities_ego0.shape[1])
    if vdim % 3 != 0:
        raise ValueError("velocities_ego0 的 vdim 必须为 3 的倍数")
    chunks = velocities_ego0.view(velocities_ego0.shape[0], vdim // 3, 3)
    rotated = torch.einsum("ij,nkj->nki", r_world_ego0, chunks)
    return rotated.reshape(velocities_ego0.shape[0], vdim)


@dataclass(frozen=True)
class TorchGaussianSet:
    """用于 Flux4D-base 的高斯集合（PyTorch 张量版）。"""

    positions: "torch.Tensor"  # (N, 3)
    rotations: "torch.Tensor"  # (N, 4) wxyz
    scales: "torch.Tensor"  # (N, 3)
    opacities: "torch.Tensor"  # (N,)
    colors: "torch.Tensor"  # (N, 3)
    timestamps: "torch.Tensor"  # (N,)

    def as_param_tensor(self) -> "torch.Tensor":
        """拼接为阶段3所需的 G_init 张量 (N, 14)。"""
        return torch.cat(
            [
                self.positions,
                self.rotations,
                self.scales,
                self.opacities[:, None],
                self.colors,
            ],
            dim=-1,
        )


@dataclass(frozen=True)
class Flux4DBaseOutput:
    """Flux4D-base 前向输出。"""

    gaussians: TorchGaussianSet
    velocities: "torch.Tensor"
    delta_g: "torch.Tensor"
    voxelization: VoxelizationResultTorch


class Flux4DBaseModel(nn.Module if nn is not None else object):
    """Flux4D-base：稀疏 3D U-Net 预测高斯残差与速度。"""

    def __init__(self, cfg: Mapping[str, object]) -> None:
        super().__init__()
        _require_torch()

        model_cfg = cfg.get("model")
        voxel_cfg = cfg.get("voxel")
        if not isinstance(model_cfg, Mapping):
            raise ValueError("cfg['model'] 缺失或格式非法")
        if not isinstance(voxel_cfg, Mapping):
            raise ValueError("cfg['voxel'] 缺失或格式非法")

        gaussian_dim = int(model_cfg.get("gaussian_dim", 14))
        time_dim = int(model_cfg.get("time_dim", 1))
        if gaussian_dim != 14 or time_dim != 1:
            raise ValueError("阶段3默认输入维度应为 gaussian_dim=14, time_dim=1")

        self._point_cloud_range = list(voxel_cfg["point_cloud_range"])
        self._voxel_size = list(voxel_cfg["voxel_size"])

        unet_cfg = model_cfg.get("unet")
        if not isinstance(unet_cfg, dict):
            raise ValueError("cfg['model']['unet'] 缺失或格式非法")
        self._unet = build_sparse_unet(in_channels=gaussian_dim + time_dim, cfg_dict=unet_cfg)

        head_cfg = model_cfg.get("head")
        if not isinstance(head_cfg, Mapping):
            raise ValueError("cfg['model']['head'] 缺失或格式非法")
        delta_g_dim = int(head_cfg.get("delta_g_dim", 14))
        motion_cfg = head_cfg.get("motion")
        if not isinstance(motion_cfg, Mapping):
            raise ValueError("cfg['model']['head']['motion'] 缺失或格式非法")
        poly_degree_l = int(motion_cfg.get("poly_degree_l", 0))
        vdim = 3 * (poly_degree_l + 1)

        unet_out_channels = int(unet_cfg["out_channels"])
        self._delta_head = nn.Linear(unet_out_channels, delta_g_dim)
        self._vel_head = nn.Linear(unet_out_channels, vdim)

    def forward(self, gaussians: TorchGaussianSet) -> Flux4DBaseOutput:
        """前向：体素化 -> Sparse U-Net -> 点级 head。

        Args:
            gaussians: 初始高斯集合（G_init + T）。

        Returns:
            Flux4D-base 输出，包含更新后的高斯、速度与体素化中间结果。
        """
        _require_torch()
        g_init = gaussians.as_param_tensor()
        t = gaussians.timestamps[:, None]
        point_features = torch.cat([g_init, t], dim=-1)

        voxelization = voxelize_points_torch(
            points_xyz=gaussians.positions,
            features=point_features,
            point_cloud_range=self._point_cloud_range,
            voxel_size=self._voxel_size,
            build_spconv=True,
        )
        if voxelization.spconv_tensor is None:
            raise ModuleNotFoundError("未检测到 spconv，无法构建 SparseConvTensor 以运行 Sparse U-Net")

        voxel_out = self._unet(voxelization.spconv_tensor)
        voxel_out_features = getattr(voxel_out, "features")

        n_points = gaussians.positions.shape[0]
        out_dim = int(voxel_out_features.shape[1])
        point_feat_out = torch.zeros((n_points, out_dim), device=voxel_out_features.device, dtype=voxel_out_features.dtype)
        valid_idx = torch.where(voxelization.valid_mask)[0]
        if valid_idx.numel() > 0:
            point_feat_out[valid_idx] = voxel_out_features[voxelization.point2voxel[valid_idx]]

        delta_g = self._delta_head(point_feat_out)
        velocities = self._vel_head(point_feat_out)

        g_updated = g_init + delta_g
        pos = g_updated[:, 0:3]
        rot = _normalize_quaternions(g_updated[:, 3:7])
        scale = g_updated[:, 7:10]
        opacity = g_updated[:, 10]
        color = g_updated[:, 11:14]
        gaussians_out = TorchGaussianSet(
            positions=pos,
            rotations=rot,
            scales=scale,
            opacities=opacity,
            colors=color,
            timestamps=gaussians.timestamps,
        )
        return Flux4DBaseOutput(
            gaussians=gaussians_out,
            velocities=velocities,
            delta_g=delta_g,
            voxelization=voxelization,
        )


@dataclass(frozen=True)
class Flux4DRefineOutput:
    """Flux4D refinement（r_θ）输出。"""

    delta_g: "torch.Tensor"
    voxelization: VoxelizationResultTorch


class Flux4DRefineModel(nn.Module if nn is not None else object):
    """Flux4D refinement（r_θ）：输入 (G, T, ∇G) 预测残差 ΔG。

    对齐补充材料 A.2/A.3 Algorithm 2：refinement 网络与 base 网络使用相同的稀疏 3D U-Net
    架构，但额外接收上一轮的高斯梯度 `G_grad` 作为输入特征。
    """

    def __init__(self, cfg: Mapping[str, object]) -> None:
        super().__init__()
        _require_torch()

        model_cfg = cfg.get("model")
        voxel_cfg = cfg.get("voxel")
        if not isinstance(model_cfg, Mapping):
            raise ValueError("cfg['model'] 缺失或格式非法")
        if not isinstance(voxel_cfg, Mapping):
            raise ValueError("cfg['voxel'] 缺失或格式非法")

        gaussian_dim = int(model_cfg.get("gaussian_dim", 14))
        time_dim = int(model_cfg.get("time_dim", 1))
        if gaussian_dim != 14 or time_dim != 1:
            raise ValueError("阶段5默认输入维度应为 gaussian_dim=14, time_dim=1")

        self._point_cloud_range = list(voxel_cfg["point_cloud_range"])
        self._voxel_size = list(voxel_cfg["voxel_size"])

        unet_cfg = model_cfg.get("unet")
        if not isinstance(unet_cfg, dict):
            raise ValueError("cfg['model']['unet'] 缺失或格式非法")
        in_channels = gaussian_dim + time_dim + gaussian_dim
        self._unet = build_sparse_unet(in_channels=in_channels, cfg_dict=unet_cfg)

        head_cfg = model_cfg.get("head")
        if not isinstance(head_cfg, Mapping):
            raise ValueError("cfg['model']['head'] 缺失或格式非法")
        delta_g_dim = int(head_cfg.get("delta_g_dim", 14))

        unet_out_channels = int(unet_cfg["out_channels"])
        self._delta_head = nn.Linear(unet_out_channels, delta_g_dim)

    def forward(self, gaussians: TorchGaussianSet, gaussians_grad: "torch.Tensor") -> Flux4DRefineOutput:
        """前向：体素化 (G,T,∇G) -> Sparse U-Net -> 点级 ΔG。

        Args:
            gaussians: 当前迭代的高斯集合（G）。
            gaussians_grad: 当前高斯参数的梯度 (N, 14)，作为 refinement 输入特征（∇G）。

        Returns:
            Flux4DRefineOutput。
        """
        _require_torch()
        if gaussians_grad.ndim != 2 or int(gaussians_grad.shape[1]) != 14:
            raise ValueError("gaussians_grad 形状必须为 (N, 14)")
        if int(gaussians_grad.shape[0]) != int(gaussians.positions.shape[0]):
            raise ValueError("gaussians_grad 与 gaussians 的 N 不一致")

        g_param = gaussians.as_param_tensor()
        t = gaussians.timestamps[:, None]
        point_features = torch.cat([g_param, t, gaussians_grad], dim=-1)

        voxelization = voxelize_points_torch(
            points_xyz=gaussians.positions,
            features=point_features,
            point_cloud_range=self._point_cloud_range,
            voxel_size=self._voxel_size,
            build_spconv=True,
        )
        if voxelization.spconv_tensor is None:
            raise ModuleNotFoundError("未检测到 spconv，无法构建 SparseConvTensor 以运行 Sparse U-Net")

        voxel_out = self._unet(voxelization.spconv_tensor)
        voxel_out_features = getattr(voxel_out, "features")

        n_points = gaussians.positions.shape[0]
        out_dim = int(voxel_out_features.shape[1])
        point_feat_out = torch.zeros((n_points, out_dim), device=voxel_out_features.device, dtype=voxel_out_features.dtype)
        valid_idx = torch.where(voxelization.valid_mask)[0]
        if valid_idx.numel() > 0:
            point_feat_out[valid_idx] = voxel_out_features[voxelization.point2voxel[valid_idx]]

        delta_g = self._delta_head(point_feat_out)
        return Flux4DRefineOutput(delta_g=delta_g, voxelization=voxelization)


def apply_delta_g_to_gaussians(gaussians: TorchGaussianSet, delta_g: "torch.Tensor") -> TorchGaussianSet:
    """将 `ΔG` 残差应用到高斯集合并规范化四元数。

    Args:
        gaussians: 输入高斯集合（G）。
        delta_g: 残差张量 (N, 14)。

    Returns:
        更新后的高斯集合（G + ΔG）。

    Raises:
        ValueError: 输入形状不匹配。
    """
    _require_torch_tensor()
    if delta_g.ndim != 2 or int(delta_g.shape[1]) != 14:
        raise ValueError("delta_g 形状必须为 (N, 14)")
    if int(delta_g.shape[0]) != int(gaussians.positions.shape[0]):
        raise ValueError("delta_g 与 gaussians 的 N 不一致")

    g_param = gaussians.as_param_tensor()
    updated = g_param + delta_g
    pos = updated[:, 0:3]
    rot = _normalize_quaternions(updated[:, 3:7])
    scale = updated[:, 7:10]
    opacity = updated[:, 10]
    color = updated[:, 11:14]
    return TorchGaussianSet(
        positions=pos,
        rotations=rot,
        scales=scale,
        opacities=opacity,
        colors=color,
        timestamps=gaussians.timestamps,
    )


def build_flux4d_base_model(cfg: Mapping[str, object]) -> Flux4DBaseModel:
    """从统一配置构建 Flux4D-base 模型。

    Args:
        cfg: `configs/flux4d.py` 中的 cfg 字典。

    Returns:
        Flux4DBaseModel 实例。
    """
    _require_torch()
    return Flux4DBaseModel(cfg)


def build_flux4d_refine_model(cfg: Mapping[str, object]) -> Flux4DRefineModel:
    """从统一配置构建 Flux4D refinement（r_θ）模型。

    Args:
        cfg: `configs/flux4d.py` 中的 cfg 字典。

    Returns:
        Flux4DRefineModel 实例。
    """
    _require_torch()
    return Flux4DRefineModel(cfg)


@dataclass(frozen=True)
class FrameTransformTorch:
    """world/ego0 的常用变换（Torch）。

    Attributes:
        t_world_ego0: SE(3) 变换 (4, 4)，表示 ego0->world（sensor->world）。
        t_ego0_world: SE(3) 变换 (4, 4)，表示 world->ego0。
        r_world_ego0: 旋转矩阵 (3, 3)，表示 ego0->world。
        r_ego0_world: 旋转矩阵 (3, 3)，表示 world->ego0。
        q_world_ego0: 四元数 (4,)，wxyz，表示 ego0->world。
        q_ego0_world: 四元数 (4,)，wxyz，表示 world->ego0。
    """

    t_world_ego0: "torch.Tensor"
    t_ego0_world: "torch.Tensor"
    r_world_ego0: "torch.Tensor"
    r_ego0_world: "torch.Tensor"
    q_world_ego0: "torch.Tensor"
    q_ego0_world: "torch.Tensor"


def build_frame_transform_from_ego0_pose(
    ego0_pose: Mapping[str, object],
    *,
    device: "torch.device",
    dtype: "torch.dtype",
) -> FrameTransformTorch:
    """从 ego0 pose 构建 FrameTransformTorch。

    Args:
        ego0_pose: ego0 帧的 LiDAR/ego pose（sensor->world）。
        device: 输出 device。
        dtype: 输出 dtype。

    Returns:
        FrameTransformTorch。
    """
    _require_torch_tensor()
    t_world_ego0, t_ego0_world, r_world_ego0, r_ego0_world = build_frame_transform_torch(
        ego0_pose, device=device, dtype=dtype
    )
    heading = ego0_pose.get("heading")
    if not isinstance(heading, Mapping):
        raise ValueError("ego0_pose.heading 格式非法")
    q_world_ego0 = torch.tensor(
        [
            float(heading["w"]),
            float(heading["x"]),
            float(heading["y"]),
            float(heading["z"]),
        ],
        device=device,
        dtype=dtype,
    )
    q_world_ego0 = _normalize_quaternions(q_world_ego0[None, :])[0]
    q_ego0_world = _quat_conjugate_wxyz(q_world_ego0)
    return FrameTransformTorch(
        t_world_ego0=t_world_ego0,
        t_ego0_world=t_ego0_world,
        r_world_ego0=r_world_ego0,
        r_ego0_world=r_ego0_world,
        q_world_ego0=q_world_ego0,
        q_ego0_world=q_ego0_world,
    )


@dataclass(frozen=True)
class Flux4DBaseOutputFrames:
    """带坐标系变换的 Flux4D-base 输出。"""

    gaussians_ego0: TorchGaussianSet
    gaussians_world: TorchGaussianSet
    velocities_ego0: "torch.Tensor"
    velocities_world: "torch.Tensor"
    delta_g_ego0: "torch.Tensor"
    voxelization: VoxelizationResultTorch


class Flux4DBaseModelFrames(nn.Module if nn is not None else object):
    """Flux4D-base（坐标系封装）：输入 world，高斯体素化在 ego0，速度输出到 world。

    Note:
        两份 PDF 未规定 canonical frame。本实现选择：
        - `G_init_world` 用于跨帧聚合与渲染。
        - `G_init_ego0` 用于体素化与 Sparse 3D U-Net 输入。
        - 网络输出 `V_ego0` 后旋转到 `V_world`，用于速度正则/推进/评测的统一语义。
    """

    def __init__(self, cfg: Mapping[str, object]) -> None:
        super().__init__()
        _require_torch_tensor()
        self._base = Flux4DBaseModel(cfg)

        coord_cfg = cfg.get("coord")
        if not isinstance(coord_cfg, Mapping):
            raise ValueError("cfg['coord'] 缺失或格式非法")
        self._transform_rotation = bool(coord_cfg.get("transform_gaussian_rotation", True))

    def forward(self, gaussians_world: TorchGaussianSet, frame: FrameTransformTorch) -> Flux4DBaseOutputFrames:
        """前向：world->ego0（点/旋转） -> base -> 速度旋转回 world -> ego0->world（输出高斯）。

        Args:
            gaussians_world: world 坐标系下的高斯集合（用于跨帧一致的主存储）。
            frame: ego0 对应的坐标变换。

        Returns:
            Flux4DBaseOutputFrames。
        """
        _require_torch_tensor()

        positions_ego0 = transform_points_torch(gaussians_world.positions, frame.t_ego0_world)
        rotations_ego0 = gaussians_world.rotations
        if self._transform_rotation:
            q_left = frame.q_ego0_world[None, :].expand_as(rotations_ego0)
            rotations_ego0 = _quat_mul_wxyz(q_left, rotations_ego0)
            rotations_ego0 = _normalize_quaternions(rotations_ego0)

        gaussians_ego0 = TorchGaussianSet(
            positions=positions_ego0,
            rotations=rotations_ego0,
            scales=gaussians_world.scales,
            opacities=gaussians_world.opacities,
            colors=gaussians_world.colors,
            timestamps=gaussians_world.timestamps,
        )

        base_out = self._base(gaussians_ego0)
        velocities_world = _rotate_motion_params_world_from_ego0(base_out.velocities, frame.r_world_ego0)

        positions_world = transform_points_torch(base_out.gaussians.positions, frame.t_world_ego0)
        rotations_world = base_out.gaussians.rotations
        if self._transform_rotation:
            q_left = frame.q_world_ego0[None, :].expand_as(rotations_world)
            rotations_world = _quat_mul_wxyz(q_left, rotations_world)
            rotations_world = _normalize_quaternions(rotations_world)

        gaussians_world_out = TorchGaussianSet(
            positions=positions_world,
            rotations=rotations_world,
            scales=base_out.gaussians.scales,
            opacities=base_out.gaussians.opacities,
            colors=base_out.gaussians.colors,
            timestamps=base_out.gaussians.timestamps,
        )

        return Flux4DBaseOutputFrames(
            gaussians_ego0=base_out.gaussians,
            gaussians_world=gaussians_world_out,
            velocities_ego0=base_out.velocities,
            velocities_world=velocities_world,
            delta_g_ego0=base_out.delta_g,
            voxelization=base_out.voxelization,
        )


def build_flux4d_base_model_frames(cfg: Mapping[str, object]) -> Flux4DBaseModelFrames:
    """从统一配置构建带坐标系封装的 Flux4D-base 模型。

    Args:
        cfg: `configs/flux4d.py` 中的 cfg 字典。

    Returns:
        Flux4DBaseModelFrames 实例。
    """
    _require_torch_tensor()
    return Flux4DBaseModelFrames(cfg)


def torch_gaussian_set_from_numpy(
    positions: "torch.Tensor",
    rotations: "torch.Tensor",
    scales: "torch.Tensor",
    opacities: "torch.Tensor",
    colors: "torch.Tensor",
    timestamps: "torch.Tensor",
) -> TorchGaussianSet:
    """从张量字段构造 TorchGaussianSet。

    Args:
        positions: (N, 3)。
        rotations: (N, 4) wxyz。
        scales: (N, 3)。
        opacities: (N,)。
        colors: (N, 3)。
        timestamps: (N,)。

    Returns:
        TorchGaussianSet 实例。
    """
    return TorchGaussianSet(
        positions=positions,
        rotations=rotations,
        scales=scales,
        opacities=opacities,
        colors=colors,
        timestamps=timestamps,
    )
