"""Flux4D 的时间推进与 gsplat 渲染封装。

本模块提供阶段3/阶段4共用的“时间推进 + 渲染”基础能力：
- 线性运动推进（Flux4D-base）。
- 高斯参数激活（sigmoid/softplus + clamp）。
- 使用 gsplat 进行 RGB/Depth 渲染（pinhole 相机）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

try:
    import torch
    import torch.nn.functional as torch_f
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    torch_f = None  # type: ignore[assignment]

from flux4d.models.flux4d_model import TorchGaussianSet
from flux4d.utils.frames import invert_se3_torch, pose_to_matrix_torch


def _require_torch() -> None:
    """确保 torch 可用。"""
    if torch is None or torch_f is None:  # pragma: no cover
        raise ModuleNotFoundError("缺少 torch：请在训练环境（如 gaussianstorm）中安装 PyTorch")


def _require_gsplat() -> object:
    """确保 gsplat 可用并返回模块对象。"""
    try:
        import gsplat  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("缺少 gsplat：请在训练环境中安装 gsplat") from exc
    return gsplat


@dataclass(frozen=True)
class CameraPinholeTorch:
    """Torch 版 pinhole 相机参数。

    Attributes:
        viewmat_world_to_cam: 视图矩阵 world->camera，形状为 (4, 4)。
        k: 内参矩阵，形状为 (3, 3)。
        width: 图像宽度（像素）。
        height: 图像高度（像素）。
    """

    viewmat_world_to_cam: "torch.Tensor"
    k: "torch.Tensor"
    width: int
    height: int


@dataclass(frozen=True)
class RenderOutputsTorch:
    """gsplat 渲染输出（Torch）。"""

    rgb: Optional["torch.Tensor"]  # (H, W, 3) or None
    depth: Optional["torch.Tensor"]  # (H, W) or None
    alpha: "torch.Tensor"  # (H, W)
    meta: Dict[str, object]


def build_pinhole_camera_from_pandaset(
    intrinsics: Mapping[str, object],
    pose_cam: Mapping[str, object],
    image_size: Tuple[int, int],
    *,
    device: "torch.device",
    dtype: "torch.dtype",
) -> CameraPinholeTorch:
    """将 PandaSet 相机参数转换为 gsplat 需要的相机张量。

    Args:
        intrinsics: PandaSet 相机内参（需包含 fx/fy/cx/cy）。
        pose_cam: 相机位姿（sensor->world）。
        image_size: 图像尺寸 (width, height)。
        device: 输出张量 device。
        dtype: 输出张量 dtype。

    Returns:
        CameraPinholeTorch。
    """
    _require_torch()
    width, height = image_size
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    k = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    cam_to_world = pose_to_matrix_torch(pose_cam, device=device, dtype=dtype)
    world_to_cam = invert_se3_torch(cam_to_world)
    return CameraPinholeTorch(
        viewmat_world_to_cam=world_to_cam,
        k=k,
        width=int(width),
        height=int(height),
    )


def apply_linear_motion(
    gaussians_world: TorchGaussianSet,
    velocities_world: "torch.Tensor",
    t_target: "torch.Tensor",
) -> TorchGaussianSet:
    """对高斯中心应用线性运动推进（Flux4D-base）。

    Args:
        gaussians_world: world 坐标系下的高斯集合（位置为 p_world(t_i)）。
        velocities_world: world 坐标系速度张量 (N, vdim)，至少包含前三维线速度。
        t_target: 目标时间标量或形状为 (N,) 的张量，单位为归一化时间（[0,1]）。

    Returns:
        推进到 t_target 的高斯集合（仅 positions 更新，其余字段保持不变）。

    Raises:
        ValueError: 输入形状不匹配。
    """
    _require_torch()
    if velocities_world.ndim != 2:
        raise ValueError("velocities_world 形状必须为 (N, vdim)")
    if velocities_world.shape[0] != gaussians_world.positions.shape[0]:
        raise ValueError("velocities_world 与 gaussians_world 的 N 不一致")
    if int(velocities_world.shape[1]) < 3:
        raise ValueError("velocities_world 的 vdim 必须至少为 3")

    timestamps = gaussians_world.timestamps
    if t_target.ndim == 0:
        delta_t = t_target - timestamps
    elif t_target.ndim == 1 and t_target.shape[0] == timestamps.shape[0]:
        delta_t = t_target - timestamps
    else:
        raise ValueError("t_target 必须为标量或形状为 (N,) 的张量")

    positions = gaussians_world.positions + velocities_world[:, 0:3] * delta_t[:, None]
    return TorchGaussianSet(
        positions=positions,
        rotations=gaussians_world.rotations,
        scales=gaussians_world.scales,
        opacities=gaussians_world.opacities,
        colors=gaussians_world.colors,
        timestamps=gaussians_world.timestamps,
    )


def activate_gaussians_for_render(
    gaussians: TorchGaussianSet,
    cfg: Mapping[str, object],
) -> TorchGaussianSet:
    """将高斯参数从“训练参数空间”激活到“渲染空间”。

    Args:
        gaussians: 高斯集合（通常包含 logit/softplus 参数）。
        cfg: `configs/flux4d.py` 中的 cfg 字典（读取 init.scale_transform 与 render.activation）。

    Returns:
        激活后的高斯集合：colors/opacities 在 [0,1]，scales 为正并按配置 clamp。

    Raises:
        ValueError: 配置字段缺失或格式非法。
    """
    _require_torch()
    init_cfg = cfg.get("init")
    render_cfg = cfg.get("render")
    if not isinstance(init_cfg, Mapping):
        raise ValueError("cfg['init'] 缺失或格式非法")
    if not isinstance(render_cfg, Mapping):
        raise ValueError("cfg['render'] 缺失或格式非法")

    scale_transform = init_cfg.get("scale_transform")
    if not isinstance(scale_transform, Mapping):
        raise ValueError("cfg['init']['scale_transform'] 缺失或格式非法")
    use_log = bool(scale_transform.get("use_log", False))
    use_softplus = bool(scale_transform.get("use_softplus", True))
    beta = float(scale_transform.get("softplus_beta", 1.0))
    eps = float(scale_transform.get("eps", 1e-6))

    activation = render_cfg.get("activation")
    if not isinstance(activation, Mapping):
        raise ValueError("cfg['render']['activation'] 缺失或格式非法")
    color_act = str(activation.get("color", "sigmoid"))
    opacity_act = str(activation.get("opacity", "sigmoid"))
    scale_clamp = activation.get("scale_clamp_m")
    if not isinstance(scale_clamp, (list, tuple)) or len(scale_clamp) != 2:
        raise ValueError("cfg['render']['activation']['scale_clamp_m'] 必须为长度 2 的序列")
    scale_min = float(scale_clamp[0])
    scale_max = float(scale_clamp[1])

    rotations = gaussians.rotations
    rot_norm = torch.linalg.norm(rotations, dim=-1, keepdim=True).clamp_min(1e-8)
    rotations = rotations / rot_norm

    scales = gaussians.scales
    if use_log and not use_softplus:
        scales = torch.exp(scales) + eps
    elif use_softplus:
        scales = torch_f.softplus(scales, beta=beta) + eps
    else:
        scales = scales.clamp_min(eps)
    scales = scales.clamp(min=scale_min, max=scale_max)

    colors = gaussians.colors
    if color_act == "sigmoid":
        colors = torch.sigmoid(colors)
    else:
        raise ValueError(f"不支持的 color activation: {color_act}")

    opacities = gaussians.opacities
    if opacity_act == "sigmoid":
        opacities = torch.sigmoid(opacities)
    else:
        raise ValueError(f"不支持的 opacity activation: {opacity_act}")

    return TorchGaussianSet(
        positions=gaussians.positions,
        rotations=rotations,
        scales=scales,
        opacities=opacities,
        colors=colors,
        timestamps=gaussians.timestamps,
    )


def render_gsplat(
    gaussians: TorchGaussianSet,
    camera: CameraPinholeTorch,
    *,
    near: float,
    far: float,
    render_mode: str,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    packed: bool = True,
    sparse_grad: bool = False,
) -> RenderOutputsTorch:
    """使用 gsplat 渲染 RGB/Depth。

    Args:
        gaussians: 渲染空间的高斯集合（scales>0，colors/opacities 已激活）。
        camera: pinhole 相机参数。
        near: 近裁剪平面（米）。
        far: 远裁剪平面（米）。
        render_mode: gsplat 渲染模式（如 "RGB+ED"）。
        radius_clip: 2D 半径阈值（像素），<=阈值的高斯跳过（加速大场景）。
        eps2d: antialiased 模式的 eps2d（像素）。
        packed: gsplat packed 模式开关。
        sparse_grad: gsplat sparse grad 开关。

    Returns:
        RenderOutputsTorch，包含 rgb/depth/alpha 与 meta。

    Raises:
        ModuleNotFoundError: gsplat 缺失。
        ValueError: render_mode 不支持或输出维度不符合预期。
    """
    _require_torch()
    gsplat = _require_gsplat()

    viewmats = camera.viewmat_world_to_cam[None, ...]
    ks = camera.k[None, ...]
    render, alpha, meta = gsplat.rasterization(
        means=gaussians.positions,
        quats=gaussians.rotations,
        scales=gaussians.scales,
        opacities=gaussians.opacities,
        colors=gaussians.colors,
        viewmats=viewmats,
        Ks=ks,
        width=int(camera.width),
        height=int(camera.height),
        near_plane=float(near),
        far_plane=float(far),
        radius_clip=float(radius_clip),
        eps2d=float(eps2d),
        render_mode=render_mode,
        packed=bool(packed),
        sparse_grad=bool(sparse_grad),
    )

    render0 = render[0]
    alpha0 = alpha[0]
    if alpha0.ndim == 3 and alpha0.shape[-1] == 1:
        alpha0 = alpha0[..., 0]

    rgb: Optional["torch.Tensor"] = None
    depth: Optional["torch.Tensor"] = None
    if render_mode == "RGB":
        if render0.shape[-1] != 3:
            raise ValueError("render_mode=RGB 时输出通道必须为 3")
        rgb = render0[..., 0:3]
    elif render_mode in ("RGB+D", "RGB+ED"):
        if render0.shape[-1] != 4:
            raise ValueError(f"render_mode={render_mode} 时输出通道必须为 4")
        rgb = render0[..., 0:3]
        depth = render0[..., 3]
    elif render_mode in ("D", "ED"):
        if render0.shape[-1] != 1:
            raise ValueError(f"render_mode={render_mode} 时输出通道必须为 1")
        depth = render0[..., 0]
    else:
        raise ValueError(f"不支持的 render_mode: {render_mode}")

    return RenderOutputsTorch(
        rgb=rgb,
        depth=depth,
        alpha=alpha0,
        meta=dict(meta),
    )
