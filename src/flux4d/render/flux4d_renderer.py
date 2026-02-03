"""Flux4D 的时间推进与 gsplat 渲染封装。

本模块提供阶段3/阶段4共用的“时间推进 + 渲染”基础能力：
- 线性运动推进（Flux4D-base）。
- 高斯参数激活（sigmoid/softplus + clamp）。
- 使用 gsplat 进行 RGB/Depth 渲染（pinhole 相机）。
- 渲染图像平面位移 `v_r`（用于阶段4门禁与阶段5动态重加权）。
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


@dataclass(frozen=True)
class RenderedVelocityOutputsTorch:
    """图像平面渲染位移（Torch）。

    Attributes:
        vr: 图像平面位移 (H, W, 2)，单位为 pixels（对应 t'->t'+Δt_norm）。
        vr_mag: 位移幅值 (H, W)，单位为 pixels。
        alpha: 可见性/权重 (H, W)，来自 gsplat 的累计 alpha。
    """

    vr: "torch.Tensor"
    vr_mag: "torch.Tensor"
    alpha: "torch.Tensor"


def project_world_to_pixel_torch(
    points_world: "torch.Tensor",
    camera: CameraPinholeTorch,
    *,
    eps: float = 1e-8,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """将 world 点投影为像素坐标与相机深度（pinhole）。

    Args:
        points_world: world 坐标系点 (N, 3)。
        camera: pinhole 相机参数（world->cam viewmat 与 K）。
        eps: 深度下界（避免除零）。

    Returns:
        (u, v, z_cam)，其中 u/v 为像素坐标 (N,)，z_cam 为相机深度 (N,)。

    Raises:
        ValueError: 输入形状非法。
    """
    _require_torch()
    if points_world.ndim != 2 or int(points_world.shape[1]) != 3:
        raise ValueError("points_world 形状必须为 (N, 3)")

    view = camera.viewmat_world_to_cam
    r_wc = view[:3, :3]
    t_wc = view[:3, 3]
    points_cam = points_world @ r_wc.transpose(0, 1) + t_wc[None, :]
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2].clamp_min(float(eps))

    k = camera.k
    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return u, v, z


def compute_rendered_velocity_per_gaussian_torch(
    points_world_t: "torch.Tensor",
    velocities_world: "torch.Tensor",
    camera: CameraPinholeTorch,
    *,
    delta_t_norm: float,
    clip_mag: float = 10.0,
    eps: float = 1e-6,
) -> "torch.Tensor":
    """计算每个高斯的图像平面位移 `v_r,i`（解析 Jacobian）。

    本实现采用“动态速度口径”：使用同一相机 pose（固定在 t'）来计算 `v_r`，
    使静态背景的 `|v_r|` 接近 0，便于阶段4门禁与阶段5动态重加权。

    Args:
        points_world_t: 高斯中心在 world 下的位置 (N, 3)，表示 p_world(t')。
        velocities_world: 高斯在 world 下的速度 (N, vdim)，至少包含前三维线速度，
            单位为 米 / 归一化时间（t∈[0,1]）。
        camera: pinhole 相机参数（world->cam viewmat 与 K），使用 t' 的相机 pose。
        delta_t_norm: 归一化时间步长（例如相邻帧 t_norm[k+1]-t_norm[k]），需为正数。
        clip_mag: 对 `||v_r,i||` 的裁剪上限（pixels）。补充材料 A.1 建议为 10。
        eps: 数值稳定项（避免除零/NaN）。

    Returns:
        每个高斯的图像平面位移 (N, 2)，单位为 pixels（对应 t'->t'+Δt_norm）。

    Raises:
        ValueError: 输入形状非法或参数不合法。
    """
    _require_torch()
    if points_world_t.ndim != 2 or int(points_world_t.shape[1]) != 3:
        raise ValueError("points_world_t 形状必须为 (N, 3)")
    if velocities_world.ndim != 2:
        raise ValueError("velocities_world 形状必须为 (N, vdim)")
    if int(velocities_world.shape[0]) != int(points_world_t.shape[0]):
        raise ValueError("velocities_world 与 points_world_t 的 N 不一致")
    if int(velocities_world.shape[1]) < 3:
        raise ValueError("velocities_world 的 vdim 必须至少为 3")
    if float(delta_t_norm) <= 0.0:
        raise ValueError("delta_t_norm 必须为正数")

    view = camera.viewmat_world_to_cam
    r_wc = view[:3, :3]
    t_wc = view[:3, 3]
    points_cam = points_world_t @ r_wc.transpose(0, 1) + t_wc[None, :]
    velocities_cam = velocities_world[:, 0:3] @ r_wc.transpose(0, 1)

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z_raw = points_cam[:, 2]
    valid_z = z_raw > float(eps)
    z = z_raw.clamp_min(float(eps))

    vx = velocities_cam[:, 0]
    vy = velocities_cam[:, 1]
    vz = velocities_cam[:, 2]

    k = camera.k
    fx = k[0, 0]
    fy = k[1, 1]

    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    du_dt = fx * (vx * z - x * vz) * inv_z2
    dv_dt = fy * (vy * z - y * vz) * inv_z2

    vr = torch.stack([du_dt, dv_dt], dim=-1) * float(delta_t_norm)
    if valid_z.numel() > 0:
        vr = torch.where(valid_z[:, None], vr, torch.zeros_like(vr))

    mag = torch.sqrt(torch.clamp(torch.sum(vr * vr, dim=-1), min=float(eps)))
    scale = torch.clamp(float(clip_mag) / mag, max=1.0)
    return vr * scale[:, None]


def render_attribute_map_gsplat(
    gaussians: TorchGaussianSet,
    camera: CameraPinholeTorch,
    attributes: "torch.Tensor",
    *,
    near: float,
    far: float,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """将每高斯的 2D 属性渲染为图像（使用与 RGB 相同的 splat/alpha 权重）。

    实现策略：将 `attributes(N,2)` 填充为 `colors(N,3)`，调用 gsplat 的 RGB 渲染，
    再取前两通道作为输出属性图。

    Args:
        gaussians: 渲染空间的高斯集合（scales>0，colors/opacities 已激活）。
        camera: pinhole 相机参数。
        attributes: 每高斯属性 (N, 2)。
        near: 近裁剪平面（米）。
        far: 远裁剪平面（米）。
        radius_clip: 2D 半径阈值（像素）。
        eps2d: antialiased 模式的 eps2d（像素）。

    Returns:
        (attr_map, alpha)，其中 attr_map 形状为 (H, W, 2)，alpha 形状为 (H, W)。

    Raises:
        ValueError: 输入形状不匹配。
    """
    _require_torch()
    if attributes.ndim != 2 or int(attributes.shape[1]) != 2:
        raise ValueError("attributes 形状必须为 (N, 2)")
    if int(attributes.shape[0]) != int(gaussians.positions.shape[0]):
        raise ValueError("attributes 与 gaussians 的 N 不一致")

    zeros = torch.zeros((int(attributes.shape[0]), 1), device=attributes.device, dtype=attributes.dtype)
    colors = torch.cat([attributes, zeros], dim=-1)
    gaussians_attr = TorchGaussianSet(
        positions=gaussians.positions,
        rotations=gaussians.rotations,
        scales=gaussians.scales,
        opacities=gaussians.opacities,
        colors=colors,
        timestamps=gaussians.timestamps,
    )
    out = render_gsplat(
        gaussians_attr,
        camera,
        near=float(near),
        far=float(far),
        render_mode="RGB",
        radius_clip=float(radius_clip),
        eps2d=float(eps2d),
    )
    if out.rgb is None:
        raise ValueError("render_mode=RGB 未输出 RGB，无法渲染属性图")
    return out.rgb[..., 0:2], out.alpha


def render_rendered_velocity_map(
    gaussians_world_t: TorchGaussianSet,
    velocities_world: "torch.Tensor",
    camera: CameraPinholeTorch,
    *,
    cfg: Mapping[str, object],
    delta_t_norm: float,
    t_target_norm: float,
) -> RenderedVelocityOutputsTorch:
    """渲染图像平面位移 `v_r`（对应 t'->t'+Δt_norm）。

    Args:
        gaussians_world_t: world 坐标系下推进到 t' 的高斯集合（仅 positions 应为 p_world(t')）。
        velocities_world: world 坐标系运动参数张量 (N, vdim)。当 `vdim>3` 时表示多项式速度参数，
            渲染 `v_r` 时会先在 `t_target_norm` 处求瞬时速度 `v(t')`。
        camera: pinhole 相机参数（使用 t' 的相机 pose）。
        cfg: `configs/flux4d.py` 中的 cfg 字典（读取 render.near/far 与 rendered_velocity.clip_mag_px）。
        delta_t_norm: 归一化时间步长（相邻帧差）。
        t_target_norm: 当前渲染时间 `t'`（归一化到 [0,1]）。

    Returns:
        RenderedVelocityOutputsTorch，包含 (v_r, |v_r|, alpha)。
    """
    _require_torch()
    render_cfg = cfg.get("render")
    if not isinstance(render_cfg, Mapping):
        raise ValueError("cfg['render'] 缺失或格式非法")
    near = float(render_cfg.get("near", 0.1))
    far = float(render_cfg.get("far", 200.0))

    rv_cfg = render_cfg.get("rendered_velocity", {})
    if rv_cfg is None:
        rv_cfg = {}
    if not isinstance(rv_cfg, Mapping):
        raise ValueError("cfg['render']['rendered_velocity'] 格式非法")
    clip_mag = float(rv_cfg.get("clip_mag_px", 10.0))
    eps = float(rv_cfg.get("eps", 1e-6))

    gaussians_act = activate_gaussians_for_render(gaussians_world_t, cfg)
    velocities_inst = evaluate_polynomial_velocity_torch(
        motion_params_world=velocities_world,
        t_target=torch.tensor(float(t_target_norm), device=velocities_world.device, dtype=velocities_world.dtype),
    )
    vr_gauss = compute_rendered_velocity_per_gaussian_torch(
        points_world_t=gaussians_world_t.positions,
        velocities_world=velocities_inst,
        camera=camera,
        delta_t_norm=float(delta_t_norm),
        clip_mag=clip_mag,
        eps=eps,
    )
    vr_map, alpha = render_attribute_map_gsplat(
        gaussians_act,
        camera,
        vr_gauss,
        near=near,
        far=far,
    )

    vr_mag = torch.sqrt(torch.clamp(torch.sum(vr_map * vr_map, dim=-1), min=float(eps)))
    scale = torch.clamp(float(clip_mag) / vr_mag, max=1.0)
    vr_map = vr_map * scale[..., None]
    vr_mag = vr_mag * scale
    return RenderedVelocityOutputsTorch(vr=vr_map, vr_mag=vr_mag, alpha=alpha)


def evaluate_polynomial_velocity_torch(
    motion_params_world: "torch.Tensor",
    t_target: "torch.Tensor",
) -> "torch.Tensor":
    """在目标时间 `t_target` 处计算多项式速度的瞬时速度 `v(t_target)`。

    补充材料 A.1 给出多项式运动模型的推进形式（积分），其对应的瞬时速度为：
    `v(t) = Σ_{j=0..ℓ} v_j * t^j`，其中 `vdim = 3(ℓ+1)`。

    Args:
        motion_params_world: 运动参数 (N, vdim)，vdim 必须为 3 的倍数。
        t_target: 目标时间标量或形状为 (N,) 的张量，范围建议为 [0,1]。

    Returns:
        瞬时线速度张量 (N, 3)，单位为 米 / 归一化时间。

    Raises:
        ValueError: 输入形状非法。
    """
    _require_torch()
    if motion_params_world.ndim != 2:
        raise ValueError("motion_params_world 形状必须为 (N, vdim)")
    vdim = int(motion_params_world.shape[1])
    if vdim < 3 or vdim % 3 != 0:
        raise ValueError("motion_params_world 的 vdim 必须为 3 的倍数且至少为 3")

    t_vec: "torch.Tensor"
    if t_target.ndim == 0:
        t_vec = t_target
    elif t_target.ndim == 1 and int(t_target.shape[0]) == int(motion_params_world.shape[0]):
        t_vec = t_target
    else:
        raise ValueError("t_target 必须为标量或形状为 (N,) 的张量")

    chunks = motion_params_world.view(motion_params_world.shape[0], vdim // 3, 3)
    v_inst = torch.zeros((motion_params_world.shape[0], 3), device=motion_params_world.device, dtype=motion_params_world.dtype)
    for j in range(vdim // 3):
        power = t_vec**j
        v_inst = v_inst + chunks[:, j, :] * (power if power.ndim == 0 else power[:, None])
    return v_inst


def apply_polynomial_motion(
    gaussians_world: TorchGaussianSet,
    motion_params_world: "torch.Tensor",
    t_target: "torch.Tensor",
) -> TorchGaussianSet:
    """对高斯中心应用多项式运动推进（补充材料 A.1 Eq.(2)）。

    位置推进形式：
    `p_i(t') = p_i(t_i) + Σ_{j=0..ℓ} v_{i,j} * (t'^{j+1} - t_i^{j+1})/(j+1)`，
    其中 `vdim = 3(ℓ+1)`，`v_{i,j}∈R^3`。

    Args:
        gaussians_world: 高斯集合（位置为 p(t_i)，时间戳为 t_i∈[0,1]）。
        motion_params_world: 多项式速度参数 (N, vdim)，vdim 必须为 3 的倍数。
        t_target: 目标时间标量或形状为 (N,) 的张量，单位为归一化时间（[0,1]）。

    Returns:
        推进到 t_target 的高斯集合（仅 positions 更新，其余字段保持不变）。

    Raises:
        ValueError: 输入形状不匹配。
    """
    _require_torch()
    if motion_params_world.ndim != 2:
        raise ValueError("motion_params_world 形状必须为 (N, vdim)")
    if motion_params_world.shape[0] != gaussians_world.positions.shape[0]:
        raise ValueError("motion_params_world 与 gaussians_world 的 N 不一致")
    vdim = int(motion_params_world.shape[1])
    if vdim < 3 or vdim % 3 != 0:
        raise ValueError("motion_params_world 的 vdim 必须为 3 的倍数且至少为 3")

    t_i = gaussians_world.timestamps
    if t_target.ndim == 0:
        t_vec = t_target
    elif t_target.ndim == 1 and t_target.shape[0] == t_i.shape[0]:
        t_vec = t_target
    else:
        raise ValueError("t_target 必须为标量或形状为 (N,) 的张量")

    terms = vdim // 3
    chunks = motion_params_world.view(motion_params_world.shape[0], terms, 3)
    delta_p = torch.zeros_like(gaussians_world.positions)
    for j in range(terms):
        exponent = j + 1
        coeff = 1.0 / float(j + 1)
        delta_scalar = (t_vec**exponent - t_i**exponent) * coeff
        delta_p = delta_p + chunks[:, j, :] * delta_scalar[:, None]

    positions = gaussians_world.positions + delta_p
    return TorchGaussianSet(
        positions=positions,
        rotations=gaussians_world.rotations,
        scales=gaussians_world.scales,
        opacities=gaussians_world.opacities,
        colors=gaussians_world.colors,
        timestamps=gaussians_world.timestamps,
    )

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
