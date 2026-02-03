#!/usr/bin/env python3
"""阶段4：Rendered Velocity（v_r）可视化门禁脚本。

该脚本用于验证阶段4“时间推进 + 图像平面位移渲染”的闭环是否正确：
- 固定相机 pose（门禁 A）：静态背景 `|v_r|≈0`，动态区域方向与相邻渲染帧差分一致；
- 逐帧相机 pose（门禁 B）：用于对齐检查，不要求背景为 0。

输出默认写入 `assets/vis/stage4_flow_sanity/`（被 git 忽略）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from flux4d.models.flux4d_model import TorchGaussianSet


def _require_torch() -> ModuleType:
    """确保 torch 可用并返回 torch 模块。"""
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("缺少 torch：请在训练环境（如 gaussianstorm）中运行该脚本") from exc
    return torch


def _require_pillow() -> object:
    """确保 Pillow 可用并返回 Image 模块。"""
    try:
        from PIL import Image  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("缺少 Pillow：请在训练环境中安装 pillow") from exc
    return Image


def _load_cfg(path: str) -> Dict[str, object]:
    """加载配置文件中的 `cfg` 字典。

    Args:
        path: 配置文件路径（如 configs/flux4d.py）。

    Returns:
        配置字典。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: 配置文件缺少 `cfg` 或格式非法。
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    spec = importlib.util.spec_from_file_location("flux4d_cfg", str(config_path))
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载配置模块: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, "cfg", None)
    if not isinstance(cfg, dict):
        raise ValueError("配置文件中未找到 dict 类型的 cfg 变量")
    return cfg


def _parse_int_ranges(text: str) -> List[int]:
    """解析形如 '0,1,2,5-10' 的整数列表字符串。"""
    out: List[int] = []
    for token in [t.strip() for t in text.split(",") if t.strip()]:
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"非法范围: {token}")
            out.extend(list(range(start, end + 1)))
        else:
            out.append(int(token))
    return sorted(set(out))


def _as_int_list(values: object, name: str) -> List[int]:
    """将对象解析为 int 列表。"""
    if not isinstance(values, list):
        raise ValueError(f"{name} 必须为 list")
    out: List[int] = []
    for item in values:
        if not isinstance(item, int):
            raise ValueError(f"{name} 必须为 int 列表")
        out.append(int(item))
    return out


def _save_rgb(path: Path, rgb: "torch.Tensor") -> None:
    """保存 RGB 图像（0~1）到磁盘。"""
    torch = _require_torch()
    Image = _require_pillow()
    array = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    Image.fromarray(array).save(path)


def _save_rgb_numpy(path: Path, rgb: np.ndarray) -> None:
    """保存 RGB 图像（NumPy，HWC）到磁盘。

    Args:
        path: 输出路径。
        rgb: RGB 图像数组，支持 uint8 或 float（0~1 / 0~255）。
    """
    Image = _require_pillow()
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb 必须为形状 (H, W, 3) 的数组")
    if rgb.dtype == np.uint8:
        array = rgb
    else:
        rgb_float = rgb.astype(np.float32)
        if float(np.nanmax(rgb_float)) <= 1.0:
            rgb_float = np.clip(rgb_float, 0.0, 1.0) * 255.0
        array = np.clip(rgb_float, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def _save_depth(path: Path, depth: "torch.Tensor", valid: Optional["torch.Tensor"] = None) -> None:
    """保存深度可视化（灰度）。"""
    torch = _require_torch()
    Image = _require_pillow()
    depth_cpu = depth.detach().cpu()
    if valid is not None:
        valid_cpu = valid.detach().cpu()
        depth_valid = depth_cpu[valid_cpu]
    else:
        depth_valid = depth_cpu[torch.isfinite(depth_cpu)]
    if depth_valid.numel() == 0:
        gray = torch.zeros_like(depth_cpu, dtype=torch.uint8)
    else:
        d_min = float(depth_valid.min())
        d_max = float(depth_valid.max())
        if d_max <= d_min:
            gray = torch.zeros_like(depth_cpu, dtype=torch.uint8)
        else:
            norm = (depth_cpu - d_min) / (d_max - d_min)
            gray_float = (1.0 - norm.clamp(0.0, 1.0)) * 255.0
            gray = gray_float.to(torch.uint8)
            if valid is not None:
                gray[~valid_cpu] = 0
    Image.fromarray(gray.numpy()).save(path)


def _hsv_to_rgb_numpy(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """将 HSV 转为 RGB（向量化，h∈[0,1)）。"""
    h = np.mod(h, 1.0)
    i = np.floor(h * 6.0).astype(np.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = np.mod(i, 6)
    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)

    mask = i_mod == 0
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
    mask = i_mod == 1
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
    mask = i_mod == 2
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
    mask = i_mod == 3
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
    mask = i_mod == 4
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
    mask = i_mod == 5
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    return np.stack([r, g, b], axis=-1)


def _save_vr_mag(path: Path, vr_mag: "torch.Tensor", alpha: "torch.Tensor", clip_mag: float) -> None:
    """保存 |v_r| 灰度图（0..clip_mag 映射到 0..255）。"""
    torch = _require_torch()
    Image = _require_pillow()
    valid = alpha > 1e-3
    gray = (vr_mag / float(clip_mag)).clamp(0.0, 1.0) * 255.0
    gray = gray.to(torch.uint8)
    gray = torch.where(valid, gray, torch.zeros_like(gray))
    Image.fromarray(gray.cpu().numpy()).save(path)


def _save_vr_hsv(path: Path, vr: "torch.Tensor", alpha: "torch.Tensor", clip_mag: float) -> None:
    """保存 v_r HSV 可视化：Hue=方向，Value=幅值。"""
    vr_np = vr.detach().cpu().numpy()
    alpha_np = alpha.detach().cpu().numpy()
    mag = np.linalg.norm(vr_np, axis=-1)
    angle = np.arctan2(vr_np[..., 1], vr_np[..., 0])  # [-pi, pi]
    hue = (angle / (2.0 * np.pi)) + 0.5  # [0, 1)
    val = np.clip(mag / float(clip_mag), 0.0, 1.0)
    sat = (alpha_np > 1e-3).astype(np.float32)
    rgb = _hsv_to_rgb_numpy(hue.astype(np.float32), sat, val.astype(np.float32))
    rgb_u8 = (rgb * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image = _require_pillow()
    Image.fromarray(rgb_u8).save(path)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Stage4 rendered velocity (v_r) visualization.")
    parser.add_argument("--config", default="configs/flux4d.py", help="Path to configs/flux4d.py")
    parser.add_argument(
        "--index-path",
        default="data/metadata/pandaset_tiny_clips.pkl",
        help="Clip index PKL path.",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Override PandaSet root (default uses cfg['data']['data_root']).",
    )
    parser.add_argument("--clip-index", type=int, default=0, help="Clip index in PKL.")
    parser.add_argument("--camera", default="front_camera", help="Camera name to render.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0.")
    parser.add_argument(
        "--ckpt",
        default="assets/vis/stage3_overfit_run2/ckpt_last.pt",
        help="Stage3 checkpoint path (ckpt_last.pt or ckpt_step_*.pt).",
    )
    parser.add_argument(
        "--out-dir",
        default="assets/vis/stage4_flow_sanity/clip_000",
        help="Output directory (under assets/vis/ by default).",
    )
    parser.add_argument(
        "--mode",
        choices=["fixed_pose", "per_frame_pose", "both"],
        default="both",
        help="Render mode: fixed camera pose, per-frame pose, or both.",
    )
    parser.add_argument("--frame-ref", type=int, default=0, help="Reference frame index for fixed_pose mode.")
    parser.add_argument(
        "--render-frames",
        default="0-15",
        help="Frames to render, e.g. '0-15' or '0,2,4,6,8,10'.",
    )
    parser.add_argument(
        "--num-sky-points",
        type=int,
        default=20000,
        help="Sky points for G_init (debug-friendly).",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=200000,
        help="Max gaussians (debug-friendly).",
    )
    return parser


def _compute_delta_t_norm(t_norm: Sequence[float], frame_index: int) -> float:
    """计算 frame_index 对应的 Δt_norm（优先使用下一帧间隔）。"""
    if frame_index < 0 or frame_index >= len(t_norm):
        raise ValueError("frame_index 超出范围")
    if len(t_norm) <= 1:
        return 1.0
    if frame_index + 1 < len(t_norm):
        return float(t_norm[frame_index + 1] - t_norm[frame_index])
    if frame_index - 1 >= 0:
        return float(t_norm[frame_index] - t_norm[frame_index - 1])
    return 1.0


def _write_summary(path: Path, payload: Mapping[str, object]) -> None:
    """写入 summary.json。"""
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2))


def _run_mode(
    *,
    mode: str,
    out_root: Path,
    render_frames: Sequence[int],
    frame_ref: int,
    t_norm: Sequence[float],
    views: Mapping[int, object],
    gaussians_world_out: "TorchGaussianSet",
    velocities_world: "torch.Tensor",
    cfg: Mapping[str, object],
    device: "torch.device",
    dtype: "torch.dtype",
) -> None:
    """渲染一个模式（fixed_pose 或 per_frame_pose）。"""
    torch = _require_torch()
    from flux4d.render.flux4d_renderer import activate_gaussians_for_render  # noqa: E402
    from flux4d.render.flux4d_renderer import apply_linear_motion  # noqa: E402
    from flux4d.render.flux4d_renderer import apply_polynomial_motion  # noqa: E402
    from flux4d.render.flux4d_renderer import build_pinhole_camera_from_pandaset  # noqa: E402
    from flux4d.render.flux4d_renderer import render_gsplat  # noqa: E402
    from flux4d.render.flux4d_renderer import render_rendered_velocity_map  # noqa: E402

    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, Mapping):
        raise ValueError("cfg['model'] 缺失或格式非法")
    head_cfg = model_cfg.get("head")
    if not isinstance(head_cfg, Mapping):
        raise ValueError("cfg['model']['head'] 缺失或格式非法")
    motion_cfg = head_cfg.get("motion")
    if not isinstance(motion_cfg, Mapping):
        raise ValueError("cfg['model']['head']['motion'] 缺失或格式非法")
    poly_degree_l = int(motion_cfg.get("poly_degree_l", 0))

    render_cfg = cfg.get("render")
    if not isinstance(render_cfg, Mapping):
        raise ValueError("cfg['render'] 缺失或格式非法")
    near = float(render_cfg.get("near", 0.1))
    far = float(render_cfg.get("far", 200.0))
    render_mode = str(render_cfg.get("mode", "RGB+ED"))
    rv_cfg = render_cfg.get("rendered_velocity", {})
    if rv_cfg is None:
        rv_cfg = {}
    if not isinstance(rv_cfg, Mapping):
        raise ValueError("cfg['render']['rendered_velocity'] 格式非法")
    clip_mag = float(rv_cfg.get("clip_mag_px", 10.0))

    out_dir = out_root / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_rgb: Optional["torch.Tensor"] = None
    for frame_index in render_frames:
        if frame_index not in views:
            raise ValueError(f"缺少 frame {frame_index} 的相机视图缓存")
        view = views[frame_ref] if mode == "fixed_pose" else views[frame_index]

        image = view.image
        height, width = image.shape[:2]
        camera = build_pinhole_camera_from_pandaset(
            intrinsics=view.intrinsics,
            pose_cam=view.pose,
            image_size=(width, height),
            device=device,
            dtype=dtype,
        )

        t_target = torch.tensor(float(t_norm[frame_index]), device=device, dtype=dtype)
        if poly_degree_l <= 0:
            gaussians_t = apply_linear_motion(gaussians_world_out, velocities_world, t_target)
        else:
            gaussians_t = apply_polynomial_motion(gaussians_world_out, velocities_world, t_target)
        gaussians_act = activate_gaussians_for_render(gaussians_t, cfg)

        render_out = render_gsplat(
            gaussians_act,
            camera,
            near=near,
            far=far,
            render_mode=render_mode,
        )
        if render_out.rgb is None:
            raise ValueError("render_mode 未输出 RGB，无法可视化")
        pred_rgb = render_out.rgb
        pred_depth = render_out.depth

        delta_t_norm = _compute_delta_t_norm(t_norm, frame_index)
        vr_out = render_rendered_velocity_map(
            gaussians_world_t=gaussians_t,
            velocities_world=velocities_world,
            camera=camera,
            cfg=cfg,
            delta_t_norm=delta_t_norm,
            t_target_norm=float(t_norm[frame_index]),
        )

        frame_dir = out_dir / f"frame_{frame_index:03d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        _save_rgb(frame_dir / "pred_rgb.png", pred_rgb.detach())
        if mode == "per_frame_pose":
            _save_rgb_numpy(frame_dir / "gt_rgb.png", image)
        if pred_depth is not None:
            _save_depth(frame_dir / "pred_depth.png", pred_depth.detach())
        _save_vr_mag(frame_dir / "vr_mag.png", vr_out.vr_mag.detach(), vr_out.alpha.detach(), clip_mag=clip_mag)
        _save_vr_hsv(frame_dir / "vr_hsv.png", vr_out.vr.detach(), vr_out.alpha.detach(), clip_mag=clip_mag)

        if mode == "fixed_pose" and prev_rgb is not None:
            diff = torch.mean(torch.abs(pred_rgb - prev_rgb), dim=-1)
            _save_vr_mag(frame_dir / "rgb_diff_prev.png", diff.detach(), vr_out.alpha.detach(), clip_mag=1.0)
        prev_rgb = pred_rgb.detach()


def main() -> int:
    """脚本入口。"""
    args = build_arg_parser().parse_args()
    torch = _require_torch()

    from flux4d.datasets.pandaset_clips import load_clip_index  # noqa: E402
    from flux4d.engine.checkpoint import load_ckpt  # noqa: E402
    from flux4d.lift.lift_lidar import build_camera_views  # noqa: E402
    from flux4d.lift.lift_lidar import build_initial_gaussians_for_clip_aggregated  # noqa: E402
    from flux4d.lift.lift_lidar import get_lidar_pose  # noqa: E402
    from flux4d.lift.lift_lidar import get_lidar_timestamp  # noqa: E402
    from flux4d.lift.lift_lidar import normalize_timestamps_to_unit_range  # noqa: E402
    from flux4d.models.flux4d_model import build_frame_transform_from_ego0_pose  # noqa: E402
    from flux4d.models.flux4d_model import build_flux4d_base_model_frames  # noqa: E402
    from flux4d.models.flux4d_model import torch_gaussian_set_from_numpy  # noqa: E402

    cfg = _load_cfg(args.config)
    data_cfg = cfg.get("data")
    init_cfg = cfg.get("init")
    coord_cfg = cfg.get("coord")
    train_cfg = cfg.get("train")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("cfg['data'] 缺失或格式非法")
    if not isinstance(init_cfg, Mapping):
        raise ValueError("cfg['init'] 缺失或格式非法")
    if not isinstance(coord_cfg, Mapping):
        raise ValueError("cfg['coord'] 缺失或格式非法")
    if not isinstance(train_cfg, Mapping):
        raise ValueError("cfg['train'] 缺失或格式非法")

    data_root = str(args.data_root) if str(args.data_root) else str(data_cfg.get("data_root", ""))
    if not data_root:
        raise ValueError("无法解析 data_root，请使用 --data-root 或在 cfg 中设置 data.data_root")

    payload = load_clip_index(str(args.index_path))
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("索引缺少 clips 列表或为空")
    if args.clip_index < 0 or args.clip_index >= len(clips):
        raise ValueError("clip_index 超出范围")
    clip = clips[int(args.clip_index)]
    if not isinstance(clip, dict):
        raise ValueError("clip 结构非法")

    input_frame_indices = _as_int_list(clip.get("input_frame_indices"), "input_frame_indices")
    clip_len_frames = int(clip.get("clip_len_frames", 0))
    if clip_len_frames <= 0:
        raise ValueError("clip_len_frames 非法")

    abs_lidar_ts = [get_lidar_timestamp(clip, i) for i in range(clip_len_frames)]
    t_norm = normalize_timestamps_to_unit_range(abs_lidar_ts)

    render_frames = _parse_int_ranges(str(args.render_frames))
    if any(i < 0 or i >= clip_len_frames for i in render_frames):
        raise ValueError("render_frames 含越界帧号")
    frame_ref = int(args.frame_ref)
    if frame_ref < 0 or frame_ref >= clip_len_frames:
        raise ValueError("frame_ref 超出范围")

    device = torch.device(str(args.device))
    dtype = torch.float32

    camera_name = str(args.camera)
    view_indices = sorted(set(render_frames + [frame_ref]))
    views: Dict[int, object] = {}
    for frame_index in view_indices:
        views_list = build_camera_views(
            clip=clip,
            frame_index=frame_index,
            data_root=data_root,
            view_names=[camera_name],
        )
        if not views_list:
            raise ValueError(f"frame {frame_index} 缺少相机视角: {camera_name}")
        views[frame_index] = views_list[0]

    downsample_cfg = init_cfg.get("downsample")
    voxel_size_m = 0.2
    if isinstance(downsample_cfg, Mapping):
        voxel_size_m = float(downsample_cfg.get("voxel_size_m", voxel_size_m))
    lidar_sensor_id = int(data_cfg.get("lidar_sensor_id", -1))
    random_seed = int(train_cfg.get("seed", 0))

    g_init = build_initial_gaussians_for_clip_aggregated(
        clip=clip,
        data_root=data_root,
        frame_indices=input_frame_indices,
        timestamp_frame_indices=list(range(clip_len_frames)),
        view_names=[camera_name],
        voxel_size=voxel_size_m,
        knn_k=int(init_cfg.get("scale_knn_k", 3)),
        default_color=(0.5, 0.5, 0.5),
        opacity_init=float(init_cfg.get("opacity_init", 0.5)),
        random_seed=random_seed,
        num_sky_points=int(args.num_sky_points),
        max_gaussians=int(args.max_gaussians),
        lidar_sensor_id=lidar_sensor_id,
    )

    gaussians_world = torch_gaussian_set_from_numpy(
        positions=torch.from_numpy(g_init.positions).to(device=device, dtype=dtype),
        rotations=torch.from_numpy(g_init.rotations).to(device=device, dtype=dtype),
        scales=torch.from_numpy(g_init.scales).to(device=device, dtype=dtype),
        opacities=torch.from_numpy(g_init.opacities).to(device=device, dtype=dtype),
        colors=torch.from_numpy(g_init.colors).to(device=device, dtype=dtype),
        timestamps=torch.from_numpy(g_init.timestamps).to(device=device, dtype=dtype),
    )

    ego0_frame_index = int(coord_cfg.get("ego0_frame_index", 0))
    ego0_pose = get_lidar_pose(clip, ego0_frame_index)
    frame = build_frame_transform_from_ego0_pose(ego0_pose, device=device, dtype=dtype)

    model = build_flux4d_base_model_frames(cfg).to(device=device)
    state = load_ckpt(str(args.ckpt), map_location=device)
    model_state = state.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint 缺少/非法字段: model_state")
    model.load_state_dict(model_state)
    model.eval()

    with torch.no_grad():
        out = model(gaussians_world, frame)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    _write_summary(
        out_root / "summary.json",
        {
            "mode": str(args.mode),
            "index_path": str(args.index_path),
            "clip_index": int(args.clip_index),
            "clip_id": clip.get("clip_id"),
            "camera": camera_name,
            "ckpt": str(args.ckpt),
            "frame_ref": frame_ref,
            "render_frames": list(render_frames),
            "num_gaussians": int(out.gaussians_world.positions.shape[0]),
        },
    )

    if str(args.mode) in ("fixed_pose", "both"):
        _run_mode(
            mode="fixed_pose",
            out_root=out_root,
            render_frames=render_frames,
            frame_ref=frame_ref,
            t_norm=t_norm,
            views=views,
            gaussians_world_out=out.gaussians_world,
            velocities_world=out.velocities_world,
            cfg=cfg,
            device=device,
            dtype=dtype,
        )
    if str(args.mode) in ("per_frame_pose", "both"):
        _run_mode(
            mode="per_frame_pose",
            out_root=out_root,
            render_frames=render_frames,
            frame_ref=frame_ref,
            t_norm=t_norm,
            views=views,
            gaussians_world_out=out.gaussians_world,
            velocities_world=out.velocities_world,
            cfg=cfg,
            device=device,
            dtype=dtype,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
