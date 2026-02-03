"""阶段3：tiny overfit 训练闭环（最小实现）。

该模块提供一个“单 clip overfit”训练入口，用于验证：
Lift(G_init) -> Flux4D-base(ΔG,V) -> gsplat 渲染 -> loss -> 反向传播

Note:
    该实现以“门禁/闭环”为第一目标，未实现完整的多场景 DataLoader、分布式训练与评测。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from flux4d.engine.checkpoint import load_ckpt, save_ckpt
from flux4d.datasets.pandaset_clips import load_clip_index
from flux4d.lift.lift_lidar import (
    build_camera_views,
    build_initial_gaussians_for_clip_aggregated,
    get_lidar_pose,
    get_lidar_timestamp,
    load_lidar_frame,
    normalize_timestamps_to_unit_range,
    project_points_to_camera,
    transform_lidar_to_camera,
)
from flux4d.losses.flux4d_losses import compute_flux4d_base_losses
from flux4d.models.flux4d_model import (
    TorchGaussianSet,
    build_frame_transform_from_ego0_pose,
    build_flux4d_base_model_frames,
    build_flux4d_refine_model,
    apply_delta_g_to_gaussians,
    torch_gaussian_set_from_numpy,
)
from flux4d.render.flux4d_renderer import (
    CameraPinholeTorch,
    activate_gaussians_for_render,
    apply_linear_motion,
    apply_polynomial_motion,
    build_pinhole_camera_from_pandaset,
    render_gsplat,
    render_rendered_velocity_map,
)


def _require_torch() -> None:
    """确保 torch 可用。"""
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("缺少 torch：请在训练环境（如 gaussianstorm）中安装 PyTorch")


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


def _render_sparse_depth_map(
    uv: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """将稀疏点投影渲染为稀疏深度图（未命中像素为 inf）。"""
    width, height = image_size
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    valid_idx = np.where(mask)[0]
    if valid_idx.size == 0:
        return depth_map
    u = np.round(uv[valid_idx, 0]).astype(np.int64)
    v = np.round(uv[valid_idx, 1]).astype(np.int64)
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    for idx, x, y in zip(valid_idx, u, v):
        current = depth_map[y, x]
        if depth[idx] < current:
            depth_map[y, x] = depth[idx]
    return depth_map


def _save_rgb(path: Path, rgb: "torch.Tensor") -> None:
    """保存 RGB 图像（0~1）到磁盘。"""
    _require_torch()
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少 Pillow，请先安装依赖") from exc
    array = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    Image.fromarray(array).save(path)


def _save_depth(path: Path, depth: "torch.Tensor", valid: Optional["torch.Tensor"] = None) -> None:
    """保存深度可视化（灰度）。"""
    _require_torch()
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少 Pillow，请先安装依赖") from exc
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


@dataclass(frozen=True)
class Stage3OverfitArgs:
    """阶段3 overfit 的关键参数。"""

    index_path: str
    clip_index: int
    data_root: str
    camera_name: str
    device: str
    iters: int
    grad_accum_steps: int
    log_every: int
    save_every: int
    output_dir: str
    num_sky_points: int
    max_gaussians: Optional[int]
    use_projected_lidar_depth: bool
    resume_from: str
    save_ckpt_every: int


def train_stage3_overfit(cfg: Mapping[str, object], args: Stage3OverfitArgs) -> None:
    """运行阶段3 tiny overfit 训练闭环。

    Args:
        cfg: `configs/flux4d.py` 中的 cfg 字典。
        args: 训练参数。
    """
    _require_torch()
    device = torch.device(args.device)
    if device.type != "cuda":  # pragma: no cover
        raise ValueError("阶段3 overfit 需要 CUDA（gsplat/spconv）")
    if not torch.cuda.is_available():  # pragma: no cover
        raise ValueError("未检测到 CUDA 设备，请在 GPU 环境中运行")

    data_cfg = cfg.get("data")
    init_cfg = cfg.get("init")
    train_cfg = cfg.get("train")
    loss_cfg = cfg.get("loss")
    render_cfg = cfg.get("render")
    coord_cfg = cfg.get("coord")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("cfg['data'] 缺失或格式非法")
    if not isinstance(init_cfg, Mapping):
        raise ValueError("cfg['init'] 缺失或格式非法")
    if not isinstance(train_cfg, Mapping):
        raise ValueError("cfg['train'] 缺失或格式非法")
    if not isinstance(loss_cfg, Mapping):
        raise ValueError("cfg['loss'] 缺失或格式非法")
    if not isinstance(render_cfg, Mapping):
        raise ValueError("cfg['render'] 缺失或格式非法")
    if not isinstance(coord_cfg, Mapping):
        raise ValueError("cfg['coord'] 缺失或格式非法")

    random.seed(int(train_cfg.get("seed", 0)))
    np.random.seed(int(train_cfg.get("seed", 0)))
    torch.manual_seed(int(train_cfg.get("seed", 0)))
    torch.cuda.manual_seed_all(int(train_cfg.get("seed", 0)))

    payload = load_clip_index(args.index_path)
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("索引缺少 clips 列表或为空")
    if args.clip_index < 0 or args.clip_index >= len(clips):
        raise ValueError("clip_index 超出范围")
    clip = clips[args.clip_index]
    if not isinstance(clip, dict):
        raise ValueError("clip 结构非法")

    input_frame_indices = _as_int_list(clip.get("input_frame_indices"), "input_frame_indices")
    target_frame_indices = _as_int_list(clip.get("target_frame_indices"), "target_frame_indices")
    if not target_frame_indices:
        raise ValueError("target_frame_indices 为空，无法做重建监督")

    clip_len_frames = int(clip.get("clip_len_frames", 0))
    if clip_len_frames <= 0:
        raise ValueError("clip_len_frames 非法")
    abs_lidar_ts = [get_lidar_timestamp(clip, i) for i in range(clip_len_frames)]
    norm_lidar_ts = normalize_timestamps_to_unit_range(abs_lidar_ts)

    lidar_sensor_id = int(data_cfg.get("lidar_sensor_id", -1))

    g_init = build_initial_gaussians_for_clip_aggregated(
        clip=clip,
        data_root=args.data_root,
        frame_indices=input_frame_indices,
        timestamp_frame_indices=list(range(clip_len_frames)),
        view_names=[args.camera_name],
        voxel_size=float(init_cfg.get("downsample", {}).get("voxel_size_m", 0.2))
        if isinstance(init_cfg.get("downsample"), Mapping)
        else 0.2,
        knn_k=int(init_cfg.get("scale_knn_k", 3)),
        default_color=(0.5, 0.5, 0.5),
        opacity_init=float(init_cfg.get("opacity_init", 0.5)),
        random_seed=int(train_cfg.get("seed", 0)),
        num_sky_points=int(args.num_sky_points),
        max_gaussians=args.max_gaussians,
        lidar_sensor_id=lidar_sensor_id,
    )

    dtype = torch.float32
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

    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, Mapping):
        raise ValueError("cfg['model'] 缺失或格式非法")
    refine_cfg = model_cfg.get("iterative_refine")
    iterative_refine_enabled = False
    refine_iters = 1
    if isinstance(refine_cfg, Mapping):
        iterative_refine_enabled = bool(refine_cfg.get("enabled", False))
        refine_iters = int(refine_cfg.get("num_iters", 3))
    refine_iters = max(1, int(refine_iters))

    head_cfg = model_cfg.get("head")
    if not isinstance(head_cfg, Mapping):
        raise ValueError("cfg['model']['head'] 缺失或格式非法")
    motion_cfg = head_cfg.get("motion")
    if not isinstance(motion_cfg, Mapping):
        raise ValueError("cfg['model']['head']['motion'] 缺失或格式非法")
    poly_degree_l = int(motion_cfg.get("poly_degree_l", 0))

    base_model = build_flux4d_base_model_frames(cfg)
    if iterative_refine_enabled:
        refine_model = build_flux4d_refine_model(cfg)

        class _Flux4DStage5Wrapper(torch.nn.Module):
            """将 base/refine 合并为一个可保存 checkpoint 的 nn.Module。"""

            def __init__(self) -> None:
                super().__init__()
                self.base = base_model
                self.refine = refine_model

            def forward(self, gaussians: TorchGaussianSet, frame_transform: object) -> object:
                return self.base(gaussians, frame_transform)

        model = _Flux4DStage5Wrapper().to(device=device)
    else:
        model = base_model.to(device=device)
    model.train()

    optimizer_cfg = train_cfg.get("optimizer")
    if not isinstance(optimizer_cfg, Mapping):
        raise ValueError("cfg['train']['optimizer'] 缺失或格式非法")
    if str(optimizer_cfg.get("type", "Adam")) != "Adam":
        raise ValueError("当前最小实现仅支持 Adam")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-3)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )

    render_mode = str(render_cfg.get("mode", "RGB+ED"))
    near = float(render_cfg.get("near", 0.1))
    far = float(render_cfg.get("far", 200.0))
    rv_cfg = render_cfg.get("rendered_velocity", {})
    if rv_cfg is None:
        rv_cfg = {}
    if not isinstance(rv_cfg, Mapping):
        raise ValueError("cfg['render']['rendered_velocity'] 格式非法")
    clip_mag_px = float(rv_cfg.get("clip_mag_px", 10.0))
    delta_t_mode = str(rv_cfg.get("delta_t_mode", "next_frame"))

    lambda_rgb = float(loss_cfg.get("lambda_rgb", 0.8))
    lambda_ssim = float(loss_cfg.get("lambda_ssim", 0.2))
    lambda_depth = float(loss_cfg.get("lambda_depth", 0.0))
    lambda_vel = float(loss_cfg.get("lambda_vel", 5e-3))
    ssim_window = int(loss_cfg.get("ssim_window", 11))

    depth_cfg = loss_cfg.get("depth")
    use_depth = bool(args.use_projected_lidar_depth) and lambda_depth > 0.0
    if isinstance(depth_cfg, Mapping):
        use_depth = use_depth and bool(depth_cfg.get("use_projected_lidar_depth", True))

    vrw_cfg = loss_cfg.get("velocity_reweighting")
    use_velocity_reweighting = False
    alpha_threshold = 1e-3
    if isinstance(vrw_cfg, Mapping):
        use_velocity_reweighting = bool(vrw_cfg.get("enabled", False))
        alpha_threshold = float(vrw_cfg.get("alpha_threshold", 1e-3))

    # 预加载 target 帧相机视图（图像+内外参），避免训练中频繁 IO。
    target_views: Dict[int, object] = {}
    for frame_index in target_frame_indices:
        views = build_camera_views(
            clip=clip,
            frame_index=frame_index,
            data_root=args.data_root,
            view_names=[args.camera_name],
        )
        if not views:
            raise ValueError(f"frame {frame_index} 缺少相机视角: {args.camera_name}")
        target_views[frame_index] = views[0]

    # 预计算 projected LiDAR depth（稀疏），仅用于深度监督。
    target_depth_maps: Dict[int, Tuple["torch.Tensor", "torch.Tensor"]] = {}
    if use_depth:
        for frame_index in target_frame_indices:
            view = target_views[frame_index]
            image = view.image
            height, width = image.shape[:2]
            lidar_paths = clip.get("lidar_paths")
            if not isinstance(lidar_paths, list):
                raise ValueError("clip.lidar_paths 格式非法")
            lidar_pose = get_lidar_pose(clip, frame_index)
            lidar_points_world, _ = load_lidar_frame(
                str(Path(args.data_root) / str(lidar_paths[frame_index])),
                sensor_id=lidar_sensor_id,
            )
            points_cam = transform_lidar_to_camera(
                lidar_points_world,
                lidar_pose=lidar_pose,
                camera_pose=view.pose,
                points_in_world=True,
            )
            uv, depth, mask = project_points_to_camera(
                points_cam,
                view.intrinsics,
                image_size=(width, height),
            )
            depth_map = _render_sparse_depth_map(uv, depth, mask, image_size=(width, height))
            depth_valid = np.isfinite(depth_map)
            target_depth_maps[frame_index] = (
                torch.from_numpy(depth_map).to(device=device, dtype=dtype),
                torch.from_numpy(depth_valid).to(device=device),
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "config": {
                    "render_mode": render_mode,
                    "use_depth": use_depth,
                    "iterative_refine": {"enabled": iterative_refine_enabled, "num_iters": refine_iters},
                    "velocity_reweighting": {"enabled": use_velocity_reweighting, "alpha_threshold": alpha_threshold},
                    "motion": {"poly_degree_l": poly_degree_l},
                },
                "index_path": args.index_path,
                "clip_index": args.clip_index,
                "clip_id": clip.get("clip_id"),
                "camera_name": args.camera_name,
                "num_gaussians": int(gaussians_world.positions.shape[0]),
                "resume_from": args.resume_from,
                "save_ckpt_every": int(args.save_ckpt_every),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    start_step = 1
    resume_from = str(args.resume_from)
    if resume_from:
        state = load_ckpt(resume_from, map_location=device)
        resume_step = int(state.get("step", 0))
        if resume_step < 0:
            raise ValueError("checkpoint.step 非法（需为非负整数）")
        if resume_step >= int(args.iters):
            raise ValueError("checkpoint.step 已达到/超过训练总步数，无法继续训练")

        model_state = state.get("model_state")
        if not isinstance(model_state, Mapping):
            raise ValueError("checkpoint 缺少/非法字段: model_state")
        optimizer_state = state.get("optimizer_state")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("checkpoint 缺少/非法字段: optimizer_state")

        if iterative_refine_enabled:
            incompatible = model.load_state_dict(model_state, strict=False)
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            if missing:
                print(f"[resume] missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                print(f"[resume] unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        else:
            model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

        rng_state = state.get("rng_state")
        if isinstance(rng_state, Mapping):
            python_random_state = rng_state.get("python_random")
            if python_random_state is not None:
                random.setstate(python_random_state)  # type: ignore[arg-type]
            numpy_state = rng_state.get("numpy")
            if numpy_state is not None:
                np.random.set_state(numpy_state)  # type: ignore[arg-type]
            torch_state = rng_state.get("torch")
            if torch_state is not None:
                torch.random.set_rng_state(torch_state)  # type: ignore[arg-type]
            cuda_state = rng_state.get("torch_cuda")
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]

        start_step = resume_step + 1
        print(f"[resume] from={resume_from} step={resume_step}")

    def _save_last_symlink(checkpoint_path: Path) -> None:
        """创建/更新 output_dir/ckpt_last.pt -> checkpoint_path.name。"""
        last_path = output_dir / "ckpt_last.pt"
        try:
            if last_path.exists() or last_path.is_symlink():
                last_path.unlink()
            last_path.symlink_to(checkpoint_path.name)
        except OSError:
            import shutil

            shutil.copy2(checkpoint_path, last_path)

    def _save_checkpoint(step: int) -> None:
        """保存模型与优化器 checkpoint。"""
        state: Dict[str, object] = {
            "step": int(step),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": {
                "python_random": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
        }
        ckpt_path = output_dir / f"ckpt_step_{step:06d}.pt"
        save_ckpt(state, ckpt_path)
        _save_last_symlink(ckpt_path)

    def _compute_delta_t_norm(frame_index: int) -> float:
        """计算 frame_index 对应的 Δt_norm（默认使用下一帧间隔）。"""
        if delta_t_mode != "next_frame":
            raise ValueError(f"不支持的 delta_t_mode: {delta_t_mode}")
        if frame_index < 0 or frame_index >= len(norm_lidar_ts):
            raise ValueError("frame_index 超出范围")
        if len(norm_lidar_ts) <= 1:
            return 1.0
        if frame_index + 1 < len(norm_lidar_ts):
            return float(norm_lidar_ts[frame_index + 1] - norm_lidar_ts[frame_index])
        if frame_index - 1 >= 0:
            return float(norm_lidar_ts[frame_index] - norm_lidar_ts[frame_index - 1])
        return 1.0

    def _camera_in_ego0(camera_world: CameraPinholeTorch) -> CameraPinholeTorch:
        """将 world->cam 的 view 矩阵左乘 ego0->world，得到 ego0->cam。"""
        view_ego0_to_cam = camera_world.viewmat_world_to_cam @ frame.t_world_ego0
        return CameraPinholeTorch(
            viewmat_world_to_cam=view_ego0_to_cam,
            k=camera_world.k,
            width=camera_world.width,
            height=camera_world.height,
        )

    def _retain_gaussian_param_grads(gaussians: TorchGaussianSet) -> None:
        """对 G 的字段调用 retain_grad，便于在 backward 后读取 ∇G。"""
        for tensor in (
            gaussians.positions,
            gaussians.rotations,
            gaussians.scales,
            gaussians.opacities,
            gaussians.colors,
        ):
            if tensor.requires_grad:
                tensor.retain_grad()

    def _extract_gaussian_param_grads(gaussians: TorchGaussianSet) -> "torch.Tensor":
        """拼接得到 ∇G（N,14），顺序与 G 的拼接一致。"""
        grads = []
        for name, tensor in (
            ("positions", gaussians.positions),
            ("rotations", gaussians.rotations),
            ("scales", gaussians.scales),
            ("opacities", gaussians.opacities),
            ("colors", gaussians.colors),
        ):
            if tensor.grad is None:
                raise ValueError(f"未获取到 {name}.grad，请确认已调用 retain_grad 且该字段参与了 loss 计算")
            if name == "opacities":
                grads.append(tensor.grad[:, None])
            else:
                grads.append(tensor.grad)
        return torch.cat(grads, dim=-1)

    def _detach_gaussians(gaussians: TorchGaussianSet) -> TorchGaussianSet:
        """对高斯字段做 detach（用于阶段5迭代 refinement 的下一轮输入）。"""
        return TorchGaussianSet(
            positions=gaussians.positions.detach(),
            rotations=gaussians.rotations.detach(),
            scales=gaussians.scales.detach(),
            opacities=gaussians.opacities.detach(),
            colors=gaussians.colors.detach(),
            timestamps=gaussians.timestamps.detach(),
        )

    for step in range(int(start_step), int(args.iters) + 1):
        accum_steps = max(1, int(args.grad_accum_steps))
        optimizer.zero_grad(set_to_none=True)

        total_sum = torch.zeros((), device=device, dtype=dtype)
        rgb_sum = torch.zeros((), device=device, dtype=dtype)
        ssim_value_sum = torch.zeros((), device=device, dtype=dtype)
        depth_sum = torch.zeros((), device=device, dtype=dtype)
        vel_sum = torch.zeros((), device=device, dtype=dtype)
        psnr_sum = torch.zeros((), device=device, dtype=dtype)

        last_pred_rgb = None
        last_target_rgb = None
        last_pred_depth = None
        last_target_depth = None
        last_target_valid = None

        for _ in range(accum_steps):
            frame_index = random.choice(target_frame_indices)
            view = target_views[frame_index]
            t_target = torch.tensor(float(norm_lidar_ts[frame_index]), device=device, dtype=dtype)

            image = view.image
            height, width = image.shape[:2]
            camera_world = build_pinhole_camera_from_pandaset(
                intrinsics=view.intrinsics,
                pose_cam=view.pose,
                image_size=(width, height),
                device=device,
                dtype=dtype,
            )
            target_rgb = torch.from_numpy(image).to(device=device, dtype=dtype)

            target_depth = None
            target_valid = None
            if use_depth:
                target_depth, target_valid = target_depth_maps[frame_index]

            if iterative_refine_enabled:
                camera = _camera_in_ego0(camera_world)
                delta_t_norm = _compute_delta_t_norm(frame_index) if use_velocity_reweighting else 1.0

                gaussians_iter: Optional[TorchGaussianSet] = None
                velocities_iter: Optional["torch.Tensor"] = None
                gaussians_grad: Optional["torch.Tensor"] = None

                for refine_step in range(refine_iters):
                    if refine_step == 0:
                        base_out = model(gaussians_world, frame)
                        gaussians_iter = base_out.gaussians_ego0
                        velocities_iter = base_out.velocities_ego0
                    else:
                        if gaussians_iter is None or velocities_iter is None or gaussians_grad is None:
                            raise ValueError("迭代 refinement 状态非法：缺少 gaussians/velocities/gaussians_grad")
                        gaussians_in = _detach_gaussians(gaussians_iter)
                        refine_out = model.refine(gaussians_in, gaussians_grad)
                        gaussians_iter = apply_delta_g_to_gaussians(gaussians_in, refine_out.delta_g)
                        velocities_iter = velocities_iter.detach()

                    if gaussians_iter is None or velocities_iter is None:
                        raise ValueError("gaussians_iter/velocities_iter 为空，训练过程异常")
                    _retain_gaussian_param_grads(gaussians_iter)

                    if poly_degree_l <= 0:
                        gaussians_t = apply_linear_motion(gaussians_iter, velocities_iter, t_target)
                    else:
                        gaussians_t = apply_polynomial_motion(gaussians_iter, velocities_iter, t_target)
                    gaussians_act = activate_gaussians_for_render(gaussians_t, cfg)

                    render_out = render_gsplat(
                        gaussians_act,
                        camera,
                        near=near,
                        far=far,
                        render_mode=render_mode,
                    )
                    if render_out.rgb is None:
                        raise ValueError("render_mode 未输出 RGB，无法进行光度监督")
                    pred_rgb = render_out.rgb
                    pred_depth = render_out.depth

                    rgb_weight_map = None
                    if use_velocity_reweighting:
                        vr_out = render_rendered_velocity_map(
                            gaussians_world_t=gaussians_t,
                            velocities_world=velocities_iter,
                            camera=camera,
                            cfg=cfg,
                            delta_t_norm=float(delta_t_norm),
                            t_target_norm=float(t_target.item()),
                        )
                        valid_mask = render_out.alpha > float(alpha_threshold)
                        rgb_weight_map = (1.0 + vr_out.vr_mag.detach()).clamp(0.0, 1.0 + float(clip_mag_px))
                        rgb_weight_map = rgb_weight_map * valid_mask.to(dtype=rgb_weight_map.dtype)

                    losses = compute_flux4d_base_losses(
                        pred_rgb=pred_rgb,
                        target_rgb=target_rgb,
                        rgb_weight_map=rgb_weight_map,
                        pred_depth=pred_depth,
                        target_depth=target_depth,
                        target_depth_valid=target_valid,
                        velocities_world=velocities_iter,
                        lambda_rgb=lambda_rgb,
                        lambda_ssim=lambda_ssim,
                        lambda_depth=lambda_depth,
                        lambda_vel=lambda_vel,
                        ssim_window=ssim_window,
                    )
                    (losses.total / float(accum_steps * refine_iters)).backward()

                    gaussians_grad = _extract_gaussian_param_grads(gaussians_iter).detach()

                    with torch.no_grad():
                        mse = torch.mean((pred_rgb - target_rgb) ** 2).clamp_min(1e-12)
                        psnr = -10.0 * torch.log10(mse)
                        ssim_value = 1.0 - losses.ssim
                        total_sum += losses.total.detach() / float(refine_iters)
                        rgb_sum += losses.rgb_l1.detach() / float(refine_iters)
                        ssim_value_sum += ssim_value.detach() / float(refine_iters)
                        depth_sum += losses.depth_l1.detach() / float(refine_iters)
                        vel_sum += losses.vel.detach() / float(refine_iters)
                        psnr_sum += psnr.detach() / float(refine_iters)

                    last_pred_rgb = pred_rgb
                    last_target_rgb = target_rgb
                    last_pred_depth = pred_depth
                    last_target_depth = target_depth
                    last_target_valid = target_valid
            else:
                out = model(gaussians_world, frame)
                velocities_world = out.velocities_world
                if poly_degree_l <= 0:
                    gaussians_t = apply_linear_motion(out.gaussians_world, velocities_world, t_target)
                else:
                    gaussians_t = apply_polynomial_motion(out.gaussians_world, velocities_world, t_target)
                gaussians_act = activate_gaussians_for_render(gaussians_t, cfg)

                render_out = render_gsplat(
                    gaussians_act,
                    camera_world,
                    near=near,
                    far=far,
                    render_mode=render_mode,
                )
                if render_out.rgb is None:
                    raise ValueError("render_mode 未输出 RGB，无法进行光度监督")
                pred_rgb = render_out.rgb
                pred_depth = render_out.depth

                rgb_weight_map = None
                if use_velocity_reweighting:
                    delta_t_norm = _compute_delta_t_norm(frame_index)
                    vr_out = render_rendered_velocity_map(
                        gaussians_world_t=gaussians_t,
                        velocities_world=velocities_world,
                        camera=camera_world,
                        cfg=cfg,
                        delta_t_norm=float(delta_t_norm),
                        t_target_norm=float(t_target.item()),
                    )
                    valid_mask = render_out.alpha > float(alpha_threshold)
                    rgb_weight_map = (1.0 + vr_out.vr_mag.detach()).clamp(0.0, 1.0 + float(clip_mag_px))
                    rgb_weight_map = rgb_weight_map * valid_mask.to(dtype=rgb_weight_map.dtype)

                losses = compute_flux4d_base_losses(
                    pred_rgb=pred_rgb,
                    target_rgb=target_rgb,
                    rgb_weight_map=rgb_weight_map,
                    pred_depth=pred_depth,
                    target_depth=target_depth,
                    target_depth_valid=target_valid,
                    velocities_world=velocities_world,
                    lambda_rgb=lambda_rgb,
                    lambda_ssim=lambda_ssim,
                    lambda_depth=lambda_depth,
                    lambda_vel=lambda_vel,
                    ssim_window=ssim_window,
                )
                (losses.total / float(accum_steps)).backward()

                with torch.no_grad():
                    mse = torch.mean((pred_rgb - target_rgb) ** 2).clamp_min(1e-12)
                    psnr = -10.0 * torch.log10(mse)
                    ssim_value = 1.0 - losses.ssim
                    total_sum += losses.total.detach()
                    rgb_sum += losses.rgb_l1.detach()
                    ssim_value_sum += ssim_value.detach()
                    depth_sum += losses.depth_l1.detach()
                    vel_sum += losses.vel.detach()
                    psnr_sum += psnr.detach()

                last_pred_rgb = pred_rgb
                last_target_rgb = target_rgb
                last_pred_depth = pred_depth
                last_target_depth = target_depth
                last_target_valid = target_valid

        optimizer.step()

        if step % int(args.log_every) == 0 or step == 1:
            denom = float(max(1, int(args.grad_accum_steps)))
            print(
                f"[{step:06d}] total={float(total_sum) / denom:.6f} "
                f"rgb={float(rgb_sum) / denom:.6f} ssim={float(ssim_value_sum) / denom:.4f} "
                f"depth={float(depth_sum) / denom:.6f} vel={float(vel_sum) / denom:.6f} "
                f"psnr={float(psnr_sum) / denom:.2f}"
            )

        if step % int(args.save_every) == 0 or step == int(args.iters):
            step_dir = output_dir / f"step_{step:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            if last_pred_rgb is None or last_target_rgb is None:
                raise ValueError("last_pred_rgb/last_target_rgb 为空，训练过程异常")
            _save_rgb(step_dir / "pred_rgb.png", last_pred_rgb.detach())
            _save_rgb(step_dir / "gt_rgb.png", last_target_rgb.detach())
            if last_pred_depth is not None:
                _save_depth(step_dir / "pred_depth.png", last_pred_depth.detach())
            if use_depth and last_target_depth is not None and last_target_valid is not None:
                _save_depth(step_dir / "gt_depth.png", last_target_depth.detach(), valid=last_target_valid)

        save_ckpt_every = int(args.save_ckpt_every)
        if step == int(args.iters) or (save_ckpt_every > 0 and step % save_ckpt_every == 0):
            _save_checkpoint(step)
