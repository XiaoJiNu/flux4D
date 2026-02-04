"""阶段6：全量训练与评测（PandaSet）。

该模块在阶段3最小闭环基础上，补齐：
- 多 clip / 多场景训练采样；
- Lift 缓存（避免每步重复 IO + Lift 计算）；
- 验证集评测：NVS（PSNR/SSIM/Depth RMSE，full + dynamic-only）与 Scene Flow 指标；
- 评测产物落盘（metrics JSON + 可视化图）。

Note:
    该实现面向“可复现与口径正确”，优先保证指标定义、帧集合、mask 口径与 GT 构造无歧义。
    训练效率（DataLoader、多进程、分布式）可在后续迭代中优化。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from flux4d.datasets.pandaset_clips import load_clip_index
from flux4d.datasets.pandaset_cuboids import (
    PandaSetCuboid,
    assign_points_to_cuboids_world,
    compute_cuboid_dynamic_flags_by_frame,
    load_pandaset_cuboids_frame,
)
from flux4d.engine.checkpoint import load_ckpt, save_ckpt
from flux4d.lift.lift_lidar import (
    build_camera_views,
    build_initial_gaussians_for_clip,
    get_lidar_pose,
    get_lidar_timestamp,
    load_lidar_frame,
    normalize_timestamps_to_unit_range,
    project_points_to_camera,
    transform_lidar_to_camera,
)
from flux4d.losses.flux4d_losses import build_dynamic_weight_map_from_flow, compute_flux4d_base_losses
from flux4d.metrics.image_metrics import compute_depth_rmse_torch, compute_psnr_torch, compute_ssim_value_torch
from flux4d.metrics.scene_flow import (
    SceneFlowMetrics,
    build_default_label_to_bucket_map,
    build_scene_flow_gt_from_cuboids,
    compute_scene_flow_metrics,
    merge_bucket_stats,
)
from flux4d.models.flux4d_model import (
    Flux4DRefineModel,
    TorchGaussianSet,
    apply_delta_g_to_gaussians,
    build_frame_transform_from_ego0_pose,
    build_flux4d_base_model_frames,
    build_flux4d_refine_model,
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


def _save_rgb(path: Path, rgb: "torch.Tensor") -> None:
    """保存 RGB 图像（0~1）到磁盘。"""
    _require_torch()
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少 Pillow，请先安装依赖") from exc
    array = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    Image.fromarray(array).save(path)


def _save_gray_u8(path: Path, gray_u8: np.ndarray) -> None:
    """保存单通道 uint8 图。"""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少 Pillow，请先安装依赖") from exc
    if gray_u8.ndim != 2 or gray_u8.dtype != np.uint8:
        raise ValueError("gray_u8 必须为 (H, W) uint8")
    Image.fromarray(gray_u8).save(path)


def _dilate_mask(mask_hw: np.ndarray, radius_px: int) -> np.ndarray:
    """对二值 mask 做膨胀（MaxFilter），避免依赖 scipy/cv2。"""
    if radius_px <= 0:
        return mask_hw
    try:
        from PIL import Image, ImageFilter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少 Pillow，请先安装依赖") from exc
    if mask_hw.ndim != 2:
        raise ValueError("mask_hw 形状必须为 (H, W)")
    size = int(radius_px) * 2 + 1
    img = Image.fromarray(mask_hw.astype(np.uint8) * 255)
    out = img.filter(ImageFilter.MaxFilter(size=size))
    return (np.asarray(out) > 0).astype(bool)


def _render_sparse_depth_map(
    uv: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """将稀疏点投影渲染为稀疏深度图（未命中像素为 inf）。

    Args:
        uv: 像素坐标 (N, 2)。
        depth: 深度 (N,)。
        mask: 有效点掩码 (N,)。
        image_size: (width, height)。

    Returns:
        (depth_map, valid_mask)，其中：
        - depth_map: (H, W) float32，未命中像素为 inf。
        - valid_mask: (H, W) bool。
    """
    width, height = image_size
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    valid_idx = np.where(mask)[0]
    if valid_idx.size == 0:
        return depth_map, np.zeros((height, width), dtype=bool)
    u = np.round(uv[valid_idx, 0]).astype(np.int64)
    v = np.round(uv[valid_idx, 1]).astype(np.int64)
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    for src, x, y in zip(valid_idx, u, v):
        current = depth_map[y, x]
        if depth[src] < current:
            depth_map[y, x] = float(depth[src])
    valid_mask = np.isfinite(depth_map)
    return depth_map, valid_mask


@dataclass(frozen=True)
class LiftCacheItem:
    """单个 clip 的 Lift 缓存项（含来源帧 meta）。"""

    positions: np.ndarray  # (N, 3) float32
    rotations: np.ndarray  # (N, 4) float32 wxyz
    scales: np.ndarray  # (N, 3) float32
    opacities: np.ndarray  # (N,) float32 (logit)
    colors: np.ndarray  # (N, 3) float32 (logit)
    timestamps: np.ndarray  # (N,) float32 in [0,1]
    source_frame_indices: np.ndarray  # (N,) int16, -1 for sky/extra


class LiftCache:
    """Lift 缓存（磁盘 npz + 内存 LRU）。"""

    def __init__(self, cache_dir: Path, *, max_items_in_memory: int = 2) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_items = max(0, int(max_items_in_memory))
        self._lru_keys: List[str] = []
        self._memory: Dict[str, LiftCacheItem] = {}

    def _touch_key(self, key: str) -> None:
        if key in self._lru_keys:
            self._lru_keys.remove(key)
        self._lru_keys.append(key)
        while self._max_items > 0 and len(self._lru_keys) > self._max_items:
            evict = self._lru_keys.pop(0)
            self._memory.pop(evict, None)

    def _path_for_key(self, key: str) -> Path:
        safe = key.replace("/", "_")
        return self._cache_dir / f"{safe}.npz"

    def get_or_build(
        self,
        *,
        key: str,
        clip: Mapping[str, object],
        data_root: str,
        input_frame_indices: Sequence[int],
        timestamp_frame_indices: Sequence[int],
        camera_names: Sequence[str],
        voxel_size_m: float,
        knn_k: int,
        opacity_init: float,
        random_seed: int,
        num_sky_points: int,
        max_gaussians: Optional[int],
        lidar_sensor_id: int,
    ) -> LiftCacheItem:
        """读取或构建 Lift 缓存项。"""
        if key in self._memory:
            self._touch_key(key)
            return self._memory[key]

        path = self._path_for_key(key)
        # 兼容旧版本 bug：临时文件名不以 ".npz" 结尾时，numpy 会自动追加扩展名，
        # 导致写出 "xxx.npz.tmp.npz" 并在 rename 前崩溃留下遗留文件。
        legacy_tmp = path.with_name(path.name + ".tmp.npz")
        if not path.exists() and legacy_tmp.exists():
            try:
                legacy_tmp.replace(path)
            except OSError:
                # 若 rename 失败则忽略，后续会重新计算并覆盖。
                pass
        if path.exists():
            item = self._load_npz(path)
            if self._max_items > 0:
                self._memory[key] = item
                self._touch_key(key)
            return item

        item = self._build_item(
            clip=clip,
            data_root=data_root,
            input_frame_indices=input_frame_indices,
            timestamp_frame_indices=timestamp_frame_indices,
            camera_names=camera_names,
            voxel_size_m=voxel_size_m,
            knn_k=knn_k,
            opacity_init=opacity_init,
            random_seed=random_seed,
            num_sky_points=num_sky_points,
            max_gaussians=max_gaussians,
            lidar_sensor_id=lidar_sensor_id,
        )
        self._save_npz(path, item)
        if self._max_items > 0:
            self._memory[key] = item
            self._touch_key(key)
        return item

    def _load_npz(self, path: Path) -> LiftCacheItem:
        payload = np.load(str(path))
        return LiftCacheItem(
            positions=payload["positions"].astype(np.float32, copy=False),
            rotations=payload["rotations"].astype(np.float32, copy=False),
            scales=payload["scales"].astype(np.float32, copy=False),
            opacities=payload["opacities"].astype(np.float32, copy=False),
            colors=payload["colors"].astype(np.float32, copy=False),
            timestamps=payload["timestamps"].astype(np.float32, copy=False),
            source_frame_indices=payload["source_frame_indices"].astype(np.int16, copy=False),
        )

    def _save_npz(self, path: Path, item: LiftCacheItem) -> None:
        # 注意：numpy.savez_compressed 会在文件名不以 ".npz" 结尾时自动追加扩展名，
        # 若使用 "xxx.npz.tmp" 会实际写出 "xxx.npz.tmp.npz" 并导致后续 replace 找不到文件。
        tmp = path.with_name(f"{path.stem}.tmp.npz")
        np.savez_compressed(
            str(tmp),
            positions=item.positions,
            rotations=item.rotations,
            scales=item.scales,
            opacities=item.opacities,
            colors=item.colors,
            timestamps=item.timestamps,
            source_frame_indices=item.source_frame_indices,
        )
        tmp.replace(path)

    def _build_item(
        self,
        *,
        clip: Mapping[str, object],
        data_root: str,
        input_frame_indices: Sequence[int],
        timestamp_frame_indices: Sequence[int],
        camera_names: Sequence[str],
        voxel_size_m: float,
        knn_k: int,
        opacity_init: float,
        random_seed: int,
        num_sky_points: int,
        max_gaussians: Optional[int],
        lidar_sensor_id: int,
    ) -> LiftCacheItem:
        per_frame = build_initial_gaussians_for_clip(
            clip=dict(clip),
            data_root=data_root,
            frame_indices=list(input_frame_indices),
            timestamp_frame_indices=list(timestamp_frame_indices),
            view_names=list(camera_names),
            voxel_size=float(voxel_size_m),
            knn_k=int(knn_k),
            opacity_init=float(opacity_init),
            random_seed=int(random_seed),
            lidar_sensor_id=int(lidar_sensor_id),
        )
        if not per_frame:
            raise ValueError("Lift 输出为空：请检查 clip/frame_indices 与数据完整性")

        source: List[np.ndarray] = []
        for frame_index, gauss in zip(input_frame_indices, per_frame):
            source.append(np.full((gauss.positions.shape[0],), int(frame_index), dtype=np.int16))
        merged_positions = np.concatenate([g.positions for g in per_frame], axis=0)
        merged_scales = np.concatenate([g.scales for g in per_frame], axis=0)
        merged_rot = np.concatenate([g.rotations for g in per_frame], axis=0)
        merged_colors = np.concatenate([g.colors for g in per_frame], axis=0)
        merged_op = np.concatenate([g.opacities for g in per_frame], axis=0)
        merged_ts = np.concatenate([g.timestamps for g in per_frame], axis=0)
        merged_source = np.concatenate(source, axis=0)

        # Sky points: reuse lift.add_sky_points logic without引入额外字段，直接采样并拼接。
        from flux4d.lift.lift_lidar import add_sky_points  # local import to avoid circular

        from flux4d.lift.lift_lidar import GaussianSet as _GaussianSet

        base_set = _GaussianSet(
            positions=merged_positions,
            scales=merged_scales,
            rotations=merged_rot,
            colors=merged_colors,
            opacities=merged_op,
            timestamps=merged_ts,
            velocities=np.zeros((merged_positions.shape[0], 3), dtype=np.float32),
        )
        rng = np.random.default_rng(int(random_seed) + 1)
        with_sky = add_sky_points(base_set, num_sky_points=int(num_sky_points), rng=rng)
        added = int(with_sky.positions.shape[0]) - int(base_set.positions.shape[0])
        if added > 0:
            merged_source = np.concatenate(
                [merged_source, np.full((added,), -1, dtype=np.int16)], axis=0
            )
        else:
            # add_sky_points 可能因 AABB 异常而不生效
            merged_source = merged_source.astype(np.int16, copy=False)

        # Downsample（需要同时下采样 source_frame_indices）
        n_total = int(with_sky.positions.shape[0])
        if max_gaussians is not None and n_total > int(max_gaussians):
            indices = rng.choice(n_total, size=int(max_gaussians), replace=False)
            indices = indices.astype(np.int64, copy=False)
        else:
            indices = None

        def _slice(arr: np.ndarray) -> np.ndarray:
            return arr if indices is None else arr[indices]

        return LiftCacheItem(
            positions=_slice(with_sky.positions).astype(np.float32, copy=False),
            rotations=_slice(with_sky.rotations).astype(np.float32, copy=False),
            scales=_slice(with_sky.scales).astype(np.float32, copy=False),
            opacities=_slice(with_sky.opacities).astype(np.float32, copy=False),
            colors=_slice(with_sky.colors).astype(np.float32, copy=False),
            timestamps=_slice(with_sky.timestamps).astype(np.float32, copy=False),
            source_frame_indices=_slice(merged_source).astype(np.int16, copy=False),
        )


def _load_camera_view_single(
    clip: Mapping[str, object],
    *,
    frame_index: int,
    data_root: str,
    camera_name: str,
) -> object:
    views = build_camera_views(
        clip=dict(clip),
        frame_index=int(frame_index),
        data_root=data_root,
        view_names=[camera_name],
    )
    if not views:
        raise ValueError(f"frame {frame_index} 缺少相机视角: {camera_name}")
    return views[0]


def _compute_delta_t_norm(t_norm: np.ndarray, frame_index: int) -> float:
    """计算 frame_index 对应的 Δt_norm（默认使用下一帧间隔）。"""
    if t_norm.size <= 1:
        return 1.0
    idx = int(frame_index)
    if idx + 1 < t_norm.shape[0]:
        return float(t_norm[idx + 1] - t_norm[idx])
    if idx - 1 >= 0:
        return float(t_norm[idx] - t_norm[idx - 1])
    return 1.0


def _integrate_polynomial_motion_delta(
    motion_params_world: "torch.Tensor",
    t0: float,
    t1: float,
) -> "torch.Tensor":
    """计算多项式运动的位移 delta_p（用于 Scene Flow 预测）。

    Args:
        motion_params_world: (N, vdim)。
        t0: 起始归一化时间。
        t1: 结束归一化时间。

    Returns:
        (N, 3) 的位移张量。
    """
    _require_torch()
    if motion_params_world.ndim != 2:
        raise ValueError("motion_params_world 形状必须为 (N, vdim)")
    vdim = int(motion_params_world.shape[1])
    if vdim < 3 or vdim % 3 != 0:
        raise ValueError("motion_params_world 的 vdim 必须为 3 的倍数且至少为 3")
    terms = vdim // 3
    chunks = motion_params_world.view(motion_params_world.shape[0], terms, 3)
    delta = torch.zeros((motion_params_world.shape[0], 3), device=motion_params_world.device, dtype=motion_params_world.dtype)
    for j in range(terms):
        exponent = j + 1
        coeff = 1.0 / float(j + 1)
        delta_scalar = (float(t1) ** exponent - float(t0) ** exponent) * coeff
        delta = delta + chunks[:, j, :] * float(delta_scalar)
    return delta


@dataclass(frozen=True)
class Stage6TrainArgs:
    """阶段6全量训练参数。"""

    index_path: str
    data_root: str
    camera_name: str
    device: str
    iters: int
    grad_accum_steps: int
    log_every: int
    save_ckpt_every: int
    output_dir: str
    cache_dir: str
    cache_max_items: int
    resume_from: str
    resume_optimizer: bool
    num_sky_points: int
    max_gaussians: Optional[int]
    max_train_clips: Optional[int]


def train_stage6_full(cfg: Mapping[str, object], args: Stage6TrainArgs) -> None:
    """阶段6：跨场景全量训练（最小实现）。

    Args:
        cfg: `configs/flux4d.py` 中的 cfg 字典。
        args: 训练参数。
    """
    _require_torch()
    device = torch.device(args.device)
    if device.type != "cuda":  # pragma: no cover
        raise ValueError("阶段6训练需要 CUDA（gsplat/spconv）")
    if not torch.cuda.is_available():  # pragma: no cover
        raise ValueError("未检测到 CUDA 设备，请在 GPU 环境中运行")

    data_cfg = cfg.get("data")
    init_cfg = cfg.get("init")
    train_cfg = cfg.get("train")
    loss_cfg = cfg.get("loss")
    render_cfg = cfg.get("render")
    coord_cfg = cfg.get("coord")
    model_cfg = cfg.get("model")
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
    if not isinstance(model_cfg, Mapping):
        raise ValueError("cfg['model'] 缺失或格式非法")

    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    payload = load_clip_index(args.index_path)
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("索引缺少 clips 列表或为空")

    train_clips: List[Mapping[str, object]] = []
    for clip in clips:
        if not isinstance(clip, Mapping):
            continue
        if str(clip.get("split")) != "train":
            continue
        setting = str(clip.get("setting", ""))
        if setting not in ("train_interpolation", "train_future"):
            continue
        train_clips.append(clip)

    if args.max_train_clips is not None:
        train_clips = train_clips[: int(args.max_train_clips)]
    if not train_clips:
        raise ValueError("未找到训练 clips：请检查 index_path/preset/split/setting")

    ego0_frame_index = int(coord_cfg.get("ego0_frame_index", 0))
    lidar_sensor_id = int(data_cfg.get("lidar_sensor_id", -1))

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
    refine_model: Optional[Flux4DRefineModel] = None
    if iterative_refine_enabled:
        refine_model = build_flux4d_refine_model(cfg)

        class _Stage6Wrapper(torch.nn.Module):
            """将 base/refine 合并为一个可保存 checkpoint 的 nn.Module。"""

            def __init__(self) -> None:
                super().__init__()
                self.base = base_model
                self.refine = refine_model

            def forward(self, gaussians: TorchGaussianSet, frame_transform: object) -> object:
                return self.base(gaussians, frame_transform)

        model = _Stage6Wrapper().to(device=device)
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

    lambda_rgb = float(loss_cfg.get("lambda_rgb", 0.8))
    lambda_ssim = float(loss_cfg.get("lambda_ssim", 0.2))
    lambda_depth = float(loss_cfg.get("lambda_depth", 0.0))
    lambda_vel = float(loss_cfg.get("lambda_vel", 5e-3))
    ssim_window = int(loss_cfg.get("ssim_window", 11))

    vrw_cfg = loss_cfg.get("velocity_reweighting")
    use_velocity_reweighting = False
    alpha_threshold = 1e-3
    blur_sigma = 0.0
    blur_window = 0
    if isinstance(vrw_cfg, Mapping):
        use_velocity_reweighting = bool(vrw_cfg.get("enabled", False))
        alpha_threshold = float(vrw_cfg.get("alpha_threshold", 1e-3))
        blur_sigma = float(vrw_cfg.get("blur_sigma", 0.0))
        blur_window = int(vrw_cfg.get("blur_window", 0))

    depth_cfg = loss_cfg.get("depth")
    use_depth = lambda_depth > 0.0
    if isinstance(depth_cfg, Mapping):
        use_depth = use_depth and bool(depth_cfg.get("use_projected_lidar_depth", True))

    downsample_cfg = init_cfg.get("downsample", {})
    voxel_size_m = float(downsample_cfg.get("voxel_size_m", 0.2)) if isinstance(downsample_cfg, Mapping) else 0.2
    knn_k = int(init_cfg.get("scale_knn_k", 3))
    opacity_init = float(init_cfg.get("opacity_init", 0.5))

    cache = LiftCache(Path(args.cache_dir), max_items_in_memory=int(args.cache_max_items))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = (output_dir / "train.log").open("a", encoding="utf-8")

    def _log(message: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        line = f"[{ts}] {message}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    start_step = 1
    if args.resume_from:
        ckpt = load_ckpt(args.resume_from, map_location="cpu")
        model_state = ckpt.get("model_state")
        if not isinstance(model_state, Mapping):
            raise ValueError("checkpoint 缺少 model_state")
        model.load_state_dict(model_state, strict=False)
        if args.resume_optimizer:
            optim_state = ckpt.get("optimizer_state")
            if isinstance(optim_state, Mapping):
                optimizer.load_state_dict(optim_state)
        step_val = ckpt.get("step")
        if isinstance(step_val, int):
            start_step = int(step_val) + 1
        _log(f"[resume] ckpt={args.resume_from} start_step={start_step} resume_optim={args.resume_optimizer}")

    def _save_last_symlink(checkpoint_path: Path) -> None:
        last_path = checkpoint_path.parent / "ckpt_last.pt"
        try:
            if last_path.exists() or last_path.is_symlink():
                last_path.unlink()
            last_path.symlink_to(checkpoint_path.name)
        except OSError:
            import shutil

            shutil.copy2(checkpoint_path, last_path)

    def _save_checkpoint(step: int) -> None:
        state: Dict[str, object] = {
            "step": int(step),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        ckpt_path = output_dir / f"ckpt_step_{step:06d}.pt"
        save_ckpt(state, ckpt_path)
        _save_last_symlink(ckpt_path)

    _log(
        f"[init] train_clips={len(train_clips)} iters={args.iters} "
        f"grad_accum={args.grad_accum_steps} refine={iterative_refine_enabled} poly_l={poly_degree_l}"
    )

    dtype = torch.float32
    for step in range(int(start_step), int(args.iters) + 1):
        optimizer.zero_grad(set_to_none=True)

        total_sum = 0.0
        rgb_sum = 0.0
        ssim_value_sum = 0.0
        depth_sum = 0.0
        vel_sum = 0.0
        psnr_sum = 0.0

        for _ in range(max(1, int(args.grad_accum_steps))):
            clip = random.choice(train_clips)
            clip_id = str(clip.get("clip_id", ""))
            if not clip_id:
                raise ValueError("clip 缺少 clip_id")

            clip_len_frames = int(clip.get("clip_len_frames", 0))
            if clip_len_frames <= 0:
                raise ValueError("clip_len_frames 非法")
            abs_lidar_ts = [get_lidar_timestamp(dict(clip), i) for i in range(clip_len_frames)]
            t_norm = normalize_timestamps_to_unit_range(abs_lidar_ts)

            input_frame_indices = clip.get("input_frame_indices")
            target_frame_indices = clip.get("target_frame_indices")
            if not isinstance(input_frame_indices, list) or not isinstance(target_frame_indices, list):
                raise ValueError("clip.input_frame_indices/target_frame_indices 格式非法")
            if not target_frame_indices:
                raise ValueError("target_frame_indices 为空")
            frame_index = int(random.choice(target_frame_indices))

            g_item = cache.get_or_build(
                key=clip_id,
                clip=clip,
                data_root=args.data_root,
                input_frame_indices=[int(x) for x in input_frame_indices],
                timestamp_frame_indices=list(range(clip_len_frames)),
                camera_names=[args.camera_name],
                voxel_size_m=float(voxel_size_m),
                knn_k=int(knn_k),
                opacity_init=float(opacity_init),
                random_seed=seed,
                num_sky_points=int(args.num_sky_points),
                max_gaussians=args.max_gaussians,
                lidar_sensor_id=lidar_sensor_id,
            )

            gaussians_world = torch_gaussian_set_from_numpy(
                positions=torch.from_numpy(g_item.positions).to(device=device, dtype=dtype),
                rotations=torch.from_numpy(g_item.rotations).to(device=device, dtype=dtype),
                scales=torch.from_numpy(g_item.scales).to(device=device, dtype=dtype),
                opacities=torch.from_numpy(g_item.opacities).to(device=device, dtype=dtype),
                colors=torch.from_numpy(g_item.colors).to(device=device, dtype=dtype),
                timestamps=torch.from_numpy(g_item.timestamps).to(device=device, dtype=dtype),
            )

            ego0_pose = get_lidar_pose(dict(clip), ego0_frame_index)
            frame = build_frame_transform_from_ego0_pose(ego0_pose, device=device, dtype=dtype)

            view = _load_camera_view_single(
                clip,
                frame_index=frame_index,
                data_root=args.data_root,
                camera_name=args.camera_name,
            )
            image = view.image
            height, width = image.shape[:2]
            target_rgb = torch.from_numpy(image).to(device=device, dtype=dtype)

            t_target = torch.tensor(float(t_norm[frame_index]), device=device, dtype=dtype)
            camera = build_pinhole_camera_from_pandaset(
                view.intrinsics,
                view.pose,
                (width, height),
                device=device,
                dtype=dtype,
            )

            target_depth = None
            target_valid = None
            if use_depth:
                lidar_paths = clip.get("lidar_paths")
                if not isinstance(lidar_paths, list):
                    raise ValueError("clip.lidar_paths 格式非法")
                lidar_pose = get_lidar_pose(dict(clip), frame_index)
                lidar_points_world, _ = load_lidar_frame(
                    str(Path(args.data_root) / str(lidar_paths[frame_index])),
                    sensor_id=lidar_sensor_id,
                )
                points_cam = transform_lidar_to_camera(
                    lidar_points_world,
                    lidar_pose,
                    view.pose,
                    points_in_world=True,
                )
                uv, depth, mask = project_points_to_camera(points_cam, view.intrinsics, (width, height))
                depth_map, depth_valid = _render_sparse_depth_map(uv, depth, mask, (width, height))
                target_depth = torch.from_numpy(depth_map).to(device=device, dtype=dtype)
                target_valid = torch.from_numpy(depth_valid).to(device=device)

            if iterative_refine_enabled and refine_model is not None:
                out0 = model.base(gaussians_world, frame)
                velocities_world = out0.velocities_world
                gaussians_iter = out0.gaussians_world
                gaussians_grad = None

                def _retain(gaussians: TorchGaussianSet) -> None:
                    for tensor in (
                        gaussians.positions,
                        gaussians.rotations,
                        gaussians.scales,
                        gaussians.opacities,
                        gaussians.colors,
                    ):
                        if tensor.requires_grad:
                            tensor.retain_grad()

                def _extract(gaussians: TorchGaussianSet) -> "torch.Tensor":
                    grads = []
                    for name, tensor in (
                        ("positions", gaussians.positions),
                        ("rotations", gaussians.rotations),
                        ("scales", gaussians.scales),
                        ("opacities", gaussians.opacities),
                        ("colors", gaussians.colors),
                    ):
                        if tensor.grad is None:
                            raise ValueError(f"未获取到 {name}.grad")
                        grads.append(tensor.grad[:, None] if name == "opacities" else tensor.grad)
                    return torch.cat(grads, dim=-1)

                for iter_idx in range(int(refine_iters)):
                    gaussians_iter = TorchGaussianSet(
                        positions=gaussians_iter.positions.detach().requires_grad_(True),
                        rotations=gaussians_iter.rotations.detach().requires_grad_(True),
                        scales=gaussians_iter.scales.detach().requires_grad_(True),
                        opacities=gaussians_iter.opacities.detach().requires_grad_(True),
                        colors=gaussians_iter.colors.detach().requires_grad_(True),
                        timestamps=gaussians_iter.timestamps.detach(),
                    )
                    if iter_idx > 0 and gaussians_grad is not None:
                        delta = model.refine(gaussians_iter, gaussians_grad).delta_g
                        gaussians_iter = apply_delta_g_to_gaussians(gaussians_iter, delta)
                    _retain(gaussians_iter)

                    if poly_degree_l <= 0:
                        gaussians_t = apply_linear_motion(gaussians_iter, velocities_world, t_target)
                    else:
                        gaussians_t = apply_polynomial_motion(gaussians_iter, velocities_world, t_target)
                    gaussians_act = activate_gaussians_for_render(gaussians_t, cfg)
                    render_out = render_gsplat(gaussians_act, camera, near=near, far=far, render_mode=render_mode)
                    if render_out.rgb is None:
                        raise ValueError("render_mode 未输出 RGB")

                    rgb_weight_map = None
                    if use_velocity_reweighting:
                        delta_t_norm = _compute_delta_t_norm(t_norm, frame_index)
                        vr_out = render_rendered_velocity_map(
                            gaussians_world_t=gaussians_t,
                            velocities_world=velocities_world,
                            camera=camera,
                            cfg=cfg,
                            delta_t_norm=float(delta_t_norm),
                            t_target_norm=float(t_target.item()),
                        )
                        rgb_weight_map = build_dynamic_weight_map_from_flow(
                            vr_out.vr_mag,
                            render_out.alpha,
                            alpha_threshold=float(alpha_threshold),
                            clip_mag=float(clip_mag_px),
                            blur_sigma=float(blur_sigma),
                            blur_window=int(blur_window),
                        )

                    losses = compute_flux4d_base_losses(
                        pred_rgb=render_out.rgb,
                        target_rgb=target_rgb,
                        rgb_weight_map=rgb_weight_map,
                        pred_depth=render_out.depth,
                        target_depth=target_depth,
                        target_depth_valid=target_valid,
                        velocities_world=velocities_world,
                        lambda_rgb=lambda_rgb,
                        lambda_ssim=lambda_ssim,
                        lambda_depth=lambda_depth,
                        lambda_vel=lambda_vel,
                        ssim_window=ssim_window,
                    )
                    (losses.total / float(args.grad_accum_steps * refine_iters)).backward()
                    gaussians_grad = _extract(gaussians_iter).detach()

                    with torch.no_grad():
                        ssim_value = 1.0 - losses.ssim
                        total_sum += float(losses.total.detach()) / float(refine_iters)
                        rgb_sum += float(losses.rgb_l1.detach()) / float(refine_iters)
                        ssim_value_sum += float(ssim_value.detach()) / float(refine_iters)
                        depth_sum += float(losses.depth_l1.detach()) / float(refine_iters)
                        vel_sum += float(losses.vel.detach()) / float(refine_iters)
                        psnr_sum += float(compute_psnr_torch(render_out.rgb, target_rgb).detach()) / float(refine_iters)
            else:
                out = model(gaussians_world, frame)
                velocities_world = out.velocities_world
                if poly_degree_l <= 0:
                    gaussians_t = apply_linear_motion(out.gaussians_world, velocities_world, t_target)
                else:
                    gaussians_t = apply_polynomial_motion(out.gaussians_world, velocities_world, t_target)
                gaussians_act = activate_gaussians_for_render(gaussians_t, cfg)
                render_out = render_gsplat(gaussians_act, camera, near=near, far=far, render_mode=render_mode)
                if render_out.rgb is None:
                    raise ValueError("render_mode 未输出 RGB")

                rgb_weight_map = None
                if use_velocity_reweighting:
                    delta_t_norm = _compute_delta_t_norm(t_norm, frame_index)
                    vr_out = render_rendered_velocity_map(
                        gaussians_world_t=gaussians_t,
                        velocities_world=velocities_world,
                        camera=camera,
                        cfg=cfg,
                        delta_t_norm=float(delta_t_norm),
                        t_target_norm=float(t_target.item()),
                    )
                    rgb_weight_map = build_dynamic_weight_map_from_flow(
                        vr_out.vr_mag,
                        render_out.alpha,
                        alpha_threshold=float(alpha_threshold),
                        clip_mag=float(clip_mag_px),
                        blur_sigma=float(blur_sigma),
                        blur_window=int(blur_window),
                    )

                losses = compute_flux4d_base_losses(
                    pred_rgb=render_out.rgb,
                    target_rgb=target_rgb,
                    rgb_weight_map=rgb_weight_map,
                    pred_depth=render_out.depth,
                    target_depth=target_depth,
                    target_depth_valid=target_valid,
                    velocities_world=velocities_world,
                    lambda_rgb=lambda_rgb,
                    lambda_ssim=lambda_ssim,
                    lambda_depth=lambda_depth,
                    lambda_vel=lambda_vel,
                    ssim_window=ssim_window,
                )
                (losses.total / float(args.grad_accum_steps)).backward()
                with torch.no_grad():
                    ssim_value = 1.0 - losses.ssim
                    total_sum += float(losses.total.detach())
                    rgb_sum += float(losses.rgb_l1.detach())
                    ssim_value_sum += float(ssim_value.detach())
                    depth_sum += float(losses.depth_l1.detach())
                    vel_sum += float(losses.vel.detach())
                    psnr_sum += float(compute_psnr_torch(render_out.rgb, target_rgb).detach())

        optimizer.step()

        if step % int(args.log_every) == 0 or step == 1:
            denom = float(max(1, int(args.grad_accum_steps)))
            _log(
                f"[{step:06d}] total={total_sum / denom:.6f} "
                f"rgb={rgb_sum / denom:.6f} ssim={ssim_value_sum / denom:.4f} "
                f"depth={depth_sum / denom:.6f} vel={vel_sum / denom:.6f} "
                f"psnr={psnr_sum / denom:.2f}"
            )

        save_every = int(args.save_ckpt_every)
        if step == int(args.iters) or (save_every > 0 and step % save_every == 0):
            _save_checkpoint(step)

    _log(f"[done] step={args.iters}")
    log_file.close()


@dataclass(frozen=True)
class Stage6EvalArgs:
    """阶段6评测参数。"""

    index_path: str
    data_root: str
    camera_name: str
    device: str
    ckpt_path: str
    output_dir: str
    cache_dir: str
    cache_max_items: int
    num_sky_points: int
    max_gaussians: Optional[int]
    max_eval_clips: Optional[int]
    save_renders: bool
    save_max_clips: int


def _as_int_list(values: object, name: str) -> List[int]:
    if not isinstance(values, list):
        raise ValueError(f"{name} 必须为 list")
    out: List[int] = []
    for item in values:
        if not isinstance(item, int):
            raise ValueError(f"{name} 必须为 int 列表")
        out.append(int(item))
    return out


def eval_stage6(cfg: Mapping[str, object], args: Stage6EvalArgs) -> Dict[str, object]:
    """阶段6：评测（NVS + Scene Flow）。

    Returns:
        metrics 字典（可直接 JSON 序列化）。
    """
    _require_torch()
    device = torch.device(args.device)
    if device.type != "cuda":  # pragma: no cover
        raise ValueError("阶段6评测需要 CUDA（gsplat/spconv）")

    data_cfg = cfg.get("data")
    init_cfg = cfg.get("init")
    loss_cfg = cfg.get("loss")
    render_cfg = cfg.get("render")
    coord_cfg = cfg.get("coord")
    model_cfg = cfg.get("model")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("cfg['data'] 缺失或格式非法")
    if not isinstance(init_cfg, Mapping):
        raise ValueError("cfg['init'] 缺失或格式非法")
    if not isinstance(loss_cfg, Mapping):
        raise ValueError("cfg['loss'] 缺失或格式非法")
    if not isinstance(render_cfg, Mapping):
        raise ValueError("cfg['render'] 缺失或格式非法")
    if not isinstance(coord_cfg, Mapping):
        raise ValueError("cfg['coord'] 缺失或格式非法")
    if not isinstance(model_cfg, Mapping):
        raise ValueError("cfg['model'] 缺失或格式非法")

    ego0_frame_index = int(coord_cfg.get("ego0_frame_index", 0))
    lidar_sensor_id = int(data_cfg.get("lidar_sensor_id", -1))

    head_cfg = model_cfg.get("head")
    if not isinstance(head_cfg, Mapping):
        raise ValueError("cfg['model']['head'] 缺失或格式非法")
    motion_cfg = head_cfg.get("motion")
    if not isinstance(motion_cfg, Mapping):
        raise ValueError("cfg['model']['head']['motion'] 缺失或格式非法")
    poly_degree_l = int(motion_cfg.get("poly_degree_l", 0))

    refine_cfg = model_cfg.get("iterative_refine")
    iterative_refine_enabled = False
    refine_iters = 1
    if isinstance(refine_cfg, Mapping):
        iterative_refine_enabled = bool(refine_cfg.get("enabled", False))
        refine_iters = int(refine_cfg.get("num_iters", 3))
    refine_iters = max(1, int(refine_iters))

    base_model = build_flux4d_base_model_frames(cfg)
    refine_model: Optional[Flux4DRefineModel] = None
    if iterative_refine_enabled:
        refine_model = build_flux4d_refine_model(cfg)

        class _Stage6Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = base_model
                self.refine = refine_model

            def forward(self, gaussians: TorchGaussianSet, frame_transform: object) -> object:
                return self.base(gaussians, frame_transform)

        model = _Stage6Wrapper().to(device=device)
    else:
        model = base_model.to(device=device)
    model.eval()

    ckpt = load_ckpt(args.ckpt_path, map_location="cpu")
    model_state = ckpt.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint 缺少 model_state")
    model.load_state_dict(model_state, strict=False)

    payload = load_clip_index(args.index_path)
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("索引缺少 clips 列表或为空")

    eval_clips: List[Mapping[str, object]] = []
    for clip in clips:
        if not isinstance(clip, Mapping):
            continue
        if str(clip.get("split")) != "test":
            continue
        if str(clip.get("setting")) != "eval_future":
            continue
        eval_clips.append(clip)
    if args.max_eval_clips is not None:
        eval_clips = eval_clips[: int(args.max_eval_clips)]
    if not eval_clips:
        raise ValueError("未找到评测 clips：请检查 index_path/preset/split/setting")

    downsample_cfg = init_cfg.get("downsample", {})
    voxel_size_m = float(downsample_cfg.get("voxel_size_m", 0.2)) if isinstance(downsample_cfg, Mapping) else 0.2
    knn_k = int(init_cfg.get("scale_knn_k", 3))
    opacity_init = float(init_cfg.get("opacity_init", 0.5))

    loss_lambda_depth = float(loss_cfg.get("lambda_depth", 0.0))
    use_depth = loss_lambda_depth > 0.0
    depth_cfg = loss_cfg.get("depth")
    if isinstance(depth_cfg, Mapping):
        use_depth = use_depth and bool(depth_cfg.get("use_projected_lidar_depth", True))
    ssim_window = int(loss_cfg.get("ssim_window", 11))

    render_mode = str(render_cfg.get("mode", "RGB+ED"))
    near = float(render_cfg.get("near", 0.1))
    far = float(render_cfg.get("far", 200.0))

    # Dynamic mask / scene flow config (documented defaults)
    dynamic_trans_thresh_m = 0.05
    dynamic_yaw_thresh_deg = 1.0
    dynamic_smooth_window = 3
    dynamic_dilate_radius_px = 5
    bucket_names = ("background", "vehicle", "pedestrian", "cyclist", "other")
    label_to_bucket = build_default_label_to_bucket_map()

    cache = LiftCache(Path(args.cache_dir), max_items_in_memory=int(args.cache_max_items))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = out_dir / "renders"
    if args.save_renders:
        renders_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float32
    psnr_full: List[float] = []
    ssim_full: List[float] = []
    depth_rmse_full: List[float] = []
    psnr_dyn: List[float] = []
    ssim_dyn: List[float] = []
    depth_rmse_dyn: List[float] = []

    scene_flow_frame_metrics: List[Dict[str, object]] = []
    sf_bucket_sum: Dict[str, float] = {}
    sf_bucket_count: Dict[str, int] = {}
    sf_3way_sum: Dict[str, float] = {}
    sf_3way_count: Dict[str, int] = {}
    sf_epe_list: List[float] = []
    sf_acc5_list: List[float] = []
    sf_acc10_list: List[float] = []
    sf_theta_list: List[float] = []

    saved_clip = 0
    for clip_idx, clip in enumerate(eval_clips):
        clip_id = str(clip.get("clip_id", f"clip_{clip_idx:03d}"))
        scene_id = str(clip.get("scene_id", ""))
        if not scene_id:
            raise ValueError("clip 缺少 scene_id")

        clip_len_frames = int(clip.get("clip_len_frames", 0))
        if clip_len_frames <= 0:
            raise ValueError("clip_len_frames 非法")
        abs_lidar_ts = [get_lidar_timestamp(dict(clip), i) for i in range(clip_len_frames)]
        t_norm = normalize_timestamps_to_unit_range(abs_lidar_ts)

        input_frame_indices = _as_int_list(clip.get("input_frame_indices"), "input_frame_indices")
        target_frame_indices = _as_int_list(clip.get("target_frame_indices"), "target_frame_indices")
        if not target_frame_indices:
            raise ValueError("target_frame_indices 为空")

        g_item = cache.get_or_build(
            key=clip_id,
            clip=clip,
            data_root=args.data_root,
            input_frame_indices=input_frame_indices,
            timestamp_frame_indices=list(range(clip_len_frames)),
            camera_names=[args.camera_name],
            voxel_size_m=float(voxel_size_m),
            knn_k=int(knn_k),
            opacity_init=float(opacity_init),
            random_seed=0,
            num_sky_points=int(args.num_sky_points),
            max_gaussians=args.max_gaussians,
            lidar_sensor_id=lidar_sensor_id,
        )

        gaussians_world = torch_gaussian_set_from_numpy(
            positions=torch.from_numpy(g_item.positions).to(device=device, dtype=dtype),
            rotations=torch.from_numpy(g_item.rotations).to(device=device, dtype=dtype),
            scales=torch.from_numpy(g_item.scales).to(device=device, dtype=dtype),
            opacities=torch.from_numpy(g_item.opacities).to(device=device, dtype=dtype),
            colors=torch.from_numpy(g_item.colors).to(device=device, dtype=dtype),
            timestamps=torch.from_numpy(g_item.timestamps).to(device=device, dtype=dtype),
        )

        ego0_pose = get_lidar_pose(dict(clip), ego0_frame_index)
        frame = build_frame_transform_from_ego0_pose(ego0_pose, device=device, dtype=dtype)

        # Preload cuboids for frames used by metrics to reduce IO.
        needed_frames = sorted(set(input_frame_indices + target_frame_indices))
        cuboids_by_frame: Dict[int, List[PandaSetCuboid]] = {}
        for fid in needed_frames:
            cuboids_by_frame[int(fid)] = load_pandaset_cuboids_frame(args.data_root, scene_id, int(fid))
        dynamic_flags_by_frame = compute_cuboid_dynamic_flags_by_frame(
            cuboids_by_frame,
            trans_thresh_m=float(dynamic_trans_thresh_m),
            yaw_thresh_deg=float(dynamic_yaw_thresh_deg),
            smoothing_window=int(dynamic_smooth_window),
        )

        with torch.no_grad():
            out0 = model.base(gaussians_world, frame) if iterative_refine_enabled else model(gaussians_world, frame)
            velocities_world = out0.velocities_world
            gaussians_final = out0.gaussians_world

        # Iterative refinement at inference time: use only observed/input frames for gradient feedback.
        if iterative_refine_enabled and refine_model is not None:
            gaussians_iter = gaussians_final
            gaussians_grad = None
            obs_frame = int(input_frame_indices[-1])  # deterministic choice: last observed frame
            view_obs = _load_camera_view_single(
                clip,
                frame_index=obs_frame,
                data_root=args.data_root,
                camera_name=args.camera_name,
            )
            obs_image = view_obs.image
            h_obs, w_obs = obs_image.shape[:2]
            target_rgb_obs = torch.from_numpy(obs_image).to(device=device, dtype=dtype)
            t_obs = torch.tensor(float(t_norm[obs_frame]), device=device, dtype=dtype)
            cam_obs = build_pinhole_camera_from_pandaset(
                view_obs.intrinsics,
                view_obs.pose,
                (w_obs, h_obs),
                device=device,
                dtype=dtype,
            )
            target_depth_obs = None
            target_valid_obs = None
            if use_depth:
                lidar_paths = clip.get("lidar_paths")
                if not isinstance(lidar_paths, list):
                    raise ValueError("clip.lidar_paths 格式非法")
                lidar_pose = get_lidar_pose(dict(clip), obs_frame)
                lidar_points_world, _ = load_lidar_frame(
                    str(Path(args.data_root) / str(lidar_paths[obs_frame])),
                    sensor_id=lidar_sensor_id,
                )
                points_cam = transform_lidar_to_camera(
                    lidar_points_world,
                    lidar_pose,
                    view_obs.pose,
                    points_in_world=True,
                )
                uv, depth, mask = project_points_to_camera(
                    points_cam, view_obs.intrinsics, (w_obs, h_obs)
                )
                depth_map, depth_valid = _render_sparse_depth_map(uv, depth, mask, (w_obs, h_obs))
                target_depth_obs = torch.from_numpy(depth_map).to(device=device, dtype=dtype)
                target_valid_obs = torch.from_numpy(depth_valid).to(device=device)

            def _retain(gaussians: TorchGaussianSet) -> None:
                for tensor in (
                    gaussians.positions,
                    gaussians.rotations,
                    gaussians.scales,
                    gaussians.opacities,
                    gaussians.colors,
                ):
                    if tensor.requires_grad:
                        tensor.retain_grad()

            def _extract(gaussians: TorchGaussianSet) -> "torch.Tensor":
                grads = []
                for name, tensor in (
                    ("positions", gaussians.positions),
                    ("rotations", gaussians.rotations),
                    ("scales", gaussians.scales),
                    ("opacities", gaussians.opacities),
                    ("colors", gaussians.colors),
                ):
                    if tensor.grad is None:
                        raise ValueError(f"未获取到 {name}.grad")
                    grads.append(tensor.grad[:, None] if name == "opacities" else tensor.grad)
                return torch.cat(grads, dim=-1)

            for iter_idx in range(int(refine_iters)):
                with torch.enable_grad():
                    gaussians_iter = TorchGaussianSet(
                        positions=gaussians_iter.positions.detach().requires_grad_(True),
                        rotations=gaussians_iter.rotations.detach().requires_grad_(True),
                        scales=gaussians_iter.scales.detach().requires_grad_(True),
                        opacities=gaussians_iter.opacities.detach().requires_grad_(True),
                        colors=gaussians_iter.colors.detach().requires_grad_(True),
                        timestamps=gaussians_iter.timestamps.detach(),
                    )
                    if iter_idx > 0 and gaussians_grad is not None:
                        delta = model.refine(gaussians_iter, gaussians_grad).delta_g
                        gaussians_iter = apply_delta_g_to_gaussians(gaussians_iter, delta)
                    _retain(gaussians_iter)

                    if poly_degree_l <= 0:
                        gaussians_t = apply_linear_motion(gaussians_iter, velocities_world, t_obs)
                    else:
                        gaussians_t = apply_polynomial_motion(gaussians_iter, velocities_world, t_obs)
                    gaussians_act = activate_gaussians_for_render(gaussians_t, cfg)
                    render_out = render_gsplat(
                        gaussians_act, cam_obs, near=near, far=far, render_mode=render_mode
                    )
                    if render_out.rgb is None:
                        raise ValueError("render_mode 未输出 RGB")
                    losses = compute_flux4d_base_losses(
                        pred_rgb=render_out.rgb,
                        target_rgb=target_rgb_obs,
                        rgb_weight_map=None,
                        pred_depth=render_out.depth,
                        target_depth=target_depth_obs,
                        target_depth_valid=target_valid_obs,
                        velocities_world=velocities_world,
                        lambda_rgb=float(loss_cfg.get("lambda_rgb", 0.8)),
                        lambda_ssim=float(loss_cfg.get("lambda_ssim", 0.2)),
                        lambda_depth=float(loss_cfg.get("lambda_depth", 0.0)),
                        lambda_vel=float(loss_cfg.get("lambda_vel", 0.0)),  # inference refine 不再强调速度正则
                        ssim_window=ssim_window,
                    )
                    losses.total.backward()
                    gaussians_grad = _extract(gaussians_iter).detach()
                gaussians_iter = TorchGaussianSet(
                    positions=gaussians_iter.positions.detach(),
                    rotations=gaussians_iter.rotations.detach(),
                    scales=gaussians_iter.scales.detach(),
                    opacities=gaussians_iter.opacities.detach(),
                    colors=gaussians_iter.colors.detach(),
                    timestamps=gaussians_iter.timestamps.detach(),
                )
            gaussians_final = gaussians_iter

        # --- NVS metrics ---
        for frame_index in target_frame_indices:
            view = _load_camera_view_single(
                clip,
                frame_index=int(frame_index),
                data_root=args.data_root,
                camera_name=args.camera_name,
            )
            image = view.image
            height, width = image.shape[:2]
            target_rgb = torch.from_numpy(image).to(device=device, dtype=dtype)
            t_target = torch.tensor(float(t_norm[int(frame_index)]), device=device, dtype=dtype)
            camera = build_pinhole_camera_from_pandaset(
                view.intrinsics,
                view.pose,
                (width, height),
                device=device,
                dtype=dtype,
            )
            if poly_degree_l <= 0:
                gaussians_t = apply_linear_motion(gaussians_final, velocities_world, t_target)
            else:
                gaussians_t = apply_polynomial_motion(gaussians_final, velocities_world, t_target)
            gaussians_act = activate_gaussians_for_render(gaussians_t, cfg)
            with torch.no_grad():
                render_out = render_gsplat(
                    gaussians_act, camera, near=near, far=far, render_mode=render_mode
                )
            if render_out.rgb is None:
                raise ValueError("render_mode 未输出 RGB")

            # Depth GT
            depth_gt = None
            depth_valid = None
            if use_depth:
                lidar_paths = clip.get("lidar_paths")
                if not isinstance(lidar_paths, list):
                    raise ValueError("clip.lidar_paths 格式非法")
                lidar_pose = get_lidar_pose(dict(clip), int(frame_index))
                lidar_points_world, _ = load_lidar_frame(
                    str(Path(args.data_root) / str(lidar_paths[int(frame_index)])),
                    sensor_id=lidar_sensor_id,
                )
                points_cam = transform_lidar_to_camera(
                    lidar_points_world,
                    lidar_pose,
                    view.pose,
                    points_in_world=True,
                )
                uv, depth, mask = project_points_to_camera(points_cam, view.intrinsics, (width, height))
                depth_map, valid_map = _render_sparse_depth_map(uv, depth, mask, (width, height))
                depth_gt = torch.from_numpy(depth_map).to(device=device, dtype=dtype)
                depth_valid = torch.from_numpy(valid_map).to(device=device)

            with torch.no_grad():
                psnr = float(compute_psnr_torch(render_out.rgb, target_rgb).cpu())
                ssim = float(
                    compute_ssim_value_torch(render_out.rgb, target_rgb, window_size=ssim_window).cpu()
                )
                if render_out.depth is not None and depth_gt is not None and depth_valid is not None:
                    depth_rmse = float(
                        compute_depth_rmse_torch(
                            render_out.depth, depth_gt, valid_mask=depth_valid
                        ).cpu()
                    )
                else:
                    depth_rmse = 0.0
                psnr_full.append(psnr)
                ssim_full.append(ssim)
                depth_rmse_full.append(depth_rmse)

            # dynamic-only mask
            dynamic_mask = None
            try:
                lidar_paths = clip.get("lidar_paths")
                if isinstance(lidar_paths, list):
                    lidar_pose = get_lidar_pose(dict(clip), int(frame_index))
                    lidar_points_world, _ = load_lidar_frame(
                        str(Path(args.data_root) / str(lidar_paths[int(frame_index)])),
                        sensor_id=lidar_sensor_id,
                    )
                    cuboids = cuboids_by_frame[int(frame_index)]
                    assigned, _ = assign_points_to_cuboids_world(lidar_points_world, cuboids)
                    dyn_flags = dynamic_flags_by_frame.get(int(frame_index), {})
                    dyn_points = np.zeros((lidar_points_world.shape[0],), dtype=bool)
                    for cub_idx, cub in enumerate(cuboids):
                        if not bool(dyn_flags.get(cub.uuid, False)):
                            continue
                        dyn_points |= assigned == int(cub_idx)
                    if np.any(dyn_points):
                        points_cam = transform_lidar_to_camera(
                            lidar_points_world,
                            lidar_pose,
                            view.pose,
                            points_in_world=True,
                        )
                        uv, _, mask = project_points_to_camera(
                            points_cam, view.intrinsics, (width, height)
                        )
                        keep = dyn_points & mask
                        mask_img = np.zeros((height, width), dtype=bool)
                        if np.any(keep):
                            u = np.clip(np.round(uv[keep, 0]).astype(np.int64), 0, width - 1)
                            v = np.clip(np.round(uv[keep, 1]).astype(np.int64), 0, height - 1)
                            mask_img[v, u] = True
                        dynamic_mask = _dilate_mask(mask_img, radius_px=int(dynamic_dilate_radius_px))
            except RuntimeError:
                dynamic_mask = None

            if dynamic_mask is not None and np.any(dynamic_mask):
                dyn_mask_t = torch.from_numpy(dynamic_mask).to(device=device)
                with torch.no_grad():
                    psnr_d = float(
                        compute_psnr_torch(render_out.rgb, target_rgb, valid_mask=dyn_mask_t).cpu()
                    )
                    ssim_d = float(
                        compute_ssim_value_torch(
                            render_out.rgb,
                            target_rgb,
                            window_size=ssim_window,
                            weight_map=dyn_mask_t.to(dtype=dtype),
                        ).cpu()
                    )
                    if render_out.depth is not None and depth_gt is not None and depth_valid is not None:
                        depth_mask = dyn_mask_t & depth_valid
                        depth_rmse_d = float(
                            compute_depth_rmse_torch(
                                render_out.depth, depth_gt, valid_mask=depth_mask
                            ).cpu()
                        )
                    else:
                        depth_rmse_d = 0.0
                    psnr_dyn.append(psnr_d)
                    ssim_dyn.append(ssim_d)
                    depth_rmse_dyn.append(depth_rmse_d)

            if args.save_renders and saved_clip < int(args.save_max_clips):
                clip_out = renders_dir / clip_id
                clip_out.mkdir(parents=True, exist_ok=True)
                _save_rgb(clip_out / f"pred_rgb_f{int(frame_index):02d}.png", render_out.rgb.detach())
                _save_rgb(clip_out / f"gt_rgb_f{int(frame_index):02d}.png", target_rgb.detach())
                if dynamic_mask is not None:
                    _save_gray_u8(
                        clip_out / f"dynamic_mask_f{int(frame_index):02d}.png",
                        (dynamic_mask.astype(np.uint8) * 255),
                    )

        if args.save_renders and saved_clip < int(args.save_max_clips):
            saved_clip += 1

        # --- Scene Flow metrics (evaluate on input frames, adjacent pairs) ---
        for frame0 in input_frame_indices:
            frame1 = int(frame0) + 1
            if frame1 >= clip_len_frames:
                continue

            # Select points originating from frame0 and exclude sky/extra (-1).
            src = g_item.source_frame_indices
            point_mask = (src == int(frame0))
            if not np.any(point_mask):
                continue

            with torch.no_grad():
                points_t0 = gaussians_final.positions[torch.from_numpy(point_mask).to(device=device)].detach().cpu().numpy()
                if poly_degree_l <= 0:
                    delta = velocities_world[torch.from_numpy(point_mask).to(device=device), 0:3] * float(
                        t_norm[frame1] - t_norm[frame0]
                    )
                else:
                    delta = _integrate_polynomial_motion_delta(
                        velocities_world[torch.from_numpy(point_mask).to(device=device)],
                        float(t_norm[frame0]),
                        float(t_norm[frame1]),
                    )
                flow_pred = delta.detach().cpu().numpy().astype(np.float32, copy=False)

            cub0 = cuboids_by_frame[int(frame0)]
            cub1 = cuboids_by_frame[int(frame1)]
            assigned0, _ = assign_points_to_cuboids_world(points_t0, cub0)
            dyn_flags0 = dynamic_flags_by_frame.get(int(frame0), {})

            # FoV filter (front camera at frame0)
            view0 = _load_camera_view_single(
                clip,
                frame_index=int(frame0),
                data_root=args.data_root,
                camera_name=args.camera_name,
            )
            h0, w0 = view0.image.shape[:2]
            lidar_pose0 = get_lidar_pose(dict(clip), int(frame0))
            points_cam0 = transform_lidar_to_camera(points_t0, lidar_pose0, view0.pose, points_in_world=True)
            uv0, _, mask0 = project_points_to_camera(points_cam0, view0.intrinsics, (w0, h0))
            fov_mask = mask0.astype(bool, copy=False)

            gt = build_scene_flow_gt_from_cuboids(
                points_world_t0=points_t0,
                assigned_cuboid_indices_t0=assigned0,
                cuboids_t0=cub0,
                cuboids_t1=cub1,
                dynamic_flags_t0=dyn_flags0,
                label_to_bucket=label_to_bucket,
                bucket_names=bucket_names,
                default_bucket="other",
            )
            valid = gt.valid_mask & fov_mask
            gt_fov = gt.__class__(
                flow_gt_world=gt.flow_gt_world,
                valid_mask=valid,
                bs_mask=gt.bs_mask & fov_mask,
                fs_mask=gt.fs_mask & fov_mask,
                fd_mask=gt.fd_mask & fov_mask,
                bucket_names=gt.bucket_names,
                bucket_indices=gt.bucket_indices,
            )
            metrics = compute_scene_flow_metrics(flow_pred, gt_fov)

            sf_epe_list.append(float(metrics.epe3d))
            sf_acc5_list.append(float(metrics.acc5))
            sf_acc10_list.append(float(metrics.acc10))
            sf_theta_list.append(float(metrics.theta_e))
            merge_bucket_stats(sf_bucket_sum, sf_bucket_count, metrics.bucketed_nepe, metrics.bucketed_nepe_count)
            merge_bucket_stats(sf_3way_sum, sf_3way_count, metrics.epe_3way, metrics.epe_3way_count)
            scene_flow_frame_metrics.append(
                {
                    "clip_id": clip_id,
                    "scene_id": scene_id,
                    "frame0": int(frame0),
                    "frame1": int(frame1),
                    "epe3d": float(metrics.epe3d),
                    "acc5": float(metrics.acc5),
                    "acc10": float(metrics.acc10),
                    "theta_e": float(metrics.theta_e),
                    "epe_3way": dict(metrics.epe_3way),
                    "epe_3way_count": dict(metrics.epe_3way_count),
                }
            )

    def _mean(values: Sequence[float]) -> float:
        return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0

    def _std(values: Sequence[float]) -> float:
        return float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0

    bucket_means: Dict[str, float] = {}
    for name in bucket_names:
        count = int(sf_bucket_count.get(name, 0))
        bucket_means[name] = float(sf_bucket_sum.get(name, 0.0) / float(count)) if count > 0 else 0.0
    epe3way_means: Dict[str, float] = {}
    for name in ("BS", "FS", "FD"):
        count = int(sf_3way_count.get(name, 0))
        epe3way_means[name] = float(sf_3way_sum.get(name, 0.0) / float(count)) if count > 0 else 0.0

    metrics_out: Dict[str, object] = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "ckpt": os.path.abspath(args.ckpt_path),
            "index_path": args.index_path,
            "num_eval_clips": len(eval_clips),
            "camera_name": args.camera_name,
        },
        "nvs": {
            "full": {
                "psnr_mean": _mean(psnr_full),
                "psnr_std": _std(psnr_full),
                "ssim_mean": _mean(ssim_full),
                "ssim_std": _std(ssim_full),
                "depth_rmse_mean": _mean(depth_rmse_full),
                "depth_rmse_std": _std(depth_rmse_full),
                "count": len(psnr_full),
            },
            "dynamic": {
                "psnr_mean": _mean(psnr_dyn),
                "psnr_std": _std(psnr_dyn),
                "ssim_mean": _mean(ssim_dyn),
                "ssim_std": _std(ssim_dyn),
                "depth_rmse_mean": _mean(depth_rmse_dyn),
                "depth_rmse_std": _std(depth_rmse_dyn),
                "count": len(psnr_dyn),
            },
        },
        "scene_flow": {
            "epe3d_mean": _mean(sf_epe_list),
            "acc5_mean": _mean(sf_acc5_list),
            "acc10_mean": _mean(sf_acc10_list),
            "theta_e_mean": _mean(sf_theta_list),
            "count": len(sf_epe_list),
            "epe_3way_mean": epe3way_means,
            "epe_3way_count": dict(sf_3way_count),
            "bucketed_nepe_mean": bucket_means,
            "bucketed_nepe_count": dict(sf_bucket_count),
            "per_pair": scene_flow_frame_metrics,
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, ensure_ascii=False))
    return metrics_out
