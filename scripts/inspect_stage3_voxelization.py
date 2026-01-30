#!/usr/bin/env python3
"""阶段3：体素化流程的离线检查脚本（不依赖 torch/spconv）。

该脚本用于验证（本复现的实现约定）：
- PandaSet 点云/初始化高斯主存储在 `world/log frame`；
- 送入稀疏 3D U-Net 前，将中心 `p_world` 变换到 `ego0`（snippet 的第 0 帧）以稳定体素范围；
- `ego0` 下的点落在配置的 `point_cloud_range` 内，体素化后有效点/体素数量合理，且特征维度为 (14+1)。

Note:
    两份 PDF 未显式规定 canonical frame；此处的 `world→ego0` 为工程实现选择。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flux4d.datasets.pandaset_clips import load_clip_index  # noqa: E402
from flux4d.lift.lift_lidar import (  # noqa: E402
    GaussianSet,
    get_lidar_pose,
    build_initial_gaussians_for_clip_aggregated,
)
from flux4d.storm.gaussian_voxelizer import voxelize_points_numpy  # noqa: E402
from flux4d.utils.frames import build_frame_transform_numpy, transform_points_numpy  # noqa: E402


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


def _pack_features(gaussians: GaussianSet) -> np.ndarray:
    """将 GaussianSet 打包为阶段3的点级特征 (N, 15)。

    Args:
        gaussians: 高斯集合。

    Returns:
        特征数组，按 [p(3), q(4), s(3), o(1), c(3), T(1)] 拼接。
    """
    return np.concatenate(
        [
            gaussians.positions.astype(np.float32),
            gaussians.rotations.astype(np.float32),
            gaussians.scales.astype(np.float32),
            gaussians.opacities.astype(np.float32)[:, None],
            gaussians.colors.astype(np.float32),
            gaussians.timestamps.astype(np.float32)[:, None],
        ],
        axis=1,
    )


def _to_ego0(gaussians_world: GaussianSet, ego0_pose: Mapping[str, object]) -> GaussianSet:
    """将高斯中心从 world 坐标系转换到 ego0。

    Args:
        gaussians_world: world 坐标系下的高斯集合。
        ego0_pose: ego0 位姿（sensor->world）。

    Returns:
        ego0 坐标系下的高斯集合（其余字段保持不变）。
    """
    frame = build_frame_transform_numpy(ego0_pose)
    positions_ego0 = transform_points_numpy(gaussians_world.positions, frame.T_ego0_world)
    return GaussianSet(
        positions=positions_ego0,
        scales=gaussians_world.scales,
        rotations=gaussians_world.rotations,
        colors=gaussians_world.colors,
        opacities=gaussians_world.opacities,
        timestamps=gaussians_world.timestamps,
        velocities=gaussians_world.velocities,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Inspect stage3 voxelization pipeline (numpy-only).")
    parser.add_argument("--config", default="configs/flux4d.py", help="Path to configs/flux4d.py")
    parser.add_argument(
        "--index-path",
        default="",
        help="Override clip index PKL path (default uses cfg['data']['index_tiny']).",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Override PandaSet root (default uses index meta.data_root).",
    )
    parser.add_argument("--clip-index", type=int, default=None, help="Clip index (default uses cfg['data']['clip_index']).")
    parser.add_argument("--ego0-frame-index", type=int, default=None, help="Ego0 frame index (default uses cfg['coord']['ego0_frame_index']).")
    parser.add_argument("--num-sky-points", type=int, default=20000, help="Sky points for debug (paper uses 1,000,000).")
    parser.add_argument("--out-json", default="", help="Optional output json path for stats.")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory for BEV plot (e.g. assets/vis/stage3_voxel_sanity).",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=200000,
        help="Max number of points used for BEV scatter plot.",
    )
    parser.add_argument(
        "--plot-window-m",
        type=float,
        default=400.0,
        help="BEV plot window size in meters (square side length, centered at ego0 origin).",
    )
    return parser


def _get_nested(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    """安全读取嵌套字典字段。"""
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"cfg['{key}'] 缺失或格式非法")
    return value


def _summarize(res: object) -> Dict[str, object]:
    """将统计信息整理为可 JSON 序列化的字典。"""
    return dict(res)


def _save_bev_plot(
    points_ego0: np.ndarray,
    point_cloud_range: Tuple[float, float, float, float, float, float],
    out_dir: Path,
    max_points: int,
    plot_window_m: float,
) -> None:
    """保存 ego0 下的 BEV 散点图用于快速检查范围与分布。

    Args:
        points_ego0: ego0 坐标系点，形状为 (N, 3)。
        point_cloud_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]。
        out_dir: 输出目录。
        max_points: 最大绘制点数，避免生成超大图片。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433
    except ModuleNotFoundError:
        print("[warn] matplotlib 不可用，跳过 BEV 可视化")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    half = float(plot_window_m) * 0.5
    if half <= 0:
        raise ValueError("--plot-window-m 必须为正数")

    in_view = (np.abs(points_ego0[:, 0]) <= half) & (np.abs(points_ego0[:, 1]) <= half)
    visible = points_ego0[in_view]
    if visible.shape[0] > max_points:
        idx = np.linspace(0, visible.shape[0] - 1, max_points, dtype=np.int64)
        sample = visible[idx]
    else:
        sample = visible

    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in point_cloud_range]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(sample[:, 0], sample[:, 1], s=0.2, c="black", alpha=0.35, linewidths=0)
    ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], c="red", lw=1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xlabel("x (ego0)")
    ax.set_ylabel("y (ego0)")
    ax.set_title(f"Stage3 voxelization sanity (ego0 BEV, window={plot_window_m:.0f}m)")
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "bev_xy.png", dpi=180)
    plt.close(fig)


def main() -> int:
    """脚本入口。"""
    args = build_arg_parser().parse_args()
    cfg = _load_cfg(args.config)
    data_cfg = _get_nested(cfg, "data")
    coord_cfg = _get_nested(cfg, "coord")
    voxel_cfg = _get_nested(cfg, "voxel")

    index_path = args.index_path or str(data_cfg["index_tiny"])
    payload = load_clip_index(index_path)
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("索引缺少 meta 字段")
    data_root = args.data_root or str(meta.get("data_root", ""))
    if not data_root:
        raise ValueError("无法解析 data_root，请使用 --data-root 显式指定")

    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("索引缺少 clips 列表或为空")

    clip_index = int(args.clip_index) if args.clip_index is not None else int(data_cfg.get("clip_index", 0))
    if clip_index < 0 or clip_index >= len(clips):
        raise ValueError("clip_index 超出范围")
    clip = clips[clip_index]
    if not isinstance(clip, dict):
        raise ValueError("clip 结构非法")

    ego0_frame_index = int(args.ego0_frame_index) if args.ego0_frame_index is not None else int(coord_cfg.get("ego0_frame_index", 0))
    ego0_pose = get_lidar_pose(clip, ego0_frame_index)

    gaussians_world = build_initial_gaussians_for_clip_aggregated(
        clip=clip,
        data_root=data_root,
        frame_indices=None,
        view_names=None,
        voxel_size=float(cfg["init"]["downsample"]["voxel_size_m"]) if isinstance(cfg.get("init"), dict) else 0.2,
        knn_k=int(cfg["init"]["scale_knn_k"]) if isinstance(cfg.get("init"), dict) else 3,
        opacity_init=float(cfg["init"]["opacity_init"]) if isinstance(cfg.get("init"), dict) else 0.5,
        random_seed=int(cfg["train"]["seed"]) if isinstance(cfg.get("train"), dict) else 0,
        num_sky_points=int(args.num_sky_points),
        max_gaussians=int(cfg["init"]["max_gaussians"]) if isinstance(cfg.get("init"), dict) else None,
    )
    gaussians_ego0 = _to_ego0(gaussians_world, ego0_pose)
    features = _pack_features(gaussians_ego0)

    voxel_res = voxelize_points_numpy(
        points_xyz=gaussians_ego0.positions,
        features=features,
        point_cloud_range=voxel_cfg["point_cloud_range"],  # type: ignore[arg-type]
        voxel_size=voxel_cfg["voxel_size"],  # type: ignore[arg-type]
    )

    num_points = int(gaussians_ego0.positions.shape[0])
    num_valid = int(np.count_nonzero(voxel_res.valid_mask))
    num_voxels = int(voxel_res.voxel_coords_xyz.shape[0])
    stats: Dict[str, object] = {
        "config": str(Path(args.config)),
        "index_path": str(Path(index_path)),
        "data_root": data_root,
        "clip_index": clip_index,
        "clip_id": clip.get("clip_id"),
        "ego0_frame_index": ego0_frame_index,
        "num_points": num_points,
        "num_valid_points": num_valid,
        "num_voxels": num_voxels,
        "feature_dim": int(features.shape[1]),
        "voxel_shape_xyz": voxel_res.voxel_shape_xyz,
        "valid_ratio": float(num_valid) / float(max(num_points, 1)),
    }
    print(json.dumps(_summarize(stats), ensure_ascii=False, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"[done] wrote: {out_path}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        _save_bev_plot(
            points_ego0=gaussians_ego0.positions,
            point_cloud_range=tuple(voxel_cfg["point_cloud_range"]),  # type: ignore[arg-type]
            out_dir=out_dir,
            max_points=int(args.max_plot_points),
            plot_window_m=float(args.plot_window_m),
        )
        print(f"[done] wrote bev plot: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
