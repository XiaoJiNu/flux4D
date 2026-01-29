#!/usr/bin/env python3
"""PandaSet Lift 对齐可视化脚本。"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flux4d.datasets.pandaset_clips import load_clip_index  # noqa: E402
from flux4d.lift.lift_lidar import (  # noqa: E402
    CameraView,
    Pose,
    build_camera_views,
    get_lidar_pose,
    get_lidar_timestamp,
    load_lidar_frame,
    project_points_to_camera,
    sample_colors_from_image,
    transform_lidar_to_camera,
    transform_world_to_ego,
    voxel_downsample_points,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        参数解析器实例。
    """
    parser = argparse.ArgumentParser(description="Visualize Lift alignment on PandaSet.")
    parser.add_argument(
        "--index-path",
        default="data/metadata/pandaset_tiny_clips.pkl",
        help="Path to the clip index PKL.",
    )
    parser.add_argument("--clip-index", type=int, default=0, help="Clip index.")
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index in clip. If omitted, random-samples will be used.",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=3,
        help="Number of random frames to sample when frame-index is not set.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sampling frames.",
    )
    parser.add_argument(
        "--camera",
        default="front_camera",
        help="Camera name for visualization.",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Override data root if different from index meta.",
    )
    parser.add_argument(
        "--lidar-sensor-id",
        type=int,
        default=-1,
        help="LiDAR sensor id: -1 (all), 0 (Pandar64), 1 (PandarGT).",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/vis/lift_alignment",
        help="Output directory for images (organized by clip/frame/camera).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.2,
        help="Voxel size for downsampling.",
    )
    parser.add_argument(
        "--max-points-raw",
        type=int,
        default=200000,
        help="Max number of raw points to render.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Max number of points to render.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=2,
        help="Point radius in pixels for overlay rendering.",
    )
    parser.add_argument(
        "--point-color-mode",
        choices=("from_image", "fixed", "depth"),
        default="depth",
        help="Point color source: from_image, fixed, or depth.",
    )
    parser.add_argument(
        "--point-color",
        default="255,0,0",
        help="Fixed RGB color (0-255) when point-color-mode is fixed, e.g. 255,0,0.",
    )
    parser.add_argument(
        "--overlay-base",
        choices=("rgb", "gray", "dark"),
        default="gray",
        help="Base image for overlay: rgb, gray, or dark.",
    )
    parser.add_argument(
        "--save-pointcloud",
        action="store_true",
        help="Save raw/downsampled point cloud visualizations (BEV in world/ego frames).",
    )
    parser.add_argument(
        "--bev-x-range",
        nargs=2,
        type=float,
        default=(-50.0, 50.0),
        metavar=("X_MIN", "X_MAX"),
        help="BEV x range in meters (min max).",
    )
    parser.add_argument(
        "--bev-y-range",
        nargs=2,
        type=float,
        default=(-50.0, 50.0),
        metavar=("Y_MIN", "Y_MAX"),
        help="BEV y range in meters (min max).",
    )
    parser.add_argument(
        "--bev-size",
        nargs=2,
        type=int,
        default=(800, 800),
        metavar=("WIDTH", "HEIGHT"),
        help="BEV output image size (width height).",
    )
    return parser


def _load_clip(payload: Dict[str, object], clip_index: int) -> Dict[str, object]:
    """从索引载入指定 clip。

    Args:
        payload: 索引内容字典。
        clip_index: clip 索引。

    Returns:
        clip 字典。

    Raises:
        ValueError: 索引格式非法或索引越界。
    """
    clips = payload.get("clips")
    if not isinstance(clips, list):
        raise ValueError("索引缺少 clips 列表")
    if clip_index < 0 or clip_index >= len(clips):
        raise ValueError("clip_index 超出范围")
    clip = clips[clip_index]
    if not isinstance(clip, dict):
        raise ValueError("clip 格式非法")
    return clip


def _resolve_data_root(payload: Dict[str, object], override: str) -> str:
    """解析数据根目录。

    Args:
        payload: 索引内容字典。
        override: 命令行覆盖路径。

    Returns:
        数据根目录字符串。

    Raises:
        ValueError: 索引 meta 格式非法。
    """
    if override:
        return override
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("索引缺少 meta 字段")
    data_root = meta.get("data_root")
    if not isinstance(data_root, str):
        raise ValueError("meta.data_root 格式非法")
    return data_root


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """在点数过大时进行均匀抽样。

    Args:
        points: 点坐标数组。
        max_points: 最大保留点数。

    Returns:
        抽样后的点坐标数组。
    """
    if points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[indices]


def _format_clip_label(clip: Dict[str, object], clip_index: int) -> str:
    """生成用于输出目录的 clip 标识。

    Args:
        clip: clip 元信息条目。
        clip_index: clip 索引。

    Returns:
        clip 标识字符串。
    """
    clip_id = clip.get("clip_id")
    if isinstance(clip_id, str) and clip_id:
        return f"clip_{clip_index:03d}_{clip_id}"
    return f"clip_{clip_index:03d}"


def _select_frame_indices(
    total_frames: int,
    frame_index: Optional[int],
    random_samples: int,
    seed: int,
) -> List[int]:
    """选择需要可视化的帧索引。

    Args:
        total_frames: clip 内总帧数。
        frame_index: 指定帧索引，None 表示使用随机采样。
        random_samples: 随机采样帧数。
        seed: 随机种子。

    Returns:
        帧索引列表。

    Raises:
        ValueError: 帧索引越界。
    """
    if frame_index is not None:
        if frame_index < 0 or frame_index >= total_frames:
            raise ValueError("frame_index 超出范围")
        return [frame_index]
    if random_samples <= 0:
        return [0]
    random.seed(seed)
    sample_count = min(random_samples, total_frames)
    indices = random.sample(range(total_frames), sample_count)
    return sorted(indices)


def _render_depth_map(
    uv: np.ndarray, depth: np.ndarray, mask: np.ndarray, image_size: Tuple[int, int]
) -> np.ndarray:
    """渲染稀疏深度图。

    Args:
        uv: 像素坐标数组。
        depth: 深度数组。
        mask: 有效点掩码。
        image_size: 图像尺寸 (width, height)。

    Returns:
        深度图数组，未命中像素为 inf。
    """
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


def _depth_to_gray(depth_map: np.ndarray) -> np.ndarray:
    """将深度图转换为灰度图。

    Args:
        depth_map: 深度图数组。

    Returns:
        灰度图数组（uint8）。
    """
    valid = np.isfinite(depth_map)
    if not np.any(valid):
        return np.zeros_like(depth_map, dtype=np.uint8)
    depth_valid = depth_map[valid]
    d_min = float(depth_valid.min())
    d_max = float(depth_valid.max())
    if d_max <= d_min:
        return np.zeros_like(depth_map, dtype=np.uint8)
    norm = (depth_map - d_min) / (d_max - d_min)
    gray = (1.0 - np.clip(norm, 0.0, 1.0)) * 255.0
    gray[~valid] = 0.0
    return gray.astype(np.uint8)


def _save_image(array: np.ndarray, path: Path) -> None:
    """保存图像数组到磁盘。

    Args:
        array: 图像数组。
        path: 输出路径。

    Raises:
        ModuleNotFoundError: Pillow 未安装。
    """
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少 Pillow，请先安装依赖") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        image = Image.fromarray(array)
    else:
        image = Image.fromarray(array)
    image.save(path)


def _parse_rgb_color(text: str) -> np.ndarray:
    """解析 RGB 颜色字符串为 0~1 浮点数组。

    Args:
        text: 颜色字符串，格式为 "R,G,B"，范围 0~255。

    Returns:
        归一化 RGB 颜色数组，形状为 (3,)。

    Raises:
        ValueError: 颜色格式非法。
    """
    parts = [item.strip() for item in text.split(",")]
    if len(parts) != 3:
        raise ValueError("point-color 必须为 R,G,B 格式")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("point-color 必须为 0~255 整数") from exc
    if any(value < 0 or value > 255 for value in values):
        raise ValueError("point-color 数值需在 0~255 范围内")
    return np.array(values, dtype=np.float32) / 255.0


def _build_overlay_base(image: np.ndarray, mode: str) -> np.ndarray:
    """生成用于叠加点的底图。

    Args:
        image: RGB 图像数组，值域 0~1。
        mode: 底图模式，支持 rgb/gray/dark。

    Returns:
        处理后的 RGB 图像数组。
    """
    if mode == "rgb":
        return image
    gray = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(
        np.float32
    )
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    if mode == "dark":
        return (gray_rgb * 0.35).astype(np.float32)
    return gray_rgb


def _colorize_by_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """根据深度生成伪彩色点。

    Args:
        depth: 深度数组，形状为 (N,)。
        mask: 有效点掩码，形状为 (N,)。

    Returns:
        颜色数组，形状为 (N, 3)。
    """
    colors = np.zeros((depth.shape[0], 3), dtype=np.float32)
    valid_idx = np.where(mask)[0]
    if valid_idx.size == 0:
        return colors
    depth_valid = depth[valid_idx]
    d_min = float(depth_valid.min())
    d_max = float(depth_valid.max())
    if d_max <= d_min:
        return colors
    t = (depth_valid - d_min) / (d_max - d_min)
    t = np.clip(1.0 - t, 0.0, 1.0)
    seg = t * 4.0

    colors_valid = np.zeros((valid_idx.size, 3), dtype=np.float32)
    idx0 = seg < 1.0
    idx1 = (seg >= 1.0) & (seg < 2.0)
    idx2 = (seg >= 2.0) & (seg < 3.0)
    idx3 = seg >= 3.0

    colors_valid[idx0, 2] = 1.0
    colors_valid[idx0, 1] = seg[idx0]

    colors_valid[idx1, 2] = 2.0 - seg[idx1]
    colors_valid[idx1, 1] = 1.0

    colors_valid[idx2, 0] = seg[idx2] - 2.0
    colors_valid[idx2, 1] = 1.0

    colors_valid[idx3, 0] = 1.0
    colors_valid[idx3, 1] = 4.0 - seg[idx3]

    colors[valid_idx] = colors_valid
    return colors


def _render_bev_height_map(
    points_xyz: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """渲染点云的 BEV 高度图（每个像素取最大高度）。

    Args:
        points_xyz: 点坐标数组，形状为 (N, 3)。
        x_range: x 轴范围 (min, max)。
        y_range: y 轴范围 (min, max)。
        image_size: 输出图像尺寸 (width, height)。

    Returns:
        RGB 图像数组（uint8），无点区域为黑色。
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz 形状必须为 (N, 3)")

    x_min, x_max = x_range
    y_min, y_max = y_range
    width, height = image_size

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    if not np.any(mask):
        return np.zeros((height, width, 3), dtype=np.uint8)

    x_sel = x[mask]
    y_sel = y[mask]
    z_sel = z[mask]

    cols = np.floor((x_sel - x_min) / (x_max - x_min) * (width - 1)).astype(np.int64)
    rows = np.floor((y_max - y_sel) / (y_max - y_min) * (height - 1)).astype(np.int64)
    cols = np.clip(cols, 0, width - 1)
    rows = np.clip(rows, 0, height - 1)

    height_map = np.full((height, width), -np.inf, dtype=np.float32)
    np.maximum.at(height_map, (rows, cols), z_sel.astype(np.float32))

    valid = np.isfinite(height_map)
    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.uint8)
    values = height_map[valid]
    z_min = float(values.min())
    z_max = float(values.max())
    if z_max <= z_min:
        z_max = z_min + 1e-3
    norm = (height_map - z_min) / (z_max - z_min)
    norm = np.clip(norm, 0.0, 1.0)
    norm[~valid] = 0.0

    from matplotlib import cm

    rgb = (cm.viridis(norm)[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _save_bev(
    points_xyz: np.ndarray,
    output_dir: Path,
    filename: str,
    title: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    image_size: Tuple[int, int],
) -> None:
    """保存点云 BEV 可视化图像。

    Args:
        points_xyz: 点坐标数组。
        output_dir: 输出目录。
        filename: 文件名（.png）。
        title: 标题文本（写入到图像上方）。
        x_range: x 轴范围。
        y_range: y 轴范围。
        image_size: 图像尺寸。
    """
    bev = _render_bev_height_map(points_xyz, x_range, y_range, image_size)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少 Pillow，请先安装依赖") from exc

    canvas = Image.fromarray(bev)
    draw = ImageDraw.Draw(canvas)
    text = f"{title}  x={x_range[0]:.1f}..{x_range[1]:.1f}  y={y_range[0]:.1f}..{y_range[1]:.1f}"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle((0, 0, canvas.size[0], 18), fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)
    _save_image(np.asarray(canvas), output_dir / filename)


def _resolve_point_colors(
    mode: str,
    fixed_color: Optional[np.ndarray],
    image: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """根据模式选择点颜色。

    Args:
        mode: 颜色模式。
        fixed_color: 固定颜色数组（0~1），仅 fixed 模式使用。
        image: RGB 图像数组。
        uv: 像素坐标数组。
        depth: 深度数组。
        mask: 有效点掩码。

    Returns:
        颜色数组。

    Raises:
        ValueError: 固定颜色为空。
    """
    if mode == "from_image":
        return sample_colors_from_image(image, uv, mask)
    if mode == "fixed":
        if fixed_color is None:
            raise ValueError("fixed 颜色模式缺少颜色配置")
        return np.tile(fixed_color[None, :], (uv.shape[0], 1)).astype(np.float32)
    return _colorize_by_depth(depth, mask)


def _summarize_projection(
    points_camera: np.ndarray, uv: np.ndarray, mask: np.ndarray, image_size: Tuple[int, int]
) -> Dict[str, float]:
    """统计投影有效性与覆盖情况。

    Args:
        points_camera: 相机坐标系点坐标。
        uv: 像素坐标数组。
        mask: 有效点掩码。
        image_size: 图像尺寸 (width, height)。

    Returns:
        统计信息字典。
    """
    total = float(mask.shape[0])
    valid = float(mask.sum())
    z_positive = float((points_camera[:, 2] > 0).sum())
    width, height = image_size
    coverage = 0.0
    if valid > 0:
        uv_valid = uv[mask]
        u = np.clip(np.round(uv_valid[:, 0]).astype(np.int64), 0, width - 1)
        v = np.clip(np.round(uv_valid[:, 1]).astype(np.int64), 0, height - 1)
        uv_int = np.stack([u, v], axis=1)
        unique_pixels = float(np.unique(uv_int, axis=0).shape[0])
        coverage = unique_pixels / float(width * height)
    return {
        "total": total,
        "valid": valid,
        "valid_ratio": valid / total if total > 0 else 0.0,
        "z_positive_ratio": z_positive / total if total > 0 else 0.0,
        "pixel_coverage": coverage,
    }


def _compute_depth_stats(
    depth: np.ndarray, mask: np.ndarray
) -> Optional[Tuple[float, float, float]]:
    """统计有效深度的最小/最大/均值。

    Args:
        depth: 深度数组。
        mask: 有效点掩码。

    Returns:
        (min, max, mean) 元组，若无有效点则返回 None。
    """
    if not np.any(mask):
        return None
    depth_valid = depth[mask]
    return float(depth_valid.min()), float(depth_valid.max()), float(depth_valid.mean())


def _print_projection_report(
    label: str,
    stats: Dict[str, float],
    depth_stats: Optional[Tuple[float, float, float]],
) -> None:
    """打印投影统计信息。

    Args:
        label: 报告标签（raw/lift）。
        stats: 投影统计信息。
        depth_stats: 深度统计信息。
    """
    print(
        f"[{label}] projection:",
        f"valid={int(stats['valid'])}/{int(stats['total'])}",
        f"ratio={stats['valid_ratio']:.3f}",
        f"z>0={stats['z_positive_ratio']:.3f}",
        f"pixel_coverage={stats['pixel_coverage']:.6f}",
    )
    if depth_stats is not None:
        print(
            f"[{label}] depth stats:",
            f"min={depth_stats[0]:.3f}",
            f"max={depth_stats[1]:.3f}",
            f"mean={depth_stats[2]:.3f}",
        )


def _overlay_points(
    image: np.ndarray,
    uv: np.ndarray,
    mask: np.ndarray,
    colors: np.ndarray,
    radius: int,
) -> np.ndarray:
    """将投影点绘制在图像上。

    Args:
        image: RGB 图像数组。
        uv: 像素坐标数组。
        mask: 有效点掩码。
        colors: 颜色数组。
        radius: 点半径（像素）。

    Returns:
        绘制后的 RGB 图像数组。

    Raises:
        ModuleNotFoundError: Pillow 未安装。
    """
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少 Pillow，请先安装依赖") from exc

    base = (image * 255.0).clip(0, 255).astype(np.uint8)
    canvas = Image.fromarray(base)
    draw = ImageDraw.Draw(canvas)

    radius = max(1, int(radius))
    valid_idx = np.where(mask)[0]
    for idx in valid_idx:
        x = int(round(float(uv[idx, 0])))
        y = int(round(float(uv[idx, 1])))
        x = max(0, min(x, base.shape[1] - 1))
        y = max(0, min(y, base.shape[0] - 1))
        color = tuple((colors[idx] * 255.0).clip(0, 255).astype(np.uint8).tolist())
        if radius <= 1:
            draw.point((x, y), fill=color)
        else:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=None,
            )
    return np.asarray(canvas)


def _render_projection_outputs(
    view: CameraView,
    points_lidar: np.ndarray,
    lidar_pose: Pose,
    base_image: np.ndarray,
    output_dir: Path,
    label: str,
    point_color_mode: str,
    fixed_color: Optional[np.ndarray],
    point_radius: int,
) -> None:
    """渲染并保存点云投影叠加图与深度图。

    Args:
        view: 相机视图。
        points_lidar: LiDAR 点坐标。
        lidar_pose: LiDAR 位姿。
        base_image: 叠加底图。
        output_dir: 输出目录。
        label: 输出标签（raw/lift）。
        point_color_mode: 点颜色模式。
        fixed_color: 固定颜色数组。
        point_radius: 点半径（像素）。
    """
    points_camera = transform_lidar_to_camera(points_lidar, lidar_pose, view.pose)
    height, width = view.image.shape[:2]
    uv, depth, mask = project_points_to_camera(points_camera, view.intrinsics, (width, height))
    colors = _resolve_point_colors(
        point_color_mode,
        fixed_color,
        view.image,
        uv,
        depth,
        mask,
    )

    depth_map = _render_depth_map(uv, depth, mask, (width, height))
    depth_gray = _depth_to_gray(depth_map)
    overlay = _overlay_points(base_image, uv, mask, colors, point_radius)

    _save_image(overlay, output_dir / f"rgb_points_{label}.png")
    _save_image(depth_gray, output_dir / f"depth_gray_{label}.png")

    stats = _summarize_projection(points_camera, uv, mask, (width, height))
    depth_stats = _compute_depth_stats(depth, mask)
    _print_projection_report(label, stats, depth_stats)


def main() -> int:
    """脚本入口：生成投影点与深度图的可视化输出。

    Returns:
        进程退出码，0 表示成功。

    Raises:
        ValueError: 索引内容格式非法或参数越界。
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    payload = load_clip_index(args.index_path)
    if not isinstance(payload, dict):
        raise ValueError("索引内容格式非法")

    data_root = _resolve_data_root(payload, args.data_root)
    clip = _load_clip(payload, args.clip_index)

    lidar_paths = clip.get("lidar_paths")
    if not isinstance(lidar_paths, list):
        raise ValueError("lidar_paths 格式非法")
    total_frames = len(lidar_paths)
    frame_indices = _select_frame_indices(
        total_frames, args.frame_index, args.random_samples, args.random_seed
    )
    clip_label = _format_clip_label(clip, args.clip_index)
    print(f"clip={clip_label} camera={args.camera} frames={frame_indices} total={total_frames}")

    fixed_color = None
    if args.point_color_mode == "fixed":
        fixed_color = _parse_rgb_color(args.point_color)

    bev_x_range = (float(args.bev_x_range[0]), float(args.bev_x_range[1]))
    bev_y_range = (float(args.bev_y_range[0]), float(args.bev_y_range[1]))
    bev_size = (int(args.bev_size[0]), int(args.bev_size[1]))

    for frame_index in frame_indices:
        camera_views = build_camera_views(clip, frame_index, data_root, view_names=[args.camera])
        if not camera_views:
            raise ValueError("未找到相机视图")

        view = camera_views[0]
        lidar_path = Path(data_root) / lidar_paths[frame_index]
        points_lidar_raw, _ = load_lidar_frame(str(lidar_path), sensor_id=args.lidar_sensor_id)
        points_lidar_raw_vis = _sample_points(points_lidar_raw, args.max_points_raw)
        points_lidar_lift = voxel_downsample_points(points_lidar_raw, args.voxel_size)
        points_lidar_lift = _sample_points(points_lidar_lift, args.max_points)

        lidar_pose = get_lidar_pose(clip, frame_index)
        frame_time = get_lidar_timestamp(clip, frame_index)
        print(f"frame={frame_index:03d} timestamp={frame_time}")
        print(
            "points raw/downsampled:",
            points_lidar_raw.shape[0],
            points_lidar_lift.shape[0],
        )

        output_dir = (
            Path(args.output_dir)
            / clip_label
            / f"camera_{args.camera}"
            / f"frame_{frame_index:03d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_image(
            (view.image * 255.0).clip(0, 255).astype(np.uint8),
            output_dir / "rgb.png",
        )
        base = _build_overlay_base(view.image, args.overlay_base)
        _render_projection_outputs(
            view=view,
            points_lidar=points_lidar_raw_vis,
            lidar_pose=lidar_pose,
            base_image=base,
            output_dir=output_dir,
            label="raw",
            point_color_mode=args.point_color_mode,
            fixed_color=fixed_color,
            point_radius=args.point_radius,
        )
        _render_projection_outputs(
            view=view,
            points_lidar=points_lidar_lift,
            lidar_pose=lidar_pose,
            base_image=base,
            output_dir=output_dir,
            label="lift",
            point_color_mode=args.point_color_mode,
            fixed_color=fixed_color,
            point_radius=args.point_radius,
        )

        if args.save_pointcloud:
            # PandaSet 点云默认在 world 坐标系，为了更直观展示 ego 系下的分布，这里同时保存 world/ego 两种 BEV。
            points_world_raw = points_lidar_raw_vis
            points_world_lift = points_lidar_lift
            points_ego_raw = transform_world_to_ego(points_world_raw, lidar_pose)
            points_ego_lift = transform_world_to_ego(points_world_lift, lidar_pose)

            _save_bev(
                points_world_raw,
                output_dir,
                "bev_world_raw.png",
                "BEV(world)-raw",
                bev_x_range,
                bev_y_range,
                bev_size,
            )
            _save_bev(
                points_world_lift,
                output_dir,
                "bev_world_lift.png",
                "BEV(world)-lift",
                bev_x_range,
                bev_y_range,
                bev_size,
            )
            _save_bev(
                points_ego_raw,
                output_dir,
                "bev_ego_raw.png",
                "BEV(ego)-raw",
                bev_x_range,
                bev_y_range,
                bev_size,
            )
            _save_bev(
                points_ego_lift,
                output_dir,
                "bev_ego_lift.png",
                "BEV(ego)-lift",
                bev_x_range,
                bev_y_range,
                bev_size,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
