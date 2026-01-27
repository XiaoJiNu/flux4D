#!/usr/bin/env python3
"""PandaSet Lift 对齐可视化脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flux4d.datasets.pandaset_clips import load_clip_index  # noqa: E402
from flux4d.lift.lift_lidar import (  # noqa: E402
    build_camera_views,
    get_lidar_pose,
    get_lidar_timestamp,
    load_lidar_frame,
    project_points_to_camera,
    sample_colors_from_image,
    transform_lidar_to_camera,
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
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index in clip.")
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
        "--output-dir",
        default="assets/vis/lift_alignment",
        help="Output directory for images.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.2,
        help="Voxel size for downsampling.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Max number of points to render.",
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


def _overlay_points(
    image: np.ndarray, uv: np.ndarray, mask: np.ndarray, colors: np.ndarray
) -> np.ndarray:
    """将投影点绘制在图像上。

    Args:
        image: RGB 图像数组。
        uv: 像素坐标数组。
        mask: 有效点掩码。
        colors: 颜色数组。

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

    valid_idx = np.where(mask)[0]
    for idx in valid_idx:
        x = int(round(float(uv[idx, 0])))
        y = int(round(float(uv[idx, 1])))
        x = max(0, min(x, base.shape[1] - 1))
        y = max(0, min(y, base.shape[0] - 1))
        color = tuple((colors[idx] * 255.0).clip(0, 255).astype(np.uint8).tolist())
        draw.point((x, y), fill=color)
    return np.asarray(canvas)


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

    camera_views = build_camera_views(
        clip, args.frame_index, data_root, view_names=[args.camera]
    )
    if not camera_views:
        raise ValueError("未找到相机视图")

    lidar_paths = clip.get("lidar_paths")
    if not isinstance(lidar_paths, list):
        raise ValueError("lidar_paths 格式非法")
    if args.frame_index >= len(lidar_paths):
        raise ValueError("frame_index 超出范围")

    lidar_path = Path(data_root) / lidar_paths[args.frame_index]
    points_lidar, _ = load_lidar_frame(str(lidar_path))
    points_lidar = voxel_downsample_points(points_lidar, args.voxel_size)
    points_lidar = _sample_points(points_lidar, args.max_points)

    lidar_pose = get_lidar_pose(clip, args.frame_index)
    frame_time = get_lidar_timestamp(clip, args.frame_index)
    print(f"frame timestamp: {frame_time}")

    view = camera_views[0]
    points_camera = transform_lidar_to_camera(points_lidar, lidar_pose, view.pose)
    height, width = view.image.shape[:2]
    uv, depth, mask = project_points_to_camera(points_camera, view.intrinsics, (width, height))
    colors = sample_colors_from_image(view.image, uv, mask)

    depth_map = _render_depth_map(uv, depth, mask, (width, height))
    depth_gray = _depth_to_gray(depth_map)
    overlay = _overlay_points(view.image, uv, mask, colors)

    output_dir = Path(args.output_dir)
    _save_image((view.image * 255.0).clip(0, 255).astype(np.uint8), output_dir / "rgb.png")
    _save_image(overlay, output_dir / "rgb_points.png")
    _save_image(depth_gray, output_dir / "depth_gray.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
