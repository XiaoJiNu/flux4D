"""PandaSet LiDAR 到高斯初始化的基础实现。"""

from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from typing import TypedDict

import numpy as np


class Vector3(TypedDict):
    """三维向量结构。"""

    x: float
    y: float
    z: float


class Quaternion(TypedDict):
    """四元数结构（w, x, y, z）。"""

    w: float
    x: float
    y: float
    z: float


class Pose(TypedDict):
    """位姿结构，包含位置与朝向。"""

    position: Vector3
    heading: Quaternion


class Intrinsics(TypedDict):
    """相机内参结构。"""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraView:
    """相机视图数据。"""

    name: str
    intrinsics: Intrinsics
    pose: Pose
    image: np.ndarray


@dataclass(frozen=True)
class GaussianSet:
    """高斯集合，用于后续 3DGS 初始化。"""

    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    timestamps: np.ndarray
    velocities: np.ndarray


def load_lidar_frame(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 LiDAR 帧并返回点坐标与强度。

    Args:
        path: LiDAR 帧文件路径（.pkl.gz）。

    Returns:
        点坐标数组与强度数组。

    Raises:
        ModuleNotFoundError: pandas 未安装。
        FileNotFoundError: LiDAR 文件不存在。
        ValueError: 文件内容格式非法或缺失必要字段。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少 pandas，请先安装依赖") from exc

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"LiDAR 文件不存在: {path_obj}")

    with gzip.open(path_obj, "rb") as fh:
        frame = pickle.load(fh)

    if not isinstance(frame, pd.DataFrame):
        raise ValueError("LiDAR 文件内容不是 pandas.DataFrame")
    for key in ("x", "y", "z"):
        if key not in frame.columns:
            raise ValueError(f"LiDAR 缺少字段: {key}")

    points = frame[["x", "y", "z"]].to_numpy(dtype=np.float32)
    if "i" in frame.columns:
        intensity = frame["i"].to_numpy(dtype=np.float32)
    else:
        intensity = np.zeros((points.shape[0],), dtype=np.float32)
    return points, intensity


def load_image_rgb(path: str) -> np.ndarray:
    """读取图像并返回 RGB 数组（0~1）。

    Args:
        path: 图像文件路径。

    Returns:
        RGB 图像数组，dtype 为 float32。

    Raises:
        ModuleNotFoundError: Pillow 未安装。
        FileNotFoundError: 图像文件不存在。
    """
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少 Pillow，请先安装依赖") from exc

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"图像文件不存在: {path_obj}")

    image = Image.open(path_obj).convert("RGB")
    array = np.asarray(image, dtype=np.float32)
    return array / 255.0


def quat_to_rot_matrix(quat: Quaternion) -> np.ndarray:
    """将四元数转换为旋转矩阵。

    Args:
        quat: 四元数（w, x, y, z）。

    Returns:
        3x3 旋转矩阵。

    Raises:
        ValueError: 四元数范数为 0。
    """
    qw = float(quat["w"])
    qx = float(quat["x"])
    qy = float(quat["y"])
    qz = float(quat["z"])
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("四元数范数为 0")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

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


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """将位姿转换为 4x4 变换矩阵。

    Args:
        pose: 位姿结构。

    Returns:
        4x4 变换矩阵，表示从传感器坐标到世界坐标。

    Note:
        默认按右乘模型构建，即 p_world = R * p_sensor + t。
    """
    rot = quat_to_rot_matrix(pose["heading"])
    pos = pose["position"]
    trans = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float32)

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rot
    matrix[:3, 3] = trans
    return matrix


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """使用 4x4 变换矩阵变换点坐标。

    Args:
        points: 点坐标数组，形状为 (N, 3)。
        transform: 4x4 变换矩阵。

    Returns:
        变换后的点坐标数组。

    Raises:
        ValueError: 输入形状非法。
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 形状必须为 (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("transform 形状必须为 (4, 4)")

    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    homo = np.concatenate([points, ones], axis=1)
    transformed = (transform @ homo.T).T
    return transformed[:, :3]


def transform_lidar_to_camera(
    points_lidar: np.ndarray, lidar_pose: Pose, camera_pose: Pose
) -> np.ndarray:
    """将 LiDAR 点从 LiDAR 坐标系转换到相机坐标系。

    Args:
        points_lidar: LiDAR 点坐标，形状为 (N, 3)。
        lidar_pose: LiDAR 位姿。
        camera_pose: 相机位姿。

    Returns:
        相机坐标系下的点坐标。

    Note:
        使用 lidar->world 和 world->camera 的组合变换。
    """
    lidar_to_world = pose_to_matrix(lidar_pose)
    camera_to_world = pose_to_matrix(camera_pose)
    world_to_camera = np.linalg.inv(camera_to_world)
    points_world = transform_points(points_lidar, lidar_to_world)
    return transform_points(points_world, world_to_camera)


def voxel_downsample_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """对点云进行体素下采样。

    Args:
        points: 点坐标数组，形状为 (N, 3)。
        voxel_size: 体素尺寸（米）。

    Returns:
        下采样后的点坐标数组。

    Raises:
        ValueError: voxel_size 非正数或点坐标形状非法。

    Note:
        使用体素内均值作为代表点，便于稳定统计尺度。
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size 必须为正数")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 形状必须为 (N, 3)")

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    voxel_map: Dict[Tuple[int, int, int], np.ndarray] = {}
    counts: Dict[Tuple[int, int, int], int] = {}

    for idx, voxel in enumerate(voxel_indices):
        key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        if key not in voxel_map:
            voxel_map[key] = points[idx].astype(np.float32)
            counts[key] = 1
        else:
            voxel_map[key] += points[idx]
            counts[key] += 1

    downsampled = []
    for key, accum in voxel_map.items():
        count = counts[key]
        downsampled.append(accum / float(count))
    return np.stack(downsampled, axis=0)


def compute_knn_mean_distance(
    points: np.ndarray, k: int, max_bruteforce_points: int = 2000
) -> np.ndarray:
    """计算每个点的 k 近邻平均距离。

    Args:
        points: 点坐标数组，形状为 (N, 3)。
        k: 近邻数量（不包含自身）。
        max_bruteforce_points: 允许使用暴力计算的最大点数。

    Returns:
        每个点的 k 近邻平均距离。

    Raises:
        ValueError: k 非法或点云为空。
        ModuleNotFoundError: 点数过大且缺少 scipy。

    Note:
        优先使用 scipy 的 KDTree，加速大规模点云计算。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 形状必须为 (N, 3)")
    if points.shape[0] == 0:
        raise ValueError("points 不能为空")

    num_points = points.shape[0]
    if num_points <= k:
        return np.full((num_points,), 0.0, dtype=np.float32)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, _ = tree.query(points, k=k + 1)
        mean_dist = distances[:, 1:].mean(axis=1)
        return mean_dist.astype(np.float32)
    except ModuleNotFoundError as exc:
        if num_points > max_bruteforce_points:
            raise ModuleNotFoundError("缺少 scipy，且点数过大无法暴力计算") from exc

    # 点数较小时使用暴力计算，避免额外依赖
    diffs = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    distances.sort(axis=1)
    mean_dist = distances[:, 1 : k + 1].mean(axis=1)
    return mean_dist.astype(np.float32)


def project_points_to_camera(
    points_camera: np.ndarray,
    intrinsics: Intrinsics,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将相机坐标系下的点投影到像素平面。

    Args:
        points_camera: 相机坐标系点坐标，形状为 (N, 3)。
        intrinsics: 相机内参。
        image_size: 图像尺寸 (width, height)。

    Returns:
        像素坐标数组、深度数组、有效点掩码。

    Raises:
        ValueError: 输入形状非法。
    """
    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError("points_camera 形状必须为 (N, 3)")

    width, height = image_size
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2]
    valid_depth = z > 0

    u = x / z * fx + cx
    v = y / z * fy + cy
    mask = (
        valid_depth
        & (u >= 0)
        & (u < float(width))
        & (v >= 0)
        & (v < float(height))
    )

    uv = np.stack([u, v], axis=1)
    return uv.astype(np.float32), z.astype(np.float32), mask


def sample_colors_from_image(
    image: np.ndarray, uv: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """根据像素坐标采样颜色。

    Args:
        image: RGB 图像数组，形状为 (H, W, 3)。
        uv: 像素坐标数组，形状为 (N, 2)。
        mask: 有效点掩码，形状为 (N,)。

    Returns:
        采样得到的颜色数组，形状为 (N, 3)。

    Raises:
        ValueError: 输入形状非法。
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image 形状必须为 (H, W, 3)")
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv 形状必须为 (N, 2)")
    if mask.shape[0] != uv.shape[0]:
        raise ValueError("mask 长度必须与 uv 一致")

    height, width = image.shape[:2]
    u = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, width - 1)
    v = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, height - 1)
    colors = np.zeros((uv.shape[0], 3), dtype=np.float32)
    valid_idx = np.where(mask)[0]
    if valid_idx.size > 0:
        colors[valid_idx] = image[v[valid_idx], u[valid_idx]]
    return colors


def build_camera_views(
    clip: Dict[str, object],
    frame_index: int,
    data_root: str,
    view_names: Optional[Sequence[str]] = None,
) -> List[CameraView]:
    """构建指定帧的相机视图列表。

    Args:
        clip: clip 元信息条目。
        frame_index: clip 内帧索引。
        data_root: PandaSet 根目录。
        view_names: 指定的相机名称列表，None 表示使用 clip 内全部视角。

    Returns:
        相机视图列表。

    Raises:
        ValueError: clip 字段格式非法或索引越界。
    """
    image_paths = clip.get("image_paths")
    intrinsics_map = clip.get("intrinsics")
    extrinsics = clip.get("extrinsics")
    views = clip.get("views")

    if not isinstance(image_paths, dict):
        raise ValueError("image_paths 格式非法")
    if not isinstance(intrinsics_map, dict):
        raise ValueError("intrinsics 格式非法")
    if not isinstance(extrinsics, dict):
        raise ValueError("extrinsics 格式非法")
    if not isinstance(views, list):
        raise ValueError("views 格式非法")

    camera_poses = extrinsics.get("camera")
    if not isinstance(camera_poses, dict):
        raise ValueError("extrinsics.camera 格式非法")

    view_list = list(view_names) if view_names is not None else views
    camera_views: List[CameraView] = []

    for view_name in view_list:
        view_paths = image_paths.get(view_name)
        view_intrinsics = intrinsics_map.get(view_name)
        view_poses = camera_poses.get(view_name)
        if not isinstance(view_paths, list):
            raise ValueError(f"image_paths 缺少视角: {view_name}")
        if not isinstance(view_intrinsics, dict):
            raise ValueError(f"intrinsics 缺少视角: {view_name}")
        if not isinstance(view_poses, list):
            raise ValueError(f"extrinsics.camera 缺少视角: {view_name}")
        if frame_index >= len(view_paths) or frame_index >= len(view_poses):
            raise ValueError("frame_index 超出范围")

        image_path = Path(data_root) / view_paths[frame_index]
        image = load_image_rgb(str(image_path))
        pose = view_poses[frame_index]
        if not isinstance(pose, dict):
            raise ValueError(f"相机位姿格式非法: {view_name}")
        camera_views.append(
            CameraView(
                name=view_name,
                intrinsics=view_intrinsics,
                pose=pose,
                image=image,
            )
        )
    return camera_views


def get_lidar_pose(clip: Dict[str, object], frame_index: int) -> Pose:
    """获取指定帧的 LiDAR 位姿。

    Args:
        clip: clip 元信息条目。
        frame_index: clip 内帧索引。

    Returns:
        LiDAR 位姿。

    Raises:
        ValueError: clip 字段格式非法或索引越界。
    """
    extrinsics = clip.get("extrinsics")
    if not isinstance(extrinsics, dict):
        raise ValueError("extrinsics 格式非法")
    lidar_poses = extrinsics.get("lidar")
    if not isinstance(lidar_poses, list):
        raise ValueError("extrinsics.lidar 格式非法")
    if frame_index >= len(lidar_poses):
        raise ValueError("frame_index 超出范围")
    pose = lidar_poses[frame_index]
    if not isinstance(pose, dict):
        raise ValueError("LiDAR 位姿格式非法")
    return pose


def get_lidar_timestamp(clip: Dict[str, object], frame_index: int) -> float:
    """获取指定帧的 LiDAR 时间戳。

    Args:
        clip: clip 元信息条目。
        frame_index: clip 内帧索引。

    Returns:
        LiDAR 时间戳。

    Raises:
        ValueError: clip 字段格式非法或索引越界。
    """
    timestamps = clip.get("timestamps")
    if not isinstance(timestamps, dict):
        raise ValueError("timestamps 格式非法")
    lidar_ts = timestamps.get("lidar")
    if not isinstance(lidar_ts, list):
        raise ValueError("timestamps.lidar 格式非法")
    if frame_index >= len(lidar_ts):
        raise ValueError("frame_index 超出范围")
    value = lidar_ts[frame_index]
    if not isinstance(value, (int, float)):
        raise ValueError("LiDAR 时间戳格式非法")
    return float(value)


def colorize_points_multi_view(
    points_lidar: np.ndarray,
    lidar_pose: Pose,
    camera_views: Sequence[CameraView],
    default_color: Tuple[float, float, float],
) -> np.ndarray:
    """使用多视角为点云赋予颜色。

    Args:
        points_lidar: LiDAR 点坐标，形状为 (N, 3)。
        lidar_pose: LiDAR 位姿。
        camera_views: 相机视图列表。
        default_color: 未命中视角时的默认颜色（0~1）。

    Returns:
        颜色数组，形状为 (N, 3)。
    """
    colors = np.tile(np.array(default_color, dtype=np.float32), (points_lidar.shape[0], 1))
    remaining = np.ones((points_lidar.shape[0],), dtype=bool)

    for view in camera_views:
        points_camera = transform_lidar_to_camera(points_lidar, lidar_pose, view.pose)
        height, width = view.image.shape[:2]
        uv, _, mask = project_points_to_camera(
            points_camera, view.intrinsics, (width, height)
        )
        valid = mask & remaining
        if not np.any(valid):
            continue
        sampled = sample_colors_from_image(view.image, uv, valid)
        colors[valid] = sampled[valid]
        remaining[valid] = False
        if not np.any(remaining):
            break

    return colors


def build_initial_gaussians_for_frame(
    points_lidar: np.ndarray,
    lidar_pose: Pose,
    camera_views: Sequence[CameraView],
    frame_timestamp: float,
    voxel_size: float = 0.2,
    knn_k: int = 8,
    default_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    opacity: float = 0.5,
) -> GaussianSet:
    """基于单帧 LiDAR 构建初始高斯集合。

    Args:
        points_lidar: LiDAR 点坐标，形状为 (N, 3)。
        lidar_pose: LiDAR 位姿。
        camera_views: 相机视图列表。
        frame_timestamp: 帧时间戳。
        voxel_size: 体素下采样尺寸。
        knn_k: kNN 近邻数量。
        default_color: 未命中视角时的默认颜色。
        opacity: 初始不透明度。

    Returns:
        初始高斯集合。
    """
    downsampled = voxel_downsample_points(points_lidar, voxel_size)
    scales_scalar = compute_knn_mean_distance(downsampled, knn_k)
    scales = np.repeat(scales_scalar[:, None], 3, axis=1)

    if camera_views:
        colors = colorize_points_multi_view(
            downsampled, lidar_pose, camera_views, default_color
        )
    else:
        colors = np.tile(np.array(default_color, dtype=np.float32), (downsampled.shape[0], 1))

    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (downsampled.shape[0], 1))
    opacities = np.full((downsampled.shape[0],), opacity, dtype=np.float32)
    timestamps = np.full((downsampled.shape[0],), frame_timestamp, dtype=np.float32)
    velocities = np.zeros((downsampled.shape[0], 3), dtype=np.float32)

    return GaussianSet(
        positions=downsampled,
        scales=scales,
        rotations=rotations,
        colors=colors,
        opacities=opacities,
        timestamps=timestamps,
        velocities=velocities,
    )


def build_initial_gaussians_for_clip(
    clip: Dict[str, object],
    data_root: str,
    frame_indices: Optional[Sequence[int]] = None,
    view_names: Optional[Sequence[str]] = None,
    voxel_size: float = 0.2,
    knn_k: int = 8,
    default_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    opacity: float = 0.5,
) -> List[GaussianSet]:
    """为 clip 内多个帧构建初始高斯集合。

    Args:
        clip: clip 元信息条目。
        data_root: PandaSet 根目录。
        frame_indices: 需要处理的帧索引列表，None 表示全帧。
        view_names: 指定的相机名称列表，None 表示使用 clip 内视角。
        voxel_size: 体素下采样尺寸。
        knn_k: kNN 近邻数量。
        default_color: 未命中视角时的默认颜色。
        opacity: 初始不透明度。

    Returns:
        初始高斯集合列表。

    Raises:
        ValueError: clip 字段格式非法或索引越界。
    """
    lidar_paths = clip.get("lidar_paths")
    if not isinstance(lidar_paths, list):
        raise ValueError("lidar_paths 格式非法")

    total_frames = len(lidar_paths)
    if frame_indices is None:
        frame_list = list(range(total_frames))
    else:
        frame_list = list(frame_indices)

    gaussians: List[GaussianSet] = []
    for frame_index in frame_list:
        if frame_index >= total_frames:
            raise ValueError("frame_index 超出范围")
        lidar_path = Path(data_root) / lidar_paths[frame_index]
        points_lidar, _ = load_lidar_frame(str(lidar_path))
        lidar_pose = get_lidar_pose(clip, frame_index)
        frame_timestamp = get_lidar_timestamp(clip, frame_index)
        camera_views = build_camera_views(clip, frame_index, data_root, view_names)
        gaussians.append(
            build_initial_gaussians_for_frame(
                points_lidar,
                lidar_pose,
                camera_views,
                frame_timestamp,
                voxel_size=voxel_size,
                knn_k=knn_k,
                default_color=default_color,
                opacity=opacity,
            )
        )
    return gaussians
