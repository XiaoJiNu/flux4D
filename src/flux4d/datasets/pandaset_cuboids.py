"""PandaSet cuboid 标注读取与几何工具。

本模块为阶段6评测服务，提供：
- 读取每帧 cuboid 标注（`annotations/cuboids/{frame}.pkl.gz`）；
- 将 cuboid 转换为 SE(3)（仅 yaw）；
- 点到 cuboid 的包含关系判定（point-in-OBB）；
- 点到 cuboid 的唯一归属分配（多重命中时按体积/中心距离打破平局）。

Note:
    PandaSet 的 cuboids 标注以 pandas.DataFrame pickle 存储，因此需要 `pandas` 才能读取。
    若当前 Python 环境无法 import pandas（例如 NumPy ABI 不兼容），请切换到项目约定的
    conda 环境（如 gaussianstorm）。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


_CUBOID_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "uuid",
    "label",
    "yaw",
    "stationary",
    "position.x",
    "position.y",
    "position.z",
    "dimensions.x",
    "dimensions.y",
    "dimensions.z",
)


def _wrap_angle_rad(angle: float) -> float:
    """将角度规整到 [-pi, pi]。

    Args:
        angle: 输入角度（弧度）。

    Returns:
        规整后的角度（弧度）。
    """
    value = float(angle)
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _rotation_z_world_from_obj(yaw_rad: float) -> np.ndarray:
    """构建绕 z 轴的旋转矩阵（object->world）。

    Args:
        yaw_rad: 绕 z 轴旋转角（弧度）。

    Returns:
        旋转矩阵，形状为 (3, 3)，dtype=float32。
    """
    yaw = float(yaw_rad)
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _find_pandaset_frame_pickle(folder: Path, frame_id: int) -> Path:
    """在标注目录中定位帧级 pkl.gz 文件。

    Args:
        folder: `.../annotations/<type>/` 目录。
        frame_id: 帧号（整数）。

    Returns:
        文件路径。

    Raises:
        FileNotFoundError: 未找到对应帧文件。
    """
    candidates = [
        folder / f"{frame_id:02d}.pkl.gz",
        folder / f"{frame_id:03d}.pkl.gz",
        folder / f"{frame_id}.pkl.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"未找到标注帧文件: {folder} frame_id={frame_id}")


@dataclass(frozen=True)
class PandaSetCuboid:
    """PandaSet 单个 cuboid（OBB）标注。"""

    uuid: str
    label: str
    yaw_rad: float
    stationary: bool
    center_world: np.ndarray  # (3,)
    dims_xyz: np.ndarray  # (3,) (x,y,z) full lengths in meters

    @property
    def volume(self) -> float:
        """返回 cuboid 体积。"""
        return float(np.prod(self.dims_xyz))

    @property
    def rotation_world_from_obj(self) -> np.ndarray:
        """返回 object->world 的旋转矩阵（绕 z 轴）。"""
        return _rotation_z_world_from_obj(self.yaw_rad)


def load_pandaset_cuboids_frame(
    data_root: str,
    scene_id: str,
    frame_id: int,
) -> List[PandaSetCuboid]:
    """读取 PandaSet 指定场景/帧的 cuboids 标注。

    Args:
        data_root: PandaSet 根目录。
        scene_id: 场景 ID（如 "001"）。
        frame_id: 场景内帧号（整数）。

    Returns:
        cuboid 列表。

    Raises:
        RuntimeError: pandas 不可用。
        FileNotFoundError: 标注文件不存在。
        ValueError: 标注文件结构/字段不符合预期。
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError(
            "pandas 不可用：请在可用的 Python 环境中运行（例如 conda 环境 gaussianstorm），"
            "或安装与 NumPy 版本兼容的 pandas。"
        ) from exc

    cuboid_dir = Path(data_root) / str(scene_id) / "annotations" / "cuboids"
    path = _find_pandaset_frame_pickle(cuboid_dir, int(frame_id))
    frame = pd.read_pickle(str(path))
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("cuboids 标注文件内容不是 pandas.DataFrame")

    missing = [name for name in _CUBOID_REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"cuboids 缺少必要字段: {missing}")

    cuboids: List[PandaSetCuboid] = []
    for _, row in frame.iterrows():
        uuid = str(row["uuid"])
        label = str(row["label"])
        yaw = float(row["yaw"])
        stationary = bool(row["stationary"])
        center = np.array(
            [float(row["position.x"]), float(row["position.y"]), float(row["position.z"])],
            dtype=np.float32,
        )
        dims = np.array(
            [float(row["dimensions.x"]), float(row["dimensions.y"]), float(row["dimensions.z"])],
            dtype=np.float32,
        )
        cuboids.append(
            PandaSetCuboid(
                uuid=uuid,
                label=label,
                yaw_rad=yaw,
                stationary=stationary,
                center_world=center,
                dims_xyz=dims,
            )
        )
    return cuboids


def index_cuboids_by_uuid(cuboids: Sequence[PandaSetCuboid]) -> Dict[str, PandaSetCuboid]:
    """按 uuid 对 cuboids 建立索引。

    Args:
        cuboids: cuboid 列表。

    Returns:
        uuid -> cuboid 的映射。
    """
    return {cuboid.uuid: cuboid for cuboid in cuboids}


def points_in_cuboid_world(points_world: np.ndarray, cuboid: PandaSetCuboid) -> np.ndarray:
    """判断 world 点是否落在 cuboid（OBB）内部。

    Args:
        points_world: world 坐标系点，形状为 (N, 3)。
        cuboid: cuboid 标注。

    Returns:
        布尔掩码，形状为 (N,)。

    Raises:
        ValueError: 输入形状非法。
    """
    if points_world.ndim != 2 or int(points_world.shape[1]) != 3:
        raise ValueError("points_world 形状必须为 (N, 3)")
    dims = cuboid.dims_xyz.astype(np.float32)
    if np.any(dims <= 0):
        return np.zeros((points_world.shape[0],), dtype=bool)

    rot_world_from_obj = cuboid.rotation_world_from_obj
    half = dims * 0.5
    rel = points_world.astype(np.float32, copy=False) - cuboid.center_world[None, :]
    points_obj = rel @ rot_world_from_obj  # (N, 3) row-vector form => apply R^T
    inside = np.all(np.abs(points_obj) <= half[None, :], axis=1)
    return inside


def assign_points_to_cuboids_world(
    points_world: np.ndarray,
    cuboids: Sequence[PandaSetCuboid],
) -> Tuple[np.ndarray, np.ndarray]:
    """将点分配到唯一 cuboid（若命中多个，按体积/中心距离选优）。

    Args:
        points_world: world 点坐标 (N, 3)。
        cuboids: cuboid 列表。

    Returns:
        (assigned_indices, point_obj_norm)：
        - assigned_indices: (N,) int 数组，值为 cuboid 索引或 -1（背景）。
        - point_obj_norm: (N,) float 数组，点到 cuboid 中心的 obj 坐标系距离（未分配时为 inf）。

    Raises:
        ValueError: 输入形状非法。
    """
    if points_world.ndim != 2 or int(points_world.shape[1]) != 3:
        raise ValueError("points_world 形状必须为 (N, 3)")
    num_points = int(points_world.shape[0])
    assigned = np.full((num_points,), -1, dtype=np.int32)
    best_volume = np.full((num_points,), np.inf, dtype=np.float32)
    best_norm = np.full((num_points,), np.inf, dtype=np.float32)

    points_world_f32 = points_world.astype(np.float32, copy=False)
    for idx, cuboid in enumerate(cuboids):
        dims = cuboid.dims_xyz.astype(np.float32)
        if np.any(dims <= 0):
            continue
        vol = float(np.prod(dims))
        rot = cuboid.rotation_world_from_obj
        rel = points_world_f32 - cuboid.center_world[None, :]
        points_obj = rel @ rot
        inside = np.all(np.abs(points_obj) <= (dims * 0.5)[None, :], axis=1)
        if not np.any(inside):
            continue
        center_norm = np.linalg.norm(points_obj, axis=1).astype(np.float32)
        update = inside & (
            (vol < best_volume)
            | ((np.isclose(vol, best_volume)) & (center_norm < best_norm))
        )
        if np.any(update):
            assigned[update] = int(idx)
            best_volume[update] = float(vol)
            best_norm[update] = center_norm[update]
    return assigned, best_norm


def compute_cuboid_dynamic_flags_by_frame(
    cuboids_by_frame: Mapping[int, Sequence[PandaSetCuboid]],
    *,
    trans_thresh_m: float,
    yaw_thresh_deg: float,
    smoothing_window: int = 1,
) -> Dict[int, Dict[str, bool]]:
    """计算每帧每个 uuid 的动态标记（基于相邻帧姿态差）。

    Args:
        cuboids_by_frame: frame_id -> cuboids 列表。
        trans_thresh_m: 平移阈值（米），超过则认为动态。
        yaw_thresh_deg: yaw 阈值（度），超过则认为动态。
        smoothing_window: 平滑窗口大小（奇数）。=1 表示不平滑。

    Returns:
        frame_id -> {uuid: is_dynamic} 的映射。

    Raises:
        ValueError: 参数非法。

    Note:
        - 优先使用 `frame_id -> frame_id+1` 的差分；若下一帧不存在则退化为 `frame_id-1 -> frame_id`。
        - 平滑使用“多数表决”：窗口内 True 数量 >= ceil(K/2) 判为 True。
    """
    if float(trans_thresh_m) < 0.0:
        raise ValueError("trans_thresh_m 不能为负")
    if float(yaw_thresh_deg) < 0.0:
        raise ValueError("yaw_thresh_deg 不能为负")
    if smoothing_window <= 0 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window 必须为正奇数")

    yaw_thresh_rad = math.radians(float(yaw_thresh_deg))
    frames = sorted(int(fid) for fid in cuboids_by_frame.keys())
    by_uuid: Dict[int, Dict[str, PandaSetCuboid]] = {
        fid: index_cuboids_by_uuid(cuboids_by_frame[fid]) for fid in frames
    }

    raw_flags: Dict[int, Dict[str, bool]] = {}
    for fid in frames:
        current = by_uuid[fid]
        next_fid = fid + 1 if (fid + 1) in by_uuid else (fid - 1 if (fid - 1) in by_uuid else None)
        flags: Dict[str, bool] = {}
        if next_fid is None:
            raw_flags[fid] = {uuid: False for uuid in current.keys()}
            continue
        other = by_uuid[next_fid]
        for uuid, cub in current.items():
            other_cub = other.get(uuid)
            if other_cub is None:
                flags[uuid] = False
                continue
            delta_t = float(np.linalg.norm(other_cub.center_world - cub.center_world))
            delta_yaw = abs(_wrap_angle_rad(float(other_cub.yaw_rad) - float(cub.yaw_rad)))
            flags[uuid] = (delta_t > float(trans_thresh_m)) or (delta_yaw > yaw_thresh_rad)
        raw_flags[fid] = flags

    if smoothing_window == 1:
        return raw_flags

    radius = smoothing_window // 2
    smoothed: Dict[int, Dict[str, bool]] = {}
    for fid in frames:
        flags_for_frame: Dict[str, bool] = {}
        current_uuids = raw_flags[fid].keys()
        for uuid in current_uuids:
            votes: List[bool] = []
            for neighbor in range(fid - radius, fid + radius + 1):
                neighbor_flags = raw_flags.get(neighbor)
                if neighbor_flags is None:
                    continue
                if uuid not in neighbor_flags:
                    continue
                votes.append(bool(neighbor_flags[uuid]))
            if not votes:
                flags_for_frame[uuid] = False
                continue
            threshold = int(math.ceil(len(votes) / 2.0))
            flags_for_frame[uuid] = sum(1 for v in votes if v) >= threshold
        smoothed[fid] = flags_for_frame
    return smoothed

