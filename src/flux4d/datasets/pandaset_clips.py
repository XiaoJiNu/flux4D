"""PandaSet clip 索引构建与加载工具。"""

from __future__ import annotations

import json
import pickle
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_frame_index(filename: str) -> Optional[int]:
    """从文件名解析帧序号。

    Args:
        filename (str): 帧文件名，支持 .pkl.gz/.jpg/.png 后缀。

    Returns:
        Optional[int]: 解析到的整数帧号，失败则返回 None。

    Raises:
        无。

    实现要点:
        - 先剥离已知后缀，再尝试转为整数。
    """
    name = filename
    for ext in (".pkl.gz", ".jpg", ".png"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    try:
        return int(name)
    except ValueError:
        return None


def _sorted_files(folder: Path, pattern: str) -> List[Path]:
    """按帧序号对目录内文件排序。

    Args:
        folder (Path): 目标目录路径。
        pattern (str): glob 匹配模式。

    Returns:
        List[Path]: 排序后的文件列表。

    Raises:
        无。

    实现要点:
        - 优先按数值帧号排序，无法解析时退化为名称排序。
    """
    files = list(folder.glob(pattern))
    if not files:
        return []

    def sort_key(path: Path) -> Tuple[int, object]:
        index = _parse_frame_index(path.name)
        if index is None:
            return (1, path.name)
        return (0, index)

    return sorted(files, key=sort_key)


def _load_json(path: Path) -> Any:
    """读取 JSON 文件并解析为对象。

    Args:
        path (Path): JSON 文件路径。

    Returns:
        Any: 解析后的 Python 对象。

    Raises:
        FileNotFoundError: 文件不存在。
        json.JSONDecodeError: JSON 格式非法。

    实现要点:
        - 使用 Path.read_text 读取后交给 json.loads。
    """
    return json.loads(path.read_text())


def _infer_fps(timestamps: List[float]) -> Optional[float]:
    """根据时间戳估计帧率。

    Args:
        timestamps (List[float]): 单调递增的时间戳列表（秒）。

    Returns:
        Optional[float]: 估计得到的 FPS，无法估计时返回 None。

    Raises:
        无。

    实现要点:
        - 以相邻差值的中位数作为采样周期。
    """
    if len(timestamps) < 2:
        return None
    diffs = [b - a for a, b in zip(timestamps[:-1], timestamps[1:])]
    median_dt = statistics.median(diffs)
    if median_dt <= 0:
        return None
    return 1.0 / median_dt


def _ensure_len(name: str, values: List[Any], expected: int, strict: bool) -> None:
    """校验列表长度与期望一致。

    Args:
        name (str): 数据名，用于错误信息。
        values (List[Any]): 待校验的列表。
        expected (int): 期望长度。
        strict (bool): 是否严格模式（严格时抛异常）。

    Returns:
        None: 无返回值。

    Raises:
        ValueError: strict=True 且长度不一致时抛出。

    实现要点:
        - 非严格模式仅打印告警，继续执行。
    """
    if len(values) == expected:
        return
    message = f"{name} length {len(values)} != {expected}"
    if strict:
        raise ValueError(message)
    print(f"[warn] {message}")


def _scene_dirs(data_root: Path) -> List[Path]:
    """扫描数据根目录下的场景子目录。

    Args:
        data_root (Path): PandaSet 根目录。

    Returns:
        List[Path]: 以数字命名的场景目录列表，按名称排序。

    Raises:
        无。

    实现要点:
        - 仅收集目录名为数字的条目。
    """
    dirs: List[Path] = []
    for item in data_root.iterdir():
        if not item.is_dir():
            continue
        if item.name.isdigit():
            dirs.append(item)
    return sorted(dirs, key=lambda p: p.name)


def _relative_paths(paths: Iterable[Path], data_root: Path) -> List[str]:
    """将路径列表转换为相对数据根目录的字符串路径。

    Args:
        paths (Iterable[Path]): 路径迭代器。
        data_root (Path): 数据根目录。

    Returns:
        List[str]: 相对路径字符串列表。

    Raises:
        ValueError: 若路径不在 data_root 下。

    实现要点:
        - 统一使用 Path.relative_to 做相对化。
    """
    return [str(path.relative_to(data_root)) for path in paths]


def _slice_list(values: List[Any], start: int, end: int) -> List[Any]:
    """对列表进行切片。

    Args:
        values (List[Any]): 原始列表。
        start (int): 起始索引（含）。
        end (int): 结束索引（不含）。

    Returns:
        List[Any]: 切片结果。

    Raises:
        无。

    实现要点:
        - 直接使用 Python 切片语法。
    """
    return values[start:end]


def _select_val_scenes(scene_ids: List[str], val_count: int) -> List[str]:
    """从场景列表中选择验证集场景。

    Args:
        scene_ids (List[str]): 按字典序排序的场景 ID 列表。
        val_count (int): 需要选取的验证场景数量。

    Returns:
        List[str]: 验证场景 ID 列表。

    Raises:
        无。

    实现要点:
        - 默认取排序后最后 val_count 个场景。
    """
    if val_count <= 0:
        return []
    if len(scene_ids) <= val_count:
        return list(scene_ids)
    return list(scene_ids[-val_count:])


def _build_scene_clips(
    data_root: Path,
    scene_path: Path,
    clip_len_s: float,
    stride_s: float,
    target_fps: Optional[float],
    val_scenes: Optional[Iterable[str]],
    strict: bool,
) -> List[Dict[str, Any]]:
    """为单个场景构建 clip 记录。

    Args:
        data_root (Path): PandaSet 根目录。
        scene_path (Path): 单个场景目录。
        clip_len_s (float): 片段长度（秒）。
        stride_s (float): 滑动步长（秒）。
        target_fps (Optional[float]): 目标帧率，None 则使用实际帧率。
        val_scenes (Optional[Iterable[str]]): 验证场景列表。
        strict (bool): 严格模式（异常即中断）。

    Returns:
        List[Dict[str, Any]]: clip 记录列表。

    Raises:
        FileNotFoundError: 关键文件缺失且 strict=True。
        ValueError: 数据长度不一致或 FPS 偏差过大且 strict=True。

    实现要点:
        - 读取 LiDAR 与各相机的时间戳/位姿/图像路径。
        - 根据目标 FPS 计算窗口帧数并滑动生成 clip。
        - 生成包含路径、内外参与时间戳的索引条目。
    """
    scene_id = scene_path.name
    lidar_dir = scene_path / "lidar"
    camera_root = scene_path / "camera"
    if not lidar_dir.exists() or not camera_root.exists():
        message = f"missing lidar/camera for scene {scene_id}"
        if strict:
            raise FileNotFoundError(message)
        print(f"[warn] {message}")
        return []

    lidar_ts = _load_json(lidar_dir / "timestamps.json")
    lidar_poses = _load_json(lidar_dir / "poses.json")
    lidar_files = _sorted_files(lidar_dir, "*.pkl.gz")
    _ensure_len("lidar timestamps", lidar_ts, len(lidar_files), strict)
    _ensure_len("lidar poses", lidar_poses, len(lidar_files), strict)

    fps_actual = _infer_fps(lidar_ts)
    if fps_actual is None:
        message = f"cannot infer fps for scene {scene_id}"
        if strict:
            raise ValueError(message)
        print(f"[warn] {message}")
        return []

    # 优先使用目标 FPS 来确定窗口大小，便于与论文设置对齐
    fps_target = target_fps if target_fps is not None else fps_actual
    if target_fps is not None and abs(fps_actual - target_fps) > 0.5:
        message = (
            f"scene {scene_id} fps {fps_actual:.3f} deviates from target {target_fps:.3f}"
        )
        if strict:
            raise ValueError(message)
        print(f"[warn] {message}")

    clip_len_frames = max(1, int(round(clip_len_s * fps_target)))
    stride_frames = max(1, int(round(stride_s * fps_target)))
    if len(lidar_files) < clip_len_frames:
        message = (
            f"scene {scene_id} has {len(lidar_files)} frames < clip_len {clip_len_frames}"
        )
        if strict:
            raise ValueError(message)
        print(f"[warn] {message}")
        return []

    cameras: Dict[str, Dict[str, Any]] = {}
    for cam_dir in sorted(camera_root.iterdir()):
        if not cam_dir.is_dir():
            continue
        cam_name = cam_dir.name
        ts_path = cam_dir / "timestamps.json"
        intr_path = cam_dir / "intrinsics.json"
        poses_path = cam_dir / "poses.json"
        if not ts_path.exists() or not intr_path.exists() or not poses_path.exists():
            message = f"missing camera files for {scene_id}/{cam_name}"
            if strict:
                raise FileNotFoundError(message)
            print(f"[warn] {message}")
            continue
        cam_ts = _load_json(ts_path)
        cam_intr = _load_json(intr_path)
        cam_poses = _load_json(poses_path)
        cam_files = _sorted_files(cam_dir, "*.jpg")
        if not cam_files:
            cam_files = _sorted_files(cam_dir, "*.png")
        _ensure_len(f"{cam_name} timestamps", cam_ts, len(cam_files), strict)
        _ensure_len(f"{cam_name} poses", cam_poses, len(cam_files), strict)
        _ensure_len(f"{cam_name} frame count", cam_files, len(lidar_files), strict)
        cameras[cam_name] = {
            "timestamps": cam_ts,
            "intrinsics": cam_intr,
            "extrinsics": cam_poses,
            "image_paths": _relative_paths(cam_files, data_root),
        }

    if not cameras:
        message = f"no cameras found for scene {scene_id}"
        if strict:
            raise ValueError(message)
        print(f"[warn] {message}")
        return []

    view_names = sorted(cameras.keys())
    val_scene_set = set(val_scenes) if val_scenes else set()
    split = "val" if scene_id in val_scene_set else "train"

    entries: List[Dict[str, Any]] = []
    for start in range(0, len(lidar_files) - clip_len_frames + 1, stride_frames):
        end = start + clip_len_frames
        frame_ids = list(range(start, end))
        entry = {
            "clip_id": f"{scene_id}_s{start:03d}_e{end - 1:03d}",
            "scene_id": scene_id,
            "split": split,
            "fps": fps_target,
            "fps_actual": fps_actual,
            "clip_len_s": clip_len_s,
            "stride_s": stride_s,
            "frame_start": start,
            "frame_end": end - 1,
            "frame_ids": frame_ids,
            "timestamps": {
                "lidar": _slice_list(lidar_ts, start, end),
                "camera": {
                    cam: _slice_list(cameras[cam]["timestamps"], start, end)
                    for cam in view_names
                },
            },
            "intrinsics": {cam: cameras[cam]["intrinsics"] for cam in view_names},
            "extrinsics": {
                "lidar": _slice_list(lidar_poses, start, end),
                "camera": {
                    cam: _slice_list(cameras[cam]["extrinsics"], start, end)
                    for cam in view_names
                },
            },
            "image_paths": {
                cam: _slice_list(cameras[cam]["image_paths"], start, end)
                for cam in view_names
            },
            "lidar_paths": _slice_list(
                _relative_paths(lidar_files, data_root), start, end
            ),
            "views": view_names,
        }
        entries.append(entry)
    return entries


def build_pandaset_clip_index(
    data_root: str,
    out_pkl_full: str,
    out_pkl_tiny: str,
    clip_len_s: float = 1.5,
    stride_s: float = 1.5,
    target_fps: Optional[float] = 10.0,
    tiny_scenes: Optional[Iterable[str]] = None,
    tiny_num_scenes: int = 2,
    val_scenes: Optional[Iterable[str]] = None,
    val_num_scenes: int = 10,
    max_scenes: Optional[int] = None,
    strict: bool = True,
) -> Tuple[int, int]:
    """构建 PandaSet 的 full/tiny clip 索引并写入 PKL。

    Args:
        data_root (str): PandaSet 根目录。
        out_pkl_full (str): full 索引输出路径。
        out_pkl_tiny (str): tiny 索引输出路径。
        clip_len_s (float): 片段长度（秒）。
        stride_s (float): 滑动步长（秒）。
        target_fps (Optional[float]): 目标帧率，None 表示使用实际帧率。
        tiny_scenes (Optional[Iterable[str]]): tiny 索引的场景列表。
        tiny_num_scenes (int): tiny 场景数量（仅在 tiny_scenes 为空时生效）。
        val_scenes (Optional[Iterable[str]]): 验证场景列表。
        val_num_scenes (int): 验证场景数量（仅在 val_scenes 为空时生效）。
        max_scenes (Optional[int]): 仅处理前 N 个场景（调试用）。
        strict (bool): 严格模式（异常即中断）。

    Returns:
        Tuple[int, int]: (full_clips, tiny_clips) 的数量统计。

    Raises:
        FileNotFoundError: 关键文件缺失且 strict=True。
        ValueError: 数据不一致或 FPS 偏差过大且 strict=True。

    实现要点:
        - 遍历场景，构建 clip 列表并组装 meta 信息。
        - full/tiny 分别打包为字典后序列化为 PKL。
    """
    data_root_path = Path(data_root)
    scenes = _scene_dirs(data_root_path)
    if max_scenes is not None:
        scenes = scenes[: max_scenes]
    scene_ids = [scene.name for scene in scenes]
    if tiny_scenes is None:
        tiny_scenes = scene_ids[:tiny_num_scenes]
    tiny_set = set(tiny_scenes)
    if val_scenes is None:
        # 默认取排序靠后的场景，保持与论文习惯一致
        val_scenes_list = _select_val_scenes(scene_ids, val_num_scenes)
    else:
        val_scenes_list = list(val_scenes)

    out_full_path = Path(out_pkl_full)
    out_tiny_path = Path(out_pkl_tiny)
    out_full_path.parent.mkdir(parents=True, exist_ok=True)
    out_tiny_path.parent.mkdir(parents=True, exist_ok=True)

    full_count = 0
    tiny_count = 0
    full_entries: List[Dict[str, Any]] = []
    tiny_entries: List[Dict[str, Any]] = []
    for scene_path in scenes:
        entries = _build_scene_clips(
            data_root_path,
            scene_path,
            clip_len_s,
            stride_s,
            target_fps,
            val_scenes_list,
            strict,
        )
        if not entries:
            continue
        full_entries.extend(entries)
        full_count += len(entries)
        if scene_path.name in tiny_set:
            tiny_entries.extend(entries)
            tiny_count += len(entries)

    created_at = datetime.now(timezone.utc).isoformat()
    full_payload = {
        "meta": {
            "data_root": str(data_root_path),
            "clip_len_s": clip_len_s,
            "stride_s": stride_s,
            "target_fps": target_fps,
            "created_at": created_at,
            "scene_ids": scene_ids,
            "val_scenes": val_scenes_list,
            "total_clips": full_count,
        },
        "clips": full_entries,
    }
    tiny_payload = {
        "meta": {
            "data_root": str(data_root_path),
            "clip_len_s": clip_len_s,
            "stride_s": stride_s,
            "target_fps": target_fps,
            "created_at": created_at,
            "scene_ids": sorted(tiny_set),
            "val_scenes": [scene for scene in val_scenes_list if scene in tiny_set],
            "total_clips": tiny_count,
        },
        "clips": tiny_entries,
    }
    with out_full_path.open("wb") as full_fh:
        pickle.dump(full_payload, full_fh)
    with out_tiny_path.open("wb") as tiny_fh:
        pickle.dump(tiny_payload, tiny_fh)
    return full_count, tiny_count


def load_clip_index(path: str) -> Dict[str, Any]:
    """加载 PKL 索引文件。

    Args:
        path (str): PKL 文件路径。

    Returns:
        Dict[str, Any]: 反序列化后的索引字典。

    Raises:
        FileNotFoundError: 文件不存在。
        pickle.UnpicklingError: 反序列化失败。

    实现要点:
        - 直接使用 pickle.load 反序列化。
    """
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def load_clip(meta_entry: Dict[str, Any], data_root: str) -> Dict[str, Any]:
    """将索引条目中的相对路径展开为绝对路径。

    Args:
        meta_entry (Dict[str, Any]): clip 元信息条目。
        data_root (str): PandaSet 根目录。

    Returns:
        Dict[str, Any]: 带绝对路径的 clip 字典。

    Raises:
        无。

    实现要点:
        - 对 lidar/image 路径拼接 data_root。
    """
    data_root_path = Path(data_root)
    clip = dict(meta_entry)
    clip["lidar_paths"] = [str(data_root_path / p) for p in clip["lidar_paths"]]
    clip["image_paths"] = {
        cam: [str(data_root_path / p) for p in paths]
        for cam, paths in clip["image_paths"].items()
    }
    return clip


def sample_views_for_train(
    clip: Dict[str, Any],
    num_views: Optional[int],
    strategy: str = "all",
) -> List[str]:
    """从 clip 中选择训练视角。

    Args:
        clip (Dict[str, Any]): clip 元信息。
        num_views (Optional[int]): 需要的视角数量，None 表示全部。
        strategy (str): 采样策略，支持 "all" 或 "random"。

    Returns:
        List[str]: 选择后的相机视角名称列表。

    Raises:
        ValueError: strategy 不支持时抛出。

    实现要点:
        - num_views 超过可用视角时退化为全部视角。
    """
    views = list(clip.get("views", []))
    if not views:
        return []
    if num_views is None or num_views >= len(views) or strategy == "all":
        return views
    if strategy == "random":
        return random.sample(views, num_views)
    raise ValueError(f"unsupported strategy: {strategy}")
