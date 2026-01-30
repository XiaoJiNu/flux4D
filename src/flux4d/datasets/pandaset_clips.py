"""PandaSet clip 索引构建与加载工具。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import pickle
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ClipEntry = Dict[str, object]

PANDASET_PAPER_TEST_SCENES: Tuple[str, ...] = (
    "001",
    "011",
    "016",
    "065",
    "084",
    "090",
    "106",
    "115",
    "123",
    "158",
)


@dataclass(frozen=True)
class ClipSetting:
    """clip 切片设置（对齐论文/补充材料）。"""

    name: str
    clip_len_frames: int
    stride_frames: int
    input_frame_indices: Tuple[int, ...]
    target_frame_indices: Tuple[int, ...]
    splits: Tuple[str, ...]
    max_clips_per_scene: Optional[int] = None


def build_pandaset_paper_settings() -> List[ClipSetting]:
    """构建论文/补充材料一致的切片设置列表。

    Returns:
        切片设置列表。

    Note:
        - Interpolation training：1s snippet（11 帧）+ 5 帧 overlap → stride=6。
        - Future training：1.5s snippet（16 帧）+ 5 帧 overlap → stride=11。
        - Eval：每 20 帧采样一个 1.5s snippet（16 帧），每个 log 取最多 4 个片段。
    """
    input_frames = (0, 2, 4, 6, 8, 10)
    return [
        ClipSetting(
            name="train_interpolation",
            clip_len_frames=11,
            stride_frames=6,
            input_frame_indices=input_frames,
            target_frame_indices=(1, 3, 5, 7, 9),
            splits=("train",),
        ),
        ClipSetting(
            name="train_future",
            clip_len_frames=16,
            stride_frames=11,
            input_frame_indices=input_frames,
            target_frame_indices=(1, 3, 5, 7, 9, 11, 12, 13, 14, 15),
            splits=("train",),
        ),
        ClipSetting(
            name="eval_future",
            clip_len_frames=16,
            stride_frames=20,
            input_frame_indices=input_frames,
            target_frame_indices=(1, 3, 5, 7, 9, 11, 12, 13, 14, 15),
            splits=("test",),
            max_clips_per_scene=4,
        ),
    ]


def _parse_frame_index(filename: str) -> Optional[int]:
    """从文件名解析帧序号。

    Args:
        filename: 帧文件名，支持 .pkl.gz/.jpg/.png 后缀。

    Returns:
        解析到的整数帧号，失败则返回 None。
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
        folder: 目标目录路径。
        pattern: glob 匹配模式。

    Returns:
        排序后的文件列表。

    Note:
        优先按数值帧号排序，无法解析时退化为名称排序。
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


def _load_json(path: Path) -> object:
    """读取 JSON 文件并解析为对象。

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的 Python 对象。
    """
    return json.loads(path.read_text())


def _infer_fps(timestamps: List[float]) -> Optional[float]:
    """根据时间戳估计帧率。

    Args:
        timestamps: 单调递增的时间戳列表（秒）。

    Returns:
        估计得到的 FPS，无法估计时返回 None。
    """
    if len(timestamps) < 2:
        return None
    diffs = [b - a for a, b in zip(timestamps[:-1], timestamps[1:])]
    median_dt = statistics.median(diffs)
    if median_dt <= 0:
        return None
    return 1.0 / median_dt


def _ensure_len(name: str, values: Sequence[object], expected: int, strict: bool) -> None:
    """校验列表长度与期望一致。

    Args:
        name: 数据名，用于错误信息。
        values: 待校验的序列。
        expected: 期望长度。
        strict: 是否严格模式（严格时抛异常）。

    Raises:
        ValueError: strict=True 且长度不一致时抛出。
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
        data_root: PandaSet 根目录。

    Returns:
        以数字命名的场景目录列表，按名称排序。
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
        paths: 路径迭代器。
        data_root: 数据根目录。

    Returns:
        相对路径字符串列表。
    """
    return [str(path.relative_to(data_root)) for path in paths]


def _slice_list(values: Sequence[object], start: int, end: int) -> List[object]:
    """对序列进行切片并返回列表。

    Args:
        values: 原始序列。
        start: 起始索引（含）。
        end: 结束索引（不含）。

    Returns:
        切片后的列表。
    """
    return list(values[start:end])


def _select_val_scenes(scene_ids: List[str], val_count: int) -> List[str]:
    """从场景列表中选择验证集场景。

    Args:
        scene_ids: 按字典序排序的场景 ID 列表。
        val_count: 需要选取的验证场景数量。

    Returns:
        验证场景 ID 列表。

    Note:
        默认取排序后最后 val_count 个场景。
    """
    if val_count <= 0:
        return []
    if len(scene_ids) <= val_count:
        return list(scene_ids)
    return list(scene_ids[-val_count:])


def _build_scene_clips(
    data_root: Path,
    scene_path: Path,
    clip_settings: Sequence[ClipSetting],
    target_fps: Optional[float],
    val_scenes: Optional[Iterable[str]],
    camera_names: Optional[Sequence[str]],
    strict: bool,
) -> List[ClipEntry]:
    """为单个场景构建 clip 记录。

    Args:
        data_root: PandaSet 根目录。
        scene_path: 单个场景目录。
        clip_settings: clip 切片设置列表。
        target_fps: 目标帧率，None 则使用实际帧率。
        val_scenes: 测试场景列表。
        camera_names: 需要保留的相机列表，None 表示保留全部。
        strict: 严格模式（异常即中断）。

    Returns:
        clip 记录列表。

    Raises:
        FileNotFoundError: 关键文件缺失且 strict=True。
        ValueError: 数据长度不一致或 FPS 偏差过大且 strict=True。
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

    if not isinstance(lidar_ts, list):
        raise ValueError("lidar timestamps 格式非法")

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

    cameras: Dict[str, Dict[str, object]] = {}
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
        if not isinstance(cam_ts, list):
            raise ValueError(f"{cam_name} timestamps 格式非法")
        cameras[cam_name] = {
            "timestamps": cam_ts,
            "intrinsics": cam_intr,
            "extrinsics": cam_poses,
            "image_paths": _relative_paths(cam_files, data_root),
        }

    if camera_names is not None:
        cameras = {name: cameras[name] for name in camera_names if name in cameras}

    if not cameras:
        message = f"no cameras found for scene {scene_id}"
        if strict:
            raise ValueError(message)
        print(f"[warn] {message}")
        return []

    view_names = sorted(cameras.keys())
    val_scene_set = set(val_scenes) if val_scenes else set()
    split = "test" if scene_id in val_scene_set else "train"

    entries: List[ClipEntry] = []
    for setting in clip_settings:
        if split not in setting.splits:
            continue
        if len(lidar_files) < setting.clip_len_frames:
            message = (
                f"scene {scene_id} has {len(lidar_files)} frames < clip_len "
                f"{setting.clip_len_frames} for setting {setting.name}"
            )
            if strict:
                raise ValueError(message)
            print(f"[warn] {message}")
            continue

        count_for_scene = 0
        for start in range(
            0, len(lidar_files) - setting.clip_len_frames + 1, setting.stride_frames
        ):
            if setting.max_clips_per_scene is not None and count_for_scene >= setting.max_clips_per_scene:
                break
            end = start + setting.clip_len_frames
            frame_ids = list(range(start, end))
            entry: ClipEntry = {
                "clip_id": f"{scene_id}_{setting.name}_s{start:03d}_e{end - 1:03d}",
                "scene_id": scene_id,
                "split": split,
                "setting": setting.name,
                "fps": fps_target,
                "fps_actual": fps_actual,
                "clip_len_frames": setting.clip_len_frames,
                "stride_frames": setting.stride_frames,
                "clip_len_s": float(setting.clip_len_frames - 1) / float(fps_target),
                "stride_s": float(setting.stride_frames) / float(fps_target),
                "input_frame_indices": list(setting.input_frame_indices),
                "target_frame_indices": list(setting.target_frame_indices),
                "input_frame_ids": [start + idx for idx in setting.input_frame_indices],
                "target_frame_ids": [start + idx for idx in setting.target_frame_indices],
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
            count_for_scene += 1
    return entries


def build_pandaset_clip_index(
    data_root: str,
    out_pkl_full: str,
    out_pkl_tiny: str,
    preset: str = "paper",
    clip_len_s: float = 1.5,
    stride_s: float = 1.5,
    target_fps: Optional[float] = 10.0,
    tiny_scenes: Optional[Iterable[str]] = None,
    tiny_num_scenes: int = 2,
    val_scenes: Optional[Iterable[str]] = None,
    val_num_scenes: int = 10,
    camera_names: Optional[Sequence[str]] = None,
    clip_settings: Optional[Sequence[ClipSetting]] = None,
    max_scenes: Optional[int] = None,
    strict: bool = True,
) -> Tuple[int, int]:
    """构建 PandaSet 的 full/tiny clip 索引并写入 PKL。

    Args:
        data_root: PandaSet 根目录。
        out_pkl_full: full 索引输出路径。
        out_pkl_tiny: tiny 索引输出路径。
        preset: 切片预设，支持 "paper" 与 "debug"。
        clip_len_s: 片段长度（秒）。
        stride_s: 滑动步长（秒）。
        target_fps: 目标帧率，None 表示使用实际帧率。
        tiny_scenes: tiny 索引的场景列表。
        tiny_num_scenes: tiny 场景数量（仅在 tiny_scenes 为空时生效）。
        val_scenes: 测试场景列表。
        val_num_scenes: 测试场景数量（仅在 val_scenes 为空时生效，仅 debug preset 使用）。
        camera_names: 相机过滤列表，None 表示保留全部（paper preset 默认仅使用 front_camera）。
        clip_settings: 自定义切片设置列表。None 表示使用 preset 的默认设置。
        max_scenes: 仅处理前 N 个场景（调试用）。
        strict: 严格模式（异常即中断）。

    Returns:
        full/tiny clip 的数量统计。
    """
    data_root_path = Path(data_root)
    scenes = _scene_dirs(data_root_path)
    if max_scenes is not None:
        scenes = scenes[: max_scenes]
    scene_ids = [scene.name for scene in scenes]
    if tiny_scenes is None:
        tiny_scenes = scene_ids[:tiny_num_scenes]
    tiny_set = set(tiny_scenes)

    if preset not in ("paper", "debug"):
        raise ValueError(f"unsupported preset: {preset}")

    if preset == "paper":
        if val_scenes is None:
            val_scenes_list = list(PANDASET_PAPER_TEST_SCENES)
        else:
            val_scenes_list = list(val_scenes)
        if camera_names is None:
            camera_names = ["front_camera"]
        if clip_settings is None:
            clip_settings = build_pandaset_paper_settings()
    else:
        if val_scenes is None:
            # 默认取排序靠后的场景，保持与旧实现一致
            val_scenes_list = _select_val_scenes(scene_ids, val_num_scenes)
        else:
            val_scenes_list = list(val_scenes)
        if clip_settings is None:
            # debug：按秒换算成帧数（注意：这与论文的 11/16 帧定义不同）
            fps_for_frames = target_fps if target_fps is not None else 10.0
            clip_len_frames = max(1, int(round(clip_len_s * fps_for_frames)))
            stride_frames = max(1, int(round(stride_s * fps_for_frames)))
            clip_settings = [
                ClipSetting(
                    name="debug",
                    clip_len_frames=clip_len_frames,
                    stride_frames=stride_frames,
                    input_frame_indices=tuple(range(clip_len_frames)),
                    target_frame_indices=tuple(),
                    splits=("train", "test"),
                )
            ]

    out_full_path = Path(out_pkl_full)
    out_tiny_path = Path(out_pkl_tiny)
    out_full_path.parent.mkdir(parents=True, exist_ok=True)
    out_tiny_path.parent.mkdir(parents=True, exist_ok=True)

    full_count = 0
    tiny_count = 0
    full_entries: List[ClipEntry] = []
    tiny_entries: List[ClipEntry] = []
    for scene_path in scenes:
        entries = _build_scene_clips(
            data_root_path,
            scene_path,
            clip_settings,
            target_fps,
            val_scenes_list,
            camera_names,
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
    full_payload: Dict[str, object] = {
        "meta": {
            "data_root": str(data_root_path),
            "preset": preset,
            "target_fps": target_fps,
            "created_at": created_at,
            "scene_ids": scene_ids,
            "test_scenes": val_scenes_list,
            "camera_names": list(camera_names) if camera_names is not None else None,
            "coord": {
                "lidar_points_frame": "world",
                "pose_convention": "sensor_to_world",
                "ego0_frame_index": 0,
            },
            "settings": [asdict(setting) for setting in clip_settings],
            "total_clips": full_count,
        },
        "clips": full_entries,
    }
    tiny_payload: Dict[str, object] = {
        "meta": {
            "data_root": str(data_root_path),
            "preset": preset,
            "target_fps": target_fps,
            "created_at": created_at,
            "scene_ids": sorted(tiny_set),
            "test_scenes": [scene for scene in val_scenes_list if scene in tiny_set],
            "camera_names": list(camera_names) if camera_names is not None else None,
            "coord": {
                "lidar_points_frame": "world",
                "pose_convention": "sensor_to_world",
                "ego0_frame_index": 0,
            },
            "settings": [asdict(setting) for setting in clip_settings],
            "total_clips": tiny_count,
        },
        "clips": tiny_entries,
    }
    with out_full_path.open("wb") as full_fh:
        pickle.dump(full_payload, full_fh)
    with out_tiny_path.open("wb") as tiny_fh:
        pickle.dump(tiny_payload, tiny_fh)
    return full_count, tiny_count


def load_clip_index(path: str) -> Dict[str, object]:
    """加载 PKL 索引文件。

    Args:
        path: PKL 文件路径。

    Returns:
        反序列化后的索引字典。
    """
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def load_clip(meta_entry: Dict[str, object], data_root: str) -> Dict[str, object]:
    """将索引条目中的相对路径展开为绝对路径。

    Args:
        meta_entry: clip 元信息条目。
        data_root: PandaSet 根目录。

    Returns:
        带绝对路径的 clip 字典。

    Raises:
        ValueError: 索引字段格式非法。
    """
    data_root_path = Path(data_root)
    clip = dict(meta_entry)
    lidar_paths = clip.get("lidar_paths")
    image_paths = clip.get("image_paths")
    if not isinstance(lidar_paths, list):
        raise ValueError("lidar_paths 格式非法")
    if not isinstance(image_paths, dict):
        raise ValueError("image_paths 格式非法")
    clip["lidar_paths"] = [str(data_root_path / p) for p in lidar_paths]
    clip["image_paths"] = {
        cam: [str(data_root_path / p) for p in paths]
        for cam, paths in image_paths.items()
    }
    return clip


def sample_views_for_train(
    clip: Dict[str, object],
    num_views: Optional[int],
    strategy: str = "all",
) -> List[str]:
    """从 clip 中选择训练视角。

    Args:
        clip: clip 元信息。
        num_views: 需要的视角数量，None 表示全部。
        strategy: 采样策略，支持 "all" 或 "random"。

    Returns:
        选择后的相机视角名称列表。

    Raises:
        ValueError: strategy 不支持时抛出。
    """
    views = list(clip.get("views", []))
    if not views:
        return []
    if num_views is None or num_views >= len(views) or strategy == "all":
        return views
    if strategy == "random":
        return random.sample(views, num_views)
    raise ValueError(f"unsupported strategy: {strategy}")
