"""PandaSet clip index builder."""

from __future__ import annotations

import json
import pickle
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _parse_frame_index(filename: str) -> Optional[int]:
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
    files = list(folder.glob(pattern))
    if not files:
        return []

    def sort_key(path: Path) -> Tuple[int, object]:
        index = _parse_frame_index(path.name)
        if index is None:
            return (1, path.name)
        return (0, index)

    return sorted(files, key=sort_key)


def _load_json(path: Path):
    return json.loads(path.read_text())


def _infer_fps(timestamps: List[float]) -> Optional[float]:
    if len(timestamps) < 2:
        return None
    diffs = [b - a for a, b in zip(timestamps[:-1], timestamps[1:])]
    median_dt = statistics.median(diffs)
    if median_dt <= 0:
        return None
    return 1.0 / median_dt


def _ensure_len(name: str, values: List, expected: int, strict: bool) -> None:
    if len(values) == expected:
        return
    message = f"{name} length {len(values)} != {expected}"
    if strict:
        raise ValueError(message)
    print(f"[warn] {message}")


def _scene_dirs(data_root: Path) -> List[Path]:
    dirs = []
    for item in data_root.iterdir():
        if not item.is_dir():
            continue
        if item.name.isdigit():
            dirs.append(item)
    return sorted(dirs, key=lambda p: p.name)


def _relative_paths(paths: Iterable[Path], data_root: Path) -> List[str]:
    return [str(path.relative_to(data_root)) for path in paths]


def _slice_list(values: List, start: int, end: int) -> List:
    return values[start:end]


def _select_val_scenes(scene_ids: List[str], val_count: int) -> List[str]:
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
):
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
        message = f"scene {scene_id} has {len(lidar_files)} frames < clip_len {clip_len_frames}"
        if strict:
            raise ValueError(message)
        print(f"[warn] {message}")
        return []

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

    entries = []
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
    data_root_path = Path(data_root)
    scenes = _scene_dirs(data_root_path)
    if max_scenes is not None:
        scenes = scenes[: max_scenes]
    scene_ids = [scene.name for scene in scenes]
    if tiny_scenes is None:
        tiny_scenes = scene_ids[:tiny_num_scenes]
    tiny_set = set(tiny_scenes)
    if val_scenes is None:
        val_scenes_list = _select_val_scenes(scene_ids, val_num_scenes)
    else:
        val_scenes_list = list(val_scenes)

    out_full_path = Path(out_pkl_full)
    out_tiny_path = Path(out_pkl_tiny)
    out_full_path.parent.mkdir(parents=True, exist_ok=True)
    out_tiny_path.parent.mkdir(parents=True, exist_ok=True)

    full_count = 0
    tiny_count = 0
    full_entries = []
    tiny_entries = []
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


def load_clip_index(path: str) -> Dict:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def load_clip(meta_entry: Dict, data_root: str) -> Dict:
    data_root_path = Path(data_root)
    clip = dict(meta_entry)
    clip["lidar_paths"] = [str(data_root_path / p) for p in clip["lidar_paths"]]
    clip["image_paths"] = {
        cam: [str(data_root_path / p) for p in paths]
        for cam, paths in clip["image_paths"].items()
    }
    return clip


def sample_views_for_train(
    clip: Dict, num_views: Optional[int], strategy: str = "all"
) -> List[str]:
    views = list(clip.get("views", []))
    if not views:
        return []
    if num_views is None or num_views >= len(views) or strategy == "all":
        return views
    if strategy == "random":
        return random.sample(views, num_views)
    raise ValueError(f"unsupported strategy: {strategy}")
