"""PandaSet 索引构建的最小单元测试。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux4d.datasets.pandaset_clips import (  # noqa: E402
    build_pandaset_clip_index,
    load_clip_index,
)


def _write_json(path: Path, obj: object) -> None:
    """写入 JSON 文件。

    Args:
        path: JSON 文件路径。
        obj: 可 JSON 序列化的对象。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _touch(path: Path) -> None:
    """创建空文件（若不存在）。

    Args:
        path: 目标文件路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_build_pandaset_clip_index(tmp_path: Path) -> None:
    """验证索引构建流程与字段一致性。

    Args:
        tmp_path: pytest 提供的临时目录。

    Note:
        构造最小化 PandaSet 目录结构并检查关键字段。
    """
    data_root = tmp_path / "pandaset"
    num_frames = 80
    timestamps = [1557539924.5 + i * 0.1 for i in range(num_frames)]
    poses = [
        {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "heading": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
        for _ in range(num_frames)
    ]

    for scene_id in ("001", "002"):
        scene = data_root / scene_id
        lidar_dir = scene / "lidar"
        camera_dir = scene / "camera" / "front_camera"
        _write_json(lidar_dir / "timestamps.json", timestamps)
        _write_json(lidar_dir / "poses.json", poses)
        for i in range(num_frames):
            _touch(lidar_dir / f"{i:02d}.pkl.gz")

        _write_json(camera_dir / "timestamps.json", timestamps)
        _write_json(camera_dir / "poses.json", poses)
        _write_json(
            camera_dir / "intrinsics.json",
            {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        )
        for i in range(num_frames):
            _touch(camera_dir / f"{i:02d}.jpg")

    out_full = tmp_path / "full.pkl"
    out_tiny = tmp_path / "tiny.pkl"
    full_count, tiny_count = build_pandaset_clip_index(
        data_root=str(data_root),
        out_pkl_full=str(out_full),
        out_pkl_tiny=str(out_tiny),
        preset="paper",
        target_fps=10.0,
        tiny_scenes=["001", "002"],
    )

    assert full_count == 22
    assert tiny_count == 22

    payload = load_clip_index(str(out_full))
    assert payload["meta"]["total_clips"] == 22
    assert payload["meta"]["preset"] == "paper"

    clips = payload["clips"]
    assert isinstance(clips, list)

    train_interp = [
        clip for clip in clips if clip.get("scene_id") == "002" and clip.get("setting") == "train_interpolation"
    ]
    train_future = [
        clip for clip in clips if clip.get("scene_id") == "002" and clip.get("setting") == "train_future"
    ]
    eval_future = [
        clip for clip in clips if clip.get("scene_id") == "001" and clip.get("setting") == "eval_future"
    ]
    assert len(train_interp) == 12
    assert len(train_future) == 6
    assert len(eval_future) == 4

    clip = eval_future[0]
    assert clip["scene_id"] == "001"
    assert clip["fps"] == 10.0
    assert clip["clip_len_frames"] == 16
    assert len(clip["frame_ids"]) == 16
    assert clip["views"] == ["front_camera"]
    assert len(clip["image_paths"]["front_camera"]) == 16
