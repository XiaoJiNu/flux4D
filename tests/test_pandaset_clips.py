"""PandaSet 索引构建的最小单元测试。"""

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux4d.datasets.pandaset_clips import (  # noqa: E402
    build_pandaset_clip_index,
    load_clip_index,
)


def _write_json(path: Path, obj: Any) -> None:
    """写入 JSON 文件。

    Args:
        path (Path): JSON 文件路径。
        obj (Any): 可 JSON 序列化的对象。

    Returns:
        None: 无返回值。

    Raises:
        TypeError: obj 无法序列化时抛出。
        OSError: 文件写入失败时抛出。

    实现要点:
        - 自动创建父目录。
        - 使用 json.dumps 序列化写入。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _touch(path: Path) -> None:
    """创建空文件（若不存在）。

    Args:
        path (Path): 目标文件路径。

    Returns:
        None: 无返回值。

    Raises:
        OSError: 文件写入失败时抛出。

    实现要点:
        - 自动创建父目录。
        - 写入空字节占位。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_build_pandaset_clip_index(tmp_path: Path) -> None:
    """验证索引构建流程与字段一致性。

    Args:
        tmp_path (Path): pytest 提供的临时目录。

    Returns:
        None: 无返回值。

    Raises:
        AssertionError: 断言失败时抛出。

    实现要点:
        - 构造最小化的 PandaSet 目录结构并写入必要文件。
        - 生成索引后检查 clip 数量与关键字段。
    """
    data_root = tmp_path / "pandaset"
    scene = data_root / "001"
    lidar_dir = scene / "lidar"
    camera_dir = scene / "camera" / "front_camera"

    num_frames = 31
    timestamps = [1557539924.5 + i * 0.1 for i in range(num_frames)]
    poses = [
        {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "heading": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
        for _ in range(num_frames)
    ]

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
        clip_len_s=1.5,
        stride_s=1.5,
        target_fps=10.0,
        tiny_scenes=["001"],
        val_scenes=["001"],
    )

    assert full_count == 2
    assert tiny_count == 2

    payload = load_clip_index(str(out_full))
    assert payload["meta"]["total_clips"] == 2

    clip = payload["clips"][0]
    assert clip["scene_id"] == "001"
    assert clip["fps"] == 10.0
    assert len(clip["frame_ids"]) == 15
    assert clip["views"] == ["front_camera"]
    assert len(clip["image_paths"]["front_camera"]) == 15
