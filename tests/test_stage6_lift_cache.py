"""阶段6 LiftCache 的 NPZ 落盘单元测试（不依赖 torch/pandas）。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux4d.engine.stage6 import LiftCache, LiftCacheItem  # noqa: E402


def test_lift_cache_save_npz_tmp_suffix_is_valid() -> None:
    """验证临时文件命名不会触发 numpy 自动追加 .npz 导致 replace 失败。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir) / "cache"
        cache = LiftCache(cache_dir, max_items_in_memory=0)

        item = LiftCacheItem(
            positions=np.zeros((2, 3), dtype=np.float32),
            rotations=np.zeros((2, 4), dtype=np.float32),
            scales=np.zeros((2, 3), dtype=np.float32),
            opacities=np.zeros((2,), dtype=np.float32),
            colors=np.zeros((2, 3), dtype=np.float32),
            timestamps=np.array([0.0, 1.0], dtype=np.float32),
            source_frame_indices=np.array([0, -1], dtype=np.int16),
        )

        path = cache_dir / "demo.npz"
        cache._save_npz(path, item)  # pylint: disable=protected-access

        assert path.exists()
        # 确保不会遗留 "demo.tmp.npz.npz" 这种错误文件名。
        assert not (cache_dir / "demo.tmp.npz").exists()
        assert not (cache_dir / "demo.tmp.npz.npz").exists()

        loaded = cache._load_npz(path)  # pylint: disable=protected-access
        np.testing.assert_allclose(loaded.positions, item.positions, atol=0.0)
        np.testing.assert_allclose(loaded.timestamps, item.timestamps, atol=0.0)
        assert loaded.source_frame_indices.dtype == np.int16


def test_lift_cache_recovers_from_legacy_tmp_npz() -> None:
    """验证可从旧版本遗留的 `xxx.npz.tmp.npz` 中自动恢复缓存。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir) / "cache"
        cache = LiftCache(cache_dir, max_items_in_memory=0)

        item = LiftCacheItem(
            positions=np.ones((1, 3), dtype=np.float32),
            rotations=np.ones((1, 4), dtype=np.float32),
            scales=np.ones((1, 3), dtype=np.float32),
            opacities=np.ones((1,), dtype=np.float32),
            colors=np.ones((1, 3), dtype=np.float32),
            timestamps=np.array([0.25], dtype=np.float32),
            source_frame_indices=np.array([7], dtype=np.int16),
        )
        path = cache._path_for_key("demo")  # pylint: disable=protected-access
        legacy = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            str(legacy),
            positions=item.positions,
            rotations=item.rotations,
            scales=item.scales,
            opacities=item.opacities,
            colors=item.colors,
            timestamps=item.timestamps,
            source_frame_indices=item.source_frame_indices,
        )
        assert legacy.exists()
        assert not path.exists()

        loaded = cache.get_or_build(
            key="demo",
            clip={},
            data_root="",
            input_frame_indices=[],
            timestamp_frame_indices=[],
            camera_names=[],
            voxel_size_m=0.2,
            knn_k=3,
            opacity_init=0.5,
            random_seed=0,
            num_sky_points=0,
            max_gaussians=None,
            lidar_sensor_id=-1,
        )
        assert path.exists()
        assert not legacy.exists()
        np.testing.assert_allclose(loaded.positions, item.positions, atol=0.0)
