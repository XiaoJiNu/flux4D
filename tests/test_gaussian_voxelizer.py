"""Gaussian 体素化模块的最小单元测试（NumPy 版本）。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux4d.storm.gaussian_voxelizer import voxelize_points_numpy  # noqa: E402


def test_voxelize_points_numpy_mean_pooling() -> None:
    """验证同体素内 mean pooling 与点到体素映射。"""
    points = np.array(
        [
            [0.05, 0.05, 0.05],  # voxel (0,0,0)
            [0.09, 0.01, 0.02],  # voxel (0,0,0)
            [0.21, 0.01, 0.02],  # voxel (2,0,0) when voxel=0.1
        ],
        dtype=np.float32,
    )
    feats = np.array(
        [
            [1.0, 0.0],
            [3.0, 2.0],
            [5.0, 4.0],
        ],
        dtype=np.float32,
    )
    res = voxelize_points_numpy(
        points_xyz=points,
        features=feats,
        point_cloud_range=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        voxel_size=[0.1, 0.1, 0.1],
    )

    assert res.voxel_coords_xyz.shape[1] == 3
    assert res.voxel_features.shape[1] == 2
    assert res.valid_mask.tolist() == [True, True, True]
    assert res.point2voxel.shape == (3,)

    # voxel (0,0,0) should pool the first two points: mean([1,0],[3,2])=[2,1]
    idx_000 = np.where((res.voxel_coords_xyz == np.array([0, 0, 0])).all(axis=1))[0]
    assert idx_000.size == 1
    np.testing.assert_allclose(res.voxel_features[idx_000[0]], np.array([2.0, 1.0]), atol=1e-6)

    # third point belongs to a different voxel and keeps its feature
    idx_other = np.where((res.voxel_coords_xyz == np.array([2, 0, 0])).all(axis=1))[0]
    assert idx_other.size == 1
    np.testing.assert_allclose(res.voxel_features[idx_other[0]], feats[2], atol=1e-6)


def test_voxelize_points_numpy_invalid_points() -> None:
    """验证超出 point_cloud_range 的点会被标记为无效。"""
    points = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    feats = np.array([[1.0]], dtype=np.float32)
    res = voxelize_points_numpy(
        points_xyz=points,
        features=feats,
        point_cloud_range=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        voxel_size=[0.1, 0.1, 0.1],
    )
    assert res.valid_mask.tolist() == [False]
    assert res.point2voxel.tolist() == [-1]
    assert res.voxel_coords_xyz.shape == (0, 3)
    assert res.voxel_features.shape == (0, 1)

