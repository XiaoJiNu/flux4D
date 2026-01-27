"""Lift 模块核心函数的最小单元测试。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux4d.lift.lift_lidar import (  # noqa: E402
    compute_knn_mean_distance,
    project_points_to_camera,
    quat_to_rot_matrix,
    voxel_downsample_points,
)


def test_quat_to_rot_matrix_identity() -> None:
    """验证单位四元数对应单位旋转矩阵。"""
    quat = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    rot = quat_to_rot_matrix(quat)
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-6)


def test_project_points_to_camera() -> None:
    """验证简单投影输出与掩码。"""
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    uv, depth, mask = project_points_to_camera(points, intrinsics, (10, 10))

    assert uv.shape == (2, 2)
    assert depth.shape == (2,)
    assert mask.tolist() == [True, True]
    np.testing.assert_allclose(uv[0], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(uv[1], [1.0, 0.0], atol=1e-6)


def test_voxel_downsample_points() -> None:
    """验证体素下采样会合并落在同体素的点。"""
    points = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.04, 0.01, 0.02],
            [1.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    downsampled = voxel_downsample_points(points, voxel_size=0.1)
    assert downsampled.shape[0] == 2


def test_compute_knn_mean_distance() -> None:
    """验证 kNN 平均距离的基本数值。"""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    distances = compute_knn_mean_distance(points, k=1, max_bruteforce_points=10)
    np.testing.assert_allclose(distances, np.array([1.0, 1.0, 1.0]), atol=1e-6)
