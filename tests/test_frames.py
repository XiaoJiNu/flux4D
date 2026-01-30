"""坐标变换工具的最小单元测试（NumPy）。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux4d.utils.frames import (  # noqa: E402
    build_frame_transform_numpy,
    transform_points_numpy,
    transform_vectors_numpy,
)


def _pose(position_xyz: np.ndarray, quat_wxyz: np.ndarray) -> dict:
    """构造最小 pose 字典。"""
    return {
        "position": {"x": float(position_xyz[0]), "y": float(position_xyz[1]), "z": float(position_xyz[2])},
        "heading": {"w": float(quat_wxyz[0]), "x": float(quat_wxyz[1]), "y": float(quat_wxyz[2]), "z": float(quat_wxyz[3])},
    }


def test_world_ego0_roundtrip_points() -> None:
    """验证点 world->ego0->world 的往返一致性。"""
    theta = np.deg2rad(90.0)
    quat = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)], dtype=np.float32)
    pose = _pose(np.array([10.0, 2.0, -1.0], dtype=np.float32), quat)
    frame = build_frame_transform_numpy(pose)

    points_world = np.array(
        [
            [10.0, 2.0, -1.0],
            [11.0, 2.0, -1.0],
            [10.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    points_ego0 = transform_points_numpy(points_world, frame.T_ego0_world)
    points_world_rt = transform_points_numpy(points_ego0, frame.T_world_ego0)
    np.testing.assert_allclose(points_world_rt, points_world, atol=1e-5)


def test_transform_vectors_ignores_translation() -> None:
    """验证向量变换只受旋转影响，不受平移影响。"""
    theta = np.deg2rad(45.0)
    quat = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)], dtype=np.float32)
    pose_a = _pose(np.array([0.0, 0.0, 0.0], dtype=np.float32), quat)
    pose_b = _pose(np.array([100.0, -30.0, 5.0], dtype=np.float32), quat)
    frame_a = build_frame_transform_numpy(pose_a)
    frame_b = build_frame_transform_numpy(pose_b)

    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    out_a = transform_vectors_numpy(vectors, frame_a.R_ego0_world)
    out_b = transform_vectors_numpy(vectors, frame_b.R_ego0_world)
    np.testing.assert_allclose(out_a, out_b, atol=1e-6)

