"""Scene Flow 伪 GT 与指标计算的最小单元测试（不依赖 torch/pandas）。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux4d.datasets.pandaset_cuboids import PandaSetCuboid, assign_points_to_cuboids_world  # noqa: E402
from flux4d.metrics.scene_flow import (  # noqa: E402
    build_scene_flow_gt_from_cuboids,
    compute_scene_flow_metrics,
)


def _cuboid(
    *,
    uuid: str,
    center_xyz: np.ndarray,
    yaw_rad: float = 0.0,
    dims_xyz: np.ndarray = np.array([2.0, 2.0, 2.0]),
    label: str = "Car",
) -> PandaSetCuboid:
    return PandaSetCuboid(
        uuid=uuid,
        label=label,
        yaw_rad=float(yaw_rad),
        stationary=False,
        center_world=center_xyz.astype(np.float32),
        dims_xyz=dims_xyz.astype(np.float32),
    )


def test_scene_flow_gt_rigid_translation() -> None:
    """验证刚体平移下的 GT flow 构造。"""
    cub0 = _cuboid(uuid="obj", center_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    cub1 = _cuboid(uuid="obj", center_xyz=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    assigned, _ = assign_points_to_cuboids_world(points, [cub0])

    gt = build_scene_flow_gt_from_cuboids(
        points_world_t0=points,
        assigned_cuboid_indices_t0=assigned,
        cuboids_t0=[cub0],
        cuboids_t1=[cub1],
        dynamic_flags_t0={"obj": True},
        label_to_bucket={"Car": "vehicle"},
        bucket_names=("background", "vehicle", "pedestrian", "cyclist", "other"),
        default_bucket="other",
    )

    # First two points move with object: +1m along x; third is background (flow=0)
    np.testing.assert_allclose(gt.flow_gt_world[0], np.array([1.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(gt.flow_gt_world[1], np.array([1.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(gt.flow_gt_world[2], np.array([0.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    assert gt.valid_mask.tolist() == [True, True, True]
    assert gt.fd_mask.tolist() == [True, True, False]
    assert gt.bs_mask.tolist() == [False, False, True]


def test_scene_flow_metrics_perfect_prediction() -> None:
    """验证完美预测下的指标。"""
    cub0 = _cuboid(uuid="obj", center_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    cub1 = _cuboid(uuid="obj", center_xyz=np.array([0.2, 0.0, 0.0], dtype=np.float32))
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    assigned, _ = assign_points_to_cuboids_world(points, [cub0])
    gt = build_scene_flow_gt_from_cuboids(
        points_world_t0=points,
        assigned_cuboid_indices_t0=assigned,
        cuboids_t0=[cub0],
        cuboids_t1=[cub1],
        dynamic_flags_t0={"obj": False},
        label_to_bucket={"Car": "vehicle"},
        bucket_names=("background", "vehicle", "pedestrian", "cyclist", "other"),
        default_bucket="other",
    )
    pred = gt.flow_gt_world.copy()
    metrics = compute_scene_flow_metrics(pred, gt, denom_min_m=0.05)

    assert abs(metrics.epe3d - 0.0) < 1e-6
    assert abs(metrics.acc5 - 1.0) < 1e-6
    assert abs(metrics.acc10 - 1.0) < 1e-6
    # Only one foreground point, marked FS.
    assert metrics.epe_3way_count["FS"] == 1

