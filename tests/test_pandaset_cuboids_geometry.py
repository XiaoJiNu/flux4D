"""PandaSet cuboids 几何工具的单元测试（不依赖 pandas）。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux4d.datasets.pandaset_cuboids import (  # noqa: E402
    PandaSetCuboid,
    assign_points_to_cuboids_world,
    compute_cuboid_dynamic_flags_by_frame,
    points_in_cuboid_world,
)


def _cuboid(
    *,
    uuid: str,
    center_xyz: np.ndarray,
    dims_xyz: np.ndarray,
    yaw_rad: float = 0.0,
    label: str = "Car",
    stationary: bool = False,
) -> PandaSetCuboid:
    """构造测试 cuboid。"""
    return PandaSetCuboid(
        uuid=uuid,
        label=label,
        yaw_rad=float(yaw_rad),
        stationary=bool(stationary),
        center_world=center_xyz.astype(np.float32),
        dims_xyz=dims_xyz.astype(np.float32),
    )


def test_points_in_cuboid_world_axis_aligned() -> None:
    """验证 axis-aligned 立方体的包含关系。"""
    cub = _cuboid(uuid="a", center_xyz=np.array([0.0, 0.0, 0.0]), dims_xyz=np.array([2.0, 2.0, 2.0]))
    points = np.array(
        [
            [0.0, 0.0, 0.0],  # inside
            [0.99, -0.5, 0.0],  # inside
            [1.01, 0.0, 0.0],  # outside (x)
            [0.0, 0.0, -1.01],  # outside (z)
        ],
        dtype=np.float32,
    )
    inside = points_in_cuboid_world(points, cub)
    assert inside.tolist() == [True, True, False, False]


def test_assign_points_to_cuboids_prefers_smaller_volume() -> None:
    """验证多重命中时优先选择体积更小的 cuboid。"""
    big = _cuboid(uuid="big", center_xyz=np.zeros(3), dims_xyz=np.array([4.0, 4.0, 4.0]))
    small = _cuboid(uuid="small", center_xyz=np.zeros(3), dims_xyz=np.array([2.0, 2.0, 2.0]))
    points = np.array([[0.1, 0.1, 0.1], [1.5, 0.0, 0.0]], dtype=np.float32)
    assigned, _ = assign_points_to_cuboids_world(points, [big, small])
    # both points inside big; only first inside small -> first should pick small, second picks big
    assert assigned.tolist() == [1, 0]


def test_compute_cuboid_dynamic_flags_by_frame_with_smoothing() -> None:
    """验证动态标记与多数表决平滑。"""
    cub0 = _cuboid(uuid="x", center_xyz=np.array([0.0, 0.0, 0.0]), dims_xyz=np.array([2.0, 2.0, 2.0]))
    cub1 = _cuboid(uuid="x", center_xyz=np.array([0.1, 0.0, 0.0]), dims_xyz=np.array([2.0, 2.0, 2.0]))
    cub2 = _cuboid(uuid="x", center_xyz=np.array([0.0, 0.0, 0.0]), dims_xyz=np.array([2.0, 2.0, 2.0]))
    by_frame = {0: [cub0], 1: [cub1], 2: [cub2]}

    raw = compute_cuboid_dynamic_flags_by_frame(
        by_frame, trans_thresh_m=0.05, yaw_thresh_deg=1.0, smoothing_window=1
    )
    assert raw[0]["x"] is True
    assert raw[1]["x"] is True
    assert raw[2]["x"] is True  # frame2 uses fallback (frame1->frame2) since next missing

    smooth = compute_cuboid_dynamic_flags_by_frame(
        by_frame, trans_thresh_m=0.05, yaw_thresh_deg=1.0, smoothing_window=3
    )
    assert smooth[0]["x"] is True
    assert smooth[1]["x"] is True
    assert smooth[2]["x"] is True

