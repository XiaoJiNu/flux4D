"""阶段4：Rendered Velocity（v_r）解析公式的最小单测（NumPy）。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _project_pinhole(points_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """简单 pinhole 投影（相机坐标系）。

    Args:
        points_cam: (N, 3)。
        fx: 焦距 x。
        fy: 焦距 y。
        cx: 主点 x。
        cy: 主点 y。

    Returns:
        (N, 2) 像素坐标。
    """
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = np.clip(points_cam[:, 2], 1e-8, None)
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return np.stack([u, v], axis=-1)


def _rendered_displacement_jacobian(
    points_cam: np.ndarray,
    velocities_cam: np.ndarray,
    fx: float,
    fy: float,
    delta_t_norm: float,
) -> np.ndarray:
    """解析 Jacobian 计算像素位移（对应 t'->t'+Δt_norm）。"""
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = np.clip(points_cam[:, 2], 1e-8, None)
    vx = velocities_cam[:, 0]
    vy = velocities_cam[:, 1]
    vz = velocities_cam[:, 2]
    du_dt = fx * (vx * z - x * vz) / (z * z)
    dv_dt = fy * (vy * z - y * vz) / (z * z)
    return np.stack([du_dt, dv_dt], axis=-1) * float(delta_t_norm)


def _clip_vector_magnitude(vec: np.ndarray, clip_mag: float, eps: float = 1e-8) -> np.ndarray:
    """按向量范数裁剪，保持方向不变。"""
    mag = np.linalg.norm(vec, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(clip_mag) / np.clip(mag, eps, None))
    return vec * scale


def test_rendered_velocity_matches_finite_difference() -> None:
    """验证解析 Jacobian 与有限差分的像素位移一致。"""
    fx, fy, cx, cy = 100.0, 120.0, 2.0, -3.0
    points_cam = np.array([[1.0, 2.0, 10.0]], dtype=np.float64)
    velocities_cam = np.array([[0.5, -0.25, 0.1]], dtype=np.float64)
    delta_t_norm = 1e-4

    uv0 = _project_pinhole(points_cam, fx=fx, fy=fy, cx=cx, cy=cy)
    points_cam_1 = points_cam + velocities_cam * float(delta_t_norm)
    uv1 = _project_pinhole(points_cam_1, fx=fx, fy=fy, cx=cx, cy=cy)
    disp_fd = uv1 - uv0

    disp_jac = _rendered_displacement_jacobian(
        points_cam=points_cam,
        velocities_cam=velocities_cam,
        fx=fx,
        fy=fy,
        delta_t_norm=delta_t_norm,
    )
    np.testing.assert_allclose(disp_jac, disp_fd, atol=1e-8)


def test_clip_vector_magnitude_preserves_direction() -> None:
    """验证裁剪会限制范数并保持方向。"""
    vec = np.array([[3.0, 4.0], [0.0, 0.0], [-6.0, 8.0]], dtype=np.float32)
    out = _clip_vector_magnitude(vec, clip_mag=5.0)
    mag = np.linalg.norm(out, axis=-1)
    assert mag[0] == 5.0
    assert mag[1] == 0.0
    assert mag[2] == 5.0

    # Non-zero vectors should keep direction (cosine similarity ~ 1).
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 1.0
        return float(np.dot(a, b) / denom)

    assert _cos(vec[0], out[0]) > 0.999
    assert _cos(vec[2], out[2]) > 0.999
