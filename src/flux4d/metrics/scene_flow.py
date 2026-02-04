"""Scene Flow 评测工具（PandaSet 口径）。

本模块提供阶段6所需的 Scene Flow 评测能力：
- 基于 PandaSet cuboids 构造“伪 GT” 3D scene flow（世界坐标系，单位：米）。
- 计算常用指标：EPE3D、Acc5、Acc10、θϵ（弧度角误差）、EPE-3way（BS/FS/FD）、
  分桶归一化 EPE（bucketed normalized EPE）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np

from flux4d.datasets.pandaset_cuboids import PandaSetCuboid, _wrap_angle_rad, index_cuboids_by_uuid


@dataclass(frozen=True)
class SceneFlowGtResult:
    """Scene Flow 伪 GT 构造结果。"""

    flow_gt_world: np.ndarray  # (N, 3) float32
    valid_mask: np.ndarray  # (N,) bool
    bs_mask: np.ndarray  # (N,) bool
    fs_mask: np.ndarray  # (N,) bool
    fd_mask: np.ndarray  # (N,) bool
    bucket_names: Sequence[str]
    bucket_indices: np.ndarray  # (N,) int32 in [0, B) aligned to bucket_names


def build_scene_flow_gt_from_cuboids(
    points_world_t0: np.ndarray,
    assigned_cuboid_indices_t0: np.ndarray,
    cuboids_t0: Sequence[PandaSetCuboid],
    cuboids_t1: Sequence[PandaSetCuboid],
    *,
    dynamic_flags_t0: Mapping[str, bool],
    label_to_bucket: Mapping[str, str],
    bucket_names: Sequence[str],
    default_bucket: str = "other",
) -> SceneFlowGtResult:
    """从 cuboids 构造相邻帧对 (t0->t1) 的 Scene Flow 伪 GT。

    Args:
        points_world_t0: 帧 t0 的点坐标（world），形状为 (N, 3)。
        assigned_cuboid_indices_t0: 每点在 t0 命中的 cuboid 索引（-1 表示背景），形状为 (N,)。
        cuboids_t0: t0 帧的 cuboid 列表。
        cuboids_t1: t1 帧的 cuboid 列表（用于 track 对应）。
        dynamic_flags_t0: t0 帧 uuid->is_dynamic 的映射（用于 FS/FD 划分）。
        label_to_bucket: cuboid label -> bucket name 的映射。
        bucket_names: bucket name 的有序列表，作为输出 `bucket_indices` 的字典序。
        default_bucket: label 不在映射中时使用的 bucket 名称。

    Returns:
        SceneFlowGtResult。

    Raises:
        ValueError: 输入形状非法或 bucket_names 缺失必要项。

    Note:
        - 背景点（assigned=-1）按“世界静止”处理：flow_gt=0，且计入 BS。
        - 前景点若在 t1 找不到同 uuid cuboid，则记为 invalid（valid_mask=False）。
        - 采用刚体假设：点在对象坐标系中保持不变，通过 `T_world_obj(t0/t1)` 做搬运。
    """
    if points_world_t0.ndim != 2 or int(points_world_t0.shape[1]) != 3:
        raise ValueError("points_world_t0 形状必须为 (N, 3)")
    if assigned_cuboid_indices_t0.ndim != 1:
        raise ValueError("assigned_cuboid_indices_t0 形状必须为 (N,)")
    if int(assigned_cuboid_indices_t0.shape[0]) != int(points_world_t0.shape[0]):
        raise ValueError("assigned_cuboid_indices_t0 与 points_world_t0 的 N 不一致")
    if "background" not in bucket_names:
        raise ValueError("bucket_names 必须包含 'background'")
    if default_bucket not in bucket_names:
        raise ValueError(f"bucket_names 必须包含 default_bucket={default_bucket!r}")

    bucket_to_index = {name: i for i, name in enumerate(bucket_names)}
    cuboids_t1_by_uuid = index_cuboids_by_uuid(cuboids_t1)

    n = int(points_world_t0.shape[0])
    flow_gt = np.zeros((n, 3), dtype=np.float32)
    valid = np.ones((n,), dtype=bool)
    bs = assigned_cuboid_indices_t0 < 0
    fs = np.zeros((n,), dtype=bool)
    fd = np.zeros((n,), dtype=bool)
    bucket_idx = np.full((n,), int(bucket_to_index["background"]), dtype=np.int32)

    # Foreground points: compute rigid transport when track exists.
    points_world_f32 = points_world_t0.astype(np.float32, copy=False)
    for cub_idx, cub in enumerate(cuboids_t0):
        point_mask = assigned_cuboid_indices_t0 == int(cub_idx)
        if not np.any(point_mask):
            continue
        other = cuboids_t1_by_uuid.get(cub.uuid)
        if other is None:
            valid[point_mask] = False
            continue

        rot0 = cub.rotation_world_from_obj
        rot1 = other.rotation_world_from_obj
        center0 = cub.center_world.astype(np.float32)
        center1 = other.center_world.astype(np.float32)
        p_rel = points_world_f32[point_mask] - center0[None, :]
        p_obj = p_rel @ rot0
        p_world_hat = p_obj @ rot1.T + center1[None, :]
        flow_gt[point_mask] = (p_world_hat - points_world_f32[point_mask]).astype(np.float32, copy=False)

        is_dyn = bool(dynamic_flags_t0.get(cub.uuid, False))
        if is_dyn:
            fd[point_mask] = True
        else:
            fs[point_mask] = True

        bucket_name = str(label_to_bucket.get(cub.label, default_bucket))
        bucket_idx[point_mask] = int(bucket_to_index.get(bucket_name, bucket_to_index[default_bucket]))

    # Background: BS, already flow=0 and bucket=background.
    bs = bs & valid
    fs = fs & valid
    fd = fd & valid
    return SceneFlowGtResult(
        flow_gt_world=flow_gt,
        valid_mask=valid,
        bs_mask=bs,
        fs_mask=fs,
        fd_mask=fd,
        bucket_names=tuple(bucket_names),
        bucket_indices=bucket_idx,
    )


@dataclass(frozen=True)
class SceneFlowMetrics:
    """Scene Flow 指标汇总。"""

    epe3d: float
    acc5: float
    acc10: float
    theta_e: float
    epe_3way: Dict[str, float]
    epe_3way_count: Dict[str, int]
    bucketed_nepe: Dict[str, float]
    bucketed_nepe_count: Dict[str, int]


def _safe_mean(values: np.ndarray) -> float:
    """对可能为空的数组求均值，空则返回 0。"""
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _l2_norm(xyz: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """计算向量 2 范数。"""
    return np.sqrt(np.clip(np.sum(xyz * xyz, axis=-1), eps, None)).astype(np.float32, copy=False)


def compute_scene_flow_metrics(
    flow_pred_world: np.ndarray,
    gt: SceneFlowGtResult,
    *,
    denom_min_m: float = 0.05,
    eps: float = 1e-12,
) -> SceneFlowMetrics:
    """计算 Scene Flow 指标。

    Args:
        flow_pred_world: 预测 flow（world，米），形状为 (N, 3)。
        gt: GT 结构（由 `build_scene_flow_gt_from_cuboids` 输出）。
        denom_min_m: 归一化 EPE 的分母下界（米），用于 bucketed normalized EPE。
        eps: 数值稳定项。

    Returns:
        SceneFlowMetrics。

    Raises:
        ValueError: 输入形状非法。
    """
    if flow_pred_world.ndim != 2 or int(flow_pred_world.shape[1]) != 3:
        raise ValueError("flow_pred_world 形状必须为 (N, 3)")
    if int(flow_pred_world.shape[0]) != int(gt.flow_gt_world.shape[0]):
        raise ValueError("flow_pred_world 与 gt 的 N 不一致")
    if float(denom_min_m) <= 0.0:
        raise ValueError("denom_min_m 必须为正数")

    valid = gt.valid_mask.astype(bool, copy=False)
    if not np.any(valid):
        return SceneFlowMetrics(
            epe3d=0.0,
            acc5=0.0,
            acc10=0.0,
            theta_e=0.0,
            epe_3way={"BS": 0.0, "FS": 0.0, "FD": 0.0},
            epe_3way_count={"BS": 0, "FS": 0, "FD": 0},
            bucketed_nepe={name: 0.0 for name in gt.bucket_names},
            bucketed_nepe_count={name: 0 for name in gt.bucket_names},
        )

    pred = flow_pred_world.astype(np.float32, copy=False)[valid]
    gt_flow = gt.flow_gt_world.astype(np.float32, copy=False)[valid]
    error = pred - gt_flow
    epe = _l2_norm(error, eps=float(eps))

    epe3d = float(np.mean(epe))
    acc5 = float(np.mean((epe <= 0.05).astype(np.float32)))
    acc10 = float(np.mean((epe <= 0.10).astype(np.float32)))

    # Angular error θϵ: compute only when both norms are non-trivial.
    pred_norm = _l2_norm(pred, eps=float(eps))
    gt_norm = _l2_norm(gt_flow, eps=float(eps))
    angle_mask = (pred_norm > 1e-6) & (gt_norm > 1e-6)
    if np.any(angle_mask):
        dot = np.sum(pred[angle_mask] * gt_flow[angle_mask], axis=-1)
        denom = (pred_norm[angle_mask] * gt_norm[angle_mask]).clip(min=float(eps))
        cos = np.clip(dot / denom, -1.0, 1.0)
        theta_e = float(np.mean(np.arccos(cos)))
    else:
        theta_e = 0.0

    # 3-way EPE
    masks = {
        "BS": gt.bs_mask[valid],
        "FS": gt.fs_mask[valid],
        "FD": gt.fd_mask[valid],
    }
    epe_3way: Dict[str, float] = {}
    epe_3way_count: Dict[str, int] = {}
    for name, mask in masks.items():
        mask_bool = mask.astype(bool, copy=False)
        epe_3way_count[name] = int(np.count_nonzero(mask_bool))
        epe_3way[name] = _safe_mean(epe[mask_bool])

    # Bucketed normalized EPE
    gt_mag = _l2_norm(gt_flow, eps=float(eps))
    denom = np.maximum(gt_mag, float(denom_min_m)).astype(np.float32, copy=False)
    nepe = (epe / denom).astype(np.float32, copy=False)
    bucketed: Dict[str, float] = {}
    bucketed_count: Dict[str, int] = {}
    bucket_idx_valid = gt.bucket_indices.astype(np.int32, copy=False)[valid]
    for bucket_index, bucket_name in enumerate(gt.bucket_names):
        mask = bucket_idx_valid == int(bucket_index)
        bucketed_count[bucket_name] = int(np.count_nonzero(mask))
        bucketed[bucket_name] = _safe_mean(nepe[mask])

    return SceneFlowMetrics(
        epe3d=epe3d,
        acc5=acc5,
        acc10=acc10,
        theta_e=theta_e,
        epe_3way=epe_3way,
        epe_3way_count=epe_3way_count,
        bucketed_nepe=bucketed,
        bucketed_nepe_count=bucketed_count,
    )


def build_default_label_to_bucket_map() -> Dict[str, str]:
    """构建一个默认的 PandaSet label->bucket 映射。

    Returns:
        label->bucket 映射。

    Note:
        PandaSet 的 label 集合较多，且不同版本可能存在细微差异。本映射采用保守策略：
        - 车辆类尽量归到 vehicle；
        - 行人归到 pedestrian；
        - 两轮/轻型出行工具归到 cyclist；
        - 其余归到 other。
    """
    vehicle = {
        "Car",
        "Pickup Truck",
        "Medium-sized Truck",
        "Semi-truck",
        "Bus",
        "Truck",
        "Other Vehicle",
    }
    pedestrian = {"Pedestrian"}
    cyclist = {"Bicycle", "Motorcycle", "Personal Mobility Device", "Cyclist"}

    mapping: Dict[str, str] = {}
    for label in vehicle:
        mapping[label] = "vehicle"
    for label in pedestrian:
        mapping[label] = "pedestrian"
    for label in cyclist:
        mapping[label] = "cyclist"
    return mapping


def merge_bucket_stats(
    total: MutableMapping[str, float],
    total_count: MutableMapping[str, int],
    values: Mapping[str, float],
    counts: Mapping[str, int],
) -> None:
    """将分桶统计累加到全局字典（用于跨 clip 聚合）。

    Args:
        total: bucket->sum(nepe) 或 sum(metric)。
        total_count: bucket->count。
        values: bucket->mean(metric)。
        counts: bucket->count。

    Note:
        该函数假设 `values` 是“均值”，因此会按 `mean * count` 累加到 total。
    """
    for bucket, count in counts.items():
        c = int(count)
        if c <= 0:
            continue
        total[bucket] = float(total.get(bucket, 0.0)) + float(values.get(bucket, 0.0)) * float(c)
        total_count[bucket] = int(total_count.get(bucket, 0)) + c

