"""Flux4D 统一实验配置（单文件，按模块分组）。

该文件承载全流程可调参数：数据、初始化（Lift）、坐标系、体素化、模型、渲染、损失、训练与评测。
配置以 Python 表达，支持派生变量（例如由点云范围与体素大小计算体素网格尺寸），便于快速迭代。
"""

from __future__ import annotations

from typing import Any, Dict, List


def _compute_voxel_shape(point_cloud_range: List[float], voxel_size: List[float]) -> List[int]:
    """根据点云范围与体素大小计算体素网格分辨率。

    Args:
        point_cloud_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]（米）。
        voxel_size: 体素大小 [vx, vy, vz]（米）。

    Returns:
        体素网格分辨率 [nx, ny, nz]。

    Raises:
        ValueError: 输入形状非法或体素大小非正。
    """
    if len(point_cloud_range) != 6:
        raise ValueError("point_cloud_range 必须为长度 6 的列表")
    if len(voxel_size) != 3:
        raise ValueError("voxel_size 必须为长度 3 的列表")
    if any(size <= 0 for size in voxel_size):
        raise ValueError("voxel_size 必须全部为正数")

    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx, vy, vz = voxel_size
    return [
        int((x_max - x_min) / vx),
        int((y_max - y_min) / vy),
        int((z_max - z_min) / vz),
    ]


data_root = "/home/yr/yr/data/automonous/pandaset"
index_full = "data/metadata/pandaset_full_clips.pkl"
index_tiny = "data/metadata/pandaset_tiny_clips.pkl"

fps = 10.0

snippet_interpolation_1s = dict(
    fps=fps,
    num_frames=11,
    overlap_frames=5,
    input_frame_ids=[0, 2, 4, 6, 8, 10],
    target_frame_ids=[1, 3, 5, 7, 9],
)

snippet_future_1p5s = dict(
    fps=fps,
    num_frames=16,
    overlap_frames=5,
    input_frame_ids=[0, 2, 4, 6, 8, 10],
    target_frame_ids=[1, 3, 5, 7, 9, 11, 12, 13, 14, 15],
)

point_cloud_range = [-60.0, -60.0, -6.0, 60.0, 60.0, 6.0]
voxel_size = [0.30, 0.30, 0.30]
voxel_shape = _compute_voxel_shape(point_cloud_range, voxel_size)

cfg: Dict[str, Any] = dict(
    data=dict(
        dataset="pandaset",
        data_root=data_root,
        index_full=index_full,
        index_tiny=index_tiny,
        clip_index=0,
        lidar_sensor_id=-1,
        views=dict(
            strategy="all",
            num_views=None,
            camera_names=None,
        ),
        snippets=dict(
            interpolation_1s=snippet_interpolation_1s,
            future_1p5s=snippet_future_1p5s,
        ),
        index_build=dict(
            target_fps=fps,
            clip_len_s=1.5,
            stride_s=1.5,
            include_endpoint=True,
            val_num_scenes=10,
            tiny_num_scenes=2,
            strict=True,
        ),
    ),
    coord=dict(
        ego0_frame_index=0,
        position_frame="ego0",
        velocity_frame="world",
        lidar_points_frame="world",
        pose_convention="sensor_to_world",
        transform_gaussian_rotation=True,
    ),
    init=dict(
        opacity_init=0.5,
        scale_knn_k=3,
        scale_transform=dict(
            use_log=True,
            use_softplus=True,
            softplus_beta=1.0,
            eps=1e-6,
        ),
        quat_init=dict(
            mode="random_normalized",
        ),
        time_norm=dict(
            mode="per_clip_minmax_0_1",
            eps=1e-8,
        ),
        downsample=dict(
            enabled=True,
            voxel_size_m=0.2,
        ),
        max_gaussians=300_000,
        sky=dict(
            enabled=True,
            num_points=1_000_000,
            sphere_radius_scale=4.0,
            upper_half=True,
            color_rgb=[0.5, 0.5, 0.5],
            opacity=0.2,
        ),
    ),
    voxel=dict(
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        voxel_shape=voxel_shape,
        pooling="mean",
    ),
    model=dict(
        backend="spconv",
        gaussian_dim=14,
        time_dim=1,
        unet=dict(
            base_channels=32,
            channels=[32, 64, 96, 128],
            num_blocks_per_level=2,
            out_channels=128,
            act="leaky_relu",
            norm="batchnorm1d",
        ),
        head=dict(
            delta_g_dim=14,
            motion=dict(
                poly_degree_l=0,
            ),
        ),
        iterative_refine=dict(
            enabled=False,
            num_iters=3,
        ),
    ),
    render=dict(
        renderer="gsplat",
        mode="RGB+ED",
        near=0.1,
        far=200.0,
        activation=dict(
            color="sigmoid",
            opacity="sigmoid",
            scale_clamp_m=[0.0, 1.0],
        ),
        rendered_velocity=dict(
            clip_mag_px=10.0,
            eps=1e-6,
            delta_t_mode="next_frame",
        ),
    ),
    loss=dict(
        lambda_rgb=0.8,
        lambda_ssim=0.2,
        lambda_depth=0.01,
        lambda_vel=5e-3,
        ssim_window=11,
        depth=dict(
            use_projected_lidar_depth=True,
        ),
    ),
    train=dict(
        iters=10_000,
        optimizer=dict(
            type="Adam",
            lr=1e-3,
            weight_decay=0.0,
        ),
        lr_schedule=dict(
            type="exp_decay",
            warmup_steps=1000,
        ),
        seed=42,
        log_every=20,
        save_every=200,
        save_ckpt_every=1000,
        output_dir="assets/vis/stage3_overfit",
    ),
)
