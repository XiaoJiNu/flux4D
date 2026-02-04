#!/usr/bin/env python3
"""阶段3：tiny clip overfit 训练入口脚本。

该脚本用于跑通最小训练闭环（不追求训练效率/完整功能）：
PandaSet Index -> Lift(G_init) -> Flux4D-base -> gsplat 渲染 -> loss -> 反向传播
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flux4d.engine.trainer import Stage3OverfitArgs, train_stage3_overfit  # noqa: E402


def _load_cfg(path: str) -> Dict[str, object]:
    """加载配置文件中的 `cfg` 字典。

    Args:
        path: 配置文件路径（如 configs/flux4d.py）。

    Returns:
        配置字典。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: 配置文件缺少 `cfg` 或格式非法。
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    spec = importlib.util.spec_from_file_location("flux4d_cfg", str(config_path))
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载配置模块: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, "cfg", None)
    if not isinstance(cfg, dict):
        raise ValueError("配置文件中未找到 dict 类型的 cfg 变量")
    return cfg


def _get_data_root(cfg: Mapping[str, object], override: str) -> str:
    """解析 data_root。"""
    if override:
        return override
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("cfg['data'] 缺失或格式非法")
    data_root = data_cfg.get("data_root")
    if not isinstance(data_root, str) or not data_root:
        raise ValueError("cfg['data']['data_root'] 缺失或为空")
    return data_root


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Stage3 tiny overfit trainer.")
    parser.add_argument("--config", default="configs/flux4d.py", help="Path to configs/flux4d.py")
    parser.add_argument(
        "--index-path",
        default="",
        help="Override clip index PKL path (default uses cfg['data']['index_tiny']).",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Override PandaSet root (default uses cfg['data']['data_root']).",
    )
    parser.add_argument("--clip-index", type=int, default=0, help="Clip index in PKL.")
    parser.add_argument("--camera", default="front_camera", help="Camera name to supervise.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0.")
    parser.add_argument("--iters", type=int, default=0, help="Override train iters (0 uses cfg['train']).")
    parser.add_argument(
        "--fixed-target-frame",
        type=int,
        default=-1,
        help=(
            "固定 overfit 的 target 帧索引（非负数表示 clip 内的 frame_index）。"
            "当为负数时，训练将从 target_frame_indices 中随机采样。"
        ),
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch size = grad_accum_steps).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Override log interval (0 uses cfg['train']).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Override save interval (0 uses cfg['train']).",
    )
    parser.add_argument(
        "--save-ckpt-every",
        type=int,
        default=-1,
        help="Checkpoint interval. -1 uses cfg['train']['save_ckpt_every']; 0 saves only the final ckpt.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Override output dir (default uses cfg['train']['output_dir']).",
    )
    parser.add_argument("--resume-from", default="", help="Resume from a checkpoint path (ckpt_step_*.pt).")
    parser.add_argument(
        "--resume-no-optim",
        action="store_true",
        help="Resume from a checkpoint but do not load optimizer/RNG state (useful when model params changed).",
    )
    parser.add_argument(
        "--num-sky-points",
        type=int,
        default=-1,
        help="Override sky points (negative uses cfg['init']['sky']).",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=-1,
        help="Override max gaussians (negative uses cfg['init']['max_gaussians']).",
    )
    parser.add_argument(
        "--no-projected-depth",
        action="store_true",
        help="Disable projected LiDAR depth supervision (stage3 depth term).",
    )
    return parser


def main() -> int:
    """脚本入口。"""
    args = build_arg_parser().parse_args()
    cfg = _load_cfg(args.config)

    data_root = _get_data_root(cfg, args.data_root)
    data_cfg = cfg.get("data")
    train_cfg = cfg.get("train")
    init_cfg = cfg.get("init")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("cfg['data'] 缺失或格式非法")
    if not isinstance(train_cfg, Mapping):
        raise ValueError("cfg['train'] 缺失或格式非法")
    if not isinstance(init_cfg, Mapping):
        raise ValueError("cfg['init'] 缺失或格式非法")

    index_path = args.index_path or str(data_cfg.get("index_tiny", ""))
    if not index_path:
        raise ValueError("无法解析 index_path，请使用 --index-path 或在 cfg 中设置 data.index_tiny")

    iters = int(args.iters) if args.iters > 0 else int(train_cfg.get("iters", 10_000))
    log_every = int(args.log_every) if args.log_every > 0 else int(train_cfg.get("log_every", 20))
    save_every = int(args.save_every) if args.save_every > 0 else int(train_cfg.get("save_every", 200))
    if int(args.save_ckpt_every) >= 0:
        save_ckpt_every = int(args.save_ckpt_every)
    else:
        save_ckpt_every = int(train_cfg.get("save_ckpt_every", 0))
    output_dir = args.output_dir or str(train_cfg.get("output_dir", "assets/vis/stage3_overfit"))

    fixed_target_frame_index: Optional[int]
    if int(args.fixed_target_frame) >= 0:
        fixed_target_frame_index = int(args.fixed_target_frame)
    else:
        fixed_target_frame_index = None

    sky_cfg = init_cfg.get("sky")
    num_sky_points: int
    if args.num_sky_points >= 0:
        num_sky_points = int(args.num_sky_points)
    elif isinstance(sky_cfg, Mapping) and bool(sky_cfg.get("enabled", True)):
        num_sky_points = int(sky_cfg.get("num_points", 0))
    else:
        num_sky_points = 0

    max_gaussians: Optional[int]
    if args.max_gaussians >= 0:
        max_gaussians = int(args.max_gaussians)
    else:
        value = init_cfg.get("max_gaussians")
        max_gaussians = int(value) if isinstance(value, int) else None

    train_args = Stage3OverfitArgs(
        index_path=index_path,
        clip_index=int(args.clip_index),
        data_root=data_root,
        camera_name=str(args.camera),
        device=str(args.device),
        iters=iters,
        fixed_target_frame_index=fixed_target_frame_index,
        grad_accum_steps=int(args.grad_accum_steps),
        log_every=log_every,
        save_every=save_every,
        output_dir=output_dir,
        num_sky_points=num_sky_points,
        max_gaussians=max_gaussians,
        use_projected_lidar_depth=not bool(args.no_projected_depth),
        resume_from=str(args.resume_from),
        resume_optimizer=not bool(args.resume_no_optim),
        save_ckpt_every=save_ckpt_every,
    )
    train_stage3_overfit(cfg, train_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
