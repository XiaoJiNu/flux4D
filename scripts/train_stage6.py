#!/usr/bin/env python3
"""阶段6：PandaSet 全量训练入口脚本。"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flux4d.engine.stage6 import Stage6TrainArgs, train_stage6_full  # noqa: E402


def _load_cfg(path: str) -> Dict[str, object]:
    """加载配置文件中的 `cfg` 字典。"""
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
    parser = argparse.ArgumentParser(description="Stage6 PandaSet full trainer.")
    parser.add_argument("--config", default="configs/flux4d.py", help="Path to configs/flux4d.py")
    parser.add_argument(
        "--index-path",
        default="",
        help="Override clip index PKL path (default uses cfg['data']['index_full']).",
    )
    parser.add_argument("--data-root", default="", help="Override PandaSet root.")
    parser.add_argument("--camera", default="front_camera", help="Camera name to supervise.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0.")
    parser.add_argument("--iters", type=int, default=30000, help="Total training iterations.")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--log-every", type=int, default=20, help="Log interval.")
    parser.add_argument(
        "--save-ckpt-every",
        type=int,
        default=1000,
        help="Checkpoint interval (0 saves only the final ckpt).",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/vis/stage6_train",
        help="Output directory (ignored by git).",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache/lift_stage6",
        help="Lift cache directory (ignored by git).",
    )
    parser.add_argument(
        "--cache-max-items",
        type=int,
        default=2,
        help="Max Lift cache items kept in memory (LRU).",
    )
    parser.add_argument("--resume-from", default="", help="Resume from a checkpoint path.")
    parser.add_argument(
        "--resume-no-optim",
        action="store_true",
        help="Resume from a checkpoint but do not load optimizer state.",
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
        "--max-train-clips",
        type=int,
        default=-1,
        help="Debug: limit the number of training clips (negative means no limit).",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = _load_cfg(args.config)

    data_root = _get_data_root(cfg, args.data_root)
    data_cfg = cfg.get("data")
    init_cfg = cfg.get("init")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("cfg['data'] 缺失或格式非法")
    if not isinstance(init_cfg, Mapping):
        raise ValueError("cfg['init'] 缺失或格式非法")

    index_path = args.index_path or str(data_cfg.get("index_full", ""))
    if not index_path:
        raise ValueError("无法解析 index_path，请使用 --index-path 或在 cfg 中设置 data.index_full")

    sky_cfg = init_cfg.get("sky")
    if args.num_sky_points >= 0:
        num_sky_points = int(args.num_sky_points)
    elif isinstance(sky_cfg, Mapping) and bool(sky_cfg.get("enabled", True)):
        num_sky_points = int(sky_cfg.get("num_points", 0))
    else:
        num_sky_points = 0

    if args.max_gaussians >= 0:
        max_gaussians: Optional[int] = int(args.max_gaussians)
    else:
        value = init_cfg.get("max_gaussians")
        max_gaussians = int(value) if isinstance(value, int) else None

    max_train_clips: Optional[int]
    if int(args.max_train_clips) >= 0:
        max_train_clips = int(args.max_train_clips)
    else:
        max_train_clips = None

    train_args = Stage6TrainArgs(
        index_path=str(index_path),
        data_root=str(data_root),
        camera_name=str(args.camera),
        device=str(args.device),
        iters=int(args.iters),
        grad_accum_steps=max(1, int(args.grad_accum_steps)),
        log_every=max(1, int(args.log_every)),
        save_ckpt_every=max(0, int(args.save_ckpt_every)),
        output_dir=str(args.output_dir),
        cache_dir=str(args.cache_dir),
        cache_max_items=max(0, int(args.cache_max_items)),
        resume_from=str(args.resume_from),
        resume_optimizer=not bool(args.resume_no_optim),
        num_sky_points=int(num_sky_points),
        max_gaussians=max_gaussians,
        max_train_clips=max_train_clips,
    )
    train_stage6_full(cfg, train_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

