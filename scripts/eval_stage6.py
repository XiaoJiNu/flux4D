#!/usr/bin/env python3
"""阶段6：PandaSet 评测入口脚本（NVS + Scene Flow）。"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flux4d.engine.stage6 import Stage6EvalArgs, eval_stage6  # noqa: E402


def _load_cfg(path: str) -> Dict[str, object]:
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
    parser = argparse.ArgumentParser(description="Stage6 evaluator (NVS + Scene Flow).")
    parser.add_argument("--config", default="configs/flux4d.py", help="Path to configs/flux4d.py")
    parser.add_argument(
        "--index-path",
        default="",
        help="Override clip index PKL path (default uses cfg['data']['index_full']).",
    )
    parser.add_argument("--data-root", default="", help="Override PandaSet root.")
    parser.add_argument("--camera", default="front_camera", help="Camera name to evaluate.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path (ckpt_last.pt or ckpt_step_*.pt).")
    parser.add_argument(
        "--out-dir",
        default="assets/vis/stage6_eval",
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
        "--max-eval-clips",
        type=int,
        default=-1,
        help="Debug: limit number of eval clips (negative means no limit).",
    )
    parser.add_argument("--save-renders", action="store_true", help="Save a few clip renders for sanity check.")
    parser.add_argument(
        "--save-max-clips",
        type=int,
        default=3,
        help="How many clips to save renders for (when --save-renders is set).",
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

    max_eval_clips: Optional[int]
    if int(args.max_eval_clips) >= 0:
        max_eval_clips = int(args.max_eval_clips)
    else:
        max_eval_clips = None

    eval_args = Stage6EvalArgs(
        index_path=str(index_path),
        data_root=str(data_root),
        camera_name=str(args.camera),
        device=str(args.device),
        ckpt_path=str(args.ckpt),
        output_dir=str(args.out_dir),
        cache_dir=str(args.cache_dir),
        cache_max_items=max(0, int(args.cache_max_items)),
        num_sky_points=int(num_sky_points),
        max_gaussians=max_gaussians,
        max_eval_clips=max_eval_clips,
        save_renders=bool(args.save_renders),
        save_max_clips=max(0, int(args.save_max_clips)),
    )
    eval_stage6(cfg, eval_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

