#!/usr/bin/env python3
"""PKL 索引检查脚本。"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Args:
        无。

    Returns:
        argparse.ArgumentParser: 参数解析器实例。

    Raises:
        无。

    实现要点:
        - 支持指定 PKL 路径、预览 clip 索引以及断点开关。
    """
    parser = argparse.ArgumentParser(description="Inspect a PKL index payload.")
    parser.add_argument(
        "--path",
        default="data/metadata/pandaset_tiny_clips.pkl",
        help="Path to the PKL file.",
    )
    parser.add_argument(
        "--clip-index",
        type=int,
        default=0,
        help="Clip index to preview.",
    )
    parser.add_argument(
        "--breakpoint",
        action="store_true",
        help="Drop into pdb breakpoint after loading.",
    )
    return parser


def _print_summary(payload: Dict[str, Any], clip_index: int) -> None:
    """打印索引概要与指定 clip 的简要信息。

    Args:
        payload (Dict[str, Any]): PKL 索引字典。
        clip_index (int): 预览的 clip 索引。

    Returns:
        None: 无返回值。

    Raises:
        无。

    实现要点:
        - 输出 meta 统计，并对 clip_index 做边界裁剪。
    """
    meta = payload.get("meta", {})
    clips = payload.get("clips", [])
    print("meta keys:", sorted(meta.keys()))
    print("total clips:", meta.get("total_clips", len(clips)))
    print("data root:", meta.get("data_root"))
    print("clip_len_s:", meta.get("clip_len_s"), "stride_s:", meta.get("stride_s"))
    print("target_fps:", meta.get("target_fps"))
    if clips:
        # 保证索引落在有效区间
        clip_index = max(0, min(clip_index, len(clips) - 1))
        clip = clips[clip_index]
        print("preview clip index:", clip_index)
        print("clip_id:", clip.get("clip_id"))
        print("scene_id:", clip.get("scene_id"))
        print("views:", clip.get("views"))
        print("num frames:", len(clip.get("frame_ids", [])))
    else:
        print("no clips found")


def main() -> int:
    """脚本入口：加载 PKL、输出概要并可进入断点。

    Args:
        无。

    Returns:
        int: 进程退出码，0 表示成功。

    Raises:
        FileNotFoundError: PKL 文件不存在。
        pickle.UnpicklingError: 反序列化失败。

    实现要点:
        - 先输出概要，再根据开关进入断点调试。
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"PKL not found: {path}")
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    _print_summary(payload, args.clip_index)

    if args.breakpoint:
        print("Entering breakpoint. Use 'payload' and 'clip' variables.")
        clips = payload.get("clips", [])
        clip = clips[args.clip_index] if clips else None
        breakpoint()  # noqa: T100
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
