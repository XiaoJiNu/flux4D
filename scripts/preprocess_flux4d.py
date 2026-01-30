#!/usr/bin/env python3
"""PandaSet 索引生成入口脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flux4d.datasets.pandaset_clips import build_pandaset_clip_index  # noqa: E402


def _parse_scene_list(text: str) -> Optional[List[str]]:
    """解析逗号分隔的场景列表。

    Args:
        text: 逗号分隔的场景字符串。

    Returns:
        解析后的场景列表，空字符串则返回 None。
    """
    if not text:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        参数解析器实例。

    Note:
        统一定义索引输出路径、FPS、场景划分等参数。
    """
    parser = argparse.ArgumentParser(description="Build PandaSet clip index PKL files.")
    parser.add_argument(
        "--preset",
        choices=("paper", "debug"),
        default="paper",
        help="Index preset: paper (align to paper/supp) or debug (legacy seconds-based slicing).",
    )
    parser.add_argument(
        "--data-root",
        default="/home/yr/yr/data/automonous/pandaset",
        help="Path to PandaSet root.",
    )
    parser.add_argument(
        "--out-full",
        default="data/metadata/pandaset_full_clips.pkl",
        help="Output PKL for full clips.",
    )
    parser.add_argument(
        "--out-tiny",
        default="data/metadata/pandaset_tiny_clips.pkl",
        help="Output PKL for tiny clips.",
    )
    parser.add_argument(
        "--clip-len-s",
        type=float,
        default=1.5,
        help="Clip length in seconds.",
    )
    parser.add_argument(
        "--stride-s",
        type=float,
        default=1.5,
        help="Stride in seconds.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=10.0,
        help="Target FPS used for clip slicing (set to 0 to use inferred).",
    )
    parser.add_argument(
        "--camera-names",
        type=str,
        default="",
        help="Comma-separated camera names to keep (paper preset defaults to front_camera).",
    )
    parser.add_argument(
        "--tiny-scenes",
        type=str,
        default="",
        help="Comma-separated scene ids for tiny index.",
    )
    parser.add_argument(
        "--tiny-num-scenes",
        type=int,
        default=2,
        help="Number of scenes used when --tiny-scenes is empty.",
    )
    parser.add_argument(
        "--val-scenes",
        type=str,
        default="",
        help="Comma-separated scene ids for test split (paper preset has a fixed default).",
    )
    parser.add_argument(
        "--val-num-scenes",
        type=int,
        default=10,
        help="Number of test scenes when --val-scenes is empty (debug preset only).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional cap on the number of scenes processed.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail on data inconsistencies.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Warn and skip inconsistent scenes.",
    )
    return parser


def main() -> int:
    """脚本入口：生成 full/tiny 索引并输出统计。

    Returns:
        进程退出码，0 表示成功。

    Note:
        target_fps=0 时退化为使用数据实际 FPS。
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    tiny_scenes = _parse_scene_list(args.tiny_scenes)
    val_scenes = _parse_scene_list(args.val_scenes)
    camera_names = _parse_scene_list(args.camera_names)
    # 约定 target_fps=0 表示使用数据实际 FPS
    target_fps = args.target_fps if args.target_fps > 0 else None
    full_count, tiny_count = build_pandaset_clip_index(
        data_root=args.data_root,
        out_pkl_full=args.out_full,
        out_pkl_tiny=args.out_tiny,
        preset=args.preset,
        clip_len_s=args.clip_len_s,
        stride_s=args.stride_s,
        target_fps=target_fps,
        tiny_scenes=tiny_scenes,
        tiny_num_scenes=args.tiny_num_scenes,
        val_scenes=val_scenes,
        val_num_scenes=args.val_num_scenes,
        camera_names=camera_names,
        max_scenes=args.max_scenes,
        strict=args.strict,
    )
    print(f"[done] full clips: {full_count}, tiny clips: {tiny_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
