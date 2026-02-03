"""Flux4D 训练 checkpoint 保存/加载工具。

该模块提供一个极简 checkpoint 格式，用于阶段3/阶段4的脚手架训练：
- 保存模型与优化器的 state_dict；
- 保存训练步数与随机数状态，便于断点续训。

Note:
    checkpoint 文件通常较大，建议输出到 `assets/vis/` 或用户指定的工作目录，
    并确保被 `.gitignore` 忽略，避免误提交。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Union

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    """确保 torch 可用。"""
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("缺少 torch：请在训练环境（如 gaussianstorm）中安装 PyTorch")


def save_ckpt(state: Mapping[str, object], out_path: Union[str, Path]) -> None:
    """保存 checkpoint 到磁盘（原子写入）。

    Args:
        state: checkpoint 状态字典（通常包含 model/optimizer/step 等字段）。
        out_path: 输出文件路径。

    Raises:
        FileNotFoundError: 输出目录不可用。
        RuntimeError: torch.save 失败。
    """
    _require_torch()
    path = Path(out_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(dict(state), tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def load_ckpt(
    path: Union[str, Path],
    *,
    map_location: Optional[object] = None,
    weights_only: Optional[bool] = None,
) -> Dict[str, object]:
    """从磁盘加载 checkpoint。

    Args:
        path: checkpoint 文件路径。
        map_location: torch.load 的 map_location 参数（可将权重映射到 CPU/GPU）。
        weights_only: 是否启用 PyTorch 的安全加载模式（weights-only）。
            - 当为 None（默认）时，本项目按“训练断点续训”的需求加载完整 checkpoint，
              等价于 `weights_only=False`（PyTorch 2.6 起默认值变更为 True，会导致包含
              NumPy RNG 等状态的 checkpoint 无法加载）。
            - 当为 True 时，仅允许加载安全白名单对象；通常只适用于“只保存 tensor 权重”的 checkpoint。

    Returns:
        checkpoint 状态字典。

    Raises:
        FileNotFoundError: checkpoint 文件不存在。
        ValueError: checkpoint 结构非法。
    """
    _require_torch()
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
    requested_weights_only = bool(weights_only) if weights_only is not None else False
    try:
        payload = torch.load(  # type: ignore[call-arg]
            ckpt_path, map_location=map_location, weights_only=requested_weights_only
        )
    except TypeError:
        # 兼容旧版本 PyTorch：torch.load 可能不支持 weights_only 参数。
        payload = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint 结构非法：期望 dict")
    return payload
