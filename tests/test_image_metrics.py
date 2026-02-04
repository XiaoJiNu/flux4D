"""图像指标（PSNR）最小单元测试（torch 可选）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux4d.metrics.image_metrics import compute_psnr_torch  # noqa: E402


@pytest.mark.skipif(torch is None, reason="torch 未安装，跳过 PSNR 单测")
def test_compute_psnr_mask_all_true_matches_unmasked() -> None:
    """验证 mask 全 True 时，masked/unmasked 的 PSNR 口径一致。"""
    pred = torch.zeros((2, 2, 3), dtype=torch.float32)
    target = torch.zeros((2, 2, 3), dtype=torch.float32)
    target[..., 0] = 1.0

    psnr_full = compute_psnr_torch(pred, target)
    psnr_masked = compute_psnr_torch(pred, target, valid_mask=torch.ones((2, 2), dtype=torch.bool))

    assert torch.allclose(psnr_full, psnr_masked, atol=1e-6)


@pytest.mark.skipif(torch is None, reason="torch 未安装，跳过 PSNR 单测")
def test_compute_psnr_raises_on_empty_mask() -> None:
    """验证空 mask 会显式报错，避免 silent bug（PSNR 虚高）。"""
    pred = torch.zeros((2, 2, 3), dtype=torch.float32)
    target = torch.zeros((2, 2, 3), dtype=torch.float32)
    empty = torch.zeros((2, 2), dtype=torch.bool)
    with pytest.raises(ValueError):
        _ = compute_psnr_torch(pred, target, valid_mask=empty)

