"""阶段3：稀疏 3D U-Net（spconv 后端）。

补充材料 Fig. A1 描述了稀疏 3D U-Net 主干：Conv3D -> BatchNorm1d -> LeakyReLU，
并通过 skip 连接（channel concat）形成 U-Net 结构。

该实现以 spconv 为后端；当 spconv 不可用时，会在构建模型时抛出明确异常。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> None:
    """确保 torch 可用。"""
    if torch is None or nn is None:  # pragma: no cover
        raise ModuleNotFoundError("缺少 torch：请在训练环境中安装 PyTorch")


def _require_spconv() -> object:
    """确保 spconv 可用并返回模块对象。"""
    try:
        import spconv.pytorch as spconv  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("缺少 spconv：阶段3需要 spconv 后端来构建稀疏 3D U-Net") from exc
    return spconv


def _replace_sparse_feature(tensor: object, features: "torch.Tensor") -> object:
    """替换 spconv SparseConvTensor 的 features（兼容不同版本）。"""
    if hasattr(tensor, "replace_feature"):
        return tensor.replace_feature(features)  # type: ignore[attr-defined]
    setattr(tensor, "features", features)
    return tensor


class SparseBNAct(nn.Module):
    """对 SparseConvTensor 的 features 应用 BN + LeakyReLU。"""

    def __init__(self, num_channels: int, negative_slope: float = 0.1) -> None:
        super().__init__()
        self._bn = nn.BatchNorm1d(num_channels)
        self._act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: object) -> object:  # noqa: D401 - 简短说明即可
        """前向：对 x.features 做 BN + 激活。"""
        feats = getattr(x, "features")
        feats = self._act(self._bn(feats))
        return _replace_sparse_feature(x, feats)


class SparseConvBlock(nn.Module):
    """稀疏卷积基本块：Conv3D -> BN -> LeakyReLU。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        subm: bool = True,
        indice_key: Optional[str] = None,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        _require_torch()
        spconv = _require_spconv()

        if subm:
            conv = spconv.SubMConv3d(  # type: ignore[attr-defined]
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
                indice_key=indice_key,
            )
        else:
            conv = spconv.SparseConv3d(  # type: ignore[attr-defined]
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                indice_key=indice_key,
            )
        self._conv = conv
        self._bn_act = SparseBNAct(out_channels, negative_slope=negative_slope)

    def forward(self, x: object) -> object:  # noqa: D401
        """前向：稀疏卷积并对 features 做 BN+激活。"""
        x = self._conv(x)
        return self._bn_act(x)


class SparseInverseConvBlock(nn.Module):
    """上采样块：InverseConv3D -> BN -> LeakyReLU。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        indice_key: str,
        kernel_size: int = 2,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        _require_torch()
        spconv = _require_spconv()
        self._conv = spconv.SparseInverseConv3d(  # type: ignore[attr-defined]
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            indice_key=indice_key,
            bias=False,
        )
        self._bn_act = SparseBNAct(out_channels, negative_slope=negative_slope)

    def forward(self, x: object) -> object:  # noqa: D401
        """前向：逆卷积上采样并对 features 做 BN+激活。"""
        x = self._conv(x)
        return self._bn_act(x)


@dataclass(frozen=True)
class SparseUNetConfig:
    """Sparse 3D U-Net 配置。"""

    base_channels: int
    channels: List[int]
    num_blocks_per_level: int
    out_channels: int
    negative_slope: float = 0.1


class SparseUNet3D(nn.Module):
    """基于 spconv 的稀疏 3D U-Net。"""

    def __init__(self, in_channels: int, cfg: SparseUNetConfig) -> None:
        super().__init__()
        _require_torch()
        _require_spconv()

        if not cfg.channels:
            raise ValueError("cfg.channels 不能为空")
        if cfg.channels[0] != cfg.base_channels:
            raise ValueError("cfg.base_channels 必须与 cfg.channels[0] 一致，便于结构对齐")

        self._cfg = cfg

        self._stem = SparseConvBlock(
            in_channels,
            cfg.channels[0],
            kernel_size=3,
            padding=1,
            subm=True,
            indice_key="subm0",
            negative_slope=cfg.negative_slope,
        )

        enc_blocks: List[nn.Module] = []
        downs: List[nn.Module] = []
        for level, (cin, cout) in enumerate(zip(cfg.channels[:-1], cfg.channels[1:])):
            blocks: List[nn.Module] = []
            for block_id in range(cfg.num_blocks_per_level):
                blocks.append(
                    SparseConvBlock(
                        cin,
                        cin,
                        kernel_size=3,
                        padding=1,
                        subm=True,
                        indice_key=f"subm{level}",
                        negative_slope=cfg.negative_slope,
                    )
                )
            enc_blocks.append(nn.Sequential(*blocks))
            downs.append(
                SparseConvBlock(
                    cin,
                    cout,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    subm=False,
                    indice_key=f"down{level}",
                    negative_slope=cfg.negative_slope,
                )
            )
        self._enc_blocks = nn.ModuleList(enc_blocks)
        self._downs = nn.ModuleList(downs)

        bottleneck: List[nn.Module] = []
        last_channels = cfg.channels[-1]
        for _ in range(cfg.num_blocks_per_level):
            bottleneck.append(
                SparseConvBlock(
                    last_channels,
                    last_channels,
                    kernel_size=3,
                    padding=1,
                    subm=True,
                    indice_key=f"subm{len(cfg.channels) - 1}",
                    negative_slope=cfg.negative_slope,
                )
            )
        self._bottleneck = nn.Sequential(*bottleneck)

        ups: List[nn.Module] = []
        dec_blocks: List[nn.Module] = []
        for level in reversed(range(len(cfg.channels) - 1)):
            cin = cfg.channels[level + 1]
            cout = cfg.channels[level]
            ups.append(
                SparseInverseConvBlock(
                    cin,
                    cout,
                    indice_key=f"down{level}",
                    kernel_size=2,
                    negative_slope=cfg.negative_slope,
                )
            )
            blocks: List[nn.Module] = []
            in_after_concat = cout + cout
            blocks.append(
                SparseConvBlock(
                    in_after_concat,
                    cout,
                    kernel_size=3,
                    padding=1,
                    subm=True,
                    indice_key=f"subm{level}",
                    negative_slope=cfg.negative_slope,
                )
            )
            for _ in range(cfg.num_blocks_per_level - 1):
                blocks.append(
                    SparseConvBlock(
                        cout,
                        cout,
                        kernel_size=3,
                        padding=1,
                        subm=True,
                        indice_key=f"subm{level}",
                        negative_slope=cfg.negative_slope,
                    )
                )
            dec_blocks.append(nn.Sequential(*blocks))
        self._ups = nn.ModuleList(ups)
        self._dec_blocks = nn.ModuleList(dec_blocks)

        spconv = _require_spconv()
        self._out_conv = spconv.SubMConv3d(  # type: ignore[attr-defined]
            cfg.channels[0],
            cfg.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            indice_key="subm0",
        )

    def forward(self, x: object) -> object:  # noqa: D401
        """前向：U-Net 编解码并输出稀疏体素特征。"""
        x = self._stem(x)

        skips: List[object] = []
        for enc, down in zip(self._enc_blocks, self._downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self._bottleneck(x)

        for up, dec, skip in zip(self._ups, self._dec_blocks, reversed(skips)):
            x = up(x)
            x = _replace_sparse_feature(x, torch.cat([getattr(x, "features"), getattr(skip, "features")], dim=1))
            x = dec(x)

        x = self._out_conv(x)
        return x


def build_sparse_unet(in_channels: int, cfg_dict: dict) -> SparseUNet3D:
    """从配置字典构建 SparseUNet3D。

    Args:
        in_channels: 输入特征维度（阶段3为 14+1=15）。
        cfg_dict: `configs/flux4d.py` 中 `cfg["model"]["unet"]` 对应字典。

    Returns:
        SparseUNet3D 实例。
    """
    cfg = SparseUNetConfig(
        base_channels=int(cfg_dict["base_channels"]),
        channels=[int(v) for v in cfg_dict["channels"]],
        num_blocks_per_level=int(cfg_dict["num_blocks_per_level"]),
        out_channels=int(cfg_dict["out_channels"]),
        negative_slope=0.1,
    )
    return SparseUNet3D(in_channels=in_channels, cfg=cfg)

