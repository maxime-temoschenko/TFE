# https://github.com/francois-rozet/sda/blob/qg/sda/utils.py#L58

r"""Neural networks"""

import torch
import torch.nn as nn
import math

from torch import Tensor
from typing import *


# https://github.com/probabilists/zuko/blob/master/zuko/nn.py
class LayerNorm(nn.Module):
    r"""Creates a normalization layer that standardizes features along a dimension.

    .. math:: y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}}

    References:
       Layer Normalization (Lei Ba et al., 2016)
       https://arxiv.org/abs/1607.06450

    Arguments:
        dim: The dimension(s) to standardize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: Union[int, Iterable[int]] = -1, eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, unbiased=True, dim=self.dim, keepdim=True)

        return (x - mean) / (variance + self.eps).sqrt()


class ResidualBlock(nn.Sequential):
    r"""Creates a residual block."""

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ModResidualBlock(nn.Module):
    r"""Creates a residual block with modulation."""

    def __init__(self, project: nn.Module, residue: nn.Module):
        super().__init__()

        self.project = project
        self.residue = residue

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + self.residue(x + self.project(y))


class ResMLP(nn.Sequential):
    r"""Creates a residual multi-layer perceptron (ResMLP).

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The number of hidden features.
        activation: The activation function constructor.
        kwargs: Keyword arguments passed to :class:`nn.Linear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = nn.ReLU,
        **kwargs,
    ):
        blocks = []

        for before, after in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            if after != before:
                blocks.append(nn.Linear(before, after, **kwargs))

            blocks.append(
                ResidualBlock(
                    LayerNorm(),
                    nn.Linear(after, after, **kwargs),
                    activation(),
                    nn.Linear(after, after, **kwargs),
                )
            )

        super().__init__(*blocks)

        self.in_features = in_features
        self.out_features = out_features


class AttentionBlock(torch.nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, spatial=2):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = LayerNorm(1)
        self.qkv = torch.nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.attention = QKVAttention()
        self.proj_out = torch.nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, emb):
        b, c, *spatial = x.shape
        # print("x", x.shape, torch.isnan(x).any())
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        # print("qkv", qkv.shape, torch.isnan(qkv).any())
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(torch.nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        # print("qkv", qkv.shape, torch.isnan(qkv).any())
        # print("scale", scale)
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        # print("weight", weight.shape, torch.isnan(weight).any())
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        # print("softmax(weight)", weight.shape, torch.isnan(weight).any())
        return torch.einsum("bts,bcs->bct", weight, v)


class UNet(torch.nn.Module):
    r"""Creates a U-Net with modulation.

    References:
        | U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
        | https://arxiv.org/abs/1505.04597

    Arguments:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        mod_features: The number of modulation features.
        hidden_channels: The number of hidden channels.
        hidden_blocks: The number of hidden blocks at each depth.
        kernel_size: The size of the convolution kernels.
        stride: The stride of the downsampling convolutions.
        activation: The activation function constructor.
        spatial: The number of spatial dimensions. Can be either 1, 2 or 3.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mod_features: int,
        hidden_channels: Sequence[int] = (32, 64, 128),
        hidden_blocks: Sequence[int] = (2, 3, 5),
        attention_levels: Sequence[int] = [],
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        activation: Callable[[], torch.nn.Module] = torch.nn.ReLU,
        spatial: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial = spatial

        # Components
        convolution = {
            1: torch.nn.Conv1d,
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d,
        }.get(spatial)

        if type(kernel_size) is int:
            kernel_size = [kernel_size] * spatial

        if type(stride) is int:
            stride = [stride] * spatial

        kwargs.update(
            kernel_size=kernel_size,
            padding=[k // 2 for k in kernel_size],
        )

        def block(channels):
            return ModResidualBlock(
                project=torch.nn.Sequential(
                    torch.nn.Linear(mod_features, channels),
                    torch.nn.Unflatten(-1, (-1,) + (1,) * spatial),
                ),
                residue=torch.nn.Sequential(
                    # torch.nn.GroupNorm(num_groups=32, num_channels=channels),
                    LayerNorm(-(spatial + 1)),
                    convolution(channels, channels, **kwargs),
                    activation(),
                    convolution(channels, channels, **kwargs),
                ),
            )

        # Layers
        heads, tails = [], []
        descent, ascent = [], []

        for i, blocks in enumerate(hidden_blocks):
            if i > 0:
                heads.append(
                    torch.nn.Sequential(
                        convolution(
                            hidden_channels[i - 1],
                            hidden_channels[i],
                            stride=stride,
                            **kwargs,
                        ),
                    )
                )

                tails.append(
                    torch.nn.Sequential(
                        # torch.nn.GroupNorm(
                        #     num_groups=32, num_channels=hidden_channels[i]
                        # ),
                        LayerNorm(-(spatial + 1)),
                        torch.nn.Upsample(scale_factor=tuple(stride), mode="nearest"),
                        convolution(
                            hidden_channels[i],
                            hidden_channels[i - 1],
                            **kwargs,
                        ),
                    )
                )
            else:
                heads.append(convolution(in_channels, hidden_channels[i], **kwargs))
                tails.append(convolution(hidden_channels[i], out_channels, **kwargs))

            descent_layers = []
            ascent_layers = []
            for bi in range(blocks):
                descent_layers.append(block(hidden_channels[i]))
                ascent_layers.append(block(hidden_channels[i]))
                if i in attention_levels:
                    descent_layers.append(AttentionBlock(hidden_channels[i]))
                    ascent_layers.append(AttentionBlock(hidden_channels[i]))

            descent.append(torch.nn.ModuleList(descent_layers))
            ascent.append(torch.nn.ModuleList(ascent_layers))

            # descent.append(
            #     torch.nn.ModuleList(block(hidden_channels[i]) for _ in range(blocks))
            # )
            # ascent.append(
            #     torch.nn.ModuleList(block(hidden_channels[i]) for _ in range(blocks))
            # )

        self.heads = torch.nn.ModuleList(heads)
        self.tails = torch.nn.ModuleList(reversed(tails))
        self.descent = torch.nn.ModuleList(descent)
        self.ascent = torch.nn.ModuleList(reversed(ascent))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        memory = []

        for head, blocks in zip(self.heads, self.descent):
            x = head(x)

            for block in blocks:
                x = block(x, y)

            memory.append(x)

        memory.pop()

        for blocks, tail in zip(self.ascent, self.tails):
            for block in blocks:
                x = block(x, y)

            if memory:
                x = tail(x) + memory.pop()
            else:
                x = tail(x)

        return x