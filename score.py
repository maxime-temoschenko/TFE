# https://github.com/francois-rozet/sda/blob/qg/sda/score.py

r"""Score modules"""

import math
import torch
import torch.nn as nn

from torch import Size, Tensor
from tqdm import tqdm
from typing import *


from nn import *

# https://github.com/probabilists/zuko/blob/master/zuko/utils.py
def broadcast(*tensors: Tensor, ignore: Union[int, Sequence[int]] = 0) -> List[Tensor]:
    r"""Broadcasts tensors together.

    The term broadcasting describes how PyTorch treats tensors with different shapes
    during arithmetic operations. In short, if possible, dimensions that have
    different sizes are expanded (without making copies) to be compatible.

    Arguments:
        tensors: The tensors to broadcast.
        ignore: The number(s) of dimensions not to broadcast.

    Returns:
        The broadcasted tensors.

    Example:
        >>> x = torch.rand(3, 1, 2)
        >>> y = torch.rand(4, 5)
        >>> x, y = broadcast(x, y, ignore=1)
        >>> x.shape
        torch.Size([3, 4, 2])
        >>> y.shape
        torch.Size([3, 4, 5])
    """

    if isinstance(ignore, int):
        ignore = [ignore] * len(tensors)

    dims = [t.dim() - i for t, i in zip(tensors, ignore)]
    common = torch.broadcast_shapes(*(t.shape[:i] for t, i in zip(tensors, dims)))

    return [torch.broadcast_to(t, common + t.shape[i:]) for t, i in zip(tensors, dims)]

class TimeEmbedding(nn.Sequential):
    r"""Creates a time embedding.

    Arguments:
        features: The number of embedding features.
    """

    def __init__(self, features: int):
        super().__init__(
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, features),
        )

        self.register_buffer('freqs', torch.pi * torch.arange(1, 32 + 1))

    def forward(self, t: Tensor) -> Tensor:
        t = self.freqs * t.unsqueeze(dim=-1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        return super().forward(t)


class ScoreNet(nn.Module):
    r"""Creates a score network.

    Arguments:
        features: The number of features.
        context: The number of context features.
        embedding: The number of time embedding features.
    """

    def __init__(self, features: int, context: int = 0, embedding: int = 16, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        self.network = ResMLP(features + context + embedding, features, **kwargs)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        t = self.embedding(t)

        if c is None:
            x, t = broadcast(x, t, ignore=1)
            x = torch.cat((x, t), dim=-1)
        else:
            x, t, c = broadcast(x, t, c, ignore=1)
            x = torch.cat((x, t, c), dim=-1)

        return self.network(x)


class ScoreUNet(nn.Module):
    r"""Creates a U-Net score network.

    Arguments:
        channels: The number of channels.
        context: The number of context channels.
        embedding: The number of time embedding features.
    """

    def __init__(self, channels: int, context: int = 0, embedding: int = 64, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        self.network = UNet(channels + context, channels, embedding, **kwargs)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        dims = self.network.spatial + 1

        if c is None:
            y = x
        else:
            y = torch.cat(broadcast(x, c, ignore=dims), dim=-dims)

        y = y.reshape(-1, *y.shape[-dims:])
        t = t.reshape(-1)
        t = self.embedding(t)

        return self.network(y, t).reshape(x.shape)


class MCScoreWrapper(nn.Module):
    r"""Disguises a `ScoreUNet` as a score network for a Markov chain."""

    def __init__(self, score: nn.Module):
        super().__init__()

        self.score = score

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W)
        t: Tensor,  # ()
        c: Tensor = None,  # TODO
    ) -> Tensor:
        return self.score(x.transpose(1, 2), t, c).transpose(1, 2)


class MCScoreNet(nn.Module):
    r"""Creates a score network for a Markov chain.

    Arguments:
        features: The number of features.
        context: The number of context features.
        order: The order of the Markov chain.
    """

    def __init__(self, features: int, context: int = 0, order: int = 1, **kwargs):
        super().__init__()

        self.order = order

        if kwargs.get('spatial', 0) > 0:
            build = ScoreUNet
        else:
            build = ScoreNet

        self.kernel = build(features * (2 * order + 1), context, **kwargs)
    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W)
        t: Tensor,  # ()
        c: Tensor = None,  # (C', H, W)
    ) -> Tensor:
        x = self.unfold(x, self.order)
        s = self.kernel(x, t, c)
        s = self.fold(s, self.order)

        return s

    @staticmethod
    @torch.jit.script_if_tracing
    def unfold(x: Tensor, order: int) -> Tensor:
        x = x.unfold(1, 2 * order + 1, 1)
        x = x.movedim(-1, 2)
        x = x.flatten(2, 3)

        return x

    @staticmethod
    @torch.jit.script_if_tracing
    def fold(x: Tensor, order: int) -> Tensor:
        x = x.unflatten(2, (2 * order  + 1, -1))

        return torch.cat((
            x[:, 0, :order],
            x[:, :, order],
            x[:, -1, -order:],
        ), dim=1)


class VPSDE(nn.Module):
    r"""Creates a noise scheduler for the variance preserving (VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = 1 - \alpha(t)^2 + \eta^2

    Arguments:
        eps: A noise estimator :math:`\epsilon_\phi(x, t)`.
        shape: The event shape.
        alpha: The choice of :math:`\alpha(t)`.
        eta: A numerical stability term.
    """

    def __init__(
        self,
        eps: nn.Module,
        shape: Size,
        alpha: str = 'cos',
        eta: float = 1e-3,
    ):
        super().__init__()

        self.eps = eps
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))
        self.eta = eta

        if alpha == 'lin':
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == 'cos':
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == 'exp':
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError()

        self.register_buffer('device', torch.empty(()))

    def mu(self, t: Tensor) -> Tensor:
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.alpha(t) ** 2 + self.eta ** 2).sqrt()

    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        r"""Samples from the perturbation kernel :math:`p(x(t) | x)`."""

        t = t.reshape(t.shape + (1,) * len(self.shape))

        eps = torch.randn_like(x)
        x = self.mu(t) * x + self.sigma(t) * eps

        if train:
            return x, eps
        else:
            return x
    def denoise(self, x_t : Tensor, t: Tensor, c:Tensor = None) -> Tensor:
        with torch.no_grad():
            t = t.reshape(t.shape + (1,)*len(self.shape))
            noise = self.eps(x_t,t,c)
            x_0 = (x_t - self.sigma(t) * noise)/self.mu(t)

            return x_0

    def ddpm_sample(self, shape: torch.Size = (), c: torch.Tensor = None, steps: int = 1000):
        x = torch.randn(shape + self.shape).to(self.device)
        x = x.reshape(-1, *self.shape)
        time = torch.linspace(1, 0, steps + 1).to(self.device)
        alpha = self.alpha(time)
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
        with torch.no_grad():
            for i, t in tqdm(enumerate(time), ncols=88):
                alpha_t = alpha[i]
                sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod[i]
                eps = self.eps(x, t, c)
                if t == 0:
                    z = 0
                else:
                    z = torch.randn_like(x)
                sigma_t = self.sigma(t)
                x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * eps) + sigma_t * z
        return x
    def sample(
        self,
        mask: Tensor,
        shape: Size = (),
        c: Tensor = None,
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            c: The optional context.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """

        x = torch.randn(shape + self.shape).to(self.device)
        x = x.reshape(-1, *self.shape)

        time = torch.linspace(1, 0, steps + 1).to(self.device)
        dt = 1 / steps

        with torch.no_grad():
            for t in tqdm(time[:-1], ncols=88):
                # Predictor
                r = self.mu(t - dt) / self.mu(t)
                eps = self.eps(x,t,c)
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * eps

                # Corrector
                for _ in range(corrections):
                    z = torch.randn_like(x)
                    eps = self.eps(x, t - dt, c)
                    delta = tau / eps.square().mean(dim=self.dims, keepdim=True)

                    x = x - (delta * eps + torch.sqrt(2 * delta) * z) * self.sigma(t - dt)
                x = x * mask
        return x.reshape(shape + self.shape)

    def loss(self, x: Tensor, c: Tensor = None, w: Tensor = None) -> Tensor:
        r"""Returns the denoising loss."""

        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device) # (BATCH_SIZE)
        x, eps = self.forward(x, t, train=True)

        err = (self.eps(x, t, c) - eps).square()

        if w is None:
            return err.mean()
        else:
            return (err * w).mean() / w.mean()


class SubVPSDE(VPSDE):
    r"""Creates a noise scheduler for the sub-variance preserving (sub-VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = (1 - \alpha(t)^2 + \eta)^2
    """

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - self.alpha(t) ** 2 + self.eta


class SubSubVPSDE(VPSDE):
    r"""Creates a noise scheduler for the sub-sub-VP SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = (1 - \alpha(t) + \eta)^2
    """

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - self.alpha(t) + self.eta


class DPSGaussianScore(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    .. math:: p(y | x) = N(y | A(x), Σ)

    References:
        | Diffusion Posterior Sampling for General Noisy Inverse Problems (Chung et al., 2022)
        | https://arxiv.org/abs/2209.14687

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`.
    """

    def __init__(
        self,
        y: Tensor,
        mask : Tensor,
        A: Callable[[Tensor], Tensor],
        sde: VPSDE,
        zeta: float = 1.0,
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('mask', mask)
        self.A = A
        self.sde = sde
        self.zeta = zeta

    def forward(self, x: Tensor, t: Tensor, c : Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.eps(x, t,c)
            x_ = (x - sigma * eps) / mu
            x_ = x_ * self.mask
            err = ( (self.y*self.mask) - (self.A(x_)*self.mask)).square().sum()

        s, = torch.autograd.grad(err, x)
        s = -s * self.zeta / err.sqrt()

        return (eps - sigma * s)*self.mask


class GaussianScore(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    .. math:: p(y | x) = N(y | A(x), Σ)

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`.
    """

    def __init__(
        self,
        y: Tensor,
        mask : Tensor,
        A: Callable[[Tensor], Tensor],
        std: Union[float, Tensor],
        sde: VPSDE,
        gamma: Union[float, Tensor] = 1e-2,
        detach: bool = False,
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('std', torch.as_tensor(std))
        self.register_buffer('gamma', torch.as_tensor(gamma))
        self.register_buffer('mask', mask)
        self.A = A
        self.sde = sde
        self.detach = detach

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        if self.detach:
            eps = self.sde.eps(x, t, c)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            if not self.detach:
                eps = self.sde.eps(x, t, c)
            mask = self.mask
            x_ = (x - sigma * eps) / mu
            x_ = x_ * mask
            err = (self.y - self.A(x_))*mask
            var = self.std ** 2 + self.gamma * (sigma / mu) ** 2

            log_p = -(err ** 2 / var).sum() / 2

        s, = torch.autograd.grad(log_p, x)

        return (eps*mask) - sigma * (s*mask)
