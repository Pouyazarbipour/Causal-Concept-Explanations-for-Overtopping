"""Conditional mediator models p(M_j | C_j, X) for counterfactual concept representations.

`ConceptFlow` holds one independent conditional RealNVP per concept (weights batched over concepts).
It is fitted by maximum likelihood on the encoder's *pre-squash* slot activations, so every sample,
after the squash, is a valid concept vector (length in [0, 1)).  With `layers=0` the model reduces to a
conditional diagonal Gaussian, which is the statistical benchmark (and the only option when D = 1,
where a coupling layer cannot split the representation).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class BLinear(nn.Module):
    """J independent linear maps applied to [B, J, in]."""

    def __init__(self, J: int, i: int, o: int, zero: bool = False):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(J, i, o) if zero else torch.randn(J, i, o) / math.sqrt(i))
        self.b = nn.Parameter(torch.zeros(J, o))

    def forward(self, x):
        return torch.einsum("bji,jio->bjo", x, self.w) + self.b


def _net(J, i, h, o):
    return nn.Sequential(BLinear(J, i, h), nn.SiLU(), BLinear(J, h, h), nn.SiLU(), BLinear(J, h, o, zero=True))


class ConceptFlow(nn.Module):
    def __init__(self, J: int, D: int, cond_dim: int, layers: int = 6, hidden: int = 64):
        super().__init__()
        self.J, self.D = J, D
        if D == 1:
            layers = 0
        self.base = _net(J, cond_dim, hidden, 2 * D)
        self.couplings = nn.ModuleList([_net(J, D + cond_dim, hidden, 2 * D) for _ in range(layers)])
        masks = torch.stack([(torch.arange(D) % 2 == (l % 2)).float() for l in range(max(layers, 1))])
        self.register_buffer("masks", masks)
        self.register_buffer("mu", torch.zeros(J, D))
        self.register_buffer("sd", torch.ones(J, D))

    def set_standardisation(self, m):  # m [N, J, D]
        self.mu.copy_(m.mean(0))
        self.sd.copy_(m.std(0).clamp_min(1e-3))

    def nll(self, m, cond):
        """Negative log-likelihood per (sample, concept): [B, J]."""
        z = (m - self.mu) / self.sd
        mu, ls = self.base(cond).chunk(2, -1)
        ls = 3.0 * torch.tanh(ls / 3.0)
        z = (z - mu) * torch.exp(-ls)
        logdet = -ls.sum(-1) - torch.log(self.sd).sum(-1)
        for l, net in enumerate(self.couplings):
            k = self.masks[l]
            s, t = net(torch.cat([z * k, cond], -1)).chunk(2, -1)
            s = torch.tanh(s) * (1 - k)
            z = z * k + (1 - k) * (z - t) * torch.exp(-s)
            logdet = logdet - s.sum(-1)
        return 0.5 * (z ** 2).sum(-1) + 0.5 * self.D * math.log(2 * math.pi) - logdet

    @torch.no_grad()
    def sample(self, cond, generator=None):
        """One draw per (sample, concept): [B, J, D] in pre-squash space."""
        z = torch.randn(cond.shape[0], self.J, self.D, generator=generator)
        for l in reversed(range(len(self.couplings))):
            k = self.masks[l]
            s, t = self.couplings[l](torch.cat([z * k, cond], -1)).chunk(2, -1)
            s = torch.tanh(s) * (1 - k)
            z = z * k + (1 - k) * (z * torch.exp(s) + t)
        mu, ls = self.base(cond).chunk(2, -1)
        ls = 3.0 * torch.tanh(ls / 3.0)
        return (z * torch.exp(ls) + mu) * self.sd + self.mu


def make_cond(labels, x, kmax: int):
    """Conditioning tensor [B, J, kmax + p]: one-hot concept state (zeros if undefined) and inputs."""
    oh = torch.nn.functional.one_hot(labels.clamp_min(0), kmax).float() * (labels >= 0).unsqueeze(-1)
    return torch.cat([oh, x.unsqueeze(1).expand(-1, labels.shape[1], -1)], -1)


def fit_flow(flow, pre, labels, x, kmax, epochs, lr, batch_size, seed, reset_std=True):
    """Maximum-likelihood fit on fixed (detached) pre-squash representations."""
    g = torch.Generator().manual_seed(seed)
    if reset_std:
        flow.set_standardisation(pre)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    hist = []
    mask = (labels >= 0).float()
    for _ in range(epochs):
        perm = torch.randperm(len(pre), generator=g)
        tot = 0.0
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            nll = flow.nll(pre[b], make_cond(labels[b], x[b], kmax))
            loss = (nll * mask[b]).sum() / mask[b].sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
        hist.append(tot / len(pre))
    return hist
