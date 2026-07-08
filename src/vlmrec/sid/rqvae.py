"""RQ-VAE: residual vector quantization of item content embeddings into semantic IDs.

An encoder MLP maps the fused item vector to a latent; L codebooks quantize it coarse-to-fine
(each level quantizes the previous level's residual); a decoder reconstructs the input. The
(c1..cL) code tuple is the item's semantic ID — similar items share prefixes (TIGER, Rajput
et al. 2023).

Codebook-collapse countermeasures (the classic VQ failure): k-means initialization from the
first batches, EMA codebook updates, dead-code re-seeding from live residuals, and per-level
utilization/perplexity logging.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def kmeans_init(x: torch.Tensor, n_codes: int, iters: int = 10, seed: int = 0) -> torch.Tensor:
    """Lloyd's k-means over rows of x -> (n_codes, dim) centroids (codebook init)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)[:n_codes]
    centroids = x[perm.to(x.device)].clone()
    for _ in range(iters):
        d = torch.cdist(x, centroids)
        assign = d.argmin(dim=1)
        for c in range(n_codes):
            m = assign == c
            if m.any():
                centroids[c] = x[m].mean(dim=0)
    return centroids


class VectorQuantizerEMA(nn.Module):
    """One codebook level: nearest-code assignment + EMA updates + dead-code re-seeding."""

    def __init__(self, n_codes: int, dim: int, decay: float = 0.99, dead_steps: int = 200):
        super().__init__()
        self.n_codes, self.dim, self.decay, self.dead_steps = n_codes, dim, decay, dead_steps
        self.register_buffer("codebook", torch.randn(n_codes, dim) * 0.05)
        self.register_buffer("ema_count", torch.zeros(n_codes))
        self.register_buffer("ema_sum", torch.zeros(n_codes, dim))
        self.register_buffer("steps_unused", torch.zeros(n_codes))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def init_from(self, x: torch.Tensor) -> None:
        self.codebook.copy_(kmeans_init(x, self.n_codes))
        self.ema_sum.copy_(self.codebook)
        self.ema_count.fill_(1.0)
        self.initialized.fill_(True)

    def assign(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, self.codebook).argmin(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.assign(x)
        quant = self.codebook[idx]
        if self.training:
            self._update(x.detach(), idx)
        return quant, idx

    @torch.no_grad()
    def _update(self, x: torch.Tensor, idx: torch.Tensor) -> None:
        onehot = F.one_hot(idx, self.n_codes).to(x.dtype)  # (B, K)
        count = onehot.sum(dim=0)  # (K,)
        summed = onehot.t() @ x  # (K, dim)
        self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(summed, alpha=1 - self.decay)
        used = count > 0
        self.codebook[used] = self.ema_sum[used] / self.ema_count[used].unsqueeze(1).clamp(1e-5)
        # dead-code re-seeding: codes unused too long restart at a live residual
        self.steps_unused[used] = 0
        self.steps_unused[~used] += 1
        dead = self.steps_unused >= self.dead_steps
        if dead.any():
            seeds = x[torch.randint(0, x.shape[0], (int(dead.sum()),), device=x.device)]
            self.codebook[dead] = seeds
            self.ema_sum[dead] = seeds
            self.ema_count[dead] = 1.0
            self.steps_unused[dead] = 0

    @torch.no_grad()
    def stats(self, x: torch.Tensor) -> dict:
        idx = self.assign(x)
        counts = torch.bincount(idx, minlength=self.n_codes).float()
        p = counts / counts.sum()
        nz = p[p > 0]
        perplexity = float(torch.exp(-(nz * nz.log()).sum()))
        return {
            "utilization": round(float((counts > 0).float().mean()), 4),
            "perplexity": round(perplexity, 1),
        }


def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class RQVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...] = (512, 256),
        latent_dim: int = 64,
        levels: int = 3,
        n_codes: int = 256,
        beta: float = 0.25,
    ):
        super().__init__()
        self.beta, self.levels = beta, levels
        self.encoder = _mlp([in_dim, *hidden, latent_dim])
        self.decoder = _mlp([latent_dim, *reversed(hidden), in_dim])
        self.quantizers = nn.ModuleList(
            [VectorQuantizerEMA(n_codes, latent_dim) for _ in range(levels)]
        )

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Residual quantization: returns (sum-of-codewords, (B, levels) code indices)."""
        residual, total, codes = z, torch.zeros_like(z), []
        for q in self.quantizers:
            quant, idx = q(residual)
            total = total + quant
            residual = residual - quant.detach()
            codes.append(idx)
        return total, torch.stack(codes, dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_q, codes = self.quantize(z)
        commit = F.mse_loss(z, z_q.detach())
        z_st = z + (z_q - z).detach()  # straight-through: decoder grads flow into the encoder
        recon = F.mse_loss(self.decoder(z_st), x)
        return recon + self.beta * commit, recon.detach(), codes

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor, batch: int = 8192) -> torch.Tensor:
        self.eval()
        outs = []
        for s in range(0, x.shape[0], batch):
            z = self.encoder(x[s : s + batch])
            _, codes = self.quantize(z)
            outs.append(codes)
        return torch.cat(outs, dim=0)  # (N, levels)


def collision_rate(codes: torch.Tensor) -> float:
    """Fraction of items whose full code tuple is shared with at least one other item."""
    _, inverse, counts = torch.unique(codes, dim=0, return_inverse=True, return_counts=True)
    return round(float((counts[inverse] > 1).float().mean()), 4)
