"""Ranking model: DIN target-attention + DCN-v2 cross network + MMoE multi-task heads.

Re-scores a (user, candidate) pair. The "multimodal + behavior" stack (candidate content, DIN
attention over the behavior sequence, pooled history profile) is gated by ``use_multimodal`` /
``use_din`` so we can ablate it against a structured/ID-only ranker. DCN-v2 and MMoE are likewise
toggleable. Two task heads by default: P(click) and P(satisfaction, rating>=4).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DINAttention(nn.Module):
    """Target attention: weight history items by relevance to the candidate (DIN)."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim * 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # query (B, d); keys (B, L, d); mask (B, L) bool (True = valid history slot)
        q = query.unsqueeze(1).expand_as(keys)
        feat = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        score = self.mlp(feat).squeeze(-1)  # (B, L)
        score = score.masked_fill(~mask, -1e9)
        attn = torch.softmax(score, dim=1).masked_fill(~mask, 0.0)
        return (attn.unsqueeze(-1) * keys).sum(1)  # (B, d)


class CrossNetV2(nn.Module):
    """DCN-v2 explicit feature crosses: x_{l+1} = x0 ⊙ (W x_l + b) + x_l."""

    def __init__(self, dim: int, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for layer in self.layers:
            x = x0 * layer(x) + x
        return x


class MMoE(nn.Module):
    """Multi-gate Mixture-of-Experts: shared experts, one softmax gate per task."""

    def __init__(self, in_dim: int, n_experts: int, expert_dim: int, n_tasks: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_dim, expert_dim), nn.ReLU()) for _ in range(n_experts)]
        )
        self.gates = nn.ModuleList([nn.Linear(in_dim, n_experts) for _ in range(n_tasks)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        experts = torch.stack([e(x) for e in self.experts], dim=1)  # (B, n_experts, expert_dim)
        outs = []
        for gate in self.gates:
            w = torch.softmax(gate(x), dim=-1).unsqueeze(-1)  # (B, n_experts, 1)
            outs.append((w * experts).sum(1))  # (B, expert_dim)
        return outs


class Ranker(nn.Module):
    def __init__(
        self,
        content_dim: int,
        cat_cardinalities: list[int],  # [n_items, price_card, category_card]
        cat_emb_dims: tuple[int, int, int] = (32, 8, 16),
        d_model: int = 128,
        cross_layers: int = 2,
        n_experts: int = 4,
        expert_dim: int = 128,
        n_tasks: int = 2,
        use_multimodal: bool = True,
        use_din: bool = True,
        use_cross: bool = True,
        use_mmoe: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_multimodal = use_multimodal
        self.use_din = use_din and use_multimodal  # DIN needs item content
        self.use_cross = use_cross
        self.use_mmoe = use_mmoe
        self.n_tasks = n_tasks
        self.pad_idx = cat_cardinalities[0]  # n_items

        self.item_id_emb = nn.Embedding(cat_cardinalities[0], cat_emb_dims[0])
        self.price_emb = nn.Embedding(cat_cardinalities[1], cat_emb_dims[1])
        self.cat_emb = nn.Embedding(cat_cardinalities[2], cat_emb_dims[2])
        self.content_proj = nn.Linear(content_dim, d_model) if use_multimodal else None
        self.din = DINAttention(d_model) if self.use_din else None

        mm_blocks = (
            (2 + (1 if self.use_din else 0)) if use_multimodal else 0
        )  # cand + profile (+din)
        in_dim = sum(cat_emb_dims) + d_model * mm_blocks

        self.cross = CrossNetV2(in_dim, cross_layers) if use_cross else None
        self.deep = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU()
        )
        head_in = (in_dim if use_cross else 0) + 128
        if use_mmoe:
            self.mmoe = MMoE(head_in, n_experts, expert_dim, n_tasks)
            self.heads = nn.ModuleList([nn.Linear(expert_dim, 1) for _ in range(n_tasks)])
        else:
            self.shared = nn.Sequential(nn.Linear(head_in, expert_dim), nn.ReLU())
            self.heads = nn.ModuleList([nn.Linear(expert_dim, 1) for _ in range(n_tasks)])

    def forward(self, cand, seq, content_pad, cat) -> torch.Tensor:
        # cand (B,) long; seq (B, L) long (pad=n_items); content_pad (N+1, Cd); cat (N, n_cat)
        parts = [self.item_id_emb(cand), self.price_emb(cat[cand, 1]), self.cat_emb(cat[cand, 2])]
        if self.use_multimodal:
            cand_c = self.content_proj(content_pad[cand])  # (B, d)
            parts.insert(0, cand_c)
            seq_proj = self.content_proj(content_pad[seq])  # (B, L, d)
            # valid history = not-pad AND not the candidate itself (leave-one-out, no leakage)
            mask = (seq != self.pad_idx) & (seq != cand.unsqueeze(1))
            if self.use_din:
                parts.append(self.din(cand_c, seq_proj, mask))
            denom = mask.sum(1, keepdim=True).clamp(min=1)
            parts.append((seq_proj * mask.unsqueeze(-1)).sum(1) / denom)  # pooled profile
        x = torch.cat(parts, dim=-1)

        reps = []
        if self.use_cross:
            reps.append(self.cross(x))
        reps.append(self.deep(x))
        h = torch.cat(reps, dim=-1)

        if self.use_mmoe:
            task_h = self.mmoe(h)
            logits = [head(th).squeeze(-1) for head, th in zip(self.heads, task_h, strict=False)]
        else:
            s = self.shared(h)
            logits = [head(s).squeeze(-1) for head in self.heads]
        return torch.stack(logits, dim=1)  # (B, n_tasks)
