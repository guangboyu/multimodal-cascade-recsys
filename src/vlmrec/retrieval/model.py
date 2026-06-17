"""Two-tower retrieval model.

feature_mode:
  * "content" — item tower from dense multimodal content (text⊕image⊕has_image); user tower
                from pooled history content. Pure multimodal → generalizes to cold/long-tail items.
  * "hybrid"  — item side also gets [item_id, price_bucket, category_leaf] embeddings
                concatenated to the content (the quantile-bucket + embedding pattern).
  * "id"      — pure collaborative baseline: learned user-id and item-id vectors, no content.
                Used for the multimodal-vs-ID ablation (the headline cold-start contrast).

Trained with temperature-scaled in-batch sampled softmax (+ optional logQ popularity correction);
all embeddings are L2-normalized so relevance is a dot product (cosine).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoTower(nn.Module):
    def __init__(
        self,
        content_dim: int,
        n_users: int,
        cat_cardinalities: list[int],  # [n_items, price_card, category_card]
        out_dim: int = 128,
        hidden: tuple[int, ...] = (256,),
        cat_emb_dims: tuple[int, int, int] = (64, 8, 16),
        feature_mode: str = "content",
        dropout: float = 0.1,
        temperature: float = 0.05,
    ):
        super().__init__()
        if feature_mode not in ("content", "hybrid", "id"):
            raise ValueError(f"bad feature_mode: {feature_mode}")
        self.feature_mode = feature_mode
        self.temperature = temperature
        self.out_dim = out_dim
        n_items = cat_cardinalities[0]

        if feature_mode == "id":
            self.user_vec = nn.Embedding(n_users, out_dim)
            self.item_vec = nn.Embedding(n_items, out_dim)
            nn.init.normal_(self.user_vec.weight, std=0.05)
            nn.init.normal_(self.item_vec.weight, std=0.05)
            return

        self.user_mlp = MLP(content_dim, list(hidden), out_dim, dropout)
        if feature_mode == "hybrid":
            self.item_id_emb = nn.Embedding(cat_cardinalities[0], cat_emb_dims[0])
            self.price_emb = nn.Embedding(cat_cardinalities[1], cat_emb_dims[1])
            self.cat_emb = nn.Embedding(cat_cardinalities[2], cat_emb_dims[2])
            item_in = content_dim + sum(cat_emb_dims)
        else:  # content
            item_in = content_dim
        self.item_mlp = MLP(item_in, list(hidden), out_dim, dropout)

    # --- item-side input assembly (content / hybrid) ---
    def _item_input(self, content_rows: torch.Tensor, cat_rows: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "content":
            return content_rows
        emb = torch.cat(
            [
                self.item_id_emb(cat_rows[:, 0]),
                self.price_emb(cat_rows[:, 1]),
                self.cat_emb(cat_rows[:, 2]),
            ],
            dim=-1,
        )
        return torch.cat([content_rows, emb], dim=-1)

    # --- training: returns normalized (user_e, item_e) for a batch of (u, i) ---
    def train_embeddings(self, u, i, user_sum, user_count, content, cat):
        if self.feature_mode == "id":
            ue, ie = self.user_vec(u), self.item_vec(i)
        else:
            cnt = (user_count[u] - 1).clamp(min=1).unsqueeze(1).float()  # leave-one-out
            pooled = (user_sum[u] - content[i]) / cnt
            ue = self.user_mlp(pooled)
            ie = self.item_mlp(self._item_input(content[i], cat[i]))
        return F.normalize(ue, dim=-1), F.normalize(ie, dim=-1)

    def loss(self, user_e, item_e, i_idx, log_q=None):
        logits = (user_e @ item_e.t()) / self.temperature  # (B, B), in-batch negatives
        if log_q is not None:
            logits = logits - log_q[i_idx].unsqueeze(0)  # logQ correction on candidate columns
        labels = torch.arange(user_e.size(0), device=user_e.device)
        return F.cross_entropy(logits, labels)

    # --- eval / serving ---
    @torch.no_grad()
    def all_item_embeddings(self, content, cat, batch: int = 8192) -> torch.Tensor:
        if self.feature_mode == "id":
            return F.normalize(self.item_vec.weight, dim=-1)
        outs = []
        for s in range(0, content.shape[0], batch):
            x = self._item_input(content[s : s + batch], cat[s : s + batch])
            outs.append(F.normalize(self.item_mlp(x), dim=-1))
        return torch.cat(outs, 0)

    @torch.no_grad()
    def user_embeddings_eval(self, user_idx, user_sum, user_count, content) -> torch.Tensor:
        if self.feature_mode == "id":
            return F.normalize(self.user_vec(user_idx), dim=-1)
        cnt = user_count[user_idx].clamp(min=1).unsqueeze(1).float()
        pooled = user_sum[user_idx] / cnt
        return F.normalize(self.user_mlp(pooled), dim=-1)
