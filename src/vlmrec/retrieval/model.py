"""Two-tower retrieval model.

feature_mode:
  * "content" — item tower from dense multimodal content (text⊕image⊕has_image); user tower
                from pooled history content. Pure multimodal → generalizes to cold/long-tail items.
  * "hybrid"  — item side also gets [item_id, price_bucket, category_leaf] embeddings
                concatenated to the content (the quantile-bucket + embedding pattern).
  * "id"      — pure collaborative baseline: learned user-id and item-id vectors, no content.
                Used for the multimodal-vs-ID ablation (the headline cold-start contrast).
  * "sid"     — item side = semantic-ID code embeddings only (RQ-VAE codes, Week 9): the
                cold-start-friendly ID replacement. User side stays a learned id vector, so the
                ablation vs "id" isolates the item-ID swap.
  * "content_sid" — content ⊕ SID code embeddings (mirrors hybrid's concat, with SIDs instead
                of raw-ID/bucket embeddings).

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
        sid_codes: torch.Tensor | None = None,  # (N, levels) int64 — required for *sid modes
        sid_emb_dim: int = 32,
    ):
        super().__init__()
        if feature_mode not in ("content", "hybrid", "id", "sid", "content_sid"):
            raise ValueError(f"bad feature_mode: {feature_mode}")
        self.feature_mode = feature_mode
        self.temperature = temperature
        self.out_dim = out_dim
        n_items = cat_cardinalities[0]

        sid_dim = 0
        if feature_mode in ("sid", "content_sid"):
            if sid_codes is None:
                raise ValueError(f"feature_mode={feature_mode} needs sid_codes (run sid-train)")
            self.register_buffer("sid_codes", sid_codes.long())
            levels = sid_codes.shape[1]
            n_codes = int(sid_codes.max()) + 1
            self.sid_embs = nn.ModuleList(
                [nn.Embedding(n_codes, sid_emb_dim) for _ in range(levels)]
            )
            sid_dim = levels * sid_emb_dim

        if feature_mode == "id":
            self.user_vec = nn.Embedding(n_users, out_dim)
            self.item_vec = nn.Embedding(n_items, out_dim)
            nn.init.normal_(self.user_vec.weight, std=0.05)
            nn.init.normal_(self.item_vec.weight, std=0.05)
            return
        if feature_mode == "sid":
            # collaborative user vector + semantic-ID item tower: isolates the item-ID swap
            self.user_vec = nn.Embedding(n_users, out_dim)
            nn.init.normal_(self.user_vec.weight, std=0.05)
            self.item_mlp = MLP(sid_dim, list(hidden), out_dim, dropout)
            return

        self.user_mlp = MLP(content_dim, list(hidden), out_dim, dropout)
        if feature_mode == "hybrid":
            self.item_id_emb = nn.Embedding(cat_cardinalities[0], cat_emb_dims[0])
            self.price_emb = nn.Embedding(cat_cardinalities[1], cat_emb_dims[1])
            self.cat_emb = nn.Embedding(cat_cardinalities[2], cat_emb_dims[2])
            item_in = content_dim + sum(cat_emb_dims)
        else:  # content / content_sid
            item_in = content_dim + sid_dim
        self.item_mlp = MLP(item_in, list(hidden), out_dim, dropout)

    def _sid_emb(self, i: torch.Tensor) -> torch.Tensor:
        codes = self.sid_codes[i]  # (B, levels)
        return torch.cat([emb(codes[:, level]) for level, emb in enumerate(self.sid_embs)], dim=-1)

    # --- item-side input assembly (content / hybrid / content_sid) ---
    def _item_input(
        self, content_rows: torch.Tensor, cat_rows: torch.Tensor, i: torch.Tensor
    ) -> torch.Tensor:
        if self.feature_mode == "content":
            return content_rows
        if self.feature_mode == "content_sid":
            return torch.cat([content_rows, self._sid_emb(i)], dim=-1)
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
        elif self.feature_mode == "sid":
            ue, ie = self.user_vec(u), self.item_mlp(self._sid_emb(i))
        else:
            cnt = (user_count[u] - 1).clamp(min=1).unsqueeze(1).float()  # leave-one-out
            pooled = (user_sum[u] - content[i]) / cnt
            ue = self.user_mlp(pooled)
            ie = self.item_mlp(self._item_input(content[i], cat[i], i))
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
            i = torch.arange(s, min(s + batch, content.shape[0]), device=content.device)
            if self.feature_mode == "sid":
                x = self._sid_emb(i)
            else:
                x = self._item_input(content[i], cat[i], i)
            outs.append(F.normalize(self.item_mlp(x), dim=-1))
        return torch.cat(outs, 0)

    @torch.no_grad()
    def user_embeddings_eval(self, user_idx, user_sum, user_count, content) -> torch.Tensor:
        if self.feature_mode in ("id", "sid"):
            return F.normalize(self.user_vec(user_idx), dim=-1)
        cnt = user_count[user_idx].clamp(min=1).unsqueeze(1).float()
        pooled = user_sum[user_idx] / cnt
        return F.normalize(self.user_mlp(pooled), dim=-1)
