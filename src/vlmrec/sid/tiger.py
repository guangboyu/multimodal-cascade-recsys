"""TIGER-lite: generative retrieval over semantic-ID tokens (demo scale).

A small decoder-only transformer reads the user's train history as SID tokens (each item =
``levels`` tokens) and generates the next item's codes autoregressively. Retrieval = beam search
constrained to the trie of real items — every generated sequence maps to catalog items, no ANN
index involved (Rajput et al., NeurIPS 2023).

Deliberately lean: one config, sampled-user eval, no sweeps. The point is the paradigm
(constant-size vocab, index-free retrieval); the two-tower remains the production retriever.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..paths import Paths
from ..retrieval.data import cfg_sources, load_retrieval_data
from ..utils import get_logger, pick_device, set_seed
from .train import sid_codes_path

log = get_logger("vlmrec.sid.tiger")


class TigerLM(nn.Module):
    """Decoder-only transformer over SID tokens: token = level * n_codes + code (+ BOS)."""

    def __init__(self, n_codes: int, levels: int, d_model=128, n_layers=4, n_heads=4, max_len=128):
        super().__init__()
        self.n_codes, self.levels = n_codes, levels
        self.vocab = n_codes * levels + 1  # + BOS
        self.bos = self.vocab - 1
        self.tok = nn.Embedding(self.vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True, norm_first=True, dropout=0.1
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, self.vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens (B, L) -> logits (B, L, vocab); causal mask keeps it autoregressive
        b, n = tokens.shape
        x = self.tok(tokens) + self.pos(torch.arange(n, device=tokens.device))
        mask = nn.Transformer.generate_square_subsequent_mask(n, device=tokens.device)
        return self.head(self.blocks(x, mask=mask, is_causal=True))


def item_tokens(sid_codes: np.ndarray) -> np.ndarray:
    """(N, levels) codes -> (N, levels) offset token ids (level l uses its own token range)."""
    levels = sid_codes.shape[1]
    n_codes = int(sid_codes.max()) + 1
    return sid_codes + np.arange(levels)[None, :] * n_codes


def build_trie_masks(tokens: np.ndarray, n_codes: int, levels: int, vocab: int):
    """Prefix -> allowed-next-token boolean masks, so decoding only produces real items."""
    masks: dict[tuple, np.ndarray] = {}
    items_by_code: dict[tuple, list[int]] = {}
    for idx, row in enumerate(tokens):
        for depth in range(levels):
            prefix = tuple(row[:depth])
            m = masks.get(prefix)
            if m is None:
                m = np.zeros(vocab, dtype=bool)
                masks[prefix] = m
            m[row[depth]] = True
        items_by_code.setdefault(tuple(row), []).append(idx)
    return masks, items_by_code


def _histories(d, tokens: np.ndarray, max_hist: int):
    """Per-user token history (last max_hist train items, time-ordered) + last item as target."""
    seqs, targets, users = [], [], []
    for u in range(d.n_users):
        items = d.seen_items[d.seen_indptr[u] : d.seen_indptr[u + 1]]
        if len(items) < 2:
            continue
        hist = items[-(max_hist + 1) : -1]
        seqs.append(tokens[hist].reshape(-1))
        targets.append(tokens[items[-1]])
        users.append(u)
    return seqs, targets, np.array(users)


def _pad_batch(seqs: list[np.ndarray], bos: int, device) -> torch.Tensor:
    # left-pad with BOS so the last position always ends the history
    n = max(len(s) for s in seqs) + 1
    out = np.full((len(seqs), n), bos, dtype=np.int64)
    for r, s in enumerate(seqs):
        out[r, n - len(s) :] = s
    return torch.as_tensor(out, device=device)


@torch.no_grad()
def constrained_beam(model, hist_tok, masks, levels, beam, device):
    """Batched beam search over the SID trie -> (B, beam, levels) codes + (B, beam) scores."""
    b = hist_tok.shape[0]
    beams = [[((), 0.0)] for _ in range(b)]  # per row: list of (prefix tuple, logprob)
    for _depth in range(levels):
        rows, seqs = [], []
        for r in range(b):
            for prefix, _ in beams[r]:
                rows.append(r)
                seqs.append(
                    torch.cat([hist_tok[r], torch.as_tensor(list(prefix), device=device).long()])
                )
        batch = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        # right-pad breaks "last position" semantics -> gather each row's true last index
        lengths = torch.as_tensor([len(s) for s in seqs], device=device)
        logits = model(batch)[torch.arange(len(seqs), device=device), lengths - 1]
        logp = F.log_softmax(logits, dim=-1).cpu().numpy()
        nxt = [[] for _ in range(b)]
        i = 0
        for r in range(b):
            for prefix, score in beams[r]:
                allowed = masks.get(prefix)
                if allowed is not None:
                    lp = np.where(allowed, logp[i], -np.inf)
                    for t in np.argpartition(-lp, min(beam, allowed.sum()) - 1)[:beam]:
                        if np.isfinite(lp[t]):
                            nxt[r].append(((*prefix, int(t)), score + float(lp[t])))
                i += 1
            nxt[r].sort(key=lambda e: -e[1])
            beams[r] = nxt[r][:beam]
    return beams


def run(cfg, paths: Paths) -> dict:
    t = cfg.tiger
    set_seed(int(cfg.seed))
    device = pick_device(str(cfg.device))
    out_dir = paths.data / "sid"
    d = load_retrieval_data(paths, sources=cfg_sources(cfg))
    sid = np.load(sid_codes_path(paths))
    levels, n_codes = sid.shape[1], int(sid.max()) + 1
    tokens = item_tokens(sid)

    model = TigerLM(
        n_codes,
        levels,
        d_model=int(t.d_model),
        n_layers=int(t.n_layers),
        n_heads=int(t.n_heads),
        max_len=(int(t.max_hist) + 2) * levels + 2,
    ).to(device)
    masks, items_by_code = build_trie_masks(tokens, n_codes, levels, model.vocab - 1)
    log.info(
        "TIGER-lite: vocab=%d params=%s | %d SID prefixes, %.3f collision groups/item",
        model.vocab,
        f"{sum(p.numel() for p in model.parameters()):,}",
        len(masks),
        len(items_by_code) / len(tokens),
    )

    seqs, targets, users = _histories(d, tokens, int(t.max_hist))
    opt = torch.optim.Adam(model.parameters(), lr=float(t.lr))
    order_pop = np.argsort(-d.item_pop, kind="stable")
    pop_rank = np.empty(d.n_items, dtype=np.int64)
    pop_rank[order_pop] = np.arange(d.n_items)

    bs = int(t.batch_size)
    t0 = time.time()
    for ep in range(1, int(t.epochs) + 1):
        model.train()
        perm = np.random.permutation(len(seqs))
        total, nb = 0.0, 0
        for s in range(0, len(perm), bs):
            idx = perm[s : s + bs]
            hist = _pad_batch([seqs[i] for i in idx], model.bos, device)
            tgt = torch.as_tensor(np.stack([targets[i] for i in idx]), device=device)
            full = torch.cat([hist, tgt], dim=1)
            logits = model(full[:, :-1])
            npos = tgt.shape[1]
            loss = F.cross_entropy(logits[:, -npos:].reshape(-1, model.vocab), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            nb += 1
        log.info("tiger ep%02d | loss %.4f | %.1fs", ep, total / nb, time.time() - t0)

    # eval: generate the TEST item for a user sample (history = train items, same as two-tower)
    model.eval()
    test_users = np.where(d.test_item >= 0)[0]
    rng = np.random.default_rng(0)
    sample = rng.choice(test_users, size=min(int(t.eval_users), len(test_users)), replace=False)
    user_row = {int(u): i for i, u in enumerate(users)}
    hits, ndcg, n_eval = 0, 0.0, 0
    for s in range(0, len(sample), 64):
        ub = [int(u) for u in sample[s : s + 64] if int(u) in user_row]
        if not ub:
            continue
        hist = _pad_batch([seqs[user_row[u]] for u in ub], model.bos, device)
        beams = constrained_beam(model, hist, masks, levels, int(t.beam), device)
        for r, u in enumerate(ub):
            seen = set(d.seen_items[d.seen_indptr[u] : d.seen_indptr[u + 1]].tolist())
            ranked: list[int] = []
            for prefix, _score in beams[r]:
                group = sorted(items_by_code.get(prefix, []), key=lambda i: pop_rank[i])
                ranked.extend(i for i in group if i not in seen and i not in ranked)
            n_eval += 1
            tgt = int(d.test_item[u])
            if tgt in ranked[:10]:
                hits += 1
                ndcg += 1.0 / math.log2(ranked.index(tgt) + 2)

    metrics = {
        "recall@10": round(hits / max(n_eval, 1), 5),
        "ndcg@10": round(ndcg / max(n_eval, 1), 5),
        "n_users_eval": n_eval,
        "beam": int(t.beam),
        "vocab": model.vocab,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "wall_clock_min": round((time.time() - t0) / 60, 1),
    }
    torch.save(model.state_dict(), out_dir / "tiger.pt")
    (out_dir / "tiger.json").write_text(json.dumps(metrics, indent=2))
    log.info("tiger demo -> %s | %s", out_dir / "tiger.json", metrics)
    return metrics
