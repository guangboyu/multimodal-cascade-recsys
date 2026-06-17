"""Precompute the two-tower's top-K retrieved candidates per user (excluding seen items).

This is the candidate set for the full retrieve -> pre-rank -> rank -> post-process cascade, and the
source of HARD negatives for training the pre-ranker / ranker (fixing the Week-3 cascade).
Saved as ``data/rerank/cand_topk.npy`` of shape (n_users, K).
"""

from __future__ import annotations

import numpy as np
import torch

from ..paths import Paths
from ..retrieval.data import load_retrieval_data
from ..retrieval.model import TwoTower
from ..utils import get_logger, pick_device, timer

log = get_logger("vlmrec.rerank.candidates")


def candidates_path(paths: Paths):
    return paths.data / "rerank" / "cand_topk.npy"


@torch.no_grad()
def precompute_candidates(cfg, paths: Paths, k: int = 200, batch_users: int = 4096) -> np.ndarray:
    device = pick_device(str(cfg.device))
    d = load_retrieval_data(paths)
    rdir = paths.data / "retrieval"
    item_e = torch.tensor(np.load(rdir / "item_emb_content.npy"), device=device)
    rr = cfg.retrieval
    rt = TwoTower(
        d.content_dim,
        d.n_users,
        d.cat_cardinalities,
        out_dim=int(rr.out_dim),
        hidden=tuple(rr.hidden),
        feature_mode="content",
        temperature=float(rr.temperature),
    ).to(device)
    rt.load_state_dict(torch.load(rdir / "model_content.pt", map_location=device))
    rt.eval()
    usum = torch.tensor(d.user_sum_content, device=device)
    ucnt = torch.tensor(d.user_count, device=device)
    content = torch.tensor(d.content, device=device)

    cand = np.zeros((d.n_users, k), dtype=np.int64)
    with timer(log, f"precompute top-{k} candidates"):
        for s in range(0, d.n_users, batch_users):
            ub = np.arange(s, min(s + batch_users, d.n_users))
            ue = rt.user_embeddings_eval(torch.as_tensor(ub, device=device), usum, ucnt, content)
            scores = ue @ item_e.t()
            rows, cols = [], []
            for r, u in enumerate(ub):
                a, b = d.seen_indptr[u], d.seen_indptr[u + 1]
                its = d.seen_items[a:b]
                rows.append(np.full(len(its), r))
                cols.append(its)
            flat = torch.as_tensor(
                np.concatenate(rows) * d.n_items + np.concatenate(cols), device=device
            )
            scores.view(-1)[flat] = -1e9
            cand[ub] = torch.topk(scores, k, dim=1).indices.cpu().numpy()

    out = candidates_path(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, cand)
    log.info("candidates -> %s  shape=%s", out, cand.shape)
    return cand


def run(cfg, paths: Paths) -> dict:
    cand = precompute_candidates(cfg, paths, k=int(cfg.get("rerank", {}).get("cand_k", 200)))
    return {"shape": list(cand.shape)}
