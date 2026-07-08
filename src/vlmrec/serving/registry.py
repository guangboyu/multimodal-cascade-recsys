"""Load all trained artifacts once (CPU) for the serving cascade.

Holds the retrieval two-tower + FAISS index, the heavy ranker, the distilled pre-ranker, and the
item/user feature tensors — everything a request needs, loaded at process startup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import load_config
from ..paths import Paths
from ..ranking.data import build_ranking_data
from ..ranking.model import Ranker
from ..retrieval.model import TwoTower
from ..utils import get_logger

log = get_logger("vlmrec.serving.registry")


def _require(path, make_target: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"missing serving artifact: {path} — run `{make_target}` first to produce it"
        )


@dataclass
class Registry:
    cfg: object
    d: object  # RetrievalData
    content: torch.Tensor  # (N+1, Cd) padded
    cat: torch.Tensor
    seq_t: torch.Tensor
    user_sum: torch.Tensor
    user_count: torch.Tensor
    item_e: np.ndarray
    index: object
    tt: object
    ranker: object
    prerank: object | None


def load_registry(cfg=None) -> Registry:
    import faiss

    cfg = cfg or load_config()
    paths = Paths(cfg)
    rdir = paths.data / "retrieval"
    _require(paths.interactions_parquet, "make week1")
    _require(paths.text_emb_npy, "make week1")
    _require(rdir / "item_emb_content.npy", "make week2")
    _require(rdir / "model_content.pt", "make week2")
    ranker_ckpt = paths.root / str(cfg.serving.ranker_ckpt)
    if not ranker_ckpt.exists():
        fallback = paths.data / "ranking" / "model_full.pt"
        log.warning("configured ranker_ckpt %s missing — falling back to %s", ranker_ckpt, fallback)
        ranker_ckpt = fallback
    _require(ranker_ckpt, "make week3")
    rdata = build_ranking_data(paths)
    d = rdata.base

    content = torch.tensor(np.vstack([d.content, np.zeros((1, d.content.shape[1]), np.float32)]))
    cat = torch.tensor(d.cat_features)
    seq_t = torch.tensor(rdata.seq)
    user_sum = torch.tensor(d.user_sum_content)
    user_count = torch.tensor(d.user_count)

    item_e = np.ascontiguousarray(np.load(rdir / "item_emb_content.npy").astype(np.float32))
    index = faiss.IndexFlatIP(item_e.shape[1])
    index.add(item_e)

    rr = cfg.retrieval
    tt = TwoTower(
        d.content_dim,
        d.n_users,
        d.cat_cardinalities,
        out_dim=int(rr.out_dim),
        hidden=tuple(rr.hidden),
        feature_mode="content",
        temperature=float(rr.temperature),
    )
    tt.load_state_dict(torch.load(rdir / "model_content.pt", map_location="cpu"))
    tt.eval()

    rk = cfg.ranking
    ranker = Ranker(
        d.content_dim,
        d.cat_cardinalities,
        d_model=int(rk.d_model),
        n_tasks=int(rk.n_tasks),
        use_multimodal=True,
        use_din=True,
        use_cross=True,
        use_mmoe=True,
    )
    ranker.load_state_dict(torch.load(ranker_ckpt, map_location="cpu"))
    ranker.eval()
    log.info("ranker ckpt: %s", ranker_ckpt)

    prerank = None
    pp = paths.data / "rerank" / "prerank.pt"
    if pp.exists():
        prerank = Ranker(
            d.content_dim,
            d.cat_cardinalities,
            d_model=64,
            n_tasks=1,
            use_multimodal=True,
            use_din=False,
            use_cross=False,
            use_mmoe=False,
        )
        prerank.load_state_dict(torch.load(pp, map_location="cpu"))
        prerank.eval()

    log.info(
        "registry loaded: items=%s users=%s prerank=%s", d.n_items, d.n_users, prerank is not None
    )
    return Registry(
        cfg, d, content, cat, seq_t, user_sum, user_count, item_e, index, tt, ranker, prerank
    )
