"""Load all trained artifacts once (CPU) for the serving cascade.

Holds the retrieval two-tower + FAISS index, the heavy ranker, the distilled pre-ranker, and the
item/user feature tensors — everything a request needs, loaded at process startup.
"""

from __future__ import annotations

import json
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


def _ckpt_meta(ckpt_path) -> dict:
    """Architecture metadata saved next to a checkpoint (e.g. ranker_cascade.json)."""
    mp = ckpt_path.with_suffix(".json")
    try:
        return json.loads(mp.read_text()) if mp.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


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
    onnx: object | None = None


def load_registry(cfg=None) -> Registry:
    import faiss

    cfg = cfg or load_config()
    paths = Paths(cfg)
    rdir = paths.data / "retrieval"
    for p in (
        paths.interactions_parquet,
        paths.item_map_parquet,
        paths.meta_parquet,
        paths.text_emb_npy,
        paths.image_emb_npy,
        paths.has_image_npy,
    ):
        _require(p, "make week1")
    _require(rdir / "item_emb_content.npy", "make week2")
    _require(rdir / "model_content.pt", "make week2")
    ranker_ckpt = paths.root / str(cfg.serving.ranker_ckpt)
    fallback = paths.data / "ranking" / "model_full.pt"
    if not ranker_ckpt.exists() and fallback != ranker_ckpt:
        log.warning("configured ranker_ckpt %s missing — falling back to %s", ranker_ckpt, fallback)
        ranker_ckpt = fallback
    _require(ranker_ckpt, "make week3")
    from ..retrieval.data import cfg_sources

    rdata = build_ranking_data(paths, sources=cfg_sources(cfg))
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
    meta = _ckpt_meta(ranker_ckpt)
    ranker = Ranker(
        d.content_dim,
        d.cat_cardinalities,
        d_model=int(meta.get("d_model", rk.d_model)),
        n_tasks=int(meta.get("n_tasks", rk.n_tasks)),
        use_multimodal=True,
        use_din=True,
        use_cross=True,
        use_mmoe=True,
        use_retrieval_score=bool(meta.get("use_retrieval_score", False)),
    )
    ranker.load_state_dict(torch.load(ranker_ckpt, map_location="cpu"))
    ranker.eval()
    log.info("ranker ckpt: %s (score_feature=%s)", ranker_ckpt, ranker.use_retrieval_score)

    prerank = None
    pp = paths.data / "rerank" / "prerank.pt"
    if pp.exists():
        pmeta = _ckpt_meta(pp)
        prerank = Ranker(
            d.content_dim,
            d.cat_cardinalities,
            d_model=int(pmeta.get("d_model", 64)),
            n_tasks=int(pmeta.get("n_tasks", 1)),
            use_multimodal=True,
            use_din=False,
            use_cross=False,
            use_mmoe=False,
            use_retrieval_score=bool(pmeta.get("use_retrieval_score", False)),
        )
        prerank.load_state_dict(torch.load(pp, map_location="cpu"))
        prerank.eval()

    onnx_sess = None
    if bool(cfg.serving.get("use_onnx", False)):
        onnx_path = paths.data / "serving" / "ranker.onnx"
        _require(onnx_path, "make export-onnx")
        import onnxruntime as ort

        onnx_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        log.info("ONNX runtime enabled: %s", onnx_path)

    log.info(
        "registry loaded: items=%s users=%s prerank=%s", d.n_items, d.n_users, prerank is not None
    )
    return Registry(
        cfg,
        d,
        content,
        cat,
        seq_t,
        user_sum,
        user_count,
        item_e,
        index,
        tt,
        ranker,
        prerank,
        onnx_sess,
    )
