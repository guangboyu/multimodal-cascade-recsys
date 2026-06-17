"""Item image embeddings via CLIP (open_clip).

Encodes whatever images were downloaded; items without an image get a zero vector and a
``has_image=0`` flag (so retrieval/ranking can learn the missing-modality case explicitly —
this is the production-realistic cold-start signal). Output aligned to item_idx.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from omegaconf import DictConfig
from PIL import Image

from ..paths import Paths
from ..utils import get_logger, pick_device, timer

log = get_logger("vlmrec.encode_image")


def run(cfg: DictConfig, paths: Paths) -> dict:
    paths.ensure()
    import open_clip
    import torch

    device = pick_device(str(cfg.device))
    n_items = pl.read_parquet(paths.item_map_parquet).height

    with timer(log, f"encode image [{cfg.image_emb.model}/{cfg.image_emb.pretrained}] on {device}"):
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.image_emb.model, pretrained=cfg.image_emb.pretrained
        )
        model = model.eval().to(device)

        # Probe output dim with a blank image so we can allocate even if 0 images exist.
        with torch.no_grad():
            blank = preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(device)
            dim = int(model.encode_image(blank).shape[1])

        emb = np.zeros((n_items, dim), dtype=np.float32)
        has_image = np.zeros(n_items, dtype=np.int8)
        batch_size = int(cfg.image_emb.batch_size)

        buf_tensors: list[torch.Tensor] = []
        buf_idx: list[int] = []

        def flush() -> None:
            if not buf_idx:
                return
            with torch.no_grad():
                batch = torch.stack(buf_tensors).to(device)
                vecs = model.encode_image(batch)
                vecs = torch.nn.functional.normalize(vecs, dim=-1).cpu().numpy().astype(np.float32)
            for j, item_idx in enumerate(buf_idx):
                emb[item_idx] = vecs[j]
                has_image[item_idx] = 1
            buf_tensors.clear()
            buf_idx.clear()

        from tqdm import tqdm

        for item_idx in tqdm(range(n_items), desc="clip", unit="item"):
            p = paths.image_path(item_idx)
            if not p.exists():
                continue
            try:
                img = Image.open(p).convert("RGB")
            except Exception:  # noqa: BLE001 - corrupt file -> treat as missing
                continue
            buf_tensors.append(preprocess(img))
            buf_idx.append(item_idx)
            if len(buf_idx) >= batch_size:
                flush()
        flush()

        np.save(paths.image_emb_npy, emb)
        np.save(paths.has_image_npy, has_image)
        n_with = int(has_image.sum())
        meta_out = {
            "model": str(cfg.image_emb.model),
            "pretrained": str(cfg.image_emb.pretrained),
            "dim": dim,
            "n_items": n_items,
            "n_with_image": n_with,
            "coverage_pct": round(100 * n_with / n_items, 2) if n_items else 0.0,
            "normalized": True,
        }
        (paths.embeddings / "image_emb.json").write_text(json.dumps(meta_out, indent=2))
    log.info(
        "image emb -> %s  shape=(%d, %d)  coverage=%.1f%%",
        paths.image_emb_npy,
        n_items,
        dim,
        meta_out["coverage_pct"],
    )
    return meta_out
