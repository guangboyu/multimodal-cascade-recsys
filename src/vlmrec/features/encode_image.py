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
        num_workers = int(cfg.image_emb.get("num_workers", 8))

        # decode+preprocess in DataLoader workers — the single-threaded PIL loop was the
        # throughput bottleneck at scale (GPU idle while one core decoded JPEGs)
        present = [i for i in range(n_items) if paths.image_path(i).exists()]

        class _ImgDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(present)

            def __getitem__(self, j):
                item_idx = present[j]
                try:
                    img = Image.open(paths.image_path(item_idx)).convert("RGB")
                except Exception:  # noqa: BLE001 - corrupt file -> treat as missing
                    return item_idx, None
                return item_idx, preprocess(img)

        def _collate(batch):
            good = [(i, t) for i, t in batch if t is not None]
            if not good:
                return [], None
            idxs, tensors = zip(*good, strict=True)
            return list(idxs), torch.stack(tensors)

        loader = torch.utils.data.DataLoader(
            _ImgDataset(),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=_collate,
        )
        from tqdm import tqdm

        for idxs, batch in tqdm(loader, desc="clip", unit="batch"):
            if batch is None:
                continue
            with torch.no_grad():
                vecs = model.encode_image(batch.to(device))
                vecs = torch.nn.functional.normalize(vecs, dim=-1).cpu().numpy().astype(np.float32)
            emb[idxs] = vecs
            has_image[idxs] = 1

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
