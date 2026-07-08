"""Download Amazon Reviews 2023 reviews + item metadata for one category.

`datasets` 5.x dropped loading-script support, and this dataset ships as plain JSONL in the
HF repo:
    raw/review_categories/<Category>.jsonl
    raw/meta_categories/meta_<Category>.jsonl
So we read those files directly (huggingface_hub + polars) — faster, and robust to the messy
metadata. Nested/variable fields (images, description, features, categories) are stored as
JSON strings to guarantee a stable parquet schema; downstream stages decode them where needed.
"""

from __future__ import annotations

import itertools
import json

import polars as pl
import requests
from huggingface_hub import hf_hub_download, hf_hub_url
from omegaconf import DictConfig

from ..paths import Paths
from ..utils import get_logger, timer

log = get_logger("vlmrec.download")

REVIEW_COLS = ["user_id", "parent_asin", "rating", "timestamp", "verified_purchase", "helpful_vote"]
_HEADERS = {"User-Agent": "vlmrec/0.1 (academic-portfolio)"}


def _review_file(category: str) -> str:
    return f"raw/review_categories/{category}.jsonl"


def _meta_file(category: str) -> str:
    return f"raw/meta_categories/meta_{category}.jsonl"


def _norm_meta(d: dict) -> dict:
    """Project + normalize one metadata record to a stable, parquet-friendly schema."""

    def js(x) -> str:
        return json.dumps(x, ensure_ascii=False)

    price = d.get("price")
    return {
        "parent_asin": d.get("parent_asin"),
        "title": d.get("title") or "",
        "main_category": d.get("main_category"),
        "store": d.get("store"),
        "price": str(price) if price not in (None, "") else None,
        "average_rating": d.get("average_rating"),
        "rating_number": d.get("rating_number"),
        "description": js(d.get("description")),  # JSON-encoded list[str]
        "features": js(d.get("features")),  # JSON-encoded list[str]
        "categories": js(d.get("categories")),  # JSON-encoded list[str]
        "images": js(d.get("images")),  # JSON-encoded list[dict] / dict
    }


def download_meta(cfg: DictConfig, paths: Paths) -> int:
    repo, cat = cfg.dataset.hf_repo, cfg.dataset.category
    with timer(log, f"download meta [{cat}]"):
        local = hf_hub_download(repo_id=repo, filename=_meta_file(cat), repo_type="dataset")
        rows = []
        with open(local, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(_norm_meta(json.loads(line)))
        df = pl.DataFrame(rows)
        df.write_parquet(paths.meta_parquet)
        n = df.height
    log.info("meta -> %s  rows=%s", paths.meta_parquet, f"{n:,}")
    return n


def download_reviews(cfg: DictConfig, paths: Paths) -> int:
    repo, cat = cfg.dataset.hf_repo, cfg.dataset.category
    cap = int(cfg.dataset.max_reviews) if cfg.dataset.max_reviews else None
    with timer(log, f"download reviews [{cat}] cap={cap}"):
        if cap:
            # Stream the JSONL over HTTP and take the first `cap` rows (avoids full download).
            url = hf_hub_url(repo_id=repo, filename=_review_file(cat), repo_type="dataset")
            rows = []
            with requests.get(url, stream=True, timeout=120, headers=_HEADERS) as resp:
                resp.raise_for_status()
                for line in itertools.islice(resp.iter_lines(), cap):
                    if line:
                        d = json.loads(line)
                        rows.append({c: d.get(c) for c in REVIEW_COLS})
            df = pl.DataFrame(rows)
            df.write_parquet(paths.reviews_parquet)
            n = df.height
        else:
            # Full corpus: download once, stream JSONL -> parquet (memory-efficient).
            local = hf_hub_download(repo_id=repo, filename=_review_file(cat), repo_type="dataset")
            lf = pl.scan_ndjson(local)
            cols = [c for c in REVIEW_COLS if c in lf.collect_schema().names()]
            lf.select(cols).sink_parquet(paths.reviews_parquet)
            n = pl.scan_parquet(paths.reviews_parquet).select(pl.len()).collect().item()
    log.info("reviews -> %s  rows=%s", paths.reviews_parquet, f"{n:,}")
    return n


def run(cfg: DictConfig, paths: Paths) -> None:
    paths.ensure()
    download_meta(cfg, paths)
    download_reviews(cfg, paths)
