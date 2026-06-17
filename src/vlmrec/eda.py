"""Summarize the built dataset into a human-readable dataset card.

Reads whatever artifacts exist (interaction stats, embedding metadata) so it can run after
a partial pipeline. Writes ``data/processed/dataset_card.md`` and prints a summary.
"""

from __future__ import annotations

import json

from omegaconf import DictConfig

from .paths import Paths
from .utils import get_logger

log = get_logger("vlmrec.eda")


def _load_json(path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - missing/partial artifacts are fine here
        return {}


def run(cfg: DictConfig, paths: Paths) -> dict:
    paths.ensure()
    stats = _load_json(paths.stats_json)
    tmeta = _load_json(paths.embeddings / "text_emb.json")
    imeta = _load_json(paths.embeddings / "image_emb.json")

    lines: list[str] = []
    out = lines.append
    out(f"# Dataset Card — {cfg.dataset.category} (Amazon Reviews 2023)")
    out("")
    out(f"- **Source**: `{cfg.dataset.hf_repo}` (configs `raw_review_*` / `raw_meta_*`)")
    out(f"- **max_reviews**: {cfg.dataset.max_reviews}  _(null = full corpus)_")
    out(f"- **Filtering**: dedup={bool(cfg.filtering.dedup)}, {cfg.filtering.k_core}-core")
    out(f"- **Split**: {cfg.split.scheme} (temporal, per user)")
    out("")
    out("## Interactions")
    if stats:
        out(
            f"- users **{stats['n_users']:,}** · items **{stats['n_items']:,}** · "
            f"interactions **{stats['n_interactions']:,}**"
        )
        out(
            f"- density {stats['density_pct']}% · avg/user {stats['avg_interactions_per_user']} · "
            f"avg/item {stats['avg_interactions_per_item']}"
        )
        out(f"- splits: `{stats['split_counts']}`")
        out(
            f"- strong-positive (rating ≥ {cfg.dataset.positive_rating}) fraction: "
            f"{stats['strong_positive_frac']}"
        )
    else:
        out("- _not built yet — run `vlmrec build-interactions`_")
    out("")
    out("## Multimodal features")
    out(
        f"- **text**: {tmeta['model']} → dim {tmeta['dim']} ({tmeta['n_items']:,} items)"
        if tmeta
        else "- text: _not encoded yet_"
    )
    out(
        f"- **image**: {imeta['model']}/{imeta['pretrained']} → dim {imeta['dim']}, "
        f"coverage {imeta['coverage_pct']}% ({imeta['n_with_image']:,}/{imeta['n_items']:,})"
        if imeta
        else "- image: _not encoded yet_"
    )
    out("")

    card = "\n".join(lines)
    paths.dataset_card.write_text(card)
    log.info("dataset card -> %s", paths.dataset_card)
    print("\n" + card)
    return {"stats": stats, "text": tmeta, "image": imeta}
