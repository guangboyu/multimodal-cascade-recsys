"""Config loading: a single YAML file plus optional CLI dotlist overrides (OmegaConf).

Deliberately lightweight — one YAML as the source of truth keeps every run reproducible
from `configs/config.yaml` + the exact overrides printed in the logs.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# repo_root/configs/config.yaml  (config.py lives at repo_root/src/vlmrec/config.py)
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(
    path: str | Path | None = None, overrides: Iterable[str] | None = None
) -> DictConfig:
    """Load the base YAML and apply ``key=value`` dotlist overrides on top."""
    cfg = OmegaConf.load(Path(path) if path else DEFAULT_CONFIG)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    assert isinstance(cfg, DictConfig)
    return cfg
