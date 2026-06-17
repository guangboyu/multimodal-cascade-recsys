"""Small shared helpers: logging, seeding, timing, device selection."""

from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager


def get_logger(name: str = "vlmrec") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # numpy optional at import time
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def pick_device(prefer: str = "auto") -> str:
    """Resolve 'auto' to 'cuda' when available, else 'cpu'."""
    if prefer == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@contextmanager
def timer(logger: logging.Logger, label: str):
    t0 = time.perf_counter()
    logger.info("▶ %s ...", label)
    try:
        yield
    finally:
        logger.info("✓ %s (%.1fs)", label, time.perf_counter() - t0)
