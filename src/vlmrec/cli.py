"""Command-line entry point: ``vlmrec <command> [-o key=value ...]``.

Commands map 1:1 to pipeline stages; ``week1`` runs them all in order. Heavy deps (torch,
sentence-transformers) are imported lazily per command so data-only commands start fast.
"""

from __future__ import annotations

import argparse

from .config import load_config
from .paths import Paths
from .utils import get_logger, set_seed

COMMANDS = [
    "download",
    "build-interactions",
    "download-images",
    "encode-text",
    "encode-image",
    "eda",
    "week1",
    "retrieval-train",
    "retrieval-eval",
    "ranking-train",
    "ranking-eval",
    "rerank",
    "vlm-profile",
    "encode-profile",
    "vlm-ablation",
    "export-onnx",
    "log-runs",
]


def _dispatch(cmd: str, cfg, paths: Paths, log) -> None:
    if cmd == "download":
        from .data import download

        download.run(cfg, paths)
    elif cmd == "build-interactions":
        from .data import build_interactions

        build_interactions.run(cfg, paths)
    elif cmd == "download-images":
        from .data import download_images

        download_images.run(cfg, paths)
    elif cmd == "encode-text":
        from .features import encode_text

        encode_text.run(cfg, paths)
    elif cmd == "encode-image":
        from .features import encode_image

        encode_image.run(cfg, paths)
    elif cmd == "eda":
        from . import eda

        eda.run(cfg, paths)
    elif cmd == "week1":
        _week1(cfg, paths, log)
    elif cmd == "retrieval-train":
        from .retrieval import train

        train.run(cfg, paths)
    elif cmd == "retrieval-eval":
        from .retrieval import eval as reval

        reval.run(cfg, paths)
    elif cmd == "ranking-train":
        from .ranking import train

        train.run(cfg, paths)
    elif cmd == "ranking-eval":
        from .ranking import eval as rkeval

        rkeval.run(cfg, paths)
    elif cmd == "rerank":
        from .rerank import cascade

        cascade.run(cfg, paths)
    elif cmd == "vlm-profile":
        from .vlm import profile

        profile.run(cfg, paths)
    elif cmd == "encode-profile":
        from .vlm import encode_profile

        encode_profile.run(cfg, paths)
    elif cmd == "vlm-ablation":
        from .vlm import ablate

        ablate.run(cfg, paths)
    elif cmd == "export-onnx":
        from .serving import export_onnx

        export_onnx.run(cfg, paths)
    elif cmd == "log-runs":
        from .mlops import log_runs

        log_runs.run(cfg, paths)


def _week1(cfg, paths: Paths, log) -> None:
    from . import eda
    from .data import build_interactions, download, download_images
    from .features import encode_image, encode_text

    log.info("=== Week 1: data + feature foundation ===")
    download.run(cfg, paths)
    build_interactions.run(cfg, paths)
    download_images.run(cfg, paths)
    encode_text.run(cfg, paths)
    encode_image.run(cfg, paths)
    eda.run(cfg, paths)
    log.info("=== Week 1 complete ===")


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="path to a config YAML")
    common.add_argument(
        "-o",
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dotlist overrides, e.g. dataset.max_reviews=200000",
    )

    parser = argparse.ArgumentParser(prog="vlmrec", description="VLM-Rec data + feature pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in COMMANDS:
        sub.add_parser(name, parents=[common], help=name)

    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(int(cfg.seed))
    paths = Paths(cfg).ensure()

    log = get_logger("vlmrec.cli")
    log.info(
        "cmd=%s | category=%s | max_reviews=%s | device=%s",
        args.cmd,
        cfg.dataset.category,
        cfg.dataset.max_reviews,
        cfg.device,
    )
    if args.override:
        log.info("overrides: %s", list(args.override))
    _dispatch(args.cmd, cfg, paths, log)


if __name__ == "__main__":
    main()
