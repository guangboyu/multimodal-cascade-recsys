"""Command-line entry point: ``vlmrec <command> [-o key=value ...]``.

Commands map 1:1 to pipeline stages; ``data`` runs the whole data/feature foundation in order.
Heavy deps (torch, sentence-transformers) are imported lazily per command so data-only commands
start fast.
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
    "data",
    "retrieval-train",
    "retrieval-eval",
    "ranking-train",
    "ranking-eval",
    "rerank",
    "vlm-profile",
    "encode-profile",
    "vlm-ablation",
    "sid-train",
    "sid-eval",
    "tiger-demo",
    "export-onnx",
    "log-runs",
    "serve",
    "serve-api",
    "demo",
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
    elif cmd == "data":
        _data(cfg, paths, log)
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
    elif cmd == "sid-train":
        from .sid import train as sid_train

        sid_train.run(cfg, paths)
    elif cmd == "sid-eval":
        from .sid import eval as sid_eval

        sid_eval.run(cfg, paths)
    elif cmd == "tiger-demo":
        from .sid import tiger

        tiger.run(cfg, paths)
    elif cmd == "export-onnx":
        from .serving import export_onnx

        export_onnx.run(cfg, paths)
    elif cmd == "log-runs":
        from .mlops import log_runs

        log_runs.run(cfg, paths)
    elif cmd == "serve":
        _serve(cfg, log)
    elif cmd == "serve-api":
        _serve_api(cfg)
    elif cmd == "demo":
        _demo(cfg)


def _serve_api(cfg) -> None:
    """Run just the FastAPI cascade (config-driven host/port)."""
    import uvicorn

    uvicorn.run("vlmrec.serving.app:app", host=str(cfg.serving.host), port=int(cfg.serving.port))


def _serve(cfg, log) -> None:
    """Launch the FastAPI cascade + the Streamlit UI together; Ctrl-C stops both.

    Starts the API as a child process, waits for ``/health`` to go green (cold start loads the
    registry — features + checkpoints), then runs the UI in the foreground pointed at it.
    """
    import importlib.util
    import os
    import subprocess
    import sys
    import time
    import urllib.error
    import urllib.request
    from pathlib import Path as _P

    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit("streamlit is not installed — run `uv sync --all-extras` first")

    host, port, ui_port = str(cfg.serving.host), int(cfg.serving.port), int(cfg.demo.port)
    health = f"http://localhost:{port}/health"
    api = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "uvicorn",
            "vlmrec.serving.app:app",
            "--host",
            host,
            "--port",
            str(port),
        ]
    )
    try:
        log.info("serving API booting on http://localhost:%d — loading registry…", port)
        # registry load (features + ranker) is slow on cold start; compose allows 120s
        timeout_s = 180
        for _ in range(timeout_s):
            if api.poll() is not None:
                raise SystemExit(f"serving API exited early (code {api.returncode})")
            try:
                with urllib.request.urlopen(health, timeout=2) as r:  # noqa: S310
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass  # port not bound / registry not ready yet — keep polling
            time.sleep(1)
        else:
            raise SystemExit(f"serving API did not become healthy within {timeout_s}s")

        log.info("API healthy ✓  starting UI on http://localhost:%d (Ctrl-C stops both)", ui_port)
        app_file = _P(__file__).parent / "serving" / "demo_app.py"
        env = {**os.environ, "VLMREC_API": f"http://localhost:{port}"}
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_file),
                "--server.port",
                str(ui_port),
                "--server.headless",
                "true",
            ],
            env=env,
            check=False,
        )
    except KeyboardInterrupt:
        pass
    finally:
        api.terminate()
        try:
            api.wait(timeout=10)
        except subprocess.TimeoutExpired:
            api.kill()
        log.info("serving API stopped")


def _demo(cfg) -> None:
    """Launch the Streamlit UI (talks to the serving API from cfg.demo over HTTP)."""
    import importlib.util
    import os
    import subprocess
    import sys
    from pathlib import Path as _P

    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit("streamlit is not installed — run `uv sync --all-extras` first")
    app_file = _P(__file__).parent / "serving" / "demo_app.py"
    env = {**os.environ, "VLMREC_API": str(cfg.demo.api)}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_file),
            "--server.port",
            str(cfg.demo.port),
            "--server.headless",
            "true",
        ],
        env=env,
        check=False,
    )


def _data(cfg, paths: Paths, log) -> None:
    from . import eda
    from .data import build_interactions, download, download_images
    from .features import encode_image, encode_text

    log.info("=== data + feature foundation ===")
    download.run(cfg, paths)
    build_interactions.run(cfg, paths)
    download_images.run(cfg, paths)
    encode_text.run(cfg, paths)
    encode_image.run(cfg, paths)
    eda.run(cfg, paths)
    log.info("=== data + feature foundation complete ===")


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
