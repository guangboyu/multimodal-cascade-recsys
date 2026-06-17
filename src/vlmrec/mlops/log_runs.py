"""Log the saved training/eval results to a local MLflow file store (./mlruns).

Each stage writes a metrics JSON; this reads them and records MLflow runs so the experiment
history is browsable with ``mlflow ui``. Keeps trainers clean (wrapping them would also work).
"""

from __future__ import annotations

import json

from ..paths import Paths
from ..utils import get_logger

log = get_logger("vlmrec.mlops.log_runs")


def _load(path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - missing artifact is fine
        return None


def _san(key: str) -> str:
    return key.replace("@", "_").replace(":", "_")


def _log_metrics(mlflow, d: dict) -> None:
    for k, v in (d or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            mlflow.log_metric(_san(k), float(v))


def run(cfg, paths: Paths) -> dict:
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # keep the lightweight ./mlruns store
    import mlflow

    mlflow.set_tracking_uri(f"file:{paths.root / 'mlruns'}")
    mlflow.set_experiment("vlmrec")
    logged = 0

    # retrieval — per feature mode
    rdir = paths.data / "retrieval"
    for mode in ["content", "hybrid", "id"]:
        m = _load(rdir / f"metrics_{mode}.json")
        if not m:
            continue
        with mlflow.start_run(run_name=f"retrieval-{mode}"):
            mlflow.log_params(
                {
                    "stage": "retrieval",
                    "feature_mode": mode,
                    "epochs": m.get("epochs"),
                    "n_params": m.get("n_params"),
                }
            )
            _log_metrics(mlflow, m.get("test"))
        logged += 1

    # ranking — ablation variants
    abl = _load(paths.data / "ranking" / "ablation.json")
    if abl:
        for name, mm in abl.get("results", {}).items():
            with mlflow.start_run(run_name=f"ranking-{name}"):
                mlflow.log_params({"stage": "ranking", "variant": name})
                _log_metrics(mlflow, mm.get("full"))
                if "cold_GAUC" in mm:
                    mlflow.log_metric("cold_GAUC", float(mm["cold_GAUC"]))
            logged += 1

    # rerank — week 4 summary
    rr = _load(paths.data / "rerank" / "results.json")
    if rr:
        with mlflow.start_run(run_name="rerank-week4"):
            mlflow.log_params({"stage": "rerank"})
            for src in ["cascade_naive", "cascade_hardneg"]:
                for k, v in (rr.get(src) or {}).items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(_san(f"{src}_{k}"), float(v))
            _log_metrics(mlflow, rr.get("prerank_consistency"))
        logged += 1

    log.info("logged %d runs to %s", logged, paths.root / "mlruns")
    return {"runs_logged": logged}
