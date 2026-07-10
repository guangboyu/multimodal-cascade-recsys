# MLOps + polish

## Goal
The production hygiene around the system: continuous integration, monitoring, and experiment tracking
— so the project behaves like something you'd actually operate, not a one-off notebook.

## What was built
- **CI — GitHub Actions** (`.github/workflows/ci.yml`): on every push/PR, runs `ruff check`,
  `ruff format --check`, and `pytest` (pure-logic tests: k-core/splits, MMR/DPP post-processing,
  AUC tie handling, negative sampling). Keeps `main` green.
- **Monitoring — Prometheus** (`src/vlmrec/serving/app.py`): a `/metrics` endpoint plus a
  `vlmrec_latency_ms` histogram (labelled per cascade stage) and a `vlmrec_requests_total` counter,
  scraped straight from the live service.
- **Experiment tracking — MLflow** (`src/vlmrec/mlops/log_runs.py`): reads the metrics JSON each stage
  already writes and logs them as MLflow runs (retrieval modes, ranking ablation variants, rerank
  summary) into a local `./mlruns` store — browsable with `mlflow ui`.

## Run it
```bash
make log-runs        # log all saved stage metrics to MLflow  (vlmrec log-runs)
make mlflow-ui    # browse runs at http://localhost:5000
make serve        # then: curl http://localhost:8000/metrics   (Prometheus format)
```
CI runs automatically on GitHub once the repo is pushed.

## Notes / extensions
- MLflow's file store now requires `MLFLOW_ALLOW_FILE_STORE=true` (set automatically) — or migrate to a
  `sqlite:///mlflow.db` backend for the newest features.
- `docker-compose.yml` is the natural place to add **Grafana** (dashboards over the Prometheus metrics)
  and **Redis** (user-embedding / hot-item cache) — wired as a follow-up.
- Consolidated cross-stage results + resume bullets live in [`RESULTS.md`](RESULTS.md).
