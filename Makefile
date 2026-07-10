.PHONY: setup data data-dev retrieval retrieval-train retrieval-eval ranking ranking-train ranking-eval rerank vlm sid sid-train sid-eval vlm-profile encode-profile vlm-ablation serve serve-api demo export-onnx log-runs mlflow-ui download interactions images encode-text encode-image eda test lint fmt clean

setup:          ## create .venv and install everything (faiss/mlflow live in extras)
	uv sync --all-extras

# --- data + item features ---
data:           ## full Video_Games build (all reviews + items)
	uv run vlmrec data

data-dev:       ## fast capped run to prove the pipeline (subsampled reviews + images)
	uv run vlmrec data -o dataset.max_reviews=500000 filtering.k_core=3 images.max_images=500

download:
	uv run vlmrec download
interactions:
	uv run vlmrec build-interactions
images:
	uv run vlmrec download-images
encode-text:
	uv run vlmrec encode-text
encode-image:
	uv run vlmrec encode-image
eda:
	uv run vlmrec eda

# --- retrieval (two-tower + FAISS + i2i) ---
retrieval:      ## train all 3 retrieval modes (content/hybrid/id) + run the ablation
	uv run vlmrec retrieval-train -o retrieval.feature_mode=content
	uv run vlmrec retrieval-train -o retrieval.feature_mode=hybrid
	uv run vlmrec retrieval-train -o retrieval.feature_mode=id
	uv run vlmrec retrieval-eval
retrieval-train:
	uv run vlmrec retrieval-train
retrieval-eval:
	uv run vlmrec retrieval-eval

# --- ranking (DIN + DCN-v2 + MMoE) ---
ranking:        ## train ranker + run the ablation
	uv run vlmrec ranking-train
	uv run vlmrec ranking-eval
ranking-train:
	uv run vlmrec ranking-train
ranking-eval:
	uv run vlmrec ranking-eval

# --- pre-ranking + post-processing + cascade consistency ---
rerank:         ## hard-neg cascade fix + pre-ranker distill + diversity (MMR/DPP)
	uv run vlmrec rerank

# --- VLM item profiles (Qwen2.5-VL) ---
vlm:            ## generate profiles + encode + feature-source ablation
	uv run vlmrec vlm-profile
	uv run vlmrec encode-profile
	uv run vlmrec vlm-ablation
vlm-profile:
	uv run vlmrec vlm-profile
encode-profile:
	uv run vlmrec encode-profile
vlm-ablation:
	uv run vlmrec vlm-ablation

# --- semantic IDs (RQ-VAE) ---
sid:            ## train RQ-VAE codes + SID-vs-ID ablation
	uv run vlmrec sid-train
	uv run vlmrec sid-eval
sid-train:
	uv run vlmrec sid-train
sid-eval:
	uv run vlmrec sid-eval

# --- serving (FastAPI cascade + Streamlit UI + ONNX) ---
serve:          ## API (:8000) + Streamlit UI (:8501) together — one command, Ctrl-C stops both
	uv run vlmrec serve
serve-api:      ## API only — FastAPI cascade at http://localhost:8000 (clients connect separately)
	uv run vlmrec serve-api
demo:           ## Streamlit UI only at http://localhost:8501 (needs an API running elsewhere)
	uv run vlmrec demo
export-onnx:    ## export ranker to ONNX (then `make serve` with -o serving.use_onnx=true)
	uv run vlmrec export-onnx

# --- MLOps (CI in .github/, Prometheus /metrics in the app) ---
log-runs:       ## log all results to MLflow
	uv run vlmrec log-runs
mlflow-ui:      ## browse tracked runs at http://localhost:5000
	uv run mlflow ui --backend-store-uri ./mlruns

test:           ## pure-logic unit tests (no network/GPU)
	uv run pytest -q
lint:
	uv run ruff check src tests
fmt:
	uv run ruff format src tests

clean:          ## remove generated artifacts (keeps code)
	rm -rf data/raw data/processed data/images data/embeddings
