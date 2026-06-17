.PHONY: setup week1 week1-dev week2 retrieval-train retrieval-eval week3 ranking-train ranking-eval week4 rerank download interactions images encode-text encode-image eda test lint fmt clean

setup:          ## create .venv and install the Week-1 stack
	uv sync

week1:          ## full Video_Games build (all reviews + items)
	uv run vlmrec week1

week1-dev:      ## fast capped run to prove the pipeline (subsampled reviews + images)
	uv run vlmrec week1 -o dataset.max_reviews=500000 filtering.k_core=3 images.max_images=500

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

# --- Week 2: retrieval (two-tower + FAISS + i2i) ---
week2:          ## train all 3 retrieval modes (content/hybrid/id) + run the ablation
	uv run vlmrec retrieval-train -o retrieval.feature_mode=content
	uv run vlmrec retrieval-train -o retrieval.feature_mode=hybrid
	uv run vlmrec retrieval-train -o retrieval.feature_mode=id
	uv run vlmrec retrieval-eval
retrieval-train:
	uv run vlmrec retrieval-train
retrieval-eval:
	uv run vlmrec retrieval-eval

# --- Week 3: ranking (DIN + DCN-v2 + MMoE) ---
week3:          ## train ranker + run the ablation
	uv run vlmrec ranking-train
	uv run vlmrec ranking-eval
ranking-train:
	uv run vlmrec ranking-train
ranking-eval:
	uv run vlmrec ranking-eval

# --- Week 4: pre-ranking + post-processing ---
week4:          ## hard-neg cascade fix + pre-ranker distill + diversity (MMR/DPP)
	uv run vlmrec rerank
rerank:
	uv run vlmrec rerank

test:           ## pure-logic unit tests (no network/GPU)
	uv run pytest -q
lint:
	uv run ruff check src tests
fmt:
	uv run ruff format src tests

clean:          ## remove generated artifacts (keeps code)
	rm -rf data/raw data/processed data/images data/embeddings
