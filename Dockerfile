# syntax=docker/dockerfile:1
# Serving image for the VLM-Rec cascade (CPU). Model/data artifacts are mounted at runtime.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
WORKDIR /app

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs

# Install runtime + serving + retrieval (FAISS) deps. (torch resolves to the pinned CUDA build,
# which runs on CPU when no GPU is present — fine for serving; swap to a CPU wheel to slim the image.)
RUN uv sync --extra serving --extra retrieval --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "vlmrec.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
