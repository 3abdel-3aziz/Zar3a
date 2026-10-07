# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Zar3a Backend — Single-stage Docker build
#
# Why single-stage?
#   A multi-stage build was attempted (builder → runtime COPY of site-packages)
#   but the ML/AI dependency tree (~15 GB) caused the BuildKit gRPC stream to
#   drop with EOF during the inter-stage COPY.  Since every package is a
#   runtime requirement (nothing to strip), single-stage is cleaner and
#   produces an identical final image.
#
# Why CPU-only PyTorch?
#   Docker Desktop (WSL2 backend) cannot access the host GPU without explicit
#   nvidia-docker runtime configuration.  The default PyPI resolution of
#   `torch>=2.0.0` pulls the CUDA 13 wheel + 16 separate NVIDIA library
#   packages (~12 GB total) that are entirely inert inside a standard
#   container.  torch is overridden to the CPU-only index in pyproject.toml,
#   reducing that footprint to ~250 MB.
#
# Why --mount=type=cache?
#   BuildKit cache mounts persist the uv download cache across builds on the
#   same host.  Subsequent `docker compose up --build` runs skip all downloads
#   and only re-install if pyproject.toml / uv.lock actually changed.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
# build-essential : native extension compilation (shapely, pyproj, lxml, …)
# libgomp1        : OpenMP runtime required by LightGBM and XGBoost at runtime
# git             : required by some VCS-pinned deps during install
# curl            : used by the docker-compose.yml healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── uv package manager ────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ── Dependency layer ──────────────────────────────────────────────────────────
# IMPORTANT: copy ONLY the manifest files before running the install.
# Docker caches this layer keyed on the contents of pyproject.toml + uv.lock.
# The heavy install step is only re-executed when those files actually change.
COPY pyproject.toml uv.lock ./

# --mount=type=cache   : BuildKit cache — persists uv's wheel/sdist cache
#                        across builds on this host (no re-downloads on rebuild)
# --system             : install into the container's system Python (no venv)
# No --no-cache flag   : allow uv to read from / write to the mounted cache dir
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system -r pyproject.toml

# ── Application source ────────────────────────────────────────────────────────
# Build context = repo root (docker-compose.yml: context: .)
# back_end/src/ → /app/src/
COPY back_end/src/ ./src/

# Expose all sub-packages under src/ as top-level importable modules.
# e.g. `from api.main import app` and `from recommendation_system.engine import …`
ENV PYTHONPATH="/app/src"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]