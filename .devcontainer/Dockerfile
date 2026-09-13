# Development image: the CLI's own deps plus the case-study stack (CPU-only
# PyTorch, notebook tooling, Typst for the paper). Kept separate from the
# repo-root Dockerfile, which stays a minimal runtime image.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git make curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install dependencies from pyproject alone first, so editing source does not
# invalidate the (large) torch layer.
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/unalloc && echo '__version__ = "0.0.0"' > src/unalloc/__init__.py \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      -e ".[dev,research]" \
 && rm -rf src

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg
