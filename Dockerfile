# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_FROZEN=1 \
    UV_CACHE_DIR=/tmp/uv \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1

# git: uv fetches liftus/snoopy from the judit repository during sync.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
# Drop the root-owned build cache so a non-root runtime uid can recreate it.
RUN uv sync --frozen --no-dev \
 && rm -rf /tmp/uv

CMD ["uv", "run", "--no-sync", "python", "-c", "print('blend image ready')"]
