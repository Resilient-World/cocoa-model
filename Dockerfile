# Multi-stage: builder (uv) → runtime (non-root API)
FROM python:3.12-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY data/external ./data/external

RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache pip setuptools wheel && \
    uv pip install --no-cache -e ".[observability]"

FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/src /app/src
COPY --from=builder /build/config /app/config
COPY --from=builder /build/data/external /app/data/external
COPY pyproject.toml README.md ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    API_PORT=8001

USER appuser
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"API_PORT\", \"8001\")}/health')" || exit 1

# Required deployment env vars:
# - COCOA_MODEL_API_KEY: long random string generated with `openssl rand -hex 32`
# - COCOA_MODEL_ALLOWED_ORIGINS: comma-separated caller origins, e.g. https://backend.example.com
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT:-8001}"]
