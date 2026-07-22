# Multi-stage: the builder resolves dependencies, the runtime carries only
# what is needed to run. Keeps the image small and the attack surface narrow.
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependency layer first, so source edits do not invalidate the install cache.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY sentinel/ ./sentinel/
COPY simulator/ ./simulator/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.11-slim AS runtime

# Run unprivileged. This process can execute remediation actions; it has no
# business running as root.
RUN groupadd --system sentinel && \
    useradd --system --gid sentinel --create-home sentinel

WORKDIR /app
COPY --from=builder --chown=sentinel:sentinel /app /app
COPY --chown=sentinel:sentinel data/ ./data/
COPY --chown=sentinel:sentinel ui/ ./ui/
COPY --chown=sentinel:sentinel mcp.config.json ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER sentinel
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"

CMD ["uvicorn", "sentinel.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
