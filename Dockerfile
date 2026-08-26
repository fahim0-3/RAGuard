# Python 3.12 is pinned deliberately: PyTorch and sentence-transformers wheels
# are not reliably published for newer CPython releases yet.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_XET=1 \
    DISABLE_SAFETENSORS_CONVERSION=1 \
    PORT=8000

WORKDIR /app

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url "${TORCH_INDEX_URL}" \
    && pip install -r requirements-api.txt

COPY src/ ./src/
COPY api/ ./api/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

RUN addgroup --system --gid 10001 raguard \
    && adduser --system --uid 10001 --ingroup raguard --home /home/raguard raguard \
    && mkdir -p /models \
    && chown -R raguard:raguard /app /models /home/raguard

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-m", "scripts.container_healthcheck"]

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
