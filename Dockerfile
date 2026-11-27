# -----------------------------------------------------------
# Multi-stage Dockerfile: build -> final (slim, non-root)
# -----------------------------------------------------------
# Build stage: install deps, prefetch embedder
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce AS builder

WORKDIR /build

# Build-time dependencies only (removed at the end of this stage)
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-setuptools && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install to a local prefix so we can copy it to the final image.
RUN pip install --no-cache-dir --timeout=300 --retries=20 \
    --prefix=/install setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=20 \
    --prefix=/install -r requirements.txt

# Prefetch the fastembed ONNX model so the first container start is instant.
# Use PYTHONPATH so the just-installed packages are importable.
ENV FASTEMBED_CACHE_PATH=/model-cache \
    HF_HOME=/model-cache/hf \
    PYTHONPATH=/install/lib/python3.11/site-packages
ARG LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='${LOCAL_EMBEDDING_MODEL}')"

# Prefetch the default Whisper model so the first voice note needs no download.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8')"

# -----------------------------------------------------------
# Final stage: runtime only
# -----------------------------------------------------------
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce AS final

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/model-cache \
    HF_HOME=/model-cache/hf

WORKDIR /app

# Copy compiled wheels from builder
COPY --from=builder /install /usr/local
# Copy prefetched embedder model
COPY --from=builder /model-cache /model-cache

# Non-root user (uid 10001). The bind-mounted cache directory must be owned by
# this user (or group 10001) so diskcache can write its SQLite file.
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g 10001 -m -s /sbin/nologin appuser && \
    chown -R appuser:appgroup /app /model-cache

# Application code (copied as appuser so COPY --chown works below)
COPY --chown=appuser:appgroup . .

# Runtime dirs (owned by appuser)
RUN mkdir -p /app/documents /app/cache && \
    chown -R appuser:appgroup /app/documents /app/cache

USER appuser

EXPOSE 8000

CMD ["python", "run.py"]