# ── Stage 1: build deps ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# system libs required by OpenCV headless + MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY web/backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/packages -r requirements.txt


# ── Stage 2: runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# same system libs needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# copy installed packages from builder
COPY --from=builder /install/packages /usr/local

# copy only what the server needs
COPY src/              /app/src/
COPY outputs/models/   /app/outputs/models/
COPY web/              /app/web/

EXPOSE 8000

# Railway injects $PORT; fall back to 8000 locally
CMD uvicorn web.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
