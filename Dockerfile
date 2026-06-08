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

# system libs + curl (needed to download models from GitHub Releases)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# copy installed packages from builder
COPY --from=builder /install/packages /usr/local

# copy source code and frontend
COPY src/  /app/src/
COPY web/  /app/web/

# ── Download model weights from GitHub Releases ────────────────────────────
# Uses /releases/latest/download/ so no tag name is needed.
# Only the two models required by the web backend are downloaded.
ARG GITHUB_REPO=SalmanAlsulami/drowsiness-detection
ARG RELEASE_BASE=https://github.com/${GITHUB_REPO}/releases/latest/download

RUN mkdir -p /app/outputs/models && \
    curl -fSL "${RELEASE_BASE}/efficientnetv2s_cbam_main_best.pth" \
         -o /app/outputs/models/efficientnetv2s_cbam_main_best.pth && \
    curl -fSL "${RELEASE_BASE}/efficientnetv2s_cbam_yawn_best.pth" \
         -o /app/outputs/models/efficientnetv2s_cbam_yawn_best.pth

EXPOSE 7860

# Railway injects $PORT; HF Spaces uses 7860; fall back to 7860 locally
CMD uvicorn web.backend.main:app --host 0.0.0.0 --port ${PORT:-7860}
