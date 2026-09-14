# YuE2-3B OpenAI-Compatible Serving Container
FROM python:3.12-slim-bookworm

# Set non-interactive and environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PORT=8000 \
    HOST=0.0.0.0 \
    YUE2_MODEL="m-a-p/YuE2-3B" \
    YUE2_VAE="m-a-p/YuE2-Vae" \
    YUE2_DEVICE="auto" \
    YUE2_MEMORY_BUDGET_GIB=24

# Install system dependencies (libsndfile for audio processing, ffmpeg, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications and wheel
COPY requirements.txt .
COPY yue2_infer-0.1.5-py3-none-any.whl .

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Start Uvicorn serving endpoint
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
