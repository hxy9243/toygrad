# YuE2-3B RunPod Serverless & OpenAI-Compatible Serving Container
FROM runpod/base:0.6.2-cuda12.4.1

# Set non-interactive and environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=8008 \
    HOST=0.0.0.0 \
    YUE2_MODEL="m-a-p/YuE2-3B" \
    YUE2_VAE="m-a-p/YuE2-Vae" \
    YUE2_DEVICE="cuda" \
    YUE2_MEMORY_BUDGET_GIB=24

# Install system dependencies (libsndfile for audio processing, ffmpeg, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependency specifications and wheel
COPY yue2/requirements.txt .
COPY yue2/yue2_infer-0.1.5-py3-none-any.whl .

# Install python dependencies (including CUDA-enabled PyTorch and RunPod SDK)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt

# Copy application modules
COPY yue2/app/ ./app/

# Expose port for optional standalone HTTP/FastAPI serving
EXPOSE 8008

# Default to RunPod Serverless worker entrypoint
# (Can be overridden to ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8008"] for HTTP serving)
CMD ["python", "-u", "-m", "app.handler"]
