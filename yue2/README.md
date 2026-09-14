# YuE2-3B OpenAI-Compatible Serving Endpoint

OpenAI API-compatible serving server for [🤗 YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) with options for:
- **ABC score only** (`return_format: "abc"`): Symbolic melody and chord plan returned in standard OpenAI chat completion text.
- **ArrayBuffer delivery** (`return_format: "arraybuffer"`): Binary in-memory archive containing all generated artifacts (`score.abc`, `audio.flac`, `plan.json`, `result.json`, `semantic.npy`, `latent.npy`), ready for browser/client `response.arrayBuffer()`.
- **Standard OpenAI Chat Completion** (`return_format: "json"`): Base64-encoded audio and generated ABC score in the standard OpenAI chat response.

---

## 🚀 Quick Start

### 1. Local Python Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and local yue2-infer wheel
pip install --upgrade pip
pip install -r requirements.txt

# Run unit and integration tests (mock mode)
pytest app/test_server.py -v

# Start the OpenAI-compatible FastAPI server locally
python -m app.server
# Or: uvicorn app.server:app --host 0.0.0.0 --port 8008
```

### 2. Docker Setup

```bash
# Build and run using Docker Compose (simulating volume mount)
docker compose up -d

# Or build manually for GPU
docker build -t yue2-serving:latest .

# Run as RunPod Serverless worker:
docker run --gpus all -v yue2-volume:/runpod-volume yue2-serving:latest

# Or run as standalone HTTP server:
docker run -p 8008:8008 --gpus all -v yue2-volume:/runpod-volume yue2-serving:latest \
  uvicorn app.server:app --host 0.0.0.0 --port 8008
```

---

## ⚡ RunPod Serverless Deployment

### 1. Architecture & Model Caching
- **Base Image**: `runpod/base:0.6.2-cuda12.4.1` with CUDA 12.4 runtime and PyTorch GPU support.
- **Model Caching (Zero Image Bloat)**:
  Instead of baking 10GB+ weights into the image, attach a **RunPod Network Volume** mounted to `/runpod-volume`.
  `app/model_cache.py` directs `HF_HOME` to `/runpod-volume/cache/huggingface`.
  - On first worker cold-start in a data center region, weights download with multi-stream `hf-transfer` into the shared volume.
  - On subsequent cold starts, any worker attaching the same volume boots instantly from disk cache.
  - If no network volume is attached, it gracefully falls back to local disk storage (`~/.cache/huggingface`).

### 2. Worker Handler (`app/handler.py`)
RunPod Serverless invokes `app.handler:handler`.

#### Example Serverless Job Payload:
```json
{
  "input": {
    "model": "YuE2-3B",
    "style": "Synthwave, 80s analog synth, driving bassline",
    "lyrics": "[Verse]\nCity of neon lights\n[Chorus]\nRunning through the night",
    "stage": "audio",
    "return_format": "json",
    "duration": 30.0
  }
}
```

#### Example Serverless Job Response:
```json
{
  "status": "success",
  "stage": "audio",
  "return_format": "json",
  "execution_time_seconds": 12.4,
  "abc": "X:1\nT:...",
  "audio_base64": "...",
  "audio_format": "flac",
  "sample_rate": 48000,
  "result": { ... }
}
```

---

## 📡 API Usage Examples (FastAPI / HTTP Mode)

### Option 1: Return ABC Notation Only (`stage="plan"` / `return_format="abc"`)

#### Python (OpenAI SDK):
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8008/v1", api_key="dummy")

response = client.chat.completions.create(
    model="YuE2-3B",
    messages=[
        {"role": "system", "content": "Style: Jazz-funk, rhodes piano, electric bass"},
        {"role": "user", "content": "[Verse]\nLate night city lights\n[Chorus]\nGrooving till sunrise"}
    ],
    extra_body={
        "return_format": "abc"  # Requests symbolic ABC score only
    }
)

print(response.choices[0].message.content)
# Output:
# X:1
# T:Tonight Awake
# M:4/4
# ...
```

#### cURL:
```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YuE2-3B",
    "style": "Jazz-funk, rhodes piano, electric bass",
    "lyrics": "[Verse]\nLate night city lights\n[Chorus]\nGrooving till sunrise",
    "return_format": "abc"
  }'
```

---

### Option 2: Return ArrayBuffer containing all generated files

#### JavaScript / Browser / Node.js:
```javascript
const response = await fetch("http://localhost:8008/v1/chat/completions", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Accept": "application/octet-stream"
  },
  body: JSON.stringify({
    model: "YuE2-3B",
    style: "Cyber Metal, heavy synthesizer, distorted guitar",
    lyrics: "[Verse]\nNeon pulse in the wire\n[Chorus]\nIgnite the fire",
    return_format: "arraybuffer",
    seed: 42
  })
});

// Binary ArrayBuffer containing the full archive:
// - score.abc
// - audio.flac (48kHz stereo)
// - plan.json
// - result.json
// - semantic.npy
// - latent.npy
const arrayBuffer = await response.arrayBuffer();

// To save or extract in Node.js:
const fs = require('fs');
fs.writeFileSync('yue2_artifacts.zip', Buffer.from(arrayBuffer));
```

#### cURL:
```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: application/octet-stream" \
  -d '{
    "model": "YuE2-3B",
    "style": "Cyber Metal, heavy synthesizer",
    "lyrics": "[Verse]\nNeon pulse\n[Chorus]\nIgnite",
    "return_format": "arraybuffer"
  }' \
  --output yue2_artifacts.zip

# Inspect all generated files
unzip -l yue2_artifacts.zip
```

---

## 💻 Local Hardware Diagnostics & Inference Status

Testing conducted on this laptop:
- **Processor**: Intel(R) Core(TM) i5-1340P (12 cores, 16 threads)
- **RAM**: 60.5 GiB RAM (30 GiB available)
- **GPU**: Intel Iris Xe Integrated Graphics (No discrete NVIDIA GPU / CUDA)
- **Framework**: PyTorch 2.10.0 + Transformers 4.57.6

### Feasibility Verdict:
1. **Memory**: **PASS** (60GB RAM exceeds the 24GB host RAM requirement).
2. **Architecture**: **PASS** (Pure PyTorch BF16 CPU matmul and SDPA attention execute cleanly without CUDA extensions).
3. **Inference by Mode**:
   - **ABC Score Planning (`stage="plan"` / `return_format="abc"`)**: **FEASIBLE on CPU** (~1-3 minutes).
   - **Full Audio Generation (`stage="audio"` / `return_format="arraybuffer"`)**: **COMPUTE LIMITED ON CPU**. A 3.6-minute song takes 71 seconds on an RTX 4090 (11 GiB VRAM), but is estimated at ~30-90+ minutes on CPU due to mobile memory bandwidth constraints.
   - For fast local testing, set `MOCK_YUE2=1`. For production full-song audio generation, deploy to an NVIDIA GPU (24GB VRAM) instance using the included `Dockerfile` / `docker-compose.yml`.
