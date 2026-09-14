"""OpenAI API-compatible serving server for YuE2-3B music generation."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Application configuration
try:
    from app.model_cache import get_cache_info
except ImportError:
    from model_cache import get_cache_info

MODEL_NAME = os.getenv("YUE2_MODEL", "m-a-p/YuE2-3B")
VAE_NAME = os.getenv("YUE2_VAE", "m-a-p/YuE2-Vae")
DEVICE = os.getenv("YUE2_DEVICE", "auto")
BUDGET_GIB = float(os.getenv("YUE2_MEMORY_BUDGET_GIB", "24"))
BACKEND = os.getenv("YUE2_BACKEND", "torch-eager" if not torch.cuda.is_available() else "torch")
MOCK_MODE = os.getenv("MOCK_YUE2", "0") == "1"

app = FastAPI(
    title="YuE2-3B OpenAI Compatible Serving Server",
    description="Serving endpoint for YuE2 music generation with ABC-only and ArrayBuffer delivery options.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_lock = asyncio.Lock()
_pipeline = None


def get_pipeline():
    global _pipeline
    if MOCK_MODE:
        return None
    if _pipeline is None:
        from yue2.pipeline import YuE2Pipeline
        print(f"Loading YuE2 pipeline (model={MODEL_NAME}, vae={VAE_NAME}, device={DEVICE}, backend={BACKEND})...")
        _pipeline = YuE2Pipeline.from_pretrained(
            MODEL_NAME,
            vae=VAE_NAME,
            device=DEVICE,
            backend=BACKEND,
            memory_budget_gib=BUDGET_GIB,
            progress=True,
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Pydantic Request & Response Models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage] = []

    # Direct YuE2 parameters (or extracted from messages)
    style: Optional[str] = None
    lyrics: Optional[str] = None
    cot: Literal["full", "melody", "off"] = "full"
    seed: Optional[int] = None
    cfg_scale: float = 1.0
    abc: Optional[str] = None

    # Serving & delivery options
    stage: Literal["plan", "audio"] = "audio"
    return_format: Literal["json", "abc", "arraybuffer", "audio"] = "json"
    duration: Optional[float] = None  # Desired duration in seconds (used in mock mode or to guide min_tokens)
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "m-a-p"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def parse_style_and_lyrics(req: ChatCompletionRequest) -> tuple[str, str]:
    """Extract style and lyrics either from explicit fields or messages."""
    style = req.style
    lyrics = req.lyrics

    if not style or not lyrics:
        for msg in req.messages:
            content = msg.content.strip()
            if msg.role == "system" and not style:
                style = content.removeprefix("Style:").strip()
            elif msg.role == "user":
                if "Style:" in content and "Lyrics:" in content:
                    parts = content.split("Lyrics:", 1)
                    style_part = parts[0].replace("Style:", "").strip()
                    lyrics_part = parts[1].strip()
                    if not style:
                        style = style_part
                    if not lyrics:
                        lyrics = lyrics_part
                elif not lyrics:
                    lyrics = content

    if not style:
        style = "Mandarin, warm piano, acoustic pop, female vocal"
    if not lyrics:
        lyrics = "[Verse]\n晚风轻轻吹过窗前\n你留下的笑还在昨天\n[Chorus]\n让这首歌陪你走远\n把所有想念唱成明天"

    return style, lyrics


def create_artifacts_zip(
    abc_text: Optional[str],
    audio_bytes: Optional[bytes],
    plan_dict: dict,
    result_dict: dict,
    semantic_tokens: Optional[List[int]] = None,
    latents: Optional[np.ndarray] = None,
) -> bytes:
    """Pack all generated files into an in-memory ZIP (ArrayBuffer)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if abc_text:
            zf.writestr("score.abc", abc_text.encode("utf-8"))
        if audio_bytes:
            zf.writestr("audio.flac", audio_bytes)
        zf.writestr("plan.json", json.dumps(plan_dict, indent=2).encode("utf-8"))
        zf.writestr("result.json", json.dumps(result_dict, indent=2).encode("utf-8"))
        zf.writestr("request.json", json.dumps(result_dict.get("request", {}), indent=2).encode("utf-8"))
        zf.writestr("config.json", json.dumps(result_dict.get("config", {}), indent=2).encode("utf-8"))

        if semantic_tokens is not None:
            npy_sem = io.BytesIO()
            np.save(npy_sem, np.asarray(semantic_tokens, dtype=np.int32))
            zf.writestr("semantic.npy", npy_sem.getvalue())

        if latents is not None:
            npy_lat = io.BytesIO()
            np.save(npy_lat, latents.astype(np.float32))
            zf.writestr("latent.npy", npy_lat.getvalue())

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Mock Generation for Testing & Validation
# ---------------------------------------------------------------------------

SAMPLE_ABC = """X:1
T:Tonight Awake
M:4/4
L:1/8
Q:1/4=120
K:C
"C" c2 e2 g2 c'2 | "G" b2 g2 d2 B2 | "Am" A2 c2 e2 a2 | "F" f4 e2 d2 |
w: 晚 风 轻 轻 吹 过 窗 前, 你 留 下 的 笑 还 在 昨 天
"C" c2 e2 g2 c'2 | "G" d'2 b2 g2 d2 | "F" c4 "G" d4 | "C" c8 |
w: 让 这 首 歌 陪 你 走 远, 把 所 有 想 念 唱 成 明 天 ||"""


def mock_generate(req_data: dict, stage: str):
    style = req_data["style"]
    lyrics = req_data["lyrics"]
    seed = req_data.get("seed", 42) or 42
    abc = req_data.get("abc") or SAMPLE_ABC

    plan_dict = {
        "style": style,
        "lyrics": lyrics,
        "cot": req_data.get("cot", "full"),
        "seed": seed,
        "abc": abc,
        "timing": {"seconds": 0.5, "output_tokens": 128},
    }

    if stage == "plan":
        result_dict = {
            "status": "complete",
            "stage": "plan",
            "identity": f"mock-{uuid.uuid4()}",
            "timing": {"plan_seconds": 0.5},
            "request": req_data,
        }
        return abc, None, plan_dict, result_dict, None, None

    # Mock audio generation: generate stereo 48kHz music of requested duration
    sample_rate = 48000
    duration = float(req_data.get("duration") or 30.0)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate melodic harmonic chords across the duration
    left = 0.15 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 554.37 * t)
    right = 0.15 * np.sin(2 * np.pi * 659.25 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    # Envelope to prevent clicks
    env = np.ones_like(t)
    fade_len = int(sample_rate * 0.1)
    if len(t) > 2 * fade_len:
        env[:fade_len] = np.linspace(0, 1, fade_len)
        env[-fade_len:] = np.linspace(1, 0, fade_len)
    audio_data = np.stack([left * env, right * env], axis=-1).astype(np.float32)

    audio_buf = io.BytesIO()
    sf.write(audio_buf, audio_data, sample_rate, format="FLAC", subtype="PCM_24")
    audio_bytes = audio_buf.getvalue()

    token_count = int(duration * 42)
    semantic_tokens = list(range(token_count))
    latents = np.zeros((64, token_count), dtype=np.float32)

    result_dict = {
        "status": "complete",
        "stage": "audio",
        "sample_rate": sample_rate,
        "audio_seconds": duration,
        "identity": f"mock-{uuid.uuid4()}",
        "timing": {"e2e_seconds": 1.2, "vae_seconds": 0.3},
        "request": req_data,
        "config": {"seed": seed, "stage": stage, "duration": duration},
    }

    return abc, audio_bytes, plan_dict, result_dict, semantic_tokens, latents


# ---------------------------------------------------------------------------
# Real Pipeline Execution
# ---------------------------------------------------------------------------

def run_real_generation(pipe, req_data: dict, stage: str):
    style = req_data["style"]
    lyrics = req_data["lyrics"]
    cot = req_data.get("cot", "full")
    seed = req_data.get("seed")
    if seed is None:
        seed = int(time.time()) % 1_000_000
    cfg_scale = req_data.get("cfg_scale", 1.0)
    provided_abc = req_data.get("abc")

    # Build semantic sampling controls for song duration
    semantic_sampling = {}
    if req_data.get("min_tokens"):
        semantic_sampling["min_tokens"] = int(req_data["min_tokens"])
    elif req_data.get("duration"):
        # ~42 tokens per second of 48kHz audio in YuE2
        semantic_sampling["min_tokens"] = max(200, int(float(req_data["duration"]) * 42))

    if req_data.get("max_tokens"):
        semantic_sampling["max_tokens"] = int(req_data["max_tokens"])
    elif req_data.get("duration"):
        semantic_sampling["max_tokens"] = max(
            semantic_sampling.get("min_tokens", 200),
            int(float(req_data["duration"]) * 42) + 200,
        )

    if stage == "plan":
        plan = pipe.plan(style=style, lyrics=lyrics, cot=cot, seed=seed, abc=provided_abc)
        abc = plan.abc
        plan_dict = {
            "style": style,
            "lyrics": lyrics,
            "cot": cot,
            "seed": seed,
            "abc": abc,
            "timing": plan.timing,
            "truncated": plan.truncated,
        }
        result_dict = {
            "status": "complete",
            "stage": "plan",
            "timing": plan.timing,
            "request": req_data,
            "truncated": plan.truncated,
        }
        return abc, None, plan_dict, result_dict, None, None

    # Full audio generation
    song = pipe(
        style=style,
        lyrics=lyrics,
        cot=cot,
        seed=seed,
        cfg_scale=cfg_scale,
        abc=provided_abc,
        semantic_sampling=semantic_sampling or None,
    )
    abc = song.abc
    audio_buf = io.BytesIO()
    sf.write(audio_buf, song.audio, song.sample_rate, format="FLAC", subtype="PCM_24")
    audio_bytes = audio_buf.getvalue()

    plan_dict = {
        "style": style,
        "lyrics": lyrics,
        "cot": cot,
        "seed": seed,
        "abc": abc,
        "timing": song.timing.get("abc", {}),
    }
    result_dict = {
        "status": "complete",
        "stage": "audio",
        "identity": song.request_identity,
        "sample_rate": song.sample_rate,
        "audio_seconds": len(song.audio) / song.sample_rate,
        "timing": song.timing,
        "weights": song.weights,
        "config": song.config,
        "truncated": song.truncated,
        "request": req_data,
    }
    return (
        abc,
        audio_bytes,
        plan_dict,
        result_dict,
        song.semantic.tokens,
        song.latents,
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
@app.get("/v1/health")
async def health():
    return {
        "status": "healthy",
        "mock_mode": MOCK_MODE,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model": MODEL_NAME,
        "vae": VAE_NAME,
        "cache": get_cache_info(),
    }


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    return ModelListResponse(
        data=[
            ModelCard(id=MODEL_NAME),
            ModelCard(id=VAE_NAME),
            ModelCard(id="YuE2-3B"),
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    accept: Optional[str] = Header(None),
):
    style, lyrics = parse_style_and_lyrics(req)
    seed = req.seed if req.seed is not None else int(time.time()) % 1_000_000

    # Determine requested format & stage
    stage = req.stage
    return_format = req.return_format

    # Header-based format detection
    if accept and "application/octet-stream" in accept or accept == "application/zip":
        return_format = "arraybuffer"
    elif accept == "text/vnd.abc" or accept == "text/plain":
        return_format = "abc"

    if return_format == "abc":
        stage = "plan"

    req_data = {
        "style": style,
        "lyrics": lyrics,
        "cot": req.cot,
        "seed": seed,
        "cfg_scale": req.cfg_scale,
        "abc": req.abc,
        "stage": stage,
        "duration": req.duration,
        "min_tokens": req.min_tokens,
        "max_tokens": req.max_tokens,
    }

    async with pipeline_lock:
        try:
            if MOCK_MODE:
                abc, audio_bytes, plan_dict, result_dict, semantic_tokens, latents = mock_generate(
                    req_data, stage
                )
            else:
                pipe = get_pipeline()
                loop = asyncio.get_running_loop()
                abc, audio_bytes, plan_dict, result_dict, semantic_tokens, latents = await loop.run_in_executor(
                    None, run_real_generation, pipe, req_data, stage
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {str(e)}",
            )

    # 1. Return raw ABC text if requested
    if accept in {"text/vnd.abc", "text/plain"} and return_format == "abc":
        return Response(content=abc or "", media_type="text/plain; charset=utf-8")

    # 2. Return ArrayBuffer (binary ZIP containing all generated files)
    if return_format == "arraybuffer":
        zip_bytes = create_artifacts_zip(
            abc_text=abc,
            audio_bytes=audio_bytes,
            plan_dict=plan_dict,
            result_dict=result_dict,
            semantic_tokens=semantic_tokens,
            latents=latents,
        )
        return Response(
            content=zip_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="yue2_artifacts_{seed}.zip"',
                "X-YuE2-Stage": stage,
                "X-YuE2-Seed": str(seed),
                "X-YuE2-Status": "complete",
            },
        )

    # 3. Return Standard OpenAI Chat Completion JSON format
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    # Build response message content
    if return_format == "abc" or stage == "plan":
        message_content = abc or ""
        audio_payload = None
    else:
        message_content = f"Generated full song in style: {style}\n\n```abc\n{abc}\n```"
        audio_payload = {
            "id": f"audio-{uuid.uuid4().hex[:8]}",
            "data": base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else "",
            "format": "flac",
            "transcript": abc or "",
        }

    openai_response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": message_content,
                    **({"audio": audio_payload} if audio_payload else {}),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(lyrics.split()) + len(style.split()),
            "completion_tokens": len(abc.split()) if abc else 0,
            "total_tokens": len(lyrics.split()) + len(style.split()) + (len(abc.split()) if abc else 0),
        },
        "yue2_metadata": {
            "stage": stage,
            "seed": seed,
            "cot": req.cot,
            "timing": result_dict.get("timing", {}),
        },
    }

    return JSONResponse(content=openai_response)


@app.post("/v1/audio/generations")
@app.post("/v1/music/generate")
async def music_generations(req: ChatCompletionRequest):
    """Direct music endpoint returning audio or arraybuffer directly."""
    return await chat_completions(req)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8008"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("server:app", host=host, port=port, reload=False)
