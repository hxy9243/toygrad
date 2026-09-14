"""RunPod Serverless Worker Handler for YuE2 music generation."""
from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch

from app.model_cache import ensure_model_cached, get_cache_dir, get_cache_info
from app.server import (
    MODEL_NAME,
    VAE_NAME,
    DEVICE,
    MOCK_MODE,
    SAMPLE_ABC,
    create_artifacts_zip,
    mock_generate,
    parse_style_and_lyrics,
    ChatCompletionRequest,
    ChatMessage,
)

_pipeline = None


def init_pipeline():
    """Warm up pipeline when worker container starts."""
    global _pipeline
    if MOCK_MODE:
        print("[RunPod Worker] MOCK_MODE enabled; skipping pipeline load.")
        return None

    if _pipeline is None:
        print("[RunPod Worker] Initializing model cache...")
        cache_info = get_cache_info()
        print(f"[RunPod Worker] Cache status: {cache_info}")

        # Ensure model weights are cached from network volume or HF
        ensure_model_cached(MODEL_NAME)
        ensure_model_cached(VAE_NAME)

        from yue2.pipeline import YuE2Pipeline

        backend = os.getenv("YUE2_BACKEND", "torch" if torch.cuda.is_available() else "torch-eager")
        device = os.getenv("YUE2_DEVICE", "cuda" if torch.cuda.is_available() else "auto")
        budget = float(os.getenv("YUE2_MEMORY_BUDGET_GIB", "24"))

        print(f"[RunPod Worker] Loading YuE2 pipeline (device={device}, backend={backend})...")
        _pipeline = YuE2Pipeline.from_pretrained(
            MODEL_NAME,
            vae=VAE_NAME,
            device=device,
            backend=backend,
            memory_budget_gib=budget,
            progress=True,
        )
        print("[RunPod Worker] Pipeline successfully loaded and ready.")
    return _pipeline


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod Serverless Handler function.
    Receives job['input'] and returns a JSON-serializable dictionary.
    """
    job_input = job.get("input", {})
    if not job_input:
        return {"status": "error", "message": "Missing 'input' in job payload."}

    # Extract messages if provided in OpenAI chat format
    raw_messages = job_input.get("messages", [])
    messages = [ChatMessage(role=m.get("role", "user"), content=m.get("content", "")) for m in raw_messages]

    req = ChatCompletionRequest(
        model=job_input.get("model", MODEL_NAME),
        messages=messages,
        style=job_input.get("style"),
        lyrics=job_input.get("lyrics"),
        cot=job_input.get("cot", "full"),
        seed=job_input.get("seed"),
        cfg_scale=float(job_input.get("cfg_scale", 1.0)),
        abc=job_input.get("abc"),
        stage=job_input.get("stage", "audio"),
        return_format=job_input.get("return_format", "json"),
        duration=float(job_input["duration"]) if job_input.get("duration") is not None else None,
        min_tokens=int(job_input["min_tokens"]) if job_input.get("min_tokens") is not None else None,
        max_tokens=int(job_input["max_tokens"]) if job_input.get("max_tokens") is not None else None,
    )

    style, lyrics = parse_style_and_lyrics(req)
    req_dict = {
        "model": req.model,
        "style": style,
        "lyrics": lyrics,
        "cot": req.cot,
        "seed": req.seed,
        "cfg_scale": req.cfg_scale,
        "abc": req.abc,
        "duration": req.duration,
        "min_tokens": req.min_tokens,
        "max_tokens": req.max_tokens,
    }

    start_t = time.perf_counter()

    if MOCK_MODE:
        abc, audio_bytes, plan_dict, result_dict, sem_tokens, latents = mock_generate(
            req_dict, stage=req.stage
        )
    else:
        pipe = init_pipeline()
        from app.server import run_real_generation
        abc, audio_bytes, plan_dict, result_dict, sem_tokens, latents = run_real_generation(
            pipe, req_dict, stage=req.stage
        )

    duration_sec = time.perf_counter() - start_t

    # Return based on requested format
    response_payload: Dict[str, Any] = {
        "status": "success",
        "stage": req.stage,
        "return_format": req.return_format,
        "execution_time_seconds": round(duration_sec, 3),
        "abc": abc,
    }

    if req.stage == "plan" or req.return_format == "abc":
        response_payload["plan"] = plan_dict
        return response_payload

    if req.return_format in ("arraybuffer", "zip"):
        zip_bytes = create_artifacts_zip(
            abc_text=abc,
            audio_bytes=audio_bytes,
            plan_dict=plan_dict,
            result_dict=result_dict,
            semantic_tokens=sem_tokens,
            latents=latents,
        )
        response_payload["zip_base64"] = base64.b64encode(zip_bytes).decode("ascii")
        return response_payload

    # Default "json" or "audio": return base64 audio and metadata
    if audio_bytes:
        response_payload["audio_base64"] = base64.b64encode(audio_bytes).decode("ascii")
        response_payload["audio_format"] = "flac"
        response_payload["sample_rate"] = 48000
    response_payload["result"] = result_dict

    return response_payload


if __name__ == "__main__":
    import runpod

    # Warm up pipeline upon worker launch
    init_pipeline()
    # Start polling RunPod serverless job queue
    runpod.serverless.start({"handler": handler})
