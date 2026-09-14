"""Model caching utilities for RunPod Serverless and GPU workers."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

RUNPOD_VOLUME_PATH = Path("/runpod-volume")
DEFAULT_CACHE_SUBDIR = "cache/huggingface"


def get_cache_dir() -> Path:
    """Determine the optimal cache directory with precedence:
    1. /runpod-volume/cache/huggingface (if RunPod Network Volume is mounted)
    2. HF_HOME environment variable
    3. ~/.cache/huggingface fallback
    """
    if RUNPOD_VOLUME_PATH.exists() and RUNPOD_VOLUME_PATH.is_dir():
        volume_cache = RUNPOD_VOLUME_PATH / DEFAULT_CACHE_SUBDIR
        try:
            volume_cache.mkdir(parents=True, exist_ok=True)
            return volume_cache
        except PermissionError:
            pass

    if os.getenv("HF_HOME"):
        p = Path(os.environ["HF_HOME"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    fallback = Path.home() / ".cache" / "huggingface"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# Configure HF environment
CACHE_DIR = get_cache_dir()
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")

# Enable high-speed hf_transfer if installed
try:
    import hf_transfer  # noqa: F401
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass


def get_cache_info() -> Dict[str, Any]:
    """Return status of model cache and storage."""
    cache_path = get_cache_dir()
    is_network_volume = str(RUNPOD_VOLUME_PATH) in str(cache_path)
    total, used, free = shutil.disk_usage(cache_path)

    return {
        "cache_dir": str(cache_path),
        "is_network_volume": is_network_volume,
        "disk_total_gib": round(total / (1024**3), 2),
        "disk_used_gib": round(used / (1024**3), 2),
        "disk_free_gib": round(free / (1024**3), 2),
    }


def ensure_model_cached(
    repo_id: str,
    revision: Optional[str] = None,
    allow_patterns: Optional[list[str]] = None,
    ignore_patterns: Optional[list[str]] = None,
) -> str:
    """Ensure the specified Hugging Face repository snapshot is cached.
    Returns the local path to the cached snapshot.
    """
    from huggingface_hub import snapshot_download

    cache_path = get_cache_dir()
    print(f"Ensuring model '{repo_id}' is cached in {cache_path}...")

    local_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_path,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        resume_download=True,
    )
    print(f"Model '{repo_id}' ready at: {local_path}")
    return local_path
