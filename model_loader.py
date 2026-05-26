"""
Model loader singleton.

Loads the LTX-Video FP8 / mixed checkpoint via diffusers LTXImageToVideoPipeline
and keeps it resident in GPU memory for the lifetime of the worker process.
The loader is thread-safe and lazy: the first call to `get_pipeline()` triggers
the download + load; subsequent calls return the cached pipeline immediately.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

_lock = threading.Lock()
_pipeline = None
_text_pipeline = None
_load_time: Optional[float] = None

# ---------------------------------------------------------------------------
# Safetensors filename map
# ---------------------------------------------------------------------------
_CKPT_MAP = {
    "fp8": "ltx-video-2b-v0.9-fp8_e4m3fn.safetensors",
    "mixed": "ltx-video-2b-v0.9-mixed.safetensors",
}


def _resolve_dtype():
    import torch

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(settings.torch_dtype, torch.bfloat16)


def _load_pipeline():
    """
    Loads LTXImageToVideoPipeline from the FP8 HF repo.
    Falls back gracefully to bfloat16 diffusers path if single-file load fails.
    """
    import torch
    from diffusers import LTXImageToVideoPipeline, LTXPipeline
    from huggingface_hub import hf_hub_download

    dtype = _resolve_dtype()
    device = settings.device
    cache_dir = Path(settings.model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ckpt_filename = _CKPT_MAP[settings.model_variant]

    logger.info(
        "model.loading",
        repo=settings.model_repo,
        variant=settings.model_variant,
        ckpt=ckpt_filename,
        dtype=str(dtype),
        device=device,
    )
    t0 = time.time()

    hf_kwargs = {}
    if settings.hf_token:
        hf_kwargs["token"] = settings.hf_token

    # Download the single safetensors checkpoint
    ckpt_path = hf_hub_download(
        repo_id=settings.model_repo,
        filename=ckpt_filename,
        cache_dir=str(cache_dir),
        **hf_kwargs,
    )
    logger.info("model.checkpoint_downloaded", path=ckpt_path)

    # For the text encoder / VAE / scheduler we borrow them from the official
    # Lightricks base repo (publicly accessible, no token needed).
    BASE_REPO = "Lightricks/LTX-Video"

    i2v_pipe = LTXImageToVideoPipeline.from_pretrained(
        BASE_REPO,
        torch_dtype=dtype,
        cache_dir=str(cache_dir),
        **hf_kwargs,
    )
    # Swap the transformer weights for the FP8 checkpoint
    from safetensors.torch import load_file as st_load

    state_dict = st_load(ckpt_path, device="cpu")
    # Filter to transformer keys only (handles both bare and prefixed checkpoints)
    transformer_sd = {
        k.removeprefix("transformer."): v
        for k, v in state_dict.items()
        if k.startswith("transformer.") or not any(
            k.startswith(p) for p in ("vae.", "text_encoder.", "scheduler.")
        )
    }
    missing, unexpected = i2v_pipe.transformer.load_state_dict(
        transformer_sd, strict=False
    )
    if missing:
        logger.warning("model.transformer.missing_keys", count=len(missing))
    if unexpected:
        logger.warning("model.transformer.unexpected_keys", count=len(unexpected))

    i2v_pipe = i2v_pipe.to(device)
    # Enable memory-efficient attention where available
    try:
        i2v_pipe.enable_xformers_memory_efficient_attention()
        logger.info("model.xformers_enabled")
    except Exception:
        pass

    # T2V pipeline shares all sub-models except the conditioning input
    t2v_pipe = LTXPipeline(
        transformer=i2v_pipe.transformer,
        vae=i2v_pipe.vae,
        text_encoder=i2v_pipe.text_encoder,
        tokenizer=i2v_pipe.tokenizer,
        scheduler=i2v_pipe.scheduler,
    ).to(device)

    elapsed = time.time() - t0
    logger.info("model.loaded", elapsed_s=round(elapsed, 2))
    return i2v_pipe, t2v_pipe


def get_pipelines():
    """Return (i2v_pipeline, t2v_pipeline), loading them on first call."""
    global _pipeline, _text_pipeline, _load_time
    if _pipeline is None:
        with _lock:
            if _pipeline is None:  # double-checked
                _pipeline, _text_pipeline = _load_pipeline()
                _load_time = time.time()
    return _pipeline, _text_pipeline


def is_model_loaded() -> bool:
    return _pipeline is not None


def model_load_time() -> Optional[float]:
    return _load_time
