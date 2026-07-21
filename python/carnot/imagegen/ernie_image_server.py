"""Local OpenAI-Images-API-compatible HTTP shim for baidu/ERNIE-Image-Turbo.

**Researcher summary:**
    Before 2026-07-21, generating a diagram meant calling a paid API
    (Gemini image-gen, GPT-Image, ...). Those tokens are no longer available.
    This module serves ``baidu/ERNIE-Image-Turbo`` (an 8B-parameter Diffusion
    Transformer text-to-image model distilled via DMD+RL for 8-step
    inference instead of the base model's 50, Apache-2.0, runs on a single
    24GB consumer GPU) behind the same HTTP contract the OpenAI Python SDK's
    ``client.images.generate(...)`` call expects, so the vendored
    ``paperbanana`` tool can be pointed at it via ``OPENAI_BASE_URL`` with
    zero changes to paperbanana's own source.

    2026-07-21 update: switched from the base ``baidu/ERNIE-Image`` (50
    steps, guidance_scale=4.0) to ``baidu/ERNIE-Image-Turbo`` (8 steps,
    guidance_scale=1.0) per explicit user directive. The guidance_scale
    change is NOT cosmetic -- distilled/turbo diffusion checkpoints
    typically bake classifier-free guidance into the distillation target,
    so reusing the base model's guidance_scale=4.0 on the Turbo checkpoint
    would very likely oversaturate/degrade output, not just run slower.
    Confirmed via the model's own HuggingFace card, not assumed.

    2026-07-21 follow-up: the prompt enhancer (``use_pe``) is explicitly
    disabled (:data:`USE_PROMPT_ENHANCER`). ``ErnieImagePipeline`` defaults
    ``use_pe=True``, which -- confirmed by reading the installed
    ``diffusers`` source, not assumed -- runs a SEPARATE auxiliary LLM
    generation call to rewrite the caller's prompt before image synthesis.
    Left at its default, that would silently rewrite paperbanana's own
    carefully-engineered prompt (venue styling, structured diagram
    description) through an opaque second LLM step. Disabled per explicit
    user directive.

**Detailed explanation for engineers:**
    paperbanana's ``OpenAIImageGen`` provider
    (``external/paperbanana/paperbanana/providers/image_gen/openai_imagen.py``)
    calls ``AsyncOpenAI(api_key=..., base_url=...).images.generate(model=,
    prompt=, n=1, size="WxH", quality=?)`` and reads
    ``result.data[0].b64_json``. This module implements exactly that request
    /response contract — nothing more — against a locally-loaded Diffusers
    pipeline. It does NOT reimplement the OpenAI API in general; only the
    single endpoint paperbanana's image-gen role actually calls.

    The model is loaded lazily (on first request, not at import time) and
    cached as a process-wide singleton, mirroring the lazy-import discipline
    used throughout ``carnot.samplers.backend`` so importing this module
    never drags in ``torch``/``diffusers``/CUDA before they're needed.

    GPU selection follows the project's existing llama-server launch
    convention: pin ``CUDA_VISIBLE_DEVICES`` in the *process* environment
    before any CUDA library is imported (this module never imports
    ``torch`` at module scope), so ``--gpu N`` behaves identically to the
    ``CUDA_VISIBLE_DEVICES=N llama-server ...`` pattern used elsewhere in
    this project (e.g. ``experiment_5768_direct_incontext_prediction_ab.py``).

Spec: REQ-PUBLISH-042, SCENARIO-PUBLISH-042
"""

from __future__ import annotations

import argparse
import base64
import functools
import io
import logging
import os
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

HF_REPO_ID = "baidu/ERNIE-Image-Turbo"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8990
# Turbo-specific settings per https://huggingface.co/baidu/ERNIE-Image-Turbo --
# NOT the base model's guidance_scale=4.0 / 50 steps (see module docstring).
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_INFERENCE_STEPS = 8
# `ErnieImagePipeline.__call__` defaults `use_pe=True` (confirmed by inspecting
# the installed diffusers==0.39.0 source directly, not assumed). When True it
# runs a SEPARATE auxiliary LLM's .generate() call (its own tokenizer + chat
# template, non-greedy sampling by default) to rewrite the caller's prompt
# before generation -- see `_enhance_prompt_with_pe` in
# diffusers/pipelines/ernie_image/pipeline_ernie_image.py. For this project
# that is actively undesirable: paperbanana's own Visualizer/Stylist agents
# already carefully engineer the prompt (venue-specific styling, structured
# diagram description); letting ERNIE silently rewrite it again would
# undermine that prompt engineering and inject non-deterministic LLM output
# into what should be a controlled generate step. Explicitly disabled (never
# left to the pipeline's own default) per 2026-07-21 user directive.
USE_PROMPT_ENHANCER = False
NATIVE_SIZE = (1024, 1024)

# paperbanana's OpenAIImageGen only ever requests one of these three
# strings (see `_RATIO_TO_SIZE` / `_size_string` in the vendored source).
# Anything else falls back to NATIVE_SIZE rather than guessing.
_KNOWN_SIZES = {"1024x1024", "1536x1024", "1024x1536"}


def ernie_image_cached() -> bool:
    """Precondition check: is ``baidu/ERNIE-Image-Turbo`` already in the local HF cache?

    **Why this exists (Pre-Launch Preconditions Discipline):** an 8B-parameter
    diffusion model is a multi-gigabyte download. This project's convention
    for every compute-bound resource is to check availability BEFORE
    attempting to use it, and fail with an honest, named ``blocked_*``
    condition rather than silently triggering a slow first-use download or
    crashing deep inside pipeline construction. Never raises; returns
    ``False`` on any lookup failure (matches the GGUF-cache-check pattern
    used throughout ``python/carnot/experiment_*.py``).
    """
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
    except Exception:
        return False
    return any(repo.repo_id == HF_REPO_ID for repo in cache.repos)


def parse_openai_size(size: str) -> tuple[int, int]:
    """Parse an OpenAI Images API ``size`` string (``"1024x1536"``) into ``(w, h)``.

    Falls back to :data:`NATIVE_SIZE` for anything unparseable or outside
    the three sizes paperbanana's OpenAI image provider ever sends, rather
    than raising — a malformed size should degrade to a working image, not
    a 500.
    """
    if size not in _KNOWN_SIZES:
        return NATIVE_SIZE
    w_str, h_str = size.lower().split("x")
    return int(w_str), int(h_str)


class ErniePipelineSingleton:
    """Lazy, process-wide singleton around the Diffusers ERNIE-Image-Turbo pipeline.

    **Detailed explanation for engineers:**
        Loading an 8B-parameter DiT pipeline takes real wall-clock time and
        VRAM; it must happen exactly once per server process, on first
        request, not per-request and not at import time. This class holds
        that single instance. Tests never exercise this class directly —
        they monkeypatch ``get`` on the class object, since instantiating
        the real pipeline requires a cached model and a GPU.

        Uses ``diffusers.ErnieImagePipeline`` explicitly (not the generic
        ``DiffusionPipeline`` auto-class-resolver) to match the exact usage
        documented on https://huggingface.co/baidu/ERNIE-Image-Turbo.
        Requires diffusers>=0.39 (confirmed available at 0.39.0 in this
        project's `.venv` as of 2026-07-21; older diffusers versions may not
        ship this pipeline class).
    """

    _pipeline: Any = None

    @classmethod
    def get(cls, gpu: Optional[int] = None) -> Any:
        if cls._pipeline is None:
            if not ernie_image_cached():
                raise RuntimeError(
                    f"{HF_REPO_ID} not found in the local HuggingFace cache. "
                    f"Fetch it first: huggingface-cli download {HF_REPO_ID}"
                )
            import torch
            from diffusers import ErnieImagePipeline

            device = f"cuda:{gpu}" if gpu is not None else "cuda"
            pipe = ErnieImagePipeline.from_pretrained(HF_REPO_ID, torch_dtype=torch.bfloat16)
            cls._pipeline = pipe.to(device)
        return cls._pipeline

    @classmethod
    def reset(cls) -> None:
        """Test-only hook: drop the cached singleton."""
        cls._pipeline = None


_request_model_lock = threading.Lock()


@functools.lru_cache(maxsize=1)
def _request_model() -> Any:
    """Build the OpenAI Images API request schema, once, as a real module global.

    FastAPI resolves a Pydantic model's fields via `get_type_hints`, which
    (for a function using `from __future__ import annotations`, as this
    module does) resolves string annotations against the function's
    `__globals__` -- for a nested function that is always the *module's*
    global namespace, never a merely-local variable of an enclosing
    function. A model class assigned only to a local inside `build_app()`
    is therefore invisible to `get_type_hints`: FastAPI silently falls back
    to treating the parameter as a query param instead of a request body
    (a real bug caught by tests/python/test_ernie_image_server.py --
    `loc: ['query', 'req']` instead of the request body; `lru_cache` alone,
    without also writing a real module global, reproduces the exact same
    bug -- verified by hand while writing this comment). The lock makes the
    "build once, publish to module globals" step below race-free; the
    `lru_cache` on top of it means every caller after the first gets the
    cheap cached return without re-acquiring the lock.
    """
    with _request_model_lock:
        existing = globals().get("ImageGenerationRequest")
        if existing is not None:
            return existing

        from pydantic import BaseModel

        class ImageGenerationRequest(BaseModel):
            model: str = "ernie-image"
            prompt: str
            n: int = 1
            size: str = "1024x1024"
            # ERNIE-Image has no OpenAI-style quality tiering; accepted for
            # request-shape compatibility with paperbanana's OpenAIImageGen
            # (which always sends it when the caller passed --image-quality),
            # but silently unused -- see generate() below.
            quality: Optional[str] = None
            response_format: Optional[str] = "b64_json"

        globals()["ImageGenerationRequest"] = ImageGenerationRequest
        return ImageGenerationRequest


def build_app(gpu: Optional[int] = None) -> Any:
    """Construct the FastAPI app. Imports FastAPI lazily (optional ``imagegen`` extra)."""
    from fastapi import FastAPI, HTTPException

    _request_model()  # ensures ImageGenerationRequest is a real module global before use

    app = FastAPI(title="carnot-ernie-image-shim")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "model_cached": ernie_image_cached()}

    @app.post("/v1/images/generations")
    def generate(req: ImageGenerationRequest) -> dict[str, Any]:
        if req.n != 1:
            # paperbanana's OpenAIImageGen always requests n=1 (see
            # openai_imagen.py). A multi-image request would mean an
            # unexpected caller; refuse rather than silently truncate.
            raise HTTPException(status_code=400, detail="this shim only supports n=1")
        width, height = parse_openai_size(req.size)
        pipe = ErniePipelineSingleton.get(gpu=gpu)
        t0 = time.time()
        result = pipe(
            prompt=req.prompt,
            width=width,
            height=height,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            num_inference_steps=DEFAULT_INFERENCE_STEPS,
            use_pe=USE_PROMPT_ENHANCER,
        )
        image = result.images[0]
        duration_s = time.time() - t0
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        logger.info(
            "ernie-image generated %dx%d in %.1fs (prompt len=%d)",
            width,
            height,
            duration_s,
            len(req.prompt),
        )
        return {"data": [{"b64_json": b64}]}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA device index to pin via CUDA_VISIBLE_DEVICES before any CUDA "
        "library is imported (mirrors the CUDA_VISIBLE_DEVICES=N llama-server "
        "launch convention used elsewhere in this project).",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        # Must happen before torch/diffusers import (deferred to first
        # request inside ErniePipelineSingleton.get) or it has no effect.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not ernie_image_cached():
        logger.warning(
            "%s not yet cached locally -- server will start but the first "
            "/v1/images/generations call will fail until you run: "
            "huggingface-cli download %s",
            HF_REPO_ID,
            HF_REPO_ID,
        )

    import uvicorn

    # CUDA_VISIBLE_DEVICES already restricts visibility to one device, so
    # the pipeline always addresses it as cuda:0 from here on.
    app = build_app(gpu=0 if args.gpu is not None else None)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
