"""Robust local GGUF generation harness for live verifier experiments.

This module is intentionally narrow: every caller gets the same local-cache
resolver, the same llama.cpp loading path, and the same generate-smoke gate.
That matters because Exp 3904 did not fail at cache discovery or import time;
it failed when the loaded GGUF was actually asked to generate.  The harness
therefore treats a model as ready only after it emits text through a real
one-token generation call.

Spec refs: REQ-INFER-SOTA-023, SCENARIO-INFER-SOTA-023-001,
SCENARIO-INFER-SOTA-023-002.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import resolve_cached_gguf

DEFAULT_PREFER_ORDER: tuple[str, ...] = (
    "gemma-4-26B-A4B-it",
    "Qwen3.6-35B-A3B",
    "gemma-4-31B-it",
)


def _hf_id_for_name(model_name: str) -> str:
    return f"unsloth/{model_name}-GGUF"


def _resolve_candidate_path(model_name: str) -> str | None:
    return resolve_cached_gguf(_hf_id_for_name(model_name))


def _resolve_candidate_paths(model_name: str) -> list[str]:
    if model_name == "gemma-4-26B-A4B-it":
        quant_order = ("IQ2_M", "IQ2_XXS", "Q3_K_M", "Q4_K_M")
    else:
        quant_order = ("Q4_K_M",)
    paths: list[str] = []
    for quant in quant_order:
        path = resolve_cached_gguf(_hf_id_for_name(model_name), preferred_quant=quant)
        if path and path not in paths and Path(path).is_file() and Path(path).stat().st_size > 0:
            paths.append(path)
    fallback = _resolve_candidate_path(model_name)
    if fallback and fallback not in paths and Path(fallback).is_file() and Path(fallback).stat().st_size > 0:
        paths.append(fallback)
    return paths


def _extract_llama_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if "text" in first:
                    return str(first["text"])
                message = first.get("message")
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"])
    return str(result)


def _completion_token_count(result: Any) -> int:
    if isinstance(result, dict):
        usage = result.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
            return int(usage["completion_tokens"])
    text = _extract_llama_text(result).strip()
    return 0 if not text else max(1, len(text.split()))


def generate(generator: Any, prompt: str, max_tokens: int) -> str:
    result = generator(str(prompt), max_tokens=int(max_tokens), temperature=0.0)
    return _extract_llama_text(result)


def load_gguf_generator(
    prefer_order: list[str] | tuple[str, ...] | None = None,
    n_ctx: int = 1024,
    max_n_gpu_layers: int = -1,
) -> tuple[Any, dict[str, object]]:
    from llama_cpp import Llama

    failures: list[str] = []
    candidates = tuple(prefer_order) if prefer_order is not None else DEFAULT_PREFER_ORDER
    requested_gpu_layers = int(max_n_gpu_layers)
    offload_levels = (0,) if requested_gpu_layers == 0 else (requested_gpu_layers, 20, 0)

    for fallback_index, model_name in enumerate(candidates):
        gguf_paths = _resolve_candidate_paths(model_name)
        if not gguf_paths:
            failures.append(f"{model_name}: no cached GGUF resolved")
            continue

        for gguf_path in gguf_paths:
            for n_gpu_layers in offload_levels:
                load_started = time.time()
                try:
                    generator = Llama(
                        model_path=str(gguf_path),
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=int(n_ctx),
                        n_batch=min(64, int(n_ctx)),
                        offload_kqv=n_gpu_layers != 0,
                        seed=3915,
                        verbose=False,
                    )
                except Exception as exc:
                    failures.append(
                        f"{model_name} n_gpu_layers={n_gpu_layers} path={gguf_path}: load failed: {exc!r}"
                    )
                    continue

                load_s = time.time() - load_started
                smoke_started = time.time()
                try:
                    smoke_result = generator("2+2=", max_tokens=1, temperature=0.0)
                    smoke_text = _extract_llama_text(smoke_result).strip()
                    smoke_tokens = _completion_token_count(smoke_result)
                    if smoke_tokens <= 0 or not smoke_text:
                        raise RuntimeError(f"smoke returned no output tokens: {smoke_result!r}")
                except Exception as exc:
                    failures.append(
                        f"{model_name} n_gpu_layers={n_gpu_layers} path={gguf_path}: smoke failed: {exc!r}"
                    )
                    continue

                return generator, {
                    "model_used": model_name,
                    "gguf_path": str(gguf_path),
                    "n_gpu_layers_used": n_gpu_layers,
                    "load_s": load_s,
                    "smoke_s": time.time() - smoke_started,
                    "smoke_tokens": smoke_tokens,
                    "fallback_index": fallback_index,
                }

    joined = "\n".join(failures) if failures else "no candidates supplied"
    raise RuntimeError(f"blocked_all_gguf_inference_failed:\n{joined}")
