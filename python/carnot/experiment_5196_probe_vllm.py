"""Standalone vLLM-native DiffusionGemma load probe for Exp 5196.

WHY THIS IS A SEPARATE SCRIPT (not part of the main experiment module):
vLLM lives in its own virtual-env (``~/.cache/vllm-venv``) with a torch build
(2.11+cu130) distinct from the Carnot ``.venv``. The two cannot share an
interpreter, so the vLLM-native attempt must run as an isolated subprocess
driven by the vllm-venv python. The parent experiment module
(``experiment_5196_diffusiongemma_vllm_native_retry_v476``) shells out to this
script and parses the JSON events it prints.

It emits one JSON object per line (newline-delimited) and flushes after each so
that, even if the process is killed by an OOM or a hard timeout mid-attempt, the
parent can still read every attempt that completed. The final line is always a
``{"event": "summary", ...}`` object when the script exits cleanly.

Each attempt records the real peak VRAM (``torch.cuda.max_memory_allocated``)
per visible device and whether a genuine forward pass (``llm.generate``)
produced text -- a load that returns an object but cannot generate does NOT
count as ``forward_pass_confirmed`` (the exp5182 precedent: ``model_loaded`` can
co-exist with a forward-pass failure).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

MODEL_ID = "google/diffusiongemma-26B-A4B-it"


def _emit(obj: dict) -> None:
    """Print one JSON event and flush, so a killed process still leaves a trail."""
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def _vram_per_gpu() -> dict:
    """Peak allocated VRAM (GiB) per visible CUDA device since last reset."""
    try:
        import torch

        out = {}
        for i in range(torch.cuda.device_count()):
            out[f"gpu{i}"] = round(
                torch.cuda.max_memory_allocated(i) / (1024**3), 3
            )
        return out
    except Exception:  # pragma: no cover - defensive; torch always present in vllm-venv
        return {}


def _reset_peak() -> None:
    try:
        import torch

        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
    except Exception:  # pragma: no cover - defensive
        pass


def _attempt(label: str, visible_devices: str, kwargs: dict) -> dict:
    """Run one vLLM load+generate attempt in the current process.

    ``visible_devices`` sets CUDA_VISIBLE_DEVICES for the attempt. Because
    vLLM initialises CUDA lazily on first LLM() construction and does not fully
    release it afterwards, the PARENT is expected to run each attempt in a fresh
    subprocess; here we still set it defensively.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    started = time.time()
    result = {
        "event": "attempt",
        "mitigation": label,
        "visible_devices": visible_devices,
        "kwargs": {k: v for k, v in kwargs.items() if k != "model"},
        "outcome": "unknown",
        "forward_pass_confirmed": False,
        "sample_output": None,
        "peak_vram_gib_per_gpu": {},
        "error_if_any": None,
        "duration_s": 0.0,
    }
    llm = None
    try:
        _reset_peak()
        from vllm import LLM, SamplingParams

        llm = LLM(model=MODEL_ID, **kwargs)
        # A load that returns is not enough -- confirm a REAL forward pass.
        out = llm.generate(
            ["def add(a, b):\n    return"],
            SamplingParams(max_tokens=8, temperature=0.0),
        )
        text = out[0].outputs[0].text if out and out[0].outputs else ""
        result["outcome"] = "forward_pass_ok"
        result["forward_pass_confirmed"] = True
        result["sample_output"] = text[:120]
    except Exception as exc:  # noqa: BLE001 - we want the exact failure string
        result["outcome"] = "load_failed"
        result["error_if_any"] = f"{type(exc).__name__}: {exc}"[:1200]
        result["traceback_tail"] = "".join(
            traceback.format_exc().splitlines(keepends=True)[-6:]
        )[:1200]
    finally:
        result["peak_vram_gib_per_gpu"] = _vram_per_gpu()
        result["duration_s"] = round(time.time() - started, 3)
        try:
            del llm
        except Exception:  # pragma: no cover - defensive
            pass
    _emit(result)
    return result


def main() -> int:
    """Run the single attempt named by argv (parent isolates each in a subprocess).

    Usage: ``probe_vllm.py <attempt_name>`` where attempt_name is one of
    ``bnb4bit_tp1_gpu0`` / ``bnb4bit_tp2_both`` / ``registry_only``.
    """
    attempt = sys.argv[1] if len(sys.argv) > 1 else "registry_only"

    if attempt == "registry_only":
        # Cheap, decisive check: does this vLLM build have a native runner?
        try:
            import vllm
            from vllm.model_executor.models.registry import ModelRegistry

            archs = set(ModelRegistry.get_supported_archs())
            _emit(
                {
                    "event": "registry",
                    "vllm_version": vllm.__version__,
                    "has_diffusiongemma_native": "DiffusionGemmaForBlockDiffusion"
                    in archs,
                }
            )
        except Exception as exc:  # noqa: BLE001
            _emit({"event": "registry", "error": f"{type(exc).__name__}: {exc}"})
        return 0

    common = dict(
        enforce_eager=True,
        max_model_len=1024,
        trust_remote_code=True,
    )
    # The vLLM DiffusionGemma recipe (recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it)
    # documents the ONE OOM-avoidance lever the block-diffusion runner needs:
    # ``max_num_seqs=4`` -- the diffusion state buffers pre-allocate
    # ``max_seqs x canvas_length x vocab_size`` logit tensors, so the default
    # (large) max_num_seqs OOMs regardless of weight quantization. We combine it
    # with 4-bit bnb weights (to fit 2x24 GiB) + the recipe entropy-bound sampler.
    recipe_common = dict(
        quantization="bitsandbytes",
        max_num_seqs=4,
        enforce_eager=True,
        max_model_len=4096,
        trust_remote_code=True,
        hf_overrides={
            "diffusion_sampler": "entropy_bound",
            "diffusion_entropy_bound": 0.1,
        },
    )
    if attempt == "bnb4bit_tp1_gpu0":
        _attempt(
            "vllm_native_bnb4bit_tp1_gpu0",
            "0",
            dict(
                quantization="bitsandbytes",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.92,
                **common,
            ),
        )
    elif attempt == "bnb4bit_tp2_both":
        _attempt(
            "vllm_native_bnb4bit_tp2_both_gpus",
            "0,1",
            dict(
                quantization="bitsandbytes",
                tensor_parallel_size=2,
                gpu_memory_utilization=0.90,
                **common,
            ),
        )
    elif attempt == "bnb4bit_tp2_recipe":
        _attempt(
            "vllm_native_bnb4bit_tp2_recipe_maxseqs4",
            "0,1",
            dict(
                tensor_parallel_size=2,
                gpu_memory_utilization=0.85,
                **recipe_common,
            ),
        )
    elif attempt == "bnb4bit_tp1_recipe":
        _attempt(
            "vllm_native_bnb4bit_tp1_gpu0_recipe_maxseqs4",
            "0",
            dict(
                tensor_parallel_size=1,
                gpu_memory_utilization=0.90,
                **recipe_common,
            ),
        )
    elif attempt == "fp8_tp2_recipe":
        # bnb is unsupported for this MoE (no get_expert_mapping); fp8 weight-only
        # is the only quantization that could fit ~25.8B params across 2x24 GiB
        # (fp8 ~= 1 byte/param => ~12.9 GiB/GPU under tp2). Ampere lacks native fp8
        # compute, so vLLM would use the fp8-Marlin weight-only path if available.
        _attempt(
            "vllm_native_fp8_tp2_recipe_maxseqs4",
            "0,1",
            dict(
                quantization="fp8",
                tensor_parallel_size=2,
                gpu_memory_utilization=0.90,
                max_num_seqs=4,
                enforce_eager=True,
                max_model_len=4096,
                trust_remote_code=True,
            ),
        )
    else:
        _emit({"event": "error", "detail": f"unknown attempt {attempt!r}"})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
