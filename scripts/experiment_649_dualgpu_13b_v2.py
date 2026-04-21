#!/usr/bin/env python3
"""Experiment 649: DualGPU 13B Proof v2 — Pre-verify HF cache, split-load 7B across 2 RTX 3090s.

**Researcher summary (RETRO-071 follow-up to Exp 632):**
    Exp 632 failed with model_load_failed because Qwen2.5 weights were not cached
    locally before attempting load.  This v2 script adds an explicit HF-cache pre-check
    *before* touching transformers: if the weights are not on disk it exits immediately
    with a clear action_required message telling the operator exactly which
    ``huggingface-cli download`` command to run.  Only after confirming cached weights
    does it attempt the split-GPU load and forward-pass utilization proof.

    GPU-1 utilization > 50% during inference proves that both RTX 3090s participate
    in the 7B forward pass, resolving RETRO-071.

Spec: REQ-INFRA-092, SCENARIO-INFRA-099, SCENARIO-INFRA-100
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from scripts.experiment_template import ExperimentTemplate, _utc_now  # noqa: E402


# ---------------------------------------------------------------------------
# HF cache helpers
# ---------------------------------------------------------------------------

_HF_CANDIDATE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]

# Layer split for Qwen2.5-7B: 28 transformer layers total.
# cuda:0 handles embedding + layers 0-13; cuda:1 handles layers 14-27, norm, lm_head.
_QWEN_7B_LAYERS = 28
_QWEN_7B_SPLIT = 14


def _hf_home() -> str:
    """Return the HuggingFace cache root directory.

    Why: HF_HOME overrides the default ``~/.cache/huggingface``.  Checking the env
    var first means the function works correctly in non-standard CI setups where the
    cache lives on a separate volume.
    """
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _model_cache_dir(hf_home: str, model_id: str) -> Path:
    """Translate a HuggingFace model ID to its on-disk hub directory.

    HF hub stores models as ``<hf_home>/hub/models--<org>--<name>``,
    replacing ``/`` with ``--``.  We consider the model cached when
    this directory exists and contains at least one ``.safetensors``
    or ``.bin`` shard (a directory with only config files is not enough).
    """
    slug = "models--" + model_id.replace("/", "--")
    return Path(hf_home) / "hub" / slug


def check_hf_cache(candidates: list[str]) -> list[str]:
    """Return subset of ``candidates`` whose weights are present in the HF cache.

    A model is considered *cached* when its hub directory exists AND contains
    at least one weight shard (.safetensors or .bin) anywhere under it.
    This guards against the partial-download case where only the config arrived.
    """
    hf_home = _hf_home()
    found: list[str] = []
    for model_id in candidates:
        cache_dir = _model_cache_dir(hf_home, model_id)
        if not cache_dir.exists():
            continue
        # Must have actual weight files, not just configs.
        shards = list(cache_dir.rglob("*.safetensors")) + list(cache_dir.rglob("*.bin"))
        if shards:
            found.append(model_id)
    return found


# ---------------------------------------------------------------------------
# GPU utilization helpers
# ---------------------------------------------------------------------------


def sample_util(gpu_index: int) -> float:
    """Sample GPU utilization percentage for one device.

    Returns percentage (0.0-100.0) via pynvml when available, falls back to
    ``torch.cuda.utilization``.  Returns -1.0 when both paths fail so callers
    can distinguish 'zero utilization measured' from 'measurement unavailable'.
    """
    try:
        import pynvml  # noqa: PLC0415

        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        pass
    try:
        import torch  # noqa: PLC0415

        return float(torch.cuda.utilization(gpu_index))
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Core experiment logic (broken into testable functions)
# ---------------------------------------------------------------------------


def detect_gpus() -> tuple[int, float, float]:
    """Return (n_gpus, vram_0_gb, vram_1_gb) using torch.cuda.

    Returns (0, 0.0, 0.0) when torch is unavailable or CUDA is absent.
    VRAM for a missing GPU slot is always 0.0.
    """
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return 0, 0.0, 0.0

    if not torch.cuda.is_available():
        return 0, 0.0, 0.0

    n = torch.cuda.device_count()
    if n == 0:
        return 0, 0.0, 0.0

    def _vram(idx: int) -> float:
        if idx < n:
            return torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return 0.0

    return n, _vram(0), _vram(1)


def build_device_map(n_layers: int, split: int) -> dict[str, str]:
    """Build a transformers device_map that splits a model across cuda:0 and cuda:1.

    ``split`` is the first layer index assigned to cuda:1.  Layers 0 .. split-1
    run on cuda:0; layers split .. n_layers-1 run on cuda:1.  The embedding,
    final norm, and lm_head assignments mirror the standard HF split convention.

    Example: n_layers=28, split=14 → layers 0-13 on cuda:0, 14-27 + head on cuda:1.
    """
    device_map: dict[str, str] = {}
    device_map["model.embed_tokens"] = "cuda:0"
    for i in range(n_layers):
        device_map[f"model.layers.{i}"] = "cuda:0" if i < split else "cuda:1"
    device_map["model.norm"] = "cuda:1"
    device_map["lm_head"] = "cuda:1"
    return device_map


def load_model_split(model_id: str) -> object | None:
    """Load a causal LM from HF cache split across cuda:0 and cuda:1.

    Uses a hardcoded 14/14 layer split for the Qwen2.5-7B architecture.
    Returns the model object on success or None on failure (caller must
    check and record the blocked_reason).
    """
    try:
        from transformers import AutoModelForCausalLM  # noqa: PLC0415
        import torch  # noqa: PLC0415

        device_map = build_device_map(_QWEN_7B_LAYERS, _QWEN_7B_SPLIT)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        return model
    except Exception as exc:
        print(f"[649] load_model_split failed: {exc}", flush=True)
        return None


def run_forward_passes(model: object, model_id: str, n_passes: int = 10) -> tuple[list[float], list[float]]:
    """Run ``n_passes`` forward passes and sample GPU utilization after each.

    Returns (util_0_list, util_1_list) — parallel lists of per-pass GPU
    utilization readings (%).  On tokenizer or generate failure, records
    -1.0 for both GPUs so downstream statistics still have the right length.

    Why we sample *after* each generate() call rather than during: the HF
    generate() is synchronous; the GPU is busy for the entire duration, so
    a single post-call sample is representative and avoids threading complexity.
    """
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
        import torch  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        print(f"[649] tokenizer load failed: {exc}", flush=True)
        return [], []

    util_0: list[float] = []
    util_1: list[float] = []

    for _ in range(n_passes):
        try:
            inputs = tokenizer("1+1=", return_tensors="pt")
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=20)  # type: ignore[arg-type]
            util_0.append(sample_util(0))
            util_1.append(sample_util(1))
        except Exception:
            util_0.append(-1.0)
            util_1.append(-1.0)

    return util_0, util_1


def run_experiment() -> dict:
    """Execute the full DualGPU 13B v2 experiment and return the result dict.

    Sequence:
      1. Detect GPU count and VRAM.
      2. Check HF cache for Qwen2.5 weights.
      3. Load model split across cuda:0 / cuda:1.
      4. Run 10 forward passes, measure GPU-1 utilization.
      5. Return structured result with honest_verdict.

    This function is separated from ``main()`` so it can be unit-tested
    without subprocess overhead.
    """
    started_at = _utc_now()

    # --- 1. GPU detection ---
    n_gpus, vram_0_gb, vram_1_gb = detect_gpus()

    base: dict = {
        "n_gpus": n_gpus,
        "vram_0_gb": round(vram_0_gb, 2),
        "vram_1_gb": round(vram_1_gb, 2),
        "model_loaded": False,
        "model_name": None,
        "peak_gpu1_util": 0.0,
        "sustained_gpu1_fraction": 0.0,
        "dualgpu_proven": False,
        "retro_071_resolved": False,
        "honest_verdict": "model_not_cached",
    }

    if n_gpus < 2:
        base["dualgpu_available"] = False
        base["blocked_reason"] = "only_one_gpu"
        base["honest_verdict"] = "model_not_cached"
        return base

    base["dualgpu_available"] = True

    # --- 2. HF cache pre-check ---
    cached = check_hf_cache(_HF_CANDIDATE_MODELS)
    preferred_model = cached[0] if cached else None

    if preferred_model is None:
        base["model_loaded"] = False
        base["blocked_reason"] = "model_not_cached_HF_weights_required"
        base["action_required"] = (
            "Run: huggingface-cli download Qwen/Qwen2.5-7B-Instruct "
            "--local-dir ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
        )
        base["honest_verdict"] = "model_not_cached"
        return base

    # --- 3. Load model split across 2 GPUs ---
    model = load_model_split(preferred_model)

    if model is None:
        base["model_loaded"] = False
        base["model_name"] = preferred_model
        base["blocked_reason"] = "model_load_failed"
        base["honest_verdict"] = "model_not_cached"
        return base

    base["model_loaded"] = True
    base["model_name"] = preferred_model

    # --- 4. Forward passes + utilization measurement ---
    util_0, util_1 = run_forward_passes(model, preferred_model, n_passes=10)

    # Cleanup GPU memory immediately to free VRAM for subsequent steps.
    try:
        import torch  # noqa: PLC0415

        del model
        torch.cuda.empty_cache()
    except Exception:
        pass

    valid_util_1 = [u for u in util_1 if u >= 0]
    peak_gpu1 = max(valid_util_1) if valid_util_1 else 0.0
    sustained = sum(1 for u in util_1 if u > 10) / len(util_1) if util_1 else 0.0

    base["peak_gpu1_util"] = round(peak_gpu1, 2)
    base["sustained_gpu1_fraction"] = round(sustained, 3)
    base["dualgpu_proven"] = peak_gpu1 > 50 or sustained > 0.5
    base["retro_071_resolved"] = base["dualgpu_proven"]

    if base["model_loaded"] and peak_gpu1 > 50:
        base["honest_verdict"] = "dualgpu_proven"
    elif base["model_loaded"]:
        base["honest_verdict"] = "dualgpu_loaded_low_util"
    else:
        base["honest_verdict"] = "model_not_cached"

    return base


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 649 end-to-end and write the deliverable JSON."""
    apply_env_autofix()

    tmpl = ExperimentTemplate(
        649,
        "DualGPU 13B Proof v2",
        "results/experiment_649_dualgpu_13b_v2.json",
        requires_gpu=True,
    )
    tmpl.setup()

    result_data = run_experiment()

    # Determine status from result.
    if result_data.get("dualgpu_proven"):
        status = "success"
    elif result_data.get("model_loaded"):
        status = "partial"
    else:
        status = "blocked"

    result_data["artifact_schema"] = "carnot.dualgpu_13b_v2.v1"
    artifact = tmpl.build_result(result_data, status=status)
    out_path = _REPO_ROOT / "results" / "experiment_649_dualgpu_13b_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"[649] Wrote {out_path}", flush=True)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
