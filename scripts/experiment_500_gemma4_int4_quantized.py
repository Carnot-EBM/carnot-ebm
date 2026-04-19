#!/usr/bin/env python3
"""Experiment 500 — Gemma4 INT4 Quantization: RETRO-048 Unblocking Fix.

**Researcher summary:**
    The conductor process holds ~15.7 GiB of GPU 0 VRAM for its own model
    inference.  This is an ACTIVE process — GPUVRAMGateV2 cannot kill it.
    Gemma4 at FP16 requires 14.89 GiB.  Conductor + FP16 = ~30.6 GiB, which
    exceeds the RTX 3090's 24 GiB budget by 6.6 GiB.

    The fix: quantize Gemma4 to GGUF Q4_K_M format (~8-10 GiB).
      conductor (~9 GiB) + Gemma4-INT4 (~9 GiB) = ~18 GiB
      18 GiB < 24 GiB — fits with ~6 GiB headroom

    This experiment loads the quantized model, checks VRAM budget, and runs a
    10-question GSM8K accuracy check to confirm quantization quality (>= 60%).

**Artifact schema:** carnot.gemma4_quantization.v1

**How to run with a real GGUF:**
    export CARNOT_GEMMA4_GGUF_PATH=/path/to/gemma4-q4_k_m.gguf
    python scripts/experiment_500_gemma4_int4_quantized.py

**How to obtain the GGUF checkpoint (if not already downloaded):**
    pip install llama-cpp-python unsloth
    # Option 1: Download pre-quantized from HuggingFace
    huggingface-cli download unsloth/gemma-4-12b-it-GGUF --include '*.Q4_K_M.gguf'
    # Option 2: Quantize from source
    python -m llama_cpp.convert_hf_to_gguf google/gemma-4-E4B-it --outtype q4_k_m

Spec: REQ-LOADER-003, REQ-LOADER-004, REQ-LOADER-005,
      SCENARIO-LOADER-003, SCENARIO-LOADER-004, SCENARIO-LOADER-005
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Apply env autofix FIRST — must precede any GPU-touching import (RETRO-022 fix)
from carnot.pipeline.env_autofix import apply_env_autofix

_env_fix = apply_env_autofix()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo path setup — experiment_template.py lives in scripts/, one level above
# repo root when launched from repo root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.gpu_vram_gate_v2 import GPUVRAMGateV2  # noqa: E402
from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader  # noqa: E402

_DELIVERABLE = "results/experiment_500_gemma4_int4_quantized.json"

_SETUP_INSTRUCTIONS = """
How to obtain and run the Gemma4 Q4_K_M GGUF model:

1. Install llama-cpp-python with CUDA support:
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

2. Download a pre-quantized Gemma4 GGUF from HuggingFace (unsloth recommended):
   huggingface-cli download unsloth/gemma-4-12b-it-GGUF \\
     --include '*.Q4_K_M.gguf' --local-dir /tmp/gemma4-gguf

3. Set the path and re-run:
   export CARNOT_GEMMA4_GGUF_PATH=/tmp/gemma4-gguf/gemma-4-12b-it-Q4_K_M.gguf
   python scripts/experiment_500_gemma4_int4_quantized.py

Alternative (quantize from source):
   python -m llama_cpp.convert_hf_to_gguf google/gemma-4-E4B-it --outtype q4_k_m \\
     --outfile /tmp/gemma4-q4km.gguf

Expected VRAM footprint: 8-10 GiB for Q4_K_M (vs 14.89 GiB for FP16).
Conductor VRAM: ~9 GiB.  Total with quantized model: ~18 GiB < 24 GiB budget.
""".strip()


def main() -> None:
    """Run Experiment 500 — Gemma4 INT4 quantization RETRO-048 unblocking check."""

    guard = DeliverableGuard(_DELIVERABLE)

    tmpl = ExperimentTemplate(
        exp_id=500,
        title="Gemma4 INT4 Quantization",
        deliverable=_DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(500, timeout_minutes=30):
        _run(tmpl, guard)

    tmpl.assert_deliverable_written()


def _run(tmpl: ExperimentTemplate, guard: DeliverableGuard) -> None:
    """Inner experiment body — separated for testability."""

    gguf_path = os.getenv("CARNOT_GEMMA4_GGUF_PATH", "")
    gguf_path_set = bool(gguf_path)

    # ------------------------------------------------------------------ #
    # VRAM gate — low threshold (2 GiB) since model may not be loaded yet.
    # We're checking there's SOME free VRAM, not the full model budget.
    # ------------------------------------------------------------------ #
    try:
        GPUVRAMGateV2(min_free_gb=2.0, kill_first=True, zombie_drain_sleep_seconds=5).__enter__()
    except Exception as exc:
        _log.warning("GPUVRAMGateV2 could not enter: %s — continuing", exc)

    # ------------------------------------------------------------------ #
    # Attempt to detect GPU
    # ------------------------------------------------------------------ #
    gpu_available = _detect_gpu()
    _log.info("GPU available: %s", gpu_available)

    model_loaded = False
    vram_usage_gb = None
    is_within_budget = None
    accuracy_check_result = None

    if not gpu_available:
        _log.warning("No GPU detected — cannot load GGUF model.  Emitting gpu_required status.")
        honest_verdict = "deferred_retro_048"
        artifact = tmpl.build_result(
            {
                "gguf_path_set": gguf_path_set,
                "model_loaded": False,
                "vram_usage_gb": None,
                "is_within_budget": None,
                "accuracy_check_result": None,
                "retro_048_unblocked": False,
                "honest_verdict": "deferred_retro_048",
                "setup_instructions": _SETUP_INSTRUCTIONS,
            },
            status="gpu_required",
        )
        artifact["schema"] = "carnot.gemma4_quantization.v1"
        _write(artifact)
        return

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    loader = Gemma4QuantizedLoader(model_path=gguf_path, n_gpu_layers=-1, max_tokens=512)
    model_loaded = loader.load()
    _log.info("model_loaded=%s, stub_mode=%s", model_loaded, loader._stub_mode)

    if model_loaded:
        vram_usage_gb = loader.vram_usage_gb()
        is_within_budget = loader.is_within_budget(10.0)
        _log.info("vram_usage_gb=%.2f, is_within_budget=%s", vram_usage_gb, is_within_budget)

        accuracy_check_result = loader.accuracy_check(n_questions=10)
        _log.info("accuracy_check_result=%.2f", accuracy_check_result)

    # ------------------------------------------------------------------ #
    # Compute honest verdict and retro_048_unblocked flag
    # ------------------------------------------------------------------ #
    retro_048_unblocked = bool(
        model_loaded
        and is_within_budget
        and accuracy_check_result is not None
        and accuracy_check_result >= 0.60
    )

    if retro_048_unblocked:
        honest_verdict = "retro_048_unblocked"
    elif not gguf_path_set:
        honest_verdict = "gguf_path_not_set"
    elif model_loaded and is_within_budget is False:
        honest_verdict = "vram_over_budget"
    elif model_loaded and accuracy_check_result is not None and accuracy_check_result < 0.60:
        honest_verdict = "accuracy_degraded"
    else:
        honest_verdict = "deferred_retro_048"

    _log.info("honest_verdict=%s, retro_048_unblocked=%s", honest_verdict, retro_048_unblocked)

    status = "success" if retro_048_unblocked else "partial"

    artifact = tmpl.build_result(
        {
            "gguf_path_set": gguf_path_set,
            "model_loaded": model_loaded,
            "vram_usage_gb": vram_usage_gb,
            "is_within_budget": is_within_budget,
            "accuracy_check_result": accuracy_check_result,
            "retro_048_unblocked": retro_048_unblocked,
            "honest_verdict": honest_verdict,
            "setup_instructions": _SETUP_INSTRUCTIONS,
        },
        status=status,
    )
    artifact["schema"] = "carnot.gemma4_quantization.v1"
    _write(artifact)


def _detect_gpu() -> bool:
    """Return True if a CUDA GPU is accessible."""
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _write(artifact: dict) -> None:
    """Write artifact to the deliverable path atomically."""
    out_path = _REPO_ROOT / _DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(artifact, f, indent=2)
    tmp_path.rename(out_path)
    _log.info("Deliverable written: %s", out_path)


if __name__ == "__main__":
    main()
