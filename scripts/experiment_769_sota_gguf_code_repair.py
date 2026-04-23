#!/usr/bin/env python3
"""Exp 769: SOTA GGUF 2-Round Code Repair — HumanEval signed improvement.

**Researcher summary (arXiv 2604.10508):**
    Exp 759 confirmed that Qwen3.5-0.8B is too small for code repair (pass@1=0.0).
    This experiment repeats the 2-round repair benchmark using the mandated SOTA
    GGUF model: unsloth/Qwen3.6-35B-A3B-GGUF (Q4_K_M, ~20GB VRAM, ~3B active
    params per token).  The arXiv paper found +4.9 to +17.1pp improvement across
    7 models all >= 7B.  This is the first credible headline code repair result
    for Carnot.

    signed_improvement = pass_at_1_round2 - pass_at_1_round1
    Where pass_at_1_round1 = initial generation pass rate (50 problems),
          pass_at_1_round2 = cumulative pass rate after at most one repair round.

**Why Qwen3.6-35B-A3B (MoE)?**
    ~3B active params per token gives high throughput on a single RTX 3090.
    Q4_K_M fits in ~20GB VRAM, leaving ~4GB headroom.  Highest capability-per-
    computation for code repair on a single consumer GPU.

**Why NOT Qwen3.5-0.8B?**
    Exp 759 produced pass@1=0.0 — too small to generate valid Python.  REQ-REPAIR-022
    mandates >= 7B models for headline code repair claims.

**honest_verdict logic:**
    - "sota_code_repair_positive"   if signed_improvement > 0, inference_mode="live_gpu"
    - "sota_code_repair_zero"       if signed_improvement = 0, inference_mode="live_gpu"
    - "sota_code_repair_negative"   if signed_improvement < 0 (unexpected — repair hurt)
    - "blocked_model_load_failed"   if llama-cpp-python load fails (VRAM/install issue)
    - "blocked_no_live_gpu"         if CARNOT_FORCE_LIVE not set

**Gate:** CARNOT_FORCE_LIVE=1 required (GPU needed for live GGUF inference).

Spec: REQ-REPAIR-022, REQ-REPAIR-023, SCENARIO-REPAIR-042, SCENARIO-REPAIR-043,
      REQ-CODE-031, REQ-CODE-032
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Force CPU JAX — EBM ops only; LLM inference via llama.cpp on CUDA.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.two_round_repair import TwoRoundCodeRepairPipeline, TwoRoundResult  # noqa: E402

# Reuse the validated 50-problem HumanEval subset from Exp 744.
from scripts.experiment_744_iterative_2round_repair import _HUMANEVAL_SUBSET  # noqa: E402

EXP_ID = 769
TITLE = "SOTA GGUF 2-Round Code Repair — HumanEval (Qwen3.6-35B-A3B-GGUF)"
DELIVERABLE = "results/experiment_769_sota_gguf_code_repair.json"
N_PROBLEMS = 50
MODEL_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
FORCE_LIVE = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"


# ---------------------------------------------------------------------------
# Pure helper functions (unit-testable; no GPU imports)
# ---------------------------------------------------------------------------


def build_repair_prompt_769(
    original_problem: str,
    failed_code: str,
    error_message: str,
) -> str:
    """Build a repair prompt that includes the error message from round 1.

    The repair prompt must differ from the generation prompt by adding:
    1. The failing code — the model needs to see what it wrote.
    2. The error/traceback — the primary repair signal (REQ-REPAIR-022).

    Keeping this minimal (3 fields instead of 6 in Exp 759) reduces prompt
    tokens while preserving the essential error signal that drives repair.

    Args:
        original_problem: The full HumanEval problem prompt (docstring + signature).
        failed_code: The Python code that failed execution.
        error_message: Traceback or error string from the failed execution.

    Returns:
        Formatted repair prompt string ready for the LLM caller.

    Spec: REQ-REPAIR-022, SCENARIO-REPAIR-042
    """
    parts = [
        "You are an expert Python programmer.  The code below has a bug.",
        "",
        "## Original Problem",
        original_problem.strip(),
        "",
        "## Failing Code",
        "```python",
        failed_code.strip(),
        "```",
        "",
        "## Error",
        error_message.strip() if error_message.strip() else "(no traceback)",
        "",
        "Fix the bug.  Return ONLY the corrected function definition.",
    ]
    return "\n".join(parts)


def compute_repair_metrics(
    results: list[TwoRoundResult],
) -> dict[str, Any]:
    """Compute pass@1 and signed_improvement from a list of TwoRoundResults.

    Definitions (REQ-REPAIR-023):
    - pass_at_1_round1: fraction passing in initial (round 0) generation.
    - pass_at_1_round2: cumulative fraction passing after at most one repair.
    - signed_improvement: pass_at_1_round2 - pass_at_1_round1.
    - n_repaired: count(NOT round0_pass AND round1_pass) — problems repaired.
    - n_round2_attempted: count(NOT round0_pass) — problems where repair was tried.

    Args:
        results: List of TwoRoundResult, one per problem.

    Returns:
        Dict with keys: pass_at_1_round1, pass_at_1_round2, signed_improvement,
        n_repaired, n_round2_attempted.

    Spec: REQ-REPAIR-023, SCENARIO-REPAIR-043
    """
    n = len(results)
    if n == 0:
        return {
            "pass_at_1_round1": 0.0,
            "pass_at_1_round2": 0.0,
            "signed_improvement": 0.0,
            "n_repaired": 0,
            "n_round2_attempted": 0,
        }
    r1 = sum(1 for r in results if r.round0_pass) / n
    r2 = sum(1 for r in results if r.round0_pass or r.round1_pass) / n
    si = round(r2 - r1, 4)
    n_repaired = sum(1 for r in results if not r.round0_pass and r.round1_pass)
    n_round2_attempted = sum(1 for r in results if not r.round0_pass)
    return {
        "pass_at_1_round1": round(r1, 4),
        "pass_at_1_round2": round(r2, 4),
        "signed_improvement": si,
        "n_repaired": n_repaired,
        "n_round2_attempted": n_round2_attempted,
    }


def classify_verdict_769(signed_improvement: float, inference_mode: str) -> str:
    """Map signed_improvement + inference_mode to an honest_verdict label.

    Verdict ladder (REQ-REPAIR-022):
    - "blocked_no_live_gpu"        — CARNOT_FORCE_LIVE not set; no inference ran.
    - "blocked_model_load_failed"  — llama-cpp load raised ImportError or FileNotFoundError.
    - "sota_code_repair_positive"  — signed_improvement > 0, live GPU confirmed.
    - "sota_code_repair_zero"      — signed_improvement = 0, live GPU ran but no gain.
    - "sota_code_repair_negative"  — signed_improvement < 0 (unexpected — repair hurt).

    Args:
        signed_improvement: pass_at_1_round2 - pass_at_1_round1.
        inference_mode: "live_gpu", "blocked", or "blocked_model_load_failed".

    Returns:
        One of the five verdict strings above.

    Spec: REQ-REPAIR-022, SCENARIO-REPAIR-042
    """
    if inference_mode == "blocked":
        return "blocked_no_live_gpu"
    if inference_mode == "blocked_model_load_failed":
        return "blocked_model_load_failed"
    if signed_improvement > 0:
        return "sota_code_repair_positive"
    if signed_improvement < 0:
        return "sota_code_repair_negative"
    return "sota_code_repair_zero"


# ---------------------------------------------------------------------------
# LLM caller (live mode — llama.cpp GGUF)
# ---------------------------------------------------------------------------


def _build_gguf_caller(model_id: str = MODEL_ID, gpu_id: int = 0):
    """Build an LLM callable for Qwen3.6-35B-A3B-GGUF via llama-cpp-python.

    Resolution order:
    1. resolve_cached_gguf() — finds local cache without network (fastest).
    2. Llama.from_pretrained() — downloads from HuggingFace (needs network).

    Returns a callable(prompt: str) -> str, or raises on failure.

    Why prefer local cache: Exp 769 runs on an isolated host where HF downloads
    may be blocked.  The resolve_cached_gguf helper checks both
    ~/.cache/huggingface/hub and <project>/models/.

    Spec: REQ-REPAIR-022, SCENARIO-REPAIR-042
    """
    from llama_cpp import Llama  # noqa: PLC0415 — optional dep

    from carnot.inference.sota_models import resolve_cached_gguf  # noqa: PLC0415

    local_path = resolve_cached_gguf(model_id, preferred_quant="Q4_K_M")
    if local_path:
        llm = Llama(
            model_path=local_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
    else:
        llm = Llama.from_pretrained(
            model_id,
            filename="*Q4_K_M*",
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )

    def _call(prompt: str) -> str:
        output = llm(prompt, max_tokens=512, temperature=0.2, echo=False)
        return output["choices"][0]["text"]

    return _call


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 769: SOTA GGUF 2-round code repair benchmark on 50 HumanEval problems."""
    apply_env_autofix()

    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()
    tmpl.check_exclusion_manifest()

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=120, result_path=DELIVERABLE):

        # Guard: CARNOT_FORCE_LIVE=1 required — REQ-REPAIR-022
        force_live_now = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
        if not force_live_now:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "blocked_no_live_gpu",
                    "blocked_reason": (
                        "CARNOT_FORCE_LIVE=1 not set — live GPU required for GGUF inference"
                    ),
                    "inference_mode": "blocked",
                    "model_id": MODEL_ID,
                    "n_problems": 0,
                    "pass_at_1_round1": 0.0,
                    "pass_at_1_round2": 0.0,
                    "signed_improvement": 0.0,
                    "n_repaired": 0,
                    "n_round2_attempted": 0,
                },
                status="blocked",
            )
            out = Path(_REPO) / DELIVERABLE
            out.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # VRAM check before loading 20GB model
        try:
            import subprocess  # noqa: PLC0415
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if smi.returncode == 0:
                free_mb = int(smi.stdout.strip().splitlines()[gpu_id := 0].strip())
                if free_mb < 20000:
                    import logging  # noqa: PLC0415
                    logging.getLogger(__name__).warning(
                        "GPU 0 free VRAM %d MB < 20000 MB — Q4_K_M may OOM; proceeding anyway",
                        free_mb,
                    )
        except Exception:
            pass  # non-fatal; proceed and let llama.cpp fail if VRAM is insufficient

        # Load GGUF model via llama-cpp-python
        try:
            llm_caller = _build_gguf_caller(MODEL_ID, gpu_id=0)
        except (ImportError, OSError, Exception) as exc:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "blocked_model_load_failed",
                    "blocked_reason": f"llama-cpp load failed: {exc}",
                    "inference_mode": "blocked_model_load_failed",
                    "model_id": MODEL_ID,
                    "n_problems": 0,
                    "pass_at_1_round1": 0.0,
                    "pass_at_1_round2": 0.0,
                    "signed_improvement": 0.0,
                    "n_repaired": 0,
                    "n_round2_attempted": 0,
                },
                status="blocked",
            )
            out = Path(_REPO) / DELIVERABLE
            out.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # Smoke test: basic code completion before committing to 50-problem run
        try:
            smoke = llm_caller("def hello():")
            if not smoke or len(smoke.strip()) == 0:
                raise RuntimeError("smoke test returned empty response")
        except Exception as exc:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "blocked_model_load_failed",
                    "blocked_reason": f"smoke test failed: {exc}",
                    "inference_mode": "blocked_model_load_failed",
                    "model_id": MODEL_ID,
                    "n_problems": 0,
                    "pass_at_1_round1": 0.0,
                    "pass_at_1_round2": 0.0,
                    "signed_improvement": 0.0,
                    "n_repaired": 0,
                    "n_round2_attempted": 0,
                },
                status="blocked",
            )
            out = Path(_REPO) / DELIVERABLE
            out.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        pipeline = TwoRoundCodeRepairPipeline()
        all_results: list[TwoRoundResult] = []

        # Resume from checkpoint if available.
        checkpoint = tmpl.checkpoint
        start_idx = 0
        if checkpoint and "results" in checkpoint:
            raw = checkpoint["results"].get("results", [])
            for r in raw:
                all_results.append(TwoRoundResult(**r))
            start_idx = len(all_results)

        problems_to_run = _HUMANEVAL_SUBSET[start_idx:]
        batch_log: list[dict] = []

        import time as _time  # noqa: PLC0415
        for i, problem_dict in enumerate(problems_to_run):
            t0 = _time.perf_counter()
            try:
                result = pipeline.run(
                    problem=problem_dict["prompt"],
                    test_cases=problem_dict["test_cases"],
                    llm_caller=llm_caller,
                )
            except Exception:
                result = TwoRoundResult(
                    round0_pass=False, round1_pass=False, round2_pass=False,
                    round0_code="", round1_code="", round2_code="",
                    error_types=["other"],
                )
            elapsed = round(_time.perf_counter() - t0, 3)
            batch_log.append({"problem_idx": start_idx + i, "time_s": elapsed})
            all_results.append(result)

            if len(all_results) % 10 == 0:
                tmpl.checkpoint_save(
                    {"results": [vars(r) for r in all_results]},
                    step=len(all_results),
                )

        metrics = compute_repair_metrics(all_results)
        verdict = classify_verdict_769(metrics["signed_improvement"], "live_gpu")

        artifact = tmpl.build_result(
            {
                "honest_verdict": verdict,
                "model_id": MODEL_ID,
                "n_problems": len(all_results),
                "pass_at_1_round1": metrics["pass_at_1_round1"],
                "pass_at_1_round2": metrics["pass_at_1_round2"],
                "signed_improvement": metrics["signed_improvement"],
                "n_repaired": metrics["n_repaired"],
                "n_round2_attempted": metrics["n_round2_attempted"],
                "inference_mode": "live_gpu",
                "batch_log": batch_log,
                "arxiv_ref": "2604.10508",
            },
            status="success",
            decision_class="repair",
        )
        out = Path(_REPO) / DELIVERABLE
        out.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
