#!/usr/bin/env python3
"""Exp 785: SOTA GGUF 2-Round Code Repair v2 — HumanEval (25 problems, zombie-kill-first).

**Researcher summary (RETRO-SOTA-GGUF-TIMEOUT fix):**
    Exp 769 timed out after 120 min for two reasons:
    (1) 15 GiB of zombie VRAM occupied GPU 0 before model load, causing OOM.
    (2) 50 problems × 3-4 min/problem = 150-200 min exceeds 120-min budget.

    This experiment (785) fixes both root causes:
    (1) Calls kill_gpu_zombies(gpu_index=0) from gpu_zombie_killer BEFORE model load.
        This is the aggressive SIGKILL approach from Exp 780, not the softer heuristic
        in ExperimentTemplate.kill_gpu_zombies(). It guarantees VRAM is freed.
    (2) Limits benchmark to 25 HumanEval problems (first 25, same as Exp 769 subset
        for comparability). 25 × 3.6 min/problem = 90 min — within the 90-min cap.

    Model selection is VRAM-based after zombie kill:
    - free_vram_mb >= 20000: Qwen3.6-35B-A3B-GGUF Q4_K_M (~20 GiB, preferred)
    - free_vram_mb < 20000:  Qwen3.5-7B-Instruct-GGUF Q4_K_M (~4 GiB, fallback)
    Both produce valid Python code (unlike Qwen3.5-0.8B which gave pass@1=0.0 in Exp 759).

    signed_improvement = pass_at_1_round2 - pass_at_1_round1
    The arXiv 2604.10508 paper found +4.9 to +17.1pp across 7 models >= 7B.

**honest_verdict logic:**
    - "sota_code_repair_positive"   if signed_improvement > 0, inference_mode="live_gpu"
    - "sota_code_repair_zero"       if signed_improvement = 0, inference_mode="live_gpu"
    - "sota_code_repair_negative"   if signed_improvement < 0 (unexpected — repair hurt)
    - "blocked_model_load_failed"   if llama-cpp load fails after zombie kill
    - "blocked_no_live_gpu"         if CARNOT_FORCE_LIVE not set

**Gate:** CARNOT_FORCE_LIVE=1 required (GPU needed for live GGUF inference).

Spec: REQ-REPAIR-024, REQ-REPAIR-025, SCENARIO-REPAIR-044, SCENARIO-REPAIR-045,
      REQ-REPAIR-022, REQ-REPAIR-023, REQ-CODE-031, REQ-CODE-032
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Force CPU JAX — EBM ops only; LLM inference via llama.cpp on CUDA.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.gpu_zombie_killer import kill_gpu_zombies  # noqa: E402
from carnot.pipeline.two_round_repair import TwoRoundCodeRepairPipeline, TwoRoundResult  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# Reuse the validated HumanEval subset from Exp 744; take first 25 for comparability with Exp 769.
from scripts.experiment_744_iterative_2round_repair import _HUMANEVAL_SUBSET  # noqa: E402

EXP_ID = 785
TITLE = "SOTA GGUF 2-Round Code Repair v2 — HumanEval 25 problems (zombie-kill-first)"
DELIVERABLE = "results/experiment_785_sota_gguf_code_repair_v2.json"

# Only use first 25 problems — half of Exp 769's 50 to fit within the 90-min budget.
N_PROBLEMS = 25

# VRAM threshold: Qwen3.6-35B Q4_K_M requires ~20 GiB. Below this, use the 7B fallback.
_LARGE_MODEL_VRAM_THRESHOLD_MB = 20_000

MODEL_LARGE = "unsloth/Qwen3.6-35B-A3B-GGUF"
MODEL_FALLBACK = "unsloth/Qwen3.5-7B-Instruct-GGUF"

# Per-problem timeout in seconds (3 min). The overall watchdog is 90 min.
_PER_PROBLEM_TIMEOUT_S = 180


# ---------------------------------------------------------------------------
# Pure helper functions (unit-testable; no GPU imports)
# ---------------------------------------------------------------------------


def build_repair_prompt_785(
    original_problem: str,
    failed_code: str,
    error_message: str,
) -> str:
    """Build a repair prompt that includes the error message from round 1.

    Why this is minimal (3 fields): The primary repair signal is the full traceback.
    The original problem context tells the model what was intended. The failing code
    shows exactly what went wrong. A plain instruction outperforms elaborate meta-prompts
    per arXiv 2604.10508 — adding more structure does not help.

    Args:
        original_problem: The full HumanEval problem prompt (docstring + signature).
        failed_code: The Python code that failed execution in round 1.
        error_message: Traceback or error string from the failed execution.

    Returns:
        Formatted repair prompt string ready for the LLM caller.

    Spec: REQ-REPAIR-024, SCENARIO-REPAIR-044
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


def select_model_by_vram(free_vram_mb: float) -> str:
    """Return the model ID to use based on available free VRAM after zombie kill.

    Why 20 GiB threshold: Qwen3.6-35B-A3B Q4_K_M occupies ~20 GiB on a single RTX 3090
    (24 GiB total). Below this threshold, loading would likely OOM. The 7B fallback at
    ~4 GiB is safe even with only a few GiB free.

    Args:
        free_vram_mb: Free GPU VRAM in MiB after kill_gpu_zombies() has run.

    Returns:
        HuggingFace model ID string for the selected model.

    Spec: REQ-REPAIR-025, SCENARIO-REPAIR-045
    """
    if free_vram_mb >= _LARGE_MODEL_VRAM_THRESHOLD_MB:
        return MODEL_LARGE
    return MODEL_FALLBACK


def compute_repair_metrics_785(
    results: list[TwoRoundResult],
) -> dict[str, Any]:
    """Compute pass@1 and signed_improvement from a list of TwoRoundResults.

    Definitions (REQ-REPAIR-023):
    - pass_at_1_round1: fraction passing in initial (round 0) generation.
    - pass_at_1_round2: cumulative fraction passing after at most one repair.
    - signed_improvement: pass_at_1_round2 - pass_at_1_round1.
    - n_repaired: count(NOT round0_pass AND round1_pass) — problems repaired.

    Args:
        results: List of TwoRoundResult, one per problem.

    Returns:
        Dict with keys: pass_at_1_round1, pass_at_1_round2, signed_improvement, n_repaired.

    Spec: REQ-REPAIR-023, REQ-REPAIR-024
    """
    n = len(results)
    if n == 0:
        return {
            "pass_at_1_round1": 0.0,
            "pass_at_1_round2": 0.0,
            "signed_improvement": 0.0,
            "n_repaired": 0,
        }
    r1 = sum(1 for r in results if r.round0_pass) / n
    r2 = sum(1 for r in results if r.round0_pass or r.round1_pass) / n
    si = round(r2 - r1, 4)
    n_repaired = sum(1 for r in results if not r.round0_pass and r.round1_pass)
    return {
        "pass_at_1_round1": round(r1, 4),
        "pass_at_1_round2": round(r2, 4),
        "signed_improvement": si,
        "n_repaired": n_repaired,
    }


def classify_verdict_785(signed_improvement: float, inference_mode: str) -> str:
    """Map signed_improvement + inference_mode to an honest_verdict label.

    Verdict ladder (REQ-REPAIR-024, REQ-REPAIR-022):
    - "blocked_no_live_gpu"        — CARNOT_FORCE_LIVE not set; no inference ran.
    - "blocked_model_load_failed"  — llama-cpp load raised ImportError or OOM.
    - "sota_code_repair_positive"  — signed_improvement > 0, live GPU confirmed.
    - "sota_code_repair_zero"      — signed_improvement = 0, live GPU ran but no gain.
    - "sota_code_repair_negative"  — signed_improvement < 0 (unexpected — repair hurt).

    Args:
        signed_improvement: pass_at_1_round2 - pass_at_1_round1.
        inference_mode: "live_gpu", "blocked", or "blocked_model_load_failed".

    Returns:
        One of the five verdict strings above.

    Spec: REQ-REPAIR-024, SCENARIO-REPAIR-044
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


def _build_gguf_caller_785(model_id: str, gpu_id: int = 0):
    """Build an LLM callable for the selected GGUF model via llama-cpp-python.

    Resolution order:
    1. resolve_cached_gguf() — finds local HF cache without network (fastest path).
    2. Llama.from_pretrained() — downloads from HuggingFace (needs network).

    The caller is valid for both Qwen3.6-35B-A3B and Qwen3.5-7B-Instruct GGUFs.
    Context length is 4096 — sufficient for HumanEval problems + repair prompts.

    Returns a callable(prompt: str) -> str, or raises on failure.

    Spec: REQ-REPAIR-024, REQ-REPAIR-025
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
    """Run Exp 785: SOTA GGUF 2-round code repair v2 on 25 HumanEval problems.

    Key changes vs Exp 769:
    - kill_gpu_zombies(gpu_index=0) called BEFORE model load (REQ-REPAIR-024).
    - 25 problems instead of 50, 90-min watchdog instead of 120-min.
    - VRAM-based model selection: 35B if >= 20 GiB free, else 7B (REQ-REPAIR-025).
    """
    apply_env_autofix()

    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()
    tmpl.check_exclusion_manifest()

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=90, result_path=DELIVERABLE):

        # Guard: CARNOT_FORCE_LIVE=1 required — REQ-REPAIR-024
        force_live_now = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
        if not force_live_now:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "blocked_no_live_gpu",
                    "blocked_reason": (
                        "CARNOT_FORCE_LIVE=1 not set — live GPU required for GGUF inference"
                    ),
                    "inference_mode": "blocked",
                    "model_used": "none",
                    "free_vram_mb_after_kill": 0.0,
                    "n_problems": 0,
                    "pass_at_1_round1": 0.0,
                    "pass_at_1_round2": 0.0,
                    "signed_improvement": 0.0,
                    "n_repaired": 0,
                },
                status="blocked",
            )
            out = Path(_REPO) / DELIVERABLE
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # REQ-REPAIR-024: kill GPU zombies BEFORE model load.
        # Use the aggressive SIGKILL approach from gpu_zombie_killer (Exp 780),
        # not the softer utilization-heuristic in ExperimentTemplate.kill_gpu_zombies().
        zombie_result = kill_gpu_zombies(gpu_index=0)

        # Read free VRAM after zombie kill to decide which model to load.
        free_vram_mb_after_kill: float = 0.0
        try:
            import subprocess  # noqa: PLC0415

            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if smi.returncode == 0:
                free_vram_mb_after_kill = float(smi.stdout.strip().splitlines()[0].strip())
        except Exception:
            # Non-fatal — proceed; select_model_by_vram will default to fallback.
            free_vram_mb_after_kill = 0.0

        # REQ-REPAIR-025: VRAM-based model selection.
        model_id = select_model_by_vram(free_vram_mb_after_kill)
        model_used = model_id.split("/")[-1]

        # Load the selected GGUF model.
        try:
            llm_caller = _build_gguf_caller_785(model_id, gpu_id=0)
        except (ImportError, OSError, Exception) as exc:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "blocked_model_load_failed",
                    "blocked_reason": f"llama-cpp load failed: {exc}",
                    "inference_mode": "blocked_model_load_failed",
                    "model_used": model_used,
                    "free_vram_mb_after_kill": free_vram_mb_after_kill,
                    "n_problems": 0,
                    "pass_at_1_round1": 0.0,
                    "pass_at_1_round2": 0.0,
                    "signed_improvement": 0.0,
                    "n_repaired": 0,
                    "zombie_kill_verdict": zombie_result.honest_verdict,
                },
                status="blocked",
            )
            out = Path(_REPO) / DELIVERABLE
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # Smoke test: verify model can produce output before 25-problem run.
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
                    "model_used": model_used,
                    "free_vram_mb_after_kill": free_vram_mb_after_kill,
                    "n_problems": 0,
                    "pass_at_1_round1": 0.0,
                    "pass_at_1_round2": 0.0,
                    "signed_improvement": 0.0,
                    "n_repaired": 0,
                    "zombie_kill_verdict": zombie_result.honest_verdict,
                },
                status="blocked",
            )
            out = Path(_REPO) / DELIVERABLE
            out.parent.mkdir(parents=True, exist_ok=True)
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

        # Take first N_PROBLEMS from the validated HumanEval subset (same ordering as Exp 769).
        problems_subset = _HUMANEVAL_SUBSET[:N_PROBLEMS]
        problems_to_run = problems_subset[start_idx:]
        batch_log: list[dict] = []

        for i, problem_dict in enumerate(problems_to_run):
            t0 = time.perf_counter()
            try:
                # Per-problem 3-minute sub-timer (REQ-REPAIR-024).
                result = tmpl.run_with_timeout(
                    lambda p=problem_dict: pipeline.run(
                        problem=p["prompt"],
                        test_cases=p["test_cases"],
                        llm_caller=llm_caller,
                    ),
                    timeout_s=_PER_PROBLEM_TIMEOUT_S,
                )
                if result is None:
                    # Timed out — treat as double-fail.
                    result = TwoRoundResult(
                        round0_pass=False, round1_pass=False, round2_pass=False,
                        round0_code="", round1_code="", round2_code="",
                        error_types=["timeout"],
                    )
            except Exception:
                result = TwoRoundResult(
                    round0_pass=False, round1_pass=False, round2_pass=False,
                    round0_code="", round1_code="", round2_code="",
                    error_types=["other"],
                )
            elapsed = round(time.perf_counter() - t0, 3)
            batch_log.append({"problem_idx": start_idx + i, "time_s": elapsed})
            all_results.append(result)

            # Checkpoint every 5 problems to survive conductor restarts.
            if len(all_results) % 5 == 0:
                tmpl.checkpoint_save(
                    {"results": [vars(r) for r in all_results]},
                    step=len(all_results),
                )

        metrics = compute_repair_metrics_785(all_results)
        verdict = classify_verdict_785(metrics["signed_improvement"], "live_gpu")

        artifact = tmpl.build_result(
            {
                "honest_verdict": verdict,
                "model_used": model_used,
                "free_vram_mb_after_kill": free_vram_mb_after_kill,
                "zombie_kill_verdict": zombie_result.honest_verdict,
                "n_problems": len(all_results),
                "pass_at_1_round1": metrics["pass_at_1_round1"],
                "pass_at_1_round2": metrics["pass_at_1_round2"],
                "signed_improvement": metrics["signed_improvement"],
                "n_repaired": metrics["n_repaired"],
                "inference_mode": "live_gpu",
                "batch_log": batch_log,
                "arxiv_ref": "2604.10508",
            },
            status="success",
            decision_class="repair",
        )
        out = Path(_REPO) / DELIVERABLE
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
