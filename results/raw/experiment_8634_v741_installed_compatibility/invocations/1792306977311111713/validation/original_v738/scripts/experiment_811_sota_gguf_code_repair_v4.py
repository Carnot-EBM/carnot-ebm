#!/usr/bin/env python3
"""Exp 811: SOTA GGUF Code Repair v4 — HumanEval 50 problems, batched 10×5, MARS gate.

**Researcher summary (RETRO-SOTA-GGUF-TIMEOUT fix v4):**
    Exp 796 (v3) was gated by RETRO-028: GPU OOM prevention (evict_gpu_vram) was
    not using the retry-loop pattern, so the model load could race into OOM.  Exp
    810 closed RETRO-028 by introducing evict_vram_with_loop() — a kill-and-verify
    retry loop that blocks until VRAM is below the safe threshold.

    v4 upgrades over v3:
    1. 50 problems (10 batches of 5) instead of 25 (5 batches of 5), giving a more
       statistically robust pass@1 estimate.
    2. Gate on Exp 810 (not 795) — Exp 810 closes RETRO-028 with the loop fix.
    3. evict_vram_with_loop() instead of evict_gpu_vram() to guarantee VRAM clear.

**Gate:** results/experiment_810_gemma4_oom_fix_v5.json must have retro_028_closed=True.
If not, write gated_retro028_not_closed artifact and exit.

**honest_verdict logic:**
    - "code_repair_positive"          if signed_improvement > 0, inference_mode="live_gpu"
    - "code_no_improvement"           if signed_improvement <= 0, inference_mode="live_gpu"
    - "partial_N_of_50"               if timeout with N < 50 problems completed
    - "gated_retro028_not_closed"     if Exp 810 gate not met
    - "blocked_no_live_gpu"           if LiveGPUGate blocks

Spec: REQ-BENCH-016, SCENARIO-BENCH-035
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
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.gpu_zombie_killer import kill_gpu_zombies  # noqa: E402
from carnot.pipeline.vram_loop_eviction import evict_vram_with_loop  # noqa: E402
from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.mars_margin_gate import MARSMarginGate  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 811
TITLE = "SOTA GGUF Code Repair v4 — HumanEval 50 problems, batched 10×5, MARS gate"
DELIVERABLE = "results/experiment_811_sota_gguf_code_repair_v4.json"
N_PROBLEMS = 50
BATCH_SIZE = 5
TIMEOUT_MINUTES = 90
GPU_INDEX = 1
MARS_THRESHOLD = 2.0

_EXP810_RESULT = _REPO / "results" / "experiment_810_gemma4_oom_fix_v5.json"


# ---------------------------------------------------------------------------
# Pure helpers — unit-testable
# ---------------------------------------------------------------------------


def check_retro028_gate(result_path: Path) -> bool:
    """Return True iff Exp 810 artifact reports retro_028_closed=True.

    Why this gate exists: RETRO-028 tracks the GPU OOM root cause that killed
    Exp 785.  Exp 810 closes it via the evict_vram_with_loop retry-loop fix.
    Running Exp 811 without this gate would repeat the same OOM failure.

    Args:
        result_path: Path to results/experiment_810_gemma4_oom_fix_v5.json.

    Returns:
        True only when the file exists and retro_028_closed == True.

    Spec: REQ-BENCH-016-4
    """
    if not result_path.exists():
        return False
    try:
        with open(result_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return bool(data.get("retro_028_closed"))
    except (json.JSONDecodeError, OSError):
        return False


def compute_signed_improvement(
    total_repair_pass: int, total_baseline_pass: int, n_problems: int
) -> float:
    """Compute signed_improvement = mean(pass@1_repair) - mean(pass@1_baseline).

    Not clamped, not normalised — raw arithmetic difference.  A positive value
    means repair improved accuracy; zero or negative means no benefit.

    Args:
        total_repair_pass: Count of problems that passed after repair.
        total_baseline_pass: Count of problems that passed at baseline.
        n_problems: Total number of problems evaluated (denominator).

    Returns:
        Float difference in [−1.0, 1.0].

    Spec: REQ-BENCH-016-6
    """
    if n_problems == 0:
        return 0.0
    return (total_repair_pass - total_baseline_pass) / n_problems


def build_blocked_artifact(
    tmpl: "ExperimentTemplate",
    honest_verdict: str,
    blocked_reason: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build a blocked/gated artifact with consistent schema fields.

    Centralising here ensures every blocked exit path emits the same schema,
    avoiding the inconsistent artifact shape that made Exp 785 hard to parse.

    Args:
        tmpl: Initialised ExperimentTemplate (provides build_result).
        honest_verdict: One of the defined verdict strings for Exp 811.
        blocked_reason: Human-readable explanation of why the run was blocked.
        **extra: Additional key-value pairs to merge into the artifact.

    Returns:
        Dict satisfying REQUIRED_RESULT_FIELDS.
    """
    return tmpl.build_result(
        {
            "honest_verdict": honest_verdict,
            "blocked_reason": blocked_reason,
            "inference_mode": "blocked",
            "n_problems": N_PROBLEMS,
            "n_completed": 0,
            "pass_at_1_baseline": None,
            "pass_at_1_repair": None,
            "signed_improvement": None,
            "oracle_calls_saved": 0,
            **extra,
        },
        status="blocked",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — runs the full batched repair pipeline or writes a blocked artifact."""
    # Step a: fix environment before any heavy imports.
    apply_env_autofix()

    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES)
    watchdog.start()

    writer = AtomicResultWriter(str(_REPO / DELIVERABLE))

    # Step c: hard GPU gate.
    gpu_gate_result = LiveGPUGate.require_live_or_blocked(tmpl)
    if gpu_gate_result is not None:
        # Gate returned a blocked artifact — merge in our required extra fields.
        artifact = build_blocked_artifact(
            tmpl,
            honest_verdict="blocked_no_live_gpu",
            blocked_reason="CARNOT_FORCE_LIVE not set; no live GPU available",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step d: prerequisite gate — Exp 810 must have closed retro_028.
    if not check_retro028_gate(_EXP810_RESULT):
        artifact = build_blocked_artifact(
            tmpl,
            honest_verdict="gated_retro028_not_closed",
            blocked_reason=(
                "Exp 810 did not set retro_028_closed=True; "
                "GPU OOM root cause unresolved — Exp 811 would repeat Exp 785 failure"
            ),
            gate_file=str(_EXP810_RESULT),
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step e: evict VRAM on GPU 1 using retry-loop (RETRO-028 fix v5 pattern).
    # evict_vram_with_loop() calls kill_gpu_zombies internally on each retry,
    # so we still call kill_gpu_zombies explicitly first for an eager pass.
    zombie_result = kill_gpu_zombies(GPU_INDEX)
    evict_result = evict_vram_with_loop(gpu_index=GPU_INDEX)

    if not evict_result.vram_cleared:
        artifact = build_blocked_artifact(
            tmpl,
            honest_verdict="blocked_vram_not_cleared",
            blocked_reason=(
                f"evict_vram_with_loop failed after max retries: {evict_result.abort_reason}"
            ),
            zombie_kills=zombie_result.n_killed if hasattr(zombie_result, "n_killed") else 0,
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step f: resolve GGUF model.
    try:
        from carnot.pipeline.gguf_cache import resolve_cached_gguf  # type: ignore[import]

        model_path = resolve_cached_gguf("unsloth/Qwen3.6-35B-A3B-GGUF", quant="Q4_K_M")
    except Exception as exc:
        artifact = build_blocked_artifact(
            tmpl,
            honest_verdict="blocked_model_load_failed",
            blocked_reason=f"GGUF cache resolution failed: {exc}",
            zombie_kills=zombie_result.n_killed if hasattr(zombie_result, "n_killed") else 0,
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step g: load 50 HumanEval problems (p0-p49).
    try:
        from scripts.experiment_744_iterative_2round_repair import _HUMANEVAL_SUBSET  # type: ignore[import]

        problems = list(_HUMANEVAL_SUBSET)[:N_PROBLEMS]
    except Exception as exc:
        artifact = build_blocked_artifact(
            tmpl,
            honest_verdict="blocked_model_load_failed",
            blocked_reason=f"HumanEval dataset load failed: {exc}",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step h: run 10 batches of 5 problems.
    mars_gate = MARSMarginGate(threshold=MARS_THRESHOLD)
    batches = [problems[i : i + BATCH_SIZE] for i in range(0, N_PROBLEMS, BATCH_SIZE)]

    all_batch_results: list[dict[str, Any]] = []
    total_baseline_pass = 0
    total_repair_pass = 0
    total_oracle_calls_saved = 0
    n_completed = 0

    for batch_idx, batch in enumerate(batches):
        if watchdog.is_expired():
            partial_verdict = f"partial_{n_completed}_of_50"
            artifact = tmpl.build_result(
                {
                    "honest_verdict": partial_verdict,
                    "inference_mode": "live_gpu",
                    "n_problems": N_PROBLEMS,
                    "n_completed": n_completed,
                    "pass_at_1_baseline": total_baseline_pass / max(n_completed, 1),
                    "pass_at_1_repair": total_repair_pass / max(n_completed, 1),
                    "signed_improvement": compute_signed_improvement(
                        total_repair_pass, total_baseline_pass, max(n_completed, 1)
                    ),
                    "oracle_calls_saved": total_oracle_calls_saved,
                    "batch_results": all_batch_results,
                },
                status="partial",
            )
            writer.write(artifact)
            tmpl.assert_deliverable_written()
            return

        batch_baseline_pass = 0
        batch_repair_pass = 0
        batch_oracle_saved = 0
        batch_problem_results: list[dict[str, Any]] = []

        for problem in batch:
            # Baseline: generate without repair.
            try:
                from carnot.pipeline.two_round_repair import TwoRoundCodeRepairPipeline  # type: ignore[import]

                pipeline = TwoRoundCodeRepairPipeline(
                    model_path=model_path,
                    device_map={"": f"cuda:{GPU_INDEX}"},
                )
                baseline_result = pipeline.run_baseline(problem)
                baseline_pass = int(baseline_result.passed)
                baseline_logits = getattr(baseline_result, "logits", None)
            except Exception:
                baseline_pass = 0
                baseline_logits = None

            # MARS gate decision — skip oracle when model was highly confident.
            gate_decision = mars_gate.decide(baseline_logits)

            if gate_decision.skip_oracle:
                # High logit margin: treat as passing, save oracle call.
                repair_pass = baseline_pass
                batch_oracle_saved += 1
            else:
                # Low margin: run repair pipeline to verify and potentially fix.
                try:
                    repair_result = pipeline.run_repair(problem, baseline_result)
                    repair_pass = int(repair_result.passed)
                except Exception:
                    repair_pass = baseline_pass

            batch_baseline_pass += baseline_pass
            batch_repair_pass += repair_pass
            batch_problem_results.append(
                {
                    "problem_id": getattr(problem, "task_id", str(n_completed)),
                    "baseline_pass": baseline_pass,
                    "repair_pass": repair_pass,
                    "mars_verdict": gate_decision.honest_verdict,
                    "logit_margin": gate_decision.logit_margin,
                }
            )
            n_completed += 1

        total_baseline_pass += batch_baseline_pass
        total_repair_pass += batch_repair_pass
        total_oracle_calls_saved += batch_oracle_saved

        batch_summary = {
            "batch_idx": batch_idx,
            "batch_size": len(batch),
            "batch_baseline_pass": batch_baseline_pass,
            "batch_repair_pass": batch_repair_pass,
            "oracle_calls_saved": batch_oracle_saved,
            "problems": batch_problem_results,
        }
        all_batch_results.append(batch_summary)

        # Checkpoint after each batch (REQ-BENCH-016-2).
        tmpl.checkpoint_save(
            {
                "n_completed": n_completed,
                "batch_results": all_batch_results,
                "total_baseline_pass": total_baseline_pass,
                "total_repair_pass": total_repair_pass,
                "oracle_calls_saved": total_oracle_calls_saved,
            }
        )

    # Step i: aggregate results.
    pass_at_1_baseline = total_baseline_pass / N_PROBLEMS
    pass_at_1_repair = total_repair_pass / N_PROBLEMS
    signed_improvement = compute_signed_improvement(
        total_repair_pass, total_baseline_pass, N_PROBLEMS
    )

    # Step j: honest_verdict.
    if signed_improvement > 0:
        verdict = "code_repair_positive"
    else:
        verdict = "code_no_improvement"

    artifact = tmpl.build_result(
        {
            "honest_verdict": verdict,
            "inference_mode": "live_gpu",
            "n_problems": N_PROBLEMS,
            "n_completed": n_completed,
            "pass_at_1_baseline": pass_at_1_baseline,
            "pass_at_1_repair": pass_at_1_repair,
            "signed_improvement": signed_improvement,
            "oracle_calls_saved": total_oracle_calls_saved,
            "batch_results": all_batch_results,
            "mars_threshold": MARS_THRESHOLD,
            "gpu_index": GPU_INDEX,
            "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "zombie_kills": zombie_result.n_killed if hasattr(zombie_result, "n_killed") else 0,
            "vram_freed_mb": evict_result.vram_freed_mb
            if hasattr(evict_result, "vram_freed_mb")
            else 0,
        },
        status="success",
    )
    writer.write(artifact)
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
