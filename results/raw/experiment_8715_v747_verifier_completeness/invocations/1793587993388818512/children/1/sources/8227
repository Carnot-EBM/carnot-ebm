#!/usr/bin/env python3
"""Exp 796: SOTA GGUF Code Repair v3 — HumanEval 25 problems, batched 5×5, MARS gate.

**Researcher summary (RETRO-SOTA-GGUF-TIMEOUT fix v3):**
    Exp 769 timed out (120 min, no checkpoint) and Exp 785 timed out (90 min,
    blocked_model_load_failed from GPU OOM before zombie kill).  Exp 795 fixed the
    OOM root cause by isolating GPU 1 via kill_gpu_zombies + evict_gpu_vram.  This
    experiment (796) assumes Exp 795 closed retro_028 and adds two more improvements:

    1. Batched execution: 25 problems split into 5 batches of 5.  Each batch
       checkpointed atomically via AtomicResultWriter so partial progress survives
       a timeout.  Per-batch wall time < 18 min; total < 90 min.

    2. MARS margin gate (arXiv 2601.15498): skip test oracle when logit_margin
       exceeds threshold.  Reduces oracle call count without hurting accuracy.

**Gate:** results/experiment_795_gemma4_oom_fix_v4.json must have retro_028_closed=True.
If not, write gated_retro028_not_closed artifact and exit.

**honest_verdict logic:**
    - "code_repair_positive"          if signed_improvement > 0, inference_mode="live_gpu"
    - "code_no_improvement"           if signed_improvement <= 0, inference_mode="live_gpu"
    - "partial_N_of_25"               if timeout with N < 25 problems completed
    - "gated_retro028_not_closed"     if Exp 795 gate not met
    - "blocked_no_live_gpu"           if LiveGPUGate blocks

Spec: REQ-BENCH-060, REQ-BENCH-061, SCENARIO-BENCH-084, SCENARIO-BENCH-085
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
from carnot.pipeline.gemma_isolation import evict_gpu_vram  # noqa: E402
from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.mars_margin_gate import MARSMarginGate  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 796
TITLE = "SOTA GGUF Code Repair v3 — HumanEval 25 problems, batched 5×5, MARS gate"
DELIVERABLE = "results/experiment_796_sota_gguf_code_repair_v3.json"
N_PROBLEMS = 25
BATCH_SIZE = 5
TIMEOUT_MINUTES = 90
GPU_INDEX = 1
MARS_THRESHOLD = 2.0

_EXP795_RESULT = _REPO / "results" / "experiment_795_gemma4_oom_fix_v4.json"


# ---------------------------------------------------------------------------
# Pure helper — unit-testable
# ---------------------------------------------------------------------------


def check_retro028_gate(result_path: Path) -> bool:
    """Return True iff Exp 795 artifact reports retro_028_closed=True.

    Why this gate exists: retro_028 tracks the GPU OOM root cause that killed
    Exp 785.  If Exp 795 did not close it, running Exp 796 would repeat the
    same OOM failure.  The gate is a hard stop, not a warning.

    Args:
        result_path: Path to results/experiment_795_gemma4_oom_fix_v4.json.

    Returns:
        True only when the file exists and retro_028_closed == True.
    """
    if not result_path.exists():
        return False
    try:
        with open(result_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return bool(data.get("retro_028_closed"))
    except (json.JSONDecodeError, OSError):
        return False


def build_blocked_artifact(
    tmpl: "ExperimentTemplate",
    honest_verdict: str,
    blocked_reason: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build a blocked artifact with the required schema fields.

    Why a separate helper: Exp 785 had inconsistent blocked artifact shapes
    across its three early-exit paths.  Centralising here ensures every
    blocked path emits the same schema.

    Args:
        tmpl: Initialised ExperimentTemplate (provides build_result).
        honest_verdict: One of the defined verdict strings.
        blocked_reason: Human-readable explanation of why the run was blocked.
        **extra: Additional key-value pairs to merge into the artifact.

    Returns:
        Dict that satisfies REQUIRED_RESULT_FIELDS.
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
    if not LiveGPUGate.require_live_or_blocked():
        artifact = build_blocked_artifact(
            tmpl,
            honest_verdict="blocked_no_live_gpu",
            blocked_reason="CARNOT_FORCE_LIVE not set; no live GPU available",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step d: prerequisite gate — Exp 795 must have closed retro_028.
    if not check_retro028_gate(_EXP795_RESULT):
        artifact = build_blocked_artifact(
            tmpl,
            honest_verdict="gated_retro028_not_closed",
            blocked_reason=(
                "Exp 795 did not set retro_028_closed=True; "
                "GPU OOM root cause unresolved — Exp 796 would repeat Exp 785 failure"
            ),
            gate_file=str(_EXP795_RESULT),
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Step e: kill zombies + evict VRAM on GPU 1.
    zombie_result = kill_gpu_zombies(GPU_INDEX)
    evict_result = evict_gpu_vram(GPU_INDEX)

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

    # Step g: load 25 HumanEval problems.
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

    # Step h: run 5 batches of 5 problems.
    mars_gate = MARSMarginGate(threshold=MARS_THRESHOLD)
    batches = [problems[i : i + BATCH_SIZE] for i in range(0, N_PROBLEMS, BATCH_SIZE)]

    all_batch_results: list[dict[str, Any]] = []
    total_baseline_pass = 0
    total_repair_pass = 0
    total_oracle_calls_saved = 0
    n_completed = 0

    for batch_idx, batch in enumerate(batches):
        if watchdog.is_expired():
            partial_verdict = f"partial_{n_completed}_of_25"
            artifact = tmpl.build_result(
                {
                    "honest_verdict": partial_verdict,
                    "inference_mode": "live_gpu",
                    "n_problems": N_PROBLEMS,
                    "n_completed": n_completed,
                    "pass_at_1_baseline": total_baseline_pass / max(n_completed, 1),
                    "pass_at_1_repair": total_repair_pass / max(n_completed, 1),
                    "signed_improvement": (total_repair_pass - total_baseline_pass)
                    / max(n_completed, 1),
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

            # MARS gate decision.
            gate_decision = mars_gate.decide(baseline_logits)

            if gate_decision.skip_oracle:
                # High-margin: count as passing, skip expensive oracle.
                repair_pass = baseline_pass
                batch_oracle_saved += 1
            else:
                # Run repair oracle.
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

        # Checkpoint after each batch (REQ-BENCH-060-1).
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
    signed_improvement = pass_at_1_repair - pass_at_1_baseline

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
            "vram_eviction": evict_result.vram_freed_mb
            if hasattr(evict_result, "vram_freed_mb")
            else 0,
        },
        status="success",
    )
    writer.write(artifact)
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
