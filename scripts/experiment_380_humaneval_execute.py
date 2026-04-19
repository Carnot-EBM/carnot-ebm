#!/usr/bin/env python3
"""Experiment 380: Live HumanEval code verification benchmark — LiveGPUGate execution.

**Researcher summary:**
    Re-executes the HumanEval code verification benchmark (Exp 369) using the
    LiveGPUGate hard gate introduced in Exp 377 (RETRO-015 fix).  Exp 226 showed
    +3.0pp on a prior live run (19/164 → 24/164); this experiment confirms or
    refutes that result with the current stack and the new gate infrastructure.

**Why this is different from Exp 369:**
    Exp 369 used the raw ``diagnose_live_gpu()`` call + manual ``os.environ.get()``
    check.  Exp 380 uses the consolidated ``LiveGPUGate.require_live_or_blocked()``
    introduced in Exp 377 — a single call that checks env var + GPU liveness and
    returns a blocked artifact immediately if either layer fails.

    This means Exp 380 CANNOT silently fall through to simulated mode.  Any
    environment where ``CARNOT_FORCE_LIVE=1`` is not set or the GPU is not live
    produces a blocked artifact and exits cleanly.

**Why code verification is reliable:**
    ArithmeticExtractor relies on finding arithmetic expressions in text — that
    pattern returns 0 violations on instruction-tuned models (Exp 328).
    CodeExtractor avoids that brittleness entirely: it runs the code against test
    cases and detects failures structurally (wrong output, runtime error, type
    mismatch).  No regex needed.  VerifyRepairPipeline feeds failure details back
    to the LLM to attempt a repaired solution.

**Pipeline per problem:**
    1. Generate code with Gemma4-E4B-it.
    2. Run official test cases (subprocess, 10s timeout) — record pass/fail.
    3. If failed: run CodeExtractor + VerifyRepairPipeline to attempt repair.
    4. Re-run official tests on repaired code — record final pass/fail.
    5. For solutions that PASS official tests: run PBT to detect unofficial bugs.

**Honest verdict rules (SCENARIO-BENCH-021):**
    ``honest_verdict="code_verification_positive"`` only when:
    1. ``inference_mode == "live_gpu"``
    2. ``signed_improvement > 0``

**Metrics:**
    - pass_at_1_before: fraction passing on first generation (before any repair)
    - pass_at_1_after: fraction passing after the verify-repair loop
    - signed_improvement: pass_at_1_after − pass_at_1_before (signed; no clamping)
    - pbt_bugs_found: count of solutions that passed official tests but failed PBT

**Output:** results/experiment_380_humaneval_execute.json

Usage:
    # Live mode (requires GPU + CARNOT_FORCE_LIVE=1):
    CARNOT_FORCE_LIVE=1 .venv/bin/python scripts/experiment_380_humaneval_execute.py

    # CI / no-GPU: produces a blocked artifact immediately
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_380_humaneval_execute.py

Spec: REQ-BENCH-004, SCENARIO-BENCH-021
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow import from python/ and scripts/ without installation
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

# Import shared helpers from Exp 369 — avoids duplicating tested code.
# These functions are stable and covered by test_experiment_369_humaneval_live.py.
from experiment_369_humaneval_live import (  # noqa: E402
    HumanEvalResult369,
    _extract_code,
    _load_model_pipeline,
    _load_problems,
    _parse_official_tests,
    _process_problem,
    _run_pbt,
    _run_tests,
    _run_tests_subprocess,
    build_humaneval_artifact_v2,
    compute_pass_at_1,
    compute_pass_at_1_after_repair,
)

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 380
EXP_TITLE = "Live HumanEval code verification benchmark — LiveGPUGate execution"
DELIVERABLE = "results/experiment_380_humaneval_execute.json"
_MODEL_IDS = ["google/gemma-4-E4B-it"]

# ---------------------------------------------------------------------------
# Artifact write helper
# ---------------------------------------------------------------------------


def _write_artifact(tmpl: ExperimentTemplate, artifact: dict[str, Any]) -> None:
    """Write the experiment artifact JSON to disk.

    Creates the results/ directory if needed, then writes the artifact as
    pretty-printed JSON.  The output path is derived from the ExperimentTemplate
    deliverable field.

    Args:
        tmpl: ExperimentTemplate instance (provides _output_path).
        artifact: Dict to serialise.

    Spec: REQ-BENCH-004
    """
    tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
    tmpl._output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", tmpl._output_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 380: live HumanEval code verification benchmark.

    Uses LiveGPUGate.require_live_or_blocked() (Exp 377 infrastructure) as the
    single consolidated hard gate.  If the gate fails (env var missing or GPU
    not live), a blocked artifact is written and the function returns immediately.

    After the gate passes, setup_gpu() pre-warms the model.  If the model is not
    healthy, a blocked artifact is written and the function returns immediately.

    There is NO simulated-mode fallback.  A blocked artifact is always better
    than a synthetic result labelled "live_gpu".

    Gate sequence:
        1. LiveGPUGate.require_live_or_blocked() — env var + GPU liveness.
        2. tmpl.setup_gpu() — model pre-warm + health check.
        3. _load_model_pipeline() — load tokenizer + model weights.
        Any gate failure produces a blocked artifact.

    Spec: REQ-BENCH-004, SCENARIO-BENCH-021
    """
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # ---------------------------------------------------------------------------
    # Hard gate: LiveGPUGate.require_live_or_blocked() (RETRO-015 fix, Exp 377)
    # Checks CARNOT_FORCE_LIVE=1 AND diagnose_live_gpu().is_live_capable.
    # Returns a blocked artifact dict on failure; None on success.
    # ---------------------------------------------------------------------------
    blocked = LiveGPUGate.require_live_or_blocked(tmpl, _MODEL_IDS)
    if blocked is not None:
        _log.error(
            "LiveGPUGate blocked Exp 380 — writing blocked artifact and exiting."
        )
        _write_artifact(tmpl, blocked)
        return

    # ---------------------------------------------------------------------------
    # GPU pre-warm via ExperimentTemplate.setup_gpu() (Exp 294 pattern).
    # The prewarm layer confirms the model is loadable and not stalled.
    # ---------------------------------------------------------------------------
    gpu_status = tmpl.setup_gpu(
        [{"name": "Gemma4-E4B-it", "hf_id": _MODEL_IDS[0], "gpu": 0}]
    )
    if not gpu_status["all_healthy"]:
        _log.error(
            "setup_gpu reports unhealthy — writing blocked artifact.  models=%s",
            gpu_status["models"],
        )
        artifact = tmpl.build_result(
            {
                "humaneval_schema": "carnot.humaneval_benchmark.v2",
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "failure_reason": "setup_gpu health check failed",
                "n_problems": 0,
                "pass_at_1_before": 0.0,
                "pass_at_1_after": 0.0,
                "signed_improvement": 0.0,
                "pbt_bugs_found": 0,
                "gpu_status": gpu_status,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    inference_mode = "live_gpu"
    _log.info("GPU gate passed — inference_mode=%s", inference_mode)

    # ---------------------------------------------------------------------------
    # Load model weights (tokenizer + causal LM).
    # ---------------------------------------------------------------------------
    tokenizer, model, device, ok = _load_model_pipeline(
        hf_id=_MODEL_IDS[0], device=0
    )
    if not ok:
        _log.error("_load_model_pipeline failed — writing blocked artifact.")
        artifact = tmpl.build_result(
            {
                "humaneval_schema": "carnot.humaneval_benchmark.v2",
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "failure_reason": "model load failed after GPU gate passed",
                "n_problems": 0,
                "pass_at_1_before": 0.0,
                "pass_at_1_after": 0.0,
                "signed_improvement": 0.0,
                "pbt_bugs_found": 0,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    # ---------------------------------------------------------------------------
    # Load problems.
    # ---------------------------------------------------------------------------
    problems = _load_problems()
    _log.info("[Exp 380] %d problems loaded.", len(problems))

    # ---------------------------------------------------------------------------
    # Process problems with checkpointing every 10.
    # ---------------------------------------------------------------------------
    results: list[HumanEvalResult369] = []
    for i, problem in enumerate(problems):
        try:
            result = _process_problem(problem, tokenizer, model, device)
            results.append(result)
        except Exception as exc:
            _log.warning("[Exp 380] problem %d error: %r", i, exc)
            results.append(
                HumanEvalResult369(
                    problem_id=problem.get("task_id", f"unknown/{i}"),
                    generated_code="",
                    passed_tests=False,
                    violations_found=0,
                    repair_attempted=False,
                    final_code="",
                    final_passed_tests=False,
                    pbt_bug_found=False,
                )
            )

        if (i + 1) % 10 == 0:
            tmpl.checkpoint_save(
                {
                    "completed": i + 1,
                    "partial_results": [asdict(r) for r in results],
                },
                step=i + 1,
            )

    humaneval_data = build_humaneval_artifact_v2(results, inference_mode)

    _log.info(
        "[Exp 380] pass@1_before=%.3f  pass@1_after=%.3f  "
        "signed_improvement=%+.3f  honest_verdict=%s  pbt_bugs=%d",
        humaneval_data["pass_at_1_before"],
        humaneval_data["pass_at_1_after"],
        humaneval_data["signed_improvement"],
        humaneval_data["honest_verdict"],
        humaneval_data["pbt_bugs_found"],
    )

    artifact = tmpl.build_result(humaneval_data, status="success")
    _write_artifact(tmpl, artifact)


if __name__ == "__main__":
    main()


# --- Exp 495 HarnessPatcher: DualGPUHarness.apply() injected — REQ-INFRA-057 ---
# Auto-injected because HarnessAudit flagged this script as loading two models
# without assigning any model to cuda:1.  apply() pins model[0] to cuda:0 and
# model[1] to cuda:1 when CARNOT_FORCE_LIVE=1 is set.  It is a no-op in CI so
# this block is safe to leave in place permanently.
try:
    from carnot.pipeline.dual_gpu_harness import DualGPUHarness as _Exp495DGH
    if "MODEL_SPECS" in vars():
        MODEL_SPECS = _Exp495DGH.from_env().apply(MODEL_SPECS)  # cuda:1 → model[1]
except Exception:  # noqa: BLE001
    pass  # best-effort injection; script continues even if harness import fails
