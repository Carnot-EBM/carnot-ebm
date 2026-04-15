#!/usr/bin/env python3
"""Experiment 353: Live GPU inference smoke test — gate before benchmark experiments.

**Researcher summary:**
    Experiments 340, 341, 346, 347 all produced artifacts labelled
    ``inference_mode="live_gpu"`` that actually contained synthetic answers.
    Exp 352 diagnosed the root cause: a silent fallback in ``setup_gpu()`` when
    model pre-warm failed.  The fix in Exp 352 added ``REQ-INFRA-014`` (explicit
    RuntimeError instead of silent fallback).

    This experiment (353) VERIFIES that the fix actually works end-to-end:
    - Runs 5 GSM8K questions through ``google/gemma-4-E4B-it`` with
      ``CARNOT_FORCE_LIVE=1``.
    - Checks that the result has ``inference_mode="live_gpu"`` (not "simulated").
    - Produces ``honest_verdict="live_confirmed"`` if successful.
    - Produces ``honest_verdict="blocked_error"`` and status="blocked" if
      GPU inference was unavailable.

    This experiment is the GATING CHECK: all subsequent benchmark experiments
    (354+) must only run after this produces ``live_confirmed``.

**CI-safe simulated mode:**
    When ``CARNOT_FORCE_LIVE`` is not set, this experiment skips GPU inference
    and writes an artifact with ``inference_mode="ci_skip"``.  This ensures
    CI never fails due to missing GPUs.

**Output:** results/experiment_353_live_gpu_smoke_test.json

Spec: REQ-BENCH-005, SCENARIO-BENCH-012, SCENARIO-BENCH-013
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure repo root is on sys.path so scripts.* and carnot.* resolve.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.smoke_test import (  # noqa: E402
    SmokeTestResult,
    build_smoke_test_artifact,
    run_smoke_test,
)

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 353
EXP_TITLE = "Live GPU inference smoke test — gate before benchmark experiments"
DELIVERABLE = "results/experiment_353_live_gpu_smoke_test.json"
MODEL_ID = "google/gemma-4-E4B-it"
N_QUESTIONS = 5
TIMEOUT_S = 300


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 353: live GPU smoke test."""
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
    _log.info(
        "Experiment %d — CARNOT_FORCE_LIVE=%s model=%s n_questions=%d",
        EXP_ID,
        "1" if force_live else "0",
        MODEL_ID,
        N_QUESTIONS,
    )

    # --- Run smoke test ---
    # run_smoke_test raises RuntimeError if CARNOT_FORCE_LIVE=1 and GPU is
    # unavailable.  We catch it here to produce a structured blocked artifact
    # rather than crashing the conductor with an unhandled exception.
    smoke_result: SmokeTestResult | None = None
    blocked_by_error: str = ""

    try:
        smoke_result = run_smoke_test(MODEL_ID, n_questions=N_QUESTIONS, timeout_s=TIMEOUT_S)
    except RuntimeError as exc:
        blocked_by_error = str(exc)
        _log.error(
            "Experiment %d BLOCKED: run_smoke_test raised RuntimeError: %s",
            EXP_ID,
            exc,
        )

    # --- Build artifact ---
    if smoke_result is not None:
        smoke_artifact = build_smoke_test_artifact(smoke_result)
        honest_verdict = smoke_artifact["honest_verdict"]
    else:
        # run_smoke_test raised — we create a synthetic blocked result so we
        # can produce a structured artifact.
        smoke_result = SmokeTestResult(
            inference_mode="blocked",
            n_questions=N_QUESTIONS,
            n_answered=0,
            elapsed_s=0.0,
            model_id=MODEL_ID,
            is_live=False,
            blocked_reason=blocked_by_error,
        )
        smoke_artifact = build_smoke_test_artifact(smoke_result)
        honest_verdict = "blocked_error"

    _log.info(
        "Experiment %d — honest_verdict=%s inference_mode=%s n_answered=%d",
        EXP_ID,
        honest_verdict,
        smoke_result.inference_mode,
        smoke_result.n_answered,
    )

    # --- Determine experiment status ---
    # Only "live_confirmed" counts as success — any other verdict is blocked.
    # We do NOT silently continue with simulated mode; that was the Exp 340 bug.
    if honest_verdict == "live_confirmed":
        status = "success"
        _log.info(
            "Experiment %d SUCCESS: live GPU inference confirmed — "
            "n_answered=%d/%d elapsed_s=%.3f",
            EXP_ID,
            smoke_result.n_answered,
            smoke_result.n_questions,
            smoke_result.elapsed_s,
        )
    else:
        status = "blocked"
        _log.warning(
            "Experiment %d BLOCKED: honest_verdict=%s — "
            "do NOT proceed to benchmark experiments with simulated mode",
            EXP_ID,
            honest_verdict,
        )

    # Build final artifact using ExperimentTemplate for standard required fields.
    artifact = tmpl.build_result(
        smoke_artifact,
        status=status,
    )

    # Write artifact.
    output_path = tmpl._output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", output_path)


if __name__ == "__main__":
    main()
