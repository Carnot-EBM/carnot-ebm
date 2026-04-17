#!/usr/bin/env python3
"""Experiment 442 — FOVER annotation on live GPU CoT data from Exp 439.

**Researcher summary:**
    FR-11 (autonomous self-learning) has been unconfirmed for 8 consecutive milestones
    because all EORM/JEPA retrains used synthetic data only (honest_verdict=synthetic_only).
    Exp 430 implemented FOVERAnnotator (Z3 step annotation on synthetic CoT).
    This experiment (442) runs FOVERAnnotator on Exp 439's LIVE GPU CoT data to produce
    REAL training pairs for downstream EORM/JEPA retrain.

**Data source decision (honest gate):**
    - If ``results/experiment_439_live_cot.json`` exists AND the companion
      ``results/experiment_439_live_precision_micro.json`` confirms inference_mode='live_gpu',
      treat the cot_responses as real data → source='live'.
    - Otherwise fall back to 100 synthetic GSM8K-style CoT responses → source='synthetic'.

**Honest verdict (REQ-LEARN-035):**
    - 'real_data_labeled'      — source='live' AND n_labeled >= 20 (FR-11 can proceed)
    - 'real_data_insufficient' — source='live' AND n_labeled < 20 (investigate regex)
    - 'synthetic_fallback'     — source='synthetic' (same as prior milestones)

**Output files:**
    - results/experiment_442_fover_live_annotation.json  (top-level artifact)
    - results/fover_labeled_steps_live.json              (training pairs — higher quality
      than Exp 430's fover_labeled_steps.json when source='live')

Spec: REQ-LEARN-035, SCENARIO-LEARN-062, SCENARIO-LEARN-063,
      REQ-INFRA-021, REQ-INFRA-022
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# apply_env_autofix() FIRST — must precede any CUDA/torch import.
# See RETRO-022: CARNOT_FORCE_LIVE=1 must be in the subprocess env before
# any GPU framework initializes.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.fover_annotator import FOVERAnnotator  # noqa: E402
from carnot.pipeline.fover_live import (  # noqa: E402
    LiveFOVERResult,
    build_live_fover_artifact,
)
from scripts.experiment_template import REQUIRED_RESULT_FIELDS  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 442
EXP_TITLE = "FOVER annotation on live GPU CoT data (Exp 439)"
DELIVERABLE = str(_REPO_ROOT / "results" / "experiment_442_fover_live_annotation.json")
LABELED_STEPS_PATH = str(_REPO_ROOT / "results" / "fover_labeled_steps_live.json")

_COT_PATH = _REPO_ROOT / "results" / "experiment_439_live_cot.json"
_COMPANION_PATH = _REPO_ROOT / "results" / "experiment_439_live_precision_micro.json"

# Number of synthetic CoT responses to generate when live data is unavailable.
_N_SYNTHETIC = 100


# ---------------------------------------------------------------------------
# Synthetic data generator (fallback)
# ---------------------------------------------------------------------------


def _generate_synthetic_responses(n: int) -> list[dict]:
    """Generate simple GSM8K-style CoT responses for synthetic fallback.

    Each response contains two arithmetic steps — one correct and one incorrect —
    so that Z3 can label at least some pairs per response.  The synthetic data is
    explicitly marked as synthetic in the artifact, so it does not affect the
    honest_verdict for FR-11.

    Args:
        n: Number of responses to generate.

    Returns:
        List of response dicts with keys: question_id, response.
    """
    responses = []
    for i in range(n):
        a = i + 1
        b = i + 2
        correct_sum = a + b
        wrong_sum = correct_sum + 1  # deliberately incorrect for labeling variety
        response = (
            f"1. First I add {a} + {b} = {correct_sum}.\n"
            f"2. Then I check: {a} + {b} = {wrong_sum}.\n"
            f"The answer is {correct_sum}."
        )
        responses.append({"question_id": f"synthetic_{i}", "response": response})
    return responses


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def _load_live_data() -> tuple[list[dict], str]:
    """Try to load Exp 439 live CoT responses.

    Returns:
        (responses, source) where source is 'live' or 'synthetic'.
    """
    # Check companion for inference_mode confirmation.
    companion_live = False
    if _COMPANION_PATH.exists():
        try:
            companion = json.loads(_COMPANION_PATH.read_text())
            if companion.get("inference_mode") == "live_gpu":
                companion_live = True
        except Exception:
            pass

    if _COT_PATH.exists() and companion_live:
        try:
            data = json.loads(_COT_PATH.read_text())
            cot_items = data.get("cot_responses", [])
            if cot_items:
                _log.info(
                    "Loaded %d live CoT responses from Exp 439 (companion confirmed live_gpu).",
                    len(cot_items),
                )
                return cot_items, "live"
        except Exception as exc:
            _log.warning("Failed to parse Exp 439 CoT file: %s", exc)

    _log.info(
        "Live CoT data unavailable or not confirmed live_gpu — using %d synthetic responses.",
        _N_SYNTHETIC,
    )
    return _generate_synthetic_responses(_N_SYNTHETIC), "synthetic"


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------


def run_experiment() -> dict:
    """Run FOVER annotation on live (or synthetic) CoT data and write results.

    Returns:
        The JSON-serializable artifact dict written to DELIVERABLE.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    # 1. Load data.
    responses, source = _load_live_data()

    # 2. Run FOVERAnnotator.
    annotator = FOVERAnnotator(z3_timeout_seconds=5)
    annotated = annotator.annotate_corpus(responses)
    training_pairs = annotator.to_training_pairs(annotated, responses)

    # 3. Compute counts.
    n_steps_found = sum(len(steps) for steps in annotated)
    n_correct = sum(1 for p in training_pairs if p["label"] == "correct")
    n_incorrect = sum(1 for p in training_pairs if p["label"] == "incorrect")
    n_labeled = n_correct + n_incorrect
    n_not_verifiable = n_steps_found - n_labeled
    labeling_rate = (n_labeled / n_steps_found) if n_steps_found > 0 else 0.0

    result = LiveFOVERResult(
        n_responses=len(responses),
        n_steps_found=n_steps_found,
        n_labeled=n_labeled,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        n_not_verifiable=n_not_verifiable,
        labeling_rate=labeling_rate,
        source=source,  # type: ignore[arg-type]
        honest_verdict="",  # filled by build_live_fover_artifact
    )
    fover_art = build_live_fover_artifact(result)

    # 4. Write labeled training pairs (separate from Exp 430's file).
    labeled_path = Path(LABELED_STEPS_PATH)
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_path.write_text(json.dumps(training_pairs, indent=2))
    _log.info(
        "Wrote %d labeled training pairs to %s", len(training_pairs), labeled_path
    )

    finished_at = datetime.now(timezone.utc).isoformat()
    duration_s = round(time.monotonic() - t0, 3)

    # 5. Build top-level artifact.
    artifact: dict[str, Any] = {
        "experiment": EXP_ID,
        "title": EXP_TITLE,
        "run_date": started_at[:10],
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "status": "success",
        "schema": "carnot.fover_live.v1",
        "honest_verdict": fover_art["honest_verdict"],
        "env_autofix": {
            "verdict": getattr(_autofix_result, "verdict", "unknown"),
        },
        "fover_live": fover_art,
        "n_responses": result.n_responses,
        "n_steps_found": result.n_steps_found,
        "n_labeled": result.n_labeled,
        "source": result.source,
        "labeled_steps_path": LABELED_STEPS_PATH,
    }

    # Validate all required fields are present.
    for field in REQUIRED_RESULT_FIELDS:
        assert field in artifact, f"BUG: missing required field '{field}'"

    out_path = Path(DELIVERABLE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Wrote artifact to %s (honest_verdict=%s)", out_path, artifact["honest_verdict"])

    return artifact


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — wraps run_experiment in ExperimentTimeoutWatchdog."""
    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30, result_path=DELIVERABLE):
        run_experiment()


if __name__ == "__main__":
    main()
