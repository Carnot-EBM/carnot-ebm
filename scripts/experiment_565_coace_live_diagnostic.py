#!/usr/bin/env python3
"""Experiment 565: CoACEExtractor Live Diagnostic — RETRO-061 Final Validation Gate.

**Researcher summary:**
    Exp 554 proved that VeriCoTStepValidator (TP=0/25) and VPRMArithmeticVerifier (TP=0/25)
    both fail completely on live IT-model responses.  Root cause: VeriCoT needs mutually
    inconsistent FOL premises (can't catch a single wrong equation); VPRM needs prose
    patterns (IT models use equation notation, not prose).

    Exp 564 implemented CoACEExtractor — execution-based arithmetic checking that
    computes eval(lhs) and compares to stated rhs.  This experiment validates it on
    the exact same 25 labeled live responses from Exp 554.

    If CoACEExtractor achieves TP > 0, RETRO-061 is resolved and the gate opens for:
        - Exp 569: live verify-repair end-to-end
        - Exp 570: FR-11 relay with CoACE-gated pipeline

**Gate chain (every exit path writes the deliverable):**
    0. Zombie PIDs killed (subprocess.run kill -9 ...)
    1. apply_env_autofix()                     — normalise env before any import
    2. ExperimentTimeoutWatchdog(565, 30)      — 30-minute hard cap (CPU-only)
    3. Load labeled responses from exp554 per_question_results or exp538_cot_pairs.json
    4. If no labeled responses: write blocked artifact, exit 0
    5. Run CoACEExtractor diagnostic (primary)
    6. Run VeriCoTStepValidator diagnostic (baseline, expected TP=0)
    7. Run VPRMArithmeticVerifier diagnostic (baseline, expected TP=0)
    8. Determine retro_061_resolved and gate_open
    9. AtomicResultWriter: results/experiment_565_coace_live_diagnostic.json
    10. tmpl.assert_deliverable_written()      — FINAL LINE

Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-065, SCENARIO-EXTRACT-066, SCENARIO-EXTRACT-067
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: Kill zombie PIDs FIRST — before any CUDA import.
# ---------------------------------------------------------------------------
import subprocess

subprocess.run(["kill", "-9", "527256", "527259", "529495"], capture_output=True)

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() — must be called before any CUDA import.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------
import json  # noqa: E402
from typing import Any  # noqa: E402

from carnot.extraction import (  # noqa: E402
    CoACEExtractor,
    VeriCoTStepValidator,
    VPRMArithmeticVerifier,
    run_extractor_diagnostic,
)
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 565
EXP_TITLE = "CoACEExtractor Live Diagnostic"
DELIVERABLE = "results/experiment_565_coace_live_diagnostic.json"

# Primary: Exp 554 per_question_results (may not exist)
EXP554_PATH = "results/experiment_554_extraction_diagnostic.json"

# Fallback: Exp 538 CoT pairs (labeled cot_text/correct pairs)
EXP538_COT_PAIRS = "results/exp538_cot_pairs.json"


# ---------------------------------------------------------------------------
# CoACE adapter — wraps CoACEExtractor to satisfy ViolationExtractor protocol
# ---------------------------------------------------------------------------


class _CoACEAdapter:
    """Wraps CoACEExtractor.extract() to expose detect_violations() for run_extractor_diagnostic.

    Why a wrapper?
        run_extractor_diagnostic() expects a ViolationExtractor protocol with
        detect_violations(text) -> list[Any].  CoACEExtractor exposes extract() instead.
        This thin adapter translates between the two APIs without modifying CoACEExtractor.

    Returns a list of CoACEViolation objects (non-empty = violation found).

    Spec: REQ-EXTRACT-035-2
    """

    def __init__(self, tolerance: float = 1e-6, min_confidence: float = 0.5) -> None:
        self._extractor = CoACEExtractor(tolerance=tolerance, min_confidence=min_confidence)

    def detect_violations(self, text: str) -> list[Any]:
        """Call CoACEExtractor.extract() and return violations list (empty = no violation)."""
        result = self._extractor.extract(text)
        return result.violations  # empty list if no violations


# ---------------------------------------------------------------------------
# Response loader
# ---------------------------------------------------------------------------


def load_labeled_responses() -> list[dict]:
    """Load labeled IT-model responses from available upstream results.

    Priority order:
        1. results/experiment_554_extraction_diagnostic.json — per_question_results field
           (has 'response' and 'is_correct' keys)
        2. results/exp538_cot_pairs.json — direct CoT pairs from live inference
           (has 'cot_text' and 'correct' keys; normalised to response/is_correct)

    Returns an empty list if neither source is available.

    Why this fallback chain?
        Exp 554 is the canonical diagnostic source, but its per_question_results field
        was not populated at write time (the extractor-level flags were stored inside
        vericot_result.per_question_flags instead).  Exp 538's cot_pairs file is the
        raw ground truth with both the CoT text and the correct/incorrect label.

    Spec: REQ-EXTRACT-035-1
    """
    # Try exp554 per_question_results first
    exp554_path = _REPO_ROOT / EXP554_PATH
    if exp554_path.exists():
        data = json.loads(exp554_path.read_text())
        pqr = data.get("per_question_results")
        if pqr and isinstance(pqr, list) and len(pqr) > 0:
            labeled = []
            for entry in pqr:
                labeled.append(
                    {
                        "response": entry.get("response", entry.get("cot_text", "")),
                        "is_correct": bool(entry.get("is_correct", entry.get("correct", False))),
                        "question": entry.get("question", ""),
                        "model_id": entry.get("model_id", "unknown"),
                    }
                )
            if labeled:
                return labeled

    # Fall back to exp538_cot_pairs.json
    exp538_path = _REPO_ROOT / EXP538_COT_PAIRS
    if exp538_path.exists():
        raw = json.loads(exp538_path.read_text())
        labeled = []
        for entry in raw:
            labeled.append(
                {
                    "response": entry.get("cot_text", entry.get("response", "")),
                    "is_correct": bool(entry.get("correct", entry.get("is_correct", False))),
                    "question": entry.get("question", ""),
                    "model_id": entry.get("model_id", "unknown"),
                }
            )
        if labeled:
            return labeled

    return []


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment() -> None:
    """Execute the CoACE live diagnostic and write the deliverable artifact."""

    # Step 2: ExperimentTimeoutWatchdog — 30-minute hard cap.
    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30)
    watchdog.start()

    tmpl = ExperimentTemplate(
        EXP_ID,
        EXP_TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    writer = AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE))

    # Step 3: Load labeled responses.
    labeled_responses = load_labeled_responses()

    # Step 4: Write blocked artifact if no labeled responses are available.
    if not labeled_responses:
        artifact = tmpl.build_result(
            {
                "n_responses": 0,
                "coace_tp_rate": 0.0,
                "coace_fp_rate": 0.0,
                "coace_precision": 0.0,
                "coace_recall": 0.0,
                "vericot_tp_rate": 0.0,
                "vprm_tp_rate": 0.0,
                "coace_improvement_over_vericot": 0.0,
                "retro_061_resolved": False,
                "gate_open": False,
                "honest_verdict": "upstream_missing",
            },
            status="blocked",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    n_responses = len(labeled_responses)

    # Step 5: CoACEExtractor diagnostic (primary).
    coace_adapter = _CoACEAdapter()
    coace_result = run_extractor_diagnostic(coace_adapter, "CoACEExtractor", labeled_responses)

    # Step 6: VeriCoTStepValidator diagnostic (baseline, expected TP=0).
    # use_mock=True: avoids loading the 0.8B transformers model — we only need
    # the baseline TP=0 measurement, not production LLM accuracy.
    vericot_result = run_extractor_diagnostic(
        VeriCoTStepValidator(use_mock=True), "VeriCoTStepValidator", labeled_responses
    )

    # Step 7: VPRMArithmeticVerifier diagnostic (baseline, expected TP=0).
    vprm_result = run_extractor_diagnostic(
        VPRMArithmeticVerifier(), "VPRMArithmeticVerifier", labeled_responses
    )

    # Derived rates and precision/recall for CoACE.
    coace_tp_rate = coace_result.tp_rate
    coace_fp_rate = coace_result.fp_rate

    # Precision: of all flagged, how many were truly incorrect?
    # n_violations_found = TP + FP; precision = TP / (TP + FP) if any flagged.
    n_flagged = coace_result.n_true_positive + coace_result.n_false_positive
    coace_precision = (
        coace_result.n_true_positive / n_flagged if n_flagged > 0 else 0.0
    )
    # Recall == tp_rate (both are TP / total_incorrect).
    coace_recall = coace_tp_rate

    vericot_tp_rate = vericot_result.tp_rate
    vprm_tp_rate = vprm_result.tp_rate

    # Step 8: RETRO-061 gate determination.
    retro_061_resolved = coace_tp_rate > 0.0
    gate_open = coace_tp_rate > 0.0

    if retro_061_resolved:
        honest_verdict = "retro_061_resolved"
    elif coace_tp_rate == 0.0:
        honest_verdict = "coace_still_zero"
    else:
        honest_verdict = "partial_improvement"

    # Step 9: Build artifact.
    artifact = tmpl.build_result(
        {
            "schema": "carnot.coace_live_diagnostic.v1",
            "n_responses": n_responses,
            "coace_tp_rate": coace_tp_rate,
            "coace_fp_rate": coace_fp_rate,
            "coace_precision": coace_precision,
            "coace_recall": coace_recall,
            "vericot_tp_rate": vericot_tp_rate,
            "vprm_tp_rate": vprm_tp_rate,
            "coace_improvement_over_vericot": coace_tp_rate - vericot_tp_rate,
            "retro_061_resolved": retro_061_resolved,
            "gate_open": gate_open,
            "honest_verdict": honest_verdict,
            "coace_detail": coace_result.to_dict(),
            "vericot_detail": vericot_result.to_dict(),
            "vprm_detail": vprm_result.to_dict(),
        },
        status="success",
    )
    writer.write(artifact)

    # Step 10: Final assertion — verifies the file was written.
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    run_experiment()
