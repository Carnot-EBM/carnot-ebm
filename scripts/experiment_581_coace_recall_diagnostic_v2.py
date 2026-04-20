#!/usr/bin/env python3
"""Experiment 581: CoACE Recall Diagnostic V2 — RETRO-064 Validation Gate.

**Researcher summary:**
    Exp 565 measured CoACEExtractor v1 recall at 5.9% on 25 known-incorrect live
    IT-model responses (1 TP out of 17 incorrect responses).  RETRO-064 identified
    three missing pattern families in v1: prose percentage/ratio patterns, multi-step
    chain tracking, and chained equality.  Exp 576 implemented CoACEExtractorV2 to
    cover all three.

    This experiment reruns the same 25 labeled responses through CoACEExtractorV2 and
    measures v2_recall.  The gate rule:
        - v2_recall >= 0.20 → gate_open=True, Exps 582+583 unblocked.
        - v2_recall < 0.20  → gate_open=False, Exps 582+583 remain blocked.
        - v2_recall >= 0.30 → retro_064_resolved=True (strong improvement).

**Gate chain (every exit path writes the deliverable):**
    0. apply_env_autofix()                        — normalise env before any import
    1. ExperimentTimeoutWatchdog(581, 20)         — 20-minute hard cap (CPU-only)
    2. ExperimentTemplate(581, ..., requires_gpu=False)
    3. Load labeled responses from exp565 per_question_results, fallback exp554,
       final fallback exp538_cot_pairs.json
    4. If no labeled responses: write blocked artifact (honest_verdict='upstream_missing')
    5. Run run_extractor_diagnostic(CoACEExtractorV2(), ...) → v2 metrics
    6. Run run_extractor_diagnostic(CoACEExtractor(), ...) → v1 baseline
    7. Compute recall_improvement, gate_open, retro_064 flags
    8. Write results/experiment_581_coace_recall_diagnostic_v2.json
    9. tmpl.assert_deliverable_written()            — FINAL LINE

Spec: REQ-EXTRACT-037, SCENARIO-EXTRACT-072, SCENARIO-EXTRACT-073, SCENARIO-EXTRACT-074
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: apply_env_autofix() FIRST — before any CUDA import.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------
import json  # noqa: E402
from typing import Any  # noqa: E402

from carnot.extraction.coace_extractor import CoACEExtractor  # noqa: E402
from carnot.extraction.coace_extractor_v2 import CoACEExtractorV2  # noqa: E402
from carnot.extraction.extraction_diagnostic import run_extractor_diagnostic  # noqa: E402
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 581
EXP_TITLE = "CoACE Recall Diagnostic V2"
DELIVERABLE = "results/experiment_581_coace_recall_diagnostic_v2.json"

# Primary: Exp 565 per_question_results
EXP565_PATH = "results/experiment_565_coace_live_diagnostic.json"

# First fallback: Exp 554 extraction diagnostic
EXP554_PATH = "results/experiment_554_extraction_diagnostic.json"

# Final fallback: raw Exp 538 CoT pairs (the ground truth used by Exp 565)
EXP538_COT_PAIRS = "results/exp538_cot_pairs.json"


# ---------------------------------------------------------------------------
# Response loader — same fallback chain as Exp 565
# ---------------------------------------------------------------------------


def load_labeled_responses() -> list[dict[str, Any]]:
    """Load 25 labeled IT-model responses from available upstream sources.

    Why this fallback chain exists:
        The Exp 565 artifact stores aggregate metrics but NOT the raw response texts
        in per_question_results (that field was empty at write time).  Exp 554 has the
        same gap.  The raw ground truth with cot_text + correct label lives in
        exp538_cot_pairs.json — the same file Exp 565 ultimately loaded from.

    Priority order:
        1. exp565 per_question_results (has response + is_correct if populated)
        2. exp554 per_question_results (same gap, but checked for completeness)
        3. exp538_cot_pairs.json (canonical ground truth, always tried last)

    Returns an empty list if no source yields labeled responses with text.

    Spec: REQ-EXTRACT-037-1
    """

    def _normalise(entry: dict[str, Any]) -> dict[str, Any]:
        """Normalise any upstream key variation to {response, is_correct, ...}."""
        return {
            "response": entry.get("response", entry.get("cot_text", "")),
            "is_correct": bool(entry.get("is_correct", entry.get("correct", False))),
            "question": entry.get("question", ""),
            "model_id": entry.get("model_id", "unknown"),
        }

    # Attempt 1: exp565 per_question_results
    exp565_path = _REPO_ROOT / EXP565_PATH
    if exp565_path.exists():
        data = json.loads(exp565_path.read_text())
        pqr = data.get("per_question_results")
        if pqr and isinstance(pqr, list) and len(pqr) > 0:
            labeled = [_normalise(e) for e in pqr if e.get("response") or e.get("cot_text")]
            if labeled:
                return labeled

    # Attempt 2: exp554 per_question_results
    exp554_path = _REPO_ROOT / EXP554_PATH
    if exp554_path.exists():
        data = json.loads(exp554_path.read_text())
        pqr = data.get("per_question_results")
        if pqr and isinstance(pqr, list) and len(pqr) > 0:
            labeled = [_normalise(e) for e in pqr if e.get("response") or e.get("cot_text")]
            if labeled:
                return labeled

    # Attempt 3: exp538_cot_pairs.json (canonical raw responses)
    exp538_path = _REPO_ROOT / EXP538_COT_PAIRS
    if exp538_path.exists():
        raw = json.loads(exp538_path.read_text())
        if isinstance(raw, list) and len(raw) > 0:
            labeled = [_normalise(e) for e in raw if e.get("cot_text") or e.get("response")]
            if labeled:
                return labeled

    return []


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment() -> None:
    """Run CoACEExtractorV2 on 25 labeled responses, measure recall, determine gate.

    All exit paths write the deliverable before returning.

    Spec: REQ-EXTRACT-037, SCENARIO-EXTRACT-072, SCENARIO-EXTRACT-073, SCENARIO-EXTRACT-074
    """
    # Step 1: hard timeout — CPU-only work should complete in minutes.
    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=20)
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

    # Step 4: Blocked artifact if no labeled responses.
    if not labeled_responses:
        artifact = tmpl.build_result(
            {
                "schema": "carnot.coace_recall_diag.v2",
                "n_responses": 0,
                "v1_recall": 0.0,
                "v2_recall": 0.0,
                "recall_improvement": 0.0,
                "v2_tp_rate": 0.0,
                "v2_fp_rate": 0.0,
                "v2_precision": 0.0,
                "retro_064_partial": False,
                "retro_064_resolved": False,
                "gate_open": False,
                "honest_verdict": "upstream_missing",
            },
            status="blocked",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    n_responses = len(labeled_responses)

    # Step 5: CoACEExtractorV2 diagnostic (v2 recall measurement).
    # CoACEExtractorV2.detect_violations() satisfies ViolationExtractor protocol directly.
    v2_result = run_extractor_diagnostic(
        CoACEExtractorV2(),
        "CoACEExtractorV2",
        labeled_responses,
    )

    # Step 6: CoACEExtractor v1 baseline (for recall_improvement comparison).
    # v1 uses an adapter pattern: wrap extract() → detect_violations().
    class _V1Adapter:
        """Thin adapter: exposes CoACEExtractor.extract() as detect_violations()."""

        def __init__(self) -> None:
            self._ext = CoACEExtractor()

        def detect_violations(self, text: str) -> list[Any]:
            return self._ext.extract(text).violations

    v1_result = run_extractor_diagnostic(
        _V1Adapter(),
        "CoACEExtractor",
        labeled_responses,
    )

    # Step 7: Compute gate metrics.
    v2_recall = v2_result.tp_rate
    v1_recall = v1_result.tp_rate
    recall_improvement = v2_recall - v1_recall

    v2_fp_rate = v2_result.fp_rate
    n_flagged = v2_result.n_true_positive + v2_result.n_false_positive
    v2_precision = (
        v2_result.n_true_positive / n_flagged if n_flagged > 0 else 0.0
    )

    retro_064_partial = v2_recall >= 0.20
    retro_064_resolved = v2_recall >= 0.30
    gate_open = v2_recall >= 0.20

    if retro_064_resolved:
        honest_verdict = "gate_open_recall_resolved"
    elif gate_open:
        honest_verdict = "gate_open_partial"
    else:
        honest_verdict = "gate_closed_still_too_low"

    # Step 8: Write artifact.
    artifact = tmpl.build_result(
        {
            "schema": "carnot.coace_recall_diag.v2",
            "n_responses": n_responses,
            "v1_recall": v1_recall,
            "v2_recall": v2_recall,
            "recall_improvement": recall_improvement,
            "v2_tp_rate": v2_recall,
            "v2_fp_rate": v2_fp_rate,
            "v2_precision": v2_precision,
            "retro_064_partial": retro_064_partial,
            "retro_064_resolved": retro_064_resolved,
            "gate_open": gate_open,
            "honest_verdict": honest_verdict,
            "v2_detail": v2_result.to_dict(),
            "v1_detail": v1_result.to_dict(),
        },
        status="success",
    )
    writer.write(artifact)

    # Step 9: FINAL LINE — asserts deliverable was written.
    tmpl.assert_deliverable_written()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment()
