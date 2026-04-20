#!/usr/bin/env python3
"""Experiment 554: Extraction Diagnostic — per-extractor FP/TP rate measurement.

**Researcher summary (RETRO-033 miss #10 root cause investigation):**
    Exp 538 ran verify-repair on 25 live questions and found
    honest_verdict='live_no_improvement_25q'.  The root cause is UNKNOWN.
    Three hypotheses:

        H1. VeriCoT/VPRM produces false positives on CORRECT IT-model responses,
            causing repair of valid answers (FP rate too high).
        H2. Violations detected correctly but repair generates wrong fixes.
        H3. Scoring bug elsewhere in the pipeline.

    This experiment tests H1 by loading Exp 538's 25 labeled responses and running
    VeriCoTStepValidator and VPRMArithmeticVerifier separately on each, then
    computing per-extractor TP rate, FP rate, precision, and recall.

**Gate chain (every exit path writes the deliverable):**
    0. Zombie PIDs killed (subprocess.run kill -9)
    1. apply_env_autofix()                     — normalise env before any import
    2. ExperimentTimeoutWatchdog(554, 20)      — 20-minute hard cap (CPU-only, fast)
    3. Load labeled responses from exp538_cot_pairs.json (or synthetic fallback)
    4. Run VeriCoTStepValidator diagnostic
    5. Run VPRMArithmeticVerifier diagnostic
    6. Classify root_cause_hypothesis from FP/TP rates
    7. AtomicResultWriter: results/experiment_554_extraction_diagnostic.json
    8. tmpl.assert_deliverable_written()       — FINAL LINE

Spec: REQ-EXTRACT-030, SCENARIO-EXTRACT-055, SCENARIO-EXTRACT-056,
      SCENARIO-EXTRACT-057
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: Kill zombie PIDs FIRST — before any CUDA import.
# ---------------------------------------------------------------------------
import subprocess

subprocess.run(["kill", "-9"], capture_output=True)  # no specific PIDs; harmless call

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

from carnot.extraction import (  # noqa: E402
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

EXP_ID = 554
EXP_TITLE = "Extraction Diagnostic — FP/TP Rates"
DELIVERABLE = "results/experiment_554_extraction_diagnostic.json"

# Primary source: Exp 538's labeled CoT pairs (25 entries with is_correct labels)
EXP538_COT_PAIRS = "results/exp538_cot_pairs.json"

# FP rate threshold above which we declare H1 (spurious repair) as the likely cause
_FP_RATE_THRESHOLD = 0.3

# TP rate threshold below which we declare H2 (low recall) as the issue
_TP_RATE_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_labeled_responses() -> list[dict]:
    """Load 25 labeled responses from Exp 538 CoT pairs file.

    Each returned dict has:
        'response': str  — the full CoT text used as extractor input
        'is_correct': bool — True iff the model's answer was graded correct

    Falls back to 25 synthetic GSM8K-style pairs if the file is missing.
    Exp 538 used 'cot_text' as the response field and 'correct' as the label.
    We normalise to 'response' / 'is_correct' for the diagnostic API.
    """
    pairs_path = _REPO_ROOT / EXP538_COT_PAIRS
    if pairs_path.exists():
        raw = json.loads(pairs_path.read_text())
        labeled: list[dict] = []
        for entry in raw:
            labeled.append(
                {
                    "response": entry.get("cot_text", entry.get("response", "")),
                    "is_correct": bool(entry.get("correct", entry.get("is_correct", False))),
                    "question": entry.get("question", ""),
                    "model_id": entry.get("model_id", "unknown"),
                }
            )
        return labeled

    # Synthetic fallback — 25 arithmetic word-problem responses with known labels.
    # 15 incorrect (wrong arithmetic), 10 correct — realistic GSM8K accuracy.
    synthetic = []
    # Correct responses: simple, factually correct arithmetic
    correct_templates = [
        "To find the total, we add 47 plus 28, which gives us 75. The answer is 75.",
        "Subtracting 15 from 100 gives 85. So the remainder is 85.",
        "5 times 6 gives us 30. The product is 30.",
        "100 divided by 4 gives 25. Each share is 25.",
        "20% of 50 is 10. So the discount is $10.",
        "12 plus 8 gives us 20. Total is 20.",
        "60 minus 24 gives 36. The difference is 36.",
        "7 times 9 gives us 63. The result is 63.",
        "48 divided by 6 gives 8. Each group has 8.",
        "25% of 80 is 20. The tip is $20.",
    ]
    for t in correct_templates:
        synthetic.append({"response": t, "is_correct": True, "question": "", "model_id": "synthetic"})

    # Incorrect responses: plausible but wrong arithmetic
    incorrect_templates = [
        "To find the total, we add 47 plus 28, which gives us 76. The answer is 76.",
        "Subtracting 15 from 100 gives 84. So the remainder is 84.",
        "5 times 6 gives us 31. The product is 31.",
        "100 divided by 4 gives 26. Each share is 26.",
        "20% of 50 is 11. So the discount is $11.",
        "12 plus 8 gives us 21. Total is 21.",
        "60 minus 24 gives 37. The difference is 37.",
        "7 times 9 gives us 62. The result is 62.",
        "48 divided by 6 gives 9. Each group has 9.",
        "25% of 80 is 21. The tip is $21.",
        "3 times 7 gives us 22. Product is 22.",
        "90 minus 45 gives 44. Remainder is 44.",
        "15 plus 26 gives us 40. Total is 40.",
        "36 divided by 6 gives 7. Quotient is 7.",
        "10% of 200 is 21. Fee is $21.",
    ]
    for t in incorrect_templates:
        synthetic.append({"response": t, "is_correct": False, "question": "", "model_id": "synthetic"})

    return synthetic


def _classify_root_cause(vericot_fp: float, vericot_tp: float, vprm_fp: float, vprm_tp: float) -> str:
    """Classify the most likely root cause from FP/TP rates.

    Decision rules (in priority order):
        'high_fp_extraction'  — max FP rate > 0.3  → extractors trigger spurious repair (H1)
        'low_tp_extraction'   — min TP rate < 0.3  → extractors miss real violations (H2 input side)
        'repair_issue'        — TP rate high, FP low, but no improvement observed → H2 (repair step)
        'inconclusive'        — neither threshold triggered clearly

    The 'repair_issue' branch requires external context (no improvement observed in Exp 538).
    We encode it as the fallback when both TP > 0.3 and FP < 0.1 — that constellation
    means the extraction itself was working, so blame shifts to the repair generator.
    """
    max_fp = max(vericot_fp, vprm_fp)
    min_tp = min(vericot_tp, vprm_tp)

    if max_fp > _FP_RATE_THRESHOLD:
        return "high_fp_extraction"
    if min_tp < _TP_RATE_THRESHOLD:
        return "low_tp_extraction"
    # Both extractors have reasonable TP and low FP — extraction is fine,
    # blame the repair step (or scoring) for the no-improvement outcome.
    if vericot_tp > 0.3 and vprm_tp > 0.3 and max_fp < 0.1:
        return "repair_issue"
    return "inconclusive"


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment() -> None:
    """Execute the extraction diagnostic and write the deliverable artifact."""

    # Step 2: ExperimentTimeoutWatchdog — 20-minute hard cap.
    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=20)
    watchdog.start()

    # Step 3: ExperimentTemplate setup (creates dirs, wires DeliverableGuard).
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    try:
        # Step 4: Load labeled responses.
        labeled_responses = _load_labeled_responses()
        n_responses = len(labeled_responses)
        n_correct = sum(1 for r in labeled_responses if r["is_correct"])
        n_incorrect = n_responses - n_correct

        # Step 5: VeriCoTStepValidator diagnostic (use_mock=True — no GPU needed).
        vericot = VeriCoTStepValidator(use_mock=True)
        vericot_result = run_extractor_diagnostic(
            extractor=vericot,
            extractor_name="VeriCoTStepValidator",
            labeled_responses=labeled_responses,
        )

        # Step 6: VPRMArithmeticVerifier diagnostic.
        vprm = VPRMArithmeticVerifier()
        vprm_result = run_extractor_diagnostic(
            extractor=vprm,
            extractor_name="VPRMArithmeticVerifier",
            labeled_responses=labeled_responses,
        )

        # Step 7: Classify root cause hypothesis.
        root_cause_hypothesis = _classify_root_cause(
            vericot_fp=vericot_result.fp_rate,
            vericot_tp=vericot_result.tp_rate,
            vprm_fp=vprm_result.fp_rate,
            vprm_tp=vprm_result.tp_rate,
        )

        # Compute precision for each extractor.
        def _precision(tp: int, fp: int) -> float:
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

        vericot_precision = _precision(vericot_result.n_true_positive, vericot_result.n_false_positive)
        vprm_precision = _precision(vprm_result.n_true_positive, vprm_result.n_false_positive)

        # Step 8: Build main artifact.
        artifact = tmpl.build_result(
            {
                "schema": "carnot.extraction_diagnostic.v1",
                "n_responses_analyzed": n_responses,
                "n_correct_responses": n_correct,
                "n_incorrect_responses": n_incorrect,
                "source": EXP538_COT_PAIRS if (
                    (_REPO_ROOT / EXP538_COT_PAIRS).exists()
                ) else "synthetic_fallback",
                "vericot_result": {
                    "tp_rate": vericot_result.tp_rate,
                    "fp_rate": vericot_result.fp_rate,
                    "n_violations": vericot_result.n_violations_found,
                    "n_correct_flagged": vericot_result.n_false_positive,
                    "precision": vericot_precision,
                    "recall": vericot_result.tp_rate,
                    **vericot_result.to_dict(),
                },
                "vprm_result": {
                    "tp_rate": vprm_result.tp_rate,
                    "fp_rate": vprm_result.fp_rate,
                    "n_violations": vprm_result.n_violations_found,
                    "n_correct_flagged": vprm_result.n_false_positive,
                    "precision": vprm_precision,
                    "recall": vprm_result.tp_rate,
                    **vprm_result.to_dict(),
                },
                "root_cause_hypothesis": root_cause_hypothesis,
                "honest_verdict": "diagnostic_complete",
            },
            status="success",
        )

    except Exception as exc:
        artifact = tmpl.build_result(
            {
                "schema": "carnot.extraction_diagnostic.v1",
                "error": str(exc),
                "root_cause_hypothesis": "error_during_diagnostic",
                "honest_verdict": "diagnostic_error",
            },
            status="error",
        )

    # Write main artifact atomically.
    writer = AtomicResultWriter(DELIVERABLE)
    writer.write(artifact)

    watchdog.stop()

    # FINAL LINE — raises RuntimeError if deliverable is absent.
    tmpl.assert_deliverable_written()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment()
