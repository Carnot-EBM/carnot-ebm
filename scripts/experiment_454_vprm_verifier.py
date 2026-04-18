#!/usr/bin/env python3
"""Experiment 454: VPRM Arithmetic Verifier — rule-based vs ArithmeticExtractor on IT prose.

Compares two approaches for detecting arithmetic errors in instruction-tuned (IT) model
chain-of-thought reasoning:

  1. ArithmeticExtractor (baseline) — regex-based, requires 'a OP b = c' equation style.
     IT models write prose, not equations, so this baseline typically gets F1 = 0.

  2. VPRMArithmeticVerifier (VPRM) — deterministic rule-based (arXiv 2601.17223).
     Matches IT-style prose patterns ('47 plus 28 equals 75') without any LLM call.
     Published improvement: 20% F1 gain over neural process reward models.

Why VPRM over a neural judge?
    Neural judges can be reward-hacked — an adversarial input can fool the judge into
    rating a wrong arithmetic step as correct.  VPRM rules are deterministic identity
    checks: impossible to fool without avoiding the recognized patterns entirely.

Why complement VeriCoT (Exp 453)?
    VeriCoT catches logical inconsistency across steps (FOL + Z3, needs an LLM call).
    VPRM catches arithmetic errors within individual steps (pure rules, no LLM).
    Running VPRM first is the fast path; VeriCoT provides deeper multi-step checking.

Protocol:
    1. apply_env_autofix() FIRST — injects CARNOT_FORCE_LIVE=1 if GPU present.
    2. ExperimentTimeoutWatchdog(454, timeout_minutes=20) — hard wall-clock cap.
    3. ExperimentTemplate(454, ...) scaffolding — output dirs, checkpoint, artifact.
    4. 20 hardcoded IT-model-style CoT steps (10 correct, 10 with arithmetic errors).
    5. Run ArithmeticExtractor on all 20 → baseline_f1.
    6. Run VPRMArithmeticVerifier on all 20 → vprm_f1.
    7. Build and write artifact with schema='carnot.vprm_verifier.v1'.

CPU-only: no GPU required (pure Python rule checks, no model loading).

Depends on: Exp 453 VeriCoT infrastructure (shares extraction/__init__.py).

Spec: REQ-EXTRACT-027, REQ-EXTRACT-028, REQ-EXTRACT-029,
      SCENARIO-EXTRACT-052, SCENARIO-EXTRACT-053, SCENARIO-EXTRACT-054
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# apply_env_autofix MUST be called before any JAX or CUDA import.
from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.extraction import VPRMArithmeticVerifier
from carnot.pipeline.extract import ArithmeticExtractor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test corpus: 20 IT-model-style CoT steps
# ---------------------------------------------------------------------------

# 10 steps with ARITHMETIC ERRORS (ground_truth = True, violation present)
# Written in IT-model prose (no 'a + b = c' style), so ArithmeticExtractor misses them.
_WRONG_SAMPLES: list[str] = [
    "we have 47 plus 28 equals 76, so the total is 76",          # correct: 75
    "5 times 6 gives us 31, so the product is 31",              # correct: 30
    "20% of 50 is 11, so the discount is 11",                    # correct: 10
    "100 minus 15 gives 90, so the remainder is 90",             # correct: 85
    "100 divided by 4 gives 26, so each share is 26",           # correct: 25
    "3 times 9 equals 28, so the result is 28",                 # correct: 27
    "15% of 200 equals 31, so the amount is 31",                # correct: 30
    "50 plus 25 gives 74, so the sum is 74",                    # correct: 75
    "subtracting 10 from 100 gives 89, so 89 remain",          # correct: 90
    "7 multiplied by 8 equals 57, so the answer is 57",        # correct: 56
]

# 10 steps that are ARITHMETICALLY CORRECT (ground_truth = False, no violation)
_CORRECT_SAMPLES: list[str] = [
    "we have 47 plus 28 equals 75, so the total is 75",
    "5 times 6 gives us 30, so the product is 30",
    "20% of 50 is 10, so the discount is 10",
    "100 minus 15 gives 85, so the remainder is 85",
    "100 divided by 4 gives 25, so each share is 25",
    "3 times 9 equals 27, so the result is 27",
    "15% of 200 equals 30, so the amount is 30",
    "50 plus 25 gives 75, so the sum is 75",
    "subtracting 10 from 100 gives 90, so 90 remain",
    "7 multiplied by 8 equals 56, so the answer is 56",
]

_ALL_SAMPLES: list[str] = _WRONG_SAMPLES + _CORRECT_SAMPLES
# True = has arithmetic error (positive class in F1 calculation)
_GROUND_TRUTH: list[bool] = [True] * 10 + [False] * 10


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def _run_experiment(tmpl: ExperimentTemplate) -> dict:
    """Run ArithmeticExtractor and VPRMArithmeticVerifier on all 20 samples.

    Returns the raw results dict (before ExperimentTemplate wraps it in an artifact).
    """
    extractor = ArithmeticExtractor()
    verifier = VPRMArithmeticVerifier()

    # --- ArithmeticExtractor baseline ---
    # ArithmeticExtractor looks for 'a + b = c' equation syntax.  IT-model prose
    # uses 'plus', 'equals', 'gives us' — no equation-style syntax.  Expected
    # result: 0 violations found → baseline_f1 = 0.0.
    baseline_predicted: list[bool] = []
    for sample in _ALL_SAMPLES:
        results = extractor.extract(sample)
        # A violation = any result with satisfied=False in metadata
        has_violation = any(
            r.metadata.get("satisfied") is False for r in results
        )
        baseline_predicted.append(has_violation)

    baseline_f1 = VPRMArithmeticVerifier.f1_score(_GROUND_TRUTH, baseline_predicted)

    # --- VPRM verifier ---
    # VPRMArithmeticVerifier uses rule-based prose patterns.  Expected result:
    # detects errors in wrong samples → higher F1 than baseline.
    vprm_predicted: list[bool] = []
    vprm_detail: list[dict] = []
    for sample in _ALL_SAMPLES:
        violations = verifier.detect_violations(sample)
        has_violation = len(violations) > 0
        vprm_predicted.append(has_violation)
        vprm_detail.append({
            "sample": sample,
            "violation_detected": has_violation,
            "violations": [
                {
                    "rule_name": v.rule_name,
                    "passed": v.passed,
                    "computed_value": v.computed_value,
                    "stated_value": v.stated_value,
                    "error_magnitude": v.error_magnitude,
                }
                for v in violations
            ],
        })

    vprm_f1 = VPRMArithmeticVerifier.f1_score(_GROUND_TRUTH, vprm_predicted)

    f1_improvement = vprm_f1 - baseline_f1
    honest_verdict = "vprm_better" if f1_improvement > 0 else "no_improvement"

    # True positives: VPRM correctly detected violations
    tp = sum(g and p for g, p in zip(_GROUND_TRUTH, vprm_predicted))
    fp = sum((not g) and p for g, p in zip(_GROUND_TRUTH, vprm_predicted))
    fn = sum(g and (not p) for g, p in zip(_GROUND_TRUTH, vprm_predicted))

    _log.info(
        "Results: baseline_f1=%.3f  vprm_f1=%.3f  improvement=%.3f  verdict=%s",
        baseline_f1, vprm_f1, f1_improvement, honest_verdict,
    )
    _log.info("VPRM: TP=%d  FP=%d  FN=%d", tp, fp, fn)

    return {
        "schema": "carnot.vprm_verifier.v1",
        "baseline_f1": baseline_f1,
        "vprm_f1": vprm_f1,
        "f1_improvement": f1_improvement,
        "honest_verdict": honest_verdict,
        "n_samples": len(_ALL_SAMPLES),
        "n_wrong": 10,
        "n_correct": 10,
        "vprm_tp": tp,
        "vprm_fp": fp,
        "vprm_fn": fn,
        "baseline_predicted": baseline_predicted,
        "vprm_predicted": vprm_predicted,
        "ground_truth": _GROUND_TRUTH,
        "vprm_detail": vprm_detail,
    }


def main() -> None:
    """Run Experiment 454."""
    tmpl = ExperimentTemplate(
        454,
        "VPRM Arithmetic Verifier",
        "results/experiment_454_vprm_verifier.json",
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(
        experiment_id=454,
        timeout_minutes=20,
        result_path="results/experiment_454_vprm_verifier.json",
    ):
        try:
            results = _run_experiment(tmpl)
        except Exception as exc:
            _log.error("Experiment 454 failed: %s", exc, exc_info=True)
            artifact = tmpl.build_result(
                {"error": str(exc), "schema": "carnot.vprm_verifier.v1"},
                status="error",
            )
            output_path = Path(__file__).resolve().parents[1] / "results" / "experiment_454_vprm_verifier.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(artifact, indent=2))
            sys.exit(1)

        artifact = tmpl.build_result(results, status="success")

    output_path = Path(__file__).resolve().parents[1] / "results" / "experiment_454_vprm_verifier.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))

    _log.info("Wrote artifact to %s", output_path)
    _log.info(
        "honest_verdict=%s  baseline_f1=%.3f  vprm_f1=%.3f",
        results["honest_verdict"],
        results["baseline_f1"],
        results["vprm_f1"],
    )


if __name__ == "__main__":
    main()
