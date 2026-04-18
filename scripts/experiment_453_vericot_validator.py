#!/usr/bin/env python3
"""Experiment 453: VeriCoT Step Validator — IT Model Detection Improvement.

**Researcher summary:**

    The ArithmeticExtractor (regex-based, Exp 47) finds zero violations on
    instruction-tuned (IT) model outputs because IT models write arithmetic in prose:

        "the total is 47 plus 28, which gives 75"

    instead of the equation style the regex expects:

        "47 + 28 = 75"

    VeriCoT (arXiv 2511.04662) fixes this by formalizing each CoT step into
    First-Order Logic (FOL) premises via an LLM call, then checking Z3 for
    consistency.  Published improvement: 46% relative pass rate improvement.

    This experiment uses use_mock=True (rule-based FOL extraction, no GPU) to
    validate the detection logic on 20 hardcoded IT-style samples.

**Design:**
    - 10 CORRECT samples: IT-style prose, arithmetic is right.
    - 10 WRONG samples: IT-style prose, arithmetic is wrong.
    - ArithmeticExtractor baseline: expected 0 detections (regex mismatch).
    - VeriCoTStepValidator(use_mock=True): expected ≥1 detection.

    This is CPU-only.  No GPU required.  Timeout: 30 minutes.

Outputs: results/experiment_453_vericot_validator.json

Spec: REQ-EXTRACT-024, REQ-EXTRACT-025, REQ-EXTRACT-026,
      SCENARIO-EXTRACT-049, SCENARIO-EXTRACT-050, SCENARIO-EXTRACT-051
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() FIRST — belt-and-suspenders RETRO-022 fix.
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix

_autofix = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------
import json

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402

from carnot.extraction import VeriCoTStepValidator  # noqa: E402
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.extract import ArithmeticExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 453
TITLE = "VeriCoT Step Validator — IT Model Detection Improvement"
RESULT_PATH = "results/experiment_453_vericot_validator.json"
TIMEOUT_MINUTES = 30

# ---------------------------------------------------------------------------
# IT-style reasoning samples (hardcoded, no GPU/API call needed)
# ---------------------------------------------------------------------------
# These mimic how Gemma4-E4B-it and Qwen3.5-0.8B write arithmetic answers.
# Format: natural prose, no "A + B = C" equations.

CORRECT_SAMPLES = [
    # [0] Basic addition
    "First, I add 47 plus 28, which gives 75.",
    # [1] Basic subtraction
    "We start with 100 and subtract 35, which gives us 65.",
    # [2] Multiplication
    "5 times 12 gives us 60.",
    # [3] Subtraction via 'from'
    "Subtracting 15 from 80 gives 65.",
    # [4] Addition with phrase variant
    "The sum of 120 plus 45 gives 165.",
    # [5] Multi-step (each step correct)
    "First, 30 plus 20 gives 50. Then, 50 minus 10 gives 40.",
    # [6] Multiplied-by phrasing
    "8 multiplied by 9 gives 72.",
    # [7] Divided-by phrasing
    "100 divided by 4 gives 25.",
    # [8] Added-to phrasing
    "17 added to 33 gives 50.",
    # [9] Large numbers
    "Adding 500 plus 250 gives 750.",
]

WRONG_SAMPLES = [
    # [0] Off-by-one addition error
    "First, I add 47 plus 28, which gives 76.",
    # [1] Wrong subtraction
    "We start with 100 and subtract 35, which gives us 66.",
    # [2] Wrong multiplication
    "5 times 12 gives us 61.",
    # [3] Wrong subtraction (from)
    "Subtracting 15 from 80 gives 66.",
    # [4] Wrong addition
    "The sum of 120 plus 45 gives 166.",
    # [5] Wrong multiplication (multiplied by)
    "8 multiplied by 9 gives 73.",
    # [6] Wrong division
    "100 divided by 4 gives 26.",
    # [7] Wrong added-to
    "17 added to 33 gives 51.",
    # [8] Large number wrong
    "Adding 500 plus 250 gives 751.",
    # [9] Wrong minus
    "200 minus 50 gives 151.",
]

ALL_SAMPLES = CORRECT_SAMPLES + WRONG_SAMPLES  # 20 total


def run_arithmetic_baseline(samples: list[str]) -> int:
    """Count violations detected by ArithmeticExtractor across all samples.

    Expected: 0.  IT prose has no "A + B = C" equations for the regex to match.
    This confirms the baseline problem that VeriCoT is designed to fix.
    """
    extractor = ArithmeticExtractor()
    total = 0
    for i, sample in enumerate(samples):
        violations = [r for r in extractor.extract(sample) if not r.metadata.get("satisfied", True)]
        _log.info("ArithmeticExtractor sample %d: %d violations", i, len(violations))
        total += len(violations)
    return total


def run_vericot_validator(samples: list[str]) -> int:
    """Count violations detected by VeriCoTStepValidator (mock mode) across all samples.

    Expected: ≥1 (specifically, should detect all 10 wrong samples).
    use_mock=True uses rule-based FOL extraction — no GPU or LLM call required.
    """
    validator = VeriCoTStepValidator(use_mock=True)
    total = 0
    for i, sample in enumerate(samples):
        violations = validator.detect_violations(sample)
        _log.info("VeriCoT sample %d: %d violations (steps: %s)", i, len(violations),
                  [v.status for v in violations])
        total += len(violations)
    return total


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=RESULT_PATH,
    )
    tmpl.setup()

    output_path = _REPO_ROOT / RESULT_PATH

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES):
        _log.info("=== Exp %d: %s ===", EXP_ID, TITLE)
        _log.info("Samples: %d total (%d correct, %d wrong)",
                  len(ALL_SAMPLES), len(CORRECT_SAMPLES), len(WRONG_SAMPLES))

        # --- Baseline: ArithmeticExtractor ---
        _log.info("--- Running ArithmeticExtractor baseline ---")
        baseline_detected = run_arithmetic_baseline(ALL_SAMPLES)
        _log.info("ArithmeticExtractor detected: %d violations", baseline_detected)

        # --- VeriCoT: VeriCoTStepValidator(use_mock=True) ---
        _log.info("--- Running VeriCoTStepValidator (mock mode) ---")
        vericot_detected = run_vericot_validator(ALL_SAMPLES)
        _log.info("VeriCoTStepValidator detected: %d violations", vericot_detected)

        # --- Build artifact ---
        n_total = len(ALL_SAMPLES)
        improvement_rate = (vericot_detected / n_total) - (baseline_detected / n_total)

        if improvement_rate > 0:
            honest_verdict = "vericot_better"
        else:
            honest_verdict = "no_improvement"

        artifact = tmpl.build_result({
            "schema": "carnot.vericot_validator.v1",
            "n_samples": n_total,
            "n_correct_samples": len(CORRECT_SAMPLES),
            "n_wrong_samples": len(WRONG_SAMPLES),
            "baseline_extractor": "ArithmeticExtractor",
            "vericot_extractor": "VeriCoTStepValidator(use_mock=True)",
            "baseline_detected": baseline_detected,
            "vericot_detected": vericot_detected,
            "improvement_rate": improvement_rate,
            "honest_verdict": honest_verdict,
            "use_mock": True,
            "requires_gpu": False,
        }, status="success")

        writer = AtomicResultWriter(str(output_path))
        writer.write(artifact)
        writer.verify_exists()

        _log.info("=== RESULT ===")
        _log.info("  baseline_detected  : %d / %d", baseline_detected, n_total)
        _log.info("  vericot_detected   : %d / %d", vericot_detected, n_total)
        _log.info("  improvement_rate   : %.3f", improvement_rate)
        _log.info("  honest_verdict     : %s", honest_verdict)
        _log.info("Written: %s", output_path)


if __name__ == "__main__":
    main()
