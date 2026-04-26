"""Experiment 874: StreamingCoT Tier 0g integration into VerifyRepairPipeline.

**Purpose:**
    Wire StreamingCoTHalluDetector (from Exp 861) into VerifyRepairPipeline.verify()
    as an advisory Tier 0g signal.  This experiment validates the wiring by running
    25 synthetic CoT questions and measuring advisory_signal_rate and skip_rate.

**Design:**
    - 12 "correct" CoT responses: short, uniform step lengths → low PHaS → is_streaming_unstable=False
    - 13 "error" CoT responses:   compounding-error steps (lengths vary dramatically) →
      high PHaS → is_streaming_unstable=True
    - CARNOT_STREAMING_COT=1 is injected via os.environ at runtime (not requiring shell)
    - VerifyRepairPipeline runs normally (Ising not skipped based on streaming signal)

**honest_verdict logic:**
    - "streaming_cot_wired" if STREAMING_COT_ENABLED and streaming_cot_unstable is
      populated in certificate for > 50% of responses
    - "wired_low_coverage" otherwise

Spec: REQ-VERIFY-140, SCENARIO-VERIFY-165, SCENARIO-VERIFY-166
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone, UTC

# Activate Tier 0g before importing the pipeline, so STREAMING_COT_ENABLED is True.
# WHY here rather than in a shell env: experiments run in-process by the conductor;
# we cannot rely on the shell having exported the variable.
os.environ["CARNOT_STREAMING_COT"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carnot.pipeline.streaming_cot import StreamingCoTHalluDetector, extract_cot_steps
from carnot.pipeline.verify_repair import VerifyRepairPipeline

EXPERIMENT_ID = 874
TITLE = "StreamingCoT Tier 0g Integration into VerifyRepairPipeline"
RESULT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "experiment_874_streaming_cot_integration.json",
)

# ---------------------------------------------------------------------------
# Synthetic CoT fixtures
# ---------------------------------------------------------------------------

# Correct CoT responses: uniform, concise steps.  PHaS EMA stays low.
CORRECT_RESPONSES = [
    "Step 1: Identify the variables.\nStep 2: Apply the formula.\nStep 3: Compute the result.",
    "1. Read the problem carefully.\n2. Set up the equation.\n3. Solve for x.",
    "Step 1: Convert units.\nStep 2: Multiply.\nStep 3: Round to nearest integer.",
    "Step 1: Expand the brackets.\nStep 2: Collect like terms.\nStep 3: Divide both sides.",
    "Step 1: Draw a diagram.\nStep 2: Label the sides.\nStep 3: Apply Pythagoras.",
    "Step 1: Note given information.\nStep 2: Choose a strategy.\nStep 3: Execute and verify.",
    "Step 1: Factor the quadratic.\nStep 2: Find the roots.\nStep 3: Check by substitution.",
    "Step 1: Identify the base case.\nStep 2: State the inductive hypothesis.\nStep 3: Complete inductive step.",
    "Step 1: Rewrite as fraction.\nStep 2: Simplify numerator.\nStep 3: Cancel common factors.",
    "Step 1: Note the symmetry.\nStep 2: Use the identity.\nStep 3: Substitute back.",
    "Step 1: Convert to radians.\nStep 2: Apply the sine rule.\nStep 3: Solve for angle.",
    "Step 1: Set up proportion.\nStep 2: Cross-multiply.\nStep 3: Divide and round.",
]

# Error CoT responses: compounding mistakes, highly variable step lengths,
# spurious detail injected in later steps → PHaS EMA rises and crosses threshold.
_SHORT = "Step 1: Hmm."
_MEDIUM = "Step 2: Let me think about this a bit more carefully and re-read the question."
_LONG = (
    "Step 3: Actually, I need to reconsider entirely because my earlier calculation "
    "overlooked a crucial detail about the boundary conditions which changes the "
    "approach fundamentally and requires a completely different formula."
)
_VERY_LONG = (
    "Step 4: Wait, I think I made an error in step 2 as well. Let me start over from "
    "scratch.  The problem says to find the area, but I was computing the perimeter. "
    "This changes everything.  The area formula for a trapezoid is (a+b)/2 * h, where "
    "a and b are parallel sides.  But I don't know h.  Actually, I do know h from the "
    "Pythagorean theorem applied to the right triangle formed by the slant and the "
    "difference of the parallel sides.  But I need to know which side is longer first. "
    "Let me re-examine the diagram I apparently forgot to draw in step 1."
)

ERROR_RESPONSES = []
for i in range(13):
    if i % 3 == 0:
        resp = "\n".join([_SHORT, _LONG, _VERY_LONG])
    elif i % 3 == 1:
        resp = "\n".join([_MEDIUM, _SHORT, _VERY_LONG, _SHORT])
    else:
        resp = "\n".join([_SHORT, _VERY_LONG, _SHORT, _LONG, _SHORT])
    ERROR_RESPONSES.append(resp)


def run_experiment() -> dict:
    """Run all 25 synthetic questions through VerifyRepairPipeline.verify()."""
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    t0 = time.monotonic()

    # Force the class attribute to True so the flag works even when the module
    # was already imported before os.environ was set (in-process conductor runs).
    VerifyRepairPipeline.STREAMING_COT_ENABLED = True

    pipeline = VerifyRepairPipeline()  # verify-only mode, no LLM model loaded

    n_correct = len(CORRECT_RESPONSES)
    n_error = len(ERROR_RESPONSES)
    all_responses = CORRECT_RESPONSES + ERROR_RESPONSES
    is_error = [False] * n_correct + [True] * n_error

    results = []
    for i, (response, expected_error) in enumerate(zip(all_responses, is_error)):
        result = pipeline.verify(
            question="Solve the problem.",
            response=response,
            domain=None,
        )
        cert = result.certificate.get("tier_0g_streaming_cot", {})
        results.append(
            {
                "idx": i,
                "expected_error": expected_error,
                "streaming_cot_unstable": result.streaming_cot_unstable,
                "streaming_cot_phas": result.streaming_cot_phas,
                "n_steps": cert.get("n_steps", 0),
                "verified": result.verified,
                "skipped": result.skipped,
            }
        )

    # Metrics
    n_total = len(results)
    n_unstable = sum(1 for r in results if r["streaming_cot_unstable"])
    n_skipped = sum(1 for r in results if r["skipped"])
    n_cert_populated = sum(1 for r in results if r["streaming_cot_phas"] is not None)

    # Among those flagged unstable, how many were actually error responses?
    unstable_results = [r for r in results if r["streaming_cot_unstable"]]
    n_unstable_correct_prediction = sum(1 for r in unstable_results if r["expected_error"])

    streaming_cot_advisory_rate = n_unstable / n_total if n_total else 0.0
    skip_rate = n_skipped / n_total if n_total else 0.0
    advisory_correct_prediction_rate = (
        n_unstable_correct_prediction / len(unstable_results) if unstable_results else 0.0
    )

    # Determine honest_verdict:
    # "streaming_cot_wired" requires:
    #   1. STREAMING_COT_ENABLED is True
    #   2. streaming_cot_unstable is populated in certificate for > 50% of responses
    coverage_rate = n_cert_populated / n_total if n_total else 0.0
    if VerifyRepairPipeline.STREAMING_COT_ENABLED and coverage_rate > 0.5:
        honest_verdict = "streaming_cot_wired"
    else:
        honest_verdict = "wired_low_coverage"

    finished_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    duration_s = round(time.monotonic() - t0, 3)

    artifact = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "run_date": datetime.now(UTC).strftime("%Y%m%d"),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "status": "success",
        "honest_verdict": honest_verdict,
        "n_questions": n_total,
        "n_correct_responses": n_correct,
        "n_error_responses": n_error,
        "streaming_cot_advisory_rate": round(streaming_cot_advisory_rate, 4),
        "advisory_correct_prediction_rate": round(advisory_correct_prediction_rate, 4),
        "skip_rate": skip_rate,
        "streaming_cot_enabled": VerifyRepairPipeline.STREAMING_COT_ENABLED,
        "per_response_results": results,
        "schema": sorted(
            [
                "experiment",
                "title",
                "run_date",
                "started_at",
                "finished_at",
                "duration_s",
                "status",
                "honest_verdict",
                "n_questions",
                "n_correct_responses",
                "n_error_responses",
                "streaming_cot_advisory_rate",
                "advisory_correct_prediction_rate",
                "skip_rate",
                "streaming_cot_enabled",
                "per_response_results",
            ]
        ),
        "invariant_violations": [],
    }
    return artifact


def main() -> None:
    artifact = run_experiment()
    out_path = os.path.abspath(RESULT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Experiment {EXPERIMENT_ID} complete: {artifact['honest_verdict']}")
    print(f"  streaming_cot_advisory_rate: {artifact['streaming_cot_advisory_rate']}")
    print(f"  advisory_correct_prediction_rate: {artifact['advisory_correct_prediction_rate']}")
    print(f"  skip_rate: {artifact['skip_rate']}")
    print(f"  Result written to: {out_path}")
    # tmpl.assert_deliverable_written() equivalent — check file exists and has required fields.
    with open(out_path) as f:
        loaded = json.load(f)
    required = ["experiment", "honest_verdict", "status", "streaming_cot_advisory_rate"]
    missing = [k for k in required if k not in loaded]
    if missing:
        raise RuntimeError(f"Deliverable missing required fields: {missing}")
    print("Deliverable validated successfully.")


if __name__ == "__main__":
    main()
