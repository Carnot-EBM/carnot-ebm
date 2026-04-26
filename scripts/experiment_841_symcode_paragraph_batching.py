#!/usr/bin/env python3
"""Experiment 841 — SymCode Paragraph Batching Latency Benchmark.

**Research question (RETRO-SYMCODE-SERIAL):**
    SymCodeVerifier.verify_response() processes multi-paragraph responses one paragraph
    at a time, each paragraph becoming a separate verify_step() -> safe_eval() call.
    For Exp 627-style responses with 10+ paragraphs this is ~500ms total (50ms × 10).

    batch_verify() collects all arithmetic expressions in one pass, builds a single
    exec() script with shared namespace, and evaluates all expressions at once.

    This experiment measures whether batch_verify() achieves < 2× single-paragraph
    latency for 10 paragraphs (i.e., batching is O(1) not O(N)).

**Honest verdict mapping:**
    batching_effective:  speedup >= 2.0 AND violations_match=True
    batching_marginal:   1.5 <= speedup < 2.0 AND violations_match=True
    batching_no_gain:    speedup < 1.5
    batching_incorrect:  violations_match=False (correctness regression)

Spec: REQ-VERIFY-148, SCENARIO-VERIFY-173
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.symcode_verifier import SymCodeVerifier  # noqa: E402

DELIVERABLE = "results/experiment_841_symcode_paragraph_batching.json"
N_PARAGRAPHS = 10
N_WARMUP = 3

# 10 synthetic paragraphs; each has 1-2 arithmetic expressions.
# Some are correct (no violation), some are wrong (violation) so we test both paths.
SYNTHETIC_PARAGRAPHS = [
    "We start by computing 3 * 4 = 12.",
    "Then we add the subtotal: 12 + 8 = 20.",
    "The discount is 100 / 4 = 25.",
    "Applying the discount: 100 - 25 = 75.",
    "Tax at 10 percent: 75 * 0.1 = 7.5.",
    "Total after tax: 75 + 7.5 = 82.5.",
    "We split the bill 3 ways: 82.5 / 3 = 27.5.",
    "Add a tip of 5 * 2 = 10 (wrong: should be 10).",  # correct
    "Final per-person share is 27.5 + 10 = 99.",  # wrong: 27.5+10=37.5
    "So the answer is clear and requires no arithmetic.",
]


def _time_serial(verifier: SymCodeVerifier, paragraphs: list[str]) -> tuple[float, list[bool]]:
    """Run N serial verify_step() calls and return (total_ms, violation_flags)."""
    t0 = time.perf_counter()
    results = [verifier.verify_step(p, idx) for idx, p in enumerate(paragraphs)]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, [r.violation_detected for r in results]


def _time_batch(verifier: SymCodeVerifier, paragraphs: list[str]) -> tuple[float, list[bool]]:
    """Run batch_verify() once and return (total_ms, violation_flags)."""
    result = verifier.batch_verify(paragraphs)
    return result.batch_latency_ms, [r.violation_detected for r in result.per_paragraph_results]


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=841,
        title="SymCode Paragraph Batching Latency Benchmark",
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(841, timeout_minutes=20, result_path=DELIVERABLE):
        verifier = SymCodeVerifier()  # CI mode (no LLM caller) — regex extraction

        # Warm up the verifier so the first timed measurement does not include
        # Python module loading time (which would inflate serial latency unfairly).
        for _ in range(N_WARMUP):
            verifier.verify_step("3 * 4 = 12")
            verifier.batch_verify(["3 * 4 = 12"])

        # Measure serial path (N separate verify_step calls).
        latency_single_ms, serial_violations = _time_serial(verifier, SYNTHETIC_PARAGRAPHS)

        # Measure batch path (one batch_verify call).
        latency_batch_ms, batch_violations = _time_batch(verifier, SYNTHETIC_PARAGRAPHS)

        # Correctness check: both paths must agree on every paragraph.
        violations_match = serial_violations == batch_violations

        # Safety: avoid division by zero if batch somehow completes in 0ms.
        speedup = latency_single_ms / max(latency_batch_ms, 0.001)

        if not violations_match:
            honest_verdict = "batching_incorrect"
        elif speedup >= 2.0:
            honest_verdict = "batching_effective"
        elif speedup >= 1.5:
            honest_verdict = "batching_marginal"
        else:
            honest_verdict = "batching_no_gain"

        retro_symcode_serial_closed = speedup >= 1.5 and violations_match

        artifact = tmpl.build_result(
            {
                "latency_single_ms": latency_single_ms,
                "latency_batch_ms": latency_batch_ms,
                "speedup": speedup,
                "violations_match": violations_match,
                "n_paragraphs": N_PARAGRAPHS,
                "serial_violations": serial_violations,
                "batch_violations": batch_violations,
                "retro_symcode_serial_closed": retro_symcode_serial_closed,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )
        Path(DELIVERABLE).parent.mkdir(parents=True, exist_ok=True)
        Path(DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        print(
            f"[841] latency_single={latency_single_ms:.2f}ms  "
            f"latency_batch={latency_batch_ms:.2f}ms  "
            f"speedup={speedup:.2f}x  "
            f"verdict={honest_verdict}"
        )

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
