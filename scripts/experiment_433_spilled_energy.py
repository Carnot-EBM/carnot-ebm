#!/usr/bin/env python3
"""Experiment 433: SpilledEnergyDetector benchmark on synthetic responses.

**Purpose:**
    Tests whether the spilled-energy signal from arXiv 2602.18671 provides a
    useful pre-filter for the verification pipeline. Uses CI-safe text mode
    (hash-based proxy) because no GPU is required for the detector's text mode.

**Protocol:**
    1. apply_env_autofix() — self-inject CARNOT_FORCE_LIVE=1 if GPU is present.
    2. ExperimentTimeoutWatchdog(433, timeout_minutes=20) — hard wall-clock cap.
    3. Create 100 synthetic responses:
       - 50 "correct": factual, confident phrasing
       - 50 "hallucinated": incoherent, random-word structure
    4. Run SpilledEnergyDetector().score_from_text(text) on each.
    5. Compute skip_rate, fn_rate, fp_rate.
    6. Compute honest_verdict:
       'spilled_energy_viable' if skip_rate > 0.20 AND fn_rate < 0.05
       else 'insufficient_signal'
    7. Build artifact with schema='carnot.spilled_energy.v1'.

**Output:** results/experiment_433_spilled_energy.json

Spec: REQ-VERIFY-092, REQ-VERIFY-093
SCENARIO-VERIFY-123, SCENARIO-VERIFY-124, SCENARIO-VERIFY-125
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# STEP 0: Apply EnvironmentAutoFix FIRST — before any env-sensitive imports
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

import json
import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.spilled_energy import SpilledEnergyDetector  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
_log = logging.getLogger(__name__)

_OUTPUT_PATH = "results/experiment_433_spilled_energy.json"


# ---------------------------------------------------------------------------
# build_synthetic_corpus
# ---------------------------------------------------------------------------


def build_synthetic_corpus() -> list[dict[str, Any]]:
    """Build 100 synthetic responses: 50 correct + 50 hallucinated.

    **Detailed explanation for engineers:**
        The "correct" responses use confident factual phrasing — short, direct
        sentences about established facts. The model should (in theory) produce
        low-uncertainty logits for such text.

        The "hallucinated" responses use incoherent word combinations with no
        semantic structure. In CI text mode (hash-based proxy), the distinction
        is reflected only in the hash of the text, so we use varied text to get
        varied proxy energies.

        Each dict has:
            - "text": the response text
            - "is_correct": True for factual responses, False for hallucinated

    Returns:
        List of 100 dicts, first 50 correct then 50 hallucinated.
    """
    correct_responses = [
        f"The answer is {i}. This is a factual, confident response with clear reasoning."
        for i in range(50)
    ]

    hallucinated_responses = [
        (
            f"Frumious bandersnatch quartz epoch {i} "
            f"galumphing slithy toves mimsy borogoves "
            f"outgrabe wabe brillig vorpal blade snicker-snack {i * 7} "
            f"jabberwock callooh callay beamish manxome foe ruminant {i * 13}"
        )
        for i in range(50)
    ]

    corpus = []
    for text in correct_responses:
        corpus.append({"text": text, "is_correct": True})
    for text in hallucinated_responses:
        corpus.append({"text": text, "is_correct": False})

    return corpus


# ---------------------------------------------------------------------------
# run_spilled_energy_benchmark
# ---------------------------------------------------------------------------


def run_spilled_energy_benchmark(
    corpus: list[dict[str, Any]],
    detector: SpilledEnergyDetector,
) -> dict[str, Any]:
    """Run SpilledEnergyDetector on all corpus items and compute metrics.

    **Detailed explanation for engineers:**
        For each item, score_from_text() returns a SpilledEnergyDetectorResult.
        We treat should_verify=True as "needs verification" (uncertain = possible
        hallucination) and should_verify=False as "skip verification" (confident).

        Metrics:
            skip_rate: fraction of responses where should_verify=False (skipped)
                Higher skip_rate = more efficient pipeline (fewer Ising calls)

            fn_rate (false negative rate): of all HALLUCINATED responses,
                what fraction were incorrectly skipped (should_verify=False)?
                Lower is better — these are the dangerous misses.

            fp_rate (false positive rate): of all CORRECT responses,
                what fraction were unnecessarily verified (should_verify=True)?
                Lower is better — these are unnecessary overhead.

    Args:
        corpus: List of {"text": str, "is_correct": bool} dicts.
        detector: SpilledEnergyDetector instance.

    Returns:
        Dict with skip_rate, fn_rate, fp_rate, n_total, n_correct, n_hallucinated,
        n_skipped, n_fn, n_fp, per_item results.
    """
    n_total = len(corpus)
    n_correct = sum(1 for item in corpus if item["is_correct"])
    n_hallucinated = n_total - n_correct

    n_skipped = 0    # should_verify=False (pipeline would skip)
    n_fn = 0         # hallucinated AND skipped (false negative — dangerous)
    n_fp = 0         # correct AND verified (false positive — unnecessary overhead)

    per_item = []

    for item in corpus:
        text = item["text"]
        is_correct = item["is_correct"]
        result = detector.score_from_text(text)

        skipped = not result.should_verify

        if skipped:
            n_skipped += 1
            if not is_correct:
                # Hallucinated but skipped → false negative (missed hallucination)
                n_fn += 1
        else:
            # Verified
            if is_correct:
                # Correct but verified → false positive (unnecessary work)
                n_fp += 1

        per_item.append(
            {
                "text_snippet": text[:60],
                "is_correct": is_correct,
                "should_verify": result.should_verify,
                "mean_spilled": result.mean_spilled,
                "high_spill_fraction": result.high_spill_fraction,
                "skipped": skipped,
            }
        )

    skip_rate = n_skipped / n_total
    fn_rate = (n_fn / n_hallucinated) if n_hallucinated > 0 else 0.0
    fp_rate = (n_fp / n_correct) if n_correct > 0 else 0.0

    return {
        "skip_rate": skip_rate,
        "fn_rate": fn_rate,
        "fp_rate": fp_rate,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_hallucinated": n_hallucinated,
        "n_skipped": n_skipped,
        "n_fn": n_fn,
        "n_fp": n_fp,
        "per_item": per_item,
    }


# ---------------------------------------------------------------------------
# compute_honest_verdict
# ---------------------------------------------------------------------------


def compute_honest_verdict(skip_rate: float, fn_rate: float) -> str:
    """Determine whether SpilledEnergyDetector shows viable hallucination detection.

    **Detailed explanation for engineers:**
        Viability requires BOTH:
        1. skip_rate > 0.20: The detector saves at least 20% of verification calls.
           If it skips nothing, it adds no value as a pre-filter.
        2. fn_rate < 0.05: The detector misses fewer than 5% of hallucinations.
           If it misses too many, it's not safe to use as a pre-filter.

        In CI text mode (hash-based proxy), these thresholds test the proxy
        behavior, not real hallucination detection. The verdict is reported
        honestly as 'insufficient_signal' if the proxy does not meet thresholds.

    Args:
        skip_rate: Fraction of responses skipped (should_verify=False).
        fn_rate: Fraction of hallucinated responses incorrectly skipped.

    Returns:
        'spilled_energy_viable' or 'insufficient_signal'.
    """
    if skip_rate > 0.20 and fn_rate < 0.05:
        return "spilled_energy_viable"
    return "insufficient_signal"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 433: SpilledEnergyDetector benchmark."""
    tmpl = ExperimentTemplate(
        exp_id=433,
        title="SpilledEnergyDetector benchmark on synthetic responses",
        deliverable=_OUTPUT_PATH,
        requires_gpu=False,  # CI-safe text mode — no GPU needed
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(433, timeout_minutes=20):
        _log.info("Exp 433: Building synthetic corpus (100 items: 50 correct + 50 hallucinated)")
        corpus = build_synthetic_corpus()

        _log.info("Exp 433: Running SpilledEnergyDetector.score_from_text() on all items")
        detector = SpilledEnergyDetector()
        metrics = run_spilled_energy_benchmark(corpus, detector)

        skip_rate = metrics["skip_rate"]
        fn_rate = metrics["fn_rate"]
        fp_rate = metrics["fp_rate"]
        honest_verdict = compute_honest_verdict(skip_rate, fn_rate)

        _log.info(
            "Exp 433: skip_rate=%.3f fn_rate=%.3f fp_rate=%.3f verdict=%s",
            skip_rate, fn_rate, fp_rate, honest_verdict,
        )

        artifact = tmpl.build_result(
            {
                "skip_rate": skip_rate,
                "fn_rate": fn_rate,
                "fp_rate": fp_rate,
                "honest_verdict": honest_verdict,
                "n_total": metrics["n_total"],
                "n_correct": metrics["n_correct"],
                "n_hallucinated": metrics["n_hallucinated"],
                "n_skipped": metrics["n_skipped"],
                "n_fn": metrics["n_fn"],
                "n_fp": metrics["n_fp"],
                "spill_threshold": detector.spill_threshold,
                "high_spill_fraction_threshold": detector.high_spill_fraction_threshold,
                "inference_mode": "ci_text_proxy",
            },
            status="success",
        )
        # Override schema with the canonical tag for this experiment type
        artifact["schema"] = "carnot.spilled_energy.v1"

        out_path = _REPO_ROOT / _OUTPUT_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        _log.info("Exp 433: Result written to %s", _OUTPUT_PATH)


if __name__ == "__main__":
    main()
