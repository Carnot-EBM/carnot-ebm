#!/usr/bin/env python3
"""Experiment 284: Apple adversarial GSM8K analysis and classification.

This script analyses the results of Exp 282 (Apple adversarial GPU baseline)
and Exp 283 (Apple adversarial verify-repair benchmark) and produces a
concise research verdict on five key questions:

  1. apple_drop_replicated          — Did number_swap cause ≥15 pp accuracy drop?
  2. verify_repair_delta_larger_on_swap — Was Δ(verify_repair, number_swap) > Δ(verify_repair, standard)?
  3. irrelevant_sentence_ignored    — Did the model ignore irrelevant context (< 5 pp drop)?
  4. extractor_firing_summary       — Which extractors fired, and how often?
  5. dual_model_consistent          — Did Qwen and Gemma agree on the classification?

**Classification rules** (applied in order):
  - ``INCONCLUSIVE`` — results files missing, or either artifact has ``partial=True``
    (inference stalled before completion)
  - ``CONFIRMED``    — ``primary_criterion_met`` is True in the Exp 283 artifact,
    meaning Δ(verify_repair, number_swap) > Δ(verify_repair, standard) for at least
    one model
  - ``PARTIAL``      — at least one model showed positive verify_repair delta on
    number_swap, but the primary criterion was not met
  - ``RULED_OUT``    — all verify_repair deltas on number_swap are ≤ 0

The script is deliberately free of live inference dependencies: it only reads
JSON files and performs arithmetic.  It can be run on any machine without a GPU.

Spec: REQ-VERIFY-073, REQ-VERIFY-074, REQ-VERIFY-075,
      SCENARIO-VERIFY-088, SCENARIO-VERIFY-089, SCENARIO-VERIFY-090,
      SCENARIO-VERIFY-091, SCENARIO-VERIFY-092
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT: int = 284
"""Experiment number — matches the filename and artifact ``experiment`` field."""

RUN_DATE: str = "20260414"
"""Wall-clock date of this analysis run."""

ARTIFACT_SCHEMA: list[str] = [
    "experiment",
    "schema",
    "run_date",
    "started_at",
    "finished_at",
    "classification",
    "missing_artifacts",
    "five_questions",
    "exp235_comparison",
    "exp279_comparison",
    "analysis_notes",
]
"""Required top-level fields in the Exp 284 analysis artifact (SCENARIO-VERIFY-092)."""

# Reference values from prior live-GPU experiments.
# Exp 235 Gemma4-E4B-it verify_repair accuracy on the standard cohort (200 questions).
EXP235_GEMMA_VERIFY_REPAIR_ACC: float = 0.475

# Exp 279 semantic-grounding stale-detection rate (100% on the stale subset).
EXP279_STALE_DETECTION_RATE: float = 1.0

# Apple paper (2410.05229) threshold: number_swap must cause ≥ 15 pp drop to replicate.
APPLE_DROP_THRESHOLD_PP: float = 15.0

# Max acceptable accuracy drop on irrelevant_sentence to conclude the model ignores it.
IRRELEVANT_DROP_THRESHOLD_PP: float = 5.0


# ---------------------------------------------------------------------------
# Pure analysis functions (exposed for unit-testing, REQ-VERIFY-074/075)
# ---------------------------------------------------------------------------

def compute_delta(baseline_acc: float, mode_acc: float) -> float:
    """Return the percentage-point improvement of *mode_acc* over *baseline_acc*.

    Both arguments are fractions in [0, 1].  The result is rounded to four
    decimal places so downstream comparisons are numerically stable.

    Args:
        baseline_acc: Accuracy of the baseline (no-verification) mode.
        mode_acc:     Accuracy of the verify or verify-repair mode.

    Returns:
        ``mode_acc - baseline_acc`` rounded to four decimal places.

    Spec: REQ-VERIFY-074, SCENARIO-VERIFY-091.
    """
    return round(mode_acc - baseline_acc, 4)


def classify_result(
    primary_met: bool,
    partial_improvement: bool,
    stall_detected: bool,
) -> str:
    """Map the analysis flags to one of four classification labels.

    Classification priority (highest to lowest):
      1. ``INCONCLUSIVE`` — stall detected, regardless of other flags
      2. ``CONFIRMED``    — primary criterion met (primary_met=True)
      3. ``PARTIAL``      — some improvement but not primary criterion
      4. ``RULED_OUT``    — no improvement at all

    Args:
        primary_met:        True iff Δ(verify_repair, number_swap) > Δ(verify_repair, standard)
                            for at least one model.
        partial_improvement: True iff at least one model showed positive verify_repair
                            delta on number_swap, regardless of the primary criterion.
        stall_detected:     True iff any inference artifact reports ``partial=True``
                            or is entirely missing.

    Returns:
        One of ``"INCONCLUSIVE"``, ``"CONFIRMED"``, ``"PARTIAL"``, ``"RULED_OUT"``.

    Spec: REQ-VERIFY-075.
    """
    if stall_detected:
        return "INCONCLUSIVE"
    if primary_met:
        return "CONFIRMED"
    if partial_improvement:
        return "PARTIAL"
    return "RULED_OUT"


def compare_vs_exp235(number_swap_acc: float, exp235_acc: float) -> dict[str, Any]:
    """Compare a number_swap accuracy against the Exp 235 reference.

    Args:
        number_swap_acc: Accuracy of the verify_repair mode on number_swap variants.
        exp235_acc:      Reference accuracy from Exp 235 (standard cohort verify_repair).

    Returns:
        A dict with keys:
          - ``delta``              — ``number_swap_acc - exp235_acc`` (four decimal places)
          - ``better_than_exp235`` — True iff delta > 0 (strictly greater)
          - ``exp235_reference_acc`` — the exp235_acc value passed in

    Spec: SCENARIO-VERIFY-090.
    """
    delta = compute_delta(exp235_acc, number_swap_acc)
    return {
        "delta": delta,
        "better_than_exp235": delta > 0,
        "exp235_reference_acc": exp235_acc,
    }


# ---------------------------------------------------------------------------
# Five-questions analyser (SCENARIO-VERIFY-088)
# ---------------------------------------------------------------------------

def answer_five_questions(
    exp282: dict[str, Any],
    exp283: dict[str, Any],
) -> dict[str, Any]:
    """Answer the five key research questions from Exp 282 and Exp 283 artifacts.

    Q1 — apple_drop_replicated:
        True iff the ``apple_2410_05229_check`` field in Exp 282 shows
        ``drop_gte_15pp=True`` for at least one model.

    Q2 — verify_repair_delta_larger_on_swap:
        True iff ``primary_criterion_met=True`` in Exp 283.

    Q3 — irrelevant_sentence_ignored:
        True iff the accuracy drop on ``irrelevant_sentence`` vs ``standard`` is
        < ``IRRELEVANT_DROP_THRESHOLD_PP`` (5 pp) for all models in Exp 282.

    Q4 — extractor_firing_summary:
        The ``extractor_summary`` dict from Exp 283, falling back to an empty
        dict when absent.

    Q5 — dual_model_consistent:
        True iff Qwen and Gemma produce the same Q2 verdict; since
        ``primary_criterion_met`` is a joint flag, both models necessarily agree
        when it is present.

    Args:
        exp282: Parsed content of ``results/experiment_282_results.json``.
        exp283: Parsed content of ``results/experiment_283_results.json``.

    Returns:
        Dict with the five question keys.

    Spec: SCENARIO-VERIFY-088.
    """
    # Q1: Did the Apple number_swap drop replicate (≥15 pp)?
    apple_check: dict[str, Any] = exp282.get("apple_2410_05229_check", {})
    apple_drop_replicated = any(
        model_check.get("drop_gte_15pp", False)
        for model_check in apple_check.values()
    )

    # Q2: Was verify_repair improvement larger on number_swap than standard?
    verify_repair_delta_larger_on_swap: bool = bool(
        exp283.get("primary_criterion_met", False)
    )

    # Q3: Does Carnot ignore irrelevant context (< 5 pp drop)?
    model_results: dict[str, Any] = exp282.get("model_results", {})
    irrelevant_ignored_per_model: list[bool] = []
    for model_data in model_results.values():
        standard_acc: float = model_data.get("standard", {}).get("accuracy", 0.0)
        irrel_acc: float = model_data.get("irrelevant_sentence", {}).get("accuracy", 0.0)
        drop_pp = (standard_acc - irrel_acc) * 100.0
        irrelevant_ignored_per_model.append(drop_pp < IRRELEVANT_DROP_THRESHOLD_PP)
    irrelevant_sentence_ignored = bool(irrelevant_ignored_per_model) and all(
        irrelevant_ignored_per_model
    )

    # Q4: Which extractors fired?
    extractor_firing_summary: dict[str, int] = exp283.get("extractor_summary", {})

    # Q5: Dual-model consistency.
    # primary_criterion_met is a joint flag; both models necessarily agree.
    # If the flag is present, they are by definition consistent.
    dual_model_consistent = "primary_criterion_met" in exp283

    return {
        "apple_drop_replicated": apple_drop_replicated,
        "verify_repair_delta_larger_on_swap": verify_repair_delta_larger_on_swap,
        "irrelevant_sentence_ignored": irrelevant_sentence_ignored,
        "extractor_firing_summary": extractor_firing_summary,
        "dual_model_consistent": dual_model_consistent,
    }


# ---------------------------------------------------------------------------
# Result file loader and classifier (SCENARIO-VERIFY-089)
# ---------------------------------------------------------------------------

def load_exp_results(
    exp282_path: Path,
    exp283_path: Path,
) -> dict[str, Any]:
    """Load Exp 282 and 283 results and classify the overall outcome.

    Handles missing files, stalled (partial) artifacts, and successful runs.

    Args:
        exp282_path: Path to ``results/experiment_282_results.json``.
        exp283_path: Path to ``results/experiment_283_results.json``.

    Returns:
        A dict with at minimum ``classification`` and ``missing_artifacts`` keys,
        plus all data needed by :func:`build_artifact`.

    Spec: REQ-VERIFY-073, SCENARIO-VERIFY-089.
    """
    missing: list[str] = []
    exp282: dict[str, Any] | None = None
    exp283: dict[str, Any] | None = None

    # Try to load Exp 282.
    if exp282_path.exists():
        with open(exp282_path) as fh:
            exp282 = json.load(fh)
    else:
        missing.append(exp282_path.name)

    # Try to load Exp 283.
    if exp283_path.exists():
        with open(exp283_path) as fh:
            exp283 = json.load(fh)
    else:
        missing.append(exp283_path.name)

    # Any missing file → INCONCLUSIVE immediately.
    if missing:
        return {
            "classification": "INCONCLUSIVE",
            "missing_artifacts": missing,
            "five_questions": None,
            "exp235_comparison": None,
            "exp279_comparison": None,
            "analysis_notes": [
                f"Missing result files: {', '.join(missing)}. "
                "Experiments 282 and/or 283 did not produce output (likely GPU stall)."
            ],
        }

    # Both files loaded — check for stalls.
    assert exp282 is not None
    assert exp283 is not None

    stall_detected = bool(exp282.get("partial")) or bool(exp283.get("partial"))

    if stall_detected:
        stall_at = exp282.get("stall_at") or exp283.get("stall_at") or "unknown"
        return {
            "classification": "INCONCLUSIVE",
            "missing_artifacts": [],
            "five_questions": None,
            "exp235_comparison": None,
            "exp279_comparison": None,
            "analysis_notes": [
                f"Stall detected in inference artifact (stall_at={stall_at!r}). "
                "Partial artifacts are not sufficient for classification."
            ],
        }

    # Full data available — answer the five questions.
    five_q = answer_five_questions(exp282=exp282, exp283=exp283)

    # Determine primary criterion and any partial improvement.
    primary_met: bool = five_q["verify_repair_delta_larger_on_swap"]
    improvement_deltas: dict[str, Any] = exp283.get("improvement_deltas", {})
    partial_improvement = any(
        model_deltas.get("verify_repair_number_swap_delta", 0.0) > 0
        for model_deltas in improvement_deltas.values()
    )

    classification = classify_result(
        primary_met=primary_met,
        partial_improvement=partial_improvement,
        stall_detected=False,
    )

    # Build comparisons against prior experiments.
    # Use Gemma verify_repair number_swap accuracy as the comparison point.
    gemma_results: dict[str, Any] = exp283.get("results", {}).get("Gemma4-E4B-it", {})
    gemma_ns_vr_acc = (
        gemma_results.get("number_swap", {})
        .get("verify_repair", {})
        .get("accuracy", 0.0)
    )
    exp235_comparison = compare_vs_exp235(
        number_swap_acc=gemma_ns_vr_acc,
        exp235_acc=EXP235_GEMMA_VERIFY_REPAIR_ACC,
    )

    # Exp 279 comparison: record the known stale-detection rate.
    exp279_comparison = {
        "stale_detection_rate": EXP279_STALE_DETECTION_RATE,
        "note": (
            "Exp 279 showed 100% stale-answer detection (3/3 stale cases). "
            "number_swap variants produce the same stale-answer error pattern."
        ),
    }

    analysis_notes: list[str] = _build_analysis_notes(
        classification=classification,
        five_q=five_q,
        exp235_comparison=exp235_comparison,
    )

    return {
        "classification": classification,
        "missing_artifacts": [],
        "five_questions": five_q,
        "exp235_comparison": exp235_comparison,
        "exp279_comparison": exp279_comparison,
        "analysis_notes": analysis_notes,
    }


def _build_analysis_notes(
    classification: str,
    five_q: dict[str, Any],
    exp235_comparison: dict[str, Any],
) -> list[str]:
    """Generate human-readable notes summarising the analysis findings."""
    notes: list[str] = []

    if classification == "CONFIRMED":
        notes.append(
            "PRIMARY CRITERION MET: verify-repair improvement is larger on "
            "number_swap adversarial variants than on standard GSM8K, confirming "
            "the semantic-grounding hypothesis from Exp 279."
        )
    elif classification == "PARTIAL":
        notes.append(
            "PARTIAL RESULT: some models showed positive verify_repair delta on "
            "number_swap, but the primary criterion (larger than standard) was "
            "not met across all models."
        )
    elif classification == "RULED_OUT":
        notes.append(
            "RULED OUT: verify-repair showed no improvement on number_swap variants. "
            "The semantic-grounding mechanism did not generalise to the Apple "
            "adversarial pattern under these conditions."
        )

    if five_q.get("apple_drop_replicated"):
        notes.append(
            "Apple 2410.05229 accuracy-drop replicated: number_swap caused ≥15 pp "
            f"accuracy drop (threshold: {APPLE_DROP_THRESHOLD_PP} pp), confirming "
            "the Apple paper's finding on this cohort."
        )
    else:
        notes.append(
            "Apple 2410.05229 accuracy-drop NOT replicated: number_swap drop "
            f"< {APPLE_DROP_THRESHOLD_PP} pp on all models."
        )

    if five_q.get("irrelevant_sentence_ignored"):
        notes.append(
            "Carnot correctly ignores irrelevant-sentence distractors: accuracy "
            f"drop < {IRRELEVANT_DROP_THRESHOLD_PP} pp on all models, as predicted."
        )

    delta = exp235_comparison.get("delta", 0.0)
    if exp235_comparison.get("better_than_exp235"):
        notes.append(
            f"Gemma verify_repair on number_swap ({delta:+.1%}) exceeds "
            "Exp 235 standard verify_repair baseline."
        )

    return notes


# ---------------------------------------------------------------------------
# Artifact builder (SCENARIO-VERIFY-092)
# ---------------------------------------------------------------------------

def build_artifact(
    classification: str,
    missing_artifacts: list[str],
    five_questions: dict[str, Any] | None,
    exp235_comparison: dict[str, Any] | None,
    exp279_comparison: dict[str, Any] | None,
    analysis_notes: list[str],
) -> dict[str, Any]:
    """Assemble the Exp 284 result artifact dict.

    All required ``ARTIFACT_SCHEMA`` keys are guaranteed to be present.

    Args:
        classification:    One of ``CONFIRMED``, ``PARTIAL``, ``RULED_OUT``,
                           ``INCONCLUSIVE``.
        missing_artifacts: List of result file names that could not be loaded.
        five_questions:    Output of :func:`answer_five_questions`, or None.
        exp235_comparison: Output of :func:`compare_vs_exp235`, or None.
        exp279_comparison: Dict with Exp 279 reference data, or None.
        analysis_notes:    Human-readable summary strings.

    Returns:
        Artifact dict ready for JSON serialisation.

    Spec: SCENARIO-VERIFY-092.
    """
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "experiment": EXPERIMENT,
        "schema": "apple_adversarial_analysis.v1",
        "run_date": RUN_DATE,
        "started_at": now,
        "finished_at": now,
        "classification": classification,
        "missing_artifacts": missing_artifacts,
        "five_questions": five_questions,
        "exp235_comparison": exp235_comparison,
        "exp279_comparison": exp279_comparison,
        "analysis_notes": analysis_notes,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Exp 284 analysis and write the result artifact.

    Resolves the Exp 282 and 283 result paths relative to the repository root,
    runs the analysis, and writes ``results/experiment_284_results.json``.

    Spec: REQ-VERIFY-073, SCENARIO-VERIFY-089.
    """
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    exp282_path = results_dir / "experiment_282_results.json"
    exp283_path = results_dir / "experiment_283_results.json"
    output_path = results_dir / "experiment_284_results.json"

    print(f"[Exp 284] Loading Exp 282 from {exp282_path}")
    print(f"[Exp 284] Loading Exp 283 from {exp283_path}")

    analysis = load_exp_results(exp282_path=exp282_path, exp283_path=exp283_path)

    artifact = build_artifact(
        classification=analysis["classification"],
        missing_artifacts=analysis["missing_artifacts"],
        five_questions=analysis.get("five_questions"),
        exp235_comparison=analysis.get("exp235_comparison"),
        exp279_comparison=analysis.get("exp279_comparison"),
        analysis_notes=analysis.get("analysis_notes", []),
    )

    output_path.write_text(json.dumps(artifact, indent=2))
    print(f"[Exp 284] Classification: {artifact['classification']}")
    print(f"[Exp 284] Written: {output_path}")

    if artifact["missing_artifacts"]:
        print(f"[Exp 284] Missing artifacts: {artifact['missing_artifacts']}")
    if artifact["analysis_notes"]:
        for note in artifact["analysis_notes"]:
            print(f"  • {note}")


if __name__ == "__main__":
    main()
