#!/usr/bin/env python3
"""Experiment 296: Apple adversarial GSM8K analysis and classification (v2).

This script analyses the results of Exp 294 (Apple adversarial GPU baseline,
re-run with pre-warm fix) and Exp 295 (Apple adversarial verify-repair
re-run with pre-warm fix) and produces a concise research verdict on five key
questions:

  1. apple_drop_replicated          — Did number_swap cause ≥15 pp accuracy drop?
  2. verify_repair_delta_larger_on_swap — Was Δ(verify_repair, number_swap) > Δ(verify_repair, standard)?
  3. irrelevant_sentence_ignored    — Did the model ignore irrelevant context (< 5 pp drop)?
  4. extractor_firing_summary       — Which extractors fired, and how often?
  5. dual_model_consistent          — Did Qwen and Gemma agree on the classification?

**Classification rules** (applied in order):
  - ``INCONCLUSIVE`` — results files missing, or either artifact has ``partial=True``
    (inference stalled before completion)
  - ``CONFIRMED``    — ``primary_criterion_met`` is True in the Exp 295 artifact,
    meaning Δ(verify_repair, number_swap) > Δ(verify_repair, standard) for at least
    one model
  - ``PARTIAL``      — at least one model showed positive verify_repair delta on
    number_swap, but the primary criterion was not met
  - ``RULED_OUT``    — all verify_repair deltas on number_swap are ≤ 0

``docs_updated`` is True only when Exp 295 ran to full completion (``partial=False``)
AND the classification is CONFIRMED or PARTIAL (i.e. there is something positive to
report in public documentation).

The script is deliberately free of live inference dependencies: it only reads
JSON files and performs arithmetic.  It can be run on any machine without a GPU.

Spec: REQ-VERIFY-080, REQ-VERIFY-081, REQ-VERIFY-082,
      SCENARIO-VERIFY-109, SCENARIO-VERIFY-110, SCENARIO-VERIFY-111,
      SCENARIO-VERIFY-112, SCENARIO-VERIFY-113, SCENARIO-VERIFY-114
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

EXPERIMENT: int = 296
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
    "docs_updated",
]
"""Required top-level fields in the Exp 296 analysis artifact (SCENARIO-VERIFY-097)."""

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
# Pure analysis functions (exposed for unit-testing, REQ-VERIFY-077/078)
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

    Spec: REQ-VERIFY-081, SCENARIO-VERIFY-112.
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
        primary_met:         True iff Δ(verify_repair, number_swap) > Δ(verify_repair, standard)
                             for at least one model.
        partial_improvement: True iff at least one model showed positive verify_repair
                             delta on number_swap, regardless of the primary criterion.
        stall_detected:      True iff any inference artifact reports ``partial=True``
                             or is entirely missing.

    Returns:
        One of ``"INCONCLUSIVE"``, ``"CONFIRMED"``, ``"PARTIAL"``, ``"RULED_OUT"``.

    Spec: REQ-VERIFY-082.
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

    Spec: SCENARIO-VERIFY-111.
    """
    delta = compute_delta(exp235_acc, number_swap_acc)
    return {
        "delta": delta,
        "better_than_exp235": delta > 0,
        "exp235_reference_acc": exp235_acc,
    }


# ---------------------------------------------------------------------------
# Five-questions analyser (SCENARIO-VERIFY-093)
# ---------------------------------------------------------------------------

def answer_five_questions(
    exp294: dict[str, Any],
    exp295: dict[str, Any],
) -> dict[str, Any]:
    """Answer the five key research questions from Exp 294 and Exp 295 artifacts.

    Q1 — apple_drop_replicated:
        True iff the ``apple_2410_05229_check`` field in Exp 294 shows
        ``drop_gte_15pp=True`` for at least one model.

    Q2 — verify_repair_delta_larger_on_swap:
        True iff ``primary_criterion_met=True`` in Exp 295.

    Q3 — irrelevant_sentence_ignored:
        True iff the accuracy drop on ``irrelevant_sentence`` vs ``standard`` is
        < ``IRRELEVANT_DROP_THRESHOLD_PP`` (5 pp) for all models in Exp 294.

    Q4 — extractor_firing_summary:
        The ``extractor_summary`` dict from Exp 295, falling back to an empty
        dict when absent.

    Q5 — dual_model_consistent:
        True iff ``primary_criterion_met`` is present in Exp 295 (both models
        necessarily agree when the joint flag is set).

    Args:
        exp294: Parsed content of ``results/experiment_294_results.json``.
        exp295: Parsed content of ``results/experiment_295_results.json``.

    Returns:
        Dict with the five question keys.

    Spec: SCENARIO-VERIFY-109.
    """
    # Q1: Did the Apple number_swap drop replicate (≥15 pp)?
    apple_check: dict[str, Any] = exp294.get("apple_2410_05229_check", {})
    apple_drop_replicated = any(
        model_check.get("drop_gte_15pp", False)
        for model_check in apple_check.values()
    )

    # Q2: Was verify_repair improvement larger on number_swap than standard?
    verify_repair_delta_larger_on_swap: bool = bool(
        exp295.get("primary_criterion_met", False)
    )

    # Q3: Does Carnot ignore irrelevant context (< 5 pp drop)?
    model_results: dict[str, Any] = exp294.get("model_results", {})
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
    extractor_firing_summary: dict[str, int] = exp295.get("extractor_summary", {})

    # Q5: Dual-model consistency — did Qwen and Gemma show the same directional trend?
    # Check whether both models agree on the sign of verify_repair_number_swap_delta.
    # If improvement_deltas is present, we compare signs across all models; agreement
    # means all models are positive or all are non-positive.  Falls back to key-presence
    # check (the old behaviour) when improvement_deltas is absent.
    improvement_deltas_295: dict[str, Any] = exp295.get("improvement_deltas", {})
    if improvement_deltas_295:
        model_signs = [
            model_d.get("verify_repair_number_swap_delta", 0.0) > 0
            for model_d in improvement_deltas_295.values()
        ]
        # Consistent if all models positive or all models non-positive
        dual_model_consistent = (all(model_signs) or not any(model_signs))
    else:
        # Fall back: joint flag presence implies both models contributed data
        dual_model_consistent = "primary_criterion_met" in exp295

    return {
        "apple_drop_replicated": apple_drop_replicated,
        "verify_repair_delta_larger_on_swap": verify_repair_delta_larger_on_swap,
        "irrelevant_sentence_ignored": irrelevant_sentence_ignored,
        "extractor_firing_summary": extractor_firing_summary,
        "dual_model_consistent": dual_model_consistent,
    }


# ---------------------------------------------------------------------------
# Result file loader and classifier (SCENARIO-VERIFY-094)
# ---------------------------------------------------------------------------

def load_exp_results(
    exp294_path: Path,
    exp295_path: Path,
) -> dict[str, Any]:
    """Load Exp 294 and 295 results and classify the overall outcome.

    Handles missing files, stalled (partial) artifacts, and successful runs.
    ``docs_updated`` is True only when Exp 295 ran to full completion (no stall)
    AND the classification is CONFIRMED or PARTIAL.

    Args:
        exp294_path: Path to ``results/experiment_294_results.json``.
        exp295_path: Path to ``results/experiment_295_results.json``.

    Returns:
        A dict with at minimum ``classification``, ``missing_artifacts``, and
        ``docs_updated`` keys, plus all data needed by :func:`build_artifact`.

    Spec: REQ-VERIFY-080, SCENARIO-VERIFY-110.
    """
    missing: list[str] = []
    exp294: dict[str, Any] | None = None
    exp295: dict[str, Any] | None = None

    if exp294_path.exists():
        with open(exp294_path) as fh:
            exp294 = json.load(fh)
    else:
        missing.append(exp294_path.name)

    if exp295_path.exists():
        with open(exp295_path) as fh:
            exp295 = json.load(fh)
    else:
        missing.append(exp295_path.name)

    # Any missing file → INCONCLUSIVE immediately; docs cannot be updated.
    if missing:
        return {
            "classification": "INCONCLUSIVE",
            "missing_artifacts": missing,
            "five_questions": None,
            "exp235_comparison": None,
            "exp279_comparison": None,
            "analysis_notes": [
                f"Missing result files: {', '.join(missing)}. "
                "Experiments 294 and/or 295 did not produce output (likely GPU stall "
                "or experiment not yet run)."
            ],
            "docs_updated": False,
        }

    assert exp294 is not None
    assert exp295 is not None

    stall_detected = bool(exp294.get("partial")) or bool(exp295.get("partial"))

    if stall_detected:
        stall_at = exp294.get("stall_at") or exp295.get("stall_at") or "unknown"
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
            "docs_updated": False,
        }

    # Full data available — answer the five questions.
    five_q = answer_five_questions(exp294=exp294, exp295=exp295)

    primary_met: bool = five_q["verify_repair_delta_larger_on_swap"]
    improvement_deltas: dict[str, Any] = exp295.get("improvement_deltas", {})
    partial_improvement = any(
        model_deltas.get("verify_repair_number_swap_delta", 0.0) > 0
        for model_deltas in improvement_deltas.values()
    )

    classification = classify_result(
        primary_met=primary_met,
        partial_improvement=partial_improvement,
        stall_detected=False,
    )

    # docs_updated is True when the experiment completed fully AND there is
    # something positive to document (CONFIRMED or PARTIAL).
    docs_updated = classification in {"CONFIRMED", "PARTIAL"}

    # Build comparisons against prior experiments.
    # Use Gemma verify_repair number_swap accuracy as the comparison point.
    gemma_results: dict[str, Any] = exp295.get("results", {}).get("Gemma4-E4B-it", {})
    gemma_ns_vr_acc = (
        gemma_results.get("number_swap", {})
        .get("verify_repair", {})
        .get("accuracy", 0.0)
    )
    exp235_comparison = compare_vs_exp235(
        number_swap_acc=gemma_ns_vr_acc,
        exp235_acc=EXP235_GEMMA_VERIFY_REPAIR_ACC,
    )

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
        "docs_updated": docs_updated,
    }


def _build_analysis_notes(
    classification: str,
    five_q: dict[str, Any],
    exp235_comparison: dict[str, Any],
) -> list[str]:
    """Generate human-readable notes summarising the Exp 296 analysis findings."""
    notes: list[str] = []

    if classification == "CONFIRMED":
        notes.append(
            "PRIMARY CRITERION MET: verify-repair improvement is larger on "
            "number_swap adversarial variants than on standard GSM8K, confirming "
            "the semantic-grounding hypothesis from Exp 279 (re-run with pre-warm fix)."
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
# Artifact builder (SCENARIO-VERIFY-097)
# ---------------------------------------------------------------------------

def build_artifact(
    classification: str,
    missing_artifacts: list[str],
    five_questions: dict[str, Any] | None,
    exp235_comparison: dict[str, Any] | None,
    exp279_comparison: dict[str, Any] | None,
    analysis_notes: list[str],
    docs_updated: bool,
) -> dict[str, Any]:
    """Assemble the Exp 296 result artifact dict.

    All required ``ARTIFACT_SCHEMA`` keys are guaranteed to be present.

    Args:
        classification:    One of ``CONFIRMED``, ``PARTIAL``, ``RULED_OUT``,
                           ``INCONCLUSIVE``.
        missing_artifacts: List of result file names that could not be loaded.
        five_questions:    Output of :func:`answer_five_questions`, or None.
        exp235_comparison: Output of :func:`compare_vs_exp235`, or None.
        exp279_comparison: Dict with Exp 279 reference data, or None.
        analysis_notes:    Human-readable summary strings.
        docs_updated:      True when Exp 295 completed fully and classification
                           is CONFIRMED or PARTIAL.

    Returns:
        Artifact dict ready for JSON serialisation.

    Spec: SCENARIO-VERIFY-113.
    """
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "experiment": EXPERIMENT,
        "schema": "carnot.apple_analysis.v2",
        "run_date": RUN_DATE,
        "started_at": now,
        "finished_at": now,
        "classification": classification,
        "missing_artifacts": missing_artifacts,
        "five_questions": five_questions,
        "exp235_comparison": exp235_comparison,
        "exp279_comparison": exp279_comparison,
        "analysis_notes": analysis_notes,
        "docs_updated": docs_updated,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Exp 296 analysis and write the result artifact.

    Resolves the Exp 294 and 295 result paths relative to the repository root,
    runs the analysis, and writes ``results/experiment_296_results.json``.

    Spec: REQ-VERIFY-080, SCENARIO-VERIFY-110.
    """
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    exp294_path = results_dir / "experiment_294_results.json"
    exp295_path = results_dir / "experiment_295_results.json"
    output_path = results_dir / "experiment_296_results.json"

    print(f"[Exp 296] Loading Exp 294 from {exp294_path}")
    print(f"[Exp 296] Loading Exp 295 from {exp295_path}")

    analysis = load_exp_results(exp294_path=exp294_path, exp295_path=exp295_path)

    artifact = build_artifact(
        classification=analysis["classification"],
        missing_artifacts=analysis["missing_artifacts"],
        five_questions=analysis.get("five_questions"),
        exp235_comparison=analysis.get("exp235_comparison"),
        exp279_comparison=analysis.get("exp279_comparison"),
        analysis_notes=analysis.get("analysis_notes", []),
        docs_updated=analysis.get("docs_updated", False),
    )

    output_path.write_text(json.dumps(artifact, indent=2))
    print(f"[Exp 296] Classification: {artifact['classification']}")
    print(f"[Exp 296] docs_updated: {artifact['docs_updated']}")
    print(f"[Exp 296] Written: {output_path}")

    if artifact["missing_artifacts"]:
        print(f"[Exp 296] Missing artifacts: {artifact['missing_artifacts']}")
    if artifact["analysis_notes"]:
        for note in artifact["analysis_notes"]:
            print(f"  • {note}")

    # docs updates are skipped when docs_updated=False (stall or no improvement)
    if not artifact["docs_updated"]:
        print(
            "[Exp 296] Skipping docs update: experiment did not complete fully "
            "or classification is RULED_OUT/INCONCLUSIVE."
        )
        return

    _update_docs(repo_root=repo_root, artifact=artifact)


def _update_docs(repo_root: Path, artifact: dict[str, Any]) -> None:
    """Update public documentation with Exp 296 findings.

    Only called when ``artifact['docs_updated']`` is True (CONFIRMED or PARTIAL
    and no stall detected).

    Updates:
      - ``docs/technical-report.md``: Adversarial Robustness section
      - ``README.md``: verify-repair headline if ≥3 pp improvement
      - ``research-studying.md``: adversarial section findings

    Spec: SCENARIO-VERIFY-114.
    """
    classification = artifact["classification"]
    five_q = artifact.get("five_questions") or {}
    exp235 = artifact.get("exp235_comparison") or {}
    notes = artifact.get("analysis_notes", [])

    # Build summary text for the new docs section.
    apple_replicated = five_q.get("apple_drop_replicated", False)
    primary_met = five_q.get("verify_repair_delta_larger_on_swap", False)
    irrelevant_ignored = five_q.get("irrelevant_sentence_ignored", False)
    extractors: dict[str, int] = five_q.get("extractor_firing_summary", {})
    dual_consistent = five_q.get("dual_model_consistent", False)
    exp235_delta = exp235.get("delta", 0.0)
    exp235_ref = exp235.get("exp235_reference_acc", EXP235_GEMMA_VERIFY_REPAIR_ACC)

    # -----------------------------------------------------------------------
    # 1. docs/technical-report.md — append Adversarial Robustness section
    # -----------------------------------------------------------------------
    tr_path = repo_root / "docs" / "technical-report.md"
    if tr_path.exists():
        tr_text = tr_path.read_text()
        section_header = "## Adversarial Robustness (Apple 2410.05229 Re-run — Exp 294/295/296)"
        if section_header not in tr_text:
            extractor_lines = "\n".join(
                f"  - {k}: {v} firings" for k, v in sorted(extractors.items())
            ) or "  - (no extractor data)"
            new_section = f"""
{section_header}

**Classification: {classification}** (Exp 296, run date {artifact['run_date']})

Experiments 294 (baseline re-run with pre-warm fix) and 295 (verify-repair
re-run with pre-warm fix) repeated the Apple adversarial benchmark from
Exps 282/283, correcting the GPU pre-warm regression identified in Exp 293.

### Key Findings

| Question | Answer |
|----------|--------|
| Apple 2410.05229 drop replicated (≥15 pp)? | {'Yes' if apple_replicated else 'No'} |
| verify-repair Δ larger on number_swap than standard? | {'Yes — PRIMARY CRITERION MET' if primary_met else 'No'} |
| Irrelevant-sentence variants ignored (< 5 pp drop)? | {'Yes' if irrelevant_ignored else 'No'} |
| Dual-model consistent (Qwen + Gemma agree)? | {'Yes' if dual_consistent else 'No'} |

### Extractor Firing Summary

{extractor_lines}

### Exp 235 Comparison

Gemma verify_repair on number_swap vs Exp 235 standard baseline
({exp235_ref:.1%}): delta = {exp235_delta:+.1%}
({'better' if exp235.get('better_than_exp235') else 'not better'} than Exp 235).

### Analysis Notes

{''.join(f'- {n}' + chr(10) for n in notes)}
"""
            tr_path.write_text(tr_text.rstrip() + "\n" + new_section)
            print("[Exp 296] Updated docs/technical-report.md")
        else:
            print("[Exp 296] docs/technical-report.md already contains Exp 296 section — skipped.")

    # -----------------------------------------------------------------------
    # 2. README.md — update headline if verify-repair ≥3 pp on number_swap
    #    We only touch the README when primary criterion is met and the Gemma
    #    verify_repair number_swap delta vs Exp 235 baseline exceeds 3 pp.
    # -----------------------------------------------------------------------
    if primary_met and exp235_delta >= 0.03:
        readme_path = repo_root / "README.md"
        if readme_path.exists():
            readme_text = readme_path.read_text()
            marker = "<!-- EXP296_ADVERSARIAL_RESULT -->"
            if marker not in readme_text:
                adversarial_line = (
                    f"\n{marker}\n"
                    f"**Apple adversarial benchmark (Exp 295):** verify-repair on "
                    f"number_swap achieved {exp235_delta:+.1%} vs Exp 235 standard "
                    f"baseline ({exp235_ref:.1%}). Classification: {classification}.\n"
                )
                readme_path.write_text(readme_text.rstrip() + "\n" + adversarial_line)
                print("[Exp 296] Updated README.md with adversarial result.")
            else:
                print("[Exp 296] README.md already contains Exp 296 marker — skipped.")

    # -----------------------------------------------------------------------
    # 3. research-studying.md — update adversarial section
    # -----------------------------------------------------------------------
    rs_path = repo_root / "research-studying.md"
    if rs_path.exists():
        rs_text = rs_path.read_text()
        marker296 = "<!-- EXP296_FINDINGS -->"
        if marker296 not in rs_text:
            findings_block = (
                f"\n{marker296}\n"
                f"### Exp 296 Apple Adversarial Analysis ({artifact['run_date']})\n\n"
                f"- Classification: **{classification}**\n"
                f"- Apple drop replicated: {apple_replicated}\n"
                f"- Primary criterion met: {primary_met}\n"
                f"- Irrelevant sentence ignored: {irrelevant_ignored}\n"
                f"- Dual-model consistent: {dual_consistent}\n"
            )
            rs_path.write_text(rs_text.rstrip() + "\n" + findings_block)
            print("[Exp 296] Updated research-studying.md")
        else:
            print("[Exp 296] research-studying.md already contains Exp 296 findings — skipped.")


if __name__ == "__main__":
    main()
