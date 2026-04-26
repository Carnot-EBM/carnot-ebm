"""Tests for Experiment 907: SVAMP Root-Cause v2 — FoVer Labeling Inapplicability.

Traces to: REQ-VER-085, SCENARIO-VER-085

REQ-VER-085: EstimationVerifier for single-step arithmetic word problems.
SCENARIO-VER-085: SVAMP answer in plausible arithmetic range => verified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_907_svamp_root_cause_v2 import (  # noqa: E402
    RESULT_PATH,
    _REQUIRED_FIELDS,
    assert_deliverable_written,
    compute_svamp_auc_post_filter,
    run_experiment,
)
from scripts.experiment_893_svamp_root_cause import (  # noqa: E402
    SVAMP_QUESTIONS,
    SVAMP_RESPONSES,
    GSM8K_QUESTIONS,
    GSM8K_RESPONSES,
    VOCAB_SIZE,
    LabelingResult,
    analyze_cohort,
    assign_honest_verdict,
    check_mismatch_confirmed,
    compute_cohort_stats,
    compute_vjepa_auc_on_labeled,
)


# ---------------------------------------------------------------------------
# Corpus sanity checks (REQ-VER-085)
# ---------------------------------------------------------------------------


def test_question_corpus_lengths() -> None:
    """SVAMP and GSM8K question lists must each have exactly 20 entries."""
    # REQ-VER-085: n_svamp_questions and n_gsm8k_questions = 20 each.
    assert len(SVAMP_QUESTIONS) == 20
    assert len(GSM8K_QUESTIONS) == 20


def test_responses_match_questions() -> None:
    """Response lists must be the same length as their question lists."""
    assert len(SVAMP_RESPONSES) == len(SVAMP_QUESTIONS)
    assert len(GSM8K_RESPONSES) == len(GSM8K_QUESTIONS)


# ---------------------------------------------------------------------------
# FoVer labeling analysis (REQ-VER-085)
# ---------------------------------------------------------------------------


def test_svamp_mean_cot_depth_is_single_step() -> None:
    """SVAMP responses must produce FoVer step depth < 2.0 (single-step structure).

    SCENARIO-VER-085: SVAMP questions like 'Tom has 5 apples, gives 2, how many?'
    produce one-sentence direct answers with no numbered step chains.
    """
    results = analyze_cohort(SVAMP_QUESTIONS, SVAMP_RESPONSES, "svamp")
    stats = compute_cohort_stats(results)
    assert stats["mean_cot_depth"] < 2.0, (
        f"SVAMP mean CoT depth {stats['mean_cot_depth']:.2f} >= 2.0: "
        "hypothesis contradicted — SVAMP responses appear multi-step."
    )


def test_gsm8k_mean_cot_depth_is_multi_step() -> None:
    """GSM8K responses must produce FoVer step depth > 4.0 (multi-step structure)."""
    results = analyze_cohort(GSM8K_QUESTIONS, GSM8K_RESPONSES, "gsm8k")
    stats = compute_cohort_stats(results)
    assert stats["mean_cot_depth"] > 4.0, (
        f"GSM8K mean CoT depth {stats['mean_cot_depth']:.2f} <= 4.0: "
        "simulated GSM8K responses are not producing sufficient step chains."
    )


def test_svamp_labeling_failure_rate_is_high() -> None:
    """FoVer must fail on the majority (> 0.5) of SVAMP pairs.

    FoVer expects equations of the form 'a OP b = c'; SVAMP responses have none.
    """
    results = analyze_cohort(SVAMP_QUESTIONS, SVAMP_RESPONSES, "svamp")
    stats = compute_cohort_stats(results)
    assert stats["labeling_failure_rate"] > 0.5, (
        f"SVAMP labeling failure rate {stats['labeling_failure_rate']:.2f} <= 0.5: "
        "FoVer is somehow labeling SVAMP single-step responses."
    )


def test_mismatch_gate_confirms_hypothesis() -> None:
    """All three gate conditions must hold, yielding labeling_mismatch_confirmed=True."""
    svamp_results = analyze_cohort(SVAMP_QUESTIONS, SVAMP_RESPONSES, "svamp")
    gsm8k_results = analyze_cohort(GSM8K_QUESTIONS, GSM8K_RESPONSES, "gsm8k")
    svamp_stats = compute_cohort_stats(svamp_results)
    gsm8k_stats = compute_cohort_stats(gsm8k_results)

    confirmed = check_mismatch_confirmed(
        mean_cot_depth_svamp=svamp_stats["mean_cot_depth"],
        mean_cot_depth_gsm8k=gsm8k_stats["mean_cot_depth"],
        labeling_failure_rate_svamp=svamp_stats["labeling_failure_rate"],
    )
    assert confirmed is True, "Mismatch gate conditions not all satisfied."


def test_honest_verdict_is_gate_open() -> None:
    """honest_verdict must be 'mismatch_confirmed_gate_open' when mismatch confirmed."""
    verdict = assign_honest_verdict(True)
    assert verdict == "mismatch_confirmed_gate_open"


def test_honest_verdict_is_investigate_when_not_confirmed() -> None:
    """honest_verdict must be 'mismatch_unconfirmed_investigate_further' otherwise."""
    verdict = assign_honest_verdict(False)
    assert verdict == "mismatch_unconfirmed_investigate_further"


# ---------------------------------------------------------------------------
# VJEPA AUC (REQ-VER-085)
# ---------------------------------------------------------------------------


def test_svamp_vjepa_auc_is_degenerate() -> None:
    """SVAMP VJEPA AUC must be 0.5 (chance) due to all-noise FoVer labels."""
    results = analyze_cohort(SVAMP_QUESTIONS, SVAMP_RESPONSES, "svamp")
    auc = compute_vjepa_auc_on_labeled(results)
    # All SVAMP labels are noise (not_verifiable), so VJEPA has no training signal.
    # AUC must be exactly 0.5 (the degenerate fallback).
    assert auc == 0.5, f"Expected SVAMP AUC=0.5, got {auc:.4f}"


def test_svamp_auc_post_filter_degenerate() -> None:
    """Post-filter SVAMP AUC must also be 0.5 (no high-confidence pairs survive)."""
    results = analyze_cohort(SVAMP_QUESTIONS, SVAMP_RESPONSES, "svamp")
    auc = compute_svamp_auc_post_filter(results)
    assert auc == 0.5, f"Expected post-filter SVAMP AUC=0.5, got {auc:.4f}"


def test_compute_svamp_auc_post_filter_empty_input() -> None:
    """compute_svamp_auc_post_filter must return 0.5 for an empty input list."""
    auc = compute_svamp_auc_post_filter([])
    assert auc == 0.5


def test_compute_svamp_auc_post_filter_single_item() -> None:
    """compute_svamp_auc_post_filter must return 0.5 for a single labeled pair."""
    single = [
        LabelingResult(
            question_id="svamp_00",
            n_cot_steps=1,
            labeling_successful=True,
            label_value=0,
            label_confidence=0.9,
            domain="svamp",
        )
    ]
    auc = compute_svamp_auc_post_filter(single)
    assert auc == 0.5


def test_compute_svamp_auc_post_filter_all_same_label() -> None:
    """compute_svamp_auc_post_filter returns 0.5 when all labels are identical."""
    pairs = [
        LabelingResult(
            question_id=f"svamp_{i:02d}",
            n_cot_steps=1,
            labeling_successful=True,
            label_value=0,
            label_confidence=0.9,
            domain="svamp",
        )
        for i in range(4)
    ]
    auc = compute_svamp_auc_post_filter(pairs)
    assert auc == 0.5


# ---------------------------------------------------------------------------
# Deliverable schema validation (REQ-VER-085)
# ---------------------------------------------------------------------------


def test_run_experiment_returns_all_required_fields() -> None:
    """run_experiment() must return a dict containing every required schema field."""
    artifact = run_experiment()
    missing = _REQUIRED_FIELDS - set(artifact.keys())
    assert not missing, f"Artifact missing required fields: {missing}"


def test_run_experiment_correct_experiment_id() -> None:
    """Artifact must identify itself as experiment 907."""
    artifact = run_experiment()
    assert artifact["experiment"] == 907


def test_run_experiment_mismatch_confirmed() -> None:
    """run_experiment() artifact must have labeling_mismatch_confirmed=True."""
    artifact = run_experiment()
    assert artifact["labeling_mismatch_confirmed"] is True


def test_run_experiment_schema_field() -> None:
    """Artifact schema field must be 'carnot-experiment-v1'."""
    artifact = run_experiment()
    assert artifact["schema"] == "carnot-experiment-v1"


def test_assert_deliverable_written_passes() -> None:
    """assert_deliverable_written() must pass if the deliverable JSON exists."""
    # This test is meaningful only if the result file was already written by
    # the experiment's __main__ entry point.  When run in CI after the script
    # has executed, RESULT_PATH exists and validation passes.
    if not RESULT_PATH.exists():
        pytest.skip("Deliverable not yet written — run the experiment script first.")
    assert_deliverable_written()
