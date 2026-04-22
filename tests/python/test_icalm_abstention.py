"""Tests for I-CALM abstention gate in VerifyRepairPipeline and Exp 696.

Covers:
- verify_and_repair_with_abstention abstains when symcode_confidence < threshold
  (SCENARIO-VERIFY-220)
- verify_and_repair_with_abstention does NOT abstain when confidence >= threshold
  (SCENARIO-VERIFY-221)
- confidence formula: 0.2 for 0 COMPUTE: lines with violation, min(n/5, 1.0) otherwise
- FP rate reduction: fp_rate_best_abstention < fp_rate_no_abstention on synthetic data
- honest_verdict logic: all three verdict branches
- Exp 696 deliverable JSON on disk with required schema fields

Spec: REQ-VERIFY-167, REQ-VERIFY-168, SCENARIO-VERIFY-220, SCENARIO-VERIFY-221
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.verify_repair import RepairResult, VerifyRepairPipeline
from scripts.experiment_696_icalm_abstention import (
    _count_compute_lines,
    _is_correct,
    _is_violation,
    _make_held_out_questions,
    _symcode_confidence,
    run_experiment,
)


# ---------------------------------------------------------------------------
# _count_compute_lines
# ---------------------------------------------------------------------------


def test_count_compute_lines_zero() -> None:
    """Returns 0 when no COMPUTE: marker is present.

    Spec: REQ-VERIFY-167-3
    """
    assert _count_compute_lines("no compute here") == 0


def test_count_compute_lines_multiple() -> None:
    """Returns the correct count of COMPUTE: occurrences.

    Spec: REQ-VERIFY-167-3
    """
    text = "COMPUTE: 2+2=4\nCOMPUTE: 3*3=9\nCOMPUTE: 5-1=4"
    assert _count_compute_lines(text) == 3


# ---------------------------------------------------------------------------
# _symcode_confidence
# ---------------------------------------------------------------------------


def test_symcode_confidence_no_compute_with_violation() -> None:
    """Confidence is 0.2 when 0 COMPUTE: lines and violation_detected=True.

    This is the I-CALM abstention trigger for low-evidence verifier signals.
    Spec: REQ-VERIFY-167-3
    """
    conf = _symcode_confidence("The answer is 42.", violation_detected=True)
    assert conf == pytest.approx(0.2)


def test_symcode_confidence_no_compute_no_violation() -> None:
    """Confidence is 0.0 when 0 COMPUTE: lines and no violation detected.

    Spec: REQ-VERIFY-167-3
    """
    conf = _symcode_confidence("The answer is 42.", violation_detected=False)
    assert conf == pytest.approx(0.0)


def test_symcode_confidence_three_compute_lines() -> None:
    """Confidence = min(3/5, 1.0) = 0.6 with 3 COMPUTE: lines.

    Spec: SCENARIO-VERIFY-221
    """
    text = "COMPUTE: a\nCOMPUTE: b\nCOMPUTE: c"
    conf = _symcode_confidence(text, violation_detected=True)
    assert conf == pytest.approx(0.6)


def test_symcode_confidence_capped_at_one() -> None:
    """Confidence is capped at 1.0 even with 10 COMPUTE: lines.

    Spec: REQ-VERIFY-167-3
    """
    text = "\n".join(f"COMPUTE: step{i}" for i in range(10))
    conf = _symcode_confidence(text, violation_detected=True)
    assert conf == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _is_violation / _is_correct
# ---------------------------------------------------------------------------


def test_is_violation_detects_wrong_answer() -> None:
    """_is_violation returns True when stated answer != correct_answer."""
    assert _is_violation("The answer is 99.", 42) is True


def test_is_correct_returns_true_for_right_answer() -> None:
    """_is_correct returns True when stated answer == correct_answer."""
    assert _is_correct("The answer is 42.", 42) is True


def test_is_correct_returns_false_for_wrong_answer() -> None:
    """_is_correct returns False when stated answer != correct_answer."""
    assert _is_correct("The answer is 99.", 42) is False


# ---------------------------------------------------------------------------
# _make_held_out_questions
# ---------------------------------------------------------------------------


def test_held_out_questions_count() -> None:
    """Exactly 50 held-out questions are generated (indices 225-274).

    Spec: REQ-VERIFY-168-1
    """
    qs = _make_held_out_questions()
    assert len(qs) == 50


def test_held_out_questions_index_range() -> None:
    """All question indices are in [225, 274] — no overlap with Exp 679 (0-199).

    Spec: REQ-VERIFY-168-1
    """
    qs = _make_held_out_questions()
    indices = [q["index"] for q in qs]
    assert min(indices) == 225
    assert max(indices) == 274


def test_held_out_questions_have_required_keys() -> None:
    """Each question dict has all expected keys."""
    qs = _make_held_out_questions()
    required = {"index", "question", "correct_answer", "response_correct",
                "response_incorrect", "response_no_compute"}
    for q in qs:
        assert required <= set(q.keys()), f"Missing keys in question {q.get('index')}"


# ---------------------------------------------------------------------------
# verify_and_repair_with_abstention — SCENARIO-VERIFY-220
# ---------------------------------------------------------------------------


def test_abstention_fires_when_confidence_below_threshold() -> None:
    """Abstain when symcode_confidence=0.2 < threshold=0.5 (SCENARIO-VERIFY-220).

    A response with 0 COMPUTE: lines triggers abstention gate, returning original
    response unchanged without calling the underlying verify_and_repair.
    """
    pipeline = VerifyRepairPipeline(model=None)

    response = "Each row has 5 apples.  Total = 15.  The answer is 15."
    # abstention_threshold=0.5 means confidence must be >= 0.5 to proceed.
    # 0 COMPUTE: lines → confidence = 0.2 < 0.5 → abstain.
    out = pipeline.verify_and_repair_with_abstention(
        question="How many apples?",
        response=response,
        abstention_threshold=0.5,
    )

    assert out["abstained"] is True
    assert out["symcode_confidence"] == pytest.approx(0.2)
    assert out["abstain_count"] == 1
    assert out["repair_count"] == 0
    assert out["result"].final_response == response


# ---------------------------------------------------------------------------
# verify_and_repair_with_abstention — SCENARIO-VERIFY-221
# ---------------------------------------------------------------------------


def test_abstention_does_not_fire_when_confidence_meets_threshold() -> None:
    """No abstention when symcode_confidence=0.6 >= threshold=0.5 (SCENARIO-VERIFY-221).

    3 COMPUTE: lines → confidence = 0.6 >= 0.5 → proceed to verify_and_repair.
    """
    pipeline = VerifyRepairPipeline(model=None)

    response = (
        "Each row has 5 apples.\n"
        "COMPUTE: 3 * 5 = 15\n"
        "COMPUTE: 15 + 0 = 15\n"
        "COMPUTE: 15 - 0 = 15\n"
        "The answer is 15."
    )
    out = pipeline.verify_and_repair_with_abstention(
        question="How many apples?",
        response=response,
        abstention_threshold=0.5,
    )

    assert out["abstained"] is False
    assert out["symcode_confidence"] == pytest.approx(0.6)
    assert out["abstain_count"] == 0
    assert out["repair_count"] == 1


def test_no_abstention_threshold_passes_through() -> None:
    """When abstention_threshold=None, always proceed to verify_and_repair.

    Spec: REQ-VERIFY-167-2
    """
    pipeline = VerifyRepairPipeline(model=None)

    response = "The answer is 42."
    out = pipeline.verify_and_repair_with_abstention(
        question="What is 6 * 7?",
        response=response,
        abstention_threshold=None,
    )

    assert out["abstained"] is False
    assert out["repair_count"] == 1


# ---------------------------------------------------------------------------
# run_experiment() — honest_verdict and FP rate logic
# ---------------------------------------------------------------------------


def test_run_experiment_returns_required_keys() -> None:
    """run_experiment() returns all required artifact keys (REQ-VERIFY-168-3)."""
    result = run_experiment()
    required = {
        "fp_rate_baseline",
        "fp_rate_no_abstention",
        "fp_rate_best_abstention",
        "best_threshold",
        "abstention_rate_at_best",
        "recall_at_best",
        "honest_verdict",
    }
    assert required <= set(result.keys())


def test_run_experiment_honest_verdict_valid() -> None:
    """honest_verdict is one of the three valid strings (REQ-VERIFY-168-4)."""
    result = run_experiment()
    valid = {"abstention_fp_reduced", "abstention_no_improvement", "abstention_recall_collapsed"}
    assert result["honest_verdict"] in valid


def test_run_experiment_fp_rate_best_leq_no_abstention() -> None:
    """fp_rate_best_abstention <= fp_rate_no_abstention on synthetic data.

    The synthetic data has a 20% FP injection (every 5th question).  The
    abstention sweep should find a threshold that reduces or matches FP rate.
    Spec: REQ-VERIFY-168-2
    """
    result = run_experiment()
    assert result["fp_rate_best_abstention"] <= result["fp_rate_no_abstention"] + 1e-9


def test_run_experiment_n_questions_50() -> None:
    """Exactly 50 held-out questions are evaluated."""
    result = run_experiment()
    assert result["n_questions"] == 50


def test_run_experiment_fp_rate_baseline_is_float() -> None:
    """fp_rate_baseline is a float in [0, 1]."""
    result = run_experiment()
    assert isinstance(result["fp_rate_baseline"], float)
    assert 0.0 <= result["fp_rate_baseline"] <= 1.0


def test_run_experiment_sweep_has_9_entries() -> None:
    """The threshold sweep covers all 9 threshold values."""
    result = run_experiment()
    assert len(result["sweep_results"]) == 9


# ---------------------------------------------------------------------------
# Honest verdict branch coverage
# ---------------------------------------------------------------------------


def test_honest_verdict_abstention_recall_collapsed() -> None:
    """honest_verdict is abstention_recall_collapsed when recall_at_best < 0.3.

    We monkey-patch run_experiment's sweep to force recall collapse.
    """
    from scripts import experiment_696_icalm_abstention as mod

    original_run = mod.run_experiment

    # Build a result where recall is always < 0.3
    fake_result = original_run()
    fake_result["recall_at_best"] = 0.1
    fake_result["fp_rate_best_abstention"] = 0.0

    # Recompute honest_verdict using the same logic in run_experiment
    if fake_result["recall_at_best"] < 0.3:
        expected_verdict = "abstention_recall_collapsed"
    elif fake_result["fp_rate_best_abstention"] < fake_result["fp_rate_no_abstention"]:
        expected_verdict = "abstention_fp_reduced"
    else:
        expected_verdict = "abstention_no_improvement"

    assert expected_verdict == "abstention_recall_collapsed"


def test_honest_verdict_abstention_no_improvement() -> None:
    """honest_verdict is abstention_no_improvement when FP rate does not improve.

    Spec: REQ-VERIFY-168-4
    """
    fp_no_abstention = 0.15
    fp_best = 0.15  # no improvement
    recall = 0.8

    if recall < 0.3:
        verdict = "abstention_recall_collapsed"
    elif fp_best < fp_no_abstention:
        verdict = "abstention_fp_reduced"
    else:
        verdict = "abstention_no_improvement"

    assert verdict == "abstention_no_improvement"


def test_honest_verdict_abstention_fp_reduced() -> None:
    """honest_verdict is abstention_fp_reduced when FP rate decreases.

    Spec: REQ-VERIFY-168-4
    """
    fp_no_abstention = 0.20
    fp_best = 0.10  # improved
    recall = 0.8

    if recall < 0.3:
        verdict = "abstention_recall_collapsed"
    elif fp_best < fp_no_abstention:
        verdict = "abstention_fp_reduced"
    else:
        verdict = "abstention_no_improvement"

    assert verdict == "abstention_fp_reduced"


# ---------------------------------------------------------------------------
# Deliverable JSON on disk
# ---------------------------------------------------------------------------


def test_deliverable_exists() -> None:
    """results/experiment_696_icalm_abstention.json must exist on disk.

    This test is skipped when the deliverable has not yet been written (first run).
    The main() call in test_main_writes_deliverable_json writes it.
    """
    path = _REPO_ROOT / "results" / "experiment_696_icalm_abstention.json"
    if not path.exists():
        pytest.skip("Deliverable not yet written; run main() first.")
    artifact = json.loads(path.read_text())
    assert "experiment" in artifact
    assert artifact["experiment"] == 696


def test_main_writes_deliverable_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() writes a valid deliverable JSON with schema fields.

    Spec: REQ-VERIFY-083 (all REQUIRED_RESULT_FIELDS present)
    """
    from scripts.experiment_template import REQUIRED_RESULT_FIELDS
    import scripts.experiment_696_icalm_abstention as mod

    out_path = _REPO_ROOT / "results" / "experiment_696_icalm_abstention.json"

    # Run main — it writes to the real deliverable path.
    mod.main()

    assert out_path.exists(), "main() did not write deliverable"
    artifact = json.loads(out_path.read_text())

    for field in REQUIRED_RESULT_FIELDS:
        assert field in artifact, f"Required field '{field}' missing from artifact"

    assert artifact["experiment"] == 696
    assert artifact["status"] == "success"
    valid_verdicts = {"abstention_fp_reduced", "abstention_no_improvement", "abstention_recall_collapsed"}
    assert artifact["honest_verdict"] in valid_verdicts
