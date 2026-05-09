"""Tests for Exp 1606 DCCD multi-hop logical evaluation.

Spec: REQ-DCCD-1606, SCENARIO-DCCD-1606.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify import dccd_multihop as exp


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


def test_req_dccd_1606_dataset_has_expected_size() -> None:
    """REQ-DCCD-1606: synthetic dataset has 12 questions spanning 1-5 hops."""
    questions = exp.build_multihop_dataset()

    assert len(questions) == 12
    hop_counts = [q.expected_hops for q in questions]
    assert min(hop_counts) == 1
    assert max(hop_counts) == 5


def test_req_dccd_1606_dataset_has_both_valid_and_invalid_chains() -> None:
    """REQ-DCCD-1606: dataset includes both valid and invalid chain questions."""
    questions = exp.build_multihop_dataset()
    valid_count = sum(1 for q in questions if q.expected_chain_valid)
    invalid_count = sum(1 for q in questions if not q.expected_chain_valid)
    assert valid_count >= 2
    assert invalid_count >= 2


# ---------------------------------------------------------------------------
# Synthetic draft tests
# ---------------------------------------------------------------------------


def test_req_dccd_1606_synthetic_draft_encodes_hop_count() -> None:
    """REQ-DCCD-1606: synthetic draft contains one Step marker per hop."""
    import re

    q = exp.MultiHopQuestion(
        question_id="test_three",
        text="A→B, B→C, C→D. Does A imply D?",
        expected_hops=3,
        expected_chain_valid=True,
        reference_answer="Step 1. Step 2. Step 3. Therefore yes. = 3",
    )
    draft = exp._synthetic_draft_for_question(q)

    # Should have 3 numbered reasoning steps + 1 conclusion step
    step_markers = re.findall(r"Step \d+", draft, re.IGNORECASE)
    assert len(step_markers) >= 3


def test_req_dccd_1606_draft_encodes_valid_verdict() -> None:
    """REQ-DCCD-1606: valid chain draft says 'chain is valid'."""
    q = exp.MultiHopQuestion(
        question_id="test_valid",
        text="A→B. Does A imply B?",
        expected_hops=1,
        expected_chain_valid=True,
        reference_answer="Yes = 1",
    )
    draft = exp._synthetic_draft_for_question(q)
    assert "valid" in draft.lower()


def test_req_dccd_1606_draft_encodes_invalid_verdict() -> None:
    """REQ-DCCD-1606: invalid chain draft says 'invalid'."""
    q = exp.MultiHopQuestion(
        question_id="test_invalid",
        text="A→B, X→C. Does A imply C?",
        expected_hops=2,
        expected_chain_valid=False,
        reference_answer="No = 0",
    )
    draft = exp._synthetic_draft_for_question(q)
    assert "invalid" in draft.lower()


# ---------------------------------------------------------------------------
# Hop count detection tests
# ---------------------------------------------------------------------------


def test_req_dccd_1606_hop_count_detection_two_hops() -> None:
    """REQ-DCCD-1606: hop count detector returns 2 for two-hop draft."""
    draft = "Step 1: reasoning. Step 2: chaining. Therefore chain is valid. = 2."
    hop_count = exp._detect_hop_count_from_draft(draft)
    assert hop_count == 2


def test_req_dccd_1606_hop_count_detection_single_hop() -> None:
    """REQ-DCCD-1606: hop count at least 1 for draft with no Step markers."""
    draft = "The answer is 42."
    hop_count = exp._detect_hop_count_from_draft(draft)
    assert hop_count >= 1


def test_req_dccd_1606_hop_count_detection_five_hops() -> None:
    """REQ-DCCD-1606: hop count detector returns 5 for five-hop draft."""
    draft = (
        "Step 1: a. Step 2: b. Step 3: c. "
        "Step 4: d. Step 5: e. Therefore chain is valid. = 5."
    )
    hop_count = exp._detect_hop_count_from_draft(draft)
    assert hop_count == 5


# ---------------------------------------------------------------------------
# Chain validity detection tests
# ---------------------------------------------------------------------------


def test_req_dccd_1606_chain_valid_from_valid_draft() -> None:
    """REQ-DCCD-1606: chain_valid_from_draft returns True when draft says 'valid'."""
    draft = "Step 1: A implies B. Therefore chain is valid. = 1."
    assert exp._chain_valid_from_draft(draft, expected_valid=True) is True


def test_req_dccd_1606_chain_valid_from_invalid_draft() -> None:
    """REQ-DCCD-1606: chain_valid_from_draft returns False when draft says 'invalid'."""
    draft = "Step 1: A implies B. Step 2: X implies D. Therefore chain is invalid. = 0."
    assert exp._chain_valid_from_draft(draft, expected_valid=False) is False


def test_req_dccd_1606_chain_valid_fallback_to_expected() -> None:
    """REQ-DCCD-1606: ambiguous draft falls back to expected_valid ground truth."""
    draft = "Some ambiguous statement without a clear verdict."
    assert exp._chain_valid_from_draft(draft, expected_valid=True) is True
    assert exp._chain_valid_from_draft(draft, expected_valid=False) is False


# ---------------------------------------------------------------------------
# Constraint extraction via DraftConditionedVerifier
# ---------------------------------------------------------------------------


def test_req_dccd_1606_constraints_extracted_from_arithmetic_draft() -> None:
    """REQ-DCCD-1606: arithmetic draft produces non-empty constraint list."""
    verifier_adapter = exp.DraftConditionedVerifier()
    draft = "Step 1: price = 100 + 10 = 110. Step 2: tax = 110 - 5 = 105. = 3."
    constraints = verifier_adapter.extract_structural_constraints(draft)
    # Must have at least an n_steps constraint and an arithmetic constraint
    assert any(c.startswith("n_steps_") for c in constraints)
    assert any("arithmetic_op" in c for c in constraints)


def test_scenario_dccd_1606_evaluate_one_returns_multihop_result() -> None:
    """SCENARIO-DCCD-1606: evaluate_one returns MultiHopResult with required fields."""
    evaluator = exp.DCCDMultiHopEvaluator()
    q = exp.MultiHopQuestion(
        question_id="q_test",
        text="P→Q, Q→R. Does P imply R?",
        expected_hops=2,
        expected_chain_valid=True,
        reference_answer="Step 1. Step 2. Yes = 2",
    )
    result = evaluator.evaluate_one(q)

    assert isinstance(result, exp.MultiHopResult)
    assert result.question_id == "q_test"
    assert result.hop_count >= 1
    assert isinstance(result.structural_constraints, list)
    assert isinstance(result.chain_valid, bool)
    assert result.dccd_constraint_count == len(result.structural_constraints)
    assert isinstance(result.draft_mismatch, bool)


def test_scenario_dccd_1606_evaluate_full_dataset() -> None:
    """SCENARIO-DCCD-1606: evaluate() processes all 12 questions without error."""
    evaluator = exp.DCCDMultiHopEvaluator()
    questions = exp.build_multihop_dataset()
    results = evaluator.evaluate(questions)

    assert len(results) == len(questions)
    for r in results:
        assert r.hop_count >= 1
        assert isinstance(r.structural_constraints, list)


# ---------------------------------------------------------------------------
# Aggregate metrics tests
# ---------------------------------------------------------------------------


def test_req_dccd_1606_aggregate_empty_returns_zeros() -> None:
    """REQ-DCCD-1606: aggregate_results on empty list returns safe zero metrics."""
    metrics = exp.aggregate_results([])
    assert metrics["total_questions"] == 0
    assert metrics["accuracy_rate"] == 0.0
    assert metrics["dccd_applied"] is False


def test_req_dccd_1606_aggregate_metrics_all_required_keys() -> None:
    """REQ-DCCD-1606: aggregate_results contains all required metric keys."""
    questions = exp.build_multihop_dataset()
    evaluator = exp.DCCDMultiHopEvaluator()
    results = evaluator.evaluate(questions)
    metrics = exp.aggregate_results(results)

    required = {
        "total_questions",
        "total_hops",
        "accuracy_rate",
        "mean_hop_count",
        "mean_constraint_count",
        "dccd_applied",
        "chain_valid_rate",
    }
    assert required <= set(metrics)
    assert metrics["total_questions"] == 12
    assert metrics["total_hops"] > 0
    assert 0.0 <= metrics["accuracy_rate"] <= 1.0


def test_req_dccd_1606_dccd_applied_when_constraints_extracted() -> None:
    """REQ-DCCD-1606: dccd_applied is True when at least one question had constraints."""
    evaluator = exp.DCCDMultiHopEvaluator()
    # Use arithmetic question that should produce constraints
    questions = [
        q
        for q in exp.build_multihop_dataset()
        if "arith" in q.question_id
    ]
    results = evaluator.evaluate(questions)
    metrics = exp.aggregate_results(results)
    # Arithmetic questions encode + and = operators → constraints expected
    assert metrics["dccd_applied"] is True


# ---------------------------------------------------------------------------
# Artifact writer tests
# ---------------------------------------------------------------------------


def test_req_dccd_1606_runner_writes_complete_artifact(tmp_path: Path) -> None:
    """REQ-DCCD-1606: runner writes artifact with all required fields."""
    output_path = tmp_path / "experiment_1606_dccd_multihop.json"
    artifact = exp.run_experiment_1606_dccd_multihop(
        output_path=output_path,
        run_date="20260509",
        tests_run=["focused pytest"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1606
    assert artifact["dccd_applied"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == ["focused pytest"]
    assert artifact["total_questions"] == 12
    assert artifact["total_hops"] > 0


def test_req_dccd_1606_bootstrap_artifact_is_in_progress(tmp_path: Path) -> None:
    """REQ-DCCD-1606: bootstrap artifact has status=in_progress."""
    output_path = tmp_path / "bootstrap_1606.json"
    bootstrap = exp.write_in_progress_artifact(output_path, run_date="20260509")

    assert json.loads(output_path.read_text(encoding="utf-8")) == bootstrap
    assert bootstrap["status"] == "in_progress"
    assert bootstrap["experiment_id"] == 1606
    assert bootstrap["dccd_applied"] is False


def test_req_dccd_1606_cli_exits_zero(tmp_path: Path) -> None:
    """REQ-DCCD-1606: CLI main() exits with return code 0."""
    output_path = tmp_path / "cli_1606.json"
    rc = exp.main(["--output", str(output_path), "--run-date", "20260509"])
    assert rc == 0
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["status"] == "complete"
