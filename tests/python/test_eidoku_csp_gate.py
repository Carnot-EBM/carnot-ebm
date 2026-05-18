"""Tests for the Eidoku CSP arithmetic gate.

Spec: REQ-VERIFY-2354, SCENARIO-VERIFY-2354
"""

from __future__ import annotations

import pytest

from carnot.verify.eidoku_csp import (
    EidokuCspGate,
    build_arithmetic_constraint_corpus,
    build_experiment_artifact,
    evaluate_corpus,
    safe_eval,
)


def test_safe_eval_accepts_arithmetic_constraints_and_blocks_calls() -> None:
    """REQ-VERIFY-2354: safe_eval evaluates arithmetic without unsafe syntax."""

    variables = {"x": 20.0, "y": 22.0, "total": 42.0}

    assert safe_eval("x + y == total", variables) is True
    assert safe_eval("(x + y) / 2", variables) == pytest.approx(21.0)
    assert safe_eval("__import__('os').system('true')", variables) is None
    assert safe_eval("(1).real == 1", variables) is None


def test_validate_reports_failed_and_unevaluable_constraints() -> None:
    """REQ-VERIFY-2354: violated or missing-variable constraints fail the gate."""

    gate = EidokuCspGate()

    passing = gate.validate("x = 20; y = 22; total = 42", ["x + y == total", "total <= 42"])
    assert passing == {"gate_passed": True, "violations": []}

    failing = gate.validate("x = 20; y = 21; total = 42", ["x + y == total", "total <= 42"])
    assert failing["gate_passed"] is False
    assert failing["violations"] == ["x + y == total"]

    missing = gate.validate("x = 20", ["x + z == 42"])
    assert missing["gate_passed"] is False
    assert missing["violations"] == ["x + z == 42"]


def test_arithmetic_constraint_corpus_has_50_seeded_examples() -> None:
    """SCENARIO-VERIFY-2354: the arithmetic corpus is deterministic and balanced."""

    corpus = build_arithmetic_constraint_corpus(seed=42)

    assert len(corpus) == 50
    assert sum(example.expected_gate_passed for example in corpus) == 25
    assert sum(not example.expected_gate_passed for example in corpus) == 25
    assert corpus == build_arithmetic_constraint_corpus(seed=42)


def test_corpus_accuracy_meets_eidoku_validation_gate() -> None:
    """SCENARIO-VERIFY-2354: accuracy is measured as correct gate classifications."""

    corpus = build_arithmetic_constraint_corpus(seed=42)
    metrics = evaluate_corpus(corpus)

    assert metrics["n_eval_examples"] == 50
    assert metrics["csp_gate_accuracy"] == pytest.approx(1.0)
    assert metrics["eidoku_gate_validated"] is True


def test_experiment_artifact_contains_required_fields_and_principles() -> None:
    """REQ-VERIFY-2354: the deliverable schema carries required fields."""

    artifact = build_experiment_artifact(seed=42)

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["eidoku_gate_validated"] is True
    assert artifact["csp_gate_accuracy"] == pytest.approx(1.0)
    assert artifact["n_eval_examples"] == 50
    assert artifact["random_seed"] == 42
    assert artifact["field_principles"]["honest_verdict"] == "Terminal-prefix required."
