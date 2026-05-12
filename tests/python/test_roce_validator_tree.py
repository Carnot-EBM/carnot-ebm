"""Tests for the Exp 1878 ROCE validator-tree compiler.

Spec: REQ-VERIFY-1878, SCENARIO-VERIFY-1878.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline import roce_validator_tree as mod


PROMPT = (
    "Return a single-line JSON object only. "
    'Use strict key order {"answer": ..., "sum": ...} and no other top-level keys. '
    'Include "approved". Do not mention "secret". Keep under 20 words. '
    "The response must state 2 + 3 = 5. "
    'If the response contains "approved", it must also contain "audited".'
)
VALID_OUTPUT = '{"answer":"approved audited","sum":"2 + 3 = 5"}'


def test_req_verify_1878_compiles_roce_constraints_to_backend_tree() -> None:
    """REQ-VERIFY-1878: ROCE constraints compile to Python/PySAT/Z3 leaves."""

    tree = mod.compile_roce_validator_tree(PROMPT, case_id="all-supported")
    tree_dict = tree.to_dict()

    assert tree_dict["compiled_backends"] == ["python", "pysat_cnf", "z3"]
    assert tree_dict["constraint_coverage_rate"] == pytest.approx(1.0)
    assert tree_dict["unsupported_constraint_types"] == []
    assert {
        "format_json",
        "json_required_keys",
        "required_text",
        "forbidden_text",
        "word_count_at_most",
        "single_line",
        "arithmetic_equality",
        "conditional_required_text",
    } <= {leaf["predicate"] for leaf in tree_dict["leaves"]}
    assert any(leaf["guard"] for leaf in tree_dict["leaves"])
    assert tree_dict["pysat_problem"]["clauses"]
    assert tree_dict["z3_problem"]["assertions"]


def test_scenario_verify_1878_validates_good_and_adversarial_outputs() -> None:
    """SCENARIO-VERIFY-1878: adversarial invalid outputs have zero false accepts."""

    tree = mod.compile_roce_validator_tree(PROMPT)
    valid = tree.validate(VALID_OUTPUT)

    assert valid.accepted is True
    assert valid.failure_ids == []

    invalid_outputs = [
        '{"answer":"approved","sum":"2 + 3 = 5"}',
        '{"sum":"2 + 3 = 5","answer":"approved audited"}',
        '{"answer":"approved audited secret","sum":"2 + 3 = 5"}',
        '{"answer":"approved audited","sum":"2 + 3 = 6"}',
        "approved audited 2 + 3 = 5",
    ]

    results = [tree.validate(output) for output in invalid_outputs]

    assert all(not result.accepted for result in results)
    assert sum(result.accepted for result in results) == 0
    assert any("conditional_required_text" in fid for fid in results[0].failure_ids)
    assert any("json_required_keys" in fid for fid in results[1].failure_ids)
    assert any("forbidden_text" in fid for fid in results[2].failure_ids)
    assert any("arithmetic_equality" in fid for fid in results[3].failure_ids)
    assert any("format_json" in fid for fid in results[4].failure_ids)


def test_req_verify_1878_conditional_guards_skip_inactive_leaf() -> None:
    """REQ-VERIFY-1878: guarded leaves are enforced only when the guard is active."""

    guarded = ConstraintResult(
        constraint_type="roce_conditional",
        description='If response contains "approved", it must also contain "audited"',
        metadata={
            "source": "roce",
            "predicate": "conditional_required_text",
            "arguments": {
                "guard": {"predicate": "contains_text", "term": "approved"},
                "then": {"predicate": "required_text", "term": "audited"},
            },
        },
    )
    tree = mod.compile_roce_validator_tree([guarded])

    inactive = tree.validate("plain response")
    active_missing = tree.validate("approved response")
    active_satisfied = tree.validate("approved audited response")

    assert inactive.accepted is True
    assert inactive.skipped_ids == ["c001-conditional_required_text"]
    assert active_missing.accepted is False
    assert active_missing.failure_ids == ["c001-conditional_required_text"]
    assert active_satisfied.accepted is True


def test_req_verify_1878_unsupported_constraints_are_recorded() -> None:
    """REQ-VERIFY-1878: unsupported categories reduce coverage and fail closed."""

    supported = ConstraintResult(
        constraint_type="roce_content",
        description="Response must contain alpha",
        metadata={
            "source": "roce",
            "predicate": "required_text",
            "arguments": {"term": "alpha"},
        },
    )
    unsupported = ConstraintResult(
        constraint_type="roce_semantic",
        description="Response must be semantically cheerful",
        metadata={
            "source": "roce",
            "predicate": "semantic_tone",
            "arguments": {"tone": "cheerful"},
        },
    )
    tree = mod.compile_roce_validator_tree([supported, unsupported])

    assert tree.constraint_coverage_rate == pytest.approx(0.5)
    assert tree.unsupported_constraint_types == ["semantic_tone"]
    assert tree.validate("alpha").accepted is False
    assert tree.validate("alpha").unsupported_constraint_types == ["semantic_tone"]


def test_req_verify_1878_artifact_writes_required_schema(tmp_path: Path) -> None:
    """REQ-VERIFY-1878: run_experiment writes the required Exp 1878 JSON."""

    output_path = tmp_path / "results" / "experiment_1878_roce_validator_tree.json"

    artifact = mod.run_experiment(
        output_path=output_path,
        tests_run=[".venv/bin/pytest tests/python/test_roce_validator_tree.py -q"],
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["validator_tree_compiler_ready"] is True
    assert artifact["zero_false_accepts"] is True
    assert artifact["false_accept_count"] == 0
    assert artifact["constraint_coverage_rate"] == pytest.approx(1.0)
    assert artifact["unsupported_constraint_types"] == []
    assert artifact["honest_verdict"].startswith("complete:")
