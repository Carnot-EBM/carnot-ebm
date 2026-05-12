"""Tests for Exp 1879 BEAVER-lite ROCE validator-tree bounds.

Spec: REQ-VERIFY-1879, SCENARIO-VERIFY-1879.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline import roce_validator_tree
from carnot.pipeline.extract import ConstraintResult
from carnot.verify import roce_beaver_lite_bounds as mod


PROMPT = (
    "Return a single-line JSON object only. "
    'Use strict key order {"answer": ..., "sum": ...} and no other top-level keys. '
    'Include "approved". Do not mention "secret". Keep under 20 words. '
    "The response must state 2 + 3 = 5. "
    'If the response contains "approved", it must also contain "audited".'
)


def test_req_verify_1879_computes_per_leaf_bounds_for_supported_tree() -> None:
    """REQ-VERIFY-1879: each executable ROCE leaf receives coverage weight."""

    tree = roce_validator_tree.compile_roce_validator_tree(PROMPT, case_id="supported")
    summary = mod.compute_tree_bounds(tree)
    payload = summary.to_dict()

    assert payload["beaver_lite_bounds_ready"] is True
    assert payload["deterministic_coverage_bound"] == pytest.approx(1.0)
    assert payload["residual_risk_bound"] == pytest.approx(0.0)
    assert len(payload["bound_rows"]) == tree.supported_constraint_count
    assert {
        row["leaf_id"] for row in payload["bound_rows"] if row["executable_validator_present"]
    } == {leaf.id for leaf in tree.leaves}
    assert all(
        row["deterministic_coverage_bound"] == pytest.approx(1.0 / tree.total_constraint_count)
        for row in payload["bound_rows"]
    )
    assert all(row["residual_risk_bound"] == pytest.approx(0.0) for row in payload["bound_rows"])


def test_req_verify_1879_unsupported_constraints_become_residual_risk() -> None:
    """REQ-VERIFY-1879: unsupported ROCE categories are residual, not accepted."""

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
    tree = roce_validator_tree.compile_roce_validator_tree([supported, unsupported])

    summary = mod.compute_tree_bounds(tree)
    payload = summary.to_dict()

    assert payload["deterministic_coverage_bound"] == pytest.approx(0.5)
    assert payload["residual_risk_bound"] == pytest.approx(0.5)
    assert payload["beaver_lite_bounds_ready"] is True
    unsupported_rows = [
        row for row in payload["bound_rows"] if row["bound_source"] == "unsupported_constraint"
    ]
    assert unsupported_rows == [
        {
            "leaf_id": "unsupported:semantic_tone",
            "predicate": "semantic_tone",
            "source_constraint_type": "unsupported",
            "deterministic_coverage_bound": 0.0,
            "residual_risk_bound": 0.5,
            "executable_validator_present": False,
            "guarded": False,
            "bound_source": "unsupported_constraint",
        }
    ]
    assert tree.validate("alpha").accepted is False


def test_scenario_verify_1879_bounds_do_not_promote_invalid_output() -> None:
    """SCENARIO-VERIFY-1879: invalid outputs remain failing VerdictRecords."""

    tree = roce_validator_tree.compile_roce_validator_tree(PROMPT)
    record = mod.verdict_record_for_output(
        tree,
        '{"answer":"approved","sum":"2 + 3 = 5"}',
    )
    payload = record.to_dict()

    assert payload["verdict"] == "fail"
    assert payload["extras"]["tree_validation_accepted"] is False
    assert payload["extras"]["acceptance_authority"] == "roce_validator_tree"
    assert payload["extras"]["acceptance_authority_unchanged"] is True
    assert payload["extras"]["beaver_lite_bounds"]["deterministic_coverage_bound"] == pytest.approx(
        1.0
    )
    assert payload["extras"]["beaver_lite_bounds"]["residual_risk_bound"] == pytest.approx(0.0)
    assert "c008-conditional_required_text" in payload["extras"]["failure_ids"]


def test_req_verify_1879_run_experiment_writes_required_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1879: Exp 1879 writes the required BEAVER-lite bound artifact."""

    output_path = tmp_path / "results" / "experiment_1879_beaver_lite_bounds.json"
    tests_run = [".venv/bin/pytest tests/python/test_roce_beaver_lite_bounds.py -q"]

    artifact = mod.run_experiment(output_path=output_path, tests_run=tests_run)

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["beaver_lite_bounds_ready"] is True
    assert artifact["deterministic_coverage_bound"] == pytest.approx(1.0)
    assert artifact["residual_risk_bound"] == pytest.approx(0.0)
    assert artifact["acceptance_authority_unchanged"] is True
    assert artifact["tests_run"] == tests_run
    assert artifact["case_results"][0]["known_bad_promoted_by_bounds"] == 0
    assert artifact["verdict_records"][0]["verdict"] == "pass"
    assert {record["verdict"] for record in artifact["verdict_records"][1:]} == {"fail"}


def test_req_verify_1879_artifact_validation_rejects_impossible_complete() -> None:
    """REQ-VERIFY-1879: complete artifacts require ready bounds and authority."""

    artifact = mod.build_artifact(tests_run=["focused"])
    mod.validate_artifact(artifact)

    with pytest.raises(AssertionError, match="complete requires ready"):
        mod.validate_artifact(dict(artifact, beaver_lite_bounds_ready=False))
    with pytest.raises(AssertionError, match="coverage out of range"):
        mod.validate_artifact(dict(artifact, deterministic_coverage_bound=1.1))
    with pytest.raises(AssertionError, match="authority"):
        mod.validate_artifact(dict(artifact, acceptance_authority_unchanged=False))


def test_req_verify_1879_fixture_loader_falls_back_to_reconstructed_cases(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1879: missing or malformed Exp 1878 artifacts reconstruct fixtures."""

    missing = mod.load_or_reconstruct_exp1878_fixture_cases(tmp_path / "missing.json")
    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text("{not json", encoding="utf-8")
    mismatched_path = tmp_path / "mismatched.json"
    mismatched_path.write_text(
        json.dumps({"case_results": [{"case_id": "not-an-exp1878-fixture"}]}),
        encoding="utf-8",
    )

    assert missing == roce_validator_tree.default_roce_fixture_cases()
    assert mod.load_or_reconstruct_exp1878_fixture_cases(bad_json_path) == missing
    assert mod.load_or_reconstruct_exp1878_fixture_cases(mismatched_path) == missing


def test_req_verify_1879_partial_artifact_and_json_safety() -> None:
    """REQ-VERIFY-1879: partial artifacts stay honest and JSON coercion is stable."""

    artifact = mod.build_artifact(cases=[], tests_run=[])

    assert artifact["status"] == "partial"
    assert artifact["beaver_lite_bounds_ready"] is False
    assert artifact["honest_verdict"].startswith("partial:")
    assert mod._json_safe(("x", float("inf"), object()))[:2] == ["x", None]
