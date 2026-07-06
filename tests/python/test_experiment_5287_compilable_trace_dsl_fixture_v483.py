"""Tests for Exp 5287 compilable trace DSL fixture.

Spec refs: REQ-VERIFY-5287, SCENARIO-VERIFY-5287.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5273_solver_fixture_rebuild_v482 as fixture_mod
from carnot import experiment_5287_compilable_trace_dsl_fixture_v483 as mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def _rows_by_type(rows: list[dict[str, object]], case_type: str) -> list[dict[str, object]]:
    return [row for row in rows if row["case_type"] == case_type]


def test_req_verify_5287_spec_declares_trace_dsl_contract() -> None:
    """REQ-VERIFY-5287: OpenSpec anchors the trace DSL fixture contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5287") : spec.index("### REQ-VERIFY-5272")]

    for marker in (
        "REQ-VERIFY-5287",
        "SCENARIO-VERIFY-5287",
        str(mod.RESULT_RELATIVE_PATH),
        "offline_deterministic_fixture_no_llm",
        "trace_dsl_ready",
        "dependency links",
        "format-valid-but-semantically-wrong",
        "localized repair",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5287_cases_reuse_exp5273_fixture_and_cover_families() -> None:
    """REQ-VERIFY-5287: trace records are built over the Exp 5273 solver cases."""

    exp5273_ids = {fixture.fixture_id for fixture in fixture_mod.fixture_set()}
    cases = mod.trace_cases()
    counts = mod.fixture_case_counts(cases)

    assert counts == {
        "positive": 6,
        "negative": 2,
        "malformed": 2,
        "semantic-error": 2,
        "repair": 2,
    }
    assert {case["fixture_id"] for case in cases} <= exp5273_ids
    assert {case["schema_version"] for case in cases} == {mod.TRACE_SCHEMA_VERSION}
    assert len({case["trace_id"] for case in cases}) == len(cases)

    positive_cases = [case for case in cases if case["case_type"] == "positive"]
    assert {case["fixture_id"] for case in positive_cases} == exp5273_ids
    for case in positive_cases:
        validation = mod.validate_trace_schema(case)
        assert validation.ok is True, validation.errors
        compiled = mod.compile_trace_to_constraint_ir(case)
        assert compiled["schema_version"] == fixture_mod.IR_SCHEMA_VERSION
        assert compiled["constraints"]


def test_scenario_verify_5287_format_validity_is_separate_from_solver_correctness() -> None:
    """SCENARIO-VERIFY-5287: syntactic validity does not imply solver acceptance."""

    summary = mod.evaluate_trace_cases(mod.trace_cases())
    rows = summary["rows"]

    malformed_rows = _rows_by_type(rows, "malformed")
    assert malformed_rows
    assert all(row["format_valid"] is False for row in malformed_rows)
    assert all(row["solver_was_run"] is False for row in malformed_rows)
    assert all(row["accepted"] is False for row in malformed_rows)

    semantic_rows = _rows_by_type(rows, "semantic-error")
    assert semantic_rows
    assert all(row["format_valid"] is True for row in semantic_rows)
    assert all(row["solver_was_run"] is True for row in semantic_rows)
    assert all(row["semantic_correct"] is False for row in semantic_rows)
    assert all(row["accepted"] is False for row in semantic_rows)
    assert any(row["solver_false_accept"] is True for row in semantic_rows)

    split = summary["format_vs_semantic_split"]
    assert split["format_valid_semantic_wrong"] == 4
    assert set(split["semantic_error_trace_ids"]) == {row["trace_id"] for row in semantic_rows}
    assert summary["unsafe_false_accepts"] == 0


def test_req_verify_5287_dependency_and_counterexample_labels_are_executable() -> None:
    """REQ-VERIFY-5287: dependency links and counterexample labels are checked."""

    negative_case = next(case for case in mod.trace_cases() if case["case_type"] == "negative")
    row = mod.check_trace(negative_case)

    assert row["format_valid"] is True
    assert row["counterexample_labels_valid"] is True
    assert row["accepted"] is True
    assert row["semantic_correct"] is True

    broken_dependency = copy.deepcopy(negative_case)
    broken_dependency["expressions"][0]["depends_on"] = ["missing_claim"]
    validation = mod.validate_trace_schema(broken_dependency)
    assert validation.ok is False
    assert any("unknown dependency missing_claim" in error for error in validation.errors)
    broken_row = mod.check_trace(broken_dependency)
    assert broken_row["format_valid"] is False
    assert broken_row["solver_was_run"] is False
    assert broken_row["accepted"] is False

    broken_counterexample = copy.deepcopy(negative_case)
    broken_counterexample["counterexamples"][0]["violated_constraints"] = []
    counterexample_row = mod.check_trace(broken_counterexample)
    assert counterexample_row["format_valid"] is True
    assert counterexample_row["counterexample_labels_valid"] is False
    assert counterexample_row["accepted"] is False


def test_req_verify_5287_schema_validation_reports_malformed_variants() -> None:
    """REQ-VERIFY-5287: malformed trace fields are rejected before compilation."""

    base = next(case for case in mod.trace_cases() if case["case_type"] == "positive")
    mutations: list[tuple[str, dict[str, object], str]] = []

    def mutated(label: str) -> dict[str, object]:
        case = copy.deepcopy(base)
        mutations.append((label, case, ""))
        return case

    case = mutated("top_level")
    case["schema_version"] = "wrong"
    case["fixture_id"] = "not_exp5273"
    case["case_type"] = "unknown"
    mutations[-1] = ("top_level", case, "schema_version must be")

    case = mutated("variables_not_object")
    case["variables"] = []
    mutations[-1] = ("variables_not_object", case, "variables must be an object")

    case = mutated("bad_variables")
    case["variables"] = {"1bad": {"type": "int"}, "x": {"type": "str"}}
    mutations[-1] = ("bad_variables", case, "invalid variable name")

    case = mutated("claims_not_list")
    case["claims"] = "claim"
    mutations[-1] = ("claims_not_list", case, "claims must be a list")

    case = mutated("claim_item_not_object")
    case["claims"] = ["claim"]
    mutations[-1] = ("claim_item_not_object", case, "claims[0] must be an object")

    case = mutated("bad_claim_id")
    case["claims"][0]["id"] = "bad-id"
    mutations[-1] = ("bad_claim_id", case, "claim has invalid id")

    case = mutated("duplicate_id")
    case["expressions"][0]["id"] = "claim_requirements"
    mutations[-1] = ("duplicate_id", case, "duplicate id claim_requirements")

    case = mutated("empty_claim_text")
    case["claims"][0]["text"] = ""
    mutations[-1] = ("empty_claim_text", case, "claim text must be non-empty")

    case = mutated("empty_expression")
    case["expressions"][0]["expr"] = ""
    mutations[-1] = ("empty_expression", case, "expression expr must be non-empty")

    case = mutated("unknown_expression")
    case["constraints"][0]["expression_id"] = "missing_expr"
    mutations[-1] = ("unknown_expression", case, "references unknown expression_id")

    case = mutated("bad_deduction")
    case["deductions"][0]["schema"] = "magic"
    case["deductions"][0]["conclusion"] = 3
    mutations[-1] = ("bad_deduction", case, "deduction schema is not allowed")

    case = mutated("bad_premises")
    case["deductions"][1]["premises"] = "not_list"
    mutations[-1] = ("bad_premises", case, "deduction premises must be a list")

    case = mutated("unknown_solver_premise")
    case["deductions"][1]["premises"] = ["missing_constraint"]
    mutations[-1] = (
        "unknown_solver_premise",
        case,
        "solver deduction premises must be constraint ids",
    )

    case = mutated("bad_counterexample_assignment")
    case["counterexamples"][0]["assignment"] = {"x": "five"}
    mutations[-1] = (
        "bad_counterexample_assignment",
        case,
        "counterexample assignment must map variables to ints",
    )

    case = mutated("bad_counterexample_violated_type")
    case["counterexamples"][0]["violated_constraints"] = "x_even"
    mutations[-1] = (
        "bad_counterexample_violated_type",
        case,
        "counterexample violated_constraints must be a list",
    )

    case = mutated("unknown_counterexample_constraint")
    case["counterexamples"][0]["violated_constraints"] = ["missing_constraint"]
    mutations[-1] = (
        "unknown_counterexample_constraint",
        case,
        "counterexample references unknown violated constraint",
    )

    repair_case = next(case for case in mod.trace_cases() if case["case_type"] == "repair")
    bad_repair = copy.deepcopy(repair_case)
    bad_repair["repairs"][0]["label"] = "global_rewrite"
    bad_repair["repairs"][0]["target_id"] = "missing_expr"
    bad_repair["repairs"][0]["replacement_expr"] = ""
    mutations.append(("bad_repair", bad_repair, "repair label is not allowed"))

    case = mutated("dependency_not_list")
    case["expressions"][0]["depends_on"] = "claim_requirements"
    mutations[-1] = ("dependency_not_list", case, "depends_on must be a list")

    for label, case, expected_error in mutations:
        validation = mod.validate_trace_schema(case)
        assert validation.ok is False, label
        assert any(expected_error in error for error in validation.errors), (
            label,
            validation.errors,
        )

    with pytest.raises(ValueError, match="schema_version must be"):
        mod.compile_trace_to_constraint_ir(mutations[0][1])

    assert (
        mod._counterexample_labels_valid(
            base,
            {"constraints": [{"id": "not_listed", "expr": "x >= 0"}]},
        )
        is False
    )
    assert (
        mod._counterexample_labels_valid(
            base,
            {"constraints": [{"id": "x_even", "expr": "x >= missing"}]},
        )
        is False
    )


def test_scenario_verify_5287_localized_repairs_recheck_with_solver() -> None:
    """SCENARIO-VERIFY-5287: repair labels change only one expression and re-solve."""

    repair_rows = _rows_by_type(mod.evaluate_trace_cases(mod.trace_cases())["rows"], "repair")

    assert repair_rows
    for row in repair_rows:
        assert row["format_valid"] is True
        assert row["semantic_correct"] is False
        assert row["repair_success"] is True
        assert row["accepted"] is True
        assert row["repair_changed_expression_ids"] == [row["repair_target_id"]]
        assert row["final_solver_status"] == row["expected_status"]


def test_scenario_verify_5287_run_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5287: ready artifact has required gate fields."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit trace dsl", "outcome": "passed"}]
    artifact = mod.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "usable" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["trace_dsl_ready"] is True
    assert artifact["fixture_case_counts"]["semantic-error"] == 2
    assert artifact["solver_correctness_metrics"]["value"]["semantic_error_rejections"] == 2
    assert artifact["solver_correctness_metrics"]["value"]["repair_successes"] == 2
    assert artifact["format_vs_semantic_split"]["value"]["format_valid_semantic_wrong"] == 4
    assert artifact["unsafe_false_accepts"]["value"] == 0
    assert artifact["tests_run"] == tests_run


def test_req_verify_5287_artifact_schema_and_blocked_path_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5287: missing solver and artifact drift fail closed."""

    artifact = mod.run(result_path=tmp_path / "ready.json", tests_run=[], write=False)
    mod.validate_artifact(artifact)

    blocked = mod.run(
        result_path=tmp_path / "blocked.json",
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
        z3_module=None,
        write=False,
    )
    mod.validate_artifact(blocked)
    assert blocked["honest_verdict"]["value"].startswith("blocked_")
    assert blocked["trace_dsl_ready"] is False
    assert "solver_available=False" in blocked["trace_dsl_ready_principle"]

    broken = dict(artifact)
    broken.pop("trace_dsl_ready")
    with pytest.raises(AssertionError, match="missing required field trace_dsl_ready"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["honest_verdict"] = {
        "value": "usable",
        "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
    }
    with pytest.raises(AssertionError, match="complete: or blocked_"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["inference_substrate"] = {
        "value": "live_llm_inference",
        "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
    }
    with pytest.raises(AssertionError, match=mod.INFERENCE_SUBSTRATE):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["trace_dsl_ready"] = "yes"
    with pytest.raises(AssertionError, match="trace_dsl_ready must be a bare bool"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["unsafe_false_accepts"] = {
        "value": 1,
        "principle": mod.FIELD_PRINCIPLES["unsafe_false_accepts"],
    }
    broken["trace_dsl_ready"] = True
    with pytest.raises(AssertionError, match="ready trace DSL requires zero unsafe false accepts"):
        mod.validate_artifact(broken)
