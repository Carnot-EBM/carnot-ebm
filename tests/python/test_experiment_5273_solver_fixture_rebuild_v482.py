"""Tests for Exp 5273 deterministic solver fixture rebuild.

Spec refs: REQ-VERIFY-5273, SCENARIO-VERIFY-5273.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5273_solver_fixture_rebuild_v482 as mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def test_req_verify_5273_spec_declares_offline_fixture_gate() -> None:
    """REQ-VERIFY-5273: OpenSpec anchors the offline fixture gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5273") : spec.index("### REQ-VERIFY-5263")
    ]

    for marker in (
        "REQ-VERIFY-5273",
        "SCENARIO-VERIFY-5273",
        str(mod.RESULT_RELATIVE_PATH),
        "offline_deterministic_certificate_no_llm",
        "solver_fixture_ready",
        "reference-copy",
        "empty-extraction",
        "deterministic shuffled",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5273_fixture_set_has_solver_labels_and_counterexamples() -> None:
    """REQ-VERIFY-5273: every fixture has executable labels and negative evidence."""

    fixtures = mod.fixture_set()
    assert len(fixtures) >= 6
    assert {fixture.expected_status for fixture in fixtures} == {"sat", "unsat"}
    assert len({fixture.fixture_id for fixture in fixtures}) == len(fixtures)

    for fixture in fixtures:
        assert fixture.natural_language
        assert fixture.reference_encoding["schema_version"] == mod.IR_SCHEMA_VERSION
        schema = mod.validate_extracted_constraints(fixture.reference_encoding)
        assert schema.ok, schema.errors

        score = mod.score_candidate(fixture, fixture.reference_encoding)
        assert score.schema_valid is True
        assert score.solver_status == fixture.expected_status
        assert score.matches_expected is True
        if fixture.expected_status == "sat":
            assert score.assignment == fixture.gold_assignment
        else:
            assert score.assignment == {}

        counterexample_rows = mod.fixture_counterexample_rows(fixture)
        assert counterexample_rows
        assert all(row["violated_constraints"] for row in counterexample_rows)

    assert mod.counterexample_coverage(fixtures) == 1.0


def test_req_verify_5273_schema_rejects_malformed_ir_before_solver_scoring() -> None:
    """REQ-VERIFY-5273: malformed extracted constraints fail before Z3 scoring."""

    fixture = mod.fixture_set()[0]
    assert mod.sha16("same") == mod.sha16(b"same")
    malformed_payloads: list[dict[str, Any]] = [
        {"variables": {"x": {"type": "int"}}, "constraints": []},
        {
            "schema_version": mod.IR_SCHEMA_VERSION,
            "variables": ["x"],
            "constraints": [],
        },
        {
            "schema_version": mod.IR_SCHEMA_VERSION,
            "variables": {"x": {"type": "int"}},
            "constraints": [{"id": "not_executable", "expr": "x is even"}],
        },
        {
            "schema_version": mod.IR_SCHEMA_VERSION,
            "variables": {"x": {"type": "str"}},
            "constraints": [{"id": "c1", "expr": "x >= 0"}],
        },
        {
            "schema_version": mod.IR_SCHEMA_VERSION,
            "variables": {"1bad": {"type": "int"}},
            "constraints": [],
        },
        {
            "schema_version": mod.IR_SCHEMA_VERSION,
            "variables": {},
            "constraints": "x >= 0",
        },
        {
            "schema_version": mod.IR_SCHEMA_VERSION,
            "variables": {"x": {"type": "int"}},
            "constraints": [3],
        },
        {
            "schema_version": mod.IR_SCHEMA_VERSION,
            "variables": {"x": {"type": "int"}},
            "constraints": [{"id": "bad-id", "expr": "x >= 0"}],
        },
        {
            "schema_version": mod.IR_SCHEMA_VERSION,
            "variables": {"x": {"type": "int"}},
            "constraints": [{"id": "empty_expr", "expr": ""}],
        },
        {
            "schema_version": mod.IR_SCHEMA_VERSION,
            "variables": {"x": {"type": "int"}},
            "constraints": [{"id": "syntax_error", "expr": "x >="}],
        },
        {
            "schema_version": mod.IR_SCHEMA_VERSION,
            "variables": {"x": {"type": "int"}},
            "constraints": [{"id": "division", "expr": "x / 2 == 1"}],
        },
        {
            "schema_version": mod.IR_SCHEMA_VERSION,
            "variables": {"x": {"type": "int"}},
            "constraints": [{"id": "call_expr", "expr": "abs(x) == 1"}],
        },
    ]

    class ExplodingZ3:
        sat = "sat"
        unsat = "unsat"

        @staticmethod
        def Solver() -> Any:
            raise AssertionError("schema-invalid payload must not reach solver")

    for payload in malformed_payloads:
        schema = mod.validate_extracted_constraints(payload)
        assert schema.ok is False
        assert schema.errors
        score = mod.score_candidate(fixture, payload, z3_module=ExplodingZ3)
        assert score.schema_valid is False
        assert score.solver_status == "schema_error"
        assert score.matches_expected is False
        assert score.counterexample["schema_errors"]


def test_scenario_verify_5273_scores_deterministic_baselines() -> None:
    """SCENARIO-VERIFY-5273: reference, empty, and shuffled controls are separate."""

    baselines = mod.score_baselines(mod.fixture_set())

    assert mod.counterexample_coverage([]) == 0.0
    assert baselines["reference_copy"]["validity_rate"] == 1.0
    assert baselines["reference_copy"]["false_accepts"] == 0
    assert baselines["reference_copy"]["counterexamples_found"] == 0
    assert baselines["empty_extraction"]["validity_rate"] == 0.5
    assert baselines["empty_extraction"]["false_accepts"] == 3
    assert baselines["empty_extraction"]["counterexamples_found"] == 3
    assert baselines["deterministic_shuffled_reference"]["validity_rate"] == pytest.approx(
        2 / 6
    )
    assert baselines["deterministic_shuffled_reference"]["false_accepts"] == 2


def test_scenario_verify_5273_run_writes_ready_artifact_with_checksums(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5273: ready artifact includes all required receipts."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit fixture", "outcome": "passed"}]

    artifact = mod.run(result_path=result_path, tests_run=tests_run, write=True)

    mod.validate_artifact(artifact)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "solver_fixture_ready true" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["solver_fixture_ready"] is True
    assert artifact["fixture_count"]["value"] == len(mod.fixture_set())
    assert artifact["baseline_validity"]["value"] == 1.0
    assert artifact["counterexample_coverage"]["value"] == 1.0
    assert artifact["schema_checks_passed"]["value"] is True
    assert artifact["fixture_checksums"]["value"]["fixture_set_sha256"]
    assert set(artifact["fixture_checksums"]["value"]["fixtures"]) == {
        fixture.fixture_id for fixture in mod.fixture_set()
    }
    assert artifact["baselines"]["empty_extraction"]["validity_rate"] == 0.5
    assert artifact["prior_v481_diagnosis"]["baseline_validity_not_useful"]
    assert artifact["prior_v481_diagnosis"]["model_validity_not_useful"]
    assert artifact["tests_run"] == tests_run


def test_req_verify_5273_artifact_schema_and_blocked_path_fail_closed(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5273: artifact validation rejects drift and blocks without Z3."""

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
    assert blocked["solver_fixture_ready"] is False
    assert blocked["schema_checks_passed"]["value"] is False
    assert mod._honest_verdict(False, True).startswith("complete: solver_fixture_ready false")
    assert mod._counterexample_for_result(
        mod.fixture_set()[3],
        "schema_error",
        {},
        False,
    ) == {"expected_status": "unsat", "solver_status": "schema_error"}

    broken = dict(artifact)
    broken.pop("fixture_count")
    with pytest.raises(AssertionError, match="missing required field fixture_count"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["solver_fixture_ready"] = "true"
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["honest_verdict"] = {
        "value": "ready",
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
    broken["fixture_count"] = {
        "value": "six",
        "principle": mod.FIELD_PRINCIPLES["fixture_count"],
    }
    with pytest.raises(AssertionError, match="fixture_count.value must be int"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["schema_checks_passed"] = {
        "value": "yes",
        "principle": mod.FIELD_PRINCIPLES["schema_checks_passed"],
    }
    with pytest.raises(AssertionError, match="schema_checks_passed.value must be bool"):
        mod.validate_artifact(broken)
