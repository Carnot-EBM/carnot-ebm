"""Tests for Exp5555 ASP/FSM nonmonotonic exact fixture.

Spec refs: REQ-VERIFY-5555, SCENARIO-VERIFY-5555.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5541_llm_fsm_exact_fixture as fsm_mod
from carnot import experiment_5555_asp_fsm_nonmonotonic_fixture as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5555_asp_fsm_nonmonotonic_fixture.py")


def _ready_upstream() -> dict:
    return fsm_mod.build_artifact(
        tests_run=[
            {
                "command": "tests/python/test_experiment_5541_llm_fsm_exact_fixture.py",
                "outcome": "passed",
            }
        ]
    )


def test_req_verify_5555_spec_declares_asp_fsm_contract() -> None:
    """REQ-VERIFY-5555: OpenSpec anchors ASP stable-model fields and no-LLM rules."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5555") : spec.index("### REQ-VERIFY-5501")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5555" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(fsm_mod.RESULT_RELATIVE_PATH) in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "`llm_invoked` SHALL be `false`" in section
    assert "SHALL NOT invoke an LLM" in section
    assert "stable-model checks" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5555_default_rows_cover_all_stable_model_classes(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5555: default ASP rows separate sat, unsat, and ambiguity."""

    upstream = fsm_mod.run(
        result_path=tmp_path / fsm_mod.RESULT_RELATIVE_PATH,
        tests_run=[
            {
                "command": "tests/python/test_experiment_5541_llm_fsm_exact_fixture.py",
                "outcome": "passed",
            }
        ],
    )
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        result_path=result_path,
        upstream_artifact=upstream,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["upstream_fsm_fixture"] == str(fsm_mod.RESULT_RELATIVE_PATH)
    assert artifact["llm_invoked"] is False
    assert artifact["no_model_specs_required"] is True
    assert artifact["asp_row_count"] == 5
    assert artifact["default_rule_count"] >= 3
    assert artifact["contradiction_row_count"] == 1
    assert artifact["stable_model_count"] == 4
    assert artifact["sat_count"] == 2
    assert artifact["unsat_count"] == 2
    assert artifact["ambiguous_count"] == 1
    assert artifact["exact_asp_validator_ready"] is True
    assert artifact["exact_fsm_fixture_extended_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert "model_specs" not in artifact
    assert str(TEST_PATH) in artifact["tests_added_or_reused"]
    assert str(mod.SPEC_PATH) in artifact["spec_files_updated_or_confirmed"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])

    by_id = {row["row_id"]: row for row in artifact["stable_model_reports"]}
    assert by_id["asp_sat_fsm_acceptance_default_guard"]["solver_status"] == "satisfiable"
    assert by_id["asp_unsat_fsm_forbidden_error"]["stable_model_count"] == 0
    assert by_id["asp_ambiguous_fsm_default_repair_choice"]["stable_model_count"] == 2
    assert by_id["asp_default_negation_no_exception"]["contains_default_negation"] is True
    assert by_id["asp_contradiction_fact_constraint"]["contradiction_row"] is True

    mod.validate_artifact(artifact)


def test_req_verify_5555_exact_stable_model_evaluator_edges() -> None:
    """REQ-VERIFY-5555: evaluator implements GL-reduct checks for limited rows."""

    fact_rule = mod.asp_rule("R_FACT", "seed")
    derived_rule = mod.asp_rule(
        "R_DERIVED", "derived", positive=("seed",), default_negated=("blocked",)
    )
    constraint = mod.asp_rule("R_CONSTRAINT", None, positive=("derived", "blocked"))
    report = mod.evaluate_asp_row(
        {
            "row_id": "unit_default",
            "description": "unit default row",
            "facts": ["seed"],
            "rules": [fact_rule, derived_rule, constraint],
            "expected_status": "satisfiable",
            "contradiction_row": False,
        }
    )

    assert report["solver_status"] == "satisfiable"
    assert report["stable_model_count"] == 1
    assert report["stable_model_samples"] == [["derived", "seed"]]
    assert report["status_matches_expected"] is True

    ambiguous = mod.evaluate_asp_row(
        {
            "row_id": "unit_ambiguous",
            "description": "unit ambiguous row",
            "facts": ["enabled"],
            "rules": [
                mod.asp_rule("R_A", "left", positive=("enabled",), default_negated=("right",)),
                mod.asp_rule("R_B", "right", positive=("enabled",), default_negated=("left",)),
            ],
            "expected_status": "ambiguous",
            "contradiction_row": False,
        }
    )
    assert ambiguous["stable_model_samples"] == [["enabled", "left"], ["enabled", "right"]]

    unsat = mod.evaluate_asp_row(
        {
            "row_id": "unit_unsat",
            "description": "unit unsat row",
            "facts": ["bad"],
            "rules": [mod.asp_rule("R_BLOCK", None, positive=("bad",))],
            "expected_status": "unsatisfiable",
            "contradiction_row": True,
        }
    )
    assert unsat["solver_status"] == "unsatisfiable"
    assert unsat["violated_constraint_count"] > 0

    with pytest.raises(ValueError, match="unsupported_atom"):
        mod.evaluate_asp_row(
            {
                "row_id": "unit_bad_atom",
                "description": "bad atom",
                "facts": ["not valid"],
                "rules": [],
                "expected_status": "satisfiable",
                "contradiction_row": False,
            }
        )


def test_req_verify_5555_validation_fails_closed_on_overclaim() -> None:
    """REQ-VERIFY-5555: validation rejects LLM use, bad counts, and unready upstream."""

    artifact = mod.build_artifact(
        upstream_artifact=_ready_upstream(),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    bad_llm = deepcopy(artifact)
    bad_llm["llm_invoked"] = True
    bad_llm["reproducibility_checksum"] = mod.payload_checksum(bad_llm)
    with pytest.raises(ValueError, match="llm_invoked"):
        mod.validate_artifact(bad_llm)

    bad_model_specs = deepcopy(artifact)
    bad_model_specs["model_specs"] = []
    bad_model_specs["reproducibility_checksum"] = mod.payload_checksum(bad_model_specs)
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(bad_model_specs)

    bad_ready = deepcopy(artifact)
    bad_ready["exact_fsm_fixture_extended_ready"] = True
    bad_ready["exact_asp_validator_ready"] = False
    bad_ready["reproducibility_checksum"] = mod.payload_checksum(bad_ready)
    with pytest.raises(ValueError, match="exact_asp_validator_ready"):
        mod.validate_artifact(bad_ready)

    bad_count = deepcopy(artifact)
    bad_count["stable_model_count"] += 1
    bad_count["reproducibility_checksum"] = mod.payload_checksum(bad_count)
    with pytest.raises(ValueError, match="stable_model_count"):
        mod.validate_artifact(bad_count)

    unready = deepcopy(_ready_upstream())
    unready["exact_fsm_fixture_ready"] = False
    blocked = mod.build_artifact(upstream_artifact=unready)
    assert blocked["exact_fsm_fixture_extended_ready"] is False
    assert blocked["honest_verdict"].startswith("blocked:")

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_verify_5555_defensive_helpers_are_deterministic(tmp_path: Path) -> None:
    """REQ-VERIFY-5555: defensive parser and helper edges stay deterministic."""

    facts = mod.fsm_facts(
        {
            "exact_check_reports": [
                [],
                {
                    "instance_id": "123",
                    "solver_status": "",
                    "trace_checks": [[], {"trace_id": "", "actual_label": ""}],
                },
            ]
        }
    )

    assert facts["123"]["status_empty"] == "a_123_status_empty"
    assert facts["123"]["trace_empty_empty"] == "a_123_trace_empty_empty"
    assert mod.stable_models([mod.asp_rule("R_FACT", "seed")]) == [["seed"]]
    assert mod._least_model([mod.asp_rule("R_CONSTRAINT", None)]) == set()
    assert mod.honest_verdict(False, upstream_ready=True).startswith("blocked: ASP")

    missing = tmp_path / "missing.json"
    assert mod._load_json(missing)["load_error"] == "missing"

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod._load_json(malformed)["load_error"] == "json_decode"

    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod._load_json(list_payload)["load_error"] == "json_not_object"

    object_payload = tmp_path / "object.json"
    object_payload.write_text('{"ok": true}', encoding="utf-8")
    assert mod._load_json(object_payload) == {"ok": True}
