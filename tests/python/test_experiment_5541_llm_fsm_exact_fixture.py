"""Tests for Exp5541 deterministic finite-state exact fixture.

Spec refs: REQ-VERIFY-5541, SCENARIO-VERIFY-5541.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5541_llm_fsm_exact_fixture as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5541_llm_fsm_exact_fixture.py")


def test_req_verify_5541_spec_declares_exact_fsm_contract() -> None:
    """REQ-VERIFY-5541: OpenSpec anchors fields, exact checks, and no-LLM rules."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5541") : spec.index("### REQ-VERIFY-5501")]

    assert "SCENARIO-VERIFY-5541" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "`no_llm_invoked` SHALL be `true`" in section
    assert "SHALL NOT invoke an LLM" in section
    assert "satisfiable instance" in section
    assert "unsatisfiable instance" in section
    assert "ambiguous or underconstrained instance" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_5541_default_family_has_three_exact_classes(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5541: default fixture separates sat, unsat, and ambiguous rows."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fsm_instances"] == 3
    assert artifact["state_counts"] == [3, 3, 4]
    assert artifact["satisfiable_instances"] == 1
    assert artifact["unsatisfiable_instances"] == 1
    assert artifact["ambiguous_instances"] == 1
    assert artifact["yaml_schema_valid"] is True
    assert artifact["reference_trace_checks_passed"] is True
    assert artifact["exact_transition_checks_passed"] is True
    assert artifact["sat_solver_checks_passed"] is True
    assert artifact["no_llm_invoked"] is True
    assert artifact["exact_fsm_fixture_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_reused"] == list(mod.TESTS_ADDED_OR_REUSED)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])

    reports = {row["instance_id"]: row for row in artifact["exact_check_reports"]}
    assert reports["fsm_sat_accept_error"]["solver_status"] == "satisfiable"
    assert reports["fsm_unsat_conflicting_transition"]["solver_status"] == "unsatisfiable"
    assert reports["fsm_ambiguous_sparse_branch"]["solver_status"] == "ambiguous"
    assert reports["fsm_ambiguous_sparse_branch"]["completion_count"] == 3
    assert reports["fsm_ambiguous_sparse_branch"]["unreachable_states"] == ["C", "D"]
    assert {
        row["actual_label"]
        for row in reports["fsm_ambiguous_sparse_branch"]["trace_checks"]
    } == {"rejected", "underdetermined"}

    for description in artifact["machine_descriptions"]:
        decoded = mod.parse_machine_description_text(description["machine_description_yaml"])
        assert mod.validate_machine_description(decoded) == []
        assert description["natural_language_summary"] == mod.natural_language_summary(decoded)

    mod.validate_artifact(artifact)


def test_req_verify_5541_exact_checks_detect_trace_and_transition_edges() -> None:
    """REQ-VERIFY-5541: exact checkers expose acceptance, contradictions, and reachability."""

    family = mod.build_fixture_family()
    sat = next(row for row in family if row["instance_id"] == "fsm_sat_accept_error")
    unsat = next(row for row in family if row["instance_id"] == "fsm_unsat_conflicting_transition")
    ambiguous = next(row for row in family if row["instance_id"] == "fsm_ambiguous_sparse_branch")

    sat_report = mod.solve_instance(sat)
    labels = {row["trace_id"]: row["actual_label"] for row in sat_report["trace_checks"]}
    assert labels == {
        "sat_empty_rejects": "rejected",
        "sat_b_accepts": "accepted",
        "sat_a_errors": "error",
    }
    assert sat_report["transition_consistency_passed"] is True
    assert sat_report["unreachable_states"] == []

    unsat_report = mod.solve_instance(unsat)
    assert unsat_report["transition_consistency_passed"] is False
    assert unsat_report["completion_count"] == 0
    assert "deterministic_conflict:S0/x:S1!=S2" in unsat_report["contradictions"]
    assert {row["actual_label"] for row in unsat_report["trace_checks"]} == {"contradiction"}

    ambiguous_report = mod.solve_instance(ambiguous)
    assert ambiguous_report["transition_consistency_passed"] is True
    assert ambiguous_report["completion_count"] == 3
    assert mod.trace_labels_across_completions(
        ambiguous,
        ambiguous_report["completion_samples"],
        ["go", "go"],
    ) == ["accepted", "rejected"]

    invalid = deepcopy(sat)
    invalid["transition_constraints"][0]["target"] = "MISSING"
    invalid_report = mod.solve_instance(invalid)
    assert invalid_report["solver_status"] == "unsatisfiable"
    assert invalid_report["transition_consistency_passed"] is False
    assert "invalid_target:TC_SAT_00:MISSING" in invalid_report["contradictions"]


def test_req_verify_5541_schema_and_artifact_validation_fail_closed() -> None:
    """REQ-VERIFY-5541: validation rejects malformed schemas, LLM claims, and checksum drift."""

    artifact = mod.build_artifact(
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}]
    )

    bad_llm = deepcopy(artifact)
    bad_llm["no_llm_invoked"] = False
    bad_llm["reproducibility_checksum"] = mod.payload_checksum(bad_llm)
    with pytest.raises(ValueError, match="no_llm_invoked"):
        mod.validate_artifact(bad_llm)

    bad_gate = deepcopy(artifact)
    bad_gate["exact_fsm_fixture_ready"] = True
    bad_gate["sat_solver_checks_passed"] = False
    bad_gate["reproducibility_checksum"] = mod.payload_checksum(bad_gate)
    with pytest.raises(ValueError, match="sat_solver_checks_passed"):
        mod.validate_artifact(bad_gate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)

    broken_machine = deepcopy(artifact["fsm_family"][0])
    broken_machine.pop("states")
    assert "missing:states" in mod.validate_machine_description(broken_machine)

    broken_yaml = artifact["machine_descriptions"][0]["machine_description_yaml"][:-4]
    with pytest.raises(ValueError, match="machine_description_yaml"):
        mod.parse_machine_description_text(broken_yaml)


def test_req_verify_5541_custom_family_keeps_underconstrained_rows_visible() -> None:
    """REQ-VERIFY-5541: configurable sparse FSMs produce generated reference labels."""

    machine = mod.build_fixture_instance(
        instance_id="unit_sparse",
        states=["A", "B"],
        alphabet=["tick"],
        start_state="A",
        accepting_states=["B"],
        error_states=[],
        required_transitions=[
            ("TC_UNIT_0", "A", "tick", "A"),
        ],
        forbidden_transitions=[],
        trace_specs=[
            ("unit_one_tick", ["tick"]),
            ("unit_two_ticks", ["tick", "tick"]),
        ],
        expected_status="ambiguous",
    )
    report = mod.solve_instance(machine)

    assert machine["state_count"] == 2
    assert machine["transition_sparsity"] == pytest.approx(0.5)
    assert [row["expected_label"] for row in machine["observable_traces"]] == [
        "rejected",
        "rejected",
    ]
    assert report["solver_status"] == "ambiguous"
    assert report["completion_count"] == 2
    assert all(row["passed"] for row in report["trace_checks"])


def test_req_verify_5541_defensive_exact_edges_are_explicit() -> None:
    """REQ-VERIFY-5541: uncommon exact edges still return deterministic labels."""

    blocked = mod.build_fixture_instance(
        instance_id="unit_forbid_all",
        states=["A"],
        alphabet=["x"],
        start_state="A",
        accepting_states=[],
        error_states=[],
        required_transitions=[],
        forbidden_transitions=[("TC_BLOCK_ALL", "A", "x", "A")],
        trace_specs=[("blocked_trace", ["x"])],
        expected_status="unsatisfiable",
    )
    assert mod.enumerate_completions(blocked) == []
    assert mod.solve_instance(blocked)["solver_status"] == "unsatisfiable"

    start_error = mod.build_fixture_instance(
        instance_id="unit_start_error",
        states=["E"],
        alphabet=["x"],
        start_state="E",
        accepting_states=[],
        error_states=["E"],
        required_transitions=[("TC_ERR_LOOP", "E", "x", "E")],
        forbidden_transitions=[],
        trace_specs=[("start_errors", [])],
        expected_status="satisfiable",
    )
    completion = mod.enumerate_completions(start_error)[0]
    assert mod.simulate_trace(start_error, completion, []) == "error"

    with pytest.raises(ValueError, match="did not decode to an object"):
        mod.parse_machine_description_text("[]")

    assert mod.honest_verdict(False).startswith("blocked:")
