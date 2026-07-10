"""Tests for Exp5545 sparse repair descriptors over exact FSM fixtures.

Spec refs: REQ-VERIFY-5545, SCENARIO-VERIFY-5545.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5541_llm_fsm_exact_fixture as fsm_mod
from carnot import experiment_5545_sparse_repair_fsm_descriptor_scale as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5545_sparse_repair_fsm_descriptor_scale.py")


def _ready_upstream() -> dict[str, object]:
    return fsm_mod.build_artifact(
        tests_run=[{"command": "tests/python/test_experiment_5541_llm_fsm_exact_fixture.py", "outcome": "passed"}]
    )


def test_req_verify_5545_spec_declares_fsm_sparse_repair_contract() -> None:
    """REQ-VERIFY-5545: OpenSpec anchors fields, exact checks, and no speedup."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5545") : spec.index("### REQ-VERIFY-5501")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5545" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(fsm_mod.RESULT_RELATIVE_PATH) in section
    assert "`exact_fsm_fixture_ready=true`" in section
    assert "SHALL NOT modify `scripts/research_conductor.py`" in section
    assert "`speedup_claim_allowed` SHALL be `false`" in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5545_descriptors_extract_fsm_conflicts_reachability_and_traces() -> None:
    """SCENARIO-VERIFY-5545: descriptors name conflicts, unreachable states, and trace issues."""

    upstream = _ready_upstream()
    descriptors = mod.build_sparse_descriptors(upstream)
    by_id = {row["source_instance_id"]: row for row in descriptors["sparse_repair_descriptors"]}

    assert descriptors["descriptor_count"] == upstream["fsm_instances"]
    assert set(by_id) == {row["instance_id"] for row in upstream["fsm_family"]}

    unsat = by_id["fsm_unsat_conflicting_transition"]
    assert "transition:S0/x" in unsat["repair_block_variables"]
    assert "trace:unsat_x_contradiction" in unsat["repair_block_variables"]
    assert "expected_status" in unsat["repair_block_variables"]
    assert {"transition_conflict", "unreachable_state", "trace_contradiction"} <= {
        row["kind"] for row in unsat["active_constraints"]
    }
    assert unsat["target_repair_assignment"] == {
        "expected_status": "ambiguous",
        "trace:unsat_x_contradiction": "accepted",
        "transition:S0/x": "S1",
    }

    ambiguous = by_id["fsm_ambiguous_sparse_branch"]
    assert "transition:B/go" in ambiguous["repair_block_variables"]
    assert "trace:amb_go_go_underconstrained" in ambiguous["repair_block_variables"]
    assert {"unreachable_state", "trace_contradiction"} <= {
        row["kind"] for row in ambiguous["active_constraints"]
    }
    assert ambiguous["target_repair_assignment"]["transition:B/go"] == "C"
    assert ambiguous["exact_fallback"]["status"] == "satisfiable"

    sat = by_id["fsm_sat_accept_error"]
    assert sat["active_constraints"] == []
    assert sat["target_repair_assignment"] == {}
    for descriptor in descriptors["sparse_repair_descriptors"]:
        mod.validate_sparse_descriptor(descriptor)


def test_scenario_verify_5545_policy_comparison_checks_every_candidate_exactly() -> None:
    """SCENARIO-VERIFY-5545: every policy candidate receives an exact FSM decision."""

    upstream = _ready_upstream()
    descriptors = mod.build_sparse_descriptors(upstream)
    comparison = mod.run_policy_comparison(upstream_artifact=upstream, descriptors=descriptors)

    assert comparison["fsm_instance_count"] == upstream["fsm_instances"]
    assert comparison["random_seed_count"] == len(mod.SEEDS)
    assert comparison["descriptor_guided_success_rate"] == pytest.approx(1.0)
    assert comparison["exact_only_success_rate"] == pytest.approx(1.0)
    assert comparison["random_block_success_rate"] < comparison["descriptor_guided_success_rate"]
    assert comparison["exact_fallback_used"] is True
    assert comparison["exact_validator_all_repairs_checked"] is True
    assert comparison["unchecked_repair_count"] == 0
    assert comparison["descriptor_mean_iterations"] <= comparison["random_mean_iterations"]

    expected_attempts = upstream["fsm_instances"] * len(mod.SEEDS)
    for policy_name in ("descriptor_guided", "random_block", "exact_only"):
        attempts = comparison["policy_results"][policy_name]
        assert len(attempts) == expected_attempts
        for attempt in attempts:
            assert attempt["candidate_checks"]
            assert {row["exact_validator_decision"] for row in attempt["candidate_checks"]} <= {
                "accepted",
                "rejected",
            }
            assert all(row["exact_checked"] for row in attempt["candidate_checks"])


def test_req_verify_5545_artifact_writes_required_json(tmp_path: Path) -> None:
    """REQ-VERIFY-5545: run writes the required sparse repair FSM result fields."""

    upstream = fsm_mod.run(
        result_path=tmp_path / fsm_mod.RESULT_RELATIVE_PATH,
        tests_run=[{"command": "tests/python/test_experiment_5541_llm_fsm_exact_fixture.py", "outcome": "passed"}],
    )
    artifact = mod.run(
        repo_root=tmp_path,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["upstream_exact_fsm_fixture_ready"] is True
    assert artifact["fsm_instance_count"] == upstream["fsm_instances"]
    assert artifact["random_seed_count"] == len(mod.SEEDS)
    assert artifact["descriptor_guided_success_rate"] == pytest.approx(1.0)
    assert artifact["exact_only_success_rate"] == pytest.approx(1.0)
    assert artifact["random_block_success_rate"] < artifact["descriptor_guided_success_rate"]
    assert artifact["exact_fallback_used"] is True
    assert artifact["exact_validator_all_repairs_checked"] is True
    assert artifact["matched_timing_available"] is False
    assert artifact["speedup_claim_allowed"] is False
    assert artifact["sparse_repair_fsm_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert str(TEST_PATH) in artifact["tests_added_or_reused"]
    assert artifact["research_conductor_modified"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    mod.validate_artifact(artifact)


def test_req_verify_5545_validation_fails_closed_on_unready_upstream_or_overclaim() -> None:
    """REQ-VERIFY-5545: validation rejects unchecked repairs and speedup overclaims."""

    artifact = mod.build_artifact(
        upstream_artifact=_ready_upstream(),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    no_upstream = deepcopy(artifact)
    no_upstream["upstream_exact_fsm_fixture_ready"] = False
    no_upstream["reproducibility_checksum"] = mod.payload_checksum(no_upstream)
    with pytest.raises(ValueError, match="upstream_exact_fsm_fixture_ready"):
        mod.validate_artifact(no_upstream)

    unchecked = deepcopy(artifact)
    unchecked["exact_validator_all_repairs_checked"] = False
    unchecked["reproducibility_checksum"] = mod.payload_checksum(unchecked)
    with pytest.raises(ValueError, match="exact_validator_all_repairs_checked"):
        mod.validate_artifact(unchecked)

    speedup = deepcopy(artifact)
    speedup["speedup_claim_allowed"] = True
    speedup["reproducibility_checksum"] = mod.payload_checksum(speedup)
    with pytest.raises(ValueError, match="speedup_claim_allowed"):
        mod.validate_artifact(speedup)

    bad_timing = deepcopy(artifact)
    bad_timing["matched_timing_available"] = True
    bad_timing["reproducibility_checksum"] = mod.payload_checksum(bad_timing)
    with pytest.raises(ValueError, match="matched_timing_available"):
        mod.validate_artifact(bad_timing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_verify_5545_readiness_blockers_and_upstream_gate_are_explicit() -> None:
    """REQ-VERIFY-5545: readiness blockers name failed gates without hidden success."""

    upstream = _ready_upstream()
    descriptors = mod.build_sparse_descriptors(upstream)
    comparison = mod.run_policy_comparison(upstream_artifact=upstream, descriptors=descriptors)

    failed_descriptor = deepcopy(comparison)
    failed_descriptor["descriptor_guided_success_rate"] = 0.0
    assert "descriptor_guided_success_rate" in mod.readiness_blockers(
        upstream,
        descriptors,
        failed_descriptor,
    )

    unchecked = deepcopy(comparison)
    unchecked["unchecked_repair_count"] = 1
    assert "unchecked_repair_count" in mod.readiness_blockers(upstream, descriptors, unchecked)

    bad_count = deepcopy(comparison)
    bad_count["fsm_instance_count"] = 1
    assert "fsm_instance_count" in mod.readiness_blockers(upstream, descriptors, bad_count)

    bad_seed_count = deepcopy(comparison)
    bad_seed_count["random_seed_count"] = 1
    assert "random_seed_count" in mod.readiness_blockers(upstream, descriptors, bad_seed_count)

    exact_failed = deepcopy(comparison)
    exact_failed["exact_only_success_rate"] = 0.0
    assert "exact_only_success_rate" in mod.readiness_blockers(upstream, descriptors, exact_failed)

    no_fallback = deepcopy(comparison)
    no_fallback["exact_fallback_used"] = False
    assert "exact_fallback_used" in mod.readiness_blockers(upstream, descriptors, no_fallback)

    not_all_checked = deepcopy(comparison)
    not_all_checked["exact_validator_all_repairs_checked"] = False
    assert "exact_validator_all_repairs_checked" in mod.readiness_blockers(
        upstream,
        descriptors,
        not_all_checked,
    )

    bad_descriptors = deepcopy(descriptors)
    bad_descriptors["descriptor_count"] = 1
    assert "descriptor_count" in mod.readiness_blockers(upstream, bad_descriptors, comparison)

    unready = deepcopy(upstream)
    unready["exact_fsm_fixture_ready"] = False
    assert "upstream_exact_fsm_fixture_ready" in mod.readiness_blockers(
        unready,
        descriptors,
        comparison,
    )
    with pytest.raises(ValueError, match="exact_fsm_fixture_ready"):
        mod.build_sparse_descriptors(unready)

    assert mod.honest_verdict(False, ["unit"]).startswith("blocked:")


def test_req_verify_5545_defensive_descriptor_edges_are_exact() -> None:
    """REQ-VERIFY-5545: trace-only repairs and target fallbacks stay exact-bounded."""

    trace_only = fsm_mod.build_fixture_instance(
        instance_id="unit_trace_only_underconstrained",
        states=["A", "B"],
        alphabet=["x"],
        start_state="A",
        accepting_states=["B"],
        error_states=[],
        required_transitions=[("TC_UNIT_A_B", "A", "x", "B")],
        forbidden_transitions=[],
        trace_specs=[("unit_two_steps", ["x", "x"])],
        expected_status="ambiguous",
    )
    descriptor = mod.build_sparse_descriptor(trace_only)

    assert descriptor["target_repair_assignment"] == {
        "expected_status": "satisfiable",
        "trace:unit_two_steps": "accepted",
        "transition:B/x": "B",
    }
    assert {
        row["kind"] for row in descriptor["active_constraints"]
    } == {"trace_transition_repair", "trace_contradiction"}
    assert descriptor["exact_fallback"]["accepted"] is True

    assert mod.choose_transition_target(
        {"accepting_states": [], "error_states": ["T2"]},
        ["T1", "T2"],
    ) == "T1"
    assert mod.choose_transition_target(
        {"accepting_states": [], "error_states": ["T1", "T2"]},
        ["T1", "T2"],
    ) == "T1"
    assert mod.transition_sparsity({"states": [], "alphabet": [], "transition_constraints": []}) == 0.0
    with pytest.raises(ValueError, match="no_allowed_trace_target"):
        mod.preferred_decisive_target(
            {"states": ["A"], "accepting_states": []},
            ("A", "x"),
            {("A", "x", "A")},
        )
