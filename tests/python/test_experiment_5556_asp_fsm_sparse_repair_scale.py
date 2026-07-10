"""Tests for Exp5556 sparse repair scale over the ASP/FSM exact fixture.

Spec refs: REQ-VERIFY-5556, SCENARIO-VERIFY-5556.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5541_llm_fsm_exact_fixture as fsm_mod
from carnot import experiment_5555_asp_fsm_nonmonotonic_fixture as asp_mod
from carnot import experiment_5556_asp_fsm_sparse_repair_scale as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5556_asp_fsm_sparse_repair_scale.py")


def _ready_upstream() -> dict[str, object]:
    fsm_artifact = fsm_mod.build_artifact(
        tests_run=[
            {
                "command": "tests/python/test_experiment_5541_llm_fsm_exact_fixture.py",
                "outcome": "passed",
            }
        ]
    )
    return asp_mod.build_artifact(
        upstream_artifact=fsm_artifact,
        tests_run=[
            {
                "command": "tests/python/test_experiment_5555_asp_fsm_nonmonotonic_fixture.py",
                "outcome": "passed",
            }
        ],
    )


def test_req_verify_5556_spec_declares_asp_sparse_repair_contract() -> None:
    """REQ-VERIFY-5556: OpenSpec anchors matched controls and no speedup."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5556") : spec.index("### REQ-VERIFY-5501")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5556" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(asp_mod.RESULT_RELATIVE_PATH) in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "`llm_invoked` SHALL be `false`" in section
    assert "`speedup_claim_allowed` SHALL be `false`" in section
    assert "same ASP rows, random seeds, and per-attempt candidate budget" in normalized
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5556_descriptors_cover_all_asp_row_families() -> None:
    """SCENARIO-VERIFY-5556: descriptors preserve row family and exact repair evidence."""

    upstream = _ready_upstream()
    descriptors = mod.build_asp_repair_descriptors(upstream)
    by_id = {row["row_id"]: row for row in descriptors["asp_repair_descriptors"]}

    assert descriptors["descriptor_count"] == upstream["asp_row_count"] == 5
    assert set(by_id) == {row["row_id"] for row in upstream["stable_model_reports"]}

    sat = by_id["asp_sat_fsm_acceptance_default_guard"]
    assert sat["target_repair_assignment"] == {
        "fact:fsm_sat_accept_error_trace_sat_empty_rejects_accepted": "absent"
    }
    assert sat["row_family_tags"] == ["satisfiable", "default_negation"]

    unsat = by_id["asp_unsat_fsm_forbidden_error"]
    assert unsat["target_repair_assignment"] == {"rule:ASP_UNSAT_01": "present"}
    assert unsat["row_family_tags"] == ["unsatisfiable"]

    ambiguous = by_id["asp_ambiguous_fsm_default_repair_choice"]
    assert ambiguous["target_repair_assignment"] == {"rule:ASP_AMB_01": "present"}
    assert ambiguous["row_family_tags"] == ["ambiguous", "default_negation"]

    default_row = by_id["asp_default_negation_no_exception"]
    assert default_row["target_repair_assignment"] == {"fact:exception_seen": "absent"}
    assert default_row["row_family_tags"] == ["satisfiable", "default_negation"]

    contradiction = by_id["asp_contradiction_fact_constraint"]
    assert contradiction["target_repair_assignment"] == {"rule:ASP_CONTRA_00": "present"}
    assert contradiction["row_family_tags"] == ["unsatisfiable"]

    for descriptor in descriptors["asp_repair_descriptors"]:
        assert descriptor["source_report"]["stable_model_checked"] is True
        assert descriptor["damaged_report"]["stable_model_checked"] is True
        assert descriptor["damaged_report"]["stable_model_samples"] != descriptor["source_report"][
            "stable_model_samples"
        ]
        mod.validate_sparse_descriptor(descriptor)


def test_scenario_verify_5556_matched_policy_comparison_records_family_breakdown() -> None:
    """SCENARIO-VERIFY-5556: all controls use matched rows, seeds, and budget."""

    upstream = _ready_upstream()
    descriptors = mod.build_asp_repair_descriptors(upstream)
    comparison = mod.run_policy_comparison(upstream_artifact=upstream, descriptors=descriptors)

    assert comparison["asp_row_count"] == upstream["asp_row_count"]
    assert comparison["random_seed_count"] == len(mod.SEEDS)
    assert comparison["candidate_budget_per_attempt"] == mod.CANDIDATE_BUDGET
    assert comparison["stable_model_checked_rate"] == pytest.approx(1.0)
    assert comparison["descriptor_guided_success_rate"] == pytest.approx(1.0)
    assert comparison["random_block_success_rate"] < comparison["descriptor_guided_success_rate"]
    assert comparison["exact_only_success_rate"] >= comparison["random_block_success_rate"]
    assert comparison["descriptor_mean_iterations"] <= comparison["random_mean_iterations"]

    expected_attempts = upstream["asp_row_count"] * len(mod.SEEDS)
    for policy_name in ("descriptor_guided", "random_block", "exact_only"):
        attempts = comparison["policy_results"][policy_name]
        assert len(attempts) == expected_attempts
        assert {attempt["candidate_budget"] for attempt in attempts} == {mod.CANDIDATE_BUDGET}
        for attempt in attempts:
            assert 1 <= attempt["iterations"] <= mod.CANDIDATE_BUDGET
            assert attempt["candidate_checks"]
            assert all(check["stable_model_checked"] for check in attempt["candidate_checks"])
            assert {check["exact_validator_decision"] for check in attempt["candidate_checks"]} <= {
                "accepted",
                "rejected",
            }

    breakdown = comparison["row_family_breakdown"]
    assert set(breakdown) >= {"satisfiable", "unsatisfiable", "ambiguous", "default_negation"}
    assert breakdown["satisfiable"]["row_count"] == 2
    assert breakdown["unsatisfiable"]["row_count"] == 2
    assert breakdown["ambiguous"]["row_count"] == 1
    assert breakdown["default_negation"]["row_count"] == 3
    for family in ("satisfiable", "unsatisfiable", "ambiguous", "default_negation"):
        assert breakdown[family]["stable_model_checked_rate"] == pytest.approx(1.0)
        assert breakdown[family]["descriptor_guided_success_rate"] == pytest.approx(1.0)


def test_req_verify_5556_artifact_writes_required_json(tmp_path: Path) -> None:
    """REQ-VERIFY-5556: run writes required fields without LLM or speedup claims."""

    fsm_artifact = fsm_mod.run(
        result_path=tmp_path / fsm_mod.RESULT_RELATIVE_PATH,
        tests_run=[
            {
                "command": "tests/python/test_experiment_5541_llm_fsm_exact_fixture.py",
                "outcome": "passed",
            }
        ],
    )
    asp_artifact = asp_mod.run(
        result_path=tmp_path / asp_mod.RESULT_RELATIVE_PATH,
        upstream_artifact=fsm_artifact,
        tests_run=[
            {
                "command": "tests/python/test_experiment_5555_asp_fsm_nonmonotonic_fixture.py",
                "outcome": "passed",
            }
        ],
    )
    artifact = mod.run(
        repo_root=tmp_path,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["upstream_asp_fsm_fixture"] == str(asp_mod.RESULT_RELATIVE_PATH)
    assert artifact["upstream_asp_fsm_fixture_ready"] is True
    assert artifact["llm_invoked"] is False
    assert artifact["no_model_specs_required"] is True
    assert "model_specs" not in artifact
    assert artifact["asp_row_count"] == asp_artifact["asp_row_count"]
    assert artifact["descriptor_guided_success_rate"] == pytest.approx(1.0)
    assert artifact["random_block_success_rate"] < artifact["descriptor_guided_success_rate"]
    assert artifact["stable_model_checked_rate"] == pytest.approx(1.0)
    assert artifact["matched_timing_available"] is False
    assert artifact["speedup_claim_allowed"] is False
    assert artifact["asp_sparse_repair_claim_allowed"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert str(TEST_PATH) in artifact["tests_added_or_reused"]
    assert artifact["research_conductor_modified"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    mod.validate_artifact(artifact)


def test_req_verify_5556_validation_fails_closed_on_unchecked_or_overclaim() -> None:
    """REQ-VERIFY-5556: validation rejects hidden model use and unsupported claims."""

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

    bad_timing = deepcopy(artifact)
    bad_timing["matched_timing_available"] = True
    bad_timing["reproducibility_checksum"] = mod.payload_checksum(bad_timing)
    with pytest.raises(ValueError, match="matched_timing_available"):
        mod.validate_artifact(bad_timing)

    speedup = deepcopy(artifact)
    speedup["speedup_claim_allowed"] = True
    speedup["reproducibility_checksum"] = mod.payload_checksum(speedup)
    with pytest.raises(ValueError, match="speedup_claim_allowed"):
        mod.validate_artifact(speedup)

    unchecked = deepcopy(artifact)
    unchecked["stable_model_checked_rate"] = 0.8
    unchecked["reproducibility_checksum"] = mod.payload_checksum(unchecked)
    with pytest.raises(ValueError, match="stable_model_checked_rate"):
        mod.validate_artifact(unchecked)

    no_signal = deepcopy(artifact)
    no_signal["random_block_success_rate"] = no_signal["descriptor_guided_success_rate"]
    no_signal["asp_sparse_repair_claim_allowed"] = True
    no_signal["reproducibility_checksum"] = mod.payload_checksum(no_signal)
    with pytest.raises(ValueError, match="asp_sparse_repair_claim_allowed"):
        mod.validate_artifact(no_signal)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_verify_5556_defensive_helpers_are_deterministic(tmp_path: Path) -> None:
    """REQ-VERIFY-5556: helper edges remain deterministic and fail closed."""

    upstream = _ready_upstream()
    descriptors = mod.build_asp_repair_descriptors(upstream)
    descriptor = descriptors["asp_repair_descriptors"][0]
    damaged = descriptor["damaged_row"]

    repaired = mod.apply_repair_assignment(damaged, descriptor["target_repair_assignment"])
    assert mod.evaluate_row_with_repair_report(repaired)["stable_model_samples"] == descriptor[
        "source_report"
    ]["stable_model_samples"]
    assert mod.ordered_domain("fact:x", ["present", "absent"], {"fact:x": "absent"}, guided=True) == [
        "absent",
        "present",
    ]
    assert mod.ordered_domain(
        "fact:x", ["present", "absent"], {"fact:x": "absent"}, guided=False
    ) == ["present", "absent"]
    assert mod.mean_iterations([]) == 0.0
    assert mod.success_rate([]) == 0.0
    assert mod.honest_verdict(False, ["unit"]).startswith("blocked:")
    with pytest.raises(ValueError, match="repair_variable"):
        mod.apply_repair_assignment(damaged, {"unknown:x": "present"})

    unready = deepcopy(upstream)
    unready["exact_fsm_fixture_extended_ready"] = False
    with pytest.raises(ValueError, match="exact_fsm_fixture_extended_ready"):
        mod.build_asp_repair_descriptors(unready)

    comparison = mod.run_policy_comparison(upstream_artifact=upstream, descriptors=descriptors)
    bad_upstream = deepcopy(upstream)
    bad_upstream["exact_fsm_fixture_extended_ready"] = False
    bad_upstream["exact_asp_validator_ready"] = False
    bad_upstream["asp_row_count"] = -1
    bad_descriptors = deepcopy(descriptors)
    bad_descriptors["descriptor_count"] = -2
    bad_comparison = deepcopy(comparison)
    bad_comparison["asp_row_count"] = -3
    bad_comparison["random_seed_count"] = -4
    bad_comparison["candidate_budget_per_attempt"] = -5
    bad_comparison["stable_model_checked_rate"] = 0.5
    bad_comparison["descriptor_guided_success_rate"] = 0.0
    bad_comparison["unchecked_repair_count"] = 1
    blockers = mod.readiness_blockers(bad_upstream, bad_descriptors, bad_comparison)
    assert {
        "upstream_asp_fsm_fixture_ready",
        "exact_asp_validator_ready",
        "descriptor_count",
        "asp_row_count",
        "random_seed_count",
        "candidate_budget_per_attempt",
        "stable_model_checked_rate",
        "descriptor_vs_random",
        "descriptor_guided_success_rate",
        "unchecked_repair_count",
    } <= set(blockers)

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
