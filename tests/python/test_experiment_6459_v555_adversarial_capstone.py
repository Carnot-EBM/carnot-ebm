"""Tests for Exp6459 V555 adversarial capstone.

Spec refs: REQ-CAPSTONE-6459,
SCENARIO-CAPSTONE-6459-INVENTORY,
SCENARIO-CAPSTONE-6459-ROW-RECOMPUTATION,
SCENARIO-CAPSTONE-6459-CLAIM-DECISIONS,
SCENARIO-CAPSTONE-6459-ATTACKS,
SCENARIO-CAPSTONE-6459-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6459_v555_adversarial_capstone as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _candidate_row(
    *,
    partition: str,
    model: str,
    problem_id: str,
    candidate_id: str,
    exact_success: bool,
    eligible: bool = True,
) -> dict[str, Any]:
    return {
        "partition": partition,
        "model_hf_id": model,
        "problem_id": problem_id,
        "candidate_id": candidate_id,
        "exact_success": exact_success,
        "eligible": eligible,
        "parse_valid": True,
        "raw_candidate_sha256": f"sha256:{partition}{problem_id}{candidate_id}".ljust(71, "0"),
    }


def _csl_row(
    *,
    unit: str,
    arm: str,
    exact_success: bool,
    future: bool = True,
    corrupt: bool = False,
) -> dict[str, Any]:
    return {
        "row_id": f"{unit}::{arm}",
        "unit_id": unit,
        "model": "fixture/model",
        "arm": arm,
        "future_eval_unit": future,
        "future_exact_outcome": exact_success if future else None,
        "exact_result": {
            "exact_success": exact_success,
            "protected_ok": True,
            "abstained": False,
        },
        "protected_outcome": {"protected_ok": True},
        "exact_sign": 1 if exact_success else -1,
        "applied_update_sign": 1 if exact_success and "frozen" not in arm else 0,
        "teacher_signal": {
            "signed_direction": -1,
            "sign_is_authoritative": False,
            "nonnegative_magnitude_evidence": 0.5,
        },
        "selection_used_post_update_state": False,
        "corrupt_event": {"scheduled": corrupt, "detected": corrupt},
        "checker_response": {"authoritative": not corrupt, "transport_corrupted": corrupt},
        "quarantine": {"quarantined": corrupt},
        "rollback": {"restored_last_good_head": corrupt, "rejected_child_head": "bad" if corrupt else ""},
        "tombstone": {"written": corrupt},
        "process": {
            "exit_code": 0,
            "recovered_from_disk": True,
            "head_hash_valid": True,
            "transaction_ancestry_valid": True,
            "inherited_memory_state_visible": False,
        },
        "path_receipts": {"path_hash_matches": not corrupt},
        "cpu_fallback": False,
        "timing": {"duration_s": 0.01},
    }


def _arc_row(
    *,
    game: str,
    prefix: str,
    arm: str,
    collision: bool,
    reachable: bool,
    source_access: int = 0,
) -> dict[str, Any]:
    return {
        "row_id": f"{game}:{prefix}|arm:{arm}",
        "game": game,
        "prefix_id": prefix,
        "seed": 6458001,
        "arm": arm,
        "state_collision": collision,
        "recorded_next_state_reachability": reachable,
        "legal_action_set": [1, 2],
        "chosen_action": 1,
        "timeout": False,
        "action_cost": 1,
        "source_access_count": source_access,
        "offline_ground_truth_bfs_count": 0,
        "per_game_adapter_count": 0,
        "recorded_next_state_used_before_action": False,
    }


def test_req_capstone_6459_spec_declares_fields_and_scenarios() -> None:
    """REQ-CAPSTONE-6459: OpenSpec owns the Exp6459 contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-6459") :]

    for marker in (
        "SCENARIO-CAPSTONE-6459-INVENTORY",
        "SCENARIO-CAPSTONE-6459-ROW-RECOMPUTATION",
        "SCENARIO-CAPSTONE-6459-CLAIM-DECISIONS",
        "SCENARIO-CAPSTONE-6459-ATTACKS",
        "SCENARIO-CAPSTONE-6459-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
    for claim in mod.CLAIM_FIELDS:
        assert f"claim_gate:{claim}" in mod.FIELD_PRINCIPLES


def test_scenario_capstone_6459_inventory_preserves_missing_blocked_and_flagged() -> None:
    """SCENARIO-CAPSTONE-6459-INVENTORY: every task state stays separate."""

    tasks = mod.load_v555_tasks(REPO)
    inventory = mod.inventory_task_artifacts(REPO, tasks)

    assert [task["id"] for task in tasks] == list(mod.V555_TASK_IDS)
    assert len(inventory) == 12
    assert inventory["exp6452-representation-objective-causal-ab"]["artifact_state"] == "missing"
    assert inventory["exp6454-held-exact-constraint-energy-selection-ab"]["artifact_state"] == "missing"
    assert inventory["exp6451-typed-fact-grounding-fixed-policy-logic-ab"]["artifact_state"] == "blocked"
    assert "sota_corpus_ready_score" in inventory[
        "exp6451-typed-fact-grounding-fixed-policy-logic-ab"
    ]["gate_check_summary"]
    assert inventory["exp6450-sota-fixed-policy-candidate-corpus"]["readiness_fields"][
        "sota_corpus_ready_score"
    ] == 0.0
    assert inventory["exp6450-sota-fixed-policy-candidate-corpus"]["current_critical_count"] >= 1
    assert inventory["exp6457-independent-verifier-bounded-csl-audit"]["current_critical_count"] >= 1
    assert inventory["exp6458-arc-representation-objective-generalization-ab"]["row_count"] == 544


def test_scenario_capstone_6459_row_reducers_ignore_upstream_aggregates() -> None:
    """SCENARIO-CAPSTONE-6459-ROW-RECOMPUTATION: rows are metric truth."""

    corpus = {
        "candidate_headroom_by_partition": {"development": {"success": -999}},
        "per_unit_rows": {
            "rows": [
                _candidate_row(
                    partition="development",
                    model="m",
                    problem_id="p0",
                    candidate_id="c0",
                    exact_success=False,
                ),
                _candidate_row(
                    partition="development",
                    model="m",
                    problem_id="p0",
                    candidate_id="c1",
                    exact_success=True,
                ),
                _candidate_row(
                    partition="selection_held",
                    model="m",
                    problem_id="p1",
                    candidate_id="c0",
                    exact_success=False,
                ),
            ]
        },
    }
    prospective = {
        "future_exact_yield_delta": {"verifier_bounded_minus_frozen": -999},
        "per_unit_rows": {
            "rows": [
                _csl_row(unit="u0", arm=mod.FROZEN_ARM, exact_success=False),
                _csl_row(unit="u0", arm=mod.TEACHER_ARM, exact_success=False),
                _csl_row(unit="u0", arm=mod.VERIFIER_ARM, exact_success=True),
            ]
        },
    }
    held = {
        "future_exact_yield_delta": {"clean_minus_frozen": -999},
        "per_unit_rows": {
            "rows": [
                _csl_row(unit="h0", arm=mod.FROZEN_ARM, exact_success=False),
                _csl_row(unit="h0", arm=mod.CLEAN_ARM, exact_success=True),
                _csl_row(unit="h0", arm=mod.GOVERNED_ARM, exact_success=True, corrupt=True),
            ]
        },
    }
    arc = {
        "collision_rates_by_arm": {mod.ARC_BASELINE_ARM: {"rate": -999}},
        "per_unit_rows": [
            _arc_row(
                game="g0",
                prefix="p0",
                arm=mod.ARC_BASELINE_ARM,
                collision=True,
                reachable=False,
            ),
            _arc_row(
                game="g0",
                prefix="p0",
                arm=mod.ARC_COMBINED_ARM,
                collision=False,
                reachable=True,
            ),
        ],
    }

    corpus_metrics = mod.reduce_sota_corpus(corpus)
    prospective_metrics = mod.reduce_prospective_csl(prospective)
    held_metrics = mod.reduce_held_csl_safety(held)
    arc_metrics = mod.reduce_arc_generalization(arc)

    assert corpus_metrics["candidate_headroom_by_partition"]["development"]["success"] == 1
    assert corpus_metrics["candidate_headroom_by_partition"]["development"]["has_headroom"] is True
    assert corpus_metrics["candidate_headroom_by_partition"]["selection_held"]["has_headroom"] is False
    assert prospective_metrics["future_exact_yield_delta"]["verifier_bounded_minus_frozen"] == 1.0
    assert prospective_metrics["future_exact_yield_delta"]["verifier_bounded_minus_teacher"] == 1.0
    assert held_metrics["future_exact_yield_delta"]["clean_minus_frozen"] == 1.0
    assert held_metrics["corrupt_event_count"] == 1
    assert held_metrics["quarantined_corrupt_event_count"] == 1
    assert arc_metrics["collision_rates_by_arm"][mod.ARC_BASELINE_ARM]["rate"] == 1.0
    assert arc_metrics["collision_rates_by_arm"][mod.ARC_COMBINED_ARM]["rate"] == 0.0
    assert arc_metrics["held_next_state_reachability_by_arm"][mod.ARC_COMBINED_ARM]["rate"] == 1.0


def test_scenario_capstone_6459_claims_do_not_pool_missing_evidence(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6459-CLAIM-DECISIONS: branches stay independent."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        date="20260815",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.25,
        tests_run=[{"command": mod.FOCUSED_TEST_COMMAND, "exit_code": 0}],
        write=True,
    )

    assert artifact["v555_capstone_ready_score"] == 1.0
    assert artifact["status"] == "complete_blocked"
    for claim in mod.CLAIM_FIELDS:
        decision = artifact[claim]
        assert decision["eligible"] is False
        assert decision["reasons"]
    assert "exp6451_blocked_gate_check_failed" in artifact["claim_ineligibility_reasons"][
        "typed_grounding_claim_eligibility"
    ]
    assert "exp6452_artifact_missing" in artifact["claim_ineligibility_reasons"][
        "objective_causal_claim_eligibility"
    ]
    assert "exp6453_blocked_gate_check_failed" in artifact["claim_ineligibility_reasons"][
        "held_allocation_claim_eligibility"
    ]
    assert "exp6454_artifact_missing" in artifact["claim_ineligibility_reasons"][
        "energy_selection_claim_eligibility"
    ]
    assert "exp6457_csl_audit_ready_score_not_1" in artifact["claim_ineligibility_reasons"][
        "prospective_csl_claim_eligibility"
    ]
    assert "exp6458_arc_ready_score_not_1" in artifact["claim_ineligibility_reasons"][
        "internal_arc_generalization_claim_eligibility"
    ]
    assert "v555_contains_no_public_arc_evidence_task" in artifact["claim_ineligibility_reasons"][
        "public_arc_claim_eligibility"
    ]
    assert "v555_contains_no_hardware_evidence_task" in artifact["claim_ineligibility_reasons"][
        "hardware_claim_eligibility"
    ]


def test_scenario_capstone_6459_attacks_joint_rows_and_schema(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6459-ATTACKS: attacks fail closed and rows are visible."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        date="20260815",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.25,
        tests_run=[{"command": mod.FOCUSED_TEST_COMMAND, "exit_code": 0}],
        write=False,
    )
    rows = artifact["per_unit_rows"]
    attacks = artifact["current_adversarial_attack_replay"]

    assert {row["row_type"] for row in rows} >= {"task", "claim"}
    assert len([row for row in rows if row["row_type"] == "task"]) == 12
    assert len([row for row in rows if row["row_type"] == "claim"]) == len(mod.CLAIM_FIELDS)
    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert all(row["fail_closed"] for row in attacks["rows"])
    assert all(not row["claim_promoted_by_attack"] for row in attacks["rows"])
    assert artifact["joint_pathway_rows_and_cofailure_moments"]["independence_assumed"] is False
    assert artifact["joint_pathway_rows_and_cofailure_moments"]["marginal_reliability_multiplied"] is False
    assert artifact["terminal_determination_preservation"]["exp6457"]["honest_verdict"].startswith(
        "complete:"
    )
    assert artifact["current_adversarial_findings"]["critical_count"] == 0


def test_scenario_capstone_6459_field_principles_and_checksum(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6459-FIELD-PRINCIPLES: artifact is self-checking."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        date="20260815",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.25,
        tests_run=[{"command": mod.FOCUSED_TEST_COMMAND, "exit_code": 0}],
        write=False,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_provenance"])
    for claim in mod.CLAIM_FIELDS:
        assert f"claim_gate:{claim}" in artifact["field_principles"]
        assert any(row["claim"] == claim for row in artifact["per_unit_rows"])
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete_blocked:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.payload_checksum(artifact) == mod.payload_checksum(artifact)


def test_req_capstone_6459_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-CAPSTONE-6459: defensive helper branches stay explicit."""

    zero = tmp_path / "zero.json"
    zero.write_bytes(b"")
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")

    assert mod._exact_success({"exact_result": {"exact_success": True}}) is True
    assert mod._protected_ok({"protected_outcome": {"protected_ok": False}}) is False
    assert mod._rows({}) == []
    assert mod._read_json(zero) == ({}, False, "zero_byte")
    assert mod._read_json(malformed)[1:] == (False, "json_decode_error:Expecting property name enclosed in double quotes")
    assert mod._row_count({"per_unit_rows": {"rows": [{}, {}]}}) == 2
    assert mod._embedded_findings({"current_adversarial_findings": {"critical_count": 1}}) == [
        {"critical_count": 1}
    ]
    assert mod._embedded_findings({"flagged_adversarial": True}) == [
        {"kind": "flagged_adversarial", "severity": "critical"}
    ]
    assert (
        mod._artifact_state("exp0000-x", zero, {}, False, "zero_byte", [])
        == "zero_byte"
    )
    assert (
        mod._artifact_state("exp0000-x", malformed, {}, False, "json_error", [])
        == "malformed"
    )
    assert (
        mod._artifact_state(
            "exp6459-v555-adversarial-capstone",
            tmp_path / "future.json",
            {},
            False,
            "missing",
            [],
        )
        == "self_pending"
    )
    assert mod._adversarial_verify_report(tmp_path / "missing.json")["ran"] is False
    assert mod._required_fields_section("no required fields here") == ""

    bad_gate = mod.gate_producer_contract_validation(
        [
            {"id": "upstream", "prompt": "REQUIRED ARTIFACT FIELDS:\nstatus"},
            {
                "id": "downstream",
                "gated_on": [{"upstream": "upstream", "artifact_field": "missing_score"}],
            },
        ]
    )
    assert bad_gate["ok"] is False
    assert bad_gate["failures"][0]["producer_declares_field"] is False

    prior = mod.prior_failure_and_exclusion_validation(
        REPO,
        [{"id": "exp9999-bad", "prior_failures": [{"experiment_id": "exp1"}]}],
        {
            "exp6448-v555-terminal-handoff-and-queue-integrity": {
                "artifact_state": "complete",
                "readiness_fields": {},
            }
        },
    )
    assert prior["ok"] is False
    assert prior["failures"][0]["ok"] is False

    arc = mod.reduce_arc_generalization(
        {
            "per_unit_rows": [
                _arc_row(
                    game="g1",
                    prefix="p1",
                    arm=mod.ARC_BASELINE_ARM,
                    collision=False,
                    reachable=False,
                )
                | {"timeout": True}
            ]
        }
    )
    assert arc["legal_action_coverage_by_arm"][mod.ARC_BASELINE_ARM]["rate"] == 1.0

    minimal_inventory = {
        task_id: {"artifact_state": "complete", "readiness_fields": {}, "current_critical_count": 0, "payload": {}}
        for task_id in mod.V555_TASK_IDS
    }
    minimal_inventory["exp6451-typed-fact-grounding-fixed-policy-logic-ab"]["artifact_state"] = "blocked"
    minimal_inventory["exp6452-representation-objective-causal-ab"]["artifact_state"] = "missing"
    minimal_inventory["exp6453-held-verifier-budget-allocation-ab"]["artifact_state"] = "blocked"
    minimal_inventory["exp6454-held-exact-constraint-energy-selection-ab"]["artifact_state"] = "missing"
    minimal_inventory["exp6458-arc-representation-objective-generalization-ab"]["payload"] = {
        "no_game_or_level_solve_claim": True
    }
    decisions = mod.make_claim_decisions(minimal_inventory)
    assert "exp6455_verifier_bounded_csl_ready_score_not_1" in decisions[
        "prospective_csl_claim_eligibility"
    ]["reasons"]
    assert "exp6456_csl_safety_replication_ready_score_not_1" in decisions[
        "held_csl_safety_claim_eligibility"
    ]["reasons"]
