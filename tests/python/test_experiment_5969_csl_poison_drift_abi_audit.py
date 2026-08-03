"""Exp5969 delayed-commit CSL poison, drift, and ABI audit tests.

Spec refs: REQ-LEARN-5969, SCENARIO-LEARN-5969-GATE,
SCENARIO-LEARN-5969-ATTACKS, SCENARIO-LEARN-5969-MATCHED-ARMS,
SCENARIO-LEARN-5969-SAFETY, SCENARIO-LEARN-5969-DRIFT-RETENTION,
SCENARIO-LEARN-5969-RECOVERY, SCENARIO-LEARN-5969-PARITY,
REQ-HW-5969, SCENARIO-HW-5969.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carnot import experiment_5969_csl_poison_drift_abi_audit as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
HARDWARE_SPEC = REPO / "openspec/capabilities/hardware/spec.md"


def test_req_5969_specs_declare_attack_audit_contract() -> None:
    """REQ-LEARN-5969/REQ-HW-5969: specs anchor the audit before code."""

    self_learning = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    hardware = HARDWARE_SPEC.read_text(encoding="utf-8")
    learn_section = self_learning[
        self_learning.index("## REQ-LEARN-5969") : self_learning.index("## REQ-LEARN-5859")
    ]
    hw_section = hardware[hardware.index("### REQ-HW-5969") : hardware.index("### REQ-HW-5861")]
    normalized = " ".join((learn_section + "\n" + hw_section).split())

    for marker in (
        "REQ-LEARN-5969",
        "SCENARIO-LEARN-5969-GATE",
        "SCENARIO-LEARN-5969-ATTACKS",
        "SCENARIO-LEARN-5969-MATCHED-ARMS",
        "SCENARIO-LEARN-5969-SAFETY",
        "SCENARIO-LEARN-5969-DRIFT-RETENTION",
        "SCENARIO-LEARN-5969-RECOVERY",
        "SCENARIO-LEARN-5969-PARITY",
        "REQ-HW-5969",
        "SCENARIO-HW-5969",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in learn_section + hw_section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in learn_section or f"`{field}`" in hw_section
        assert " ".join(principle.split()) in normalized


def test_scenario_5969_gate_binds_exact_exp5968_state_ledger_and_abi() -> None:
    """SCENARIO-LEARN-5969-GATE: attacks start only from ready Exp5968."""

    receipt = mod.gate_replay_receipt()
    hashes = mod.selected_policy_state_ledger_and_abi_hashes()
    preconditions = mod.preconditions_checked(REPO / mod.RESULT_RELATIVE_PATH)

    assert receipt["path"] == mod.EXP5968_RESULT_RELATIVE_PATH.as_posix()
    assert receipt["ready_score"] == 1.0
    assert receipt["status"] == "complete_ready"
    assert receipt["gate_passed"] is True
    assert hashes["selected_policy"] == "delayed_commit"
    assert hashes["one_immutable_clean_starting_point"] is True
    assert hashes["exp5968"]["prospective_csl_ready_score"] == 1.0
    assert hashes["exp5926"]["abi_ready_score"] == 1.0
    assert preconditions["preconditions_ready"] is True
    assert preconditions["llm_loaded"] is False
    assert preconditions["protected_prefix_corpus"]["protected_prefix_count"] == mod.PROTECTED_PREFIX_COUNT


def test_scenario_5969_attack_manifest_is_sealed_before_execution() -> None:
    """SCENARIO-LEARN-5969-ATTACKS: attack families and seeds are frozen."""

    manifest = mod.preregistered_attack_manifest_and_seeds()
    families = set(manifest["attack_families"])

    assert manifest["sealed_before_execution"] is True
    assert manifest["attack_seed"] == mod.ATTACK_SEED
    assert manifest["attack_seed_seal"].startswith("sha256:")
    assert families == set(mod.ATTACK_FAMILIES)
    assert manifest["max_poison_rate"] <= mod.MAX_POISON_RATE
    assert manifest["outcome_fields_present_before_execution"] is False
    for name in mod.ATTACK_FAMILIES:
        assert manifest["attacks"][name]["seed"] == mod.ATTACK_SEED
        assert manifest["attacks"][name]["expected_selected_policy_reason"]


def test_scenario_5969_matched_arms_expose_poison_without_selected_propagation() -> None:
    """SCENARIO-LEARN-5969-SAFETY: delayed commit rejects poison propagation."""

    replay = mod.run_attacked_replay()
    matching = mod.delayed_commit_write_through_fixed_and_clean_arm_matching(replay)
    metrics = mod.poison_admission_propagation_detection_and_quarantine_metrics(replay)

    assert matching["all_arms_matched"] is True
    assert matching["arm_names"] == list(mod.ARM_NAMES)
    assert len(set(matching["per_arm_capacity"].values())) == 1
    assert len(set(matching["per_arm_attack_budget"].values())) == 1
    assert metrics["selected_policy"] == "delayed_commit"
    assert metrics["delayed_commit"]["unsafe_accept_count"] == 0
    assert metrics["delayed_commit"]["poison_propagation_count"] == 0
    assert metrics["delayed_commit"]["quarantine_precision"] == 1.0
    assert metrics["delayed_commit"]["quarantine_recall"] == 1.0
    assert metrics["same_event_write_through"]["unsafe_accept_count"] > 0
    assert metrics["same_event_write_through"]["poison_propagation_count"] > 0
    assert mod.unsafe_accept_count(replay) == 0
    assert mod.poison_propagation_count(replay) == 0


def test_scenario_5969_drift_retention_conflict_capacity_metrics_are_explicit() -> None:
    """SCENARIO-LEARN-5969-DRIFT-RETENTION: recovery is distinct from retention."""

    replay = mod.run_attacked_replay()
    drift = mod.abrupt_gradual_drift_and_recovery_metrics(replay)
    lifecycle = mod.conflict_duplicate_stale_capacity_and_eviction_metrics(replay)
    retention = mod.protected_prefix_and_clean_utility_retention(replay)

    assert drift["abrupt_drift"]["recovered"] is True
    assert drift["gradual_drift"]["recovered"] is True
    assert drift["post_attack_recovery"]["selected_policy_recovered"] is True
    assert lifecycle["all_lifecycle_edges_fail_closed"] is True
    assert lifecycle["delayed_commit"]["conflict_rejection_count"] > 0
    assert lifecycle["delayed_commit"]["duplicate_rejection_count"] > 0
    assert lifecycle["delayed_commit"]["stale_replay_rejection_count"] > 0
    assert lifecycle["delayed_commit"]["capacity_eviction_count"] > 0
    assert retention["selected_policy_retention_ready"] is True
    assert retention["delayed_commit"]["protected_prefix_retention"] >= retention["floor"]
    assert retention["delayed_commit"]["clean_utility_retention"] >= mod.CLEAN_UTILITY_RETENTION_FLOOR


def test_scenario_5969_recovery_parity_and_hardware_mapping_fail_closed() -> None:
    """SCENARIO-LEARN-5969-PARITY/SCENARIO-HW-5969: ABI traces match exactly."""

    replay = mod.run_attacked_replay()
    matrix = mod.crash_restart_tamper_and_rollback_matrix(replay)
    parity = mod.python_rust_pyo3_attacked_trace_parity(replay)
    hardware = mod.hardware_abi_mapping_receipt(parity)

    assert matrix["all_fail_closed_or_exactly_recovered"] is True
    assert matrix["crash_phase_count"] == len(mod.CRASH_PHASES)
    assert matrix["ledger_tamper"]["tamper_detected"] is True
    assert matrix["rollback_after_late_failure"]["rollback_exact"] is True
    assert parity["all_operation_version_reason_hash_and_energy_parity"] is True
    assert parity["fail_closed_on_unsupported_operation"] is True
    assert parity["parity_failures"] == []
    assert set(parity["backend_receipts"]) == {"python", "rust", "pyo3"}
    assert len({receipt["final_energy"] for receipt in parity["backend_receipts"].values()}) == 1
    assert hardware["fixed_width_portability_only"] is True
    assert hardware["hardware_execution_claimed"] is False
    assert hardware["tsu_execution_claimed"] is False


def test_req_5969_artifact_schema_ready_score_and_reproducibility(tmp_path: Path) -> None:
    """REQ-LEARN-5969: the result artifact is complete, ready, and checksummed."""

    result_path = tmp_path / "experiment_5969.json"
    artifact = mod.run(
        result_path=result_path,
        duration_s=0.0,
        test_commands=mod.DEFAULT_TEST_COMMANDS,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=True,
    )

    assert result_path.is_file()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["unsafe_accept_count"] == 0
    assert artifact["poison_propagation_count"] == 0
    assert artifact["rollback_and_recovery_ready_score"] == 1.0
    assert artifact["retirement_decision"]["promotion_readiness_retired"] is False
    assert artifact["retirement_decision"]["exp5968_clean_result_preserved"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    blocked = dict(artifact)
    blocked["test_exit_codes"] = {**artifact["test_exit_codes"], "forced_failure": 1}
    blocked["rollback_and_recovery_ready_score"] = mod.ready_score(blocked)
    blocked["status"] = mod.status(blocked)
    blocked["honest_verdict"] = mod.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    assert blocked["rollback_and_recovery_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("complete_partial:")
    retired = {**artifact, "retirement_decision": {"promotion_readiness_retired": True}}
    assert mod.honest_verdict(retired).startswith("retired:")
    with pytest.raises(ValueError, match="rollback_and_recovery_ready_score"):
        mod.validate_artifact({**artifact, "rollback_and_recovery_ready_score": 0.0})
