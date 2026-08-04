"""Exp6120 outcome-committed reduced-order CSL tests.

Spec refs: REQ-LEARN-6120,
SCENARIO-LEARN-6120-STATE,
SCENARIO-LEARN-6120-SNAPSHOT,
SCENARIO-LEARN-6120-TRANSACTION,
SCENARIO-LEARN-6120-ARMS,
SCENARIO-LEARN-6120-PROMOTION,
SCENARIO-LEARN-6120-SAFETY-PARITY.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carnot import experiment_6120_outcome_committed_reduced_order_csl as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"


def test_req_6120_spec_declares_outcome_committed_contract() -> None:
    """REQ-LEARN-6120: the capability spec owns the reduced-order contract."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-6120") : text.index("## REQ-LEARN-5859")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-6120",
        "SCENARIO-LEARN-6120-STATE",
        "SCENARIO-LEARN-6120-SNAPSHOT",
        "SCENARIO-LEARN-6120-TRANSACTION",
        "SCENARIO-LEARN-6120-ARMS",
        "SCENARIO-LEARN-6120-PROMOTION",
        "SCENARIO-LEARN-6120-SAFETY-PARITY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6120_state_schema_is_fixed_width_and_bounded() -> None:
    """SCENARIO-LEARN-6120-STATE: reduced state stays fixed as history grows."""

    replay = mod.run_chronological_replay()
    schema = mod.reduced_order_state_schema_dimension_version_and_bytes(replay)

    assert schema["schema_version"] == mod.REDUCED_STATE_SCHEMA_VERSION
    assert schema["dimension"] == mod.REDUCED_STATE_DIMENSION
    assert len(schema["coordinate_names"]) == mod.REDUCED_STATE_DIMENSION
    assert schema["dimension_constant_over_history"] is True
    assert schema["raw_exact_events_runtime_state"] is False
    assert schema["raw_exact_event_audit_ledger_count"] == mod.EVENT_COUNT
    assert schema["max_serialized_state_bytes"] <= mod.REDUCED_STATE_BYTE_BOUND
    assert "task:access_control" in schema["coordinate_names"]
    assert "polarity:satisfiable" in schema["coordinate_names"]
    assert "dynamics:rollback" in schema["coordinate_names"]


def test_scenario_6120_preconditions_bind_fixtures_order_authority_and_abi() -> None:
    """REQ-LEARN-6120: preconditions hash all upstream authority."""

    receipt = mod.immutable_fixture_event_order_authority_code_and_abi_hashes()
    preconditions = mod.preconditions_checked(REPO / mod.RESULT_RELATIVE_PATH)

    assert receipt["exp5967"]["ready_score"] == 1.0
    assert receipt["exp5968"]["ready_score"] == 1.0
    assert receipt["exp5969"]["ready_score"] == 1.0
    assert receipt["event_order"]["chronological"] is True
    assert receipt["event_order"]["event_count"] == mod.EVENT_COUNT
    assert receipt["exact_outcome_authority"]["coverage_rate"] == 1.0
    assert receipt["memory_implementation"]["current_python_paths_exist"] is True
    assert receipt["abi"]["rust_paths_exist"] is True
    assert preconditions["preconditions_ready"] is True
    assert preconditions["no_llm_modules_loaded"] is True
    assert preconditions["model_weight_immutability_confirmed"] is True
    assert preconditions["root_clutter"]["root_py_file_count"] == 0
    assert preconditions["prompt_path_mismatches"]["exp5967_prompt_path_exists"] is False


def test_scenario_6120_decision_snapshots_are_read_only() -> None:
    """SCENARIO-LEARN-6120-SNAPSHOT: decisions read frozen versions only."""

    replay = mod.run_chronological_replay()
    receipts = mod.decision_snapshot_freeze_and_no_same_decision_write_receipts(replay)

    assert receipts["total_decision_count"] == len(mod.SEEDS) * mod.EVENT_COUNT * 2
    assert receipts["current_label_visible_before_decision_count"] == 0
    assert receipts["same_decision_read_after_write_count"] == 0
    assert receipts["snapshot_mutation_count"] == 0
    assert receipts["all_decisions_used_frozen_snapshot"] is True
    assert receipts["sample_receipts"]
    for sample in receipts["sample_receipts"]:
        assert sample["snapshot_hash_before"] == sample["snapshot_hash_after"]
        assert sample["label_visible_before_decision"] is False


def test_scenario_6120_transactions_commit_only_after_exact_future_outcome() -> None:
    """SCENARIO-LEARN-6120-TRANSACTION: post-outcome credit is transactional."""

    replay = mod.run_chronological_replay()
    transactions = mod.exact_post_outcome_transaction_commit_and_rollback_receipts(replay)

    assert transactions["commit_count"] > 0
    assert transactions["rollback_count"] > 0
    assert transactions["all_commits_after_exact_future_outcome"] is True
    assert transactions["no_same_decision_read_after_write"] is True
    assert transactions["transaction_hash_chain_valid"] is True
    assert transactions["rollback_exact"] is True
    for sample in transactions["sample_commit_receipts"]:
        assert sample["outcome_event_index"] > sample["decision_event_index"]
        assert sample["before_state_hash"] != sample["after_state_hash"]
        assert sample["exact_future_outcome_visible"] is True


def test_scenario_6120_matched_arms_pareto_gate_and_controls() -> None:
    """SCENARIO-LEARN-6120-ARMS/PROMOTION: write-through is the comparator."""

    replay = mod.run_chronological_replay()
    arms = mod.arm_definitions_seed_event_and_aa_determinism_counts(replay)
    utility = mod.future_event_utility_learning_speed_final_utility_and_paired_intervals(replay)
    controls = mod.write_through_delayed_fixed_shuffled_and_no_memory_controls(replay)

    assert arms["arm_names"] == list(mod.ARM_NAMES)
    assert arms["seed_count"] == len(mod.SEEDS)
    assert arms["all_arms_matched"] is True
    assert arms["aa_determinism"]["matching_checksum"] is True
    assert utility["paired_vs_write_through"]["equal_utility_lower_state_pareto"] is True
    assert utility["paired_vs_write_through"]["promotion_gate_passed"] is True
    assert utility["reduced_order_post_outcome_commit"]["final_utility"] == utility["write_through"]["final_utility"]
    assert utility["reduced_order_post_outcome_commit"]["state_bytes"] < utility["write_through"]["state_bytes"]
    assert controls["write_through"]["primary_comparator"] is True
    assert controls["delayed_commit"]["from_exp5968"] is True
    assert controls["shuffled_retrieval"]["state_volume_matched"] is True
    assert controls["no_memory"]["memory_disabled"] is True
    assert mod._ci95([0.25]) == (0.25, 0.25)
    assert mod._time_to_threshold([0.0] * mod.PROTECTED_PREFIX_COUNT) is None


def test_scenario_6120_safety_parity_feedback_and_weights_are_preserved() -> None:
    """SCENARIO-LEARN-6120-SAFETY-PARITY: utility cannot buy regressions."""

    replay = mod.run_chronological_replay()
    feedback = mod.feedback_coverage_contamination_and_state_size(replay)
    safety = mod.unsafe_accept_poison_rollback_replay_retention_and_nonforgetting_metrics(replay)
    parity = mod.python_rust_pyo3_fixed_width_abi_parity(replay)
    weights = mod.model_weight_immutability_receipt()

    assert feedback["feedback_coverage_rate"] == 1.0
    assert feedback["contamination_count"] == 0
    assert feedback["bounded_state_ok"] is True
    assert safety["unsafe_accept_count"] == 0
    assert safety["poison_propagation_count"] == 0
    assert safety["rollback"]["rollback_exact"] is True
    assert safety["replay_retention"]["protected_prefix_retention"] == 1.0
    assert safety["nonforgetting"]["nonforgetting_ready"] is True
    assert parity["all_operation_version_reason_hash_and_energy_parity"] is True
    assert parity["hardware_execution_claimed"] is False
    assert weights["all_unchanged"] is True
    assert weights["weight_update_count"] == 0


def test_req_6120_artifact_schema_ready_score_and_reproducibility(tmp_path: Path) -> None:
    """REQ-LEARN-6120: the result artifact is complete and checksummed."""

    result_path = tmp_path / "experiment_6120.json"
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
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["outcome_committed_csl_ready_score"] == 1.0
    assert artifact["retirement_triggered"] is False
    assert artifact["qualification_gate_matrix"]["all_gates_passed"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    blocked = dict(artifact)
    blocked["test_exit_codes"] = {**artifact["test_exit_codes"], "forced_failure": 1}
    blocked["outcome_committed_csl_ready_score"] = mod.ready_score(blocked)
    blocked["status"] = mod.status(blocked)
    blocked["retirement_triggered"] = mod.retirement_triggered(blocked)
    blocked["qualification_gate_matrix"] = mod.qualification_gate_matrix(blocked)
    blocked["honest_verdict"] = mod.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    assert blocked["outcome_committed_csl_ready_score"] == 0.0
    assert blocked["retirement_triggered"] is True
    assert blocked["honest_verdict"].startswith("retired:")
    with pytest.raises(ValueError, match="outcome_committed_csl_ready_score"):
        mod.validate_artifact({**artifact, "outcome_committed_csl_ready_score": 0.0})
