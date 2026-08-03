"""Exp5968 delayed-commit prospective CSL tests.

Spec refs: REQ-LEARN-5968, SCENARIO-LEARN-5968-GATE,
SCENARIO-LEARN-5968-CHRONOLOGY, SCENARIO-LEARN-5968-ARMS,
SCENARIO-LEARN-5968-CONTROLS, SCENARIO-LEARN-5968-PROMOTION.
"""

from __future__ import annotations

from pathlib import Path

from carnot import experiment_5968_delayed_commit_csl_prospective as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"


def test_req_5968_spec_declares_prospective_gate_contract() -> None:
    """REQ-LEARN-5968: the capability spec owns the gate before code runs."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-5968") : text.index("## REQ-LEARN-5859")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5968",
        "SCENARIO-LEARN-5968-GATE",
        "SCENARIO-LEARN-5968-CHRONOLOGY",
        "SCENARIO-LEARN-5968-ARMS",
        "SCENARIO-LEARN-5968-CONTROLS",
        "SCENARIO-LEARN-5968-PROMOTION",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_5968_gate_preconditions_bind_stream_state_and_abi() -> None:
    """SCENARIO-LEARN-5968-GATE: upstream receipts are exact before replay."""

    receipt = mod.gate_replay_receipt()
    immutable = mod.immutable_stream_state_abi_hashes()
    preconditions = mod.preconditions_checked(REPO / mod.RESULT_RELATIVE_PATH)

    assert receipt["ready_score"] == 1.0
    assert receipt["path"] == mod.EXP5967_RESULT_RELATIVE_PATH.as_posix()
    assert receipt["gate_passed"] is True
    assert immutable["exp5920"]["row_count"] == 198
    assert immutable["exp5920"]["prefix_chain_valid"] is True
    assert immutable["exp5924"]["ready_score"] == 1.0
    assert immutable["exp5926"]["ready_score"] == 1.0
    assert preconditions["preconditions_ready"] is True
    assert preconditions["llm_loaded"] is False
    assert preconditions["seeds"] == list(mod.SEEDS)


def test_scenario_5968_arms_are_matched_and_predictions_are_pre_event() -> None:
    """SCENARIO-LEARN-5968-ARMS: treatment arms differ only by memory policy."""

    replay = mod.run_five_seed_replay()
    matching = mod.five_arm_capacity_compute_and_event_matching(replay)
    timing = mod.pre_event_prediction_and_post_seal_label_timing(replay)

    assert set(replay["replicates"]) == set(mod.SEEDS)
    assert matching["all_arms_matched"] is True
    assert matching["arm_names"] == list(mod.ARM_NAMES)
    assert len(set(matching["per_arm_retrieval_count"].values())) == 1
    assert len(set(matching["per_arm_verifier_call_count"].values())) == 1
    assert timing["pre_event_prediction_count"] == len(mod.ARM_NAMES) * len(mod.SEEDS) * 198
    assert timing["current_label_visible_before_prediction_count"] == 0
    assert timing["proposal_sealed_before_label_reveal_count"] == timing["pre_event_prediction_count"]


def test_scenario_5968_future_validation_excludes_same_event_and_protected_prefix() -> None:
    """SCENARIO-LEARN-5968-CHRONOLOGY: delayed credit is future-only."""

    replay = mod.run_five_seed_replay()
    contract = mod.semantic_neighborhood_future_validation_contract(replay)
    lifecycle = mod.promotion_rejection_quarantine_state_growth_and_retrieval_metrics(replay)
    metrics = mod.per_arm_prequential_learning_curve_and_final_metrics(replay)

    assert contract["all_delayed_promotions_future_disjoint"] is True
    assert contract["same_event_validator_count"] == 0
    assert contract["protected_prefix_validator_count"] == 0
    assert contract["promoted_update_count"] > 0
    assert lifecycle["delayed_commit"]["promotion_count"] > 0
    assert lifecycle["delayed_commit"]["quarantine_count"] > 0
    assert lifecycle["delayed_commit"]["retrieval_hit_utility"] > 0.0
    assert metrics["same_event_write_through"]["online_auc"] > metrics["delayed_commit"]["online_auc"]


def test_scenario_5968_controls_and_paired_intervals_credit_delayed_only() -> None:
    """SCENARIO-LEARN-5968-CONTROLS: shortcuts and state volume are uncredited."""

    replay = mod.run_five_seed_replay()
    controls = mod.label_order_retrieval_same_event_noop_capacity_and_random_controls(replay)
    deltas = mod.paired_deltas_intervals_and_power(replay)
    retention = mod.protected_prefix_retention(replay)

    assert controls["same_event_only_utility"]["credited_to_delayed_commit"] is False
    assert controls["same_event_only_utility"]["delayed_same_event_credit_count"] == 0
    assert controls["retrieval_shuffle"]["state_volume_matched"] is True
    assert controls["retrieval_shuffle"]["explains_delayed_lift"] is False
    assert controls["label_permutation"]["improvement_vanishes"] is True
    assert controls["random_admission"]["explains_delayed_lift"] is False
    assert deltas["promotion_gate_passed"] is True
    assert deltas["delayed_commit_vs_no_memory"]["online_auc_delta_ci95"][0] > 0.0
    assert deltas["delayed_commit_vs_fixed_validated_memory"]["online_auc_delta_ci95"][0] > 0.0
    assert retention["delayed_commit"]["retention"] >= retention["no_memory"]["retention"]
    assert retention["not_regressed"] is True
    assert mod._ci95([0.5]) == (0.5, 0.5)
    assert mod._time_to_threshold([0.0] * mod.PROTECTED_PREFIX_COUNT) is None


def test_req_5968_artifact_schema_ready_score_and_reproducibility(tmp_path: Path) -> None:
    """REQ-LEARN-5968: the result artifact is complete, ready, and checksummed."""

    result_path = tmp_path / "experiment_5968.json"
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
    assert artifact["prospective_csl_ready_score"] == 1.0
    assert artifact["immutable_model_weights_receipt"]["all_unchanged"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
