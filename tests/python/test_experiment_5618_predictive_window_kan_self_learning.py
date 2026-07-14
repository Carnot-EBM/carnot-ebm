"""Tests for Exp5618 predictive-window KAN self-learning controller.

Spec refs: REQ-LEARN-5618,
SCENARIO-LEARN-5618-CAUSAL-CONTROLLER,
SCENARIO-LEARN-5618-CONTROLS,
SCENARIO-LEARN-5618-SAFETY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5618_predictive_window_kan_self_learning as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5618_predictive_window_kan_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5618_predictive_window_kan_self_learning.py "
    "-m pytest tests/python/test_experiment_5618_predictive_window_kan_self_learning.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5618_predictive_window_kan_self_learning.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5618_predictive_window_kan_self_learning.json"
)
TESTS_ADDED_OR_REUSED = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
]


def _artifact(tmp_path: Path) -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        checkpoint_dir=tmp_path / "checkpoints",
    )


def test_req_learn_5618_spec_declares_predictive_window_contract() -> None:
    """REQ-LEARN-5618: OpenSpec anchors fields, gates, arms, and substrate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5618") :]

    for marker in (
        "REQ-LEARN-5618",
        "SCENARIO-LEARN-5618-CAUSAL-CONTROLLER",
        "SCENARIO-LEARN-5618-CONTROLS",
        "SCENARIO-LEARN-5618-SAFETY",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "future-aware oracle selector",
        "held-out stream roster",
        "future labels, held-out outcomes, and external teachers as excluded",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle == mod.FIELD_PRINCIPLES[field]


def test_scenario_learn_5618_controller_contract_freezes_roster_without_leakage() -> None:
    """SCENARIO-LEARN-5618-CAUSAL-CONTROLLER: roster and features freeze before outcomes."""

    gates = mod.freeze_predictive_window_gates(root=REPO)
    contract = gates["controller_feature_contract"]

    assert gates["heldout_roster_frozen_before_outcomes"] is True
    assert gates["outcome_fields_materialized_for_contract"] is False
    assert gates["minimum_learner_seeds"] == 5
    assert gates["instances_per_condition_floor"] == 32
    assert contract["future_leakage_excluded"] is True
    assert contract["excluded_sources"] == list(mod.EXCLUDED_CONTROLLER_SOURCES)
    assert contract["allowed_feature_families"] == list(mod.ALLOWED_FEATURE_FAMILIES)
    assert set(contract["action_space"]) == set(mod.ACTION_NAMES)
    assert gates["heldout_stream_roster_count"] == 288
    assert len(gates["heldout_stream_roster_sha256"]) == 71

    fixture = mod.load_predictive_fixture(gates, root=REPO)
    assert fixture.heldout_replicates_per_condition == 40
    with pytest.raises(ValueError, match="controller_feature_contract_frozen"):
        mod.load_predictive_fixture({"controller_feature_contract_frozen": False}, root=REPO)
    leaked = deepcopy(gates)
    leaked["controller_feature_contract"]["future_leakage_excluded"] = False
    with pytest.raises(ValueError, match="controller_feature_contract"):
        mod.load_predictive_fixture(leaked, root=REPO)


def test_scenario_learn_5618_controller_beats_fixed_and_oracle_is_ceiling(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5618-CONTROLS: controller compares against fixed arms and oracle."""

    result = mod.run_predictive_window_experiment(
        mod.load_predictive_fixture(mod.freeze_predictive_window_gates(root=REPO), root=REPO),
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert result["models_tested"]["fixed_exp5617_arms"] == list(mod.FIXED_ARM_NAMES)
    assert result["models_tested"]["causal_controller"] == mod.CONTROLLER_ARM
    assert result["models_tested"]["future_aware_oracle"] == mod.ORACLE_ARM
    assert set(mod.CONTROL_ARM_NAMES) <= set(result["models_tested"]["controls"])
    assert result["seeds"] == list(mod.DEFAULT_LEARNER_SEEDS)
    assert result["instances_per_condition"]["replicated_heldout_streams"] >= 32
    assert result["optimization_budget"]["matched_across_non_oracle_arms"] is True
    assert result["optimization_budget"]["exact_validation_calls_matched"] is True

    assert set(mod.FIXED_ARM_NAMES) <= set(result["ale_by_arm"])
    assert mod.CONTROLLER_ARM in result["ale_by_arm"]
    assert mod.ORACLE_ARM in result["ale_by_arm"]
    assert result["delta_ale_vs_best_fixed"]["mean"] > 0.0
    assert result["regret_to_oracle"]["mean"] > 0.0
    assert result["oracle_selector"]["future_aware"] is True
    assert result["oracle_selector"]["excluded_from_headline"] is True
    assert result["valid_adaptation_latency"][mod.CONTROLLER_ARM]["mean"] <= result[
        "valid_adaptation_latency"
    ][mod.RESET_ARM]["mean"]
    assert result["forward_transfer_delta"]["mean"] > 0.0
    assert result["backward_retention_delta"]["mean"] > 0.0
    assert result["forgetting_delta"]["mean"] <= 0.0
    assert result["update_frequency"][mod.CONTROLLER_ARM]["mean"] > 0.0
    assert result["rollback_burden"][mod.CONTROLLER_ARM]["mean"] >= 0.0
    assert result["compute_memory_cost"][mod.CONTROLLER_ARM]["memory_bytes"] > 0

    ledger = result["immutable_decision_ledger"]
    assert ledger
    assert all(row["ledger_hash"] == mod.decision_ledger_hash(row) for row in ledger)
    assert all(row["chosen_action"] in mod.ACTION_NAMES for row in ledger)
    assert all(set(row["feature_names"]).issubset(mod.CONTROLLER_FEATURE_NAMES) for row in ledger)
    assert all(not set(row["feature_names"]).intersection(mod.FORBIDDEN_FEATURE_NAMES) for row in ledger)


def test_scenario_learn_5618_safety_checkpoint_and_replay_controls(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5618-SAFETY: poison, rollback, replay, and regression gates hold."""

    artifact = _artifact(tmp_path)

    assert artifact["unsafe_false_accept_count"]["total"] == 0
    assert artifact["poison_update_disposition"]["accepted"] == 0
    assert artifact["poison_update_disposition"]["rejected"] > 0
    assert artifact["poison_update_disposition"]["rolled_back"] >= 1
    assert artifact["rollback_positive_control"]["passed"] is True
    assert artifact["delayed_regression_passed"] is True
    assert artifact["recurring_valid_rule_recovered"] is True
    assert artifact["lazy_identity_guard_passed"] is True
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["adversarial_scenarios"]["poison"]["present"] is True
    assert artifact["adversarial_scenarios"]["transient_drift"]["present"] is True
    assert artifact["adversarial_scenarios"]["recurring_old_rule"]["present"] is True
    assert artifact["adversarial_scenarios"]["delayed_regression"]["present"] is True
    assert mod.verify_checkpoint_replay(artifact["checkpoint_replay_receipts"]) is True
    assert all(
        row["checkpoint_hash"] == mod.sha256_file(row["checkpoint_path"])
        for row in artifact["checkpoint_replay_receipts"]
    )
    bad_file_hash = deepcopy(artifact["checkpoint_replay_receipts"])
    bad_file_hash[0]["checkpoint_hash"] = "sha256:bad"
    assert mod.verify_checkpoint_replay(bad_file_hash) is False
    bad_model_hash = deepcopy(artifact["checkpoint_replay_receipts"])
    bad_model_hash[0]["model_checksum"] = "bad"
    assert mod.verify_checkpoint_replay(bad_model_hash) is False


def test_scenario_learn_5618_artifact_fields_stable_write_and_validation(tmp_path: Path) -> None:
    """REQ-LEARN-5618: artifact exposes required fields and validates gates."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(
        root=REPO,
        result_path=destination,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        checkpoint_dir=tmp_path / "checkpoints",
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert mod.validate_artifact(artifact) is True
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field] == mod.REQUIRED_FIELD_PRINCIPLES[field]
    assert artifact["upstream_gate_receipt"]["prerequisite_values_exact"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED


def test_req_learn_5618_validation_fails_closed_on_required_gates(tmp_path: Path) -> None:
    """REQ-LEARN-5618: no safety or leakage failure can headline."""

    artifact = _artifact(tmp_path)
    assert mod.validate_artifact(artifact) is True

    bad_cases = [
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("continuous_self_learning_ready", False, "continuous_self_learning_ready"),
        ("no_model_weight_mutation", False, "no_model_weight_mutation"),
        ("lazy_identity_guard_passed", False, "lazy_identity_guard_passed"),
        ("unsafe_false_accept_count", {"total": 1}, "unsafe_false_accept_count"),
        ("models_tested", {"causal_controller": "wrong"}, "models_tested"),
        ("ale_by_arm", {}, "ale_by_arm"),
        ("delta_ale_vs_best_fixed", {"mean": 0.0}, "delta_ale_vs_best_fixed"),
        ("regret_to_oracle", {"mean": 0.0}, "regret_to_oracle"),
        ("forward_transfer_delta", {"mean": 0.0}, "forward_transfer_delta"),
        ("backward_retention_delta", {"mean": 0.0}, "backward_retention_delta"),
        ("forgetting_delta", {"mean": 0.1}, "forgetting_delta"),
        (
            "poison_update_disposition",
            {"accepted": 1, "rejected": 0, "rolled_back": 0},
            "poison_update_disposition",
        ),
        ("rollback_positive_control", {"passed": False}, "rollback_positive_control"),
        ("delayed_regression_passed", False, "delayed_regression_passed"),
        ("upstream_gate_receipt", {"prerequisite_values_exact": False}, "upstream_gate_receipt"),
        ("seeds", [5618, 5619, 5620, 5621], "seeds"),
        (
            "instances_per_condition",
            {"replicated_heldout_streams": 31},
            "instances_per_condition",
        ),
    ]
    for field, value, expected in bad_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["honest_verdict"] = mod.honest_verdict(bad)
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    leaked = deepcopy(artifact)
    leaked["controller_feature_contract"]["future_leakage_excluded"] = False
    leaked["honest_verdict"] = mod.honest_verdict(leaked)
    leaked["reproducibility_checksum"] = mod.reproducibility_checksum(leaked)
    with pytest.raises(ValueError, match="controller_feature_contract"):
        mod.validate_artifact(leaked)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("ale_by_arm")
    missing_principle["honest_verdict"] = mod.honest_verdict(missing_principle)
    missing_principle["reproducibility_checksum"] = mod.reproducibility_checksum(
        missing_principle
    )
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing_principle)

    missing_required = deepcopy(artifact)
    missing_required.pop("models_tested")
    missing_required["honest_verdict"] = mod.honest_verdict(missing_required)
    missing_required["reproducibility_checksum"] = mod.reproducibility_checksum(missing_required)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_required)

    stale_verdict = deepcopy(artifact)
    stale_verdict["honest_verdict"] = "complete: stale"
    stale_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(stale_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(stale_verdict)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)
