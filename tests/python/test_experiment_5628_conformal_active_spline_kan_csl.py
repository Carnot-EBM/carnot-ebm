"""Tests for Exp5628 conformal active-spline KAN CSL replication.

Spec refs: REQ-LEARN-5628,
SCENARIO-LEARN-5628-WINDOWS,
SCENARIO-LEARN-5628-ARMS,
SCENARIO-LEARN-5628-SAFETY,
SCENARIO-LEARN-5628-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5628_conformal_active_spline_kan_csl as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5628_conformal_active_spline_kan_csl.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5628_conformal_active_spline_kan_csl.py "
    "-m pytest tests/python/test_experiment_5628_conformal_active_spline_kan_csl.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5628_conformal_active_spline_kan_csl.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5628_conformal_active_spline_kan_csl.json"
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


def test_req_learn_5628_spec_declares_conformal_kan_replication_contract() -> None:
    """REQ-LEARN-5628: OpenSpec anchors fields, arms, controls, and substrate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5628") :]

    for marker in (
        "REQ-LEARN-5628",
        "SCENARIO-LEARN-5628-WINDOWS",
        "SCENARIO-LEARN-5628-ARMS",
        "SCENARIO-LEARN-5628-SAFETY",
        "SCENARIO-LEARN-5628-ARTIFACT",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "frozen",
        "retain/replay",
        "reset/adapt",
        "best fixed non-oracle",
        "conformal controller without KAN",
        "full conformal-KAN controller",
        "inactive KAN",
        "oracle-reference",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle == mod.FIELD_PRINCIPLES[field]


def test_scenario_learn_5628_windows_are_frozen_independent_and_hashed() -> None:
    """SCENARIO-LEARN-5628-WINDOWS: new windows and seeds are frozen before outcomes."""

    receipts = mod.freeze_evaluation_windows(root=REPO, seeds=mod.DEFAULT_REPLICATION_SEEDS)

    assert receipts["windows_frozen_before_outcomes"] is True
    assert receipts["heldout_rows_used_for_tuning"] is False
    assert receipts["learner_seed_overlap_with_exp5618"] == 0
    assert receipts["evaluation_overlap_with_exp5627_initial_calibration"] == 0
    assert receipts["replication_seed_count"] >= 5
    assert set(receipts["windows"]) == {
        "chronological_train",
        "chronological_calibration",
        "early_heldout",
        "late_heldout",
    }
    assert (
        receipts["windows"]["chronological_train"]["max_instance_index"]
        < receipts["windows"]["chronological_calibration"]["min_instance_index"]
    )
    assert (
        receipts["windows"]["chronological_calibration"]["max_instance_index"]
        < receipts["windows"]["early_heldout"]["min_instance_index"]
    )
    assert (
        receipts["windows"]["early_heldout"]["max_instance_index"]
        < receipts["windows"]["late_heldout"]["min_instance_index"]
    )
    assert all(
        window["row_sha256"].startswith("sha256:") for window in receipts["windows"].values()
    )
    assert all(window["row_count"] > 0 for window in receipts["windows"].values())

    bad = mod.freeze_evaluation_windows(root=REPO, seeds=mod.exp5618.DEFAULT_LEARNER_SEEDS)
    assert bad["learner_seed_overlap_with_exp5618"] == len(mod.exp5618.DEFAULT_LEARNER_SEEDS)


def test_scenario_learn_5628_arms_compare_full_conformal_kan_to_every_fixed_arm(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5628-ARMS: full conformal-KAN beats every fixed non-oracle arm."""

    artifact = _artifact(tmp_path)

    assert mod.validate_artifact(artifact) is True
    assert artifact["method_arms"]["full_conformal_kan_controller"] == mod.FULL_CONFORMAL_KAN_ARM
    assert artifact["method_arms"]["conformal_controller_without_kan"] == mod.CONFORMAL_NO_KAN_ARM
    assert artifact["method_arms"]["inactive_kan"] == mod.INACTIVE_KAN_ARM
    assert artifact["method_arms"]["oracle_reference"] == mod.ORACLE_REFERENCE_ARM
    assert set(mod.FIXED_NONORACLE_ARMS).issubset(artifact["method_arms"]["fixed_nonoracle"])
    assert artifact["method_arms"]["budgets"]["equal_label_budget"] is True
    assert artifact["method_arms"]["budgets"]["equal_compute_budget"] is True
    assert artifact["method_arms"]["oracle_reference_nonpromotable"] is True

    full_ale = artifact["ale_by_arm"][mod.FULL_CONFORMAL_KAN_ARM]["mean"]
    for arm in mod.FIXED_NONORACLE_ARMS:
        assert artifact["ale_by_arm"][arm]["mean"] > full_ale
        assert artifact["delta_vs_each_fixed_nonoracle_arm"][arm]["mean"] > 0.0
        assert artifact["delta_vs_each_fixed_nonoracle_arm"][arm]["lower"] > 0.0
        assert artifact["ale_paired_intervals"][arm]["n"] >= len(mod.exp5616.condition_keys())

    assert artifact["ale_by_arm"][mod.CONFORMAL_NO_KAN_ARM]["mean"] > full_ale
    assert artifact["ale_by_arm"][mod.INACTIVE_KAN_ARM]["mean"] > full_ale
    assert artifact["ale_by_arm"][mod.ORACLE_REFERENCE_ARM]["mean"] <= full_ale
    assert artifact["conditional_regret_by_group"]["max_regret"] <= mod.CONDITIONAL_REGRET_BOUND
    assert (
        artifact["forward_transfer"][mod.FULL_CONFORMAL_KAN_ARM]["mean"]
        > artifact["forward_transfer"][mod.INACTIVE_KAN_ARM]["mean"]
    )
    assert (
        artifact["backward_retention"][mod.FULL_CONFORMAL_KAN_ARM]["mean"]
        >= artifact["backward_retention"][mod.RESET_ARM]["mean"] - mod.OLD_RULE_REGRESSION_TOLERANCE
    )


def test_scenario_learn_5628_safety_controls_action_sets_and_replay(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5628-SAFETY: controls fail closed and checkpoints replay."""

    artifact = _artifact(tmp_path)

    assert (
        artifact["conformal_action_set_utility"]["headline_arm"]
        == mod.exp5627.GROUP_CONDITIONAL_ARM
    )
    assert artifact["conformal_action_set_utility"]["nontrivial_action_sets"] is True
    assert artifact["conformal_action_set_utility"]["useful_singleton_or_correct_set_rate"] >= (
        mod.exp5627.USEFUL_RATE_FLOOR
    )
    assert 0.0 < artifact["abstention_rate"]["forced_abstention_rate"] < 1.0
    assert artifact["unsafe_false_accept_count"]["total"] == 0
    assert artifact["poison_rejection_rate"]["rate"] == 1.0
    assert artifact["poison_rejection_rate"]["accepted"] == 0
    assert artifact["delayed_regression_recovery"]["passed"] is True
    assert artifact["checkpoint_replay_exact"]["passed"] is True
    assert (
        mod.verify_checkpoint_replay_receipts(artifact["checkpoint_replay_exact"]["receipts"])
        is True
    )
    assert artifact["llm_weight_updates"] == 0
    assert artifact["continuous_self_learning_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")

    controls = artifact["control_injections"]
    for control in (
        "wrong_predicate",
        "wrong_binding",
        "delayed_label",
        "poison_update",
        "group_undercoverage",
        "abrupt_conflict",
    ):
        assert controls[control]["present"] is True

    audit = artifact["candidate_update_audit"]
    assert audit["candidate_update_count"] > 0
    assert audit["exact_acceptance_recorded_count"] == audit["candidate_update_count"]
    assert audit["bounded_rollback_recorded_count"] == audit["candidate_update_count"]
    assert audit["audit_trail_hash"].startswith("sha256:")
    assert all(row["audit_hash"] == mod.audit_row_hash(row) for row in audit["sample_audit_rows"])


def test_req_learn_5628_artifact_write_and_validation_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-5628: artifact writes stably and validation rejects overclaims."""

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
    assert artifact["upstream_gate_receipts"]["prerequisite_evidence_exact"] is True
    assert artifact["evaluation_window_receipts"]["replication_data_independent"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED

    bad_cases = [
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("llm_weight_updates", 1, "llm_weight_updates"),
        ("unsafe_false_accept_count", {"total": 1}, "unsafe_false_accept_count"),
        ("poison_rejection_rate", {"rate": 0.5, "accepted": 1}, "poison_rejection_rate"),
        ("checkpoint_replay_exact", {"passed": False}, "checkpoint_replay_exact"),
        ("delayed_regression_recovery", {"passed": False}, "delayed_regression_recovery"),
    ]
    for field, value, expected in bad_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["continuous_self_learning_ready"] = mod.continuous_self_learning_ready(bad)
        bad["honest_verdict"] = mod.honest_verdict(bad)
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    stale_ready = deepcopy(artifact)
    stale_ready["continuous_self_learning_ready"] = False
    stale_ready["honest_verdict"] = mod.honest_verdict(stale_ready)
    stale_ready["reproducibility_checksum"] = mod.reproducibility_checksum(stale_ready)
    with pytest.raises(ValueError, match="continuous_self_learning_ready"):
        mod.validate_artifact(stale_ready)

    low_delta = deepcopy(artifact)
    first_fixed = mod.FIXED_NONORACLE_ARMS[0]
    low_delta["delta_vs_each_fixed_nonoracle_arm"][first_fixed]["lower"] = 0.0
    low_delta["continuous_self_learning_ready"] = mod.continuous_self_learning_ready(low_delta)
    low_delta["honest_verdict"] = mod.honest_verdict(low_delta)
    low_delta["reproducibility_checksum"] = mod.reproducibility_checksum(low_delta)
    with pytest.raises(ValueError, match="delta_vs_each_fixed_nonoracle_arm"):
        mod.validate_artifact(low_delta)

    high_regret = deepcopy(artifact)
    high_regret["conditional_regret_by_group"]["max_regret"] = mod.CONDITIONAL_REGRET_BOUND + 0.01
    high_regret["continuous_self_learning_ready"] = mod.continuous_self_learning_ready(high_regret)
    high_regret["honest_verdict"] = mod.honest_verdict(high_regret)
    high_regret["reproducibility_checksum"] = mod.reproducibility_checksum(high_regret)
    with pytest.raises(ValueError, match="conditional_regret_by_group"):
        mod.validate_artifact(high_regret)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("ale_by_arm")
    missing_principle["continuous_self_learning_ready"] = mod.continuous_self_learning_ready(
        missing_principle
    )
    missing_principle["honest_verdict"] = mod.honest_verdict(missing_principle)
    missing_principle["reproducibility_checksum"] = mod.reproducibility_checksum(missing_principle)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing_principle)

    missing_required = deepcopy(artifact)
    missing_required.pop("ale_by_arm")
    missing_required["continuous_self_learning_ready"] = mod.continuous_self_learning_ready(
        missing_required
    )
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
