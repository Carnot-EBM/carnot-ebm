"""Tests for Exp5608 KAN-only longitudinal exact-gated adaptation.

Spec refs: REQ-LEARN-5608,
SCENARIO-LEARN-5608-SESSIONS,
SCENARIO-LEARN-5608-ARMS,
SCENARIO-LEARN-5608-LEDGER,
SCENARIO-LEARN-5608-POISON,
SCENARIO-LEARN-5608-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5608_kan_longitudinal_self_learning as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5608_kan_longitudinal_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5608_kan_longitudinal_self_learning.py "
    "-m pytest tests/python/test_experiment_5608_kan_longitudinal_self_learning.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5608_kan_longitudinal_self_learning.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5608_kan_longitudinal_self_learning.json"
)
TESTS_ADDED_OR_REUSED = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
]


def _manifest() -> dict[str, object]:
    return exp.build_session_manifest(root=REPO)


def _artifact(tmp_path: Path) -> dict[str, object]:
    return exp.build_artifact(
        root=REPO,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        checkpoint_dir=tmp_path / "checkpoints",
    )


def test_req_learn_5608_spec_declares_longitudinal_contract() -> None:
    """REQ-LEARN-5608: OpenSpec anchors exact-gated longitudinal KAN work."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5608") : spec.index("## REQ-LEARN-5571")]

    for marker in (
        "REQ-LEARN-5608",
        "SCENARIO-LEARN-5608-SESSIONS",
        "SCENARIO-LEARN-5608-ARMS",
        "SCENARIO-LEARN-5608-LEDGER",
        "SCENARIO-LEARN-5608-POISON",
        "SCENARIO-LEARN-5608-ARTIFACT",
        str(exp.RESULT_RELATIVE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "reuse Exp5570's active-spline updater",
        "known-bad poisoned update",
    ):
        assert marker in section
    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_learn_5608_sessions_manifest_fixes_order_and_budgets() -> None:
    """SCENARIO-LEARN-5608-SESSIONS: sessions, holdouts, and budgets are fixed."""

    manifest = _manifest()
    sessions = manifest["ordered_sessions"]
    online_ids = {
        row_id
        for session in sessions
        for row_id in session["online_observation_ids"]
    }
    gate_ids = {
        row_id
        for session in sessions
        for row_id in session["gate_holdout_ids"]
    }
    independent_ids = {
        row_id
        for session in sessions
        for row_id in session["independent_heldout_ids"]
    }

    assert manifest["family_order"] == list(exp.ORDERED_FAMILIES)
    assert len(sessions) == 4
    assert manifest["exact_validator"]["source_artifact"] == str(exp.DATASET_RELATIVE_PATH)
    assert manifest["exact_validator"]["feedback_source"] == "accepted_by_exact_validator"
    assert manifest["seeds"] == list(exp.DEFAULT_SEEDS)
    assert manifest["sample_budget"]["online_observations_per_family"] == exp.ONLINE_ROWS_PER_FAMILY
    assert manifest["sample_budget"]["gate_holdout_per_family"] == exp.GATE_ROWS_PER_FAMILY
    assert manifest["sample_budget"]["independent_heldout_per_family"] == exp.HELDOUT_ROWS_PER_FAMILY
    assert manifest["adaptation_budget"]["update_budget_per_arm_seed"] == exp.UPDATE_BUDGET
    assert manifest["adaptation_budget"]["checkpoint_cadence"] == "before_each_proposed_update"
    assert manifest["delayed_replay_schedule"]
    assert online_ids.isdisjoint(gate_ids)
    assert online_ids.isdisjoint(independent_ids)
    assert gate_ids.isdisjoint(independent_ids)


def test_scenario_learn_5608_arms_metrics_and_safety_gates(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5608-ARMS: exact-gated KAN reports independent deltas."""

    result = exp.run_longitudinal_experiment(
        _manifest(),
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert result["arms"] == list(exp.ARM_NAMES)
    assert result["heldout_delta_by_arm"][exp.EXACT_GATED_ARM]["mean"] > 0.0
    assert result["heldout_delta_by_arm"][exp.EXACT_GATED_ARM]["ci"]["lower"] > 0.0
    assert result["heldout_delta_by_arm"][exp.FROZEN_ARM]["mean"] == 0.0
    assert result["forward_transfer_delta"] >= 0.0
    assert result["backward_retention_delta"] >= 0.0
    assert result["forgetting_delta"] <= 0.0
    assert result["cost_by_arm"][exp.EXACT_GATED_ARM]["proposed_updates"] == exp.UPDATE_BUDGET
    assert result["cost_by_arm"][exp.FROZEN_ARM]["proposed_updates"] == 0
    assert result["unsafe_false_accept_count"] == 0
    assert result["delayed_regression_passed"] is True
    assert result["no_model_weight_mutation"] is True
    assert result["kan_longitudinal_ready"] is True


def test_scenario_learn_5608_decision_ledger_is_immutable_and_attributable(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5608-LEDGER: every proposed update has a ledger row."""

    result = exp.run_longitudinal_experiment(
        _manifest(),
        checkpoint_dir=tmp_path / "checkpoints",
    )
    ledger = result["decision_ledger"]
    non_frozen_arms = {exp.SHUFFLED_ARM, exp.ALWAYS_UPDATE_ARM, exp.EXACT_GATED_ARM}
    rows_by_arm = {arm: [row for row in ledger if row["arm"] == arm] for arm in non_frozen_arms}

    assert len(ledger) == len(exp.DEFAULT_SEEDS) * len(non_frozen_arms) * exp.UPDATE_BUDGET + 1
    for arm, rows in rows_by_arm.items():
        expected = len(exp.DEFAULT_SEEDS) * exp.UPDATE_BUDGET
        if arm == exp.EXACT_GATED_ARM:
            expected += 1
        assert len(rows) == expected, arm
    for row in ledger:
        for field in exp.REQUIRED_LEDGER_FIELDS:
            assert field in row
        assert row["ledger_hash"] == exp.ledger_hash(row)
        assert row["checkpoint_hash"] == row["rollback_target"]["checkpoint_hash"]
        assert isinstance(row["active_spline_indices"], list)
        assert row["cost"]["touched_spline_count"] == len(row["active_spline_indices"])
        assert row["decision"] in {"accepted", "rejected", "rolled_back"}
        assert row["reason"]
        assert row["observations"]

    exact_rows = rows_by_arm[exp.EXACT_GATED_ARM]
    assert any(row["decision"] == "accepted" for row in exact_rows)
    assert any(row["decision"] in {"rejected", "rolled_back"} for row in exact_rows)

    tampered = dict(ledger[0])
    tampered["decision"] = "accepted" if ledger[0]["decision"] != "accepted" else "rejected"
    assert exp.ledger_hash(tampered) != ledger[0]["ledger_hash"]


def test_scenario_learn_5608_poison_update_rejected_or_rolled_back(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5608-POISON: bad adaptation cannot silently persist."""

    result = exp.run_longitudinal_experiment(
        _manifest(),
        checkpoint_dir=tmp_path / "checkpoints",
    )
    disposition = result["poison_update_disposition"]
    poison_rows = [row for row in result["decision_ledger"] if row["is_poison"]]

    assert disposition["injected"] is True
    assert disposition["disposition"] in {"rejected", "rolled_back"}
    assert disposition["persisted"] is False
    assert disposition["poison_ledger_ids"]
    assert result["rollback_positive_control"] is True
    assert result["rollback_positive_control_receipt"]["outputs_match"] is True
    assert result["rollback_positive_control_receipt"]["pre_update_hash"] == result[
        "rollback_positive_control_receipt"
    ]["restored_hash"]
    assert poison_rows
    assert len(poison_rows) == 1
    assert all(row["decision"] in {"rejected", "rolled_back"} for row in poison_rows)
    assert all("poison" in row["reason"] for row in poison_rows)


def test_scenario_learn_5608_artifact_fields_and_stable_write(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5608-ARTIFACT: receipt exposes required gate evidence."""

    destination = tmp_path / exp.RESULT_RELATIVE_PATH.name
    artifact = exp.run(
        root=REPO,
        result_path=destination,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        checkpoint_dir=tmp_path / "checkpoints",
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert exp.validate_artifact(artifact) is True
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field] == exp.REQUIRED_FIELD_PRINCIPLES[field]

    assert artifact["continuous_self_learning_task"] is True
    assert artifact["session_manifest"]["family_order"] == list(exp.ORDERED_FAMILIES)
    assert artifact["adaptation_budget"]["update_budget_per_arm_seed"] == exp.UPDATE_BUDGET
    assert artifact["decision_ledger"]
    assert artifact["heldout_delta_by_arm"][exp.EXACT_GATED_ARM]["ci"]["lower"] > 0.0
    assert artifact["forward_transfer_delta"] >= 0.0
    assert artifact["backward_retention_delta"] >= 0.0
    assert artifact["forgetting_delta"] <= 0.0
    assert artifact["unsafe_false_accept_count"] == 0
    assert artifact["poison_update_disposition"]["persisted"] is False
    assert artifact["rollback_positive_control"] is True
    assert artifact["delayed_regression_passed"] is True
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["kan_longitudinal_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED


def test_req_learn_5608_artifact_gate_fails_closed(tmp_path: Path) -> None:
    """REQ-LEARN-5608-6: promotion gate cannot pass contradictory evidence."""

    artifact = _artifact(tmp_path)
    assert exp.validate_artifact(artifact) is True

    bad_cases = [
        ("continuous_self_learning_task", False, "continuous_self_learning_task"),
        ("session_manifest", {}, "session_manifest"),
        ("adaptation_budget", {}, "adaptation_budget"),
        ("decision_ledger", [], "decision_ledger"),
        ("heldout_delta_by_arm", {}, "heldout_delta_by_arm"),
        ("unsafe_false_accept_count", 1, "unsafe_false_accept_count"),
        ("poison_update_disposition", {"persisted": True}, "poison_update_disposition"),
        ("rollback_positive_control", False, "rollback_positive_control"),
        ("delayed_regression_passed", False, "delayed_regression_passed"),
        ("no_model_weight_mutation", False, "no_model_weight_mutation"),
        ("inference_substrate", "llm", "inference_substrate"),
    ]
    for field, value, expected in bad_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["kan_longitudinal_ready"] = exp.kan_longitudinal_ready_from_artifact(bad)
        bad["honest_verdict"] = exp.honest_verdict(bad)
        bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    bad_heldout_gate = deepcopy(artifact)
    bad_heldout_gate["heldout_delta_by_arm"][exp.EXACT_GATED_ARM]["ci"]["lower"] = 0.0
    bad_heldout_gate["kan_longitudinal_ready"] = True
    bad_heldout_gate["honest_verdict"] = "complete: invalid"
    bad_heldout_gate["reproducibility_checksum"] = exp.reproducibility_checksum(
        bad_heldout_gate
    )
    with pytest.raises(ValueError, match="kan_longitudinal_ready"):
        exp.validate_artifact(bad_heldout_gate)

    blocked = deepcopy(artifact)
    blocked["heldout_delta_by_arm"][exp.EXACT_GATED_ARM]["ci"]["lower"] = 0.0
    blocked["kan_longitudinal_ready"] = False
    blocked["honest_verdict"] = exp.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    assert exp.validate_artifact(blocked) is True
    assert blocked["honest_verdict"].startswith("bounded_null:")

    invalid_claim = deepcopy(blocked)
    invalid_claim["kan_longitudinal_ready"] = True
    invalid_claim["honest_verdict"] = "complete: invalid"
    invalid_claim["reproducibility_checksum"] = exp.reproducibility_checksum(invalid_claim)
    with pytest.raises(ValueError, match="kan_longitudinal_ready"):
        exp.validate_artifact(invalid_claim)

    missing = deepcopy(artifact)
    missing.pop("honest_verdict")
    missing["reproducibility_checksum"] = exp.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = exp.reproducibility_checksum(bad_principles)
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principles)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    bad_ledger_hash = deepcopy(artifact)
    bad_ledger_hash["decision_ledger"][0]["decision"] = "tampered"
    bad_ledger_hash["reproducibility_checksum"] = exp.reproducibility_checksum(bad_ledger_hash)
    with pytest.raises(ValueError, match="decision_ledger"):
        exp.validate_artifact(bad_ledger_hash)

    model = exp.exp5570.OnlineKANEnergyModel(seed=5608, n_params=exp.exp5570.FEATURE_DIM)
    assert exp.exact_energy(model, []) == 0.0
    assert exp.exact_error(model, []) == 0.0
    assert exp.confidence_interval([0.25]) == {
        "mean": 0.25,
        "lower": 0.25,
        "upper": 0.25,
        "n": 1,
    }
    assert exp.rollback_positive_control_receipt([{"decision_ledger": []}]) == {
        "passed": False,
        "outputs_match": False,
        "pre_update_hash": "",
        "restored_hash": "",
    }
    assert (
        exp.accept_reason(
            exp.EXACT_GATED_ARM,
            False,
            train_pre=0.1,
            train_post=0.2,
            heldout_pre=0.1,
            heldout_post=0.1,
            unsafe_pre=0,
            unsafe_post=0,
        )
        == "rejected_exact_train_energy_regression"
    )
    assert (
        exp.accept_reason(
            exp.EXACT_GATED_ARM,
            False,
            train_pre=0.1,
            train_post=0.1,
            heldout_pre=0.1,
            heldout_post=0.1,
            unsafe_pre=0,
            unsafe_post=0,
        )
        == "rejected_exact_gate_no_improvement"
    )
