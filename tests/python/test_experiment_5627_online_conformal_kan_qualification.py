"""Tests for Exp5627 online-conformal KAN qualification.

Spec refs: REQ-LEARN-5627,
SCENARIO-LEARN-5627-CHRONOLOGY,
SCENARIO-LEARN-5627-GROUPS,
SCENARIO-LEARN-5627-CONTROLS,
SCENARIO-LEARN-5627-SAFETY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5627_online_conformal_kan_qualification as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5627_online_conformal_kan_qualification.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5627_online_conformal_kan_qualification.py "
    "-m pytest tests/python/test_experiment_5627_online_conformal_kan_qualification.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5627_online_conformal_kan_qualification.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5627_online_conformal_kan_qualification.json"
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
        scratch_dir=tmp_path,
    )


def test_req_learn_5627_spec_declares_online_conformal_contract() -> None:
    """REQ-LEARN-5627: OpenSpec anchors fields, arms, groups, gates, and substrate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5627") :]

    for marker in (
        "REQ-LEARN-5627",
        "SCENARIO-LEARN-5627-CHRONOLOGY",
        "SCENARIO-LEARN-5627-GROUPS",
        "SCENARIO-LEARN-5627-CONTROLS",
        "SCENARIO-LEARN-5627-SAFETY",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "global",
        "rolling-window",
        "group-conditional",
        "inactive",
        "shuffled-label",
        "undercoverage",
        "delayed-label",
        "order-permutation",
    ):
        assert marker in section
    for action in mod.ACTIONS:
        assert f"`{action}`" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle == mod.FIELD_PRINCIPLES[field]


def test_scenario_learn_5627_chronology_receipts_freeze_before_coverage() -> None:
    """SCENARIO-LEARN-5627-CHRONOLOGY: frozen split hashes exclude future leakage."""

    rows = mod.load_fixture_rows(REPO)
    receipts = mod.freeze_chronological_splits(rows, root=REPO)

    assert receipts["windows_frozen_before_conformal_scoring"] is True
    assert receipts["future_rows_in_initial_calibration"] == 0
    assert receipts["stream_id_overlap_count"] == 0
    assert receipts["state_id_overlap_count"] == 0
    assert receipts["update_id_overlap_count"] == 0
    assert receipts["train_window"]["max_instance_index"] < receipts["calibration_window"]["min_instance_index"]
    assert receipts["calibration_window"]["max_instance_index"] < receipts["heldout_window"]["min_instance_index"]
    assert set(receipts["split_hashes"]) == {"calibration", "heldout", "train"}
    assert all(value.startswith("sha256:") for value in receipts["split_hashes"].values())

    groups = mod.preregister_groups(rows)
    assert groups["group_axes"] == list(mod.GROUP_AXES)
    assert groups["sparse_group_backoff"]["levels"][0] == mod.EXACT_GROUP_LEVEL
    assert groups["adequately_powered_threshold"] == mod.ADEQUATELY_POWERED_DENOMINATOR

    sparse = mod.select_backoff_history(
        "missing|missing|missing|d999",
        {"global": [0.1, 0.2, 0.3]},
        min_count=mod.ADEQUATELY_POWERED_DENOMINATOR,
    )
    assert sparse["level"] == "global"
    assert sparse["history"] == [0.1, 0.2, 0.3]


def test_scenario_learn_5627_group_conditional_metrics_and_controls(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5627-GROUPS: headline coverage, group denominators, and controls report."""

    artifact = _artifact(tmp_path)
    headline = mod.GROUP_CONDITIONAL_ARM

    assert mod.validate_artifact(artifact) is True
    assert artifact["method_arms"]["headline"] == headline
    assert set(mod.CONFORMAL_ARMS).issubset(artifact["method_arms"]["conformal"])
    assert set(mod.CONTROL_ARMS).issubset(artifact["method_arms"]["controls"])
    assert artifact["marginal_coverage"][headline]["heldout"]["coverage"] >= 0.90
    assert artifact["worst_group_coverage"][headline]["coverage"] >= 0.90
    assert artifact["worst_group_coverage"][headline]["adequately_powered_groups_only"] is True
    assert artifact["coverage_intervals"][headline]["heldout"]["n"] > 0
    assert artifact["training_conditional_regret"][headline]["mean"] >= 0.0
    assert artifact["detection_delay"][headline]["mean"] >= 0.0
    assert artifact["useful_singleton_or_correct_set_rate"][headline] >= mod.USEFUL_RATE_FLOOR
    assert artifact["conformal_qualification_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete:")

    powered_group = artifact["worst_group_coverage"][headline]["group"]
    assert artifact["group_definitions"]["denominators"][powered_group]["heldout"] >= (
        mod.ADEQUATELY_POWERED_DENOMINATOR
    )
    assert powered_group in artifact["action_set_size_by_group"][headline]
    assert powered_group in artifact["abstention_rate_by_group"][headline]
    assert artifact["leakage_controls"]["order_permutation_control_nonpromotable"] is True
    assert artifact["leakage_controls"]["undercoverage_control_nonpromotable"] is True


def test_scenario_learn_5627_safety_fail_closed_and_control_rows_abstain(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5627-SAFETY: exact invalid controls cannot be legalized."""

    artifact = _artifact(tmp_path)
    safety = artifact["exact_validator_authority"]

    assert artifact["exact_unsafe_accept_count"]["total"] == 0
    assert safety["invalid_control_rows_seen"] > 0
    assert safety["invalid_control_rows_restricted_to_abstain"] == safety["invalid_control_rows_seen"]
    assert safety["conformal_can_legalize_invalid_action"] is False

    invalid = next(row for row in mod.load_fixture_rows(REPO) if row["control_kind"] == "wrong_predicate")
    assert mod.oracle_action(invalid) == "abstain"
    assert mod.safe_action_set(invalid, ["retain", "reset", "abstain"]) == ["abstain"]

    valid = next(row for row in mod.load_fixture_rows(REPO) if row["row_role"] == "stream_update")
    assert mod.oracle_action(valid) in set(mod.ACTIONS) - {"abstain"}
    assert mod.safe_action_set(valid, ["retain", "abstain"]) == ["retain", "abstain"]


def test_req_learn_5627_artifact_write_and_validation_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-5627: artifact writes stably and validation rejects overclaim gates."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(
        root=REPO,
        result_path=destination,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        scratch_dir=tmp_path,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert mod.validate_artifact(artifact) is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field] == mod.REQUIRED_FIELD_PRINCIPLES[field]

    bad_cases = [
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("leakage_control_pass", False, "leakage_control_pass"),
        ("exact_unsafe_accept_count", {"total": 1}, "exact_unsafe_accept_count"),
        ("conformal_qualification_ready_score", 0.5, "conformal_qualification_ready_score"),
    ]
    for field, value, expected in bad_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["honest_verdict"] = mod.honest_verdict(bad)
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    low_marginal = deepcopy(artifact)
    low_marginal["marginal_coverage"][mod.GROUP_CONDITIONAL_ARM]["heldout"]["coverage"] = 0.89
    low_marginal["honest_verdict"] = mod.honest_verdict(low_marginal)
    low_marginal["reproducibility_checksum"] = mod.reproducibility_checksum(low_marginal)
    with pytest.raises(ValueError, match="marginal_coverage"):
        mod.validate_artifact(low_marginal)

    low_group = deepcopy(artifact)
    low_group["worst_group_coverage"][mod.GROUP_CONDITIONAL_ARM]["coverage"] = 0.89
    low_group["honest_verdict"] = mod.honest_verdict(low_group)
    low_group["reproducibility_checksum"] = mod.reproducibility_checksum(low_group)
    with pytest.raises(ValueError, match="worst_group_coverage"):
        mod.validate_artifact(low_group)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("fixture_path")
    missing_principle["honest_verdict"] = mod.honest_verdict(missing_principle)
    missing_principle["reproducibility_checksum"] = mod.reproducibility_checksum(
        missing_principle
    )
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing_principle)

    missing_required = deepcopy(artifact)
    missing_required.pop("fixture_path")
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


def test_req_learn_5627_helper_edge_cases_are_bounded() -> None:
    """REQ-LEARN-5627: helper edge cases stay deterministic and conservative."""

    assert mod.conformal_quantile([], alpha=0.1) == 1.0
    assert mod.conformal_quantile([0.2], alpha=0.1) == 0.2
    assert mod.wilson_interval(0, 0) == {"coverage": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    assert mod.interval_from_values([0.5]) == {"mean": 0.5, "lower": 0.5, "upper": 0.5, "n": 1}

    synthetic = {
        "accepted_by_exact_validator": True,
        "space_shift_family": "shared_rule",
        "temporal_drift_type": "persistent_drift",
        "duration": 4,
        "control_kind": "none",
        "row_role": "stream_update",
        "seed": 1,
        "step_index": 0,
    }
    assert mod.group_key(synthetic) == "shared_rule|persistent_drift|shared|d4"
    assert mod.oracle_action(synthetic) == "adapt"
    scores = mod.action_nonconformity_scores(synthetic)
    assert scores["adapt"] < scores["reset"]
