"""Tests for Exp5617 active-spline KAN critical task duration mapping.

Spec refs: REQ-KAN-5617, SCENARIO-KAN-5617.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5617_kan_critical_task_duration_map as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5617_kan_critical_task_duration_map.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5617_kan_critical_task_duration_map.py "
    "-m pytest tests/python/test_experiment_5617_kan_critical_task_duration_map.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5617_kan_critical_task_duration_map.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5617_kan_critical_task_duration_map.json"
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


def test_req_kan_5617_spec_declares_boundary_contract() -> None:
    """REQ-KAN-5617: OpenSpec anchors arms, fixture, metrics, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-KAN-5617") :]

    for marker in (
        "REQ-KAN-5617",
        "SCENARIO-KAN-5617",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "frozen_no_update",
        "retain_exact_replay",
        "reset_adapt",
        "loss_smoothed_adaptation",
        "update_substitution_control",
        "frozen_spline_control",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in mod.FIELD_PRINCIPLES[field]


def test_scenario_kan_5617_structured_gates_freeze_before_fixture_rows_load() -> None:
    """SCENARIO-KAN-5617: gates freeze splits, cells, metrics, and rules first."""

    gates = mod.freeze_structured_gates(root=REPO)

    assert gates["structured_gates_frozen"] is True
    assert gates["fixture_rows_loaded"] is False
    assert gates["duration_cells"] == list(mod.DURATION_CELLS)
    assert gates["split_names"] == list(mod.SPLIT_NAMES)
    assert gates["models_tested"] == list(mod.ARM_NAMES)
    assert gates["minimum_learner_seeds"] == 5
    assert gates["promotion_rules"]["selection_label_source"] == "calibration_only"
    assert gates["promotion_rules"]["future_heldout_labels_for_selection"] is False
    for metric in mod.DISAGGREGATED_METRICS:
        assert metric in gates["metrics_frozen"]

    fixture = mod.load_frozen_fixture(gates, root=REPO)
    assert fixture.fixture_hash == gates["fixture_hash"]
    assert fixture.stream_count == 1152
    assert fixture.heldout_replicates_per_condition == 40
    assert set(fixture.rows_by_split) == set(mod.SPLIT_NAMES)
    with pytest.raises(ValueError, match="structured_gates_frozen"):
        mod.load_frozen_fixture({"structured_gates_frozen": False}, root=REPO)


def test_scenario_kan_5617_duration_map_metrics_controls_and_switches(tmp_path: Path) -> None:
    """SCENARIO-KAN-5617: matched arms produce disaggregated boundary metrics."""

    result = mod.run_duration_map(
        mod.load_frozen_fixture(mod.freeze_structured_gates(root=REPO), root=REPO),
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert result["models_tested"] == list(mod.ARM_NAMES)
    assert result["seeds"] == list(mod.DEFAULT_LEARNER_SEEDS)
    assert result["instances_per_condition"]["replicated_heldout_streams"] >= 32
    assert result["optimization_budget"]["matched_across_mutable_arms"] is True
    assert result["optimization_budget"]["exact_validation_calls_matched"] is True
    assert result["unsafe_false_accept_count"]["total"] == 0
    assert result["lazy_identity_guard_passed"] is True
    assert result["control_credit_guard"]["update_substitution_control_credit"] == 0.0
    assert result["control_credit_guard"]["frozen_spline_control_credit"] == 0.0
    assert len(result["empirical_switch_durations"]) >= 2
    assert len(result["nondegenerate_switch_cases"]) >= 2
    assert 0.0 <= result["critical_duration_fit_r2"] <= 1.0
    assert result["critical_task_duration"] in mod.DURATION_CELLS

    cell_key = mod.cell_key("conflicting_rule", "persistent_drift", 16)
    for metric_name in (
        "ale_by_arm_and_cell",
        "instability_by_arm_and_cell",
        "transient_error_by_arm_and_cell",
        "time_to_valid_adaptation_by_arm_and_cell",
        "update_rollback_counts_by_arm_and_cell",
    ):
        metric = result[metric_name]
        assert set(mod.ARM_NAMES) <= set(metric)
        assert cell_key in metric[mod.RETAIN_REPLAY_ARM]

    short_cell = mod.cell_key("shared_rule", "no_drift", 1)
    long_cell = mod.cell_key("conflicting_rule", "persistent_drift", 32)
    assert result["transient_error_by_arm_and_cell"][mod.RETAIN_REPLAY_ARM][short_cell] <= result[
        "transient_error_by_arm_and_cell"
    ][mod.RESET_ARM][short_cell]
    assert result["ale_by_arm_and_cell"][mod.RESET_ARM][long_cell] <= result[
        "ale_by_arm_and_cell"
    ][mod.RETAIN_REPLAY_ARM][long_cell]

    ledger = result["immutable_update_ledger"]
    assert ledger
    assert all(row["ledger_hash"] == mod.ledger_hash(row) for row in ledger)
    assert any(row["arm"] == mod.RETAIN_REPLAY_ARM and row["decision"] == "accepted" for row in ledger)
    assert any(row["arm"] == mod.FROZEN_SPLINE_CONTROL_ARM for row in ledger)
    assert result["checkpoint_receipts"]

    tampered = dict(ledger[0])
    tampered["decision"] = "accepted" if ledger[0]["decision"] != "accepted" else "rejected"
    assert mod.ledger_hash(tampered) != ledger[0]["ledger_hash"]


def test_scenario_kan_5617_artifact_fields_stable_write_and_validation(tmp_path: Path) -> None:
    """SCENARIO-KAN-5617: terminal artifact exposes every required field."""

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
    assert artifact["fixture_hash"] == mod.freeze_structured_gates(root=REPO)["fixture_hash"]
    assert artifact["models_tested"] == list(mod.ARM_NAMES)
    assert artifact["duration_cells"] == list(mod.DURATION_CELLS)
    assert artifact["seeds"] == list(mod.DEFAULT_LEARNER_SEEDS)
    assert artifact["random_seeds"] == list(mod.DEFAULT_LEARNER_SEEDS)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED


def test_req_kan_5617_helper_edge_cases_are_deterministic() -> None:
    """REQ-KAN-5617: helper edge cases remain bounded and deterministic."""

    model = mod.initialized_model(mod.DEFAULT_LEARNER_SEEDS[0], mod.RETAIN_REPLAY_ARM)

    assert mod.exact_energy(model, (), label_name="label") == 0.0
    assert mod.exact_error(model, (), label_name="label") == 0.0
    assert mod.confidence_interval([0.25]) == {
        "mean": 0.25,
        "lower": 0.25,
        "upper": 0.25,
        "n": 1,
    }
    assert mod.switch_fit_r2([]) == 0.0
    assert mod.switch_fit_r2(
        [{"switch_duration": 4.0}, {"switch_duration": 4.0}]
    ) == 1.0
    assert mod.mean([]) == 0.0


def test_req_kan_5617_validation_fails_closed_on_required_gates(tmp_path: Path) -> None:
    """REQ-KAN-5617: artifact gates reject overclaim and safety regressions."""

    artifact = _artifact(tmp_path)
    assert mod.validate_artifact(artifact) is True

    bad_cases = [
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("lazy_identity_guard_passed", False, "lazy_identity_guard_passed"),
        ("unsafe_false_accept_count", {"total": 1}, "unsafe_false_accept_count"),
        ("models_tested", list(mod.ARM_NAMES[:-1]), "models_tested"),
        ("duration_cells", [1, 2], "duration_cells"),
        ("seeds", [1, 2, 3, 4], "seeds"),
        (
            "instances_per_condition",
            {"replicated_heldout_streams": 31},
            "instances_per_condition",
        ),
        ("ale_by_arm_and_cell", {}, "ale_by_arm_and_cell"),
        ("backward_retention_by_arm", {}, "backward_retention_by_arm"),
        ("forward_transfer_by_arm", {}, "forward_transfer_by_arm"),
        ("critical_duration_fit_r2", "bad", "critical_duration_fit_r2"),
    ]
    for field, value, expected in bad_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["honest_verdict"] = mod.honest_verdict(bad)
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    critical_without_switches = deepcopy(artifact)
    critical_without_switches["nondegenerate_switch_cases"] = []
    critical_without_switches["empirical_switch_durations"] = []
    critical_without_switches["critical_task_duration"] = 8
    critical_without_switches["honest_verdict"] = mod.honest_verdict(critical_without_switches)
    critical_without_switches["reproducibility_checksum"] = mod.reproducibility_checksum(
        critical_without_switches
    )
    with pytest.raises(ValueError, match="nondegenerate_switch_cases"):
        mod.validate_artifact(critical_without_switches)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("fixture_hash")
    missing_principle["honest_verdict"] = mod.honest_verdict(missing_principle)
    missing_principle["reproducibility_checksum"] = mod.reproducibility_checksum(missing_principle)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing_principle)

    missing_required = deepcopy(artifact)
    missing_required.pop("fixture_hash")
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
