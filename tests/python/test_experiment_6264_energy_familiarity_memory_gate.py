"""Tests for Exp6264 energy-familiarity memory admission.

Spec refs: REQ-LEARN-6264, SCENARIO-LEARN-6264-THRESHOLDS,
SCENARIO-LEARN-6264-GATES, SCENARIO-LEARN-6264-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6264_energy_familiarity_memory_gate as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, object]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=write,
    )


def _refresh(artifact: dict[str, object]) -> dict[str, object]:
    artifact["familiarity_gate_ready_score"] = mod.ready_score(artifact)
    artifact["status"] = mod.status(artifact)
    artifact["honest_verdict"] = mod.honest_verdict(artifact)
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def _event(
    index: int,
    *,
    energy: float,
    unsafe: int = 0,
    family: str = "known_family",
    partition: str = "train",
    poisoned: bool = False,
) -> mod.EnergyEvent:
    return mod.EnergyEvent(
        row_id=f"row-{index}",
        event_id=f"event-{index}",
        model_hf_id="model-a",
        family=family,
        partition=partition,
        source_partition="calibration" if partition == "train" else "future_known",
        chronological_index=index,
        unsafe_label=unsafe,
        energy=energy,
        task_key=family,
        source_disposition="clean",
        content_addressed_row_id=f"sha256:{index:064x}",
        variant_kind="normal",
        control_kind="normal",
        poisoned=poisoned,
    )


def test_req_6264_spec_declares_artifact_contract_and_scenarios() -> None:
    """REQ-LEARN-6264: OpenSpec owns the 6264 artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-6264") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-6264-1",
        "REQ-LEARN-6264-8",
        "SCENARIO-LEARN-6264-THRESHOLDS",
        "SCENARIO-LEARN-6264-GATES",
        "SCENARIO-LEARN-6264-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        *mod.ARM_NAMES,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6264_artifact_writes_required_receipts(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6264-THRESHOLDS: thresholds freeze before held scoring."""

    artifact = _artifact(tmp_path, write=True)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["source_mutation_count"] == 0
    assert type(artifact["source_mutation_count"]) is int
    assert artifact["weight_mutation_count"] == 0
    assert type(artifact["weight_mutation_count"]) is int
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False

    bridge = artifact["upstream_bridge_path_and_hash"]
    assert bridge["sha256"] == mod.EXPECTED_BRIDGE_SHA256
    assert bridge["exact_hash_matched"] is True

    splits = artifact["chronological_split_hashes"]
    assert splits["partition_overlap_count"] == 0
    assert splits["row_count_by_partition"] == {"test": 160, "train": 192, "validation": 128}
    assert splits["quarantined_rows_entered_headline_count"] == 0

    preconditions = artifact["preconditions_checked"]
    assert preconditions["exact_bridge_hash_verified"] is True
    assert preconditions["partition_non_overlap_verified"] is True
    assert preconditions["sample_sizes_verified"] is True
    assert preconditions["no_llm_or_weight_mutation_verified"] is True

    energy = artifact["energy_definition_and_direction"]
    assert energy["direction"] == "lower_energy_is_more_familiar"
    assert energy["direction_validation"]["selected_direction"] == "lower_is_familiar"
    assert (
        energy["direction_validation"]["selected_utility_per_row"]
        > energy["direction_validation"]["reversed_direction_utility_per_row"]
    )

    fit = artifact["threshold_fit_partition_and_receipts"]
    assert fit["fit_partitions"] == ["train"]
    assert fit["held_partitions_used_for_threshold_count"] == 0
    assert fit["global_threshold"]["fit_row_count"] == 192
    assert fit["task_threshold_count"] == 4

    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert mod.validate_artifact(artifact) is True


def test_scenario_6264_held_arms_reduce_shifted_unsafe_advice(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6264-GATES: task thresholds withhold OOD advice."""

    artifact = _artifact(tmp_path)
    fires = artifact["treatment_fire_counts"]
    known = artifact["known_family_coverage_by_arm"]
    shifted = artifact["shifted_family_unsafe_advice_by_arm"]
    utility = artifact["exact_utility_by_arm"]
    transfer = artifact["negative_transfer_by_arm"]

    assert fires["no_memory"]["held"]["fire_count"] == 0
    assert fires["unconditional_advice"]["held"]["fire_count"] == 288
    assert 0 < fires["global_threshold"]["held"]["fire_count"] < 288
    assert 0 < fires["task_conditional_thresholds"]["held"]["fire_count"] < 288

    assert known["task_conditional_thresholds"]["coverage"] > 0.0
    assert known["task_conditional_thresholds"]["preregistered_known_family_regression"] is False
    assert (
        shifted["task_conditional_thresholds"]["unsafe_advice_count"]
        < shifted["unconditional_advice"]["unsafe_advice_count"]
    )
    assert transfer["task_conditional_thresholds"]["utility_delta_vs_no_memory"] >= 0.0
    assert (
        utility["task_conditional_thresholds"]["held"]["utility_per_row"]
        > utility["unconditional_advice"]["held"]["utility_per_row"]
    )

    inactive = artifact["inactive_gate_control"]
    assert inactive["thresholds_disabled"] is True
    assert inactive["matches_unconditional_advice"] is True

    ood = artifact["ood_positive_control"]
    assert ood["task_conditional_unseen_task_fire_count"] == 0
    assert ood["passed"] is True

    intervals = artifact["paired_intervals_and_sample_sizes"]
    assert intervals["task_conditional_vs_unconditional_shifted_unsafe_advice"]["n"] == 160
    assert intervals["task_conditional_vs_unconditional_known_coverage"]["n"] == 128
    assert intervals["task_conditional_vs_global_held_utility"]["n"] == 288


def test_scenario_6264_gate_edges_fail_closed() -> None:
    """SCENARIO-LEARN-6264-CONTROLS: edge cases cannot leak advice."""

    train = [
        _event(0, energy=-2.0, unsafe=0),
        _event(1, energy=-1.5, unsafe=0),
        _event(2, energy=2.0, unsafe=1),
        _event(3, energy=2.5, unsafe=1),
    ]
    fit = mod.fit_familiarity_thresholds(train)
    safe = _event(10, energy=-1.7, unsafe=0, partition="validation")
    unsafe = _event(11, energy=2.2, unsafe=1, partition="validation")
    unseen = _event(12, energy=-1.7, unsafe=0, family="unseen", partition="validation")
    poisoned = _event(13, energy=-1.7, unsafe=0, partition="validation", poisoned=True)

    assert fit["direction_control"]["selected_direction"] == "lower_is_familiar"
    assert (
        fit["direction_control"]["selected_utility_per_row"]
        > fit["direction_control"]["reversed_direction_utility_per_row"]
    )
    assert mod.advice_fires("global_threshold", safe, fit) is True
    assert mod.advice_fires("global_threshold", unsafe, fit) is False
    assert mod.advice_fires("task_conditional_thresholds", safe, fit) is True
    assert mod.advice_fires("task_conditional_thresholds", unseen, fit) is False
    assert mod.advice_fires("unconditional_advice", poisoned, fit) is False

    empty_fit = mod.fit_familiarity_thresholds([])
    assert empty_fit["global_threshold"]["threshold"] is None
    assert mod.advice_fires("global_threshold", safe, empty_fit) is False

    inactive = mod.FamiliarityGate(
        mode="global",
        direction=mod.LOWER_IS_FAMILIAR,
        global_threshold=None,
        task_thresholds={},
        inactive=True,
    )
    assert inactive.admit(safe) is True
    assert inactive.admit(poisoned) is False


def test_req_6264_deterministic_replay_and_validation_guards(tmp_path: Path) -> None:
    """REQ-LEARN-6264-7: validation rejects mutation and gate drift."""

    output_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    first = mod.run(
        result_path=output_path,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
    )
    second = mod.run(
        result_path=output_path,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
    )
    assert first == second
    assert mod.validate_artifact(first) is True

    missing = dict(first)
    missing.pop("off_policy_limitation")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_checksum = deepcopy(first)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_source = deepcopy(first)
    bad_source["source_mutation_count"] = {"value": 0}
    bad_source["reproducibility_checksum"] = mod.reproducibility_checksum(bad_source)
    with pytest.raises(ValueError, match="source_mutation_count"):
        mod.validate_artifact(bad_source)

    bad_weight = deepcopy(first)
    bad_weight["weight_mutation_count"] = 1
    _refresh(bad_weight)
    with pytest.raises(ValueError, match="weight_mutation_count"):
        mod.validate_artifact(bad_weight)

    bad_ready = deepcopy(first)
    bad_ready["treatment_fire_counts"]["task_conditional_thresholds"]["held"][
        "fire_count"
    ] = 0
    bad_ready["reproducibility_checksum"] = mod.reproducibility_checksum(bad_ready)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_ready)

    known_regression = deepcopy(first)
    known_regression["known_family_coverage_by_arm"]["task_conditional_thresholds"][
        "preregistered_known_family_regression"
    ] = True
    _refresh(known_regression)
    assert known_regression["status"] == "complete_null"
    assert mod.validate_artifact(known_regression) is True

    failed_test_receipt = deepcopy(first)
    failed_test_receipt["test_exit_codes"][mod.GLOBAL_PYTEST_COMMAND] = 2
    _refresh(failed_test_receipt)
    assert failed_test_receipt["status"] == "complete_null"
    assert "recorded test command failed" in failed_test_receipt["honest_verdict"]
    assert mod.validate_artifact(failed_test_receipt) is True


def test_req_6264_helper_branches_and_blocked_validation(tmp_path: Path) -> None:
    """REQ-LEARN-6264-1/5: helper guard branches are explicit."""

    assert mod.sha256_file(tmp_path / "missing.bin") is None
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(list_json)

    assert mod._std([1.0]) == 0.0
    assert mod._candidate_thresholds([]) == []
    assert mod._file_receipt(tmp_path / "missing.bin")["exists"] is False
    assert _event(99, energy=0.0).to_json()["row_id"] == "row-99"
    assert mod._resolve_path("results/example.json") == mod.REPO_ROOT / "results/example.json"
    with pytest.raises(ValueError, match="unknown energy direction"):
        mod._passes_threshold(0.0, 0.0, "sideways")

    fit = mod.fit_familiarity_thresholds([])
    event = _event(1, energy=0.0, partition="validation")
    assert mod._admission_probability("global_threshold", event, fit) == 0.0
    with pytest.raises(ValueError, match="unknown arm"):
        mod.advice_fires("mystery_arm", event, fit)
    with pytest.raises(ValueError, match="unknown arm"):
        mod._admission_probability("mystery_arm", event, fit)
    assert mod._ece([], []) == 0.0
    assert mod._paired_interval([], seed=1)["n"] == 0

    empty_events, receipt = mod._materialize_energy_events(
        {"rows": [{"source_disposition": "quarantine"}]}
    )
    assert empty_events == []
    assert receipt["fit_row_count"] == 0

    timed = mod.run(
        result_path=tmp_path / "timed.json",
        test_exit_codes=_passing_exit_codes(),
    )
    assert timed["duration_s"] >= 0.001

    blocked = _artifact(tmp_path)
    blocked["upstream_bridge_path_and_hash"]["exact_hash_matched"] = False
    _refresh(blocked)
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(blocked) is True

    bad_status = deepcopy(timed)
    bad_status["status"] = "blocked"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_verdict = deepcopy(timed)
    bad_verdict["honest_verdict"] = "complete: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_configs = deepcopy(timed)
    bad_configs["no_memory_unconditional_global_and_task_conditional_arm_configs"][
        "arm_names"
    ] = []
    bad_configs["reproducibility_checksum"] = mod.reproducibility_checksum(bad_configs)
    with pytest.raises(ValueError, match="arm config"):
        mod.validate_artifact(bad_configs)

    bad_principles = deepcopy(timed)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = mod.reproducibility_checksum(bad_principles)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    bad_provenance_type = deepcopy(timed)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_provenance = deepcopy(timed)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance)
