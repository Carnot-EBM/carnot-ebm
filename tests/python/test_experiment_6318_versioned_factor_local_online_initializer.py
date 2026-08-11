"""Tests for Exp6318 versioned factor-local online initializer.

Spec refs: REQ-CSL-6318, REQ-CSL-6318-STREAM,
REQ-CSL-6318-PREDECISION, REQ-CSL-6318-VERSIONS,
REQ-CSL-6318-BUDGETS, REQ-CSL-6318-RELEASE,
REQ-CSL-6318-CONTROLS, REQ-CSL-6318-READY,
REQ-CSL-6318-PROVENANCE, SCENARIO-CSL-6318-CHRONOLOGY,
SCENARIO-CSL-6318-LINEAGE, SCENARIO-CSL-6318-BUDGET-PARITY,
SCENARIO-CSL-6318-BOUNDARY, SCENARIO-CSL-6318-ROLLBACK,
SCENARIO-CSL-6318-NO-TRANSFER.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6318_versioned_factor_local_online_initializer as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, object]:
    return mod.run(
        date="20260811",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=write,
    )


def _path(receipt: dict[str, object]) -> Path:
    return Path(str(receipt["path"]))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _refresh(artifact: dict[str, object]) -> dict[str, object]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_csl_6318_spec_declares_artifact_contract() -> None:
    """REQ-CSL-6318-PROVENANCE: OpenSpec owns fields and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CSL-6318") :]

    for token in (
        "REQ-CSL-6318-STREAM",
        "REQ-CSL-6318-PREDECISION",
        "REQ-CSL-6318-VERSIONS",
        "REQ-CSL-6318-BUDGETS",
        "REQ-CSL-6318-RELEASE",
        "REQ-CSL-6318-CONTROLS",
        "REQ-CSL-6318-READY",
        "SCENARIO-CSL-6318-CHRONOLOGY",
        "SCENARIO-CSL-6318-LINEAGE",
        "SCENARIO-CSL-6318-BUDGET-PARITY",
        "SCENARIO-CSL-6318-BOUNDARY",
        "SCENARIO-CSL-6318-ROLLBACK",
        "SCENARIO-CSL-6318-NO-TRANSFER",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        *mod.ARM_NAMES,
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_csl_6318_chronology_snapshots_precede_outcomes(tmp_path: Path) -> None:
    """SCENARIO-CSL-6318-CHRONOLOGY: labels open only after snapshots."""

    artifact = _artifact(tmp_path)
    manifest = json.loads(
        _path(artifact["sealed_stream_manifest_path_and_hash"]).read_text(encoding="utf-8")
    )
    snapshots = _jsonl(_path(artifact["immutable_predecision_snapshots"]))
    outcomes = _jsonl(_path(artifact["postdecision_exact_outcome_receipts"]))

    assert len(snapshots) == manifest["event_count"] * len(mod.ARM_NAMES)
    assert len(outcomes) == manifest["event_count"]
    assert artifact["chronological_partition_contract"]["label_visibility"] == "postdecision_only"
    assert artifact["immutable_predecision_snapshots"]["chronology_leak_count"] == 0
    assert all(row["phase"] == "predecision" for row in snapshots)
    assert all("target_state" not in row and row["label_visible"] is False for row in snapshots)
    assert all(row["active_version"] for row in snapshots)
    latest_snapshot_by_event = {
        row["event_id"]: max(
            snap["snapshot_sequence"] for snap in snapshots if snap["event_id"] == row["event_id"]
        )
        for row in outcomes
    }
    assert all(
        row["reveal_sequence"] > latest_snapshot_by_event[row["event_id"]] for row in outcomes
    )


def test_scenario_csl_6318_version_lineage_and_factor_attribution(
    tmp_path: Path,
) -> None:
    """SCENARIO-CSL-6318-LINEAGE: versions have parents and factors."""

    artifact = _artifact(tmp_path)
    registry = _jsonl(_path(artifact["version_registry_path_and_hash"]))
    receipts = artifact["version_parent_and_changed_factor_receipts"]
    seen: set[str] = set()
    full_changed = 0
    factor_changed = 0

    for row in registry:
        version_id = str(row["version_id"])
        assert version_id not in seen
        if row["parent_version_id"] is not None:
            assert row["parent_version_id"] in seen
            assert row["changed_factor_set"]
            assert set(row["changed_factor_set"]) <= set(mod.FACTOR_NAMES)
            assert row["movement_cost"]["changed_factor_count"] == len(row["changed_factor_set"])
            assert row["created_after_outcome_sequence"] >= 1
            if row["arm"] == mod.FULL_STATE_ARM:
                full_changed += len(row["changed_factor_set"])
            if row["arm"] == mod.FACTOR_LOCAL_ARM:
                factor_changed += len(row["changed_factor_set"])
        seen.add(version_id)

    assert receipts["all_non_root_versions_have_existing_parent"] is True
    assert receipts["all_candidates_have_changed_factor_set"] is True
    assert receipts["root_version_count"] == len(mod.LEARNING_ARMS)
    assert factor_changed < full_changed


def test_scenario_csl_6318_budget_parity_cost_and_readiness(tmp_path: Path) -> None:
    """SCENARIO-CSL-6318-BUDGET-PARITY: learning arms stay comparable."""

    artifact = _artifact(tmp_path)
    budgets = artifact["matched_update_and_verifier_budgets"]
    metrics = artifact[
        "first_attempt_exact_rate_refinement_work_regret_retention_forgetting_and_negative_transfer_by_arm_and_partition"
    ]
    intervals = artifact["paired_intervals_and_sample_sizes"]
    costs = artifact["movement_memory_and_update_cost_by_arm"]

    assert budgets[mod.FULL_STATE_ARM]["authenticated_update_opportunities"] == budgets[
        mod.FACTOR_LOCAL_ARM
    ]["authenticated_update_opportunities"]
    assert budgets[mod.FULL_STATE_ARM]["exact_verifier_call_count"] == budgets[
        mod.FACTOR_LOCAL_ARM
    ]["exact_verifier_call_count"]
    assert budgets[mod.FULL_STATE_ARM]["validation_window_size"] == budgets[
        mod.FACTOR_LOCAL_ARM
    ]["validation_window_size"]
    assert metrics["forward_transfer_by_arm"][mod.FACTOR_LOCAL_ARM][
        "future_same_template_delta_vs_frozen"
    ] > 0.0
    assert metrics["forward_transfer_by_arm"][mod.FACTOR_LOCAL_ARM][
        "held_template_delta_vs_frozen"
    ] > 0.0
    assert metrics["forward_transfer_by_arm"][mod.FACTOR_LOCAL_ARM][
        "unseen_family_delta_vs_frozen"
    ] > 0.0
    assert intervals["factor_local_vs_full_state_utility"]["mean_delta"] >= (
        -mod.NONINFERIORITY_MARGIN
    )
    assert costs[mod.FACTOR_LOCAL_ARM]["total_movement_cost"] < costs[mod.FULL_STATE_ARM][
        "total_movement_cost"
    ]
    assert artifact["versioned_factor_local_learning_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")


def test_scenario_csl_6318_boundary_release_rollback_and_restart(
    tmp_path: Path,
) -> None:
    """SCENARIO-CSL-6318-ROLLBACK: degradation restores parent bytes."""

    artifact = _artifact(tmp_path)
    pairings = artifact["champion_challenger_pairing_and_decisions"]
    releases = artifact["task_boundary_release_receipts"]
    monitoring = artifact["monitoring_degradation_and_parent_rollback_receipts"]
    stress = artifact["reversal_poison_restart_and_rollback_results"]

    assert pairings["paired_decision_count"] >= 1
    assert all(row["paired"] is True for row in pairings["decisions"])
    assert releases["activation_count"] >= 1
    assert all(row["activated_at_task_boundary"] is True for row in releases["releases"])
    assert all(row["release_index"] > row["created_at_index"] for row in releases["releases"])
    assert all(row["release_index"] > row["validation_window_end"] for row in releases["releases"])
    assert monitoring["rollback_count"] >= 2
    assert {"planted_reversal", "natural_retention_dip"} <= set(
        monitoring["degradation_classes"]
    )
    assert all(row["byte_exact_parent_restore"] is True for row in monitoring["rollbacks"])
    assert monitoring["restart_matches_active_versions"] is True
    assert stress[mod.FACTOR_LOCAL_ARM]["rollback_count"] >= 2
    assert stress[mod.FACTOR_LOCAL_ARM]["poison_quarantine_count"] >= 1
    assert stress[mod.FACTOR_LOCAL_ARM]["restart_identity"] is True


def test_scenario_csl_6318_controls_schema_checksum_and_cli(tmp_path: Path) -> None:
    """SCENARIO-CSL-6318-NO-TRANSFER: controls do not mutate base state."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260811", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["unsafe_commit_count"] == 0
    assert type(artifact["unsafe_commit_count"]) is int
    assert artifact["cross_family_transfer_count"] == 0
    assert type(artifact["cross_family_transfer_count"]) is int
    assert artifact["source_model_weight_mutation_count"] == 0
    assert type(artifact["source_model_weight_mutation_count"]) is int
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is None

    missing = dict(artifact)
    missing.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing)

    bad_zero = json.loads(json.dumps(artifact))
    bad_zero["cross_family_transfer_count"] = True
    _refresh(bad_zero)
    with pytest.raises(ValueError, match="cross_family_transfer_count"):
        mod.validate_artifact(bad_zero)

    bad_ready = json.loads(json.dumps(artifact))
    bad_ready["movement_memory_and_update_cost_by_arm"][mod.FACTOR_LOCAL_ARM][
        "total_movement_cost"
    ] = bad_ready["movement_memory_and_update_cost_by_arm"][mod.FULL_STATE_ARM][
        "total_movement_cost"
    ]
    _refresh(bad_ready)
    assert bad_ready["versioned_factor_local_learning_ready_score"] == 0.0
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact({**bad_ready, "status": "complete_positive"})

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_csl_6318_deterministic_helpers_and_error_paths(tmp_path: Path) -> None:
    """REQ-CSL-6318-STREAM: deterministic helper branches stay stable."""

    events = mod.build_sealed_stream()
    assert len(events) == mod.EVENT_COUNT
    assert mod.exact_validate_event(events[0]) in mod.TARGET_STATES
    assert mod._paired_interval([]) == {"n": 0, "mean_delta": 0.0, "lower": 0.0, "upper": 0.0}
    assert mod._paired_interval([1.0]) == {
        "n": 1,
        "mean_delta": 1.0,
        "lower": 1.0,
        "upper": 1.0,
    }
    assert mod._paired_interval([0.0, 1.0])["n"] == 2
    assert mod._path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod._confusion_rates([], "empty") == {
        "row_count": 0,
        "exact_count": 0,
        "exact_rate": 0.0,
    }
    assert mod._predict_from_parameters([[0.0, 0.0, 0.0]], [1.0]) == "accept"

    bad_target = mod.StreamEvent(
        event_id="bad-target",
        chronological_index=99,
        partition="poison",
        task_family=mod.TASK_FAMILY,
        subfamily="bad_subfamily",
        template_id="bad_template",
        task_boundary=False,
        features=(1, 0, 0, 0, 0, 0),
        asp_program="invalid.",
        target_state="maybe",
        validator_key="wrong:key",
        update_allowed=False,
        poison=True,
        degradation_class=None,
    )
    with pytest.raises(ValueError, match="target_state"):
        mod.exact_validate_event(bad_target)

    bad_key = mod.StreamEvent(
        event_id="bad-key",
        chronological_index=100,
        partition="poison",
        task_family=mod.TASK_FAMILY,
        subfamily="bad_subfamily",
        template_id="bad_template",
        task_boundary=False,
        features=(1, 0, 0, 0, 0, 0),
        asp_program="invalid.",
        target_state="accept",
        validator_key="wrong:key",
        update_allowed=False,
        poison=True,
        degradation_class=None,
    )
    with pytest.raises(ValueError, match="validator_key"):
        mod.exact_validate_event(bad_key)

    with pytest.raises(ValueError, match="forced"):
        mod._require(False, "forced")

    no_tests = _artifact(tmp_path, write=False)
    no_tests["test_exit_codes"] = {mod.DEFAULT_TEST_COMMANDS[0]: 2}
    _refresh(no_tests)
    assert no_tests["versioned_factor_local_learning_ready_score"] == 0.0

    artifact = _artifact(tmp_path, write=False)
    for field in (
        "first_attempt_exact_rate_refinement_work_regret_retention_forgetting_and_negative_transfer_by_arm_and_partition",
        "paired_intervals_and_sample_sizes",
        "movement_memory_and_update_cost_by_arm",
        "task_boundary_release_receipts",
        "monitoring_degradation_and_parent_rollback_receipts",
        "test_exit_codes",
        "protected_files_unchanged",
    ):
        malformed = json.loads(json.dumps(artifact))
        malformed[field] = []
        assert mod.ready_score(malformed) == 0.0

    malformed = json.loads(json.dumps(artifact))
    metrics_key = (
        "first_attempt_exact_rate_refinement_work_regret_retention_forgetting_and_negative_transfer_by_arm_and_partition"
    )
    malformed[metrics_key]["forward_transfer_by_arm"] = []
    assert mod.ready_score(malformed) == 0.0

    malformed = json.loads(json.dumps(artifact))
    malformed[metrics_key]["forward_transfer_by_arm"][mod.FACTOR_LOCAL_ARM] = []
    assert mod.ready_score(malformed) == 0.0

    for arm in (mod.FACTOR_LOCAL_ARM, mod.FULL_STATE_ARM):
        malformed = json.loads(json.dumps(artifact))
        malformed["movement_memory_and_update_cost_by_arm"][arm] = []
        assert mod.ready_score(malformed) == 0.0

    pending = [{"status": "pending_validation", "paired": False}]
    mod._finish_unreleased_candidates(pending)
    assert pending[0]["status"] == "incomplete_validation_rejected"

    boundary_event = events[0]
    root_state = mod._reference_parameters()
    active_versions = {mod.FULL_STATE_ARM: "root"}
    releases: list[dict[str, object]] = []
    row = {
        "arm": mod.FULL_STATE_ARM,
        "parent_version_id": "root",
        "version_id": "child",
        "status": "eligible_for_boundary_release",
        "validation_window_end": boundary_event.chronological_index,
        "created_at_index": -1,
    }
    mod._release_at_boundary(
        event=boundary_event,
        version_rows=[row],
        states_by_version={"child": root_state},
        active_versions=active_versions,
        releases=releases,
    )
    assert releases == []

    root_rows = []
    root_states = {}
    root_active = {}
    for arm in mod.LEARNING_ARMS:
        version_id = f"{arm}:root"
        root_active[arm] = version_id
        root_states[version_id] = root_state
        root_rows.append(
            mod._version_row(
                arm=arm,
                version_id=version_id,
                parent_version_id=None,
                state=root_state,
                changed_factor_set=[],
                created_at_index=-1,
                created_after_outcome_sequence=0,
                movement_cost={
                    "changed_factor_count": 0,
                    "changed_parameter_count": 0,
                    "changed_state_bytes": 0,
                    "l1_movement": 0.0,
                    "total": 0.0,
                },
                champion_version_id=None,
                validation_window=[],
                status_value="active_root",
            )
        )
    root_monitoring = mod._monitor_and_rollback(
        version_rows=root_rows,
        states_by_version=root_states,
        active_versions=root_active,
    )
    assert root_monitoring["exact_parent_rollback"] is True

    large = [[100.0, 0.0, 0.0] for _ in mod.FEATURE_NAMES]
    projected = mod._project_to_reference_radius(large)
    assert projected != large
    assert len(projected) == len(mod.FEATURE_NAMES)
    assert (
        mod._chronological_forgetting_count(
            [
                {"exact": True, "partition": "replay", "target_state": "accept"},
                {"exact": False, "partition": "reversal", "target_state": "repair"},
            ]
        )
        == 1
    )

    output = tmp_path / "cli-no-validate.json"
    assert mod.main(["--date", "20260811", "--output", str(output)]) == 0
    assert output.exists()
