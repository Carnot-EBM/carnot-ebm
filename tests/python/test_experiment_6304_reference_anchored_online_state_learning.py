"""Tests for Exp6304 reference-anchored online state learning.

Spec refs: REQ-CSL-6304, REQ-CSL-6304-STREAM,
REQ-CSL-6304-PREDECISION, REQ-CSL-6304-UPDATE,
REQ-CSL-6304-CONTROLS, REQ-CSL-6304-READY,
REQ-CSL-6304-PROVENANCE, SCENARIO-CSL-6304-CHRONOLOGY,
SCENARIO-CSL-6304-PARITY, SCENARIO-CSL-6304-ROLLBACK,
SCENARIO-CSL-6304-READY.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6304_reference_anchored_online_state_learning as mod


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


def _path_from_receipt(receipt: dict[str, object]) -> Path:
    return Path(str(receipt["path"]))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _refresh(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reference_anchored_online_learning_ready_score"] = mod.ready_score(artifact)
    artifact["status"] = mod.status(artifact)
    artifact["honest_verdict"] = mod.honest_verdict(artifact)
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_req_csl_6304_spec_declares_artifact_contract() -> None:
    """REQ-CSL-6304-PROVENANCE: OpenSpec owns fields and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CSL-6304") :]

    for token in (
        "REQ-CSL-6304-STREAM",
        "REQ-CSL-6304-PREDECISION",
        "REQ-CSL-6304-UPDATE",
        "REQ-CSL-6304-CONTROLS",
        "REQ-CSL-6304-READY",
        "SCENARIO-CSL-6304-CHRONOLOGY",
        "SCENARIO-CSL-6304-PARITY",
        "SCENARIO-CSL-6304-ROLLBACK",
        "SCENARIO-CSL-6304-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        *mod.ARM_NAMES,
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_csl_6304_chronology_snapshots_precede_outcomes(tmp_path: Path) -> None:
    """SCENARIO-CSL-6304-CHRONOLOGY: labels open only after snapshots."""

    artifact = _artifact(tmp_path)
    manifest = artifact["sealed_stream_manifest_path_and_hash"]
    snapshot_receipt = artifact["immutable_predecision_snapshot_receipts"]
    outcome_receipt = artifact["postdecision_exact_outcome_receipts"]
    manifest_rows = json.loads(
        _path_from_receipt(manifest).read_text(encoding="utf-8")
    )["events"]
    snapshots = _jsonl(_path_from_receipt(snapshot_receipt))
    outcomes = _jsonl(_path_from_receipt(outcome_receipt))

    expected_decisions = len(manifest_rows) * len(mod.ARM_NAMES)
    assert len(snapshots) == expected_decisions
    assert len(outcomes) == len(manifest_rows)
    assert artifact["chronological_partition_contract"]["label_visibility"] == "postdecision_only"
    assert all(row["phase"] == "predecision" for row in snapshots)
    assert all("target_state" not in row for row in snapshots)
    latest_snapshot_by_event = {
        row["event_id"]: max(
            snap["snapshot_sequence"] for snap in snapshots if snap["event_id"] == row["event_id"]
        )
        for row in outcomes
    }
    assert all(
        row["reveal_sequence"] > latest_snapshot_by_event[row["event_id"]] for row in outcomes
    )
    assert artifact["immutable_predecision_snapshot_receipts"]["chronology_leak_count"] == 0


def test_scenario_csl_6304_update_parity_and_future_transfer(tmp_path: Path) -> None:
    """SCENARIO-CSL-6304-PARITY: budgets match and transfer is prospective."""

    artifact = _artifact(tmp_path)
    budget = artifact["matched_update_budget"]
    forward = artifact["forward_transfer_by_arm"]
    retention = artifact["retention_and_forgetting_by_arm"]
    negative = artifact["negative_transfer_by_arm"]
    utility_delta = artifact["paired_intervals_and_sample_sizes"][
        "reference_anchored_vs_unanchored_utility"
    ]

    assert budget["unanchored"]["authenticated_update_opportunities"] == budget[
        "reference_anchored"
    ]["authenticated_update_opportunities"]
    assert budget["unanchored"]["step_budget"] == budget["reference_anchored"]["step_budget"]
    assert forward["reference_anchored"]["future_same_template_delta_vs_frozen"] > 0.0
    assert forward["reference_anchored"]["held_template_delta_vs_frozen"] > 0.0
    assert forward["reference_anchored"]["unseen_family_delta_vs_frozen"] > 0.0
    assert utility_delta["mean_delta"] >= -mod.NONINFERIORITY_MARGIN
    assert retention["reference_anchored"]["forgetting_rate"] < retention["unanchored"][
        "forgetting_rate"
    ]
    assert negative["reference_anchored"]["negative_transfer_count"] < negative["unanchored"][
        "negative_transfer_count"
    ]
    assert artifact["reference_anchored_online_learning_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete_positive:")


def test_scenario_csl_6304_false_pass_poison_rollback_and_no_base_mutation(
    tmp_path: Path,
) -> None:
    """SCENARIO-CSL-6304-ROLLBACK: unsafe updates fail closed."""

    artifact = _artifact(tmp_path)
    counts = artifact["commit_reject_quarantine_and_rollback_counts"]
    rollback = artifact["rollback_and_restart_identity"]

    assert counts["reference_anchored"]["unsafe_commit_count"] == 0
    assert counts["reference_anchored"]["false_pass_injection_rejected"] is True
    assert counts["reference_anchored"]["poison_quarantine_count"] >= 1
    assert counts["reference_anchored"]["rollback_count"] >= 1
    assert rollback["exact_rollback"] is True
    assert rollback["restart_matches_active_state"] is True
    assert artifact["source_model_weight_mutation_count"] == 0
    assert type(artifact["source_model_weight_mutation_count"]) is int
    assert artifact["learned_initializer_mutation_counts"]["frozen"] == 0
    assert artifact["learned_initializer_mutation_counts"]["reference_anchored"] > 0
    assert artifact["protected_files_unchanged"]["unchanged"] is True


def test_req_csl_6304_artifact_schema_controls_and_sidecars(tmp_path: Path) -> None:
    """REQ-CSL-6304: terminal artifact and sidecars are stable."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(
        date="20260811",
        result_path=result_path,
        duration_s=2.0,
        test_exit_codes=_passing_exit_codes(),
        write=True,
    )

    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["status"] == "complete_positive"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["continuous_relaxation_path_hash_and_terminal_class"]["path"].endswith(
        "experiment_6287_asp_continuous_relaxation.json"
    )
    assert artifact["reference_snapshot_path_and_hash"]["sha256"] == mod.sha256_file(
        _path_from_receipt(artifact["reference_snapshot_path_and_hash"])
    )
    assert artifact["sealed_stream_manifest_path_and_hash"]["row_count"] == mod.EVENT_COUNT
    assert artifact["preconditions_checked"]["stream_frozen_before_fitting"] is True
    assert artifact["reversal_and_poison_results_by_arm"]["reference_anchored"][
        "unsafe_commit_count"
    ] == 0


def test_req_csl_6304_validate_artifact_fails_closed(tmp_path: Path) -> None:
    """REQ-CSL-6304-READY: validator rejects bypass-shaped artifacts."""

    artifact = _artifact(tmp_path, write=False)

    missing = dict(artifact)
    missing.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing)

    bad_oracle = dict(artifact)
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["reproducibility_checksum"] = mod.payload_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_oracle)

    bad_weight = dict(artifact)
    bad_weight["source_model_weight_mutation_count"] = 1
    bad_weight["reproducibility_checksum"] = mod.payload_checksum(bad_weight)
    with pytest.raises(ValueError, match="source_model_weight_mutation_count"):
        mod.validate_artifact(bad_weight)

    replay_only = dict(artifact)
    replay_only["forward_transfer_by_arm"] = json.loads(
        json.dumps(replay_only["forward_transfer_by_arm"])
    )
    replay_only["forward_transfer_by_arm"]["reference_anchored"][
        "held_template_delta_vs_frozen"
    ] = 0.0
    replay_only["forward_transfer_by_arm"]["reference_anchored"][
        "unseen_family_delta_vs_frozen"
    ] = 0.0
    _refresh(replay_only)
    assert replay_only["reference_anchored_online_learning_ready_score"] == 0.0
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact({**replay_only, "status": "complete_positive"})

    unsafe = dict(artifact)
    unsafe["commit_reject_quarantine_and_rollback_counts"] = json.loads(
        json.dumps(unsafe["commit_reject_quarantine_and_rollback_counts"])
    )
    unsafe["commit_reject_quarantine_and_rollback_counts"]["reference_anchored"][
        "unsafe_commit_count"
    ] = 1
    _refresh(unsafe)
    assert unsafe["reference_anchored_online_learning_ready_score"] == 0.0

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_csl_6304_deterministic_helpers_and_defensive_branches(tmp_path: Path) -> None:
    """REQ-CSL-6304-UPDATE: helper branches stay deterministic."""

    events = mod.build_sealed_stream()
    assert len(events) == mod.EVENT_COUNT
    assert mod.exact_validate_event(events[0]) in mod.TARGET_STATES
    assert mod._predict_from_parameters([[0.0, 0.0, 0.0]], [1.0]) == "accept"
    assert mod._confusion_rates([], "accept") == {
        "row_count": 0,
        "exact_count": 0,
        "exact_rate": 0.0,
    }
    assert mod._paired_interval([]) == {
        "n": 0,
        "mean_delta": 0.0,
        "lower": 0.0,
        "upper": 0.0,
    }
    assert mod._project_to_radius([3.0, 4.0], radius=5.0) == [3.0, 4.0]
    assert mod._project_to_radius([6.0, 8.0], radius=5.0) == [3.0, 4.0]

    bad_event = mod.StreamEvent(
        event_id="bad",
        chronological_index=99,
        partition="poison",
        family="bad_family",
        template_id="bad_template",
        features=(1, 0, 0, 0, 0, 0),
        asp_program="invalid.",
        target_state="accept",
        validator_key="wrong:key",
        update_allowed=True,
        poison=True,
        repeated_template=False,
    )
    with pytest.raises(ValueError, match="validator_key"):
        mod.exact_validate_event(bad_event)

    bad_target = mod.StreamEvent(
        event_id="bad-target",
        chronological_index=100,
        partition="poison",
        family="bad_family",
        template_id="bad_template",
        features=(1, 0, 0, 0, 0, 0),
        asp_program="invalid.",
        target_state="maybe",
        validator_key="wrong:key",
        update_allowed=False,
        poison=True,
        repeated_template=False,
    )
    with pytest.raises(ValueError, match="target_state"):
        mod.exact_validate_event(bad_target)

    output = tmp_path / "cli.json"
    assert mod.main(["--date", "20260811", "--output", str(output), "--validate"]) == 0
    assert output.exists()
