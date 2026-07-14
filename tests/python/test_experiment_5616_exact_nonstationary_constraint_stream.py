"""Tests for Exp5616 exact nonstationary constraint-stream fixture.

Spec refs: REQ-BENCH-5616, SCENARIO-BENCH-5616-SCHEMA,
SCENARIO-BENCH-5616-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5616_exact_nonstationary_constraint_stream as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/benchmarks/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5616_exact_nonstationary_constraint_stream.py")


def _jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_req_bench_5616_spec_declares_exact_nonstationary_contract() -> None:
    """REQ-BENCH-5616: OpenSpec anchors axes, counts, controls, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-BENCH-5616") : spec.index("### REQ-BENCH-3389")]
    normalized = " ".join(section.split())

    assert "SCENARIO-BENCH-5616-SCHEMA" in section
    assert "SCENARIO-BENCH-5616-CONTROLS" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.DATASET_RELATIVE_PATH) in section
    assert "`fixture_ready_score` SHALL be exactly `1.0`" in section
    assert "`inference_substrate` SHALL be `deterministic_verifier`" in section
    assert "SHALL NOT run an LLM" in section
    assert "SHALL NOT fit a policy" in section
    for family in mod.SPACE_SHIFT_FAMILIES:
        assert family.replace("_", "-") in normalized.replace("_", "-")
    for drift_type in mod.TEMPORAL_DRIFT_TYPES:
        assert drift_type.replace("_", "-") in normalized.replace("_", "-")
    for duration in mod.TASK_DURATIONS:
        assert str(duration) in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_5616_run_writes_replayable_exact_fixture(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5616-SCHEMA: replay preserves schema, counts, splits, and hashes."""

    artifact = mod.run(
        repo_root=tmp_path,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    dataset_path = tmp_path / mod.DATASET_RELATIVE_PATH
    written = json.loads(result_path.read_text(encoding="utf-8"))
    rows = _jsonl(dataset_path)
    replay = mod.replay_dataset(dataset_path)

    assert written == artifact
    assert replay["row_count"] == len(rows) == artifact["dataset_row_count"]
    assert replay["dataset_sha256"] == artifact["dataset_sha256"]
    assert artifact["dataset_path"] == mod.DATASET_RELATIVE_PATH.as_posix()
    assert artifact["schema_version"] == mod.ROW_SCHEMA_VERSION
    assert artifact["space_shift_families"] == list(mod.SPACE_SHIFT_FAMILIES)
    assert artifact["temporal_drift_types"] == list(mod.TEMPORAL_DRIFT_TYPES)
    assert artifact["task_durations"] == list(mod.TASK_DURATIONS)
    assert artifact["instances_per_condition"] == mod.INSTANCES_PER_CONDITION
    assert artifact["stream_count"] == (
        len(mod.SPACE_SHIFT_FAMILIES)
        * len(mod.TEMPORAL_DRIFT_TYPES)
        * len(mod.TASK_DURATIONS)
        * mod.INSTANCES_PER_CONDITION
    )
    assert len(artifact["random_seeds"]) == artifact["stream_count"]
    assert len(set(artifact["random_seeds"])) == artifact["stream_count"]
    assert artifact["exact_oracle_label_count"] == len(rows) * 6
    assert artifact["oracle_label_error_count"] == 0
    assert artifact["fixture_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["llm_invoked"] is False
    assert artifact["policy_fit"] is False
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])

    for condition, count in artifact["stream_condition_counts"].items():
        assert count == mod.INSTANCES_PER_CONDITION, condition
    assert artifact["family_duration_counts"] == replay["family_duration_counts"]
    assert artifact["content_hashes"]["dataset_sha256"] == mod.sha256_file(dataset_path)
    assert artifact["content_hashes"]["stream_order_sha256"] == mod.sha256_json(
        artifact["stream_ordering"]
    )
    assert artifact["replay_loader"]["module"] == (
        "carnot.experiment_5616_exact_nonstationary_constraint_stream"
    )
    assert artifact["replay_loader"]["function"] == "replay_dataset"

    split_receipts = artifact["split_receipts"]
    assert split_receipts["stream_id_overlap_count"] == 0
    assert split_receipts["state_id_overlap_count"] == 0
    assert split_receipts["update_id_overlap_count"] == 0
    assert split_receipts["streams_per_split"] == {"calibration": 288, "heldout": 288, "train": 576}
    for condition, counts in split_receipts["per_condition_streams"].items():
        assert counts == {"calibration": 8, "heldout": 8, "train": 16}, condition

    assert rows == sorted(rows, key=mod.row_sort_key)
    for row in rows:
        assert row["schema_version"] == mod.ROW_SCHEMA_VERSION
        assert row["space_shift_family"] in mod.SPACE_SHIFT_FAMILIES
        assert row["temporal_drift_type"] in mod.TEMPORAL_DRIFT_TYPES
        assert row["duration"] in mod.TASK_DURATIONS
        assert row["split"] in mod.SPLITS
        assert set(row["state_labels"]) == {"current_rule", "future_rule", "old_rule"}
        assert set(row["update_labels"]) == {"current_rule", "future_rule", "old_rule"}
        assert row["row_sha256"] == mod.row_content_hash(row)
        assert mod.validate_dataset_row(row)["accepted"] == row["accepted_by_exact_validator"]

    mod.validate_artifact(artifact, repo_root=tmp_path)


def test_scenario_bench_5616_corruption_controls_fail_closed() -> None:
    """SCENARIO-BENCH-5616-CONTROLS: corruptions reject and valid controls accept."""

    rows = mod.build_dataset_rows()
    summary = mod.summarize_rows(rows)

    assert summary["oracle_label_error_count"] == 0
    assert summary["corruption_controls"]["known_valid"]["accepted"] > 0
    assert summary["corruption_controls"]["known_valid"]["rejected"] == 0
    for kind in mod.CORRUPTION_KINDS:
        receipt = summary["corruption_controls"][kind]
        assert receipt["injected"] > 0
        assert receipt["accepted"] == 0
        assert receipt["rejected"] == receipt["injected"]

    valid = next(row for row in rows if row["control_kind"] == "known_valid")
    assert mod.validate_dataset_row(valid)["accepted"] is True

    wrong_predicate = deepcopy(valid)
    wrong_predicate["update"]["predicate_id"] = "wrong_predicate"
    assert mod.validate_dataset_row(wrong_predicate)["accepted"] is False

    wrong_binding = deepcopy(valid)
    wrong_binding["update"]["entity_id"] = "wrong_entity"
    assert mod.validate_dataset_row(wrong_binding)["accepted"] is False

    delayed = deepcopy(valid)
    delayed["update"]["label_step_index"] = int(valid["step_index"]) - 1
    assert mod.validate_dataset_row(delayed)["accepted"] is False

    poison = deepcopy(valid)
    poison["update"]["poison_update"] = True
    assert mod.validate_dataset_row(poison)["accepted"] is False


def test_req_bench_5616_validation_fails_closed_on_gate_or_hash_drift(tmp_path: Path) -> None:
    """REQ-BENCH-5616: fixture readiness blocks on bad substrate, counts, or replay."""

    artifact = mod.run(repo_root=tmp_path)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["reproducibility_checksum"] = mod.payload_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate, repo_root=tmp_path)

    bad_score = deepcopy(artifact)
    bad_score["fixture_ready_score"] = 1.0
    bad_score["oracle_label_error_count"] = 1
    bad_score["reproducibility_checksum"] = mod.payload_checksum(bad_score)
    with pytest.raises(ValueError, match="fixture_ready_score"):
        mod.validate_artifact(bad_score, repo_root=tmp_path)

    bad_seed = deepcopy(artifact)
    bad_seed["random_seeds"] = bad_seed["random_seeds"][:-1]
    bad_seed["reproducibility_checksum"] = mod.payload_checksum(bad_seed)
    with pytest.raises(ValueError, match="random_seeds"):
        mod.validate_artifact(bad_seed, repo_root=tmp_path)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum, repo_root=tmp_path)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("dataset_path")
    missing_principle["reproducibility_checksum"] = mod.payload_checksum(missing_principle)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing_principle, repo_root=tmp_path)

    dataset_path = tmp_path / mod.DATASET_RELATIVE_PATH
    dataset_path.write_text(dataset_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_sha256"):
        mod.validate_artifact(artifact, repo_root=tmp_path)
