"""Tests for the Exp6263 clean local-SOTA replay bridge.

Spec refs: REQ-LEARN-6263, SCENARIO-LEARN-6263-BRIDGE,
SCENARIO-LEARN-6263-QUARANTINE, SCENARIO-LEARN-6263-NEGATIVES,
SCENARIO-LEARN-6263-REPLAY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from carnot import experiment_6263_clean_sota_event_replay_bridge as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"


def _jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.write_text(
        "".join(mod.canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _copy_clean_rows(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for hf_id, source in mod.CLEAN_ROW_SIDECARS.items():
        target = tmp_path / source.name
        shutil.copyfile(REPO / source, target)
        paths[hf_id] = target
    return paths


def _run_tmp(tmp_path: Path, **kwargs: Any) -> JsonDict:
    return mod.run(
        result_path=tmp_path / "bridge.json",
        row_manifest_path=tmp_path / "bridge.rows.jsonl",
        quarantine_manifest_path=tmp_path / "bridge.quarantine.json",
        run_date="20260810",
        duration_s=1.25,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        **kwargs,
    )


def test_req_learn_6263_spec_declares_bridge_and_fields() -> None:
    """REQ-LEARN-6263: OpenSpec names the bridge contract."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6263") :]

    for marker in (
        "REQ-LEARN-6263",
        "SCENARIO-LEARN-6263-BRIDGE",
        "SCENARIO-LEARN-6263-QUARANTINE",
        "SCENARIO-LEARN-6263-NEGATIVES",
        "SCENARIO-LEARN-6263-REPLAY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_6263_builds_clean_bridge_and_quarantine(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6263-BRIDGE: clean rows produce a ready bridge."""

    artifact = _run_tmp(tmp_path, write=True)
    row_manifest = tmp_path / "bridge.rows.jsonl"
    quarantine_manifest = tmp_path / "bridge.quarantine.json"

    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["event_replay_bridge_ready_score"] == 1.0
    assert type(artifact["duplicate_count"]) is int and artifact["duplicate_count"] == 0
    assert type(artifact["time_reversal_count"]) is int and artifact["time_reversal_count"] == 0
    assert type(artifact["train_validation_test_overlap_count"]) is int
    assert artifact["train_validation_test_overlap_count"] == 0
    assert type(artifact["source_mutation_count"]) is int and artifact["source_mutation_count"] == 0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["no_model_load_receipt"]["model_load_count"] == 0
    assert artifact["no_model_load_receipt"]["llm_loaded"] is False
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    assert row_manifest.exists()
    assert quarantine_manifest.exists()
    assert artifact["immutable_row_manifest_path_and_hash"]["row_count"] == 480
    assert artifact["immutable_row_manifest_path_and_hash"]["sha256"] == mod.path_sha256(
        row_manifest
    )
    assert len(_jsonl(row_manifest)) == 480
    assert json.loads(quarantine_manifest.read_text(encoding="utf-8"))[
        "quarantined_source_ids_and_reasons"
    ] == artifact["quarantined_source_ids_and_reasons"]

    model_ids = [row["hf_id"] for row in artifact["model_specs"]]
    assert model_ids == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ]
    assert artifact["clean_source_ids"] == [
        "exp6160:artifact",
        "exp6160:rows:unsloth/Qwen3.6-35B-A3B-GGUF",
        "exp6160:rows:unsloth/gemma-4-26B-A4B-it-GGUF",
        "exp6162:artifact",
    ]
    assert all(
        row["source_disposition"] == "clean"
        for row in _jsonl(row_manifest)
    )
    assert all(
        source["source_id"].startswith("exp6146:")
        for source in artifact["quarantined_source_ids_and_reasons"]
    )
    assert {
        row["bridge_partition"]
        for row in _jsonl(row_manifest)
    } == {"train", "validation", "test"}
    assert artifact["protected_files_unchanged"]["scripts_research_conductor_py_untouched"] is True
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_learn_6263_negative_controls_are_recorded(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6263-NEGATIVES: built-in attacks reject."""

    artifact = _run_tmp(tmp_path)
    controls = artifact["replay_negative_controls"]

    assert set(controls) == {
        "duplicate",
        "reorder",
        "alias_collision",
        "row_loss",
        "parser_failure_disposition",
        "source_mutation",
    }
    assert all(row["accepted"] is False for row in controls.values())
    assert controls["duplicate"]["duplicate_count"] > 0
    assert controls["reorder"]["time_reversal_count"] > 0
    assert controls["alias_collision"]["alias_collision_count"] > 0
    assert controls["row_loss"]["row_count_mismatch_count"] > 0
    assert controls["parser_failure_disposition"]["parser_failure_mismatch_count"] > 0
    assert controls["source_mutation"]["source_mutation_count"] > 0
    assert artifact["replay_positive_control"]["accepted"] is True


def test_scenario_learn_6263_mutated_sources_block_readiness(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6263-NEGATIVES: source attack classes fail closed."""

    row_paths = _copy_clean_rows(tmp_path)

    duplicate_rows = _jsonl(row_paths[mod.QWEN_HF_ID])
    duplicate_rows.append(deepcopy(duplicate_rows[0]))
    _write_jsonl(row_paths[mod.QWEN_HF_ID], duplicate_rows)
    duplicate = _run_tmp(tmp_path, row_path_overrides=row_paths)
    assert duplicate["status"] == "blocked"
    assert duplicate["duplicate_count"] > 0
    assert duplicate["event_replay_bridge_ready_score"] == 0.0

    row_paths = _copy_clean_rows(tmp_path / "reorder")
    reordered_rows = _jsonl(row_paths[mod.QWEN_HF_ID])
    reordered_rows[0], reordered_rows[1] = reordered_rows[1], reordered_rows[0]
    _write_jsonl(row_paths[mod.QWEN_HF_ID], reordered_rows)
    reordered = _run_tmp(tmp_path / "reorder", row_path_overrides=row_paths)
    assert reordered["time_reversal_count"] > 0
    assert reordered["status"] == "blocked"

    row_paths = _copy_clean_rows(tmp_path / "alias")
    alias_rows = _jsonl(row_paths[mod.QWEN_HF_ID])
    alias_rows[1]["visible_event_hash"] = alias_rows[0]["visible_event_hash"]
    _write_jsonl(row_paths[mod.QWEN_HF_ID], alias_rows)
    alias = _run_tmp(tmp_path / "alias", row_path_overrides=row_paths)
    assert alias["chronological_order_receipts"]["alias_collision_count"] > 0
    assert alias["status"] == "blocked"

    row_paths = _copy_clean_rows(tmp_path / "loss")
    lost_rows = _jsonl(row_paths[mod.QWEN_HF_ID])[:-1]
    _write_jsonl(row_paths[mod.QWEN_HF_ID], lost_rows)
    loss = _run_tmp(tmp_path / "loss", row_path_overrides=row_paths)
    assert loss["source_artifact_paths_hashes_and_terminal_classes"]["row_count_mismatch_count"] > 0
    assert loss["status"] == "blocked"

    row_paths = _copy_clean_rows(tmp_path / "parser")
    parser_rows = _jsonl(row_paths[mod.QWEN_HF_ID])
    parser_row = next(row for row in parser_rows if row["invalid_output"] is True)
    parser_row["invalid_output"] = False
    parser_row["answer_parse_state"] = "complete"
    parser_row["strategy_parse_state"] = "complete"
    parser_row["terminal_parse_status"] = "complete"
    _write_jsonl(row_paths[mod.QWEN_HF_ID], parser_rows)
    parser = _run_tmp(tmp_path / "parser", row_path_overrides=row_paths)
    disposition = parser["malformed_or_parser_failure_count_by_disposition"]
    assert disposition["parser_failure_mismatch_count"] > 0
    assert parser["status"] == "blocked"

    row_paths = _copy_clean_rows(tmp_path / "mutation")
    expected = {str(path): mod.path_sha256(path) for path in row_paths.values()}
    row_paths[mod.QWEN_HF_ID].write_text(
        row_paths[mod.QWEN_HF_ID].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    mutation = _run_tmp(
        tmp_path / "mutation",
        row_path_overrides=row_paths,
        expected_source_hashes=expected,
    )
    assert mutation["source_mutation_count"] > 0
    assert mutation["status"] == "blocked"


def test_scenario_learn_6263_byte_identical_replay(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6263-REPLAY: repeated materialization is byte-stable."""

    first = _run_tmp(tmp_path / "first", write=True)
    second = _run_tmp(tmp_path / "second", write=True)

    first_rows = (tmp_path / "first" / "bridge.rows.jsonl").read_bytes()
    second_rows = (tmp_path / "second" / "bridge.rows.jsonl").read_bytes()

    assert first_rows == second_rows
    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert first["immutable_row_manifest_path_and_hash"]["sha256"] == second[
        "immutable_row_manifest_path_and_hash"
    ]["sha256"]


def test_req_learn_6263_validation_rejects_bad_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-6263: validation keeps readiness conjunctive."""

    artifact = _run_tmp(tmp_path)
    assert mod._line_count(tmp_path / "missing.jsonl") is None

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    bad_score = deepcopy(artifact)
    bad_score["event_replay_bridge_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="event_replay_bridge_ready_score"):
        mod.validate_artifact(bad_score)

    bad_duplicate = deepcopy(artifact)
    bad_duplicate["duplicate_count"] = {"value": 0}
    bad_duplicate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_duplicate)
    with pytest.raises(ValueError, match="duplicate_count"):
        mod.validate_artifact(bad_duplicate)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete_ready: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["event_replay_bridge_ready_score"] = mod._ready_score(bad_substrate)
    bad_substrate["status"] = mod._status(bad_substrate)
    bad_substrate["honest_verdict"] = mod._honest_verdict(bad_substrate)
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = True
    bad_oracle["event_replay_bridge_ready_score"] = mod._ready_score(bad_oracle)
    bad_oracle["status"] = mod._status(bad_oracle)
    bad_oracle["honest_verdict"] = mod._honest_verdict(bad_oracle)
    bad_oracle["reproducibility_checksum"] = mod.reproducibility_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_oracle)

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"]["status"] = "wrong"
    bad_principles["reproducibility_checksum"] = mod.reproducibility_checksum(bad_principles)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"] = []
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance)

    bad_provenance_field = deepcopy(artifact)
    bad_provenance_field["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance_field["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_field
    )
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance_field)

    bad_tests = deepcopy(artifact)
    command = next(iter(bad_tests["test_exit_codes"]))
    bad_tests["test_exit_codes"][command] = 1
    bad_tests["event_replay_bridge_ready_score"] = mod._ready_score(bad_tests)
    bad_tests["status"] = mod._status(bad_tests)
    bad_tests["honest_verdict"] = mod._honest_verdict(bad_tests)
    bad_tests["reproducibility_checksum"] = mod.reproducibility_checksum(bad_tests)
    assert "test_exit_codes" in bad_tests["honest_verdict"]
    assert mod.validate_artifact(bad_tests) is True

    bad_protected = deepcopy(artifact)
    bad_protected["protected_files_unchanged"]["unchanged"] = False
    bad_protected["event_replay_bridge_ready_score"] = mod._ready_score(bad_protected)
    bad_protected["status"] = mod._status(bad_protected)
    bad_protected["honest_verdict"] = mod._honest_verdict(bad_protected)
    bad_protected["reproducibility_checksum"] = mod.reproducibility_checksum(bad_protected)
    assert "protected_files_changed" in bad_protected["honest_verdict"]
    assert mod.validate_artifact(bad_protected) is True

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)
