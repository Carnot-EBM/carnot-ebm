"""Tests for Exp6179 retention-safe continuous strategy learning.

Spec refs: REQ-CL-6179-MANDATORY-EXECUTION, REQ-CL-6179-LOCAL-GGUF,
REQ-CL-6179-IMMUTABLE-WEIGHTS, REQ-CL-6179-EXTERNAL-MEMORY,
REQ-CL-6179-POST-OUTCOME-WRITE, REQ-CL-6179-BOUNDED-REPLAY,
REQ-CL-6179-RETENTION, REQ-CL-6179-POISON-QUARANTINE,
REQ-CL-6179-ROLLBACK, REQ-CL-6179-PROTECTED-FILES,
REQ-CL-6179-ARMS, REQ-CL-6179-RECEIPTS,
SCENARIO-CL-6179-SEALED-ARMS,
SCENARIO-CL-6179-RETENTION-AFTER-UPDATE,
SCENARIO-CL-6179-POISON-ROLLBACK, SCENARIO-CL-6179-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6179_retention_safe_continuous_strategy_learning_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _cache_records() -> list[dict[str, Any]]:
    rows = []
    for index, spec in enumerate(mod.MODEL_SPECS):
        rows.append(
            {
                "hf_id": spec["hf_id"],
                "name": spec["name"],
                "role": spec["role"],
                "quantization": spec["quantization"],
                "revision": f"fixture-revision-{index}",
                "path": f"/tmp/exp6179/{mod.model_slug(spec['hf_id'])}.gguf",
                "exists": True,
                "size_bytes": 16_000_000_000 + index,
                "checksum": mod.sha256_text(spec["hf_id"]),
                "checksum_source": "fixture-cache-oid",
                "usable_for_local_gguf": True,
            }
        )
    return rows


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        memory_dir=tmp_path / "experiment_6179_memory",
        model_cache_records=_cache_records(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.25,
        write=write,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["retention_safe_continuous_strategy_learning_ready_score"] = mod.ready_score(
        artifact
    )
    artifact["missing_verifier_gaps"] = mod.missing_verifier_gaps(artifact)
    artifact["status"] = mod.status(artifact)
    artifact["honest_verdict"] = mod.honest_verdict(artifact)
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def test_req_6179_spec_declares_retention_safe_contract() -> None:
    """REQ-CL-6179-RECEIPTS: OpenSpec owns the Exp6179 artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-CL-6179-MANDATORY-EXECUTION") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-CL-6179-MANDATORY-EXECUTION",
        "REQ-CL-6179-LOCAL-GGUF",
        "REQ-CL-6179-IMMUTABLE-WEIGHTS",
        "REQ-CL-6179-EXTERNAL-MEMORY",
        "REQ-CL-6179-POST-OUTCOME-WRITE",
        "REQ-CL-6179-BOUNDED-REPLAY",
        "REQ-CL-6179-RETENTION",
        "REQ-CL-6179-POISON-QUARANTINE",
        "REQ-CL-6179-ROLLBACK",
        "REQ-CL-6179-PROTECTED-FILES",
        "REQ-CL-6179-ARMS",
        "SCENARIO-CL-6179-SEALED-ARMS",
        "SCENARIO-CL-6179-RETENTION-AFTER-UPDATE",
        "SCENARIO-CL-6179-POISON-ROLLBACK",
        "SCENARIO-CL-6179-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MODEL_SPECS[0]["hf_id"],
        mod.MODEL_SPECS[1]["hf_id"],
        *mod.ARM_NAMES,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6179_sealed_stream_five_arms_and_post_outcome_writes(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6179-SEALED-ARMS: five arms share one sealed stream."""

    artifact = _artifact(tmp_path, write=True)

    assert (tmp_path / mod.RESULT_RELATIVE_PATH.name).is_file()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["mandatory_artifact_written"] is True
    assert artifact["MODEL_SPECS"] == mod.MODEL_SPECS
    assert artifact["model_specs"] == mod.MODEL_SPECS

    stream = artifact["sealed_chronological_stream_receipt"]
    assert stream["chronological"] is True
    assert stream["sealed"] is True
    assert stream["current_label_visible_before_decision_count"] == 0
    assert stream["event_count"] == len(mod.default_stream())

    memory = artifact["task_owned_external_memory_receipt"]
    assert memory["task_owned_external_memory_only"] is True
    assert memory["weight_memory_boundary"] == "external_strategy_state_only"
    assert all(Path(path).is_file() for path in memory["written_sidecar_paths"].values())

    arms = artifact["arm_definitions_and_resource_matching"]
    assert arms["arm_names"] == list(mod.ARM_NAMES)
    assert arms["arm_count"] == 5
    assert arms["all_arms_matched"] is True

    writes = artifact["exact_post_outcome_write_receipts"]
    assert writes["same_decision_write_count"] == 0
    assert writes["all_commits_after_exact_outcome"] is True
    assert writes["commit_count"] == artifact["prior_family_retention_after_every_update"][
        "selected_arm_commit_count"
    ]
    assert writes["commit_count"] > 0

    retention = artifact["prior_family_retention_after_every_update"]
    assert retention["selected_arm"] == "replay"
    assert retention["selected_arm_min_prior_family_retention"] == 1.0
    assert retention["measured_after_every_admitted_update"] is True
    assert len(retention["by_arm"]["replay"]) == writes["commit_count"]

    rollback = artifact["rollback_and_quarantine_receipts"]
    assert rollback["rollback_exact"] is True
    assert rollback["rollback_past_root_failed_closed"] is True
    assert rollback["poison_propagation_count"] == 0
    assert rollback["quarantine_precision"] == 1.0
    assert rollback["quarantine_recall"] == 1.0

    assert artifact["retention_safe_continuous_strategy_learning_ready_score"] == 1.0
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact


def test_req_6179_store_quarantine_duplicate_eviction_and_rollback() -> None:
    """REQ-CL-6179-ROLLBACK/POISON-QUARANTINE: store operations fail closed."""

    store = mod.BoundedStrategyStore(
        max_records=3,
        protected_families=mod.PROTECTED_FAMILIES,
    )
    events = mod.default_stream()
    first = store.apply_event(events[0], exact_outcome_seen=True)
    duplicate = store.apply_event(events[0], exact_outcome_seen=True)
    protected = store.apply_event(events[2], exact_outcome_seen=True)
    poison = store.apply_event(events[5], exact_outcome_seen=True)
    failed = store.apply_event(
        mod.StreamEvent(
            "evt-failed",
            99,
            "geometry",
            "failed exact outcome case",
            "bad_geometry_rule",
            "rejected",
        ),
        exact_outcome_seen=True,
    )
    for event in (events[1], events[3], events[4], events[6]):
        store.apply_event(event, exact_outcome_seen=True)

    assert first["action"] == "commit"
    assert duplicate["action"] == "duplicate"
    assert duplicate["before_state_hash"] == duplicate["after_state_hash"]
    assert protected["protected"] is True
    assert poison["action"] == "quarantine"
    assert poison["rollback_exact"] is True
    assert failed["reason"] == "failed_exact_outcome"
    assert len(store.records) <= 3
    assert any(record.family in mod.PROTECTED_FAMILIES for record in store.records)

    protected_overflow = mod.BoundedStrategyStore(
        max_records=1,
        protected_families=("protected_safety",),
        records=[
            mod.StrategyRecord("r1", "p1", 1, "protected_safety", "a", "h1", True),
            mod.StrategyRecord("r2", "p2", 2, "protected_safety", "b", "h2", True),
        ],
    )
    assert protected_overflow._evict_if_needed() == []
    assert len(protected_overflow.records) == 2

    rollback = store.clone().rollback_to(first["after_state_hash"])
    assert rollback["restored_state_hash"] == first["after_state_hash"]
    assert rollback["rollback_exact"] is True
    with pytest.raises(ValueError, match="unknown rollback target"):
        store.rollback_to(mod.sha256_text("missing-state"))


def test_req_6179_ready_score_rejects_utility_without_retention_or_safety(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6179-RETENTION-AFTER-UPDATE: utility cannot hide forgetting."""

    artifact = _artifact(tmp_path)
    assert mod.ready_score(artifact) == 1.0

    bad_retention = deepcopy(artifact)
    bad_retention["prior_family_retention_after_every_update"][
        "selected_arm_min_prior_family_retention"
    ] = 0.75
    _refresh(bad_retention)
    assert bad_retention["retention_safe_continuous_strategy_learning_ready_score"] == 0.0
    assert bad_retention["status"] == "complete_null"
    assert "prior_family_retention_regression" in bad_retention["missing_verifier_gaps"]
    assert mod.validate_artifact(bad_retention) is True

    bad_utility = deepcopy(artifact)
    bad_utility["utility_by_arm_family_and_model"]["by_model"][mod.MODEL_SPECS[0]["hf_id"]][
        "replay_minus_write_through_ci95"
    ][0] = 0.0
    _refresh(bad_utility)
    assert "replay_positive_utility_not_met" in bad_utility["missing_verifier_gaps"]

    bad_poison = deepcopy(artifact)
    bad_poison["rollback_and_quarantine_receipts"]["poison_propagation_count"] = 1
    _refresh(bad_poison)
    assert "poison_propagation" in bad_poison["missing_verifier_gaps"]

    bad_weight = deepcopy(artifact)
    bad_weight["model_weight_immutability_receipt"]["all_unchanged"] = False
    _refresh(bad_weight)
    assert "model_weight_immutability_failed" in bad_weight["missing_verifier_gaps"]

    bad_test = deepcopy(artifact)
    bad_test["test_exit_codes"][mod.DEFAULT_TEST_COMMANDS[0]] = 1
    _refresh(bad_test)
    assert "test_failure" in bad_test["missing_verifier_gaps"]

    combined = deepcopy(artifact)
    combined["MODEL_SPECS"][0]["hf_id"] = "Qwen/Qwen3.5-0.8B"
    combined["arm_definitions_and_resource_matching"]["all_arms_matched"] = False
    combined["sealed_chronological_stream_receipt"]["sealed"] = False
    combined["task_owned_external_memory_receipt"]["task_owned_external_memory_only"] = False
    combined["exact_post_outcome_write_receipts"]["all_commits_after_exact_outcome"] = False
    combined["rollback_and_quarantine_receipts"]["rollback_exact"] = False
    combined["state_bound_receipt"]["within_bounds"] = False
    combined["protected_files_unchanged"]["unchanged"] = False
    gaps = mod.missing_verifier_gaps(combined)
    for gap in (
        "model_identity_mismatch",
        "arm_matching_failed",
        "stream_not_sealed",
        "external_memory_boundary_failed",
        "post_outcome_write_failed",
        "rollback_failed",
        "state_bound_exceeded",
        "protected_files_changed",
    ):
        assert gap in gaps


def test_scenario_6179_schema_validation_rejects_bypasses(tmp_path: Path) -> None:
    """SCENARIO-CL-6179-SCHEMA: validation rejects hidden bypasses."""

    artifact = _artifact(tmp_path)

    missing = dict(artifact)
    missing.pop("mandatory_artifact_written")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_model = deepcopy(artifact)
    bad_model["MODEL_SPECS"][0]["hf_id"] = "Qwen/Qwen3.5-0.8B"
    bad_model["reproducibility_checksum"] = mod.reproducibility_checksum(bad_model)
    with pytest.raises(ValueError, match="MODEL_SPECS"):
        mod.validate_artifact(bad_model)

    bad_arms = deepcopy(artifact)
    bad_arms["arm_definitions_and_resource_matching"]["arm_names"].pop()
    bad_arms["reproducibility_checksum"] = mod.reproducibility_checksum(bad_arms)
    with pytest.raises(ValueError, match="arm_definitions"):
        mod.validate_artifact(bad_arms)

    bad_write = deepcopy(artifact)
    bad_write["exact_post_outcome_write_receipts"]["same_decision_write_count"] = 1
    bad_write["reproducibility_checksum"] = mod.reproducibility_checksum(bad_write)
    with pytest.raises(ValueError, match="same-decision"):
        mod.validate_artifact(bad_write)

    bad_protected = deepcopy(artifact)
    bad_protected["protected_files_unchanged"]["unchanged"] = False
    bad_protected["reproducibility_checksum"] = mod.reproducibility_checksum(bad_protected)
    with pytest.raises(ValueError, match="protected_files"):
        mod.validate_artifact(bad_protected)

    bad_score = deepcopy(artifact)
    bad_score["retention_safe_continuous_strategy_learning_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_score)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_continuous = deepcopy(artifact)
    bad_continuous["continuous_self_learning_task"] = False
    bad_continuous["reproducibility_checksum"] = mod.reproducibility_checksum(bad_continuous)
    with pytest.raises(ValueError, match="continuous_self_learning_task"):
        mod.validate_artifact(bad_continuous)

    bad_mandatory = deepcopy(artifact)
    bad_mandatory["mandatory_artifact_written"] = False
    bad_mandatory["reproducibility_checksum"] = mod.reproducibility_checksum(bad_mandatory)
    with pytest.raises(ValueError, match="mandatory_artifact_written"):
        mod.validate_artifact(bad_mandatory)

    bad_mirror = deepcopy(artifact)
    bad_mirror["model_specs"] = []
    bad_mirror["reproducibility_checksum"] = mod.reproducibility_checksum(bad_mirror)
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(bad_mirror)

    bad_post_outcome = deepcopy(artifact)
    bad_post_outcome["exact_post_outcome_write_receipts"][
        "all_commits_after_exact_outcome"
    ] = False
    bad_post_outcome["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_post_outcome
    )
    with pytest.raises(ValueError, match="post-outcome"):
        mod.validate_artifact(bad_post_outcome)

    bad_gaps = deepcopy(artifact)
    bad_gaps["missing_verifier_gaps"] = ["wrong"]
    bad_gaps["reproducibility_checksum"] = mod.reproducibility_checksum(bad_gaps)
    with pytest.raises(ValueError, match="missing_verifier_gaps"):
        mod.validate_artifact(bad_gaps)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_provenance_type = deepcopy(artifact)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance)


def test_req_6179_cache_snapshot_load_json_and_blocked_edges(tmp_path: Path) -> None:
    """REQ-CL-6179-LOCAL-GGUF: cache and JSON helpers are explicit."""

    assert mod.sha256_file(tmp_path / "missing.bin") is None

    cache_root = tmp_path / "hub"
    qwen_path = (
        cache_root
        / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
        / "snapshots"
        / "rev-qwen"
        / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    )
    gemma_path = (
        cache_root
        / "models--unsloth--gemma-4-26B-A4B-it-GGUF"
        / "snapshots"
        / "rev-gemma"
        / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    )
    qwen_path.parent.mkdir(parents=True)
    gemma_path.parent.mkdir(parents=True)
    qwen_path.write_bytes(b"qwen")
    gemma_path.write_bytes(b"gemma")

    snapshot = mod.snapshot_model_caches(cache_root=cache_root)
    assert snapshot["all_usable"] is True
    assert [row["hf_id"] for row in snapshot["records"]] == [
        spec["hf_id"] for spec in mod.MODEL_SPECS
    ]
    assert snapshot["records"][0]["revision"] == "rev-qwen"
    assert snapshot["records"][1]["revision"] == "rev-gemma"
    assert snapshot["records"][0]["checksum"] == mod.sha256_file(qwen_path)

    object_path = tmp_path / "object.json"
    object_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps([]), encoding="utf-8")
    assert mod.load_json(tmp_path / "missing.json") == {}
    assert mod.load_json(object_path) == {"ok": True}
    with pytest.raises(ValueError, match="did not contain"):
        mod.load_json(list_path)
    assert mod.model_slug("unsloth/gemma-4-26B-A4B-it-GGUF") == "gemma_4_26b_a4b_it"

    symlink_cache = tmp_path / "symlink_hub"
    blob = symlink_cache / "models--unsloth--Qwen3.6-35B-A3B-GGUF" / "blobs" / ("a" * 64)
    link = (
        symlink_cache
        / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
        / "snapshots"
        / "rev-qwen"
        / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    )
    gemma = (
        symlink_cache
        / "models--unsloth--gemma-4-26B-A4B-it-GGUF"
        / "snapshots"
        / "rev-gemma"
        / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    )
    blob.parent.mkdir(parents=True)
    link.parent.mkdir(parents=True)
    gemma.parent.mkdir(parents=True)
    blob.write_bytes(b"x")
    link.symlink_to("../../blobs/" + ("a" * 64))
    gemma.write_bytes(b"gemma")
    symlink_snapshot = mod.snapshot_model_caches(cache_root=symlink_cache)
    assert symlink_snapshot["records"][0]["checksum"] == "sha256:" + ("a" * 64)
    assert symlink_snapshot["records"][0]["checksum_source"] == "huggingface_cache_blob_oid"

    class _GitResult:
        stdout = "?? " + str(tmp_path / "ignored.json") + "\n M python/carnot/example.py\n"

    original_run = mod.subprocess.run
    try:
        mod.subprocess.run = lambda *args, **kwargs: _GitResult()
        git_status = mod._git_status(tmp_path / "ignored.json", tmp_path / "ignored_dir")
        assert git_status["raw_filtered_task_outputs"] == [" M python/carnot/example.py"]
        mod.subprocess.run = lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
        assert mod._git_status(tmp_path / "ignored.json", tmp_path / "ignored_dir")[
            "raw_filtered_task_outputs"
        ] == []
    finally:
        mod.subprocess.run = original_run

    timed = mod.run(
        result_path=tmp_path / "timed.json",
        memory_dir=tmp_path / "timed_memory",
        model_cache_records=_cache_records(),
        test_exit_codes=_passing_exit_codes(),
    )
    assert timed["duration_s"] >= 0.0

    blocked = mod.run(
        result_path=tmp_path / "blocked.json",
        memory_dir=tmp_path / "blocked_memory",
        model_cache_records=[{**row, "usable_for_local_gguf": False} for row in _cache_records()],
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.0,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert "local_gguf_cache_not_usable" in blocked["missing_verifier_gaps"]
    assert mod.validate_artifact(blocked) is True
