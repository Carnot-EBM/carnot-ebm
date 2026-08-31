"""Tests for stepwise read-only ARC strategy accrual.

Spec refs: REQ-AGENTIC-6810-2 and
SCENARIO-AGENTIC-6810-2-READ-ONLY-LIVE-PATH.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6819_arc_stepwise_strategy_accrual as exp


def _model(tmp_path: Path) -> dict:
    spec = exp.MODEL_SPECS[0]
    path = tmp_path / spec["filename"]
    path.write_bytes(b"GGUF fixture")
    return {
        **deepcopy(spec),
        "model_path": str(path),
        "model_sha256": spec["expected_sha256"],
        "revision": spec["revision"],
        "resolved": True,
        "tokenizer": {
            "source": "llama.cpp_embedded_gguf",
            "loadable": True,
            "chat_template": spec["chat_template"],
        },
    }


def _preflight(tmp_path: Path, *, passed: bool = True) -> dict:
    model = _model(tmp_path)
    selected = {
        "index": 0,
        "uuid": exp.EXPECTED_GPU_UUIDS[0],
        "name": "NVIDIA GeForce RTX 3090",
        "memory_free_mb": 24_000,
        "memory_used_mb": 100,
        "active_compute_processes": [],
    }
    check = {
        "check": "task_owned_gpu_lease",
        "expected": "one unowned RTX 3090",
        "observed": selected if passed else {"owners": [991]},
        "passed": passed,
    }
    return {
        "all_passed": passed,
        "checks": [check],
        "models": [model],
        "device_selection_receipt": {"selected_device": selected if passed else None},
        "registry_precheck": {
            "passed": True,
            "complete_levels_scheduled": [],
            "registry_sha256": "sha256:" + "1" * 64,
        },
        "source_artifact_hashes": {
            "operational_obligation": "sha256:" + "2" * 64,
            "durable_evidence": "sha256:" + "3" * 64,
        },
    }


def _complete_row(cell: dict, *, pid: int) -> dict:
    action = {"kind": 1, "data": None}
    action_hash = exp.sha256_json(action)
    active = cell["attempt"] == "active"
    row = {
        **deepcopy(cell),
        "status": "complete",
        "durable_reuse": False,
        "strategy_hash": cell["strategy_hash"],
        "strategy_reads": 1 if cell["guidance_mode"] == "episode_start_static" else 2,
        "production_action": action,
        "production_action_hash": action_hash,
        "selected_action_hash": action_hash,
        "action_influenced": False,
        "exact_obligation_receipt": exp.build_exact_obligation_receipt(action),
        "exact_next_action_outcome": (
            {
                "observed": True,
                "frame_before_hash": "sha256:" + "4" * 64,
                "frame_after_hash": "sha256:" + "5" * 64,
                "frame_changed": True,
                "level_before": 0,
                "level_after": 0,
            }
            if active
            else None
        ),
        "process_receipt": {
            "pid": pid,
            "ppid": 100,
            "fresh_process": True,
            "task_owned": True,
            "external_process_reused": False,
        },
        "gpu_receipt": {
            "device_uuid": exp.EXPECTED_GPU_UUIDS[0],
            "lease_owner": {"pid": pid, "task_id": cell["row_id"], "owner_bound": True},
            "gpu_layers": {"requested": 999, "offloaded": 65, "total": 65},
            "tokens": {"prompt": 64, "predicted": 8},
            "lease_release": {"released": True},
            "vram_recovery": {"passed": True},
            "unrelated_processes_signaled": [],
        },
        "checkpoint_receipt": {
            "atomic_replace": True,
            "restart_loaded": True,
            "manifest_hash_match": True,
            "sha256": "sha256:" + "6" * 64,
        },
        "active_episode_write_count": 0,
        "adapter_use_count": 0,
        "source_access_count": 0,
        "tokens": {"prompt": 64, "predicted": 8},
        "actions": 1,
        "teardown_passed": True,
        "solve_claim": False,
        "solve_provenance": "live_agent_self_discovery",
        "failure_class": None,
        "row_hash": "",
    }
    row["row_hash"] = exp.row_checksum(row)
    return row


def test_req_6810_2_model_and_manifest_are_frozen_exactly() -> None:
    """REQ-AGENTIC-6810-2 fixes the model, cells, limits, and strategy bytes."""

    manifest = exp.freeze_manifest()

    assert [row["model_id"] for row in exp.MODEL_SPECS] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF"
    ]
    assert manifest["MODEL_SPECS"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert manifest["guidance_modes"] == ["episode_start_static", "stepwise_read_only"]
    assert manifest["attempts"] == ["shadow", "active"]
    assert manifest["source_access_allowed"] is False
    assert manifest["adapter_allowed"] is False
    assert manifest["censoring"]["rule"] == "bounded_action_or_wall_clock"
    assert manifest["manifest_sha256"] == exp.manifest_checksum(manifest)
    assert set(manifest["immutable_strategy_hashes"]) == set(manifest["guidance_modes"])


def test_scenario_6810_2_immutable_static_and_stepwise_retrieval() -> None:
    """SCENARIO-AGENTIC-6810-2-READ-ONLY-LIVE-PATH fixes read behavior."""

    snapshot = exp.ImmutableStrategySnapshot.freeze({"advice": ["prefer novel effects"]})
    before = snapshot.canonical_bytes
    static = exp.StrategyEpisode(snapshot, "episode_start_static")
    stepwise = exp.StrategyEpisode(snapshot, "stepwise_read_only")

    static.begin()
    assert static.read(0) == snapshot.payload
    assert static.read(1) is None
    static.end()
    stepwise.begin()
    assert stepwise.read(0) == snapshot.payload
    assert stepwise.read(1) == snapshot.payload
    stepwise.end()

    assert static.read_count == 1
    assert stepwise.read_count == 2
    assert snapshot.canonical_bytes == before
    assert snapshot.sha256 == exp.sha256_bytes(before)


def test_scenario_6810_2_active_episode_write_is_rejected() -> None:
    """REQ-AGENTIC-6810-2 rejects durable writes during an active episode."""

    episode = exp.StrategyEpisode(
        exp.ImmutableStrategySnapshot.freeze({"advice": ["observe"]}),
        "stepwise_read_only",
    )
    episode.begin()

    with pytest.raises(exp.ActiveEpisodeWriteRejected, match="active_episode_write_rejected"):
        episode.persist({"advice": ["changed"]})

    assert episode.write_count == 0
    episode.end()


def test_scenario_6810_2_shadow_action_identity_and_exact_authority() -> None:
    """REQ-AGENTIC-6810-2 keeps strategy outside action authority."""

    action = {"kind": 6, "data": {"x": 12, "y": 8}}
    shadow = exp.preserve_shadow_action(action, {"rank": ["RESET", "ACTION6"]})
    obligation = exp.build_exact_obligation_receipt(action)

    assert shadow["production_action_hash"] == shadow["selected_action_hash"]
    assert shadow["action_influenced"] is False
    assert shadow["strategy_authorized_action"] is False
    assert obligation["authority"] == "operational_obligation_v3"
    assert obligation["hard_violation_count"] == 0
    assert obligation["selected_action"] == action
    assert "strategy" not in json.dumps(obligation).lower()

    with pytest.raises(exp.AccrualEvidenceError, match="exact_authority_mismatch"):
        exp.verify_exact_authority(action, {**obligation, "selected_action": {"kind": 1}})


def test_req_6810_2_source_adapter_and_process_guards_fail_closed() -> None:
    """REQ-AGENTIC-6810-2 rejects source, adapter, and process boundary loss."""

    assert exp.prohibition_errors(source_access_count=0, adapter_use_count=0) == []
    assert exp.prohibition_errors(source_access_count=1, adapter_use_count=0) == [
        "source_access"
    ]
    assert exp.prohibition_errors(source_access_count=0, adapter_use_count=1) == [
        "adapter_use"
    ]
    receipts = [
        {
            "row_id": "a",
            "pid": 201,
            "fresh_process": True,
            "task_owned": True,
            "external_process_reused": False,
        },
        {
            "row_id": "b",
            "pid": 202,
            "fresh_process": True,
            "task_owned": True,
            "external_process_reused": False,
        },
    ]
    assert exp.process_isolation_errors(receipts) == []
    receipts[1]["pid"] = 201
    receipts[1]["external_process_reused"] = True
    assert set(exp.process_isolation_errors(receipts)) == {"duplicate_pid", "external_reuse:b"}


def test_req_6810_2_checkpoint_restart_is_atomic_and_manifest_bound(tmp_path: Path) -> None:
    """REQ-AGENTIC-6810-2 restores only an atomic checkpoint for this manifest."""

    checkpoint = tmp_path / "checkpoint.json"
    written = exp.write_checkpoint_atomic(
        checkpoint,
        manifest_sha256="sha256:" + "a" * 64,
        rows=[{"row_id": "row-0"}],
    )
    loaded, receipt = exp.load_checkpoint(
        checkpoint,
        manifest_sha256="sha256:" + "a" * 64,
    )

    assert loaded == [{"row_id": "row-0"}]
    assert written["atomic_replace"] is True
    assert receipt["restart_loaded"] is True
    assert receipt["manifest_hash_match"] is True
    assert not list(tmp_path.glob(f".{checkpoint.name}.*"))

    with pytest.raises(exp.AccrualEvidenceError, match="checkpoint_manifest_mismatch"):
        exp.load_checkpoint(checkpoint, manifest_sha256="sha256:" + "b" * 64)


def test_req_6810_2_durable_filter_reuses_only_complete_compatible_rows() -> None:
    """REQ-AGENTIC-6810-2 freezes the durable filter before bounded top-up."""

    manifest = exp.freeze_manifest()
    cell = manifest["cells"][0]
    compatible = _complete_row(cell, pid=301)
    compatible["durable_reuse"] = True
    compatible["row_hash"] = exp.row_checksum(compatible)
    incomplete = {**deepcopy(compatible), "row_id": "incomplete", "status": "blocked"}
    wrong_model = {**deepcopy(compatible), "row_id": "wrong-model", "model_id": "other/model"}

    reused, receipt = exp.filter_compatible_durable_rows(
        [{"rows": [compatible, incomplete, wrong_model]}], manifest
    )
    top_up = exp.bounded_top_up_cells(manifest, reused)

    assert [row["row_id"] for row in reused] == [cell["row_id"]]
    assert receipt["eligible_row_count"] == 1
    assert receipt["filter_sha256"] == manifest["durable_row_filter"]["filter_sha256"]
    assert cell["row_id"] not in {row["row_id"] for row in top_up}
    assert len(top_up) == len(manifest["cells"]) - 1


def test_req_6810_2_complete_rows_set_ready_independent_of_effect(tmp_path: Path) -> None:
    """REQ-AGENTIC-6810-2 makes completeness, not effect sign, control readiness."""

    manifest = exp.freeze_manifest()
    rows = [_complete_row(cell, pid=400 + index) for index, cell in enumerate(manifest["cells"])]
    artifact = exp.build_artifact(
        manifest=manifest,
        models=[_model(tmp_path)],
        preflight=_preflight(tmp_path),
        rows=rows,
        durable_filter_receipt={
            "filter_sha256": manifest["durable_row_filter"]["filter_sha256"],
            "eligible_row_count": 0,
            "inspected_artifact_count": 1,
        },
        duration_s=12.5,
        commands=["experiment command"],
    )

    assert artifact["stepwise_strategy_accrual_ready"] is True
    assert artifact["shadow_action_identity"]["passed"] is True
    assert artifact["exact_next_action_outcomes"]["complete"] is True
    assert artifact["active_episode_write_count"] == 0
    assert artifact["adapter_use_count"] == 0
    assert artifact["source_access_count"] == 0
    assert artifact["solve_claim"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "null"
    assert set(artifact["field_principles"]) == set(artifact)
    assert exp.validate_artifact(artifact) == []


def test_req_6810_2_row_completeness_and_tamper_detection(tmp_path: Path) -> None:
    """REQ-AGENTIC-6810-2 emits every game-seed-mode-attempt row exactly once."""

    manifest = exp.freeze_manifest()
    rows = [_complete_row(cell, pid=500 + index) for index, cell in enumerate(manifest["cells"])]
    rows[-1]["teardown_passed"] = False
    rows[-1]["row_hash"] = exp.row_checksum(rows[-1])
    artifact = exp.build_artifact(
        manifest=manifest,
        models=[_model(tmp_path)],
        preflight=_preflight(tmp_path),
        rows=rows,
        durable_filter_receipt={
            "filter_sha256": manifest["durable_row_filter"]["filter_sha256"],
            "eligible_row_count": 0,
            "inspected_artifact_count": 1,
        },
        duration_s=1.0,
        commands=[],
    )

    assert [row["row_id"] for row in artifact["rows"]] == [
        row["row_id"] for row in manifest["cells"]
    ]
    assert artifact["stepwise_strategy_accrual_ready"] is False
    assert artifact["verdict_class"] == "partial"
    assert artifact["gate_check_summary"]["failed_check"].startswith("teardown")

    artifact["rows"].pop()
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert "row_denominator_or_order" in exp.validate_artifact(artifact)


def test_req_6810_2_blocked_preflight_writes_complete_artifact(tmp_path: Path) -> None:
    """REQ-AGENTIC-6810-2 starts no process after a bounded lease failure."""

    calls: list[object] = []
    artifact = exp.run(
        result_path=tmp_path / "blocked.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        durable_artifact_paths=[],
        preflight_fn=lambda: _preflight(tmp_path, passed=False),
        worker_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        clock=iter((1_000_000_000, 2_500_000_000)).__next__,
    )

    assert calls == []
    assert artifact["status"] == "complete_blocked_arc_stepwise_strategy_accrual"
    assert artifact["honest_verdict"].startswith(
        "complete_blocked_arc_stepwise_strategy_accrual:"
    )
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_check": "task_owned_gpu_lease",
        "expected": "one unowned RTX 3090",
        "observed": {"owners": [991]},
    }
    assert artifact["stepwise_strategy_accrual_ready"] is False
    assert artifact["rows"] == []
    assert artifact["verdict_class"] == "blocked"
    assert exp.validate_artifact(artifact) == []

