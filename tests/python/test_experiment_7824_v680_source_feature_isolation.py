"""Current source isolation checks for REQ-REPORT-7824."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_7824_v680_source_feature_isolation as task
from carnot.verify.evidence_views import prepare_views, serialize_pair


def fixture_row() -> dict:
    pair = serialize_pair(prepare_views(b"Alpha fact. Beta fact.", b"Alpha fact."))
    return {
        "family_id": "sha256:fixture",
        "role": "fit",
        "label": 1,
        "confidence": 0.99,
        "generator_identity": "fixture model",
        "error_annotations": [{"wrong": True}],
        "view_a": pair["a"],
        "view_b": pair["b"],
    }


def test_scenario_report_7824_isolation() -> None:
    """SCENARIO-REPORT-7824-ISOLATION: metadata cannot change tensors."""
    original = fixture_row()
    public = task.project(original)
    baseline = task.extract(public)
    assert baseline["feature_dim"] == 132
    tensors = {key: value for key, value in baseline.items() if key != "family_id"}
    for name in task.FORBIDDEN:
        changed = dict(original, **{name: "mutated"})
        assert {
            k: v for k, v in task.extract(task.project(changed)).items() if k != "family_id"
        } == tensors
        changed.pop(name)
        assert {
            k: v for k, v in task.extract(task.project(changed)).items() if k != "family_id"
        } == tensors


def test_scenario_report_7824_public_type_guard() -> None:
    """SCENARIO-REPORT-7824-ISOLATION: public types fail before extraction."""
    public = task.project(fixture_row())
    public["view_a"]["source_bytes"] = 42
    with pytest.raises((TypeError, ValueError)):
        task.extract(public)
    with pytest.raises((TypeError, ValueError)):
        task.extract(dict(public, confidence=0.99))


def test_scenario_report_7824_mask() -> None:
    """SCENARIO-REPORT-7824-MASK: unknown labels have no local gradient."""
    logits = [0.7, -0.4]
    zero = task.masked_local_loss(logits, [0, 1], [1, 0])
    one = task.masked_local_loss(logits, [0, 0], [1, 0])
    assert zero == one
    assert zero[1][1] == 0
    flipped = task.masked_local_loss(logits, [1, 1], [1, 0])
    assert flipped[0] != zero[0]
    assert zero[1][0] != 0


def test_scenario_report_7824_manifest_and_dispatch(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7824-TERMINAL: exact command roster and bytes survive replay."""
    manifest = task.load_command_manifest()
    assert manifest["commands"][-1]["classification"] == "diagnostic"
    with pytest.raises(ValueError, match="manifest"):
        task.validate_command_manifest(dict(manifest, commands=[]))
    log = tmp_path / "sealed.log"
    log.write_bytes(b"passed\n")
    calls = []

    def record(command: dict, index: int, scope: dict) -> dict:
        calls.append((command["name"], command["argv"], command["classification"]))
        return {
            "name": command["name"],
            "command_argv": command["argv"],
            "classification": command["classification"],
            "log_path": str(log),
            "log_sha256": task.sha256_file(log),
            "passed": True,
            "exit_code": 0,
        }

    task.dispatch(manifest, record)
    assert calls == [(x["name"], x["argv"], x["classification"]) for x in manifest["commands"]]
    log.write_bytes(b"mutated\n")
    with pytest.raises(ValueError, match="log_drift"):
        task.validate_log_receipt({"log_path": str(log), "log_sha256": "sha256:wrong"})


def test_scenario_report_7824_cold_replay_rejects_changed_public(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7824-TERMINAL: public byte mutation breaks a sealed manifest."""
    row = task.project(fixture_row())
    path = tmp_path / "public.jsonl"
    path.write_text(json.dumps(row) + "\n")
    manifest = {"public_path": str(path), "public_sha256": task.sha256_file(path)}
    task.check_public_manifest(manifest)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="public_hash"):
        task.check_public_manifest(manifest)
