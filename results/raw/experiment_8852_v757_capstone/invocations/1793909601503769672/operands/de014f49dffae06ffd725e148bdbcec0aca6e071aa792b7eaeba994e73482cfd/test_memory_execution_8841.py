"""REQ-REPORT-8841: qualify execution without changing historical authorities."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import memory_execution_8841 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_initialization() -> None:
    """SCENARIO-REPORT-8841-CONTRACT: recipes reconstruct exact in-process heads."""
    recipes = r.initializations()
    assert len(recipes) == 15
    for row in recipes:
        assert row["parameters"] == r.old.c.head(row["seed"], row["objective"])
        assert row["sha256"] == canonical_hash(row["parameters"])
    assert {row["objective"] for row in recipes} == {"mixed", "categorical", "linear"}


def test_original_missing_paths_real_children(tmp_path: Path) -> None:
    """REQ-REPORT-8841: real children execute original81/290 with no bypass."""
    rows = r.rejections(tmp_path)
    assert [row["diagnostic"] for row in rows] == [
        "objective_candidate_ready_score:observed_zero",
        "rejected_producer",
        "validation_log",
    ]
    assert all(row["passed"] and row["unchanged"] for row in rows)
    assert [row["original_statement"] for row in rows] == [77, 81, 290]
    for row in rows:
        assert r.old.bound(row["response_ref"])["diagnostic"] == row["diagnostic"]


def test_probe_rejects_unexpected(tmp_path: Path) -> None:
    """REQ-REPORT-8841: a missing expected rejection is an internal failure."""
    request = dict(mode="zero", fixture=str(tmp_path / "fixture.json"), raw=str(tmp_path))
    atomic_json(Path(request["fixture"]), {})
    with patch.object(r.old, "preflight", return_value=({}, {}, {})):
        with pytest.raises(RuntimeError, match="expected_rejection_missing"):
            r.probe(request)
    with patch.object(r.old, "preflight", side_effect=PermissionError("wrong")):
        with pytest.raises(RuntimeError, match="rejection_drift"):
            r.probe(request)
    with patch.object(r.old.validation, "commands", return_value=[dict(passed=False)]):
        with pytest.raises(RuntimeError, match="rejection_child"):
            r.rejections(tmp_path / "failed")


def test_preconditions(tmp_path: Path) -> None:
    """REQ-REPORT-8841: authenticate original raw, source, config and method bytes."""
    historical, evidence, pre = r.preflight(tmp_path)
    assert historical["memory_execution_ready_score"] == 0
    assert len(evidence["fits"]) == 45
    assert pre["feedback_delay"] == 8
    assert pre["time_budget_s"] == 4200
    assert pre["configuration"]["lr"] == 0.1


def test_validation_adapters(tmp_path: Path) -> None:
    """REQ-REPORT-8841: reuse the original owned/E2E/health command manifest."""
    with patch.object(r.old.validation, "validate", return_value=dict(passed=True)):
        assert r.validate(tmp_path)["passed"]
    with patch.object(r.old.validation, "terminal", return_value=dict(passed=True)):
        assert r.terminal(tmp_path / "candidate.json", tmp_path)["passed"]
    with patch.object(r.old.validation, "commands", return_value=[dict(passed=True)]):
        assert r.upstream_replay(tmp_path)["passed"]


def test_replay_guards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8841-CONTRACT: replay rederives recipes and rejection probes."""
    value = dict(
        raw_shard_hashes=[],
        validation_receipts=[],
        code_config_hashes=[],
        source_artifact_hashes=[],
        initialization_recipes=r.initializations(),
        configuration_hashes=dict(
            configuration=canonical_hash(None), initialization=canonical_hash(r.initializations())
        ),
        preconditions_checked={},
        original_missing_statement_rows=[],
    )
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    with patch.object(r.old, "replay", return_value=dict(passed=True)):
        with patch.object(r, "rejections", return_value=[]):
            assert r.replay(path)["passed"]
            operand = tmp_path / "operand.json"
            atomic_json(operand, {})
            shard = tmp_path / "shard.json"
            atomic_json(shard, dict(operand=r.old.reference(operand)))
            atomic_json(path, dict(value, raw_shard_hashes=[r.old.reference(shard)]))
            assert r.replay(path)["passed"]
            atomic_json(path, dict(value, configuration_hashes={}))
            with pytest.raises(ValueError, match="configuration_drift"):
                r.replay(path)
        atomic_json(path, dict(value, initialization_recipes=[]))
        with pytest.raises(ValueError, match="initialization_drift"):
            r.replay(path)
        atomic_json(path, value)
        with patch.object(r, "rejections", return_value=[dict(diagnostic="wrong")]):
            with pytest.raises(ValueError, match="rejection_drift"):
                r.replay(path)


def test_main_contract(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-8841: internal failures disqualify; external absence blocks."""
    historical = json.loads(r.HISTORICAL.read_bytes())
    work = r.old.bound(historical["raw_shard_hashes"][0])
    pre = historical["preconditions_checked"]
    monkeypatch.setattr(r, "preflight", lambda raw: (deepcopy(historical), {}, pre))
    monkeypatch.setattr(r.old, "execute", lambda *a: deepcopy(work))
    monkeypatch.setattr(r.old, "restart", lambda raw: work["restart_rows"])
    monkeypatch.setattr(r, "rejections", lambda raw: [])
    monkeypatch.setattr(r, "upstream_replay", lambda raw: dict(passed=True, duration_s=1))
    checked = dict(passed=True, receipts=[], coverage=dict(complete=True, percent=100))
    monkeypatch.setattr(r, "validate", lambda raw: deepcopy(checked))
    monkeypatch.setattr(r, "terminal", lambda *a: dict(passed=True, receipts=[]))
    output = tmp_path / (r.NAME + ".json")
    args = ["--private-output", str(output), "--raw-dir", str(tmp_path / "raw")]
    assert r.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["experiment_id"] == 8841 and value["memory_execution_ready_score"] == 1
    assert value["inference_substrate_class"] == "no_model_load"
    assert value["execution_venue"] == "host"
    assert r.main(["--date", "20000101"]) == 1
    for mode in ("owned", "terminal", "upstream", "control", "missing", "internal"):
        checked["passed"] = mode != "owned"
        monkeypatch.setattr(r, "terminal", lambda *a: dict(passed=mode != "terminal", receipts=[]))
        monkeypatch.setattr(
            r, "upstream_replay", lambda raw: dict(passed=mode != "upstream", duration_s=1)
        )
        monkeypatch.setattr(
            r.old, "execute", lambda *a: dict(work, controls=[dict(passed=mode != "control")])
        )
        if mode in {"missing", "internal"}:

            def fail(raw: Path) -> Any:
                raise FileNotFoundError("absent") if mode == "missing" else RuntimeError("broken")

            monkeypatch.setattr(r, "preflight", fail)
        assert r.main(args) == 0
        rejected = json.loads(output.read_bytes())
        assert rejected["memory_execution_ready_score"] == 0
        assert rejected["verdict_class"] == ("blocked" if mode == "missing" else "disqualified")
    with patch.object(r, "replay", return_value=dict(passed=True)):
        assert r.main(["--replay", str(output)]) == 0
    assert r.main(["--replay", str(tmp_path / "missing")]) == 1
