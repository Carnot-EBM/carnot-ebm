"""REQ-REPORT-7911: reuse authenticated cutoff bytes without fresh science."""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7911_v686_arc_supervisor_delta as cli


def authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use immutable private authorities so tests cannot rewrite research results."""
    prior = json.loads(cli.PRIOR.read_text())
    inventory = tmp_path / "inventory.json"
    atomic_json(inventory, {"baseline": {}, "current": {}, "producer_paths": []})
    prior["receipt_inventory_path"] = str(inventory)
    prior["source_artifact_hashes"] = {str(inventory): sha256_file(inventory)}
    prior["cutoff_receipt_hashes"] = {}
    path = tmp_path / "prior.json"
    atomic_json(path, prior)
    monkeypatch.setattr(cli, "PRIOR", path)
    monkeypatch.setattr(cli, "EXPECTED_PRIOR", sha256_file(path))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    return path


# SCENARIO-REPORT-7911-CUTOFF: all gate operands survive failed custody.
def test_authenticated_empty_and_missing_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = authority(tmp_path, monkeypatch)
    prior, checks, failures = cli.precheck()
    assert failures == [] and checks
    delta = cli.audit(prior)
    assert delta["new_outcome_count"] == 0 and delta["null_fast_path"]
    assert delta["new_level_solves"] == 0
    path.write_text("{}")
    assert cli.precheck()[2][0]["artifact_field"] == "sha256"
    path.unlink()
    assert cli.precheck()[2][0]["observed"] == "missing"


# SCENARIO-REPORT-7911-CUTOFF: fresh producer bytes delegate to the qualified reducer.
def test_new_producer_and_cutoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority(tmp_path, monkeypatch)
    producer = tmp_path / "results/experiment_9001_arc.json"
    atomic_json(producer, {"source_artifact_hashes": {}, "verdict_class": "null"})
    prior = cli.precheck()[0]
    prior["outcome_rows"] = [
        {"event_id": "old", "content_sha256": "sha256:old", "source_sha256": "sha256:raw"}
    ]
    delta = cli.audit(prior)
    assert delta["null_fast_path"] is False and delta["new_outcome_count"] == 0
    assert delta["cutoff_receipt_hashes"]["old"] == "sha256:old"
    assert delta["cutoff_receipt_hashes"]["raw:sha256:raw"] == "sha256:raw"


# SCENARIO-REPORT-7911-VALIDATION: real private routes and forged replay fail honestly.
def test_private_modes(tmp_path: Path) -> None:
    producer = tmp_path / "producer.json"
    atomic_json(producer, {"source_artifact_hashes": {}, "verdict_class": "null"})
    output = tmp_path / "delta.json"
    args = ["--date", "20260930", "--reduce-ledger", str(tmp_path), "--producer", str(producer)]
    assert cli.main([*args, "--output", str(output)]) == 0
    assert cli.main(["--cold-replay", str(output)]) == 0
    data = json.loads(output.read_text())
    data["firings"] = 1
    atomic_json(output, data)
    assert cli.main(["--cold-replay", str(output)]) == 1
    with pytest.raises(SystemExit):
        cli.main(args)


# SCENARIO-REPORT-7911-VALIDATION: the manifest freezes dates, private paths and coverage.
def test_manifest(tmp_path: Path) -> None:
    rows = cli.commands(tmp_path)
    assert all(row["deadline_s"] > 0 and row["expected_exit"] in {0, 2} for row in rows)
    assert len([row for row in rows if "--include=" + cli.INCLUDE in row["argv"]]) == 5
    for row in rows:
        if row["name"].startswith("e2e_016"):
            assert row["argv"][row["argv"].index("--date") + 1] == "20260930"
    assert any(row["name"] == "full_python_suite" for row in rows)


# SCENARIO-REPORT-7911-VALIDATION: null, external block and owned failures stay distinct.
@pytest.mark.parametrize("state", ["null", "blocked", "required_failure", "terminal_failure"])
def test_terminal_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    authority(tmp_path, monkeypatch)
    if state == "blocked":
        cli.PRIOR.unlink()
    monkeypatch.setattr(cli, "DURABLE", tmp_path / "durable")
    monkeypatch.setattr(
        cli,
        "commands",
        lambda _: [
            {"name": "unit", "argv": ["true"], "classification": "required", "expected_exit": 0}
        ],
    )
    calls = 0

    def run(row: dict, _private: Path, _started: float) -> dict:
        nonlocal calls
        calls += 1
        passed = not (state == "required_failure" and row["name"] == "unit")
        if state == "terminal_failure" and row["name"] == "terminal_adversarial" and calls == 2:
            passed = False
        return {
            "name": row["name"],
            "command_argv": row["argv"],
            "passed": passed,
            "exit_code": int(not passed),
            "classification": row["classification"],
            "expected_exit": 0,
        }

    monkeypatch.setattr(cli, "_run", run)
    output = tmp_path / "output.json"
    rc = cli.main(["--date", "20260930", "--output", str(output)])
    data = json.loads(output.read_text())
    expected = "disqualified" if "failure" in state else state
    assert data["verdict_class"] == expected
    assert rc == int(state != "null")
    assert data["new_live_outcome_count"] == 0 and data["MODEL_SPECS"] == []
    assert data["duration_s"] > 0 and data["run_date"] == "20260930"
    assert data["field_principles"]
    assert data["historical_required_failures"] or state == "blocked"
    assert sha256_file(output) == sha256_file(tmp_path / "durable/terminal_candidate.json")
    cli.progress(time.monotonic(), "verified", 1)
