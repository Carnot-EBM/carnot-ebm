"""REQ-REPORT-7949: empty live ledgers terminate without invented work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_supervisor_v689_delta as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7949_v689_arc_supervisor_delta as cli


def authorities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Private pinned bytes let failures be tested without changing authorities."""
    root = tmp_path
    (root / "results").mkdir()
    inventory = root / "inventory.json"
    seen = {"raw:sha256:old": "sha256:old"}
    atomic_json(inventory, dict(receipt_inventory=seen, seen_receipt_hashes=seen))
    prior = root / "prior.json"
    atomic_json(
        prior,
        dict(
            verdict_class="null",
            flagged_adversarial=False,
            arc_evidence_ready_score=1,
            receipt_inventory=seen,
            seen_receipt_hashes=seen,
            source_artifact_hashes={},
            historical_required_failures=[{"name": "old_failure"}],
        ),
    )
    registry = root / "registry.yaml"
    registry.write_text("games:\n  g1:\n    levels_reproduced: 2\n")
    monkeypatch.setattr(task, "ROOT", root)
    monkeypatch.setattr(task, "PRIOR", prior)
    monkeypatch.setattr(task, "INVENTORY", inventory)
    monkeypatch.setattr(task, "REGISTRY", registry)
    monkeypatch.setattr(
        task, "PINNED", {str(p): sha256_file(p) for p in (prior, inventory, registry)}
    )


# SCENARIO-REPORT-7949-1: authenticate actual operands and reject changed authorities.
def test_authenticate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authorities(tmp_path, monkeypatch)
    checked = task.inputs()
    assert not checked["failures"] and checked["registry_precheck"] == {"g1": 2}
    value = json.loads(task.PRIOR.read_text())
    value["arc_evidence_ready_score"] = 0
    atomic_json(task.PRIOR, value)
    task.PINNED[str(task.PRIOR)] = sha256_file(task.PRIOR)
    assert task.inputs()["failures"][0]["artifact_field"] == "arc_evidence_ready_score"
    value["arc_evidence_ready_score"] = 1
    value["seen_receipt_hashes"] = {}
    atomic_json(task.PRIOR, value)
    task.PINNED[str(task.PRIOR)] = sha256_file(task.PRIOR)
    assert task.inputs()["failures"]
    task.INVENTORY.write_text("{}")
    assert task.inputs()["failures"]
    task.REGISTRY.unlink()
    assert any(r["observed"] == "missing" for r in task.inputs()["failures"])


# SCENARIO-REPORT-7949-2: replay rejects every derived claim that lacks primitive support.
def test_private_cli(tmp_path: Path) -> None:
    root = tmp_path / "success"
    root.mkdir()
    producer = root / "producer.json"
    atomic_json(
        producer, dict(run_date="20260930", verdict_class="null", source_artifact_hashes={})
    )
    output = root / task.OUTPUT.name
    args = ["--reduce-ledger", str(root), "--producer", str(producer), "--output", str(output)]
    assert cli.main(args) == 0
    assert cli.main(["--cold-replay", str(output)]) == 0
    assert cli.main(["--terminal-recheck", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["fixture_claim_scope"] == "circular_positive"
    value["identity_filter_count"] = 1
    atomic_json(output, value)
    assert cli.main(["--cold-replay", str(output)]) == 1
    with pytest.raises(SystemExit):
        cli.main(["--reduce-ledger", str(root)])
    with pytest.raises(SystemExit):
        cli.main(["--date", "20260929"])
    assert task.replay(dict(rows=[], new_event_rows=[{}]))


# SCENARIO-REPORT-7949-1: same-day unseen content is new; fixtures cannot supply live evidence.
def test_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authorities(tmp_path, monkeypatch)
    checked = task.inputs()
    private = tmp_path / "scratch"
    assert task.scan(tmp_path, [], checked, private)["identity_filter_count"] == 0
    raw = tmp_path / "results/raw/live/rows.json"
    episode = dict(
        game="g1",
        seed=1,
        invocation_id="call",
        receipt_id="receipt",
        solve_provenance="live_agent_self_discovery",
        event_sequence=2,
        sequence_scope="run",
        live_agent_provenance=dict(
            policy_class="E3AgentPolicy", agent_factory="make_carnot_agent", execution_mode="live"
        ),
        trajectory_supervisor=dict(
            enabled=True,
            mode="applied",
            redirects=[
                dict(
                    id="r",
                    arm="drop_goal_bias",
                    fired=True,
                    resolved_by_levelup=True,
                    actions_to_levelup=3,
                )
            ],
        ),
    )
    producer = tmp_path / "results/experiment_7948_arc.json"

    def write() -> None:
        atomic_json(raw, {"rows": [episode]})
        atomic_json(
            producer,
            dict(
                run_date="20260930",
                verdict_class="null",
                source_artifact_hashes={str(raw): sha256_file(raw)},
            ),
        )

    write()
    delta = task.scan(tmp_path, [producer], checked, private)
    assert delta["identity_filter_count"] == 1 and not task.replay(delta)
    assert delta["new_event_rows"][0]["actions_to_levelup"] == 3
    assert delta["refinement_decisions"][0]["decision"] == "unchanged"
    checked["prior"]["source_artifact_hashes"] = {str(producer): sha256_file(producer)}
    assert task.scan(tmp_path, [producer], checked, private)["identity_filter_count"] == 0
    checked["prior"]["source_artifact_hashes"] = {}
    episode["live_agent_provenance"]["execution_mode"] = "offline"
    write()
    assert task.scan(tmp_path, [producer], checked, private)["identity_filter_count"] == 0
    raw.unlink()
    assert task.scan(tmp_path, [producer], checked, private)["scan_failures"]
    monkeypatch.setattr(task, "SCAN_CAP_S", -1)
    assert task.scan(tmp_path, [producer], checked, private)["scan_failures"]


# SCENARIO-REPORT-7949-3: freeze private routes, historical dates and identical includes.
def test_commands(tmp_path: Path) -> None:
    specs = task.commands(tmp_path)
    assert all(s["deadline_s"] > 0 for s in specs)
    includes = {a for s in specs for a in s["argv"] if a.startswith("--include=")}
    assert includes == {"--include=" + task.INCLUDE}
    assert any(s["classification"] == "repository_health" for s in specs)
    for spec in specs:
        if spec["name"].startswith("e2e_016"):
            assert spec["argv"][spec["argv"].index("--date") + 1] == "20260929"


# SCENARIO-REPORT-7949-3: validation gates preserve failures and bind actual reader bytes.
@pytest.mark.parametrize("state", ["null", "blocked", "failure", "coverage", "new"])
def test_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    authorities(tmp_path, monkeypatch)
    private = tmp_path / "private"
    private.mkdir()
    output = tmp_path / "results" / task.OUTPUT.name

    def commands(directory: Path) -> list[dict[str, Any]]:
        atomic_json(
            directory / "coverage.json",
            {
                "files": {
                    name: {
                        "summary": {"num_statements": 1, "covered_lines": 1},
                        "missing_lines": [],
                    }
                    for name in task.ADDED
                }
            },
        )
        return [
            dict(
                name="unit",
                argv=["true"],
                expected_exit=0,
                expected_text=None,
                deadline_s=5,
                classification="required",
            )
        ]

    def run(spec: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        passed = state != "failure" or spec["name"] != "unit"
        return dict(
            spec,
            passed=passed,
            exit_code=int(not passed),
            command_argv=spec["argv"],
            output_tail="",
            log_path="private.log",
            log_sha256="sha256:private",
        )

    monkeypatch.setattr(task, "commands", commands)
    monkeypatch.setattr(task, "run", run)
    monkeypatch.setattr(task.validation, "dependency_hashes", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        task, "sha256_file", lambda p: sha256_file(p) if p.is_file() else "sha256:private"
    )
    if state == "blocked":
        task.INVENTORY.unlink()
    if state == "coverage":
        monkeypatch.setattr(task.validation, "coverage_complete", lambda *_args, **_kwargs: False)
    if state == "new":
        original = task.scan

        def changed(*args: Any) -> dict[str, Any]:
            value = original(*args)
            value["identity_filter_count"] = 1
            return value

        monkeypatch.setattr(task, "scan", changed)
    result = task.execute(output, private)
    value = json.loads(output.read_text())
    expected = "disqualified" if state in {"failure", "coverage", "new"} else state
    assert value["verdict_class"] == expected
    assert result == int(expected != "null")
    assert value["experiment_id"] == 7949 and value["milestone"] == "2026.09.689"
    assert value["model_invocation_counts"] == 0 and value["new_level_solves_claimed"] == 0
    assert value["solve_provenance"] == [] and value["historical_required_failures"]
    receipt = json.loads(
        (output.parent / "raw" / output.stem / "primary_resolution_receipt.json").read_text()
    )
    assert receipt["gate_sha256"] == sha256_file(output) and receipt["passed"]
    monkeypatch.setattr(cli.task, "execute", lambda *_args: 0)
    assert cli.main(["--output", str(output)]) == 0


# SCENARIO-REPORT-7949-3: a wrong failure reason cannot qualify an expected exit.
def test_reason_and_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        task.validation, "run_check", lambda *_args: dict(passed=True, output_tail="wrong")
    )
    assert not task.run(dict(expected_text="required"), tmp_path, tmp_path)["passed"]
    assert task.run(dict(expected_text=None), tmp_path, tmp_path)["passed"]
    monkeypatch.setattr(task, "run", lambda *_args: dict(passed=False))
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, dict(rows=[]))
    assert task.terminal(candidate, tmp_path, tmp_path)["passed"] is False


# SCENARIO-REPORT-7949-3: rejected final bytes are rechecked; uniqueness remains required.
@pytest.mark.parametrize("conflict", [False, True])
def test_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conflict: bool
) -> None:
    authorities(tmp_path, monkeypatch)
    private = tmp_path / "private"
    private.mkdir()
    output = tmp_path / "results" / task.OUTPUT.name

    def commands(directory: Path) -> list[dict[str, Any]]:
        atomic_json(
            directory / "coverage.json",
            {
                "files": {
                    name: {
                        "summary": {"num_statements": 1, "covered_lines": 1},
                        "missing_lines": [],
                    }
                    for name in task.ADDED
                }
            },
        )
        return []

    calls = 0

    def run(spec: dict[str, Any], *_args: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return dict(spec, passed=calls > 2, log_path="private.log", log_sha256="sha256:private")

    monkeypatch.setattr(task, "commands", commands)
    monkeypatch.setattr(task, "run", run)
    monkeypatch.setattr(task.validation, "dependency_hashes", lambda *_args, **_kwargs: {})
    if conflict:
        atomic_json(output.parent / "experiment_7949_conflict.json", {})
        with pytest.raises(ValueError, match="conflicting_primary"):
            task.execute(output, private)
    else:
        assert task.execute(output, private) == 1
        value = json.loads(output.read_text())
        assert value["verdict_class"] == "disqualified" and value["arc_evidence_ready_score"] == 0
        assert calls == 4 and value["flagged_adversarial"] is False


# SCENARIO-REPORT-7949-1: cached, malformed and missing candidate bytes fail without retries.
def test_scan_custody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authorities(tmp_path, monkeypatch)
    checked = task.inputs()
    private = tmp_path / "scratch"
    producer = tmp_path / "results/experiment_7948_arc.json"
    raw = tmp_path / "results/raw/live.json"
    atomic_json(raw, {"rows": []})
    digest = sha256_file(raw)
    atomic_json(
        producer,
        dict(
            run_date="20260930",
            verdict_class="null",
            source_artifact_hashes={"code.py": "sha256:ignored", str(raw): {"sha256": digest}},
        ),
    )
    checked["inventory"]["receipt_inventory"]["raw:" + digest] = digest
    assert not task.scan(tmp_path, [producer], checked, private)["rows"]
    checked["inventory"]["receipt_inventory"] = {}
    assert not task.scan(tmp_path, [producer, producer], checked, private)["scan_failures"]
    raw.write_text("invalid json")
    value = json.loads(producer.read_text())
    value["source_artifact_hashes"] = {str(raw): sha256_file(raw)}
    atomic_json(producer, value)
    assert task.scan(tmp_path, [producer], checked, private)["scan_failures"]
    producer.write_text("invalid json")
    assert task.scan(tmp_path, [producer], checked, private)["scan_failures"]
    atomic_json(producer, {"source_artifact_hashes": []})
    assert task.scan(tmp_path, [producer], checked, private)["scan_failures"]
    producer.unlink()
    assert task.scan(tmp_path, [producer], checked, private)["scan_failures"]


# SCENARIO-REPORT-7949-1: current registry list rows keep their banked level operands.
def test_current_registry_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authorities(tmp_path, monkeypatch)
    task.REGISTRY.write_text("games:\n- game: g1\n  levels_reproduced: 2\n")
    task.PINNED[str(task.REGISTRY)] = sha256_file(task.REGISTRY)
    assert task.inputs()["registry_precheck"] == {"g1": 2}
