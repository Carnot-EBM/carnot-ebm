"""REQ-REPORT-8001 and REQ-VERIFY-8001: qualify before reading outcomes."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest

from carnot.reporting import arc_supervisor_qualification as task
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from scripts.experiments import experiment_8001_v693_arc_supervisor_qualification as cli
from test_arc_supervisor_refinement_7936 import episode, producer


def live_episode() -> dict[str, Any]:
    """Protocol control binds the wrapper application to visible frame outcomes."""
    value = episode()
    value["live_agent_provenance"] = dict(
        policy_class="E3AgentPolicy", agent_factory="make_carnot_agent", execution_mode="live"
    )
    redirect = value["trajectory_supervisor"]["redirects"][0]
    redirect["action_index"] = 1
    frames = [dict(action_index=i, levels_completed=int(i == 4), frame=[[i]]) for i in range(1, 5)]
    value["run_receipt"] = dict(
        game=value["game"],
        seed=value["seed"],
        invocation_id=value["invocation_id"],
        entrypoint="CarnotAgent.choose_action",
        execution_mode="live",
        frames_sha256=canonical_hash(frames),
        applications=[dict(redirect_id="r1", arm=redirect["arm"], action_index=1)],
    )
    value["action_frames"] = frames
    return value


# SCENARIO-REPORT-8001-AUTHENTICATION: frame evidence, wrappers and fixture exclusions.
@pytest.mark.parametrize(
    "mutation",
    [
        "valid",
        "fixture",
        "missing",
        "identity",
        "hash",
        "gap",
        "empty_frame",
        "distance",
        "resolved",
        "application",
        "mode",
        "negative",
    ],
)
def test_authentication(mutation: str) -> None:
    value = live_episode()
    if mutation == "fixture":
        value["fixture_claim_scope"] = "circular_positive"
    elif mutation == "missing":
        value.pop("run_receipt")
    elif mutation == "identity":
        value["run_receipt"]["invocation_id"] = "other"
    elif mutation == "hash":
        value["run_receipt"]["frames_sha256"] = "sha256:wrong"
    elif mutation in {"gap", "empty_frame"}:
        value["action_frames"][1]["action_index" if mutation == "gap" else "frame"] = (
            9 if mutation == "gap" else []
        )
        value["run_receipt"]["frames_sha256"] = canonical_hash(value["action_frames"])
    elif mutation in {"distance", "resolved"}:
        value["trajectory_supervisor"]["redirects"][0][
            "actions_to_levelup" if mutation == "distance" else "resolved_by_levelup"
        ] = 9 if mutation == "distance" else False
    elif mutation == "application":
        value["run_receipt"]["applications"] = []
    elif mutation == "mode":
        value["run_receipt"]["entrypoint"] = "offline_proxy"
    elif mutation == "negative":
        for frame in value["action_frames"]:
            frame["levels_completed"] = 0
        value["run_receipt"]["frames_sha256"] = canonical_hash(value["action_frames"])
        value["trajectory_supervisor"]["redirects"][0].update(
            resolved_by_levelup=False, actions_to_levelup=None
        )
    assert bool(task.authenticate_episode(value)) == (mutation not in {"valid", "negative"})


# SCENARIO-REPORT-8001-AUTHENTICATION: content reuse never adds independent events.
def test_scan_and_controls(tmp_path: Path) -> None:
    value = live_episode()
    duplicate = deepcopy(value)
    duplicate.update(invocation_id="call2", receipt_id="receipt2")
    duplicate["run_receipt"]["invocation_id"] = "call2"
    fixture = deepcopy(value)
    fixture.update(invocation_id="fixture", receipt_id="fixture", synthetic=True)
    path = producer(tmp_path, [value, value, duplicate, fixture])
    checked = dict(prior={}, inventory={})
    result = task.scan(tmp_path, [path], checked, tmp_path / "private", current_date="20261002")
    assert result["identity_filter_count"] == 1
    assert {r.get("reason") for r in result["rows"]} >= {
        "duplicate_outcome_bytes",
        "synthetic_fixture",
        "identical_retry",
    }
    assert result["new_event_rows"][0]["frame_evidence_verified"] is True
    value.pop("run_receipt")
    path = producer(tmp_path, [value])
    rejected = task.scan(tmp_path, [path], checked, tmp_path / "again", current_date="20261002")
    assert rejected["identity_filter_count"] == 0
    assert rejected["rows"][0]["reason"] == "missing_live_wrapper_receipt"
    empty = task.scan(tmp_path, [], checked, tmp_path / "empty", current_date="20261002")
    assert empty["identity_filter_count"] == 0
    path = producer(tmp_path, [live_episode()])
    document = json.loads(path.read_text())
    document["verdict_class"] = "circular_positive"
    atomic_json(path, document)
    circular = task.scan(tmp_path, [path], checked, tmp_path / "circular", current_date="20261002")
    assert circular["identity_filter_count"] == 0
    assert circular["rows"][0]["reason"] == "synthetic_fixture"


# SCENARIO-REPORT-8001-QUALIFICATION: original shared script entrypoint is measured.
@pytest.mark.parametrize("script", [task.CLI, task.runner.CLI, task.history.CLI])
def test_actual_script_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str
) -> None:
    source = tmp_path / "producer.json"
    atomic_json(source, dict(run_date="20261002", verdict_class="null", source_artifact_hashes={}))
    output = tmp_path / "out.json"
    date = "20261002" if script == task.CLI else "20261001"
    args = [
        script,
        "--date",
        date,
        "--reduce-ledger",
        str(tmp_path),
        "--producer",
        str(source),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as good:
        runpy.run_path(script, run_name="__main__")
    assert good.value.code == 0
    assert json.loads(output.read_text())["fixture_claim_scope"] == "circular_positive"
    monkeypatch.setattr(sys, "argv", [script, "--cold-replay", str(output)])
    with pytest.raises(SystemExit) as replay:
        runpy.run_path(script, run_name="__main__")
    assert replay.value.code == 0


# SCENARIO-REPORT-8001-QUALIFICATION: private receipt errors retain asserted routes.
def test_cli_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "producer.json"
    output = tmp_path / "out.json"
    args = ["--reduce-ledger", str(tmp_path), "--producer", str(source), "--output", str(output)]
    assert cli.main(args) == 1
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    source.write_text("{broken")
    assert cli.main(args) == 1
    assert json.loads(output.read_text())["gate_check_summary"]
    with pytest.raises(SystemExit) as missing:
        cli.main(["--reduce-ledger", str(tmp_path)])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as date:
        cli.main(["--date", "20261001"])
    assert date.value.code == 2
    with pytest.raises(json.JSONDecodeError):
        cli.main(["--cold-replay", str(source)])
    with pytest.raises(FileNotFoundError):
        cli.main(["--cold-replay", str(tmp_path / "missing.json")])
    monkeypatch.setattr(task, "execute", lambda *_args: 0)
    assert cli.main([]) == 0


# SCENARIO-REPORT-8001-NULL: headroom stays a protocol control and cannot inflate N.
def test_fields_and_replay() -> None:
    value = dict(
        task.previous.reduce([]),
        rows=[],
        source_artifact_hashes={},
        scan_source_hashes={},
        new_firing_count=0,
        new_helped_count=0,
        live_path_receipts=[],
        verdict_class="null",
        honest_verdict="complete_null_no_new_supervisor_outcomes",
        validation_receipts=[dict(name="coverage_report", classification="required", passed=True)],
    )
    value.update(task.artifact_fields(value, {}))
    assert value["honest_verdict"] == "complete_null_no_new_outcomes"
    assert value["no_new_outcomes"] is True
    assert value["positive_control_results"]["passed"] is True
    assert value["proposed_generalization_refinement"] is None
    assert not task.replay(value)
    value["no_new_outcomes"] = False
    assert "no_new_outcomes" in task.replay(value)
    value["no_new_outcomes"] = True
    value["verdict_class"] = "disqualified"
    assert (
        task.artifact_fields(value, {})["retire_if_same_qualification_failure"]["retired"] is True
    )
    value["verdict_class"] = "blocked"
    assert task.artifact_fields(value, {})["honest_verdict"].startswith("complete_blocked")


# SCENARIO-REPORT-8001-QUALIFICATION: owned commands freeze E2E and old denominator.
def test_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    specs = task.commands(tmp_path)
    assert any(s["name"] == "e2e_017" for s in specs)
    assert set(task.history.ADDED) <= set(task.ADDED)
    assert any("--fail-under=100" in s["argv"] for s in specs)
    assert task.QUALIFY_BEFORE_SCAN is True
    assert not any(s["name"].startswith("e2e_016") for s in specs)
    assert any(s["name"] == "full_python_suite" for s in specs)
    checked = task.inputs()
    assert not checked["failures"]
    assert checked["owned_repository_health"][0]["exit_code"] != 0
    assert checked["historical_coverage_failure"]["covered_statements"] == 268
    assert checked["historical_coverage_failure"]["statements"] == 269
    assert len(checked["original_source_snapshots"]) == 4
    assert all(Path(p).is_file() for p in checked["original_source_snapshots"])
    assert checked["prior"]["historical_required_failures"][-1]["verdict_class"] == "disqualified"
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(task, "RECENT_HISTORY", missing)
    monkeypatch.setitem(task.PINNED, str(missing), "sha256:" + "0" * 64)
    assert any(r["observed"] == "missing" for r in task.inputs()["failures"])


# SCENARIO-VERIFY-8001-READINESS: qualification precedes scanning on real runner paths.
@pytest.mark.parametrize("passed", [True, False])
def test_qualification_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passed: bool) -> None:
    from test_arc_supervisor_delta_7962 import authority

    authority(tmp_path, monkeypatch)
    events: list[str] = []
    scope = task.runner
    monkeypatch.setattr(scope, "QUALIFY_BEFORE_SCAN", True, raising=False)
    monkeypatch.setattr(scope, "FREEZE_SOURCES", True, raising=False)
    monkeypatch.setattr(scope.validation, "dependency_hashes", lambda *_args, **_kwargs: {})
    for label in scope.ADDED:
        path = tmp_path / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((task.ROOT / label).read_bytes())
    original = scope.previous.scan

    def scan(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("scan")
        assert events[0] == "qualify"
        assert bool(args[1]) is passed
        return original(*args, **kwargs)

    def commands(private: Path) -> list[dict[str, Any]]:
        atomic_json(
            private / "coverage.json",
            dict(
                files={
                    n: dict(summary=dict(num_statements=1, covered_lines=1), missing_lines=[])
                    for n in scope.ADDED
                }
            ),
        )
        return [dict(name="owned", classification="required", argv=[], expected_exit=0)]

    def run(spec: dict[str, Any], *_args: Any) -> dict[str, Any]:
        events.append("qualify")
        return dict(
            spec,
            passed=passed,
            command_argv=[],
            log_path=str(tmp_path / "check.log"),
            log_sha256="sha256:x",
            exit_code=int(not passed),
        )

    monkeypatch.setattr(scope, "commands", commands)
    monkeypatch.setattr(scope.previous, "run", run)
    monkeypatch.setattr(scope.previous, "scan", scan)
    monkeypatch.setattr(scope.previous, "terminal", lambda *_args: dict(passed=True, reports=[]))
    atomic_json(tmp_path / "results/experiment_7900_arc.json", dict(source_artifact_hashes={}))
    output = tmp_path / "results" / scope.OUTPUT.name
    assert scope.execute(output, tmp_path / "private") == int(not passed)
    assert json.loads(output.read_text())["arc_evidence_ready_score"] == int(passed)
    assert json.loads(output.read_text())["phase_spans"][0]["phase"] == "supervisor_qualification"
    assert (output.parent / "raw" / output.stem / "source_snapshots" / scope.CLI).is_file()


# SCENARIO-REPORT-8001-SEAL: delegation and cold child preserve exact current scope.
def test_execute_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        task.runner, "execute", lambda output, private, scope: int(scope.EXPERIMENT_ID != 8001)
    )
    assert task.execute(tmp_path / task.OUTPUT.name, tmp_path) == 0
    monkeypatch.setattr(task.previous, "terminal", lambda *_args: dict(passed=True, reports=[]))

    def run(spec: dict[str, Any], *_args: Any) -> dict[str, Any]:
        assert str(task.ROOT / task.CLI) in spec["argv"]
        return dict(name=spec["name"], passed=True)

    monkeypatch.setattr(task.previous, "run", run)
    atomic_json(tmp_path / "candidate.json", dict(task.previous.reduce([]), rows=[]))
    assert task.terminal(tmp_path / "candidate.json", tmp_path, tmp_path)["passed"]


# SCENARIO-REPORT-8001-AUTHENTICATION: actual private CLI applies the new evidence gate.
def test_cli_frame_gate(tmp_path: Path) -> None:
    path = producer(tmp_path, [live_episode()])
    output = tmp_path / "private.json"
    args = ["--reduce-ledger", str(tmp_path), "--producer", str(path), "--output", str(output)]
    assert cli.main(args) == 0
    value = json.loads(output.read_text())
    assert value["identity_filter_count"] == 1
    assert value["new_event_rows"][0]["frame_evidence_verified"] is True
    assert cli.main(["--cold-replay", str(output)]) == 0
    value["identity_filter_count"] = 2
    atomic_json(output, value)
    assert cli.main(["--cold-replay", str(output)]) == 1
    fixture = live_episode()
    fixture["synthetic"] = True
    path = producer(tmp_path, [fixture])
    assert cli.main(args) == 0
    assert json.loads(output.read_text())["identity_filter_count"] == 0


# SCENARIO-REPORT-8001-AUTHENTICATION: wrapper checks share the same scan budget.
def test_scan_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = producer(tmp_path, [live_episode()])
    checked = dict(prior={}, inventory={})
    baseline = task.previous.scan(
        tmp_path, [path], checked, tmp_path / "first", current_date="20261002"
    )
    monkeypatch.setattr(task.previous, "scan", lambda *_args, **_kwargs: baseline)
    times = iter([0.0, 121.0, 121.0])
    monkeypatch.setattr(task.time, "monotonic", lambda: next(times, 121.0))
    value = task.scan(tmp_path, [path], checked, tmp_path / "deadline", current_date="20261002")
    assert value["identity_filter_count"] == 0
    assert value["scan_failures"][0]["artifact_field"] == "scan_deadline_s"
    assert value["rows"][0]["reason"] == "scan_deadline"
