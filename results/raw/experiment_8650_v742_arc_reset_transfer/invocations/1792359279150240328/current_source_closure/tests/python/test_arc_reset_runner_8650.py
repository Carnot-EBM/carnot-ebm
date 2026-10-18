"""REQ-VERIFY-8650: private children exercise rejection and publication paths."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import arc_reset_transfer_8650 as e
from carnot.reporting import arc_reset_runner_8650 as runner
from carnot.reporting.current_work_receipt import atomic_json


def test_private_cli_actual_children(tmp_path):
    """SCENARIO-VERIFY-8650-REPLAY: execute outside the checkout without PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)

    def run(*args):
        return subprocess.run(
            [sys.executable, str(e.ROOT / e.CLI), *map(str, args)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

    assert run("--date", "invalid").returncode == 2
    assert run("--cold-replay", tmp_path / "missing").returncode == 1
    assert run("--private-control", "--root", tmp_path).returncode == 2
    for failed in (False, True):
        output = tmp_path / str(failed) / (e.NAME + ".json")
        args = ["--private-control", "--root", tmp_path / "absent", "--output", output]
        if failed:
            args.append("--owned-failure")
        result = run(*args)
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert run("--cold-replay", output).returncode == 0
        value["completed_count"] = 999
        atomic_json(output, value)
        assert run("--cold-replay", output).returncode == 1


def test_official_episode_entry_and_public_reset_hashes(tmp_path, monkeypatch):
    """REQ-REPORT-8650: SDK entry preserves actual reset observations and head counts."""
    from carnot.reporting import arc_reset_live_8649 as live
    from carnot.reporting import arc_supervisor_live_panel_8384 as sdk
    from carnot.reporting.arc_observer_fixtures_8624 import frame

    unit = dict(e.units()[1], directory=str(tmp_path))
    path = tmp_path / "unit.json"
    atomic_json(path, unit)
    value = dict(
        unit=unit,
        legacy={},
        actions=[dict(kind="RESET")],
        observations=[{}],
        steps=[],
        events=[],
        end_reason="cutoff",
        observer_rows=[],
        termination=dict(error=None),
        model_tripwire_attempts=[],
    )

    def collect(unit, sdk, raw):
        (raw / "steps.jsonl").write_text(
            json.dumps(
                dict(
                    index=1,
                    action=dict(name="RESET"),
                    observation=dict(pixels=frame("navigation", 1, "cutoff").frame.tolist()),
                )
            )
            + "\n"
        )
        return value

    class SDK:
        pass

    monkeypatch.setattr(sdk, "sdk", SDK)
    monkeypatch.setattr(live, "collect", collect)
    assert runner.main(["--episode", str(path)]) == 0
    result = json.loads((tmp_path / "transfer_stream.json").read_bytes())
    assert len(result["reset_observations"]) == 1
    value["termination"]["error"] = "sdk_failure"
    assert runner.episode(path) == 1
    value["observer_rows"] = [dict(drift=True)]
    with pytest.raises(ValueError, match="original_observer_replay_drift"):
        runner.episode(path)


def test_measure_owned_orchestration(tmp_path, monkeypatch):
    """REQ-VERIFY-8650: current binding precedes every bounded episode child."""
    monkeypatch.setattr(runner.validation, "preconditions", lambda *args: ([], {}, {}))
    monkeypatch.setattr(
        e,
        "authenticate",
        lambda root: dict(
            arc_inputs_ready_score=1, reset_provenance_ready_score=1, historical_model_provenance={}
        ),
    )
    monkeypatch.setattr(e, "units", lambda: [dict(episode_id="fixture")])

    def child(name, argv, logs, **kwargs):
        unit = json.loads(Path(argv[-1]).read_bytes())
        atomic_json(Path(unit["directory"]) / "transfer_stream.json", dict(unit=unit))
        return dict(name=name, passed=True, duration_s=0.01)

    monkeypatch.setattr(runner, "child", child)
    work = runner.measure(e.ROOT, tmp_path / "raw", tmp_path / "scratch", "20261018")
    assert len(work["references"]) == 1 and not work["failures"]
    assert work["code_config_hashes"]
    monkeypatch.setattr(runner.time, "monotonic_ns", iter([0, 3601 * 10**9]).__next__)
    with pytest.raises(TimeoutError):
        runner.measure(e.ROOT, tmp_path / "deadline", tmp_path / "scratch", "20261018")


def test_failed_external_authentication_and_checkpoint(tmp_path, monkeypatch):
    """REQ-VERIFY-8650: absent and changed CPU inputs remain explicit blocked gates."""
    monkeypatch.setattr(runner.validation, "preconditions", lambda *args: ([], {}, {}))

    def reject(root):
        raise ValueError("changed_source_closure")

    monkeypatch.setattr(e, "authenticate", reject)
    work = runner.measure(e.ROOT, tmp_path / "auth", tmp_path / "scratch", "20261018")
    assert work["failures"][0]["check"] == "upstream_authentication"
    original = runner.sha256_file
    monkeypatch.setattr(
        runner, "sha256_file", lambda p: "changed" if p.suffix == ".pt" else original(p)
    )
    work = runner.measure(
        tmp_path / "absent", tmp_path / "checkpoint", tmp_path / "scratch", "20261018"
    )
    assert any(r["check"] == "numerical_checkpoint" for r in work["failures"])
    assert any(r["observed"] is None for r in work["failures"])
    assert not work["references"]


def test_controller_production_validation_path_and_publication_rejection(tmp_path, monkeypatch):
    """REQ-VERIFY-8650: orchestration reaches real publication without nested validation."""
    work = dict(measurement_receipts=[])
    monkeypatch.setattr(runner, "measure", lambda *args: work)
    monkeypatch.setattr(runner.inherited, "validate_checks", lambda *args: [])
    monkeypatch.setattr(e, "build", lambda *args: dict(completed_count=16))
    monkeypatch.setattr(runner, "publish", lambda *args: dict(passed=True))
    assert runner.main(["--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    monkeypatch.undo()

    def child(name, argv, logs, **kwargs):
        log = tmp_path / (name + ".log")
        log.write_text("{}")
        return dict(name=name, passed=False, stdout_path=str(log))

    monkeypatch.setattr(runner, "child", child)
    value = dict(
        experiment_id=8650,
        task_id=e.TASK,
        honest_verdict="complete_null_control",
        verdict_class="null",
        reset_transfer_rows_ready_score=0,
    )
    with pytest.raises(ValueError, match="candidate_rejected"):
        runner.publish(value, tmp_path / "reject" / (e.NAME + ".json"), tmp_path / "raw")
