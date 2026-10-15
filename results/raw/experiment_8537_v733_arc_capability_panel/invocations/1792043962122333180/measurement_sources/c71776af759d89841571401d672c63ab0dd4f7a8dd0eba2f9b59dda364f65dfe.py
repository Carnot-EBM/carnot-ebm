"""REQ-VERIFY-8537: private CLI and rejection paths use actual consumers."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from carnot.reporting import arc_capability_panel_8537 as e
from carnot.reporting import arc_capability_panel_runner_8537 as runner
from carnot.reporting.current_work_receipt import atomic_json


def test_cli_dates_and_rejections(tmp_path):
    """SCENARIO-REPORT-8537-REPLAY crosses the actual standalone boundary."""
    for args, expected in [
        (["--date", "20261014"], 2),
        (["--deliberate-error"], 7),
        (["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        result = subprocess.run(
            [sys.executable, str(e.ROOT / e.CLI), *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == expected, result.stdout + result.stderr
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    assert runner.main(["--cold-replay", str(bad)]) == 1
    assert runner.main(["--deliberate-error"]) == 7


@pytest.mark.parametrize("failed", [False, True])
def test_private_block_and_owned_failure(tmp_path, monkeypatch, failed):
    """REQ-VERIFY-8537: blocked and failed work reach unchanged byte consumers."""
    monkeypatch.setattr(
        runner, "validate_checks", lambda *_: [dict(name="private", passed=not failed)]
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "missing"), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("disqualified" if failed else "blocked")
    assert value["arc_capability_panel_ready_score"] == 0
    assert value["completed_count"] == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert side["passed"] and side["consumer"]["passed"]
    assert side["consumer"]["gates"][0]["artifact_field"] == "arc_capability_panel_ready_score"


def test_validation_collects_original_union(tmp_path, monkeypatch):
    """REQ-VERIFY-8537 keeps assertions disjoint and coverage scoped to owned code."""
    calls = []

    def execute(name, argv, logs, **kwargs):
        calls.append((name, argv))
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (name + ".stdout")
        log.write_text(
            "tests/python/test_private.py::test_original\n" if name.startswith("collect") else ""
        )
        return dict(name=name, passed=True, stdout_path=str(log))

    monkeypatch.setattr(runner.shipped, "child", execute)
    receipts = runner.validate_checks(tmp_path / "raw", tmp_path / "private")
    assert all(r["passed"] for r in receipts)
    assert any(name == "coverage_report" and "--fail-under=100" in argv for name, argv in calls)
    assert all("--no-cov" in argv for name, argv in calls if name.startswith("historical"))
    monkeypatch.setattr(runner.shipped, "child", lambda *_a, **_k: dict(passed=False))
    assert not any(
        r["passed"] for r in runner.validate_checks(tmp_path / "failed", tmp_path / "scratch")
    )


def test_observation_adapter_records_attempts(tmp_path, monkeypatch):
    """REQ-REPORT-8537: record actual policy attempts without substituting actions."""

    class Agent:
        def __init__(self):
            self._policy = SimpleNamespace(induction_attempts=[dict(skipped="disabled_by_env")])

        def choose_action(self, *_):
            return "original_action"

    monkeypatch.setattr(
        e.history.runtime.agent_module, "make_carnot_agent", lambda *_a, **_k: Agent
    )
    monkeypatch.setattr(e.history.panel_runtime, "sdk", lambda: "sdk")
    path = tmp_path / "unit.json"
    atomic_json(path, dict(directory=str(tmp_path), capability_mask=True))

    def run(unit, sdk, raw):
        assert os.environ["CARNOT_ARC_DISABLE_INDUCTION"] == "1"
        assert os.environ["CARNOT_ARC_SUPERVISOR_CAPABILITY_MASK"] == "1"
        agent = e.history.runtime.agent_module.make_carnot_agent()()
        assert agent.choose_action([], None) == "original_action"
        return dict(error=None, model_tripwire_attempts=[])

    monkeypatch.setattr(e.history.runtime, "episode", run)
    assert runner.main(["--episode", str(path)]) == 0
    assert json.loads((tmp_path / "attempts.jsonl").read_text()) == [
        dict(skipped="disabled_by_env")
    ]


def test_measured_adapter_and_complete_wall_cost(tmp_path, monkeypatch):
    """REQ-REPORT-8537: retain startup/teardown in the complete episode wall cost."""
    monkeypatch.setattr(e, "authenticate", lambda *_: dict(protocol=e.qualification.protocol()))
    monkeypatch.setattr(e.history.shipped, "preconditions", lambda *_: ([], {}, {}))
    monkeypatch.setattr(
        e.history.panel_runtime,
        "sdk",
        lambda: SimpleNamespace(
            available_environments=[
                SimpleNamespace(game_id=g) for g in e.qualification.protocol()["games"]
            ]
        ),
    )
    calls = []

    def execute(name, argv, logs, **kwargs):
        calls.append((name, kwargs))
        unit = json.loads(Path(argv[-1]).read_bytes())
        directory = Path(unit["directory"])
        atomic_json(directory / "episode.json", dict(unit=unit))
        (directory / "steps.jsonl").write_text("")
        return dict(name=name, passed=True, duration_s=4.0)

    monkeypatch.setattr(e.history, "child", execute)
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path / "private")
    assert len(work["references"]) == 16
    assert all(k["deadline"] <= 130 for _, k in calls)
    assert all(r["complete_wall_s"] == 4.0 for r in work["references"])


def test_episode_early_owned_failure_is_recorded(tmp_path, monkeypatch):
    """REQ-VERIFY-8537: a failed child retains its empty attempt stream and fails."""
    path = tmp_path / "unit.json"
    atomic_json(path, dict(directory=str(tmp_path), capability_mask=False))
    monkeypatch.setattr(e.history.panel_runtime, "sdk", lambda: None)
    monkeypatch.setattr(
        e.history.runtime,
        "episode",
        lambda *_: dict(error="owned_failure", model_tripwire_attempts=[]),
    )
    assert e.episode(path) == 1
    assert (tmp_path / "attempts.jsonl").read_bytes() == b""
    assert json.loads((tmp_path / "episode.json").read_bytes())["error"] == "owned_failure"
