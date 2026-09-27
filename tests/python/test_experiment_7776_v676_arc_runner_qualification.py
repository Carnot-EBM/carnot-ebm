"""REQ-REPORT-7776 and REQ-ARC-WMTE-7776 scored-path qualification."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import pytest

from carnot.experiment_7776_v676_arc_runner_qualification import (
    gate_decision,
    reduce_probe_evidence,
)
from carnot.experiment_7763_v675_arc_runner_qualification import schedule_rows
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands


def test_scenario_report_7776_custody_rejects_changed_probes() -> None:
    """SCENARIO-REPORT-7776-CUSTODY: an absent or invented unit cannot count."""
    schedule = schedule_rows()
    assert len(schedule) == 48
    assert reduce_probe_evidence(schedule, []) == {
        "intended": 48,
        "started": 0,
        "completed": 0,
        "actions": 0,
    }
    with pytest.raises(ValueError, match="schedule_changed"):
        reduce_probe_evidence(schedule[:-1], [])
    probe = {
        "episode_id": schedule[0]["episode_id"],
        "actions": [{"action": "RESET"}],
        "actions_charged": 1,
        "error": None,
        "counter_event_rows": [
            {
                "event": "observation",
                "provenance": "organic",
                "seen": 1,
                "organic_seen": 1,
                "replay_seen": 0,
                "reset_seen": 0,
            },
            {
                "event": "observation",
                "provenance": "replay",
                "seen": 2,
                "organic_seen": 1,
                "replay_seen": 1,
                "reset_seen": 0,
            },
        ],
    }
    assert reduce_probe_evidence(schedule, [probe])["completed"] == 1
    assert (
        reduce_probe_evidence(schedule, [dict(probe, episode_id="fixture:67500:off")])["completed"]
        == 1
    )
    assert (
        reduce_probe_evidence(schedule, [dict(probe, episode_id="r11l:67501:organic")])["completed"]
        == 1
    )
    with pytest.raises(ValueError, match="unscheduled_probe"):
        reduce_probe_evidence(schedule, [dict(probe, episode_id="alien")])
    with pytest.raises(ValueError, match="duplicate_probe"):
        reduce_probe_evidence(schedule, [probe, probe])
    with pytest.raises(ValueError, match="action_count"):
        reduce_probe_evidence(schedule, [dict(probe, actions_charged=2)])
    wrong = [*probe["counter_event_rows"]]
    wrong[1] = dict(wrong[1], organic_seen=2)
    with pytest.raises(ValueError, match="counter_provenance"):
        reduce_probe_evidence(schedule, [dict(probe, counter_event_rows=wrong)])


def test_scenario_report_7776_scope_requires_every_current_receipt() -> None:
    """SCENARIO-REPORT-7776-SCOPE: historical collection debt never counts as green."""
    required = ("focused_pytest", "changed_module_coverage", "adversarial_verify")
    receipts = [
        {"name": name, "exit_code": 0, "passed": True, "log_sha256": "sha256:ok"}
        for name in required
    ]
    assert gate_decision(receipts, required, sdk_ok=True) == (True, [])
    assert gate_decision(receipts, required, sdk_ok=False) == (False, ["sdk_transport"])
    assert gate_decision(receipts[:-1], required, sdk_ok=True) == (
        False,
        ["adversarial_verify"],
    )
    failed = [*receipts[:-1], dict(receipts[-1], exit_code=1, passed=False)]
    assert gate_decision(failed, required, sdk_ok=True) == (
        False,
        ["adversarial_verify"],
    )
    duplicate = [*receipts, receipts[0]]
    assert gate_decision(duplicate, required, sdk_ok=True) == (False, ["focused_pytest"])


def test_scenario_report_7776_child_owned_timeout_and_private_basetemp(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7776-CHILD: a real timed-out child dies; logs stay private."""
    parent = tmp_path / "nested"
    parent.mkdir(parents=True)
    child = tmp_path / "test_child.py"
    child.write_text("def test_child():\n    assert True\n")
    receipts = run_commands(
        tmp_path,
        [
            CommandSpec(
                "basetemp_child",
                (
                    sys.executable,
                    "-m",
                    "pytest",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={parent / 'child'}",
                    str(child),
                    "-q",
                ),
                "private_fixture",
                60,
            )
        ],
        log_dir=tmp_path / "logs",
        heartbeat_s=0.05,
    )
    assert receipts[0]["passed"]
    pid_path = tmp_path / "owned.pid"
    code = f"import os,time;open({str(pid_path)!r},'w').write(str(os.getpid()));time.sleep(20)"
    killed = run_commands(
        tmp_path,
        [CommandSpec("owned_timeout", (sys.executable, "-u", "-c", code), "kill_fixture", 0.3)],
        log_dir=tmp_path / "kill_logs",
        heartbeat_s=0.05,
    )[0]
    assert killed["timed_out"] and not killed["passed"]
    assert pid_path.is_file()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)
    log = tmp_path / killed["log_path"]
    assert killed["log_sha256"] == "sha256:" + hashlib.sha256(log.read_bytes()).hexdigest()
