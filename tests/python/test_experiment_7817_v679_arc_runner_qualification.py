"""Current direct SDK qualification checks for REQ-REPORT-7817."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

import carnot.experiment_7817_v679_arc_runner_qualification as qualification
from scripts.experiments import experiment_7817_v679_arc_runner_qualification as cli
from carnot.experiment_7817_v679_arc_runner_qualification import (
    ObservedArchive,
    command_plan,
    freeze_panel,
    make_agent_factory,
    positive_selector_fixture,
    run_probe,
    seal_log,
    sdk_probe_ok,
    validate_receipts,
    verify_log,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "results/raw/experiment_7817_v679_arc_runner_qualification/validation_command_manifest.json"
)
PANEL = ROOT / "results/raw/experiment_7790_v677_arc_runner_qualification/arc_panel_manifest.json"


def test_scenario_report_7817_dispatch_real_cli_matches_frozen_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7817-DISPATCH: inspect the actual entrypoint's complete child plan."""
    output = tmp_path / "recorded.json"
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}"}
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-u",
            "scripts/experiments/experiment_7817_v679_arc_runner_qualification.py",
            "--record-dispatch",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    frozen = json.loads(MANIFEST.read_text())["commands"]
    coverage_files = [
        next(
            arg.split("=", 1)[1]
            for arg in next(row for row in frozen if row["name"] == name)["argv"]
            if arg.startswith("--data-file=")
        )
        for name in ("changed_module_coverage", "cli_coverage")
    ]
    combine = next(row for row in frozen if row["name"] == "coverage_combine")["argv"]
    assert combine[-2:] == coverage_files
    assert json.loads(output.read_text()) == [
        {"name": row["name"], "argv": row["argv"], "classification": row["classification"]}
        for row in frozen
    ]
    assert command_plan(MANIFEST) == json.loads(output.read_text())
    assert [row["name"] for row in frozen][-1] == "cold_replay"
    assert (
        next(row for row in frozen if row["name"] == "repository_health_full_python_suite")[
            "classification"
        ]
        == "diagnostic"
    )
    assert not any(
        arg == "scripts/experiments/experiment_7803_v678_arc_runner_qualification.py"
        for row in frozen
        for arg in row["argv"]
    )


def test_scenario_report_7817_dispatch_rejects_undeclared_child(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7817-DISPATCH: no child can escape the frozen list."""
    data = json.loads(MANIFEST.read_text())
    data["commands"].append(
        {"name": "surprise", "argv": ["true"], "classification": "required", "timeout_s": 1}
    )
    altered = tmp_path / "manifest.json"
    altered.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="undeclared_child"):
        command_plan(altered)


def test_scenario_report_7817_dispatch_rejects_wrong_names_and_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7817-DISPATCH: the reader checks list content after byte custody."""
    data = json.loads(MANIFEST.read_text())
    altered = tmp_path / "manifest.json"
    data["commands"][0]["name"] = "undeclared"
    altered.write_text(json.dumps(data))
    monkeypatch.setattr(
        qualification, "FROZEN_MANIFEST_SHA256", hashlib.sha256(altered.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="undeclared_child"):
        command_plan(altered)
    data["commands"][0]["name"] = "probe_r11l_off"
    data["commands"][0]["classification"] = "diagnostic"
    altered.write_text(json.dumps(data))
    monkeypatch.setattr(
        qualification, "FROZEN_MANIFEST_SHA256", hashlib.sha256(altered.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="required_classification_changed"):
        command_plan(altered)


def test_scenario_report_7817_custody_retry_parent_and_mutation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7817-CUSTODY: a later attempt cannot rewrite a sealed log."""
    missing_parent = tmp_path / "missing" / "pytest"
    with pytest.raises(FileNotFoundError):
        seal_log(missing_parent, "focused_pytest", 1, b"first")
    missing_parent.mkdir(parents=True)
    first = seal_log(missing_parent, "focused_pytest", 1, b"first")
    later = seal_log(missing_parent, "focused_pytest", 2, b"second")
    assert first != later
    assert verify_log(first, hashlib.sha256(b"first").hexdigest())
    assert verify_log(later, hashlib.sha256(b"second").hexdigest())
    first.write_bytes(b"firsu")
    assert not verify_log(first, hashlib.sha256(b"first").hexdigest())
    assert later.read_bytes() == b"second"


def test_scenario_arc_wmte_7817_panel_and_selector() -> None:
    """SCENARIO-ARC-WMTE-7817-PANEL/SELECTOR: preserve history and prove choice divergence."""
    panel = freeze_panel(PANEL)
    assert panel["games"] == ["cd82", "dc22", "lf52", "m0r0", "sk48", "tn36", "sb26", "sc25"]
    assert len(panel["rows"]) == 48
    assert {row["status"] for row in panel["rows"]} == {"unstarted"}
    fixture = positive_selector_fixture()
    assert fixture["off_archive"] is None
    assert fixture["total_prefix"] == [{"action": 2, "data": None}]
    assert fixture["organic_prefix"] == [{"action": 3, "data": None}]
    assert fixture["organic_seen_before_replay"] == fixture["organic_seen_after_replay"]
    assert fixture["replay_seen_after_replay"] > 0
    assert (
        ObservedArchive("total", [], selector="organic_visits")._select_via_selector([((0,), {})])
        is None
    )


def test_scenario_arc_wmte_7817_factory_defaults_and_local_options() -> None:
    """SCENARIO-ARC-WMTE-7817-SCORED: each arm reaches E3 with explicit local archive settings."""

    class Base:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

    for arm in ("off", "total", "organic"):
        agent = make_agent_factory(arm, [])(Base, cascade=True, proposer=None)(game_id="r11l")
        assert type(agent._policy).__name__ == "E3AgentPolicy"
        archive = agent._policy.explorer.go_explore_archive
        assert (archive is None) == (arm == "off")
        if archive is not None:
            assert archive.selector == "organic_visits"
    with pytest.raises(ValueError, match="unknown_arm"):
        make_agent_factory("invalid", [])


def test_scenario_report_7817_gate_missing_duplicate_and_failed() -> None:
    """SCENARIO-REPORT-7817-GATE: a failed required receipt keeps readiness zero."""
    names = [
        row["name"]
        for row in json.loads(MANIFEST.read_text())["commands"]
        if row["classification"] == "required"
    ]
    good = [{"name": name, "exit_code": 0, "passed": True, "timed_out": False} for name in names]
    assert validate_receipts(MANIFEST, good, sdk_ok=True) == []
    assert validate_receipts(MANIFEST, good[:-1], sdk_ok=True) == [names[-1]]
    assert validate_receipts(MANIFEST, [*good, good[0]], sdk_ok=True) == [names[0]]
    assert validate_receipts(
        MANIFEST, [dict(good[0], exit_code=1, passed=False), *good[1:]], sdk_ok=True
    ) == [names[0]]
    assert validate_receipts(MANIFEST, good, sdk_ok=False) == ["sdk_transport"]


def test_scenario_arc_wmte_7817_real_sdk_probe() -> None:
    """SCENARIO-ARC-WMTE-7817-SCORED: visible SDK frames cross the direct wrapper."""
    from carnot.agentic.arc_solver_kit import offline_arcade

    row = run_probe("r11l", "off", offline_arcade())
    assert row["error"] is None
    assert row["counts"]["sdk_transitions"] == 12
    assert row["policy_entry"]["policy_class"] == "E3AgentPolicy"
    assert row["new_solve_credit"] is False
    assert all(action["induction_attempt_count"] == 0 for action in row["actions"])
    assert sdk_probe_ok([row]) is False
    six = [
        dict(row, game=game, arm=arm)
        for game in ("r11l", "cd82")
        for arm in ("off", "total", "organic")
    ]
    assert sdk_probe_ok(six) is True


def test_scenario_report_7817_owned_child_timeout_and_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7817-CUSTODY: timeout kills only the spawned process group."""
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    private = tmp_path / "private"
    durable = tmp_path / "durable"
    private.mkdir()
    started = __import__("time").monotonic()
    good = {
        "name": "good",
        "argv": [__import__("sys").executable, "-c", "print('done')"],
        "classification": "required",
        "timeout_s": 5,
    }
    receipt = cli.run_child(
        good,
        [{key: good[key] for key in ("name", "argv", "classification")}],
        private,
        durable,
        1,
        started,
        0,
    )
    assert receipt["passed"] is True
    assert verify_log(tmp_path / receipt["log_path"], receipt["log_sha256"])
    slow = {
        "name": "slow",
        "argv": [__import__("sys").executable, "-c", "import time; time.sleep(5)"],
        "classification": "required",
        "timeout_s": 0.2,
    }
    receipt = cli.run_child(
        slow,
        [{key: slow[key] for key in ("name", "argv", "classification")}],
        private,
        durable,
        2,
        started,
        1,
    )
    assert receipt["timed_out"] is True
    assert receipt["exit_code"] != 0
    assert receipt["passed"] is False


def test_scenario_report_7817_recorded_entrypoint_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7817-DISPATCH/GATE: actual CLI loop dispatches every frozen command."""
    panel = PANEL
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "RAW", tmp_path / "raw")
    monkeypatch.setattr(cli, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(cli, "OLD_PANEL", panel)
    monkeypatch.setattr(
        cli,
        "preflight",
        lambda _start: ([{"passed": True, "field": "fixture"}], {}, {"host": "fixture"}),
    )
    observed: list[dict] = []
    failed_name: list[str] = []
    missing_probe: list[str] = []
    run_number = [1]

    def recording_executor(spec, plan, private, durable, attempt, start, completed):
        del private, durable, start, completed
        assert {key: spec[key] for key in ("name", "argv", "classification")} in plan
        observed.append({key: spec[key] for key in ("name", "argv", "classification")})
        name = spec["name"]
        if name.startswith("probe_") and name not in missing_probe:
            _, game, arm = name.split("_")
            row = {
                "episode_id": f"{game}:67501:{arm}",
                "game": game,
                "seed": 67501,
                "arm": arm,
                "max_actions": 12,
                "max_seconds": 75,
                "actions": [
                    {
                        "action": "ACTION1",
                        "goal_firing": None,
                        "induction_attempt_count": 0,
                        "actual_observation": {"frame_sha256": "visible"},
                    }
                ],
                "actions_charged": 1,
                "counter_event_rows": [],
                "raw_metrics": {"peak_level": 0, "elapsed_s": 0.1},
                "counts": {"sdk_transitions": 1},
                "censoring": None,
                "exclusions": [],
                "error": None,
                "policy_entry": {"policy_class": "E3AgentPolicy"},
                "solve_provenance": "live_agent_self_discovery",
                "new_solve_credit": False,
            }
            data = ("PROBE_JSON=" + json.dumps(row) + "\n").encode()
        else:
            data = b"recorded\n"
        path = tmp_path / f"logs{run_number[0]}"
        path.mkdir(exist_ok=True)
        sealed = seal_log(path, name, attempt, data)
        passed = name not in failed_name
        return {
            "name": name,
            "command_argv": spec["argv"],
            "classification": spec["classification"],
            "exit_code": 0 if passed else 1,
            "passed": passed,
            "timed_out": False,
            "log_path": str(sealed.relative_to(tmp_path)),
            "log_sha256": hashlib.sha256(data).hexdigest(),
            "duration_s": 0.01,
        }

    monkeypatch.setattr(cli, "run_child", recording_executor)
    assert cli.main(["--date", "20260928"]) == 0
    value = json.loads((tmp_path / "result.json").read_text())
    assert value["validation_command_manifest_path"] == str(MANIFEST.relative_to(ROOT))
    assert (
        value["validation_command_manifest_sha256"]
        == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    )
    assert value["organic_runner_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert observed == command_plan(MANIFEST)
    failed_name.append("ruff_check")
    run_number[0] = 2
    observed.clear()
    assert cli.main(["--date", "20260928"]) == 0
    value = json.loads((tmp_path / "result.json").read_text())
    assert value["organic_runner_ready_score"] == 0
    assert value["verdict_class"] == "disqualified"
    assert value["gate_check_summary"][0]["field"] == "ruff_check"
    assert observed == command_plan(MANIFEST)
    failed_name.clear()
    missing_probe.append("probe_r11l_off")
    run_number[0] = 3
    observed.clear()
    assert cli.main(["--date", "20260928"]) == 0
    value = json.loads((tmp_path / "result.json").read_text())
    assert value["organic_runner_ready_score"] == 0
    assert value["verdict_class"] == "disqualified"
    assert any(row["field"] == "sdk_transport" for row in value["gate_check_summary"])
    assert observed == command_plan(MANIFEST)
    monkeypatch.setattr(
        cli,
        "preflight",
        lambda _start: ([{"passed": False, "field": "fixture"}], {}, {"host": "fixture"}),
    )
    run_number[0] = 4
    observed.clear()
    assert cli.main(["--date", "20260928"]) == 0
    value = json.loads((tmp_path / "result.json").read_text())
    assert value["organic_runner_ready_score"] == 0
    assert value["verdict_class"] == "blocked"
    assert observed == []


def test_scenario_report_7817_real_preflight_custody(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7817-CUSTODY: named inputs and external roles get exact receipts."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    checks, sources, resources = cli.preflight(time.monotonic())
    by_field = {row["field"]: row for row in checks}
    assert by_field["cpu_backend"]["passed"] is True
    assert by_field["frozen_panel"]["passed"] is True
    assert by_field["registry_game:r11l"]["passed"] is True
    assert by_field["sdk_game:r11l"]["passed"] is True
    assert (
        sources["AGENTS.md"]["sha256"]
        == hashlib.sha256((ROOT / "AGENTS.md").read_bytes()).hexdigest()
    )
    assert sources["results/experiment_7749_arc_organic_measurement.json"]["eligible"] is False
    assert resources["backend"] == "offline_arcade_cpu"


def test_scenario_report_7817_child_guard_and_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7817-CUSTODY: escalation is scoped to the owned process group."""
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    spec = {
        "name": "e2e_009_smoke",
        "argv": [
            "python",
            f"--basetemp={tmp_path / 'pytest' / 'case'}",
            f"--data-file={tmp_path / 'coverage' / '.coverage'}",
            str(tmp_path / "smoke" / "output.json"),
        ],
        "classification": "required",
        "timeout_s": 0,
    }
    private = tmp_path / "private"
    private.mkdir()
    with pytest.raises(ValueError, match="undeclared_child"):
        cli.run_child(spec, [], private, tmp_path / "logs", 1, time.monotonic(), 0)

    class StubbornChild:
        pid = 7817
        waits = 0

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(spec["argv"], timeout)
            return -signal.SIGKILL

    child = StubbornChild()
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    plan = [{key: spec[key] for key in ("name", "argv", "classification")}]
    receipt = cli.run_child(spec, plan, private, tmp_path / "logs", 1, time.monotonic(), 0)
    assert kills == [(child.pid, signal.SIGTERM), (child.pid, signal.SIGKILL)]
    assert receipt["timed_out"] is True
    assert receipt["passed"] is False
    assert (tmp_path / "pytest").is_dir()
    assert (tmp_path / "coverage").is_dir()
    assert (tmp_path / "smoke").is_dir()


def test_scenario_report_7817_cold_replay_mutation_and_cli_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7817-CUSTODY: cold readers reject changed raw and sealed bytes."""
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    schedule = freeze_panel(PANEL)["rows"]
    summary = cli.reduce_probe_evidence(schedule, [])
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"schedule": schedule, "probes": [], "summary": summary}))
    assert cli.cold_reduce(raw) == summary
    assert cli.main(["--cold-reduce", raw.as_posix()]) == 0
    log = tmp_path / "child.log"
    log.write_bytes(b"sealed")
    candidate = tmp_path / "candidate.json"
    value = {
        "raw_probes_path": raw.name,
        "raw_probes_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "validation_receipts": [
            {
                "name": "one",
                "log_path": log.name,
                "log_sha256": hashlib.sha256(b"sealed").hexdigest(),
            }
        ],
        "raw_reduction": summary,
    }
    candidate.write_text(json.dumps(value))
    assert cli.main(["--cold-replay", candidate.as_posix()]) == 0
    log.write_bytes(b"changed")
    with pytest.raises(ValueError, match="sealed_log_mutated:one"):
        cli.cold_replay(candidate)
    log.write_bytes(b"sealed")
    value["raw_reduction"] = dict(summary, completed=-1)
    candidate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="cold_reduction_changed"):
        cli.cold_replay(candidate)
    value["raw_reduction"] = summary
    raw.write_bytes(raw.read_bytes() + b" ")
    candidate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="raw_probes_mutated"):
        cli.cold_replay(candidate)
    raw.write_text(json.dumps({"schedule": schedule, "probes": [], "summary": {}}))
    with pytest.raises(ValueError, match="raw_reduction_mismatch"):
        cli.cold_reduce(raw)
    with pytest.raises(ValueError, match="probe_row_missing"):
        cli.extract_probe({"name": "one", "log_path": log.name})


def test_scenario_report_7817_cli_dispatch_and_probe_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7817-DISPATCH: direct CLI modes retain the frozen plan and probe result."""
    with pytest.raises(ValueError, match="run_date_must_match_milestone"):
        cli.main(["--date", "20260927"])
    assert cli.main(["--list-commands"]) == 0
    assert '"probe_r11l_off"' in capsys.readouterr().out
    recorded = tmp_path / "plan.json"
    assert cli.main(["--record-dispatch", str(recorded)]) == 0
    assert json.loads(recorded.read_text()) == command_plan(MANIFEST)
    monkeypatch.setattr(cli, "offline_arcade", lambda: object())
    row = {"actions": [{"action": "ACTION1"}], "error": None}
    monkeypatch.setattr(cli, "run_probe", lambda game, arm, arcade: row)
    assert cli.main(["--probe", "r11l", "off"]) == 0
    assert "PROBE_JSON=" in capsys.readouterr().out
    row["error"] = "sdk_failure"
    assert cli.main(["--probe", "r11l", "off"]) == 1
