"""REQ-REPORT-8330 / REQ-VERIFY-8330: private controls never imply board work."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import gatemate_change_ledger_8330 as h
from carnot.reporting import gatemate_ledger_execution_8330 as cli
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from test_gatemate_obligation_8316 import panel as old_panel


def panel(changed=False):
    """Keep scripted evidence private while advancing only its test frontier."""
    data = old_panel(changed)
    data["receipt_rows"], data["physical_change"] = h.select_change(data)
    data["authority"] = dict(activated=True, fixture=True)
    data["authority_refs"] = []
    return data


def invoke(*args):
    """A real child outside the checkout exercises runner imports and publication."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(h.ROOT / h.CLI), *map(str, args)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.parametrize("changed", [False, True])
def test_real_cli_and_tamper(tmp_path, changed):
    """SCENARIO-REPORT-8330-CLI: valid replay, tamper, owned failure and recovery."""
    source, output = tmp_path / "fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, panel(changed))
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["experiment_id"] == 8330 and value["milestone"] == "2026.10.718"
    assert value["run_date"] == "20261009"
    assert value["honest_verdict"] == "complete_blocked_" + (
        "gatemate_device_preflight" if changed else "gatemate_physical_change"
    )
    assert value["intended_count"] == value["excluded_count"] == len(value["rows"]) == 1
    assert value["current_jtag_retry_count"] == value["execution_ready_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["current_model_calls"] == 0
    assert value["independent_generalization_score"] == 0
    assert value["reopen_contract_path"] == h.REOPEN
    assert value["adversarial_findings"] == []
    assert invoke("--cold-replay", output).returncode == 0
    for field in ["experiment_id", "task_id", "schema", "config", "completed_count"]:
        altered = dict(value, **{field: "wrong"})
        altered["reproducibility_checksum"] = cli.checksum(altered)
        atomic_json(output, altered)
        with pytest.raises(ValueError):
            cli.replay(output)
    assert invoke("--cold-replay", output).returncode == 1
    atomic_json(output, dict(value, reproducibility_checksum="bad"))
    with pytest.raises(ValueError, match="checksum_drift"):
        cli.replay(output)
    failed = deepcopy(value)
    failed.update(
        honest_verdict="complete_disqualified_owned_checks",
        verdict_class="disqualified",
        required_checks_passed=False,
        gatemate_obligation_ready_score=0,
    )
    failed["future_probe_contract"]["eligible"] = False
    failed["reproducibility_checksum"] = cli.checksum(failed)
    atomic_json(output, failed)
    assert cli.replay(output)["passed"]
    if changed:
        primitive = Path(value["replay_input_reference"]["path"])
        data = json.loads(primitive.read_bytes())
        data["candidate_rows"][0]["raw_receipt"]["operator_authored"] = False
        atomic_json(primitive, data)
        altered = deepcopy(value)
        altered["replay_input_reference"] = reference(primitive)
        altered["raw_shard_hashes"][0] = reference(primitive)
        altered["reproducibility_checksum"] = cli.checksum(altered)
        atomic_json(output, altered)
        assert invoke("--cold-replay", output).returncode == 1
    assert invoke("--date", "bad").returncode == 2
    assert invoke("--input", source, "--output", h.ROOT / "results" / output.name).returncode == 1


def test_physical_frontier():
    """SCENARIO-VERIFY-8330-FRONTIER: instructions and host rebuilds never qualify."""
    data = panel(True)
    assert h.select_change(data)[1]["exists"]
    data["candidate_rows"][0]["raw_receipt"]["changed_physical_fields"] = ["dirtyjtag"]
    assert not h.select_change(data)[1]["exists"]
    data = panel(True)
    data["candidate_rows"][0]["raw_receipt"]["receipt_timestamp"] = "bad"
    assert not h.select_change(data)[1]["exists"]
    missing = h.empty_data()
    assert h.reduce(missing)["honest_verdict"] == "complete_blocked_gatemate_history"


def test_live_history_and_rehashed_sidecar(tmp_path, monkeypatch):
    """REQ-VERIFY-8330: immutable live history is read without physical commands."""
    data = h.load(h.ROOT, tmp_path / "raw")
    assert data["history_ready"] and data["authority"]["activated"]
    h.verify_primitives(data)
    value = h.reduce(data)
    assert value["gatemate_obligation_ready_score"] == 1
    assert value["original_transcript_sha256"] == h.TRANSCRIPT_PIN
    side = Path(data["terminal_reference"]["path"])
    side.chmod(0o600)
    atomic_json(side, dict(publication=dict(primary_sha256="bad")))
    data["terminal_reference"] = reference(side)
    with pytest.raises(ValueError, match="sidecar_drift"):
        h.verify_primitives(data)
    missing = h.load(tmp_path / "absent", tmp_path / "missing")
    assert not missing["history_ready"]
    monkeypatch.setattr(h, "BOUND_PIN", "sha256:wrong")
    assert not h.load(h.ROOT, tmp_path / "bad-pin")["history_ready"]


def test_failure_recovery_tools_and_children(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8330-CLI: failed checks preserve prior bytes until recovery."""
    plan = cli.commands(tmp_path / "plan")
    assert "e2e018" in {spec.name for spec in plan}
    assert not any(spec.argv[1:3] == ("tests/python", "-q") for spec in plan)
    health = cli.CommandSpec("repository_health_once", ("false",), "health", 1)
    assert cli.execute([health], tmp_path / "health") == []
    child = cli.CommandSpec(
        "private_child", (sys.executable, "-u", "-c", "print('owned')"), "owned", 5
    )
    receipt = cli.execute([child], tmp_path / "child")[0]
    assert receipt["passed"] and receipt["stdout_sha256"]
    timeout = cli.CommandSpec(
        "deadline", (sys.executable, "-u", "-c", "import time;time.sleep(2)"), "owned", 0.01
    )
    assert cli.execute([timeout], tmp_path / "timeout")[0]["timed_out"]
    value = dict(
        experiment_id=8330,
        task_id="exp8330-gatemate-change-ledger",
        milestone="2026.10.718",
        run_date="20261009",
        code_config_hashes=[],
        honest_verdict="complete_disqualified_owned_checks",
        verdict_class="disqualified",
        required_checks_passed=False,
        gate_check_summary=[],
        precondition_receipts=[
            dict(
                passed=False,
                normal_exit=True,
                actual_exit=1,
                stdout_path=str(tmp_path / "missing-tool"),
            )
        ],
    )
    blocked = cli.normalize(value)
    assert blocked["honest_verdict"] == "complete_blocked_required_tools"
    source, output = tmp_path / "fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, panel())
    assert invoke("--input", source, "--output", output).returncode == 0
    blocked.update(json.loads(output.read_bytes()))
    blocked.update(
        honest_verdict="complete_blocked_required_tools", gatemate_obligation_ready_score=0
    )
    blocked["future_probe_contract"]["eligible"] = False
    blocked["reproducibility_checksum"] = cli.checksum(blocked)
    atomic_json(output, blocked)
    assert cli.replay(output)["passed"]
    original = output.read_bytes()
    monkeypatch.setattr(cli, "commands", lambda private: [child])
    monkeypatch.setattr(h, "load", lambda root, raw: panel())
    monkeypatch.setattr(cli.previous.qualified, "validators", lambda path: [])
    (tmp_path / "failure").write_text("deliberate failed owned check")
    monkeypatch.setattr(
        cli,
        "_execute",
        lambda plan, raw: [
            dict(
                passed=spec.scope == "preconditions",
                normal_exit=True,
                actual_exit=0 if spec.scope == "preconditions" else 1,
                stdout_path=str(tmp_path / "failure"),
            )
            for spec in plan
        ],
    )
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    monkeypatch.setattr(
        cli, "_execute", lambda plan, raw: [dict(passed=True, normal_exit=True) for spec in plan]
    )
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert output.read_bytes() != original


@pytest.mark.parametrize("raw_report,exit_code", [("bad", 1), ('{"reports": []}', 2)])
def test_malformed_verifier_reports(tmp_path, monkeypatch, raw_report, exit_code):
    """REQ-REPORT-8330: process errors and malformed findings fail closed."""
    candidate, stdout = tmp_path / "candidate.json", tmp_path / "audit.stdout"
    atomic_json(candidate, {})
    stdout.write_text(raw_report)
    spec = cli.CommandSpec("adversarial", ("unused", str(candidate)), "terminal", 1)
    monkeypatch.setattr(
        cli,
        "_execute",
        lambda plan, raw: [dict(stdout_path=str(stdout), actual_exit=exit_code, normal_exit=True)],
    )
    assert not cli.execute([spec], tmp_path / "logs")[0]["passed"]


def test_transcript_and_authority_drift(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8330-FRONTIER: fresh digests cannot repair altered authority."""
    data = h.load(h.ROOT, tmp_path / "raw")
    data["authority"]["digest"] = "wrong"
    with pytest.raises(ValueError, match="authority_drift"):
        h.verify_primitives(data)
    monkeypatch.setattr(h, "TRANSCRIPT_PIN", "sha256:wrong")
    assert not h.load(h.ROOT, tmp_path / "transcript")["history_ready"]


def test_historical_replay_inside_current_adapter(tmp_path, monkeypatch):
    """REQ-VERIFY-8330: historical replay retains its own identity during the real CLI."""
    monkeypatch.setattr(cli.previous, "h", h)
    monkeypatch.setattr(cli.previous, "replay", cli.replay)
    assert h.load(h.ROOT, tmp_path / "history")["history_ready"]
