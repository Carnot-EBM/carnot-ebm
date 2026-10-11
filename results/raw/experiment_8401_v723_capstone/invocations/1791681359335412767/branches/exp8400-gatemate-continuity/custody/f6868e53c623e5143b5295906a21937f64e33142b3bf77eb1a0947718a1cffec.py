"""REQ-REPORT-8344 / REQ-VERIFY-8344: private controls grant no board success."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import gatemate_change_ledger_8344 as h
from carnot.reporting import gatemate_ledger_execution_8344 as cli
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from test_gatemate_change_ledger_8330 import panel as old_panel


def panel(changed=False):
    """Only private scripted operator evidence may drive a fixture branch."""
    data = old_panel(changed)
    data["receipt_rows"], data["physical_change"] = h.select_change(data)
    return data


def invoke(*args):
    """External cwd exercises the actual runner without ambient PYTHONPATH."""
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
def test_cli_negative_rehashed_and_recovery(tmp_path, changed):
    """SCENARIO-REPORT-8344-REPLAY: missing observations never become zero science."""
    source, output = tmp_path / "fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, panel(changed))
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["experiment_id"] == 8344 and value["milestone"] == "2026.10.719"
    assert value["run_date"] == "20261009"
    assert value["honest_verdict"] == "complete_blocked_" + (
        "gatemate_device_preflight" if changed else "gatemate_physical_change"
    )
    assert value["intended_count"] == value["excluded_count"] == len(value["rows"]) == 1
    assert value["completed_count"] == value["current_jtag_retry_count"] == 0
    assert value["execution_ready_score"] == value["independent_generalization_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["current_model_calls"] == 0
    assert value["reopen_contract_path"] == h.REOPEN
    assert invoke("--cold-replay", output).returncode == 0

    for field in ["experiment_id", "task_id", "schema", "config", "completed_count"]:
        altered = dict(value, **{field: "wrong"})
        altered["reproducibility_checksum"] = cli.checksum(altered)
        atomic_json(output, altered)
        with pytest.raises(ValueError):
            cli.replay(output)
    assert invoke("--cold-replay", output).returncode == 1
    atomic_json(output, dict(value, reproducibility_checksum="wrong"))
    with pytest.raises(ValueError, match="checksum_drift"):
        cli.replay(output)
    for verdict in ["complete_disqualified_owned_checks", "complete_blocked_required_tools"]:
        altered = deepcopy(value)
        altered.update(
            honest_verdict=verdict,
            required_checks_passed=False,
            verdict_class="disqualified" if "disqualified" in verdict else "blocked",
            gatemate_obligation_ready_score=0,
        )
        altered["future_probe_contract"]["eligible"] = False
        altered["reproducibility_checksum"] = cli.checksum(altered)
        atomic_json(output, altered)
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


@pytest.mark.parametrize(
    "flags,exit_code,passed",
    [
        ([], 0, True),
        ([], 1, False),
        ([], 2, False),
        ([dict(severity="warn", kind="UNKNOWN", detail="warning")], 1, False),
        ([dict(severity="unknown", kind="NEW", detail="unknown")], 1, False),
        (
            [dict(severity="info", kind="IMPLAUSIBLE_PERFECT", detail="dense_sparse_error_max=0")],
            1,
            False,
        ),
    ],
)
def test_unchanged_finding_consumer(tmp_path, flags, exit_code, passed):
    """SCENARIO-VERIFY-8344-FINDINGS: process errors and false-zero fail closed."""
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, {})
    report = dict(
        candidate_sha256=reference(candidate)["sha256"],
        verifier_sha256=cli.policy.verifier_hash(),
        reports=[dict(loaded=True, artifact=str(candidate), flags=flags, flag_count=len(flags))],
    )
    found = cli.policy.consume(report, candidate, exit_code, {})
    assert found["passed"] is passed and found["findings"] == flags
    report["candidate_sha256"] = "wrong"
    assert not cli.policy.consume(report, candidate, exit_code, {})["passed"]
    assert not cli.policy.consume(dict(reports="bad"), candidate, 0, {})["passed"]


@pytest.mark.parametrize("stdout_text,exit_code", [("bad", 1), ('{"reports": []}', 2)])
def test_bad_verifier_receipts(tmp_path, monkeypatch, stdout_text, exit_code):
    """REQ-VERIFY-8344: the existing finding adapter cannot bless malformed output."""
    candidate, stdout = tmp_path / "candidate.json", tmp_path / "audit.stdout"
    atomic_json(candidate, {})
    stdout.write_text(stdout_text)
    spec = cli.CommandSpec("adversarial", ("unused", str(candidate)), "terminal", 1)
    monkeypatch.setattr(
        cli.previous,
        "_execute",
        lambda plan, raw: [dict(stdout_path=str(stdout), actual_exit=exit_code, normal_exit=True)],
    )
    assert not cli.execute([spec], tmp_path / "logs")[0]["passed"]


def test_plan_children_owned_failure_and_recovery(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8344-CLI: bounded children and failed owned work stay explicit."""
    plan = cli.commands(tmp_path / "plan")
    assert "e2e018" in {s.name for s in plan}
    assert not any(s.argv[1:3] == ("tests/python", "-q") for s in plan)
    health = cli.CommandSpec("repository_health_once", ("false",), "health", 1)
    assert cli.execute([health], tmp_path / "health") == []
    child = cli.CommandSpec("child", (sys.executable, "-u", "-c", "print('owned')"), "owned", 5)
    assert cli.execute([child], tmp_path / "child")[0]["passed"]
    timeout = cli.CommandSpec(
        "deadline", (sys.executable, "-u", "-c", "import time;time.sleep(2)"), "owned", 0.01
    )
    assert cli.execute([timeout], tmp_path / "timeout")[0]["timed_out"]
    monkeypatch.setattr(cli, "commands", lambda private: [child])
    monkeypatch.setattr(h, "load", lambda root, raw: panel(True))
    monkeypatch.setattr(cli.previous.previous.qualified, "validators", lambda path: [])
    monkeypatch.setattr(
        cli.previous,
        "_execute",
        lambda plan, raw: [
            dict(
                passed=s.scope == "preconditions",
                normal_exit=True,
                actual_exit=1,
                stdout_path=str(tmp_path / "failed"),
            )
            for s in plan
        ],
    )
    output = tmp_path / (h.NAME + ".json")
    (tmp_path / "failed").write_text("deliberate owned failure")
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    monkeypatch.setattr(
        cli.previous,
        "_execute",
        lambda plan, raw: [dict(passed=True, normal_exit=True) for s in plan],
    )
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["gatemate_obligation_ready_score"] == 1


def test_tools_and_terminal_recovery(tmp_path, monkeypatch):
    """REQ-REPORT-8344: missing tools block and rejected publication preserves bytes."""
    value = dict(
        experiment_id=8344,
        task_id="exp8344-gatemate-change-ledger",
        honest_verdict="complete_blocked_required_tools",
        verdict_class="blocked",
        required_checks_passed=False,
        code_config_hashes=[],
        gate_check_summary=[],
        precondition_receipts=[
            dict(passed=False, normal_exit=True, actual_exit=1, stdout_path=str(tmp_path / "tool"))
        ],
    )
    assert cli.normalize(value)["gatemate_obligation_ready_score"] == 0
    source, output = tmp_path / "fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, panel())
    assert invoke("--input", source, "--output", output).returncode == 0
    original = output.read_bytes()
    (tmp_path / "failed").write_text("deliberate terminal rejection")
    monkeypatch.setattr(cli, "commands", lambda private: [])
    monkeypatch.setattr(
        cli.previous,
        "_execute",
        lambda plan, raw: [
            dict(
                passed=s.scope == "preconditions",
                normal_exit=True,
                actual_exit=1,
                stdout_path=str(tmp_path / "failed"),
            )
            for s in plan
        ],
    )
    assert cli.main(["--input", str(source), "--output", str(output)]) == 1
    assert output.read_bytes() == original


def test_live_load_and_tamper(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8344-FRONTIER: original bytes and actual authority stay bound."""
    data = h.load(h.ROOT, tmp_path / "raw")
    assert data["history_ready"], data["checks"]
    assert data["authority"]["task"]["id"] == "exp8344-gatemate-change-ledger"
    h.verify_primitives(data)
    value = h.reduce(data)
    assert value["gatemate_obligation_ready_score"] == 1
    assert value["original_transcript_sha256"] == h.TRANSCRIPT_PIN
    assert not value["physical_change_evidence"]["exists"]
    with monkeypatch.context() as m:
        m.setattr(h, "TRANSCRIPT_PIN", "sha256:wrong")
        with pytest.raises(ValueError, match="original_transcript_drift"):
            h.verify_primitives(data)
    altered = deepcopy(data)
    altered["authority"]["digest"] = "wrong"
    with pytest.raises(ValueError, match="authority_drift"):
        h.verify_primitives(altered)
    side = Path(data["terminal_reference"]["path"])
    side.chmod(0o600)
    atomic_json(side, dict(publication=dict(primary_sha256="wrong")))
    data["terminal_reference"] = reference(side)
    with pytest.raises(ValueError, match="sidecar_drift"):
        h.verify_primitives(data)
    missing = h.load(tmp_path / "absent", tmp_path / "missing")
    assert not missing["history_ready"]
    assert h.reduce(missing)["verdict_class"] == "blocked"
    for name in ["PIN", "BOUND_PIN", "TRANSCRIPT_PIN"]:
        with monkeypatch.context() as m:
            m.setattr(h, name, "sha256:wrong")
            assert not h.load(h.ROOT, tmp_path / name)["history_ready"]


def test_nonphysical_and_missing_cli(tmp_path):
    """REQ-VERIFY-8344: software updates cannot satisfy a physical obligation."""
    data = panel(True)
    data["candidate_rows"][0]["raw_receipt"]["changed_physical_fields"] = ["dirtyjtag"]
    assert not h.select_change(data)[1]["exists"]
    source, output = tmp_path / "missing.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, h.empty_data())
    assert invoke("--input", source, "--output", output).returncode == 0
    assert json.loads(output.read_bytes())["honest_verdict"] == "complete_blocked_gatemate_history"
    assert invoke("--cold-replay", output).returncode == 0


def test_historical_replay_under_current_execution_binding(tmp_path, monkeypatch):
    """REQ-VERIFY-8344: production adapters cannot replace historical replay identity."""
    monkeypatch.setattr(cli.previous, "h", h)
    assert h.load(h.ROOT, tmp_path / "inside-current-runner")["history_ready"]
