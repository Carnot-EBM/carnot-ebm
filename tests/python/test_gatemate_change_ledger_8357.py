"""REQ-REPORT-8357 / REQ-VERIFY-8357: history is separate from physical evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import gatemate_change_ledger_8357 as h
from carnot.reporting import gatemate_ledger_execution_8357 as cli
from carnot.reporting import gatemate_history_8357 as history
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from test_gatemate_change_ledger_8344 import panel as old_panel


def panel(changed=False):
    """Private controls exercise physical semantics without becoming observations."""
    data = old_panel(changed)
    data.update(authority={}, authority_refs=[], history_authentication={"passed": True})
    return data


def invoke(*args):
    """Run the actual CLI from outside the checkout to test its import boundary."""
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
def test_real_cli_and_rehashed_controls(tmp_path, changed):
    """SCENARIO-REPORT-8357-REPLAY: stored claims cannot invent board success."""
    source, output = tmp_path / "fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, panel(changed))
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["experiment_id"] == 8357 and value["milestone"] == "2026.10.720"
    assert value["MODEL_SPECS"] == [] and value["current_model_calls"] == 0
    assert value["execution_ready_score"] == value["current_jtag_retry_count"] == 0
    assert value["reopen_contract_path"] == h.REOPEN
    assert invoke("--cold-replay", output).returncode == 0
    for field in [
        "experiment_id",
        "task_id",
        "config",
        "completed_count",
        "history_authentication",
    ]:
        altered = dict(value, **{field: "wrong"})
        altered["reproducibility_checksum"] = cli.checksum(altered)
        atomic_json(output, altered)
        with pytest.raises(ValueError):
            cli.replay(output)
    assert invoke("--cold-replay", output).returncode == 1
    atomic_json(output, dict(value, reproducibility_checksum="wrong"))
    assert invoke("--cold-replay", output).returncode == 1
    assert invoke("--date", "wrong").returncode == 2
    assert invoke("--input", source, "--output", h.ROOT / "results" / output.name).returncode == 1


def test_immutable_closure_and_missing(tmp_path):
    """SCENARIO-VERIFY-8357-CLOSURE: exact copies qualify; changed bytes do not."""
    source = tmp_path / "producer.py"
    source.write_text("CONFIG = {'date': '20261009'}\n")
    ref = reference(source)
    result = history.recover(ref, tmp_path, tmp_path / "raw", {}, [])
    assert (
        result["available"]
        and Path(result["reference"]["path"]).read_bytes() == source.read_bytes()
    )
    source.write_text("CONFIG = {'date': '20261010'}\n")
    missing = history.recover(ref, tmp_path, tmp_path / "missing", {}, [])
    assert not missing["available"] and missing["expected_sha256"] == ref["sha256"]
    assert missing["observed_sha256"] != ref["sha256"]
    Path(result["reference"]["path"]).chmod(0o600)
    Path(result["reference"]["path"]).write_text("drift\n")
    with pytest.raises(ValueError):
        history.verify([result])


def test_live_custody_and_fresh_immutable_replay(tmp_path):
    """SCENARIO-VERIFY-8357-CLOSURE: a producer closure works outside mutable bindings."""
    data = h.load(h.ROOT, tmp_path / "natural")
    assert not data["history_ready"] and not data["document_rows"]
    assert len(data["history_authentication"]["missing_hashes"]) == 2
    assert data["board"]["blocked_idcode"] == "0xffffffff"
    h.verify_primitives(data)
    bundle = data["history_authentication"]["bundles"][0]
    assert bundle["passed"]
    assert history.replay_bundle(bundle, tmp_path / "immutable")["passed"]
    for field in ["expected_sha256", "available"]:
        altered = deepcopy(data)
        altered["history_authentication"]["bundles"][0]["rows"][3][field] = "wrong"
        with pytest.raises(ValueError):
            h.verify_primitives(altered)
    altered = deepcopy(data)
    altered["history_authentication"]["missing_hashes"] = []
    with pytest.raises(ValueError, match="history_authentication_drift"):
        h.verify_primitives(altered)
    altered = deepcopy(data)
    altered["authority"]["activated"] = False
    with pytest.raises(ValueError, match="authority_drift"):
        h.verify_primitives(altered)
    altered = deepcopy(data)
    altered["board"]["source_transcript_sha256"] = "wrong"
    with pytest.raises(ValueError, match="original_transcript_drift"):
        h.verify_primitives(altered)
    altered = deepcopy(data)
    altered["history_authentication"]["bundles"][0]["primary_sha256"] = "wrong"
    with pytest.raises(ValueError, match="historical_primary_drift"):
        h.verify_primitives(altered)


def test_missing_history_and_suppressed_physical_scan(tmp_path):
    """SCENARIO-VERIFY-8357-CLOSURE: unavailable history never triggers a physical scan."""
    data = h.load(tmp_path, tmp_path / "absent")
    assert not data["history_ready"] and data["authority"] == {}
    assert h.reduce(data)["honest_verdict"] == "complete_blocked_gatemate_history"
    assert any(not row["passed"] and row["artifact_field"] == "authority" for row in data["checks"])
    data.update(fixture=True, history_authentication={"passed": False}, document_rows=[{}])
    with pytest.raises(ValueError, match="physical_scan_before_history"):
        h.verify_primitives(data)


def test_seal_and_observed_tampering(tmp_path):
    """REQ-VERIFY-8357: missing and changed hashes retain distinct failure paths."""
    source = tmp_path / "source.py"
    source.write_text("original\n")
    ref = reference(source)
    source.write_text("changed\n")
    row = history.recover(ref, tmp_path, tmp_path / "diagnostic", {}, [])
    row["reference"] = reference(source)
    with pytest.raises(ValueError, match="missing_closure_drift"):
        history.verify([row])
    row["reference"] = None
    row["observed_sha256"] = "wrong"
    with pytest.raises(ValueError, match="observed_closure_drift"):
        history.verify([row])
    row.update(available=True, reference=ref)
    row["expected_sha256"] = "wrong"
    with pytest.raises(ValueError, match="closure_seal_drift"):
        history.verify([row])
    source.unlink()
    absent = history.recover(ref, tmp_path, tmp_path / "absent", {}, [])
    assert absent["observed_sha256"] is None
    with pytest.raises(ValueError, match="historical_primary_drift"):
        history.authenticate(h.ROOT / h.UPSTREAM, "wrong", tmp_path / "bad", {})


def test_plan_children_failure_and_atomic_rejection(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8357-CLI: actual child errors and rejected publication stay visible."""
    from carnot.reporting import gatemate_ledger_execution_8246 as base
    from carnot.reporting import gatemate_ledger_execution_8330 as executor

    assert "e2e018" in {s.name for s in cli.commands(tmp_path / "plan")}
    child = cli.CommandSpec("child", (sys.executable, "-u", "-c", "print('owned')"), "owned", 5)
    assert cli.execute([child], tmp_path / "child")[0]["passed"]
    timeout = cli.CommandSpec(
        "deadline", (sys.executable, "-c", "import time;time.sleep(2)"), "owned", 0.01
    )
    assert cli.execute([timeout], tmp_path / "timeout")[0]["timed_out"]
    source, output = tmp_path / "fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, panel())
    assert invoke("--input", source, "--output", output).returncode == 0
    original = output.read_bytes()
    monkeypatch.setattr(cli, "commands", lambda private: [child])
    monkeypatch.setattr(h, "load", lambda root, raw: panel())
    log = tmp_path / "failed"
    log.write_text("deliberate failure")
    monkeypatch.setattr(
        executor,
        "_execute",
        lambda plan, raw: [
            dict(passed=False, normal_exit=True, actual_exit=1, stdout_path=str(log)) for s in plan
        ],
    )
    assert cli.main(["--input", str(source), "--output", str(output)]) == 1
    assert output.read_bytes() == original
    monkeypatch.setattr(base, "validators", lambda path: [])
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert (
        value["verdict_class"] == "blocked"
        and value["honest_verdict"] == "complete_blocked_required_tools"
    )
    assert cli.replay(output)["passed"]
    monkeypatch.setattr(
        executor,
        "_execute",
        lambda plan, raw: [
            dict(
                passed=s.scope == "preconditions",
                normal_exit=True,
                actual_exit=0,
                stdout_path=str(log),
            )
            for s in plan
        ],
    )
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert cli.replay(output)["passed"]


def test_authenticated_frontier_scans_only_physical_documents(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8357-CLOSURE: private parser controls cannot become device evidence."""
    bundle = history.authenticate(h.ROOT / h.UPSTREAM, h.PIN, tmp_path / "closure", {})
    monkeypatch.setattr(history, "authenticate", lambda *args: deepcopy(bundle))
    monkeypatch.setattr(history, "replay_bundle", lambda *args: dict(passed=True))
    data = h.load(h.ROOT, tmp_path / "parser-control")
    data["fixture"] = True
    assert data["history_ready"] and len(data["document_rows"]) >= len(h.DOCS)
    h.verify_primitives(data)
    value = h.reduce(data)
    assert value["honest_verdict"] == "complete_blocked_gatemate_physical_change"
    assert value["gatemate_obligation_ready_score"] == 1 and value["execution_ready_score"] == 0


def test_terminal_sidecar_drift(tmp_path, monkeypatch):
    """REQ-VERIFY-8357: changed terminal custody cannot be authenticated."""
    monkeypatch.setattr(
        history, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False))
    )
    with pytest.raises(ValueError, match="historical_terminal_drift"):
        history.authenticate(h.ROOT / h.UPSTREAM, h.PIN, tmp_path / "bad-sidecar", {})


def test_sidecar_seal_and_incomplete_replay(tmp_path, monkeypatch):
    """REQ-VERIFY-8357: unavailable sources and changed terminal seals stay blocked."""
    with monkeypatch.context() as patcher:
        patcher.setitem(history.SEALS, h.PIN, ("wrong", "wrong"))
        with pytest.raises(ValueError, match="historical_sidecar_seal_drift"):
            history.authenticate(h.ROOT / h.UPSTREAM, h.PIN, tmp_path / "seal", {})
    with pytest.raises(ValueError, match="incomplete_history_closure"):
        history.replay_bundle(dict(passed=False), tmp_path / "incomplete")


def test_self_consistently_rehashed_primitive(tmp_path):
    """SCENARIO-REPORT-8357-REPLAY: renewed hashes cannot bless changed physical meaning."""
    source, output = tmp_path / "fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, panel(True))
    assert invoke("--input", source, "--output", output).returncode == 0
    value = json.loads(output.read_bytes())
    primitive = Path(value["replay_input_reference"]["path"])
    data = json.loads(primitive.read_bytes())
    data["candidate_rows"][0]["raw_receipt"]["operator_authored"] = False
    atomic_json(primitive, data)
    value["replay_input_reference"] = reference(primitive)
    value["raw_shard_hashes"][0] = reference(primitive)
    value["reproducibility_checksum"] = cli.checksum(value)
    atomic_json(output, value)
    assert invoke("--cold-replay", output).returncode == 1


def test_rehashed_terminal_membership_and_replay_claim(tmp_path):
    """SCENARIO-VERIFY-8357-CLOSURE: new hashes cannot replace terminal seals or replay."""
    data = h.load(h.ROOT, tmp_path / "natural")
    altered = deepcopy(data)
    row = altered["history_authentication"]["bundles"][0]["rows"][1]
    row.update(expected_sha256=row["observed_sha256"])
    row["original_path"] = "changed-terminal-location"
    with pytest.raises(ValueError, match="terminal_membership_drift"):
        h.verify_primitives(altered)
    altered = deepcopy(data)
    altered["history_authentication"]["bundles"][0]["replay"]["passed"] = False
    with pytest.raises(ValueError, match="historical_replay_drift"):
        h.verify_primitives(altered)
