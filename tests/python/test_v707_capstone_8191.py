"""REQ-REPORT-8191, REQ-VERIFY-8191: private accounting and replay controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import pytest

from carnot.reporting import v707_capstone as c
from carnot.reporting import v707_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary


def fixture(tmp_path):
    """Use real immutable authority while keeping every synthetic result private."""
    root = tmp_path / "private"
    receipt = json.loads((e.ROOT / e.INPUT).read_bytes())
    receipt = {
        k: receipt[k]
        for k in (
            "task_contract",
            "authority_snapshots",
            "canonical_tasks_sha256",
            "literature_mapping",
            "historical_hash_failures",
            "prior_scope_ledger",
        )
    }
    tasks = receipt["task_contract"]
    for role in ("design", "active"):
        ref = receipt["authority_snapshots"][role]
        target = root / (role + ".bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(ref["snapshot_path"]).read_bytes())
        ref.update(snapshot_path=str(target), sha256=sha256_file(target))
    for n, task in enumerate(tasks[:-1], 8178):
        path = root / task["deliverable"]
        side = path.parent / "raw" / path.stem / "terminal.json"
        value = dict(
            experiment_id=n,
            task_id=task["id"],
            honest_verdict="complete_null_private",
            verdict_class="null",
            required_checks_passed=True,
            flagged_adversarial=False,
            gate_check_summary=[],
            raw_shard_hashes=[],
            rows=[],
            terminal_validation_sidecar_path=str(side),
        )
        value.update(
            {
                g["artifact_field"]: 1
                for t in tasks
                for g in t["gated_on"]
                if g["upstream"] == task["id"]
            }
        )
        if n == 8178:
            value.update(receipt)
        pub = publish_primary(path, value, lambda _: dict(passed=True))
        atomic_json(side, dict(publication=pub))
    return root, tasks


def replay_value(tmp_path, data, passed=True):
    """Bind private rows and logs exactly as a cold consumer would read them."""
    source, output = tmp_path / "input.json", tmp_path / "out.json"
    atomic_json(source, data)
    value = e.reduce(data)
    value.update(
        required_checks_passed=passed,
        validation_receipts=[],
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        replay_input_reference=e.reference(source),
    )
    c.qualify(value, passed)
    atomic_json(output, value)
    return source, output, value


def test_mixed_dispositions_and_authority(tmp_path):
    """SCENARIO-REPORT-8191: null, skip, disqualified and absent stay distinct."""
    root, tasks = fixture(tmp_path)
    (root / tasks[9]["deliverable"]).unlink()
    path = root / tasks[2]["deliverable"]
    v = json.loads(path.read_bytes())
    v.update(
        required_checks_passed=False,
        verdict_class="disqualified",
        honest_verdict="complete_disqualified_owned_validation",
    )
    atomic_json(path, v)
    skip = root / tasks[8]["deliverable"]
    skip.rename(skip.with_name("experiment_8186_calibrated_online_memory.json"))
    atomic_json(
        skip.with_name("experiment_8186_calibrated_online_memory.json"),
        dict(
            schema="blocked_gate_check_v1",
            task_id=tasks[8]["id"],
            failed_field="calibrated_memory_ready_score",
            failed_upstream=tasks[2]["id"],
            failed_evidence_path=str(path),
            failed_evidence_sha256=sha256_file(path),
            failed_operator="==",
            failed_expected=1,
            failed_observed=0,
        ),
    )
    data = e.load(root, tmp_path / "raw")
    value = e.reduce(data)
    assert len(value["rows"]) == value["completed_count"] == 14
    assert value["rows"][9]["primary_honest_verdict"] == "<missing>"
    assert value["rows"][2]["verdict_class"] == "disqualified"
    assert value["rows"][8]["disposition"] == "conductor_skip"
    assert value["rows"][1]["verdict_class"] == "null"
    assert value["H1"]["status"] == value["H2"]["status"] == "blocked"
    assert value["science_ready_score"] == value["independent_generalization_score"] == 0
    assert len(value["gap_decisions"]) == 3
    assert all(not r["closed"] for r in value["gap_decisions"].values())
    assert not any(r["retire_method_family"] for r in value["retirement_decisions"])
    p = root / e.INPUT
    v = json.loads(p.read_bytes())
    v["canonical_tasks_sha256"] = "changed"
    atomic_json(p, v)
    assert not e.load(root, tmp_path / "mutated")["authority"]["activated"]
    p.unlink()
    assert not e.load(root, tmp_path / "absent")["authority"]["activated"]


def test_real_primitives_and_historical_scopes(tmp_path):
    """REQ-VERIFY-8191: current H1 null survives historical integrity failures."""
    data = e.load(e.ROOT, tmp_path / "raw")
    value = e.reduce(data)
    assert value["authority"]["activated"]
    assert value["H1"]["status"] == "completed_null"
    assert value["H1"]["completed_count"] == 97
    assert value["H1"]["original_missing_mask"].count(True) == 31
    assert 0.05 < value["H1"]["raw_p_value"] < 1.0
    assert value["H1"]["raw_p_value_method"] == "source_bootstrap_margin_0.02"
    assert value["H2"]["status"] == "blocked"
    assert value["H2"]["learning_execution"] is False
    assert value["h1_development_signal_score"] == value["h2_development_signal_score"] == 0
    assert value["historical_hash_failures"]
    assert (
        value["service_evidence_scope"]["conditional_exact_repeat"]["cached_service_ready_score"]
        == 1
    )
    assert len(value["board_obligations"]) == 5
    assert value["rows"][2]["verdict_class"] == "disqualified"
    for n in (8185, 8188, 8190):
        t = data["tasks"][n - 8178]
        assert e.primitive(data["primaries"][t["id"]], n) == data["audits"][t["id"]]
    altered = deepcopy(data["primaries"][data["tasks"][7]["id"]])
    altered["paired_intervals"]["all_slot"]["mean_gain"] = 99
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        e.primitive(altered, 8185)
    assert e.primitive({}, 8187) == {"available": False}
    assert all(r["prior_sha256_matches_snapshot"] for r in value["retirement_decisions"])


def test_worker_and_replay_outside_checkout(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8191: external CLI and cold replay need no PYTHONPATH."""
    root, _ = fixture(tmp_path)
    data = e.load(root, tmp_path / "raw")
    source, output, value = replay_value(tmp_path, data)
    argv = [
        str(c.ROOT / ".venv/bin/python"),
        str(c.ROOT / c.CLI),
        "--worker-input",
        str(source),
        "--output",
        str(tmp_path / "worker.json"),
    ]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    print("[private_e2e] before_subprocess completed=0 pending=1", flush=True)
    child = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True)
    print("[private_e2e] after_subprocess completed=1 pending=0", flush=True)
    assert child.returncode == 0, child.stdout + child.stderr
    monkeypatch.setattr(sys, "argv", argv[1:])
    with pytest.raises(SystemExit) as info:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert info.value.code == 0
    assert c.replay(output)["passed"]
    assert c.main(["--cold-replay", str(output)]) == 0
    value["H1"]["status"] = "invented"
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        c.replay(output)
    source.write_text("{}")
    with pytest.raises(ValueError, match="input_hash_drift"):
        c.replay(output)


@pytest.mark.parametrize("passed", [True, False])
def test_main_and_owned_validation_failure(tmp_path, monkeypatch, passed):
    """REQ-REPORT-8191: owned failure zeros readiness and leaves terminal science."""
    root, _ = fixture(tmp_path)
    gates = dict(
        gates={k: dict(pass_=k != "G2") for k in ("G1", "G2", "G3", "G4")},
        paper_ready=False,
        unmet_gates=["G2"],
    )

    def execute(plan, raw):
        receipts = []
        for spec in plan:
            log = raw / (spec.name + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps(gates) if spec.name == "publication_gate" else "checked\n")
            if spec.name == "independent_reduction":
                source = Path(spec.argv[spec.argv.index("--worker-input") + 1])
                target = Path(spec.argv[spec.argv.index("--output") + 1])
                atomic_json(target, e.reduce(json.loads(source.read_bytes())))
            receipts.append(
                dict(
                    name=spec.name,
                    scope=spec.scope,
                    passed=passed if spec.scope == "owned" else True,
                    normal_exit=True,
                    log_path=str(log),
                    log_sha256=sha256_file(log),
                )
            )
        return receipts

    monkeypatch.setattr(c, "execute", execute)
    output = tmp_path / "experiment_8191_v707_capstone.json"
    args = ["--root", str(root), "--output", str(output), "--fixture-e2e"]
    assert c.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] is passed
    assert value["capstone_execution_ready_score"] == int(passed)
    assert value["MODEL_SPECS"] == value["call_ledger"] == []
    assert c.main(["--cold-replay", str(output)]) == 0
    assert c.main(args) == 0
    value["validation_receipts"][0]["log_sha256"] = "changed"
    atomic_json(output, value)
    assert c.main(["--cold-replay", str(output)]) == 1


def test_error_controls_and_commands(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8191: missing evidence, unsafe fixtures and audit drift fail."""
    assert c.main(["--fixture-e2e"]) == 1
    assert c.main(["--fixture-e2e", "--output", str(tmp_path / "out.json")]) == 1
    assert (
        c.main(
            ["--worker-input", str(tmp_path / "missing"), "--output", str(tmp_path / "out.json")]
        )
        == 1
    )
    assert c.main(["--worker-input", str(tmp_path / "missing")]) == 1
    root, tasks = fixture(tmp_path)
    real = e.primitive
    monkeypatch.setattr(e, "primitive", lambda *_: (_ for _ in ()).throw(ValueError("bad")))
    assert not any(r["qualified"] for r in e.load(root, tmp_path / "bad")["dispositions"])
    monkeypatch.setattr(e, "primitive", real)
    p = root / tasks[2]["deliverable"]
    v = json.loads(p.read_bytes())
    side = Path(v["terminal_validation_sidecar_path"])
    validator = Path(json.loads(side.read_bytes())["publication"]["sidecar_path"])
    report = json.loads(validator.read_bytes())
    report["report"]["passed"] = False
    atomic_json(validator, report)
    assert not e.load(root, tmp_path / "failed")["dispositions"][2]["qualified"]
    data = e.load(root, tmp_path / "raw")
    data["audits"][tasks[0]["id"]] = dict(available=True, result={})
    _, output, _ = replay_value(tmp_path, data, False)
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        c.replay(output)
    monkeypatch.setattr(c, "execute", lambda *_: [dict(passed=False, normal_exit=True)])
    assert not c.terminal(output)["passed"]
    assert (
        c.main(["--root", str(root), "--output", str(tmp_path / "failed.json"), "--fixture-e2e"])
        == 1
    )
    plan = c.commands(tmp_path / "scratch")
    assert all(
        "::" not in a
        for s in plan
        if s.name in ("ruff_check", "ruff_format", "scoped_spec_coverage")
        for a in s.argv
    )


def test_child_wait_counts(tmp_path, monkeypatch, capsys):
    """SCENARIO-REPORT-8191: waits report actual completed and pending children."""

    def slow(plan, raw):
        time.sleep(0.03)
        return [dict(passed=True, normal_exit=True)]

    monkeypatch.setattr(c, "run_checks", slow)
    plan = [c.CommandSpec("one", ("true",), "owned"), c.CommandSpec("two", ("true",), "owned")]
    assert len(c.execute(plan, tmp_path, heartbeat_s=0.001)) == 2
    lines = capsys.readouterr().out
    assert "waiting_one completed=0 pending=2" in lines
    assert "waiting_two completed=1 pending=1" in lines
