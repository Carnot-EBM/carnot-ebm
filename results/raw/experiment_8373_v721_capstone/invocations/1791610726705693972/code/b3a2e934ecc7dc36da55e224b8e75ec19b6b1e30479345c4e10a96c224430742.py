"""REQ-REPORT-8373 / REQ-VERIFY-8373: terminal evidence must survive fresh readers."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v721_capstone_evidence as e
from carnot.reporting import v721_capstone as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def private_root(root):
    """Real authority copies exercise mechanics without inventing science rows."""
    for name in [e.DESIGN, e.ACTIVE, e.PROTOCOL]:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((e.ROOT / name).read_bytes())
    return root


def cli(*args):
    """A real script process checks imports, output flushing and dispatch."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_private_cli_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8373-REPLAY: absence cannot be rehashed into evidence."""
    root = private_root(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    done = cli("--root", root, "--output", output, "--private-control")
    assert done.returncode == 0, done.stdout + done.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["missing_output_count"] == 13
    assert value["actual_executed_task_count"] == 1
    assert len(value["rows"]) == 14
    assert value["MODEL_SPECS"] == []
    assert not any(value["model_invocation_counts"].values())
    assert value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert cli("--cold-replay", output).returncode == 0
    for key in ["rows", "H1", "paper_ready", "validation_receipts"]:
        changed = deepcopy(value)
        if key == "rows":
            changed[key][0]["disposition"] = "qualified"
        elif key == "validation_receipts":
            changed[key][0]["passed"] = False
        elif key == "H1":
            changed[key]["intended_count"] = 96
        else:
            changed[key] = not changed[key]
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        assert cli("--cold-replay", path).returncode == 1
    assert cli("--date", "wrong").returncode == 2
    assert not e.replay(tmp_path / "absent")


def test_owned_failure_and_authority(tmp_path):
    """SCENARIO-REPORT-8373-ACCOUNTING: owned failure and task drift stay visible."""
    root = private_root(tmp_path / "root")
    work = e.measure(root, tmp_path / "raw")
    value = e.build(
        work,
        [dict(passed=False, name="actual_failure")],
        tmp_path / "raw",
        tmp_path / (e.NAME + ".json"),
    )
    assert value["verdict_class"] == "disqualified"
    assert value["capstone_execution_ready_score"] == 0
    changed = deepcopy(work)
    changed["tasks"][0]["title"] += " foreign"
    with pytest.raises(ValueError, match="authority"):
        e.reduce(changed, [dict(passed=True)])
    changed = deepcopy(work)
    changed["memory_measurements"]["parent_growth_mb"] = 501
    assert e.reduce(changed, [dict(passed=True)])["verdict_class"] == "disqualified"
    assert runner.manifest(tmp_path)[0]["name"] == "owned_tests"
    assert e.MODEL_SPECS == []


def test_current_fourteen_and_qualified_null(tmp_path):
    """SCENARIO-REPORT-8373-ACCOUNTING: authentic current nulls keep every missing slot."""
    work = e.measure(e.ROOT, tmp_path / "current")
    value = e.build(work, [dict(passed=True)], tmp_path / "current", tmp_path / (e.NAME + ".json"))
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8360, 8374))
    assert value["actual_executed_task_count"] == 8
    assert value["pre_gate_count"] == 2
    assert value["missing_output_count"] == 4
    assert value["H1"]["qualified"]
    assert value["H1"]["qualified_count"] == 97
    assert value["H2"]["qualified_count"] == 67
    assert value["H2"]["retention_windows"] == [0, 32, 64, 96]
    assert value["science_ready_score"] == 1
    assert (
        value["completed_count"]
        + value["failed_count"]
        + value["censored_count"]
        + value["excluded_count"]
        == 14
    )
    assert all(not g["closed"] for g in value["three_prd_gaps"])
    assert value["deployment_results"]["atomic_recovery"] is None
    assert value["rows"][11]["verdict_class"] == "disqualified"
    assert value["original_utility_dispositions"][0]["verdict_class"] == "disqualified"


def test_actual_failed_child_and_cli_guards(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8373-REPLAY: real owned failure stays disqualified after publication."""
    root = private_root(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control", "--failed-child")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert not value["validation_receipts"][0]["passed"]
    assert value["validation_receipts"][0]["exit_code"] == 1
    assert e.replay(output)
    forbidden = cli("--private-control")
    assert forbidden.returncode == 2
    with pytest.raises(SystemExit):
        runner.main(["--date", "wrong"])


def test_frozen_reads_and_missing_source(tmp_path):
    """SCENARIO-VERIFY-8373-REPLAY: mutable authority never replaces frozen operands."""
    source = tmp_path / "source.json"
    atomic_json(source, dict(answer=1))
    ref = e.freeze(source, tmp_path / "custody")
    assert e.freeze(source, tmp_path / "custody") == ref
    atomic_json(source, dict(answer=2))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with e.frozen_inputs([ref], scratch):
        assert json.loads(source.read_bytes()) == dict(answer=1)
        source.write_bytes(b"private rewritten bytes")
        with pytest.raises(ValueError, match="mutable_authority"):
            (e.ROOT / "research-roadmap.yaml").read_bytes()
        assert (tmp_path / "unknown").exists() is False
    assert json.loads(source.read_bytes()) == dict(answer=2)
    assert e.load(ref) == dict(answer=1)
    assert len(e.references(dict(a=[ref], b="text"))) == 1
    assert e.freeze(tmp_path / "absent", tmp_path / "custody")["sha256"] is None


def test_history_and_exact_retirement(tmp_path):
    """REQ-REPORT-8373: only authenticated exact repeated verdicts retire narrow scopes."""
    source = next((e.ROOT / "results").glob("experiment_8355_*.json"))
    receipt = e.historical(source, tmp_path / "history")
    assert receipt["authenticated"]
    assert receipt["honest_verdict"] == "complete_null_no_supervisor_outcomes"
    assert receipt["readiness_imported"] is False
    task = dict(
        id="exp8370-arc-outcome-delta",
        prior_failures=[
            dict(
                experiment_id="exp8355-arc-supervisor-frontier",
                verdict=receipt["honest_verdict"],
                retire_if_same_verdict=True,
            )
        ],
    )
    current = dict(honest_verdict=receipt["honest_verdict"], eligible=True)
    records = e.retire(task, current, [receipt])
    assert len(records) == 1
    assert records[0]["hypothesis_retired"] is False
    assert e.retire(task, dict(current, eligible=False), [receipt]) == []
    assert e.retire(task, dict(current, honest_verdict="changed"), [receipt]) == []


def test_worker_authentication_and_sealed_tamper(tmp_path):
    """SCENARIO-VERIFY-8373-REPLAY: repaired hashes cannot replace terminal authority."""
    from carnot.reporting import v721_contract_methods as contract

    task = next(
        t
        for t in contract.authority(e.ROOT, tmp_path / "authority")["tasks"]
        if t["id"].startswith("exp8372-")
    )
    primary = e.freeze(e.ROOT / task["deliverable"], tmp_path / "primary")
    summary = e.outcome(task, primary, tmp_path / "capture")
    request, output = tmp_path / "request.json", tmp_path / "summary.json"
    args = dict(
        task=task, task_sha256=canonical_hash(task), primary=primary, closure=summary["closure"]
    )
    atomic_json(request, args)
    assert e.worker(request, output) == 0
    args["task_sha256"] = "wrong"
    atomic_json(request, args)
    assert cli("--worker-request", request, "--worker-output", output).returncode == 1
    missing = e.freeze(tmp_path / "absent", tmp_path / "primary")
    args.update(task_sha256=canonical_hash(task), primary=missing, closure=None)
    atomic_json(request, args)
    assert e.worker(request, output) == 1
    foreign = deepcopy(task)
    foreign["id"] = "exp8372-foreign"
    with pytest.raises(ValueError, match="wrong_authority"):
        e.outcome(foreign, primary, tmp_path / "foreign")
    value = e.load(primary)
    value["execution_ready_score"] = 1
    changed = tmp_path / "tamper.json"
    atomic_json(changed, value)
    bad = e.freeze(changed, tmp_path / "tamper")
    bad["path"] = primary["path"]
    with pytest.raises(ValueError, match="terminal_binding"):
        e.outcome(task, bad, tmp_path / "tamper", summary["closure"])
    with pytest.raises(ValueError, match="terminal_binding"):
        e.outcome(task, bad, tmp_path / "tamper_initial")


def test_append_only_manifest_and_note(tmp_path, monkeypatch):
    """REQ-REPORT-8373: current retirement never removes historical entries or broadens scope."""
    import yaml

    manifest = tmp_path / "ops/exclusion_manifest.yaml"
    manifest.parent.mkdir()
    manifest.write_text("retired:\n- experiment_id: old\n  reason: preserve original\n")
    original = manifest.read_bytes()
    output = tmp_path / (e.NAME + ".json")
    root = private_root(tmp_path / "root")
    raw = tmp_path / "raw"
    work = e.measure(root, raw)
    value = e.build(work, [dict(passed=True)], raw, output)
    value["retirements"] = [
        dict(
            task_id="exp8370-arc-outcome-delta",
            prior_task_id="exp8355-arc-supervisor-frontier",
            exact_verdict="complete_null_no_supervisor_outcomes",
            scope="unchanged inspection only",
            hypothesis_retired=False,
            reopening_condition="new authenticated outcomes",
        )
    ]
    atomic_json(output, value)
    runner.append_retirements(output, manifest)
    assert manifest.read_bytes().startswith(original)
    assert len(yaml.safe_load(manifest.read_bytes())["retired"]) == 2
    runner.append_retirements(output, manifest)
    assert len(yaml.safe_load(manifest.read_bytes())["retired"]) == 2
    monkeypatch.setattr(e, "ROOT", tmp_path)
    (tmp_path / "docs/research-notes").mkdir(parents=True)
    runner.write_note(output)
    note = (tmp_path / "docs/research-notes/v721-capstone.md").read_text()
    assert "Three PRD gaps remain open" in note
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: 0)
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0


def test_actual_lost_worker_and_exact_contract(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8373-REPLAY: a real failed child cannot erase its intended producer."""
    root = private_root(tmp_path / "root")
    work = e.measure(root, tmp_path / "raw")
    task = work["tasks"][12]
    source = e.ROOT / task["deliverable"]
    primary = e.freeze(source, tmp_path / "primary")
    original_child = e.child
    monkeypatch.setattr(
        e,
        "child",
        lambda name, argv, logs, **kwargs: original_child(
            name, [sys.executable, "-u", "-c", "raise SystemExit(9)"], logs, **kwargs
        ),
    )
    summary, receipt = e.invoke(task, primary, tmp_path / "failed_worker")
    assert receipt["exit_code"] == 9
    assert not receipt["passed"]
    assert summary["honest_verdict"] == e.load(primary)["honest_verdict"]
    assert summary["branch_replay"]["status"] == "disqualified_owned_child"
    assert summary["producer_executed"]
    original_authority = e.contract.authority
    actual = original_authority(root, tmp_path / "assessment")
    actual["tasks"] = actual["tasks"][:-1]
    monkeypatch.setattr(e.contract, "authority", lambda *args: actual)
    with pytest.raises(ValueError, match="exact_fourteen"):
        e.measure(root, tmp_path / "wrong_count")


def test_utility_and_source_controls(tmp_path):
    """SCENARIO-VERIFY-8373-REPLAY: real natural inputs precede repaired source and metric controls."""
    tasks = e.contract.authority(e.ROOT, tmp_path / "authority")["tasks"]
    task = tasks[2]
    primary = e.freeze(e.ROOT / task["deliverable"], tmp_path / "primary")
    summary = e.outcome(task, primary, tmp_path / "capture")
    assert summary["branch_replay"]["passed"]
    source = next(
        r
        for r in summary["closure"]
        if r["path"].endswith("/python/carnot/reporting/threshold_guard_8362.py")
    )
    changed = deepcopy(summary["closure"])
    row = next(r for r in changed if r["path"] == source["path"])
    target = tmp_path / "changed_source.py"
    target.write_bytes(b"changed immutable source")
    row["snapshot_path"] = str(target)
    row["sha256"] = e.sha256_file(target)
    blocked = e.outcome(task, primary, tmp_path / "missing_source", changed)
    assert blocked["branch_replay"]["status"] == "blocked_missing_source_closure"
    row["expected_sha256"] = row["sha256"]
    changed_source = e.outcome(task, primary, tmp_path / "changed_source", changed)
    assert changed_source["branch_replay"]["status"] == "blocked_changed_source"
    audit = json.loads((e.ROOT / tasks[1]["deliverable"]).read_bytes())
    audit["H1"]["qualified_count"] = 128
    with pytest.raises(ValueError, match="independent_utility"):
        e.utility(audit, [])
    pre = e.freeze(e.ROOT / "results/experiment_8363_atomic_table_state.json", tmp_path / "pre")
    with pytest.raises(ValueError, match="pre_gate_identity"):
        e.outcome(tasks[4], pre, tmp_path / "foreign_pre")
    wrong = dict(primary, sha256="sha256:wrong")
    with pytest.raises(ValueError):
        e.load(wrong)


def test_natural_cold_replay_and_rehashed_primitives(tmp_path):
    """SCENARIO-VERIFY-8373-REPLAY: sealed natural branches reconstruct in fresh child processes."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    path = tmp_path / "valid.json"
    atomic_json(path, value)
    assert e.replay(path)
    changed = deepcopy(work)
    changed["inputs"][0]["summary"]["selected"]["current_contract_ready_score"] = 0
    candidate = e.build(
        changed, [dict(passed=True)], tmp_path / "tamper", tmp_path / (e.NAME + ".json")
    )
    atomic_json(tmp_path / "tamper.json", candidate)
    assert not e.replay(tmp_path / "tamper.json")


def test_real_coverage_receipt_path(tmp_path):
    """REQ-VERIFY-8373: persist a genuine disk-backed coverage report alongside validation."""
    root = private_root(tmp_path / "root")
    probe = tmp_path / "coverage_probe.py"
    probe.write_text("print('actual measured private child')\n")
    data = tmp_path / "probe.coverage"
    for args in [
        ["run", "--data-file=" + str(data), str(probe)],
        [
            "json",
            "--data-file=" + str(data),
            "--include=" + str(probe),
            "-o",
            str(tmp_path / "coverage.json"),
        ],
    ]:
        result = subprocess.run(
            [sys.executable, "-m", "coverage", *args], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(root, output, tmp_path, control=True) == 0
    value = json.loads(output.read_bytes())
    assert any("coverage" in r["snapshot_path"] for r in value["code_config_hashes"])
