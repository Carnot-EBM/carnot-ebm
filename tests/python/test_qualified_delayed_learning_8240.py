"""REQ-VERIFY-8240 / REQ-REPORT-8240: qualify before causal execution."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import qualified_delayed_learning_8240 as e


def cli(tmp_path, *args):
    """Exercise real imports and process exits from outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private8240 CLI", flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
    )
    print("after private8240 CLI", result.returncode, flush=True)
    return result


def test_qualification_and_changed_external_bytes(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8240-QUALIFICATION: qualification binds exact code and receipts."""
    work = dict(checks=[], refs=[])
    e.qualification(e.ROOT, tmp_path / "good", work)
    assert work["execution_qualification"]["learning_execution_ready_score"] == 1
    assert all(c["passed"] for c in work["checks"])
    with pytest.raises(ValueError, match="exists"):
        e.qualification(tmp_path, tmp_path / "absent", dict(checks=[], refs=[]))
    private = tmp_path / "copy"
    source = e.ROOT / e.QUALIFICATION
    target = private / e.QUALIFICATION
    target.parent.mkdir(parents=True)
    value = json.loads(source.read_bytes())
    for field, changed in [
        ("learning_execution_ready_score", 0),
        ("experiment_id", 999),
        ("required_checks_passed", False),
    ]:
        atomic_json(target, dict(value, **{field: changed}))
        with pytest.raises(ValueError, match=field):
            e.qualification(private, tmp_path / field, dict(checks=[], refs=[]))
    atomic_json(target, dict(value, task_id="foreign"))
    with pytest.raises(ValueError, match="task_id"):
        e.qualification(private, tmp_path / "task", dict(checks=[], refs=[]))
    target.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sha256"):
        e.qualification(private, tmp_path / "hash", dict(checks=[], refs=[]))
    atomic_json(target, [])
    with pytest.raises(ValueError, match="input_schema"):
        e.qualification(private, tmp_path / "schema", dict(checks=[], refs=[]))


def test_private_cli_causal_recovery_and_tamper(tmp_path):
    """SCENARIO-REPORT-8240-CLI: real children qualify causal readiness without benefit."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert (value["experiment_id"], value["task_id"]) == (8240, e.TASK)
    assert value["utility_trajectory_ready_score"] == 1 and value["verdict_class"] == "null"
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert value["retention_labels_opened"] is False
    assert (
        value["generalized_learning_benefit_score"]
        == value["independent_generalization_score"]
        == 0
    )
    assert [r["actual_exit"] for r in value["child_exit_rows"]] == [0, 73, 0, 73, 0]
    assert all(r["passed"] for r in value["restart_state_hashes"])
    assert len(value["storage_cost_rows"]) == 5
    assert all(r["durable_write_bytes"] > 0 for r in value["storage_cost_rows"])
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    work = json.loads((raw / "measurement.json").read_bytes())
    original_bytes = (raw / "measurement.json").read_bytes()
    work["storage_cost_rows"][0]["durable_write_bytes"] += 1
    atomic_json(raw / "measurement.json", work)
    altered = e.build(work, raw, value["validation_receipts"], fixture=True)
    atomic_json(output, altered)
    assert not e.replay(output)
    (raw / "measurement.json").write_bytes(original_bytes)
    atomic_json(output, value)
    changed = dict(value, utility_trajectory_ready_score=9)
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    receipt = value["child_exit_rows"][0]
    log = Path(receipt["stdout_path"])
    log.write_bytes(log.read_bytes() + b"tamper")
    assert not e.replay(output)
    assert not e.replay(tmp_path / "missing.json")


def test_portable_coefficient_reads():
    """REQ-VERIFY-8240: count every clipped patch and both mixture heads."""
    head = dict(kind="global", scale=1, intercept=0)
    patched = dict(kind="patch", base=head, patches=[{}, {}])
    assert e.coefficient_reads(dict(kind="input")) == 0
    assert e.coefficient_reads(head) == 2
    assert e.coefficient_reads(patched) == 4
    assert e.coefficient_reads(dict(kind="mixture", base=head, candidate=patched, step=0.5)) == 7


def test_block_worker_and_frozen_manifest(tmp_path):
    """REQ-REPORT-8240: external absence blocks; failed owned checks disqualify."""
    output = tmp_path / (e.NAME + ".json")
    assert cli(tmp_path, "--root", tmp_path / "missing", "--fixture-output", output).returncode == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["utility_trajectory_ready_score"] == 0
    assert any(c["observed"] is None for c in value["gate_check_summary"])
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    assert cli(tmp_path, "--date", "19000101").returncode == 2
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/never-write.json").returncode == 2
    worker = tmp_path / "worker/measurement.json"
    assert cli(tmp_path, "--root", tmp_path / "missing", "--worker-output", worker).returncode == 0
    work = json.loads(worker.read_bytes())
    assert e.build(work, worker.parent, [dict(passed=False)])["verdict_class"] == "disqualified"
    specs = e.manifest(tmp_path, output)
    commands = {c["name"]: c for c in specs["commands"]}
    assert "--fail-under=100" in commands["coverage_report"]["argv"]
    assert "--strict" in commands["strict_mypy"]["argv"]
    assert "--files" in commands["spec_coverage"]["argv"]
    assert (
        "tests/python/test_restricted_decision_audit_8210.py"
        in commands["consumer_and_E2E015_019"]["argv"]
    )
    assert specs["repository_health"]["classification"] == "diagnostic"
