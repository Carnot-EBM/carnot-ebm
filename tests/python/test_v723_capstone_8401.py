"""REQ-REPORT-8401 / REQ-VERIFY-8401: scheduling cannot substitute for measured science."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v723_capstone_evidence as e
from carnot.reporting import v723_capstone as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def cli(*args):
    """Real processes cover import, deadline and standalone invocation behavior."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        capture_output=True, text=True, timeout=300,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    root = tmp_path_factory.mktemp("capstone-root")
    for name in [e.DESIGN, e.ACTIVE, e.PROTOCOL]:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((e.ROOT / name).read_bytes())
    output = root.parent / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    return root, output, value


def test_private_cli_missing_and_failed(private, tmp_path):
    """SCENARIO-VERIFY-8401-CONTROLS: external absence differs from an owned failure."""
    root, output, value = private
    assert value["verdict_class"] == "blocked"
    assert value["missing_output_count"] == 13
    assert value["actual_executed_task_count"] == 1
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert len(value["rows"]) == 14
    assert e.replay(output) and cli("--cold-replay", output).returncode == 0
    failed = tmp_path / (e.NAME + ".json")
    result = cli("--root", root, "--output", failed, "--private-control", "--failed-child")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"
    assert cli("--date", "wrong").returncode == 2
    assert cli("--private-control").returncode == 2
    assert not e.replay(tmp_path / "absent.json")
    assert set(value) <= set(value["field_principles"])


@pytest.mark.parametrize("field", ["rows", "canonical_tasks_sha256", "g1", "validation_receipts", "H1"])
def test_rehashed_result_rejects(private, tmp_path, field):
    """SCENARIO-VERIFY-8401-CONTROLS: a checksum cannot authorize a changed conclusion."""
    changed = deepcopy(private[2])
    changed[field] = "forged"
    changed["reproducibility_checksum"] = canonical_hash({k: v for k, v in changed.items() if k != "reproducibility_checksum"})
    p = tmp_path / "tampered.json"
    atomic_json(p, changed)
    assert not e.replay(p)


def test_accounting_from_bound_sources(tmp_path, private):
    """SCENARIO-REPORT-8401-ACCOUNTING: receipts, cascade and original failures stay separate."""
    tasks = e.contract.authority(e.ROOT, tmp_path / "auth")["tasks"]
    rows = []
    for task in tasks[:-1]:
        number = int(task["id"][3:7])
        path = e.ROOT / e.RECEIPTS.get(number, task["deliverable"])
        ref = e.freeze(path, tmp_path / "sources")
        rows.append(e.slot(task, ref, e.CASCADE if number == 8394 else None))
    assert sum(r["producer_executed"] for r in rows) == 8
    assert sum(r["disposition"] == "pre_gate_receipt" for r in rows) == 4
    assert rows[6]["disposition"] == "cascade_skip"
    assert rows[2]["verdict_class"] == rows[3]["verdict_class"] == "disqualified"
    assert rows[2]["metrics"]["direct_state_ready_score"] == 0
    work = e.load(private[2]["work_reference"])
    work["slots"] = rows
    value = e.reduce(work, private[2]["validation_receipts"])
    assert value["actual_executed_task_count"] == 9
    assert value["pre_gate_count"] == 4 and value["cascade_skip_count"] == 1
    assert value["missing_output_count"] == 1
    assert sum(value[k] for k in ["completed_count", "failed_count", "censored_count", "excluded_count"]) == 14
    wrong = deepcopy(work)
    wrong["tasks"][0]["prompt"] += "forged"
    with pytest.raises(ValueError, match="authority"):
        e.reduce(wrong, [])
    assert runner.manifest(tmp_path / "plan")[0]["name"] == "owned_tests"


def test_worker_and_primitive_errors(tmp_path, private):
    """SCENARIO-VERIFY-8401-CONTROLS: real worker failures and primitive replacement are rejected."""
    task = e.contract.authority(e.ROOT, tmp_path / "auth")["tasks"][4]
    primary = e.freeze(e.ROOT / e.RECEIPTS[8392], tmp_path / "raw")
    request, output = tmp_path / "request.json", tmp_path / "worker.json"
    atomic_json(request, dict(task=task, task_sha256="wrong", primary=primary))
    assert cli("--worker-request", request, "--worker-output", output).returncode == 1
    assert json.loads(output.read_bytes())["memory"]["peak_bound_mb"] == 1500
    atomic_json(request, dict(task=task, task_sha256=canonical_hash(task), primary=primary))
    assert cli("--worker-request", request, "--worker-output", output).returncode == 0
    work = e.load(private[2]["work_reference"])
    work["slots"][0]["disposition"] = "forged"
    value = e.build(work, private[2]["validation_receipts"], tmp_path / "changed", tmp_path / (e.NAME + ".json"))
    candidate = tmp_path / "primitive-tamper.json"
    atomic_json(candidate, value)
    assert not e.replay(candidate)
