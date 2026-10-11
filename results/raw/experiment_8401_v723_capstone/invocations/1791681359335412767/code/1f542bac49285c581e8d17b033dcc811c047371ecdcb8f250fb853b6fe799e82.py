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
        capture_output=True,
        text=True,
        timeout=300,
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


@pytest.mark.parametrize(
    "field", ["rows", "canonical_tasks_sha256", "g1", "validation_receipts", "H1"]
)
def test_rehashed_result_rejects(private, tmp_path, field):
    """SCENARIO-VERIFY-8401-CONTROLS: a checksum cannot authorize a changed conclusion."""
    changed = deepcopy(private[2])
    changed[field] = "forged"
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
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
    assert (
        sum(
            value[k]
            for k in ["completed_count", "failed_count", "censored_count", "excluded_count"]
        )
        == 14
    )
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
    value = e.build(
        work, private[2]["validation_receipts"], tmp_path / "changed", tmp_path / (e.NAME + ".json")
    )
    candidate = tmp_path / "primitive-tamper.json"
    atomic_json(candidate, value)
    assert not e.replay(candidate)


def test_sealed_absence_and_positive_reduce(private, tmp_path):
    """SCENARIO-REPORT-8401-SEALED-REDUCTION: preserve every unit and terminal class."""
    work = e.load(private[2]["work_reference"])
    task = work["tasks"][0]
    absent = e.slot(task, e.freeze(tmp_path / "missing.json", tmp_path / "raw"), None)
    assert absent["metrics"] is None and absent["disposition"] == "absent_primary"
    altered = deepcopy(work)
    for row in altered["slots"]:
        row.update(eligible=True, verdict_class="null", disposition="executed_producer")
    reduced = e.reduce(altered, private[2]["validation_receipts"])
    assert reduced["verdict_class"] == "null"
    assert reduced["independent_generalization_score"] == 0
    assert reduced["retirements"] == []


@pytest.fixture(scope="module")
def natural(private, tmp_path_factory):
    """SCENARIO-REPORT-8401-ACCOUNTING: capture current primitive sources exactly once."""
    directory = tmp_path_factory.mktemp("natural-capstone")
    work = e.measure(e.ROOT, directory / "natural")
    value = e.build(
        work, private[2]["validation_receipts"], directory / "built", directory / (e.NAME + ".json")
    )
    return work, value


def test_natural_measurement_and_replay(natural, private, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8401-MEMORY: sealed current sources and fresh workers preserve failures."""
    work, value = natural
    assert value["actual_executed_task_count"] == 9
    assert value["pre_gate_count"] == 4 and value["cascade_skip_count"] == 1
    assert value["historical_replay_qualification"]["original_v722_disqualification_preserved"]
    assert len(value["historical_replay_qualification"]["original_v722_failures"]) == 2
    assert not value["repository_health"]["current_suite_launched"]
    assert any(
        "coverage_combination_mode" in citation["fields_imported"]
        for citation in value["cited_upstream_artifacts"]
    )
    path = tmp_path / "natural-candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    changed = deepcopy(work)
    changed["slots"][0]["task"] = deepcopy(changed["slots"][0]["task"])
    changed["slots"][0]["task"]["prompt"] += "wrong"
    candidate = e.build(
        changed,
        private[2]["validation_receipts"],
        tmp_path / "changed-task",
        tmp_path / (e.NAME + ".json"),
    )
    atomic_json(path, candidate)
    assert not e.replay(path)
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: 0)
    assert runner.main(["--output", str(tmp_path / (e.NAME + ".json"))]) == 0


def test_declared_pregate_operands(tmp_path):
    """SCENARIO-REPORT-8401-SEALED-REDUCTION: gate receipts bind declared fields and actual values."""
    task = e.contract.authority(e.ROOT, tmp_path / "authority")["tasks"][4]
    primary = e.freeze(e.ROOT / e.RECEIPTS[8392], tmp_path / "source")
    changed = deepcopy(task)
    changed["gated_on"][0]["artifact_field"] = "invented_field"
    with pytest.raises(ValueError, match="declared_gate"):
        e.slot(changed, primary, None)
    value = e.load(primary)
    value["gates_evaluated"][0]["actual"] = 1
    path = tmp_path / "receipt.json"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="gate_observed"):
        e.slot(task, e.freeze(path, tmp_path / "tamper"), None)


def test_current_process_memory():
    """SCENARIO-VERIFY-8401-MEMORY: exec-inherited host peaks cannot masquerade as worker RSS."""
    current = e.memory()
    assert current["measurement_source"] == "/proc/self/status"
    assert current["current_rss_mb"] > 0
    assert current["peak_rss_mb"] >= current["current_rss_mb"]


def test_authority_and_receipt_controls(private, tmp_path):
    """SCENARIO-VERIFY-8401-CONTROLS: forged argv and missing authority fail independently."""
    with pytest.raises(ValueError, match="authority"):
        e.measure(tmp_path / "missing-root", tmp_path / "scratch")
    changed = deepcopy(private[2])
    changed["validation_receipts"][0]["argv"].append("invented")
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    path = tmp_path / "receipt-tamper.json"
    atomic_json(path, changed)
    assert not e.replay(path)
    changed = deepcopy(private[2])
    changed["code_config_hashes"][0]["sha256"] = "sha256:" + "0" * 64
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    atomic_json(path, changed)
    assert not e.replay(path)


def test_native_coverage_and_findings_preserved(tmp_path):
    """REQ-REPORT-8401: native topology and original findings remain qualified source claims."""
    task = e.contract.authority(e.ROOT, tmp_path / "auth")["tasks"][3]
    primary = e.freeze(e.ROOT / task["deliverable"], tmp_path / "native")
    row = e.slot(task, primary, None)
    source = e.load(primary)
    assert row["metrics"]["coverage_combination_mode"] == source["coverage_combination_mode"]
    assert row["metrics"]["coverage_file_manifest"] == source["coverage_file_manifest"]
    assert row["metrics"]["adversarial_findings"] == source["adversarial_findings"]
    assert row["verdict_class"] == "disqualified"


@pytest.mark.parametrize("name,deadline", [("terminal_cold", 180), ("strict_rows", 60)])
def test_bounded_terminal_replay_deadline(tmp_path, name, deadline):
    """REQ-VERIFY-8401: complete branch replay has a declared bounded deadline and real logs."""
    receipt = runner.terminal_child(
        name,
        [sys.executable, "-u", "-c", "print('actual child', flush=True)"],
        tmp_path / "child",
        deadline=60,
    )
    assert receipt["passed"] and receipt["deadline_s"] == deadline
    assert (
        e.load(
            e.freeze(Path(receipt["stdout_path"]).with_suffix(".receipt.json"), tmp_path / "sealed")
        )["argv"]
        == receipt["argv"]
    )


@pytest.mark.parametrize("control", ["history_verdict", "utility_mean", "owned_child"])
def test_historical_and_owned_failure_controls(natural, private, tmp_path, monkeypatch, control):
    """SCENARIO-VERIFY-8401-CONTROLS: sealed historical conclusions and child failures cannot grant readiness."""
    work = deepcopy(natural[0])
    if control == "history_verdict":
        work["history"][0]["honest_verdict"] = "complete_null_forged"
    elif control == "utility_mean":
        work["history"][0]["selected"]["H1"]["bootstrap_summary"]["all_intended"]["mean_gain"] = 1
    else:
        monkeypatch.setattr(e, "invoke", lambda *args, **kwargs: ({}, {"passed": False}))
    value = e.build(
        work, private[2]["validation_receipts"], tmp_path / "built", tmp_path / (e.NAME + ".json")
    )
    path = tmp_path / "rehashed-primitive-control.json"
    atomic_json(path, value)
    assert not e.replay(path)
