"""REQ-REPORT-8345 / REQ-VERIFY-8345: honest compact evidence and cold replay."""

from copy import deepcopy
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from carnot.reporting import v719_capstone as runner
from carnot.reporting import v719_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(root):
    root.mkdir()
    for name in [e.DESIGN, e.ACTIVE, e.PROTOCOL]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((e.ROOT / name).read_bytes())
    return root


def cli(parent, *args):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("CARNOT_8345_REPOSITORY_SUITE_RECEIPT", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_private_cli_and_rehashed_controls(tmp_path):
    """SCENARIO-VERIFY-8345-REPLAY: actual CLI checks every frozen disposition."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    result = cli(
        tmp_path, "--date", "20261009", "--root", root, "--output", output, "--private-fixture"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8332, 8346))
    assert value["missing_output_count"] == 13
    assert all(r["honest_verdict"] is None for r in value["rows"][:-1])
    assert value["verdict_class"] == "blocked"
    assert value["H1"]["intended_count"] == 128
    assert value["H2"]["retention_windows"] == [0, 32, 64, 96]
    assert value["capacity_scope"]["included_in_H1_H2"] is False
    assert value["science_ready_score"] == 0
    assert value["MODEL_SPECS"] == []
    assert not any(value["model_invocation_counts"].values())
    assert value["memory_bounds"]["growth_mb"] == 500
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    for key in ["rows", "validation_receipts", "paper_ready"]:
        changed = deepcopy(value)
        if key == "rows":
            changed[key][-1]["verdict_class"] = "positive"
        elif key == "validation_receipts":
            changed[key][0]["passed"] = False
        else:
            changed[key] = not changed[key]
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        assert cli(tmp_path, "--cold-replay", path).returncode == 1
    assert not e.replay(tmp_path / "absent")
    assert cli(tmp_path, "--date", "wrong").returncode == 2


def test_manifest_and_preflight(tmp_path, monkeypatch):
    """REQ-VERIFY-8345: owned-only coverage and immutable historical assertions."""
    plan = runner.manifest(tmp_path)
    assert {p["name"] for p in plan} >= {
        "private_E2E021",
        "private_E2E018_consumers",
        "v718_mixed_history",
    }
    assert "patch=subprocess" in (tmp_path / "coverage.ini").read_text()
    assert runner.preflight([dict(argv=["/absent/tool"])])


def test_owned_failure_and_primitive_tamper(tmp_path):
    """SCENARIO-REPORT-8345-ACCOUNTING: failed work remains disqualified."""
    work = e.measure(fixture(tmp_path / "root"), tmp_path / "raw")
    value = e.build(work, [], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    changed = deepcopy(work)
    changed["tasks"][0]["title"] = "foreign"
    with pytest.raises(ValueError, match="contract"):
        e.reduce(changed, [])
    changed = deepcopy(work)
    changed["inputs"][0]["summary"]["row"]["honest_verdict"] = "complete_null_fake"
    with pytest.raises(ValueError, match="primitive"):
        e.reduce(changed, [])


def test_current_fourteen_and_gates(tmp_path):
    """SCENARIO-REPORT-8345-ACCOUNTING: actual primaries remain byte-bound."""
    work = e.measure(e.ROOT, tmp_path / "actual")
    value = e.build(
        work, [dict(passed=True, scope="owned")], tmp_path / "actual", tmp_path / (e.NAME + ".json")
    )
    assert len(value["rows"]) == 14
    assert (
        value["actual_executed_task_count"]
        + value["pre_gate_count"]
        + value["missing_output_count"]
        == 14
    )
    assert value["science_ready_score"] == 0
    assert value["memory_measurements"]["retained_payload_bytes"] < 8_000_000
    assert value["static_ready_score"] == 1
    assert value["H1"]["sealed_predictions"]["completed_source_count"] == 97
    assert value["H1"]["sealed_predictions"]["arm_count"] == 6
    assert value["H1"]["sealed_predictions"]["evaluator_access_count"] == 0
    assert all(m["after"]["peak_rss_mb"] >= 0 for m in value["memory_measurements"]["workers"])
    changed = deepcopy(work)
    changed["references"][3]["sha256"] = "sha256:wrong"
    # A source hash mutation first fails immutable-byte authentication.
    with pytest.raises(ValueError, match="hash_drift"):
        e.reduce(changed, [])
    changed = deepcopy(work)
    changed["memory_measurements"]["parent_growth_mb"] = 501
    assert (
        e.reduce(changed, [dict(passed=True, scope="owned")])["capstone_execution_ready_score"] == 0
    )
    changed = deepcopy(work)
    changed["references"][4]["expected_sha256"] = "sha256:wrong"
    assert any(
        g.get("artifact_field") == "primitive_sha256"
        for g in e.reduce(changed, [])["gate_check_summary"]
    )
    pregate = next(i for i in work["inputs"] if i["summary"]["conductor_gates"])
    changed = deepcopy(work)
    target = next(i for i in changed["inputs"] if i["reference"] == pregate["reference"])
    target["summary"]["conductor_gates"][0]["artifact_sha256"] = "sha256:foreign"
    target["summary_sha256"] = canonical_hash(target["summary"])
    assert any(r["disposition"] == "unbound_pre_gate" for r in e.reduce(changed, [])["rows"])


def test_protocol_missing_authority_and_suite_receipt(tmp_path, monkeypatch):
    """REQ-REPORT-8345: external absence and full-suite failure remain distinct."""
    root = fixture(tmp_path / "root")
    (root / e.PROTOCOL).write_text("{}")
    suite = tmp_path / "repository_suite.json"
    atomic_json(suite, dict(passed=False, actual_exit=124, argv=["pytest", "tests/python", "-q"]))
    monkeypatch.setenv("CARNOT_8345_REPOSITORY_SUITE_RECEIPT", str(suite))
    work = e.measure(root, tmp_path / "protocol")
    assert any(
        g["artifact_field"] == "protocol_sha256" for g in e.reduce(work, [])["gate_check_summary"]
    )
    assert work["repository_suite_attempt"]["actual_exit"] == 124
    (root / e.ACTIVE).unlink()
    (root / e.DESIGN).unlink()
    absent = e.measure(root, tmp_path / "absent")
    assert not e.authority(absent)["activated"]
    (root / e.DESIGN).write_text(
        (e.ROOT / e.DESIGN).read_text().replace("exp8332-contract-replay", "exp9999-foreign")
    )
    with pytest.raises(ValueError, match="fourteen"):
        e.measure(root, tmp_path / "foreign")


def test_memory_worker_bound(tmp_path, monkeypatch):
    """REQ-VERIFY-8345: measured worker growth cannot be hidden by child isolation."""
    root = fixture(tmp_path / "root")
    task = e.parse_design((root / e.DESIGN).read_text(), milestone=e.MILESTONE)[1][0]
    item = e.reader.bind(root / task["deliverable"], tmp_path / "raw", [])
    request = tmp_path / "request.json"
    atomic_json(request, dict(task=task, item=item))
    values = iter(
        [dict(current_rss_mb=10, peak_rss_mb=10), dict(current_rss_mb=20, peak_rss_mb=511)]
    )
    monkeypatch.setattr(e, "memory", lambda: next(values))
    assert e.worker(request, tmp_path / "summary.json") == 1
    assert not json.loads((tmp_path / "summary.json").read_bytes())["memory"]["passed"]


def test_worker_failure_recovery_and_resources(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8345-REPLAY: preserve child failure and publisher recovery."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")

    def failed_manifest(private):
        return [
            dict(
                name="actual_failed_child",
                argv=[sys.executable, "-u", "-c", "raise SystemExit(1)"],
                expected=0,
                deadline=10,
                scope="owned",
            )
        ]

    audit = runner.adapter.qualified.audit
    calls = []

    def reject_once(candidate, logs, proof):
        report = audit(candidate, logs, proof)
        calls.append(candidate)
        if len(calls) == 1:
            report["passed"] = report["receipt"]["passed"] = False
        return report

    monkeypatch.setattr(runner, "manifest", failed_manifest)
    monkeypatch.setattr(runner.adapter.qualified, "audit", reject_once)
    assert runner.main(["--date", "20261009", "--root", str(root), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["capstone_execution_ready_score"] == 0
    assert len(calls) == 2
    assert e.replay(output)
    monkeypatch.setattr(
        runner.adapter.shutil, "disk_usage", lambda path: type("Disk", (), dict(free=0))()
    )
    assert runner.preflight([])[0]["artifact_field"] == "private_scratch_capacity_bytes"


def test_compact_worker_error_and_hist_receipt(tmp_path, monkeypatch):
    """REQ-VERIFY-8345: no missing worker or unmeasured history earns readiness."""
    root = fixture(tmp_path / "root")
    tasks = e.parse_design((root / e.DESIGN).read_text(), milestone=e.MILESTONE)[1]
    refs = []
    source = root / tasks[0]["deliverable"]
    atomic_json(source, {})
    item = e.reader.bind(source, tmp_path / "raw", refs)
    request = tmp_path / "request.json"
    atomic_json(request, dict(task=tasks[0], item=item))
    assert (
        cli(
            tmp_path, "--worker-request", request, "--worker-output", tmp_path / "summary.json"
        ).returncode
        == 0
    )
    assert json.loads((tmp_path / "summary.json").read_bytes())["row"]["honest_verdict"] is None
    monkeypatch.setattr(e, "child", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="owned_worker_failed"):
        e.compact(tasks[0], item, tmp_path / "failure", "absent_child")
    # Exercise the historical worker's failure path without changing its real
    # memory threshold or removing any original assertion from qualification.
    monkeypatch.setattr(
        runner.runpy,
        "run_path",
        lambda p: {
            "test_actual_mixed_history": lambda p: (_ for _ in ()).throw(
                AssertionError("failed assertion")
            )
        },
    )
    assert runner.main(["--historical-check", str(tmp_path / "history.json")]) == 1
    assert not json.loads((tmp_path / "history.json").read_bytes())["assertions_passed"]


def test_frozen_memory_and_rehashed_primitive(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8345-REPLAY: byte-bound memory and source reductions replay."""
    work = e.measure(fixture(tmp_path / "root"), tmp_path / "raw")
    receipt = e.child(
        "v718_mixed_history",
        [sys.executable, "-u", "-c", "print('memory_receipt={\"memory_passed\": false}')"],
        tmp_path / "logs",
        deadline=10,
    )
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, [receipt], tmp_path / "raw", output)
    atomic_json(output, value)
    assert value["memory_measurements"]["historical_workers"] == [dict(memory_passed=False)]
    assert value["verdict_class"] == "disqualified"
    assert e.replay(output)
    changed = deepcopy(work)
    changed["inputs"][0]["summary"]["row"]["honest_verdict"] = "complete_null_fabricated"
    changed["inputs"][0]["summary_sha256"] = canonical_hash(changed["inputs"][0]["summary"])
    changed["frozen_validation_receipts"] = [receipt]
    primitive = tmp_path / "changed_primitive.json"
    atomic_json(primitive, changed)
    ref = e.snapshot(primitive, tmp_path / "custody", "rehashed")
    mutated = e.build(changed, [receipt], tmp_path / "rehashed_build", output)
    assert mutated["rows"][0]["honest_verdict"] == "complete_null_fabricated"
    mutated["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in mutated.items() if k != "reproducibility_checksum"}
    )
    path = tmp_path / "changed.json"
    atomic_json(path, mutated)
    assert cli(tmp_path, "--cold-replay", path).returncode == 1
    original = e.compact

    def failed(*args):
        summary, _ = original(*args)
        return summary, dict(passed=False)

    monkeypatch.setattr(e, "compact", failed)
    assert not e.replay(output)
