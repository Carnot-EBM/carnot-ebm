"""REQ-VERIFY-8221 / REQ-REPORT-8221: exact frozen mechanics and private CLI."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import utility_kernel_8221 as k
from carnot.verify import utility_kernel_qualification_8221 as e


def sample(p=0.8, y=0, n=16):
    """Independent private source IDs test support without implying natural data."""
    return [
        dict(
            unit_id=str(i),
            source_cluster_id=str(i),
            p=p,
            y=y,
            baseline_p=p,
            baseline_action="reject",
        )
        for i in range(n)
    ]


def test_exact_patch_and_sequential_clipping():
    """SCENARIO-VERIFY-8221-KERNEL: fixed membership and clipping survive replay."""
    rows = sample()
    model = k.fit_patches(rows)
    assert len(model["patches"]) == 4
    assert [v["group"]["name"] for v in model["patches"]] == ["global"] * 4
    assert all(v["delta"] == -0.05 for v in model["patches"])
    assert k.predict(model, rows[0]) == pytest.approx(0.6)
    stopped = k.fit_patches(sample(0.001, 0))
    assert not stopped["patches"]
    assert not k.fit_patches([])["patches"]
    assert not k.fit_patches(sample(n=7))["patches"]
    duplicates = sample()
    for row in duplicates:
        row["source_cluster_id"] = "same"
    assert not k.fit_patches(duplicates)["patches"]
    assert not k.fit_patches(rows, groups=[k.GROUPS[-1]])["patches"]
    with pytest.raises(ValueError, match="group"):
        k.fit_patches(rows, groups=[dict(name="invented", interval=None, reject_only=False)])
    with pytest.raises(ValueError, match="group"):
        k.fit_patches(rows, groups=[k.GROUPS[0], k.GROUPS[0]])
    manual = dict(
        kind="patch",
        base=dict(kind="input"),
        patches=[dict(group=k.GROUPS[0], delta=0.05), dict(group=k.GROUPS[0], delta=-0.05)],
    )
    assert k.predict(manual, sample(0.99)[0]) == 1 - 1e-6 - 0.05
    assert k.predict(manual, dict(rows[0], p=None)) is None
    for p in [1e-6, 0.1, 0.5, 1 - 1e-6]:
        good, bad = k.energies(p)
        assert good == -math.log1p(-p) and bad == -math.log(p)
        assert abs(k.rule.probability(good, bad, 1) - p) <= 1e-10
    for mode in ["original", "global", "local", "random"]:
        assert k.fit_patches(rows, mode=mode, seed=101)["kind"] == "patch"
    with pytest.raises(ValueError, match="mode"):
        k.fit_patches(rows, mode="unknown")


def test_causal_state_and_rejected_write():
    """SCENARIO-VERIFY-8221-CAUSAL: admission labels never fit candidates."""
    rows, labels = k.fixture("learnable")
    state = k.run(rows, labels, 101)
    assert k.summary(state)["passed"]
    assert len(state["issued"]) == 256 and len(state["pending"]) == 20
    assert set(state["training"]).isdisjoint(state["used_admission"])
    commits = [v for v in state["events"] if v["kind"] == "commit_candidate"]
    assert [v["slot"] for v in commits] == [64, 144]
    for event in commits:
        assert len(event["fit_ids"]) <= 64
        assert all(i <= event["slot"] - 20 for i in event["fit_ids"])
    rejected_rows, rejected_labels = k.fixture("rejected")
    rejected = k.run(rejected_rows, rejected_labels, 101)
    attempts = [v for v in rejected["events"] if v["kind"] == "admit_once"]
    assert attempts and all(v["step"] == 0 for v in attempts)
    assert all(v["before_hash"] == v["after_hash"] for v in attempts)
    for case in ["late_label", "no_signal", "single_class", "missing"]:
        r, y = k.fixture(case)
        assert k.run(r, y, 101)["cursor"] == 257
    with pytest.raises(ValueError, match="stream_schema"):
        k.run(rows[:-1], labels, 101)
    with pytest.raises(ValueError, match="baseline_hash"):
        k.run(rows, labels, 101, state=dict(state, baseline_hash="wrong"))
    mixture = dict(
        kind="mixture", base=dict(kind="input"), candidate=k.fit_patches(sample()), step=0.5
    )
    assert k.predict(mixture, sample()[0]) == pytest.approx(0.7)


def test_real_crashes_and_measurement(tmp_path):
    """SCENARIO-VERIFY-8221-CAUSAL: actual exits73 resume identical states."""
    work = e.measure(e.ROOT, tmp_path / "raw", fixture=True)
    assert all(c["passed"] for c in work["checks"])
    assert work["static"]["passed"] and work["causal"]["passed"]
    assert [r["actual_exit"] for r in work["causal"]["receipts"]] == [0, 73, 0, 73, 0]
    assert all(v["passed"] and v["pending_ids"] for v in work["causal"]["restart_state_hashes"])
    value = e.build(work, tmp_path / "raw", [dict(passed=True)], fixture=True)
    assert value["static_kernel_ready_score"] == value["causal_kernel_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["independent_count"] == value["generalized_learning_benefit_score"] == 0
    failed = e.build(work, tmp_path / "raw", [dict(passed=False)], fixture=True)
    assert failed["verdict_class"] == "disqualified" and failed["static_kernel_ready_score"] == 0
    changed = deepcopy(work)
    changed["causal"]["passed"] = False
    separated = e.build(changed, tmp_path / "raw", [dict(passed=True)], fixture=True)
    assert (
        separated["static_kernel_ready_score"] == 1 and separated["causal_kernel_ready_score"] == 0
    )


def cli(tmp_path, *args):
    """Private process execution crosses the public CLI without ambient imports."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private CLI", flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    print("after private CLI", result.returncode, flush=True)
    return result


def test_real_cli_replay_blocking_and_schema(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8221-CLI: cold replay rejects even rehashed edits."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["causal_kernel_ready_score"] == 1
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    changed = dict(value, static_kernel_ready_score=9)
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    assert not e.replay(tmp_path / "missing.json")
    ref = value["raw_shard_hashes"][0]
    path = Path(ref["path"])
    saved = path.read_bytes()
    path.write_bytes(saved + b" ")
    assert not e.replay(output)
    path.write_bytes(saved)
    assert e.replay(output)
    work_path = Path(value["measurement_reference"]["path"])
    work = json.loads(work_path.read_bytes())
    work["static"]["energy_probability_error"] = 1
    atomic_json(work_path, work)
    atomic_json(output, e.build(work, work_path.parent, value["validation_receipts"], fixture=True))
    assert not e.replay(output)
    blocked = tmp_path / "blocked" / output.name
    assert cli(tmp_path, "--root", tmp_path / "absent", "--fixture-output", blocked).returncode == 0
    blocked_value = json.loads(blocked.read_bytes())
    assert blocked_value["verdict_class"] == "blocked"
    assert blocked_value["gate_check_summary"][-1]["observed"] is None
    assert cli(tmp_path, "--date", "19000101").returncode == 2
    assert (
        cli(tmp_path, "--fixture-output", e.ROOT / "results" / "never-write.json").returncode == 2
    )
    worker = tmp_path / "worker" / "measurement.json"
    assert cli(tmp_path, "--worker-output", worker).returncode == 0
    assert json.loads(worker.read_bytes())["static"]["passed"]
    monkeypatch.setattr(e, "PROTOCOL_VALUE", {})
    measured = e.measure(e.ROOT, tmp_path / "schema", fixture=True)
    assert not all(c["passed"] for c in measured["checks"])


def test_manifest_and_receipt_preservation(tmp_path):
    """REQ-REPORT-8221: exact owned commands include typing, coverage and E2E."""
    plan = e.manifest(tmp_path, tmp_path / (e.NAME + ".json"))
    commands = {v["name"]: v for v in plan["commands"]}
    assert "--files" in commands["spec_coverage"]["argv"]
    assert "--fail-under=100" in commands["coverage_report"]["argv"]
    assert "--strict" in commands["strict_mypy"]["argv"]
    assert "patch = _exit" in (tmp_path / "coverage.ini").read_text()
    assert (
        "tests/python/test_hard_exit_learning_qualification_8206.py"
        in commands["consumer_and_E2E015_019"]["argv"]
    )
    report = tmp_path / "coverage.json"
    atomic_json(report, dict(private_fixture=True))
    receipt = e.run_check(
        e.ROOT,
        dict(
            name="coverage_json",
            argv=["/bin/true", "-o", str(report)],
            deadline_s=5,
            expected_exit=0,
        ),
        tmp_path,
        tmp_path / "logs",
    )
    assert (
        receipt["passed"]
        and Path(receipt["coverage_reference"]["path"]).read_bytes() == report.read_bytes()
    )


def test_missing_denominator_and_exact_admission_grid():
    """REQ-VERIFY-8221: missing rows preserve n and mixtures preserve final probabilities."""
    rows = sample(n=8)
    rows.append(dict(rows[0], unit_id="missing", source_cluster_id="missing", p=None))
    model = k.fit_patches(rows)
    assert model["patches"][0]["witness"]["denominator"] == 9
    assert model["patches"][0]["witness"]["numerator"] == pytest.approx(-6.4)
    for p in [-1, 2, float("nan")]:
        with pytest.raises(ValueError, match="probability"):
            k.clip(p)
    with pytest.raises(ValueError, match="model_kind"):
        k.predict(dict(kind="invalid", base=dict(kind="input")), sample()[0])
    records = sample(0.05, 0, n=10) + sample(0.49, 1, n=2)
    for row in records[:10]:
        row["baseline_action"] = "accept"
    for row in records[10:]:
        row["baseline_action"] = "escalate"
    candidate = dict(
        kind="patch", base=dict(kind="input"), patches=[dict(group=k.GROUPS[0], delta=0.05)] * 3
    )
    installed = dict(kind="input")
    accepted, receipt = k.admission(candidate, installed, records)
    assert receipt["step"] == 0.25
    assert accepted["candidate"] == candidate and accepted["base"] == installed
    assert k.predict(accepted, records[0]) == pytest.approx(0.0875)
    assert k.admission(candidate, installed, records[:3])[0] == installed
    invalid_labels = [4] * 256
    r, y = k.fixture("learnable")
    with pytest.raises(ValueError, match="stream_schema"):
        k.run(r, invalid_labels, 101)


def test_replay_receipt_state_and_checksum_failure(tmp_path):
    """SCENARIO-REPORT-8221-CLI: rebuilding headlines cannot authenticate altered trajectories."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, fixture=True)
    receipt = e.run_check(
        e.ROOT,
        dict(name="actual_pass", argv=["/bin/true"], deadline_s=5, expected_exit=0),
        tmp_path,
        raw / "logs",
    )
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, raw, [receipt], fixture=True)
    atomic_json(output, value)
    assert e.replay(output)
    atomic_json(output, dict(value, completed_count=42))
    assert not e.replay(output)
    atomic_json(output, value)
    path = Path(receipt["stdout_path"])
    saved = path.read_bytes()
    path.write_bytes(b"changed")
    assert not e.replay(output)
    path.write_bytes(saved)
    changed = deepcopy(work)
    changed["causal"]["final"]["cursor"] = 0
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [receipt], fixture=True))
    assert not e.replay(output)
    atomic_json(raw / "measurement.json", work)
    final_path = raw / "restart90/final.json"
    saved = final_path.read_bytes()
    altered = json.loads(saved)
    altered["cursor"] = 0
    atomic_json(final_path, altered)
    altered_work = deepcopy(work)
    for ref in altered_work["raw_shard_hashes"]:
        if ref["path"] == str(final_path):
            ref["sha256"] = e.reference(final_path)["sha256"]
    atomic_json(raw / "measurement.json", altered_work)
    atomic_json(output, e.build(altered_work, raw, [receipt], fixture=True))
    assert not e.replay(output)
    final_path.write_bytes(saved)
    blocked = e.measure(e.ROOT, tmp_path / "mutated", mutation="source")
    assert e.build(blocked, tmp_path / "mutated", [dict(passed=True)])["verdict_class"] == "blocked"


def test_owned_numerical_failure_disqualifies(tmp_path, monkeypatch):
    """REQ-REPORT-8221: a failed owned benchmark cannot earn readiness."""

    def fail():
        raise ValueError("private owned numerical failure")

    monkeypatch.setattr(e, "static_checks", fail)
    logs = tmp_path / "failed" / "logs"
    logs.mkdir(parents=True)
    (logs / "measurement.stdout").write_text("private live stream fixture")
    (logs / "measurement.stderr").write_text("")
    work = e.measure(e.ROOT, tmp_path / "failed", fixture=True)
    assert not any(
        Path(r["path"]).name in {"measurement.stdout", "measurement.stderr"}
        for r in work["raw_shard_hashes"]
    )
    value = e.build(work, tmp_path / "failed", [dict(passed=True)], fixture=True)
    assert value["verdict_class"] == "disqualified"
    assert value["static_kernel_ready_score"] == value["causal_kernel_ready_score"] == 0


def test_rehashed_protocol_and_fixture_input_rejection(tmp_path):
    """SCENARIO-REPORT-8221-CLI: fixed protocol and fixture inputs defeat rehashed edits."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, fixture=True)
    output = tmp_path / (e.NAME + ".json")
    receipts = [dict(passed=True)]
    changed = deepcopy(work)
    changed["protocol"]["frozen_date"] = "tampered"
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, receipts, fixture=True))
    assert not e.replay(output)
    inputs = raw / "restart-input.json"
    altered = json.loads(inputs.read_bytes())
    altered["labels"][0] = 1 - altered["labels"][0]
    atomic_json(inputs, altered)
    changed = deepcopy(work)
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(inputs):
            ref["sha256"] = e.reference(inputs)["sha256"]
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, receipts, fixture=True))
    assert not e.replay(output)
