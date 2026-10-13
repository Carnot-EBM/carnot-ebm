"""REQ-REPORT-8484 / REQ-VERIFY-8484: administrative completion grants no science credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import yaml

from carnot.reporting import v729_capstone_evidence as e
from carnot.reporting import v729_capstone as runner
from carnot.reporting.current_work_receipt import atomic_json


def cli(*args, outside=False):
    """A real child detects missing checkout imports and validates private publication bytes."""
    env = dict(os.environ)
    if outside:
        env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp" if outside else e.ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    root = tmp_path_factory.mktemp("v729-capstone")
    for name in [e.methods.DESIGN, e.methods.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    output = root / "results" / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return root, output, json.loads(output.read_bytes())


def test_accounting(private, tmp_path):
    """SCENARIO-REPORT-8484-ACCOUNTING: missing evidence never becomes a measured zero."""
    root, output, value = private
    assert value["honest_verdict"] == "complete_blocked_v729_capstone"
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8471, 8485))
    assert value["absent_count"] == 13 and value["producer_count"] == 0
    assert value["rows"][-1]["disposition"] == "current_reconciliation"
    assert value["run_date"] == "20261013" and not value["MODEL_SPECS"]
    assert not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert set(value) <= set(value["field_principles"])
    assert all(
        r["metric"] is None
        for r in value["scientific_claims"]["objective_learning"]["retention_windows"]
    )
    assert e.replay(output)
    assert cli("--cold-replay", output, outside=True).returncode == 0
    failed = tmp_path / (e.NAME + ".json")
    result = cli("--root", root, "--output", failed, "--private-control", "--failed-child")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"
    for args, code in [
        (["--date", "20260101"], 2),
        (["--private-control"], 2),
        (["--deliberate-error"], 7),
        (["--worker-request", tmp_path / "x"], 2),
        (["--cold-replay", tmp_path / "absent.json"], 1),
    ]:
        assert cli(*args).returncode == code


def test_tamper(private, tmp_path):
    """SCENARIO-VERIFY-8484-CONTROLS: a repaired checksum cannot replace independent operands."""
    _, _, value = private
    for key, replacement in [
        ("task_count", 13),
        ("scientific_claims", {}),
        ("contract_digest", "wrong"),
    ]:
        changed = dict(value, **{key: replacement})
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        assert not e.replay(path)
    work = e.load(value["primitive_reference"])
    work["tasks"][0]["title"] += " altered"
    path = tmp_path / "work.json"
    atomic_json(path, work)
    changed = dict(value, primitive_reference=e.reference(path))
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(tmp_path / "changed.json", changed)
    assert not e.replay(tmp_path / "changed.json")


def test_manifest(tmp_path):
    """REQ-VERIFY-8484: exact collected unions and genuine child coverage govern acceptance."""
    plan = runner.manifest(tmp_path)
    assert "--fail-under=100" in next(p for p in plan if p["name"] == "coverage_report")["argv"]
    for row in plan:
        assert row["deadline"] <= 900
    for suite in runner.SUITES:
        collection = next(p for p in plan if p.get("suite") == suite and "original_node_ids" in p)
        shards = [p for p in plan if p.get("suite") == suite and "shard_node_ids" in p]
        assert [n for p in shards for n in p["shard_node_ids"]] == collection["original_node_ids"]


def test_current_producer_replay(tmp_path):
    """SCENARIO-REPORT-8484-CUSTODY: current seals and readers qualify every existing branch."""
    tasks = yaml.safe_load((e.ROOT / e.methods.ACTIVE).read_bytes())["tasks"]
    for number in e.PINS:
        task = next(t for t in tasks if t["id"].startswith(f"exp{number}-"))
        ref = e.freeze(next((e.ROOT / "results").glob(f"experiment_{number}_*.json")), tmp_path)
        row, receipt = e.invoke(task, ref, tmp_path / str(number))
        assert receipt["passed"] and row["memory"]["passed"]
        assert row["producer_executed"] and row["eligible"] == (number != 8482)
        with patch.dict(e.PINS, {number: "wrong"}):
            with pytest.raises(ValueError, match="primary_pin"):
                e.outcome(task, ref, tmp_path)
        with pytest.raises(ValueError, match="task_identity"):
            e.outcome(dict(task, id=f"exp{number}-wrong"), ref, tmp_path)
    task = dict(id="exp8471-contract-methods")
    prior = next((e.ROOT / "results").glob("experiment_8459_*.json"))
    ref = e.freeze(prior, tmp_path)
    with patch.dict(e.PINS, {8471: ref["sha256"][7:]}):
        assert e.outcome(task, ref, tmp_path)["disposition"] == "pre_gate_receipt"
    task = next(t for t in tasks if t["id"].startswith("exp8472-"))
    ref = e.freeze(next((e.ROOT / "results").glob("experiment_8472_*.json")), tmp_path)
    row, receipt = e.invoke(task, ref, tmp_path / "pregate")
    assert receipt["passed"] and row["disposition"] == "pre_gate_receipt"
    authority = e.methods.protocol()["historical_authorities"]["V726"][1]
    tasks = yaml.safe_load(Path(authority["path"]).read_bytes())["tasks"]
    task = next(t for t in tasks if t["id"].startswith("exp8437-"))
    ref = e.freeze(next((e.ROOT / "results").glob("experiment_8437_*.json")), tmp_path)
    assert e.outcome(task, ref, tmp_path)["eligible"]


def test_measure_limits(tmp_path):
    """REQ-VERIFY-8484: exact authority, resources, scheduling and time bounds are explicit."""
    tasks = yaml.safe_load((e.ROOT / e.methods.ACTIVE).read_bytes())["tasks"]
    root = tmp_path / "root"
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True)
    log.write_text(
        "old GATE_BLOCK "
        + tasks[2]["id"]
        + "\nMilestone 2026.10.729 activated\nGATE_BLOCK "
        + tasks[3]["id"]
    )
    results = root / "results"
    results.mkdir()
    for n in [8471, 8437]:
        path = next((e.ROOT / "results").glob(f"experiment_{n}_*.json"))
        (results / path.name).write_bytes(path.read_bytes())
    authority = dict(tasks=tasks, activated=False, gate_check_summary=[])
    with patch.object(e.methods, "authority", return_value=authority):
        work = e.measure(root, tmp_path / "raw", tmp_path)
    assert work["slots"][0]["disposition"] == "qualified_producer"
    assert work["slots"][3]["disposition"] == "cascade_skip"
    assert work["slots"][2]["disposition"] == "absent_artifact"
    assert work["historical_slots"][0]["experiment_id"] == 8437
    altered_protocol = deepcopy(e.methods.protocol())
    altered_protocol["historical_authorities"]["V728"][1]["sha256"] = "wrong"
    with (
        patch.object(e.methods, "authority", return_value=authority),
        patch.object(e.history.previous, "resources", return_value=dict(disk_backed=False)),
        patch.object(e.methods, "protocol", return_value=altered_protocol),
    ):
        blocked = e.measure(root, tmp_path / "blocked", tmp_path)
    assert any(g["check"] == "required_resources" for g in blocked["checks"])
    assert any(g["check"] == "original_authority_pin" for g in blocked["checks"])
    with patch.object(
        e.methods,
        "authority",
        return_value=dict(authority, tasks=[*tasks[:-1], dict(tasks[-1], requires=["no"])]),
    ):
        with pytest.raises(ValueError, match="exact_unconditional"):
            e.measure(root, tmp_path / "bad", tmp_path)
    with (
        patch.object(e.methods, "authority", return_value=authority),
        patch.object(e.time, "monotonic_ns", side_effect=[0, 3600_000_000_000]),
    ):
        with pytest.raises(TimeoutError, match="measurement_deadline"):
            e.measure(root, tmp_path / "deadline", tmp_path)


def test_independent_replay_rejection(private, tmp_path):
    """SCENARIO-VERIFY-8484-CONTROLS: independent historical pins survive local rehashing."""
    _, _, value = private

    def candidate(work):
        path = tmp_path / "work.json"
        atomic_json(path, work)
        changed = dict(value, primitive_reference=e.reference(path))
        changed["reproducibility_checksum"] = e.checksum(changed)
        target = tmp_path / "candidate.json"
        atomic_json(target, changed)
        return target

    work = e.load(value["primitive_reference"])
    work["predecessors"][0]["sha256"] = "wrong"
    assert not e.replay(candidate(work))
    work = e.load(value["primitive_reference"])
    work["historical_slots"] = [dict(task={}, primary_reference={}, metric=1)]
    with patch.object(e, "invoke", return_value=({}, dict(passed=False))):
        assert not e.replay(candidate(work))
    with patch.object(e, "invoke", return_value=(dict(metric=2), dict(passed=True))):
        assert not e.replay(candidate(work))
    with (
        patch.object(e, "invoke", return_value=(dict(metric=1), dict(passed=True))),
        patch.object(e.base, "replay", return_value=True),
    ):
        assert e.replay(candidate(work))


def test_report_preserves_scopes(private):
    """REQ-REPORT-8484: historical nulls and current failed qualification remain separate."""
    from carnot.reporting.v729_capstone_report import build

    _, _, value = private
    work = e.load(value["primitive_reference"])
    original = e.load(work["predecessors"][2])
    work["historical_slots"] = [
        r for r in original["rows"] if r["experiment_id"] in [8432, 8435, 8436, 8437]
    ]
    prior = e.load(work["predecessors"][-1])
    work["historical_slots"] += [
        r for r in prior["rows"] if r["experiment_id"] in [8464, 8465, 8466]
    ]
    task = work["tasks"][0]
    ref = e.freeze(
        next((e.ROOT / "results").glob("experiment_8471_*.json")),
        Path(work["raw"]) / "test-current",
    )
    work["slots"][0] = e.slot(task, ref, None, e.outcome(task, ref, Path(work["raw"])))
    result = build(
        work, value["validation_receipts"], Path(work["raw"]), Path(value["publication_output"])
    )
    assert (
        result["scientific_claims"]["historical_journal_null"]["journal_speed_benefit_score"] == 0
    )
    assert result["scientific_claims"]["V717_closed"]["H1"] == -0.00390625
    assert (
        result["scientific_claims"]["V726_preserved"]["8435"]["original_verdict"]
        == original["scientific_claims"]["bounded_feedback_learning"]["original_verdict"]
    )
    assert any(r["task_id"] == task["id"] for r in result["retirement_recommendations"])


def test_complete_child_budget(tmp_path):
    """REQ-VERIFY-8484: whole replay gets a bounded budget; individual workers remain unchanged."""
    with patch.object(runner, "_CHILD", return_value=dict(passed=True)) as child:
        assert runner.complete_child("terminal_cold", ["python"], tmp_path, deadline=60)["passed"]
        assert child.call_args.kwargs["deadline"] == 600
        assert runner.complete_child("other", ["python"], tmp_path, deadline=30)["passed"]
        assert child.call_args.kwargs["deadline"] == 30
