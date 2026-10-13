"""REQ-REPORT-8470 / REQ-VERIFY-8470: bind administrative accounting to actual evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import v728_capstone_evidence as e
from carnot.reporting import v728_capstone as runner
from carnot.reporting.current_work_receipt import atomic_json


def cli(*args, outside=False):
    """Real children verify import and publication behavior with private output paths."""
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
    root = tmp_path_factory.mktemp("v728-capstone")
    for name in [e.methods.DESIGN, e.methods.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    output = root / "results" / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return root, output, json.loads(output.read_bytes())


def test_private_accounting(private, tmp_path):
    """SCENARIO-REPORT-8470-ACCOUNTING: external absence stays blocked with fourteen slots."""
    root, output, value = private
    assert value["honest_verdict"] == "complete_blocked_v728_capstone"
    assert value["run_date"] == "20261013"
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8457, 8471))
    assert value["absent_count"] == 13 and value["producer_count"] == 0
    assert value["rows"][-1]["disposition"] == "current_reconciliation"
    assert not value["MODEL_SPECS"] and not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert set(value) <= set(value["field_principles"])
    assert e.replay(output)
    result = cli("--cold-replay", output, outside=True)
    assert result.returncode == 0, result.stdout + result.stderr
    failed = tmp_path / (e.NAME + ".json")
    result = cli("--root", root, "--output", failed, "--private-control", "--failed-child")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"
    assert cli("--date", "20260101").returncode == 2
    assert cli("--private-control").returncode == 2
    assert cli("--deliberate-error").returncode == 7
    assert cli("--cold-replay", tmp_path / "absent.json").returncode == 1


def test_rehashed_controls(private, tmp_path):
    """SCENARIO-VERIFY-8470-CONTROLS: repaired checksums cannot alter reduced science."""
    _, output, value = private
    for key, replacement in [
        ("task_count", 13),
        ("scientific_claims", {}),
        ("contract_digest", "wrong"),
    ]:
        changed = deepcopy(value)
        changed[key] = replacement
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        assert not e.replay(path)
    work = e.load(value["primitive_reference"])
    changed = deepcopy(work)
    changed["slots"][0]["disposition"] = "qualified_producer"
    path = tmp_path / "work.json"
    atomic_json(path, changed)
    tampered = deepcopy(value)
    tampered["primitive_reference"] = e.reference(path)
    tampered["reproducibility_checksum"] = e.checksum(tampered)
    candidate = tmp_path / "rehashed.json"
    atomic_json(candidate, tampered)
    assert not e.replay(candidate)
    assert e.replay(output)


def test_manifest(tmp_path):
    """REQ-VERIFY-8470: unchanged private checks and owned-only coverage are shipped."""
    plan = runner.manifest(tmp_path)
    assert {"owned_tests", "coverage_report", "private_E2E018_consumers", "private_E2E021"} <= {
        p["name"] for p in plan
    }
    assert "--fail-under=100" in next(p for p in plan if p["name"] == "coverage_report")["argv"]
    assert all(p["deadline"] <= 900 for p in plan)


def test_original_reducers(tmp_path):
    """REQ-REPORT-8470: pre-gates, derivative evidence and three boards keep producer scope."""
    import yaml

    tasks = yaml.safe_load((e.ROOT / e.methods.ACTIVE).read_bytes())["tasks"]
    for number in [8459, 8461, 8468, 8469, 8437]:
        task = next(
            t
            for t in (
                tasks
                if number >= 8457
                else yaml.safe_load((e.ROOT / e.previous.history.ACTIVE).read_bytes())["tasks"]
            )
            if t["id"].startswith(f"exp{number}-")
        )
        path = next((e.ROOT / "results").glob(f"experiment_{number}_*.json"))
        ref = e.freeze(path, tmp_path / str(number))
        row, receipt = e.invoke(task, ref, tmp_path / f"work-{number}")
        assert receipt["passed"] and row["memory"]["passed"]
        assert row["disposition"] == (
            "pre_gate_receipt"
            if number == 8459
            else "disqualified_producer"
            if number in [8461, 8468]
            else "qualified_producer"
        )
        if number == 8459:
            actual, receipt = e.invoke(task, ref, tmp_path / "real-worker")
            assert receipt["passed"] and actual["memory"]["passed"]
        if number == 8469:
            assert row["selected"]["gatemate_obligation_rows"]
        if number == 8461:
            assert not row["eligible"] and row["selected"]["objective_kernel_ready_score"] == 0
        with patch.dict(e.PINS, {number: "wrong"}):
            if number >= 8457:
                with pytest.raises(ValueError, match="primary_pin"):
                    e.outcome(task, ref, tmp_path)
    task = next(t for t in tasks if t["id"].startswith("exp8469-"))
    ref = e.freeze(next((e.ROOT / "results").glob("experiment_8469_*.json")), tmp_path)
    with pytest.raises(ValueError, match="task_identity"):
        e.outcome(dict(task, id="exp8469-wrong"), ref, tmp_path)


def test_measure_boundaries(tmp_path):
    """SCENARIO-REPORT-8470-ACCOUNTING: exact authority, resources and time bounds fail closed."""
    import yaml

    tasks = yaml.safe_load((e.ROOT / e.methods.ACTIVE).read_bytes())["tasks"]
    root = tmp_path / "root"
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True)
    log.write_text(
        "old GATE_BLOCK "
        + tasks[3]["title"][:48]
        + "\nMilestone 2026.10.728 activated\nGATE_BLOCK "
        + tasks[3]["title"][:48]
    )
    pre = next((e.ROOT / "results").glob("experiment_8459_*.json"))
    target = root / "results" / pre.name
    target.parent.mkdir()
    target.write_bytes(pre.read_bytes())
    derivative = next((e.ROOT / "results").glob("experiment_8461_*.json"))
    (target.parent / derivative.name).write_bytes(derivative.read_bytes())
    authority = dict(tasks=tasks, activated=True, gate_check_summary=[])
    with patch.object(e.methods, "authority", return_value=authority):
        work = e.measure(root, tmp_path / "raw", tmp_path)
    assert work["slots"][3]["disposition"] == "cascade_skip"
    assert work["slots"][2]["disposition"] == "pre_gate_receipt"
    with (
        patch.object(e.methods, "authority", return_value=authority),
        patch.object(e.previous, "resources", return_value=dict(disk_backed=False)),
    ):
        blocked = e.measure(root, tmp_path / "blocked", tmp_path)
    assert any(g["check"] == "required_resources" for g in blocked["checks"])
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


def test_replay_independent_pins(private, tmp_path):
    """SCENARIO-VERIFY-8470-CONTROLS: rehashed work cannot replace independent authority pins."""
    _, _, value = private
    original = e.load(value["primitive_reference"])

    def candidate(work):
        path = tmp_path / "work.json"
        atomic_json(path, work)
        changed = dict(value, primitive_reference=e.reference(path))
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / "candidate.json"
        atomic_json(path, changed)
        return path

    work = deepcopy(original)
    work["tasks"][0]["title"] += " tampered"
    assert not e.replay(candidate(work))
    work = deepcopy(original)
    work["predecessors"][0]["sha256"] = "wrong"
    assert not e.replay(candidate(work))
    work = deepcopy(original)
    row = dict(task={}, primary_reference={}, metric=1)
    work["historical_slots"] = [row]
    with patch.object(e, "invoke", return_value=({}, dict(passed=False))):
        assert not e.replay(candidate(work))
    with patch.object(e, "invoke", return_value=(dict(metric=2), dict(passed=True))):
        assert not e.replay(candidate(work))
    with (
        patch.object(e, "invoke", return_value=(dict(metric=1), dict(passed=True))),
        patch.object(e.previous, "_REPLAY", return_value=True),
    ):
        assert e.replay(candidate(work))


def test_runner_bindings(tmp_path):
    """REQ-VERIFY-8470: current date and disk scratch reach the unchanged runner."""
    with patch.object(runner, "run", return_value=0) as run:
        assert (
            runner.main(
                [
                    "--date",
                    "20261013",
                    "--root",
                    str(tmp_path),
                    "--output",
                    str(tmp_path / (e.NAME + ".json")),
                ]
            )
            == 0
        )
        assert run.call_args.args[0] == tmp_path
    with pytest.raises(SystemExit):
        runner.main(["--date"])
    with patch.object(runner, "_RUN", return_value=0) as run:
        assert runner.run(tmp_path, tmp_path / "out", tmp_path, control=True, failed=True) == 0
        assert run.call_args.kwargs == dict(control=True, failed=True)


def test_report_qualification_and_retirement(private, tmp_path):
    """REQ-REPORT-8470: owned upstream defects cannot promote effects or erase historical nulls."""
    from types import SimpleNamespace
    from carnot.reporting.v728_capstone_report import build

    _, _, value = private
    work = e.load(value["primitive_reference"])
    task = work["tasks"][4]
    ref = e.freeze(next((e.ROOT / "results").glob("experiment_8461_*.json")), tmp_path)
    with patch.object(
        e.importlib, "import_module", return_value=SimpleNamespace(replay=lambda _: False)
    ):
        summary = e.outcome(task, ref, tmp_path)
    work["slots"][4] = e.slot(task, ref, None, summary)
    prior = e.load(work["predecessors"][2])
    work["historical_slots"] = [
        r for r in prior["rows"] if r["experiment_id"] in [8432, 8435, 8436, 8437]
    ]
    result = build(
        work, value["validation_receipts"], Path(work["raw"]), Path(value["publication_output"])
    )
    assert result["scientific_claims"]["objective_qualification"]["missing_reason"]
    assert (
        result["scientific_claims"]["objective_learning"]["qualified_retrospective_benefit"] is None
    )
    assert len(result["scientific_claims"]["objective_learning"]["retention_windows"]) == 5
    assert (
        result["scientific_claims"]["V726_preserved"]["8435"]["metrics"]["gain_vs_frozen"][
            "estimate"
        ]
        < 0
    )
    assert any(
        r["task_id"] == task["id"] and r["predecessor"]["sha256"] == ref["sha256"]
        for r in result["retirement_recommendations"]
    )
    assert result["scientific_claims"]["durable_service"]["complete_journal_speedup"] is None
