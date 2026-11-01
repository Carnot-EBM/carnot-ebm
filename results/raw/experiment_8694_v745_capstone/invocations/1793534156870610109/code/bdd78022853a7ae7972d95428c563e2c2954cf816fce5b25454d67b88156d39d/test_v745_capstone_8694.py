"""REQ-REPORT-8694 / REQ-VERIFY-8694: scheduling never substitutes for science."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v745_capstone_inputs as e
from carnot.reporting import v745_capstone as runner
from carnot.reporting import v745_capstone_report as report
from carnot.reporting import v745_capstone_reduce as reduction


def cli(*args):
    """Exercise standalone import and real children outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """SCENARIO-VERIFY-8694-REPLAY: publish a real private candidate once."""
    path = tmp_path_factory.mktemp("v745") / (e.NAME + ".json")
    result = cli("--output", path, "--private-control", "--date", "20261101")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_accounting(private):
    """SCENARIO-REPORT-8694-ACCOUNTING: stamps and literal absences survive."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert value["intended_count"] == len(value["task_outcome_rows"]) == 14
    assert value["completed_count"] == 14 and value["independent_count"] == 0
    assert value["verdict_class"] == "blocked" and value["reconciliation_ready_score"] == 1
    assert value["MODEL_SPECS"] == [] and value["current_language_model_calls"] == 0
    assert value["paper_ready"] == all(value[g]["pass"] for g in ("G1", "G2", "G3", "G4"))
    assert len(value["prd_gap_rows"]) == 3
    assert set(value) <= set(value["field_principles"])
    assert value["arc_decision_rows"]["complete_horizon_count"] == 0
    assert value["arc_decision_rows"]["censored_count"] == 16
    assert value["precision_decision_rows"][2]["passed"] is True
    rows = {r["experiment_id"]: r for r in value["rows"]}
    assert rows[8681]["flagged_adversarial"] is True
    assert rows[8691]["flagged_adversarial"] is True
    assert rows[8683]["honest_verdict"] is None
    assert "GATE_BLOCK" in rows[8683]["conductor_disposition"]
    assert rows[8692]["scientific_result_measured"] is True
    assert value["physical_hardware_ready_score"] == 0
    assert all(value[k] == n and type(n) is int for k, n in value["branch_gate_values"].items())
    assert not any(r["retired_scientific_hypothesis"] for r in value["retirement_decisions"])
    for key in ["learning_decision_rows", "semantic_decision_rows", "deployment_decision_rows"]:
        assert value[key][0]["measured"] is False


@pytest.mark.parametrize("mode", ["rows", "primitive", "receipts", "missing"])
def test_tampering(private, tmp_path, mode):
    """SCENARIO-VERIFY-8694-REPLAY: self-consistent rehashes cannot grant authority."""
    _, original = private
    value = deepcopy(original)
    if mode == "rows":
        value["rows"][0]["eligible"] = True
    elif mode == "receipts":
        value["validation_receipts"][0]["passed"] = False
    else:
        work = e.read(value["primitive_reference"])
        work["duration_s"] += 1
        p = tmp_path / "measurement.json"
        if mode == "primitive":
            e.atomic_json(p, work)
        value["primitive_reference"] = (
            e.reference(p) if p.is_file() else dict(path=str(p), sha256=None)
        )
    value["reproducibility_checksum"] = e.checksum(value)
    target = tmp_path / "candidate.json"
    e.atomic_json(target, value)
    assert cli("--cold-replay", target).returncode == 1


def test_rejection_and_owned_failure(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8694: real failed children disqualify, private aliases reject."""
    path, value = private
    work = e.read(value["primitive_reference"])
    assert report.build(work, [], path)["verdict_class"] == "disqualified"
    for args in [
        ["--date", "123"],
        ["--date", "20260230"],
        ["--private-control"],
        ["--cold-replay", tmp_path / "missing"],
    ]:
        assert cli(*args).returncode == 1
    alias = tmp_path / "checkout"
    alias.symlink_to(e.ROOT, target_is_directory=True)
    assert (
        cli("--private-control", "--output", alias / "results" / (e.NAME + ".json")).returncode == 1
    )
    monkeypatch.setattr(runner, "measure", lambda *a: work)
    monkeypatch.setattr(runner, "validate", lambda *a: [{"name": "owned", "passed": True}])
    monkeypatch.setattr(runner, "terminal", lambda *a: {"passed": True})
    failed = tmp_path / "failed" / (e.NAME + ".json")
    assert runner.run(failed, "20261101", False, True) == 1
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"


def test_gate_states_and_custody(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8694-ACCOUNTING: distinguish zero, drift and flagged authority."""
    path = tmp_path / "experiment_8681_fixture.json"
    task = dict(id="exp8681-contract-methods")
    assert e.observe(path, task, "ready")["operand_state"] == "missing_file"
    e.atomic_json(path, {})
    assert e.observe(path, task, "ready")["operand_state"] == "missing_field"
    e.atomic_json(path, dict(ready=0))
    assert e.observe(path, task, "ready")["operand_state"] == "observed_zero"
    e.atomic_json(path, dict(ready=1))
    assert (
        e.observe(path, task, "ready", expected_hash="sha256:wrong")["operand_state"]
        == "changed_bytes"
    )
    value = dict(
        experiment_id=8681,
        task_id=task["id"],
        ready=1,
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        flagged_adversarial=False,
        required_checks_passed=True,
    )
    from carnot.reporting.primary_publication import publish_primary

    path.unlink()
    pub = publish_primary(path, value, lambda candidate: {"passed": True})
    side = tmp_path / "terminal.json"
    value["terminal_validation_sidecar_path"] = str(side)
    pub = publish_primary(path, value, lambda candidate: {"passed": True})
    e.atomic_json(side, dict(publication=pub))
    assert e.custody(path, task)["sealed"]
    assert e.observe(path, task, "ready")["passed"]
    assert not e.custody(path, dict(id="exp8681-forged"))["sealed"]
    value["flagged_adversarial"] = True
    pub = publish_primary(path, value, lambda candidate: {"passed": True})
    e.atomic_json(side, dict(publication=pub))
    assert not e.observe(path, task, "ready")["passed"]


def test_resolution_and_primary_identity(tmp_path, monkeypatch):
    """REQ-REPORT-8694: unexpected siblings and undocumented absence cannot enter rows."""
    monkeypatch.setattr(e, "ROOT", tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    task = dict(
        id="exp8683-learning-union-qualification",
        deliverable="results/experiment_8683_v745_learning_union_qualification.json",
        title="Original union qualification",
    )
    with pytest.raises(ValueError, match="missing_slot_disposition"):
        e.resolve(task, tmp_path / "raw", [])
    row = e.resolve(
        task, tmp_path / "raw", ["| Original union qualification | GATE_BLOCK | upstream retired |"]
    )
    assert row["honest_verdict"] is None and not row["producer_executed"]
    (results / "experiment_8683_wrong.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected_primary_identity"):
        e.locate(task)


@pytest.mark.parametrize("mode", ["source", "clock", "log", "reduction"])
def test_replay_rejections(private, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8694-REPLAY: original clocks, logs and arithmetic stay bound."""
    _, value = private
    work = deepcopy(e.read(value["primitive_reference"]))
    read = e.read
    if mode == "source":
        work["code_refs"] = []
    elif mode == "clock":
        work["end_ns"] += 1
    elif mode == "log":
        work["publication_receipt"]["stdout_sha256"] = "sha256:wrong"
    monkeypatch.setattr(
        e, "read", lambda ref: work if ref == value["primitive_reference"] else read(ref)
    )
    monkeypatch.setattr(reduction, "reduce", lambda *a: {})
    path = tmp_path / "candidate.json"
    e.atomic_json(path, value)
    with pytest.raises(
        ValueError,
        match={
            "source": "independent_source_seal",
            "clock": "authority_or_clock_drift",
            "log": "child_log_drift",
            "reduction": "cold_reduction_drift",
        }[mode],
    ):
        runner.replay(path)


def test_validation_wrapper(tmp_path, monkeypatch):
    """REQ-VERIFY-8694: delegate bounded collection without changing original nodes."""
    monkeypatch.setattr(runner.previous, "validate", lambda raw: [{"passed": True}])
    assert runner.validate(tmp_path) == [{"passed": True}]


@pytest.mark.parametrize("mode", ["arc", "producer", "inventory", "checkpoint"])
def test_reducer_rejections(monkeypatch, mode):
    """SCENARIO-REPORT-8694-REDUCTION: missing rows and inconsistent producers reject."""
    numerical = dict(
        primitive_reference="numeric",
        validation_receipts=[],
        publication_output="/var/tmp/numeric.json",
    )
    items = dict(
        history=dict(primitive_reference="old"),
        old=dict(slots=[dict(experiment_id=8678, primary_reference="panel")]),
        panel=dict(work_reference="episodes", complete_horizon_count=0),
        episodes=dict(references=[]),
        producer=numerical,
        hardware={},
    )
    work = dict(
        history_reference="history",
        slots=[
            dict(experiment_id=n, producer_executed=True, primary_reference=r)
            for n, r in [(8692, "producer"), (8693, "hardware")]
        ],
    )
    items["numeric"] = dict(precision_rows=[], trajectory_rows=[], rows=[])
    monkeypatch.setattr(e, "read", lambda ref: deepcopy(items[ref]))
    monkeypatch.setattr(reduction.arc, "reduce", lambda episodes: {"complete_horizon_count": 0})
    monkeypatch.setattr(reduction.precision, "build", lambda *a: numerical)
    if mode == "arc":
        monkeypatch.setattr(
            reduction.arc, "reduce", lambda episodes: {"complete_horizon_count": 999}
        )
    elif mode == "producer":
        monkeypatch.setattr(reduction.precision, "build", lambda *a: {})
    elif mode == "checkpoint":
        for bits in [16, 24, 32]:
            items["numeric"]["rows"].append(dict(bits=bits, saturation=0, overflow=0))
            items["numeric"]["precision_rows"].extend(
                dict(
                    bits=bits,
                    head_index=h,
                    sequence=seq,
                    update_index=i,
                    probability_error=0,
                    decision="accept",
                    float64_decision="accept",
                    decision_parity=True,
                    finite=True,
                    local_writes=37,
                    fallback=False,
                )
                for h in range(5)
                for seq, count in [
                    ("one_step", 512),
                    ("a", 256),
                    ("b", 256),
                    ("c", 256),
                    ("d", 256),
                ]
                for i in range(count)
            )
        monkeypatch.setattr(reduction.precision, "authenticate", lambda: {"failures": ["bad"]})
    with pytest.raises(
        ValueError,
        match={
            "arc": "original_arc_reduction",
            "producer": "numerical_producer_construction",
            "inventory": "numerical_row_inventory",
            "checkpoint": "numerical_checkpoint_authentication",
        }[mode],
    ):
        reduction.reduce(work)


@pytest.mark.parametrize("mode", ["publication", "deadline"])
def test_measure_failure_paths(tmp_path, monkeypatch, mode):
    """REQ-VERIFY-8694: failed gate execution and measurement overruns cannot publish."""
    log = tmp_path / "publication.json"
    log.write_text("{}")
    ref = dict(path=str(log), snapshot_path=str(log), sha256=None, exists=True)
    monkeypatch.setattr(e, "freeze", lambda *a: dict(ref))
    monkeypatch.setattr(e, "reference", lambda *a: dict(ref))
    monkeypatch.setattr(e, "read", lambda *a: {})
    tasks = [
        dict(id=f"exp{n}-fixture", title="fixture", deliverable="fixture")
        for n in range(8681, 8695)
    ]
    monkeypatch.setattr(e, "protocol", lambda: dict(task_map_v745=tasks))
    monkeypatch.setattr(e.methods, "authority", lambda root: dict(ready=False))
    monkeypatch.setattr(
        e,
        "resolve",
        lambda task, *a: dict(
            experiment_id=int(task["id"][3:7]),
            task_id=task["id"],
            producer_executed=False,
            producer_qualified=False,
            flagged_adversarial=None,
            declared_reference=ref,
            primary_reference=ref,
            evidence_references=[],
        ),
    )
    monkeypatch.setattr(
        runner,
        "resources",
        lambda path: dict(
            disk_backed=True,
            tools_available=True,
            free_disk_bytes=2**40,
            memory_available_bytes=2**40,
        ),
    )
    monkeypatch.setattr(reduction, "reduce", lambda work: {})
    monkeypatch.setattr(
        runner, "child", lambda *a, **kw: dict(passed=mode == "deadline", stdout_path=str(log))
    )
    if mode == "deadline":
        ticks = iter([0, 3601_000_000_000])
        monkeypatch.setattr(runner.time, "monotonic_ns", lambda: next(ticks))
    with pytest.raises(
        (ValueError, TimeoutError), match="publication_gate_execution|measurement_deadline"
    ):
        runner.measure(tmp_path / "raw", "20261101")
