"""REQ-REPORT-8666 / REQ-VERIFY-8666: original evidence governs closure."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v743_capstone_inputs as e
from carnot.reporting import v743_capstone_report as report
from carnot.reporting import v743_capstone as runner


def cli(*args):
    """Exercise standalone imports and rejection paths from outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )


def test_operand_states_and_identity(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8666-ACCOUNTING: absence and zero require distinct diagnoses."""
    path = tmp_path / "missing.json"
    assert e.observe(path, "up", "ready")["operand_state"] == "missing_file"
    e.atomic_json(path, {"other": 0})
    assert e.observe(path, "up", "ready")["operand_state"] == "missing_field"
    e.atomic_json(path, {"ready": 0})
    assert e.observe(path, "up", "ready")["operand_state"] == "observed_zero"
    assert not e.observe(path, "up", "ready")["passed"]
    e.atomic_json(path, {"ready": 1})
    assert e.observe(path, "up", "ready")["passed"]
    ref = e.freeze(path, tmp_path / "frozen")
    e.verify(ref)
    path.write_text("changed")
    with pytest.raises(ValueError):
        e.verify(ref)
    task = {"id": "exp1234-a-task", "deliverable": "results/experiment_1234_declared.json"}
    monkeypatch.setattr(e, "ROOT", tmp_path)
    fallback = tmp_path / "results/experiment_1234_a_task.json"
    assert e.locate(task).name == "experiment_1234_declared.json"
    e.atomic_json(fallback, {"blocked_at_layer": "conductor_pre_gate"})
    assert e.locate(task) == fallback
    e.atomic_json(tmp_path / task["deliverable"], {})
    with pytest.raises(ValueError):
        e.locate(task)


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """REQ-VERIFY-8666: private controls use real operands without recursive tests."""
    path = tmp_path_factory.mktemp("v743") / (e.NAME + ".json")
    result = cli("--date", "20261019", "--output", path, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_accounting_and_scientific_limits(private):
    """SCENARIO-REPORT-8666-ACCOUNTING: reconcile slots without inventing science."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert len(value["rows"]) == value["intended_count"] == value["completed_count"] == 14
    assert value["reconciliation_ready_score"] == 1 and value["required_checks_passed"]
    assert value["verdict_class"] == "blocked"
    assert value["original_union_audit"]["learning"]["missing_count"] == 25
    assert value["cold_reductions"]["coverage"]["recovered_count"] == 55
    assert value["cold_reductions"]["installed"]["session_count"] == 120
    assert value["cold_reductions"]["reset"]["authentic_reset_count"] == 236
    assert not value["cold_reductions"]["reset"]["qualified"]
    assert len(value["per_task_results"]) == 14 and value["per_gap_results"]
    assert all(type(v) is int and value[k] == v for k, v in value["branch_gate_values"].items())
    assert set(value) <= set(value["field_principles"])
    assert not value["publication_authorized"] and value["MODEL_SPECS"] == []


@pytest.mark.parametrize("field", ["rows", "cold_reductions", "primitive_reference", "receipts"])
def test_rehashed_tampering(private, tmp_path, field):
    """SCENARIO-VERIFY-8666-REPLAY: local checksums cannot replace original authority."""
    _, value = private
    altered = deepcopy(value)
    if field == "primitive_reference":
        work = e.read(value[field])
        work["duration_s"] += 1
        e.atomic_json(tmp_path / "measurement.json", work)
        altered[field] = e.reference(tmp_path / "measurement.json")
    elif field == "receipts":
        altered["validation_receipts"][0]["passed"] = False
    elif field == "rows":
        altered[field][0]["eligible"] = not altered[field][0]["eligible"]
    else:
        altered[field]["coverage"]["recovered_count"] += 1
    altered["reproducibility_checksum"] = e.checksum(altered)
    e.atomic_json(tmp_path / "candidate.json", altered)
    assert cli("--cold-replay", tmp_path / "candidate.json").returncode == 1


def test_original_cause_scope(private):
    """REQ-REPORT-8666: changed callback errors do not become a qualified null."""
    _, value = private
    assert value["original_union_audit"]["installed"]["independently_verified"]
    assert value["original_union_audit"]["streaming"]["independently_verified"]
    row = next(r for r in value["per_task_results"] if r["task_id"] == "exp8664-arc-reset-transfer")
    assert row["normalized_underlying_cause"] == "duplicate_observation_clean_observer_failure"
    assert not row["normalized_cause_repeat"]
    assert value["cold_reductions"]["reset"]["historical_missing_causes"] == 236


def test_private_errors_and_owned_failure(private, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8666-REPLAY: failed owned work cannot receive readiness."""
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
    monkeypatch.setattr(runner, "measure", lambda *a: work)
    monkeypatch.setattr(runner, "validate", lambda *a: [{"name": "owned", "passed": False}])
    monkeypatch.setattr(runner, "publish_primary", lambda *a: {})
    monkeypatch.setattr(runner, "reader_receipt", lambda *a, **kw: {"passed": True})
    monkeypatch.setattr(runner.e, "atomic_json", lambda *a: None)
    assert runner.main(["--output", str(tmp_path / (e.NAME + ".json")), "--failed-child"]) == 1
    assert runner.run(tmp_path / (e.NAME + ".json"), "20261019", False, False) == 1


def test_snapshot_and_missing_obligations(tmp_path, monkeypatch):
    """REQ-VERIFY-8666: original identities survive replacement and missing receipts."""
    absent = {"path": str(tmp_path / "absent"), "exists": False, "sha256": None}
    e.verify(absent)
    with pytest.raises(ValueError, match="missing_operand"):
        e.verify(dict(absent, exists=True))
    original = tmp_path / "source.json"
    e.atomic_json(original, {"ready": 1})
    frozen = e.freeze(original, tmp_path / "raw")
    Path(frozen["snapshot_path"]).write_text("bad")
    with pytest.raises(ValueError, match="snapshot_hash"):
        e.verify(frozen)
    assert e.gates({"acceptance_gates": {"x": {"passed": False}}}) == {"x": False}
    monkeypatch.setattr(e, "ROOT", tmp_path)
    task = dict(
        id="exp1111-example",
        title="No scheduling authority",
        deliverable="results/experiment_1111_example.json",
    )
    with pytest.raises(ValueError, match="missing_slot_disposition"):
        e.resolve(task, tmp_path / "missing", [])
    ref = {"path": str(original), "sha256": "sha256:wrong"}
    refs = {}
    e.closure([ref], tmp_path / "closure", refs)
    assert not refs[str(original)]["declared_source_matches"]


def test_validation_scope_and_capture(tmp_path, monkeypatch):
    """REQ-VERIFY-8666: collection failure cannot accidentally run the repository suite."""
    from coverage import CoverageData

    nodes = [e.TESTS[0] + "::test_operand_states_and_identity", *runner.E2E]
    selected = {}
    calls = []
    empty = [False]

    def child(name, argv, logs, **kwargs):
        calls.append(name)
        logs.mkdir(parents=True, exist_ok=True)
        out = logs / (name + ".stdout")
        out.write_text("\n".join(nodes) if name == "collect_once" and not empty[0] else "")
        if name.startswith("tests_"):
            assert any(arg in nodes for arg in argv)
            xml = Path(next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--junitxml=")))
            selected[str(xml)] = [a for a in argv if a in nodes]
            config = Path(next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--rcfile=")))
            coverage = CoverageData(basename=str(config.parent / ".coverage.control"))
            coverage.add_lines({str(e.ROOT / e.OWNED[0]): {1}})
            coverage.write()
        return dict(
            name=name, passed=not (empty[0] and name == "collect_once"), stdout_path=str(out)
        )

    monkeypatch.setattr(runner, "child", child)
    monkeypatch.setattr(runner, "xml_nodes", lambda p: selected[str(p)])
    receipts = runner.validate(tmp_path / "valid")
    assert all(r["passed"] for r in receipts)
    capture = next(r for r in receipts if r["name"] == "preserved_raw_child_sqlite")
    assert {r["role"] for r in capture["shards"]} == {"owned", "consumers"}
    assert all(
        Path(r["reference"]["path"]).read_bytes().startswith(b"SQLite") for r in capture["shards"]
    )
    empty[0] = True
    calls.clear()
    receipts = runner.validate(tmp_path / "failed")
    assert not all(r["passed"] for r in receipts)
    assert not any(n.startswith("tests_") for n in calls)


def test_bounded_measurement_rejections(tmp_path, monkeypatch):
    """REQ-VERIFY-8666: bad publication execution and an exhausted measurement cap stop."""
    from types import SimpleNamespace

    log = tmp_path / "log"
    log.write_text("")
    ref = dict(e.reference(log), exists=True, snapshot_path=str(log))
    monkeypatch.setattr(e.methods, "protocol", lambda: {"task_map_v743": []})
    monkeypatch.setattr(e, "authority", lambda p: {"ready": True})
    monkeypatch.setattr(e, "freeze", lambda *a: ref)
    monkeypatch.setattr(e, "read", lambda *a: {})
    monkeypatch.setattr(runner.reduction, "reduce", lambda *a: ({}, {}, []))
    monkeypatch.setattr(runner, "child", lambda *a, **kw: {"passed": False})
    with pytest.raises(ValueError, match="publication_gate_execution"):
        runner.measure(tmp_path / "failed", "20261019")
    e.atomic_json(tmp_path / "publication.json", {"gates": {}})
    monkeypatch.setattr(
        runner,
        "child",
        lambda *a, **kw: {"passed": True, "stdout_path": str(tmp_path / "publication.json")},
    )
    ticks = iter([0, 3600_000_000_000])
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic_ns=lambda: next(ticks)))
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        runner.measure(tmp_path / "deadline", "20261019")


@pytest.mark.parametrize("mode", ["clock", "log", "reduction"])
def test_replay_owned_operand_rejections(private, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8666-REPLAY: original clocks, child logs and cold operands constrain admission."""
    path, original = private
    value = deepcopy(original)
    work = deepcopy(e.read(value["primitive_reference"]))
    read = e.read
    validation = {"receipts": value["validation_receipts"]}
    if mode == "clock":
        work["end_ns"] += 1
    if mode == "log":
        stream = tmp_path / "child.stdout"
        stream.write_text("actual output")
        receipt = dict(
            name="child", passed=True, stdout_path=str(stream), stdout_sha256="sha256:wrong"
        )
        value["validation_receipts"] = validation["receipts"] = [receipt]
    candidate = tmp_path / "candidate.json"
    e.atomic_json(candidate, value)
    monkeypatch.setattr(
        e,
        "read",
        lambda ref: (
            work
            if ref == value["primitive_reference"]
            else validation
            if ref == value["validation_reference"]
            else read(ref)
        ),
    )
    monkeypatch.setattr(e, "verify", lambda ref: None)
    monkeypatch.setattr(e.methods, "protocol", lambda: work["protocol"])
    monkeypatch.setattr(e, "authority", lambda p: work["contract"])
    monkeypatch.setattr(runner.reduction, "reduce", lambda *a: ({}, {}, []))
    with pytest.raises(
        ValueError,
        match={
            "clock": "authority_or_clock_drift",
            "log": "child_log_drift",
            "reduction": "cold_reduction_drift",
        }[mode],
    ):
        runner.replay(candidate)


@pytest.mark.parametrize("mode", ["brier", "feedback", "source", "installed", "reset"])
def test_cold_reduction_rejects_changed_science(private, tmp_path, monkeypatch, mode):
    """REQ-REPORT-8666: a changed loss, coverage count, cost cell or callback cannot be admitted."""
    from types import SimpleNamespace
    from carnot.reporting import v743_capstone_reduce as reduction

    _, value = private
    work = e.read(value["primitive_reference"])
    values = {
        str(r["experiment_id"]): e.read(r["primary_reference"])
        for r in work["slots"]
        if r["producer_executed"]
    }

    def decode(payload):
        decoded = json.loads(payload)
        if (
            mode == "brier"
            and isinstance(decoded, dict)
            and isinstance(decoded.get("rows"), list)
            and decoded["rows"]
            and "brier" in decoded["rows"][0]
        ):
            decoded["rows"][0]["brier"] += 1
        if (
            mode == "feedback"
            and isinstance(decoded, list)
            and decoded
            and "log_loss" in decoded[0]
            and "primitive" in decoded[0]
        ):
            decoded[0]["log_loss"] += 1
        if mode == "reset" and isinstance(decoded, dict) and "observer_issues" in decoded:
            return {}
        return decoded

    monkeypatch.setattr(reduction, "json", SimpleNamespace(loads=decode))
    monkeypatch.setattr(reduction.source, "inputs", lambda *a: {})
    monkeypatch.setattr(
        reduction.source, "reduce", lambda *a: {"recovered_count": -1} if mode == "source" else {}
    )
    costs = {
        k: values["8661"][k]
        for k in ("per_session_rows", "complete_operation_spans", "paired_throughput_intervals")
    }
    if mode == "installed":
        costs["per_session_rows"] = []
    monkeypatch.setattr(reduction.installed, "reduce", lambda *a: costs)
    monkeypatch.setattr(reduction, "installed_cells", lambda *a: [])
    monkeypatch.setattr(reduction.transfer, "reduce", lambda *a: {"complete_horizon_count": 1})
    with pytest.raises(
        ValueError,
        match={
            "brier": "historical_brier_reduction",
            "feedback": "historical_feedback_reduction",
            "source": "complete_source_reduction",
            "installed": "installed_cost_reduction",
            "reset": "reset_callback_reduction",
        }[mode],
    ):
        reduction.reduce(values, work["protocol"], tmp_path / "operands")


def test_authority_and_original_union_rejections(tmp_path, monkeypatch):
    """REQ-VERIFY-8666: absent original nodes and substituted authority remain fatal."""
    from carnot.reporting import v743_capstone_reduce as reduction

    with pytest.raises(ValueError, match="original_junit_union"):
        reduction.original_union({"original_node_union": {}, "validation_receipts": []}, {})
    authority = e.authority(tmp_path / "authority")
    assert authority["ready"]
    (tmp_path / "authority" / e.methods.DESIGN).write_text("altered original")
    with pytest.raises(ValueError, match="original_authority_drift"):
        e.authority(tmp_path / "authority")
    monkeypatch.setattr(e, "ROOT", tmp_path)
    candidate = tmp_path / "candidate.json"
    e.atomic_json(candidate, {"different": True})
    terminal = tmp_path / "terminal.json"
    e.atomic_json(terminal, {"candidate_reference": e.reference(candidate), "passed": True})
    primary = tmp_path / "results/experiment_1111_example.json"
    e.atomic_json(primary, {"terminal_validation_sidecar_path": str(terminal)})
    task = dict(
        id="exp1111-example", title="Example", deliverable=str(primary.relative_to(tmp_path))
    )
    with pytest.raises(ValueError, match="terminal_candidate_identity"):
        e.resolve(task, tmp_path / "snapshots", [])
