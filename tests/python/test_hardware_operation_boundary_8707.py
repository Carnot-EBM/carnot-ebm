"""REQ-REPORT-8707, REQ-VERIFY-8707: separate CPU mapping and physical evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import hardware_operation_inputs_8707 as i
from carnot.reporting import hardware_operation_boundary_8707 as e
from carnot.reporting import hardware_operation_runner_8707 as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child


def producer(tmp_path):
    """Use real atomic publication so terminal identity has independent checked bytes."""
    output = tmp_path / "results/experiment_8706_v746_spline_lifecycle_stress.json"
    value = dict(
        experiment_id=8706,
        task_id="exp8706-spline-lifecycle-stress",
        verdict_class="circular_positive",
        honest_verdict="complete_circular_positive_fixture",
        required_checks_passed=True,
        flagged_adversarial=False,
        acceptance_gates={"owned": True},
        adversarial_findings=[],
        lifecycle_rows_ready_score=1,
        numerical_lifecycle_ready_score=1,
        code_config_hashes=[],
        source_artifact_hashes=[],
        raw_shard_hashes=[],
    )
    publish_primary(output, value, lambda p: dict(passed=True))
    return output, value


def test_operand_states_and_terminal(tmp_path):
    """REQ-REPORT-8707: absence, zero, changed bytes and historical flags differ."""
    path = tmp_path / "missing.json"
    assert i.gate(i.reference(path), "x", 1, "p")["operand_state"] == "missing_file"
    for value, state in [
        ({}, "missing_field"),
        ({"x": 0}, "observed_zero"),
        ({"x": 1}, "observed"),
        ({"x": True}, "observed"),
    ]:
        atomic_json(path, value)
        row = i.gate(i.reference(path), "x", 1, "p")
        assert row["operand_state"] == state
        assert row["passed"] == (type(value.get("x")) is int and value.get("x") == 1)
    ref = i.reference(path)
    atomic_json(path, dict(x=2))
    assert i.gate(ref, "x", 1, "p")["operand_state"] == "changed_bytes"
    output, value = producer(tmp_path)
    assert i.terminal(i.reference(output), value)
    side = (
        output.parent
        / "raw"
        / output.stem
        / "validators"
        / (i.reference(output)["sha256"][7:] + ".json")
    )
    atomic_json(
        side, dict(primary_path=str(output), primary_sha256="bad", report=dict(passed=True))
    )
    with pytest.raises(ValueError):
        i.terminal(i.reference(output), value)
    atomic_json(
        side,
        dict(
            primary_path="foreign",
            primary_sha256=i.reference(output)["sha256"],
            report=dict(passed=True),
        ),
    )
    with pytest.raises(ValueError, match="producer_terminal_acceptance"):
        i.terminal(i.reference(output), value)


def test_authority_and_current_inputs(tmp_path):
    """SCENARIO-REPORT-8707-OPERATIONS: authentic lifecycle survives missing siblings."""
    contract = i.authority(i.ROOT)
    assert contract["ready"] and len(contract["tasks"]) == 14
    inputs = i.authenticate(i.ROOT)
    assert inputs["board_qualified"], inputs["failures"]
    branches = inputs["branches"]
    assert [b["qualified"] for b in branches] == [False, False, True, True]
    assert branches[0]["scheduling_disposition"]["science_evidence"] is False
    assert len(inputs["lifecycle_cost_rows"]) == 715
    assert len(inputs["historical_cost_rows"]) == 1260
    assert not i.authenticate(tmp_path)["board_qualified"]
    with pytest.raises((OSError, ValueError)):
        i.authority(tmp_path)


def test_admission_rejections(tmp_path, monkeypatch):
    """REQ-REPORT-8707: a score cannot replace producer construction and original checks."""
    path, value = producer(tmp_path)
    monkeypatch.setattr(i, "producer_operands", lambda v, p: [])
    task = dict(id=value["task_id"], deliverable=str(path.relative_to(tmp_path)))
    assert i.admit(tmp_path, task, "lifecycle_rows_ready_score")["qualified"]
    for changes in [
        dict(flagged_adversarial=True),
        dict(required_checks_passed=False),
        dict(acceptance_gates={"owned": False}),
        dict(verdict_class="blocked"),
        dict(lifecycle_rows_ready_score=0),
        dict(adversarial_findings=[dict(severity="warning")]),
    ]:
        publish_primary(path, {**value, **changes}, lambda p: dict(passed=True))
        assert not i.admit(tmp_path, task, "lifecycle_rows_ready_score")["qualified"]
    publish_primary(path, value, lambda p: dict(passed=True))
    publish_primary(
        path,
        {
            **value,
            "acceptance_gates": {"owned": True, "numerical_criteria": False},
            "numerical_lifecycle_ready_score": 0,
        },
        lambda p: dict(passed=True),
    )
    assert i.admit(tmp_path, task, "lifecycle_rows_ready_score")["qualified"]
    assert not i.admit(tmp_path, task, "numerical_lifecycle_ready_score")["qualified"]
    publish_primary(path, value, lambda p: dict(passed=True))
    monkeypatch.setattr(
        i, "producer_operands", lambda v, p: (_ for _ in ()).throw(ValueError("bad seal"))
    )
    assert not i.admit(tmp_path, task, "lifecycle_rows_ready_score")["qualified"]


def test_costs_and_build(tmp_path):
    """SCENARIO-REPORT-8707-OPERATIONS: descriptive CPU spans never imply fabric speed."""
    work = r.measure(tmp_path / "raw" / e.NAME / "invocations" / "unit", "20261101", i.ROOT, False)
    saved = deepcopy(work)
    value = e.build(work, [dict(passed=True)], tmp_path / (e.NAME + ".json"))
    assert work == saved
    assert value["hardware_mapping_ready_score"] == 1
    assert value["physical_hardware_ready_score"] == 0 and value["verdict_class"] == "blocked"
    assert len(value["operation_support_rows"]) == len(e.OPERATIONS) * 5
    assert (
        len({(x["board"], x["operation"]) for x in value["operation_support_rows"]})
        == len(e.OPERATIONS) * 5
    )
    assert all(x["conditional_ceiling"] is None for x in value["conditional_acceleration_bounds"])
    assert value["preserved_quarantines"]["Exp8681"]["flagged_adversarial"]
    assert (
        e.build(work, [dict(passed=False)], tmp_path / (e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )
    assert e.bound(100, 50, True)["conditional_ceiling"] == 2
    assert e.bound(100, 50, False)["conditional_ceiling"] is None
    assert e.bound(None, None, False)["compatible_fraction"] is None
    assert e.bound(100, 100, True)["conditional_ceiling"] is None
    for total, part in [(0, 0), (10, 11), (10, -1)]:
        with pytest.raises(ValueError):
            e.bound(total, part, True)


def test_actual_private_cli_replay_and_rejections(tmp_path):
    """SCENARIO-VERIFY-8707-REPLAY: real child success and independent-seal rejection."""
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    logs = tmp_path / "logs"
    assert child(
        "private",
        argv + ["--private-control", "--root", str(tmp_path / "absent"), "--output", str(output)],
        logs,
    )["passed"]
    assert child("replay", argv + ["--cold-replay", str(output)], logs)["passed"]
    original = json.loads(output.read_bytes())
    raw = Path(original["primitive_reference"]["path"]).parent
    work_path = raw / "measurement.json"
    work_bytes = work_path.read_bytes()
    work = json.loads(work_bytes)
    work["run_date"] = "forged"
    atomic_json(work_path, work)
    atomic_json(output, e.build(work, original["validation_receipts"], output))
    assert child("rehashed", argv + ["--cold-replay", str(output)], logs, expected=1)["passed"]
    work_path.write_bytes(work_bytes)
    atomic_json(output, original)
    work_path.unlink()
    assert child("missing", argv + ["--cold-replay", str(output)], logs, expected=1)["passed"]
    assert child("date", argv + ["--date", "bad"], logs, expected=1)["passed"]
    assert child("guard", argv + ["--private-control"], logs, expected=1)["passed"]
    assert r.main(["--date", "20260230"]) == 1


def test_terminal_adapter_and_run_bindings(tmp_path, monkeypatch):
    """REQ-VERIFY-8707: retain the real adapter while controlling delegated exits."""
    reports = []

    def fake_child(name, argv, logs, **kwargs):
        path = tmp_path / (name + ".json")
        atomic_json(path, dict(flagged_count=0, reports=[]))
        reports.append(argv)
        return dict(passed=True, stdout_path=str(path))

    monkeypatch.setattr(r, "child", fake_child)
    assert r.terminal(tmp_path / "candidate.json", tmp_path / "terminal")["passed"]
    assert len(reports) == 3
    monkeypatch.setattr(
        r, "measure", lambda *a: dict(raw=str(tmp_path / "raw" / e.NAME / "invocations" / "fake"))
    )
    monkeypatch.setattr(r, "validate", lambda *a: [dict(passed=True)])
    monkeypatch.setattr(
        e,
        "build",
        lambda *a: dict(
            completed_count=0,
            censored_count=4,
            required_checks_passed=True,
            verdict_class="blocked",
            hardware_mapping_ready_score=0,
        ),
    )
    monkeypatch.setattr(r, "publish_primary", lambda *a: dict(primary_sha256="sha256:fixture"))
    monkeypatch.setattr(r, "reader_receipt", lambda *a, **k: dict(passed=True))
    monkeypatch.setattr(i, "ROOT", tmp_path)
    assert r.run(tmp_path / "results" / (e.NAME + ".json"), "20261101", tmp_path, False) == 0
    monkeypatch.setattr(r, "run", lambda *a: 0)
    assert r.main([]) == 0


def test_validation_custody_with_failed_child(tmp_path, monkeypatch):
    """REQ-VERIFY-8707: a failed child stays failed even when SQLite survives cleanup."""
    import configparser
    from coverage import CoverageData

    calls = []

    def controlled(name, argv, logs, **kwargs):
        calls.append((name, argv))
        logs.mkdir(parents=True, exist_ok=True)
        out = logs / (name + ".stdout")
        out.write_text(
            e.TESTS[0]
            + "::test_operand_states_and_terminal\n"
            + "tests/python/test_primary_publication_7928.py::test_fixture\n"
        )
        if name.startswith("tests_"):
            data_path = Path(
                next(a.split("=", 1)[1] for a in argv if a.startswith("COVERAGE_FILE="))
            )
            data = CoverageData(basename=str(data_path) + ".unit")
            data.add_lines({str(i.ROOT / e.OWNED[0]): {1}})
            data.write()
        if name == "combine":
            parser = configparser.ConfigParser()
            parser.read(next(a.split("=", 1)[1] for a in argv if a.startswith("--rcfile=")))
            data = CoverageData(basename=parser["run"]["data_file"])
            data.add_lines({str(i.ROOT / e.OWNED[0]): {1}})
            data.write()
        if name == "coverage_json":
            atomic_json(Path(argv[-1]), dict(totals=dict(percent_covered=100)))
        return dict(name=name, passed=name != "tests_consumers", stdout_path=str(out))

    monkeypatch.setattr(r, "child", controlled)
    receipts = r.validate(tmp_path / "raw")
    assert not next(x for x in receipts if x["name"] == "tests_consumers")["passed"]
    assert receipts[1]["references"] and receipts[2]["references"]
    assert all(Path(ref["path"]).is_file() for row in receipts for ref in row.get("references", []))
    assert not set(receipts[1]["collected_nodes"]) & set(receipts[2]["collected_nodes"])
    assert next(x for x in receipts if x.get("name") == "retained_final_sqlite")["passed"]


@pytest.mark.parametrize("mode", ["board_flag", "method", "contract", "head_flag", "costs"])
def test_original_authority_rejections(tmp_path, monkeypatch, mode):
    """REQ-REPORT-8707: independent original flags, registration and object identity bind."""
    original = i.load

    def changed(ref):
        value = original(ref)
        if mode == "board_flag" and value.get("experiment_id") == 8665:
            value["flagged_adversarial"] = True
        if mode == "method" and "objective_updates" in value:
            value["objective_updates"]["learning_rate"] = 0.2
        if mode == "head_flag" and value.get("experiment_id") == 8431:
            value["flagged_adversarial"] = True
        return value

    monkeypatch.setattr(i, "load", changed)
    if mode == "costs":
        from carnot.reporting import hardware_boundary_costs_8679 as costs

        monkeypatch.setattr(
            costs, "source_cells", lambda *a: (_ for _ in ()).throw(ValueError("missing endpoints"))
        )
    if mode == "contract":
        monkeypatch.setattr(
            i, "authority", lambda root: (_ for _ in ()).throw(ValueError("changed contract"))
        )
    monkeypatch.setattr(i, "admit", lambda *a: dict(refs=[], value={}, qualified=False))
    assert i.authenticate(i.ROOT)["failures"]


def test_original_seals_and_construction_rejections(monkeypatch):
    """SCENARIO-VERIFY-8707-REPLAY: original seals outrank a consistently rehashed report."""
    from carnot.reporting import spline_lifecycle_study_8706 as original

    path = i.ROOT / "results/experiment_8706_v746_spline_lifecycle_stress.json"
    value = json.loads(path.read_bytes())
    real = i.load

    def altered(ref):
        doc = real(ref)
        if "/seals/" in ref["path"]:
            doc["measurement"] = {}
        return doc

    monkeypatch.setattr(i, "load", altered)
    with pytest.raises(ValueError, match="original_independent_seal"):
        i.producer_operands(value, path)
    monkeypatch.setattr(i, "load", real)
    monkeypatch.setattr(original, "build", lambda *a: {})
    with pytest.raises(ValueError, match="original_producer_construction"):
        i.producer_operands(value, path)
    with pytest.raises(ValueError, match="changed_bytes"):
        i.load_bytes(dict(path=str(path), sha256="changed"))


def test_authority_digest_rejection(monkeypatch):
    """REQ-REPORT-8707: matching tables cannot override the independently fixed digest."""
    monkeypatch.setattr(i, "PIN", "0" * 64)
    with pytest.raises(ValueError, match="v746_full_object_authority"):
        i.authority(i.ROOT)


@pytest.mark.parametrize("mode", ["clean", "source", "clock", "validation", "report"])
def test_cold_replay_separate_checks(tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8707-REPLAY: source, clock, validation and reduction bind independently."""
    output = tmp_path / (e.NAME + ".json")
    raw = tmp_path / "raw" / e.NAME / "invocations" / "replay"
    work = r.measure(raw, "20261101", tmp_path / "absent", True)
    retained = raw / "retained.txt"
    retained.write_text("real retained bytes")
    receipts = [
        dict(
            passed=True,
            references=[i.reference(retained)],
            stdout_path=str(retained),
            stdout_sha256=i.reference(retained)["sha256"],
        )
    ]
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    atomic_json(
        raw.parents[1] / "validation_seals" / "replay.json",
        dict(validation=i.reference(raw / "validation.json")),
    )
    if mode in {"source", "clock"}:
        if mode == "clock":
            work["duration_s"] = -1
        atomic_json(raw / "measurement.json", work)
        atomic_json(
            raw.parents[1] / "seals" / "replay.json",
            dict(
                measurement=i.reference(raw / "measurement.json"),
                code_refs=[] if mode == "source" else work["code_refs"],
            ),
        )
    value = e.build(work, receipts, output)
    if mode == "validation":
        value["validation_reference"] = dict(path="foreign", sha256="foreign")
    if mode == "report":
        value["hardware_mapping_ready_score"] = 99
    atomic_json(output, value)
    if mode == "clean":
        r.replay(output)
    else:
        with pytest.raises(ValueError):
            r.replay(output)


def test_resource_failure_deadline_and_output_guards(tmp_path, monkeypatch):
    """REQ-VERIFY-8707: resource absence and expired measurement never grant readiness."""

    def unavailable(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        out = logs / "filesystem.stdout"
        out.write_text("tmpfs tmpfs")
        return dict(passed=False, stdout_path=str(out))

    monkeypatch.setattr(r, "child", unavailable)
    raw = tmp_path / "raw" / e.NAME / "invocations" / "resource"
    work = r.measure(raw, "20261101", tmp_path / "absent", True)
    assert work["resource_failures"]
    with pytest.raises(ValueError, match="fixed"):
        r.run(tmp_path / "foreign.json", "20261101", i.ROOT, False)
    with pytest.raises(ValueError, match="outside checkout"):
        r.run(i.ROOT / "results" / (e.NAME + ".json"), "20261101", i.ROOT, True)
    clocks = iter([0, 3601 * 10**9])
    monkeypatch.setattr(r.time, "monotonic_ns", lambda: next(clocks))
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        r.measure(raw, "20261101", tmp_path / "absent", True)
