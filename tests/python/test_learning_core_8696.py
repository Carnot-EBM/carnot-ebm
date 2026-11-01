"""REQ-REPORT-8696, REQ-VERIFY-8696: preserve original authority and real children."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import learning_core_8696 as e
from carnot.reporting import learning_core_runner_8696 as runner


def test_original_inputs_and_partition(tmp_path):
    """SCENARIO-REPORT-8696-CORE: original operands and remainder never change."""
    work = e.prepare(tmp_path / "raw", "20261101")
    assert not work["input_failures"]
    assert len(work["partition"]["cohorts"][0]["node_ids"]) == 39
    assert len(work["partition"]["cohorts"][1]["node_ids"]) == 25
    assert len(work["partition"]["original_runner_node_ids"]) == 27
    assert work["preconditions"]["private_disk_backed_scratch"]
    assert e.MODEL_SPECS == []
    assert e.check_bundle(work["frozen_source_bundle"])
    e.c.atomic_json(tmp_path / "work.json", work)


def test_fixture_clock_is_current_and_scientific_bytes_are_original(tmp_path):
    """REQ-VERIFY-8696: a previous boot's clock cannot become a current duration."""
    p = e.c.read(
        dict(
            path=str(e.ROOT / e.authority.original.PROTOCOL),
            sha256=e.authority.original.PROTOCOL_PIN,
        )
    )
    partition = e.c.read(p["cohort_partition_reference"])
    reference = e.c.read(partition["historical_primitive_reference"])["mechanics_fixture"]
    before = e.c.read(reference)
    lower = e.time.monotonic_ns()
    fresh = e.fresh_fixture(reference, tmp_path / "fixture")
    after = e.c.read(fresh)
    upper = e.time.monotonic_ns()
    assert lower <= after["started_monotonic_ns"] == after["ended_monotonic_ns"] <= upper
    changed = {key for key in before if before[key] != after[key]}
    assert changed <= {"raw", "started_monotonic_ns", "ended_monotonic_ns"}
    assert e.c.read(reference) == before
    for name in (
        "stream_rows.json",
        "released_feedback.json",
        "retention_rows.json",
        "coefficient_transitions.json",
    ):
        assert (Path(before["raw"]) / name).read_bytes() == (Path(after["raw"]) / name).read_bytes()
    assert after["cells"] == before["cells"]
    assert after["restart_rows"] == before["restart_rows"]


@pytest.mark.parametrize("mode", ["missing", "field", "zero", "changed", "flagged"])
def test_external_states(tmp_path, mode):
    """SCENARIO-REPORT-8696-CORE: absence and zero never share an operand state."""
    path = tmp_path / "upstream.json"
    if mode != "missing":
        value = {} if mode == "field" else dict(learning_authority_ready_score=0)
        if mode in {"changed", "flagged"}:
            value = json.loads(e.UPSTREAM.read_bytes())
            value["flagged_adversarial"] = mode == "flagged"
        path.write_text(json.dumps(value))
    work = e.prepare(tmp_path / "raw", "20261101", path)
    assert work["input_failures"]
    states = {r["operand_state"] for r in work["input_failures"]}
    assert (
        dict(missing="missing_file", field="missing_field", zero="observed_zero").get(
            mode, "changed_bytes"
        )
        in states
    )


def test_seal_and_qualification(tmp_path):
    """SCENARIO-VERIFY-8696-REPLAY: a self-rehashed replacement cannot gain authority."""
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    bundle = e.freeze_sources([e.c.ref(source)], tmp_path / "checkout")
    assert e.check_bundle(bundle)
    source.write_text("value = 2\n")
    assert not e.check_bundle(bundle)
    assert not e.check_bundle(dict(path=str(tmp_path / "absent"), sha256="wrong"))


def test_actual_child_and_failed_child(tmp_path):
    """SCENARIO-REPORT-8696-CORE: real SQLite and failed exits stay separate."""
    work = e.prepare(tmp_path / "raw", "20261101", tmp_path / "missing")
    source = e.ROOT / "python/carnot/reporting/current_work_receipt.py"
    work["sources"] = [e.c.ref(source)]
    node = "tests/python/test_current_work_receipt.py::test_canonical_hash_is_order_independent"
    # The collected inventory determines the exact existing assertion spelling.
    nodes = e.collect(work, ["tests/python/test_current_work_receipt.py"])
    node = nodes[0]
    assert e.instrument(work, "real", [node], [e.c.ref(source)], 90, "A")["passed"]
    e.recompute(work, tmp_path / "combine")
    assert work["outcomes"][0]["node_id"] == node
    assert work["children"][0]["files"]
    assert e.instrument(work, "failed", [node], [e.c.ref(source)], 0.001, "A")["timed_out"]
    assert work["child_errors"]


def test_report_failures_and_purity(tmp_path):
    """REQ-REPORT-8696: external absence blocks; owned failure takes precedence."""
    work = e.prepare(tmp_path / "raw", "20261101", tmp_path / "missing")
    runner.freeze(work)
    before = deepcopy(work)
    value = e.build(work, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked" and value["learning_core_ready_score"] == 0
    assert work == before
    assert value["training_eligible"] is False
    work["receipts"] = [dict(name="failed_owned", passed=False)]
    assert e.build(work, tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"


def test_private_cli_and_rehashed_rejection(tmp_path):
    """SCENARIO-VERIFY-8696-REPLAY: real CLI publishes blocked bytes and rejects tampering."""
    output = tmp_path / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = e.c.supervise(
        "private",
        argv
        + [
            "--date",
            "20261101",
            "--private-output",
            str(output),
            "--upstream",
            str(tmp_path / "missing"),
            "--private-control",
        ],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and runner.replay(output)
    assert runner.main(["--cold-replay", str(output)]) == 0
    for field in ["learning_core_ready_score", "duration_s", "completed_count"]:
        changed = deepcopy(value)
        changed[field] += 1
        e.c.atomic_json(output, changed)
        assert not runner.replay(output)
    work = e.c.read(value["primitive_reference"])
    work["duration_s"] += 1
    e.c.atomic_json(Path(value["primitive_reference"]["path"]), work)
    value["primitive_reference"] = e.c.ref(Path(value["primitive_reference"]["path"]))
    e.c.atomic_json(output, value)
    assert not runner.replay(output)
    assert not runner.replay(tmp_path / "missing")


@pytest.mark.parametrize(
    "args,exit_code",
    [
        (["--date", "bad"], 2),
        (["--date", "20260230"], 2),
        (["--date", "20261102"], 2),
        (["--private-control"], 2),
        (["--private-output", str(e.ROOT / "results/unsafe.json")], 2),
        (["--deliberate-error"], 7),
    ],
)
def test_cli_rejections(args, exit_code):
    """SCENARIO-VERIFY-8696-REPLAY: dates and private destinations fail before work."""
    assert runner.main(args) == exit_code


def test_execution_dispatch_and_deadline(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8696-CORE: preserve every module and its complete budget."""
    work = e.prepare(tmp_path / "raw", "20261101", tmp_path / "missing")
    monkeypatch.setattr(e, "collect", lambda *a: work["partition"]["original_node_ids"])
    calls = []
    monkeypatch.setattr(e, "instrument", lambda *a: calls.append(a))
    e.execute(work)
    assert len(calls) == 3
    assert sum(len(call[2]) for call in calls) == 39
    work["started_monotonic_ns"] = 0
    e.execute(work)
    assert not work["launch_budget_rows"][-1]["admitted"]


def test_source_drift_on_child_boundaries(tmp_path, monkeypatch):
    """REQ-VERIFY-8696: post-seal edits invalidate the affected receipt."""
    work = e.prepare(tmp_path / "raw", "20261101", tmp_path / "missing")
    monkeypatch.setattr(e, "check_bundle", lambda *a: False)
    with pytest.raises(ValueError, match="source_drift_before"):
        e.instrument(work, "before", [], [], 1, "A")
    answers = iter([True, False])
    monkeypatch.setattr(e, "check_bundle", lambda *a: next(answers))
    monkeypatch.setattr(e.core, "instrument", lambda *a: dict(passed=True))
    e.instrument(work, "after", [], [], 1, "A")
    assert work["source_reuse_refusals"]


def test_full_core_readiness_is_only_cohort_a(tmp_path):
    """REQ-REPORT-8696: all original assertions grant only bounded core readiness."""
    work = e.prepare(tmp_path / "raw", "20261101", tmp_path / "missing")
    work.update(
        input_failures=[],
        outcomes=[
            dict(node_id=n, outcome="passed") for n in work["partition"]["cohorts"][0]["node_ids"]
        ],
        original_coverage=dict(files={str(e.ROOT / e.core.old_report.RUNNER): dict(executed=[1])}),
        owned_coverage=dict(complete=True),
        children=[dict(kind="A", files=[])],
        launch_budget_rows=[dict(admitted=True)],
        receipts=[dict(name=n, passed=True) for n in e.REQUIRED],
    )
    runner.freeze(work)
    value = e.build(work, tmp_path / (e.NAME + ".json"))
    assert value["learning_core_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert not value["training_eligible"] and len(value["remainder_node_ids"]) == 25
    work["outcomes"][0]["outcome"] = "failed"
    assert e.build(work, tmp_path / (e.NAME + ".json"))["failed_count"] == 1


def test_prepare_rejects_method_partition_and_source_drift(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8696-CORE: original registration cannot be silently weakened."""
    original_read = e.c.read
    for mode in ("method", "partition", "operand"):

        def changed(ref):
            value = original_read(ref)
            if mode == "method" and "inherited_operand_checks" in value:
                value["objective_updates"] = {}
            if mode == "partition" and "cohorts" in value:
                value["original_runner_node_ids"] = []
            if mode == "operand" and "cohorts" in value:
                value["source_references"][0]["sha256"] = "wrong"
            return value

        monkeypatch.setattr(e.c, "read", changed)
        with pytest.raises(ValueError, match="original_"):
            e.prepare(tmp_path / mode, "20261101", tmp_path / "missing")
    monkeypatch.undo()
    monkeypatch.setattr(e.shutil, "disk_usage", lambda *a: type("Disk", (), {"free": 0})())
    assert any(
        r["field"] == "cpu_resources"
        for r in e.prepare(tmp_path / "disk", "20261101")["input_failures"]
    )


def test_validation_and_run_dispatch(tmp_path, monkeypatch):
    """REQ-VERIFY-8696: production measures, validates and publishes in that order."""
    work = e.prepare(tmp_path / "raw", "20261101", tmp_path / "missing")
    calls = []
    monkeypatch.setattr(e, "instrument", lambda *a: calls.append("owned"))
    monkeypatch.setattr(e.c, "supervise", lambda name, *a, **kw: dict(name=name, passed=True))
    runner.validation(work)
    assert calls == ["owned"] and len(work["receipts"]) == 6
    monkeypatch.undo()
    work["receipts"] = []
    work["input_failures"] = []
    monkeypatch.setattr(e, "prepare", lambda *a: work)
    monkeypatch.setattr(e, "execute", lambda *a: calls.append("execute"))
    monkeypatch.setattr(runner, "validation", lambda *a: calls.append("validate"))
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(output, "20261101") == 0
    assert calls[-2:] == ["execute", "validate"]
    assert runner.replay(output)
    monkeypatch.setattr(runner, "run", lambda *a, **kw: 0)
    assert runner.main([]) == 0


def test_recompute_mutations_and_receipt_bytes(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8696-REPLAY: raw configs, nodes and logs remain immutable."""
    work = e.prepare(tmp_path / "raw", "20261101", tmp_path / "missing")
    node = "tests/python/test_current_work_receipt.py::test_scenario_report_7395_current_keeps_sidecar_counts_external"
    source = e.c.ref(e.ROOT / "python/carnot/reporting/current_work_receipt.py")
    assert e.instrument(work, "owned", [node], [source], 90, "owned")["passed"]
    for mode in ("source", "operand", "config", "node", "combine"):
        changed = deepcopy(work)
        if mode == "source":
            monkeypatch.setattr(e, "check_bundle", lambda *a: False)
        elif mode == "operand":
            changed["input_references"][0]["sha256"] = "wrong"
        elif mode == "config":
            changed["children"][0]["receipt"]["coverage_config_reference"]["sha256"] = "wrong"
        elif mode == "node":
            changed["children"][0]["node_ids"] = []
        else:
            monkeypatch.setattr(
                e.c, "combine_copies", lambda *a: dict(second_combine_invariance=dict(passed=False))
            )
        with pytest.raises(ValueError):
            e.recompute(changed, tmp_path / mode)
        monkeypatch.undo()
    e.recompute(work, tmp_path / "valid")
    runner.freeze(work)
    output = tmp_path / (e.NAME + ".json")
    e.atomic_json(output, e.build(work, output))
    assert runner.replay(output)
    Path(work["receipts"][0]["stdout_path"]).write_text("changed")
    assert not runner.replay(output)
