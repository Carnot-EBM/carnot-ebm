"""REQ-REPORT-8708 / REQ-VERIFY-8708: terminal accounting preserves claim limits."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v746_capstone_inputs as e
from carnot.reporting import v746_capstone as runner
from carnot.reporting import v746_capstone_report as report


def cli(*args):
    """Run the actual standalone CLI so import and private-path guards are measured."""
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
    """SCENARIO-VERIFY-8708-REPLAY: publish a private final-byte candidate once."""
    path = tmp_path_factory.mktemp("v746") / (e.NAME + ".json")
    result = cli("--output", path, "--private-control", "--date", "20261101")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_terminal_accounting(private):
    """SCENARIO-REPORT-8708-ACCOUNTING: scheduling success cannot erase missing science."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert len(value["task_outcome_rows"]) == value["completed_count"] == 14
    assert value["verdict_class"] == "blocked"
    assert value["reconciliation_ready_score"] == 0  # A private control is not production proof.
    assert value["MODEL_SPECS"] == [] and value["current_language_model_calls"] == 0
    assert value["independent_count"] == 0
    assert set(value) <= set(value["field_principles"])
    rows = {r["experiment_id"]: r for r in value["rows"]}
    assert rows[8697]["verdict_class"] == rows[8702]["verdict_class"] == "disqualified"
    assert not rows[8702]["producer_qualified"] and rows[8702]["flagged_adversarial"]
    assert all(rows[n]["scientific_verdict"] is None for n in (8699, 8700, 8703, 8704))
    assert all(not rows[n]["producer_executed"] for n in (8699, 8700, 8703, 8704))
    assert value["paper_ready"] == all(value[g]["pass"] for g in ("G1", "G2", "G3", "G4"))
    assert value["physical_hardware_ready_score"] == 0
    assert value["arc_decision_rows"]["complete_horizon_count"] == 0
    assert value["learning_decision_rows"]["future_brier"] is None
    assert value["deployment_decision_rows"]["same_batch_rust_ratio"] is None
    assert value["historical_stamped_outcomes"]["exp8681"]["flagged_adversarial"]
    assert value["historical_stamped_outcomes"]["exp8691"]["flagged_adversarial"]
    assert value["preserved_v743_batch_one_null"]
    assert all(type(v) is int and value[k] == v for k, v in value["branch_gate_values"].items())
    assert len(value["next_decision_rows"]) == 14


@pytest.mark.parametrize("mode", ["rows", "receipts", "primitive", "missing"])
def test_actual_rehashed_rejection(private, tmp_path, mode):
    """SCENARIO-VERIFY-8708-REPLAY: rehashing a forged envelope cannot replace a seal."""
    _, original = private
    value = deepcopy(original)
    if mode == "rows":
        value["rows"][0]["producer_qualified"] = False
    elif mode == "receipts":
        value["validation_receipts"][0]["passed"] = False
    else:
        work = e.load(value["primitive_reference"])
        work["duration_s"] += 1
        target = tmp_path / "measurement.json"
        if mode == "primitive":
            e.atomic_json(target, work)
        value["primitive_reference"] = e.reference(target)
    value["reproducibility_checksum"] = e.digest(value)
    candidate = tmp_path / "candidate.json"
    e.atomic_json(candidate, value)
    assert cli("--cold-replay", candidate).returncode == 1


def test_private_guards_and_failed_child(private, tmp_path):
    """REQ-VERIFY-8708: exercise actual private success, invalid date and failed subprocess."""
    for args in [
        ["--date", "123"],
        ["--date", "20260230"],
        ["--private-control"],
        ["--cold-replay", tmp_path / "absent"],
    ]:
        assert cli(*args).returncode == 1
    alias = tmp_path / "alias"
    alias.symlink_to(e.ROOT, target_is_directory=True)
    assert (
        cli("--private-control", "--output", alias / "results" / (e.NAME + ".json")).returncode == 1
    )
    path = tmp_path / "failed" / (e.NAME + ".json")
    result = cli("--private-control", "--failed-child", "--output", path)
    assert result.returncode == 1, result.stdout + result.stderr
    value = json.loads(path.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["failed_count"] == 1
    assert value["validation_receipts"][-1]["actual_exit"] == 1


def test_gate_states(tmp_path):
    """SCENARIO-REPORT-8708-ACCOUNTING: absence, zero and changed bytes remain different."""
    path = tmp_path / "operand.json"
    assert e.observe(e.reference(path), "exp0", "ready")["operand_state"] == "missing_file"
    e.atomic_json(path, {})
    assert e.observe(e.reference(path), "exp0", "ready")["operand_state"] == "missing_field"
    e.atomic_json(path, dict(ready=0))
    assert e.observe(e.reference(path), "exp0", "ready")["operand_state"] == "observed_zero"
    ref = e.reference(path)
    e.atomic_json(path, dict(ready=1))
    assert e.observe(ref, "exp0", "ready")["operand_state"] == "changed_bytes"
    assert e.observe(e.reference(path), "exp0", "ready")["passed"]


def test_input_identity_and_original_validation(tmp_path):
    """REQ-REPORT-8708: a forged identity, pin, empty validation or changed receipt is rejected."""
    from carnot.reporting.primary_publication import publish_primary

    task = dict(id="exp1-fixture", title="fixture", deliverable="results/experiment_1_fixture.json")
    results = tmp_path / "results"
    results.mkdir()
    primitive, validation = tmp_path / "primitive.json", tmp_path / "validation.json"
    e.atomic_json(primitive, {})
    e.atomic_json(validation, dict(receipts=[dict(passed=True)]))
    path = results / "experiment_1_fixture.json"
    value = dict(
        experiment_id=1,
        task_id=task["id"],
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        primitive_reference=e.reference(primitive),
        validation_reference=e.reference(validation),
        validation_receipts=[dict(passed=True)],
        publication_output=str(path),
    )
    publish_primary(path, value, lambda p: dict(passed=True))
    assert e.authenticate(path, task)["authenticated"]
    assert not e.authenticate(path, dict(id="exp1-wrong"))["authenticated"]
    assert not e.authenticate(path, dict(id="exp8695-forged"))["authenticated"]
    e.atomic_json(validation, dict(receipts=[]))
    value["validation_reference"] = e.reference(validation)
    publish_primary(path, value, lambda p: dict(passed=True))
    assert "producer_validation_drift" in e.authenticate(path, task)["errors"]
    value["validation_receipts"] = []
    publish_primary(path, value, lambda p: dict(passed=True))
    assert "missing_owned_validation" in e.authenticate(path, task)["errors"]
    assert e.resolve(task, tmp_path / "copies", [], tmp_path)["verdict_class"] == "disqualified"
    path.unlink()
    bare = results / "experiment_1.json"
    bare.write_text("{}")
    assert e.locate(task, tmp_path) == bare
    (results / "experiment_1_wrong.json").write_text("{}")
    with pytest.raises(ValueError, match="ambiguous_primary"):
        e.locate(task, tmp_path)


def test_independent_seals_and_clocks(private, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8708-REPLAY: source, clock, absent operands and logs remain sealed."""
    path, value = private
    original = e.load(value["primitive_reference"])
    actual_load = e.load
    for mode in ("source", "clock", "log", "appeared", "snapshot"):
        work = deepcopy(original)
        if mode == "source":
            work["code_refs"] = []
        elif mode == "clock":
            work["end_ns"] += 1
        elif mode == "log":
            work["publication_receipt"]["stdout_sha256"] = "sha256:wrong"
        elif mode == "appeared":
            present = tmp_path / "appeared"
            present.write_text("new")
            work["source_refs"].append(dict(path=str(present), sha256=None))
        else:
            work["source_refs"][0]["snapshot_reference"] = dict(
                path=str(tmp_path / "absent"), sha256="wrong"
            )
        monkeypatch.setattr(
            e, "load", lambda ref: work if ref == value["primitive_reference"] else actual_load(ref)
        )
        with pytest.raises(ValueError):
            runner.replay(path)
    monkeypatch.setattr(e, "load", actual_load)
    candidate = deepcopy(value)
    candidate["validation_reference"] = dict(path="absent", sha256="wrong")
    target = tmp_path / "candidate.json"
    e.atomic_json(target, candidate)
    with pytest.raises(ValueError, match="independent_validation_seal"):
        runner.replay(target)


def test_measured_validator_retains_disjoint_children(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8708: exercise the real retention adapter with controlled child execution."""
    from coverage import CoverageData

    nodes = [e.TESTS[0] + "::test_control", *runner.E2E]

    def controlled_child(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        stdout, stderr = logs / (name + ".stdout"), logs / (name + ".stderr")
        stdout.write_text("\n".join(nodes) if name == "collect_once" else "controlled child")
        stderr.write_text("")
        if name.startswith("tests_"):
            config = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--rcfile=")))
            folder = Path(
                next(
                    s.split("=", 1)[1]
                    for s in config.read_text().splitlines()
                    if s.startswith("data_file=")
                )
            ).parent
            data = CoverageData(basename=str(folder / ".coverage.test"))
            data.add_lines({str(e.ROOT / e.OWNED[0]): {1}})
            data.write()
            selected = [n for n in nodes if n in argv]
            (folder / "junit.xml").write_text(
                "<testsuite>"
                + "".join('<testcase name="' + n.rsplit("::", 1)[-1] + '" />' for n in selected)
                + "</testsuite>"
            )
        return dict(
            name=name,
            passed=True,
            normal_exit=True,
            timed_out=False,
            actual_exit=0,
            expected_exit=0,
            stdout_path=str(stdout),
            stderr_path=str(stderr),
            stdout_sha256=e.sha256_file(stdout),
            stderr_sha256=e.sha256_file(stderr),
        )

    monkeypatch.setattr(runner, "child", controlled_child)
    receipts = runner.validate(tmp_path / "raw")
    assert all(r["passed"] for r in receipts)
    assert (
        next(r for r in receipts if r["name"] == "exact_disjoint_node_union")["node_ids"] == nodes
    )
    assert len(list((tmp_path / "raw/validation/children").glob("*/child-*.sqlite"))) == 3
    path, value = private
    real_load = e.load
    monkeypatch.setattr(
        e,
        "load",
        lambda ref: (
            dict(receipts=receipts) if ref == value["validation_reference"] else real_load(ref)
        ),
    )
    monkeypatch.setattr(report, "build", lambda *a: value)
    runner.replay(path)
    next((tmp_path / "raw/validation/children").glob("*/child-*.sqlite")).write_bytes(
        b"tampered SQLite"
    )
    with pytest.raises(ValueError, match="retained_child_evidence"):
        runner.replay(path)
    monkeypatch.setattr(e, "load", real_load)
    nodes.pop()
    with pytest.raises(ValueError, match="empty_required_shard"):
        runner.validate(tmp_path / "incomplete")
    nodes.clear()
    with pytest.raises(ValueError, match="single_exact_collection_failed"):
        runner.validate(tmp_path / "empty")


def test_production_adapter_and_time_limit(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8708: production readiness needs measured validation; timeout cannot qualify."""
    path, value = private
    work = deepcopy(e.load(value["primitive_reference"]))
    work["private_control"] = False
    assert report.build(work, [], path)["verdict_class"] == "disqualified"
    measured = [dict(name="owned", passed=True)]
    qualified = report.build(work, measured, path)
    assert qualified["reconciliation_ready_score"] == 1
    monkeypatch.setattr(runner, "measure", lambda *a: work)
    monkeypatch.setattr(runner, "validate", lambda *a: measured)
    monkeypatch.setattr(runner, "terminal", lambda *a: dict(passed=True))
    output = tmp_path / "production_adapter" / (e.NAME + ".json")
    assert runner.run(output, "20261101", False, False) == 0
    assert json.loads(output.read_bytes())["reconciliation_ready_score"] == 1


def test_measurement_budget_and_failed_publication(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8708: actual measurement rejects failed gate commands and the3600s boundary."""
    _, value = private
    work = e.load(value["primitive_reference"])
    monkeypatch.setattr(runner, "child", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="publication_gate_execution"):
        runner.measure(tmp_path / "failed_publication", "20261101", True)
    monkeypatch.setattr(runner, "child", lambda *a, **k: work["publication_receipt"])
    ticks = iter([0, 3600_000_000_000])
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        runner.measure(tmp_path / "timed_out", "20261101", True)


def test_original_method_absence_and_tampering(tmp_path):
    """SCENARIO-REPORT-8708-ACCOUNTING: original external absence stays blocked and distinct."""
    missing = e.original_inputs(tmp_path)
    assert not missing["ready"]
    assert {r["operand_state"] for r in missing["gate_check_summary"]} == {
        "missing_file",
        "missing_field",
    }
    source = tmp_path / e.authority.PROTOCOL
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("{}")
    altered = e.original_inputs(tmp_path)
    assert not altered["ready"]
    assert altered["gate_check_summary"][4]["operand_state"] == "changed_bytes"
