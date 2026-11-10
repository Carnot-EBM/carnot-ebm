"""REQ-REPORT-8967 / SCENARIO-REPORT-8967-CONTRACT: retained paired audit."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import online_benefit_audit_8967 as d
from carnot.reporting import online_benefit_runner_8967 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def evidence():
    """REQ-REPORT-8967: real pinned evidence is the only scientific test input."""
    value = d.authenticate(d.ROOT)
    assert not value["failures"]
    return value


@pytest.fixture(scope="module")
def reduced(evidence):
    """REQ-REPORT-8967: reduce retained observations once without fitting anything."""
    return d.reduce(evidence)


def test_all_real_rows_and_bounds(evidence, reduced):
    """REQ-REPORT-8967: preserve the complete roster and every prespecified bound."""
    assert reduced["completed_count"] == reduced["intended_count"] == 39936
    assert reduced["independent_count"] == 24
    assert len(reduced["paired_stream_rows"]) == 24 * 9
    assert len(reduced["shared_panel_rows"]) == 128 * 6
    assert len(reduced["acceptance_gate_rows"]) == 9
    assert reduced["online_benefit_score"] == 0
    assert reduced["mechanism_diagnostics"]["feedback_positions"] == 24 * 7 * 4
    assert reduced["mechanism_diagnostics"]["frozen_future_brier_headroom"] > 0
    assert (
        reduced["multiplicity_ledger"]
        == evidence["producers"][d.METHOD_TASK]["multiplicity_ledger"]
    )
    assert all(r["conditional_on_panel_upper"] is not None for r in reduced["acceptance_gate_rows"])
    assert d.reduce(evidence) == reduced


def test_bootstrap_matches_independent_two_axis_calculation():
    """REQ-REPORT-8967: one graph draw is shared across streams and endpoints."""
    grid = np.arange(24 * 128, dtype=float).reshape(24, 128) / 100000
    sampled, conditional = d.bootstrap(grid, 766060)
    rng = np.random.default_rng(766060)
    expected, fixed = [], []
    for _ in range(10000):
        streams = rng.integers(24, size=24)
        graphs = rng.integers(128, size=128)
        expected.append(grid[streams][:, graphs].mean())
        fixed.append(grid[streams].mean())
    assert np.allclose(sampled, expected, atol=1e-15, rtol=0)
    assert np.allclose(conditional, fixed, atol=1e-15, rtol=0)
    assert not np.array_equal(sampled, conditional)


def test_gate_equalities_and_integrity():
    """REQ-REPORT-8967: strict attribution cannot pass equality or failed controls."""
    bounds = [dict(upper=-0.01, strict=False), dict(upper=0, strict=True)]
    assert d.accepts(-0.01, bounds[0])
    assert not d.accepts(0, bounds[1])
    assert d.accepts(-1e-15, bounds[1])
    assert (
        d.benefit([True] * 9, dict(feedback_positions_differ=2, frozen_future_brier_headroom=0.1))
        == 1
    )
    assert (
        d.benefit([True] * 8, dict(feedback_positions_differ=2, frozen_future_brier_headroom=0.1))
        == 0
    )
    assert (
        d.benefit([True] * 9, dict(feedback_positions_differ=0, frozen_future_brier_headroom=0.1))
        == 0
    )
    assert (
        d.benefit([True] * 9, dict(feedback_positions_differ=2, frozen_future_brier_headroom=0))
        == 0
    )


@pytest.mark.parametrize("field,value", [("rows", []), ("independent_count", 25), ("streams", [])])
def test_reconstruction_refuses_bad_roster(evidence, field, value):
    """REQ-REPORT-8967: repeated rows or restart controls cannot enlarge the census."""
    producers = dict(evidence["producers"])
    producers[d.SOURCE_TASK] = dict(producers[d.SOURCE_TASK], **{field: value})
    with pytest.raises(ValueError):
        d.reconstruct(producers)


def test_reconstruction_refuses_changed_metric(evidence):
    """REQ-REPORT-8967: rehashed aggregate values cannot replace sealed query scores."""
    producers = dict(evidence["producers"])
    source = dict(producers[d.SOURCE_TASK])
    rows = list(source["rows"])
    rows[0] = dict(rows[0], value=123)
    source["rows"] = rows
    producers[d.SOURCE_TASK] = source
    with pytest.raises(ValueError, match="query_rows"):
        d.reconstruct(producers)


@pytest.mark.parametrize("gate", [None, 0, False, True])
def test_exact_external_failure(tmp_path, monkeypatch, gate):
    """REQ-REPORT-8967: missing, zero and boolean gates report the exact field."""
    name, _, field, task = d.INPUTS[0]
    path = tmp_path / name
    value = dict(task_id=task, required_checks_passed=True)
    if gate is not None:
        value[field] = gate
    atomic_json(path, value)
    monkeypatch.setattr(d, "INPUTS", [(name, d.ref(path)["sha256"], field, task)])
    failure = d.authenticate(tmp_path)["failures"][0]
    assert failure["artifact_field"] == field
    assert failure["observed"] == gate and failure["expected"] == 1
    assert failure["check"] == (
        "missing_field" if gate is None else "observed_zero" if gate == 0 else "rejected_producer"
    )


def test_missing_malformed_and_operand_drift(tmp_path, monkeypatch):
    """REQ-REPORT-8967: original bytes remain mandatory even if a producer is rehashed."""
    name, pin, field, task = d.INPUTS[0]
    monkeypatch.setattr(d, "INPUTS", [(name, pin, field, task)])
    assert d.authenticate(tmp_path)["failures"][0]["check"] == "missing_file"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text("broken-json")
    monkeypatch.setattr(d, "INPUTS", [(name, d.ref(path)["sha256"], field, task)])
    assert d.authenticate(tmp_path)["failures"][0]["artifact_field"] == "producer_json"
    operand = tmp_path / "original.json"
    atomic_json(operand, dict(original=True))
    value = dict(
        task_id=task,
        required_checks_passed=True,
        **{field: 1},
        source_artifact_hashes=[d.ref(operand)],
        code_config_hashes=[],
        raw_shard_hashes=[],
    )
    atomic_json(path, value)
    monkeypatch.setattr(d, "INPUTS", [(name, d.ref(path)["sha256"], field, task)])
    operand.write_text("changed")
    failure = d.authenticate(tmp_path)["failures"][0]
    assert failure["path"] == str(operand) and failure["artifact_field"] == "sha256"


def test_terminal_readiness_and_retirement(evidence, reduced):
    """REQ-REPORT-8967: a qualified null is ready; only exact protocol scope retires."""
    validation = dict(
        receipts=[dict(passed=True)],
        coverage=dict(
            complete=True,
            percent=100,
            files=[dict(path=p, statements=1, missing=[], excluded=[]) for p in d.OWNED],
        ),
    )
    work = dict(evidence=evidence, result=reduced, owned_failure=False, prior_terminal={})
    ready, cls = d.disposition(work, validation, False)
    assert ready and cls == "null"
    assert d.disposition(work, validation, True) == (False, "disqualified")
    assert d.disposition(dict(work, owned_failure=True), validation, False) == (
        False,
        "disqualified",
    )
    assert d.disposition(work, dict(validation, receipts=[]), False) == (False, "disqualified")
    assert d.disposition(
        dict(work, evidence=dict(failures=[dict(check="missing_file")])), validation, False
    ) == (False, "blocked")
    result = dict(reduced, online_benefit_score=1)
    assert d.disposition(dict(work, result=result), validation, False) == (
        True,
        "circular_positive",
    )


def test_evaluation_failures_are_owned(evidence, tmp_path, monkeypatch):
    """REQ-REPORT-8967: failed reducers disqualify rather than masquerading as absence."""
    assert runner.evaluation(dict(failures=[{}]))[0]["censored_count"] == 39936
    with monkeypatch.context() as m:
        m.setattr(d, "reduce", lambda _: (_ for _ in ()).throw(ValueError("bad_owned")))
        local = dict(evidence, failures=[])
        result, failed = runner.evaluation(local)
    assert failed and result["completed_count"] == 0
    assert local["failures"][0]["check"] == "owned_reduction"


def test_private_cli_publication_replay_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8967-CONTRACT: actual foreign-directory CLI refuses changed metrics."""
    output = tmp_path / (d.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(d.ROOT / d.CLI)]
    ran = subprocess.run(
        command
        + [
            "--date",
            "20261110",
            "--private-output",
            str(output),
            "--sources",
            str(tmp_path / "missing"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked"
    assert value["online_audit_ready_score"] == value["online_benefit_score"] == 0
    assert value["intended_count"] == value["censored_count"] == len(value["rows"]) == 39936
    assert value["numerical_training_calls"] == 0
    assert (
        subprocess.run(
            command + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=120,
        ).returncode
        == 0
    )
    value["online_audit_ready_score"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]
    value["online_audit_ready_score"] = 0
    value["completed_count"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert (
        subprocess.run(
            command + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=120,
        ).returncode
        == 1
    )
    assert not runner.replay(tmp_path / "absent")["passed"]
    assert (
        subprocess.run(
            command + ["--date", "wrong"], cwd=tmp_path, env=env, capture_output=True, timeout=30
        ).returncode
        == 2
    )
    assert (
        subprocess.run(
            command
            + ["--date", "20261110", "--private-output", str(d.ROOT / "never-written.json")],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=30,
        ).returncode
        == 2
    )


def test_production_orchestration_private_scratch(tmp_path, monkeypatch):
    """REQ-REPORT-8967: exercise production validation without touching live publication."""
    root = d.ROOT
    for name in d.OWNED + d.TESTS + runner.HELPERS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    monkeypatch.setattr(d, "ROOT", tmp_path)

    def plan(private, **kwargs):
        (private / ".coverage").touch()
        return []

    monkeypatch.setattr(runner, "validation_plan", plan)

    def commands(root, plan, **kwargs):
        return [dict(passed=True, name="private_test")]

    monkeypatch.setattr(runner, "run_commands", commands)
    monkeypatch.setattr(runner, "measured", lambda *args: dict(complete=False, percent=0, files=[]))
    assert runner.main(["--date", "20261110"]) == 0
    output = tmp_path / "results" / (d.NAME + ".json")
    first = json.loads(output.read_text())
    assert first["verdict_class"] == "blocked"
    assert runner.main(["--date", "20261110"]) == 0
    assert json.loads(output.read_text())["honest_verdict"] == first["honest_verdict"]


def test_real_private_replay_refuses_rehashed_bound(tmp_path):
    """SCENARIO-REPORT-8967-CONTRACT: replay recalculates a real bound after rehashing."""
    output = tmp_path / (d.NAME + ".json")
    assert runner.main(["--date", "20261110", "--private-output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["completed_count"] == 39936 and value["fixture_only"]
    assert value["verdict_class"] == "disqualified"
    assert len(value["acceptance_gate_rows"]) == 9
    value["acceptance_gate_rows"][0]["upper"] = -1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]


def test_build_covers_null_failed_owned_and_preserved_verdict(tmp_path, evidence, reduced):
    """REQ-REPORT-8967: unchanged failure retirement preserves its original terminal name."""
    work = runner.freeze(tmp_path, tmp_path / "raw")
    work.update(evidence=dict(failures=[], references=[]), result=reduced)
    coverage = dict(
        complete=True,
        percent=100,
        files=[dict(path=p, statements=1, missing=[], excluded=[]) for p in d.OWNED],
    )
    validation = dict(receipts=[dict(passed=True)], coverage=coverage)
    value = d.build(work, validation, False)
    assert value["verdict_class"] == "null" and value["online_audit_ready_score"] == 1
    assert value["continuation_decision"]["action"] == "retire_exact_protocol"
    assert not value["continuation_decision"]["self_learning_in_general_retired"]
    failed = d.build(dict(work, owned_failure=True), validation, False)
    work["prior_terminal"] = dict(
        failed, honest_verdict="complete_disqualified_retained_exact_scope"
    )
    assert (
        d.build(dict(work, owned_failure=True), validation, False)["honest_verdict"]
        == "complete_disqualified_retained_exact_scope"
    )
