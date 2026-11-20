"""REQ-REPORT-8987-ADMISSION: development admission never implies fresh evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from carnot.reporting import spline_development_admission_8987 as d
from carnot.reporting import spline_development_runner_8987 as runner
from carnot.reporting.current_work_receipt import atomic_json

pytestmark = pytest.mark.memory_watchdog_skip


def test_four_fixed_gates_and_separate_readiness():
    """SCENARIO-REPORT-8987-READINESS: each fixed margin can independently close collection."""
    rows = []
    for arm, probabilities in {
        "spline": [0.1, 0.9],
        "converged_linear": [0.2, 0.8],
        "shared_logistic": [0.3, 0.7],
        "prevalence": [0.5, 0.5],
    }.items():
        for i, p in enumerate(probabilities):
            rows.append(
                dict(
                    unit_id=str(i),
                    arm=arm,
                    condition="exposed_development_guard",
                    error=i,
                    p_error=p,
                    value=(p - i) ** 2,
                    numerator=(p - i) ** 2,
                )
            )
    metrics, gates = d.admission(rows)
    assert all(g["passed"] for g in gates) and len(gates) == 4
    assert metrics["auroc"] == 1
    for name in ("auroc", "energy_brier"):
        bad = deepcopy(rows)
        for row in bad:
            if row["arm"] == "spline":
                row["p_error"] = 1 - row["p_error"]
                row["value"] = row["numerator"] = (row["p_error"] - row["error"]) ** 2
        assert not all(g["passed"] for g in d.admission(bad)[1])
    with pytest.raises(ValueError, match="primitive"):
        d.admission([dict(r, numerator=99) for r in rows])


def test_trace_keeps_initial_and_every_iteration(monkeypatch):
    """REQ-REPORT-8987-ADMISSION: record actual derivatives, never optimizer success alone."""
    trace = []
    phi = np.column_stack([np.ones(8), np.arange(8) / 8])
    y = np.asarray([0] * 4 + [1] * 4, dtype=float)
    result = d.traced_minimize(
        d.n.objective,
        np.zeros(2),
        args=(phi, y),
        method="L-BFGS-B",
        jac=True,
        options=d.n.OPTIONS,
        trace=trace,
    )
    assert len(trace) == result.nit + 1
    assert [r["iteration"] for r in trace] == list(range(result.nit + 1))
    assert trace[-1]["gradient_inf"] <= 1e-6


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8987-ADMISSION: natural cached inputs and real bounded construction."""
    return runner.freeze(d.ROOT, tmp_path_factory.mktemp("8987-work"))


def test_natural_fits_and_bound_manifests(work):
    """REQ-REPORT-8987-ADMISSION: six fits share320 graphs; future1024 carry no responses."""
    assert not work["owned_failure"] and not work["evidence"]["failures"]
    result = work["result"]
    assert len(result["head_rows"]) == 6
    assert all(h["gradient_inf"] <= 1e-6 for h in result["head_rows"])
    assert len(result["optimizer_rows"]) == sum(h["iterations"] + 1 for h in result["head_rows"])
    assert result["construction"]["complete"]
    assert len(result["graph_manifest"]) == 1024
    assert len({r["stream_id"] for r in result["role_manifest"] if r["role"] == "online"}) == 24
    assert all(
        r["trained_state_sha256"] == result["trained_state_reference"]["sha256"]
        for r in result["role_manifest"]
    )
    d.verify_result(work["evidence"], result)
    for field in ("development_metrics", "development_admission_rows"):
        bad = deepcopy(result)
        if field == "development_metrics":
            bad[field]["auroc"] = 0
        else:
            bad[field][0]["passed"] = False
        with pytest.raises(ValueError, match="reduction"):
            d.verify_result(work["evidence"], bad)
    bad = deepcopy(result)
    bad["head_rows"][0]["coefficients"][0] += 1
    with pytest.raises(ValueError, match="state"):
        d.verify_result(work["evidence"], bad)


def test_missing_input_is_terminal_blocked(tmp_path):
    """SCENARIO-REPORT-8987-READINESS: preserve actual missing operands without fitting."""
    evidence = d.authenticate(tmp_path, tmp_path / "raw")
    assert evidence["failures"]
    assert any(r["observed"] == "missing_file" for r in evidence["failures"])


def test_gate_type_and_zero_refusals(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8987-READINESS: booleans, absent fields and zero do not authorize."""
    for value in (True, 0, None):
        path = tmp_path / d.INPUT
        atomic_json(path, dict(spline_methods_ready_score=value, required_checks_passed=True))
        monkeypatch.setattr(d, "INPUT_PIN", d.reference(path)["sha256"])
        evidence = d.authenticate(tmp_path, tmp_path / str(value))
        checks = [
            r for r in evidence["checks"] if r["artifact_field"] == "spline_methods_ready_score"
        ]
        assert checks and not checks[0]["passed"]


def test_readiness_and_validation_failures(work, monkeypatch):
    """SCENARIO-REPORT-8987-READINESS: convergence and admission remain different operands."""
    from carnot.reporting import matched_corpus_a_8949 as qualification

    monkeypatch.setattr(qualification, "qualified", lambda *a: True)
    validation = dict(receipts=[dict(name="private_control")], coverage={})
    value = d.build(work, validation, False)
    assert value["fitted_states_ready_score"] == 1
    assert value["collection_ready_score"] == 1
    failed = deepcopy(work)
    failed["result"]["development_admission_rows"][0]["passed"] = False
    value = d.build(failed, validation, False)
    assert value["fitted_states_ready_score"] == 1 and value["collection_ready_score"] == 0
    assert value["verdict_class"] == "null" and value["unchanged_protocol_retired"]
    failed["owned_failure"] = "nonconvergence:gradient_inf=0.2"
    value = d.build(failed, validation, False)
    assert value["fitted_states_ready_score"] == value["collection_ready_score"] == 0
    assert value["convergence_failure_residual"] == failed["owned_failure"]
    assert value["verdict_class"] == "disqualified"
    failed = deepcopy(work)
    failed["result"]["construction"]["complete"] = False
    assert "construction_exhausted" in d.build(failed, validation, False)["honest_verdict"]


def test_private_foreign_cli_success_missing_tamper_and_replay(tmp_path):
    """SCENARIO-REPORT-8987-READINESS: real children run outside checkout without PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    output = tmp_path / (d.NAME + ".json")

    def child(*args):
        print("before_private_child", args, flush=True)
        process = subprocess.Popen(
            [sys.executable, "-u", str(d.ROOT / d.CLI), *args],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while True:
            try:
                stdout, _ = process.communicate(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print("private_child completed=0 pending=1", flush=True)
        print(stdout, "after_private_child", process.returncode, flush=True)
        return process.returncode

    assert child("--date", "wrong") == 2
    assert child("--date", "20261111", "--fixture-e2e", str(output)) == 0
    value = json.loads(output.read_text())
    assert value["fixture_only"] and value["completed_count"] == 320
    assert child("--cold-replay", str(output)) == 0
    changed = deepcopy(value)
    changed["rows"][0]["value"] += 0.1
    changed["reproducibility_checksum"] = d.checksum(changed)
    atomic_json(output, changed)
    assert child("--cold-replay", str(output)) == 1
    atomic_json(output, value)
    ref = value["trained_state_reference"]
    saved = Path(ref["path"]).read_bytes()
    Path(ref["path"]).write_text("{}")
    assert child("--cold-replay", str(output)) == 1
    Path(ref["path"]).write_bytes(saved)
    env["CARNOT_8986_INPUT_ROOT"] = str(tmp_path / "absent")
    blocked = tmp_path / "blocked" / output.name
    assert child("--date", "20261111", "--fixture-e2e", str(blocked)) == 0
    b = json.loads(blocked.read_text())
    assert b["verdict_class"] == "blocked" and b["collection_ready_score"] == 0
    assert child("--cold-replay", str(blocked)) == 0


def test_validation_plan_and_main_binding(tmp_path, monkeypatch):
    """REQ-REPORT-8987-ADMISSION: the reused runner receives only this producer's paths."""
    commands = runner.validation_plan(tmp_path)
    assert set(d.OWNED) <= set(a for c in commands for a in c.argv)
    assert all(
        not any(a.startswith("tests/python/") for a in c.argv)
        for c in commands
        if "pytest" not in c.argv
        and "spec_coverage" != c.name
        and c.name not in {"ruff_check", "ruff_format", "e2e018", "e2e021"}
    )
    monkeypatch.setattr(runner.base, "main", lambda argv: 17)
    assert runner.main([]) == 17


def test_replay_and_semantic_controls(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8987-READINESS: rehashing summaries cannot change pinned natural operands."""
    candidate = tmp_path / (d.NAME + ".json")
    w = deepcopy(work)
    validation = dict(receipts=[], coverage={})
    raw = Path(w["raw"])
    atomic_json(raw / "work.json", w)
    atomic_json(raw / "validation.json", validation)
    w["raw_references"] = [d.reference(raw / "work.json"), d.reference(raw / "validation.json")]
    with runner.binding():
        value = runner.base.artifact(w, validation, True)
    atomic_json(candidate, value)
    assert runner.replay(candidate)["passed"]
    bad = deepcopy(value)
    bad["code_config_hashes"][0]["source_path"] = str(tmp_path / "missing.py")
    atomic_json(candidate, bad)
    assert runner.replay(candidate)["reason"] == "executed_bytes_drift"
    bad = deepcopy(w)
    bad["evidence"]["spline_methods"]["spline_methods_ready_score"] = 0
    atomic_json(raw / "work.json", bad)
    bad["raw_references"] = [d.reference(raw / "work.json"), d.reference(raw / "validation.json")]
    with runner.binding():
        atomic_json(candidate, runner.base.artifact(bad, validation, True))
    assert runner.replay(candidate)["reason"] == "pinned_operand_drift"
    bad = deepcopy(w)
    bad["evidence"]["work"]["foreign_operand"] = True
    atomic_json(raw / "work.json", bad)
    bad["raw_references"] = [d.reference(raw / "work.json"), d.reference(raw / "validation.json")]
    with runner.binding():
        atomic_json(candidate, runner.base.artifact(bad, validation, True))
    assert runner.replay(candidate)["reason"] == "pinned_work_drift"


def test_stationarity_manifest_and_graph_refusals(work, monkeypatch):
    """SCENARIO-REPORT-8987-READINESS: retained states, roles and constructions are operands."""
    result = work["result"]
    bad = deepcopy(result)
    bad["head_rows"][0]["loss"] += 0.1
    state = json.loads(Path(result["trained_state_reference"]["path"]).read_bytes())
    state["heads"] = bad["head_rows"]
    original_loads = d.json.loads
    monkeypatch.setattr(
        d.json,
        "loads",
        lambda b: (
            state
            if b == Path(result["trained_state_reference"]["path"]).read_bytes()
            else original_loads(b)
        ),
    )
    with pytest.raises(ValueError, match="stationarity"):
        d.verify_result(work["evidence"], bad)
    monkeypatch.undo()
    bad = deepcopy(result)
    bad["source_manifest"]["graph_manifest_hash"] = "changed"
    with pytest.raises(ValueError, match="manifest"):
        d.verify_result(work["evidence"], bad)
    bad = deepcopy(result)
    bad["role_manifest"][0]["role"] = "wrong"
    bad["source_manifest"]["role_manifest_hash"] = d.canonical_hash(bad["role_manifest"])
    with pytest.raises(ValueError, match="role"):
        d.verify_result(work["evidence"], bad)
    bad = deepcopy(result)
    bad["graph_manifest"][0]["start"] = "ZZ"
    bad["source_manifest"]["graph_manifest_hash"] = d.canonical_hash(bad["graph_manifest"])
    with pytest.raises(ValueError, match="graph"):
        d.verify_result(work["evidence"], bad)


def test_ensemble_prediction_tamper(work):
    """SCENARIO-REPORT-8987-READINESS: internally consistent fake losses still fail actual states."""
    bad = deepcopy(work["result"])
    for row in bad["rows"]:
        if row["condition"] == "exposed_development_guard" and row["arm"] == "spline":
            row["p_error"] = 0.5
            row["value"] = row["numerator"] = (row["p_error"] - row["error"]) ** 2
    bad["development_metrics"], bad["development_admission_rows"] = d.admission(bad["rows"])
    with pytest.raises(ValueError, match="prediction"):
        d.verify_result(work["evidence"], bad)


def test_primitive_and_owned_checks_are_fail_closed(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8987-READINESS: failed owned checks disqualify without claiming partial science."""
    from carnot.reporting import matched_corpus_a_8949 as qualification

    monkeypatch.setattr(qualification, "qualified", lambda *a: False)
    value = d.build(work, dict(receipts=[dict(name="failed")], coverage={}), False)
    assert value["verdict_class"] == "disqualified" and value["collection_ready_score"] == 0
    assert not runner.replay(tmp_path / "absent.json")["passed"]


def test_role_binding_tamper(work):
    """SCENARIO-REPORT-8987-READINESS: a different state cannot be attached to a frozen role."""
    bad = deepcopy(work["result"])
    bad["role_manifest"][0]["trained_state_sha256"] = "changed"
    bad["source_manifest"]["role_manifest_hash"] = d.canonical_hash(bad["role_manifest"])
    with pytest.raises(ValueError, match="role_binding"):
        d.verify_result(work["evidence"], bad)


def test_failed_fit_retains_actual_residual_and_trace(work, tmp_path, monkeypatch):
    """REQ-REPORT-8987-ADMISSION: false optimizer success cannot erase failed fit evidence."""
    monkeypatch.setattr(
        d,
        "minimize",
        lambda fun, x0, **kw: SimpleNamespace(x=x0, success=True, message="false success", nit=0),
    )
    with pytest.raises(ValueError, match="nonconvergence:gradient_inf="):
        d.evaluate(work["evidence"], tmp_path)
    trace = json.loads((tmp_path / "optimizer_traces/fit-1.json").read_bytes())
    assert len(trace) == 1 and trace[0]["gradient_inf"] > 1e-6
