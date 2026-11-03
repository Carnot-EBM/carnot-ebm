"""REQ-REPORT-8770 / REQ-VERIFY-8770: optimized costs need exact custody."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import optimized_cost_attribution_8770 as e
from carnot.reporting import optimized_cost_attribution_measure_8770 as m
from carnot.reporting import optimized_cost_attribution_runner_8770 as r
from carnot.reporting.v709_execution import child


def synthetic():
    """SCENARIO-REPORT-8770-COST: repeats are processes, never semantic cases."""
    return [
        dict(
            s,
            status="completed",
            passed=True,
            duration_s=1.0,
            child_duration_s=0.8,
            components={"feature_extraction": 0.6},
            child_unexplained_s=0.2,
            startup_s=0.1,
            shutdown_s=0.1,
            request_digest="same",
            output_digest="same",
        )
        for s in e.roster()
    ]


def test_roster_bounds_and_overhead():
    """SCENARIO-REPORT-8770-COST: attribution cannot manufacture a speed headline."""
    rows = synthetic()
    assert len(rows) == len({s["unit_id"] for s in rows}) == 12
    assert sum(s["item_evaluations"] for s in rows) == 12288
    summary = e.aggregate(rows)
    assert summary["complete"] and summary["ready"]
    assert len(summary["observer_overhead_rows"]) == 2
    assert summary["dominant_component"]["operation"] == "feature_extraction"
    assert summary["next_optimization_decision"]["selected_target"] == "feature_extraction"
    assert max(b["ideal_speedup"] for b in summary["ideal_speed_bounds"]) < 10
    for row in rows:
        if row["instrumented"]:
            row["duration_s"] = 1.3
    assert not e.aggregate(rows)["ready"]
    assert e.aggregate(rows)["next_optimization_decision"]["selected_target"] is None
    rows[0]["status"] = "censored"
    assert not e.aggregate(rows)["complete"]


def test_inputs(tmp_path, monkeypatch):
    """REQ-REPORT-8770: both qualified primaries and their operands are required."""
    good = e.inputs(e.ROOT)
    assert good["passed"], good["gate_check_summary"]
    assert good["workload"]["configuration_template"]["reuse_request_features"] is True
    assert not e.inputs(tmp_path)["passed"]
    monkeypatch.setattr(e, "COST_PIN", "sha256:changed")
    assert not e.inputs(e.ROOT)["passed"]


def test_cli_blocked_and_replay(tmp_path):
    """SCENARIO-VERIFY-8770-CUSTODY: actual CLI children publish private blocked rows."""
    output = tmp_path / (e.NAME + ".json")
    receipt = child(
        "blocked",
        [sys.executable, "-u", e.CLI, "--root", str(tmp_path), "--private-output", str(output)],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 12
    assert r.replay(output, value["primitive_reference"])
    assert not r.replay(output, None)
    e.atomic_json(output, dict(value, completed_count=1))
    assert not r.replay(output, value["primitive_reference"])


def test_real_measured_cli_replay(tmp_path):
    """SCENARIO-VERIFY-8770-CUSTODY: measured private CLI rows must cold-replay too."""
    output = tmp_path / (e.NAME + ".json")
    receipt = child(
        "measured",
        [sys.executable, "-u", e.CLI, "--private-output", str(output)],
        tmp_path / "logs",
        deadline=240,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["completed_count"] == 12
    assert r.replay(output, value["primitive_reference"])


def test_installed_e2e003_e2e004(tmp_path):
    """REQ-VERIFY-8770: real installed Python/native wire serialization and parity."""
    inputs = e.inputs(e.ROOT)
    measured = m.measure(tmp_path / "measurement", tmp_path, inputs, 2**63)
    assert measured["owned_error"] is None
    assert len(measured["process_rows"]) == 12
    assert all(s["status"] == "completed" for s in measured["process_rows"])
    assert all(s["passed"] for s in measured["parity_rows"])
    assert len(measured["parity_rows"]) == 12288
    first = measured["process_rows"][0]
    assert first["child_steady_scoring_p95_ms"] > 0
    assert first["parent_steady_request_p95_ms"] >= first["child_steady_scoring_p95_ms"]
    primitive = e.load(first["session_reference"])
    assert m.reduce(primitive, inputs)["process_row"] == first
    for change in ["clock", "mode", "profile", "parity"]:
        bad = deepcopy(primitive)
        if change == "clock":
            bad["profile"]["spans"] = [dict(operation="bad", start_ns=0, end_ns=2**63)]
        elif change == "mode":
            bad["slot"]["mode"] = "baseline"
        elif change == "profile":
            bad["profile"]["actual_exit"] = 5
        else:
            bad["observations"][0]["response"]["responses"][0]["probability"] = 5
        with pytest.raises(ValueError):
            m.reduce(bad, inputs)


def test_profile_leaves_and_actual_entrypoint(tmp_path, monkeypatch):
    """REQ-VERIFY-8770: exercise exclusive leaves and the real module entrypoint."""
    import runpy
    from types import SimpleNamespace
    from carnot.inference import optimized_profile_8770 as p

    base = p.base
    numeric = base.wire.numeric
    for owner, names in [
        (
            base.wire,
            ["read_frame", "write_frame", "sha256_file", "configuration", "decide", "json"],
        ),
        (base.batch, ["evaluate", "json"]),
        (
            numeric,
            [
                "extract",
                "canonical_hash",
                "normalized",
                "scale",
                "basis",
                "parameters",
                "predict",
                "batch",
                "json",
            ],
        ),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, getattr(owner, name))
    recorder = base.Recorder()
    p.install(recorder)
    inputs = e.inputs(e.ROOT)
    head, rows = inputs["head"], inputs["requests"][:16]
    assert len(base.batch.evaluate(head, rows, None, 16, reuse_features=True)[0]) == 16
    basis = numeric.basis
    import numpy as np

    native = SimpleNamespace(
        nine_input_residual_8533=lambda theta, x: (
            basis(np.asarray(x)) @ np.asarray(theta)
        ).tolist()
    )
    assert len(base.batch.evaluate(head, rows, native, 16, reuse_features=True)[0]) == 16
    assert any(s["operation"] == "native_crossing_and_residual" for s in recorder.spans)
    assert all(a["end_ns"] <= b["start_ns"] for a, b in zip(recorder.spans, recorder.spans[1:]))
    monkeypatch.setattr(base, "main", lambda argv: 0)
    original = base.install
    assert p.main([]) == 0 and base.install is original
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("carnot.inference.optimized_profile_8770", run_name="__main__")
    assert exit_info.value.code == 0


def test_rejected_pairs_reconciliation_and_missingness():
    """SCENARIO-REPORT-8770-COST: coordinated denominators and overlaps reject."""
    rows = synthetic()
    rows[0]["unit_id"] = rows[1]["unit_id"]
    rows[0]["repeat"] = rows[1]["repeat"] = 8
    with pytest.raises(ValueError, match="process_roster"):
        e.aggregate(rows)
    rows = synthetic()
    rows[0]["output_digest"] = "changed"
    with pytest.raises(ValueError, match="paired_exact_parity"):
        e.aggregate(rows)
    rows = synthetic()
    rows[1]["complete_components"] = dict(overlap=2.0)
    with pytest.raises(ValueError, match="cost_reconciliation"):
        e.aggregate(rows)


def test_component_order_survives_canonical_json():
    """SCENARIO-VERIFY-8770-CUSTODY: sorted JSON keys cannot reorder report rows."""
    rows = synthetic()
    for row in rows:
        row["components"] = dict(z_scoring=0.1, feature_extraction=0.5)
    assert e.aggregate(rows) == e.aggregate(json.loads(json.dumps(rows, sort_keys=True)))


def test_input_terminal_rejection_and_owned_failure(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8770-CUSTODY: rejected producers cannot be recertified."""
    original = e.load

    def reject(ref):
        value = original(ref)
        if ref["path"].endswith("terminal_validation.json"):
            raise ValueError("rejected terminal")
        return value

    with monkeypatch.context() as context:
        context.setattr(e, "load", reject)
        assert not e.inputs(e.ROOT)["passed"]
    inputs = e.inputs(e.ROOT)
    measured = m.measure(tmp_path / "failed", tmp_path, inputs, 0)
    assert measured["owned_error"] == "measurement_admission_deadline"
    assert measured["process_rows"][0]["status"] == "failed"
    assert sum(s["status"] == "censored" for s in measured["process_rows"]) == 11


def test_reducer_and_install_rejections(tmp_path, monkeypatch):
    """REQ-VERIFY-8770: retained original assertions and installed bytes fail closed."""
    config = tmp_path / "configuration.json"
    e.atomic_json(config, dict(reuse_request_features=True))
    primitive = dict(slot=dict(mode="reuse"), config_reference=e.reference(config))
    row = dict(duration_s=0.01, startup_s=0.1, shutdown_s=0.1)
    monkeypatch.setattr(m.diagnostic, "reduce", lambda *args: dict(process_row=row))
    monkeypatch.setattr(m.installed, "reduce", lambda *args: dict(process_row=dict(passed=False)))
    with pytest.raises(ValueError, match="original_numerical_parity"):
        m.reduce(primitive, {})
    monkeypatch.setattr(m.installed, "reduce", lambda *args: dict(process_row=dict(passed=True)))
    primitive.update(ready_ns=0, observations=[dict(end_ns=1)], profile=dict(spans=[]))
    with pytest.raises(ValueError, match="parent_span_overlap"):
        m.reduce(primitive, {})
    original = m.shutil.copyfile

    def corrupt(source, target):
        if Path(source).name == "optimized_profile_8770.py":
            Path(target).write_bytes(b"corrupted diagnostic")
            return target
        return original(source, target)

    monkeypatch.setattr(m.shutil, "copyfile", corrupt)
    with pytest.raises(ValueError, match="diagnostic_install_hash"):
        m.measure(tmp_path / "corrupted", tmp_path, e.inputs(e.ROOT), 2**63)


def test_qualified_cost_seal_and_workload_mutations(monkeypatch):
    """SCENARIO-VERIFY-8770-CUSTODY: rehashed seals cannot change the qualified work."""
    with monkeypatch.context() as context:
        context.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
        assert not e.inputs(e.ROOT)["passed"]
    cost = e.load(e.reference(e.ROOT / e.COST))
    original = e.load
    primitive = original(cost["primitive_reference"])
    for mode in ["seal", "workload"]:
        altered = deepcopy(primitive)
        altered["inputs"]["workload"]["configuration_template"]["item_budget"] = 1

        def changed(ref):
            value = original(ref)
            if ref == primitive["seal_reference"]:
                return (
                    {}
                    if mode == "seal"
                    else {k: v for k, v in altered.items() if k != "seal_reference"}
                )
            if mode == "workload" and ref == cost["primitive_reference"]:
                return deepcopy(altered)
            return value

        with monkeypatch.context() as context:
            context.setattr(e, "load", changed)
            assert not e.inputs(e.ROOT)["passed"]
