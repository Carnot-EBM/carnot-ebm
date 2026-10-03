"""REQ-REPORT-8075: private geometry and terminal controls supply no science credit."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.verify import constraint_projection_8075 as kernel


def release(source="s", slot=1, **extra):
    """A receipt makes the permitted label-access time explicit."""
    return dict(
        source_id=source,
        release_slot=slot,
        observed_slot=slot,
        role="update",
        eligible=True,
        y=1,
        phi=[1.0, 0.0],
        **extra,
    )


def test_calibration_and_sign():
    """SCENARIO-REPORT-8075-PROJECTION: the intercept stays frozen."""
    phi, w0 = kernel.calibrated(np.array([[2.0, 1.0]]), [0.2, -0.1], [0.7, 1.3])
    assert phi[0] @ w0 == pytest.approx(1.09)
    for y in (0, 1):
        row = kernel.constraint(phi[0], w0, y, "s")
        s = 2 * y - 1
        assert row["rhs"] == pytest.approx(min(s * 1.09, 0 if y else np.log(9)))
    with pytest.raises(ValueError, match="label"):
        kernel.constraint(phi[0], w0, 2, "s")
    with pytest.raises(ValueError, match="nonfinite"):
        kernel.constraint([np.nan], [0.0], 1, "s")
    with pytest.raises(ValueError, match="shape"):
        kernel.constraint([1, 2], [0], 1, "s")


def test_memory_atomic_replay_late_duplicate(tmp_path):
    """SCENARIO-REPORT-8075-MEMORY: original slots decide bounded membership."""
    path = tmp_path / "memory.json"
    memory = kernel.Memory(path, [0.0, 0.0])
    for i in range(66):
        memory.add(release(f"s{i:03}", i))
    assert len(memory.rows) == 64
    assert memory.rows[0]["source_id"] == "s002"
    memory.add(release("late", 0))
    assert "late" not in [r["source_id"] for r in memory.rows]
    before = path.read_bytes()
    assert memory.add(release("late", 0)) == "duplicate"
    assert before == path.read_bytes()
    assert kernel.Memory(path, [0.0, 0.0]).state == memory.state
    conflict = release("late", 0)
    conflict["y"] = 0
    with pytest.raises(ValueError, match="duplicate_conflict"):
        memory.add(conflict)
    for role in ("admission", "retention"):
        row = release(role)
        row["role"] = role
        with pytest.raises(ValueError, match="release_contract"):
            memory.add(row)
    row = release("future")
    row["observed_slot"] = 0
    with pytest.raises(ValueError, match="release_contract"):
        memory.add(row)
    row = release("ineligible")
    row["eligible"] = False
    with pytest.raises(ValueError, match="release_contract"):
        memory.add(row)
    with pytest.raises(ValueError, match="memory_identity"):
        kernel.Memory(path, [1.0, 0.0])
    value = json.loads(path.read_text())
    value["state"]["events"].pop()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="memory_hash"):
        kernel.Memory(path, [0.0, 0.0])


def test_projection_fallback_and_invalid():
    """SCENARIO-REPORT-8075-PROJECTION: a finite budget is not a certificate."""
    rows = [dict(source_id="a", normal=[1.0], rhs=0.0)]
    result = kernel.project([-2.0], rows, [0.0], [0.2], seed=7)
    assert result["feasible"] and result["point"] == [0.0]
    assert result["projection_steps"] == 1
    assert result["cost"]["full_residual_checks"] >= 4
    assert result["termination"] == "residual"
    for incumbent, fallback in [([0.2], "incumbent"), ([-0.2], "initial")]:
        result = kernel.project([-0.4], rows, [0.0], incumbent, seed=7, budget=0)
        assert result["fallback"] == fallback and not result["candidate_feasible"]
        assert result["termination"] == "budget_exhausted" and result["feasible"]
    result = kernel.project([np.nan], rows, [0.0], [0.2], seed=7)
    assert result["fallback"] == "incumbent" and result["termination"] == "nonfinite_proposal"
    assert kernel.project([0.1], [], [0.0], [0.0], seed=1)["feasible"]
    frozen = kernel.project([0.1, 5], [], [0.0, 1], [0.0, 1], seed=1, frozen_last=True)
    assert frozen["point"][-1] == 1
    for bad, reason in [
        ([dict(source_id="z", normal=[0.0], rhs=1.0)], "zero_norm_violated"),
        ([dict(source_id="z", normal=[1.0], rhs=1.0)], "initial_infeasible"),
        ([dict(source_id="z", normal=[np.nan], rhs=0.0)], "nonfinite_constraint"),
    ]:
        with pytest.raises(ValueError, match=reason):
            kernel.project([0.0], bad, [0.0], [0.0], seed=1)
    with pytest.raises(ValueError, match="projection_contract"):
        kernel.project([0.0], rows, [0.0], [0.0], seed=1, budget=257)
    with pytest.raises(ValueError, match="projection_shape"):
        kernel.project([0, 0], rows, [0], [0], seed=1)
    with pytest.raises(ValueError, match="nonfinite_initial"):
        kernel.project([0], rows, [np.nan], [0], seed=1)


def test_qp_and_measurement_reduction(tmp_path):
    """SCENARIO-REPORT-8075-REFERENCE: distance and feasibility are distinct."""
    from carnot import experiment_8075_v699_constraint_projection_kernel as exp

    evidence = exp.measure(tmp_path, fixture=True)
    assert len(evidence["rows"]) == 64
    assert all(r["feasible"] and r["distance_gap"] >= -1e-6 for r in evidence["rows"])
    assert all(r["passed"] for r in evidence["controls"])
    assert exp.reduce(evidence)
    evidence["rows"][0]["max_residual"] = 1.0
    assert not exp.reduce(evidence)
    infeasible = exp.qp([0.0], [dict(normal=[0.0], rhs=1.0)], [0.0])
    assert not infeasible["feasible"]


def test_cli_success_blocked_mutation_and_replay(tmp_path):
    """SCENARIO-REPORT-8075-TERMINAL: real script runs outside checkout."""
    from carnot import experiment_8075_v699_constraint_projection_kernel as exp

    cli = exp.ROOT / exp.CLI
    env = dict(__import__("os").environ, PYTHONUNBUFFERED="1", PYTHONPATH="")
    config = __import__("os").environ.get("CARNOT_8075_COVERAGE_CONFIG")
    prefix = [sys.executable]
    if config:
        prefix += ["-m", "coverage", "run", "--rcfile=" + config, "--parallel-mode"]

    def run(*args):
        return subprocess.run(
            [*prefix, str(cli), *map(str, args)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=300,
        )

    output = tmp_path / (exp.NAME + ".json")
    assert run("--fixture-output", output).returncode == 0
    assert run("--cold-replay", output).returncode == 0
    artifact = json.loads(output.read_text())
    assert artifact["generalized_learning_benefit_score"] == 0
    assert artifact["independent_count"] == 0
    blocked = tmp_path / "blocked" / output.name
    assert run("--fixture-output", blocked, "--root", tmp_path / "absent").returncode == 0
    value = json.loads(blocked.read_text())
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["observed"] == "missing"
    assert run("--cold-replay", blocked).returncode == 0
    mutated = tmp_path / "mutation" / output.name
    assert run("--fixture-output", mutated, "--mutate").returncode == 0
    assert json.loads(mutated.read_text())["verdict_class"] == "disqualified"
    artifact["completed_count"] = 2
    output.write_text(json.dumps(artifact))
    assert run("--cold-replay", output).returncode == 1
    assert run("--fixture-output", mutated).returncode == 1


def test_replay_and_memory_tamper(tmp_path):
    """SCENARIO-REPORT-8075-MEMORY: a recomputed envelope cannot hide lost events."""
    from carnot import experiment_8075_v699_constraint_projection_kernel as exp

    memory = kernel.Memory(tmp_path / "memory.json", [0.0, 0.0])
    memory.add(release())
    value = json.loads(memory.path.read_text())
    value["state"]["rows"] = []
    value["sha256"] = kernel.canonical_hash(value["state"])
    memory.path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="memory_replay"):
        kernel.Memory(memory.path, [0.0, 0.0])
    assert not exp.replay(tmp_path / "absent")


def test_independent_reducer_rejects_primitive_mutations(tmp_path):
    """SCENARIO-REPORT-8075-TERMINAL: altered equations cannot retain readiness."""
    from copy import deepcopy
    from carnot import experiment_8075_v699_constraint_projection_kernel as exp

    evidence = exp.measure(tmp_path)
    assert exp.reduce(evidence)
    mutations = [
        lambda e: e["systems"].clear(),
        lambda e: e["controls"][0].update(passed=False),
        lambda e: e["systems"][0]["projection"]["projection_rows"][0]["before"].__setitem__(0, 5),
        lambda e: e["systems"][0]["projection"]["projection_rows"][0]["sample"].reverse(),
        lambda e: e["systems"][0]["projection"]["projection_rows"][0]["after"].__setitem__(0, 5),
        lambda e: e["systems"][0]["projection"].update(candidate_feasible=True),
        lambda e: e["systems"][0]["qp"].update(reference_certified=False),
        lambda e: e["systems"][0]["constraints"][0]["release_receipt"].update(role="admission"),
        lambda e: e["systems"][0]["constraints"][0].update(rhs=5),
        lambda e: e["systems"][0]["gradient"].__setitem__(0, 5),
        lambda e: e["systems"][0]["qp"]["dual_multipliers"].__setitem__(0, -1),
        lambda e: e["rows"][0].update(distance=5),
    ]
    for mutation in mutations:
        altered = deepcopy(evidence)
        mutation(altered)
        assert not exp.reduce(altered)


def test_prerequisite_operands(tmp_path):
    """SCENARIO-REPORT-8075-TERMINAL: invalid terminal bindings remain blocked."""
    from carnot import experiment_8075_v699_constraint_projection_kernel as exp

    root = tmp_path / "root"
    for label in exp.INPUTS:
        path = root / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("input")
        if label.startswith("results/"):
            terminal = tmp_path / (path.stem + "-terminal.json")
            terminal.write_text("{}")
            path.write_text(
                json.dumps(
                    dict(
                        terminal_validation_sidecar_path=str(terminal),
                        learning_protocol_ready_score=0,
                    )
                )
            )
    plan = exp.prerequisites(root, tmp_path / "raw")
    assert any(r["observed"] == "invalid_terminal_binding" for r in plan["failures"])
    assert any(r["field"] == "learning_protocol_ready_score" for r in plan["failures"])


def test_owned_main_and_failed_child(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8075-TERMINAL: owned failures cannot export readiness."""
    from carnot import experiment_8075_v699_constraint_projection_kernel as exp
    from carnot.reporting.current_work_receipt import atomic_json, sha256_file

    monkeypatch.setenv(
        "CARNOT_8075_COVERAGE_CONFIG",
        __import__("os").environ.get("CARNOT_8075_COVERAGE_CONFIG", ""),
    )

    monkeypatch.setattr(exp, "terminal", lambda path: dict(passed=exp.replay(path)))

    def commands(private):
        return [dict(name="coverage_json", argv=[], deadline_s=1, expected_exit=0)]

    monkeypatch.setattr(exp, "manifest", commands)
    failed = [False]

    def run_check(root, spec, private, durable):
        durable.mkdir(parents=True, exist_ok=True)
        log = durable / (spec["name"] + ".log")
        log.write_text("unit-test dependency substitution\n")
        if spec["name"] == "measurement_normal_exit" and not failed[0]:
            exp.worker(exp.ROOT, durable.parent, fixture=False, mutate=False)
        if spec["name"] == "coverage_json":
            atomic_json(
                private / "coverage.json",
                dict(
                    files={
                        str(exp.ROOT / p): dict(summary=dict(num_statements=1, missing_lines=0))
                        for p in exp.OWNED
                    }
                ),
            )
        return dict(
            name=spec["name"], passed=not failed[0], log_path=str(log), log_sha256=sha256_file(log)
        )

    monkeypatch.setattr(exp, "run_check", run_check)
    output = tmp_path / (exp.NAME + ".json")
    assert exp.main(["--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["projection_kernel_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert exp.replay(output)
    original = json.loads(output.read_text())
    raw = Path(original["terminal_validation_sidecar_path"]).parent
    work = json.loads((raw / "work.json").read_text())
    failure = exp.build(work, raw, original["validation_receipts"], {}, fixture=False)
    assert (
        failure["verdict_class"] == "disqualified" and failure["projection_kernel_ready_score"] == 0
    )
    assert any(r["check"] == "owned_statement_coverage" for r in failure["gate_check_summary"])
    for mutation in [
        lambda v: v["source_artifact_hashes"][0].update(sha256="bad"),
        lambda v: v["code_config_hashes"].update({exp.MODULE: "bad"}),
        lambda v: v["validation_receipts"][0].update(log_sha256="bad"),
    ]:
        value = __import__("copy").deepcopy(original)
        mutation(value)
        output.write_text(json.dumps(value))
        assert not exp.replay(output)
    failed[0] = True
    failed_output = tmp_path / "failed" / output.name
    assert exp.main(["--output", str(failed_output)]) == 0
    assert json.loads(failed_output.read_text())["verdict_class"] == "disqualified"


def test_measurement_budget_and_control_detection(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8075-REFERENCE: time and rejection budgets are enforced."""
    from carnot import experiment_8075_v699_constraint_projection_kernel as exp

    original = kernel.project

    def changed(proposal, rows, *args, **kwargs):
        if rows and rows[0]["source_id"] == "zero":
            return {}
        return original(proposal, rows, *args, **kwargs)

    monkeypatch.setattr(kernel, "project", changed)
    assert not exp.reduce(exp.measure(tmp_path / "control"))
    times = iter([0, 400])
    monkeypatch.setattr(exp, "progress", lambda *args: None)
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="fixture_budget"):
        exp.measure(tmp_path / "timeout")


def test_external_environment_and_invalid_json(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8075-TERMINAL: missing tools are exact external operands."""
    from carnot import experiment_8075_v699_constraint_projection_kernel as exp

    root = tmp_path / "root"
    for label in exp.INPUTS:
        path = root / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("invalid input")
    monkeypatch.setattr(exp, "ROOT", root)
    monkeypatch.setattr(exp.sys, "version_info", (3, 10))
    plan = exp.prerequisites(root, tmp_path / "raw")
    assert any(r["check"] == "input_json" for r in plan["failures"])
    assert any(r["check"] == "required_tool" for r in plan["failures"])
    assert any(r["check"] == "python_version" for r in plan["failures"])
