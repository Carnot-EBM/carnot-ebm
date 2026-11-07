"""REQ-REPORT-8906: a refusal cannot remove a registered future certificate."""

from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import finite_budget_methods_8906 as m
from carnot.reporting import finite_budget_runner_8906 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_protocol_and_exact_decisions():
    """REQ-REPORT-8906: all arms share exact arithmetic and one final event."""
    p = m.protocol()
    assert p["look_schedule"] == [120, 240, 480, 960]
    assert p["replicates"] == 40 and p["population_per_group"] == 1024
    assert p["selection_per_group"] == 240 and len(p["policy_menu"]) == 7
    assert sum(Fraction(x["events"], x["alpha_denominator"]) for x in m.alpha_ledger()) == Fraction(
        1, 100
    )
    assert not m.passes(0, 0, 5600)
    assert not m.passes(168, 0, 5600)
    assert m.passes(169, 0, 5600)
    for n in range(1, 961):
        for e in {0, min(n, n // 20), n}:
            assert m.passes(n, e, 5600) == (m.frozen.base.old.upper(n, e, m.frozen.FINAL) <= 0.05)
    for args in [(-1, 0, 5600), (1, 2, 5600), (True, 0, 5600), (1, 0, 0)]:
        with pytest.raises(ValueError):
            m.passes(*args)


def test_paths_edges_and_irrevocable_pass():
    """SCENARIO-REPORT-8906-CONTRACT: favorable suffixes use accepts and errors together."""
    safe = [(n, 0) for n in m.LOOKS]
    assert m.decide(safe)["candidate_draws"] == 240
    final_only = [(0, 0), (0, 0), (0, 0), (169, 0)]
    assert m.decide(final_only, refusal=True)["candidate_draws"] == 960
    assert m.decide(final_only, refusal=True)["certificate_issued"]
    empty = m.decide([(0, 0)] * 4, refusal=True)
    assert empty["candidate_draws"] == 960 and not empty["certificate_issued"]
    failed = m.decide([(n, n) for n in m.LOOKS], refusal=True)
    assert failed["candidate_draws"] == 120 and failed["reason"] == "impossible_future"
    assert len(failed["looks"][0]["remaining_tests"]) == 3
    retained = m.decide([(120, 0), (240, 0), (480, 240), (960, 720)], refusal=True)
    assert retained["certificate_issued"] and retained["candidate_draws"] == 240
    assert not m.decide(safe, fixed=True)["candidate_draws"] < 960
    for stats in [[], [(121, 0)] * 4, [(0, 0), (1, 0), (1, 1), (1, 1)]]:
        with pytest.raises(ValueError, match="statistic"):
            m.decide(stats)


def test_qualification_and_cost_identity():
    """REQ-REPORT-8906: exhaustive paths qualify the proof independently of savings."""
    q = m.qualify()
    assert all(x["metric"] == 1 for x in q["rows"])
    assert len(q["exhaustive_path_rows"]) == 2**8 + 3**8
    assert len(q["randomized_state_rows"]) == 4096
    assert q["unsafe_control"]["lost_certificate"]
    assert all(x["metric"] == 1 for x in q["bound_monotonicity_rows"])
    assert all(x["metric"] == 1 for x in q["cost_identity_rows"])
    assert q["scientific_benefit_claim"] is False
    stops = [
        dict(policy=i, group=g, candidate_draws=240 if i < 6 else 960)
        for g in m.frozen.base.GROUPS
        for i in range(7)
    ]
    assert m.costs(stops)["shared_union_labels"] == 4800
    for x in stops:
        if x["policy"] == 6:
            x["candidate_draws"] = 120
    assert m.costs(stops)["shared_union_labels"] == 1920


def test_external_inputs(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8906-CONTRACT: external absence and observed zero are distinct."""
    assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    for value, check in [
        ({}, "missing_field"),
        ({r.FIELD: 0}, "observed_zero"),
        ({r.FIELD: 1}, "rejected_producer"),
    ]:
        atomic_json(tmp_path / (r.SOURCE + ".json"), value)
        assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == check
    p = r.preconditions(tmp_path / "original", m.ROOT / "results")
    assert p["external_ready"], p["diagnostics"]
    assert len(p["references"]) >= 16
    path = tmp_path / (r.SOURCE + ".json")
    atomic_json(
        path,
        {
            r.FIELD: 1,
            "experiment_id": 8896,
            "task_id": "exp8896-shared-terminal-evaluation",
            "verdict_class": "circular_positive",
            "required_checks_passed": True,
            "code_config_hashes": [],
            "raw_shard_hashes": [],
        },
    )
    monkeypatch.setattr(r, "PIN", r.sha256_file(path))
    assert r.preconditions(tmp_path / "incomplete", tmp_path)["diagnostics"]


def test_private_cli_and_semantic_tampering(tmp_path):
    """SCENARIO-REPORT-8906-CONTRACT: actual cold replay rejects rehashed false readiness."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    output = tmp_path / (m.NAME + ".json")
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]

    def child(options):
        return subprocess.run(
            command + options, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )

    run = child(["--private-output", str(output), "--sources", str(tmp_path / "absent")])
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["finite_budget_ready_score"] == 0
    assert child(["--cold-replay", str(output)]).returncode == 0
    value["finite_budget_ready_score"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert child(["--cold-replay", str(output)]).returncode == 1
    assert child(["--date", "wrong"]).returncode != 0
    assert child(["--private-output", str(m.ROOT / "results" / output.name)]).returncode != 0


def test_bound_failure_is_closed(monkeypatch):
    """REQ-REPORT-8906: numerical failure grants neither a pass nor a refusal."""
    m.bound.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(m.frozen.base.old, "upper", lambda *a: float("nan"))
        with pytest.raises(ValueError, match="invalid_exact_bound"):
            m.decide([(0, 0)] * 4, refusal=True)
    m.bound.cache_clear()


@pytest.fixture
def qualified_fixture(monkeypatch):
    """REQ-REPORT-8906: a small private proof exercises publication without science claims."""
    result = dict(
        r.scheduled("fixture"),
        rows=[m.row("fixture", True)],
        proof_obligations=dict(proof=m.PROOF, passed=True),
    )
    coverage = dict(
        complete=True,
        percent=100,
        files=[dict(path=p, statements=1, missing=[], excluded=[]) for p in m.OWNED],
    )
    validation = dict(
        coverage=coverage,
        union=dict(passed=True),
        receipts=[dict(name="fixture", passed=True)],
        raw_references=[],
    )
    pre = dict(
        external_ready=True,
        diagnostics=[],
        references=[],
        census_reference=None,
        resources=dict(passed=True),
    )
    monkeypatch.setattr(m, "qualify", lambda: deepcopy(result))
    monkeypatch.setattr(r, "preconditions", lambda *a: deepcopy(pre))
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(validation))
    monkeypatch.setattr(r, "HEALTH", Path("/tmp/carnot-8906-nonexistent-health"))
    return pre, result, validation


def test_production_flow_and_rehashed_tampering(tmp_path, monkeypatch, qualified_fixture):
    """SCENARIO-REPORT-8906-CONTRACT: readiness needs every owned and terminal check."""
    from types import SimpleNamespace

    output = tmp_path / (m.NAME + ".json")
    terminal = SimpleNamespace(
        terminal=lambda *a: [
            dict(name="terminal", passed=True, stdout_path=str(log), stderr_path=str(log))
        ]
    )
    log = tmp_path / "terminal.log"
    log.write_text("private test evidence")
    monkeypatch.setattr(r, "namespace", lambda *a: terminal)
    assert r.main(["--output", str(output)]) == 0
    original = json.loads(output.read_bytes())
    assert original["finite_budget_ready_score"] == 1 and original["verdict_class"] == "null"
    assert r.replay(output)["passed"]
    for key, changed, error in [
        ("experiment_id", 8907, "producer_identity"),
        ("protocol", {}, "protocol_drift"),
        ("independent_count", 40, "compute_scope_drift"),
        ("completed_count", 2, "count_drift"),
        ("intended_count", 9, "count_drift"),
        ("proof_obligations", {}, "reduction_drift"),
        ("finite_budget_ready_score", 0, "readiness_drift"),
        ("verdict_class", "blocked", "readiness_drift"),
    ]:
        value = deepcopy(original)
        value[key] = changed
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        atomic_json(output, value)
        with pytest.raises(ValueError, match=error):
            r.replay(output)
    atomic_json(output, dict(original, reproducibility_checksum="wrong"))
    with pytest.raises(ValueError, match="artifact_checksum"):
        r.replay(output)
    value = deepcopy(original)
    value["current_work_receipt"]["event_count"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="current_receipt"):
        r.replay(output)
    terminal.terminal = lambda *a: [
        dict(name="failed", passed=False, stdout_path=str(log), stderr_path=str(log))
    ]
    assert r.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_owned_failure_and_validation_adapters(tmp_path, monkeypatch, qualified_fixture):
    """REQ-REPORT-8906: failures in owned proof or controls cannot become external blocks."""
    output = tmp_path / (m.NAME + ".json")
    pre, result, validation = qualified_fixture
    result["proof_obligations"]["passed"] = False
    assert r.main(["--private-output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["failed_count"] == 1
    pre["resources"]["passed"] = False
    assert r.main(["--private-output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_validation_wrapper(tmp_path, monkeypatch):
    """REQ-REPORT-8906: use the inherited validator with precisely the owned paths."""
    from types import SimpleNamespace

    adapter = SimpleNamespace(validate=lambda *a: dict(actual_validator=True))
    monkeypatch.setattr(r, "namespace", lambda *a: adapter)
    assert r.validate(tmp_path, tmp_path) == dict(actual_validator=True)
    assert adapter.m is m


def test_original_operands_and_malformed_producers(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8906-CONTRACT: intact producer hashes never excuse altered imports."""
    producer = tmp_path / (r.SOURCE + ".json")
    for contents in ["{", "[]"]:
        producer.write_text(contents)
        assert (
            r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"]
            == "rejected_producer"
        )
    operand = tmp_path / "population.json"
    atomic_json(operand, dict(inputs=[dict(split="population")] * 4096))
    value = {
        r.FIELD: 1,
        "experiment_id": 8896,
        "task_id": "exp8896-shared-terminal-evaluation",
        "verdict_class": "circular_positive",
        "required_checks_passed": True,
        "code_config_hashes": [],
        "raw_shard_hashes": [r.reference(operand)],
    }
    atomic_json(producer, value)
    monkeypatch.setattr(r, "PIN", r.sha256_file(producer))
    assert r.preconditions(tmp_path / "valid", tmp_path)["external_ready"]
    operand.write_text("altered")
    assert (
        r.preconditions(tmp_path / "altered", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )
    operand.unlink()
    assert (
        r.preconditions(tmp_path / "absent", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    )


def test_measured_private_cli(tmp_path):
    """SCENARIO-REPORT-8906-CONTRACT: real proof and child replay stay private and unqualified."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    output = tmp_path / (m.NAME + ".json")
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    run = subprocess.run(
        command + ["--private-output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["completed_count"] > 0 and value["proof_obligations"]["passed"]
    assert value["verdict_class"] == "disqualified" and value["finite_budget_ready_score"] == 0
    replay = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["exhaustive_path_rows"][0]["fixed_certificate"] = False
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    rejected = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert rejected.returncode == 1 and "reduction_drift" in rejected.stdout


def test_health_receipt_and_qualification_failures(tmp_path, monkeypatch, qualified_fixture):
    """REQ-REPORT-8906: repository failures stay separate and their exact logs remain bound."""
    output = tmp_path / (m.NAME + ".json")
    log = tmp_path / "health.log"
    log.write_text("unrelated repository failure")
    health = tmp_path / "health.json"
    atomic_json(health, dict(log=r.reference(log), passed=False))
    monkeypatch.setattr(r, "HEALTH", health)
    assert r.main(["--private-output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["repository_health"]["passed"] is False
    assert Path(value["repository_health"]["log"]["path"]).read_text() == log.read_text()
    value["required_checks_passed"] = True
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="qualification_drift"):
        r.replay(output)
    log.write_text("changed log")
    with pytest.raises(ValueError, match="repository_health_log_drift"):
        r.main(["--private-output", str(output)])
