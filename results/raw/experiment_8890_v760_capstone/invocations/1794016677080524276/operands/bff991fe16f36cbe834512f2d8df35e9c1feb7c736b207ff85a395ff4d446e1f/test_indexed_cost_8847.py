"""REQ-REPORT-8847: costs retain construction, censoring and paired world uncertainty."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import indexed_cost_8847 as m
from carnot.reporting import indexed_cost_runner_8847 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def units():
    return m.stress()[:4]


def test_stress_and_schedule():
    """SCENARIO-REPORT-8847-CONTRACT: stress has 120 worlds, not 600 scheduling replicas."""
    cohort = m.stress()
    assert len(cohort) == 480 and len({u["cluster_id"] for u in cohort}) == 120
    for size in [16, 32, 64]:
        subset = [u for u in cohort if u["size"] == size]
        assert len(subset) == 160
        assert all(len(u["input"]["facts"]) == size for u in subset)
        assert {u["condition"] for u in subset} == {
            "missing_hop",
            "contradiction",
            "binding_swap",
            "valid_alternative",
        }
    assert m.schedule(units(), 11) == m.schedule(units(), 11)
    assert m.schedule(units(), 11) != m.schedule(units(), 13)


def test_paid_attempts():
    """REQ-REPORT-8847: complete bytes, index build and disposal are paid even on failure."""
    for unit in units():
        payload = json.dumps(unit["input"], sort_keys=True)
        pair = [m.attempt(payload, arm) for arm in m.ARMS]
        assert pair[0]["decision"] == pair[1]["decision"]
        for row in pair:
            assert row["primitive_operations"] == sum(row["operation_counts"].values())
            assert row["operation_counts"]["request_disposal"] > 0
            assert row["wall_time_s"] > 0 and row["cpu_time_s"] > 0
        for cap in [0, 1, pair[1]["primitive_operations"] - 1]:
            row = m.attempt(payload, m.ARMS[1], cap)
            assert row["budget_exhausted"] and row["decision"] == "insufficient"
            assert row["primitive_operations"] == cap
    assert pair[1]["operation_counts"]["index_build"] > 0
    for arm in m.ARMS:
        exhausted = m.attempt(payload, arm, 2 * len(payload.encode()) + 1)
        assert exhausted["budget_exhausted"] and exhausted["decision"] == "insufficient"
        row = m.attempt('{"facts":[]}', arm)
        assert row["status"] == "failed" and row["decision"] == "insufficient"
    with pytest.raises(ValueError, match="arm"):
        m.attempt("{}", "invented")


def test_measure_and_intervals():
    """REQ-REPORT-8847: all schedules count; bootstraps resample worlds, not repeated runs."""
    rows = m.score(m.measure(units()), units())
    assert len(rows) == 40
    result = m.reduce(rows)
    assert len(result["per_world_results"]) == 1
    assert result["paired_intervals"]["paid_operation_reduction"]["draws"] == 10000
    assert result["completed_decision_parity"]
    assert result["paired_intervals"]["coverage_loss"]["upper"] == 0
    assert result["paired_intervals"]["semantic_false_accept_increase"]["upper"] == 0
    assert result["wall_speed_ratio"]["confidence"] == 0.99
    altered = deepcopy(rows)
    altered[0]["decision"] = "support" if rows[0]["decision"] != "support" else "insufficient"
    assert not m.reduce(altered)["completed_decision_parity"]
    for row in altered:
        row.update(status="censored", budget_exhausted=True, decision="insufficient")
    assert m.reduce(altered)["wall_speed_ratio"]["lower"] is None
    with pytest.raises(ValueError, match="roster"):
        m.reduce(rows[:-1])
    assert m.reduce([])["indexed_cost_benefit_score"] == 0


def test_preconditions(tmp_path, monkeypatch):
    """REQ-REPORT-8847: original indexed authority and preserved V756 operands authenticate."""
    pre = r.preconditions(tmp_path / "raw")
    assert pre["external_ready"] and pre["resources"]["passed"], pre
    assert len(pre["evaluation_worlds"]) == 240
    monkeypatch.setattr(r, "UPSTREAM", tmp_path / "missing")
    pre = r.preconditions(tmp_path / "absent")
    assert pre["diagnostics"][0]["check"] == "missing_file"
    assert not pre["external_ready"]


def test_private_cli(tmp_path):
    """SCENARIO-REPORT-8847-CONTRACT: private publication and cold replay run real children."""
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    for args in [
        ["--private-output", "--private-unit-limit", "2", "--output", str(output)],
        ["--cold-replay", str(output)],
    ]:
        child = subprocess.run(
            command + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
        )
        assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_bytes())
    assert value["indexed_cost_ready_score"] == 0 and len(value["rows"]) == 20
    value["rows"][0]["decision"] = "invented"
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert not r.replay(output)
    assert not r.replay(tmp_path / "absent")


def test_cli_errors(tmp_path):
    """REQ-REPORT-8847: dates, output isolation and subsets are validated before execution."""
    for args in [
        ["--date", "20250101"],
        ["--private-output"],
        ["--private-unit-limit", "1"],
        ["--private-output", "--private-unit-limit", "0", "--output", str(tmp_path / "x.json")],
    ]:
        with pytest.raises(SystemExit):
            r.main(args)
    assert r.main(["--cold-replay", str(tmp_path / "absent")]) == 1


def seal_private(tmp_path, monkeypatch):
    """Create small real operands while replacing only process scheduling in unit controls."""

    def local(raw, sample):
        atomic_json(raw / "requests.json", dict(rows=sample))
        return m.measure(sample)

    monkeypatch.setattr(r, "process_measure", local)
    raw = tmp_path / "raw"
    work = r.prepare(raw, 2)
    validation = r.unqualified()
    work.update(code_config_hashes=[], raw_shard_hashes=[])
    output = tmp_path / (m.NAME + ".json")

    def seal():
        atomic_json(raw / "work.json", work)
        atomic_json(raw / "validation.json", validation)
        value = r.artifact(
            work,
            validation,
            "20261105",
            r.reference(raw / "work.json"),
            r.reference(raw / "validation.json"),
        )
        atomic_json(output, value)
        return value

    return work, validation, output, raw, seal


def test_rehashed_attacks(tmp_path, monkeypatch):
    """REQ-REPORT-8847: repaired transport hashes cannot hide altered counts, gates or frozen inputs."""
    work, validation, output, raw, seal = seal_private(tmp_path, monkeypatch)
    original = deepcopy(work)
    seal()
    assert r.replay(output)
    for key in ["evaluation", "stress", "rows"]:
        work.clear()
        work.update(deepcopy(original))
        if key == "rows":
            work["rows"][0]["primitive_operations"] += 1
        elif key == "evaluation":
            work["evaluation"]["indexed_cost_benefit_score"] = (
                1 - original["evaluation"]["indexed_cost_benefit_score"]
            )
        else:
            work["stress"]["16"]["indexed_cost_benefit_score"] = 1
        seal()
        assert not r.replay(output), key
    work.clear()
    work.update(deepcopy(original))
    sealed = json.loads((raw / "sealed_predictions.json").read_bytes())
    sealed["rows"][0]["decision"] = "invented"
    sealed["commitment"] = canonical_hash(sealed["rows"])
    atomic_json(raw / "sealed_predictions.json", sealed)
    work["rows"] = m.score(sealed["rows"], json.loads((raw / "requests.json").read_bytes())["rows"])
    seal()
    assert not r.replay(output)
    work.clear()
    work.update(deepcopy(original))
    seal()
    with monkeypatch.context() as patch:
        patch.setattr(r, "sha256_file", lambda p: "changed")
        work["code_config_hashes"] = [dict(path=str(output), sha256="old")]
        seal()
        assert not r.replay(output)
    work["code_config_hashes"] = []
    seal()
    with monkeypatch.context() as patch:
        patch.setattr(r, "validate_current_work_receipt", lambda *args: ["bad"])
        assert not r.replay(output)
    frozen = json.loads((raw / "frozen_inputs.json").read_bytes())
    frozen["cap"] = 1
    atomic_json(raw / "frozen_inputs.json", frozen)
    seal()
    assert not r.replay(output)


def test_authority_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8847: absent fields, zero readiness, rejected producers and drift are distinct."""
    authority = json.loads(r.UPSTREAM.read_bytes())
    for value, check in [
        ({}, "missing_field"),
        ({"indexed_witness_ready_score": 0}, "observed_zero"),
        (dict(authority, task_id="wrong"), "rejected_producer"),
    ]:
        path = tmp_path / "authority.json"
        atomic_json(path, value)
        with monkeypatch.context() as patch:
            patch.setattr(r, "UPSTREAM", path)
            assert r.preconditions(tmp_path / check)["diagnostics"][-1]["check"] == check
    with monkeypatch.context() as patch:
        patch.setattr(r, "sha256_file", lambda *args: "changed")
        assert r.preconditions(tmp_path / "drift")["diagnostics"][-1]["error"] == "operand_drift"
    with monkeypatch.context() as patch:
        patch.setattr(m, "stress", lambda: [])
        pre = r.preconditions(tmp_path / "roster")
        assert len(pre["evaluation_worlds"]) == 240


def test_readiness_null_and_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8847: qualified honest nulls are ready; owned failures disqualify."""
    work, validation, output, raw, seal = seal_private(tmp_path, monkeypatch)
    good = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True),
        raw_references=[],
    )
    work["private_unit_limit"] = None
    work["rows"] *= 960
    assert len(work["rows"]) == 19200
    work["evaluation"]["indexed_cost_benefit_score"] = 0
    assert r.artifact(work, good, "20261105", {}, {})["verdict_class"] == "null"
    work["evaluation"]["indexed_cost_benefit_score"] = 1
    assert r.artifact(work, good, "20261105", {}, {})["verdict_class"] == "circular_positive"
    assert r.artifact(work, validation, "20261105", {}, {})["verdict_class"] == "disqualified"
    monkeypatch.setattr(r, "UPSTREAM", tmp_path / "absent")
    blocked = r.prepare(tmp_path / "blocked")
    blocked.update(code_config_hashes=[], raw_shard_hashes=[])
    assert r.artifact(blocked, validation, "20261105", {}, {})["verdict_class"] == "blocked"


def test_main_internal_failure_and_validation(tmp_path, monkeypatch):
    """REQ-REPORT-8847: failed owned execution cannot become an external block or a ready result."""
    from types import SimpleNamespace

    configured = SimpleNamespace(validate=lambda *args: dict(actual=True))
    with monkeypatch.context() as patch:
        patch.setattr(r, "namespace", lambda *args: configured)
        assert r.validate(tmp_path, tmp_path)["actual"]
        assert configured.m is m
    original = r.prepare

    def failure(raw, limit=None, measurement=True):
        if measurement:
            raise ValueError("owned_failure")
        return original(raw, limit, False)

    monkeypatch.setattr(r, "prepare", failure)
    monkeypatch.setattr(r, "validate", lambda *args: r.unqualified())
    monkeypatch.setattr(r, "publish_primary", lambda *args: dict(unit_boundary=True))
    monkeypatch.setattr(r, "HEALTH", tmp_path / "health.json")
    atomic_json(r.HEALTH, dict(passed=False))
    assert r.main(["--output", str(tmp_path / (m.NAME + ".json"))]) == 0
    for args in [
        ["--measure-arm", "always_joint"],
        ["--measure-arm", "always_joint", "--schedule-seed", "2"],
    ]:
        with pytest.raises(SystemExit):
            r.main(args)


def test_deadline_heartbeat_roster_and_child_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8847: progress never resets the deadline, and failed processes stay failures."""
    ticks = iter([0, 31, 62, 93, 124, 155, 186, 217, 248, 279, 310])
    with monkeypatch.context() as patch:
        patch.setattr(m.time, "monotonic", lambda: next(ticks))
        with pytest.raises(ValueError, match="measurement_deadline"):
            m.measure(units())
    rows = m.score(m.measure(units()), units())
    with pytest.raises(ValueError, match="roster_duplicate"):
        m.reduce(rows + [rows[0]])
    monkeypatch.setattr(r, "stream", lambda *args, **kwargs: dict(passed=False))
    with pytest.raises(ValueError, match="measurement_child_failure"):
        r.process_measure(tmp_path, units())


def test_registered_gain_and_all_attempt_denominators():
    """REQ-REPORT-8847: fixed benefit gates use paid work and all requests, including censors."""
    rows = m.score(m.measure(units()), units())
    for row in rows:
        row["primitive_operations"] = 100 if row["arm"] == "always_joint" else 50
        row["wall_time_s"] = 2 if row["arm"] == "always_joint" else 1
    result = m.reduce(rows)
    assert result["indexed_cost_benefit_score"] == 1
    assert result["paired_intervals"]["paid_operation_reduction"]["lower"] == 0.5
    assert result["wall_speed_ratio"]["lower"] == 2
    assert result["wall_speed_ratio"]["claim"]
    for row in rows:
        if row["arm"] == "indexed_witness":
            row.update(decision="insufficient", status="censored", budget_exhausted=True)
    result = m.reduce(rows)
    assert result["indexed_cost_benefit_score"] == 0
    assert result["paired_intervals"]["coverage_loss"]["upper"] == 0.25
    assert result["per_world_results"][0]["arms"]["indexed_witness"]["denominator"] == 20


def test_roster_and_validation_replay_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8847: malformed internal rosters and unproved owned qualification are rejected."""
    from types import SimpleNamespace

    original = r.previous.preconditions

    def malformed(raw):
        value = original(raw)
        atomic_json(raw / "inputs.json", dict(rows=[]))
        return value

    with monkeypatch.context() as patch:
        patch.setattr(r.previous, "preconditions", malformed)
        with pytest.raises(ValueError, match="evaluation_roster"):
            r.preconditions(tmp_path / "bad_roster")
    work, validation, output, raw, seal = seal_private(tmp_path, monkeypatch)
    validation.update(
        receipts=[dict(passed=True)], union=dict(passed=True), coverage=dict(complete=True)
    )
    seal()
    monkeypatch.setattr(
        r, "namespace", lambda *args: SimpleNamespace(validation_evidence=lambda value: False)
    )
    assert not r.replay(output)


def test_label_heartbeat(monkeypatch, capsys):
    """REQ-REPORT-8847: label verification reports actual progress inside a long loop."""
    ticks = iter([0, 31, 62, 93, 124])
    monkeypatch.setattr(m.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(m.witness.reference.base, "truth", lambda payload: "support")
    assert m.score([], units()) == []
    output = capsys.readouterr().out
    assert "phase=independent_labels completed=4 pending=0" in output
