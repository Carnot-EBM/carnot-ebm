"""REQ-REPORT-8857: independent stages must preserve paid labels and refused rules."""

from copy import deepcopy
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import binom

from carnot.reporting import split_certificate_methods_8857 as m
from carnot.reporting import split_certificate_runner_8857 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(size=12):
    """A complete deterministic roster permits exact tests without a fitted model."""
    scores = [
        dict(
            unit_id=f"u{i:04d}",
            cluster_id=f"w{i}",
            arm=a,
            seed=s,
            candidate="support",
            probabilities=[0.96, 0.02, 0.02],
            status="completed",
            missing_reason=None,
        )
        for a, s in m.POLICIES
        for i in range(size)
    ]
    truth = {f"u{i:04d}": "support" for i in range(size)}
    return scores, truth, m.draws(size, replicates=2)


def test_protocol_draws_and_disjoint_freeze():
    """SCENARIO-REPORT-8857-CONTRACT: population construction precedes replacement draws."""
    protocol = m.protocol()
    assert protocol["familywise_allocation"]["total"] == 0.01
    assert len(protocol["policy_menu"]) == 7
    trace = m.draws(12)
    assert len(trace["replicates"]) == 60
    for rep in trace["replicates"]:
        cal, ev = rep["calibration"], rep["evaluation"]
        assert len(cal) == 720 and len(ev) == 240
        assert len({x["draw_id"] for x in cal + ev}) == 960
        assert len({x["index"] for x in cal}) < 720
        assert cal[:240] != cal[240:480]
    assert trace == m.draws(12)
    first = m.freeze(set(), population_count=4, fit_count=2)
    exposed = {w["structure"] for w in first["worlds"]}
    second = m.freeze(exposed, population_count=4, fit_count=2)
    assert exposed.isdisjoint(w["structure"] for w in second["worlds"])
    assert all(w["world_id"].startswith("v758/8857/") for w in second["worlds"])


def test_exhaustive_binomial_boundaries():
    """REQ-REPORT-8857: inversion agrees with exhaustive Bernoulli enumeration."""
    for alpha in [0.01 / 84, 0.01 / 14, 0.2]:
        assert m.upper(0, 0, alpha) == 1
        for n in range(1, 7):
            for bits in product([0, 1], repeat=n):
                errors = sum(bits)
                bound = m.upper(n, errors, alpha)
                assert (bound <= 0.05) == (binom.cdf(errors, n, 0.05) <= alpha)
                if errors == n:
                    assert bound == 1
            for risk in [0.1, 0.5, 0.9]:
                failure = sum(
                    risk ** sum(b) * (1 - risk) ** (n - sum(b))
                    for b in product([0, 1], repeat=n)
                    if m.upper(n, sum(b), alpha) < risk
                )
                assert failure <= alpha + 1e-12
    for n, e, alpha in [(-1, 0, 0.1), (1, 2, 0.1), (1, 0, 0)]:
        with pytest.raises(ValueError, match="binomial"):
            m.upper(n, e, alpha)


def test_selection_certification_independence_and_refusal():
    """SCENARIO-REPORT-8857-CONTRACT: certification cannot replace its selected rule."""
    scores, truth, trace = fixture(1024)
    selected = m.select(scores, truth, trace)
    assert all(x["threshold"] == 0.95 for x in selected["choices"])
    changed = deepcopy(trace)
    for rep in changed["replicates"]:
        for draw in rep["calibration"][240:]:
            draw["index"] = 0
    assert m.select(scores, truth, changed) == selected
    certificates = m.certify(scores, truth, trace, selected)
    assert all(x["certificate_issued"] for x in certificates["choices"])
    damaged = deepcopy(selected)
    damaged["choices"][0]["threshold"] = 0.5
    with pytest.raises(ValueError, match="selection"):
        m.certify(scores, truth, trace, damaged)
    broken_truth = dict.fromkeys(truth, "refute")
    refused = m.certify(scores, broken_truth, trace, selected)
    assert not any(x["certificate_issued"] for x in refused["choices"])
    assert all(
        x["selected_threshold"] == 0.95 for x in refused["choices"] if x["protocol"] == "split"
    )
    rows = m.evaluate(scores, truth, trace, refused)
    assert all(x["metric"] == 0 and x["served_risk"] is None for x in rows)
    census = m.census(scores, truth, rows)
    assert all(x["census_risk"] is None for x in census)
    for score in scores:
        score["probabilities"] = [0, 0, 1]
    empty = m.select(scores, truth, trace)
    assert all(x["threshold"] is None for x in empty["choices"])
    cert = m.certify(scores, truth, trace, empty)
    assert all(x["threshold"] is None for x in cert["choices"])
    with pytest.raises(ValueError, match="score"):
        m.select([dict(x, probabilities=[float("nan"), 0, 0]) for x in scores], truth, trace)
    with pytest.raises(ValueError, match="roster"):
        m.select(scores[:-1], truth, trace)


def test_evaluation_bootstrap_and_duplicate_denominators():
    """REQ-REPORT-8857: all policies share each replicate and refusals contribute zero."""
    scores, truth, trace = fixture()
    cert = m.certify(scores, truth, trace, m.select(scores, truth, trace))
    rows = m.census(scores, truth, m.evaluate(scores, truth, trace, cert))
    assert len(rows) == 28
    assert all(x["denominator"] == 240 and x["accepted"] == 240 for x in rows)
    result = m.summarize(rows)
    assert result["primary_contrast"]["estimate"] == 0
    assert result["protocol_benefit_score"] == 0
    assert len(result["policy_summaries"]) == 7
    for row in rows:
        row["metric"] = float(row["protocol"] == "split")
    assert m.summarize(rows)["primary_contrast"]["lower_99"] == 1
    assert m.summarize(rows)["protocol_benefit_score"] == 1
    assert np.isfinite(result["primary_contrast"]["upper_99"])


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    """Tiny authenticated inputs keep integration tests independent of live primaries."""
    monkeypatch.setattr(m.old.p, "labels", lambda units: {u["unit_id"]: "support" for u in units})
    monkeypatch.setattr(
        m.old,
        "fit",
        lambda cohort: dict(
            parameters=[],
            failed_seeds=[],
            numerical_training_calls=0,
            training_labels=m.old.p.labels([u for u in cohort["inputs"] if u["split"] == "fit"]),
        ),
    )
    monkeypatch.setattr(
        m.old,
        "predict",
        lambda cohort, fitted: [
            dict(
                unit_id=u["unit_id"],
                cluster_id=u["cluster_id"],
                arm=a,
                seed=s,
                candidate=u["candidates"][u["candidate_index"]],
                probabilities=[0.96, 0.02, 0.02],
                status="completed",
            )
            for a, s in m.POLICIES
            for u in cohort["inputs"]
            if u["split"] == "population"
        ],
    )
    cohort = m.freeze(set(), population_count=3, fit_count=2)
    population = tmp_path / "population.json"
    atomic_json(population, dict(value=cohort))
    source = tmp_path / "upstream.json"
    value = dict(
        task_id="exp8855-selective-ceiling",
        split_study_feasible_score=1,
        ceiling_rows_ready_score=1,
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[r.reference(population)],
    )
    atomic_json(source, value)
    monkeypatch.setattr(r, "PIN", r.sha256_file(source))
    monkeypatch.setattr(r, "SIZE", 12)
    monkeypatch.setattr(r, "REPLICATES", 2)
    return source, value


def test_authentication_and_measurement(tmp_path, monkeypatch, upstream):
    """REQ-REPORT-8857: missing, zero, foreign and changed inputs cannot authorize work."""
    source, value = upstream
    raw = tmp_path / "raw"
    pre = r.preconditions(raw, source)
    assert pre["external_ready"]
    measured = r.measure(raw)
    assert len(measured["rows"]) == 28
    assert r.reduction(raw) == measured
    population = r.load(r.reference(raw / "population.json"))
    assert len([w for w in population["worlds"] if w["split"] == "population"]) == 12
    assert all(x["status"] == "completed" for x in measured["rows"])
    assert (raw / "selection.json").stat().st_mtime_ns <= (
        raw / "certificates.json"
    ).stat().st_mtime_ns
    for changed, expected in [
        ({}, "missing_field"),
        (dict(value, split_study_feasible_score=0), "observed_zero"),
        (dict(value, task_id="foreign"), "rejected_producer"),
    ]:
        atomic_json(source, changed)
        assert r.preconditions(tmp_path / expected, source)["diagnostics"][0]["check"] == expected
    assert (
        r.preconditions(tmp_path / "missing", tmp_path / "absent")["diagnostics"][0]["check"]
        == "missing_file"
    )
    atomic_json(source, value)
    Path(value["raw_shard_hashes"][0]["path"]).write_text("tampered")
    assert not r.preconditions(tmp_path / "tampered", source)["external_ready"]
    with monkeypatch.context() as control:
        control.setattr(r, "MEASUREMENT_DEADLINE_S", -1)
        with pytest.raises(ValueError, match="deadline"):
            r.measure(raw)


def test_publication_and_semantic_tamper(tmp_path, monkeypatch, upstream):
    """SCENARIO-REPORT-8857-CONTRACT: rehashing cannot change the frozen study or result."""
    source, _ = upstream
    good = dict(
        coverage=dict(complete=True), union=dict(passed=True), receipts=[], raw_references=[]
    )
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(good))
    log = tmp_path / "log"
    log.write_text("test control")
    health = tmp_path / "health.json"
    atomic_json(
        health,
        dict(
            passed=False, scope="separate_global_health", stdout_path=str(log), stderr_path=str(log)
        ),
    )
    monkeypatch.setattr(r, "HEALTH", health)
    monkeypatch.setattr(
        r, "stream", lambda *a, **kw: dict(passed=True, stdout_path=str(log), stderr_path=str(log))
    )
    output = tmp_path / (m.NAME + ".json")
    args = ["--output", str(output), "--upstream", str(source)]
    assert r.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["split_methods_ready_score"] == 1
    assert r.main(["--cold-replay", str(output)]) == 0
    for key, changed in [
        ("experiment_id", 1),
        ("completed_count", 0),
        ("protocol_hash", "bad"),
        ("current_work_receipt", {}),
        ("protocol_benefit_score", 9),
        ("rows", []),
        ("split_methods_ready_score", 9),
        ("required_checks_passed", False),
    ]:
        bad = deepcopy(value)
        bad[key] = changed
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        with pytest.raises(ValueError):
            r.replay(output)
    atomic_json(output, dict(value, reproducibility_checksum="bad"))
    assert r.main(["--cold-replay", str(output)]) == 1
    atomic_json(output, value)
    raw = Path(value["population_manifest"]["path"]).parent
    saved = (raw / "draws.json").read_bytes()
    atomic_json(raw / "draws.json", dict(changed=True))
    with pytest.raises(ValueError, match="operand"):
        r.replay(output)
    (raw / "draws.json").write_bytes(saved)
    with monkeypatch.context() as control:
        control.setattr(r, "PIN", "bad")
        with pytest.raises(ValueError, match="pin"):
            r.replay(output)
    monkeypatch.setattr(
        r, "stream", lambda *a, **kw: dict(passed=False, stdout_path=str(log), stderr_path=str(log))
    )
    assert r.main(args) == 0
    assert json.loads(output.read_bytes())["split_methods_ready_score"] == 0
    monkeypatch.setattr(r, "measure", lambda *a: (_ for _ in ()).throw(ValueError("owned_failure")))
    assert r.main(args) == 0
    assert json.loads(output.read_bytes())["owned_errors"] == ["owned_failure"]


def test_private_cli_and_cold_replay(tmp_path):
    """REQ-REPORT-8857: private CLI works outside checkout without PYTHONPATH."""
    output = tmp_path / (m.NAME + ".json")
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for argv, code in [
        (["--date", "20261105"], 2),
        (["--private-output", str(m.ROOT / "results" / output.name)], 2),
        (["--private-output", str(output), "--upstream", str(tmp_path / "absent")], 0),
        (["--cold-replay", str(output)], 0),
    ]:
        child = subprocess.run(
            command + argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )
        assert child.returncode == code, child.stdout + child.stderr
    value = json.loads(output.read_bytes())
    assert value["honest_verdict"].startswith("complete_blocked_missing_file")
    assert len(value["rows"]) == 840 and value["completed_count"] == 0


def test_stage_rehash_rejection_and_deadlines(tmp_path, monkeypatch, upstream):
    """REQ-REPORT-8857: coherent byte updates cannot change frozen draws or stage order."""
    source, _ = upstream
    raw = tmp_path / "raw"
    r.preconditions(raw, source)
    r.measure(raw)
    saved = {p.name: p.read_bytes() for p in raw.glob("*.json")}
    for name, change, message in [
        ("protocol", lambda v: dict(v, seed=0), "protocol"),
        ("draws", lambda v: dict(v, seed=0), "population_or_draw"),
        (
            "population",
            lambda v: dict(v, exposed_invariants=[v["worlds"][0]["structure"]]),
            "population_or_draw",
        ),
        ("access_order", lambda v: dict(events=list(reversed(v["events"]))), "access_order"),
        ("selection", lambda v: dict(v, choices=[]), "stage_reduction"),
    ]:
        value = json.loads(saved[name + ".json"])
        atomic_json(raw / (name + ".json"), change(value))
        if name == "selection":
            order = json.loads(saved["access_order.json"])
            for event in order["events"]:
                if event["stage"] == name:
                    event.update(r.reference(raw / (name + ".json")))
            atomic_json(raw / "access_order.json", order)
        with pytest.raises(ValueError, match=message):
            r.reduction(raw)
        for filename, data in saved.items():
            (raw / filename).write_bytes(data)
    with monkeypatch.context() as control:
        clock = iter([0.0, 2.0])
        control.setattr(r, "MEASUREMENT_DEADLINE_S", 1)
        control.setattr(r.time, "monotonic", lambda: next(clock))
        with pytest.raises(ValueError, match="measurement_deadline"):
            r.measure(raw)


def test_validation_adapter(tmp_path, monkeypatch):
    """REQ-REPORT-8857: unchanged shared readers receive only the current owned paths."""
    from types import SimpleNamespace

    checker = SimpleNamespace(validate=lambda *a: dict(passed=True))
    monkeypatch.setattr(r, "namespace", lambda *a: checker)
    assert r.validate(tmp_path, tmp_path)["passed"]
    assert checker.m is m


def test_bound_schema_and_label_replay(tmp_path, monkeypatch, upstream):
    """REQ-REPORT-8857: frozen gates and input-derived labels survive full artifact rehashes."""
    source, _ = upstream
    good = dict(
        coverage=dict(complete=True), union=dict(passed=True), receipts=[], raw_references=[]
    )
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(good))
    log = tmp_path / "log"
    log.write_text("test control")
    monkeypatch.setattr(
        r, "stream", lambda *a, **kw: dict(passed=True, stdout_path=str(log), stderr_path=str(log))
    )
    output = tmp_path / (m.NAME + ".json")
    r.main(["--output", str(output), "--upstream", str(source)])
    value = json.loads(output.read_bytes())
    for key, changed in [
        ("acceptance_gates", {}),
        ("familywise_allocation", {}),
        ("verdict_class", "blocked"),
        ("run_date", "20261105"),
    ]:
        bad = dict(value, **{key: changed})
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        with pytest.raises(ValueError):
            r.replay(output)
    bad = deepcopy(value)
    bad["completed_count"] = 0
    bad["failed_count"] = len(bad["rows"])
    bad["rows"] = [dict(row, status="failed") for row in bad["rows"]]
    bad["split_methods_ready_score"] = 0
    bad["protocol_benefit_score"] = 1
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(output, bad)
    with pytest.raises(ValueError, match="missing_measurement"):
        r.replay(output)
    raw = Path(value["population_manifest"]["path"]).parent
    retained = r.load(r.reference(raw / "census.json"))
    atomic_json(raw / "census.json", dict(retained, numerical_training_calls=99))
    order = r.load(r.reference(raw / "access_order.json"))
    for event in order["events"]:
        if event["stage"] == "census":
            event.update(r.reference(raw / "census.json"))
    atomic_json(raw / "access_order.json", order)
    bad = deepcopy(value)
    bad["raw_shard_hashes"] = [r.reference(Path(ref["path"])) for ref in bad["raw_shard_hashes"]]
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(output, bad)
    with pytest.raises(ValueError, match="retained_reduction"):
        r.replay(output)
