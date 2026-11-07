"""REQ-REPORT-8859: sealed split calibration keeps refusals and all paid draws."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import split_certificate_evaluation_8859 as m
from carnot.reporting import split_certificate_evaluation_runner_8859 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(size=12):
    """Uniform scores make accepted counts independent of replacement collisions."""
    scores = [
        dict(
            unit_id=f"u{i:04d}",
            cluster_id=f"w{i}",
            arm=a,
            seed=s,
            candidate="support",
            probabilities=[0.96, 0.02, 0.02],
            status="completed",
        )
        for a, s in m.POLICIES
        for i in range(size)
    ]
    truth = {f"u{i:04d}": "support" for i in range(size)}
    return scores, truth, m.draws(size, replicates=2)


def test_protocol_and_denominators():
    """SCENARIO-REPORT-8859-CONTRACT: original allocation survives the evaluation bootstrap setting."""
    assert canonical_hash(m.registered.protocol()) == m.METHOD_PROTOCOL_PIN
    assert m.protocol()["bootstrap"]["draws"] == 10000
    assert m.protocol()["familywise_allocation"] == m.registered.protocol()["familywise_allocation"]
    scores, truth, trace = fixture()
    selection = m.select(scores, truth, trace)
    cert = m.certify(scores, truth, trace, selection)
    rows = m.census(scores, truth, m.evaluate(scores, truth, trace, cert))
    result = m.summarize(rows)
    assert result["primary_contrast"]["estimate"] == 0
    assert result["split_protocol_benefit_score"] == 0
    assert len(result["paired_replicate_rows"]) == 2
    assert len(result["firing_intervals"]) == 14
    assert result["familywise_failure_interval"]["denominator"] == 2
    assert all(x["denominator"] == 2 for x in result["firing_intervals"])
    draws = m.per_draw(scores, truth, trace, cert)
    assert len(draws) == 28 * 240
    assert all(x["accepted"] and not x["error"] for x in draws)
    assert len({x["draw_id"] for x in draws}) == 480
    for row in rows:
        row["metric"] = float(row["protocol"] == "split")
    assert m.summarize(rows)["split_protocol_benefit_score"] == 1
    rows[0]["census_risk"] = 0.2
    assert m.summarize(rows)["split_protocol_benefit_score"] == 0


def test_failed_fit_refusal_and_empty_risk():
    """REQ-REPORT-8859: failed seeds remain present and serve no answers."""
    scores, truth, trace = fixture()
    scores = [
        dict(x, probabilities=None, status="failed", missing_reason="fit_failed")
        if x["arm"] == "learned_energy"
        else dict(x, probabilities=[0, 0, 1])
        for x in scores
    ]
    cert = m.certify(scores, truth, trace, m.select(scores, truth, trace))
    rows = m.census(scores, truth, m.evaluate(scores, truth, trace, cert))
    assert len(rows) == 28
    assert all(x["metric"] == 0 and x["served_risk"] is None for x in rows)
    assert sum(x["status"] == "failed" for x in rows) == 20
    assert all(x["estimate"] is None for x in m.summarize(rows)["conditional_failure_intervals"])
    assert all(
        not x["accepted"] and x["error"] is None for x in m.per_draw(scores, truth, trace, cert)
    )
    assert sum(x["status"] == "failed" for x in m.per_draw(scores, truth, trace, cert)) == 4800


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    """Small local worlds exercise custody and phase ordering without expensive training."""
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
    population = tmp_path / "population.json"
    atomic_json(population, m.registered.freeze(set(), population_count=3, fit_count=2))
    source = tmp_path / "source.json"
    value = dict(
        task_id="exp8857-split-certificate-methods",
        split_methods_ready_score=1,
        protocol_hash=m.METHOD_PROTOCOL_PIN,
        population_manifest=r.reference(population),
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[r.reference(population)],
    )
    atomic_json(source, value)
    with monkeypatch.context() as control:
        atomic_json(source, dict(value, protocol_hash="bad"))
        control.setattr(r, "PIN", r.sha256_file(source))
        assert (
            r.preconditions(tmp_path / "protocol", source)["diagnostics"][0]["check"]
            == "rejected_producer"
        )
    atomic_json(source, value)
    monkeypatch.setattr(r, "PIN", r.sha256_file(source))
    monkeypatch.setattr(r, "SIZE", 12)
    monkeypatch.setattr(r, "REPLICATES", 2)
    return source, value


def test_preconditions_measure_and_replay(tmp_path, monkeypatch, upstream):
    """REQ-REPORT-8859: phase commitments and draw decisions replay from bound bytes."""
    source, value = upstream
    raw = tmp_path / "raw"
    assert r.preconditions(raw, source)["external_ready"]
    result = r.measure(raw)
    assert len(result["rows"]) == 28
    assert r.reduction(raw) == result
    assert len(r.load(result["per_draw_results"])["rows"]) == 6720
    for altered, check in [
        ({}, "missing_field"),
        (dict(value, split_methods_ready_score=0), "observed_zero"),
        (dict(value, task_id="foreign"), "rejected_producer"),
        (dict(value, protocol_hash="bad"), "rejected_producer"),
    ]:
        atomic_json(source, altered)
        assert r.preconditions(tmp_path / check, source)["diagnostics"][0]["check"] == check
    assert (
        r.preconditions(tmp_path / "missing", tmp_path / "absent")["diagnostics"][0]["check"]
        == "missing_file"
    )
    atomic_json(source, value)
    Path(value["population_manifest"]["path"]).write_text("changed")
    assert not r.preconditions(tmp_path / "changed", source)["external_ready"]
    with monkeypatch.context() as control:
        control.setattr(r, "MEASUREMENT_DEADLINE_S", -1)
        with pytest.raises(ValueError, match="deadline"):
            r.measure(raw)


def test_publication_tamper_and_owned_failures(tmp_path, monkeypatch, upstream):
    """SCENARIO-REPORT-8859-CONTRACT: rehashing cannot change scientific rows or readiness."""
    source, _ = upstream
    good = dict(
        coverage=dict(complete=True), union=dict(passed=True), receipts=[], raw_references=[]
    )
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(good))
    log = tmp_path / "log"
    log.write_text("control")
    health = tmp_path / "health.json"
    atomic_json(
        health,
        dict(
            scope="separate_global_health", passed=False, stdout_path=str(log), stderr_path=str(log)
        ),
    )
    monkeypatch.setattr(r, "HEALTH", health)
    monkeypatch.setattr(
        r, "stream", lambda *a, **kw: dict(passed=True, stdout_path=str(log), stderr_path=str(log))
    )
    output = tmp_path / (m.NAME + ".json")
    argv = ["--output", str(output), "--upstream", str(source)]
    assert r.main(argv) == 0
    value = json.loads(output.read_bytes())
    assert value["selective_rows_ready_score"] == 1
    assert r.main(["--cold-replay", str(output)]) == 0
    for key, changed in [
        ("experiment_id", 1),
        ("completed_count", 0),
        ("protocol_hash", "bad"),
        ("current_work_receipt", {}),
        ("split_protocol_benefit_score", 9),
        ("rows", []),
        ("selective_rows_ready_score", 9),
        ("required_checks_passed", False),
        ("acceptance_gates", {}),
        ("run_date", "20261105"),
        ("protocol", {}),
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
        with pytest.raises(ValueError, match="producer_pin"):
            r.replay(output)
    missing = dict(
        value,
        population_manifest=None,
        selective_rows_ready_score=0,
        split_protocol_benefit_score=1,
    )
    missing.pop("reproducibility_checksum")
    missing["reproducibility_checksum"] = canonical_hash(missing)
    atomic_json(output, missing)
    with pytest.raises(ValueError, match="missing_measurement"):
        r.replay(output)
    atomic_json(output, value)
    old_census, old_order = (
        (raw / "census.json").read_bytes(),
        (raw / "access_order.json").read_bytes(),
    )
    retained = json.loads(old_census)
    retained["numerical_training_calls"] = 99
    atomic_json(raw / "census.json", retained)
    order = json.loads(old_order)
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
    (raw / "census.json").write_bytes(old_census)
    (raw / "access_order.json").write_bytes(old_order)
    monkeypatch.setattr(
        r, "stream", lambda *a, **kw: dict(passed=False, stdout_path=str(log), stderr_path=str(log))
    )
    assert r.main(argv) == 0
    assert json.loads(output.read_bytes())["selective_rows_ready_score"] == 0
    monkeypatch.setattr(r, "measure", lambda *a: (_ for _ in ()).throw(ValueError("owned_failure")))
    assert r.main(argv) == 0
    assert json.loads(output.read_bytes())["owned_errors"] == ["owned_failure"]


def test_private_cli(tmp_path):
    """REQ-REPORT-8859: direct CLI and replay work outside checkout without PYTHONPATH."""
    output = tmp_path / (m.NAME + ".json")
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for args, code in [
        (["--date", "20261105"], 2),
        (["--private-output", str(m.ROOT / "results" / output.name)], 2),
        (["--private-output", str(output), "--upstream", str(tmp_path / "absent")], 0),
        (["--cold-replay", str(output)], 0),
    ]:
        child = subprocess.run(
            command + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )
        assert child.returncode == code, child.stdout + child.stderr
    value = json.loads(output.read_bytes())
    assert value["honest_verdict"].startswith("complete_blocked_missing_file")
    assert len(value["rows"]) == 840 and value["completed_count"] == 0


def test_stage_semantics_and_deadline(tmp_path, monkeypatch, upstream):
    """REQ-REPORT-8859: refreshed hashes cannot authorize changed labels or draw decisions."""
    source, _ = upstream
    raw = tmp_path / "raw"
    r.preconditions(raw, source)
    r.measure(raw)
    saved = {p.name: p.read_bytes() for p in raw.glob("*.json")}
    for name, alter in [
        ("scores", lambda x: dict(rows=[])),
        ("protocol", lambda x: dict(x, seed=0)),
        ("draws", lambda x: dict(x, seed=0)),
        ("access_order", lambda x: dict(events=list(reversed(x["events"])))),
        ("selection", lambda x: dict(x, choices=[])),
        ("census_labels", lambda x: dict(labels={k: "unknown" for k in x["labels"]})),
        ("per_draw", lambda x: dict(rows=[])),
    ]:
        atomic_json(raw / (name + ".json"), alter(json.loads(saved[name + ".json"])))
        with pytest.raises(ValueError):
            r.reduction(raw)
        for filename, data in saved.items():
            (raw / filename).write_bytes(data)
    with monkeypatch.context() as control:
        clock = iter([0.0, 2.0])
        control.setattr(r, "MEASUREMENT_DEADLINE_S", 1)
        control.setattr(r.time, "monotonic", lambda: next(clock))
        with pytest.raises(ValueError, match="deadline"):
            r.measure(raw)


def test_validation_adapter(tmp_path, monkeypatch):
    """REQ-REPORT-8859: shared checks retain their original private historical consumers."""
    from types import SimpleNamespace

    checker = SimpleNamespace(validate=lambda *a: dict(passed=True))
    monkeypatch.setattr(r, "namespace", lambda *a: checker)
    assert r.validate(tmp_path, tmp_path)["passed"]
    assert checker.m is m
