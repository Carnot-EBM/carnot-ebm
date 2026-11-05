"""REQ-REPORT-8844: finite-population availability has explicit denominators."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import selective_availability_8844 as m
from carnot.reporting import selective_availability_runner_8844 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture():
    """Keep a tiny complete roster while using the genuine replacement RNG."""
    cohort = m.p.freeze(set(), fit_count=3, population_count=12)
    units = sorted(u["unit_id"] for u in cohort["inputs"] if u["split"] == "population")
    predictions = [
        dict(
            unit_id=u,
            cluster_id=u,
            arm=a,
            seed=s,
            probabilities=[1.0, 0.0, 0.0],
            candidate="support",
            candidate_index=0,
        )
        for a, s in m.POLICIES
        for u in units
    ]
    return cohort, predictions, {u: "support" for u in units}, m.p.draws(12, replicates=2)


def test_intervals_and_nulls():
    """SCENARIO-REPORT-8844-CONTRACT: zero denominators never imply zero risk."""
    assert m.interval(0, 0) == dict(
        numerator=0, denominator=0, estimate=None, lower=None, upper=None
    )
    assert m.interval(30, 0)["upper"] > 0.01
    assert m.interval(30, 30)["lower"] < 1
    assert 0 < m.interval(30, 2)["lower"] < m.interval(30, 2)["upper"] < 1
    assert m.p.FLOOR == 185
    with pytest.raises(ValueError, match="counts"):
        m.interval(1, 2)


def test_selection_and_evaluation():
    """REQ-REPORT-8844: select before census; retain repeated draws and refusals."""
    _, predictions, truth, trace = fixture()
    selected = m.select(predictions, truth, trace)
    assert len(selected["selected_cutoff_rows"]) == 42
    assert all(
        x["threshold"] is None for x in selected["selected_cutoff_rows"] if x["budget"] < 360
    )
    assert all(
        x["threshold"] == 0.5 for x in selected["selected_cutoff_rows"] if x["budget"] == 360
    )
    result = m.evaluate(predictions, truth, trace, selected)
    assert len(result["rows"]) == 42
    assert len(result["per_draw_results"]) == 42 * 240
    assert len(result["population_score_rows"]) == 12 * 7
    assert result["rows"][0]["served_risk"] is None
    assert any(x["census_risk"] == 0 for x in result["rows"])
    assert all(x["denominator"] == 240 for x in result["rows"])
    assert len(result["paired_comparisons"]) == 10
    assert result["learned_selective_benefit_score"] == 0
    wrong = dict.fromkeys(truth, "refute")
    bad = m.evaluate(predictions, wrong, trace, selected)
    assert any(x["certificate_failure"] for x in bad["certificate_event_rows"])
    assert all(
        x["threshold"] is None for x in m.select(predictions, wrong, trace)["selected_cutoff_rows"]
    )
    with pytest.raises(ValueError, match="roster"):
        m.select(predictions[:-1], truth, trace)
    malicious = deepcopy(trace)
    malicious["replicates"][0]["calibration"]["indices"] = [0] * 360
    with pytest.raises(ValueError, match="draw"):
        m.select(predictions, truth, malicious)


def test_training_label_control():
    """REQ-REPORT-8844: shuffled fitting labels cannot grant readiness."""
    cohort, _, _, _ = fixture()
    truth = m.p.labels([u for u in cohort["inputs"] if u["split"] == "fit"])
    fitted = dict(training_labels=truth)
    m.audit_fit(cohort, fitted)
    values = list(truth.values())
    fitted["training_labels"] = dict(zip(truth, values[1:] + values[:1], strict=True))
    with pytest.raises(ValueError, match="fitting_label"):
        m.audit_fit(cohort, fitted)


def test_failed_fit_keeps_arm(tmp_path, monkeypatch):
    """REQ-REPORT-8844: a failed fit keeps every replicate and budget slot."""
    cohort, predictions, truth, trace = fixture()

    def fit(c, seeds):
        if seeds == [11]:
            raise ValueError("failed_training")
        return dict(
            parameters=[dict(seed=seeds[0], models=[])],
            training_labels={},
            numerical_training_calls=3,
        )

    monkeypatch.setattr(m.p, "fit", fit)
    fitted = m.fit(cohort)
    assert fitted["failed_seeds"] == [dict(seed=11, error="failed_training")]
    assert fitted["numerical_training_calls"] == 12
    monkeypatch.setattr(m.p, "predict", lambda *a: [r for r in predictions if r["seed"] != 11])
    failed = m.predict(cohort, fitted)
    result = m.evaluate(failed, truth, trace, m.select(failed, truth, trace))
    assert len(result["rows"]) == 42
    assert sum(row["status"] == "failed" for row in result["rows"]) == 6
    assert all(not row["certificate_issued"] for row in result["rows"] if row["status"] == "failed")


def test_private_cli(tmp_path):
    """SCENARIO-REPORT-8844-CONTRACT: actual CLI rejects cold tampering."""
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    output = tmp_path / (m.NAME + ".json")
    for options, code in [
        (["--date", "20261104"], 2),
        (["--private-output", str(output), "--upstream", str(tmp_path / "absent")], 0),
    ]:
        child = subprocess.run(
            command + options, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120
        )
        assert child.returncode == code, child.stdout + child.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked"
    assert value["selective_rows_ready_score"] == 0
    assert value["inference_substrate_class"] == "blocked_no_run"
    for tamper, code in [(False, 0), (True, 1)]:
        if tamper:
            value["completed_count"] = 1
            atomic_json(output, value)
        child = subprocess.run(
            command + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert child.returncode == code, child.stdout + child.stderr


def test_authentication(tmp_path):
    """REQ-REPORT-8844: absent fields, zero gates and rejected producers stay blocked."""
    path = tmp_path / "source.json"
    assert not r.preconditions(tmp_path / "raw", path)["external_ready"]
    for value in [
        {},
        {"selective_protocol_ready_score": 0},
        {"selective_protocol_ready_score": 1, "task_id": "foreign"},
    ]:
        atomic_json(path, value)
        assert not r.preconditions(tmp_path / "raw", path)["external_ready"]


def test_bound_operands(tmp_path, monkeypatch):
    """REQ-REPORT-8844: byte changes and protocol drift cannot gain readiness."""
    protocol = tmp_path / "protocol.json"
    atomic_json(protocol, dict(frozen=True))
    source = tmp_path / "source.json"
    value = dict(
        task_id="exp8842-selective-population",
        selective_protocol_ready_score=1,
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[r.reference(protocol)],
    )
    atomic_json(source, value)
    monkeypatch.setattr(r, "PIN", r.sha256_file(source))
    monkeypatch.setattr(r, "PROTOCOL_PIN", r.sha256_file(protocol))
    assert r.preconditions(tmp_path / "raw", source)["external_ready"]
    monkeypatch.setattr(r, "PROTOCOL_PIN", "bad")
    assert not r.preconditions(tmp_path / "raw", source)["external_ready"]
    protocol.write_text("changed")
    assert not r.preconditions(tmp_path / "raw", source)["external_ready"]


def test_measure_publish_and_attacks(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8844-CONTRACT: measured operands reject semantic rehash attacks."""
    cohort, predictions, _, trace = fixture()
    upstream = {}
    for key, value in [("population_manifest", cohort), ("draw_manifest", trace)]:
        path = tmp_path / (key + ".json")
        atomic_json(path, value)
        upstream[key] = r.reference(path)
    real_audit, real_fit = m.p.audit_draws, m.p.fit
    monkeypatch.setattr(m.p, "audit_draws", lambda t, size: real_audit(t, t["population_size"]))
    monkeypatch.setattr(m.p, "fit", lambda c, seeds: real_fit(c, seeds=seeds, steps=1))
    monkeypatch.setattr(m.p, "predict", lambda *a: deepcopy(predictions))
    pre = dict(
        external_ready=True,
        resources=dict(passed=True, measurement_deadline_s=60),
        references=[],
        diagnostics=[],
        frozen_protocol=None,
    )

    def setup(raw, source):
        raw.mkdir(parents=True)
        atomic_json(raw / "upstream.json", upstream)
        return deepcopy(pre)

    monkeypatch.setattr(r, "preconditions", setup)
    good = dict(
        coverage=dict(complete=True), union=dict(passed=True), receipts=[], raw_references=[]
    )
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(good))
    output = tmp_path / (m.NAME + ".json")
    log = tmp_path / "log"
    log.write_text("actual test control")
    monkeypatch.setattr(
        r, "stream", lambda *a, **kw: dict(passed=False, stdout_path=str(log), stderr_path=str(log))
    )
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] is False
    assert r.replay(output)["passed"]
    assert r.main(["--cold-replay", str(output)]) == 0
    for key, changed, message in [
        ("experiment_id", 9999, "identity"),
        ("completed_count", 99, "count"),
        ("certificate_firing_count", 999, "reduction"),
        ("current_work_receipt", {}, "receipt"),
        ("selective_rows_ready_score", 1, "readiness"),
        ("code_config_hashes", [dict(path=str(log), sha256="bad")], "operand"),
    ]:
        bad = deepcopy(value)
        bad[key] = changed
        del bad["reproducibility_checksum"]
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        with pytest.raises(ValueError, match=message):
            r.replay(output)
    bad = deepcopy(value)
    bad["reproducibility_checksum"] = "bad"
    atomic_json(output, bad)
    assert r.main(["--cold-replay", str(output)]) == 1
    raw = Path(value["measurement_manifest"]["path"]).parent
    scores = raw / "scores.json"
    original = scores.read_bytes()
    atomic_json(scores, dict(rows=[]))
    bad = deepcopy(value)
    bad["raw_shard_hashes"] = [
        r.reference(scores) if ref["path"] == str(scores) else ref
        for ref in bad["raw_shard_hashes"]
    ]
    del bad["reproducibility_checksum"]
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(output, bad)
    with pytest.raises(ValueError, match="score_drift"):
        r.replay(output)
    scores.write_bytes(original)
    atomic_json(output, value)
    for filename, message in [("draws.json", "frozen_population"), ("fit.json", "fitting_label")]:
        operand = raw / filename
        saved = operand.read_bytes()
        altered = json.loads(saved)
        if filename == "draws.json":
            altered["replicates"][0]["calibration"]["indices"] = [0] * 360
        else:
            labels = altered["training_labels"]
            values = list(labels.values())
            altered["training_labels"] = dict(zip(labels, values[1:] + values[:1], strict=True))
        atomic_json(operand, altered)
        bad = deepcopy(value)
        bad["raw_shard_hashes"] = [
            r.reference(operand) if ref["path"] == str(operand) else ref
            for ref in bad["raw_shard_hashes"]
        ]
        del bad["reproducibility_checksum"]
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        with pytest.raises(ValueError, match=message):
            r.replay(output)
        operand.write_bytes(saved)
    atomic_json(output, value)
    pre["resources"]["measurement_deadline_s"] = -1
    with pytest.raises(ValueError, match="deadline"):
        r.measure(raw, pre)
    monkeypatch.setattr(r, "measure", lambda *a: (_ for _ in ()).throw(ValueError("owned_failure")))
    assert r.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["owned_errors"] == ["owned_failure"]
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])


def test_validation_adapter(tmp_path, monkeypatch):
    """REQ-REPORT-8844: reuse the unchanged consumer coverage adapter."""
    from types import SimpleNamespace

    checker = SimpleNamespace(validate=lambda *a: dict(passed=True))
    monkeypatch.setattr(r, "namespace", lambda *a: checker)
    assert r.validate(tmp_path, tmp_path)["passed"]
    assert checker.m is m
