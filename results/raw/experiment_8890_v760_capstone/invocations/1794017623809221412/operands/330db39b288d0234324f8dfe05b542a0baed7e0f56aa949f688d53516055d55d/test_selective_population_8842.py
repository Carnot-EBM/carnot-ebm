"""REQ-REPORT-8842: independent draws, sealed policies and honest certificates."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest
from scipy.stats import binomtest

from carnot.reporting import selective_population_8842 as m
from carnot.reporting import selective_population_runner_8842 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_exact_boundary():
    """SCENARIO-REPORT-8842-CONTRACT: zero answers never certify zero risk."""
    assert m.upper(0, 0) == m.upper(10, 10) == 1
    assert m.upper(m.FLOOR, 0) <= 0.05 < m.upper(m.FLOOR - 1, 0)
    assert m.upper(200, 2) == pytest.approx(
        binomtest(2, 200, alternative="less").proportion_ci(1 - m.ALPHA).high
    )
    with pytest.raises(ValueError, match="counts"):
        m.upper(2, 3)
    assert all(x["passed"] for x in m.boundaries())
    assert all(x["risk_upper"] == pytest.approx(x["scipy_upper"]) for x in m.boundaries())
    assert m.SEEDS == [11, 23, 37, 47, 59]


def test_freeze_draws_and_attacks():
    """REQ-REPORT-8842: construction rejection precedes independent index draws."""
    c = m.freeze(set(), fit_count=3, population_count=12)
    again = m.freeze(set(), fit_count=3, population_count=12)
    assert canonical_hash(c) == canonical_hash(again)
    assert len(c["canaries"]) == 12
    assert len({w["structure"] for w in c["worlds"]}) == 27
    assert not any("truth" in u for u in c["inputs"])
    assert {u["condition"] for u in c["canaries"]} == {
        "sign",
        "entity_binding",
        "missing_evidence",
        "valid_alternative",
    }
    assert sum(u["sign"] for u in c["canaries"]) == 6
    assert all(
        m.graph.truth(u["input"]) == "support"
        for u in c["canaries"]
        if u["condition"] == "valid_alternative"
    )
    extra = m.freeze({c["worlds"][0]["structure"]}, fit_count=3, population_count=60)
    assert not ({w["structure"] for w in extra["worlds"]} & {c["worlds"][0]["structure"]})
    with pytest.raises(ValueError, match="pilot"):
        m.freeze(set(), fit_count=3, population_count=12, pilot_s=0)
    d = m.draws(12, replicates=2)
    m.audit_draws(d, 12)
    assert len(d["replicates"][0]["calibration"]["indices"]) == 360
    assert len(set(d["replicates"][0]["calibration"]["indices"])) < 360
    for attack in ["duplicate", "alias", "state"]:
        bad = deepcopy(d)
        rep = bad["replicates"][0]
        if attack == "duplicate":
            rep["calibration"]["indices"] = [0] * 360
        elif attack == "alias":
            rep["evaluation"] = rep["calibration"]
        else:
            rep["calibration"]["final_state"] = {}
        with pytest.raises(ValueError, match="draw"):
            m.audit_draws(bad, 12)


def test_training_prediction_and_reduction():
    """SCENARIO-REPORT-8842-CONTRACT: only fit labels reach unchanged training."""
    c = m.freeze(set(), fit_count=3, population_count=12)
    fitted = m.fit(c, seeds=[11], steps=1)
    assert len(fitted["parameters"]) == 1
    predictions = m.predict(c, fitted)
    labels = m.labels(c["inputs"])
    assert m.labels(c["inputs"] * 4) == labels
    m.audit_predictions(predictions, c, labels, fitted)
    released = list(labels.values())
    bad = dict(zip(labels, released[1:] + released[:1], strict=True))
    with pytest.raises(ValueError, match="label"):
        m.audit_predictions(predictions, c, bad, fitted)
    with pytest.raises(ValueError, match="prediction"):
        m.audit_predictions(predictions + predictions[:1], c, labels, fitted)
    d = m.draws(12, replicates=2)
    result = m.reduce(predictions, labels, d)
    with pytest.raises(ValueError, match="roster"):
        m.reduce(predictions[:-1], labels, d)
    assert len(result["rows"]) == 2 * 3 * 3
    assert all(row["threshold"] is None for row in result["rows"])
    artificial = [dict(p, probabilities=[1.0, 0.0, 0.0], candidate="support") for p in predictions]
    correct = {p["unit_id"]: p["candidate"] for p in artificial}
    selected = m.reduce(artificial, correct, d)
    assert any(row["threshold"] == 0.5 for row in selected["rows"])
    assert all(row["census_risk"] == 0 for row in selected["rows"] if row["threshold"])
    assert all(row["denominator"] == 240 for row in result["rows"])


def test_private_cli_and_invalid_date(tmp_path):
    """SCENARIO-REPORT-8842-CONTRACT: actual child runs outside the checkout."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    output = tmp_path / (m.NAME + ".json")
    for options, expected in [
        (["--date", "20261104"], 2),
        (["--private-output", str(output), "--upstream", str(tmp_path / "absent")], 0),
    ]:
        child = subprocess.run(
            command + options, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )
        assert child.returncode == expected, child.stdout + child.stderr
    value = json.loads(output.read_text())
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["selective_protocol_ready_score"] == 0
    assert value["model_invocation_counts"]["model_loads_attempted"] == 0
    child = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value["completed_count"] = 1
    atomic_json(output, value)
    child = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 1


def test_authentication(tmp_path):
    """REQ-REPORT-8842: missing and rejected authorities cannot grant admission."""
    raw = tmp_path / "raw"
    missing = r.preconditions(raw, tmp_path / "missing")
    assert not missing["external_ready"]
    assert missing["diagnostics"][0]["check"] == "missing_file"
    path = tmp_path / "bad.json"
    for value in [
        {},
        {"selective_protocol_ready_score": 0},
        {"selective_protocol_ready_score": 1, "task_id": "foreign"},
    ]:
        atomic_json(path, value)
        assert not r.preconditions(raw, path)["external_ready"]


def test_bound_authorities(tmp_path, monkeypatch):
    """REQ-REPORT-8842: authenticate binary operands and reject drift at each boundary."""
    (tmp_path / "results").mkdir()
    (tmp_path / "research-roadmap.yaml").write_text(
        "milestone: 2026.11.757\nmilestone_doc: design.md\ntasks:\n- id: exp8842-selective-population\n"
    )
    (tmp_path / "design.md").write_text("Frozen private authority fixture")
    historical = tmp_path / "results/experiment_8832_v756_selective_availability.json"
    operand = tmp_path / "operand.bin"
    operand.write_bytes(b"\x8a retained binary coverage")
    old = dict(
        verdict_class="disqualified",
        inference_substrate_class="small_ebm_training",
        raw_shard_hashes=[r.reference(operand)],
    )
    atomic_json(historical, old)
    upstream = tmp_path / "upstream.json"
    value = dict(
        task_id="exp8829-selective-methods",
        verdict_class="null",
        selective_protocol_ready_score=1,
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
    )
    atomic_json(upstream, value)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(r, "PIN", r.sha256_file(upstream))
    monkeypatch.setattr(r, "urlopen", lambda *a, **kw: BytesIO(b"Certified by Abstention"))
    assert r.preconditions(tmp_path / "raw", upstream)["external_ready"]
    monkeypatch.setattr(r, "PIN", "bad")
    assert not r.preconditions(tmp_path / "raw", upstream)["external_ready"]
    monkeypatch.setattr(r, "PIN", r.sha256_file(upstream))
    value["raw_shard_hashes"] = [dict(path=str(operand), sha256="bad")]
    atomic_json(upstream, value)
    monkeypatch.setattr(r, "PIN", r.sha256_file(upstream))
    assert not r.preconditions(tmp_path / "raw", upstream)["external_ready"]
    value["raw_shard_hashes"] = []
    atomic_json(upstream, value)
    monkeypatch.setattr(r, "PIN", r.sha256_file(upstream))
    old["verdict_class"] = "null"
    atomic_json(historical, old)
    assert not r.preconditions(tmp_path / "raw", upstream)["external_ready"]
    old["verdict_class"] = "disqualified"
    atomic_json(historical, old)
    operand.write_bytes(b"changed")
    assert not r.preconditions(tmp_path / "raw", upstream)["external_ready"]
    old["raw_shard_hashes"] = []
    atomic_json(historical, old)
    monkeypatch.setattr(r, "urlopen", lambda *a, **kw: BytesIO(b"wrong paper"))
    assert not r.preconditions(tmp_path / "raw", upstream)["external_ready"]
    (tmp_path / "research-roadmap.yaml").write_text("milestone: old\ntasks: []\n")
    assert not r.preconditions(tmp_path / "raw", upstream)["external_ready"]


def test_measure_publish_and_rehashed_attacks(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8842-CONTRACT: real private operands support semantic cold replay."""
    cohort = m.freeze(set(), fit_count=3, population_count=12)
    real_fit, real_draws = m.fit, m.draws
    monkeypatch.setattr(m, "freeze", lambda *a, **kw: deepcopy(cohort))
    monkeypatch.setattr(m, "fit", lambda c: real_fit(c, seeds=[11], steps=1))
    monkeypatch.setattr(m, "draws", lambda n, **kw: real_draws(n, replicates=2))
    monkeypatch.setattr(r.inherited, "exposed_structures", lambda refs: set())
    pre = dict(
        external_ready=True,
        resources=dict(passed=True, measurement_deadline_s=60),
        references=[],
        diagnostics=[],
    )
    monkeypatch.setattr(r, "preconditions", lambda *a: deepcopy(pre))
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--private-output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert r.replay(output)["passed"]
    assert r.main(["--cold-replay", str(output)]) == 0
    corrupt = deepcopy(value)
    corrupt["reproducibility_checksum"] = "bad"
    atomic_json(output, corrupt)
    with pytest.raises(ValueError, match="checksum"):
        r.replay(output)
    assert r.main(["--cold-replay", str(output)]) == 1
    atomic_json(output, value)
    canary_path = Path(value["canary_manifest"]["path"])
    original_canary = canary_path.read_bytes()
    canary = json.loads(original_canary)
    canary["rows"][0]["source"]["sha256"] = "bad"
    atomic_json(canary_path, canary)
    bad = deepcopy(value)
    bad["canary_manifest"] = r.reference(canary_path)
    bad["raw_shard_hashes"] = [
        r.reference(canary_path) if ref["path"] == str(canary_path) else ref
        for ref in bad["raw_shard_hashes"]
    ]
    del bad["reproducibility_checksum"]
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(output, bad)
    with pytest.raises(ValueError, match="canary_source_hash"):
        r.replay(output)
    canary_path.write_bytes(original_canary)
    for key, changed, message in [
        ("experiment_id", 9999, "identity"),
        ("completed_count", 99, "count"),
        ("rows", [], "count"),
        ("certificate_firing_count", 999, "reduction"),
        ("current_work_receipt", {}, "receipt"),
        ("code_config_hashes", [dict(path=str(output), sha256="bad")], "operand"),
    ]:
        bad = deepcopy(value)
        bad[key] = changed
        del bad["reproducibility_checksum"]
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        with pytest.raises(ValueError, match=message):
            r.replay(output)
    atomic_json(output, value)
    raw = tmp_path / "timeout"
    raw.mkdir()
    pre["resources"]["measurement_deadline_s"] = -1
    with pytest.raises(ValueError, match="deadline"):
        r.measure(raw, pre)
    pre["resources"]["measurement_deadline_s"] = 60
    good = dict(
        coverage=dict(complete=True), union=dict(passed=True), receipts=[], raw_references=[]
    )
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(good))
    monkeypatch.setattr(r, "stream", lambda *a, **kw: dict(passed=False))
    assert r.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["required_checks_passed"] is False
    monkeypatch.setattr(r, "measure", lambda *a: (_ for _ in ()).throw(ValueError("owned_failure")))
    assert r.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["owned_errors"] == ["owned_failure"]
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])


def test_validation_adapter(tmp_path, monkeypatch):
    """REQ-REPORT-8842: private validation retains the unchanged consumer workflow."""
    checker = SimpleNamespace(validate=lambda raw, private: dict(passed=True))
    monkeypatch.setattr(r, "namespace", lambda *a: checker)
    assert r.validate(tmp_path, tmp_path)["passed"]
    assert checker.m is m
