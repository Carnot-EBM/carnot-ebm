"""REQ-REPORT-8855: exact diagnostics must retain the frozen population and costs."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from scipy.stats import beta, binom

from carnot.reporting import selective_ceiling_8855 as m
from carnot.reporting import selective_ceiling_runner_8855 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_exact_boundaries_and_power():
    """SCENARIO-REPORT-8855-CONTRACT: exact tails include empty and error boundaries."""
    for name, budget, alpha, selection in m.PROTOCOLS:
        for errors in [0, 1, 2, 10]:
            n = m.minimum_accepted(errors, alpha)
            assert binom.cdf(errors, n, 0.05) <= alpha
            assert binom.cdf(errors, n - 1, 0.05) > alpha
            assert beta.isf(alpha, errors + 1, n - errors) <= 0.05 + 1e-14
        assert budget + selection in [360, 720]
        assert m.firing_probability(budget, 0, None, alpha) == 0
        assert m.firing_probability(budget, 1, 0, alpha) == 1
        assert m.firing_probability(budget, 1, 1, alpha) == 0
        q, risk = 0.7, 0.01
        direct = sum(
            float(binom.pmf(n, budget, q))
            * sum(
                float(binom.pmf(e, n, risk))
                for e in range(int(n * 0.05) + 1)
                if n and binom.cdf(e, n, 0.05) <= alpha
            )
            for n in range(budget + 1)
        )
        assert m.firing_probability(budget, q, risk, alpha) == pytest.approx(direct, abs=1e-12)
    with pytest.raises(ValueError, match="counts"):
        m.minimum_accepted(-1, 0.01)


def score_fixture():
    """A complete small roster exposes refusal, safe rules and discriminative errors."""
    return [
        dict(
            unit_id=f"u{i}",
            cluster_id=f"w{i}",
            arm=a,
            seed=s,
            probabilities=([1.0, 0.0, 0.0] if i < 8 else [0.0, 0.0, 1.0]),
            expected_action="accept",
            status="completed",
            missing_reason=None,
        )
        for a, s in m.old.POLICIES
        for i in range(12)
    ]


def test_census_and_decomposition():
    """REQ-REPORT-8855: a census diagnoses discrimination without selecting deployment."""
    scores = score_fixture()
    result = m.diagnose(scores)
    assert len(result["population_cutoff_rows"]) == 42
    assert result["split_study_feasible_score"] == 1
    assert all(x["risk"] == 0 for x in result["population_cutoff_rows"])
    assert all(
        x["total_labels"] == 720
        for x in result["availability_power_rows"]
        if x["protocol"] != "historical"
    )
    assert all(x["numerical_check_passed"] for x in result["availability_power_rows"])
    assert len(result["failure_decomposition"]) == 7
    for row in scores:
        row["probabilities"] = [0.0, 0.0, 1.0]
    empty = m.diagnose(scores)
    assert all(x["risk"] is None for x in empty["population_cutoff_rows"])
    assert empty["split_study_feasible_score"] == 0
    for row in scores:
        row["probabilities"] = [1.0, 0.0, 0.0]
        row["expected_action"] = "reject"
    assert all(
        x["binding_limitation"] == "discrimination"
        for x in m.diagnose(scores)["failure_decomposition"]
    )
    with pytest.raises(ValueError, match="roster"):
        m.diagnose(scores[:-1])
    with pytest.raises(ValueError, match="roster"):
        m.diagnose([])
    scores[0]["probabilities"] = [float("nan"), 0.0, 0.0]
    with pytest.raises(ValueError, match="score"):
        m.diagnose(scores)


def test_private_cli_missing_and_replay(tmp_path):
    """SCENARIO-REPORT-8855-CONTRACT: outside-checkout replay rejects rehashed edits."""
    output = tmp_path / (m.NAME + ".json")
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for argv, code in [
        (["--date", "20261105"], 2),
        (["--private-output", str(output), "--upstream", str(tmp_path / "absent")], 0),
    ]:
        child = subprocess.run(
            command + argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )
        assert child.returncode == code, child.stdout + child.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_missing_file")
    assert len(value["rows"]) == 42
    assert value["inference_substrate_class"] == "blocked_no_run"
    for tamper, expected in [(False, 0), (True, 1)]:
        if tamper:
            value["ceiling_rows_ready_score"] = 1
            del value["reproducibility_checksum"]
            value["reproducibility_checksum"] = canonical_hash(value)
            atomic_json(output, value)
        child = subprocess.run(
            command + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert child.returncode == expected, child.stdout + child.stderr


@pytest.fixture
def historical(tmp_path, monkeypatch):
    """Keep real historical reducers with tiny worlds and the genuine random streams."""
    monkeypatch.setattr(r, "SIZE", 12)
    monkeypatch.setattr(r, "REPLICATES", 2)
    cohort = m.old.p.freeze(set(), fit_count=3, population_count=12)
    units = sorted(
        (u for u in cohort["inputs"] if u["split"] == "population"), key=lambda u: u["unit_id"]
    )
    truth = m.old.p.labels(units)
    scores = [
        dict(
            unit_id=u["unit_id"],
            cluster_id=u["cluster_id"],
            arm=a,
            seed=s,
            candidate=u["candidates"][u["candidate_index"]],
            probabilities=[1.0, 0.0, 0.0],
            status="completed",
            missing_reason=None,
        )
        for a, s in m.old.POLICIES
        for u in units
    ]
    trace = m.old.p.draws(12, replicates=2)
    values = dict(
        scores=dict(rows=scores),
        labels=dict(labels=truth),
        population=cohort,
        draws=trace,
        protocol=dict(
            alpha=m.old.p.ALPHA, thresholds=m.old.p.THRESHOLDS, seeds=m.old.p.SEEDS, population=12
        ),
    )
    refs = {}
    for key, value in values.items():
        path = tmp_path / (key + ".json")
        atomic_json(path, value)
        refs[key] = r.reference(path)
    calculated = m.old.evaluate(scores, truth, trace, m.old.select(scores, truth, trace))
    source = dict(
        calculated,
        task_id="exp8844-selective-availability",
        selective_rows_ready_score=1,
        frozen_protocol=refs["protocol"],
        acceptance_gates=dict(risk_upper=0.05, coverage=0.30),
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[refs["scores"], refs["protocol"], refs["labels"]],
    )
    population = dict(
        task_id="exp8842-selective-population",
        selective_protocol_ready_score=1,
        population_manifest=refs["population"],
        labels_manifest=refs["labels"],
        draw_manifest=refs["draws"],
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[refs["population"], refs["draws"]],
    )
    sources = [tmp_path / (n + ".json") for n in ["availability", "population_source"]]
    for path, value in zip(sources, [source, population], strict=True):
        atomic_json(path, value)
    monkeypatch.setattr(r, "PINS", [r.sha256_file(p) for p in sources])
    monkeypatch.setattr(r, "PROTOCOL_PIN", refs["protocol"]["sha256"])
    return sources, source, population


def test_authentication_and_measurement(tmp_path, monkeypatch, historical):
    """REQ-REPORT-8855: missing, zero and changed historical operands remain explicit."""
    sources, source, population = historical
    raw = tmp_path / "raw"
    assert r.preconditions(raw, *sources)["external_ready"]
    result = r.measure(raw)
    assert len(result["rows"]) == 42
    assert r.measure(raw, write=False) == result
    score_path = raw / "scores.json"
    saved = score_path.read_bytes()
    atomic_json(score_path, dict(value=[]))
    with pytest.raises(ValueError, match="retained_operand"):
        r.measure(raw, write=False)
    score_path.write_bytes(saved)
    monkeypatch.setattr(r, "PROTOCOL_PIN", "wrong")
    with pytest.raises(ValueError, match="protocol"):
        r.measure(raw)
    for changed, expected in [
        ({}, "missing_field"),
        (dict(selective_rows_ready_score=0), "observed_zero"),
        (dict(source, task_id="foreign"), "rejected_producer"),
    ]:
        atomic_json(sources[0], changed)
        pre = r.preconditions(raw, *sources)
        assert pre["diagnostics"][0]["check"] == expected
    atomic_json(sources[0], source)
    path = Path(source["raw_shard_hashes"][0]["path"])
    path.write_text("changed")
    assert r.preconditions(raw, *sources)["diagnostics"][0]["check"] == "rejected_producer"


def test_publication_and_semantic_attacks(tmp_path, monkeypatch, historical):
    """SCENARIO-REPORT-8855-CONTRACT: rehashed reductions cannot become new evidence."""
    sources, _, _ = historical
    good = dict(
        coverage=dict(complete=True), union=dict(passed=True), receipts=[], raw_references=[]
    )
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(good))
    log = tmp_path / "log"
    log.write_text("owned test control")
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
    args = ["--output", str(output), "--upstream", str(sources[0]), "--population", str(sources[1])]
    assert r.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"]
    assert value["ceiling_rows_ready_score"] == 1
    assert r.main(["--cold-replay", str(output)]) == 0
    for key, changed, message in [
        ("experiment_id", 123, "identity"),
        ("completed_count", 0, "count"),
        ("split_study_feasible_score", 9, "reduction"),
        ("current_work_receipt", {}, "receipt"),
        ("source_artifact_hashes", [dict(path=str(log), sha256="wrong")], "operand"),
    ]:
        bad = deepcopy(value)
        bad[key] = changed
        del bad["reproducibility_checksum"]
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        with pytest.raises(ValueError, match=message):
            r.replay(output)
    atomic_json(output, dict(value, reproducibility_checksum="bad"))
    assert r.main(["--cold-replay", str(output)]) == 1
    atomic_json(output, value)
    with monkeypatch.context() as control:
        control.setattr(r, "PINS", ["wrong", "wrong"])
        with pytest.raises(ValueError, match="producer_pin"):
            r.replay(output)
    bad = deepcopy(value)
    bad["measurement_manifest"] = None
    del bad["reproducibility_checksum"]
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(output, bad)
    with pytest.raises(ValueError, match="missing_measurement"):
        r.replay(output)
    raw = Path(value["measurement_manifest"]["path"]).parent
    reduction = raw / "reduction.json"
    original = reduction.read_bytes()
    atomic_json(reduction, dict(changed=True))
    bad = deepcopy(value)
    bad["measurement_manifest"] = r.reference(reduction)
    bad["raw_shard_hashes"] = [
        r.reference(reduction) if ref["path"] == str(reduction) else ref
        for ref in bad["raw_shard_hashes"]
    ]
    del bad["reproducibility_checksum"]
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(output, bad)
    with pytest.raises(ValueError, match="retained_reduction"):
        r.replay(output)
    reduction.write_bytes(original)
    monkeypatch.setattr(
        r, "stream", lambda *a, **kw: dict(passed=False, stdout_path=str(log), stderr_path=str(log))
    )
    assert r.main(args) == 0
    assert json.loads(output.read_bytes())["ceiling_rows_ready_score"] == 0
    monkeypatch.setattr(
        r, "measure", lambda *a, **kw: (_ for _ in ()).throw(ValueError("owned_failure"))
    )
    assert r.main(args) == 0
    assert json.loads(output.read_bytes())["owned_errors"] == ["owned_failure"]
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])


def test_validation_adapter(tmp_path, monkeypatch):
    """REQ-REPORT-8855: validation retains the unchanged private consumer suite."""
    from types import SimpleNamespace

    checker = SimpleNamespace(validate=lambda *a: dict(passed=True))
    monkeypatch.setattr(r, "namespace", lambda *a: checker)
    assert r.validate(tmp_path, tmp_path)["passed"]
    assert checker.m is m


def test_owned_control_failures(tmp_path, monkeypatch, historical):
    """REQ-REPORT-8855: internal inconsistencies and numerical failures cannot pass."""
    sources, _, _ = historical
    raw = tmp_path / "raw"
    assert r.preconditions(raw, *sources)["external_ready"]
    result = r.measure(raw)
    source_path = raw / "source_0.json"
    original = source_path.read_bytes()
    source = json.loads(original)
    source["certificate_firing_count"] += 1
    atomic_json(source_path, source)
    with pytest.raises(ValueError, match="historical_reduction"):
        r.measure(raw)
    source_path.write_bytes(original)
    with monkeypatch.context() as control:
        control.setattr(r, "REPLICATES", 3)
        with pytest.raises(ValueError, match="population_or_draw"):
            r.measure(raw)
    with monkeypatch.context() as control:
        control.setattr(r, "MEASUREMENT_DEADLINE_S", -1)
        with pytest.raises(ValueError, match="measurement_deadline"):
            r.measure(raw)
    label_ref = next(
        ref for ref in source["raw_shard_hashes"] if ref["path"].endswith("/labels.json")
    )
    labels = json.loads(Path(label_ref["path"]).read_bytes())["labels"]
    with monkeypatch.context() as control:
        original_load = r.load
        responses = iter([dict(labels={}), dict(labels=labels)])
        control.setattr(
            r, "load", lambda ref: next(responses) if ref == label_ref else original_load(ref)
        )
        with pytest.raises(ValueError, match="historical_label"):
            r.measure(raw)
    scores_path = next(
        Path(ref["path"])
        for ref in source["raw_shard_hashes"]
        if ref["path"].endswith("/scores.json")
    )
    original_scores = scores_path.read_bytes()
    changed = json.loads(original_scores)
    first = changed["rows"][0]["unit_id"]
    for row in changed["rows"]:
        if row["unit_id"] == first:
            row["unit_id"] = "foreign_unit"
    atomic_json(scores_path, changed)
    altered = json.loads(original)
    altered["raw_shard_hashes"] = [
        r.reference(scores_path) if ref["path"] == str(scores_path) else ref
        for ref in altered["raw_shard_hashes"]
    ]
    atomic_json(source_path, altered)
    with pytest.raises(ValueError, match="population_unit"):
        r.measure(raw)
    scores_path.write_bytes(original_scores)
    source_path.write_bytes(original)
    with monkeypatch.context() as control:
        control.setattr(m, "firing_probability", lambda *a: 0.123)
        with pytest.raises(ValueError, match="numerical_control"):
            m.diagnose(score_fixture())
    assert result["historical_recomputation"]["selection_sha256"]
