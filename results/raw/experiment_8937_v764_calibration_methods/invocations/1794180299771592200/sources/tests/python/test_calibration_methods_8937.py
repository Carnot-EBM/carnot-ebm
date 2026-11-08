"""REQ-REPORT-8937; SCENARIO-REPORT-8937-CONTRACT: independent calibration methods."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.autoresearch import graph_calibration_benchmark as b
from carnot.reporting import calibration_methods_8937 as m


def test_feature_boundary_and_missing_values():
    """REQ-REPORT-8937: rehashing leaked truth cannot turn it into a detector input."""
    valid = dict(mean_token_nll=0.2, first_answer_top2_margin=0.8)
    assert b.features([valid]).shape == (1, 2)
    for field in ["label", "distance", "path", "graph", "split", "node_identity"]:
        injected = dict(valid, **{field: 1})
        envelope = dict(features=[injected], sha256=m.canonical_hash(injected))
        with pytest.raises(ValueError, match="feature_schema"):
            b.features(envelope["features"])
    for rows in [
        [],
        [{}],
        [dict(valid, mean_token_nll=None)],
        [dict(valid, mean_token_nll=True)],
        [dict(valid, mean_token_nll=-1)],
        [dict(valid, first_answer_top2_margin=float("nan"))],
    ]:
        with pytest.raises(ValueError):
            b.features(rows)
    assert b.nuisance([dict(token_count=2, format_valid=True)]).tolist() == [[2, 1]]
    for row in [
        dict(token_count=0, format_valid=True),
        dict(token_count=17, format_valid=True),
        dict(token_count=1, format_valid="correct"),
        dict(token_count=1, format_valid=True, parser_category="correct"),
    ]:
        with pytest.raises(ValueError):
            b.nuisance([row])
    for labels in [[0], [True, False], [0, 2], [0, float("nan")]]:
        with pytest.raises(ValueError):
            b.labels(labels, 2)


@pytest.fixture(scope="module")
def development():
    """REQ-REPORT-8937: development fixtures carry no live quality evidence."""
    return m.development()


def test_population_loss_training_and_frozen_states(development):
    """REQ-REPORT-8937: preserve the prior, train scaling and every initial parameter."""
    data, fit = development["fixture"], development["fit"]
    train = data["train"]
    assert fit == b.fit(
        train["features"],
        train["labels"],
        train["nuisance"],
        data["calibration"]["features"],
        data["calibration"]["labels"],
    )
    assert fit["temperature"] in [0.5, 1, 2, 4]
    assert fit["prevalence"] == np.mean(train["labels"])
    assert fit["scaling"]["mean"] == np.mean(b.features(train["features"]), axis=0).tolist()
    assert fit["initial_states"]["logistic"] == [0, 0, 0]
    assert fit["initial_states"]["nuisance"] == [0, 0, 0]
    assert np.asarray(fit["initial_states"]["energy"][0]).shape == (4, 2)
    assert fit["numerical_training_calls"] == 600
    logits = np.array([-2.0, 0.3, 1.0])
    y = np.array([0, 1, 1])
    expected = np.mean(np.logaddexp(0, logits) - y * logits)
    assert float(b.population_loss(logits, y)) == pytest.approx(expected)
    with pytest.raises(ValueError, match="degenerate_class"):
        b.fit(
            train["features"],
            [1] * len(train["labels"]),
            train["nuisance"],
            data["calibration"]["features"],
            data["calibration"]["labels"],
        )
    assert b.temperature([0, 0], [0, 1]) == 0.5
    assert all(r["passed"] for r in development["control_rows"])
    constant = next(r for r in development["control_rows"] if r["control"] == "constant_score")
    assert constant["score_range"] == 0 and constant["brier"] >= constant["prevalence_brier"]
    assert {r["control"] for r in development["control_rows"]} == {
        "planted_signal",
        "shuffled_labels",
        "constant_score",
        "missing_feature",
        "degenerate_class",
        "rehashed_leakage",
    }


def test_decisions_confidence_ties_and_cluster_bounds(development):
    """SCENARIO-REPORT-8937-CONTRACT: exact coverage and simultaneous paired bounds."""
    assert b.decisions([0, 0.1, 0.10001, 0.89999, 0.9, 1]) == [
        "accept",
        "accept",
        "escalate",
        "escalate",
        "reject",
        "reject",
    ]
    ids = ["d", "a", "c", "b"]
    chosen = b.half_selection([0.5] * 4, ids)
    assert sum(chosen) == 2
    shuffled = [3, 1, 0, 2]
    again = b.half_selection([0.5] * 4, [ids[i] for i in shuffled])
    assert {u for u, yes in zip(ids, chosen) if yes} == {
        ids[i] for i, yes in zip(shuffled, again) if yes
    }
    for p, names in [
        ([0.2], ["a"]),
        ([0.1, 0.2], ["a"]),
        ([float("nan"), 0.1], ["a", "b"]),
        ([-1, 0.2], ["a", "b"]),
    ]:
        with pytest.raises(ValueError):
            b.half_selection(p, names)
    rows = development["rows"]
    result = b.summarize(rows)
    assert result == development["fixture_diagnostics"]
    assert result["draws"] == 10000 and result["alpha_per_contrast"] == 0.05 / 3
    assert result["independent_clusters"] == 40
    assert set(result["contrasts"]) == {
        "brier_vs_prevalence",
        "brier_vs_logistic",
        "accepted_error_vs_logistic",
    }
    doubled = [dict(r, unit_id=r["unit_id"] + suffix) for suffix in ("-1", "-2") for r in rows]
    assert b.summarize(doubled)["independent_clusters"] == 40
    with pytest.raises(ValueError, match="comparison_roster"):
        b.summarize(rows[:-1])
    bad = deepcopy(rows)
    bad[0]["value"] += 0.1
    with pytest.raises(ValueError, match="row_metric"):
        b.summarize(bad)
    altered = deepcopy(development["fit"])
    altered["energy"][0] = [[float("nan"), 1]] * 4
    with pytest.raises(ValueError, match="state"):
        b.score(
            altered,
            development["fixture"]["diagnostic"]["features"],
            development["fixture"]["diagnostic"]["nuisance"],
        )
    with pytest.raises(ValueError, match="scoring_roster"):
        b.score(
            development["fit"],
            development["fixture"]["diagnostic"]["features"],
            development["fixture"]["diagnostic"]["nuisance"][:1],
        )
    with pytest.raises(ValueError, match="comparison_roster"):
        b.summarize(rows + [rows[0]])
    bad = deepcopy(rows)
    for r in bad:
        if r["unit_id"] == "fixture-00":
            r["cluster_id"] = "fixture-01"
    with pytest.raises(ValueError, match="equal_even_graph_roster"):
        b.summarize(bad)


def test_scoring_child_imports_no_oracle(tmp_path, development):
    """REQ-REPORT-8937: the actual isolated scoring child receives two features only."""
    path = tmp_path / "score.json"
    data = development["fixture"]["diagnostic"]
    m.atomic_json(
        path, dict(state=development["fit"], features=data["features"], nuisance=data["nuisance"])
    )
    code = """import json,sys
from carnot.autoresearch import graph_calibration_benchmark as b
v=json.load(open(sys.argv[1]))
p=b.score(v['state'],v['features'],v['nuisance'])
assert not any('graph_source' in n or 'bfs_' in n or 'pcib' in n or 'calibrated_decision_benchmark' in n for n in sys.modules)
print(json.dumps(p))
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    child = subprocess.run(
        [sys.executable, "-u", "-c", code, str(path)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert set(json.loads(child.stdout)) == {"energy", "logistic", "prevalence", "nuisance"}
    for field in ("label", "distance", "path"):
        leaked = deepcopy(data["features"])
        leaked[0][field] = 1
        m.atomic_json(
            path,
            dict(
                state=development["fit"],
                features=leaked,
                nuisance=data["nuisance"],
                hash=m.canonical_hash(leaked),
            ),
        )
        rejected = subprocess.run(
            [sys.executable, "-u", "-c", code, str(path)],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert rejected.returncode != 0 and "feature_schema" in rejected.stderr


def test_authentication_failures_and_changed_producers(tmp_path, monkeypatch):
    """REQ-REPORT-8937: external absence is terminal, while observed zeros stay distinct."""
    assert m.authenticate(m.ROOT)["ready"]
    root = tmp_path / "inputs"
    root.mkdir()
    monkeypatch.setattr(m.branch, "authority", lambda _: dict(ready=True, failures=[]))
    absent = m.authenticate(root)
    assert absent["failures"][0]["check"] == "missing_file"
    target = root / m.INPUT
    target.parent.mkdir(parents=True)
    original = json.loads((m.ROOT / m.INPUT).read_text())
    for mutate, check in [
        (lambda v: v.pop("source_methods_ready_score"), "missing_field"),
        (lambda v: v.update(source_methods_ready_score=0), "observed_zero"),
        (lambda v: v.update(source_methods_ready_score=True), "rejected_producer"),
    ]:
        value = deepcopy(original)
        mutate(value)
        m.atomic_json(target, value)
        assert any(r["check"] == check for r in m.authenticate(root)["failures"])
    target.write_text("[]")
    assert not m.authenticate(root)["ready"]
    target.write_text("{broken")
    assert not m.authenticate(root)["ready"]
    for existing, expected_check in [(False, "missing_file"), (True, "rejected_producer")]:
        value = deepcopy(original)
        operand = tmp_path / "original-operand.json"
        if existing:
            operand.write_text("changed")
        value["source_artifact_hashes"][0]["path"] = str(operand)
        m.atomic_json(target, value)
        assert any(
            r["check"] == expected_check and r["upstream"] == "exp8931-original-operand"
            for r in m.authenticate(root)["failures"]
        )


def test_reduction_readiness_and_semantic_tampering(tmp_path, development, monkeypatch):
    """SCENARIO-REPORT-8937-CONTRACT: fixtures never authorize scientific positives."""
    work = m.freeze(m.ROOT, tmp_path / "raw")
    validation = dict(receipts=[], coverage={})
    value = m.build(work, development, validation)
    assert value["verdict_class"] == "disqualified"
    assert value["calibration_methods_ready_score"] == 0
    assert not m.qualified(dict(receipts=[], coverage={"invalid": 1}))
    monkeypatch.setattr(m, "qualified", lambda _: True)
    value = m.build(work, development, validation)
    assert value["calibration_methods_ready_score"] == 1 and value["verdict_class"] == "null"
    assert value["independent_count"] == 0 and value["verifier_is_oracle"] is False
    assert value["MODEL_SPECS"] == [] and value["numerical_training_calls"] == 1200
    assert value["intended_count"] == value["completed_count"] == len(value["rows"])
    assert value["failed_count"] == value["censored_count"] == value["excluded_count"] == 0
    assert value["inference_substrate_class"] == "no_model_load"
    m.atomic_json(tmp_path / "candidate.json", value)
    assert m.replay(tmp_path / "candidate.json")["passed"]
    value["rows"][0]["value"] += 0.01
    value["reproducibility_checksum"] = m.checksum(value)
    m.atomic_json(tmp_path / "candidate.json", value)
    assert not m.replay(tmp_path / "candidate.json")["passed"]
    changed = m.build(work, development, validation)
    changed["calibration_methods_ready_score"] = 0
    m.atomic_json(tmp_path / "candidate.json", changed)
    assert not m.replay(tmp_path / "candidate.json")["passed"]
    m.atomic_json(tmp_path / "candidate.json", m.build(work, development, validation))
    Path(work["sources"][0]["path"]).write_text("changed retained source")
    assert not m.replay(tmp_path / "candidate.json")["passed"]
    work["authentication"]["failures"] = [dict(check="missing_file")]
    blocked = m.build(work, {}, validation)
    assert blocked["verdict_class"] == "blocked" and blocked["gate_check_summary"]
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["calibration_methods_ready_score"] == 0
    assert not m.replay(tmp_path / "absent.json")["passed"]


def test_failed_owned_controls_disqualify(tmp_path, development, monkeypatch):
    """REQ-REPORT-8937: a broken rejection path cannot qualify on a planted fixture."""
    original = b.features

    def permissive(rows):
        return original([{k: r.get(k, 0.0) for k in b.FEATURES} for r in rows])

    monkeypatch.setattr(b, "features", permissive)
    monkeypatch.setattr(b, "fit", lambda *_: deepcopy(development["fit"]))
    result = m.development()
    assert not all(r["passed"] for r in result["control_rows"])
    work = m.freeze(m.ROOT, tmp_path / "raw")
    monkeypatch.setattr(m, "qualified", lambda _: True)
    value = m.build(work, result, dict(receipts=[], coverage={}))
    assert (
        value["calibration_methods_ready_score"] == 0 and value["verdict_class"] == "disqualified"
    )


def test_actual_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8937-CONTRACT: cold replay is outside checkout without PYTHONPATH."""
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]

    def child(*args):
        return subprocess.run(
            command + list(args),
            cwd=tmp_path,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )

    got = child("--date", "20261108", "--private-output", str(output))
    assert got.returncode == 0, got.stdout
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified" and not value["required_checks_passed"]
    assert child("--cold-replay", str(output)).returncode == 0
    value["calibration_methods_ready_score"] = 1
    value["reproducibility_checksum"] = m.checksum(value)
    m.atomic_json(output, value)
    assert child("--cold-replay", str(output)).returncode == 1
    assert child("--date", "20261107").returncode == 2
    assert (
        child("--date", "20261108", "--private-output", str(m.ROOT / "forbidden.json")).returncode
        == 2
    )


def test_production_qualification_and_terminal_rejection(tmp_path, development, monkeypatch):
    """REQ-REPORT-8937: measured checks govern publication and failed scanners disqualify."""
    root, original_freeze = m.ROOT, m.freeze
    commands = m.validation_plan(tmp_path / "plan")
    assert {c.name for c in commands} == m.CHECKS
    config = (tmp_path / "plan" / "coverage.ini").read_text()
    assert all(p in config for p in m.OWNED) and "exclude_lines =\n" in config
    receipts = [dict(name=c.name, passed=True, exit_code=0, timed_out=False) for c in commands]
    monkeypatch.setattr(m, "verify_measured", lambda *_: True)
    assert m.qualified(dict(receipts=receipts, coverage={}))
    assert not m.qualified(dict(receipts=receipts[:-1], coverage={}))
    receipts[0]["exit_code"] = 1
    assert not m.qualified(dict(receipts=receipts, coverage={}))
    receipts[0]["exit_code"] = 0

    def frozen(_, raw):
        with monkeypatch.context() as context:
            context.setattr(m, "ROOT", root)
            return original_freeze(root, raw)

    def run(_, specs, **kwargs):
        specs = list(specs)
        if specs[0].name == "owned_unit_child":
            coverage = next(
                a.split("=", 1)[1] for a in specs[0].argv if a.startswith("--data-file=")
            )
            Path(coverage).touch()
        return [
            dict(
                name=c.name,
                passed=c.name != "adversarial",
                exit_code=int(c.name == "adversarial"),
                timed_out=False,
            )
            for c in specs
        ]

    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "freeze", frozen)
    monkeypatch.setattr(m, "development", lambda: deepcopy(development))
    monkeypatch.setattr(m, "run_commands", run)
    monkeypatch.setattr(m, "measured", lambda *_: dict(complete=True))
    assert m.main(["--date", "20261108"]) == 0
    output = tmp_path / "results" / (m.NAME + ".json")
    value = json.loads(output.read_text())
    assert (
        value["verdict_class"] == "disqualified" and value["calibration_methods_ready_score"] == 0
    )
    assert m.replay(output)["passed"]
    monkeypatch.setattr(m, "publish_primary", lambda *_: (_ for _ in ()).throw(ValueError("other")))
    with pytest.raises(ValueError, match="other"):
        m.main(["--date", "20261108"])
    with pytest.raises(SystemExit):
        m.main([])
