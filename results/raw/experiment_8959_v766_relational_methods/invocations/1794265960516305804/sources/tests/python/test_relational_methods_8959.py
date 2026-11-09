"""REQ-REPORT-8959; SCENARIO-REPORT-8959-CONTRACT: freeze before fresh evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.autoresearch import relational_graph_encoder as g
from carnot.reporting import relational_methods_8959 as d
from carnot.reporting import relational_methods_runner_8959 as runner
from carnot.reporting.current_work_receipt import atomic_json


def fixture_rows():
    """REQ-REPORT-8959: adjacency supplies information absent from constant confidence."""
    return [
        dict(
            nodes=["AA", "BB", "CC", "DD"],
            edges=[["AA", "BB"], ["BB", "CC"], ["CC", "DD"]],
            query_node="AA",
            proposed_node="BB" if i % 2 else "DD",
            features=dict(mean_token_nll=1.0, first_answer_top2_margin=1.0),
        )
        for i in range(32)
    ]


@pytest.fixture(scope="module")
def states():
    """REQ-REPORT-8959: each matched architecture trains from identical seed identities."""
    rows = fixture_rows()
    return {
        arm: g.fit(rows, [i % 2 for i in range(32)], 71, relational, energy)
        for arm, relational, energy in d.TRAINED_ARMS
    }


def test_architecture_learning_and_invariance(states):
    """SCENARIO-REPORT-8959-CONTRACT: real gradients and symmetry are owned controls."""
    rows = fixture_rows()
    for arm, state in states.items():
        p = g.predict(state, rows)
        assert np.isfinite(p).all()
        assert state["steps"] == 200 and state["rate"] == 0.01
        if arm.startswith("relational"):
            assert state["gradient_norm"] > 0
            assert state["encoder_gradient_norm"] > 0
            assert np.mean((p - np.arange(32) % 2) ** 2) < 0.25
        renamed = deepcopy(rows)
        names = dict(zip(rows[0]["nodes"], ["WW", "XX", "YY", "ZZ"]))
        for row in renamed:
            row["nodes"] = [names[n] for n in row["nodes"]][::-1]
            row["edges"] = [[names[b], names[a]] for a, b in row["edges"]][::-1]
            row["query_node"] = names[row["query_node"]]
            row["proposed_node"] = names[row["proposed_node"]]
        assert np.max(np.abs(g.predict(state, renamed) - p)) <= 1e-6
    missing = deepcopy(rows[:1])
    missing[0]["proposed_node"] = None
    assert np.isfinite(g.predict(states["relational_energy"], missing)).all()


@pytest.mark.parametrize("key", ["label", "distance", "bfs", "answer_set", "graph_id", "split"])
def test_forbidden_detector_fields(key):
    """REQ-REPORT-8959: rehashed metadata cannot cross the exact feature boundary."""
    rows = fixture_rows()
    rows[0][key] = 1
    with pytest.raises(ValueError, match="detector_schema"):
        g.pack(rows)


def test_frozen_protocol():
    """REQ-REPORT-8959: all five comparisons and all fresh roles are fixed in advance."""
    protocol = d.protocol()
    assert len(protocol["detector_bounds"]) == 5
    assert protocol["bootstrap_draws"] == 10000
    assert protocol["family_alpha"] == 0.05
    assert protocol["temperature_grid"] == [0.5, 1.0, 2.0, 4.0]
    assert protocol["seeds"] == [71, 72, 73]


def test_external_absence_is_terminal(tmp_path):
    """SCENARIO-REPORT-8959-CONTRACT: missing evidence is blocked with exact custody fields."""
    evidence = d.authenticate(tmp_path)
    assert evidence["failures"]
    assert all(
        {"upstream", "path", "hash", "check", "artifact_field", "expected", "observed"} <= set(row)
        for row in evidence["failures"]
    )


@pytest.mark.parametrize(
    "fault", ["empty", "nodes", "duplicate", "source", "answer", "loop", "edge", "parallel"]
)
def test_input_validation(fault):
    """REQ-REPORT-8959: malformed adjacency and marker identities fail at the boundary."""
    rows = fixture_rows()[:1]
    if fault == "empty":
        rows = []
    elif fault == "nodes":
        rows[0]["nodes"] = []
    elif fault == "duplicate":
        rows[0]["nodes"].append("AA")
    elif fault == "source":
        rows[0]["query_node"] = "ZZ"
    elif fault == "answer":
        rows[0]["proposed_node"] = "ZZ"
    elif fault == "loop":
        rows[0]["edges"] = [["AA", "AA"]]
    elif fault == "edge":
        rows[0]["edges"] = [["AA", "ZZ"]]
    else:
        rows[0]["edges"] *= 2
    with pytest.raises(ValueError):
        g.pack(rows)


def test_calibration_is_grid_only(states):
    """REQ-REPORT-8959: shared temperatures use calibration operands and average seed probabilities."""
    rows = fixture_rows()
    group = [states["relational_energy"]] * 3
    assert np.allclose(g.ensemble(group, rows), g.predict(group[0], rows))
    assert g.select_temperature(group, rows, [i % 2 for i in range(32)]) in g.b.GRID
    with pytest.raises(ValueError, match="temperature_grid"):
        g.predict(group[0], rows, 0)
    with pytest.raises(ValueError, match="training_class_support"):
        g.fit(rows, [0] * 32, 71, True, True)


@pytest.fixture(scope="module")
def evidence():
    """REQ-REPORT-8959: original source and development bytes must remain authenticated."""
    value = d.authenticate(d.ROOT)
    assert value["failures"] == []
    return value


def test_authentication_and_fit_custody(evidence, monkeypatch):
    """REQ-REPORT-8959: no guard label or metadata enters any of the twelve fits."""
    calls = []

    def fit(rows, labels, seed, relational, energy):
        calls.append((rows, labels, seed, relational, energy))
        return dict(seed=seed)

    monkeypatch.setattr(g, "fit", fit)
    monkeypatch.setattr(g, "ensemble", lambda states, rows, temperature=1: np.full(len(rows), 0.5))
    result = d.train(evidence)
    fit_ids = {r["unit_id"] for r in evidence["developer"] if r["development_role"] == "fit"}
    assert set(result["fit_ids"]) == fit_ids
    assert not fit_ids & set(result["guard_ids"])
    assert len(calls) == 12 and {c[2] for c in calls} == {71, 72, 73}
    assert all(len(c[0]) == len(c[1]) == 256 and all(set(r) == g.KEYS for r in c[0]) for c in calls)
    changed = deepcopy(evidence)
    changed["developer"].pop()
    with pytest.raises(ValueError, match="development_roles"):
        d.train(changed)


def test_private_controls():
    """SCENARIO-REPORT-8959-CONTRACT: fixture learning and permutation sensitivity are real measurements."""
    controls = d.fixture_controls()
    assert controls["passed"] and len(controls["rows"]) == 11
    assert controls["numerical_training_calls"] == 800


@pytest.mark.parametrize("fault", ["hash", "field", "operand", "structure"])
def test_authentication_refuses_changes(evidence, tmp_path, monkeypatch, fault):
    """REQ-REPORT-8959: missing readiness, byte changes and malformed producers cannot qualify."""
    reference = d.calibration.reference
    if fault in {"hash", "field", "structure"}:
        for name, _, field in d.INPUTS:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(
                path, {"required_checks_passed": True, field: 0} if fault == "field" else {}
            )
        if fault != "hash":
            pins = {str(tmp_path / name): pin for name, pin, _ in d.INPUTS}
            monkeypatch.setattr(
                d.calibration, "reference", lambda p: dict(path=str(p), sha256=pins[str(p)])
            )
        result = d.authenticate(tmp_path)
    else:
        target = evidence["references"][1]["path"]
        monkeypatch.setattr(
            d.calibration,
            "reference",
            lambda p: dict(path=str(p), sha256="changed") if str(p) == target else reference(p),
        )
        result = d.authenticate(d.ROOT)
    assert result["failures"]


def test_graph_exhaustion_and_isomorphism(monkeypatch):
    """SCENARIO-REPORT-8959-CONTRACT: exhaustion never substitutes an easier population."""
    graph = dict(
        nodes=["AA", "BB", "CC", "DD", "EE"],
        edges=[["AA", "BB"], ["BB", "CC"], ["CC", "DD"], ["DD", "EE"]],
        start="AA",
    )
    empty = dict(old_graphs=[], old_seeds=[])
    monkeypatch.setattr(d.source, "candidate", lambda seed: deepcopy(graph))
    result = d.graph_protocol(empty)
    assert len(result["graph_manifest"]) == 1
    assert result["terminal_reason"] == "proposal_budget_exhausted"
    assert not result["complete"]
    assert d.verify_graphs(result, empty)
    renamed = dict(
        nodes=["VV", "WW", "XX", "YY", "ZZ"],
        edges=[["VV", "WW"], ["WW", "XX"], ["XX", "YY"], ["YY", "ZZ"]],
        start="VV",
    )
    rejected = d.graph_protocol(dict(old_graphs=[renamed], old_seeds=[]))
    assert not rejected["graph_manifest"]
    assert rejected["proposal_rejection_rows"][0]["reason"] == "isomorphic_relabeling"
    result["graph_manifest"][0]["role"] = "changed"
    assert not d.verify_graphs(result, empty)


def test_construction_time_limit(monkeypatch):
    """REQ-REPORT-8959: history reconstruction and collection share one600-second ceiling."""
    ticks = iter([0, 0, 601, 602])
    monkeypatch.setattr(d.time, "monotonic", lambda: next(ticks))
    result = d.graph_protocol(dict(old_graphs=[], old_seeds=[1]))
    assert result["terminal_reason"] == "construction_time_exhausted"


def test_private_cli_publication_cold_replay_and_tamper(tmp_path):
    """SCENARIO-REPORT-8959-CONTRACT: private children publish, replay and refuse rehashed changes."""
    output = tmp_path / (d.NAME + ".json")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    def run(*args):
        return subprocess.run(
            [sys.executable, "-u", str(d.ROOT / d.CLI), *args],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=240,
        )

    result = run("--date", "20261109", "--fixture-e2e", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["fixture_only"] and value["relational_methods_ready_score"] == 0
    assert value["completed_count"] == 1024
    roles = value["role_manifest"]
    for shard in ("A", "B"):
        assert {
            kind: sum(r["shard"] == shard and r["role"] == kind for r in roles)
            for kind in ("calibration", "test", "online")
        } == dict(calibration=64, test=64, online=384)
    streams = {r["stream_id"] for r in roles if r["stream_id"]}
    assert len(streams) == 24
    assert all(
        sorted(r["stream_position"] for r in roles if r["stream_id"] == s) == list(range(32))
        for s in streams
    )
    assert len(value["trained_state_hashes"]) == 12 and len(value["development_guard_rows"]) == 320
    assert value["numerical_training_calls"] == 3200
    replay = run("--cold-replay", str(output))
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["relational_methods_ready_score"] = 1
    value["reproducibility_checksum"] = d.calibration.checksum(value)
    atomic_json(output, value)
    assert run("--cold-replay", str(output)).returncode == 1
    assert run("--date", "wrong").returncode == 2
    assert (
        run("--date", "20261109", "--fixture-e2e", str(d.ROOT / "forbidden.json")).returncode == 2
    )


def test_freeze_missing_and_terminal_classes(tmp_path, monkeypatch):
    """REQ-REPORT-8959: missing external science blocks, while failed owned controls disqualify."""
    missing = d.authenticate(tmp_path)
    monkeypatch.setattr(d, "authenticate", lambda _: deepcopy(missing))
    work = runner.freeze(d.ROOT, tmp_path / "raw")
    validation = dict(receipts=[], coverage={})
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    monkeypatch.setattr(d.diagnosis.admission.shard_a, "qualified", lambda *_: True)
    value = d.build(work, validation, False)
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 1024
    work["evidence"]["failures"] = []
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    work["controls"]["passed"] = True
    assert d.build(work, validation, False)["verdict_class"] == "null"


@pytest.mark.parametrize("coverage", [False, True])
def test_nonfixture_pipeline_and_failed_publication(tmp_path, monkeypatch, coverage):
    """SCENARIO-REPORT-8959-CONTRACT: real publication requires owned receipts; health runs only once."""
    missing = d.authenticate(tmp_path)
    monkeypatch.setattr(d, "authenticate", lambda _: missing)
    work = runner.freeze(d.ROOT, tmp_path / "raw")
    monkeypatch.setattr(runner, "freeze", lambda *_: deepcopy(work))
    monkeypatch.setattr(d, "ROOT", tmp_path)
    private = tmp_path / "private"
    private.mkdir()
    if coverage:
        (private / ".coverage").touch()
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **_: str(private))
    monkeypatch.setattr(
        runner.shared,
        "validation_plan",
        lambda *a, **k: [runner.CommandSpec("fixture", ("true",), "owned")],
    )
    monkeypatch.setattr(runner, "run_commands", lambda *a, **k: [])
    monkeypatch.setattr(runner, "measured", lambda *a, **k: {})
    with pytest.raises(ValueError, match="owned_checks_failed"):
        runner.main(["--date", "20261109"])
    assert not (tmp_path / "results" / (d.NAME + ".json")).exists()
    monkeypatch.setattr(d.diagnosis.admission.shard_a, "qualified", lambda *_: True)
    monkeypatch.setattr(runner, "replay", lambda _: dict(passed=True))
    assert runner.main(["--date", "20261109"]) == 0
    value = json.loads((tmp_path / "results" / (d.NAME + ".json")).read_text())
    assert value["verdict_class"] == "blocked" and value["global_test_debt"]["repeated"] is False


@pytest.mark.parametrize(
    "fault", ["missing", "bytes", "executed", "frozen", "trained", "construction"]
)
def test_cold_replay_refuses_rehashed_operands(tmp_path, monkeypatch, fault):
    """SCENARIO-REPORT-8959-CONTRACT: freshly hashed lies fail primitive recomputation."""
    path = tmp_path / "primary.json"
    if fault == "missing":
        assert not runner.replay(path)["passed"]
        return
    raw = tmp_path / "raw"
    raw.mkdir()
    measurement = raw / "measurement.json"
    atomic_json(measurement, {})
    evidence = dict(failures=[], old_seeds=[], old_graphs=[])
    trained = dict(states={}, numerical_training_calls=0)
    graphs = dict(
        graph_manifest=[],
        complete=fault == "construction",
        proposal_rejection_rows=[],
        terminal_reason=None,
    )
    atomic_json(raw / "frozen_protocol.json", d.protocol())
    atomic_json(raw / "trained_states.json", trained)
    work = dict(evidence=evidence, trained=trained, graphs=graphs, raw_references=[])
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", {})
    ref = d.calibration.reference(measurement)
    if fault == "bytes":
        ref["sha256"] = "changed"
    if fault == "executed":
        changed = raw / "changed.py"
        changed.write_text("changed")
        ref.update(executed_source=True, source_path=str(changed))
    if fault == "frozen":
        atomic_json(raw / "frozen_protocol.json", {})
    value = dict(
        source_artifact_hashes=[], code_config_hashes=[], raw_shard_hashes=[ref], fixture_only=True
    )
    atomic_json(path, value)
    monkeypatch.setattr(d, "build", lambda *_: value)
    monkeypatch.setattr(d, "authenticate", lambda _: evidence)
    monkeypatch.setattr(
        d, "train", lambda _: dict(trained, changed=True) if fault == "trained" else trained
    )
    monkeypatch.setattr(d, "verify_graphs", lambda *_: True)
    monkeypatch.setattr(d, "graph_protocol", lambda _: dict(graphs, complete=False))
    assert not runner.replay(path)["passed"]


def test_construction_progress_and_collection_ceiling(monkeypatch):
    """REQ-REPORT-8959: both history and proposal loops expose bounded completed/pending counts."""
    graph = dict(
        nodes=["AA", "BB", "CC", "DD"], edges=[["AA", "BB"], ["BB", "CC"], ["CC", "DD"]], start="AA"
    )
    monkeypatch.setattr(d.source, "candidate", lambda _: deepcopy(graph))
    ticks = iter([0, 0, 35, 35, 35])
    monkeypatch.setattr(d.time, "monotonic", lambda: next(ticks, 35))
    assert (
        d.graph_protocol(dict(old_seeds=[1], old_graphs=[]))["terminal_reason"]
        == "proposal_budget_exhausted"
    )
    counter = iter(np.arange(0, 550, 0.1).tolist())
    monkeypatch.setattr(d.time, "monotonic", lambda: next(counter))
    assert (
        d.graph_protocol(dict(old_seeds=[], old_graphs=[]))["terminal_reason"]
        == "proposal_budget_exhausted"
    )
    ticks = iter([0, 0, 601, 602])
    monkeypatch.setattr(d.time, "monotonic", lambda: next(ticks))
    assert (
        d.graph_protocol(dict(old_seeds=[], old_graphs=[]))["terminal_reason"]
        == "construction_time_exhausted"
    )
