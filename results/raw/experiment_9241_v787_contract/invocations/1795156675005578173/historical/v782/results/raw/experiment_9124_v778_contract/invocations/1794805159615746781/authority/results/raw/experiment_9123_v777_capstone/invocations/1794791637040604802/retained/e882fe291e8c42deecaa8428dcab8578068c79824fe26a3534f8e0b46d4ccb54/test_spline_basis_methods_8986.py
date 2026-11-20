"""REQ-KAN-8986 / REQ-REPORT-8986 / REQ-VERIFY-8986: convex mechanics and custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit

from carnot.autoresearch import spline_basis_8986 as n
from carnot.reporting import spline_basis_methods_8986 as d
from carnot.reporting import spline_basis_runner_8986 as runner
from carnot.reporting.current_work_receipt import atomic_json

pytestmark = pytest.mark.memory_watchdog_skip


def test_basis_scaling_and_constants():
    """SCENARIO-KAN-8986-NUMERIC: endpoints, partition and fixed dimensions survive constants."""
    x = np.tile(np.linspace(-1, 1, 33)[:, None], (1, 27))
    x[:, 3] = 5
    scaling = n.fit_scaling(x)
    scaled, clipping = n.scale(x, scaling)
    assert np.array_equal(scaled[:, 3], np.zeros(33))
    assert sum(clipping["below"]) + sum(clipping["above"]) == 0
    later = x.copy()
    later[0, 0], later[-1, 1], later[0, 3] = -5, 5, 99
    future, clipped = n.scale(later, scaling)
    assert future[0, 0] == -1 and future[-1, 1] == 1 and future[0, 3] == 0
    assert clipped["below"][0] == clipped["above"][1] == clipped["above"][3] == 1
    phi = n.design(scaled, True)
    basis = phi[:, 28:].reshape(33, 27, 7)
    assert phi.shape == (33, 217) and n.design(scaled, False).shape == (33, 28)
    np.testing.assert_allclose(basis.sum(axis=2), 1, atol=1e-15)
    assert np.array_equal(basis[0, 0], [1, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(basis[-1, 0], [0, 0, 0, 0, 0, 0, 1])
    for bad in (np.zeros((3, 26)), np.full((2, 27), np.nan), np.zeros(27)):
        with pytest.raises(ValueError, match="features"):
            n.fit_scaling(bad)
    with pytest.raises(ValueError, match="scaled_domain"):
        n.design(np.full((2, 27), 2.0), True)


def test_gradient_hessian_duplicates_and_orientation(monkeypatch):
    """SCENARIO-KAN-8986-NUMERIC: actual derivatives, ridge duplicates and error labels."""
    x = np.tile(np.linspace(-1, 1, 16)[:, None], (1, 27))
    x[:, 5] = 0
    phi = n.design(x, True)
    y = np.asarray([0.0] * 8 + [1.0] * 8)
    c = np.random.default_rng(71).normal(0, 0.1, 217)
    loss, grad = n.objective(c, phi, y)
    numeric = []
    for i in range(len(c)):
        delta = np.eye(1, len(c), i)[0] * 1e-6
        numeric.append(
            (n.objective(c + delta, phi, y)[0] - n.objective(c - delta, phi, y)[0]) / 2e-6
        )
    np.testing.assert_allclose(grad, numeric, atol=1e-8)
    p = expit(phi @ c)
    hessian = phi.T @ ((p * (1 - p))[:, None] * phi) / len(y)
    hessian += np.diag([0.0] + [0.01] * 216)
    assert np.linalg.eigvalsh(hessian).min() >= -1e-12 and np.isfinite(loss)
    state = n.fit(phi, y)
    assert state["converged"] and state["gradient_inf"] <= 1e-6
    assert expit(phi @ state["coefficients"])[-1] > expit(phi @ state["coefficients"])[0]
    assert n.numeric_audit()["passed"]
    monkeypatch.setattr(
        n,
        "minimize",
        lambda *a, **kw: SimpleNamespace(
            x=np.zeros(217), success=True, message="false success", nit=0
        ),
    )
    with pytest.raises(ValueError, match="nonconvergence"):
        n.fit(phi, y)
    for bad in (np.full(16, 2), np.zeros(15), np.full(16, np.nan)):
        with pytest.raises(ValueError, match="fit_operands"):
            n.fit(phi, bad)


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8986: real immutable historical observations qualify numerical mechanics."""
    return runner.freeze(d.ROOT, tmp_path_factory.mktemp("8986-work"))


def test_natural_freeze_and_exact_permutation(work):
    """SCENARIO-VERIFY-8986-INVARIANCE: permutation is exact; source and truth cannot enter."""
    assert not work["evidence"]["failures"]
    r = work["result"]
    assert len(r["head_rows"]) == 6 and all(h["gradient_inf"] <= 1e-6 for h in r["head_rows"])
    assert len(r["rows"]) == 320 * 5 and len(r["encoder_hashes"]) == 3
    assert all(row["passed"] for row in r["invariant_rows"])
    assert r["summary_recomputation"]["passed"]
    state = work["evidence"]["work"]["evidence"]["states"]["relational_logistic"][0]
    row = work["evidence"]["work"]["evidence"]["views"]["guard"]["rows"][0]
    renamed = {name: f"new-{i}" for i, name in enumerate(row["nodes"])}
    permuted = dict(
        row,
        nodes=[renamed[v] for v in row["nodes"][::-1]],
        edges=[[renamed[b], renamed[a]] for a, b in row["edges"][::-1]],
        query_node=renamed[row["query_node"]],
        proposed_node=renamed[row["proposed_node"]] if row["proposed_node"] is not None else None,
    )
    assert np.array_equal(n.features(state, [row]), n.features(state, [permuted]))
    for field in (
        "BFS",
        "label",
        "labels",
        "graph_id",
        "distance",
        "shortest_distance",
        "answer_membership",
        "source_metadata",
    ):
        with pytest.raises(ValueError):
            n.features(state, [dict(row, **{field: 1})])
    fixture = d.build(work, dict(receipts=[], coverage={}), True)
    assert fixture["spline_methods_ready_score"] == fixture["independent_generalization_score"] == 0
    assert fixture["verdict_class"] == "null" and fixture["MODEL_SPECS"] == []
    assert not any(fixture["model_invocation_counts"].values())
    assert d.build(work, dict(receipts=[], coverage={}), False)["verdict_class"] == "disqualified"
    p = json.loads((d.ROOT / d.PROTOCOL).read_text())
    assert len(p["seven_bound_contract"]["bounds"]) == 7 and p["optimizer"]["bounds"] is None
    assert set(fixture) <= set(fixture["field_principles"])


def test_missing_changed_and_typed_gate(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8986-CUSTODY: missing and altered external bytes have actual operands."""
    evidence = d.authenticate(tmp_path, tmp_path / "raw")
    assert evidence["failures"][0]["observed"] == "missing_file"
    monkeypatch.setattr(d.prior, "INPUT", "upstream.json")
    path = tmp_path / "upstream.json"
    atomic_json(path, {})
    assert d.authenticate(tmp_path, tmp_path / "raw")["failures"][0]["observed"] == "changed_bytes"
    monkeypatch.setattr(d.prior, "PIN", d.reference(path)["sha256"])
    assert d.authenticate(tmp_path, tmp_path / "raw")["failures"][0]["observed"] == "missing_field"


def test_private_cli_replay_tamper_and_dates(tmp_path):
    """SCENARIO-REPORT-8986-CUSTODY: real children from a foreign cwd cover publication."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    output = tmp_path / (d.NAME + ".json")

    def child(*args):
        print("before_private_child", args, flush=True)
        completed = subprocess.run(
            [sys.executable, "-u", str(d.ROOT / d.CLI), *args],
            cwd=tmp_path,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        print(completed.stdout, "after_private_child", completed.returncode, flush=True)
        return completed.returncode

    assert child("--date", "wrong") == 2
    assert child("--date", "20261111", "--fixture-e2e", str(d.ROOT / (d.NAME + ".json"))) == 2
    assert child("--date", "20261111", "--fixture-e2e", str(output)) == 0
    value = json.loads(output.read_text())
    assert value["fixture_only"] and value["completed_count"] == 320
    assert child("--cold-replay", str(output)) == 0
    changed = deepcopy(value)
    changed["rows"][0]["value"] += 0.1
    changed["reproducibility_checksum"] = d.checksum(changed)
    atomic_json(output, changed)
    assert child("--cold-replay", str(output)) == 1
    atomic_json(output, value)
    ref = value["trained_state_reference"]
    saved = Path(ref["path"]).read_bytes()
    Path(ref["path"]).write_text("{}")
    assert child("--cold-replay", str(output)) == 1
    Path(ref["path"]).write_bytes(saved)
    # Repairing aggregate hashes cannot change pinned observations or the frozen reduction.
    workpath = Path(value["work_reference"]["path"])
    w = json.loads(workpath.read_text())
    w["result"]["rows"][0]["value"] += 0.1
    atomic_json(workpath, w)
    w["raw_references"] = [d.reference(Path(r["path"])) for r in value["raw_shard_hashes"]]
    changed = runner.artifact(w, dict(receipts=[], coverage={}), True)
    atomic_json(output, changed)
    assert child("--cold-replay", str(output)) == 1
    blocked = tmp_path / "blocked" / output.name
    env["CARNOT_8986_INPUT_ROOT"] = str(tmp_path / "absent")
    assert child("--date", "20261111", "--fixture-e2e", str(blocked)) == 0
    b = json.loads(blocked.read_text())
    assert b["verdict_class"] == "blocked" and b["spline_methods_ready_score"] == 0
    assert b["gate_check_summary"][0]["observed"] == "missing_file"
    assert len(b["rows"]) == 320 * 5 and all(r["value"] is None for r in b["rows"])
    assert child("--cold-replay", str(blocked)) == 0


def test_replay_and_invalid_rosters(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8986-CUSTODY: independent reduction binds fit roles and features."""
    raw = Path(work["raw"])
    assert d.evaluate(work["evidence"], raw, replay=True) == work["result"]
    original = deepcopy(work["evidence"])
    views = work["evidence"]["work"]["evidence"]["views"]
    monkeypatch.setitem(views["fit"], "unit_ids", ["bad", *views["fit"]["unit_ids"][1:]])
    with pytest.raises(ValueError, match="development_roles"):
        d.evaluate(work["evidence"], tmp_path)
    assert runner.replay(tmp_path / "absent")["passed"] is False
    blocked = dict(
        work,
        evidence=dict(original, failures=[dict(check="test", observed="missing_file")]),
        result={},
    )
    value = d.build(blocked, dict(receipts=[], coverage={}), True)
    assert value["honest_verdict"] == "complete_blocked_test"


def test_network_failure_and_owned_measurement_refusal(work, tmp_path, monkeypatch):
    """REQ-REPORT-8986: external network failure blocks; owned numerical failure disqualifies."""
    evidence = deepcopy(work["evidence"])
    evidence["method_source_rows"] = evidence["method_source_rows"][:1]

    def offline(*args, **kwargs):
        raise OSError("private network unavailable")

    monkeypatch.setattr(d.urllib.request, "urlopen", offline)
    d.preflight(tmp_path, evidence)
    assert evidence["failures"][-1]["observed"] == "private network unavailable"
    monkeypatch.setattr(d, "authenticate", lambda *a: deepcopy(work["evidence"]))
    monkeypatch.setattr(d, "preflight", lambda *a: work["preconditions"])

    def refuse(*args, **kwargs):
        raise ValueError("nonconvergence")

    monkeypatch.setattr(d, "evaluate", refuse)
    refused = runner.freeze(d.ROOT, tmp_path / "refused")
    assert refused["owned_failure"] == "nonconvergence"
    assert d.build(refused, dict(receipts=[], coverage={}), True)["verdict_class"] == "disqualified"


def test_encoder_and_cached_feature_refusal(work, tmp_path, monkeypatch):
    """REQ-VERIFY-8986: neither changed encoder bytes nor foreign features can pass."""
    evidence = deepcopy(work["evidence"])
    state = evidence["work"]["evidence"]["states"]["relational_logistic"][0]
    state["params"]["encoder"]["bias"][0] += 0.1
    with pytest.raises(ValueError, match="encoder_bytes"):
        d.evaluate(evidence, tmp_path)
    monkeypatch.setattr(d.n, "features", lambda state, rows: np.full((len(rows), 27), 999.0))
    with pytest.raises(ValueError, match="historical_features"):
        d.evaluate(work["evidence"], tmp_path)


def test_replay_refuses_code_authority_and_protocol(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8986-CUSTODY: coordinated rehashing cannot replace immutable authority."""
    raw = tmp_path / "retained"
    value_path = tmp_path / (d.NAME + ".json")
    # Full upstream replay is covered by the CLI; these private cases isolate early refusals.
    monkeypatch.setattr(runner.historical, "load", lambda evidence, ref: evidence["work"])

    def candidate(w):
        w["evidence"]["work"] = {}
        w["evidence"]["context"] = {}
        w["raw"] = str(raw)
        atomic_json(raw / "work.json", w)
        atomic_json(raw / "validation.json", dict(receipts=[], coverage={}))
        w["raw_references"] = [d.reference(raw / "work.json"), d.reference(raw / "validation.json")]
        atomic_json(value_path, runner.artifact(w, dict(receipts=[], coverage={}), True))
        return runner.replay(value_path)

    changed = deepcopy(work)
    changed["sources"][0]["source_path"] = str(tmp_path / "missing-code.py")
    assert candidate(changed)["reason"] == "executed_bytes_drift"
    changed = deepcopy(work)
    changed["evidence"]["producer"]["diagnosis_ready_score"] = 0
    assert candidate(changed)["reason"] == "pinned_operand_drift"
    changed = deepcopy(work)
    ref = next(
        r
        for r in changed["evidence"]["references"]
        if r.get("source_path") == str(d.ROOT / d.PROTOCOL)
    )
    ref["source_path"] = "removed-source"
    changed["evidence"]["references"].append(
        dict(d.reference(d.ROOT / d.CONTEXT), source_path=str(d.ROOT / d.PROTOCOL))
    )
    assert candidate(changed)["reason"] == "protocol_operand_drift"
    changed = deepcopy(work)
    changed["evidence"]["protocol"]["optimizer"]["maxiter"] = 2
    assert candidate(changed)["reason"] == "protocol_operand_drift"


def test_production_validation_dispatch_stays_unqualified(work, tmp_path, monkeypatch):
    """REQ-REPORT-8986: mocked dispatch covers orchestration but cannot supply readiness."""
    from carnot.reporting import v756_branch_qualification as coverage_helpers

    private = tmp_path / "checks"
    private.mkdir()
    (private / ".coverage").write_bytes(b"private dispatch fixture; not SQLite evidence")
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **kw: str(private))
    monkeypatch.setattr(d, "ROOT", tmp_path)
    fixture_work = deepcopy(work)
    fixture_work["evidence"]["work"] = {}
    fixture_work["evidence"]["producer"] = {}
    fixture_work["evidence"]["context"] = {}
    monkeypatch.setattr(
        runner, "freeze", lambda root, raw: dict(deepcopy(fixture_work), raw=str(raw))
    )
    monkeypatch.setattr(coverage_helpers, "measured", lambda *a: {})
    calls = []

    def commands(root, specs, **kwargs):
        calls.append([s.name for s in specs])
        return []

    def publish(output, value, validator):
        assert value["spline_methods_ready_score"] == 0 and value["verdict_class"] == "disqualified"
        candidate = tmp_path / "candidate.json"
        atomic_json(candidate, value)
        validator(candidate)
        return dict(primary_path=str(output))

    monkeypatch.setattr(runner, "run_commands", commands)
    monkeypatch.setattr(runner, "publish_primary", publish)
    assert runner.main(["--date", "20261111"]) == 0
    assert calls[-1] == ["cold_replay", "adversarial", "strict_rows"]
    assert calls[-2] == ["full_python_suite"]
