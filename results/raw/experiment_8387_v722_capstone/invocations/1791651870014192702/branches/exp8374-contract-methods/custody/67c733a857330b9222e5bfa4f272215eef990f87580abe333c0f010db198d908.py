"""REQ-KAN-8334 / REQ-VERIFY-8334 / REQ-REPORT-8334: static fit custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import json
from pathlib import Path

import numpy as np
import pytest

from carnot.verify import sentence_spline_fit_8334 as n
from carnot.reporting import sentence_spline_fit_8334 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting import sentence_spline_execution_8334 as runner
from carnot.reporting import v718_replay_history as history
from carnot.reporting import v718_replay_runner as finding_runner


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Real cached operands stay separate from private test outputs."""
    raw = tmp_path_factory.mktemp("sentence8334")
    return e.measure(e.ROOT, raw), raw


def test_numerics_and_exact_optimizer():
    """SCENARIO-KAN-8334-FIT: qualify basis and gradient independently."""
    audit = n.numeric_audit()
    assert audit["passed"]
    assert audit["basis_error"] < 1e-12
    assert audit["gradient_error"] < 1e-7
    control = n.optimizer_control()
    assert control["passed"]
    assert control["final"]["loss"] < control["initial"]["loss"]
    assert control["initial_action"] != control["final_action"]
    for arm in n.ARMS:
        x = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0]])
        phi = n.matrix(arm, x, {"centers": [[0.0] * 4] * 32, "width": 1.0})
        head = n.fit(phi, np.array([0.0, 1.0]))
        assert len(head["coefficients"]) == n.COUNTS[arm]
        assert len(head["loss_rows"]) == 401
        assert np.isfinite(head["coefficients"]).all()
    with pytest.raises(ValueError):
        n.matrix("bad", x, {})


def test_geometry_policy_and_separation():
    """REQ-KAN-8334: center selection ignores labels and policy ties escalate."""
    x = np.array([[i / 40.0] * 4 for i in range(40)])
    g = n.geometry(x, [f"{i:03}" for i in range(40)])
    assert g["center_ids"][0] == "000"
    assert len(g["centers"]) == 32
    assert g == n.geometry(x[::-1], [f"{i:03}" for i in range(39, -1, -1)])
    with pytest.raises(ValueError, match="geometry"):
        n.geometry(x[:20], list(map(str, range(20))))
    assert [n.action(p) for p in [None, 0.25, 0.75, 0.1, 0.9]] == [
        "escalate",
        "escalate",
        "escalate",
        "accept",
        "reject",
    ]
    assert n.cost("accept", 1) == n.cost("reject", 0) == 1
    assert n.cost("escalate", 0) == 0.5
    assert n.cost("accept", 0) == 0
    h = {"arm": "scalar2", "coefficients": [1.0, 0.0], "temperature": 1.0, "geometry": {}}
    assert n.predict(h, {"features": [0.0] * 5}) == 0.5
    for bad in [
        {"features": [0.0] * 5, "y": 1},
        {"features": [0.0] * 5, "unit_id": "x"},
        {"features": [0.0, 2.0, 0.0, 0.0, 0.0]},
        {"features": [float("nan")] * 5},
    ]:
        with pytest.raises(ValueError):
            n.predict(h, bad)
    assert n.predict(h, {"features": None}) is None


def test_authentic_fit_and_cold_reduction(measured, tmp_path):
    """SCENARIO-VERIFY-8334-CUSTODY: recount only original fit/tune roles."""
    work, raw = measured
    assert not work["failures"]
    assert [work["support"][k]["usable"] for k in e.ROLES] == [104, 25, 28]
    assert work["trained"]["selected_comparator"] in n.ARMS[1:]
    value = e.build(work, raw, [{"passed": True}])
    assert value["heads_ready_score"] == 1
    assert value["independent_generalization_score"] == 0
    assert value["sigmoid_equivalence_error"] < 1e-12
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    for key, changed in [("heads_ready_score", 99), ("selected_comparator", "scalar2_bad")]:
        bad = dict(value, **{key: changed})
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)
    assert not e.replay(tmp_path / "absent")
    atomic_json(path, {})
    assert not e.replay(path)
    assert e.build(work, raw, [{"passed": False}])["verdict_class"] == "disqualified"
    coverage = tmp_path / "coverage.json"
    atomic_json(coverage, dict(totals=dict(percent_covered=0), files={}))
    assert (
        e.build(
            dict(work, owned_coverage_reference=e.reference(coverage)), raw, [{"passed": True}]
        )["heads_ready_score"]
        == 0
    )
    atomic_json(
        coverage,
        dict(
            totals=dict(percent_covered=100),
            files={p: dict(summary=dict(missing_lines=0)) for p in e.OWNED},
        ),
    )
    assert (
        e.build(
            dict(work, owned_coverage_reference=e.reference(coverage)), raw, [{"passed": True}]
        )["heads_ready_score"]
        == 1
    )
    blocked = e.measure(tmp_path / "absent", tmp_path / "blocked")
    assert e.build(blocked, tmp_path / "blocked", [{"passed": True}])["verdict_class"] == "blocked"
    assert blocked["failures"][0]["observed"] is None
    bad = deepcopy(work)
    bad["optimizer_control"]["passed"] = False
    assert (
        e.build(bad, raw, [{"passed": True}])["honest_verdict"] == "complete_null_optimizer_control"
    )


def test_shard_tampering_and_information_boundary(measured, tmp_path):
    """REQ-VERIFY-8334: even rehashed source and label substitutions fail."""
    work, _ = measured
    bundle = deepcopy(work["bundle"])
    for field in ["source_cluster_id", "y"]:
        bad = deepcopy(bundle)
        target = bad["evaluators"]["fit"][0]
        target[field] = "wrong"
        with pytest.raises(ValueError):
            e.validate_bundle(bad)
    bad = deepcopy(bundle)
    bad["predictors"]["fit"][0]["unit_id"] = "wrong"
    with pytest.raises(ValueError):
        e.validate_bundle(bad)
    for role in e.ROLES:
        assert all("y" not in p for p in bundle["predictors"][role])
    bad = deepcopy(bundle)
    for row in bad["evaluators"]["fit"]:
        row["y"] = None
    assert e.validate_bundle(bad)["fit"]["usable"] == 0


def test_real_cli_and_publication_recovery(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8334-CLI: real children replay and recover rejected bytes."""
    output = tmp_path / (e.NAME + ".json")
    prefix = [sys.executable]
    if "COVERAGE_RCFILE" in os.environ:
        prefix += ["-m", "coverage", "run", "--rcfile=" + os.environ["COVERAGE_RCFILE"]]
    result = subprocess.run(
        prefix + [str(e.ROOT / e.CLI), "--private-run", "--output", str(output)],
        capture_output=True,
        timeout=60,
        cwd="/tmp",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert e.replay(output)
    raw = Path(value["work_reference"]["path"]).parent
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    for args in [["--date", "20000101"], ["--private-run"]]:
        with pytest.raises(SystemExit):
            runner.main(args)
    bad = dict(value, heads_ready_score=99)
    runner.publish(bad, output, raw)
    recovered = json.loads(output.read_bytes())
    assert recovered["verdict_class"] == "disqualified"
    assert recovered["heads_ready_score"] == 0
    assert e.replay(output)
    with pytest.raises(ValueError, match="producer_identity"):
        runner.publish(dict(value, experiment_id=99), output, raw)
    original = runner.manifest

    def bounded(private, candidate):
        plan = original(private, candidate)
        plan["commands"] = [dict(name="actual_failure", argv=["/bin/false"], deadline_s=10)]
        atomic_json(private / "coverage.json", dict(totals=dict(percent_covered=0), files={}))
        return plan

    monkeypatch.setattr(runner, "manifest", bounded)
    blocked = tmp_path / "other" / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "missing"), "--output", str(blocked)]) == 0
    assert json.loads(blocked.read_bytes())["verdict_class"] == "disqualified"


def test_finding_consumer_qualified_here(tmp_path):
    """REQ-VERIFY-8334: retained exact-zero info never authorizes false zero."""
    path = tmp_path / "zero.json"
    value = dict(
        experiment_id=8334,
        task_id=e.TASK,
        run_date="20261009",
        honest_verdict="complete_circular_positive_control",
        verdict_class="circular_positive",
        verifier_is_oracle=True,
        inference_substrate_class="no_model_load",
        inference_substrate="aggregation_from_upstream_artifacts",
        MODEL_SPECS=[],
        dense_sparse_error_max=0.0,
        duration_s=1.0,
        preconditions_checked=True,
        methodology_note="Constructed arithmetic control with separately recomputed primitives and deliberate-error rejection.",
    )
    atomic_json(path, value)
    real = finding_runner.audit(
        path, tmp_path / "legitimate", dict(recomputed=True, deliberate_error_rejected=True)
    )
    false = finding_runner.audit(
        path, tmp_path / "false", dict(recomputed=False, deliberate_error_rejected=True)
    )
    assert real["passed"] and real["findings"]
    assert not false["passed"] and false["findings"]
    report = real["report"]
    for change in [
        dict(candidate_sha256="wrong"),
        dict(verifier_sha256="wrong"),
        dict(reports=[]),
        dict(reports="malformed"),
    ]:
        assert not history.consume(dict(report, **change), path, 1, {})["passed"]
    for severity in ["warn", "critical", "unknown"]:
        changed = deepcopy(report)
        changed["reports"][0]["flags"][0]["severity"] = severity
        assert not history.consume(
            changed, path, 1, dict(recomputed=True, deliberate_error_rejected=True)
        )["passed"]
    assert not history.consume(report, path, 2, {})["passed"]
    unknown = deepcopy(report)
    unknown["reports"][0]["flags"][0]["kind"] = "UNKNOWN_KIND"
    assert not history.consume(
        unknown, path, 1, dict(recomputed=True, deliberate_error_rejected=True)
    )["passed"]
    evidence = runner.qualify_findings(tmp_path / "qualification")
    assert all(r["passed"] for r in evidence["receipts"])
    assert evidence["audits"][0]["passed"] and not evidence["audits"][1]["passed"]


def test_structural_failures_and_rehashed_primitives(measured, tmp_path, monkeypatch):
    """REQ-VERIFY-8334: preserve failure gates and reject rehashed primitive drift."""
    work, _ = measured
    for mutation in ["length", "role", "feature"]:
        b = deepcopy(work["bundle"])
        if mutation == "length":
            b["predictors"]["fit"].pop()
        elif mutation == "role":
            b["predictors"]["fit"][0]["role"] = "reserved"
        else:
            b["predictors"]["fit"][0]["x"] = [float("nan")] * 16
        with pytest.raises(ValueError):
            e.validate_bundle(b)
    assert not e.np_finite([float("inf")])
    identity = deepcopy(work["bundle"])
    identity["predictors"]["fit"][0]["source_sha256"] = "changed"
    with pytest.raises(ValueError, match="source_identity"):
        e.validate_bundle(identity)
    path = tmp_path / "changed_work.json"
    value = e.build(work, Path(work["checkpoint_reference"]["path"]).parent, [dict(passed=True)])
    value["work_reference"]["sha256"] = "wrong"
    atomic_json(path, value)
    assert not e.replay(path)
    for mutation in ["ref", "bundle", "train", "audit", "checkpoint"]:
        raw = tmp_path / mutation
        raw.mkdir()
        bad = deepcopy(work)
        if mutation == "ref":
            bad["refs"][0]["sha256"] = "wrong"
        elif mutation == "bundle":
            bad["bundle"]["evaluators"]["fit"][0]["y"] ^= 1
        elif mutation == "train":
            bad["trained"]["heads"][0]["coefficients"][2] += 0.01
        elif mutation == "audit":
            bad["numeric_audit"]["basis_error"] = 99
        else:
            atomic_json(raw / "checkpoint.json", {})
            bad["checkpoint_reference"] = e.reference(raw / "checkpoint.json")
        atomic_json(raw / "measurement.json", bad)
        path = raw / "candidate.json"
        atomic_json(path, e.build(bad, raw, [dict(passed=True)]))
        assert not e.replay(path)
    actual = n.train
    monkeypatch.setattr(
        n, "train", lambda bundle: (_ for _ in ()).throw(ValueError("insufficient_geometry"))
    )
    blocked = e.measure(e.ROOT, tmp_path / "geometry")
    assert blocked["failures"][0]["upstream"] == "fit_geometry"
    monkeypatch.setattr(n, "train", actual)
    monkeypatch.setattr(
        e, "validate_primary", lambda *args: (_ for _ in ()).throw(ValueError("broken_terminal"))
    )
    broken = e.measure(e.ROOT, tmp_path / "broken")
    assert broken["failures"][0]["artifact_field"] == "structure"
