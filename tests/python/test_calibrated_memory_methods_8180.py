"""REQ-VERIFY-8180 / REQ-REPORT-8180: private qualification precedes natural fitting."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import calibrated_memory_methods_8180 as e


def test_convex_scalar_reference_and_bounds():
    """SCENARIO-VERIFY-8180-FIXTURES: independent equations detect fit errors."""
    rows, labels = e.fixture("learnable")
    state = e.genesis(rows, 101)
    pool = [dict(r, y=y) for r, y in zip(rows[:64], labels[:64], strict=True)]
    h = state["arms"]["error_center"]
    x, y = e.operands(h, state["geometry"], pool)
    theta = np.array([1.0, 0.0, *([0.0] * 16)])
    loss, gradient = e.objective(theta, x, y)
    ref_loss, ref_gradient = e.objective(theta, x, y, reference=True)
    assert abs(loss - ref_loss) < 1e-8
    np.testing.assert_allclose(gradient, ref_gradient, atol=1e-12)
    trained = e.train(h, state["geometry"], pool)
    assert trained["fit"]["converged"]
    assert trained["fit"]["projected_gradient"] <= 1e-6
    assert trained["fit"]["objective_agreement"] < 1e-8
    assert 0 <= trained["scale"] <= 2 and -8 <= trained["intercept"] <= 8
    assert max(map(abs, trained["weights"])) <= 4
    stopped = e.train(h, state["geometry"], pool, maxiter=0)
    assert not stopped["fit"]["converged"]
    with pytest.raises(ValueError, match="optimizer_operands"):
        e.train(h, state["geometry"], [])
    for v in [rows[0]["values"], [1000.0, *([0.0] * 8)], [-1000.0, *([0.0] * 8)]]:
        assert (
            abs(
                e.probability(trained, state["geometry"], v)
                - e.scalar_probability(trained, state["geometry"], v)
            )
            < 1e-12
        )
    c = dict(base=h, head=trained)
    mid = e.interpolate(c, 0.5)
    assert mid["scale"] == (h["scale"] + trained["scale"]) / 2


def test_schedule_admission_and_overflow(monkeypatch):
    """REQ-VERIFY-8180: only converged heads reach twelve future admission labels."""
    rows, labels = e.fixture("learnable")
    s = e.run(rows, labels, 101)
    f = e.fixture_summary(s, rows)
    assert f["install_slot"] <= 208
    assert f["changed_later_decisions"] >= 32
    assert len(s["arms"]["calibration_only"]["centers"]) == 16
    assert len(s["arms"]["error_center"]["centers"]) == 24
    assert len(s["used_admission"]) == len(set(s["used_admission"]))
    assert all(e.engine.bucket(rows[i - 1]) == 0 for i in s["used_admission"])
    assert all(e.engine.bucket(rows[i - 1]) != 0 for i in s["training"])
    assert all(h["fit"]["projected_gradient"] <= 1e-6 for h in s["arms"].values() if "fit" in h)
    assert e.run(rows, labels, 101, capacity=8)["lost"]
    original = e.train

    def failed(*args, **kwargs):
        h = original(*args, **kwargs)
        h["fit"]["converged"] = False
        return h

    monkeypatch.setattr(e, "train", failed)
    rejected = e.run(rows, labels, 101)
    assert not rejected["used_admission"]
    assert all(h["scale"] == 1 for h in rejected["arms"].values())


def test_no_signal_and_ineligible():
    """SCENARIO-VERIFY-8180-FIXTURES: circular data cannot prove new signal."""
    rows, labels = e.fixture("no_signal")
    summary = e.fixture_summary(e.run(rows, labels, 101), rows)
    assert summary["benefit_claim"] is False
    _, labels = e.fixture("learnable")
    labels = [None] * len(labels)
    s = e.run(rows, labels, 101)
    assert not s["candidates"] and not s["used_admission"]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8180: generated fixtures remain in private temporary custody."""
    raw = tmp_path_factory.mktemp("8180-measurement")
    return e.measure(e.ROOT, raw, fixture=True), raw


def test_measure_build_and_replay(work):
    """SCENARIO-REPORT-8180-CLI: independently reduced primaries reject forgery."""
    measured, raw = work
    assert measured["future_decision_fixture_score"] == 1
    assert measured["restart_fixture"]["passed"]
    assert measured["optimizer_fixture_rows"]
    value = e.build(measured, raw, [dict(passed=True, normal_exit=True)])
    assert value["verdict_class"] == "circular_positive"
    assert value["calibrated_memory_ready_score"] == 1
    assert value["stream_input_ready_score"] == 1
    assert value["MODEL_SPECS"] == [] and value["call_ledger"] == []
    assert (
        value["generalized_learning_benefit_score"]
        == value["independent_generalization_score"]
        == 0
    )
    assert value["H2"]["status"] == "registered_not_measured"
    path = raw / "artifact.json"
    atomic_json(path, value)
    assert e.replay(path)
    assert not e.replay(raw / "absent.json")
    for kind in [
        "rows",
        "future_decision_fixture_score",
        "completed_count",
        "saturation_rows",
        "code_config_hashes",
    ]:
        bad = deepcopy(value)
        if kind == "rows":
            bad[kind][0]["numerator"] += 1
        elif kind == "saturation_rows":
            bad[kind][0]["residual"] += 1
        elif kind == "code_config_hashes":
            bad[kind][e.MODULE] = "sha256:wrong"
        else:
            bad[kind] += 1
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)
    atomic_json(path, value)
    assert (
        e.build(measured, raw, [dict(passed=True, normal_exit=False)])["verdict_class"]
        == "disqualified"
    )


def test_external_blocks(tmp_path):
    """REQ-REPORT-8180: an external failed operand is terminal and explicit."""
    missing = e.measure(tmp_path, tmp_path / "missing", fixture=True)
    value = e.build(missing, tmp_path, [dict(passed=True, normal_exit=True)])
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][-1]["observed"] is False
    assert value["calibrated_memory_ready_score"] == 0
    original = json.loads((e.ROOT / e.UPSTREAM).read_text())
    original["learning_audit_ready_score"] = 0
    atomic_json(tmp_path / e.UPSTREAM, original)
    changed = e.measure(tmp_path, tmp_path / "gate", fixture=True)
    assert changed["gate_check_summary"][-1]["artifact_field"] == "learning_audit_ready_score"


def test_private_cli_and_manifest(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8180-CLI: E2E-016 success/block/tamper/replay outside checkout."""
    from carnot.reporting import calibrated_memory_execution_8180 as runner

    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    if env.get("COVERAGE_RCFILE"):
        env["COVERAGE_PROCESS_START"] = env["COVERAGE_RCFILE"]
    cli = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]

    def child(args, expected):
        print("[test8180] before_subprocess", args, flush=True)
        result = subprocess.run(
            cli + args, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=240
        )
        print("[test8180] after_subprocess", result.returncode, flush=True)
        assert result.returncode == expected, result.stdout + result.stderr

    p = tmp_path / (e.NAME + ".json")
    child(["--fixture-output", str(p)], 0)
    child(["--cold-replay", str(p)], 0)
    bad = json.loads(p.read_text())
    bad["rows"][0]["numerator"] += 1
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(p, bad)
    child(["--cold-replay", str(p)], 1)
    child(
        [
            "--fixture-output",
            str(tmp_path / "blocked" / (e.NAME + ".json")),
            "--root",
            str(tmp_path / "absent"),
        ],
        0,
    )
    child(["--fixture-output", str(e.ROOT / "results/forbidden.json")], 2)
    child(
        [
            "--root",
            str(tmp_path / "absent"),
            "--worker-output",
            str(tmp_path / "worker/measurement.json"),
        ],
        0,
    )
    private = tmp_path / "validation"
    private.mkdir()
    specs = runner.manifest(private, tmp_path / "candidate.json")
    assert all(
        "::" not in a
        for s in specs["commands"]
        if s["name"] in ["ruff_check", "ruff_format", "strict_mypy", "spec_coverage"]
        for a in s["argv"]
    )
    assert specs["repository_health"]["argv"][-2:] == ["tests/python", "-q"]
    # The reused supervisor owns publication; this adapter must propagate its status.
    monkeypatch.setattr(runner.previous, "main", lambda argv: 0)
    assert runner.main([]) == 0


def test_terminal_custody_and_rehashed_private_evidence(work, tmp_path):
    """REQ-REPORT-8180: resealed private bytes cannot replace independent replay."""
    from carnot.reporting.current_work_receipt import sha256_file

    measured, raw = work
    binder = e.engine.methods.Custody(tmp_path)
    assert e.authenticate(e.ROOT, binder, False)["experiment_id"] == 8172
    value = e.build(measured, raw, [dict(passed=True, normal_exit=True)])
    path = tmp_path / "forged.json"

    def check(bad):
        bad.pop("reproducibility_checksum", None)
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)

    broken = deepcopy(value)
    broken["reproducibility_checksum"] = "wrong"
    atomic_json(path, broken)
    assert not e.replay(path)
    for field, wrong in [("experiment_id", 8181), ("protocol_sha256", "wrong")]:
        check(dict(value, **{field: wrong}))
    log = tmp_path / "log.txt"
    log.write_text("changed private validation log\n")
    check(
        dict(
            value,
            validation_receipts=[
                dict(passed=True, normal_exit=True, log_path=str(log), log_sha256="wrong")
            ],
        )
    )
    fixture_path = Path(value["fixture_states"][0]["state"]["path"])
    saved_bytes = fixture_path.read_bytes()
    fixture_path.write_text("{}\n")
    check(deepcopy(value))
    fixture_path.write_bytes(saved_bytes)
    # A forged trajectory with a fresh shard hash must still fail arithmetic replay.
    primitive_ref = next(
        r for r in value["raw_shard_hashes"] if r["path"].endswith("primitive_evidence.json")
    )
    primitive_path = Path(primitive_ref["path"])
    primitive_bytes = primitive_path.read_bytes()
    altered = json.loads(saved_bytes)
    altered["issued"][100]["predictions"]["error_center"] += 0.1
    atomic_json(fixture_path, altered)
    bad = deepcopy(value)
    digest = sha256_file(fixture_path)
    bad["fixture_states"][0]["state"]["sha256"] = digest
    primitive = json.loads(primitive_bytes)
    primitive["fixture_states"] = bad["fixture_states"]
    atomic_json(primitive_path, primitive)
    for ref in bad["raw_shard_hashes"]:
        if ref["path"] in [str(primitive_path), str(fixture_path)]:
            ref["sha256"] = sha256_file(Path(ref["path"]))
    check(bad)
    fixture_path.write_bytes(saved_bytes)
    primitive_path.write_bytes(primitive_bytes)


def test_missing_fixture_roster_disqualifies(tmp_path):
    """SCENARIO-VERIFY-8180-ROSTER: declared success cannot replace twenty seeds."""
    truncated = dict(
        input_ready=1,
        future_decision_fixture_score=1,
        overflow_fixture=dict(passed=True),
        restart_fixture=dict(passed=True),
        fixture_states=[],
        gate_check_summary=[],
        rows=[],
    )
    result = e.build(truncated, tmp_path, [dict(passed=True, normal_exit=True)])
    assert result["calibrated_memory_ready_score"] == 0
    assert result["verdict_class"] == "disqualified"


def test_rehashed_metric_and_diagnosis_shards(work, tmp_path):
    """SCENARIO-VERIFY-8180-ROSTER: forged reductions fail beyond their new hashes."""
    from carnot.reporting.current_work_receipt import sha256_file

    measured, raw = work
    value = e.build(measured, raw, [dict(passed=True, normal_exit=True)])
    primitive_path = Path(
        next(
            r["path"]
            for r in value["raw_shard_hashes"]
            if r["path"].endswith("primitive_evidence.json")
        )
    )
    original_bytes = primitive_path.read_bytes()
    original = json.loads(original_bytes)
    for case in ["metrics", "diagnosis"]:
        bad = deepcopy(value)
        primitive = deepcopy(original)
        # Removing every trajectory tests truncation without re-running valid trajectories.
        for key in ["fixture_states", "fixture_summaries", "optimizer_fixture_rows"]:
            bad[key] = primitive[key] = []
        if case == "diagnosis":
            bad["rows"] = primitive["rows"] = []
            bad["saturation_rows"][0]["residual"] += 0.1
            primitive["saturation_rows"] = bad["saturation_rows"]
        atomic_json(primitive_path, primitive)
        for ref in bad["raw_shard_hashes"]:
            if ref["path"] == str(primitive_path):
                ref["sha256"] = sha256_file(primitive_path)
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        path = tmp_path / "forged.json"
        atomic_json(path, bad)
        assert not e.replay(path)
        primitive_path.write_bytes(original_bytes)


def test_coverage_collector_and_health_custody(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8180-HEALTH-CUSTODY: retry never repeats repository health."""
    from carnot.reporting import calibrated_memory_execution_8180 as runner
    from carnot.reporting.current_work_receipt import sha256_file

    private = tmp_path / "validation"
    private.mkdir()
    frozen = runner.manifest(private, tmp_path / "candidate.json")
    assert all(
        "--data-file=" + str(private / ".coverage") in s["argv"]
        for s in frozen["commands"]
        if s["name"].startswith("coverage_")
    )
    log = tmp_path / "suite.log"
    log.write_text("repository health failed normally\n")
    health = dict(
        frozen["repository_health"],
        passed=False,
        actual_exit=1,
        log_path=str(log),
        log_sha256=sha256_file(log),
    )
    source = tmp_path / "measurement.json"
    atomic_json(source, dict(global_health=health))
    monkeypatch.setenv("CARNOT_8180_HEALTH_WORK", str(source))
    reused = runner.check(e.ROOT, frozen["repository_health"], private, tmp_path)
    assert reused["actual_exit"] == 1 and not reused["passed"] and reused["reused"]
    monkeypatch.setattr(runner, "BASE_CHECK", lambda *args, **kwargs: dict(passed=True))
    assert runner.check(e.ROOT, dict(name="other"), private, tmp_path)["passed"]
    monkeypatch.setattr(
        runner.time, "sleep", lambda seconds: atomic_json(source, dict(global_health=health))
    )
    atomic_json(source, {})
    assert runner.check(e.ROOT, frozen["repository_health"], private, tmp_path)["reused"]
    log.write_text("tampered\n")
    with pytest.raises(ValueError, match="health_custody"):
        runner.check(e.ROOT, frozen["repository_health"], private, tmp_path)
