"""REQ-VERIFY-8212 / REQ-REPORT-8212: original source reductions and direct CLI."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import memory_benefit_audit_8212 as e


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-VERIFY-8212: natural primitives are reduced only into private storage."""
    raw = tmp_path_factory.mktemp("audit-8212")
    preflight = raw / "preflight"
    preflight.mkdir()
    atomic_json(preflight / "probe.json", dict(private_writable=True))
    os.environ.setdefault("CARNOT_8212_PREFLIGHT", str(preflight))
    return e.measure(e.ROOT, raw), raw


def test_natural_null_and_failed_owned_validation(work):
    """SCENARIO-VERIFY-8212-REDUCTION: readiness and useful decisions differ."""
    measured, raw = work
    value = e.build(measured, raw, [dict(passed=True)], fixture=True)
    assert value["input_ready"] == value["learning_audit_ready_score"] == 1
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert value["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["intended_count"] == 192
    assert value["completed_count"] == value["independent_count"] == 158
    assert value["completed_count"] + value["excluded_count"] + value["censored_count"] == 192
    assert value["H2"]["paired_gain_interval"]["draws"] == 10000
    assert value["H2"]["paired_gain_interval"]["valid_draws"] >= 9500
    assert value["verdict_class"] == "null" and not value["h2_development_signal_score"]
    assert (
        value["causal_checks"]["restart_equal"]
        and value["causal_checks"]["future_mutation_invariant"]
    )
    assert value["structural_memory_effect"]["action_movement"] >= 0
    assert value["calibration_only_effect"]["probability_movement"] > 0
    assert value["activation_retention_correlation"]["diagnostic_only"]
    failed = e.build(measured, raw, [dict(passed=False)])
    assert failed["verdict_class"] == "disqualified" and not failed["learning_audit_ready_score"]


def test_external_blocks_and_owned_faults(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8212-REDUCTION: absent and zero are distinct operands."""
    for observed in ["absent", None, 0]:
        root = tmp_path / str(observed)
        if observed != "absent":
            value = json.loads((e.ROOT / e.UPSTREAM).read_bytes())
            value.pop("learning_trajectory_ready_score")
            if observed == 0:
                value["learning_trajectory_ready_score"] = 0
            atomic_json(root / e.UPSTREAM, value)
        measured = e.measure(root, root / "raw", fixture=True)
        result = e.build(measured, root / "raw", [dict(passed=True)])
        assert result["verdict_class"] == "blocked" and not result["learning_audit_ready_score"]
        check = next(r for r in result["gate_check_summary"] if not r["passed"])
        if observed != "absent":
            assert check["artifact_field"] == "learning_trajectory_ready_score"
            assert check["observed"] == observed
    monkeypatch.setattr(e, "reconstruct", lambda *args: (_ for _ in ()).throw(ValueError("fault")))
    measured = e.measure(e.ROOT, tmp_path / "owned-failure")
    assert e.build(measured, tmp_path, [dict(passed=True)])["verdict_class"] == "disqualified"


def test_statistics_controls_missingness_and_seed_credit(work):
    """SCENARIO-VERIFY-8212-REDUCTION: the frozen thresholds never relax."""
    rows = e.legacy.control_rows(improved=True)
    rows += [dict(r, arm="calibration_only") for r in rows if r["arm"] == "fixed_public_center"]
    result = e.statistics(rows)
    assert result["h2_passed"] and result["retention_passed"]
    assert (
        e.statistics(rows + [dict(r, seed=r["seed"] + 20) for r in rows])["completed_count"] == 192
    )
    missing = deepcopy(rows)
    for row in missing:
        if row["slot"] % 2 == 0:
            row["status"] = "excluded"
    assert not e.statistics(missing)["support_sufficient"]
    unsafe = deepcopy(rows)
    for row in unsafe:
        if (
            row["condition"] == "retention"
            and row["arm"] == "error_center"
            and row["metric"] == "typed_cost"
        ):
            row["numerator"] = 10
    assert not e.statistics(unsafe)["retention_passed"]
    duplicate = deepcopy(rows)
    duplicate[0]["source_cluster_id"] = duplicate[20]["source_cluster_id"]
    with pytest.raises(ValueError):
        e.statistics(duplicate)
    measured, _ = work
    assert not e.statistics(measured["legacy_rows"])["h2_passed"]


def test_replay_rejects_rehashed_changes(work, tmp_path):
    """SCENARIO-VERIFY-8212-REDUCTION: rehashing cannot certify changed rows."""
    measured, raw = work
    value = e.build(measured, raw, [dict(passed=True)])
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    for field in [
        "H2",
        "rows",
        "learning_audit_ready_score",
        "calibration_only_effect",
        "experiment_id",
    ]:
        bad = deepcopy(value)
        bad[field] = "changed"
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)
    assert not e.replay(tmp_path / "absent")
    bad = dict(value, reproducibility_checksum="wrong")
    atomic_json(path, bad)
    assert not e.replay(path)


def test_manifest_and_real_cli(work, tmp_path):
    """SCENARIO-REPORT-8212-CLI: child statements execute outside ambient imports."""
    from carnot.reporting import memory_benefit_execution_8212 as runner

    measured, raw = work
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, e.build(measured, raw, [dict(passed=True)]))
    specs = runner.manifest(tmp_path, path)
    assert specs["repository_health"]["argv"][-2:] == ["tests/python", "-q"]
    assert "--files" in next(r for r in specs["commands"] if r["name"] == "spec_coverage")["argv"]
    cli = ["/usr/bin/env", "-u", "PYTHONPATH"]
    if os.environ.get("COVERAGE_RCFILE"):
        cli.append("COVERAGE_PROCESS_START=" + os.environ["COVERAGE_RCFILE"])
    cli.extend([str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)])
    for name, args, expected in [
        ("replay", ["--cold-replay", str(path)], 0),
        ("private", ["--fixture-output", str(tmp_path / "private" / path.name)], 0),
        (
            "blocked",
            [
                "--root",
                str(tmp_path / "absent"),
                "--fixture-output",
                str(tmp_path / "blocked" / path.name),
            ],
            0,
        ),
        (
            "worker",
            [
                "--root",
                str(tmp_path / "absent"),
                "--worker-output",
                str(tmp_path / "worker/measurement.json"),
            ],
            0,
        ),
        ("date", ["--date", "20261005"], 2),
    ]:
        receipt = e.run_check(
            e.ROOT,
            dict(name=name, argv=cli + args, expected_exit=expected, deadline_s=180),
            tmp_path,
            tmp_path / "logs",
        )
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(path.read_text())
    value["completed_count"] = 0
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    receipt = e.run_check(
        e.ROOT,
        dict(
            name="tamper", argv=cli + ["--cold-replay", str(path)], expected_exit=1, deadline_s=180
        ),
        tmp_path,
        tmp_path / "logs",
    )
    assert receipt["passed"]


def test_receipt_hash_and_upstream_runtime_rejection(work, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8212-REDUCTION: failed runtime checks cannot earn credit."""
    measured, raw = work
    path = tmp_path / (e.NAME + ".json")
    stream = tmp_path / "stdout"
    stream.write_text("real child output")
    value = e.build(
        measured, raw, [dict(passed=True, stdout_path=str(stream), stdout_sha256="wrong")]
    )
    atomic_json(path, value)
    assert not e.replay(path)
    atomic_json(path, e.build(measured, raw, [dict(passed=True)]))
    monkeypatch.setattr(e.upstream, "replay", lambda path: False)
    assert not e.replay(path)
