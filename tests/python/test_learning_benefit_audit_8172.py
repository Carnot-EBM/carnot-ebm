"""REQ-VERIFY-8172 / REQ-REPORT-8172: private evidence qualifies the reader."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import admission_horizon_methods_8152 as schedule
from carnot.verify import learning_audit_8144 as historical
from carnot.verify import learning_benefit_audit_8172 as audit


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    """SCENARIO-VERIFY-8172-E2E016: known labels stay in private temporary files."""
    raw = tmp_path_factory.mktemp("benefit8172-input")
    rows, targets = schedule.fixture("positive")
    retained, retention_targets = schedule.engine.fixture(64, "retention")
    features, labels = {}, {}
    for role, public, truth in [
        ("stream", rows, targets),
        ("retention", retained, retention_targets),
    ]:
        features[role] = schedule.engine.shard(raw, dict(rows=public))
        labels[role] = schedule.engine.shard(
            raw, dict(rows=[dict(r, y=y) for r, y in zip(public, truth, strict=True)])
        )
    states = [
        dict(seed=s, state=schedule.engine.shard(raw, schedule.run(rows, targets, s)))
        for s in [101, 102]
    ]
    value = dict(
        experiment_id=8171,
        learning_trajectory_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        input_manifests=dict(
            stream_feature_manifest=features["stream"],
            retention_feature_manifest=features["retention"],
            evaluator_label_manifests=labels,
        ),
        state_manifest=states,
    )
    root = raw / "root"
    atomic_json(root / audit.UPSTREAM, value)
    return root, rows, value


def test_source_gates_and_masks():
    """REQ-VERIFY-8172: source counts and frozen-control safety survive repetition."""
    rows = historical.control_rows()
    result = audit.statistics(rows)
    assert result["h2_passed"] and result["retention_passed"]
    assert result["completed_count"] == 192
    assert result["paired_gain_interval"]["valid_draws"] == 10000
    assert set(result["other_control_cost_increases"]) == set(audit.ARMS) - {"error_center"}
    duplicate = [dict(r, seed=r["seed"] + 100) for r in rows]
    assert audit.statistics(rows + duplicate)["completed_count"] == 192
    assert audit.statistics(historical.control_rows(improved=False))["h2_passed"] is False
    for row in rows:
        if row["condition"] == "later_stream" and row["slot"] == 70:
            row.update(status="excluded", exclusion_reason="missing", numerator=0)
    masked = audit.statistics(rows)
    assert masked["completed_count"] == 191
    assert masked["per_source_results"][5]["gain"] is None
    assert audit.statistics([])["h2_passed"] is False
    assert [i["block_length"] for i in result["paired_intervals"]] == [16, 8, 32]


@pytest.mark.parametrize("mutation", ["future", "reuse", "mask", "prediction"])
def test_causal_mutations(panel, mutation):
    """SCENARIO-VERIFY-8172-E2E016: producer fields cannot override scalar events."""
    _, rows, value = panel
    rows = deepcopy(rows)
    state = json.loads(Path(value["state_manifest"][0]["state"]["path"]).read_text())
    if mutation == "future":
        rows[0]["y"] = 1
    elif mutation == "reuse":
        state["used_admission"].append(state["used_admission"][0])
    elif mutation == "mask":
        rows.pop(70)
    else:
        state["issued"][149]["predictions"]["error_center"] += 0.1
    labels = Path(value["input_manifests"]["evaluator_label_manifests"]["stream"]["path"])
    with pytest.raises((ValueError, KeyError)):
        audit.reconstruct(state, rows, labels)


@pytest.fixture(scope="module")
def measured(panel, tmp_path_factory):
    """REQ-VERIFY-8172: all retained predictions seal before target decoding."""
    raw = tmp_path_factory.mktemp("benefit8172-measured")
    return audit.measure(panel[0], raw, fixture=True), raw


def test_measure_build_and_replay(measured):
    """SCENARIO-REPORT-8172-CLI: independently rebuilt headlines bind evidence."""
    work, raw = measured
    assert "owned_reconstruction_error" not in work
    value = audit.build(work, raw, [dict(passed=True, normal_exit=True)])
    assert value["learning_audit_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["h2_development_signal_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert value["independent_generalization_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["call_ledger"] == []
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert value["installation_exposure_summary"]["installed_head_count"] > 0
    path = raw / (audit.NAME + ".json")
    atomic_json(path, value)
    assert audit.replay(path)
    assert not audit.replay(raw / "absent.json")
    for key in ["completed_count", "experiment_id", "rows", "installation_exposure_summary"]:
        bad = deepcopy(value)
        if key == "rows":
            bad[key][0]["numerator"] += 0.1
        elif key == "installation_exposure_summary":
            bad[key]["installed_head_count"] += 1
        else:
            bad[key] += 1
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not audit.replay(path)
    atomic_json(path, value)
    failed = audit.build(work, raw, [dict(passed=True, normal_exit=False)])
    assert failed["verdict_class"] == "disqualified"
    assert failed["learning_audit_ready_score"] == 0


def test_external_block(tmp_path, panel):
    """REQ-REPORT-8172: unchanged external failure terminates with exact operands."""
    work = audit.measure(tmp_path, tmp_path / "missing")
    value = audit.build(work, tmp_path / "missing", [dict(passed=True)])
    assert value["verdict_class"] == "blocked"
    failed = next(r for r in value["gate_check_summary"] if not r["passed"])
    assert failed["check"] == "resource_exists" and failed["observed"] is False
    assert audit.build(work, tmp_path, [dict(passed=False)])["verdict_class"] == "disqualified"
    original = deepcopy(panel[2])
    for kind in ["gate", "seed"]:
        bad = deepcopy(original)
        if kind == "gate":
            bad["learning_trajectory_ready_score"] = 0
        else:
            bad["state_manifest"].append(bad["state_manifest"][0])
        root = tmp_path / kind
        atomic_json(root / audit.UPSTREAM, bad)
        measured = audit.measure(root, root / "raw", fixture=True)
        blocked = audit.build(measured, root / "raw", [dict(passed=True)])
        assert blocked["verdict_class"] == "blocked"
        assert blocked["learning_audit_ready_score"] == 0


def test_private_direct_cli(panel, tmp_path):
    """SCENARIO-REPORT-8172-CLI: actual children work without checkout PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    config = env.get("COVERAGE_RCFILE")
    if config:
        env["COVERAGE_PROCESS_START"] = config
    cli = [str(audit.ROOT / ".venv/bin/python"), "-u", str(audit.ROOT / audit.CLI)]

    def run(args, expected):
        print("[test8172] before_subprocess", args, flush=True)
        child = subprocess.run(
            cli + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )
        print("[test8172] after_subprocess", child.returncode, flush=True)
        assert child.returncode == expected, child.stdout + child.stderr
        return child

    output = tmp_path / (audit.NAME + ".json")
    run(["--fixture-output", str(output), "--root", str(panel[0])], 0)
    run(["--cold-replay", str(output)], 0)
    value = json.loads(output.read_text())
    assert value["learning_audit_ready_score"] == 1
    value["completed_count"] += 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    run(["--cold-replay", str(output)], 1)
    blocked = tmp_path / "blocked" / (audit.NAME + ".json")
    run(["--fixture-output", str(blocked), "--root", str(tmp_path / "absent")], 0)
    run(["--cold-replay", str(blocked)], 0)
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    run(["--date", "wrong"], 2)
    run(["--fixture-output", str(audit.ROOT / "results" / "private.json")], 2)


def test_frozen_manifest_and_main(tmp_path, monkeypatch):
    """REQ-REPORT-8172: only pytest receives test selectors; other tools use paths."""
    from carnot.reporting import learning_benefit_execution_8172 as runner

    specs = runner.manifest(tmp_path, tmp_path / "candidate.json")
    for spec in specs["commands"]:
        if spec["name"] in ["ruff_check", "ruff_format", "strict_mypy", "spec_coverage"]:
            assert all("::" not in arg for arg in spec["argv"])
    assert str(audit.ROOT / audit.CLI) in specs["terminal_commands"][0]["argv"]
    monkeypatch.setattr(runner.previous, "main", lambda argv: 23)
    assert runner.main([]) == 23


def test_unchanged_protocol_and_rejected_heads(tmp_path):
    """REQ-VERIFY-8172: immutable parameters and missing public slots stay intact."""
    binder = audit.Custody(tmp_path / "custody")
    binder.require(tmp_path, "numerical_protocol", {}, schedule.protocol())
    assert binder.checks[-1]["passed"]
    for case in ["positive", "rejected"]:
        rows, targets = schedule.fixture(case)
        rows[180]["values"] = None
        state = schedule.run(rows, targets, 101)
        path = tmp_path / (case + ".json")
        atomic_json(path, dict(rows=[dict(r, y=y) for r, y in zip(rows, targets, strict=True)]))
        result = audit.reconstruct(state, rows, path)
        assert result["passed"]
        assert result["overflow_count"] == result["admission_reuse_count"] == 0
        assert result["pending"] == list(range(237, 257))
