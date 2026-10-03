"""REQ-REPORT-8077: private primitive, causal and publication checks."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from carnot import experiment_8077_v699_projected_learning_audit as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import projected_learning_audit_8077 as a
from carnot.verify import projected_online_8076 as producer
from test_fresh_feedback_8064 import data


def test_independent_reconstruction(tmp_path, monkeypatch):
    """REQ-REPORT-8077: no producer transition or aggregate enters cold replay."""
    value = data()
    original = producer.measure(value, tmp_path / "trajectory")
    monkeypatch.setattr(producer, "reduce", lambda *_: pytest.fail("producer reduction"))
    monkeypatch.setattr(producer, "trajectory", lambda *_: pytest.fail("producer transition"))
    rebuilt = a.reconstruct(tmp_path / "trajectory", value["labels"])
    for key in producer.FIELDS.values():
        assert rebuilt[key] == original[key]
    assert len(rebuilt["rows"]) == 1024
    assert rebuilt["later_source_rows"]
    assert rebuilt["reconstructed_projection_residuals"]
    mutated = deepcopy(value["labels"])
    mutated["110"] ^= 1
    with pytest.raises(ValueError):
        a.reconstruct(tmp_path / "trajectory", mutated)


def test_h2_retention_and_mask():
    """SCENARIO-REPORT-8077-H2/RETENTION: paired seeds never enlarge support."""
    rows = []
    for slot in range(100):
        for seed in (101, 102):
            for arm in a.ARMS:
                rows.append(
                    dict(
                        source=str(slot),
                        slot=slot,
                        seed=seed,
                        arm=arm,
                        y=slot % 2,
                        denominator=1,
                        numerator=0.5 if arm == "ray_fresh" else 0,
                        brier=0.0,
                        action="escalate" if arm == "ray_fresh" else "reject",
                    )
                )
    retained = deepcopy(rows[: 64 * 8])
    for row in retained:
        row["numerator"] = 0.0
    result = a.comparisons(rows, retained)
    assert result["H2"]["support_count"] == 100
    assert result["H2"]["qualified_benefit"]
    assert result["H2"]["tests"][0]["slot_count"] == 256
    for row in retained:
        if row["arm"] == "ray_fresh":
            row["numerator"] = 0.6
    assert not a.comparisons(rows, retained)["H2"]["qualified_benefit"]
    assert a.comparisons([], [])["H2"]["support_passed"] is False
    assert a.bootstrap([float("nan")] * 256, 32)["censored_draws"] == 10000


def cli(private, *args):
    """SCENARIO-REPORT-8077-TERMINAL: real imports work outside checkout."""
    command = [str(e.ROOT / ".venv/bin/python")]
    config = os.environ.get("CARNOT_8076_COVERAGE_CONFIG")
    if config:
        command += ["-m", "coverage", "run", "--rcfile=" + config, "--parallel-mode"]
    command += [str(e.ROOT / e.CLI), *map(str, args)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    print("8077 subprocess before", command, flush=True)
    result = subprocess.run(
        command, cwd=private, env=env, capture_output=True, text=True, timeout=180
    )
    print(
        "8077 subprocess after",
        result.returncode,
        result.stdout[-1500:],
        result.stderr[-1500:],
        flush=True,
    )
    return result


def test_private_cli_success_blocked_mutation_replay(tmp_path):
    """SCENARIO-REPORT-8077-TERMINAL: exact oracle and external absences differ."""
    source, output = tmp_path / "input.json", tmp_path / (e.NAME + ".json")
    atomic_json(source, data())
    assert cli(tmp_path, "--fixture-input", source, "--fixture-output", output).returncode == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["verifier_is_oracle"] and value["generalized_learning_benefit_score"] == 0
    assert len(value["recovery_rows"]) == 10
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    for key in ("raw_shard_hashes", "code_config_hashes"):
        changed = deepcopy(value)
        if key == "raw_shard_hashes":
            changed[key][0]["sha256"] = "sha256:forged"
        else:
            changed[key][e.MODULE] = "sha256:forged"
        atomic_json(tmp_path / "hash_mutant.json", changed)
        assert e.main(["--cold-replay", str(tmp_path / "hash_mutant.json")]) == 1
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    validation_path = raw / "validation.json"
    validation = json.loads(validation_path.read_text())
    changed = deepcopy(validation)
    changed["receipts"][0]["log_sha256"] = "sha256:forged"
    atomic_json(validation_path, changed)
    assert e.main(["--cold-replay", str(output)]) == 1
    atomic_json(validation_path, validation)
    work_path = raw / "work.json"
    work = json.loads(work_path.read_text())
    changed = deepcopy(work)
    changed["evidence"]["H2"]["support_count"] += 1
    atomic_json(work_path, changed)
    assert e.main(["--cold-replay", str(output)]) == 1
    atomic_json(work_path, work)
    changed = deepcopy(value)
    changed["H2"]["support_count"] += 1
    atomic_json(tmp_path / "mutant.json", changed)
    assert cli(tmp_path, "--cold-replay", tmp_path / "mutant.json").returncode == 1
    assert cli(tmp_path, "--fixture-input", source, "--fixture-output", output).returncode == 1
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert cli(tmp_path, "--root", tmp_path, "--fixture-output", blocked).returncode == 0
    missing = json.loads(blocked.read_text())
    assert missing["verdict_class"] == "blocked" and missing["gate_check_summary"]
    assert cli(tmp_path, "--cold-replay", blocked).returncode == 0


def test_build_owned_failure(tmp_path):
    """SCENARIO-REPORT-8077-TERMINAL: failed owned checks cannot earn readiness."""
    work = dict(
        data={},
        evidence={},
        failures=[],
        owned_failure=True,
        source_artifact_hashes=[],
        raw_shard_hashes=[],
        code_config_hashes={},
        phase_spans=[],
        duration_s=0,
    )
    result = e.build(work, tmp_path, [], {}, fixture=False)
    assert result["verdict_class"] == "disqualified"
    assert result["learning_audit_ready_score"] == 0
    assert not e.replay(tmp_path / "missing.json")


def test_projection_fallback(tmp_path):
    """SCENARIO-REPORT-8077-RECOVERY: budget exhaustion validates fallback separately."""
    from carnot.verify import constraint_projection_8075 as kernel

    rows = [dict(source_id="x", normal=[1.0, 0.0], rhs=0.0)]
    feasible = kernel.project([0.3, 1], rows, [0, 1], [0, 1], seed=3, frozen_last=True)
    assert a.projection(feasible, rows, np.asarray([0, 1.0]), np.asarray([0, 1.0]), 3)["feasible"]
    for previous, expected in [([0.2, 1], "incumbent"), ([-0.2, 1], "initial")]:
        observed = kernel.project(
            [-0.3, 1], rows, [0, 1], previous, seed=3, budget=0, frozen_last=True
        )
        rebuilt = a.projection(
            observed, rows, np.asarray([0, 1.0]), np.asarray(previous), 3, budget=0
        )
        assert rebuilt["fallback"] == expected
        observed["point"][0] = -1
        with pytest.raises(ValueError):
            a.projection(observed, rows, np.asarray([0, 1.0]), np.asarray(previous), 3, budget=0)


def test_independent_admission_reset_and_harm():
    """SCENARIO-REPORT-8077-RETENTION: keeping/resetting a head is a null control."""
    head = dict(parameters=[0.0], calibration=[0.0, 1.0])
    x = np.tile([-1.0, 1.0], 6).reshape(12, 1)
    y = np.tile([0, 1], 6)
    zero, good = np.zeros(1), np.ones(1)
    constraints = [dict(normal=[1.0, 0.0], rhs=0.0)]
    assert a.admission(head, zero, good, zero, x, y, [], False)["alpha"] == 1
    assert a.admission(head, zero, -20 * good, zero, x, y, [], False)["alpha"] == 0
    result = a.admission(head, -20 * good, -30 * good, zero, x, y, constraints, True)
    assert result["fallback"] == "initial" and result["parameters"] == [0.0]
    assert a.admission(head, zero, good, zero, x, np.zeros(12), [], True)["fallback"] == "incumbent"


def test_build_supported_positive_and_unsupported(tmp_path, monkeypatch):
    """REQ-REPORT-8077: support blocks, safety failures stay null, oracle cannot earn benefit."""
    monkeypatch.setattr(e.driver, "OWNED", e.OWNED)
    atomic_json(
        tmp_path / "validation_commands.json",
        dict(commands=[dict(name="measurement_normal_exit", classification="required")]),
    )
    h = dict(
        support_count=100,
        retention_support_count=64,
        support_passed=True,
        safety_passed=True,
        qualified_benefit=True,
        beneficial_changed_sources=10,
        tests=[dict(gain=0.2, raw_p=0.001)],
        retention_checks=[dict(arm="ray_fresh", brier_drift=0.0, cost_drift=0.0)],
    )
    work = dict(
        data=dict(seeds=[101]),
        evidence=dict(H2=h, final_head_seals=[], rows=[]),
        failures=[],
        owned_failure=False,
        source_artifact_hashes=[],
        raw_shard_hashes=[],
        code_config_hashes={},
        phase_spans=[],
        duration_s=1,
    )
    receipts = [dict(name="measurement_normal_exit", passed=True)]
    coverage = {p: dict(missing_lines=0) for p in e.OWNED}
    assert (
        e.build(work, tmp_path, receipts, coverage, fixture=False)[
            "projected_learning_benefit_score"
        ]
        == 1
    )
    assert (
        e.build(work, tmp_path, receipts, coverage, fixture=True)[
            "projected_learning_benefit_score"
        ]
        == 0
    )
    h.update(support_passed=False, qualified_benefit=False)
    assert e.build(work, tmp_path, receipts, coverage, fixture=False)["verdict_class"] == "blocked"
    h.update(support_passed=True, safety_passed=False)
    h["retention_checks"][0]["brier_drift"] = 0.2
    assert e.build(work, tmp_path, receipts, coverage, fixture=False)["verdict_class"] == "null"


def test_recovery_deadline(tmp_path):
    """SCENARIO-REPORT-8077-RECOVERY: unfinished recovery cannot earn readiness."""
    with pytest.raises(TimeoutError, match="recovery_budget"):
        e.recovery(data(), tmp_path, budget_s=-1)


def test_missing_projection_fallback_fails_recovery(tmp_path):
    """SCENARIO-REPORT-8077-RECOVERY: an unvisited crash boundary cannot pass."""
    value = data()
    for row in value["sources"]:
        row["features"] = [0.3] * 8
    rows = e.recovery(value, tmp_path)
    assert len(rows) == 10
    assert not all(r["passed"] for r in rows)
    assert all(r["projection_control"] == "force" for r in rows)


def test_deferred_censored_and_one_class(tmp_path):
    """REQ-REPORT-8077: excluded and admission masks remain in the original timeline."""
    value = data()
    value["sources"][0]["public_eligible"] = False
    value["labels"]["1"] = None
    for row in value["sources"]:
        if a.prior.role(row) == "admission":
            value["labels"][row["family_id"]] = 0
    producer.measure(value, tmp_path / "oneclass")
    result = a.reconstruct(tmp_path / "oneclass", value["labels"])
    assert all(r["alpha"] is None for r in result["durable_commit_rows"] if r["arm"] in a.ARMS[2:])
    with pytest.raises(TimeoutError):
        a.reconstruct(tmp_path / "oneclass", value["labels"], budget_s=-1)
    for row in value["sources"]:
        row["eligible"] = False
    producer.measure(value, tmp_path / "empty")
    assert a.reconstruct(tmp_path / "empty", value["labels"])["pending_update_rows"]
    value = data()
    for row in value["sources"]:
        if row["slot"] >= 65 and a.prior.role(row) == "admission":
            row["eligible"] = False
    producer.measure(value, tmp_path / "censored")
    rebuilt = a.reconstruct(tmp_path / "censored", value["labels"])
    assert any(r["status"] == "censored" for r in rebuilt["pending_update_rows"])


def test_external_worker_and_owned_label_rejection(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8077-TERMINAL: exact external operands and leaked labels differ."""
    value = data()
    trajectory = tmp_path / "upstream"
    producer.measure(value, trajectory)
    parent = dict(
        experiment_id=8076,
        task_id="exp8076-projected-online-learning",
        learning_trajectory_ready_score=1,
        required_checks_passed=True,
        verifier_is_oracle=False,
        raw_shard_hashes=[],
        source_artifact_hashes=[],
        trajectory_directory=str(trajectory),
    )
    primary = tmp_path / "results/experiment_8076_v699_projected_online_learning.json"
    atomic_json(primary, parent)
    monkeypatch.setattr(e.driver, "prerequisites", lambda *_: ([], []))
    monkeypatch.setattr(e.historical, "load_inputs", lambda *_: (deepcopy(value), []))
    monkeypatch.setattr(e, "measure", lambda *_: dict(H2={"support_count": 1}))
    assert e.worker(tmp_path, tmp_path / "success")["evidence"]
    parent["learning_trajectory_ready_score"] = 0
    parent["raw_shard_hashes"] = [dict(path=str(tmp_path / "absent"), sha256="sha256:absent")]
    atomic_json(primary, parent)
    assert len(e.worker(tmp_path, tmp_path / "blocked")["failures"]) == 2

    def fail(*_):
        raise ValueError("label leak")

    monkeypatch.setattr(e, "measure", fail)
    source = tmp_path / "fixture.json"
    atomic_json(source, value)
    assert e.worker(tmp_path, tmp_path / "owned", source)["owned_failure"]


def test_public_label_leak(tmp_path):
    """REQ-REPORT-8077: public label leaks are rejected before scientific scoring."""
    value = data()
    producer.measure(value, tmp_path / "upstream")
    value["trajectory"] = str(tmp_path / "upstream")
    value["retention"][0]["y"] = 0
    with pytest.raises(ValueError, match="public_label_fields"):
        e.measure(value, tmp_path / "audit")


def test_public_feature_rebuild(tmp_path, monkeypatch):
    """REQ-REPORT-8077: literal source/answer bytes independently determine public features."""
    value = data()
    row = value["sources"][0]
    row.update(
        source_bytes=b"A public source contains one fact.".hex(), answer_bytes=b"One fact.".hex()
    )
    row["features"] = e.features.extract(
        {k: row[k] for k in ("source_bytes", "answer_bytes", "family_id")}
    )["values"]
    producer.measure(value, tmp_path / "upstream")
    value["trajectory"] = str(tmp_path / "upstream")
    monkeypatch.setattr(e, "recovery", lambda *_args, **_kwargs: [dict(passed=True)])
    assert e.measure(value, tmp_path / "audit")["H2"]
