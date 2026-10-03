"""REQ-REPORT-8056: terminal accounting must not invent scientific success."""

import copy
import gzip
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v697_capstone as cap
from carnot.reporting import v697_capstone_reduction as red
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def fixture(root):
    """SCENARIO-REPORT-8056-CUSTODY: preserve a private immutable invocation."""
    snapshots = {}
    for role, name in (("active", "active.yaml"), ("design", "design.md")):
        p = root / name
        p.write_bytes(
            gzip.decompress((cap.ROOT / "tests/fixtures/v697" / (name + ".gz")).read_bytes())
        )
        snapshots[role] = dict(exists=True, snapshot_path=str(p), sha256=sha256_file(p))
    tasks = yaml.safe_load((root / "active.yaml").read_bytes())["tasks"]
    atomic_json(
        root / cap.INPUT,
        dict(
            authority_snapshots=snapshots,
            canonical_tasks_sha256=cap.lifecycle.tasks_digest(tasks),
            method_freeze={},
            experiment_id=8044,
            task_id=tasks[0]["id"],
            honest_verdict="complete_null_private",
            verdict_class="null",
            flagged_adversarial=False,
        ),
    )
    return tasks


def test_authorities_and_absence(tmp_path):
    """SCENARIO-REPORT-8056-CUSTODY: prompts and missing resources are operands."""
    tasks = fixture(tmp_path)
    assert cap.authorities(tmp_path)[0] == tasks
    p = tmp_path / cap.INPUT
    v = json.loads(p.read_text())
    v["canonical_tasks_sha256"] = "wrong"
    atomic_json(p, v)
    with pytest.raises(ValueError, match="authority"):
        cap.authorities(tmp_path)
    assert any(not r["passed"] for r in cap.preconditions(tmp_path))


def test_statistics_and_unsafe_credit():
    """SCENARIO-REPORT-8056-SCIENCE: margin failures and missing tests keep p=1."""
    p = [
        red.bootstrap([0.009] * 80, 0.01),
        red.bootstrap([0.03] * 80, 0.02),
        red.bootstrap([], 0.02),
    ]
    h = red.family(p, [True, True, False])
    assert not h[0]["positive_claim"] and h[1]["positive_claim"] and h[2]["family_p"] == 1
    assert not any(r["positive_claim"] for r in red.family(p, [False] * 3))
    assert red.independent({}, 8050)["measurement_available"] is False
    assert red.independent({}, 8052)["measurement_available"] is False


def test_blocked_build_and_seal(tmp_path):
    """SCENARIO-REPORT-8056-TERMINAL: a correct reader may report blocked science."""
    fixture(tmp_path)
    v = cap.build(tmp_path, "20261003", tmp_path / "raw")
    assert len(v["rows"]) == 13 and not v["rows"][-1]["completed"]
    assert v["verdict_class"] == "blocked" and all(
        h["family_p"] == 1 for h in v["primary_hypothesis_results"]
    )
    counts = {p: dict(num_statements=1, covered_lines=1) for p in cap.OWNED}
    cap.complete(v, [dict(passed=True, classification="required")], counts)
    assert v["capstone_execution_ready_score"] == 1 and v["completed_count"] == 13
    assert v["science_ready"] is False and v["generalized_learning_benefit_score"] == 0
    cap.seal(v, tmp_path / "raw")
    assert cap.cold_replay(v) == []
    changed = copy.deepcopy(v)
    changed["science_ready"] = True
    assert cap.cold_replay(changed)
    changed = copy.deepcopy(v)
    changed["independent_reduction_sha256"] = "wrong"
    assert "independent_reduction_drift" in cap.cold_replay(changed)
    bad = copy.deepcopy(v)
    cap.complete(bad, [dict(passed=False)], counts)
    assert bad["verdict_class"] == "disqualified"
    Path(v["checkpoint_references"][0]["path"]).write_text("{}")
    assert cap.cold_replay(v) == ["source_bytes_changed"]


def test_independent_equations(tmp_path):
    """SCENARIO-REPORT-8056-SCIENCE: rebuild causal heads and every retention operand."""
    from test_feedback_constrained_8051 import data
    from carnot.verify import feedback_constrained_8051 as producer

    inputs = data(96)
    raw = tmp_path / "trajectory"
    producer.measure(inputs, raw)
    targets = tmp_path / "targets.json"
    atomic_json(
        targets, dict(rows=[dict(family_id=k, eligible_y=v) for k, v in inputs["labels"].items()])
    )
    bundle = dict(
        trajectory=str(raw),
        target_reference=dict(path=str(targets), sha256=sha256_file(targets)),
        retention_public=[],
        retention_target=dict(path=str(targets), sha256=sha256_file(targets)),
        historical_exposure={},
    )
    replay = red.learning.replay(raw, inputs["labels"])
    # A private empty retention vault is a valid insufficient-support control.
    empty = tmp_path / "empty.json"
    atomic_json(empty, dict(rows=[]))
    bundle["retention_target"] = dict(path=str(empty), sha256=sha256_file(empty))
    seal = tmp_path / "seal.json"
    retained = red.learning.retention(bundle, replay, seal)
    compared = red.learning.compare(replay["rows"], retained)
    bp = tmp_path / "bundle.json"
    atomic_json(bp, bundle)
    v = dict(
        replay,
        retention_rows=retained,
        **compared,
        audit_bundle=dict(path=str(bp), sha256=sha256_file(bp)),
        retention_prediction_seal=dict(path=str(seal), sha256=sha256_file(seal)),
    )
    r = red.independent(v, 8052)
    assert r["measurement_available"] and r["primary"]["slot_count"] == 256
    assert not r["scientific_qualified"] and r["per_seed_false_accept_rows"]
    v["rows"][0]["probability"] = 0.99
    with pytest.raises(ValueError, match="producer_drift"):
        red.independent(v, 8052)


def test_cli_missing_and_date(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8056-TERMINAL: actual entrypoint returns normal negative exits."""
    monkeypatch.setattr(sys, "argv", [cap.CLI, "--cold-replay", str(tmp_path / "missing")])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(cap.ROOT / cap.CLI), run_name="__main__")
    assert result.value.code == 1
    with pytest.raises(ValueError, match="date"):
        cap.main(["--date", "20261002"])


def test_real_cold_routes(tmp_path):
    """SCENARIO-REPORT-8056-TERMINAL: real blocked/null/valid/tampered replay processes."""
    fixture(tmp_path)
    v = cap.build(tmp_path, "20261003", tmp_path / "raw")
    counts = {p: dict(num_statements=1, covered_lines=1) for p in cap.OWNED}
    cap.complete(v, [dict(passed=True)], counts)
    for state in ("blocked", "null"):
        v.update(verdict_class=state, honest_verdict="complete_" + state + "_private")
        cap.seal(v, tmp_path / state)
        path = tmp_path / (state + ".json")
        atomic_json(path, v)
        env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
        argv = [
            str(cap.ROOT / ".venv/bin/python"),
            "-u",
            str(cap.ROOT / cap.CLI),
            "--cold-replay",
            str(path),
        ]
        result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        bad = copy.deepcopy(v)
        bad["primary_hypothesis_results"][0]["family_p"] = 0
        atomic_json(path, bad)
        assert subprocess.run(argv, env=env, capture_output=True, timeout=30).returncode == 1


def test_resource_and_code_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8056-CUSTODY: unknown authority and stale code fail closed."""
    missing = cap.build(tmp_path, "20261003", tmp_path / "missing_raw")
    assert missing["honest_verdict"] == "complete_blocked_immutable_authority"
    assert missing["gate_check_summary"]
    fixture(tmp_path)
    v = cap.build(tmp_path, "20261003", tmp_path / "raw")
    toy = tmp_path / "toy.py"
    toy.write_text("x=1\n")
    ref = cap.save(dict(path=str(toy), role="current_code"), tmp_path / "raw")
    v["code_config_hashes"].append(ref)
    cap.seal(v, tmp_path / "raw")
    toy.write_text("x=2\n")
    assert cap.cold_replay(v) == ["code_configuration_changed"]
    huge = tmp_path / "huge"
    with huge.open("wb") as stream:
        stream.truncate(90_000_000)
    sharded = cap.save(dict(path=str(huge)), tmp_path / "raw")
    manifest = json.loads(Path(sharded["path"]).read_text())
    assert manifest["original_sha256"] == sha256_file(huge)
    assert sum(Path(r["path"]).stat().st_size for r in manifest["shards"]) == 90_000_000
    assert all(Path(r["path"]).stat().st_size < 90_000_000 for r in manifest["shards"])
    v["source_artifact_hashes"].append(sharded)
    v["code_config_hashes"].pop()
    cap.seal(v, tmp_path / "raw")
    assert cap.cold_replay(v) == []
    original_manifest = copy.deepcopy(manifest)
    manifest["original_sha256"] = "wrong"
    atomic_json(Path(sharded["path"]), manifest)
    sharded["sha256"] = sha256_file(Path(sharded["path"]))
    assert cap.cold_replay(v) == ["source_bytes_changed"]
    atomic_json(Path(sharded["path"]), original_manifest)
    sharded["sha256"] = sha256_file(Path(sharded["path"]))
    Path(manifest["shards"][0]["path"]).write_bytes(b"tampered")
    assert cap.cold_replay(v) == ["source_bytes_changed"]
    row = dict(
        task_id="exp8052-learning-benefit-audit",
        path=str(tmp_path / cap.INPUT),
        gate_check_summary=[],
    )
    monkeypatch.setattr(cap.previous, "collect", lambda *a: ([row], [], [], []))
    monkeypatch.setattr(
        red, "independent", lambda *a: (_ for _ in ()).throw(ValueError("bad primitive"))
    )
    rows, _, failed, audit = cap.collect(tmp_path, [])
    assert not rows[0]["eligible"] and failed[0]["observed"] == "bad primitive"
    assert not audit[0]["measurement_available"]


def test_main_publication_and_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8056-TERMINAL: final-byte consumers qualify a private reader."""
    fixture(tmp_path)
    frozen = cap.manifest(tmp_path / "workspace")
    assert frozen["coverage_includes"] == cap.OWNED
    assert any(s["name"] == "full_suite" and s["deadline_s"] <= 90 for s in frozen["commands"])
    original = cap.run_check
    private = tmp_path / "workspace"
    private.mkdir()
    counts = {p: dict(summary=dict(num_statements=1, covered_lines=1)) for p in cap.OWNED}

    def plan(scratch):
        atomic_json(scratch / "coverage.json", dict(files=counts))
        return dict(commands=[frozen["commands"][0]], dependency_hashes={})

    monkeypatch.setattr(cap, "manifest", plan)
    out = tmp_path / "results/experiment_8056_v697_capstone.json"
    health = dict(passed=False, classification="diagnostic", actual_exit=124, name="full_suite")
    atomic_json(
        out.parent / "raw" / out.stem / "repository_health_receipt.json", dict(receipts=[health])
    )
    assert cap.main(["--root", str(tmp_path), "--output", str(out)]) == 0
    value = json.loads(out.read_text())
    assert value["capstone_execution_ready_score"] == 1
    assert value["repository_health"] == [health]
    side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert side["publication"]["primary_sha256"] == sha256_file(out)
    monkeypatch.setattr(cap, "run_check", lambda *a: dict(passed=False))
    with pytest.raises(ValueError, match="owned_validation_failed"):
        cap.main(["--root", str(tmp_path), "--output", str(out)])
    monkeypatch.setattr(cap, "run_check", original)


def test_terminal_rejections(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8056-TERMINAL: validator flags and reader drift prevent approval."""
    fixture(tmp_path)
    v = cap.build(tmp_path, "20261003", tmp_path / "raw")
    counts = {p: dict(num_statements=1, covered_lines=1) for p in cap.OWNED}
    cap.complete(v, [dict(passed=True)], counts)
    cap.seal(v, tmp_path / "raw")
    log = tmp_path / "log.json"
    atomic_json(log, dict(flagged_count=1))
    monkeypatch.setattr(cap, "run_check", lambda *a: dict(passed=True, log_path=str(log)))
    out = tmp_path / "results/experiment_8056_v697_capstone.json"
    with pytest.raises(ValueError, match="terminal_validation_failed"):
        cap.publish(v, out, tmp_path / "private", tmp_path / "raw")
    atomic_json(log, dict(flagged_count=0))
    monkeypatch.setattr(cap, "reader_receipt", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="published_reader_drift"):
        cap.publish(v, out, tmp_path / "private", tmp_path / "raw")


def test_cost_primitives_and_hardware_limits(tmp_path):
    """SCENARIO-REPORT-8056-SCIENCE: paired cost sums cannot establish full deployment."""
    rows = []
    components = {
        k: 10
        for k in ("gradient_arithmetic_ns", "gradient_construction_ns", "guard_scans_ns", "ffi_ns")
    }
    for arm, ns in (("python", 1000), ("native", 1100)):
        for repetition in range(2):
            rows.append(
                dict(
                    arm=arm,
                    transaction_ns=ns,
                    repetition=repetition,
                    condition="feedback_constrained",
                    transaction_class="accepted",
                    natural=True,
                    excluded=False,
                    reset=False,
                    components=components,
                    guard_count=8,
                    stored_bytes=100,
                )
            )
    config = dict(seed=123, draws=10000)
    raw = tmp_path / "cost"
    atomic_json(raw / "transaction_rows.json", dict(rows=rows))
    v = dict(
        red.transaction.summarize(rows, config),
        config=config,
        raw_directory=str(raw),
        missing_service_components=["likelihood acquisition"],
    )
    r = red.independent(v, 8053)
    assert r["costs"]["complete_numerical_transaction_speedup"][0]["numerator"] == 2000
    assert r["complete_service_measured"] is False
    v["timing_components"] = []
    with pytest.raises(ValueError, match="transaction_drift"):
        red.independent(v, 8053)
    board = dict(
        rows=[{}],
        board_rows=[dict(board="GateMate", blocker="0xffffffff")],
        guard_fallback_counts={},
        guard_fallback_fraction=0.5,
        missing_cost_components=["transfer"],
    )
    assert red.independent(board, 8055)["board_rows"][0]["blocker"] == "0xffffffff"


def test_substantive_retirement(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8056-CUSTODY: renamed capstones retain unchanged missing operands."""
    prior = tmp_path / "prior.json"
    gate = dict(
        artifact_field="declared_primary_present", expected=True, observed=False, passed=False
    )
    atomic_json(
        prior,
        dict(
            honest_verdict="complete_blocked_old_name",
            verdict_class="blocked",
            gate_check_summary=[gate],
        ),
    )
    side = tmp_path / "side.json"
    atomic_json(side, dict(passed=True, primary_sha256=sha256_file(prior)))
    r = dict(
        task_id="exp8056-capstone",
        prior_path=str(prior),
        prior_sha256=sha256_file(prior),
        prior_verdict="complete_blocked_old_name",
        prior_authenticated=True,
        retire_if_same_verdict=True,
        retire=False,
        reopen_condition="New qualified token capture",
        terminal_verdict="complete_blocked_v697_capstone",
    )
    monkeypatch.setattr(cap.previous, "retirements", lambda *a: [dict(r)])
    rows = [dict(task_id="exp8056-capstone", verdict_class="blocked", gate_check_summary=[gate])]
    assert cap.retirements(tmp_path, [], rows, [gate])[0]["retire"]
    rows[0]["verdict_class"] = "null"
    assert not cap.retirements(tmp_path, [], rows, [gate])[0]["retire"]
