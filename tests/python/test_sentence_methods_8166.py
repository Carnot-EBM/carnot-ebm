"""REQ-REPORT-8166 / SCENARIO-VERIFY-8166-REPLAY: private custody and CLI."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import sentence_methods_8166 as e


@pytest.fixture
def world(tmp_path):
    """Use private original-role text; fixture truth gives no scientific credit."""
    root = tmp_path / "world"
    refs = {}
    for role, count in e.ROLES.items():
        rows = [
            dict(
                family_id=f"{role}-{i}",
                source_bytes=b"A fact.".hex(),
                answer_bytes=b"A fact. A qualifier may apply.".hex(),
            )
            for i in range(count)
        ]
        roster = [
            dict(unit_id=r["family_id"], source_cluster_id=r["family_id"], role=role, slot=i + 1)
            for i, r in enumerate(rows)
        ]
        path = root / (role + ".json")
        atomic_json(path, dict(request_rows=rows, roster=roster))
        refs[role] = e.reference(path)
    decisions = [
        dict(
            unit_id=f"tune-{i}",
            source_cluster_id=f"tune-{i}",
            role="tune",
            arm=arm,
            numerator=cost,
            denominator=1,
            status="completed",
        )
        for arm, cost in [("scalar_span", 0.4), ("linear12", 0.3), ("radial16", 0.2)]
        for i in range(64)
    ]
    path = root / "primitive_decisions.json"
    atomic_json(path, dict(rows=decisions))
    for filename in e.PINS:
        value = dict(
            required_checks_passed=True,
            flagged_adversarial=False,
            source_role_manifests=refs,
            rows=[],
            raw_shard_hashes=[e.reference(path)],
            MODEL_SPECS=["historical-fixture"],
            model_invocation_counts={"generate": 7},
        )
        atomic_json(root / filename, value)
    return root


def test_freeze_measure_reduce_and_terminal_classification(world, tmp_path):
    """REQ-VERIFY-8166: tune-only control choice and full source denominators."""
    work = e.measure(world, tmp_path / "raw", fixture=True)
    value = e.build(work, tmp_path / "raw", [dict(passed=True)], fixture=True)
    assert value["sentence_protocol_ready_score"] == 1
    assert value["completed_count"] == 320 and value["independent_count"] == 320
    assert value["baseline_manifest"]["selected_control"] == "radial16"
    assert value["MODEL_SPECS"] == [] and value["call_ledger"] == []
    assert value["generalized_learning_benefit_score"] == 0
    assert value["verifier_is_oracle"] and value["verdict_class"] == "circular_positive"
    assert len(value["feature_schema"]) == 16 and value["trained_head_specs"] == []
    assert value["statistical_plan"]["one_sided_alpha"] == 0.025
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"
    path = tmp_path / "primary.json"
    atomic_json(path, value)
    assert e.replay(path)
    original = deepcopy(value)
    for field in ["completed_count", "baseline_manifest", "claim_scope"]:
        changed = deepcopy(original)
        changed[field] = "tampered"
        atomic_json(path, changed)
        assert not e.replay(path)
    atomic_json(path, original)
    Path(original["raw_shard_hashes"][0]["path"]).write_text("tampered")
    assert not e.replay(path)
    assert not e.replay(tmp_path / "absent")


def test_authentic_terminal_routes_and_structural_rejection(world, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8166-REPLAY: use the real bound sidecar and input parser."""
    pins = {}
    for name in e.PINS:
        path = world / name
        raw = path.parent / "raw" / path.stem
        terminal, sidecar = raw / "terminal.json", raw / "validators/check.json"
        value = json.loads(path.read_text())
        value["terminal_validation_sidecar_path"] = str(terminal)
        atomic_json(path, value)
        atomic_json(terminal, dict(publication=dict(sidecar_path=str(sidecar))))
        atomic_json(sidecar, dict(primary_sha256=sha256_file(path), report=dict(passed=True)))
        pins[name] = sha256_file(path)
    monkeypatch.setattr(e, "PINS", pins)
    plan = e.inputs(world, tmp_path / "authentic")
    assert all(c["passed"] for c in plan["checks"])
    first = world / next(iter(pins))
    value = json.loads(first.read_text())
    value.pop("source_role_manifests")
    atomic_json(first, value)
    plan = e.inputs(world, tmp_path / "bad_structure", fixture=True)
    assert plan["checks"][-1]["check"] == "input_structure"
    work = e.measure(world, tmp_path / "blocked_structure", fixture=True)
    saved = e.build(work, tmp_path / "blocked_structure", [dict(passed=True)], fixture=True)
    path = tmp_path / "blocked.json"
    atomic_json(path, saved)
    assert e.replay(path)


def test_rehashed_primitive_and_log_tamper(world, tmp_path):
    """SCENARIO-VERIFY-8166-REPLAY: changing a hash does not replace original text."""
    raw = tmp_path / "raw"
    work = e.measure(world, raw, fixture=True)
    log = tmp_path / "log"
    log.write_text("normal exit")
    receipts = [dict(passed=True, log_path=str(log), log_sha256=sha256_file(log))]
    value = e.build(work, raw, receipts, fixture=True)
    path = tmp_path / "primary.json"
    atomic_json(path, value)
    assert e.replay(path)
    log.write_text("tampered log")
    assert not e.replay(path)
    log.write_text("normal exit")
    work["rows"][0]["answer_bytes"] = b"Changed answer.".hex()
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, raw, receipts, fixture=True)
    atomic_json(path, value)
    assert not e.replay(path)
    work["rows"] = e.freeze_sources(work["plan"]["sources"])
    work["config"]["H1"]["minimum_sources"] = 1
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, raw, receipts, fixture=True)
    atomic_json(path, value)
    assert not e.replay(path)


def test_baseline_forgery_and_runtime_operands(world, tmp_path, monkeypatch):
    """REQ-VERIFY-8166: a rehashed measurement cannot substitute baseline outputs."""
    raw = tmp_path / "raw"
    work = e.measure(world, raw, fixture=True)
    work["plan"]["baselines"][0]["numerator"] = 0.49
    work["baseline"] = e.select_control(work["plan"]["baselines"])
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, raw, [dict(passed=True)], fixture=True)
    path = tmp_path / "forged.json"
    atomic_json(path, value)
    assert not e.replay(path)
    monkeypatch.setattr(e.os, "access", lambda *_: False)
    plan = e.inputs(world, tmp_path / "runtime", fixture=True)
    assert any(
        c["check"] == "runtime_tool_executable" and c["observed"] is False for c in plan["checks"]
    )


def test_external_block_role_and_source_tamper(world, tmp_path):
    """SCENARIO-VERIFY-8166-REPLAY: report actual failed operands."""
    for mutation in ["labels", "roles", "slots", "source"]:
        work = e.measure(world, tmp_path / mutation, fixture=True, mutation=mutation)
        value = e.build(work, tmp_path / mutation, [dict(passed=True)], fixture=True)
        assert value["verdict_class"] == "blocked" and value["sentence_protocol_ready_score"] == 0
        assert any(not c["passed"] for c in value["gate_check_summary"])
    work = e.measure(tmp_path / "missing", tmp_path / "blocked", fixture=True)
    value = e.build(work, tmp_path / "blocked", [dict(passed=True)])
    assert value["honest_verdict"] == "complete_blocked_upstream_exists"
    assert value["gate_check_summary"][0]["observed"] is False
    plan = e.inputs(world, tmp_path / "production", fixture=False)
    assert any(c["check"] == "input_sha256" and not c["passed"] for c in plan["checks"])


def test_matched_control_uses_missing_slots_and_ignores_reserved():
    """REQ-VERIFY-8166: 64 tune sources remain the selector denominator."""
    rows = [
        dict(unit_id="tune", role="tune", arm="scalar_span", numerator=0, status="completed"),
        dict(
            unit_id="reserved",
            role="evaluation",
            arm="radial16",
            numerator=-100,
            status="completed",
        ),
    ]
    baseline = e.select_control(rows)
    assert baseline["selected_control"] == "scalar_span"
    assert baseline["control_costs"]["scalar_span"] == 63 * 0.5 / 64
    extra = dict(rows[0], arm="additive_cubic")
    assert "additive_cubic" in e.select_control([*rows, extra])["retained_controls"]
    with pytest.raises(ValueError):
        e.select_control(rows + rows[:1])


def test_manifest_and_private_cli_cold_replay(world, tmp_path):
    """SCENARIO-REPORT-8166-CLI: outside-checkout publication, block and replay."""
    specs = e.manifest(tmp_path, tmp_path / "candidate")
    assert all("::" not in p for c in specs["commands"][5:] for p in c["argv"])
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    command = [sys.executable]
    if env.get("COVERAGE_RCFILE"):
        command += ["-m", "coverage", "run", "--rcfile=" + env["COVERAGE_RCFILE"]]

    def run(*args):
        print("before private CLI subprocess", flush=True)
        done = subprocess.run(
            [*command, str(e.ROOT / e.CLI), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print("after private CLI subprocess exit=" + str(done.returncode), flush=True)
        return done

    output = tmp_path / (e.NAME + ".json")
    assert run("--root", str(world), "--fixture-output", str(output)).returncode == 0
    assert run("--cold-replay", str(output)).returncode == 0
    output.write_text("{}")
    assert run("--cold-replay", str(output)).returncode == 1
    output.unlink()
    assert run("--root", str(tmp_path / "missing"), "--fixture-output", str(output)).returncode == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    assert run("--fixture-output", str(e.ROOT / "results/private.json")).returncode == 2
    worker = tmp_path / "worker/measurement.json"
    assert run("--root", str(tmp_path / "missing"), "--worker-output", str(worker)).returncode == 0
