"""REQ-VERIFY-8137 / REQ-REPORT-8137: source custody without model work."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify import evidence_protocol_8124 as previous
from carnot.verify import source_protocol_8137 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from test_evidence_protocol_8124 import runtime_world
from test_methods_stream_custody_8111 import world, primary  # noqa: F401


def test_private_and_production_permission_boundary(tmp_path):
    """SCENARIO-VERIFY-8137: private tampering cannot relax production protection."""
    private = tmp_path / "private.json"
    ref = previous.capture_seal(private, dict(rows=[]), fixture=True)
    assert private.stat().st_mode & 0o200
    private.write_bytes(b"tampered")
    assert sha256_file(private) != ref["sha256"]
    sealed = tmp_path / "production.json"
    previous.capture_seal(sealed, dict(rows=[]), fixture=False)
    assert not sealed.stat().st_mode & 0o222
    with pytest.raises(PermissionError):
        sealed.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="private_capture_required"):
        previous.capture_seal(e.ROOT / "results/private-8137.json", {}, fixture=True)


@pytest.mark.parametrize("missing", [False, True])
def test_actual_production_identity_order(world, tmp_path, monkeypatch, missing):
    """REQ-VERIFY-8137: authenticated expectation precedes actual cache observer."""
    model = runtime_world(world)
    value = json.loads((world / previous.STREAM).read_text())
    if missing:
        value["runtime_identity"].pop("model_revision")
        primary(world / previous.STREAM, value)
    pins = {name: sha256_file(world / name) for name in previous.PINS}
    monkeypatch.setattr(previous, "PINS", pins)
    events = []
    original_read = previous.qualified.Custody.read

    def read(binder, path, *args, **kwargs):
        events.append(str(path))
        return original_read(binder, path, *args, **kwargs)

    def cache():
        assert all(str(world / n) in events for n in pins)
        events.append("cache")
        return dict(model_path=str(model))

    monkeypatch.setattr(previous.qualified.Custody, "read", read)
    monkeypatch.setattr(previous, "cached_current_model", cache)
    plan = previous.preconditions(world, tmp_path / "identity", fixture=False)
    assert events[-1] == "cache"
    assert plan["expected_runtime_identity"]["model_revision"] == (
        None if missing else "revision-fixture"
    )
    assert all(r["passed"] for r in plan["checks"]) == (not missing)


def test_source_only_work_and_terminal_replay(world, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8137: source reduction loads no stream/retention features."""
    runtime_world(world)

    def forbidden(*args, **kwargs):
        pytest.fail("stream worker must not run")

    monkeypatch.setattr(e.qualified, "authenticate_stream", forbidden)
    monkeypatch.setattr(e.qualified, "measure", forbidden)
    raw = tmp_path / "raw"
    work = e.measure(world, raw, fixture=True)
    assert len(work["rows"]) == 320
    assert len(work["protocol_conformance_rows"]) == 640
    assert set(work["source_role_manifests"]) == {"fit", "tune", "evaluation"}
    assert "stream_feature_manifest" not in work
    assert all(r["entailment_label"] is None for r in work["protocol_conformance_rows"])
    assert work["evidence_protocol"] == previous.protocol()
    assert work["fixture_permission_rows"][0]["mutation_rejected"]
    receipts = [dict(passed=True)]
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    value = e.build(work, raw, receipts, fixture=True)
    assert value["source_protocol_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == value["call_ledger"] == value["trained_head_specs"] == []
    assert not any(value["model_invocation_counts"].values())
    assert value["independent_count"] == value["completed_count"] == 320
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    manifest = Path(work["capture_manifest"]["path"])
    saved = manifest.read_bytes()
    manifest.write_bytes(b"changed")
    assert not e.replay(output)
    manifest.write_bytes(saved)
    value["completed_count"] += 1
    atomic_json(output, value)
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")
    disqualified = e.build(work, raw, [dict(passed=False)], fixture=True)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["source_protocol_ready_score"] == 0
    assert e.build(work, raw, receipts)["verdict_class"] == "null"


def test_external_block_and_mutated_source(world, tmp_path, monkeypatch):
    """REQ-REPORT-8137: external blockers and owned mutations stay distinct."""
    runtime_world(world)
    changed = e.measure(world, tmp_path / "mutation", fixture=True, mutation="source")
    assert e.build(changed, tmp_path, [dict(passed=True)])["verdict_class"] == "disqualified"
    absent = e.measure(tmp_path / "missing-root", tmp_path / "blocked", fixture=True)
    blocked = e.build(absent, tmp_path, [dict(passed=True)])
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["source_protocol_ready_score"] == 0
    assert blocked["intended_count"] == blocked["excluded_count"] == 320
    assert any(
        r["expected"] is True and r["observed"] is False for r in absent["gate_check_summary"]
    )
    bad = deepcopy(changed["rows"])
    bad[0]["denominator"] = 2
    with pytest.raises(ValueError, match="source_denominator"):
        e.reduce_rows(bad)
    monkeypatch.setattr(previous, "cached_current_model", lambda: None)
    production = e.measure(tmp_path / "missing-root", tmp_path / "production")
    assert e.build(production, tmp_path, [dict(passed=True)])["verdict_class"] == "blocked"


def test_private_cli_success_block_mutation_and_cold_replay(world, tmp_path):
    """SCENARIO-REPORT-8137: script path imports work outside the checkout."""
    runtime_world(world)
    output = tmp_path / (e.NAME + ".json")
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)

    def run(*args):
        command = [sys.executable]
        if env.get("COVERAGE_RCFILE"):
            command += ["-m", "coverage", "run", "--rcfile=" + env["COVERAGE_RCFILE"]]
        print("before Exp8137 private subprocess", flush=True)
        done = subprocess.run(
            [*command, str(e.ROOT / e.CLI), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print("after Exp8137 private subprocess exit=" + str(done.returncode), flush=True)
        return done

    args = ["--root", str(world), "--fixture-output", str(output)]
    done = run(*args)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(output.read_text())["source_protocol_ready_score"] == 1
    assert run("--cold-replay", str(output)).returncode == 0
    value = json.loads(output.read_text())
    value["source_protocol_ready_score"] = 0
    atomic_json(output, value)
    assert run("--cold-replay", str(output)).returncode == 1
    assert run(*args, "--mutation", "labels").returncode == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    args[1] = str(tmp_path / "absent-world")
    assert run(*args).returncode == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
