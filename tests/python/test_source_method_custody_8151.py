"""REQ-VERIFY-8151 / REQ-REPORT-8151: immutable methods and source-only replay."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify import development_methods_8098 as methods
from carnot.verify import source_method_custody_8151 as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from test_evidence_protocol_8124 import runtime_world
from test_methods_stream_custody_8111 import world  # noqa: F401


def historical(root, raw):
    """Make private source history so no test changes production evidence."""
    runtime_world(root)
    work = e.previous.measure(root, raw, fixture=True)
    value = e.previous.build(work, raw, [dict(passed=True)], fixture=True)
    atomic_json(root / e.HISTORY, value)
    return value


def test_immutable_method_identity_and_stored_indexerror(tmp_path):
    """SCENARIO-VERIFY-8151 reproduces the stored failure before the fix."""
    mutable = tmp_path / "research-roadmap-vNEXT.md"
    mutable.write_text("## Unrelated future headings\n")
    with pytest.raises(IndexError):
        mutable.read_text().split("## Frozen data and numerical protocol", 1)[1]
    baseline = methods.methods()
    pinned = tmp_path / "v701.md"
    pinned.write_bytes(methods.METHOD_PATH.read_bytes())
    assert methods.methods(pinned, methods.METHOD_SHA256) == baseline
    mutable.write_text("## Another unrelated roadmap\n")
    assert methods.methods() == baseline
    with pytest.raises(ValueError, match="method_identity"):
        methods.methods(pinned, "sha256:unknown")
    pinned.write_text(pinned.read_text() + "changed")
    with pytest.raises(ValueError, match="method_identity"):
        methods.methods(pinned, methods.METHOD_SHA256)


def test_source_only_custody_and_replay(world, tmp_path, monkeypatch):
    """REQ-VERIFY-8151 keeps exact prompts without learning or reserved access."""
    history = historical(world, tmp_path / "history")
    stream = world / e.previous.previous.STREAM
    saved_stream = stream.read_bytes()
    stream.unlink()
    monkeypatch.setattr(e.qualified.cohort, "select", lambda *a: pytest.fail("resplit"))
    raw = tmp_path / "owned"
    work = e.measure(world, raw, fixture=True)
    stream.write_bytes(saved_stream)
    assert work["protocol_conformance_rows"] == history["protocol_conformance_rows"]
    assert work["rows"] == history["rows"]
    assert work["source_custody_ready_score"] == 1
    assert work["expected_runtime_identity"]["model_revision"] == "revision-fixture"
    receipts = [dict(passed=True)]
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    value = e.build(work, raw, receipts, fixture=True)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    assert value["source_protocol_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == value["trained_head_specs"] == value["call_ledger"] == []
    assert value["run_date"] == "20261005"
    for field in ("completed_count", "source_custody_ready_score"):
        changed = deepcopy(value)
        changed[field] -= 1
        atomic_json(output, changed)
        assert not e.replay(output)
    atomic_json(output, value)
    shard = Path(work["capture_manifest"]["path"])
    shard.write_bytes(b"tampered")
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")


def test_source_blocks_and_owned_failure(world, tmp_path):
    historical(world, tmp_path / "history")
    work = e.measure(world, tmp_path / "raw", fixture=True, mutation="source")
    assert (
        e.build(work, tmp_path, [dict(passed=True)], fixture=True)["verdict_class"]
        == "disqualified"
    )
    assert (
        e.build(work, tmp_path, [dict(passed=False)], fixture=True)["source_custody_ready_score"]
        == 0
    )
    missing = e.measure(tmp_path / "missing", tmp_path / "blocked", fixture=True)
    value = e.build(missing, tmp_path, [dict(passed=True)], fixture=True)
    assert value["verdict_class"] == "blocked"
    failed = next(r for r in value["gate_check_summary"] if not r["passed"])
    assert value["honest_verdict"] == "complete_blocked_" + failed["check"]


def test_private_cli_success_block_tamper_cold_replay(world, tmp_path):
    """SCENARIO-REPORT-8151 uses real dated scripts without PYTHONPATH."""
    historical(world, tmp_path / "history")
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)

    def run(*args):
        command = [sys.executable]
        if env.get("COVERAGE_RCFILE"):
            command += ["-m", "coverage", "run", "--rcfile=" + env["COVERAGE_RCFILE"]]
        print("before private subprocess", flush=True)
        done = subprocess.run(
            [*command, str(e.ROOT / e.CLI), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print("after private subprocess exit=" + str(done.returncode), flush=True)
        return done

    output = tmp_path / (e.NAME + ".json")
    args = ["--date", "20261005", "--root", str(world), "--fixture-output", str(output)]
    done = run(*args)
    assert done.returncode == 0, done.stdout + done.stderr
    assert run("--cold-replay", str(output)).returncode == 0
    value = json.loads(output.read_text())
    value["completed_count"] += 1
    atomic_json(output, value)
    assert run("--cold-replay", str(output)).returncode == 1
    assert run(*args, "--mutation", "labels").returncode == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    args[3] = str(tmp_path / "missing")
    assert run(*args).returncode == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"


def test_whole_source_overlength_mask_and_quote_parser():
    """REQ-VERIFY-8151 excludes both arms without changing source bytes or labels."""
    assert e.whole_source_mask([6000, 5999])["status"] == "eligible"
    excluded = e.whole_source_mask([6000, 6001])
    assert excluded == dict(
        status="excluded",
        exclusion_reason="whole_source_overlength",
        arm_mask=[0, 0],
        entailment_label=None,
    )
    for invalid in ([1], [True, 2], [-1, 0]):
        with pytest.raises(ValueError, match="input_token_counts"):
            e.whole_source_mask(invalid)
    parser = e.previous.previous.parse
    source = "évidence fact.".encode()
    valid = parser(
        json.dumps(dict(p_hallucination=0.2, quote="fact", byte_start=10, byte_end=14)),
        source,
        "source_span",
    )
    assert valid["valid_quote"] == 1 and valid["entailment_label"] is None
    invalid = parser(
        json.dumps(dict(p_hallucination=0.2, quote="fact", byte_start=9, byte_end=13)),
        source,
        "source_span",
    )
    assert invalid["p_hallucination"] == 0.2 and invalid["valid_quote"] == 0
    for payload in (
        "[]",
        '{"p_hallucination":true}',
        '{"p_hallucination":NaN}',
        '{"p_hallucination":2}',
    ):
        assert parser(payload, source, "holistic")["status"] == "missing_probability"


def test_production_cache_observation_uses_frozen_expectation(world, tmp_path, monkeypatch):
    """REQ-VERIFY-8151: cache observations cannot supply their own expectation."""
    historical(world, tmp_path / "history")
    monkeypatch.setattr(e, "HISTORY_SHA256", sha256_file(world / e.HISTORY))
    model = world / "cache/revision-fixture/model.gguf"
    monkeypatch.setattr(
        e.previous.previous, "cached_current_model", lambda: dict(model_path=str(model))
    )
    plan = e.preconditions(world, tmp_path / "checked", fixture=False)
    assert plan["expected_runtime_identity"]["model_revision"] == "revision-fixture"
    assert all(r["passed"] for r in plan["checks"])
    monkeypatch.setattr(
        e.previous.previous,
        "cached_current_model",
        lambda: dict(model_path=str(tmp_path / "wrong-revision/model.gguf")),
    )
    plan = e.preconditions(world, tmp_path / "rejected", fixture=False)
    failed = next(r for r in plan["checks"] if not r["passed"])
    assert failed["check"] == "cache_revision"
    assert failed["expected"] == "revision-fixture" and failed["observed"] == "wrong-revision"


def test_completed_health_receipt_is_reused_without_rerun(tmp_path, monkeypatch):
    """REQ-REPORT-8151 keeps a stopped full-suite diagnostic separate from owned checks."""
    path = tmp_path / "health.json"
    atomic_json(path, dict(actual_exit=2, passed=False, classification="diagnostic"))
    monkeypatch.setenv("CARNOT_8151_HEALTH_RECEIPT", str(path))
    monkeypatch.setattr(
        e.execution,
        "main",
        lambda argv: e.execution.run_check(
            e.ROOT, dict(name="repository_full_suite"), tmp_path, tmp_path
        )["actual_exit"],
    )
    assert e.main([]) == 2
    monkeypatch.setattr(
        e.execution,
        "main",
        lambda argv: e.execution.run_check(e.ROOT, dict(name="owned"), tmp_path, tmp_path)[
            "actual_exit"
        ],
    )
    monkeypatch.setattr(e.execution, "run_check", lambda *a, **kw: dict(actual_exit=0))
    assert e.main([]) == 0


def test_original_method_seal_matches_pinned_document(world, tmp_path):
    """REQ-VERIFY-8151 retains the original production method seal byte identity."""
    historical(world, tmp_path / "history")
    work = e.measure(world, tmp_path / "raw", fixture=True)
    seal = next(r for r in work["raw_shard_hashes"] if r["path"].endswith("v701_methods.json"))
    assert (
        seal["sha256"] == "sha256:10d31f444cb10b91b5360f5edbe5e3b825df283fa9c4552cbc99416eeb272cb7"
    )


def test_manifest_instruments_consumers_and_dates_e2e_replay(tmp_path, monkeypatch):
    """REQ-REPORT-8151 measures edited consumer statements and freezes runnable replay."""
    monkeypatch.setattr(e.execution, "e", e)
    monkeypatch.setattr(e.execution, "OWNED", e.OWNED)
    commands = {r["name"]: r for r in e.manifest(tmp_path, tmp_path / "candidate.json")["commands"]}
    argv = commands["consumer_and_E2E015_019"]["argv"]
    assert "run" in argv and "coverage" in argv[2]
    replay = commands["E2E016_cold_replay"]["argv"]
    assert replay[replay.index("--date") + 1] == "20260929"
