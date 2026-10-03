"""REQ-REPORT-7995: private script CLI and immutable original receipts."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot import experiment_7995_v693_qwen_development_capture as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from test_qwen_development_capture_7995 import Runtime, views


def cli(args, cwd):
    argv = [sys.executable, str(e.ROOT / e.OWNED[-1]), *map(str, args)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    config = env.get("CARNOT_7995_COVERAGE_CONFIG")
    if config:
        argv[1:1] = ["-m", "coverage", "run", "--rcfile=" + config]
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)


def test_private_cli_success_blocked_and_cold_replay(tmp_path):
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, views())
    output = tmp_path / "out" / (e.NAME + ".json")
    run = cli(["--fixture", fixture, "--output", output], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["capture_ready_score"] == 0
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert cli(["--cold-replay", output], tmp_path).returncode == 0
    changed = copy.deepcopy(value)
    changed["rows"][0]["request"]["max_tokens"] = 95
    atomic_json(output, changed)
    assert cli(["--cold-replay", output], tmp_path).returncode == 1
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert cli(["--root", tmp_path / "absent", "--output", blocked], tmp_path).returncode == 0
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    assert cli(["--cold-replay", blocked], tmp_path).returncode == 0
    assert (
        cli(
            ["--fixture", fixture, "--output", e.ROOT / "results" / output.name], tmp_path
        ).returncode
        == 1
    )
    assert cli(["--date", "20260930"], tmp_path).returncode == 2


def test_authenticated_receipt_and_validation_failure(tmp_path):
    ledger = e.c.Ledger(tmp_path / "ledger.json")
    slots = e.c.freeze(views())[:4]
    rows = e.c.capture(slots, Runtime(), tmp_path / "slots", "id", ledger=ledger)
    result = dict(
        rows=rows,
        checks=[],
        measured_duration_s=11,
        model_loads_completed=1,
        cleanup=dict(leak_free=True),
        model_identity_receipt=dict(authenticated=True),
        ledger=ledger.rows,
    )
    value = e.build(result, dict(checks=[], references=[]), tmp_path, 0, 11)
    assert value["verdict_class"] == "null"
    assert value["capture_ready_score"] == 0
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 4
    e.apply_validation(value, [dict(passed=False, required=True, name="owned", exit_code=1)])
    assert value["verdict_class"] == "disqualified"
    assert all(value[k] == 0 for k in e.READY)
    result["measured_duration_s"] = 1
    assert (
        e.build(result, dict(checks=[], references=[]), tmp_path, 0, 1)["verdict_class"]
        == "disqualified"
    )
    result["checks"] = [dict(passed=False, field="identity")]
    result["measured_duration_s"] = 11
    assert (
        e.build(result, dict(checks=[], references=[]), tmp_path, 0, 11)["verdict_class"]
        == "blocked"
    )


def test_authentication_contract_and_exact_hashes(tmp_path):
    plan = e.authenticate(tmp_path)
    assert plan["checks"] and not all(c["passed"] for c in plan["checks"])
    for n, pin in [(7994, e.COHORT_PIN), (7969, e.HISTORY_PIN)]:
        p = e.ROOT / e.SOURCES[n]
        assert sha256_file(p) == pin
    assert e.authenticate(e.ROOT)["checks"]
    assert all(c["passed"] for c in e.authenticate(e.ROOT)["checks"])
    assert (
        e.authenticate(e.ROOT)["protocol_fingerprint"]
        == json.loads((e.ROOT / e.SOURCES[7969]).read_text())["protocol_fingerprint"]
    )


def test_original_ledger_mutations_fail(tmp_path):
    ledger = e.c.Ledger(tmp_path / "ledger.json")
    rows = e.c.capture(e.c.freeze(views())[:4], Runtime(), tmp_path / "slots", "id", ledger=ledger)
    value = e.build(
        dict(
            rows=rows,
            checks=[],
            ledger=ledger.rows,
            measured_duration_s=11,
            model_loads_completed=1,
        ),
        dict(checks=[], references=[]),
        tmp_path,
        0,
        11,
    )
    value["original_capture_code_snapshot"] = []
    atomic_json(tmp_path / "original.json", dict(ledger=ledger.rows))
    value["original_call_receipts"] = e.reference(tmp_path / "original.json")
    value["raw_response_shards"] = [
        e.reference(p) for p in sorted((tmp_path / "slots").glob("slot-*.json"))
    ]
    e.replay(value)
    value["model_invocation_counts"]["generation_calls_attempted"] = 3
    with pytest.raises(ValueError, match="ledger_count"):
        e.replay(value)


def test_replay_mutations_and_preflight(tmp_path):
    """REQ-REPORT-7995: original snapshots remain independent of validators."""
    preflight = e.preflight(tmp_path)
    assert preflight["passed"]
    candidate = Path(preflight["candidate"]["path"])
    value = json.loads(candidate.read_text())
    e.replay(value)
    value["generated_tokens"] += 1
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(value)
    value["generated_tokens"] -= 1
    value["verdict_class"] = "disqualified"
    value["capture_ready_score"] = 1
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(value)
    value["capture_ready_score"] = 0
    raw = tmp_path / "transcript"
    original = raw / "original.json"
    atomic_json(original, dict(ledger=[]))
    value["original_call_receipts"] = e.reference(original)
    with pytest.raises(ValueError, match="original_ledger"):
        e.replay(value)
    atomic_json(original, dict(ledger=value["current_invocation_ledger"]))
    with pytest.raises(ValueError, match="original_receipt_hash"):
        e.replay(value)
    value["original_call_receipts"] = e.reference(original)
    shard = Path(value["raw_response_shards"][0]["path"])
    shard.write_text("{}")
    with pytest.raises(ValueError, match="shard_hash"):
        e.replay(value)
    with patch.object(
        e, "_classify_current_task_inference_claim", return_value=dict(state="contradictory")
    ):
        with pytest.raises(ValueError, match="preflight_provenance"):
            e.preflight(tmp_path / "bad")


def test_live_adapter_records_load_outcomes(tmp_path):
    """REQ-VERIFY-7995: load failure and success both leave owned receipts."""
    public = tmp_path / "public.json"
    atomic_json(public, views()["calibration"])
    plan = dict(public_role_manifests=dict(calibration=dict(path=str(public))))

    class BaseRuntime:
        def __init__(self, model, scratch, gpu):
            self.model = model

        def load(self):
            return dict(authenticated=True)

    def invoke(plan, raw, scratch):
        runtime = e.legacy.QwenRuntime(Path("model.gguf"), scratch, 0)
        runtime.load()
        return dict(rows=[])

    with (
        patch.object(e.legacy, "QwenRuntime", BaseRuntime),
        patch.object(e.legacy, "live_capture", invoke),
    ):
        result = e.live_capture(plan, tmp_path / "success", tmp_path)
    assert result["ledger"][0]["status"] == "completed"
    BaseRuntime.load = lambda self: (_ for _ in ()).throw(RuntimeError("identity"))
    with (
        patch.object(e.legacy, "QwenRuntime", BaseRuntime),
        patch.object(e.legacy, "live_capture", invoke),
    ):
        with pytest.raises(RuntimeError, match="identity"):
            e.live_capture(plan, tmp_path / "failure", tmp_path)
    assert e.c.Ledger(tmp_path / "failure/ledger.json").counts()["model_loads_failed"] == 1


def test_main_validation_archive_and_owned_failure(tmp_path):
    """REQ-REPORT-7995: real CLI logic archives exits and suppresses readiness."""

    def commands(root, specs, *, log_dir, extra_env, heartbeat_s):
        log_dir.mkdir(parents=True)
        log = log_dir / "check.log"
        log.write_text("private test receipt")
        atomic_json(
            log_dir.parent / "coverage.json",
            dict(totals=dict(num_statements=1, covered_lines=1, percent_covered=100)),
        )
        return [
            dict(name="owned", scope="owned", passed=True, exit_code=0, log_path=str(log)),
            dict(
                name="health",
                scope="repository_health",
                passed=False,
                exit_code=1,
                log_path=str(log),
            ),
        ]

    def capture(plan, raw, scratch):
        ledger = e.c.Ledger(raw / "ledger.json")
        ledger.start("model_load", "private-load", {})
        ledger.finish("private-load", "completed", {})
        rows = e.c.capture(e.c.freeze(views())[:4], Runtime(), raw / "slots", "id", ledger=ledger)
        return dict(rows=rows, ledger=ledger.rows, checks=[], measured_duration_s=11)

    with (
        patch.object(e, "run_commands", commands),
        patch.object(e, "live_capture", capture),
        patch.object(e, "terminal", return_value=dict(passed=True, receipts=[])),
    ):
        output = tmp_path / "success" / (e.NAME + ".json")
        assert e.main(["--output", str(output)]) == 0
        value = json.loads(output.read_text())
        assert value["verdict_class"] == "null"
        assert value["repository_health"][0]["exit_code"] == 1
        assert value["coverage_statement_counts"]["num_statements"] == 1

    def failed(*args, **kwargs):
        receipts = commands(*args, **kwargs)
        receipts[0]["passed"] = False
        return receipts

    with (
        patch.object(e, "run_commands", failed),
        patch.object(e, "terminal", return_value=dict(passed=True, receipts=[])),
    ):
        output = tmp_path / "disqualified" / (e.NAME + ".json")
        assert e.main(["--output", str(output)]) == 0
        assert json.loads(output.read_text())["verdict_class"] == "disqualified"


@pytest.mark.parametrize("failure", ["terminal", "private_reader", "published_reader"])
def test_main_terminal_and_reader_rejection(tmp_path, failure):
    """REQ-REPORT-7995: failed owned publication checks cannot imply readiness."""
    output = tmp_path / "out" / (e.NAME + ".json")
    report = dict(passed=failure != "terminal", receipts=[])
    readers = [dict(passed=failure != "private_reader"), dict(passed=failure != "published_reader")]
    with (
        patch.object(e, "terminal", return_value=report),
        patch.object(e, "reader_receipt", side_effect=readers),
    ):
        assert e.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 1
    if failure == "terminal":
        assert not output.exists()
    else:
        assert (output.parent / "raw" / e.NAME).exists()
