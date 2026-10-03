"""REQ-REPORT-8011, SCENARIO-REPORT-8011-PAIRED/CUSTODY: private source trials."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_8011_v694_qwen_source_sensitivity as e
from carnot.reporting.current_work_receipt import atomic_json
from test_qwen_development_capture_7995 import views


def panel():
    return e.p.freeze(views()["stream"])


def recorded(tmp_path, n=192):
    frozen = panel()
    frozen["slots"] = frozen["slots"][:n]
    ledger = e.c.Ledger(tmp_path / "ledger.json")
    rows = e.c.capture(
        frozen["slots"],
        e.p.upstream.FixtureRuntime(),
        tmp_path / "slots",
        e.canonical_hash(frozen),
        ledger=ledger,
        token_budget=192 * 96,
    )
    return frozen, rows, ledger


def cli(args, tmp_path):
    argv = [sys.executable, str(e.ROOT / e.CLI), *map(str, args)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    config = env.get("CARNOT_8011_COVERAGE_CONFIG")
    if config:
        argv[1:1] = ["-m", "coverage", "run", "--rcfile=" + config]
    return subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)


def test_paired_floor_and_control(tmp_path):
    """SCENARIO-REPORT-8011-PAIRED: duplicates never add independent sources."""
    frozen, rows, _ = recorded(tmp_path)
    result = e.reduce(rows)
    assert result["sample_size_budget"]["independent"] == 64
    assert len(result["sensitivity_rows"]) == len(result["duplicate_control_rows"]) == 64
    assert result["paired_confidence_intervals"]["swap_minus_duplicate"]["mean"] == 0
    assert result["placebo_instability"]["absolute_mean"] == 0
    for row in rows:
        if row["arm"] == "swap":
            reply = json.loads(row["raw_response"]["choices"][0]["message"]["content"])
            reply["unsupported_probability"] = 0.9
            row["raw_response"]["choices"][0]["message"]["content"] = json.dumps(reply)
            row["parsed"] = e.c.risk.transport.parse_response(
                row["raw_response"], row["visible_ids"]
            )
    result = e.reduce(rows)
    assert result["paired_confidence_intervals"]["swap_minus_duplicate"]["lower95"] > 0
    incomplete = e.reduce(rows[:141])
    assert incomplete["sample_size_budget"]["independent"] == 47
    assert all(v["mean"] is None for v in incomplete["paired_confidence_intervals"].values())
    rows[0]["parsed"] = {}
    with pytest.raises(ValueError, match="parse_drift"):
        e.reduce(rows)
    assert frozen["methods"] == e.p.METHODS


def test_reducer_failures_exclusions_and_identity(tmp_path):
    """REQ-REPORT-8011: parsing errors and nonstarts remain in denominators."""
    _, rows, _ = recorded(tmp_path, 6)
    rows[0].update(started=False, status="censored", raw_response={})
    rows[1].update(started=False, status="excluded", public_eligible=False, raw_response={})
    rows[2]["raw_response"]["model"] = "wrong-model"
    for row in rows:
        row["parsed"] = e.c.risk.transport.parse_response(row["raw_response"], row["visible_ids"])
    result = e.reduce(rows)
    assert result["sample_size_budget"]["censored"] == 1
    assert result["sample_size_budget"]["excluded"] == 1
    assert result["parse_failure_count"] == 1
    assert len(result["censor_rows"]) == 3
    with pytest.raises(ValueError, match="slot_identity"):
        e.reduce(rows + rows[:1])


def test_authentication_and_command_freeze(tmp_path):
    """SCENARIO-REPORT-8011-CUSTODY: exact upstream and scoped validation are frozen."""
    plan = e.authenticate(e.ROOT)
    assert all(r["passed"] for r in plan["checks"])
    assert len(plan["panel"]["slots"]) == 192
    blocked = e.authenticate(tmp_path)
    assert blocked["checks"] and not blocked["checks"][0]["passed"]
    commands = e.commands(tmp_path)
    assert any(s.scope == "repository_health" for s in commands)
    assert any("--strict" in s.argv for s in commands)
    assert any("--fail-under=100" in s.argv for s in commands)


def test_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8011-CUSTODY: real CLI fixtures cannot become natural science."""
    public = tmp_path / "public.json"
    atomic_json(public, views()["stream"])
    output = tmp_path / "out" / (e.NAME + ".json")
    run = cli(["--fixture", public, "--output", output], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["source_sensitivity_ready_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert value["sample_size_budget"]["independent"] == 64
    assert cli(["--cold-replay", output], tmp_path).returncode == 0
    value["sensitivity_rows"][0]["swap_minus_original"] = 99
    atomic_json(output, value)
    assert cli(["--cold-replay", output], tmp_path).returncode == 1
    assert cli(["--cold-replay", tmp_path / "missing"], tmp_path).returncode == 1
    assert cli(["--date", "20261001"], tmp_path).returncode == 2
    assert cli(["--fixture", public], tmp_path).returncode == 1


def test_live_adapter_records_load_and_budgets(tmp_path, monkeypatch):
    """REQ-REPORT-8011: qualified runtime owns load starts and frozen reservations."""
    frozen = panel()
    monkeypatch.setattr(e.legacy.QwenRuntime, "load", lambda self: dict(authenticated=True))
    monkeypatch.setattr(e.legacy.QwenRuntime, "count", lambda self, text: 30)
    monkeypatch.setattr(
        e.legacy.QwenRuntime,
        "generate",
        lambda self, request: e.p.upstream.FixtureRuntime().generate(request),
    )

    def fake_capture(plan, raw, scratch):
        assert scratch != raw and scratch.parent == raw
        runtime = e.legacy.QwenRuntime(tmp_path / "fake.gguf", scratch, 0)
        runtime.load()
        slots = e.legacy.capture.freeze({})
        assert slots == frozen["slots"]
        rows = e.legacy.capture.capture(slots, runtime, raw / "slots", plan["capture_identity"])
        return dict(rows=rows, checks=[], measured_duration_s=11)

    monkeypatch.setattr(e.legacy, "live_capture", fake_capture)
    result = e.live(frozen, dict(protocol={}), tmp_path)
    assert result["ledger"][0]["operation"] == "model_load"
    assert result["ledger"][0]["status"] == "completed"
    assert sum(r["started"] for r in result["rows"]) == 192
    monkeypatch.setattr(
        e.legacy.QwenRuntime,
        "load",
        lambda self: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    with pytest.raises(RuntimeError):
        e.live(frozen, dict(protocol={}), tmp_path / "failed")
    ledger = e.c.Ledger(tmp_path / "failed/ledger.json")
    assert ledger.counts()["model_loads_failed"] == 1


def test_main_owned_checks_block_and_terminal_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8011-CUSTODY: validation never starts another model call."""
    frozen = panel()
    monkeypatch.setattr(
        e, "authenticate", lambda root: dict(panel=frozen, checks=[], references=[], protocol={})
    )
    mode = ["ok"]

    def fake_live(panel, plan, raw):
        rows = e.c.capture(
            panel["slots"],
            None,
            raw / "slots",
            e.canonical_hash(panel),
            ledger=e.c.Ledger(raw / "ledger.json"),
            deadline_s=0,
        )
        return dict(
            rows=rows,
            ledger=[],
            checks=[dict(passed=False, field="lease", expected=True, observed=False)],
            measured_duration_s=0,
        )

    monkeypatch.setattr(e, "live", fake_live)
    original = e.run_commands

    def run(root, specs, **kwargs):
        if specs[0].name == "focused":
            scratch = Path(kwargs["extra_env"]["COVERAGE_FILE"]).parent
            atomic_json(
                scratch / "coverage.json",
                dict(totals=dict(num_statements=1, covered_lines=1, percent_covered=100), files={}),
            )
            return [dict(name="controlled_check", scope="owned", passed=mode[0] != "owned_fail")]
        if specs[0].name == "fresh_process_cold_reduction" and mode[0] == "cold_fail":
            return [dict(passed=False)]
        return original(root, specs, **kwargs)

    monkeypatch.setattr(e, "run_commands", run)
    for case in ["ok", "owned_fail", "cold_fail"]:
        mode[0] = case
        output = tmp_path / case / (e.NAME + ".json")
        assert e.main(["--output", str(output)]) == int(case == "cold_fail")
        if output.exists():
            value = json.loads(output.read_text())
            assert value["verdict_class"] == ("blocked" if case == "ok" else "disqualified")
            assert value["source_sensitivity_ready_score"] == 0
            assert len(value["rows"]) == 192
    mode[0] = "ok"
    monkeypatch.setattr(e, "terminal", lambda path: dict(passed=False))
    assert e.main(["--output", str(tmp_path / "terminal" / (e.NAME + ".json"))]) == 1
    monkeypatch.setattr(e, "terminal", lambda path: dict(passed=True))
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    assert e.main(["--output", str(tmp_path / "reader" / (e.NAME + ".json"))]) == 1

    def changed_calls(root, specs, **kwargs):
        if specs[0].name == "focused":
            ledger = e.c.Ledger(kwargs["log_dir"].parent / "ledger.json")
            ledger.start("model_load", "unexpected_validation_load", {})
            return []
        return original(root, specs, **kwargs)

    monkeypatch.setattr(e, "run_commands", changed_calls)
    assert e.main(["--output", str(tmp_path / "drift" / (e.NAME + ".json"))]) == 1


def candidate(tmp_path):
    frozen, rows, ledger = recorded(tmp_path)
    result = dict(rows=rows, ledger=ledger.rows)
    atomic_json(tmp_path / "panel.json", frozen)
    atomic_json(tmp_path / "capture.json", result)
    atomic_json(tmp_path / "validation_commands.json", dict(code_config_hashes=[]))
    atomic_json(tmp_path / "plan.json", dict(checks=[], references=[]))
    atomic_json(tmp_path / "validation_receipts.json", dict(receipts=[]))
    return e.build(frozen, dict(checks=[], references=[]), result, tmp_path, [], 1, True)


def test_cold_custody_mutations(tmp_path):
    """SCENARIO-REPORT-8011-CUSTODY: altered aggregates and joins fail independently."""
    value = candidate(tmp_path)
    e.replay(value)
    mutations = [
        (lambda v: v.update(protocol_fingerprint="drift"), "protocol_drift"),
        (lambda v: v["current_invocation_ledger"].append({}), "original_ledger_drift"),
        (
            lambda v: v["calls_after_validation"].update(generation_calls_attempted=1),
            "current_call_drift",
        ),
        (lambda v: v["request_rows"][0].update(response_id="changed"), "request_model_ids"),
        (lambda v: v["rows"][0].update(request={}), "published_rows_drift"),
        (
            lambda v: v.update(source_sensitivity_ready_score=1, verdict_class="blocked"),
            "unsafe_readiness",
        ),
        (lambda v: v.update(fixture_scope=None), "original_ledger_drift"),
    ]
    for mutate, message in mutations:
        changed = copy.deepcopy(value)
        mutate(changed)
        with pytest.raises(ValueError, match=message):
            e.replay(changed)
    changed = copy.deepcopy(value)
    changed.update(fixture_scope=None, verdict_class="disqualified")
    original = json.loads((tmp_path / "capture.json").read_text())
    atomic_json(tmp_path / "capture.json", dict(original, ledger=[]))
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(tmp_path / "capture.json"):
            ref["sha256"] = e.sha256_file(tmp_path / "capture.json")
    with pytest.raises(ValueError, match="response_identity"):
        e.replay(changed)
    atomic_json(tmp_path / "capture.json", original)
    checkpoint = Path(value["checkpoint_references"][0]["path"])
    original_bytes = checkpoint.read_bytes()
    checkpoint.write_text("{}")
    with pytest.raises(ValueError, match="custody_hash_drift"):
        e.replay(value)
    checkpoint.write_bytes(original_bytes)
    for frozen_drift in [False, True]:
        changed = copy.deepcopy(value)
        altered = copy.deepcopy(original)
        if frozen_drift:
            altered["rows"][0]["request"] = {}
            atomic_json(checkpoint, altered["rows"][0])
            changed["checkpoint_references"][0]["sha256"] = e.sha256_file(checkpoint)
        else:
            altered["rows"] = []
        atomic_json(tmp_path / "capture.json", altered)
        for ref in changed["raw_shard_hashes"]:
            if ref["path"] == str(tmp_path / "capture.json"):
                ref["sha256"] = e.sha256_file(tmp_path / "capture.json")
        with pytest.raises(
            ValueError, match="frozen_request_drift" if frozen_drift else "checkpoint_roster_drift"
        ):
            e.replay(changed)
        checkpoint.write_bytes(original_bytes)
        atomic_json(tmp_path / "capture.json", original)
    rows = original["rows"]
    rows[0]["raw_response"]["usage"]["completion_tokens"] = 33
    rows[0]["parsed"] = e.c.risk.transport.parse_response(
        rows[0]["raw_response"], rows[0]["visible_ids"]
    )
    with pytest.raises(ValueError, match="decode_budget"):
        e.reduce(rows)


def test_failed_prerequisite_cli_and_task_cap(tmp_path, monkeypatch):
    """REQ-REPORT-8011: blocked inputs cannot launch, and task deadlines cannot publish."""
    frozen = panel()
    check = e.operand(
        8010, tmp_path / "absent", "protocol_ready_score", 1, "missing_field_contract_error"
    )
    monkeypatch.setattr(
        e, "authenticate", lambda root: dict(panel=frozen, checks=[check], references=[])
    )
    output = tmp_path / "blocked" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and len(value["rows"]) == 192
    assert value["gate_check_summary"][0]["artifact_field"] == "protocol_ready_score"
    actual_time = e.time.monotonic
    offset = [0]
    monkeypatch.setattr(
        e, "time", type("Clock", (), {"monotonic": staticmethod(lambda: actual_time() + offset[0])})
    )

    def terminal(path):
        offset[0] = 4801
        return dict(passed=True)

    monkeypatch.setattr(e, "terminal", terminal)
    assert (
        e.main(
            ["--root", str(tmp_path), "--output", str(tmp_path / "timeout" / (e.NAME + ".json"))]
        )
        == 1
    )
    assert not (tmp_path / "timeout" / (e.NAME + ".json")).exists()


def test_replay_binds_runtime_fields_and_frozen_commands(tmp_path):
    """SCENARIO-REPORT-8011-CUSTODY: summaries cannot replace owned CUDA evidence."""
    value = candidate(tmp_path)
    for key in [
        "model_identity_receipt",
        "gguf_sha256",
        "model_revision",
        "gpu_lease_receipt",
        "offload_evidence",
        "runtime_receipts",
        "cleanup",
        "capacity_receipt",
        "server_log",
    ]:
        changed = copy.deepcopy(value)
        changed[key] = "tampered"
        with pytest.raises(ValueError, match="runtime_receipt_drift"):
            e.replay(changed)
    manifest = tmp_path / "validation_commands.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="custody_hash_drift"):
        e.replay(value)


def test_fixture_readiness_promotion_is_rejected(tmp_path):
    """REQ-REPORT-8011: scripted evidence cannot pass a natural readiness gate."""
    value = candidate(tmp_path)
    value.update(
        source_sensitivity_ready_score=1,
        verdict_class="positive",
        honest_verdict="complete_positive_source_sensitivity",
    )
    value["acceptance_gate_results"].update(source_comparison=True, source_dependence=True)
    with pytest.raises(ValueError, match="claim_gate_drift"):
        e.replay(value)


def test_referenced_runtime_file_hashes(tmp_path):
    """SCENARIO-REPORT-8011-CUSTODY: log and native-map byte drift remains visible."""
    path = tmp_path / "server.log"
    path.write_text("owned worker log")
    original_reference = e.reference(path)
    e.verify_references(
        dict(
            nested=[
                e.reference(path),
                dict(log_path=str(path), log_sha256=e.sha256_file(path)),
                None,
            ]
        )
    )
    path.write_text("changed")
    with pytest.raises(ValueError, match="receipt_file_hash"):
        e.verify_references(original_reference)
    with pytest.raises(ValueError, match="receipt_file_hash"):
        e.verify_references(dict(log_path=str(path), log_sha256="drift"))
