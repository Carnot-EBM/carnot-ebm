"""REQ-VERIFY-8118; REQ-REPORT-8118: private evidence never writes results/."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from carnot import experiment_8118_v702_fresh_acquisition_cost as e
from carnot.verify import fresh_acquisition_cost_8118 as c
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def views():
    """Reuse qualified public fixtures without admitting their evaluator labels."""
    return runpy.run_path(str(e.ROOT / "tests/python/test_fit_source_capture_8112.py"))["views"]()


def runtime():
    """Scripted transport exercises custody, with zero scientific model credit."""
    return runpy.run_path(str(e.ROOT / "tests/python/test_fit_source_capture_8099.py"))["Runtime"]()


def identity():
    """Exact versions make every invalidation dimension independently testable."""
    return dict(gguf_sha256="model", runtime_sha256="runtime", chat_template_sha256="template")


def capture(tmp_path, slots=None, **kwargs):
    """All primitive fixtures remain in pytest's private directory."""
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    rows = c.capture(
        c.freeze(views()) if slots is None else slots,
        runtime(),
        tmp_path / "slots",
        "private",
        ledger=ledger,
        key_identity=identity(),
        **kwargs,
    )
    return rows, ledger


def test_frozen_panel_and_exact_key():
    """SCENARIO-VERIFY-8118-CACHE preserves24 sources with two unique calls."""
    slots = c.freeze(views())
    assert len(slots) == 48 and slots == c.freeze(views())
    assert len({r["source_cluster_id"] for r in slots}) == 24
    assert len({r["family_id"] for r in slots}) == 48
    assert [r["arm"] for r in slots[:2]] == ["full_source", "source_order_permuted"]
    assert json.loads(slots[1]["request"]["messages"][1]["content"])["complete_source"] == (
        "Third sentence. Second sentence. First sentence."
    )
    with pytest.raises(ValueError):
        c.freeze({})
    key = c.judgment_key(slots[0], identity())
    for field in identity():
        assert c.judgment_key(slots[0], {**identity(), field: "changed"}) != key
    for field in ["source_bytes", "answer_bytes", "arm"]:
        assert c.judgment_key({**slots[0], field: "changed"}, identity()) != key
    assert c.config()["call_limit"] == 48 and c.config()["output_tokens"] == 4608


def test_current_acquisition_reuse_and_reduction(tmp_path):
    """REQ-VERIFY-8118: repeats cannot increase the independent denominator."""
    rows, ledger = capture(tmp_path)
    reduced = c.reduce(rows)
    assert reduced["completed_count"] == 48 and reduced["independent_count"] == 24
    assert reduced["fit_capture_ready_score"] == 1
    assert ledger.counts()["generation_calls_attempted"] == 48
    assert all(r["reuse"]["model_calls"] == 0 and r["reuse"]["exact_hit"] for r in rows)
    assert all(r["reuse"]["changed_rejected"] and r["reuse"]["stale_model_rejected"] for r in rows)
    assert all(r["human_target"] is None for r in rows)
    assert c.reduce(rows[:30])["fit_capture_ready_score"] == 0
    for field in ["numerator", "judgment_key", "parsed"]:
        bad = deepcopy(rows)
        bad[0][field] = "tampered"
        with pytest.raises(ValueError):
            c.reduce(bad)
    with pytest.raises(ValueError):
        c.reduce(rows + rows[:1])
    with pytest.raises(ValueError, match="empty_service_cache"):
        c.capture(c.freeze(views()), runtime(), tmp_path / "slots", "again", ledger=ledger)


def test_blocked_and_budget_preserve_slots(tmp_path):
    """SCENARIO-VERIFY-8118-CACHE retains all intended unattempted slots."""
    rows, ledger = capture(tmp_path, blocked_reason="absent_external")
    assert len(rows) == 48 and ledger.counts()["generation_calls_attempted"] == 0
    assert c.reduce(rows)["excluded_count"] == 48
    assert all(r["exclusion_reason"] == "absent_external" for r in rows)
    rows, ledger = capture(tmp_path / "budget", token_budget=0)
    assert c.reduce(rows)["censored_count"] == 48
    assert ledger.counts()["generation_calls_attempted"] == 0


def test_private_cli_and_cold_mutation(tmp_path):
    """SCENARIO-REPORT-8118-TERMINAL invokes actual CLI outside checkout."""
    public = tmp_path / "public.json"
    atomic_json(public, views())
    output = tmp_path / (e.NAME + ".json")
    cli = [str(e.ROOT / ".venv/bin/python"), str(e.ROOT / e.OWNED[2]), "--date", "20261004"]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        cli + ["--fixture", str(public), "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive" and value["verifier_is_oracle"]
    assert value["MODEL_SPECS"] == [] and value["acquisition_cost_ready_score"] == 0
    assert (
        subprocess.run(
            cli + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        ).returncode
        == 0
    )
    value["completed_count"] += 1
    atomic_json(output, value)
    assert (
        subprocess.run(
            cli + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        ).returncode
        == 1
    )


@pytest.mark.parametrize("branch", ["zero", "load", "generation", "owned", "fixture"])
def test_builder_terminal_branches(tmp_path, branch):
    """SCENARIO-REPORT-8118-TERMINAL declares only owned invocation receipts."""
    plan = dict(
        checks=[], references=[], slots=c.freeze(views()), protocol=identity(), manifests={}
    )
    rows, ledger = capture(tmp_path / "capture")
    if branch == "zero":
        plan["checks"] = [
            e.qualified.operand("external", tmp_path / "absent", "exists", True, False)
        ]
    if branch in ("load", "generation", "owned"):
        ledger.start("model_load", "owned-load", {})
        ledger.finish("owned-load", "completed", {})
    if branch in ("zero", "load"):
        rows = []
        ledger.rows = [r for r in ledger.rows if r["operation"] == "model_load"]
    result = dict(
        rows=rows,
        ledger=ledger.rows,
        checks=[],
        measured_duration_s=12,
        model_identity_receipt=dict(
            authenticated=True, offload_layers=[1], props=dict(chat_template="fixture")
        ),
        gpu_lease_receipt=dict(owner_pid=os.getpid()),
        resident_gpu_receipt=dict(output_tail="12000\n"),
        unloaded_gpu_receipt=dict(output_tail="4\n"),
    )
    value = e.build(
        plan,
        result,
        dict(passed=branch != "owned", receipts=[]),
        tmp_path,
        12,
        fixture=branch == "fixture",
    )
    assert value["acquisition_cost_ready_score"] == int(branch == "generation")
    assert value["gpu_memory_delta_mb"] == 11996
    assert e.replay_value(value)
    for field, changed in [
        ("completed_count", 999),
        ("MODEL_SPECS", ["stale"]),
        ("exact_judgment_keys", ["stale"]),
        ("acquisition_cost_ready_score", 1),
    ]:
        bad = deepcopy(value)
        bad[field] = changed
        if bad != value and not (field == "acquisition_cost_ready_score" and branch == "load"):
            with pytest.raises(ValueError):
                e.replay_value(bad)
    bad = deepcopy(value)
    bad["rows"] = [{"tampered": True}]
    with pytest.raises(ValueError, match="raw_rows_drift"):
        e.replay_value(bad)
    bad = deepcopy(value)
    bad["raw_shard_hashes"][0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="raw_hash_drift"):
        e.replay_value(bad)
    log = tmp_path / "validation.log"
    log.write_text("normal exit\n")
    from carnot.reporting.current_work_receipt import sha256_file

    value["validation_receipts"] = [dict(log_path=str(log), log_sha256=sha256_file(log))]
    assert e.replay_value(value)
    log.write_text("changed\n")
    with pytest.raises(ValueError, match="validation_log_drift"):
        e.replay_value(value)


def test_public_preconditions_and_frozen_commands(tmp_path, monkeypatch):
    """REQ-REPORT-8118 authenticates actual bytes, sidecars and executable paths."""
    from carnot.reporting.primary_publication import publish_primary
    from carnot.reporting.current_work_receipt import sha256_file

    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    root = tmp_path / "checkout"
    root.mkdir()
    public = views()
    manifests = {}
    for role, view in public.items():
        path = root / (role + ".json")
        atomic_json(path, view)
        manifests[role] = dict(path=str(path), sha256=sha256_file(path))
    pins = {}
    for n, name in [(8111, e.METHODS), (8102, e.HISTORY)]:
        path = root / name
        value = dict(
            experiment_id=n,
            task_id=f"exp{n}-fixture",
            honest_verdict="complete_null_fixture",
            verdict_class="null",
            flagged_adversarial=False,
            required_checks_passed=True,
            role_manifests=manifests,
            **identity(),
            runtime_identity=dict(model_revision="revision"),
        )
        terminal = root / f"terminal-{n}.json"
        value["terminal_validation_sidecar_path"] = str(terminal)
        receipt = publish_primary(path, value, lambda p: dict(passed=True))
        atomic_json(terminal, dict(publication=receipt))
        pins[name] = sha256_file(path)
    monkeypatch.setattr(e, "PINS", pins)
    monkeypatch.setattr(e, "INPUTS", ["absent-input"])
    plan = e.preconditions(root)
    assert len(plan["slots"]) == 48 and plan["protocol"]["model_revision"] == "revision"
    Path(json.loads((root / e.METHODS).read_text())["terminal_validation_sidecar_path"]).unlink()
    assert any(
        "authenticated_terminal" in r["check"] and not r["passed"]
        for r in e.preconditions(root)["checks"]
    )
    Path(manifests["fit"]["path"]).unlink()
    assert e.preconditions(root)["slots"] == []
    private = tmp_path / "validation"
    private.mkdir()
    specs = e.commands(private)
    assert specs[0]["name"] == "collect_consumers"
    assert any("test_primary_publication_7928.py" in a for a in specs[0]["argv"])
    assert {
        "cli_success",
        "cli_blocked",
        "cli_mutation",
        "cli_cold_replay",
        "repository_full_suite",
    } <= {r["name"] for r in specs}
    monkeypatch.setattr(e.qualified, "commands", e.commands)
    assert e.commands(private)[0]["name"] == "collect_consumers"


def test_normal_child_receipt_and_main_scope(tmp_path, monkeypatch):
    """REQ-REPORT-8118 covers the real supervisor boundary and scoped CLI wiring."""
    monkeypatch.setattr(e, "_run_check", lambda *a, **k: dict(actual_exit=0, timed_out=False))
    assert e.run_check(tmp_path, dict(name="private"), tmp_path, tmp_path)["normal_exit"]
    monkeypatch.setattr(e.transport, "main", lambda argv: 0)
    assert e.main(["--date", "20261004"]) == 0


def test_owned_runtime_version_and_progress(tmp_path, monkeypatch):
    """REQ-REPORT-8118 checks served template identity before a current call."""

    class Runtime:
        def load(self):
            return dict(props=dict(chat_template="native"))

    monkeypatch.setattr(e.qualified.legacy, "QwenRuntime", Runtime)
    plan = dict(protocol=dict(chat_template_sha256=canonical_hash("native")))
    raw = tmp_path / "raw"
    raw.mkdir()
    ledger = c.prior.Ledger(raw / "ledger.json")

    def owned(plan, raw, scratch):
        e.qualified.legacy.progress("before_model_load", 0)
        e.qualified.legacy.QwenRuntime().load()
        ledger.start("model_load", "load", {})
        ledger.finish("load", "completed", {})
        e.qualified.legacy.progress("after_model_load", 0)
        return dict(ledger=ledger.rows, checks=[])

    monkeypatch.setattr(e, "_live", owned)
    assert not e.live_capture(plan, raw, tmp_path)["owned_failure"]
    plan["protocol"]["chat_template_sha256"] = "stale"
    with pytest.raises(ValueError, match="chat_template_drift"):
        e.live_capture(plan, raw, tmp_path)
    monkeypatch.setattr(
        e, "_live", lambda *a: dict(ledger=ledger.rows, checks=[dict(passed=False)])
    )
    assert e.live_capture(plan, raw, tmp_path)["owned_failure"]


def test_failures_and_cache_mutation(tmp_path):
    """SCENARIO-VERIFY-8118-CACHE forbids silent retries and stale cache credit."""
    slots = c.freeze(views())
    key = c.judgment_key(slots[0], identity())
    rows, ledger = capture(tmp_path, slots[:2], historical_keys=[key])
    assert rows[0]["identical_historical_key"] == key
    rows[0]["reuse"]["model_calls"] = 1
    with pytest.raises(ValueError, match="cache_reuse_drift"):
        c.reduce(rows)

    class FailedRuntime:
        def count(self, text):
            return 1

        def generate(self, request):
            raise TimeoutError("owned_call_deadline")

    failed_ledger = c.prior.Ledger(tmp_path / "failed.json")
    rows = c.capture(
        slots[:1], FailedRuntime(), tmp_path / "failure", "private", ledger=failed_ledger
    )
    assert c.reduce(rows)["failed_count"] == 1
    assert rows[0]["component_costs"]["request_failure_s"] > 0


@pytest.mark.parametrize("failure", [False, True])
def test_owned_terminal_failure_is_disqualified(tmp_path, monkeypatch, failure):
    """SCENARIO-REPORT-8118-TERMINAL requires normal owned auditor exits."""
    rows, ledger = capture(tmp_path / "capture")
    plan = dict(checks=[], references=[], slots=c.freeze(views()), protocol={}, manifests={})
    value = e.build(
        plan,
        dict(rows=rows, ledger=ledger.rows, checks=[]),
        dict(passed=True, receipts=[]),
        tmp_path / "raw",
        1,
        fixture=True,
    )

    def check(root, spec, private, durable, **kwargs):
        log = tmp_path / (spec["name"] + ".log")
        log.write_text("normal exit\n")
        from carnot.reporting.current_work_receipt import sha256_file

        return dict(
            spec,
            passed=not failure,
            actual_exit=int(failure),
            normal_exit=True,
            duration_s=0.001,
            log_path=str(log),
            log_sha256=sha256_file(log),
        )

    monkeypatch.setattr(e, "run_check", check)
    output = tmp_path / (e.NAME + ".json")
    e.terminal_publish(output, value, tmp_path / "raw")
    published = json.loads(output.read_text())
    assert published["verdict_class"] == ("disqualified" if failure else "circular_positive")
    assert published["acquisition_cost_ready_score"] == 0
