"""REQ-VERIFY-8264 / REQ-REPORT-8264: current evidence and costs stay separate."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import evidence_view_execution_8264 as e


def test_triplets_and_null_custody(tmp_path):
    """SCENARIO-VERIFY-8264-CUSTODY: zero effect is a complete syntax result."""
    work = e.private_work(tmp_path)
    result = e.reduce(work)
    assert result["complete_triplets"] == 12
    assert result["syntax_yield"] == dict(numerator=36, denominator=36)
    assert result["selected_probability_changes"] == [0.0] * 12
    assert result["custody_passed"]
    for mutate in ["request_id", "answer_bytes", "transcript", "source_bytes"]:
        bad = deepcopy(work)
        bad["calls"][0][mutate] = "drift"
        assert not e.reduce(bad)["custody_passed"]
    missing = deepcopy(work)
    missing["calls"][0]["status"] = "failed"
    missing["calls"][0]["error"] = "timeout"
    assert e.reduce(missing)["complete_triplets"] == 11
    empty = deepcopy(work)
    empty["plans"][0]["plan"] = dict(status="unavailable", reason="missing_control", views={})
    empty["calls"] = empty["calls"][3:]
    assert e.reduce(empty)["missing_control_frequency"] == dict(numerator=1, denominator=12)


def test_conservative_branch_budget():
    """SCENARIO-VERIFY-8264-BUDGET: full focal rosters have independent allowances."""
    roster = {
        role: [dict(input_tokens=100, output_tokens=20)] * (count * 3)
        for role, count in [("fit", 128), ("tune", 64), ("reserved", 128)]
    }
    timings = [dict(prompt_n=100, prompt_ms=1000, predicted_n=20, predicted_ms=5000)]
    result = e.forecast(roster, timings, dict(load=20, setup=5, shutdown=5, serialization=1))
    assert result["fit"]["ready_score"] == 0
    assert result["tune"]["ready_score"] == 1
    assert result["reserved"]["ready_score"] == 0
    assert result["fit"]["intended_sources"] == 128
    assert e.forecast(roster, [], {})["fit"]["projected_seconds"] is None
    assert e.allow_capture(2390, 100) is False
    assert e.allow_capture(0, 100) is True


def test_roster_order_and_token_block():
    """SCENARIO-VERIFY-8264-BUDGET: no control failure can replace a source."""
    slots = [
        dict(
            unit_id=str(i),
            source_cluster_id=f"{12 - i:02}",
            source_bytes=b"One. Two.".hex(),
            answer_bytes=b"One.".hex(),
            role="fit",
        )
        for i in range(12)
    ]
    cache = [dict(unit_id=str(i), sentence_index=0, p_unsupported=0.8) for i in range(12)]
    plans = e.plan_slots(slots, cache, lambda _: 1)
    assert [p["source_cluster_id"] for p in plans] == sorted(p["source_cluster_id"] for p in plans)
    assert all(p["plan"]["status"] == "completed" for p in plans)
    assert all(
        p["plan"]["status"] == "unavailable" for p in e.plan_slots(slots, cache, lambda _: 99999)
    )
    assert e.plan_slots(slots, [], lambda _: 1) == []


def test_parse_exception_and_replay_mutations(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8264-CUSTODY: malformed parser and rehashed custody fail closed."""
    work = e.measure(tmp_path, tmp_path / "private", fixture=True)
    original_parse = e.d.focal.parse
    monkeypatch.setattr(e.d.focal, "parse", lambda *args: (_ for _ in ()).throw(ValueError("bad")))
    assert e.reduce(work["evidence"])["complete_triplets"] == 0
    monkeypatch.setattr(e.d.focal, "parse", original_parse)
    raw = tmp_path / "private"
    value = e.build(work, raw, [dict(name="private", passed=True)], fixture=True)
    primary = tmp_path / "primary.json"
    atomic_json(primary, value)
    assert e.replay(primary)

    def save(bad):
        bad.pop("reproducibility_checksum", None)
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(primary, bad)

    atomic_json(primary, dict(value, reproducibility_checksum="bad"))
    assert not e.replay(primary)
    bad = deepcopy(value)
    bad["code_config_hashes"][0]["sha256"] = "bad"
    save(bad)
    assert not e.replay(primary)
    log = tmp_path / "stdout"
    log.write_bytes(b"actual")
    bad = deepcopy(value)
    bad["validation_receipts"][0].update(stdout_path=str(log), stdout_sha256="bad")
    save(bad)
    assert not e.replay(primary)
    primitive = json.loads((raw / "primitive_evidence.json").read_bytes())
    altered = deepcopy(primitive)
    altered["calls"][0]["transcript"] = "tampered"
    atomic_json(raw / "primitive_evidence.json", altered)
    bad = deepcopy(value)
    for ref in bad["raw_shard_hashes"]:
        ref["sha256"] = e.sha256_file(Path(ref["path"]))
    save(bad)
    assert not e.replay(primary)
    altered = deepcopy(primitive)
    altered["plans"][0]["plan"]["selected_sentence_index"] = 99
    altered_work = dict(work, evidence=altered)
    atomic_json(raw / "measurement.json", altered_work)
    atomic_json(raw / "primitive_evidence.json", altered)
    atomic_json(
        primary, e.build(altered_work, raw, [dict(name="private", passed=True)], fixture=True)
    )
    assert not e.replay(primary)
    altered = deepcopy(primitive)
    altered["plans"][0]["plan"]["views"]["original"]["sentence_map"][0]["original_byte_start"] = 3
    slots = e.d.schedule(altered["plans"])
    altered["calls"] = e.d.decorate(slots, altered["calls"])
    altered_work = dict(work, evidence=altered)
    atomic_json(raw / "measurement.json", altered_work)
    atomic_json(raw / "primitive_evidence.json", altered)
    bad = e.build(altered_work, raw, [dict(name="private", passed=True)], fixture=True)
    atomic_json(primary, bad)
    assert not e.replay(primary)
    spec = dict(
        name="actual_child",
        argv=[str(e.ROOT / ".venv/bin/python"), "-c", "print('actual')"],
        deadline_s=10,
    )
    assert e.run_check(e.ROOT, spec, tmp_path, tmp_path / "logs")["passed"]


def cli(tmp_path, *args):
    """SCENARIO-REPORT-8264-CLI: execute actual children outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_real_cli_replay_negative_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8264-CLI: primitive changes cannot survive rehashing."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--date", "20261008", "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "null" and value["view_canary_ready_score"] == 0
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    bad = deepcopy(value)
    bad["complete_triplets"] = 11
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    tamper = tmp_path / "tamper.json"
    atomic_json(tamper, bad)
    assert cli(tmp_path, "--cold-replay", tamper).returncode == 1
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing.json").returncode == 1
    assert cli(tmp_path, "--date", "20261007").returncode == 2


def test_missing_external_and_owned_failure(tmp_path):
    """SCENARIO-REPORT-8264-CLI: external blocking differs from owned failures."""
    work = e.measure(tmp_path, tmp_path / "raw")
    receipts = [dict(name="private_check", passed=True)]
    value = e.build(work, tmp_path / "raw", receipts, fixture=True)
    assert value["verdict_class"] == "blocked" and value["intended_count"] == 36
    assert len(value["rows"]) == 36
    value = e.build(work, tmp_path / "raw", [dict(name="failed", passed=False)], fixture=True)
    assert value["verdict_class"] == "disqualified"


def test_cuda_refusal_keeps_roster_and_cold_replays(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8264-CLI: a complete external block preserves all intended slots."""
    work = e.measure(tmp_path, tmp_path / "blocked", fixture=True)
    work["evidence"]["calls"] = []
    monkeypatch.setattr(
        e.qualified, "run_check", lambda *args, **kwargs: dict(passed=False, actual_exit=1)
    )
    e.cuda_preflight(work, tmp_path / "blocked", tmp_path)
    raw = tmp_path / "blocked"
    atomic_json(raw / "primitive_evidence.json", work["evidence"])
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, raw, [dict(name="private", passed=True)], fixture=True)
    assert value["verdict_class"] == "blocked"
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert value["intended_count"] == value["censored_count"] == 36
    assert value["model_invocation_counts"]["model_loads_attempted"] == 0
    path = tmp_path / "blocked.json"
    atomic_json(path, value)
    assert e.replay(path)
    blocked = deepcopy(work["evidence"])
    for plan in blocked["plans"]:
        plan["plan"] = dict(status="unavailable", reason="token_budget", views={})
    assert e.reduce(blocked)["missing_control_frequency"] == dict(numerator=0, denominator=12)


def test_live_adapter_private_transport(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8264-CUSTODY: actual adapter statements use a declared private peer."""
    import threading
    from carnot.verify import evidence_view_live_8264 as live
    from carnot.reporting import v709_execution

    primitive = e.private_work(tmp_path / "original")
    evidence = deepcopy(primitive)
    evidence["calls"] = []
    evidence["rosters"] = {"fit": [dict(prompt="public", input_tokens=1, output_tokens=1)]}
    work = dict(
        evidence=evidence,
        checks=[],
        tokenizer=dict(metadata={"tokenizer.chat_template": "template"}),
    )
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

    class Worker:
        def post_json(self, endpoint, payload, deadline):
            if endpoint == "/apply-template":
                return dict(prompt="rendered " + payload["messages"][0]["content"])
            threading.Event().wait(0.15)
            return dict(
                content="0|B|0.50|[]",
                timings=dict(prompt_n=10, prompt_ms=20, predicted_n=10, predicted_ms=30),
            )

    class Runtime:
        def __init__(self, *args):
            self.worker = Worker()
            self.receipts = []

        def load(self):
            return dict(props=dict(chat_template="template"))

        def close(self):
            return dict(leak_free=True)

        def count(self, text):
            return 10

    def lifecycle(plan, raw, private):
        runtime = live.legacy.QwenRuntime()
        loaded = runtime.load()
        slots = live.legacy.capture.freeze({})
        rows = live.legacy.capture.capture(slots, runtime, raw, {})
        return dict(
            rows=rows,
            model_identity_receipt=loaded,
            cleanup=runtime.close(),
            runtime_receipts=runtime.receipts,
            model_loads_attempted=1,
            model_loads_completed=1,
            checks=[dict(field="owned", upstream_id="peer", passed=True)],
        )

    monkeypatch.setattr(live.legacy, "QwenRuntime", Runtime)
    monkeypatch.setattr(live.legacy, "live_capture", lifecycle)
    monkeypatch.setattr(
        live, "cached_current_model", lambda: dict(model_path=str(live.qualified.TOKENIZER_PATH))
    )
    monkeypatch.setattr(
        v709_execution,
        "child",
        lambda *args, **kwargs: dict(name="private_gpu_monitor", passed=True),
    )
    result = live.acquire(work, tmp_path / "adapter", tmp_path)
    assert e.reduce(evidence)["complete_triplets"] == 12
    assert len(result["runtime_receipts"]) == 36
    assert all(r["telemetry"] for r in result["runtime_receipts"])
    assert all(
        r["payload"]["n_predict"] == 64 and not r["payload"]["cache_prompt"]
        for r in result["runtime_receipts"]
    )
    evidence["rosters"] = {
        "fit": [dict(prompt="public", input_tokens=1, output_tokens=1), dict(prompt=None)]
    }
    work["tokenizer"]["metadata"]["tokenizer.chat_template"] = "changed"
    with pytest.raises(ValueError, match="template_drift"):
        live.acquire(dict(work, checks=[]), tmp_path / "template_drift", tmp_path)
    work["tokenizer"]["metadata"]["tokenizer.chat_template"] = "template"
    monkeypatch.setattr(Runtime, "count", lambda *args: 9000)
    with pytest.raises(ValueError, match="context_budget"):
        live.acquire(dict(work, checks=[]), tmp_path / "context", tmp_path)
    monkeypatch.setattr(live, "cached_current_model", lambda: {})
    assert live.acquire(dict(work, checks=[]), tmp_path / "missing", tmp_path) == {}


def test_current_public_input_authentication_and_prepare(tmp_path):
    """SCENARIO-REPORT-8264-CLI: actual current fields and hashes bind public inputs."""
    from carnot.verify import evidence_view_live_8264 as live

    work = dict(checks=[], refs=[])
    inputs = live.public_inputs(e.ROOT, work)
    assert all(c["passed"] for c in work["checks"])
    assert all(
        "human_target" not in row
        for data in [inputs["fit"], inputs["evaluation"]]
        for row in data["cache"]
    )
    prepared = live.prepare(inputs, lambda _: 1)
    assert len(prepared["plans"]) == 12
    assert len({p["unit_id"] for p in prepared["plans"]}) == 12
    assert {k: len(v) for k, v in prepared["rosters"].items()} == dict(
        fit=384, tune=192, reserved=384
    )


def test_real_worker_and_production_supervision(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8264-CLI: normal child work and owned commands remain bounded."""
    from carnot.verify import evidence_view_live_8264 as live

    monkeypatch.setattr(live, "public_inputs", lambda root, work: {})
    monkeypatch.setattr(e, "cuda_preflight", lambda *args: None)
    monkeypatch.setattr(e.qualified, "tokenizer", lambda work: (lambda _: 1, {}))
    primitive = e.private_work(tmp_path / "primitive")
    monkeypatch.setattr(live, "prepare", lambda *args: primitive)
    monkeypatch.setattr(live, "acquire", lambda *args: {})
    work = e.measure(tmp_path, tmp_path / "measured")
    assert len(work["evidence"]["plans"]) == 12
    monkeypatch.setattr(
        live, "public_inputs", lambda *args: (_ for _ in ()).throw(ValueError("external"))
    )
    assert not all(c["passed"] for c in e.measure(tmp_path, tmp_path / "invalid")["checks"])
    monkeypatch.setattr(e, "measure", lambda *args, **kwargs: work)
    assert e.main(["--worker-output", str(tmp_path / "worker.json")]) == 0
    monkeypatch.setattr(
        e, "run_check", lambda *args, **kwargs: dict(name=args[1]["name"], passed=True)
    )
    monkeypatch.setattr(e, "build", lambda *args, **kwargs: dict(private=True))
    monkeypatch.setattr(e.execution, "publish", lambda *args: None)
    from carnot.reporting.current_work_receipt import atomic_json as actual_atomic

    def record(path, value):
        actual_atomic(path, value)
        if path.name == "validation_commands.json":
            actual_atomic(path.parent / "measurement.json", work)

    monkeypatch.setattr(e, "atomic_json", record)
    assert e.main(["--root", str(tmp_path), "--output", str(tmp_path / "primary.json")]) == 0
    health = tmp_path / "prior_health.json"
    atomic_json(
        health,
        dict(
            argv=[str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
            passed=False,
            actual_exit=-9,
        ),
    )
    assert (
        e.main(
            [
                "--root",
                str(tmp_path),
                "--output",
                str(tmp_path / "repaired.json"),
                "--repository-health-receipt",
                str(health),
            ]
        )
        == 0
    )
    actual_manifest = e.manifest

    def exhausted(private, candidate):
        value = actual_manifest(private, candidate)
        for spec in value["commands"]:
            spec["deadline_s"] = 901
        return value

    monkeypatch.setattr(e, "manifest", exhausted)
    assert e.main(["--root", str(tmp_path), "--output", str(tmp_path / "exhausted.json")]) == 0
    with pytest.raises(SystemExit):
        e.main(["--fixture-output", str(e.ROOT / "results" / (e.NAME + ".json"))])
