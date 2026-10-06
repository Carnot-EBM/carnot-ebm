"""REQ-VERIFY-8188 / REQ-REPORT-8188: private service evidence earns no live credit."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.verify import exact_request_8188 as e


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """REQ-VERIFY-8188: use the original panel without creating result fixtures."""
    return e.inputs(e.ROOT, tmp_path_factory.mktemp("8188-inputs"), fixture=True)


@pytest.fixture(scope="module")
def native(data):
    """SCENARIO-VERIFY-8188-CACHE: E2E-003 crosses the loaded Rust extension."""
    return e.prior.shared.host.old.prior.old.host.load_binding(data)[0]


def test_identity_and_durable_recovery(data, native, tmp_path):
    """REQ-VERIFY-8188: evidence survives restart while every acquisition operand binds."""
    slot = deepcopy(data["slots"][0])
    identity = e.identity(data, slot)
    for field in identity:
        changed = deepcopy(identity)
        changed[field] = "different"
        assert e.key(changed) != e.key(identity)
    head = deepcopy(data)
    head["head"]["intercept"] += 0.5
    assert e.identity(head, slot) == identity
    cache = e.Cache(tmp_path / "cache")
    assert cache.read(identity) is None
    runtime = e.prior.shared.prior.FixtureRuntime()
    ledger = e.prior.shared.Ledger(tmp_path / "ledger.json")
    cold = e.request(data, slot, "cold_miss", runtime, native, cache, tmp_path, ledger, "cold")
    hit = e.request(
        head, slot, "exact_repeat", runtime, native, e.Cache(cache.path), tmp_path, ledger, "hit"
    )
    assert cold["status"] == hit["status"] == "completed"
    assert hit["cache_hit"] and not cold["cache_hit"]
    assert len(ledger.rows) == 1 and cold["head_sha256"] != hit["head_sha256"]
    assert cold["probability"] != hit["probability"]
    evidence = cache.path / (e.key(identity).split(":")[1] + ".json")
    evidence.write_text("broken")
    assert e.Cache(cache.path).read(identity) is None
    recovered = e.request(
        data, slot, "exact_repeat", runtime, native, cache, tmp_path, ledger, "recovered"
    )
    assert recovered["status"] == "completed" and not recovered["cache_hit"]
    assert list(cache.path.glob("*.corrupt-*"))
    envelope = json.loads(evidence.read_text())
    envelope["evidence"]["usage"]["completion_tokens"] += 1
    e.prior.atomic_json(evidence, envelope)
    assert cache.read(identity) is None
    cache.write(identity, cold["raw_response"])
    changed = json.loads(evidence.read_text())
    changed["identity"] = {}
    e.prior.atomic_json(evidence, changed)
    assert cache.read(identity) is None
    cache.write(identity, cold["raw_response"])
    refresh = e.request(
        data, slot, "forced_refresh", runtime, native, cache, tmp_path, ledger, "refresh"
    )
    assert not refresh["cache_hit"] and len(ledger.rows) == 3


def test_measure_reduction_and_tamper(data, native, tmp_path):
    """REQ-VERIFY-8188-REPLAY: source bootstrap cannot count repeats as sources."""
    panel = deepcopy(data)
    panel["slots"] = panel["slots"][:2]
    ledger = e.prior.shared.Ledger(tmp_path / "ledger.json")
    work = e.measure(panel, e.prior.shared.prior.FixtureRuntime(), native, tmp_path, ledger)
    assert len(work["requests"]) == 8 and len(ledger.rows) == 8
    reduction = e.reduce(work)
    assert reduction["completed_count"] == 8 and reduction["independent_count"] == 2
    assert reduction["equivalent_behavior_score"] == 1
    assert reduction["nfr01_met"] is False
    assert len(reduction["amortized_cost_scenarios"]) == 4
    assert reduction["measured_repeat_frequency"] is None
    assert e.validate_work(panel, work)
    for mutation in ("numerator", "probability", "cache_key", "source_cluster_id"):
        bad = deepcopy(work)
        bad["requests"][0][mutation] = "forged"
        assert not e.validate_work(panel, bad)
    duplicate = deepcopy(work)
    duplicate["requests"].append(duplicate["requests"][0])
    assert not e.validate_work(panel, duplicate)
    divergent = deepcopy(work)
    next(r for r in divergent["requests"] if r["condition"] == "forced_refresh")[
        "generated_text"
    ] = "different"
    assert e.reduce(divergent)["equivalent_behavior_score"] == 0
    cutoff = e.measure(
        panel, runtime=None, native=native, raw=tmp_path / "cutoff", ledger=ledger, started=0
    )
    assert e.reduce(cutoff)["censored_count"] == 8
    assert e.validate_work(panel, cutoff)
    assert e.reduce({"requests": []})["paired_speed_intervals"] == []


def test_acquisition_failure(data, native, tmp_path):
    """SCENARIO-VERIFY-8188-CACHE: a failed generation stays visible and is never cached."""

    class Broken:
        def generate(self, request):
            raise RuntimeError("transport")

    ledger = e.prior.shared.Ledger(tmp_path / "ledger.json")
    row = e.request(
        data,
        data["slots"][0],
        "cold_miss",
        Broken(),
        native,
        e.Cache(tmp_path / "cache"),
        tmp_path,
        ledger,
        "failed",
    )
    assert row["status"] == "failed" and ledger.rows[0]["status"] == "failed"
    assert row["numerator"] is None
    assert e.validate_work(data, {"requests": [row]})


def test_inputs_and_live_adapter(data, tmp_path, monkeypatch):
    """REQ-REPORT-8188: external blocks identify operands; live adapters own their lease."""
    missing = e.inputs(tmp_path, tmp_path / "missing", fixture=True)
    assert not missing["ready"] and not missing["checks"][0]["passed"]
    monkeypatch.setattr(e, "PIN", "wrong")
    assert not e.inputs(e.ROOT, tmp_path / "drift", fixture=True)["ready"]
    monkeypatch.setattr(e, "PIN", e.prior.sha256_file(e.ROOT / e.UPSTREAM))
    monkeypatch.setattr(
        e.prior.shared,
        "tokenizer_counts",
        lambda d: (dict(data["runtime_freeze"]), [1] * len(d["slots"])),
    )
    monkeypatch.setattr(e, "load_template", lambda d: "{{ messages[0]['content'] }}")
    live_data = e.inputs(e.ROOT, tmp_path / "live")
    assert live_data["ready"] and live_data["chat_template"]
    monkeypatch.setattr(
        e.prior.shared, "tokenizer_counts", lambda d: (_ for _ in ()).throw(ValueError("drift"))
    )
    assert not e.inputs(e.ROOT, tmp_path / "runtime")["ready"]
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
    assert not e.live(deepcopy(data), tmp_path, tmp_path)["work"]
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    legacy = e.prior.shared.prior.qualified.legacy
    monkeypatch.setattr(legacy, "live_capture", lambda *a, **k: dict(task=legacy.TASK))
    monkeypatch.setattr(legacy.QwenRuntime, "load", lambda s: dict(authenticated=True))

    def adapter(d, raw, private):
        legacy.progress("waiting", 0)
        assert legacy.QwenRuntime.load(object())["authenticated"]
        assert e.prior.shared.prior.capture is e.measure
        return legacy.live_capture({}, raw, private)

    monkeypatch.setattr(e.prior.shared.prior, "live", adapter)
    assert e.live(data, tmp_path, tmp_path)["task"] == "exp8188-exact-request-service"


def test_build_replay_and_owned_failure(data, native, tmp_path):
    """REQ-REPORT-8188: fixture clocks and forged readiness cannot become live claims."""
    panel = deepcopy(data)
    panel["slots"] = panel["slots"][:1]
    ledger = e.prior.shared.Ledger(tmp_path / "ledger.json")
    work = e.measure(panel, e.prior.shared.prior.FixtureRuntime(), native, tmp_path, ledger)
    receipts = [
        dict(name=n, passed=True, normal_exit=True, actual_exit=0) for n in e.REQUIRED_CHECK_NAMES
    ]
    value = e.build(
        panel, dict(work=work, ledger=ledger.rows), tmp_path, receipts, "20261006", 11, True
    )
    output = tmp_path / (e.NAME + ".json")
    e.prior.atomic_json(output, value)
    assert e.replay(output)
    assert not e.replay(tmp_path / "absent")
    for field in ("cached_service_ready_score", "completed_count", "amortized_cost_scenarios"):
        bad = deepcopy(value)
        bad[field] = "forged"
        bad["reproducibility_checksum"] = e.prior.checksum(bad)
        e.prior.atomic_json(output, bad)
        assert not e.replay(output)
    failed = e.build(panel, {}, tmp_path / "bad", [], "20261006", 1, False)
    assert failed["verdict_class"] == "disqualified"
    missing = e.inputs(tmp_path, tmp_path / "blocked", fixture=True)
    blocked = e.build(missing, {}, tmp_path / "blocked", receipts, "20261006", 1, False)
    assert blocked["verdict_class"] == "blocked"


def test_cli_and_validation_plan(tmp_path):
    """SCENARIO-REPORT-8188-CLI: private success, missing, tamper and replay run from /tmp."""
    import os
    import runpy
    import subprocess
    import sys
    from carnot.reporting import exact_request_execution_8188 as runner

    plan = runner.validation_plan(tmp_path)
    assert {c.name for c in plan} >= set(e.REQUIRED_CHECK_NAMES)
    for c in plan:
        if c.name in {"ruff_check", "ruff_format", "scoped_spec_coverage"}:
            assert not any("::" in a for a in c.argv)
    env = dict(os.environ, PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    for args, expected in [
        (["--fixture-e2e", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (
            [
                "--fixture-e2e",
                str(tmp_path / "block" / output.name),
                "--root",
                str(tmp_path / "absent"),
            ],
            0,
        ),
        (["--fixture-e2e", str(e.ROOT / "results" / output.name)], 2),
        (["--date", "invalid"], 2),
    ]:
        print("[8188-test] subprocess before", args, flush=True)
        child = subprocess.run(
            command + args, cwd=tmp_path, env=env, capture_output=True, timeout=120
        )
        print("[8188-test] subprocess after", child.returncode, flush=True)
        assert child.returncode == expected, child.stdout.decode() + child.stderr.decode()
    value = json.loads(output.read_text())
    value["rows"][0]["numerator"] = -1
    e.prior.atomic_json(output, value)
    assert not e.replay(output)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sys, "argv", [e.CLI, "--cold-replay", str(tmp_path / "absent")])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(e.ROOT / e.CLI), run_name="__main__")
    assert error.value.code == 1
    monkey.undo()


def test_template_and_replay_rejections(data, native, tmp_path, monkeypatch):
    """REQ-VERIFY-8188-REPLAY: parsed evidence, durable records and custody all bind."""
    import struct

    model = tmp_path / "embedded.gguf"
    template = "{{ messages[0]['content'] }}"
    model.write_bytes(struct.pack("<Q", len(template)) + template.encode())
    metadata = dict(
        field_provenance=dict(metadata_keys={"tokenizer.chat_template": dict(value_offset=0)})
    )
    monkeypatch.setattr(
        e.prior.shared.prior.qualified.legacy, "read_gguf_metadata", lambda p: metadata
    )
    assert e.load_template(dict(runtime_freeze=dict(model_path=str(model)))) == template
    templated = deepcopy(data)
    templated["chat_template"] = template
    assert e.render(templated, data["slots"][0])
    with pytest.raises(ValueError):
        e.features(data["slots"][0], dict(choices=[dict(message=dict(content="broken"))]))
    panel = deepcopy(data)
    panel["slots"] = panel["slots"][:1]
    ledger = e.prior.shared.Ledger(tmp_path / "ledger.json")
    work = e.measure(panel, e.prior.shared.prior.FixtureRuntime(), native, tmp_path, ledger)
    for field, value in [
        ("action", "forged"),
        ("generated_text", "forged"),
        ("evidence", [dict(path=str(model), sha256="bad")]),
        ("response_ns", 0),
        ("head_sha256", "bad"),
    ]:
        bad = deepcopy(work)
        bad["requests"][0][field] = value
        assert not e.validate_work(panel, bad)
    first = work["requests"][0]
    response = next(
        Path(r["path"]) for r in first["evidence"] if Path(r["path"]).name == "response.json"
    )
    response.write_text("{}")
    changed = deepcopy(work)
    next(r for r in changed["requests"][0]["evidence"] if Path(r["path"]) == response)["sha256"] = (
        e.sha256_file(response)
    )
    assert not e.validate_work(panel, changed)
    e.atomic_json(response, first["raw_response"])
    store = Path(first["store"]["path"])
    original = json.loads(store.read_text())
    forged = deepcopy(original)
    forged["state"]["records"][0]["request_id"] = "another"
    forged["state"]["causal_hash"] = e.key(forged["state"]["records"][0])
    forged["sha256"] = e.key(forged["state"])
    e.atomic_json(store, forged)
    changed = deepcopy(work)
    changed["requests"][0]["store"]["sha256"] = e.sha256_file(store)
    assert not e.validate_work(panel, changed)
    e.atomic_json(store, original)
    receipts = [
        dict(name=n, passed=True, normal_exit=True, actual_exit=0) for n in e.REQUIRED_CHECK_NAMES
    ]
    value = e.build(
        panel, dict(work=work, ledger=ledger.rows), tmp_path, receipts, "20261006", 11, True
    )
    output = tmp_path / (e.NAME + ".json")
    value["source_artifact_hashes"][0]["sha256"] = "bad"
    e.atomic_json(output, value)
    assert not e.replay(output)


def test_main_paths_and_sealed_logs(data, native, tmp_path, monkeypatch):
    """REQ-REPORT-8188: close owned failures and recover evidence without new calls."""
    from carnot.reporting import exact_request_execution_8188 as runner

    panel = deepcopy(data)
    panel["slots"] = panel["slots"][:2]
    raw = tmp_path / "saved"
    ledger = e.prior.shared.Ledger(raw / "ledger.json")
    work = e.measure(panel, e.prior.shared.prior.FixtureRuntime(), native, raw, ledger)
    result = dict(work=work, ledger=ledger.rows, checks=[])
    e.atomic_json(raw / "live_result.json", result)
    e.atomic_json(raw / "input_data.json", panel)
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--fixture-e2e", str(output)]) == 0
    assert runner.main(["--fixture-e2e", str(output), "--resume-evidence", str(raw)]) == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1

    def receipts(plan, path):
        return [dict(name=c.name, passed=True, normal_exit=True, actual_exit=0) for c in plan]

    monkeypatch.setattr(runner, "execute", receipts)
    monkeypatch.setattr(e, "inputs", lambda *a, **k: deepcopy(panel))
    monkeypatch.setattr(e, "live", lambda *a: deepcopy(result))
    assert runner.main(["--output", str(output)]) == 0
    monkeypatch.setattr(
        runner, "execute", lambda plan, path: [dict(passed=False, normal_exit=True)]
    )
    assert runner.main(["--output", str(tmp_path / "failed" / output.name)]) == 1
    monkeypatch.setattr(e, "validate_work", lambda *a: False)
    with pytest.raises(ValueError, match="resume_evidence"):
        runner.main(["--fixture-e2e", str(output), "--resume-evidence", str(raw)])
    log = tmp_path / "owned.log"
    log.write_text("actual completed validation log")
    monkeypatch.setattr(runner.qualified, "execute", lambda *a: [dict(log_path=str(log))])
    # Use the original function, because the failure-path test replaced its name.
    function = __import__("importlib").reload(runner).execute
    for _ in range(2):
        assert (
            Path(function([], tmp_path / "sealed")[0]["log_path"]).read_bytes() == log.read_bytes()
        )


def test_complete_panel_budget_and_restart(data, native, tmp_path, monkeypatch):
    """REQ-VERIFY-8188: exactly 24 source clusters, at most 74 calls and a child restart."""
    import os
    import subprocess
    from carnot.reporting import exact_request_execution_8188 as runner

    ledger = e.prior.shared.Ledger(tmp_path / "ledger.json")
    work = e.measure(data, e.prior.shared.prior.FixtureRuntime(), native, tmp_path, ledger)
    assert len(work["requests"]) == 96 and len(ledger.rows) == 74
    assert e.reduce(work)["independent_count"] == 24
    assert e.validate_work(data, work)
    operand = e.identity(data, data["slots"][0])
    path = tmp_path / "identity.json"
    e.atomic_json(path, operand)
    py = str(e.ROOT / ".venv/bin/python")
    code = (
        "import json,sys;from pathlib import Path;"
        "sys.path[:0]=[sys.argv[1]+'/python',sys.argv[1]];"
        "from carnot.verify.exact_request_8188 import Cache;"
        "raise SystemExit(Cache(Path(sys.argv[2])).read(json.loads(Path(sys.argv[3]).read_text())) is None)"
    )
    print("[8188-test] before cache child restart completed=0 pending=1", flush=True)
    child = subprocess.run(
        [py, "-u", "-c", code, str(e.ROOT), str(tmp_path / "evidence_cache"), str(path)],
        cwd=tmp_path,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
        timeout=60,
    )
    print("[8188-test] after cache child restart completed=1 pending=0", flush=True)
    assert child.returncode == 0
    for field in (
        "model_gguf",
        "model_revision",
        "rendered_prompt",
        "seed",
        "grammar",
        "answer_bytes",
        "tokenizer",
        "runtime",
        "request_schema",
        "generation_parameters",
    ):
        changed = deepcopy(operand)
        changed[field] = "changed"
        assert e.Cache(tmp_path / "evidence_cache").read(changed) is None
    receipts = [
        dict(name=n, passed=True, normal_exit=True, actual_exit=0) for n in e.REQUIRED_CHECK_NAMES
    ]
    zero = dict(e.prior.shared.prior.ZERO_INVOCATION_COUNTS)
    monkeypatch.setattr(
        e.prior.shared.Ledger,
        "counts",
        lambda s: dict(
            zero,
            generation_calls_attempted=74,
            generation_calls_completed=74,
            model_loads_attempted=1,
            model_loads_completed=1,
        ),
    )
    value = e.build(
        data,
        dict(work=work, ledger=[], gpu_lease_receipt=dict(private_fixture=True)),
        tmp_path,
        receipts,
        "20261006",
        11,
        False,
    )
    assert value["cached_service_ready_score"] == 1 and value["verdict_class"] == "positive"
    output = tmp_path / (e.NAME + ".json")
    assert not output.exists()
    e.atomic_json(output, value)
    assert not e.replay(output)
    with pytest.raises(SystemExit) as error:
        runner.main(["--fixture-e2e", str(e.ROOT / "results" / output.name)])
    assert error.value.code == 2


def test_call_identity_reconciliation(data, native, tmp_path):
    """REQ-VERIFY-8188-REPLAY: cached hits cannot invent current model calls."""
    panel = deepcopy(data)
    panel["slots"] = panel["slots"][:2]
    ledger = e.prior.shared.Ledger(tmp_path / "ledger.json")
    work = e.measure(panel, e.prior.shared.prior.FixtureRuntime(), native, tmp_path, ledger)
    assert e.validate_ledger(work, ledger.rows)
    for field in ("request_sha256", "response_sha256", "started_monotonic_ns", "input_tokens"):
        bad = deepcopy(ledger.rows)
        bad[0][field] = "forged"
        assert not e.validate_ledger(work, bad)
    assert not e.validate_ledger(work, ledger.rows[:-1])
    assert not e.validate_ledger(work, ledger.rows + ledger.rows)
    assert e.validate_ledger({}, [])
    load = dict(operation="model_load", call_id="load", status="completed")
    assert e.validate_ledger(work, ledger.rows + [load])
    assert not e.validate_ledger(work, ledger.rows + [load, load])
