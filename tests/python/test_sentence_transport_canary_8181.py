"""REQ-REPORT-8181 / REQ-VERIFY-8181: private paired transport custody."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import sentence_transport_8179 as transport
from carnot.verify import sentence_transport_canary_8181 as e


def source(i=0, sentences=2):
    """Keep fixture bytes private and independent of real development labels."""
    return dict(
        unit_id=f"u{i}",
        source_cluster_id=f"s{i}",
        role="fit",
        source_bytes=b"Source evidence. Second evidence.".hex(),
        answer_bytes=("A claim. " * sentences).encode().hex(),
    )


class Runtime:
    """Provide scripted transport only; it cannot earn a live invocation."""

    def __init__(self, mode="ok"):
        self.mode = mode
        self.worker = self

    def post_json(self, endpoint, payload, timeout):
        if endpoint == "/apply-template":
            return dict(prompt=payload["messages"][0]["content"])
        assert endpoint == "/tokenize"
        return dict(tokens=list(range(6001 if self.mode == "overflow" else 30)))

    def generate(self, payload):
        indices = json.loads(payload["messages"][0]["content"])["answer_sentence_indices"]
        text = "\n".join(f"{i}|E|0.20|[0]" for i in indices)
        if self.mode == "partial":
            text = text.split("\n")[0]
        if self.mode == "index":
            text = text.replace("[0]", "[99]")
        return dict(
            choices=[
                dict(
                    message=dict(content=text),
                    finish_reason="length" if self.mode == "length" else "stop",
                )
            ],
            usage=dict(prompt_tokens=30, completion_tokens=20),
        )


def slots(n=24):
    rows = [dict(source(i), **transport.requests(source(i), lambda _: 30)) for i in range(n)]
    config = dict(
        source_ids=[r["unit_id"] for r in rows],
        arm_order=[dict(unit_id=r["unit_id"], arms=["grammar", "unconstrained"]) for r in rows],
    )
    return e.select(rows, config)


def test_paired_capture_and_reduction(tmp_path):
    """SCENARIO-VERIFY-8181-PAIRED: count sources, preserve paired prompts."""
    chosen = slots()
    rows = e.capture(chosen, Runtime(), tmp_path, dict(fixture=True))
    reduced = e.reduce(chosen, rows)
    assert reduced["completed_count"] == 24
    assert reduced["grammar_complete_sources"] == 24
    assert len(rows) == 48 and len(reduced["rows"]) == 48
    assert rows[0]["rendered_prompt"] == rows[1]["rendered_prompt"]
    assert reduced["structural_loss_delta"] == 0
    bad = deepcopy(rows)
    bad[0]["response"]["choices"][0]["message"]["content"] = "0|E|0.20|[99]"
    assert e.reduce(chosen, bad)["grammar_complete_sources"] == 23
    assert all(r["human_target"] is None for r in reduced["response_schema_rows"])


@pytest.mark.parametrize("mode", ["partial", "length", "index", "overflow", "error", "choices"])
def test_failures_never_retry_or_invent_probabilities(tmp_path, mode):
    """REQ-VERIFY-8181: partial arrays and bounded failures remain failures."""
    runtime = Runtime(mode)
    if mode == "error":
        runtime.generate = lambda _: (_ for _ in ()).throw(RuntimeError("owned failure"))
    if mode == "choices":
        runtime.generate = lambda _: dict(choices=[], usage={})
    chosen = slots(1)
    rows = e.capture(chosen, runtime, tmp_path, dict(fixture=True))
    reduced = e.reduce(chosen, rows)
    assert reduced["completed_count"] == 0
    assert len(rows) == 2
    assert reduced["grammar_complete_sources"] == 0


def test_frozen_selection_and_launch_deadline(tmp_path):
    """REQ-VERIFY-8181: no label selection or calls after frozen deadline."""
    with pytest.raises(ValueError):
        e.select([], dict(source_ids=["missing"], arm_order=[]))
    chosen = slots(1)
    rows = e.capture(chosen, Runtime(), tmp_path, {}, started=-10000)
    assert all(r["status"] == "censored" for r in rows)
    assert e.reduce(chosen, rows)["censored_count"] == 1


def test_owned_validation_and_terminal_verdicts(tmp_path):
    """REQ-REPORT-8181: owned failure overrides live readiness."""
    chosen = slots()
    rows = e.capture(chosen, Runtime(), tmp_path / "calls", dict(fixture=True))
    work = e.fixture_work(chosen, rows, tmp_path)
    value = e.build(work, tmp_path, [dict(passed=True)], fixture=True)
    assert value["transport_canary_ready_score"] == 0
    assert value["verifier_is_oracle"] and value["model_invocation_counts"]["generate"] == 0
    assert value["independent_generalization_score"] == 0
    assert e.build(work, tmp_path, [dict(passed=False)])["verdict_class"] == "disqualified"
    work["checks"] = [dict(check="sentence_protocol_ready_score", passed=False, observed=0)]
    assert (
        e.build(work, tmp_path, [dict(passed=True)])["honest_verdict"]
        == "complete_blocked_sentence_protocol_ready_score"
    )


def test_missing_inputs_and_frozen_manifest(tmp_path):
    """SCENARIO-REPORT-8181-CUSTODY: preserve actual missing gate operand."""
    plan = e.inputs(tmp_path, tmp_path / "raw")
    assert any(not c["passed"] and c["observed"] is None for c in plan["checks"])
    private = tmp_path / "private"
    private.mkdir()
    specs = e.manifest(private, private / "candidate.json")
    assert e.TEST in specs["commands"][0]["argv"]
    assert all("::" not in x for s in specs["commands"][5:] for x in s["argv"])


def test_private_cli_cold_replay_and_tamper(tmp_path):
    """E2E-010/015/019: private success and failures use direct subprocesses."""
    import os
    import subprocess
    import sys

    chosen = slots()
    calls = e.capture(chosen, Runtime(), tmp_path / "calls", dict(fixture=True))
    atomic_json(tmp_path / "canary-fixture.json", dict(rows=chosen, calls=calls))
    output = tmp_path / (e.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    prefix = [sys.executable]
    if env.get("COVERAGE_RCFILE"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + env["COVERAGE_RCFILE"]]

    def run(args):
        return subprocess.run(
            prefix + [str(e.ROOT / e.CLI), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    result = run(["--root", str(tmp_path), "--fixture-output", str(output)])
    assert result.returncode == 0, result.stderr
    assert run(["--cold-replay", str(output)]).returncode == 0
    value = json.loads(output.read_text())
    value["completed_count"] = -1
    atomic_json(output, value)
    assert run(["--cold-replay", str(output)]).returncode == 1
    assert (
        run(
            [
                "--root",
                str(tmp_path / "missing"),
                "--fixture-output",
                str(tmp_path / "missing" / (e.NAME + ".json")),
            ]
        ).returncode
        == 0
    )


def test_replay_seals_logs_and_primitives(tmp_path):
    """REQ-REPORT-8181: rehashed response changes cannot evade independent rows."""
    from carnot.reporting.current_work_receipt import sha256_file

    chosen = slots(1)
    calls = e.capture(chosen, Runtime(), tmp_path / "calls", dict(fixture=True))
    raw = tmp_path / "raw"
    work = e.fixture_work(chosen, calls, raw)
    log = tmp_path / "owned.log"
    log.write_text("normal exit")
    receipts = [dict(passed=True, log_path=str(log), log_sha256=sha256_file(log))]
    value = e.build(work, raw, receipts)
    primary = tmp_path / "primary.json"
    atomic_json(primary, value)
    assert e.replay(primary)
    log.write_text("changed")
    assert not e.replay(primary)
    log.write_text("normal exit")
    original = deepcopy(work)
    for kind in ["calls", "slots", "source", "grammar"]:
        work = deepcopy(original)
        if kind == "calls":
            work["calls"][0]["response"]["choices"][0]["message"]["content"] = "0|B|0.90|[]"
        elif kind == "slots":
            work["slots"][0]["arms"] = ["unconstrained", "grammar"]
        elif kind == "source":
            work["slots"][0]["source_bytes"] = b"Changed evidence.".hex()
            atomic_json(raw / "source_plan.json", dict(rows=work["slots"]))
            work["raw_shard_hashes"][1] = e.reference(raw / "source_plan.json")
        else:
            work["slots"][0]["requests"][0]["grammar"] = "changed grammar"
            atomic_json(raw / "source_plan.json", dict(rows=work["slots"]))
            work["raw_shard_hashes"][1] = e.reference(raw / "source_plan.json")
            # Both rehashed shards agree, so replay must rebuild the grammar from
            # source bytes rather than reject only the copied response shard.
            atomic_json(raw / "primitive_calls.json", dict(rows=work["calls"]))
            work["raw_shard_hashes"][0] = e.reference(raw / "primitive_calls.json")
        atomic_json(raw / "measurement.json", work)
        atomic_json(primary, e.build(work, raw, receipts))
        assert not e.replay(primary)
    Path(value["raw_shard_hashes"][0]["path"]).write_text("tampered response bytes")
    atomic_json(primary, value)
    assert not e.replay(primary)
    assert not e.replay(tmp_path / "absent")


def test_live_measure_boundary_and_readiness(tmp_path, monkeypatch):
    """REQ-VERIFY-8181: only real owned loads and sufficient yield grant readiness."""
    chosen = slots()
    calls = e.capture(chosen, Runtime(), tmp_path / "calls", {})
    plan = dict(slots=chosen, checks=[], refs=[], upstream=[], identity={}, grammar_receipt={})
    monkeypatch.setattr(e, "inputs", lambda root, raw: deepcopy(plan))
    monkeypatch.setattr(e, "preflight", lambda plan, raw: None)
    result = dict(
        rows=calls,
        checks=[],
        model_loads_attempted=1,
        model_loads_completed=1,
        runtime_receipts=[dict(input_tokens=30, output_tokens=20)] * len(calls),
    )
    monkeypatch.setattr(e, "live", lambda plan, raw: result)
    raw = tmp_path / "work"
    work = e.measure(tmp_path, raw)
    work["duration_s"] = 12
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, raw, [dict(passed=True)])
    assert value["transport_canary_ready_score"] == 1
    assert (
        value["qualified_transport_sha256"] and value["model_invocation_counts"]["generate"] == 48
    )
    assert value["qualified_transport_configuration"]["acquisition_cost_charged_once"]
    assert (
        e.measure(tmp_path, tmp_path / "mutated", mutation="source")["checks"][-1]["passed"]
        is False
    )
    monkeypatch.setattr(e.execution, "main", lambda argv: 0)
    assert e.main([]) == 0


def test_authentication_and_external_primitive_tamper(tmp_path, monkeypatch):
    """REQ-REPORT-8181: source custody is authenticated before model preflight."""
    from carnot.reporting.current_work_receipt import sha256_file

    chosen = slots()
    protocol = tmp_path / e.methods.PROTOCOL
    config = dict(
        canary=dict(
            source_ids=[r["unit_id"] for r in chosen],
            arm_order=[dict(unit_id=r["unit_id"], arms=r["arms"]) for r in chosen],
        )
    )
    atomic_json(protocol, config)
    primitive = tmp_path / "requests.json"
    atomic_json(primitive, dict(rows=chosen))
    terminal = tmp_path / "terminal.json"
    atomic_json(terminal, dict(publication=dict(sidecar_path="unused")))
    value = dict(
        sentence_protocol_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        terminal_validation_sidecar_path=str(terminal),
        measurement_reference=e.reference(primitive),
        raw_shard_hashes=[e.reference(primitive)],
        source_manifest=dict(fit=e.reference(primitive)),
        tokenizer_receipt={},
    )
    path = tmp_path / e.UPSTREAM
    atomic_json(path, value)
    monkeypatch.setattr(e, "PIN", sha256_file(path))
    monkeypatch.setattr(e.methods, "PROTOCOL_PIN", sha256_file(protocol))
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=True)))
    plan = e.inputs(tmp_path, tmp_path / "raw")
    assert len(plan["slots"]) == 24 and all(c["passed"] for c in plan["checks"])
    primitive.write_text("changed")
    assert any(not c["passed"] for c in e.inputs(tmp_path, tmp_path / "tamper")["checks"])
    value["terminal_validation_sidecar_path"] = str(tmp_path / "absent")
    atomic_json(path, value)
    monkeypatch.setattr(e, "PIN", sha256_file(path))
    assert e.inputs(tmp_path, tmp_path / "malformed")["checks"][-1]["check"] == "upstream_structure"


def test_CPU_grammar_preflight_and_blocking(tmp_path, monkeypatch):
    """REQ-VERIFY-8181: CPU compilation precedes any CUDA ownership."""
    import llama_cpp
    from carnot.reporting.current_work_receipt import sha256_file

    model = tmp_path / "revision" / "model.gguf"
    model.parent.mkdir()
    model.write_bytes(b"private vocabulary")
    receipt = dict(path=str(model), sha256=sha256_file(model))
    plan = dict(checks=[], slots=slots(1), tokenizer_receipt=receipt, refs=[])
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    monkeypatch.setattr(
        e.qualified.legacy,
        "cached_current_model",
        lambda: dict(model_path=str(model), hf_id=e.MODEL_SPECS[0]),
    )

    class Vocab:
        metadata = {"tokenizer.chat_template": "enable_thinking supported"}
        _model = SimpleNamespace(vocab="private")

        def tokenize(self, data, **kwargs):
            return [0] * 30

        def close(self):
            assert True

    monkeypatch.setattr(llama_cpp, "Llama", lambda **kwargs: Vocab())
    monkeypatch.setattr(llama_cpp.llama_cpp, "llama_sampler_init_grammar", lambda *args: 1)
    monkeypatch.setattr(llama_cpp.llama_cpp, "llama_sampler_free", lambda *args: None)
    e.preflight(plan, tmp_path)
    assert plan["grammar_receipt"]["compiled_requests"] == 1
    assert all(c["passed"] for c in plan["checks"])
    monkeypatch.setattr(llama_cpp.llama_cpp, "llama_sampler_init_grammar", lambda *args: None)
    bad = dict(checks=[], slots=slots(1), tokenizer_receipt=receipt, refs=[])
    e.preflight(bad, tmp_path)
    assert bad["checks"][-1]["check"] == "runtime_grammar_support"
    model.unlink()
    bad = dict(checks=[], slots=slots(1), tokenizer_receipt=receipt, refs=[])
    e.preflight(bad, tmp_path)
    assert any(not c["passed"] for c in bad["checks"])


def test_live_adapter_reuses_owned_runtime(tmp_path, monkeypatch):
    """REQ-VERIFY-8181: the qualified lease runner receives the frozen schedule."""
    from carnot.reporting.current_work_receipt import canonical_hash

    plan = dict(
        slots=slots(1),
        identity=dict(chat_template_sha256=canonical_hash("template")),
        started=e.time.monotonic(),
    )

    class Model:
        def __init__(self, *args):
            assert args

        def load(self):
            return dict(props=dict(chat_template="template"))

    monkeypatch.setattr(e.qualified.legacy, "QwenRuntime", Model)

    def run(plan, raw, private):
        assert e.qualified.legacy.QwenRuntime("model").load()["started_monotonic_ns"] > 0
        e.qualified.legacy.progress("heartbeat", 0)
        frozen = e.qualified.legacy.capture.freeze({})
        calls = e.qualified.legacy.capture.capture(frozen, Runtime(), raw, {}, started=0)
        return dict(rows=calls, checks=[dict(upstream_id="runtime", field="offload", passed=True)])

    monkeypatch.setattr(e.qualified.legacy, "live_capture", run)
    result = e.live(plan, tmp_path)
    assert len(result["rows"]) == 2 and result["checks"][0]["check"] == "runtime_offload"
    plan["identity"]["chat_template_sha256"] = "wrong"
    with pytest.raises(ValueError, match="served_chat_template_drift"):
        e.live(plan, tmp_path)


def test_freeze_worker_deadline_before_subprocess(tmp_path, monkeypatch):
    """REQ-REPORT-8181: required argv and deadlines are sealed before measurement."""
    atomic_json(tmp_path / "validation_commands.json", {})
    monkeypatch.setattr(e, "BASE_RUN_CHECK", lambda *args, **kwargs: dict(passed=True))
    spec = dict(name="measurement", deadline_s=180)
    assert e.supervise(tmp_path, spec, tmp_path, tmp_path / "logs")["passed"]
    assert (
        json.loads((tmp_path / "validation_commands.json").read_text())["measurement"]["deadline_s"]
        == 3300
    )
    assert e.supervise(tmp_path, dict(name="unit"), tmp_path, tmp_path / "logs")["passed"]
