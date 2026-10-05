"""REQ-VERIFY-8153 / REQ-REPORT-8153: fixed capture without outcome selection."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.verify import fit_evidence_capture_8153 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify.qwen_development_capture_7995 import Ledger


def public():
    """Keep fixture identity fixed independently of its probability or targets."""
    rows = []
    for role, n in e.ROLES.items():
        for i in range(n):
            source = f"é fact {role} {i}"
            for order, arm in enumerate(("holistic", "source_span")):
                prompt = e.protocol.protocol()["prompt_prefixes"][arm] + json.dumps(
                    dict(source=source, answer="fact"), ensure_ascii=False
                )
                rows.append(
                    dict(
                        unit_id=f"{role}-{i}",
                        source_cluster_id=f"sha256:{i + (128 if role == 'tune' else 0):064x}",
                        role=role,
                        arm=arm,
                        order=order,
                        prompt=prompt,
                        prompt_sha256=canonical_hash(prompt),
                        source_bytes=source.encode().hex(),
                        answer_bytes=b"fact".hex(),
                        max_tokens=128,
                        status="frozen_not_invoked",
                        entailment_label=None,
                    )
                )
    return rows


class Runtime:
    def count(self, text):
        return 3

    def generate(self, request):
        return dict(
            choices=[
                dict(
                    message=dict(
                        content='{"p_hallucination":0.2,"quote":"wrong","byte_start":0,"byte_end":5}'
                    ),
                    finish_reason="stop",
                )
            ],
            usage=dict(prompt_tokens=3, completion_tokens=23),
        )

    def close(self):
        return dict(leak_free=True)


def test_fixed_identity_and_unlabeled_interventions():
    """SCENARIO-VERIFY-8153 freezes fixed donors without ranking outcomes."""
    rows = public()
    slots = e.freeze(rows)
    assert len(slots) == 432
    assert slots[256]["slot"] == 1 and slots[382]["slot"] == 64
    assert [r["role"] for r in slots[:384]].count("fit") == 256
    assert all(r["human_target"] is None for r in slots)
    assert slots[384]["request"] == slots[0]["request"]
    assert json.loads(slots[385]["prompt"].split("Input JSON follows:\n")[1])["source"] == ""
    assert slots[386]["source_bytes"] == slots[2]["source_bytes"]
    assert slots[-1]["source_bytes"] == slots[0]["source_bytes"]
    for change in ("prompt_sha256", "role", "order"):
        bad = deepcopy(rows)
        bad[0][change] = "changed"
        with pytest.raises(ValueError):
            e.freeze(bad)
    with pytest.raises(ValueError):
        e.freeze(rows[:-1])


def test_capture_parser_pair_support_and_timeout(tmp_path):
    """REQ-VERIFY-8153 quote failure retains probability and source independence."""
    slots = e.freeze(public())
    ledger = Ledger(tmp_path / "ledger.json")
    rows = e.capture(slots, Runtime(), tmp_path / "slots", "identity", ledger=ledger)
    reduced = e.reduce(rows)
    assert reduced["completed_count"] == 192
    assert reduced["pair_support"] == dict(fit=128, tune=64)
    assert all(r["valid_quote"] == 0 for r in reduced["quote_validity_rows"])
    assert len(rows) == 432 and ledger.counts()["generation_calls_completed"] == 432
    labels = {r["unit_id"]: i % 2 for i, r in enumerate(reduced["rows"])}
    assert e.trainability(reduced["rows"], labels)["fit_trainable_score"] == 1
    assert e.trainability(reduced["rows"], {})["fit_trainable_score"] == 0
    bad = deepcopy(rows)
    bad[0]["probability"] = 0.9
    with pytest.raises(ValueError):
        e.reduce(bad)
    for method in ("count", "generate"):

        class Failing(Runtime):
            def count(self, text):
                if method == "count":
                    raise ValueError("tokenizer")
                return 6001 if "é fact fit 0" in text else 3

            def generate(self, request):
                raise TimeoutError("owned_deadline")

        failed_ledger = Ledger(tmp_path / (method + ".json"))
        failed = e.capture(slots, Failing(), tmp_path / method, "id", ledger=failed_ledger)
        assert e.reduce(failed)["completed_count"] == 0
        assert failed[0]["status"] in ("excluded", "failed")
        assert all(not r["started"] for r in failed if r["unit_id"] == "fit-0")
        if method == "generate":
            assert failed_ledger.rows[0]["status"] == "cancelled"
            assert failed_ledger.rows[0]["output_tokens"] is None
    cutoff = e.capture(
        slots,
        Runtime(),
        tmp_path / "cutoff",
        "id",
        ledger=Ledger(tmp_path / "cutoff.json"),
        started=0,
    )
    assert all(r["status"] == "censored" for r in cutoff)


def world(tmp_path):
    """Seal private custody files without modifying any production result."""
    root = tmp_path / "root"
    raw = tmp_path / "upstream"
    raw.mkdir(parents=True)
    captured = raw / "manifest.json"
    atomic_json(captured, dict(rows=public()))
    refs = []
    for name in ("methods", "fit", "tune"):
        path = raw / (name + ".json")
        atomic_json(path, {})
        refs.append(e.reference(path))
    value = dict(
        source_protocol_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        capture_manifest=e.reference(captured),
        method_freeze=refs[0],
        source_role_manifests=dict(fit=refs[1], tune=refs[2]),
        expected_runtime_identity={},
    )
    atomic_json(root / e.UPSTREAM, value)
    return root


def test_inputs_blocks_and_shard_authentication(tmp_path, monkeypatch):
    """REQ-REPORT-8153 names exact operands and preserves the original upstream."""
    root = world(tmp_path)
    raw = tmp_path / "owned"
    raw.mkdir()
    assert len(e.inputs(root, raw, fixture=True)["slots"]) == 432
    path = root / e.UPSTREAM
    v = json.loads(path.read_text())
    original = path.read_bytes()
    v["source_protocol_ready_score"] = 0
    atomic_json(path, v)
    plan = e.inputs(root, raw, fixture=True)
    assert (
        next(r for r in plan["checks"] if r["check"] == "source_protocol_ready_score")["observed"]
        == 0
    )
    v["source_protocol_ready_score"] = 1
    v["capture_manifest"]["sha256"] = "changed"
    atomic_json(path, v)
    assert e.inputs(root, raw, fixture=True)["slots"] == []
    path.write_text("{invalid")
    assert any(
        r["check"] == "authenticated_source_inputs"
        for r in e.inputs(root, raw, fixture=True)["checks"]
    )
    path.write_bytes(original)
    v = json.loads(path.read_text())
    terminal = tmp_path / "terminal.json"
    atomic_json(terminal, dict(publication=dict(sidecar_path=str(tmp_path / "sidecar.json"))))
    v["terminal_validation_sidecar_path"] = str(terminal)
    atomic_json(path, v)
    atomic_json(tmp_path / "sidecar.json", {})
    monkeypatch.setattr(e, "PIN", e.reference(path)["sha256"])
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=True)))
    assert len(e.inputs(root, raw)["slots"]) == 432
    assert e.inputs(tmp_path / "missing", raw)["slots"] == []


def test_owned_build_replay_and_private_cli(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8153 real external scripts cover success, block and tamper."""
    import os
    import subprocess
    import sys

    root = world(tmp_path)
    raw = tmp_path / "raw-work"
    work = e.measure(root, raw, fixture=True)
    value = e.build(work, raw, [dict(passed=True)], fixture=True)
    assert value["fit_capture_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert e.build(work, raw, [dict(passed=False)], fixture=True)["verdict_class"] == "disqualified"
    failed = deepcopy(work)
    failed["plan"]["checks"][0]["passed"] = False
    assert e.build(failed, raw, [dict(passed=True)], fixture=True)["verdict_class"] == "blocked"
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    for field in ("completed_count", "fit_capture_ready_score"):
        v = deepcopy(value)
        v[field] += 1
        atomic_json(output, v)
        assert not e.replay(output)
    v = deepcopy(value)
    v["code_config_hashes"][e.MODULE] = "changed"
    atomic_json(output, v)
    assert not e.replay(output)
    atomic_json(output, value)
    shard = Path(work["raw_shard_hashes"][1]["path"])
    old = shard.read_bytes()
    shard.chmod(0o600)
    shard.write_text("{}")
    assert not e.replay(output)
    shard.write_bytes(old)
    assert not e.replay(tmp_path / "absent")
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)

    def run(*args):
        cmd = [sys.executable]
        if env.get("COVERAGE_RCFILE"):
            cmd += ["-m", "coverage", "run", "--rcfile=" + env["COVERAGE_RCFILE"]]
        print("before private subprocess", flush=True)
        result = subprocess.run(
            [*cmd, str(e.ROOT / e.CLI), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print("after private subprocess exit=" + str(result.returncode), flush=True)
        return result

    args = ["--root", str(root), "--fixture-output", str(output)]
    done = run(*args)
    assert done.returncode == 0, done.stdout + done.stderr
    assert run("--cold-replay", str(output)).returncode == 0
    v = json.loads(output.read_text())
    v["completed_count"] += 1
    atomic_json(output, v)
    assert run("--cold-replay", str(output)).returncode == 1
    assert run(*args, "--mutation", "source").returncode == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    args[1] = str(tmp_path / "absent")
    assert run(*args).returncode == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    assert run("--date", "20261004").returncode == 2
    assert run("--fixture-output", str(e.ROOT / "results/private.json")).returncode == 2


def test_runtime_freeze_and_one_owned_load(tmp_path, monkeypatch):
    """REQ-VERIFY-8153 freezes the build before one load and keeps failed loads."""
    import struct

    template = "frozen template"
    model = tmp_path / "revision" / "model.gguf"
    model.parent.mkdir()
    model.write_bytes(struct.pack("<Q", len(template)) + template.encode())
    binary = tmp_path / "llama-server"
    binary.write_text("private binary")
    binary.chmod(0o700)
    plan = dict(
        checks=[],
        refs=[],
        slots=e.freeze(public()),
        manifests={},
        capture_identity="fixed",
        protocol=dict(
            model_path=str(model),
            native_binary=dict(path=str(binary)),
            model_revision="revision",
            gguf_sha256=e.reference(model)["sha256"],
            runtime_sha256=e.reference(binary)["sha256"],
            chat_template_sha256=canonical_hash(template),
        ),
    )
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    monkeypatch.setattr(
        e.legacy,
        "cached_current_model",
        lambda: dict(hf_id=e.MODEL_SPECS[0], model_path=str(model)),
    )
    monkeypatch.setattr(
        e.legacy,
        "read_gguf_metadata",
        lambda _: dict(
            field_provenance=dict(metadata_keys={"tokenizer.chat_template": dict(value_offset=0)})
        ),
    )
    log = tmp_path / "cuda.log"
    log.write_text("CUDA private fixture")
    monkeypatch.setattr(
        e.execution,
        "run_check",
        lambda *a, **k: dict(
            passed=True,
            output_tail="CUDA",
            log_path=str(log),
            log_sha256=e.reference(log)["sha256"],
        ),
    )
    e.runtime_preflight(plan, tmp_path, tmp_path)
    assert all(r["passed"] for r in plan["checks"])
    assert plan["runtime_freeze"]["tokenizer"] == "embedded_GGUF"
    assert (tmp_path / "runtime_freeze.json").is_file()
    no_live = deepcopy(plan)
    no_live["checks"] = []
    monkeypatch.delenv("CARNOT_FORCE_LIVE")
    e.runtime_preflight(no_live, tmp_path, tmp_path)
    assert not no_live["checks"][-1]["passed"]
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

    class FakeRuntime:
        command = ["private"]

        def __init__(self, model, scratch, gpu):
            self.model = model

        def load(self):
            return dict(props=dict(chat_template=template))

        def count(self, text):
            return 3

        def generate(self, request):
            return Runtime().generate(request)

    def worker(p, raw, scratch):
        runtime = e.legacy.QwenRuntime(model, scratch, 0)
        e.legacy.progress("private_owned_heartbeat", 0)
        result = dict(rows=[], checks=[dict(upstream_id="private", field="fixture", passed=True)])
        try:
            result["identity"] = runtime.load()
            result["rows"] = e.legacy.capture.capture(
                p["slots"],
                runtime,
                raw / "calls",
                p["capture_identity"],
                started=e.time.monotonic(),
            )
        except (ValueError, TimeoutError):
            pass
        return result

    monkeypatch.setattr(e.legacy, "QwenRuntime", FakeRuntime)
    monkeypatch.setattr(e.legacy, "live_capture", worker)
    result = e.live(plan, tmp_path / "live", tmp_path)
    assert result["ledger"][0]["operation"] == "model_load"
    assert len(result["rows"]) == 432
    plan["protocol"]["chat_template_sha256"] = "changed"
    failed = e.live(plan, tmp_path / "bad-template", tmp_path)
    assert failed["ledger"][0]["status"] == "failed"
    monkeypatch.setattr(
        FakeRuntime, "load", lambda _: (_ for _ in ()).throw(TimeoutError("deadline"))
    )
    failed = e.live(plan, tmp_path / "timeout", tmp_path)
    assert failed["ledger"][0]["status"] == "failed"


def test_probability_usage_failure_and_ledger_readiness(tmp_path):
    """REQ-VERIFY-8153 records every call even when probability or usage fails."""
    slots = e.freeze(public())

    class Invalid(Runtime):
        def generate(self, request):
            if "source_span" == next(r["arm"] for r in slots if r["request"] == request):
                return dict(
                    choices=[dict(message=dict(content="invalid"))],
                    usage=dict(prompt_tokens=3, completion_tokens=128),
                )
            raise RuntimeError("transport")

    calls = e.capture(
        slots,
        Invalid(),
        tmp_path / "invalid",
        "fixed",
        ledger=Ledger(tmp_path / "invalid-ledger.json"),
    )
    assert e.reduce(calls)["failed_count"] == 192
    assert all(r["status"] == "failed" for r in calls)


def test_production_measure_and_normal_main(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8153 frozen owned commands run normally before publication."""
    root = world(tmp_path)
    original_inputs = e.inputs
    monkeypatch.setattr(
        e, "inputs", lambda root, raw, **k: original_inputs(root, raw, fixture=True)
    )
    monkeypatch.setattr(e, "runtime_preflight", lambda *a: None)

    def worker(plan, raw, private):
        ledger = Ledger(raw / "ledger.json")
        calls = e.capture(
            plan["slots"], Runtime(), raw / "calls", plan["capture_identity"], ledger=ledger
        )
        return dict(rows=calls, checks=[], ledger=ledger.rows)

    monkeypatch.setattr(e, "live", worker)

    class Binder:
        refs = []

        def read(self, path, pin):
            assert (self.raw / "primitive_calls.json").is_file()
            return dict(
                evaluator_label_manifests={r: dict(path=r, sha256="private") for r in e.ROLES}
            )

        def __init__(self, raw):
            self.raw = raw

    monkeypatch.setattr(e.source.qualified, "Custody", Binder)
    monkeypatch.setattr(
        e.source.qualified,
        "read_ref",
        lambda b, ref: dict(
            rows=[dict(unit_id=f"{ref['path']}-{i}", y=i % 2) for i in range(e.ROLES[ref["path"]])]
        ),
    )
    monkeypatch.setattr(
        e.source.qualified,
        "upstream",
        lambda *a: pytest.fail("historical code must not be rebound to current mutable files"),
    )
    work = e.measure(root, tmp_path / "production")
    assert len(work["result"]["rows"]) == 432
    log = tmp_path / "normal.log"
    log.write_text("private normally exited commands")
    monkeypatch.setattr(
        e.execution,
        "run_check",
        lambda *a, **k: dict(passed=True, log_path=str(log), log_sha256=e.reference(log)["sha256"]),
    )
    monkeypatch.setattr(e, "measure", lambda *a, **k: work)
    monkeypatch.setattr(e, "build", lambda *a, **k: dict(required_checks_passed=True))
    monkeypatch.setattr(e.execution, "publish", lambda *a: None)
    assert e.main(["--root", str(root), "--output", str(tmp_path / "normal.json")]) == 0


def test_empty_denominator_and_request_tamper(tmp_path):
    """REQ-REPORT-8153 missing custody retains all192 source slots."""
    empty = e.reduce([])
    assert empty["excluded_count"] == len(empty["rows"]) == 192
    assert empty["independent_count"] == 0
    slots = e.freeze(public())
    calls = e.capture(
        slots, Runtime(), tmp_path / "tamper", "fixed", ledger=Ledger(tmp_path / "tamper.json")
    )
    calls[0]["request"]["messages"][0]["content"] = "changed prompt"
    with pytest.raises(ValueError, match="request_identity"):
        e.reduce(calls)


def test_no_cuda_readiness_and_ledger_tamper(tmp_path):
    """REQ-VERIFY-8153 a generation counter alone cannot establish live CUDA."""
    root = world(tmp_path)
    raw = tmp_path / "cuda-readiness"
    work = e.measure(root, raw, fixture=True)
    value = e.build(work, raw, [dict(passed=True)])
    assert value["fit_capture_ready_score"] == 0
    work["result"]["ledger"][0]["request_sha256"] = "changed"
    with pytest.raises(ValueError, match="ledger_request_binding"):
        e.build(work, raw, [dict(passed=True)])


def test_closure_reserves_child_cleanup_and_no_launch(tmp_path, monkeypatch):
    """REQ-VERIFY-8153 reserve cleanup time rather than launch near hard closure."""
    slots = e.freeze(public())
    monkeypatch.setattr(e.time, "monotonic", lambda: 2960)
    rows = e.capture(
        slots,
        Runtime(),
        tmp_path / "closure",
        "fixed",
        ledger=Ledger(tmp_path / "closure.json"),
        started=0,
    )
    assert all(not r["started"] and r["status"] == "censored" for r in rows)


def test_late_admission_and_response_receipt_tamper(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8153 counts tokens before cutoff but never launches late."""
    slots = e.freeze(public())
    now = [0]

    class Late(Runtime):
        def count(self, text):
            now[0] = 3000
            return 3

    monkeypatch.setattr(e.time, "monotonic", lambda: now[0])
    calls = e.capture(
        slots, Late(), tmp_path / "late", "fixed", ledger=Ledger(tmp_path / "late.json"), started=0
    )
    assert all(not r["started"] for r in calls)
    monkeypatch.undo()
    root = world(tmp_path)
    raw = tmp_path / "receipts"
    work = e.measure(root, raw, fixture=True)
    work["result"]["ledger"][0]["response_sha256"] = "changed"
    with pytest.raises(ValueError, match="ledger_response_binding"):
        e.build(work, raw, [dict(passed=True)])


def test_log_and_primitive_cold_replay_binding(tmp_path):
    """SCENARIO-REPORT-8153 rejects tampered logs and inconsistent sealed calls."""
    root = world(tmp_path)
    raw = tmp_path / "bound"
    work = e.measure(root, raw, fixture=True)
    log = tmp_path / "log.txt"
    log.write_text("normal validation exit0")
    receipts = [dict(passed=True, log_path=str(log), log_sha256=e.reference(log)["sha256"])]
    value = e.build(work, raw, receipts, fixture=True)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    log.write_text("changed")
    assert not e.replay(output)
    log.write_text("normal validation exit0")
    work["result"]["rows"][0]["input_tokens"] = 999
    atomic_json(raw / "measurement.json", work)
    value["measurement_reference"] = e.reference(raw / "measurement.json")
    atomic_json(output, value)
    assert not e.replay(output)


def test_blocked_slots_never_invoke_runtime(tmp_path):
    """REQ-REPORT-8153 blocked prerequisites retain requests without fake calls."""
    slots = e.freeze(public())
    ledger = Ledger(tmp_path / "blocked.json")

    class Unavailable(Runtime):
        def count(self, text):
            pytest.fail("blocked tokenizer invoked")

        def generate(self, request):
            pytest.fail("blocked model invoked")

    calls = e.capture(
        slots,
        Unavailable(),
        tmp_path / "blocked",
        "fixed",
        ledger=ledger,
        blocked_reason="CARNOT_FORCE_LIVE",
    )
    assert all(
        r["status"] == "excluded" and r["exclusion_reason"] == "CARNOT_FORCE_LIVE" for r in calls
    )
    assert ledger.rows == [] and e.reduce(calls)["excluded_count"] == 192
