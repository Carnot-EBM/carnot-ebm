"""REQ-VERIFY-8167 / REQ-REPORT-8167: private source transport and custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import fit_sentence_capture_8167 as e
from carnot.verify.qwen_development_capture_7995 import Ledger


def public():
    """Create disjoint original slots; no labels enter the model-visible view."""
    return [
        dict(
            unit_id=f"{role}-{i}",
            family_id=f"{role}-{i}",
            role=role,
            slot=i + 1,
            source_cluster_id=f"{role}-cluster-{i}",
            source_bytes=f"é fact {role} {i}.".encode().hex(),
            answer_bytes=b"Fact one. Fact two. Fact three. Fact four. Fact five.".hex(),
        )
        for role, n in e.ROLES.items()
        for i in range(n)
    ]


class Runtime:
    """A scripted transport exercises fields without earning live model credit."""

    def count(self, text):
        return 30

    def generate(self, request):
        payload = json.loads(request["messages"][0]["content"])
        source = payload["source"]
        rows = [
            dict(
                sentence_index=r["sentence_index"],
                p_unsupported=0.2,
                relation="entailed",
                quote=source or None,
                byte_start=0 if source else None,
                byte_end=len(source.encode()) if source else None,
            )
            for r in payload["sentences"]
        ]
        return dict(
            choices=[dict(message=dict(content=json.dumps(rows)), finish_reason="stop")],
            usage=dict(prompt_tokens=30, completion_tokens=120),
        )

    def close(self):
        return dict(leak_free=True)


def world(tmp_path):
    """Seal public fixture manifests under a private root, outside results."""
    root = tmp_path / "root"
    refs = {}
    for role in e.ROLES:
        selected = [r for r in public() if r["role"] == role]
        path = tmp_path / (role + ".json")
        atomic_json(
            path,
            dict(
                roster=[
                    {
                        k: v
                        for k, v in r.items()
                        if k not in ("family_id", "source_bytes", "answer_bytes")
                    }
                    for r in selected
                ],
                request_rows=[
                    {k: r[k] for k in ("family_id", "source_bytes", "answer_bytes")}
                    for r in selected
                ],
            ),
        )
        refs[role] = e.reference(path)
    path = tmp_path / "historical.json"
    atomic_json(path, dict(rows=[dict(r, condition="original", arm="holistic") for r in public()]))
    historical = dict(
        required_checks_passed=True,
        flagged_adversarial=False,
        capture_manifest=e.reference(path),
        plan=dict(protocol={}, manifests=refs),
        raw_shard_hashes=[],
        MODEL_SPECS=e.MODEL_SPECS,
        model_invocation_counts={},
    )
    atomic_json(root / e.HISTORICAL, historical)
    protocol = dict(
        required_checks_passed=True,
        flagged_adversarial=False,
        sentence_protocol_ready_score=1,
        source_manifest=refs,
        protocol_path=str(e.ROOT / e.methods.PROTOCOL),
        protocol_sha256=e.methods.PROTOCOL_HASH,
    )
    atomic_json(root / e.UPSTREAM, protocol)
    return root


def captured(tmp_path, runtime=None, **kwargs):
    slots = e.freeze(public(), {r["unit_id"]: r for r in public()})
    ledger = Ledger(tmp_path / "ledger.json")
    rows = e.capture(
        slots, runtime or Runtime(), tmp_path / "slots", "identity", ledger=ledger, **kwargs
    )
    return rows, ledger


def test_order_diagnostics_and_inputs(tmp_path):
    """SCENARIO-VERIFY-8167-TRANSPORT keeps roles, byte identity and fixed donors."""
    rows = public()
    slots = e.freeze(rows, {r["unit_id"]: r for r in rows})
    assert len(slots) == 208
    assert [r["unit_id"] for r in slots[:192]] == [r["unit_id"] for r in rows]
    assert slots[192]["source_bytes"] == ""
    assert slots[193]["source_bytes"] == rows[1]["source_bytes"]
    assert all(r["human_target"] is None for r in slots)
    for key in ("role", "slot", "source_cluster_id"):
        bad = deepcopy(rows)
        bad[0][key] = rows[1][key] if key == "source_cluster_id" else "changed"
        with pytest.raises(ValueError):
            e.freeze(bad, {})
    with pytest.raises(ValueError):
        e.freeze(rows[:-1], {})
    root = world(tmp_path)
    plan = e.inputs(root, tmp_path / "owned", fixture=True)
    assert len(plan["slots"]) == 208 and all(r["passed"] for r in plan["checks"])
    path = root / e.UPSTREAM
    value = json.loads(path.read_text())
    value["sentence_protocol_ready_score"] = 0
    atomic_json(path, value)
    blocked = e.inputs(root, tmp_path / "blocked", fixture=True)
    assert (
        next(r for r in blocked["checks"] if r["check"] == "sentence_protocol_ready_score")[
            "observed"
        ]
        == 0
    )
    path.write_text("{invalid")
    assert e.inputs(root, tmp_path / "invalid", fixture=True)["slots"] == []
    assert e.inputs(tmp_path / "missing", tmp_path / "absent")["slots"] == []


def test_complete_sentences_and_support(tmp_path):
    """REQ-VERIFY-8167 counts whole sources independently of requests and sentences."""
    rows, ledger = captured(tmp_path)
    reduced = e.reduce(rows)
    assert reduced["completed_count"] == 192
    assert reduced["pair_support"] == dict(fit=128, tune=64)
    assert len(reduced["sentence_rows"]) == 960
    assert len(reduced["diagnostic_rows"]) == 16
    assert ledger.counts()["generation_calls_attempted"] == 400
    labels = {r["unit_id"]: i % 2 for i, r in enumerate(reduced["rows"])}
    assert e.trainability(reduced["rows"], labels)["fit_trainable_score"] == 1
    assert e.trainability(reduced["rows"], {})["fit_trainable_score"] == 0
    assert json.loads((tmp_path / "slots/checkpoint.json").read_text())["completed_slots"] == 208


@pytest.mark.parametrize(
    "mode",
    [
        "parser",
        "usage",
        "output_budget",
        "overlength",
        "tokenizer",
        "timeout",
        "transport",
        "censored",
    ],
)
def test_failure_dispositions(tmp_path, mode):
    """SCENARIO-VERIFY-8167-TRANSPORT keeps transport, parser and budget failures distinct."""

    class Failing(Runtime):
        attempts = 0

        def count(self, text):
            if mode == "tokenizer":
                raise ValueError("tokenizer")
            return 6001 if mode == "overlength" else 30

        def generate(self, request):
            self.attempts += 1
            if mode == "timeout":
                raise TimeoutError("uncertain")
            if mode == "transport" and self.attempts == 1:
                raise OSError("connection refused")
            response = super().generate(request)
            if mode == "parser":
                response["choices"][0]["message"]["content"] = "broken"
            if mode == "output_budget":
                response["choices"][0]["finish_reason"] = "length"
            if mode == "usage":
                response["usage"]["completion_tokens"] = 257
            return response

    runtime = Failing()
    rows, ledger = captured(tmp_path, runtime, started=0 if mode == "censored" else None)
    expected = {
        "parser": "parser_failure",
        "usage": "invalid_usage",
        "output_budget": "output_token_budget",
        "overlength": "input_token_limit",
        "tokenizer": "tokenizer_failure",
        "timeout": "transport_timeout",
        "censored": "launch_cutoff",
    }
    if mode == "transport":
        assert rows[0]["status"] == "completed"
        assert ledger.rows[0]["status"] == "failed" and ledger.rows[1]["status"] == "completed"
        assert ledger.counts()["generation_calls_attempted"] <= 400
    else:
        assert rows[0]["exclusion_reason"] == expected[mode]
        assert e.reduce(rows)["completed_count"] == 0


def test_tamper_and_missing_history(tmp_path):
    """SCENARIO-VERIFY-8167-REPLAY rejects prompt, transcript and sentence changes."""
    rows, _ = captured(tmp_path)
    for mutate in (
        lambda r: r["calls"][0].update(prompt="changed"),
        lambda r: r["calls"][0].update(transcript="[]"),
        lambda r: r.update(human_target=1),
        lambda r: r["sentences"][0].update(sentence_bytes="00"),
    ):
        bad = deepcopy(rows)
        mutate(bad[0])
        with pytest.raises(ValueError):
            e.reduce(bad)
    slots = e.freeze(public(), {})
    missing = e.capture(
        slots, Runtime(), tmp_path / "missing", "id", ledger=Ledger(tmp_path / "missing.json")
    )
    assert all(r["exclusion_reason"] == "absent_historical_source" for r in missing[:192])


def test_build_replay_and_external_cli(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8167-CLI exercises private success, external block and tampered replay."""
    root = world(tmp_path)
    work = e.measure(root, tmp_path / "work", fixture=True)
    value = e.build(work, tmp_path / "work", [dict(passed=True)], fixture=True)
    assert value["fit_capture_ready_score"] == value["fit_trainable_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert (
        e.build(work, tmp_path / "work", [dict(passed=False)], fixture=True)["verdict_class"]
        == "disqualified"
    )
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    changed = deepcopy(value)
    changed["completed_count"] += 1
    atomic_json(output, changed)
    assert not e.replay(output)
    atomic_json(output, value)
    shard = Path(value["raw_shard_hashes"][1]["path"])
    original = shard.read_bytes()
    shard.chmod(0o600)
    shard.write_text("{}")
    assert not e.replay(output)
    shard.write_bytes(original)
    changed = deepcopy(value)
    changed["code_config_hashes"][e.MODULE] = "changed"
    atomic_json(output, changed)
    assert not e.replay(output)
    monkeypatch.setattr(e.execution, "run_check", lambda *a, **k: dict(passed=True))
    for missing in (False, True):
        cli_output = tmp_path / ("blocked" if missing else "success") / (e.NAME + ".json")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        command = [
            sys.executable,
            str(e.ROOT / e.CLI),
            "--fixture-output",
            str(cli_output),
            "--root",
            str(tmp_path / "absent" if missing else root),
        ]
        rc = os.environ.get("COVERAGE_RCFILE")
        if rc:
            command[1:1] = ["-m", "coverage", "run", "--rcfile=" + rc]
        print("before private CLI", flush=True)
        child = subprocess.run(
            command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        print("after private CLI", child.returncode, flush=True)
        assert child.returncode == 0, child.stdout + child.stderr
        result = json.loads(cli_output.read_text())
        assert result["verdict_class"] == ("blocked" if missing else "circular_positive")
        command = [sys.executable, str(e.ROOT / e.CLI), "--cold-replay", str(cli_output)]
        if rc:
            command[1:1] = ["-m", "coverage", "run", "--rcfile=" + rc]
        assert (
            subprocess.run(
                command, cwd=tmp_path, env=env, capture_output=True, timeout=60
            ).returncode
            == 0
        )
        result["completed_count"] += 1
        atomic_json(cli_output, result)
        assert (
            subprocess.run(
                command, cwd=tmp_path, env=env, capture_output=True, timeout=60
            ).returncode
            == 1
        )


def test_owned_live_adapter_and_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8167 records the qualified owned load without legacy fallback."""
    plan = dict(
        slots=e.freeze(public(), {r["unit_id"]: r for r in public()})[:1],
        protocol=dict(chat_template_sha256=canonical_hash("template")),
        manifests={},
        capture_started_monotonic=e.time.monotonic(),
        capture_identity="id",
    )

    class Worker:
        def __init__(self, *args):
            self.model = tmp_path / "model.gguf"

        def load(self):
            return dict(props=dict(chat_template="template"))

    def own(p, raw, private):
        e.qualified.legacy.progress("heartbeat", 0)
        worker = e.qualified.legacy.QwenRuntime()
        loaded = worker.load()
        return dict(
            rows=e.qualified.legacy.capture.capture(
                p["slots"], Runtime(), raw / "calls", "id", started=0
            ),
            checks=[dict(upstream_id="owned", field="identity", passed=True)],
            receipt=loaded,
        )

    monkeypatch.setattr(e.qualified.legacy, "QwenRuntime", Worker)
    monkeypatch.setattr(e.qualified.legacy, "live_capture", own)
    monkeypatch.setattr(e.qualified.legacy, "bounded", lambda fn, timeout: fn())
    result = e.live(plan, tmp_path / "live", tmp_path)
    assert result["ledger"][0]["status"] == "completed"
    assert result["checks"][0]["check"] == "owned_identity"
    plan["protocol"]["chat_template_sha256"] = "changed"
    with pytest.raises(ValueError):
        e.live(plan, tmp_path / "failed", tmp_path)
    assert (
        json.loads((tmp_path / "failed/ledger.json").read_text())["rows"][0]["status"] == "failed"
    )


def test_authenticated_nonfixture_inputs_and_evaluator(tmp_path, monkeypatch):
    """REQ-REPORT-8167 checks terminal identity and opens only sealed fit/tune targets."""
    root = world(tmp_path)
    labels = tmp_path / "fit_tune_targets.json"
    atomic_json(labels, {r["unit_id"]: i % 2 for i, r in enumerate(public())})
    terminal = tmp_path / "terminal.json"
    sidecar = tmp_path / "sidecar.json"
    atomic_json(sidecar, {})
    atomic_json(terminal, dict(publication=dict(sidecar_path=str(sidecar))))
    for name in e.PINS:
        path = root / name
        value = json.loads(path.read_text())
        value["terminal_validation_sidecar_path"] = str(terminal)
        if name == e.HISTORICAL:
            value["raw_shard_hashes"] = [e.reference(labels)]
        atomic_json(path, value)
    monkeypatch.setattr(e, "PINS", {n: e.reference(root / n)["sha256"] for n in e.PINS})
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=True)))
    plan = e.inputs(root, tmp_path / "nonfixture")
    assert len(plan["slots"]) == 208
    monkeypatch.setattr(e.qualified, "runtime_preflight", lambda *a: None)

    def live(plan, raw, private):
        ledger = Ledger(raw / "ledger.json")
        ledger.start("model_load", "load", {})
        ledger.finish("load", "completed", {})
        rows = e.capture(plan["slots"], Runtime(), raw / "calls", "id", ledger=ledger)
        return dict(
            rows=rows,
            checks=[],
            ledger=ledger.rows,
            model_identity_receipt=dict(authenticated=True),
            resolved_library=dict(path="fixture"),
            gpu_lease_receipt=dict(owner="fixture"),
            cleanup=dict(leak_free=True),
        )

    monkeypatch.setattr(e, "live", live)
    work = e.measure(root, tmp_path / "measurement")
    assert work["labels"] == json.loads(labels.read_text())
    work["duration_s"] = 11
    value = e.build(work, tmp_path / "measurement", [dict(passed=True)])
    assert value["fit_trainable_score"] == 1
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 400
    assert value["inference_mode"] == "live_gpu"
    bad = deepcopy(work)
    bad["result"]["ledger"][1]["request_sha256"] = "changed"
    with pytest.raises(ValueError):
        e.build(bad, tmp_path / "measurement", [dict(passed=True)])
    monkeypatch.setattr(
        e, "live", lambda *a: dict(rows=[], checks=[dict(check="no_cuda", passed=False)], ledger=[])
    )
    blocked = e.measure(root, tmp_path / "cuda-block")
    assert (
        e.build(blocked, tmp_path / "cuda-block", [dict(passed=True)])["verdict_class"] == "blocked"
    )


def test_manifest_main_and_replay_receipt_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8167-CLI owns validation failures and keeps global health separate."""
    root = world(tmp_path)
    work = e.measure(root, tmp_path / "reference", fixture=True)
    specs = e.manifest(tmp_path, tmp_path / "candidate.json")
    assert all(
        "::" not in arg
        for s in specs["commands"]
        if s["name"] != "owned_unit_and_private_CLI"
        for arg in s["argv"]
    )
    minimal = dict(
        commands=[dict(name="owned", expected_exit=0)],
        terminal_commands=[],
        repository_health=dict(name="health"),
    )
    monkeypatch.setattr(e, "manifest", lambda *a: minimal)
    monkeypatch.setattr(e.execution, "publish", lambda *a: None)
    monkeypatch.setattr(e, "measure", lambda *a, **k: deepcopy(work))
    for passed in (True, False):
        monkeypatch.setattr(
            e.execution, "run_check", lambda *a, **k: dict(passed=passed, duration_s=0.1)
        )
        assert (
            e.main(
                ["--root", str(root), "--output", str(tmp_path / str(passed) / (e.NAME + ".json"))]
            )
            == 0
        )
    with pytest.raises(SystemExit):
        e.main(["--fixture-output", str(e.ROOT / "results" / (e.NAME + ".json"))])
    with pytest.raises(SystemExit):
        e.main(["--date", "20261004"])
    output = tmp_path / (e.NAME + ".json")
    log = tmp_path / "log.txt"
    log.write_text("original")
    value = e.build(
        work,
        tmp_path / "reference",
        [dict(passed=True, log_path=str(log), log_sha256=e.reference(log)["sha256"])],
        fixture=True,
    )
    atomic_json(output, value)
    log.write_text("changed")
    assert not e.replay(output)
    output.write_text("{broken")
    assert not e.replay(output)


def test_additional_transport_and_rehashed_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8167-REPLAY checks frozen identity beyond headline hashes."""
    rows = public()
    bad = deepcopy(rows)
    bad[0]["answer_bytes"] = b"changed".hex()
    with pytest.raises(ValueError):
        e.freeze(bad, {r["unit_id"]: r for r in rows})
    slots = e.freeze(rows, {r["unit_id"]: r for r in rows})[:1]
    request_builder = e.sentence.requests

    def incomplete(row, count):
        value = request_builder(row, count)
        value["requests"] = value["requests"][:1]
        return value

    with patch.object(e.sentence, "requests", incomplete):
        captured_rows = e.capture(
            slots,
            Runtime(),
            tmp_path / "incomplete",
            "id",
            ledger=Ledger(tmp_path / "incomplete.json"),
        )
    assert captured_rows[0]["exclusion_reason"] == "incomplete_sentence_coverage"
    good = e.capture(
        slots, Runtime(), tmp_path / "good", "id", ledger=Ledger(tmp_path / "good.json")
    )
    bad = deepcopy(good)
    bad[0]["local_features"] = [99]
    with pytest.raises(ValueError):
        e.reduce(bad)
    root = world(tmp_path)
    work = e.measure(root, tmp_path / "work", fixture=True)
    value = e.build(work, tmp_path / "work", [dict(passed=True)], fixture=True)
    output = tmp_path / (e.NAME + ".json")
    changed_work = deepcopy(work)
    changed_work["result"]["rows"][0]["status"] = "excluded"
    measurement = Path(value["measurement_reference"]["path"])
    atomic_json(measurement, changed_work)
    value["measurement_reference"] = e.reference(measurement)
    atomic_json(output, value)
    assert not e.replay(output)


def test_raw_reply_binding_and_original_label_gate(tmp_path):
    """REQ-REPORT-8167 binds transcripts to replies and hashes target custody before dispatch."""
    slots = e.freeze(public(), {r["unit_id"]: r for r in public()})[:1]
    rows = e.capture(
        slots, Runtime(), tmp_path / "calls", "id", ledger=Ledger(tmp_path / "ledger.json")
    )
    changed = deepcopy(rows)
    changed[0]["calls"][0]["raw_response"]["choices"][0]["message"]["content"] = "changed"
    with pytest.raises(ValueError):
        e.reduce(changed)
    changed = deepcopy(rows)
    changed[0]["source_sha256"] = "changed"
    with pytest.raises(ValueError):
        e.reduce(changed)


def test_rehashed_slot_forgery(tmp_path):
    """SCENARIO-VERIFY-8167-REPLAY binds source slots to the pre-call manifest."""
    root = world(tmp_path)
    work = e.measure(root, tmp_path / "work", fixture=True)
    work["result"]["rows"][0]["slot"] = 999
    primitive = Path(work["raw_shard_hashes"][1]["path"])
    primitive.chmod(0o600)
    atomic_json(primitive, dict(rows=work["result"]["rows"]))
    work["raw_shard_hashes"][1] = e.reference(primitive)
    atomic_json(tmp_path / "work/measurement.json", work)
    value = e.build(work, tmp_path / "work", [dict(passed=True)], fixture=True)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert not e.replay(output)
