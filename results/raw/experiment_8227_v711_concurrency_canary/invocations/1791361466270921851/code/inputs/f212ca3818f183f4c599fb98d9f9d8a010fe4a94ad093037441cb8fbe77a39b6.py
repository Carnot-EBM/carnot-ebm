"""REQ-VERIFY-8227 / REQ-REPORT-8227: private evidence tests never write results."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from carnot.verify import concurrency_canary_8227 as e
from carnot.reporting import concurrency_execution_8227 as runner
from carnot.inference import concurrency_runtime_8227 as live
from carnot.verify.recorder_fixtures_8213 import synthetic


def roster():
    """SCENARIO-VERIFY-8227-PROTOCOL: fixture identities contain no human targets."""
    return [
        dict(
            role="fit",
            source_cluster_id=f"sha256:{i:064x}",
            source_id=str(i),
            response_id=str(i),
            slot=i,
            condition="original",
            source_bytes=b"source".hex(),
            answer_bytes=b"answer".hex(),
            requests=[
                dict(
                    payload=json.dumps(
                        dict(
                            answer_sentence_indices=[0],
                            answer_offsets=[[0, 6]],
                            source_segments=["source"],
                        )
                    )
                )
            ],
        )
        for i in range(28)
    ]


def test_protocol(tmp_path):
    """SCENARIO-VERIFY-8227-PROTOCOL: roles and all96 obligations precede outcomes."""
    identity = synthetic(0)["runtime_identity"]
    p = e.freeze(roster(), identity)
    assert len(p["canary"]) == 8 and len(p["benchmark"]) == 96
    assert len({r["source_cluster_id"] for r in p["benchmark"]}) == 24
    assert not (
        {r["source_cluster_id"] for r in p["canary"]}
        & {r["source_cluster_id"] for r in p["benchmark"]}
    )
    assert p["launch_order"] == ["s1_serial", "s1_concurrent", "s2_concurrent", "s2_serial"]
    assert all(r["envelope"]["payload"]["cache_prompt"] is False for r in p["canary"])
    assert all(r["envelope"]["payload"]["max_tokens"] == 128 for r in p["benchmark"])
    for bad in (roster()[:27], [roster()[0]] * 28):
        with pytest.raises(ValueError, match="source_count"):
            e.freeze(bad, identity)


def test_scripted_isolation(tmp_path):
    """SCENARIO-VERIFY-8227-ISOLATION: queues, interleaving and durable restarts work."""
    work = e.qualify(tmp_path)
    assert work["passed"] and work["verdict_class"] == "circular_positive"
    assert work["checks"]["interleaving"] and work["checks"]["partial_tail"]
    assert work["checks"]["restart"] and work["checks"]["queued_failure"]
    rows = work["rows"]
    assert [r["status"] for r in rows] == ["completed", "completed", "error", "completed"]
    assert all(
        r["clocks"]["issue"]
        <= r["clocks"]["queue"]
        <= r["clocks"]["start"]
        <= r["clocks"]["end"]
        <= r["clocks"]["durability"]
        for r in rows
    )
    assert {r["result"]["id_slot"] for r in rows if r["status"] == "completed"} == {0, 1}


def test_scheduler_failure_censor_and_drift(tmp_path):
    """SCENARIO-VERIFY-8227-ISOLATION: never replace a failed or censored request."""
    p = e.freeze(roster(), synthetic(0)["runtime_identity"])
    rows = e.acquire(p["canary"][:4], lambda payload, slot: {}, tmp_path / "bad", 2)
    assert all(r["status"] == "error" for r in rows)
    rows = e.acquire(p["canary"][:4], lambda payload, slot: {}, tmp_path / "cap", 1, cap_s=0)
    assert all(r["status"] == "censored" for r in rows)
    changed = deepcopy(p["canary"][:1])
    changed[0]["envelope"]["payload"]["seed"] += 1
    with pytest.raises(ValueError, match="envelope"):
        e.acquire(changed, lambda payload, slot: {}, tmp_path / "drift", 1)


def cli(*args):
    """SCENARIO-REPORT-8227-CLI: run the actual script from outside the checkout."""
    return subprocess.run(
        [sys.executable, str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/tmp",
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8227-CLI: publication and tamper checks run in real children."""
    output = tmp_path / (e.NAME + ".json")
    child = cli("--fixture-e2e", output)
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["concurrent_canary_ready_score"] == 0
    assert value["model_invocation_counts"]["generation_calls"] == 0
    assert cli("--cold-replay", output).returncode == 0
    changed = deepcopy(value)
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = runner.checksum(changed)
    e.atomic_json(output, changed)
    assert cli("--cold-replay", output).returncode == 1
    assert cli("--date", "20261006").returncode == 2
    assert cli("--fixture-e2e", e.ROOT / "results" / "bad.json").returncode == 2


def response(slot=0, identity=None):
    """SCENARIO-VERIFY-8227-ISOLATION: explicit synthetic token clocks test custody."""
    return dict(
        id=identity or str(time.monotonic_ns()),
        id_slot=slot,
        first_token_monotonic_ns=time.monotonic_ns(),
        choices=[dict(message=dict(content="fixture"), finish_reason="stop")],
        usage=dict(prompt_tokens=1, completion_tokens=1),
    )


def test_scheduler_semantic_failures(tmp_path):
    """REQ-VERIFY-8227: cross-slot, duplicate response and absent clocks fail closed."""
    p = e.freeze(roster(), synthetic(0)["runtime_identity"])
    for mode in ["slot", "duplicate", "clock", "length"]:

        def generate(payload, slot):
            result = response(slot, "same" if mode == "duplicate" else None)
            if mode == "slot":
                result["id_slot"] = 8
            if mode == "clock":
                result["first_token_monotonic_ns"] = None
            if mode == "length":
                result["choices"][0]["finish_reason"] = "length"
            return result

        rows = e.acquire(p["canary"][:4], generate, tmp_path / mode, 1)
        assert sum(r["status"] == "error" for r in rows) == (3 if mode == "duplicate" else 4)
        assert all("response" in r["result"] for r in rows if r["status"] == "error")
        value = runner.build(dict(protocol=p, checks=[]), dict(rows=rows), tmp_path, [], 1, True)
        assert value["failed_count"] == (3 if mode == "duplicate" else 4)


def test_preflight_and_commands(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8227-PROTOCOL: cache, hardware and binary operands stay exact."""
    monkeypatch.setattr(live, "cached_current_model", lambda **kw: None)
    assert live.preflight({}, tmp_path)["model"] is None
    model = tmp_path / "Qwen3.8-Q4_K_M.gguf"
    template = "fixture template"
    model.write_bytes(b"00000000" + template.encode())
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fixture binary")
    identity = dict(
        model_sha256=e.sha256_file(model),
        chat_template_sha256=e.key(template),
        runtime_sha256=e.sha256_file(binary),
    )
    metadata = dict(
        quantization="Q4_K_M",
        field_provenance=dict(
            metadata_keys={
                "tokenizer.chat_template": dict(
                    value_offset=0, value_end_offset=model.stat().st_size
                )
            }
        ),
    )
    monkeypatch.setattr(live, "cached_current_model", lambda **kw: dict(model_path=str(model)))
    monkeypatch.setattr(live, "BINARY", binary)
    monkeypatch.setattr(live, "read_gguf_metadata", lambda path: metadata)

    def execute(commands, raw):
        raw.mkdir(parents=True, exist_ok=True)
        rows = []
        for spec in commands:
            path = raw / (spec.name + ".stdout")
            path.write_text(
                "0, GPU-fixture, NVIDIA RTX 3090, 24000, 4\ninvalid\n"
                if spec.name == "gpu_memory"
                else "--parallel --cache-ram --no-cache-prompt"
            )
            rows.append(dict(name=spec.name, stdout_path=str(path), exit_code=0, passed=True))
        return rows

    monkeypatch.setattr(live, "execute", execute)
    good = live.preflight(identity, tmp_path / "good")
    assert all(c["passed"] for c in good["checks"]) and good["gpu"]["index"] == 0
    argv = live.command(model, tmp_path, 0)
    assert argv[argv.index("--parallel") + 1] == "2" and argv[argv.index("-c") + 1] == "8192"
    assert argv[-1] == "--slots"
    identity["model_sha256"] = "drift"
    assert not all(c["passed"] for c in live.preflight(identity, tmp_path / "bad")["checks"])


def test_native_streaming(tmp_path, monkeypatch):
    """REQ-VERIFY-8227: real parser statements retain first token and native tail bytes."""
    from io import BytesIO

    terminal = dict(id_slot=1, stop=True, tokens_evaluated=5, tokens_predicted=2)
    wire = (
        b"\n"
        + b'data: {"content":"fixture"}\n\n'
        + b"data: "
        + json.dumps(terminal).encode()
        + b"\n"
    )
    monkeypatch.setattr(live, "urlopen", lambda *a, **kw: BytesIO(wire))
    runtime = SimpleNamespace(
        port=1, worker=SimpleNamespace(post_json=lambda *a: dict(prompt="rendered"))
    )
    result = live.generate(runtime, dict(messages=[], grammar="fixture"), 1)
    assert result["usage"]["completion_tokens"] == 2 and result["id_slot"] == 1
    assert result["first_token_monotonic_ns"] > 0 and result["wire_bytes_hex"]


def fake_live(tmp_path, monkeypatch, mode="valid"):
    """SCENARIO-VERIFY-8227-PROTOCOL: private fake ownership exercises supervisor wiring."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    p = e.freeze(roster(), synthetic(0)["runtime_identity"])
    p["server_argv"] = {
        "canary_" + arm: ["fixture-server", "--port", "100"] for arm in ["serial", "concurrent"]
    }

    class Lease:
        def __init__(self):
            self.document = dict(phase="preflight")

        def owner_receipt(self):
            return dict(fixture=True)

        def transition(self, phase, **kw):
            self.document["phase"] = phase

        def release(self):
            return dict(released=True, phase=self.document["phase"])

    def acquire(**kw):
        if mode == "lease":
            raise live.LeaseError("fixture occupied lease")
        return Lease()

    monkeypatch.setattr(live, "GpuLease", SimpleNamespace(acquire=acquire))

    class Runtime:
        def __init__(self, model, scratch, gpu):
            self.scratch, self.log = scratch, scratch / "server.log"

        def load(self):
            self.log.write_text("fixture loaded")
            if mode == "load":
                raise RuntimeError("fixture unsupported server")
            return dict(
                props=dict(
                    total_slots=1 if mode == "slots" else 2,
                    default_generation_settings=dict(n_ctx=4096),
                    chat_template="fixture template",
                )
            )

        def close(self):
            return dict(leak_free=mode != "cleanup")

    p["identity"]["chat_template_sha256"] = e.key("fixture template")
    monkeypatch.setattr(live, "QwenRuntime", Runtime)
    monkeypatch.setattr(live, "generate", lambda runtime, payload, slot: response(slot))

    def execute(commands, raw):
        path = raw / "resident.stdout"
        path.write_text("25000" if mode == "memory" else "1000")
        return [dict(stdout_path=str(path), passed=True)]

    monkeypatch.setattr(live, "execute", execute)
    resources = dict(
        gpu=dict(index=0, uuid="fixture-gpu", used_mb=4, free_mb=24000),
        model=dict(model_path=str(model)),
    )
    return p, resources


@pytest.mark.parametrize("mode", ["valid", "lease", "load", "slots", "memory", "cleanup"])
def test_owned_runtime_paths(tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8227-PROTOCOL: owned resources include each failure and teardown."""
    protocol, resources = fake_live(tmp_path, monkeypatch, mode)
    work = live.live(protocol, resources, tmp_path / "work")
    if mode == "valid":
        assert len(work["rows"]) == 8 and all(r["status"] == "completed" for r in work["rows"])
        assert work["actual_launch_order"] == ["canary_serial", "canary_concurrent"]
        assert work["gpu_lease_release"]["phase"] == "terminal_complete"
    else:
        assert any(not c["passed"] for c in work["checks"])


def test_readiness_and_owned_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8227: readiness needs real paired evidence and never raises benefit."""
    protocol, resources = fake_live(tmp_path, monkeypatch)
    work = live.live(protocol, resources, tmp_path / "work")
    work["qualification"] = dict(passed=True)
    work["phase_spans"] = [dict(phase="generation", start_ns=0, end_ns=11_000_000_000)]
    data = dict(protocol=protocol, checks=[], refs=[], code=[])
    good = runner.build(data, work, tmp_path, [dict(passed=True)], 20, False)
    assert good["concurrent_canary_ready_score"] == 1 and good["generated_tokens"] == 8
    assert good["independent_count"] == 4 and good["generalized_learning_benefit_score"] == 0
    bad = runner.build(data, work, tmp_path, [dict(passed=False)], 20, False)
    assert bad["verdict_class"] == "disqualified" and bad["concurrent_canary_ready_score"] == 0


def test_replay_primitive_and_aggregate_drift(tmp_path, monkeypatch):
    """REQ-REPORT-8227: rehashed summaries cannot impersonate native journal responses."""
    protocol, resources = fake_live(tmp_path, monkeypatch)
    work = live.live(protocol, resources, tmp_path / "work")
    for row in work["rows"]:
        row["journal_path"] = str(
            tmp_path / "work" / ("canary_" + row["arm"]) / "requests/events.jsonl"
        )
    data = dict(protocol=protocol, checks=[], refs=[], code=[])
    e.atomic_json(tmp_path / "data.json", data)
    e.atomic_json(tmp_path / "result.json", work)
    value = runner.build(data, work, tmp_path, [], 20, False)
    output = tmp_path / (e.NAME + ".json")

    def write(v):
        v["reproducibility_checksum"] = runner.checksum(v)
        e.atomic_json(output, v)

    write(value)
    assert runner.replay(output)
    changed_work = deepcopy(work)
    changed_work["rows"][0]["clocks"]["issue"] += 1
    e.atomic_json(tmp_path / "result.json", changed_work)
    assert not runner.replay(output)
    e.atomic_json(tmp_path / "result.json", work)
    changed = deepcopy(value)
    changed["reproducibility_checksum"] = "drift"
    e.atomic_json(output, changed)
    assert not runner.replay(output)
    changed = deepcopy(value)
    changed["raw_shard_hashes"] = [dict(path=str(tmp_path / "data.json"), sha256="drift")]
    write(changed)
    assert not runner.replay(output)
    write(value)
    work["rows"][0]["result"]["usage"]["completion_tokens"] = 9
    e.atomic_json(tmp_path / "result.json", work)
    assert not runner.replay(output)
    output.write_text("{invalid")
    assert not runner.replay(output)


def test_attempted_operations_determine_substrate(tmp_path):
    """REQ-REPORT-8227: one attempted load is current work despite an external block."""
    work = dict(loads=[dict(completed=False)], rows=[], checks=[])
    data = dict(checks=[dict(passed=False, artifact_field="unsupported")])
    value = runner.build(data, work, tmp_path, [], 1, False)
    assert value["inference_substrate_class"] == "model_load_no_generation"
    assert value["model_invocation_counts"]["model_loads_attempted"] == 1


def test_external_missing_cli(tmp_path):
    """REQ-REPORT-8227: external absence is blocked with missing rather than zero."""
    output = tmp_path / (e.NAME + ".json")
    child = cli("--root", tmp_path / "missing", "--output", output)
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["honest_verdict"].startswith(
        "complete_blocked_"
    )
    assert value["gate_check_summary"][0]["observed"] is None
    assert value["censored_count"] == 8 and value["completed_count"] == 0
    assert cli("--cold-replay", output).returncode == 0


@pytest.mark.parametrize("mode", ["valid", "terminal", "schema", "resources", "owned"])
def test_natural_runner_private_wiring(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8227-CLI: private injected operands exercise natural orchestration."""
    protocol, resources = fake_live(tmp_path, monkeypatch)
    natural = live.live(protocol, resources, tmp_path / "natural")
    data = dict(ready=True, checks=[], refs=[], roster=roster(), identity=protocol["identity"])
    monkeypatch.setattr(runner.qualified, "inputs", lambda *a: deepcopy(data))
    monkeypatch.setattr(
        runner.runtime,
        "preflight",
        lambda *a: dict(
            resources,
            receipts=[],
            checks=[
                dict(
                    passed=mode != "resources",
                    artifact_field="private_resource",
                    path=str(tmp_path),
                    expected=True,
                    observed=False,
                )
            ],
        ),
    )
    monkeypatch.setattr(runner.runtime, "command", lambda *a: ["fixture-server", "--port", "100"])
    monkeypatch.setattr(runner.runtime, "live", lambda *a: natural)
    monkeypatch.setattr(runner, "PROTOCOL", str(tmp_path / "protocol.json"))
    monkeypatch.setattr(e, "qualify", lambda *a: dict(passed=True))
    monkeypatch.setattr(
        runner,
        "validation_plan",
        lambda *a: [runner.CommandSpec("private-owned", ("true",), "owned", 5)],
    )

    def execute(commands, raw):
        return [
            dict(
                name=c.name,
                passed=not (
                    (mode == "terminal" and c.scope == "terminal")
                    or (mode == "owned" and c.name == "private-owned")
                ),
                normal_exit=True,
            )
            for c in commands
        ]

    monkeypatch.setattr(runner, "execute", execute)
    original = json.loads
    if mode == "schema":

        def loads(text, **kw):
            value = original(text, **kw)
            if isinstance(value, dict) and value.get("experiment_id") == 8214:
                raise ValueError("private bad schema")
            return value

        monkeypatch.setattr(runner.json, "loads", loads)
    output = tmp_path / (e.NAME + ".json")
    log = tmp_path / "health.stdout"
    log.write_text("private prior health diagnostic")
    receipt = dict(
        command_argv=[str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        passed=False,
        stdout_path=str(log),
        stdout_sha256=e.sha256_file(log),
        stderr_path=str(log),
        stderr_sha256=e.sha256_file(log),
    )
    saved = tmp_path / "prior_health.json"
    e.atomic_json(saved, [receipt])
    e.atomic_json(output, dict(experiment_id=8227, task_id=e.TASK))
    argv = ["--output", str(output)]
    if mode != "valid":
        argv += ["--repository-health-receipt", str(saved)]
    exit_code = runner.main(argv)
    assert exit_code == (1 if mode == "terminal" else 0)
    if exit_code == 0:
        value = original(output.read_text())
        assert value["concurrent_canary_ready_score"] == 0
        if mode in {"schema", "resources"}:
            assert value["verdict_class"] == "blocked"
        if mode == "owned":
            assert value["verdict_class"] == "disqualified"


def test_health_receipt_rejects_unbound_diagnostics(tmp_path):
    """REQ-REPORT-8227: diagnostic reuse rejects changed logs before current work."""
    saved = tmp_path / "bad_health.json"
    e.atomic_json(saved, [dict(command_argv=["wrong command"])])
    assert cli("--repository-health-receipt", saved).returncode == 2
    log = tmp_path / "drift.stdout"
    log.write_text("changed private log")
    e.atomic_json(
        saved,
        [
            dict(
                command_argv=[str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
                stdout_path=str(log),
                stdout_sha256="sha256:drift",
            )
        ],
    )
    assert cli("--repository-health-receipt", saved).returncode == 2


def test_partial_tail_control_fails_if_recorder_accepts_it(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8227-ISOLATION: a permissive recorder cannot qualify isolation."""
    journal = e.recorder.Journal
    monkeypatch.setattr(
        e.recorder,
        "Journal",
        lambda path: SimpleNamespace() if path.name == "partial.jsonl" else journal(path),
    )
    value = e.qualify(tmp_path)
    assert not value["passed"] and not value["checks"]["partial_tail"]
