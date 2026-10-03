"""REQ-REPORT-7969: isolated CLI, exact publication and external absence."""

import json
from pathlib import Path

from carnot import experiment_7969_v691_qwen_calibration_capture as producer
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from test_qwen_calibration_capture_7969 import views


def test_fixture_blocked_replay_and_terminal(tmp_path):
    """SCENARIO-REPORT-7969-TERMINAL: only final checked bytes reach readers."""
    source = tmp_path / "input.json"
    atomic_json(source, views())
    output = tmp_path / "fixture" / (producer.NAME + ".json")
    assert producer.main(["--fixture-input", str(source), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["qwen_capture_ready_score"] == 0
    assert value["current_evaluation_call_count"] == 0
    assert producer.replay(value)["sample_size_budget"]["completed"] == 384
    assert read_bound_sidecar(output, Path(value["terminal_validation_sidecar_path"]))[
        "primary_sha256"
    ] == sha256_file(output)
    assert producer.main(["--cold-replay", str(output)]) == 0
    value["generated_tokens"] += 1
    atomic_json(tmp_path / "bad.json", value)
    assert producer.main(["--cold-replay", str(tmp_path / "bad.json")]) == 1
    blocked = tmp_path / "blocked" / output.name
    assert producer.main(["--root", str(tmp_path / "missing"), "--output", str(blocked)]) == 0
    value = json.loads(blocked.read_text())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert value["MODEL_SPECS"] == []
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0


def test_real_input_gate_and_required_failure():
    """SCENARIO-REPORT-7969-VALIDATION: gate readiness never claims benefit."""
    failures, upstream = producer.authenticate(producer.ROOT)
    assert not failures and len(producer.load_public(upstream["public_role_manifests"])) == 4
    value = producer.base([])
    producer.apply_validation(value, [{"passed": False, "required": True, "command_argv": []}])
    assert value["verdict_class"] == "disqualified"
    assert value["qwen_capture_ready_score"] == 0


def test_partial_publishes_checked_resumable_identity(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7969-TERMINAL: expired owned work has a partial primary."""
    value = producer.base([])
    value.update(verdict_class="partial", honest_verdict="complete_partial_owned_capture_budget")
    monkeypatch.setattr(producer, "terminal_check", lambda _: dict(passed=True))
    output = tmp_path / (producer.NAME + ".json")
    producer.publish(output, value)
    assert json.loads(output.read_text())["verdict_class"] == "partial"
    assert read_bound_sidecar(output, Path(value["terminal_validation_sidecar_path"]))


def runtime_fixture(tmp_path, monkeypatch):
    """Use explicit scripted native boundaries, with no model provenance claim."""
    from unittest.mock import MagicMock

    from test_qwen_calibration_capture_7969 import CountingRuntime
    from test_qwen_response_risk_7958 import reply

    model = tmp_path / "revision" / "model.gguf"
    model.parent.mkdir(exist_ok=True)
    model.write_bytes(b"fixture GGUF bytes")
    plan = dict(
        public_role_manifests={},
        capture_identity="identity",
        protocol=dict(gguf_sha256=sha256_file(model), model_revision="revision"),
    )
    monkeypatch.setattr(producer, "load_public", lambda _: views())
    monkeypatch.setattr(
        producer,
        "cached_current_model",
        lambda: dict(model_path=str(model), hf_id=producer.MODEL_SPECS[0]),
    )
    monkeypatch.setattr(producer, "read_gguf_metadata", lambda _: dict(quantization="Q4_K_M"))
    monkeypatch.setattr(
        producer,
        "run_commands",
        lambda *a, **k: [
            dict(
                passed=True,
                output_tail="0, GPU-fixture, 4, 24000\n1, GPU-two, 4, 24000",
                command_argv=[],
                exit_code=0,
            )
        ],
    )
    lease = MagicMock()
    lease.document = dict(phase="inferencing")
    lease.owner_receipt.return_value = {}
    lease.release.return_value = dict(released=True)
    monkeypatch.setattr(producer.GpuLease, "acquire", lambda **_: lease)
    monkeypatch.setattr(producer.prior.completion.prior, "gpu_memory", lambda *a: (16000, {}))

    class Native(CountingRuntime):
        def __init__(self, *args):
            super().__init__()
            self.command = [str(model)]
            self.log = tmp_path / "server.log"
            self.log.write_text("offloaded 65/65 layers to GPU")
            self.receipts = []

        def load(self):
            return dict(
                authenticated=True,
                offload_layers=[65, 65],
                owner=dict(pid=123, start_time_ticks=456),
            )

        def generate(self, request):
            self.receipts.append(dict(request=request))
            return reply()

        def close(self):
            return dict(leak_free=True)

    monkeypatch.setattr(producer, "QwenRuntime", Native)
    monkeypatch.setattr(
        producer, "loaded_libraries", lambda *a: dict(libraries=[producer.reference(model)])
    )
    return plan, model, lease, Native


def test_owned_runtime_success_and_external_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7969-VALIDATION: private worker and idle lease gates."""
    from unittest.mock import MagicMock
    import threading

    plan, model, lease, native = runtime_fixture(tmp_path, monkeypatch)
    # Execute a single lease heartbeat without waiting fifteen real seconds.
    event = MagicMock()
    event.wait.side_effect = [False, True]
    monkeypatch.setattr(threading, "Event", lambda: event)
    thread = MagicMock()
    monkeypatch.setattr(threading, "Thread", lambda target, **k: (target(), thread)[1])
    raw = tmp_path / "success"
    result = producer.live_capture(plan, raw, tmp_path)
    assert result["model_loads_completed"] == 1 and len(result["rows"]) == 384
    assert lease.release.called and lease.heartbeat.called
    assert result["resolved_library"]["libraries"][0]["sha256"] == sha256_file(model)
    monkeypatch.setattr(producer, "cached_current_model", lambda: None)
    assert not producer.live_capture(plan, tmp_path / "missing", tmp_path)["model_loads_attempted"]
    monkeypatch.setattr(
        producer,
        "cached_current_model",
        lambda: dict(model_path=str(model), hf_id=producer.MODEL_SPECS[0]),
    )
    drift = {**plan, "protocol": {**plan["protocol"], "gguf_sha256": "sha256:changed"}}
    assert not producer.live_capture(drift, tmp_path / "hash", tmp_path)["model_loads_attempted"]
    monkeypatch.setattr(
        producer, "run_commands", lambda *a, **k: [dict(passed=False, output_tail="")]
    )
    assert not producer.live_capture(plan, tmp_path / "busy", tmp_path)["model_loads_attempted"]
    monkeypatch.setattr(
        producer,
        "run_commands",
        lambda *a, **k: [dict(passed=True, output_tail="0, GPU-fixture, 4, 24000")],
    )
    monkeypatch.setattr(
        producer.GpuLease, "acquire", lambda **k: (_ for _ in ()).throw(producer.LeaseError("busy"))
    )
    assert not producer.live_capture(plan, tmp_path / "lease", tmp_path)["model_loads_attempted"]


def test_owned_load_error_releases_only_owner(tmp_path, monkeypatch):
    """REQ-REPORT-7969: failed owned load retains attempts and cleanup evidence."""
    plan, _, lease, native = runtime_fixture(tmp_path, monkeypatch)
    native.load = lambda _: (_ for _ in ()).throw(RuntimeError("load failed"))
    lease.document = dict(phase="loading")
    result = producer.live_capture(plan, tmp_path / "error", tmp_path)
    assert result["model_loads_attempted"] == 1 and result["model_loads_completed"] == 0
    assert lease.release.called and result["checks"][-2]["passed"] is False


def test_live_build_null_blocked_partial_and_floor(tmp_path):
    """REQ-REPORT-7969: completion never reports calibration or decision benefit."""
    from test_qwen_calibration_capture_7969 import CountingRuntime

    raw = tmp_path / "raw"
    raw.mkdir()
    frozen = producer.capture.freeze(views())
    atomic_json(raw / "request_manifest.json", dict(rows=frozen, capture_identity="identity"))
    rows = producer.capture.capture(frozen, CountingRuntime(), raw / "slots", "identity")
    result = dict(
        rows=rows,
        capture_identity="identity",
        checks=[],
        model_loads_attempted=1,
        model_loads_completed=1,
        model_identity_receipt=dict(offload_layers=[65, 65]),
        resident_gpu_receipt={},
        measured_duration_s=12,
    )
    value = producer.build(raw, result, {})
    assert value["verdict_class"] == "null" and value["qwen_capture_ready_score"] == 1
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 384
    result["child_receipt"] = dict(command_argv=["owned-python", "capture-child"])
    assert producer.build(raw, result, {})["observed_child_commands"] == [
        ["owned-python", "capture-child"]
    ]
    assert value["acceptance_gate_results"]["decision_benefit"] is None
    result["measured_duration_s"] = 1
    assert producer.build(raw, result, {})["verdict_class"] == "disqualified"
    rows[-1]["status"] = "censored"
    rows[-1]["started"] = False
    rows[-1]["raw_response"] = {}
    rows[-1]["parsed"] = producer.prior.risk.transport.parse_response({}, rows[-1]["visible_ids"])
    assert producer.build(raw, result, {})["verdict_class"] == "partial"
    result["checks"] = [dict(passed=False)]
    assert producer.build(raw, result, {})["verdict_class"] == "blocked"


def test_replay_mutations_and_private_route_errors(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7969-TERMINAL: drift and unsafe readiness fail closed."""
    from copy import deepcopy
    import pytest

    source = tmp_path / "input.json"
    atomic_json(source, views())
    output = tmp_path / "success" / (producer.NAME + ".json")
    assert producer.main(["--fixture-input", str(source), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    shard = Path(value["raw_response_shards"][0]["path"])
    saved = json.loads(shard.read_text())
    changed = deepcopy(saved)
    changed["request"]["seed"] = 1
    atomic_json(shard, changed)
    mutated = {
        **value,
        "raw_response_shards": [producer.reference(shard), *value["raw_response_shards"][1:]],
    }
    with pytest.raises(ValueError, match="request_drift"):
        producer.replay(mutated)
    atomic_json(shard, saved)
    manifest = Path(value["request_manifest"]["path"])
    frozen = json.loads(manifest.read_text())
    atomic_json(manifest, {"rows": []})
    with pytest.raises(ValueError, match="request_manifest_drift"):
        producer.replay({**value, "request_manifest": producer.reference(manifest)})
    atomic_json(manifest, frozen)
    with pytest.raises(ValueError, match="rows_drift"):
        producer.replay({**value, "rows": []})
    with pytest.raises(ValueError, match="unsafe_readiness"):
        producer.replay({**producer.base([]), "qwen_capture_ready_score": 1})
    monkeypatch.setattr(producer.prior.targets, "check_receipts", lambda _: None)
    producer.replay(
        {
            **value,
            "verdict_class": "null",
            "qwen_capture_ready_score": 1,
            "inference_mode": "live_gpu",
        }
    )
    with pytest.raises(ValueError, match="unsafe_readiness"):
        producer.replay({**value, "verdict_class": "null", "qwen_capture_ready_score": 1})
    for args, code in [
        (["--capture-manifest", str(source)], 1),
        (["--fixture-input", str(tmp_path / "missing")], 1),
    ]:
        assert producer.main(args) == code
    monkeypatch.setattr(producer, "live_capture", lambda *a: dict(rows=[]))
    atomic_json(source, dict(raw=str(tmp_path / "child")))
    assert (
        producer.main(["--capture-manifest", str(source), "--runtime-scratch", str(tmp_path)]) == 0
    )
    monkeypatch.setattr(producer, "reader_receipt", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="primary_resolution"):
        producer.publish(tmp_path / "bad-reader" / output.name, producer.base([]))


def test_protocol_custody_exception_is_blocked(monkeypatch):
    """REQ-REPORT-7969: present but changed public bytes are external blocks."""
    monkeypatch.setattr(
        producer, "load_public", lambda _: (_ for _ in ()).throw(ValueError("changed_public_hash"))
    )
    failures, _ = producer.authenticate(producer.ROOT)
    assert failures[-1]["field"] == "authenticated_public_protocol"


def test_historical_prompt_identity_drift(monkeypatch):
    """REQ-REPORT-7969: authenticate full-source wording against frozen replies."""
    original = producer.prior.risk.freeze

    def drift(*args):
        rows = original(*args)
        rows[0]["requests"]["full_source"]["seed"] = 0
        return rows

    monkeypatch.setattr(producer.prior.risk, "freeze", drift)
    failures, _ = producer.authenticate(producer.ROOT)
    assert failures[-1]["observed"] == "historical_prompt_drift"


def test_partial_guard_rejects_unsafe_identity(tmp_path, monkeypatch):
    """REQ-REPORT-7969: partial cannot bypass primary naming or validation."""
    import pytest

    value = producer.base([])
    value.update(verdict_class="partial", honest_verdict="complete_partial_owned_capture_budget")
    with pytest.raises(ValueError, match="partial_publication_identity"):
        producer.publish(tmp_path / "wrong.json", value)
    monkeypatch.setattr(producer, "terminal_check", lambda _: dict(passed=False))
    with pytest.raises(ValueError, match="candidate_rejected"):
        producer.publish(tmp_path / (producer.NAME + ".json"), value)


def test_host_main_qualification_and_nonempty_coverage(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7969-VALIDATION: collect genuine exits, never hide failures."""
    from copy import deepcopy

    real_root = producer.ROOT
    original_reference = producer.reference
    root = tmp_path / "repo"
    root.mkdir()
    output = root / "results" / (producer.NAME + ".json")
    upstream = dict(
        checks=[],
        public_role_manifests={},
        protocol=dict(model_revision="revision", gguf_sha256="sha256:fixture"),
    )
    monkeypatch.setattr(producer, "ROOT", root)
    monkeypatch.setattr(producer, "authenticate", lambda _: ([], upstream))
    monkeypatch.setattr(producer, "load_public", lambda _: views())

    def reference(path):
        if not path.is_file() and path.is_relative_to(root):
            path = real_root / path.relative_to(root)
        return original_reference(path)

    monkeypatch.setattr(producer, "reference", reference)
    original_freeze = producer.freeze_commands
    state = {}

    def freeze(raw, scratch, data):
        plan = original_freeze(raw, scratch, data)
        state.update(raw=raw, scratch=scratch)
        return plan

    monkeypatch.setattr(producer, "freeze_commands", freeze)

    def child(*args, **kwargs):
        atomic_json(
            state["raw"] / "runtime_receipt.json",
            dict(rows=[], checks=[], capture_identity="fixture"),
        )
        return [dict(passed=True, command_argv=[], exit_code=0)]

    monkeypatch.setattr(producer, "run_commands", child)
    published = []
    monkeypatch.setattr(producer, "publish", lambda path, value: published.append(deepcopy(value)))

    def qualify(plan, logs):
        atomic_json(
            state["scratch"] / "coverage.json",
            {
                "files": {
                    path: {"summary": {"missing_lines": 0, "num_statements": 1, "covered_lines": 1}}
                    for path in producer.OWNED
                }
            },
        )
        (state["scratch"] / ".coverage.fixture").write_bytes(b"fixture coverage archive")
        return [dict(required=True, passed=True, command_argv=[])]

    monkeypatch.setattr(producer.prior.targets.prior, "execute_commands", qualify)
    assert producer.main(["--root", str(root), "--output", str(output)]) == 0
    assert published[-1]["coverage_statement_counts"]
    assert published[-1]["model_invocation_counts"]["generation_calls_attempted"] == 0
    monkeypatch.setattr(producer.prior.targets.prior, "execute_commands", lambda *a: [])
    assert producer.main(["--root", str(root), "--output", str(output)]) == 0
    assert published[-1]["verdict_class"] == "disqualified"
    assert published[-1]["validation_receipts"][-1]["name"] == "nonempty_complete_coverage"


def test_loaded_library_binds_pid_start_and_exact_bytes(tmp_path, monkeypatch):
    """REQ-REPORT-7969: dynamic CUDA libraries belong to the owned process."""
    import pytest

    proc = tmp_path / "proc"
    maps = proc / "42" / "maps"
    maps.parent.mkdir(parents=True)
    library = tmp_path / "libggml-cuda.so"
    library.write_bytes(b"fixture native library")
    maps.write_text(f"0-1 r-xp 0 0:0 0 {library}\n")
    monkeypatch.setattr(producer, "proc_start_ticks", lambda _: 7)
    owner = dict(pid=42, start_time_ticks=7)
    receipt = producer.loaded_libraries(owner, tmp_path / "raw", proc)
    assert receipt["libraries"][0]["sha256"] == sha256_file(library)
    with pytest.raises(RuntimeError, match="owned_pid_start_identity"):
        producer.loaded_libraries({**owner, "start_time_ticks": 8}, tmp_path / "raw", proc)
    maps.write_text("0-1 r-xp 0 0:0 0 /usr/lib/libc.so\n")
    with pytest.raises(RuntimeError, match="loaded_cuda_library"):
        producer.loaded_libraries(owner, tmp_path / "raw", proc)
