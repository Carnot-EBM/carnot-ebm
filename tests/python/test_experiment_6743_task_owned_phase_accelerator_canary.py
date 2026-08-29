"""Focused tests for the task-owned accelerator canary.

Spec refs: REQ-INFRA-6743, SCENARIO-INFRA-6743-MONOTONIC-ELAPSED,
SCENARIO-INFRA-6743-ACCELERATOR-INTEGRITY, and
SCENARIO-INFRA-6743-BLOCKED-PREFLIGHT.
"""

from __future__ import annotations

from copy import deepcopy
import builtins
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from carnot import experiment_6743_task_owned_phase_accelerator_canary as exp


def _clock(start: int = 1_000_000_000):
    value = start

    def now() -> int:
        nonlocal value
        value += 10_000_000
        return value

    return now


def _resolved_models(tmp_path: Path) -> list[dict]:
    rows = []
    for index, spec in enumerate(exp.MODEL_SPECS):
        path = tmp_path / f"model-{index}-{spec['quantization']}.gguf"
        path.write_bytes(f"GGUF fixture {spec['model_id']}".encode())
        rows.append(exp.model_identity_receipt(spec, path))
    return rows


def _passing_row(model: dict, index: int, start_ns: int) -> dict:
    pid = 7_400 + index
    device_uuid = f"GPU-fixture-{model['device_index']}"
    raw = f"CANARY-{index}"
    row = {
        **model,
        "owned_pid": pid,
        "parent_pid": 7_000,
        "assigned_device": {
            "index": model["device_index"],
            "uuid": device_uuid,
            "cuda_visible_devices": str(model["device_index"]),
        },
        "gpu_layers": {"requested": -1, "offloaded": 64, "total": 64},
        "peak_vram_mb": 16_384,
        "prompt_sha256": exp.PROMPT_SHA256,
        "prompt_tokens": 9,
        "output_tokens": 2,
        "stop_reason": "length",
        "raw_output": raw,
        "raw_output_sha256": exp.sha256_text(raw),
        "first_token_reached": True,
        "decode_completed": True,
        "supports_gpu_offload": True,
        "process_exit_code": 0,
        "owned_process_absent": True,
        "teardown_completed": True,
        "clocks": {
            "subprocess_started_ns": start_ns,
            "model_loaded_ns": start_ns + 10,
            "first_token_ns": start_ns + 20,
            "decode_complete_ns": start_ns + 30,
            "teardown_complete_ns": start_ns + 40,
        },
        "gpu_receipts": {
            "before": {
                "phase": "before",
                "monotonic_ns": start_ns + 1,
                "pid": pid,
                "pid_present": False,
                "device_uuid": device_uuid,
                "pid_memory_mb": 0,
            },
            "during": {
                "phase": "during",
                "monotonic_ns": start_ns + 21,
                "pid": pid,
                "pid_present": True,
                "device_uuid": device_uuid,
                "pid_memory_mb": 16_384,
            },
            "after": {
                "phase": "after",
                "monotonic_ns": start_ns + 41,
                "pid": pid,
                "pid_present": False,
                "device_uuid": device_uuid,
                "pid_memory_mb": 0,
            },
        },
        "backend_stderr_sha256": exp.sha256_text("fixture stderr"),
        "runtime_error": None,
    }
    row["row_sha256"] = exp.row_checksum(row)
    return row


def _passing_rows(tmp_path: Path) -> list[dict]:
    return [
        _passing_row(model, index, 2_000 + index * 100)
        for index, model in enumerate(_resolved_models(tmp_path))
    ]


def _phase_rows(rows: list[dict]) -> list[dict]:
    phases = [{"phase": "preflight", "monotonic_ns": 1_000, "model_id": None}]
    phases.extend(
        {
            "phase": "cache_resolved",
            "monotonic_ns": 1_100 + index * 10,
            "model_id": row["model_id"],
        }
        for index, row in enumerate(rows)
    )
    for row in rows:
        phases.extend(
            {
                "phase": phase,
                "monotonic_ns": row["clocks"][f"{phase}_ns"],
                "model_id": row["model_id"],
            }
            for phase in exp.MODEL_PHASES
        )
    phases.append({"phase": "artifact_write", "monotonic_ns": 3_000, "model_id": None})
    return phases


def _passing_preflight(models: list[dict]) -> dict:
    checks = [
        {"check": "llama_cpp_cuda_offload", "expected": True, "observed": True, "passed": True}
    ]
    checks.extend(
        {
            "check": f"cache:{model['model_id']}",
            "expected": True,
            "observed": True,
            "passed": True,
        }
        for model in models
    )
    return {"all_passed": True, "checks": checks, "gpu_inventory": []}


def test_req_infra_6743_model_policy_and_field_principles() -> None:
    """REQ-INFRA-6743 fixes the three local families and explains every field."""

    assert [row["model_id"] for row in exp.MODEL_SPECS] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]
    assert exp.INFERENCE_SUBSTRATE == "local llama.cpp CUDA GGUF"
    assert exp.MAX_OUTPUT_TOKENS > 0
    assert exp.PROMPT_SHA256 == exp.sha256_text(exp.FIXED_PROMPT)
    assert set(exp.FIELD_PRINCIPLES) == set(exp.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_infra_6743_monotonic_phase_order_and_real_elapsed(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6743-MONOTONIC-ELAPSED rejects reordered or zero clocks."""

    rows = _passing_rows(tmp_path)
    phases = _phase_rows(rows)
    assert exp.phase_clock_is_monotonic(phases) is True
    assert exp.phase_errors(phases, rows) == []
    artifact = exp.build_artifact(
        rows=rows,
        task_phase_rows=phases,
        preflight=_passing_preflight(rows),
        started_ns=1_000,
        artifact_write_ns=3_000,
    )
    assert artifact["duration_s"] == 0.000002
    assert artifact["duration_s"] > 0
    assert artifact["phase_clock_monotonic"] is True

    reversed_rows = deepcopy(phases)
    reversed_rows[2]["monotonic_ns"] = 1
    assert exp.phase_clock_is_monotonic(reversed_rows) is False
    assert "phase_clock_not_monotonic" in exp.phase_errors(reversed_rows, rows)

    zero_rows = deepcopy(phases)
    zero_rows[-1]["monotonic_ns"] = zero_rows[0]["monotonic_ns"]
    assert "phase_duration_not_positive" in exp.phase_errors(zero_rows, rows)


def test_scenario_infra_6743_accelerator_receipt_integrity(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6743-ACCELERATOR-INTEGRITY binds PID, UUID, layers, and teardown."""

    rows = _passing_rows(tmp_path)
    assert all(exp.model_row_errors(row) == [] for row in rows)
    assert exp.reduce_accelerator_readiness(rows) is True

    attacks = {
        "wrong_pid": ("gpu_receipts", "during", "pid", 99_999),
        "wrong_uuid": ("gpu_receipts", "during", "device_uuid", "GPU-other"),
        "no_layers": ("gpu_layers", "offloaded", None, 0),
        "empty_output": ("output_tokens", None, None, 0),
        "owned_process_live": ("owned_process_absent", None, None, False),
    }
    for path in attacks.values():
        mutated = deepcopy(rows)
        target = mutated[0]
        if path[1] is None:
            target[path[0]] = path[3]
        elif path[2] is None:
            target[path[0]][path[1]] = path[3]
        else:
            target[path[0]][path[1]][path[2]] = path[3]
        target["row_sha256"] = exp.row_checksum(target)
        assert exp.model_row_errors(target)
        assert exp.reduce_accelerator_readiness(mutated) is False


def test_scenario_infra_6743_blocked_preflight_never_runs_model(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6743-BLOCKED-PREFLIGHT writes observed failures without fallback."""

    models = _resolved_models(tmp_path)
    calls: list[str] = []

    def blocked_preflight(_: list[dict]) -> dict:
        return {
            "all_passed": False,
            "checks": [
                {
                    "check": "llama_cpp_cuda_offload",
                    "expected": True,
                    "observed": False,
                    "passed": False,
                }
            ],
            "gpu_inventory": [],
        }

    def forbidden_runner(model: dict) -> dict:
        calls.append(model["model_id"])
        raise AssertionError("blocked preflight must not invoke a model")

    artifact = exp.run(
        result_path=tmp_path / "blocked.json",
        resolver=lambda: models,
        preflight_fn=blocked_preflight,
        model_runner=forbidden_runner,
        clock=_clock(),
    )
    assert calls == []
    assert artifact["honest_verdict"].startswith("complete_blocked_accelerator_canary")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["live_model_invoked"] is False
    assert artifact["accelerator_receipt_ready"] is False
    assert artifact["gate_check_summary"] == [blocked_preflight(models)["checks"][0]]
    assert [row["model_id"] for row in artifact["rows"]] == [
        row["model_id"] for row in exp.MODEL_SPECS
    ]
    assert json.loads((tmp_path / "blocked.json").read_text()) == artifact
    assert exp.validate_artifact(artifact) == []


def test_req_infra_6743_run_is_sequential_and_artifact_valid(tmp_path: Path) -> None:
    """REQ-INFRA-6743 completes each model teardown before the next subprocess."""

    models = _resolved_models(tmp_path)
    now = _clock(10_000)
    calls: list[str] = []

    def runner(model: dict) -> dict:
        calls.append(model["model_id"])
        start = now()
        return _passing_row(model, len(calls), start)

    artifact = exp.run(
        result_path=tmp_path / "complete.json",
        resolver=lambda: models,
        preflight_fn=_passing_preflight,
        model_runner=runner,
        clock=now,
    )
    assert calls == [row["model_id"] for row in exp.MODEL_SPECS]
    assert artifact["accelerator_receipt_ready"] is True
    assert artifact["live_model_invoked"] is True
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete")
    assert artifact["claim_boundary"] == exp.CLAIM_BOUNDARY
    assert len(artifact["rows"]) == 3
    assert len(artifact["gpu_receipts"]) == 3
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact["rows"])
    assert exp.validate_artifact(artifact) == []

    mutated = deepcopy(artifact)
    mutated["rows"][0]["raw_output"] = "changed"
    assert "row_invalid:unsloth/Qwen3.6-35B-A3B-GGUF" in exp.validate_artifact(mutated)


def test_req_infra_6743_cache_resolution_uses_pair_pattern(tmp_path: Path) -> None:
    """REQ-INFRA-6743 resolves both pair shapes and preserves snapshot identity."""

    paths = {}
    for spec in exp.MODEL_SPECS:
        path = tmp_path / "snapshots" / "revision-123" / f"{spec['role']}.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(spec["model_id"].encode())
        paths[spec["model_id"]] = path
    calls = []

    def cached_pair(**kwargs):
        calls.append(kwargs)
        ids = (
            [exp.MODEL_SPECS[0]["model_id"], exp.MODEL_SPECS[2]["model_id"]]
            if kwargs["model_indices"] is None
            else [exp.MODEL_SPECS[0]["model_id"], exp.MODEL_SPECS[1]["model_id"]]
        )
        return [{"hf_id": model_id, "model_path": str(paths[model_id])} for model_id in ids]

    rows = exp.resolve_model_specs(cached_pair)
    assert [call["model_indices"] for call in calls] == [None, (0, 2)]
    assert all(row["resolved"] is True for row in rows)
    assert all(row["file_identity"]["snapshot_revision"] == "revision-123" for row in rows)
    assert all(row["file_identity"]["head_1m_sha256"] for row in rows)
    assert exp._snapshot_revision(tmp_path / "snapshots") is None

    missing = exp.resolve_model_specs(lambda **kwargs: None)
    assert all(row["resolved"] is False for row in missing)
    assert all(row["resolved_path_sha256"] == exp.sha256_text("") for row in missing)
    direct_missing = exp.model_identity_receipt(exp.MODEL_SPECS[0], tmp_path / "absent.gguf")
    assert direct_missing["resolved"] is False


def test_req_infra_6743_host_command_inventory_and_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6743 preserves host probe failures and checks observed free VRAM."""

    good_stdout = (
        "0, GPU-zero, RTX 3090, 24576, 4, 24120, 610.57\n"
        "bad,row\n"
        "x, GPU-bad, RTX 3090, nope, 4, 2, 610.57\n"
    )
    completed = SimpleNamespace(returncode=0, stdout=good_stdout, stderr="")
    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: completed)
    command = exp._run_text_command(["probe"])
    assert command["ok"] is True
    inventory = exp.nvidia_smi_inventory()
    assert [row["index"] for row in inventory["devices"]] == [0]

    def fail_run(*args, **kwargs):
        raise OSError("missing tool")

    monkeypatch.setattr(exp.subprocess, "run", fail_run)
    assert exp._run_text_command(["missing"])["ok"] is False

    models = _resolved_models(tmp_path)
    for model in models:
        model["required_vram_mb"] = 100
        model["device_index"] = 0
    good_inventory = {
        "ok": True,
        "devices": [{"index": 0, "memory_free_mb": 1_000}],
        "stdout": "",
        "stderr": "",
    }
    monkeypatch.setattr(exp, "nvidia_smi_inventory", lambda: good_inventory)
    assert exp.live_preflight(models)["all_passed"] is True

    with monkeypatch.context() as context:
        real_import = builtins.__import__

        def fail_llama_import(name, *args, **kwargs):
            if name == "llama_cpp":
                raise ImportError("fixture missing")
            return real_import(name, *args, **kwargs)

        context.setattr(builtins, "__import__", fail_llama_import)
        blocked = exp.live_preflight([{**models[0], "resolved": False, "device_index": 9}])
    assert blocked["all_passed"] is False
    assert any(check["observed"] is None for check in blocked["checks"])


def test_req_infra_6743_pid_bound_gpu_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-INFRA-6743 parses PID-bound compute rows and ignores malformed samples."""

    inventory = {
        "ok": True,
        "devices": [
            {
                "index": 1,
                "uuid": "GPU-one",
                "name": "RTX 3090",
                "memory_used_mb": 123,
                "memory_free_mb": 24_000,
            }
        ],
    }
    process_receipt = {
        "ok": True,
        "stdout": "7400, GPU-one, 2048, python\nbad\nx, GPU-one, nope, python\n",
    }
    monkeypatch.setattr(exp, "nvidia_smi_inventory", lambda: inventory)
    monkeypatch.setattr(exp, "_run_text_command", lambda command: process_receipt)
    sample = exp.gpu_snapshot_for_pid(1, 7400, "during", clock=lambda: 55)
    assert sample["pid_present"] is True
    assert sample["pid_memory_mb"] == 2048
    assert sample["monotonic_ns"] == 55
    absent = exp.gpu_snapshot_for_pid(2, 99, "after", clock=lambda: 56)
    assert absent["pid_present"] is False
    assert absent["device_uuid"] is None


class _FakeLlama:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False

    def tokenize(self, value: bytes, add_bos: bool = True) -> list[int]:
        return [1, 2] if value else []

    def create_completion(self, *args, **kwargs):
        yield {"choices": [{"text": ""}]}
        yield {"choices": [{"text": "CANARY"}]}
        yield {"choices": [{"text": "", "finish_reason": "length"}]}

    def close(self) -> None:
        self.closed = True


class _EmptyLlama(_FakeLlama):
    close = None

    def create_completion(self, *args, **kwargs):
        yield {"choices": [{}]}


def _worker_payload() -> dict:
    return {
        "model_id": exp.MODEL_SPECS[0]["model_id"],
        "resolved_path": "/cache/model.gguf",
        "device_index": 0,
        "prompt": exp.FIXED_PROMPT,
        "random_seed": exp.RANDOM_SEED,
    }


def _snapshot(device: int, pid: int, phase: str) -> dict:
    return {
        "phase": phase,
        "monotonic_ns": 1,
        "pid": pid,
        "pid_present": phase == "during",
        "device_uuid": "GPU-zero",
        "pid_memory_mb": 2048 if phase == "during" else 0,
    }


def test_req_infra_6743_worker_decode_and_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-INFRA-6743 worker emits first-token data and a structured failure row."""

    receipt = exp._worker_decode(
        _worker_payload(),
        llama_factory=_FakeLlama,
        snapshot_fn=_snapshot,
        supports_gpu_offload=True,
        clock=_clock(),
    )
    assert receipt["raw_output"] == "CANARY"
    assert receipt["first_token_reached"] is True
    assert receipt["stop_reason"] == "length"

    empty = exp._worker_decode(
        _worker_payload(),
        llama_factory=_EmptyLlama,
        snapshot_fn=_snapshot,
        supports_gpu_offload=False,
        clock=_clock(),
    )
    assert empty["first_token_reached"] is False
    assert empty["stop_reason"] == "unknown"
    assert empty["output_tokens"] == 0

    emitted = []
    assert (
        exp.worker_main(
            json.dumps(_worker_payload()),
            llama_factory=_FakeLlama,
            snapshot_fn=_snapshot,
            supports_gpu_offload=True,
            emit=emitted.append,
        )
        == 0
    )
    assert json.loads(emitted[-1])["decode_completed"] is True
    assert exp.worker_main("not-json", emit=emitted.append) == 1
    assert "runtime_error" in json.loads(emitted[-1])

    fake_module = ModuleType("llama_cpp")
    fake_module.Llama = _FakeLlama
    fake_module.llama_cpp = SimpleNamespace(llama_supports_gpu_offload=lambda: True)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    assert (
        exp.worker_main(json.dumps(_worker_payload()), snapshot_fn=_snapshot, emit=emitted.append)
        == 0
    )


def test_req_infra_6743_live_parent_receipt_and_timeout(tmp_path: Path) -> None:
    """REQ-INFRA-6743 parent binds worker exit, stderr layers, and process absence."""

    model = _resolved_models(tmp_path)[0]
    child = {
        "owned_pid": 8_400,
        "parent_pid": 8_000,
        "supports_gpu_offload": True,
        "clocks": {
            "subprocess_started_ns": 100,
            "model_loaded_ns": 200,
            "first_token_ns": 300,
            "decode_complete_ns": 400,
        },
        "gpu_receipts": {
            "before": {
                "pid": 8_400,
                "pid_present": False,
                "device_uuid": "GPU-zero",
                "pid_memory_mb": 0,
            },
            "during": {
                "pid": 8_400,
                "pid_present": True,
                "device_uuid": "GPU-zero",
                "pid_memory_mb": 2048,
            },
        },
        "prompt_tokens": 3,
        "output_tokens": 1,
        "stop_reason": "length",
        "raw_output": "CANARY",
        "raw_output_sha256": exp.sha256_text("CANARY"),
        "first_token_reached": True,
        "decode_completed": True,
    }

    class Process:
        pid = 8_400
        returncode = 0

        def communicate(self, timeout=None):
            return exp.canonical_json(child), "offloaded 64/64 layers to GPU"

    launches = []

    def popen(*args, **kwargs):
        launches.append((args, kwargs))
        return Process()

    after = {
        "phase": "after",
        "monotonic_ns": 450,
        "pid": 8_400,
        "pid_present": False,
        "device_uuid": "GPU-zero",
        "pid_memory_mb": 0,
    }
    row = exp.run_live_model(
        model,
        popen_factory=popen,
        snapshot_fn=lambda *args: after,
        clock=lambda: 500,
        proc_root=tmp_path,
    )
    assert row["gpu_layers"] == {"requested": -1, "offloaded": 64, "total": 64}
    assert row["teardown_completed"] is True
    assert row["peak_vram_mb"] == 2048
    assert launches[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == str(model["device_index"])

    class TimeoutProcess(Process):
        returncode = -9

        def __init__(self) -> None:
            self.calls = 0
            self.killed = False

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["worker"], 1)
            return "noise\n[]", "no layer receipt"

        def kill(self) -> None:
            self.killed = True

    timed = TimeoutProcess()
    blocked = exp.run_live_model(
        model,
        popen_factory=lambda *args, **kwargs: timed,
        snapshot_fn=lambda *args: after,
        clock=lambda: 500,
        proc_root=tmp_path,
    )
    assert timed.killed is True
    assert blocked["runtime_error"] == "worker_json_receipt_missing"
    assert blocked["gpu_layers"]["offloaded"] == 0
    assert blocked["teardown_completed"] is False
    assert exp._parse_worker_stdout("bad\n[]") == {"runtime_error": "worker_json_receipt_missing"}


def test_scenario_infra_6743_negative_phase_and_row_diagnostics(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6743-ACCELERATOR-INTEGRITY names every failed row class."""

    assert {
        "phase_duration_not_positive",
        "missing_phase:preflight",
        "missing_phase:cache_resolved",
        "missing_phase:artifact_write",
    }.issubset(exp.phase_errors([], []))
    rows = _passing_rows(tmp_path)
    phases = _phase_rows(rows)
    phases[5]["model_id"] = "wrong"
    assert "phase_sequence_mismatch" in exp.phase_errors(phases, rows)

    bad = deepcopy(rows[0])
    bad.update(
        {
            "resolved": False,
            "resolved_path_sha256": "bad",
            "prompt_sha256": "bad",
            "raw_output_sha256": "bad",
            "prompt_tokens": 0,
            "output_tokens": 0,
            "first_token_reached": False,
            "decode_completed": False,
            "supports_gpu_offload": False,
            "process_exit_code": 1,
            "owned_process_absent": False,
            "teardown_completed": False,
            "runtime_error": "fixture",
        }
    )
    bad["gpu_layers"]["offloaded"] = 0
    bad["clocks"] = {}
    bad["gpu_receipts"]["before"]["pid_present"] = True
    bad["gpu_receipts"]["after"]["pid"] = 999
    errors = set(exp.model_row_errors(bad))
    assert {
        "cache_not_resolved",
        "resolved_path_hash_mismatch",
        "prompt_hash_mismatch",
        "raw_output_hash_mismatch",
        "prompt_tokens_nonpositive",
        "output_tokens_nonpositive",
        "first_token_missing",
        "decode_incomplete",
        "cuda_offload_unsupported",
        "gpu_layers_not_offloaded",
        "model_clock_missing",
        "before_gpu_receipt_invalid",
        "after_gpu_receipt_invalid",
        "teardown_incomplete",
    }.issubset(errors)
    reversed_clocks = deepcopy(rows[0])
    reversed_clocks["clocks"]["first_token_ns"] = 1
    assert "model_clock_not_monotonic" in exp.model_row_errors(reversed_clocks)


def test_req_infra_6743_partial_and_validator_mutations(tmp_path: Path) -> None:
    """REQ-INFRA-6743 keeps partial evidence valid and rejects derived-field drift."""

    rows = _passing_rows(tmp_path)
    phases = _phase_rows(rows)
    base = exp.build_artifact(
        rows=rows,
        task_phase_rows=phases,
        preflight=_passing_preflight(rows),
        started_ns=1_000,
        artifact_write_ns=3_000,
    )
    partial_rows = deepcopy(rows)
    partial_rows[0]["output_tokens"] = 0
    partial_rows[0]["row_sha256"] = exp.row_checksum(partial_rows[0])
    partial = exp.build_artifact(
        rows=partial_rows,
        task_phase_rows=phases,
        preflight=_passing_preflight(rows),
        started_ns=1_000,
        artifact_write_ns=3_000,
    )
    assert partial["verdict_class"] == "partial"
    assert exp.validate_artifact(partial) == []

    missing = dict(base)
    missing.pop("duration_s")
    assert "missing_field:duration_s" in exp.validate_artifact(missing)
    attacks = {
        "field_principles_mismatch": ("field_principles", {}),
        "inference_substrate_mismatch": ("inference_substrate", "remote"),
        "random_seed_mismatch": ("random_seed", 0),
        "claim_boundary_mismatch": ("claim_boundary", "ranking"),
        "models_used_mismatch": ("models_used", []),
        "row_model_order_mismatch": ("rows", list(reversed(base["rows"]))),
        "phase_clock_boolean_mismatch": ("phase_clock_monotonic", False),
        "duration_mismatch": ("duration_s", 0),
        "live_model_invoked_mismatch": ("live_model_invoked", False),
        "accelerator_receipt_ready_mismatch": ("accelerator_receipt_ready", False),
        "gpu_receipts_mismatch": ("gpu_receipts", []),
        "reproducibility_checksum_mismatch": ("reproducibility_checksum", "bad"),
        "positive_verdict_mismatch": ("verdict_class", "null"),
    }
    for expected, (field, value) in attacks.items():
        mutated = deepcopy(base)
        mutated[field] = value
        assert expected in exp.validate_artifact(mutated)

    no_phases = deepcopy(partial)
    no_phases["task_phase_rows"] = []
    assert "duration_mismatch" in exp.validate_artifact(no_phases)
    wrong_partial = deepcopy(partial)
    wrong_partial["verdict_class"] = "blocked"
    assert "partial_verdict_mismatch" in exp.validate_artifact(wrong_partial)

    blocked = exp.run(
        result_path=tmp_path / "blocked-validator.json",
        resolver=lambda: _resolved_models(tmp_path),
        preflight_fn=lambda models: {
            "all_passed": False,
            "checks": [{"check": "x", "expected": True, "observed": False, "passed": False}],
        },
        clock=_clock(),
    )
    wrong_blocked = deepcopy(blocked)
    wrong_blocked["honest_verdict"] = "complete"
    assert "blocked_verdict_mismatch" in exp.validate_artifact(wrong_blocked)
    no_summary = deepcopy(blocked)
    no_summary["gate_check_summary"] = []
    assert "blocked_gate_summary_missing_observed" in exp.validate_artifact(no_summary)


def test_req_infra_6743_runner_exception_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-INFRA-6743 converts launch exceptions and exposes all CLI modes."""

    models = _resolved_models(tmp_path)
    artifact = exp.run(
        result_path=tmp_path / "exception.json",
        resolver=lambda: models,
        preflight_fn=_passing_preflight,
        model_runner=lambda model: (_ for _ in ()).throw(RuntimeError("launch failed")),
        clock=_clock(),
    )
    assert artifact["verdict_class"] == "partial"
    assert all("RuntimeError" in row["runtime_error"] for row in artifact["rows"])

    monkeypatch.setattr(exp, "worker_main", lambda payload: 7)
    assert exp.main(["--worker", "{}"]) == 7

    valid_path = tmp_path / "valid.json"
    valid = exp.build_artifact(
        rows=_passing_rows(tmp_path),
        task_phase_rows=_phase_rows(_passing_rows(tmp_path)),
        preflight=_passing_preflight(models),
        started_ns=1_000,
        artifact_write_ns=3_000,
    )
    exp.write_artifact(valid_path, valid)
    assert exp.main(["--validate", "--result-path", str(valid_path)]) == 0
    invalid_path = tmp_path / "invalid.json"
    exp.write_artifact(invalid_path, {})
    assert exp.main(["--validate", "--result-path", str(invalid_path)]) == 1

    monkeypatch.setattr(exp, "run", lambda result_path: valid)
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: [])
    assert exp.main(["--result-path", str(valid_path)]) == 0
    assert "accelerator_receipt_ready" in capsys.readouterr().out
