"""Focused admission tests for the exclusive ARC load canary.

Spec refs: REQ-INFRA-6764, SCENARIO-INFRA-6764-*,
REQ-ARC-WMTE-6764, and SCENARIO-ARC-WMTE-6764-*.
"""

from __future__ import annotations

import builtins
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from carnot import experiment_6764_arc_exclusive_load_preflight as exp
from carnot import gpu_lease_phase_journal as lease_api
from carnot.agentic.arc_induction_tools import (
    InductionToolSession,
    dispatch_tool,
    parse_xml_tool_calls,
)
from carnot.experiment_6752_arc_code_carrying_tool_preflight import (
    expected_xml_call,
    fixture_transitions,
)


def _models(tmp_path: Path) -> list[dict]:
    rows = []
    for index, spec in enumerate(exp.MODEL_SPECS):
        path = tmp_path / spec["filename"]
        path.write_bytes(f"model-{index}".encode())
        rows.append(
            {
                **spec,
                "model_path": str(path),
                "model_sha256": spec["expected_sha256"],
                "model_size_bytes": path.stat().st_size,
                "resolved": True,
                "tokenizer": {
                    "source": "llama.cpp_embedded_gguf",
                    "loadable": True,
                    "detail": "fixture tokenizer",
                },
            }
        )
    return rows


def _devices() -> list[dict]:
    return [
        {
            "index": 0,
            "uuid": exp.EXPECTED_GPU_UUIDS[0],
            "name": "NVIDIA GeForce RTX 3090",
            "memory_total_mb": 24_576,
            "memory_used_mb": 900,
            "memory_free_mb": 23_100,
            "temperature_c": 67,
            "utilization_pct": 3,
            "active_compute_processes": [{"pid": 91, "used_memory_mb": 100}],
        },
        {
            "index": 1,
            "uuid": exp.EXPECTED_GPU_UUIDS[1],
            "name": "NVIDIA GeForce RTX 3090",
            "memory_total_mb": 24_576,
            "memory_used_mb": 600,
            "memory_free_mb": 23_300,
            "temperature_c": 72,
            "utilization_pct": 1,
            "active_compute_processes": [],
        },
    ]


def _selfparse_receipt() -> dict:
    raw = expected_xml_call()
    calls, blocks, unparsed = parse_xml_tool_calls(raw)
    call = calls[0]["function"]
    session = InductionToolSession(fixture_transitions(), cell=1)
    result = dispatch_tool(session, call["name"], call["arguments"])
    bounded = "<tool_response>\n" + json.dumps(result, sort_keys=True) + "\n</tool_response>"
    return exp.build_production_selfparse_receipt(
        [
            {
                "raw_emission": raw,
                "parsed_tool": call["name"],
                "parsed_arguments": json.loads(call["arguments"]),
                "dispatch_result": result,
                "bounded_response": bounded,
            }
        ],
        blocks_seen=blocks,
        blocks_unparsed=unparsed,
    )


def _phase_history() -> list[dict]:
    rows = []
    previous = None
    for ordinal, phase in enumerate(lease_api.COMPLETE_PHASE_SEQUENCE):
        rows.append(
            {
                "phase": phase,
                "previous_phase": previous,
                "monotonic_ns": 1_000 + ordinal,
                "event_checksum": f"sha256:{ordinal:064x}",
            }
        )
        previous = phase
    return rows


def _gpu_receipt(tmp_path: Path, index: int) -> dict:
    model = _models(tmp_path)[index]
    owner_pid = 7_000 + index
    server_pid = 8_000 + index
    before = 100
    after = 120
    row = {
        "model_id": model["model_id"],
        "role": model["role"],
        "model_path": model["model_path"],
        "model_sha256": model["model_sha256"],
        "observed_model_path": model["model_path"],
        "inference_substrate": exp.INFERENCE_SUBSTRATE,
        "llama_cpp_cuda": True,
        "server_path": "/fixture/llama-server",
        "server_sha256": f"sha256:{9:064x}",
        "device": deepcopy(_devices()[1]),
        "worker_process": {
            "pid": owner_pid,
            "pid_start_ticks": 123 + index,
            "exit_code": 0,
            "absent_after_exit": True,
        },
        "model_process": {
            "pid": server_pid,
            "pid_start_ticks": 223 + index,
            "exit_code": 0,
            "absent_after_exit": True,
        },
        "lease_owner": {
            "pid": owner_pid,
            "pid_start_ticks": 123 + index,
            "device_uuid": exp.EXPECTED_GPU_UUIDS[1],
            "expected_model": model["model_path"],
            "signals_sent": [],
        },
        "phase_history": _phase_history(),
        "lease_release": {
            "released": True,
            "phase": "terminal_complete",
            "device_uuid": exp.EXPECTED_GPU_UUIDS[1],
            "signals_sent": [],
        },
        "runtime_context": exp.CONTEXT_REQUESTED,
        "gpu_layers": {
            "requested": 999,
            "offloaded": 66 - (index * 25),
            "total": 66 - (index * 25),
        },
        "peak_owned_vram_mb": 17_900 + (index * 3_700),
        "resident_owned_vram_mb": 17_800 + (index * 3_700),
        "duration_s": 10.0 + index,
        "live_model_invoked": True,
        "first_token_observed": True,
        "production_selfparse": _selfparse_receipt(),
        "vram_recovery": exp.build_vram_recovery_receipt(
            before_used_mb=before,
            after_used_mb=after,
            owned_pid_present=False,
        ),
        "full_load": index == 0,
        "transport_canary": index == 1,
        "unrelated_processes_signaled": [],
        "errors": [],
    }
    row["receipt_sha256"] = exp.gpu_receipt_checksum(row)
    return row


def _preflight(models: list[dict], selection: dict, *, passed: bool = True) -> dict:
    checks = [
        {"check": "exp6752_preflight_ready", "expected": True, "observed": True, "passed": True},
        {"check": "exp6647_task_owned_admission", "expected": 1.0, "observed": 1.0, "passed": True},
        {
            "check": "least_used_eligible_rtx3090",
            "expected": {"free_vram_mb_at_least": exp.FROZEN_FREE_VRAM_THRESHOLD_MB},
            "observed": selection.get("selected_device"),
            "passed": passed,
        },
    ]
    return {
        "all_passed": passed,
        "checks": checks,
        "models": models,
        "device_inventory_before": _devices(),
        "device_selection_receipt": selection,
    }


def test_scenario_infra_6764_least_used_selection_is_frozen() -> None:
    """SCENARIO-INFRA-6764-LEAST-USED-SELECTION ranks only eligible fixed GPUs."""
    receipt = exp.rank_eligible_devices(_devices())
    assert receipt["frozen_free_vram_threshold_mb"] == 22_610
    assert receipt["rank_policy"] == ["free_vram_desc", "temperature_asc", "active_compute_asc"]
    assert receipt["selected_device"]["uuid"] == exp.EXPECTED_GPU_UUIDS[1]
    assert [row["uuid"] for row in receipt["ranked_eligible_devices"]] == [
        exp.EXPECTED_GPU_UUIDS[1],
        exp.EXPECTED_GPU_UUIDS[0],
    ]

    tied = _devices()
    tied[0]["memory_free_mb"] = tied[1]["memory_free_mb"]
    assert exp.rank_eligible_devices(tied)["selected_device"]["uuid"] == exp.EXPECTED_GPU_UUIDS[0]

    blocked = _devices()
    for row in blocked:
        row["memory_free_mb"] = exp.FROZEN_FREE_VRAM_THRESHOLD_MB - 1
    blocked_receipt = exp.rank_eligible_devices(blocked)
    assert blocked_receipt["selected_device"] is None
    assert blocked_receipt["eligible_device_count"] == 0


def test_scenario_infra_6764_lease_excludes_a_second_owner(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6764-LEASE-EXCLUSION-AND-NO-PREEMPTION uses GpuLease."""
    selected = _devices()[1]
    owner = exp.acquire_selected_lease(
        runtime_dir=tmp_path,
        task_id="exp6764-owner",
        selected_device=selected,
        expected_model="model-a.gguf",
    )
    with pytest.raises(lease_api.LeaseBusy):
        exp.acquire_selected_lease(
            runtime_dir=tmp_path,
            task_id="exp6764-contender",
            selected_device=selected,
            expected_model="model-b.gguf",
        )
    assert owner.owner_receipt()["signals_sent"] == []
    owner.transition("terminal_blocked")
    assert owner.release()["signals_sent"] == []


class _OwnedProcess:
    def __init__(self, *, survives_term: bool) -> None:
        self.pid = 55
        self.returncode = None
        self.survives_term = survives_term
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self.survives_term:
            self.returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float):
        if self.returncode is None:
            raise exp.subprocess.TimeoutExpired(["owned"], timeout)
        return self.returncode


@pytest.mark.parametrize("survives_term", [False, True])
def test_scenario_infra_6764_cleanup_targets_only_owned_child(survives_term: bool) -> None:
    """SCENARIO-INFRA-6764-LEASE-EXCLUSION-AND-NO-PREEMPTION never accepts foreign PIDs."""
    process = _OwnedProcess(survives_term=survives_term)
    receipt = exp.terminate_owned_process(process, terminate_timeout_s=0.01)
    assert receipt["pid"] == 55
    assert receipt["terminate_sent"] is True
    assert receipt["kill_sent"] is survives_term
    assert receipt["unrelated_processes_signaled"] == []
    assert process.terminate_calls == 1
    assert process.kill_calls == int(survives_term)
    exited = _OwnedProcess(survives_term=False)
    exited.returncode = 0
    exited_receipt = exp.terminate_owned_process(exited)
    assert exited_receipt["absent_after_exit"] is True
    assert exited.terminate_calls == 0


def test_scenario_arc_wmte_6764_production_selfparse_dispatch_is_real() -> None:
    """SCENARIO-ARC-WMTE-6764-FULL-LOAD-SELFPARSE uses parser, dispatch, and bounds."""
    receipt = _selfparse_receipt()
    assert receipt["production_route"] == "induce_with_tool_loop/selfparse/dispatch_tool"
    assert receipt["parsed_tool"] == "find_objects"
    assert receipt["parsed_arguments"]["t"] == 0
    assert receipt["parsed_arguments"]["max_objects"] == 8
    assert receipt["dispatch_result"]["ok"] is True
    assert receipt["bounded_response_bytes"] > 0
    assert receipt["blocks_seen"] == 1
    assert receipt["blocks_unparsed"] == 0
    assert receipt["success"] is True
    assert exp.production_selfparse_errors(receipt) == []


def test_scenario_arc_wmte_6764_owned_environment_sets_32k_before_load(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6764 fixes 32K, CUDA, model, and selfparse in each worker."""
    model = _models(tmp_path)[0]
    env = exp.worker_environment({"KEEP": "yes"}, model, _devices()[1], port=45_001)
    assert env["KEEP"] == "yes"
    assert env["CARNOT_ARC_INDUCE_N_CTX"] == "32768"
    assert env["CARNOT_ARC_INDUCE_TOOL_LOOP"] == "selfparse"
    assert env["CARNOT_ARC_GENERATOR_CUDA_GPU"] == "1"
    assert env["CARNOT_ARC_GENERATOR_REQUIRE_CUDA"] == "1"
    assert env["CARNOT_ARC_GGUF_PATH"] == model["model_path"]
    assert env["CARNOT_ARC_EXCLUSIVE_PORT"] == "45001"


def test_scenario_infra_6764_teardown_and_vram_recovery_are_required(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6764-TEARDOWN-RECOVERY fails closed on residue or a live process."""
    receipt = _gpu_receipt(tmp_path, 0)
    assert exp.gpu_receipt_errors(receipt) == []
    assert receipt["vram_recovery"]["passed"] is True

    for field, value in (
        ("model_process", {**receipt["model_process"], "absent_after_exit": False}),
        ("lease_release", {**receipt["lease_release"], "released": False}),
        (
            "vram_recovery",
            exp.build_vram_recovery_receipt(
                before_used_mb=100,
                after_used_mb=700,
                owned_pid_present=False,
            ),
        ),
    ):
        changed = deepcopy(receipt)
        changed[field] = value
        changed["receipt_sha256"] = exp.gpu_receipt_checksum(changed)
        assert exp.gpu_receipt_errors(changed)


def test_scenario_arc_wmte_6764_ready_requires_separate_complete_canaries(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6764-FLAGSHIP-CANARY keeps two receipts and timings separate."""
    receipts = [_gpu_receipt(tmp_path, 0), _gpu_receipt(tmp_path, 1)]
    assert exp.reduce_arc_exclusive_load_ready(receipts) is True
    changed = deepcopy(receipts)
    changed[1]["first_token_observed"] = False
    changed[1]["receipt_sha256"] = exp.gpu_receipt_checksum(changed[1])
    assert exp.reduce_arc_exclusive_load_ready(changed) is False


def test_scenario_infra_6764_blocked_artifact_stops_before_workers(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6764-BLOCKED-ARTIFACT preserves the failed observed value."""
    models = _models(tmp_path)
    devices = _devices()
    for row in devices:
        row["memory_free_mb"] = 6_235
    selection = exp.rank_eligible_devices(devices)
    preflight = _preflight(models, selection, passed=False)
    preflight["device_inventory_before"] = devices
    calls = []
    artifact = exp.run(
        result_path=tmp_path / "blocked.json",
        date="20260829",
        preflight_fn=lambda: preflight,
        worker_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        clock=iter((1_000, 2_000)).__next__,
    )
    assert calls == []
    assert artifact["honest_verdict"] == "complete_blocked_arc_exclusive_load"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["arc_exclusive_load_ready"] is False
    assert artifact["live_model_invoked"] is False
    assert artifact["gpu_receipts"] == []
    assert artifact["gate_check_summary"][-1]["observed"] is None
    assert artifact["unrelated_processes_signaled"] == []
    assert exp.validate_artifact(artifact) == []
    assert json.loads((tmp_path / "blocked.json").read_text()) == artifact


def test_scenario_infra_6764_missing_exact_model_writes_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6764-BLOCKED-ARTIFACT retains a missing-model observation."""
    models = _models(tmp_path)
    models[0].update(
        model_path=None,
        model_sha256="missing",
        model_size_bytes=0,
        resolved=False,
        tokenizer={
            "source": "llama.cpp_embedded_gguf",
            "loadable": False,
            "detail": "exact cached GGUF missing",
        },
    )
    selection = exp.rank_eligible_devices(_devices())
    preflight = _preflight(models, selection)
    preflight["all_passed"] = False
    preflight["checks"].append(
        {
            "check": "exact_cached_model:qwen3.8_27b",
            "expected": exp.MODEL_SPECS[0]["expected_sha256"],
            "observed": "missing",
            "passed": False,
        }
    )
    calls = []
    artifact = exp.run(
        result_path=tmp_path / "missing-model.json",
        date="20260829",
        preflight_fn=lambda: preflight,
        worker_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        clock=iter((1_000, 2_000)).__next__,
    )
    assert calls == []
    assert artifact["honest_verdict"] == "complete_blocked_arc_exclusive_load"
    assert artifact["gate_check_summary"] == [preflight["checks"][-1]]
    assert artifact["models_used"][0]["model_sha256"] == "missing"
    assert exp.validate_artifact(artifact) == []
    assert json.loads((tmp_path / "missing-model.json").read_text()) == artifact


def test_scenario_arc_wmte_6764_admission_only_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6764-ADMISSION-ONLY reduces lifecycle evidence, not quality."""
    models = _models(tmp_path)
    selection = exp.rank_eligible_devices(_devices())
    preflight = _preflight(models, selection)
    receipts = [_gpu_receipt(tmp_path, 0), _gpu_receipt(tmp_path, 1)]
    artifact = exp.run(
        result_path=tmp_path / "complete.json",
        date="20260829",
        preflight_fn=lambda: preflight,
        worker_runner=lambda model, device, port, runtime_dir: deepcopy(
            receipts[0 if model["model_id"] == receipts[0]["model_id"] else 1]
        ),
        clock=iter((1_000_000_000, 4_000_000_000)).__next__,
    )
    assert artifact["arc_exclusive_load_ready"] is True
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"] == "complete_arc_exclusive_load_ready"
    assert artifact["model_specs"] == artifact["models_used"]
    assert artifact["runtime_context_by_model"] == {row["model_id"]: 32_768 for row in receipts}
    assert len(artifact["lease_owner_receipts"]) == 2
    assert len(artifact["lease_release_receipts"]) == 2
    assert len(artifact["vram_recovery_receipts"]) == 2
    assert len(artifact["phase_rows"]) == 16
    assert artifact["rows"] == artifact["phase_rows"]
    assert artifact["owned_processes_terminated"] is True
    assert artifact["unrelated_processes_signaled"] == []
    assert artifact["verifier_is_oracle"] is False
    assert "quality" in artifact["claim_boundary"]
    assert "solve" in artifact["claim_boundary"]
    assert "pooled" in artifact["claim_boundary"]
    assert set(artifact).issubset(artifact["field_principles"])
    assert exp.validate_artifact(artifact) == []


def test_req_arc_wmte_6764_validator_rejects_substitution_and_claim_drift(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-6764 rejects a changed model, pooled claim, or forged readiness."""
    models = _models(tmp_path)
    selection = exp.rank_eligible_devices(_devices())
    receipts = [_gpu_receipt(tmp_path, 0), _gpu_receipt(tmp_path, 1)]
    artifact = exp.build_artifact(
        date="20260829",
        preflight=_preflight(models, selection),
        gpu_receipts=receipts,
        started_ns=1,
        finished_ns=2,
    )
    assert exp.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["models_used"][0]["model_id"] = "legacy/model"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "models_used" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["claim_boundary"] = "ARC quality passed"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "claim_boundary" in exp.validate_artifact(changed)


def test_req_infra_6764_phase_rows_bind_owner_and_model(tmp_path: Path) -> None:
    """REQ-INFRA-6764 emits one ordered row for every model and lease phase."""
    receipt = _gpu_receipt(tmp_path, 0)
    rows = exp.phase_rows_for_receipt(receipt)
    assert [row["phase"] for row in rows] == list(lease_api.COMPLETE_PHASE_SEQUENCE)
    assert all(row["model_id"] == receipt["model_id"] for row in rows)
    assert all(row["owner_pid"] == receipt["lease_owner"]["pid"] for row in rows)
    reversed_receipt = deepcopy(receipt)
    reversed_receipt["phase_history"] = list(reversed(receipt["phase_history"]))
    reversed_receipt["receipt_sha256"] = exp.gpu_receipt_checksum(reversed_receipt)
    assert "phase_sequence" in exp.gpu_receipt_errors(reversed_receipt)


def test_req_infra_6764_all_required_artifact_fields_have_principles() -> None:
    """REQ-INFRA-6764 keeps every required field and gate reason self-explanatory."""
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(exp.FIELD_PRINCIPLES)
    assert exp.INFERENCE_SUBSTRATE == "task-owned local llama.cpp CUDA GGUF"
    assert exp.VRAM_RECOVERY_TOLERANCE_MB == 512
    assert exp.VERDICT_CLASSES == {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }


def test_req_infra_6764_worker_launcher_passes_frozen_device_and_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6764 launches one fresh worker and retains its exact output receipt."""
    model = _models(tmp_path)[0]
    device = _devices()[1]
    expected = _gpu_receipt(tmp_path, 0)
    captured = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.pid = 90_001
            self.returncode = 0
            self.command = command
            captured["command"] = command
            captured["env"] = kwargs["env"]
            output = Path(command[command.index("--worker-output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(expected))

        def communicate(self, timeout=None):
            return "ok", ""

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    def fake_popen(command, **kwargs):
        return FakePopen(command, **kwargs)

    monkeypatch.setattr(exp.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(exp.lease_api, "proc_start_ticks", lambda pid: 321)
    row = exp.run_model_worker(
        model,
        device,
        port=45_002,
        runtime_dir=tmp_path / "runtime",
        timeout_s=1,
    )
    assert "--worker" in captured["command"]
    assert captured["env"]["CARNOT_ARC_INDUCE_N_CTX"] == "32768"
    assert captured["env"]["CARNOT_ARC_GENERATOR_CUDA_GPU"] == "1"
    assert row["model_id"] == expected["model_id"]


def test_req_infra_6764_host_probe_adapters_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6764 host adapters retain exact files, inventory, resources, and ports."""
    payload = tmp_path / "payload.json"
    payload.write_text('{"ok": true}')
    assert exp._load_json(payload) == {"ok": True}
    payload.write_text("[")
    assert exp._load_json(payload) == {}
    payload.unlink()
    assert exp._load_json(payload) == {}

    blob = tmp_path / "blob"
    blob.write_bytes(b"abc")
    assert exp.sha256_file(blob) == exp.sha256_text("abc")
    assert exp.sha256_file(tmp_path / "missing") == "missing"

    completed = SimpleNamespace(returncode=0, stdout="x" * 17_000, stderr="warning")
    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: completed)
    command = exp._run_command(("probe",))
    assert command["exit_code"] == 0
    assert len(command["stdout"]) == 16_000

    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(["probe"], 1)

    monkeypatch.setattr(exp.subprocess, "run", timed_out)
    assert exp._run_command(("probe",))["exit_code"] == 124
    monkeypatch.setattr(
        exp.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
    )
    assert exp._run_command(("probe",))["exit_code"] == 127

    model_module = sys.modules["carnot.agentic.arc_executable_world_model"]
    monkeypatch.setattr(exp, "resolve_cached_gguf", lambda *args: "/cache/moe.gguf")
    monkeypatch.setattr(model_module, "_resolve_gguf", lambda name: f"/cache/{name}.gguf")
    assert exp._model_path(exp.MODEL_SPECS[1]) == "/cache/moe.gguf"
    assert exp._model_path(exp.MODEL_SPECS[0]).endswith("Qwen3.8-27B.gguf")

    models = _models(tmp_path)
    by_id = {row["model_id"]: row for row in models}
    monkeypatch.setattr(exp, "_model_path", lambda spec: by_id[spec["model_id"]]["model_path"])
    monkeypatch.setattr(
        exp,
        "sha256_file",
        lambda path: next(
            row["expected_sha256"] for row in models if row["filename"] == Path(path).name
        ),
    )
    monkeypatch.setattr(exp, "gguf_tokenizer_loadable", lambda path: (True, "embedded"))
    resolved = exp.resolve_model_specs()
    assert all(row["resolved"] and row["tokenizer"]["loadable"] for row in resolved)

    responses = iter(
        (
            {
                "stdout": (
                    "bad\n"
                    f"0, {exp.EXPECTED_GPU_UUIDS[0]}, NVIDIA GeForce RTX 3090, "
                    "24576, 100, 24476, 60, 1\n"
                    f"x, {exp.EXPECTED_GPU_UUIDS[1]}, NVIDIA GeForce RTX 3090, "
                    "bad, 100, 24476, 60, 1"
                )
            },
            {
                "stdout": (
                    "bad\n"
                    f"{exp.EXPECTED_GPU_UUIDS[0]}, 123, llama-server, 100\n"
                    f"{exp.EXPECTED_GPU_UUIDS[0]}, bad, llama-server, 100"
                )
            },
        )
    )
    monkeypatch.setattr(exp, "_run_command", lambda *args, **kwargs: next(responses))
    inventory = exp.nvidia_smi_inventory()
    assert inventory["devices"][0]["active_compute_processes"][0]["pid"] == 123

    wrong = deepcopy(_devices()[0])
    wrong["uuid"] = "GPU-wrong"
    ranked = exp.rank_eligible_devices([wrong])
    assert ranked["evaluated_devices"][0]["ineligibility_reasons"] == ["unexpected_device_identity"]

    ports = exp.choose_free_ports(2)
    assert len(set(ports)) == 2 and all(exp.port_is_free(port) for port in ports)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        assert exp.port_is_free(listener.getsockname()[1]) is False
    finally:
        listener.close()

    resources = exp._host_resources(tmp_path)
    assert resources["ram_total_bytes"] > 0 and resources["disk_total_bytes"] > 0
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_text", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
        assert exp._host_resources(tmp_path)["ram_total_bytes"] == 0


def test_req_infra_6764_llama_and_precondition_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6764 checks every immutable input before opening a lease."""
    server = tmp_path / "llama-server"
    server.write_bytes(b"server")
    server.chmod(0o755)
    monkeypatch.setenv("CARNOT_LLAMA_SERVER", str(server))
    monkeypatch.setattr(
        exp,
        "_run_command",
        lambda *args, **kwargs: {"stdout": "libggml-cuda libcuda", "stderr": "", "exit_code": 0},
    )
    import llama_cpp

    monkeypatch.setattr(llama_cpp.llama_cpp, "llama_supports_gpu_offload", lambda: True)
    assert exp._llama_cpp_receipt()["cuda_linked"] is True
    with monkeypatch.context() as scoped:
        scoped.setitem(sys.modules, "llama_cpp", None)
        assert "ModuleNotFoundError" in str(exp._llama_cpp_receipt()["python_cuda_offload"])

    models = _models(tmp_path)
    prior_6752 = {
        "arc_context_tool_preflight_ready": True,
        "context_observed_by_model": {row["model_id"]: exp.CONTEXT_REQUESTED for row in models},
        "models_used": [
            {"model_id": row["model_id"], "model_sha256": row["model_sha256"]} for row in models
        ],
    }
    prior_6647 = {"task_owned_admission_ready_score": 1.0}
    monkeypatch.setattr(
        exp,
        "_load_json",
        lambda path: prior_6752 if path == exp.EXP6752_PATH else prior_6647,
    )
    monkeypatch.setattr(exp.exp6752, "validate_artifact", lambda artifact: [])
    monkeypatch.setattr(exp.exp6647, "validate_artifact", lambda artifact: [])
    monkeypatch.setattr(exp, "resolve_model_specs", lambda: deepcopy(models))
    monkeypatch.setattr(
        exp,
        "nvidia_smi_inventory",
        lambda: {
            "devices": _devices(),
            "device_query": {"ok": True},
            "process_query": {"ok": True},
        },
    )
    monkeypatch.setattr(
        exp,
        "_llama_cpp_receipt",
        lambda: {
            "exists": True,
            "executable": True,
            "cuda_linked": True,
            "python_cuda_offload": True,
        },
    )
    monkeypatch.setattr(
        exp,
        "_host_resources",
        lambda root: {
            "ram_available_bytes": exp.RAM_AVAILABLE_FLOOR_BYTES,
            "disk_free_bytes": exp.DISK_FREE_FLOOR_BYTES,
        },
    )
    monkeypatch.setattr(exp, "choose_free_ports", lambda count: [45_010, 45_011])
    monkeypatch.setattr(exp, "port_is_free", lambda port: True)
    monkeypatch.setattr(exp, "sha256_file", lambda path: f"sha256:{1:064x}")
    receipt = exp.collect_preconditions(tmp_path)
    assert receipt["all_passed"] is True
    assert len(receipt["checks"]) == 15
    assert receipt["device_selection_receipt"]["selected_device"]["index"] == 1

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "carnot.agentic.arc_executable_world_model":
            raise ImportError("blocked fixture")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    blocked = exp.collect_preconditions(tmp_path)
    imports_check = next(
        row for row in blocked["checks"] if row["check"] == "production_selfparse_imports"
    )
    assert imports_check["passed"] is False


def test_req_infra_6764_process_snapshots_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFRA-6764-TEARDOWN-RECOVERY polls only the recorded owner."""
    device = _devices()[1]
    device["active_compute_processes"] = [{"pid": 88, "used_memory_mb": 321}]
    monkeypatch.setattr(
        exp,
        "nvidia_smi_inventory",
        lambda: {"devices": [device], "device_query": {}, "process_query": {}},
    )
    snapshot = exp._gpu_snapshot(device["uuid"], 88)
    assert snapshot["owned_pid_present"] is True
    assert snapshot["owned_pid_vram_mb"] == 321

    process = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    identity = exp._process_identity(process)
    assert identity["pid"] == os.getpid() and identity["executable"]
    with monkeypatch.context() as scoped:
        scoped.setattr(exp.os, "readlink", lambda path: (_ for _ in ()).throw(OSError()))
        assert exp._process_identity(process)["executable"] == ""
    assert exp._empty_model_process()["absent_after_exit"] is True

    samples = iter(
        (
            {"memory_used_mb": 900, "owned_pid_present": True, "observed_monotonic_ns": 1},
            {"memory_used_mb": 100, "owned_pid_present": False, "observed_monotonic_ns": 2},
        )
    )
    monkeypatch.setattr(exp, "_gpu_snapshot", lambda *args: next(samples))
    monotonic = iter((0.0, 1.0))
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(exp.time, "sleep", lambda seconds: None)
    recovery, after = exp._wait_for_vram_recovery(device["uuid"], 88, 100, timeout_s=10)
    assert recovery["passed"] is True and after["observed_monotonic_ns"] == 2


class _FakeLease:
    def __init__(self, model: dict, device: dict, *, fail_phase: str | None = None) -> None:
        self.document = {"phase": "preflight"}
        self.model = model
        self.device = device
        self.fail_phase = fail_phase
        self.journal_path = Path("/fixture/journal")
        self.closed = False

    def owner_receipt(self) -> dict:
        return {
            "pid": 400,
            "pid_start_ticks": 500,
            "device_uuid": self.device["uuid"],
            "expected_model": self.model["model_path"],
            "signals_sent": [],
        }

    def transition(self, phase: str, **kwargs) -> None:
        if phase == self.fail_phase:
            raise lease_api.LeaseError(f"failed {phase}")
        self.document["phase"] = phase

    def release(self) -> dict:
        return {
            "released": True,
            "phase": self.document["phase"],
            "device_uuid": self.device["uuid"],
            "signals_sent": [],
        }

    def close(self) -> None:
        self.closed = True


def _install_live_worker_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "success",
    fail_phase: str | None = None,
) -> tuple[dict, dict, _FakeLease]:
    model = _models(tmp_path)[0]
    device = _devices()[1]
    fake_lease = _FakeLease(model, device, fail_phase=fail_phase)
    model_process = _OwnedProcess(survives_term=False)
    model_process.pid = 401
    log_path = tmp_path / "server.log"
    log_path.write_text("offload fixture")

    class FakeProposer:
        def __init__(self, **kwargs):
            self.port = kwargs["port"] + int(mode == "changed_port")
            self._proc = None if mode == "missing_process" else model_process
            self._stderr_log_path = log_path
            self.last_tool_loop_stats = {}

        def _ensure_server(self):
            return mode != "load_failed"

        def observed_n_ctx(self):
            return exp.CONTEXT_REQUESTED

        def observed_model_path(self):
            return model["model_path"]

    model_module = sys.modules["carnot.agentic.arc_executable_world_model"]
    loop_module = sys.modules["carnot.agentic.arc_induction_tool_loop"]
    monkeypatch.setattr(model_module, "LocalGGUFProposer", FakeProposer)

    valid = _selfparse_receipt()

    def fake_induce(proposer, *args, tool_event_sink, **kwargs):
        tool_event_sink.append(
            {
                "raw_emission": valid["raw_emission"],
                "parsed_tool": valid["parsed_tool"],
                "parsed_arguments": valid["parsed_arguments"],
                "dispatch_result": valid["dispatch_result"],
                "bounded_response": valid["bounded_response"],
            }
        )
        proposer.last_tool_loop_stats = {
            "selfparse_blocks_seen": 1,
            "selfparse_blocks_unparsed": 0,
        }
        time.sleep(0.12)

    monkeypatch.setattr(loop_module, "induce_with_tool_loop", fake_induce)
    monkeypatch.setattr(
        exp.lease_api, "current_process_identity", lambda: {"pid": 400, "pid_start_ticks": 500}
    )
    monkeypatch.setattr(exp, "acquire_selected_lease", lambda **kwargs: fake_lease)
    monkeypatch.setattr(
        exp,
        "nvidia_smi_inventory",
        lambda: {"devices": _devices(), "device_query": {}, "process_query": {}},
    )
    monkeypatch.setattr(exp, "port_is_free", lambda port: mode != "busy_port")
    monkeypatch.setattr(
        exp,
        "_llama_cpp_receipt",
        lambda: {"cuda_linked": mode != "no_cuda", "path": "/llama", "sha256": f"sha256:{1:064x}"},
    )

    def snapshot(uuid, owned_pid=0):
        return {
            **device,
            "owned_pid": owned_pid,
            "owned_pid_present": bool(owned_pid) and mode != "no_cuda",
            "owned_pid_vram_mb": 20_000 if owned_pid and mode != "no_cuda" else 0,
            "observed_monotonic_ns": 99,
        }

    monkeypatch.setattr(exp, "_gpu_snapshot", snapshot)
    monkeypatch.setattr(
        exp,
        "_wait_for_vram_recovery",
        lambda *args, **kwargs: (
            exp.build_vram_recovery_receipt(
                before_used_mb=device["memory_used_mb"],
                after_used_mb=device["memory_used_mb"],
                owned_pid_present=False,
            ),
            {**device, "observed_monotonic_ns": 100},
        ),
    )
    monkeypatch.setattr(
        exp,
        "_process_identity",
        lambda process: {
            "pid": process.pid,
            "pid_start_ticks": 501,
            "parent_pid": 400,
            "executable": "/llama",
            "exit_code": None,
            "absent_after_exit": False,
        },
    )
    monkeypatch.setattr(
        exp.exp6752,
        "_gpu_layers_from_log",
        lambda text, requested: {"requested": requested, "offloaded": 66, "total": 66},
    )
    monkeypatch.setattr(
        exp.lease_api, "read_journal", lambda path: {"phase_history": _phase_history()}
    )
    return model, device, fake_lease


def test_scenario_arc_wmte_6764_live_worker_success_is_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-6764-FULL-LOAD-SELFPARSE exercises the live worker lifecycle."""
    model, device, _ = _install_live_worker_fakes(tmp_path, monkeypatch)
    receipt = exp.run_live_model_worker(model, device, port=45_020, lease_runtime_dir=tmp_path)
    assert receipt["errors"] == []
    assert receipt["live_model_invoked"] is True
    assert receipt["lease_release"]["phase"] == "terminal_complete"
    assert receipt["model_process"]["absent_after_exit"] is True
    assert receipt["peak_owned_vram_mb"] == 20_000


@pytest.mark.parametrize(
    "mode,error",
    [
        ("busy_port", "selected_port_no_longer_free"),
        ("load_failed", "llama_server_load_failed"),
        ("changed_port", "llama_server_changed_frozen_port"),
        ("missing_process", "llama_server_process_missing"),
        ("no_cuda", "owner_bound_cuda_residency_missing"),
    ],
)
def test_req_infra_6764_live_worker_blocks_on_runtime_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, error: str
) -> None:
    """REQ-INFRA-6764 blocks fresh-worker drift without signaling unrelated work."""
    model, device, _ = _install_live_worker_fakes(tmp_path, monkeypatch, mode=mode)
    receipt = exp.run_live_model_worker(model, device, port=45_021, lease_runtime_dir=tmp_path)
    assert any(error in item for item in receipt["errors"])
    assert receipt["unrelated_processes_signaled"] == []


def test_req_infra_6764_live_worker_blocks_if_selection_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6764 rechecks that the leased UUID remains first eligible."""
    model, device, _ = _install_live_worker_fakes(tmp_path, monkeypatch)
    changed = _devices()
    changed[0]["memory_free_mb"] = 24_000
    monkeypatch.setattr(
        exp,
        "nvidia_smi_inventory",
        lambda: {"devices": changed, "device_query": {}, "process_query": {}},
    )
    receipt = exp.run_live_model_worker(model, device, port=45_022, lease_runtime_dir=tmp_path)
    assert "selected_device_no_longer_first_eligible" in receipt["errors"][0]


@pytest.mark.parametrize("fail_phase", ["unloading", "validating"])
def test_req_infra_6764_live_worker_retains_lease_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_phase: str
) -> None:
    """REQ-INFRA-6764 retains transition errors and closes its owner lease."""
    model, device, lease = _install_live_worker_fakes(tmp_path, monkeypatch, fail_phase=fail_phase)
    receipt = exp.run_live_model_worker(model, device, port=45_023, lease_runtime_dir=tmp_path)
    assert any("LeaseError" in item for item in receipt["errors"])
    assert lease.closed is True


def test_req_infra_6764_live_worker_retains_journal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6764 fails closed when its durable phase journal cannot be read."""
    model, device, _ = _install_live_worker_fakes(tmp_path, monkeypatch)
    monkeypatch.setattr(
        exp.lease_api,
        "read_journal",
        lambda path: (_ for _ in ()).throw(lease_api.LeaseError("journal")),
    )
    receipt = exp.run_live_model_worker(model, device, port=45_024, lease_runtime_dir=tmp_path)
    assert any("journal" in item for item in receipt["errors"])


def test_req_infra_6764_worker_timeout_and_group_identity_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-6764-LEASE-EXCLUSION-AND-NO-PREEMPTION binds group signals to PID start."""
    process = SimpleNamespace(pid=90_100, wait=lambda timeout: 0)
    monkeypatch.setattr(exp.lease_api, "proc_start_ticks", lambda pid: 2)
    mismatch = exp._terminate_worker_group(process, 1)
    assert mismatch["identity_mismatch"] is True and mismatch["term_sent"] is False

    sent = []
    monkeypatch.setattr(exp.lease_api, "proc_start_ticks", lambda pid: 1)
    monkeypatch.setattr(exp.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    exp._terminate_worker_group(process, 1)
    assert sent == [(90_100, exp.signal.SIGTERM)]

    class TimeoutProcess:
        pid = 90_101

        def __init__(self):
            self.calls = 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["worker"], timeout)
            return -9

    sent.clear()
    exp._terminate_worker_group(TimeoutProcess(), 1)
    assert sent == [(90_101, exp.signal.SIGTERM), (90_101, exp.signal.SIGKILL)]

    monkeypatch.setattr(
        exp.os,
        "killpg",
        lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()),
    )
    assert exp._terminate_worker_group(process, 1)["term_sent"] is False

    model = _models(tmp_path)[0]
    device = _devices()[1]

    class TimeoutPopen:
        def __init__(self, command, **kwargs):
            self.pid = 90_102
            self.returncode = -15
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["worker"], timeout)
            return "", "timed out"

        def poll(self):
            return self.returncode

    monkeypatch.setattr(exp.subprocess, "Popen", TimeoutPopen)
    monkeypatch.setattr(exp.lease_api, "proc_start_ticks", lambda pid: 1)
    monkeypatch.setattr(
        exp,
        "_terminate_worker_group",
        lambda process, ticks: {"term_sent": True, "unrelated_processes_signaled": []},
    )
    receipt = exp.run_model_worker(model, device, 45_025, tmp_path / "runtime", timeout_s=0.01)
    assert receipt["errors"] and receipt["worker_process"]["timeout_cleanup"]["term_sent"]


def test_req_arc_wmte_6764_receipt_validator_failure_matrix(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6764 rejects every forged full-load admission component."""
    valid = _gpu_receipt(tmp_path, 0)
    mutations = {
        "receipt_sha256": lambda row: row.update(receipt_sha256="bad"),
        "model_id": lambda row: row.update(model_id="legacy/model"),
        "role": lambda row: row.update(role="wrong"),
        "model_sha256": lambda row: row.update(model_sha256="bad"),
        "observed_model_path": lambda row: row.update(observed_model_path="bad"),
        "inference_substrate": lambda row: row.update(inference_substrate="cpu"),
        "llama_cpp_cuda": lambda row: row.update(llama_cpp_cuda=False),
        "server_sha256": lambda row: row.update(server_sha256="bad"),
        "device_uuid": lambda row: row["device"].update(uuid="GPU-wrong"),
        "lease_owner": lambda row: row["lease_owner"].update(pid=-1),
        "worker_process": lambda row: row["worker_process"].update(exit_code=1),
        "model_process": lambda row: row["model_process"].update(exit_code=None),
        "phase_sequence": lambda row: row.update(phase_history=[]),
        "lease_release": lambda row: row["lease_release"].update(released=False),
        "runtime_context": lambda row: row.update(runtime_context=1),
        "gpu_layers": lambda row: row.update(gpu_layers={}),
        "peak_owned_vram_mb": lambda row: row.update(peak_owned_vram_mb=0),
        "resident_owned_vram_mb": lambda row: row.update(resident_owned_vram_mb=0),
        "duration_s": lambda row: row.update(duration_s=0),
        "first_token": lambda row: row.update(first_token_observed=False),
        "production_selfparse": lambda row: row.update(production_selfparse={}),
        "vram_recovery": lambda row: row.update(vram_recovery={}),
        "unrelated_processes_signaled": lambda row: row.update(unrelated_processes_signaled=[999]),
        "errors": lambda row: row.update(errors=["failed"]),
        "full_load": lambda row: row.update(full_load=False),
    }
    for expected_error, mutate in mutations.items():
        changed = deepcopy(valid)
        mutate(changed)
        if expected_error != "receipt_sha256":
            changed["receipt_sha256"] = exp.gpu_receipt_checksum(changed)
        assert expected_error in exp.gpu_receipt_errors(changed)

    canary = _gpu_receipt(tmp_path, 1)
    canary["transport_canary"] = False
    canary["receipt_sha256"] = exp.gpu_receipt_checksum(canary)
    assert "transport_canary" in exp.gpu_receipt_errors(canary)
    assert exp.production_selfparse_errors({}) == [
        "production_route",
        "parsed_tool",
        "parsed_arguments",
        "dispatch_result",
        "bounded_response",
        "xml_blocks",
        "raw_emission_sha256",
        "bounded_response_sha256",
        "transcript_sha256",
    ]
    assert exp.phase_rows_for_receipt({"phase_history": [None]}) == []


def test_req_infra_6764_artifact_validator_failure_matrix(tmp_path: Path) -> None:
    """REQ-INFRA-6764 recomputes every top-level readiness and row-consistency field."""
    models = _models(tmp_path)
    receipts = [_gpu_receipt(tmp_path, 0), _gpu_receipt(tmp_path, 1)]
    artifact = exp.build_artifact(
        date="20260829",
        preflight=_preflight(models, exp.rank_eligible_devices(_devices())),
        gpu_receipts=receipts,
        started_ns=1,
        finished_ns=2,
    )
    mutations = {
        "missing_field:random_seed": lambda row: row.pop("random_seed"),
        "field_principles": lambda row: row.update(field_principles={}),
        "inference_substrate": lambda row: row.update(inference_substrate="cpu"),
        "verifier_is_oracle": lambda row: row.update(verifier_is_oracle=True),
        "verdict_class": lambda row: row.update(verdict_class="unknown"),
        "claim_boundary": lambda row: row.update(claim_boundary="quality"),
        "duration_s": lambda row: row.update(duration_s=-1),
        "models_used": lambda row: row["models_used"][0].update(model_sha256="bad"),
        "model_specs": lambda row: row.update(model_specs=[]),
        "device_selection_receipt": lambda row: row.update(device_selection_receipt={}),
        "arc_exclusive_load_ready": lambda row: row.update(arc_exclusive_load_ready=False),
        "rows": lambda row: row.update(rows=[]),
        "phase_rows": lambda row: row.update(phase_rows=[]),
        "lease_owner_receipts": lambda row: row.update(lease_owner_receipts=[]),
        "lease_release_receipts": lambda row: row.update(lease_release_receipts=[]),
        "vram_recovery_receipts": lambda row: row.update(vram_recovery_receipts=[]),
        "runtime_context_by_model": lambda row: row.update(runtime_context_by_model={}),
        "production_selfparse_receipt": lambda row: row.update(production_selfparse_receipt={}),
        "owned_processes_terminated": lambda row: row.update(owned_processes_terminated=False),
        "unrelated_processes_signaled": lambda row: row.update(unrelated_processes_signaled=[7]),
        "live_model_invoked": lambda row: row.update(live_model_invoked=False),
        "gate_check_summary": lambda row: row.update(gate_check_summary=[]),
        "honest_verdict": lambda row: row.update(honest_verdict="complete_wrong"),
        "selected_device_binding": lambda row: row["gpu_receipts"][0]["device"].update(
            uuid=exp.EXPECTED_GPU_UUIDS[0]
        ),
        "reproducibility_checksum": lambda row: row.update(reproducibility_checksum="bad"),
    }
    for expected_error, mutate in mutations.items():
        changed = deepcopy(artifact)
        mutate(changed)
        if expected_error != "reproducibility_checksum":
            changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected_error in exp.validate_artifact(changed)

    partial = exp.build_artifact(
        date="20260829",
        preflight=_preflight(models, exp.rank_eligible_devices(_devices())),
        gpu_receipts=[receipts[0]],
        started_ns=1,
        finished_ns=2,
    )
    assert partial["verdict_class"] == "partial"


def test_req_infra_6764_run_fail_closed_and_cli_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-INFRA-6764 stops after a failed first worker and exposes parent/worker CLIs."""
    models = _models(tmp_path)
    preflight = _preflight(models, exp.rank_eligible_devices(_devices()))
    bad = _gpu_receipt(tmp_path, 0)
    bad["errors"] = ["load failed"]
    bad["receipt_sha256"] = exp.gpu_receipt_checksum(bad)
    calls = []

    artifact = exp.run(
        result_path=tmp_path / "partial.json",
        preflight_fn=lambda: preflight,
        worker_runner=lambda *args: calls.append(args) or deepcopy(bad),
        clock=iter((1, 2)).__next__,
    )
    assert len(calls) == 1 and artifact["verdict_class"] == "partial"

    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        exp.run(
            result_path=tmp_path / "invalid.json",
            preflight_fn=lambda: {**preflight, "all_passed": False},
            clock=iter((1, 2)).__next__,
        )

    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: [])
    no_selection = deepcopy(preflight)
    no_selection["device_selection_receipt"]["selected_device"] = None
    no_selection_artifact = exp.run(
        result_path=tmp_path / "no-selection.json",
        preflight_fn=lambda: no_selection,
        clock=iter((1, 2)).__next__,
    )
    assert no_selection_artifact["gpu_receipts"] == []

    model_path = tmp_path / "worker-model.json"
    device_path = tmp_path / "worker-device.json"
    output_path = tmp_path / "worker-output.json"
    model_path.write_text(json.dumps(models[0]))
    device_path.write_text(json.dumps(_devices()[1]))
    monkeypatch.setattr(
        exp,
        "run_live_model_worker",
        lambda *args, **kwargs: {"errors": []},
    )
    assert exp._worker_entry(model_path, device_path, output_path, 45_030, tmp_path) == 0
    monkeypatch.setattr(
        exp,
        "run_live_model_worker",
        lambda *args, **kwargs: {"errors": ["failed"]},
    )
    assert exp._worker_entry(model_path, device_path, output_path, 45_030, tmp_path) == 2

    monkeypatch.setattr(exp, "_worker_entry", lambda *args: 7)
    assert (
        exp.main(
            [
                "--worker",
                "--worker-model",
                str(model_path),
                "--worker-device",
                str(device_path),
                "--worker-output",
                str(output_path),
                "--port",
                "45030",
            ]
        )
        == 7
    )
    with pytest.raises(SystemExit):
        exp.main(["--worker"])

    monkeypatch.setattr(
        exp,
        "run",
        lambda date: {
            "arc_exclusive_load_ready": False,
            "honest_verdict": "complete_blocked_arc_exclusive_load",
        },
    )
    assert exp.main(["--date", "20260829"]) == 0
    assert "complete_blocked_arc_exclusive_load" in capsys.readouterr().out
