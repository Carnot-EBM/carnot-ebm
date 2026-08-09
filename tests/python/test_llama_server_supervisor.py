"""Tests for the reusable native llama-server supervisor.

Spec refs: REQ-INFRA-6228,
SCENARIO-INFRA-6228-DEAD-PORT-SLOW-LOAD-AND-EARLY-EXIT-ARE-BOUNDED,
SCENARIO-INFRA-6228-OWNERSHIP-REFUSES-PID-REUSE-AND-UNRELATED-OWNERS,
SCENARIO-INFRA-6228-CUDA-READINESS-USES-LOGS-AND-GPU-INTERVALS.
"""

from __future__ import annotations

from pathlib import Path
import signal
from typing import Any

from carnot.inference import llama_server_supervisor as sup


class FakeProcessOps:
    """SCENARIO-INFRA-6228-OWNERSHIP-REFUSES-PID-REUSE-AND-UNRELATED-OWNERS."""

    def __init__(self, waits: list[str] | None = None) -> None:
        self.waits = list(waits or ["exited"])
        self.signals: list[dict[str, Any]] = []

    def send_signal(self, pid: int, sig: signal.Signals, *, process_group: bool) -> None:
        self.signals.append(
            {"pid": pid, "signal": sig.name, "process_group": process_group}
        )

    def wait_for_exit(self, pid: int, timeout_s: float) -> str:
        assert pid == 100
        assert timeout_s >= 0
        return self.waits.pop(0) if self.waits else "timeout"


def _contract() -> sup.JsonDict:
    return sup.supervisor_contract(
        health_timeout_s=3.0,
        token_timeout_s=2.0,
        cleanup_grace_s=0.5,
        kill_after_cleanup_timeout_s=0.25,
        retry_budget=1,
        endurance_interval_s=2.0,
        endurance_sample_count=3,
    )


def _identity(*, command_hash: str | None = None, owned: bool = True) -> sup.JsonDict:
    command = ["llama-server", "--model", "/models/family-Q4_K_M.gguf"]
    return {
        "pid": 100,
        "exists": True,
        "start_time_ticks": 555,
        "uid": 1000,
        "command": command,
        "command_hash": command_hash or sup.command_hash(command),
        "process_group_id": 100,
        "parent_identity": {"pid": 9, "start_time_ticks": 123},
        "owned_by_task": owned,
    }


def test_req_infra_6228_contract_classifies_bounded_failures() -> None:
    """REQ-INFRA-6228: retry, wait, and cleanup limits are explicit."""

    contract = _contract()
    stat = "100 (llama server) S 9 100 100 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 98765 0 0"
    parsed = sup.parse_proc_stat(stat)

    assert contract["retry_budget"] == 1
    assert contract["health_timeout_s"] == 3.0
    assert contract["pid_reuse_guard"] == (
        "pid_start_time_uid_command_hash_process_group_parent_identity"
    )
    assert sup.classify_runtime_event(exception_name="ConnectionRefusedError") == (
        "connection_refused"
    )
    assert sup.classify_runtime_event(text="server_exited_before_health") == "early_exit"
    assert sup.classify_runtime_event(text="RemoteDisconnected") == "server_died_mid_request"
    assert sup.classify_runtime_event(text="request timed out") == "request_timeout"
    assert sup.classify_runtime_event(deadline_expired=True) == "deadline_expired"
    assert sup.classify_runtime_event(text="retry budget exhausted") == "retry_exhausted"
    assert sup.classify_runtime_event(exit_code=-15) == "external_signal_sigterm"
    assert sup.classify_runtime_event("", 0) == "no_failure"
    assert sup.classify_runtime_event("odd stderr", 2) == "unclassified_runtime_failure"
    assert sup.should_retry("connection_refused", 0, contract) is True
    assert sup.should_retry("connection_refused", 1, contract) is False
    assert sup.should_retry("early_exit", 0, contract) is True
    assert sup.should_retry("unclassified_runtime_failure", 0, contract) is False
    assert parsed["pid"] == 100
    assert parsed["comm"] == "llama server"
    assert parsed["process_group_id"] == 100
    assert parsed["start_time_ticks"] == 98765


def test_scenario_infra_6228_cleanup_refuses_identity_drift_and_leaks() -> None:
    """SCENARIO-INFRA-6228-OWNERSHIP-REFUSES-PID-REUSE-AND-UNRELATED-OWNERS."""

    contract = _contract()
    recorded = _identity()
    current = _identity()
    ops = FakeProcessOps(["exited"])

    receipt = sup.cleanup_recorded_identity(
        recorded,
        lambda _pid: current,
        ops,
        contract=contract,
    )

    assert receipt["action"] == "terminated"
    assert receipt["leak_free"] is True
    assert receipt["unrelated_process_kill_count_delta"] == 0
    assert ops.signals == [{"pid": 100, "signal": "SIGTERM", "process_group": True}]

    reused_ops = FakeProcessOps()
    reused = {**current, "start_time_ticks": 556}
    refused = sup.cleanup_recorded_identity(
        recorded,
        lambda _pid: reused,
        reused_ops,
        contract=contract,
    )
    assert refused["action"] == "refused"
    assert refused["reason"] == "identity_mismatch"
    assert reused_ops.signals == []

    unowned_ops = FakeProcessOps()
    unowned = sup.cleanup_recorded_identity(
        {**recorded, "owned_by_task": False},
        lambda _pid: current,
        unowned_ops,
        contract=contract,
    )
    assert unowned["action"] == "refused"
    assert unowned["reason"] == "unowned_process"
    assert unowned_ops.signals == []

    drift_ops = FakeProcessOps()
    command_drift = {**current, "command_hash": sup.command_hash(["other"])}
    drift = sup.cleanup_recorded_identity(
        recorded,
        lambda _pid: command_drift,
        drift_ops,
        contract=contract,
    )
    assert drift["action"] == "refused"
    assert drift["reason"] == "identity_mismatch"
    assert drift_ops.signals == []

    already_gone = sup.cleanup_recorded_identity(
        recorded,
        lambda _pid: {"pid": 100, "exists": False},
        FakeProcessOps(),
        contract=contract,
    )
    assert already_gone["action"] == "already_exited"
    assert already_gone["leak_free"] is True

    killed_ops = FakeProcessOps(["timeout", "exited"])
    force_killed = sup.cleanup_recorded_identity(
        recorded,
        lambda _pid: current,
        killed_ops,
        contract=contract,
    )
    assert force_killed["action"] == "force_killed"
    assert force_killed["leak_free"] is True
    assert [row["signal"] for row in killed_ops.signals] == ["SIGTERM", "SIGKILL"]

    leak_ops = FakeProcessOps(["timeout", "timeout"])
    leak = sup.cleanup_recorded_identity(
        recorded,
        lambda _pid: current,
        leak_ops,
        contract=contract,
    )
    assert leak["action"] == "cleanup_leak"
    assert leak["leak_free"] is False
    assert leak["bounded"] is True


def test_scenario_infra_6228_cuda_needs_logs_and_gpu_intervals() -> None:
    """SCENARIO-INFRA-6228-CUDA-READINESS-USES-LOGS-AND-GPU-INTERVALS."""

    intervals = [
        {
            "label": "during",
            "compute_apps": [
                {
                    "pid": 100,
                    "process_name": "llama-server",
                    "used_memory_mb": 9400,
                    "owned_by_task": True,
                }
            ],
        }
    ]
    log = "llm_load_tensors: offloaded 64/64 layers to GPU\nCUDA0 buffer size = 9400 MiB"

    evidence = sup.parse_cuda_placement(
        "qwen3_35b_a3b_moe",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        log,
        intervals,
        owned_pids={100},
    )

    assert evidence["cuda_placement_confirmed"] is True
    assert evidence["cuda_layers_offloaded"] == 64
    assert evidence["log_cuda_evidence_present"] is True
    assert evidence["gpu_interval_owned_vram_confirmed"] is True

    flags_only = sup.parse_cuda_placement(
        "qwen3_35b_a3b_moe",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "--n-gpu-layers all",
        intervals,
        owned_pids={100},
    )
    assert flags_only["cuda_placement_confirmed"] is False
    assert flags_only["log_cuda_evidence_present"] is False

    log_only = sup.parse_cuda_placement(
        "qwen3_35b_a3b_moe",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        log,
        [{"label": "during", "compute_apps": []}],
        owned_pids={100},
    )
    assert log_only["cuda_placement_confirmed"] is False
    assert log_only["gpu_interval_owned_vram_confirmed"] is False

    tensor_marker = sup.parse_cuda_placement(
        "gemma4_31b_dense",
        "unsloth/gemma-4-31B-it-GGUF",
        "ggml_cuda_init: CUDA found\nCUDA1 compute buffer size = 1024 MiB",
        intervals,
        owned_pids={100},
    )
    assert tensor_marker["cuda_tensor_or_buffer_confirmed"] is True
    assert tensor_marker["cuda_placement_confirmed"] is True


def test_scenario_infra_6228_repeated_tokens_and_family_readiness(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6228-ENDURANCE-AND-RECOVERY-QUALIFY-EACH-FAMILY."""

    token_path = tmp_path / "token.bin"
    token_path.write_bytes(b"A")
    assert sup.sha256_file(token_path) == sup.sha256_bytes(b"A")
    samples = [
        sup.raw_token_receipt(b"A", latency_s=0.1 + index, path=token_path, sample_index=index)
        for index in range(3)
    ]
    token_summary = sup.summarize_repeated_tokens(samples, min_samples=3)
    cuda = sup.parse_cuda_placement(
        "qwen3_35b_a3b_moe",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "offloaded 64/64 layers to GPU\nCUDA0 buffer size = 10 MiB",
        [{"compute_apps": [{"pid": 100, "used_memory_mb": 1000, "owned_by_task": True}]}],
        owned_pids={100},
    )
    family = {
        "ownership": {"owned_process": True},
        "cuda": cuda,
        "tokens": token_summary,
        "endurance": {"passed": True, "health_sample_count": 3},
        "recovery": {"controlled_failure_bounded": True, "recovery_success": True},
        "leak_check": {"leak_free": True},
    }

    assert token_summary["deterministic_repeated_output"] is True
    assert sup.family_runtime_ready(family) is True

    not_recovered = {**family, "recovery": {"controlled_failure_bounded": True}}
    assert sup.family_runtime_ready(not_recovered) is False
