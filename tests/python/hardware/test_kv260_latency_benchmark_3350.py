"""Tests for Exp 3350 KV260 Latency Benchmark."""

import json
from pathlib import Path
from unittest.mock import patch

from carnot.hardware.kv260_latency_benchmark_3350 import (
    build_problem_payload,
    run_cpu_baseline,
    run_experiment,
)


def test_build_problem_payload() -> None:
    payload = build_problem_payload()
    assert len(payload["problems"]) == 100
    problem = payload["problems"][0]
    assert problem["n_spins"] == 64
    assert len(problem["upload"]["adjacency"]) == 64
    assert len(problem["upload"]["adjacency"][0]) == 16


def test_run_cpu_baseline() -> None:
    payload = build_problem_payload()
    median_latency = run_cpu_baseline(payload["problems"][:5])
    assert median_latency > 0


@patch("carnot.hardware.kv260_latency_benchmark_3350._ssh")
@patch("carnot.hardware.kv260_latency_benchmark_3350._scp")
def test_run_experiment_ssh_blocked(mock_scp, mock_ssh, tmp_path) -> None:
    # SSH returns non-zero code to block immediately
    mock_ssh.returncode = 1

    class DummyResult:
        returncode = 1
        stdout = ""
        stderr = "connection refused"

    mock_ssh.return_value = DummyResult()

    with patch(
        "carnot.hardware.kv260_latency_benchmark_3350.RESULT_PATH", tmp_path / "result.json"
    ):
        result = run_experiment()
    assert result["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert "SSH test failed" in result["blocked_reasons"]


@patch("carnot.hardware.kv260_latency_benchmark_3350._ssh")
@patch("carnot.hardware.kv260_latency_benchmark_3350._scp")
def test_run_experiment_success(mock_scp, mock_ssh, tmp_path) -> None:
    class DummySSHResult:
        def __init__(self, code=0, stdout="", stderr=""):
            self.returncode = code
            self.stdout = stdout
            self.stderr = stderr

    def ssh_side_effect(cmd, timeout=30):
        if cmd == "true":
            return DummySSHResult()
        elif cmd == "uptime":
            return DummySSHResult(0, "up 1 day")
        elif "xmutil" in cmd:
            return DummySSHResult()
        elif "sudo python3" in cmd:
            board_out = json.dumps({"latencies_us": [10.0, 11.0], "median_latency_us": 10.5})
            return DummySSHResult(0, f"ignoring first part\n{board_out}\n")
        return DummySSHResult()

    mock_ssh.side_effect = ssh_side_effect

    class DummySCPResult:
        returncode = 0
        stdout = ""
        stderr = ""

    mock_scp.return_value = DummySCPResult()

    with patch(
        "carnot.hardware.kv260_latency_benchmark_3350.RESULT_PATH", tmp_path / "result.json"
    ):
        result = run_experiment()
        assert result["honest_verdict"] == "success: hardware latency benchmark complete"
        assert result["hardware_latency_us"] == 10.5
        assert result["speedup_vs_cpu"] > 0
        assert result["cpu_latency_us"] > 0
