"""Tests for the Exp 1207 llama.cpp GPU-offload verification artifact builder.

Spec traces: REQ-REPORT-015, SCENARIO-REPORT-012.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import llama_cpp_gpu_offload_fix as exp1207


def _kwargs(**overrides: object) -> dict[str, object]:
    """Build a baseline kwargs dict and overlay test-specific overrides."""

    base: dict[str, object] = {
        "llama_cpp_version": "0.3.22",
        "cuda_version_detected": "12.6",
        "cuda_support_compiled": True,
        "install_method": "already-installed",
        "llama_supports_gpu_offload": True,
        "throughput_tokens_per_sec": 302.0,
    }
    base.update(overrides)
    return base


def test_gpu_offload_verified_requires_cuda_and_throughput_req_report_015() -> None:
    """REQ-REPORT-015: verification requires both CUDA compile and >=50 tok/s."""

    assert exp1207.gpu_offload_verified(True, 50.0) is True
    assert exp1207.gpu_offload_verified(True, 49.999) is False
    assert exp1207.gpu_offload_verified(False, 1000.0) is False


def test_honest_verdict_three_outcomes_req_report_015() -> None:
    """REQ-REPORT-015: honest_verdict has exactly three outcomes."""

    assert exp1207.honest_verdict(True, 302.0) == "gpu_offload_verified"
    assert exp1207.honest_verdict(True, 10.0) == "partial_offload_cpu_fallback"
    assert exp1207.honest_verdict(False, 999.0) == "gpu_offload_failed"


def test_build_artifact_full_pass_scenario_report_012() -> None:
    """SCENARIO-REPORT-012: a healthy install yields the verified artifact."""

    artifact = exp1207.build_artifact(**_kwargs())

    assert artifact["experiment"] == "1207_llama_cpp_gpu_offload_fix_v3"
    assert artifact["schema"] == "llama_cpp_gpu_offload_fix_v3"
    assert artifact["status"] == "success"
    assert artifact["llama_cpp_version"] == "0.3.22"
    assert artifact["cuda_version_detected"] == "12.6"
    assert artifact["cuda_support_compiled"] is True
    assert artifact["install_method"] == "already-installed"
    assert artifact["llama_supports_gpu_offload"] is True
    assert artifact["throughput_tokens_per_sec"] == 302.0
    assert artifact["throughput_floor_tokens_per_sec"] == 50.0
    assert artifact["llama_cpp_gpu_offload_verified"] is True
    assert artifact["honest_verdict"] == "gpu_offload_verified"
    assert "notes" not in artifact


def test_build_artifact_partial_offload_marks_blocked_req_report_015() -> None:
    """REQ-REPORT-015: CUDA compiled but slow inference is partial fallback."""

    artifact = exp1207.build_artifact(
        **_kwargs(throughput_tokens_per_sec=12.5),
        notes="LD_LIBRARY_PATH was unset for libcudart.so.12",
    )

    assert artifact["status"] == "blocked"
    assert artifact["llama_cpp_gpu_offload_verified"] is False
    assert artifact["honest_verdict"] == "partial_offload_cpu_fallback"
    assert artifact["notes"] == "LD_LIBRARY_PATH was unset for libcudart.so.12"


def test_build_artifact_no_cuda_marks_failed_req_report_015() -> None:
    """REQ-REPORT-015: a CPU-only wheel produces gpu_offload_failed."""

    artifact = exp1207.build_artifact(
        **_kwargs(
            cuda_support_compiled=False,
            llama_supports_gpu_offload=False,
            throughput_tokens_per_sec=8.4,
        )
    )

    assert artifact["llama_cpp_gpu_offload_verified"] is False
    assert artifact["honest_verdict"] == "gpu_offload_failed"
    assert artifact["status"] == "blocked"


def test_build_artifact_rejects_unknown_install_method_req_report_015() -> None:
    """REQ-REPORT-015: install_method is restricted to the documented set."""

    with pytest.raises(ValueError):
        exp1207.build_artifact(**_kwargs(install_method="conda-channel"))


def test_write_artifact_round_trips_json_req_report_015(tmp_path: Path) -> None:
    """REQ-REPORT-015: the written JSON re-loads to the same dict."""

    artifact = exp1207.build_artifact(**_kwargs())
    out = tmp_path / "results" / "experiment_1207_llama_cpp_gpu_offload_fix_v3.json"

    exp1207.write_artifact(artifact, out)
    reloaded = json.loads(out.read_text(encoding="utf-8"))

    assert reloaded == artifact


def test_main_writes_artifact_and_returns_zero_for_verified(tmp_path: Path) -> None:
    """REQ-REPORT-015: CLI returns 0 and writes a verified artifact on success."""

    out = tmp_path / "artifact.json"
    code = exp1207.main(
        [
            "--llama-cpp-version",
            "0.3.22",
            "--cuda-version-detected",
            "12.6",
            "--cuda-support-compiled",
            "--install-method",
            "already-installed",
            "--llama-supports-gpu-offload",
            "--throughput-tokens-per-sec",
            "302.0",
            "--notes",
            "smoke test on RTX 3090",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "gpu_offload_verified"
    assert written["notes"] == "smoke test on RTX 3090"


def test_main_returns_one_for_partial_offload(tmp_path: Path) -> None:
    """REQ-REPORT-015: CLI returns 1 when CUDA is compiled but throughput is low."""

    out = tmp_path / "artifact.json"
    code = exp1207.main(
        [
            "--llama-cpp-version",
            "0.3.22",
            "--cuda-version-detected",
            "12.6",
            "--cuda-support-compiled",
            "--install-method",
            "already-installed",
            "--llama-supports-gpu-offload",
            "--throughput-tokens-per-sec",
            "12.5",
            "--out",
            str(out),
        ]
    )

    assert code == 1
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "partial_offload_cpu_fallback"


def test_main_returns_two_for_no_cuda(tmp_path: Path) -> None:
    """REQ-REPORT-015: CLI returns 2 when CUDA support is missing."""

    out = tmp_path / "artifact.json"
    code = exp1207.main(
        [
            "--llama-cpp-version",
            "0.3.22",
            "--cuda-version-detected",
            "12.6",
            "--install-method",
            "source-cmake-cuda",
            "--throughput-tokens-per-sec",
            "5.0",
            "--out",
            str(out),
        ]
    )

    assert code == 2
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "gpu_offload_failed"
    assert written["cuda_support_compiled"] is False
