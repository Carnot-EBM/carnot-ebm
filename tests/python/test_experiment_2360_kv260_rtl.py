"""Tests for Exp 2360 KV260 Ising RTL lint/simulation artifact.

Spec traces: REQ-HW-037, REQ-HW-038
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from carnot.hardware import kv260_ising_rtl_lint_sim as exp2360


def _completed(
    cmd: list[str],
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    """Build a subprocess result for mocked HDL commands."""

    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


def _passing_sim_warning_lint_runner(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
    """Pretend lint emits one warning-fatal error line and simulation passes."""

    if cmd[0] == "verilator":
        return _completed(
            cmd,
            returncode=1,
            stderr=(
                "%Warning-WIDTH: sample_ising.v:1: width mismatch\n"
                "%Error: Exiting due to 1 warning(s)\n"
            ),
        )
    if cmd[0] == "iverilog":
        return _completed(cmd, stdout="compiled\n")
    if cmd[0] == "vvp":
        return _completed(cmd, stdout="SIMULATION RESULT: PASS\n")
    raise AssertionError(cmd)


def test_discover_ising_verilog_files_excludes_other_rtl(tmp_path: Path) -> None:
    """REQ-HW-037/038: Exp 2360 scopes source discovery to Ising RTL."""

    (tmp_path / "rtl").mkdir()
    (tmp_path / "hardware" / "kv260").mkdir(parents=True)
    (tmp_path / "rtl" / "sample_ising.v").write_text("module sample_ising; endmodule\n")
    (tmp_path / "hardware" / "kv260" / "potts_sampler_v1.v").write_text(
        "module potts_sampler_v1; endmodule\n"
    )
    (tmp_path / "hardware" / "kv260" / "ising_sampler_v4.v").write_text(
        "module ising_sampler_v4; endmodule\n"
    )

    discovered = [path.as_posix() for path in exp2360.discover_ising_verilog_files(tmp_path)]

    assert discovered == [
        "hardware/kv260/ising_sampler_v4.v",
        "rtl/sample_ising.v",
    ]


def test_lint_diagnostic_counts_follow_prompt_rules() -> None:
    """REQ-HW-037/038: lint pass logic uses Error-line counts and warning counts."""

    result = {
        "stdout": "",
        "stderr": (
            "%Warning-WIDTH: sample_ising.v:1: width mismatch\n"
            "%Warning-BLKSEQ: sample_ising.v:2: blocking assignment\n"
            "%Error: Exiting due to 2 warning(s)\n"
        ),
        "error": "",
    }

    assert exp2360.count_lint_errors(result) == 1
    assert exp2360.count_lint_warnings(result) == 2


def test_run_experiment_writes_required_artifact_fields(tmp_path: Path) -> None:
    """REQ-HW-037/038: runner writes the Exp 2360 terminal schema."""

    (tmp_path / "rtl").mkdir()
    (tmp_path / "rtl" / "sample_ising.v").write_text("module sample_ising; endmodule\n")
    output = tmp_path / "results" / "experiment_2360_kv260_rtl.json"

    artifact = exp2360.run_experiment(
        project_root=tmp_path,
        output_path=output,
        runner=_passing_sim_warning_lint_runner,
        tool_paths={"verilator": "/mock/verilator", "iverilog": "/mock/iverilog"},
    )

    assert exp2360.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["rtl_lint_passed"] is False
    assert artifact["lint_errors_count"] == 1
    assert artifact["lint_warnings_count"] == 1
    assert artifact["simulation_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["hardware_execution_performed"] is False
    assert artifact["hardware_claim_allowed"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
