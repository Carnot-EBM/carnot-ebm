"""Tests for Exp 1451 Discrete SB RTL lint/simulation rerun evidence.

Spec traces: REQ-ISING-026, SCENARIO-ISING-036
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.hardware import discrete_sb_rtl_lint_sim_rerun as exp1451


def _completed(cmd: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""):
    """Build a subprocess result for mocked HDL commands."""

    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def _write_exp1441(root: Path, *, source_created: bool = True) -> Path:
    """Create the Exp 1441 source artifact that gates the rerun."""

    artifact_path = root / "results" / "experiment_1441_discrete_sb_rtl_source_implementation.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "rtl_source_created": source_created,
                "rtl_source_path": "hardware/kv260/discrete_sb_256.v",
                "testbench_created": True,
                "testbench_path": "hardware/kv260/discrete_sb_256_tb.v",
                "honest_verdict": "rtl_source_and_testbench_created_lint_and_smoke_sim_passed_no_kv260_execution_claim",
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def _write_rtl_pair(root: Path) -> None:
    """Create placeholder RTL and testbench files for command-selection tests."""

    kv260 = root / "hardware" / "kv260"
    kv260.mkdir(parents=True, exist_ok=True)
    (kv260 / "discrete_sb_256.v").write_text(
        "module discrete_sb_256; endmodule\n",
        encoding="utf-8",
    )
    (kv260 / "discrete_sb_256_tb.v").write_text(
        "module discrete_sb_256_tb; endmodule\n",
        encoding="utf-8",
    )


def _passing_runner(cmd: list[str], **_: Any):
    """Pretend Verilator, Icarus, Yosys, and Vivado are visible and checks pass."""

    if cmd == ["verilator", "--version"]:
        return _completed(cmd, stdout="Verilator 5.000")
    if cmd == ["iverilog", "-V"]:
        return _completed(cmd, stdout="Icarus Verilog version 12")
    if cmd == ["yosys", "--version"]:
        return _completed(cmd, stdout="Yosys 0.45")
    if cmd == ["vivado", "-version"]:
        return _completed(cmd, stdout="Vivado v2024.2")
    if cmd[0] == "verilator":
        return _completed(cmd, stdout="lint clean")
    if cmd[0] == "iverilog":
        return _completed(cmd, stdout="compiled")
    if cmd[0] == "vvp":
        return _completed(cmd, stdout="SIMULATION RESULT: PASS")
    raise AssertionError(cmd)


def _missing_tools_runner(cmd: list[str], **_: Any):
    """Pretend no local HDL tools are available."""

    raise FileNotFoundError(cmd[0])


def _yosys_only_runner(cmd: list[str], **_: Any):
    """Pretend Yosys is the only available lint path."""

    if cmd == ["yosys", "--version"]:
        return _completed(cmd, stdout="Yosys 0.45")
    if cmd[0] == "yosys":
        return _completed(cmd, stdout="hierarchy ok")
    raise FileNotFoundError(cmd[0])


def _iverilog_only_runner(cmd: list[str], **_: Any):
    """Pretend Icarus is the only available lint and simulation path."""

    if cmd == ["iverilog", "-V"]:
        return _completed(cmd, stdout="Icarus Verilog version 12")
    if cmd[0] == "iverilog":
        return _completed(cmd, stdout="compiled")
    if cmd[0] == "vvp":
        return _completed(cmd, stdout="SIMULATION RESULT: PASS")
    raise FileNotFoundError(cmd[0])


def _failing_runner(cmd: list[str], **_: Any):
    """Pretend lint and simulation both execute but fail distinctly."""

    if cmd == ["verilator", "--version"]:
        return _completed(cmd, stdout="Verilator 5.000")
    if cmd == ["iverilog", "-V"]:
        return _completed(cmd, stdout="Icarus Verilog version 12")
    if cmd in (["yosys", "--version"], ["vivado", "-version"]):
        raise FileNotFoundError(cmd[0])
    if cmd[0] == "verilator":
        return _completed(cmd, returncode=1, stderr="syntax error near row_idx")
    if cmd[0] == "iverilog":
        return _completed(cmd, stdout="compiled")
    if cmd[0] == "vvp":
        return _completed(cmd, returncode=1, stderr="CHECK FAIL: spin mismatch")
    raise AssertionError(cmd)


def test_req_ising_026_spec_anchor_exists() -> None:
    """REQ-ISING-026, SCENARIO-ISING-036: rerun work is spec-anchored."""

    spec = (exp1451.PROJECT_ROOT / "openspec/capabilities/ising-backend/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-ISING-026" in spec
    assert "SCENARIO-ISING-036" in spec
    assert "experiment_1451_discrete_sb_rtl_lint_sim_rerun.json" in spec


def test_req_ising_026_writes_in_progress_marker(tmp_path: Path) -> None:
    """REQ-ISING-026: the rerun writes a bootstrap marker before inspection."""

    output = tmp_path / "results" / "experiment_1451_discrete_sb_rtl_lint_sim_rerun.json"

    marker = exp1451.write_in_progress_artifact(output)

    assert marker == {"status": "in_progress"}
    assert json.loads(output.read_text(encoding="utf-8")) == marker


def test_scenario_ising_036_runs_lint_and_simulation_when_source_exists(tmp_path: Path) -> None:
    """SCENARIO-ISING-036: source-present rerun records clean lint and sim evidence."""

    _write_rtl_pair(tmp_path)
    exp1441 = _write_exp1441(tmp_path)
    output = tmp_path / "results" / "artifact.json"

    artifact = exp1451.run_experiment(
        project_root=tmp_path,
        exp1441_path=exp1441,
        output_path=output,
        runner=_passing_runner,
    )

    assert exp1451.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["rtl_source_present"] is True
    assert artifact["exp1441_rtl_source_created"] is True
    assert artifact["rtl_lint_complete"] is True
    assert artifact["simulation_complete"] is True
    assert artifact["tools_available"]["verilator"] is True
    assert artifact["tools_available"]["iverilog"] is True
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["hardware_execution_performed"] is False
    assert artifact["lint_errors"] == []
    assert artifact["simulation_errors"] == []
    assert "verilator --lint-only" in artifact["lint_command"]
    assert "iverilog -g2012" in artifact["simulation_command"]
    assert "vvp /tmp/discrete_sb_256_tb_1451.vvp" in artifact["simulation_command"]
    assert artifact["honest_verdict"] == "rtl_lint_and_simulation_complete_no_hardware_execution_no_kv260_claim"
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_req_ising_026_does_not_run_lint_without_confirmed_source(tmp_path: Path) -> None:
    """REQ-ISING-026: missing source or false Exp 1441 source flag blocks rerun."""

    exp1441 = _write_exp1441(tmp_path, source_created=False)

    artifact = exp1451.run_experiment(
        project_root=tmp_path,
        exp1441_path=exp1441,
        output_path=tmp_path / "results" / "artifact.json",
        runner=_passing_runner,
    )

    assert artifact["status"] == "complete"
    assert artifact["rtl_source_present"] is False
    assert artifact["exp1441_rtl_source_created"] is False
    assert artifact["rtl_lint_complete"] is False
    assert artifact["simulation_complete"] is False
    assert artifact["lint_command"] == ""
    assert artifact["simulation_command"] == ""
    assert artifact["lint_errors"][0]["error_class"] == "prerequisite_missing"
    assert artifact["simulation_errors"][0]["error_class"] == "prerequisite_missing"
    assert artifact["honest_verdict"] == "blocked_missing_or_unconfirmed_discrete_sb_rtl_source_no_hardware_claim"


def test_req_ising_026_records_missing_testbench(tmp_path: Path) -> None:
    """REQ-ISING-026: source without a testbench records why simulation did not run."""

    kv260 = tmp_path / "hardware" / "kv260"
    kv260.mkdir(parents=True, exist_ok=True)
    (kv260 / "discrete_sb_256.v").write_text("module discrete_sb_256; endmodule\n", encoding="utf-8")
    exp1441 = _write_exp1441(tmp_path)

    artifact = exp1451.run_experiment(
        project_root=tmp_path,
        exp1441_path=exp1441,
        output_path=tmp_path / "results" / "artifact.json",
        runner=_passing_runner,
    )

    assert artifact["rtl_lint_complete"] is True
    assert artifact["simulation_complete"] is False
    assert artifact["simulation_errors"][0]["error_class"] == "testbench_missing"


def test_req_ising_026_records_missing_tools_and_yosys_fallback(tmp_path: Path) -> None:
    """REQ-ISING-026: tool availability and lint fallback remain explicit."""

    _write_rtl_pair(tmp_path)
    exp1441 = _write_exp1441(tmp_path)

    missing = exp1451.run_experiment(
        project_root=tmp_path,
        exp1441_path=exp1441,
        output_path=tmp_path / "results" / "missing.json",
        runner=_missing_tools_runner,
    )
    assert missing["tools_available"] == {
        "verilator": False,
        "iverilog": False,
        "yosys": False,
        "vivado": False,
    }
    assert missing["lint_errors"][0]["error_class"] == "no_lint_tool"
    assert missing["simulation_errors"][0]["error_class"] == "no_simulator"
    assert missing["honest_verdict"] == "blocked_no_local_lint_or_simulation_tool_no_hardware_claim"

    yosys = exp1451.run_experiment(
        project_root=tmp_path,
        exp1441_path=exp1441,
        output_path=tmp_path / "results" / "yosys.json",
        runner=_yosys_only_runner,
    )
    assert yosys["rtl_lint_complete"] is True
    assert yosys["simulation_complete"] is False
    assert yosys["lint_command"] == (
        "yosys -q -p read_verilog -sv hardware/kv260/discrete_sb_256.v; hierarchy -check"
    )
    assert yosys["simulation_errors"][0]["error_class"] == "no_simulator"

    iverilog = exp1451.run_experiment(
        project_root=tmp_path,
        exp1441_path=exp1441,
        output_path=tmp_path / "results" / "iverilog.json",
        runner=_iverilog_only_runner,
    )
    assert iverilog["rtl_lint_complete"] is True
    assert iverilog["simulation_complete"] is True
    assert iverilog["lint_command"] == "iverilog -tnull -g2012 hardware/kv260/discrete_sb_256.v"


def test_req_ising_026_records_lint_and_simulation_failures(tmp_path: Path) -> None:
    """REQ-ISING-026: nonzero lint and testbench exits are captured precisely."""

    _write_rtl_pair(tmp_path)
    exp1441 = _write_exp1441(tmp_path)

    artifact = exp1451.run_experiment(
        project_root=tmp_path,
        exp1441_path=exp1441,
        output_path=tmp_path / "results" / "artifact.json",
        runner=_failing_runner,
    )

    assert artifact["status"] == "complete"
    assert artifact["rtl_lint_complete"] is False
    assert artifact["simulation_complete"] is False
    assert artifact["lint_errors"][0]["error_class"] == "nonzero_exit"
    assert "syntax error" in artifact["lint_errors"][0]["stderr_summary"]
    assert artifact["simulation_errors"][0]["stage"] == "simulation"
    assert artifact["simulation_errors"][0]["error_class"] == "nonzero_exit"
    assert "spin mismatch" in artifact["simulation_errors"][0]["stderr_summary"]
    assert artifact["honest_verdict"] == "blocked_rtl_lint_and_simulation_failed_no_hardware_claim"


def test_req_ising_026_command_summaries_and_claim_validation(tmp_path: Path) -> None:
    """REQ-ISING-026: command errors are summarized and hardware claims are gated."""

    timeout_result = exp1451.run_command(
        ["verilator", "--version"],
        cwd=tmp_path,
        runner=lambda cmd, **_: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd=cmd, timeout=1, output="o" * 5000, stderr="late")
        ),
        timeout=1,
    )
    os_error_result = exp1451.run_command(
        ["vivado", "-version"],
        cwd=tmp_path,
        runner=lambda cmd, **_: (_ for _ in ()).throw(OSError("permission denied")),
        timeout=1,
    )
    missing_result = exp1451.run_command(
        ["yosys", "--version"],
        cwd=tmp_path,
        runner=lambda cmd, **_: (_ for _ in ()).throw(FileNotFoundError(cmd[0])),
        timeout=1,
    )

    assert timeout_result["error_class"] == "timeout"
    assert timeout_result["stdout_summary"].endswith("...[truncated]")
    assert os_error_result["error_class"] == "os_error"
    assert missing_result["error_class"] == "not_found"

    valid = {
        field: None
        for field in exp1451.REQUIRED_ARTIFACT_FIELDS
    }
    valid.update(
        {
            "status": "complete",
            "hardware_claim_allowed": False,
            "hardware_execution_performed": False,
        }
    )
    exp1451.validate_artifact(valid)

    missing = dict(valid)
    missing.pop("commands_run")
    with pytest.raises(ValueError, match="missing"):
        exp1451.validate_artifact(missing)

    bad_status = dict(valid)
    bad_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        exp1451.validate_artifact(bad_status)

    dishonest = dict(valid)
    dishonest["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="hardware_claim_allowed"):
        exp1451.validate_artifact(dishonest)

    dishonest_execution = dict(valid)
    dishonest_execution["hardware_execution_performed"] = True
    with pytest.raises(ValueError, match="hardware_execution_performed"):
        exp1451.validate_artifact(dishonest_execution)
