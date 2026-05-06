"""Tests for Exp 1437 Discrete SB KV260 RTL lint/simulation evidence.

Spec traces: REQ-ISING-024, SCENARIO-ISING-034
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.hardware import discrete_sb_rtl_lint_sim as exp1437


def _write_exp1422(root: Path) -> Path:
    """Create the prior Exp 1422 artifact that Exp 1437 must inspect."""

    path = root / "results" / "experiment_1422_discrete_sb_kv260_rtl_spec.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "rtl_spec_path": "hardware/kv260/discrete_sb_rtl_spec.md",
                "hardware_execution_performed": False,
                "hardware_claim_allowed": False,
                "honest_verdict": "rtl_spec_complete_budget_fits_no_synthesis_or_board_execution",
            }
        ),
        encoding="utf-8",
    )
    spec_path = root / "hardware" / "kv260" / "discrete_sb_rtl_spec.md"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        "Future source: `hardware/kv260/discrete_sb_256.v`\n",
        encoding="utf-8",
    )
    return path


def _completed(cmd: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""):
    """Build a subprocess result for mocked RTL tool execution."""

    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def _all_tools_missing(cmd: list[str], **_: Any):
    """Pretend none of the local RTL tools are on PATH."""

    raise FileNotFoundError(cmd[0])


def _verilator_only_runner(cmd: list[str], **_: Any):
    """Pretend Verilator is available and lint succeeds."""

    if cmd == ["verilator", "--version"]:
        return _completed(cmd, stdout="Verilator 5.000")
    if cmd[0] == "verilator" and "--lint-only" in cmd:
        return _completed(cmd, stdout="lint clean")
    raise FileNotFoundError(cmd[0])


def _iverilog_sim_runner(cmd: list[str], **_: Any):
    """Pretend iverilog syntax and testbench simulation both pass."""

    if cmd == ["iverilog", "-V"]:
        return _completed(cmd, stdout="Icarus Verilog version 12")
    if cmd[0] == "iverilog" and "-tnull" in cmd:
        return _completed(cmd, stdout="syntax ok")
    if cmd[0] == "iverilog" and "-o" in cmd:
        return _completed(cmd, stdout="compiled")
    if cmd[0] == "vvp":
        return _completed(cmd, stdout="simulation passed")
    raise FileNotFoundError(cmd[0])


def _yosys_only_runner(cmd: list[str], **_: Any):
    """Pretend only yosys is available for syntax/hierarchy checking."""

    if cmd == ["yosys", "--version"]:
        return _completed(cmd, stdout="Yosys 0.45")
    if cmd[0] == "yosys" and any("hierarchy -check" in part for part in cmd):
        return _completed(cmd, stdout="hierarchy ok")
    raise FileNotFoundError(cmd[0])


def _iverilog_sim_fail_runner(cmd: list[str], **_: Any):
    """Pretend iverilog syntax passes but testbench execution fails."""

    if cmd == ["iverilog", "-V"]:
        return _completed(cmd, stdout="Icarus Verilog version 12")
    if cmd[0] == "iverilog":
        return _completed(cmd, stdout="compiled")
    if cmd[0] == "vvp":
        return _completed(cmd, returncode=1, stderr="assertion failed")
    raise FileNotFoundError(cmd[0])


def _nonzero_iverilog_runner(cmd: list[str], **_: Any):
    """Pretend iverilog exists but syntax checking fails."""

    if cmd == ["iverilog", "-V"]:
        return _completed(cmd, stdout="Icarus Verilog version 12")
    if cmd[0] == "iverilog":
        return _completed(cmd, returncode=1, stderr="syntax error")
    raise FileNotFoundError(cmd[0])


def test_req_ising_024_writes_in_progress_marker(tmp_path: Path) -> None:
    """REQ-ISING-024: the experiment can write the required in-progress marker."""

    output = tmp_path / "results" / "experiment_1437_discrete_sb_kv260_rtl_lint_sim.json"

    marker = exp1437.write_in_progress_artifact(output)

    assert marker["status"] == "in_progress"
    assert json.loads(output.read_text(encoding="utf-8")) == marker


def test_req_ising_024_blocks_when_exp1422_planned_source_is_missing(tmp_path: Path) -> None:
    """REQ-ISING-024: missing Discrete SB RTL source is recorded as a blocker."""

    exp1422_path = _write_exp1422(tmp_path)
    output = tmp_path / "results" / "experiment_1437_discrete_sb_kv260_rtl_lint_sim.json"

    artifact = exp1437.run_experiment(
        project_root=tmp_path,
        exp1422_path=exp1422_path,
        output_path=output,
        runner=_all_tools_missing,
    )

    assert exp1437.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["rtl_lint_complete"] is False
    assert artifact["simulation_complete"] is False
    assert artifact["synthesis_attempted"] is False
    assert artifact["yosys_available"] is False
    assert artifact["verilator_available"] is False
    assert artifact["vivado_available"] is False
    assert artifact["hardware_execution_performed"] is False
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["honest_verdict"] == "blocked_missing_discrete_sb_rtl_source"
    assert artifact["rtl_sources_checked"] == [
        {
            "path": "hardware/kv260/discrete_sb_256.v",
            "exists": False,
            "source": "exp1422_planned_discrete_sb_source",
        }
    ]
    assert "discrete_sb_256.v" in artifact["next_bitfile_step"]
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_ising_034_runs_verilator_lint_when_source_exists(tmp_path: Path) -> None:
    """SCENARIO-ISING-034: available lint tool runs against discovered RTL source."""

    exp1422_path = _write_exp1422(tmp_path)
    source = tmp_path / "hardware" / "kv260" / "discrete_sb_256.v"
    source.write_text("module discrete_sb_256; endmodule\n", encoding="utf-8")

    artifact = exp1437.run_experiment(
        project_root=tmp_path,
        exp1422_path=exp1422_path,
        output_path=tmp_path / "results" / "artifact.json",
        runner=_verilator_only_runner,
    )

    assert artifact["status"] == "complete"
    assert artifact["rtl_lint_complete"] is True
    assert artifact["simulation_complete"] is False
    assert artifact["verilator_available"] is True
    assert artifact["command_results"]["rtl_lint"]["command"] == [
        "verilator",
        "--lint-only",
        "--timing",
        "hardware/kv260/discrete_sb_256.v",
    ]
    assert artifact["honest_verdict"] == "rtl_lint_complete_simulation_not_run_no_testbench"
    assert artifact["hardware_claim_allowed"] is False


def test_scenario_ising_034_runs_iverilog_syntax_and_testbench_sim(tmp_path: Path) -> None:
    """SCENARIO-ISING-034: iverilog syntax plus testbench execution can complete."""

    exp1422_path = _write_exp1422(tmp_path)
    source = tmp_path / "hardware" / "kv260" / "discrete_sb_256.v"
    testbench = tmp_path / "hardware" / "kv260" / "discrete_sb_256_tb.v"
    source.write_text("module discrete_sb_256; endmodule\n", encoding="utf-8")
    testbench.write_text("module discrete_sb_256_tb; endmodule\n", encoding="utf-8")

    artifact = exp1437.run_experiment(
        project_root=tmp_path,
        exp1422_path=exp1422_path,
        output_path=tmp_path / "results" / "artifact.json",
        runner=_iverilog_sim_runner,
    )

    assert artifact["status"] == "complete"
    assert artifact["rtl_lint_complete"] is True
    assert artifact["simulation_complete"] is True
    assert artifact["iverilog_available"] is True
    assert artifact["command_results"]["simulation"]["command"][0] == "vvp"
    assert artifact["honest_verdict"] == "rtl_lint_and_simulation_complete_no_hardware_execution"
    assert "Vivado synthesis" in artifact["next_bitfile_step"]


def test_req_ising_024_uses_yosys_read_verilog_when_it_is_the_available_tool(
    tmp_path: Path,
) -> None:
    """REQ-ISING-024: yosys syntax/hierarchy check is a valid bounded lint path."""

    exp1422_path = _write_exp1422(tmp_path)
    source = tmp_path / "hardware" / "kv260" / "discrete_sb_256.v"
    source.write_text("module discrete_sb_256; endmodule\n", encoding="utf-8")

    artifact = exp1437.run_experiment(
        project_root=tmp_path,
        exp1422_path=exp1422_path,
        output_path=tmp_path / "results" / "artifact.json",
        runner=_yosys_only_runner,
    )

    assert artifact["rtl_lint_complete"] is True
    assert artifact["yosys_available"] is True
    assert artifact["command_results"]["rtl_lint"]["command"] == [
        "yosys",
        "-q",
        "-p",
        "read_verilog -sv hardware/kv260/discrete_sb_256.v; hierarchy -check",
    ]


def test_req_ising_024_blocks_when_testbench_simulation_fails(tmp_path: Path) -> None:
    """REQ-ISING-024: a failing testbench prevents complete simulation evidence."""

    exp1422_path = _write_exp1422(tmp_path)
    source = tmp_path / "hardware" / "kv260" / "discrete_sb_256.v"
    testbench = tmp_path / "hardware" / "kv260" / "discrete_sb_256_tb.v"
    source.write_text("module discrete_sb_256; endmodule\n", encoding="utf-8")
    testbench.write_text("module discrete_sb_256_tb; endmodule\n", encoding="utf-8")

    artifact = exp1437.run_experiment(
        project_root=tmp_path,
        exp1422_path=exp1422_path,
        output_path=tmp_path / "results" / "artifact.json",
        runner=_iverilog_sim_fail_runner,
    )

    assert artifact["status"] == "blocked"
    assert artifact["rtl_lint_complete"] is True
    assert artifact["simulation_complete"] is False
    assert artifact["honest_verdict"] == "blocked_simulation_failed"
    assert "assertion failed" in artifact["command_results"]["simulation"]["stderr_summary"]


def test_req_ising_024_blocks_when_sources_exist_but_tools_are_missing(tmp_path: Path) -> None:
    """REQ-ISING-024: source without local RTL tools records a no-tool blocker."""

    exp1422_path = _write_exp1422(tmp_path)
    source = tmp_path / "hardware" / "kv260" / "discrete_sb_256.v"
    source.write_text("module discrete_sb_256; endmodule\n", encoding="utf-8")

    artifact = exp1437.run_experiment(
        project_root=tmp_path,
        exp1422_path=exp1422_path,
        output_path=tmp_path / "results" / "artifact.json",
        runner=_all_tools_missing,
    )

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_no_rtl_lint_or_sim_tool"
    assert "yosys/verilator/iverilog" in artifact["next_bitfile_step"]


def test_req_ising_024_blocks_on_failed_lint_and_rejects_dishonest_claims(
    tmp_path: Path,
) -> None:
    """REQ-ISING-024: lint failure and hardware-claim validation stay explicit."""

    exp1422_path = _write_exp1422(tmp_path)
    source = tmp_path / "hardware" / "kv260" / "discrete_sb_256.v"
    source.write_text("module discrete_sb_256; endmodule\n", encoding="utf-8")

    artifact = exp1437.run_experiment(
        project_root=tmp_path,
        exp1422_path=exp1422_path,
        output_path=tmp_path / "results" / "artifact.json",
        runner=_nonzero_iverilog_runner,
    )

    assert artifact["status"] == "blocked"
    assert artifact["rtl_lint_complete"] is False
    assert artifact["honest_verdict"] == "blocked_rtl_lint_failed"
    assert "syntax error" in artifact["command_results"]["rtl_lint"]["stderr_summary"]

    missing = dict(artifact)
    missing.pop("rtl_sources_checked")
    with pytest.raises(ValueError, match="missing"):
        exp1437.validate_artifact(missing)

    dishonest_execution = dict(artifact)
    dishonest_execution["hardware_execution_performed"] = True
    with pytest.raises(ValueError, match="hardware_execution_performed"):
        exp1437.validate_artifact(dishonest_execution)

    dishonest_claim = dict(artifact)
    dishonest_claim["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="hardware_claim_allowed"):
        exp1437.validate_artifact(dishonest_claim)

    bad_status = dict(artifact)
    bad_status["status"] = "in_progress"
    with pytest.raises(ValueError, match="status"):
        exp1437.validate_artifact(bad_status)


def test_req_ising_024_command_summaries_cover_timeout_oserror_and_truncation(
    tmp_path: Path,
) -> None:
    """REQ-ISING-024: command summaries preserve bounded blocker evidence."""

    timeout_result = exp1437.run_command(
        ["yosys", "--version"],
        cwd=tmp_path,
        runner=lambda cmd, **_: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd=cmd, timeout=1, output="o" * 5000, stderr="late")
        ),
        timeout=1,
    )
    oserror_result = exp1437.run_command(
        ["vivado", "-version"],
        cwd=tmp_path,
        runner=lambda cmd, **_: (_ for _ in ()).throw(OSError("permission denied")),
        timeout=1,
    )

    assert timeout_result["timed_out"] is True
    assert timeout_result["stdout_summary"].endswith("...[truncated]")
    assert oserror_result["error"] == "permission denied"
