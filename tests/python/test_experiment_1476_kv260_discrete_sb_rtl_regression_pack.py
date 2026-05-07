"""Tests for Exp 1476 KV260 Discrete SB RTL regression packaging.

Spec traces: REQ-ISING-027, SCENARIO-ISING-037
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.hardware import discrete_sb_regression_pack as exp1476


def _completed(cmd: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""):
    """Build a subprocess result for mocked HDL commands."""

    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def _write_prior_artifacts(
    root: Path,
    *,
    exp1451_hardware_claim: bool = False,
    exp1460_source_only: bool = True,
) -> tuple[Path, Path]:
    """Create prior artifacts that anchor the no-board claim boundary."""

    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    exp1451 = results / "experiment_1451_discrete_sb_rtl_lint_sim_rerun.json"
    exp1451.write_text(
        json.dumps(
            {
                "status": "complete",
                "rtl_lint_complete": True,
                "simulation_complete": True,
                "hardware_execution_performed": exp1451_hardware_claim,
                "hardware_claim_allowed": exp1451_hardware_claim,
                "honest_verdict": "rtl_lint_and_simulation_complete_no_hardware_execution_no_kv260_claim",
            }
        ),
        encoding="utf-8",
    )

    if exp1460_source_only:
        active_tracks = [
            {
                "track_id": "kv260_discrete_sb_rtl_sim",
                "claim_boundary": (
                    "No KV260 board execution, bitfile, or latency claim until "
                    "Vivado synthesis, bitfile flashing, and board commands are captured."
                ),
            }
        ]
    else:
        active_tracks = [{"track_id": "kv260_discrete_sb_rtl_sim", "claim_boundary": "board ok"}]

    exp1460 = results / "experiment_1460_hardware_portfolio_narrowing.json"
    exp1460.write_text(
        json.dumps(
            {
                "status": "complete",
                "active_hardware_tracks": active_tracks,
                "honest_verdict": "active_tracks_narrowed_to_3",
            }
        ),
        encoding="utf-8",
    )
    return exp1451, exp1460


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
    """Pretend local HDL tools are available and all source checks pass."""

    if cmd == ["verilator", "--version"]:
        return _completed(cmd, stdout="Verilator 5.047")
    if cmd == ["iverilog", "-V"]:
        return _completed(cmd, stdout="Icarus Verilog version 14")
    if cmd == ["yosys", "--version"]:
        return _completed(cmd, stdout="Yosys 0.64")
    if cmd[0] == "verilator":
        return _completed(cmd, stdout="lint clean")
    if cmd[0] == "iverilog":
        return _completed(cmd, stdout="compiled")
    if cmd[0] == "vvp":
        return _completed(cmd, stdout="SIMULATION RESULT: PASS")
    if cmd[0] == "yosys":
        return _completed(cmd, stdout="hierarchy ok")
    raise AssertionError(cmd)


def _missing_tools_runner(cmd: list[str], **_: Any):
    """Pretend no local HDL tools are available."""

    raise FileNotFoundError(cmd[0])


def _failing_runner(cmd: list[str], **_: Any):
    """Pretend tool probes work but lint and simulation fail."""

    if cmd == ["verilator", "--version"]:
        return _completed(cmd, stdout="Verilator 5.047")
    if cmd == ["iverilog", "-V"]:
        return _completed(cmd, stdout="Icarus Verilog version 14")
    if cmd == ["yosys", "--version"]:
        return _completed(cmd, stdout="Yosys 0.64")
    if cmd[0] == "verilator":
        return _completed(cmd, returncode=1, stderr="syntax error")
    if cmd[0] == "iverilog":
        return _completed(cmd, stdout="compiled")
    if cmd[0] == "vvp":
        return _completed(cmd, returncode=1, stderr="CHECK FAIL")
    if cmd[0] == "yosys":
        return _completed(cmd, returncode=1, stderr="hierarchy failed")
    raise AssertionError(cmd)


def test_req_ising_027_spec_anchor_exists() -> None:
    """REQ-ISING-027, SCENARIO-ISING-037: regression pack is spec-anchored."""

    spec = (exp1476.PROJECT_ROOT / "openspec/capabilities/ising-backend/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-ISING-027" in spec
    assert "SCENARIO-ISING-037" in spec
    assert "experiment_1476_kv260_discrete_sb_rtl_regression_pack.json" in spec


def test_req_ising_027_writes_in_progress_marker(tmp_path: Path) -> None:
    """REQ-ISING-027: the regression pack writes a bootstrap marker first."""

    output = tmp_path / "results" / "experiment_1476_kv260_discrete_sb_rtl_regression_pack.json"

    marker = exp1476.write_in_progress_artifact(output)

    assert marker["status"] == "in_progress"
    assert marker["board_execution_performed"] is False
    assert marker["bitfile_produced"] is False
    assert marker["latency_claimed"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == marker


def test_scenario_ising_037_packages_successful_regression(tmp_path: Path) -> None:
    """SCENARIO-ISING-037: passing local RTL checks produce the manifest artifact."""

    _write_rtl_pair(tmp_path)
    exp1451, exp1460 = _write_prior_artifacts(tmp_path)
    output = tmp_path / "results" / "artifact.json"
    manifest = tmp_path / "hardware" / "kv260" / "discrete_sb_regression_manifest.md"

    artifact = exp1476.run_regression_pack(
        project_root=tmp_path,
        exp1451_path=exp1451,
        exp1460_path=exp1460,
        output_path=output,
        manifest_path=manifest,
        runner=_passing_runner,
    )

    assert exp1476.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["rtl_files"] == ["hardware/kv260/discrete_sb_256.v"]
    assert artifact["testbench_files"] == ["hardware/kv260/discrete_sb_256_tb.v"]
    assert artifact["rtl_regression_complete"] is True
    assert artifact["verilator_lint_passed"] is True
    assert artifact["icarus_sim_passed"] is True
    assert artifact["yosys_available"] is True
    assert artifact["yosys_probe_passed"] is True
    assert artifact["board_execution_performed"] is False
    assert artifact["bitfile_produced"] is False
    assert artifact["latency_claimed"] is False
    assert (
        artifact["regression_manifest_path"] == "hardware/kv260/discrete_sb_regression_manifest.md"
    )
    assert artifact["honest_verdict"] == (
        "rtl_regression_manifest_complete_source_level_only_no_board_bitfile_or_latency_claim"
    )
    assert "verilator --lint-only --timing -Wall" in artifact["verilator_lint_command"]
    assert "iverilog -g2012 -o /tmp/discrete_sb_256_tb_1476.vvp" in artifact["icarus_sim_command"]
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    manifest_text = manifest.read_text(encoding="utf-8")
    assert "REQ-ISING-027" in manifest_text
    assert "SCENARIO-ISING-037" in manifest_text
    assert "Expected output: return code 0 and `SIMULATION RESULT: PASS`" in manifest_text
    assert "Board execution performed: `false`" in manifest_text


def test_req_ising_027_blocks_on_missing_source_or_prior_boundary(tmp_path: Path) -> None:
    """REQ-ISING-027: missing source or unsafe prior claims block completion."""

    exp1451, exp1460 = _write_prior_artifacts(tmp_path)
    missing = exp1476.run_regression_pack(
        project_root=tmp_path,
        exp1451_path=exp1451,
        exp1460_path=exp1460,
        output_path=tmp_path / "results" / "missing.json",
        manifest_path=tmp_path / "hardware" / "kv260" / "missing.md",
        runner=_passing_runner,
    )
    assert missing["rtl_regression_complete"] is False
    assert missing["rtl_files"] == []
    assert missing["testbench_files"] == []
    assert missing["rtl_errors"][0]["error_class"] == "rtl_or_testbench_missing"
    assert (
        missing["honest_verdict"]
        == "blocked_missing_discrete_sb_rtl_or_testbench_no_hardware_claim"
    )

    _write_rtl_pair(tmp_path)
    unsafe1451, unsafe1460 = _write_prior_artifacts(
        tmp_path,
        exp1451_hardware_claim=True,
        exp1460_source_only=False,
    )
    unsafe = exp1476.run_regression_pack(
        project_root=tmp_path,
        exp1451_path=unsafe1451,
        exp1460_path=unsafe1460,
        output_path=tmp_path / "results" / "unsafe.json",
        manifest_path=tmp_path / "hardware" / "kv260" / "unsafe.md",
        runner=_passing_runner,
    )
    assert unsafe["rtl_regression_complete"] is False
    assert unsafe["prior_boundary_preserved"] is False
    assert unsafe["board_execution_performed"] is False
    assert unsafe["honest_verdict"] == "blocked_prior_artifact_claim_boundary_not_preserved"


def test_req_ising_027_records_missing_tools_and_command_failures(tmp_path: Path) -> None:
    """REQ-ISING-027: tool absence and nonzero exits remain explicit."""

    _write_rtl_pair(tmp_path)
    exp1451, exp1460 = _write_prior_artifacts(tmp_path)

    missing_tools = exp1476.run_regression_pack(
        project_root=tmp_path,
        exp1451_path=exp1451,
        exp1460_path=exp1460,
        output_path=tmp_path / "results" / "missing_tools.json",
        manifest_path=tmp_path / "hardware" / "kv260" / "missing_tools.md",
        runner=_missing_tools_runner,
    )
    assert missing_tools["verilator_lint_passed"] is False
    assert missing_tools["icarus_sim_passed"] is False
    assert missing_tools["yosys_available"] is False
    assert missing_tools["rtl_errors"][0]["error_class"] == "verilator_unavailable"
    assert missing_tools["rtl_errors"][1]["error_class"] == "iverilog_unavailable"

    failing = exp1476.run_regression_pack(
        project_root=tmp_path,
        exp1451_path=exp1451,
        exp1460_path=exp1460,
        output_path=tmp_path / "results" / "failing.json",
        manifest_path=tmp_path / "hardware" / "kv260" / "failing.md",
        runner=_failing_runner,
    )
    assert failing["verilator_lint_passed"] is False
    assert failing["icarus_sim_passed"] is False
    assert failing["yosys_available"] is True
    assert failing["yosys_probe_passed"] is False
    assert any(error["stage"] == "verilator_lint" for error in failing["rtl_errors"])
    assert any(error["stage"] == "icarus_simulation" for error in failing["rtl_errors"])
    assert failing["honest_verdict"] == "blocked_rtl_regression_checks_failed_no_hardware_claim"


def test_req_ising_027_artifact_validation_enforces_claim_boundary() -> None:
    """REQ-ISING-027: terminal schema and claim fields are validated."""

    valid = {field: None for field in exp1476.REQUIRED_ARTIFACT_FIELDS}
    valid.update(
        {
            "status": "complete",
            "rtl_regression_complete": True,
            "verilator_lint_passed": True,
            "icarus_sim_passed": True,
            "yosys_available": True,
            "board_execution_performed": False,
            "bitfile_produced": False,
            "latency_claimed": False,
        }
    )
    exp1476.validate_artifact(valid)

    missing = dict(valid)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing"):
        exp1476.validate_artifact(missing)

    bad_status = dict(valid)
    bad_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        exp1476.validate_artifact(bad_status)

    for claim_field in ("board_execution_performed", "bitfile_produced", "latency_claimed"):
        dishonest = dict(valid)
        dishonest[claim_field] = True
        with pytest.raises(ValueError, match=claim_field):
            exp1476.validate_artifact(dishonest)


def test_req_ising_027_command_error_summaries_and_helper_edges(tmp_path: Path) -> None:
    """REQ-ISING-027: command wrapper and boundary helpers classify edge cases."""

    timeout_result = exp1476._run_command(
        ["verilator", "--version"],
        cwd=tmp_path,
        runner=lambda cmd, **_: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd=cmd, timeout=1, output="o" * 5000, stderr="late")
        ),
        timeout=1,
    )
    os_error_result = exp1476._run_command(
        ["yosys", "--version"],
        cwd=tmp_path,
        runner=lambda cmd, **_: (_ for _ in ()).throw(OSError("permission denied")),
        timeout=1,
    )

    assert timeout_result["error_class"] == "timeout"
    assert timeout_result["stdout_summary"].endswith("...[truncated]")
    assert os_error_result["error_class"] == "os_error"
    assert exp1476._exp1460_keeps_kv260_source_only({"active_hardware_tracks": []}) is False
    assert exp1476._relative_to_root(Path("/tmp/outside.md"), tmp_path) == "/tmp/outside.md"
