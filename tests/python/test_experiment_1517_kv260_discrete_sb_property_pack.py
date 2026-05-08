"""Tests for Exp 1517 KV260 Discrete SB source-level property pack.

Spec traces: REQ-ISING-028, SCENARIO-ISING-038
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.hardware import kv260_discrete_sb_property_pack as exp1517


def _completed(cmd: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""):
    """Build a subprocess result for mocked HDL commands."""

    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def _write_exp1506(root: Path, *, active: bool = True) -> Path:
    """Create the milestone gate artifact for Exp 1517."""

    path = root / "results" / "experiment_1506_115_completion_archive_116_activation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "prior_kv260_source_track_active": active,
                "honest_verdict": "complete: milestone_116_activation_complete",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_exp1460(root: Path, *, requested_name: bool = False) -> Path:
    """Create the Exp 1460 source-only hardware portfolio artifact."""

    filename = (
        "experiment_1460_hardware_track_priority_retro.json"
        if requested_name
        else "experiment_1460_hardware_portfolio_narrowing.json"
    )
    path = root / "results" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "active_hardware_tracks": [
                    {
                        "track_id": "kv260_discrete_sb_rtl_sim",
                        "claim_boundary": (
                            "No KV260 board execution, bitfile, or latency claim until "
                            "Vivado synthesis, bitfile flashing, and board commands are captured."
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_rtl_bundle(root: Path) -> None:
    """Create a compact source bundle with the Exp 1517 property tokens."""

    kv260 = root / "hardware" / "kv260"
    kv260.mkdir(parents=True, exist_ok=True)
    (kv260 / "discrete_sb_256.v").write_text(
        """
module discrete_sb_256 #(
    parameter integer N_VARIABLES = 256,
    parameter integer COUPLING_BITS = 8
) (
    input wire clk,
    input wire rst,
    input wire start,
    input wire load_init,
    input wire [4:0] init_word_index,
    input wire [31:0] init_word_data,
    input wire load_coupling,
    input wire [15:0] coupling_addr,
    input wire signed [COUPLING_BITS-1:0] coupling_data,
    input wire [15:0] max_steps,
    input wire signed [15:0] eta_q1_15,
    input wire signed [15:0] pressure_start_q1_15,
    input wire signed [15:0] pressure_delta_q1_15,
    output reg busy,
    output reg done,
    output reg [N_VARIABLES-1:0] spin_out,
    output reg [15:0] step_count,
    output reg [7:0] row_index
);
localparam integer COUPLING_COUNT = N_VARIABLES * N_VARIABLES;
localparam [1:0] STATE_IDLE = 2'd0;
localparam [1:0] STATE_ROW = 2'd1;
localparam [1:0] STATE_COMMIT = 2'd2;
reg [1:0] state;
reg [N_VARIABLES-1:0] spin_cur;
reg [N_VARIABLES-1:0] spin_snapshot;
reg [N_VARIABLES-1:0] spin_next;
reg [7:0] row_idx;
reg [7:0] col_idx;
reg signed [31:0] field_acc;
reg [15:0] max_steps_active;
reg signed [15:0] eta_active;
reg signed [15:0] pressure_q1_15;
reg signed [15:0] pressure_delta_active;
reg signed [COUPLING_BITS-1:0] j_matrix [0:COUPLING_COUNT-1];
reg signed [49:0] candidate_q1_15;
always @(posedge clk) begin
    if (rst) begin
        state <= STATE_IDLE;
        busy <= 1'b0;
        done <= 1'b0;
        spin_cur <= {N_VARIABLES{1'b1}};
        spin_snapshot <= {N_VARIABLES{1'b1}};
        spin_next <= {N_VARIABLES{1'b0}};
        spin_out <= {N_VARIABLES{1'b1}};
        row_idx <= 8'd0;
        col_idx <= 8'd0;
        row_index <= 8'd0;
        field_acc <= 32'd0;
        step_count <= 16'd0;
    end else if (start) begin
        state <= STATE_ROW;
        spin_snapshot <= spin_cur;
        spin_next <= {N_VARIABLES{1'b0}};
        max_steps_active <= (max_steps == 16'd0) ? 16'd128 : max_steps;
        eta_active <= eta_q1_15;
        pressure_q1_15 <= pressure_start_q1_15;
        pressure_delta_active <= pressure_delta_q1_15;
    end else if (state == STATE_ROW) begin
        if (spin_snapshot[col_idx]) begin
            field_acc <= field_acc + j_matrix[(row_idx * N_VARIABLES) + col_idx];
        end
        if (col_idx == (N_VARIABLES - 1)) begin
            spin_next[row_idx] <= (candidate_q1_15 >= 50'sd0);
            if (row_idx == (N_VARIABLES - 1)) begin
                state <= STATE_COMMIT;
            end
        end
    end else if (state == STATE_COMMIT) begin
        spin_cur <= spin_next;
        spin_snapshot <= spin_next;
        spin_out <= spin_next;
        step_count <= step_count + 16'd1;
        if ((step_count + 16'd1) >= max_steps_active) begin
            state <= STATE_IDLE;
        end
    end
end
endmodule
""",
        encoding="utf-8",
    )
    (kv260 / "discrete_sb_256_tb.v").write_text(
        'module discrete_sb_256_tb; initial $display("SIMULATION RESULT: PASS"); endmodule\n',
        encoding="utf-8",
    )
    (kv260 / "discrete_sb_256_property_tb.sv").write_text(
        """
module discrete_sb_256_property_tb;
// PROP_RESET_KNOWN_STATE
// PROP_BOUNDED_ONE_STEP_DONE
// PROP_SNAPSHOT_STABLE_DURING_ROW_UPDATE
// PROP_SHAPE_WIDTH_DEFAULTS
endmodule
""",
        encoding="utf-8",
    )
    (kv260 / "discrete_sb_regression_manifest.md").write_text(
        "Spec traces: REQ-ISING-027, SCENARIO-ISING-037\n",
        encoding="utf-8",
    )


def _passing_runner(cmd: list[str], **_: Any):
    """Pretend local HDL tools are available and all source-level checks pass."""

    if cmd == ["verilator", "--version"]:
        return _completed(cmd, stdout="Verilator 5.047")
    if cmd == ["iverilog", "-V"]:
        return _completed(cmd, stdout="Icarus Verilog version 14")
    if cmd == ["yosys", "--version"]:
        return _completed(cmd, stdout="Yosys 0.64")
    if cmd[0] == "verilator":
        return _completed(cmd, stdout="property lint clean")
    if cmd[0] == "iverilog" and "-tnull" in cmd:
        return _completed(cmd, stdout="property parse clean")
    if cmd[0] == "iverilog":
        return _completed(cmd, stdout="property compiled")
    if cmd[0] == "vvp":
        return _completed(cmd, stdout="PROPERTY RESULT: PASS")
    if cmd[0] == "yosys":
        return _completed(cmd, stdout="parse ok")
    raise AssertionError(cmd)


def _missing_tools_runner(cmd: list[str], **_: Any):
    """Pretend no local HDL tools are available."""

    raise FileNotFoundError(cmd[0])


def _failing_runner(cmd: list[str], **_: Any):
    """Pretend tools are present but property lint and simulation fail."""

    if cmd == ["verilator", "--version"]:
        return _completed(cmd, stdout="Verilator 5.047")
    if cmd == ["iverilog", "-V"]:
        return _completed(cmd, stdout="Icarus Verilog version 14")
    if cmd == ["yosys", "--version"]:
        return _completed(cmd, stdout="Yosys 0.64")
    if cmd[0] == "verilator":
        return _completed(cmd, returncode=1, stderr="property lint failed")
    if cmd[0] == "iverilog" and "-tnull" in cmd:
        return _completed(cmd, returncode=1, stderr="property parse failed")
    if cmd[0] == "iverilog":
        return _completed(cmd, stdout="compiled")
    if cmd[0] == "vvp":
        return _completed(cmd, returncode=1, stderr="PROPERTY FAIL")
    if cmd[0] == "yosys":
        return _completed(cmd, returncode=1, stderr="parse failed")
    raise AssertionError(cmd)


def _compile_failing_runner(cmd: list[str], **_: Any):
    """Pretend lint and parse work but the property simulation compile fails."""

    if cmd == ["verilator", "--version"]:
        return _completed(cmd, stdout="Verilator 5.047")
    if cmd == ["iverilog", "-V"]:
        return _completed(cmd, stdout="Icarus Verilog version 14")
    if cmd == ["yosys", "--version"]:
        return _completed(cmd, stdout="Yosys 0.64")
    if cmd[0] == "verilator":
        return _completed(cmd, stdout="property lint clean")
    if cmd[0] == "iverilog" and "-tnull" in cmd:
        return _completed(cmd, stdout="property parse clean")
    if cmd[0] == "iverilog":
        return _completed(cmd, returncode=1, stderr="compile failed")
    raise AssertionError(cmd)


def test_req_ising_028_spec_anchor_exists() -> None:
    """REQ-ISING-028, SCENARIO-ISING-038: property-pack work is spec-anchored."""

    spec = (exp1517.PROJECT_ROOT / "openspec/capabilities/ising-backend/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-ISING-028" in spec
    assert "SCENARIO-ISING-038" in spec
    assert "experiment_1517_kv260_discrete_sb_rtl_property_pack_v2.json" in spec
    assert "kv260_discrete_sb_property_manifest_1517.json" in spec


def test_req_ising_028_writes_in_progress_marker(tmp_path: Path) -> None:
    """REQ-ISING-028: the terminal artifact starts with source-only boundaries."""

    output = tmp_path / "results" / "experiment_1517.json"

    marker = exp1517.write_in_progress_artifact(output)

    assert exp1517.REQUIRED_ARTIFACT_FIELDS <= set(marker)
    assert marker["status"] == "in_progress"
    assert marker["source_level_only"] is True
    assert marker["no_board_execution"] is True
    assert marker["no_bitstream_claim"] is True
    assert marker["kv260_property_pack_ready"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == marker


def test_scenario_ising_038_writes_successful_property_manifest(tmp_path: Path) -> None:
    """SCENARIO-ISING-038: passing source checks produce the property manifest."""

    _write_exp1506(tmp_path)
    _write_exp1460(tmp_path)
    _write_rtl_bundle(tmp_path)
    output = tmp_path / "results" / "experiment_1517.json"
    manifest = tmp_path / "results" / "kv260_discrete_sb_property_manifest_1517.json"

    artifact = exp1517.run_property_pack(
        project_root=tmp_path,
        output_path=output,
        manifest_path=manifest,
        runner=_passing_runner,
    )

    assert exp1517.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["kv260_property_pack_ready"] is True
    assert artifact["gated_inputs_present"] is True
    assert artifact["source_level_only"] is True
    assert artifact["no_board_execution"] is True
    assert artifact["no_bitstream_claim"] is True
    assert (
        artifact["property_manifest_path"]
        == "results/kv260_discrete_sb_property_manifest_1517.json"
    )
    assert len(artifact["properties_defined"]) == 4
    assert {item["category"] for item in artifact["properties_defined"]} == {
        "bounded_behavior",
        "reset_behavior",
        "deterministic_update_ordering",
        "shape_width_assumptions",
    }
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["source_level_only"] is True
    assert payload["no_board_execution"] is True
    assert payload["no_bitstream_claim"] is True
    assert payload["prior_artifacts"]["exp1460"]["path"] == (
        "results/experiment_1460_hardware_portfolio_narrowing.json"
    )
    assert payload["path_mismatches"] == [
        {
            "requested": "results/experiment_1460_hardware_track_priority_retro.json",
            "actual": "results/experiment_1460_hardware_portfolio_narrowing.json",
        }
    ]
    assert any(
        result["stage"] == "verilator_property_lint" for result in payload["lint_or_parse_results"]
    )
    assert any(
        result["stage"] == "iverilog_property_parse" for result in payload["lint_or_parse_results"]
    )
    assert payload["simulations_run"][0]["stage"] == "icarus_property_simulation"


def test_req_ising_028_records_requested_exp1460_path_when_present(tmp_path: Path) -> None:
    """REQ-ISING-028: no path mismatch is recorded when the requested path exists."""

    _write_exp1506(tmp_path)
    requested_exp1460 = _write_exp1460(tmp_path, requested_name=True)
    _write_rtl_bundle(tmp_path)

    artifact = exp1517.run_property_pack(
        project_root=tmp_path,
        exp1460_requested_path=requested_exp1460,
        exp1460_fallback_path=tmp_path / "results" / "unused.json",
        output_path=tmp_path / "results" / "experiment_1517.json",
        manifest_path=tmp_path / "results" / "manifest.json",
        runner=_passing_runner,
    )

    manifest = json.loads((tmp_path / "results" / "manifest.json").read_text(encoding="utf-8"))
    assert artifact["kv260_property_pack_ready"] is True
    assert manifest["path_mismatches"] == []
    assert manifest["prior_artifacts"]["exp1460"]["path"] == (
        "results/experiment_1460_hardware_track_priority_retro.json"
    )


def test_req_ising_028_blocks_when_prior_gate_is_absent(tmp_path: Path) -> None:
    """REQ-ISING-028: inactive Exp 1506 gate writes a terminal gated artifact."""

    exp1506 = _write_exp1506(tmp_path, active=False)
    output = tmp_path / "results" / "experiment_1517.json"

    artifact = exp1517.run_property_pack(
        project_root=tmp_path,
        exp1506_path=exp1506,
        output_path=output,
        manifest_path=tmp_path / "results" / "manifest.json",
        runner=_passing_runner,
    )

    assert artifact["status"] == "complete"
    assert artifact["kv260_property_pack_ready"] is False
    assert artifact["gated_inputs_present"] is False
    assert artifact["property_manifest_path"] == ""
    assert artifact["blockers"][0]["error_class"] == "prior_kv260_source_track_inactive"
    assert artifact["source_level_only"] is True
    assert artifact["no_board_execution"] is True
    assert artifact["no_bitstream_claim"] is True
    assert artifact["honest_verdict"] == "complete: gated_prior_kv260_source_track_inactive"
    assert not (tmp_path / "results" / "manifest.json").exists()


def test_req_ising_028_records_missing_tools_and_failed_commands(tmp_path: Path) -> None:
    """REQ-ISING-028: local tool absence and command failures remain explicit."""

    _write_exp1506(tmp_path)
    _write_exp1460(tmp_path)
    _write_rtl_bundle(tmp_path)

    missing = exp1517.run_property_pack(
        project_root=tmp_path,
        output_path=tmp_path / "results" / "missing.json",
        manifest_path=tmp_path / "results" / "missing_manifest.json",
        runner=_missing_tools_runner,
    )
    assert missing["kv260_property_pack_ready"] is False
    assert {item["error_class"] for item in missing["blockers"]} == {
        "no_verilator",
        "no_iverilog",
        "no_source_level_command_executed",
    }
    assert all(result["returncode"] is None for result in missing["lint_or_parse_results"])
    assert missing["simulations_run"][0]["returncode"] is None

    failing = exp1517.run_property_pack(
        project_root=tmp_path,
        output_path=tmp_path / "results" / "failing.json",
        manifest_path=tmp_path / "results" / "failing_manifest.json",
        runner=_failing_runner,
    )
    assert failing["kv260_property_pack_ready"] is False
    assert any(blocker["stage"] == "verilator_property_lint" for blocker in failing["blockers"])
    assert any(blocker["stage"] == "iverilog_property_parse" for blocker in failing["blockers"])
    assert any(blocker["stage"] == "icarus_property_simulation" for blocker in failing["blockers"])


def test_req_ising_028_records_missing_source_and_bad_exp1460_boundary(tmp_path: Path) -> None:
    """REQ-ISING-028: source and prior-boundary blockers are terminal evidence."""

    _write_exp1506(tmp_path)

    artifact = exp1517.run_property_pack(
        project_root=tmp_path,
        exp1460_requested_path=tmp_path / "results" / "missing_requested.json",
        exp1460_fallback_path=tmp_path / "results" / "missing_fallback.json",
        output_path=tmp_path / "results" / "missing_source.json",
        manifest_path=tmp_path / "results" / "missing_source_manifest.json",
        runner=_passing_runner,
    )

    assert artifact["kv260_property_pack_ready"] is False
    assert artifact["gated_inputs_present"] is False
    assert {blocker["error_class"] for blocker in artifact["blockers"]} >= {
        "exp1460_source_only_boundary_missing",
        "source_file_missing",
        "property_tokens_missing",
    }
    assert all(
        result["error_class"] == "source_bundle_missing"
        for result in artifact["lint_or_parse_results"]
    )
    assert artifact["simulations_run"][0]["error_class"] == "source_bundle_missing"


def test_req_ising_028_records_property_compile_failure(tmp_path: Path) -> None:
    """REQ-ISING-028: simulation compile failures skip vvp and block readiness."""

    _write_exp1506(tmp_path)
    _write_exp1460(tmp_path)
    _write_rtl_bundle(tmp_path)

    artifact = exp1517.run_property_pack(
        project_root=tmp_path,
        output_path=tmp_path / "results" / "compile_failed.json",
        manifest_path=tmp_path / "results" / "compile_failed_manifest.json",
        runner=_compile_failing_runner,
    )

    sim = artifact["simulations_run"][0]
    assert sim["returncode"] == 1
    assert sim["run_result"]["error_class"] == "compile_failed"
    assert any(blocker["stage"] == "icarus_property_simulation" for blocker in artifact["blockers"])


def test_req_ising_028_command_error_classes_are_json_safe(tmp_path: Path) -> None:
    """REQ-ISING-028: subprocess errors are captured as compact JSON-safe records."""

    not_found = exp1517.run_command(["missing-tool"], cwd=tmp_path, runner=_missing_tools_runner)
    assert not_found["error_class"] == "not_found"
    assert not_found["returncode"] is None

    def timeout_runner(cmd: list[str], **_: Any):
        raise subprocess.TimeoutExpired(cmd, timeout=1, output="partial", stderr="late")

    timed_out = exp1517.run_command(["slow-tool"], cwd=tmp_path, runner=timeout_runner)
    assert timed_out["error_class"] == "timeout"
    assert timed_out["stdout_summary"] == "partial"
    assert timed_out["stderr_summary"] == "late"

    def os_error_runner(cmd: list[str], **_: Any):
        raise OSError(f"cannot exec {cmd[0]}")

    os_error = exp1517.run_command(["broken-tool"], cwd=tmp_path, runner=os_error_runner)
    assert os_error["error_class"] == "os_error"
    assert os_error["error"] == "cannot exec broken-tool"

    def long_output_runner(cmd: list[str], **_: Any):
        return _completed(cmd, stdout="x" * (exp1517.SUMMARY_LIMIT + 20))

    long_output = exp1517.run_command(["chatty-tool"], cwd=tmp_path, runner=long_output_runner)
    assert long_output["returncode"] == 0
    assert long_output["stdout_summary"].endswith("...[truncated]")


def test_req_ising_028_artifact_validation_enforces_schema_and_claim_boundary() -> None:
    """REQ-ISING-028: terminal artifacts must keep source-only claim fields true."""

    valid = {
        "status": "complete",
        "kv260_property_pack_ready": True,
        "gated_inputs_present": True,
        "source_level_only": True,
        "no_board_execution": True,
        "no_bitstream_claim": True,
        "rtl_files_checked": [],
        "properties_defined": [],
        "simulations_run": [],
        "lint_or_parse_results": [],
        "property_manifest_path": "results/manifest.json",
        "blockers": [],
        "honest_verdict": "complete: ok",
    }
    exp1517.validate_terminal_artifact(valid)

    missing = dict(valid)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing"):
        exp1517.validate_terminal_artifact(missing)

    for field in ("source_level_only", "no_board_execution", "no_bitstream_claim"):
        dishonest = dict(valid)
        dishonest[field] = False
        with pytest.raises(ValueError, match=field):
            exp1517.validate_terminal_artifact(dishonest)

    bad_status = dict(valid)
    bad_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        exp1517.validate_terminal_artifact(bad_status)

    bad_verdict = dict(valid)
    bad_verdict["honest_verdict"] = "blocked_without_allowed_prefix"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1517.validate_terminal_artifact(bad_verdict)

    outside = exp1517._relative_to_root(Path("/tmp/outside.json"), Path("/var/tmp/root"))
    assert outside == "/tmp/outside.json"
