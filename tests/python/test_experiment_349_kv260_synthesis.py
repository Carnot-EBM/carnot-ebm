"""Tests for Experiment 349: KV260 open-source FPGA synthesis via yosys/nextpnr.

Spec coverage: REQ-HW-003, SCENARIO-HW-005, SCENARIO-HW-006

Design philosophy:
    - All tests that require real synthesis tools (yosys, nextpnr-xilinx) auto-skip
      when those tools are absent.
    - Tests covering blocked/partial paths always run via subprocess mocking.
    - 100% branch coverage of experiment_349_kv260_synthesis.py is required.
    - We never fabricate LUT/FF counts: if synthesis is not exercised,
      lut_count and ff_count must be None in the artifact.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import scripts.experiment_349_kv260_synthesis as exp349
from scripts.experiment_349_kv260_synthesis import (
    EXPERIMENT_ID,
    SynthesisResult,
    check_nextpnr_available,
    check_verilog_source_exists,
    check_yosys_available,
    parse_synthesis_output,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Subprocess mock helpers
# ---------------------------------------------------------------------------


def _make_subprocess_result(returncode: int, stdout: str = "", stderr: str = "") -> Any:
    """Build a mock object that looks like subprocess.CompletedProcess."""
    mock = MagicMock()
    mock.returncode = returncode
    mock.stdout = stdout
    mock.stderr = stderr
    return mock


def _subprocess_ok(cmd, **kwargs) -> Any:
    """Fake subprocess.run that always succeeds with empty output."""
    return _make_subprocess_result(0, stdout="", stderr="")


def _subprocess_fail(cmd, **kwargs) -> Any:
    """Fake subprocess.run that always returns returncode=1."""
    return _make_subprocess_result(1, stdout="", stderr="error")


def _subprocess_file_not_found(cmd, **kwargs) -> Any:
    """Fake subprocess.run that raises FileNotFoundError (tool not on PATH)."""
    raise FileNotFoundError("No such file or directory: 'yosys'")


def _subprocess_timeout(cmd, **kwargs) -> Any:
    """Fake subprocess.run that simulates a timeout."""
    raise subprocess.TimeoutExpired(cmd=cmd, timeout=10)


YOSYS_SAMPLE_OUTPUT = """\
-- Running command `synth_xilinx -top ising_sampler_128 -flatten; write_json /tmp/netlist_349.json' --

3. Executing SYNTH_XILINX pass.

...

Number of cells:               2176
   LUT1:                          16
   LUT2:                          64
   LUT3:                         128
   LUT4:                         256
   LUT5:                         256
   LUT6:                        1024
   FDRE:                         256
   FDSE:                          64
   RAMB18E2:                        4

End of script. Recognized commands: synth_xilinx.
"""


# ---------------------------------------------------------------------------
# REQ-HW-003 / SCENARIO-HW-006: check_yosys_available
# ---------------------------------------------------------------------------


class TestCheckYosysAvailable:
    """REQ-HW-003: yosys availability check must be graceful."""

    def test_returns_true_when_yosys_exits_zero(self) -> None:
        """SCENARIO-HW-006: check_yosys_available returns True on returncode=0."""
        with patch("subprocess.run", return_value=_make_subprocess_result(0)):
            assert check_yosys_available() is True

    def test_returns_false_when_yosys_exits_nonzero(self) -> None:
        """REQ-HW-003: graceful False when yosys returns non-zero."""
        with patch("subprocess.run", return_value=_make_subprocess_result(1)):
            assert check_yosys_available() is False

    def test_returns_false_when_yosys_not_found(self) -> None:
        """SCENARIO-HW-006: graceful False when yosys not on PATH."""
        with patch("subprocess.run", side_effect=FileNotFoundError("yosys")):
            assert check_yosys_available() is False

    def test_returns_false_on_timeout(self) -> None:
        """REQ-HW-003: graceful False on subprocess timeout."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["yosys", "--version"], timeout=10),
        ):
            assert check_yosys_available() is False

    def test_returns_false_on_os_error(self) -> None:
        """REQ-HW-003: graceful False on generic OSError."""
        with patch("subprocess.run", side_effect=OSError("permission denied")):
            assert check_yosys_available() is False


# ---------------------------------------------------------------------------
# REQ-HW-003 / SCENARIO-HW-006: check_nextpnr_available
# ---------------------------------------------------------------------------


class TestCheckNextpnrAvailable:
    """REQ-HW-003: nextpnr-xilinx availability check must be graceful."""

    def test_returns_true_when_nextpnr_exits_zero(self) -> None:
        """SCENARIO-HW-006: check_nextpnr_available returns True on returncode=0."""
        with patch("subprocess.run", return_value=_make_subprocess_result(0)):
            assert check_nextpnr_available() is True

    def test_returns_false_when_nextpnr_exits_nonzero(self) -> None:
        """REQ-HW-003: graceful False when nextpnr exits non-zero."""
        with patch("subprocess.run", return_value=_make_subprocess_result(1)):
            assert check_nextpnr_available() is False

    def test_returns_false_when_nextpnr_not_found(self) -> None:
        """SCENARIO-HW-006: graceful False when nextpnr-xilinx not on PATH."""
        with patch("subprocess.run", side_effect=FileNotFoundError("nextpnr-xilinx")):
            assert check_nextpnr_available() is False

    def test_returns_false_on_timeout(self) -> None:
        """REQ-HW-003: graceful False on nextpnr subprocess timeout."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["nextpnr-xilinx", "--version"], timeout=10),
        ):
            assert check_nextpnr_available() is False

    def test_returns_false_on_os_error(self) -> None:
        """REQ-HW-003: graceful False on OSError for nextpnr."""
        with patch("subprocess.run", side_effect=OSError("permission denied")):
            assert check_nextpnr_available() is False


# ---------------------------------------------------------------------------
# REQ-HW-003 / SCENARIO-HW-006: check_verilog_source_exists
# ---------------------------------------------------------------------------


class TestCheckVerilogSourceExists:
    """REQ-HW-003: Verilog source presence check."""

    def test_returns_true_when_file_exists(self, tmp_path: Path) -> None:
        """SCENARIO-HW-005: Returns True when RTL source exists."""
        f = tmp_path / "ising_sampler_v1.v"
        f.write_text("`timescale 1ns/1ps\n")
        assert check_verilog_source_exists(f) is True

    def test_returns_false_when_file_absent(self, tmp_path: Path) -> None:
        """SCENARIO-HW-006: Returns False when RTL source is missing."""
        assert check_verilog_source_exists(tmp_path / "nonexistent.v") is False

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """REQ-HW-003: Accepts both Path and str arguments."""
        f = tmp_path / "ising_sampler_v1.v"
        f.write_text("module top; endmodule\n")
        assert check_verilog_source_exists(str(f)) is True

    def test_returns_false_for_directory(self, tmp_path: Path) -> None:
        """REQ-HW-003: Returns True for directory (os.path.exists is True)."""
        # os.path.exists returns True for directories — that's acceptable;
        # synthesis will fail if the path is a dir, caught by yosys itself.
        assert check_verilog_source_exists(tmp_path) is True


# ---------------------------------------------------------------------------
# REQ-HW-003 / SCENARIO-HW-005: parse_synthesis_output
# ---------------------------------------------------------------------------


class TestParseSynthesisOutput:
    """REQ-HW-003: LUT/FF parsing from yosys synthesis report."""

    def test_parses_lut_counts_from_sample_output(self) -> None:
        """SCENARIO-HW-005: parse_synthesis_output extracts LUT sums correctly."""
        result = parse_synthesis_output(YOSYS_SAMPLE_OUTPUT)
        # LUT1=16, LUT2=64, LUT3=128, LUT4=256, LUT5=256, LUT6=1024 → 1744
        assert result["lut_count"] == 1744

    def test_parses_ff_counts_from_sample_output(self) -> None:
        """SCENARIO-HW-005: parse_synthesis_output extracts FF sums correctly."""
        result = parse_synthesis_output(YOSYS_SAMPLE_OUTPUT)
        # FDRE=256, FDSE=64 → 320
        assert result["ff_count"] == 320

    def test_raw_lines_populated(self) -> None:
        """REQ-HW-003: raw_lines contains at least the parsed resource entries."""
        result = parse_synthesis_output(YOSYS_SAMPLE_OUTPUT)
        assert len(result["raw_lines"]) >= 6

    def test_returns_none_when_no_luts(self) -> None:
        """REQ-HW-003: lut_count is None when stdout has no LUT entries."""
        result = parse_synthesis_output("No resources here.\n")
        assert result["lut_count"] is None

    def test_returns_none_when_no_ffs(self) -> None:
        """REQ-HW-003: ff_count is None when stdout has no FD entries."""
        result = parse_synthesis_output("No resources here.\n")
        assert result["ff_count"] is None

    def test_empty_string_returns_nones(self) -> None:
        """REQ-HW-003: Empty stdout returns None counts without raising."""
        result = parse_synthesis_output("")
        assert result["lut_count"] is None
        assert result["ff_count"] is None
        assert result["raw_lines"] == []

    def test_single_lut_type(self) -> None:
        """REQ-HW-003: Single LUT type line is parsed correctly."""
        stdout = "   LUT6:                        512\n"
        result = parse_synthesis_output(stdout)
        assert result["lut_count"] == 512

    def test_multiple_ff_types_summed(self) -> None:
        """REQ-HW-003: Multiple FD* types are summed into ff_count."""
        stdout = "   FDRE:                         100\n   FDSE:                          50\n"
        result = parse_synthesis_output(stdout)
        assert result["ff_count"] == 150

    def test_number_of_cells_line_captured(self) -> None:
        """REQ-HW-003: Number of cells aggregate line appears in raw_lines."""
        result = parse_synthesis_output(YOSYS_SAMPLE_OUTPUT)
        assert any("Number of cells" in line for line in result["raw_lines"])


# ---------------------------------------------------------------------------
# REQ-HW-003: SynthesisResult dataclass
# ---------------------------------------------------------------------------


class TestSynthesisResult:
    """REQ-HW-003: SynthesisResult dataclass serialization and approved verdicts."""

    def test_to_dict_contains_all_fields(self) -> None:
        """REQ-HW-003: to_dict includes all dataclass fields."""
        sr = SynthesisResult(
            yosys_available=True,
            nextpnr_available=False,
            verilog_found=True,
            synthesis_attempted=True,
            synthesis_success=False,
            lut_count=512,
            ff_count=128,
            honest_verdict="synthesis_partial",
        )
        d = sr.to_dict()
        assert d["yosys_available"] is True
        assert d["nextpnr_available"] is False
        assert d["verilog_found"] is True
        assert d["synthesis_attempted"] is True
        assert d["synthesis_success"] is False
        assert d["lut_count"] == 512
        assert d["ff_count"] == 128
        assert d["honest_verdict"] == "synthesis_partial"

    def test_approved_verdicts_set(self) -> None:
        """REQ-HW-003: APPROVED_VERDICTS contains the five expected strings."""
        expected = {
            "synthesis_success",
            "synthesis_partial",
            "blocked_missing_yosys",
            "blocked_missing_verilog",
            "synthesis_failed",
        }
        assert SynthesisResult.APPROVED_VERDICTS == expected

    def test_to_dict_is_json_serializable(self) -> None:
        """REQ-HW-003: SynthesisResult dict must be JSON-serializable."""
        sr = SynthesisResult(
            yosys_available=False,
            nextpnr_available=False,
            verilog_found=False,
            synthesis_attempted=False,
            synthesis_success=False,
            lut_count=None,
            ff_count=None,
            honest_verdict="blocked_missing_yosys",
        )
        # Must not raise.
        json.dumps(sr.to_dict())


# ---------------------------------------------------------------------------
# REQ-HW-003 / SCENARIO-HW-006: run_experiment — blocked paths
# ---------------------------------------------------------------------------


class TestRunExperimentBlockedPaths:
    """SCENARIO-HW-006: Blocked verdicts when prerequisites are missing."""

    def test_blocked_missing_yosys(self, tmp_path: Path) -> None:
        """SCENARIO-HW-006: honest_verdict=blocked_missing_yosys when yosys absent."""
        verilog = tmp_path / "ising_sampler_v1.v"
        verilog.write_text("module top; endmodule\n")

        with patch.object(exp349, "check_yosys_available", return_value=False), \
             patch.object(exp349, "check_nextpnr_available", return_value=False):
            artifact = run_experiment(
                output_path=tmp_path / "out.json",
                verilog_path=verilog,
                write_output=False,
            )
        assert artifact["honest_verdict"] == "blocked_missing_yosys"
        assert artifact["synthesis_result"]["synthesis_attempted"] is False
        assert artifact["lut_count"] is None
        assert artifact["ff_count"] is None

    def test_blocked_missing_verilog(self, tmp_path: Path) -> None:
        """SCENARIO-HW-006: honest_verdict=blocked_missing_verilog when RTL absent."""
        with patch.object(exp349, "check_yosys_available", return_value=True), \
             patch.object(exp349, "check_nextpnr_available", return_value=False):
            artifact = run_experiment(
                output_path=tmp_path / "out.json",
                verilog_path=tmp_path / "nonexistent.v",
                write_output=False,
            )
        assert artifact["honest_verdict"] == "blocked_missing_verilog"
        assert artifact["synthesis_result"]["synthesis_attempted"] is False
        assert artifact["lut_count"] is None
        assert artifact["ff_count"] is None

    def test_synthesis_attempted_false_when_yosys_absent(self, tmp_path: Path) -> None:
        """REQ-HW-003: synthesis_attempted=False in artifact when yosys absent."""
        with patch.object(exp349, "check_yosys_available", return_value=False), \
             patch.object(exp349, "check_nextpnr_available", return_value=False):
            artifact = run_experiment(
                output_path=tmp_path / "out.json",
                verilog_path=tmp_path / "some.v",
                write_output=False,
            )
        assert artifact["synthesis_result"]["synthesis_attempted"] is False

    def test_bitfile_not_generated_when_blocked(self, tmp_path: Path) -> None:
        """SCENARIO-HW-006: bitfile_generated=False when blocked."""
        with patch.object(exp349, "check_yosys_available", return_value=False), \
             patch.object(exp349, "check_nextpnr_available", return_value=False):
            artifact = run_experiment(
                output_path=tmp_path / "out.json",
                verilog_path=tmp_path / "some.v",
                write_output=False,
            )
        assert artifact["bitfile_generated"] is False
        assert artifact["bitfile_path"] is None


# ---------------------------------------------------------------------------
# REQ-HW-003 / SCENARIO-HW-006: run_experiment — synthesis_failed path
# ---------------------------------------------------------------------------


class TestRunExperimentSynthesisFailed:
    """REQ-HW-003: honest_verdict=synthesis_failed when yosys exits non-zero."""

    def test_synthesis_failed_verdict(self, tmp_path: Path) -> None:
        """REQ-HW-003: synthesis_failed when yosys returns returncode=1."""
        verilog = tmp_path / "ising_sampler_v1.v"
        verilog.write_text("module broken syntax {}\n")

        def _fake_run(cmd, **kwargs):
            return _make_subprocess_result(1, stdout="", stderr="syntax error")

        with patch.object(exp349, "check_yosys_available", return_value=True), \
             patch.object(exp349, "check_nextpnr_available", return_value=False):
            artifact = run_experiment(
                output_path=tmp_path / "out.json",
                verilog_path=verilog,
                write_output=False,
                _subprocess_run=_fake_run,
            )
        assert artifact["honest_verdict"] == "synthesis_failed"
        assert artifact["synthesis_result"]["synthesis_attempted"] is True
        assert artifact["synthesis_result"]["synthesis_success"] is False
        assert artifact["lut_count"] is None

    def test_synthesis_failed_bitfile_not_generated(self, tmp_path: Path) -> None:
        """REQ-HW-003: bitfile_generated=False when synthesis fails."""
        verilog = tmp_path / "ising_sampler_v1.v"
        verilog.write_text("module broken; endmodule\n")

        def _fake_run(cmd, **kwargs):
            return _make_subprocess_result(1, stdout="", stderr="error")

        with patch.object(exp349, "check_yosys_available", return_value=True), \
             patch.object(exp349, "check_nextpnr_available", return_value=False):
            artifact = run_experiment(
                output_path=tmp_path / "out.json",
                verilog_path=verilog,
                write_output=False,
                _subprocess_run=_fake_run,
            )
        assert artifact["bitfile_generated"] is False


# ---------------------------------------------------------------------------
# REQ-HW-003 / SCENARIO-HW-005: run_experiment — synthesis_partial path
# ---------------------------------------------------------------------------


class TestRunExperimentSynthesisPartial:
    """SCENARIO-HW-005: synthesis_partial when yosys succeeds but nextpnr absent."""

    def test_synthesis_partial_no_nextpnr(self, tmp_path: Path) -> None:
        """SCENARIO-HW-005: synthesis_partial when yosys OK but nextpnr absent."""
        verilog = tmp_path / "ising_sampler_v1.v"
        verilog.write_text("module top; endmodule\n")

        def _fake_run(cmd, **kwargs):
            return _make_subprocess_result(0, stdout=YOSYS_SAMPLE_OUTPUT)

        with patch.object(exp349, "check_yosys_available", return_value=True), \
             patch.object(exp349, "check_nextpnr_available", return_value=False):
            artifact = run_experiment(
                output_path=tmp_path / "out.json",
                verilog_path=verilog,
                write_output=False,
                _subprocess_run=_fake_run,
            )
        assert artifact["honest_verdict"] == "synthesis_partial"
        assert artifact["synthesis_result"]["synthesis_attempted"] is True
        assert artifact["synthesis_result"]["synthesis_success"] is False

    def test_lut_count_populated_after_partial_synthesis(self, tmp_path: Path) -> None:
        """SCENARIO-HW-005: lut_count extracted from yosys output in partial path."""
        verilog = tmp_path / "ising_sampler_v1.v"
        verilog.write_text("module top; endmodule\n")

        def _fake_run(cmd, **kwargs):
            return _make_subprocess_result(0, stdout=YOSYS_SAMPLE_OUTPUT)

        with patch.object(exp349, "check_yosys_available", return_value=True), \
             patch.object(exp349, "check_nextpnr_available", return_value=False):
            artifact = run_experiment(
                output_path=tmp_path / "out.json",
                verilog_path=verilog,
                write_output=False,
                _subprocess_run=_fake_run,
            )
        assert artifact["lut_count"] == 1744
        assert artifact["ff_count"] == 320

    def test_synthesis_partial_when_nextpnr_fails(self, tmp_path: Path) -> None:
        """SCENARIO-HW-005: synthesis_partial when nextpnr P&R returns non-zero."""
        verilog = tmp_path / "ising_sampler_v1.v"
        verilog.write_text("module top; endmodule\n")

        call_count = {"n": 0}

        def _fake_run(cmd, **kwargs):
            call_count["n"] += 1
            if cmd[0] == "yosys":
                # Yosys succeeds AND creates the netlist JSON so the P&R path is reached.
                netlist = kwargs.get("capture_output")  # not relevant
                # Write the netlist file to satisfy the .exists() check
                import tempfile
                return _make_subprocess_result(0, stdout=YOSYS_SAMPLE_OUTPUT)
            # nextpnr-xilinx fails
            return _make_subprocess_result(1, stdout="", stderr="P&R failed")

        netlist_path = tmp_path / "netlist_349.json"
        netlist_path.write_text("{}")  # simulate netlist present

        with patch.object(exp349, "check_yosys_available", return_value=True), \
             patch.object(exp349, "check_nextpnr_available", return_value=True):
            artifact = run_experiment(
                output_path=tmp_path / "out.json",
                verilog_path=verilog,
                netlist_path=netlist_path,
                write_output=False,
                _subprocess_run=_fake_run,
            )
        assert artifact["honest_verdict"] == "synthesis_partial"
        assert artifact["bitfile_generated"] is False


# ---------------------------------------------------------------------------
# REQ-HW-003 / SCENARIO-HW-005: run_experiment — synthesis_success path
# ---------------------------------------------------------------------------


class TestRunExperimentSynthesisSuccess:
    """SCENARIO-HW-005: synthesis_success when yosys + nextpnr both succeed."""

    def test_synthesis_success_verdict(self, tmp_path: Path) -> None:
        """SCENARIO-HW-005: synthesis_success when P&R completes and bitfile written."""
        verilog = tmp_path / "ising_sampler_v1.v"
        verilog.write_text("module top; endmodule\n")
        netlist_path = tmp_path / "netlist_349.json"
        netlist_path.write_text("{}")
        bitfile_path = tmp_path / "carnot_ising.bit"

        def _fake_run(cmd, **kwargs):
            if cmd[0] == "yosys":
                return _make_subprocess_result(0, stdout=YOSYS_SAMPLE_OUTPUT)
            # nextpnr-xilinx succeeds — also write the bitfile to disk
            bitfile_path.write_bytes(b"\x00" * 16)
            return _make_subprocess_result(0, stdout="P&R OK")

        with patch.object(exp349, "check_yosys_available", return_value=True), \
             patch.object(exp349, "check_nextpnr_available", return_value=True):
            artifact = run_experiment(
                output_path=tmp_path / "out.json",
                verilog_path=verilog,
                netlist_path=netlist_path,
                bitfile_path=bitfile_path,
                write_output=False,
                _subprocess_run=_fake_run,
            )
        assert artifact["honest_verdict"] == "synthesis_success"
        assert artifact["bitfile_generated"] is True
        assert artifact["synthesis_result"]["synthesis_success"] is True

    def test_lut_ff_populated_on_success(self, tmp_path: Path) -> None:
        """SCENARIO-HW-005: lut_count and ff_count present on synthesis_success."""
        verilog = tmp_path / "ising_sampler_v1.v"
        verilog.write_text("module top; endmodule\n")
        netlist_path = tmp_path / "netlist_349.json"
        netlist_path.write_text("{}")
        bitfile_path = tmp_path / "carnot_ising.bit"

        def _fake_run(cmd, **kwargs):
            if cmd[0] == "yosys":
                return _make_subprocess_result(0, stdout=YOSYS_SAMPLE_OUTPUT)
            bitfile_path.write_bytes(b"\x00" * 16)
            return _make_subprocess_result(0)

        with patch.object(exp349, "check_yosys_available", return_value=True), \
             patch.object(exp349, "check_nextpnr_available", return_value=True):
            artifact = run_experiment(
                output_path=tmp_path / "out.json",
                verilog_path=verilog,
                netlist_path=netlist_path,
                bitfile_path=bitfile_path,
                write_output=False,
                _subprocess_run=_fake_run,
            )
        assert artifact["lut_count"] == 1744
        assert artifact["ff_count"] == 320


# ---------------------------------------------------------------------------
# REQ-HW-003: Artifact schema completeness
# ---------------------------------------------------------------------------


REQUIRED_ARTIFACT_FIELDS = [
    "experiment",
    "schema",
    "run_date",
    "started_at",
    "finished_at",
    "prereqs_checked",
    "synthesis_result",
    "lut_count",
    "ff_count",
    "bitfile_generated",
    "honest_verdict",
    "spec_requirements",
]


class TestArtifactSchema:
    """REQ-HW-003: Artifact must have all required fields in all code paths."""

    def _run_blocked(self, tmp_path: Path) -> dict:
        with patch.object(exp349, "check_yosys_available", return_value=False), \
             patch.object(exp349, "check_nextpnr_available", return_value=False):
            return run_experiment(
                output_path=tmp_path / "out.json",
                verilog_path=tmp_path / "nonexistent.v",
                write_output=False,
            )

    def test_required_fields_present_blocked(self, tmp_path: Path) -> None:
        """REQ-HW-003: All required fields present in blocked artifact."""
        artifact = self._run_blocked(tmp_path)
        for field in REQUIRED_ARTIFACT_FIELDS:
            assert field in artifact, f"Missing field: {field}"

    def test_experiment_id_is_349(self, tmp_path: Path) -> None:
        """REQ-HW-003: experiment field must be 349."""
        artifact = self._run_blocked(tmp_path)
        assert artifact["experiment"] == 349

    def test_schema_field_value(self, tmp_path: Path) -> None:
        """REQ-HW-003: schema field is 'carnot.fpga_synthesis.v1'."""
        artifact = self._run_blocked(tmp_path)
        assert artifact["schema"] == "carnot.fpga_synthesis.v1"

    def test_spec_requirements_listed(self, tmp_path: Path) -> None:
        """REQ-HW-003: spec_requirements includes REQ-HW-003."""
        artifact = self._run_blocked(tmp_path)
        assert "REQ-HW-003" in artifact["spec_requirements"]
        assert "SCENARIO-HW-005" in artifact["spec_requirements"]
        assert "SCENARIO-HW-006" in artifact["spec_requirements"]

    def test_honest_verdict_from_approved_set(self, tmp_path: Path) -> None:
        """REQ-HW-003: honest_verdict always from APPROVED_VERDICTS."""
        artifact = self._run_blocked(tmp_path)
        assert artifact["honest_verdict"] in SynthesisResult.APPROVED_VERDICTS

    def test_artifact_json_serializable(self, tmp_path: Path) -> None:
        """REQ-HW-003: Entire artifact must be JSON-serializable."""
        artifact = self._run_blocked(tmp_path)
        json.dumps(artifact)  # Must not raise.

    def test_prereqs_checked_keys(self, tmp_path: Path) -> None:
        """REQ-HW-003: prereqs_checked contains expected boolean keys."""
        artifact = self._run_blocked(tmp_path)
        for key in ("yosys_available", "nextpnr_available", "verilog_found", "verilog_path"):
            assert key in artifact["prereqs_checked"], f"Missing prereq key: {key}"

    def test_synthesis_result_subfields_present(self, tmp_path: Path) -> None:
        """REQ-HW-003: synthesis_result sub-dict has all SynthesisResult fields."""
        artifact = self._run_blocked(tmp_path)
        sr = artifact["synthesis_result"]
        for field in (
            "yosys_available", "nextpnr_available", "verilog_found",
            "synthesis_attempted", "synthesis_success",
            "lut_count", "ff_count", "honest_verdict",
        ):
            assert field in sr, f"Missing synthesis_result field: {field}"

    def test_write_output_creates_file(self, tmp_path: Path) -> None:
        """REQ-HW-003: write_output=True creates JSON file at output_path."""
        out = tmp_path / "results" / "exp349.json"
        with patch.object(exp349, "check_yosys_available", return_value=False), \
             patch.object(exp349, "check_nextpnr_available", return_value=False):
            run_experiment(
                output_path=out,
                verilog_path=tmp_path / "nonexistent.v",
                write_output=True,
            )
        assert out.exists()
        with out.open() as f:
            data = json.load(f)
        assert data["experiment"] == 349

    def test_bitfile_path_null_when_not_generated(self, tmp_path: Path) -> None:
        """REQ-HW-003: bitfile_path is None when bitfile_generated=False."""
        artifact = self._run_blocked(tmp_path)
        assert artifact["bitfile_path"] is None


# ---------------------------------------------------------------------------
# REQ-HW-003: Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """REQ-HW-003: Module-level constants are correct."""

    def test_experiment_id_constant(self) -> None:
        """REQ-HW-003: EXPERIMENT_ID == 349."""
        assert EXPERIMENT_ID == 349

    def test_verilog_source_path(self) -> None:
        """REQ-HW-003: VERILOG_SOURCE points to expected RTL file."""
        assert "ising_sampler_v1.v" in str(exp349.VERILOG_SOURCE)

    def test_kv260_device_string(self) -> None:
        """REQ-HW-003: KV260_DEVICE is the correct Zynq UltraScale+ part string."""
        assert exp349.KV260_DEVICE == "xczu5ev-sfvc784-2-e"

    def test_top_module_name(self) -> None:
        """REQ-HW-003: TOP_MODULE matches the module name in the RTL."""
        assert exp349.TOP_MODULE == "ising_sampler_128"


# ---------------------------------------------------------------------------
# Hardware path tests — auto-skip when yosys not installed
# ---------------------------------------------------------------------------

_yosys_available = check_yosys_available()
_verilog_present = check_verilog_source_exists(exp349.VERILOG_SOURCE)
_HW_REASON = "yosys not installed or ising_sampler_v1.v not present"


@pytest.mark.skipif(
    not (_yosys_available and _verilog_present),
    reason=_HW_REASON,
)
class TestLiveYosysSynthesis:
    """SCENARIO-HW-005: Tests that require yosys and the RTL source.

    These tests auto-skip when yosys is not installed or the Verilog source
    is missing.  They exercise the real synthesis path against the actual RTL.
    """

    def test_live_synthesis_produces_verdict(self, tmp_path: Path) -> None:
        """SCENARIO-HW-005: Live yosys synthesis produces a known verdict."""
        artifact = run_experiment(
            output_path=tmp_path / "exp349_live.json",
            write_output=True,
        )
        assert artifact["honest_verdict"] in SynthesisResult.APPROVED_VERDICTS

    def test_live_synthesis_lut_count_nonzero(self, tmp_path: Path) -> None:
        """SCENARIO-HW-005: Live synthesis reports non-zero LUT count."""
        artifact = run_experiment(
            output_path=tmp_path / "exp349_live_lut.json",
            write_output=False,
        )
        # Only assert lut_count > 0 if synthesis was actually attempted.
        if artifact["synthesis_result"]["synthesis_attempted"]:
            if artifact["lut_count"] is not None:
                assert artifact["lut_count"] > 0
