"""Tests for Exp 794: FPGA Toolchain Install and minimal Ising synthesis.

Spec traces: REQ-HW-032, REQ-HW-033, REQ-HW-034, SCENARIO-HW-032

These tests cover the logic added in scripts/experiment_794_fpga_toolchain_install.py.
All subprocess calls are mocked so the suite runs without actual FPGA tools installed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers — we import the module-level functions directly so we can
# unit-test them in isolation without running main().
# ---------------------------------------------------------------------------

import sys
import os

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from scripts.experiment_794_fpga_toolchain_install import (
    _ISING2_VERILOG,
    attempt_pacman_install,
    check_tools,
    classify_verdict,
    run_synthesis,
    _check_tool,
)


# ---------------------------------------------------------------------------
# _check_tool
# ---------------------------------------------------------------------------


class TestCheckTool:
    """REQ-HW-032-1, REQ-HW-033-1, REQ-HW-034-1: tool presence detection."""

    def test_tool_present_exits_zero(self):
        """Tool that exits 0 with output is reported as present."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "yosys 0.38\n"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            present, ver = _check_tool(["yosys", "--version"])
        assert present is True
        assert "yosys" in ver

    def test_tool_missing_file_not_found(self):
        """FileNotFoundError from subprocess signals tool is absent."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            present, ver = _check_tool(["yosys", "--version"])
        assert present is False
        assert "not found" in ver

    def test_tool_present_exits_nonzero_with_output(self):
        """icepack --help exits 1 but produces output — still treated as present (REQ-HW-034-1)."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Usage: icepack [input] [output]\n"
        with patch("subprocess.run", return_value=mock_result):
            present, ver = _check_tool(["icepack", "--help"])
        assert present is True

    def test_tool_timeout_returns_not_present(self):
        """TimeoutExpired is reported as absent, not as an error exception."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["tool"], 10)):
            present, ver = _check_tool(["tool", "--version"])
        assert present is False
        assert "timeout" in ver


# ---------------------------------------------------------------------------
# check_tools
# ---------------------------------------------------------------------------


class TestCheckTools:
    """REQ-HW-032, REQ-HW-033, REQ-HW-034: all three tools checked together."""

    def test_all_tools_present(self):
        """When all three binaries exist, all entries have present=True."""
        ok = MagicMock(returncode=0, stdout="version 1.0", stderr="")
        with patch("subprocess.run", return_value=ok):
            result = check_tools()
        assert result["yosys"]["present"] is True
        assert result["nextpnr-ice40"]["present"] is True
        assert result["icepack"]["present"] is True

    def test_all_tools_missing(self):
        """When no binary exists, all entries have present=False."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = check_tools()
        assert result["yosys"]["present"] is False
        assert result["nextpnr-ice40"]["present"] is False
        assert result["icepack"]["present"] is False

    def test_partial_presence(self):
        """yosys present but nextpnr/icepack absent — partial state is correctly captured."""
        def side_effect(cmd, **kwargs):
            if "yosys" in cmd:
                m = MagicMock(returncode=0, stdout="yosys 0.38", stderr="")
                return m
            raise FileNotFoundError

        with patch("subprocess.run", side_effect=side_effect):
            result = check_tools()
        assert result["yosys"]["present"] is True
        assert result["nextpnr-ice40"]["present"] is False
        assert result["icepack"]["present"] is False


# ---------------------------------------------------------------------------
# attempt_pacman_install
# ---------------------------------------------------------------------------


class TestAttemptPacmanInstall:
    """REQ-HW-032-2, REQ-HW-033-2, REQ-HW-034-2: install attempt via pacman."""

    def test_install_succeeds(self):
        """pacman exits 0 → returns (True, output)."""
        mock_result = MagicMock(returncode=0, stdout=":: Resolving dependencies...\n", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            success, output = attempt_pacman_install()
        assert success is True
        assert output != ""

    def test_install_fails_nonzero(self):
        """pacman exits non-zero → returns (False, output)."""
        mock_result = MagicMock(returncode=1, stdout="", stderr="error: package not found\n")
        with patch("subprocess.run", return_value=mock_result):
            success, output = attempt_pacman_install()
        assert success is False

    def test_sudo_not_found(self):
        """If sudo is absent, returns (False, message) without raising (REQ-HW-032-2)."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            success, output = attempt_pacman_install()
        assert success is False
        assert "sudo not found" in output

    def test_pacman_timeout(self):
        """Pacman timeout is handled gracefully, returning False."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(["pacman"], 300),
        ):
            success, output = attempt_pacman_install()
        assert success is False
        assert "timed out" in output


# ---------------------------------------------------------------------------
# run_synthesis
# ---------------------------------------------------------------------------


class TestRunSynthesis:
    """REQ-HW-032, SCENARIO-HW-032: minimal synthesis execution and result parsing."""

    def test_synthesis_clean(self, tmp_path):
        """yosys exits 0 without ERROR in stderr → success=True (SCENARIO-HW-032)."""
        netlist = {
            "modules": {
                "test_ising2": {
                    "cells": {
                        "c1": {"type": "SB_LUT4"},
                        "c2": {"type": "SB_LUT4"},
                    }
                }
            }
        }
        netlist_json = json.dumps(netlist)

        mock_result = MagicMock(returncode=0, stdout="End of script.\n", stderr="")

        def fake_run(cmd, **kwargs):
            # The -p argument is "synth_ice40; write_json /path/to/file.json"
            # Extract the JSON path from the script argument.
            for arg in cmd:
                if "write_json" in arg:
                    json_out = arg.split("write_json")[-1].strip()
                    Path(json_out).write_text(netlist_json)
            return mock_result

        with patch("subprocess.run", side_effect=fake_run):
            result = run_synthesis(_ISING2_VERILOG)

        assert result["success"] is True
        assert result["lut_count"] == 2

    def test_synthesis_error_in_stderr(self):
        """ERROR keyword in stderr → success=False."""
        mock_result = MagicMock(
            returncode=0,
            stdout="",
            stderr="ERROR: syntax error in module",
        )
        with patch("subprocess.run", return_value=mock_result):
            result = run_synthesis(_ISING2_VERILOG)
        assert result["success"] is False
        assert result["lut_count"] is None

    def test_synthesis_nonzero_returncode(self):
        """Non-zero exit from yosys → success=False."""
        mock_result = MagicMock(returncode=1, stdout="", stderr="fatal error")
        with patch("subprocess.run", return_value=mock_result):
            result = run_synthesis(_ISING2_VERILOG)
        assert result["success"] is False

    def test_synthesis_timeout(self):
        """yosys timeout is handled gracefully → success=False."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(["yosys"], 120),
        ):
            result = run_synthesis(_ISING2_VERILOG)
        assert result["success"] is False
        assert "timed out" in result["stderr_snippet"]

    def test_synthesis_clean_no_lut4_cells(self, tmp_path):
        """Synthesis success but netlist has no SB_LUT4 cells → lut_count=0."""
        netlist = {"modules": {"test_ising2": {"cells": {}}}}
        netlist_json = json.dumps(netlist)

        mock_result = MagicMock(returncode=0, stdout="End.\n", stderr="")

        def fake_run(cmd, **kwargs):
            for arg in cmd:
                if "write_json" in arg:
                    json_out = arg.split("write_json")[-1].strip()
                    Path(json_out).write_text(netlist_json)
            return mock_result

        with patch("subprocess.run", side_effect=fake_run):
            result = run_synthesis(_ISING2_VERILOG)

        assert result["success"] is True
        assert result["lut_count"] == 0


# ---------------------------------------------------------------------------
# classify_verdict
# ---------------------------------------------------------------------------


class TestClassifyVerdict:
    """REQ-HW-043-4 vocabulary extended: all five verdict strings are reachable."""

    def _present_tools(self) -> dict:
        return {
            "yosys": {"present": True, "version": "0.38"},
            "nextpnr-ice40": {"present": True, "version": "0.7"},
            "icepack": {"present": True, "version": ""},
        }

    def _absent_tools(self) -> dict:
        return {
            "yosys": {"present": False, "version": "not found"},
            "nextpnr-ice40": {"present": False, "version": "not found"},
            "icepack": {"present": False, "version": "not found"},
        }

    def test_all_present_synth_clean(self):
        """SCENARIO-HW-032: all tools present + clean synth → tools_installed_synthesis_clean."""
        synth = {"success": True, "lut_count": 2, "stderr_snippet": "", "stdout_snippet": ""}
        verdict = classify_verdict(self._present_tools(), False, False, synth)
        assert verdict == "tools_installed_synthesis_clean"

    def test_all_present_synth_failed(self):
        synth = {"success": False, "lut_count": None, "stderr_snippet": "ERROR", "stdout_snippet": ""}
        verdict = classify_verdict(self._present_tools(), False, False, synth)
        assert verdict == "tools_installed_synthesis_failed"

    def test_all_present_synth_skipped(self):
        verdict = classify_verdict(self._present_tools(), False, False, None)
        assert verdict == "tools_installed_synthesis_skipped"

    def test_missing_install_attempted(self):
        verdict = classify_verdict(self._absent_tools(), True, False, None)
        assert verdict == "tools_not_installed_install_attempted"

    def test_missing_install_skipped(self):
        """No sudo → install not attempted → tools_not_installed_install_skipped."""
        verdict = classify_verdict(self._absent_tools(), False, False, None)
        assert verdict == "tools_not_installed_install_skipped"
