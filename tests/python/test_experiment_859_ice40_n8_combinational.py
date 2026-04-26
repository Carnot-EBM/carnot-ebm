"""Tests for scripts/experiment_859_ice40_n8_combinational.py.

These tests mock all subprocess calls so they run in CI without OSS-CAD-Suite
installed.  They verify:
    - LUT count parsing from yosys and nextpnr output strings.
    - Sequential-logic detection from yosys output.
    - honest_verdict selection logic.
    - Bitstream-generated check using a temporary file.
    - The main() path for a successful synthesis run end-to-end.
    - The main() path for synthesis failure (returncode != 0).
    - The main() path for P&R failure.

Spec: REQ-FPGA-030, SCENARIO-FPGA-040
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.  We need its private helpers.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.experiment_859_ice40_n8_combinational as exp859


# ---------------------------------------------------------------------------
# Unit tests for LUT-count parsers
# ---------------------------------------------------------------------------


class TestParseSynthLutCount:
    """Tests for _parse_synth_lut_count — extracts SB_LUT4 count from yosys stdout.
    Spec: REQ-FPGA-030"""

    def test_typical_output(self) -> None:
        """Parses the standard yosys cell-statistics format."""
        output = (
            "Chip area for module '\\ising_energy_n8_comb':\n"
            "     1234   SB_LUT4\n"
            "       45   SB_CARRY\n"
        )
        assert exp859._parse_synth_lut_count(output) == 1234

    def test_exact_match_from_exp859(self) -> None:
        """Parses the actual LUT count produced by Exp 859 (132)."""
        output = "      132   SB_LUT4\n        2   SB_CARRY\n"
        assert exp859._parse_synth_lut_count(output) == 132

    def test_missing_lut_line_returns_zero(self) -> None:
        """Returns 0 when the SB_LUT4 line is absent (synthesis failed)."""
        assert exp859._parse_synth_lut_count("Error: something went wrong\n") == 0

    def test_zero_luts(self) -> None:
        """Handles degenerate case of empty design reporting 0 LUTs."""
        assert exp859._parse_synth_lut_count("        0   SB_LUT4\n") == 0


class TestParsePnrLutCount:
    """Tests for _parse_pnr_lut_count — extracts ICESTORM_LC count from nextpnr output.
    Spec: REQ-FPGA-030"""

    def test_typical_nextpnr_output(self) -> None:
        """Parses the standard nextpnr utilisation table format."""
        pnr_output = (
            "Info: Device utilisation:\n"
            "Info: \t         ICESTORM_LC:     134/   7680     1%\n"
            "Info: \t        ICESTORM_RAM:       0/     32     0%\n"
        )
        assert exp859._parse_pnr_lut_count(pnr_output) == 134

    def test_large_count(self) -> None:
        """Parses a large LC count (reproduces Exp 851 failure scenario)."""
        pnr_output = "Info: \t         ICESTORM_LC:   12258/   7680   159%\n"
        assert exp859._parse_pnr_lut_count(pnr_output) == 12258

    def test_missing_line_returns_zero(self) -> None:
        """Returns 0 when the utilisation line is absent (P&R failed)."""
        assert exp859._parse_pnr_lut_count("ERROR: placement failed\n") == 0


class TestHasSequentialLogic:
    """Tests for _has_sequential_logic — detects DFF instances in yosys output.
    Spec: REQ-FPGA-030"""

    def test_no_dff_present(self) -> None:
        """Returns False when no SB_DFF count line appears (pure combinational)."""
        output = "      132   SB_LUT4\n        2   SB_CARRY\n"
        assert exp859._has_sequential_logic(output) is False

    def test_nonzero_dff_detected(self) -> None:
        """Returns True when yosys reports one or more DFF cells."""
        output = "      132   SB_LUT4\n       16   SB_DFF\n"
        assert exp859._has_sequential_logic(output) is True

    def test_zero_dff_line_treated_as_combinational(self) -> None:
        """Returns False when the DFF count line is explicitly zero."""
        output = "      132   SB_LUT4\n        0   SB_DFF\n"
        assert exp859._has_sequential_logic(output) is False

    def test_module_definition_line_not_counted(self) -> None:
        """Does not false-positive on 'SB_DFF' in the cell model header lines.

        The yosys log includes lines like 'Generating RTLIL representation for
        module SB_DFF' before the stats table.  These are not cell instances.
        Only numeric count lines of the form '<digits>   SB_DFF' trigger True.
        """
        header_only = (
            "Generating RTLIL representation for module `\\SB_DFF'.\n      132   SB_LUT4\n"
        )
        assert exp859._has_sequential_logic(header_only) is False


# ---------------------------------------------------------------------------
# Integration test: main() with mocked subprocess and file I/O
# ---------------------------------------------------------------------------

_SYNTH_STDOUT_SUCCESS = "      132   SB_LUT4\n        2   SB_CARRY\nEnd of script.\n"
_PNR_STDOUT_SUCCESS = (
    "Info: Device utilisation:\n"
    "Info: \t         ICESTORM_LC:     134/   7680     1%\n"
    "24 warnings, 0 errors\n"
    "Info: Program finished normally.\n"
)
_ICEPACK_STDOUT_SUCCESS = ""


def _make_completed_process(returncode: int, stdout: str, stderr: str = "") -> MagicMock:
    """Build a mock CompletedProcess-like object returned by subprocess.run."""
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


class TestMainSuccessPath:
    """Tests for main() when yosys, nextpnr, and icepack all succeed.
    Spec: SCENARIO-FPGA-040"""

    def test_fpga_oracle_ready_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() writes artifact with honest_verdict=fpga_oracle_ready on full success."""
        # Run the experiment with tmp_path as the working directory so the
        # "results/..." relative path resolves inside the temp directory.
        (tmp_path / "results").mkdir()
        monkeypatch.chdir(tmp_path)

        fake_bin = tmp_path / "ising_energy_n8_comb.bin"

        def fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
            cmd_str = cmd[0] if cmd else ""
            if "yosys" in cmd_str:
                return _make_completed_process(0, _SYNTH_STDOUT_SUCCESS)
            elif "nextpnr" in cmd_str:
                return _make_completed_process(0, _PNR_STDOUT_SUCCESS)
            elif "icepack" in cmd_str:
                fake_bin.write_bytes(b"\xff" * 135100)
                return _make_completed_process(0, _ICEPACK_STDOUT_SUCCESS)
            return _make_completed_process(1, "", "unknown command")

        monkeypatch.setattr(exp859, "BIN_OUT", fake_bin)
        with patch("subprocess.run", side_effect=fake_run):
            with patch(
                "scripts.experiment_template.ExperimentTemplate.assert_deliverable_written"
            ) as mock_assert:
                mock_assert.return_value = None
                exp859.main()

        out_file = tmp_path / "results" / "experiment_859_ice40_n8_combinational.json"
        assert out_file.exists(), "Artifact JSON was not written"
        artifact = json.loads(out_file.read_text())
        assert artifact["honest_verdict"] == "fpga_oracle_ready"
        assert artifact["bitstream_generated"] is True
        assert artifact["lut_count"] == 134
        assert artifact["sequential_logic_present"] is False
        assert artifact["synthesis_lut_count"] == 132


class TestMainSynthesisFailure:
    """Tests for main() when yosys returns non-zero exit code.
    Spec: REQ-FPGA-030"""

    def test_synthesis_failed_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() writes synthesis_failed artifact when yosys exits non-zero."""
        (tmp_path / "results").mkdir()
        monkeypatch.chdir(tmp_path)

        def fake_run_fail(cmd: list[str], **_kwargs: object) -> MagicMock:
            return _make_completed_process(1, "", "ERROR: synthesis failed")

        with patch("subprocess.run", side_effect=fake_run_fail):
            with patch(
                "scripts.experiment_template.ExperimentTemplate.assert_deliverable_written"
            ) as mock_assert:
                mock_assert.return_value = None
                exp859.main()

        out_file = tmp_path / "results" / "experiment_859_ice40_n8_combinational.json"
        assert out_file.exists()
        artifact = json.loads(out_file.read_text())
        assert artifact["honest_verdict"] == "synthesis_failed"
        assert artifact["bitstream_generated"] is False


class TestMainPnrFailure:
    """Tests for main() when nextpnr returns non-zero (e.g. over-budget design).
    Spec: REQ-FPGA-030"""

    def test_pnr_failed_verdict(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() writes pnr_failed artifact when nextpnr exits non-zero."""
        (tmp_path / "results").mkdir()
        monkeypatch.chdir(tmp_path)

        call_count = {"n": 0}

        def fake_run_pnr_fail(cmd: list[str], **_kwargs: object) -> MagicMock:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_completed_process(0, _SYNTH_STDOUT_SUCCESS)
            pnr_overflow = (
                "Info: Device utilisation:\n"
                "Info: \t         ICESTORM_LC:   12258/   7680   159%\n"
                "ERROR: Failed to expand region\n"
            )
            return _make_completed_process(255, pnr_overflow, "ERROR: Failed to expand region")

        with patch("subprocess.run", side_effect=fake_run_pnr_fail):
            with patch(
                "scripts.experiment_template.ExperimentTemplate.assert_deliverable_written"
            ) as mock_assert:
                mock_assert.return_value = None
                exp859.main()

        out_file = tmp_path / "results" / "experiment_859_ice40_n8_combinational.json"
        assert out_file.exists()
        artifact = json.loads(out_file.read_text())
        assert artifact["honest_verdict"] == "pnr_failed"
        assert artifact["bitstream_generated"] is False
        assert artifact["synthesis_lut_count"] == 132


class TestLutOverBudget:
    """Tests the lut_over_budget verdict branch when P&R succeeds but LC count >= 500."""

    def test_lut_over_budget_verdict(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """honest_verdict = lut_over_budget when pnr_lut_count >= 500."""
        (tmp_path / "results").mkdir()
        monkeypatch.chdir(tmp_path)

        fake_bin = tmp_path / "ising_energy_n8_comb.bin"
        monkeypatch.setattr(exp859, "BIN_OUT", fake_bin)

        bloated_pnr = (
            "Info: Device utilisation:\n"
            "Info: \t         ICESTORM_LC:     600/   7680     7%\n"
            "Info: Program finished normally.\n"
        )
        call_count = {"n": 0}

        def fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_completed_process(0, _SYNTH_STDOUT_SUCCESS)
            elif call_count["n"] == 2:
                return _make_completed_process(0, bloated_pnr)
            fake_bin.write_bytes(b"\xff" * 135100)
            return _make_completed_process(0, "")

        with patch("subprocess.run", side_effect=fake_run):
            with patch(
                "scripts.experiment_template.ExperimentTemplate.assert_deliverable_written"
            ) as mock_assert:
                mock_assert.return_value = None
                exp859.main()

        out_file = tmp_path / "results" / "experiment_859_ice40_n8_combinational.json"
        assert out_file.exists()
        artifact = json.loads(out_file.read_text())
        assert artifact["honest_verdict"] == "lut_over_budget"
        assert artifact["lut_count"] == 600
