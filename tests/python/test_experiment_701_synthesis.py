"""Tests for experiment 701 — KV260 Ising v3 synthesis.

Covers all public functions in scripts/experiment_701_kv260_synthesis.py
with enough cases to achieve 100% branch coverage on the added code.

Spec: REQ-HW-037, REQ-HW-038,
      SCENARIO-HW-037, SCENARIO-HW-038, SCENARIO-HW-039
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_701_kv260_synthesis import (
    DELIVERABLE,
    EXP_ID,
    KV260_TOTAL_LUTS,
    LUT_BUDGET,
    LUT_BUDGET_PCT,
    PART_NUMBER,
    SCHEMA,
    TIMING_TARGET_MHZ,
    append_known_issue,
    check_vivado,
    check_yosys,
    compute_honest_verdict,
    parse_timing_report,
    parse_utilization_report,
    parse_yosys_stat,
    write_vivado_tcl,
    write_yosys_script,
)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_lut_budget_is_20_pct() -> None:
    """REQ-HW-038: LUT budget must be 20% of KV260 fabric (23,424 LUTs)."""
    assert LUT_BUDGET == int(KV260_TOTAL_LUTS * LUT_BUDGET_PCT / 100)
    assert LUT_BUDGET == 23_424


def test_timing_target_mhz() -> None:
    """REQ-HW-037: timing target must be 50 MHz."""
    assert TIMING_TARGET_MHZ == 50


def test_part_number_kv260() -> None:
    """REQ-HW-037: synthesis must target the correct KV260 device part."""
    assert PART_NUMBER == "xck26-sfvc784-2LV-c"


# ---------------------------------------------------------------------------
# check_vivado
# ---------------------------------------------------------------------------


class TestCheckVivado:
    """SCENARIO-HW-037: Vivado availability check."""

    def test_vivado_present(self) -> None:
        """Returns True when vivado exits with returncode 0."""
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            assert check_vivado() is True
        mock_run.assert_called_once()

    def test_vivado_nonzero_rc(self) -> None:
        """Returns False when vivado exits with nonzero returncode."""
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            assert check_vivado() is False

    def test_vivado_not_found(self) -> None:
        """Returns False when vivado is not on PATH (FileNotFoundError)."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert check_vivado() is False

    def test_vivado_timeout(self) -> None:
        """Returns False when vivado times out."""
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("vivado", 30)):
            assert check_vivado() is False


# ---------------------------------------------------------------------------
# check_yosys
# ---------------------------------------------------------------------------


class TestCheckYosys:
    """SCENARIO-HW-038: yosys availability check."""

    def test_yosys_present(self) -> None:
        """Returns True when yosys exits with returncode 0."""
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            assert check_yosys() is True

    def test_yosys_nonzero_rc(self) -> None:
        """Returns False when yosys exits with nonzero returncode."""
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            assert check_yosys() is False

    def test_yosys_not_found(self) -> None:
        """Returns False when yosys is not on PATH."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert check_yosys() is False

    def test_yosys_timeout(self) -> None:
        """Returns False when yosys times out."""
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("yosys", 30)):
            assert check_yosys() is False


# ---------------------------------------------------------------------------
# write_vivado_tcl
# ---------------------------------------------------------------------------


def test_write_vivado_tcl(tmp_path: Path) -> None:
    """TCL script must reference the part number and RTL path."""
    tcl_path = tmp_path / "tcl" / "synth_ising_v3.tcl"
    write_vivado_tcl(tcl_path)
    assert tcl_path.exists()
    content = tcl_path.read_text()
    assert PART_NUMBER in content
    assert "ising_sampler_v3" in content
    assert "report_utilization" in content
    assert "report_timing_summary" in content


# ---------------------------------------------------------------------------
# write_yosys_script
# ---------------------------------------------------------------------------


def test_write_yosys_script(tmp_path: Path) -> None:
    """yosys script must contain synth command and stat."""
    ys_path = tmp_path / "synth.ys"
    write_yosys_script(ys_path)
    assert ys_path.exists()
    content = ys_path.read_text()
    assert "synth" in content
    assert "ising_sampler_v3" in content
    assert "stat" in content


# ---------------------------------------------------------------------------
# parse_utilization_report
# ---------------------------------------------------------------------------


class TestParseUtilizationReport:
    """SCENARIO-HW-037, SCENARIO-HW-038: parse Vivado utilization report."""

    def test_file_missing_returns_nones(self, tmp_path: Path) -> None:
        """Missing report file returns all None values."""
        result = parse_utilization_report(tmp_path / "nonexistent.rpt")
        assert result == {"LUT_count": None, "FF_count": None, "BRAM_count": None}

    def test_parses_lut_count(self, tmp_path: Path) -> None:
        """Extracts LUT as Logic count from report text.

        REQ-HW-038: LUT count must be < 23,424 to pass the 20% budget.
        """
        report = tmp_path / "util.rpt"
        report.write_text(
            "| LUT as Logic             |  12345 |  117120 |  10.55 |\n"
            "| Register as Flip Flop    |   8000 |  234240 |   3.41 |\n"
            "| Block RAM Tile           |      2 |     144 |   1.39 |\n"
        )
        result = parse_utilization_report(report)
        assert result["LUT_count"] == 12345
        assert result["FF_count"] == 8000
        assert result["BRAM_count"] == 2

    def test_parses_comma_formatted_numbers(self, tmp_path: Path) -> None:
        """Handles comma-separated numbers (Vivado uses these for large counts)."""
        report = tmp_path / "util.rpt"
        report.write_text(
            "| LUT as Logic             |  23,000 |  117,120 |  19.64 |\n"
        )
        result = parse_utilization_report(report)
        assert result["LUT_count"] == 23000

    def test_missing_fields_return_none(self, tmp_path: Path) -> None:
        """Fields not present in report return None without crashing."""
        report = tmp_path / "util.rpt"
        report.write_text("Some report with no utilization data.")
        result = parse_utilization_report(report)
        assert result["LUT_count"] is None
        assert result["FF_count"] is None
        assert result["BRAM_count"] is None


# ---------------------------------------------------------------------------
# parse_timing_report
# ---------------------------------------------------------------------------


class TestParseTimingReport:
    """SCENARIO-HW-037: parse Vivado timing summary report."""

    def test_file_missing_returns_nones(self, tmp_path: Path) -> None:
        """Missing report file returns (None, None)."""
        wns, met = parse_timing_report(tmp_path / "nonexistent.rpt")
        assert wns is None
        assert met is None

    def test_positive_wns_timing_met(self, tmp_path: Path) -> None:
        """WNS >= 0 means timing closed — REQ-HW-037 satisfied."""
        report = tmp_path / "timing.rpt"
        report.write_text(
            "WNS(ns)  TNS(ns)  TNS Failing Endpoints\n"
            "  2.345  0.000    0\n"
        )
        wns, met = parse_timing_report(report)
        assert wns == pytest.approx(2.345)
        assert met is True

    def test_negative_wns_timing_failed(self, tmp_path: Path) -> None:
        """WNS < 0 means timing violation — REQ-HW-037 not satisfied."""
        report = tmp_path / "timing.rpt"
        report.write_text(
            "WNS(ns)  TNS(ns)  TNS Failing Endpoints\n"
            " -1.234  -5.678   3\n"
        )
        wns, met = parse_timing_report(report)
        assert wns == pytest.approx(-1.234)
        assert met is False

    def test_zero_wns_timing_met(self, tmp_path: Path) -> None:
        """WNS == 0 exactly counts as timing met (boundary case)."""
        report = tmp_path / "timing.rpt"
        report.write_text(
            "WNS(ns)  TNS(ns)  TNS Failing Endpoints\n"
            "  0.000  0.000    0\n"
        )
        wns, met = parse_timing_report(report)
        assert wns == pytest.approx(0.0)
        assert met is True

    def test_no_matching_lines_returns_nones(self, tmp_path: Path) -> None:
        """Report with no parseable timing data returns (None, None)."""
        report = tmp_path / "timing.rpt"
        report.write_text("Timing report with no data.")
        wns, met = parse_timing_report(report)
        assert wns is None
        assert met is None


# ---------------------------------------------------------------------------
# parse_yosys_stat
# ---------------------------------------------------------------------------


class TestParseYosynStat:
    """SCENARIO-HW-038: parse yosys stat output for LUT proxy estimate."""

    def test_parses_number_of_cells(self) -> None:
        """Primary path: 'Number of cells: N' is extracted correctly."""
        stdout = (
            "=== ising_sampler_v3 ===\n"
            "\n"
            "   Number of wires:              1024\n"
            "   Number of cells:              4567\n"
            "   $_DFF_P_                        56\n"
        )
        assert parse_yosys_stat(stdout) == 4567

    def test_fallback_sums_cell_types(self) -> None:
        """Fallback path: sums all $_<type>_ counts when Number of cells absent."""
        stdout = (
            "=== ising_sampler_v3 ===\n"
            "   $_DFF_P_                        56\n"
            "   $_MUX_                         100\n"
            "   $_NOT_                          44\n"
        )
        result = parse_yosys_stat(stdout)
        assert result == 200

    def test_empty_stdout_returns_none(self) -> None:
        """Empty output means synthesis produced no stat — returns None."""
        assert parse_yosys_stat("") is None

    def test_comma_formatted_cell_count(self) -> None:
        """Handles comma-separated large cell counts (e.g. 1,234)."""
        stdout = "   Number of cells:              1,234\n"
        assert parse_yosys_stat(stdout) == 1234


# ---------------------------------------------------------------------------
# compute_honest_verdict
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    """All honest_verdict branches."""

    def test_vivado_timing_met(self) -> None:
        """REQ-HW-037 satisfied: Vivado + WNS >= 0."""
        assert compute_honest_verdict("vivado", True, 10000) == "synthesis_timing_met"

    def test_vivado_timing_failed(self) -> None:
        """REQ-HW-037 not met: Vivado + WNS < 0."""
        assert compute_honest_verdict("vivado", False, 10000) == "synthesis_timing_failed"

    def test_vivado_timing_none(self) -> None:
        """Vivado ran but timing parse failed — treated as failed."""
        assert compute_honest_verdict("vivado", None, 10000) == "synthesis_timing_failed"

    def test_yosys_lut_estimate(self) -> None:
        """REQ-HW-038 proxy: yosys + cell count available."""
        assert compute_honest_verdict("yosys", None, 5000) == "synthesis_lut_estimate_only"

    def test_yosys_parse_failed(self) -> None:
        """yosys ran but stat parse failed."""
        assert (
            compute_honest_verdict("yosys", None, None)
            == "synthesis_blocked_yosys_parse_failed"
        )

    def test_no_tool(self) -> None:
        """SCENARIO-HW-039: no synthesis tool available."""
        assert (
            compute_honest_verdict("none_available", None, None)
            == "synthesis_blocked_no_tool"
        )


# ---------------------------------------------------------------------------
# append_known_issue
# ---------------------------------------------------------------------------


def test_append_known_issue_appends(tmp_path: Path) -> None:
    """Note is appended when not already present."""
    ki_path = tmp_path / "known-issues.md"
    ki_path.write_text("## Existing content\n")
    with patch(
        "scripts.experiment_701_kv260_synthesis.KNOWN_ISSUES_PATH", ki_path
    ):
        append_known_issue("## RETRO-072: test note")
    assert "RETRO-072: test note" in ki_path.read_text()


def test_append_known_issue_no_duplicate(tmp_path: Path) -> None:
    """Note is not appended a second time if already present."""
    ki_path = tmp_path / "known-issues.md"
    note = "## RETRO-072: test note"
    ki_path.write_text(f"{note}\n")
    with patch(
        "scripts.experiment_701_kv260_synthesis.KNOWN_ISSUES_PATH", ki_path
    ):
        append_known_issue(note)
    assert ki_path.read_text().count(note) == 1


def test_append_known_issue_file_missing(tmp_path: Path) -> None:
    """Silently does nothing when known-issues.md does not exist."""
    missing_path = tmp_path / "nonexistent.md"
    with patch(
        "scripts.experiment_701_kv260_synthesis.KNOWN_ISSUES_PATH", missing_path
    ):
        append_known_issue("## RETRO-072 note")  # must not raise


# ---------------------------------------------------------------------------
# Integration: main() with no synthesis tool (SCENARIO-HW-039)
# ---------------------------------------------------------------------------


def test_main_no_tool_produces_deliverable(tmp_path: Path) -> None:
    """main() produces a valid deliverable when no synthesis tool is available.

    This exercises the full SCENARIO-HW-039 path end-to-end without requiring
    Vivado or yosys to be installed on the CI host.

    REQ-HW-037, REQ-HW-038: even in the blocked case the artifact must be
    written with all required schema fields.
    """
    import scripts.experiment_701_kv260_synthesis as mod

    deliverable_path = tmp_path / "results" / "experiment_701_kv260_ising_v3_synthesis.json"

    with (
        patch.object(mod, "check_vivado", return_value=False),
        patch.object(mod, "check_yosys", return_value=False),
        patch.object(mod, "KNOWN_ISSUES_PATH", tmp_path / "known-issues.md"),
        patch.object(mod, "_REPO_ROOT", tmp_path),
        patch.object(mod, "RESULTS_DIR", tmp_path / "results"),
        patch.object(mod, "RTL_PATH", tmp_path / "rtl" / "ising_sampler_v3.v"),
    ):
        # The known-issues file must exist for append_known_issue to write.
        (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
        (tmp_path / "known-issues.md").write_text("")

        # Patch the deliverable path in ExperimentTemplate
        with patch("scripts.experiment_template._get_repo_root", return_value=tmp_path):
            (tmp_path / "results").mkdir(parents=True, exist_ok=True)
            mod.main()

    assert deliverable_path.exists(), "Deliverable JSON must be written"
    artifact = json.loads(deliverable_path.read_text())
    assert artifact["honest_verdict"] == "synthesis_blocked_no_tool"
    assert artifact["retro_072_resolved"] is False
    assert artifact["status"] == "blocked"
    assert artifact["experiment"] == EXP_ID
    # REQUIRED_RESULT_FIELDS check
    for field in ("experiment", "schema", "run_date", "started_at",
                  "finished_at", "duration_s", "status", "title"):
        assert field in artifact, f"Missing required field: {field}"
