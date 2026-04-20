"""Tests for experiment_584_kv260_synthesis.py — targeted 100% coverage of new helpers.

Spec: REQ-SAMPLE-032, SCENARIO-SAMPLE-052, SCENARIO-SAMPLE-053, SCENARIO-SAMPLE-054
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.experiment_584_kv260_synthesis import (
    BITFILE_PATH,
    CPU_BASELINE_LATENCY_US,
    DELIVERABLE,
    EXPERIMENT_ID,
    EXPORT_COMMAND_PATH,
    OUTPUT_DIR,
    TCL_PATH,
    VIVADO_INSTALL_STEPS,
    VERILOG_PATH,
    _TCL_MINIMAL_LINE_THRESHOLD,
    build_vivado_result,
    check_vivado_available,
    determine_honest_verdict,
    run_experiment,
    tcl_is_complete,
)


# ---------------------------------------------------------------------------
# check_vivado_available
# ---------------------------------------------------------------------------


def test_check_vivado_available_true():
    """SCENARIO-SAMPLE-052: vivado found on PATH returns True."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        assert check_vivado_available() is True
        mock_run.assert_called_once_with(["which", "vivado"], capture_output=True, timeout=10)


def test_check_vivado_available_false():
    """SCENARIO-SAMPLE-052: vivado not on PATH returns False."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    with patch("subprocess.run", return_value=mock_result):
        assert check_vivado_available() is False


# ---------------------------------------------------------------------------
# tcl_is_complete
# ---------------------------------------------------------------------------


def test_tcl_is_complete_true(tmp_path):
    """SCENARIO-SAMPLE-052: TCL with write_bitstream and enough lines is complete."""
    tcl = tmp_path / "synth.tcl"
    # Build content with more than _TCL_MINIMAL_LINE_THRESHOLD non-blank lines
    lines = ["# line %d" % i for i in range(_TCL_MINIMAL_LINE_THRESHOLD + 2)]
    lines.append("write_bitstream -force output/carnot_ising.bit")
    tcl.write_text("\n".join(lines))
    assert tcl_is_complete(str(tcl)) is True


def test_tcl_is_complete_missing_write_bitstream(tmp_path):
    """TCL without write_bitstream command is not considered complete."""
    tcl = tmp_path / "synth.tcl"
    lines = ["# line %d" % i for i in range(_TCL_MINIMAL_LINE_THRESHOLD + 2)]
    tcl.write_text("\n".join(lines))
    assert tcl_is_complete(str(tcl)) is False


def test_tcl_is_complete_file_missing():
    """Non-existent TCL file returns False."""
    assert tcl_is_complete("/nonexistent/path/synth.tcl") is False


def test_tcl_is_complete_too_few_lines(tmp_path):
    """TCL with write_bitstream but fewer than threshold lines is not complete."""
    tcl = tmp_path / "synth.tcl"
    tcl.write_text("write_bitstream output.bit\n")
    assert tcl_is_complete(str(tcl)) is False


# ---------------------------------------------------------------------------
# build_vivado_result
# ---------------------------------------------------------------------------


def test_build_vivado_result_success(tmp_path):
    """SCENARIO-SAMPLE-053: Vivado runs and produces bitfile."""
    bitfile = tmp_path / "output" / "carnot_ising.bit"
    bitfile.parent.mkdir(parents=True)
    bitfile.write_bytes(b"\x00")  # Simulate bitfile existing

    mock_proc = MagicMock()
    mock_proc.stdout = "Finished writing bitstream to output/carnot_ising.bit\nSynthesis complete."
    mock_proc.stderr = ""
    mock_proc.returncode = 0

    with patch("subprocess.run", return_value=mock_proc):
        result = build_vivado_result(
            tcl_path="hardware/kv260/synth_ising.tcl",
            output_dir=str(tmp_path / "output"),
            bitfile_path=str(bitfile),
            timeout_s=10,
        )

    assert result["vivado_ran"] is True
    assert result["bitfile_built"] is True
    assert result["vivado_returncode"] == 0
    assert result["timed_out"] is False
    assert "Finished writing bitstream" in result["synthesis_stdout"]


def test_build_vivado_result_failed_returncode(tmp_path):
    """Vivado returns non-zero and bitfile is absent."""
    mock_proc = MagicMock()
    mock_proc.stdout = "ERROR: synthesis failed"
    mock_proc.stderr = "some error"
    mock_proc.returncode = 1

    with patch("subprocess.run", return_value=mock_proc):
        result = build_vivado_result(
            tcl_path="hardware/kv260/synth_ising.tcl",
            output_dir=str(tmp_path / "output"),
            bitfile_path=str(tmp_path / "output" / "x.bit"),
            timeout_s=10,
        )

    assert result["vivado_ran"] is True
    assert result["bitfile_built"] is False
    assert result["vivado_returncode"] == 1


def test_build_vivado_result_timeout(tmp_path):
    """SCENARIO-SAMPLE-053: Vivado times out returns timed_out=True."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("vivado", 10)):
        result = build_vivado_result(
            tcl_path="hardware/kv260/synth_ising.tcl",
            output_dir=str(tmp_path / "output"),
            bitfile_path=str(tmp_path / "output" / "x.bit"),
            timeout_s=10,
        )

    assert result["vivado_ran"] is True
    assert result["timed_out"] is True
    assert result["vivado_returncode"] == -1


def test_build_vivado_result_exception(tmp_path):
    """Unexpected exception from subprocess returns vivado_ran=False."""
    with patch("subprocess.run", side_effect=FileNotFoundError("vivado not found")):
        result = build_vivado_result(
            tcl_path="hardware/kv260/synth_ising.tcl",
            output_dir=str(tmp_path / "output"),
            bitfile_path=str(tmp_path / "output" / "x.bit"),
            timeout_s=10,
        )

    assert result["vivado_ran"] is False
    assert result["vivado_returncode"] == -1


def test_build_vivado_result_stdout_truncated(tmp_path):
    """synthesis_stdout is truncated to 2000 chars."""
    mock_proc = MagicMock()
    mock_proc.stdout = "x" * 5000
    mock_proc.stderr = ""
    mock_proc.returncode = 0

    with patch("subprocess.run", return_value=mock_proc):
        result = build_vivado_result(
            tcl_path="t.tcl",
            output_dir=str(tmp_path / "out"),
            bitfile_path=str(tmp_path / "out" / "x.bit"),
            timeout_s=10,
        )

    assert len(result["synthesis_stdout"]) == 2000


# ---------------------------------------------------------------------------
# determine_honest_verdict
# ---------------------------------------------------------------------------


def test_determine_honest_verdict_bitfile_built():
    """SCENARIO-SAMPLE-053: bitfile_built=True → 'bitfile_built'."""
    assert determine_honest_verdict(True, True, True) == "bitfile_built"


def test_determine_honest_verdict_attempted_failed():
    """SCENARIO-SAMPLE-053: Vivado ran, no bitfile → 'synthesis_attempted_failed'."""
    assert determine_honest_verdict(True, False, True) == "synthesis_attempted_failed"


def test_determine_honest_verdict_not_installed():
    """SCENARIO-SAMPLE-054: Vivado not available → 'vivado_not_installed'."""
    assert determine_honest_verdict(False, False, False) == "vivado_not_installed"


def test_determine_honest_verdict_available_but_not_ran():
    """Vivado available but did not run (pre-check failure) → 'vivado_not_installed'."""
    assert determine_honest_verdict(True, False, False) == "vivado_not_installed"


# ---------------------------------------------------------------------------
# run_experiment integration (mocked I/O)
# ---------------------------------------------------------------------------


def test_run_experiment_vivado_not_installed(tmp_path):
    """SCENARIO-SAMPLE-054: when Vivado is absent the artifact has install steps."""
    # Minimal TCL for tcl_is_complete check
    hw_dir = tmp_path / "hardware" / "kv260"
    hw_dir.mkdir(parents=True)
    lines = ["# line %d" % i for i in range(_TCL_MINIMAL_LINE_THRESHOLD + 2)]
    lines.append("write_bitstream -force output/carnot_ising.bit")
    (hw_dir / "synth_ising.tcl").write_text("\n".join(lines))

    (tmp_path / "results").mkdir(parents=True)

    with patch(
        "scripts.experiment_584_kv260_synthesis.check_vivado_available",
        return_value=False,
    ):
        artifact = run_experiment(repo_root=tmp_path)

    assert artifact["vivado_available"] is False
    assert artifact["bitfile_built"] is False
    assert artifact["bitfile_path"] is None
    assert artifact["honest_verdict"] == "vivado_not_installed"
    assert isinstance(artifact["vivado_install_steps"], list)
    assert len(artifact["vivado_install_steps"]) > 0

    # Deliverable must be written
    assert (tmp_path / DELIVERABLE).exists()
    data = json.loads((tmp_path / DELIVERABLE).read_text())
    assert data["honest_verdict"] == "vivado_not_installed"
    assert data["cpu_baseline_latency_us"] == CPU_BASELINE_LATENCY_US


def test_run_experiment_vivado_available_bitfile_built(tmp_path):
    """SCENARIO-SAMPLE-052/053: Vivado installed and synthesis succeeds."""
    hw_dir = tmp_path / "hardware" / "kv260"
    hw_dir.mkdir(parents=True)
    lines = ["# line %d" % i for i in range(_TCL_MINIMAL_LINE_THRESHOLD + 2)]
    lines.append("write_bitstream -force output/carnot_ising.bit")
    (hw_dir / "synth_ising.tcl").write_text("\n".join(lines))
    (tmp_path / "results").mkdir(parents=True)

    # Pre-create the bitfile so build_vivado_result finds it
    out = tmp_path / OUTPUT_DIR
    out.mkdir(parents=True)
    (out / "carnot_ising.bit").write_bytes(b"\x00")

    fake_synthesis = {
        "vivado_ran": True,
        "synthesis_stdout": "Finished writing bitstream",
        "synthesis_stderr": "",
        "vivado_returncode": 0,
        "timed_out": False,
        "bitfile_built_from_stdout": True,
        "bitfile_built": True,
    }

    with (
        patch("scripts.experiment_584_kv260_synthesis.check_vivado_available", return_value=True),
        patch(
            "scripts.experiment_584_kv260_synthesis.build_vivado_result",
            return_value=fake_synthesis,
        ),
    ):
        artifact = run_experiment(repo_root=tmp_path)

    assert artifact["vivado_available"] is True
    assert artifact["bitfile_built"] is True
    assert artifact["honest_verdict"] == "bitfile_built"
    assert artifact["bitfile_path"] is not None

    # Export command file must be written
    export_path = tmp_path / EXPORT_COMMAND_PATH
    assert export_path.exists()
    assert "CARNOT_KV260_BITFILE" in export_path.read_text()


def test_run_experiment_vivado_available_synthesis_failed(tmp_path):
    """Vivado runs but no bitfile produced → 'synthesis_attempted_failed'."""
    hw_dir = tmp_path / "hardware" / "kv260"
    hw_dir.mkdir(parents=True)
    lines = ["# line %d" % i for i in range(_TCL_MINIMAL_LINE_THRESHOLD + 2)]
    lines.append("write_bitstream -force output/carnot_ising.bit")
    (hw_dir / "synth_ising.tcl").write_text("\n".join(lines))
    (tmp_path / "results").mkdir(parents=True)

    fake_synthesis = {
        "vivado_ran": True,
        "synthesis_stdout": "ERROR: synthesis failed",
        "synthesis_stderr": "place failed",
        "vivado_returncode": 2,
        "timed_out": False,
        "bitfile_built_from_stdout": False,
        "bitfile_built": False,
    }

    with (
        patch("scripts.experiment_584_kv260_synthesis.check_vivado_available", return_value=True),
        patch(
            "scripts.experiment_584_kv260_synthesis.build_vivado_result",
            return_value=fake_synthesis,
        ),
    ):
        artifact = run_experiment(repo_root=tmp_path)

    assert artifact["bitfile_built"] is False
    assert artifact["honest_verdict"] == "synthesis_attempted_failed"
    # Export command file must NOT be written when no bitfile
    assert not (tmp_path / EXPORT_COMMAND_PATH).exists()


# ---------------------------------------------------------------------------
# Constant values sanity checks
# ---------------------------------------------------------------------------


def test_constants():
    """Verify constants match task specification."""
    assert EXPERIMENT_ID == 584
    assert CPU_BASELINE_LATENCY_US == 289608.0
    assert "results/experiment_584" in DELIVERABLE
    assert len(VIVADO_INSTALL_STEPS) >= 4
    assert "vivado" in VIVADO_INSTALL_STEPS[-1]
