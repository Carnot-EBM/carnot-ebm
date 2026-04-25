"""Tests for Experiment 827 — KV260 nextpnr-xilinx Attempt and iCE40 Bitstream Fallback.

Covers:
  - valid_header check with mocked file bytes (REQ-HW-039-4)
  - fallback path triggers when nextpnr-xilinx is not available (REQ-HW-039-1, REQ-HW-039-3)
  - bitstream_path is copied to hardware/kv260/ when generated (REQ-HW-039-5)
  - gate fails when Exp 816 artifact is missing (REQ-HW-039)
  - gate fails when Exp 816 verdict is wrong (REQ-HW-039)
  - honest_verdict logic for all outcome branches (REQ-HW-039-6)

Spec: REQ-HW-039, SCENARIO-HW-035
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent.parent))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import experiment_827_kv260_nextpnr_xilinx_v3 as exp827  # noqa: E402


# ---------------------------------------------------------------------------
# validate_exp816_gate — REQ-HW-039
# ---------------------------------------------------------------------------


def test_gate_fails_when_artifact_missing(tmp_path: Path) -> None:
    """Gate returns (False, None) when Exp 816 artifact file does not exist.

    A missing artifact is treated conservatively as gate failure so we never
    attempt hardware synthesis without confirming prior results.

    Spec: REQ-HW-039
    """
    ok, lut = exp827.validate_exp816_gate(tmp_path)
    assert ok is False
    assert lut is None


def test_gate_fails_when_verdict_wrong(tmp_path: Path) -> None:
    """Gate returns False when Exp 816 honest_verdict is not synthesis_clean_n32.

    A different verdict (e.g. synthesis_errors_n32) means the prior run was
    broken and we should not proceed.

    Spec: REQ-HW-039
    """
    artifact = tmp_path / "results" / "experiment_816_kv260_synthesis_v2.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({
        "honest_verdict": "synthesis_errors_n32",
        "lut_count_n32": 3952,
    }))
    # Temporarily redirect the constant to the temp path.
    with patch.object(exp827, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"):
        ok, lut = exp827.validate_exp816_gate(tmp_path)
    assert ok is False
    assert lut == 3952


def test_gate_fails_when_lut_count_wrong(tmp_path: Path) -> None:
    """Gate returns False when lut_count_n32 != 3952 even if verdict matches.

    This guards against a re-run that somehow produced a different LUT count,
    which would invalidate our N=32 budget analysis.

    Spec: REQ-HW-039
    """
    artifact = tmp_path / "results" / "experiment_816_kv260_synthesis_v2.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({
        "honest_verdict": "synthesis_clean_n32",
        "lut_count_n32": 1234,
    }))
    with patch.object(exp827, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"):
        ok, lut = exp827.validate_exp816_gate(tmp_path)
    assert ok is False
    assert lut == 1234


def test_gate_passes_when_correct(tmp_path: Path) -> None:
    """Gate returns (True, 3952) when Exp 816 artifact matches expected values.

    Spec: REQ-HW-039
    """
    artifact = tmp_path / "results" / "experiment_816_kv260_synthesis_v2.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({
        "honest_verdict": "synthesis_clean_n32",
        "lut_count_n32": 3952,
    }))
    with patch.object(exp827, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"):
        ok, lut = exp827.validate_exp816_gate(tmp_path)
    assert ok is True
    assert lut == 3952


# ---------------------------------------------------------------------------
# check_xilinx_available — REQ-HW-039-1
# ---------------------------------------------------------------------------


def test_xilinx_not_available_when_binary_absent(tmp_path: Path) -> None:
    """check_xilinx_available returns False when nextpnr-xilinx binary does not exist.

    OSS-CAD-Suite releases do not always include nextpnr-xilinx; the binary
    check must gracefully report unavailable rather than crashing.

    Spec: REQ-HW-039-1
    """
    result = exp827.check_xilinx_available(tmp_path)
    assert result is False


def test_xilinx_available_when_binary_present_and_exits_zero(tmp_path: Path) -> None:
    """check_xilinx_available returns True when nextpnr-xilinx exits 0 on --help.

    Spec: REQ-HW-039-1
    """
    fake_binary = tmp_path / "nextpnr-xilinx"
    fake_binary.touch()
    fake_binary.chmod(0o755)

    with patch.object(exp827, "_run", return_value=(0, "nextpnr-xilinx help text", "")):
        result = exp827.check_xilinx_available(tmp_path)
    assert result is True


def test_xilinx_not_available_when_binary_present_but_exits_nonzero(tmp_path: Path) -> None:
    """check_xilinx_available returns False when nextpnr-xilinx exits non-zero on --help.

    A non-zero exit on --help indicates the binary is present but broken or
    does not recognise --help (some older builds use --version only).

    Spec: REQ-HW-039-1
    """
    fake_binary = tmp_path / "nextpnr-xilinx"
    fake_binary.touch()
    fake_binary.chmod(0o755)

    with patch.object(exp827, "_run", return_value=(1, "", "unrecognised option")):
        result = exp827.check_xilinx_available(tmp_path)
    assert result is False


# ---------------------------------------------------------------------------
# valid_header validation — REQ-HW-039-4
# ---------------------------------------------------------------------------


def test_valid_header_true_for_ice40_magic(tmp_path: Path) -> None:
    """run_ice40_bitstream reports valid_header=True when output starts with 0xFF 0x00.

    iCE40 bitstreams always begin with 0xFF 0x00 as the magic marker.  icepack
    produces this header on every valid pack operation.

    Spec: REQ-HW-039-4
    """
    # Build a fake .bin content with valid iCE40 magic header.
    fake_bin_bytes = bytes([0xFF, 0x00, 0x00, 0xFF]) + b"\x00" * 100

    def fake_run(cmd, timeout=300):
        cmd_str = " ".join(str(c) for c in cmd)
        if "yosys" in cmd_str:
            return 0, "3952   SB_LUT4\nEnd of script.\n", ""
        if "nextpnr-ice40" in cmd_str:
            # Simulate creating the ASC file.
            for part in cmd:
                part = str(part)
                if part.endswith(".asc"):
                    Path(part).write_text(".comment test\n.device hx8k\n")
            return 0, "nextpnr-ice40 ok", ""
        if "icepack" in cmd_str:
            # Locate the .bin output path from the command args and write fake bytes.
            parts = [str(c) for c in cmd]
            bin_path = parts[-1]
            Path(bin_path).write_bytes(fake_bin_bytes)
            return 0, "", ""
        return 0, "", ""

    with patch.object(exp827, "_run", side_effect=fake_run):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            # Create fake binaries so existence checks pass.
            for name in ("yosys", "nextpnr-ice40", "icepack"):
                (tmp_dir / name).touch()
            # Write a minimal RTL source.
            rtl_source = tmp_dir / "ising_sampler_v3.v"
            rtl_source.write_text("module ising_sampler_v3(); endmodule\n")
            work_dir = tmp_dir / "work"
            work_dir.mkdir()
            generated, valid, size, log = exp827.run_ice40_bitstream(
                tmp_dir, rtl_source, work_dir
            )

    assert generated is True
    assert valid is True
    assert size == len(fake_bin_bytes)


def test_valid_header_false_for_wrong_magic(tmp_path: Path) -> None:
    """run_ice40_bitstream reports valid_header=False when output does not start 0xFF 0x00.

    A truncated or corrupted bitstream that fails the magic check must NOT be
    reported as programmable hardware — we need an explicit failure signal.

    Spec: REQ-HW-039-4
    """
    bad_bytes = bytes([0x00, 0xFF, 0x00, 0xFF]) + b"\x00" * 100

    def fake_run(cmd, timeout=300):
        cmd_str = " ".join(str(c) for c in cmd)
        if "yosys" in cmd_str:
            return 0, "3952   SB_LUT4\n", ""
        if "nextpnr-ice40" in cmd_str:
            for part in cmd:
                part = str(part)
                if part.endswith(".asc"):
                    Path(part).write_text(".comment test\n")
            return 0, "ok", ""
        if "icepack" in cmd_str:
            parts = [str(c) for c in cmd]
            Path(parts[-1]).write_bytes(bad_bytes)
            return 0, "", ""
        return 0, "", ""

    with patch.object(exp827, "_run", side_effect=fake_run):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            rtl_source = tmp_dir / "ising_sampler_v3.v"
            rtl_source.write_text("module ising_sampler_v3(); endmodule\n")
            work_dir = tmp_dir / "work"
            work_dir.mkdir()
            generated, valid, size, log = exp827.run_ice40_bitstream(
                tmp_dir, rtl_source, work_dir
            )

    assert generated is True
    assert valid is False


# ---------------------------------------------------------------------------
# bitstream_path copy — REQ-HW-039-5
# ---------------------------------------------------------------------------


def test_bitstream_copied_to_hardware_kv260(tmp_path: Path) -> None:
    """run_experiment copies a valid bitstream to hardware/kv260/ising_n32.bin.

    The hardware/kv260/ directory is the canonical location for KV260-targeted
    bitstreams in the repository so developers can find them easily.

    Spec: REQ-HW-039-5
    """
    # Set up a mock repo root with required structure.
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    hw_dir = tmp_path / "hardware" / "kv260"
    hw_dir.mkdir(parents=True)

    # Write a passing Exp 816 gate artifact.
    gate_art = results_dir / "experiment_816_kv260_synthesis_v2.json"
    gate_art.write_text(json.dumps({
        "honest_verdict": "synthesis_clean_n32",
        "lut_count_n32": 3952,
    }))

    # Write a minimal RTL source.
    rtl = hw_dir / "ising_sampler_v3.v"
    rtl.write_text("module ising_sampler_v3(); endmodule\n")

    fake_bin_bytes = bytes([0xFF, 0x00, 0x00, 0xFF]) + b"\x00" * 200

    def fake_run(cmd, timeout=300):
        cmd_str = " ".join(str(c) for c in cmd)
        if "yosys" in cmd_str:
            return 0, "3952   SB_LUT4\n", ""
        if "nextpnr-ice40" in cmd_str:
            for part in cmd:
                part = str(part)
                if part.endswith(".asc"):
                    Path(part).write_text(".comment test\n")
            return 0, "ok", ""
        if "icepack" in cmd_str:
            parts = [str(c) for c in cmd]
            Path(parts[-1]).write_bytes(fake_bin_bytes)
            return 0, "", ""
        return 0, "", ""

    # Build a fake OSS-CAD bin directory so existence checks pass.
    fake_oss = tmp_path / "oss-cad-suite" / "bin"
    fake_oss.mkdir(parents=True)
    for name in ("yosys", "nextpnr-ice40", "icepack"):
        (fake_oss / name).touch()
    # nextpnr-xilinx is absent — this forces the fallback path.

    mock_tmpl = MagicMock()
    mock_tmpl._repo_root = str(tmp_path)

    with (
        patch.object(exp827, "OSS_CAD_BIN", fake_oss),
        patch.object(exp827, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"),
        patch.object(exp827, "RTL_SOURCE", "hardware/kv260/ising_sampler_v3.v"),
        patch.object(exp827, "BITSTREAM_DEST", "hardware/kv260/ising_n32.bin"),
        patch.object(exp827, "_run", side_effect=fake_run),
    ):
        fields, status = exp827.run_experiment(mock_tmpl)

    assert status == "success"
    assert fields["honest_verdict"] == "ice40_bitstream_generated"
    assert fields["ice40_bitstream_generated"] is True
    assert fields["valid_header"] is True
    dest_bin = tmp_path / "hardware" / "kv260" / "ising_n32.bin"
    assert dest_bin.exists(), "bitstream was not copied to hardware/kv260/"
    assert dest_bin.read_bytes() == fake_bin_bytes


# ---------------------------------------------------------------------------
# honest_verdict logic — REQ-HW-039-6
# ---------------------------------------------------------------------------


def test_honest_verdict_synthesis_blocked_when_gate_fails(tmp_path: Path) -> None:
    """honest_verdict is 'synthesis_blocked' when the Exp 816 gate fails.

    Spec: REQ-HW-039-6
    """
    mock_tmpl = MagicMock()
    mock_tmpl._repo_root = str(tmp_path)

    with (
        patch.object(exp827, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"),
    ):
        fields, status = exp827.run_experiment(mock_tmpl)

    assert status == "blocked"
    assert fields["honest_verdict"] == "synthesis_blocked"


def test_honest_verdict_bitstream_invalid_header(tmp_path: Path) -> None:
    """honest_verdict is 'bitstream_invalid_header' when .bin has wrong magic.

    Spec: REQ-HW-039-6
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    hw_dir = tmp_path / "hardware" / "kv260"
    hw_dir.mkdir(parents=True)

    gate_art = results_dir / "experiment_816_kv260_synthesis_v2.json"
    gate_art.write_text(json.dumps({
        "honest_verdict": "synthesis_clean_n32",
        "lut_count_n32": 3952,
    }))
    rtl = hw_dir / "ising_sampler_v3.v"
    rtl.write_text("module ising_sampler_v3(); endmodule\n")

    bad_bytes = bytes([0xDE, 0xAD, 0xBE, 0xEF])

    def fake_run(cmd, timeout=300):
        cmd_str = " ".join(str(c) for c in cmd)
        if "yosys" in cmd_str:
            return 0, "3952   SB_LUT4\n", ""
        if "nextpnr-ice40" in cmd_str:
            for part in cmd:
                part = str(part)
                if part.endswith(".asc"):
                    Path(part).write_text(".comment test\n")
            return 0, "ok", ""
        if "icepack" in cmd_str:
            parts = [str(c) for c in cmd]
            Path(parts[-1]).write_bytes(bad_bytes)
            return 0, "", ""
        return 0, "", ""

    fake_oss = tmp_path / "oss-cad-suite" / "bin"
    fake_oss.mkdir(parents=True)
    for name in ("yosys", "nextpnr-ice40", "icepack"):
        (fake_oss / name).touch()

    mock_tmpl = MagicMock()
    mock_tmpl._repo_root = str(tmp_path)

    with (
        patch.object(exp827, "OSS_CAD_BIN", fake_oss),
        patch.object(exp827, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"),
        patch.object(exp827, "RTL_SOURCE", "hardware/kv260/ising_sampler_v3.v"),
        patch.object(exp827, "BITSTREAM_DEST", "hardware/kv260/ising_n32.bin"),
        patch.object(exp827, "_run", side_effect=fake_run),
    ):
        fields, status = exp827.run_experiment(mock_tmpl)

    assert fields["honest_verdict"] == "bitstream_invalid_header"
    assert fields["ice40_bitstream_generated"] is True
    assert fields["valid_header"] is False
