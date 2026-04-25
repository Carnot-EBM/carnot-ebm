"""Tests for Experiment 839 — iCE40 HX8K Place-and-Route + Bitstream Generation.

Covers:
  - validate_exp816_gate: gate passes/fails based on Exp 816 artifact (REQ-HW-040-1)
  - check_nextpnr_ice40_available: binary presence and exit-code checks (REQ-HW-040-3)
  - run_pnr_and_pack: bitstream header validation for valid and invalid magic (REQ-HW-040-5)
  - run_experiment: honest_verdict for all outcome branches (REQ-HW-040-6)
  - run_experiment: bitstream copied to hardware/kv260/ on valid header (REQ-HW-040-5)

Spec: REQ-HW-040, SCENARIO-HW-044
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

import experiment_839_kv260_ice40_bitstream as exp839  # noqa: E402


# ---------------------------------------------------------------------------
# validate_exp816_gate — REQ-HW-040-1
# ---------------------------------------------------------------------------


def test_gate_fails_when_artifact_missing(tmp_path: Path) -> None:
    """Gate returns (False, None) when Exp 816 artifact does not exist.

    A missing artifact is treated conservatively as gate failure so we never
    attempt PnR without confirming prior synthesis results.

    Spec: REQ-HW-040-1
    """
    ok, lut = exp839.validate_exp816_gate(tmp_path)
    assert ok is False
    assert lut is None


def test_gate_fails_when_verdict_wrong(tmp_path: Path) -> None:
    """Gate returns False when Exp 816 honest_verdict is not synthesis_clean_n32.

    Spec: REQ-HW-040-1
    """
    artifact = tmp_path / "results" / "experiment_816_kv260_synthesis_v2.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({
        "honest_verdict": "synthesis_errors_n32",
        "lut_count_n32": 3952,
    }))
    with patch.object(exp839, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"):
        ok, lut = exp839.validate_exp816_gate(tmp_path)
    assert ok is False
    assert lut == 3952


def test_gate_fails_when_lut_count_wrong(tmp_path: Path) -> None:
    """Gate returns False when lut_count_n32 != 3952 even if verdict matches.

    Spec: REQ-HW-040-1
    """
    artifact = tmp_path / "results" / "experiment_816_kv260_synthesis_v2.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({
        "honest_verdict": "synthesis_clean_n32",
        "lut_count_n32": 9999,
    }))
    with patch.object(exp839, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"):
        ok, lut = exp839.validate_exp816_gate(tmp_path)
    assert ok is False
    assert lut == 9999


def test_gate_fails_when_artifact_malformed(tmp_path: Path) -> None:
    """Gate returns (False, None) when artifact JSON cannot be parsed.

    Spec: REQ-HW-040-1
    """
    artifact = tmp_path / "results" / "experiment_816_kv260_synthesis_v2.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("not valid json {{")
    with patch.object(exp839, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"):
        ok, lut = exp839.validate_exp816_gate(tmp_path)
    assert ok is False
    assert lut is None


def test_gate_passes_when_correct(tmp_path: Path) -> None:
    """Gate returns (True, 3952) when Exp 816 artifact matches expected values.

    Spec: REQ-HW-040-1
    """
    artifact = tmp_path / "results" / "experiment_816_kv260_synthesis_v2.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({
        "honest_verdict": "synthesis_clean_n32",
        "lut_count_n32": 3952,
    }))
    with patch.object(exp839, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"):
        ok, lut = exp839.validate_exp816_gate(tmp_path)
    assert ok is True
    assert lut == 3952


# ---------------------------------------------------------------------------
# check_nextpnr_ice40_available — REQ-HW-040-3
# ---------------------------------------------------------------------------


def test_nextpnr_not_available_when_binaries_absent(tmp_path: Path) -> None:
    """check_nextpnr_ice40_available returns False when nextpnr-ice40/icepack missing.

    OSS-CAD-Suite must have BOTH nextpnr-ice40 and icepack for the full pipeline.

    Spec: REQ-HW-040-3
    """
    result = exp839.check_nextpnr_ice40_available(tmp_path)
    assert result is False


def test_nextpnr_not_available_when_icepack_missing(tmp_path: Path) -> None:
    """check_nextpnr_ice40_available returns False when icepack is absent.

    nextpnr-ice40 alone cannot complete the pipeline without icepack.

    Spec: REQ-HW-040-3
    """
    (tmp_path / "nextpnr-ice40").touch()
    result = exp839.check_nextpnr_ice40_available(tmp_path)
    assert result is False


def test_nextpnr_available_when_both_present_and_exit_zero(tmp_path: Path) -> None:
    """check_nextpnr_ice40_available returns True when both binaries exist and --help exits 0.

    Spec: REQ-HW-040-3
    """
    (tmp_path / "nextpnr-ice40").touch()
    (tmp_path / "icepack").touch()

    with patch.object(exp839, "_run", return_value=(0, "nextpnr-ice40 help", "")):
        result = exp839.check_nextpnr_ice40_available(tmp_path)
    assert result is True


def test_nextpnr_not_available_when_help_exits_nonzero(tmp_path: Path) -> None:
    """check_nextpnr_ice40_available returns False when --help exits non-zero.

    Spec: REQ-HW-040-3
    """
    (tmp_path / "nextpnr-ice40").touch()
    (tmp_path / "icepack").touch()

    with patch.object(exp839, "_run", return_value=(1, "", "error")):
        result = exp839.check_nextpnr_ice40_available(tmp_path)
    assert result is False


# ---------------------------------------------------------------------------
# run_pnr_and_pack — bitstream header validation — REQ-HW-040-5
# ---------------------------------------------------------------------------


def _make_fake_run(bin_bytes: bytes | None, *, pnr_rc: int = 0, yosys_rc: int = 0):
    """Build a fake _run() side-effect for run_pnr_and_pack tests."""
    def fake_run(cmd, timeout=300):  # noqa: ANN001
        cmd_str = " ".join(str(c) for c in cmd)
        if "yosys" in cmd_str:
            return yosys_rc, "3952   SB_LUT4\n", ""
        if "nextpnr-ice40" in cmd_str:
            if pnr_rc != 0:
                return pnr_rc, "", "nextpnr-ice40 ERROR: placement failed"
            # Simulate ASC file creation from the --asc flag.
            parts = [str(c) for c in cmd]
            for i, part in enumerate(parts):
                if part == "--asc" and i + 1 < len(parts):
                    Path(parts[i + 1]).write_text(".comment test\n.device hx8k\n")
            return 0, "Info: Placed 3952 cells\n", ""
        if "icepack" in cmd_str:
            parts = [str(c) for c in cmd]
            bin_path = parts[-1]
            if bin_bytes is not None:
                Path(bin_path).write_bytes(bin_bytes)
            return 0, "", ""
        return 0, "", ""
    return fake_run


def test_bitstream_valid_header_for_ice40_magic(tmp_path: Path) -> None:
    """run_pnr_and_pack reports bitstream_valid_header=True for 0xFF 0x00 magic.

    The iCE40 bitstream format requires the first two bytes to be 0xFF 0x00.
    icepack always produces this header for valid HX8K bitstreams.

    Spec: REQ-HW-040-5
    """
    fake_bin = bytes([0xFF, 0x00, 0x00, 0xFF]) + b"\x00" * 100
    rtl_source = tmp_path / "ising_sampler_v3.v"
    rtl_source.write_text(
        "module ising_sampler_v3 #(parameter integer N = 64) (); endmodule\n"
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with patch.object(exp839, "_run", side_effect=_make_fake_run(fake_bin)):
        pnr_ok, timing_met, gen, size, valid, log = exp839.run_pnr_and_pack(
            tmp_path, rtl_source, work_dir
        )

    assert pnr_ok is True
    assert gen is True
    assert valid is True
    assert size == len(fake_bin)


def test_bitstream_invalid_header_for_wrong_magic(tmp_path: Path) -> None:
    """run_pnr_and_pack reports bitstream_valid_header=False when magic is wrong.

    A corrupted or non-iCE40 file must NOT be reported as valid hardware artifact.

    Spec: REQ-HW-040-5
    """
    bad_bin = bytes([0x00, 0xFF, 0xAA, 0xBB]) + b"\x00" * 100
    rtl_source = tmp_path / "ising_sampler_v3.v"
    rtl_source.write_text(
        "module ising_sampler_v3 #(parameter integer N = 64) (); endmodule\n"
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with patch.object(exp839, "_run", side_effect=_make_fake_run(bad_bin)):
        pnr_ok, timing_met, gen, size, valid, log = exp839.run_pnr_and_pack(
            tmp_path, rtl_source, work_dir
        )

    assert pnr_ok is True
    assert gen is True
    assert valid is False


def test_pnr_failed_when_nextpnr_exits_nonzero(tmp_path: Path) -> None:
    """run_pnr_and_pack returns pnr_complete=False when nextpnr-ice40 exits non-zero.

    A non-zero PnR exit means routing could not complete; no ASC/bin is produced.

    Spec: REQ-HW-040-4
    """
    rtl_source = tmp_path / "ising_sampler_v3.v"
    rtl_source.write_text(
        "module ising_sampler_v3 #(parameter integer N = 64) (); endmodule\n"
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with patch.object(exp839, "_run", side_effect=_make_fake_run(None, pnr_rc=1)):
        pnr_ok, timing_met, gen, size, valid, log = exp839.run_pnr_and_pack(
            tmp_path, rtl_source, work_dir
        )

    assert pnr_ok is False
    assert gen is False


def test_timing_not_met_when_constraint_violation_in_log(tmp_path: Path) -> None:
    """run_pnr_and_pack reports timing_met=False when PnR log contains 'constraint not met'.

    nextpnr-ice40 prints this phrase when the --freq target cannot be achieved.
    We treat any such message as timing failure, even if routing completed.

    Spec: REQ-HW-040-4
    """
    fake_bin = bytes([0xFF, 0x00, 0x00, 0xFF]) + b"\x00" * 50

    def fake_run_timing(cmd, timeout=300):  # noqa: ANN001
        cmd_str = " ".join(str(c) for c in cmd)
        if "yosys" in cmd_str:
            return 0, "3952   SB_LUT4\n", ""
        if "nextpnr-ice40" in cmd_str:
            parts = [str(c) for c in cmd]
            for i, p in enumerate(parts):
                if p == "--asc" and i + 1 < len(parts):
                    Path(parts[i + 1]).write_text(".comment timing test\n")
            # Include 'constraint not met' in stderr to trigger timing failure.
            return 0, "Placed ok\n", "Warning: constraint not met\n"
        if "icepack" in cmd_str:
            parts = [str(c) for c in cmd]
            Path(parts[-1]).write_bytes(fake_bin)
            return 0, "", ""
        return 0, "", ""

    rtl_source = tmp_path / "ising_sampler_v3.v"
    rtl_source.write_text(
        "module ising_sampler_v3 #(parameter integer N = 64) (); endmodule\n"
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with patch.object(exp839, "_run", side_effect=fake_run_timing):
        pnr_ok, timing_met, gen, size, valid, log = exp839.run_pnr_and_pack(
            tmp_path, rtl_source, work_dir
        )

    assert pnr_ok is True
    assert timing_met is False


# ---------------------------------------------------------------------------
# run_experiment — honest_verdict logic — REQ-HW-040-6
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path, gate_passes: bool = True) -> Path:
    """Create a minimal repo structure for run_experiment tests."""
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    hw_dir = tmp_path / "hardware" / "kv260"
    hw_dir.mkdir(parents=True, exist_ok=True)
    rtl = hw_dir / "ising_sampler_v3.v"
    rtl.write_text(
        "module ising_sampler_v3 #(parameter integer N = 64) (); endmodule\n"
    )
    if gate_passes:
        art = tmp_path / "results" / "experiment_816_kv260_synthesis_v2.json"
        art.write_text(json.dumps({
            "honest_verdict": "synthesis_clean_n32",
            "lut_count_n32": 3952,
        }))
    return tmp_path


def _make_mock_tmpl(tmp_path: Path) -> MagicMock:
    mock = MagicMock()
    mock._repo_root = str(tmp_path)
    return mock


def test_honest_verdict_synthesis_artifact_missing_when_gate_fails(tmp_path: Path) -> None:
    """honest_verdict is 'synthesis_artifact_missing' when Exp 816 gate fails.

    Spec: REQ-HW-040-6
    """
    _make_repo(tmp_path, gate_passes=False)
    mock_tmpl = _make_mock_tmpl(tmp_path)

    with patch.object(exp839, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"):
        fields, status = exp839.run_experiment(mock_tmpl)

    assert fields["honest_verdict"] == "synthesis_artifact_missing"
    assert status == "blocked"


def test_honest_verdict_nextpnr_not_available(tmp_path: Path) -> None:
    """honest_verdict is 'nextpnr_not_available' when nextpnr-ice40 is missing.

    Spec: REQ-HW-040-6
    """
    _make_repo(tmp_path)
    mock_tmpl = _make_mock_tmpl(tmp_path)
    fake_oss = tmp_path / "oss-cad" / "bin"
    fake_oss.mkdir(parents=True)
    # Only create yosys, not nextpnr-ice40 or icepack.
    (fake_oss / "yosys").touch()

    with (
        patch.object(exp839, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"),
        patch.object(exp839, "OSS_CAD_BIN", fake_oss),
    ):
        fields, status = exp839.run_experiment(mock_tmpl)

    assert fields["honest_verdict"] == "nextpnr_not_available"
    assert fields["nextpnr_available"] is False
    assert status == "blocked"


def test_honest_verdict_pnr_failed(tmp_path: Path) -> None:
    """honest_verdict is 'pnr_failed' when nextpnr-ice40 exits non-zero.

    Spec: REQ-HW-040-6
    """
    _make_repo(tmp_path)
    mock_tmpl = _make_mock_tmpl(tmp_path)
    fake_oss = tmp_path / "oss-cad" / "bin"
    fake_oss.mkdir(parents=True)
    for name in ("yosys", "nextpnr-ice40", "icepack"):
        (fake_oss / name).touch()

    with (
        patch.object(exp839, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"),
        patch.object(exp839, "OSS_CAD_BIN", fake_oss),
        patch.object(exp839, "check_nextpnr_ice40_available", return_value=True),
        patch.object(exp839, "run_pnr_and_pack",
                     return_value=(False, False, False, 0, False, "pnr error")),
    ):
        fields, status = exp839.run_experiment(mock_tmpl)

    assert fields["honest_verdict"] == "pnr_failed"


def test_honest_verdict_bitstream_generated(tmp_path: Path) -> None:
    """honest_verdict is 'bitstream_generated' when .bin has valid iCE40 magic.

    The bitstream must be copied to hardware/kv260/ and output_path set.

    Spec: REQ-HW-040-6, REQ-HW-040-5
    """
    _make_repo(tmp_path)
    mock_tmpl = _make_mock_tmpl(tmp_path)
    fake_oss = tmp_path / "oss-cad" / "bin"
    fake_oss.mkdir(parents=True)
    for name in ("yosys", "nextpnr-ice40", "icepack"):
        (fake_oss / name).touch()

    fake_bin = bytes([0xFF, 0x00, 0x00, 0xFF]) + b"\x00" * 200

    def fake_pnr_and_pack(oss_cad_bin, rtl_source, tmp_dir):  # noqa: ANN001
        bin_out = tmp_dir / "carnot_ising_n32.bin"
        bin_out.write_bytes(fake_bin)
        return True, True, True, len(fake_bin), True, "[yosys ok]\n[nextpnr ok]\n[icepack ok]"

    with (
        patch.object(exp839, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"),
        patch.object(exp839, "OSS_CAD_BIN", fake_oss),
        patch.object(exp839, "BITSTREAM_DEST", "hardware/kv260/ising_n32_exp839.bin"),
        patch.object(exp839, "check_nextpnr_ice40_available", return_value=True),
        patch.object(exp839, "run_pnr_and_pack", side_effect=fake_pnr_and_pack),
    ):
        fields, status = exp839.run_experiment(mock_tmpl)

    assert fields["honest_verdict"] == "bitstream_generated"
    assert fields["bitstream_generated"] is True
    assert fields["bitstream_valid_header"] is True
    assert status == "success"
    dest = tmp_path / "hardware" / "kv260" / "ising_n32_exp839.bin"
    assert dest.exists(), "bitstream not copied to hardware/kv260/"


def test_honest_verdict_bitstream_generated_invalid_header(tmp_path: Path) -> None:
    """honest_verdict is 'bitstream_generated_invalid_header' when magic is wrong.

    Spec: REQ-HW-040-6
    """
    _make_repo(tmp_path)
    mock_tmpl = _make_mock_tmpl(tmp_path)
    fake_oss = tmp_path / "oss-cad" / "bin"
    fake_oss.mkdir(parents=True)
    for name in ("yosys", "nextpnr-ice40", "icepack"):
        (fake_oss / name).touch()

    with (
        patch.object(exp839, "EXP816_ARTIFACT", "results/experiment_816_kv260_synthesis_v2.json"),
        patch.object(exp839, "OSS_CAD_BIN", fake_oss),
        patch.object(exp839, "BITSTREAM_DEST", "hardware/kv260/ising_n32_exp839.bin"),
        patch.object(exp839, "check_nextpnr_ice40_available", return_value=True),
        patch.object(exp839, "run_pnr_and_pack",
                     return_value=(True, True, True, 256, False, "[log]")),
    ):
        fields, status = exp839.run_experiment(mock_tmpl)

    assert fields["honest_verdict"] == "bitstream_generated_invalid_header"
    assert fields["bitstream_valid_header"] is False
