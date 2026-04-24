"""Tests for Experiment 804 — KV260 N=32 Yosys Open-Source Synthesis Attempt.

Covers:
  - LUT count parsing from yosys stat output (REQ-HW-035)
  - Gate logic: blocked when Exp 794 reports tools not installed (REQ-HW-035)
  - Gate logic: pass when Exp 794 reports tools installed (REQ-HW-035)
  - RTL parameter patching for N override (REQ-HW-035)

Spec: REQ-HW-035, SCENARIO-HW-033
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent.parent))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import experiment_804_kv260_synthesis_attempt as exp804  # noqa: E402


# ---------------------------------------------------------------------------
# Test: _parse_lut_count extracts SB_LUT4 count correctly (REQ-HW-035)
# ---------------------------------------------------------------------------


def test_parse_lut_count_sb_lut4() -> None:
    """_parse_lut_count returns the SB_LUT4 count from yosys stat output.

    Spec: REQ-HW-035
    """
    yosys_out = """
2.25. Printing statistics.

=== ising_sampler_v3 ===

   Number of wires:               1234
   Number of wire bits:           5678
   Number of public wires:          42
   Number of public wire bits:     420
   Number of cells:               1812
     SB_CARRY                      200
     SB_DFF                        400
     SB_LUT4                       612
"""
    result = exp804._parse_lut_count(yosys_out)
    assert result == 612, "Should parse SB_LUT4 count as 612"


def test_parse_lut_count_prefers_sb_lut4_over_cells() -> None:
    """_parse_lut_count prefers SB_LUT4 over total 'Number of cells' fallback.

    The cell count includes flip-flops and carry chains, not just LUTs.
    SB_LUT4 is the correct metric for iCE40 LUT budget comparison.

    Spec: REQ-HW-035
    """
    yosys_out = """
   Number of cells:               1812
     SB_LUT4                       300
"""
    result = exp804._parse_lut_count(yosys_out)
    assert result == 300, "SB_LUT4 (300) should be preferred over Number of cells (1812)"


def test_parse_lut_count_fallback_to_cells() -> None:
    """_parse_lut_count falls back to 'Number of cells' when SB_LUT4 not present.

    Some yosys versions omit SB_LUT4 in the stat output when targeting generic
    backends. The fallback ensures we still get a usable count.

    Spec: REQ-HW-035
    """
    yosys_out = "   Number of cells:                 999\n"
    result = exp804._parse_lut_count(yosys_out)
    assert result == 999


def test_parse_lut_count_missing_returns_none() -> None:
    """_parse_lut_count returns None when neither pattern is present.

    This happens when yosys exits before emitting stat output (e.g. parse error).

    Spec: REQ-HW-035
    """
    result = exp804._parse_lut_count("Error: module not found\n")
    assert result is None


# ---------------------------------------------------------------------------
# Test: gate blocks when Exp 794 honest_verdict is tools_not_installed (REQ-HW-035)
# ---------------------------------------------------------------------------


def test_gate_blocks_when_tools_not_installed(tmp_path: Path) -> None:
    """run_experiment returns honest_verdict='tools_not_installed' when gate blocks.

    Gate condition: Exp 794 honest_verdict not in
    ['tools_installed_synthesis_clean', 'tools_installed_synthesis_failed'].

    Spec: REQ-HW-035, SCENARIO-HW-033
    """
    # Write a fake Exp 794 artifact that mirrors the actual failure state
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    gate_artifact = {
        "experiment": 794,
        "honest_verdict": "tools_not_installed_install_attempted",
        "tools_installed": False,
    }
    (results_dir / "experiment_794_fpga_toolchain_install.json").write_text(
        json.dumps(gate_artifact)
    )

    # Create a minimal ExperimentTemplate that points at tmp_path
    from experiment_template import ExperimentTemplate  # noqa: PLC0415
    tmpl = ExperimentTemplate(
        exp_id=804,
        title="test",
        deliverable="results/experiment_804_kv260_synthesis_attempt.json",
        requires_gpu=False,
        repo_root=tmp_path,
    )

    data, status = exp804.run_experiment(tmpl)

    assert status == "blocked", "Status must be 'blocked' when gate fails"
    assert data["honest_verdict"] == "tools_not_installed"
    assert data["gate_passed"] is False
    assert "block_reason" in data


def test_gate_blocks_when_artifact_missing(tmp_path: Path) -> None:
    """run_experiment blocks when Exp 794 artifact does not exist.

    A missing gate artifact is treated as tools_installed=False to be safe —
    we cannot proceed with synthesis if we cannot confirm tools are present.

    Spec: REQ-HW-035
    """
    # No artifact written — results dir doesn't even exist
    from experiment_template import ExperimentTemplate  # noqa: PLC0415
    tmpl = ExperimentTemplate(
        exp_id=804,
        title="test",
        deliverable="results/experiment_804_kv260_synthesis_attempt.json",
        requires_gpu=False,
        repo_root=tmp_path,
    )

    data, status = exp804.run_experiment(tmpl)

    assert status == "blocked"
    assert data["honest_verdict"] == "tools_not_installed"
    assert data["gate_passed"] is False


# ---------------------------------------------------------------------------
# Test: gate passes when Exp 794 verdict is in allowed set (REQ-HW-035)
# ---------------------------------------------------------------------------


def test_gate_passes_with_synthesis_failed_verdict(tmp_path: Path) -> None:
    """run_experiment proceeds past the gate when verdict='tools_installed_synthesis_failed'.

    Both 'tools_installed_synthesis_clean' and 'tools_installed_synthesis_failed' are
    allowed gate-pass values because both indicate yosys is actually installed — the
    synthesis outcome varies per RTL, not per tool availability.

    Spec: REQ-HW-035
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    gate_artifact = {
        "experiment": 794,
        "honest_verdict": "tools_installed_synthesis_failed",
        "tools_installed": True,
    }
    (results_dir / "experiment_794_fpga_toolchain_install.json").write_text(
        json.dumps(gate_artifact)
    )

    # Copy the actual RTL file so the experiment can read it
    hw_dir = tmp_path / "hardware" / "kv260"
    hw_dir.mkdir(parents=True)
    real_rtl = REPO_ROOT / "hardware" / "kv260" / "ising_sampler_v3.v"
    if real_rtl.exists():
        (hw_dir / "ising_sampler_v3.v").write_text(real_rtl.read_text())
    else:
        # Stub RTL for pure gate test — yosys will fail but gate should pass
        (hw_dir / "ising_sampler_v3.v").write_text("module stub(); endmodule\n")

    from experiment_template import ExperimentTemplate  # noqa: PLC0415
    tmpl = ExperimentTemplate(
        exp_id=804,
        title="test",
        deliverable="results/experiment_804_kv260_synthesis_attempt.json",
        requires_gpu=False,
        repo_root=tmp_path,
    )

    # Patch yosys not found so the test doesn't require the toolchain installed
    with patch.object(exp804, "_find_yosys", return_value=(False, "")):
        data, status = exp804.run_experiment(tmpl)

    # Gate passed but no yosys → blocked on tools_not_installed (different path)
    assert data["gate_passed"] is True, "Gate must pass when verdict is tools_installed_synthesis_failed"


# ---------------------------------------------------------------------------
# Test: _patch_n_spins overrides N parameter (REQ-HW-035)
# ---------------------------------------------------------------------------


def test_patch_n_spins_overrides_parameter() -> None:
    """_patch_n_spins replaces N=64 with the requested N in RTL text.

    The yosys synthesis uses this to target N=32 without editing the source file.

    Spec: REQ-HW-035
    """
    rtl = "parameter integer N              = 64,\n"
    patched = exp804._patch_n_spins(rtl, n_spins=32, max_degree=8)
    assert "= 32" in patched
    assert "= 64" not in patched


def test_patch_n_spins_leaves_other_params_unchanged() -> None:
    """_patch_n_spins only changes the N parameter, not EMA_ALPHA_NUM or similar.

    Spec: REQ-HW-035
    """
    rtl = (
        "parameter integer N              = 64,\n"
        "parameter integer EMA_ALPHA_NUM  = 7,\n"
        "parameter integer EMA_ALPHA_DEN  = 8,\n"
    )
    patched = exp804._patch_n_spins(rtl, n_spins=32, max_degree=8)
    assert "EMA_ALPHA_NUM  = 7" in patched
    assert "EMA_ALPHA_DEN  = 8" in patched


# ---------------------------------------------------------------------------
# Test: load_gate_artifact handles corrupt JSON gracefully (REQ-HW-035)
# ---------------------------------------------------------------------------


def test_load_gate_artifact_handles_corrupt_json(tmp_path: Path) -> None:
    """load_gate_artifact returns {} on corrupt JSON, causing gate to block.

    A corrupt artifact is treated the same as missing — we cannot confirm tools
    are present, so we block rather than synthesize with unknown tool state.

    Spec: REQ-HW-035
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "experiment_794_fpga_toolchain_install.json").write_text(
        "{ this is not valid json }"
    )
    result = exp804.load_gate_artifact(tmp_path)
    assert result == {}
