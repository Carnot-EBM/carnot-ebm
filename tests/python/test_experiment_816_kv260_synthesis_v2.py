"""Tests for Experiment 816 — KV260 Ising Sampler v3 Synthesis via OSS-CAD-Suite Yosys.

Covers:
  - LUT count parsing from yosys stat output (REQ-HW-038)
  - RTL N-parameter patching (REQ-HW-038)
  - Gate blocks when Exp 807 verdict is invalid (REQ-HW-038-3)
  - Gate blocks when Exp 807 artifact is missing (REQ-HW-038-3)
  - Gate passes when Exp 807 verdict is 'tools_installed_synthesis_clean' (REQ-HW-038-3)
  - Gate passes when Exp 807 verdict is 'already_installed' (REQ-HW-038-3)
  - Blocked when yosys binary is absent at expected path (REQ-HW-038-2)

Spec: REQ-HW-038, SCENARIO-HW-035
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent.parent))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import experiment_816_kv260_synthesis_v2 as exp816  # noqa: E402


# ---------------------------------------------------------------------------
# _parse_lut_count — REQ-HW-038
# ---------------------------------------------------------------------------


def test_parse_lut_count_sb_lut4() -> None:
    """_parse_lut_count extracts SB_LUT4 count from yosys stat output.

    Spec: REQ-HW-038
    """
    yosys_out = (
        "=== ising_sampler_v3 ===\n"
        "   Number of wires:               1234\n"
        "   Number of cells:               1812\n"
        "     SB_CARRY                      200\n"
        "     SB_DFF                        400\n"
        "     SB_LUT4                       612\n"
    )
    assert exp816._parse_lut_count(yosys_out) == 612


def test_parse_lut_count_prefers_sb_lut4_over_cells() -> None:
    """_parse_lut_count returns SB_LUT4, not the larger Number of cells total.

    SB_LUT4 is the correct iCE40 LUT metric; cells includes flip-flops etc.

    Spec: REQ-HW-038
    """
    yosys_out = "   Number of cells:               1812\n     SB_LUT4                       300\n"
    assert exp816._parse_lut_count(yosys_out) == 300


def test_parse_lut_count_fallback_to_cells() -> None:
    """_parse_lut_count falls back to Number of cells when SB_LUT4 is absent.

    Spec: REQ-HW-038
    """
    assert exp816._parse_lut_count("   Number of cells:                 999\n") == 999


def test_parse_lut_count_reversed_format() -> None:
    """_parse_lut_count handles yosys 0.64+ format where count precedes the name.

    yosys 0.64+ (OSS-CAD-Suite) emits stat lines as:
        3952   SB_LUT4
    rather than the older:
        SB_LUT4   3952

    Spec: REQ-HW-038
    """
    yosys_out = (
        "=== ising_sampler_v3 ===\n"
        "     6897 cells\n"
        "     1216   SB_CARRY\n"
        "        1   SB_DFF\n"
        "     3952   SB_LUT4\n"
    )
    assert exp816._parse_lut_count(yosys_out) == 3952


def test_parse_lut_count_returns_none_when_absent() -> None:
    """_parse_lut_count returns None when neither pattern is found.

    Spec: REQ-HW-038
    """
    assert exp816._parse_lut_count("Error: module not found\n") is None


def test_parse_lut_count_empty_string() -> None:
    """_parse_lut_count returns None on empty string input.

    Spec: REQ-HW-038
    """
    assert exp816._parse_lut_count("") is None


# ---------------------------------------------------------------------------
# _patch_n_spins — REQ-HW-038
# ---------------------------------------------------------------------------


def test_patch_n_spins_replaces_default() -> None:
    """_patch_n_spins replaces the N parameter default with the requested value.

    Spec: REQ-HW-038
    """
    rtl = "parameter integer N              = 64,\n"
    patched = exp816._patch_n_spins(rtl, 32)
    assert "= 32" in patched
    assert "= 64" not in patched


def test_patch_n_spins_leaves_other_params_unchanged() -> None:
    """_patch_n_spins only modifies the N parameter, not EMA_ALPHA_NUM etc.

    Spec: REQ-HW-038
    """
    rtl = (
        "parameter integer N              = 64,\n"
        "parameter integer EMA_ALPHA_NUM  = 7,\n"
        "parameter integer EMA_ALPHA_DEN  = 8,\n"
    )
    patched = exp816._patch_n_spins(rtl, 32)
    assert "EMA_ALPHA_NUM  = 7" in patched
    assert "EMA_ALPHA_DEN  = 8" in patched


def test_patch_n_spins_n64() -> None:
    """_patch_n_spins correctly sets N=64 as well as N=32.

    Spec: REQ-HW-038
    """
    rtl = "parameter integer N              = 64,\n"
    assert "= 64" in exp816._patch_n_spins(rtl, 64)


# ---------------------------------------------------------------------------
# Gate logic — REQ-HW-038-3
# ---------------------------------------------------------------------------


def _make_tmpl(tmp_path: Path) -> "exp816.ExperimentTemplate":  # type: ignore[name-defined]
    from experiment_template import ExperimentTemplate  # noqa: PLC0415

    return ExperimentTemplate(
        exp_id=816,
        title="test",
        deliverable="results/experiment_816_kv260_synthesis_v2.json",
        requires_gpu=False,
        repo_root=tmp_path,
    )


def test_gate_blocks_when_verdict_invalid(tmp_path: Path) -> None:
    """run_experiment returns gated_tools_not_installed when Exp 807 verdict is invalid.

    Spec: REQ-HW-038-3
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "experiment_807_oss_cad_suite_install.json").write_text(
        json.dumps({"experiment": 807, "honest_verdict": "some_unknown_verdict"})
    )
    tmpl = _make_tmpl(tmp_path)
    data, status = exp816.run_experiment(tmpl)

    assert status == "blocked"
    assert data["honest_verdict"] == "gated_tools_not_installed"
    assert data["gate_passed"] is False


def test_gate_blocks_when_artifact_missing(tmp_path: Path) -> None:
    """run_experiment blocks when Exp 807 artifact does not exist.

    A missing artifact is treated conservatively as tools_not_installed.

    Spec: REQ-HW-038-3
    """
    tmpl = _make_tmpl(tmp_path)
    data, status = exp816.run_experiment(tmpl)

    assert status == "blocked"
    assert data["honest_verdict"] == "gated_tools_not_installed"
    assert data["gate_passed"] is False


def test_gate_passes_with_tools_installed_synthesis_clean(tmp_path: Path) -> None:
    """Gate passes when Exp 807 verdict is 'tools_installed_synthesis_clean'.

    Spec: REQ-HW-038-3
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "experiment_807_oss_cad_suite_install.json").write_text(
        json.dumps({"experiment": 807, "honest_verdict": "tools_installed_synthesis_clean"})
    )
    # Copy RTL so experiment doesn't block on missing source
    hw_dir = tmp_path / "hardware" / "kv260"
    hw_dir.mkdir(parents=True)
    real_rtl = REPO_ROOT / "hardware" / "kv260" / "ising_sampler_v3.v"
    if real_rtl.exists():
        (hw_dir / "ising_sampler_v3.v").write_text(real_rtl.read_text())
    else:
        (hw_dir / "ising_sampler_v3.v").write_text("module stub(); endmodule\n")

    tmpl = _make_tmpl(tmp_path)

    # Patch yosys binary check so the test does not require OSS-CAD installed.
    with patch.object(Path, "exists", return_value=True):
        with patch.object(exp816, "_run_yosys_synth", return_value=(True, 450, "SB_LUT4 450")):
            data, status = exp816.run_experiment(tmpl)

    assert data["gate_passed"] is True, "Gate must pass for tools_installed_synthesis_clean"


def test_gate_passes_with_already_installed(tmp_path: Path) -> None:
    """Gate passes when Exp 807 verdict is 'already_installed'.

    Spec: REQ-HW-038-3
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "experiment_807_oss_cad_suite_install.json").write_text(
        json.dumps({"experiment": 807, "honest_verdict": "already_installed"})
    )
    hw_dir = tmp_path / "hardware" / "kv260"
    hw_dir.mkdir(parents=True)
    (hw_dir / "ising_sampler_v3.v").write_text("module stub(); endmodule\n")

    tmpl = _make_tmpl(tmp_path)

    with patch.object(Path, "exists", return_value=True):
        with patch.object(exp816, "_run_yosys_synth", return_value=(True, 400, "SB_LUT4 400")):
            data, status = exp816.run_experiment(tmpl)

    assert data["gate_passed"] is True


# ---------------------------------------------------------------------------
# Blocked when yosys binary absent — REQ-HW-038-2
# ---------------------------------------------------------------------------


def test_blocked_when_yosys_binary_absent(tmp_path: Path) -> None:
    """run_experiment returns blocked_tools_not_at_expected_path when yosys absent.

    Spec: REQ-HW-038-2
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "experiment_807_oss_cad_suite_install.json").write_text(
        json.dumps({"experiment": 807, "honest_verdict": "tools_installed_synthesis_clean"})
    )
    hw_dir = tmp_path / "hardware" / "kv260"
    hw_dir.mkdir(parents=True)
    (hw_dir / "ising_sampler_v3.v").write_text("module stub(); endmodule\n")

    tmpl = _make_tmpl(tmp_path)

    # Do NOT patch Path.exists — the real OSS-CAD yosys path likely does not exist in CI.
    # If it does exist on this machine, force it to appear absent for the test.
    orig_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if "oss-cad-suite" in str(self) and self.name == "yosys":
            return False
        return orig_exists(self)

    with patch.object(Path, "exists", fake_exists):
        data, status = exp816.run_experiment(tmpl)

    assert status == "blocked"
    assert data["honest_verdict"] == "blocked_tools_not_at_expected_path"


# ---------------------------------------------------------------------------
# load_gate_artifact handles corrupt JSON (REQ-HW-038-3)
# ---------------------------------------------------------------------------


def test_load_gate_artifact_handles_corrupt_json(tmp_path: Path) -> None:
    """load_gate_artifact returns {} on corrupt JSON, causing gate to block.

    Spec: REQ-HW-038-3
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "experiment_807_oss_cad_suite_install.json").write_text(
        "{ not valid json }"
    )
    assert exp816.load_gate_artifact(tmp_path) == {}


def test_load_gate_artifact_handles_missing_file(tmp_path: Path) -> None:
    """load_gate_artifact returns {} when the file does not exist.

    Spec: REQ-HW-038-3
    """
    assert exp816.load_gate_artifact(tmp_path) == {}
