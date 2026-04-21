"""Tests for Exp 636: FPGA TCL v2 Update.

100% targeted coverage on functions added in
scripts/experiment_636_fpga_tcl_v2.py:
  - check_tcl_v2_content()
  - run_synthesis()
  - run_python_simulation()
  - main()

All tests run without Vivado hardware by mocking subprocess calls.

Spec: REQ-SAMPLE-039, SCENARIO-SAMPLE-065
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

os.environ["CARNOT_IS_CI"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import scripts.experiment_636_fpga_tcl_v2 as exp636  # noqa: E402


# ---------------------------------------------------------------------------
# check_tcl_v2_content
# ---------------------------------------------------------------------------


class TestCheckTclV2Content:
    """REQ-SAMPLE-039: synth_ising_v2.tcl exists and targets v2 correctly."""

    def test_tcl_file_missing(self, tmp_path: Path) -> None:
        # SCENARIO: file does not exist — all flags False.
        with patch.object(exp636, "TCL_V2_PATH", str(tmp_path / "missing.tcl")):
            result = exp636.check_tcl_v2_content()
        assert result["tcl_exists"] is False
        assert result["top_module_ok"] is False
        assert result["rtl_file_ok"] is False
        assert result["output_dir_ok"] is False

    def test_tcl_file_correct_content(self, tmp_path: Path) -> None:
        # SCENARIO-SAMPLE-065: file exists with all required v2 strings.
        tcl = tmp_path / "synth_ising_v2.tcl"
        tcl.write_text(
            'set top_module "ising_sampler_128_sync"\n'
            'set rtl_files [list "hardware/kv260/ising_sampler_v2.v"]\n'
            'set output_dir "output/carnot_ising_synth_v2"\n'
        )
        with patch.object(exp636, "TCL_V2_PATH", str(tcl)):
            result = exp636.check_tcl_v2_content()
        assert result["tcl_exists"] is True
        assert result["top_module_ok"] is True
        assert result["rtl_file_ok"] is True
        assert result["output_dir_ok"] is True

    def test_tcl_file_v1_content(self, tmp_path: Path) -> None:
        # SCENARIO: file exists but still points to v1 — flags reflect reality.
        tcl = tmp_path / "synth_ising_v2.tcl"
        tcl.write_text('set top_module "ising_sampler_128"\n')
        with patch.object(exp636, "TCL_V2_PATH", str(tcl)):
            result = exp636.check_tcl_v2_content()
        assert result["tcl_exists"] is True
        assert result["top_module_ok"] is False
        assert result["rtl_file_ok"] is False
        assert result["output_dir_ok"] is False


# ---------------------------------------------------------------------------
# run_synthesis
# ---------------------------------------------------------------------------


class TestRunSynthesis:
    """REQ-SAMPLE-039: Synthesis invokes Vivado with correct arguments."""

    def test_synthesis_success(self, tmp_path: Path) -> None:
        # SCENARIO: Vivado returns 0 and bitfile appears on disk.
        bitfile = tmp_path / "carnot_ising_v2.bit"
        bitfile.touch()
        util_rpt = tmp_path / "utilization.rpt"
        util_rpt.touch()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = ""

        with (
            patch("scripts.experiment_636_fpga_tcl_v2.subprocess.run",
                  return_value=mock_proc) as mock_run,
            patch.object(exp636, "BITFILE_PATH", str(bitfile)),
            patch.object(exp636, "VIVADO_OUTPUT_DIR", str(tmp_path)),
        ):
            result = exp636.run_synthesis("hardware/kv260/synth_ising_v2.tcl")

        assert result["synthesis_succeeded"] is True
        assert result["returncode"] == 0
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "vivado" in call_args
        assert "-mode" in call_args
        assert "batch" in call_args

    def test_synthesis_vivado_fails(self, tmp_path: Path) -> None:
        # SCENARIO: Vivado returns non-zero — synthesis_succeeded=False.
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "ERROR: some synthesis error"

        with (
            patch("scripts.experiment_636_fpga_tcl_v2.subprocess.run",
                  return_value=mock_proc),
            patch.object(exp636, "BITFILE_PATH", str(tmp_path / "missing.bit")),
            patch.object(exp636, "VIVADO_OUTPUT_DIR", str(tmp_path)),
        ):
            result = exp636.run_synthesis("hardware/kv260/synth_ising_v2.tcl")

        assert result["synthesis_succeeded"] is False

    def test_synthesis_timeout(self) -> None:
        # SCENARIO: Vivado times out — synthesis_succeeded=False, error recorded.
        import subprocess as sp
        with patch(
            "scripts.experiment_636_fpga_tcl_v2.subprocess.run",
            side_effect=sp.TimeoutExpired(cmd="vivado", timeout=3600),
        ):
            result = exp636.run_synthesis("hardware/kv260/synth_ising_v2.tcl",
                                          timeout_s=3600)
        assert result["synthesis_succeeded"] is False
        assert "timeout" in result["error"]

    def test_synthesis_exception(self) -> None:
        # SCENARIO: Vivado not found — OSError captured and recorded.
        with patch(
            "scripts.experiment_636_fpga_tcl_v2.subprocess.run",
            side_effect=OSError("vivado: not found"),
        ):
            result = exp636.run_synthesis("hardware/kv260/synth_ising_v2.tcl")
        assert result["synthesis_succeeded"] is False
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# run_python_simulation
# ---------------------------------------------------------------------------


class TestRunPythonSimulation:
    """REQ-SAMPLE-037: SynchronousIsingSampler provides simulation baseline."""

    def test_returns_expected_keys(self) -> None:
        # SCENARIO-SAMPLE-065: simulation returns sync and async energy keys.
        result = exp636.run_python_simulation()
        assert "sync_mean_energy" in result
        assert "async_mean_energy" in result

    def test_energy_values_are_finite(self) -> None:
        # SCENARIO: energy values are finite floats (not NaN or inf).
        result = exp636.run_python_simulation()
        import math
        assert math.isfinite(result["sync_mean_energy"])
        assert math.isfinite(result["async_mean_energy"])


# ---------------------------------------------------------------------------
# main — integration
# ---------------------------------------------------------------------------


class TestMain:
    """Integration: main() produces deliverable JSON with correct schema."""

    def _make_sim_result(self) -> dict:
        return {"sync_mean_energy": 0.01, "async_mean_energy": -0.32,
                "energy_gap": 0.33}

    def test_main_vivado_absent(self, tmp_path: Path) -> None:
        # SCENARIO: Vivado not installed — honest_verdict=tcl_updated_synthesis_deferred.
        deliverable = tmp_path / "experiment_636_fpga_tcl_v2.json"
        tcl = tmp_path / "synth_ising_v2.tcl"
        tcl.write_text(
            'set top_module "ising_sampler_128_sync"\n'
            'set rtl_files [list "hardware/kv260/ising_sampler_v2.v"]\n'
            'set output_dir "output/carnot_ising_synth_v2"\n'
        )

        mock_vivado_fail = MagicMock()
        mock_vivado_fail.returncode = 1

        with (
            patch.object(exp636, "DELIVERABLE", str(deliverable)),
            patch.object(exp636, "TCL_V2_PATH", str(tcl)),
            patch("scripts.experiment_636_fpga_tcl_v2.subprocess.run",
                  return_value=mock_vivado_fail),
            patch.object(exp636, "run_python_simulation",
                         return_value=self._make_sim_result()),
        ):
            exp636.main()

        assert deliverable.exists()
        artifact = json.loads(deliverable.read_text())
        assert artifact["experiment"] == 636
        assert artifact["artifact_type"] == "carnot.fpga_tcl_v2.v1"
        assert isinstance(artifact["schema"], list)
        assert artifact["vivado_installed"] is False
        assert artifact["synthesis_succeeded"] == "not_attempted"
        assert artifact["honest_verdict"] == "tcl_updated_synthesis_deferred"
        assert artifact["simulation_validated"] is True
        assert artifact["est_lut_reduction"] == 0.50

    def test_main_vivado_present_success(self, tmp_path: Path) -> None:
        # SCENARIO: Vivado installed and synthesis succeeds — honest_verdict=synthesis_complete.
        deliverable = tmp_path / "experiment_636_fpga_tcl_v2.json"
        tcl = tmp_path / "synth_ising_v2.tcl"
        tcl.write_text(
            'set top_module "ising_sampler_128_sync"\n'
            'set rtl_files [list "hardware/kv260/ising_sampler_v2.v"]\n'
            'set output_dir "output/carnot_ising_synth_v2"\n'
        )
        bitfile = tmp_path / "carnot_ising_v2.bit"
        bitfile.touch()

        mock_vivado_ok = MagicMock()
        mock_vivado_ok.returncode = 0

        synth_result_ok = {
            "synthesis_succeeded": True,
            "returncode": 0,
            "utilization_report": None,
            "timing_report": None,
            "stderr_tail": "",
        }

        with (
            patch.object(exp636, "DELIVERABLE", str(deliverable)),
            patch.object(exp636, "TCL_V2_PATH", str(tcl)),
            patch.object(exp636, "BITFILE_PATH", str(bitfile)),
            patch.object(exp636, "VIVADO_OUTPUT_DIR", str(tmp_path)),
            patch("scripts.experiment_636_fpga_tcl_v2.subprocess.run",
                  return_value=mock_vivado_ok),
            patch.object(exp636, "run_synthesis", return_value=synth_result_ok),
            patch.object(exp636, "run_python_simulation",
                         return_value=self._make_sim_result()),
        ):
            exp636.main()

        artifact = json.loads(deliverable.read_text())
        assert artifact["honest_verdict"] == "synthesis_complete"
        assert artifact["synthesis_succeeded"] is True

    def test_main_vivado_present_failed(self, tmp_path: Path) -> None:
        # SCENARIO: Vivado installed but synthesis fails — honest_verdict=synthesis_attempted_failed.
        deliverable = tmp_path / "experiment_636_fpga_tcl_v2.json"
        tcl = tmp_path / "synth_ising_v2.tcl"
        tcl.write_text("dummy content")

        mock_vivado_ok = MagicMock()
        mock_vivado_ok.returncode = 0

        synth_result_fail = {
            "synthesis_succeeded": False,
            "returncode": 1,
            "utilization_report": None,
            "timing_report": None,
            "stderr_tail": "ERROR: constraint unsatisfied",
        }

        with (
            patch.object(exp636, "DELIVERABLE", str(deliverable)),
            patch.object(exp636, "TCL_V2_PATH", str(tcl)),
            patch("scripts.experiment_636_fpga_tcl_v2.subprocess.run",
                  return_value=mock_vivado_ok),
            patch.object(exp636, "run_synthesis", return_value=synth_result_fail),
            patch.object(exp636, "run_python_simulation",
                         return_value=self._make_sim_result()),
        ):
            exp636.main()

        artifact = json.loads(deliverable.read_text())
        assert artifact["honest_verdict"] == "synthesis_attempted_failed"
        assert artifact["synthesis_succeeded"] is False

    def test_artifact_required_fields_present(self, tmp_path: Path) -> None:
        # SCENARIO: All REQUIRED_RESULT_FIELDS are present in the artifact.
        from scripts.experiment_template import REQUIRED_RESULT_FIELDS

        deliverable = tmp_path / "experiment_636_fpga_tcl_v2.json"
        tcl = tmp_path / "synth_ising_v2.tcl"
        tcl.write_text("dummy")

        mock_fail = MagicMock()
        mock_fail.returncode = 1

        with (
            patch.object(exp636, "DELIVERABLE", str(deliverable)),
            patch.object(exp636, "TCL_V2_PATH", str(tcl)),
            patch("scripts.experiment_636_fpga_tcl_v2.subprocess.run",
                  return_value=mock_fail),
            patch.object(exp636, "run_python_simulation",
                         return_value=self._make_sim_result()),
        ):
            exp636.main()

        artifact = json.loads(deliverable.read_text())
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"
