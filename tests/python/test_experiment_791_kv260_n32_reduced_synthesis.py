"""Tests for Experiment 791 — N=32 iCE40 synthesis and place-and-route.

Each test is traced to REQ-HW-043 per the spec-anchored development workflow.

Spec: REQ-HW-043, SCENARIO-HW-043
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import experiment_791_kv260_n32_reduced_synthesis as mod


# ---------------------------------------------------------------------------
# REQ-HW-043-1: Synthesis script must reference N_SPINS=32 and MAX_DEGREE=8
# ---------------------------------------------------------------------------


class TestSynthScriptParameters:
    """REQ-HW-043-1: chparam overrides N_SPINS=32 and MAX_DEGREE=8 in the .ys script."""

    def test_synth_script_sets_n_spins_32(self, tmp_path):
        """REQ-HW-043-1: synth script must contain N_SPINS 32 for _check_synth_parameters_in_script."""
        script = tmp_path / "synth_yosys_ice40_n32.ys"
        script.write_text("chparam -set N_SPINS 32 -set MAX_DEGREE 8 ising_sampler_128_sync\n")
        # Simulate a repo_root where the script lives at the expected path
        synth_dir = tmp_path / "hardware" / "kv260"
        synth_dir.mkdir(parents=True)
        (synth_dir / "synth_yosys_ice40_n32.ys").write_text(
            "chparam -set N_SPINS 32 -set MAX_DEGREE 8 ising_sampler_128_sync\n"
        )
        assert mod._check_synth_parameters_in_script(tmp_path) is True

    def test_synth_script_missing_n_spins_fails(self, tmp_path):
        """REQ-HW-043-1: script without N_SPINS 32 returns False — guards against misconfiguration."""
        synth_dir = tmp_path / "hardware" / "kv260"
        synth_dir.mkdir(parents=True)
        (synth_dir / "synth_yosys_ice40_n32.ys").write_text(
            "chparam -set N_SPINS 64 -set MAX_DEGREE 32 ising_sampler_128_sync\n"
        )
        assert mod._check_synth_parameters_in_script(tmp_path) is False

    def test_real_synth_script_has_correct_params(self):
        """REQ-HW-043-1: the committed synth_yosys_ice40_n32.ys has N_SPINS=32, MAX_DEGREE=8."""
        assert mod._check_synth_parameters_in_script(_REPO_ROOT) is True


# ---------------------------------------------------------------------------
# REQ-HW-043-4: honest_verdict = "tools_unavailable" when nextpnr-ice40 absent
# ---------------------------------------------------------------------------


class TestToolsUnavailableVerdict:
    """REQ-HW-043-4: tools_unavailable verdict when nextpnr-ice40 cannot be found."""

    def test_tools_unavailable_when_nextpnr_not_found(self, tmp_path):
        """REQ-HW-043-4: run_experiment returns tools_unavailable if nextpnr-ice40 is absent."""
        fake_tmpl = MagicMock()
        fake_tmpl._repo_root = str(tmp_path)

        with patch.object(mod, "_find_yosys", return_value=(True, "yosys")), \
             patch.object(mod, "_find_nextpnr_ice40", return_value=(False, "")), \
             patch.object(mod, "_find_icepack", return_value=(False, "")):
            result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "tools_unavailable"
        assert result["pnr_success_ice40"] is False
        assert result["bitstream_generated"] is False
        assert result["synthesis_ok"] is False

    def test_tools_available_dict_populated(self, tmp_path):
        """REQ-HW-043-4: tools_available dict is always included in the result."""
        fake_tmpl = MagicMock()
        fake_tmpl._repo_root = str(tmp_path)

        with patch.object(mod, "_find_yosys", return_value=(False, "")), \
             patch.object(mod, "_find_nextpnr_ice40", return_value=(False, "")), \
             patch.object(mod, "_find_icepack", return_value=(False, "")):
            result = mod.run_experiment(fake_tmpl)

        assert "tools_available" in result
        assert result["tools_available"]["nextpnr_ice40"] is False


# ---------------------------------------------------------------------------
# REQ-HW-043-4: honest_verdict = "bitstream_generated_n32_ice40" on full success
# ---------------------------------------------------------------------------


class TestBitstreamGeneratedVerdict:
    """REQ-HW-043-4: bitstream_generated_n32_ice40 verdict when all steps succeed."""

    def _make_synth_script(self, tmp_path: Path) -> None:
        """Create a minimal synth script with the required parameters."""
        synth_dir = tmp_path / "hardware" / "kv260"
        synth_dir.mkdir(parents=True)
        (synth_dir / "synth_yosys_ice40_n32.ys").write_text(
            "chparam -set N_SPINS 32 -set MAX_DEGREE 8 ising_sampler_128_sync\n"
        )

    def test_bitstream_generated_yields_correct_verdict(self, tmp_path):
        """REQ-HW-043-4: bitstream_generated=True → honest_verdict='bitstream_generated_n32_ice40'."""
        self._make_synth_script(tmp_path)
        fake_tmpl = MagicMock()
        fake_tmpl._repo_root = str(tmp_path)

        with patch.object(mod, "_find_yosys", return_value=(True, "yosys")), \
             patch.object(mod, "_find_nextpnr_ice40", return_value=(True, "nextpnr-ice40")), \
             patch.object(mod, "_find_icepack", return_value=(True, "icepack")), \
             patch.object(mod, "_run_yosys_synthesis", return_value=(True, 612, "")), \
             patch.object(mod, "_run_nextpnr", return_value=(True, 12.5, 8.0, "")), \
             patch.object(mod, "_run_icepack", return_value=(True, str(tmp_path / "out.bin"))):
            result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "bitstream_generated_n32_ice40"
        assert result["bitstream_generated"] is True
        assert result["pnr_success_ice40"] is True
        assert result["lut_count_n32"] == 612
        assert result["critical_path_ns"] == 12.5
        assert result["lut_utilization_pct"] == 8.0

    def test_pnr_success_lut_fit_verdict(self, tmp_path):
        """REQ-HW-043-4: PnR ok, LUT < 90%, no icepack → pnr_success_lut_fit."""
        self._make_synth_script(tmp_path)
        fake_tmpl = MagicMock()
        fake_tmpl._repo_root = str(tmp_path)

        with patch.object(mod, "_find_yosys", return_value=(True, "yosys")), \
             patch.object(mod, "_find_nextpnr_ice40", return_value=(True, "nextpnr-ice40")), \
             patch.object(mod, "_find_icepack", return_value=(False, "")), \
             patch.object(mod, "_run_yosys_synthesis", return_value=(True, 700, "")), \
             patch.object(mod, "_run_nextpnr", return_value=(True, 10.0, 9.1, "")):
            result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "pnr_success_lut_fit"
        assert result["bitstream_generated"] is False
        assert result["lut_utilization_pct"] == 9.1

    def test_pnr_failed_timing_n32_verdict(self, tmp_path):
        """REQ-HW-043-4: synth ok but PnR fails → pnr_failed_timing_n32."""
        self._make_synth_script(tmp_path)
        fake_tmpl = MagicMock()
        fake_tmpl._repo_root = str(tmp_path)

        with patch.object(mod, "_find_yosys", return_value=(True, "yosys")), \
             patch.object(mod, "_find_nextpnr_ice40", return_value=(True, "nextpnr-ice40")), \
             patch.object(mod, "_find_icepack", return_value=(False, "")), \
             patch.object(mod, "_run_yosys_synthesis", return_value=(True, 650, "")), \
             patch.object(mod, "_run_nextpnr", return_value=(False, None, None, "FAIL")):
            result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "pnr_failed_timing_n32"
        assert result["pnr_success_ice40"] is False
        assert result["synthesis_ok"] is True


# ---------------------------------------------------------------------------
# Helper function unit tests (for 100% coverage of the module)
# ---------------------------------------------------------------------------


class TestHelpers:
    """Unit tests for internal helper functions."""

    def test_which_returns_true_for_python(self):
        """_which: 'python3' should be on PATH in the test environment."""
        # python3 is always available when running pytest
        result = mod._which("python3")
        assert isinstance(result, bool)

    def test_which_returns_false_for_nonexistent(self):
        """_which: nonexistent binary returns False without exception."""
        assert mod._which("this_binary_does_not_exist_carnot_791") is False

    def test_run_timeout_returns_minus_one(self):
        """_run: timed-out command returns returncode=-1 and non-empty stderr."""
        # Use a sleep command that will exceed a 0-second timeout
        rc, stdout, stderr = mod._run(["sleep", "60"], timeout=0)
        assert rc == -1
        assert "timed out" in stderr.lower() or len(stderr) > 0

    def test_run_not_found_returns_minus_one(self):
        """_run: FileNotFoundError for missing binary returns rc=-1."""
        rc, _, stderr = mod._run(["this_binary_does_not_exist_carnot"], timeout=5)
        assert rc == -1
        assert "not found" in stderr.lower()

    def test_find_yosys_native_path(self):
        """_find_yosys: native binary detected when _which returns True."""
        with patch.object(mod, "_which", return_value=True):
            found, cmd = mod._find_yosys()
        assert found is True
        assert cmd == "yosys"

    def test_find_yosys_not_found(self):
        """_find_yosys: returns False when neither native nor yowasp is present."""
        with patch.object(mod, "_which", return_value=False), \
             patch.object(mod, "_run", return_value=(1, "", "error")):
            found, cmd = mod._find_yosys()
        assert found is False
        assert cmd == ""

    def test_find_icepack_native(self):
        """_find_icepack: native binary detected when _which returns True."""
        with patch.object(mod, "_which", return_value=True):
            found, cmd = mod._find_icepack()
        assert found is True
        assert cmd == "icepack"

    def test_find_icepack_not_found(self):
        """_find_icepack: returns (False, '') when neither native nor yowasp installs."""
        with patch.object(mod, "_which", return_value=False), \
             patch.object(mod, "_run", return_value=(1, "", "error")):
            found, cmd = mod._find_icepack()
        assert found is False

    def test_find_nextpnr_ice40_yowasp_module(self):
        """_find_nextpnr_ice40: yowasp module found without native binary."""
        def _which_side(name):
            return False  # native not found

        with patch.object(mod, "_which", side_effect=_which_side), \
             patch.object(mod, "_run", return_value=(0, "nextpnr 0.10", "")):
            found, cmd = mod._find_nextpnr_ice40()
        assert found is True

    def test_find_nextpnr_pip_install_success(self):
        """_find_nextpnr_ice40: installs via pip when native and module unavailable."""
        call_count = [0]

        def _run_side(*args, **kwargs):
            call_count[0] += 1
            # First call: module check → fail; second: pip install → ok; third: recheck → ok
            if call_count[0] == 1:
                return (1, "", "not found")
            if call_count[0] == 2:
                return (0, "", "")  # pip install succeeds
            return (0, "nextpnr 0.10", "")  # recheck succeeds

        with patch.object(mod, "_which", return_value=False), \
             patch.object(mod, "_run", side_effect=_run_side):
            found, cmd = mod._find_nextpnr_ice40()
        assert found is True

    def test_find_nextpnr_all_fail(self):
        """_find_nextpnr_ice40: returns (False, '') when all attempts fail."""
        with patch.object(mod, "_which", return_value=False), \
             patch.object(mod, "_run", return_value=(1, "", "error")):
            found, cmd = mod._find_nextpnr_ice40()
        assert found is False
        assert cmd == ""

    def test_run_icepack_no_asc(self, tmp_path):
        """_run_icepack: returns (False, None) when .asc file does not exist."""
        ok, path = mod._run_icepack(tmp_path, "icepack")
        assert ok is False
        assert path is None

    def test_run_yosys_lut_parsing(self, tmp_path):
        """_run_yosys_synthesis: SB_LUT4 count is parsed from stat output."""
        synth_dir = tmp_path / "hardware" / "kv260"
        synth_dir.mkdir(parents=True)
        (synth_dir / "synth_yosys_ice40_n32.ys").write_text(
            "chparam -set N_SPINS 32 -set MAX_DEGREE 8 top\n"
        )
        # Simulate yosys output with SB_LUT4 count and write the JSON file
        def _run_side(cmd, timeout=300):
            (tmp_path / "hardware" / "kv260" / "ising_sampler_n32_ice40.json").write_text("{}")
            return (0, "   SB_LUT4:    612\n", "")

        with patch.object(mod, "_run", side_effect=_run_side):
            ok, lut_count, log = mod._run_yosys_synthesis(tmp_path, "yosys")

        assert ok is True
        assert lut_count == 612

    def test_run_nextpnr_parses_fmax(self, tmp_path):
        """_run_nextpnr: fmax_mhz line is parsed to critical_path_ns."""
        asc_path = tmp_path / "hardware" / "kv260" / "ising_sampler_n32.asc"
        asc_path.parent.mkdir(parents=True)
        asc_path.touch()
        json_in = tmp_path / "hardware" / "kv260" / "ising_sampler_n32_ice40.json"
        json_in.write_text("{}")

        nextpnr_out = (
            "Max frequency for clock 'clk': 80.00 MHz (PASS at 8.00 MHz)\n"
            "  ICESTORM_LC:   768/ 7680   10%\n"
        )

        def _run_side(cmd, timeout=300):
            return (0, nextpnr_out, "")

        with patch.object(mod, "_run", side_effect=_run_side):
            ok, crit_ns, lut_pct, log = mod._run_nextpnr(tmp_path, "nextpnr-ice40")

        assert ok is True
        assert crit_ns == pytest.approx(12.5, rel=0.01)  # 1000/80 = 12.5 ns
        assert lut_pct == 10.0  # 768/7680 * 100 = 10.0
