"""Tests for Experiment 776 — nextpnr place-and-route of Ising sampler.

Spec: REQ-HW-042, SCENARIO-HW-042
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Allow running from repo root
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import experiment_776_kv260_nextpnr_synthesis as mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exp758_json(synthesis_errors: int = 0, status: str = "success") -> str:
    return json.dumps({"synthesis_errors": synthesis_errors, "status": status})


# ---------------------------------------------------------------------------
# REQ-HW-042: blocked gate when Exp 758 synthesis failed
# ---------------------------------------------------------------------------


class TestExp758Gate:
    """REQ-HW-042: run_experiment must return blocked_yosys_synthesis_failed when
    Exp 758 result is absent or reports synthesis_errors > 0."""

    def test_blocks_when_exp758_missing(self, tmp_path):
        """REQ-HW-042: absent Exp 758 result → blocked_yosys_synthesis_failed."""
        # _check_exp758_success returns False when the file does not exist.
        result = mod._check_exp758_success(tmp_path)
        assert result is False

    def test_blocks_when_exp758_has_errors(self, tmp_path):
        """REQ-HW-042: synthesis_errors=1 → _check_exp758_success returns False."""
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "experiment_758_yosys_synthesis.json").write_text(
            _make_exp758_json(synthesis_errors=1)
        )
        assert mod._check_exp758_success(tmp_path) is False

    def test_passes_when_exp758_clean(self, tmp_path):
        """REQ-HW-042: synthesis_errors=0 + status=success → True."""
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "experiment_758_yosys_synthesis.json").write_text(
            _make_exp758_json(synthesis_errors=0)
        )
        assert mod._check_exp758_success(tmp_path) is True

    def test_run_experiment_returns_blocked_verdict(self, tmp_path):
        """REQ-HW-042: run_experiment emits honest_verdict=blocked_yosys_synthesis_failed
        when Exp 758 result is absent — no nextpnr tools are invoked."""
        # No Exp 758 file written — _check_exp758_success returns False.
        fake_tmpl = MagicMock()
        fake_tmpl._repo_root = str(tmp_path)

        result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "blocked_yosys_synthesis_failed"
        assert result["pnr_success_ice40"] is False
        assert result["bitstream_generated"] is False


# ---------------------------------------------------------------------------
# REQ-HW-042: nextpnr_ice40_found detection
# ---------------------------------------------------------------------------


class TestNextpnrIce40Found:
    """REQ-HW-042: _find_nextpnr_ice40 reads subprocess result correctly."""

    def test_native_binary_found(self):
        """REQ-HW-042: _which('nextpnr-ice40')=True → found=True, cmd='nextpnr-ice40'."""
        with patch.object(mod, "_which", return_value=True):
            found, cmd = mod._find_nextpnr_ice40()
        assert found is True
        assert cmd == "nextpnr-ice40"

    def test_yowasp_module_found(self):
        """REQ-HW-042: native absent but yowasp module importable → found=True."""
        def _which_side(name):
            return name != "nextpnr-ice40"

        with patch.object(mod, "_which", side_effect=_which_side), \
             patch.object(mod, "_run", return_value=(0, "nextpnr 0.10", "")):
            found, cmd = mod._find_nextpnr_ice40()
        assert found is True

    def test_not_found_when_pip_fails(self):
        """REQ-HW-042: native absent + pip install fails → found=False."""
        with patch.object(mod, "_which", return_value=False), \
             patch.object(mod, "_run", return_value=(1, "", "error")):
            found, cmd = mod._find_nextpnr_ice40()
        assert found is False
        assert cmd == ""


# ---------------------------------------------------------------------------
# REQ-HW-042: honest_verdict mapping
# ---------------------------------------------------------------------------


class TestHonestVerdictMapping:
    """REQ-HW-042: honest_verdict correctly maps bitstream_generated to verdict string."""

    def _make_exp758(self, tmp_path: Path) -> None:
        (tmp_path / "results").mkdir(exist_ok=True)
        (tmp_path / "results" / "experiment_758_yosys_synthesis.json").write_text(
            _make_exp758_json()
        )

    def test_bitstream_generated_yields_ice40_verdict(self, tmp_path):
        """REQ-HW-042: bitstream_generated=True → honest_verdict='bitstream_generated_ice40'."""
        self._make_exp758(tmp_path)
        fake_tmpl = MagicMock()
        fake_tmpl._repo_root = str(tmp_path)

        # Stub the full pipeline: nextpnr found, synthesis ok, PnR ok, icepack ok.
        with patch.object(mod, "_find_nextpnr_ice40", return_value=(True, "nextpnr-ice40")), \
             patch.object(mod, "_find_yosys", return_value=(True, "yosys")), \
             patch.object(mod, "_find_icepack", return_value=(True, "icepack")), \
             patch.object(mod, "_which", return_value=False), \
             patch.object(mod, "_run_ice40_synthesis", return_value=(True, "")), \
             patch.object(mod, "_run_nextpnr_ice40", return_value=(True, 26.3, 38.5, "")), \
             patch.object(mod, "_run_icepack", return_value=(True, str(tmp_path / "out.bin"))):
            result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "bitstream_generated_ice40"
        assert result["bitstream_generated"] is True
        assert result["pnr_success_ice40"] is True
        assert result["critical_path_ns"] == 26.3
        assert result["lut_utilization_pct"] == 38.5

    def test_pnr_success_no_icepack_yields_no_bitstream_verdict(self, tmp_path):
        """REQ-HW-042: PnR ok but icepack unavailable → pnr_successful_no_bitstream."""
        self._make_exp758(tmp_path)
        fake_tmpl = MagicMock()
        fake_tmpl._repo_root = str(tmp_path)

        with patch.object(mod, "_find_nextpnr_ice40", return_value=(True, "nextpnr-ice40")), \
             patch.object(mod, "_find_yosys", return_value=(True, "yosys")), \
             patch.object(mod, "_find_icepack", return_value=(False, "")), \
             patch.object(mod, "_which", return_value=False), \
             patch.object(mod, "_run_ice40_synthesis", return_value=(True, "")), \
             patch.object(mod, "_run_nextpnr_ice40", return_value=(True, 30.0, 40.0, "")):
            result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "pnr_successful_no_bitstream"
        assert result["bitstream_generated"] is False

    def test_pnr_failed_timing_verdict(self, tmp_path):
        """REQ-HW-042: PnR returns False with critical_path_ns → pnr_failed_timing."""
        self._make_exp758(tmp_path)
        fake_tmpl = MagicMock()
        fake_tmpl._repo_root = str(tmp_path)

        with patch.object(mod, "_find_nextpnr_ice40", return_value=(True, "nextpnr-ice40")), \
             patch.object(mod, "_find_yosys", return_value=(True, "yosys")), \
             patch.object(mod, "_find_icepack", return_value=(False, "")), \
             patch.object(mod, "_which", return_value=False), \
             patch.object(mod, "_run_ice40_synthesis", return_value=(True, "")), \
             patch.object(mod, "_run_nextpnr_ice40", return_value=(False, 55.0, None, "FAIL")):
            result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "pnr_failed_timing"
        assert result["pnr_success_ice40"] is False

    def test_nextpnr_not_installable_verdict(self, tmp_path):
        """REQ-HW-042: nextpnr_ice40_found=False → honest_verdict='nextpnr_not_installable'."""
        self._make_exp758(tmp_path)
        fake_tmpl = MagicMock()
        fake_tmpl._repo_root = str(tmp_path)

        with patch.object(mod, "_find_nextpnr_ice40", return_value=(False, "")), \
             patch.object(mod, "_find_yosys", return_value=(True, "yosys")), \
             patch.object(mod, "_find_icepack", return_value=(False, "")), \
             patch.object(mod, "_which", return_value=False):
            result = mod.run_experiment(fake_tmpl)

        assert result["honest_verdict"] == "nextpnr_not_installable"
        assert result["nextpnr_ice40_found"] is False
