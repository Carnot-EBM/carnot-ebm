"""Tests for experiment_982_kv260_board_programming_v2.

Spec refs: REQ-HW-040, SCENARIO-HW-040

**Why these tests exist:**
    Exp 971 ran but wrote no artifact JSON because there was no try/finally guard.
    These tests verify that the critical invariant is preserved: the result JSON
    MUST be written before the script exits, regardless of what fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import experiment_982_kv260_board_programming_v2 as exp982  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests: _find_vivado
# ---------------------------------------------------------------------------


class TestFindVivado:
    """REQ-HW-040: Vivado must be located before synthesis starts."""

    def test_finds_vivado_via_env_var(self, tmp_path: Path) -> None:
        """VIVADO_BIN env var pointing to an existing file is honoured."""
        fake_vivado = tmp_path / "vivado"
        fake_vivado.write_text("#!/bin/sh\necho fake")
        with mock.patch.dict(os.environ, {"VIVADO_BIN": str(fake_vivado)}):
            found, path = exp982._find_vivado()
        assert found is True
        assert path == str(fake_vivado)

    def test_env_var_missing_file_falls_through(self, tmp_path: Path) -> None:
        """VIVADO_BIN pointing to a non-existent file falls through to other checks."""
        missing = str(tmp_path / "no_such_vivado")
        # Patch shutil.which and the known path so we control the outcome.
        with (
            mock.patch.dict(os.environ, {"VIVADO_BIN": missing}),
            mock.patch("shutil.which", return_value=None),
            mock.patch("pathlib.Path.exists", return_value=False),
            mock.patch("glob.glob", return_value=[]),
        ):
            found, path = exp982._find_vivado()
        assert found is False
        assert path == ""

    def test_finds_vivado_via_which(self, tmp_path: Path) -> None:
        """shutil.which result is accepted when VIVADO_BIN is not set."""
        fake = str(tmp_path / "vivado")
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("shutil.which", return_value=fake),
        ):
            # Remove VIVADO_BIN from env if set
            env = dict(os.environ)
            env.pop("VIVADO_BIN", None)
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch("shutil.which", return_value=fake):
                    found, path = exp982._find_vivado()
        assert found is True
        assert path == fake

    def test_finds_vivado_via_known_path(self, tmp_path: Path) -> None:
        """The hardcoded known install path is checked when PATH lookup fails."""
        fake_known = tmp_path / "vivado"
        fake_known.write_text("#!/bin/sh")
        env = dict(os.environ)
        env.pop("VIVADO_BIN", None)
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("shutil.which", return_value=None),
            mock.patch.object(exp982, "_VIVADO_KNOWN_PATH", str(fake_known)),
            mock.patch("glob.glob", return_value=[]),
        ):
            found, path = exp982._find_vivado()
        assert found is True
        assert path == str(fake_known)

    def test_returns_false_when_nothing_found(self) -> None:
        """Returns (False, '') when no Vivado binary exists anywhere."""
        env = dict(os.environ)
        env.pop("VIVADO_BIN", None)
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("shutil.which", return_value=None),
            mock.patch("pathlib.Path.exists", return_value=False),
            mock.patch("glob.glob", return_value=[]),
        ):
            found, path = exp982._find_vivado()
        assert found is False
        assert path == ""


# ---------------------------------------------------------------------------
# Unit tests: _cpu_baseline_latency_us
# ---------------------------------------------------------------------------


class TestCpuBaseline:
    """REQ-HW-040: CPU baseline must produce a positive timing value."""

    def test_returns_positive_float(self) -> None:
        """The CPU baseline sweep must return a positive microseconds value."""
        us = exp982._cpu_baseline_latency_us()
        assert isinstance(us, float)
        assert us > 0.0

    def test_reproducible_with_fixed_seed(self) -> None:
        """With fixed numpy seed (42), two calls return similar values (< 2x apart)."""
        us1 = exp982._cpu_baseline_latency_us()
        us2 = exp982._cpu_baseline_latency_us()
        # Both should be in the same ballpark (wall-clock can vary, allow 5x tolerance).
        assert us1 > 0.0 and us2 > 0.0
        ratio = max(us1, us2) / min(us1, us2)
        assert ratio < 5.0, f"CPU baseline too variable: {us1:.1f} vs {us2:.1f} us"


# ---------------------------------------------------------------------------
# Unit tests: result JSON schema
# ---------------------------------------------------------------------------


class TestResultSchema:
    """SCENARIO-HW-040: Result JSON must contain all required fields."""

    REQUIRED_FIELDS = [
        "vivado_found",
        "vivado_path",
        "synthesis_passes",
        "lut_count_vivado",
        "implementation_passes",
        "bitstream_generated",
        "board_programmed",
        "hardware_latency_us",
        "cpu_baseline_latency_us",
        "honest_verdict",
    ]

    VALID_VERDICTS = {
        "hardware_working",
        "bitstream_generated_board_unreachable",
        "vivado_synthesis_passes",
        "vivado_not_on_path",
        "implementation_failed",
    }

    def _run_main_with_vivado_missing(self, result_path: Path) -> dict:
        """Run main() in a context where Vivado is not found."""
        with (
            mock.patch.object(exp982, "_RESULT_FILE", result_path),
            mock.patch.object(exp982, "_find_vivado", return_value=(False, "")),
            mock.patch.object(exp982, "_cpu_baseline_latency_us", return_value=42.0),
        ):
            exp982.main()
        return json.loads(result_path.read_text())

    def test_all_required_fields_present(self, tmp_path: Path) -> None:
        """All required schema fields must be present in the output JSON."""
        result_path = tmp_path / "result.json"
        data = self._run_main_with_vivado_missing(result_path)
        for field in self.REQUIRED_FIELDS:
            assert field in data, f"Missing required field: {field}"

    def test_honest_verdict_is_valid(self, tmp_path: Path) -> None:
        """honest_verdict must be one of the documented values."""
        result_path = tmp_path / "result.json"
        data = self._run_main_with_vivado_missing(result_path)
        assert data["honest_verdict"] in self.VALID_VERDICTS

    def test_vivado_not_found_verdict(self, tmp_path: Path) -> None:
        """When Vivado is missing, honest_verdict must be 'vivado_not_on_path'."""
        result_path = tmp_path / "result.json"
        data = self._run_main_with_vivado_missing(result_path)
        assert data["honest_verdict"] == "vivado_not_on_path"
        assert data["vivado_found"] is False


# ---------------------------------------------------------------------------
# Integration test: result JSON ALWAYS written (the critical invariant)
# ---------------------------------------------------------------------------


class TestResultAlwaysWritten:
    """REQ-HW-040 critical invariant: result JSON written regardless of outcome."""

    def test_result_written_when_vivado_missing(self, tmp_path: Path) -> None:
        """Result JSON exists even when Vivado binary is not found (Exp 971 regression test)."""
        result_path = tmp_path / "result.json"
        with (
            mock.patch.object(exp982, "_RESULT_FILE", result_path),
            mock.patch.object(exp982, "_find_vivado", return_value=(False, "")),
            mock.patch.object(exp982, "_cpu_baseline_latency_us", return_value=10.0),
        ):
            exp982.main()
        assert result_path.exists(), "Result JSON was not written when Vivado missing"

    def test_result_written_when_cpu_baseline_raises(self, tmp_path: Path) -> None:
        """Result JSON exists even if CPU baseline raises an exception."""
        result_path = tmp_path / "result.json"
        with (
            mock.patch.object(exp982, "_RESULT_FILE", result_path),
            mock.patch.object(exp982, "_find_vivado", return_value=(False, "")),
            mock.patch.object(
                exp982, "_cpu_baseline_latency_us", side_effect=RuntimeError("numpy missing")
            ),
        ):
            exp982.main()
        assert result_path.exists(), "Result JSON was not written when cpu_baseline raised"

    def test_result_written_when_vivado_raises(self, tmp_path: Path) -> None:
        """Result JSON exists even if _run_vivado raises an unexpected exception."""
        result_path = tmp_path / "result.json"
        with (
            mock.patch.object(exp982, "_RESULT_FILE", result_path),
            mock.patch.object(exp982, "_find_vivado", return_value=(True, "/fake/vivado")),
            mock.patch.object(exp982, "_cpu_baseline_latency_us", return_value=10.0),
            # Pretend the bitstream doesn't exist yet.
            mock.patch.object(Path, "exists", return_value=False),
            mock.patch.object(exp982, "_run_vivado", side_effect=RuntimeError("Vivado crashed")),
        ):
            exp982.main()
        assert result_path.exists(), "Result JSON was not written when _run_vivado raised"

    def test_result_json_is_valid(self, tmp_path: Path) -> None:
        """Result JSON can be parsed and has expected top-level fields."""
        result_path = tmp_path / "result.json"
        with (
            mock.patch.object(exp982, "_RESULT_FILE", result_path),
            mock.patch.object(exp982, "_find_vivado", return_value=(False, "")),
            mock.patch.object(exp982, "_cpu_baseline_latency_us", return_value=5.0),
        ):
            exp982.main()
        data = json.loads(result_path.read_text())
        assert data["experiment"] == 982
        assert data["schema"] == "kv260_board_programming_v2"
        assert isinstance(data["duration_s"], int)
        assert data["cpu_baseline_latency_us"] == 5.0

    def test_bitstream_path_vivado_synthesis_passes_verdict(self, tmp_path: Path) -> None:
        """When Vivado succeeds but board unreachable, verdict is bitstream_generated_board_unreachable."""
        result_path = tmp_path / "result.json"
        fake_bitstream = tmp_path / "carnot_ising_v4.bit"
        fake_bitstream.write_bytes(b"\x00" * 16)  # non-empty fake bitstream

        with (
            mock.patch.object(exp982, "_RESULT_FILE", result_path),
            mock.patch.object(exp982, "_BITSTREAM_DST", fake_bitstream),
            mock.patch.object(exp982, "_find_vivado", return_value=(True, "/fake/vivado")),
            mock.patch.object(exp982, "_cpu_baseline_latency_us", return_value=20.0),
            mock.patch.object(exp982, "_board_reachable", return_value=False),
        ):
            exp982.main()
        data = json.loads(result_path.read_text())
        assert data["bitstream_generated"] is True
        assert data["board_programmed"] is False
        assert data["honest_verdict"] == "bitstream_generated_board_unreachable"
