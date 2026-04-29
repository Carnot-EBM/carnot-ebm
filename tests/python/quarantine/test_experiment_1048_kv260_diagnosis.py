"""Tests for Exp 1048: KV260 UIO device diagnosis and energy distribution helpers.

Spec coverage: REQ-SAMPLE-036, SCENARIO-SAMPLE-060

These tests cover the pure-Python logic in experiment_1048 without requiring
board hardware. The UIO read path is tested via the energy_distribution logic
and the run_on_board plumbing is tested with mocked subprocess.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the experiment module without executing main()
# ---------------------------------------------------------------------------

_SCRIPT = (
    Path(__file__).parent.parent.parent / "scripts" / "experiment_1048_kv260_v4_dtbo_rebuild.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("exp1048", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


exp1048 = _load_module()


# ---------------------------------------------------------------------------
# Tests: pure-Python helpers (no board required)
# ---------------------------------------------------------------------------


class TestCheckDtboV4Exists:
    """REQ-SAMPLE-036: v4 DTBO check works when directory is absent."""

    def test_returns_false_when_no_dtbo(self, tmp_path, monkeypatch):
        """SCENARIO-SAMPLE-060: Missing output dir returns False."""
        monkeypatch.chdir(tmp_path)
        # No output/carnot_ising_v4_bd directory exists
        assert exp1048.check_dtbo_v4_exists() is False

    def test_returns_true_when_dtbo_present(self, tmp_path, monkeypatch):
        """SCENARIO-SAMPLE-060: Present DTBO file returns True."""
        dtbo_dir = tmp_path / "output" / "carnot_ising_v4_bd"
        dtbo_dir.mkdir(parents=True)
        (dtbo_dir / "carnot_ising_v4.dtbo").write_bytes(b"\x00" * 16)
        # Patch __file__ location so the module resolves relative to tmp_path
        monkeypatch.setattr(exp1048, "__file__", str(tmp_path / "scripts" / "fake.py"))
        assert exp1048.check_dtbo_v4_exists() is True


class TestCheckVivadoAvailable:
    """REQ-SAMPLE-036: Vivado availability detection via PATH."""

    def test_vivado_not_found(self, monkeypatch):
        """SCENARIO-SAMPLE-060: `which vivado` failure → returns False."""
        with mock.patch("subprocess.run") as m:
            m.return_value = mock.Mock(returncode=1)
            assert exp1048.check_vivado_available() is False

    def test_vivado_found(self):
        """SCENARIO-SAMPLE-060: `which vivado` success → returns True."""
        with mock.patch("subprocess.run") as m:
            m.return_value = mock.Mock(returncode=0)
            assert exp1048.check_vivado_available() is True


class TestSshReachable:
    """REQ-SAMPLE-036: SSH reachability probe."""

    def test_unreachable_on_timeout(self):
        """SCENARIO-SAMPLE-060: Subprocess exception → returns False."""
        with mock.patch.object(exp1048, "run_remote", side_effect=Exception("timeout")):
            assert exp1048.ssh_reachable("192.168.99.99") is False

    def test_reachable_when_alive_in_output(self):
        """SCENARIO-SAMPLE-060: 'alive' in output → returns True."""
        with mock.patch.object(exp1048, "run_remote", return_value=(0, "alive", "")):
            assert exp1048.ssh_reachable("192.168.99.99") is True

    def test_not_reachable_when_rc_nonzero(self):
        """SCENARIO-SAMPLE-060: Non-zero returncode → returns False."""
        with mock.patch.object(exp1048, "run_remote", return_value=(1, "", "")):
            assert exp1048.ssh_reachable("192.168.99.99") is False


class TestRunOnBoard:
    """REQ-SAMPLE-036: JSON extraction from board script output."""

    def test_parses_json_from_stdout(self):
        """SCENARIO-SAMPLE-060: Valid JSON on stdout is parsed correctly."""
        payload = {"energy_iqr": 5.0, "energy_distribution_nonuniform": True}
        with mock.patch.object(exp1048, "run_remote", return_value=(0, json.dumps(payload), "")):
            result = exp1048.run_on_board("host", "script")
        assert result["energy_iqr"] == 5.0
        assert result["energy_distribution_nonuniform"] is True

    def test_returns_error_on_no_output(self):
        """SCENARIO-SAMPLE-060: Empty stdout returns error dict."""
        with mock.patch.object(exp1048, "run_remote", return_value=(1, "", "crash")):
            result = exp1048.run_on_board("host", "script")
        assert "error" in result

    def test_extracts_json_from_last_line(self):
        """SCENARIO-SAMPLE-060: JSON on last line extracted even with debug lines above."""
        stdout = "debug info\nmore debug\n" + json.dumps({"ok": True})
        with mock.patch.object(exp1048, "run_remote", return_value=(0, stdout, "")):
            result = exp1048.run_on_board("host", "script")
        assert result.get("ok") is True


class TestUioDiagnosisIntegrity:
    """REQ-SAMPLE-036: Artifact fields match spec requirements."""

    def test_artifact_fields_present(self, tmp_path):
        """SCENARIO-SAMPLE-060: All required schema fields are present in artifact."""
        hw_data = {
            "uio_device": "/dev/uio4",
            "spin_count": 32,
            "n_samples": 100,
            "unique_spin_words": 90,
            "energy_mean": -0.29,
            "energy_std": 3.46,
            "energy_iqr": 5.0,
            "energy_min": -8.0,
            "energy_max": 8.0,
            "energy_distribution_nonuniform": True,
            "std_first_half": 3.94,
            "std_second_half": 2.87,
            "mpemba_speedup": 2.04,
            "energies_sample": [],
        }
        uio_map = "uio0: axi-pmon\nuio4: ising_sampler"

        with (
            mock.patch.object(exp1048, "ssh_reachable", return_value=True),
            mock.patch.object(exp1048, "check_dtbo_v4_exists", return_value=False),
            mock.patch.object(exp1048, "check_vivado_available", return_value=False),
            mock.patch.object(exp1048, "run_remote", return_value=(0, uio_map, "")),
            mock.patch.object(exp1048, "run_on_board", return_value=hw_data),
            mock.patch.object(exp1048, "_write_artifact") as write_mock,
        ):
            exp1048.main()

        assert write_mock.called
        artifact = write_mock.call_args[0][0]

        required_fields = [
            "ssh_reachable",
            "uio_uniform_constant_diagnosed",
            "dtbo_v4_exists",
            "vivado_available",
            "bitstream_rebuilt",
            "bitstream_loaded",
            "smoke_test_passed",
            "energy_distribution_nonuniform",
            "mpemba_convergence_speedup",
            "kanele_note",
            "kv260_guide_written",
            "honest_verdict",
        ]
        for field in required_fields:
            assert field in artifact, f"Missing required field: {field}"

        assert artifact["honest_verdict"] == "first_light_hardware_computing"
        assert artifact["smoke_test_passed"] is True
        assert artifact["uio_uniform_constant_diagnosed"] is True
