"""Tests for Experiment 1068 — KV260 Ising sampler smoke test v9.

REQ-HW-040: KV260 Ising sampler exposes AXI-Lite control and SPIN_OUT
            registers and produces non-uniform sample distributions.
SCENARIO-HW-040: 100-sample smoke test on the deployed bitstream
            yields >1 unique value and bounded round-trip latency.

Tests cover host-side logic only — no board access required:
- derive_artifact: every honest_verdict branch (passed, smoke fail,
  ssh_unreachable, ssh_reachable_no_probe, error path).
- derive_artifact: artifact schema completeness (all required fields
  named in the experiment spec are present).
- write_guide_v4: file is written and contains the runtime values.
- check_ssh_reachable: subprocess timeout / nonzero return paths.
- run_on_board_program: handles ssh failure, malformed JSON, and
  embedded-error JSON correctly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import experiment_1068_kv260_smoke_test_v9 as exp  # noqa: E402


# Required fields named in the experiment spec — used by the conductor
# reconciler. Missing any of these would silently downgrade the
# experiment to "schema_invalid".
REQUIRED_FIELDS = {
    "experiment",
    "title",
    "run_date",
    "schema",
    "duration_s",
    "board_ip",
    "ssh_reachable",
    "control_register_found",
    "reset_method_tried",
    "reset_deasserted",
    "smoke_test_passed",
    "unique_values",
    "hardware_latency_us",
    "guide_v4_written",
    "honest_verdict",
}


# ---------------------------------------------------------------------------
# derive_artifact — verdict branches
# ---------------------------------------------------------------------------


class TestDeriveArtifactVerdicts:
    def test_ssh_unreachable_verdict(self) -> None:
        a = exp.derive_artifact(
            host="1.2.3.4", ssh_reachable=False, smoke_result=None, duration_s=2.0
        )
        assert a["honest_verdict"] == "ssh_unreachable"
        assert a["ssh_reachable"] is False
        assert a["smoke_test_passed"] is False
        assert a["unique_values"] is None
        assert a["hardware_latency_us"] is None

    def test_ssh_reachable_no_probe_verdict(self) -> None:
        # Defensive branch: should not normally trigger because main()
        # always passes a probe result when SSH is up, but the function
        # must still emit a coherent artifact in this case.
        a = exp.derive_artifact(
            host="1.2.3.4", ssh_reachable=True, smoke_result=None, duration_s=1.0
        )
        assert a["honest_verdict"] == "ssh_reachable_no_probe"
        assert a["ssh_reachable"] is True

    def test_smoke_passed_with_latency(self) -> None:
        smoke = {
            "stats": {
                "samples": 100,
                "unique_values": 77,
                "min_value": 0,
                "max_value": 0xFFFFFFFF,
                "min_popcount": 15,
                "max_popcount": 31,
                "mean_popcount": 16.27,
                "energy_range": 16,
                "first_5_hex": ["0x1", "0x2", "0x3", "0x4", "0x5"],
                "latency_us_min": 23.9,
                "latency_us_max": 39.3,
                "latency_us_mean": 26.6,
                "pre_status": 1,
                "pre_control": 0,
                "failed": 0,
            }
        }
        a = exp.derive_artifact(
            host="1.2.3.4", ssh_reachable=True, smoke_result=smoke, duration_s=3.0
        )
        assert a["honest_verdict"] == "smoke_test_passed_latency_measured"
        assert a["smoke_test_passed"] is True
        assert a["energy_distribution_nonuniform"] is True
        assert a["unique_values"] == 77
        assert a["hardware_latency_us"] == pytest.approx(26.6)
        assert a["reset_deasserted"] is True

    def test_smoke_passed_latency_pending(self) -> None:
        # Distribution non-uniform but latency missing — should not
        # happen in practice but the verdict table covers it.
        smoke = {
            "stats": {
                "samples": 100,
                "unique_values": 50,
                "min_value": 0,
                "max_value": 99,
                "min_popcount": 0,
                "max_popcount": 32,
                "mean_popcount": 16.0,
                "energy_range": 32,
                "first_5_hex": ["0x1", "0x2", "0x3", "0x4", "0x5"],
                "latency_us_mean": None,
                "failed": 0,
            }
        }
        a = exp.derive_artifact(
            host="1.2.3.4", ssh_reachable=True, smoke_result=smoke, duration_s=3.0
        )
        assert a["honest_verdict"] == "smoke_test_passed_latency_pending"
        assert a["smoke_test_passed"] is True
        assert a["hardware_latency_us"] is None

    def test_smoke_failed_constant_distribution(self) -> None:
        smoke = {
            "stats": {
                "samples": 100,
                "unique_values": 1,
                "min_value": 0,
                "max_value": 0,
                "min_popcount": 0,
                "max_popcount": 0,
                "mean_popcount": 0.0,
                "energy_range": 0,
                "first_5_hex": ["0x0", "0x0", "0x0", "0x0", "0x0"],
                "latency_us_mean": 25.0,
                "failed": 0,
            }
        }
        a = exp.derive_artifact(
            host="1.2.3.4", ssh_reachable=True, smoke_result=smoke, duration_s=3.0
        )
        assert a["honest_verdict"] == "reset_deasserted_smoke_fail"
        assert a["smoke_test_passed"] is False
        assert a["energy_distribution_nonuniform"] is False
        # reset_deasserted is True because the probe ran without error
        # — the test failure is about distribution, not register access.
        assert a["reset_deasserted"] is True

    def test_smoke_probe_returned_error(self) -> None:
        smoke = {"error": "uio_device_missing", "uio": "/dev/uio4"}
        a = exp.derive_artifact(
            host="1.2.3.4", ssh_reachable=True, smoke_result=smoke, duration_s=1.0
        )
        assert a["honest_verdict"] == "reset_deasserted_smoke_fail"
        assert a["smoke_error"] == "uio_device_missing"
        assert a["smoke_test_passed"] is False


# ---------------------------------------------------------------------------
# derive_artifact — schema completeness
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    def test_all_required_fields_present_when_unreachable(self) -> None:
        a = exp.derive_artifact(host="x", ssh_reachable=False, smoke_result=None, duration_s=1.0)
        missing = REQUIRED_FIELDS - set(a.keys())
        assert not missing, f"missing fields: {missing}"

    def test_all_required_fields_present_when_passed(self) -> None:
        smoke = {
            "stats": {
                "samples": 100,
                "unique_values": 70,
                "min_value": 0,
                "max_value": 0xFFFFFFFF,
                "min_popcount": 15,
                "max_popcount": 31,
                "mean_popcount": 16.0,
                "energy_range": 16,
                "first_5_hex": ["0x0"] * 5,
                "latency_us_mean": 25.0,
                "failed": 0,
            }
        }
        a = exp.derive_artifact(host="x", ssh_reachable=True, smoke_result=smoke, duration_s=2.0)
        missing = REQUIRED_FIELDS - set(a.keys())
        assert not missing, f"missing fields: {missing}"

    def test_experiment_id_and_schema_constants(self) -> None:
        a = exp.derive_artifact(host="x", ssh_reachable=False, smoke_result=None, duration_s=1.0)
        assert a["experiment"] == 1068
        assert a["schema"] == "kv260_smoke_test_v9"


# ---------------------------------------------------------------------------
# write_guide_v4
# ---------------------------------------------------------------------------


class TestWriteGuideV4:
    def test_writes_file_with_runtime_values(self, tmp_path: Path) -> None:
        a = exp.derive_artifact(host="x", ssh_reachable=False, smoke_result=None, duration_s=1.0)
        out = tmp_path / "guide.md"
        exp.write_guide_v4(out, a)
        body = out.read_text()
        assert "KV260 Ising Sampler Bring-Up Guide" in body
        assert a["honest_verdict"] in body
        # Register-map facts must appear so a future operator can
        # bring up the board without re-reading the RTL.
        assert "0x0000" in body and "0xA010" in body
        assert "/dev/uio4" in body


# ---------------------------------------------------------------------------
# check_ssh_reachable
# ---------------------------------------------------------------------------


class TestCheckSshReachable:
    def test_returns_true_on_zero_rc_with_ok_marker(self) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="SSH_OK\n", stderr="")
        with patch.object(subprocess, "run", return_value=fake):
            assert exp.check_ssh_reachable("h") is True

    def test_returns_false_on_nonzero_rc(self) -> None:
        fake = subprocess.CompletedProcess(
            args=[], returncode=255, stdout="", stderr="conn refused"
        )
        with patch.object(subprocess, "run", return_value=fake):
            assert exp.check_ssh_reachable("h") is False

    def test_returns_false_on_timeout(self) -> None:
        with patch.object(
            subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=10)
        ):
            assert exp.check_ssh_reachable("h") is False

    def test_returns_false_when_ssh_missing(self) -> None:
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            assert exp.check_ssh_reachable("h") is False


# ---------------------------------------------------------------------------
# run_on_board_program
# ---------------------------------------------------------------------------


class TestRunOnBoardProgram:
    def test_parses_well_formed_json(self) -> None:
        payload = json.dumps(
            {
                "samples": 100,
                "unique_values": 77,
                "energy_range": 16,
                "latency_us_mean": 25.0,
                "min_value": 0,
                "max_value": 0xFFFFFFFF,
                "min_popcount": 15,
                "max_popcount": 31,
                "mean_popcount": 16.0,
                "first_5_hex": ["0x1"] * 5,
                "failed": 0,
            }
        )
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=payload, stderr="")
        with patch.object(subprocess, "run", return_value=fake):
            r = exp.run_on_board_program("h")
        assert "stats" in r and r["stats"]["unique_values"] == 77

    def test_handles_ssh_nonzero_rc(self) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=255, stdout="", stderr="oops")
        with patch.object(subprocess, "run", return_value=fake):
            r = exp.run_on_board_program("h")
        assert "error" in r and r["error"].startswith("ssh_rc=")

    def test_handles_no_json_in_output(self) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="hello world", stderr="")
        with patch.object(subprocess, "run", return_value=fake):
            r = exp.run_on_board_program("h")
        assert r["error"] == "no_json_in_output"

    def test_handles_malformed_json(self) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="{not: json}", stderr="")
        with patch.object(subprocess, "run", return_value=fake):
            r = exp.run_on_board_program("h")
        assert r["error"].startswith("bad_json")

    def test_propagates_embedded_error(self) -> None:
        payload = json.dumps({"error": "uio_device_missing"})
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=payload, stderr="")
        with patch.object(subprocess, "run", return_value=fake):
            r = exp.run_on_board_program("h")
        assert r["error"] == "uio_device_missing"

    def test_handles_ssh_timeout(self) -> None:
        with patch.object(
            subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=60)
        ):
            r = exp.run_on_board_program("h")
        assert r["error"] == "ssh_timeout"

    def test_handles_ssh_missing(self) -> None:
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            r = exp.run_on_board_program("h")
        assert r["error"] == "ssh_not_installed"
