"""Tests for Experiment 288: KV260 FPGA overlay bring-up with 60s timeout.

Spec coverage: REQ-SAMPLE-009,
               SCENARIO-SAMPLE-018, SCENARIO-SAMPLE-019
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.experiment_288_kv260_bringup as exp288
from scripts.experiment_288_kv260_bringup import (
    BRINGUP_TIMEOUT_SECONDS,
    DEFAULT_BITFILE_ENV,
    EXPERIMENT_ID,
    OVERLAY_IP_NAME,
    build_blocked_artifact,
    build_problem,
    check_env_var,
    run_experiment,
    spins_to_pm1,
    validate_spin_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _software_overlay_factory(bitfile_path: Any) -> Any:
    """Return a SoftwareFPGAOverlay for all paths — used by tests that need a transport."""
    from carnot.samplers.fpga_ising import SoftwareFPGAOverlay

    return SoftwareFPGAOverlay()


def _none_factory(bitfile_path: Any) -> None:
    return None


# ---------------------------------------------------------------------------
# REQ-SAMPLE-009: Blocked artifact when env var is missing
# SCENARIO-SAMPLE-018
# ---------------------------------------------------------------------------


class TestBlockedArtifact:
    """SCENARIO-SAMPLE-018: Blocked immediately when CARNOT_KV260_BITFILE is unset."""

    def test_check_env_var_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCENARIO-SAMPLE-018: Missing env var returns None without side effects."""
        monkeypatch.delenv(DEFAULT_BITFILE_ENV, raising=False)
        assert check_env_var() is None

    def test_check_env_var_returns_path_when_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-018: Env var present returns the configured path."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv(DEFAULT_BITFILE_ENV, str(bitfile))
        assert check_env_var() == str(bitfile)

    def test_blocked_artifact_has_required_fields(self) -> None:
        """SCENARIO-SAMPLE-018: Blocked artifact must include execution_path, missing, next_step."""
        artifact = build_blocked_artifact(
            missing=DEFAULT_BITFILE_ENV,
            next_step="Set CARNOT_KV260_BITFILE to the KV260 bitfile path.",
        )
        assert artifact["execution_path"] == "blocked"
        assert artifact["missing"] == DEFAULT_BITFILE_ENV
        assert "next_step" in artifact
        assert len(artifact["next_step"]) > 0

    def test_blocked_artifact_omits_overlay_load_ms(self) -> None:
        """SCENARIO-SAMPLE-018: overlay_load_ms absent or null for blocked runs."""
        artifact = build_blocked_artifact(
            missing=DEFAULT_BITFILE_ENV,
            next_step="Set CARNOT_KV260_BITFILE to the KV260 bitfile path.",
        )
        assert artifact.get("overlay_load_ms") is None

    def test_run_experiment_blocked_when_no_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-018: run_experiment emits blocked with no PYNQ import attempted."""
        monkeypatch.delenv(DEFAULT_BITFILE_ENV, raising=False)
        output_path = tmp_path / "results" / "exp288.json"
        payload = run_experiment(
            output_path=output_path,
            bitfile_path=None,
            overlay_loader=_none_factory,
        )
        assert payload["execution_path"] == "blocked"
        assert payload.get("overlay_load_ms") is None

    def test_run_experiment_blocked_artifact_written_to_disk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-018: Artifact JSON is written even when blocked."""
        monkeypatch.delenv(DEFAULT_BITFILE_ENV, raising=False)
        output_path = tmp_path / "results" / "exp288.json"
        run_experiment(
            output_path=output_path,
            bitfile_path=None,
            overlay_loader=_none_factory,
            write_output=True,
        )
        assert output_path.exists()
        with output_path.open() as f:
            data = json.load(f)
        assert data["execution_path"] == "blocked"


# ---------------------------------------------------------------------------
# REQ-SAMPLE-009: Hardware labeling
# ---------------------------------------------------------------------------


class TestHardwareLabeling:
    """REQ-SAMPLE-009: execution_path field must be 'hardware' only when PYNQ MMIO is real."""

    def test_software_model_not_labeled_hardware(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-009: SoftwareFPGAOverlay transport must yield software_model label."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv(DEFAULT_BITFILE_ENV, str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp288.json",
            bitfile_path=str(bitfile),
            overlay_loader=_software_overlay_factory,
        )
        assert payload["execution_path"] == "software_model"
        assert payload["execution_path"] != "hardware"

    def test_hardware_label_when_non_software_transport(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-009: Non-SoftwareFPGAOverlay transport is labeled hardware."""
        from carnot.samplers.fpga_ising import AXILiteRegisterMap, SoftwareFPGAOverlay

        # Use a thin proxy object that is NOT a SoftwareFPGAOverlay instance
        real = SoftwareFPGAOverlay()
        regmap = AXILiteRegisterMap()

        class HardwareProxy:
            def write(self, offset: int, value: int) -> None:
                real.write(offset, value)

            def read(self, offset: int) -> int:
                return real.read(offset)

        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv(DEFAULT_BITFILE_ENV, str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp288.json",
            bitfile_path=str(bitfile),
            overlay_loader=lambda _p: HardwareProxy(),
        )
        assert payload["execution_path"] == "hardware"


# ---------------------------------------------------------------------------
# REQ-SAMPLE-009: Register round-trip shape validation
# ---------------------------------------------------------------------------


class TestRegisterRoundTrip:
    """REQ-SAMPLE-009: Register write/read round-trip must preserve the compiled problem shape."""

    def test_register_roundtrip_shape_matches_problem(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-009: n_spins written to SPIN_COUNT register matches compiled problem."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv(DEFAULT_BITFILE_ENV, str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp288.json",
            bitfile_path=str(bitfile),
            overlay_loader=_software_overlay_factory,
        )
        assert payload["round_trip"] is not None
        rt = payload["round_trip"]
        # sample shape rows × cols must equal n_samples × n_spins
        biases, _ = build_problem()
        n_spins = int(biases.shape[0])
        assert rt["sample_shape"][1] == n_spins

    def test_register_roundtrip_records_overlay_load_ms(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-009: overlay_load_ms present and non-negative when load succeeds."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv(DEFAULT_BITFILE_ENV, str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp288.json",
            bitfile_path=str(bitfile),
            overlay_loader=_software_overlay_factory,
        )
        assert payload["overlay_load_ms"] is not None
        assert float(payload["overlay_load_ms"]) >= 0.0

    def test_register_roundtrip_records_register_roundtrip_us(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-009: register_roundtrip_us present when bring-up completes."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv(DEFAULT_BITFILE_ENV, str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp288.json",
            bitfile_path=str(bitfile),
            overlay_loader=_software_overlay_factory,
        )
        assert payload["round_trip"] is not None
        assert float(payload["round_trip"]["register_roundtrip_us"]) >= 0.0


# ---------------------------------------------------------------------------
# REQ-SAMPLE-009 / SCENARIO-SAMPLE-019: Spin state validity (+1 / −1)
# ---------------------------------------------------------------------------


class TestSpinStateValidity:
    """SCENARIO-SAMPLE-019: All spin values must be exactly +1 or −1 after conversion."""

    def test_spins_to_pm1_maps_true_to_plus_one(self) -> None:
        """SCENARIO-SAMPLE-019: True booleans map to +1."""
        arr = np.array([True, True, False], dtype=bool)
        result = spins_to_pm1(arr)
        np.testing.assert_array_equal(result, np.array([1, 1, -1], dtype=np.int8))

    def test_spins_to_pm1_maps_false_to_minus_one(self) -> None:
        """SCENARIO-SAMPLE-019: False booleans map to -1."""
        arr = np.array([False, False, True], dtype=bool)
        result = spins_to_pm1(arr)
        np.testing.assert_array_equal(result, np.array([-1, -1, 1], dtype=np.int8))

    def test_validate_spin_state_accepts_pm1_only(self) -> None:
        """SCENARIO-SAMPLE-019: Validation passes when all values are ±1."""
        spins = np.array([1, -1, 1, 1, -1], dtype=np.int8)
        assert validate_spin_state(spins) is True

    def test_validate_spin_state_rejects_zero(self) -> None:
        """SCENARIO-SAMPLE-019: Any value outside {+1,-1} invalidates the spin state."""
        spins = np.array([1, 0, -1], dtype=np.int8)
        assert validate_spin_state(spins) is False

    def test_artifact_records_spin_state_valid_true_after_roundtrip(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-019: spin_state_valid is True in artifact after bring-up completes."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv(DEFAULT_BITFILE_ENV, str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp288.json",
            bitfile_path=str(bitfile),
            overlay_loader=_software_overlay_factory,
        )
        assert payload.get("spin_state_valid") is True


# ---------------------------------------------------------------------------
# REQ-SAMPLE-009: 60s timeout constant and timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """REQ-SAMPLE-009: The 60s hard timeout must be enforced."""

    def test_bringup_timeout_constant_is_60(self) -> None:
        """REQ-SAMPLE-009: BRINGUP_TIMEOUT_SECONDS must equal 60."""
        assert BRINGUP_TIMEOUT_SECONDS == 60.0

    def test_timeout_emits_blocked_artifact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-009: A transport that never asserts DONE within timeout → blocked."""
        from carnot.samplers.fpga_ising import AXILiteRegisterMap

        class StallTransport:
            def write(self, offset: int, value: int) -> None:
                pass

            def read(self, offset: int) -> int:
                return 0  # never sets STATUS_DONE

        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv(DEFAULT_BITFILE_ENV, str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp288.json",
            bitfile_path=str(bitfile),
            overlay_loader=lambda _p: StallTransport(),
            timeout_seconds=0.01,  # tiny timeout for test speed
        )
        assert payload["execution_path"] == "blocked"
        assert payload.get("spin_state_valid") is not True

    def test_blocked_artifact_from_overlay_load_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-009: Exception during overlay load produces a blocked artifact."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv(DEFAULT_BITFILE_ENV, str(bitfile))

        def _raise(_path: Any) -> None:
            raise RuntimeError("simulated pynq load failure")

        payload = run_experiment(
            output_path=tmp_path / "exp288.json",
            bitfile_path=str(bitfile),
            overlay_loader=_raise,
        )
        assert payload["execution_path"] == "blocked"


# ---------------------------------------------------------------------------
# REQ-SAMPLE-009: Artifact schema completeness
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    """REQ-SAMPLE-009: Artifact has required top-level fields."""

    def test_artifact_has_required_fields_when_blocked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-009: Blocked artifacts have experiment, execution_path, missing, next_step."""
        monkeypatch.delenv(DEFAULT_BITFILE_ENV, raising=False)
        payload = run_experiment(
            output_path=tmp_path / "exp288.json",
            bitfile_path=None,
            overlay_loader=_none_factory,
        )
        for field in ("experiment", "execution_path", "missing", "next_step"):
            assert field in payload, f"Missing field: {field}"
        assert payload["experiment"] == EXPERIMENT_ID

    def test_artifact_has_required_fields_when_complete(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-009: Complete artifacts include round_trip, overlay_load_ms, spin_state_valid."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv(DEFAULT_BITFILE_ENV, str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp288.json",
            bitfile_path=str(bitfile),
            overlay_loader=_software_overlay_factory,
        )
        for field in ("experiment", "execution_path", "overlay_load_ms", "round_trip", "spin_state_valid"):
            assert field in payload, f"Missing field: {field}"
        assert payload["experiment"] == EXPERIMENT_ID
        assert payload["execution_path"] in {"hardware", "software_model", "blocked"}
