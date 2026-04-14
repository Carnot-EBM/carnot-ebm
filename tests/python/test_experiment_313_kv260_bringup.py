"""Tests for Experiment 313: KV260 FPGA hardware bring-up with honest_verdict.

Spec coverage: REQ-SAMPLE-012,
               SCENARIO-SAMPLE-025, SCENARIO-SAMPLE-026

Design philosophy:
    This test suite follows the honest_verdict pattern from Exp 303:
    - All tests that require physical KV260 hardware auto-skip when the
      hardware is unavailable (CARNOT_KV260_BITFILE not set or pynq absent).
    - Tests that cover the blocked / CPU-fallback path always run.
    - We never fabricate latency numbers: if the hardware path is not
      exercised, hardware_latency_us must be None in the artifact.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import scripts.experiment_313_kv260_bringup as exp313
from scripts.experiment_313_kv260_bringup import (
    EXPERIMENT_ID,
    LATENCY_TRIALS,
    N_SPINS,
    detect_kv260_hardware,
    run_experiment,
    spin_validity_check,
)


# ---------------------------------------------------------------------------
# Helper factories used by multiple test classes
# ---------------------------------------------------------------------------


def _software_overlay_factory(bitfile_path: Any) -> Any:
    """Return a SoftwareFPGAOverlay — simulates KV260 without real hardware."""
    from carnot.samplers.fpga_ising import SoftwareFPGAOverlay

    return SoftwareFPGAOverlay()


def _raising_factory(bitfile_path: Any) -> None:
    raise RuntimeError("simulated pynq load failure")


def _none_factory(bitfile_path: Any) -> None:
    return None


# ---------------------------------------------------------------------------
# REQ-SAMPLE-012 / SCENARIO-SAMPLE-026: detect_kv260_hardware
# ---------------------------------------------------------------------------


class TestDetectKv260Hardware:
    """REQ-SAMPLE-012: Hardware detection records each check independently."""

    def test_no_bitfile_env_var_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCENARIO-SAMPLE-026: honest_verdict=blocked_no_bitfile when env var unset."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        result = detect_kv260_hardware(overlay_factory=_none_factory)
        assert result["honest_verdict"] == "blocked_no_bitfile"
        assert result["kv260_detected"] is False

    def test_bitfile_set_pynq_absent_blocked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-026: honest_verdict=blocked_pynq when pynq not importable."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        # Simulate pynq not installed: overlay_factory raises ImportError
        def _import_error_factory(path: Any) -> None:
            raise ImportError("No module named 'pynq'")

        result = detect_kv260_hardware(overlay_factory=_import_error_factory)
        assert result["honest_verdict"] == "blocked_pynq"
        assert result["kv260_detected"] is False

    def test_overlay_load_fails_blocked_overlay(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-026: honest_verdict=blocked_overlay on RuntimeError during load."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        result = detect_kv260_hardware(overlay_factory=_raising_factory)
        assert result["honest_verdict"] == "blocked_overlay"
        assert result["kv260_detected"] is False

    def test_overlay_returns_none_blocked_overlay(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-026: None transport → blocked_overlay."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        result = detect_kv260_hardware(overlay_factory=_none_factory)
        assert result["honest_verdict"] == "blocked_overlay"
        assert result["kv260_detected"] is False

    def test_software_overlay_detected_as_hardware_candidate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: Software overlay transport reaches hardware-candidate state."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        result = detect_kv260_hardware(overlay_factory=_software_overlay_factory)
        # The software overlay is a valid transport — it passes detection.
        # honest_verdict here is determined by the full run_experiment flow,
        # but detect_kv260_hardware returns a transport and kv260_detected=True.
        assert result["transport"] is not None
        assert result["kv260_detected"] is True

    def test_detection_result_has_required_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-SAMPLE-012: detect_kv260_hardware always returns required keys."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        result = detect_kv260_hardware(overlay_factory=_none_factory)
        for key in ("honest_verdict", "kv260_detected", "transport", "bitfile_path"):
            assert key in result, f"Missing key: {key}"

    def test_detection_records_bitfile_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: bitfile_path recorded in detection result."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        result = detect_kv260_hardware(overlay_factory=_none_factory)
        assert result["bitfile_path"] == str(bitfile)

    def test_detection_bitfile_path_none_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-SAMPLE-012: bitfile_path is None when env var is unset."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        result = detect_kv260_hardware(overlay_factory=_none_factory)
        assert result["bitfile_path"] is None


# ---------------------------------------------------------------------------
# REQ-SAMPLE-012 / SCENARIO-SAMPLE-025: spin_validity_check
# ---------------------------------------------------------------------------


class TestSpinValidityCheck:
    """REQ-SAMPLE-012: All returned spins must be exactly +1 or -1."""

    def test_valid_pm1_array_passes(self) -> None:
        """SCENARIO-SAMPLE-025: Array of ±1 int8 values passes."""
        spins = np.array([1, -1, 1, 1, -1, 1], dtype=np.int8)
        valid, n_spins, shape_ok = spin_validity_check(spins, expected_n=6)
        assert valid is True
        assert n_spins == 6
        assert shape_ok is True

    def test_wrong_shape_fails(self) -> None:
        """REQ-SAMPLE-012: Shape mismatch is detected and recorded."""
        spins = np.array([1, -1, 1], dtype=np.int8)
        valid, n_spins, shape_ok = spin_validity_check(spins, expected_n=6)
        assert shape_ok is False

    def test_zero_value_fails_validity(self) -> None:
        """SCENARIO-SAMPLE-025: Any value outside {+1,-1} invalidates spins."""
        spins = np.array([1, 0, -1, 1], dtype=np.int8)
        valid, n_spins, shape_ok = spin_validity_check(spins, expected_n=4)
        assert valid is False

    def test_float_pm1_passes(self) -> None:
        """REQ-SAMPLE-012: Float ±1.0 values are accepted by the check."""
        spins = np.array([1.0, -1.0, 1.0], dtype=np.float32)
        valid, n_spins, shape_ok = spin_validity_check(spins, expected_n=3)
        assert valid is True

    def test_all_minus_one_passes(self) -> None:
        """REQ-SAMPLE-012: Uniform -1 spin state is valid."""
        spins = np.full(N_SPINS, -1, dtype=np.int8)
        valid, n_spins, shape_ok = spin_validity_check(spins, expected_n=N_SPINS)
        assert valid is True
        assert n_spins == N_SPINS

    def test_all_plus_one_passes(self) -> None:
        """REQ-SAMPLE-012: Uniform +1 spin state is valid."""
        spins = np.ones(N_SPINS, dtype=np.int8)
        valid, n_spins, shape_ok = spin_validity_check(spins, expected_n=N_SPINS)
        assert valid is True


# ---------------------------------------------------------------------------
# REQ-SAMPLE-012 / SCENARIO-SAMPLE-026: CPU fallback always measured
# ---------------------------------------------------------------------------


class TestCpuFallbackAlwaysMeasured:
    """SCENARIO-SAMPLE-026: cpu_fallback_latency_us present regardless of HW status."""

    def test_cpu_fallback_present_when_no_bitfile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-026: CPU fallback measured even when blocked_no_bitfile."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_none_factory,
            _cpu_trials=2,
        )
        assert "cpu_fallback_latency_us" in payload
        assert payload["cpu_fallback_latency_us"] is not None
        assert float(payload["cpu_fallback_latency_us"]) >= 0.0

    def test_cpu_fallback_present_when_overlay_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-026: CPU fallback measured even when overlay load fails."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_raising_factory,
            _cpu_trials=2,
        )
        assert payload["cpu_fallback_latency_us"] is not None
        assert float(payload["cpu_fallback_latency_us"]) >= 0.0

    def test_cpu_fallback_measured_with_software_overlay(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-026: CPU fallback measured alongside software overlay run."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_software_overlay_factory,
            _cpu_trials=2,
        )
        assert payload["cpu_fallback_latency_us"] is not None
        assert float(payload["cpu_fallback_latency_us"]) >= 0.0


# ---------------------------------------------------------------------------
# REQ-SAMPLE-012: Latency measurement
# ---------------------------------------------------------------------------


class TestLatencyMeasurement:
    """REQ-SAMPLE-012: Latency measured over LATENCY_TRIALS trials."""

    def test_latency_trials_constant_is_100(self) -> None:
        """REQ-SAMPLE-012: LATENCY_TRIALS must be 100."""
        assert LATENCY_TRIALS == 100

    def test_n_spins_constant_is_100(self) -> None:
        """REQ-SAMPLE-012: N_SPINS must be 100 for the hardware target."""
        assert N_SPINS == 100

    def test_cpu_fallback_reports_mean_and_p99(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: CPU fallback reports both mean_latency_us and p99_latency_us."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_none_factory,
            _cpu_trials=2,
        )
        assert "cpu_fallback_mean_latency_us" in payload
        assert "cpu_fallback_p99_latency_us" in payload
        assert payload["cpu_fallback_mean_latency_us"] >= 0.0
        assert payload["cpu_fallback_p99_latency_us"] >= payload["cpu_fallback_mean_latency_us"]

    def test_hardware_latency_null_when_blocked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: hardware_latency_us is None when hardware not exercised."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_none_factory,
            _cpu_trials=2,
        )
        assert payload.get("hardware_latency_us") is None

    def test_hardware_latency_not_fabricated_on_software_overlay(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: hardware_latency_us is None for software_model path (not real hardware)."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_software_overlay_factory,
            _cpu_trials=2,
        )
        # Software overlay is NOT real hardware — hardware_latency_us must be null.
        # honest_verdict must not claim hardware_working.
        assert payload["honest_verdict"] != "hardware_working"
        assert payload.get("hardware_latency_us") is None


# ---------------------------------------------------------------------------
# REQ-SAMPLE-012: honest_verdict values
# ---------------------------------------------------------------------------


class TestHonestVerdict:
    """REQ-SAMPLE-012: honest_verdict must come from the approved set."""

    APPROVED_VERDICTS = frozenset(
        [
            "hardware_working",
            "blocked_no_bitfile",
            "blocked_pynq",
            "blocked_overlay",
            "blocked_timeout",
        ]
    )

    def test_verdict_blocked_no_bitfile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: honest_verdict=blocked_no_bitfile when env var missing."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_none_factory,
            _cpu_trials=2,
        )
        assert payload["honest_verdict"] == "blocked_no_bitfile"

    def test_verdict_blocked_pynq_on_import_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: honest_verdict=blocked_pynq when pynq not available."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        def _import_error(path: Any) -> None:
            raise ImportError("No module named 'pynq'")

        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_import_error,
            _cpu_trials=2,
        )
        assert payload["honest_verdict"] == "blocked_pynq"

    def test_verdict_blocked_overlay_on_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: honest_verdict=blocked_overlay when overlay raises RuntimeError."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_raising_factory,
            _cpu_trials=2,
        )
        assert payload["honest_verdict"] == "blocked_overlay"

    def test_verdict_blocked_timeout_on_stall_transport(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: honest_verdict=blocked_timeout when STATUS_DONE never asserts."""
        from carnot.samplers.fpga_ising import AXILiteRegisterMap

        class StallTransport:
            def write(self, offset: int, value: int) -> None:
                pass

            def read(self, offset: int) -> int:
                return 0  # STATUS_DONE never set

        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=lambda _p: StallTransport(),
            roundtrip_timeout_seconds=0.01,
            _cpu_trials=2,
        )
        assert payload["honest_verdict"] == "blocked_timeout"

    def test_verdict_is_from_approved_set_blocked_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: honest_verdict is always from the approved vocabulary."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_none_factory,
            _cpu_trials=2,
        )
        assert payload["honest_verdict"] in self.APPROVED_VERDICTS

    def test_verdict_is_from_approved_set_software_overlay(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: honest_verdict from approved set even for software overlay."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_software_overlay_factory,
            _cpu_trials=2,
        )
        assert payload["honest_verdict"] in self.APPROVED_VERDICTS


# ---------------------------------------------------------------------------
# REQ-SAMPLE-012: Artifact schema completeness
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    """REQ-SAMPLE-012: Artifact has all required top-level fields."""

    REQUIRED_FIELDS = [
        "experiment",
        "schema",
        "run_date",
        "started_at",
        "finished_at",
        "honest_verdict",
        "kv260_detected",
        "bringup_steps_passed",
        "hardware_latency_us",
        "cpu_fallback_latency_us",
        "cpu_fallback_mean_latency_us",
        "cpu_fallback_p99_latency_us",
    ]

    def test_required_fields_present_blocked_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: All required fields present in blocked artifact."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_none_factory,
            _cpu_trials=2,
        )
        for field in self.REQUIRED_FIELDS:
            assert field in payload, f"Missing field: {field}"

    def test_required_fields_present_software_overlay(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: All required fields present when software overlay used."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_software_overlay_factory,
            _cpu_trials=2,
        )
        for field in self.REQUIRED_FIELDS:
            assert field in payload, f"Missing field: {field}"

    def test_experiment_id_correct(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: experiment field must be 313."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_none_factory,
            _cpu_trials=2,
        )
        assert payload["experiment"] == 313

    def test_kv260_detected_false_when_no_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-026: kv260_detected=False when env var unset."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_none_factory,
            _cpu_trials=2,
        )
        assert payload["kv260_detected"] is False

    def test_kv260_detected_false_when_pynq_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SCENARIO-SAMPLE-026: kv260_detected=False when pynq import fails."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        def _import_error(path: Any) -> None:
            raise ImportError("No module named 'pynq'")

        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_import_error,
            _cpu_trials=2,
        )
        assert payload["kv260_detected"] is False

    def test_bringup_steps_passed_is_zero_when_no_bitfile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: Zero steps passed when blocked at env-var check."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_none_factory,
            _cpu_trials=2,
        )
        assert payload["bringup_steps_passed"] == 0

    def test_bringup_steps_passed_greater_when_overlay_loads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: bringup_steps_passed > 0 when overlay loads successfully."""
        bitfile = tmp_path / "carnot_ising.bit"
        bitfile.write_bytes(b"bitstream")
        monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))

        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_software_overlay_factory,
            _cpu_trials=2,
        )
        assert payload["bringup_steps_passed"] > 0

    def test_artifact_written_to_disk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: Artifact JSON written to the specified path."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        out = tmp_path / "results" / "exp313.json"
        run_experiment(
            output_path=out,
            overlay_factory=_none_factory,
            write_output=True,
            _cpu_trials=2,
        )
        assert out.exists()
        with out.open() as f:
            data = json.load(f)
        assert data["experiment"] == 313

    def test_schema_field_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """REQ-SAMPLE-012: schema field identifies artifact version."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        payload = run_experiment(
            output_path=tmp_path / "exp313.json",
            overlay_factory=_none_factory,
            _cpu_trials=2,
        )
        assert isinstance(payload["schema"], dict)
        assert "artifact" in payload["schema"]


# ---------------------------------------------------------------------------
# Hardware path tests — auto-skip when KV260 not present
# ---------------------------------------------------------------------------


HW_REASON = "KV260 hardware not available: CARNOT_KV260_BITFILE not set or pynq absent"
_hw_available = (
    exp313._check_bitfile_env() is not None
    and exp313._try_import_pynq() is True
)


@pytest.mark.skipif(not _hw_available, reason=HW_REASON)
class TestHardwarePath:
    """SCENARIO-SAMPLE-025: Tests that require real KV260 hardware.

    These tests auto-skip when CARNOT_KV260_BITFILE is unset or pynq is
    not importable. They only run on a physical KV260 with the Carnot
    overlay loaded.
    """

    def test_hardware_working_verdict(self) -> None:
        """SCENARIO-SAMPLE-025: honest_verdict=hardware_working on real KV260."""
        import os
        from pathlib import Path

        payload = run_experiment(
            output_path=Path("/tmp/exp313_hw_test.json"),
        )
        assert payload["honest_verdict"] == "hardware_working"

    def test_hardware_latency_within_100us(self) -> None:
        """SCENARIO-SAMPLE-025: mean_latency_us ≤ 100 for 100-spin sampling."""
        import os
        from pathlib import Path

        payload = run_experiment(
            output_path=Path("/tmp/exp313_hw_latency.json"),
        )
        assert payload["honest_verdict"] == "hardware_working"
        assert payload["hardware_latency_us"] is not None
        hw = payload["hardware_latency_us"]
        assert hw["mean_latency_us"] <= 100.0, (
            f"mean_latency_us={hw['mean_latency_us']:.1f} exceeds 100μs target"
        )
        assert hw["p99_latency_us"] <= 200.0, (
            f"p99_latency_us={hw['p99_latency_us']:.1f} exceeds 200μs guard"
        )

    def test_hardware_spin_validity(self) -> None:
        """SCENARIO-SAMPLE-025: All returned spins are ±1 on real hardware."""
        from pathlib import Path

        payload = run_experiment(
            output_path=Path("/tmp/exp313_hw_spin.json"),
        )
        assert payload["honest_verdict"] == "hardware_working"
        assert payload["hardware_latency_us"]["spin_state_valid"] is True
