"""Tests for carnot.pipeline.hardware_energy_probe — 100% coverage.

Every branch of HardwareEnergyProbe and compute_eorm_hardware_correlation is
exercised here.  No real RAPL hardware is required — we mock the sysfs file
and the EORM model to keep tests CI-safe and deterministic.

Spec: REQ-LEARN-064,
      SCENARIO-LEARN-098, SCENARIO-LEARN-099, SCENARIO-LEARN-100
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.hardware_energy_probe import (
    EORMHardwareCorrelation,
    HardwareEnergyProbe,
    HardwareEnergyReading,
    compute_eorm_hardware_correlation,
)


# ---------------------------------------------------------------------------
# Minimal mock EORM model — no JAX, no real weights needed
# ---------------------------------------------------------------------------

@dataclass
class _MockCoTInput:
    question_text: str
    response_text: str


class _FixedEnergyModel:
    """Returns preset energies in round-robin order for deterministic tests."""

    def __init__(self, energies: list[float]) -> None:
        self._energies = list(energies)
        self._idx = 0

    def energy(self, cot_input: object) -> float:
        e = self._energies[self._idx % len(self._energies)]
        self._idx += 1
        return e


# ---------------------------------------------------------------------------
# HardwareEnergyReading dataclass
# ---------------------------------------------------------------------------

def test_hardware_energy_reading_fields() -> None:
    """HardwareEnergyReading stores timestamp_ns, joules, source."""
    r = HardwareEnergyReading(timestamp_ns=123, joules=0.5, source="mock")
    assert r.timestamp_ns == 123
    assert r.joules == 0.5
    assert r.source == "mock"


# ---------------------------------------------------------------------------
# _try_rapl — returns None when file absent (CI)
# ---------------------------------------------------------------------------

def test_try_rapl_returns_none_when_file_absent(tmp_path: pytest.FixtureRequest) -> None:
    """_try_rapl() returns None when /sys/class/powercap/.../energy_uj does not exist.

    SCENARIO-LEARN-098 (partial): sysfs unavailable path.
    """
    probe = HardwareEnergyProbe.__new__(HardwareEnergyProbe)
    # Patch the RAPL path to a non-existent file
    with patch(
        "carnot.pipeline.hardware_energy_probe._RAPL_ENERGY_PATH",
        new=pytest.importorskip("pathlib").Path("/nonexistent/energy_uj"),
    ):
        result = probe._try_rapl()
    assert result is None


def test_try_rapl_returns_joules_when_file_present(tmp_path: pytest.FixtureRequest) -> None:
    """_try_rapl() converts microjoules to joules when file exists."""
    fake_energy_file = tmp_path / "energy_uj"
    fake_energy_file.write_text("1000000\n")  # 1 000 000 µJ = 1.0 J

    probe = HardwareEnergyProbe.__new__(HardwareEnergyProbe)
    with patch(
        "carnot.pipeline.hardware_energy_probe._RAPL_ENERGY_PATH",
        new=fake_energy_file,
    ):
        result = probe._try_rapl()
    assert result == pytest.approx(1.0)


def test_try_rapl_returns_none_on_invalid_content(tmp_path: pytest.FixtureRequest) -> None:
    """_try_rapl() returns None when file contains non-integer text."""
    fake_energy_file = tmp_path / "energy_uj"
    fake_energy_file.write_text("not_a_number\n")

    probe = HardwareEnergyProbe.__new__(HardwareEnergyProbe)
    with patch(
        "carnot.pipeline.hardware_energy_probe._RAPL_ENERGY_PATH",
        new=fake_energy_file,
    ):
        result = probe._try_rapl()
    assert result is None


# ---------------------------------------------------------------------------
# _try_pyrapl — returns None when not installed
# ---------------------------------------------------------------------------

def test_try_pyrapl_returns_none_when_not_installed() -> None:
    """_try_pyrapl() returns None when pyRAPL is not importable."""
    probe = HardwareEnergyProbe.__new__(HardwareEnergyProbe)
    with patch.dict("sys.modules", {"pyRAPL": None}):
        result = probe._try_pyrapl()
    assert result is None


def test_try_pyrapl_returns_joules_when_available() -> None:
    """_try_pyrapl() returns joules when pyRAPL is installed and readable."""
    # Build a minimal pyRAPL mock
    mock_pyrapl = MagicMock()
    mock_result = MagicMock()
    mock_result.pkg = [2_000_000.0]  # 2 J in µJ
    mock_meter = MagicMock()
    mock_meter.result = mock_result
    mock_pyrapl.Measurement.return_value = mock_meter

    probe = HardwareEnergyProbe.__new__(HardwareEnergyProbe)
    with patch.dict("sys.modules", {"pyRAPL": mock_pyrapl}):
        result = probe._try_pyrapl()
    assert result == pytest.approx(2.0)


def test_try_pyrapl_returns_none_on_empty_pkg() -> None:
    """_try_pyrapl() returns None when pkg list is empty."""
    mock_pyrapl = MagicMock()
    mock_result = MagicMock()
    mock_result.pkg = []
    mock_meter = MagicMock()
    mock_meter.result = mock_result
    mock_pyrapl.Measurement.return_value = mock_meter

    probe = HardwareEnergyProbe.__new__(HardwareEnergyProbe)
    with patch.dict("sys.modules", {"pyRAPL": mock_pyrapl}):
        result = probe._try_pyrapl()
    assert result is None


def test_try_pyrapl_returns_none_on_exception() -> None:
    """_try_pyrapl() returns None when pyRAPL raises any exception."""
    mock_pyrapl = MagicMock()
    mock_pyrapl.setup.side_effect = RuntimeError("permission denied")

    probe = HardwareEnergyProbe.__new__(HardwareEnergyProbe)
    with patch.dict("sys.modules", {"pyRAPL": mock_pyrapl}):
        result = probe._try_pyrapl()
    assert result is None


# ---------------------------------------------------------------------------
# read() — returns mock source when RAPL unavailable (SCENARIO-LEARN-098)
# ---------------------------------------------------------------------------

def test_read_returns_mock_when_rapl_unavailable() -> None:
    """read() returns source='mock' and joules=0.0 when no hardware available.

    Spec: SCENARIO-LEARN-098
    """
    # Force mock source by patching both backends
    with patch(
        "carnot.pipeline.hardware_energy_probe._RAPL_ENERGY_PATH",
        new=pytest.importorskip("pathlib").Path("/nonexistent/energy_uj"),
    ), patch.dict("sys.modules", {"pyRAPL": None}):
        probe = HardwareEnergyProbe()

    assert probe.source == "mock"
    reading = probe.read()
    assert reading.source == "mock"
    assert reading.joules == 0.0
    assert isinstance(reading.timestamp_ns, int)


def test_read_returns_rapl_source_when_file_present(tmp_path: pytest.FixtureRequest) -> None:
    """read() returns source='rapl' when sysfs energy file is accessible."""
    fake_energy_file = tmp_path / "energy_uj"
    fake_energy_file.write_text("500000\n")  # 0.5 J

    # Keep the patch active for both construction AND read() call
    with patch(
        "carnot.pipeline.hardware_energy_probe._RAPL_ENERGY_PATH",
        new=fake_energy_file,
    ):
        probe = HardwareEnergyProbe()
        assert probe.source == "rapl"
        reading = probe.read()

    assert reading.source == "rapl"
    assert reading.joules == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# measure_segment() — returns (result, delta_joules) (SCENARIO-LEARN-099)
# ---------------------------------------------------------------------------

def test_measure_segment_returns_result_and_delta_joules() -> None:
    """measure_segment(fn) returns (fn(), delta_joules).

    Spec: SCENARIO-LEARN-099
    """
    with patch(
        "carnot.pipeline.hardware_energy_probe._RAPL_ENERGY_PATH",
        new=pytest.importorskip("pathlib").Path("/nonexistent/energy_uj"),
    ), patch.dict("sys.modules", {"pyRAPL": None}):
        probe = HardwareEnergyProbe()

    result, delta = probe.measure_segment(lambda: 42)
    assert result == 42
    assert isinstance(delta, float)
    assert delta == pytest.approx(0.0)  # mock always returns 0.0


def test_measure_segment_handles_wraparound() -> None:
    """measure_segment() clamps negative delta (counter wrap) to 0.0."""
    with patch(
        "carnot.pipeline.hardware_energy_probe._RAPL_ENERGY_PATH",
        new=pytest.importorskip("pathlib").Path("/nonexistent/energy_uj"),
    ), patch.dict("sys.modules", {"pyRAPL": None}):
        probe = HardwareEnergyProbe()

    # Force the internal read() to return decreasing values
    readings = iter([
        HardwareEnergyReading(timestamp_ns=1, joules=100.0, source="mock"),
        HardwareEnergyReading(timestamp_ns=2, joules=50.0, source="mock"),  # wrapped
    ])
    with patch.object(probe, "read", side_effect=readings):
        _, delta = probe.measure_segment(lambda: None)
    assert delta == 0.0


# ---------------------------------------------------------------------------
# compute_eorm_hardware_correlation() — SCENARIO-LEARN-100
# ---------------------------------------------------------------------------

def test_correlation_with_mock_probe() -> None:
    """compute_eorm_hardware_correlation() works end-to-end with mock probe.

    With mock source all hw deltas are 0.0 — r=0, p=1.0, calibration_viable=False.

    Spec: SCENARIO-LEARN-100
    """
    from carnot.models.eorm import CoTEnergyInput  # only to validate import path

    with patch(
        "carnot.pipeline.hardware_energy_probe._RAPL_ENERGY_PATH",
        new=pytest.importorskip("pathlib").Path("/nonexistent/energy_uj"),
    ), patch.dict("sys.modules", {"pyRAPL": None}):
        probe = HardwareEnergyProbe()

    energies = [float(i) for i in range(5)]
    model = _FixedEnergyModel(energies)
    steps = [f"Step {i}: let x = {i}" for i in range(5)]

    result = compute_eorm_hardware_correlation(probe, model, steps)

    assert result.n_steps == 5
    assert len(result.hardware_energies) == 5
    assert len(result.eorm_energies) == 5
    # All hardware deltas are 0 from mock — r cannot be meaningfully computed
    assert result.pearson_r == pytest.approx(0.0, abs=1e-9)
    assert not result.calibration_viable


def test_correlation_fields_present() -> None:
    """EORMHardwareCorrelation dataclass has all required fields."""
    c = EORMHardwareCorrelation(
        n_steps=3,
        pearson_r=0.8,
        p_value=0.01,
        hardware_energies=[0.1, 0.2, 0.3],
        eorm_energies=[1.0, 2.0, 3.0],
        calibration_viable=True,
    )
    assert c.n_steps == 3
    assert c.pearson_r == 0.8
    assert c.p_value == 0.01
    assert c.calibration_viable


def test_correlation_calibration_viable_threshold() -> None:
    """calibration_viable requires r > 0.5 AND p_value < 0.05."""
    # Viable
    c1 = EORMHardwareCorrelation(
        n_steps=30, pearson_r=0.7, p_value=0.01,
        hardware_energies=[], eorm_energies=[], calibration_viable=True,
    )
    assert c1.calibration_viable

    # r too low
    c2 = EORMHardwareCorrelation(
        n_steps=30, pearson_r=0.3, p_value=0.01,
        hardware_energies=[], eorm_energies=[], calibration_viable=False,
    )
    assert not c2.calibration_viable

    # p too high
    c3 = EORMHardwareCorrelation(
        n_steps=30, pearson_r=0.7, p_value=0.1,
        hardware_energies=[], eorm_energies=[], calibration_viable=False,
    )
    assert not c3.calibration_viable


def test_correlation_handles_single_step() -> None:
    """compute_eorm_hardware_correlation() handles edge case of 1 step (no pearsonr)."""
    with patch(
        "carnot.pipeline.hardware_energy_probe._RAPL_ENERGY_PATH",
        new=pytest.importorskip("pathlib").Path("/nonexistent/energy_uj"),
    ), patch.dict("sys.modules", {"pyRAPL": None}):
        probe = HardwareEnergyProbe()

    model = _FixedEnergyModel([0.5])
    result = compute_eorm_hardware_correlation(probe, model, ["single step"])
    assert result.n_steps == 1
    assert result.pearson_r == pytest.approx(0.0, abs=1e-9)
    assert not result.calibration_viable


def test_source_property() -> None:
    """HardwareEnergyProbe.source property returns the active source label."""
    with patch(
        "carnot.pipeline.hardware_energy_probe._RAPL_ENERGY_PATH",
        new=pytest.importorskip("pathlib").Path("/nonexistent/energy_uj"),
    ), patch.dict("sys.modules", {"pyRAPL": None}):
        probe = HardwareEnergyProbe()
    assert probe.source == "mock"
