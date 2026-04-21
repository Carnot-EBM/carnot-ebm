"""Unit tests for PlattScaledJEPA (carnot.models.jepa_platt).

Tests trace to REQ-VERIFY-151.

Coverage target: 100% of python/carnot/models/jepa_platt.py.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

from carnot.models.eorm import CoTEnergyInput
from carnot.models.jepa_platt import PlattScaledJEPA


def _make_stub_eorm(raw_energy: float = 2.0) -> MagicMock:
    """Return an EORMModel stub that always returns ``raw_energy`` from .energy()."""
    stub = MagicMock()
    stub.energy.return_value = raw_energy
    return stub


class TestPlattScaledJEPAInit(unittest.TestCase):
    """REQ-VERIFY-151: PlattScaledJEPA construction."""

    def test_stores_temperature(self):
        """Temperature is stored after construction."""
        model = PlattScaledJEPA(_make_stub_eorm(), temperature=1.5)
        self.assertAlmostEqual(model.temperature, 1.5)

    def test_zero_temperature_raises(self):
        """Temperature <= 0 must raise ValueError (avoids division-by-zero / sign-flip)."""
        with self.assertRaises(ValueError):
            PlattScaledJEPA(_make_stub_eorm(), temperature=0.0)

    def test_negative_temperature_raises(self):
        """Negative temperature must raise ValueError."""
        with self.assertRaises(ValueError):
            PlattScaledJEPA(_make_stub_eorm(), temperature=-1.0)

    def test_env_override_applied(self):
        """JEPA_TIER2_PLATT_TEMPERATURE env var overrides the constructor temperature."""
        with unittest.mock.patch.dict(os.environ, {"JEPA_TIER2_PLATT_TEMPERATURE": "2.5"}):
            model = PlattScaledJEPA(_make_stub_eorm(), temperature=1.0)
        self.assertAlmostEqual(model.temperature, 2.5)

    def test_env_override_invalid_float_raises(self):
        """Non-numeric env var value must raise ValueError."""
        with unittest.mock.patch.dict(
            os.environ, {"JEPA_TIER2_PLATT_TEMPERATURE": "not_a_number"}
        ):
            with self.assertRaises(ValueError):
                PlattScaledJEPA(_make_stub_eorm(), temperature=1.0)

    def test_env_override_zero_raises(self):
        """Env var set to '0' must raise ValueError."""
        with unittest.mock.patch.dict(os.environ, {"JEPA_TIER2_PLATT_TEMPERATURE": "0.0"}):
            with self.assertRaises(ValueError):
                PlattScaledJEPA(_make_stub_eorm(), temperature=1.0)


class TestPlattScaledJEPAEnergy(unittest.TestCase):
    """REQ-VERIFY-151-1: PlattScaledJEPA.energy() divides raw energy by temperature."""

    def _cot(self, q: str = "q", r: str = "r") -> CoTEnergyInput:
        return CoTEnergyInput(question_text=q, response_text=r)

    def test_energy_divides_by_temperature(self):
        """energy() returns raw_energy / temperature."""
        stub = _make_stub_eorm(raw_energy=3.0)
        model = PlattScaledJEPA(stub, temperature=1.5)
        result = model.energy(self._cot())
        self.assertAlmostEqual(result, 2.0)  # 3.0 / 1.5

    def test_energy_passes_cot_input_to_eorm(self):
        """The CoTEnergyInput is forwarded to the underlying EORM model."""
        stub = _make_stub_eorm(raw_energy=1.0)
        model = PlattScaledJEPA(stub, temperature=1.0)
        cot = self._cot(q="What is 2+2?", r="It is 4.")
        model.energy(cot)
        stub.energy.assert_called_once_with(cot)

    def test_energy_temperature_one_returns_raw(self):
        """With temperature=1.0, output equals the raw EORM energy."""
        raw = 4.75
        stub = _make_stub_eorm(raw_energy=raw)
        model = PlattScaledJEPA(stub, temperature=1.0)
        self.assertAlmostEqual(model.energy(self._cot()), raw)

    def test_energy_high_temperature_reduces_score(self):
        """T > 1 compresses the energy scale (over-confident model pulled toward 0)."""
        stub = _make_stub_eorm(raw_energy=10.0)
        model = PlattScaledJEPA(stub, temperature=5.0)
        self.assertAlmostEqual(model.energy(self._cot()), 2.0)

    def test_energy_small_temperature_amplifies_score(self):
        """T < 1 expands the energy scale (under-confident model pushed outward)."""
        stub = _make_stub_eorm(raw_energy=1.0)
        model = PlattScaledJEPA(stub, temperature=0.5)
        self.assertAlmostEqual(model.energy(self._cot()), 2.0)

    def test_energy_returns_float(self):
        """energy() always returns a Python float, not a JAX array or numpy scalar."""
        stub = _make_stub_eorm(raw_energy=1.0)
        model = PlattScaledJEPA(stub, temperature=1.0)
        result = model.energy(self._cot())
        self.assertIsInstance(result, float)


class TestPlattScaledJEPAExport(unittest.TestCase):
    """REQ-VERIFY-151: PlattScaledJEPA must be importable from carnot.models."""

    def test_importable_from_carnot_models(self):
        """carnot.models.__all__ exports PlattScaledJEPA."""
        import carnot.models as m

        self.assertIn("PlattScaledJEPA", m.__all__)
        self.assertIs(m.PlattScaledJEPA, PlattScaledJEPA)


if __name__ == "__main__":
    unittest.main()
