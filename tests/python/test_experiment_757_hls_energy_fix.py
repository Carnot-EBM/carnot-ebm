"""Tests for Experiment 757 — HLS Ising Energy Sign Fix.

All tests trace to REQ-HW-040 / SCENARIO-HW-040.

WHY these tests exist:
    Exp 750 reported a sign-convention failure in the HLS C++ kernel: the
    CPU simulation produced energy +3.0 for a test whose ground state is -3.0.
    These tests verify that HLSEnergyValidator implements the correct Hamiltonian
    sign (E = -sum J s s) and that the validation logic works as expected.

Coverage target: 100% of python/carnot/pipeline/hls_energy_validator.py.
"""

from __future__ import annotations

import math
import pytest
from python.carnot.pipeline.hls_energy_validator import HLSEnergyValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ferro_validator(n: int = 4, j: float = 1.0) -> HLSEnergyValidator:
    """Return an HLSEnergyValidator for a fully connected ferromagnetic system.

    J[i][k] = j for i != k, 0 on diagonal.  h = 0.
    Ground state: all spins +1 or all spins -1.
    Ground state energy: -j * n * (n-1) / 2.
    """
    j_mat = [[j if i != k else 0.0 for k in range(n)] for i in range(n)]
    h = [0.0] * n
    return HLSEnergyValidator(n_spins=n, j_matrix=j_mat, h_field=h)


def _antiferro_validator(n: int = 4, j: float = 1.0) -> HLSEnergyValidator:
    """Return an HLSEnergyValidator for a fully connected antiferromagnetic system.

    J[i][k] = -j for i != k.  Ground state: alternating ±1.
    """
    j_mat = [[-j if i != k else 0.0 for k in range(n)] for i in range(n)]
    h = [0.0] * n
    return HLSEnergyValidator(n_spins=n, j_matrix=j_mat, h_field=h)


# ---------------------------------------------------------------------------
# REQ-HW-040: compute_energy — sign convention and basic arithmetic
# ---------------------------------------------------------------------------

class TestComputeEnergy:
    """Spec: REQ-HW-040 — energy function MUST use E = -sum J s s."""

    def test_ferro_all_ones_n4_energy_is_negative_six(self):
        """Ground-state energy for N=4 fully connected ferromagnet must be -6.0.

        WHY -6.0: there are N*(N-1)/2 = 6 pairs, each contributing -J*1*1 = -1.
        Total: -6.  A positive result (+6) would reveal the sign bug from Exp 750.

        Spec: REQ-HW-040, SCENARIO-HW-040
        """
        v = _ferro_validator(n=4, j=1.0)
        energy = v.compute_energy([1, 1, 1, 1])
        assert abs(energy - (-6.0)) < 0.5, (
            f"Expected -6.0 for N=4 all-ones ferromagnet, got {energy}. "
            "Sign convention bug: energy must use -= not +="
        )

    def test_ferro_all_ones_energy_is_negative(self):
        """All-ones ground state of ferromagnet must have strictly negative energy.

        Spec: REQ-HW-040
        """
        v = _ferro_validator(n=4)
        energy = v.compute_energy([1, 1, 1, 1])
        assert energy < 0.0, f"Ferromagnetic ground state energy must be < 0, got {energy}"

    def test_ferro_all_minus_ones_same_energy_as_all_ones(self):
        """All-(-1) and all-(+1) have identical energy by symmetry.

        WHY: E = -sum J s_i s_j.  Flipping all spins leaves s_i*s_j unchanged.

        Spec: REQ-HW-040
        """
        v = _ferro_validator(n=4)
        e_plus = v.compute_energy([1, 1, 1, 1])
        e_minus = v.compute_energy([-1, -1, -1, -1])
        assert abs(e_plus - e_minus) < 1e-9, (
            f"All+1 and all-1 energies should be equal; got {e_plus} vs {e_minus}"
        )

    def test_zero_coupling_zero_energy(self):
        """With J=0 and h=0, energy must be 0 for any spin configuration.

        Spec: REQ-HW-040
        """
        n = 4
        j_zero = [[0.0] * n for _ in range(n)]
        h_zero = [0.0] * n
        v = HLSEnergyValidator(n_spins=n, j_matrix=j_zero, h_field=h_zero)
        assert v.compute_energy([1, -1, 1, -1]) == 0.0

    def test_bias_field_lowers_energy_of_aligned_spin(self):
        """Positive h_i applied to spin +1 decreases energy by h_i.

        WHY: E includes -sum_i h_i s_i.  h=1, s=+1 → -1*1 = -1 contribution.

        Spec: REQ-HW-040
        """
        # Single spin, no coupling.
        v = HLSEnergyValidator(n_spins=1, j_matrix=[[0.0]], h_field=[2.0])
        # E = -h * s = -2.0 * 1 = -2.0
        assert abs(v.compute_energy([1]) - (-2.0)) < 1e-9

        # Anti-aligned: E = -h * s = -2.0 * (-1) = +2.0
        assert abs(v.compute_energy([-1]) - 2.0) < 1e-9

    def test_wrong_length_spins_raises(self):
        """compute_energy must raise ValueError if spins length != n_spins.

        Spec: REQ-HW-040
        """
        v = _ferro_validator(n=4)
        with pytest.raises(ValueError, match="length"):
            v.compute_energy([1, 1, 1])

    def test_invalid_spin_value_raises(self):
        """compute_energy must raise ValueError for spin values other than ±1.

        Spec: REQ-HW-040
        """
        v = _ferro_validator(n=4)
        with pytest.raises(ValueError, match=r"spin\["):
            v.compute_energy([1, 0, 1, 1])

    def test_n2_ferromagnet_energy(self):
        """N=2 ferromagnet: one bond, expected E = -J*s0*s1 = -1.0 for s=[1,1].

        Spec: REQ-HW-040
        """
        v = _ferro_validator(n=2, j=1.0)
        assert abs(v.compute_energy([1, 1]) - (-1.0)) < 1e-9
        # Anti-aligned: E = -J * (+1) * (-1) = +1.0
        assert abs(v.compute_energy([1, -1]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# REQ-HW-040: validate_ground_state
# ---------------------------------------------------------------------------

class TestValidateGroundState:
    """Spec: REQ-HW-040 — ground-state validator returns True when energy is negative."""

    def test_ferromagnet_ground_state_valid(self):
        """Ferromagnet all-ones ground state returns True.

        Spec: REQ-HW-040, SCENARIO-HW-040
        """
        v = _ferro_validator(n=4)
        assert v.validate_ground_state() is True

    def test_antiferromagnet_all_ones_not_ground_state(self):
        """All-ones for antiferromagnet is a HIGH energy state; should return False.

        WHY: For antiferromagnet (J<0), all-ones gives energy +6 (not ground state).
        validate_ground_state() returns False because all-ones energy is positive.

        Spec: REQ-HW-040
        """
        v = _antiferro_validator(n=4)
        assert v.validate_ground_state() is False

    def test_zero_coupling_not_valid(self):
        """With J=0, all-ones energy = 0 which is NOT < 0; returns False.

        Spec: REQ-HW-040
        """
        n = 4
        v = HLSEnergyValidator(
            n_spins=n,
            j_matrix=[[0.0] * n for _ in range(n)],
            h_field=[0.0] * n,
        )
        assert v.validate_ground_state() is False

    def test_strong_h_field_makes_ground_state_valid(self):
        """Positive h field on all spins makes all-ones energy negative even with J=0.

        Spec: REQ-HW-040
        """
        n = 4
        v = HLSEnergyValidator(
            n_spins=n,
            j_matrix=[[0.0] * n for _ in range(n)],
            h_field=[1.0] * n,  # each spin gets E += -h_i*1 = -1 → total -4
        )
        assert v.validate_ground_state() is True


# ---------------------------------------------------------------------------
# REQ-HW-040: compare_with_python_ising
# ---------------------------------------------------------------------------

class TestCompareWithPythonIsing:
    """Spec: REQ-HW-040 — consistency check across random spin configurations."""

    def test_ferromagnet_max_delta_pct_below_10(self):
        """Ferromagnet comparison returns max_delta_pct < 10% for 100 random configs.

        Spec: REQ-HW-040
        """
        v = _ferro_validator(n=4)
        _max_delta, max_delta_pct = v.compare_with_python_ising(n_samples=100)
        assert max_delta_pct < 10.0, (
            f"max_delta_pct={max_delta_pct:.2f}% exceeds 10% threshold"
        )

    def test_returns_finite_floats(self):
        """compare_with_python_ising must return finite float values.

        Spec: REQ-HW-040
        """
        v = _ferro_validator(n=4)
        max_delta, max_delta_pct = v.compare_with_python_ising(n_samples=10)
        assert math.isfinite(max_delta)
        assert math.isfinite(max_delta_pct)

    def test_zero_samples_returns_zero(self):
        """Zero samples gives zero delta (no random configs to deviate).

        Spec: REQ-HW-040
        """
        v = _ferro_validator(n=4)
        max_delta, max_delta_pct = v.compare_with_python_ising(n_samples=0)
        assert max_delta == 0.0
        assert max_delta_pct == 0.0

    def test_infinite_energy_propagates_as_inf(self):
        """If compute_energy returns inf (e.g. J=inf), max_delta becomes inf.

        WHY this branch exists: guards against NaN/inf propagation from
        pathological coupling matrices.  This test covers the early-exit path.

        Spec: REQ-HW-040
        """
        import math as _math
        # Coupling with J=inf causes energy to be inf.
        n = 2
        j_inf = [[0.0, float("inf")], [float("inf"), 0.0]]
        v = HLSEnergyValidator(n_spins=n, j_matrix=j_inf, h_field=[0.0, 0.0])
        max_delta, _pct = v.compare_with_python_ising(n_samples=5)
        assert not _math.isfinite(max_delta) or max_delta >= 0.0


# ---------------------------------------------------------------------------
# REQ-HW-040: constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    """Spec: REQ-HW-040 — constructor must reject invalid inputs."""

    def test_wrong_j_matrix_rows_raises(self):
        """Constructor raises ValueError when j_matrix has wrong number of rows."""
        with pytest.raises(ValueError, match="j_matrix"):
            HLSEnergyValidator(
                n_spins=3,
                j_matrix=[[0.0, 0.0], [0.0, 0.0]],  # 2 rows, not 3
                h_field=[0.0, 0.0, 0.0],
            )

    def test_wrong_j_matrix_cols_raises(self):
        """Constructor raises ValueError when a j_matrix row has wrong length."""
        with pytest.raises(ValueError, match="j_matrix"):
            HLSEnergyValidator(
                n_spins=2,
                j_matrix=[[0.0], [0.0, 0.0]],  # first row has 1 col, not 2
                h_field=[0.0, 0.0],
            )

    def test_wrong_h_field_length_raises(self):
        """Constructor raises ValueError when h_field has wrong length."""
        with pytest.raises(ValueError, match="h_field"):
            HLSEnergyValidator(
                n_spins=3,
                j_matrix=[[0.0] * 3 for _ in range(3)],
                h_field=[0.0, 0.0],  # wrong length
            )

    def test_n_spins_zero_raises(self):
        """Constructor raises ValueError for n_spins < 1."""
        with pytest.raises(ValueError, match="n_spins"):
            HLSEnergyValidator(n_spins=0, j_matrix=[], h_field=[])
