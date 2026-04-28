"""Tests for the Phase 2 transpiler API surface (HardwareSpec / IsingSpec).

Spec: REQ-PHASE2-001 (HardwareSpec descriptor), REQ-PHASE2-002 (IsingSpec
output schema).
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.hardware.transpiler import HardwareSpec, IsingSpec


# REQ-PHASE2-001
def test_hardware_spec_sparse_defaults_locality_2() -> None:
    """A sparse HardwareSpec without explicit locality should default to
    2-local couplings — the conservative KV260/Z1/ECP5 native primitive.
    """
    spec = HardwareSpec(kind="sparse", max_spins=1024, beta_range=(0.1, 5.0))
    assert spec.locality == 2
    assert spec.vendor_target == "synthetic-cpu"


# REQ-PHASE2-001
def test_hardware_spec_rejects_zero_max_spins() -> None:
    """Construction must fail rather than silently produce a useless spec
    when the spin budget is zero or negative.
    """
    with pytest.raises(ValueError, match="max_spins"):
        HardwareSpec(kind="sparse", max_spins=0, beta_range=(0.1, 5.0))


# REQ-PHASE2-001
def test_hardware_spec_rejects_invalid_beta_range() -> None:
    with pytest.raises(ValueError, match="beta_range"):
        HardwareSpec(kind="sparse", max_spins=1024, beta_range=(1.0, 0.5))


# REQ-PHASE2-002
def test_ising_spec_validates_symmetric_zero_diag() -> None:
    """Standard Ising convention: J symmetric, diagonal zero. The spec
    must catch violations at construction; otherwise the downstream
    sampler can produce subtly wrong distributions.
    """
    n = 8
    rng = np.random.default_rng(0)
    J = rng.normal(size=(n, n))
    # Asymmetric J should be rejected
    with pytest.raises(ValueError, match="symmetric"):
        IsingSpec(
            J=J,
            h=np.zeros(n),
            phi=lambda z: np.zeros((1, n)),
            psi=lambda s: np.zeros((1, 2)),
        )

    # Symmetric but non-zero diagonal should be rejected
    Jsym = 0.5 * (J + J.T)
    np.fill_diagonal(Jsym, 1.0)
    with pytest.raises(ValueError, match="diagonal"):
        IsingSpec(
            J=Jsym,
            h=np.zeros(n),
            phi=lambda z: np.zeros((1, n)),
            psi=lambda s: np.zeros((1, 2)),
        )


# REQ-PHASE2-002
def test_ising_spec_energy_matches_quadratic_form() -> None:
    """For a known small J, h, the energy ``-s^T J s - h^T s`` should be
    exactly reproduced by IsingSpec.energy. Both single and batched.
    """
    J = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, -0.3], [0.5, -0.3, 0.0]])
    h = np.array([0.1, -0.2, 0.05])
    spec = IsingSpec(J=J, h=h, phi=lambda z: np.zeros((1, 3)), psi=lambda s: np.zeros((1, 2)))
    s = np.array([1.0, -1.0, 1.0])
    expected_single = -float(s @ J @ s) - float(h @ s)
    assert np.isclose(spec.energy(s), expected_single)

    batch = np.array([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]])
    expected_batch = -np.einsum("bi,ij,bj->b", batch, J, batch) - batch @ h
    np.testing.assert_allclose(spec.energy(batch), expected_batch)


# REQ-PHASE2-002
def test_ising_spec_rejects_h_shape_mismatch() -> None:
    """h must match J's spin count or downstream sampling silently
    indexes into out-of-bounds memory regions.
    """
    J = np.zeros((4, 4))
    with pytest.raises(ValueError, match="h shape"):
        IsingSpec(
            J=J,
            h=np.zeros(3),  # mismatched
            phi=lambda z: np.zeros((1, 4)),
            psi=lambda s: np.zeros((1, 2)),
        )
