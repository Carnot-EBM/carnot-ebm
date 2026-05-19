import pytest
from carnot.verify.ising import IsingVerifier

def test_ising_verifier_energy():
    """Test IsingVerifier energy calculation."""
    v = IsingVerifier(n_spins=4)
    # The default J_ij = 1.0, h_i = 0.0.
    # state = [1, -1, 1, -1]
    # Coupling pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Products: (1*-1) + (1*1) + (1*-1) + (-1*1) + (-1*-1) + (1*-1)
    #         = -1 + 1 - 1 - 1 + 1 - 1 = -2
    # e = - (-2) = 2.0
    energy = v.energy([1, -1, 1, -1])
    assert energy == 2.0

def test_ising_verifier_invalid_state():
    """Test IsingVerifier raises error on invalid state length."""
    v = IsingVerifier(n_spins=4)
    with pytest.raises(ValueError):
        v.energy([1, -1])
