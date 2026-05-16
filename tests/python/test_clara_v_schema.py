import pytest
import numpy as np

from carnot.phase3.continuous_ebm import ContinuousEBM
from carnot.pipeline.clara_v_schema import EnergyVector, ContinuousLatentState

def test_energy_vector_total_energy():
    ev = EnergyVector(components=np.array([1.5, -0.5, 2.0]))
    assert ev.total_energy == 3.0

def test_continuous_latent_state_from_dimensions():
    dim = 5
    state = ContinuousLatentState.from_dimensions(dim)
    assert state.z.shape == (dim,)
    assert state.energy.components.shape == (dim,)
    assert np.all(state.z == 0.0)
    assert np.all(state.energy.components == 0.0)

def test_evaluate_ebm_energy():
    dim = 2
    coupling = np.array([[2.0, 0.5], [0.5, 1.0]])
    bias = np.array([1.0, -1.0])
    ebm = ContinuousEBM(variables=dim, coupling=coupling, bias=bias)
    
    state = ContinuousLatentState.from_dimensions(dim)
    # With z = [0, 0], energy should be 0
    assert state.evaluate_ebm_energy(ebm) == 0.0
    
    state.z = np.array([1.0, 2.0])
    # E(x) = -0.5 * x^T * J * x - h^T * x
    # J = [[2, 0.5], [0.5, 1]]
    # x^T * J = [1*2 + 2*0.5, 1*0.5 + 2*1] = [3, 2.5]
    # x^T * J * x = 3*1 + 2.5*2 = 8
    # h^T * x = 1*1 + -1*2 = -1
    # E(x) = -0.5 * 8 - (-1) = -4 + 1 = -3.0
    assert state.evaluate_ebm_energy(ebm) == -3.0

def test_evaluate_ebm_energy_dimension_mismatch():
    dim = 2
    coupling = np.array([[2.0, 0.5], [0.5, 1.0]])
    bias = np.array([1.0, -1.0])
    ebm = ContinuousEBM(variables=dim, coupling=coupling, bias=bias)
    
    state = ContinuousLatentState.from_dimensions(dim + 1)
    with pytest.raises(ValueError):
        state.evaluate_ebm_energy(ebm)
