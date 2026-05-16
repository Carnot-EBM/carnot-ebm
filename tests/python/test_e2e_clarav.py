"""
E2E-008: CLaRa-V Continuous Latent Space Evaluation
Spec refs: REQ-KONA-040, SCENARIO-KONA-041.
"""

import numpy as np
from carnot.pipeline.clara_v_schema import ContinuousLatentState
from carnot.phase3.continuous_ebm import ContinuousEBM
from carnot.models.pinet_1995 import project_clara_state

def test_clarav_e2e_evaluation():
    """
    Test the full E2E path for CLaRa-V continuous latent variables mapping to EBM instances,
    evaluating energy, and projecting constraints.
    """
    dim = 4
    # 1. Instantiate a ContinuousLatentState
    state = ContinuousLatentState.from_dimensions(dim)
    state.z = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float64)
    
    # 2. Instantiate a ContinuousEBM
    coupling = np.eye(dim) * -0.1
    bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    ebm = ContinuousEBM(variables=dim, coupling=coupling, bias=bias)
    
    # 3. Evaluate the continuous latent state energies with the EBM
    energy = state.evaluate_ebm_energy(ebm)
    assert isinstance(energy, float)
    
    # 4. Ensure ContinuousLatentState can project using the DouglasRachfordPiNetLayer
    # Let's enforce equality constraint: x_0 + x_1 = 0.5
    eq_mat = np.zeros((1, dim))
    eq_mat[0, 0] = 1.0
    eq_mat[0, 1] = 1.0
    eq_tgt = np.array([0.5])
    
    projected_state = project_clara_state(
        state=state,
        equality_matrix=eq_mat,
        equality_target=eq_tgt
    )
    
    assert isinstance(projected_state, ContinuousLatentState)
    assert projected_state.z.shape == (dim,)
    
    # Verify projection
    proj_sum = float(projected_state.z[0] + projected_state.z[1])
    assert abs(proj_sum - 0.5) < 1e-4

