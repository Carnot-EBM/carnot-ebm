import numpy as np
from carnot.samplers.dikin_langevin import DikinLangevinSampler

def test_dikin_langevin_sampler_basic():
    """Test that DikinLangevinSampler runs and respects constraints."""
    sampler = DikinLangevinSampler(kT=1.0, dt=0.01, n_steps=100, random_seed=42)
    
    # 2D quadratic bowl
    J = np.eye(2)
    def grad_energy_fn(x):
        return x @ J
        
    def project_fn(x):
        return np.clip(x, -1, 1)
        
    init_x = np.array([[0.5, 0.5], [-0.5, -0.5]])
    
    samples = sampler.sample(grad_energy_fn, init_x, project_fn)
    
    assert samples.shape == init_x.shape
    assert np.all(samples >= -1.0)
    assert np.all(samples <= 1.0)
