import numpy as np
from carnot.samplers.de_psgld import DEPSGLDSampler

def test_de_psgld_sampler():
    sampler = DEPSGLDSampler(dt=0.01, n_steps=10, random_seed=42)
    init_x = np.zeros((2, 2))
    
    def grad_energy_fn(x):
        return x
        
    def project_fn(x):
        return np.clip(x, -1, 1)
        
    samples = sampler.sample(grad_energy_fn, init_x, project_fn)
    assert samples.shape == (2, 2)
    assert np.all(samples >= -1.0)
    assert np.all(samples <= 1.0)
