import numpy as np
from carnot.samplers.kinetic_langevin import KineticLangevinSampler

def test_kinetic_langevin_initialization():
    sampler = KineticLangevinSampler(gamma=2.0, kT=0.5, dt=0.05, n_steps=100, random_seed=42)
    assert sampler.gamma == 2.0
    assert sampler.kT == 0.5
    assert sampler.dt == 0.05
    assert sampler.n_steps == 100
    assert sampler.random_seed == 42

def test_kinetic_langevin_sample():
    sampler = KineticLangevinSampler(gamma=1.0, kT=1.0, dt=0.01, n_steps=10, random_seed=42)
    
    def grad_energy_fn(x):
        return x
        
    def project_fn(x):
        return np.clip(x, -1.0, 1.0)
        
    init_x = np.array([[0.5, -0.5], [2.0, -2.0]])
    final_x = sampler.sample(grad_energy_fn, init_x, project_fn)
    
    assert final_x.shape == (2, 2)
    assert np.all(final_x >= -1.0)
    assert np.all(final_x <= 1.0)

def test_kinetic_langevin_no_project():
    sampler = KineticLangevinSampler(gamma=1.0, kT=1.0, dt=0.01, n_steps=10, random_seed=42)
    
    def grad_energy_fn(x):
        return x
        
    init_x = np.array([[0.5, -0.5]])
    final_x = sampler.sample(grad_energy_fn, init_x)
    
    assert final_x.shape == (1, 2)
