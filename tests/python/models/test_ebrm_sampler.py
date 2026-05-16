import jax.numpy as jnp
from carnot.models.ebrm_sampler import EBRMSampler

def test_ebrm_sampler_small_logic_task():
    """
    Test the EBRM sampler against a small logic task.
    
    SCENARIO-KONA-040: Verify gradient-based latent refinement.
    REQ-KONA-040: Model refines latents.
    """
    # Target state representing the correct logical conclusion
    target = jnp.array([1.0, -1.0, 0.5])
    
    # Energy function: Mean squared error to target
    def energy_fn(latents):
        return jnp.sum((latents - target) ** 2)
        
    sampler = EBRMSampler(energy_fn=energy_fn, learning_rate=0.1, steps=100)
    
    # Initial state (random/uninformed)
    init_latents = jnp.array([0.0, 0.0, 0.0])
    
    # Run sampler
    refined_latents = sampler.sample(init_latents)
    
    # Assert convergence
    assert jnp.allclose(refined_latents, target, atol=1e-3)
