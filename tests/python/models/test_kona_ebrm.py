from carnot.models.kona_ebrm import KonaEBRM
import jax.numpy as jnp

def test_kona_ebrm():
    """Spec: REQ-KONA-040, SCENARIO-KONA-040"""
    model = KonaEBRM(trace_length=5, dim=2)
    init_trace = jnp.zeros((5, 2))
    target_final = jnp.array([1.0, -1.0])
    
    refined = model.refine_trace(init_trace, target_final, steps=200, lr=0.05)
    
    assert refined.shape == (5, 2)
    # The final step should be closer to target
    assert jnp.allclose(refined[-1], target_final, atol=0.3)
