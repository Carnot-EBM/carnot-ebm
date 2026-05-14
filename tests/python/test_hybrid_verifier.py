import jax.numpy as jnp
from carnot.solvers.hybrid_verifier import HybridVerifier

def test_hybrid_verifier_pipeline():
    """
    Test the hybrid verifier pipeline.
    References: REQ-VERIFY-1673, SCENARIO-VERIFY-1673
    """
    # Simple constraints: x0 + x1 = 1, x1 + x2 = 1, x0 = 1
    # Solution should be x0=1, x1=0, x2=1
    A = jnp.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0]
    ])
    b = jnp.array([1.0, 1.0, 1.0])
    
    verifier = HybridVerifier(A, b)
    
    # 1. Test generate_prediction
    pred = verifier.generate_prediction(seed=42)
    assert pred.shape == (3,)
    
    # 2. Test project_pinet
    proj = verifier.project_pinet(pred)
    assert proj.shape == (3,)
    
    # 3. Test verify_z3
    # Our simple projection might not be perfect Boolean due to DR convergence,
    # but the solution to these equations is exactly 1, 0, 1.
    is_verified = verifier.verify_z3(proj)
    assert is_verified is True
    
    # 4. Test run_pipeline
    is_verified_pipe, latency = verifier.run_pipeline(seed=42)
    assert is_verified_pipe is True
    assert latency > 0.0
