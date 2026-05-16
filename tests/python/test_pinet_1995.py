import json
import jax.numpy as jnp
from carnot.models.pinet_1995 import project_clara_state, build_experiment_1995_artifact
from carnot.pipeline.clara_v_schema import ContinuousLatentState, EnergyVector
import numpy as np

def test_pinet_projection_gradient_flow():
    # REQ-KONA-041 / SCENARIO-KONA-041
    # Check that gradient flows through the projection of a ContinuousLatentState
    
    # 1. Define synthetic continuous state
    state = ContinuousLatentState(
        z=np.array([1.5, -0.5], dtype=np.float64),
        energy=EnergyVector(components=np.array([0.0, 0.0], dtype=np.float64))
    )
    
    # 2. Define equality and inequality constraints for projection
    # E.g. z[0] + z[1] = 0.5 (Equality)
    equality_matrix = np.array([[1.0, 1.0]], dtype=np.float64)
    equality_target = np.array([0.5], dtype=np.float64)
    inequality_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    inequality_bound = np.array([1.0, 1.0], dtype=np.float64)
    
    # 3. Check forward pass
    projected_state = project_clara_state(
        state, 
        equality_matrix=equality_matrix,
        equality_target=equality_target,
        inequality_matrix=inequality_matrix,
        inequality_bound=inequality_bound
    )
    
    assert projected_state.z.shape == (2,)
    # projected_state.z should satisfy z[0] + z[1] = 0.5 approximately
    assert abs(projected_state.z[0] + projected_state.z[1] - 0.5) < 1e-4

    # 4. Check differentiability via jax.grad
    import jax
    def loss(z_input):
        temp_state = ContinuousLatentState(
            z=z_input,
            energy=EnergyVector(components=jnp.zeros_like(z_input))
        )
        proj = project_clara_state(
            temp_state,
            equality_matrix=equality_matrix,
            equality_target=equality_target,
            inequality_matrix=inequality_matrix,
            inequality_bound=inequality_bound
        )
        return jnp.sum(proj.z * proj.z)
    
    grad_fn = jax.grad(loss)
    grad = grad_fn(jnp.array([1.5, -0.5]))
    
    assert jnp.all(jnp.isfinite(grad))

def test_experiment_1995_artifact(tmp_path):
    artifact = build_experiment_1995_artifact(str(tmp_path / "experiment_1995_pinet_projection.json"))
    
    assert artifact["schema"] == "carnot.model_layer.v1"
    assert artifact["experiment"] == 1995
    assert artifact["honest_verdict"].startswith("SUCCESS:")
    
    with open(tmp_path / "experiment_1995_pinet_projection.json", "r") as f:
        data = json.load(f)
        assert data["experiment"] == 1995
