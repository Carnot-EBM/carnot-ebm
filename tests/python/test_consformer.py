import jax
import jax.numpy as jnp
from carnot.models.consformer import ConsFormerRefiner, refinement_loop

def test_consformer_refiner_forward():
    model = ConsFormerRefiner(d_model=16, num_heads=2, num_layers=1)
    
    n_nodes = 4
    x = jnp.array([[0.1], [0.5], [1.2], [-0.5]])
    
    # 4x4 adjacency matrix
    adj_matrix = jnp.array([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    
    key = jax.random.PRNGKey(0)
    params = model.init(key, x, adj_matrix)
    
    update = model.apply(params, x, adj_matrix)
    
    assert update.shape == (n_nodes,)
    # Update should be real values
    assert not jnp.isnan(update).any()

def test_refinement_loop():
    model = ConsFormerRefiner(d_model=16, num_heads=2, num_layers=1)
    n_nodes = 3
    init_x = jnp.array([0.0, 1.0, 2.0])
    
    adj_matrix = jnp.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    
    key = jax.random.PRNGKey(1)
    # Init expects (n_nodes, 1)
    x_in = jnp.expand_dims(init_x, axis=-1)
    params = model.init(key, x_in, adj_matrix)
    
    num_steps = 5
    final_x, trajectory = refinement_loop(
        params, model, init_x, adj_matrix, num_steps=num_steps, step_size=0.1
    )
    
    assert final_x.shape == (n_nodes,)
    assert trajectory.shape == (num_steps, n_nodes)
    
    # After step > 0, final_x should not be exactly init_x if there's an update
    # Note: it's possible but unlikely that init_x is a fixed point of random init.
