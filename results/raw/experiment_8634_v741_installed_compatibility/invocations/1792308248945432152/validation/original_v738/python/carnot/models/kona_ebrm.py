import jax
import jax.numpy as jnp

class KonaEBRM:
    """Non-autoregressive Reasoning Model that refines an entire reasoning trace simultaneously.
    
    Spec: REQ-KONA-040, SCENARIO-KONA-040
    """
    def __init__(self, trace_length: int, dim: int):
        self.trace_length = trace_length
        self.dim = dim
        
    def energy(self, trace: jnp.ndarray, target_final: jnp.ndarray) -> jnp.ndarray:
        """Energy function: penalizes transitions and final state mismatch.
        trace: [trace_length, dim]
        """
        # Consistency: adjacent steps should be close or follow a specific logic
        # For a simple puzzle, let's say step i+1 should be step i passed through a non-linearity
        diff = trace[1:] - jnp.tanh(trace[:-1])
        transition_energy = jnp.sum(diff ** 2)
        
        # Final state should match target
        final_energy = jnp.sum((trace[-1] - target_final) ** 2)
        
        return transition_energy + final_energy

    def refine_trace(self, init_trace: jnp.ndarray, target_final: jnp.ndarray, steps: int = 100, lr: float = 0.1):
        """Refine trace via gradient descent on the energy."""
        def loss_fn(t):
            return self.energy(t, target_final)
            
        grad_fn = jax.grad(loss_fn)
        
        trace = init_trace
        for _ in range(steps):
            grads = grad_fn(trace)
            trace = trace - lr * grads
            
        return trace
