import jax
import jax.numpy as jnp
import z3
import time
from typing import Tuple
from carnot.solvers.pinet_prototype import douglas_rachford_splitting

class HybridVerifier:
    def __init__(self, A: jnp.ndarray, b: jnp.ndarray):
        """
        Initialize the Hybrid Verifier with linear equality constraints Ax = b.
        """
        self.A = jnp.array(A)
        self.b = jnp.array(b)
        self.num_vars = self.A.shape[1]
        
    def generate_prediction(self, seed: int = 0) -> jnp.ndarray:
        """
        Neural generator predicts an initial assignment.
        """
        key = jax.random.PRNGKey(seed)
        return jax.random.uniform(key, shape=(self.num_vars,))
        
    def project_pinet(self, prediction: jnp.ndarray) -> jnp.ndarray:
        """
        PiNet projects the prediction to satisfy the constraints continuously.
        """
        projected = douglas_rachford_splitting(self.A, self.b, max_iter=200)
        return projected
        
    def verify_z3(self, projected: jnp.ndarray, threshold: float = 0.5) -> bool:
        """
        Z3 verifies the boolean interpretation of the projected continuous constraints.
        """
        # Round the projected continuous values to boolean
        bool_vals = (projected > threshold).astype(int)
        
        # Build Z3 formulation
        solver = z3.Solver()
        z3_vars = [z3.Int(f"x_{i}") for i in range(self.num_vars)]
        
        # Domain constraints (boolean)
        for var in z3_vars:
            solver.add(z3.Or(var == 0, var == 1))
            
        # Add constraint Ax = b
        A_np = self.A.tolist()
        b_np = self.b.tolist()
        
        for row_idx, row in enumerate(A_np):
            lhs = z3.Sum([int(row[col_idx]) * z3_vars[col_idx] for col_idx in range(self.num_vars)])
            solver.add(lhs == int(b_np[row_idx]))
            
        # Add the specific assignment we got from projection
        for i, val in enumerate(bool_vals):
            solver.add(z3_vars[i] == int(val))
            
        return solver.check() == z3.sat

    def run_pipeline(self, seed: int = 0) -> Tuple[bool, float]:
        """
        Run the full pipeline and return (is_verified, latency).
        """
        start_time = time.time()
        
        pred = self.generate_prediction(seed)
        proj = self.project_pinet(pred)
        is_verified = self.verify_z3(proj)
        
        latency = time.time() - start_time
        return is_verified, latency
