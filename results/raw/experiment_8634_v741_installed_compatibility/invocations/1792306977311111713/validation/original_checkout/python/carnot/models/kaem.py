"""Kolmogorov-Arnold Energy Model (KAEM) structure."""
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jrandom

class KAEMEnergy:
    """KAEM energy model using 1D B-splines instead of dense layers."""
    def __init__(self, n_vars: int, n_knots: int = 10, key: jax.Array | None = None):
        if n_vars < 1:
            raise ValueError("n_vars must be >= 1")
        if n_knots < 3:
            raise ValueError("n_knots must be >= 3")
        self.n_vars = n_vars
        self.n_knots = n_knots
        if key is None:
            key = jrandom.PRNGKey(0)
        
        # Initialize spline control points for each variable
        # shape: (n_vars, n_knots)
        self.control_points = jrandom.normal(key, (n_vars, n_knots)) * 0.1

    def _eval_spline(self, ctrl: jax.Array, x: jax.Array) -> jax.Array:
        # Evaluate 1D B-spline (piecewise linear interpolation for simplicity)
        # x is assumed to be in [-1, 1]
        x_scaled = (x + 1.0) / 2.0 * (self.n_knots - 1)
        idx = jnp.floor(x_scaled).astype(jnp.int32)
        idx = jnp.clip(idx, 0, self.n_knots - 2)
        t = x_scaled - idx
        
        val = ctrl[idx] * (1.0 - t) + ctrl[idx + 1] * t
        return val

    def energy(self, x: jax.Array) -> jax.Array:
        """Evaluate energy for a single sample x of shape (n_vars,)."""
        # Sum of univariate 1D splines
        total_energy = jnp.array(0.0)
        for i in range(self.n_vars):
            total_energy += self._eval_spline(self.control_points[i], x[i])
        return total_energy

    def inverse_transform_sample(self, n_samples: int, key: jax.Array) -> jax.Array:
        """Sample using inverse transform sampling to bypass MCMC."""
        N_QUAD = 256
        uniforms = jrandom.uniform(key, (n_samples, self.n_vars))
        samples = np.zeros((n_samples, self.n_vars), dtype=np.float32)
        
        # We process each variable independently
        for i in range(self.n_vars):
            ctrl = np.array(self.control_points[i])
            grid = np.linspace(-1.0, 1.0, N_QUAD)
            
            # evaluate energy for variable i
            x_scaled = (grid + 1.0) / 2.0 * (self.n_knots - 1)
            idx = np.floor(x_scaled).astype(np.int32)
            idx = np.clip(idx, 0, self.n_knots - 2)
            t = x_scaled - idx
            energies = ctrl[idx] * (1.0 - t) + ctrl[idx + 1] * t
            
            energies = energies - np.max(energies)
            density = np.exp(-energies)
            
            # Cumulative trapezoid integration
            cdf_vals = np.zeros(N_QUAD, dtype=np.float64)
            for k in range(1, N_QUAD):
                dx = grid[k] - grid[k - 1]
                cdf_vals[k] = cdf_vals[k - 1] + 0.5 * (density[k - 1] + density[k]) * dx
            
            total = cdf_vals[-1]
            cdf_vals /= total
            
            for s in range(n_samples):
                u = float(uniforms[s, i])
                idx_cdf = int(np.searchsorted(cdf_vals, u))
                idx_cdf = int(np.clip(idx_cdf, 1, N_QUAD - 1))
                
                x0, x1 = grid[idx_cdf - 1], grid[idx_cdf]
                c0, c1 = cdf_vals[idx_cdf - 1], cdf_vals[idx_cdf]
                
                if abs(c1 - c0) < 1e-12:
                    samples[s, i] = x0
                else:
                    t_interp = (u - c0) / (c1 - c0)
                    samples[s, i] = x0 + t_interp * (x1 - x0)
                    
        return jnp.array(samples)
