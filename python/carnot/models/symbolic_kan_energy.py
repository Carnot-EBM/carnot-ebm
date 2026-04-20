"""SymbolicKANEnergy — KAN energy tier with discrete symbolic activations.

**Researcher summary (arXiv 2603.23854, Symbolic-KAN, March 2026):**
    Standard KAN models use continuous B-spline activations which are opaque:
    the learned edge function is a curve with no interpretable form. Symbolic-KAN
    replaces those splines with a discrete selection from a small set of named
    symbolic functions (linear, quadratic, tanh, relu, abs). The result is a
    human-readable energy function — instead of an opaque curve, the user sees
    '2.3 * x + 1.1' or 'abs(1.5 * x - 0.2)'. This makes constraint violations
    explainable: the pipeline can say 'constraint fires because |x1+x2-1| = 0.87'.

**Why symbolic selection works:**
    For each input variable, we try each candidate activation type and measure
    how well it fits the training residuals (MSE). The type with lowest MSE wins.
    This is a discrete model selection step, not continuous optimisation — no
    gradient is needed to pick the symbolic form. Only the scale (coefficient)
    and shift (bias) are fitted by least squares once the form is chosen.

**Why this is useful for EBM verification:**
    Carnot's users currently cannot understand WHY the pipeline flags a response.
    SymbolicKANEnergy provides an explain() string that maps directly to the
    constraint structure in natural language. The energy is no longer a black box.

Spec: REQ-MODEL-020, REQ-MODEL-021,
      SCENARIO-MODEL-030, SCENARIO-MODEL-031, SCENARIO-MODEL-032
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp
import numpy as np

# The five symbolic activation types supported. Adding more types here requires
# corresponding changes to _apply_activation() and _fit_single_type().
SymbolicActivationType = Literal["linear", "quadratic", "tanh", "relu", "abs"]

_ALL_TYPES: list[SymbolicActivationType] = ["linear", "quadratic", "tanh", "relu", "abs"]


def _apply_activation(act_type: SymbolicActivationType, coef: float, bias: float, x: np.ndarray) -> np.ndarray:
    """Apply a named activation with scalar coefficient and bias to a 1-D array.

    Returns coef * f(x) + bias where f is determined by act_type.
    The coefficient and bias are fitted by least squares AFTER the nonlinear
    transform is computed, so 'coef' always scales the post-transform output.
    """
    if act_type == "linear":
        z = x
    elif act_type == "quadratic":
        z = x ** 2
    elif act_type == "tanh":
        z = np.tanh(x)
    elif act_type == "relu":
        z = np.maximum(0.0, x)
    elif act_type == "abs":
        z = np.abs(x)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")
    return coef * z + bias


def _make_formula_str(act_type: SymbolicActivationType, coef: float, bias: float) -> str:
    """Build a human-readable formula string for one activation.

    Examples:
        linear(2.3, 1.1)    -> '2.30 * x + 1.10'
        quadratic(1.5, 0.0) -> '1.50 * x^2 + 0.00'
        abs(-0.7, 0.3)      -> 'abs(-0.70 * x) + 0.30'
    """
    c = f"{coef:.2f}"
    b = f"{bias:.2f}"
    if act_type == "linear":
        return f"{c} * x + {b}"
    elif act_type == "quadratic":
        return f"{c} * x^2 + {b}"
    elif act_type == "tanh":
        return f"tanh({c} * x) + {b}"
    elif act_type == "relu":
        return f"relu({c} * x) + {b}"
    elif act_type == "abs":
        return f"abs({c} * x) + {b}"
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


@dataclass
class SymbolicActivation:
    """Fitted symbolic activation for one input variable.

    Attributes:
        activation_type: Which symbolic form was selected (minimum MSE winner).
        coefficient: Scale applied to the post-transform value. Fitted by OLS.
        bias: Additive offset. Fitted by OLS.
        formula_str: Human-readable formula auto-generated at fit time.
    """

    activation_type: SymbolicActivationType
    coefficient: float
    bias: float
    formula_str: str = field(init=False)

    def __post_init__(self) -> None:
        self.formula_str = _make_formula_str(self.activation_type, self.coefficient, self.bias)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Evaluate this activation on a 1-D numpy array."""
        return _apply_activation(self.activation_type, self.coefficient, self.bias, x)


def _fit_single_type(
    act_type: SymbolicActivationType, x_data: np.ndarray, y_data: np.ndarray
) -> tuple[float, float, float]:
    """Fit coefficient and bias for one activation type by ordinary least squares.

    Returns (coefficient, bias, mse).

    We minimise ||coef * f(x) + bias - y||^2 by solving the 2-column linear
    system [f(x), 1] @ [coef, bias]^T = y via numpy lstsq.
    """
    if act_type == "linear":
        z = x_data
    elif act_type == "quadratic":
        z = x_data ** 2
    elif act_type == "tanh":
        z = np.tanh(x_data)
    elif act_type == "relu":
        z = np.maximum(0.0, x_data)
    elif act_type == "abs":
        z = np.abs(x_data)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")

    A = np.column_stack([z, np.ones_like(z)])
    result, _, _, _ = np.linalg.lstsq(A, y_data, rcond=None)
    coef, bias = float(result[0]), float(result[1])
    pred = coef * z + bias
    mse = float(np.mean((pred - y_data) ** 2))
    return coef, bias, mse


class SymbolicKANLayer:
    """One layer of Symbolic-KAN: one fitted activation per input variable.

    Each variable gets its own independently fitted symbolic function. The layer
    output is the sum of all per-variable activations — this is the same additive
    decomposition used by KAEM but with symbolic rather than spline functions.

    Attributes:
        n_vars: Number of input variables this layer handles.
        activation_candidates: Symbolic types to try during fitting.
        activations: List of fitted SymbolicActivation (one per variable), set
            after fit_activation() has been called for each variable.
    """

    def __init__(
        self,
        n_vars: int,
        activation_candidates: list[SymbolicActivationType] | None = None,
    ) -> None:
        self.n_vars = n_vars
        self.activation_candidates: list[SymbolicActivationType] = (
            activation_candidates if activation_candidates is not None else list(_ALL_TYPES)
        )
        # Populated by fit_activation() calls; one entry per variable.
        self.activations: list[SymbolicActivation] = []

    def fit_activation(self, x_data: jnp.ndarray, y_data: jnp.ndarray) -> SymbolicActivation:
        """Select the best symbolic activation for ONE variable by minimum MSE.

        Tries each candidate type, fits OLS coefficients, picks the winner.

        Args:
            x_data: 1-D array of input values for this variable (N,).
            y_data: 1-D array of target output values (N,).

        Returns:
            The fitted SymbolicActivation with lowest MSE on the training data.
        """
        x_np = np.asarray(x_data, dtype=np.float64).ravel()
        y_np = np.asarray(y_data, dtype=np.float64).ravel()

        best_act: SymbolicActivation | None = None
        best_mse = float("inf")

        for act_type in self.activation_candidates:
            coef, bias, mse = _fit_single_type(act_type, x_np, y_np)
            if mse < best_mse:
                best_mse = mse
                best_act = SymbolicActivation(
                    activation_type=act_type,
                    coefficient=coef,
                    bias=bias,
                )

        assert best_act is not None  # guaranteed because candidates is non-empty
        return best_act

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply fitted per-variable activations and sum them.

        Args:
            x: Input array of shape (n_vars,) or (batch, n_vars).

        Returns:
            Scalar (or batch vector) — sum of all per-variable activations.
        """
        x_np = np.asarray(x, dtype=np.float64)
        if x_np.ndim == 1:
            # Single sample: x_np shape (n_vars,)
            total = sum(act.apply(x_np[i : i + 1])[0] for i, act in enumerate(self.activations))
            return jnp.array(total, dtype=jnp.float32)
        else:
            # Batch: x_np shape (batch, n_vars)
            totals = np.zeros(x_np.shape[0])
            for i, act in enumerate(self.activations):
                totals += act.apply(x_np[:, i])
            return jnp.array(totals, dtype=jnp.float32)

    def get_formula(self) -> str:
        """Return a human-readable formula joining all per-variable activations.

        Variables are labelled x1, x2, ... (1-indexed for readability).
        Example: '2.30 * x1 + 1.10 + abs(-0.70 * x2) + 0.30'
        """
        import re

        parts = []
        for i, act in enumerate(self.activations):
            # Replace the bare token 'x' (word boundary) with 'x{i+1}'.
            var_formula = re.sub(r"\bx\b", f"x{i + 1}", act.formula_str)
            parts.append(var_formula)
        return " + ".join(parts)


class SymbolicKANEnergy:
    """Energy-based model using Symbolic-KAN: discrete symbolic activations.

    Replaces the opaque B-spline activations of standard KAN with human-readable
    symbolic functions (linear, quadratic, tanh, relu, abs). This makes the energy
    function fully interpretable: explain() returns a formula string that a user
    can read and understand without EBM expertise.

    **Architecture:**
        n_layers stacked SymbolicKANLayers. Each layer takes the raw input x and
        produces a scalar energy contribution. The total energy is the sum across
        layers. This multi-layer structure lets each layer capture a different
        aspect of the constraint (e.g., layer 1 captures linear trend, layer 2
        captures residual nonlinearity).

    **Fitting strategy:**
        Layer i is fitted on the residuals left by layers 0..i-1. This greedy
        forward-fitting approach avoids the need for backprop through symbolic
        selection (which is not differentiable).

    Attributes:
        energy_interpretable: Always True — distinguishes this tier from spline KAN.
    """

    energy_interpretable: bool = True

    def __init__(self, n_vars: int, n_layers: int = 2) -> None:
        self.n_vars = n_vars
        self.n_layers = n_layers
        self.layers: list[SymbolicKANLayer] = [
            SymbolicKANLayer(n_vars) for _ in range(n_layers)
        ]

    def energy(self, x: jnp.ndarray) -> float:
        """Compute the scalar energy for a single input vector x.

        Args:
            x: Input array of shape (n_vars,).

        Returns:
            Scalar energy value (sum of all layer outputs).
        """
        total = 0.0
        for layer in self.layers:
            total += float(layer.forward(x))
        return total

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """Fit all layers to training data using greedy residual fitting.

        Each layer is fitted on the residuals not explained by previous layers.
        Within each layer, per-variable activations are fitted independently.

        Args:
            X: Input matrix of shape (N, n_vars).
            y: Target energy values of shape (N,).
        """
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64).ravel()

        residuals = y_np.copy()

        for layer in self.layers:
            # For this layer, fit each variable's activation to the residuals.
            # Per-variable target: residuals / n_vars so each variable carries
            # an equal share of the unexplained variance.
            var_target = residuals / self.n_vars

            for var_idx in range(self.n_vars):
                act = layer.fit_activation(
                    jnp.array(X_np[:, var_idx]),
                    jnp.array(var_target),
                )
                layer.activations.append(act)

            # Update residuals: subtract this layer's predictions
            layer_pred = np.asarray(layer.forward(jnp.array(X_np)), dtype=np.float64)
            residuals = residuals - layer_pred

    def explain(self) -> str:
        """Return the full symbolic energy formula as a human-readable string.

        Example output:
            E(x) = 2.10*x1 + 0.80*x2^2 + tanh(1.30*x3) + ...

        Returns:
            Human-readable formula string. Non-empty after fit() has been called.
        """
        layer_formulas = [layer.get_formula() for layer in self.layers if layer.activations]
        body = " + ".join(f"({f})" for f in layer_formulas)
        return f"E(x) = {body}"
