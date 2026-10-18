"""Symbolic-KAN routing layer for discrete primitive embeddings.

Spec references: REQ-KAN-1749, SCENARIO-KAN-1749.

This prototype maps the Symbolic-KAN idea from arXiv:2603.23854 into Carnot's
JAX tensor space. Each route forms a learned scalar projection, evaluates a
finite library of analytic primitives on that scalar, and stores the symbolic
choice as a gate tensor over primitive names. Soft gates support training-time
mixtures; hard gates expose the one-hot symbolic structure used for
interpretation and downstream artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


def _repo_root() -> Path:
    """Return the repository root from this nested KAN package module."""

    return Path(__file__).resolve().parents[4]


DEFAULT_RESULT_PATH = _repo_root() / "results/experiment_1749_symbolic_kan.json"
SUPPORTED_PRIMITIVES = ("identity", "square", "sin", "cos", "abs")


def _primitive_value(name: str, projection: jax.Array) -> jax.Array:
    """Evaluate one named analytic primitive on a projection tensor."""

    if name == "identity":
        return projection
    if name == "square":
        return projection**2
    if name == "sin":
        return jnp.sin(projection)
    if name == "cos":
        return jnp.cos(projection)
    if name == "abs":
        return jnp.abs(projection)
    raise ValueError(f"unknown primitive: {name}")


@dataclass(frozen=True)
class SymbolicKANConfig:
    """Configuration for a Symbolic-KAN primitive routing layer."""

    input_dim: int = 2
    n_routes: int = 2
    primitives: tuple[str, ...] = ("identity", "square", "sin")
    temperature: float = 1.0

    @property
    def n_primitives(self) -> int:
        """Number of analytic primitives available to each route."""

        return len(self.primitives)

    def validate(self) -> None:
        """Validate configuration before tensors are allocated or evaluated."""

        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.n_routes <= 0:
            raise ValueError("n_routes must be > 0")
        if not self.primitives:
            raise ValueError("at least one primitive is required")
        unknown = [name for name in self.primitives if name not in SUPPORTED_PRIMITIVES]
        if unknown:
            raise ValueError(f"unknown primitive: {unknown[0]}")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")


@dataclass(frozen=True)
class SymbolicKANParams:
    """Trainable tensors for scalar projections and primitive routing gates."""

    projection_weights: jax.Array
    projection_bias: jax.Array
    route_logits: jax.Array
    route_scales: jax.Array
    output_bias: jax.Array


@dataclass(frozen=True)
class SymbolicRoutingOutput:
    """Forward-pass tensor report for Symbolic-KAN routing."""

    energy: jax.Array
    projections: jax.Array
    primitive_values: jax.Array
    gates: jax.Array
    route_values: jax.Array


class SymbolicRoutingLayer:
    """Route scalar projections through a discrete symbolic primitive library."""

    def __init__(
        self,
        config: SymbolicKANConfig,
        key: jax.Array | None = None,
        params: SymbolicKANParams | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.params = self.init_params(key) if params is None else params
        self._validate_params(self.params)

    def init_params(self, key: jax.Array | None = None) -> SymbolicKANParams:
        """Create deterministic JAX tensors for a new routing layer."""

        if key is None:
            key = jrandom.PRNGKey(0)
        weight_key, logit_key = jrandom.split(key)
        scale = jnp.sqrt(jnp.asarray(self.config.input_dim, dtype=jnp.float32))
        return SymbolicKANParams(
            projection_weights=jrandom.normal(
                weight_key,
                (self.config.n_routes, self.config.input_dim),
                dtype=jnp.float32,
            )
            / scale,
            projection_bias=jnp.zeros((self.config.n_routes,), dtype=jnp.float32),
            route_logits=0.01
            * jrandom.normal(
                logit_key,
                (self.config.n_routes, self.config.n_primitives),
                dtype=jnp.float32,
            ),
            route_scales=jnp.ones((self.config.n_routes,), dtype=jnp.float32),
            output_bias=jnp.array(0.0, dtype=jnp.float32),
        )

    def _validate_params(self, params: SymbolicKANParams) -> None:
        """Ensure parameter tensors match the layer's routing shape."""

        expected = (
            (self.config.n_routes, self.config.input_dim),
            (self.config.n_routes,),
            (self.config.n_routes, self.config.n_primitives),
            (self.config.n_routes,),
            (),
        )
        actual = (
            tuple(params.projection_weights.shape),
            tuple(params.projection_bias.shape),
            tuple(params.route_logits.shape),
            tuple(params.route_scales.shape),
            tuple(jnp.asarray(params.output_bias).shape),
        )
        if actual != expected:
            raise ValueError(f"params shapes must match {expected}, got {actual}")

    def primitive_values(self, projections: jax.Array) -> jax.Array:
        """Evaluate every primitive for every route projection.

        Args:
            projections: Tensor with shape `(batch, n_routes)`.

        Returns:
            Tensor with shape `(batch, n_routes, n_primitives)`.
        """

        return jnp.stack(
            [_primitive_value(name, projections) for name in self.config.primitives],
            axis=-1,
        )

    def gates(
        self,
        hard: bool = False,
        temperature: float | None = None,
        params: SymbolicKANParams | None = None,
    ) -> jax.Array:
        """Return soft or hard primitive gates for each route."""

        params = self.params if params is None else params
        gate_temperature = self.config.temperature if temperature is None else temperature
        if gate_temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        soft = jax.nn.softmax(params.route_logits / gate_temperature, axis=-1)
        if hard:
            indices = jnp.argmax(soft, axis=-1)
            return jax.nn.one_hot(indices, self.config.n_primitives, dtype=jnp.float32)
        return soft

    def structure_embedding(self, hard: bool = True) -> jax.Array:
        """Embed symbolic primitive choices as a route-by-primitive gate tensor."""

        return self.gates(hard=hard)

    def selected_primitives(self, params: SymbolicKANParams | None = None) -> tuple[str, ...]:
        """Return the primitive name selected by each route's maximum logit."""

        params = self.params if params is None else params
        indices = np.asarray(jnp.argmax(params.route_logits, axis=-1), dtype=np.int32)
        return tuple(self.config.primitives[int(index)] for index in indices)

    def forward(
        self,
        x: jax.Array,
        hard: bool = False,
        temperature: float | None = None,
        return_aux: bool = False,
        params: SymbolicKANParams | None = None,
    ) -> jax.Array | SymbolicRoutingOutput:
        """Evaluate Symbolic-KAN routing on a vector or batch tensor."""

        params = self.params if params is None else params
        self._validate_params(params)
        x_tensor = jnp.asarray(x, dtype=jnp.float32)
        if x_tensor.ndim not in (1, 2):
            raise ValueError("x must be a 1-D or 2-D tensor")
        if x_tensor.shape[-1] != self.config.input_dim:
            raise ValueError(f"x last dimension must equal input_dim={self.config.input_dim}")

        was_vector = x_tensor.ndim == 1
        batch = x_tensor[jnp.newaxis, :] if was_vector else x_tensor
        projections = batch @ params.projection_weights.T + params.projection_bias
        primitive_values = self.primitive_values(projections)
        gates = self.gates(hard=hard, temperature=temperature, params=params)
        route_values = jnp.sum(primitive_values * gates[jnp.newaxis, :, :], axis=-1)
        energies = route_values @ params.route_scales + params.output_bias

        if return_aux:
            return SymbolicRoutingOutput(
                energy=energies[0] if was_vector else energies,
                projections=projections[0] if was_vector else projections,
                primitive_values=primitive_values[0] if was_vector else primitive_values,
                gates=gates,
                route_values=route_values[0] if was_vector else route_values,
            )
        return energies[0] if was_vector else energies

    def symbolic_regularization(self, hard: bool = False) -> jax.Array:
        """Return normalized gate entropy; lower means sharper symbolic choices."""

        gates = self.gates(hard=hard)
        clipped = jnp.clip(gates, 1e-8, 1.0)
        entropy = -jnp.sum(gates * jnp.log(clipped), axis=-1)
        normalizer = jnp.log(jnp.asarray(self.config.n_primitives, dtype=jnp.float32))
        return jnp.mean(entropy / normalizer)

    def discrete_structure(self) -> list[dict[str, object]]:
        """Return JSON-safe metadata for each discretized symbolic route."""

        hard_gates = np.asarray(self.structure_embedding(hard=True), dtype=np.float64)
        weights = np.asarray(self.params.projection_weights, dtype=np.float64)
        biases = np.asarray(self.params.projection_bias, dtype=np.float64)
        scales = np.asarray(self.params.route_scales, dtype=np.float64)
        selected = self.selected_primitives()
        return [
            {
                "route_index": route_index,
                "primitive": selected[route_index],
                "gate_weight": float(hard_gates[route_index].max()),
                "projection_weights": weights[route_index].tolist(),
                "projection_bias": float(biases[route_index]),
                "route_scale": float(scales[route_index]),
            }
            for route_index in range(self.config.n_routes)
        ]

    def explain(self) -> str:
        """Return a compact formula-like explanation of hard-routed structure."""

        parts = []
        weights = np.asarray(self.params.projection_weights, dtype=np.float64)
        for route_index, primitive in enumerate(self.selected_primitives()):
            projection = " + ".join(
                f"{float(weight):.3f}*feature_{feature_index}"
                for feature_index, weight in enumerate(weights[route_index])
            )
            parts.append(f"route_{route_index}: {primitive}({projection})")
        return "; ".join(parts)


def _to_list(array: jax.Array) -> list[object]:
    """Convert a tensor to a JSON-safe nested list of Python floats."""

    return np.asarray(array, dtype=np.float64).tolist()


def build_experiment_1749_artifact(run_date: str = "20260510") -> dict[str, object]:
    """Build the stable Exp 1749 Symbolic-KAN routing artifact payload."""

    config = SymbolicKANConfig(input_dim=2, n_routes=2, primitives=("identity", "square", "sin"))
    params = SymbolicKANParams(
        projection_weights=jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        projection_bias=jnp.array([0.0, 0.0], dtype=jnp.float32),
        route_logits=jnp.array([[0.0, 0.0, 12.0], [0.0, 12.0, 0.0]], dtype=jnp.float32),
        route_scales=jnp.array([1.0, 1.0], dtype=jnp.float32),
        output_bias=jnp.array(0.0, dtype=jnp.float32),
    )
    layer = SymbolicRoutingLayer(config, params=params)

    inputs = jnp.array([[0.0, 0.0], [0.5, 0.25], [1.0, -0.5]], dtype=jnp.float32)
    report = layer.forward(inputs, hard=True, return_aux=True)
    targets = jnp.sin(inputs[:, 0]) + inputs[:, 1] ** 2
    max_abs_error = float(jnp.max(jnp.abs(report.energy - targets)))
    hard_gates = np.asarray(layer.structure_embedding(hard=True), dtype=np.float64)

    return {
        "schema": "carnot.symbolic_kan.experiment_1749.v1",
        "status": "complete",
        "experiment_id": 1749,
        "run_date": run_date,
        "spec_traces": ["REQ-KAN-1749", "SCENARIO-KAN-1749"],
        "paper": "arXiv:2603.23854",
        "module": "python/carnot/models/kan/symbolic_kan.py",
        "artifact_path": "results/experiment_1749_symbolic_kan.json",
        "config": {
            "input_dim": config.input_dim,
            "n_routes": config.n_routes,
            "primitives": list(config.primitives),
            "temperature": config.temperature,
        },
        "selected_primitives": list(layer.selected_primitives()),
        "structure_embedding_shape": list(hard_gates.shape),
        "hard_gates_one_hot": bool(
            np.allclose(hard_gates.sum(axis=1), 1.0)
            and np.all((hard_gates == 0.0) | (hard_gates == 1.0))
        ),
        "soft_regularization": float(layer.symbolic_regularization(hard=False)),
        "hard_regularization": abs(float(layer.symbolic_regularization(hard=True))),
        "discrete_structure": layer.discrete_structure(),
        "explanation": layer.explain(),
        "toy_mapping": {
            "target": "sin(feature_0) + square(feature_1)",
            "inputs": _to_list(inputs),
            "energies": _to_list(report.energy),
            "targets": _to_list(targets),
            "max_abs_error": max_abs_error,
        },
        "honest_verdict": "complete: symbolic_kan_routing_layer_embeds_discrete_primitives",
    }


def write_experiment_1749_artifact(
    output_path: str | Path = DEFAULT_RESULT_PATH,
    run_date: str = "20260510",
) -> dict[str, object]:
    """Write the stable Exp 1749 artifact to disk."""

    artifact = build_experiment_1749_artifact(run_date=run_date)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


__all__ = [
    "DEFAULT_RESULT_PATH",
    "SUPPORTED_PRIMITIVES",
    "SymbolicKANConfig",
    "SymbolicKANParams",
    "SymbolicRoutingLayer",
    "SymbolicRoutingOutput",
    "build_experiment_1749_artifact",
    "write_experiment_1749_artifact",
]
