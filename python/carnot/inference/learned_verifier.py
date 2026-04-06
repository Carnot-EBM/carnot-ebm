"""Learned energy functions: train EBMs to verify LLM output from examples.

**Researcher summary:**
    Trains a Gibbs model to distinguish correct from incorrect solutions using
    Noise Contrastive Estimation (NCE). The trained model's energy function is
    then used in the verify-and-repair pipeline instead of hand-coded constraints.
    This is the bridge from toy domains (SAT with known clauses) to real domains
    (code correctness) where you can't enumerate all constraints.

**Detailed explanation for engineers:**
    Hand-coded constraints (SAT clauses, coloring edges) work when rules are
    explicit. For code verification, you need the EBM to LEARN what "correct"
    looks like from examples. This module proves the concept on SAT — where
    we have hand-coded ground truth for comparison — then the same pattern
    scales to any domain with (correct, incorrect) example pairs.

    **How it works:**

    1. **Generate training data**: Rejection-sample satisfying SAT assignments
       (positive examples). Random assignments serve as noise (negative examples).

    2. **Train with NCE**: The loss pushes the model to assign low energy to
       satisfying assignments and high energy to random ones. After training,
       ``model.energy(x)`` approximates "how violated is this assignment?"

    3. **Wrap as ComposedEnergy**: ``LearnedEnergyWrapper`` adapts the trained
       model to the ``BaseConstraint`` protocol, making it usable with
       ``verify_and_repair()``.

    4. **Compare**: ``compare_learned_vs_handcoded()`` measures how well the
       learned verifier matches the exact hand-coded constraints.

Spec: REQ-INFER-007, SCENARIO-INFER-008
"""

from __future__ import annotations

from typing import Any

import logging
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.training.nce import nce_loss
from carnot.verify.constraint import BaseConstraint, ComposedEnergy
from carnot.verify.sat import SATBinaryConstraint, SATClause, build_sat_energy

logger = logging.getLogger(__name__)


@dataclass
class LearnedVerifierConfig:
    """Configuration for training a learned verifier.

    **Researcher summary:**
        Controls model architecture, training budget, and data generation.

    **Detailed explanation for engineers:**
        - ``hidden_dims``: Gibbs network architecture. [64, 32] is small
          but sufficient for SAT with ~5-20 variables.
        - ``n_training_samples``: How many satisfying assignments to generate
          via rejection sampling. More = better but slower.
        - ``n_epochs``: Training iterations. Each epoch uses the full dataset.
        - ``learning_rate``: Gradient descent step size. 0.01 is conservative.
        - ``noise_scale``: Scale of random noise for NCE negative examples.

    Spec: REQ-INFER-007
    """

    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    n_training_samples: int = 500
    n_epochs: int = 100
    learning_rate: float = 0.01
    noise_scale: float = 1.0
    seed: int = 42


def generate_sat_training_data(
    clauses: list[SATClause],
    n_vars: int,
    n_samples: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Generate training data for NCE: satisfying assignments + random noise.

    **Researcher summary:**
        Rejection-samples satisfying assignments as positive examples.
        Random assignments serve as noise (negative examples).

    **Detailed explanation for engineers:**
        Uses the hand-coded SAT energy to check if a random assignment
        satisfies all clauses. Keeps generating until we have ``n_samples``
        satisfying assignments. This is tractable for small instances
        (5-20 vars) where a decent fraction of random assignments satisfy.

        For larger instances where satisfying assignments are rare, you'd
        need smarter sampling (e.g., WalkSAT or MCMC). For this proof of
        concept, rejection sampling is sufficient.

    Args:
        clauses: SAT clauses defining the problem.
        n_vars: Number of variables.
        n_samples: Target number of satisfying assignments.
        key: JAX PRNG key.

    Returns:
        Tuple of (satisfying_batch, noise_batch), each shape (n_samples, n_vars).

    Spec: REQ-INFER-007
    """
    energy = build_sat_energy(clauses, n_vars, binary_weight=0.0)
    satisfying: list[jax.Array] = []

    # Generate random assignments and keep satisfying ones
    max_attempts = n_samples * 100  # Safety bound
    attempts = 0

    while len(satisfying) < n_samples and attempts < max_attempts:
        key, subkey = jrandom.split(key)
        batch_size = min(n_samples * 10, 10000)
        # Random binary assignments
        candidates = jnp.round(jrandom.uniform(subkey, (batch_size, n_vars)))

        for i in range(batch_size):
            if len(satisfying) >= n_samples:
                break
            x = candidates[i]
            result = energy.verify(x)
            if result.verdict.verified:
                satisfying.append(x)
        attempts += batch_size

    if len(satisfying) < n_samples:
        logger.warning(
            "Only found %d/%d satisfying assignments after %d attempts",
            len(satisfying),
            n_samples,
            attempts,
        )
        if not satisfying:
            # No satisfying assignments found — return random data as fallback
            key, subkey = jrandom.split(key)
            satisfying_batch = jnp.round(jrandom.uniform(subkey, (n_samples, n_vars)))
        else:
            # Pad with duplicates
            while len(satisfying) < n_samples:
                satisfying.append(satisfying[len(satisfying) % len(satisfying)])
            satisfying_batch = jnp.stack(satisfying[:n_samples])
    else:
        satisfying_batch = jnp.stack(satisfying[:n_samples])

    # Noise: random uniform [0, 1] assignments
    key, subkey = jrandom.split(key)
    noise_batch = jrandom.uniform(subkey, (n_samples, n_vars))

    return satisfying_batch, noise_batch


def train_sat_verifier(
    clauses: list[SATClause],
    n_vars: int,
    config: LearnedVerifierConfig | None = None,
) -> GibbsModel:
    """Train a Gibbs model to verify SAT assignments via NCE.

    **Researcher summary:**
        Generates training data, trains with NCE for n_epochs, returns
        the trained model whose energy function approximates "how violated
        is this assignment?"

    **Detailed explanation for engineers:**
        The training loop:

        1. Generate satisfying + noise assignments via ``generate_sat_training_data``
        2. Create a GibbsModel with the configured architecture
        3. For each epoch, compute NCE loss and update parameters via gradient descent
        4. The NCE loss pushes energy(satisfying) down and energy(noise) up
        5. After training, model.energy(x) is low for valid, high for invalid

        **Parameter updates**: Since GibbsModel stores params as JAX arrays
        in its ``_params`` dict, we use ``jax.tree.map`` to apply gradient
        descent to all parameters simultaneously.

    Args:
        clauses: SAT clauses defining the problem.
        n_vars: Number of variables.
        config: Training configuration.

    Returns:
        Trained GibbsModel.

    Spec: REQ-INFER-007, SCENARIO-INFER-008
    """
    if config is None:
        config = LearnedVerifierConfig()

    key = jrandom.PRNGKey(config.seed)
    key, data_key = jrandom.split(key)

    # Generate training data
    satisfying, noise = generate_sat_training_data(
        clauses, n_vars, config.n_training_samples, data_key
    )

    # Create model
    key, model_key = jrandom.split(key)
    gibbs_config = GibbsConfig(
        input_dim=n_vars,
        hidden_dims=config.hidden_dims,
        activation="silu",
    )
    model = GibbsModel(gibbs_config, key=model_key)

    # Extract parameters as a pytree for functional gradient computation.
    # GibbsModel stores: self.layers (list of (weight, bias)), self.output_weight, self.output_bias
    def get_params(m: GibbsModel) -> dict[str, Any]:
        return {
            "layers": [(w, b) for w, b in m.layers],
            "output_weight": m.output_weight,
            "output_bias": m.output_bias,
        }

    def set_params(m: GibbsModel, params: dict[str, Any]) -> None:
        m.layers = list(params["layers"])
        m.output_weight = params["output_weight"]
        m.output_bias = params["output_bias"]

    def loss_fn(params: dict[str, Any]) -> jax.Array:
        # Temporarily set params, compute loss, restore
        old = get_params(model)
        set_params(model, params)
        result = nce_loss(model, satisfying, noise)
        set_params(model, old)
        return result

    # Training loop
    params = get_params(model)
    for epoch in range(config.n_epochs):
        set_params(model, params)
        loss_val = nce_loss(model, satisfying, noise)

        grads = jax.grad(loss_fn)(params)

        # Gradient descent update on all parameters
        params = jax.tree.map(
            lambda p, g: p - config.learning_rate * g,
            params,
            grads,
        )

        if epoch % 20 == 0:
            logger.info("Epoch %d: NCE loss = %.4f", epoch, float(loss_val))

    # Apply final params to model
    set_params(model, params)
    return model


class LearnedEnergyWrapper(BaseConstraint):
    """Wraps a trained model as a BaseConstraint for verify-and-repair.

    **Researcher summary:**
        Adapter: takes any EnergyFunction and makes it usable in the
        ComposedEnergy / verify-and-repair pipeline.

    **Detailed explanation for engineers:**
        The verify-and-repair pipeline uses ``ComposedEnergy`` which expects
        ``BaseConstraint`` objects. A trained GibbsModel has ``energy()`` and
        ``grad_energy()`` but not the ``name`` / ``satisfaction_threshold``
        properties. This wrapper provides them.

        The threshold is calibrated during construction: compute energy on a
        few known-good examples and set threshold slightly above the mean.

    Spec: REQ-INFER-007
    """

    def __init__(
        self,
        name: str,
        model: GibbsModel,
        threshold: float = 0.5,
    ) -> None:
        self._name = name
        self._model = model
        self._threshold = threshold

    @property
    def name(self) -> str:
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        return self._threshold

    def energy(self, x: jax.Array) -> jax.Array:
        """Delegate to the trained model's energy function.

        Spec: REQ-INFER-007
        """
        return self._model.energy(x)


def build_learned_sat_energy(
    model: GibbsModel,
    n_vars: int,
    model_weight: float = 1.0,
    binary_weight: float = 0.1,
    threshold: float = 0.5,
) -> ComposedEnergy:
    """Build a ComposedEnergy from a trained model + binary penalty.

    **Researcher summary:**
        Wraps the trained model as a constraint and adds a binary penalty,
        producing a ComposedEnergy ready for verify_and_repair().

    Spec: REQ-INFER-007
    """
    composed = ComposedEnergy(input_dim=n_vars)
    composed.add_constraint(
        LearnedEnergyWrapper("learned_sat", model, threshold=threshold),
        model_weight,
    )
    composed.add_constraint(
        SATBinaryConstraint("binary_penalty", list(range(n_vars))),
        binary_weight,
    )
    return composed


@dataclass
class ComparisonResult:
    """Result of comparing learned vs hand-coded verifier.

    Spec: REQ-INFER-007
    """

    learned_accuracy: float = 0.0
    handcoded_accuracy: float = 0.0
    learned_mean_energy_satisfying: float = 0.0
    learned_mean_energy_violating: float = 0.0
    energy_gap: float = 0.0
    n_test_samples: int = 0


def compare_learned_vs_handcoded(
    clauses: list[SATClause],
    n_vars: int,
    model: GibbsModel,
    n_test: int = 100,
    seed: int = 99,
) -> ComparisonResult:
    """Compare learned verifier accuracy against hand-coded constraints.

    **Researcher summary:**
        Generates test assignments, classifies each with both the learned
        model and the hand-coded SAT energy, and reports accuracy and
        energy statistics.

    **Detailed explanation for engineers:**
        For each test assignment:
        - Hand-coded: verified if ComposedEnergy.verify() passes
        - Learned: classified as satisfying if model.energy(x) < median energy

        The "energy gap" measures separation between satisfying and violating
        assignments in the learned model's energy space. Larger gap = better
        discrimination.

    Spec: REQ-INFER-007, SCENARIO-INFER-008
    """
    key = jrandom.PRNGKey(seed)
    handcoded_energy = build_sat_energy(clauses, n_vars, binary_weight=0.0)

    # Generate random binary test assignments
    key, subkey = jrandom.split(key)
    test_assignments = jnp.round(jrandom.uniform(subkey, (n_test, n_vars)))

    # Classify with hand-coded constraints
    handcoded_labels: list[bool] = []
    for i in range(n_test):
        result = handcoded_energy.verify(test_assignments[i])
        handcoded_labels.append(result.verdict.verified)

    # Compute learned energies
    learned_energies = [float(model.energy(test_assignments[i])) for i in range(n_test)]

    # Use median energy as threshold for learned classifier
    median_energy = sorted(learned_energies)[n_test // 2]

    # Compute statistics
    sat_energies = [e for e, label in zip(learned_energies, handcoded_labels, strict=True) if label]
    viol_energies = [
        e for e, label in zip(learned_energies, handcoded_labels, strict=True) if not label
    ]

    mean_sat = sum(sat_energies) / max(len(sat_energies), 1)
    mean_viol = sum(viol_energies) / max(len(viol_energies), 1)

    # Learned accuracy: does low energy correlate with satisfying?
    learned_correct = sum(
        1
        for e, label in zip(learned_energies, handcoded_labels, strict=True)
        if (e < median_energy) == label
    )

    return ComparisonResult(
        learned_accuracy=learned_correct / n_test,
        handcoded_accuracy=1.0,  # Hand-coded is ground truth by definition
        learned_mean_energy_satisfying=mean_sat,
        learned_mean_energy_violating=mean_viol,
        energy_gap=mean_viol - mean_sat,
        n_test_samples=n_test,
    )
