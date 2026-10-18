"""Self-improving code verification via autoresearch.

**Researcher summary:**
    Connects the code verifier to the autoresearch pipeline. Defines a
    benchmark function, baseline builder, hypothesis templates for
    architecture exploration, and a full autoresearch loop that improves
    code verification accuracy autonomously.

**Detailed explanation for engineers:**
    This module is the capstone: it connects all three Carnot pillars into
    a self-improvement loop for code verification:

    1. **EBM constraints** (from ``verify.python_types``): Execution-based
       energy terms that check Python code correctness.
    2. **Learned verification** (from ``inference.code_verifier``): NCE-trained
       Gibbs model that generalizes from correct/buggy code patterns.
    3. **Autoresearch** (from ``autoresearch.orchestrator``): Autonomous
       hypothesis evaluation loop with sandbox, three-gate evaluator,
       experiment logging, and circuit breaker.

    **How the loop works:**
    1. Train a baseline code verifier with default hyperparameters.
    2. Evaluate it on held-out test data (accuracy = 1 - energy).
    3. Generate hypotheses that tweak hyperparameters (wider model, more data, etc.).
    4. The autoresearch loop evaluates each hypothesis in a sandbox.
    5. If a hypothesis improves accuracy, it becomes the new baseline.

    **Why is this interesting?**
    The code verifier is verifying *code*, and the autoresearch loop is
    improving the *verifier*. This is a meta-level self-improvement: the
    system learns to be a better judge of code correctness, autonomously.

Spec: REQ-CODE-005
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
from carnot.autoresearch.orchestrator import AutoresearchConfig, LoopResult, run_loop
from carnot.inference.code_verifier import (
    CodeVerifierConfig,
    generate_code_training_data,
    train_code_verifier,
)


def code_verification_benchmark(
    model: object,
    test_correct: object,
    test_buggy: object,
) -> dict[str, float]:
    """Evaluate a code verifier model on held-out test data.

    **Researcher summary:**
        Computes accuracy of the learned verifier on test pairs. Returns
        a benchmark dict compatible with the autoresearch evaluator:
        ``{"final_energy": 1 - accuracy, "wall_clock_seconds": ...}``.

    **Detailed explanation for engineers:**
        For each pair of (correct_embedding, buggy_embedding):
        - If model.energy(correct) < model.energy(buggy), it's a correct
          classification (the model correctly identifies which is the real code).
        - Accuracy = fraction of correct classifications.
        - Final energy = 1 - accuracy (so lower is better, matching the
          autoresearch convention where lower energy = better performance).

        The ``model``, ``test_correct``, and ``test_buggy`` are typed as
        ``object`` to keep the function signature flexible for sandbox use,
        but they should be a GibbsModel and JAX arrays respectively.

    Args:
        model: Trained GibbsModel (typed as object for sandbox compatibility).
        test_correct: Correct code embeddings, shape (n_test, vocab_size).
        test_buggy: Buggy code embeddings, shape (n_test, vocab_size).

    Returns:
        Dict with "final_energy" (1 - accuracy) and "wall_clock_seconds".

    Spec: REQ-CODE-005
    """
    import jax

    start = time.monotonic()

    # Cast to JAX arrays for vmap
    correct_energies = jax.vmap(model.energy)(test_correct)  # type: ignore[attr-defined]
    buggy_energies = jax.vmap(model.energy)(test_buggy)  # type: ignore[attr-defined]

    # Accuracy: fraction where correct energy < buggy energy
    n_correct = int((correct_energies < buggy_energies).sum())
    n_total = len(correct_energies)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    wall_clock = time.monotonic() - start

    return {
        "final_energy": 1.0 - accuracy,
        "wall_clock_seconds": wall_clock,
    }


def build_code_verification_baselines(
    config: CodeVerifierConfig | None = None,
) -> tuple[BaselineRecord, dict[str, Any]]:
    """Train a default verifier and create baseline performance record.

    **Researcher summary:**
        Trains a code verifier with default config, evaluates on test data,
        and packages the results as a BaselineRecord for the autoresearch
        evaluator.

    **Detailed explanation for engineers:**
        This function bootstraps the autoresearch loop:
        1. Generate training data with the default config
        2. Train a Gibbs model via NCE
        3. Generate held-out test data (different seed)
        4. Evaluate accuracy on the test data
        5. Package as a BaselineRecord + context dict

        The context dict contains the test data and model so that hypothesis
        code can access them in the sandbox.

    Args:
        config: Verifier configuration. Uses defaults if None.

    Returns:
        Tuple of (baseline_record, benchmark_data) where benchmark_data
        contains test arrays and the trained model.

    Spec: REQ-CODE-005
    """
    if config is None:
        config = CodeVerifierConfig()

    # Generate training and test data
    correct, buggy = generate_code_training_data(
        config.n_training_samples, config.seed, config.vocab_size
    )
    test_correct, test_buggy = generate_code_training_data(
        50, config.seed + 1000, config.vocab_size
    )

    # Train baseline model
    model = train_code_verifier(correct, buggy, config)

    # Evaluate baseline
    metrics = code_verification_benchmark(model, test_correct, test_buggy)

    # Package as baseline record
    baselines = BaselineRecord(version="0.1.0")
    baselines.benchmarks["code_verification"] = BenchmarkMetrics(
        benchmark_name="code_verification",
        final_energy=metrics["final_energy"],
        convergence_steps=config.n_epochs,
        wall_clock_seconds=metrics["wall_clock_seconds"],
    )

    benchmark_data: dict[str, Any] = {
        "test_correct": test_correct,
        "test_buggy": test_buggy,
        "vocab_size": config.vocab_size,
        "seed": config.seed,
    }

    return baselines, benchmark_data


def code_verification_hypothesis_template(strategy: str) -> str:
    """Return hypothesis code for a given improvement strategy.

    **Researcher summary:**
        Generates Python code strings that the autoresearch sandbox will
        execute. Each strategy tweaks a hyperparameter of the code verifier
        training pipeline.

    **Detailed explanation for engineers:**
        The autoresearch loop evaluates hypothesis code in a sandbox. Each
        hypothesis must define a ``run(benchmark_data) -> dict`` function
        that trains a verifier with modified hyperparameters and returns
        benchmark metrics.

        Available strategies:
        - ``"wider_model"``: Use wider hidden layers (256, 128 instead of 128, 64)
        - ``"deeper_model"``: Add a third hidden layer
        - ``"more_epochs"``: Train for 200 epochs instead of 100
        - ``"more_data"``: Use 400 training samples instead of 200

    Args:
        strategy: One of "wider_model", "deeper_model", "more_epochs", "more_data".

    Returns:
        Python code string with a ``run(benchmark_data)`` function.

    Raises:
        ValueError: If strategy is not recognized.

    Spec: REQ-CODE-005
    """
    templates: dict[str, str] = {
        "wider_model": _WIDER_MODEL_TEMPLATE,
        "deeper_model": _DEEPER_MODEL_TEMPLATE,
        "more_epochs": _MORE_EPOCHS_TEMPLATE,
        "more_data": _MORE_DATA_TEMPLATE,
    }

    if strategy not in templates:
        msg = f"Unknown strategy: {strategy}. Choose from: {list(templates.keys())}"
        raise ValueError(msg)

    return templates[strategy]


_HYPOTHESIS_PREAMBLE = """
import time
import jax
import jax.numpy as jnp
import jax.random as jrandom
from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.training.nce import nce_loss
from carnot.inference.code_verifier import generate_code_training_data
"""

_WIDER_MODEL_TEMPLATE = (
    _HYPOTHESIS_PREAMBLE
    + """
def run(benchmark_data):
    vocab_size = benchmark_data.get("vocab_size", 256)
    seed = benchmark_data.get("seed", 42)
    correct, buggy = generate_code_training_data(200, seed, vocab_size)
    cfg = GibbsConfig(input_dim=vocab_size, hidden_dims=[256, 128], activation="silu")
    model = GibbsModel(cfg, key=jrandom.PRNGKey(seed))
    for epoch in range(100):
        def loss_fn(layers, out_w, out_b):
            def energy_fn(x):
                h = x
                for w, b in layers:
                    h = (w @ h + b)
                    h = h * jax.nn.sigmoid(h)
                return out_w @ h + out_b
            return nce_loss(energy_fn, correct, buggy)
        layers = [(w, b) for w, b in model.layers]
        out_w = model.output_weight
        out_b = jnp.float32(model.output_bias)
        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(layers, out_w, out_b)
        new_layers = []
        for (w, b), (gw, gb) in zip(model.layers, grads[0]):
            new_layers.append((w - 0.01 * gw, b - 0.01 * gb))
        model.layers = new_layers
        model.output_weight = out_w - 0.01 * grads[1]
        model.output_bias = float(out_b - 0.01 * grads[2])
    test_c = benchmark_data["test_correct"]
    test_b = benchmark_data["test_buggy"]
    start = time.monotonic()
    c_e = jax.vmap(model.energy)(test_c)
    b_e = jax.vmap(model.energy)(test_b)
    acc = float((c_e < b_e).sum()) / len(c_e)
    wall = time.monotonic() - start
    return {"code_verification": {"final_energy": 1.0 - acc, "wall_clock_seconds": wall}}
"""
)

_DEEPER_MODEL_TEMPLATE = (
    _HYPOTHESIS_PREAMBLE
    + """
def run(benchmark_data):
    vocab_size = benchmark_data.get("vocab_size", 256)
    seed = benchmark_data.get("seed", 42)
    correct, buggy = generate_code_training_data(200, seed, vocab_size)
    cfg = GibbsConfig(input_dim=vocab_size, hidden_dims=[128, 64, 32], activation="silu")
    model = GibbsModel(cfg, key=jrandom.PRNGKey(seed))
    for epoch in range(100):
        def loss_fn(layers, out_w, out_b):
            def energy_fn(x):
                h = x
                for w, b in layers:
                    h = (w @ h + b)
                    h = h * jax.nn.sigmoid(h)
                return out_w @ h + out_b
            return nce_loss(energy_fn, correct, buggy)
        layers = [(w, b) for w, b in model.layers]
        out_w = model.output_weight
        out_b = jnp.float32(model.output_bias)
        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(layers, out_w, out_b)
        new_layers = []
        for (w, b), (gw, gb) in zip(model.layers, grads[0]):
            new_layers.append((w - 0.01 * gw, b - 0.01 * gb))
        model.layers = new_layers
        model.output_weight = out_w - 0.01 * grads[1]
        model.output_bias = float(out_b - 0.01 * grads[2])
    test_c = benchmark_data["test_correct"]
    test_b = benchmark_data["test_buggy"]
    start = time.monotonic()
    c_e = jax.vmap(model.energy)(test_c)
    b_e = jax.vmap(model.energy)(test_b)
    acc = float((c_e < b_e).sum()) / len(c_e)
    wall = time.monotonic() - start
    return {"code_verification": {"final_energy": 1.0 - acc, "wall_clock_seconds": wall}}
"""
)

_MORE_EPOCHS_TEMPLATE = (
    _HYPOTHESIS_PREAMBLE
    + """
def run(benchmark_data):
    vocab_size = benchmark_data.get("vocab_size", 256)
    seed = benchmark_data.get("seed", 42)
    correct, buggy = generate_code_training_data(200, seed, vocab_size)
    cfg = GibbsConfig(input_dim=vocab_size, hidden_dims=[128, 64], activation="silu")
    model = GibbsModel(cfg, key=jrandom.PRNGKey(seed))
    for epoch in range(200):
        def loss_fn(layers, out_w, out_b):
            def energy_fn(x):
                h = x
                for w, b in layers:
                    h = (w @ h + b)
                    h = h * jax.nn.sigmoid(h)
                return out_w @ h + out_b
            return nce_loss(energy_fn, correct, buggy)
        layers = [(w, b) for w, b in model.layers]
        out_w = model.output_weight
        out_b = jnp.float32(model.output_bias)
        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(layers, out_w, out_b)
        new_layers = []
        for (w, b), (gw, gb) in zip(model.layers, grads[0]):
            new_layers.append((w - 0.01 * gw, b - 0.01 * gb))
        model.layers = new_layers
        model.output_weight = out_w - 0.01 * grads[1]
        model.output_bias = float(out_b - 0.01 * grads[2])
    test_c = benchmark_data["test_correct"]
    test_b = benchmark_data["test_buggy"]
    start = time.monotonic()
    c_e = jax.vmap(model.energy)(test_c)
    b_e = jax.vmap(model.energy)(test_b)
    acc = float((c_e < b_e).sum()) / len(c_e)
    wall = time.monotonic() - start
    return {"code_verification": {"final_energy": 1.0 - acc, "wall_clock_seconds": wall}}
"""
)

_MORE_DATA_TEMPLATE = (
    _HYPOTHESIS_PREAMBLE
    + """
def run(benchmark_data):
    vocab_size = benchmark_data.get("vocab_size", 256)
    seed = benchmark_data.get("seed", 42)
    correct, buggy = generate_code_training_data(400, seed, vocab_size)
    cfg = GibbsConfig(input_dim=vocab_size, hidden_dims=[128, 64], activation="silu")
    model = GibbsModel(cfg, key=jrandom.PRNGKey(seed))
    for epoch in range(100):
        def loss_fn(layers, out_w, out_b):
            def energy_fn(x):
                h = x
                for w, b in layers:
                    h = (w @ h + b)
                    h = h * jax.nn.sigmoid(h)
                return out_w @ h + out_b
            return nce_loss(energy_fn, correct, buggy)
        layers = [(w, b) for w, b in model.layers]
        out_w = model.output_weight
        out_b = jnp.float32(model.output_bias)
        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(layers, out_w, out_b)
        new_layers = []
        for (w, b), (gw, gb) in zip(model.layers, grads[0]):
            new_layers.append((w - 0.01 * gw, b - 0.01 * gb))
        model.layers = new_layers
        model.output_weight = out_w - 0.01 * grads[1]
        model.output_bias = float(out_b - 0.01 * grads[2])
    test_c = benchmark_data["test_correct"]
    test_b = benchmark_data["test_buggy"]
    start = time.monotonic()
    c_e = jax.vmap(model.energy)(test_c)
    b_e = jax.vmap(model.energy)(test_b)
    acc = float((c_e < b_e).sum()) / len(c_e)
    wall = time.monotonic() - start
    return {"code_verification": {"final_energy": 1.0 - acc, "wall_clock_seconds": wall}}
"""
)


@dataclass
class _AutoresearchCodeConfig:
    """Internal config for autoresearch code verification loop.

    Spec: REQ-CODE-005
    """

    verifier_config: CodeVerifierConfig = field(default_factory=CodeVerifierConfig)
    n_hypotheses: int = 5
    strategies: list[str] = field(
        default_factory=lambda: ["wider_model", "deeper_model", "more_epochs", "more_data"]
    )


def run_code_verification_autoresearch(
    config: CodeVerifierConfig | None = None,
    n_hypotheses: int = 5,
) -> LoopResult:
    """Run the full autoresearch loop to improve code verification.

    **Researcher summary:**
        Bootstraps a baseline verifier, generates improvement hypotheses
        (wider model, deeper model, more epochs, more data), and runs the
        autoresearch orchestrator to evaluate them.

    **Detailed explanation for engineers:**
        This is the top-level function that ties everything together:

        1. **Build baselines**: Train a default code verifier, evaluate on
           held-out test data, record the baseline performance.
        2. **Generate hypotheses**: Create hypothesis code strings from
           templates, each exploring a different hyperparameter change.
        3. **Run autoresearch**: Pass hypotheses to ``run_loop()``, which
           sandboxes, evaluates, and logs each one.
        4. **Return**: The LoopResult shows how many hypotheses were
           accepted/rejected, whether the circuit breaker tripped, and
           the updated baselines.

        The number of hypotheses is capped at ``n_hypotheses``. Strategies
        are cycled if more hypotheses are requested than strategies exist.

    Args:
        config: Verifier configuration for the baseline. Uses defaults if None.
        n_hypotheses: Number of hypotheses to evaluate. Default 5.

    Returns:
        LoopResult from the autoresearch orchestrator.

    Spec: REQ-CODE-005
    """
    if config is None:
        config = CodeVerifierConfig()

    # Build baselines
    baselines, benchmark_data = build_code_verification_baselines(config)

    # Generate hypothesis list
    strategies = ["wider_model", "deeper_model", "more_epochs", "more_data"]
    hypotheses: list[tuple[str, str]] = []
    for i in range(n_hypotheses):
        strategy = strategies[i % len(strategies)]
        code = code_verification_hypothesis_template(strategy)
        hypotheses.append((f"code_verifier_{strategy}", code))

    # Run autoresearch loop with relaxed evaluation tolerance
    auto_config = AutoresearchConfig(
        max_iterations=n_hypotheses,
        max_consecutive_failures=n_hypotheses + 1,
        energy_regression_tolerance=0.5,
    )

    return run_loop(
        hypotheses=hypotheses,
        baselines=baselines,
        benchmark_data=benchmark_data,
        config=auto_config,
    )
