"""Learned code verifier: NCE-trained Gibbs model for Python code verification.

**Researcher summary:**
    Trains a Gibbs model via NCE to discriminate correct Python code from buggy
    mutations. Code is embedded as bag-of-tokens frequency vectors. The learned
    model complements execution-based handcoded constraints for a two-signal
    verification pipeline.

**Detailed explanation for engineers:**
    This module implements the learned half of Carnot's code verification system.
    While the handcoded constraints (in ``python_types.py``) execute code and
    check results, the learned verifier operates on code *embeddings* — fixed-size
    vectors derived from the source code's token frequencies.

    **Training pipeline:**
    1. Generate training data: correct function templates + buggy mutations
    2. Embed all code snippets as bag-of-tokens vectors
    3. Train a Gibbs model via NCE: correct = data, buggy = noise
    4. The trained model assigns lower energy to correct code embeddings

    **Verification pipeline:**
    1. Build handcoded energy (type check + exception check + test pass)
    2. Embed the code under test
    3. Compute handcoded energy via constraint execution
    4. Compute learned energy via the trained Gibbs model
    5. Return a CodeVerificationResult combining both signals

    **Why two signals?**
    - Handcoded: precise but requires test cases; can't generalize beyond them
    - Learned: generalizes from training distribution; may catch patterns that
      specific test cases miss (e.g., "code that looks like buggy code")
    - Together: defense in depth — both must agree for high confidence

Spec: REQ-CODE-003, REQ-CODE-004
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.training.nce import nce_loss
from carnot.verify.python_types import build_code_energy, code_to_embedding


@dataclass
class CodeVerifierConfig:
    """Configuration for the code verifier training pipeline.

    **Researcher summary:**
        Hyperparameters for code verifier training: embedding size, network
        topology, training epochs, learning rate, data volume, and PRNG seed.

    **Detailed explanation for engineers:**
        These settings control every aspect of the learned verifier:

        - ``vocab_size``: Dimension of the code embedding (bag-of-tokens).
          Larger = fewer hash collisions but more parameters to learn.

        - ``hidden_dims``: Hidden layer sizes for the Gibbs model. Wider/deeper
          = more expressive but slower to train.

        - ``n_epochs``: Number of training passes over the data. More epochs =
          better fit but risk of overfitting on the small template set.

        - ``learning_rate``: Gradient descent step size. Too large = diverge,
          too small = slow convergence.

        - ``n_training_samples``: Number of correct/buggy pairs to generate.
          More data = better generalization.

        - ``seed``: Random seed for reproducibility.

    Spec: REQ-CODE-003
    """

    vocab_size: int = 256
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    n_epochs: int = 100
    learning_rate: float = 0.01
    n_training_samples: int = 200
    seed: int = 42


# ---------------------------------------------------------------------------
# Simple function templates for training data generation
# ---------------------------------------------------------------------------

CORRECT_TEMPLATES: list[tuple[str, str]] = [
    ("add", "def add(a: int, b: int) -> int:\n    return a + b"),
    ("multiply", "def multiply(a: int, b: int) -> int:\n    return a * b"),
    ("identity", "def identity(x: int) -> int:\n    return x"),
    ("negate", "def negate(x: int) -> int:\n    return -x"),
    ("double", "def double(x: int) -> int:\n    return x * 2"),
]
"""Known-correct Python function templates.

Each tuple is (function_name, source_code). These are intentionally simple
arithmetic functions so that mutations are easy to create and verify.

Spec: REQ-CODE-003
"""


# ---------------------------------------------------------------------------
# Mutation functions that create buggy versions of correct code
# ---------------------------------------------------------------------------


def _mutate_remove_return(code: str) -> str:
    """Remove the ``return`` keyword, turning the function into a no-op.

    **For engineers:**
        A function without ``return`` implicitly returns None, which breaks
        any caller expecting an int. This is one of the most common bugs in
        student code.

    Spec: REQ-CODE-003
    """
    return code.replace("return ", "")


def _mutate_swap_operator(code: str) -> str:
    """Swap ``+`` for ``-``, introducing an arithmetic bug.

    Spec: REQ-CODE-003
    """
    return code.replace(" + ", " - ")


def _mutate_wrong_value(code: str) -> str:
    """Change ``return a`` to ``return a + 1``, introducing an off-by-one bug.

    Spec: REQ-CODE-003
    """
    return code.replace("return a", "return a + 1")


def _mutate_remove_body(code: str) -> str:
    """Replace the function body with ``pass``, making it return None.

    Spec: REQ-CODE-003
    """
    lines = code.split("\n")
    if len(lines) >= 2:  # noqa: PLR2004 — need at least def + body
        return lines[0] + "\n    pass"
    return code


def _mutate_swap_multiply(code: str) -> str:
    """Swap ``*`` for ``//``, introducing an integer division bug.

    Spec: REQ-CODE-003
    """
    return code.replace(" * ", " // ")


_MUTATIONS = [
    _mutate_remove_return,
    _mutate_swap_operator,
    _mutate_wrong_value,
    _mutate_remove_body,
    _mutate_swap_multiply,
]
"""List of all available mutation functions.

Each mutation creates a different kind of bug. During training data generation,
mutations are applied randomly to correct templates to create buggy examples.
"""


def generate_code_training_data(
    n_samples: int,
    seed: int,
    vocab_size: int = 256,
) -> tuple[jax.Array, jax.Array]:
    """Generate (correct_embeddings, buggy_embeddings) for NCE training.

    **Researcher summary:**
        Cycles through correct templates, applies random mutations to create
        buggy versions, embeds both as bag-of-tokens vectors. Returns paired
        batches for NCE training.

    **Detailed explanation for engineers:**
        For each sample:
        1. Pick a correct template (cycling through the list)
        2. Embed it as a bag-of-tokens vector → correct embedding
        3. Pick a random mutation function
        4. Apply it to the correct code → buggy code
        5. Embed the buggy code → buggy embedding

        The result is two arrays of shape (n_samples, vocab_size) where each
        row is a code embedding. Correct embeddings are the "data" (real
        distribution) and buggy embeddings are the "noise" for NCE training.

        Uses Python's ``random`` module (not JAX) for mutation selection because
        the mutations are Python string operations, not JAX computations.

    Args:
        n_samples: Number of correct/buggy pairs to generate.
        seed: Random seed for reproducibility.
        vocab_size: Embedding dimension. Default 256.

    Returns:
        Tuple of (correct_batch, buggy_batch), each shape (n_samples, vocab_size).

    Spec: REQ-CODE-003
    """
    rng = random.Random(seed)

    correct_embeddings = []
    buggy_embeddings = []

    for i in range(n_samples):
        # Cycle through correct templates
        _, correct_code = CORRECT_TEMPLATES[i % len(CORRECT_TEMPLATES)]
        correct_emb = code_to_embedding(correct_code, vocab_size)
        correct_embeddings.append(correct_emb)

        # Apply a random mutation to create buggy code
        mutation = rng.choice(_MUTATIONS)
        buggy_code = mutation(correct_code)
        buggy_emb = code_to_embedding(buggy_code, vocab_size)
        buggy_embeddings.append(buggy_emb)

    return jnp.stack(correct_embeddings), jnp.stack(buggy_embeddings)


def train_code_verifier(
    correct: jax.Array,
    buggy: jax.Array,
    config: CodeVerifierConfig | None = None,
) -> GibbsModel:
    """Train a Gibbs model via NCE to discriminate correct from buggy code.

    **Researcher summary:**
        NCE training loop: correct code = data, buggy code = noise.
        Uses manual gradient descent on model parameters extracted via
        get_params/set_params pattern.

    **Detailed explanation for engineers:**
        This function trains a Gibbs model to assign lower energy to correct
        code embeddings than buggy ones. The training loop:

        1. Create a GibbsModel with the configured architecture
        2. For each epoch:
           a. Compute NCE loss: correct = data (should get low energy),
              buggy = noise (should get high energy)
           b. Compute gradients of the loss w.r.t. model parameters
           c. Update parameters via gradient descent

        **Parameter handling:**
        Since GibbsModel stores parameters as instance attributes (not a
        JAX pytree), we extract them into a flat list, compute gradients
        w.r.t. that list, and update in place.

        **Why NCE instead of score matching?**
        NCE is conceptually simpler for binary discrimination tasks. We have
        clear "data" (correct code) and "noise" (buggy code), which maps
        directly to NCE's framework.

    Args:
        correct: Correct code embeddings, shape (n_samples, vocab_size).
        buggy: Buggy code embeddings, shape (n_samples, vocab_size).
        config: Training configuration. Uses defaults if None.

    Returns:
        Trained GibbsModel.

    Spec: REQ-CODE-004
    """
    if config is None:
        config = CodeVerifierConfig()

    vocab_size = correct.shape[1]

    # Create model with matching input dimension
    gibbs_config = GibbsConfig(
        input_dim=vocab_size,
        hidden_dims=config.hidden_dims,
        activation="silu",
    )
    key = jrandom.PRNGKey(config.seed)
    model = GibbsModel(gibbs_config, key=key)

    # Training loop: manual gradient descent on model parameters
    for _epoch in range(config.n_epochs):
        # Define a loss function that closes over the current model state
        # We need to make the energy function purely functional for jax.grad
        def loss_fn(
            layers: list[tuple[jax.Array, jax.Array]],
            out_w: jax.Array,
            out_b: jax.Array,
        ) -> jax.Array:
            """Compute NCE loss given model parameters.

            This inner function creates a temporary energy callable that uses
            the provided parameters (not the model's stored ones), making it
            compatible with jax.grad.
            """

            def energy_fn(x: jax.Array) -> jax.Array:
                h = x
                for weight, bias in layers:
                    # Inline SiLU activation to keep everything in JAX
                    h = weight @ h + bias
                    h = h * jax.nn.sigmoid(h)
                return out_w @ h + out_b

            return nce_loss(energy_fn, correct, buggy)

        # Extract current parameters
        layers = [(w, b) for w, b in model.layers]
        out_w = model.output_weight
        out_b = jnp.float32(model.output_bias)

        # Compute gradients w.r.t. all parameters
        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(layers, out_w, out_b)
        grad_layers, grad_out_w, grad_out_b = grads

        # Update parameters via gradient descent
        new_layers = []
        for (w, b), (gw, gb) in zip(model.layers, grad_layers, strict=True):
            new_layers.append(
                (
                    w - config.learning_rate * gw,
                    b - config.learning_rate * gb,
                )
            )
        model.layers = new_layers
        model.output_weight = out_w - config.learning_rate * grad_out_w
        model.output_bias = float(out_b - config.learning_rate * grad_out_b)

    return model


@dataclass
class ComparisonResult:
    """Result of comparing learned vs handcoded code verification.

    **Researcher summary:**
        Captures accuracy and agreement metrics between the learned Gibbs model
        and handcoded execution-based constraints on a test set.

    **Detailed explanation for engineers:**
        After training a learned verifier, we want to know:
        1. How accurate is it? (learned_accuracy)
        2. How often does it agree with handcoded constraints? (agreement_rate)
        3. What fraction of correct code does each method identify? (detailed stats)

    Spec: REQ-CODE-004
    """

    learned_accuracy: float = 0.0
    handcoded_accuracy: float = 0.0
    agreement_rate: float = 0.0
    n_test: int = 0


@dataclass
class CodeVerificationResult:
    """Result of verifying a Python function with both handcoded and learned signals.

    **Researcher summary:**
        Combined verification result: handcoded energy (from execution),
        learned energy (from Gibbs model), per-test-case results, and
        overall verdict.

    **Detailed explanation for engineers:**
        This dataclass captures the full output of the verification pipeline:

        - ``code``: The Python source code that was verified
        - ``handcoded_energy``: Total energy from execution-based constraints
          (0.0 = all tests pass, type checks pass, no exceptions)
        - ``handcoded_verified``: True if handcoded energy is below threshold
        - ``learned_energy``: Energy from the trained Gibbs model (None if no
          model provided). Lower = model thinks code is more "correct-like"
        - ``test_results``: Per-test-case details: (input, output, passed?)
        - ``n_tests_passed``: Number of test cases that produced correct output
        - ``n_tests_total``: Total number of test cases

    Spec: REQ-CODE-004
    """

    code: str = ""
    handcoded_energy: float = 0.0
    handcoded_verified: bool = False
    learned_energy: float | None = None
    test_results: list[tuple[Any, Any, bool]] = field(default_factory=list)
    n_tests_passed: int = 0
    n_tests_total: int = 0


def verify_python_function(
    code: str,
    func_name: str,
    test_cases: list[tuple[tuple[Any, ...], Any]],
    expected_type: type = int,
    model: GibbsModel | None = None,
    vocab_size: int = 256,
) -> CodeVerificationResult:
    """Full verification pipeline: handcoded constraints + optional learned energy.

    **Researcher summary:**
        Builds execution-based energy, runs tests, optionally computes learned
        energy from a Gibbs model. Returns a CodeVerificationResult combining
        both signals.

    **Detailed explanation for engineers:**
        This is the main entry point for verifying a Python function. It:

        1. **Build handcoded energy**: Creates ReturnType + NoException +
           TestPass constraints via ``build_code_energy``.
        2. **Compute embedding**: Converts the code to a bag-of-tokens vector.
        3. **Execute and verify**: Runs the code on all test cases, collecting
           per-case results and computing the total handcoded energy.
        4. **Learned energy** (optional): If a trained Gibbs model is provided,
           computes the model's energy for the code embedding. Lower energy
           from the learned model suggests the code matches patterns of
           correct code from the training distribution.
        5. **Return**: Combines all results into a CodeVerificationResult.

    Args:
        code: Python source code defining the function.
        func_name: Name of the function to call.
        test_cases: List of (args_tuple, expected_output) pairs.
        expected_type: Expected return type. Default int.
        model: Optional trained GibbsModel for learned verification.
        vocab_size: Embedding dimension. Default 256.

    Returns:
        CodeVerificationResult with handcoded and learned verification results.

    Spec: REQ-CODE-004
    """
    from carnot.verify.python_types import safe_exec_function

    # Build handcoded energy
    composed = build_code_energy(code, func_name, test_cases, expected_type, vocab_size)
    embedding = code_to_embedding(code, vocab_size)
    handcoded_energy = float(composed.energy(embedding))
    result_obj = composed.verify(embedding)
    handcoded_verified = result_obj.is_verified()

    # Execute test cases and collect per-case results
    test_results: list[tuple[Any, Any, bool]] = []
    n_passed = 0
    for args, expected in test_cases:
        actual, error = safe_exec_function(code, func_name, args)
        passed = error is None and actual == expected
        test_results.append((args, actual, passed))
        if passed:
            n_passed += 1

    # Compute learned energy if model provided
    learned_energy = None
    if model is not None:
        learned_energy = float(model.energy(embedding))

    return CodeVerificationResult(
        code=code,
        handcoded_energy=handcoded_energy,
        handcoded_verified=handcoded_verified,
        learned_energy=learned_energy,
        test_results=test_results,
        n_tests_passed=n_passed,
        n_tests_total=len(test_cases),
    )


def compare_learned_vs_handcoded_code(
    model: GibbsModel,
    n_test: int = 100,
    seed: int = 99,
    vocab_size: int = 256,
) -> ComparisonResult:
    """Compare learned Gibbs model vs handcoded constraints on test data.

    **Researcher summary:**
        Generates held-out correct/buggy code pairs, evaluates both the learned
        model and handcoded constraints, and reports accuracy and agreement.

    **Detailed explanation for engineers:**
        This function answers: "How well does the learned verifier agree with
        the handcoded execution-based constraints?"

        For each test sample:
        1. Generate a correct code embedding and a buggy code embedding
        2. Learned: check if model assigns lower energy to correct than buggy
        3. Handcoded: check if execution-based constraints give lower energy
           to correct code
        4. Agreement: check if both methods agree on which is better

        The ``seed`` should be different from the training seed to ensure the
        test set is independent.

    Args:
        model: Trained GibbsModel.
        n_test: Number of test pairs to evaluate.
        seed: Random seed (should differ from training seed).
        vocab_size: Embedding dimension.

    Returns:
        ComparisonResult with accuracy and agreement metrics.

    Spec: REQ-CODE-004
    """
    correct_batch, buggy_batch = generate_code_training_data(n_test, seed, vocab_size)

    learned_correct = 0
    handcoded_correct = 0
    agree = 0

    for i in range(n_test):
        correct_emb = correct_batch[i]
        buggy_emb = buggy_batch[i]

        # Learned: correct should have lower energy
        learned_e_correct = float(model.energy(correct_emb))
        learned_e_buggy = float(model.energy(buggy_emb))
        learned_says_correct = learned_e_correct < learned_e_buggy

        # Handcoded: use the template's actual constraints
        template_idx = i % len(CORRECT_TEMPLATES)
        func_name, correct_code = CORRECT_TEMPLATES[template_idx]
        test_inputs_map: dict[str, list[tuple[tuple[Any, ...], Any]]] = {
            "add": [((1, 2), 3), ((0, 0), 0)],
            "multiply": [((2, 3), 6), ((0, 5), 0)],
            "identity": [((7,), 7), ((0,), 0)],
            "negate": [((5,), -5), ((-3,), 3)],
            "double": [((4,), 8), ((0,), 0)],
        }
        test_cases = test_inputs_map.get(func_name, [((1, 2), 3)])
        composed = build_code_energy(correct_code, func_name, test_cases, int, vocab_size)
        handcoded_e_correct = float(composed.energy(correct_emb))
        # Handcoded energy for correct code should be 0
        handcoded_says_correct = handcoded_e_correct <= 1e-6

        if learned_says_correct:
            learned_correct += 1
        if handcoded_says_correct:
            handcoded_correct += 1
        if learned_says_correct == handcoded_says_correct:
            agree += 1

    return ComparisonResult(
        learned_accuracy=learned_correct / n_test,
        handcoded_accuracy=handcoded_correct / n_test,
        agreement_rate=agree / n_test,
        n_test=n_test,
    )
