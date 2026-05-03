"""Tests for the Exp 1168 SC-Energy verifier.

Spec: REQ-VERIFY-1168, SCENARIO-VERIFY-1168
"""

from __future__ import annotations

import numpy as np

from carnot.verify import sc_energy_verifier as scv
from carnot.verify.sc_energy_verifier import SCEnergyVerifier


def test_encode_returns_one_projected_embedding_per_statement() -> None:
    """REQ-VERIFY-1168-2: encode returns a 2-D array with one row per statement."""
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=16)

    embeddings = verifier.encode(["Alice has 2 apples.", "Alice gets 3 more apples."])

    assert embeddings.shape == (2, 16)
    assert embeddings.dtype == np.float32
    assert np.all(np.isfinite(embeddings))


def test_encode_handles_empty_statement_sets() -> None:
    """REQ-VERIFY-1168-2: empty statement sets keep the same embedding rank."""
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=8)

    embeddings = verifier.encode([])

    assert embeddings.shape == (0, 8)
    assert embeddings.dtype == np.float32


def test_energy_is_higher_for_less_compatible_response() -> None:
    """REQ-VERIFY-1168-3: higher energy means lower context/response compatibility."""
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=32)
    context = "Alice has 2 apples. Bob gives Alice 3 more apples."

    coherent = verifier.energy("Alice now has 5 apples.", context)
    incoherent = verifier.energy("The train travels 90 miles.", context)

    assert coherent < incoherent


def test_train_returns_self_and_widens_margin_gap() -> None:
    """REQ-VERIFY-1168-4: train optimizes the incoherent-minus-coherent margin."""
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=32)
    coherent_pairs = [
        ("Alice now has 5 apples.", "Alice has 2 apples. Alice gets 3 apples."),
        ("The shelf has 7 books.", "The shelf starts with 4 books. 3 books arrive."),
    ]
    incoherent_pairs = [
        ("The train travels 90 miles.", "Alice has 2 apples. Alice gets 3 apples."),
        ("The soup needs 2 cups of water.", "The shelf starts with 4 books. 3 books arrive."),
    ]

    before = verifier.contrastive_loss(coherent_pairs, incoherent_pairs)
    returned = verifier.train(coherent_pairs, incoherent_pairs, n_epochs=8)
    after = verifier.contrastive_loss(coherent_pairs, incoherent_pairs)

    assert returned is verifier
    assert after < before


def test_statement_set_training_inputs_are_accepted() -> None:
    """REQ-VERIFY-1168-1: train accepts statement-set style contrastive pairs."""
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=24)
    coherent_sets = [["A box has 4 pens.", "A second box has 6 pens.", "There are 10 pens total."]]
    incoherent_sets = [["A box has 4 pens.", "A second box has 6 pens.", "Clouds cover the sky."]]

    verifier.train(coherent_sets, incoherent_sets, n_epochs=2)

    assert verifier.energy("There are 10 pens total.", "A box has 4 pens.") < verifier.energy(
        "Clouds cover the sky.", "A box has 4 pens."
    )


def test_constraint_term_compatibility_helpers() -> None:
    """REQ-VERIFY-1168-1: SCEnergyVerifier exposes the ConstraintTerm-adjacent helpers."""
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=64)

    assert verifier.name == "SCEnergyVerifier"
    assert verifier.satisfaction_threshold == 0.5
    assert verifier.is_satisfied("Alice has apples.", "Alice has fruit.") is True
    assert verifier.is_satisfied("A rocket launched.", "Alice has fruit.") is False
    assert np.array_equal(verifier.grad_energy(np.ones(3)), np.zeros(3))


def test_empty_training_and_loss_history_paths() -> None:
    """REQ-VERIFY-1168-4: empty contrastive data is handled deterministically."""
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=8)

    assert verifier.contrastive_loss([], []) == 0.0
    assert verifier.train([], [], n_epochs=4) is verifier
    assert verifier.loss_history == [0.0]
    assert verifier.energy("", "") == 1.0


def test_pair_input_shapes_cover_dict_pair_and_fallback_forms() -> None:
    """REQ-VERIFY-1168-1: pair coercion supports artifact-style row formats."""
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=16)

    dict_pair = {"response": "Alice has 5 apples.", "context": "Alice has apples."}
    statement_dict = {"statements": ["A box has 4 pens.", "There are 4 pens."]}
    explicit_pair = scv._Pair("The shelf has books.", "The shelf is wooden.")
    one_statement = ["Only one statement."]

    assert verifier.contrastive_loss([dict_pair], [statement_dict]) >= 0.0
    assert verifier.contrastive_loss([explicit_pair], [one_statement]) >= 0.0
    assert verifier.contrastive_loss(["plain response"], [("response", "context")]) >= 0.0


def test_pair_feature_edge_shapes() -> None:
    """REQ-VERIFY-1168-3: set-energy features cover empty, unary, and pairwise cases."""
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=12)

    assert verifier.energy("", "Context only.") < 1.0
    assert verifier.energy("One statement only.") < 1.0
    assert verifier.energy("First statement. Second statement.") < 1.0
    assert scv._pair_feature(
        np.zeros((0, 12), dtype=np.float32), np.zeros((0, 12), dtype=np.float32)
    ).shape == (0,)


def test_projection_branch_and_backend_fallback(monkeypatch) -> None:
    """REQ-VERIFY-1168-2: non-hidden-size encoder outputs are projected."""

    class TinyBackend:
        name = "tiny"

        def encode_cls(self, statements: list[str]) -> np.ndarray:
            return np.ones((len(statements), 3), dtype=np.float32)

    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=5)
    verifier._backend = TinyBackend()

    projected = verifier.encode(["one", "two"])

    assert projected.shape == (2, 5)
    assert verifier._projection is not None

    class RaisingBackend:
        def __init__(self, model_name: str) -> None:
            raise RuntimeError(model_name)

    monkeypatch.setattr(scv, "_TransformersCLSBackend", RaisingBackend)
    backend = scv._load_backend("roberta-base", 7)
    assert backend.name == "deterministic_cls"
    assert backend.encode_cls([]).shape == (0, 7)


def test_arithmetic_markers_and_safe_eval_paths() -> None:
    """REQ-VERIFY-1168-2: deterministic CLS embeddings include arithmetic consistency markers."""
    assert scv._equation_marker("2 + 2 = 4") == "__arith_valid__"
    assert scv._equation_marker("2 + 2 = 5") == "__arith_invalid__"
    assert scv._equation_marker("2 + = 4") == "__arith_valid__"
    assert scv._equation_marker("no equation here") == ""
    assert scv._safe_eval_arithmetic("-(2 + 3) * 4 / 2") == -10.0
    assert scv._safe_eval_arithmetic("5 - 3") == 2.0
    try:
        scv._safe_eval_arithmetic("1 / 0")
    except ValueError as exc:
        assert "division by zero" in str(exc)
    try:
        scv._safe_eval_arithmetic("abs(1)")
    except ValueError as exc:
        assert "unsupported" in str(exc)

    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=10)
    assert verifier.encode(["2 + 2 = 4"]).shape == (1, 10)
    assert np.array_equal(verifier.encode(["!!!"]), np.zeros((1, 10), dtype=np.float32))
