"""Tests for EORM verification layer."""

from __future__ import annotations

import jax.random as jrandom
import pytest

from carnot.models.eorm import EORMModel
from carnot.pipeline.eorm_verifier import EORMVerifier


def test_eorm_verifier_initialization():
    """Test that the verifier can be initialized."""
    model = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64)
    verifier = EORMVerifier(model)
    assert verifier.model is model


def test_eorm_verifier_empty_candidates():
    """Test that empty candidates raise ValueError."""
    model = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64)
    verifier = EORMVerifier(model)
    with pytest.raises(ValueError, match="Must provide at least one candidate"):
        verifier.verify_and_rerank("What is 2+2?", [])


def test_eorm_verifier_rerank():
    """Test that the verifier correctly ranks candidates."""
    model = EORMModel(
        embed_dim=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=32,
        vocab_size=64,
        key=jrandom.PRNGKey(42),
    )
    verifier = EORMVerifier(model)
    
    question = "Solve 3x + 5 = 14"
    candidates = [
        "Subtract 5 to get 3x = 9, then x = 3.",
        "Subtract 5 to get 3x = 9, then divide by 3 to get x = 3.",
        "Divide by 3 first to get x + 5/3 = 14/3.",
    ]
    
    result = verifier.verify_and_rerank(question, candidates)
    
    assert "best_candidate" in result
    assert "best_energy" in result
    assert "ranked_candidates" in result
    assert "energies" in result
    
    assert len(result["ranked_candidates"]) == 3
    assert len(result["energies"]) == 3
    
    # Energies should be sorted in ascending order
    energies = result["energies"]
    assert all(energies[i] <= energies[i+1] for i in range(len(energies)-1))
    
    assert result["best_candidate"] == result["ranked_candidates"][0]
    assert result["best_energy"] == energies[0]
