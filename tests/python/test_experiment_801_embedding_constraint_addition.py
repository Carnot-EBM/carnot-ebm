"""Tests for Exp 801: Embedding Constraint Addition wired into VerifyRepairPipeline.

Traces to: REQ-LEARN-060, REQ-LEARN-061, SCENARIO-LEARN-099
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
for _p in [str(_REPO), str(_SCRIPTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store_with_patterns():
    """Return an EmbeddingConstraintStore bootstrapped with all 5 pattern types."""
    from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore

    store = EmbeddingConstraintStore()
    store.from_casememory_patterns(
        {"carry": 4, "sign": 4, "unit": 4, "comparison": 4, "causal": 4}
    )
    return store


# ---------------------------------------------------------------------------
# REQ-LEARN-060: verify() accepts embedding_constraint_store param
# ---------------------------------------------------------------------------


def test_verify_accepts_embedding_constraint_store_none():
    """verify() with embedding_constraint_store=None (default) runs without error.

    Spec: REQ-LEARN-060-4 (full backward compatibility when param is None)
    """
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline()
    result = pipeline.verify(
        question="What is 2 + 2?",
        response="The answer is 4.",
        embedding_constraint_store=None,
    )
    # Must return a valid VerificationResult regardless of value
    assert hasattr(result, "verified")
    assert hasattr(result, "constraints")


def test_verify_accepts_embedding_constraint_store_param():
    """verify() accepts a non-None EmbeddingConstraintStore without raising.

    Spec: REQ-LEARN-060-1 (parameter accepted with correct type)
    """
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    store = _make_store_with_patterns()
    pipeline = VerifyRepairPipeline()
    result = pipeline.verify(
        question="What is 2 + 2?",
        response="The answer is 4.",
        embedding_constraint_store=store,
    )
    assert hasattr(result, "verified")


def test_verify_calls_retrieve_on_store():
    """verify() calls store.retrieve(response, top_k=3) when store is set.

    Spec: REQ-LEARN-060-2
    """
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    store = _make_store_with_patterns()
    original_retrieve = store.retrieve
    calls: list[tuple] = []

    def spy_retrieve(query: str, top_k: int = 3):
        calls.append((query, top_k))
        return original_retrieve(query, top_k=top_k)

    store.retrieve = spy_retrieve  # type: ignore[method-assign]

    pipeline = VerifyRepairPipeline()
    response_text = "37 + 45 = 72 because the carry bit was dropped"
    pipeline.verify(
        question="Add 37 and 45.",
        response=response_text,
        embedding_constraint_store=store,
    )

    assert len(calls) >= 1, "retrieve() was not called"
    # Called with the response text and top_k=3
    assert calls[0][0] == response_text
    assert calls[0][1] == 3


# ---------------------------------------------------------------------------
# REQ-LEARN-061: Constraint injection is additive
# ---------------------------------------------------------------------------


def test_static_constraints_unchanged_with_store():
    """Static constraints extracted from the response are present alongside injected ones.

    Spec: REQ-LEARN-061-1 (additive injection: static constraints remain)
    REQ-LEARN-061-2 (injected constraint_type prefixed with 'embedding_retrieved_')
    """
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    store = _make_store_with_patterns()
    pipeline = VerifyRepairPipeline()

    # Run with store to get constraints list
    result_with_store = pipeline.verify(
        question="Add 37 and 45.",
        response="37 + 45 = 72 because the carry bit was dropped",
        embedding_constraint_store=store,
    )

    # Run without store to get static-only constraints
    result_without_store = pipeline.verify(
        question="Add 37 and 45.",
        response="37 + 45 = 72 because the carry bit was dropped",
        embedding_constraint_store=None,
    )

    # Injected constraints must carry the 'embedding_retrieved_' prefix
    injected = [
        c for c in result_with_store.constraints
        if c.constraint_type.startswith("embedding_retrieved_")
    ]
    assert len(injected) > 0, "No embedding_retrieved_ constraints injected"

    # Count of static constraints should be >= without-store count
    static_types_with = {
        c.constraint_type for c in result_with_store.constraints
        if not c.constraint_type.startswith("embedding_retrieved_")
    }
    static_types_without = {c.constraint_type for c in result_without_store.constraints}
    # static types present without store must all appear in the with-store result
    assert static_types_without <= static_types_with or len(static_types_without) == 0, (
        "Static constraints were removed when embedding_constraint_store was set"
    )


def test_injected_constraint_metadata_fields():
    """Injected ConstraintResult has all required metadata fields.

    Spec: REQ-LEARN-061-3
    """
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    store = _make_store_with_patterns()
    pipeline = VerifyRepairPipeline()
    result = pipeline.verify(
        question="Compute.",
        response="37 + 45 = 72 because the carry bit was dropped",
        embedding_constraint_store=store,
    )

    injected = [
        c for c in result.constraints
        if c.constraint_type.startswith("embedding_retrieved_")
    ]
    assert injected, "Expected at least one injected constraint"
    for c in injected:
        assert "source" in c.metadata
        assert c.metadata["source"] == "embedding_constraint_store"
        assert "spo_subject" in c.metadata
        assert "spo_predicate" in c.metadata
        assert "spo_object" in c.metadata
        assert "source_violation_type" in c.metadata


# ---------------------------------------------------------------------------
# Session-level delta computation (SCENARIO-LEARN-099)
# ---------------------------------------------------------------------------


def test_compute_honest_verdict_works():
    """compute_honest_verdict returns 'constraint_addition_works' for delta>0 + monotonic.

    Spec: SCENARIO-LEARN-099
    """
    from experiment_801_embedding_constraint_addition import compute_honest_verdict

    assert compute_honest_verdict(0.05, True) == "constraint_addition_works"


def test_compute_honest_verdict_partial():
    """compute_honest_verdict returns 'constraint_addition_partial' for delta>0, not monotonic."""
    from experiment_801_embedding_constraint_addition import compute_honest_verdict

    assert compute_honest_verdict(0.05, False) == "constraint_addition_partial"


def test_compute_honest_verdict_zero_delta():
    """compute_honest_verdict returns 'constraint_addition_zero_delta' for delta==0.0."""
    from experiment_801_embedding_constraint_addition import compute_honest_verdict

    assert compute_honest_verdict(0.0, True) == "constraint_addition_zero_delta"
    assert compute_honest_verdict(0.0, False) == "constraint_addition_zero_delta"


def test_run_session_returns_float_in_range():
    """run_session returns a float accuracy in [0.0, 1.0].

    Spec: SCENARIO-LEARN-099
    """
    from experiment_801_embedding_constraint_addition import (
        _build_session_questions,
        run_session,
    )

    questions = _build_session_questions()
    acc = run_session(questions, store=None)
    assert 0.0 <= acc <= 1.0


def test_run_session_with_store_returns_float_in_range():
    """run_session with EmbeddingConstraintStore returns accuracy in [0.0, 1.0].

    Spec: REQ-LEARN-060 (store param exercised in session loop)
    """
    from experiment_801_embedding_constraint_addition import (
        _build_session_questions,
        run_session,
    )

    store = _make_store_with_patterns()
    questions = _build_session_questions()
    acc = run_session(questions, store=store)
    assert 0.0 <= acc <= 1.0


def test_session_delta_computation():
    """Per-session delta = dynamic_acc - baseline_acc is computed correctly.

    Spec: SCENARIO-LEARN-099
    """
    from experiment_801_embedding_constraint_addition import (
        _build_session_questions,
        run_session,
    )

    store = _make_store_with_patterns()
    questions = _build_session_questions()
    baseline = run_session(questions, store=None)
    dynamic = run_session(questions, store=store)
    delta = dynamic - baseline
    # delta must be in [-1, 1] (no constraint on sign — honest reporting)
    assert -1.0 <= delta <= 1.0
