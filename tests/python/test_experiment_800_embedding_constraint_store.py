"""Tests for Exp 800: EmbeddingConstraintStore SPO + Orthogonality Regularization.

Verifies that EmbeddingConstraintStore encodes constraints as SPO tuples,
applies orthogonality regularization, retrieves by cosine similarity, and
correctly bootstraps from legacy CaseMemory pattern dicts.

Spec: REQ-LEARN-057, REQ-LEARN-058, REQ-LEARN-059, SCENARIO-LEARN-098
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent.parent))
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.embedding_constraint_store import (  # noqa: E402
    ConstraintSPOTuple,
    EmbeddingConstraintStore,
    _ci_hash_embedding,
    _cosine_similarity,
    _dot,
    _l2norm,
    _normalize,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _make_store() -> EmbeddingConstraintStore:
    """Return a fresh store in ci_hash mode (no heavy dependency required)."""
    store = EmbeddingConstraintStore.__new__(EmbeddingConstraintStore)
    store.model_name = "all-MiniLM-L6-v2"
    store._store = []
    store._encoder = None
    store.embedding_mode = "ci_hash"
    return store


def _make_spo(vtype: str) -> ConstraintSPOTuple:
    """Return a minimal SPO tuple with no embedding set."""
    return ConstraintSPOTuple(
        subject="s",
        predicate="p",
        object="o",
        embedding=None,
        source_violation_type=vtype,
    )


# ---------------------------------------------------------------------------
# REQ-LEARN-057: ConstraintSPOTuple dataclass fields
# ---------------------------------------------------------------------------


class TestConstraintSPOTuple:
    """Spec: REQ-LEARN-057"""

    def test_fields_present(self) -> None:
        """ConstraintSPOTuple must have all six documented fields."""
        spo = ConstraintSPOTuple(
            subject="arithmetic_carry",
            predicate="violates",
            object="carry_propagation",
            embedding=None,
            source_violation_type="carry",
        )
        assert spo.subject == "arithmetic_carry"
        assert spo.predicate == "violates"
        assert spo.object == "carry_propagation"
        assert spo.embedding is None
        assert spo.source_violation_type == "carry"
        assert isinstance(spo.timestamp, str) and len(spo.timestamp) > 0

    def test_embedding_field_accepts_list(self) -> None:
        """embedding field must accept a list[float] (populated after store())."""
        spo = ConstraintSPOTuple(
            subject="x",
            predicate="y",
            object="z",
            embedding=[0.1, -0.2, 0.3],
            source_violation_type="unit",
        )
        assert spo.embedding == [0.1, -0.2, 0.3]


# ---------------------------------------------------------------------------
# REQ-LEARN-058: Orthogonality regularization
# ---------------------------------------------------------------------------


class TestOrthogonalize:
    """Spec: REQ-LEARN-058"""

    def test_orthogonalize_reduces_dot_product(self) -> None:
        """After storing one embedding, the next should be near-orthogonal to it.

        Why this matters: if two stored embeddings are NOT orthogonal, queries
        that sit between them will return ambiguous top-1 results, lowering AUC.
        Orthogonalization forces them into distinct subspaces.
        """
        store = _make_store()

        # Store first constraint — gets L2-normalized but not projected (store is empty).
        spo1 = ConstraintSPOTuple(
            subject="arithmetic_carry",
            predicate="violates",
            object="carry_propagation",
            embedding=None,
            source_violation_type="carry",
        )
        store.store(spo1)
        assert spo1.embedding is not None
        e1 = spo1.embedding

        # Store second constraint — should be projected out of e1's direction.
        spo2 = ConstraintSPOTuple(
            subject="numeric_sign",
            predicate="violates",
            object="sign_preservation",
            embedding=None,
            source_violation_type="sign",
        )
        store.store(spo2)
        assert spo2.embedding is not None
        e2 = spo2.embedding

        # Dot product between stored embeddings should be < 0.05 (near-orthogonal).
        dot = _dot(e1, e2)
        assert abs(dot) < 0.05, f"Stored embeddings not near-orthogonal: dot={dot:.4f}"

    def test_orthogonalize_result_is_unit_norm(self) -> None:
        """The orthogonalized embedding must be L2-normalized (norm ≈ 1.0)."""
        store = _make_store()
        spo = ConstraintSPOTuple(
            subject="unit_label",
            predicate="violates",
            object="unit_consistency",
            embedding=None,
            source_violation_type="unit",
        )
        store.store(spo)
        assert spo.embedding is not None
        norm = _l2norm(spo.embedding)
        assert abs(norm - 1.0) < 1e-6, f"Stored embedding norm = {norm:.6f}, expected 1.0"

    def test_empty_store_orthogonalize_is_identity(self) -> None:
        """With no prior entries, _orthogonalize returns a normalized version of the input."""
        store = _make_store()
        v = [1.0, 2.0, 3.0]
        result = store._orthogonalize(v)
        # Must be unit norm
        assert abs(_l2norm(result) - 1.0) < 1e-6
        # Must point in same direction as v
        original_norm = _l2norm(v)
        expected = [x / original_norm for x in v]
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-6


# ---------------------------------------------------------------------------
# REQ-LEARN-059: Cosine-similarity retrieval
# ---------------------------------------------------------------------------


class TestRetrieve:
    """Spec: REQ-LEARN-059"""

    def test_retrieve_returns_top_k(self) -> None:
        """retrieve() must return exactly top_k results when store has >= top_k entries."""
        store = _make_store()
        for vtype in ["carry", "sign", "unit", "comparison", "causal"]:
            spo = ConstraintSPOTuple(
                subject=vtype,
                predicate="violates",
                object=f"{vtype}_rule",
                embedding=None,
                source_violation_type=vtype,
            )
            store.store(spo)

        results = store.retrieve("test query about carry", top_k=3)
        assert len(results) == 3

    def test_retrieve_returns_constraintspo_instances(self) -> None:
        """retrieve() must return ConstraintSPOTuple objects."""
        store = _make_store()
        spo = ConstraintSPOTuple(
            subject="arithmetic_carry",
            predicate="violates",
            object="carry_propagation",
            embedding=None,
            source_violation_type="carry",
        )
        store.store(spo)
        results = store.retrieve("carry error query", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], ConstraintSPOTuple)

    def test_retrieve_empty_store_returns_empty(self) -> None:
        """retrieve() on an empty store must return an empty list."""
        store = _make_store()
        results = store.retrieve("any query", top_k=3)
        assert results == []

    def test_retrieval_auc_empty(self) -> None:
        """retrieval_auc() with empty queries or empty store returns 0.0."""
        store = _make_store()
        assert store.retrieval_auc([], []) == 0.0
        # Store has entries but queries is empty
        spo = ConstraintSPOTuple(
            subject="s",
            predicate="p",
            object="o",
            embedding=None,
            source_violation_type="carry",
        )
        store.store(spo)
        assert store.retrieval_auc([], []) == 0.0

    def test_retrieval_auc_perfect_on_identical_query(self) -> None:
        """When there is only one stored constraint, all queries must retrieve it (AUC=1.0)."""
        store = _make_store()
        spo = ConstraintSPOTuple(
            subject="arithmetic_carry",
            predicate="violates",
            object="carry_propagation",
            embedding=None,
            source_violation_type="carry",
        )
        store.store(spo)
        # With only one entry in store, every query retrieves it → AUC = 1.0
        queries = ["carry error", "carry propagation missed", "missing carry bit"]
        labels = ["carry", "carry", "carry"]
        auc = store.retrieval_auc(queries, labels)
        assert auc == 1.0


# ---------------------------------------------------------------------------
# REQ-LEARN-057: from_casememory_patterns produces 5 SPO tuples
# ---------------------------------------------------------------------------


class TestFromCasememoryPatterns:
    """Spec: REQ-LEARN-057"""

    def test_produces_five_spo_tuples(self) -> None:
        """from_casememory_patterns() must produce exactly 5 stored tuples for all 5 keys."""
        store = _make_store()
        store.from_casememory_patterns(
            {"carry": 4, "sign": 4, "unit": 4, "comparison": 4, "causal": 4}
        )
        assert len(store._store) == 5

    def test_violation_types_present(self) -> None:
        """All 5 source_violation_type values must appear in the store."""
        store = _make_store()
        store.from_casememory_patterns(
            {"carry": 1, "sign": 1, "unit": 1, "comparison": 1, "causal": 1}
        )
        vtypes = {spo.source_violation_type for spo in store._store}
        assert vtypes == {"carry", "sign", "unit", "comparison", "causal"}

    def test_spo_fields_populated_correctly(self) -> None:
        """Each stored SPO must have non-empty subject/predicate/object fields."""
        store = _make_store()
        store.from_casememory_patterns({"carry": 1})
        assert len(store._store) == 1
        spo = store._store[0]
        assert spo.subject == "arithmetic_carry"
        assert spo.predicate == "violates"
        assert spo.object == "carry_propagation"

    def test_unknown_keys_ignored(self) -> None:
        """Keys not in the known SPO map must be silently ignored."""
        store = _make_store()
        store.from_casememory_patterns({"carry": 1, "unknown_error": 99})
        assert len(store._store) == 1
        assert store._store[0].source_violation_type == "carry"

    def test_empty_patterns_produces_no_entries(self) -> None:
        """An empty patterns dict must result in an empty store."""
        store = _make_store()
        store.from_casememory_patterns({})
        assert len(store._store) == 0

    def test_embeddings_set_after_from_casememory(self) -> None:
        """All stored SPO tuples must have embeddings (not None) after from_casememory_patterns."""
        store = _make_store()
        store.from_casememory_patterns({"carry": 1, "sign": 1})
        for spo in store._store:
            assert spo.embedding is not None
            assert isinstance(spo.embedding, list)
            assert len(spo.embedding) > 0


# ---------------------------------------------------------------------------
# CI hash mode determinism (REQ-LEARN-057 fallback)
# ---------------------------------------------------------------------------


class TestCIHashMode:
    """Spec: REQ-LEARN-057-3 (ci_hash fallback determinism)"""

    def test_ci_hash_deterministic(self) -> None:
        """Same text must produce the same 384-dim float vector every time."""
        v1 = _ci_hash_embedding("carry_error test")
        v2 = _ci_hash_embedding("carry_error test")
        assert v1 == v2

    def test_ci_hash_different_text_different_vector(self) -> None:
        """Different texts must produce different vectors."""
        v1 = _ci_hash_embedding("carry_error")
        v2 = _ci_hash_embedding("sign_error")
        assert v1 != v2

    def test_ci_hash_produces_384_dims(self) -> None:
        """ci_hash must produce exactly 384-dimensional vectors."""
        v = _ci_hash_embedding("unit error")
        assert len(v) == 384

    def test_ci_hash_values_in_range(self) -> None:
        """ci_hash values must be in [-0.5, 0.5] (uniform distribution)."""
        v = _ci_hash_embedding("comparison error test")
        assert all(-0.5 <= x <= 0.5 for x in v)

    def test_store_uses_ci_hash_when_encoder_none(self) -> None:
        """When _encoder is None, embedding_mode must be 'ci_hash'."""
        store = _make_store()
        assert store.embedding_mode == "ci_hash"
        assert store._encoder is None

    def test_store_ci_hash_encodes_deterministically(self) -> None:
        """_encode() in ci_hash mode must return same vector for same text."""
        store = _make_store()
        v1 = store._encode("test carry error")
        v2 = store._encode("test carry error")
        assert v1 == v2
