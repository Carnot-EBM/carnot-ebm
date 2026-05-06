"""Tests for Exp 800: EmbeddingConstraintStore SPO + L2-normalized retrieval.

Verifies that EmbeddingConstraintStore encodes constraints as SPO tuples,
preserves semantic embedding direction, retrieves by cosine similarity, and
correctly bootstraps from legacy CaseMemory pattern dicts.

Spec: REQ-LEARN-057, REQ-LEARN-058, REQ-LEARN-059, REQ-VERIFY-150,
SCENARIO-LEARN-098
"""

from __future__ import annotations

import math
import os
import sys
import types
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
# REQ-LEARN-058 / REQ-VERIFY-150: L2 normalization without orthogonalizing store writes
# ---------------------------------------------------------------------------


class TestStoreNormalization:
    """Spec: REQ-LEARN-058, REQ-VERIFY-150"""

    def test_store_preserves_semantic_direction_after_l2_normalization(self) -> None:
        """Stored embeddings stay aligned with their original SPO encoding.

        Exp 847 superseded the old Exp 800 orthogonalization path because
        Gram-Schmidt projection made later queries nearly orthogonal to matching
        stored constraints.  The write path should normalize, not deflect.
        """
        store = _make_store()
        spo1 = ConstraintSPOTuple(
            subject="arithmetic_carry",
            predicate="violates",
            object="carry_propagation",
            embedding=None,
            source_violation_type="carry",
        )
        store.store(spo1)
        assert spo1.embedding is not None
        expected = _normalize(
            _ci_hash_embedding("(arithmetic_carry) (violates) (carry_propagation)")
        )

        assert _cosine_similarity(spo1.embedding, expected) > 0.999999

    def test_store_result_is_unit_norm(self) -> None:
        """The stored embedding must be L2-normalized (norm approximately 1.0)."""
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

    def test_empty_store_orthogonalize_helper_is_identity(self) -> None:
        """The historical diagnostic helper still normalizes an empty-store input."""
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

    def test_constructor_ci_hash_mode_does_not_load_encoder(self) -> None:
        """REQ-LEARN-057-5: explicit ci_hash mode avoids MiniLM weight loading."""
        store = EmbeddingConstraintStore(embedding_mode="ci_hash")
        assert store.embedding_mode == "ci_hash"
        assert store._encoder is None

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

    def test_invalid_embedding_mode_raises(self) -> None:
        """REQ-LEARN-057-5: unsupported embedding modes fail clearly."""
        with pytest.raises(ValueError, match="embedding_mode"):
            EmbeddingConstraintStore(embedding_mode="invalid")

    def test_sentence_transformer_mode_uses_mocked_encoder(self, monkeypatch) -> None:
        """REQ-LEARN-057-2: sentence_transformer mode calls the configured encoder."""

        class FakeEncoder:
            def encode(self, text: str) -> list[float]:
                return [1.0, 2.0, 3.0]

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = lambda model_name: FakeEncoder()
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        store = EmbeddingConstraintStore(embedding_mode="sentence_transformer")
        assert store.embedding_mode == "sentence_transformer"
        assert store._encode("anything") == [1.0, 2.0, 3.0]

    def test_auto_mode_falls_back_to_ci_hash_when_encoder_missing(self, monkeypatch) -> None:
        """REQ-LEARN-057-3: missing sentence_transformers falls back to ci_hash."""
        original_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sentence_transformers":
                raise ImportError("missing optional dependency")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", guarded_import)
        store = EmbeddingConstraintStore(embedding_mode="auto")
        assert store.embedding_mode == "ci_hash"

    def test_zero_vector_helpers_are_safe(self) -> None:
        """REQ-VERIFY-150: zero vectors do not produce NaN normalization or cosine."""
        assert _normalize([0.0, 0.0]) == [0.0, 0.0]
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_orthogonalize_helper_branches_remain_diagnostic_only(self) -> None:
        """REQ-LEARN-058-3: historical helper handles empty and degenerate entries."""
        store = _make_store()
        store._store = [
            ConstraintSPOTuple(
                subject="empty",
                predicate="violates",
                object="none",
                embedding=None,
                source_violation_type="empty",
            ),
            ConstraintSPOTuple(
                subject="zero",
                predicate="violates",
                object="zero",
                embedding=[0.0, 0.0],
                source_violation_type="zero",
            ),
            ConstraintSPOTuple(
                subject="axis",
                predicate="violates",
                object="x",
                embedding=[1.0, 0.0],
                source_violation_type="axis",
            ),
        ]

        result = store._orthogonalize([1.0, 1.0])
        assert abs(_l2norm(result) - 1.0) < 1e-6
        assert abs(_dot(result, [1.0, 0.0])) < 1e-6
