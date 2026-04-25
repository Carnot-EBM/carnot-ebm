"""Tests for Exp 865 — Constraint memory bank compression K=32.

Spec: REQ-STORE-020, SCENARIO-STORE-030

Coverage targets:
    - MemoryBankCompressor.compress() output shape and label length
    - MemoryBankCompressor.compression_ratio()
    - MemoryBankCompressor.compress() no-op path (N <= K)
    - EmbeddingConstraintStore.compress() in-place mutation
    - EmbeddingConstraintStore.add_constraint() normalisation
    - EmbeddingConstraintStore.retrieve() top-k ranking
    - EmbeddingConstraintStore.embeddings property (empty store)
    - EmbeddingConstraintStore.__len__()
    - Full AUROC benchmark (before < 1.0, after > 0.75)
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from carnot.stores.embedding_constraint_store import EmbeddingConstraintStore
from carnot.stores.memory_bank_compressor import MemoryBankCompressor


# ---------------------------------------------------------------------------
# MemoryBankCompressor tests
# ---------------------------------------------------------------------------

class TestMemoryBankCompressor:
    """REQ-STORE-020: MemoryBankCompressor compress() contract."""

    def test_compress_output_shape(self) -> None:
        """compress() returns (K, D) centroids and K labels when N > K."""
        rng = np.random.default_rng(0)
        N, D, K = 200, 16, 8
        emb = rng.standard_normal((N, D)).astype(np.float32)
        labels = [bool(i % 2) for i in range(N)]

        compressor = MemoryBankCompressor(k=K)
        centroids, new_labels = compressor.compress(emb, labels)

        assert centroids.shape == (K, D), f"Expected ({K}, {D}), got {centroids.shape}"
        assert len(new_labels) == K

    def test_compress_labels_are_from_input(self) -> None:
        """Majority-vote labels must be a subset of the input label values."""
        rng = np.random.default_rng(1)
        N, D, K = 100, 8, 4
        emb = rng.standard_normal((N, D)).astype(np.float32)
        labels = [i % 3 for i in range(N)]  # multi-class labels

        compressor = MemoryBankCompressor(k=K)
        _, new_labels = compressor.compress(emb, labels)

        valid = set(labels)
        for lbl in new_labels:
            assert lbl in valid, f"Label {lbl!r} not in original label set {valid}"

    def test_compress_noop_when_n_le_k(self) -> None:
        """No compression when N <= K: original arrays returned unchanged."""
        # REQ-STORE-020: N <= k is a no-op
        rng = np.random.default_rng(2)
        N, D, K = 10, 8, 32  # N < K
        emb = rng.standard_normal((N, D)).astype(np.float32)
        labels = [bool(i % 2) for i in range(N)]

        compressor = MemoryBankCompressor(k=K)
        centroids, new_labels = compressor.compress(emb, labels)

        assert centroids.shape == (N, D)
        assert len(new_labels) == N
        assert new_labels == labels

    def test_compress_noop_when_n_equals_k(self) -> None:
        """Exact equality N == K is also a no-op."""
        rng = np.random.default_rng(3)
        K = 5
        emb = rng.standard_normal((K, 4)).astype(np.float32)
        labels = [True] * K

        compressor = MemoryBankCompressor(k=K)
        centroids, new_labels = compressor.compress(emb, labels)

        assert centroids.shape == (K, 4)

    def test_compression_ratio(self) -> None:
        """compression_ratio() returns original_n / k."""
        compressor = MemoryBankCompressor(k=32)
        assert compressor.compression_ratio(1000) == pytest.approx(31.25)
        assert compressor.compression_ratio(32) == pytest.approx(1.0)
        assert compressor.compression_ratio(64) == pytest.approx(2.0)

    def test_compression_ratio_custom_k(self) -> None:
        """compression_ratio() respects custom k values."""
        compressor = MemoryBankCompressor(k=10)
        assert compressor.compression_ratio(100) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# EmbeddingConstraintStore tests
# ---------------------------------------------------------------------------

class TestEmbeddingConstraintStore:
    """REQ-STORE-010, REQ-STORE-011, REQ-STORE-020."""

    def test_add_constraint_normalises(self) -> None:
        """Stored embeddings are unit-normalised regardless of input scale."""
        store = EmbeddingConstraintStore()
        raw = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5
        store.add_constraint(raw, True)

        stored = store.embeddings
        norm = np.linalg.norm(stored[0])
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_len(self) -> None:
        """__len__ returns number of stored constraints."""
        store = EmbeddingConstraintStore()
        assert len(store) == 0
        store.add_constraint(np.array([1.0, 0.0]), True)
        assert len(store) == 1
        store.add_constraint(np.array([0.0, 1.0]), False)
        assert len(store) == 2

    def test_embeddings_property_empty(self) -> None:
        """embeddings property returns (0, 0) array on empty store."""
        store = EmbeddingConstraintStore()
        emb = store.embeddings
        assert emb.shape == (0, 0)

    def test_embeddings_property_nonempty(self) -> None:
        """embeddings property stacks stored vectors into (N, D) array."""
        store = EmbeddingConstraintStore()
        store.add_constraint(np.array([1.0, 0.0, 0.0]), True)
        store.add_constraint(np.array([0.0, 1.0, 0.0]), False)
        emb = store.embeddings
        assert emb.shape == (2, 3)

    def test_retrieve_empty_store(self) -> None:
        """retrieve() returns empty list from an empty store."""
        store = EmbeddingConstraintStore()
        result = store.retrieve(np.array([1.0, 0.0]))
        assert result == []

    def test_retrieve_returns_top_k(self) -> None:
        """retrieve() returns at most top_k results."""
        store = EmbeddingConstraintStore()
        rng = np.random.default_rng(10)
        for i in range(20):
            store.add_constraint(rng.standard_normal(8).astype(np.float32), i % 2 == 0)
        query = rng.standard_normal(8).astype(np.float32)
        result = store.retrieve(query, top_k=3)
        assert len(result) == 3

    def test_retrieve_ranking_correct(self) -> None:
        """retrieve() ranks the most similar embedding first."""
        store = EmbeddingConstraintStore()
        # Two orthogonal vectors; query is identical to the first.
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        store.add_constraint(v1, True)
        store.add_constraint(v2, False)

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = store.retrieve(query, top_k=2)

        # First result should be the True label (v1 is closest to query).
        assert result[0][1] is True
        assert result[0][0] > result[1][0]  # scores are descending

    def test_compress_in_place(self) -> None:
        """compress() replaces store contents with K centroids in-place."""
        rng = np.random.default_rng(20)
        store = EmbeddingConstraintStore()
        for i in range(200):
            store.add_constraint(rng.standard_normal(16).astype(np.float32), i % 2 == 0)

        assert len(store) == 200
        store.compress(k=8)
        assert len(store) == 8

    def test_compress_noop_when_store_small(self) -> None:
        """compress() does nothing when store has fewer constraints than k."""
        store = EmbeddingConstraintStore()
        store.add_constraint(np.array([1.0, 0.0]), True)
        store.add_constraint(np.array([0.0, 1.0]), False)

        store.compress(k=32)  # 2 <= 32 → no-op
        assert len(store) == 2

    def test_dim_is_set_after_first_add(self) -> None:
        """dim attribute is set after the first add_constraint call."""
        store = EmbeddingConstraintStore()
        assert store.dim is None
        store.add_constraint(np.ones(16, dtype=np.float32), True)
        assert store.dim == 16


# ---------------------------------------------------------------------------
# SCENARIO-STORE-030: full AUROC benchmark
# ---------------------------------------------------------------------------

class TestAUROCBenchmark:
    """SCENARIO-STORE-030: Compression preserves retrieval AUROC > 0.75."""

    def _build_store_and_queries(
        self,
    ) -> tuple[EmbeddingConstraintStore, np.ndarray, list[bool]]:
        """Build synthetic 10-session store and 50 held-out queries."""
        rng = np.random.default_rng(42)
        N_SESSIONS, PER_SESSION, D, N_CLUSTERS = 10, 100, 64, 5
        store = EmbeddingConstraintStore()

        centres = rng.standard_normal((N_CLUSTERS, D)).astype(np.float32)
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        cluster_label = [c % 2 == 1 for c in range(N_CLUSTERS)]

        for i in range(N_SESSIONS * PER_SESSION):
            c = i % N_CLUSTERS
            noise = rng.standard_normal(D).astype(np.float32) * 0.15
            store.add_constraint(centres[c] + noise, cluster_label[c])

        queries = []
        gt_labels = []
        for i in range(50):
            c = i % N_CLUSTERS
            noise = rng.standard_normal(D).astype(np.float32) * 0.15
            q = centres[c] + noise
            q /= np.linalg.norm(q)
            queries.append(q)
            gt_labels.append(cluster_label[c])

        return store, np.stack(queries), gt_labels

    def _score(self, store: EmbeddingConstraintStore, queries: np.ndarray) -> list[float]:
        """Score each query as the fraction of True labels among top-5 neighbours.

        Why fraction-of-True rather than max similarity:
            Max cosine similarity is high for ALL queries that are close to ANY
            cluster centre, regardless of the cluster's label.  This means True
            and False queries both get high scores, so roc_auc_score sees no
            discrimination signal.  Fraction-of-True gives a score in [0, 1]
            that is naturally high for queries near True clusters and low for
            queries near False clusters — exactly what AUROC measures.
        """
        scores = []
        for q in queries:
            neighbours = store.retrieve(q, top_k=5)
            if not neighbours:
                scores.append(0.0)
            else:
                frac_true = sum(1 for _, lbl in neighbours if lbl) / len(neighbours)
                scores.append(frac_true)
        return scores

    def test_auroc_before_compression_above_chance(self) -> None:
        """Retrieval AUROC before compression must be well above 0.5."""
        store, queries, gt_labels = self._build_store_and_queries()
        scores = self._score(store, queries)
        auroc = roc_auc_score(gt_labels, scores)
        assert auroc > 0.7, f"Pre-compression AUROC too low: {auroc:.4f}"

    def test_auroc_after_compression_viable(self) -> None:
        """SCENARIO-STORE-030: AUROC after K=32 compression must exceed 0.75."""
        store, queries, gt_labels = self._build_store_and_queries()
        store.compress(k=32)
        scores = self._score(store, queries)
        auroc = roc_auc_score(gt_labels, scores)
        assert auroc > 0.75, (
            f"Post-compression AUROC {auroc:.4f} is below the 0.75 viability threshold. "
            "This means K=32 centroid compression degrades retrieval too much on this "
            "synthetic benchmark. Consider increasing K or re-evaluating the compression strategy."
        )

    def test_store_size_after_compression(self) -> None:
        """After compression, store holds exactly K=32 constraints."""
        store, _, _ = self._build_store_and_queries()
        assert len(store) == 1000
        store.compress(k=32)
        assert len(store) == 32

    def test_compression_ratio_1000_to_32(self) -> None:
        """Compression ratio from 1000 to 32 is ~31.25."""
        compressor = MemoryBankCompressor(k=32)
        ratio = compressor.compression_ratio(1000)
        assert abs(ratio - 31.25) < 1e-3
