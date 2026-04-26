"""EmbeddingConstraintStore — dense vector store for EBM constraint embeddings.

**Why this module exists:**
    The Carnot verify-repair pipeline needs to retrieve past constraints that are
    semantically similar to a new candidate output.  Instead of keyword search
    (brittle, language-specific), we embed each constraint as a float32 vector
    and retrieve by cosine similarity.  This mirrors how neural retrieval systems
    (DPR, FAISS) work, but stays lightweight: pure NumPy, no external index.

**Known bug — RETRO-CONSTRAINT-ZERO-DELTA:**
    The retrieve() method currently computes raw dot products without normalising
    the query vector to unit length first.  When all stored embeddings are already
    unit-normalised (as they are after add_constraint()), the dot product equals
    the cosine similarity — BUT only if the query is also unit-length.  If the
    query arrives un-normalised (e.g. from a model that returns L2 norms != 1),
    all similarity scores collapse to near-zero, producing a flat ranking.  This
    is the "delta=0" symptom in Exp 847/848.

    DO NOT fix this bug in this module without a dedicated experiment: the fix
    belongs in the query path and needs its own AUROC measurement to confirm
    the repair did not introduce regressions elsewhere.

Spec: REQ-STORE-010 (add_constraint), REQ-STORE-011 (retrieve cosine),
      REQ-STORE-020 (compression via MemoryBankCompressor),
      SCENARIO-STORE-010, SCENARIO-STORE-020, SCENARIO-STORE-030
"""

from __future__ import annotations

import numpy as np


class EmbeddingConstraintStore:
    """Dense embedding store for LLM output constraints.

    Each constraint is stored as a unit-normalised float32 embedding vector plus
    a boolean label (True = constraint satisfied, False = constraint violated).
    Retrieval is by cosine similarity, which equals the dot product when both
    query and stored vectors are unit-length.

    Attributes
    ----------
    embeddings : np.ndarray of shape (N, D) or empty (0, 0)
        All stored constraint embeddings, unit-normalised at add time.
    labels : list[bool]
        Parallel list of constraint-satisfaction labels.
    dim : int or None
        Embedding dimension, inferred from the first added constraint.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self._embeddings: list[np.ndarray] = []
        self.labels: list[bool] = []
        self.dim: int | None = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_constraint(self, embedding: np.ndarray, label: bool) -> None:
        """Store a constraint embedding with its satisfaction label.

        The embedding is L2-normalised before storage so that all stored
        vectors are unit-length.  This makes cosine similarity equivalent
        to a dot product during retrieval, which is faster on large stores.

        Parameters
        ----------
        embedding : np.ndarray, shape (D,)
            Raw embedding vector.  Need not be pre-normalised.
        label : bool
            True if the constraint was satisfied; False if violated.
        """
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        if self.dim is None:
            self.dim = len(vec)
        self._embeddings.append(vec)
        self.labels.append(bool(label))

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> np.ndarray:
        """Return all stored embeddings as a (N, D) float32 array.

        Returns an empty (0, 0) array when the store is empty so callers
        can always call .shape without guarding for None.
        """
        if not self._embeddings:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(self._embeddings, axis=0)

    def __len__(self) -> int:
        return len(self._embeddings)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: np.ndarray, top_k: int = 5) -> list[tuple[float, bool]]:
        """Return the top-k most similar constraints by cosine similarity.

        **Note on the RETRO-CONSTRAINT-ZERO-DELTA bug:**
            This method does NOT normalise the query.  If the query vector is
            not unit-length, similarity scores will be scaled by ||query||,
            which changes rankings when stored vectors have varying norms
            (they don't — stored vectors are unit-length — but the query's
            scale still affects the absolute score values used by AUROC).
            Until a dedicated fix experiment confirms normalising here is
            safe, leave this as-is and note it in experiment artefacts.

        Parameters
        ----------
        query : np.ndarray, shape (D,)
            Query embedding.
        top_k : int
            Number of nearest neighbours to return.

        Returns
        -------
        list of (similarity_score, label) tuples, descending by score.
        """
        if len(self._embeddings) == 0:
            return []
        stored = self.embeddings  # (N, D)
        q = np.asarray(query, dtype=np.float32).ravel()
        # dot product — equals cosine similarity when both sides are unit-length
        scores = stored @ q  # (N,)
        top_k = min(top_k, len(scores))
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [(float(scores[i]), self.labels[i]) for i in idx]

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(self, k: int = 32) -> None:
        """Compress the store in-place using K-means centroid compression.

        After many sessions of accumulation the store may hold thousands of
        embeddings, most of which are redundant near-duplicates.  K-means
        compression replaces the full set with K centroids (majority-vote
        labels), reducing memory by ~N/K while preserving retrieval AUROC.

        Uses MemoryBankCompressor under the hood (REQ-STORE-020).

        Parameters
        ----------
        k : int
            Target number of centroids.  No-op when len(self) <= k.
        """
        from carnot.stores.memory_bank_compressor import MemoryBankCompressor  # noqa: PLC0415

        if len(self) <= k:
            return
        emb = self.embeddings
        compressor = MemoryBankCompressor(k=k)
        centroids, new_labels = compressor.compress(emb, self.labels)
        self._embeddings = [centroids[i] for i in range(len(centroids))]
        self.labels = list(new_labels)
