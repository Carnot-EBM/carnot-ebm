"""MemoryBankCompressor — online K-means compression for EBM constraint memory banks.

**Why this module exists:**
    Inspired by arXiv 2601.00756 (Memory Bank Compression for Continual Adaptation).
    After 10+ verification sessions, EmbeddingConstraintStore may hold thousands of
    embeddings.  Most are redundant near-duplicates clustered around a small number
    of semantic concepts.  Replacing N embeddings with K=32 centroids (majority-vote
    labels) achieves ~31x storage reduction while the retrieval AUROC stays above
    0.75 — the threshold we define as "compression viable."

**How it works:**
    1. MiniBatchKMeans (sklearn) fits K cluster centres to the (N, D) embedding matrix.
    2. Each centroid inherits the majority label from the cluster members it absorbed.
    3. The caller replaces the store's full embedding list with the K centroids.

**What it does NOT fix:**
    The RETRO-CONSTRAINT-ZERO-DELTA bug is in the retrieve() query path (un-normalised
    query vector), not the storage path.  Compressing the storage does not touch the
    query path and will not fix or worsen that bug.

Spec: REQ-STORE-020, SCENARIO-STORE-030
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import MiniBatchKMeans


class MemoryBankCompressor:
    """Compress a large embedding memory bank to K centroids via online K-means.

    Parameters
    ----------
    k : int
        Target number of centroids.  Default 32 per arXiv 2601.00756 §4.2.

    Attributes
    ----------
    k : int
        Number of target centroids.
    kmeans : MiniBatchKMeans
        Fitted sklearn clusterer (available after compress() is called).
    """

    def __init__(self, k: int = 32) -> None:
        """Initialise with the target centroid count."""
        self.k = k
        self.kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)

    # ------------------------------------------------------------------
    # Core compression
    # ------------------------------------------------------------------

    def compress(
        self,
        embeddings: np.ndarray,
        labels: list,
    ) -> tuple[np.ndarray, list]:
        """Compress N embeddings into K centroids with majority-vote labels.

        When N <= K, compression is a no-op: we return the original arrays
        unchanged so callers can always call compress() unconditionally without
        checking first.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
            Constraint embeddings to compress.  Need not be unit-length — the
            K-means distance metric works on raw coordinates.
        labels : list of length N
            Per-embedding constraint satisfaction labels (bool or int).

        Returns
        -------
        centroids : np.ndarray, shape (min(N, K), D)
            Cluster centres from MiniBatchKMeans.
        centroid_labels : list of length min(N, K)
            Majority-vote label for each centroid's cluster.
        """
        n = len(embeddings)
        if n <= self.k:
            # Nothing to compress — return the original data as-is.
            return embeddings, list(labels)

        self.kmeans.fit(embeddings)
        centroids = self.kmeans.cluster_centers_  # (K, D)
        cluster_assignments = self.kmeans.labels_  # (N,)

        centroid_labels: list = []
        for c in range(self.k):
            member_indices = [i for i in range(n) if cluster_assignments[i] == c]
            if member_indices:
                member_label_vals = [labels[i] for i in member_indices]
                majority = Counter(member_label_vals).most_common(1)[0][0]
            else:
                # Empty cluster (rare with MiniBatchKMeans, but handle it).
                majority = labels[0] if labels else False
            centroid_labels.append(majority)

        return centroids, centroid_labels

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def compression_ratio(self, original_n: int) -> float:
        """Compute the storage reduction factor N / K.

        A ratio of 31.25 means 1000 embeddings compressed to 32 centroids.
        Values < 1 indicate the store was smaller than K (no compression
        would have occurred).

        Parameters
        ----------
        original_n : int
            Number of embeddings before compression.
        """
        return original_n / self.k
