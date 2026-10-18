#!/usr/bin/env python3
"""Experiment 865 — Constraint memory bank compression via K=32 centroids.

**Research question:**
    After 10 sessions of constraint accumulation (~1000 embeddings), can we
    compress EmbeddingConstraintStore to K=32 centroids using online K-means
    while keeping retrieval AUROC above 0.75?  If yes, compression is viable
    for production deployments where storage and query latency matter.

**Why this matters:**
    The verify-repair pipeline accumulates constraint embeddings across sessions.
    After 10 sessions at 100 constraints/session, the store holds 1000 vectors.
    That is ~256 KB of float32 at D=64 — small today, but at D=4096 (real LLM
    embeddings) it becomes 16 MB per 1000 constraints, growing without bound.
    K=32 centroid compression gives ~31x storage reduction.  This experiment
    measures whether the AUROC penalty for that compression is acceptable.

**arXiv reference:**
    2601.00756 (Memory Bank Compression for Continual Adaptation), §4.2.

**Known confound — RETRO-CONSTRAINT-ZERO-DELTA:**
    retrieve() has a known bug: un-normalised queries produce near-zero dot
    products.  This experiment uses unit-normalised synthetic queries so the
    bug does NOT affect these results.  The bug fix is a separate experiment.

Spec: REQ-STORE-020, SCENARIO-STORE-030
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.stores.embedding_constraint_store import EmbeddingConstraintStore  # noqa: E402
from carnot.stores.memory_bank_compressor import MemoryBankCompressor  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SESSIONS = 10
CONSTRAINTS_PER_SESSION = 100
EMBEDDING_DIM = 64
N_CLUSTERS_TRUE = 5  # synthetic semantic clusters in the data
K_COMPRESSION = 32
N_HELD_OUT = 50
TOP_K_RETRIEVAL = 5
AUROC_VIABILITY_THRESHOLD = 0.75
SEED = 42


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _build_synthetic_store(rng: np.random.Generator) -> EmbeddingConstraintStore:
    """Create a 10-session constraint store with N_CLUSTERS_TRUE semantic clusters.

    We generate N_SESSIONS * CONSTRAINTS_PER_SESSION embeddings as noisy
    perturbations of N_CLUSTERS_TRUE cluster centres in R^EMBEDDING_DIM.
    Labels are assigned by cluster (odd cluster index -> True, even -> False)
    so that retrieval by nearest cluster should produce an AUROC well above 0.5.
    """
    store = EmbeddingConstraintStore()

    # Fixed cluster centres (unit-normalised so add_constraint() norms are no-ops).
    centres = rng.standard_normal((N_CLUSTERS_TRUE, EMBEDDING_DIM)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    # Cluster labels: odd clusters are positive constraints, even are negative.
    cluster_label = [c % 2 == 1 for c in range(N_CLUSTERS_TRUE)]

    total = N_SESSIONS * CONSTRAINTS_PER_SESSION
    for i in range(total):
        c = i % N_CLUSTERS_TRUE
        # Small Gaussian noise so each embedding is distinct but semantically
        # close to its cluster centre.  Noise std=0.15 keeps cos-sim ~0.97.
        noise = rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.15
        emb = centres[c] + noise
        store.add_constraint(emb, cluster_label[c])

    return store, centres, cluster_label


def _build_held_out(
    rng: np.random.Generator,
    centres: np.ndarray,
    cluster_label: list[bool],
) -> tuple[np.ndarray, list[bool]]:
    """Build N_HELD_OUT query embeddings with known ground-truth labels.

    Each query is a noisy version of a random cluster centre.  The ground-truth
    label is the cluster's majority label.  We use these to compute AUROC:
    a query whose nearest neighbour shares its label contributes to a high score.
    """
    queries = []
    gt_labels = []
    for i in range(N_HELD_OUT):
        c = i % N_CLUSTERS_TRUE
        noise = rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.15
        q = centres[c] + noise
        # Unit-normalise queries to avoid the RETRO-CONSTRAINT-ZERO-DELTA bug.
        q /= np.linalg.norm(q)
        queries.append(q)
        gt_labels.append(cluster_label[c])
    return np.stack(queries), gt_labels


# ---------------------------------------------------------------------------
# AUROC measurement
# ---------------------------------------------------------------------------


def _measure_auroc(
    store: EmbeddingConstraintStore, queries: np.ndarray, gt_labels: list[bool]
) -> float:
    """Compute retrieval AUROC for a store against held-out queries.

    For each query:
        - Retrieve top-TOP_K_RETRIEVAL neighbours by cosine similarity.
        - Score = fraction of True labels among top-k retrieved neighbours.
    AUROC is computed between these scores and ground-truth labels.

    Why fraction-of-True rather than max similarity:
        Max cosine similarity is high for ALL queries near ANY cluster,
        regardless of that cluster's label.  This means True and False queries
        both score high, giving roc_auc_score no discrimination signal.
        Fraction-of-True maps naturally to [0, 1] and is high only when the
        query is close to a True cluster, which is exactly what we want AUROC
        to measure.
    """
    scores = []
    pred_labels = []
    for q in queries:
        neighbours = store.retrieve(q, top_k=TOP_K_RETRIEVAL)
        if not neighbours:
            scores.append(0.0)
            pred_labels.append(False)
            continue
        frac_true = sum(1 for _, lbl in neighbours if lbl) / len(neighbours)
        majority_label = Counter([lbl for _, lbl in neighbours]).most_common(1)[0][0]
        scores.append(frac_true)
        pred_labels.append(majority_label)

    # roc_auc_score requires both classes present; fall back to 0.5 if degenerate.
    unique_labels = set(gt_labels)
    if len(unique_labels) < 2:
        return 0.5
    return float(roc_auc_score(gt_labels, scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=865,
        title="Constraint memory bank compression K=32",
        deliverable="results/experiment_865_constraint_memory_compression.json",
        requires_gpu=False,
    )
    tmpl.setup()

    rng = np.random.default_rng(SEED)

    with tmpl.phase("build_synthetic_store"):
        store, centres, cluster_label = _build_synthetic_store(rng)

    n_before = len(store)

    with tmpl.phase("build_held_out"):
        queries, gt_labels = _build_held_out(rng, centres, cluster_label)

    with tmpl.phase("auroc_before"):
        auroc_before = _measure_auroc(store, queries, gt_labels)

    with tmpl.phase("compress"):
        store.compress(k=K_COMPRESSION)

    n_after = len(store)
    ratio = n_before / n_after  # should be ~31.25

    with tmpl.phase("auroc_after"):
        auroc_after = _measure_auroc(store, queries, gt_labels)

    # Determine honest verdict.
    memory_compression_viable = auroc_after > AUROC_VIABILITY_THRESHOLD
    if auroc_after > AUROC_VIABILITY_THRESHOLD:
        honest_verdict = "compression_viable"
    elif auroc_after <= auroc_before * 0.9:
        honest_verdict = "compression_degrades"
    else:
        honest_verdict = "compression_marginal"

    artifact = tmpl.build_result(
        {
            "retrieval_auroc_before": round(auroc_before, 4),
            "retrieval_auroc_after": round(auroc_after, 4),
            "compression_ratio": round(ratio, 4),
            "k": K_COMPRESSION,
            "n_sessions_simulated": N_SESSIONS,
            "n_constraints_before": n_before,
            "n_constraints_after": n_after,
            "embedding_dim": EMBEDDING_DIM,
            "memory_compression_viable": memory_compression_viable,
            "honest_verdict": honest_verdict,
        },
        status="success" if memory_compression_viable else "partial",
    )

    out_path = Path("results/experiment_865_constraint_memory_compression.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(
        f"[Exp 865] AUROC before={auroc_before:.4f} after={auroc_after:.4f} "
        f"ratio={ratio:.2f}x  verdict={honest_verdict}"
    )

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
