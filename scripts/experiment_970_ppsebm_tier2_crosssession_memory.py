#!/usr/bin/env python3
"""Experiment 970 — PPSEBM-inspired progressive parameter selection for Tier 2 cross-session memory.

**Researcher summary (why this experiment exists):**
    Exps 748 and 761 showed that the EmbeddingConstraintStore saturates after session 1:
    templates are added in S1, then zero additions in S2-S10 even when new error patterns
    appear.  The root cause (from arXiv 2512.15658 PPSEBM analysis) is that the single
    shared parameter space gets saturated — new constraint types interfere with the
    already-learned embedding space, exactly like catastrophic forgetting in continual
    learning.

**What PPSEBM progressive parameter selection does:**
    Instead of one flat embedding store, we maintain separate "parameter groups" — one
    per discovered constraint TYPE cluster.  When a new constraint arrives whose cluster
    centroid is more than 0.5 cosine distance from ALL existing cluster centroids, we
    create a fresh parameter group (a new EmbeddingConstraintStore slice) for that type.
    Each group learns its own subspace without interfering with others.

**Replay to prevent forgetting:**
    Before each session, we generate N=5 synthetic replay examples per group by querying
    each group's centroid embedding.  This "rehearsal" keeps the retrieval quality high
    for old constraint types while new ones are being learned.

**What this experiment measures:**
    - templates_added_per_session: did we keep adding templates after session 1?
    - cluster_count_per_session: is the cluster count growing (=new types discovered)?
    - precision_per_session: does precision stay high as we add more constraint types?
    - plateau_broken: True if templates were added in >= 3 sessions beyond session 1.

Spec: REQ-STORE-010, REQ-STORE-011
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np

# JAX_PLATFORMS=cpu is set in the environment per CLAUDE.md for reproducibility.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# ---------------------------------------------------------------------------
# Progressive Parameter-Selected EmbeddingConstraintStore
# ---------------------------------------------------------------------------

_COSINE_DISTANCE_THRESHOLD = 0.5  # above this = new cluster


@dataclass
class _ParameterGroup:
    """One isolated parameter group = one constraint-type cluster.

    **Why isolated groups matter:**
        In a flat store, a new constraint type's embedding is stored in the same
        space as all other types.  At retrieval time, the query gets "confused"
        by semantically distant stored vectors and the similarity scores collapse,
        making it impossible to distinguish whether a new pattern matches anything.
        An isolated group only contains embeddings of similar semantic class,
        so cosine similarity within the group stays meaningful.

    Attributes:
        centroid: Running mean of all embeddings in this group (unit-normalised).
        embeddings: Raw stored embeddings (unit-normalised).
        labels: Parallel satisfaction labels.
        constraint_type: The constraint_type tag that seeded this group.
    """

    centroid: np.ndarray
    embeddings: list[np.ndarray] = field(default_factory=list)
    labels: list[bool] = field(default_factory=list)
    constraint_type: str = ""

    def add(self, vec: np.ndarray, label: bool) -> None:
        """Add an embedding to this group and update the running centroid."""
        v = _unit(vec)
        self.embeddings.append(v)
        self.labels.append(label)
        # Update centroid as running mean, then renormalise.
        n = len(self.embeddings)
        self.centroid = _unit(self.centroid * (n - 1) / n + v / n)

    def replay_anchors(self, n: int = 5) -> list[np.ndarray]:
        """Return up to n stored embeddings closest to the centroid.

        These serve as "replay examples" — by retrieving the most centroid-like
        embeddings we rehearse the most representative constraint patterns for
        this group, preventing forgetting during subsequent session updates.
        """
        if not self.embeddings:
            return []
        embs = np.stack(self.embeddings, axis=0)  # (K, D)
        scores = embs @ self.centroid  # (K,)
        top_k = min(n, len(self.embeddings))
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [self.embeddings[i] for i in idx]


def _unit(v: np.ndarray) -> np.ndarray:
    """Return v normalised to unit L2 length.  Zero vector stays zero."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance = 1 - cosine_similarity.  Range [0, 2]."""
    sim = float(np.dot(_unit(a), _unit(b)))
    return 1.0 - sim


class ProgressiveEmbeddingConstraintStore:
    """EmbeddingConstraintStore with PPSEBM-inspired progressive parameter groups.

    **Detailed explanation for engineers:**
        This replaces the flat EmbeddingConstraintStore with a collection of
        isolated parameter groups, one per discovered constraint-type cluster.
        The algorithm mirrors the Progressive Parameter Selection approach from
        arXiv 2512.15658 (PPSEBM):

        1. When a new embedding arrives, compute its cosine distance to every
           existing group centroid.
        2. If the minimum distance is <= threshold: add to the closest group.
        3. If the minimum distance is > threshold (or no groups exist): create
           a NEW group — a fresh, dedicated embedding subspace for this type.

        By isolating parameter spaces, new constraint types can be learned without
        overwriting the geometry of existing types.  Replay keeps old types fresh.

    Attributes:
        groups: List of isolated parameter groups, one per cluster.
        threshold: Cosine distance above which a new group is spawned.
    """

    def __init__(self, threshold: float = _COSINE_DISTANCE_THRESHOLD) -> None:
        self.groups: list[_ParameterGroup] = []
        self.threshold = threshold

    def add_constraint(self, embedding: np.ndarray, label: bool, constraint_type: str = "") -> bool:
        """Store a constraint and return True if a new parameter group was created.

        **Returns True for new groups** so callers can count how many new constraint
        types were discovered in a session — that is the signal this experiment tracks.

        Args:
            embedding: Raw embedding vector for the constraint text.
            label: True if constraint was satisfied, False if violated.
            constraint_type: Hint tag (e.g. "arithmetic") used to label new groups.

        Returns:
            True if a new parameter group was spawned, False if the constraint
            was absorbed into an existing group.
        """
        vec = _unit(np.asarray(embedding, dtype=np.float32).ravel())

        if not self.groups:
            self.groups.append(
                _ParameterGroup(centroid=vec.copy(), constraint_type=constraint_type)
            )
            self.groups[-1].add(vec, label)
            return True

        # Find the closest existing group.
        distances = [_cosine_distance(vec, g.centroid) for g in self.groups]
        min_dist = min(distances)
        closest_idx = int(np.argmin(distances))

        if min_dist <= self.threshold:
            self.groups[closest_idx].add(vec, label)
            return False
        else:
            # New constraint type: spawn an isolated parameter group.
            new_group = _ParameterGroup(centroid=vec.copy(), constraint_type=constraint_type)
            new_group.add(vec, label)
            self.groups.append(new_group)
            return True

    def replay(self, n_per_group: int = 5) -> list[np.ndarray]:
        """Collect n_per_group replay anchors from each existing parameter group.

        **Why replay matters:**
            Without rehearsal, adding many new groups in later sessions shifts the
            overall retrieval landscape.  By re-presenting the centroid anchors of
            old groups before each session, we keep those constraints "fresh" in the
            system's attention — the direct parallel to experience replay in
            continual learning.

        Returns:
            Flat list of anchor embeddings (unit-normalised) from all groups.
        """
        anchors: list[np.ndarray] = []
        for g in self.groups:
            anchors.extend(g.replay_anchors(n_per_group))
        return anchors

    @property
    def cluster_count(self) -> int:
        """Number of active parameter groups (= discovered constraint type clusters)."""
        return len(self.groups)


# ---------------------------------------------------------------------------
# Synthetic CPU model for the 10-session relay
# ---------------------------------------------------------------------------

# We use random embeddings keyed to synthetic constraint-type vocabularies.
# The vocabulary grows session by session so that new semantic types appear
# (triggering new parameter groups) without requiring a real LLM.
_RNG = np.random.default_rng(970)  # deterministic for reproducibility

_BASE_CONSTRAINT_TYPES = [
    "arithmetic_add",
    "arithmetic_mul",
    "type_check_int",
    "type_check_str",
    "implication_causal",
]

# New types that emerge in later sessions, simulating a fresh error pattern the
# model starts making that the existing store has never seen before.
_LATE_CONSTRAINT_TYPES_BY_SESSION = {
    3: ["loop_bound_off_by_one"],
    5: ["factual_population", "negation_double"],
    7: ["universal_quantifier", "return_type_mismatch"],
    9: ["disjunction_exclusive"],
}

# Cluster centroids per type (unit-normalised, 64-d).  These are fixed so that
# the cosine-distance check is deterministic across sessions.
_DIM = 64
_TYPE_CENTROIDS: dict[str, np.ndarray] = {}


def _get_centroid(ctype: str) -> np.ndarray:
    """Return (or create) a stable unit-normalised centroid for a constraint type.

    **Why stable centroids?**
        In a real system each constraint type has a natural semantic cluster in
        embedding space.  For this synthetic test we fix one random centroid per
        type so that the ProgressiveEmbeddingConstraintStore sees a consistent
        geometry across sessions — mimicking what a real embedding model would
        produce for semantically coherent constraint classes.
    """
    if ctype not in _TYPE_CENTROIDS:
        v = _RNG.standard_normal(_DIM).astype(np.float32)
        _TYPE_CENTROIDS[ctype] = _unit(v)
    return _TYPE_CENTROIDS[ctype]


def _sample_embedding(ctype: str, noise: float = 0.15) -> np.ndarray:
    """Sample a noisy embedding near the centroid for a given constraint type.

    A small amount of noise (default 0.15) gives realistic intra-cluster
    variance without pushing the sample outside the cluster's cosine ball.
    """
    centroid = _get_centroid(ctype)
    noise_vec = _RNG.standard_normal(_DIM).astype(np.float32) * noise
    return _unit(centroid + noise_vec)


def _generate_session_constraints(
    session_idx: int, n_questions: int
) -> list[tuple[np.ndarray, bool, str]]:
    """Generate synthetic constraints for one session.

    **Session design:**
        - Every session sees the base 5 constraint types (old patterns).
        - Sessions 3, 5, 7, 9 introduce one or more new constraint types
          whose centroids are intentionally far (cosine dist > 0.5) from
          all existing centroids.  This simulates the scenario where the LLM
          starts making a class of error it never made before — exactly the
          scenario that caused plateau in Exp 748.
        - Each question produces 1-3 constraints of mixed types.
        - Labels (satisfied/violated) are ~70% satisfied to match real-pipeline
          base rates observed in Exps 748 and 761.

    Returns:
        List of (embedding, label, constraint_type) tuples.
    """
    # Accumulate active types up to this session.
    active_types = list(_BASE_CONSTRAINT_TYPES)
    for sess_threshold, new_types in _LATE_CONSTRAINT_TYPES_BY_SESSION.items():
        if session_idx >= sess_threshold:
            active_types.extend(new_types)

    constraints = []
    per_q = max(1, n_questions // len(active_types))
    for i in range(n_questions):
        ctype = active_types[i % len(active_types)]
        n_per_q = _RNG.integers(1, 4)
        for _ in range(n_per_q):
            emb = _sample_embedding(ctype)
            label = bool(_RNG.random() < 0.7)
            constraints.append((emb, label, ctype))

    return constraints


# ---------------------------------------------------------------------------
# Precision metric
# ---------------------------------------------------------------------------


def _compute_precision(store: ProgressiveEmbeddingConstraintStore) -> float:
    """Compute retrieval precision: fraction of top-1 retrievals that match label.

    **What this measures:**
        For each stored embedding, retrieve the top-1 neighbour from the store
        (excluding the query itself via leave-one-out approximation).  A match
        means the retrieved constraint has the same satisfaction label.

        This is the same metric used in Exps 748/761 so results are comparable.
        Perfect grouping within isolated parameter groups should keep this >= 0.9.
    """
    total = 0
    correct = 0
    for gidx, group in enumerate(store.groups):
        for i, (emb, lbl) in enumerate(zip(group.embeddings, group.labels)):
            # Build a temporary store without this exact embedding (LOO).
            candidates = []
            for j, (e2, l2) in enumerate(zip(group.embeddings, group.labels)):
                if j != i:
                    candidates.append((float(np.dot(emb, e2)), l2))
            if not candidates:
                continue
            best = max(candidates, key=lambda x: x[0])
            total += 1
            if best[1] == lbl:
                correct += 1

    return correct / total if total > 0 else 1.0


# ---------------------------------------------------------------------------
# 10-session relay
# ---------------------------------------------------------------------------


def run_relay(
    n_sessions: int = 10,
    n_questions_per_session: int = 20,
    replay_n: int = 5,
) -> dict:
    """Run the 10-session cross-session relay with progressive parameter selection.

    **How the relay works:**
        Each "session" simulates a round of LLM-generated outputs being verified
        and the resulting constraint embeddings stored.  Between sessions:
        1. Replay anchors are fetched from all existing parameter groups.
        2. Replay embeddings are re-inserted (no label change) to keep groups active.
        3. New constraints from the current session are added, potentially spawning
           new groups if the constraint type hasn't been seen before.

        We record how many NEW groups (= templates) were added each session.
        Exp 748 baseline: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0].
        Target: non-zero additions in sessions 3-7 (sessions indexed 1-10).

    Args:
        n_sessions: Number of simulated sessions.
        n_questions_per_session: Synthetic questions per session.
        replay_n: Replay anchors per parameter group between sessions.

    Returns:
        Result dict matching the required experiment schema.
    """
    store = ProgressiveEmbeddingConstraintStore(threshold=_COSINE_DISTANCE_THRESHOLD)

    templates_added_per_session: list[int] = []
    cluster_count_per_session: list[int] = []
    precision_per_session: list[float] = []

    for sess_idx in range(1, n_sessions + 1):
        new_groups_this_session = 0

        # --- Step 1: Replay before adding new constraints ---
        # This rehearsal step is the key PPSEBM addition vs Exp 748.
        # Without it, old groups' centroids drift as the overall geometry grows,
        # reducing recall on earlier constraint types.
        if sess_idx > 1:
            replay_embs = store.replay(n_per_group=replay_n)
            for remb in replay_embs:
                # Replay embeddings re-use their group's majority label.
                # We approximate by using True (most constraints are satisfied).
                store.add_constraint(remb, True, constraint_type="replay")

        # --- Step 2: Add new constraints from this session ---
        constraints = _generate_session_constraints(sess_idx, n_questions_per_session)
        for emb, lbl, ctype in constraints:
            is_new = store.add_constraint(emb, lbl, constraint_type=ctype)
            if is_new:
                new_groups_this_session += 1

        templates_added_per_session.append(new_groups_this_session)
        cluster_count_per_session.append(store.cluster_count)
        precision_per_session.append(_compute_precision(store))

    # --- Analyse results ---
    # sessions_with_new_templates: how many sessions (not counting session 1) saw additions.
    # The target is sessions 3-7 (indices 2-6 in 0-based).
    sessions_with_new_templates = sum(1 for v in templates_added_per_session[1:] if v > 0)

    # plateau_broken: True if at least 3 non-session-1 sessions have new templates.
    plateau_broken = sessions_with_new_templates >= 3

    if plateau_broken:
        honest_verdict = "ppsebm_plateau_broken"
    else:
        honest_verdict = "ppsebm_plateau_persists"

    return {
        "templates_added_per_session": templates_added_per_session,
        "cluster_count_per_session": cluster_count_per_session,
        "precision_per_session": [round(p, 4) for p in precision_per_session],
        "sessions_with_new_templates": sessions_with_new_templates,
        "plateau_broken": plateau_broken,
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 970 and write the result JSON."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.perf_counter()

    result = run_relay(n_sessions=10, n_questions_per_session=20, replay_n=5)

    duration_s = round(time.perf_counter() - t0, 3)
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    run_date = time.strftime("%Y%m%d", time.gmtime())

    artifact = {
        "experiment": 970,
        "title": "PPSEBM Progressive Parameter Selection: Tier 2 Cross-Session Memory",
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "status": "success",
        # Required schema fields
        "templates_added_per_session": result["templates_added_per_session"],
        "cluster_count_per_session": result["cluster_count_per_session"],
        "precision_per_session": result["precision_per_session"],
        "sessions_with_new_templates": result["sessions_with_new_templates"],
        "plateau_broken": result["plateau_broken"],
        "honest_verdict": result["honest_verdict"],
        # Provenance
        "n_sessions": 10,
        "n_questions_per_session": 20,
        "replay_n_per_group": 5,
        "cosine_distance_threshold": _COSINE_DISTANCE_THRESHOLD,
        "embedding_dim": _DIM,
        "prior_experiments": [748, 761],
        "ppsebm_reference": "arXiv 2512.15658",
        "schema": [
            "cluster_count_per_session",
            "cosine_distance_threshold",
            "duration_s",
            "embedding_dim",
            "experiment",
            "finished_at",
            "honest_verdict",
            "n_questions_per_session",
            "n_sessions",
            "plateau_broken",
            "precision_per_session",
            "ppsebm_reference",
            "prior_experiments",
            "replay_n_per_group",
            "run_date",
            "sessions_with_new_templates",
            "started_at",
            "status",
            "templates_added_per_session",
            "title",
        ],
    }

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "experiment_970_ppsebm_tier2_crosssession_memory.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(artifact, fh, indent=2)
        fh.write("\n")

    print(f"[Exp 970] honest_verdict: {artifact['honest_verdict']}")
    print(f"[Exp 970] templates_added_per_session: {artifact['templates_added_per_session']}")
    print(f"[Exp 970] cluster_count_per_session: {artifact['cluster_count_per_session']}")
    print(f"[Exp 970] sessions_with_new_templates: {artifact['sessions_with_new_templates']}")
    print(f"[Exp 970] plateau_broken: {artifact['plateau_broken']}")
    print(f"[Exp 970] Written to {out_path}")


if __name__ == "__main__":
    main()
