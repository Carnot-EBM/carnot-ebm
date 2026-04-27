"""Tests for Exp 970 — PPSEBM progressive parameter selection for Tier 2 cross-session memory.

**Why these tests exist:**
    Exp 748 showed plateau (zero template additions after session 1).  This experiment
    introduces ProgressiveEmbeddingConstraintStore to break that plateau by isolating
    parameter groups per constraint-type cluster.  These tests verify the core invariants
    of the new implementation: group spawning, replay, precision, and the plateau outcome.

Spec: REQ-STORE-010, REQ-STORE-011
"""

from __future__ import annotations

import sys
import os
import json

import numpy as np
import pytest

# Allow importing the experiment script as a module.
_SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)
sys.path.insert(0, _SCRIPT_DIR)

from experiment_970_ppsebm_tier2_crosssession_memory import (  # noqa: E402
    ProgressiveEmbeddingConstraintStore,
    _ParameterGroup,
    _cosine_distance,
    _unit,
    _sample_embedding,
    _get_centroid,
    _compute_precision,
    run_relay,
)


# ---------------------------------------------------------------------------
# _unit helper
# ---------------------------------------------------------------------------


class TestUnit:
    """Tests for the _unit normalisation helper.  Spec: REQ-STORE-010"""

    def test_unit_unit_length(self):
        """Output of _unit is always unit L2 length.

        Spec: REQ-STORE-010
        """
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = _unit(v)
        assert pytest.approx(np.linalg.norm(result), abs=1e-6) == 1.0

    def test_unit_zero_vector_unchanged(self):
        """Zero vector stays zero — _unit does not divide by zero.

        Spec: REQ-STORE-010
        """
        z = np.zeros(4, dtype=np.float32)
        result = _unit(z)
        assert np.allclose(result, 0.0)


# ---------------------------------------------------------------------------
# _cosine_distance helper
# ---------------------------------------------------------------------------


class TestCosineDistance:
    """Tests for _cosine_distance.  Spec: REQ-STORE-011"""

    def test_identical_vectors_distance_zero(self):
        """Cosine distance between identical vectors is 0.

        Spec: REQ-STORE-011
        """
        v = _unit(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert pytest.approx(_cosine_distance(v, v), abs=1e-6) == 0.0

    def test_orthogonal_vectors_distance_one(self):
        """Orthogonal vectors have cosine distance 1.0.

        Spec: REQ-STORE-011
        """
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert pytest.approx(_cosine_distance(a, b), abs=1e-6) == 1.0

    def test_opposite_vectors_distance_two(self):
        """Opposite vectors have cosine distance 2.0.

        Spec: REQ-STORE-011
        """
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert pytest.approx(_cosine_distance(a, b), abs=1e-6) == 2.0


# ---------------------------------------------------------------------------
# _ParameterGroup
# ---------------------------------------------------------------------------


class TestParameterGroup:
    """Tests for _ParameterGroup isolated parameter group.  Spec: REQ-STORE-010"""

    def test_add_updates_centroid(self):
        """Adding embeddings updates the centroid toward the mean.

        Spec: REQ-STORE-010
        """
        c = _unit(np.array([1.0, 0.0], dtype=np.float32))
        group = _ParameterGroup(centroid=c.copy())
        v1 = _unit(np.array([0.9, 0.1], dtype=np.float32))
        v2 = _unit(np.array([0.8, 0.2], dtype=np.float32))
        group.add(v1, True)
        group.add(v2, False)
        assert len(group.embeddings) == 2
        assert len(group.labels) == 2
        # Centroid should be unit-length after updates.
        assert pytest.approx(np.linalg.norm(group.centroid), abs=1e-5) == 1.0

    def test_replay_anchors_length(self):
        """replay_anchors returns at most n embeddings.

        Spec: REQ-STORE-010
        """
        c = _unit(np.ones(4, dtype=np.float32))
        group = _ParameterGroup(centroid=c.copy())
        for i in range(10):
            v = _unit(np.random.default_rng(i).standard_normal(4).astype(np.float32))
            group.add(v, True)
        anchors = group.replay_anchors(n=3)
        assert len(anchors) == 3

    def test_replay_anchors_empty_group(self):
        """replay_anchors returns empty list for an empty group.

        Spec: REQ-STORE-010
        """
        c = _unit(np.ones(4, dtype=np.float32))
        group = _ParameterGroup(centroid=c.copy())
        assert group.replay_anchors(n=5) == []


# ---------------------------------------------------------------------------
# ProgressiveEmbeddingConstraintStore
# ---------------------------------------------------------------------------


class TestProgressiveEmbeddingConstraintStore:
    """Tests for the PPSEBM-inspired progressive parameter store.  Spec: REQ-STORE-010, REQ-STORE-011"""

    def test_first_constraint_spawns_new_group(self):
        """First add_constraint always creates a new parameter group.

        Spec: REQ-STORE-010
        """
        store = ProgressiveEmbeddingConstraintStore()
        v = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        is_new = store.add_constraint(v, True, "test")
        assert is_new is True
        assert store.cluster_count == 1

    def test_similar_constraint_absorbed_into_group(self):
        """A constraint very similar to an existing centroid is absorbed (no new group).

        Spec: REQ-STORE-010
        """
        store = ProgressiveEmbeddingConstraintStore(threshold=0.5)
        base = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        store.add_constraint(base, True, "type_a")
        # Slightly perturbed version of the same vector.
        perturbed = _unit(base + 0.01 * np.array([0.0, 1.0, 0.0], dtype=np.float32))
        is_new = store.add_constraint(perturbed, False, "type_a")
        assert is_new is False
        assert store.cluster_count == 1

    def test_distant_constraint_spawns_new_group(self):
        """A constraint whose cosine distance > threshold spawns a new group.

        Spec: REQ-STORE-010
        """
        store = ProgressiveEmbeddingConstraintStore(threshold=0.5)
        a = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        b = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        store.add_constraint(a, True, "type_a")
        is_new = store.add_constraint(b, False, "type_b")
        assert is_new is True
        assert store.cluster_count == 2

    def test_replay_returns_embeddings_per_group(self):
        """replay() returns anchors from all groups.

        Spec: REQ-STORE-010
        """
        store = ProgressiveEmbeddingConstraintStore(threshold=0.5)
        a = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        b = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        # Add 3 embeddings near a.
        for _ in range(3):
            store.add_constraint(
                _unit(a + 0.01 * np.random.default_rng(0).standard_normal(3).astype(np.float32)),
                True,
                "type_a",
            )
        # Add 3 embeddings near b.
        for _ in range(3):
            store.add_constraint(
                _unit(b + 0.01 * np.random.default_rng(1).standard_normal(3).astype(np.float32)),
                False,
                "type_b",
            )
        anchors = store.replay(n_per_group=2)
        # 2 groups × 2 anchors = 4 total.
        assert len(anchors) == 4

    def test_cluster_count_monotone_non_decreasing(self):
        """cluster_count can only stay the same or increase; it never shrinks.

        Spec: REQ-STORE-010
        """
        store = ProgressiveEmbeddingConstraintStore(threshold=0.5)
        rng = np.random.default_rng(42)
        prev = 0
        for _ in range(20):
            v = _unit(rng.standard_normal(16).astype(np.float32))
            store.add_constraint(v, True)
            assert store.cluster_count >= prev
            prev = store.cluster_count


# ---------------------------------------------------------------------------
# _compute_precision
# ---------------------------------------------------------------------------


class TestComputePrecision:
    """Tests for the leave-one-out precision metric.  Spec: REQ-STORE-011"""

    def test_precision_perfect_homogeneous_groups(self):
        """All-True or all-False groups should achieve precision=1.0.

        Spec: REQ-STORE-011
        """
        store = ProgressiveEmbeddingConstraintStore(threshold=0.1)
        rng = np.random.default_rng(1)
        base = _unit(rng.standard_normal(16).astype(np.float32))
        # Add a tight cluster of 5 all-True embeddings.
        for i in range(5):
            v = _unit(base + 0.001 * rng.standard_normal(16).astype(np.float32))
            store.add_constraint(v, True)
        prec = _compute_precision(store)
        assert prec == pytest.approx(1.0, abs=0.01)

    def test_precision_empty_store(self):
        """Empty store returns precision=1.0 (no errors, by convention).

        Spec: REQ-STORE-011
        """
        store = ProgressiveEmbeddingConstraintStore()
        assert _compute_precision(store) == 1.0

    def test_precision_single_element_group(self):
        """Single-element groups are skipped; precision still returns 1.0.

        Spec: REQ-STORE-011
        """
        store = ProgressiveEmbeddingConstraintStore()
        store.add_constraint(_unit(np.array([1.0, 0.0], dtype=np.float32)), True)
        assert _compute_precision(store) == 1.0


# ---------------------------------------------------------------------------
# run_relay integration
# ---------------------------------------------------------------------------


class TestRunRelay:
    """Integration test for the 10-session relay.  Spec: REQ-STORE-010, REQ-STORE-011"""

    def test_relay_schema_fields_present(self):
        """run_relay result contains all required schema fields.

        Spec: REQ-STORE-010
        """
        result = run_relay(n_sessions=3, n_questions_per_session=5, replay_n=2)
        required = [
            "templates_added_per_session",
            "cluster_count_per_session",
            "precision_per_session",
            "sessions_with_new_templates",
            "plateau_broken",
            "honest_verdict",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_relay_list_lengths_match_sessions(self):
        """Per-session lists have exactly n_sessions entries.

        Spec: REQ-STORE-010
        """
        n = 4
        result = run_relay(n_sessions=n, n_questions_per_session=5, replay_n=2)
        assert len(result["templates_added_per_session"]) == n
        assert len(result["cluster_count_per_session"]) == n
        assert len(result["precision_per_session"]) == n

    def test_relay_session1_has_new_templates(self):
        """Session 1 always adds templates (the store is empty at start).

        Spec: REQ-STORE-010
        """
        result = run_relay(n_sessions=5, n_questions_per_session=10, replay_n=2)
        assert result["templates_added_per_session"][0] > 0

    def test_relay_cluster_count_monotone(self):
        """Cluster count is monotonically non-decreasing across sessions.

        Spec: REQ-STORE-010
        """
        result = run_relay(n_sessions=5, n_questions_per_session=10, replay_n=2)
        counts = result["cluster_count_per_session"]
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1]

    def test_relay_full_10_session_plateau_broken(self):
        """Full 10-session relay must break the plateau (honest_verdict check).

        This is the primary success criterion for Exp 970.
        Spec: REQ-STORE-010, REQ-STORE-011
        """
        result = run_relay(n_sessions=10, n_questions_per_session=20, replay_n=5)
        # plateau_broken requires sessions_with_new_templates >= 3 (beyond session 1).
        assert result["plateau_broken"] is True
        assert result["sessions_with_new_templates"] >= 3
        assert result["honest_verdict"] == "ppsebm_plateau_broken"

    def test_relay_precision_values_valid(self):
        """Precision values are valid floats in [0.0, 1.0] for all sessions.

        **Why not a hard floor?**
            Precision is computed by LOO retrieval within isolated groups.  In
            session 1 many groups contain only 1-2 embeddings with randomly
            assigned labels (70% satisfied), so the LOO estimate is noisy.  This
            test validates shape and range, not absolute magnitude.  The
            plateau-broken result is the primary success criterion for Exp 970.

        Spec: REQ-STORE-011
        """
        result = run_relay(n_sessions=5, n_questions_per_session=10, replay_n=2)
        for sess, prec in enumerate(result["precision_per_session"], start=1):
            assert 0.0 <= prec <= 1.0, f"Session {sess} precision {prec} out of [0, 1]"

    def test_honest_verdict_values(self):
        """honest_verdict is one of the two valid values.

        Spec: REQ-STORE-010
        """
        result = run_relay(n_sessions=3, n_questions_per_session=5, replay_n=2)
        assert result["honest_verdict"] in ("ppsebm_plateau_broken", "ppsebm_plateau_persists")
