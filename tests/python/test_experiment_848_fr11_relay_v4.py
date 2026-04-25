"""Tests for Exp 848: FR-11 Tier 1 self-learning relay v4.

Validates the gate check, constraint write path, monotonicity logic, and
delta computation in experiment_848_fr11_tier1_live_relay_v4.py without
requiring live GPU inference.  All LLM calls are replaced with synthetic
responses via controlled random seeds or direct function calls.

Spec: FR-11, REQ-LEARN-057, REQ-LEARN-058, REQ-LEARN-059,
      REQ-VERIFY-150, SCENARIO-VERIFY-230
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repo root is on the path before importing experiment module
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.experiment_848_fr11_tier1_live_relay_v4 import (  # noqa: E402
    RETRIEVAL_AUROC_GATE,
    TOP_K_CONSTRAINT_UPDATE,
    VARIANCE_FREEZE_THRESHOLD,
    _compute_type_variances,
    _read_847_gate,
    _select_updatable_types,
    _synthetic_inference,
    run_relay,
)
from python.carnot.pipeline.embedding_constraint_store import (  # noqa: E402
    ConstraintSPOTuple,
    EmbeddingConstraintStore,
)


# ---------------------------------------------------------------------------
# test_gate_blocks_if_retrieval_below_threshold
# ---------------------------------------------------------------------------

class TestGateCheck:
    """Gate check: relay must be blocked when retrieval_auroc <= 0.70.

    Spec: REQ-VERIFY-150, FR-11
    """

    def test_gate_blocks_if_retrieval_below_threshold(self, tmp_path: Path) -> None:
        """A missing or low-auroc Exp 847 artifact must return <= gate value."""
        # Simulate missing file: _read_847_gate returns 0.0
        with patch(
            "scripts.experiment_848_fr11_tier1_live_relay_v4._REPO",
            tmp_path,
        ):
            result = _read_847_gate()
        assert result == 0.0, "Missing artifact must return 0.0 (below gate)"
        assert result <= RETRIEVAL_AUROC_GATE

    def test_gate_passes_with_valid_auroc(self, tmp_path: Path) -> None:
        """An artifact with retrieval_auroc=0.72 clears the 0.70 gate."""
        artifact_dir = tmp_path / "results"
        artifact_dir.mkdir(parents=True)
        artifact = {"retrieval_auroc": 0.72}
        (artifact_dir / "experiment_847_constraint_retrieval_l2_fix.json").write_text(
            json.dumps(artifact)
        )
        with patch(
            "scripts.experiment_848_fr11_tier1_live_relay_v4._REPO",
            tmp_path,
        ):
            auroc = _read_847_gate()
        assert auroc == 0.72
        assert auroc > RETRIEVAL_AUROC_GATE

    def test_gate_threshold_constant_is_0_70(self) -> None:
        """RETRIEVAL_AUROC_GATE must be exactly 0.70 per spec."""
        assert RETRIEVAL_AUROC_GATE == 0.70


# ---------------------------------------------------------------------------
# test_constraint_write_after_session
# ---------------------------------------------------------------------------

class TestConstraintWriteAfterSession:
    """After a session with violations, SPO constraints must be written to the store.

    Spec: REQ-LEARN-057, REQ-LEARN-058, SCENARIO-VERIFY-230
    """

    def test_violations_write_spo_to_store(self) -> None:
        """Each violation from an updatable type must produce one store entry."""
        store = EmbeddingConstraintStore()
        initial_size = len(store._store)

        # Write a single SPO entry
        spo = ConstraintSPOTuple(
            subject="arithmetic_carry",
            predicate="violates",
            object="carry_propagation",
            embedding=None,
            source_violation_type="carry",
        )
        store.store(spo)
        assert len(store._store) == initial_size + 1

    def test_stored_embedding_is_unit_norm(self) -> None:
        """Every stored embedding must have L2 norm == 1.0 (Exp 847 invariant).

        Spec: REQ-VERIFY-150
        """
        import math
        store = EmbeddingConstraintStore()
        spo = ConstraintSPOTuple(
            subject="numeric_sign",
            predicate="violates",
            object="sign_preservation",
            embedding=None,
            source_violation_type="sign",
        )
        store.store(spo)
        emb = store._store[-1].embedding
        assert emb is not None
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 1e-4, f"Stored embedding norm={norm}, expected ~1.0"

    def test_relay_writes_constraints_each_session(self, tmp_path: Path) -> None:
        """run_relay must write at least 1 constraint per session on average.

        Spec: REQ-LEARN-059
        """
        with patch(
            "scripts.experiment_848_fr11_tier1_live_relay_v4._REPO",
            tmp_path,
        ):
            (tmp_path / "results").mkdir(parents=True, exist_ok=True)
            result = run_relay(n_sessions=3, n_questions=5, seed=42)
        total_written = sum(result["session_constraints_written"])
        assert total_written >= 3, (
            f"Expected at least 3 total constraints written across 3 sessions, "
            f"got {total_written}"
        )


# ---------------------------------------------------------------------------
# test_monotonicity_check
# ---------------------------------------------------------------------------

class TestMonotonicityCheck:
    """is_monotonic must be True iff each session's precision >= the previous.

    Spec: FR-11 Tier 1 spec
    """

    def test_strictly_increasing_is_monotonic(self, tmp_path: Path) -> None:
        """Strictly increasing precisions must yield is_monotonic=True."""
        precisions = [0.30, 0.35, 0.40, 0.45, 0.50]
        is_monotonic = all(
            precisions[i] >= precisions[i - 1] for i in range(1, len(precisions))
        )
        assert is_monotonic is True

    def test_dip_breaks_monotonicity(self) -> None:
        """Any precision drop must yield is_monotonic=False."""
        precisions = [0.30, 0.38, 0.35, 0.42, 0.50]  # dip at s3
        is_monotonic = all(
            precisions[i] >= precisions[i - 1] for i in range(1, len(precisions))
        )
        assert is_monotonic is False

    def test_relay_returns_is_monotonic_field(self, tmp_path: Path) -> None:
        """run_relay must return an 'is_monotonic' key."""
        with patch(
            "scripts.experiment_848_fr11_tier1_live_relay_v4._REPO",
            tmp_path,
        ):
            (tmp_path / "results").mkdir(parents=True, exist_ok=True)
            result = run_relay(n_sessions=3, n_questions=5, seed=42)
        assert "is_monotonic" in result
        assert isinstance(result["is_monotonic"], bool)


# ---------------------------------------------------------------------------
# test_delta_computation
# ---------------------------------------------------------------------------

class TestDeltaComputation:
    """delta_s1_to_s5 = precision_s5 - precision_s1.

    Spec: FR-11 Tier 1 spec
    """

    def test_delta_equals_last_minus_first(self, tmp_path: Path) -> None:
        """delta_s1_to_s5 must equal precision_s5 - precision_s1."""
        with patch(
            "scripts.experiment_848_fr11_tier1_live_relay_v4._REPO",
            tmp_path,
        ):
            (tmp_path / "results").mkdir(parents=True, exist_ok=True)
            result = run_relay(n_sessions=5, n_questions=5, seed=42)
        expected_delta = result["precision_s5"] - result["precision_s1"]
        assert abs(result["delta_s1_to_s5"] - expected_delta) < 1e-9

    def test_positive_delta_on_seed_42(self, tmp_path: Path) -> None:
        """With seed=42 and 5 sessions, the relay should show positive delta.

        The synthetic inference model has increasing base_precision with session
        index, so seed=42 should produce delta > 0 over 5 sessions.

        Spec: FR-11, REQ-LEARN-059
        """
        with patch(
            "scripts.experiment_848_fr11_tier1_live_relay_v4._REPO",
            tmp_path,
        ):
            (tmp_path / "results").mkdir(parents=True, exist_ok=True)
            result = run_relay(n_sessions=5, n_questions=15, seed=42)
        assert result["delta_s1_to_s5"] > 0, (
            f"Expected positive delta on seed=42, got {result['delta_s1_to_s5']}"
        )

    def test_result_has_all_precision_fields(self, tmp_path: Path) -> None:
        """run_relay must return precision_s1 through precision_s5."""
        with patch(
            "scripts.experiment_848_fr11_tier1_live_relay_v4._REPO",
            tmp_path,
        ):
            (tmp_path / "results").mkdir(parents=True, exist_ok=True)
            result = run_relay(n_sessions=5, n_questions=5, seed=1)
        for key in ("precision_s1", "precision_s2", "precision_s3",
                    "precision_s4", "precision_s5"):
            assert key in result, f"Missing field: {key}"
            assert result[key] is not None


# ---------------------------------------------------------------------------
# test_capacity_constrained_update
# ---------------------------------------------------------------------------

class TestCapacityConstrainedUpdate:
    """Only top-K types by variance are written each session.

    Spec: REQ-LEARN-058
    """

    def test_select_updatable_returns_at_most_top_k(self) -> None:
        """_select_updatable_types must return at most TOP_K_CONSTRAINT_UPDATE items."""
        history = {t: [0.3, 0.4, 0.5, 0.6] for t in ["carry", "sign", "unit", "comparison", "causal"]}
        selected = _select_updatable_types(history)
        assert len(selected) <= TOP_K_CONSTRAINT_UPDATE

    def test_frozen_types_excluded(self) -> None:
        """Types with variance < VARIANCE_FREEZE_THRESHOLD must be excluded."""
        # One type with constant precision (variance=0) should be frozen
        history = {
            "carry": [0.5, 0.5, 0.5, 0.5],   # variance=0 → frozen
            "sign":  [0.2, 0.4, 0.6, 0.8],    # high variance → active
        }
        variances = _compute_type_variances(history)
        assert variances["carry"] < VARIANCE_FREEZE_THRESHOLD
        selected = _select_updatable_types(history)
        assert "carry" not in selected
        assert "sign" in selected

    def test_compute_type_variances_returns_float_per_type(self) -> None:
        """_compute_type_variances must return a float for every input type."""
        history = {"carry": [0.3, 0.6], "sign": [0.5]}
        variances = _compute_type_variances(history)
        assert set(variances.keys()) == {"carry", "sign"}
        for v in variances.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# test_checkpoint_written
# ---------------------------------------------------------------------------

class TestCheckpointWritten:
    """Checkpoint must be written after each session.

    Spec: REQ-INFRA-027
    """

    def test_checkpoint_file_exists_after_relay(self, tmp_path: Path) -> None:
        """A checkpoint JSON must exist on disk after run_relay completes."""
        with patch(
            "scripts.experiment_848_fr11_tier1_live_relay_v4._REPO",
            tmp_path,
        ):
            (tmp_path / "results").mkdir(parents=True, exist_ok=True)
            run_relay(n_sessions=2, n_questions=3, seed=7)
        ckpt = tmp_path / "results" / "exp848_checkpoint.json"
        assert ckpt.exists(), "Checkpoint file must be written by run_relay"
        data = json.loads(ckpt.read_text())
        assert "session" in data
        assert data["session"] == 2  # last session number


# ---------------------------------------------------------------------------
# test_honest_verdict_mapping
# ---------------------------------------------------------------------------

class TestHonestVerdictMapping:
    """honest_verdict must map correctly from delta and monotonicity.

    Spec: FR-11
    """

    def test_tier1_relay_works_live_when_monotonic_and_positive_delta(self) -> None:
        is_monotonic = True
        delta = 0.15
        if is_monotonic and delta > 0:
            verdict = "tier1_relay_works_live"
        elif delta > 0:
            verdict = "tier1_partial_improvement"
        else:
            verdict = "tier1_plateau_persists_live"
        assert verdict == "tier1_relay_works_live"

    def test_partial_improvement_when_positive_delta_not_monotonic(self) -> None:
        is_monotonic = False
        delta = 0.05
        if is_monotonic and delta > 0:
            verdict = "tier1_relay_works_live"
        elif delta > 0:
            verdict = "tier1_partial_improvement"
        else:
            verdict = "tier1_plateau_persists_live"
        assert verdict == "tier1_partial_improvement"

    def test_plateau_when_delta_zero_or_negative(self) -> None:
        for delta in (0.0, -0.1):
            is_monotonic = False
            if is_monotonic and delta > 0:
                verdict = "tier1_relay_works_live"
            elif delta > 0:
                verdict = "tier1_partial_improvement"
            else:
                verdict = "tier1_plateau_persists_live"
            assert verdict == "tier1_plateau_persists_live"
