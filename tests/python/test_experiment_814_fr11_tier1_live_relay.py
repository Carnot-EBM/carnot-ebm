"""Tests for Exp 814: FR-11 Tier 1 Live Relay — capacity-constrained update.

Covers:
- selective_update only updates specified constraint types  (REQ-LEARN-814-001)
- compute_precision handles tp=fp=0 case                   (REQ-LEARN-814-001)
- check_monotonic correctly identifies monotone sequences   (REQ-LEARN-814-001)
- map_honest_verdict maps correctly to each verdict string  (REQ-LEARN-814-001)
- Gate blocks when Exp 813 delta_overall is None or <= 0   (REQ-LEARN-814-001)

Spec: REQ-LEARN-814-001, SCENARIO-LEARN-814-001
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.experiment_814_fr11_tier1_live_relay import (
    compute_precision,
    selective_update,
    check_monotonic,
    map_honest_verdict,
    _load_exp813_gate,
)
from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore


# ---------------------------------------------------------------------------
# selective_update: only updates specified constraint types
# ---------------------------------------------------------------------------


class TestSelectiveUpdate:
    """REQ-LEARN-814-001: selective_update MUST only update allowed constraint types."""

    def _make_store(self) -> EmbeddingConstraintStore:
        store = EmbeddingConstraintStore()
        store.from_casememory_patterns({"carry": 1, "sign": 1})
        return store

    def test_only_allowed_types_are_added(self) -> None:
        """selective_update adds SPO entries only for types in update_types.

        Spec: REQ-LEARN-814-001
        """
        store = self._make_store()
        initial_count = len(store._store)
        # Violations include "carry" and "sign"; only allow "carry"
        added = selective_update(store, ["carry", "sign", "carry"], update_types=["carry"])
        assert added == 2  # two "carry" violations added
        assert len(store._store) == initial_count + 2

    def test_excluded_types_are_frozen(self) -> None:
        """Types NOT in update_types must not be added to the store.

        Spec: REQ-LEARN-814-001
        """
        store = self._make_store()
        initial_count = len(store._store)
        # violations contains "sign" but update_types does NOT include it
        added = selective_update(store, ["sign", "sign"], update_types=["carry"])
        assert added == 0
        assert len(store._store) == initial_count

    def test_empty_violations_adds_nothing(self) -> None:
        """selective_update with no violations returns 0 and leaves store unchanged.

        Spec: REQ-LEARN-814-001
        """
        store = self._make_store()
        initial_count = len(store._store)
        added = selective_update(store, [], update_types=["carry", "sign"])
        assert added == 0
        assert len(store._store) == initial_count

    def test_unknown_violation_type_is_ignored(self) -> None:
        """Violation type not in _SPO_MAP is silently skipped.

        Spec: REQ-LEARN-814-001
        """
        store = self._make_store()
        initial_count = len(store._store)
        added = selective_update(store, ["unknown_type"], update_types=["unknown_type"])
        assert added == 0
        assert len(store._store) == initial_count


# ---------------------------------------------------------------------------
# compute_precision: tp=fp=0 edge case
# ---------------------------------------------------------------------------


class TestComputePrecision:
    """REQ-LEARN-814-001: compute_precision must handle edge cases correctly."""

    def test_zero_tp_zero_fp_returns_one(self) -> None:
        """When tp=0 and fp=0, precision must be 1.0 (vacuously perfect).

        Spec: REQ-LEARN-814-001
        """
        assert compute_precision(0, 0) == 1.0

    def test_normal_precision(self) -> None:
        """Standard precision computation: tp/(tp+fp).

        Spec: REQ-LEARN-814-001
        """
        assert compute_precision(6, 4) == pytest.approx(0.6)

    def test_perfect_precision(self) -> None:
        """When fp=0, precision is 1.0.

        Spec: REQ-LEARN-814-001
        """
        assert compute_precision(8, 0) == pytest.approx(1.0)

    def test_zero_precision(self) -> None:
        """When tp=0 and fp>0, precision is 0.0.

        Spec: REQ-LEARN-814-001
        """
        assert compute_precision(0, 5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# check_monotonic: monotone sequence detection
# ---------------------------------------------------------------------------


class TestCheckMonotonic:
    """REQ-LEARN-814-001: check_monotonic must correctly identify non-decreasing sequences."""

    def test_strictly_increasing_is_monotonic(self) -> None:
        """A strictly increasing sequence is monotonically non-decreasing.

        Spec: REQ-LEARN-814-001
        """
        assert check_monotonic([0.5, 0.6, 0.7, 0.8, 0.9]) is True

    def test_flat_sequence_is_monotonic(self) -> None:
        """A constant sequence is monotonically non-decreasing.

        Spec: REQ-LEARN-814-001
        """
        assert check_monotonic([0.6, 0.6, 0.6, 0.6, 0.6]) is True

    def test_one_dip_breaks_monotonic(self) -> None:
        """A sequence with any decrease is NOT monotonically non-decreasing.

        Spec: REQ-LEARN-814-001
        """
        assert check_monotonic([0.5, 0.7, 0.6, 0.8, 0.9]) is False

    def test_single_element_is_monotonic(self) -> None:
        """A single-element sequence is trivially monotonic.

        Spec: REQ-LEARN-814-001
        """
        assert check_monotonic([0.7]) is True


# ---------------------------------------------------------------------------
# map_honest_verdict: verdict string mapping
# ---------------------------------------------------------------------------


class TestMapHonestVerdict:
    """REQ-LEARN-814-001: map_honest_verdict must map statistics to correct verdict strings."""

    def test_monotonic_and_positive_by_s3_gives_relay_works(self) -> None:
        """Monotonic non-decrease AND positive by session 3 → tier1_relay_works_live.

        Spec: SCENARIO-LEARN-814-001
        """
        verdict = map_honest_verdict(
            is_monotonically_non_decreasing=True,
            delta_positive_by_s3=True,
            delta_s1_to_s5=0.1,
        )
        assert verdict == "tier1_relay_works_live"

    def test_positive_by_s3_but_not_monotonic_gives_partial(self) -> None:
        """Positive by session 3 but not monotonic → tier1_partial_improvement_live.

        Spec: SCENARIO-LEARN-814-001
        """
        verdict = map_honest_verdict(
            is_monotonically_non_decreasing=False,
            delta_positive_by_s3=True,
            delta_s1_to_s5=0.05,
        )
        assert verdict == "tier1_partial_improvement_live"

    def test_no_improvement_gives_plateau_persists(self) -> None:
        """No improvement from session 1 to 5 → tier1_plateau_persists_live.

        Spec: SCENARIO-LEARN-814-001
        """
        verdict = map_honest_verdict(
            is_monotonically_non_decreasing=False,
            delta_positive_by_s3=False,
            delta_s1_to_s5=0.0,
        )
        assert verdict == "tier1_plateau_persists_live"

    def test_negative_delta_gives_plateau_persists(self) -> None:
        """Negative delta → tier1_plateau_persists_live (precision regressed).

        Spec: SCENARIO-LEARN-814-001
        """
        verdict = map_honest_verdict(
            is_monotonically_non_decreasing=False,
            delta_positive_by_s3=False,
            delta_s1_to_s5=-0.05,
        )
        assert verdict == "tier1_plateau_persists_live"


# ---------------------------------------------------------------------------
# Gate: blocks when Exp 813 delta_overall is None or <= 0
# ---------------------------------------------------------------------------


class TestExp813Gate:
    """REQ-LEARN-814-001: Gate MUST block when Exp 813 delta_overall is None or <= 0."""

    def _make_tmpl(self) -> MagicMock:
        tmpl = MagicMock()
        tmpl.build_result.side_effect = lambda *args, **kwargs: {
            "status": kwargs.get("status", "blocked"),
            **kwargs,
        }
        return tmpl

    def test_gate_blocks_when_file_missing(self, tmp_path: Path) -> None:
        """Gate returns blocked artifact when Exp 813 result file does not exist.

        Spec: REQ-LEARN-814-001
        """
        missing = tmp_path / "no_such_file.json"
        tmpl = self._make_tmpl()
        with patch("scripts.experiment_814_fr11_tier1_live_relay.EXP_813_PATH", missing):
            result = _load_exp813_gate(tmpl)
        assert result is not None
        assert result["honest_verdict"] == "blocked_no_delta"
        assert result["status"] == "blocked"

    def test_gate_blocks_when_delta_is_none(self, tmp_path: Path) -> None:
        """Gate blocks when delta_overall is None (Exp 813 was itself blocked).

        Spec: REQ-LEARN-814-001
        """
        exp813 = {"delta_overall": None, "honest_verdict": "injection_not_wired"}
        p = tmp_path / "experiment_813.json"
        p.write_text(json.dumps(exp813))
        tmpl = self._make_tmpl()
        with patch("scripts.experiment_814_fr11_tier1_live_relay.EXP_813_PATH", p):
            result = _load_exp813_gate(tmpl)
        assert result is not None
        assert result["honest_verdict"] == "blocked_no_delta"

    def test_gate_blocks_when_delta_is_zero(self, tmp_path: Path) -> None:
        """Gate blocks when delta_overall is exactly 0.0.

        Spec: REQ-LEARN-814-001
        """
        exp813 = {"delta_overall": 0.0, "honest_verdict": "constraint_addition_no_delta_live"}
        p = tmp_path / "experiment_813.json"
        p.write_text(json.dumps(exp813))
        tmpl = self._make_tmpl()
        with patch("scripts.experiment_814_fr11_tier1_live_relay.EXP_813_PATH", p):
            result = _load_exp813_gate(tmpl)
        assert result is not None
        assert result["honest_verdict"] == "blocked_no_delta"

    def test_gate_passes_when_delta_positive(self, tmp_path: Path) -> None:
        """Gate returns None (pass) when delta_overall > 0.

        Spec: REQ-LEARN-814-001
        """
        exp813 = {"delta_overall": 0.083, "honest_verdict": "constraint_addition_works_live"}
        p = tmp_path / "experiment_813.json"
        p.write_text(json.dumps(exp813))
        tmpl = self._make_tmpl()
        with patch("scripts.experiment_814_fr11_tier1_live_relay.EXP_813_PATH", p):
            result = _load_exp813_gate(tmpl)
        assert result is None  # None means gate passes — proceed
