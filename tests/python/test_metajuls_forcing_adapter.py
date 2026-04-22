"""Tests for MetaJuLSForcingAdapter and ForcingFeedback.

Every test traces to a specific REQ-* or SCENARIO-* from the autoresearch spec.

Spec coverage:
    REQ-LEARN-085  — adapter updates forcing strategy from live feedback
    REQ-LEARN-086  — adapted strategy improves COMPUTE: recall vs static forcer
    SCENARIO-LEARN-133 — low recall domain triggers CRITICAL: emphasis
    SCENARIO-LEARN-134 — high recall domain keeps base addendum unchanged
    SCENARIO-LEARN-135 — save/load round-trip preserves state
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from carnot.pipeline.metajuls_forcing_adapter import (
    ForcingFeedback,
    LOW_RECALL_THRESHOLD,
    MetaJuLSForcingAdapter,
)

# Base addendum from the real forcer — use a compact version to keep tests readable.
_BASE = "IMPORTANT: Write every arithmetic step as COMPUTE: X op Y = result."


# ---------------------------------------------------------------------------
# ForcingFeedback dataclass tests
# ---------------------------------------------------------------------------


class TestForcingFeedback:
    """REQ-LEARN-085: ForcingFeedback captures all fields needed for adaptation."""

    def test_fields_present(self):
        fb = ForcingFeedback(
            question="What is 2+2?",
            compute_lines_found=1,
            total_arithmetic_ops=1,
            recall=1.0,
            domain="arithmetic",
        )
        assert fb.question == "What is 2+2?"
        assert fb.compute_lines_found == 1
        assert fb.total_arithmetic_ops == 1
        assert fb.recall == 1.0
        assert fb.domain == "arithmetic"

    def test_recall_zero_allowed(self):
        # SCENARIO-LEARN-133: zero recall is valid input (model wrote no COMPUTE: lines)
        fb = ForcingFeedback(
            question="What is 15% of 100?",
            compute_lines_found=0,
            total_arithmetic_ops=2,
            recall=0.0,
            domain="percentage",
        )
        assert fb.recall == 0.0


# ---------------------------------------------------------------------------
# MetaJuLSForcingAdapter — initialisation
# ---------------------------------------------------------------------------


class TestAdapterInit:
    """REQ-LEARN-085: adapter initialises with correct defaults."""

    def test_base_addendum_stored(self):
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        assert adapter._base_addendum == _BASE

    def test_learning_rate_default(self):
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        assert adapter._learning_rate == 0.1

    def test_custom_learning_rate(self):
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE, learning_rate=0.05)
        assert adapter._learning_rate == 0.05

    def test_no_domain_recalls_initially(self):
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        assert adapter.domain_recalls == {}

    def test_no_domain_emphasis_initially(self):
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        assert adapter.domain_emphasis == {}


# ---------------------------------------------------------------------------
# update — low recall path (SCENARIO-LEARN-133)
# ---------------------------------------------------------------------------


class TestUpdateLowRecall:
    """SCENARIO-LEARN-133: low mean recall triggers CRITICAL: emphasis installation."""

    def _low_recall_fb(self, recall: float = 0.10, domain: str = "percentage") -> ForcingFeedback:
        return ForcingFeedback(
            question="What is 15% of 200?",
            compute_lines_found=int(recall * 5),
            total_arithmetic_ops=5,
            recall=recall,
            domain=domain,
        )

    def test_emphasis_installed_below_threshold(self):
        # REQ-LEARN-085: when mean recall < 0.30, domain emphasis must be set.
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        adapter.update(self._low_recall_fb(recall=0.10))
        assert "percentage" in adapter.domain_emphasis

    def test_emphasis_contains_critical(self):
        # SCENARIO-LEARN-133: the emphasis string escalates to CRITICAL wording.
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        adapter.update(self._low_recall_fb(recall=0.10))
        assert "CRITICAL" in adapter.domain_emphasis["percentage"]

    def test_emphasis_contains_domain_name(self):
        # SCENARIO-LEARN-133: domain name appears in emphasis for specificity.
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        adapter.update(self._low_recall_fb(recall=0.10, domain="multi_step"))
        assert "multi_step" in adapter.domain_emphasis["multi_step"]

    def test_recall_recorded(self):
        # REQ-LEARN-085: the recall value is stored in domain_recalls.
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        adapter.update(self._low_recall_fb(recall=0.20))
        assert 0.20 in adapter.domain_recalls["percentage"]

    def test_multiple_low_recalls_accumulate(self):
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        for recall in [0.10, 0.15, 0.20]:
            adapter.update(self._low_recall_fb(recall=recall))
        assert len(adapter.domain_recalls["percentage"]) == 3

    def test_emphasis_stays_installed_on_subsequent_low_recalls(self):
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        adapter.update(self._low_recall_fb(recall=0.10))
        adapter.update(self._low_recall_fb(recall=0.15))
        assert "percentage" in adapter.domain_emphasis


# ---------------------------------------------------------------------------
# update — high recall path (SCENARIO-LEARN-134)
# ---------------------------------------------------------------------------


class TestUpdateHighRecall:
    """SCENARIO-LEARN-134: adequate mean recall keeps base addendum unchanged."""

    def _high_recall_fb(self, domain: str = "arithmetic") -> ForcingFeedback:
        return ForcingFeedback(
            question="What is 47 + 28?",
            compute_lines_found=1,
            total_arithmetic_ops=1,
            recall=1.0,
            domain=domain,
        )

    def test_no_emphasis_installed_for_high_recall(self):
        # SCENARIO-LEARN-134: domains above threshold must not get CRITICAL: emphasis.
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        adapter.update(self._high_recall_fb())
        assert "arithmetic" not in adapter.domain_emphasis

    def test_emphasis_removed_when_domain_recovers(self):
        # If a previously low-recall domain recovers, emphasis must be removed.
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        # Drive mean below threshold with many low recalls.
        for _ in range(5):
            adapter.update(
                ForcingFeedback(
                    question="q",
                    compute_lines_found=0,
                    total_arithmetic_ops=1,
                    recall=0.10,
                    domain="arithmetic",
                )
            )
        assert "arithmetic" in adapter.domain_emphasis

        # Now flood with high recalls to push mean above threshold.
        for _ in range(50):
            adapter.update(self._high_recall_fb())
        assert "arithmetic" not in adapter.domain_emphasis

    def test_recall_still_recorded_for_high_recall(self):
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        adapter.update(self._high_recall_fb())
        assert len(adapter.domain_recalls["arithmetic"]) == 1


# ---------------------------------------------------------------------------
# get_adapted_addendum (REQ-LEARN-086)
# ---------------------------------------------------------------------------


class TestGetAdaptedAddendum:
    """REQ-LEARN-086: adapted addendum improves recall by extending base addendum."""

    def test_returns_base_when_no_emphasis(self):
        # No adaptation → caller receives the original addendum unchanged.
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        result = adapter.get_adapted_addendum(question="q", domain="arithmetic")
        assert result == _BASE

    def test_returns_base_when_domain_is_none(self):
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        result = adapter.get_adapted_addendum(question="q", domain=None)
        assert result == _BASE

    def test_returns_extended_addendum_for_low_recall_domain(self):
        # SCENARIO-LEARN-133: adapted addendum is longer than base addendum.
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        adapter.update(
            ForcingFeedback(
                question="15% of 200?",
                compute_lines_found=0,
                total_arithmetic_ops=2,
                recall=0.0,
                domain="percentage",
            )
        )
        result = adapter.get_adapted_addendum(question="q", domain="percentage")
        assert len(result) > len(_BASE)

    def test_extended_addendum_contains_base(self):
        # REQ-LEARN-086: base instructions are preserved in the extended addendum.
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        adapter.update(
            ForcingFeedback(
                question="q", compute_lines_found=0, total_arithmetic_ops=1,
                recall=0.10, domain="percentage"
            )
        )
        result = adapter.get_adapted_addendum(question="q", domain="percentage")
        assert _BASE in result

    def test_unknown_domain_returns_base(self):
        # A domain that has never been observed → no emphasis → base returned.
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        result = adapter.get_adapted_addendum(question="q", domain="unknown_domain")
        assert result == _BASE

    def test_question_arg_unused_in_rule_based_policy(self):
        # Question text does not affect output in the current rule-based policy.
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        r1 = adapter.get_adapted_addendum(question="question one", domain="arithmetic")
        r2 = adapter.get_adapted_addendum(question="question two", domain="arithmetic")
        assert r1 == r2


# ---------------------------------------------------------------------------
# save_state / load_state (SCENARIO-LEARN-135)
# ---------------------------------------------------------------------------


class TestSaveLoadState:
    """SCENARIO-LEARN-135: save/load round-trip preserves all adapter state."""

    def _adapter_with_data(self) -> MetaJuLSForcingAdapter:
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE, learning_rate=0.05)
        adapter.update(
            ForcingFeedback(
                question="q", compute_lines_found=0, total_arithmetic_ops=2,
                recall=0.10, domain="percentage"
            )
        )
        adapter.update(
            ForcingFeedback(
                question="q", compute_lines_found=1, total_arithmetic_ops=1,
                recall=1.0, domain="arithmetic"
            )
        )
        return adapter

    def test_save_creates_file(self):
        adapter = self._adapter_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "state.json")
            adapter.save_state(path)
            assert Path(path).exists()

    def test_saved_file_is_valid_json(self):
        adapter = self._adapter_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "state.json")
            adapter.save_state(path)
            state = json.loads(Path(path).read_text())
            assert "base_addendum" in state
            assert "domain_recalls" in state
            assert "domain_emphasis" in state

    def test_load_restores_base_addendum(self):
        adapter = self._adapter_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "state.json")
            adapter.save_state(path)
            restored = MetaJuLSForcingAdapter.load_state(path)
            assert restored._base_addendum == _BASE

    def test_load_restores_learning_rate(self):
        adapter = self._adapter_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "state.json")
            adapter.save_state(path)
            restored = MetaJuLSForcingAdapter.load_state(path)
            assert restored._learning_rate == 0.05

    def test_load_restores_domain_recalls(self):
        # SCENARIO-LEARN-135: accumulated recall history survives save/load.
        adapter = self._adapter_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "state.json")
            adapter.save_state(path)
            restored = MetaJuLSForcingAdapter.load_state(path)
            assert "percentage" in restored.domain_recalls
            assert 0.10 in restored.domain_recalls["percentage"]

    def test_load_restores_domain_emphasis(self):
        # SCENARIO-LEARN-135: installed emphasis strings survive save/load.
        adapter = self._adapter_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "state.json")
            adapter.save_state(path)
            restored = MetaJuLSForcingAdapter.load_state(path)
            assert "percentage" in restored.domain_emphasis

    def test_load_restores_functional_adapter(self):
        # Restored adapter must produce the same addendum as before saving.
        adapter = self._adapter_with_data()
        original_addendum = adapter.get_adapted_addendum(question="q", domain="percentage")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "state.json")
            adapter.save_state(path)
            restored = MetaJuLSForcingAdapter.load_state(path)
            restored_addendum = restored.get_adapted_addendum(question="q", domain="percentage")
        assert original_addendum == restored_addendum


# ---------------------------------------------------------------------------
# LOW_RECALL_THRESHOLD constant
# ---------------------------------------------------------------------------


class TestLowRecallThreshold:
    """Sanity checks for the module-level threshold constant."""

    def test_threshold_is_float(self):
        assert isinstance(LOW_RECALL_THRESHOLD, float)

    def test_threshold_value(self):
        # 0.30 is the boundary below which CRITICAL: emphasis is triggered.
        assert LOW_RECALL_THRESHOLD == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# domain_recalls / domain_emphasis properties
# ---------------------------------------------------------------------------


class TestProperties:
    """domain_recalls and domain_emphasis properties return independent copies."""

    def test_domain_recalls_property_returns_copy(self):
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        adapter.update(
            ForcingFeedback(question="q", compute_lines_found=1, total_arithmetic_ops=1,
                            recall=1.0, domain="arithmetic")
        )
        copy = adapter.domain_recalls
        copy["arithmetic"].append(999.0)
        # Internal state must be unaffected by modifying the returned copy.
        assert 999.0 not in adapter._domain_recalls["arithmetic"]

    def test_domain_emphasis_property_returns_copy(self):
        adapter = MetaJuLSForcingAdapter(base_addendum=_BASE)
        adapter.update(
            ForcingFeedback(question="q", compute_lines_found=0, total_arithmetic_ops=1,
                            recall=0.0, domain="arithmetic")
        )
        copy = adapter.domain_emphasis
        copy["injected"] = "hack"
        assert "injected" not in adapter._domain_emphasis
