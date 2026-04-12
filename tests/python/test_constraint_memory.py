"""Tests for carnot.pipeline.memory -- ConstraintMemory (Tier 2).

Covers Tier 2 cross-session pattern learning:
- Recording patterns (domain, error_type, constraint description)
- Auto-promotion when frequency >= PATTERN_THRESHOLD
- Suggesting learned constraints before verification
- JSON persistence (save/load round-trip)
- Summary reporting
- Integration with VerifyRepairPipeline via memory= parameter

Spec: REQ-LEARN-003, SCENARIO-LEARN-003
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from carnot.pipeline.memory import PATTERN_THRESHOLD, ConstraintMemory
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# REQ-LEARN-003: record_pattern accumulates counts
# ---------------------------------------------------------------------------


class TestRecordPattern:
    """REQ-LEARN-003: record_pattern tracks (domain, error_type) frequency."""

    def test_first_record_creates_entry(self) -> None:
        """SCENARIO-LEARN-003: First record_pattern creates the pattern entry."""
        mem = ConstraintMemory()
        mem.record_pattern("arithmetic", "arithmetic", "47 + 28 = 74 (correct: 75)")
        summary = mem.summary()
        assert "arithmetic" in summary
        assert summary["arithmetic"]["total_patterns"] == 1

    def test_frequency_increments(self) -> None:
        """REQ-LEARN-003: Frequency counter increments on each call."""
        mem = ConstraintMemory()
        for i in range(5):
            mem.record_pattern("arithmetic", "arithmetic", f"expr_{i} = wrong")
        summary = mem.summary()
        # The top pattern should have frequency 5.
        top = summary["arithmetic"]["top_patterns"][0]
        assert top["error_type"] == "arithmetic"
        assert top["frequency"] == 5

    def test_multiple_error_types_per_domain(self) -> None:
        """REQ-LEARN-003: Multiple error types stored independently per domain."""
        mem = ConstraintMemory()
        mem.record_pattern("code", "initialization", "x used but not assigned")
        mem.record_pattern("code", "return_type", "f() returns str, expected int")
        mem.record_pattern("code", "initialization", "y used but not assigned")
        summary = mem.summary()
        assert summary["code"]["total_patterns"] == 2

    def test_multiple_domains_independent(self) -> None:
        """REQ-LEARN-003: Patterns in different domains are kept separate."""
        mem = ConstraintMemory()
        mem.record_pattern("arithmetic", "arithmetic", "3 + 4 = 8")
        mem.record_pattern("code", "initialization", "z is unset")
        assert len(mem.summary()) == 2

    def test_constraint_examples_stored(self) -> None:
        """REQ-LEARN-003: Distinct constraint descriptions are stored as examples."""
        mem = ConstraintMemory()
        mem.record_pattern("logic", "implication", "If A then B -- violated")
        mem.record_pattern("logic", "implication", "If X then Y -- violated")
        # Two distinct examples should be stored.
        suggestions = mem._patterns["logic"]["implication"].constraint_examples
        assert "If A then B -- violated" in suggestions
        assert "If X then Y -- violated" in suggestions

    def test_duplicate_description_not_stored_twice(self) -> None:
        """REQ-LEARN-003: Same description string is not stored as two examples."""
        mem = ConstraintMemory()
        desc = "47 + 28 = 74 (correct: 75)"
        mem.record_pattern("arithmetic", "arithmetic", desc)
        mem.record_pattern("arithmetic", "arithmetic", desc)
        examples = mem._patterns["arithmetic"]["arithmetic"].constraint_examples
        assert examples.count(desc) == 1
        # But frequency IS incremented twice.
        assert mem._patterns["arithmetic"]["arithmetic"].frequency == 2


# ---------------------------------------------------------------------------
# REQ-LEARN-003: auto-promotion at PATTERN_THRESHOLD
# ---------------------------------------------------------------------------


class TestAutoPromotion:
    """REQ-LEARN-003: Patterns are promoted when frequency >= PATTERN_THRESHOLD."""

    def test_below_threshold_not_promoted(self) -> None:
        """SCENARIO-LEARN-003: Pattern below threshold is not auto_generated."""
        mem = ConstraintMemory()
        for i in range(PATTERN_THRESHOLD - 1):
            mem.record_pattern("arithmetic", "arithmetic", f"err_{i}")
        assert not mem._patterns["arithmetic"]["arithmetic"].auto_generated

    def test_at_threshold_promoted(self) -> None:
        """SCENARIO-LEARN-003: Exactly at threshold triggers auto_generated=True."""
        mem = ConstraintMemory()
        for i in range(PATTERN_THRESHOLD):
            mem.record_pattern("arithmetic", "arithmetic", f"err_{i}")
        assert mem._patterns["arithmetic"]["arithmetic"].auto_generated

    def test_above_threshold_stays_promoted(self) -> None:
        """REQ-LEARN-003: Pattern stays promoted after exceeding threshold."""
        mem = ConstraintMemory()
        for i in range(PATTERN_THRESHOLD + 5):
            mem.record_pattern("code", "initialization", f"var_{i} unset")
        assert mem._patterns["code"]["initialization"].auto_generated


# ---------------------------------------------------------------------------
# REQ-LEARN-003: suggest_constraints returns learned constraints
# ---------------------------------------------------------------------------


class TestSuggestConstraints:
    """REQ-LEARN-003: suggest_constraints returns mature patterns only."""

    def _make_mature_memory(self, domain: str, error_type: str) -> ConstraintMemory:
        """Helper: create memory with one mature pattern."""
        mem = ConstraintMemory()
        for i in range(PATTERN_THRESHOLD):
            mem.record_pattern(domain, error_type, f"example_{i}")
        return mem

    def test_no_suggestions_for_unknown_domain(self) -> None:
        """SCENARIO-LEARN-003: Unknown domain returns empty list."""
        mem = ConstraintMemory()
        suggestions = mem.suggest_constraints("anything", "arithmetic")
        assert suggestions == []

    def test_no_suggestions_below_threshold(self) -> None:
        """SCENARIO-LEARN-003: Immature pattern yields no suggestions."""
        mem = ConstraintMemory()
        for i in range(PATTERN_THRESHOLD - 1):
            mem.record_pattern("arithmetic", "arithmetic", f"err_{i}")
        suggestions = mem.suggest_constraints("12 + 9 = 21", "arithmetic")
        assert suggestions == []

    def test_suggestion_returned_when_mature(self) -> None:
        """SCENARIO-LEARN-003: Mature pattern produces a ConstraintResult."""
        mem = self._make_mature_memory("arithmetic", "arithmetic")
        suggestions = mem.suggest_constraints("5 + 3 = 9", "arithmetic")
        assert len(suggestions) == 1
        cr = suggestions[0]
        assert cr.constraint_type == "learned"
        assert "arithmetic" in cr.description
        assert cr.metadata["domain"] == "arithmetic"
        assert cr.metadata["error_type"] == "arithmetic"
        assert cr.metadata["frequency"] >= PATTERN_THRESHOLD
        assert cr.metadata["auto_generated"] is True

    def test_suggestion_includes_examples(self) -> None:
        """REQ-LEARN-003: Suggestions carry the stored constraint examples."""
        mem = self._make_mature_memory("code", "initialization")
        suggestions = mem.suggest_constraints("def f(): pass", "code")
        assert len(suggestions) == 1
        examples = suggestions[0].metadata["examples"]
        assert len(examples) > 0

    def test_multiple_mature_patterns_all_returned(self) -> None:
        """REQ-LEARN-003: All mature patterns for a domain are returned."""
        mem = ConstraintMemory()
        for error_type in ["initialization", "return_type", "bound"]:
            for i in range(PATTERN_THRESHOLD):
                mem.record_pattern("code", error_type, f"{error_type}_{i}")
        suggestions = mem.suggest_constraints("code snippet", "code")
        returned_types = {s.metadata["error_type"] for s in suggestions}
        assert returned_types == {"initialization", "return_type", "bound"}

    def test_text_parameter_accepted_but_not_required(self) -> None:
        """REQ-LEARN-003: text parameter is accepted without error (forward-compat)."""
        mem = self._make_mature_memory("nl", "factual")
        # Should not raise even with arbitrary text.
        result = mem.suggest_constraints("Paris is the capital of France.", "nl")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Persistence: save / load round-trip
# ---------------------------------------------------------------------------


class TestPersistence:
    """REQ-LEARN-003: ConstraintMemory persists and loads correctly via JSON."""

    def test_save_creates_file(self) -> None:
        """SCENARIO-LEARN-003: save() creates a readable JSON file."""
        mem = ConstraintMemory()
        mem.record_pattern("arithmetic", "arithmetic", "1 + 1 = 3")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mem.save(path)
            assert os.path.exists(path)
            with open(path) as f:
                payload = json.load(f)
            assert payload["version"] == 1
            assert "arithmetic" in payload["patterns"]
        finally:
            os.unlink(path)

    def test_load_restores_patterns(self) -> None:
        """SCENARIO-LEARN-003: load() restores exact patterns from save()."""
        mem = ConstraintMemory()
        for i in range(PATTERN_THRESHOLD):
            mem.record_pattern("arithmetic", "arithmetic", f"carry_err_{i}")
        mem.record_pattern("code", "initialization", "x unset")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mem.save(path)
            restored = ConstraintMemory.load(path)

            # Pattern structure matches.
            assert "arithmetic" in restored._patterns
            assert "code" in restored._patterns
            arith = restored._patterns["arithmetic"]["arithmetic"]
            assert arith.frequency == PATTERN_THRESHOLD
            assert arith.auto_generated is True
            code = restored._patterns["code"]["initialization"]
            assert code.frequency == 1
            assert code.auto_generated is False
        finally:
            os.unlink(path)

    def test_load_invalid_version_raises(self) -> None:
        """REQ-LEARN-003: load() raises ValueError for bad file version."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"version": 99, "patterns": {}}, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="version"):
                ConstraintMemory.load(path)
        finally:
            os.unlink(path)

    def test_load_missing_version_raises(self) -> None:
        """REQ-LEARN-003: load() raises ValueError when version key is absent."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"patterns": {}}, f)
            path = f.name
        try:
            with pytest.raises(ValueError):
                ConstraintMemory.load(path)
        finally:
            os.unlink(path)

    def test_round_trip_suggestions_work(self) -> None:
        """REQ-LEARN-003: Suggestions work correctly after save/load cycle."""
        mem = ConstraintMemory()
        for i in range(PATTERN_THRESHOLD):
            mem.record_pattern("logic", "implication", f"if A then B #{i}")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mem.save(path)
            restored = ConstraintMemory.load(path)
            suggestions = restored.suggest_constraints("test", "logic")
            assert len(suggestions) == 1
            assert suggestions[0].constraint_type == "learned"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    """REQ-LEARN-003: summary() returns accurate per-domain statistics."""

    def test_empty_summary(self) -> None:
        """SCENARIO-LEARN-003: Empty memory returns empty summary dict."""
        mem = ConstraintMemory()
        assert mem.summary() == {}

    def test_summary_counts(self) -> None:
        """REQ-LEARN-003: summary reports total and mature pattern counts."""
        mem = ConstraintMemory()
        # One mature, one immature pattern in "arithmetic".
        for i in range(PATTERN_THRESHOLD):
            mem.record_pattern("arithmetic", "arithmetic", f"e{i}")
        mem.record_pattern("arithmetic", "carry", "carry error once")  # immature

        s = mem.summary()
        assert s["arithmetic"]["total_patterns"] == 2
        assert s["arithmetic"]["mature_patterns"] == 1

    def test_summary_top_patterns_ordering(self) -> None:
        """REQ-LEARN-003: top_patterns sorted by frequency descending."""
        mem = ConstraintMemory()
        # error_a: 5 times, error_b: 2 times
        for i in range(5):
            mem.record_pattern("code", "error_a", f"e{i}")
        for i in range(2):
            mem.record_pattern("code", "error_b", f"b{i}")

        top = mem.summary()["code"]["top_patterns"]
        assert top[0]["error_type"] == "error_a"
        assert top[0]["frequency"] == 5
        assert top[1]["error_type"] == "error_b"
        assert top[1]["frequency"] == 2

    def test_summary_top_patterns_capped_at_five(self) -> None:
        """REQ-LEARN-003: top_patterns returns at most 5 entries."""
        mem = ConstraintMemory()
        for i in range(10):
            mem.record_pattern("code", f"error_{i}", "desc")
        top = mem.summary()["code"]["top_patterns"]
        assert len(top) <= 5


# ---------------------------------------------------------------------------
# VerifyRepairPipeline integration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """REQ-LEARN-003: VerifyRepairPipeline memory= parameter integrates correctly."""

    def test_pipeline_accepts_memory_none(self) -> None:
        """SCENARIO-LEARN-003: memory=None (default) does not break pipeline."""
        pipeline = VerifyRepairPipeline(domains=["arithmetic"])
        result = pipeline.verify("q", "2 + 3 = 5")
        assert result.verified  # correct arithmetic

    def test_pipeline_accepts_memory_instance(self) -> None:
        """SCENARIO-LEARN-003: Pipeline constructed with ConstraintMemory."""
        mem = ConstraintMemory()
        pipeline = VerifyRepairPipeline(domains=["arithmetic"], memory=mem)
        result = pipeline.verify("q", "2 + 3 = 5")
        assert result.verified

    def test_pipeline_records_violations_into_memory(self) -> None:
        """REQ-LEARN-003: After a violation, memory accumulates the pattern."""
        mem = ConstraintMemory()
        pipeline = VerifyRepairPipeline(domains=["arithmetic"], memory=mem)
        # Incorrect arithmetic: 47 + 28 = 74 (correct is 75).
        pipeline.verify("q", "47 + 28 = 74")
        # Memory should have recorded the arithmetic pattern.
        summary = mem.summary()
        assert "arithmetic" in summary
        assert summary["arithmetic"]["total_patterns"] >= 1

    def test_pipeline_memory_promotes_after_threshold(self) -> None:
        """REQ-LEARN-003: After PATTERN_THRESHOLD violations, memory promotes pattern."""
        mem = ConstraintMemory()
        pipeline = VerifyRepairPipeline(domains=["arithmetic"], memory=mem)
        # Trigger arithmetic violations PATTERN_THRESHOLD times.
        bad_expressions = [
            "10 + 5 = 14",
            "20 + 3 = 22",
            "7 + 8 = 14",
        ]
        # Ensure we have at least PATTERN_THRESHOLD distinct violations.
        assert len(bad_expressions) >= PATTERN_THRESHOLD
        for expr in bad_expressions[:PATTERN_THRESHOLD]:
            pipeline.verify("q", expr)
        summary = mem.summary()
        assert summary["arithmetic"]["mature_patterns"] >= 1

    def test_pipeline_memory_suggestions_prepended(self) -> None:
        """REQ-LEARN-003: Learned constraints appear in verification constraints list."""
        mem = ConstraintMemory()
        # Manually inject a mature pattern so we don't need to loop.
        for i in range(PATTERN_THRESHOLD):
            mem.record_pattern("arithmetic", "arithmetic", f"carry_error_{i}")

        pipeline = VerifyRepairPipeline(domains=["arithmetic"], memory=mem)
        result = pipeline.verify("q", "5 + 3 = 8")  # correct arithmetic
        # Learned constraints should appear in the constraint list.
        constraint_types = [c.constraint_type for c in result.constraints]
        assert "learned" in constraint_types

    def test_pipeline_no_memory_no_learned_constraints(self) -> None:
        """REQ-LEARN-003: Without memory, no 'learned' constraints appear."""
        pipeline = VerifyRepairPipeline(domains=["arithmetic"])
        result = pipeline.verify("q", "5 + 3 = 8")
        constraint_types = [c.constraint_type for c in result.constraints]
        assert "learned" not in constraint_types

    def test_memory_stores_violation_description(self) -> None:
        """REQ-LEARN-003: Memory stores the actual violation description."""
        mem = ConstraintMemory()
        pipeline = VerifyRepairPipeline(domains=["arithmetic"], memory=mem)
        pipeline.verify("q", "3 + 4 = 8")
        # The pattern for 'arithmetic' domain should have an example.
        arith_patterns = mem._patterns.get("arithmetic", {})
        assert len(arith_patterns) > 0
        any_example = any(
            len(rec.constraint_examples) > 0
            for rec in arith_patterns.values()
        )
        assert any_example


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    """REQ-LEARN-003: ConstraintMemory has a useful repr."""

    def test_repr_empty(self) -> None:
        mem = ConstraintMemory()
        assert "domains=0" in repr(mem)
        assert "patterns=0" in repr(mem)

    # REQ-LEARN-003: repr reflects tracked domains and patterns after updates
    def test_repr_with_patterns(self) -> None:
        mem = ConstraintMemory()
        mem.record_pattern("arithmetic", "arithmetic", "err")
        mem.record_pattern("code", "init", "var missing")
        r = repr(mem)
        assert "domains=2" in r
        assert "patterns=2" in r
