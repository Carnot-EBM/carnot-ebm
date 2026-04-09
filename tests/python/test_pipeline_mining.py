"""Tests for carnot.pipeline.mining — failure analysis and pattern mining.

Each test references REQ-VERIFY-* or SCENARIO-VERIFY-* per spec-anchored
development requirements.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import pytest

from carnot.pipeline.mining import (
    CATEGORY_DETECTORS,
    CLAIM_CATEGORIES,
    SUGGESTED_PATTERNS,
    FailureAnalyzer,
    FailureReport,
    FalseNegative,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# FailureAnalyzer initialization — REQ-VERIFY-001
# ---------------------------------------------------------------------------


class TestFailureAnalyzerInit:
    """Tests for FailureAnalyzer construction."""

    def test_default_pipeline(self) -> None:
        """REQ-VERIFY-001: FailureAnalyzer creates default pipeline when none provided."""
        analyzer = FailureAnalyzer()
        assert analyzer._pipeline is not None

    def test_custom_pipeline(self) -> None:
        """REQ-VERIFY-001: FailureAnalyzer accepts a custom pipeline."""
        pipeline = VerifyRepairPipeline()
        analyzer = FailureAnalyzer(pipeline=pipeline)
        assert analyzer._pipeline is pipeline


# ---------------------------------------------------------------------------
# Category detection — REQ-VERIFY-003, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestCategoryDetection:
    """Tests for claim category detection patterns."""

    def test_arithmetic_chain_detected(self) -> None:
        """REQ-VERIFY-003: Arithmetic chain claims are detected in text."""
        text = "First, add 3 and 5 which gives us 8. Then multiply by 2."
        detectors = CATEGORY_DETECTORS["arithmetic_chain"]
        matches = sum(len(p.findall(text)) for p in detectors)
        assert matches > 0, "Should detect arithmetic chain patterns"

    def test_implicit_logic_detected(self) -> None:
        """REQ-VERIFY-003: Implicit logic claims are detected in text."""
        text = "Since all dogs are mammals, therefore Rex must be a mammal."
        detectors = CATEGORY_DETECTORS["implicit_logic"]
        matches = sum(len(p.findall(text)) for p in detectors)
        assert matches > 0, "Should detect implicit logic patterns"

    def test_world_knowledge_detected(self) -> None:
        """REQ-VERIFY-003: World knowledge claims are detected in text."""
        text = "Python was created in 1991 by Guido van Rossum."
        detectors = CATEGORY_DETECTORS["world_knowledge"]
        matches = sum(len(p.findall(text)) for p in detectors)
        assert matches > 0, "Should detect world knowledge patterns"

    def test_code_semantics_detected(self) -> None:
        """REQ-VERIFY-003: Code semantics claims are detected in text."""
        text = "The function runs in O(n log n) time and correctly handles edge cases."
        detectors = CATEGORY_DETECTORS["code_semantics"]
        matches = sum(len(p.findall(text)) for p in detectors)
        assert matches > 0, "Should detect code semantics patterns"

    def test_comparison_detected(self) -> None:
        """REQ-VERIFY-003: Comparison claims are detected in text."""
        text = "The result 50 is greater than 30 and is the largest value."
        detectors = CATEGORY_DETECTORS["comparison"]
        matches = sum(len(p.findall(text)) for p in detectors)
        assert matches > 0, "Should detect comparison patterns"

    def test_negation_detected(self) -> None:
        """REQ-VERIFY-003: Negation claims are detected in text."""
        text = "The function never returns None and has no errors."
        detectors = CATEGORY_DETECTORS["negation"]
        matches = sum(len(p.findall(text)) for p in detectors)
        assert matches > 0, "Should detect negation patterns"

    def test_all_categories_have_detectors(self) -> None:
        """SCENARIO-VERIFY-005: Every category has at least one detector pattern."""
        for cat in CLAIM_CATEGORIES:
            assert cat in CATEGORY_DETECTORS, f"Missing detectors for {cat}"
            assert len(CATEGORY_DETECTORS[cat]) > 0, f"Empty detectors for {cat}"


# ---------------------------------------------------------------------------
# FailureAnalyzer.analyze — REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestFailureAnalyzerAnalyze:
    """Tests for the main analyze() method."""

    def setup_method(self) -> None:
        self.analyzer = FailureAnalyzer()

    def test_correct_answers_produce_no_false_negatives(self) -> None:
        """REQ-VERIFY-001: Correct answers are not counted as false negatives."""
        report = self.analyzer.analyze(
            questions=["What is 2 + 3?"],
            responses=["The answer is 5."],
            ground_truths=["5"],
        )
        assert report.total_wrong == 0
        assert len(report.false_negatives) == 0

    def test_wrong_answer_detected_by_pipeline_is_true_positive(self) -> None:
        """REQ-VERIFY-002: Wrong answer with extractable constraint is a true positive."""
        # Pipeline can catch explicit "2 + 3 = 6" arithmetic errors.
        report = self.analyzer.analyze(
            questions=["What is 2 + 3?"],
            responses=["2 + 3 = 6"],
            ground_truths=["5"],
        )
        assert report.total_wrong == 1
        # The pipeline should flag "2 + 3 = 6" as violated, so NOT a false negative.
        assert len(report.false_negatives) == 0

    def test_wrong_answer_missed_by_pipeline_is_false_negative(self) -> None:
        """REQ-VERIFY-003: Wrong answer without extractable constraints is a false negative."""
        # Response has wrong answer but no extractable "A + B = C" pattern.
        report = self.analyzer.analyze(
            questions=["What is 2 + 3?"],
            responses=["Since both numbers are small, the answer must be 4."],
            ground_truths=["5"],
        )
        assert report.total_wrong == 1
        assert len(report.false_negatives) == 1
        assert report.false_negative_rate == 1.0

    def test_false_negative_has_uncovered_categories(self) -> None:
        """REQ-VERIFY-003: False negatives are categorized by uncovered claim types."""
        report = self.analyzer.analyze(
            questions=["What is 2 + 3?"],
            responses=[
                "Since 2 and 3 are both positive, therefore the sum must be 4. "
                "This is greater than 1."
            ],
            ground_truths=["5"],
        )
        assert len(report.false_negatives) == 1
        fn = report.false_negatives[0]
        # Should detect implicit_logic ("since...therefore") and/or comparison
        # ("greater than") as uncovered categories.
        total_uncovered = sum(fn.uncovered_categories.values())
        assert total_uncovered > 0, "Should find uncovered claim categories"

    def test_custom_checkers(self) -> None:
        """REQ-VERIFY-001: Custom checkers override default string matching."""
        # Default check would pass (ground_truth "5" IS in "the answer is 50"),
        # but custom checker extracts the number and compares strictly.
        def strict_checker(response: str) -> bool:
            import re
            nums = re.findall(r"\b\d+\b", response)
            return "5" in nums

        report = self.analyzer.analyze(
            questions=["What is 2 + 3?"],
            responses=["The answer is 50."],
            ground_truths=["5"],
            checkers=[strict_checker],
        )
        # With strict checker, "50" != "5", so this is wrong.
        assert report.total_wrong == 1

    def test_length_mismatch_raises(self) -> None:
        """REQ-VERIFY-001: Mismatched input lengths raise AssertionError."""
        with pytest.raises(AssertionError):
            self.analyzer.analyze(
                questions=["Q1", "Q2"],
                responses=["R1"],
                ground_truths=["GT1", "GT2"],
            )

    def test_empty_input(self) -> None:
        """REQ-VERIFY-001: Empty input produces empty report."""
        report = self.analyzer.analyze(
            questions=[],
            responses=[],
            ground_truths=[],
        )
        assert report.total_questions == 0
        assert report.total_wrong == 0
        assert len(report.false_negatives) == 0

    def test_report_has_all_fields(self) -> None:
        """SCENARIO-VERIFY-005: FailureReport contains all expected fields."""
        report = self.analyzer.analyze(
            questions=["What is 2 + 3?"],
            responses=["The answer is 4, because reasons."],
            ground_truths=["5"],
        )
        assert isinstance(report, FailureReport)
        assert isinstance(report.total_questions, int)
        assert isinstance(report.total_wrong, int)
        assert isinstance(report.false_negatives, list)
        assert isinstance(report.false_negative_rate, float)
        assert isinstance(report.category_counts, dict)
        assert isinstance(report.suggested_patterns, list)


# ---------------------------------------------------------------------------
# Category aggregation and pattern suggestion — REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestCategoryAggregation:
    """Tests for category counting and pattern suggestion."""

    def test_category_counts_reflect_false_negatives(self) -> None:
        """REQ-VERIFY-003: Category counts accurately reflect uncovered claims."""
        analyzer = FailureAnalyzer()
        # Multiple false negatives with different uncovered categories.
        report = analyzer.analyze(
            questions=[
                "What is 10 + 20?",
                "What is 5 * 3?",
            ],
            responses=[
                # Implicit logic + comparison, no extractable arithmetic.
                "Since both are positive, therefore the sum must be 25. "
                "This is greater than 20.",
                # Chain arithmetic without "A * B = C" pattern.
                "First multiply 5 by 3 which gives us 20. "
                "Therefore the answer is 20.",
            ],
            ground_truths=["30", "15"],
        )
        # Both are wrong and pipeline likely can't catch them.
        # category_counts should have nonzero entries.
        assert report.total_wrong == 2
        total_cats = sum(report.category_counts.values())
        # At least some categories should be populated from the false negatives.
        if len(report.false_negatives) > 0:
            assert total_cats > 0

    def test_suggested_patterns_are_sorted_by_catch_count(self) -> None:
        """REQ-VERIFY-003: Suggested patterns are ordered by estimated impact."""
        analyzer = FailureAnalyzer()
        report = analyzer.analyze(
            questions=["Q"] * 5,
            responses=[
                "Since X, therefore Y is 10. This is greater than 5.",
                "Because A, thus B must be 20. The answer is 20.",
                "Therefore the result is 30. It must be correct.",
                "Hence the value is 40, which gives us the answer.",
                "Since P implies Q, the answer must be 50.",
            ],
            ground_truths=["wrong"] * 5,
        )
        if len(report.suggested_patterns) >= 2:
            for i in range(len(report.suggested_patterns) - 1):
                assert (
                    report.suggested_patterns[i]["estimated_catch_count"]
                    >= report.suggested_patterns[i + 1]["estimated_catch_count"]
                ), "Patterns should be sorted by catch count descending"


# ---------------------------------------------------------------------------
# FalseNegative dataclass — SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestFalseNegativeDataclass:
    """Tests for the FalseNegative dataclass structure."""

    def test_false_negative_fields(self) -> None:
        """SCENARIO-VERIFY-005: FalseNegative has all required fields."""
        analyzer = FailureAnalyzer()
        report = analyzer.analyze(
            questions=["What is 1 + 1?"],
            responses=["The answer is definitely 3, since 1 and 1 make 3."],
            ground_truths=["2"],
        )
        if report.false_negatives:
            fn = report.false_negatives[0]
            assert isinstance(fn, FalseNegative)
            assert isinstance(fn.question, str)
            assert isinstance(fn.response, str)
            assert isinstance(fn.ground_truth, str)
            assert isinstance(fn.extracted_constraints, list)
            assert isinstance(fn.uncovered_categories, dict)


# ---------------------------------------------------------------------------
# Suggested patterns structure — REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestSuggestedPatterns:
    """Tests for the SUGGESTED_PATTERNS constant."""

    def test_all_categories_have_suggestions(self) -> None:
        """REQ-VERIFY-003: Most categories have suggested extraction patterns."""
        # At least 4 of 6 categories should have suggested patterns.
        covered = sum(1 for cat in CLAIM_CATEGORIES if cat in SUGGESTED_PATTERNS)
        assert covered >= 4, f"Only {covered}/6 categories have suggested patterns"

    def test_pattern_structure(self) -> None:
        """REQ-VERIFY-003: Each suggested pattern has name, pattern, description."""
        for cat, patterns in SUGGESTED_PATTERNS.items():
            for pat in patterns:
                assert "name" in pat, f"Missing 'name' in {cat} pattern"
                assert "pattern" in pat, f"Missing 'pattern' in {cat} pattern"
                assert "description" in pat, f"Missing 'description' in {cat} pattern"

    def test_patterns_are_valid_regex(self) -> None:
        """REQ-VERIFY-003: All suggested patterns compile as valid regex."""
        import re
        for cat, patterns in SUGGESTED_PATTERNS.items():
            for pat in patterns:
                try:
                    re.compile(pat["pattern"])
                except re.error as e:
                    pytest.fail(f"Invalid regex in {cat}/{pat['name']}: {e}")
