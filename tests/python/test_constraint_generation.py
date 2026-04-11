"""Tests for Exp 173 improvements to NegationConstraint v2 and CarryChainConstraint v2.

Covers:
- NegationConstraint v2: satisfied/violated detection for "is not", "not all", "no A are B"
- CarryChainConstraint v2: subtraction borrowing, digit count check, negative result detection
- _count_borrows helper
- _digit_count helper
- Regression: Exp 141 dedup fix still intact (no extra constraints of static types)

Spec: REQ-LEARN-003, REQ-LEARN-004, SCENARIO-LEARN-003
"""

from __future__ import annotations

import pytest

from carnot.pipeline.generation import (
    PATTERN_ARITHMETIC_CARRY,
    PATTERN_NEGATION_SCOPE,
    CarryChainConstraint,
    NegationConstraint,
    _count_borrows,
    _digit_count,
)


# ---------------------------------------------------------------------------
# _count_borrows helper tests
# REQ-LEARN-004: subtraction borrow detection
# ---------------------------------------------------------------------------


class TestCountBorrows:
    """REQ-LEARN-004: _count_borrows counts cascading borrow operations."""

    def test_no_borrow(self) -> None:
        """SCENARIO-LEARN-003: 5 - 3 has no borrow."""
        assert _count_borrows(5, 3) == 0

    def test_single_borrow(self) -> None:
        """REQ-LEARN-004: 10 - 1 has one borrow (units digit 0 < 1)."""
        assert _count_borrows(10, 1) == 1

    def test_two_borrows(self) -> None:
        """REQ-LEARN-004: 100 - 1 = 99 has two cascading borrows."""
        assert _count_borrows(100, 1) == 2

    def test_three_borrows(self) -> None:
        """REQ-LEARN-004: 1000 - 1 = 999 has three cascading borrows."""
        assert _count_borrows(1000, 1) == 3

    def test_four_borrows(self) -> None:
        """REQ-LEARN-004: 10000 - 1 has four cascading borrows."""
        assert _count_borrows(10000, 1) == 4

    def test_zero_subtrahend(self) -> None:
        """REQ-LEARN-004: Subtracting zero has no borrow."""
        assert _count_borrows(999, 0) == 0

    def test_equal_values(self) -> None:
        """REQ-LEARN-004: 5 - 5 has no borrow (result is 0)."""
        assert _count_borrows(5, 5) == 0

    def test_negative_values_use_abs(self) -> None:
        """REQ-LEARN-004: Negative values treated as absolute for borrow count."""
        # abs(1000) - abs(-1) = 1000 - 1 → 3 borrows
        assert _count_borrows(1000, -1) == 3
        assert _count_borrows(-1000, 1) == 3

    def test_swap_when_b_greater(self) -> None:
        """REQ-LEARN-004: When b > a, swapped to count borrows in b - a."""
        # 1 - 1000: swapped to 1000 - 1 → 3 borrows
        assert _count_borrows(1, 1000) == 3


# ---------------------------------------------------------------------------
# _digit_count helper tests
# REQ-LEARN-004: digit count for overflow detection
# ---------------------------------------------------------------------------


class TestDigitCount:
    """REQ-LEARN-004: _digit_count returns number of decimal digits."""

    def test_single_digit(self) -> None:
        assert _digit_count(5) == 1

    def test_two_digits(self) -> None:
        assert _digit_count(42) == 2

    def test_three_digits(self) -> None:
        assert _digit_count(999) == 3

    def test_zero(self) -> None:
        """REQ-LEARN-004: Zero has 1 digit."""
        assert _digit_count(0) == 1

    def test_negative(self) -> None:
        """REQ-LEARN-004: Negative values use absolute value."""
        assert _digit_count(-123) == 3

    def test_power_of_ten(self) -> None:
        """REQ-LEARN-004: 1000 has 4 digits."""
        assert _digit_count(1000) == 4


# ---------------------------------------------------------------------------
# CarryChainConstraint v2 — subtraction borrow chain tests
# REQ-LEARN-004: borrow chain detection
# ---------------------------------------------------------------------------


class TestCarryChainConstraintV2Subtraction:
    """REQ-LEARN-004: CarryChainConstraint v2 detects subtraction borrow chains."""

    def setup_method(self) -> None:
        self.ext = CarryChainConstraint()

    def test_borrow_chain_correct(self) -> None:
        """REQ-LEARN-004: 1000 - 1 = 999 is satisfied (correct borrow chain)."""
        results = self.ext.extract("1000 - 1 = 999")
        assert len(results) == 1
        r = results[0]
        assert r.constraint_type == PATTERN_ARITHMETIC_CARRY
        assert r.metadata["satisfied"] is True
        assert r.metadata["borrow_count"] == 3
        assert r.metadata["op"] == "-"

    def test_borrow_chain_wrong(self) -> None:
        """REQ-LEARN-004: 1000 - 1 = 1099 is violated (wrong borrow result)."""
        results = self.ext.extract("1000 - 1 = 1099")
        assert len(results) == 1
        r = results[0]
        assert r.metadata["satisfied"] is False
        assert r.metadata["correct_result"] == 999
        assert "correct: 999" in r.description

    def test_two_borrow_correct(self) -> None:
        """REQ-LEARN-004: 100 - 1 = 99 is satisfied (2 borrows >= min_carries)."""
        results = self.ext.extract("100 - 1 = 99")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True
        assert results[0].metadata["borrow_count"] == 2

    def test_single_borrow_skipped(self) -> None:
        """REQ-LEARN-004: 10 - 1 = 9 is NOT flagged (only 1 borrow < min_carries)."""
        # 1 borrow only, no negative-result issue
        results = self.ext.extract("10 - 1 = 9")
        assert results == []

    def test_simple_subtraction_skipped(self) -> None:
        """REQ-LEARN-004: 5 - 3 = 2 is NOT flagged (no borrow)."""
        results = self.ext.extract("5 - 3 = 2")
        assert results == []

    def test_borrow_chain_metadata_fields(self) -> None:
        """REQ-LEARN-004: Metadata has a, b, op, borrow_count, correct_result."""
        results = self.ext.extract("100 - 1 = 98")
        assert len(results) == 1
        m = results[0].metadata
        assert m["a"] == 100
        assert m["b"] == 1
        assert m["op"] == "-"
        assert m["borrow_count"] == 2
        assert m["correct_result"] == 99
        assert m["claimed_result"] == 98
        assert m["satisfied"] is False


# ---------------------------------------------------------------------------
# CarryChainConstraint v2 — negative result detection tests
# REQ-LEARN-004: B > A subtraction must yield negative
# ---------------------------------------------------------------------------


class TestCarryChainConstraintV2NegativeResult:
    """REQ-LEARN-004: CarryChainConstraint v2 detects B > A positive-result errors."""

    def setup_method(self) -> None:
        self.ext = CarryChainConstraint()

    def test_negative_result_correct(self) -> None:
        """REQ-LEARN-004: 1 - 5 = -4 is satisfied (correct negative result)."""
        results = self.ext.extract("1 - 5 = -4")
        assert len(results) == 1
        r = results[0]
        assert r.metadata["satisfied"] is True
        assert r.metadata["negative_violated"] is False

    def test_negative_result_violated_positive(self) -> None:
        """REQ-LEARN-004: 1 - 5 = 4 is violated (B > A, result must be negative)."""
        results = self.ext.extract("1 - 5 = 4")
        assert len(results) == 1
        r = results[0]
        assert r.metadata["satisfied"] is False
        assert r.metadata["negative_violated"] is True
        assert "should be negative" in r.description

    def test_negative_result_violated_zero(self) -> None:
        """REQ-LEARN-004: 1 - 5 = 0 is violated (B > A, result should be -4)."""
        results = self.ext.extract("1 - 5 = 0")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is False

    def test_negative_result_correct_result_value(self) -> None:
        """REQ-LEARN-004: Correct result stored in metadata for negative case."""
        results = self.ext.extract("3 - 10 = -7")
        assert len(results) == 1
        assert results[0].metadata["correct_result"] == -7
        assert results[0].metadata["satisfied"] is True

    def test_b_equals_a_zero_result_ok(self) -> None:
        """REQ-LEARN-004: 5 - 5 = 0 is satisfied (result is zero, not negative)."""
        # 0 borrows, a == b so not negative_violated
        results = self.ext.extract("5 - 5 = 0")
        assert results == []  # 0 borrows, not negative violated → skipped


# ---------------------------------------------------------------------------
# CarryChainConstraint v2 — digit count check tests
# REQ-LEARN-004: addition result digit count overflow detection
# ---------------------------------------------------------------------------


class TestCarryChainConstraintV2DigitCount:
    """REQ-LEARN-004: CarryChainConstraint v2 detects digit-count violations."""

    def setup_method(self) -> None:
        self.ext = CarryChainConstraint()

    def test_digit_count_ok(self) -> None:
        """REQ-LEARN-004: 99 + 2 = 101 is satisfied (3 digits, max expected 3)."""
        results = self.ext.extract("99 + 2 = 101")
        # carry_count for 99+2: 2 carries → flagged; 101 is correct
        assert len(results) == 1
        r = results[0]
        assert r.metadata["digit_count_ok"] is True
        assert r.metadata["satisfied"] is True

    def test_digit_count_violated(self) -> None:
        """REQ-LEARN-004: 99 + 2 = 10099 is violated (5 digits, max expected 3)."""
        results = self.ext.extract("99 + 2 = 10099")
        assert len(results) == 1
        r = results[0]
        assert r.metadata["digit_count_ok"] is False
        assert r.metadata["satisfied"] is False
        assert r.metadata["claimed_digits"] == 5
        assert r.metadata["max_expected_digits"] == 3

    def test_digit_count_metadata(self) -> None:
        """REQ-LEARN-004: Metadata includes claimed_digits and max_expected_digits."""
        results = self.ext.extract("99 + 1 = 100")
        assert len(results) == 1
        m = results[0].metadata
        assert "claimed_digits" in m
        assert "max_expected_digits" in m
        assert "digit_count_ok" in m


# ---------------------------------------------------------------------------
# NegationConstraint v2 — violation detection tests
# REQ-LEARN-004: negation satisfied vs violated
# ---------------------------------------------------------------------------


class TestNegationConstraintV2Violation:
    """REQ-LEARN-004: NegationConstraint v2 correctly detects satisfied/violated."""

    def setup_method(self) -> None:
        self.ext = NegationConstraint()

    def test_satisfied_when_negation_respected(self) -> None:
        """REQ-LEARN-004: 'NOT 42' + later asserts '17' → satisfied=True."""
        text = "The answer is NOT 42. The answer is 17."
        results = self.ext.extract(text)
        assert len(results) >= 1
        is_not_results = [r for r in results if r.metadata.get("pattern") == "is_not"]
        assert len(is_not_results) >= 1
        # All is_not results should be satisfied (17 ≠ 42)
        for r in is_not_results:
            assert r.metadata["satisfied"] is True

    def test_violated_when_positive_asserted(self) -> None:
        """REQ-LEARN-004: 'NOT 42' + later asserts '42' → satisfied=False."""
        text = "The answer is NOT 42. The answer is 42."
        results = self.ext.extract(text)
        assert len(results) >= 1
        is_not_results = [r for r in results if r.metadata.get("pattern") == "is_not"]
        assert len(is_not_results) >= 1
        # At least one is_not result should be violated (42 asserted again)
        violated = [r for r in is_not_results if r.metadata["satisfied"] is False]
        assert len(violated) >= 1

    def test_satisfied_no_positive_form(self) -> None:
        """REQ-LEARN-004: 'X is not Y' with no subsequent positive assertion."""
        results = self.ext.extract("Paris is not the capital of Germany.")
        is_not_results = [r for r in results if r.metadata.get("pattern") == "is_not"]
        assert len(is_not_results) >= 1
        for r in is_not_results:
            assert r.metadata["satisfied"] is True

    def test_is_not_metadata_has_satisfied(self) -> None:
        """REQ-LEARN-004: is_not metadata always contains 'satisfied' key."""
        results = self.ext.extract("The result is not correct.")
        is_not = [r for r in results if r.metadata.get("pattern") == "is_not"]
        assert len(is_not) >= 1
        for r in is_not:
            assert "satisfied" in r.metadata


# ---------------------------------------------------------------------------
# NegationConstraint v2 — "not all A are B" pattern tests
# REQ-LEARN-004: universal negation extraction
# ---------------------------------------------------------------------------


class TestNegationConstraintV2NotAll:
    """REQ-LEARN-004: NegationConstraint v2 extracts 'not all A are B' patterns."""

    def setup_method(self) -> None:
        self.ext = NegationConstraint()

    def test_not_all_extracted(self) -> None:
        """REQ-LEARN-004: 'not all dogs are mammals' is extracted."""
        results = self.ext.extract("not all dogs are mammals.")
        not_all = [r for r in results if r.metadata.get("pattern") == "not_all"]
        assert len(not_all) == 1
        assert not_all[0].constraint_type == PATTERN_NEGATION_SCOPE

    def test_not_all_metadata(self) -> None:
        """REQ-LEARN-004: 'not_all' metadata has subject and negated_predicate."""
        results = self.ext.extract("not all swans are white.")
        not_all = [r for r in results if r.metadata.get("pattern") == "not_all"]
        assert len(not_all) == 1
        m = not_all[0].metadata
        assert m["subject"] == "swans"
        assert m["negated_predicate"] == "white"
        assert "satisfied" in m

    def test_not_all_satisfied_no_positive(self) -> None:
        """REQ-LEARN-004: 'not all A are B' with no 'all A are B' → satisfied=True."""
        results = self.ext.extract("not all dogs are cats.")
        not_all = [r for r in results if r.metadata.get("pattern") == "not_all"]
        assert len(not_all) >= 1
        for r in not_all:
            assert r.metadata["satisfied"] is True

    def test_not_all_violated(self) -> None:
        """REQ-LEARN-004: 'not all A are B' + 'all A are B' later → satisfied=False."""
        text = "not all mammals are warm-blooded. In fact, all mammals are warm-blooded."
        results = self.ext.extract(text)
        not_all = [r for r in results if r.metadata.get("pattern") == "not_all"]
        assert len(not_all) >= 1
        violated = [r for r in not_all if r.metadata["satisfied"] is False]
        assert len(violated) >= 1


# ---------------------------------------------------------------------------
# NegationConstraint v2 — "no A are/is/has B" pattern tests
# REQ-LEARN-004: universal negation ("no A are B")
# ---------------------------------------------------------------------------


class TestNegationConstraintV2NoAreB:
    """REQ-LEARN-004: NegationConstraint v2 extracts 'no A are/is/has B' patterns."""

    def setup_method(self) -> None:
        self.ext = NegationConstraint()

    def test_no_A_are_B_extracted(self) -> None:
        """REQ-LEARN-004: 'no cats are blue' is extracted."""
        results = self.ext.extract("no cats are blue.")
        no_are = [r for r in results if r.metadata.get("pattern") == "no_A_are_B"]
        assert len(no_are) == 1
        assert no_are[0].constraint_type == PATTERN_NEGATION_SCOPE

    def test_no_A_is_B_extracted(self) -> None:
        """REQ-LEARN-004: 'no answer is correct' is extracted."""
        results = self.ext.extract("no answer is correct.")
        no_are = [r for r in results if r.metadata.get("pattern") == "no_A_are_B"]
        assert len(no_are) == 1

    def test_no_A_are_B_metadata(self) -> None:
        """REQ-LEARN-004: 'no_A_are_B' metadata has subject and negated_predicate."""
        results = self.ext.extract("no birds are reptiles.")
        no_are = [r for r in results if r.metadata.get("pattern") == "no_A_are_B"]
        assert len(no_are) == 1
        m = no_are[0].metadata
        assert m["subject"] == "birds"
        assert m["negated_predicate"] == "reptiles"
        assert "satisfied" in m

    def test_no_A_are_B_satisfied(self) -> None:
        """REQ-LEARN-004: 'no A are B' with no positive form → satisfied=True."""
        results = self.ext.extract("no cats are dogs.")
        no_are = [r for r in results if r.metadata.get("pattern") == "no_A_are_B"]
        assert len(no_are) >= 1
        for r in no_are:
            assert r.metadata["satisfied"] is True

    def test_no_A_are_B_violated(self) -> None:
        """REQ-LEARN-004: 'no cats are blue' + 'cats are blue' later → satisfied=False."""
        text = "no cats are blue. But actually cats are blue in this context."
        results = self.ext.extract(text)
        no_are = [r for r in results if r.metadata.get("pattern") == "no_A_are_B"]
        assert len(no_are) >= 1
        violated = [r for r in no_are if r.metadata["satisfied"] is False]
        assert len(violated) >= 1

    def test_no_text_no_false_positive(self) -> None:
        """REQ-LEARN-004: 'The sky is blue.' does not trigger no_A_are_B."""
        results = self.ext.extract("The sky is blue.")
        no_are = [r for r in results if r.metadata.get("pattern") == "no_A_are_B"]
        assert no_are == []


# ---------------------------------------------------------------------------
# NegationConstraint v2 — "not equal" pattern backward compatibility
# REQ-LEARN-004: existing not-equal patterns still work
# ---------------------------------------------------------------------------


class TestNegationConstraintV2NotEqual:
    """REQ-LEARN-004: not-equal pattern still works after v2 refactor."""

    def setup_method(self) -> None:
        self.ext = NegationConstraint()

    def test_unicode_not_equal_extracted(self) -> None:
        """REQ-LEARN-004: '5 ≠ 3' is extracted with pattern='not_equal'."""
        results = self.ext.extract("5 ≠ 3")
        assert len(results) == 1
        assert results[0].metadata["pattern"] == "not_equal"
        assert results[0].constraint_type == PATTERN_NEGATION_SCOPE

    def test_ascii_not_equal_extracted(self) -> None:
        """REQ-LEARN-004: '5 != 3' is extracted with pattern='not_equal'."""
        results = self.ext.extract("5 != 3")
        assert len(results) == 1
        assert results[0].metadata["pattern"] == "not_equal"

    def test_not_equal_satisfied(self) -> None:
        """REQ-LEARN-004: '5 ≠ 3' with no '5 = 3' elsewhere → satisfied=True."""
        results = self.ext.extract("We know 5 ≠ 3.")
        assert len(results) >= 1
        ne = [r for r in results if r.metadata.get("pattern") == "not_equal"]
        assert all(r.metadata["satisfied"] is True for r in ne)

    def test_not_equal_metadata_has_satisfied(self) -> None:
        """REQ-LEARN-004: not_equal metadata always contains 'satisfied' key."""
        results = self.ext.extract("7 != 8")
        assert len(results) >= 1
        for r in results:
            assert "satisfied" in r.metadata


# ---------------------------------------------------------------------------
# NegationConstraint v2 — existing behavior backward compatibility
# REQ-LEARN-004: all pre-v2 tests still pass
# ---------------------------------------------------------------------------


class TestNegationConstraintV2BackwardCompat:
    """REQ-LEARN-004: v2 changes are backward-compatible with v1 test cases."""

    def setup_method(self) -> None:
        self.ext = NegationConstraint()

    def test_is_not_pattern_still_extracted(self) -> None:
        """REQ-LEARN-004: 'The answer is not 42.' is still extracted."""
        results = self.ext.extract("The answer is not 42.")
        assert len(results) >= 1
        assert any(r.constraint_type == PATTERN_NEGATION_SCOPE for r in results)

    def test_no_negation_still_empty(self) -> None:
        """REQ-LEARN-004: 'The sky is blue.' still returns empty list."""
        results = self.ext.extract("The sky is blue.")
        assert results == []

    def test_deduplication_within_text(self) -> None:
        """REQ-LEARN-004: Same negation phrase repeated is extracted once."""
        text = "The answer is not 42. The answer is not 42."
        results = self.ext.extract(text)
        is_not_descs = [r.description for r in results if "42" in r.description]
        assert len(set(is_not_descs)) == len(is_not_descs)

    def test_empty_text_returns_empty(self) -> None:
        """REQ-LEARN-004: Empty text returns empty list."""
        assert self.ext.extract("") == []

    def test_is_not_metadata_has_subject_and_predicate(self) -> None:
        """REQ-LEARN-004: is_not metadata still has 'subject' and 'negated_predicate'."""
        results = self.ext.extract("The result is not correct.")
        is_not = [r for r in results if r.metadata.get("pattern") == "is_not"]
        assert len(is_not) >= 1
        r = is_not[0]
        assert "subject" in r.metadata
        assert "negated_predicate" in r.metadata


# ---------------------------------------------------------------------------
# CarryChainConstraint v2 — backward compatibility
# REQ-LEARN-004: pre-v2 addition tests still pass
# ---------------------------------------------------------------------------


class TestCarryChainConstraintV2BackwardCompat:
    """REQ-LEARN-004: v2 changes are backward-compatible with v1 addition tests."""

    def setup_method(self) -> None:
        self.ext = CarryChainConstraint()

    def test_addition_carry_chain_correct(self) -> None:
        """REQ-LEARN-004: 99 + 1 = 100 still satisfied in v2."""
        results = self.ext.extract("99 + 1 = 100")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True
        assert results[0].metadata["carry_count"] == 2

    def test_addition_carry_chain_wrong(self) -> None:
        """REQ-LEARN-004: 99 + 1 = 90 still violated in v2."""
        results = self.ext.extract("99 + 1 = 90")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is False
        assert results[0].metadata["correct_result"] == 100

    def test_single_carry_addition_skipped(self) -> None:
        """REQ-LEARN-004: 5 + 7 = 12 still not flagged (single carry, digit ok)."""
        results = self.ext.extract("5 + 7 = 12")
        assert results == []

    def test_no_carry_addition_skipped(self) -> None:
        """REQ-LEARN-004: 2 + 3 = 5 still not flagged in v2."""
        results = self.ext.extract("2 + 3 = 5")
        assert results == []

    def test_addition_metadata_has_op_field(self) -> None:
        """REQ-LEARN-004: v2 addition metadata includes 'op' field."""
        results = self.ext.extract("99 + 1 = 100")
        assert len(results) == 1
        assert results[0].metadata["op"] == "+"

    def test_multiple_carry_chains_still_work(self) -> None:
        """REQ-LEARN-004: Multiple carry-chain expressions still extracted."""
        text = "First: 99 + 1 = 100. Then: 999 + 1 = 1000."
        results = self.ext.extract(text)
        assert len(results) == 2
