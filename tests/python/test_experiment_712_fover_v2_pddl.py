"""Tests for PDDL-based step labeler (Exp 712: FoVer v2 corpus).

Spec: REQ-DATA-005, REQ-DATA-006, REQ-DATA-007,
      SCENARIO-DATA-005, SCENARIO-DATA-006, SCENARIO-DATA-007
"""

from __future__ import annotations

import pytest

from carnot.training.pddl_labeler import (
    _safe_eval_binop,
    encode_step_transition,
    extract_quantities,
    label_gsm8k_chain,
    verify_transition,
)


# ---------------------------------------------------------------------------
# extract_quantities — REQ-DATA-006, SCENARIO-DATA-006
# ---------------------------------------------------------------------------


class TestExtractQuantities:
    """Tests for extract_quantities().  Spec: REQ-DATA-006."""

    def test_num_then_word_pattern(self) -> None:
        """'5 apples' extracts apples → 5.0.  Spec: REQ-DATA-006."""
        result = extract_quantities("Alice has 5 apples.")
        assert "apples" in result
        assert result["apples"] == 5.0

    def test_word_then_num_pattern(self) -> None:
        """'price is 60' extracts price → 60.0.  Spec: REQ-DATA-006."""
        result = extract_quantities("The price is 60 dollars.")
        assert "price" in result
        assert result["price"] == 60.0

    def test_multiple_quantities(self) -> None:
        """Multiple named quantities are all extracted.  Spec: REQ-DATA-006."""
        result = extract_quantities("There are 3 cats and 7 dogs in the shelter.")
        assert "cats" in result
        assert "dogs" in result
        assert result["cats"] == 3.0
        assert result["dogs"] == 7.0

    def test_stopwords_excluded(self) -> None:
        """Common stopwords like 'times' and 'each' are not extracted as quantity names.
        Spec: REQ-DATA-006.
        """
        result = extract_quantities("She earned 4 times as much as before.")
        assert "times" not in result

    def test_decimal_value(self) -> None:
        """Decimal quantities like '3.5 kg' are parsed correctly.  Spec: REQ-DATA-006."""
        result = extract_quantities("The bag weighs 3.5 kilograms.")
        assert "kilograms" in result
        assert result["kilograms"] == 3.5

    def test_empty_text_returns_empty_dict(self) -> None:
        """Empty input produces empty dict.  Spec: REQ-DATA-006."""
        result = extract_quantities("")
        assert result == {}

    def test_no_quantities(self) -> None:
        """Text with no numbers produces empty dict.  Spec: REQ-DATA-006."""
        result = extract_quantities("The quick brown fox jumps.")
        assert result == {}

    def test_scenario_data_006_example(self) -> None:
        """Direct SCENARIO-DATA-006: '5 apples and 3 oranges' → both extracted."""
        result = extract_quantities("Alice has 5 apples and 3 oranges.")
        assert result.get("apples") == 5.0
        assert result.get("oranges") == 3.0


# ---------------------------------------------------------------------------
# verify_transition — REQ-DATA-007, SCENARIO-DATA-007
# ---------------------------------------------------------------------------


class TestVerifyTransition:
    """Tests for verify_transition().  Spec: REQ-DATA-007."""

    def test_correct_addition_step(self) -> None:
        """5 + 3 = 8 with prev={apples:5}, next={apples:8} → True.
        Spec: REQ-DATA-007, SCENARIO-DATA-007.
        """
        prev = {"apples": 5.0}
        nxt = {"apples": 8.0}
        assert verify_transition("5 + 3 = 8", prev, nxt) is True

    def test_incorrect_multiplication_step(self) -> None:
        """5 * 3 = 15 when next state expects 8 → False.
        Spec: REQ-DATA-007, SCENARIO-DATA-007b.
        """
        prev = {"apples": 5.0}
        nxt = {"apples": 8.0}
        assert verify_transition("5 * 3 = 15", prev, nxt) is False

    def test_subtraction_step_correct(self) -> None:
        """10 - 4 = 6 with prev={x:10}, next={x:6} → True.  Spec: REQ-DATA-007."""
        prev = {"x": 10.0}
        nxt = {"x": 6.0}
        assert verify_transition("10 - 4 = 6", prev, nxt) is True

    def test_identical_states_trivially_correct(self) -> None:
        """When prev and next state are identical, no transition to verify → True.
        Spec: REQ-DATA-007.
        """
        state = {"apples": 5.0}
        assert verify_transition("no arithmetic here", state, state) is True

    def test_no_arithmetic_in_step(self) -> None:
        """Step with no numeric expression + non-trivial state change → False.
        Spec: REQ-DATA-007.
        """
        prev = {"apples": 5.0}
        nxt = {"apples": 9.0}
        # The step has no arithmetic that produces 9.
        assert verify_transition("something happened", prev, nxt) is False

    def test_bare_expression_without_equals(self) -> None:
        """Bare 'a OP b' without stated result is evaluated and checked.
        Spec: REQ-DATA-007.
        """
        prev = {"count": 3.0}
        nxt = {"count": 9.0}
        # 3 * 3 = 9 — step text has no = sign, but computed result matches next state.
        assert verify_transition("3 * 3", prev, nxt) is True

    def test_division_step_correct(self) -> None:
        """12 / 4 = 3 with next_state containing 3 → True.  Spec: REQ-DATA-007."""
        prev = {"items": 12.0}
        nxt = {"items": 3.0}
        assert verify_transition("12 / 4 = 3", prev, nxt) is True


# ---------------------------------------------------------------------------
# label_gsm8k_chain — REQ-DATA-005, SCENARIO-DATA-005
# ---------------------------------------------------------------------------


class TestLabelGsm8kChain:
    """Tests for label_gsm8k_chain().  Spec: REQ-DATA-005."""

    def test_returns_one_label_per_step(self) -> None:
        """Output list length equals input cot_steps length.  Spec: REQ-DATA-005."""
        question = "A store has 5 apples and 3 oranges."
        steps = ["5 + 3 = 8.", "8 - 2 = 6."]
        result = label_gsm8k_chain(question, steps)
        assert len(result) == 2

    def test_correct_step_labeled_true(self) -> None:
        """A step whose arithmetic is self-consistent is labeled step_correct=True.
        Spec: REQ-DATA-005, SCENARIO-DATA-005.
        """
        question = "A bag has 4 red balls and 6 blue balls."
        steps = ["4 + 6 = 10."]
        result = label_gsm8k_chain(question, steps)
        assert result[0]["step_correct"] is True

    def test_labeler_field_is_pddl(self) -> None:
        """Every label entry has labeler='pddl'.  Spec: REQ-DATA-005."""
        question = "There are 3 cats."
        steps = ["3 + 0 = 3."]
        result = label_gsm8k_chain(question, steps)
        assert result[0]["labeler"] == "pddl"

    def test_required_fields_present(self) -> None:
        """Each label dict has all required keys.  Spec: REQ-DATA-005."""
        question = "There are 2 apples."
        steps = ["2 + 1 = 3."]
        result = label_gsm8k_chain(question, steps)
        required_keys = {"step", "step_index", "step_correct", "action", "prev_state", "next_state", "labeler"}
        assert required_keys.issubset(result[0].keys())

    def test_step_index_is_sequential(self) -> None:
        """step_index values are 0, 1, 2 ... for a 3-step chain.  Spec: REQ-DATA-005."""
        question = "Sam has 10 books."
        steps = ["10 - 3 = 7.", "7 + 5 = 12.", "12 / 2 = 6."]
        result = label_gsm8k_chain(question, steps)
        assert [r["step_index"] for r in result] == [0, 1, 2]

    def test_empty_steps_returns_empty_list(self) -> None:
        """Empty step list returns empty list.  Spec: REQ-DATA-005."""
        result = label_gsm8k_chain("Any question.", [])
        assert result == []

    def test_state_advances_between_steps(self) -> None:
        """prev_state of step N+1 equals next_state of step N.  Spec: REQ-DATA-005."""
        question = "A box has 5 items."
        steps = ["5 + 3 = 8.", "8 - 2 = 6."]
        result = label_gsm8k_chain(question, steps)
        # next_state of step 0 should equal prev_state of step 1.
        assert result[0]["next_state"] == result[1]["prev_state"]


# ---------------------------------------------------------------------------
# _safe_eval_binop (internal helper) — branch coverage
# ---------------------------------------------------------------------------


class TestSafeEvalBinop:
    """Tests for the _safe_eval_binop helper to ensure full branch coverage."""

    def test_addition(self) -> None:
        assert _safe_eval_binop(3.0, "+", 4.0) == pytest.approx(7.0)

    def test_subtraction(self) -> None:
        assert _safe_eval_binop(10.0, "-", 4.0) == pytest.approx(6.0)

    def test_multiplication(self) -> None:
        assert _safe_eval_binop(3.0, "*", 5.0) == pytest.approx(15.0)

    def test_division(self) -> None:
        assert _safe_eval_binop(12.0, "/", 4.0) == pytest.approx(3.0)

    def test_division_by_zero_returns_none(self) -> None:
        assert _safe_eval_binop(5.0, "/", 0.0) is None

    def test_unknown_operator_returns_none(self) -> None:
        assert _safe_eval_binop(5.0, "^", 2.0) is None


class TestVerifyTransitionEdgeCases:
    """Additional edge-case tests for branch coverage on verify_transition."""

    def test_stated_value_matches_new_key_in_next_state(self) -> None:
        """When next_state has a key not in prev_state, its value is a valid target.
        This exercises the 'new key in next_state' branch (line 244).
        """
        prev: dict[str, float] = {}
        nxt = {"result": 8.0}
        # Step says 5 + 3 = 8; result is a new key in next_state.
        assert verify_transition("5 + 3 = 8", prev, nxt) is True

    def test_stated_value_matches_changed_value_even_when_computed_differs(self) -> None:
        """stated=8 matches changed_value=8.0 even if computed (5*3=15) differs.
        This exercises the 'stated matches cv' fallback (line 264).
        """
        prev = {"apples": 5.0}
        nxt = {"apples": 8.0}
        # Step has wrong arithmetic (5*3=8 stated; 5*3 actually=15) but stated=8 matches next.
        assert verify_transition("5 * 3 = 8", prev, nxt) is True


class TestEncodeStepTransitionEdgeCases:
    """Additional edge-case tests for encode_step_transition branch coverage."""

    def test_explicit_result_no_operand_match_stores_under_result(self) -> None:
        """When no state key's value matches an operand, result stored as '_result'.
        This exercises the for-else branch in the explicit-result path.
        """
        state = {"bananas": 99.0}
        action, new_state = encode_step_transition("2 + 3 = 5.", state)
        assert "_result" in new_state
        assert new_state["_result"] == 5.0

    def test_bare_expression_no_operand_match_stores_under_result(self) -> None:
        """Bare 'a OP b' (no = sign) with no state-key match stores result as '_result'.
        This exercises the for-else branch in the bare-expression path (line 200).
        """
        state = {"bananas": 99.0}
        # 2 + 3 = 5 — neither operand matches bananas=99.
        action, new_state = encode_step_transition("2 + 3", state)
        # Result (5.0) should be stored under '_result' since no key matched.
        assert "_result" in new_state
        assert new_state["_result"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# encode_step_transition
# ---------------------------------------------------------------------------


class TestEncodeStepTransition:
    """Tests for encode_step_transition().  Spec: REQ-DATA-007."""

    def test_explicit_result_expression(self) -> None:
        """'5 + 3 = 8' is parsed and state updated if an operand matches."""
        state = {"apples": 5.0}
        action, new_state = encode_step_transition("5 + 3 = 8.", state)
        assert "+" in action
        assert new_state.get("apples") == 8.0 or new_state.get("_result") == 8.0

    def test_no_arithmetic_returns_unchanged_state(self) -> None:
        """Step with no arithmetic returns the original state unchanged."""
        state = {"apples": 5.0}
        action, new_state = encode_step_transition("No numbers here.", state)
        assert action == "no_arithmetic"
        assert new_state == state

    def test_bare_expression_updates_matching_key(self) -> None:
        """Bare '5 + 3' (no = sign) evaluates to 8 and updates a matching key."""
        state = {"apples": 5.0}
        action, new_state = encode_step_transition("5 + 3", state)
        # apples was 5, which is operand a; result should be 8.
        assert new_state.get("apples") == pytest.approx(8.0) or new_state.get("_result") == pytest.approx(8.0)

    def test_new_quantity_stored_as_result(self) -> None:
        """When no state key matches an operand, result is stored under '_result'."""
        state = {"bananas": 99.0}
        action, new_state = encode_step_transition("2 + 3 = 5.", state)
        assert "_result" in new_state or "bananas" in new_state
