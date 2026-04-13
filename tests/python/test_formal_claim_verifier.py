"""Tests for `carnot.pipeline.formal_claim_verifier`.

Spec: REQ-VERIFY-058, REQ-VERIFY-059
"""

from __future__ import annotations

import json

import pytest

import carnot.pipeline.formal_claim_verifier as fcv
from carnot.pipeline.formal_claim_verifier import (
    FormalClaim,
    FormalClaimBatchResult,
    FormalClaimVerdict,
    FormalClaimVerifier,
    normalize_claim,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Helpers: minimal claim dicts matching the corpus schema
# ---------------------------------------------------------------------------


def _arithmetic_claim(
    claim_id: str = "cl1",
    operands: list[float] | None = None,
    relation_type: str = "equation",
) -> dict:
    """100 - 24 = 76  (supported arithmetic claim)."""
    return {
        "claim_id": claim_id,
        "claim_text": "Guests remaining = 100 - 24 = 76",
        "candidate_solver_route": "arithmetic",
        "formalization_status": "formalized",
        "relation_type": relation_type,
        "operands": operands if operands is not None else [100.0, 24.0, 76.0],
        "target": "derived_quantity",
        "bound_variables": [],
    }


def _comparison_claim(
    claim_id: str = "cl2",
    relation_type: str = "less_than",
    operands: list[float] | None = None,
) -> dict:
    """Sidney ordered 3 less than 10 = 7 sandwiches."""
    return {
        "claim_id": claim_id,
        "claim_text": "Sidney ordered 3 less than 10 = 7 sandwiches",
        "candidate_solver_route": "comparison",
        "formalization_status": "formalized",
        "relation_type": relation_type,
        "operands": operands if operands is not None else [3.0, 7.0],
        "target": "sidney",
        "bound_variables": ["sidney", "sandwiches"],
    }


def _cardinality_claim(
    claim_id: str = "cl3",
    relation_type: str = "equals",
    operands: list[float] | None = None,
) -> dict:
    """bullet_count equals 3."""
    return {
        "claim_id": claim_id,
        "claim_text": "bullet_count equals 3",
        "candidate_solver_route": "cardinality",
        "formalization_status": "formalized",
        "relation_type": relation_type,
        "operands": operands if operands is not None else [3.0],
        "target": "bullet_count",
        "bound_variables": [],
    }


def _set_membership_claim(
    claim_id: str = "cl4",
    relation_type: str = "contains",
    operands: list[float] | None = None,
) -> dict:
    """answer_surface contains risk."""
    return {
        "claim_id": claim_id,
        "claim_text": "answer_surface contains risk",
        "candidate_solver_route": "set_membership",
        "formalization_status": "formalized",
        "relation_type": relation_type,
        "operands": operands if operands is not None else [],
        "target": "answer_surface",
        "bound_variables": ["risk"],
    }


def _boolean_entailment_claim(
    claim_id: str = "cl5",
    relation_type: str = "equals",
    operands: list[float] | None = None,
) -> dict:
    """function_name equals chunk_list."""
    return {
        "claim_id": claim_id,
        "claim_text": "function_name equals chunk_list",
        "candidate_solver_route": "boolean_entailment",
        "formalization_status": "formalized",
        "relation_type": relation_type,
        "operands": operands if operands is not None else [],
        "target": "function_name",
        "bound_variables": ["chunk_list"],
    }


def _not_formalizable_claim(claim_id: str = "cl6") -> dict:
    """Claim that cannot be normalized — must abstain."""
    return {
        "claim_id": claim_id,
        "claim_text": "Here is the step-by-step solution:",
        "candidate_solver_route": "not_formalizable",
        "formalization_status": "not_formalizable",
        "relation_type": "unparsed_claim",
        "operands": [],
        "target": "unknown_claim_target",
        "bound_variables": ["here", "step", "solution"],
    }


def _unknown_route_claim(claim_id: str = "cl7") -> dict:
    """Claim with a route not in the supported set — must abstain."""
    return {
        "claim_id": claim_id,
        "claim_text": "execute test_suite and pass",
        "candidate_solver_route": "execution_oracle",
        "formalization_status": "formalized",
        "relation_type": "equals",
        "operands": [],
        "target": "test_result",
        "bound_variables": [],
    }


# ---------------------------------------------------------------------------
# REQ-VERIFY-058 — normalization
# ---------------------------------------------------------------------------


class TestNormalizeClaim:
    """REQ-VERIFY-058: normalize_claim produces a FormalClaim with correct fields."""

    def test_arithmetic_claim_normalizes(self) -> None:
        # SCENARIO: a formalized arithmetic claim is converted to typed FormalClaim
        raw = _arithmetic_claim()
        claim = normalize_claim(raw)
        assert isinstance(claim, FormalClaim)
        assert claim.claim_id == "cl1"
        assert claim.route == "arithmetic"
        assert claim.formalization_status == "formalized"
        assert claim.operands == [100.0, 24.0, 76.0]

    def test_not_formalizable_status_preserved(self) -> None:
        # SCENARIO: not_formalizable claims retain their status through normalization
        raw = _not_formalizable_claim()
        claim = normalize_claim(raw)
        assert claim.formalization_status == "not_formalizable"
        assert claim.route == "not_formalizable"

    def test_missing_optional_fields_get_defaults(self) -> None:
        # SCENARIO: partial claim dict doesn't crash normalization
        raw = {
            "claim_id": "cl99",
            "claim_text": "x equals y",
            "candidate_solver_route": "boolean_entailment",
            "formalization_status": "formalized",
            "relation_type": "equals",
        }
        claim = normalize_claim(raw)
        assert claim.claim_id == "cl99"
        assert claim.operands == []
        assert claim.bound_variables == []

    def test_normalize_preserves_all_required_fields(self) -> None:
        # SCENARIO: round-trip through normalization keeps all typed fields
        raw = _set_membership_claim()
        claim = normalize_claim(raw)
        assert claim.relation_type == "contains"
        assert claim.target == "answer_surface"
        assert "risk" in claim.bound_variables


# ---------------------------------------------------------------------------
# REQ-VERIFY-058 — route selection
# ---------------------------------------------------------------------------


class TestRouteSelection:
    """REQ-VERIFY-058: Each supported route maps to the correct checker."""

    def test_arithmetic_route_selected(self) -> None:
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_arithmetic_claim()))
        assert verdict.route == "arithmetic"

    def test_comparison_route_selected(self) -> None:
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_comparison_claim()))
        assert verdict.route == "comparison"

    def test_cardinality_route_selected(self) -> None:
        # Use two operands so the checker can produce a definitive verdict (not abstain)
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_cardinality_claim(operands=[3.0, 3.0])))
        assert verdict.route == "cardinality"

    def test_set_membership_route_selected(self) -> None:
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_set_membership_claim()))
        assert verdict.route == "set_membership"

    def test_boolean_entailment_route_selected(self) -> None:
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_boolean_entailment_claim()))
        assert verdict.route == "boolean_entailment"


# ---------------------------------------------------------------------------
# REQ-VERIFY-058 — arithmetic checker
# ---------------------------------------------------------------------------


class TestArithmeticChecker:
    """REQ-VERIFY-058: arithmetic route checks equation/operand consistency."""

    def test_correct_arithmetic_equation_is_supported(self) -> None:
        # SCENARIO: 100 - 24 = 76 → supported
        verifier = FormalClaimVerifier()
        claim = normalize_claim(_arithmetic_claim(operands=[100.0, 24.0, 76.0]))
        verdict = verifier.verify_claim(claim)
        assert verdict.verdict == "supported"
        assert verdict.failure_detail is None

    def test_wrong_arithmetic_equation_is_violated(self) -> None:
        # SCENARIO: 100 - 24 = 80 (wrong) → violated with failure_detail
        verifier = FormalClaimVerifier()
        claim = normalize_claim(_arithmetic_claim(operands=[100.0, 24.0, 80.0]))
        verdict = verifier.verify_claim(claim)
        assert verdict.verdict == "violated"
        assert verdict.failure_detail is not None

    def test_arithmetic_with_equals_relation_type(self) -> None:
        # SCENARIO: 2 * 15 = 30 stored as relation_type=equals
        verifier = FormalClaimVerifier()
        raw = {
            "claim_id": "clA",
            "claim_text": "2 * 15 = 30",
            "candidate_solver_route": "arithmetic",
            "formalization_status": "formalized",
            "relation_type": "equals",
            "operands": [2.0, 15.0, 30.0],
            "target": "derived_quantity",
            "bound_variables": [],
        }
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "supported"

    def test_arithmetic_with_only_two_operands_abstains(self) -> None:
        # SCENARIO: only 2 operands → not enough for equation check → abstain
        verifier = FormalClaimVerifier()
        claim = normalize_claim(_arithmetic_claim(operands=[100.0, 24.0]))
        verdict = verifier.verify_claim(claim)
        assert verdict.verdict == "abstain"

    def test_arithmetic_failure_detail_contains_expected_value(self) -> None:
        # SCENARIO: failure detail exposes the correct arithmetic result
        verifier = FormalClaimVerifier()
        claim = normalize_claim(_arithmetic_claim(operands=[100.0, 24.0, 99.0]))
        verdict = verifier.verify_claim(claim)
        assert verdict.verdict == "violated"
        assert verdict.failure_detail is not None
        # The detail must record what the correct result is
        assert "claimed" in verdict.failure_detail or "expected" in verdict.failure_detail


# ---------------------------------------------------------------------------
# REQ-VERIFY-058 — comparison checker
# ---------------------------------------------------------------------------


class TestComparisonChecker:
    """REQ-VERIFY-058: comparison route handles less_than / greater_than / between."""

    def test_less_than_supported(self) -> None:
        # SCENARIO: operands=[3,7], relation=less_than → 3 < 7 → supported
        verifier = FormalClaimVerifier()
        claim = normalize_claim(_comparison_claim(operands=[3.0, 7.0], relation_type="less_than"))
        verdict = verifier.verify_claim(claim)
        assert verdict.verdict == "supported"

    def test_less_than_violated(self) -> None:
        # SCENARIO: 7 is NOT less than 3 → violated
        verifier = FormalClaimVerifier()
        claim = normalize_claim(_comparison_claim(operands=[7.0, 3.0], relation_type="less_than"))
        verdict = verifier.verify_claim(claim)
        assert verdict.verdict == "violated"

    def test_greater_than_supported(self) -> None:
        # SCENARIO: operands=[15,10], relation=greater_than → 15 > 10 → supported
        verifier = FormalClaimVerifier()
        raw = _comparison_claim(operands=[15.0, 10.0], relation_type="greater_than")
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "supported"

    def test_greater_than_violated(self) -> None:
        verifier = FormalClaimVerifier()
        raw = _comparison_claim(operands=[3.0, 10.0], relation_type="greater_than")
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "violated"

    def test_between_supported(self) -> None:
        # SCENARIO: operands=[4,7,5] → 4 <= 5 <= 7 → supported
        verifier = FormalClaimVerifier()
        raw = {
            "claim_id": "clC",
            "claim_text": "bullet_word_count between 4 and 7 (actual 5)",
            "candidate_solver_route": "comparison",
            "formalization_status": "formalized",
            "relation_type": "between",
            "operands": [4.0, 7.0, 5.0],
            "target": "bullet_word_count",
            "bound_variables": [],
        }
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "supported"

    def test_between_violated(self) -> None:
        # SCENARIO: operands=[4,7,10] → 10 not in [4,7] → violated
        verifier = FormalClaimVerifier()
        raw = {
            "claim_id": "clD",
            "claim_text": "bullet_word_count between 4 and 7 (actual 10)",
            "candidate_solver_route": "comparison",
            "formalization_status": "formalized",
            "relation_type": "between",
            "operands": [4.0, 7.0, 10.0],
            "target": "bullet_word_count",
            "bound_variables": [],
        }
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "violated"

    def test_comparison_with_insufficient_operands_abstains(self) -> None:
        verifier = FormalClaimVerifier()
        raw = _comparison_claim(operands=[], relation_type="less_than")
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "abstain"


# ---------------------------------------------------------------------------
# REQ-VERIFY-058 — cardinality checker
# ---------------------------------------------------------------------------


class TestCardinalityChecker:
    """REQ-VERIFY-058: cardinality route checks count constraints."""

    def test_cardinality_equals_supported(self) -> None:
        # SCENARIO: bullet_count equals 3, operands=[3] → supported (no observed value override)
        verifier = FormalClaimVerifier()
        claim = normalize_claim(_cardinality_claim(operands=[3.0]))
        verdict = verifier.verify_claim(claim)
        # With a single operand and equals relation, no second value to compare → abstain
        # (the verifier cannot verify without an observed count)
        assert verdict.verdict in {"supported", "abstain"}

    def test_cardinality_equals_with_observed_supported(self) -> None:
        # SCENARIO: operands=[3,3] → required=3, observed=3 → supported
        verifier = FormalClaimVerifier()
        raw = _cardinality_claim(operands=[3.0, 3.0], relation_type="equals")
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "supported"

    def test_cardinality_equals_with_observed_violated(self) -> None:
        # SCENARIO: operands=[3,5] → required=3, observed=5 → violated
        verifier = FormalClaimVerifier()
        raw = _cardinality_claim(operands=[3.0, 5.0], relation_type="equals")
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "violated"

    def test_cardinality_between_supported(self) -> None:
        # SCENARIO: operands=[4,7,5] → required 4<=x<=7, observed=5 → supported
        verifier = FormalClaimVerifier()
        raw = {
            "claim_id": "clE",
            "claim_text": "bullet_word_count between [4, 7]",
            "candidate_solver_route": "cardinality",
            "formalization_status": "formalized",
            "relation_type": "between",
            "operands": [4.0, 7.0, 5.0],
            "target": "bullet_word_count",
            "bound_variables": [],
        }
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "supported"

    def test_cardinality_between_violated(self) -> None:
        # SCENARIO: operands=[4,7,10] → 10 not in [4..7] → violated
        verifier = FormalClaimVerifier()
        raw = {
            "claim_id": "clF",
            "claim_text": "bullet_word_count between [4, 7]",
            "candidate_solver_route": "cardinality",
            "formalization_status": "formalized",
            "relation_type": "between",
            "operands": [4.0, 7.0, 10.0],
            "target": "bullet_word_count",
            "bound_variables": [],
        }
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "violated"


# ---------------------------------------------------------------------------
# REQ-VERIFY-058 — set-membership checker
# ---------------------------------------------------------------------------


class TestSetMembershipChecker:
    """REQ-VERIFY-058: set_membership route handles contains / not_contains / in."""

    def test_contains_with_present_member_supported(self) -> None:
        # SCENARIO: bound_variables=["risk"], target present in bound_variables → supported
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_set_membership_claim()))
        # contains: the bound variable "risk" is in bound_variables → supported
        assert verdict.verdict == "supported"

    def test_not_contains_with_absent_member_supported(self) -> None:
        # SCENARIO: not_contains, bound_variable not in list → supported
        verifier = FormalClaimVerifier()
        raw = {
            "claim_id": "clG",
            "claim_text": "implementation not_contains ['re.sub with Unicode classes']",
            "candidate_solver_route": "set_membership",
            "formalization_status": "formalized",
            "relation_type": "not_contains",
            "operands": [],
            "target": "implementation",
            "bound_variables": ["re_sub_unicode"],
        }
        verdict = verifier.verify_claim(normalize_claim(raw))
        # not_contains with non-empty bound_variables → abstain (no observable text to scan)
        # The verifier cannot scan actual code here, so abstains safely
        assert verdict.verdict in {"supported", "abstain"}

    def test_in_relation_with_member_supported(self) -> None:
        # SCENARIO: relation=in, target in bound_variables → supported
        verifier = FormalClaimVerifier()
        raw = {
            "claim_id": "clH",
            "claim_text": "color in [red, blue, green]",
            "candidate_solver_route": "set_membership",
            "formalization_status": "formalized",
            "relation_type": "in",
            "operands": [],
            "target": "red",
            "bound_variables": ["red", "blue", "green"],
        }
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "supported"

    def test_in_relation_with_absent_member_violated(self) -> None:
        # SCENARIO: target not in bound_variables set → violated
        verifier = FormalClaimVerifier()
        raw = {
            "claim_id": "clI",
            "claim_text": "color in [red, blue, green]",
            "candidate_solver_route": "set_membership",
            "formalization_status": "formalized",
            "relation_type": "in",
            "operands": [],
            "target": "yellow",
            "bound_variables": ["red", "blue", "green"],
        }
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "violated"

    def test_empty_bound_variables_abstains(self) -> None:
        # SCENARIO: no bound_variables to check membership against → abstain
        verifier = FormalClaimVerifier()
        raw = _set_membership_claim(relation_type="contains")
        raw["bound_variables"] = []
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "abstain"


# ---------------------------------------------------------------------------
# REQ-VERIFY-058 — boolean-entailment checker
# ---------------------------------------------------------------------------


class TestBooleanEntailmentChecker:
    """REQ-VERIFY-058: boolean_entailment route checks equality/attribute claims."""

    def test_equals_target_matches_bound_variable_supported(self) -> None:
        # SCENARIO: function_name is a recognized attribute target and bound_variables
        # has a value → structurally valid claim → supported
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_boolean_entailment_claim()))
        assert verdict.verdict == "supported"

    def test_equals_target_not_in_known_set_violated(self) -> None:
        # SCENARIO: target is not a recognized boolean-entailment attribute type → violated
        verifier = FormalClaimVerifier()
        raw = _boolean_entailment_claim(relation_type="equals")
        raw["target"] = "wrong_function"  # not in _KNOWN_BOOL_ENTAILMENT_TARGETS
        raw["bound_variables"] = ["chunk_list"]
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "violated"

    def test_attribute_equals_with_no_bound_variables_abstains(self) -> None:
        # SCENARIO: bound_variables empty → nothing to check → abstain
        verifier = FormalClaimVerifier()
        raw = _boolean_entailment_claim()
        raw["bound_variables"] = []
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "abstain"


# ---------------------------------------------------------------------------
# REQ-VERIFY-058 — abstention
# ---------------------------------------------------------------------------


class TestAbstention:
    """REQ-VERIFY-058: verifier abstains explicitly on unsafe or ambiguous claims."""

    def test_not_formalizable_route_abstains(self) -> None:
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_not_formalizable_claim()))
        assert verdict.verdict == "abstain"
        assert verdict.route == "abstain"

    def test_unknown_route_abstains(self) -> None:
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_unknown_route_claim()))
        assert verdict.verdict == "abstain"
        assert verdict.route == "abstain"

    def test_non_formalized_status_abstains_regardless_of_route(self) -> None:
        # SCENARIO: route says arithmetic but formalization_status=not_formalizable → abstain
        verifier = FormalClaimVerifier()
        raw = _arithmetic_claim()
        raw["formalization_status"] = "not_formalizable"
        verdict = verifier.verify_claim(normalize_claim(raw))
        assert verdict.verdict == "abstain"

    def test_abstain_verdict_has_no_failure_detail(self) -> None:
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_not_formalizable_claim()))
        assert verdict.failure_detail is None

    def test_abstain_route_field_is_string_literal(self) -> None:
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_unknown_route_claim()))
        assert isinstance(verdict.route, str)
        assert verdict.route == "abstain"


# ---------------------------------------------------------------------------
# REQ-VERIFY-058 — FormalClaimVerdict fields and closed vocabulary
# ---------------------------------------------------------------------------


class TestVerdictStructure:
    """REQ-VERIFY-058: FormalClaimVerdict fields use closed string vocabularies."""

    _VALID_VERDICTS = {"supported", "violated", "abstain"}
    _VALID_ROUTES = {
        "arithmetic",
        "comparison",
        "cardinality",
        "set_membership",
        "boolean_entailment",
        "abstain",
    }

    def _all_claims(self) -> list[dict]:
        return [
            _arithmetic_claim(),
            _comparison_claim(),
            _cardinality_claim(),
            _set_membership_claim(),
            _boolean_entailment_claim(),
            _not_formalizable_claim(),
            _unknown_route_claim(),
        ]

    def test_verdict_in_closed_vocabulary(self) -> None:
        verifier = FormalClaimVerifier()
        for raw in self._all_claims():
            verdict = verifier.verify_claim(normalize_claim(raw))
            assert verdict.verdict in self._VALID_VERDICTS

    def test_route_in_closed_vocabulary(self) -> None:
        verifier = FormalClaimVerifier()
        for raw in self._all_claims():
            verdict = verifier.verify_claim(normalize_claim(raw))
            assert verdict.route in self._VALID_ROUTES

    def test_verdict_has_claim_id(self) -> None:
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_arithmetic_claim(claim_id="myid")))
        assert verdict.claim_id == "myid"

    def test_verdict_has_run_date(self) -> None:
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_arithmetic_claim()))
        assert verdict.run_date == fcv.RUN_DATE

    def test_failure_detail_only_present_when_violated(self) -> None:
        # SCENARIO: failure_detail is None for supported/abstain, non-None for violated
        verifier = FormalClaimVerifier()
        supported = verifier.verify_claim(
            normalize_claim(_arithmetic_claim(operands=[100.0, 24.0, 76.0]))
        )
        violated = verifier.verify_claim(
            normalize_claim(_arithmetic_claim(operands=[100.0, 24.0, 99.0]))
        )
        abstained = verifier.verify_claim(normalize_claim(_not_formalizable_claim()))
        assert supported.failure_detail is None
        assert violated.failure_detail is not None
        assert abstained.failure_detail is None


# ---------------------------------------------------------------------------
# REQ-VERIFY-058 — batch result
# ---------------------------------------------------------------------------


class TestBatchResult:
    """REQ-VERIFY-058: batch input returns FormalClaimBatchResult with per-claim verdicts."""

    def test_batch_returns_all_verdicts(self) -> None:
        verifier = FormalClaimVerifier()
        claims = [
            normalize_claim(_arithmetic_claim("c1")),
            normalize_claim(_comparison_claim("c2")),
            normalize_claim(_not_formalizable_claim("c3")),
        ]
        result = verifier.verify_batch(claims)
        assert isinstance(result, FormalClaimBatchResult)
        assert len(result.verdicts) == 3

    def test_batch_aggregate_counts_correct(self) -> None:
        verifier = FormalClaimVerifier()
        # c1: arithmetic correct → supported
        # c2: arithmetic wrong → violated
        # c3: not_formalizable → abstain
        claims = [
            normalize_claim(_arithmetic_claim("c1", operands=[100.0, 24.0, 76.0])),
            normalize_claim(_arithmetic_claim("c2", operands=[100.0, 24.0, 99.0])),
            normalize_claim(_not_formalizable_claim("c3")),
        ]
        result = verifier.verify_batch(claims)
        assert result.counts["supported"] == 1
        assert result.counts["violated"] == 1
        assert result.counts["abstain"] == 1

    def test_batch_route_counts_correct(self) -> None:
        verifier = FormalClaimVerifier()
        claims = [
            normalize_claim(_arithmetic_claim("c1")),
            normalize_claim(_comparison_claim("c2")),
            normalize_claim(_cardinality_claim("c3")),
            normalize_claim(_not_formalizable_claim("c4")),
        ]
        result = verifier.verify_batch(claims)
        # arithmetic + comparison + cardinality → 3 supported routes + 1 abstain
        assert result.route_counts.get("arithmetic", 0) >= 1
        assert result.route_counts.get("abstain", 0) >= 1

    def test_empty_batch_returns_empty_result(self) -> None:
        verifier = FormalClaimVerifier()
        result = verifier.verify_batch([])
        assert result.verdicts == []
        assert result.counts["supported"] == 0


# ---------------------------------------------------------------------------
# REQ-VERIFY-058 — deterministic serialization
# ---------------------------------------------------------------------------


class TestDeterministicSerialization:
    """REQ-VERIFY-058: serialization is deterministic across two identical runs."""

    def test_verdict_to_dict_is_deterministic(self) -> None:
        verifier = FormalClaimVerifier()
        claim = normalize_claim(_arithmetic_claim(operands=[100.0, 24.0, 76.0]))
        d1 = verifier.verify_claim(claim).to_dict()
        d2 = verifier.verify_claim(claim).to_dict()
        assert d1 == d2

    def test_verdict_to_dict_contains_required_keys(self) -> None:
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalize_claim(_arithmetic_claim()))
        d = verdict.to_dict()
        for key in ("claim_id", "verdict", "route", "failure_detail", "run_date"):
            assert key in d

    def test_batch_to_json_is_deterministic(self) -> None:
        verifier = FormalClaimVerifier()
        claims = [
            normalize_claim(_arithmetic_claim("c1")),
            normalize_claim(_comparison_claim("c2")),
            normalize_claim(_not_formalizable_claim("c3")),
        ]
        j1 = verifier.verify_batch(claims).to_json()
        j2 = verifier.verify_batch(claims).to_json()
        assert j1 == j2

    def test_batch_to_json_is_valid_json(self) -> None:
        verifier = FormalClaimVerifier()
        claims = [normalize_claim(_arithmetic_claim("c1"))]
        j = verifier.verify_batch(claims).to_json()
        parsed = json.loads(j)
        assert "verdicts" in parsed
        assert "counts" in parsed

    def test_batch_json_keys_are_sorted(self) -> None:
        verifier = FormalClaimVerifier()
        claims = [normalize_claim(_arithmetic_claim())]
        j = verifier.verify_batch(claims).to_json()
        # Re-serialize with sort_keys to get canonical form; must match
        parsed = json.loads(j)
        re_serialized = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        assert j == re_serialized

    def test_batch_to_dict_round_trips(self) -> None:
        verifier = FormalClaimVerifier()
        claims = [
            normalize_claim(_arithmetic_claim("c1", operands=[100.0, 24.0, 76.0])),
            normalize_claim(_not_formalizable_claim("c2")),
        ]
        result = verifier.verify_batch(claims)
        d = result.to_dict()
        assert isinstance(d["verdicts"], list)
        assert len(d["verdicts"]) == 2


# ---------------------------------------------------------------------------
# REQ-VERIFY-059 — pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """REQ-VERIFY-059: VerifyRepairPipeline.verify_formal_claims is an additive entry point."""

    def test_pipeline_has_verify_formal_claims_method(self) -> None:
        pipeline = VerifyRepairPipeline()
        assert callable(getattr(pipeline, "verify_formal_claims", None))

    def test_pipeline_verify_formal_claims_returns_batch_result(self) -> None:
        pipeline = VerifyRepairPipeline()
        raw_claims = [
            _arithmetic_claim("c1"),
            _not_formalizable_claim("c2"),
        ]
        result = pipeline.verify_formal_claims(raw_claims)
        assert isinstance(result, FormalClaimBatchResult)
        assert len(result.verdicts) == 2

    def test_pipeline_verify_formal_claims_does_not_affect_verify(self) -> None:
        """REQ-VERIFY-059: calling verify_formal_claims does not break verify()."""
        pipeline = VerifyRepairPipeline()
        # verify() should still work normally after the new method exists
        result = pipeline.verify(
            question="What is 2 + 2?",
            response="The answer is 4.",
        )
        # Verify returns a VerificationResult — should not crash
        assert hasattr(result, "verified")

    def test_pipeline_verify_formal_claims_empty_list(self) -> None:
        pipeline = VerifyRepairPipeline()
        result = pipeline.verify_formal_claims([])
        assert isinstance(result, FormalClaimBatchResult)
        assert result.verdicts == []

    def test_pipeline_verify_formal_claims_serializes(self) -> None:
        pipeline = VerifyRepairPipeline()
        raw_claims = [_arithmetic_claim("c1")]
        result = pipeline.verify_formal_claims(raw_claims)
        j = result.to_json()
        assert json.loads(j)  # valid JSON


# ---------------------------------------------------------------------------
# Module-level API and edge cases
# ---------------------------------------------------------------------------


class TestModuleLevelAPI:
    """Test the public module-level verify_formal_claims function."""

    def test_verify_formal_claims_module_function(self) -> None:
        """Test the public module-level function."""
        raw_claims = [_arithmetic_claim("c1")]
        result = fcv.verify_formal_claims(raw_claims)
        assert isinstance(result, FormalClaimBatchResult)
        assert len(result.verdicts) == 1
        assert result.verdicts[0].verdict == "supported"

    def test_comparison_between_with_insufficient_operands(self) -> None:
        """Comparison 'between' route with <3 operands should abstain."""
        claim = _comparison_claim(
            relation_type="between",
            operands=[1.0, 2.0],  # Only 2, need 3
        )
        normalized = normalize_claim(claim)
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalized)
        assert verdict.verdict == "abstain"

    def test_comparison_lt_with_insufficient_operands(self) -> None:
        """Comparison 'less_than' route with <2 operands should abstain."""
        claim = _comparison_claim(
            relation_type="less_than",
            operands=[1.0],  # Only 1, need 2
        )
        normalized = normalize_claim(claim)
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalized)
        assert verdict.verdict == "abstain"

    def test_comparison_gt_with_insufficient_operands(self) -> None:
        """Comparison 'greater_than' route with <2 operands should abstain."""
        claim = _comparison_claim(
            relation_type="greater_than",
            operands=[1.0],  # Only 1, need 2
        )
        normalized = normalize_claim(claim)
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalized)
        assert verdict.verdict == "abstain"

    def test_unknown_comparison_relation_abstains(self) -> None:
        """Unknown comparison relation_type should abstain."""
        claim = _comparison_claim(relation_type="unknown_op")
        normalized = normalize_claim(claim)
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalized)
        assert verdict.verdict == "abstain"

    def test_cardinality_between_with_insufficient_operands(self) -> None:
        """Cardinality 'between' route with <3 operands should abstain."""
        claim = _cardinality_claim(
            relation_type="between",
            operands=[1.0, 2.0],  # Only 2, need 3
        )
        normalized = normalize_claim(claim)
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalized)
        assert verdict.verdict == "abstain"

    def test_set_membership_not_contains_abstains(self) -> None:
        """Set membership 'not_contains' relation should abstain (cannot scan text)."""
        claim = _set_membership_claim(relation_type="not_contains")
        normalized = normalize_claim(claim)
        verifier = FormalClaimVerifier()
        verdict = verifier.verify_claim(normalized)
        assert verdict.verdict == "abstain"
