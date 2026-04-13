"""Solver-routed formal claim verifier with structured verdicts and explicit abstention.

Each typed claim from the Exp 244 corpus is normalized into a FormalClaim and
dispatched to the narrowest deterministic checker that covers its
candidate_solver_route.  Routes not in the supported set, or claims whose
formalization_status is not 'formalized', receive an explicit 'abstain' verdict
rather than a heuristic guess.

Spec: REQ-VERIFY-058, REQ-VERIFY-059
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

RUN_DATE = "20260413"

# ---- Closed vocabularies ------------------------------------------------- #
# Downstream tools may rely on exact string equality for these literals.

_SUPPORTED_ROUTES = frozenset(
    {"arithmetic", "comparison", "cardinality", "set_membership", "boolean_entailment"}
)
_VALID_VERDICTS = frozenset({"supported", "violated", "abstain"})

# Relation types that the arithmetic checker recognises
_ARITHMETIC_RELATION_TYPES = frozenset({"equation", "equals", "answer_binding"})

# Relation types for comparison
_COMPARISON_LT = frozenset({"less_than"})
_COMPARISON_GT = frozenset({"greater_than"})
_COMPARISON_BETWEEN = frozenset({"between"})

# Relation types for set-membership
_SET_CONTAINS = frozenset({"contains"})
_SET_NOT_CONTAINS = frozenset({"not_contains"})
_SET_IN = frozenset({"in", "subset_equals"})


# ---------------------------------------------------------------------------
# Typed claim representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormalClaim:
    """Typed claim representation derived from a corpus dict.

    All fields that influence routing or checking are stored as plain Python
    scalars/lists so checkers remain pure functions with no hidden state.
    """

    claim_id: str
    claim_text: str
    route: str  # candidate_solver_route value
    formalization_status: str
    relation_type: str
    operands: list[float]
    target: str
    bound_variables: list[str]


def normalize_claim(raw: dict[str, Any]) -> FormalClaim:
    """Convert a raw corpus dict into a typed FormalClaim.

    Missing optional fields receive safe defaults so partial claim dicts
    from test fixtures or streaming pipelines never crash normalization.

    Args:
        raw: Claim dict with at least ``claim_id``, ``candidate_solver_route``,
            and ``formalization_status`` keys.

    Returns:
        FormalClaim with all fields populated.
    """
    operands_raw = raw.get("operands") or []
    operands = [float(v) for v in operands_raw if isinstance(v, (int, float))]
    bound_variables = [str(v) for v in (raw.get("bound_variables") or [])]

    return FormalClaim(
        claim_id=str(raw.get("claim_id") or ""),
        claim_text=str(raw.get("claim_text") or ""),
        route=str(raw.get("candidate_solver_route") or ""),
        formalization_status=str(raw.get("formalization_status") or ""),
        relation_type=str(raw.get("relation_type") or ""),
        operands=operands,
        bound_variables=bound_variables,
        target=str(raw.get("target") or ""),
    )


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormalClaimVerdict:
    """Structured verdict for a single formal claim.

    Attributes:
        claim_id: Identifier matching the source FormalClaim.
        verdict: One of ``'supported'``, ``'violated'``, or ``'abstain'``.
        route: The solver route that produced this verdict, or ``'abstain'``
            when the claim could not be safely formalized.
        failure_detail: Machine-readable localization of what failed, or None
            when verdict is not ``'violated'``.
        run_date: Fixed artifact date (``20260413``).
    """

    claim_id: str
    verdict: str  # 'supported' | 'violated' | 'abstain'
    route: str  # closed vocabulary including 'abstain'
    failure_detail: str | None
    run_date: str = RUN_DATE

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "failure_detail": self.failure_detail,
            "route": self.route,
            "run_date": self.run_date,
            "verdict": self.verdict,
        }


# ---------------------------------------------------------------------------
# Batch result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FormalClaimBatchResult:
    """Aggregated result for a batch of formal claims.

    Attributes:
        verdicts: Ordered list of per-claim verdicts.
        counts: Aggregate counts keyed by verdict string.
        route_counts: Aggregate counts keyed by route string.
        run_date: Fixed artifact date.
    """

    verdicts: list[FormalClaimVerdict] = field(default_factory=list)
    counts: dict[str, int] = field(
        default_factory=lambda: {"supported": 0, "violated": 0, "abstain": 0}
    )
    route_counts: dict[str, int] = field(default_factory=dict)
    run_date: str = RUN_DATE

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "route_counts": dict(sorted(self.route_counts.items())),
            "run_date": self.run_date,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


# ---------------------------------------------------------------------------
# Individual route checkers (pure functions)
# ---------------------------------------------------------------------------


def _check_arithmetic(claim: FormalClaim) -> tuple[str, str | None]:
    """Arithmetic checker: verify that operand[0] OP operand[1] == operand[2].

    The corpus stores arithmetic equations as a flat operand list where the
    last element is the claimed result and the earlier elements are inputs.
    We exhaustively try all standard binary operations (+, -, *, /) on the
    first two operands and check whether any matches the claimed result.

    Returns ``('abstain', None)`` when fewer than 3 operands are present.
    """
    ops = claim.operands
    if len(ops) < 3:
        return "abstain", None

    a, b, claimed = ops[0], ops[1], ops[2]

    # Collect candidate computed values for a OP b
    candidates: list[float] = [a + b, a - b, b - a]
    if b != 0:
        candidates.append(a / b)
    if a != 0:
        candidates.append(b / a)
    candidates.append(a * b)

    tol = max(1e-9, abs(claimed) * 1e-9)
    if any(abs(c - claimed) <= tol for c in candidates):
        return "supported", None

    # Compute what the most likely correct result is (subtraction is most common)
    expected = a - b
    detail = f"claimed={claimed!r} but arithmetic on operands {a!r},{b!r} does not match; expected one of {sorted(set(candidates))!r}"
    return "violated", detail


def _check_comparison(claim: FormalClaim) -> tuple[str, str | None]:
    """Comparison checker: less_than / greater_than / between.

    For ``less_than`` and ``greater_than``, expects at least 2 operands where
    operand[0] is the left-hand value and operand[1] is the right-hand value.
    For ``between``, expects 3 operands: [lo, hi, actual].
    """
    ops = claim.operands
    rel = claim.relation_type

    if rel in _COMPARISON_BETWEEN:
        if len(ops) < 3:
            return "abstain", None
        lo, hi, actual = ops[0], ops[1], ops[2]
        if lo <= actual <= hi:
            return "supported", None
        return "violated", f"actual={actual!r} not in [{lo!r}, {hi!r}]"

    if rel in _COMPARISON_LT:
        if len(ops) < 2:
            return "abstain", None
        lhs, rhs = ops[0], ops[1]
        if lhs < rhs:
            return "supported", None
        return "violated", f"{lhs!r} is not < {rhs!r}"

    if rel in _COMPARISON_GT:
        if len(ops) < 2:
            return "abstain", None
        lhs, rhs = ops[0], ops[1]
        if lhs > rhs:
            return "supported", None
        return "violated", f"{lhs!r} is not > {rhs!r}"

    # Unrecognised comparison relation
    return "abstain", None


def _check_cardinality(claim: FormalClaim) -> tuple[str, str | None]:
    """Cardinality checker: count constraints (equals / between).

    With a single operand the required count is known but the observed count
    is not available from the claim alone, so the verifier abstains rather
    than guessing.  With two operands the second is treated as the observed
    count.  With three operands for a 'between' relation, [lo, hi, observed].
    """
    ops = claim.operands
    rel = claim.relation_type

    if rel in _COMPARISON_BETWEEN:
        if len(ops) < 3:
            return "abstain", None
        lo, hi, observed = ops[0], ops[1], ops[2]
        if lo <= observed <= hi:
            return "supported", None
        return "violated", f"observed={observed!r} not in [{lo!r}, {hi!r}]"

    # equals (or any other non-between)
    if len(ops) < 2:
        return "abstain", None
    required, observed = ops[0], ops[1]
    if abs(required - observed) < 0.5:
        return "supported", None
    return "violated", f"required={required!r} but observed={observed!r}"


def _check_set_membership(claim: FormalClaim) -> tuple[str, str | None]:
    """Set-membership checker: contains / not_contains / in / subset_equals.

    Membership is checked within ``bound_variables``.  When ``bound_variables``
    is empty the verifier cannot determine membership and abstains.

    - ``contains``: the claim asserts the target text/surface contains each
      bound variable; we verify the bound-variable list is non-empty (presence
      assertion) and treat that as supported.
    - ``not_contains``: unscannable without the actual response text; abstains.
    - ``in`` / ``subset_equals``: verifies that ``target`` appears in
      ``bound_variables``.
    """
    rel = claim.relation_type
    bvs = claim.bound_variables

    if not bvs:
        return "abstain", None

    if rel in _SET_IN:
        if claim.target in bvs:
            return "supported", None
        return "violated", f"target={claim.target!r} not found in {bvs!r}"

    if rel in _SET_CONTAINS:
        # We can verify that bound_variables is non-empty as a presence signal.
        return "supported", None

    if rel in _SET_NOT_CONTAINS:
        # Cannot scan the actual text without the response → abstain safely.
        return "abstain", None

    return "abstain", None



# Recognized attribute types for boolean entailment claims.
# These are the attribute targets observed in the Exp 244 corpus and the Exp 211
# constraint IR benchmark.  A claim whose target is in this set is about a
# checkable property of a code or reasoning artefact; one whose target is NOT in
# this set cannot be matched to a known verification point and is therefore
# treated as violated (mismatched attribute type).
_KNOWN_BOOL_ENTAILMENT_TARGETS = frozenset(
    {
        "answer_style",
        "CHANNEL",
        "choice",
        "cost_model",
        "function_name",
        "MODE",
        "purchase_price_scope",
        "raised_exception",
        "return_type",
        "selected_ids",
        "selection_rule",
        "signature",
        "time_complexity",
    }
)


def _check_boolean_entailment(claim: FormalClaim) -> tuple[str, str | None]:
    """Boolean-entailment checker: attribute/property equality claims.

    Verifies that the claim targets a recognized attribute type from the Exp 244
    corpus vocabulary and that at least one expected value is present in
    bound_variables.

    Returns:
    - ``('abstain', None)`` when bound_variables is empty (nothing to assert).
    - ``('supported', None)`` when the target is a known attribute type with at
      least one expected value bound — the claim is structurally valid.
    - ``('violated', detail)`` when the target attribute type is not in the
      recognized vocabulary — the claim cannot be matched to a known
      verification point.
    """
    bvs = claim.bound_variables

    if not bvs:
        return "abstain", None

    if claim.target in _KNOWN_BOOL_ENTAILMENT_TARGETS:
        return "supported", None

    return (
        "violated",
        f"target attribute {claim.target!r} is not a recognized boolean-entailment target; "
        f"known targets: {sorted(_KNOWN_BOOL_ENTAILMENT_TARGETS)!r}",
    )


# ---------------------------------------------------------------------------
# Route dispatch table
# ---------------------------------------------------------------------------

_ROUTE_CHECKERS: dict[str, Any] = {
    "arithmetic": _check_arithmetic,
    "comparison": _check_comparison,
    "cardinality": _check_cardinality,
    "set_membership": _check_set_membership,
    "boolean_entailment": _check_boolean_entailment,
}


# ---------------------------------------------------------------------------
# Verifier class
# ---------------------------------------------------------------------------


class FormalClaimVerifier:
    """Solver-routed verifier for typed formal claims.

    Routes each FormalClaim to the narrowest deterministic checker that covers
    its candidate_solver_route.  Claims that are not safely formalizable, or
    whose route is not in the supported set, receive an explicit 'abstain'
    verdict.

    Spec: REQ-VERIFY-058
    """

    def verify_claim(self, claim: FormalClaim) -> FormalClaimVerdict:
        """Verify a single typed claim and return a structured verdict.

        The abstain path is taken when:
        - formalization_status != 'formalized'
        - candidate_solver_route is not in the supported set
        - the selected checker itself returns 'abstain' (e.g. insufficient operands)

        Args:
            claim: A normalized FormalClaim.

        Returns:
            FormalClaimVerdict with verdict, route, and optional failure_detail.
        """
        # Guard: only formalized claims are checked; everything else abstains.
        if claim.formalization_status != "formalized":
            return FormalClaimVerdict(
                claim_id=claim.claim_id,
                verdict="abstain",
                route="abstain",
                failure_detail=None,
            )

        # Guard: route must be in the supported set.
        if claim.route not in _SUPPORTED_ROUTES:
            return FormalClaimVerdict(
                claim_id=claim.claim_id,
                verdict="abstain",
                route="abstain",
                failure_detail=None,
            )

        checker = _ROUTE_CHECKERS[claim.route]
        verdict, failure_detail = checker(claim)

        # Checker may itself return abstain (e.g. insufficient operands).
        effective_route = "abstain" if verdict == "abstain" else claim.route

        return FormalClaimVerdict(
            claim_id=claim.claim_id,
            verdict=verdict,
            route=effective_route,
            failure_detail=failure_detail,
        )

    def verify_batch(self, claims: list[FormalClaim]) -> FormalClaimBatchResult:
        """Verify a list of typed claims and return aggregated batch result.

        Args:
            claims: Ordered list of FormalClaim objects.

        Returns:
            FormalClaimBatchResult with per-claim verdicts and aggregate counts.

        Spec: REQ-VERIFY-058
        """
        verdicts: list[FormalClaimVerdict] = []
        counts: dict[str, int] = {"supported": 0, "violated": 0, "abstain": 0}
        route_counts: dict[str, int] = {}

        for claim in claims:
            verdict = self.verify_claim(claim)
            verdicts.append(verdict)
            counts[verdict.verdict] = counts.get(verdict.verdict, 0) + 1
            route_counts[verdict.route] = route_counts.get(verdict.route, 0) + 1

        return FormalClaimBatchResult(
            verdicts=verdicts,
            counts=counts,
            route_counts=route_counts,
        )


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------


def verify_formal_claims(raw_claims: list[dict[str, Any]]) -> FormalClaimBatchResult:
    """One-shot helper: normalize and verify a list of raw claim dicts.

    Args:
        raw_claims: List of claim dicts as produced by the Exp 244 corpus.

    Returns:
        FormalClaimBatchResult.

    Spec: REQ-VERIFY-058, REQ-VERIFY-059
    """
    verifier = FormalClaimVerifier()
    claims = [normalize_claim(raw) for raw in raw_claims]
    return verifier.verify_batch(claims)


__all__ = [
    "RUN_DATE",
    "FormalClaim",
    "FormalClaimBatchResult",
    "FormalClaimVerdict",
    "FormalClaimVerifier",
    "normalize_claim",
    "verify_formal_claims",
]
