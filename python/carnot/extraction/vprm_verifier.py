"""VPRM Arithmetic Verifier — rule-based arithmetic step checking immune to reward hacking.

**Why rule-based over neural (the key insight from arXiv 2601.17223):**

    Neural verifiers (reward models, judge LLMs) can be fooled by adversarial inputs.
    A model trained to say "correct" can be coaxed into saying "correct" for wrong answers
    through reward hacking, where the model learns the surface features of correct answers
    rather than the underlying arithmetic.  A deterministic rule CANNOT be hacked — it
    either applies the arithmetic identity or it doesn't.

    VPRM (Verifiable Process Reward Models) replaces neural judges for arithmetic
    step verification with six families of deterministic rules:
    addition, subtraction, multiplication, division, percentage, unit consistency.

    Published improvement: 20% F1 gain over neural process reward models (arXiv 2601.17223).

**Why VPRM complements VeriCoT (different failure modes):**

    VeriCoT (arXiv 2511.04662) catches LOGICAL inconsistency across steps — it extracts
    FOL premises via an LLM call, then feeds them to Z3 for satisfiability checking.
    VeriCoT's weak point is the LLM extraction step: an LLM adds latency and can
    hallucinate premises that don't match the step text.

    VPRM catches ARITHMETIC errors within individual steps — no LLM call, purely
    deterministic regex + arithmetic identity checks.  VPRM's weak point is coverage:
    it only detects claims in patterns it was written to recognize.

    Together:
    - VPRM: "does the arithmetic in this step compute correctly?" (deterministic, fast)
    - VeriCoT: "are the logical claims across steps mutually consistent?" (LLM-assisted)
    - ArithmeticExtractor: "is there an explicit 'a OP b = c' equation?" (regex, base model)

Spec: REQ-EXTRACT-027, REQ-EXTRACT-028, REQ-EXTRACT-029,
      SCENARIO-EXTRACT-052, SCENARIO-EXTRACT-053, SCENARIO-EXTRACT-054
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# RuleVerdict — result of one rule's check on one step
# ---------------------------------------------------------------------------


@dataclass
class RuleVerdict:
    """Result of applying one arithmetic rule to one reasoning step.

    Attributes
    ----------
    rule_name : str
        Which rule produced this verdict (e.g. ``'addition'``, ``'percentage'``).
    passed : bool
        ``True`` iff the arithmetic identity holds — the stated result matches
        the computed result within floating-point tolerance (1e-6 relative error).
        ``False`` iff a mismatch was detected (arithmetic error).
    computed_value : float | None
        The value the rule computed from the operands.  ``None`` if the rule
        could not extract a computable expression.
    stated_value : float | None
        The value stated in the text as the result.  ``None`` if not found.
    error_magnitude : float | None
        ``abs(computed_value - stated_value)`` when both are available.
        ``None`` otherwise.  Useful for ranking violations by severity.

    Why include computed_value and stated_value?
        Downstream repair logic needs to know WHAT the correct value is, not just
        that an error exists.  Providing ``computed_value`` allows the repair loop
        to substitute the correct value without recomputing.

    Spec: REQ-EXTRACT-027, SCENARIO-EXTRACT-052
    """

    rule_name: str
    passed: bool
    computed_value: float | None
    stated_value: float | None
    error_magnitude: float | None


# ---------------------------------------------------------------------------
# ArithmeticRule — abstract base
# ---------------------------------------------------------------------------

# Tolerance for floating-point comparisons (handles e.g. 1/3 ≈ 0.333)
_FLOAT_TOL = 1e-6

# Shared result-clause pattern: "gives N", "gives us N", "equals N", "is N", "= N"
_RESULT_RE = re.compile(
    r"(?:gives\s+us|gives|equals|is|=)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _is_close(a: float, b: float) -> bool:
    """Return True iff a and b agree within _FLOAT_TOL relative error (or absolute 1e-9).

    Why not use math.isclose directly?
        math.isclose requires both absolute AND relative tolerance tuning.
        For arithmetic verification, we want to catch errors of magnitude >= 1
        (a wrong integer answer) while ignoring floating-point rounding.
        The combined test below handles both near-zero values (abs tol) and
        large values (rel tol) correctly.
    """
    abs_diff = abs(a - b)
    if abs_diff < 1e-9:
        return True
    denom = max(abs(a), abs(b), 1.0)
    return abs_diff / denom < _FLOAT_TOL


class ArithmeticRule(ABC):
    """Abstract base for deterministic arithmetic step-checking rules.

    Each subclass recognises one family of arithmetic claims in natural-language
    IT model output and verifies the stated result against the computed result.

    Why abstract over six rule types instead of one monolithic extractor?
        Each rule family has distinct prose patterns and correctness semantics.
        An abstract base lets downstream code treat all rules uniformly while
        keeping the pattern logic isolated and independently testable.

    Spec: REQ-EXTRACT-027
    """

    @abstractmethod
    def check(self, step_text: str) -> RuleVerdict | None:
        """Apply this rule to a single reasoning step.

        Parameters
        ----------
        step_text : str
            One natural-language reasoning step (a single sentence or clause).

        Returns
        -------
        RuleVerdict | None
            ``RuleVerdict`` when the rule's pattern matched the step text.
            ``None`` when the pattern did not match (rule not applicable).

        Why Optional return instead of always returning a verdict?
            Returning None signals "this rule has nothing to say about this step".
            A rule that always returns a verdict would flood verify_step() with
            spurious "no-match" verdicts, making it impossible to distinguish
            "rule found an error" from "rule couldn't parse the step".
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# AdditionRule
# ---------------------------------------------------------------------------

_ADD_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s+(?:plus|added to)\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class AdditionRule(ArithmeticRule):
    """Detect incorrect addition claims in IT-style natural language.

    Matches patterns like:
    - "47 plus 28 equals 75"
    - "the total is 100 added to 50 gives 150"

    Why prose patterns rather than the '47 + 28 = 75' regex from ArithmeticExtractor?
        Instruction-tuned models rarely write 'a + b = c'; they write '47 plus 28
        gives 75'.  ArithmeticExtractor finds ZERO violations on IT output because
        its regex requires the equation-style syntax.  This rule covers IT prose.

    Spec: REQ-EXTRACT-027, SCENARIO-EXTRACT-052, SCENARIO-EXTRACT-053
    """

    def check(self, step_text: str) -> RuleVerdict | None:
        """Match 'A plus/added to B [equals/gives/is] C' and verify A + B == C."""
        op_match = _ADD_RE.search(step_text)
        if not op_match:
            return None
        res_match = _RESULT_RE.search(step_text, op_match.end())
        if not res_match:
            return None

        a = float(op_match.group(1))
        b = float(op_match.group(2))
        c = float(res_match.group(1))
        computed = a + b

        return RuleVerdict(
            rule_name="addition",
            passed=_is_close(computed, c),
            computed_value=computed,
            stated_value=c,
            error_magnitude=abs(computed - c),
        )


# ---------------------------------------------------------------------------
# SubtractionRule
# ---------------------------------------------------------------------------

_SUB_MINUS_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s+(?:minus|subtracted by)\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_SUB_FROM_RE = re.compile(
    r"subtract(?:ing)?\s+(-?\d+(?:\.\d+)?)\s+from\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class SubtractionRule(ArithmeticRule):
    """Detect incorrect subtraction claims in IT-style natural language.

    Matches:
    - "100 minus 15 gives 85"
    - "subtracting 15 from 100 gives 85"

    Spec: REQ-EXTRACT-027
    """

    def check(self, step_text: str) -> RuleVerdict | None:
        """Match subtraction prose and verify A - B == C."""
        # Try 'A minus B' first
        op_match = _SUB_MINUS_RE.search(step_text)
        if op_match:
            a = float(op_match.group(1))
            b = float(op_match.group(2))
        else:
            # Try 'subtracting B from A' (note: operand order is swapped)
            op_match = _SUB_FROM_RE.search(step_text)
            if not op_match:
                return None
            b = float(op_match.group(1))  # the subtrahend
            a = float(op_match.group(2))  # the minuend

        res_match = _RESULT_RE.search(step_text, op_match.end())
        if not res_match:
            return None

        c = float(res_match.group(1))
        computed = a - b

        return RuleVerdict(
            rule_name="subtraction",
            passed=_is_close(computed, c),
            computed_value=computed,
            stated_value=c,
            error_magnitude=abs(computed - c),
        )


# ---------------------------------------------------------------------------
# MultiplicationRule
# ---------------------------------------------------------------------------

_MUL_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s+(?:times|multiplied by)\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class MultiplicationRule(ArithmeticRule):
    """Detect incorrect multiplication claims in IT-style natural language.

    Matches:
    - "5 times 6 gives us 30"
    - "7 multiplied by 8 equals 56"

    Spec: REQ-EXTRACT-027, SCENARIO-EXTRACT-052
    """

    def check(self, step_text: str) -> RuleVerdict | None:
        """Match multiplication prose and verify A * B == C."""
        op_match = _MUL_RE.search(step_text)
        if not op_match:
            return None
        res_match = _RESULT_RE.search(step_text, op_match.end())
        if not res_match:
            return None

        a = float(op_match.group(1))
        b = float(op_match.group(2))
        c = float(res_match.group(1))
        computed = a * b

        return RuleVerdict(
            rule_name="multiplication",
            passed=_is_close(computed, c),
            computed_value=computed,
            stated_value=c,
            error_magnitude=abs(computed - c),
        )


# ---------------------------------------------------------------------------
# DivisionRule
# ---------------------------------------------------------------------------

_DIV_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s+divided by\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class DivisionRule(ArithmeticRule):
    """Detect incorrect division claims in IT-style natural language.

    Matches:
    - "100 divided by 4 gives 25"
    - "20 divided by 5 equals 4"

    Division by zero is silently skipped (returns None) because it is
    undefined and cannot be verified against a numeric result.

    Spec: REQ-EXTRACT-027
    """

    def check(self, step_text: str) -> RuleVerdict | None:
        """Match division prose and verify A / B == C."""
        op_match = _DIV_RE.search(step_text)
        if not op_match:
            return None

        a = float(op_match.group(1))
        b = float(op_match.group(2))

        if abs(b) < 1e-12:
            # Division by zero is undefinable; don't flag as error
            return None

        res_match = _RESULT_RE.search(step_text, op_match.end())
        if not res_match:
            return None

        c = float(res_match.group(1))
        computed = a / b

        return RuleVerdict(
            rule_name="division",
            passed=_is_close(computed, c),
            computed_value=computed,
            stated_value=c,
            error_magnitude=abs(computed - c),
        )


# ---------------------------------------------------------------------------
# PercentageRule
# ---------------------------------------------------------------------------

_PCT_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)%\s+of\s+(-?\d+(?:\.\d+)?)\s+(?:is|equals|gives(?:\s+us)?)\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class PercentageRule(ArithmeticRule):
    """Detect incorrect percentage claims in IT-style natural language.

    Matches:
    - "20% of 50 is 10"
    - "15% of 200 equals 30"

    Why percentage needs its own rule?
        IT models frequently make percentage errors — the failure mode is confusing
        the divisor (X% of Y = X/100 * Y) with a different operation.  A dedicated
        rule catches "20% of 50 is 11" which an addition or multiplication rule
        would never match.

    Spec: REQ-EXTRACT-027, SCENARIO-EXTRACT-052, SCENARIO-EXTRACT-053
    """

    def check(self, step_text: str) -> RuleVerdict | None:
        """Match 'X% of Y is Z' and verify X/100 * Y == Z."""
        m = _PCT_RE.search(step_text)
        if not m:
            return None

        pct = float(m.group(1))
        base = float(m.group(2))
        c = float(m.group(3))
        computed = (pct / 100.0) * base

        return RuleVerdict(
            rule_name="percentage",
            passed=_is_close(computed, c),
            computed_value=computed,
            stated_value=c,
            error_magnitude=abs(computed - c),
        )


# ---------------------------------------------------------------------------
# UnitConsistencyRule
# ---------------------------------------------------------------------------

# Units that are commonly confused in arithmetic word problems
_UNIT_CLASSES: dict[str, set[str]] = {
    "distance": {"km", "m", "miles", "mile", "feet", "ft", "yards", "yd"},
    "mass": {"kg", "g", "lb", "lbs", "oz", "pounds", "pound"},
    "volume": {"liters", "liter", "litres", "litre", "ml", "gallons", "gallon"},
}

# Maps each unit string to its class name
_UNIT_TO_CLASS: dict[str, str] = {
    unit: cls for cls, units in _UNIT_CLASSES.items() for unit in units
}

# Matches "N unit [plus/minus] M unit" — captures both operand units
_UNIT_OP_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s+([a-zA-Z]+)\s+(?:plus|added to|minus|subtracted by)\s+"
    r"(\d+(?:\.\d+)?)\s+([a-zA-Z]+)",
    re.IGNORECASE,
)


class UnitConsistencyRule(ArithmeticRule):
    """Detect unit-mixing errors in arithmetic steps.

    Catches steps where two operands have different physical units in the same
    family (e.g., km and miles being added together without conversion).

    Matches:
    - "5 km plus 3 miles gives 8 km"  → FAIL (km ≠ miles, same distance family)
    - "5 km plus 3 km gives 8 km"     → PASS (consistent units)

    Why unit consistency matters:
        IT models trained on word problems sometimes conflate unit labels when
        producing reasoning traces.  Adding "5 km + 3 miles" is physically
        meaningless without a conversion factor; this rule flags it as an
        inconsistency without needing to know the conversion ratio.

    Spec: REQ-EXTRACT-027
    """

    def check(self, step_text: str) -> RuleVerdict | None:
        """Match binary operation with two unit labels and check unit class consistency."""
        m = _UNIT_OP_RE.search(step_text)
        if not m:
            return None

        unit_a = m.group(2).lower()
        unit_b = m.group(4).lower()

        class_a = _UNIT_TO_CLASS.get(unit_a)
        class_b = _UNIT_TO_CLASS.get(unit_b)

        # Only flag when both units are in the same physical family but are
        # different unit tokens (e.g. km vs miles, not km vs kg which are
        # different families and might be intentional in a multi-dimensional problem)
        if class_a is None or class_b is None:
            return None
        if class_a != class_b:
            return None

        units_match = unit_a == unit_b

        return RuleVerdict(
            rule_name="unit_consistency",
            passed=units_match,
            computed_value=None,
            stated_value=None,
            error_magnitude=None,
        )


# ---------------------------------------------------------------------------
# VPRMArithmeticVerifier — top-level API
# ---------------------------------------------------------------------------

_DEFAULT_RULES: list[ArithmeticRule] = [
    AdditionRule(),
    SubtractionRule(),
    MultiplicationRule(),
    DivisionRule(),
    PercentageRule(),
    UnitConsistencyRule(),
]

# Split on newlines or sentence-ending punctuation (same splitter as VeriCoT)
_STEP_SPLITTER = re.compile(r"\n+|(?<=[.?!;])\s+")


def _split_steps(cot_text: str) -> list[str]:
    """Split CoT text into individual reasoning steps.

    Replicates the VeriCoT splitter so VPRM and VeriCoT process the same step
    boundaries when run in combination.
    """
    parts = _STEP_SPLITTER.split(cot_text.strip())
    return [p.strip() for p in parts if p.strip()]


class VPRMArithmeticVerifier:
    """Deterministic rule-based arithmetic step verifier implementing VPRM.

    Applies six families of arithmetic rules to natural-language reasoning steps
    and returns structured verdicts without any LLM call.

    **Why deterministic over neural (arXiv 2601.17223)?**
        Process reward models that use a neural judge to rate reasoning steps
        can be reward-hacked: an adversarial input can fool the judge into
        rating a wrong step as correct.  VPRM rules are deterministic identity
        checks — the addition rule either finds A + B = C or it doesn't.  The
        only way to fool a VPRM rule is to avoid the patterns it recognizes,
        which means the rule simply returns None (no verdict) rather than a
        wrong verdict.

    **Complementary to VeriCoT (different failure modes, not competing):**
        VeriCoT (vericot_validator.py) catches multi-step LOGICAL inconsistency
        by translating steps to FOL and running Z3.  Its weakness is the LLM
        extraction step, which adds latency and hallucination risk.
        VPRM catches single-step ARITHMETIC errors with zero LLM overhead.
        Running both in sequence gives complementary coverage:
        VPRM first (fast, catches arithmetic), VeriCoT second (slower, catches logic).

    Parameters
    ----------
    rules : list[ArithmeticRule] | None
        List of rules to apply.  ``None`` (default) uses all six built-in rules:
        addition, subtraction, multiplication, division, percentage, unit consistency.

    Spec: REQ-EXTRACT-027, REQ-EXTRACT-028, REQ-EXTRACT-029,
          SCENARIO-EXTRACT-052, SCENARIO-EXTRACT-053, SCENARIO-EXTRACT-054
    """

    def __init__(self, rules: list[ArithmeticRule] | None = None) -> None:
        self.rules: list[ArithmeticRule] = (
            rules if rules is not None else _DEFAULT_RULES
        )

    def verify_step(self, step_text: str) -> list[RuleVerdict]:
        """Apply all rules to one step and return verdicts for rules that matched.

        Returns only verdicts where the rule's pattern was found in the step text
        (i.e., rules that returned non-None from check()).  Steps with no matching
        rules return an empty list.

        Parameters
        ----------
        step_text : str
            One natural-language reasoning step.

        Returns
        -------
        list[RuleVerdict]
            One RuleVerdict per rule that matched this step (passed or failed).
            Empty list means no arithmetic claim was recognizable by any rule.

        Spec: REQ-EXTRACT-028, SCENARIO-EXTRACT-052
        """
        verdicts: list[RuleVerdict] = []
        for rule in self.rules:
            verdict = rule.check(step_text)
            if verdict is not None:
                verdicts.append(verdict)
        return verdicts

    def detect_violations(self, cot_text: str) -> list[RuleVerdict]:
        """Detect arithmetic violations across all steps in a CoT trace.

        Splits the CoT text into steps, applies all rules to each step, and
        returns only verdicts where ``passed=False`` (arithmetic error detected).

        Empty return list means either all arithmetic is correct or no arithmetic
        claims were recognized by any rule.

        Parameters
        ----------
        cot_text : str
            Full chain-of-thought text from an IT model response.

        Returns
        -------
        list[RuleVerdict]
            Only failed verdicts (``passed=False``).

        Spec: REQ-EXTRACT-028, SCENARIO-EXTRACT-053
        """
        violations: list[RuleVerdict] = []
        for step in _split_steps(cot_text):
            for verdict in self.verify_step(step):
                if not verdict.passed:
                    violations.append(verdict)
        return violations

    @staticmethod
    def f1_score(ground_truth: list[bool], predicted: list[bool]) -> float:
        """Compute binary F1 score from ground truth and predicted violation flags.

        F1 = 2 * precision * recall / (precision + recall)

        ``True`` means "violation present" (positive class).
        ``False`` means "no violation" (negative class).

        Returns 0.0 when the denominator is zero (no positives predicted or
        no positives in ground truth), matching sklearn convention.

        Parameters
        ----------
        ground_truth : list[bool]
            One bool per sample: True iff the sample has an arithmetic error.
        predicted : list[bool]
            One bool per sample: True iff VPRM detected a violation.

        Spec: REQ-EXTRACT-029, SCENARIO-EXTRACT-054
        """
        if len(ground_truth) != len(predicted):
            raise ValueError(
                f"ground_truth length {len(ground_truth)} != "
                f"predicted length {len(predicted)}"
            )

        tp = sum(g and p for g, p in zip(ground_truth, predicted))
        fp = sum((not g) and p for g, p in zip(ground_truth, predicted))
        fn = sum(g and (not p) for g, p in zip(ground_truth, predicted))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)
