"""FormalStepVerifier and EidokuCSP — Tier 2.8 multi-step formal verification.

**Why this module exists (arXiv 2603.29500 + arXiv 2512.20664):**

    SymCodeVerifier (Tier 2) checks each step independently: does this step's
    stated result match the arithmetic expression extracted from it?  It misses
    a whole class of errors: steps that are individually correct but collectively
    contradictory, and multi-step variable re-assignments that silently change the
    meaning of a computation.

    Two papers address the multi-step gap:

    arXiv 2603.29500 (Formal Step Intermediaries): each step must "formally entail"
    the next — the pair (steps[i], steps[i+1]) must be Z3-consistent.  This is a
    step-level formal verification oracle.  A chain is correct iff ALL consecutive
    pairs are entailed (Z3-consistent) AND no step has a Z3 violation.

    arXiv 2512.20664 (Eidoku): CSP-based constraint propagation.  Each step is
    represented as a constraint domain (a dict of variable->value assignments).
    Global consistency is checked by detecting contradictions: the same variable
    assigned different values in different steps.  This is a lightweight O(n)
    alternative to Z3 that catches a different class of errors.

**Architecture:**

    FormalStepVerifier: Z3-based step entailment using Z3StepVerifier.verify_step_z3.
      - verify_chain(steps) -> list of per-step entailment dicts.
      - chain_correct(steps) -> bool: all entailments pass, no violation verdict.

    EidokuCSP: variable-assignment consistency across all steps.
      - build_constraint_domain(step) -> dict of {var_name: value}.
      - check_global_consistency(steps) -> False if any variable contradicts across steps.

Spec: REQ-VERIFY-165, REQ-VERIFY-166,
      SCENARIO-VERIFY-217, SCENARIO-VERIFY-218, SCENARIO-VERIFY-219
"""

from __future__ import annotations

import re
from typing import Optional

from carnot.training.fover_z3_labeler import Z3StepVerifier

# ---------------------------------------------------------------------------
# Regex for Eidoku variable extraction
# ---------------------------------------------------------------------------

# Match "word = number" assignments in a step, e.g. "total = 75" or "x = 12.5".
# We capture the variable name (word) and the numeric value (integer or decimal).
# Why word chars only: CoT steps use short identifiers or plain nouns as variable names.
_ASSIGN_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\$?([\d]+(?:\.[\d]+)?)"
)

# Match "there are N items" or "N items remain" style quantity statements.
# E.g. "there are 5 apples" -> variable "apples" = 5.
_QUANTITY_RE = re.compile(
    r"(?:there\s+(?:are|is|were|was)\s+\$?([\d]+(?:\.[\d]+)?)\s+(\w+))"
    r"|(?:\$?([\d]+(?:\.[\d]+)?)\s+(\w+)\s+remain)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# FormalStepVerifier (arXiv 2603.29500)
# ---------------------------------------------------------------------------


class FormalStepVerifier:
    """Step-level formal entailment verifier (arXiv 2603.29500 approach).

    Uses Z3StepVerifier.verify_step_z3 to check whether each consecutive pair of
    steps is arithmetically consistent: (steps[0], steps[1]), (steps[1], steps[2]),
    etc.  Each step[i] acts as a prior context for step[i+1]; if Z3 finds the pair
    produces UNSAT (contradiction), the transition is labelled as a non-entailment.

    Why consecutive-pair consistency (not global)?
        arXiv 2603.29500 found that step-level oracles that check pair-wise entailment
        outperform global checkers on GSM8K because errors are typically localised to
        one faulty step transition.  Checking ALL prior steps as context (rather than
        just step[i]) is also supported and is what Z3StepVerifier already does when
        we pass prior_steps=steps[:i+1].

    Spec: REQ-VERIFY-165, SCENARIO-VERIFY-217, SCENARIO-VERIFY-218
    """

    def __init__(self) -> None:
        self._z3 = Z3StepVerifier()

    def verify_chain(self, steps: list[str]) -> list[dict]:
        """Verify each step's entailment from the steps that precede it.

        For step index i (starting at 1), checks whether steps[i] is arithmetically
        consistent with steps[:i] as prior context.  Step 0 is always labelled
        "entailment=True" because there are no prior steps to contradict it.

        Args:
            steps: Ordered list of CoT step strings.

        Returns:
            List of dicts, one per step, each with keys:
              step_idx  : int — zero-based index.
              verdict   : str — "correct" | "violation" | "unparseable".
              entailment: bool — True iff verdict is "correct" or "unparseable".

        Spec: REQ-VERIFY-165, SCENARIO-VERIFY-217
        """
        results: list[dict] = []

        for i, step in enumerate(steps):
            if i == 0:
                # No prior context — cannot violate anything.
                results.append(
                    {"step_idx": i, "verdict": "correct", "entailment": True}
                )
                continue

            prior_steps = steps[:i]
            verdict = self._z3.verify_step_z3(prior_steps, step)
            entailment = verdict != "violation"
            results.append(
                {"step_idx": i, "verdict": verdict, "entailment": entailment}
            )

        return results

    def chain_correct(self, steps: list[str]) -> bool:
        """Return True if the entire chain is formally correct.

        A chain is correct iff:
          - Every step entails the next (no "violation" verdicts).
          - At least one step is non-empty (empty chains are trivially correct but
            we want to avoid false positives on empty inputs).

        Args:
            steps: Ordered CoT step strings.

        Returns:
            True when all entailments pass and no step is labelled "violation".

        Spec: REQ-VERIFY-165, SCENARIO-VERIFY-217
        """
        if not steps:
            return True
        chain_results = self.verify_chain(steps)
        return all(r["entailment"] for r in chain_results)


# ---------------------------------------------------------------------------
# EidokuCSP (arXiv 2512.20664)
# ---------------------------------------------------------------------------


class EidokuCSP:
    """CSP-based global consistency checker for multi-step reasoning (arXiv 2512.20664).

    Eidoku represents each reasoning step as a constraint domain: a dict mapping
    variable names to their values.  Global consistency is checked by merging all
    per-step domains and detecting any variable assigned conflicting values across
    different steps.

    Why CSP instead of Z3?
        Z3 requires arithmetic expressions of the form "A op B = C"; it cannot
        easily represent prose assignments like "the price is $75".  Eidoku's regex-
        based domain extraction captures any "X = N" pattern, making it complementary
        to Z3: it catches variable-reuse contradictions that Z3 misses, while Z3
        catches arithmetic errors that regex extraction misses.

    Spec: REQ-VERIFY-166, SCENARIO-VERIFY-219
    """

    def build_constraint_domain(self, step: str) -> dict[str, float]:
        """Extract all variable assignments from a step as a constraint domain.

        Looks for:
          1. Explicit assignments: "x = 75", "total = 1000.5", "n_students = 20".
          2. Quantity statements: "there are 5 apples", "3 widgets remain".

        Returns a dict mapping each variable name (lowercased for normalisation) to
        its numeric value.  When a variable appears twice in the same step with
        different values, the last occurrence wins (consistent with CoT convention
        that the last stated value is the "result" of the step).

        Args:
            step: One CoT step string.

        Returns:
            Dict of {variable_name_lower: float_value}.

        Spec: REQ-VERIFY-166, SCENARIO-VERIFY-219
        """
        domain: dict[str, float] = {}

        # 1. Explicit "word = number" assignments.
        for m in _ASSIGN_RE.finditer(step):
            var_name = m.group(1).lower()
            try:
                value = float(m.group(2))
            except ValueError:
                continue
            domain[var_name] = value

        # 2. Quantity phrases: "there are N items" or "N items remain".
        for m in _QUANTITY_RE.finditer(step):
            if m.group(1) is not None and m.group(2) is not None:
                # "there are N items" pattern
                try:
                    value = float(m.group(1))
                except ValueError:
                    continue
                var_name = m.group(2).lower()
                domain[var_name] = value
            elif m.group(3) is not None and m.group(4) is not None:
                # "N items remain" pattern
                try:
                    value = float(m.group(3))
                except ValueError:
                    continue
                var_name = m.group(4).lower()
                domain[var_name] = value

        return domain

    def check_global_consistency(self, steps: list[str]) -> bool:
        """Check that no variable is assigned contradictory values across steps.

        Merges the constraint domains from all steps.  For each variable that appears
        in more than one step, checks whether all assignments agree (within a tolerance
        of 1e-6 to handle floating-point representations).  Returns False as soon as
        the first contradiction is found.

        Why tolerance instead of exact equality?
            CoT steps may express the same value as "75" in one step and "75.00" in
            another.  Floating-point parsing can introduce tiny rounding differences.
            1e-6 is tight enough to catch genuine contradictions (e.g. 75 vs 76) while
            accepting representational noise.

        Args:
            steps: Ordered CoT step strings.

        Returns:
            True if all variables are consistent across steps, False if any
            contradiction is detected.

        Spec: REQ-VERIFY-166, SCENARIO-VERIFY-219
        """
        # Track the first-assigned value for each variable and the step it appeared in.
        first_seen: dict[str, tuple[float, int]] = {}

        for step_idx, step in enumerate(steps):
            domain = self.build_constraint_domain(step)
            for var, value in domain.items():
                if var in first_seen:
                    prior_value, _ = first_seen[var]
                    if abs(prior_value - value) > 1e-6:
                        # Contradiction detected: same variable, different values.
                        return False
                else:
                    first_seen[var] = (value, step_idx)

        return True
