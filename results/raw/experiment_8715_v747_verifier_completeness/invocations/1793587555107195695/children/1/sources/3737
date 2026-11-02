"""Tier 1 online constraint-addition agent: detects high-signal constraint types and
adds specialized constraints to the pipeline.

**Researcher summary:**
    Exp 134 showed that *reweighting* existing constraints does not improve precision.
    The fix (from research-program.md) is constraint ADDITION: when the agent sees that
    constraint type X fires in >60% of wrong responses but <20% of correct ones, it
    adds a new specialised constraint Y derived from X's firing context.

    This module provides ConstraintAdditionAgent, which:
    1. Observes verification results during a learning pass over labelled samples.
    2. Tracks per-constraint-type firing rates split by ground-truth label.
    3. Detects types that exceed the high-signal thresholds (wrong_rate > 0.6,
       correct_rate < 0.2) — these are the types to add.
    4. Returns a list of ConstraintResult objects representing the added constraints
       so the caller can inject them into VerifyRepairPipeline's active set.

**Why threshold-based detection (not continuous weighting):**
    A continuous weight adjustment (exp134) sees no improvement because the weight
    is multiplied against the same constraint set — if a type is missing, raising its
    weight to infinity still adds zero detections.  The threshold triggers only when
    empirical evidence is strong enough to justify adding a *new* constraint type,
    preventing noise-driven proliferation.

    The 60%/20% thresholds were chosen to require a 3:1 signal-to-noise ratio.  A
    type that fires equally on wrong and correct (50%/50%) is pure noise; one that
    fires 83% wrong / 15% correct (as arithmetic violations do on FoVer) is clearly
    discriminative.

Spec: REQ-LEARN-1212, SCENARIO-LEARN-1212
"""

from __future__ import annotations

from dataclasses import dataclass, field

from carnot.pipeline.extract import ConstraintResult


# ---------------------------------------------------------------------------
# ConstraintFiringStats — per-type firing rates
# ---------------------------------------------------------------------------


@dataclass
class ConstraintFiringStats:
    """Firing statistics for one constraint type over an observation window.

    **Detailed explanation for engineers:**
        Each call to ConstraintAdditionAgent.observe() with a set of fired
        constraint types updates the per-type counters here.  At query time,
        wrong_rate and correct_rate measure how discriminative the type is.

        A type is "high signal" when it fires reliably on wrong responses
        (high wrong_rate) AND rarely on correct ones (low correct_rate).

    Attributes:
        constraint_type: The constraint type tag (e.g. "arithmetic").
        n_wrong_fired: Count of wrong responses where this type fired.
        n_correct_fired: Count of correct responses where this type fired.
        n_wrong_total: Total wrong responses observed so far.
        n_correct_total: Total correct responses observed so far.
    """

    constraint_type: str
    n_wrong_fired: int = 0
    n_correct_fired: int = 0
    n_wrong_total: int = 0
    n_correct_total: int = 0

    @property
    def wrong_rate(self) -> float:
        """Fraction of wrong responses where this type fired."""
        return self.n_wrong_fired / max(1, self.n_wrong_total)

    @property
    def correct_rate(self) -> float:
        """Fraction of correct responses where this type fired."""
        return self.n_correct_fired / max(1, self.n_correct_total)

    def is_high_signal(self, wrong_threshold: float = 0.6, correct_threshold: float = 0.2) -> bool:
        """Return True when this type reliably flags wrong responses but not correct ones."""
        return self.wrong_rate > wrong_threshold and self.correct_rate < correct_threshold


# ---------------------------------------------------------------------------
# ConstraintAdditionAgent
# ---------------------------------------------------------------------------


class ConstraintAdditionAgent:
    """Online agent that detects high-signal constraint types and generates new constraints.

    **Detailed explanation for engineers:**
        Usage pattern:

            agent = ConstraintAdditionAgent()
            # Learning phase — iterate over labelled samples
            for text, label in learning_samples:
                fired_types = get_fired_types(text)   # e.g. {"arithmetic"}
                agent.observe(fired_types, is_correct=(label == "correct"))

            # Detect which types should be added
            high_signal = agent.detect_additions()   # e.g. ["arithmetic"]
            n_added = agent.n_constraints_added      # e.g. 1

            # Build specialised ConstraintResult objects for injection
            added_constraints = agent.build_addition_constraints(text)

        The ``build_addition_constraints`` method generates one ConstraintResult per
        detected high-signal type.  These results are injected into the pipeline's
        active constraint list, causing the pipeline to flag any violation of that
        type as a definitive signal.

    Args:
        wrong_threshold: Minimum firing rate on wrong responses to qualify as
            high signal.  Defaults to 0.6 (fires in >60% of wrong responses).
        correct_threshold: Maximum firing rate on correct responses to qualify
            as high signal.  Defaults to 0.2 (fires in <20% of correct responses).

    Spec: REQ-LEARN-1212
    """

    def __init__(self, wrong_threshold: float = 0.6, correct_threshold: float = 0.2) -> None:
        self._wrong_threshold = wrong_threshold
        self._correct_threshold = correct_threshold
        # Per-constraint-type firing stats, keyed by type name.
        self._stats: dict[str, ConstraintFiringStats] = {}
        self._n_wrong_total = 0
        self._n_correct_total = 0

    # -----------------------------------------------------------------------
    # Learning API
    # -----------------------------------------------------------------------

    def observe(self, fired_types: set[str], is_correct: bool) -> None:
        """Record which constraint types fired on one labelled response.

        **Detailed explanation for engineers:**
            Call this once per response in the learning corpus.  fired_types
            is the set of constraint type tags (e.g. {"arithmetic"}) that
            produced at least one violation on this response.

            The global wrong/correct totals are incremented on every call
            regardless of which types fired, so the denominators for
            wrong_rate / correct_rate are always the total number of samples
            seen in each class — not just the subset where the type fired.

        Args:
            fired_types: Set of constraint type strings that fired (had
                violations) on this response.
            is_correct: True if this response is ground-truth correct.
        """
        if is_correct:
            self._n_correct_total += 1
        else:
            self._n_wrong_total += 1

        # Update totals for ALL known types (so denominators stay current).
        for stats in self._stats.values():
            if is_correct:
                stats.n_correct_total += 1
            else:
                stats.n_wrong_total += 1

        # Record firings for newly-seen and previously-seen types.
        for ctype in fired_types:
            if ctype not in self._stats:
                self._stats[ctype] = ConstraintFiringStats(
                    constraint_type=ctype,
                    n_wrong_total=self._n_wrong_total,
                    n_correct_total=self._n_correct_total,
                )
            s = self._stats[ctype]
            if is_correct:
                s.n_correct_fired += 1
            else:
                s.n_wrong_fired += 1

    # -----------------------------------------------------------------------
    # Detection API
    # -----------------------------------------------------------------------

    def detect_additions(self) -> list[str]:
        """Return constraint type names that exceed the high-signal thresholds.

        **Detailed explanation for engineers:**
            Only types where wrong_rate > wrong_threshold AND
            correct_rate < correct_threshold are returned.  These are the types
            whose firing is a reliable indicator of a wrong response — adding
            them as active constraints will improve pipeline precision.

        Returns:
            List of constraint type strings to add to the active pipeline.

        Spec: REQ-LEARN-1212
        """
        return [
            ctype
            for ctype, stats in self._stats.items()
            if stats.is_high_signal(self._wrong_threshold, self._correct_threshold)
        ]

    @property
    def n_constraints_added(self) -> int:
        """Number of high-signal constraint types detected (to be added)."""
        return len(self.detect_additions())

    def firing_stats(self) -> dict[str, ConstraintFiringStats]:
        """Return a copy of the per-type firing statistics for inspection."""
        return dict(self._stats)

    # -----------------------------------------------------------------------
    # Constraint generation
    # -----------------------------------------------------------------------

    def build_addition_constraints(
        self, text: str, added_types: list[str] | None = None
    ) -> list[ConstraintResult]:
        """Build ConstraintResult objects for each high-signal type that fires.

        **Detailed explanation for engineers:**
            For each high-signal constraint type, we re-run the appropriate
            extractor on the given text and return any violations.  These
            ConstraintResult objects are ready to inject into the pipeline's
            active constraint list.

            The ``added_types`` argument lets the caller override which types
            are checked (defaults to whatever detect_additions() returns).

        Args:
            text: The response text to check.
            added_types: Override list of constraint types.  If None, uses
                the types returned by detect_additions().

        Returns:
            List of ConstraintResult objects representing violations found
            under the high-signal constraint types.

        Spec: REQ-LEARN-1212
        """
        from carnot.pipeline.extract import (  # noqa: PLC0415
            ArithmeticExtractor,
            CodeExtractor,
            LogicExtractor,
            NLExtractor,
        )

        extractor_map = {
            "arithmetic": ArithmeticExtractor(),
            "code": CodeExtractor(),
            "logic": LogicExtractor(),
            "nl": NLExtractor(),
            # Subtypes all delegate to arithmetic
            "type_check": CodeExtractor(),
            "bound": CodeExtractor(),
        }

        types_to_check = added_types if added_types is not None else self.detect_additions()
        results: list[ConstraintResult] = []

        for ctype in types_to_check:
            # Map subtype strings to their parent extractor
            root = ctype.split("_")[0]
            ext = extractor_map.get(ctype) or extractor_map.get(root)
            if ext is None:
                continue
            for constraint in ext.extract(text):
                if constraint.constraint_type == ctype or constraint.constraint_type.startswith(
                    root
                ):
                    # Only include violations (unsatisfied constraints)
                    if not constraint.metadata.get("satisfied", True):
                        results.append(constraint)

        return results
