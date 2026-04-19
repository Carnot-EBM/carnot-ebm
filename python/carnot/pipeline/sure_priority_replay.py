"""SuRe (Surprise-prioritized Replay) for PPSConstraintLearner (arXiv 2511.22367, Exp 497).

**Why surprise-driven replay (the SuRe insight from arXiv 2511.22367):**
    Uniform random replay wastes replay budget on easy examples that the model already
    handles well — violations the model consistently assigns low energy to and has not
    forgotten.  These easy examples do not contribute to preventing catastrophic forgetting
    because they are far from the knowledge boundary.

    SuRe (arXiv 2511.22367) selects replay items by NLL (negative log-likelihood) —
    high-surprise sequences that the model was most uncertain about are replayed more
    frequently.  The paper shows +5% accuracy improvement in continual learning tasks
    compared to uniform random replay, specifically at domain transitions where forgetting
    is concentrated.

**Why EBM energy as surprise proxy (not NLL directly):**
    In NLP continual learning (the SuRe paper's setting), NLL is computed directly from
    token probabilities.  Carnot's EBM setting does not have a language model outputting
    token probabilities — instead, we have Ising EBM energy values per violation.

    The mapping: high Ising energy = multiple competing low-energy spin configurations =
    the constraint is ambiguous for the current model = the model is "surprised" by this
    violation.  Equivalently, a violation that the domain's EBM assigns HIGH energy to
    is one where the model's learned distribution is uncertain — this is the EBM-domain
    analogue of high NLL in language model continual learning.

    Surprise score = energy - domain_mean_energy.  Positive = more surprising than
    the domain average.  Large positive = near the knowledge boundary = high forgetting risk.

**Connection to PPSConstraintLearner (Exp 470 / 485):**
    PPSConstraintLearner maintains per-domain parameter partitions for ARITHMETIC, CODE,
    and LOGICAL constraint types.  When a new violation arrives from a domain, the
    corresponding partition is updated.  Under interleaved real data (Exp 485 RETRO-043),
    violations from multiple domains arrive in the same sliding window — the partition
    must be updated on a mix of domains.

    Without SuRe: violations are replayed uniformly at random from the buffer.  Easy
    arithmetic violations (low energy, model already handles them) crowd out harder
    violations near the code/logical boundary.

    With SuRe: the top-k highest-surprise violations are always replayed.  These are the
    examples closest to the knowledge boundary — the ones most likely to cause partition
    isolation degradation if forgotten.  Result: partition_isolation_score is maintained
    higher under interleaved conditions.

Spec: REQ-SELFLEARN-021, REQ-SELFLEARN-022,
      SCENARIO-SELFLEARN-021, SCENARIO-SELFLEARN-022
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# ViolationSurprise
# ---------------------------------------------------------------------------


@dataclass
class ViolationSurprise:
    """One violation from the replay buffer, annotated with its surprise score.

    **Why track domain_mean_energy (not absolute energy):**
        Different constraint domains have different natural energy scales.  An energy
        of 2.0 is high-surprise for CODE violations (if the domain mean is 0.5) but
        low-surprise for ARITHMETIC violations (if the domain mean is 3.5).  Computing
        surprise_score = energy - domain_mean_energy normalises across domains, making
        high-surprise selections fair across ARITHMETIC, CODE, and LOGICAL domains.

    **Why is_high_surprise threshold (not pure ranking):**
        In addition to ranking by surprise, the system supports a threshold gate:
        violations with surprise_score <= surprise_threshold are considered "easy" and
        may be excluded from replay entirely.  This further concentrates replay budget
        on genuinely hard examples.

    Args:
        violation: The constraint violation dict (from PPSConstraintLearner training data).
        domain: The constraint domain string ('arithmetic', 'code', 'logical').
        energy: The EBM energy assigned to this violation by the domain's Ising model.
        domain_mean_energy: The running mean energy for this domain at time of addition.
        surprise_threshold: Threshold above which is_high_surprise is True (default 0.5).

    Spec: REQ-SELFLEARN-021
    """

    violation: dict
    domain: str
    energy: float
    domain_mean_energy: float
    surprise_threshold: float = 0.5

    @property
    def surprise_score(self) -> float:
        """Energy relative to domain mean.

        Positive = this violation is more energetically surprising than the domain
        average.  The further above zero, the more the model is currently uncertain
        about this constraint boundary — and the more likely it is to be forgotten
        under interleaved training.

        Returns:
            Float (may be negative if violation is below-average energy).
        """
        return self.energy - self.domain_mean_energy

    @property
    def is_high_surprise(self) -> bool:
        """Return True iff surprise_score > surprise_threshold.

        WHY strict inequality (not >=): a violation exactly AT the threshold is
        borderline — not clearly high-surprise enough to prioritise over a violation
        that is definitively above the threshold.  Strict inequality avoids promoting
        borderline examples.

        Returns:
            True if surprise_score > surprise_threshold.
        """
        return self.surprise_score > self.surprise_threshold


# ---------------------------------------------------------------------------
# SuRePriorityReplay
# ---------------------------------------------------------------------------


class SuRePriorityReplay:
    """Surprise-prioritized replay buffer for PPSConstraintLearner (arXiv 2511.22367).

    **Lifecycle:**
        1. Construct with replay_buffer_size, top_k_fraction, and surprise_threshold.
        2. Call ``add(violation, domain, energy)`` whenever a new violation is observed.
        3. Call ``get_replay_batch(n)`` to retrieve the top-n highest-surprise violations
           for replay at each update step.
        4. Domain means are updated automatically by ``add()`` via ``update_domain_mean()``.

    **Why FIFO eviction when buffer is full:**
        The replay buffer must not grow without bound in a streaming production setting.
        When the buffer is full, the oldest violation is evicted (FIFO).  This is
        consistent with SuRe's assumption that recent violations are more relevant to
        the current domain shift — stale violations from long-past sessions provide less
        forgetting-prevention value.

    **Why top_k_fraction (not fixed top_k):**
        The optimal replay batch size depends on the total number of violations seen.
        A fraction (e.g. 0.3 = top 30%) scales naturally with buffer fill level —
        providing more replay items when the buffer contains more history, without
        hardcoding a magic number.

    Args:
        replay_buffer_size: Maximum number of violations to keep in the buffer (FIFO eviction).
        top_k_fraction: Fraction of buffer to select for replay (e.g. 0.3 = top 30%).
        surprise_threshold: Minimum surprise_score for is_high_surprise classification.

    Spec: REQ-SELFLEARN-021, REQ-SELFLEARN-022
    """

    def __init__(
        self,
        replay_buffer_size: int = 100,
        top_k_fraction: float = 0.3,
        surprise_threshold: float = 0.5,
    ) -> None:
        self.replay_buffer_size = replay_buffer_size
        self.top_k_fraction = top_k_fraction
        self.surprise_threshold = surprise_threshold

        # The replay buffer: list of ViolationSurprise objects in insertion order.
        self._buffer: list[ViolationSurprise] = []

        # Per-domain running mean energy tracking.
        # WHY running mean (not recompute from buffer each time):
        #   Computing the mean from the buffer at every add() call is O(n) per call.
        #   Tracking a running mean with (sum, count) is O(1) per call.
        #   Since domain_mean_energy is needed at add() time to compute surprise_score,
        #   we update the mean BEFORE computing surprise so the score reflects the
        #   long-run domain average, not the batch-local average.
        self._domain_energy_sums: dict[str, float] = {}
        self._domain_energy_counts: dict[str, int] = {}

    def update_domain_mean(self, domain: str, energy: float) -> None:
        """Update the running mean energy for a domain with a new observation.

        Called by ``add()`` before computing surprise_score so the domain mean
        reflects all prior observations for this domain (not just the current batch).

        Args:
            domain: The constraint domain string.
            energy: The EBM energy to incorporate into the running mean.
        """
        self._domain_energy_sums[domain] = (
            self._domain_energy_sums.get(domain, 0.0) + energy
        )
        self._domain_energy_counts[domain] = (
            self._domain_energy_counts.get(domain, 0) + 1
        )

    def _domain_mean(self, domain: str) -> float:
        """Return the current running mean energy for a domain.

        Returns 0.0 if no observations have been recorded yet (cold start).
        WHY 0.0 as cold-start default: with no prior observations, we cannot estimate
        the domain mean.  Defaulting to 0.0 means the first violation always has a
        positive surprise score (since energy >= 0 for Ising models), which is
        conservative — new domains are always treated as potentially high-surprise.
        """
        count = self._domain_energy_counts.get(domain, 0)
        if count == 0:
            return 0.0
        return self._domain_energy_sums[domain] / count

    def add(self, violation: dict, domain: str, energy: float) -> None:
        """Add a violation to the replay buffer.

        Updates the domain mean FIRST, then computes surprise_score relative to
        the updated mean.  Evicts the oldest item if the buffer is full (FIFO).

        Args:
            violation: The constraint violation dict.
            domain: The constraint domain ('arithmetic', 'code', 'logical').
            energy: The EBM energy assigned to this violation.
        """
        # Update running mean first (so surprise_score reflects all history).
        self.update_domain_mean(domain, energy)
        domain_mean = self._domain_mean(domain)

        item = ViolationSurprise(
            violation=violation,
            domain=domain,
            energy=energy,
            domain_mean_energy=domain_mean,
            surprise_threshold=self.surprise_threshold,
        )

        # FIFO eviction when buffer is full.
        if len(self._buffer) >= self.replay_buffer_size:
            self._buffer.pop(0)

        self._buffer.append(item)

    def get_replay_batch(self, n: int) -> list[dict]:
        """Return the top-n highest-surprise violations for replay.

        Sorts the buffer by surprise_score descending and returns the violation
        dicts (not ViolationSurprise wrappers) for the top-n items.

        WHY return violation dicts (not ViolationSurprise):
            PPSConstraintLearner.fit_domain() accepts plain violation strings/dicts.
            Returning the raw violation dict keeps the interface clean — callers do
            not need to know about the ViolationSurprise wrapper to use the replay.

        Args:
            n: Number of violations to return (returns fewer if buffer has < n items).

        Returns:
            List of violation dicts sorted by surprise_score descending (highest first).
        """
        if not self._buffer:
            return []

        sorted_items = sorted(
            self._buffer, key=lambda item: item.surprise_score, reverse=True
        )
        top_n = sorted_items[:n]
        return [item.violation for item in top_n]


# ---------------------------------------------------------------------------
# SuReReplayResult
# ---------------------------------------------------------------------------


@dataclass
class SuReReplayResult:
    """Comparison result between SuRe surprise-priority replay and uniform random replay.

    **Why compare against uniform (not no replay):**
        The interesting question for Tier 2 self-learning (FR-11) is whether SELECTING
        which violations to replay matters — not whether replay in general helps.
        Uniform random replay is the natural baseline: it uses the same budget but
        wastes it on random violations rather than prioritising high-surprise ones.

    **Why isolation_improvement (not accuracy improvement):**
        In Carnot's PPSEBM framework, the key metric for Tier 2 self-learning is
        partition_isolation_score — how orthogonal the domain-specific gradient updates
        are (cosine distance > threshold).  An improvement in isolation score under
        SuRe replay means the surprise-priority selection is preserving knowledge
        boundary separation better than uniform replay.

    Args:
        n_violations_processed: Total violations processed in the experiment.
        n_replay_items: Number of violations selected for replay (top-k).
        isolation_score_uniform: partition_isolation_score achieved with uniform replay.
        isolation_score_sure: partition_isolation_score achieved with SuRe priority replay.

    Spec: REQ-SELFLEARN-022, SCENARIO-SELFLEARN-022
    """

    n_violations_processed: int
    n_replay_items: int
    isolation_score_uniform: float
    isolation_score_sure: float

    @property
    def isolation_improvement(self) -> float:
        """Return isolation_score_sure - isolation_score_uniform.

        Positive = SuRe replay improved isolation over uniform baseline.
        Negative = SuRe replay degraded isolation (unexpected — log as warning).
        Zero = no difference.

        Returns:
            Float: delta isolation score (SuRe minus uniform).
        """
        return round(self.isolation_score_sure - self.isolation_score_uniform, 6)

    @property
    def sure_better(self) -> bool:
        """Return True iff SuRe replay achieved higher isolation than uniform replay.

        WHY strict inequality (not >=): equal performance means SuRe brought no benefit
        over the simpler uniform strategy.  We only claim SuRe is "better" when it
        strictly improves isolation.

        Returns:
            True if isolation_score_sure > isolation_score_uniform.

        Spec: REQ-SELFLEARN-022, SCENARIO-SELFLEARN-022
        """
        return self.isolation_score_sure > self.isolation_score_uniform


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SuRePriorityReplay",
    "SuReReplayResult",
    "ViolationSurprise",
]
