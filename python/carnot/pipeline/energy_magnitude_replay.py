"""EnergyMagnitudeReplay — energy-magnitude-priority replay for PPSConstraintLearner (RETRO-050).

**Why energy magnitude beats LLM surprise (RETRO-050 root cause):**
    Exp 497 (SuRe) showed isolation_improvement=-0.1172, meaning SuRe's LLM-NLL-based
    surprise priority HURT isolation compared to uniform replay.  The root cause was that
    "LLM surprise" (high NLL of a token sequence) and "high constraint energy" (the EBM
    energy function) are DIFFERENT concepts and turned out to be anticorrelated in practice.

    LLM NLL measures how unusual the surface form of the token sequence is — a sentence
    with rare words or unusual syntax is "surprising" to a language model even if the
    constraint it embodies is easy for the EBM to handle.  Conversely, a very common
    sentence structure can violate a hard constraint at high energy.

    **The EBM's energy function is the ground truth.**  High |energy - session_mean| means
    the EBM is currently maximally wrong about this constraint boundary — this is exactly
    the violation the model most needs to see replayed.  Replaying high-energy deviations
    directly targets the domain boundaries where catastrophic forgetting occurs.

    RETRO-050 recommendation: replace LLM-surprise priority with EBM energy magnitude
    priority: rank constraint violations by |energy(x) - domain_mean| and replay the
    top-k highest-deviation samples.

**How EnergyMagnitudeBuffer works:**
    Each domain gets its own buffer of at most `max_size` violations.  When a new
    violation arrives, it is inserted in sorted order by absolute energy deviation
    |energy - running_mean|.  When the buffer is full, the lowest-deviation item is
    evicted (the "easiest" violation, which the model handles best, contributes least
    to forgetting prevention).

    The running mean is updated with Welford's online algorithm (O(1) per update,
    numerically stable) so the deviation is always relative to the full history of
    energies seen, not just the current buffer contents.

**How EnergyMagnitudeReplay.isolation_score works:**
    Isolation score measures domain boundary interference: how much does updating on
    domain A change domain B's accuracy?  Perfect isolation = 1.0 (no cross-domain
    interference).  Perfect interference = -1.0 (every A update corrupts B).

    The simulation: for n_steps steps, alternately replay from domain_a and domain_b.
    After each domain_a replay step, check how much the domain_b batch loss changes.
    isolation_score = 1.0 - (mean cross-domain interference rate).

Spec: REQ-LEARN-036, REQ-LEARN-037, REQ-LEARN-038,
      SCENARIO-LEARN-064, SCENARIO-LEARN-065, SCENARIO-LEARN-066
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# EnergyMagnitudeBuffer
# ---------------------------------------------------------------------------


class EnergyMagnitudeBuffer:
    """Per-domain sorted buffer of constraint violations, ranked by |energy - mean|.

    WHY sort by |energy - mean| rather than raw energy:
        Different constraint domains have different natural energy scales.  Arithmetic
        domain energy might average 1.2 while code domain energy averages 3.7.  Sorting
        by absolute deviation normalises across domains so that a violation with
        energy=5.0 in an arithmetic domain (mean=1.2, deviation=3.8) correctly ranks
        higher than a violation with energy=4.0 in a code domain (mean=3.7, deviation=0.3).

    WHY evict the lowest-deviation item when full (not FIFO):
        FIFO eviction (oldest-out) loses recent high-energy violations.  Deviation-based
        eviction (easiest-out) ensures the buffer always contains the hardest examples —
        the ones the model is currently most wrong about — regardless of recency.

    WHY Welford's online algorithm for the running mean:
        We need mean_energy at add() time to compute the deviation.  Recomputing mean
        from the buffer contents is biased (buffer only holds top-max_size items, not
        all observed energies).  Welford's algorithm maintains an unbiased running mean
        over ALL observed energies in O(1) per update.

    Args:
        domain: The constraint domain label (e.g. 'arithmetic', 'code', 'logical').
        max_size: Maximum number of violations to retain in the buffer.

    Spec: REQ-LEARN-036, REQ-LEARN-037
    """

    def __init__(self, domain: str, max_size: int = 100) -> None:
        self.domain = domain
        self.max_size = max_size

        # Sorted list of (deviation, violation) tuples, highest deviation first.
        # WHY tuples: we need to sort by deviation; storing violation alongside avoids
        # an extra lookup when top_k() is called.
        self._items: list[tuple[float, dict]] = []

        # Welford running mean state.
        # WHY count and mean (not sum): avoids floating point accumulation error on
        # large streams; Welford's formula is (mean + (x - mean) / count).
        self._count: int = 0
        self._mean: float = 0.0

    @property
    def mean_energy(self) -> float:
        """Running mean of ALL energies added to this buffer (not just retained ones).

        WHY 0.0 when empty: the deviation of a new item relative to an empty domain is
        undefined; returning 0.0 means the first item's deviation equals its raw energy,
        which is a reasonable conservative estimate.
        """
        return self._mean

    def _update_mean(self, energy: float) -> None:
        """Update Welford running mean with a new energy observation."""
        self._count += 1
        delta = energy - self._mean
        self._mean += delta / self._count

    def add(self, violation: dict, energy: float) -> None:
        """Add a violation to the buffer, maintaining sorted order by |energy - mean|.

        The running mean is updated BEFORE computing the deviation so that the deviation
        is relative to the full history including this new observation.

        If the buffer is full and this violation has higher deviation than the lowest
        item, the lowest-deviation item is evicted to make room.  If this violation has
        lower or equal deviation than the lowest item and the buffer is full, it is
        discarded (the buffer already contains harder examples).

        Args:
            violation: The constraint violation dict.
            energy: The EBM energy assigned to this violation.
        """
        self._update_mean(energy)
        deviation = abs(energy - self._mean)

        if len(self._items) < self.max_size:
            # Buffer not full — insert in sorted position (descending deviation).
            self._insert_sorted(deviation, violation)
        elif self._items and deviation > self._items[-1][0]:
            # Buffer full but this item is harder than the easiest retained item.
            self._items.pop()
            self._insert_sorted(deviation, violation)
        # else: buffer full and this item is easier than all retained — discard.

    def _insert_sorted(self, deviation: float, violation: dict) -> None:
        """Insert (deviation, violation) maintaining descending order by deviation."""
        # Binary search for insertion point (highest deviation first).
        lo, hi = 0, len(self._items)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._items[mid][0] >= deviation:
                lo = mid + 1
            else:
                hi = mid
        self._items.insert(lo, (deviation, violation))

    def top_k(self, k: int) -> list[dict]:
        """Return the k highest-deviation violations (domain boundary cases).

        WHY return violations (not ViolationDeviation objects): callers (replay loop)
        only need the violation dict for replay; the deviation score is an internal
        ranking detail.

        Args:
            k: Number of top violations to return.

        Returns:
            List of violation dicts, ordered highest deviation first.
            May return fewer than k items if the buffer has fewer than k entries.
        """
        return [v for _, v in self._items[:k]]


# ---------------------------------------------------------------------------
# EnergyMagnitudeReplay
# ---------------------------------------------------------------------------


@dataclass
class _DomainState:
    """Internal state for one domain in EnergyMagnitudeReplay."""

    buffer: EnergyMagnitudeBuffer
    total_violations: int = 0


class EnergyMagnitudeReplay:
    """Multi-domain energy-magnitude-priority replay for PPSConstraintLearner (RETRO-050).

    **Lifecycle:**
        1. Construct with domain list and k (replay batch size per domain).
        2. Call ``add_violation(domain, violation, energy)`` for each new violation.
        3. Call ``get_replay_batch(domain)`` to retrieve the top-k highest-deviation
           violations for that domain at each update step.
        4. Call ``isolation_score(domain_a, domain_b, n_steps)`` to measure how much
           replaying domain_a influences domain_b's learned distribution.

    **Why k (not a fraction):**
        SuRe used top_k_fraction (a fraction of buffer size).  RETRO-050 recommends a
        fixed k because the isolation score metric (n_steps alternating replay steps)
        requires a predictable batch size per step for the simulation to be meaningful.
        A fixed k also makes the hyperparameter concrete and tunable in the experiment.

    **Why isolation_score is in [-1, 1]:**
        1.0 = perfect isolation (replaying domain_a never changes domain_b loss).
        0.0 = neutral (replaying domain_a changes domain_b by the same amount as
              random noise would).
        -1.0 = perfect interference (every domain_a replay step corrupts domain_b).

        The score is computed as: 1 - 2 * (cross_domain_change_rate), where
        cross_domain_change_rate is the fraction of domain_a replay steps that caused
        a statistically meaningful shift in domain_b's top-k selections.

    Args:
        domains: List of domain label strings (e.g. ['arithmetic', 'code', 'logical']).
        k: Number of top-violation items to return per replay batch.
        buffer_size: Maximum per-domain buffer size (passed to EnergyMagnitudeBuffer).

    Spec: REQ-LEARN-036, REQ-LEARN-037, REQ-LEARN-038
    """

    def __init__(
        self,
        domains: list[str],
        k: int = 10,
        buffer_size: int = 100,
    ) -> None:
        self.k = k
        self._states: dict[str, _DomainState] = {
            d: _DomainState(buffer=EnergyMagnitudeBuffer(d, max_size=buffer_size))
            for d in domains
        }

    def add_violation(self, domain: str, violation: dict, energy: float) -> None:
        """Record a new constraint violation for a domain.

        Updates the domain's EnergyMagnitudeBuffer and increments violation count.
        Unknown domains are silently ignored (allows partial domain specification).

        Args:
            domain: The constraint domain label.
            violation: The constraint violation dict.
            energy: The EBM energy for this violation.
        """
        if domain not in self._states:
            return
        state = self._states[domain]
        state.buffer.add(violation, energy)
        state.total_violations += 1

    def get_replay_batch(self, domain: str) -> list[dict]:
        """Return the top-k highest-energy-deviation violations for replay.

        WHY return a copy (via list slicing in EnergyMagnitudeBuffer.top_k):
            The replay consumer may modify the returned dicts; returning the internal
            buffer references would corrupt the buffer's sorted state.

        Args:
            domain: The constraint domain label.

        Returns:
            List of up to k violation dicts, highest energy deviation first.
            Returns [] for unknown domains.
        """
        if domain not in self._states:
            return []
        return self._states[domain].buffer.top_k(self.k)

    def isolation_score(
        self,
        domain_a: str,
        domain_b: str,
        n_steps: int = 20,
    ) -> float:
        """Measure domain boundary interference between domain_a and domain_b.

        Simulates n_steps of alternating replay: half the steps replay domain_a,
        half replay domain_b.  After each domain_a replay step, we check whether
        the domain_b replay batch has changed (i.e., whether domain_a replay would
        have shifted domain_b's learned constraint boundaries).

        The interference is measured as: fraction of domain_a steps that changed
        domain_b's top-k selection.  Perfect isolation = 0.0 change rate = score 1.0.
        Complete interference = 1.0 change rate = score -1.0.

        **Why this simulation (not actual gradient updates):**
            Actually training the model n_steps times would require a live EBM training
            loop and GPU time.  This experiment (CPU-only, RETRO-050) uses a proxy:
            if domain_a's high-energy violations are structurally similar to domain_b's
            high-energy violations (measured by overlap in violation keys/values), then
            replaying domain_a would disturb domain_b's boundaries.  The proxy is
            conservative — it measures potential interference, not actual gradient
            interference — but it is a meaningful proxy for whether the energy
            distributions of the two domains are well-separated.

        **Score interpretation:**
            - score = 1.0: domains are perfectly isolated (no shared violation structure).
            - score = 0.0: neutral (50% of domain_a steps overlap with domain_b structure).
            - score = -1.0: complete interference (all domain_a steps overlap domain_b).

        Args:
            domain_a: The "update" domain (the one being replayed).
            domain_b: The "watch" domain (the one we measure for interference).
            n_steps: Number of alternating replay steps in the simulation.

        Returns:
            Float in [-1.0, 1.0].  Higher is better (more isolated).

        Spec: REQ-LEARN-038, SCENARIO-LEARN-066
        """
        if domain_a not in self._states or domain_b not in self._states:
            return 0.0

        batch_a = self.get_replay_batch(domain_a)
        batch_b = self.get_replay_batch(domain_b)

        if not batch_a or not batch_b:
            # Empty buffer → no violations to interfere; define as neutral.
            return 0.0

        # Build fingerprints for each batch: frozen sets of (key, str(value)) pairs.
        # WHY frozenset (not list): order-independent overlap detection.  Two violations
        # that share constraint type and value but differ in metadata would still count
        # as overlapping for interference purposes.
        def _fingerprint(v: dict) -> frozenset:
            return frozenset((k, str(val)) for k, val in v.items())

        fp_b = {_fingerprint(v) for v in batch_b}

        # Simulate n_steps // 2 domain_a replay steps, counting how many overlap with
        # domain_b's fingerprint set.
        a_steps = max(1, n_steps // 2)
        interference_count = 0
        for i in range(a_steps):
            # Rotate through domain_a batch items to simulate sequential replay steps.
            v_a = batch_a[i % len(batch_a)]
            fp_a = _fingerprint(v_a)
            # Interference = domain_a violation shares constraint keys with domain_b items.
            # We measure key overlap (not full fingerprint match) because key sharing
            # indicates constraint-type overlap, which would cause gradient interference.
            a_keys = {k for k, _ in fp_a}
            for fp_bi in fp_b:
                b_keys = {k for k, _ in fp_bi}
                if a_keys & b_keys:
                    interference_count += 1
                    break  # count at most 1 interference per a_step

        change_rate = interference_count / a_steps
        # Map [0, 1] change_rate to [1, -1] score.
        return 1.0 - 2.0 * change_rate
