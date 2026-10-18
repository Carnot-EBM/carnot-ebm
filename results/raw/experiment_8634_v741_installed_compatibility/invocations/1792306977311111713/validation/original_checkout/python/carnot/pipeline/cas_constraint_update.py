"""CAS (Compress-Add-Smooth) updates for Carnot constraint template observation banks.

**Researcher summary:**
    arXiv:2604.00067 proposes the CAS recursion for bounded memory updates.
    Applied to Carnot's ConstraintTemplateLibrary: instead of unbounded count
    accumulation, each update step Compresses existing evidence (decay),
    Adds new observations, and Smooths the result (blend + cap).  This keeps
    the template observation bank bounded while incorporating new constraint
    signal from each verification session.

**Detailed explanation for engineers:**
    The CAS update solves a practical problem with ConstraintTemplateLibrary:
    without a forgetting mechanism, observation counts grow without bound across
    sessions.  Templates activated many milestones ago dominate over more recent
    signal.  The three-step CAS recursion fixes this:

    1. Compress — multiply all existing counts by compress_factor ∈ (0, 1).
       Old evidence decays; after k CAS steps it contributes only
       compress_factor^k of its original weight.  A factor of 0.9 over 10
       steps reduces a count-100 entry to ≈ 34.9.

    2. Add — incorporate new (pattern_key, model_id) → count observations.
       These arrive at full weight before the Smooth step blends them.

    3. Smooth — blend the post-add counts toward smooth_target (default 0.0)
       using blend weight smooth_alpha, then cap at max_count:
           smoothed = (1 - alpha) * raw + alpha * target
           clipped  = min(smoothed, max_count)
       This prevents any single pattern from permanently dominating the
       activation gate and guarantees a finite bound on all counts.

    Activation semantics are preserved: a pattern whose count stays above
    its template's min_frequency after the smooth step remains active.  A
    pattern whose decayed count drops below min_frequency becomes inactive,
    clearing stale activations automatically.

    Memory safety: regardless of how many CAS steps are applied, no entry in
    _observations ever exceeds max_count.

Spec: REQ-CAS-001, REQ-CAS-001-1, REQ-CAS-001-2, REQ-CAS-001-3,
      REQ-CAS-001-4, SCENARIO-CAS-001, SCENARIO-CAS-002
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary


class CASConstraintUpdater:
    """Apply Compress-Add-Smooth updates to a ConstraintTemplateLibrary.

    **Researcher summary:**
        Wraps three targeted operations — compress, add, smooth — into a
        single cas_update() call that keeps the library's observation bank
        bounded and recency-weighted.

    **Detailed explanation for engineers:**
        Attributes control the three update phases:

        compress_factor : float in (0, 1)
            Multiplicative decay applied to every existing observation count
            in the Compress step.  Smaller values = faster forgetting.
            Typical: 0.9 (10% decay per step).

        smooth_alpha : float in [0, 1]
            Blend weight toward smooth_target in the Smooth step.  Larger
            values = stronger pull toward target (more aggressive smoothing).
            Typical: 0.05–0.2.

        smooth_target : float
            The target value that smooth_alpha blends toward.  Default 0.0
            means counts decay toward zero under repeated smooth steps.
            Set higher to maintain a non-zero floor for active templates.

        max_count : float
            Hard upper bound on any single observation count after smoothing.
            Guarantees REQ-CAS-001-3 (bounded memory).

    Spec: REQ-CAS-001
    """

    def __init__(
        self,
        compress_factor: float = 0.9,
        smooth_alpha: float = 0.1,
        smooth_target: float = 0.0,
        max_count: float = 100.0,
    ) -> None:
        """Initialise the CAS updater.

        Args:
            compress_factor: Multiplicative decay for existing counts.
                             Must be in (0, 1).
            smooth_alpha:    Blend weight toward smooth_target.
                             Must be in [0, 1].
            smooth_target:   Target value for the smooth blend.
            max_count:       Maximum allowed observation count after smoothing.

        Spec: REQ-CAS-001
        """
        if not (0.0 < compress_factor < 1.0):
            raise ValueError(f"compress_factor must be in (0, 1), got {compress_factor}")
        if not (0.0 <= smooth_alpha <= 1.0):
            raise ValueError(f"smooth_alpha must be in [0, 1], got {smooth_alpha}")
        if max_count <= 0:
            raise ValueError(f"max_count must be positive, got {max_count}")

        self.compress_factor = compress_factor
        self.smooth_alpha = smooth_alpha
        self.smooth_target = smooth_target
        self.max_count = max_count

    def compress(self, library: "ConstraintTemplateLibrary") -> None:
        """Decay existing observation counts by compress_factor.

        **Detailed explanation for engineers:**
            Multiplies every value in library._observations by compress_factor.
            This is the "forgetting" step: old evidence shrinks geometrically
            so that counts from the distant past no longer dominate activation.

            After this step, counts are floats even if they started as ints.
            This is intentional — the get_active_templates() comparison
            (count >= min_frequency) works correctly for float counts.

        Args:
            library: The ConstraintTemplateLibrary whose observation bank
                     should be decayed in-place.

        Spec: REQ-CAS-001-1
        """
        for key in list(library._observations.keys()):
            library._observations[key] = library._observations[key] * self.compress_factor

    def add(
        self,
        library: "ConstraintTemplateLibrary",
        new_observations: dict[tuple[str, str], float],
    ) -> None:
        """Incorporate new constraint observations into the library.

        **Detailed explanation for engineers:**
            Calls library.observe_pattern() for each (pattern_key, model_id)
            → count entry in new_observations.  This preserves the library's
            existing observation accumulation semantics (additive counts).

            New observations arrive at FULL weight — no decay is applied here.
            The Compress step before Add means the old evidence has already been
            weakened; the Add step injects fresh signal at its original strength.

        Args:
            library:          The ConstraintTemplateLibrary to update.
            new_observations: Mapping from (pattern_key, model_id) tuples to
                              observation counts to add.  Counts may be floats.

        Spec: REQ-CAS-001-2
        """
        for (pattern_key, model_id), count in new_observations.items():
            # observe_pattern only accepts int, so we use direct dict access
            # to support float counts from the Compress step accumulation.
            key = (pattern_key, model_id)
            current = library._observations.get(key, 0.0)
            library._observations[key] = current + count

    def smooth(self, library: "ConstraintTemplateLibrary") -> None:
        """Blend observation counts toward smooth_target and cap at max_count.

        **Detailed explanation for engineers:**
            For each observation count c:
                smoothed = (1 - alpha) * c + alpha * target
                clipped  = min(smoothed, max_count)

            The blend prevents any single outlier count from staying extremely
            high across many iterations.  The cap enforces the hard memory bound
            required by REQ-CAS-001-3.

            Counts that fall below 0.0 after smoothing are set to 0.0 — the
            observation bank never stores negative evidence.

        Args:
            library: The ConstraintTemplateLibrary to smooth in-place.

        Spec: REQ-CAS-001-3
        """
        alpha = self.smooth_alpha
        target = self.smooth_target
        for key in list(library._observations.keys()):
            raw = library._observations[key]
            blended = (1.0 - alpha) * raw + alpha * target
            capped = min(blended, self.max_count)
            library._observations[key] = max(capped, 0.0)

    def cas_update(
        self,
        library: "ConstraintTemplateLibrary",
        new_observations: dict[tuple[str, str], float],
    ) -> dict[tuple[str, str], float]:
        """Apply a full Compress-Add-Smooth update step.

        **Detailed explanation for engineers:**
            Runs compress → add → smooth in that order.  This ordering ensures:
            1. Stale evidence is weakened BEFORE new evidence is added (so the
               new signal is not immediately decayed).
            2. The smooth step normalises the combined result and enforces the
               max_count cap.

            Returns the updated observation mapping as a plain dict so that
            callers can inspect or serialize the post-update state without
            modifying the library again.

        Args:
            library:          The ConstraintTemplateLibrary to update in-place.
            new_observations: New (pattern_key, model_id) → count observations.

        Returns:
            Dict mapping (pattern_key, model_id) → float count after the update.

        Spec: REQ-CAS-001-4
        """
        self.compress(library)
        self.add(library, new_observations)
        self.smooth(library)
        return dict(library._observations)


__all__ = ["CASConstraintUpdater"]
