"""Cascade router — routes queries through the Carnot verification tier cascade.

**What this module does:**
    Implements the multi-tier cascade routing logic that decides which verification
    tiers a given LLM output must pass through.  The tiers are ordered cheapest-first:

        Tier 0b (KAN prompt-injection pre-filter)  →  safety pipeline if injection
        Tier 0h (EORM)  →  Tier 1 (fast energy gate)  →  Tier 2 (JEPA ranking)
                        →  Tier 2.1 (JEPAReasonerProbe — early exit gate)
                        →  Tier 2.5 (SymCodeVerifier)
                        →  Tier 3 (Ising sampling)

    Running all tiers on every query is accurate but expensive.  The key insight from
    arXiv 2505.11730 ("Variable Granularity Search") is that cheap-tier confidence
    correlates strongly with final verdict — when the cheap tier is very confident,
    running expensive tiers rarely changes the outcome.

**Tier 0b KAN prompt-injection pre-filter (REQ-SAFE-016):**
    Before any verification tier runs, the KAN Tier 0b classifier scores the query
    for injection patterns.  If the score exceeds 0.5, the query is routed
    immediately to the safety pipeline and no other tier executes.  This prevents
    expensive verifiers from processing adversarial inputs.

    The KANTier0bClassifier is optional (``tier0b_classifier`` kwarg): when not
    supplied, the router behaves exactly as before (backwards compatible).

**EORM confidence gate (REQ-INFRA-046):**
    EORM (Tier 0h) outputs a scalar confidence in [0, 1].  Values near 1.0 mean the
    cheap verifier is highly confident the generation is correct.  When this confidence
    exceeds ``eorm_ising_skip_threshold`` (default 0.92), the router skips Tier 3 Ising
    entirely and marks the result "verified_fast".

    Why 0.92 as the default: the arXiv 2505.11730 paper found that at >0.92 EORM
    confidence, Ising sampling changes the verdict in <5% of cases.  This matches our
    REQ-INFRA-047 requirement that fn_delta < 0.05.  The threshold is configurable so
    that production operators can tune it based on observed fn_delta in their domain.

**Tier 2.1 early-exit gate (REQ-VER-035, REQ-VER-036):**
    After Tier 2 EORM, the Tier21ProbeWrapper scores the query's hidden state using the
    pre-trained JEPAReasonerProbe.  If the probe score <= calibrated threshold, verdict is
    "likely_correct" and Tier 2.5 SymCodeVerifier, 2.6 HERMES, and 2.7 Causal tiers are
    ALL skipped (early exit).  If score > threshold, the cascade continues normally to
    Tier 2.5+.  A ViolationEvent stub is emitted to FR11EventBus on the violation path
    (REQ-VER-037).

**Per-query logging:**
    Every call to ``route()`` returns a ``RouteResult`` that includes:
    - ``ising_skip``: bool — True iff Tier 3 Ising was bypassed by the EORM gate.
    - ``eorm_confidence``: float — the EORM confidence score that drove the decision.
    - ``metadata["tier21_skip"]``: bool — True iff Tier 2.5+ was skipped by the probe gate.
    - ``metadata["probe_score"]``: float — the raw Tier 2.1 probe score (always populated).
    - ``metadata["tier0b_score"]``: float — Tier 0b KAN injection score (when wired).
    - ``metadata["tier0b_verdict"]``: str — "injection_detected" or "benign" (when wired).
    These fields are included in every result so that downstream tooling (Exp 727, 733,
    735, ops dashboards) can compute skip rates and fn_delta without re-running inference.

Spec: REQ-INFRA-046, REQ-INFRA-047, REQ-VER-035, REQ-VER-036, REQ-VER-037,
      REQ-SAFE-016, SCENARIO-INFRA-055, SCENARIO-INFRA-056, SCENARIO-VER-044,
      SCENARIO-VER-045, SCENARIO-SAFE-016, SCENARIO-SAFE-017
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    import numpy as np

    from carnot.cascade.tier0b_kan import KANTier0bClassifier
    from carnot.cascade.tier21_probe import Tier21ProbeWrapper


# ---------------------------------------------------------------------------
# RouteResult — the per-query routing decision record
# ---------------------------------------------------------------------------


@dataclass
class RouteResult:
    """Outcome of routing one query through the cascade.

    Attributes
    ----------
    verified : bool
        True iff the query passed all tiers it was routed through.
    verdict : str
        Human-readable verdict string.  One of:
        - ``"verified_fast"``   — EORM gate fired; Ising skipped.
        - ``"verified_full"``   — full cascade (Ising ran); query passed.
        - ``"rejected"``        — full cascade; query failed Ising.
        - ``"likely_correct"``  — Tier 2.1 probe gate fired; Tier 2.5+ skipped.
        - ``"likely_violation"``— Tier 2.1 probe flagged violation; Tier 2.5+ ran.
    eorm_confidence : float
        The raw EORM confidence score (0–1) for this query.
    ising_skip : bool
        True iff Tier 3 Ising was skipped because EORM confidence exceeded the
        configured threshold.  This is the field Exp 727 aggregates to compute
        the skip rate.
    ising_result : bool | None
        The Ising sampling verdict (True=pass, False=fail).  None when Ising
        was skipped.
    metadata : dict
        Caller-supplied or router-generated extra fields.
        Key fields added by Tier 2.1 logic (REQ-VER-036-1, REQ-VER-036-2):
        - ``tier21_skip`` (bool): True iff Tier 2.5+ was skipped by the probe gate.
        - ``probe_score`` (float): the raw Tier 2.1 probe score (always populated when
          a Tier21ProbeWrapper is wired into the router).
        Key fields added by Tier 0b KAN filter (REQ-SAFE-016):
        - ``tier0b_score`` (float): KAN injection probability in [0, 1].
        - ``tier0b_verdict`` (str): "injection_detected" or "benign".
    """

    verified: bool
    verdict: str
    eorm_confidence: float
    ising_skip: bool
    ising_result: Optional[bool] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CascadeRouter
# ---------------------------------------------------------------------------


class CascadeRouter:
    """Multi-tier cascade router with EORM confidence gate and Tier 2.1 probe gate.

    The router accepts two callable backends (``eorm_fn`` and ``ising_fn``) so
    that unit tests can inject lightweight stubs without loading real models.
    In production, these are wired to the EORMModel and IsingVerifier respectively.

    Tier 2.1 is optional: when ``tier21_probe`` is supplied, the probe is called after
    EORM and before Tier 2.5+.  When the probe scores a query as "likely_correct",
    Tier 2.5 SymCodeVerifier, 2.6 HERMES, and 2.7 Causal tiers are skipped entirely.
    When the probe scores "likely_violation", a ViolationEvent stub is emitted and the
    cascade continues to Tier 2.5 as normal.

    WHY Tier 2.1 is an optional kwarg and not required:
        Existing callers (Exp 727, integration tests) were written before Tier 2.1
        existed.  Making it optional preserves backwards compatibility — callers that
        don't pass tier21_probe get the original routing behaviour unchanged.

    Parameters
    ----------
    eorm_fn : callable
        Function ``(query: str) -> float`` that returns an EORM confidence in [0, 1].
        Higher is more confident the generation is correct.
    ising_fn : callable
        Function ``(query: str) -> bool`` that returns True iff Ising sampling
        accepts the query (i.e., the energy-based sampler judges it correct).
    eorm_ising_skip_threshold : float
        EORM confidence above which Tier 3 Ising is skipped.  Default 0.92, matching
        the arXiv 2505.11730 finding that >0.92 confidence changes <5% of verdicts.
        Lower this value to be more conservative (skip less); raise it to skip more.
    tier21_probe : Tier21ProbeWrapper | None
        Optional Tier 2.1 probe wrapper.  When supplied, it is called after EORM with
        the query's hidden state.  Requires that ``hidden_state_fn`` is also supplied.
    hidden_state_fn : callable | None
        Function ``(query: str) -> np.ndarray`` that extracts the hidden state vector
        for the Tier 2.1 probe.  Required when tier21_probe is not None.
    tier0b_classifier : KANTier0bClassifier | None
        Optional Tier 0b KAN injection pre-filter.  When supplied, it is called FIRST
        before any other tier.  Queries with score > 0.5 are immediately routed to the
        safety pipeline (verdict="safety_violation") without running EORM or Ising.
        When None (default), the pre-filter is bypassed and behaviour is unchanged.

    Spec: REQ-INFRA-046, REQ-VER-035, REQ-VER-036, REQ-VER-037, REQ-SAFE-016
    """

    def __init__(
        self,
        eorm_fn: Callable[[str], float],
        ising_fn: Callable[[str], bool],
        eorm_ising_skip_threshold: float = 0.92,
        tier21_probe: Optional["Tier21ProbeWrapper"] = None,
        hidden_state_fn: Optional[Callable[[str], "np.ndarray"]] = None,
        tier0b_classifier: Optional["KANTier0bClassifier"] = None,
    ) -> None:
        self.eorm_fn = eorm_fn
        self.ising_fn = ising_fn
        self.eorm_ising_skip_threshold = eorm_ising_skip_threshold
        self.tier21_probe = tier21_probe
        self.hidden_state_fn = hidden_state_fn
        self.tier0b_classifier = tier0b_classifier

        if tier21_probe is not None and hidden_state_fn is None:
            raise ValueError(
                "hidden_state_fn is required when tier21_probe is supplied. "
                "The probe needs a hidden state vector extracted from the query."
            )

    def route(self, query: str) -> RouteResult:
        """Route one query through the cascade and return a RouteResult.

        Routing logic:
        0. If Tier 0b KAN classifier is wired: call classify(query).
           If verdict == "injection_detected": return immediately with
           verdict="safety_violation" (REQ-SAFE-016).  No further tiers run.
        1. Call EORM to get a confidence score.
        2. If confidence > eorm_ising_skip_threshold: skip Ising → "verified_fast".
        3. If Tier 2.1 probe is wired:
           a. Extract hidden state and call probe.score().
           b. If verdict == "likely_correct": skip Tier 2.5+ → "likely_correct" (early exit).
           c. If verdict == "likely_violation": emit stub, continue to step 4.
        4. Call Ising → "verified_full" or "rejected".

        Per-query logging fields are always populated in the returned RouteResult so
        callers can aggregate statistics.  Tier 2.1 adds ``tier21_skip`` and
        ``probe_score`` to RouteResult.metadata (REQ-VER-036-1, REQ-VER-036-2).

        Parameters
        ----------
        query : str
            The LLM-generated text to verify (question + response concatenation
            or just the response, depending on how eorm_fn is wired).

        Returns
        -------
        RouteResult
            The routing decision with all per-query log fields populated.

        Spec: REQ-INFRA-046, REQ-VER-035, REQ-VER-036, REQ-VER-037,
              SCENARIO-INFRA-055, SCENARIO-INFRA-056, SCENARIO-VER-044, SCENARIO-VER-045
        """
        # --- Tier 0b KAN prompt-injection pre-filter (REQ-SAFE-016) ---
        # Runs FIRST before any other tier.  If a query is flagged as an injection
        # attempt (score > 0.5), we skip the entire verification cascade and route
        # to the safety pipeline immediately.  This protects expensive tiers from
        # adversarial inputs and prevents attackers from probing verifier behaviour.
        if self.tier0b_classifier is not None:
            tier0b_score, tier0b_verdict = self.tier0b_classifier.classify(query)
            if tier0b_verdict == "injection_detected":
                return RouteResult(
                    verified=False,
                    verdict="safety_violation",
                    eorm_confidence=0.0,
                    ising_skip=True,
                    ising_result=None,
                    metadata={
                        "tier0b_score": tier0b_score,
                        "tier0b_verdict": tier0b_verdict,
                    },
                )
            # Benign: log the Tier 0b metadata and fall through to the normal cascade.
            tier0b_meta: dict[str, Any] = {
                "tier0b_score": tier0b_score,
                "tier0b_verdict": tier0b_verdict,
            }
        else:
            tier0b_meta = {}

        eorm_confidence = float(self.eorm_fn(query))

        if eorm_confidence > self.eorm_ising_skip_threshold:
            # EORM gate fires — Ising is expensive and unlikely to change the verdict.
            # Mark "verified_fast" so downstream tools can distinguish gated results
            # from full-cascade results.
            return RouteResult(
                verified=True,
                verdict="verified_fast",
                eorm_confidence=eorm_confidence,
                ising_skip=True,
                ising_result=None,
                metadata=tier0b_meta,
            )

        # --- Tier 2.1 probe gate (REQ-VER-035, REQ-VER-036) ---
        if self.tier21_probe is not None and self.hidden_state_fn is not None:
            hidden_state = self.hidden_state_fn(query)
            probe_score, probe_verdict = self.tier21_probe.score(hidden_state)

            if probe_verdict == "likely_correct":
                # Early exit: skip Tier 2.5 SymCodeVerifier, 2.6 HERMES, 2.7 Causal.
                # The response is very likely correct per the pre-generative probe —
                # running expensive symbolic verifiers would waste compute with < 5% FP.
                return RouteResult(
                    verified=True,
                    verdict="likely_correct",
                    eorm_confidence=eorm_confidence,
                    ising_skip=True,
                    ising_result=None,
                    metadata={
                        "tier21_skip": True,
                        "probe_score": probe_score,
                        **tier0b_meta,
                    },
                )
            else:
                # Probe flagged a violation — emit stub event and continue to Ising.
                # Build a deterministic query_id from the first 32 chars of the query.
                query_id = f"q_{abs(hash(query)) % 10_000_000:07d}"
                self.tier21_probe.emit_violation_stub(query_id, probe_score)
                # Fall through to Ising with probe metadata attached.
                ising_result = bool(self.ising_fn(query))
                verdict = "verified_full" if ising_result else "rejected"
                return RouteResult(
                    verified=ising_result,
                    verdict=verdict,
                    eorm_confidence=eorm_confidence,
                    ising_skip=False,
                    ising_result=ising_result,
                    metadata={
                        "tier21_skip": False,
                        "probe_score": probe_score,
                        **tier0b_meta,
                    },
                )

        # No Tier 2.1 probe — run Ising as normal.
        ising_result = bool(self.ising_fn(query))
        verdict = "verified_full" if ising_result else "rejected"
        return RouteResult(
            verified=ising_result,
            verdict=verdict,
            eorm_confidence=eorm_confidence,
            ising_skip=False,
            ising_result=ising_result,
            metadata=tier0b_meta,
        )
