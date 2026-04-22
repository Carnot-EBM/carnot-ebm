"""Cascade router — routes queries through the Carnot verification tier cascade.

**What this module does:**
    Implements the multi-tier cascade routing logic that decides which verification
    tiers a given LLM output must pass through.  The tiers are ordered cheapest-first:

        Tier 0h (EORM)  →  Tier 1 (fast energy gate)  →  Tier 2 (JEPA ranking)
                        →  Tier 3 (Ising sampling)

    Running all tiers on every query is accurate but expensive.  The key insight from
    arXiv 2505.11730 ("Variable Granularity Search") is that cheap-tier confidence
    correlates strongly with final verdict — when the cheap tier is very confident,
    running expensive tiers rarely changes the outcome.

**EORM confidence gate (REQ-INFRA-046):**
    EORM (Tier 0h) outputs a scalar confidence in [0, 1].  Values near 1.0 mean the
    cheap verifier is highly confident the generation is correct.  When this confidence
    exceeds ``eorm_ising_skip_threshold`` (default 0.92), the router skips Tier 3 Ising
    entirely and marks the result "verified_fast".

    Why 0.92 as the default: the arXiv 2505.11730 paper found that at >0.92 EORM
    confidence, Ising sampling changes the verdict in <5% of cases.  This matches our
    REQ-INFRA-047 requirement that fn_delta < 0.05.  The threshold is configurable so
    that production operators can tune it based on observed fn_delta in their domain.

**Per-query logging:**
    Every call to ``route()`` returns a ``RouteResult`` that includes:
    - ``ising_skip``: bool — True iff Tier 3 Ising was bypassed by the EORM gate.
    - ``eorm_confidence``: float — the EORM confidence score that drove the decision.
    These fields are included in every result so that downstream tooling (Exp 727,
    ops dashboards) can compute skip rates and fn_delta without re-running inference.

Spec: REQ-INFRA-046, REQ-INFRA-047, SCENARIO-INFRA-055, SCENARIO-INFRA-056
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


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
        Caller-supplied or router-generated extra fields (e.g. latency_ms).
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
    """Multi-tier cascade router with EORM confidence gate for Tier 3 Ising skip.

    The router accepts two callable backends (``eorm_fn`` and ``ising_fn``) so
    that unit tests can inject lightweight stubs without loading real models.
    In production, these are wired to the EORMModel and IsingVerifier respectively.

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

    Spec: REQ-INFRA-046
    """

    def __init__(
        self,
        eorm_fn: Callable[[str], float],
        ising_fn: Callable[[str], bool],
        eorm_ising_skip_threshold: float = 0.92,
    ) -> None:
        self.eorm_fn = eorm_fn
        self.ising_fn = ising_fn
        self.eorm_ising_skip_threshold = eorm_ising_skip_threshold

    def route(self, query: str) -> RouteResult:
        """Route one query through the cascade and return a RouteResult.

        Routing logic:
        1. Call EORM to get a confidence score.
        2. If confidence > eorm_ising_skip_threshold: skip Ising → "verified_fast".
        3. Else: call Ising → "verified_full" or "rejected".

        Per-query logging fields ``ising_skip`` and ``eorm_confidence`` are always
        populated in the returned RouteResult so callers can aggregate statistics.

        Parameters
        ----------
        query : str
            The LLM-generated text to verify (question + response concatenation
            or just the response, depending on how eorm_fn is wired).

        Returns
        -------
        RouteResult
            The routing decision with all per-query log fields populated.

        Spec: REQ-INFRA-046, SCENARIO-INFRA-055, SCENARIO-INFRA-056
        """
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
            )

        # Below threshold — run Ising as normal.
        ising_result = bool(self.ising_fn(query))
        verdict = "verified_full" if ising_result else "rejected"
        return RouteResult(
            verified=ising_result,
            verdict=verdict,
            eorm_confidence=eorm_confidence,
            ising_skip=False,
            ising_result=ising_result,
        )
