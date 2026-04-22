"""Tier 2.1 JEPAReasonerProbe wrapper — pre-generative constraint gate.

WHY THIS MODULE EXISTS (REQ-VER-035):
    Tier 2 EORM scores responses after generation.  Tier 2.1 fires BEFORE Tier 2.5
    SymCodeVerifier and uses the hidden state of the LLM at the last input token to
    predict constraint violations in sub-1ms on CPU.  When the probe says "likely_correct",
    the cascade skips Tier 2.5, 2.6, and 2.7 entirely — saving expensive verifier calls
    on responses that are almost certainly fine.

THRESHOLD CALIBRATION (REQ-VER-035-1):
    The threshold is the 95th percentile of correct-step probe scores on the FoVer v2
    corpus (sourced from Exp 732 cross-validation).  At this percentile, < 5% of correct
    steps score ABOVE the threshold (false positive rate < 5%).  A step scoring ABOVE
    the threshold is routed as "likely_violation" — sending it through the expensive
    downstream tiers.  A step scoring AT OR BELOW the threshold exits early as
    "likely_correct", saving compute.

    Why 95th percentile for the correct-step distribution:
        The probe outputs P(violation): high scores = violation, low scores = correct.
        "likely_violation" fires when score > threshold.  For < 5% of correct steps
        to be mis-routed as violations, threshold must sit at the 95th percentile of
        correct-step scores — 95% of correct steps score at or below it (they exit
        early) and only 5% exceed it (false positives).  If threshold were at the 5th
        percentile, 95% of correct steps would be mis-routed as violations.

FR11EventBus STUB (REQ-VER-037):
    When a violation is detected, a ViolationEvent stub is emitted to the FR11EventBus.
    Until Exp 734 implements the real bus, the stub appends to an in-memory list passed
    at construction time, or calls a user-supplied callable.  Callers replace the stub
    by passing a real bus callable — the Tier21ProbeWrapper interface stays the same.

Spec: REQ-VER-035, REQ-VER-036, REQ-VER-037,
      SCENARIO-VER-044, SCENARIO-VER-045, SCENARIO-VER-046
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np

from carnot.samplers.jepa_reasoner_probe import JEPAReasonerProbe
from carnot.pipeline.fr11_event_bus import FR11EventBus, ViolationEvent


# ---------------------------------------------------------------------------
# ViolationEvent stub — minimal record emitted when probe flags a violation
# ---------------------------------------------------------------------------


class ViolationEventStub:
    """Minimal violation event record for the FR11EventBus stub.

    WHY a dataclass-like object instead of a plain dict:
        Exp 734 will replace this with a real ViolationEvent that adds fields like
        tier_source, error_code, and bus routing metadata.  Using a class now means
        callers can do isinstance checks and attribute access, not dict key lookups.
        The interface is stable even as the internals grow.

    Attributes
    ----------
    query_id : str
        Caller-supplied identifier for the query (e.g. GSM8K question index).
    probe_score : float
        The raw probe output that triggered the violation verdict.
    timestamp_utc : str
        ISO-8601 UTC timestamp at the moment the event was emitted.
    """

    def __init__(self, query_id: str, probe_score: float, timestamp_utc: str) -> None:
        self.query_id = query_id
        self.probe_score = probe_score
        self.timestamp_utc = timestamp_utc

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "probe_score": self.probe_score,
            "timestamp_utc": self.timestamp_utc,
        }


# ---------------------------------------------------------------------------
# Tier21ProbeWrapper — the production Tier 2.1 gate
# ---------------------------------------------------------------------------


class Tier21ProbeWrapper:
    """Wrap JEPAReasonerProbe as a drop-in Tier 2.1 gate in the cascade pipeline.

    The wrapper handles three responsibilities:
      1. Scoring: extract hidden state (or accept pre-extracted) and run the MLP probe.
      2. Thresholding: compare score to the calibrated threshold and emit a verdict.
      3. Stub event bus: fire emit_violation_stub() for every likely_violation verdict.

    The JEPAReasonerProbe inside must already have probe weights loaded (either via
    train_probe() or via load_probe_weights()).  This wrapper does NOT train — it only
    runs inference.

    Parameters
    ----------
    probe : JEPAReasonerProbe
        A fully-initialised probe with weights loaded.  Must be able to accept a
        numpy hidden state via predict().
    threshold : float
        The calibrated probe score threshold (5th percentile of correct-step scores
        on FoVer v2, per REQ-VER-035-1).  Scores <= threshold → "likely_correct".
        Scores > threshold → "likely_violation".
    event_bus : callable | None
        Optional callable that accepts a ViolationEventStub.  If None, a default
        in-memory stub list is used.  Exp 734 replaces this with the real FR11EventBus.
    violation_log : list | None
        When event_bus is None, violation events are appended to this list.  If also
        None, a new empty list is created internally and accessible as self.violation_log.

    Spec: REQ-VER-035, REQ-VER-037
    """

    def __init__(
        self,
        probe: JEPAReasonerProbe,
        threshold: float,
        event_bus: Optional[Callable[[ViolationEventStub], None]] = None,
        violation_log: Optional[list[ViolationEventStub]] = None,
        fr11_bus: Optional[FR11EventBus] = None,
    ) -> None:
        self._probe = probe
        self.threshold = threshold
        self._fr11_bus = fr11_bus

        # In-memory stub event bus — kept for backward-compatibility with callers
        # that have not yet wired fr11_bus.  Exp 734 passes fr11_bus; when present
        # the stub path is bypassed and ViolationEvent (not ViolationEventStub) is
        # published to the real bus.
        if violation_log is None:
            self.violation_log: list[ViolationEventStub] = []
        else:
            self.violation_log = violation_log

        # If a real bus is supplied use it; otherwise append to the in-memory list.
        if event_bus is not None:
            self._event_bus = event_bus
        else:
            self._event_bus = self.violation_log.append  # type: ignore[assignment]

    def score(self, hidden_state: np.ndarray) -> tuple[float, str]:
        """Score a pre-extracted hidden state and return (probe_score, verdict).

        WHY hidden_state not query_text:
            Hidden state extraction requires the LLM forward pass, which is expensive
            and owned by the calling pipeline (it must happen once regardless of what
            downstream tiers run).  Accepting a pre-extracted vector keeps this wrapper
            pure CPU and sub-1ms, matching REQ-VER-034-2.

        Parameters
        ----------
        hidden_state : np.ndarray
            Float32 vector of shape (hidden_dim,) from JEPAReasonerProbe.extract_hidden_state().

        Returns
        -------
        tuple[float, str]
            (probe_score, verdict) where verdict is "likely_correct" or "likely_violation".

        Spec: REQ-VER-035-3, REQ-VER-036
        """
        probe_score: float = self._probe.predict(hidden_state)
        # Score <= threshold means the step is below the violation band → likely correct.
        # Score > threshold means the step enters the violation band → route to Tier 2.5+.
        if probe_score <= self.threshold:
            verdict = "likely_correct"
        else:
            verdict = "likely_violation"
        return probe_score, verdict

    def emit_violation_stub(
        self,
        query_id: str,
        probe_score: float,
        constraint_type: str = "probe_violation",
        question_domain: str = "unknown",
        step_index: int = 0,
        energy_score: float = 0.0,
    ) -> None:
        """Emit a ViolationEvent to the FR11EventBus (REQ-VER-037, REQ-FR11-001).

        When ``fr11_bus`` was supplied at construction time, this publishes a full
        ViolationEvent to the real FR11EventBus (Exp 734 wiring).  Otherwise it falls
        back to the legacy stub path (in-memory list or user-supplied callable) for
        backward-compatibility with callers that have not yet migrated.

        Parameters
        ----------
        query_id : str
            Identifier for the query that triggered the violation (e.g. "gsm8k_0042").
        probe_score : float
            The raw probe score that exceeded the threshold.
        constraint_type : str
            Coarse category of the violation (default "probe_violation").
        question_domain : str
            Topic area of the question (default "unknown").
        step_index : int
            Which reasoning step index triggered the violation.
        energy_score : float
            EORM confidence for context.

        Spec: REQ-VER-037-1, REQ-VER-037-3, REQ-FR11-001
        """
        from datetime import datetime, timezone  # noqa: PLC0415

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if self._fr11_bus is not None:
            # Real FR11EventBus path — full ViolationEvent with all fields.
            real_event = ViolationEvent(
                query_id=query_id,
                step_index=step_index,
                energy_score=energy_score,
                probe_confidence=probe_score,
                constraint_type=constraint_type,
                question_domain=question_domain,
                timestamp=ts,
            )
            self._fr11_bus.publish(real_event)
        else:
            # Legacy stub path — backward-compatible with pre-Exp734 callers.
            stub = ViolationEventStub(query_id=query_id, probe_score=probe_score, timestamp_utc=ts)
            self._event_bus(stub)
