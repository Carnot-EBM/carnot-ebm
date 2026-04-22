"""FR-11 Event Bus — pub-sub relay for Tier 2.1 violation events to Tier 1 feedback.

**Why this module exists (REQ-FR11-001):**
    FR-11 (Autonomous Self-Learning Loop) has been blocked for 15+ milestones because
    the violation detector (JEPA) never reached reliable AUC.  Tier 2.1 (JEPAReasonerProbe,
    AUC >= 0.75 confirmed by Exps 732-733) is now the violation signal source.

    This bus is the wiring layer: Tier 2.1 publishes a ViolationEvent whenever the probe
    exceeds the calibrated threshold.  Subscribers (PerModelFPTracker, SessionMemory)
    react in under 200ms.  The design is synchronous and single-process — no threading,
    no queues — because latency requirements are soft (200ms) and the number of
    subscribers is small (2-4).  Async complexity is not justified here.

**How the pub-sub contract works:**
    - ``subscribe(fn)`` registers a callable that accepts a ViolationEvent.
    - ``publish(event)`` iterates subscribers and calls each one.  Each successful
      call (no exception raised) increments ``events_acked``.
    - ``events_acked`` < ``events_published`` means at least one subscriber errored.
      The conductor monitors this ratio to detect wiring failures.

Spec: REQ-FR11-001, SCENARIO-FR11-001
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List


# ---------------------------------------------------------------------------
# ViolationEvent — the message passed through the bus
# ---------------------------------------------------------------------------


@dataclass
class ViolationEvent:
    """One violation detected by the Tier 2.1 JEPAReasonerProbe.

    **Why these fields and not others:**
        - ``query_id``: lets subscribers look up additional context if needed.
        - ``step_index``: which reasoning step triggered the violation (for
          multi-step CoT; 0 when not applicable).
        - ``energy_score``: raw EORM confidence that preceded the Tier 2.1 check.
          Stored so Tier 1 can weight updates proportionally to energy if desired.
        - ``probe_confidence``: the Tier 2.1 probe output (P(violation)).  Used by
          PerModelFPTracker to log severity.
        - ``constraint_type``: coarse category of the violation (e.g. "carry_check",
          "sign_check", "unit_consistency").  This is the key Tier 1 uses to
          increment the right constraint weight.
        - ``question_domain``: topic area of the question (e.g. "arithmetic",
          "algebra").  SessionMemory uses this to cluster cross-domain patterns.
        - ``timestamp``: ISO-8601 UTC string at emission time.  Used for latency
          measurement in Exp 734.

    Spec: REQ-FR11-001
    """

    query_id: str
    step_index: int
    energy_score: float
    probe_confidence: float
    constraint_type: str
    question_domain: str
    timestamp: str


# ---------------------------------------------------------------------------
# FR11EventBus
# ---------------------------------------------------------------------------


class FR11EventBus:
    """Pub-sub bus delivering ViolationEvents from Tier 2.1 to all registered subscribers.

    **Detailed explanation for engineers:**
        This is the core wiring for FR-11.  The Tier 2.1 probe calls ``publish()``
        whenever it detects a likely violation.  All subscribers registered via
        ``subscribe()`` are called synchronously in registration order.  The bus
        does NOT catch exceptions from subscribers — a subscriber that raises will
        propagate the exception to the caller.  This is intentional: silent failure
        in a subscriber is harder to detect than a raised exception.

        ``events_acked`` counts how many subscriber invocations completed without
        error.  ``events_published`` counts how many ``publish()`` calls were made.
        If ``events_acked < events_published * len(subscribers)``, some subscriber
        errored.

        **Why synchronous?**
            The 200ms latency requirement (REQ-FR11-001) is easy to meet with a
            synchronous call since each subscriber (weight update, cache write) is
            O(1) in-memory work.  Async would add queue management complexity with
            no benefit at this scale.

    Attributes:
        events_published: Total number of ViolationEvents published (incremented on
                          each ``publish()`` call).
        events_acked:     Total number of successful subscriber invocations across
                          all publish() calls.

    Spec: REQ-FR11-001, SCENARIO-FR11-001, SCENARIO-FR11-002
    """

    def __init__(self) -> None:
        self._subscribers: List[Callable[[ViolationEvent], None]] = []
        self.events_published: int = 0
        self.events_acked: int = 0

    def subscribe(self, subscriber_fn: Callable[[ViolationEvent], None]) -> None:
        """Register a subscriber callable.

        **Why duplicates are allowed:**
            Preventing duplicate registration requires tracking subscriber identity,
            which is non-trivial for lambdas and bound methods.  The experiment
            setup scripts are responsible for not registering the same subscriber
            twice.

        Args:
            subscriber_fn: Callable that accepts a ViolationEvent.  Called
                           synchronously on each publish().

        Spec: REQ-FR11-001-1
        """
        self._subscribers.append(subscriber_fn)

    def publish(self, event: ViolationEvent) -> None:
        """Deliver a ViolationEvent to all registered subscribers.

        Calls each subscriber synchronously in registration order.  Each call
        that completes without raising increments ``events_acked``.

        Args:
            event: The ViolationEvent to deliver.

        Spec: REQ-FR11-001-2, REQ-FR11-001-3
        """
        self.events_published += 1
        for fn in self._subscribers:
            fn(event)
            self.events_acked += 1

    def measure_publish_latency_ms(self, event: ViolationEvent) -> float:
        """Publish an event and return the wall-clock time for all subscribers to complete.

        Used by Exp 734 to measure relay_latency_p99_ms.

        Returns:
            Elapsed milliseconds for all subscribers to complete.

        Spec: REQ-FR11-001 (200ms delivery requirement)
        """
        t0 = time.perf_counter()
        self.publish(event)
        return (time.perf_counter() - t0) * 1000.0
