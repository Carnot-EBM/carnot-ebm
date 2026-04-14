"""Z3-gated repair pipeline: cheap Z3 first-pass gate before expensive Ising repair.

**Researcher summary:**
    Exp 311 benchmarked all three extractors and confirmed that NL2Z3Extractor
    produces the lowest false-positive rate (Z3 either proves UNSAT — a
    definitive contradiction — or returns SAT/unknown, so unverified responses
    are never repaired).  This module wires Z3 as a mandatory cheap gate in
    front of the full Ising + LLM repair loop.

    Gate logic (in order of decreasing cheapness):
    1. Run NL2Z3Extractor → Z3Result (one LLM call + subprocess; ~100–500 ms).
    2. If sat_status="sat":  skip Ising entirely (fast exit, ~0 ms).
    3. If sat_status="unsat": trigger ConfidenceVerifier + Ising repair.
    4. If sat_status="unknown" or "error": fallback to confidence-weighted Ising.

**Detailed explanation for engineers:**
    WHY this two-gate design matters:
    - In a 30-question benchmark run, roughly half the responses will be
      consistent (SAT).  Those never need repair.  The gate eliminates
      half the Ising + LLM calls at the cost of one Z3 pass each.
    - Z3 UNSAT is a *hard* proof of contradiction — it has effectively zero
      false positives.  We can trigger the full repair path confidently.
    - Z3 unknown/error is conservative: we don't skip; we fall back to the
      existing confidence-weighted path (REQ-VERIFY-082).

    Components:
    - ``Z3GatedRepairResult``: captures z3_status, ising_triggered,
      repair outcome, and runtime so callers can compute aggregate metrics.
    - ``Z3GatedRepair``: the gate orchestrator.  Accepts an injectable
      NL2Z3Extractor and an injectable Ising pipeline for testability.
    - ``compute_skip_rate(results)``: helper to compute the Z3 skip fraction
      from a list of Z3GatedRepairResult records.

    The ising_pipeline is expected to implement
    ``verify_and_repair_confident(question, response, domain, threshold)``,
    matching the VerifyRepairPipeline contract.

    CI mode:
    - NL2Z3Extractor always returns "unknown" when CARNOT_FORCE_LIVE is not
      set.  In CI, every question takes the fallback path (Ising triggered
      but no real LLM calls, so verify_and_repair_confident is exercised with
      the same mocked-model behaviour as unit tests).

Spec: REQ-REPAIR-010, REQ-REPAIR-011,
      SCENARIO-REPAIR-020, SCENARIO-REPAIR-021, SCENARIO-REPAIR-022
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carnot.pipeline.nl2z3_extractor import NL2Z3Extractor


# ---------------------------------------------------------------------------
# Z3GatedRepairResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class Z3GatedRepairResult:
    """Outcome of one Z3-gated repair call.

    **Detailed explanation for engineers:**
        Captures everything needed for per-question logging and aggregate
        metric computation (skip_rate, trigger_rate, net_improvement).

        Fields:
        - z3_status:       Z3 verdict — "sat", "unsat", "unknown", or "error".
        - z3_code:         Z3 Python code generated and executed (may be "").
        - ising_triggered: True when the Ising repair path was invoked.
                           False only when z3_status=="sat" (fast exit).
        - ising_violations: Number of violations found by Ising when triggered;
                            0 when Ising was not triggered or found none.
        - repair_attempted: True when an LLM repair attempt was made.
        - repaired:         True when the response was changed and passes
                            verification after repair.
        - improvement:      Integer delta — 1 if repaired, 0 otherwise.
                            Reported honestly; 0 is never suppressed.
        - runtime_ms:       Total wall-clock time for this gate call in ms.

    Spec: REQ-REPAIR-010, SCENARIO-REPAIR-020, SCENARIO-REPAIR-022
    """

    z3_status: str          # "sat" | "unsat" | "unknown" | "error"
    z3_code: str            # the Z3 Python code that was generated
    ising_triggered: bool   # True iff Ising pipeline was invoked
    ising_violations: int   # violation count from first Ising check (0 if not triggered)
    repair_attempted: bool  # True iff an LLM repair attempt was made
    repaired: bool          # True iff the final response passes verification
    improvement: int        # 1 if repaired, 0 otherwise (honest — never suppressed)
    runtime_ms: float       # wall-clock time for the entire gate call


# ---------------------------------------------------------------------------
# skip_rate helper
# ---------------------------------------------------------------------------


def compute_skip_rate(results: list[Z3GatedRepairResult]) -> float:
    """Compute the fraction of questions where Z3 SAT skipped Ising repair.

    **Detailed explanation for engineers:**
        The skip rate is the primary efficiency metric for the Z3 gate.  A
        high skip rate means most responses were consistent (SAT) and we
        avoided expensive Ising + LLM repair calls.

        Edge case: empty list returns 0.0 (no questions processed).

    Args:
        results: List of Z3GatedRepairResult from a benchmark run.

    Returns:
        Float in [0, 1]: fraction where ising_triggered is False.

    Spec: REQ-REPAIR-010
    """
    if not results:
        return 0.0
    skipped = sum(1 for r in results if not r.ising_triggered)
    return skipped / len(results)


# ---------------------------------------------------------------------------
# Z3GatedRepair orchestrator
# ---------------------------------------------------------------------------


class Z3GatedRepair:
    """Z3 first-gate orchestrator for the verify-repair pipeline.

    **Detailed explanation for engineers:**
        Wraps an NL2Z3Extractor (the Z3 gate) and a VerifyRepairPipeline-like
        object (the Ising repair stage).  On each call to ``repair()``:

        1. Run NL2Z3Extractor to get a Z3Result.
        2. If sat: return immediately (ising_triggered=False).
        3. If unsat: call ising_pipeline.verify_and_repair_confident() with
           the caller-supplied confidence_threshold.
        4. If unknown/error: same as unsat — invoke the confidence-weighted
           repair path as a conservative fallback.

        Injectable dependencies:
        - ``nl2z3_extractor``: Any object with ``extract(q, r, domain)``
          and ``last_z3_result`` — matches NL2Z3Extractor's public API.
          In production: a real NL2Z3Extractor.
          In tests: a MagicMock returning canned Z3Results.
        - ``ising_pipeline``: Any object with
          ``verify_and_repair_confident(question, response, domain, threshold)``.
          In production: a VerifyRepairPipeline instance.
          In tests: a MagicMock.

    Args:
        nl2z3_extractor:      The Z3 gate extractor.
        ising_pipeline:       The Ising + LLM repair pipeline.
        confidence_threshold: Minimum confidence to trigger Ising repair when
                              Z3 returns unsat/unknown.  Default 0.8.

    Spec: REQ-REPAIR-010, REQ-REPAIR-011, SCENARIO-REPAIR-020,
          SCENARIO-REPAIR-021, SCENARIO-REPAIR-022
    """

    def __init__(
        self,
        nl2z3_extractor: Any,
        ising_pipeline: Any,
        confidence_threshold: float = 0.8,
    ) -> None:
        self._extractor = nl2z3_extractor
        self._ising = ising_pipeline
        self.confidence_threshold = confidence_threshold

    def repair(
        self,
        question: str,
        response: str,
        domain: str | None = None,
    ) -> Z3GatedRepairResult:
        """Run the Z3 gate and optionally invoke Ising repair.

        **Detailed explanation for engineers:**
            The three code paths share the same timing wrapper so runtime_ms
            always reflects the full cost of this call:

            Path A — SAT (fast exit):
                NL2Z3Extractor says the reasoning is consistent.  We trust Z3
                and return without touching the Ising pipeline.  This is the
                "cheap path" that justifies having the gate at all.

            Path B — UNSAT (triggered repair):
                Z3 proved a contradiction.  We have high confidence the
                response is wrong, so we invoke the full Ising + LLM repair
                loop.  The repair may or may not succeed; we record the outcome
                honestly either way.

            Path C — UNKNOWN / ERROR (conservative fallback):
                Z3 could not reach a verdict (CI mode, timeout, or bad code).
                We fall back to the confidence-weighted Ising path rather than
                silently skipping, because skipping on uncertainty would be
                wrong in the same direction as over-repairing: both hide errors.

        Args:
            question: The original question posed to the LLM.
            response: The LLM response to evaluate and potentially repair.
            domain:   Optional domain hint for NL2Z3Extractor.

        Returns:
            Z3GatedRepairResult with full outcome and timing metadata.

        Spec: REQ-REPAIR-010, REQ-REPAIR-011, SCENARIO-REPAIR-020,
              SCENARIO-REPAIR-021, SCENARIO-REPAIR-022
        """
        start = time.monotonic()

        # Step 1: Run NL2Z3Extractor to get the Z3 verdict.
        self._extractor.extract(question, response, domain)
        z3_result = self._extractor.last_z3_result

        # Defensive: if last_z3_result is None (shouldn't happen, but guard anyway)
        if z3_result is None:
            z3_status = "unknown"
            z3_code = ""
        else:
            z3_status = z3_result.sat_status
            z3_code = z3_result.z3_code

        # Step 2: SAT → fast exit (skip Ising entirely).
        if z3_status == "sat":
            runtime_ms = (time.monotonic() - start) * 1000.0
            return Z3GatedRepairResult(
                z3_status="sat",
                z3_code=z3_code,
                ising_triggered=False,
                ising_violations=0,
                repair_attempted=False,
                repaired=False,
                improvement=0,
                runtime_ms=runtime_ms,
            )

        # Steps 3 & 4: UNSAT or UNKNOWN/ERROR → invoke confidence-weighted Ising.
        # Both unsat (proved contradiction) and unknown/error (conservative fallback)
        # go through the same Ising path; the distinction is captured in z3_status.
        ising_result = self._ising.verify_and_repair_confident(
            question,
            response,
            domain,
            self.confidence_threshold,
        )

        # Count violations from first history entry (if any)
        ising_violations = 0
        if getattr(ising_result, "history", None):
            first_check = ising_result.history[0]
            ising_violations = len(getattr(first_check, "violations", []))

        repaired = bool(getattr(ising_result, "repaired", False))
        repair_attempted = bool(
            getattr(ising_result, "iterations", 0) > 0
            or getattr(ising_result, "repair_attempted", repaired)
        )

        runtime_ms = (time.monotonic() - start) * 1000.0
        return Z3GatedRepairResult(
            z3_status=z3_status,
            z3_code=z3_code,
            ising_triggered=True,
            ising_violations=ising_violations,
            repair_attempted=repair_attempted,
            repaired=repaired,
            improvement=1 if repaired else 0,
            runtime_ms=runtime_ms,
        )
