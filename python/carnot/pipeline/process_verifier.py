"""Process-integrity verifier for typed reasoning traces and code-repair traces.

Detects "correct answer, invalid process" patterns and other structural defects
in step-by-step reasoning and iterative repair without touching the existing
semantic or code verifiers.

Six defect kinds are detected:
- unsupported_step:               n_unsupported_claims > 0 in process_evidence
- missing_premise_jump:           max_premise_support < 1.0 AND unsupported claims exist
- contradictory_intermediate:     verifier_verdict == "violated" in process_evidence
- outcome_correct_process_invalid: outcome_label == "correct" AND
                                   process_label == "right_answer_wrong_process"
- repair_regression:              repair_context.prior_outcome == "correct"
                                  AND current outcome_label == "incorrect"
- repair_stall:                   repair_context.prior_outcome == "incorrect"
                                  AND current outcome_label == "incorrect"

The verifier is purely additive.  Existing semantic and code verifiers are
unaffected: call them independently and optionally merge results.

Spec: REQ-VERIFY-061, REQ-VERIFY-062
SCENARIO-VERIFY-065, SCENARIO-VERIFY-066, SCENARIO-VERIFY-067,
SCENARIO-VERIFY-068, SCENARIO-VERIFY-069
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carnot.pipeline.typed_reasoning import TypedReasoningIR

# Fixed run-date embedded in every result for traceability.
RUN_DATE = "20260413"

# ---------------------------------------------------------------------------
# Closed vocabulary for defect kinds
# ---------------------------------------------------------------------------

UNSUPPORTED_STEP = "unsupported_step"
MISSING_PREMISE_JUMP = "missing_premise_jump"
CONTRADICTORY_INTERMEDIATE = "contradictory_intermediate"
OUTCOME_CORRECT_PROCESS_INVALID = "outcome_correct_process_invalid"
REPAIR_REGRESSION = "repair_regression"
REPAIR_STALL = "repair_stall"

# All valid defect kind literals so callers can rely on exact string equality.
ALL_DEFECT_KINDS: frozenset[str] = frozenset(
    {
        UNSUPPORTED_STEP,
        MISSING_PREMISE_JUMP,
        CONTRADICTORY_INTERMEDIATE,
        OUTCOME_CORRECT_PROCESS_INVALID,
        REPAIR_REGRESSION,
        REPAIR_STALL,
    }
)

# process_label values that indicate an invalid reasoning process.
_INVALID_PROCESS_LABELS: frozenset[str] = frozenset(
    {
        "right_answer_wrong_process",
        "wrong_answer_wrong_process",
    }
)


# ---------------------------------------------------------------------------
# Defect record
# ---------------------------------------------------------------------------


@dataclass
class ProcessDefect:
    """One detected process-integrity defect.

    Attributes:
        kind:     One of the six defect kind strings from ``ALL_DEFECT_KINDS``.
        detail:   Human-readable explanation of why the defect was raised.
        step_id:  Optional reference to the step or claim that triggered it.
        evidence: Machine-readable key-value pairs for downstream auditing.
    """

    kind: str
    detail: str
    step_id: str | None = None
    evidence: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministically ordered dict (sorted keys)."""
        return {
            "evidence": dict(sorted(self.evidence.items())),
            "kind": self.kind,
            "detail": self.detail,
            "step_id": self.step_id,
        }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProcessVerificationResult:
    """Structured result for one process-integrity verification call.

    Attributes:
        process_valid:  ``True`` only when zero defects are detected.
        outcome_correct: ``True``/``False`` from outcome_label, or ``None``
                         when outcome is unknown.
        defects:        All detected defects (may be empty).
        process_label:  The raw process_label from the corpus row, or
                        ``"unknown"`` when the row did not carry one.
        run_date:       Fixed string ``"20260413"`` for traceability.
    """

    process_valid: bool
    outcome_correct: bool | None
    defects: list[ProcessDefect]
    process_label: str
    run_date: str = RUN_DATE

    def to_dict(self) -> dict[str, object]:
        """Return a deterministically ordered dict (sorted keys)."""
        return {
            "defects": [d.to_dict() for d in self.defects],
            "outcome_correct": self.outcome_correct,
            "process_label": self.process_label,
            "process_valid": self.process_valid,
            "run_date": self.run_date,
        }

    def to_json(self) -> str:
        """Deterministic JSON serialization with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _outcome_correct(outcome_label: str | None) -> bool | None:
    """Map 'correct'/'incorrect' → True/False; anything else → None."""
    if outcome_label == "correct":
        return True
    if outcome_label == "incorrect":
        return False
    return None


def _detect_evidence_defects(
    evidence: dict[str, Any],
    outcome_label: str | None,
    process_label: str,
) -> list[ProcessDefect]:
    """Detect defects from pre-computed process_evidence and labels.

    This is the primary path for corpus rows that carry numeric evidence
    produced by Exp 235 / 248 pipelines.

    Args:
        evidence:      The ``process_evidence`` sub-dict from the corpus row.
        outcome_label: ``"correct"`` or ``"incorrect"``.
        process_label: The corpus process_label string.

    Returns:
        List of ``ProcessDefect`` objects (may be empty).
    """
    defects: list[ProcessDefect] = []

    n_unsupported: int = int(evidence.get("n_unsupported_claims", 0))
    max_support: float = float(evidence.get("max_premise_support", 1.0))
    verifier_verdict: str = str(evidence.get("verifier_verdict", ""))

    # 1. Unsupported steps: any claim without premise support.
    if n_unsupported > 0:
        defects.append(
            ProcessDefect(
                kind=UNSUPPORTED_STEP,
                detail=(
                    f"{n_unsupported} claim(s) had no premise support "
                    f"in the reasoning trace."
                ),
                evidence={
                    "n_unsupported_claims": n_unsupported,
                    "max_premise_support": max_support,
                },
            )
        )

    # 2. Missing-premise jump: max_support < 1.0 combined with unsupported claims.
    if max_support < 1.0 and n_unsupported > 0:
        defects.append(
            ProcessDefect(
                kind=MISSING_PREMISE_JUMP,
                detail=(
                    f"At least one reasoning step was accepted with only "
                    f"{max_support:.2f} premise support (threshold: 1.0)."
                ),
                evidence={
                    "max_premise_support": max_support,
                    "n_unsupported_claims": n_unsupported,
                },
            )
        )

    # 3. Contradictory intermediate state: verifier found a violation inside
    #    the trace, regardless of the final answer.
    if verifier_verdict == "violated":
        defects.append(
            ProcessDefect(
                kind=CONTRADICTORY_INTERMEDIATE,
                detail=(
                    "The semantic verifier reported a violation inside the "
                    "reasoning trace (verifier_verdict='violated')."
                ),
                evidence={"verifier_verdict": verifier_verdict},
            )
        )

    # 4. Outcome-correct but process-invalid: the model lucked into the right
    #    answer via a provably flawed process.
    if outcome_label == "correct" and process_label in _INVALID_PROCESS_LABELS:
        defects.append(
            ProcessDefect(
                kind=OUTCOME_CORRECT_PROCESS_INVALID,
                detail=(
                    f"Final answer is correct but the process was labeled "
                    f"'{process_label}' — the answer cannot be trusted to "
                    f"generalise from this trace."
                ),
                evidence={
                    "outcome_label": outcome_label,
                    "process_label": process_label,
                },
            )
        )

    return defects


def _detect_repair_defects(
    repair_context: dict[str, Any] | None,
    outcome_label: str | None,
) -> list[ProcessDefect]:
    """Detect repair-specific defects from repair_context.

    Args:
        repair_context: The ``repair_context`` sub-dict from the corpus row,
                        or ``None`` when no repair was attempted.
        outcome_label:  ``"correct"`` or ``"incorrect"`` for this iteration.

    Returns:
        List of ``ProcessDefect`` objects (may be empty).
    """
    if repair_context is None:
        return []

    defects: list[ProcessDefect] = []
    prior_outcome: str | None = repair_context.get("prior_outcome")

    # Repair regression: previous iteration was correct, this one is not.
    if prior_outcome == "correct" and outcome_label == "incorrect":
        defects.append(
            ProcessDefect(
                kind=REPAIR_REGRESSION,
                detail=(
                    "A repair iteration turned a previously correct answer "
                    "incorrect (prior_outcome='correct' → outcome='incorrect')."
                ),
                evidence={
                    "prior_outcome": prior_outcome,
                    "current_outcome": outcome_label,
                },
            )
        )

    # Repair stall: both previous and current iterations are incorrect.
    if prior_outcome == "incorrect" and outcome_label == "incorrect":
        defects.append(
            ProcessDefect(
                kind=REPAIR_STALL,
                detail=(
                    "A repair iteration failed to recover a correct answer "
                    "(prior_outcome='incorrect' → outcome='incorrect')."
                ),
                evidence={
                    "prior_outcome": prior_outcome,
                    "current_outcome": outcome_label,
                },
            )
        )

    return defects


def _detect_ir_defects(ir: TypedReasoningIR) -> list[ProcessDefect]:
    """Detect defects from a TypedReasoningIR without pre-computed evidence.

    Cross-references claim-to-step grounding: claims that carry no step_id
    are treated as unsupported.  When some claims are ungrounded, a
    missing_premise_jump is also raised because the final answer has an
    incomplete support chain.

    Args:
        ir: A validated ``TypedReasoningIR`` from
            ``carnot.pipeline.typed_reasoning``.

    Returns:
        List of ``ProcessDefect`` objects (may be empty).
    """
    defects: list[ProcessDefect] = []

    # Collect step ids for cross-reference (TypedReasoningIR uses reasoning_steps).
    step_ids: frozenset[str] = frozenset(s.step_id for s in ir.reasoning_steps)

    # Identify claims that are not grounded to any known step
    # (TypedReasoningIR uses atomic_claims).
    ungrounded = [
        c
        for c in ir.atomic_claims
        if c.step_id is None or c.step_id not in step_ids
    ]

    n_unsupported = len(ungrounded)
    if n_unsupported > 0:
        sample_claim_ids = [c.claim_id for c in ungrounded[:3]]
        defects.append(
            ProcessDefect(
                kind=UNSUPPORTED_STEP,
                detail=(
                    f"{n_unsupported} claim(s) are not grounded to any "
                    f"reasoning step in the typed IR."
                ),
                evidence={
                    "n_unsupported_claims": n_unsupported,
                    "sample_ungrounded_claim_ids": sample_claim_ids,
                },
            )
        )
        defects.append(
            ProcessDefect(
                kind=MISSING_PREMISE_JUMP,
                detail=(
                    f"{n_unsupported} ungrounded claim(s) create a gap in "
                    f"the premise chain before the final answer."
                ),
                evidence={
                    "n_unsupported_claims": n_unsupported,
                    "sample_ungrounded_claim_ids": sample_claim_ids,
                },
            )
        )

    return defects


# ---------------------------------------------------------------------------
# Public verifier class
# ---------------------------------------------------------------------------


class ProcessVerifier:
    """Detect process-integrity defects in reasoning and code-repair traces.

    All methods are pure functions over their inputs (no hidden state).
    The class exists as a namespace for discoverability and to mirror the
    pattern used by ``FormalClaimVerifier`` and ``SpecCodeVerifier``.

    Spec: REQ-VERIFY-061
    """

    def verify_reasoning_trace(
        self,
        corpus_row: dict[str, Any],
    ) -> ProcessVerificationResult:
        """Verify a reasoning-trace corpus row for process integrity.

        Checks pre-computed evidence fields from the Exp 248 corpus schema:
        ``process_evidence``, ``outcome_label``, and ``process_label``.

        Args:
            corpus_row: One row dict from ``process_integrity_corpus_248.jsonl``
                        or any dict following the same schema.

        Returns:
            ``ProcessVerificationResult`` with all detected defects.

        Spec: REQ-VERIFY-061, SCENARIO-VERIFY-065, SCENARIO-VERIFY-066,
              SCENARIO-VERIFY-067
        """
        evidence: dict[str, Any] = corpus_row.get("process_evidence") or {}
        outcome_label: str | None = corpus_row.get("outcome_label")
        process_label: str = str(corpus_row.get("process_label") or "unknown")
        repair_context: dict[str, Any] | None = corpus_row.get("repair_context")

        defects: list[ProcessDefect] = []
        defects.extend(_detect_evidence_defects(evidence, outcome_label, process_label))
        defects.extend(_detect_repair_defects(repair_context, outcome_label))

        return ProcessVerificationResult(
            process_valid=len(defects) == 0,
            outcome_correct=_outcome_correct(outcome_label),
            defects=defects,
            process_label=process_label,
            run_date=RUN_DATE,
        )

    def verify_code_repair_trace(
        self,
        corpus_row: dict[str, Any],
    ) -> ProcessVerificationResult:
        """Verify a code-repair trace corpus row for process integrity.

        Applies the same evidence checks as ``verify_reasoning_trace`` plus
        repair-specific regression and stall detection.  The two methods
        produce identical output for rows that carry no ``repair_context``.

        Args:
            corpus_row: A corpus row dict that may include ``repair_context``
                        with a ``prior_outcome`` field.

        Returns:
            ``ProcessVerificationResult`` with all detected defects.

        Spec: REQ-VERIFY-061, SCENARIO-VERIFY-068
        """
        # Code-repair traces use the same schema; delegate to the shared path.
        return self.verify_reasoning_trace(corpus_row)

    def verify_typed_reasoning(
        self,
        ir: TypedReasoningIR,
        outcome_correct: bool | None = None,
        process_evidence: dict[str, Any] | None = None,
    ) -> ProcessVerificationResult:
        """Verify a TypedReasoningIR for process integrity.

        Detects unsupported and missing-premise defects by cross-referencing
        claim-to-step grounding in the IR.  When ``process_evidence`` is
        provided, also applies the evidence-based checks from
        ``verify_reasoning_trace``.

        Args:
            ir:               A validated ``TypedReasoningIR`` instance.
            outcome_correct:  Whether the final answer is correct, or ``None``
                              when unknown.
            process_evidence: Optional pre-computed evidence dict (same schema
                              as the ``process_evidence`` field in corpus rows).

        Returns:
            ``ProcessVerificationResult`` with all detected defects.

        Spec: REQ-VERIFY-061
        """
        defects: list[ProcessDefect] = _detect_ir_defects(ir)

        if process_evidence:
            # Determine process_label from evidence when it's not a named argument.
            outcome_label: str | None = (
                "correct" if outcome_correct is True
                else "incorrect" if outcome_correct is False
                else None
            )
            defects.extend(
                _detect_evidence_defects(process_evidence, outcome_label, "unknown")
            )

        process_valid = len(defects) == 0
        return ProcessVerificationResult(
            process_valid=process_valid,
            outcome_correct=outcome_correct,
            defects=defects,
            process_label="unknown",
            run_date=RUN_DATE,
        )


# ---------------------------------------------------------------------------
# Convenience helper (mirrors verify_formal_claims pattern)
# ---------------------------------------------------------------------------


def verify_process_integrity(
    corpus_row: dict[str, Any],
) -> ProcessVerificationResult:
    """One-shot helper: verify a corpus row for process integrity.

    Selects between reasoning-trace and code-repair-trace paths based on
    whether ``repair_context`` is present.  Either way, all six defect kinds
    are checked.

    Args:
        corpus_row: One corpus row dict.

    Returns:
        ``ProcessVerificationResult``.

    Spec: REQ-VERIFY-061
    """
    verifier = ProcessVerifier()
    if corpus_row.get("repair_context") is not None:
        return verifier.verify_code_repair_trace(corpus_row)
    return verifier.verify_reasoning_trace(corpus_row)
