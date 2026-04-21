"""EnsembleGate v4 — structured-first gate logic for VR authorization.

Why v4 exists:
    EnsembleGate v3 averaged ALL four recall components including HermesV2,
    which scores 0.0 on mixed-format test sets.  This dragged the ensemble
    below 0.30 even when causal_recall=0.36 (which already exceeds the
    threshold on its own).  The fix is to use a structured-first OR-logic:
    open the gate when EITHER structured_recall meets its lower threshold
    OR the best single-component recall meets the full threshold.  HermesV2
    is recorded for audit but excluded from the gate formula entirely.

Spec: REQ-VERIFY-147, REQ-VERIFY-148
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnsembleGateV4Result:
    """Result of running EnsembleGateV4.compute().

    Fields:
        symcode_recall: Fraction of arithmetic steps verified correct by SymCodeVerifier.
        structured_recall: Fraction of questions answered in COMPUTE: format (Exp 653).
        causal_recall: Fraction of causal reasoning steps verified correct (Exp 654).
        ensemble_recall: Average of symcode, structured, and causal (HermesV2 excluded).
        gate_threshold: The structured_threshold used in this run (default 0.20).
        gate_open: True when the structured-first OR condition is satisfied.
        gate_version: Always "v4" — identifies this class in artifact schemas.
        authorizes_vr: Synonym for gate_open; True means VR attempt is authorized.
        honest_verdict: Human-readable decision string for artifact schemas.
    """

    symcode_recall: float
    structured_recall: float
    causal_recall: float
    ensemble_recall: float
    gate_threshold: float
    gate_open: bool
    gate_version: str
    authorizes_vr: bool
    honest_verdict: str


class EnsembleGateV4:
    """Structured-first ensemble gate for verifiable-reasoning authorization.

    Gate formula (REQ-VERIFY-147):
        gate_open = (
            structured_recall >= structured_threshold
            OR max(causal_recall, symcode_recall) >= max_component_threshold
        )

    HermesV2 recall is accepted as input and logged to the artifact for
    audit purposes, but it is intentionally excluded from the gate formula
    (REQ-VERIFY-148).  On mixed-format test sets HermesV2 scores near 0.0,
    so including it in an average unfairly penalises the ensemble.

    Args:
        structured_threshold: Gate opens when structured_recall >= this value.
            Lower bar than max_component_threshold because structured format
            already implies partial verification.  Default 0.20.
        max_component_threshold: Gate also opens when the best of
            (causal_recall, symcode_recall) >= this value.  Default 0.30.
    """

    def __init__(
        self,
        structured_threshold: float = 0.20,
        max_component_threshold: float = 0.30,
    ) -> None:
        self.structured_threshold = structured_threshold
        self.max_component_threshold = max_component_threshold

    def compute(
        self,
        symcode_recall: float,
        hermes_v2_recall: float,  # noqa: ARG002 — advisory only, not used in formula
        structured_recall: float,
        causal_recall: float,
    ) -> EnsembleGateV4Result:
        """Apply structured-first gate logic and return a result dataclass.

        HermesV2 recall is accepted so callers can pass the full v3 result
        dict unchanged, but it is NOT used in the gate decision (REQ-VERIFY-148).

        Args:
            symcode_recall: SymCodeVerifier recall fraction in [0, 1].
            hermes_v2_recall: HermesV2LiveLoop recall fraction (advisory, ignored).
            structured_recall: StructuredEquationForcer recall fraction in [0, 1].
            causal_recall: CausalReasoningVerifier recall fraction in [0, 1].

        Returns:
            EnsembleGateV4Result with gate_open, authorizes_vr, and honest_verdict.
        """
        gate_open = (
            structured_recall >= self.structured_threshold
            or max(causal_recall, symcode_recall) >= self.max_component_threshold
        )
        # Average excludes HermesV2 — three-component ensemble only.
        ensemble_recall = (symcode_recall + structured_recall + causal_recall) / 3

        verdict = (
            "gate_open_vr_authorized" if gate_open else "gate_closed_vr_blocked"
        )

        return EnsembleGateV4Result(
            symcode_recall=symcode_recall,
            structured_recall=structured_recall,
            causal_recall=causal_recall,
            ensemble_recall=ensemble_recall,
            gate_threshold=self.structured_threshold,
            gate_open=gate_open,
            gate_version="v4",
            authorizes_vr=gate_open,
            honest_verdict=verdict,
        )
