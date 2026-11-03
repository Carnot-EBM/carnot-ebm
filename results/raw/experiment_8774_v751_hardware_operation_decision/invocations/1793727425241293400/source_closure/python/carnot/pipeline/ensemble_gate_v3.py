"""Ensemble Recall Gate v3 — combines symcode, hermes_v2_structured, and causal signals.

**Researcher summary:**
    Gate v2 (Exp 643) used interwhen + hermes_v2 + causal signals to decide whether
    to authorise VR attempt #17.  That gate opened because causal_recall reached 0.36.

    Gate v3 replaces interwhen with the new structured-equation signal (Exp 654).
    StructuredEquationForcer forces the LLM to emit arithmetic in COMPUTE: format,
    making violations machine-checkable at generation time — a stronger signal than
    the post-hoc interwhen monitor.

    Weighted ensemble formula (weights sum to 1.0):
        ensemble_recall = (
            0.3 * symcode_recall   +   # SymCodeVerifier: Python-executable CoT checks
            0.4 * structured_recall +   # HermesV2StructuredLoop: COMPUTE: forcing
            0.3 * causal_recall         # CausalReasoningVerifier: step-to-step entailment
        )
    Gate opens when ensemble_recall >= threshold (default 0.30).

    hermes_v2_recall is tracked for historical traceability but intentionally excluded
    from the weighted formula — it has been consistently 0.0 across live runs (Exps 641,
    643), so including it would only dilute the meaningful signals.

Spec: REQ-VERIFY-149, SCENARIO-VERIFY-200, SCENARIO-VERIFY-201
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnsembleGateV3Result:
    """Recall signals and gate decision from EnsembleRecallGateV3.compute().

    Fields
    ------
    symcode_recall:
        Fraction of incorrect questions where SymCodeVerifier detected a violation.
        Source: Exp 619 or Exp 630.
    hermes_v2_recall:
        Fraction of incorrect questions where HermesV2LiveLoop detected a violation.
        Source: Exp 641.  Historically 0.0 — tracked but excluded from ensemble weight.
    structured_recall:
        Fraction of incorrect questions where HermesV2StructuredLoop (COMPUTE: forcing)
        detected a violation.  Source: Exp 654.
    causal_recall:
        Fraction of incorrect questions where CausalReasoningVerifier detected a
        step-level entailment break.  Source: Exp 642.
    ensemble_recall:
        Weighted combination: 0.3*symcode + 0.4*structured + 0.3*causal.
    gate_open:
        True when ensemble_recall >= threshold.  When True, VR attempt #18 is authorised.
    gate_version:
        Always 'v3' — identifies which gate formula produced this result.
    """

    symcode_recall: float
    hermes_v2_recall: float
    structured_recall: float
    causal_recall: float
    ensemble_recall: float
    gate_open: bool
    gate_version: str = field(default="v3")


class EnsembleRecallGateV3:
    """Compute the v3 ensemble recall and decide whether to open the VR gate.

    The three weighted signals represent orthogonal error-detection strategies:
    - symcode: executable Python checks on CoT arithmetic
    - structured: COMPUTE:-forced generation with per-line verification
    - causal: step-to-step entailment checks between consecutive CoT sentences

    Because these strategies catch different error classes (execution errors,
    format errors, and logical-flow errors respectively), their weighted combination
    is more robust than any single signal alone.

    Why exclude hermes_v2_recall from the weights?
    hermes_v2_recall has been 0.0 in every live run (Exps 641, 643).  Including it
    with any positive weight would reduce the ensemble below the level justified by
    the working signals.  It is stored in EnsembleGateV3Result for auditability.

    Parameters
    ----------
    symcode_weight:
        Weight for the SymCodeVerifier recall signal (default 0.3).
    structured_weight:
        Weight for the HermesV2StructuredLoop recall signal (default 0.4).
    causal_weight:
        Weight for the CausalReasoningVerifier recall signal (default 0.3).
    threshold:
        Minimum ensemble_recall to open the gate (default 0.30).
    """

    def __init__(
        self,
        symcode_weight: float = 0.3,
        structured_weight: float = 0.4,
        causal_weight: float = 0.3,
        threshold: float = 0.30,
    ) -> None:
        self.weights: dict[str, float] = {
            "symcode": symcode_weight,
            "structured": structured_weight,
            "causal": causal_weight,
        }
        self.threshold = threshold

    def compute(
        self,
        symcode_recall: float,
        hermes_v2_recall: float,
        structured_recall: float,
        causal_recall: float,
    ) -> EnsembleGateV3Result:
        """Compute weighted ensemble recall and gate decision.

        Parameters
        ----------
        symcode_recall:
            TP rate from SymCodeVerifier on the 25-question incorrect set.
        hermes_v2_recall:
            TP rate from HermesV2LiveLoop (tracked but not included in ensemble weight).
        structured_recall:
            TP rate from HermesV2StructuredLoop COMPUTE: forcing.
        causal_recall:
            TP rate from CausalReasoningVerifier entailment checking.

        Returns
        -------
        EnsembleGateV3Result
            Contains all recall signals, the computed ensemble_recall, and gate_open.
        """
        ensemble_recall = (
            self.weights["symcode"] * symcode_recall
            + self.weights["structured"] * structured_recall
            + self.weights["causal"] * causal_recall
        )
        gate_open = ensemble_recall >= self.threshold
        return EnsembleGateV3Result(
            symcode_recall=symcode_recall,
            hermes_v2_recall=hermes_v2_recall,
            structured_recall=structured_recall,
            causal_recall=causal_recall,
            ensemble_recall=ensemble_recall,
            gate_open=gate_open,
        )
