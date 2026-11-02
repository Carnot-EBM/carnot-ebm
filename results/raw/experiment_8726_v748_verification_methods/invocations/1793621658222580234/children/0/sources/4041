"""VerificationCertificate — per-step verification provenance record (arXiv 2601.17223).

**Researcher summary:**
    arXiv 2601.17223 "Beyond Outcome Verification" introduces the Verifiable PRM design:
    rather than only recording a binary pass/fail for a full response, a verifiable PRM
    emits a certificate for each individual reasoning step.  The certificate captures
    the specific constraint that was evaluated and the formal verdict from the solver.

    Carnot's VerificationCertificate maps directly to this design:
      - step_id:           unique string identifier (domain + step index)
      - jepa_energy_delta: cosine distance from JEPAv23Predictor.predict_energy();
                           higher = model thinks the step is less aligned with the prefix
      - constraint_type:   semantic category of the constraint checked ("arithmetic",
                           "code_logic", "planning")
      - z3_verdict:        formal solver verdict ("sat" = constraint satisfied, "unsat" =
                           violated, "unknown" = solver timeout or undecidable)
      - confidence_score:  float in [0, 1] derived from energy_delta via sigmoid;
                           high confidence means the model is certain about its verdict

    WHY a namedtuple instead of a dataclass:
        namedtuple is immutable and hashable, which makes certificates safe to store in
        sets or as dict keys.  The fields are fixed by the arXiv 2601.17223 spec and
        should never be mutated after emission.

Spec: REQ-LEARN-052, SCENARIO-LEARN-061
"""

from __future__ import annotations

import math
from typing import NamedTuple


class VerificationCertificate(NamedTuple):
    """Immutable per-step verification provenance record.

    Implements the Verifiable PRM certificate format from arXiv 2601.17223.

    Fields
    ------
    step_id : str
        Unique identifier for the evaluated step, e.g. "gsm8k_step_003".
    jepa_energy_delta : float
        Cosine distance produced by JEPAv23Predictor.predict_energy(prefix, step).
        Range [0, 2]; higher = step less aligned with prefix = more likely incorrect.
    constraint_type : str
        Semantic category of the constraint evaluated: "arithmetic", "code_logic",
        or "planning".  Determines which Z3 constraint family was applied.
    z3_verdict : str
        Formal solver verdict: "sat" (constraint satisfied), "unsat" (violated),
        or "unknown" (solver timed out or problem is undecidable).
    confidence_score : float
        Scalar in [0, 1].  Derived from jepa_energy_delta via:
            confidence_score = sigmoid(2.0 - jepa_energy_delta * 2.0)
        so that low energy (step looks correct) → high confidence, and
        high energy (step looks wrong) → low confidence.
    """

    step_id: str
    jepa_energy_delta: float
    constraint_type: str
    z3_verdict: str
    confidence_score: float


def make_certificate(
    step_id: str,
    jepa_energy_delta: float,
    constraint_type: str,
) -> VerificationCertificate:
    """Construct a VerificationCertificate from energy delta, deriving other fields.

    WHY this factory instead of constructing VerificationCertificate directly:
        The z3_verdict and confidence_score follow deterministic rules from
        jepa_energy_delta; exposing those rules as a factory prevents callers
        from accidentally supplying inconsistent values.

    Args:
        step_id:            Unique string identifier for the step.
        jepa_energy_delta:  Cosine distance from JEPAv23Predictor.predict_energy().
        constraint_type:    One of "arithmetic", "code_logic", "planning".

    Returns:
        VerificationCertificate with z3_verdict and confidence_score derived
        from jepa_energy_delta.
    """
    # Derive z3_verdict from energy threshold:
    #   energy < 0.5  → step is close to prefix → "sat" (constraint satisfied)
    #   energy >= 0.5 → step is distant from prefix → "unsat" (constraint violated)
    if jepa_energy_delta < 0.5:
        z3_verdict = "sat"
    elif jepa_energy_delta < 1.5:
        z3_verdict = "unsat"
    else:
        z3_verdict = "unknown"

    # confidence_score: sigmoid of (2.0 - energy * 2.0) maps [0,2] energy
    # range to a [0,1] confidence value where low energy → high confidence.
    x = 2.0 - jepa_energy_delta * 2.0
    confidence_score = 1.0 / (1.0 + math.exp(-x))

    return VerificationCertificate(
        step_id=step_id,
        jepa_energy_delta=jepa_energy_delta,
        constraint_type=constraint_type,
        z3_verdict=z3_verdict,
        confidence_score=confidence_score,
    )
