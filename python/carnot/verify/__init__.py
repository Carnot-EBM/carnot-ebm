"""Verifiable reasoning: constraints as energy terms + landscape certification.

Spec: REQ-VERIFY-001 through REQ-VERIFY-007, REQ-INFER-001, REQ-INFER-002
"""

from carnot.verify.constraint import (
    ComposedEnergy,
    ConstraintReport,
    ConstraintTerm,
    Verdict,
    VerificationResult,
    repair,
)
from carnot.verify.graph_coloring import (
    ColorDifferenceConstraint,
    ColorRangeConstraint,
    build_coloring_energy,
)
from carnot.verify.landscape import LandscapeCertificate, certify_landscape
from carnot.verify.sat import (
    SATBinaryConstraint,
    SATClause,
    SATClauseConstraint,
    build_sat_energy,
    parse_dimacs,
)

__all__ = [
    "ColorDifferenceConstraint",
    "ColorRangeConstraint",
    "ComposedEnergy",
    "ConstraintReport",
    "ConstraintTerm",
    "LandscapeCertificate",
    "SATBinaryConstraint",
    "SATClause",
    "SATClauseConstraint",
    "Verdict",
    "VerificationResult",
    "build_coloring_energy",
    "build_sat_energy",
    "certify_landscape",
    "parse_dimacs",
    "repair",
]
