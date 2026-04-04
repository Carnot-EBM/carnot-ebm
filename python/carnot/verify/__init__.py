"""Verifiable reasoning: constraints as energy terms + landscape certification.

Spec: REQ-VERIFY-001 through REQ-VERIFY-007, REQ-INFER-001, REQ-INFER-002, REQ-CODE-001
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
from carnot.verify.python_types import (
    NoExceptionConstraint,
    ReturnTypeConstraint,
    TestPassConstraint,
    build_code_energy,
    code_to_embedding,
    safe_exec_function,
)
from carnot.verify.sat import (
    SATBinaryConstraint,
    SATClause,
    SATClauseConstraint,
    build_sat_energy,
    parse_dimacs,
)

__all__ = [
    "NoExceptionConstraint",
    "ReturnTypeConstraint",
    "TestPassConstraint",
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
    "build_code_energy",
    "build_coloring_energy",
    "code_to_embedding",
    "build_sat_energy",
    "certify_landscape",
    "parse_dimacs",
    "repair",
    "safe_exec_function",
]
