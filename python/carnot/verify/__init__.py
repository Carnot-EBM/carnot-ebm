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
from carnot.verify.convergence import (
    ConvergenceCertificate,
    certify_repair_convergence,
    compute_absorbing_radius,
    estimate_jacobian_bound,
)
from carnot.verify.landscape import LandscapeCertificate, certify_landscape
from carnot.verify.property_test import (
    PropertyTestConstraint,
    PropertyTestResult,
    format_violations_for_llm,
    property_test,
)
from carnot.verify.python_types import (
    NoExceptionConstraint,
    ReturnTypeConstraint,
    TestPassConstraint,
    ast_code_to_embedding,
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
    "PropertyTestConstraint",
    "PropertyTestResult",
    "NoExceptionConstraint",
    "ReturnTypeConstraint",
    "TestPassConstraint",
    "ConvergenceCertificate",
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
    "ast_code_to_embedding",
    "code_to_embedding",
    "format_violations_for_llm",
    "build_sat_energy",
    "certify_landscape",
    "certify_repair_convergence",
    "compute_absorbing_radius",
    "estimate_jacobian_bound",
    "parse_dimacs",
    "property_test",
    "repair",
    "safe_exec_function",
]
