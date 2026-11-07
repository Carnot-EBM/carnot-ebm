"""Verifiable reasoning with optional evaluators loaded only when requested.

REQ-REPORT-8858: explicit record verification must import without research or
training initialization. Existing named exports keep their original modules.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ParaphrasticConsistencyVerifier": "carnot.verify.tier0w_paraphrase_consistency",
    "TemporalCausalConsistencyVerifier": "carnot.verify.tier0z_temporal_causal",
    "ConformalCalibrationVerifier": "carnot.verify.tier0y_conformal_calibration",
    "PartialStateDiffusionScorer": "carnot.verify.partial_state_diffusion_scorer",
    "ComposedEnergy": "carnot.verify.constraint",
    "ConstraintReport": "carnot.verify.constraint",
    "ConstraintTerm": "carnot.verify.constraint",
    "Verdict": "carnot.verify.constraint",
    "VerificationResult": "carnot.verify.constraint",
    "repair": "carnot.verify.constraint",
    "ConvergenceCertificate": "carnot.verify.convergence",
    "certify_repair_convergence": "carnot.verify.convergence",
    "compute_absorbing_radius": "carnot.verify.convergence",
    "estimate_jacobian_bound": "carnot.verify.convergence",
    "ColorDifferenceConstraint": "carnot.verify.graph_coloring",
    "ColorRangeConstraint": "carnot.verify.graph_coloring",
    "build_coloring_energy": "carnot.verify.graph_coloring",
    "LandscapeCertificate": "carnot.verify.landscape",
    "certify_landscape": "carnot.verify.landscape",
    "PropertyTestConstraint": "carnot.verify.property_test",
    "PropertyTestResult": "carnot.verify.property_test",
    "format_violations_for_llm": "carnot.verify.property_test",
    "property_test": "carnot.verify.property_test",
    "NoExceptionConstraint": "carnot.verify.python_types",
    "ReturnTypeConstraint": "carnot.verify.python_types",
    "TestPassConstraint": "carnot.verify.python_types",
    "ast_code_to_embedding": "carnot.verify.python_types",
    "build_code_energy": "carnot.verify.python_types",
    "code_to_embedding": "carnot.verify.python_types",
    "safe_exec_function": "carnot.verify.python_types",
    "SATBinaryConstraint": "carnot.verify.sat",
    "SATClause": "carnot.verify.sat",
    "SATClauseConstraint": "carnot.verify.sat",
    "build_sat_energy": "carnot.verify.sat",
    "parse_dimacs": "carnot.verify.sat",
    "EORMVerifier": "carnot.verify.tier0e_eorm",
    "SemanticCalibratedVerifier": "carnot.verify.tier0f_semantic_calibration",
    "SemanticEnergyVerifier": "carnot.verify.tier0g_semantic_energy",
    "LinearProbeCalibrator": "carnot.verify.linear_probe_calibrator",
    "DeentangledEnsemble": "carnot.verify.deentangled_ensemble",
    "Tier0rVerifier": "carnot.verify.tier0r_curry_howard",
    "Tier0sVerifier": "carnot.verify.tier0s_halluguard",
    "Tier0uVerifier": "carnot.verify.tier0u_logical_consistency",
    "SetConsistencyVerifier": "carnot.verify.tier0v_set_consistency",
    "verify_indexed_witness": "carnot.verify.indexed_witness",
}

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
    "EORMVerifier",
    "SemanticCalibratedVerifier",
    "SemanticEnergyVerifier",
    "LinearProbeCalibrator",
    "DeentangledEnsemble",
    "Tier0rVerifier",
    "Tier0sVerifier",
    "Tier0uVerifier",
    "SetConsistencyVerifier",
    "PartialStateDiffusionScorer",
    "verify_indexed_witness",
]


def __getattr__(name: str) -> Any:
    """Resolve an existing public name on demand to keep pure requests lightweight."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
