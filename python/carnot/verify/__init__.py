"""Verifiable reasoning: constraints as energy terms + landscape certification.

Spec: REQ-VERIFY-001 through REQ-VERIFY-007, REQ-INFER-001, REQ-INFER-002, REQ-CODE-001
"""

try:
    from carnot.verify.constraint import (
        ComposedEnergy,
        ConstraintReport,
        ConstraintTerm,
        Verdict,
        VerificationResult,
        repair,
    )
    from carnot.verify.convergence import (
        ConvergenceCertificate,
        certify_repair_convergence,
        compute_absorbing_radius,
        estimate_jacobian_bound,
    )
    from carnot.verify.graph_coloring import (
        ColorDifferenceConstraint,
        ColorRangeConstraint,
        build_coloring_energy,
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
    from carnot.verify.tier0e_eorm import EORMVerifier
    from carnot.verify.tier0f_semantic_calibration import SemanticCalibratedVerifier
    from carnot.verify.tier0g_semantic_energy import SemanticEnergyVerifier
    from carnot.verify.linear_probe_calibrator import LinearProbeCalibrator
    from carnot.verify.deentangled_ensemble import DeentangledEnsemble
    from carnot.verify.tier0r_curry_howard import Tier0rVerifier
    from carnot.verify.tier0s_halluguard import Tier0sVerifier
    from carnot.verify.tier0u_logical_consistency import Tier0uVerifier
    from carnot.verify.tier0v_set_consistency import SetConsistencyVerifier
except ModuleNotFoundError as exc:  # pragma: no cover - raw system Python without JAX.
    if exc.name != "jax":
        raise
    __all__ = []
else:
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
    ]
from .tier0w_paraphrase_consistency import ParaphrasticConsistencyVerifier
from .tier0z_temporal_causal import TemporalCausalConsistencyVerifier

from .tier0y_conformal_calibration import ConformalCalibrationVerifier

from .tier0y_conformal_calibration import ConformalCalibrationVerifier

from .partial_state_diffusion_scorer import PartialStateDiffusionScorer

try:
    __all__.append("PartialStateDiffusionScorer")
except NameError:  # pragma: no cover - only possible if early optional imports fail.
    __all__ = ["PartialStateDiffusionScorer"]

from .tier0y_conformal_calibration import ConformalCalibrationVerifier

from .tier0y_conformal_calibration import ConformalCalibrationVerifier

from .tier0y_conformal_calibration import ConformalCalibrationVerifier

from .tier0y_conformal_calibration import ConformalCalibrationVerifier

from .tier0y_conformal_calibration import ConformalCalibrationVerifier

from .tier0y_conformal_calibration import ConformalCalibrationVerifier
