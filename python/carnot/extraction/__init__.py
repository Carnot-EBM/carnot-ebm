"""carnot.extraction — pluggable constraint extractors for LLM reasoning traces.

This package houses extractors that go beyond regex-based arithmetic scanning
to handle instruction-tuned (IT) model outputs written in natural language.

**Why a separate package from carnot.pipeline?**
    pipeline/ contains the full verify-repair orchestration stack.  extraction/
    isolates the constraint-extraction primitives so they can be imported without
    pulling in heavy pipeline dependencies (transformers, JAX, etc.).  This keeps
    test-time imports fast and makes use_mock=True isolation straightforward.

Exports
-------
FOLPremise       — one formalized First-Order Logic premise from a CoT step
StepVerdict      — Z3 satisfiability verdict for one CoT step
VeriCoTStepValidator — full VeriCoT pipeline: extract FOL → check Z3

Spec: REQ-EXTRACT-024, REQ-EXTRACT-025, REQ-EXTRACT-026
"""

from carnot.extraction.vericot_validator import (
    FOLPremise,
    StepVerdict,
    VeriCoTStepValidator,
)
from carnot.extraction.vprm_verifier import (
    ArithmeticRule,
    RuleVerdict,
    VPRMArithmeticVerifier,
)
from carnot.extraction.extraction_diagnostic import (
    ExtractionDiagnosticResult,
    run_extractor_diagnostic,
)
from carnot.extraction.confidence_filter import (
    ViolationConfidence,
    ConfidenceWeightedExtractor,
    score_violation,
)

__all__ = [
    "FOLPremise",
    "StepVerdict",
    "VeriCoTStepValidator",
    "ArithmeticRule",
    "RuleVerdict",
    "VPRMArithmeticVerifier",
    "ExtractionDiagnosticResult",
    "run_extractor_diagnostic",
    "ViolationConfidence",
    "ConfidenceWeightedExtractor",
    "score_violation",
]
