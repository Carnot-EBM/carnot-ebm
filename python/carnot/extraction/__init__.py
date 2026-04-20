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
ArithmeticEquation   — one parsed arithmetic equation from prose CoT
CoACEViolation       — one detected arithmetic violation (computed != stated)
CoACEResult          — aggregated extraction result from CoACEExtractor
CoACEExtractor       — execution-based extractor (Caco arXiv 2510.04081)

Spec: REQ-EXTRACT-024, REQ-EXTRACT-025, REQ-EXTRACT-026,
      REQ-EXTRACT-033, REQ-EXTRACT-034
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
from carnot.extraction.coace_extractor import (
    ArithmeticEquation,
    CoACEViolation,
    CoACEResult,
    CoACEExtractor,
)
from carnot.extraction.coace_extractor_v2 import (
    CoACEExtractorV2,
)
from carnot.extraction.coace_extractor_v3 import (
    CoACEExtractorV3,
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
    "ArithmeticEquation",
    "CoACEViolation",
    "CoACEResult",
    "CoACEExtractor",
    "CoACEExtractorV2",
    "CoACEExtractorV3",
]
