"""Error hierarchy for the Carnot verification pipeline.

**Researcher summary:**
    Structured exception classes for graceful degradation in the
    verify-repair pipeline. Each error type corresponds to a distinct
    failure mode (extraction, verification, repair, model loading,
    timeout) so callers can handle them selectively.

**Detailed explanation for engineers:**
    The pipeline can fail at several stages: constraint extraction
    (bad input, unsupported domain), verification (JAX computation
    errors), repair (LLM generation failures), model loading
    (missing model, OOM), or timeout (pipeline exceeds wall-clock
    budget). This module provides a single base class (CarnotError)
    and five specific subclasses so callers can catch broadly or
    narrowly as needed.

    All errors carry a ``details`` dict for structured metadata
    (e.g., which extractor failed, what input triggered the error)
    to aid debugging without exposing internals in the message string.

Spec: REQ-VERIFY-001, REQ-VERIFY-003
"""

from __future__ import annotations

from carnot.errors import (
    CarnotError,
    ExtractionError,
    ModelLoadError,
    PipelineTimeoutError,
    RepairError,
    VerificationError,
)

__all__ = [
    "CarnotError",
    "ExtractionError",
    "ModelLoadError",
    "PipelineTimeoutError",
    "RepairError",
    "VerificationError",
]
