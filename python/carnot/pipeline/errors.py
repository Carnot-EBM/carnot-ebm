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

from typing import Any


class CarnotError(Exception):
    """Base exception for all Carnot pipeline errors.

    **Detailed explanation for engineers:**
        All pipeline-specific exceptions inherit from this class.
        Callers who want to catch "anything the pipeline throws"
        can catch CarnotError. The ``details`` dict carries
        structured context (input snippets, domain, stage) for
        logging and diagnostics without bloating the message string.

    Attributes:
        details: Optional dict with structured error context.

    Spec: REQ-VERIFY-003
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details: dict[str, Any] = details or {}


class ExtractionError(CarnotError):
    """Raised when constraint extraction fails on the input text.

    **Detailed explanation for engineers:**
        Covers cases like: input that causes a regex catastrophic
        backtrack, an AST parse that raises an unexpected error
        (not SyntaxError, which CodeExtractor already handles),
        or an extractor that raises internally. The pipeline catches
        these and degrades to "no constraints extracted" rather than
        crashing.

    Spec: REQ-VERIFY-001
    """


class VerificationError(CarnotError):
    """Raised when constraint evaluation / energy computation fails.

    **Detailed explanation for engineers:**
        Covers JAX computation errors during ComposedEnergy.verify(),
        such as shape mismatches or NaN energies. The pipeline catches
        these and returns a VerificationResult with verified=False and
        the error recorded in the certificate.

    Spec: REQ-VERIFY-003
    """


class RepairError(CarnotError):
    """Raised when the LLM repair loop encounters an unrecoverable error.

    **Detailed explanation for engineers:**
        Covers failures during _generate() that are not simple
        RuntimeError (no model). For example, the model produces
        garbage tokens, the tokenizer fails on the repair prompt,
        or the model raises an OOM during generation. The pipeline
        catches these and returns the best response so far rather
        than crashing.

    Spec: REQ-VERIFY-003
    """


class ModelLoadError(CarnotError):
    """Raised when loading a HuggingFace model fails.

    **Detailed explanation for engineers:**
        Wraps ImportError (torch/transformers not installed),
        OSError (model not found on hub or disk), and OOM errors
        during model loading. Surfaced at construction time so
        callers know immediately that the pipeline cannot do repair.

    Spec: REQ-VERIFY-001
    """


class PipelineTimeoutError(CarnotError):
    """Raised when a pipeline operation exceeds its wall-clock budget.

    **Detailed explanation for engineers:**
        Named PipelineTimeoutError (not TimeoutError) to avoid
        shadowing the built-in TimeoutError. Raised by the pipeline
        when a verify() or verify_and_repair() call exceeds the
        configured timeout_seconds. Uses signal.alarm on Unix or
        threading.Timer as fallback.

    Spec: REQ-VERIFY-003
    """
