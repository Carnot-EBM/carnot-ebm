"""Shared Carnot exception hierarchy."""

from __future__ import annotations

from typing import Any


class CarnotError(Exception):
    """Base exception for all Carnot pipeline errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details: dict[str, Any] = details or {}


class ExtractionError(CarnotError):
    """Raised when constraint extraction fails on the input text."""


class VerificationError(CarnotError):
    """Raised when constraint evaluation or energy computation fails."""


class RepairError(CarnotError):
    """Raised when the LLM repair loop encounters an unrecoverable error."""


class ModelLoadError(CarnotError):
    """Raised when loading a HuggingFace model fails."""


class PipelineTimeoutError(CarnotError):
    """Raised when a pipeline operation exceeds its wall-clock budget."""


__all__ = [
    "CarnotError",
    "ExtractionError",
    "ModelLoadError",
    "PipelineTimeoutError",
    "RepairError",
    "VerificationError",
]
