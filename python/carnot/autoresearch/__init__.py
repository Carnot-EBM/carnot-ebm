"""Autoresearch: autonomous self-improvement pipeline.

Spec: REQ-AUTO-001 through REQ-AUTO-010
"""

from carnot.autoresearch.sandbox import SandboxConfig, SandboxResult, run_in_sandbox
from carnot.autoresearch.evaluator import EvalResult, evaluate_hypothesis
from carnot.autoresearch.baselines import BaselineRecord

__all__ = [
    "SandboxConfig",
    "SandboxResult",
    "run_in_sandbox",
    "EvalResult",
    "evaluate_hypothesis",
    "BaselineRecord",
]
