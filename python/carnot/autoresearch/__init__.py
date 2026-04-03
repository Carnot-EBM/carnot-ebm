"""Autoresearch: autonomous self-improvement pipeline.

**Researcher summary:**
    Sandboxed hypothesis evaluation with three-gate protocol, structured
    experiment logging, circuit breaker, and orchestration loop. Supports
    two sandbox backends: process-level (dev) and Docker+gVisor (production).

**Detailed explanation for engineers:**
    This package implements Carnot's self-learning loop. The core idea:

    1. Generate a hypothesis (new sampler config, training tweak, etc.)
    2. Run it safely in a sandbox (can't crash, can't escape, times out)
    3. Score it against baselines (did energy improve? within time budget?)
    4. Log everything (full audit trail, rejected registry)
    5. If it passed, update baselines so the bar keeps rising

    **Two sandbox backends:**

    - ``run_in_sandbox()`` — Process-level isolation (SIGALRM timeout, import
      blocking). Fast, no dependencies, good for development and testing.
      NOT secure against determined adversaries.

    - ``run_in_docker()`` — Docker+gVisor container isolation. No network,
      read-only filesystem, hard memory/CPU limits, syscall interception.
      Use for production autoresearch runs.

    The ``run_loop()`` function in the orchestrator ties it all together.

Spec: REQ-AUTO-001 through REQ-AUTO-010
"""

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
from carnot.autoresearch.evaluator import EvalResult, evaluate_hypothesis
from carnot.autoresearch.experiment_log import ExperimentEntry, ExperimentLog
from carnot.autoresearch.orchestrator import AutoresearchConfig, LoopResult, run_loop
from carnot.autoresearch.sandbox import SandboxConfig, SandboxResult, run_in_sandbox
from carnot.autoresearch.sandbox_docker import DockerSandboxConfig, run_in_docker

__all__ = [
    "AutoresearchConfig",
    "BaselineRecord",
    "BenchmarkMetrics",
    "DockerSandboxConfig",
    "EvalResult",
    "ExperimentEntry",
    "ExperimentLog",
    "LoopResult",
    "SandboxConfig",
    "SandboxResult",
    "evaluate_hypothesis",
    "run_in_docker",
    "run_in_sandbox",
    "run_loop",
]
