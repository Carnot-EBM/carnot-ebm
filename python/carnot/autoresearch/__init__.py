"""Autoresearch: autonomous self-improvement pipeline.

**Researcher summary:**
    Complete self-learning loop: sandboxed evaluation, three-gate protocol,
    experiment logging, cross-language validation, automatic rollback,
    circuit breaker, and orchestration. Two sandbox backends (process/Docker).

**Detailed explanation for engineers:**
    This package implements Carnot's full self-learning pipeline:

    1. Generate a hypothesis (new sampler config, training tweak, etc.)
    2. Run it safely in a sandbox (can't crash, can't escape, times out)
    3. Score it against baselines (did energy improve? within time budget?)
    4. Log everything (full audit trail, rejected registry)
    5. If it passed, generate test vectors for cross-language validation
    6. Validate the Rust transpilation matches JAX outputs
    7. Deploy and monitor — rollback automatically if energy regresses

Spec: REQ-AUTO-001 through REQ-AUTO-010
"""

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
from carnot.autoresearch.evaluator import EvalResult, evaluate_hypothesis
from carnot.autoresearch.experiment_log import ExperimentEntry, ExperimentLog
from carnot.autoresearch.orchestrator import (
    AutoresearchConfig,
    LoopResult,
    run_loop,
    run_loop_with_generator,
)
from carnot.autoresearch.rollback import RollbackConfig, RollbackResult, monitor_and_rollback
from carnot.autoresearch.sandbox import SandboxConfig, SandboxResult, run_in_sandbox
from carnot.autoresearch.sandbox_docker import DockerSandboxConfig, run_in_docker
from carnot.autoresearch.transpile import (
    ConformanceResult,
    TestVectorSet,
    generate_test_vectors,
    validate_conformance,
    validate_performance,
)

__all__ = [
    "AutoresearchConfig",
    "BaselineRecord",
    "BenchmarkMetrics",
    "ConformanceResult",
    "DockerSandboxConfig",
    "EvalResult",
    "ExperimentEntry",
    "ExperimentLog",
    "LoopResult",
    "RollbackConfig",
    "RollbackResult",
    "SandboxConfig",
    "SandboxResult",
    "TestVectorSet",
    "evaluate_hypothesis",
    "generate_test_vectors",
    "monitor_and_rollback",
    "run_in_docker",
    "run_in_sandbox",
    "run_loop",
    "run_loop_with_generator",
    "validate_conformance",
    "validate_performance",
]
