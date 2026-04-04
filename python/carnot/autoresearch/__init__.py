"""Autoresearch: autonomous self-improvement pipeline.

**Researcher summary:**
    Complete self-learning loop: sandboxed evaluation, three-gate protocol,
    experiment logging, cross-language validation, automatic rollback,
    circuit breaker, orchestration, trajectory analysis, skill directory,
    and hierarchical lesson consolidation (Trace2Skill). Two sandbox
    backends (process/Docker).

**Detailed explanation for engineers:**
    This package implements Carnot's full self-learning pipeline:

    1. Generate a hypothesis (new sampler config, training tweak, etc.)
    2. Run it safely in a sandbox (can't crash, can't escape, times out)
    3. Score it against baselines (did energy improve? within time budget?)
    4. Log everything (full audit trail, rejected registry)
    5. Analyze trajectories (error/success analysts extract structured lessons)
    6. Consolidate lessons (hierarchical merge, dedup, conflict resolution)
    7. Accumulate knowledge in a skill directory (persistent optimization playbook)
    8. If it passed, generate test vectors for cross-language validation
    9. Validate the Rust transpilation matches JAX outputs
    10. Deploy and monitor — rollback automatically if energy regresses

Spec: REQ-AUTO-001 through REQ-AUTO-014
"""

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
from carnot.autoresearch.consolidator import ConsolidatorConfig, consolidate_lessons
from carnot.autoresearch.evaluator import EvalResult, evaluate_hypothesis
from carnot.autoresearch.experiment_log import ExperimentEntry, ExperimentLog
from carnot.autoresearch.orchestrator import (
    AutoresearchConfig,
    LoopResult,
    run_loop,
    run_loop_with_generator,
    run_loop_with_skills,
)
from carnot.autoresearch.rollback import RollbackConfig, RollbackResult, monitor_and_rollback
from carnot.autoresearch.sandbox import SandboxConfig, SandboxResult, run_in_sandbox
from carnot.autoresearch.sandbox_docker import DockerSandboxConfig, run_in_docker
from carnot.autoresearch.skill_directory import SkillDirectory, SkillDirectoryConfig, SkillPatch
from carnot.autoresearch.trajectory_analyst import (
    AnalystConfig,
    Lesson,
    analyze_batch,
    analyze_error,
    analyze_success,
)
from carnot.autoresearch.transpile import (
    ConformanceResult,
    TestVectorSet,
    generate_test_vectors,
    validate_conformance,
    validate_performance,
)

__all__ = [
    "AnalystConfig",
    "AutoresearchConfig",
    "BaselineRecord",
    "BenchmarkMetrics",
    "ConformanceResult",
    "ConsolidatorConfig",
    "DockerSandboxConfig",
    "EvalResult",
    "ExperimentEntry",
    "ExperimentLog",
    "Lesson",
    "LoopResult",
    "RollbackConfig",
    "RollbackResult",
    "SandboxConfig",
    "SandboxResult",
    "SkillDirectory",
    "SkillDirectoryConfig",
    "SkillPatch",
    "TestVectorSet",
    "analyze_batch",
    "analyze_error",
    "analyze_success",
    "consolidate_lessons",
    "evaluate_hypothesis",
    "generate_test_vectors",
    "monitor_and_rollback",
    "run_in_docker",
    "run_in_sandbox",
    "run_loop",
    "run_loop_with_generator",
    "run_loop_with_skills",
    "validate_conformance",
    "validate_performance",
]
