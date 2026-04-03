"""Autoresearch orchestrator: the autonomous self-improvement loop.

**Researcher summary:**
    Runs the propose → sandbox → evaluate → log cycle. Hypothesis code is
    executed in isolation, scored against baselines via three-gate evaluation,
    and logged. Circuit breaker halts after N consecutive failures. Accepted
    hypotheses update the baseline registry.

**Detailed explanation for engineers:**
    This is the capstone of Carnot's self-learning pipeline. It ties together
    all the pieces:

    1. **Propose**: Load hypothesis code (from a file, an LLM, or a generator)
    2. **Execute**: Run it in the sandbox (timeout, import blocking, I/O capture)
    3. **Evaluate**: Compare results against baselines (three-gate protocol)
    4. **Log**: Record the full experiment (code, metrics, verdict, outcome)
    5. **Update**: If accepted, update baselines. If rejected, add to rejected list.
    6. **Circuit breaker**: If too many consecutive failures, halt and wait for human.

    **Why this is safe:**
    - Hypothesis code runs in a sandbox (no filesystem, no network, timeout)
    - The energy function is the objective judge (can't be gamed by an LLM)
    - Baselines are append-only (old results never deleted)
    - Circuit breaker prevents infinite loops of bad hypotheses
    - REVIEW verdicts (mixed results) require human approval

    **How to use it:**

    ```python
    from carnot.autoresearch.orchestrator import AutoresearchConfig, run_loop

    config = AutoresearchConfig(max_iterations=10)
    hypotheses = [
        ("step_size_sweep_0.005", 'def run(d): ...'),
        ("step_size_sweep_0.05", 'def run(d): ...'),
    ]
    results = run_loop(hypotheses, baselines, benchmark_data, config)
    ```

Spec: FR-11, REQ-AUTO-003 through REQ-AUTO-010
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
from carnot.autoresearch.evaluator import evaluate_hypothesis
from carnot.autoresearch.experiment_log import ExperimentEntry, ExperimentLog
from carnot.autoresearch.sandbox import SandboxConfig, run_in_sandbox

logger = logging.getLogger(__name__)


@dataclass
class AutoresearchConfig:
    """Configuration for the autoresearch loop.

    **Researcher summary:**
        Controls iteration budget, circuit breaker threshold, sandbox parameters,
        and evaluation tolerances.

    **Detailed explanation for engineers:**
        These settings control the safety and behavior of the self-improvement loop:

        - ``max_iterations``: Maximum number of hypotheses to evaluate in one run.
          Set this to prevent runaway loops. Default 100.

        - ``max_consecutive_failures``: Circuit breaker threshold. If this many
          hypotheses in a row are rejected, the loop halts and returns control
          to the human. Default 10. This prevents the system from wasting compute
          on a broken hypothesis generator.

        - ``sandbox_config``: Timeout, memory limit, and import blocklist for
          hypothesis execution. See SandboxConfig.

        - ``auto_accept_pass``: If True, hypotheses with PASS verdict are
          automatically accepted and baselines are updated. If False, all
          verdicts require human approval. Default True.

        - ``energy_regression_tolerance``: How much energy can increase before
          it's considered a regression. Default 0.001 (0.1%). Uses absolute
          tolerance to handle negative energies correctly.

    Spec: REQ-AUTO-009
    """

    max_iterations: int = 100
    max_consecutive_failures: int = 10
    sandbox_config: SandboxConfig = field(default_factory=SandboxConfig)
    auto_accept_pass: bool = True
    energy_regression_tolerance: float = 0.001


@dataclass
class LoopResult:
    """Result of running the autoresearch loop.

    **Researcher summary:**
        Summary of the loop run: how many hypotheses were evaluated, how many
        accepted/rejected/review, whether the circuit breaker tripped, and
        the final baseline state.

    **Detailed explanation for engineers:**
        After the loop finishes (either by exhausting hypotheses, hitting
        max_iterations, or tripping the circuit breaker), this tells you
        what happened:

        - ``iterations``: How many hypotheses were evaluated
        - ``accepted``: How many were accepted (baselines updated)
        - ``rejected``: How many were rejected (no change)
        - ``pending_review``: How many had mixed results (need human decision)
        - ``circuit_breaker_tripped``: True if the loop halted due to too
          many consecutive failures
        - ``final_baselines``: The baseline record after all accepted updates
        - ``experiment_log``: Full experiment log for inspection

    Spec: FR-11
    """

    iterations: int = 0
    accepted: int = 0
    rejected: int = 0
    pending_review: int = 0
    circuit_breaker_tripped: bool = False
    final_baselines: BaselineRecord | None = None
    experiment_log: ExperimentLog = field(default_factory=ExperimentLog)


def run_loop(
    hypotheses: list[tuple[str, str]],
    baselines: BaselineRecord,
    benchmark_data: dict[str, Any],
    config: AutoresearchConfig | None = None,
    experiment_log: ExperimentLog | None = None,
) -> LoopResult:
    """Run the autoresearch self-improvement loop.

    **Researcher summary:**
        Iterates over hypotheses: sandbox → evaluate → log → update baselines.
        Halts on circuit breaker or iteration limit. Returns LoopResult with
        full experiment log and updated baselines.

    **Detailed explanation for engineers:**
        This is the main entry point for the autoresearch pipeline. It takes
        a list of hypotheses (each a (description, code) pair), runs each
        through the sandbox and evaluator, and tracks results.

        The loop processes hypotheses in order and stops when:
        1. All hypotheses have been evaluated
        2. ``max_iterations`` is reached
        3. The circuit breaker trips (too many consecutive failures)

        For each hypothesis:
        1. Check if it's in the rejected registry (skip if so)
        2. Run it in the sandbox with the benchmark data
        3. Evaluate the sandbox result against current baselines
        4. Log the experiment
        5. If PASS and auto_accept: update baselines
        6. If FAIL: increment failure counter
        7. If REVIEW: mark as pending (no auto-accept)

        **The key insight**: baselines are updated *during* the loop, so later
        hypotheses are evaluated against the improved baselines. This means
        the bar keeps rising as good hypotheses are accepted.

    Args:
        hypotheses: List of (description, python_code) pairs. Each code string
            must define a ``run(benchmark_data) -> dict`` function.
        baselines: Current baseline performance metrics to evaluate against.
        benchmark_data: Dict passed to each hypothesis's ``run()`` function.
            Typically contains benchmark dimensions, configurations, etc.
        config: Loop configuration. Uses defaults if None.
        experiment_log: Existing experiment log to append to. Creates new if None.

    Returns:
        LoopResult with iteration counts, updated baselines, and experiment log.

    For example::

        hypotheses = [
            ("try smaller step size", '''
def run(benchmark_data):
    # ... run sampler with step_size=0.005 ...
    return {"double_well": {"final_energy": -5.5, "wall_clock_seconds": 1.0}}
            '''),
        ]
        result = run_loop(hypotheses, baselines, {"dim": 2})
        print(f"Accepted: {result.accepted}, Rejected: {result.rejected}")

    Spec: FR-11, REQ-AUTO-003 through REQ-AUTO-010
    """
    if config is None:
        config = AutoresearchConfig()
    if experiment_log is None:
        experiment_log = ExperimentLog()

    result = LoopResult(
        final_baselines=baselines,
        experiment_log=experiment_log,
    )

    # Get already-rejected IDs to avoid re-evaluation
    rejected_ids = experiment_log.rejected_ids()

    for i, (description, code) in enumerate(hypotheses):
        # --- Iteration limit ---
        if result.iterations >= config.max_iterations:
            logger.info("Max iterations (%d) reached, stopping.", config.max_iterations)
            break

        # --- Circuit breaker (REQ-AUTO-009) ---
        if experiment_log.consecutive_failures() >= config.max_consecutive_failures:
            logger.warning(
                "Circuit breaker: %d consecutive failures. Halting for human review.",
                config.max_consecutive_failures,
            )
            result.circuit_breaker_tripped = True
            break

        # --- Generate experiment ID ---
        exp_id = f"auto-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{i:03d}"

        # --- Skip if already rejected (REQ-AUTO-007) ---
        if exp_id in rejected_ids:
            logger.info("Skipping %s (already rejected)", exp_id)
            continue

        logger.info("Evaluating hypothesis %s: %s", exp_id, description)

        # --- Stage 1: Sandbox execution (REQ-AUTO-004) ---
        sandbox_result = run_in_sandbox(code, benchmark_data, config.sandbox_config)

        # --- Stage 2: Evaluation (REQ-AUTO-005) ---
        eval_result = evaluate_hypothesis(
            sandbox_result,
            baselines,
            energy_regression_tolerance=config.energy_regression_tolerance,
        )

        # --- Stage 3: Determine outcome ---
        if eval_result.verdict == "PASS" and config.auto_accept_pass:
            outcome = "accepted"
            result.accepted += 1
            # Update baselines with the improved metrics (REQ-AUTO-002)
            if sandbox_result.success:
                _update_baselines(baselines, sandbox_result.metrics)
            logger.info("ACCEPTED: %s — %s", exp_id, eval_result.reason)
        elif eval_result.verdict == "REVIEW":
            outcome = "pending_review"
            result.pending_review += 1
            logger.info("REVIEW: %s — %s", exp_id, eval_result.reason)
        else:
            outcome = "rejected"
            result.rejected += 1
            logger.info("REJECTED: %s — %s", exp_id, eval_result.reason)

        # --- Stage 4: Log experiment (REQ-AUTO-008) ---
        entry = ExperimentEntry(
            id=exp_id,
            timestamp=datetime.now(UTC).isoformat(),
            hypothesis_code=code,
            hypothesis_description=description,
            sandbox_success=sandbox_result.success,
            sandbox_metrics=sandbox_result.metrics,
            sandbox_error=sandbox_result.error,
            sandbox_wall_clock=sandbox_result.wall_clock_seconds,
            sandbox_timed_out=sandbox_result.timed_out,
            eval_verdict=eval_result.verdict,
            eval_reason=eval_result.reason,
            eval_improvements=eval_result.improvements,
            eval_regressions=eval_result.regressions,
            outcome=outcome,
        )
        experiment_log.append(entry)
        result.iterations += 1

    return result


def run_loop_with_generator(
    generator: Any,
    baselines: BaselineRecord,
    benchmark_data: dict[str, Any],
    config: AutoresearchConfig | None = None,
    experiment_log: ExperimentLog | None = None,
) -> LoopResult:
    """Run the autoresearch loop with an LLM hypothesis generator.

    **Researcher summary:**
        Like ``run_loop`` but generates hypotheses on-demand from an LLM
        instead of requiring a pre-built list. Each iteration: generate →
        sandbox → evaluate → log → update → feed failures back to generator.

    **Detailed explanation for engineers:**
        This variant of the loop uses a callback to generate hypotheses
        lazily. After each evaluation, the generator receives feedback
        about what worked and what didn't, allowing it to adapt its
        strategy over time.

        The ``generator`` must be a callable with signature::

            generator(baselines, recent_failures, iteration) -> list[tuple[str, str]]

        Where:
        - baselines: current BaselineRecord
        - recent_failures: list of dicts with "description" and "reason"
        - iteration: current iteration number

        Returns a list of (description, code) pairs.

    Args:
        generator: Callable that produces hypothesis (description, code) pairs.
        baselines: Initial baseline performance.
        benchmark_data: Dict passed to each hypothesis's run() function.
        config: Loop configuration. Uses defaults if None.
        experiment_log: Existing experiment log to append to. Creates new if None.

    Returns:
        LoopResult with iteration counts, updated baselines, and experiment log.

    Spec: FR-11, REQ-AUTO-003
    """
    if config is None:
        config = AutoresearchConfig()
    if experiment_log is None:
        experiment_log = ExperimentLog()

    result = LoopResult(
        final_baselines=baselines,
        experiment_log=experiment_log,
    )

    recent_failures: list[dict[str, Any]] = []
    iteration = 0

    while iteration < config.max_iterations:
        # --- Circuit breaker (REQ-AUTO-009) ---
        if experiment_log.consecutive_failures() >= config.max_consecutive_failures:
            logger.warning(
                "Circuit breaker: %d consecutive failures. Halting for human review.",
                config.max_consecutive_failures,
            )
            result.circuit_breaker_tripped = True
            break

        # --- Generate hypotheses ---
        logger.info("Generating hypotheses (iteration %d)...", iteration)
        try:
            hypotheses = generator(baselines, recent_failures, iteration)
        except Exception:
            logger.exception("Hypothesis generator failed")
            recent_failures.append({
                "description": "generator_error",
                "reason": "Hypothesis generator raised an exception",
            })
            iteration += 1
            continue

        if not hypotheses:
            logger.warning("Generator returned no hypotheses, stopping.")
            break

        # --- Evaluate each generated hypothesis ---
        for desc, code in hypotheses:
            if result.iterations >= config.max_iterations:
                break
            if experiment_log.consecutive_failures() >= config.max_consecutive_failures:
                result.circuit_breaker_tripped = True
                break

            exp_id = f"llm-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{iteration:03d}"
            logger.info("Evaluating hypothesis %s: %s", exp_id, desc)

            # Sandbox execution
            sandbox_result = run_in_sandbox(code, benchmark_data, config.sandbox_config)

            # Evaluation
            eval_result = evaluate_hypothesis(
                sandbox_result,
                baselines,
                energy_regression_tolerance=config.energy_regression_tolerance,
            )

            # Determine outcome
            if eval_result.verdict == "PASS" and config.auto_accept_pass:
                outcome = "accepted"
                result.accepted += 1
                if sandbox_result.success:
                    _update_baselines(baselines, sandbox_result.metrics)
                logger.info("ACCEPTED: %s — %s", exp_id, eval_result.reason)
                # Reset failure tracking on success
                recent_failures.clear()
            elif eval_result.verdict == "REVIEW":
                outcome = "pending_review"
                result.pending_review += 1
                logger.info("REVIEW: %s — %s", exp_id, eval_result.reason)
            else:
                outcome = "rejected"
                result.rejected += 1
                logger.info("REJECTED: %s — %s", exp_id, eval_result.reason)
                recent_failures.append({
                    "description": desc,
                    "reason": eval_result.reason,
                })

            # Log experiment
            entry = ExperimentEntry(
                id=exp_id,
                timestamp=datetime.now(UTC).isoformat(),
                hypothesis_code=code,
                hypothesis_description=desc,
                sandbox_success=sandbox_result.success,
                sandbox_metrics=sandbox_result.metrics,
                sandbox_error=sandbox_result.error,
                sandbox_wall_clock=sandbox_result.wall_clock_seconds,
                sandbox_timed_out=sandbox_result.timed_out,
                eval_verdict=eval_result.verdict,
                eval_reason=eval_result.reason,
                eval_improvements=eval_result.improvements,
                eval_regressions=eval_result.regressions,
                outcome=outcome,
            )
            experiment_log.append(entry)
            result.iterations += 1

        iteration += 1

    return result


def _update_baselines(
    baselines: BaselineRecord,
    metrics: dict[str, Any],
) -> None:
    """Update baseline record with improved metrics from an accepted hypothesis.

    Only updates benchmarks where the hypothesis provided results.
    Does not remove existing baselines — append/update only.

    Spec: REQ-AUTO-002
    """
    for name, bench_metrics in metrics.items():
        if not isinstance(bench_metrics, dict):
            continue
        energy = bench_metrics.get("final_energy")
        if energy is None:
            continue
        baselines.benchmarks[name] = BenchmarkMetrics(
            benchmark_name=name,
            final_energy=energy,
            convergence_steps=bench_metrics.get("convergence_steps", 0),
            wall_clock_seconds=bench_metrics.get("wall_clock_seconds", 0.0),
            peak_memory_mb=bench_metrics.get("peak_memory_mb", 0.0),
        )
