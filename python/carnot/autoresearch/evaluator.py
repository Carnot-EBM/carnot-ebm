"""Hypothesis evaluation protocol.

Three-gate evaluation: energy improvement, time budget, memory budget.

Spec: REQ-AUTO-005, REQ-AUTO-007
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from carnot.autoresearch.baselines import BaselineRecord
from carnot.autoresearch.sandbox import SandboxResult


@dataclass
class EvalResult:
    """Result of evaluating a hypothesis against baselines.

    Spec: REQ-AUTO-005
    """

    verdict: str  # "PASS", "FAIL", "REVIEW"
    reason: str = ""
    primary_gate: bool = False  # energy improvement
    secondary_gate: bool = False  # time budget
    tertiary_gate: bool = False  # memory budget
    improvements: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)


def evaluate_hypothesis(
    sandbox_result: SandboxResult,
    baselines: BaselineRecord,
    energy_regression_tolerance: float = 0.001,
    time_budget_multiplier: float = 2.0,
    memory_budget_multiplier: float = 2.0,
) -> EvalResult:
    """Evaluate a sandbox result against baselines.

    Three gates:
    1. Primary: energy must be <= baseline (within tolerance)
    2. Secondary: wall-clock time must be <= 2x baseline
    3. Tertiary: memory must be <= 2x baseline

    Spec: REQ-AUTO-005
    """
    if not sandbox_result.success:
        return EvalResult(
            verdict="FAIL",
            reason=f"Sandbox failed: {sandbox_result.error}",
        )

    metrics = sandbox_result.metrics
    improvements: list[str] = []
    regressions: list[str] = []

    # Primary gate: energy check per benchmark
    primary_pass = True
    for name, baseline in baselines.benchmarks.items():
        if name not in metrics:
            continue
        bench_metrics = metrics[name]
        if not isinstance(bench_metrics, dict):
            continue

        bench_energy = bench_metrics.get("final_energy")
        if bench_energy is None:
            continue

        # Use absolute tolerance for correct handling of negative energies
        abs_tol = abs(baseline.final_energy) * energy_regression_tolerance
        if bench_energy > baseline.final_energy + abs_tol:
            primary_pass = False
            regressions.append(name)
        elif bench_energy < baseline.final_energy - abs_tol:
            improvements.append(name)

    # Secondary gate: time budget
    secondary_pass = True
    for name, baseline in baselines.benchmarks.items():
        if name not in metrics:
            continue
        bench_metrics = metrics[name]
        if not isinstance(bench_metrics, dict):
            continue
        bench_time = bench_metrics.get("wall_clock_seconds")
        if bench_time is not None and baseline.wall_clock_seconds > 0:
            if bench_time > baseline.wall_clock_seconds * time_budget_multiplier:
                secondary_pass = False

    # Tertiary gate: memory budget
    tertiary_pass = True
    peak_memory = metrics.get("peak_memory_mb")
    if peak_memory is not None:
        max_baseline_memory = max(
            (b.peak_memory_mb for b in baselines.benchmarks.values() if b.peak_memory_mb > 0),
            default=0,
        )
        if max_baseline_memory > 0 and peak_memory > max_baseline_memory * memory_budget_multiplier:
            tertiary_pass = False

    # Determine verdict
    if regressions and improvements:
        verdict = "REVIEW"
        reason = f"Mixed: improved {improvements}, regressed {regressions}"
    elif not primary_pass:
        verdict = "FAIL"
        reason = f"Energy regression on: {', '.join(regressions)}"
    elif not secondary_pass:
        verdict = "FAIL"
        reason = "Time budget exceeded"
    elif not tertiary_pass:
        verdict = "FAIL"
        reason = "Memory budget exceeded"
    else:
        verdict = "PASS"
        reason = f"Improved: {', '.join(improvements)}" if improvements else "No regression"

    return EvalResult(
        verdict=verdict,
        reason=reason,
        primary_gate=primary_pass,
        secondary_gate=secondary_pass,
        tertiary_gate=tertiary_pass,
        improvements=improvements,
        regressions=regressions,
    )
