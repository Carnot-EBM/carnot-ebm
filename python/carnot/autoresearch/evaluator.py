"""Hypothesis evaluation protocol -- three-gate assessment.

**Researcher summary:**
    Evaluates sandbox results against baseline records using three gates:
    (1) energy improvement (primary), (2) wall-clock time budget (secondary),
    (3) memory budget (tertiary). Produces PASS / FAIL / REVIEW verdicts.

**Detailed explanation for engineers:**
    In the autoresearch self-improvement pipeline, after a hypothesis runs in
    the sandbox (sandbox.py), its results need to be compared against known
    baselines to determine if it represents an improvement. This module
    implements that comparison using a three-gate evaluation protocol.

    **The self-improvement pipeline context:**
    1. Baseline: The current best-known results for each benchmark (stored
       in a BaselineRecord — see baselines.py)
    2. Hypothesis: A proposed improvement (new architecture, training method,
       sampler, etc.) that was executed in the sandbox
    3. Evaluation (this module): Compares hypothesis metrics against baselines

    **The three gates:**

    1. **Primary gate (energy):** The hypothesis must not produce worse energy
       values than the baseline. This is the core quality check — if the model
       got worse at its actual task, it fails. A small tolerance
       (``energy_regression_tolerance``) accounts for noise.

    2. **Secondary gate (time):** The hypothesis must not take more than 2x
       the baseline's wall-clock time. We want improvements that are practical,
       not just theoretically better but 100x slower.

    3. **Tertiary gate (memory):** The hypothesis must not use more than 2x
       the baseline's peak memory. An improvement that requires a 256GB GPU
       when the baseline runs on 8GB is not practical.

    **Verdicts:**
    - ``PASS``: All gates passed. The hypothesis is an improvement (or at
      least not a regression). Safe to adopt.
    - ``FAIL``: At least one gate failed with no offsetting improvements.
      The hypothesis should be discarded.
    - ``REVIEW``: Mixed results — some benchmarks improved, others regressed.
      A human (or more sophisticated agent) should review.

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

    **Researcher summary:**
        Contains verdict (PASS/FAIL/REVIEW), gate results, and lists of
        improved/regressed benchmarks.

    **Detailed explanation for engineers:**
        After running the three-gate evaluation, this dataclass captures
        the complete outcome:

    Attributes:
        verdict: One of "PASS", "FAIL", or "REVIEW".
            - "PASS": All gates passed, no regressions.
            - "FAIL": At least one gate failed.
            - "REVIEW": Mixed results (some improved, some regressed).
        reason: Human-readable explanation of the verdict.
        primary_gate: Did the energy gate pass? (True/False)
        secondary_gate: Did the time budget gate pass? (True/False)
        tertiary_gate: Did the memory budget gate pass? (True/False)
        improvements: List of benchmark names where the hypothesis
            improved on the baseline (lower energy).
        regressions: List of benchmark names where the hypothesis
            regressed (higher energy).

    For example::

        result = evaluate_hypothesis(sandbox_result, baselines)
        if result.verdict == "PASS":
            print(f"Hypothesis improved on: {result.improvements}")
        elif result.verdict == "REVIEW":
            print(f"Mixed: improved {result.improvements}, regressed {result.regressions}")

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
    jit_grace_seconds: float = 1.0,
) -> EvalResult:
    """Evaluate a sandbox result against baselines using three gates.

    **Researcher summary:**
        Three-gate evaluation: (1) energy within tolerance of baseline,
        (2) time <= 2x baseline, (3) memory <= 2x baseline. Returns
        PASS/FAIL/REVIEW verdict.

    **Detailed explanation for engineers:**
        This function implements the core evaluation logic for the autoresearch
        pipeline. It takes the SandboxResult (from running a hypothesis) and
        a BaselineRecord (known-good results), and systematically checks three
        gates:

        **Gate 1 — Energy (Primary):**
        For each benchmark in the baseline, check if the hypothesis produced
        a worse (higher) final energy. Uses absolute tolerance to correctly
        handle negative energies:
            allowed = baseline_energy + |baseline_energy| * tolerance
        If the hypothesis energy exceeds this, it's a regression.

        **Gate 2 — Time (Secondary):**
        For each benchmark, check if the hypothesis took more than
        ``time_budget_multiplier`` times the baseline's wall-clock time.

        **Gate 3 — Memory (Tertiary):**
        Check if the hypothesis's peak memory exceeds
        ``memory_budget_multiplier`` times the maximum baseline memory.

        **Verdict logic:**
        - If both improvements AND regressions exist: "REVIEW" (mixed results)
        - If any gate fails with no improvements: "FAIL"
        - Otherwise: "PASS"

    Args:
        sandbox_result: The result from running the hypothesis in the sandbox.
            Must have ``success=True`` for evaluation to proceed.
        baselines: Known-good baseline results for comparison.
        energy_regression_tolerance: Relative tolerance for energy regression.
            Default 0.001 (0.1%). A hypothesis is allowed to be this much
            worse than the baseline without being flagged as a regression.
        time_budget_multiplier: How much slower than baseline is allowed.
            Default 2.0 (2x). A hypothesis taking 2.1x baseline time fails.
        memory_budget_multiplier: How much more memory than baseline is allowed.
            Default 2.0 (2x).
        jit_grace_seconds: Extra seconds added to the time budget to account
            for JIT compilation overhead. JAX compiles computation graphs on
            first call, which adds ~0.2-1.0s of one-time cost that baselines
            (which run with warm JIT) don't pay. Default 1.0s.

    Returns:
        EvalResult with verdict, gate outcomes, and improvement/regression lists.

    For example::

        from carnot.autoresearch.sandbox import SandboxResult
        from carnot.autoresearch.baselines import BaselineRecord, BenchmarkBaseline

        sandbox_result = SandboxResult(
            success=True,
            metrics={"bench1": {"final_energy": 0.45, "wall_clock_seconds": 1.0}},
        )
        baselines = BaselineRecord(benchmarks={
            "bench1": BenchmarkBaseline(final_energy=0.5, wall_clock_seconds=1.0),
        })
        result = evaluate_hypothesis(sandbox_result, baselines)
        assert result.verdict == "PASS"  # 0.45 < 0.5, so it improved!

    Spec: REQ-AUTO-005
    """
    # If the sandbox itself failed (crash, timeout, bad return type),
    # immediately fail — there are no metrics to evaluate.
    if not sandbox_result.success:
        return EvalResult(
            verdict="FAIL",
            reason=f"Sandbox failed: {sandbox_result.error}",
        )

    metrics = sandbox_result.metrics
    improvements: list[str] = []
    regressions: list[str] = []

    # === Primary gate: energy check per benchmark ===
    # For each benchmark in the baselines, compare final_energy values.
    primary_pass = True
    for name, baseline in baselines.benchmarks.items():
        if name not in metrics:
            continue  # Benchmark not present in hypothesis results — skip
        bench_metrics = metrics[name]
        if not isinstance(bench_metrics, dict):
            continue  # Unexpected format — skip

        bench_energy = bench_metrics.get("final_energy")
        if bench_energy is None:
            continue  # No energy metric reported — skip

        # Compute absolute tolerance: |baseline_energy| * tolerance_fraction.
        # Using absolute tolerance (not relative) correctly handles negative
        # energies. For example, if baseline is -10.0 and tolerance is 0.001,
        # the allowed range is up to -10.0 + 0.01 = -9.99.
        abs_tol = abs(baseline.final_energy) * energy_regression_tolerance
        if bench_energy > baseline.final_energy + abs_tol:
            # Hypothesis produced higher (worse) energy — regression
            primary_pass = False
            regressions.append(name)
        elif bench_energy < baseline.final_energy - abs_tol:
            # Hypothesis produced lower (better) energy — improvement!
            improvements.append(name)
        # If within tolerance band: neither improvement nor regression

    # === Secondary gate: time budget ===
    # Check that the hypothesis doesn't take too much longer than baseline.
    secondary_pass = True
    for name, baseline in baselines.benchmarks.items():
        if name not in metrics:
            continue
        bench_metrics = metrics[name]
        if not isinstance(bench_metrics, dict):
            continue
        bench_time = bench_metrics.get("wall_clock_seconds")
        if bench_time is not None and baseline.wall_clock_seconds > 0:
            # Time budget = multiplier * baseline + JIT grace period.
            # The grace period accounts for JAX JIT compilation on first call,
            # which hypotheses pay but warm baselines don't.
            budget = baseline.wall_clock_seconds * time_budget_multiplier + jit_grace_seconds
            if bench_time > budget:
                secondary_pass = False

    # === Tertiary gate: memory budget ===
    # Check that peak memory doesn't exceed the multiplier * max baseline memory.
    tertiary_pass = True
    peak_memory = metrics.get("peak_memory_mb")
    if peak_memory is not None:
        # Find the maximum baseline memory across all benchmarks
        max_baseline_memory = max(
            (b.peak_memory_mb for b in baselines.benchmarks.values() if b.peak_memory_mb > 0),
            default=0,
        )
        if max_baseline_memory > 0 and peak_memory > max_baseline_memory * memory_budget_multiplier:
            tertiary_pass = False

    # === Determine verdict ===
    if regressions and improvements:
        # Mixed bag: some benchmarks got better, some got worse.
        # Needs human review to decide if the trade-off is acceptable.
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
        # All gates passed!
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
