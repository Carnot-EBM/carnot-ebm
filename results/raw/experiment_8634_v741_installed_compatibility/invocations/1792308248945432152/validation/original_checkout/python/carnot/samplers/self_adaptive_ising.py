"""Self-adaptive Ising probe for FoVer arithmetic constraints.

**Researcher summary:**
    Implements the arXiv:2501.04971 self-adaptive Ising idea for a small
    CPU-only FoVer probe. Each FoVer arithmetic equation is converted into a
    binary answer-state problem: Ising bits encode a candidate integer answer,
    the base energy prefers a plausible wrong answer, and the constraint
    requires the encoded answer to equal the verified arithmetic target.

**Detailed explanation for engineers:**
    Static Ising constraint verifiers usually pick one penalty value before
    sampling. If the value is too small, the sampler can prefer low base energy
    states that violate the arithmetic constraint. The self-adaptive loop keeps
    the same weak quadratic penalty but adds a Lagrange multiplier that changes
    after each readout:

        L(s, lambda) = E_ising(s) + lambda * g(s) + (rho / 2) * g(s)^2
        lambda <- lambda + eta * g(s)

    where ``g(s)`` is the normalized arithmetic residual. This module uses an
    exact low-temperature CPU readout over small bit-width states so the probe is
    deterministic and fast. The measured quantity is not hardware sampling
    throughput; it is whether the adaptive energy landscape reaches a feasible
    arithmetic state with fewer sweeps than a weak static-penalty baseline.

Spec: REQ-VERIFY-1385, SCENARIO-VERIFY-1385
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_FOVER_EQUATION_RE = re.compile(r"<<([^<>]+?)=(-?\d+(?:\.\d+)?)>>")
_SAFE_OPERATORS: dict[type[ast.operator | ast.unaryop], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@dataclass(frozen=True)
class FoVerArithmeticConstraintProblem:
    """Single FoVer arithmetic equation encoded as a binary Ising problem.

    ``target`` is the verified arithmetic answer extracted from a FoVer
    ``<<expr=target>>`` annotation. ``preferred_value`` is a deterministic
    wrong answer used by the base Ising objective to model the current verifier
    prior pulling toward an infeasible state. The augmented Lagrangian must
    overcome that prior without manually retuning the static penalty.

    Spec: REQ-VERIFY-1385-4
    """

    problem_id: str
    expression: str
    target: int
    bit_width: int
    preferred_value: int
    source: str
    row_label: str
    text_excerpt: str

    @property
    def max_value(self) -> int:
        """Largest integer representable by the answer bits."""
        return (1 << self.bit_width) - 1

    @property
    def value_scale(self) -> float:
        """Scale used to keep residuals comparable across different targets."""
        return float(max(1, self.max_value))


@dataclass(frozen=True)
class IsingRunResult:
    """Convergence result for one static or adaptive Ising run."""

    converged: bool
    convergence_steps: int
    final_value: int
    final_constraint_violation: float
    lambda_updates: int
    lambda_history: tuple[float, ...]
    value_history: tuple[int, ...]
    violation_history: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for experiment artifacts."""
        return {
            "converged": self.converged,
            "convergence_steps": self.convergence_steps,
            "final_value": self.final_value,
            "final_constraint_violation": self.final_constraint_violation,
            "lambda_updates": self.lambda_updates,
            "lambda_history": list(self.lambda_history),
            "value_history": list(self.value_history),
            "violation_history": list(self.violation_history),
        }


def _safe_eval_arithmetic(expression: str) -> float:
    """Evaluate a FoVer arithmetic expression without exposing Python builtins.

    FoVer GSM8K rows usually use annotations such as ``<<4*3=12>>``. A full
    symbolic math parser would be unnecessary for this probe, but ``eval`` would
    be the wrong tool because corpus text is data. This AST walker accepts only
    numeric literals and arithmetic operators.
    """

    cleaned = expression.replace(",", "").replace("^", "**").strip()
    tree = ast.parse(cleaned, mode="eval")
    return float(_eval_ast_node(tree.body))


def _eval_ast_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported arithmetic operator: {ast.dump(node.op)}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        return float(op(left, right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {ast.dump(node.op)}")
        return float(op(_eval_ast_node(node.operand)))
    raise ValueError(f"unsupported arithmetic expression: {ast.dump(node)}")


def _read_fover_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, list):
        raise ValueError(f"expected FoVer list at {path}, got {type(loaded).__name__}")
    return [row for row in loaded if isinstance(row, dict)]


def _preferred_value(target: int, bit_width: int, index: int) -> int:
    """Pick a deterministic wrong attractor that remains representable."""
    max_value = (1 << bit_width) - 1
    offset = max(5, int(round(max_value * 0.08)))
    direction = 1 if index % 2 == 0 else -1
    candidate = target + direction * offset
    if not 0 <= candidate <= max_value:
        candidate = target - direction * offset
    if candidate == target:
        candidate = min(max_value, target + 1) if target < max_value else max(0, target - 1)
    return int(candidate)


def load_fover_arithmetic_constraint_problems(
    *,
    repo_root: str | Path,
    limit: int = 8,
    bit_width: int = 8,
) -> list[FoVerArithmeticConstraintProblem]:
    """Load local FoVer arithmetic equations as small Ising constraint problems.

    The loader prefers the structured v4 FoVer splits because they contain
    explicit GSM8K arithmetic annotations in the ``<<expr=answer>>`` format.
    Only equations whose expression evaluates to the target are used, so the
    Ising problem is anchored to a verified arithmetic constraint rather than
    to an arbitrary number extracted from free text.

    Spec: REQ-VERIFY-1385-4
    """

    root = Path(repo_root)
    candidate_paths = [
        root / "data" / "fover_train_v4.json",
        root / "data" / "fover_test_v4.json",
        root / "data" / "fover_corpus.jsonl",
    ]
    max_value = (1 << bit_width) - 1
    problems: list[FoVerArithmeticConstraintProblem] = []
    seen: set[tuple[str, int]] = set()

    for path in candidate_paths:
        if not path.exists():
            continue
        for row in _read_fover_rows(path):
            text = str(row.get("step_text") or row.get("response") or "")
            for expression, raw_target in _FOVER_EQUATION_RE.findall(text):
                try:
                    evaluated = _safe_eval_arithmetic(expression)
                    target_float = float(raw_target)
                except (SyntaxError, ValueError, OverflowError, ZeroDivisionError):
                    continue
                if not math.isfinite(evaluated) or not math.isfinite(target_float):
                    continue
                if abs(evaluated - target_float) > 1e-9:
                    continue
                if abs(target_float - round(target_float)) > 1e-9:
                    continue
                target = int(round(target_float))
                if not 0 <= target <= max_value:
                    continue
                key = (expression, target)
                if key in seen:
                    continue
                seen.add(key)
                problem_id = str(row.get("question_id") or f"{path.stem}:{len(problems)}")
                preferred = _preferred_value(target, bit_width, len(problems))
                problems.append(
                    FoVerArithmeticConstraintProblem(
                        problem_id=problem_id,
                        expression=expression.strip(),
                        target=target,
                        bit_width=bit_width,
                        preferred_value=preferred,
                        source=str(path.relative_to(root)),
                        row_label=str(row.get("label", "unknown")),
                        text_excerpt=text[:240],
                    )
                )
                if len(problems) >= limit:
                    return problems

    if len(problems) < limit:
        raise ValueError(
            f"found only {len(problems)} FoVer arithmetic equations, expected at least {limit}"
        )
    return problems


def lagrange_relaxation_update(
    *,
    lambda_value: float,
    raw_violation: float,
    eta: float,
    value_scale: float,
) -> float:
    """Apply ``lambda_{k+1} = lambda_k + eta * g(x_k)``.

    ``raw_violation`` is the signed arithmetic residual ``value - target``.
    Dividing by ``value_scale`` gives the normalized residual ``g(x_k)`` used by
    both the energy and the update. Equality constraints use an unconstrained
    real-valued multiplier, so negative residuals are allowed to reduce lambda.

    Spec: REQ-VERIFY-1385-2
    """

    return float(lambda_value + eta * (raw_violation / value_scale))


def augmented_lagrangian_energy(
    *,
    base_energy: float,
    normalized_violation: float,
    lambda_value: float,
    rho: float,
) -> float:
    """Compute ``E + lambda*g + rho/2*g^2`` for one equality constraint.

    Spec: REQ-VERIFY-1385-3
    """

    return float(
        base_energy + lambda_value * normalized_violation + 0.5 * rho * normalized_violation**2
    )


@dataclass
class SelfAdaptiveIsingMachine:
    """Small CPU self-adaptive Ising machine for one FoVer arithmetic equation.

    The answer bits encode an integer ``y``. The base Ising objective prefers
    ``preferred_value`` while the arithmetic constraint requires ``y == target``.
    Since ``(sum_i 2^i s_i - c)^2`` is a QUBO, both the base preference and the
    quadratic penalty are Ising-compatible. The implementation enumerates all
    bit states as a deterministic low-temperature readout.

    Spec: REQ-VERIFY-1385
    """

    problem: FoVerArithmeticConstraintProblem
    objective_weight: float = 8.0
    rho: float = 0.2
    eta: float = 1.5
    max_steps: int = 50
    threshold: float = 0.0

    def _base_energy_for_value(self, value: int) -> float:
        residual = (value - self.problem.preferred_value) / self.problem.value_scale
        return float(self.objective_weight * residual**2)

    def _normalized_violation_for_value(self, value: int) -> float:
        return float((value - self.problem.target) / self.problem.value_scale)

    def _raw_violation_for_value(self, value: int) -> float:
        return float(value - self.problem.target)

    def _energy_for_value(self, value: int, lambda_value: float, rho: float) -> float:
        return augmented_lagrangian_energy(
            base_energy=self._base_energy_for_value(value),
            normalized_violation=self._normalized_violation_for_value(value),
            lambda_value=lambda_value,
            rho=rho,
        )

    def _best_value(self, *, lambda_value: float, rho: float) -> int:
        best_value = 0
        best_energy = float("inf")
        for value in range(self.problem.max_value + 1):
            energy = self._energy_for_value(value, lambda_value, rho)
            if energy < best_energy:
                best_value = value
                best_energy = energy
        return best_value

    def run_static_penalty(self, *, rho: float | None = None) -> IsingRunResult:
        """Run the weak static-penalty baseline with fixed lambda=0.

        The static landscape does not change across sweeps. If the first
        low-temperature readout is infeasible, the baseline is recorded as a
        timeout at ``max_steps + 1``. This makes the speedup calculation honest:
        the baseline did not reach the threshold within the same step budget.

        Spec: REQ-VERIFY-1385-5
        """

        penalty = self.rho if rho is None else rho
        value = self._best_value(lambda_value=0.0, rho=penalty)
        violation = abs(self._raw_violation_for_value(value))
        converged = violation <= self.threshold
        steps = 1 if converged else self.max_steps + 1
        return IsingRunResult(
            converged=converged,
            convergence_steps=steps,
            final_value=value,
            final_constraint_violation=violation,
            lambda_updates=0,
            lambda_history=(0.0,),
            value_history=(value,),
            violation_history=(violation,),
        )

    def run_adaptive_lagrange(self) -> IsingRunResult:
        """Run adaptive Lagrange relaxation, updating lambda after each readout.

        Spec: REQ-VERIFY-1385-2, REQ-VERIFY-1385-3, REQ-VERIFY-1385-5
        """

        lambda_value = 0.0
        lambda_history: list[float] = [lambda_value]
        value_history: list[int] = []
        violation_history: list[float] = []
        lambda_updates = 0

        for step in range(1, self.max_steps + 1):
            value = self._best_value(lambda_value=lambda_value, rho=self.rho)
            raw_violation = self._raw_violation_for_value(value)
            violation = abs(raw_violation)
            value_history.append(value)
            violation_history.append(violation)
            if violation <= self.threshold:
                return IsingRunResult(
                    converged=True,
                    convergence_steps=step,
                    final_value=value,
                    final_constraint_violation=violation,
                    lambda_updates=lambda_updates,
                    lambda_history=tuple(lambda_history),
                    value_history=tuple(value_history),
                    violation_history=tuple(violation_history),
                )
            lambda_value = lagrange_relaxation_update(
                lambda_value=lambda_value,
                raw_violation=raw_violation,
                eta=self.eta,
                value_scale=self.problem.value_scale,
            )
            lambda_updates += 1
            lambda_history.append(lambda_value)

        final_value = value_history[-1]
        final_violation = violation_history[-1]
        return IsingRunResult(
            converged=False,
            convergence_steps=self.max_steps + 1,
            final_value=final_value,
            final_constraint_violation=final_violation,
            lambda_updates=lambda_updates,
            lambda_history=tuple(lambda_history),
            value_history=tuple(value_history),
            violation_history=tuple(violation_history),
        )

    def static_penalty_tuning_iterations_saved(
        self,
        *,
        penalty_ladder: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    ) -> int:
        """Estimate how many manual static-penalty trials adaptive avoids.

        The base probe starts at ``rho=0.2``. A manual static baseline would
        usually try larger penalties until the low-temperature readout becomes
        feasible. The returned value is the number of extra ladder settings that
        would have been tested after the base penalty.

        Spec: REQ-VERIFY-1385-6
        """

        for index, penalty in enumerate(penalty_ladder):
            if self.run_static_penalty(rho=penalty).converged:
                return max(0, index)
        return max(0, len(penalty_ladder) - 1)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def run_self_adaptive_ising_probe(
    *,
    repo_root: str | Path,
    limit: int = 8,
    run_date: str = "20260505",
) -> dict[str, Any]:
    """Run the Exp 1385 FoVer side-by-side convergence probe.

    Spec: REQ-VERIFY-1385, SCENARIO-VERIFY-1385
    """

    problems = load_fover_arithmetic_constraint_problems(repo_root=repo_root, limit=limit)
    per_problem: list[dict[str, Any]] = []

    for problem in problems:
        machine = SelfAdaptiveIsingMachine(problem=problem)
        static = machine.run_static_penalty()
        adaptive = machine.run_adaptive_lagrange()
        speedup = static.convergence_steps / max(1, adaptive.convergence_steps)
        static_final = static.final_constraint_violation
        adaptive_final = adaptive.final_constraint_violation
        if static_final > 0.0:
            violation_reduction = (static_final - adaptive_final) / static_final
        else:
            violation_reduction = 0.0
        per_problem.append(
            {
                "problem_id": problem.problem_id,
                "source": problem.source,
                "expression": problem.expression,
                "target": problem.target,
                "preferred_value": problem.preferred_value,
                "static": static.as_dict(),
                "adaptive": adaptive.as_dict(),
                "speedup": speedup,
                "constraint_violation_reduction": violation_reduction,
                "penalty_tuning_iterations_saved": machine.static_penalty_tuning_iterations_saved(),
            }
        )

    faster_count = sum(1 for result in per_problem if result["speedup"] > 1.0)
    adaptive_ising_viable = faster_count >= math.ceil(len(per_problem) / 2)
    speedups = [float(result["speedup"]) for result in per_problem]
    reductions = [float(result["constraint_violation_reduction"]) for result in per_problem]
    lambda_updates = [int(result["adaptive"]["lambda_updates"]) for result in per_problem]
    tuning_saved = [int(result["penalty_tuning_iterations_saved"]) for result in per_problem]

    if adaptive_ising_viable:
        verdict = "adaptive_lagrange_viable_on_fover_arithmetic_probe"
    else:
        verdict = "adaptive_lagrange_not_viable_on_this_fover_slice"

    return {
        "status": "complete",
        "run_date": run_date,
        "constraint_problems_tested": [result["problem_id"] for result in per_problem],
        "static_penalty_convergence_steps": [
            int(result["static"]["convergence_steps"]) for result in per_problem
        ],
        "adaptive_lagrange_convergence_steps": [
            int(result["adaptive"]["convergence_steps"]) for result in per_problem
        ],
        "convergence_speedup": _mean(speedups),
        "constraint_violation_reduction": _mean(reductions),
        "lagrange_multiplier_iterations": _mean([float(value) for value in lambda_updates]),
        "penalty_tuning_iterations_saved": _mean([float(value) for value in tuning_saved]),
        "adaptive_ising_viable": adaptive_ising_viable,
        "honest_verdict": verdict,
        "per_problem_results": per_problem,
        "arxiv_2501_04971_update_rule": "lambda_{k+1} = lambda_k + eta * g(x_k)",
        "energy_form": "E_ising(s) + lambda^T g(s) + (rho/2) * |g(s)|^2",
        "fover_problem_count": len(per_problem),
        "viability_gate": "speedup > 1.0 on at least half of tested problems",
    }
