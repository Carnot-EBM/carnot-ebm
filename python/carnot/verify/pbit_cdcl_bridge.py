"""Exp 5089 p-bit guided SAT assumption bridge.

This diagnostic checks a narrow pattern: a stochastic Ising/p-bit sampler can
suggest SAT assumptions, but an exact solver remains the only authority for
truth labels and satisfying assignments. The module intentionally stays CPU
only and does not call an LLM.

Spec: REQ-VERIFY-5089, SCENARIO-VERIFY-5089.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import z3


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]

ARTIFACT_FILENAME = "experiment_5089_pbit_guided_cdcl_bridge_v467.json"
INSTANCE_FAMILY = "planted_consensus_3sat_v1"
INFERENCE_SUBSTRATE = "deterministic_solver_plus_stochastic_sampler"
EXACT_SOLVER_USED = "z3"
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "n_instances",
    "instance_family",
    "exact_solver_used",
    "correctness_preserved",
    "pure_solver_effort",
    "random_assumption_effort",
    "pbit_guided_effort",
    "delta_effort_vs_pure",
    "fallback_rate",
    "helps_declared_family",
    "flagged_adversarial",
)


@dataclass(frozen=True)
class CnfInstance:
    """A small signed-literal CNF instance for the diagnostic family."""

    instance_id: str
    family: str
    n_vars: int
    clauses: tuple[tuple[int, ...], ...]
    planted_assignment: tuple[bool, ...]
    expected_satisfiable: bool


@dataclass(frozen=True)
class SolveResult:
    """One exact solver check, optionally under temporary assumptions."""

    status: str
    assignment: tuple[bool, ...] | None
    assumptions: tuple[int, ...]
    duration_s: float
    conflicts: int | None
    propagations: int | None
    decisions: int | None
    solver_stats: JsonDict

    @property
    def probe_count(self) -> int:
        """Every solver check is one deterministic probe."""

        return 1


@dataclass(frozen=True)
class ArmResult:
    """The visible result for a pure or assumption-guided arm."""

    status: str
    assignment: tuple[bool, ...] | None
    assumptions: tuple[int, ...]
    primary_status: str
    fallback_used: bool
    solution_verified: bool
    effort: JsonDict

    def to_json(self) -> JsonDict:
        return {
            "status": self.status,
            "assignment": list(self.assignment) if self.assignment is not None else None,
            "assumptions": list(self.assumptions),
            "primary_status": self.primary_status,
            "fallback_used": self.fallback_used,
            "solution_verified": self.solution_verified,
            "effort": dict(self.effort),
        }


class ExactSatAuthority:
    """Z3-backed SAT authority for labels, models, and model verification."""

    solver_name = EXACT_SOLVER_USED

    def __init__(self, *, collect_solver_stats: bool = False) -> None:
        self.collect_solver_stats = collect_solver_stats

    def solve(
        self,
        instance: CnfInstance,
        assumptions: Sequence[int] = (),
        *,
        clock: ClockFn = time.perf_counter,
    ) -> SolveResult:
        """Run Z3 once on the CNF with optional temporary assumptions."""

        variables = _z3_variables(instance.n_vars)
        solver = z3.Solver()
        for clause in instance.clauses:
            solver.add(z3.Or(*[_z3_literal(variables, literal) for literal in clause]))
        started_at = clock()
        check_result = solver.check(*[_z3_literal(variables, literal) for literal in assumptions])
        duration_s = clock() - started_at
        status = str(check_result)
        assignment = None
        if check_result == z3.sat:
            model = solver.model()
            assignment = tuple(bool(z3.is_true(model.eval(variables[i], model_completion=True))) for i in range(instance.n_vars))
        stats = _solver_stats(solver) if self.collect_solver_stats else {}
        return SolveResult(
            status=status,
            assignment=assignment,
            assumptions=tuple(assumptions),
            duration_s=duration_s,
            conflicts=_first_stat(stats, ("conflicts", "sat conflicts")),
            propagations=_first_stat(stats, ("propagations", "sat propagations", "binary propagations")),
            decisions=_first_stat(stats, ("decisions", "sat decisions")),
            solver_stats=stats,
        )

    def verify_assignment(self, instance: CnfInstance, assignment: Sequence[bool]) -> bool:
        """Accept an assignment only when Z3 proves the CNF plus bindings SAT."""

        if len(assignment) != instance.n_vars:
            return False
        variables = _z3_variables(instance.n_vars)
        solver = z3.Solver()
        for clause in instance.clauses:
            solver.add(z3.Or(*[_z3_literal(variables, literal) for literal in clause]))
        for index, value in enumerate(assignment):
            solver.add(variables[index] == bool(value))
        return solver.check() == z3.sat


class PBitConsensusSampler:
    """Local p-bit/Gibbs sampler that proposes high-consensus SAT literals."""

    def __init__(
        self,
        *,
        seed: int = 5089,
        n_samples: int = 96,
        burn_in: int = 24,
        beta: float = 2.5,
        consensus_threshold: float = 0.7,
        max_assumptions: int = 4,
    ) -> None:
        self.seed = seed
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.beta = beta
        self.consensus_threshold = consensus_threshold
        self.max_assumptions = max_assumptions

    def propose_assumptions(self, instance: CnfInstance) -> tuple[int, ...]:
        """Return high-consensus signed literals from stochastic samples."""

        rng = random.Random(self.seed + _instance_seed(instance))
        state = [rng.choice((False, True)) for _ in range(instance.n_vars)]
        counts = [0 for _ in range(instance.n_vars)]
        total_steps = self.burn_in + self.n_samples
        for step in range(total_steps):
            order = list(range(instance.n_vars))
            rng.shuffle(order)
            for var_index in order:
                state[var_index] = self._draw_var(instance, state, var_index, rng)
            if step >= self.burn_in:
                for index, value in enumerate(state):
                    counts[index] += int(value)

        candidates: list[tuple[float, int]] = []
        for index, true_count in enumerate(counts):
            true_rate = true_count / max(1, self.n_samples)
            if true_rate >= self.consensus_threshold:
                candidates.append((true_rate, index + 1))
            elif true_rate <= 1.0 - self.consensus_threshold:
                candidates.append((1.0 - true_rate, -(index + 1)))
        candidates.sort(key=lambda item: (-item[0], abs(item[1])))
        return tuple(literal for _, literal in candidates[: self.max_assumptions])

    def _draw_var(
        self,
        instance: CnfInstance,
        state: list[bool],
        var_index: int,
        rng: random.Random,
    ) -> bool:
        state[var_index] = False
        false_energy = _cnf_energy(instance, state)
        state[var_index] = True
        true_energy = _cnf_energy(instance, state)
        p_true = 1.0 / (1.0 + math.exp(self.beta * (true_energy - false_energy)))
        return rng.random() < p_true


def build_declared_family() -> tuple[CnfInstance, ...]:
    """Build deterministic planted 3-literal CNFs with exact labels."""

    planted_rows = (
        (True, False, True, True, False, False, True, False),
        (False, True, True, False, True, False, True, True),
        (True, True, False, False, True, True, False, False),
        (False, False, True, True, True, False, False, True),
    )
    return tuple(_planted_instance(index, row) for index, row in enumerate(planted_rows, start=1))


def run_diagnostic(
    *,
    family: Sequence[CnfInstance] | None = None,
    authority: ExactSatAuthority | None = None,
    sampler: PBitConsensusSampler | None = None,
    started_at: float | None = None,
    clock: ClockFn = time.perf_counter,
) -> JsonDict:
    """Run pure, random-assumption, and p-bit-guided arms on the CNF family."""

    active_family = tuple(family or build_declared_family())
    exact = authority or ExactSatAuthority()
    pbit_sampler = sampler or PBitConsensusSampler()
    started = clock() if started_at is None else started_at
    rows: list[JsonDict] = []
    arm_totals = _empty_arm_totals()
    correctness_preserved = True
    flagged_adversarial = False

    for index, instance in enumerate(active_family):
        label_result = exact.solve(instance, clock=clock)
        pure = _pure_arm(instance, exact, clock=clock)
        random_arm = _assumption_arm(
            instance,
            exact,
            _random_assumptions(instance, seed=811 + index),
            clock=clock,
        )
        pbit_arm = _assumption_arm(
            instance,
            exact,
            pbit_sampler.propose_assumptions(instance),
            clock=clock,
        )
        for name, arm in (
            ("pure_solver", pure),
            ("random_assumption", random_arm),
            ("pbit_guided", pbit_arm),
        ):
            _add_arm_totals(arm_totals[name], arm)
            if arm.status == "sat" and not arm.solution_verified:  # pragma: no cover - guarded by exact authority.
                correctness_preserved = False
                flagged_adversarial = True
        exact_status = label_result.status == "sat"
        if exact_status is not instance.expected_satisfiable:  # pragma: no cover - guarded by declared fixture tests.
            correctness_preserved = False
            flagged_adversarial = True
        rows.append(
            {
                "instance_id": instance.instance_id,
                "known_satisfiable": instance.expected_satisfiable,
                "exact_status": exact_status,
                "n_vars": instance.n_vars,
                "n_clauses": len(instance.clauses),
                "pure_solver": pure.to_json(),
                "random_assumption": random_arm.to_json(),
                "pbit_guided": pbit_arm.to_json(),
            }
        )

    pure_effort = _summarize_effort(arm_totals["pure_solver"])
    random_effort = _summarize_effort(arm_totals["random_assumption"])
    pbit_effort = _summarize_effort(arm_totals["pbit_guided"])
    random_delta = random_effort["total_effort_score"] - pure_effort["total_effort_score"]
    pbit_delta = pbit_effort["total_effort_score"] - pure_effort["total_effort_score"]
    helps = bool(correctness_preserved and pbit_delta < 0)
    verdict = (
        f"success_pbit_guided_cdcl_effort_reduction_{INSTANCE_FAMILY}"
        if helps
        else "complete_pbit_guided_cdcl_distribution_sensitive_no_win"
    )
    duration_s = max(0.0, clock() - started)
    artifact: JsonDict = {
        "honest_verdict": verdict,
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_instances": len(active_family),
        "instance_family": INSTANCE_FAMILY,
        "exact_solver_used": exact.solver_name,
        "correctness_preserved": correctness_preserved,
        "pure_solver_effort": pure_effort,
        "random_assumption_effort": random_effort,
        "pbit_guided_effort": pbit_effort,
        "delta_effort_vs_pure": {
            "random_assumption": random_delta,
            "pbit_guided": pbit_delta,
        },
        "fallback_rate": _rate(arm_totals["pbit_guided"]["fallbacks"], len(active_family)),
        "helps_declared_family": helps,
        "flagged_adversarial": flagged_adversarial,
        "fallback_rates_by_arm": {
            "random_assumption": _rate(arm_totals["random_assumption"]["fallbacks"], len(active_family)),
            "pbit_guided": _rate(arm_totals["pbit_guided"]["fallbacks"], len(active_family)),
        },
        "sampler_config": {
            "kind": "local_boolean_gibbs_pbit_consensus",
            "seed": pbit_sampler.seed,
            "n_samples": pbit_sampler.n_samples,
            "burn_in": pbit_sampler.burn_in,
            "beta": pbit_sampler.beta,
            "consensus_threshold": pbit_sampler.consensus_threshold,
            "max_assumptions": pbit_sampler.max_assumptions,
        },
        "per_instance_results": rows,
        "field_principles": _field_principles(),
        "spec_refs": ["REQ-VERIFY-5089", "SCENARIO-VERIFY-5089"],
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    output_path: Path | None = None,
    repo_root: Path | None = None,
) -> JsonDict:
    """Run the diagnostic and write the Exp 5089 terminal JSON artifact."""

    root = repo_root or Path(__file__).resolve().parents[3]
    destination = output_path or root / "results" / ARTIFACT_FILENAME
    payload = run_diagnostic()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the artifact violates the Exp 5089 terminal contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] == "live_llm_inference":
        raise ValueError("Exp 5089 must not claim live_llm_inference")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    if artifact["exact_solver_used"] != EXACT_SOLVER_USED:
        raise ValueError(f"exact_solver_used must be {EXACT_SOLVER_USED}")
    if not isinstance(artifact["correctness_preserved"], bool):
        raise ValueError("correctness_preserved must be a boolean")
    if not 0.0 <= float(artifact["fallback_rate"]) <= 1.0:
        raise ValueError("fallback_rate must be in [0, 1]")
    verdict = str(artifact["honest_verdict"])
    prefixes = (
        "success_pbit_guided_cdcl_effort_reduction_",
        "complete_pbit_guided_cdcl_distribution_sensitive_no_win",
    )
    if not verdict.startswith(prefixes):
        raise ValueError("honest_verdict has no accepted Exp 5089 terminal prefix")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_ARTIFACT_FIELDS) <= set(principles):
        raise ValueError("field_principles must annotate every required field")


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    payload = write_artifact()
    print(json.dumps({field: payload[field] for field in REQUIRED_ARTIFACT_FIELDS}, indent=2, sort_keys=True))
    return 0


def _planted_instance(index: int, planted_assignment: Sequence[bool]) -> CnfInstance:
    n_vars = len(planted_assignment)
    clauses: list[tuple[int, ...]] = []
    for var_index, value in enumerate(planted_assignment, start=1):
        literal = var_index if value else -var_index
        clauses.append((literal, literal, literal))
    windows = (
        (1, 2, 3),
        (2, 4, 5),
        (3, 5, 6),
        (4, 6, 7),
        (1, 7, 8),
        (2, 6, 8),
        (3, 4, 8),
    )
    for offset, window in enumerate(windows):
        clause: list[int] = []
        for position, var_index in enumerate(window):
            planted_value = planted_assignment[var_index - 1]
            keep_planted = (index + offset + position) % 3 != 0
            literal_positive = planted_value if keep_planted else not planted_value
            clause.append(var_index if literal_positive else -var_index)
        if not _clause_satisfied(tuple(clause), planted_assignment):  # pragma: no cover - deterministic rows satisfy.
            first_var = abs(clause[0])
            clause[0] = first_var if planted_assignment[first_var - 1] else -first_var
        clauses.append(tuple(clause))
    return CnfInstance(
        instance_id=f"{INSTANCE_FAMILY}_{index:02d}",
        family=INSTANCE_FAMILY,
        n_vars=n_vars,
        clauses=tuple(clauses),
        planted_assignment=tuple(planted_assignment),
        expected_satisfiable=True,
    )


def _pure_arm(instance: CnfInstance, authority: ExactSatAuthority, *, clock: ClockFn) -> ArmResult:
    result = authority.solve(instance, clock=clock)
    verified = result.assignment is not None and authority.verify_assignment(instance, result.assignment)
    return ArmResult(
        status=result.status,
        assignment=result.assignment,
        assumptions=(),
        primary_status=result.status,
        fallback_used=False,
        solution_verified=verified,
        effort=_effort_from_results([result], fallback_used=False),
    )


def _assumption_arm(
    instance: CnfInstance,
    authority: ExactSatAuthority,
    assumptions: Sequence[int],
    *,
    clock: ClockFn,
) -> ArmResult:
    primary = authority.solve(instance, assumptions=assumptions, clock=clock)
    verified = primary.assignment is not None and authority.verify_assignment(instance, primary.assignment)
    if primary.status == "sat" and verified:
        return ArmResult(
            status=primary.status,
            assignment=primary.assignment,
            assumptions=tuple(assumptions),
            primary_status=primary.status,
            fallback_used=False,
            solution_verified=True,
            effort=_effort_from_results([primary], fallback_used=False),
        )
    fallback = authority.solve(instance, clock=clock)
    fallback_verified = fallback.assignment is not None and authority.verify_assignment(instance, fallback.assignment)
    return ArmResult(
        status=fallback.status,
        assignment=fallback.assignment,
        assumptions=tuple(assumptions),
        primary_status=primary.status,
        fallback_used=True,
        solution_verified=fallback_verified,
        effort=_effort_from_results([primary, fallback], fallback_used=True),
    )


def _random_assumptions(instance: CnfInstance, *, seed: int, max_assumptions: int = 4) -> tuple[int, ...]:
    rng = random.Random(seed + _instance_seed(instance))
    variables = list(range(1, instance.n_vars + 1))
    rng.shuffle(variables)
    literals = []
    for var_index in variables[:max_assumptions]:
        literals.append(var_index if rng.choice((False, True)) else -var_index)
    return tuple(literals)


def _cnf_energy(instance: CnfInstance, assignment: Sequence[bool]) -> int:
    return sum(not _clause_satisfied(clause, assignment) for clause in instance.clauses)


def _clause_satisfied(clause: Sequence[int], assignment: Sequence[bool]) -> bool:
    for literal in clause:
        value = assignment[abs(literal) - 1]
        if (literal > 0 and value) or (literal < 0 and not value):
            return True
    return False


def _z3_variables(n_vars: int) -> tuple[z3.BoolRef, ...]:
    return tuple(z3.Bool(f"x_{index}") for index in range(1, n_vars + 1))


def _z3_literal(variables: Sequence[z3.BoolRef], literal: int) -> z3.BoolRef:
    variable = variables[abs(literal) - 1]
    return variable if literal > 0 else z3.Not(variable)


def _solver_stats(solver: z3.Solver) -> JsonDict:  # pragma: no cover - optional Z3 stats path.
    stats = solver.statistics()
    result: JsonDict = {}
    for key in stats.keys():
        value = stats.get_key_value(key)
        if isinstance(value, int | float):
            result[key] = value
    return result


def _first_stat(stats: Mapping[str, Any], names: Sequence[str]) -> int | None:
    for name in names:
        value = stats.get(name)
        if isinstance(value, int | float):
            return int(value)  # pragma: no cover - optional stats path.
    return None


def _effort_from_results(results: Sequence[SolveResult], *, fallback_used: bool) -> JsonDict:
    conflicts = _sum_optional(result.conflicts for result in results)
    propagations = _sum_optional(result.propagations for result in results)
    decisions = _sum_optional(result.decisions for result in results)
    probe_count = sum(result.probe_count for result in results)
    stat_score = sum(value or 0 for value in (conflicts, propagations, decisions))
    if stat_score > 0:  # pragma: no cover - optional stats path.
        metric = "solver_conflicts_propagations_decisions"
        total_score = stat_score
    else:
        metric = "probe_count_proxy"
        total_score = probe_count
    return {
        "metric": metric,
        "total_effort_score": total_score,
        "probe_count": probe_count,
        "duration_s": round(sum(result.duration_s for result in results), 6),
        "conflicts": conflicts,
        "propagations": propagations,
        "decisions": decisions,
        "fallback_used": fallback_used,
    }


def _sum_optional(values: Sequence[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)  # pragma: no cover - optional stats path.


def _empty_arm_totals() -> dict[str, JsonDict]:
    return {
        "pure_solver": _new_total(),
        "random_assumption": _new_total(),
        "pbit_guided": _new_total(),
    }


def _new_total() -> JsonDict:
    return {
        "instances": 0,
        "total_effort_score": 0,
        "probe_count": 0,
        "duration_s": 0.0,
        "conflicts": 0,
        "propagations": 0,
        "decisions": 0,
        "stat_fields_seen": False,
        "fallbacks": 0,
    }


def _add_arm_totals(total: JsonDict, arm: ArmResult) -> None:
    effort = arm.effort
    total["instances"] += 1
    total["total_effort_score"] += effort["total_effort_score"]
    total["probe_count"] += effort["probe_count"]
    total["duration_s"] += effort["duration_s"]
    total["fallbacks"] += int(arm.fallback_used)
    for field in ("conflicts", "propagations", "decisions"):
        value = effort[field]
        if value is not None:  # pragma: no cover - optional stats path.
            total[field] += value
            total["stat_fields_seen"] = True


def _summarize_effort(total: Mapping[str, Any]) -> JsonDict:
    stat_fields_seen = bool(total["stat_fields_seen"])
    return {
        "metric": "solver_conflicts_propagations_decisions" if stat_fields_seen else "probe_count_proxy",
        "total_effort_score": total["total_effort_score"],
        "probe_count": total["probe_count"],
        "duration_s": round(float(total["duration_s"]), 6),
        "conflicts": total["conflicts"] if stat_fields_seen else None,
        "propagations": total["propagations"] if stat_fields_seen else None,
        "decisions": total["decisions"] if stat_fields_seen else None,
        "fallbacks": total["fallbacks"],
        "instances": total["instances"],
    }


def _rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else round(count / total, 6)


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "Terminal prefix: success only for verified effort reduction, otherwise complete no-win.",
        "duration_s": "Measured local CPU diagnostic runtime in seconds.",
        "inference_substrate": "Must be deterministic_solver_plus_stochastic_sampler, not live_llm_inference.",
        "n_instances": "Number of declared CNF instances evaluated.",
        "instance_family": "Declared deterministic small SAT/CNF family.",
        "exact_solver_used": "Exact authority used for labels and model verification.",
        "correctness_preserved": "True only when exact labels and all claimed solutions verify.",
        "pure_solver_effort": "Exact solver effort with no temporary assumptions.",
        "random_assumption_effort": "Exact solver effort after random temporary assumptions plus fallback.",
        "pbit_guided_effort": "Exact solver effort after p-bit consensus assumptions plus fallback.",
        "delta_effort_vs_pure": "Signed arm effort minus pure effort; negative means less effort than pure.",
        "fallback_rate": "P-bit-guided fraction requiring pure exact fallback.",
        "helps_declared_family": "True only when verified p-bit effort is below pure effort on this family.",
        "flagged_adversarial": "True when labels or claimed models fail exact verification.",
    }


def _instance_seed(instance: CnfInstance) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(instance.instance_id))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
