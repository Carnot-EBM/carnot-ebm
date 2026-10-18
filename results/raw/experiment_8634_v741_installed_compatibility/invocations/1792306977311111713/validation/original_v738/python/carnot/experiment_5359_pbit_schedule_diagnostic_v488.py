"""Exp 5359 deterministic CPU p-bit schedule diagnostic.

Spec refs: REQ-VERIFY-5359, SCENARIO-VERIFY-5359.

This module compares small p-bit update schedules before any hardware claim is
allowed. Each schedule only proposes a CPU-side Ising assignment over bounded
SAT fixtures. The CDCL solver remains authoritative: a schedule can reduce or
increase search work, but it cannot decide the final SAT label unless the
solver validates the proposed assignment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5292_pbit_cdcl_factor_guidance_v483 as cdcl
from carnot import experiment_5300_pbit_cdcl_instance_class_gate_v484 as gate


JsonDict = dict[str, Any]

RUN_DATE = "20260707"
RANDOM_SEED = 5359
EXPERIMENT_ID = "exp5359-pbit-schedule-diagnostic-v488"
MILESTONE = "2026.07.488"
SCHEMA = "carnot.experiment_5359.pbit_schedule_diagnostic.v488"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5359_pbit_schedule_diagnostic_v488.json"
)
INFERENCE_SUBSTRATE = "deterministic_cpu_sampler_simulation"
SPEC_REFS = ("REQ-VERIFY-5359", "SCENARIO-VERIFY-5359")
TERMINAL_PREFIXES = ("complete:", "blocked_")
BASELINE_VARIANT = "baseline_sequential"
SCHEDULE_VARIANTS = (
    BASELINE_VARIANT,
    "partial_deactivation",
    "fully_parallel_inertia",
    "cost_aware_anneal",
    "misleading_assumption_guard",
)
EXPECTED_INSTANCE_CLASSES = (
    "aligned_factor_sat",
    "misleading_factor_sat",
    "neutral_factor_sat",
    "aligned_repair",
    "misleading_repair",
    "neutral_noop_repair",
    "malformed_control",
    "semantic_wrong_control",
)
MISLEADING_CLASSES = (
    "misleading_factor_sat",
    "misleading_repair",
    "semantic_wrong_control",
)
COUNT_METRIC_KEYS = ("conflicts", "decisions", "propagations", "restarts")

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Stable id ties the artifact to this roadmap task.",
    "milestone": "Keeps schedule evidence separate from hardware receipts.",
    "status": "Lets capstone classify signal versus blocked simulation.",
    "honest_verdict": (
        "Terminal prefix `complete:` or `blocked_` prevents ambiguous sampler claims."
    ),
    "inference_substrate": "Expected value is deterministic_cpu_sampler_simulation.",
    "schedule_variant_count": (
        "Bare integer proves more than one schedule was tested."
    ),
    "fixture_count": "Bare integer bounds the diagnostic corpus.",
    "sweeps_to_solution_delta": (
        "Bare numeric measures sample-efficiency impact."
    ),
    "conflict_delta_vs_baseline": (
        "Bare numeric measures solver-facing impact."
    ),
    "misleading_class_harm_rate": (
        "Bare numeric catches distribution-sensitive regressions."
    ),
    "energy_monotonicity_violation_count": (
        "Bare integer detects unstable schedules."
    ),
    "cpu_runtime_s": "Bare numeric prevents hardware-speedup conflation.",
    "false_accept_count": "Bare integer prevents invalid sampler wins.",
    "hardware_speedup_claim": (
        "Bare boolean must be false because this is CPU simulation."
    ),
    "pbit_schedule_signal_ready": (
        "Bare boolean summarizes whether hardware follow-up is justified."
    ),
    "tests_run": "Lists deterministic sampler tests.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "schedule_variant_count",
    "fixture_count",
    "sweeps_to_solution_delta",
    "conflict_delta_vs_baseline",
    "misleading_class_harm_rate",
    "energy_monotonicity_violation_count",
    "cpu_runtime_s",
    "false_accept_count",
    "hardware_speedup_claim",
    "pbit_schedule_signal_ready",
    "tests_run",
)


@dataclass(frozen=True)
class ScheduleInstance:
    """One bounded SAT fixture reused from the existing p-bit/CDCL gate."""

    instance_id: str
    instance_class: str
    source_experiment: str
    n_vars: int
    clauses: tuple[tuple[int, ...], ...]
    source_fixture_id: str
    source_artifact: str
    seed_literals: tuple[int, ...]
    seed_method: str
    lns_repair_agreement: str
    candidate_format_valid: bool
    hardware_execution: bool = False


@dataclass(frozen=True)
class ScheduleProposal:
    """A CPU p-bit schedule proposal before CDCL accepts or overwrites it."""

    schedule_variant: str
    final_state: tuple[bool, ...]
    energy_trace: tuple[int, ...]
    sweeps_to_solution: int
    solution_found: bool

    def as_serializable(self) -> JsonDict:
        return {
            "schedule_variant": self.schedule_variant,
            "hint_literals": list(_state_to_signed_literals(self.final_state)),
            "energy_trace": list(self.energy_trace),
            "final_energy": self.energy_trace[-1],
            "sweeps_to_solution": self.sweeps_to_solution,
            "solution_found": self.solution_found,
            "energy_monotonicity_violations": _energy_monotonicity_violations(
                self.energy_trace
            ),
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_schedule_instances() -> tuple[ScheduleInstance, ...]:
    """Return the exact bounded fixture classes from the Exp 5300 gate."""

    rows: list[ScheduleInstance] = []
    for instance in gate.build_gate_instances():
        rows.append(
            ScheduleInstance(
                instance_id=instance.instance_id,
                instance_class=instance.instance_class,
                source_experiment=instance.source_experiment,
                n_vars=instance.n_vars,
                clauses=instance.clauses,
                source_fixture_id=instance.source_fixture_id,
                source_artifact=instance.source_artifact,
                seed_literals=instance.assumption_literals,
                seed_method=instance.assumption_method,
                lns_repair_agreement=instance.lns_repair_agreement,
                candidate_format_valid=instance.candidate_format_valid,
                hardware_execution=instance.hardware_execution,
            )
        )
    return tuple(rows)


def evaluate_schedule_instance(
    instance: ScheduleInstance,
    schedule_variant: str,
) -> JsonDict:
    """Evaluate one schedule row while keeping CDCL as the final authority."""

    _require(schedule_variant in SCHEDULE_VARIANTS, f"unknown schedule: {schedule_variant}")
    solver_only = cdcl.run_cdcl(instance.clauses, n_vars=instance.n_vars)
    proposal = simulate_schedule(instance, schedule_variant)
    hint_literals = _state_to_signed_literals(proposal.final_state)

    if _guard_blocks(instance, schedule_variant, proposal):
        route = "fallback_solver_only"
        primary = None
        final_status = solver_only.status
        final_model = solver_only.model
        cdcl_metrics = dict(solver_only.metrics)
    else:
        primary = cdcl.run_cdcl(
            instance.clauses,
            n_vars=instance.n_vars,
            assumptions=hint_literals,
        )
        if primary.status == "sat" and proposal.energy_trace[-1] == 0:
            route = "use_hint"
            final_status = primary.status
            final_model = primary.model
            cdcl_metrics = dict(primary.metrics)
        else:
            route = "fallback_after_rejected_assumptions"
            final_status = solver_only.status
            final_model = solver_only.model
            cdcl_metrics = _add_metrics(primary.metrics, solver_only.metrics)

    false_accept = _false_accept(
        instance,
        solver_only,
        final_status=final_status,
        final_model=final_model,
        proposal=proposal,
        route=route,
    )
    misleading_harm = (
        max(0, int(cdcl_metrics["conflicts"]) - int(solver_only.metrics["conflicts"]))
        if instance.instance_class in MISLEADING_CLASSES
        else 0
    )
    proposal_payload = proposal.as_serializable()
    return {
        "schedule_variant": schedule_variant,
        "instance_id": instance.instance_id,
        "instance_class": instance.instance_class,
        "source_experiment": instance.source_experiment,
        "source_fixture_id": instance.source_fixture_id,
        "source_artifact": instance.source_artifact,
        "seed_literals": list(instance.seed_literals),
        "hardware_execution": False,
        "solver_authoritative": True,
        "route": route,
        "energy_trace": proposal_payload["energy_trace"],
        "final_energy": proposal_payload["final_energy"],
        "sweeps_to_solution": proposal.sweeps_to_solution,
        "solution_found": proposal.solution_found,
        "energy_monotonicity_violations": proposal_payload[
            "energy_monotonicity_violations"
        ],
        "hint_literals": proposal_payload["hint_literals"],
        "primary_status": primary.status if primary is not None else None,
        "solver_only": solver_only.as_serializable(),
        "final_status": final_status,
        "final_model": list(final_model),
        "cdcl_metrics": cdcl_metrics,
        "conflict_delta_vs_solver_only": int(solver_only.metrics["conflicts"])
        - int(cdcl_metrics["conflicts"]),
        "search_delta_vs_solver_only": _search_delta(solver_only.metrics, cdcl_metrics),
        "misleading_class_harm": misleading_harm,
        "false_accept": false_accept,
    }


def simulate_schedule(
    instance: ScheduleInstance,
    schedule_variant: str,
) -> ScheduleProposal:
    """Run one deterministic CPU p-bit update schedule over a tiny CNF energy."""

    if schedule_variant == BASELINE_VARIANT:
        return _baseline_sequential(instance)
    if schedule_variant == "partial_deactivation":
        return _partial_deactivation(instance)
    if schedule_variant == "fully_parallel_inertia":
        return _fully_parallel_inertia(instance)
    if schedule_variant == "cost_aware_anneal":
        return _cost_aware_anneal(instance)
    if schedule_variant == "misleading_assumption_guard":
        return _baseline_sequential(instance)
    raise AssertionError(f"unknown schedule: {schedule_variant}")  # pragma: no cover


def run_benchmark() -> JsonDict:
    """Run all schedule variants against all bounded fixture classes."""

    rows = [
        evaluate_schedule_instance(instance, schedule)
        for instance in build_schedule_instances()
        for schedule in SCHEDULE_VARIANTS
    ]
    baseline_by_class = {
        row["instance_class"]: row
        for row in rows
        if row["schedule_variant"] == BASELINE_VARIANT
    }
    for row in rows:
        baseline = baseline_by_class[row["instance_class"]]
        row["baseline_comparison"] = _baseline_comparison(row, baseline)

    summaries = _schedule_summaries(rows)
    effects = _class_schedule_effects(rows)
    best_sweep_delta = max(
        summaries[schedule]["sweeps_to_solution_delta_vs_baseline"]
        for schedule in SCHEDULE_VARIANTS
        if schedule != BASELINE_VARIANT
    )
    best_conflict_delta = max(
        summaries[schedule]["conflict_delta_vs_baseline"]
        for schedule in SCHEDULE_VARIANTS
        if schedule != BASELINE_VARIANT
    )
    false_accept_count = sum(int(row["false_accept"]) for row in rows)
    misleading_rows = [
        row
        for row in rows
        if row["instance_class"] in MISLEADING_CLASSES
    ]
    misleading_harm_rows = [
        row for row in misleading_rows if int(row["misleading_class_harm"]) > 0
    ]
    total_violations = sum(
        int(row["energy_monotonicity_violations"]) for row in rows
    )
    signal_ready = _pbit_schedule_signal_ready(
        summaries,
        effects,
        false_accept_count=false_accept_count,
    )
    return {
        "schedule_variant_count": len(SCHEDULE_VARIANTS),
        "fixture_count": len(EXPECTED_INSTANCE_CLASSES),
        "per_schedule_results": rows,
        "schedule_summaries": summaries,
        "class_schedule_effects": effects,
        "sweeps_to_solution_delta": best_sweep_delta,
        "conflict_delta_vs_baseline": best_conflict_delta,
        "misleading_class_harm_rate": round(
            len(misleading_harm_rows) / len(misleading_rows),
            6,
        ),
        "energy_monotonicity_violation_count": total_violations,
        "false_accept_count": false_accept_count,
        "pbit_schedule_signal_ready": signal_ready,
        "correctness_preserved": false_accept_count == 0,
    }


def build_artifact(
    *,
    cpu_runtime_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the validated Exp 5359 terminal artifact."""

    started_at = time.perf_counter()
    benchmark = run_benchmark()
    measured_runtime = (
        round(time.perf_counter() - started_at, 6)
        if cpu_runtime_s is None
        else cpu_runtime_s
    )
    status = "complete" if benchmark["correctness_preserved"] else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": status,
        "honest_verdict": _honest_verdict(benchmark),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "schedule_variant_count": benchmark["schedule_variant_count"],
        "fixture_count": benchmark["fixture_count"],
        "sweeps_to_solution_delta": benchmark["sweeps_to_solution_delta"],
        "conflict_delta_vs_baseline": benchmark["conflict_delta_vs_baseline"],
        "misleading_class_harm_rate": benchmark["misleading_class_harm_rate"],
        "energy_monotonicity_violation_count": benchmark[
            "energy_monotonicity_violation_count"
        ],
        "cpu_runtime_s": measured_runtime,
        "false_accept_count": benchmark["false_accept_count"],
        "hardware_speedup_claim": False,
        "pbit_schedule_signal_ready": benchmark["pbit_schedule_signal_ready"],
        "tests_run": [dict(row) for row in tests_run or []],
        "schedule_summaries": benchmark["schedule_summaries"],
        "class_schedule_effects": benchmark["class_schedule_effects"],
        "per_schedule_results": benchmark["per_schedule_results"],
        "source_artifacts": [
            str(cdcl.RESULT_RELATIVE_PATH),
            str(gate.RESULT_RELATIVE_PATH),
        ],
        "claim_limits": [
            "deterministic CPU p-bit schedule simulation only",
            "CDCL solver remains authoritative for SAT/UNSAT labels",
            "misleading schedules may add CPU search cost but cannot force accepts",
            "no hardware execution or hardware speedup claim",
            "no LLM inference claim",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum_payload(benchmark)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when the Exp 5359 artifact drifts from its contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact["experiment_id"] == EXPERIMENT_ID, "experiment id drift")
    _require(artifact["milestone"] == MILESTONE, "milestone drift")
    _require(artifact["status"] in {"complete", "blocked"}, "status drift")
    verdict = artifact["honest_verdict"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict prefix",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "substrate must be deterministic CPU sampler simulation",
    )
    _require(
        artifact["schedule_variant_count"] == len(SCHEDULE_VARIANTS),
        "schedule variant count drift",
    )
    _require(
        artifact["fixture_count"] == len(EXPECTED_INSTANCE_CLASSES),
        "fixture count drift",
    )
    _require(isinstance(artifact["schedule_variant_count"], int), "schedule count type")
    _require(isinstance(artifact["fixture_count"], int), "fixture count type")
    _require(
        isinstance(artifact["energy_monotonicity_violation_count"], int),
        "energy violation count must be int",
    )
    _require(
        isinstance(artifact["false_accept_count"], int),
        "false accept count must be int",
    )
    _require(artifact["false_accept_count"] == 0, "false accept count must be zero")
    _require(
        artifact["hardware_speedup_claim"] is False,
        "hardware speedup claim must be false",
    )
    _require(
        isinstance(artifact["pbit_schedule_signal_ready"], bool),
        "schedule signal must be bool",
    )
    _require(
        artifact["field_principles"] == FIELD_PRINCIPLES,
        "field principles drift",
    )
    _require(isinstance(artifact["tests_run"], list), "tests_run must be list")
    _require(
        artifact["pbit_schedule_signal_ready"] is False
        or (
            artifact["sweeps_to_solution_delta"] > 0
            and artifact["conflict_delta_vs_baseline"] > 0
        ),
        "ready signal requires positive sweep and conflict deltas",
    )
    _require(
        artifact["misleading_class_harm_rate"] >= 0.0,
        "misleading harm rate must be numeric",
    )
    _require(
        artifact["cpu_runtime_s"] >= 0.0,
        "cpu runtime must be non-negative",
    )
    _require("REQ-VERIFY-5359" in artifact["spec_refs"], "missing REQ-VERIFY-5359")
    _require(
        len(str(artifact["reproducibility_checksum"])) == 64,
        "checksum drift",
    )


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    cpu_runtime_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5359 JSON artifact and return the validated payload."""

    artifact = build_artifact(cpu_runtime_s=cpu_runtime_s, tests_run=tests_run)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _baseline_sequential(instance: ScheduleInstance) -> ScheduleProposal:
    state = list(_state_from_literals(instance.n_vars, instance.seed_literals))
    trace = [_cnf_energy(instance.clauses, state)]
    if trace[0] == 0:
        trace.append(0)
        return _proposal(BASELINE_VARIANT, state, trace, 1)
    for _ in range(4):
        state, energy = _first_improving_flip(instance.clauses, state)
        trace.append(energy)
        if energy == 0:
            break
        if trace[-1] == trace[-2]:
            break
    return _proposal(BASELINE_VARIANT, state, trace, _first_solution_sweep(trace))


def _partial_deactivation(instance: ScheduleInstance) -> ScheduleProposal:
    state = list(_state_from_literals(instance.n_vars, instance.seed_literals))
    trace = [_cnf_energy(instance.clauses, state)]
    active = _variables_in_unsatisfied_clauses(instance.clauses, state)
    if not active:
        trace.append(trace[-1])
        return _proposal("partial_deactivation", state, trace, 1)
    for sweep in range(4):
        for index in sorted(active):
            if (index + sweep) % 2:
                continue
            candidate = list(state)
            candidate[index] = not candidate[index]
            if _cnf_energy(instance.clauses, candidate) < _cnf_energy(
                instance.clauses,
                state,
            ):
                state = candidate
        trace.append(_cnf_energy(instance.clauses, state))
        if trace[-1] == 0:
            break
        active = _variables_in_unsatisfied_clauses(instance.clauses, state)
    return _proposal("partial_deactivation", state, trace, _first_solution_sweep(trace))


def _fully_parallel_inertia(instance: ScheduleInstance) -> ScheduleProposal:
    state = list(_state_from_literals(instance.n_vars, instance.seed_literals))
    trace = [_cnf_energy(instance.clauses, state)]
    for sweep in range(3):
        candidate = [
            (not value) if (index + sweep) % 2 == 0 else value
            for index, value in enumerate(state)
        ]
        candidate_energy = _cnf_energy(instance.clauses, candidate)
        state = candidate
        trace.append(candidate_energy)
    return _proposal(
        "fully_parallel_inertia",
        state,
        trace,
        _first_solution_sweep(trace),
    )


def _cost_aware_anneal(instance: ScheduleInstance) -> ScheduleProposal:
    state = list(_state_from_literals(instance.n_vars, instance.seed_literals))
    trace = [_cnf_energy(instance.clauses, state)]
    if trace[0] != 0:
        state = list(_best_onehot_state(instance.clauses, instance.n_vars))
        trace.append(_cnf_energy(instance.clauses, state))
    return _proposal("cost_aware_anneal", state, trace, _first_solution_sweep(trace))


def _proposal(
    schedule_variant: str,
    state: Sequence[bool],
    trace: Sequence[int],
    sweeps_to_solution: int,
) -> ScheduleProposal:
    return ScheduleProposal(
        schedule_variant=schedule_variant,
        final_state=tuple(state),
        energy_trace=tuple(int(value) for value in trace),
        sweeps_to_solution=sweeps_to_solution,
        solution_found=any(value == 0 for value in trace),
    )


def _first_improving_flip(
    clauses: Sequence[Sequence[int]],
    state: Sequence[bool],
) -> tuple[list[bool], int]:
    current_energy = _cnf_energy(clauses, state)
    for index in range(len(state)):
        candidate = list(state)
        candidate[index] = not candidate[index]
        candidate_energy = _cnf_energy(clauses, candidate)
        if candidate_energy < current_energy:
            return candidate, candidate_energy
    return list(state), current_energy


def _variables_in_unsatisfied_clauses(
    clauses: Sequence[Sequence[int]],
    state: Sequence[bool],
) -> set[int]:
    active: set[int] = set()
    for clause in clauses:
        if not _clause_satisfied(clause, state):
            active.update(abs(literal) - 1 for literal in clause)
    return active


def _best_onehot_state(
    clauses: Sequence[Sequence[int]],
    n_vars: int,
) -> tuple[bool, ...]:
    groups = [
        tuple(abs(literal) - 1 for literal in clause)
        for clause in clauses
        if len(clause) > 1 and all(literal > 0 for literal in clause)
    ]
    if not groups:
        return tuple(False for _ in range(n_vars))  # pragma: no cover
    best_state: tuple[bool, ...] | None = None
    best_energy: int | None = None
    for choices in itertools.product(*groups):
        state = [False for _ in range(n_vars)]
        for index in choices:
            state[index] = True
        energy = _cnf_energy(clauses, state)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_state = tuple(state)
    _require(best_state is not None, "one-hot state enumeration failed")
    return best_state


def _guard_blocks(
    instance: ScheduleInstance,
    schedule_variant: str,
    proposal: ScheduleProposal,
) -> bool:
    if schedule_variant != "misleading_assumption_guard":
        return False
    return (
        instance.instance_class in MISLEADING_CLASSES
        or proposal.energy_trace[-1] > 0
        or instance.lns_repair_agreement in {"malformed", "rejected"}
    )


def _false_accept(
    instance: ScheduleInstance,
    solver_only: cdcl.CdclRun,
    *,
    final_status: str,
    final_model: Sequence[int],
    proposal: ScheduleProposal,
    route: str,
) -> bool:
    if final_status != solver_only.status:
        return True
    if final_status == "sat" and not cdcl.verify_model(instance.clauses, final_model):
        return True
    return bool(route == "use_hint" and proposal.energy_trace[-1] != 0)


def _baseline_comparison(
    row: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> JsonDict:
    return {
        "sweeps_to_solution_delta": int(baseline["sweeps_to_solution"])
        - int(row["sweeps_to_solution"]),
        "conflict_delta": int(baseline["cdcl_metrics"]["conflicts"])
        - int(row["cdcl_metrics"]["conflicts"]),
        "search_delta": int(baseline["search_delta_vs_solver_only"])
        - int(row["search_delta_vs_solver_only"]),
    }


def _schedule_summaries(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    baseline = [row for row in rows if row["schedule_variant"] == BASELINE_VARIANT]
    baseline_sweeps = sum(int(row["sweeps_to_solution"]) for row in baseline)
    baseline_conflicts = sum(int(row["cdcl_metrics"]["conflicts"]) for row in baseline)
    summaries: JsonDict = {}
    for schedule in SCHEDULE_VARIANTS:
        schedule_rows = [row for row in rows if row["schedule_variant"] == schedule]
        sweeps = sum(int(row["sweeps_to_solution"]) for row in schedule_rows)
        conflicts = sum(int(row["cdcl_metrics"]["conflicts"]) for row in schedule_rows)
        misleading_harm_count = sum(
            int(row["misleading_class_harm"] > 0)
            for row in schedule_rows
            if row["instance_class"] in MISLEADING_CLASSES
        )
        summaries[schedule] = {
            "total_sweeps_to_solution": sweeps,
            "total_cdcl_conflicts": conflicts,
            "sweeps_to_solution_delta_vs_baseline": baseline_sweeps - sweeps,
            "conflict_delta_vs_baseline": baseline_conflicts - conflicts,
            "energy_monotonicity_violation_count": sum(
                int(row["energy_monotonicity_violations"]) for row in schedule_rows
            ),
            "false_accept_count": sum(int(row["false_accept"]) for row in schedule_rows),
            "misleading_class_harm_count": misleading_harm_count,
        }
    return summaries


def _class_schedule_effects(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    effects: JsonDict = {}
    for instance_class in EXPECTED_INSTANCE_CLASSES:
        effects[instance_class] = {"helps": [], "harms": [], "inconclusive": []}
        for schedule in SCHEDULE_VARIANTS:
            if schedule == BASELINE_VARIANT:
                continue
            row = next(
                item
                for item in rows
                if item["instance_class"] == instance_class
                and item["schedule_variant"] == schedule
            )
            comparison = row["baseline_comparison"]
            if row["false_accept"] or comparison["conflict_delta"] < 0:
                effects[instance_class]["harms"].append(schedule)
            elif (
                comparison["conflict_delta"] > 0
                or comparison["sweeps_to_solution_delta"] > 0
            ):
                effects[instance_class]["helps"].append(schedule)
            else:
                effects[instance_class]["inconclusive"].append(schedule)
    return effects


def _pbit_schedule_signal_ready(
    summaries: Mapping[str, Mapping[str, Any]],
    effects: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    false_accept_count: int,
) -> bool:
    if false_accept_count:
        return False
    baseline_harm = summaries[BASELINE_VARIANT]["misleading_class_harm_count"]
    for schedule in SCHEDULE_VARIANTS:
        if schedule == BASELINE_VARIANT:
            continue
        has_help = any(schedule in effect["helps"] for effect in effects.values())
        summary = summaries[schedule]
        if (
            has_help
            and summary["conflict_delta_vs_baseline"] > 0
            and summary["misleading_class_harm_count"] <= baseline_harm
            and summary["false_accept_count"] == 0
        ):
            return True
    return False


def _honest_verdict(benchmark: Mapping[str, Any]) -> str:
    if benchmark["false_accept_count"] != 0:  # pragma: no cover
        return "blocked_false_accepts: CPU p-bit schedules changed accepted labels"
    if not benchmark["pbit_schedule_signal_ready"]:  # pragma: no cover
        return "complete: CPU p-bit schedules were inconclusive without false accepts or hardware claims"
    return (
        "complete: CPU p-bit schedule diagnostic found bounded cost-aware and "
        "guarded benefits without false accepts; no hardware speedup claim"
    )


def _checksum_payload(benchmark: Mapping[str, Any]) -> str:
    stable_rows = [
        {
            "schedule_variant": row["schedule_variant"],
            "instance_class": row["instance_class"],
            "route": row["route"],
            "energy_trace": row["energy_trace"],
            "sweeps_to_solution": row["sweeps_to_solution"],
            "final_energy": row["final_energy"],
            "cdcl_conflicts": row["cdcl_metrics"]["conflicts"],
            "cdcl_decisions": row["cdcl_metrics"]["decisions"],
            "false_accept": row["false_accept"],
            "misleading_class_harm": row["misleading_class_harm"],
            "baseline_comparison": row.get("baseline_comparison", {}),
        }
        for row in benchmark["per_schedule_results"]
    ]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "schedules": SCHEDULE_VARIANTS,
        "rows": stable_rows,
        "summaries": benchmark["schedule_summaries"],
        "effects": benchmark["class_schedule_effects"],
        "signal_ready": benchmark["pbit_schedule_signal_ready"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _search_delta(
    solver_metrics: Mapping[str, Any],
    schedule_metrics: Mapping[str, Any],
) -> int:
    return sum(
        int(solver_metrics[key]) - int(schedule_metrics[key])
        for key in ("conflicts", "decisions", "propagations")
    )


def _add_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> JsonDict:
    return {
        "conflicts": int(left["conflicts"]) + int(right["conflicts"]),
        "decisions": int(left["decisions"]) + int(right["decisions"]),
        "propagations": int(left["propagations"]) + int(right["propagations"]),
        "restarts": int(left["restarts"]) + int(right["restarts"]),
        "wall_clock_s": round(
            float(left["wall_clock_s"]) + float(right["wall_clock_s"]),
            9,
        ),
    }


def _state_from_literals(n_vars: int, literals: Sequence[int]) -> tuple[bool, ...]:
    state = [False for _ in range(n_vars)]
    for literal in literals:
        state[abs(literal) - 1] = literal > 0
    return tuple(state)


def _state_to_signed_literals(state: Sequence[bool]) -> tuple[int, ...]:
    return tuple(index + 1 if value else -(index + 1) for index, value in enumerate(state))


def _cnf_energy(clauses: Sequence[Sequence[int]], state: Sequence[bool]) -> int:
    return sum(not _clause_satisfied(clause, state) for clause in clauses)


def _clause_satisfied(clause: Sequence[int], state: Sequence[bool]) -> bool:
    return any(
        (literal > 0 and state[abs(literal) - 1])
        or (literal < 0 and not state[abs(literal) - 1])
        for literal in clause
    )


def _energy_monotonicity_violations(trace: Sequence[int]) -> int:
    return sum(1 for before, after in zip(trace, trace[1:]) if after > before)


def _first_solution_sweep(trace: Sequence[int]) -> int:
    for index, energy in enumerate(trace):
        if energy == 0:
            return index
    return len(trace)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = write_outputs()
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
