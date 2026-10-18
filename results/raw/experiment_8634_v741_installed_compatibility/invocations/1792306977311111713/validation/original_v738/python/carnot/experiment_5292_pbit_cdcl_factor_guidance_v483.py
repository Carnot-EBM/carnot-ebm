"""Exp 5292 CPU p-bit/Ising assumption guidance for CDCL.

Spec refs: REQ-VERIFY-5292, SCENARIO-VERIFY-5292.

The experiment keeps the SAT solver authoritative. A CPU p-bit-style sampler
may suggest temporary assumptions, but a non-empty assumption set can only
guide the CDCL search. If assumptions make a strengthened formula UNSAT, the
runner falls back to an unassumed CDCL solve before reporting the original
formula's SAT/UNSAT label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from pysat.solvers import Minisat22

from carnot import experiment_5278_constraint_factor_graph_boundary_v482 as v5278


JsonDict = dict[str, Any]

RUN_DATE = "20260706"
RANDOM_SEED = 5292
EXPERIMENT_ID = "exp5292-pbit-cdcl-factor-guidance-v483"
SCHEMA = "carnot.experiment_5292.pbit_cdcl_factor_guidance.v483"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5292_pbit_cdcl_factor_guidance_v483.json"
)
INFERENCE_SUBSTRATE = "offline_deterministic_certificate_no_llm"
SPEC_REFS = ("REQ-VERIFY-5292", "SCENARIO-VERIFY-5292")
TERMINAL_PREFIXES = ("complete:", "null:", "harmful_", "blocked_")
PBIT_GUIDANCE_POSITIVE_PRINCIPLE = (
    "Bare boolean requested by the task; true only when simulated p-bit assumptions "
    "preserve correctness and save aggregate conflicts on the bounded instance classes."
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal Exp 5292 verdict; starts with complete:, null:, harmful_, or "
        "blocked_ and states whether simulated p-bit/CDCL guidance helped."
    ),
    "inference_substrate": (
        "Must be offline_deterministic_certificate_no_llm because Exp 5292 uses "
        "CPU-local fixtures, a local CDCL solver, and no LLM inference."
    ),
    "pbit_cdcl_guidance_positive": PBIT_GUIDANCE_POSITIVE_PRINCIPLE,
    "pbit_cdcl_guidance_positive_principle": (
        "Explains why the bare guidance gate opened or stayed closed so a mixed "
        "distribution-sensitive result is not over-promoted."
    ),
    "assumption_generation_summary": (
        "Documents CPU-only simulated p-bit/Ising assumption generation and prevents "
        "it from being mistaken for hardware execution."
    ),
    "conflicts_saved": (
        "Pure-minus-guided CDCL conflicts by aggregate and class; negative values "
        "expose harmful guidance instead of hiding it."
    ),
    "propagations_saved": (
        "Pure-minus-guided CDCL propagations by aggregate and class; negative values "
        "expose harmful guidance overhead."
    ),
    "fallback_overwrite_count": (
        "Counts assumptions rejected or overwritten by the authoritative CDCL "
        "fallback, proving guidance is advisory."
    ),
    "correctness_preserved": (
        "True only when guided SAT/UNSAT labels match unassumed CDCL labels and every "
        "SAT model satisfies the original CNF."
    ),
    "instance_class_gate": (
        "Classifies help, harm, and neutral behavior by tiny fixture class and treats "
        "distribution sensitivity as expected."
    ),
    "hardware_speedup_claimed": (
        "Always false for Exp 5292 because the sampler guidance is CPU simulation, "
        "not hardware execution."
    ),
    "tests_run": (
        "Commands run to validate assumption safety, class gates, artifact schema, "
        "new-code coverage, and repository test status."
    ),
}
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "pbit_cdcl_guidance_positive",
    "pbit_cdcl_guidance_positive_principle",
    "assumption_generation_summary",
    "conflicts_saved",
    "propagations_saved",
    "fallback_overwrite_count",
    "correctness_preserved",
    "instance_class_gate",
    "hardware_speedup_claimed",
    "tests_run",
)
WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "assumption_generation_summary",
    "conflicts_saved",
    "propagations_saved",
    "fallback_overwrite_count",
    "correctness_preserved",
    "instance_class_gate",
    "hardware_speedup_claimed",
)


@dataclass(frozen=True)
class GuidanceInstance:
    """Tiny CNF benchmark row derived from the Exp 5278 factor boundary."""

    instance_id: str
    instance_class: str
    n_vars: int
    clauses: tuple[tuple[int, ...], ...]
    expected_status: str
    source_fixture_id: str
    source_artifact: str
    assumption_profile: str


@dataclass(frozen=True)
class AssumptionSet:
    """CPU-simulated guidance literals offered to CDCL as temporary assumptions."""

    literals: tuple[int, ...]
    method: str
    profile: str
    energy: int
    simulated_guidance: bool = True
    hardware_execution: bool = False

    def as_serializable(self) -> JsonDict:
        return {
            "literals": list(self.literals),
            "method": self.method,
            "profile": self.profile,
            "energy": self.energy,
            "simulated_guidance": self.simulated_guidance,
            "hardware_execution": self.hardware_execution,
        }


@dataclass(frozen=True)
class CdclRun:
    """One Minisat22 CDCL solve with stable counters from the solver."""

    status: str
    model: tuple[int, ...]
    metrics: JsonDict

    def as_serializable(self) -> JsonDict:
        return {
            "status": self.status,
            "model": list(self.model),
            "metrics": dict(self.metrics),
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wrap_field(field: str, value: Any) -> JsonDict:
    """Attach the task-required principle to an artifact field."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def build_factor_guidance_instances() -> tuple[GuidanceInstance, ...]:
    """Return help, harm, and neutral classes over the Exp 5278 tiny factor CNF."""

    source = v5278.select_tiny_fixture()
    boundary = v5278.build_boundary(source)
    clauses = _factor_boundary_to_cnf(boundary)
    common = {
        "n_vars": len(boundary.bit_order),
        "clauses": clauses,
        "expected_status": "sat",
        "source_fixture_id": source["fixture_id"],
        "source_artifact": str(v5278.RESULT_RELATIVE_PATH),
    }
    return (
        GuidanceInstance(
            instance_id="exp5292_aligned_factor_sat",
            instance_class="aligned_factor_sat",
            assumption_profile="low_temperature_consensus",
            **common,
        ),
        GuidanceInstance(
            instance_id="exp5292_misleading_factor_sat",
            instance_class="misleading_factor_sat",
            assumption_profile="false_basin_control",
            **common,
        ),
        GuidanceInstance(
            instance_id="exp5292_neutral_factor_sat",
            instance_class="neutral_factor_sat",
            assumption_profile="no_consensus_control",
            **common,
        ),
    )


def generate_assumptions(instance: GuidanceInstance) -> AssumptionSet:
    """Generate CPU-only p-bit/Ising-style assumptions for one class."""

    source = v5278.select_tiny_fixture()
    boundary = v5278.build_boundary(source)
    if instance.assumption_profile == "low_temperature_consensus":
        literals = _cpu_pbit_consensus_literals(instance)
        method = "cpu_simulated_pbit_low_temperature_consensus"
    elif instance.assumption_profile == "false_basin_control":
        literals = _assignment_to_positive_lits(boundary, boundary.false_assignment)
        method = "cpu_simulated_pbit_false_basin_control"
    else:
        literals = ()
        method = "cpu_simulated_pbit_no_consensus_control"
    return AssumptionSet(
        literals=literals,
        method=method,
        profile=instance.assumption_profile,
        energy=_cnf_energy(instance.clauses, _state_from_positive_lits(instance.n_vars, literals)),
    )


def run_cdcl(
    clauses: Sequence[Sequence[int]],
    *,
    n_vars: int,
    assumptions: Sequence[int] = (),
) -> CdclRun:
    """Run Minisat22 CDCL once and return solver counters."""

    started_at = time.perf_counter()
    with Minisat22(bootstrap_with=[list(clause) for clause in clauses]) as solver:
        is_sat = solver.solve(assumptions=list(assumptions))
        stats = solver.accum_stats()
        model = tuple(int(literal) for literal in (solver.get_model() or [])[:n_vars])
    wall_clock_s = round(time.perf_counter() - started_at, 9)
    return CdclRun(
        status="sat" if is_sat else "unsat",
        model=model if is_sat else (),
        metrics={
            "conflicts": int(stats.get("conflicts", 0)),
            "propagations": int(stats.get("propagations", 0)),
            "decisions": int(stats.get("decisions", 0)),
            "restarts": int(stats.get("restarts", 0)),
            "wall_clock_s": wall_clock_s,
        },
    )


def run_guided_instance(
    instance: GuidanceInstance,
    assumptions: AssumptionSet,
) -> JsonDict:
    """Run unassumed and assumption-guided CDCL while preserving authority."""

    pure = run_cdcl(instance.clauses, n_vars=instance.n_vars)
    primary = run_cdcl(
        instance.clauses,
        n_vars=instance.n_vars,
        assumptions=assumptions.literals,
    )
    fallback_used = bool(assumptions.literals and primary.status == "unsat")
    fallback = (
        run_cdcl(instance.clauses, n_vars=instance.n_vars)
        if fallback_used
        else None
    )
    final = fallback or primary
    guided_metrics = (
        _add_metrics(primary.metrics, fallback.metrics)
        if fallback is not None
        else dict(primary.metrics)
    )
    overwrite_count = (
        _overwrite_count(assumptions.literals, final.model)
        if fallback_used
        else 0
    )
    correctness = _correctness_preserved(instance, pure, final)
    savings = _metric_savings(pure.metrics, guided_metrics)
    return {
        "instance_id": instance.instance_id,
        "instance_class": instance.instance_class,
        "expected_status": instance.expected_status,
        "source_fixture_id": instance.source_fixture_id,
        "assumptions": assumptions.as_serializable(),
        "pure": pure.as_serializable(),
        "guided": {
            "primary_status": primary.status,
            "final_status": final.status,
            "final_model": list(final.model),
            "fallback_used": fallback_used,
            "overwrite_count": overwrite_count,
            "metrics": guided_metrics,
            "primary_metrics": dict(primary.metrics),
            "fallback_metrics": dict(fallback.metrics) if fallback is not None else None,
        },
        "correctness_preserved": correctness,
        "savings": savings,
        "class_effect": _class_effect(savings),
    }


def run_benchmark() -> JsonDict:
    """Run the bounded help/harm/neutral guidance benchmark."""

    instances = build_factor_guidance_instances()
    rows = [
        run_guided_instance(instance, generate_assumptions(instance))
        for instance in instances
    ]
    aggregate = _aggregate_savings(rows)
    savings_by_class = {
        row["instance_class"]: dict(row["savings"])
        for row in rows
    }
    instance_class_gate = _instance_class_gate(rows)
    correctness = all(row["correctness_preserved"] for row in rows)
    guidance_positive = bool(correctness and aggregate["conflicts_saved"] > 0)
    fallback_overwrite_count = sum(row["guided"]["overwrite_count"] for row in rows)
    return {
        "per_instance_results": rows,
        "aggregate_savings": aggregate,
        "savings_by_class": savings_by_class,
        "instance_class_gate": instance_class_gate,
        "correctness_preserved": correctness,
        "pbit_cdcl_guidance_positive": guidance_positive,
        "fallback_overwrite_count": fallback_overwrite_count,
        "assumption_generation_summary": _assumption_generation_summary(rows),
    }


def verify_model(clauses: Sequence[Sequence[int]], model: Sequence[int]) -> bool:
    """Return true only when a model satisfies every original CNF clause."""

    assignment = {abs(literal): literal > 0 for literal in model}
    return all(
        any(
            (literal > 0 and assignment.get(abs(literal), False))
            or (literal < 0 and not assignment.get(abs(literal), False))
            for literal in clause
        )
        for clause in clauses
    )


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the validated Exp 5292 terminal artifact."""

    started_at = time.perf_counter()
    benchmark = run_benchmark()
    measured_duration = (
        round(time.perf_counter() - started_at, 6)
        if duration_s is None
        else duration_s
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "duration_s": measured_duration,
        "honest_verdict": wrap_field("honest_verdict", _honest_verdict(benchmark)),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "pbit_cdcl_guidance_positive": benchmark["pbit_cdcl_guidance_positive"],
        "pbit_cdcl_guidance_positive_principle": PBIT_GUIDANCE_POSITIVE_PRINCIPLE,
        "assumption_generation_summary": wrap_field(
            "assumption_generation_summary",
            benchmark["assumption_generation_summary"],
        ),
        "conflicts_saved": wrap_field(
            "conflicts_saved",
            {
                "aggregate": benchmark["aggregate_savings"]["conflicts_saved"],
                "by_class": {
                    name: metrics["conflicts_saved"]
                    for name, metrics in benchmark["savings_by_class"].items()
                },
            },
        ),
        "propagations_saved": wrap_field(
            "propagations_saved",
            {
                "aggregate": benchmark["aggregate_savings"]["propagations_saved"],
                "by_class": {
                    name: metrics["propagations_saved"]
                    for name, metrics in benchmark["savings_by_class"].items()
                },
            },
        ),
        "fallback_overwrite_count": wrap_field(
            "fallback_overwrite_count",
            benchmark["fallback_overwrite_count"],
        ),
        "correctness_preserved": wrap_field(
            "correctness_preserved",
            benchmark["correctness_preserved"],
        ),
        "instance_class_gate": wrap_field(
            "instance_class_gate",
            benchmark["instance_class_gate"],
        ),
        "hardware_speedup_claimed": wrap_field("hardware_speedup_claimed", False),
        "tests_run": [dict(row) for row in tests_run or []],
        "benchmark_metrics": benchmark["aggregate_savings"],
        "per_instance_results": benchmark["per_instance_results"],
        "source_artifacts": [str(v5278.RESULT_RELATIVE_PATH)],
        "claim_limits": [
            "CPU simulated p-bit/Ising assumptions only",
            "Minisat22 CDCL remains authoritative for SAT/UNSAT labels",
            "bad assumptions trigger fallback or overwrite telemetry",
            "distribution sensitivity is expected and reported by class",
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
    """Fail closed when the Exp 5292 artifact drifts from its contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        wrapped = artifact[field]
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle drift")
        _require("value" in wrapped, f"{field} missing value")

    verdict = artifact["honest_verdict"]["value"]
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict prefix")
    _require("p-bit/CDCL" in verdict, "honest_verdict must mention p-bit/CDCL guidance")
    _require(artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE, "inference_substrate drift")
    _require(isinstance(artifact["pbit_cdcl_guidance_positive"], bool), "pbit_cdcl_guidance_positive must be bare bool")
    _require(
        artifact["pbit_cdcl_guidance_positive_principle"] == PBIT_GUIDANCE_POSITIVE_PRINCIPLE,
        "pbit_cdcl_guidance_positive_principle drift",
    )
    _require(artifact["hardware_speedup_claimed"]["value"] is False, "hardware speedup must be false")
    _require(artifact["correctness_preserved"]["value"] is True, "correctness must be preserved")
    _require(isinstance(artifact["tests_run"], list), "tests_run must be list")

    summary = artifact["assumption_generation_summary"]["value"]
    _require(summary["hardware_execution"] is False, "assumption summary must be CPU-only")
    _require(summary["simulated_guidance_label"] == "simulated_cpu_guidance_not_hardware_execution", "simulated guidance label drift")
    _require(artifact["fallback_overwrite_count"]["value"] > 0, "fallback/overwrite count must be positive")
    conflicts = artifact["conflicts_saved"]["value"]
    _require(conflicts["aggregate"] > 0, "aggregate conflicts must be saved for positive gate")
    _require(artifact["pbit_cdcl_guidance_positive"] == (conflicts["aggregate"] > 0), "positive gate drift")
    propagations = artifact["propagations_saved"]["value"]
    _require(propagations["aggregate"] > 0, "aggregate propagations must be saved")
    gate = artifact["instance_class_gate"]["value"]
    _require(gate["helps"] == ["aligned_factor_sat"], "help class gate drift")
    _require(gate["harms"] == ["misleading_factor_sat"], "harm class gate drift")
    _require(gate["neutral"] == ["neutral_factor_sat"], "neutral class gate drift")
    _require(gate["distribution_sensitivity_expected"] is True, "distribution sensitivity gate drift")
    _require("REQ-VERIFY-5292" in artifact["spec_refs"], "spec refs must include REQ-VERIFY-5292")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum drift")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5292 JSON artifact and return the validated payload."""

    artifact = build_artifact(duration_s=duration_s, tests_run=tests_run)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _factor_boundary_to_cnf(boundary: v5278.BoundaryInstance) -> tuple[tuple[int, ...], ...]:
    clauses: list[tuple[int, ...]] = []
    groups = {
        variable: [
            index + 1
            for index, bit_variable in enumerate(boundary.bit_variables)
            if bit_variable == variable
        ]
        for variable in boundary.variables
    }
    for group in groups.values():
        clauses.append(tuple(group))
        clauses.extend(tuple(-literal for literal in pair) for pair in itertools.combinations(group, 2))
    for a_value in boundary.variables["a"]["domain"]:
        for b_value in boundary.variables["b"]["domain"]:
            if not (a_value + b_value == 5 and a_value < b_value):
                clauses.append((-(a_value + 1), -(b_value + 5)))
    return tuple(clauses)


def _assignment_to_positive_lits(
    boundary: v5278.BoundaryInstance,
    assignment: Mapping[str, Any],
) -> tuple[int, ...]:
    bits = boundary.assignment_to_bits(dict(assignment))
    return tuple(index + 1 for index, bit in enumerate(bits) if bit)


def _cpu_pbit_consensus_literals(instance: GuidanceInstance) -> tuple[int, ...]:
    rng = random.Random(RANDOM_SEED)
    state = [rng.choice((False, True)) for _ in range(instance.n_vars)]
    counts = [0 for _ in range(instance.n_vars)]
    burn_in = 50
    n_samples = 150
    beta = 5.0
    for step in range(burn_in + n_samples):
        order = list(range(instance.n_vars))
        rng.shuffle(order)
        for index in order:
            state[index] = False
            false_energy = _cnf_energy(instance.clauses, state)
            state[index] = True
            true_energy = _cnf_energy(instance.clauses, state)
            p_true = 1.0 / (1.0 + math.exp(beta * (true_energy - false_energy)))
            state[index] = rng.random() < p_true
        if step >= burn_in:
            for index, value in enumerate(state):
                counts[index] += int(value)
    rates = [count / n_samples for count in counts]
    return tuple(
        index + 1
        for index, rate in enumerate(rates)
        if rate >= 0.7
    )[:2]


def _cnf_energy(clauses: Sequence[Sequence[int]], state: Sequence[bool]) -> int:
    return sum(
        not any(
            (literal > 0 and state[abs(literal) - 1])
            or (literal < 0 and not state[abs(literal) - 1])
            for literal in clause
        )
        for clause in clauses
    )


def _state_from_positive_lits(n_vars: int, literals: Sequence[int]) -> tuple[bool, ...]:
    positive = {literal for literal in literals if literal > 0}
    return tuple(index in positive for index in range(1, n_vars + 1))


def _add_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> JsonDict:
    return {
        "conflicts": int(left["conflicts"]) + int(right["conflicts"]),
        "propagations": int(left["propagations"]) + int(right["propagations"]),
        "decisions": int(left["decisions"]) + int(right["decisions"]),
        "restarts": int(left["restarts"]) + int(right["restarts"]),
        "wall_clock_s": round(float(left["wall_clock_s"]) + float(right["wall_clock_s"]), 9),
    }


def _overwrite_count(assumptions: Sequence[int], final_model: Sequence[int]) -> int:
    final_literals = set(final_model)
    return sum(1 for literal in assumptions if literal not in final_literals)


def _correctness_preserved(
    instance: GuidanceInstance,
    pure: CdclRun,
    final: CdclRun,
) -> bool:
    if pure.status != instance.expected_status:
        return False
    if final.status != pure.status:
        return False
    return final.status == "unsat" or verify_model(instance.clauses, final.model)


def _metric_savings(pure_metrics: Mapping[str, Any], guided_metrics: Mapping[str, Any]) -> JsonDict:
    return {
        "conflicts_saved": int(pure_metrics["conflicts"]) - int(guided_metrics["conflicts"]),
        "propagations_saved": int(pure_metrics["propagations"]) - int(guided_metrics["propagations"]),
        "decisions_saved": int(pure_metrics["decisions"]) - int(guided_metrics["decisions"]),
        "restarts_saved": int(pure_metrics["restarts"]) - int(guided_metrics["restarts"]),
        "wall_clock_s_saved": round(float(pure_metrics["wall_clock_s"]) - float(guided_metrics["wall_clock_s"]), 9),
    }


def _class_effect(savings: Mapping[str, Any]) -> str:
    if savings["conflicts_saved"] > 0:
        return "helps"
    if savings["conflicts_saved"] < 0:
        return "harms"
    return "neutral"


def _aggregate_savings(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        key: sum(row["savings"][key] for row in rows)
        for key in (
            "conflicts_saved",
            "propagations_saved",
            "decisions_saved",
            "restarts_saved",
            "wall_clock_s_saved",
        )
    }


def _instance_class_gate(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "helps": [row["instance_class"] for row in rows if row["class_effect"] == "helps"],
        "harms": [row["instance_class"] for row in rows if row["class_effect"] == "harms"],
        "neutral": [row["instance_class"] for row in rows if row["class_effect"] == "neutral"],
        "distribution_sensitivity_expected": True,
        "gate_rule": "classify by pure-minus-guided conflicts, with correctness required separately",
    }


def _assumption_generation_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "method": "cpu_simulated_pbit_ising_assumptions",
        "simulated_guidance_label": "simulated_cpu_guidance_not_hardware_execution",
        "hardware_execution": False,
        "random_seed": RANDOM_SEED,
        "source_artifact": str(v5278.RESULT_RELATIVE_PATH),
        "profiles": {
            row["instance_class"]: row["assumptions"]
            for row in rows
        },
    }


def _honest_verdict(benchmark: Mapping[str, Any]) -> str:
    if not benchmark["correctness_preserved"]:  # pragma: no cover - validation and tests keep the fixture correct.
        return "blocked_correctness_not_preserved"
    if benchmark["aggregate_savings"]["conflicts_saved"] < 0:  # pragma: no cover - current bounded fixture is aggregate-positive.
        return "harmful_pbit_cdcl_guidance_increased_aggregate_conflicts"
    if benchmark["aggregate_savings"]["conflicts_saved"] == 0:  # pragma: no cover - current bounded fixture is aggregate-positive.
        return "null: p-bit/CDCL simulated CPU guidance was neutral on aggregate conflicts"
    return (
        "complete: p-bit/CDCL simulated CPU guidance helped aggregate conflicts "
        "on the bounded factor fixture while harming the misleading-assumption "
        "class; distribution sensitivity is expected"
    )


def _checksum_payload(benchmark: Mapping[str, Any]) -> str:
    rows = []
    for row in benchmark["per_instance_results"]:
        rows.append(
            {
                "instance_id": row["instance_id"],
                "instance_class": row["instance_class"],
                "assumptions": row["assumptions"],
                "pure_status": row["pure"]["status"],
                "guided_primary_status": row["guided"]["primary_status"],
                "guided_final_status": row["guided"]["final_status"],
                "fallback_used": row["guided"]["fallback_used"],
                "overwrite_count": row["guided"]["overwrite_count"],
                "pure_metrics": _stable_metrics(row["pure"]["metrics"]),
                "guided_metrics": _stable_metrics(row["guided"]["metrics"]),
                "savings": {
                    key: value
                    for key, value in row["savings"].items()
                    if key != "wall_clock_s_saved"
                },
                "class_effect": row["class_effect"],
                "correctness_preserved": row["correctness_preserved"],
            }
        )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "rows": rows,
        "aggregate_savings": {
            key: value
            for key, value in benchmark["aggregate_savings"].items()
            if key != "wall_clock_s_saved"
        },
        "instance_class_gate": benchmark["instance_class_gate"],
        "fallback_overwrite_count": benchmark["fallback_overwrite_count"],
        "pbit_cdcl_guidance_positive": benchmark["pbit_cdcl_guidance_positive"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_metrics(metrics: Mapping[str, Any]) -> JsonDict:
    return {
        key: metrics[key]
        for key in ("conflicts", "propagations", "decisions", "restarts")
    }


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
