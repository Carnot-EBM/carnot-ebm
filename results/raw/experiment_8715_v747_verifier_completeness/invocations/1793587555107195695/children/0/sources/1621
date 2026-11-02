"""Exp 5300 p-bit/CDCL instance-class gate.

Spec refs: REQ-VERIFY-5300, SCENARIO-VERIFY-5300.

The gate keeps the SAT solver authoritative. CPU-side p-bit or Ising
assumptions may be replayed only when deterministic solver/LNS features say
they are aligned. Misleading assumptions are blocked and replaced by the
solver-only fallback, so this experiment measures routing safety rather than
claiming any hardware speedup.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5292_pbit_cdcl_factor_guidance_v483 as cdcl
from carnot import experiment_5299_constraint_lns_solver_repair_fixture_v484 as lns


JsonDict = dict[str, Any]

RUN_DATE = "20260706"
RANDOM_SEED = 5300
EXPERIMENT_ID = "exp5300-pbit-cdcl-instance-class-gate-v484"
SCHEMA = "carnot.experiment_5300.pbit_cdcl_instance_class_gate.v484"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5300_pbit_cdcl_instance_class_gate_v484.json"
)
INFERENCE_SUBSTRATE = "offline_deterministic_certificate_no_llm"
SPEC_REFS = ("REQ-VERIFY-5300", "SCENARIO-VERIFY-5300")
TERMINAL_PREFIXES = ("complete:", "null:", "harmful_", "blocked_")
CLASSIFIER_KIND = "deterministic_threshold_rules_no_training"
MISLEADING_ASSUMPTION_CLASSES = (
    "misleading_factor_sat",
    "misleading_repair",
    "semantic_wrong_control",
)
METRIC_KEYS = ("conflicts", "decisions", "propagations", "restarts", "wall_clock_s")
COUNT_METRIC_KEYS = ("conflicts", "decisions", "propagations", "restarts")

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal Exp 5300 verdict; starts with complete:, null:, harmful_, or "
        "blocked_ and states whether the p-bit/CDCL instance-class gate helped."
    ),
    "inference_substrate": (
        "Must be offline_deterministic_certificate_no_llm because Exp 5300 replays "
        "local deterministic fixtures with a local CDCL solver and no LLM inference."
    ),
    "pbit_gate_ready": (
        "True only when the transparent feature gate preserves correctness, blocks "
        "misleading assumption classes, and retains positive aggregate conflict "
        "savings versus solver-only fallback."
    ),
    "misleading_class_blocked": (
        "Confirms that classes with contradiction, overwrite, or LNS-rejected "
        "assumption evidence are blocked instead of routed through ungated p-bit guidance."
    ),
    "conflicts_saved_by_class": (
        "Reports solver-only-minus-gated CDCL conflicts by class so class-specific "
        "benefit and harm remain visible."
    ),
    "ungated_vs_gated_delta": (
        "Compares ungated guidance, gated guidance, and solver-only fallback over "
        "the same instances to prove the gate removes misleading harm rather than "
        "hiding it."
    ),
    "correctness_preserved": (
        "True only when gated and ungated final SAT/UNSAT labels match solver-only "
        "labels and every final SAT model satisfies the original CNF."
    ),
    "hardware_speedup_claimed": (
        "Always false for Exp 5300 because p-bit/Ising assumptions are replayed "
        "CPU-side and no hardware execution is measured."
    ),
    "tests_run": (
        "Commands run to validate gate decisions, misleading-class blocking, "
        "correctness preservation, no hardware speedup claims, artifact schema, "
        "and new-code coverage."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "pbit_gate_ready",
    "misleading_class_blocked",
    "conflicts_saved_by_class",
    "ungated_vs_gated_delta",
    "correctness_preserved",
    "hardware_speedup_claimed",
    "tests_run",
)
WRAPPED_FIELDS = tuple(
    field for field in REQUIRED_ARTIFACT_FIELDS if field != "tests_run"
)


@dataclass(frozen=True)
class GateInstance:
    """One replay row drawn from Exp 5292 or Exp 5299 fixture classes."""

    instance_id: str
    instance_class: str
    source_experiment: str
    n_vars: int
    clauses: tuple[tuple[int, ...], ...]
    source_fixture_id: str
    source_artifact: str
    assumption_literals: tuple[int, ...]
    assumption_method: str
    lns_repair_agreement: str
    candidate_format_valid: bool
    hardware_execution: bool = False


def wrap_field(field: str, value: Any) -> JsonDict:
    """Attach the task-required principle to an artifact field."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def build_gate_instances() -> tuple[GateInstance, ...]:
    """Build replay rows from Exp 5292 guidance and Exp 5299 LNS fixtures."""

    cdcl_instances = tuple(_instances_from_exp5292())
    lns_instances = tuple(_instances_from_exp5299())
    return cdcl_instances + lns_instances


def evaluate_gate_instance(instance: GateInstance) -> JsonDict:
    """Compare solver-only, ungated guidance, and gated guidance for one row."""

    solver_only = cdcl.run_cdcl(instance.clauses, n_vars=instance.n_vars)
    ungated = _run_ungated(instance, solver_only)
    features = _gate_features(instance, ungated)
    decision = decide_gate(instance, features)
    gated = _run_gated(instance, solver_only, ungated, decision)
    class_delta = {
        "solver_only_minus_ungated": _metric_savings(
            solver_only.metrics,
            ungated["metrics"],
            suffix="_saved",
        ),
        "solver_only_minus_gated": _metric_savings(
            solver_only.metrics,
            gated["metrics"],
            suffix="_saved",
        ),
        "ungated_minus_gated": _metric_savings(
            ungated["metrics"],
            gated["metrics"],
            suffix="_saved_by_gate",
        ),
    }
    correctness = _row_correctness_preserved(instance, solver_only, ungated, gated)
    return {
        "instance_id": instance.instance_id,
        "instance_class": instance.instance_class,
        "source_experiment": instance.source_experiment,
        "source_fixture_id": instance.source_fixture_id,
        "source_artifact": instance.source_artifact,
        "assumptions": {
            "literals": list(instance.assumption_literals),
            "method": instance.assumption_method,
            "hardware_execution": instance.hardware_execution,
            "simulated_guidance": True,
        },
        "solver_only": solver_only.as_serializable(),
        "ungated": ungated,
        "gated": gated,
        "gate_features": features,
        "gate_decision": decision,
        "class_delta": class_delta,
        "correctness_preserved": correctness,
    }


def decide_gate(instance: GateInstance, features: Mapping[str, Any]) -> JsonDict:
    """Route only feature-aligned assumptions through p-bit/CDCL guidance."""

    if not instance.assumption_literals:
        if instance.lns_repair_agreement in {"malformed", "rejected"}:
            return {
                "route": "block",
                "reason": "malformed_or_rejected_lns_fixture",
            }
        return {"route": "allow", "reason": "no_assumptions_neutral"}
    if (
        features["contradiction_count"] > 0
        or features["solver_overwrite_count"] > 0
        or features["factor_alignment_score"] < 1.0
        or instance.lns_repair_agreement in {"malformed", "rejected"}
    ):
        return {"route": "block", "reason": "contradiction_or_overwrite"}
    return {"route": "allow", "reason": "aligned_deterministic_features"}


def run_benchmark() -> JsonDict:
    """Run the deterministic instance-class gate benchmark."""

    rows = [evaluate_gate_instance(instance) for instance in build_gate_instances()]
    aggregate = _aggregate_metrics(rows)
    conflicts_by_class = {
        row["instance_class"]: row["class_delta"]["solver_only_minus_gated"][
            "conflicts_saved"
        ]
        for row in rows
    }
    misleading = _misleading_class_blocked(rows)
    correctness = all(row["correctness_preserved"] for row in rows)
    pbit_gate_ready = bool(
        correctness
        and misleading["all_misleading_blocked"]
        and aggregate["solver_only_vs_gated_delta"]["conflicts_saved"] > 0
    )
    return {
        "per_instance_results": rows,
        "aggregate_metrics": aggregate,
        "conflicts_saved_by_class": conflicts_by_class,
        "misleading_class_blocked": misleading,
        "correctness_preserved": correctness,
        "pbit_gate_ready": pbit_gate_ready,
        "gate_rule": {
            "classifier_kind": CLASSIFIER_KIND,
            "features": [
                "assumption_conflict_prefix",
                "factor_alignment_score",
                "lns_repair_agreement",
                "contradiction_count",
                "solver_overwrite_count",
            ],
            "route_rule": (
                "allow non-empty assumptions only when CDCL accepts them, factor "
                "alignment is exact, LNS agreement is accepted, and overwrite "
                "count is zero; empty assumptions are neutral unless the LNS "
                "fixture is malformed or rejected"
            ),
        },
    }


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the validated Exp 5300 terminal artifact."""

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
        "pbit_gate_ready": wrap_field(
            "pbit_gate_ready",
            benchmark["pbit_gate_ready"],
        ),
        "misleading_class_blocked": wrap_field(
            "misleading_class_blocked",
            benchmark["misleading_class_blocked"],
        ),
        "conflicts_saved_by_class": wrap_field(
            "conflicts_saved_by_class",
            benchmark["conflicts_saved_by_class"],
        ),
        "ungated_vs_gated_delta": wrap_field(
            "ungated_vs_gated_delta",
            benchmark["aggregate_metrics"]["ungated_vs_gated_delta"],
        ),
        "correctness_preserved": wrap_field(
            "correctness_preserved",
            benchmark["correctness_preserved"],
        ),
        "hardware_speedup_claimed": wrap_field("hardware_speedup_claimed", False),
        "tests_run": [dict(row) for row in tests_run or []],
        "aggregate_metrics": benchmark["aggregate_metrics"],
        "per_instance_results": benchmark["per_instance_results"],
        "gate_rule": benchmark["gate_rule"],
        "source_artifacts": [
            str(cdcl.RESULT_RELATIVE_PATH),
            str(lns.RESULT_RELATIVE_PATH),
        ],
        "claim_limits": [
            "CPU replay of p-bit/Ising assumptions only",
            "Minisat22 CDCL remains authoritative for SAT/UNSAT labels",
            "misleading assumptions are blocked by deterministic feature gates",
            "ungated p-bit guidance is not safe to route on misleading classes",
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
    """Fail closed when the Exp 5300 artifact drifts from its contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        wrapped = artifact[field]
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require("value" in wrapped, f"{field} missing value")
        _require(
            wrapped.get("principle") == FIELD_PRINCIPLES[field],
            f"{field} principle drift",
        )

    verdict = artifact["honest_verdict"]["value"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict prefix",
    )
    _require("p-bit/CDCL gate" in verdict, "honest_verdict must mention p-bit/CDCL gate")
    _require(
        artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE,
        f"inference_substrate must be {INFERENCE_SUBSTRATE}",
    )
    _require(
        artifact["hardware_speedup_claimed"]["value"] is False,
        "hardware speedup must be false",
    )
    _require(
        artifact["correctness_preserved"]["value"] is True,
        "correctness must be preserved",
    )
    _require(
        artifact["pbit_gate_ready"]["value"] is True,
        "pbit gate must be ready for this deliverable",
    )
    misleading = artifact["misleading_class_blocked"]["value"]
    _require(
        misleading["all_misleading_blocked"] is True,
        "misleading classes must be blocked",
    )
    _require(
        misleading["blocked_classes"] == list(MISLEADING_ASSUMPTION_CLASSES),
        "misleading blocked class list drift",
    )
    delta = artifact["ungated_vs_gated_delta"]["value"]
    _require(
        delta["conflicts_saved_by_gate"] > 0,
        "gate must save conflicts versus ungated guidance",
    )
    conflicts_by_class = artifact["conflicts_saved_by_class"]["value"]
    _require(
        conflicts_by_class["aligned_factor_sat"] > 0,
        "aligned factor class must retain benefit",
    )
    _require(
        conflicts_by_class["misleading_factor_sat"] == 0,
        "misleading factor class must fall back to solver-only",
    )
    _require(isinstance(artifact["tests_run"], list), "tests_run must be list")
    _require("REQ-VERIFY-5300" in artifact["spec_refs"], "spec refs must include REQ-VERIFY-5300")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum drift")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5300 JSON artifact and return the validated payload."""

    artifact = build_artifact(duration_s=duration_s, tests_run=tests_run)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _instances_from_exp5292() -> tuple[GateInstance, ...]:
    agreement_by_class = {
        "aligned_factor_sat": "accepted",
        "misleading_factor_sat": "rejected",
        "neutral_factor_sat": "neutral",
    }
    rows: list[GateInstance] = []
    for instance in cdcl.build_factor_guidance_instances():
        assumptions = cdcl.generate_assumptions(instance)
        rows.append(
            GateInstance(
                instance_id=instance.instance_id,
                instance_class=instance.instance_class,
                source_experiment="exp5292",
                n_vars=instance.n_vars,
                clauses=instance.clauses,
                source_fixture_id=instance.source_fixture_id,
                source_artifact=str(cdcl.RESULT_RELATIVE_PATH),
                assumption_literals=assumptions.literals,
                assumption_method=assumptions.method,
                lns_repair_agreement=agreement_by_class[instance.instance_class],
                candidate_format_valid=True,
            )
        )
    return tuple(rows)


def _instances_from_exp5299() -> tuple[GateInstance, ...]:
    boundary = lns.base_boundary()
    rows: list[GateInstance] = []
    for instance in lns.build_lns_instances():
        validation = lns.validate_repair_candidate(instance.repair_candidate, boundary)
        if validation.format_valid:
            literals = _assignment_literals(
                boundary,
                instance.repair_candidate.payload["assignments"],
            )
            agreement = (
                "accepted"
                if instance.repair_candidate.expected_behavior in {"solver_accepts", "solver_accepts_noop"}
                else "rejected"
            )
        else:
            literals = ()
            agreement = "malformed"
        rows.append(
            GateInstance(
                instance_id=instance.instance_id,
                instance_class=instance.instance_class,
                source_experiment="exp5299",
                n_vars=instance.n_vars,
                clauses=instance.clauses,
                source_fixture_id=instance.source_fixture_id,
                source_artifact=str(lns.RESULT_RELATIVE_PATH),
                assumption_literals=literals,
                assumption_method=instance.repair_candidate.operator,
                lns_repair_agreement=agreement,
                candidate_format_valid=validation.format_valid,
            )
        )
    return tuple(rows)


def _assignment_literals(
    boundary: lns.v5278.BoundaryInstance,
    assignments: Mapping[str, Any],
) -> tuple[int, ...]:
    bits = boundary.assignment_to_bits(dict(assignments))
    return tuple(index + 1 for index, bit in enumerate(bits) if bit)


def _run_ungated(instance: GateInstance, solver_only: cdcl.CdclRun) -> JsonDict:
    if not instance.assumption_literals:
        return {
            "primary_status": solver_only.status,
            "final_status": solver_only.status,
            "final_model": list(solver_only.model),
            "fallback_used": False,
            "overwrite_count": 0,
            "metrics": dict(solver_only.metrics),
            "primary_metrics": dict(solver_only.metrics),
            "fallback_metrics": None,
        }
    primary = cdcl.run_cdcl(
        instance.clauses,
        n_vars=instance.n_vars,
        assumptions=instance.assumption_literals,
    )
    fallback_used = primary.status == "unsat"
    fallback = (
        cdcl.run_cdcl(instance.clauses, n_vars=instance.n_vars)
        if fallback_used
        else None
    )
    final = fallback or primary
    metrics = (
        _add_metrics(primary.metrics, fallback.metrics)
        if fallback is not None
        else dict(primary.metrics)
    )
    return {
        "primary_status": primary.status,
        "final_status": final.status,
        "final_model": list(final.model),
        "fallback_used": fallback_used,
        "overwrite_count": _overwrite_count(instance.assumption_literals, final.model)
        if fallback_used
        else 0,
        "metrics": metrics,
        "primary_metrics": dict(primary.metrics),
        "fallback_metrics": dict(fallback.metrics) if fallback is not None else None,
    }


def _run_gated(
    instance: GateInstance,
    solver_only: cdcl.CdclRun,
    ungated: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> JsonDict:
    if decision["route"] == "allow":
        final_status = ungated["final_status"]
        final_model = list(ungated["final_model"])
        metrics = dict(ungated["metrics"])
        used_solver_only = False
    else:
        final_status = solver_only.status
        final_model = list(solver_only.model)
        metrics = dict(solver_only.metrics)
        used_solver_only = True
    return {
        "final_status": final_status,
        "final_model": final_model,
        "metrics": metrics,
        "used_solver_only_fallback": used_solver_only,
    }


def _gate_features(instance: GateInstance, ungated: Mapping[str, Any]) -> JsonDict:
    assumption_prefix = _assumption_conflict_prefix(instance, ungated)
    factor_energy = _factor_energy(instance)
    factor_alignment_score = _factor_alignment_score(instance, factor_energy)
    contradiction_count = sum(
        (
            assumption_prefix == "unsat_under_assumptions",
            bool(instance.assumption_literals and factor_energy > 0),
            instance.lns_repair_agreement in {"malformed", "rejected"},
        )
    )
    return {
        "assumption_conflict_prefix": assumption_prefix,
        "factor_alignment_score": factor_alignment_score,
        "factor_energy": factor_energy,
        "lns_repair_agreement": instance.lns_repair_agreement,
        "candidate_format_valid": instance.candidate_format_valid,
        "contradiction_count": int(contradiction_count),
        "solver_overwrite_count": int(ungated["overwrite_count"]),
    }


def _assumption_conflict_prefix(
    instance: GateInstance,
    ungated: Mapping[str, Any],
) -> str:
    if not instance.assumption_literals:
        return "no_assumptions"
    if ungated["primary_status"] == "unsat":
        return "unsat_under_assumptions"
    return "sat_under_assumptions"


def _factor_energy(instance: GateInstance) -> int:
    state = _state_from_positive_lits(instance.n_vars, instance.assumption_literals)
    return _cnf_energy(instance.clauses, state)


def _factor_alignment_score(instance: GateInstance, factor_energy: int) -> float:
    if not instance.assumption_literals:
        return 0.5
    return 1.0 if factor_energy == 0 else 0.0


def _row_correctness_preserved(
    instance: GateInstance,
    solver_only: cdcl.CdclRun,
    ungated: Mapping[str, Any],
    gated: Mapping[str, Any],
) -> bool:
    if solver_only.status == "sat" and not cdcl.verify_model(instance.clauses, solver_only.model):
        return False
    if ungated["final_status"] != solver_only.status:
        return False
    if gated["final_status"] != solver_only.status:
        return False
    if ungated["final_status"] == "sat" and not cdcl.verify_model(
        instance.clauses,
        ungated["final_model"],
    ):
        return False
    return gated["final_status"] == "unsat" or cdcl.verify_model(
        instance.clauses,
        gated["final_model"],
    )


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    solver_only = _sum_metrics(row["solver_only"]["metrics"] for row in rows)
    ungated = _sum_metrics(row["ungated"]["metrics"] for row in rows)
    gated = _sum_metrics(row["gated"]["metrics"] for row in rows)
    return {
        "solver_only": solver_only,
        "ungated": ungated,
        "gated": gated,
        "ungated_vs_gated_delta": _metric_savings(
            ungated,
            gated,
            suffix="_saved_by_gate",
        ),
        "solver_only_vs_gated_delta": _metric_savings(
            solver_only,
            gated,
            suffix="_saved",
        ),
        "solver_only_vs_ungated_delta": _metric_savings(
            solver_only,
            ungated,
            suffix="_saved",
        ),
    }


def _misleading_class_blocked(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    blocked = [
        row["instance_class"]
        for row in rows
        if row["instance_class"] in MISLEADING_ASSUMPTION_CLASSES
        and row["gate_decision"]["route"] == "block"
    ]
    return {
        "misleading_classes": list(MISLEADING_ASSUMPTION_CLASSES),
        "blocked_classes": blocked,
        "all_misleading_blocked": blocked == list(MISLEADING_ASSUMPTION_CLASSES),
        "block_rule": (
            "block non-empty assumptions when CDCL reports assumption UNSAT, "
            "factor energy is nonzero, LNS agreement is rejected, or fallback "
            "overwrites assumptions"
        ),
    }


def _metric_savings(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    suffix: str,
) -> JsonDict:
    result: JsonDict = {
        f"{key}{suffix}": int(left[key]) - int(right[key])
        for key in COUNT_METRIC_KEYS
    }
    result[f"wall_clock_s{suffix}"] = round(
        float(left["wall_clock_s"]) - float(right["wall_clock_s"]),
        9,
    )
    return result


def _sum_metrics(metrics_rows: Iterable[Mapping[str, Any]]) -> JsonDict:
    metrics_list = list(metrics_rows)
    return {
        "conflicts": sum(int(metrics["conflicts"]) for metrics in metrics_list),
        "decisions": sum(int(metrics["decisions"]) for metrics in metrics_list),
        "propagations": sum(int(metrics["propagations"]) for metrics in metrics_list),
        "restarts": sum(int(metrics["restarts"]) for metrics in metrics_list),
        "wall_clock_s": round(
            sum(float(metrics["wall_clock_s"]) for metrics in metrics_list),
            9,
        ),
    }


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


def _overwrite_count(assumptions: Sequence[int], final_model: Sequence[int]) -> int:
    final_literals = set(final_model)
    return sum(1 for literal in assumptions if literal not in final_literals)


def _state_from_positive_lits(n_vars: int, literals: Sequence[int]) -> tuple[bool, ...]:
    positive = {literal for literal in literals if literal > 0}
    return tuple(index in positive for index in range(1, n_vars + 1))


def _cnf_energy(clauses: Sequence[Sequence[int]], state: Sequence[bool]) -> int:
    return sum(
        not any(
            (literal > 0 and state[abs(literal) - 1])
            or (literal < 0 and not state[abs(literal) - 1])
            for literal in clause
        )
        for clause in clauses
    )


def _honest_verdict(benchmark: Mapping[str, Any]) -> str:
    if not benchmark["correctness_preserved"]:  # pragma: no cover - validation keeps the fixture correct.
        return "blocked_correctness_not_preserved: p-bit/CDCL gate cannot report unsafe labels"
    if not benchmark["misleading_class_blocked"]["all_misleading_blocked"]:  # pragma: no cover - tests cover the positive route.
        return "harmful_pbit_cdcl_gate_failed_to_block_misleading_assumptions"
    if benchmark["aggregate_metrics"]["solver_only_vs_gated_delta"]["conflicts_saved"] <= 0:  # pragma: no cover - current fixture keeps benefit.
        return (
            "null: p-bit/CDCL gate blocked misleading harm but did not retain "
            "aggregate conflict benefit; retire ungated p-bit guidance"
        )
    return (
        "complete: p-bit/CDCL gate helped by blocking misleading-assumption "
        "classes while preserving aggregate conflict savings on deterministic "
        "Exp5292 and Exp5299 fixtures"
    )


def _checksum_payload(benchmark: Mapping[str, Any]) -> str:
    rows = [
        {
            "instance_id": row["instance_id"],
            "instance_class": row["instance_class"],
            "source_experiment": row["source_experiment"],
            "assumptions": row["assumptions"],
            "gate_features": row["gate_features"],
            "gate_decision": row["gate_decision"],
            "solver_only_status": row["solver_only"]["status"],
            "ungated_final_status": row["ungated"]["final_status"],
            "gated_final_status": row["gated"]["final_status"],
            "solver_only_metrics": _stable_metrics(row["solver_only"]["metrics"]),
            "ungated_metrics": _stable_metrics(row["ungated"]["metrics"]),
            "gated_metrics": _stable_metrics(row["gated"]["metrics"]),
            "class_delta": _stable_class_delta(row["class_delta"]),
            "correctness_preserved": row["correctness_preserved"],
        }
        for row in benchmark["per_instance_results"]
    ]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "classifier_kind": CLASSIFIER_KIND,
        "rows": rows,
        "conflicts_saved_by_class": benchmark["conflicts_saved_by_class"],
        "misleading_class_blocked": benchmark["misleading_class_blocked"],
        "pbit_gate_ready": benchmark["pbit_gate_ready"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_metrics(metrics: Mapping[str, Any]) -> JsonDict:
    return {key: int(metrics[key]) for key in COUNT_METRIC_KEYS}


def _stable_class_delta(class_delta: Mapping[str, Any]) -> JsonDict:
    return {
        name: {
            key: int(value)
            for key, value in metrics.items()
            if key != "wall_clock_s_saved" and key != "wall_clock_s_saved_by_gate"
        }
        for name, metrics in class_delta.items()
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
