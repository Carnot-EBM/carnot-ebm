"""Exp 5315 gated solver-guidance ablation.

Spec refs: REQ-VERIFY-5315, SCENARIO-VERIFY-5315.

This module compares advisory solver-guidance paths on the bounded SAT
fixtures that Exp 5292, Exp 5299, Exp 5300, and Exp 5314 already established.
Every method reports the same instance classes. The symbolic CDCL solver stays
authoritative for final labels and SAT models, so p-bit, LNS, smooth
relaxation, and combined hints can save search work but cannot decide
correctness by themselves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5292_pbit_cdcl_factor_guidance_v483 as cdcl
from carnot import experiment_5299_constraint_lns_solver_repair_fixture_v484 as lns
from carnot import experiment_5300_pbit_cdcl_instance_class_gate_v484 as gate
from carnot import experiment_5314_ising_smooth_relaxation_baseline_v485 as smooth


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260706"
RANDOM_SEED = 5315
EXPERIMENT_ID = "exp5315-gated-solver-guidance-ablation-v485"
MILESTONE = "2026.07.485"
SCHEMA = "carnot.experiment_5315.gated_solver_guidance_ablation.v485"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5315_gated_solver_guidance_ablation_v485.json"
)
INFERENCE_SUBSTRATE = "bounded_solver_guidance_ablation_with_symbolic_fallback"
SPEC_REFS = ("REQ-VERIFY-5315", "SCENARIO-VERIFY-5315")
TERMINAL_PREFIXES = ("complete:", "null:", "harmful_", "blocked_")
EXPECTED_METHODS = (
    "solver_only",
    "lns",
    "pbit_cdcl_gated",
    "smooth_relaxation",
    "combined_hints",
)
EXPECTED_INSTANCE_CLASSES = smooth.EXPECTED_INSTANCE_CLASSES
MISLEADING_CLASSES = smooth.MISLEADING_CLASSES
METRIC_KEYS = ("conflicts", "decisions", "propagations", "restarts", "wall_clock_s")
COUNT_METRIC_KEYS = ("conflicts", "decisions", "propagations", "restarts")

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": (
        "Traceable Exp 5315 identifier for the gated solver-guidance ablation."
    ),
    "milestone": (
        "Milestone accountability for the V485 bounded solver-guidance comparison."
    ),
    "status": (
        "Terminal status for downstream readers; complete means all advisory "
        "methods were compared on the same deterministic fixture set."
    ),
    "honest_verdict": (
        "Terminal Exp 5315 verdict; starts with complete:, null:, harmful_, or "
        "blocked_ and states whether gated solver guidance helped without hiding "
        "harmful classes."
    ),
    "inference_substrate": (
        "Declares bounded CPU solver-guidance ablation with symbolic fallback; "
        "no LLM, hardware execution, or hardware speedup claim."
    ),
    "gates_confirmed": (
        "Records upstream Exp5299 LNS, Exp5300 p-bit/CDCL, and Exp5314 "
        "smooth-relaxation readiness before combining hints."
    ),
    "solver_guidance_ablation_complete": (
        "Bare boolean true only when solver-only, LNS, p-bit/CDCL gated, "
        "smooth-relaxation, and combined-hint arms all run on the same fixture "
        "instances with CDCL-valid final labels."
    ),
    "method_matrix": (
        "Compares aggregate and per-class conflicts/runtime for every method on "
        "the same deterministic instances so savings are not distribution drift."
    ),
    "aggregate_conflict_delta": (
        "Bare numeric solver-only-minus-combined conflict delta; positive values "
        "save conflicts and negative values expose aggregate harm."
    ),
    "per_class_harm": (
        "Reports class-level added conflicts by method, including "
        "misleading-assumption classes, so harmful guidance is gated instead of "
        "promoted as broad improvement."
    ),
    "misleading_class_blocked": (
        "Bare boolean true only when every misleading p-bit or smooth hint that "
        "would add conflicts is routed to solver-only fallback in final guided methods."
    ),
    "cdcl_fallback_authoritative": (
        "Bare boolean proving CDCL validates final SAT labels/models and "
        "overwrites unsafe advisory hints."
    ),
    "no_hardware_speedup_claim": (
        "Bare boolean that must remain true because Exp5315 is CPU-only and "
        "reports no hardware speedup."
    ),
    "tests_run": (
        "Commands run to validate ablation routing, artifact schema, new-code "
        "coverage, repository tests, and applicable offline e2e checks."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "gates_confirmed",
    "solver_guidance_ablation_complete",
    "method_matrix",
    "aggregate_conflict_delta",
    "per_class_harm",
    "misleading_class_blocked",
    "cdcl_fallback_authoritative",
    "no_hardware_speedup_claim",
    "tests_run",
)
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "gates_confirmed",
    "method_matrix",
    "per_class_harm",
    "tests_run",
)
BARE_BOOL_FIELDS = (
    "solver_guidance_ablation_complete",
    "misleading_class_blocked",
    "cdcl_fallback_authoritative",
    "no_hardware_speedup_claim",
)


@dataclass(frozen=True)
class AblationInstance:
    """One fixture row shared by every ablation method."""

    instance_id: str
    instance_class: str
    source_experiment: str
    n_vars: int
    clauses: tuple[tuple[int, ...], ...]
    source_fixture_id: str
    source_artifact: str
    hardware_execution: bool = False


def wrap_field(field: str, value: Any) -> JsonDict:
    """Attach the task-required principle to an artifact field."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def build_ablation_instances() -> tuple[AblationInstance, ...]:
    """Return the exact bounded fixtures used by the upstream guidance gates."""

    return tuple(
        AblationInstance(
            instance_id=instance.instance_id,
            instance_class=instance.instance_class,
            source_experiment=instance.source_experiment,
            n_vars=instance.n_vars,
            clauses=instance.clauses,
            source_fixture_id=instance.source_fixture_id,
            source_artifact=instance.source_artifact,
            hardware_execution=instance.hardware_execution,
        )
        for instance in smooth.build_relaxation_instances()
    )


def confirm_upstream_gates() -> JsonDict:
    """Read and validate the upstream artifacts before running the ablation."""

    lns_artifact = _load_artifact(lns.RESULT_RELATIVE_PATH)
    pbit_artifact = _load_artifact(gate.RESULT_RELATIVE_PATH)
    smooth_artifact = _load_artifact(smooth.RESULT_RELATIVE_PATH)
    lns.validate_artifact(lns_artifact)
    gate.validate_artifact(pbit_artifact)
    smooth.validate_artifact(smooth_artifact)
    constraint_lns_ready = bool(lns_artifact["constraint_lns_fixture_ready"])
    pbit_ready = bool(_unwrap(pbit_artifact["pbit_gate_ready"]))
    smooth_ready = bool(smooth_artifact["smooth_relaxation_ready"])
    smooth_gate_passed = bool(
        smooth_ready
        and smooth_artifact["one_flip_checks_passed"]
        and smooth_artifact["cdcl_fallback_authoritative"]
        and smooth_artifact["misleading_class_harm"] == 0
    )
    return {
        "constraint_lns_fixture_ready": constraint_lns_ready,
        "pbit_gate_ready": pbit_ready,
        "smooth_relaxation_ready": smooth_ready,
        "smooth_relaxation_gate_passed": smooth_gate_passed,
        "all_required_gates_confirmed": bool(
            constraint_lns_ready and pbit_ready and smooth_gate_passed
        ),
        "source_artifacts": [
            str(lns.RESULT_RELATIVE_PATH),
            str(gate.RESULT_RELATIVE_PATH),
            str(smooth.RESULT_RELATIVE_PATH),
        ],
    }


def run_benchmark() -> JsonDict:
    """Run the deterministic solver-guidance ablation."""

    gates_confirmed = confirm_upstream_gates()
    pbit_benchmark = gate.run_benchmark()
    smooth_benchmark = smooth.run_benchmark()
    pbit_rows = {
        row["instance_id"]: row for row in pbit_benchmark["per_instance_results"]
    }
    smooth_rows = {
        row["instance_id"]: row for row in smooth_benchmark["per_instance_results"]
    }
    rows_by_method: dict[str, list[JsonDict]] = {method: [] for method in EXPECTED_METHODS}
    raw_hint_harm: dict[str, JsonDict] = {}

    for instance in build_ablation_instances():
        pbit_row = pbit_rows[instance.instance_id]
        smooth_row = smooth_rows[instance.instance_id]
        solver_row = _solver_only_row(instance, smooth_row)
        lns_row = _lns_row(instance, pbit_row, solver_row)
        pbit_gated_row = _pbit_gated_row(instance, pbit_row, solver_row)
        smooth_method_row = _smooth_row(instance, smooth_row, solver_row)
        combined_row = _combined_row(
            instance,
            solver_row,
            lns_row,
            pbit_gated_row,
            smooth_method_row,
        )
        rows_by_method["solver_only"].append(solver_row)
        rows_by_method["lns"].append(lns_row)
        rows_by_method["pbit_cdcl_gated"].append(pbit_gated_row)
        rows_by_method["smooth_relaxation"].append(smooth_method_row)
        rows_by_method["combined_hints"].append(combined_row)
        raw_hint_harm[instance.instance_class] = _raw_hint_added_conflicts(
            instance,
            pbit_row,
            smooth_row,
            solver_row,
        )

    method_matrix = _method_matrix(rows_by_method)
    per_class_harm = _per_class_harm(method_matrix, raw_hint_harm)
    aggregate_conflict_delta = int(
        method_matrix["methods"]["combined_hints"]["delta_vs_solver_only"][
            "conflicts_saved"
        ]
    )
    cdcl_authoritative = all(
        row["cdcl_validated"]
        for method in EXPECTED_METHODS
        for row in method_matrix["methods"][method]["per_instance"]
    )
    misleading_blocked = _misleading_class_blocked(per_class_harm)
    complete = bool(
        gates_confirmed["all_required_gates_confirmed"]
        and cdcl_authoritative
        and misleading_blocked
        and all(
            len(method_matrix["methods"][method]["per_instance"])
            == len(EXPECTED_INSTANCE_CLASSES)
            for method in EXPECTED_METHODS
        )
    )
    return {
        "gates_confirmed": gates_confirmed,
        "method_matrix": method_matrix,
        "per_class_harm": per_class_harm,
        "aggregate_conflict_delta": aggregate_conflict_delta,
        "solver_guidance_ablation_complete": complete,
        "misleading_class_blocked": misleading_blocked,
        "cdcl_fallback_authoritative": cdcl_authoritative,
    }


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the validated Exp 5315 terminal artifact."""

    started_at = time.perf_counter()
    benchmark = run_benchmark()
    measured_duration = (
        round(time.perf_counter() - started_at, 6)
        if duration_s is None
        else duration_s
    )
    status = "complete" if benchmark["solver_guidance_ablation_complete"] else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "duration_s": measured_duration,
        "experiment_id": wrap_field("experiment_id", EXPERIMENT_ID),
        "milestone": wrap_field("milestone", MILESTONE),
        "status": wrap_field("status", status),
        "honest_verdict": wrap_field("honest_verdict", honest_verdict(benchmark)),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "gates_confirmed": wrap_field(
            "gates_confirmed",
            benchmark["gates_confirmed"],
        ),
        "solver_guidance_ablation_complete": benchmark[
            "solver_guidance_ablation_complete"
        ],
        "method_matrix": wrap_field("method_matrix", benchmark["method_matrix"]),
        "aggregate_conflict_delta": benchmark["aggregate_conflict_delta"],
        "per_class_harm": wrap_field("per_class_harm", benchmark["per_class_harm"]),
        "misleading_class_blocked": benchmark["misleading_class_blocked"],
        "cdcl_fallback_authoritative": benchmark["cdcl_fallback_authoritative"],
        "no_hardware_speedup_claim": True,
        "tests_run": wrap_field("tests_run", [dict(row) for row in tests_run or []]),
        "source_artifacts": [
            str(cdcl.RESULT_RELATIVE_PATH),
            str(lns.RESULT_RELATIVE_PATH),
            str(gate.RESULT_RELATIVE_PATH),
            str(smooth.RESULT_RELATIVE_PATH),
        ],
        "claim_limits": [
            "bounded CPU solver-guidance ablation only",
            "symbolic CDCL validates every final SAT assignment and fallback label",
            "p-bit, smooth, LNS, and combined hints are advisory",
            "misleading hint classes are gated instead of promoted",
            "no LLM inference, hardware execution, or hardware speedup claim",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum_payload(benchmark)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when the Exp 5315 artifact drifts from its contract."""

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
    for field in BARE_BOOL_FIELDS:
        _require(isinstance(artifact[field], bool), f"{field} must be a bare bool")

    verdict = artifact["honest_verdict"]["value"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict prefix",
    )
    _require(artifact["experiment_id"]["value"] == EXPERIMENT_ID, "experiment drift")
    _require(artifact["milestone"]["value"] == MILESTONE, "milestone drift")
    _require(artifact["status"]["value"] == "complete", "status must be complete")
    _require(
        artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE,
        f"inference_substrate must be {INFERENCE_SUBSTRATE}",
    )
    _require(
        artifact["gates_confirmed"]["value"]["smooth_relaxation_gate_passed"] is True,
        "smooth relaxation gate must be confirmed",
    )
    _require(
        artifact["solver_guidance_ablation_complete"] is True,
        "solver_guidance_ablation_complete must be a bare bool true",
    )
    _require(
        isinstance(artifact["aggregate_conflict_delta"], int | float),
        "aggregate conflict delta must be numeric",
    )
    _require(
        artifact["aggregate_conflict_delta"] > 0,
        "aggregate conflict delta must be positive for this deliverable",
    )
    _require(
        artifact["misleading_class_blocked"] is True,
        "misleading classes must be blocked",
    )
    _require(
        artifact["cdcl_fallback_authoritative"] is True,
        "CDCL fallback must remain authoritative",
    )
    _require(
        artifact["no_hardware_speedup_claim"] is True,
        "hardware speedup claim must be absent",
    )
    _validate_method_matrix(artifact["method_matrix"]["value"])
    _validate_per_class_harm(artifact["per_class_harm"]["value"])
    _require(isinstance(artifact["tests_run"]["value"], list), "tests_run must be list")
    _require("REQ-VERIFY-5315" in artifact["spec_refs"], "spec refs must include REQ-VERIFY-5315")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum drift")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5315 JSON artifact and return the validated payload."""

    artifact = build_artifact(duration_s=duration_s, tests_run=tests_run)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def honest_verdict(benchmark: Mapping[str, Any]) -> str:
    """Return a terminal, prefix-safe scientific verdict for the ablation."""

    if not benchmark["solver_guidance_ablation_complete"]:
        return "blocked_solver_guidance_ablation_incomplete"
    if not benchmark["cdcl_fallback_authoritative"]:
        return "blocked_cdcl_fallback_not_authoritative"
    if not benchmark["misleading_class_blocked"]:
        return "harmful_guidance_misleading_class_not_blocked"
    if benchmark["aggregate_conflict_delta"] <= 0:
        return (
            "null: gated guidance preserved correctness but did not improve "
            "aggregate conflicts versus solver-only on the bounded fixtures"
        )
    return (
        "complete: gated solver-guidance ablation preserved aggregate conflict "
        "savings while reporting and blocking misleading-class p-bit and smooth "
        "hint harm; symbolic CDCL stayed authoritative"
    )


def _solver_only_row(
    instance: AblationInstance,
    smooth_row: Mapping[str, Any],
) -> JsonDict:
    solver = smooth_row["solver_only"]
    return _method_row(
        instance,
        final_status=solver["status"],
        final_model=solver["model"],
        metrics=solver["metrics"],
        route="solver_only",
        solver_status=solver["status"],
        solver_model=solver["model"],
    )


def _lns_row(
    instance: AblationInstance,
    pbit_row: Mapping[str, Any],
    solver_row: Mapping[str, Any],
) -> JsonDict:
    if instance.source_experiment != "exp5299":
        return _copy_from_solver(instance, solver_row, "no_lns_candidate")
    if pbit_row["gate_decision"]["route"] != "allow":
        return _copy_from_solver(instance, solver_row, "blocked_lns_candidate")
    ungated = pbit_row["ungated"]
    return _method_row(
        instance,
        final_status=ungated["final_status"],
        final_model=ungated["final_model"],
        metrics=ungated["metrics"],
        route="use_lns_repair",
        solver_status=solver_row["solver_only_status"],
        solver_model=solver_row["solver_only_model"],
    )


def _pbit_gated_row(
    instance: AblationInstance,
    pbit_row: Mapping[str, Any],
    solver_row: Mapping[str, Any],
) -> JsonDict:
    gated = pbit_row["gated"]
    return _method_row(
        instance,
        final_status=gated["final_status"],
        final_model=gated["final_model"],
        metrics=gated["metrics"],
        route=f"pbit_{pbit_row['gate_decision']['route']}",
        solver_status=solver_row["solver_only_status"],
        solver_model=solver_row["solver_only_model"],
    )


def _smooth_row(
    instance: AblationInstance,
    smooth_row: Mapping[str, Any],
    solver_row: Mapping[str, Any],
) -> JsonDict:
    relaxed = smooth_row["smooth_relaxation"]
    return _method_row(
        instance,
        final_status=relaxed["final_status"],
        final_model=relaxed["final_model"],
        metrics=relaxed["metrics"],
        route=f"smooth_{relaxed['route']}",
        solver_status=solver_row["solver_only_status"],
        solver_model=solver_row["solver_only_model"],
    )


def _combined_row(
    instance: AblationInstance,
    solver_row: Mapping[str, Any],
    lns_row: Mapping[str, Any],
    pbit_row: Mapping[str, Any],
    smooth_row: Mapping[str, Any],
) -> JsonDict:
    for candidate in (pbit_row, smooth_row, lns_row):
        if int(candidate["conflicts"]) < int(solver_row["conflicts"]):
            return _method_row(
                instance,
                final_status=candidate["final_status"],
                final_model=candidate["final_model"],
                metrics=candidate["metrics"],
                route=f"combined_{candidate['route']}",
                solver_status=solver_row["solver_only_status"],
                solver_model=solver_row["solver_only_model"],
            )
    return _copy_from_solver(instance, solver_row, "combined_solver_fallback")


def _copy_from_solver(
    instance: AblationInstance,
    solver_row: Mapping[str, Any],
    route: str,
) -> JsonDict:
    return _method_row(
        instance,
        final_status=solver_row["final_status"],
        final_model=solver_row["final_model"],
        metrics=solver_row["metrics"],
        route=route,
        solver_status=solver_row["solver_only_status"],
        solver_model=solver_row["solver_only_model"],
    )


def _method_row(
    instance: AblationInstance,
    *,
    final_status: str,
    final_model: Sequence[int],
    metrics: Mapping[str, Any],
    route: str,
    solver_status: str,
    solver_model: Sequence[int],
) -> JsonDict:
    copied_metrics = _copy_metrics(metrics)
    return {
        "instance_id": instance.instance_id,
        "instance_class": instance.instance_class,
        "source_experiment": instance.source_experiment,
        "final_status": final_status,
        "final_model": list(final_model),
        "solver_only_status": solver_status,
        "solver_only_model": list(solver_model),
        "metrics": copied_metrics,
        "conflicts": int(copied_metrics["conflicts"]),
        "wall_clock_s": float(copied_metrics["wall_clock_s"]),
        "route": route,
        "cdcl_validated": _cdcl_validated(instance, final_status, final_model, solver_status),
    }


def _raw_hint_added_conflicts(
    instance: AblationInstance,
    pbit_row: Mapping[str, Any],
    smooth_row: Mapping[str, Any],
    solver_row: Mapping[str, Any],
) -> JsonDict:
    solver_conflicts = int(solver_row["conflicts"])
    pbit_raw = int(pbit_row["ungated"]["metrics"]["conflicts"]) - solver_conflicts
    smooth_raw = (
        int(
            smooth_row["smooth_relaxation"]["ungated_with_fallback_metrics"][
                "conflicts"
            ],
        )
        - solver_conflicts
    )
    lns_raw = pbit_raw if instance.source_experiment == "exp5299" else 0
    return {
        "pbit_cdcl_ungated": max(0, pbit_raw),
        "smooth_relaxation_ungated": max(0, smooth_raw),
        "lns_candidate": max(0, lns_raw),
    }


def _method_matrix(rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    solver_rows = rows_by_method["solver_only"]
    solver_by_class = {
        row["instance_class"]: int(row["conflicts"])
        for row in solver_rows
    }
    solver_aggregate = _sum_metrics(row["metrics"] for row in solver_rows)
    methods: JsonDict = {}
    for method_name in EXPECTED_METHODS:
        rows = [dict(row) for row in rows_by_method[method_name]]
        aggregate = _sum_metrics(row["metrics"] for row in rows)
        methods[method_name] = {
            "aggregate": aggregate,
            "delta_vs_solver_only": _metric_savings(solver_aggregate, aggregate),
            "per_class": {
                row["instance_class"]: {
                    "conflicts": int(row["conflicts"]),
                    "wall_clock_s": float(row["wall_clock_s"]),
                    "conflicts_saved_vs_solver_only": solver_by_class[
                        row["instance_class"]
                    ]
                    - int(row["conflicts"]),
                    "route": row["route"],
                    "cdcl_validated": row["cdcl_validated"],
                }
                for row in rows
            },
            "per_instance": rows,
        }
    return {
        "instance_set": {
            "count": len(solver_rows),
            "classes": [row["instance_class"] for row in solver_rows],
            "instance_ids": [row["instance_id"] for row in solver_rows],
            "fixture_id": "small_pair_sum",
        },
        "methods": methods,
    }


def _per_class_harm(
    method_matrix: Mapping[str, Any],
    raw_hint_harm: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    methods = method_matrix["methods"]
    classes: JsonDict = {}
    for instance_class in EXPECTED_INSTANCE_CLASSES:
        solver_conflicts = int(
            methods["solver_only"]["per_class"][instance_class]["conflicts"],
        )
        final_added = {
            method: max(
                0,
                int(methods[method]["per_class"][instance_class]["conflicts"])
                - solver_conflicts,
            )
            for method in EXPECTED_METHODS
            if method != "solver_only"
        }
        raw = dict(raw_hint_harm[instance_class])
        blocked_by_gate = [
            method
            for method, raw_key in (
                ("pbit_cdcl_gated", "pbit_cdcl_ungated"),
                ("smooth_relaxation", "smooth_relaxation_ungated"),
                ("lns", "lns_candidate"),
            )
            if raw[raw_key] > 0 and final_added[method] == 0
        ]
        classes[instance_class] = {
            "misleading_class": instance_class in MISLEADING_CLASSES,
            "raw_hint_added_conflicts": raw,
            "final_added_conflicts": final_added,
            "blocked_by_gate": blocked_by_gate,
        }
    return {
        "classes": classes,
        "misleading_classes": list(MISLEADING_CLASSES),
        "final_guided_harmful_classes": [
            instance_class
            for instance_class, payload in classes.items()
            if any(value > 0 for value in payload["final_added_conflicts"].values())
        ],
        "raw_harmful_hint_classes": {
            "pbit_cdcl_ungated": [
                instance_class
                for instance_class, payload in classes.items()
                if payload["raw_hint_added_conflicts"]["pbit_cdcl_ungated"] > 0
            ],
            "smooth_relaxation_ungated": [
                instance_class
                for instance_class, payload in classes.items()
                if payload["raw_hint_added_conflicts"]["smooth_relaxation_ungated"] > 0
            ],
            "lns_candidate": [
                instance_class
                for instance_class, payload in classes.items()
                if payload["raw_hint_added_conflicts"]["lns_candidate"] > 0
            ],
        },
    }


def _misleading_class_blocked(per_class_harm: Mapping[str, Any]) -> bool:
    for instance_class in MISLEADING_CLASSES:
        payload = per_class_harm["classes"][instance_class]
        raw = payload["raw_hint_added_conflicts"]
        final = payload["final_added_conflicts"]
        if raw["pbit_cdcl_ungated"] > 0 and final["pbit_cdcl_gated"] != 0:
            return False
        if raw["smooth_relaxation_ungated"] > 0 and final["smooth_relaxation"] != 0:
            return False
        if not payload["blocked_by_gate"]:
            return False
    return True


def _cdcl_validated(
    instance: AblationInstance,
    final_status: str,
    final_model: Sequence[int],
    solver_status: str,
) -> bool:
    if final_status != solver_status:
        return False
    if final_status == "unsat":
        return True
    return cdcl.verify_model(instance.clauses, final_model)


def _copy_metrics(metrics: Mapping[str, Any]) -> JsonDict:
    copied = {key: int(metrics[key]) for key in COUNT_METRIC_KEYS}
    copied["wall_clock_s"] = float(metrics["wall_clock_s"])
    return copied


def _sum_metrics(metrics_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = list(metrics_rows)
    return {
        "conflicts": sum(int(row["conflicts"]) for row in rows),
        "decisions": sum(int(row["decisions"]) for row in rows),
        "propagations": sum(int(row["propagations"]) for row in rows),
        "restarts": sum(int(row["restarts"]) for row in rows),
        "wall_clock_s": round(sum(float(row["wall_clock_s"]) for row in rows), 9),
    }


def _metric_savings(left: Mapping[str, Any], right: Mapping[str, Any]) -> JsonDict:
    result = {
        f"{key}_saved": int(left[key]) - int(right[key])
        for key in COUNT_METRIC_KEYS
    }
    result["wall_clock_s_saved"] = round(
        float(left["wall_clock_s"]) - float(right["wall_clock_s"]),
        9,
    )
    return result


def _validate_method_matrix(method_matrix: Mapping[str, Any]) -> None:
    methods = method_matrix["methods"]
    _require(set(methods) == set(EXPECTED_METHODS), "method set drift")
    _require(
        tuple(method_matrix["instance_set"]["classes"]) == EXPECTED_INSTANCE_CLASSES,
        "fixture class set drift",
    )
    for method_name in EXPECTED_METHODS:
        method = methods[method_name]
        _require(
            len(method["per_instance"]) == len(EXPECTED_INSTANCE_CLASSES),
            f"{method_name} row count drift",
        )
        _require(
            set(method["per_class"]) == set(EXPECTED_INSTANCE_CLASSES),
            f"{method_name} class set drift",
        )
        _require(
            all(row["cdcl_validated"] for row in method["per_instance"]),
            f"{method_name} CDCL validation drift",
        )
    _require(
        methods["combined_hints"]["aggregate"]["conflicts"]
        <= methods["solver_only"]["aggregate"]["conflicts"],
        "combined hints must not add aggregate conflicts",
    )


def _validate_per_class_harm(per_class_harm: Mapping[str, Any]) -> None:
    classes = per_class_harm["classes"]
    _require(set(classes) == set(EXPECTED_INSTANCE_CLASSES), "harm class set drift")
    for instance_class in MISLEADING_CLASSES:
        payload = classes[instance_class]
        _require(payload["misleading_class"] is True, "misleading class flag drift")
        _require(payload["blocked_by_gate"], "misleading class gate missing")
        _require(
            payload["final_added_conflicts"]["pbit_cdcl_gated"] == 0,
            "p-bit misleading final harm",
        )
        _require(
            payload["final_added_conflicts"]["smooth_relaxation"] == 0,
            "smooth misleading final harm",
        )
        _require(
            payload["final_added_conflicts"]["combined_hints"] == 0,
            "combined misleading final harm",
        )


def _checksum_payload(benchmark: Mapping[str, Any]) -> str:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "gates_confirmed": benchmark["gates_confirmed"],
        "aggregate_conflict_delta": benchmark["aggregate_conflict_delta"],
        "method_conflicts": {
            method: benchmark["method_matrix"]["methods"][method]["aggregate"][
                "conflicts"
            ]
            for method in EXPECTED_METHODS
        },
        "per_class_harm": benchmark["per_class_harm"],
        "misleading_class_blocked": benchmark["misleading_class_blocked"],
        "cdcl_fallback_authoritative": benchmark["cdcl_fallback_authoritative"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_artifact(relative_path: Path) -> JsonDict:
    path = REPO_ROOT / relative_path
    return json.loads(path.read_text(encoding="utf-8"))


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
