"""Exp 5299 deterministic constraint-LNS repair fixture.

Spec refs: REQ-VERIFY-5299, SCENARIO-VERIFY-5299.

This module wraps the tiny Exp 5278 factor fixture in a deliberately small
Large Neighborhood Search loop. The destroy and repair operators are simple on
purpose: they let the tests prove that aligned repairs are accepted, bad
structured repairs are rejected, and the CDCL solver remains the final
authority for every SAT model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5278_constraint_factor_graph_boundary_v482 as v5278
from carnot import experiment_5292_pbit_cdcl_factor_guidance_v483 as cdcl


JsonDict = dict[str, Any]

RUN_DATE = "20260706"
RANDOM_SEED = 5299
EXPERIMENT_ID = "exp5299-constraint-lns-solver-repair-fixture-v484"
SCHEMA = "carnot.experiment_5299.constraint_lns_solver_repair_fixture.v484"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5299_constraint_lns_solver_repair_fixture_v484.json"
)
INFERENCE_SUBSTRATE = "offline_deterministic_certificate_no_llm"
SPEC_REFS = ("REQ-VERIFY-5299", "SCENARIO-VERIFY-5299")
TERMINAL_PREFIXES = ("complete:", "blocked_")
CANDIDATE_FORMAT_VERSION = "constraint_lns_repair_v1"
INSTANCE_CLASS_ORDER = (
    "aligned_repair",
    "misleading_repair",
    "neutral_noop_repair",
    "malformed_control",
    "semantic_wrong_control",
)
COUNTER_KEYS = ("conflicts", "decisions", "propagations", "restarts")

CONSTRAINT_LNS_READY_PRINCIPLE = (
    "Bare gate for exp5300; true only when aligned, misleading, neutral, malformed, "
    "and semantic-wrong repair classes are measured, solver correctness is preserved, "
    "classical baselines are present, and unsafe false accepts are zero."
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal Exp 5299 verdict; starts with complete: or blocked_ and states "
        "whether the constraint-LNS fixture is usable."
    ),
    "inference_substrate": (
        "Must be offline_deterministic_certificate_no_llm because Exp 5299 uses "
        "CPU-local deterministic fixtures, a local CDCL solver, and no LLM inference."
    ),
    "constraint_lns_fixture_ready": CONSTRAINT_LNS_READY_PRINCIPLE,
    "constraint_lns_fixture_ready_principle": (
        "Explains why the deterministic constraint-LNS fixture can or cannot gate "
        "downstream p-bit/CDCL guidance experiments."
    ),
    "instance_class_counts": (
        "Counts aligned repair, misleading repair, neutral no-op repair, malformed "
        "control, and semantic-wrong control instances so downstream runs cannot "
        "silently drop a class."
    ),
    "destroy_repair_telemetry": (
        "Records destroy operators, destroyed variables and clauses, repair operators, "
        "repair candidates, solver accept/reject decisions, fallback/overwrite counts, "
        "and CDCL counters."
    ),
    "classical_baseline_results": (
        "Reports solver-only classical baseline results over the same instances so LNS "
        "repair behavior is compared against an authoritative non-guided path."
    ),
    "solver_correctness_preserved": (
        "True only when every final LNS result matches the solver-only baseline label "
        "and every final SAT model satisfies the original CNF."
    ),
    "unsafe_false_accepts": (
        "Counts malformed, semantic-wrong, or misleading repair candidates accepted as "
        "safe repairs; must be zero for the fixture-ready gate."
    ),
    "tests_run": (
        "Commands run to validate LNS repair acceptance, rejection, fallback correctness, "
        "artifact schema, new-code coverage, and repository test status."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "constraint_lns_fixture_ready",
    "constraint_lns_fixture_ready_principle",
    "instance_class_counts",
    "destroy_repair_telemetry",
    "classical_baseline_results",
    "solver_correctness_preserved",
    "unsafe_false_accepts",
    "tests_run",
)
WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "instance_class_counts",
    "destroy_repair_telemetry",
    "classical_baseline_results",
    "solver_correctness_preserved",
    "unsafe_false_accepts",
)


@dataclass(frozen=True)
class RepairCandidate:
    """Structured repair proposal before the symbolic solver checks it."""

    candidate_id: str
    operator: str
    payload: JsonDict
    safety_negative: bool
    expected_behavior: str

    def as_serializable(self) -> JsonDict:
        return {
            "candidate_id": self.candidate_id,
            "operator": self.operator,
            "payload": dict(self.payload),
            "safety_negative": self.safety_negative,
            "expected_behavior": self.expected_behavior,
        }


@dataclass(frozen=True)
class CandidateValidation:
    """Format validation result for a structured repair proposal."""

    format_valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class LnsInstance:
    """One deterministic destroy/repair case over the tiny factor CNF."""

    instance_id: str
    instance_class: str
    n_vars: int
    clauses: tuple[tuple[int, ...], ...]
    source_fixture_id: str
    source_artifact: str
    destroy_operator: str
    repair_operator: str
    repair_candidate: RepairCandidate
    constraint_groups: tuple[JsonDict, ...]


def wrap_field(field: str, value: Any) -> JsonDict:
    """Attach the task-required principle to an artifact field."""

    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def base_boundary() -> v5278.BoundaryInstance:
    """Return the reused Exp 5278 small factor boundary."""

    return v5278.build_boundary(v5278.select_tiny_fixture())


def base_clauses() -> tuple[tuple[int, ...], ...]:
    """Return the same tiny CNF used by the Exp 5292 CDCL guidance fixture."""

    return cdcl.build_factor_guidance_instances()[0].clauses


def verify_model(clauses: Sequence[Sequence[int]], model: Sequence[int]) -> bool:
    """Return true only when a SAT model satisfies every original CNF clause."""

    return cdcl.verify_model(clauses, model)


def constraint_group_metadata(
    boundary: v5278.BoundaryInstance | None = None,
    clauses: Sequence[Sequence[int]] | None = None,
) -> tuple[JsonDict, ...]:
    """Describe declarative constraint groups beside the flat CNF clauses."""

    active_boundary = base_boundary() if boundary is None else boundary
    active_clauses = tuple(base_clauses() if clauses is None else clauses)
    by_variable = {
        variable: {
            index + 1
            for index, bit_variable in enumerate(active_boundary.bit_variables)
            if bit_variable == variable
        }
        for variable in active_boundary.variables
    }
    groups = {
        "a_one_hot": {
            "group_id": "a_one_hot",
            "group_type": "one_hot_domain",
            "variables": [
                bit
                for bit in active_boundary.bit_order
                if bit.startswith("a_")
            ],
            "clause_indices": [],
            "authority": "Minisat22 CDCL over original CNF",
        },
        "b_one_hot": {
            "group_id": "b_one_hot",
            "group_type": "one_hot_domain",
            "variables": [
                bit
                for bit in active_boundary.bit_order
                if bit.startswith("b_")
            ],
            "clause_indices": [],
            "authority": "Minisat22 CDCL over original CNF",
        },
        "sum_and_order_relation": {
            "group_id": "sum_and_order_relation",
            "group_type": "decoded_sum_and_order",
            "variables": list(active_boundary.bit_order),
            "clause_indices": [],
            "authority": "Minisat22 CDCL over original CNF",
        },
    }
    for index, clause in enumerate(active_clauses):
        literal_variables = {abs(literal) for literal in clause}
        if literal_variables <= by_variable["a"]:
            groups["a_one_hot"]["clause_indices"].append(index)
        elif literal_variables <= by_variable["b"]:
            groups["b_one_hot"]["clause_indices"].append(index)
        else:
            groups["sum_and_order_relation"]["clause_indices"].append(index)
    return tuple(groups[group_id] for group_id in groups)


def build_lns_instances() -> tuple[LnsInstance, ...]:
    """Build aligned, misleading, no-op, malformed, and semantic-wrong LNS cases."""

    boundary = base_boundary()
    clauses = base_clauses()
    groups = constraint_group_metadata(boundary, clauses)
    solver_assignment = {
        variable: int(value)
        for variable, value in boundary.solver_assignment.items()
    }
    false_assignment = {
        variable: int(value)
        for variable, value in boundary.false_assignment.items()
    }
    semantic_wrong_assignment = {"a": 1, "b": 3}
    common = {
        "n_vars": len(boundary.bit_order),
        "clauses": clauses,
        "source_fixture_id": boundary.fixture_id,
        "source_artifact": str(v5278.RESULT_RELATIVE_PATH),
        "constraint_groups": groups,
    }
    return (
        LnsInstance(
            instance_id="exp5299_aligned_repair",
            instance_class="aligned_repair",
            destroy_operator="destroy_full_assignment_neighborhood",
            repair_operator="structured_exact_repair",
            repair_candidate=_candidate(
                "aligned_solver_assignment",
                solver_assignment,
                safety_negative=False,
                expected_behavior="solver_accepts",
            ),
            **common,
        ),
        LnsInstance(
            instance_id="exp5299_misleading_repair",
            instance_class="misleading_repair",
            destroy_operator="destroy_full_assignment_neighborhood",
            repair_operator="structured_false_basin_repair",
            repair_candidate=_candidate(
                "misleading_false_assignment",
                false_assignment,
                safety_negative=True,
                expected_behavior="solver_rejects_then_fallback",
            ),
            **common,
        ),
        LnsInstance(
            instance_id="exp5299_neutral_noop_repair",
            instance_class="neutral_noop_repair",
            destroy_operator="destroy_none_noop",
            repair_operator="structured_noop_repair",
            repair_candidate=_candidate(
                "neutral_preserve_assignment",
                solver_assignment,
                safety_negative=False,
                expected_behavior="solver_accepts_noop",
            ),
            **common,
        ),
        LnsInstance(
            instance_id="exp5299_malformed_control",
            instance_class="malformed_control",
            destroy_operator="destroy_full_assignment_neighborhood",
            repair_operator="structured_output_schema_control",
            repair_candidate=RepairCandidate(
                candidate_id="malformed_missing_assignments",
                operator="structured_output_schema_control",
                payload={"format_version": CANDIDATE_FORMAT_VERSION},
                safety_negative=True,
                expected_behavior="schema_rejects_then_fallback",
            ),
            **common,
        ),
        LnsInstance(
            instance_id="exp5299_semantic_wrong_control",
            instance_class="semantic_wrong_control",
            destroy_operator="destroy_full_assignment_neighborhood",
            repair_operator="structured_semantic_negative_control",
            repair_candidate=_candidate(
                "semantic_wrong_assignment",
                semantic_wrong_assignment,
                safety_negative=True,
                expected_behavior="solver_rejects_then_fallback",
            ),
            **common,
        ),
    )


def instance_class_counts(instances: Sequence[LnsInstance]) -> JsonDict:
    """Count required LNS fixture families without artifact wrapping."""

    counts = Counter(instance.instance_class for instance in instances)
    return {name: int(counts.get(name, 0)) for name in INSTANCE_CLASS_ORDER}


def validate_repair_candidate(
    candidate: RepairCandidate,
    boundary: v5278.BoundaryInstance | None = None,
) -> CandidateValidation:
    """Validate structured repair output before symbolic solver assumptions."""

    active_boundary = base_boundary() if boundary is None else boundary
    payload = candidate.payload
    errors: list[str] = []
    if payload.get("format_version") != CANDIDATE_FORMAT_VERSION:
        errors.append(f"format_version must be {CANDIDATE_FORMAT_VERSION}")
    assignments = payload.get("assignments")
    if not isinstance(assignments, Mapping):
        errors.append("assignments must be an object")
        return CandidateValidation(format_valid=False, errors=tuple(errors))
    expected_variables = set(active_boundary.variables)
    actual_variables = set(assignments)
    missing = sorted(expected_variables - actual_variables)
    extra = sorted(actual_variables - expected_variables)
    if missing:
        errors.append("assignments missing variables: " + ",".join(missing))
    if extra:
        errors.append("assignments contains unknown variables: " + ",".join(extra))
    for variable in sorted(expected_variables & actual_variables):
        value = assignments[variable]
        domain = active_boundary.variables[variable]["domain"]
        if not isinstance(value, int):
            errors.append(f"assignment {variable} must be int")
        elif value not in domain:
            errors.append(f"assignment {variable} outside declared domain")
    return CandidateValidation(format_valid=not errors, errors=tuple(errors))


def run_lns_instance(instance: LnsInstance) -> JsonDict:
    """Run one destroy/repair attempt and keep solver-only fallback authoritative."""

    boundary = base_boundary()
    destroyed_variables, destroyed_clauses = _destroyed_neighborhood(instance, boundary)
    baseline = _solver_only_baseline(instance)
    validation = validate_repair_candidate(instance.repair_candidate, boundary)
    assumptions: tuple[int, ...] = ()
    primary_status = "not_run"
    primary_counters = _zero_counters()
    candidate_solver_accepted = False
    final_run: cdcl.CdclRun
    fallback_run: cdcl.CdclRun | None = None

    if validation.format_valid:
        assumptions = _candidate_assumptions(instance.repair_candidate, boundary)
        primary = cdcl.run_cdcl(
            instance.clauses,
            n_vars=instance.n_vars,
            assumptions=assumptions,
        )
        primary_status = primary.status
        primary_counters = _solver_counters(primary.metrics)
        candidate_solver_accepted = bool(
            primary.status == "sat" and verify_model(instance.clauses, primary.model)
        )
        if candidate_solver_accepted:
            final_run = primary
        else:
            fallback_run = cdcl.run_cdcl(instance.clauses, n_vars=instance.n_vars)
            final_run = fallback_run
    else:
        fallback_run = cdcl.run_cdcl(instance.clauses, n_vars=instance.n_vars)
        final_run = fallback_run

    fallback_used = fallback_run is not None
    overwrite_count = (
        _overwrite_count(assumptions, final_run.model)
        if fallback_used and validation.format_valid
        else 0
    )
    final_model_valid = final_run.status == "unsat" or verify_model(
        instance.clauses,
        final_run.model,
    )
    solver_correctness_preserved = bool(
        final_run.status == baseline["status"] and final_model_valid
    )
    solver_decision = "accepted" if candidate_solver_accepted else "rejected"
    rejection_reason = "" if candidate_solver_accepted else _rejection_reason(validation)
    return {
        "instance_id": instance.instance_id,
        "instance_class": instance.instance_class,
        "source_fixture_id": instance.source_fixture_id,
        "constraint_groups": list(instance.constraint_groups),
        "telemetry": {
            "destroy_operator": instance.destroy_operator,
            "destroyed_variables": destroyed_variables,
            "destroyed_clauses": destroyed_clauses,
            "repair_operator": instance.repair_operator,
            "repair_candidate": instance.repair_candidate.as_serializable(),
            "solver_decision": solver_decision,
            "fallback_count": int(fallback_used),
            "overwrite_count": overwrite_count,
            "solver_counters": primary_counters,
        },
        "repair": {
            "candidate": instance.repair_candidate.as_serializable(),
            "candidate_format_valid": validation.format_valid,
            "candidate_safety_negative": instance.repair_candidate.safety_negative,
            "validation_errors": list(validation.errors),
            "assumption_literals": list(assumptions),
            "primary_status": primary_status,
            "candidate_solver_accepted": candidate_solver_accepted,
            "solver_decision": solver_decision,
            "rejection_reason": rejection_reason,
        },
        "fallback": {
            "used": fallback_used,
            "overwrite_count": overwrite_count,
            "status": fallback_run.status if fallback_run is not None else "not_run",
            "solver_counters": _solver_counters(fallback_run.metrics)
            if fallback_run is not None
            else _zero_counters(),
        },
        "baseline": baseline,
        "final": {
            "status": final_run.status,
            "model": list(final_run.model),
            "model_valid": final_model_valid,
            "solver_counters": _solver_counters(final_run.metrics),
        },
        "solver_correctness_preserved": solver_correctness_preserved,
    }


def count_unsafe_false_accepts(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count safety-negative repair candidates that were accepted as repairs."""

    return sum(
        1
        for row in rows
        if row["repair"]["candidate_safety_negative"]
        and row["repair"]["candidate_solver_accepted"]
    )


def run_benchmark() -> JsonDict:
    """Run the deterministic LNS fixture and compute readiness metrics."""

    instances = build_lns_instances()
    rows = [run_lns_instance(instance) for instance in instances]
    counts = instance_class_counts(instances)
    baseline_results = _classical_baseline_results(rows)
    unsafe_false_accepts = count_unsafe_false_accepts(rows)
    correctness_preserved = all(row["solver_correctness_preserved"] for row in rows)
    ready = _fixture_ready(counts, rows, baseline_results, correctness_preserved, unsafe_false_accepts)
    return {
        "per_instance_results": rows,
        "instance_class_counts": counts,
        "destroy_repair_telemetry": [row["telemetry"] for row in rows],
        "classical_baseline_results": baseline_results,
        "solver_correctness_preserved": correctness_preserved,
        "unsafe_false_accepts": unsafe_false_accepts,
        "constraint_lns_fixture_ready": ready,
        "constraint_lns_fixture_ready_principle": _ready_principle(
            ready,
            counts,
            correctness_preserved,
            unsafe_false_accepts,
        ),
    }


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the validated Exp 5299 terminal artifact."""

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
        "constraint_lns_fixture_ready": benchmark["constraint_lns_fixture_ready"],
        "constraint_lns_fixture_ready_principle": benchmark[
            "constraint_lns_fixture_ready_principle"
        ],
        "instance_class_counts": wrap_field(
            "instance_class_counts",
            benchmark["instance_class_counts"],
        ),
        "destroy_repair_telemetry": wrap_field(
            "destroy_repair_telemetry",
            benchmark["destroy_repair_telemetry"],
        ),
        "classical_baseline_results": wrap_field(
            "classical_baseline_results",
            benchmark["classical_baseline_results"],
        ),
        "solver_correctness_preserved": wrap_field(
            "solver_correctness_preserved",
            benchmark["solver_correctness_preserved"],
        ),
        "unsafe_false_accepts": wrap_field(
            "unsafe_false_accepts",
            benchmark["unsafe_false_accepts"],
        ),
        "tests_run": [dict(row) for row in tests_run or []],
        "per_instance_results": benchmark["per_instance_results"],
        "constraint_group_metadata": list(constraint_group_metadata()),
        "source_artifacts": [
            str(v5278.RESULT_RELATIVE_PATH),
            str(cdcl.RESULT_RELATIVE_PATH),
        ],
        "claim_limits": [
            "offline deterministic LNS fixture only",
            "Minisat22 CDCL remains authoritative for candidate and final correctness",
            "malformed and semantic-wrong structured repair controls are rejected",
            "classical baseline is solver-only fallback over the same CNF",
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
    """Fail closed when the Exp 5299 artifact drifts from its contract."""

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
    _require("usable" in verdict, "honest_verdict must state whether fixture is usable")
    _require(
        artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE,
        f"inference_substrate must be {INFERENCE_SUBSTRATE}",
    )
    _require(
        isinstance(artifact["constraint_lns_fixture_ready"], bool),
        "constraint_lns_fixture_ready must be a bare bool",
    )
    _require(
        isinstance(artifact["constraint_lns_fixture_ready_principle"], str)
        and artifact["constraint_lns_fixture_ready_principle"],
        "constraint_lns_fixture_ready_principle must be non-empty",
    )
    _require(isinstance(artifact["tests_run"], list), "tests_run must be list")
    _require(
        artifact["unsafe_false_accepts"]["value"] == 0,
        "unsafe false accepts must be zero",
    )
    _require(
        artifact["solver_correctness_preserved"]["value"] is True,
        "solver correctness must be preserved",
    )

    counts = artifact["instance_class_counts"]["value"]
    _require(
        all(counts.get(instance_class, 0) > 0 for instance_class in INSTANCE_CLASS_ORDER),
        "all instance classes must be present",
    )
    baseline = artifact["classical_baseline_results"]["value"]
    _require(
        baseline.get("baseline_name") == "solver_only_fallback",
        "solver-only baseline missing",
    )
    _require(
        baseline.get("all_baseline_models_valid") is True,
        "baseline models must validate",
    )
    _require(
        len(artifact["destroy_repair_telemetry"]["value"]) == sum(counts.values()),
        "telemetry count must match instances",
    )
    if artifact["constraint_lns_fixture_ready"]:
        _require(
            artifact["unsafe_false_accepts"]["value"] == 0,
            "ready fixture requires zero unsafe false accepts",
        )
    _require("REQ-VERIFY-5299" in artifact["spec_refs"], "spec refs must include REQ-VERIFY-5299")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum drift")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5299 JSON artifact and return the validated payload."""

    artifact = build_artifact(duration_s=duration_s, tests_run=tests_run)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _candidate(
    candidate_id: str,
    assignments: Mapping[str, int],
    *,
    safety_negative: bool,
    expected_behavior: str,
) -> RepairCandidate:
    return RepairCandidate(
        candidate_id=candidate_id,
        operator="structured_domain_assignment",
        payload={
            "format_version": CANDIDATE_FORMAT_VERSION,
            "assignments": dict(assignments),
        },
        safety_negative=safety_negative,
        expected_behavior=expected_behavior,
    )


def _destroyed_neighborhood(
    instance: LnsInstance,
    boundary: v5278.BoundaryInstance,
) -> tuple[list[str], list[int]]:
    if instance.destroy_operator == "destroy_none_noop":
        return [], []
    return list(boundary.bit_order), list(range(len(instance.clauses)))


def _candidate_assumptions(
    candidate: RepairCandidate,
    boundary: v5278.BoundaryInstance,
) -> tuple[int, ...]:
    assignments = candidate.payload["assignments"]
    bits = boundary.assignment_to_bits(dict(assignments))
    return tuple(index + 1 for index, bit in enumerate(bits) if bit)


def _solver_only_baseline(instance: LnsInstance) -> JsonDict:
    result = cdcl.run_cdcl(instance.clauses, n_vars=instance.n_vars)
    model_valid = result.status == "unsat" or verify_model(instance.clauses, result.model)
    return {
        "baseline_name": "solver_only_fallback",
        "status": result.status,
        "model": list(result.model),
        "model_valid": model_valid,
        "solver_counters": _solver_counters(result.metrics),
    }


def _classical_baseline_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "baseline_name": "solver_only_fallback",
        "instance_count": len(rows),
        "all_baseline_models_valid": all(row["baseline"]["model_valid"] for row in rows),
        "lns_matches_baseline_count": sum(
            1 for row in rows if row["baseline"]["status"] == row["final"]["status"]
        ),
        "per_instance": {
            row["instance_class"]: {
                "status": row["baseline"]["status"],
                "model_valid": row["baseline"]["model_valid"],
                "solver_counters": row["baseline"]["solver_counters"],
            }
            for row in rows
        },
    }


def _fixture_ready(
    counts: Mapping[str, int],
    rows: Sequence[Mapping[str, Any]],
    baseline_results: Mapping[str, Any],
    correctness_preserved: bool,
    unsafe_false_accepts: int,
) -> bool:
    decisions = {
        row["instance_class"]: row["repair"]["solver_decision"]
        for row in rows
    }
    return bool(
        all(counts.get(instance_class, 0) > 0 for instance_class in INSTANCE_CLASS_ORDER)
        and decisions["aligned_repair"] == "accepted"
        and decisions["neutral_noop_repair"] == "accepted"
        and decisions["misleading_repair"] == "rejected"
        and decisions["malformed_control"] == "rejected"
        and decisions["semantic_wrong_control"] == "rejected"
        and baseline_results["all_baseline_models_valid"]
        and correctness_preserved
        and unsafe_false_accepts == 0
    )


def _ready_principle(
    ready: bool,
    counts: Mapping[str, int],
    correctness_preserved: bool,
    unsafe_false_accepts: int,
) -> str:
    if ready:
        return (
            "ready: deterministic destroy/repair telemetry covers aligned, misleading, "
            "neutral, malformed, and semantic-wrong classes; solver-only baseline "
            "matches final labels; unsafe_false_accepts=0 for exp5300."
        )
    missing = ",".join(
        instance_class
        for instance_class in INSTANCE_CLASS_ORDER
        if counts.get(instance_class, 0) == 0
    )
    return (  # pragma: no cover - current fixture is ready; validation tests cover fail-closed schema.
        "blocked: missing_classes="
        + missing
        + f"; correctness_preserved={correctness_preserved}; "
        + f"unsafe_false_accepts={unsafe_false_accepts}"
    )


def _honest_verdict(benchmark: Mapping[str, Any]) -> str:
    if benchmark["constraint_lns_fixture_ready"]:
        return "complete: constraint-LNS fixture usable for exp5300 solver-repair guidance"
    return "blocked_constraint_lns_unusable: constraint-LNS fixture usable=false"  # pragma: no cover


def _rejection_reason(validation: CandidateValidation) -> str:
    if validation.errors:
        return "; ".join(validation.errors)
    return "solver_rejected_candidate_against_original_cnf"


def _overwrite_count(assumptions: Sequence[int], final_model: Sequence[int]) -> int:
    final_literals = set(final_model)
    return sum(1 for literal in assumptions if literal not in final_literals)


def _solver_counters(metrics: Mapping[str, Any]) -> JsonDict:
    return {key: int(metrics.get(key, 0)) for key in COUNTER_KEYS}


def _zero_counters() -> JsonDict:
    return {key: 0 for key in COUNTER_KEYS}


def _checksum_payload(benchmark: Mapping[str, Any]) -> str:
    rows = [
        {
            "instance_id": row["instance_id"],
            "instance_class": row["instance_class"],
            "repair": {
                "candidate_format_valid": row["repair"]["candidate_format_valid"],
                "primary_status": row["repair"]["primary_status"],
                "candidate_solver_accepted": row["repair"]["candidate_solver_accepted"],
                "solver_decision": row["repair"]["solver_decision"],
                "assumption_literals": row["repair"]["assumption_literals"],
            },
            "fallback": {
                "used": row["fallback"]["used"],
                "overwrite_count": row["fallback"]["overwrite_count"],
            },
            "final": {
                "status": row["final"]["status"],
                "model_valid": row["final"]["model_valid"],
            },
            "baseline_status": row["baseline"]["status"],
            "solver_correctness_preserved": row["solver_correctness_preserved"],
        }
        for row in benchmark["per_instance_results"]
    ]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "instance_class_counts": benchmark["instance_class_counts"],
        "rows": rows,
        "unsafe_false_accepts": benchmark["unsafe_false_accepts"],
        "constraint_lns_fixture_ready": benchmark["constraint_lns_fixture_ready"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
