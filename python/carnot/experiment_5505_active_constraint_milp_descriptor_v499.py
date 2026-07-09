"""Exp5505 active-constraint MILP/MaxSAT/CSP descriptor rows.

Spec refs: REQ-VERIFY-5505, SCENARIO-VERIFY-5505.

The descriptor lane is a handoff format, not a performance result. This module
builds tiny finite-domain MILP-, MaxSAT-, and CSP-style rows that hardware
receipt tasks can load later. Each row is solved by a local exact enumerator so
the JSON carries expected outputs and partition/update fields without implying
that any board or solver has accelerated the workload.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5499_preference_maxsat_minimal_fixture_v499 as fixture_mod


JsonDict = dict[str, Any]
AssignmentValue = int | str
Assignment = dict[str, AssignmentValue]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5505_active_constraint_milp_descriptor_v499.json")
DESCRIPTOR_RELATIVE_PATH = Path("results/active_constraint_milp_descriptor_5505/descriptors.json")
SCHEMA_RELATIVE_PATH = Path("results/active_constraint_milp_descriptor_5505/schema.json")

SCHEMA = "carnot.experiment_5505.active_constraint_milp_descriptor.v499"
DESCRIPTOR_SCHEMA = "carnot.descriptor.active_constraint_milp_maxsat_csp.v1"
EXPERIMENT = 5505
EXPERIMENT_ID = "exp5505-active-constraint-milp-descriptor-v499"
MILESTONE = "2026.07.499"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5505
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
EXACT_SOLVER_NAME = "deterministic_finite_domain_enumerator"
SPEC_REFS = ("REQ-VERIFY-5505", "SCENARIO-VERIFY-5505", "REQ-VERIFY-5499")
TERMINAL_PREFIXES = ("complete:", "blocked:")
HARDWARE_TARGETS = ("cpu", "cuda", "kv260", "gatemate", "polarfire")
DESCRIPTOR_STYLES = ("milp", "maxsat", "csp")
TEST_PATHS = ("tests/python/test_experiment_5505_active_constraint_milp_descriptor_v499.py",)

FIELD_PRINCIPLES: dict[str, str] = {
    "descriptor_paths": (
        "points to executable descriptor payloads rather than relying on embedded prose."
    ),
    "schema_paths": "points to the schema contract hardware receipt tasks should validate.",
    "test_paths": "identifies the REQ/SCENARIO tests for this descriptor lane.",
    "num_descriptor_rows": "bounds row coverage and prevents silent descriptor loss.",
    "milp_style_rows": "confirms finite-domain MILP-style rows are present.",
    "maxsat_style_rows": "confirms Exp5499-derived hard/soft Preference-MaxSAT rows are present.",
    "csp_style_rows": "confirms relational finite-domain CSP rows are present.",
    "exact_fallback_agreement_rate": "keeps exact fallback as final authority.",
    "partition_update_fields_present": "guards downstream board receipt compatibility.",
    "descriptor_ready_for_hardware": (
        "downstream gate for CPU/CUDA/KV260/GateMate/PolarFire receipt tasks."
    ),
    "hardware_speedup_claim": ("must remain false without authenticated matched timing evidence."),
    "inference_substrate": (
        "declares verifier checks over cached/exact candidates rather than live model inference."
    ),
    "honest_verdict": (
        "terminal status; start with complete: or blocked: and avoid solver or hardware overclaim."
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_DESCRIPTOR_FIELDS = (
    "descriptor_id",
    "descriptor_style",
    "schema",
    "source_artifact",
    "typed_variables",
    "domains",
    "hard_constraints",
    "soft_preferences",
    "objective_weights",
    "update_schedule",
    "partition_id",
    "partition_update",
    "admissible_hardware_mapping",
    "expected_outputs",
    "exact_fallback",
    "status",
)


def canonical_json(payload: Any) -> str:
    """Serialize JSON deterministically so hashes change only when content changes."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(payload: Any) -> str:
    """Hash a JSON-compatible payload after deterministic serialization."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_schema() -> JsonDict:
    """Return the compact schema contract written beside the descriptor payload."""

    return {
        "schema": "carnot.schema.active_constraint_descriptor_bundle.v1",
        "descriptor_schema": DESCRIPTOR_SCHEMA,
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "required_descriptor_fields": list(REQUIRED_DESCRIPTOR_FIELDS),
        "descriptor_styles": list(DESCRIPTOR_STYLES),
        "constraint_types": [
            "all_different",
            "clause",
            "linear_eq",
            "linear_ge",
            "linear_le",
            "not_equal",
        ],
        "preference_types": ["linear_reward", "pairwise_difference_reward", "value_reward"],
        "hardware_targets": list(HARDWARE_TARGETS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "hardware_speedup_claim_allowed": False,
    }


def build_descriptors() -> list[JsonDict]:
    """Build all executable descriptor rows for the Exp5505 deliverable."""

    descriptors: list[JsonDict] = []
    descriptors.extend(build_milp_descriptors())
    descriptors.extend(build_maxsat_descriptors())
    descriptors.extend(build_csp_descriptors())
    for descriptor in descriptors:
        validate_descriptor(descriptor)
    return descriptors


def build_milp_descriptors() -> list[JsonDict]:
    """Build small finite-domain MILP-style rows with linear hard constraints."""

    rows = [
        {
            "descriptor_id": "milp:bounded_shipping_mix",
            "partition_id": "milp_shipping_partition",
            "variables": [
                _variable("ship_a", "integer", [0, 1, 2, 3], "partition_control"),
                _variable("ship_b", "integer", [0, 1, 2], "partition_control"),
                _variable("expedite", "binary", [0, 1], "boundary_context"),
            ],
            "hard_constraints": [
                _linear("HC_DEMAND_MIN", "linear_ge", {"ship_a": 1, "ship_b": 1, "expedite": 1}, 3),
                _linear(
                    "HC_CAPACITY_MAX", "linear_le", {"ship_a": 2, "ship_b": 3, "expedite": 2}, 7
                ),
                _linear("HC_EXPEDITE_NEEDS_B", "linear_le", {"expedite": 1, "ship_b": -1}, 0),
            ],
            "soft_preferences": [
                _linear_reward("SP_SHIPMENT_VALUE", {"ship_a": 4, "ship_b": 7, "expedite": 1}, 1),
                _value_reward("SP_AVOID_EXPEDITE", "expedite", 0, 1),
            ],
            "partition_scope": ["ship_a", "ship_b"],
            "boundary_variables": ["expedite"],
        },
        {
            "descriptor_id": "milp:batch_allocation_no_speedup",
            "partition_id": "milp_batch_partition",
            "variables": [
                _variable("route_cpu", "binary", [0, 1], "partition_control"),
                _variable("route_accel", "binary", [0, 1], "partition_control"),
                _variable("tiles", "integer", [1, 2, 3], "boundary_context"),
            ],
            "hard_constraints": [
                _linear("HC_ONE_ROUTE", "linear_eq", {"route_cpu": 1, "route_accel": 1}, 1),
                _linear("HC_TILE_BUDGET", "linear_le", {"tiles": 2, "route_accel": 2}, 7),
                _linear("HC_MIN_TILES", "linear_ge", {"tiles": 1}, 2),
            ],
            "soft_preferences": [
                _linear_reward("SP_TILE_COVERAGE", {"tiles": 5}, 1),
                _value_reward("SP_ACCEL_ADVISORY", "route_accel", 1, 3),
                _value_reward("SP_CPU_BASELINE", "route_cpu", 1, 1),
            ],
            "partition_scope": ["route_cpu", "route_accel"],
            "boundary_variables": ["tiles"],
        },
    ]
    return [
        build_descriptor(
            descriptor_style="milp",
            source_artifact="built_in_finite_domain_milp_rows",
            source_instance_id=row["descriptor_id"],
            typed_variables=row["variables"],
            hard_constraints=row["hard_constraints"],
            soft_preferences=row["soft_preferences"],
            partition_id=row["partition_id"],
            partition_scope=row["partition_scope"],
            boundary_variables=row["boundary_variables"],
            update_schedule_type="board_neutral_milp_batch",
            descriptor_id=row["descriptor_id"],
        )
        for row in rows
    ]


def build_maxsat_descriptors() -> list[JsonDict]:
    """Build MaxSAT-style descriptors directly from the Exp5499 exact fixture."""

    descriptors: list[JsonDict] = []
    fixture = fixture_mod.build_fixture()
    fixture_mod.validate_fixture(fixture)
    for index, instance in enumerate(fixture["instances"]):
        typed_variables = [
            _variable(
                str(claim["name"]),
                "categorical",
                [str(value) for value in claim["domain"]],
                "partition_control",
            )
            for claim in instance["typed_claims"]
        ]
        reference = fixture_mod.solve_reference(instance)
        expected = _expected_from_reference(reference)
        descriptor = build_descriptor(
            descriptor_style="maxsat",
            source_artifact=fixture_mod.RESULT_RELATIVE_PATH.as_posix(),
            source_instance_id=str(instance["instance_id"]),
            typed_variables=typed_variables,
            hard_constraints=[dict(row) for row in instance["hard_constraints"]],
            soft_preferences=[dict(row) for row in instance["soft_preferences"]],
            partition_id=f"maxsat_claim_partition_{index}",
            partition_scope=[str(item["name"]) for item in instance["typed_claims"]],
            boundary_variables=[],
            update_schedule_type="board_neutral_maxsat_batch",
            descriptor_id=f"maxsat:exp5499:{instance['instance_id']}",
            expected_outputs=expected,
            exp5499_reference=reference,
        )
        descriptors.append(descriptor)
    return descriptors


def build_csp_descriptors() -> list[JsonDict]:
    """Build relational finite-domain CSP rows with advisory soft preferences."""

    rows = [
        {
            "descriptor_id": "csp:micro_schedule_all_different",
            "partition_id": "csp_schedule_partition",
            "variables": [
                _variable("task_a", "integer", [0, 1, 2], "partition_control"),
                _variable("task_b", "integer", [0, 1, 2], "partition_control"),
                _variable("task_c", "integer", [0, 1, 2], "boundary_context"),
            ],
            "hard_constraints": [
                {
                    "id": "HC_DISTINCT_TASKS",
                    "type": "all_different",
                    "variables": ["task_a", "task_b", "task_c"],
                },
                {"id": "HC_A_NOT_C", "type": "not_equal", "left": "task_a", "right": "task_c"},
            ],
            "soft_preferences": [
                _value_reward("SP_A_EARLY", "task_a", 0, 3),
                _value_reward("SP_B_MIDDLE", "task_b", 1, 2),
                _value_reward("SP_C_LATE", "task_c", 2, 2),
            ],
            "partition_scope": ["task_a", "task_b"],
            "boundary_variables": ["task_c"],
        },
        {
            "descriptor_id": "csp:boundary_coloring",
            "partition_id": "csp_coloring_partition",
            "variables": [
                _variable(
                    "color_left", "categorical", ["red", "green", "blue"], "partition_control"
                ),
                _variable(
                    "color_mid", "categorical", ["red", "green", "blue"], "partition_control"
                ),
                _variable(
                    "color_right", "categorical", ["red", "green", "blue"], "boundary_context"
                ),
            ],
            "hard_constraints": [
                {
                    "id": "HC_LEFT_MID_DIFF",
                    "type": "not_equal",
                    "left": "color_left",
                    "right": "color_mid",
                },
                {
                    "id": "HC_MID_RIGHT_DIFF",
                    "type": "not_equal",
                    "left": "color_mid",
                    "right": "color_right",
                },
                {
                    "id": "HC_LEFT_NOT_BLUE",
                    "type": "clause",
                    "literals": [
                        {"variable": "color_left", "equals": "red"},
                        {"variable": "color_left", "equals": "green"},
                    ],
                },
            ],
            "soft_preferences": [
                _value_reward("SP_LEFT_RED", "color_left", "red", 3),
                _value_reward("SP_MID_GREEN", "color_mid", "green", 3),
                _value_reward("SP_RIGHT_RED", "color_right", "red", 2),
                {
                    "id": "SP_EDGE_DIVERSITY_ADVISORY",
                    "type": "pairwise_difference_reward",
                    "left": "color_left",
                    "right": "color_right",
                    "weight": 1,
                },
            ],
            "partition_scope": ["color_left", "color_mid"],
            "boundary_variables": ["color_right"],
        },
    ]
    return [
        build_descriptor(
            descriptor_style="csp",
            source_artifact="built_in_finite_domain_csp_rows",
            source_instance_id=row["descriptor_id"],
            typed_variables=row["variables"],
            hard_constraints=row["hard_constraints"],
            soft_preferences=row["soft_preferences"],
            partition_id=row["partition_id"],
            partition_scope=row["partition_scope"],
            boundary_variables=row["boundary_variables"],
            update_schedule_type="board_neutral_csp_batch",
            descriptor_id=row["descriptor_id"],
        )
        for row in rows
    ]


def build_descriptor(
    *,
    descriptor_style: str,
    source_artifact: str,
    source_instance_id: str,
    typed_variables: Sequence[Mapping[str, Any]],
    hard_constraints: Sequence[Mapping[str, Any]],
    soft_preferences: Sequence[Mapping[str, Any]],
    partition_id: str,
    partition_scope: Sequence[str],
    boundary_variables: Sequence[str],
    update_schedule_type: str,
    descriptor_id: str,
    expected_outputs: Mapping[str, Any] | None = None,
    exp5499_reference: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble one descriptor and attach its exact fallback output."""

    domains = {str(variable["name"]): list(variable["domain"]) for variable in typed_variables}
    base: JsonDict = {
        "schema": DESCRIPTOR_SCHEMA,
        "descriptor_id": descriptor_id,
        "descriptor_style": descriptor_style,
        "source_artifact": source_artifact,
        "source_instance_id": source_instance_id,
        "typed_variables": [dict(variable) for variable in typed_variables],
        "domains": domains,
        "hard_constraints": [dict(row) for row in hard_constraints],
        "soft_preferences": [dict(row) for row in soft_preferences],
        "objective_weights": objective_weights(soft_preferences),
        "update_schedule": {
            "type": update_schedule_type,
            "update_count": len(hard_constraints) + len(soft_preferences),
            "board_neutral": True,
        },
        "partition_id": partition_id,
        "admissible_hardware_mapping": {
            "advisory_only": True,
            "admissible_targets": list(HARDWARE_TARGETS),
            "board_timing_collected": False,
            "speedup_claim_allowed": False,
            "solver_speedup_claim_allowed": False,
            "mapping_note": "bounded descriptor workload only; no board timing or speedup claim",
        },
        "status": "pending_exact_fallback",
    }
    exact = solve_descriptor_exact(base)
    expected = (
        dict(expected_outputs) if expected_outputs is not None else _expected_from_exact(exact)
    )
    partition_update = build_partition_update(
        descriptor_id=descriptor_id,
        partition_id=partition_id,
        partition_scope=partition_scope,
        boundary_variables=boundary_variables,
        update_schedule=base["update_schedule"],
        descriptor_inputs={
            "typed_variables": base["typed_variables"],
            "hard_constraints": base["hard_constraints"],
            "soft_preferences": base["soft_preferences"],
            "objective_weights": base["objective_weights"],
        },
        expected_outputs=expected,
    )
    fallback = {
        "required": True,
        "complete": True,
        "solver": EXACT_SOLVER_NAME,
        "status": exact["status"],
        "solution": exact["solution"],
        "objective_score": exact["objective_score"],
        "solution_hash": exact["solution_hash"],
        "feasible_assignment_count": exact["feasible_assignment_count"],
        "agreement_with_expected": _outputs_agree(exact, expected),
    }
    base["partition_update"] = partition_update
    base["expected_outputs"] = expected
    base["exact_fallback"] = fallback
    if exp5499_reference is not None:
        base["exp5499_reference"] = dict(exp5499_reference)
    base["status"] = "ready" if fallback["agreement_with_expected"] else "blocked"
    validate_descriptor(base)
    return base


def build_partition_update(
    *,
    descriptor_id: str,
    partition_id: str,
    partition_scope: Sequence[str],
    boundary_variables: Sequence[str],
    update_schedule: Mapping[str, Any],
    descriptor_inputs: Mapping[str, Any],
    expected_outputs: Mapping[str, Any],
) -> JsonDict:
    """Build board-neutral fields that later receipt tasks can echo and hash."""

    return {
        "partition_id": partition_id,
        "partition_scope": list(partition_scope),
        "boundary_variables": list(boundary_variables),
        "update_schedule_type": update_schedule["type"],
        "update_count": update_schedule["update_count"],
        "descriptor_input_hash": sha256_json(
            {"descriptor_id": descriptor_id, "descriptor_inputs": dict(descriptor_inputs)}
        ),
        "expected_output_hash": sha256_json({"expected_outputs": dict(expected_outputs)}),
        "receipt_targets": list(HARDWARE_TARGETS),
        "target_receipt_fields": {
            target: [
                f"{target}_partition_id",
                f"{target}_input_hash",
                f"{target}_output_hash",
                f"{target}_status",
            ]
            for target in HARDWARE_TARGETS
        },
        "board_neutral": True,
    }


def solve_descriptor_exact(descriptor: Mapping[str, Any]) -> JsonDict:
    """Exhaustively solve the finite-domain descriptor with hard rows first."""

    variable_names = [str(variable["name"]) for variable in descriptor["typed_variables"]]
    domains = {name: list(descriptor["domains"][name]) for name in variable_names}
    feasible: list[JsonDict] = []
    for values in itertools.product(*(domains[name] for name in variable_names)):
        assignment = dict(zip(variable_names, values, strict=True))
        if constraints_satisfied(assignment, descriptor["hard_constraints"]):
            feasible.append(
                {
                    "assignment": assignment,
                    "objective_score": score_feasible_assignment(descriptor, assignment),
                    "solution_hash": solution_hash(assignment),
                }
            )
    if not feasible:
        return {
            "status": "infeasible",
            "solution": None,
            "objective_score": None,
            "solution_hash": None,
            "feasible_assignment_count": 0,
        }
    feasible.sort(
        key=lambda row: (-float(row["objective_score"]), canonical_json(row["assignment"]))
    )
    best = feasible[0]
    return {
        "status": "optimal",
        "solution": best["assignment"],
        "objective_score": best["objective_score"],
        "solution_hash": best["solution_hash"],
        "feasible_assignment_count": len(feasible),
    }


def constraints_satisfied(
    assignment: Mapping[str, AssignmentValue],
    constraints: Sequence[Mapping[str, Any]],
) -> bool:
    """Return true only when all hard rows accept the assignment."""

    return all(constraint_satisfied(assignment, constraint) for constraint in constraints)


def constraint_satisfied(
    assignment: Mapping[str, AssignmentValue],
    constraint: Mapping[str, Any],
) -> bool:
    """Evaluate one portable hard row over a finite-domain assignment."""

    kind = constraint.get("type")
    if kind == "clause":
        return any(
            assignment[str(literal["variable"])] == literal["equals"]
            for literal in constraint["literals"]
        )
    if kind == "linear_le":
        return linear_value(assignment, constraint["terms"]) <= float(constraint["rhs"])
    if kind == "linear_ge":
        return linear_value(assignment, constraint["terms"]) >= float(constraint["rhs"])
    if kind == "linear_eq":
        return linear_value(assignment, constraint["terms"]) == float(constraint["rhs"])
    if kind == "all_different":
        values = [assignment[str(variable)] for variable in constraint["variables"]]
        return len(set(values)) == len(values)
    if kind == "not_equal":
        return assignment[str(constraint["left"])] != assignment[str(constraint["right"])]
    raise ValueError("constraint_type")


def score_feasible_assignment(
    descriptor: Mapping[str, Any],
    assignment: Mapping[str, AssignmentValue],
) -> float:
    """Score soft rows after hard feasibility has already been established."""

    return round(
        sum(
            preference_score(assignment, preference)
            for preference in descriptor["soft_preferences"]
        ),
        6,
    )


def preference_score(
    assignment: Mapping[str, AssignmentValue],
    preference: Mapping[str, Any],
) -> float:
    """Evaluate one advisory soft row without giving it correctness authority."""

    weight = float(preference["weight"])
    kind = preference.get("type")
    if kind == "value_reward":
        return weight if assignment[str(preference["variable"])] == preference["value"] else 0.0
    if kind == "linear_reward":
        return weight * (
            linear_value(assignment, preference["terms"]) + float(preference.get("constant", 0.0))
        )
    if kind == "pairwise_difference_reward":
        return (
            weight
            if assignment[str(preference["left"])] != assignment[str(preference["right"])]
            else 0.0
        )
    raise ValueError("preference_type")


def linear_value(
    assignment: Mapping[str, AssignmentValue],
    terms: Sequence[Mapping[str, Any]],
) -> float:
    """Compute a numeric linear form over integer or binary descriptor variables."""

    return sum(
        float(term["coefficient"]) * float(assignment[str(term["variable"])]) for term in terms
    )


def objective_weights(soft_preferences: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose objective weights explicitly for downstream descriptor consumers."""

    return {str(preference["id"]): float(preference["weight"]) for preference in soft_preferences}


def summarize_descriptors(descriptors: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate descriptor readiness metrics for the terminal artifact."""

    _require(bool(descriptors), "descriptor_rows")
    for descriptor in descriptors:
        validate_descriptor(descriptor)
    style_counts = {
        style: sum(int(row["descriptor_style"] == style) for row in descriptors)
        for style in DESCRIPTOR_STYLES
    }
    agreement_count = sum(int(descriptor_exact_fallback_agrees(row)) for row in descriptors)
    partition_count = sum(
        int(descriptor_partition_update_fields_present(row)) for row in descriptors
    )
    summary = {
        "num_descriptor_rows": len(descriptors),
        "milp_style_rows": style_counts["milp"],
        "maxsat_style_rows": style_counts["maxsat"],
        "csp_style_rows": style_counts["csp"],
        "exact_fallback_agreement_rate": _rate(agreement_count, len(descriptors)),
        "partition_update_fields_present": partition_count == len(descriptors),
    }
    summary["descriptor_ready_for_hardware"] = not readiness_blockers(summary)
    summary["readiness_blockers"] = readiness_blockers(summary)
    return summary


def build_descriptor_payload(descriptors: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the standalone descriptor payload written for hardware receipt tasks."""

    return {
        "schema": DESCRIPTOR_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "descriptor_rows": [dict(row) for row in descriptors],
        "hardware_speedup_claim": False,
        "source_artifacts": sorted({str(row["source_artifact"]) for row in descriptors}),
    }


def build_artifact(
    *,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5505 artifact with required bare fields."""

    descriptors = build_descriptors()
    summary = summarize_descriptors(descriptors)
    ready = bool(summary["descriptor_ready_for_hardware"])
    descriptor_payload = build_descriptor_payload(descriptors)
    schema_payload = build_schema()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "descriptor_paths": [DESCRIPTOR_RELATIVE_PATH.as_posix()],
        "schema_paths": [SCHEMA_RELATIVE_PATH.as_posix()],
        "test_paths": list(TEST_PATHS),
        "num_descriptor_rows": summary["num_descriptor_rows"],
        "milp_style_rows": summary["milp_style_rows"],
        "maxsat_style_rows": summary["maxsat_style_rows"],
        "csp_style_rows": summary["csp_style_rows"],
        "exact_fallback_agreement_rate": summary["exact_fallback_agreement_rate"],
        "partition_update_fields_present": summary["partition_update_fields_present"],
        "descriptor_ready_for_hardware": ready,
        "hardware_speedup_claim": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, summary["readiness_blockers"]),
        "descriptor_rows": descriptors,
        "descriptor_payload_sha256": sha256_json(descriptor_payload),
        "schema_payload_sha256": sha256_json(schema_payload),
        "readiness_blockers": summary["readiness_blockers"],
        "tests_run": [dict(row) for row in tests_run],
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "research_conductor_modified": False,
        "claim_limits": [
            "descriptor rows are deterministic CPU-local finite-domain fixtures",
            "exact fallback is the final authority for optimal and infeasible rows",
            "partition/update fields are board-neutral receipt inputs only",
            "no board timing, solver speed, or hardware speedup claim is made",
        ],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str | None = None,
    descriptor_path: Path | str | None = None,
    schema_path: Path | str | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write descriptor payload, schema payload, and terminal result JSON."""

    artifact = build_artifact(tests_run=tests_run)
    descriptor_payload = build_descriptor_payload(artifact["descriptor_rows"])
    schema_payload = build_schema()
    descriptor_output = (
        Path(descriptor_path)
        if descriptor_path is not None
        else repo_root / DESCRIPTOR_RELATIVE_PATH
    )
    schema_output = (
        Path(schema_path) if schema_path is not None else repo_root / SCHEMA_RELATIVE_PATH
    )
    result_output = (
        Path(result_path) if result_path is not None else repo_root / RESULT_RELATIVE_PATH
    )
    descriptor_output.parent.mkdir(parents=True, exist_ok=True)
    schema_output.parent.mkdir(parents=True, exist_ok=True)
    result_output.parent.mkdir(parents=True, exist_ok=True)
    descriptor_output.write_text(
        json.dumps(descriptor_payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    schema_output.write_text(
        json.dumps(schema_payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    result_output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal result against descriptors and required fields."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        artifact.get("descriptor_paths") == [DESCRIPTOR_RELATIVE_PATH.as_posix()],
        "descriptor_paths",
    )
    _require(artifact.get("schema_paths") == [SCHEMA_RELATIVE_PATH.as_posix()], "schema_paths")
    _require(artifact.get("test_paths") == list(TEST_PATHS), "test_paths")
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(
        str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict"
    )
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    descriptors = artifact.get("descriptor_rows")
    _require(isinstance(descriptors, list) and bool(descriptors), "descriptor_rows")
    summary = summarize_descriptors(descriptors)
    for field in (
        "num_descriptor_rows",
        "milp_style_rows",
        "maxsat_style_rows",
        "csp_style_rows",
        "exact_fallback_agreement_rate",
        "partition_update_fields_present",
        "descriptor_ready_for_hardware",
    ):
        _require(artifact.get(field) == summary[field], field)
    _require(
        artifact.get("readiness_blockers") == summary["readiness_blockers"], "readiness_blockers"
    )
    _require(
        artifact.get("descriptor_payload_sha256")
        == sha256_json(build_descriptor_payload(descriptors)),
        "descriptor_payload_sha256",
    )
    _require(
        artifact.get("schema_payload_sha256") == sha256_json(build_schema()),
        "schema_payload_sha256",
    )
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def validate_descriptor(descriptor: Mapping[str, Any]) -> None:
    """Validate one descriptor row and recompute its exact fallback."""

    for field in REQUIRED_DESCRIPTOR_FIELDS:
        _require(field in descriptor, field)
    _require(descriptor.get("schema") == DESCRIPTOR_SCHEMA, "schema")
    _require(descriptor.get("descriptor_style") in DESCRIPTOR_STYLES, "descriptor_style")
    _require(descriptor.get("status") == "ready", "status")
    _require(bool(descriptor.get("typed_variables")), "typed_variables")
    _require(bool(descriptor.get("hard_constraints")), "hard_constraints")
    _require(bool(descriptor.get("soft_preferences")), "soft_preferences")
    for variable in descriptor["typed_variables"]:
        _require(bool(variable.get("name")) and bool(variable.get("domain")), "typed_variable")
    mapping = descriptor["admissible_hardware_mapping"]
    _require(mapping.get("advisory_only") is True, "advisory_only")
    _require(mapping.get("board_timing_collected") is False, "board_timing_collected")
    _require(mapping.get("speedup_claim_allowed") is False, "speedup_claim_allowed")
    _require(mapping.get("solver_speedup_claim_allowed") is False, "solver_speedup_claim_allowed")
    _require(descriptor_partition_update_fields_present(descriptor), "partition_update")
    exact = solve_descriptor_exact(descriptor)
    fallback = descriptor["exact_fallback"]
    _require(
        fallback.get("required") is True and fallback.get("complete") is True, "exact_fallback"
    )
    _require(fallback.get("solver") == EXACT_SOLVER_NAME, "exact_fallback")
    _require(fallback.get("status") == exact["status"], "exact_fallback")
    _require(fallback.get("solution") == exact["solution"], "exact_fallback")
    _require(fallback.get("objective_score") == exact["objective_score"], "exact_fallback")
    _require(fallback.get("solution_hash") == exact["solution_hash"], "exact_fallback")
    _require(descriptor_exact_fallback_agrees(descriptor), "exact_fallback")
    if descriptor.get("descriptor_style") == "maxsat":
        reference = descriptor["exp5499_reference"]
        _require(reference.get("status") == exact["status"], "exp5499_reference")
        _require(reference.get("assignment") == exact["solution"], "exp5499_reference")
        _require(reference.get("objective_score") == exact["objective_score"], "exp5499_reference")


def descriptor_exact_fallback_agrees(descriptor: Mapping[str, Any]) -> bool:
    """Return whether exact fallback, expected output, and source reference agree."""

    exact = solve_descriptor_exact(descriptor)
    expected = descriptor["expected_outputs"]
    fallback = descriptor["exact_fallback"]
    if not _outputs_agree(exact, expected):
        return False
    if not _outputs_agree(exact, fallback):
        return False
    if descriptor.get("descriptor_style") == "maxsat":
        reference = descriptor.get("exp5499_reference", {})
        return (
            reference.get("status") == exact["status"]
            and reference.get("assignment") == exact["solution"]
            and reference.get("objective_score") == exact["objective_score"]
        )
    return True


def descriptor_partition_update_fields_present(descriptor: Mapping[str, Any]) -> bool:
    """Check board-neutral partition/update fields for all receipt targets."""

    partition_update = descriptor.get("partition_update")
    if not isinstance(partition_update, Mapping):
        return False
    required = (
        "partition_id",
        "partition_scope",
        "boundary_variables",
        "update_schedule_type",
        "update_count",
        "descriptor_input_hash",
        "expected_output_hash",
        "receipt_targets",
        "target_receipt_fields",
        "board_neutral",
    )
    if any(field not in partition_update for field in required):
        return False
    if partition_update.get("receipt_targets") != list(HARDWARE_TARGETS):
        return False
    receipt_fields = partition_update.get("target_receipt_fields")
    return (
        isinstance(receipt_fields, Mapping)
        and set(receipt_fields) == set(HARDWARE_TARGETS)
        and partition_update.get("board_neutral") is True
        and int(partition_update.get("update_count", 0)) > 0
    )


def readiness_blockers(summary: Mapping[str, Any]) -> list[str]:
    """Return precise blockers for the hardware-ready descriptor gate."""

    checks = (
        (int(summary["num_descriptor_rows"]) > 0, "descriptor_rows_missing"),
        (int(summary["milp_style_rows"]) > 0, "milp_style_rows_missing"),
        (int(summary["maxsat_style_rows"]) > 0, "maxsat_style_rows_missing"),
        (int(summary["csp_style_rows"]) > 0, "csp_style_rows_missing"),
        (summary["exact_fallback_agreement_rate"] == 1.0, "exact_fallback_disagreement"),
        (summary["partition_update_fields_present"] is True, "partition_update_fields_missing"),
    )
    return [name for passed, name in checks if not passed]


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return a terminal verdict that avoids solver and hardware overclaim."""

    if ready:
        return (
            "complete: active-constraint MILP/MaxSAT/CSP descriptors are exact-fallback "
            "checked, Exp5499 MaxSAT-aligned, board-neutral for receipt tasks, and carry "
            "no solver or hardware speedup claim"
        )
    return "blocked: active_constraint_milp_descriptors_not_ready_" + "_".join(blockers)


def solution_hash(solution: Mapping[str, AssignmentValue]) -> str:
    """Hash a solved assignment independently from descriptor metadata."""

    return sha256_json({"solution": dict(solution)})


def _expected_from_exact(exact: Mapping[str, Any]) -> JsonDict:
    return {
        "status": exact["status"],
        "solution": exact["solution"],
        "objective_score": exact["objective_score"],
        "solution_hash": exact["solution_hash"],
    }


def _expected_from_reference(reference: Mapping[str, Any]) -> JsonDict:
    return {
        "status": reference["status"],
        "solution": reference["assignment"],
        "objective_score": reference["objective_score"],
        "solution_hash": (
            solution_hash(reference["assignment"])
            if isinstance(reference.get("assignment"), Mapping)
            else None
        ),
    }


def _outputs_agree(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        observed.get("status") == expected.get("status")
        and observed.get("solution") == expected.get("solution")
        and observed.get("objective_score") == expected.get("objective_score")
        and observed.get("solution_hash") == expected.get("solution_hash")
    )


def _variable(
    name: str,
    variable_type: str,
    domain: Sequence[AssignmentValue],
    role: str,
) -> JsonDict:
    return {
        "name": name,
        "variable_type": variable_type,
        "domain": list(domain),
        "role": role,
    }


def _linear(
    constraint_id: str,
    kind: str,
    coefficients: Mapping[str, float],
    rhs: float,
) -> JsonDict:
    return {
        "id": constraint_id,
        "type": kind,
        "terms": [
            {"variable": variable, "coefficient": coefficient}
            for variable, coefficient in coefficients.items()
        ],
        "rhs": rhs,
    }


def _linear_reward(
    preference_id: str,
    coefficients: Mapping[str, float],
    weight: float,
) -> JsonDict:
    return {
        "id": preference_id,
        "type": "linear_reward",
        "terms": [
            {"variable": variable, "coefficient": coefficient}
            for variable, coefficient in coefficients.items()
        ],
        "weight": weight,
        "constant": 0,
    }


def _value_reward(
    preference_id: str,
    variable: str,
    value: AssignmentValue,
    weight: float,
) -> JsonDict:
    return {
        "id": preference_id,
        "type": "value_reward",
        "variable": variable,
        "value": value,
        "weight": weight,
    }


def _rate(numerator: int | float, denominator: int | float) -> float:
    _require(float(denominator) > 0.0, "rate_denominator")
    return round(float(numerator) / float(denominator), 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
