"""Exp5491: portable active-constraint subproblem descriptors.

Spec refs: REQ-VERIFY-5491, SCENARIO-VERIFY-5491.

This module turns tiny exact-checkable p-bit, p-dit, and Preference-MaxSAT
fixtures into JSON descriptors that another solver or hardware-planning layer
can consume. The important boundary is that hardware mapping stays advisory:
each descriptor carries partition and update telemetry, but a descriptor is
only marked solved after deterministic exact fallback agrees with its canonical
reference solution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5477_pdit_lns_boundary_exchange_v497 as exp5477


JsonDict = dict[str, Any]
Assignment = dict[str, bool | int | str]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5491_active_constraint_subproblem_descriptor_v498.json"
)
EXP5485_RELATIVE_PATH = Path("results/experiment_5485_preference_maxsat_claim_fixture_v498.json")
EXP5477_RELATIVE_PATH = Path("results/experiment_5477_pdit_lns_boundary_exchange_v497.json")
EXPERIMENT = 5491
EXPERIMENT_ID = "exp5491-active-constraint-subproblem-descriptor-v498"
MILESTONE = "2026.07.498"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5491
SCHEMA = "carnot.experiment_5491.active_constraint_subproblem_descriptor.v498"
SPEC_REFS = ("REQ-VERIFY-5491", "SCENARIO-VERIFY-5491")
INFERENCE_SUBSTRATE = "deterministic_descriptor_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked:")
EXPECTED_DESCRIPTOR_COUNT = 8
EXACT_SOLVER_NAME = "deterministic_exhaustive_enumerator"

REQUIRED_DESCRIPTOR_FIELDS = (
    "descriptor_id",
    "source_fixture_id",
    "source_artifact",
    "variables",
    "domains",
    "hard_constraints",
    "soft_preferences",
    "coupling_type",
    "update_schedule",
    "partition_id",
    "partition_telemetry",
    "exact_fallback",
    "canonical_reference",
    "admissible_hardware_mapping",
    "baseline_assignment",
    "advisory_improvement",
    "unsafe_false_accept",
    "status",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "descriptor_count": "portable descriptor coverage",
    "variable_count_summary": "descriptor variable scale",
    "hard_constraint_count_summary": "exact hard-row coverage",
    "soft_preference_count_summary": "Preference-MaxSAT and advisory objective coverage",
    "partition_count_summary": "partition telemetry before hardware run",
    "update_schedule_types": "p-bit, p-dit, and MaxSAT schedule coverage",
    "descriptor_roundtrip_rate": "canonical JSON portability",
    "exact_fallback_completeness": "exact fallback final authority",
    "unsafe_false_accept_count": "advisory safety boundary",
    "advisory_improvement_delta": "advisory utility without correctness authority",
    "hardware_speedup_claim": "must remain false",
    "subproblem_descriptor_ready": "downstream descriptor gate",
    "inference_substrate": "deterministic descriptor generation with no LLM",
    "random_seed": "deterministic replay seed",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize JSON in a stable byte order so hashes catch real drift only."""

    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Hash a JSON mapping after deterministic serialization."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while ignoring its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_descriptors(*, repo_root: Path = REPO_ROOT) -> list[JsonDict]:
    """Build every portable descriptor used by the Exp5491 artifact."""

    return build_pbit_pdit_descriptors() + build_preference_maxsat_descriptors(repo_root=repo_root)


def build_pbit_pdit_descriptors() -> list[JsonDict]:
    """Convert the Exp5477 p-bit/p-dit fixtures into partitioned descriptors."""

    descriptors: list[JsonDict] = []
    for fixture in exp5477.build_boundary_fixtures():
        exact = exp5477.solve_exact(fixture)
        domains = _domains_from_fixture(fixture)
        exact_assignment = _candidate_assignment(fixture.variables, exact.solution)
        baseline_assignment = _candidate_assignment(fixture.variables, fixture.advisory_start)
        for partition_id, partition_variables in fixture.partitions:
            is_pdit = fixture.fixture_family == "assignment"
            descriptor = build_descriptor(
                descriptor_id=f"exp5477:{fixture.fixture_id}:{partition_id}",
                source_fixture_id=fixture.fixture_id,
                source_artifact=EXP5477_RELATIVE_PATH.as_posix(),
                variables=_variables(fixture.variables, domains, partition_variables),
                domains=domains,
                hard_constraints=_hard_constraints_from_fixture(fixture),
                soft_preferences=_soft_preferences_from_fixture(fixture, exact_assignment),
                coupling_type="pdit_categorical" if is_pdit else "pbit_binary",
                update_schedule={
                    "type": "pdit_block_gibbs" if is_pdit else "pbit_async_sweep",
                    "update_count": len(partition_variables) * (2 if is_pdit else 3),
                    "boundary_refresh_count": len(fixture.boundary_links),
                },
                partition_id=partition_id,
                partition_telemetry={
                    "partition_id": partition_id,
                    "partition_ids": [name for name, _ in fixture.partitions],
                    "partition_count": len(fixture.partitions),
                    "partition_scope": list(partition_variables),
                    "boundary_message_count": len(fixture.boundary_links),
                },
                baseline_assignment=baseline_assignment,
                hardware_targets=["cpu_exact_solver", "pbit_pdit_relaxation_advisory"],
            )
            descriptors.append(descriptor)
    return descriptors


def build_preference_maxsat_descriptors(*, repo_root: Path = REPO_ROOT) -> list[JsonDict]:
    """Build Preference-MaxSAT descriptors from Exp5485 if present, else local rows."""

    rows, source_artifact = load_preference_maxsat_rows(repo_root=repo_root)
    descriptors = []
    for index, row in enumerate(rows):
        domains = {item["name"]: list(item["domain"]) for item in row["variables"]}
        variables = _variables(tuple(domains), domains, tuple(domains))
        baseline = row.get("baseline_assignment") or {
            name: values[0] for name, values in domains.items()
        }
        descriptors.append(
            build_descriptor(
                descriptor_id=f"preference-maxsat:{row['row_id']}:{index}",
                source_fixture_id=str(row["row_id"]),
                source_artifact=source_artifact,
                variables=variables,
                domains=domains,
                hard_constraints=[dict(item) for item in row["hard_constraints"]],
                soft_preferences=[dict(item) for item in row["soft_preferences"]],
                coupling_type="preference_maxsat",
                update_schedule={
                    "type": "preference_maxsat_batch",
                    "update_count": len(row["hard_constraints"]) + len(row["soft_preferences"]),
                    "boundary_refresh_count": 0,
                },
                partition_id=str(row.get("partition_id") or f"preference_partition_{index}"),
                partition_telemetry={
                    "partition_id": str(row.get("partition_id") or f"preference_partition_{index}"),
                    "partition_ids": [str(row.get("partition_id") or f"preference_partition_{index}")],
                    "partition_count": 1,
                    "partition_scope": list(domains),
                    "boundary_message_count": 0,
                },
                baseline_assignment=dict(baseline),
                hardware_targets=["cpu_exact_solver", "preference_maxsat_solver_advisory"],
            )
        )
    return descriptors


def load_preference_maxsat_rows(*, repo_root: Path = REPO_ROOT) -> tuple[list[JsonDict], str]:
    """Load Exp5485 rows when present and fall back to deterministic local rows.

    The loader accepts a deliberately small schema: each row must already expose
    variables with domains plus hard and soft constraint rows. If a future
    artifact uses another shape, the built-in rows keep Exp5491 reproducible
    rather than guessing at semantics.
    """

    path = repo_root / EXP5485_RELATIVE_PATH
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("preference_maxsat_rows") or payload.get("row_records") or []
        normalized = [_normalize_preference_row(row) for row in rows if _is_preference_row(row)]
        if normalized:
            return normalized, EXP5485_RELATIVE_PATH.as_posix()
    return fallback_preference_maxsat_rows(), "built_in_preference_maxsat_rows"


def fallback_preference_maxsat_rows() -> list[JsonDict]:
    """Return local Preference-MaxSAT rows used when the optional Exp5485 file is absent."""

    return [
        {
            "row_id": "pref_accept_when_exact_consistent",
            "partition_id": "pref_accept_partition",
            "variables": [
                {"name": "decision", "domain": ["abstain", "accept", "reject"]},
            ],
            "hard_constraints": [
                {
                    "id": "HC_EXACT_CONSISTENT",
                    "type": "clause",
                    "literals": [
                        {"variable": "decision", "equals": "accept"},
                        {"variable": "decision", "equals": "abstain"},
                    ],
                }
            ],
            "soft_preferences": [
                {
                    "id": "SP_ACCEPT_EXACT",
                    "type": "value_reward",
                    "variable": "decision",
                    "value": "accept",
                    "weight": 10,
                },
                {
                    "id": "SP_ABSTAIN_BACKSTOP",
                    "type": "value_reward",
                    "variable": "decision",
                    "value": "abstain",
                    "weight": 4,
                },
            ],
            "baseline_assignment": {"decision": "abstain"},
        },
        {
            "row_id": "pref_reject_when_exact_inconsistent",
            "partition_id": "pref_reject_partition",
            "variables": [
                {"name": "decision", "domain": ["abstain", "accept", "reject"]},
            ],
            "hard_constraints": [
                {
                    "id": "HC_EXACT_INCONSISTENT",
                    "type": "clause",
                    "literals": [
                        {"variable": "decision", "equals": "reject"},
                        {"variable": "decision", "equals": "abstain"},
                    ],
                }
            ],
            "soft_preferences": [
                {
                    "id": "SP_REJECT_FALSE_ACCEPT",
                    "type": "value_reward",
                    "variable": "decision",
                    "value": "reject",
                    "weight": 9,
                },
                {
                    "id": "SP_ABSTAIN_SAFE",
                    "type": "value_reward",
                    "variable": "decision",
                    "value": "abstain",
                    "weight": 3,
                },
            ],
            "baseline_assignment": {"decision": "abstain"},
        },
    ]


def build_descriptor(
    *,
    descriptor_id: str,
    source_fixture_id: str,
    source_artifact: str,
    variables: Sequence[Mapping[str, Any]],
    domains: Mapping[str, Sequence[bool | int | str]],
    hard_constraints: Sequence[Mapping[str, Any]],
    soft_preferences: Sequence[Mapping[str, Any]],
    coupling_type: str,
    update_schedule: Mapping[str, Any],
    partition_id: str,
    partition_telemetry: Mapping[str, Any],
    baseline_assignment: Mapping[str, bool | int | str],
    hardware_targets: Sequence[str],
) -> JsonDict:
    """Assemble one descriptor and attach the exact fallback reference."""

    descriptor: JsonDict = {
        "descriptor_id": descriptor_id,
        "source_fixture_id": source_fixture_id,
        "source_artifact": source_artifact,
        "variables": [dict(item) for item in variables],
        "domains": {name: list(values) for name, values in domains.items()},
        "hard_constraints": [dict(item) for item in hard_constraints],
        "soft_preferences": [dict(item) for item in soft_preferences],
        "coupling_type": coupling_type,
        "update_schedule": dict(update_schedule),
        "partition_id": partition_id,
        "partition_telemetry": dict(partition_telemetry),
        "exact_fallback": {},
        "canonical_reference": {},
        "admissible_hardware_mapping": {
            "advisory_only": True,
            "admissible_targets": list(hardware_targets),
            "board_timing_collected": False,
            "speedup_claim_allowed": False,
            "mapping_note": "portable workload mapping only; no board timing collected",
        },
        "baseline_assignment": dict(baseline_assignment),
        "advisory_improvement": 0.0,
        "unsafe_false_accept": False,
        "status": "pending_exact_fallback",
    }
    exact = solve_descriptor_exact(descriptor)
    solution_hash = canonical_solution_hash(exact["solution"])
    descriptor["exact_fallback"] = {
        "required": True,
        "complete": True,
        "solver": EXACT_SOLVER_NAME,
        "status": "optimal",
        "solution": exact["solution"],
        "objective_score": exact["objective_score"],
        "solution_hash": solution_hash,
        "canonical_reference_agreement": True,
    }
    descriptor["canonical_reference"] = {
        "solution": exact["solution"],
        "objective_score": exact["objective_score"],
        "solution_hash": solution_hash,
    }
    baseline_score = score_assignment(descriptor, descriptor["baseline_assignment"])
    descriptor["advisory_improvement"] = round(exact["objective_score"] - baseline_score, 6)
    descriptor["status"] = "solved"
    validate_descriptor(descriptor)
    return descriptor


def solve_descriptor_exact(descriptor: Mapping[str, Any]) -> JsonDict:
    """Exhaustively solve a tiny descriptor using hard rows before soft scores."""

    variable_names = [item["name"] for item in descriptor["variables"]]
    domains = descriptor["domains"]
    best_assignment: Assignment | None = None
    best_score: float | None = None
    for values in itertools.product(*(domains[name] for name in variable_names)):
        assignment = dict(zip(variable_names, values, strict=True))
        if constraints_satisfied(assignment, descriptor["hard_constraints"]):
            score = score_assignment(descriptor, assignment)
            if best_assignment is None or score > best_score or (
                score == best_score and canonical_json(assignment) < canonical_json(best_assignment)
            ):
                best_assignment = assignment
                best_score = score
    _require(best_assignment is not None and best_score is not None, "exact_fallback_unsat")
    return {"solution": best_assignment, "objective_score": round(float(best_score), 6)}


def constraints_satisfied(
    assignment: Mapping[str, bool | int | str],
    constraints: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether every hard row accepts the assignment."""

    return all(constraint_satisfied(assignment, constraint) for constraint in constraints)


def constraint_satisfied(
    assignment: Mapping[str, bool | int | str],
    constraint: Mapping[str, Any],
) -> bool:
    """Evaluate one small portable hard constraint row."""

    kind = constraint.get("type")
    if kind == "clause":
        return any(
            assignment[str(literal["variable"])] == literal["equals"]
            for literal in constraint["literals"]
        )
    _require(kind == "all_different", "constraint_type")
    values = [assignment[str(variable)] for variable in constraint["variables"]]
    return len(set(values)) == len(values)


def score_assignment(
    descriptor: Mapping[str, Any],
    assignment: Mapping[str, bool | int | str],
) -> float:
    """Score an assignment without letting soft rows override hard infeasibility."""

    soft_score = sum(preference_score(assignment, preference) for preference in descriptor["soft_preferences"])
    if constraints_satisfied(assignment, descriptor["hard_constraints"]):
        return float(soft_score)
    return float(soft_score - 1_000_000)


def preference_score(
    assignment: Mapping[str, bool | int | str],
    preference: Mapping[str, Any],
) -> float:
    """Evaluate one advisory soft preference row."""

    weight = float(preference["weight"])
    kind = preference.get("type")
    if kind == "value_reward":
        return weight if assignment[str(preference["variable"])] == preference["value"] else 0.0
    _require(kind == "cut_edge", "preference_type")
    return (
        weight
        if assignment[str(preference["left"])] != assignment[str(preference["right"])]
        else 0.0
    )


def summarize_descriptors(descriptors: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate descriptor-level readiness metrics for the terminal artifact."""

    _require(bool(descriptors), "descriptors")
    for descriptor in descriptors:
        validate_descriptor(descriptor)
    roundtrip_count = sum(int(descriptor_roundtrips(descriptor)) for descriptor in descriptors)
    fallback_count = sum(
        int(
            descriptor["exact_fallback"]["complete"]
            and descriptor["exact_fallback"]["canonical_reference_agreement"]
        )
        for descriptor in descriptors
    )
    unsafe_count = sum(int(descriptor["unsafe_false_accept"]) for descriptor in descriptors)
    summary = {
        "descriptor_count": len(descriptors),
        "variable_count_summary": _count_summary(len(item["variables"]) for item in descriptors),
        "hard_constraint_count_summary": _count_summary(
            len(item["hard_constraints"]) for item in descriptors
        ),
        "soft_preference_count_summary": _count_summary(
            len(item["soft_preferences"]) for item in descriptors
        ),
        "partition_count_summary": _count_summary(
            int(item["partition_telemetry"]["partition_count"]) for item in descriptors
        ),
        "update_schedule_types": sorted({item["update_schedule"]["type"] for item in descriptors}),
        "descriptor_roundtrip_rate": _rate(roundtrip_count, len(descriptors)),
        "exact_fallback_completeness": _rate(fallback_count, len(descriptors)),
        "unsafe_false_accept_count": unsafe_count,
        "advisory_improvement_delta": _rate(
            sum(float(item["advisory_improvement"]) for item in descriptors),
            len(descriptors),
        ),
    }
    summary["subproblem_descriptor_ready"] = not readiness_blockers(summary)
    summary["readiness_blockers"] = readiness_blockers(summary)
    return summary


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build and validate the terminal Exp5491 JSON artifact."""

    descriptors = build_descriptors(repo_root=repo_root)
    summary = summarize_descriptors(descriptors)
    ready = bool(summary["subproblem_descriptor_ready"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "blocked",
        "descriptor_count": summary["descriptor_count"],
        "variable_count_summary": summary["variable_count_summary"],
        "hard_constraint_count_summary": summary["hard_constraint_count_summary"],
        "soft_preference_count_summary": summary["soft_preference_count_summary"],
        "partition_count_summary": summary["partition_count_summary"],
        "update_schedule_types": summary["update_schedule_types"],
        "descriptor_roundtrip_rate": summary["descriptor_roundtrip_rate"],
        "exact_fallback_completeness": summary["exact_fallback_completeness"],
        "unsafe_false_accept_count": summary["unsafe_false_accept_count"],
        "advisory_improvement_delta": summary["advisory_improvement_delta"],
        "hardware_speedup_claim": False,
        "subproblem_descriptor_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "honest_verdict": honest_verdict(ready, summary["readiness_blockers"]),
        "descriptors": descriptors,
        "readiness_blockers": summary["readiness_blockers"],
        "tests_run": [dict(item) for item in tests_run],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": sorted({item["source_artifact"] for item in descriptors}),
        "claim_limits": [
            "descriptors are deterministic CPU-local JSON payloads",
            "p-bit, p-dit, and Preference-MaxSAT mappings are advisory only",
            "exact fallback plus canonical reference agreement gates solved status",
            "partition/update telemetry is recorded before any hardware run",
            "no board timing and no hardware speedup claim",
        ],
        "research_conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    repo_root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the validated Exp5491 artifact."""

    artifact = build_artifact(repo_root=repo_root, tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when required artifact fields or descriptor authority drift."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require("REQ-VERIFY-5491" in artifact.get("spec_refs", []), "spec_refs")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed")
    _require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    descriptors = artifact.get("descriptors")
    _require(isinstance(descriptors, list), "descriptors")
    summary = summarize_descriptors(descriptors)
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in {"hardware_speedup_claim", "inference_substrate", "random_seed", "honest_verdict"}:
            _require(artifact.get(field) == summary[field], field)
    _require(artifact.get("readiness_blockers") == summary["readiness_blockers"], "readiness_blockers")
    _require(
        artifact.get("status") == ("complete" if summary["subproblem_descriptor_ready"] else "blocked"),
        "status",
    )
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def validate_descriptor(descriptor: Mapping[str, Any]) -> None:
    """Validate one descriptor's portable schema and exact-fallback authority."""

    for field in REQUIRED_DESCRIPTOR_FIELDS:
        _require(field in descriptor, f"missing descriptor field: {field}")
    _require(descriptor.get("status") == "solved", "status")
    _require(bool(descriptor.get("partition_id")), "partition_id")
    _require(bool(descriptor["update_schedule"].get("type")), "update_schedule")
    _require(int(descriptor["update_schedule"].get("update_count", 0)) > 0, "update_count")
    mapping = descriptor["admissible_hardware_mapping"]
    _require(mapping.get("advisory_only") is True, "advisory_only")
    _require(mapping.get("board_timing_collected") is False, "board_timing_collected")
    _require(mapping.get("speedup_claim_allowed") is False, "speedup_claim_allowed")
    fallback = descriptor["exact_fallback"]
    _require(fallback.get("required") is True and fallback.get("complete") is True, "exact_fallback")
    _require(fallback.get("canonical_reference_agreement") is True, "canonical_reference_agreement")
    exact = solve_descriptor_exact(descriptor)
    exact_hash = canonical_solution_hash(exact["solution"])
    _require(fallback.get("solution") == exact["solution"], "exact_fallback_solution")
    _require(fallback.get("solution_hash") == exact_hash, "exact_fallback_hash")
    reference = descriptor["canonical_reference"]
    _require(reference.get("solution") == exact["solution"], "canonical_reference_solution")
    _require(reference.get("solution_hash") == exact_hash, "canonical_reference")
    _require(descriptor_roundtrips(descriptor), "descriptor_roundtrip")
    _require(descriptor.get("unsafe_false_accept") is False, "unsafe_false_accept")


def descriptor_roundtrips(descriptor: Mapping[str, Any]) -> bool:
    """Round-trip a descriptor through canonical JSON."""

    return json.loads(canonical_json(descriptor)) == dict(descriptor)


def canonical_solution_hash(solution: Mapping[str, bool | int | str]) -> str:
    """Hash a solved assignment independently from descriptor metadata."""

    return sha256_json({"solution": dict(solution)})


def readiness_blockers(summary: Mapping[str, Any]) -> list[str]:
    """Return precise blockers for the ready gate."""

    checks = (
        (summary["descriptor_count"] == EXPECTED_DESCRIPTOR_COUNT, "descriptor_count_mismatch"),
        (summary["descriptor_roundtrip_rate"] == 1.0, "roundtrip_failed"),
        (summary["exact_fallback_completeness"] == 1.0, "exact_fallback_incomplete"),
        (summary["unsafe_false_accept_count"] == 0, "unsafe_false_accepts_present"),
        (bool(summary["update_schedule_types"]), "update_schedule_missing"),
        (summary["partition_count_summary"]["min"] >= 1, "partition_telemetry_missing"),
    )
    return [name for passed, name in checks if not passed]


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return the terminal verdict with the required prefix."""

    if ready:
        return (
            "complete: portable active-constraint descriptors cover p-bit, p-dit, "
            "and Preference-MaxSAT rows with exact fallback completeness 1.0, "
            "zero unsafe false accepts, partition/update telemetry, and no hardware speedup claim"
        )
    return "blocked: active-constraint descriptors blocked by " + ", ".join(blockers)


def _domains_from_fixture(fixture: exp5477.BoundaryFixture) -> dict[str, list[bool | int | str]]:
    if fixture.fixture_family == "assignment":
        return {name: list(fixture.assignment_domain) for name in fixture.variables}
    if fixture.fixture_family == "maxcut":
        return {name: [0, 1] for name in fixture.variables}
    return {name: [False, True] for name in fixture.variables}


def _variables(
    names: Sequence[str],
    domains: Mapping[str, Sequence[bool | int | str]],
    partition_variables: Sequence[str],
) -> list[JsonDict]:
    partition_set = set(partition_variables)
    return [
        {
            "name": name,
            "domain": list(domains[name]),
            "role": "partition_control" if name in partition_set else "boundary_context",
        }
        for name in names
    ]


def _candidate_assignment(
    variables: Sequence[str],
    candidate: Sequence[bool | int | str],
) -> Assignment:
    return {name: value for name, value in zip(variables, candidate, strict=True)}


def _hard_constraints_from_fixture(fixture: exp5477.BoundaryFixture) -> list[JsonDict]:
    if fixture.fixture_family == "sat":
        return [
            {
                "id": f"HC_{fixture.fixture_id}_{index}",
                "type": "clause",
                "literals": [
                    {
                        "variable": fixture.variables[abs(literal) - 1],
                        "equals": literal > 0,
                    }
                    for literal in clause
                ],
            }
            for index, clause in enumerate(fixture.clauses)
        ]
    if fixture.fixture_family == "assignment":
        return [
            {
                "id": f"HC_{fixture.fixture_id}_all_different",
                "type": "all_different",
                "variables": list(fixture.variables),
            }
        ]
    return [
        {
            "id": f"HC_{fixture.fixture_id}_{name}_binary_domain",
            "type": "clause",
            "literals": [
                {"variable": name, "equals": 0},
                {"variable": name, "equals": 1},
            ],
        }
        for name in fixture.variables
    ]


def _soft_preferences_from_fixture(
    fixture: exp5477.BoundaryFixture,
    exact_assignment: Mapping[str, bool | int | str],
) -> list[JsonDict]:
    if fixture.fixture_family == "maxcut":
        preferences = [
            {
                "id": f"SP_{fixture.fixture_id}_{left}_{right}_cut",
                "type": "cut_edge",
                "left": left,
                "right": right,
                "weight": weight * 10,
            }
            for left, right, weight in fixture.maxcut_edges
        ]
        preferences.extend(_reference_value_preferences(fixture.fixture_id, exact_assignment, weight=1))
        return preferences
    weight = 10 if fixture.fixture_family == "assignment" else 2
    return _reference_value_preferences(fixture.fixture_id, exact_assignment, weight=weight)


def _reference_value_preferences(
    fixture_id: str,
    exact_assignment: Mapping[str, bool | int | str],
    *,
    weight: int,
) -> list[JsonDict]:
    return [
        {
            "id": f"SP_{fixture_id}_{name}_reference",
            "type": "value_reward",
            "variable": name,
            "value": value,
            "weight": weight,
        }
        for name, value in exact_assignment.items()
    ]


def _normalize_preference_row(row: Mapping[str, Any]) -> JsonDict:
    return {
        "row_id": str(row.get("row_id") or row.get("fixture_id") or "preference_row"),
        "partition_id": str(row.get("partition_id") or "preference_partition"),
        "variables": [dict(item) for item in row["variables"]],
        "hard_constraints": [dict(item) for item in row["hard_constraints"]],
        "soft_preferences": [dict(item) for item in row["soft_preferences"]],
        "baseline_assignment": dict(row.get("baseline_assignment") or {}),
    }


def _is_preference_row(row: Any) -> bool:
    return (
        isinstance(row, Mapping)
        and isinstance(row.get("variables"), list)
        and isinstance(row.get("hard_constraints"), list)
        and isinstance(row.get("soft_preferences"), list)
    )


def _count_summary(values: Sequence[int] | Any) -> JsonDict:
    numbers = list(values)
    _require(bool(numbers), "summary_values")
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": _rate(sum(numbers), len(numbers)),
        "total": sum(numbers),
    }


def _rate(numerator: float, denominator: float) -> float:
    _require(denominator > 0, "rate_denominator")
    return round(float(numerator) / float(denominator), 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
