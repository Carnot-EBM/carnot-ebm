"""Exp5761 exact constraint-acquisition benchmark.

Spec refs: REQ-BENCH-5761, REQ-LEARN-5761, REQ-STORE-5761,
SCENARIO-BENCH-5761, SCENARIO-BENCH-5761-CONTROLS,
SCENARIO-LEARN-5761, SCENARIO-LEARN-5761-MINIMAL-QUERIES,
SCENARIO-STORE-5761.

This module derives a local, sealed constraint-acquisition fixture from the
Exp5746 exact finite-domain benchmark.  It does not learn from MPMMine, call an
LLM, or use model prose as evidence.  The benchmark exists to test whether a
future learner can notice missing rules, reject spurious over-fit rules, and
produce the expected safe lifecycle repair operation under exact validators.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

from carnot import experiment_5746_exact_proposal_utility_benchmark as exp5746


JsonDict = dict[str, Any]
Probe = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5761_exact_constraint_acquisition_benchmark.json")
BENCHMARK_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_5761_exact_constraint_acquisition_benchmark.instances.jsonl"
)

SCHEMA = "carnot.experiment_5761.exact_constraint_acquisition_benchmark.v1"
MANIFEST_SCHEMA = SCHEMA + ".case"
EXPERIMENT = 5761
EXPERIMENT_ID = "experiment_5761_exact_constraint_acquisition_benchmark"
MILESTONE = "2026.07.514"
RUN_DATE = "20260721"
GENERATOR_VERSION = "exp5761_exact_constraint_acquisition_benchmark_v1"
PRIMARY_SOLVER_VERSION = "carnot_exact_ca_primary_enumerator_v1"
INDEPENDENT_SOLVER_VERSION = "carnot_exact_ca_reversed_domain_validator_v1"
INFERENCE_SUBSTRATE = "deterministic_exact_solver_dataset_generation_no_llm"

REQUIRED_FAMILIES = exp5746.REQUIRED_FAMILIES
SPLITS = exp5746.SPLITS
INSTANCE_COUNT = 120
CASES_PER_SPLIT_FAMILY = 10
VARIANT_KINDS = ("faithful", "incomplete", "overfit", "mixed")
PRODUCER_GATE_FIELDS = (
    "ca_benchmark_ready_score",
    "exact_validator_disagreement_count",
    "train_dev_science_disjoint_score",
)
ADVERSARIAL_CONTROL_TYPES = (
    "duplicate",
    "contradictory",
    "implied_constraint",
    "infeasible_model",
    "vacuous_global",
    "objective_sign",
    "leakage",
    "split_collision",
    "shortcut",
)
SPEC_REFS = (
    "REQ-BENCH-5761",
    "REQ-LEARN-5761",
    "REQ-STORE-5761",
    "SCENARIO-BENCH-5761",
    "SCENARIO-BENCH-5761-CONTROLS",
    "SCENARIO-LEARN-5761",
    "SCENARIO-LEARN-5761-MINIMAL-QUERIES",
    "SCENARIO-STORE-5761",
)
RANDOM_SEEDS: JsonDict = {
    "source_selection_seed": 5761001,
    "mutation_seed": 5761002,
    "query_seed": 5761003,
    "control_seed": 5761004,
    "base_seed": 5761,
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5761_exact_constraint_acquisition_benchmark.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5761_exact_constraint_acquisition_benchmark.py -m pytest tests/python/test_experiment_5761_exact_constraint_acquisition_benchmark.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5761_exact_constraint_acquisition_benchmark.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5761_exact_constraint_acquisition_benchmark.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "spec_refs",
    "upstream_artifact_hashes",
    "generator_version",
    "generator_source_hash",
    "solver_versions",
    "benchmark_manifest_path",
    "benchmark_manifest_hash",
    "instance_count",
    "family_counts",
    "variant_counts",
    "faithful_model_hashes",
    "incomplete_model_hashes",
    "overfit_model_hashes",
    "mixed_model_hashes",
    "domain_artifact_hashes",
    "positive_assignment_count",
    "negative_assignment_count",
    "membership_query_count",
    "distinguishing_query_receipts",
    "expected_repair_receipts",
    "hard_soft_role_receipts",
    "split_manifest",
    "train_dev_science_disjoint_score",
    "science_row_hashes",
    "adversarial_controls",
    "structure_receipt_failure_count",
    "solution_receipt_failure_count",
    "exact_validator_disagreement_count",
    "ca_benchmark_ready_score",
    "producer_gate_fields",
    "llm_inference_used",
    "verifier_is_oracle",
    "inference_substrate",
    "random_seeds",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Pins the Exp5761 artifact schema for downstream validators.",
    "experiment": "Numeric experiment id for result indexing.",
    "experiment_id": "Stable slug for conductor and artifact lookup.",
    "milestone": "Milestone accountability for this benchmark build.",
    "run_date": "Absolute run date 20260721 avoids relative-date ambiguity.",
    "result_path": "Names the JSON artifact written by this workflow.",
    "field_principles": "Every artifact field states the evidence boundary it protects.",
    "status": "Bare terminal state used by downstream gates.",
    "preconditions_checked": "Records upstream replay, solver, provenance, seed, RAM, disk, and reconstruction gates before generation.",
    "spec_refs": "Binds the artifact to REQ-BENCH-5761, REQ-LEARN-5761, and REQ-STORE-5761.",
    "upstream_artifact_hashes": "Seals Exp5746 artifact, manifest, and generator source inputs.",
    "generator_version": "Pins the deterministic Exp5761 generator version.",
    "generator_source_hash": "Seals the Exp5761 source file that generated the rows.",
    "solver_versions": "Pins primary and independent exact validator implementations.",
    "benchmark_manifest_path": "Points to the full sealed acquisition manifest.",
    "benchmark_manifest_hash": "Seals the full manifest bytes.",
    "benchmark_row_hashes": "Seals every selected acquisition case row.",
    "instance_count": "Records the number of faithful source cases.",
    "family_counts": "Proves balanced coverage across finite CSP, MaxSAT, packing, and planning.",
    "variant_counts": "Proves faithful, incomplete, over-fit, and mixed variants exist per case.",
    "faithful_model_hashes": "Seals canonical faithful typed model AST/text by case.",
    "incomplete_model_hashes": "Seals deleted-constraint variants by case.",
    "overfit_model_hashes": "Seals spurious-constraint variants by case.",
    "mixed_model_hashes": "Seals variants that combine deletion and spurious restriction by case.",
    "domain_artifact_hashes": "Seals natural-language, variable-domain, and transition artifacts by case.",
    "positive_assignment_count": "Counts exact accepted assignment witnesses for every variant.",
    "negative_assignment_count": "Counts exact rejected assignment witnesses for every variant.",
    "membership_query_count": "Counts minimal discriminating membership queries for non-faithful variants.",
    "distinguishing_query_receipts": "Seals query hashes and minimality evidence for each variant.",
    "expected_repair_receipts": "Seals expected lifecycle add/remove/noop operations for each variant.",
    "hard_soft_role_receipts": "Proves mutations preserve hard versus soft roles.",
    "split_manifest": "Seals family-stratified train/dev/science membership.",
    "train_dev_science_disjoint_score": "Bare scalar proving selected split memberships are disjoint.",
    "science_row_hashes": "Exposes held-out science case commitments for downstream learner audits.",
    "adversarial_controls": "Records duplicate, contradictory, no-op, infeasible, leakage, split, shortcut, and objective controls.",
    "structure_receipt_failure_count": "Blocks semantic no-op mutations or role drift.",
    "solution_receipt_failure_count": "Blocks unsatisfiable variants or missing assignment witnesses.",
    "exact_validator_disagreement_count": "Bare scalar blocking primary versus independent validator disagreement.",
    "ca_benchmark_ready_score": "Bare scalar that unlocks downstream acquisition consumers only when every exact gate passes.",
    "producer_gate_fields": "Lists top-level bare scalars exported to conductor gates.",
    "llm_inference_used": "Bare false proves no LLM contaminated benchmark generation.",
    "verifier_is_oracle": "Bare true records exact validators as the only correctness authority.",
    "inference_substrate": "Declares deterministic exact solver dataset generation with no LLM.",
    "random_seeds": "Records source selection, mutation, query, and control seeds.",
    "random_seed": "Legacy scalar seed for methodology linters.",
    "model_specs": "Declares the offline exact generator in place of any LLM model.",
    "test_commands": "Records focused, coverage, full-suite, spec, adversarial, and root-clutter commands.",
    "test_exit_codes": "Records observed or preregistered verification command exit codes.",
    "methodology_note": "Explains exact 1.0 gate scores as deterministic replay gates, not statistical classifier claims.",
    "reproducibility_checksum": "Hashes the artifact with its checksum field blanked.",
    "honest_verdict": "Terminal state starts complete: or blocked: and names the readiness boundary.",
    "blocked_reasons": "Lists mechanical blockers when the benchmark is not ready.",
}


class ManifestReplayError(ValueError):
    """Raised when the acquisition manifest no longer matches artifact commitments."""


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file in chunks so replay does not trust metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = 512
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_mb = int(pages * page_size / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": required_mb, "ok": available_mb >= required_mb}


def _disk_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = 512
    usage = shutil.disk_usage(REPO_ROOT)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": required_mb, "ok": available_mb >= required_mb}


def _exp5746_row_hash(row: Mapping[str, Any]) -> str:
    stable = dict(row)
    stable["row_hash"] = ""
    return exp5746.sha256_json(stable)


def _read_json_object(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _load_upstream(
    upstream_artifact_path: str | Path,
    upstream_manifest_path: str | Path,
) -> tuple[JsonDict, list[JsonDict]]:
    artifact = _read_json_object(upstream_artifact_path)
    rows = exp5746.read_benchmark_manifest(upstream_manifest_path)
    exp5746.validate_artifact(artifact)
    exp5746.verify_benchmark_manifest(rows, artifact)
    return artifact, rows


def _reconstruct_faithful_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures: list[JsonDict] = []
    for row in rows:
        pool = exp5746.candidate_pool_receipt(row)
        structure = exp5746.structure_receipt(row)
        solution = exp5746.solution_receipt(row)
        hard = exp5746.hard_constraint_receipt(row, solution)
        soft = exp5746.soft_objective_receipt(row, solution)
        optimum = exp5746.exact_optimum_receipt(row, solution)
        checks = {
            "row_hash": _exp5746_row_hash(row) == row.get("row_hash"),
            "candidate_pool": pool == row.get("candidate_pool_receipt"),
            "structure": structure == row.get("structure_receipt"),
            "solution": solution == row.get("solution_receipt"),
            "hard": hard == row.get("hard_constraint_receipt"),
            "soft": soft == row.get("soft_objective_receipt"),
            "optimum": optimum == row.get("exact_optimum_receipt"),
        }
        if not all(checks.values()):
            failures.append({"source_instance_id": str(row["instance_id"]), "checks": checks})
    return {
        "checked_row_count": len(rows),
        "failure_count": len(failures),
        "failures": failures[:10],
        "ok": not failures,
    }


def collect_preconditions(
    *,
    upstream_artifact_path: str | Path = REPO_ROOT / exp5746.RESULT_RELATIVE_PATH,
    upstream_manifest_path: str | Path = REPO_ROOT / exp5746.BENCHMARK_MANIFEST_RELATIVE_PATH,
    memory_probe: Probe = _memory_probe,
    disk_probe: Probe = _disk_probe,
) -> JsonDict:
    """Collect preconditions before deriving any acquisition rows."""

    artifact_path = Path(upstream_artifact_path)
    manifest_path = Path(upstream_manifest_path)
    memory = memory_probe()
    disk = disk_probe()
    blocked: list[str] = []
    exp5746_replay: JsonDict
    seed_receipt: JsonDict
    family_balance: JsonDict
    faithful_reconstruction: JsonDict
    license_provenance: JsonDict

    try:
        upstream_artifact, upstream_rows = _load_upstream(artifact_path, manifest_path)
        manifest_hash_ok = upstream_artifact.get("benchmark_manifest_hash") == sha256_file(manifest_path)
        artifact_hash = sha256_file(artifact_path)
        family_balance = {
            "family_counts": dict(upstream_artifact.get("family_counts") or {}),
            "expected_family_counts": {
                family: exp5746.INSTANCES_PER_FAMILY for family in REQUIRED_FAMILIES
            },
            "ok": dict(upstream_artifact.get("family_counts") or {})
            == {family: exp5746.INSTANCES_PER_FAMILY for family in REQUIRED_FAMILIES},
        }
        seed_receipt = {
            "upstream_random_seeds": dict(upstream_artifact.get("random_seeds") or {}),
            "expected_upstream_random_seeds": dict(exp5746.RANDOM_SEEDS),
            "exp5761_random_seeds": dict(RANDOM_SEEDS),
            "ok": dict(upstream_artifact.get("random_seeds") or {}) == dict(exp5746.RANDOM_SEEDS),
        }
        faithful_reconstruction = _reconstruct_faithful_receipts(upstream_rows)
        license_path = REPO_ROOT / "LICENSE"
        license_provenance = {
            "license_path": str(license_path),
            "license_hash": sha256_file(license_path) if license_path.exists() else "",
            "local_exp5746_generated": True,
            "mpmmine_result_imported": False,
            "llm_inference_used": upstream_artifact.get("llm_inference_used") is True,
            "ok": license_path.exists() and upstream_artifact.get("llm_inference_used") is False,
        }
        exp5746_replay = {
            "artifact_path": str(artifact_path),
            "manifest_path": str(manifest_path),
            "artifact_hash": artifact_hash,
            "manifest_hash": sha256_file(manifest_path),
            "manifest_hash_ok": manifest_hash_ok,
            "row_count": len(upstream_rows),
            "row_hashes_ok": all(_exp5746_row_hash(row) == row.get("row_hash") for row in upstream_rows),
            "generator_version": upstream_artifact.get("generator_version"),
            "solver_versions": dict(upstream_artifact.get("solver_versions") or {}),
            "benchmark_ready_score": upstream_artifact.get("benchmark_ready_score"),
            "ok": (
                manifest_hash_ok
                and len(upstream_rows) == exp5746.INSTANCE_COUNT
                and upstream_artifact.get("benchmark_ready_score") == 1.0
            ),
        }
    except (OSError, ValueError, exp5746.ManifestReplayError) as exc:
        blocked.append("exp5746_replay_failed")
        exp5746_replay = {
            "artifact_path": str(artifact_path),
            "manifest_path": str(manifest_path),
            "ok": False,
            "error": str(exc),
        }
        seed_receipt = {"ok": False}
        family_balance = {"ok": False}
        faithful_reconstruction = {"checked_row_count": 0, "failure_count": 1, "ok": False}
        license_provenance = {"mpmmine_result_imported": False, "ok": False}

    if memory.get("ok") is not True:
        blocked.append("insufficient_free_ram")
    if disk.get("ok") is not True:
        blocked.append("insufficient_free_disk")
    if exp5746_replay.get("ok") is not True:
        blocked.append("exp5746_replay_failed")
    if family_balance.get("ok") is not True:
        blocked.append("family_balance_failed")
    if seed_receipt.get("ok") is not True:
        blocked.append("deterministic_seed_replay_failed")
    if faithful_reconstruction.get("ok") is not True:
        blocked.append("faithful_model_structure_reconstruction_failed")
    if license_provenance.get("ok") is not True:
        blocked.append("license_or_provenance_failed")

    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "receipt_emitted_before_benchmark_generation": True,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "memory": memory,
        "disk": disk,
        "exact_solvers_available": True,
        "solver_versions": {
            "primary_exact_solver": PRIMARY_SOLVER_VERSION,
            "independent_exact_solver": INDEPENDENT_SOLVER_VERSION,
            "upstream_primary_exact_solver": exp5746.PRIMARY_SOLVER_VERSION,
            "upstream_independent_exact_solver": exp5746.INDEPENDENT_SOLVER_VERSION,
        },
        "exp5746_replay": exp5746_replay,
        "family_balance": family_balance,
        "license_provenance": license_provenance,
        "deterministic_seeds": seed_receipt,
        "faithful_structure_reconstruction": faithful_reconstruction,
        "preconditions_ready": not sorted(set(blocked)),
        "blocked_reasons": sorted(set(blocked)),
    }


def fixture_preconditions() -> JsonDict:
    """Return deterministic preconditions for unit tests using real sealed Exp5746 inputs."""

    return collect_preconditions(
        memory_probe=lambda: {"available_mb": 8192, "required_mb": 512, "ok": True},
        disk_probe=lambda: {"available_mb": 8192, "required_mb": 512, "ok": True},
    )


def _candidate_by_id(source_row: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {str(candidate["candidate_id"]): dict(candidate) for candidate in source_row["candidate_pool"]}


def _literal_satisfied(assignment: Mapping[str, Any], literal: Sequence[Any]) -> bool:
    name, positive = literal
    return bool(assignment[str(name)]) is bool(positive)


def _simulate_plan(start: str, actions: Sequence[str], transitions: Sequence[Mapping[str, Any]]) -> list[str]:
    table = {(str(row["from"]), str(row["action"])): str(row["to"]) for row in transitions}
    states = [start]
    current = start
    for action in actions:
        current = table[(current, str(action))]
        states.append(current)
    return states


def _constraint_holds(
    model_ast: Mapping[str, Any],
    constraint: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> bool:
    kind = str(constraint["type"])
    if kind == "equals":
        return assignment[str(constraint["var"])] == constraint["value"]
    if kind == "not_equal":
        a, b = constraint["vars"]
        return assignment[str(a)] != assignment[str(b)]
    if kind == "clause":
        return any(_literal_satisfied(assignment, literal) for literal in constraint["literals"])
    if kind == "at_most_one":
        return sum(int(assignment[str(name)]) for name in constraint["vars"]) <= 1
    if kind == "capacity":
        variables = [row["name"] for row in model_ast["variables"]]
        return (
            sum(
                int(assignment[str(name)]) * int(weight)
                for name, weight in zip(variables, constraint["weights"], strict=True)
            )
            <= int(constraint["capacity"])
        )
    if kind == "requires_item":
        return int(assignment[str(constraint["var"])]) == 1
    if kind == "not_both":
        a, b = constraint["vars"]
        return not (int(assignment[str(a)]) and int(assignment[str(b)]))
    if kind == "final_state":
        actions = [str(assignment[f"a{i}"]) for i in range(3)]
        return (
            _simulate_plan(str(constraint["start"]), actions, model_ast["transitions"])[-1]
            == str(constraint["goal"])
        )
    if kind == "max_action_count":
        actions = [str(assignment[f"a{i}"]) for i in range(3)]
        return actions.count(str(constraint["action"])) <= int(constraint["limit"])
    if kind == "forbid_assignment":
        return dict(assignment) != dict(constraint["assignment"])
    if kind == "vacuous_global":
        return True
    raise ValueError(f"unsupported constraint type: {kind}")


def _objective_value(
    source_row: Mapping[str, Any],
    model_ast: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> int:
    objective = 0
    for preference in model_ast["soft_preferences"]:
        kind = str(preference["type"])
        if kind == "prefer_equals":
            objective += int(preference["weight"]) if assignment[str(preference["var"])] == preference["value"] else 0
        elif kind == "weighted_clause":
            objective += int(preference["weight"]) if any(_literal_satisfied(assignment, literal) for literal in preference["literals"]) else 0
        elif kind == "linear_utility":
            variables = [row["name"] for row in model_ast["variables"]]
            objective += sum(
                int(assignment[str(name)]) * int(utility)
                for name, utility in zip(variables, preference["utilities"], strict=True)
            )
        elif kind == "light_item_bonus":
            variables = [row["name"] for row in model_ast["variables"]]
            weights = next(row["weights"] for row in model_ast["hard_constraints"] if row["type"] == "capacity")
            objective += int(preference["weight"]) * sum(
                int(assignment[str(name)])
                for name, weight in zip(variables, weights, strict=True)
                if int(weight) <= int(preference["max_weight"])
            )
        elif kind == "action_reward":
            actions = [str(assignment[f"a{i}"]) for i in range(3)]
            objective += int(preference["weight"]) * actions.count(str(preference["action"]))
        elif kind == "action_penalty":
            actions = [str(assignment[f"a{i}"]) for i in range(3)]
            objective -= int(preference["weight"]) * actions.count(str(preference["action"]))
        else:
            raise ValueError(f"unsupported soft preference type: {kind}")
    if source_row["family"] == "finite_domain_csp":
        objective += 1 if assignment["A"] != assignment["C"] else 0
    return objective


def evaluate_model_candidate(
    source_row: Mapping[str, Any],
    model_ast: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> JsonDict:
    """Evaluate one candidate against one typed acquisition model."""

    assignment = candidate["assignment"]
    hard_violations = [
        str(constraint["id"])
        for constraint in model_ast["hard_constraints"]
        if not _constraint_holds(model_ast, constraint, assignment)
    ]
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "feasible": not hard_violations,
        "hard_violations": hard_violations,
        "hard_violation_count": len(hard_violations),
        "objective_value": _objective_value(source_row, model_ast, assignment),
    }


def model_solution_receipt(
    source_row: Mapping[str, Any],
    model_ast: Mapping[str, Any],
    *,
    solver_version: str = PRIMARY_SOLVER_VERSION,
    independent_order: bool = False,
) -> JsonDict:
    """Enumerate the finite candidate pool and summarize model semantics."""

    candidates = [dict(candidate) for candidate in source_row["candidate_pool"]]
    if independent_order:
        candidates = sorted(
            candidates,
            key=lambda candidate: canonical_json(candidate["assignment"]),
            reverse=True,
        )
    evaluations = {
        str(candidate["candidate_id"]): evaluate_model_candidate(source_row, model_ast, candidate)
        for candidate in candidates
    }
    feasible_ids = sorted(
        candidate_id for candidate_id, row in evaluations.items() if row["feasible"] is True
    )
    optimum_value = (
        max(int(evaluations[candidate_id]["objective_value"]) for candidate_id in feasible_ids)
        if feasible_ids
        else None
    )
    optimal_ids = [
        candidate_id
        for candidate_id in feasible_ids
        if optimum_value is not None
        and int(evaluations[candidate_id]["objective_value"]) == optimum_value
    ]
    return {
        "solver_version": solver_version,
        "candidate_count": len(candidates),
        "all_candidates_checked": len(evaluations) == len(source_row["candidate_pool"]),
        "satisfiable": bool(feasible_ids),
        "feasible_candidate_count": len(feasible_ids),
        "feasible_candidate_ids": feasible_ids,
        "feasible_candidate_ids_hash": sha256_json(feasible_ids),
        "optimum_value": optimum_value,
        "optimal_candidate_ids": optimal_ids,
        "optimal_candidate_ids_hash": sha256_json(optimal_ids),
        "objective_values_hash": sha256_json(
            {candidate_id: row["objective_value"] for candidate_id, row in evaluations.items()}
        ),
        "hard_violation_hash": sha256_json(
            {candidate_id: row["hard_violations"] for candidate_id, row in evaluations.items()}
        ),
        "solution_hash": sha256_json(evaluations),
    }


def _model_text(model_ast: Mapping[str, Any]) -> str:
    hard_ids = ", ".join(str(row["id"]) for row in model_ast["hard_constraints"])
    soft_ids = ", ".join(str(row["id"]) for row in model_ast["soft_preferences"])
    return (
        f"family={model_ast['family']}; hard=[{hard_ids}]; "
        f"soft=[{soft_ids}]; objective={canonical_json(model_ast['soft_objective'])}"
    )


def _model_hash(model_ast: Mapping[str, Any], model_text: str) -> str:
    return sha256_json({"model_ast": model_ast, "model_text": model_text})


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _feasible_set(solution: Mapping[str, Any]) -> set[str]:
    return set(str(candidate_id) for candidate_id in solution["feasible_candidate_ids"])


def _assignment_for_candidate(source_row: Mapping[str, Any], candidate_id: str) -> JsonDict:
    return dict(_candidate_by_id(source_row)[candidate_id]["assignment"])


def _candidate_sort_key(candidate_id: str) -> tuple[int, str]:
    suffix = candidate_id.rsplit("-", 1)[-1]
    return (int(suffix) if suffix.isdigit() else 0, candidate_id)


def _deletion_options(source_row: Mapping[str, Any], faithful_ast: Mapping[str, Any]) -> list[JsonDict]:
    faithful_solution = model_solution_receipt(source_row, faithful_ast)
    faithful_set = _feasible_set(faithful_solution)
    options: list[JsonDict] = []
    for index, constraint in enumerate(faithful_ast["hard_constraints"]):
        if constraint["type"] == "capacity":
            continue
        candidate_ast = _copy_json(faithful_ast)
        deleted = candidate_ast["hard_constraints"].pop(index)
        solution = model_solution_receipt(source_row, candidate_ast)
        variant_set = _feasible_set(solution)
        extra = sorted(variant_set - faithful_set, key=_candidate_sort_key)
        if extra and solution["satisfiable"] is True:
            options.append(
                {
                    "model_ast": candidate_ast,
                    "deleted_constraint": dict(deleted),
                    "extra_candidate_id": extra[0],
                    "solution_receipt": solution,
                }
            )
    return options


def _addition_candidates(
    source_row: Mapping[str, Any],
    faithful_ast: Mapping[str, Any],
) -> list[JsonDict]:
    faithful_solution = model_solution_receipt(source_row, faithful_ast)
    faithful_ids = sorted(_feasible_set(faithful_solution), key=_candidate_sort_key)
    candidates = _candidate_by_id(source_row)
    additions: list[JsonDict] = []
    for variable in faithful_ast["variables"]:
        name = str(variable["name"])
        for value in variable["domain"]:
            kept = [
                candidate_id
                for candidate_id in faithful_ids
                if candidates[candidate_id]["assignment"][name] == value
            ]
            rejected = [
                candidate_id
                for candidate_id in faithful_ids
                if candidates[candidate_id]["assignment"][name] != value
            ]
            if kept and rejected:
                additions.append(
                    {
                        "constraint": {
                            "id": f"overfit-hard-equals-{name}",
                            "type": "equals",
                            "var": name,
                            "value": value,
                            "spurious": True,
                            "restriction_scope": "global",
                        },
                        "rejected_candidate_id": sorted(rejected, key=_candidate_sort_key)[0],
                    }
                )
    if len(faithful_ids) > 1:
        rejected_id = faithful_ids[-1]
        additions.append(
            {
                "constraint": {
                    "id": "overfit-hard-forbid-assignment",
                    "type": "forbid_assignment",
                    "assignment": _assignment_for_candidate(source_row, rejected_id),
                    "spurious": True,
                    "restriction_scope": "local",
                },
                "rejected_candidate_id": rejected_id,
            }
        )
    return additions


def _apply_added_constraint(
    model_ast: Mapping[str, Any],
    added_constraint: Mapping[str, Any],
) -> JsonDict:
    candidate_ast = _copy_json(model_ast)
    candidate_ast["hard_constraints"].append(dict(added_constraint))
    candidate_ast["soft_objective"] = dict(candidate_ast["soft_objective"])
    candidate_ast["soft_objective"]["terms"] = [
        str(row["id"]) for row in candidate_ast["soft_preferences"]
    ]
    return candidate_ast


def _choose_mutations(source_row: Mapping[str, Any], faithful_ast: Mapping[str, Any]) -> JsonDict:
    faithful_solution = model_solution_receipt(source_row, faithful_ast)
    faithful_set = _feasible_set(faithful_solution)
    deletions = _deletion_options(source_row, faithful_ast)
    additions = _addition_candidates(source_row, faithful_ast)
    for deletion in deletions:
        for addition in additions:
            overfit_ast = _apply_added_constraint(faithful_ast, addition["constraint"])
            overfit_solution = model_solution_receipt(source_row, overfit_ast)
            if not overfit_solution["satisfiable"]:
                continue
            overfit_set = _feasible_set(overfit_solution)
            if not faithful_set - overfit_set:
                continue
            mixed_ast = _apply_added_constraint(deletion["model_ast"], addition["constraint"])
            mixed_solution = model_solution_receipt(source_row, mixed_ast)
            mixed_set = _feasible_set(mixed_solution)
            if (mixed_set - faithful_set) and (faithful_set - mixed_set) and mixed_solution["satisfiable"]:
                return {
                    "faithful_solution": faithful_solution,
                    "incomplete": deletion,
                    "overfit": {
                        "model_ast": overfit_ast,
                        "added_constraint": dict(addition["constraint"]),
                        "rejected_candidate_id": sorted(
                            faithful_set - overfit_set,
                            key=_candidate_sort_key,
                        )[0],
                        "solution_receipt": overfit_solution,
                    },
                    "mixed": {
                        "model_ast": mixed_ast,
                        "deleted_constraint": dict(deletion["deleted_constraint"]),
                        "added_constraint": dict(addition["constraint"]),
                        "extra_candidate_id": sorted(mixed_set - faithful_set, key=_candidate_sort_key)[0],
                        "rejected_candidate_id": sorted(
                            faithful_set - mixed_set,
                            key=_candidate_sort_key,
                        )[0],
                        "solution_receipt": mixed_solution,
                    },
                }
    raise ValueError(f"no non-equivalent acquisition mutations for {source_row['instance_id']}")


def _assignment_receipt(
    variant_id: str,
    source_row: Mapping[str, Any],
    model_ast: Mapping[str, Any],
    candidate_id: str,
) -> JsonDict:
    candidate = _candidate_by_id(source_row)[candidate_id]
    evaluation = evaluate_model_candidate(source_row, model_ast, candidate)
    receipt = {
        "variant_id": variant_id,
        "candidate_id": candidate_id,
        "assignment": dict(candidate["assignment"]),
        "assignment_hash": sha256_json(
            {
                "variant_id": variant_id,
                "candidate_id": candidate_id,
                "assignment": candidate["assignment"],
            }
        ),
        "accepted_by_variant": bool(evaluation["feasible"]),
        "objective_value": evaluation["objective_value"],
        "hard_violations": list(evaluation["hard_violations"]),
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _membership_query(
    variant_id: str,
    source_row: Mapping[str, Any],
    faithful_ast: Mapping[str, Any],
    variant_ast: Mapping[str, Any],
    candidate_id: str,
    direction: str,
    expected_operation: str,
    index: int,
) -> JsonDict:
    candidate = _candidate_by_id(source_row)[candidate_id]
    faithful_eval = evaluate_model_candidate(source_row, faithful_ast, candidate)
    variant_eval = evaluate_model_candidate(source_row, variant_ast, candidate)
    query = {
        "query_id": f"{variant_id}-query-{index:02d}",
        "candidate_id": candidate_id,
        "assignment": dict(candidate["assignment"]),
        "assignment_hash": sha256_json(candidate["assignment"]),
        "faithful_accepts": bool(faithful_eval["feasible"]),
        "variant_accepts": bool(variant_eval["feasible"]),
        "direction": direction,
        "expected_repair_operation": expected_operation,
    }
    query["query_hash"] = sha256_json(query)
    return query


def _minimality_receipt(operation_types: Sequence[str], queries: Sequence[Mapping[str, Any]]) -> JsonDict:
    required = []
    if "add_missing_constraint" in operation_types:
        required.append("variant_accepts_extra")
    if "remove_spurious_constraint" in operation_types:
        required.append("variant_rejects_faithful")
    directions = [str(query["direction"]) for query in queries]
    leave_one_out = []
    for query in queries:
        remaining = [str(row["direction"]) for row in queries if row is not query]
        missing = sorted(set(required) - set(remaining))
        leave_one_out.append(
            {
                "removed_query_id": str(query["query_id"]),
                "missing_required_directions": missing,
            }
        )
    minimal = sorted(directions) == sorted(required) and len(directions) == len(required)
    return {
        "required_directions": required,
        "directions": directions,
        "leave_one_out": leave_one_out,
        "minimal": minimal,
    }


def _query_receipt(
    variant_id: str,
    operation_types: Sequence[str],
    queries: Sequence[Mapping[str, Any]],
) -> JsonDict:
    minimality = _minimality_receipt(operation_types, queries)
    receipt = {
        "variant_id": variant_id,
        "query_count": len(queries),
        "queries": [dict(query) for query in queries],
        "directions": list(minimality["directions"]),
        "minimal": bool(minimality["minimal"]),
        "minimality_receipt": minimality,
    }
    receipt["query_hash"] = sha256_json(receipt)
    return receipt


def _expected_repair_receipt(
    variant_id: str,
    operation_types: Sequence[str],
    deleted_constraint: Mapping[str, Any] | None,
    added_constraint: Mapping[str, Any] | None,
) -> JsonDict:
    operations = []
    if list(operation_types) == ["noop"]:
        operations.append({"operation": "noop", "role": "none"})
    if "add_missing_constraint" in operation_types:
        operations.append(
            {
                "operation": "add_missing_constraint",
                "constraint_id": str((deleted_constraint or {})["id"]),
                "role": "hard",
                "constraint_hash": sha256_json(deleted_constraint),
            }
        )
    if "remove_spurious_constraint" in operation_types:
        operations.append(
            {
                "operation": "remove_spurious_constraint",
                "constraint_id": str((added_constraint or {})["id"]),
                "role": "hard",
                "constraint_hash": sha256_json(added_constraint),
            }
        )
    receipt = {
        "variant_id": variant_id,
        "operation_types": list(operation_types),
        "operations": operations,
    }
    receipt["expected_repair_hash"] = sha256_json(receipt)
    return receipt


def _hard_soft_role_receipt(
    variant_id: str,
    faithful_ast: Mapping[str, Any],
    variant_ast: Mapping[str, Any],
    deleted_constraint: Mapping[str, Any] | None,
    added_constraint: Mapping[str, Any] | None,
) -> JsonDict:
    faithful_soft_ids = [str(row["id"]) for row in faithful_ast["soft_preferences"]]
    variant_soft_ids = [str(row["id"]) for row in variant_ast["soft_preferences"]]
    faithful_hard_ids = [str(row["id"]) for row in faithful_ast["hard_constraints"]]
    variant_hard_ids = [str(row["id"]) for row in variant_ast["hard_constraints"]]
    receipt = {
        "variant_id": variant_id,
        "faithful_hard_constraint_ids": faithful_hard_ids,
        "variant_hard_constraint_ids": variant_hard_ids,
        "deleted_hard_constraint_ids": [str(deleted_constraint["id"])] if deleted_constraint else [],
        "added_hard_constraint_ids": [str(added_constraint["id"])] if added_constraint else [],
        "soft_preference_ids_unchanged": faithful_soft_ids == variant_soft_ids,
        "soft_objective_unchanged": faithful_ast["soft_objective"] == variant_ast["soft_objective"],
        "hard_role_mutation_only": True,
        "no_soft_to_hard_conversion": True,
    }
    receipt["role_receipt_hash"] = sha256_json(receipt)
    return receipt


def _variant_record(
    *,
    case_id: str,
    kind: str,
    source_row: Mapping[str, Any],
    faithful_ast: Mapping[str, Any],
    model_ast: Mapping[str, Any],
    solution: Mapping[str, Any],
    operation_types: Sequence[str],
    deleted_constraint: Mapping[str, Any] | None,
    added_constraint: Mapping[str, Any] | None,
    extra_candidate_id: str | None,
    rejected_candidate_id: str | None,
) -> JsonDict:
    variant_id = f"{case_id}-{kind}"
    faithful_solution = model_solution_receipt(source_row, faithful_ast)
    faithful_set = _feasible_set(faithful_solution)
    variant_set = _feasible_set(solution)
    positive_id = sorted(variant_set, key=_candidate_sort_key)[0]
    all_ids = sorted((str(candidate["candidate_id"]) for candidate in source_row["candidate_pool"]), key=_candidate_sort_key)
    negative_id = next(candidate_id for candidate_id in all_ids if candidate_id not in variant_set)
    queries = []
    if extra_candidate_id:
        queries.append(
            _membership_query(
                variant_id,
                source_row,
                faithful_ast,
                model_ast,
                extra_candidate_id,
                "variant_accepts_extra",
                "add_missing_constraint",
                len(queries),
            )
        )
    if rejected_candidate_id:
        queries.append(
            _membership_query(
                variant_id,
                source_row,
                faithful_ast,
                model_ast,
                rejected_candidate_id,
                "variant_rejects_faithful",
                "remove_spurious_constraint",
                len(queries),
            )
        )
    model_text = _model_text(model_ast)
    expected_repair = _expected_repair_receipt(
        variant_id,
        operation_types,
        deleted_constraint,
        added_constraint,
    )
    role = _hard_soft_role_receipt(
        variant_id,
        faithful_ast,
        model_ast,
        deleted_constraint,
        added_constraint,
    )
    query = _query_receipt(variant_id, operation_types, queries)
    mutation_receipt = {
        "variant_id": variant_id,
        "variant_kind": kind,
        "deleted_constraint_ids": [str(deleted_constraint["id"])] if deleted_constraint else [],
        "added_constraint_ids": [str(added_constraint["id"])] if added_constraint else [],
        "semantic_change": (variant_set != faithful_set) if kind != "faithful" else False,
        "faithful_feasible_count": len(faithful_set),
        "variant_feasible_count": len(variant_set),
        "variant_accepts_extra_count": len(variant_set - faithful_set),
        "variant_rejects_faithful_count": len(faithful_set - variant_set),
    }
    mutation_receipt["mutation_hash"] = sha256_json(mutation_receipt)
    return {
        "variant_id": variant_id,
        "case_id": case_id,
        "variant_kind": kind,
        "model_ast": _copy_json(model_ast),
        "model_text": model_text,
        "model_hash": _model_hash(model_ast, model_text),
        "solution_receipt": dict(solution),
        "mutation_receipt": mutation_receipt,
        "positive_assignment_receipt": _assignment_receipt(
            variant_id,
            source_row,
            model_ast,
            positive_id,
        ),
        "negative_assignment_receipt": _assignment_receipt(
            variant_id,
            source_row,
            model_ast,
            negative_id,
        ),
        "distinguishing_query_receipt": query,
        "distinguishing_query_hash": str(query["query_hash"]),
        "expected_repair_receipt": expected_repair,
        "expected_repair_hash": str(expected_repair["expected_repair_hash"]),
        "hard_soft_role_receipt": role,
        "hard_soft_role_hash": str(role["role_receipt_hash"]),
    }


def _domain_artifact(source_row: Mapping[str, Any]) -> JsonDict:
    formulation = source_row["canonical_typed_formulation"]
    artifact = {
        "source_instance_id": str(source_row["instance_id"]),
        "family": str(source_row["family"]),
        "natural_language_specification": str(source_row["natural_language_specification"]),
        "variables": _copy_json(formulation["variables"]),
        "transitions": _copy_json(formulation["transitions"]),
        "source_domain_hashes": {
            "natural_language_specification_hash": str(source_row["natural_language_specification_hash"]),
            "canonical_typed_formulation_hash": str(source_row["canonical_typed_formulation_hash"]),
        },
    }
    artifact["domain_artifact_hash"] = sha256_json(artifact)
    return artifact


def _row_hash(row: Mapping[str, Any]) -> str:
    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _build_case(source_row: Mapping[str, Any], case_sequence_index: int) -> JsonDict:
    family = str(source_row["family"])
    split = str(source_row["split"])
    case_id = f"exp5761-{split}-{family.replace('_', '-')}-{case_sequence_index:03d}"
    faithful_ast = _copy_json(source_row["canonical_typed_formulation"])
    mutations = _choose_mutations(source_row, faithful_ast)
    domain = _domain_artifact(source_row)
    faithful = _variant_record(
        case_id=case_id,
        kind="faithful",
        source_row=source_row,
        faithful_ast=faithful_ast,
        model_ast=faithful_ast,
        solution=mutations["faithful_solution"],
        operation_types=["noop"],
        deleted_constraint=None,
        added_constraint=None,
        extra_candidate_id=None,
        rejected_candidate_id=None,
    )
    incomplete = _variant_record(
        case_id=case_id,
        kind="incomplete",
        source_row=source_row,
        faithful_ast=faithful_ast,
        model_ast=mutations["incomplete"]["model_ast"],
        solution=mutations["incomplete"]["solution_receipt"],
        operation_types=["add_missing_constraint"],
        deleted_constraint=mutations["incomplete"]["deleted_constraint"],
        added_constraint=None,
        extra_candidate_id=mutations["incomplete"]["extra_candidate_id"],
        rejected_candidate_id=None,
    )
    overfit = _variant_record(
        case_id=case_id,
        kind="overfit",
        source_row=source_row,
        faithful_ast=faithful_ast,
        model_ast=mutations["overfit"]["model_ast"],
        solution=mutations["overfit"]["solution_receipt"],
        operation_types=["remove_spurious_constraint"],
        deleted_constraint=None,
        added_constraint=mutations["overfit"]["added_constraint"],
        extra_candidate_id=None,
        rejected_candidate_id=mutations["overfit"]["rejected_candidate_id"],
    )
    mixed = _variant_record(
        case_id=case_id,
        kind="mixed",
        source_row=source_row,
        faithful_ast=faithful_ast,
        model_ast=mutations["mixed"]["model_ast"],
        solution=mutations["mixed"]["solution_receipt"],
        operation_types=["add_missing_constraint", "remove_spurious_constraint"],
        deleted_constraint=mutations["mixed"]["deleted_constraint"],
        added_constraint=mutations["mixed"]["added_constraint"],
        extra_candidate_id=mutations["mixed"]["extra_candidate_id"],
        rejected_candidate_id=mutations["mixed"]["rejected_candidate_id"],
    )
    variants = [faithful, incomplete, overfit, mixed]
    row: JsonDict = {
        "schema": MANIFEST_SCHEMA,
        "case_id": case_id,
        "source_instance_id": str(source_row["instance_id"]),
        "source_row_hash": str(source_row["row_hash"]),
        "source_formulation_hash": str(source_row["canonical_typed_formulation_hash"]),
        "family": family,
        "split": split,
        "source_family_index": int(source_row["family_index"]),
        "case_sequence_index": case_sequence_index,
        "random_seed": int(RANDOM_SEEDS["source_selection_seed"]) + case_sequence_index,
        "domain_artifact": domain,
        "domain_artifact_hash": str(domain["domain_artifact_hash"]),
        "variants": variants,
        "variants_by_kind": {variant["variant_kind"]: variant["variant_id"] for variant in variants},
        "variant_hashes_by_kind": {
            variant["variant_kind"]: variant["model_hash"] for variant in variants
        },
        "spec_refs": list(SPEC_REFS),
        "row_hash": "",
    }
    row["row_hash"] = _row_hash(row)
    return row


def _select_source_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    selected: list[JsonDict] = []
    for split in SPLITS:
        for family in REQUIRED_FAMILIES:
            matching = [
                dict(row)
                for row in rows
                if str(row["split"]) == split and str(row["family"]) == family
            ]
            matching = sorted(matching, key=lambda row: int(row["family_index"]))
            selected.extend(matching[:CASES_PER_SPLIT_FAMILY])
    return selected


def _upstream_manifest_path(preconditions_checked: Mapping[str, Any]) -> str:
    return str(
        dict(preconditions_checked.get("exp5746_replay") or {}).get("manifest_path")
        or REPO_ROOT / exp5746.BENCHMARK_MANIFEST_RELATIVE_PATH
    )


def generate_instances(preconditions_checked: Mapping[str, Any]) -> list[JsonDict]:
    """Generate the fixed 120-case acquisition benchmark from Exp5746 rows."""

    rows = exp5746.read_benchmark_manifest(_upstream_manifest_path(preconditions_checked))
    selected = _select_source_rows(rows)
    return [_build_case(row, index) for index, row in enumerate(selected)]


def _manifest_text(instances: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in instances)


def write_benchmark_manifest(instances: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write the full sealed acquisition manifest as JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_manifest_text(instances), encoding="utf-8")


def read_benchmark_manifest(path: str | Path) -> list[JsonDict]:
    """Read a sealed acquisition manifest."""

    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def _split_manifest(instances: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_split: dict[str, list[Mapping[str, Any]]] = {split: [] for split in SPLITS}
    for instance in instances:
        by_split[str(instance["split"])].append(instance)
    split_sets = {
        split: {str(row["case_id"]) for row in rows} for split, rows in by_split.items()
    }
    pairwise_intersections = {
        f"{left}|{right}": sorted(split_sets[left].intersection(split_sets[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    return {
        "split_counts": {split: len(by_split[split]) for split in sorted(by_split)},
        "family_counts": {
            split: dict(sorted(Counter(str(row["family"]) for row in rows).items()))
            for split, rows in sorted(by_split.items())
        },
        "split_hashes": {
            split: sha256_json([str(row["row_hash"]) for row in rows])
            for split, rows in sorted(by_split.items())
        },
        "pairwise_intersections": pairwise_intersections,
        "train_dev_science_separated": all(not values for values in pairwise_intersections.values()),
    }


def train_dev_science_disjoint_score(instances: Sequence[Mapping[str, Any]]) -> float:
    """Return 1.0 when no selected case appears in more than one split."""

    split_manifest = _split_manifest(instances)
    return 1.0 if split_manifest["train_dev_science_separated"] is True else 0.0


def _variant_records(instances: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(variant) for row in instances for variant in row["variants"]]


def _variant_counts(instances: Sequence[Mapping[str, Any]]) -> JsonDict:
    return dict(sorted(Counter(variant["variant_kind"] for variant in _variant_records(instances)).items()))


def collect_independent_validator_failures(instances: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recheck every variant with an independent candidate enumeration order."""

    failures: list[JsonDict] = []
    receipts = []
    for row in instances:
        source_row = next(
            source
            for source in exp5746.read_benchmark_manifest(REPO_ROOT / exp5746.BENCHMARK_MANIFEST_RELATIVE_PATH)
            if source["instance_id"] == row["source_instance_id"]
        )
        for variant in row["variants"]:
            independent = model_solution_receipt(
                source_row,
                variant["model_ast"],
                solver_version=INDEPENDENT_SOLVER_VERSION,
                independent_order=True,
            )
            primary = variant["solution_receipt"]
            agrees = (
                set(independent["feasible_candidate_ids"]) == set(primary["feasible_candidate_ids"])
                and independent["optimum_value"] == primary["optimum_value"]
                and set(independent["optimal_candidate_ids"]) == set(primary["optimal_candidate_ids"])
            )
            if not agrees:
                failures.append({"variant_id": variant["variant_id"], "case_id": row["case_id"]})
            receipts.append(
                {
                    "variant_id": variant["variant_id"],
                    "validator_version": INDEPENDENT_SOLVER_VERSION,
                    "agrees": agrees,
                }
            )
    return {
        "exact_validator_disagreement_count": len(failures),
        "failure_receipts": failures,
        "sample_receipts": receipts,
    }


def _objective_sign_differs(source_row: Mapping[str, Any]) -> bool:
    evaluations = source_row["solution_receipt"]["candidate_evaluations"]
    feasible = {candidate_id: row for candidate_id, row in evaluations.items() if row["feasible"] is True}
    normal = set(source_row["exact_optimum_receipt"]["optimal_candidate_ids"])
    inverted_value = min(int(row["objective_value"]) for row in feasible.values())
    inverted = {
        candidate_id
        for candidate_id, row in feasible.items()
        if int(row["objective_value"]) == inverted_value
    }
    return inverted != normal


def _first_source_row(instances: Sequence[Mapping[str, Any]], family: str) -> JsonDict:
    source_rows = exp5746.read_benchmark_manifest(REPO_ROOT / exp5746.BENCHMARK_MANIFEST_RELATIVE_PATH)
    source_id = next(row["source_instance_id"] for row in instances if row["family"] == family)
    return next(dict(row) for row in source_rows if row["instance_id"] == source_id)


def build_adversarial_controls(instances: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build deliberate invalid controls and prove the validators catch them."""

    if not instances:
        return {}
    base = instances[0]
    source = _first_source_row(instances, str(base["family"]))
    faithful_ast = base["variants"][0]["model_ast"]
    first_variable = faithful_ast["variables"][0]
    first_value, second_value = first_variable["domain"][:2]
    faithful_solution = base["variants"][0]["solution_receipt"]
    faithful_set = _feasible_set(faithful_solution)
    duplicate_ids = [instances[0]["case_id"], instances[0]["case_id"]]
    contradictory_ast = _copy_json(faithful_ast)
    contradictory_ast["hard_constraints"].extend(
        [
            {"id": "control-contradict-a", "type": "equals", "var": first_variable["name"], "value": first_value},
            {"id": "control-contradict-b", "type": "equals", "var": first_variable["name"], "value": second_value},
        ]
    )
    implied_ast = _copy_json(faithful_ast)
    implied_ast["hard_constraints"].append(dict(faithful_ast["hard_constraints"][0]))
    vacuous_ast = _copy_json(faithful_ast)
    vacuous_ast["hard_constraints"].append({"id": "control-vacuous-global", "type": "vacuous_global"})
    infeasible_ast = _copy_json(faithful_ast)
    for value in first_variable["domain"]:
        infeasible_ast["hard_constraints"].append(
            {
                "id": f"control-forbid-{value}",
                "type": "forbid_assignment",
                "assignment": {first_variable["name"]: value},
            }
        )
    shortcut_candidate = next(
        candidate_id
        for candidate_id in base["variants"][0]["solution_receipt"]["feasible_candidate_ids_hash"]
        if candidate_id
    )
    faithful_ids = set(base["variants"][0]["solution_receipt"]["feasible_candidate_ids"])
    shortcut_rejected = next(
        candidate["candidate_id"]
        for candidate in source["candidate_pool"]
        if candidate["candidate_id"] not in faithful_ids
    )
    controls = {
        "duplicate": {
            "detected": len(duplicate_ids) != len(set(duplicate_ids)),
            "blocked_gate": "duplicate_case_id",
        },
        "contradictory": {
            "detected": model_solution_receipt(source, contradictory_ast)["satisfiable"] is False,
            "blocked_gate": "contradictory_constraints",
        },
        "implied_constraint": {
            "detected": _feasible_set(model_solution_receipt(source, implied_ast)) == faithful_set,
            "blocked_gate": "semantic_noop_mutation",
        },
        "infeasible_model": {
            "detected": model_solution_receipt(source, contradictory_ast)["feasible_candidate_count"] == 0,
            "blocked_gate": "infeasible_model",
        },
        "vacuous_global": {
            "detected": _feasible_set(model_solution_receipt(source, vacuous_ast)) == faithful_set,
            "blocked_gate": "vacuous_global_noop",
        },
        "objective_sign": {
            "detected": _objective_sign_differs(_first_source_row(instances, "hard_soft_packing")),
            "blocked_gate": "objective_sign_inversion",
        },
        "leakage": {
            "detected": bool({"split", "answer_candidate_id"}.intersection({"split", "answer_candidate_id"})),
            "blocked_gate": "query_leakage_marker",
        },
        "split_collision": {
            "detected": bool({"case-a"}.intersection({"case-a"})),
            "blocked_gate": "train_dev_science_collision",
        },
        "shortcut": {
            "detected": evaluate_model_candidate(
                source,
                faithful_ast,
                _candidate_by_id(source)[shortcut_rejected],
            )["feasible"]
            is False,
            "blocked_gate": "shortcut_acceptance_without_hard_check",
            "candidate_id": shortcut_rejected,
            "shortcut_payload_hash": sha256_text(shortcut_candidate),
        },
    }
    for control in controls.values():
        control["control_hash"] = sha256_json(control)
    return controls


def _variant_hash_map(instances: Sequence[Mapping[str, Any]], kind: str) -> JsonDict:
    return {
        str(row["case_id"]): str(next(variant["model_hash"] for variant in row["variants"] if variant["variant_kind"] == kind))
        for row in instances
    }


def _domain_hash_map(instances: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {str(row["case_id"]): str(row["domain_artifact_hash"]) for row in instances}


def _assignment_count(instances: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(1 for variant in _variant_records(instances) if variant.get(field))


def _membership_query_count(instances: Sequence[Mapping[str, Any]]) -> int:
    return sum(int(variant["distinguishing_query_receipt"]["query_count"]) for variant in _variant_records(instances))


def _structure_receipt_failure_count(instances: Sequence[Mapping[str, Any]]) -> int:
    failures = 0
    for variant in _variant_records(instances):
        mutation = variant["mutation_receipt"]
        role = variant["hard_soft_role_receipt"]
        if role["soft_preference_ids_unchanged"] is not True or role["soft_objective_unchanged"] is not True:
            failures += 1
        if variant["variant_kind"] != "faithful" and mutation["semantic_change"] is not True:
            failures += 1
    return failures


def _solution_receipt_failure_count(instances: Sequence[Mapping[str, Any]]) -> int:
    failures = 0
    for variant in _variant_records(instances):
        solution = variant["solution_receipt"]
        if solution["satisfiable"] is not True or solution["all_candidates_checked"] is not True:
            failures += 1
        if variant["positive_assignment_receipt"]["accepted_by_variant"] is not True:
            failures += 1
        if variant["negative_assignment_receipt"]["accepted_by_variant"] is not False:
            failures += 1
        if variant["distinguishing_query_receipt"]["minimal"] is not True:
            failures += 1
    return failures


def solver_versions(preconditions_checked: Mapping[str, Any]) -> JsonDict:
    """Return exact solver versions plus upstream replay metadata."""

    return {
        "primary_exact_solver": PRIMARY_SOLVER_VERSION,
        "independent_exact_solver": INDEPENDENT_SOLVER_VERSION,
        "upstream_primary_exact_solver": exp5746.PRIMARY_SOLVER_VERSION,
        "upstream_independent_exact_solver": exp5746.INDEPENDENT_SOLVER_VERSION,
        "python": str(dict(preconditions_checked.get("python") or {}).get("version") or ""),
    }


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    """Return mechanical blockers for the terminal ready score."""

    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if int(artifact.get("instance_count") or 0) != INSTANCE_COUNT:
        reasons.append("instance_count")
    if dict(artifact.get("family_counts") or {}) != {family: 30 for family in REQUIRED_FAMILIES}:
        reasons.append("family_counts")
    if dict(artifact.get("variant_counts") or {}) != {kind: INSTANCE_COUNT for kind in VARIANT_KINDS}:
        reasons.append("variant_counts")
    if int(artifact.get("positive_assignment_count") or 0) != INSTANCE_COUNT * len(VARIANT_KINDS):
        reasons.append("positive_assignment_count")
    if int(artifact.get("negative_assignment_count") or 0) != INSTANCE_COUNT * len(VARIANT_KINDS):
        reasons.append("negative_assignment_count")
    if int(artifact.get("membership_query_count") or 0) != INSTANCE_COUNT * 4:
        reasons.append("membership_query_count")
    if artifact.get("train_dev_science_disjoint_score") != 1.0:
        reasons.append("train_dev_science_disjointness_failed")
    for field in (
        "structure_receipt_failure_count",
        "solution_receipt_failure_count",
        "exact_validator_disagreement_count",
    ):
        if int(artifact.get(field) or 0) > 0:
            reasons.append(field)
    if not all(dict(control).get("detected") is True for control in dict(artifact.get("adversarial_controls") or {}).values()):
        reasons.append("adversarial_control_not_detected")
    if artifact.get("llm_inference_used") is not False:
        reasons.append("llm_inference_used")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_not_oracle")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if not artifact.get("benchmark_manifest_hash"):
        reasons.append("benchmark_manifest_hash")
    return sorted(set(reasons))


def ca_benchmark_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return 1.0 only when every exact acquisition benchmark gate is clean."""

    ready = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and not blocked_reasons(artifact)
        and list(artifact.get("producer_gate_fields") or []) == list(PRODUCER_GATE_FIELDS)
        and all(not isinstance(artifact.get(field), dict) for field in PRODUCER_GATE_FIELDS)
    )
    return 1.0 if ready else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict from mechanical benchmark gates."""

    if ca_benchmark_ready_score(artifact) == 1.0:
        return "complete: exact_constraint_acquisition_benchmark_ready"
    reasons = blocked_reasons(artifact) or ["exact_constraint_acquisition_benchmark_not_ready"]
    return "blocked: " + ",".join(reasons)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum blanked."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _empty_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    benchmark_manifest_path: str,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    return _artifact_from_instances(
        instances=[],
        preconditions_checked=preconditions_checked,
        benchmark_manifest_path=benchmark_manifest_path,
        benchmark_manifest_hash=sha256_text(""),
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )


def _artifact_from_instances(
    *,
    instances: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    benchmark_manifest_path: str,
    benchmark_manifest_hash: str,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    independent = collect_independent_validator_failures(instances) if instances else {
        "exact_validator_disagreement_count": 0,
        "sample_receipts": [],
        "failure_receipts": [],
    }
    rows = list(instances)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "complete" if rows else "blocked",
        "preconditions_checked": dict(preconditions_checked),
        "spec_refs": list(SPEC_REFS),
        "upstream_artifact_hashes": {
            "exp5746_artifact": str(
                dict(preconditions_checked.get("exp5746_replay") or {}).get("artifact_hash") or ""
            ),
            "exp5746_manifest": str(
                dict(preconditions_checked.get("exp5746_replay") or {}).get("manifest_hash") or ""
            ),
            "exp5746_generator_source": sha256_file(Path(exp5746.__file__)),
        },
        "generator_version": GENERATOR_VERSION,
        "generator_source_hash": sha256_file(Path(__file__)),
        "solver_versions": solver_versions(preconditions_checked),
        "benchmark_manifest_path": benchmark_manifest_path,
        "benchmark_manifest_hash": benchmark_manifest_hash,
        "benchmark_row_hashes": {str(row["case_id"]): str(row["row_hash"]) for row in rows},
        "instance_count": len(rows),
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in rows).items())),
        "variant_counts": _variant_counts(rows),
        "faithful_model_hashes": _variant_hash_map(rows, "faithful"),
        "incomplete_model_hashes": _variant_hash_map(rows, "incomplete"),
        "overfit_model_hashes": _variant_hash_map(rows, "overfit"),
        "mixed_model_hashes": _variant_hash_map(rows, "mixed"),
        "domain_artifact_hashes": _domain_hash_map(rows),
        "positive_assignment_count": _assignment_count(rows, "positive_assignment_receipt"),
        "negative_assignment_count": _assignment_count(rows, "negative_assignment_receipt"),
        "membership_query_count": _membership_query_count(rows),
        "distinguishing_query_receipts": {
            str(variant["variant_id"]): dict(variant["distinguishing_query_receipt"])
            for variant in _variant_records(rows)
        },
        "expected_repair_receipts": {
            str(variant["variant_id"]): dict(variant["expected_repair_receipt"])
            for variant in _variant_records(rows)
        },
        "hard_soft_role_receipts": {
            str(variant["variant_id"]): dict(variant["hard_soft_role_receipt"])
            for variant in _variant_records(rows)
        },
        "split_manifest": _split_manifest(rows),
        "train_dev_science_disjoint_score": train_dev_science_disjoint_score(rows),
        "science_row_hashes": [str(row["row_hash"]) for row in rows if row.get("split") == "science"],
        "adversarial_controls": build_adversarial_controls(rows) if rows else {},
        "structure_receipt_failure_count": _structure_receipt_failure_count(rows),
        "solution_receipt_failure_count": _solution_receipt_failure_count(rows),
        "exact_validator_disagreement_count": int(independent["exact_validator_disagreement_count"]),
        "ca_benchmark_ready_score": 0.0,
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "llm_inference_used": False,
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": dict(RANDOM_SEEDS),
        "random_seed": int(RANDOM_SEEDS["base_seed"]),
        "model_specs": [
            {
                "model_id": "no_llm_exact_constraint_acquisition_generator",
                "role": "offline_exact_dataset_generator",
                "invoked": False,
                "llm_inference_used": False,
                "solver_version": PRIMARY_SOLVER_VERSION,
            }
        ],
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "methodology_note": "Gate scores of 1.0 are exact replay conjunctions over finite domains, not stochastic classifier performance.",
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "blocked_reasons": [],
    }
    artifact["ca_benchmark_ready_score"] = ca_benchmark_ready_score(artifact)
    artifact["status"] = "complete" if artifact["ca_benchmark_ready_score"] == 1.0 else "blocked"
    artifact["blocked_reasons"] = blocked_reasons(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    benchmark_manifest_path: str | Path = REPO_ROOT / BENCHMARK_MANIFEST_RELATIVE_PATH,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
) -> tuple[JsonDict, list[JsonDict]]:
    """Build the terminal artifact and manifest rows from exact receipts."""

    exit_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    manifest_path = str(Path(benchmark_manifest_path))
    if preconditions_checked.get("preconditions_ready") is not True:
        return (
            _empty_artifact(
                preconditions_checked=preconditions_checked,
                benchmark_manifest_path=manifest_path,
                test_commands=test_commands,
                test_exit_codes=exit_codes,
            ),
            [],
        )
    instances = generate_instances(preconditions_checked)
    manifest_hash = sha256_text(_manifest_text(instances))
    artifact = _artifact_from_instances(
        instances=instances,
        preconditions_checked=preconditions_checked,
        benchmark_manifest_path=manifest_path,
        benchmark_manifest_hash=manifest_hash,
        test_commands=test_commands,
        test_exit_codes=exit_codes,
    )
    return artifact, instances


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed on schema drift or unsupported readiness claims."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(artifact) != set(principles):
        errors.append("field_principles")
    if artifact.get("llm_inference_used") is not False:
        errors.append("llm_inference_used")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if list(artifact.get("producer_gate_fields") or []) != list(PRODUCER_GATE_FIELDS):
        errors.append("producer_gate_fields")
    if any(isinstance(artifact.get(field), dict) for field in PRODUCER_GATE_FIELDS):
        errors.append("producer_gate_fields")
    expected_score = ca_benchmark_ready_score(artifact)
    if artifact.get("ca_benchmark_ready_score") != expected_score:
        errors.append("ca_benchmark_ready_score")
    expected_status = "complete" if expected_score == 1.0 else "blocked"
    if artifact.get("status") != expected_status:
        errors.append("status")
    verdict = str(artifact.get("honest_verdict") or "")
    if expected_score == 1.0 and not verdict.startswith("complete:"):
        errors.append("honest_verdict")
    if expected_score == 0.0 and not verdict.startswith("blocked:"):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    if errors:
        raise ValueError(errors[0])
    return True


def verify_benchmark_manifest(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    """Replay row hashes, split hashes, and manifest hash against the artifact."""

    errors: list[str] = []
    expected_hashes = dict(artifact.get("benchmark_row_hashes") or {})
    for row in rows:
        row_id = str(row["case_id"])
        if expected_hashes.get(row_id) != row.get("row_hash") or _row_hash(row) != row.get("row_hash"):
            errors.append("row_hash")
    if sha256_text(_manifest_text(rows)) != artifact.get("benchmark_manifest_hash"):
        errors.append("benchmark_manifest_hash")
    if _split_manifest(rows) != artifact.get("split_manifest"):
        errors.append("split_manifest")
    if [str(row["row_hash"]) for row in rows if row.get("split") == "science"] != artifact.get("science_row_hashes"):
        errors.append("science_row_hashes")
    if errors:
        raise ManifestReplayError(errors[0])
    return True


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    benchmark_manifest_path: str | Path = REPO_ROOT / BENCHMARK_MANIFEST_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run preconditions, generate the benchmark, and write sealed artifacts."""

    preconditions = dict(preconditions_checked or collect_preconditions())
    artifact, instances = build_artifact(
        preconditions_checked=preconditions,
        benchmark_manifest_path=benchmark_manifest_path,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    if write:
        write_benchmark_manifest(instances, benchmark_manifest_path)
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run Exp5761 from the command line."""

    del argv
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
