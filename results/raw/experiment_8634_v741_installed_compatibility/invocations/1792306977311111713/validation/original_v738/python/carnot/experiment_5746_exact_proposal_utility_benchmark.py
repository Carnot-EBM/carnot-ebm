"""Exp5746 exact hard/soft proposal utility benchmark.

Spec refs: REQ-VERIFY-5746, SCENARIO-VERIFY-5746.

This module builds an offline benchmark, not a model evaluation.  The point is
to freeze complete finite candidate domains with exact hard-feasibility and
soft-objective receipts before a future GGUF proposal run consumes the rows.
Solver success is never treated as enough evidence by itself: every candidate
gets a replayable feasibility and objective receipt, and a separate
structure-side receipt checks that the typed formulation did not omit declared
variables, domains, constraints, preferences, or planning transitions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from typing import Any


JsonDict = dict[str, Any]
CommandRunner = Callable[[str], str]
Probe = Callable[[], JsonDict]
CollisionProbe = Callable[[Sequence[str]], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5746_exact_proposal_utility_benchmark.json")
BENCHMARK_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_5746_exact_proposal_utility_benchmark.instances.jsonl"
)
PREFLIGHT_RELATIVE_PATH = Path(
    "results/experiment_5746_exact_proposal_utility_benchmark.preflight.json"
)

SCHEMA = "carnot.experiment_5746.exact_proposal_utility_benchmark.v1"
MANIFEST_SCHEMA = SCHEMA + ".instance"
EXPERIMENT = 5746
EXPERIMENT_ID = "experiment_5746_exact_proposal_utility_benchmark"
MILESTONE = "2026.07.513"
RUN_DATE = "20260720"
GENERATOR_VERSION = "exp5746_exact_proposal_utility_benchmark_v1"
PRIMARY_SOLVER_VERSION = "carnot_exhaustive_hard_soft_solver_v1"
INDEPENDENT_SOLVER_VERSION = "carnot_independent_stratified_enumerator_v1"
ENERGY_HEURISTIC_VERSION = "deterministic_hard_penalty_soft_reward_energy_v1"
SPEC_REFS = ("REQ-VERIFY-5746", "SCENARIO-VERIFY-5746")

REQUIRED_FAMILIES = (
    "finite_domain_csp",
    "weighted_maxsat",
    "hard_soft_packing",
    "finite_state_planning",
)
SPLITS = ("train", "dev", "science")
INSTANCE_COUNT = 180
INSTANCES_PER_FAMILY = 45
INSTANCES_PER_SPLIT_FAMILY = 15
ADVERSARIAL_CONTROL_TYPES = (
    "omitted_constraint",
    "omitted_candidate",
    "duplicate_candidate",
    "infeasible_best_score",
    "shortcut",
    "objective_sign",
)
RANDOM_SEEDS: JsonDict = {
    "dataset_seed": 5746001,
    "split_seed": 5746002,
    "permutation_seed": 5746003,
    "control_seed": 5746004,
    "base_seed": 5746,
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5746_exact_proposal_utility_benchmark.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/experiment_5746_exact_proposal_utility_benchmark.py -m pytest tests/python/test_experiment_5746_exact_proposal_utility_benchmark.py -q --no-cov -n 0 && .venv/bin/coverage report --include=python/carnot/experiment_5746_exact_proposal_utility_benchmark.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5746_exact_proposal_utility_benchmark.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)
DOWNSTREAM_METRIC_DEFINITIONS: JsonDict = {
    "top_1_feasible_discovery": "1 when the first proposed candidate is hard-feasible.",
    "top_k_feasible_discovery": "1 when any of the first k proposals is hard-feasible; k is preregistered by the future run.",
    "nodes_to_first_valid": "1-indexed proposal position of the first hard-feasible candidate, or candidate_count+1 if absent.",
    "nodes_to_optimal": "1-indexed proposal position of the first exact-optimal feasible candidate, or candidate_count+1 if absent.",
    "verifier_calls": "number of exact candidate validation calls consumed by the proposal policy.",
    "hard_violation_count": "sum of hard-constraint violations on the selected candidate.",
    "normalized_optimality_gap": "max(0, optimum_value-selected_value) divided by the feasible objective range.",
    "wall_clock_s": "descriptive timing only; it never admits or rejects a candidate.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "spec_refs",
    "generator_version",
    "solver_versions",
    "random_seeds",
    "instance_count",
    "family_counts",
    "split_manifest",
    "science_row_hashes",
    "disjoint_from_v512_score",
    "candidate_pool_receipts",
    "structure_receipts",
    "solution_receipts",
    "hard_constraint_receipts",
    "soft_objective_receipts",
    "exact_optimum_receipts",
    "baseline_orderings",
    "adversarial_controls",
    "candidate_domain_incomplete_count",
    "structure_receipt_failure_count",
    "solution_receipt_failure_count",
    "validator_disagreement_count",
    "benchmark_manifest_path",
    "benchmark_manifest_hash",
    "benchmark_ready_score",
    "llm_inference_used",
    "verifier_is_oracle",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "names the artifact schema version for downstream validators.",
    "experiment": "numeric experiment id for conductor and result indexing.",
    "experiment_id": "stable experiment slug for traceability.",
    "milestone": "milestone accountability for this held-out benchmark build.",
    "run_date": "absolute run date prevents relative-date ambiguity.",
    "result_path": "records where the terminal artifact is expected to live.",
    "field_principles": "every artifact field states the receipt boundary or preregistered metric it protects.",
    "preconditions_checked": "records Python, Rust, solver, RAM, disk, and v512 disjointness checks before dataset generation.",
    "spec_refs": "binds the artifact to REQ-VERIFY-5746 and SCENARIO-VERIFY-5746.",
    "generator_version": "pins the deterministic benchmark generator implementation.",
    "solver_versions": "pins primary and independent exact solver implementations and host tool versions.",
    "random_seeds": "records dataset, split, permutation, and control seeds.",
    "random_seed": "legacy scalar seed for methodology linters that do not unwrap random_seeds.",
    "model_specs": "declares that no LLM model was invoked and names the exact offline generator instead.",
    "instance_count": "records the held-out benchmark denominator.",
    "family_counts": "proves balanced coverage across finite CSP, MaxSAT, packing, and planning families.",
    "split_manifest": "seals train/dev/science separation and split row hashes.",
    "science_row_hashes": "exposes the held-out science split commitment for future GGUF runs.",
    "disjoint_from_v512_score": "blocks row or experiment collisions with Exp5733/Exp5734.",
    "candidate_pool_receipts": "proves every bounded candidate pool is complete and duplicate-free.",
    "structure_receipts": "proves every declared variable, domain, hard constraint, soft preference, and transition is represented.",
    "solution_receipts": "proves feasibility and objective value for every candidate.",
    "hard_constraint_receipts": "separates hard feasibility receipts from soft preference scoring.",
    "soft_objective_receipts": "separates soft objective receipts from hard feasibility.",
    "exact_optimum_receipts": "records exact feasible sets, optimum values, and optimal candidate ids.",
    "baseline_orderings": "freezes solver-native, random, and deterministic energy-heuristic orderings before model use.",
    "adversarial_controls": "records omitted-constraint, omitted-candidate, duplicate, infeasible-best, shortcut, and objective-sign control detection.",
    "candidate_domain_incomplete_count": "blocks any incomplete finite candidate domain.",
    "structure_receipt_failure_count": "blocks formulation-structure omissions independent of solver success.",
    "solution_receipt_failure_count": "blocks candidate feasibility or objective receipt failures.",
    "validator_disagreement_count": "blocks primary versus independent exact-validator disagreements.",
    "benchmark_manifest_path": "points to the full sealed instance manifest.",
    "benchmark_manifest_hash": "seals the full manifest bytes.",
    "benchmark_row_hashes": "seals each individual manifest row by instance id.",
    "downstream_metric_definitions": "preregisters decision-utility metrics before any GGUF run.",
    "benchmark_ready_score": "strict benchmark-readiness scalar, not proposal accuracy.",
    "llm_inference_used": "bare false proves no GGUF or other LLM run contaminated benchmark generation.",
    "verifier_is_oracle": "bare true records exact validators as the only authority.",
    "test_commands": "records the focused, coverage, full-suite, spec, adversarial, and root-clutter commands.",
    "test_exit_codes": "records observed or preregistered zero exit codes for verification commands.",
    "reproducibility_checksum": "hashes the artifact with its checksum field blanked.",
    "honest_verdict": "terminal state starts complete: or blocked: and names the readiness boundary.",
    "blocked_reasons": "lists mechanical blockers when the benchmark is not ready.",
}


class ManifestReplayError(ValueError):
    """Raised when a benchmark manifest no longer matches artifact commitments."""


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
    """Hash a local file in chunks so manifest replay does not trust metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def planned_instance_ids() -> list[str]:
    """Return the deterministic row ids used for preflight collision checks."""

    ids: list[str] = []
    for split_index, split in enumerate(SPLITS):
        for split_family_index in range(INSTANCES_PER_SPLIT_FAMILY):
            family_index = split_index * INSTANCES_PER_SPLIT_FAMILY + split_family_index
            for family in REQUIRED_FAMILIES:
                ids.append(f"exp5746-{split}-{family.replace('_', '-')}-{family_index:02d}")
    return ids


def _run_version_command(name: str) -> str:  # pragma: no cover - host-dependent preflight.
    completed = subprocess.run(
        [name, "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip().splitlines()[0]


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


def _v512_collision_probe(planned_ids: Sequence[str]) -> JsonDict:  # pragma: no cover - artifact-dependent preflight.
    existing: set[str] = set()
    source_paths = [
        REPO_ROOT / "results/experiment_5733_sota_finite_choice_proposal_channel.json",
        REPO_ROOT / "results/experiment_5734_sota_exact_proposal_stream.json",
        REPO_ROOT / "results/experiment_5734_sota_exact_proposal_stream.rows.jsonl",
    ]
    source_presence = {path.name: path.exists() for path in source_paths}
    if source_paths[0].exists():
        artifact = json.loads(source_paths[0].read_text(encoding="utf-8"))
        existing.add(str(artifact.get("experiment_id") or ""))
        existing.update(str(row.get("control_id") or "") for row in artifact.get("control_manifest", []))
    if source_paths[1].exists():
        artifact = json.loads(source_paths[1].read_text(encoding="utf-8"))
        existing.add(str(artifact.get("experiment_id") or ""))
        existing.update(str(row.get("row_id") or "") for row in artifact.get("preregistered_panel", []))
    if source_paths[2].exists():
        for line in source_paths[2].read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing.add(str(row.get("row_id") or ""))
    collisions = sorted(set(planned_ids).intersection(existing))
    sources_present = all(source_presence.values())
    return {
        "source_artifacts_present": sources_present,
        "source_presence": source_presence,
        "collision_count": len(collisions),
        "colliding_ids": collisions,
        "score": 1.0 if sources_present and not collisions else 0.0,
    }


def collect_preconditions(
    *,
    planned_instance_ids: Sequence[str],
    command_runner: CommandRunner = _run_version_command,
    memory_probe: Probe = _memory_probe,
    disk_probe: Probe = _disk_probe,
    v512_collision_probe: CollisionProbe = _v512_collision_probe,
    python_version_ok: bool | None = None,
    exact_solvers_available: bool = True,
) -> JsonDict:
    """Collect the preflight receipt before full dataset generation starts."""

    command_receipts: JsonDict = {}
    for name in ("rustc", "cargo"):
        try:
            command_receipts[name] = {"available": True, "version": command_runner(name)}
        except (OSError, subprocess.SubprocessError) as exc:
            command_receipts[name] = {"available": False, "version": "", "error": str(exc)}
    memory = memory_probe()
    disk = disk_probe()
    collision = v512_collision_probe(planned_instance_ids)
    python_ok = (sys.version_info >= (3, 11)) if python_version_ok is None else python_version_ok
    blocked_reasons = []
    if not python_ok:
        blocked_reasons.append("python_version_too_old")
    for name, receipt in command_receipts.items():
        if receipt["available"] is not True:
            blocked_reasons.append(f"{name}_unavailable")
    if memory.get("ok") is not True:
        blocked_reasons.append("insufficient_free_ram")
    if disk.get("ok") is not True:
        blocked_reasons.append("insufficient_free_disk")
    if collision.get("score") != 1.0:
        blocked_reasons.append("v512_id_collision_or_missing_sources")
    if not exact_solvers_available:
        blocked_reasons.append("required_exact_solver_unavailable")
    return {
        "schema": SCHEMA + ".preflight",
        "run_date": RUN_DATE,
        "receipt_emitted_before_dataset_generation": True,
        "planned_instance_count": len(planned_instance_ids),
        "planned_instance_id_hash": sha256_json(list(planned_instance_ids)),
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": python_ok,
        },
        "rust": command_receipts,
        "memory": memory,
        "disk": disk,
        "exact_solvers_available": exact_solvers_available,
        "exact_solvers": {
            "primary": PRIMARY_SOLVER_VERSION,
            "independent": INDEPENDENT_SOLVER_VERSION,
        },
        "v512_collision_receipt": collision,
        "preflight_ready": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
    }


def fixture_preflight_receipt() -> JsonDict:
    """Return a deterministic preflight receipt for tests and manifest replay."""

    ids = planned_instance_ids()
    return {
        "schema": SCHEMA + ".preflight",
        "run_date": RUN_DATE,
        "receipt_emitted_before_dataset_generation": True,
        "planned_instance_count": len(ids),
        "planned_instance_id_hash": sha256_json(ids),
        "python": {
            "available": True,
            "version": "3.12.13-fixture",
            "executable": ".venv/bin/python",
            "ok": True,
        },
        "rust": {
            "rustc": {"available": True, "version": "rustc 1.97.0 fixture"},
            "cargo": {"available": True, "version": "cargo 1.97.0 fixture"},
        },
        "memory": {"available_mb": 8192, "required_mb": 512, "ok": True},
        "disk": {"available_mb": 8192, "required_mb": 512, "ok": True},
        "exact_solvers_available": True,
        "exact_solvers": {
            "primary": PRIMARY_SOLVER_VERSION,
            "independent": INDEPENDENT_SOLVER_VERSION,
        },
        "v512_collision_receipt": {"collision_count": 0, "colliding_ids": [], "score": 1.0},
        "preflight_ready": True,
        "blocked_reasons": [],
    }


def _variables(names: Sequence[str], domain: Sequence[Any]) -> list[JsonDict]:
    return [{"name": name, "domain": list(domain)} for name in names]


def _finite_domain_csp_formulation(index: int) -> tuple[str, JsonDict]:
    colors = ["red", "green", "blue"]
    anchor = colors[index % len(colors)]
    third = colors[(index + 1) % len(colors)]
    variables = _variables(("A", "B", "C"), colors)
    hard = [
        {"id": "csp-hard-anchor", "type": "equals", "var": "A", "value": anchor},
        {"id": "csp-hard-edge-ab", "type": "not_equal", "vars": ["A", "B"]},
        {"id": "csp-hard-edge-bc", "type": "not_equal", "vars": ["B", "C"]},
    ]
    soft = [
        {"id": "csp-soft-c-target", "type": "prefer_equals", "var": "C", "value": third, "weight": 5 + index % 4},
        {"id": "csp-soft-b-blue", "type": "prefer_equals", "var": "B", "value": "blue", "weight": 2 + index % 3},
    ]
    spec = (
        f"Color A, B, and C red/green/blue. A must be {anchor}; A and B differ; "
        f"B and C differ. Prefer C={third} and then B=blue by listed weights."
    )
    return spec, _formulation("finite_domain_csp", variables, hard, soft, [])


def _weighted_maxsat_formulation(index: int) -> tuple[str, JsonDict]:
    variables = _variables(("p", "q", "r", "s"), [0, 1])
    hard = [
        {"id": "sat-hard-coverage", "type": "clause", "literals": [["p", True], ["q", True]]},
        {"id": "sat-hard-link", "type": "clause", "literals": [["p", False], ["r", True]]},
        {"id": "sat-hard-guard", "type": "clause", "literals": [["q", False], ["s", True]]},
    ]
    if index % 2:
        hard.append({"id": "sat-hard-not-both", "type": "at_most_one", "vars": ["r", "s"]})
    soft = [
        {"id": "sat-soft-p", "type": "weighted_clause", "literals": [["p", True]], "weight": 2 + index % 5},
        {"id": "sat-soft-r", "type": "weighted_clause", "literals": [["r", True]], "weight": 3 + index % 4},
        {"id": "sat-soft-not-s", "type": "weighted_clause", "literals": [["s", False]], "weight": 1 + index % 3},
    ]
    spec = "Choose p,q,r,s in {0,1}; satisfy hard clauses, then maximize weighted satisfied soft clauses."
    return spec, _formulation("weighted_maxsat", variables, hard, soft, [])


def _hard_soft_packing_formulation(index: int) -> tuple[str, JsonDict]:
    item_count = 5
    variables = _variables([f"item{i}" for i in range(item_count)], [0, 1])
    weights = [1 + ((index + i * 2) % 5) for i in range(item_count)]
    utilities = [2 + ((index * 3 + i * 4) % 9) for i in range(item_count)]
    capacity = 6 + index % 4
    required = f"item{index % item_count}"
    conflict = [f"item{(index + 1) % item_count}", f"item{(index + 3) % item_count}"]
    hard = [
        {"id": "pack-hard-capacity", "type": "capacity", "weights": weights, "capacity": capacity},
        {"id": "pack-hard-required", "type": "requires_item", "var": required},
        {"id": "pack-hard-conflict", "type": "not_both", "vars": conflict},
    ]
    soft = [
        {
            "id": "pack-soft-utility",
            "type": "linear_utility",
            "utilities": utilities,
            "weight": 1,
        },
        {
            "id": "pack-soft-light-bonus",
            "type": "light_item_bonus",
            "max_weight": 2 + index % 2,
            "weight": 2,
        },
    ]
    spec = (
        f"Pack five items with capacity {capacity}, require {required}, forbid "
        f"selecting {conflict[0]} with {conflict[1]}, then maximize utility and light-item bonus."
    )
    return spec, _formulation("hard_soft_packing", variables, hard, soft, [])


def _finite_state_planning_formulation(index: int) -> tuple[str, JsonDict]:
    states = ["S0", "S1", "S2", "S3"]
    actions = ["L", "R", "H"]
    transitions = [
        {"from": "S0", "action": "L", "to": "S1"},
        {"from": "S0", "action": "R", "to": "S2"},
        {"from": "S0", "action": "H", "to": "S0"},
        {"from": "S1", "action": "L", "to": "S3"},
        {"from": "S1", "action": "R", "to": "S0"},
        {"from": "S1", "action": "H", "to": "S1"},
        {"from": "S2", "action": "L", "to": "S0"},
        {"from": "S2", "action": "R", "to": "S3"},
        {"from": "S2", "action": "H", "to": "S2"},
        {"from": "S3", "action": "L", "to": "S2"},
        {"from": "S3", "action": "R", "to": "S1"},
        {"from": "S3", "action": "H", "to": "S3"},
    ]
    planted = [["L", "L", "H"], ["R", "R", "H"], ["H", "L", "L"], ["H", "R", "R"]][index % 4]
    start = states[index % len(states)]
    goal = _simulate_plan(start, planted, transitions)[-1]
    variables = _variables(("a0", "a1", "a2"), actions)
    hard = [
        {"id": "plan-hard-goal", "type": "final_state", "start": start, "goal": goal},
        {"id": "plan-hard-h-limit", "type": "max_action_count", "action": "H", "limit": 1 + index % 2},
    ]
    soft = [
        {"id": "plan-soft-left", "type": "action_reward", "action": "L", "weight": 2 + index % 3},
        {"id": "plan-soft-right", "type": "action_reward", "action": "R", "weight": 1 + index % 4},
        {"id": "plan-soft-short-hold", "type": "action_penalty", "action": "H", "weight": 2},
    ]
    spec = (
        f"Start at {start}; choose three actions from L/R/H, follow the transition table, "
        f"end at {goal}, respect the H limit, then maximize action rewards."
    )
    return spec, _formulation("finite_state_planning", variables, hard, soft, transitions)


def _formulation(
    family: str,
    variables: Sequence[Mapping[str, Any]],
    hard_constraints: Sequence[Mapping[str, Any]],
    soft_preferences: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "family": family,
        "variables": [dict(row) for row in variables],
        "hard_constraints": [dict(row) for row in hard_constraints],
        "soft_preferences": [dict(row) for row in soft_preferences],
        "soft_objective": {
            "sense": "max",
            "terms": [str(row["id"]) for row in soft_preferences],
            "authority": "exact_candidate_evaluator",
        },
        "transitions": [dict(row) for row in transitions],
    }


def _base_formulation(family: str, index: int) -> tuple[str, JsonDict]:
    builders = {
        "finite_domain_csp": _finite_domain_csp_formulation,
        "weighted_maxsat": _weighted_maxsat_formulation,
        "hard_soft_packing": _hard_soft_packing_formulation,
        "finite_state_planning": _finite_state_planning_formulation,
    }
    return builders[family](index)


def _candidate_assignments(formulation: Mapping[str, Any]) -> list[JsonDict]:
    variables = list(formulation["variables"])
    names = [str(row["name"]) for row in variables]
    domains = [list(row["domain"]) for row in variables]
    assignments: list[JsonDict] = []
    for values in itertools.product(*domains):
        assignments.append({name: value for name, value in zip(names, values, strict=True)})
    return assignments


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


def evaluate_candidate(instance: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    """Evaluate one candidate against exact hard and soft receipts."""

    formulation = instance["canonical_typed_formulation"]
    assignment = candidate["assignment"]
    family = str(instance["family"])
    hard_violations: list[str] = []
    for constraint in formulation["hard_constraints"]:
        kind = str(constraint["type"])
        ok = True
        if kind == "equals":
            ok = assignment[str(constraint["var"])] == constraint["value"]
        elif kind == "not_equal":
            a, b = constraint["vars"]
            ok = assignment[str(a)] != assignment[str(b)]
        elif kind == "clause":
            ok = any(_literal_satisfied(assignment, literal) for literal in constraint["literals"])
        elif kind == "at_most_one":
            ok = sum(int(assignment[str(name)]) for name in constraint["vars"]) <= 1
        elif kind == "capacity":
            variables = [row["name"] for row in formulation["variables"]]
            ok = sum(int(assignment[str(name)]) * weight for name, weight in zip(variables, constraint["weights"], strict=True)) <= int(constraint["capacity"])
        elif kind == "requires_item":
            ok = int(assignment[str(constraint["var"])]) == 1
        elif kind == "not_both":
            a, b = constraint["vars"]
            ok = not (int(assignment[str(a)]) and int(assignment[str(b)]))
        elif kind == "final_state":
            actions = [str(assignment[f"a{i}"]) for i in range(3)]
            ok = _simulate_plan(str(constraint["start"]), actions, formulation["transitions"])[-1] == str(constraint["goal"])
        elif kind == "max_action_count":
            actions = [str(assignment[f"a{i}"]) for i in range(3)]
            ok = actions.count(str(constraint["action"])) <= int(constraint["limit"])
        if not ok:
            hard_violations.append(str(constraint["id"]))
    objective = 0
    for preference in formulation["soft_preferences"]:
        kind = str(preference["type"])
        if kind == "prefer_equals":
            objective += int(preference["weight"]) if assignment[str(preference["var"])] == preference["value"] else 0
        elif kind == "weighted_clause":
            objective += int(preference["weight"]) if any(_literal_satisfied(assignment, literal) for literal in preference["literals"]) else 0
        elif kind == "linear_utility":
            variables = [row["name"] for row in formulation["variables"]]
            objective += sum(int(assignment[str(name)]) * utility for name, utility in zip(variables, preference["utilities"], strict=True))
        elif kind == "light_item_bonus":
            variables = [row["name"] for row in formulation["variables"]]
            weights = next(row["weights"] for row in formulation["hard_constraints"] if row["type"] == "capacity")
            objective += int(preference["weight"]) * sum(
                int(assignment[str(name)]) for name, weight in zip(variables, weights, strict=True) if int(weight) <= int(preference["max_weight"])
            )
        elif kind == "action_reward":
            actions = [str(assignment[f"a{i}"]) for i in range(3)]
            objective += int(preference["weight"]) * actions.count(str(preference["action"]))
        elif kind == "action_penalty":
            actions = [str(assignment[f"a{i}"]) for i in range(3)]
            objective -= int(preference["weight"]) * actions.count(str(preference["action"]))
    if family == "finite_domain_csp":
        objective += 1 if assignment["A"] != assignment["C"] else 0
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "feasible": not hard_violations,
        "hard_violations": hard_violations,
        "hard_violation_count": len(hard_violations),
        "objective_value": objective,
    }


def _candidate_pool(instance_id: str, formulation: Mapping[str, Any]) -> list[JsonDict]:
    candidates: list[JsonDict] = []
    for index, assignment in enumerate(_candidate_assignments(formulation)):
        payload = {"instance_id": instance_id, "native_index": index, "assignment": assignment}
        candidates.append(
            {
                "candidate_id": f"{instance_id}-cand-{index:03d}",
                "native_order_index": index,
                "assignment": assignment,
                "candidate_hash": sha256_json(payload),
            }
        )
    return candidates


def candidate_pool_receipt(instance: Mapping[str, Any]) -> JsonDict:
    """Prove the candidate pool exactly covers the finite cross-product domain."""

    formulation = instance["canonical_typed_formulation"]
    candidates = list(instance["candidate_pool"])
    expected_assignments = _candidate_assignments(formulation)
    assignment_keys = [canonical_json(row["assignment"]) for row in candidates]
    expected_keys = [canonical_json(row) for row in expected_assignments]
    ids = [str(row["candidate_id"]) for row in candidates]
    complete = (
        len(candidates) == len(expected_assignments)
        and len(set(ids)) == len(ids)
        and len(set(assignment_keys)) == len(assignment_keys)
        and set(assignment_keys) == set(expected_keys)
    )
    return {
        "instance_id": str(instance["instance_id"]),
        "candidate_count": len(candidates),
        "expected_candidate_count": len(expected_assignments),
        "duplicate_candidate_count": len(candidates) - len(set(assignment_keys)),
        "duplicate_id_count": len(ids) - len(set(ids)),
        "domain_complete": complete,
        "candidate_pool_hash": sha256_json(candidates),
    }


def structure_receipt(instance: Mapping[str, Any]) -> JsonDict:
    """Check formulation-side coverage independently from solution values."""

    formulation = instance["canonical_typed_formulation"]
    variables = [str(row["name"]) for row in formulation["variables"]]
    candidate_var_sets = {tuple(sorted(candidate["assignment"])) for candidate in instance["candidate_pool"]}
    represented_domains = {
        name: sorted({candidate["assignment"][name] for candidate in instance["candidate_pool"]})
        for name in variables
    }
    declared_domains = {str(row["name"]): sorted(row["domain"]) for row in formulation["variables"]}
    hard_match = list(formulation["hard_constraints"]) == list(instance["hard_constraints"])
    soft_match = list(formulation["soft_preferences"]) == list(instance["soft_preferences"])
    transition_match = list(formulation["transitions"]) == list(instance["transitions"])
    complete = (
        candidate_var_sets == {tuple(sorted(variables))}
        and represented_domains == declared_domains
        and hard_match
        and soft_match
        and transition_match
    )
    return {
        "instance_id": str(instance["instance_id"]),
        "declared_variables": variables,
        "domains_represented": represented_domains == declared_domains,
        "hard_constraints_represented": hard_match,
        "soft_preferences_represented": soft_match,
        "transitions_represented": transition_match,
        "structure_complete": complete,
        "structure_hash": sha256_json(
            {
                "variables": formulation["variables"],
                "hard": formulation["hard_constraints"],
                "soft": formulation["soft_preferences"],
                "transitions": formulation["transitions"],
            }
        ),
    }


def solution_receipt(instance: Mapping[str, Any]) -> JsonDict:
    """Evaluate feasibility and objective value for every candidate."""

    evaluations = {
        str(candidate["candidate_id"]): evaluate_candidate(instance, candidate)
        for candidate in instance["candidate_pool"]
    }
    candidate_ids = {str(candidate["candidate_id"]) for candidate in instance["candidate_pool"]}
    return {
        "instance_id": str(instance["instance_id"]),
        "candidate_count": len(candidate_ids),
        "candidate_evaluations": evaluations,
        "all_candidates_checked": set(evaluations) == candidate_ids,
        "solution_hash": sha256_json(evaluations),
    }


def hard_constraint_receipt(instance: Mapping[str, Any], solution: Mapping[str, Any]) -> JsonDict:
    """Summarize hard feasibility separately from the soft objective."""

    evaluations = dict(solution["candidate_evaluations"])
    violation_counts = {
        candidate_id: int(row["hard_violation_count"]) for candidate_id, row in evaluations.items()
    }
    return {
        "instance_id": str(instance["instance_id"]),
        "hard_constraint_ids": [str(row["id"]) for row in instance["hard_constraints"]],
        "candidate_count": len(evaluations),
        "feasible_candidate_count": sum(1 for row in evaluations.values() if row["feasible"] is True),
        "all_candidates_checked": len(evaluations) == len(instance["candidate_pool"]),
        "hard_violation_hash": sha256_json(violation_counts),
    }


def soft_objective_receipt(instance: Mapping[str, Any], solution: Mapping[str, Any]) -> JsonDict:
    """Summarize objective values separately from hard feasibility."""

    evaluations = dict(solution["candidate_evaluations"])
    objective_values = {
        candidate_id: int(row["objective_value"]) for candidate_id, row in evaluations.items()
    }
    feasible_values = [int(row["objective_value"]) for row in evaluations.values() if row["feasible"] is True]
    return {
        "instance_id": str(instance["instance_id"]),
        "objective_sense": str(instance["soft_objective"]["sense"]),
        "soft_preference_ids": [str(row["id"]) for row in instance["soft_preferences"]],
        "candidate_count": len(evaluations),
        "all_candidates_scored": len(evaluations) == len(instance["candidate_pool"]),
        "best_feasible_value": max(feasible_values),
        "objective_values_hash": sha256_json(objective_values),
    }


def exact_optimum_receipt(instance: Mapping[str, Any], solution: Mapping[str, Any]) -> JsonDict:
    """Record exact feasible set and all optimal feasible candidates."""

    evaluations = dict(solution["candidate_evaluations"])
    feasible_ids = [candidate_id for candidate_id, row in evaluations.items() if row["feasible"] is True]
    optimum_value = max(int(evaluations[candidate_id]["objective_value"]) for candidate_id in feasible_ids)
    optimal_ids = [
        candidate_id
        for candidate_id in feasible_ids
        if int(evaluations[candidate_id]["objective_value"]) == optimum_value
    ]
    return {
        "instance_id": str(instance["instance_id"]),
        "solver_version": PRIMARY_SOLVER_VERSION,
        "solver_exit": "success",
        "solver_exit_success": True,
        "feasible_candidate_count": len(feasible_ids),
        "feasible_candidate_ids": feasible_ids,
        "feasible_candidate_ids_hash": sha256_json(feasible_ids),
        "optimum_value": optimum_value,
        "optimal_candidate_ids": optimal_ids,
        "optimal_candidate_ids_hash": sha256_json(optimal_ids),
    }


def baseline_ordering_receipt(
    instance: Mapping[str, Any],
    solution: Mapping[str, Any],
    optimum: Mapping[str, Any],
) -> JsonDict:
    """Freeze native, random, and deterministic energy-heuristic orderings."""

    del optimum
    evaluations = dict(solution["candidate_evaluations"])
    native_order = [str(candidate["candidate_id"]) for candidate in instance["candidate_pool"]]
    seed = int(RANDOM_SEEDS["permutation_seed"]) + int(instance["sequence_index"])
    random_order = list(native_order)
    random.Random(seed).shuffle(random_order)
    energy_scores = {
        candidate_id: int(row["hard_violation_count"]) * 1000 - int(row["objective_value"])
        for candidate_id, row in evaluations.items()
    }
    heuristic_order = sorted(native_order, key=lambda candidate_id: (energy_scores[candidate_id], candidate_id))
    return {
        "instance_id": str(instance["instance_id"]),
        "exact_solver_native_order": native_order,
        "random_permutation_seed": seed,
        "random_permutation_order": random_order,
        "energy_heuristic_version": ENERGY_HEURISTIC_VERSION,
        "energy_heuristic_scores": energy_scores,
        "energy_heuristic_order": heuristic_order,
    }


def _row_hash(row: Mapping[str, Any]) -> str:
    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _build_instance(split: str, family: str, family_index: int, sequence_index: int) -> JsonDict:
    instance_id = f"exp5746-{split}-{family.replace('_', '-')}-{family_index:02d}"
    natural_language_spec, formulation = _base_formulation(family, family_index)
    candidates = _candidate_pool(instance_id, formulation)
    instance: JsonDict = {
        "schema": MANIFEST_SCHEMA,
        "instance_id": instance_id,
        "sequence_index": sequence_index,
        "family": family,
        "family_index": family_index,
        "split": split,
        "random_seed": int(RANDOM_SEEDS["dataset_seed"]) + sequence_index,
        "natural_language_specification": natural_language_spec,
        "natural_language_specification_hash": sha256_text(natural_language_spec),
        "canonical_typed_formulation": formulation,
        "canonical_typed_formulation_hash": sha256_json(formulation),
        "hard_constraints": list(formulation["hard_constraints"]),
        "hard_constraints_hash": sha256_json(formulation["hard_constraints"]),
        "soft_preferences": list(formulation["soft_preferences"]),
        "soft_objective": dict(formulation["soft_objective"]),
        "soft_objective_hash": sha256_json(formulation["soft_objective"]),
        "transitions": list(formulation["transitions"]),
        "candidate_pool": candidates,
        "candidate_pool_hash": sha256_json(candidates),
        "spec_refs": list(SPEC_REFS),
    }
    pool = candidate_pool_receipt(instance)
    structure = structure_receipt(instance)
    solution = solution_receipt(instance)
    hard = hard_constraint_receipt(instance, solution)
    soft = soft_objective_receipt(instance, solution)
    optimum = exact_optimum_receipt(instance, solution)
    ordering = baseline_ordering_receipt(instance, solution, optimum)
    instance.update(
        {
            "candidate_pool_receipt": pool,
            "structure_receipt": structure,
            "solution_receipt": solution,
            "hard_constraint_receipt": hard,
            "soft_objective_receipt": soft,
            "exact_optimum_receipt": optimum,
            "exact_feasible_set": list(optimum["feasible_candidate_ids"]),
            "baseline_ordering": ordering,
            "independent_sampled": family_index % 7 == 0,
            "row_hash": "",
        }
    )
    instance["row_hash"] = _row_hash(instance)
    return instance


def generate_instances() -> list[JsonDict]:
    """Generate the fixed 180-row held-out benchmark dataset."""

    rows: list[JsonDict] = []
    sequence_index = 0
    for split_index, split in enumerate(SPLITS):
        for split_family_index in range(INSTANCES_PER_SPLIT_FAMILY):
            family_index = split_index * INSTANCES_PER_SPLIT_FAMILY + split_family_index
            for family in REQUIRED_FAMILIES:
                rows.append(_build_instance(split, family, family_index, sequence_index))
                sequence_index += 1
    return rows


def _independent_optimum_receipt(instance: Mapping[str, Any]) -> JsonDict:
    formulation = instance["canonical_typed_formulation"]
    reversed_variables = list(reversed(formulation["variables"]))
    independent_formulation = dict(formulation)
    independent_formulation["variables"] = reversed_variables
    original_ids = {
        canonical_json(candidate["assignment"]): str(candidate["candidate_id"])
        for candidate in instance["candidate_pool"]
    }
    candidates = _candidate_pool(str(instance["instance_id"]) + "-independent", independent_formulation)
    translated_candidates = []
    for candidate in candidates:
        translated = dict(candidate)
        translated["candidate_id"] = original_ids[canonical_json(candidate["assignment"])]
        translated_candidates.append(translated)
    replay = dict(instance)
    replay["candidate_pool"] = translated_candidates
    solution = solution_receipt(replay)
    return exact_optimum_receipt(replay, solution)


def collect_independent_validator_failures(instances: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Double-check a stratified sample with an independent enumeration order."""

    sample_receipts = []
    disagreement_count = 0
    for instance in instances:
        sampled = bool(instance.get("independent_sampled"))
        agrees = True
        if sampled:
            independent = _independent_optimum_receipt(instance)
            exact = instance["exact_optimum_receipt"]
            agrees = (
                independent["optimum_value"] == exact["optimum_value"]
                and set(independent["optimal_candidate_ids"]) == set(exact["optimal_candidate_ids"])
            )
            disagreement_count += 0 if agrees else 1
        sample_receipts.append(
            {
                "instance_id": str(instance["instance_id"]),
                "family": str(instance["family"]),
                "sampled": sampled,
                "validator_version": INDEPENDENT_SOLVER_VERSION,
                "agrees": agrees,
            }
        )
    return {
        "validator_disagreement_count": disagreement_count,
        "sample_receipts": sample_receipts,
    }


def _copy_instance(instance: Mapping[str, Any]) -> JsonDict:
    return json.loads(canonical_json(instance))


def _first_infeasible_best_candidate(instance: Mapping[str, Any]) -> str:
    solution = instance["solution_receipt"]["candidate_evaluations"]
    infeasible = [
        (candidate_id, row)
        for candidate_id, row in solution.items()
        if row["feasible"] is not True
    ]
    return max(infeasible, key=lambda item: (int(item[1]["objective_value"]), item[0]))[0]


def _objective_sign_differs(instance: Mapping[str, Any]) -> bool:
    evaluations = instance["solution_receipt"]["candidate_evaluations"]
    feasible = {candidate_id: row for candidate_id, row in evaluations.items() if row["feasible"] is True}
    normal = set(instance["exact_optimum_receipt"]["optimal_candidate_ids"])
    inverted_value = min(int(row["objective_value"]) for row in feasible.values())
    inverted = {
        candidate_id
        for candidate_id, row in feasible.items()
        if int(row["objective_value"]) == inverted_value
    }
    return inverted != normal


def build_adversarial_controls(instances: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build deliberate invalid controls and prove that receipts catch them."""

    base = {str(row["family"]): row for row in instances}
    omitted_constraint = _copy_instance(base["finite_domain_csp"])
    omitted_constraint["canonical_typed_formulation"]["hard_constraints"] = omitted_constraint["canonical_typed_formulation"]["hard_constraints"][:-1]
    omitted_candidate = _copy_instance(base["weighted_maxsat"])
    omitted_candidate["candidate_pool"] = omitted_candidate["candidate_pool"][:-1]
    duplicate_candidate = _copy_instance(base["hard_soft_packing"])
    duplicate_candidate["candidate_pool"][-1] = dict(duplicate_candidate["candidate_pool"][0])
    infeasible_candidate = _first_infeasible_best_candidate(base["hard_soft_packing"])
    shortcut_candidate = _first_infeasible_best_candidate(base["finite_state_planning"])
    objective_instance = next(row for row in instances if _objective_sign_differs(row))
    controls = {
        "omitted_constraint": {
            "control_instance_id": omitted_constraint["instance_id"],
            "blocked_gate": "structure_receipt_failure",
            "detected": structure_receipt(omitted_constraint)["structure_complete"] is False,
        },
        "omitted_candidate": {
            "control_instance_id": omitted_candidate["instance_id"],
            "blocked_gate": "candidate_domain_incomplete",
            "detected": candidate_pool_receipt(omitted_candidate)["domain_complete"] is False,
        },
        "duplicate_candidate": {
            "control_instance_id": duplicate_candidate["instance_id"],
            "blocked_gate": "candidate_domain_duplicate",
            "detected": candidate_pool_receipt(duplicate_candidate)["duplicate_candidate_count"] > 0,
        },
        "infeasible_best_score": {
            "control_instance_id": base["hard_soft_packing"]["instance_id"],
            "blocked_gate": "hard_constraint_receipt",
            "candidate_id": infeasible_candidate,
            "detected": base["hard_soft_packing"]["solution_receipt"]["candidate_evaluations"][infeasible_candidate]["feasible"] is False,
        },
        "shortcut": {
            "control_instance_id": base["finite_state_planning"]["instance_id"],
            "blocked_gate": "hard_constraint_receipt",
            "candidate_id": shortcut_candidate,
            "detected": base["finite_state_planning"]["solution_receipt"]["candidate_evaluations"][shortcut_candidate]["hard_violation_count"] > 0,
        },
        "objective_sign": {
            "control_instance_id": objective_instance["instance_id"],
            "blocked_gate": "soft_objective_receipt",
            "detected": _objective_sign_differs(objective_instance),
        },
    }
    for control in controls.values():
        control["control_hash"] = sha256_json(control)
    return controls


def _manifest_text(instances: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in instances)


def write_benchmark_manifest(instances: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write the full sealed instance manifest as JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_manifest_text(instances), encoding="utf-8")


def read_benchmark_manifest(path: str | Path) -> list[JsonDict]:
    """Read a sealed benchmark manifest."""

    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def _split_manifest(instances: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_split: dict[str, list[Mapping[str, Any]]] = {split: [] for split in SPLITS}
    for instance in instances:
        by_split[str(instance["split"])].append(instance)
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
        "train_dev_science_separated": all(by_split.values()),
    }


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    count_fields = (
        "candidate_domain_incomplete_count",
        "structure_receipt_failure_count",
        "solution_receipt_failure_count",
        "validator_disagreement_count",
    )
    for field in count_fields:
        if int(artifact.get(field) or 0) > 0:
            reasons.append(field)
    if artifact.get("disjoint_from_v512_score") != 1.0:
        reasons.append("v512_disjointness_failed")
    if not all(dict(control).get("detected") is True for control in dict(artifact.get("adversarial_controls") or {}).values()):
        reasons.append("adversarial_control_not_detected")
    if artifact.get("llm_inference_used") is not False:
        reasons.append("llm_inference_used")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_not_oracle")
    return sorted(set(reasons))


def benchmark_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return 1.0 only when every exact benchmark readiness gate is clean."""

    ready = (
        int(artifact.get("instance_count") or 0) == INSTANCE_COUNT
        and dict(artifact.get("family_counts") or {}) == {family: INSTANCES_PER_FAMILY for family in REQUIRED_FAMILIES}
        and dict(artifact.get("preconditions_checked") or {}).get("preflight_ready") is True
        and artifact.get("disjoint_from_v512_score") == 1.0
        and int(artifact.get("candidate_domain_incomplete_count") or 0) == 0
        and int(artifact.get("structure_receipt_failure_count") or 0) == 0
        and int(artifact.get("solution_receipt_failure_count") or 0) == 0
        and int(artifact.get("validator_disagreement_count") or 0) == 0
        and artifact.get("llm_inference_used") is False
        and artifact.get("verifier_is_oracle") is True
        and all(dict(control).get("detected") is True for control in dict(artifact.get("adversarial_controls") or {}).values())
        and bool(artifact.get("benchmark_manifest_hash"))
        and not _blocked_reasons(artifact)
    )
    return 1.0 if ready else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict from mechanical benchmark gates."""

    if benchmark_ready_score(artifact) == 1.0:
        return "complete: exact_proposal_utility_benchmark_ready"
    reasons = _blocked_reasons(artifact) or ["exact_proposal_utility_benchmark_not_ready"]
    return "blocked: " + ",".join(reasons)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum blanked."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def solver_versions(preconditions_checked: Mapping[str, Any]) -> JsonDict:
    """Return exact solver versions plus the preflight host tool versions."""

    rust = dict(preconditions_checked.get("rust") or {})
    return {
        "primary_exact_solver": PRIMARY_SOLVER_VERSION,
        "independent_exact_solver": INDEPENDENT_SOLVER_VERSION,
        "energy_heuristic": ENERGY_HEURISTIC_VERSION,
        "python": str(dict(preconditions_checked.get("python") or {}).get("version") or ""),
        "rustc": str(dict(rust.get("rustc") or {}).get("version") or ""),
        "cargo": str(dict(rust.get("cargo") or {}).get("version") or ""),
    }


def _empty_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    benchmark_manifest_path: str,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    artifact = _artifact_from_instances(
        instances=[],
        preconditions_checked=preconditions_checked,
        benchmark_manifest_path=benchmark_manifest_path,
        benchmark_manifest_hash=sha256_text(""),
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    return artifact


def _artifact_from_instances(
    *,
    instances: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    benchmark_manifest_path: str,
    benchmark_manifest_hash: str,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    independent = collect_independent_validator_failures(instances)
    candidate_pool_receipts = {str(row["instance_id"]): dict(row["candidate_pool_receipt"]) for row in instances}
    structure_receipts = {str(row["instance_id"]): dict(row["structure_receipt"]) for row in instances}
    solution_receipts = {str(row["instance_id"]): dict(row["solution_receipt"]) for row in instances}
    hard_receipts = {str(row["instance_id"]): dict(row["hard_constraint_receipt"]) for row in instances}
    soft_receipts = {str(row["instance_id"]): dict(row["soft_objective_receipt"]) for row in instances}
    optimum_receipts = {str(row["instance_id"]): dict(row["exact_optimum_receipt"]) for row in instances}
    baseline_orderings = {str(row["instance_id"]): dict(row["baseline_ordering"]) for row in instances}
    row_hashes = {str(row["instance_id"]): str(row["row_hash"]) for row in instances}
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": dict(preconditions_checked),
        "spec_refs": list(SPEC_REFS),
        "generator_version": GENERATOR_VERSION,
        "solver_versions": solver_versions(preconditions_checked),
        "random_seeds": dict(RANDOM_SEEDS),
        "random_seed": int(RANDOM_SEEDS["base_seed"]),
        "model_specs": [
            {
                "model_id": "no_llm_exact_enumerator",
                "role": "offline_exact_benchmark_generator",
                "invoked": False,
                "llm_inference_used": False,
                "solver_version": PRIMARY_SOLVER_VERSION,
            }
        ],
        "instance_count": len(instances),
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in instances).items())),
        "split_manifest": _split_manifest(instances),
        "science_row_hashes": [str(row["row_hash"]) for row in instances if row.get("split") == "science"],
        "disjoint_from_v512_score": float(dict(preconditions_checked.get("v512_collision_receipt") or {}).get("score") or 0.0),
        "candidate_pool_receipts": candidate_pool_receipts,
        "structure_receipts": structure_receipts,
        "solution_receipts": solution_receipts,
        "hard_constraint_receipts": hard_receipts,
        "soft_objective_receipts": soft_receipts,
        "exact_optimum_receipts": optimum_receipts,
        "baseline_orderings": baseline_orderings,
        "adversarial_controls": build_adversarial_controls(instances) if instances else {},
        "candidate_domain_incomplete_count": sum(1 for row in candidate_pool_receipts.values() if row["domain_complete"] is not True),
        "structure_receipt_failure_count": sum(1 for row in structure_receipts.values() if row["structure_complete"] is not True),
        "solution_receipt_failure_count": sum(1 for row in solution_receipts.values() if row["all_candidates_checked"] is not True),
        "validator_disagreement_count": int(independent["validator_disagreement_count"]),
        "benchmark_manifest_path": benchmark_manifest_path,
        "benchmark_manifest_hash": benchmark_manifest_hash,
        "benchmark_row_hashes": row_hashes,
        "downstream_metric_definitions": dict(DOWNSTREAM_METRIC_DEFINITIONS),
        "benchmark_ready_score": 0.0,
        "llm_inference_used": False,
        "verifier_is_oracle": True,
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "blocked_reasons": [],
    }
    artifact["benchmark_ready_score"] = benchmark_ready_score(artifact)
    artifact["blocked_reasons"] = _blocked_reasons(artifact)
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
    if preconditions_checked.get("preflight_ready") is not True:
        return (
            _empty_artifact(
                preconditions_checked=preconditions_checked,
                benchmark_manifest_path=manifest_path,
                test_commands=test_commands,
                test_exit_codes=exit_codes,
            ),
            [],
        )
    instances = generate_instances()
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
    expected_score = benchmark_ready_score(artifact)
    if artifact.get("benchmark_ready_score") != expected_score:
        errors.append("benchmark_ready_score")
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
        row_id = str(row["instance_id"])
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
    preflight_receipt_path: str | Path = REPO_ROOT / PREFLIGHT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the preflight, generate the benchmark, and write sealed artifacts."""

    planned_ids = planned_instance_ids()
    preflight = dict(preconditions_checked or collect_preconditions(planned_instance_ids=planned_ids))
    if write:
        preflight_path = Path(preflight_receipt_path)
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(
            json.dumps(preflight, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    artifact, instances = build_artifact(
        preconditions_checked=preflight,
        benchmark_manifest_path=benchmark_manifest_path,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    if write:
        write_benchmark_manifest(instances, benchmark_manifest_path)
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run Exp5746 from the command line."""

    del argv
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
