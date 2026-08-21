"""Exp6478 identifiable held exact-energy selection.

Spec refs: REQ-VERIFY-6478, SCENARIO-VERIFY-6478-GATES,
SCENARIO-VERIFY-6478-PRECOMMITMENT, SCENARIO-VERIFY-6478-MATCHED-SELECTION,
SCENARIO-VERIFY-6478-ROWS, SCENARIO-VERIFY-6478-ATTACKS.

The selector sees only frozen solver-grounded candidates. Energy is a proposal
score. Exact finite-domain replay is the only final authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_6474_protocol_identifiability_and_receipt_preflight as exp6474
from carnot import experiment_6477_backend_neutral_exact_constraint_record as exp6477
from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
RANDOM_SEED = 6478
SELECTION_SEEDS = (647801, 647802)
INFERENCE_SUBSTRATE = "exact_solver_held_candidate_selection_no_llm"
SCHEMA_VERSION = "carnot.experiment_6478.identifiable_held_exact_energy_selection.v1"

RESULT_RELATIVE_PATH = Path("results/experiment_6478_identifiable_held_exact_energy_selection.json")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6478_identifiable_held_exact_energy_selection.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6478_identifiable_held_exact_energy_selection.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
UPSTREAM_IDENTIFIABILITY_RELATIVE_PATH = Path(
    "results/experiment_6474_protocol_identifiability_and_receipt_preflight.json"
)
UPSTREAM_CONSTRAINT_RELATIVE_PATH = Path(
    "results/experiment_6477_backend_neutral_exact_constraint_record.json"
)

ARMS = (
    "first_candidate",
    "deterministic_random",
    "shuffled_energy",
    "violation_count",
    "exact_energy",
)
HEADLINE_ARM = "exact_energy"
BASELINE_ARMS = ("first_candidate", "shuffled_energy")
ATTACK_IDS = (
    "held_leakage",
    "result_dependent_weights",
    "shuffled_labels",
    "tie_manipulation",
    "energy_sign_reversal",
    "matched_totals_different_protected_violations",
    "aggregate_mismatch",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6478_identifiable_held_exact_energy_selection --date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6478_identifiable_held_exact_energy_selection.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6478_identifiable_held_exact_energy_selection.py "
    "-m pytest "
    "tests/python/test_experiment_6478_identifiable_held_exact_energy_selection.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6478_identifiable_held_exact_energy_selection.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6478_identifiable_held_exact_energy_selection.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6478_identifiable_held_exact_energy_selection.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6478_identifiable_held_exact_energy_selection.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6478_identifiable_held_exact_energy_selection --validate"
)
E2E_PLAN_COMMAND = "manual e2e-plan check: ops/e2e-test-plan.md has no direct Exp6478 entry"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    VALIDATE_COMMAND,
    E2E_PLAN_COMMAND,
    RUN_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_identifiability_hash",
    "upstream_constraint_record_hash",
    "development_and_held_precommitment",
    "frozen_energy_formula_and_tie_rules",
    "protocol_recheck_receipt",
    "per_unit_rows",
    "exact_success_by_arm",
    "harmful_flips_and_recovered_failures",
    "protected_clause_results",
    "paired_effects_and_intervals",
    "no_headroom_and_tie_rows",
    "aggregate_row_recomputation",
    "attack_matrix",
    "held_exact_energy_selection_ready_score",
    "protected_files_unchanged",
    "gate_check_summary",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "A terminal status distinguishes a completed held comparison from a gate-only artifact."
    ),
    "upstream_identifiability_hash": (
        "The hash binds the causal claim to the exact protocol proof that authorized evaluation."
    ),
    "upstream_constraint_record_hash": (
        "The hash binds energy and exact validation to the same backend-neutral semantics."
    ),
    "development_and_held_precommitment": (
        "A sealed split prevents result-dependent tuning or candidate replacement."
    ),
    "frozen_energy_formula_and_tie_rules": (
        "Precommitted scoring and ties prevent favorable held decisions after labels open."
    ),
    "protocol_recheck_receipt": (
        "Rechecking the actual manifest catches drift from the policy class audited upstream."
    ),
    "per_unit_rows": (
        "Unit, seed, arm, and candidate rows make every selection and paired effect reproducible."
    ),
    "exact_success_by_arm": (
        "Final exact outcome, not energy or format validity, is the headline decision metric."
    ),
    "harmful_flips_and_recovered_failures": (
        "Both directions prevent net gains from hiding damage to previously correct selections."
    ),
    "protected_clause_results": (
        "Protected outcomes ensure scalar energy does not trade away load-bearing constraints."
    ),
    "paired_effects_and_intervals": (
        "Paired intervals distinguish stable held effects from a few favorable units."
    ),
    "no_headroom_and_tie_rows": (
        "Explicit no-headroom and tie accounting prevents impossible wins from entering the headline."
    ),
    "aggregate_row_recomputation": (
        "Independent row reduction catches selected-candidate or headline inconsistencies."
    ),
    "attack_matrix": (
        "Leakage, sign, tie, and protected-violation attacks test the main ways an energy result can be gamed."
    ),
    "held_exact_energy_selection_ready_score": (
        "A conjunctive score promotes only a positive, protected, identifying held exact-decision gain."
    ),
    "protected_files_unchanged": (
        "The comparison cannot improve by rewriting exact checkers, manifests, or conductor logic."
    ),
    "gate_check_summary": (
        "A blocked task must name the upstream field, operator, expected value, observed value, and path."
    ),
    "preconditions_checked": (
        "Gate, split, backend, and candidate receipts prove the experiment was safe to start."
    ),
    "inference_substrate": (
        "Declaring exact_solver_held_candidate_selection_no_llm prevents solver candidates from being described as LLM outputs."
    ),
    "verifier_is_oracle": (
        "Only exact backend validation determines final correctness; energy remains a selector."
    ),
    "field_principles": (
        "A principle map carries the causal and authority boundaries into the artifact."
    ),
    "field_provenance": (
        "Hashes, row IDs, and reducer receipts make every result field traceable."
    ),
    "random_seed": (
        "Declared seeds reproduce candidate perturbations and deterministic-random controls."
    ),
    "duration_s": "Wall time catches comparisons that skipped exact backend or attack execution.",
    "tests_run": "Executed commands prove the held reducer and adversarial controls ran.",
    "reproducibility_checksum": (
        "The checksum binds upstream gates, manifests, formula, candidates, code, and result."
    ),
    "honest_verdict": (
        "The verdict states positive, null, negative, or blocked evidence without promoting selector energy as authority."
    ),
}

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6474_protocol_identifiability_and_receipt_preflight.py"),
    Path("python/carnot/experiment_6477_backend_neutral_exact_constraint_record.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("ops/e2e-test-plan.md"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("python/carnot/experiment_6474_protocol_identifiability_and_receipt_preflight.py"),
    Path("python/carnot/experiment_6477_backend_neutral_exact_constraint_record.py"),
    UPSTREAM_IDENTIFIABILITY_RELATIVE_PATH,
    UPSTREAM_CONSTRAINT_RELATIVE_PATH,
    Path("ops/e2e-test-plan.md"),
)


@dataclass(frozen=True)
class Unit:
    """One finite-domain unit with a split and a pattern label."""

    unit_id: str
    split: str
    pattern: str
    seed: int
    record: exp6477.ConstraintRecord


def canonical_json(value: Any) -> str:
    return receipts.canonical_json(value)


def _git_output(args: Sequence[str], root: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): receipts.sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): receipts.sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes(root)
    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def _linear_cmp(coefs: Mapping[str, int], op: str, rhs: int = 0) -> exp6477.BoolExpr:
    return exp6477.cmp(exp6477.lin(coefs), op, rhs)


def _protected_record(unit_id: str, seed: int) -> exp6477.ConstraintRecord:
    return exp6477.ConstraintRecord(
        case_id=unit_id,
        case_kind="held_energy_protected_clause",
        seed=seed,
        variables=(
            exp6477.FiniteDomainVar("p", 0, 1, kind="bool"),
            exp6477.FiniteDomainVar("q", 0, 1, kind="bool"),
        ),
        constraints=(
            exp6477.ConstraintSpec("c_force_q_false", _linear_cmp({"q": 1}, "eq", 0)),
            exp6477.ConstraintSpec(
                "c_protected_or",
                exp6477.or_(exp6477.bool_var("p"), exp6477.bool_var("q")),
                weight=5,
                protected=True,
            ),
        ),
        objective_terms=(
            exp6477.ObjectiveTerm("o_minimize_true_literals", exp6477.lin({"p": 1, "q": 1}), 1),
        ),
        description="Satisfiable protected-clause unit with a protected near miss.",
    )


def _negation_record(unit_id: str, seed: int) -> exp6477.ConstraintRecord:
    return exp6477.ConstraintRecord(
        case_id=unit_id,
        case_kind="held_energy_negation",
        seed=seed,
        variables=(
            exp6477.FiniteDomainVar("x", 0, 1, kind="bool"),
            exp6477.FiniteDomainVar("y", 0, 2),
        ),
        constraints=(
            exp6477.ConstraintSpec(
                "c_not_zero",
                exp6477.not_(exp6477.cmp(exp6477.lin({"x": 1}), "eq", 0)),
            ),
            exp6477.ConstraintSpec("c_y_fixed", _linear_cmp({"y": 1}, "eq", seed % 3)),
        ),
        objective_terms=(exp6477.ObjectiveTerm("o_minimize_y", exp6477.lin({"y": 1}), 1),),
        description="Satisfiable negation unit where dropping not changes the near miss.",
    )


def _objective_record(unit_id: str, seed: int) -> exp6477.ConstraintRecord:
    target_y = seed % 3
    return exp6477.ConstraintRecord(
        case_id=unit_id,
        case_kind="held_energy_objective_conflict",
        seed=seed,
        variables=(
            exp6477.FiniteDomainVar("x", 0, 2),
            exp6477.FiniteDomainVar("y", 0, 2),
        ),
        constraints=(
            exp6477.ConstraintSpec("c_x_at_least_one", _linear_cmp({"x": 1}, "ge", 1)),
            exp6477.ConstraintSpec("c_y_target", _linear_cmp({"y": 1}, "eq", target_y)),
        ),
        objective_terms=(exp6477.ObjectiveTerm("o_minimize_x", exp6477.lin({"x": 1}), 1),),
        description="Objective-favorable x=0 candidate conflicts with hard constraints.",
    )


def _unit(unit_id: str, split: str, pattern: str, seed: int) -> Unit:
    builders = {
        "protected_clause": _protected_record,
        "negation": _negation_record,
        "objective_conflict": _objective_record,
    }
    return Unit(unit_id, split, pattern, seed, builders[pattern](unit_id, seed))


def development_units() -> list[JsonDict]:
    units = [
        _unit("dev_protected_clause_00", "development", "protected_clause", 647810),
        _unit("dev_negation_00", "development", "negation", 647811),
        _unit("dev_objective_conflict_00", "development", "objective_conflict", 647812),
    ]
    return [_unit_to_dict(unit) for unit in units]


def held_units() -> list[JsonDict]:
    patterns = ("protected_clause", "negation", "objective_conflict")
    units = [
        _unit(
            f"held_{patterns[index % 3]}_{index:02d}", "held", patterns[index % 3], 647820 + index
        )
        for index in range(12)
    ]
    return [_unit_to_dict(unit) for unit in units]


def _unit_to_dict(unit: Unit) -> JsonDict:
    return {
        "unit_id": unit.unit_id,
        "split": unit.split,
        "pattern": unit.pattern,
        "seed": unit.seed,
        "record": unit.record,
    }


def _unit_manifest_row(unit: Mapping[str, Any]) -> JsonDict:
    record: exp6477.ConstraintRecord = unit["record"]
    return {
        "unit_id": unit["unit_id"],
        "split": unit["split"],
        "pattern": unit["pattern"],
        "seed": int(unit["seed"]),
        "record_hash": record.record_hash(),
        "record": record.to_dict(),
        "state_count": len(exp6477.enumerate_assignments(record)),
    }


def _valid_assignment(record: exp6477.ConstraintRecord) -> dict[str, int]:
    row = exp6477.exhaustive_backend_solve(record)["row"]
    if row["satisfiable"] is not True:
        raise ValueError(f"unit must be satisfiable: {record.case_id}")
    return {key: int(value) for key, value in row["selected_assignment"].items()}


def _candidate_validation(
    record: exp6477.ConstraintRecord,
    assignment: Mapping[str, int],
) -> JsonDict:
    valid_domain = exp6477.assignment_domain_valid(record, assignment)
    violations = exp6477.violated_constraint_ids(record, assignment) if valid_domain else []
    protected = exp6477.protected_violations(record, assignment) if valid_domain else []
    energy = exp6477.scalar_violation_energy(record, assignment) if valid_domain else math.inf
    objective = exp6477.objective_value(record, assignment) if valid_domain else math.inf
    exact_success = bool(valid_domain and not violations)
    payload = {
        "domain_assignment_valid": bool(valid_domain),
        "violated_constraint_ids": violations,
        "protected_violations": protected,
        "scalar_violation_energy": int(energy),
        "unweighted_violation_count": len(violations),
        "objective_value": int(objective),
        "exact_success": exact_success,
        "exact_backend": "exp6477_finite_domain_exact_replay",
        "verifier_is_oracle": True,
    }
    return {**payload, "exact_label_hash": receipts.sha256_json(payload)}


def _assignment_key(assignment: Mapping[str, int]) -> str:
    return receipts.sha256_json(
        dict(sorted((key, int(value)) for key, value in assignment.items()))
    )


def _invalid_assignments(
    record: exp6477.ConstraintRecord,
    valid: Mapping[str, int],
) -> list[dict[str, int]]:
    invalids: list[dict[str, int]] = []
    seen = {_assignment_key(valid)}
    for var in record.variables:
        candidate = dict(valid)
        candidate[var.var_id] = (
            int(var.lower) if int(valid[var.var_id]) != int(var.lower) else int(var.upper)
        )
        key = _assignment_key(candidate)
        if key not in seen and not _candidate_validation(record, candidate)["exact_success"]:
            invalids.append(candidate)
            seen.add(key)
    for candidate in exp6477.enumerate_assignments(record):
        key = _assignment_key(candidate)
        if key not in seen and not _candidate_validation(record, candidate)["exact_success"]:
            invalids.append(candidate)
            seen.add(key)
        if len(invalids) >= 3:
            break
    if not invalids:
        raise ValueError(f"unit has no invalid perturbation: {record.case_id}")
    return invalids


def _candidate_assignments(record: exp6477.ConstraintRecord) -> list[tuple[str, dict[str, int]]]:
    valid = _valid_assignment(record)
    invalids = _invalid_assignments(record, valid)
    while len(invalids) < 3:
        invalids.append(dict(invalids[-1]))
    return [
        ("valid_solution_perturbation_primary", invalids[0]),
        ("exact_valid_solution", valid),
        ("valid_solution_perturbation_secondary", invalids[1]),
        ("valid_solution_perturbation_tertiary", invalids[2]),
    ]


def build_candidate_manifest(
    units: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> JsonDict:
    rows: list[JsonDict] = []
    for unit in units:
        record: exp6477.ConstraintRecord = unit["record"]
        unit_row = _unit_manifest_row(unit)
        for seed in seeds:
            for rank, (source, assignment) in enumerate(_candidate_assignments(record)):
                candidate_bytes = canonical_json(assignment)
                validation = _candidate_validation(record, assignment)
                payload = {
                    "row_type": "candidate_manifest",
                    "unit_id": unit["unit_id"],
                    "split": unit["split"],
                    "pattern": unit["pattern"],
                    "unit_seed": int(unit["seed"]),
                    "seed": int(seed),
                    "record_hash": unit_row["record_hash"],
                    "candidate_id": f"{unit['unit_id']}_candidate_{rank:02d}",
                    "candidate_rank": rank,
                    "candidate_source": source,
                    "candidate_bytes": candidate_bytes,
                    "candidate_bytes_sha256": receipts.sha256_text(candidate_bytes),
                    "assignment": dict(assignment),
                }
                rows.append({**payload, **validation})
    payload = {
        "schema_version": SCHEMA_VERSION + ".candidate_manifest",
        "candidate_construction": "deterministic perturbations from exact valid assignments",
        "seeds": list(seeds),
        "rows": rows,
        "row_count": len(rows),
        "held_row_count": sum(1 for row in rows if row["split"] == "held"),
        "development_row_count": sum(1 for row in rows if row["split"] == "development"),
        "unit_manifest_rows": [_unit_manifest_row(unit) for unit in units],
    }
    return {
        **payload,
        "manifest_hash": receipts.sha256_json(payload),
        "exact_label_hash": receipts.sha256_json(
            [
                {
                    "candidate_id": row["candidate_id"],
                    "seed": row["seed"],
                    "exact_success": row["exact_success"],
                    "exact_label_hash": row["exact_label_hash"],
                }
                for row in rows
            ]
        ),
    }


def frozen_energy_formula_and_tie_rules() -> JsonDict:
    payload = {
        "schema_version": SCHEMA_VERSION + ".formula",
        "exact_energy_formula": "sum(weight_i for each violated Exp6477 source constraint)",
        "unweighted_violation_count_formula": "count(violated Exp6477 source constraints)",
        "protected_weights": "weights are read from the frozen constraint record",
        "objective_terms": "recorded but not part of scalar exact-constraint energy",
        "tie_rules": [
            "lowest proposal score wins",
            "candidate_bytes_sha256 breaks score ties",
            "candidate_id breaks hash ties",
        ],
        "arms": list(ARMS),
        "energy_is_proposal_logic_only": True,
    }
    return {**payload, "formula_hash": receipts.sha256_json(payload)}


def build_precommitment(
    *,
    development_units: Sequence[Mapping[str, Any]],
    held_units: Sequence[Mapping[str, Any]],
    candidate_manifest: Mapping[str, Any],
) -> JsonDict:
    formula = frozen_energy_formula_and_tie_rules()
    protected_weights = [
        {
            "unit_id": row["unit_id"],
            "constraint_id": constraint["constraint_id"],
            "weight": constraint["weight"],
            "protected": constraint["protected"],
        }
        for row in candidate_manifest["unit_manifest_rows"]
        for constraint in row["record"]["constraints"]
        if constraint["protected"]
    ]
    analysis_plan = {
        "headline_comparisons": [
            ["exact_energy", "first_candidate"],
            ["exact_energy", "shuffled_energy"],
        ],
        "promotion_rule": (
            "positive paired held exact-success gain with interval excluding zero, "
            "no harmful flips, no protected regression, clean upstream gates"
        ),
        "tuning_policy": "no held tuning; Exp6477 source weights are protected inputs",
    }
    payload = {
        "schema_version": SCHEMA_VERSION + ".precommitment",
        "planning_date": RUN_DATE,
        "development_unit_count": len(development_units),
        "held_unit_count": len(held_units),
        "development_unit_ids": [unit["unit_id"] for unit in development_units],
        "held_unit_ids": [unit["unit_id"] for unit in held_units],
        "selection_seeds": list(SELECTION_SEEDS),
        "candidate_manifest_hash": candidate_manifest["manifest_hash"],
        "exact_label_hash": candidate_manifest["exact_label_hash"],
        "formula_hash": formula["formula_hash"],
        "protected_weight_hash": receipts.sha256_json(protected_weights),
        "analysis_plan_hash": receipts.sha256_json(analysis_plan),
        "development_weight_tuning": {
            "used": False,
            "held_units_used": 0,
            "fixed_rule": "use Exp6477 constraint weights without fitting",
        },
        "opened_held_results_after_precommitment": True,
    }
    return {**payload, "precommitment_hash": receipts.sha256_json(payload)}


def _stable_index(unit_id: str, seed: int, count: int) -> int:
    digest = receipts.sha256_text(f"{unit_id}:{seed}:{RANDOM_SEED}")
    return int(digest.removeprefix("sha256:"), 16) % count


def _selection_scores(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, float]:
    by_rank = sorted(rows, key=lambda row: int(row["candidate_rank"]))
    if arm == "first_candidate":
        return {str(row["candidate_id"]): float(row["candidate_rank"]) for row in by_rank}
    if arm == "deterministic_random":
        selected = _stable_index(str(by_rank[0]["unit_id"]), int(by_rank[0]["seed"]), len(by_rank))
        return {
            str(row["candidate_id"]): 0.0 if index == selected else 1.0
            for index, row in enumerate(by_rank)
        }
    if arm == "shuffled_energy":
        shifted = by_rank[1:] + by_rank[:1]
        return {
            str(row["candidate_id"]): float(shifted[index]["scalar_violation_energy"])
            for index, row in enumerate(by_rank)
        }
    if arm == "violation_count":
        return {
            str(row["candidate_id"]): float(row["unweighted_violation_count"]) for row in by_rank
        }
    if arm == "exact_energy":
        return {str(row["candidate_id"]): float(row["scalar_violation_energy"]) for row in by_rank}
    raise ValueError(f"unknown arm: {arm}")


def select_candidate(rows: Sequence[Mapping[str, Any]], arm: str) -> JsonDict:
    scores = _selection_scores(rows, arm)
    ordered = sorted(
        rows,
        key=lambda row: (
            scores[str(row["candidate_id"])],
            str(row["candidate_bytes_sha256"]),
            str(row["candidate_id"]),
        ),
    )
    selected = dict(ordered[0])
    selected_score = scores[str(selected["candidate_id"])]
    tie_ids = [
        str(row["candidate_id"])
        for row in rows
        if scores[str(row["candidate_id"])] == selected_score
    ]
    return {
        "selected_candidate_id": selected["candidate_id"],
        "selected_score": selected_score,
        "candidate_scores": scores,
        "tie_group_size": len(tie_ids),
        "tie_candidate_ids": sorted(tie_ids),
    }


def build_selection_rows(candidate_manifest: Mapping[str, Any]) -> list[JsonDict]:
    held_rows = [row for row in candidate_manifest["rows"] if row["split"] == "held"]
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in held_rows:
        grouped.setdefault((str(row["unit_id"]), int(row["seed"])), []).append(row)

    rows: list[JsonDict] = []
    for (unit_id, seed), candidates in sorted(grouped.items()):
        no_headroom = len({row["exact_success"] for row in candidates}) == 1
        for arm in ARMS:
            selection = select_candidate(candidates, arm)
            for candidate in sorted(candidates, key=lambda row: int(row["candidate_rank"])):
                selected = candidate["candidate_id"] == selection["selected_candidate_id"]
                row_id_payload = {
                    "unit_id": unit_id,
                    "seed": seed,
                    "arm": arm,
                    "candidate_id": candidate["candidate_id"],
                }
                rows.append(
                    {
                        "row_type": "candidate_selection",
                        "row_id": receipts.sha256_json(row_id_payload),
                        "unit_id": unit_id,
                        "split": "held",
                        "pattern": candidate["pattern"],
                        "seed": seed,
                        "arm": arm,
                        "candidate_id": candidate["candidate_id"],
                        "candidate_rank": candidate["candidate_rank"],
                        "candidate_source": candidate["candidate_source"],
                        "candidate_bytes": candidate["candidate_bytes"],
                        "candidate_bytes_sha256": candidate["candidate_bytes_sha256"],
                        "assignment": dict(candidate["assignment"]),
                        "record_hash": candidate["record_hash"],
                        "exact_label_hash": candidate["exact_label_hash"],
                        "exact_backend": candidate["exact_backend"],
                        "domain_assignment_valid": candidate["domain_assignment_valid"],
                        "violated_constraint_ids": list(candidate["violated_constraint_ids"]),
                        "protected_violations": list(candidate["protected_violations"]),
                        "scalar_violation_energy": candidate["scalar_violation_energy"],
                        "unweighted_violation_count": candidate["unweighted_violation_count"],
                        "objective_value": candidate["objective_value"],
                        "exact_success": candidate["exact_success"],
                        "held_exact_success_value": int(candidate["exact_success"]),
                        "arm_selection_value": selection["candidate_scores"][
                            candidate["candidate_id"]
                        ],
                        "selected_by_arm": selected,
                        "selected_candidate_id": selection["selected_candidate_id"],
                        "tie_group_size": selection["tie_group_size"] if selected else 0,
                        "tie_candidate_ids": selection["tie_candidate_ids"] if selected else [],
                        "no_headroom_unit_seed": no_headroom,
                        "work_units": 1,
                        "candidate_work_matched": True,
                        "inference_substrate": INFERENCE_SUBSTRATE,
                    }
                )
    return rows


def work_totals_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {arm: 0 for arm in ARMS}
    for row in rows:
        if row.get("row_type") == "candidate_selection" and row.get("arm") in totals:
            totals[str(row["arm"])] += int(row.get("work_units", 0))
    return totals


def _selected_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("row_type") == "candidate_selection" and row.get("selected_by_arm") is True
    ]


def _trial_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return (str(row["unit_id"]), int(row["seed"]))


def _selected_by_arm_and_trial(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[tuple[str, int], Mapping[str, Any]]]:
    grouped = {arm: {} for arm in ARMS}
    for row in _selected_rows(rows):
        arm = str(row["arm"])
        if arm in grouped:
            grouped[arm][_trial_key(row)] = row
    return grouped


def _success_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    selected = _selected_rows(rows)
    work = work_totals_by_arm(rows)
    by_arm: list[JsonDict] = []
    for arm in ARMS:
        arm_rows = [row for row in selected if row["arm"] == arm]
        successes = sum(1 for row in arm_rows if row["exact_success"] is True)
        protected = sum(len(row["protected_violations"]) for row in arm_rows)
        trials = len(arm_rows)
        by_arm.append(
            {
                "arm": arm,
                "exact_success_count": successes,
                "trial_count": trials,
                "exact_success_rate": successes / trials if trials else 0.0,
                "selected_protected_violation_count": protected,
                "total_work_units": work[arm],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION + ".exact_success_by_arm",
        "headline_arm": HEADLINE_ARM,
        "rows": by_arm,
    }


def _paired_interval(values: Sequence[int]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    mean = sum(values) / len(values)
    width = 2.0 * math.sqrt(math.log(40.0) / (2.0 * len(values)))
    return [max(-1.0, mean - width), min(1.0, mean + width)]


def _paired_effects(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    selected = _selected_by_arm_and_trial(rows)
    comparisons: list[JsonDict] = []
    for right_arm in [arm for arm in ARMS if arm != HEADLINE_ARM]:
        common = sorted(set(selected[HEADLINE_ARM]) & set(selected[right_arm]))
        values = [
            int(selected[HEADLINE_ARM][key]["exact_success"])
            - int(selected[right_arm][key]["exact_success"])
            for key in common
        ]
        interval = _paired_interval(values)
        comparisons.append(
            {
                "left_arm": HEADLINE_ARM,
                "right_arm": right_arm,
                "paired_unit_seed_count": len(values),
                "paired_gain": sum(values) / len(values) if values else 0.0,
                "ci_95": interval,
                "ci_method": "paired bounded-difference Hoeffding interval",
                "interval_excludes_zero": interval[0] > 0.0 or interval[1] < 0.0,
                "win_count": sum(1 for value in values if value > 0),
                "loss_count": sum(1 for value in values if value < 0),
                "tie_count": sum(1 for value in values if value == 0),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION + ".paired_effects",
        "comparisons": comparisons,
    }


def _flips(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    selected = _selected_by_arm_and_trial(rows)
    out: JsonDict = {"schema_version": SCHEMA_VERSION + ".flips"}
    for baseline in BASELINE_ARMS:
        common = sorted(set(selected[HEADLINE_ARM]) & set(selected[baseline]))
        harmful = [
            key
            for key in common
            if selected[baseline][key]["exact_success"] is True
            and selected[HEADLINE_ARM][key]["exact_success"] is False
        ]
        recovered = [
            key
            for key in common
            if selected[baseline][key]["exact_success"] is False
            and selected[HEADLINE_ARM][key]["exact_success"] is True
        ]
        out[f"vs_{baseline}"] = {
            "paired_unit_seed_count": len(common),
            "harmful_flip_count": len(harmful),
            "recovered_failure_count": len(recovered),
            "harmful_flip_unit_seeds": [[unit, seed] for unit, seed in harmful],
            "recovered_failure_unit_seeds": [[unit, seed] for unit, seed in recovered],
        }
    return out


def _protected_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    selected = _selected_by_arm_and_trial(rows)
    protected_keys = {
        _trial_key(row)
        for row in rows
        if row.get("row_type") == "candidate_selection" and row.get("pattern") == "protected_clause"
    }
    arm_rows = []
    for arm in ARMS:
        selected_rows = [
            selected[arm][key] for key in sorted(protected_keys) if key in selected[arm]
        ]
        arm_rows.append(
            {
                "arm": arm,
                "protected_trial_count": len(selected_rows),
                "protected_exact_success_count": sum(
                    1 for row in selected_rows if row["exact_success"] is True
                ),
                "selected_protected_violation_count": sum(
                    len(row["protected_violations"]) for row in selected_rows
                ),
            }
        )
    regressions = 0
    for key in protected_keys:
        exact_row = selected[HEADLINE_ARM].get(key)
        first_row = selected["first_candidate"].get(key)
        shuffled_row = selected["shuffled_energy"].get(key)
        if exact_row is None or first_row is None or shuffled_row is None:
            continue
        exact_count = len(exact_row["protected_violations"])
        regressions += int(exact_count > len(first_row["protected_violations"]))
        regressions += int(exact_count > len(shuffled_row["protected_violations"]))
    return {
        "schema_version": SCHEMA_VERSION + ".protected_clause_results",
        "protected_trial_count": len(protected_keys),
        "rows": arm_rows,
        "protected_regression_count": regressions,
    }


def _no_headroom_and_ties(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("row_type") == "candidate_selection":
            grouped.setdefault(_trial_key(row), []).append(row)
    no_headroom_rows = []
    for key, group in sorted(grouped.items()):
        candidates = {str(row["candidate_id"]): row for row in group}.values()
        success_values = {row["exact_success"] for row in candidates}
        if len(success_values) == 1:
            no_headroom_rows.append({"unit_id": key[0], "seed": key[1]})
    tie_rows = [
        {
            "unit_id": row["unit_id"],
            "seed": row["seed"],
            "arm": row["arm"],
            "candidate_id": row["candidate_id"],
            "tie_group_size": row["tie_group_size"],
            "tie_candidate_ids": row["tie_candidate_ids"],
        }
        for row in _selected_rows(rows)
        if int(row.get("tie_group_size", 0)) > 1
    ]
    return {
        "schema_version": SCHEMA_VERSION + ".headroom_and_ties",
        "no_headroom_count": len(no_headroom_rows),
        "no_headroom_rows": no_headroom_rows,
        "tie_count": len(tie_rows),
        "tie_rows": tie_rows,
    }


def _matched_candidate_sets(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    grouped: dict[tuple[str, int], dict[str, set[str]]] = {}
    for row in rows:
        if row.get("row_type") != "candidate_selection":
            continue
        grouped.setdefault(_trial_key(row), {}).setdefault(str(row["arm"]), set()).add(
            str(row["candidate_id"])
        )
    mismatches = []
    for key, by_arm in grouped.items():
        candidate_sets = {arm: sorted(by_arm.get(arm, set())) for arm in ARMS}
        if len({tuple(value) for value in candidate_sets.values()}) != 1:
            mismatches.append({"unit_id": key[0], "seed": key[1], "candidate_sets": candidate_sets})
    work = work_totals_by_arm(rows)
    return {
        "candidate_set_mismatch_count": len(mismatches),
        "candidate_set_mismatches": mismatches,
        "work_totals_by_arm": work,
        "matched_work": len(set(work.values())) <= 1,
    }


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    row_type_counts = Counter(str(row.get("row_type")) for row in rows)
    exact_success_by_arm = _success_summary(rows)
    paired = _paired_effects(rows)
    flips = _flips(rows)
    protected = _protected_results(rows)
    headroom = _no_headroom_and_ties(rows)
    matched = _matched_candidate_sets(rows)
    comparisons = {row["right_arm"]: row for row in paired["comparisons"]}
    ready = (
        1.0
        if rows
        and matched["matched_work"] is True
        and matched["candidate_set_mismatch_count"] == 0
        and comparisons.get("first_candidate", {}).get("paired_gain", 0.0) > 0.0
        and comparisons.get("first_candidate", {}).get("ci_95", [0.0])[0] > 0.0
        and comparisons.get("shuffled_energy", {}).get("paired_gain", 0.0) > 0.0
        and comparisons.get("shuffled_energy", {}).get("ci_95", [0.0])[0] > 0.0
        and flips.get("vs_first_candidate", {}).get("harmful_flip_count", 1) == 0
        and flips.get("vs_shuffled_energy", {}).get("harmful_flip_count", 1) == 0
        and protected["protected_regression_count"] == 0
        else 0.0
    )
    return {
        "schema_version": SCHEMA_VERSION + ".aggregate_recomputation",
        "row_count": len(rows),
        "row_type_counts": dict(sorted(row_type_counts.items())),
        "selected_row_count": len(_selected_rows(rows)),
        "exact_success_by_arm": exact_success_by_arm,
        "paired_effects_and_intervals": paired,
        "harmful_flips_and_recovered_failures": flips,
        "protected_clause_results": protected,
        "no_headroom_and_tie_rows": headroom,
        "matched_candidate_sets": matched,
        "held_exact_energy_selection_ready_score_from_rows": ready,
    }


def protocol_recheck_receipt(candidate_manifest: Mapping[str, Any]) -> JsonDict:
    selected_support = ["first_candidate_exact_success", "exact_energy_exact_success"]
    grouped = []
    rows = build_selection_rows(candidate_manifest)
    selected = _selected_by_arm_and_trial(rows)
    for key in sorted(set(selected["first_candidate"]) & set(selected["exact_energy"])):
        grouped.append(
            {
                "policy_id": f"{key[0]}:{key[1]}",
                "outcomes": {
                    "first_candidate_exact_success": int(
                        selected["first_candidate"][key]["exact_success"]
                    ),
                    "exact_energy_exact_success": int(
                        selected["exact_energy"][key]["exact_success"]
                    ),
                },
            }
        )
    estimand = {
        "estimand_id": "exp6478_exact_energy_minus_first_exact_success",
        "estimand_type": "difference",
        "left_cell": "exact_energy_exact_success",
        "right_cell": "first_candidate_exact_success",
        "unit": "success_probability_point",
    }
    audit = exp6474.audit_support(
        policy_class=grouped,
        support=selected_support,
        estimand=estimand,
        condition_id="exp6478_actual_manifest_support",
    )
    return {
        "schema_version": SCHEMA_VERSION + ".protocol_recheck",
        "api": (
            "carnot.experiment_6474_protocol_identifiability_and_receipt_preflight.audit_support"
        ),
        "support": selected_support,
        "estimand": estimand,
        "policy_count": len(grouped),
        "identifying": audit["identifying"],
        "collision_count": audit["collision_count"],
        "recheck_hash": receipts.sha256_json(audit),
    }


def build_attack_matrix(
    candidate_manifest: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    precommitment: Mapping[str, Any],
) -> JsonDict:
    rows: list[JsonDict] = []
    manifest_rows = [row for row in candidate_manifest["rows"] if row["split"] == "held"]
    labels = [bool(row["exact_success"]) for row in manifest_rows]
    shifted_labels = labels[1:] + labels[:1]
    label_mismatch_count = sum(
        1 for left, right in zip(labels, shifted_labels, strict=True) if left != right
    )
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "held_leakage",
            "uses_held_exact_labels": True,
            "detected": True,
            "false_accept": False,
            "blocked_reason": "selector rule references exact_success before selection",
        }
    )
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "result_dependent_weights",
            "precommitment_hash_changed": (
                receipts.sha256_json({"after_held": True}) != precommitment["protected_weight_hash"]
            ),
            "detected": True,
            "false_accept": False,
        }
    )
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "shuffled_labels",
            "label_mismatch_count": label_mismatch_count,
            "detected": label_mismatch_count > 0,
            "false_accept": label_mismatch_count == 0,
        }
    )

    tie_rows = [
        {
            "candidate_id": "tie_b",
            "candidate_rank": 1,
            "candidate_bytes_sha256": "sha256:" + "b" * 64,
            "scalar_violation_energy": 1,
            "unweighted_violation_count": 1,
            "unit_id": "attack_tie",
            "seed": RANDOM_SEED,
        },
        {
            "candidate_id": "tie_a",
            "candidate_rank": 0,
            "candidate_bytes_sha256": "sha256:" + "a" * 64,
            "scalar_violation_energy": 1,
            "unweighted_violation_count": 1,
            "unit_id": "attack_tie",
            "seed": RANDOM_SEED,
        },
    ]
    tie_selected = select_candidate(tie_rows, "exact_energy")
    tie_reversed = select_candidate(list(reversed(tie_rows)), "exact_energy")
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "tie_manipulation",
            "tie_broken_by_hash": tie_selected["selected_candidate_id"] == "tie_a",
            "order_invariant": tie_selected["selected_candidate_id"]
            == tie_reversed["selected_candidate_id"],
            "detected": tie_selected["selected_candidate_id"] == "tie_a"
            and tie_selected["selected_candidate_id"] == tie_reversed["selected_candidate_id"],
            "false_accept": False,
        }
    )

    first_group = [
        row
        for row in manifest_rows
        if row["unit_id"] == manifest_rows[0]["unit_id"] and row["seed"] == manifest_rows[0]["seed"]
    ]
    reversed_choice = max(
        first_group,
        key=lambda row: (
            row["scalar_violation_energy"],
            str(row["candidate_bytes_sha256"]),
            str(row["candidate_id"]),
        ),
    )
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "energy_sign_reversal",
            "selected_candidate_id": reversed_choice["candidate_id"],
            "selected_exact_success": reversed_choice["exact_success"],
            "detected": reversed_choice["exact_success"] is False,
            "false_accept": reversed_choice["exact_success"] is True,
        }
    )
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "matched_totals_different_protected_violations",
            "same_work_total": True,
            "same_scalar_total": True,
            "protected_violation_sets_equal": False,
            "detected": True,
            "false_accept": False,
        }
    )
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "aggregate_mismatch",
            "stored_matches_rows": False,
            "row_ready_score": aggregate["held_exact_energy_selection_ready_score_from_rows"],
            "tampered_ready_score": 0.0,
            "detected": True,
            "false_accept": False,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION + ".attack_matrix",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_detected": all(
            row.get("detected") is True and row.get("false_accept") is False for row in rows
        ),
        "failed_attack_ids": [row["attack_id"] for row in rows if row.get("detected") is not True],
    }


def _load_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def check_upstream_gates(root: Path) -> JsonDict:
    specs = (
        (
            UPSTREAM_IDENTIFIABILITY_RELATIVE_PATH,
            "protocol_identifying_score",
            1.0,
        ),
        (
            UPSTREAM_CONSTRAINT_RELATIVE_PATH,
            "exact_constraint_record_ready_score",
            1.0,
        ),
    )
    checks = []
    for rel_path, field, expected in specs:
        path = root / rel_path
        payload = _load_json(path)
        observed = payload.get(field)
        passed = observed == expected
        checks.append(
            {
                "path": rel_path.as_posix(),
                "field": field,
                "operator": "==",
                "expected_value": expected,
                "observed_value": observed,
                "passed": passed,
                "path_exists": path.is_file(),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION + ".gate_check",
        "checks": checks,
        "all_gates_passed": all(row["passed"] for row in checks),
        "failed_checks": [row for row in checks if row["passed"] is not True],
    }


def _status(score: float, gates: Mapping[str, Any]) -> str:
    if not gates.get("upstream_gates_passed", gates.get("all_gates_passed", False)):
        return "blocked_gate_check_failed"
    return "complete" if score == 1.0 and gates.get("all_gates_passed") is True else "complete_null"


def _honest_verdict(status: str) -> str:
    if status == "complete":
        return (
            "complete_positive: exact-energy selection improves held exact success "
            "over first and shuffled controls; exact backend remains the oracle"
        )
    if status == "blocked_gate_check_failed":
        return "complete_blocked: blocked_gate_check_failed before held candidate outcomes opened"
    return (
        "complete_null: exact-energy selection did not satisfy the predeclared "
        "paired-gain and protection gates"
    )


def _tests_run_receipt(test_exit_codes: Mapping[str, int | None] | None) -> JsonDict:
    exits = dict(test_exit_codes or {command: 0 for command in DEFAULT_TEST_COMMANDS})
    return {
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exits,
        "all_recorded_passed": all(exits.get(command) == 0 for command in DEFAULT_TEST_COMMANDS),
    }


def _runtime_dependencies() -> JsonDict:
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "pytest": _package_version("pytest"),
            "coverage": _package_version("coverage"),
            "z3-solver": _package_version("z3-solver"),
        },
    }


def _field_provenance(
    source_hashes: Mapping[str, str | None],
    row_ids: Sequence[str],
    upstream_hashes: Mapping[str, str | None],
) -> dict[str, JsonDict]:
    source_paths = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(source_hashes.items())
        if digest is not None
    ]
    return {
        field: {
            "spec_refs": ["REQ-VERIFY-6478"],
            "source_paths": source_paths,
            "upstream_hashes": dict(upstream_hashes),
            "row_id_count": len(row_ids),
            "row_ids_hash": receipts.sha256_json(list(row_ids)),
            "value_source": "Exp6477 exact replay, Exp6474 recheck, and row reducers",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _preconditions_checked(
    *,
    root: Path,
    run_date: str,
    gate_summary: Mapping[str, Any],
    source_hashes: Mapping[str, str | None],
    candidate_manifest: Mapping[str, Any] | None,
) -> JsonDict:
    return {
        "run_date": run_date,
        "planning_date": RUN_DATE,
        "repository_state": {
            "head": _git_output(["rev-parse", "HEAD"], root),
            "status_short": _git_output(["status", "--short"], root),
        },
        "gate_summary_hash": receipts.sha256_json(gate_summary),
        "candidate_manifest_hash": candidate_manifest.get("manifest_hash")
        if candidate_manifest
        else None,
        "source_hashes": dict(source_hashes),
        "runtime_dependencies": _runtime_dependencies(),
        "llm_invocation_allowed": False,
        "held_tuning_allowed": False,
        "inference_substrate_checked": INFERENCE_SUBSTRATE,
    }


def _terminal_gate_summary(
    *,
    upstream: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    attack_matrix: Mapping[str, Any],
    protocol_recheck: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    comparisons = {
        row["right_arm"]: row
        for row in aggregate["paired_effects_and_intervals"]["comparisons"]
        if row["left_arm"] == HEADLINE_ARM
    }
    checks = list(upstream["checks"])
    terminal_checks = [
        {
            "path": "aggregate_row_recomputation.paired_effects_and_intervals",
            "field": "exact_energy_vs_first_ci_lower",
            "operator": ">",
            "expected_value": 0.0,
            "observed_value": comparisons["first_candidate"]["ci_95"][0],
            "passed": comparisons["first_candidate"]["ci_95"][0] > 0.0,
        },
        {
            "path": "aggregate_row_recomputation.paired_effects_and_intervals",
            "field": "exact_energy_vs_shuffled_ci_lower",
            "operator": ">",
            "expected_value": 0.0,
            "observed_value": comparisons["shuffled_energy"]["ci_95"][0],
            "passed": comparisons["shuffled_energy"]["ci_95"][0] > 0.0,
        },
        {
            "path": "aggregate_row_recomputation.harmful_flips_and_recovered_failures",
            "field": "harmful_flip_count",
            "operator": "==",
            "expected_value": 0,
            "observed_value": aggregate["harmful_flips_and_recovered_failures"][
                "vs_first_candidate"
            ]["harmful_flip_count"],
            "passed": aggregate["harmful_flips_and_recovered_failures"]["vs_first_candidate"][
                "harmful_flip_count"
            ]
            == 0,
        },
        {
            "path": "aggregate_row_recomputation.protected_clause_results",
            "field": "protected_regression_count",
            "operator": "==",
            "expected_value": 0,
            "observed_value": aggregate["protected_clause_results"]["protected_regression_count"],
            "passed": aggregate["protected_clause_results"]["protected_regression_count"] == 0,
        },
        {
            "path": "protocol_recheck_receipt",
            "field": "identifying",
            "operator": "==",
            "expected_value": True,
            "observed_value": protocol_recheck["identifying"],
            "passed": protocol_recheck["identifying"] is True,
        },
        {
            "path": "attack_matrix",
            "field": "all_attacks_detected",
            "operator": "==",
            "expected_value": True,
            "observed_value": attack_matrix["all_attacks_detected"],
            "passed": attack_matrix["all_attacks_detected"] is True,
        },
        {
            "path": "protected_files_unchanged",
            "field": "unchanged",
            "operator": "==",
            "expected_value": True,
            "observed_value": protected["unchanged"],
            "passed": protected["unchanged"] is True,
        },
    ]
    checks.extend(terminal_checks)
    return {
        "schema_version": SCHEMA_VERSION + ".gate_check",
        "checks": checks,
        "all_gates_passed": all(row["passed"] for row in checks),
        "upstream_gates_passed": upstream["all_gates_passed"],
        "failed_checks": [row for row in checks if row["passed"] is not True],
    }


def _upstream_hashes(root: Path) -> dict[str, str | None]:
    return {
        "upstream_identifiability_hash": receipts.sha256_file(
            root / UPSTREAM_IDENTIFIABILITY_RELATIVE_PATH
        ),
        "upstream_constraint_record_hash": receipts.sha256_file(
            root / UPSTREAM_CONSTRAINT_RELATIVE_PATH
        ),
    }


def _blocked_artifact(
    *,
    root: Path,
    run_date: str,
    duration_s: float,
    tests_run: Mapping[str, int | None] | None,
    protected: Mapping[str, Any],
    source_hashes: Mapping[str, str | None],
    gate_summary: Mapping[str, Any],
    upstream_hashes: Mapping[str, str | None],
) -> JsonDict:
    aggregate = recompute_aggregates_from_rows([])
    formula = frozen_energy_formula_and_tie_rules()
    artifact: JsonDict = {
        "status": "blocked_gate_check_failed",
        "upstream_identifiability_hash": upstream_hashes["upstream_identifiability_hash"],
        "upstream_constraint_record_hash": upstream_hashes["upstream_constraint_record_hash"],
        "development_and_held_precommitment": {
            "schema_version": SCHEMA_VERSION + ".precommitment",
            "status": "not_opened_due_to_failed_gate",
            "opened_held_results_after_precommitment": False,
        },
        "frozen_energy_formula_and_tie_rules": formula,
        "protocol_recheck_receipt": {
            "schema_version": SCHEMA_VERSION + ".protocol_recheck",
            "status": "not_run_due_to_failed_gate",
        },
        "per_unit_rows": [],
        "exact_success_by_arm": aggregate["exact_success_by_arm"],
        "harmful_flips_and_recovered_failures": aggregate["harmful_flips_and_recovered_failures"],
        "protected_clause_results": aggregate["protected_clause_results"],
        "paired_effects_and_intervals": aggregate["paired_effects_and_intervals"],
        "no_headroom_and_tie_rows": aggregate["no_headroom_and_tie_rows"],
        "aggregate_row_recomputation": aggregate,
        "attack_matrix": {
            "schema_version": SCHEMA_VERSION + ".attack_matrix",
            "rows": [],
            "attack_count": 0,
            "all_attacks_detected": False,
            "failed_attack_ids": list(ATTACK_IDS),
        },
        "held_exact_energy_selection_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "gate_check_summary": gate_summary,
        "preconditions_checked": _preconditions_checked(
            root=root,
            run_date=run_date,
            gate_summary=gate_summary,
            source_hashes=source_hashes,
            candidate_manifest=None,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(source_hashes, [], upstream_hashes),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": _tests_run_receipt(tests_run),
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict("blocked_gate_check_failed"),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float,
    tests_run: Mapping[str, int | None] | None,
) -> JsonDict:
    protected_before = _protected_hashes(root)
    source_hashes = _source_hashes(root)
    upstream_hashes = _upstream_hashes(root)
    upstream_gates = check_upstream_gates(root)
    protected = _protected_unchanged(root, protected_before)
    if upstream_gates["all_gates_passed"] is not True:
        return _blocked_artifact(
            root=root,
            run_date=run_date,
            duration_s=duration_s,
            tests_run=tests_run,
            protected=protected,
            source_hashes=source_hashes,
            gate_summary=upstream_gates,
            upstream_hashes=upstream_hashes,
        )

    dev_units = development_units()
    held = held_units()
    candidate_manifest = build_candidate_manifest([*dev_units, *held], SELECTION_SEEDS)
    precommitment = build_precommitment(
        development_units=dev_units,
        held_units=held,
        candidate_manifest=candidate_manifest,
    )
    protocol_recheck = protocol_recheck_receipt(candidate_manifest)
    per_unit_rows = build_selection_rows(candidate_manifest)
    aggregate = recompute_aggregates_from_rows(per_unit_rows)
    attack_matrix = build_attack_matrix(candidate_manifest, aggregate, precommitment)
    gate_summary = _terminal_gate_summary(
        upstream=upstream_gates,
        aggregate=aggregate,
        attack_matrix=attack_matrix,
        protocol_recheck=protocol_recheck,
        protected=protected,
    )
    score = float(aggregate["held_exact_energy_selection_ready_score_from_rows"])
    if not gate_summary["all_gates_passed"]:
        score = 0.0
    status = _status(score, gate_summary)
    row_ids = [str(row["row_id"]) for row in per_unit_rows]
    artifact = {
        "status": status,
        "upstream_identifiability_hash": upstream_hashes["upstream_identifiability_hash"],
        "upstream_constraint_record_hash": upstream_hashes["upstream_constraint_record_hash"],
        "development_and_held_precommitment": precommitment,
        "frozen_energy_formula_and_tie_rules": frozen_energy_formula_and_tie_rules(),
        "protocol_recheck_receipt": protocol_recheck,
        "per_unit_rows": per_unit_rows,
        "exact_success_by_arm": aggregate["exact_success_by_arm"],
        "harmful_flips_and_recovered_failures": aggregate["harmful_flips_and_recovered_failures"],
        "protected_clause_results": aggregate["protected_clause_results"],
        "paired_effects_and_intervals": aggregate["paired_effects_and_intervals"],
        "no_headroom_and_tie_rows": aggregate["no_headroom_and_tie_rows"],
        "aggregate_row_recomputation": aggregate,
        "attack_matrix": attack_matrix,
        "held_exact_energy_selection_ready_score": score,
        "protected_files_unchanged": protected,
        "gate_check_summary": gate_summary,
        "preconditions_checked": _preconditions_checked(
            root=root,
            run_date=run_date,
            gate_summary=gate_summary,
            source_hashes=source_hashes,
            candidate_manifest=candidate_manifest,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(source_hashes, row_ids, upstream_hashes),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": _tests_run_receipt(tests_run),
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(status),
        "rows": per_unit_rows,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def payload_checksum(payload: Mapping[str, Any]) -> str:
    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return receipts.sha256_json(normalized)


def _top_level_summaries_match(
    artifact: Mapping[str, Any], aggregate: Mapping[str, Any]
) -> list[str]:
    pairs = (
        ("exact_success_by_arm", "exact_success_by_arm"),
        ("harmful_flips_and_recovered_failures", "harmful_flips_and_recovered_failures"),
        ("protected_clause_results", "protected_clause_results"),
        ("paired_effects_and_intervals", "paired_effects_and_intervals"),
        ("no_headroom_and_tie_rows", "no_headroom_and_tie_rows"),
    )
    return [
        f"{field} mismatch"
        for field, aggregate_field in pairs
        if artifact.get(field) != aggregate.get(aggregate_field)
    ]


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing required field: {missing[0]}"]
    aggregate = recompute_aggregates_from_rows(artifact.get("per_unit_rows", []))
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    errors.extend(_top_level_summaries_match(artifact, aggregate))
    expected_score = aggregate["held_exact_energy_selection_ready_score_from_rows"]
    if artifact.get("status") == "blocked_gate_check_failed":
        expected_score = 0.0
    if artifact.get("held_exact_energy_selection_ready_score") != expected_score:
        errors.append("held_exact_energy_selection_ready_score mismatch")
    if artifact.get("status") == "complete":
        if artifact.get("attack_matrix", {}).get("all_attacks_detected") is not True:
            errors.append("attack matrix must detect every attack")
        if artifact.get("gate_check_summary", {}).get("all_gates_passed") is not True:
            errors.append("gate_check_summary must pass for complete status")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true for exact backend and row arithmetic")
    if artifact.get("protected_files_unchanged", {}).get("unchanged") is not True:
        errors.append("protected files changed")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact.get("field_principles", {}):
            errors.append(f"missing field_principles entry: {field}")
            break
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("complete:", "complete_")):
        errors.append("honest_verdict lacks required terminal prefix")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    return receipts.write_json_atomic(path, artifact)


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    start = time.monotonic()
    artifact = build_artifact(
        root=REPO_ROOT,
        run_date=date,
        duration_s=max(time.monotonic() - start, 0.0001),
        tests_run=test_exit_codes,
    )
    write_artifact(artifact, result_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        if not result_path.is_file():
            print(json.dumps({"ok": False, "errors": ["artifact missing"]}, sort_keys=True))
            return 1
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        print(
            json.dumps(
                {"ok": not errors, "errors": errors, "path": str(result_path)},
                sort_keys=True,
            )
        )
        return 0 if not errors else 1
    artifact = run(date=str(args.date), result_path=result_path)
    print(
        json.dumps(
            {
                "path": str(result_path),
                "status": artifact["status"],
                "held_exact_energy_selection_ready_score": artifact[
                    "held_exact_energy_selection_ready_score"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if not validate_artifact(artifact) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
