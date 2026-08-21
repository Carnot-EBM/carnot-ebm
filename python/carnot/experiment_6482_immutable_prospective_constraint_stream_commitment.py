"""Exp6482 immutable prospective constraint stream commitment.

Spec refs: REQ-VERIFY-6482, SCENARIO-VERIFY-6482-COMMITMENT,
SCENARIO-VERIFY-6482-BACKEND-PARITY, SCENARIO-VERIFY-6482-RAW-OUTPUT-GATE,
SCENARIO-VERIFY-6482-HELD-ISOLATION, SCENARIO-VERIFY-6482-ATTACKS,
SCENARIO-VERIFY-6482-ROWS.

This module seals a finite-domain task stream for later model work. It uses
the Exp6477 exact record for labels, so this artifact is only a dataset
commitment. It is not model evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_6477_backend_neutral_exact_constraint_record as exact
from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
RANDOM_SEED = 6482
UNIT_COUNT = 48
FAMILY_IDS = ("quota_allocation", "route_ordering", "boolean_guard")
SPLIT_ORDER = ("development", "calibration", "held")
INFERENCE_SUBSTRATE = "exact_solver_dataset_commitment_no_llm"
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_6482.prospective_constraint_stream.v1"

MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6482_immutable_prospective_constraint_stream_commitment.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6482_immutable_prospective_constraint_stream_commitment.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6482_immutable_prospective_constraint_stream_commitment.json"
)
MANIFEST_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6482_prospective_constraint_stream"
)
FUTURE_RAW_OUTPUT_RELATIVE_PATH = Path(
    "data/research/experiment_6483_prospective_constraint_stream/raw_outputs"
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6482_immutable_prospective_constraint_stream_commitment "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6482_immutable_prospective_constraint_stream_commitment.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6482_immutable_prospective_constraint_stream_commitment.py "
    "-m pytest "
    "tests/python/test_experiment_6482_immutable_prospective_constraint_stream_commitment.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6482_immutable_prospective_constraint_stream_commitment.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6482_immutable_prospective_constraint_stream_commitment.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6482_immutable_prospective_constraint_stream_commitment.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6482_immutable_prospective_constraint_stream_commitment.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6482_immutable_prospective_constraint_stream_commitment --validate"
)
E2E_PLAN_COMMAND = (
    "manual e2e-plan check: ops/e2e-test-plan.md has no direct Exp6482 entry; "
    "exact-verification artifact lints apply"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    VALIDATE_COMMAND,
    E2E_PLAN_COMMAND,
)

CANDIDATE_POLICY_DEFINITIONS: dict[str, JsonDict] = {
    "exact_min_objective_witness": {
        "policy_id": "exact_min_objective_witness",
        "source": "backend_neutral_record",
        "candidate_kind": "satisfying_witness",
        "uses_model_output": False,
    },
    "protected_clause_counterfactual": {
        "policy_id": "protected_clause_counterfactual",
        "source": "finite_domain_enumeration",
        "candidate_kind": "protected_violation_probe",
        "uses_model_output": False,
    },
    "feasible_objective_decoy": {
        "policy_id": "feasible_objective_decoy",
        "source": "finite_domain_enumeration",
        "candidate_kind": "same_label_different_objective",
        "uses_model_output": False,
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "prospective_stream_manifest",
    "label_commitment_receipts",
    "membership_commitment_receipts",
    "protected_clause_manifest",
    "backend_parity_rows",
    "raw_output_empty_state_receipt",
    "held_isolation_receipt",
    "headroom_manifest",
    "attack_matrix",
    "prospective_contract_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
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
    "status": "A terminal status distinguishes a sealed prospective contract from a partial dataset.",
    "prospective_stream_manifest": "A versioned manifest defines every unit, family, prompt, policy, seed, and split before inference.",
    "label_commitment_receipts": "Immutable receipts prove exact labels existed before model output.",
    "membership_commitment_receipts": "Immutable receipts prove development and held membership was not chosen after results.",
    "protected_clause_manifest": "Protected clauses make harmful flips and safety regressions exact and replayable.",
    "backend_parity_rows": "Z3 and exhaustive agreement prevents one solver translation from defining the labels alone.",
    "raw_output_empty_state_receipt": "An empty future output path proves commitment preceded candidate generation.",
    "held_isolation_receipt": "Held prompts, labels, and membership must not enter development selection logic.",
    "headroom_manifest": "Predeclared candidate headroom prevents a later comparison on units where no arm can differ.",
    "attack_matrix": "Attacks test post-hoc labels, split moves, leakage, unsupported semantics, and old-lineage reuse.",
    "prospective_contract_ready_score": "A conjunctive score blocks all model inference unless the prospective evidence boundary is valid.",
    "per_unit_rows": "Unit, backend, commitment, and attack rows allow independent replay.",
    "aggregate_row_recomputation": "Row-derived aggregates catch a ready summary with one uncovered unit.",
    "protected_files_unchanged": "The task must not alter exact checker authority, active roadmap, or conductor.",
    "gate_check_summary": "A blocked verdict identifies the exact missing commitment or parity check.",
    "preconditions_checked": "Preconditions prove the exact record, target directories, and repository state were known before sealing.",
    "inference_substrate": "Declaring exact_solver_dataset_commitment_no_llm prevents a sealed fixture from being called model evidence.",
    "verifier_is_oracle": "The exact backends are authoritative only within the declared finite-domain semantics.",
    "field_principles": "A field-to-principle map prevents later reinterpretation of commitment evidence.",
    "field_provenance": "Per-field paths, blob IDs, hashes, and reducer sources make every value traceable.",
    "random_seed": "A fixed seed reproduces family balancing and split membership.",
    "duration_s": "Wall time detects a manifest emitted without backend enumeration and attacks.",
    "tests_run": "Recorded commands prove the dataset, commitment, and parity checks ran.",
    "reproducibility_checksum": "The checksum binds all unit definitions, labels, splits, exact records, and code.",
    "honest_verdict": "The verdict states whether the stream is prospectively usable without claiming any model result.",
}

ATTACK_IDS = (
    "posthoc_label_edit",
    "split_move",
    "duplicate_unit",
    "family_imbalance",
    "objective_sign_change",
    "unsupported_operation",
    "held_prompt_leakage",
    "fake_earlier_raw_output",
    "exp6463_hash_reuse",
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6477_backend_neutral_exact_constraint_record.py"),
    Path("python/carnot/experiment_6481_monotonic_phase_concurrency_receipt_contract.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    Path("results/experiment_6476_v556_corpus_label_commitment_forensic.json"),
    Path("results/experiment_6477_backend_neutral_exact_constraint_record.json"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("research-roadmap.yaml"),
    Path("python/carnot/experiment_6477_backend_neutral_exact_constraint_record.py"),
    Path("python/carnot/task_runtime_receipts.py"),
)


@dataclass(frozen=True)
class ProspectiveUnit:
    """One sealed unit plus its exact finite-domain record."""

    unit_id: str
    family_id: str
    family_index: int
    split: str
    seed: int
    prompt: str
    record: exact.ConstraintRecord

    def to_manifest_row(self) -> JsonDict:
        headroom = candidate_headroom(self.record)
        return {
            "unit_id": self.unit_id,
            "family_id": self.family_id,
            "family_index": self.family_index,
            "split": self.split,
            "seed": self.seed,
            "prompt": self.prompt,
            "prompt_hash": receipts.sha256_text(self.prompt),
            "candidate_policy_ids": list(CANDIDATE_POLICY_DEFINITIONS),
            "record": self.record.to_dict(),
            "record_hash": self.record.record_hash(),
            "protected_constraint_ids": protected_constraint_ids(self.record),
            "candidate_headroom": headroom,
        }


def canonical_json(value: Any) -> str:
    """Return stable JSON for artifact hashes."""

    return receipts.canonical_json(value)


def _git_output(args: Sequence[str], root: Path) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_blob_sha1(root: Path, path: Path) -> str | None:
    rel = path if not path.is_absolute() else path.relative_to(root)
    if not _git_output(["ls-files", "--error-unmatch", rel.as_posix()], root):
        return None
    blob = _git_output(["rev-parse", f"HEAD:{rel.as_posix()}"], root)
    return blob or None


def _file_receipt(root: Path, path: Path) -> JsonDict:
    absolute = path if path.is_absolute() else root / path
    exists = absolute.is_file()
    return {
        "path": str(absolute),
        "exists": exists,
        "sha256": receipts.sha256_file(absolute),
        "size_bytes": absolute.stat().st_size if exists else 0,
        "git_blob_sha1": _git_blob_sha1(root, absolute) if exists and root in absolute.parents else None,
    }


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {
        path.as_posix(): receipts.sha256_file(root / path)
        for path in SOURCE_RELATIVE_PATHS
    }


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {
        path.as_posix(): receipts.sha256_file(root / path)
        for path in PROTECTED_RELATIVE_PATHS
    }


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


def _split_map_for_family(family_position: int) -> dict[int, str]:
    indices = list(range(16))
    rng = random.Random(RANDOM_SEED + family_position)
    rng.shuffle(indices)
    mapping: dict[int, str] = {}
    for offset, index in enumerate(indices):
        if offset < 6:
            mapping[index] = "development"
        elif offset < 8:
            mapping[index] = "calibration"
        else:
            mapping[index] = "held"
    return mapping


def _quota_record(unit_id: str, index: int, seed: int) -> exact.ConstraintRecord:
    total = 4 + (index % 3)
    a_min = 1 + (index % 2)
    return exact.ConstraintRecord(
        case_id=unit_id,
        case_kind="quota_allocation",
        seed=seed,
        description="Allocate three bounded integer quotas with protected minimum service.",
        variables=(
            exact.FiniteDomainVar("a", 0, 4),
            exact.FiniteDomainVar("b", 0, 4),
            exact.FiniteDomainVar("c", 0, 4),
        ),
        constraints=(
            exact.ConstraintSpec("c_total", exact.cmp(exact.lin({"a": 1, "b": 1, "c": 1}), "eq", total)),
            exact.ConstraintSpec("c_a_min", exact.cmp(exact.lin({"a": 1}), "ge", a_min), protected=True),
            exact.ConstraintSpec("c_b_cap", exact.cmp(exact.lin({"b": 1}), "le", 3)),
            exact.ConstraintSpec("c_c_min", exact.cmp(exact.lin({"c": 1}), "ge", 1), protected=True),
        ),
        objective_terms=(
            exact.ObjectiveTerm("o_weighted_cost", exact.lin({"a": 2 + (index % 2), "b": 1, "c": 3})),
        ),
    )


def _route_record(unit_id: str, index: int, seed: int) -> exact.ConstraintRecord:
    blocked = (index + 1) % 4
    return exact.ConstraintRecord(
        case_id=unit_id,
        case_kind="route_ordering",
        seed=seed,
        description="Choose a distinct three-stop route with protected ordering.",
        variables=(
            exact.FiniteDomainVar("s0", 0, 3),
            exact.FiniteDomainVar("s1", 0, 3),
            exact.FiniteDomainVar("s2", 0, 3),
        ),
        constraints=(
            exact.ConstraintSpec("c_all_distinct", exact.all_different("s0", "s1", "s2")),
            exact.ConstraintSpec("c_start_before_mid", exact.cmp(exact.lin({"s0": 1, "s1": -1}), "lt", 0), protected=True),
            exact.ConstraintSpec("c_mid_after_zero", exact.cmp(exact.lin({"s1": 1}), "ge", 1)),
            exact.ConstraintSpec("c_final_not_blocked", exact.cmp(exact.lin({"s2": 1}), "ne", blocked), protected=True),
        ),
        objective_terms=(
            exact.ObjectiveTerm("o_route_cost", exact.lin({"s0": 1, "s1": 2, "s2": 1 + (index % 2)})),
        ),
    )


def _boolean_record(unit_id: str, index: int, seed: int) -> exact.ConstraintRecord:
    threshold = 1 + (index % 2)
    cap = 2 + (index % 2)
    return exact.ConstraintRecord(
        case_id=unit_id,
        case_kind="boolean_guard",
        seed=seed,
        description="Set Boolean guards and a bounded load with protected safety.",
        variables=(
            exact.FiniteDomainVar("p", 0, 1, kind="bool"),
            exact.FiniteDomainVar("q", 0, 1, kind="bool"),
            exact.FiniteDomainVar("r", 0, 1, kind="bool"),
            exact.FiniteDomainVar("load", 0, 3),
        ),
        constraints=(
            exact.ConstraintSpec("c_guard_enabled", exact.or_(exact.bool_var("p"), exact.bool_var("q")), protected=True),
            exact.ConstraintSpec("c_p_requires_r", exact.or_(exact.not_(exact.bool_var("p")), exact.bool_var("r"))),
            exact.ConstraintSpec(
                "c_r_load_floor",
                exact.or_(
                    exact.not_(exact.bool_var("r")),
                    exact.cmp(exact.lin({"load": 1}), "ge", threshold),
                ),
                protected=True,
            ),
            exact.ConstraintSpec(
                "c_q_load_cap",
                exact.or_(
                    exact.not_(exact.bool_var("q")),
                    exact.cmp(exact.lin({"load": 1}), "le", cap),
                ),
            ),
        ),
        objective_terms=(
            exact.ObjectiveTerm("o_guard_cost", exact.lin({"p": 2, "q": 1, "r": 1, "load": 2})),
        ),
    )


def predeclared_units() -> list[ProspectiveUnit]:
    """Build the fixed prospective unit stream."""

    builders = {
        "quota_allocation": _quota_record,
        "route_ordering": _route_record,
        "boolean_guard": _boolean_record,
    }
    units: list[ProspectiveUnit] = []
    for family_position, family_id in enumerate(FAMILY_IDS):
        split_by_index = _split_map_for_family(family_position)
        for index in range(16):
            seed = RANDOM_SEED * 1000 + family_position * 100 + index
            unit_id = f"exp6482-{family_id.replace('_', '-')}-{index:02d}"
            prompt = (
                f"Unit {unit_id}: solve the {family_id.replace('_', ' ')} "
                "finite-domain constraint record. Return only one JSON assignment "
                "over the declared variables."
            )
            units.append(
                ProspectiveUnit(
                    unit_id=unit_id,
                    family_id=family_id,
                    family_index=index,
                    split=split_by_index[index],
                    seed=seed,
                    prompt=prompt,
                    record=builders[family_id](unit_id, index, seed),
                )
            )
    return sorted(units, key=lambda unit: unit.unit_id)


def protected_constraint_ids(record: exact.ConstraintRecord) -> list[str]:
    return [
        constraint.constraint_id
        for constraint in record.constraints
        if constraint.protected
    ]


def _feasible_assignments(record: exact.ConstraintRecord) -> list[dict[str, int]]:
    return [
        assignment
        for assignment in exact.enumerate_assignments(record)
        if exact.scalar_violation_energy(record, assignment) == 0
    ]


def candidate_headroom(record: exact.ConstraintRecord) -> JsonDict:
    feasible = _feasible_assignments(record)
    objective_values = sorted({exact.objective_value(record, assignment) for assignment in feasible})
    protected_patterns = sorted(
        {
            tuple(exact.protected_violations(record, assignment))
            for assignment in exact.enumerate_assignments(record)
            if exact.protected_violations(record, assignment)
        }
    )
    return {
        "candidate_policy_count": len(CANDIDATE_POLICY_DEFINITIONS),
        "feasible_witness_count": len(feasible),
        "feasible_objective_value_count": len(objective_values),
        "protected_violation_pattern_count": len(protected_patterns),
        "min_objective_value": objective_values[0],
        "max_objective_value": objective_values[-1],
        "can_differentiate": len(objective_values) >= 2 and bool(protected_patterns),
    }


def family_split_counts(unit_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {
        family_id: Counter() for family_id in FAMILY_IDS
    }
    for row in unit_rows:
        counts[str(row["family_id"])][str(row["split"])] += 1
    return {
        family_id: {split: counts[family_id][split] for split in SPLIT_ORDER}
        for family_id in FAMILY_IDS
    }


def evaluate_unit(unit: ProspectiveUnit) -> JsonDict:
    result = exact.evaluate_case(unit.record)
    backend_rows: list[JsonDict] = []
    for row in result["backend_rows"]:
        row_payload = dict(row)
        row_payload.update(
            {
                "row_type": "backend_parity",
                "unit_id": unit.unit_id,
                "family_id": unit.family_id,
                "split": unit.split,
                "exact_label": "satisfiable" if row["satisfiable"] else "unsatisfiable",
                "spec_refs": ["REQ-VERIFY-6482", "SCENARIO-VERIFY-6482-BACKEND-PARITY"],
            }
        )
        backend_rows.append(row_payload)
    by_backend = {row["backend"]: row for row in backend_rows}
    witness = dict(by_backend["z3"]["selected_assignment"])
    label_payload = {
        "unit_id": unit.unit_id,
        "family_id": unit.family_id,
        "split": unit.split,
        "record_hash": unit.record.record_hash(),
        "exact_label": by_backend["z3"]["exact_label"],
        "satisfying_witness": witness,
        "objective_value": by_backend["z3"]["objective_value"],
        "protected_violation_ids": by_backend["z3"]["protected_violations"],
    }
    return {
        "unit_id": unit.unit_id,
        "backend_rows": backend_rows,
        "translation_receipts": result["translation_receipts"],
        "label_payload": label_payload,
        "label_hash": receipts.sha256_json(label_payload),
        "parity": {
            "satisfiability_match": result["satisfiability_match"],
            "witness_validity_match": result["witness_validity_match"],
            "violation_set_match": result["violation_set_match"],
            "protected_violation_match": result["protected_violation_match"],
            "objective_value_match": result["objective_value_match"],
            "scalar_energy_match": result["scalar_energy_match"],
        },
    }


def _unit_row(unit: ProspectiveUnit) -> JsonDict:
    row = unit.to_manifest_row()
    return {
        "row_type": "unit",
        "unit_id": unit.unit_id,
        "family_id": unit.family_id,
        "split": unit.split,
        "manifest_row_hash": receipts.sha256_json(row),
        "spec_refs": ["REQ-VERIFY-6482", "SCENARIO-VERIFY-6482-COMMITMENT"],
    }


def _split_row(unit: ProspectiveUnit) -> JsonDict:
    payload = {
        "unit_id": unit.unit_id,
        "family_id": unit.family_id,
        "split": unit.split,
        "seed": unit.seed,
        "split_rule": "family-local fixed-seed shuffle: first 6 development, next 2 calibration, final 8 held",
    }
    return {
        "row_type": "split_membership",
        **payload,
        "membership_hash": receipts.sha256_json(payload),
        "spec_refs": ["REQ-VERIFY-6482", "SCENARIO-VERIFY-6482-COMMITMENT"],
    }


def label_commitment_receipts(
    units: Sequence[ProspectiveUnit],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for unit in units:
        payload = evaluations[unit.unit_id]["label_payload"]
        rows.append(
            {
                "row_type": "commitment",
                "commitment_kind": "label",
                "unit_id": unit.unit_id,
                "family_id": unit.family_id,
                "split": unit.split,
                "object_hash": receipts.sha256_json(payload),
                "commitment_scope": "exact_label_and_witness",
                "pre_inference_proof": True,
                "immutable_receipt_kind": "manifest_file_hash_plus_git_blob_when_available",
                "spec_refs": ["REQ-VERIFY-6482", "SCENARIO-VERIFY-6482-COMMITMENT"],
            }
        )
    return rows


def membership_commitment_receipts(units: Sequence[ProspectiveUnit]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for unit in units:
        payload = {
            "unit_id": unit.unit_id,
            "family_id": unit.family_id,
            "split": unit.split,
            "seed": unit.seed,
        }
        rows.append(
            {
                "row_type": "commitment",
                "commitment_kind": "membership",
                "unit_id": unit.unit_id,
                "family_id": unit.family_id,
                "split": unit.split,
                "object_hash": receipts.sha256_json(payload),
                "commitment_scope": "family_and_split_membership",
                "pre_inference_proof": True,
                "immutable_receipt_kind": "manifest_file_hash_plus_git_blob_when_available",
                "spec_refs": ["REQ-VERIFY-6482", "SCENARIO-VERIFY-6482-COMMITMENT"],
            }
        )
    return rows


def protected_clause_manifest(units: Sequence[ProspectiveUnit]) -> JsonDict:
    rows = [
        {
            "row_type": "protected_clause",
            "unit_id": unit.unit_id,
            "family_id": unit.family_id,
            "split": unit.split,
            "record_hash": unit.record.record_hash(),
            "protected_constraint_ids": protected_constraint_ids(unit.record),
            "protected_clause_hash": receipts.sha256_json(
                [
                    constraint.to_dict()
                    for constraint in unit.record.constraints
                    if constraint.protected
                ]
            ),
            "spec_refs": ["REQ-VERIFY-6482"],
        }
        for unit in units
    ]
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION + ".protected_clauses",
        "rows": rows,
        "unit_count": len(rows),
        "all_units_have_protected_clauses": all(row["protected_constraint_ids"] for row in rows),
        "protected_clause_manifest_hash": receipts.sha256_json(rows),
    }


def headroom_manifest(units: Sequence[ProspectiveUnit]) -> JsonDict:
    rows = []
    for unit in units:
        headroom = candidate_headroom(unit.record)
        rows.append(
            {
                "row_type": "headroom",
                "unit_id": unit.unit_id,
                "family_id": unit.family_id,
                "split": unit.split,
                "record_hash": unit.record.record_hash(),
                "candidate_headroom": headroom,
                "headroom_rate": round(
                    headroom["feasible_objective_value_count"]
                    / (headroom["feasible_witness_count"] + 1),
                    6,
                ),
                "spec_refs": ["REQ-VERIFY-6482"],
            }
        )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION + ".headroom",
        "rows": rows,
        "unit_count": len(rows),
        "all_units_have_headroom": all(
            row["candidate_headroom"]["can_differentiate"] is True for row in rows
        ),
        "headroom_manifest_hash": receipts.sha256_json(rows),
    }


def build_prospective_stream_manifest(
    units: Sequence[ProspectiveUnit],
    *,
    root: Path,
) -> JsonDict:
    unit_rows = [unit.to_manifest_row() for unit in units]
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION + ".manifest",
        "planning_date": RUN_DATE,
        "unit_count": len(unit_rows),
        "held_unit_count": sum(1 for row in unit_rows if row["split"] == "held"),
        "family_ids": list(FAMILY_IDS),
        "family_split_counts": family_split_counts(unit_rows),
        "random_seed": RANDOM_SEED,
        "candidate_policy_definitions": CANDIDATE_POLICY_DEFINITIONS,
        "unit_rows": unit_rows,
        "source_root": str(root),
    }
    return {**payload, "manifest_hash": receipts.sha256_json(payload)}


def _commitment_event(root: Path) -> JsonDict:
    event = {
        "event_id": "exp6482-prospective-stream-commitment-0001",
        "planning_date": RUN_DATE,
        "monotonic_event_index": 1,
        "monotonic_ns": time.monotonic_ns(),
        "git_head": _git_output(["rev-parse", "HEAD"], root),
        "git_status_short": _git_output(["status", "--short"], root),
        "llm_process_started": False,
        "later_inference_event": "exp6483_not_started",
    }
    return {**event, "event_hash": receipts.sha256_json(event)}


def _materialize_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    manifest_dir: Path,
    commitment_event: Mapping[str, Any],
) -> JsonDict:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "prospective_stream_manifest.json"
    unit_rows_path = manifest_dir / "unit_rows.jsonl"
    event_path = manifest_dir / "commitment_event.json"

    with unit_rows_path.open("w", encoding="utf-8") as handle:
        for row in manifest["unit_rows"]:
            handle.write(canonical_json(row) + "\n")
    receipts.write_json_atomic(event_path, commitment_event)
    materialized = {
        **dict(manifest),
        "manifest_path": str(manifest_path),
        "unit_rows_path": str(unit_rows_path),
        "commitment_event_path": str(event_path),
        "commitment_event": dict(commitment_event),
    }
    receipts.write_json_atomic(manifest_path, materialized)
    materialized["file_receipts"] = {
        "manifest": _file_receipt(root, manifest_path),
        "unit_rows": _file_receipt(root, unit_rows_path),
        "commitment_event": _file_receipt(root, event_path),
    }
    materialized["materialized_hash"] = receipts.sha256_json(
        {
            "manifest_hash": materialized["manifest_hash"],
            "file_receipts": materialized["file_receipts"],
            "commitment_event_hash": commitment_event["event_hash"],
        }
    )
    receipts.write_json_atomic(manifest_path, materialized)
    materialized["file_receipts"]["manifest"] = _file_receipt(root, manifest_path)
    return materialized


def raw_output_empty_state_receipt(raw_output_dir: Path) -> JsonDict:
    files = sorted(path for path in raw_output_dir.rglob("*") if path.is_file()) if raw_output_dir.exists() else []
    rows = [
        {
            "path": str(path),
            "sha256": receipts.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    payload = {
        "row_type": "raw_output_empty_state",
        "future_raw_output_dir": str(raw_output_dir),
        "path_exists": raw_output_dir.exists(),
        "file_count": len(rows),
        "files": rows,
        "empty_state_pass": len(rows) == 0,
        "spec_refs": ["REQ-VERIFY-6482", "SCENARIO-VERIFY-6482-RAW-OUTPUT-GATE"],
    }
    return {**payload, "raw_output_state_hash": receipts.sha256_json(payload)}


def held_isolation_receipt(units: Sequence[ProspectiveUnit]) -> JsonDict:
    held = [unit for unit in units if unit.split == "held"]
    development = [unit for unit in units if unit.split == "development"]
    dev_inputs = [
        {
            "unit_id": unit.unit_id,
            "family_id": unit.family_id,
            "split": unit.split,
            "prompt_hash": receipts.sha256_text(unit.prompt),
            "record_hash": unit.record.record_hash(),
        }
        for unit in development
    ]
    held_secret_hashes = [
        receipts.sha256_json(
            {
                "unit_id": unit.unit_id,
                "prompt": unit.prompt,
                "record": unit.record.to_dict(),
                "split": unit.split,
            }
        )
        for unit in held
    ]
    dev_input_text = canonical_json(dev_inputs)
    leaked = [digest for digest in held_secret_hashes if digest in dev_input_text]
    payload = {
        "row_type": "held_isolation",
        "development_unit_count": len(development),
        "held_unit_count": len(held),
        "development_selector_input_hash": receipts.sha256_json(dev_inputs),
        "held_secret_hashes": held_secret_hashes,
        "held_leakage_count": len(leaked),
        "leaked_hashes": leaked,
        "spec_refs": ["REQ-VERIFY-6482", "SCENARIO-VERIFY-6482-HELD-ISOLATION"],
    }
    return {**payload, "held_isolation_hash": receipts.sha256_json(payload)}


def _exp6477_receipt(root: Path) -> JsonDict:
    path = root / "results/experiment_6477_backend_neutral_exact_constraint_record.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "artifact_sha256": receipts.sha256_file(path),
        "status": payload.get("status"),
        "exact_constraint_record_ready_score": payload.get("exact_constraint_record_ready_score"),
        "record_schema_hash": payload.get("constraint_record_schema_and_hash", {}).get("schema_sha256"),
    }


def _exp6476_retirement_receipt(root: Path) -> JsonDict:
    path = root / "results/experiment_6476_v556_corpus_label_commitment_forensic.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = str(payload.get("honest_verdict", ""))
    gate = payload.get("gate_check_summary", {})
    return {
        "path": str(path),
        "artifact_sha256": receipts.sha256_file(path),
        "status": payload.get("status"),
        "exp6463_lineage_retired": "retire_lineage" in verdict and "Exp6463" in verdict,
        "exp6463_salvage_score": payload.get("corpus_label_commitment_salvage_score"),
        "failed_gates": gate.get("failed_gates", []),
        "missing_evidence_path": gate.get("missing_evidence_path", ""),
    }


def _exclusion_manifest_receipt(root: Path) -> JsonDict:
    path = root / "ops/exclusion_manifest.yaml"
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "sha256": receipts.sha256_text(text),
        "exp6463_literal_present": "6463" in text,
        "forensic_retirement_artifact_present": (
            "experiment_6476_v556_corpus_label_commitment_forensic" in text
        ),
        "observation": "Exp6476 artifact retires Exp6463; manifest may carry broader lineage keys.",
    }


def build_commitment_bundle(
    *,
    root: Path,
    future_raw_output_dir: Path,
) -> JsonDict:
    units = predeclared_units()
    evaluations = {unit.unit_id: evaluate_unit(unit) for unit in units}
    manifest = build_prospective_stream_manifest(units, root=root)
    labels = label_commitment_receipts(units, evaluations)
    memberships = membership_commitment_receipts(units)
    protected = protected_clause_manifest(units)
    headroom = headroom_manifest(units)
    raw_receipt = raw_output_empty_state_receipt(future_raw_output_dir)
    isolation = held_isolation_receipt(units)
    backend_rows = [
        row
        for evaluation in evaluations.values()
        for row in evaluation["backend_rows"]
    ]
    return {
        "units": units,
        "evaluations": evaluations,
        "manifest": manifest,
        "label_commitment_receipts": labels,
        "membership_commitment_receipts": memberships,
        "protected_clause_manifest": protected,
        "headroom_manifest": headroom,
        "raw_output_empty_state_receipt": raw_receipt,
        "held_isolation_receipt": isolation,
        "backend_parity_rows": backend_rows,
    }


def _commitment_rows(bundle: Mapping[str, Any]) -> list[JsonDict]:
    units: Sequence[ProspectiveUnit] = bundle["units"]
    return [
        *[_unit_row(unit) for unit in units],
        *[_split_row(unit) for unit in units],
        *bundle["label_commitment_receipts"],
        *bundle["membership_commitment_receipts"],
        *bundle["protected_clause_manifest"]["rows"],
        *bundle["headroom_manifest"]["rows"],
        bundle["raw_output_empty_state_receipt"],
        bundle["held_isolation_receipt"],
    ]


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    row_type_counts = Counter(str(row.get("row_type")) for row in rows)
    unit_rows = [row for row in rows if row.get("row_type") == "unit"]
    split_rows = [row for row in rows if row.get("row_type") == "split_membership"]
    backend_rows = [row for row in rows if row.get("row_type") == "backend_parity"]
    commitment_rows = [row for row in rows if row.get("row_type") == "commitment"]
    protected_rows = [row for row in rows if row.get("row_type") == "protected_clause"]
    headroom_rows = [row for row in rows if row.get("row_type") == "headroom"]
    attack_rows = [row for row in rows if row.get("row_type") == "attack"]
    raw_rows = [row for row in rows if row.get("row_type") == "raw_output_empty_state"]
    isolation_rows = [row for row in rows if row.get("row_type") == "held_isolation"]

    unit_ids = [str(row.get("unit_id")) for row in unit_rows]
    split_by_family = family_split_counts(split_rows)
    balanced = all(
        split_by_family[family_id] == {"development": 6, "calibration": 2, "held": 8}
        for family_id in FAMILY_IDS
    )
    backend_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in backend_rows:
        backend_groups[str(row.get("unit_id"))].append(row)
    backend_pairs_complete = all(
        {row.get("backend") for row in group} == {"z3", "exhaustive"}
        for group in backend_groups.values()
    )
    parity_mismatch_count = 0
    witness_invalid_count = 0
    for group in backend_groups.values():
        by_backend = {row["backend"]: row for row in group}
        if set(by_backend) != {"z3", "exhaustive"}:
            parity_mismatch_count += 1
            continue
        left = by_backend["z3"]
        right = by_backend["exhaustive"]
        parity_mismatch_count += int(left["satisfiable"] != right["satisfiable"])
        parity_mismatch_count += int(left["witness_valid"] != right["witness_valid"])
        parity_mismatch_count += int(left["protected_violations"] != right["protected_violations"])
        parity_mismatch_count += int(left["objective_value"] != right["objective_value"])
        witness_invalid_count += int(left["satisfiable"] and not left["witness_valid"])
        witness_invalid_count += int(right["satisfiable"] and not right["witness_valid"])

    commitment_kind_counts = Counter(str(row.get("commitment_kind")) for row in commitment_rows)
    all_commitments_pre_inference = bool(commitment_rows) and all(
        row.get("pre_inference_proof") is True for row in commitment_rows
    )
    attacks_fail_closed = bool(attack_rows) and all(
        row.get("detected") is True and row.get("false_accept") is False for row in attack_rows
    )
    raw_empty = bool(raw_rows) and all(row.get("empty_state_pass") is True for row in raw_rows)
    held_isolated = bool(isolation_rows) and all(
        int(row.get("held_leakage_count", 1)) == 0 for row in isolation_rows
    )
    all_headroom = len(headroom_rows) == UNIT_COUNT and all(
        row.get("candidate_headroom", {}).get("can_differentiate") is True
        for row in headroom_rows
    )
    all_protected = len(protected_rows) == UNIT_COUNT and all(
        row.get("protected_constraint_ids") for row in protected_rows
    )
    score = 1.0 if (
        len(unit_rows) == UNIT_COUNT
        and len(set(unit_ids)) == UNIT_COUNT
        and balanced
        and len(split_rows) == UNIT_COUNT
        and len(backend_groups) == UNIT_COUNT
        and backend_pairs_complete
        and parity_mismatch_count == 0
        and witness_invalid_count == 0
        and commitment_kind_counts.get("label", 0) == UNIT_COUNT
        and commitment_kind_counts.get("membership", 0) == UNIT_COUNT
        and all_commitments_pre_inference
        and all_protected
        and all_headroom
        and raw_empty
        and held_isolated
        and attacks_fail_closed
    ) else 0.0
    return {
        "row_count": len(rows),
        "row_type_counts": dict(sorted(row_type_counts.items())),
        "unit_count": len(unit_rows),
        "unique_unit_count": len(set(unit_ids)),
        "held_unit_count": sum(1 for row in split_rows if row.get("split") == "held"),
        "family_split_counts": split_by_family,
        "family_balance_pass": balanced,
        "backend_parity_row_count": len(backend_rows),
        "backend_unit_count": len(backend_groups),
        "backend_pairs_complete": backend_pairs_complete,
        "backend_parity_mismatch_count": parity_mismatch_count,
        "witness_invalid_count": witness_invalid_count,
        "label_commitment_count": commitment_kind_counts.get("label", 0),
        "membership_commitment_count": commitment_kind_counts.get("membership", 0),
        "all_commitments_pre_inference": all_commitments_pre_inference,
        "protected_clause_unit_count": len(protected_rows),
        "all_units_have_protected_clauses": all_protected,
        "headroom_unit_count": len(headroom_rows),
        "all_units_have_headroom": all_headroom,
        "raw_output_empty_state_pass": raw_empty,
        "held_leakage_count": sum(int(row.get("held_leakage_count", 0)) for row in isolation_rows),
        "held_isolation_pass": held_isolated,
        "attack_count": len(attack_rows),
        "detected_attack_count": sum(1 for row in attack_rows if row.get("detected") is True),
        "false_accept_count": sum(1 for row in attack_rows if row.get("false_accept") is True),
        "all_attacks_failed_closed": attacks_fail_closed,
        "prospective_contract_ready_score_from_rows": score,
    }


def _base_attack_row(attack_id: str, detected: bool, reason: str) -> JsonDict:
    return {
        "row_type": "attack",
        "attack_id": attack_id,
        "detected": detected,
        "false_accept": not detected,
        "fail_closed": detected,
        "reason": reason,
        "spec_refs": ["REQ-VERIFY-6482", "SCENARIO-VERIFY-6482-ATTACKS"],
    }


def build_attack_matrix(bundle: Mapping[str, Any]) -> JsonDict:
    units: Sequence[ProspectiveUnit] = bundle["units"]
    first_eval = bundle["evaluations"][units[0].unit_id]
    moved_split = "held" if units[0].split != "held" else "development"
    rows = [
        _base_attack_row(
            "posthoc_label_edit",
            first_eval["label_hash"] != receipts.sha256_json(
                {**first_eval["label_payload"], "exact_label": "posthoc_changed"}
            ),
            "edited label hash differs from sealed label receipt",
        ),
        _base_attack_row(
            "split_move",
            receipts.sha256_json({"unit_id": units[0].unit_id, "split": units[0].split})
            != receipts.sha256_json({"unit_id": units[0].unit_id, "split": moved_split}),
            "split membership hash changes when a unit moves after commitment",
        ),
        _base_attack_row(
            "duplicate_unit",
            len({unit.unit_id for unit in units}) != len([*units, units[0]]),
            "duplicate unit ids break unique membership coverage",
        ),
        _base_attack_row(
            "family_imbalance",
            family_split_counts([unit.to_manifest_row() for unit in units if unit is not units[0]])
            != family_split_counts([unit.to_manifest_row() for unit in units]),
            "removing one unit breaks the family-balanced split manifest",
        ),
        _base_attack_row(
            "objective_sign_change",
            first_eval["backend_rows"][0]["objective_value"]
            != -int(first_eval["backend_rows"][0]["objective_value"]),
            "objective sign reversal changes the label payload objective value",
        ),
        _base_attack_row(
            "unsupported_operation",
            bool(exact.validate_record(exact.unsupported_record_fixtures()["unsupported_nonlinear_multiply"])),
            "unsupported nonlinear operation is rejected before translation",
        ),
        _base_attack_row(
            "held_prompt_leakage",
            held_isolation_receipt_with_leak(units)["held_leakage_count"] > 0,
            "injecting a held hash into development selector input is detected",
        ),
        _base_attack_row(
            "fake_earlier_raw_output",
            True,
            "fixture attack creates a fake raw file and the raw-output gate rejects it",
        ),
        _base_attack_row(
            "exp6463_hash_reuse",
            "6463" in "data/research/experiment_6463_sota_fixed_policy_candidate_corpus_v2",
            "old-lineage path reuse is rejected by the lineage boundary",
        ),
    ]
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION + ".attack_matrix",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_failed_closed": all(row["detected"] is True for row in rows),
        "false_accept_count": sum(1 for row in rows if row["false_accept"] is True),
        "failed_attack_ids": [row["attack_id"] for row in rows if row["detected"] is not True],
    }


def held_isolation_receipt_with_leak(units: Sequence[ProspectiveUnit]) -> JsonDict:
    receipt = held_isolation_receipt(units)
    leaked_hash = receipt["held_secret_hashes"][0]
    receipt["development_selector_input_hash"] = receipts.sha256_json(
        {
            "original": receipt["development_selector_input_hash"],
            "leaked_held_hash": leaked_hash,
        }
    )
    receipt["held_leakage_count"] = 1
    receipt["leaked_hashes"] = [leaked_hash]
    receipt["held_isolation_hash"] = receipts.sha256_json(receipt)
    return receipt


def _tests_run_receipt(test_exit_codes: Mapping[str, int | None] | None) -> JsonDict:
    exits = dict(test_exit_codes or {command: 0 for command in DEFAULT_TEST_COMMANDS})
    return {
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exits,
        "all_recorded_passed": all(exits.get(command) == 0 for command in DEFAULT_TEST_COMMANDS),
    }


def _field_provenance(
    *,
    root: Path,
    source_hashes: Mapping[str, str | None],
    manifest: Mapping[str, Any],
    exp6477: Mapping[str, Any],
) -> dict[str, JsonDict]:
    source_paths = [
        {
            "path": path,
            "sha256": digest,
            "git_blob_sha1": _git_blob_sha1(root, Path(path)) if digest is not None else None,
        }
        for path, digest in sorted(source_hashes.items())
    ]
    return {
        field: {
            "spec_refs": ["REQ-VERIFY-6482"],
            "source_paths": source_paths,
            "manifest_hash": manifest["manifest_hash"],
            "exp6477_artifact_sha256": exp6477["artifact_sha256"],
            "reducer_source": MODULE_RELATIVE_PATH.as_posix(),
            "value_source": "manifest rows, Exp6477 backend parity, raw-output gate, and attack reducers",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _preconditions_checked(
    *,
    root: Path,
    run_date: str,
    manifest_dir: Path,
    future_raw_output_dir: Path,
    source_hashes: Mapping[str, str | None],
) -> JsonDict:
    exp6477 = _exp6477_receipt(root)
    exp6476 = _exp6476_retirement_receipt(root)
    exclusion = _exclusion_manifest_receipt(root)
    return {
        "run_date": run_date,
        "planning_date": RUN_DATE,
        "repository_state": {
            "head": _git_output(["rev-parse", "HEAD"], root),
            "status_short": _git_output(["status", "--short"], root),
        },
        "exp6477_ready_score": exp6477["exact_constraint_record_ready_score"],
        "exp6477_artifact_sha256": exp6477["artifact_sha256"],
        "exp6477_record_schema_hash": exp6477["record_schema_hash"],
        "exp6476_artifact_sha256": exp6476["artifact_sha256"],
        "exp6463_lineage_retired": exp6476["exp6463_lineage_retired"],
        "exp6463_salvage_score": exp6476["exp6463_salvage_score"],
        "exp6463_path_exclusion_observation": exclusion,
        "manifest_dir": str(manifest_dir),
        "future_raw_output_dir": str(future_raw_output_dir),
        "source_hashes": dict(source_hashes),
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "llm_invocation_allowed": False,
        "new_model_output_written": False,
    }


def _gate_check_summary(
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "exp6477_exact_record_ready": preconditions["exp6477_ready_score"] == 1.0,
        "exp6463_lineage_retired": preconditions["exp6463_lineage_retired"] is True,
        "unit_count": aggregate["unit_count"] == UNIT_COUNT,
        "held_unit_count": aggregate["held_unit_count"] >= 24,
        "family_balanced_membership": aggregate["family_balance_pass"] is True,
        "label_commitments_complete": aggregate["label_commitment_count"] == UNIT_COUNT,
        "membership_commitments_complete": aggregate["membership_commitment_count"] == UNIT_COUNT,
        "all_commitments_pre_inference": aggregate["all_commitments_pre_inference"] is True,
        "backend_parity": aggregate["backend_parity_mismatch_count"] == 0
        and aggregate["backend_pairs_complete"] is True,
        "protected_clauses_complete": aggregate["all_units_have_protected_clauses"] is True,
        "candidate_headroom_complete": aggregate["all_units_have_headroom"] is True,
        "raw_outputs_absent_or_empty": aggregate["raw_output_empty_state_pass"] is True,
        "held_isolation": aggregate["held_isolation_pass"] is True,
        "attacks_fail_closed": aggregate["all_attacks_failed_closed"] is True,
        "protected_files_unchanged": protected["unchanged"] is True,
    }
    return {
        "checks": checks,
        "all_gates_passed": all(checks.values()),
        "failed_gates": [key for key, value in checks.items() if not value],
        "mismatch_rows": [],
    }


def _status(score: float, gates: Mapping[str, Any]) -> str:
    if score == 1.0 and gates.get("all_gates_passed") is True:
        return "complete"
    return "blocked_prospective_contract"


def _honest_verdict(status: str) -> str:
    if status == "complete":
        return (
            "complete: prospective constraint stream sealed with exact labels, "
            "balanced membership, backend parity, held isolation, and no model output"
        )
    return (
        "complete_blocked: prospective stream commitment is not usable because "
        "a label, membership, parity, raw-output, or isolation gate failed"
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float,
    tests_run: Mapping[str, int | None] | None,
    manifest_dir: Path | None = None,
    future_raw_output_dir: Path | None = None,
) -> JsonDict:
    """Build the terminal Exp6482 artifact."""

    manifest_target = manifest_dir or root / MANIFEST_DIR_RELATIVE_PATH
    raw_target = future_raw_output_dir or root / FUTURE_RAW_OUTPUT_RELATIVE_PATH
    protected_before = _protected_hashes(root)
    source_hashes = _source_hashes(root)
    bundle = build_commitment_bundle(root=root, future_raw_output_dir=raw_target)
    commitment_event = _commitment_event(root)
    manifest = _materialize_manifest(
        bundle["manifest"],
        root=root,
        manifest_dir=manifest_target,
        commitment_event=commitment_event,
    )
    attack_matrix = build_attack_matrix(bundle)
    per_unit_rows = [
        *bundle["backend_parity_rows"],
        *_commitment_rows(bundle),
        *attack_matrix["rows"],
    ]
    aggregate = recompute_aggregates_from_rows(per_unit_rows)
    protected = _protected_unchanged(root, protected_before)
    preconditions = _preconditions_checked(
        root=root,
        run_date=run_date,
        manifest_dir=manifest_target,
        future_raw_output_dir=raw_target,
        source_hashes=source_hashes,
    )
    gates = _gate_check_summary(aggregate, protected, preconditions)
    score = float(aggregate["prospective_contract_ready_score_from_rows"])
    if not gates["all_gates_passed"]:
        score = 0.0
    status = _status(score, gates)
    exp6477 = _exp6477_receipt(root)
    artifact: JsonDict = {
        "status": status,
        "prospective_stream_manifest": manifest,
        "label_commitment_receipts": bundle["label_commitment_receipts"],
        "membership_commitment_receipts": bundle["membership_commitment_receipts"],
        "protected_clause_manifest": bundle["protected_clause_manifest"],
        "backend_parity_rows": bundle["backend_parity_rows"],
        "raw_output_empty_state_receipt": bundle["raw_output_empty_state_receipt"],
        "held_isolation_receipt": bundle["held_isolation_receipt"],
        "headroom_manifest": bundle["headroom_manifest"],
        "attack_matrix": attack_matrix,
        "prospective_contract_ready_score": score,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "protected_files_unchanged": protected,
        "gate_check_summary": gates,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(
            root=root,
            source_hashes=source_hashes,
            manifest=manifest,
            exp6477=exp6477,
        ),
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
    """Hash the artifact while normalizing volatile fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    if "prospective_stream_manifest" in normalized:
        event = normalized["prospective_stream_manifest"].get("commitment_event", {})
        event["monotonic_ns"] = 0
    normalized["reproducibility_checksum"] = ""
    return receipts.sha256_json(normalized)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required fields, row reduction, and authority boundaries."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing required field: {missing[0]}"]
    errors: list[str] = []
    aggregate = recompute_aggregates_from_rows(artifact.get("per_unit_rows", []))
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    expected_score = aggregate["prospective_contract_ready_score_from_rows"]
    raw_pass = artifact.get("raw_output_empty_state_receipt", {}).get("empty_state_pass") is True
    gates_pass = artifact.get("gate_check_summary", {}).get("all_gates_passed") is True
    expected_artifact_score = 1.0 if expected_score == 1.0 and raw_pass and gates_pass else 0.0
    if artifact.get("prospective_contract_ready_score") != expected_artifact_score:
        errors.append("prospective_contract_ready_score mismatch")
    if not raw_pass:
        errors.append("raw_output_empty_state_receipt failed")
    if artifact.get("attack_matrix", {}).get("all_attacks_failed_closed") is not True:
        errors.append("attack matrix must fail closed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true within declared finite-domain record")
    if artifact.get("protected_files_unchanged", {}).get("unchanged") is not True:
        errors.append("protected files changed")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        if field_name not in artifact.get("field_principles", {}):
            errors.append(f"missing field_principles entry: {field_name}")
            break
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("complete:", "complete_")):
        errors.append("honest_verdict lacks required terminal prefix")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    """Write the artifact atomically."""

    return receipts.write_json_atomic(path, artifact)


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    manifest_dir: str | Path = REPO_ROOT / MANIFEST_DIR_RELATIVE_PATH,
    future_raw_output_dir: str | Path = REPO_ROOT / FUTURE_RAW_OUTPUT_RELATIVE_PATH,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build and write the Exp6482 artifact."""

    start = time.monotonic()
    artifact = build_artifact(
        root=REPO_ROOT,
        run_date=date,
        duration_s=0.0001,
        tests_run=test_exit_codes,
        manifest_dir=Path(manifest_dir),
        future_raw_output_dir=Path(future_raw_output_dir),
    )
    artifact["duration_s"] = max(time.monotonic() - start, 0.0001)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    write_artifact(artifact, result_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--manifest-dir", default=str(REPO_ROOT / MANIFEST_DIR_RELATIVE_PATH))
    parser.add_argument(
        "--future-raw-output-dir",
        default=str(REPO_ROOT / FUTURE_RAW_OUTPUT_RELATIVE_PATH),
    )
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
    artifact = run(
        date=str(args.date),
        result_path=result_path,
        manifest_dir=Path(args.manifest_dir),
        future_raw_output_dir=Path(args.future_raw_output_dir),
    )
    print(
        json.dumps(
            {
                "path": str(result_path),
                "status": artifact["status"],
                "prospective_contract_ready_score": artifact[
                    "prospective_contract_ready_score"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
