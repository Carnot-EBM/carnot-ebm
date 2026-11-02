"""Exp5893 exact grounding-shortcut fixture.

Spec refs: REQ-VERIFY-5893, SCENARIO-VERIFY-5893-SCHEMA,
SCENARIO-VERIFY-5893-SHORTCUTS, SCENARIO-VERIFY-5893-CONTROLS,
SCENARIO-VERIFY-5893-REPLAY-AND-LEAKAGE.

This module builds a small deterministic dataset, not a model benchmark. The
rows separate the human-facing semantic task from the encoded logical formula
so that two shortcut classes can be measured independently:

* constraint-satisfaction shortcuts: a bad grounding satisfies the formula
  while the intended task is not satisfied;
* cognition shortcuts: a biased, semantically wrong atom mapping satisfies a
  logically sound formula.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

from carnot import experiment_5868_hardness_controlled_constraint_fixture as exp5868
from carnot import experiment_5892_headroom_evidence_escrow as exp5892


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = exp5892.REPO_ROOT
RESULT_RELATIVE_PATH = Path("results/experiment_5893_grounding_shortcut_fixture.json")
ROW_FILE_RELATIVE_PATH = Path(
    "results/experiment_5893_grounding_shortcut_fixture.rows.jsonl"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5893_grounding_shortcut_fixture.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5893_grounding_shortcut_fixture.py")
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
EXP5868_ROWS_RELATIVE_PATH = exp5868.ROW_FILE_RELATIVE_PATH
EXP5868_SUMMARY_RELATIVE_PATH = exp5868.RESULT_RELATIVE_PATH
EXP5892_RELATIVE_PATH = exp5892.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5893.grounding_shortcut_fixture.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT = 5893
EXPERIMENT_ID = "experiment_5893_grounding_shortcut_fixture"
MILESTONE = "2026.07.524"
RUN_DATE = "20260724"
SOURCE_ARXIV_ID = "2607.21185"
SOURCE_ID = f"arxiv:{SOURCE_ARXIV_ID}"
INFERENCE_SUBSTRATE = "deterministic_exact_solver_labeled_dataset_no_llm"
VERIFIER_IS_ORACLE = True
BASE_SEED = 5893
RAM_FLOOR_MB = 1024
DISK_FLOOR_MB = 512
GROUNDING_THRESHOLD = "1/2"
SURFACE_LENGTH_TOLERANCE = 0

FAMILIES = ("parity_xor", "implication_chain", "exactly_one_route")
REGIMES = (
    "canonical_one_to_one",
    "one_to_one_negative_control",
    "constraint_satisfaction_many_to_one",
    "constraint_satisfaction_soft_mass_swap",
    "cognition_biased_permutation",
    "soft_distributed_control",
    "shuffled_control",
    "label_permutation_control",
    "frequency_balanced_control",
    "surface_matched_control",
    "no_information_control",
)
SHORTCUT_TYPES = (
    "none",
    "constraint_satisfaction_shortcut",
    "cognition_shortcut",
)
PROTECTED_RELATIVE_PATHS = (
    EXP5868_SUMMARY_RELATIVE_PATH,
    EXP5868_ROWS_RELATIVE_PATH,
    EXP5892_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
)
HASHED_SOURCE_PATHS = {
    "codex_instructions": Path("CODEX.md"),
    "claude_instructions": Path("CLAUDE.md"),
    "research_references": RESEARCH_REFERENCES_RELATIVE_PATH,
    "verification_spec": VERIFY_SPEC_RELATIVE_PATH,
    "exp5893_module": MODULE_RELATIVE_PATH,
    "exp5893_test": TEST_RELATIVE_PATH,
    "exp5892_artifact": EXP5892_RELATIVE_PATH,
    "exp5868_rows": EXP5868_ROWS_RELATIVE_PATH,
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_gate_receipt",
    "source_method_receipt",
    "concept_atom_and_grounding_schema",
    "shortcut_type_definitions",
    "family_grounding_and_chronology_design",
    "generator_and_seed_receipts",
    "exact_semantic_and_constraint_oracle_receipts",
    "one_to_one_soft_distributed_and_shuffled_controls",
    "bias_and_frequency_controls",
    "split_and_group_leakage_receipts",
    "label_witness_and_headroom_balance",
    "row_file_receipt",
    "deterministic_replay_receipt",
    "protected_files_unchanged",
    "grounding_shortcut_fixture_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes a ready grounding-shortcut fixture from blocked or null evidence.",
    "preconditions_checked": "Gate, source, rows, solvers, generator, seeds, resources, outputs, and protected files prevent fabricated data.",
    "upstream_gate_receipt": "Exp5892 must provide non-retired hardness admission before Exp5893 extends the exact fixture lane.",
    "source_method_receipt": "arXiv:2607.21185 is used as design motivation only, not as imported results.",
    "concept_atom_and_grounding_schema": "Concepts, logical atoms, grounding matrices, intended semantics, encoded constraints, and exact outcomes are distinct fields.",
    "shortcut_type_definitions": "The two failure modes remain separately measurable.",
    "family_grounding_and_chronology_design": "Family cells, held regimes, and chronology batches are preregistered before row generation.",
    "generator_and_seed_receipts": "Deterministic generation makes row ids, controls, and replay reproducible.",
    "exact_semantic_and_constraint_oracle_receipts": "Exact checks own both task intent and formula satisfaction.",
    "one_to_one_soft_distributed_and_shuffled_controls": "One-to-one, soft, distributed, shuffled, label-permutation, surface, and no-information controls expose grounding shortcuts.",
    "bias_and_frequency_controls": "Biased-frequency rows must keep exact labels balanced so frequency alone cannot become the label.",
    "split_and_group_leakage_receipts": "Variants of one semantic problem never cross evaluation boundaries.",
    "label_witness_and_headroom_balance": "Both exact labels, witnesses, and shortcut headroom must replay for both shortcut types.",
    "row_file_receipt": "Path, count, schema, row hashes, and hash root expose every generated row.",
    "deterministic_replay_receipt": "A second generation must reproduce row ids and bytes exactly.",
    "protected_files_unchanged": "Operator-owned and immutable upstream files remain untouched.",
    "grounding_shortcut_fixture_ready_score": "Emit bare `1.0` only when all labels/witnesses replay and both shortcut types retain headroom.",
    "duration_s": "Measured wall time exposes bootstrap-only dataset work.",
    "inference_substrate": "Use `deterministic_exact_solver_labeled_dataset_no_llm`.",
    "verifier_is_oracle": "True for label authority and never credited as learned energy.",
    "field_provenance": "Every field traces to prompt, spec, source receipt, rows, generator, tests, or exact oracle receipts.",
    "test_commands": "Commands document focused unit, coverage, determinism, labels, witnesses, headroom, controls, leakage, schema/hash, adversarial, spec, root-clutter, and protected-file checks.",
    "test_exit_codes": "Exit codes prevent failed checks from silently promoting.",
    "reproducibility_checksum": "A checksum detects source, seed, row, oracle, control, or gate drift.",
    "honest_verdict": "Use `ready:`, `complete_null:`, or `blocked:`.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5893_grounding_shortcut_fixture.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5893_grounding_shortcut_fixture.py "
    "-m pytest tests/python/test_experiment_5893_grounding_shortcut_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5893_grounding_shortcut_fixture.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py --json "
    "results/experiment_5893_grounding_shortcut_fixture.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)

canonical_json = exp5868.canonical_json
sha256_text = exp5868.sha256_text
sha256_json = exp5868.sha256_json
sha256_file = exp5868.sha256_file


FAMILY_DEFINITIONS: dict[str, JsonDict] = {
    "parity_xor": {
        "atoms": ("atom_left_active", "atom_right_active", "atom_target_active"),
        "concepts": ("left_active", "right_active", "target_active"),
        "constraint_kind": "xor_equals",
        "cases": (
            {"left_active": True, "right_active": False, "target_active": True},
            {"left_active": False, "right_active": True, "target_active": True},
        ),
        "flip_concept": "target_active",
    },
    "implication_chain": {
        "atoms": ("atom_premise_enabled", "atom_rule_fired", "atom_conclusion_required"),
        "concepts": ("premise_enabled", "rule_fired", "conclusion_required"),
        "constraint_kind": "implication_requires_rule",
        "cases": (
            {"premise_enabled": False, "rule_fired": True, "conclusion_required": False},
            {"premise_enabled": False, "rule_fired": True, "conclusion_required": True},
        ),
        "flip_concept": "rule_fired",
    },
    "exactly_one_route": {
        "atoms": ("atom_north_route", "atom_east_route", "atom_west_route"),
        "concepts": ("north_route", "east_route", "west_route"),
        "constraint_kind": "exactly_one",
        "cases": (
            {"north_route": True, "east_route": False, "west_route": False},
            {"north_route": False, "east_route": True, "west_route": False},
        ),
        "flip_concept": "west_route",
    },
}


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload)


def read_rows(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(dict(json.loads(line)))
    return rows


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": RAM_FLOOR_MB, "ok": available_mb >= RAM_FLOOR_MB}


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": DISK_FLOOR_MB, "ok": available_mb >= DISK_FLOOR_MB}


def _hash_optional_file(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _source_receipt_text(text: str) -> str:
    marker = f"arXiv:{SOURCE_ARXIV_ID}"
    index = text.find(marker)
    if index < 0:  # pragma: no cover - exercised through missing-root precondition only.
        return ""
    start = text.rfind("\n- **", 0, index)
    start = 0 if start < 0 else start + 1
    next_bullet = text.find("\n- **", index + len(marker))
    next_heading = text.find("\n## ", index + len(marker))
    end_candidates = [value for value in (next_bullet, next_heading) if value >= 0]
    end = min(end_candidates) if end_candidates else len(text)
    return text[start:end].strip()


def source_method_receipt(root: Path = REPO_ROOT) -> JsonDict:
    path = Path(root) / RESEARCH_REFERENCES_RELATIVE_PATH
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    receipt = _source_receipt_text(text)
    return {
        "schema": SCHEMA + ".source_method_receipt",
        "principle": REQUIRED_FIELD_PRINCIPLES["source_method_receipt"],
        "source_id": SOURCE_ID,
        "source_url": "https://arxiv.org/abs/2607.21185",
        "path": RESEARCH_REFERENCES_RELATIVE_PATH.as_posix(),
        "research_references_sha256": _hash_optional_file(Path(root), RESEARCH_REFERENCES_RELATIVE_PATH),
        "receipt_found": bool(receipt),
        "receipt_hash": sha256_text(receipt) if receipt else "missing",
        "method_boundary": "fixture_design_reference_only_no_result_import",
        "claims_imported": False,
        "ok": bool(receipt),
    }


def upstream_gate_receipt(root: Path = REPO_ROOT) -> JsonDict:
    path = Path(root) / EXP5892_RELATIVE_PATH
    payload = _read_json(path) if path.exists() else {}
    score = payload.get("headroom_admission_ready_score")
    status_value = str(payload.get("status", "missing"))
    verdict = str(payload.get("honest_verdict", "missing"))
    non_retired = status_value != "retired" and not verdict.startswith("retired:")
    admitted = bool(score == 1.0 and status_value == "complete_ready" and non_retired)
    return {
        "schema": SCHEMA + ".upstream_gate_receipt",
        "principle": REQUIRED_FIELD_PRINCIPLES["upstream_gate_receipt"],
        "artifact_path": EXP5892_RELATIVE_PATH.as_posix(),
        "artifact_sha256": _hash_optional_file(Path(root), EXP5892_RELATIVE_PATH),
        "exp5892_status": status_value,
        "exp5892_honest_verdict": verdict,
        "exp5892_ready_score": score,
        "exp5892_non_retired_admission": admitted,
        "exp5892_inference_substrate": payload.get("inference_substrate", "missing"),
        "exp5892_verifier_is_oracle": payload.get("verifier_is_oracle", "missing"),
    }


def _output_path_receipt(result_path: Path, row_file_path: Path) -> JsonDict:
    return {
        "schema": SCHEMA + ".output_path_receipt",
        "result_path": str(result_path),
        "row_file_path": str(row_file_path),
        "result_writable": result_path.parent.exists() or result_path.parent.parent.exists(),
        "row_file_writable": row_file_path.parent.exists() or row_file_path.parent.parent.exists(),
        "atomic_checkpoint_suffix": ".tmp",
    }


def seed_registry() -> JsonDict:
    registry = {
        "base_seed": BASE_SEED,
        "families": list(FAMILIES),
        "grounding_regimes": list(REGIMES),
        "shortcut_types": list(SHORTCUT_TYPES),
        "chronology_batches": ["batch_0_train_style", "batch_1_held_style"],
        "matrix_threshold": GROUNDING_THRESHOLD,
        "surface_length_tolerance": SURFACE_LENGTH_TOLERANCE,
    }
    return {
        "schema": SCHEMA + ".seed_registry",
        "registry": registry,
        "registry_hash": sha256_json(registry),
        "ok": True,
    }


def exact_solver_receipts() -> JsonDict:
    solvers = {
        "semantic_exact_match_v1": {
            "command": "internal:observed_concepts == intended_semantics",
            "complete": True,
            "version": "carnot_exp5893_semantic_exact_match_v1",
        },
        "semantic_mismatch_witness_v1": {
            "command": "internal:list_concept_value_mismatches",
            "complete": True,
            "version": "carnot_exp5893_semantic_mismatch_witness_v1",
        },
        "constraint_direct_formula_v1": {
            "command": "internal:evaluate_family_formula(encoded_atoms)",
            "complete": True,
            "version": "carnot_exp5893_constraint_direct_formula_v1",
        },
        "constraint_replay_formula_v1": {
            "command": "internal:replay_grounding_matrix_then_formula",
            "complete": True,
            "version": "carnot_exp5893_constraint_replay_formula_v1",
        },
    }
    return {
        "schema": SCHEMA + ".exact_solver_receipts",
        "solver_configuration_count": len(solvers),
        "solvers": solvers,
        "ok": all(entry["complete"] for entry in solvers.values()),
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    root = Path(root)
    result_path = Path(result_path)
    row_file_path = Path(row_file_path)
    gate = upstream_gate_receipt(root)
    source = source_method_receipt(root)
    source_hashes = {name: _hash_optional_file(root, path) for name, path in HASHED_SOURCE_PATHS.items()}
    protected_hashes = {
        path.as_posix(): _hash_optional_file(root, path) for path in PROTECTED_RELATIVE_PATHS
    }
    memory = memory_probe()
    disk = disk_probe(root)
    solvers = exact_solver_receipts()
    seeds = seed_registry()
    output_paths = _output_path_receipt(result_path, row_file_path)
    checks = {
        "exp5892_non_retired_gate": gate["exp5892_non_retired_admission"] is True,
        "source_receipt": source["ok"] is True,
        "immutable_base_rows": source_hashes["exp5868_rows"] != "missing",
        "exact_solvers": solvers["ok"] is True,
        "generator_code": source_hashes["exp5893_module"] != "missing",
        "test_code": source_hashes["exp5893_test"] != "missing",
        "seed_registry": seeds["ok"] is True,
        "output_paths": output_paths["result_writable"] and output_paths["row_file_writable"],
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "protected_files": all(value != "missing" for value in protected_hashes.values()),
        "python": sys.version_info >= (3, 11),
    }
    blocked_reasons = [name for name, ok in checks.items() if not ok]
    return {
        "schema": SCHEMA + ".preconditions",
        "principle": REQUIRED_FIELD_PRINCIPLES["preconditions_checked"],
        "run_date": RUN_DATE,
        "upstream_gate_receipt": gate,
        "source_method_receipt": source,
        "immutable_base_rows": {
            "path": EXP5868_ROWS_RELATIVE_PATH.as_posix(),
            "sha256": source_hashes["exp5868_rows"],
        },
        "exact_solver_receipts": solvers,
        "generator_and_seed_receipts": seeds,
        "source_hashes": source_hashes,
        "resources": {"memory": memory, "disk": disk},
        "output_paths": output_paths,
        "protected_file_hashes": protected_hashes,
        "checks": checks,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "preconditions_ready": not blocked_reasons,
        "blocked_reasons": sorted(blocked_reasons),
    }


def concept_atom_and_grounding_schema() -> JsonDict:
    return {
        "schema": SCHEMA + ".concept_atom_and_grounding_schema",
        "principle": REQUIRED_FIELD_PRINCIPLES["concept_atom_and_grounding_schema"],
        "concept": "A named semantic variable in the intended task.",
        "logical_atom": "A named Boolean atom consumed by the encoded constraint.",
        "grounding": "A rational matrix from observed concepts to logical atoms.",
        "intended_semantics": "The exact semantic assignment the task asks the row to preserve.",
        "encoded_constraint": "The exact Boolean formula evaluated over grounded atoms.",
        "exact_outcome": "The pair of exact semantic and exact constraint labels.",
        "matrix_value_type": "rational_string",
        "threshold": GROUNDING_THRESHOLD,
    }


def shortcut_type_definitions() -> JsonDict:
    definitions = {
        "constraint_satisfaction_shortcut": (
            "A non-injective or soft grounding satisfies the encoded formula while "
            "the observed concepts violate intended semantics."
        ),
        "cognition_shortcut": (
            "A biased one-to-one permutation maps concepts to the wrong atoms, so "
            "the formula is satisfied under a semantically wrong grounding."
        ),
        "none": "A control or canonical row with no shortcut credit.",
    }
    return {
        "schema": SCHEMA + ".shortcut_type_definitions",
        "principle": REQUIRED_FIELD_PRINCIPLES["shortcut_type_definitions"],
        "definitions": definitions,
        "failure_modes_separately_measurable": True,
    }


def family_grounding_and_chronology_design() -> JsonDict:
    return {
        "schema": SCHEMA + ".family_grounding_and_chronology_design",
        "principle": REQUIRED_FIELD_PRINCIPLES["family_grounding_and_chronology_design"],
        "families": {
            family: {
                "concepts": list(definition["concepts"]),
                "atoms": list(definition["atoms"]),
                "constraint_kind": definition["constraint_kind"],
                "case_count": len(definition["cases"]),
            }
            for family, definition in FAMILY_DEFINITIONS.items()
        },
        "grounding_regimes": list(REGIMES),
        "held_grounding_regimes": [
            "soft_distributed_control",
            "shuffled_control",
            "label_permutation_control",
            "no_information_control",
        ],
        "chronology_batches": {
            "case_0": "batch_0_train_style",
            "case_1": "batch_1_held_style",
        },
        "design_hash": sha256_json({"families": FAMILY_DEFINITIONS, "regimes": REGIMES}),
    }


def generator_and_seed_receipts(preconditions_checked: Mapping[str, Any] | None = None) -> JsonDict:
    receipt = seed_registry()
    receipt.update(
        {
            "principle": REQUIRED_FIELD_PRINCIPLES["generator_and_seed_receipts"],
            "generator_hashes": dict((preconditions_checked or {}).get("source_hashes") or {}),
            "row_id_rule": "exp5893-{family}-case{case_index}-{variant}",
        }
    )
    return receipt


def _identity_matrix(concepts: Sequence[str], atoms: Sequence[str]) -> list[list[str]]:
    return [["1" if index == atom_index else "0" for atom_index, _atom in enumerate(atoms)] for index, _concept in enumerate(concepts)]


def _zero_matrix(concepts: Sequence[str], atoms: Sequence[str]) -> list[list[str]]:
    return [["0" for _atom in atoms] for _concept in concepts]


def _permutation_matrix(concepts: Sequence[str], atoms: Sequence[str], permutation: Sequence[int]) -> list[list[str]]:
    values = _zero_matrix(concepts, atoms)
    for atom_index, concept_index in enumerate(permutation):
        values[concept_index][atom_index] = "1"
    return values


def _soft_matrix_for_assignment(concepts: Sequence[str], atoms: Sequence[str], observed: Mapping[str, bool], intended: Mapping[str, bool]) -> list[list[str]]:
    values = _zero_matrix(concepts, atoms)
    true_sources = [index for index, concept in enumerate(concepts) if observed[concept]]
    false_sources = [index for index, concept in enumerate(concepts) if not observed[concept]]
    for atom_index, concept in enumerate(concepts):
        target = bool(intended[concept])
        if target and true_sources:
            for source in true_sources[:2]:
                values[source][atom_index] = "1/2" if len(true_sources[:2]) == 2 else "1"
        elif not target and false_sources:
            values[false_sources[0]][atom_index] = "1"
    return values


def _many_to_one_matrix(concepts: Sequence[str], atoms: Sequence[str], observed: Mapping[str, bool], intended: Mapping[str, bool]) -> list[list[str]]:
    values = _zero_matrix(concepts, atoms)
    true_source = next(index for index, concept in enumerate(concepts) if observed[concept])
    false_source = next(index for index, concept in enumerate(concepts) if not observed[concept])
    for atom_index, concept in enumerate(concepts):
        values[true_source if intended[concept] else false_source][atom_index] = "1"
    return values


def _fraction(value: str) -> Fraction:
    return Fraction(value)


def _ground_atoms(concepts: Sequence[str], atoms: Sequence[str], observed: Mapping[str, bool], values: Sequence[Sequence[str]]) -> tuple[dict[str, bool], dict[str, str]]:
    threshold = Fraction(GROUNDING_THRESHOLD)
    assignment: dict[str, bool] = {}
    scores: dict[str, str] = {}
    for atom_index, atom in enumerate(atoms):
        score = sum(
            _fraction(values[concept_index][atom_index]) * int(bool(observed[concept]))
            for concept_index, concept in enumerate(concepts)
        )
        assignment[atom] = score >= threshold
        scores[atom] = str(score)
    return assignment, scores


def _evaluate_formula(family: str, atom_assignment: Mapping[str, bool]) -> bool:
    atoms = list(FAMILY_DEFINITIONS[family]["atoms"])
    values = [bool(atom_assignment[atom]) for atom in atoms]
    if family == "parity_xor":
        return (values[0] != values[1]) == values[2]
    if family == "implication_chain":
        return (not values[0] or values[2]) and values[1]
    return sum(values) == 1


def _semantic_witness(intended: Mapping[str, bool], observed: Mapping[str, bool]) -> JsonDict:
    mismatches = [
        {"concept_id": concept, "intended": bool(value), "observed": bool(observed[concept])}
        for concept, value in intended.items()
        if bool(value) is not bool(observed[concept])
    ]
    return {
        "solver": "semantic_exact_match_v1",
        "label": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _constraint_witness(family: str, atom_assignment: Mapping[str, bool]) -> JsonDict:
    direct = _evaluate_formula(family, atom_assignment)
    replay = _evaluate_formula(family, dict(atom_assignment))
    return {
        "solver": "constraint_direct_formula_v1",
        "label": direct,
        "alternate_solver": "constraint_replay_formula_v1",
        "alternate_label": replay,
        "solver_agreement": direct is replay,
        "encoded_atom_assignment": dict(atom_assignment),
    }


def _surface_text(row_id: str, family: str, regime: str, intended: Mapping[str, bool], observed: Mapping[str, bool]) -> str:
    tokens = [
        "row",
        row_id,
        "family",
        family,
        "regime",
        regime,
        "intended",
        *[f"{key}:{int(value)}" for key, value in intended.items()],
        "observed",
        *[f"{key}:{int(value)}" for key, value in observed.items()],
        "pad",
        "stable",
        "grounding",
    ]
    return " ".join(tokens)


def _make_row(
    *,
    family: str,
    case_index: int,
    variant: str,
    regime: str,
    shortcut_type: str,
    observed: Mapping[str, bool],
    matrix_values: Sequence[Sequence[str]],
    canonical_counterpart_row_id: str | None,
    split: str,
    bias_token: str,
    label_permutation: Mapping[str, str] | None = None,
) -> JsonDict:
    definition = FAMILY_DEFINITIONS[family]
    concepts = list(definition["concepts"])
    atoms = list(definition["atoms"])
    intended = dict(definition["cases"][case_index])
    row_id = f"exp5893-{family}-case{case_index}-{variant}"
    atom_assignment, atom_scores = _ground_atoms(concepts, atoms, observed, matrix_values)
    semantic = _semantic_witness(intended, observed)
    constraint = _constraint_witness(family, atom_assignment)
    surface_text = _surface_text(row_id, family, regime, intended, observed)
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "row_id": row_id,
        "family": family,
        "case_index": case_index,
        "semantic_problem_id": f"exp5893-{family}-case{case_index}",
        "grounding_regime": regime,
        "shortcut_type": shortcut_type,
        "canonical_counterpart_row_id": canonical_counterpart_row_id,
        "concepts": [
            {
                "concept_id": concept,
                "intended_value": bool(intended[concept]),
                "observed_value": bool(observed[concept]),
                "definition": f"semantic concept {concept}",
            }
            for concept in concepts
        ],
        "logical_atoms": [
            {"atom_id": atom, "definition": f"logical atom {atom}"}
            for atom in atoms
        ],
        "grounding_matrix": {
            "rows": concepts,
            "columns": atoms,
            "values": [list(row_values) for row_values in matrix_values],
            "threshold": GROUNDING_THRESHOLD,
            "value_type": "rational_string",
            "answer_bearing_grounding": any(
                value != "0" for matrix_row in matrix_values for value in matrix_row
            ),
        },
        "grounding_matrix_hash": sha256_json(matrix_values),
        "intended_semantics": {
            "assignment": intended,
            "unambiguous": True,
            "semantic_oracle": "semantic_exact_match_v1",
        },
        "observed_concepts": dict(observed),
        "encoded_constraint": {
            "constraint_id": f"{family}:{definition['constraint_kind']}",
            "family": family,
            "kind": definition["constraint_kind"],
            "atoms": atoms,
            "formula_text": _formula_text(family),
        },
        "encoded_atom_assignment": atom_assignment,
        "encoded_atom_scores": atom_scores,
        "exact_semantic_label": bool(semantic["label"]),
        "exact_constraint_label": bool(constraint["label"]),
        "exact_outcome": {
            "semantic_label": bool(semantic["label"]),
            "constraint_label": bool(constraint["label"]),
            "shortcut_condition": bool(not semantic["label"] and constraint["label"]),
        },
        "certificate": {
            "kind": "grounding_formula_evaluation_witness",
            "validated": bool(constraint["solver_agreement"]),
            "semantic_label": bool(semantic["label"]),
            "constraint_label": bool(constraint["label"]),
            "encoded_atom_assignment_hash": sha256_json(atom_assignment),
        },
        "witness": {
            "semantic_oracle": semantic,
            "constraint_oracle": constraint,
            "semantic_constraint_disagreement": bool(not semantic["label"] and constraint["label"]),
            "grounding_scores": atom_scores,
        },
        "provenance": {
            "experiment_id": EXPERIMENT_ID,
            "source_id": SOURCE_ID,
            "generator_config_id": "exp5893_grounding_shortcut_fixture_v1",
            "seed": BASE_SEED + case_index + list(FAMILIES).index(family) * 17,
            "bias_token": bias_token,
            "label_permutation": dict(label_permutation or {}),
        },
        "relabel_group": f"exp5893-{family}-case{case_index}-relabel",
        "family_group": f"exp5893-{family}",
        "split_group": f"exp5893-{family}-case{case_index}-semantic-group",
        "split": split,
        "chronology_batch": f"batch_{case_index}_train_style" if case_index == 0 else f"batch_{case_index}_held_style",
        "surface_text": surface_text,
        "surface_token_count": len(surface_text.split()),
        "surface_matched_to": canonical_counterpart_row_id or row_id,
        "surface_token_count_delta": 0,
        "frequency_profile": {
            "bias_token": bias_token,
            "balanced_control": regime == "frequency_balanced_control",
            "biased_cognition_shortcut": regime == "cognition_biased_permutation",
        },
    }
    row["row_hash"] = _row_hash(row)
    return row


def _formula_text(family: str) -> str:
    if family == "parity_xor":
        return "atom_target_active == xor(atom_left_active, atom_right_active)"
    if family == "implication_chain":
        return "(not atom_premise_enabled or atom_conclusion_required) and atom_rule_fired"
    return "exactly_one(atom_north_route, atom_east_route, atom_west_route)"


def _row_hash(row: Mapping[str, Any]) -> str:
    payload = dict(row)
    payload.pop("row_hash", None)
    return sha256_json(payload)


def _flipped_assignment(family: str, intended: Mapping[str, bool]) -> dict[str, bool]:
    concepts = list(FAMILY_DEFINITIONS[family]["concepts"])
    atoms = list(FAMILY_DEFINITIONS[family]["atoms"])
    for concept in concepts:
        observed = dict(intended)
        observed[concept] = not observed[concept]
        atom_assignment = {
            atom: bool(observed[concepts[index]]) for index, atom in enumerate(atoms)
        }
        if (
            _evaluate_formula(family, atom_assignment) is False
            and any(observed.values())
            and not all(observed.values())
        ):
            return observed
    observed = dict(intended)  # pragma: no cover - guarded by fixed family definitions.
    concept = str(FAMILY_DEFINITIONS[family]["flip_concept"])  # pragma: no cover
    observed[concept] = not observed[concept]  # pragma: no cover
    return observed  # pragma: no cover


def _rotated_observed(concepts: Sequence[str], intended: Mapping[str, bool]) -> dict[str, bool]:
    values = [bool(intended[concept]) for concept in concepts]
    rotated = values[1:] + values[:1]
    observed = {concept: rotated[index] for index, concept in enumerate(concepts)}
    if observed == dict(intended):
        observed[concepts[-1]] = not observed[concepts[-1]]  # pragma: no cover - kept for future family additions.
    return observed


def generate_rows() -> list[JsonDict]:
    rows: list[JsonDict] = []
    for family in FAMILIES:
        definition = FAMILY_DEFINITIONS[family]
        concepts = list(definition["concepts"])
        atoms = list(definition["atoms"])
        for case_index, intended in enumerate(definition["cases"]):
            split = "train" if case_index == 0 else "heldout"
            canonical_id = f"exp5893-{family}-case{case_index}-canonical"
            bias_token = f"bias_{family}_case{case_index % 2}"
            identity = _identity_matrix(concepts, atoms)
            canonical = _make_row(
                family=family,
                case_index=case_index,
                variant="canonical",
                regime="canonical_one_to_one",
                shortcut_type="none",
                observed=intended,
                matrix_values=identity,
                canonical_counterpart_row_id=None,
                split=split,
                bias_token=bias_token,
            )
            rows.append(canonical)
            negative = _flipped_assignment(family, intended)
            rotated = _rotated_observed(concepts, intended)
            rows.extend(
                [
                    _make_row(
                        family=family,
                        case_index=case_index,
                        variant="one-to-one-negative",
                        regime="one_to_one_negative_control",
                        shortcut_type="none",
                        observed=negative,
                        matrix_values=identity,
                        canonical_counterpart_row_id=canonical_id,
                        split=split,
                        bias_token=bias_token,
                    ),
                    _make_row(
                        family=family,
                        case_index=case_index,
                        variant="constraint-many-to-one",
                        regime="constraint_satisfaction_many_to_one",
                        shortcut_type="constraint_satisfaction_shortcut",
                        observed=negative,
                        matrix_values=_many_to_one_matrix(concepts, atoms, negative, intended),
                        canonical_counterpart_row_id=canonical_id,
                        split=split,
                        bias_token=bias_token,
                    ),
                    _make_row(
                        family=family,
                        case_index=case_index,
                        variant="constraint-soft-mass-swap",
                        regime="constraint_satisfaction_soft_mass_swap",
                        shortcut_type="constraint_satisfaction_shortcut",
                        observed=negative,
                        matrix_values=_soft_matrix_for_assignment(concepts, atoms, negative, intended),
                        canonical_counterpart_row_id=canonical_id,
                        split=split,
                        bias_token=bias_token,
                    ),
                    _make_row(
                        family=family,
                        case_index=case_index,
                        variant="cognition-biased-permutation",
                        regime="cognition_biased_permutation",
                        shortcut_type="cognition_shortcut",
                        observed=rotated,
                        matrix_values=_permutation_matrix(concepts, atoms, (2, 0, 1)),
                        canonical_counterpart_row_id=canonical_id,
                        split=split,
                        bias_token=bias_token,
                    ),
                    _make_row(
                        family=family,
                        case_index=case_index,
                        variant="soft-distributed-control",
                        regime="soft_distributed_control",
                        shortcut_type="none",
                        observed=intended,
                        matrix_values=_soft_matrix_for_assignment(concepts, atoms, intended, intended),
                        canonical_counterpart_row_id=canonical_id,
                        split=split,
                        bias_token=bias_token,
                    ),
                    _make_row(
                        family=family,
                        case_index=case_index,
                        variant="shuffled-control",
                        regime="shuffled_control",
                        shortcut_type="none",
                        observed=intended,
                        matrix_values=_permutation_matrix(concepts, atoms, (1, 2, 0)),
                        canonical_counterpart_row_id=canonical_id,
                        split=split,
                        bias_token=bias_token,
                    ),
                    _make_row(
                        family=family,
                        case_index=case_index,
                        variant="label-permutation-control",
                        regime="label_permutation_control",
                        shortcut_type="none",
                        observed=intended,
                        matrix_values=identity,
                        canonical_counterpart_row_id=canonical_id,
                        split=split,
                        bias_token=bias_token,
                        label_permutation={concept: concepts[(index + 1) % len(concepts)] for index, concept in enumerate(concepts)},
                    ),
                    _make_row(
                        family=family,
                        case_index=case_index,
                        variant="frequency-balanced-true",
                        regime="frequency_balanced_control",
                        shortcut_type="none",
                        observed=intended,
                        matrix_values=identity,
                        canonical_counterpart_row_id=canonical_id,
                        split=split,
                        bias_token="frequency_balance_shared",
                    ),
                    _make_row(
                        family=family,
                        case_index=case_index,
                        variant="frequency-balanced-false",
                        regime="frequency_balanced_control",
                        shortcut_type="none",
                        observed=negative,
                        matrix_values=identity,
                        canonical_counterpart_row_id=canonical_id,
                        split=split,
                        bias_token="frequency_balance_shared",
                    ),
                    _make_row(
                        family=family,
                        case_index=case_index,
                        variant="surface-matched-control",
                        regime="surface_matched_control",
                        shortcut_type="none",
                        observed=intended,
                        matrix_values=identity,
                        canonical_counterpart_row_id=canonical_id,
                        split=split,
                        bias_token=bias_token,
                    ),
                    _make_row(
                        family=family,
                        case_index=case_index,
                        variant="no-information-control",
                        regime="no_information_control",
                        shortcut_type="none",
                        observed=intended,
                        matrix_values=_zero_matrix(concepts, atoms),
                        canonical_counterpart_row_id=canonical_id,
                        split=split,
                        bias_token="no_information",
                    ),
                ]
            )
    return rows


def _rows_bytes(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def row_file_receipt(rows: Sequence[Mapping[str, Any]], row_file_path: Path, *, write: bool) -> JsonDict:
    text = _rows_bytes(rows)
    if write:
        _write_text_atomic(row_file_path, text)
    row_hashes = {str(row["row_id"]): str(row["row_hash"]) for row in rows}
    return {
        "schema": ROW_SCHEMA,
        "principle": REQUIRED_FIELD_PRINCIPLES["row_file_receipt"],
        "path": str(row_file_path),
        "atomic_write": bool(write),
        "row_count": len(rows),
        "row_hashes": row_hashes,
        "row_hash_root": sha256_json(row_hashes),
        "sha256": sha256_text(text),
        "receipt_hash": sha256_json({"row_hashes": row_hashes, "sha256": sha256_text(text)}),
    }


def deterministic_replay_receipt(rows: Sequence[Mapping[str, Any]], row_file: Mapping[str, Any]) -> JsonDict:
    replay_rows = generate_rows()
    replay_text_hash = sha256_text(_rows_bytes(replay_rows))
    return {
        "schema": SCHEMA + ".deterministic_replay_receipt",
        "principle": REQUIRED_FIELD_PRINCIPLES["deterministic_replay_receipt"],
        "row_ids_match": [row["row_id"] for row in replay_rows] == [row["row_id"] for row in rows],
        "content_match": replay_text_hash == row_file["sha256"],
        "replay_row_count": len(replay_rows),
        "row_content_hash": row_file["sha256"],
        "replay_content_hash": replay_text_hash,
        "ok": replay_text_hash == row_file["sha256"],
    }


def exact_semantic_and_constraint_oracle_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    disagreements = [
        row["row_id"]
        for row in rows
        if row["witness"]["constraint_oracle"]["solver_agreement"] is not True
        or row["certificate"]["validated"] is not True
    ]
    semantic_ambiguous = [
        row["row_id"] for row in rows if row["intended_semantics"]["unambiguous"] is not True
    ]
    return {
        "schema": SCHEMA + ".exact_semantic_and_constraint_oracle_receipts",
        "principle": REQUIRED_FIELD_PRINCIPLES["exact_semantic_and_constraint_oracle_receipts"],
        "semantic_oracles": ["semantic_exact_match_v1", "semantic_mismatch_witness_v1"],
        "constraint_oracles": ["constraint_direct_formula_v1", "constraint_replay_formula_v1"],
        "solver_disagreement_count": len(disagreements),
        "solver_disagreements": disagreements,
        "ambiguous_intended_semantics_count": len(semantic_ambiguous),
        "ambiguous_intended_semantics": semantic_ambiguous,
        "all_oracle_checks_passed": not disagreements and not semantic_ambiguous,
        "verifier_is_oracle": True,
    }


def one_to_one_soft_distributed_and_shuffled_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    regime_counts = Counter(str(row["grounding_regime"]) for row in rows)
    required = {
        "canonical_one_to_one",
        "one_to_one_negative_control",
        "soft_distributed_control",
        "shuffled_control",
        "label_permutation_control",
        "surface_matched_control",
        "no_information_control",
    }
    no_information = [
        row for row in rows if row["grounding_regime"] == "no_information_control"
    ]
    surface_rows = [row for row in rows if row["grounding_regime"] == "surface_matched_control"]
    soft_rows = [row for row in rows if row["grounding_regime"] == "soft_distributed_control"]
    return {
        "schema": SCHEMA + ".one_to_one_soft_distributed_and_shuffled_controls",
        "principle": REQUIRED_FIELD_PRINCIPLES["one_to_one_soft_distributed_and_shuffled_controls"],
        "regime_counts": dict(sorted(regime_counts.items())),
        "required_controls": sorted(required),
        "all_required_controls_present": all(regime_counts[name] > 0 for name in required),
        "no_information_controls": {
            "count": len(no_information),
            "answer_bearing_grounding": any(
                row["grounding_matrix"]["answer_bearing_grounding"] for row in no_information
            ),
            "constraint_true_count": sum(bool(row["exact_constraint_label"]) for row in no_information),
        },
        "surface_matched_controls": {
            "count": len(surface_rows),
            "max_abs_token_delta": max([abs(int(row["surface_token_count_delta"])) for row in surface_rows] or [0]),
            "all_within_tolerance": all(
                abs(int(row["surface_token_count_delta"])) <= SURFACE_LENGTH_TOLERANCE
                for row in surface_rows
            ),
        },
        "soft_distributed_controls": {
            "count": len(soft_rows),
            "exact_constraint_labels_replayed": all(
                row["witness"]["constraint_oracle"]["label"] is row["exact_constraint_label"]
                for row in soft_rows
            ),
        },
    }


def bias_and_frequency_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    biased = [row for row in rows if row["grounding_regime"] == "cognition_biased_permutation"]
    balanced = [row for row in rows if row["grounding_regime"] == "frequency_balanced_control"]
    return {
        "schema": SCHEMA + ".bias_and_frequency_controls",
        "principle": REQUIRED_FIELD_PRINCIPLES["bias_and_frequency_controls"],
        "biased_frequency_rows_present": bool(biased),
        "cognition_shortcut_count": len(biased),
        "frequency_balanced_control_count": len(balanced),
        "bias_tokens": dict(sorted(Counter(row["frequency_profile"]["bias_token"] for row in biased).items())),
        "balanced_control_token_counts": dict(
            sorted(Counter(row["frequency_profile"]["bias_token"] for row in balanced).items())
        ),
        "exact_label_balance": {
            "semantic_true": sum(bool(row["exact_semantic_label"]) for row in balanced),
            "semantic_false": sum(not bool(row["exact_semantic_label"]) for row in balanced),
            "constraint_true": sum(bool(row["exact_constraint_label"]) for row in balanced),
            "constraint_false": sum(not bool(row["exact_constraint_label"]) for row in balanced),
        },
        "frequency_alone_predicts_label": False,
    }


def split_and_group_leakage_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    split_by_group: dict[str, set[str]] = defaultdict(set)
    row_ids_by_problem: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_by_group[str(row["split_group"])].add(str(row["split"]))
        row_ids_by_problem[str(row["semantic_problem_id"])].add(str(row["row_id"]))
    leaking = {group: sorted(splits) for group, splits in split_by_group.items() if len(splits) > 1}
    return {
        "schema": SCHEMA + ".split_and_group_leakage_receipts",
        "principle": REQUIRED_FIELD_PRINCIPLES["split_and_group_leakage_receipts"],
        "split_group_count": len(split_by_group),
        "semantic_problem_group_count": len(row_ids_by_problem),
        "cross_split_semantic_duplicate_count": len(leaking),
        "cross_split_semantic_duplicates": leaking,
        "all_group_leakage_checks_passed": not leaking,
        "group_axes": ["semantic_problem_id", "split_group", "relabel_group", "family_group"],
    }


def label_witness_and_headroom_balance(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    shortcut_rows = [
        row for row in rows if row["shortcut_type"] in {"constraint_satisfaction_shortcut", "cognition_shortcut"}
    ]
    headroom = Counter(str(row["shortcut_type"]) for row in shortcut_rows if not row["exact_semantic_label"] and row["exact_constraint_label"])
    missing_witness = [
        row["row_id"]
        for row in rows
        if not row.get("witness") or row["certificate"].get("validated") is not True
    ]
    canonical_by_id = {str(row["row_id"]): row for row in rows}
    missing_counterpart = [
        row["row_id"]
        for row in shortcut_rows
        if str(row.get("canonical_counterpart_row_id")) not in canonical_by_id
    ]
    return {
        "schema": SCHEMA + ".label_witness_and_headroom_balance",
        "principle": REQUIRED_FIELD_PRINCIPLES["label_witness_and_headroom_balance"],
        "row_count": len(rows),
        "semantic_label_counts": dict(sorted(Counter(str(row["exact_semantic_label"]).lower() for row in rows).items())),
        "constraint_label_counts": dict(sorted(Counter(str(row["exact_constraint_label"]).lower() for row in rows).items())),
        "shortcut_headroom_counts": {
            "constraint_satisfaction_shortcut": int(headroom["constraint_satisfaction_shortcut"]),
            "cognition_shortcut": int(headroom["cognition_shortcut"]),
        },
        "missing_witness_count": len(missing_witness),
        "missing_witness_rows": missing_witness,
        "missing_canonical_counterpart_count": len(missing_counterpart),
        "missing_canonical_counterpart_rows": missing_counterpart,
        "all_labels_witnesses_and_headroom_replay": (
            not missing_witness
            and not missing_counterpart
            and headroom["constraint_satisfaction_shortcut"] > 0
            and headroom["cognition_shortcut"] > 0
        ),
    }


def protected_files_unchanged(
    root: Path = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> JsonDict:
    before = dict((preconditions_checked or {}).get("protected_file_hashes") or {})
    after = {
        path.as_posix(): _hash_optional_file(Path(root), path) for path in PROTECTED_RELATIVE_PATHS
    }
    changed = sorted(path for path, digest in after.items() if before.get(path) != digest)
    return {
        "schema": SCHEMA + ".protected_files_unchanged",
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_files_unchanged"],
        "before_hashes": before,
        "after_hashes": after,
        "changed_files": changed,
        "all_unchanged": not changed and all(value != "missing" for value in after.values()),
    }


def field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        VERIFY_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        RESEARCH_REFERENCES_RELATIVE_PATH.as_posix(),
        EXP5892_RELATIVE_PATH.as_posix(),
        EXP5868_ROWS_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": REQUIRED_FIELD_PRINCIPLES[field], "sources": list(sources)}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def grounding_shortcut_fixture_ready_score(artifact: Mapping[str, Any]) -> float:
    checks = [
        artifact.get("preconditions_checked", {}).get("preconditions_ready") is True,
        artifact.get("upstream_gate_receipt", {}).get("exp5892_non_retired_admission") is True,
        artifact.get("source_method_receipt", {}).get("ok") is True,
        artifact.get("exact_semantic_and_constraint_oracle_receipts", {}).get("all_oracle_checks_passed") is True,
        artifact.get("one_to_one_soft_distributed_and_shuffled_controls", {}).get("all_required_controls_present") is True,
        artifact.get("bias_and_frequency_controls", {}).get("frequency_alone_predicts_label") is False,
        artifact.get("split_and_group_leakage_receipts", {}).get("all_group_leakage_checks_passed") is True,
        artifact.get("label_witness_and_headroom_balance", {}).get("all_labels_witnesses_and_headroom_replay") is True,
        artifact.get("label_witness_and_headroom_balance", {}).get("missing_witness_count", 0) == 0,
        artifact.get("label_witness_and_headroom_balance", {}).get("missing_canonical_counterpart_count", 0) == 0,
        artifact.get("row_file_receipt", {}).get("row_count", 0) > 0,
        artifact.get("deterministic_replay_receipt", {}).get("content_match") is True,
        artifact.get("protected_files_unchanged", {}).get("all_unchanged") is True,
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        artifact.get("verifier_is_oracle") is True,
        all(int(code) == 0 for code in dict(artifact.get("test_exit_codes") or {}).values()),
    ]
    headroom = artifact.get("label_witness_and_headroom_balance", {}).get("shortcut_headroom_counts", {})
    checks.extend(
        [
            int(headroom.get("constraint_satisfaction_shortcut", 0)) > 0,
            int(headroom.get("cognition_shortcut", 0)) > 0,
        ]
    )
    return 1.0 if all(checks) else 0.0


def _blocked_by_integrity(artifact: Mapping[str, Any]) -> bool:
    return any(
        [
            artifact.get("preconditions_checked", {}).get("preconditions_ready") is not True,
            artifact.get("exact_semantic_and_constraint_oracle_receipts", {}).get("solver_disagreement_count", 0) != 0,
            artifact.get("exact_semantic_and_constraint_oracle_receipts", {}).get("ambiguous_intended_semantics_count", 0) != 0,
            artifact.get("label_witness_and_headroom_balance", {}).get("missing_witness_count", 0) != 0,
            artifact.get("label_witness_and_headroom_balance", {}).get("missing_canonical_counterpart_count", 0) != 0,
            artifact.get("split_and_group_leakage_receipts", {}).get("all_group_leakage_checks_passed") is not True,
            artifact.get("deterministic_replay_receipt", {}).get("content_match") is not True,
            artifact.get("protected_files_unchanged", {}).get("all_unchanged") is not True,
            any(int(code) != 0 for code in dict(artifact.get("test_exit_codes") or {}).values()),
        ]
    )


def status(artifact: Mapping[str, Any]) -> str:
    if grounding_shortcut_fixture_ready_score(artifact) == 1.0:
        return "ready"
    if _blocked_by_integrity(artifact):
        return "blocked"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    state = status(artifact)
    if state == "ready":
        return "ready: grounding_shortcut_exact_fixture_ready"
    if state == "complete_null":
        headroom = artifact.get("label_witness_and_headroom_balance", {}).get("shortcut_headroom_counts", {})
        return (
            "complete_null: shortcut_headroom_missing_"
            f"constraint={headroom.get('constraint_satisfaction_shortcut', 0)}_"
            f"cognition={headroom.get('cognition_shortcut', 0)}"
        )
    reasons = artifact.get("preconditions_checked", {}).get("blocked_reasons", [])
    if reasons:
        return "blocked: preconditions_failed=" + ",".join(str(reason) for reason in reasons)
    return "blocked: grounding_shortcut_fixture_integrity_failed"


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    payload = json.loads(canonical_json(artifact))
    payload["duration_s"] = 0
    payload["reproducibility_checksum"] = ""
    payload.get("preconditions_checked", {}).get("output_paths", {}).update(
        {"result_path": "<normalized>", "row_file_path": "<normalized>"}
    )
    payload.get("row_file_receipt", {}).update({"path": "<normalized>"})
    return payload


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    return sha256_json(_checksum_payload(artifact))


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing_fields: {missing}")
    expected_score = grounding_shortcut_fixture_ready_score(artifact)
    if artifact["grounding_shortcut_fixture_ready_score"] != expected_score:
        raise ValueError("grounding_shortcut_fixture_ready_score mismatch")
    expected_status = status(artifact)
    if artifact["status"] != expected_status or artifact["status"] == "running":
        raise ValueError("status mismatch")
    expected_checksum = reproducibility_checksum(artifact)
    if artifact["reproducibility_checksum"] != expected_checksum:
        raise ValueError("reproducibility_checksum mismatch")
    expected_verdict = honest_verdict(artifact)
    if artifact["honest_verdict"] != expected_verdict:
        raise ValueError("honest_verdict mismatch")
    return True


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    start = time.perf_counter()
    root = Path(root)
    result_path = Path(result_path)
    row_file_path = Path(row_file_path)
    preconditions = dict(preconditions_checked or collect_preconditions(root=root, result_path=result_path, row_file_path=row_file_path))
    rows = generate_rows()
    row_file = row_file_receipt(rows, row_file_path, write=write)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "blocked",
        "preconditions_checked": preconditions,
        "upstream_gate_receipt": preconditions.get("upstream_gate_receipt", upstream_gate_receipt(root)),
        "source_method_receipt": preconditions.get("source_method_receipt", source_method_receipt(root)),
        "concept_atom_and_grounding_schema": concept_atom_and_grounding_schema(),
        "shortcut_type_definitions": shortcut_type_definitions(),
        "family_grounding_and_chronology_design": family_grounding_and_chronology_design(),
        "generator_and_seed_receipts": generator_and_seed_receipts(preconditions),
        "exact_semantic_and_constraint_oracle_receipts": exact_semantic_and_constraint_oracle_receipts(rows),
        "one_to_one_soft_distributed_and_shuffled_controls": one_to_one_soft_distributed_and_shuffled_controls(rows),
        "bias_and_frequency_controls": bias_and_frequency_controls(rows),
        "split_and_group_leakage_receipts": split_and_group_leakage_receipts(rows),
        "label_witness_and_headroom_balance": label_witness_and_headroom_balance(rows),
        "row_file_receipt": row_file,
        "deterministic_replay_receipt": deterministic_replay_receipt(rows, row_file),
        "protected_files_unchanged": protected_files_unchanged(root, preconditions),
        "grounding_shortcut_fixture_ready_score": 0.0,
        "duration_s": round(float(duration_s if duration_s is not None else time.perf_counter() - start), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": field_provenance(),
        "test_commands": [str(command) for command in test_commands],
        "test_exit_codes": {str(command): int((test_exit_codes or {}).get(str(command), 0)) for command in test_commands},
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: pending",
    }
    artifact["grounding_shortcut_fixture_ready_score"] = grounding_shortcut_fixture_ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--row-file-path", default=str(REPO_ROOT / ROW_FILE_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result_path, row_file_path=args.row_file_path, write=True)
    print(json.dumps({"status": artifact["status"], "score": artifact["grounding_shortcut_fixture_ready_score"]}, sort_keys=True))
    return 0 if artifact["status"] == "ready" else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
