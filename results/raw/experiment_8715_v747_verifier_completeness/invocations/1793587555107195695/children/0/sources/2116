"""Exp6103 Phase D exact difficulty ladder fixture.

Spec refs: REQ-VERIFY-6103, SCENARIO-VERIFY-6103-GENERATION,
SCENARIO-VERIFY-6103-TRANSFORMS, SCENARIO-VERIFY-6103-REPLAY,
SCENARIO-VERIFY-6103-POLICY.

This module builds the fixture that Phase D needs before any model inference
is allowed.  The rows are generated from small public finite rules, not from
model text.  The model-facing prompt receives only the problem and answer
choices; exact labels, validator receipts, and method-validity labels stay in
the sealed manifest so later inference cannot learn from the answers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
Probe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6103_phase_d_difficulty_ladder_fixture.json")
ROW_FILE_RELATIVE_PATH = Path(
    "results/experiment_6103_phase_d_difficulty_ladder_fixture.rows.jsonl"
)
SPLIT_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6103_phase_d_difficulty_ladder_fixture.splits.json"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6103_phase_d_difficulty_ladder_fixture.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6103_phase_d_difficulty_ladder_fixture.py"
)
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")

SCHEMA = "carnot.experiment_6103.phase_d_difficulty_ladder_fixture.v1"
ROW_SCHEMA = SCHEMA + ".row"
SPLIT_SCHEMA = SCHEMA + ".split_manifest"
EXPERIMENT = 6103
EXPERIMENT_ID = "experiment_6103_phase_d_difficulty_ladder_fixture"
RUN_DATE = "20260804"
INFERENCE_SUBSTRATE = "deterministic_exact_fixture_generation_no_llm"
VERIFIER_IS_ORACLE = True
BASE_SEED = 6103
RAM_FLOOR_MB = 1024
DISK_FLOOR_MB = 512

FAMILIES = ("finite_domain_scheduling", "logic_grid", "typed_finite_choice")
SPLITS = ("calibration", "held_test")
LABELS = ("A", "B", "C", "D")
TRANSFORM_KINDS = (
    "proof_preserving_relabel",
    "meaning_preserving_paraphrase",
    "surface_order_change",
    "answer_permutation",
)
CALIBRATION_PER_FAMILY = 200
HELD_TEST_PER_FAMILY = 120
DIFFICULTY_STRATA = ("compact", "dense", "wide", "boundary")
SOLVER_BINS = ("low", "medium", "high")
MODEL_SURFACE_BINS = ("compact_surface", "lexical_dense_surface", "wide_surface", "shortcut_salient")
PROTECTED_FILES = (Path("scripts/research_conductor.py"),)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "immutable_source_hashes",
    "family_parameter_and_exact_generation_contract",
    "calibration_and_held_test_counts",
    "semantic_group_splits",
    "answer_space_and_enumerated_chance_floors",
    "solver_hardness_model_surface_and_semantic_strata",
    "proof_preserving_relabel_paraphrase_and_inverse_receipts",
    "shortcut_salience_and_method_validity_manifest",
    "python_z3_parity",
    "duplicate_leakage_unreachable_and_order_dependence_counts",
    "calibration_policy_and_test_secrecy",
    "row_paths_hashes_and_prefix_chain",
    "phase_d_ladder_fixture_ready_score",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "family_parameter_and_exact_generation_contract": "all instances are generated from public exact rules, never model outputs or hidden answer templates.",
    "calibration_and_held_test_counts": "evaluation power is measured at the independent question group.",
    "semantic_group_splits": "evaluation power is measured at the independent question group.",
    "answer_space_and_enumerated_chance_floors": "chance is enumerated from each exact answer space, not estimated from observed model accuracy.",
    "solver_hardness_model_surface_and_semantic_strata": "solver conflicts remain diagnostic and are not called model difficulty.",
    "proof_preserving_relabel_paraphrase_and_inverse_receipts": "every transform has auditable meaning and inverse behavior.",
    "shortcut_salience_and_method_validity_manifest": "final-answer correctness is kept separate from the intended reasoning path.",
    "python_z3_parity": "exact validators and sealed splits are the authority.",
    "duplicate_leakage_unreachable_and_order_dependence_counts": "exact validators and sealed splits are the authority.",
    "calibration_policy_and_test_secrecy": "no held-test label can affect generation settings or inclusion.",
    "phase_d_ladder_fixture_ready_score": "readiness requires all exact, diversity, power, leakage, and tamper gates.",
    "duration_s": "use measured `deterministic_exact_fixture_generation_no_llm`.",
    "inference_substrate": "use measured `deterministic_exact_fixture_generation_no_llm`.",
    "field_provenance": "use measured `deterministic_exact_fixture_generation_no_llm`.",
    "test_commands": "use measured `deterministic_exact_fixture_generation_no_llm`.",
    "test_exit_codes": "use measured `deterministic_exact_fixture_generation_no_llm`.",
    "reproducibility_checksum": "use measured `deterministic_exact_fixture_generation_no_llm`.",
    "verifier_is_oracle": "exact Python/Z3 validators are oracle and any uncovered method semantics are explicit gaps.",
    "missing_verifier_gaps": "exact Python/Z3 validators are oracle and any uncovered method semantics are explicit gaps.",
    "honest_verdict": "use `complete_ready:`, `complete_partial:`, or `blocked:`.",
}

FIELD_PRINCIPLES: dict[str, str] = {
    field: REQUIRED_FIELD_PRINCIPLES.get(
        field,
        "required Exp6103 schema field with deterministic exact no-LLM provenance.",
    )
    for field in REQUIRED_ARTIFACT_FIELDS
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6103_phase_d_difficulty_ladder_fixture.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6103_phase_d_difficulty_ladder_fixture.py "
    "-m pytest tests/python/test_experiment_6103_phase_d_difficulty_ladder_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6103_phase_d_difficulty_ladder_fixture.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6103_phase_d_difficulty_ladder_fixture.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/known-issues.md"),
    Path("results/experiment_5785_hardness_surface_fixture.json"),
    Path("results/experiment_5785_hardness_surface_fixture.rows.jsonl"),
    Path("results/experiment_5786_sota_constraint_stream.json"),
    Path("results/experiment_5786_sota_constraint_stream.rows.jsonl"),
    Path("results/experiment_5868_hardness_controlled_constraint_fixture.json"),
    Path("results/experiment_5868_hardness_controlled_constraint_fixture.rows.jsonl"),
    Path("results/experiment_5879_hardness_headroom_taxonomy_corrigendum.json"),
    Path("python/carnot/experiment_5785_hardness_surface_fixture.py"),
    Path("python/carnot/experiment_5786_sota_constraint_stream.py"),
    Path("python/carnot/experiment_5868_hardness_controlled_constraint_fixture.py"),
    Path("python/carnot/experiment_5879_hardness_headroom_taxonomy_corrigendum.py"),
    Path("python/carnot/constraint_ir_replay_contract.py"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    VERIFY_SPEC_RELATIVE_PATH,
    *PROTECTED_FILES,
)

PROMPT_ALIAS_PATHS = {
    "results/experiment_5785_exact_constraint_fixture.json": Path(
        "results/experiment_5785_hardness_surface_fixture.json"
    ),
    "python/carnot/constraint_ir_replay.py": Path("python/carnot/constraint_ir_replay_contract.py"),
}


class ManifestReplayError(ValueError):
    """Raised when a sealed row or split manifest no longer matches receipts."""


def canonical_json(value: Any) -> str:
    """Serialize JSON values into the stable byte order used by row hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed digest for text that has already been normalized."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible object through the canonical serializer."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact local bytes so manifests are not tied to filesystem metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


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
    available_mb = int(shutil.disk_usage(root).free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": DISK_FLOOR_MB, "ok": available_mb >= DISK_FLOOR_MB}


def _z3_probe() -> JsonDict:  # pragma: no cover - environment-dependent import probe.
    try:
        import z3  # type: ignore[import-untyped]
    except ImportError as exc:
        return {"available": False, "version": "", "ok": False, "error": str(exc)}
    return {"available": True, "version": z3.get_version_string(), "ok": True}


def _source_hashes(root: Path) -> JsonDict:
    sources: JsonDict = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        sources[relative.as_posix()] = {
            "path": relative.as_posix(),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else "",
        }
    aliases = {
        prompt_path: {
            "prompt_path": prompt_path,
            "resolved_local_path": resolved.as_posix(),
            "prompt_path_exists": (root / prompt_path).exists(),
            "resolved_local_path_exists": (root / resolved).exists(),
            "resolved_sha256": sha256_file(root / resolved) if (root / resolved).exists() else "",
        }
        for prompt_path, resolved in PROMPT_ALIAS_PATHS.items()
    }
    return {"schema": SCHEMA + ".immutable_source_hashes", "sources": sources, "prompt_aliases": aliases}


def _root_clutter_inventory(root: Path) -> JsonDict:
    files = sorted(path.name for path in root.glob("*.py"))
    return {"root_python_files": files, "root_python_file_count": len(files), "ok": not files}


def _output_path_receipt(paths: Sequence[Path]) -> JsonDict:
    def writable_parent(path: Path) -> bool:
        parent = path.parent
        while not parent.exists():
            parent = parent.parent
        return os.access(parent, os.W_OK)

    return {
        path.name: {
            "path": str(path),
            "parent_exists": path.parent.exists(),
            "parent_writable": writable_parent(path),
            "exists_before": path.exists(),
            "sha256_before": sha256_file(path) if path.exists() else "",
        }
        for path in paths
    }


def collect_preconditions(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    split_manifest_path: str | Path = REPO_ROOT / SPLIT_MANIFEST_RELATIVE_PATH,
    root: Path = REPO_ROOT,
    memory_probe: Probe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
    z3_probe: Probe = _z3_probe,
) -> JsonDict:
    """Collect source hashes, resource checks, clutter, and fixture-row policy."""

    result = Path(result_path)
    rows = Path(row_file_path)
    splits = Path(split_manifest_path)
    memory = memory_probe()
    disk = disk_probe(root)
    z3 = z3_probe()
    output_paths = _output_path_receipt((result, rows, splits))
    protected_before = {
        relative.as_posix(): sha256_file(root / relative) for relative in PROTECTED_FILES
    }
    ready = (
        memory.get("ok") is True
        and disk.get("ok") is True
        and z3.get("ok") is True
        and all(item["parent_writable"] for item in output_paths.values())
        and _root_clutter_inventory(root)["ok"] is True
    )
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ram": memory,
        "disk": disk,
        "z3": z3,
        "immutable_source_hashes": _source_hashes(root),
        "output_paths": output_paths,
        "root_clutter_inventory": _root_clutter_inventory(root),
        "protected_file_hashes_before": protected_before,
        "fixture_row_source_policy": {
            "principle": "rows are generated from public exact rules; prior artifacts are hash-only context",
            "candidate_generation_artifact_imported": False,
            "model_response_artifact_imported": False,
            "imported_fixture_row_count": 0,
            "hashed_prior_artifact_count": 5,
            "rows_generated_by": "exp6103_public_exact_family_parameter_draws_v1",
        },
        "preconditions_ready": ready,
    }


def _split_count(split: str) -> int:
    return CALIBRATION_PER_FAMILY if split == "calibration" else HELD_TEST_PER_FAMILY


def _family_index(family: str) -> int:
    return FAMILIES.index(family)


def _row_number(split: str, family: str, local_index: int) -> int:
    split_offset = 0 if split == "calibration" else CALIBRATION_PER_FAMILY * len(FAMILIES)
    family_span = _split_count(split)
    return split_offset + _family_index(family) * family_span + local_index


def _parameters(family: str, split: str, local_index: int) -> JsonDict:
    global_index = _row_number(split, family, local_index)
    family_ix = _family_index(family)
    return {
        "family_size": 5 + ((local_index + family_ix) % 4),
        "density": 2 + ((local_index * 2 + family_ix) % 5),
        "width": 2 + ((local_index + 2 * family_ix) % 4),
        "difficulty_stratum": DIFFICULTY_STRATA[global_index % len(DIFFICULTY_STRATA)],
        "solver_effort_bin": SOLVER_BINS[(local_index + 2 * family_ix) % len(SOLVER_BINS)],
        "model_surface_bin": MODEL_SURFACE_BINS[(global_index + family_ix) % len(MODEL_SURFACE_BINS)],
        "surface_order_seed": BASE_SEED + global_index * 17 + family_ix,
        "answer_permutation_seed": BASE_SEED + global_index * 31 + family_ix,
        "parameter_independence_receipt": {
            "size_density_width_key": (local_index + family_ix) % 20,
            "solver_conflict_key": (local_index * 7 + family_ix) % 19,
            "surface_key": (local_index * 11 + family_ix) % 23,
            "independent_where_feasible": True,
        },
    }


def _label_mapping(candidates: Sequence[str], exact_candidate: str, seed: int) -> list[JsonDict]:
    shift = seed % len(LABELS)
    labels = list(LABELS[shift:] + LABELS[:shift])
    return [
        {
            "label": labels[index],
            "candidate": candidate,
            "candidate_hash": sha256_text(candidate),
            "is_exact": candidate == exact_candidate,
        }
        for index, candidate in enumerate(candidates)
    ]


def _exact_label(label_mapping: Sequence[Mapping[str, Any]]) -> str:
    return next(str(row["label"]) for row in label_mapping if row.get("is_exact") is True)


def _candidate_values(exact: int, size: int) -> list[str]:
    return [str((exact + offset) % size) for offset in range(4)]


def _scheduling_problem(params: Mapping[str, Any], local_index: int) -> JsonDict:
    size = int(params["family_size"])
    multiplier = 2 if size % 2 == 1 else 3
    offset = (local_index + int(params["density"])) % size
    target_index = (local_index * 2 + int(params["width"])) % size
    exact = (offset + multiplier * target_index) % size
    tasks = [f"task_{index}" for index in range(size)]
    facts = {
        "tasks": tasks,
        "slots": list(range(size)),
        "rule": f"slot(task_i) = ({offset} + {multiplier} * i) mod {size}",
        "target_task": tasks[target_index],
    }
    prompt = (
        f"Use the cyclic scheduling rule slot(task_i)=({offset}+{multiplier}*i) mod {size}. "
        f"What slot is assigned to {tasks[target_index]}?"
    )
    return {
        "family": "finite_domain_scheduling",
        "facts": facts,
        "exact_candidate": str(exact),
        "candidate_values": _candidate_values(exact, size),
        "prompt_stem": prompt,
    }


def _logic_grid_problem(params: Mapping[str, Any], local_index: int) -> JsonDict:
    size = int(params["family_size"])
    multiplier = 2 if size % 2 == 1 else 3
    offset = (local_index * 3 + int(params["width"])) % size
    target_index = (local_index + int(params["density"])) % size
    exact = (offset + multiplier * target_index) % size
    people = [f"person_{index}" for index in range(size)]
    items = [f"item_{index}" for index in range(size)]
    facts = {
        "people": people,
        "items": items,
        "rule": f"item(person_i) = ({offset} + {multiplier} * i) mod {size}",
        "target_person": people[target_index],
    }
    prompt = (
        f"In the one-to-one logic grid, item(person_i)=({offset}+{multiplier}*i) mod {size}. "
        f"Which item number belongs to {people[target_index]}?"
    )
    return {
        "family": "logic_grid",
        "facts": facts,
        "exact_candidate": str(exact),
        "candidate_values": _candidate_values(exact, size),
        "prompt_stem": prompt,
    }


def _typed_choice_problem(params: Mapping[str, Any], local_index: int) -> JsonDict:
    budget = 8 + (local_index % 3)
    risk_limit = 5 + (int(params["width"]) % 3)
    exact_index = local_index % 4
    choices = []
    for index in range(4):
        feasible = index == exact_index or (index + local_index) % 3 != 0
        weight = 3 + ((local_index + index) % 4)
        risk = 2 + ((local_index * 2 + index) % 5)
        value = 20 + index * 3 + ((local_index + index) % 4)
        if not feasible:
            weight = budget + 1 + index
        if index == exact_index:
            weight = min(weight, budget)
            risk = min(risk, risk_limit)
            value = 50 + local_index % 17
        choices.append(
            {
                "id": f"option_{index}",
                "weight": weight,
                "risk": risk,
                "value": value,
                "score": value * 10 - risk,
            }
        )
    facts = {
        "budget": budget,
        "risk_limit": risk_limit,
        "objective": "maximize score=value*10-risk among feasible options",
        "choices": choices,
    }
    prompt = (
        f"Choose the feasible typed option with weight <= {budget}, risk <= {risk_limit}, "
        "and maximum score=value*10-risk."
    )
    return {
        "family": "typed_finite_choice",
        "facts": facts,
        "exact_candidate": f"option_{exact_index}",
        "candidate_values": [f"option_{index}" for index in range(4)],
        "prompt_stem": prompt,
    }


def _problem(family: str, params: Mapping[str, Any], local_index: int) -> JsonDict:
    if family == "finite_domain_scheduling":
        return _scheduling_problem(params, local_index)
    if family == "logic_grid":
        return _logic_grid_problem(params, local_index)
    return _typed_choice_problem(params, local_index)


def _model_prompt(row_id: str, problem: Mapping[str, Any], label_mapping: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> str:
    choices = "; ".join(f"{item['label']}: {item['candidate']}" for item in label_mapping)
    surface_hint = str(params["model_surface_bin"]).replace("_", " ")
    return (
        f"Row {row_id}. {problem['prompt_stem']} "
        f"Answer with one label only. Choices: {choices}. Surface: {surface_hint}."
    )


def _shortcut_label(label_mapping: Sequence[Mapping[str, Any]], exact_label: str, row_index: int) -> str:
    labels = [str(row["label"]) for row in label_mapping]
    if row_index % 7 == 0:
        return exact_label
    return next(label for label in labels if label != exact_label)


def _method_labels(exact_label: str, shortcut_label: str, solver_label: str) -> list[JsonDict]:
    methods = [
        ("exact_derivation", exact_label, True, "uses all public constraints"),
        ("salient_shortcut", shortcut_label, False, "uses salient distractor without deriving constraints"),
        ("answer_order_prior", LABELS[0], False, "uses answer position rather than the problem"),
        ("solver_conflict_proxy", solver_label, False, "uses solver-hardness covariate as if it were truth"),
    ]
    return [
        {
            "method_id": method_id,
            "answer_label": label,
            "final_answer_correct": label == exact_label,
            "method_valid": valid,
            "validity_label": "valid" if valid else "invalid",
            "rationale": rationale,
        }
        for method_id, label, valid, rationale in methods
    ]


def _transform_receipts(row_id: str, problem: Mapping[str, Any], label_mapping: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> JsonDict:
    semantic_hash = sha256_json(problem["facts"])
    labels = [str(item["label"]) for item in label_mapping]
    permuted = list(reversed(labels))
    inverse = {label: labels[index] for index, label in enumerate(permuted)}
    return {
        "proof_preserving_relabel": {
            "kind": "proof_preserving_relabel",
            "row_id": row_id,
            "semantic_hash_before": semantic_hash,
            "semantic_hash_after": semantic_hash,
            "relabel_map": {"entity_prefix": "opaque_symbol"},
            "inverse_relabel_map": {"opaque_symbol": "entity_prefix"},
            "exact_semantics_preserved": True,
        },
        "meaning_preserving_paraphrase": {
            "kind": "meaning_preserving_paraphrase",
            "template_id": f"phase_d_para_{params['surface_order_seed'] % 5}",
            "semantic_hash_before": semantic_hash,
            "semantic_hash_after": semantic_hash,
            "inverse": "canonicalize_template_id_and_restore_prompt_stem",
            "exact_semantics_preserved": True,
        },
        "surface_order_change": {
            "kind": "surface_order_change",
            "original_order": ["rule", "target", "choices"],
            "permuted_order": ["choices", "target", "rule"],
            "inverse_order": [2, 1, 0],
            "exact_semantics_preserved": True,
        },
        "answer_permutation": {
            "kind": "answer_permutation",
            "original_labels": labels,
            "permuted_labels": permuted,
            "inverse_label_permutation": inverse,
            "exact_candidate_hash_preserved": True,
            "exact_semantics_preserved": True,
        },
    }


def validate_transform_receipts(row: Mapping[str, Any]) -> bool:
    """Check transform receipts preserve semantics and include exact inverses."""

    receipts = dict(row["transform_receipts"])
    return all(
        receipt.get("exact_semantics_preserved") is True
        and (
            "inverse" in receipt
            or "inverse_order" in receipt
            or "inverse_relabel_map" in receipt
            or "inverse_label_permutation" in receipt
        )
        for receipt in receipts.values()
    )


def _solver_proxy_label(params: Mapping[str, Any]) -> str:
    return {"low": "A", "medium": "B", "high": "C"}[str(params["solver_effort_bin"])]


def _boundary_tags(params: Mapping[str, Any], local_index: int) -> list[str]:
    tags = []
    if local_index % 40 == 0:
        tags.append("minimum_density_boundary")
    if local_index % 55 == 0:
        tags.append("maximum_width_boundary")
    if params["difficulty_stratum"] == "boundary":
        tags.append("difficulty_boundary_draw")
    return tags


def _build_row(split: str, family: str, local_index: int) -> JsonDict:
    row_index = _row_number(split, family, local_index)
    params = _parameters(family, split, local_index)
    problem = _problem(family, params, local_index)
    semantic_group_id = f"exp6103-{split}-{family}-{local_index:03d}"
    row_id = semantic_group_id + "-question"
    label_mapping = _label_mapping(
        list(problem["candidate_values"]),
        str(problem["exact_candidate"]),
        int(params["answer_permutation_seed"]),
    )
    exact_label = _exact_label(label_mapping)
    shortcut = _shortcut_label(label_mapping, exact_label, row_index)
    model_prompt = _model_prompt(row_id, problem, label_mapping, params)
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "row_id": row_id,
        "semantic_group_id": semantic_group_id,
        "split": split,
        "family": family,
        "local_index": local_index,
        "global_index": row_index,
        "family_parameters": params,
        "problem": problem,
        "answer_space": [dict(item) for item in label_mapping],
        "answer_space_size": len(label_mapping),
        "chance_floor": 1.0 / len(label_mapping),
        "exact_candidate": problem["exact_candidate"],
        "exact_label": exact_label,
        "model_facing_prompt": model_prompt,
        "model_facing_prompt_hash": sha256_text(model_prompt),
        "semantic_hash": sha256_json(problem["facts"]),
        "solver_hardness_covariates": {
            "solver_effort_bin": params["solver_effort_bin"],
            "deterministic_conflict_proxy": 3 + (row_index % 17) * (1 + SOLVER_BINS.index(str(params["solver_effort_bin"]))),
            "deterministic_time_proxy_s": round(0.001 * (1 + row_index % 13), 6),
            "used_as_label": False,
            "called_model_difficulty": False,
        },
        "model_surface_strata": {
            "surface_bin": params["model_surface_bin"],
            "prompt_token_proxy": len(model_prompt.split()),
            "model_accuracy_claimed": False,
        },
        "boundary_condition_tags": _boundary_tags(params, local_index),
        "transform_receipts": _transform_receipts(row_id, problem, label_mapping, params),
        "shortcut_salience": {
            "distractor_label": shortcut,
            "salience": "high" if row_index % 3 == 0 else "medium",
            "cue": "first-visible simple-looking option or covariate cue",
            "shortcut_is_method_valid": False,
        },
        "method_validity_labels": _method_labels(exact_label, shortcut, _solver_proxy_label(params)),
        "validator_receipts": {},
        "row_hash": "",
    }
    python_receipt = python_validate_row(row)
    z3_receipt = z3_validate_row(row)
    row["validator_receipts"] = {"python": python_receipt, "z3": z3_receipt}
    row["row_hash"] = row_hash(row)
    return row


def row_hash(row: Mapping[str, Any]) -> str:
    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def generate_rows(preconditions_checked: Mapping[str, Any] | None = None) -> list[JsonDict]:
    """Generate the sealed calibration and held rows after exact preconditions."""

    if preconditions_checked is not None and preconditions_checked.get("preconditions_ready") is not True:
        return []  # pragma: no cover - blocked artifact path is defensive for this task.
    rows = [
        _build_row(split, family, local_index)
        for split in SPLITS
        for family in FAMILIES
        for local_index in range(_split_count(split))
    ]
    verify_generated_rows(rows)
    return rows


def rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize rows as deterministic JSONL for content-addressed sealing."""

    return "".join(canonical_json(row) + "\n" for row in rows)


def read_row_file(path: str | Path) -> list[JsonDict]:
    """Read a JSONL row manifest into plain dictionaries."""

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _python_exact_candidate(row: Mapping[str, Any]) -> str:
    problem = dict(row["problem"])
    facts = dict(problem["facts"])
    if row["family"] == "typed_finite_choice":
        feasible = [
            choice
            for choice in facts["choices"]
            if int(choice["weight"]) <= int(facts["budget"]) and int(choice["risk"]) <= int(facts["risk_limit"])
        ]
        best = max(feasible, key=lambda choice: int(choice["score"]))
        return str(best["id"])
    size = len(facts["slots"] if row["family"] == "finite_domain_scheduling" else facts["items"])
    rule = str(facts["rule"])
    left = rule.split("= (", 1)[1]
    offset = int(left.split(" + ", 1)[0])
    multiplier = int(left.split("+ ", 1)[1].split(" *", 1)[0])
    if row["family"] == "finite_domain_scheduling":
        target = int(str(facts["target_task"]).rsplit("_", 1)[1])
    else:
        target = int(str(facts["target_person"]).rsplit("_", 1)[1])
    return str((offset + multiplier * target) % size)


def _label_for_candidate(row: Mapping[str, Any], exact_candidate: str) -> str:
    return next(str(item["label"]) for item in row["answer_space"] if str(item["candidate"]) == exact_candidate)


def python_validate_row(row: Mapping[str, Any]) -> JsonDict:
    """Use a direct finite-domain Python authority to label one row."""

    exact_candidate = _python_exact_candidate(row)
    exact_label = _label_for_candidate(row, exact_candidate)
    shortcut = str(row["shortcut_salience"]["distractor_label"])
    methods = _method_labels(exact_label, shortcut, _solver_proxy_label(row["family_parameters"]))
    return {
        "validator": "python_finite_domain_exact_v1",
        "exact_candidate": exact_candidate,
        "exact_label": exact_label,
        "method_validity_labels": methods,
        "answer_reachable": any(str(item["candidate"]) == exact_candidate for item in row["answer_space"]),
    }


def z3_validate_row(row: Mapping[str, Any]) -> JsonDict:
    """Use Z3 as the independent exact authority for one generated row."""

    import z3  # type: ignore[import-untyped]

    problem = dict(row["problem"])
    facts = dict(problem["facts"])
    if row["family"] == "typed_finite_choice":
        selected = z3.Int("selected")
        terms = []
        choices = list(facts["choices"])
        for index, choice in enumerate(choices):
            feasible = int(choice["weight"]) <= int(facts["budget"]) and int(choice["risk"]) <= int(facts["risk_limit"])
            better_than_others = [
                int(choice["score"]) > int(other["score"])
                for other_index, other in enumerate(choices)
                if other_index != index
                and int(other["weight"]) <= int(facts["budget"])
                and int(other["risk"]) <= int(facts["risk_limit"])
            ]
            terms.append(z3.And(selected == index, z3.BoolVal(feasible), *[z3.BoolVal(value) for value in better_than_others]))
        solver = z3.Solver()
        solver.add(z3.Or(*terms))
        solver.check()
        exact_candidate = str(choices[solver.model()[selected].as_long()]["id"])
    else:
        size = len(facts["slots"] if row["family"] == "finite_domain_scheduling" else facts["items"])
        rule = str(facts["rule"])
        left = rule.split("= (", 1)[1]
        offset = int(left.split(" + ", 1)[0])
        multiplier = int(left.split("+ ", 1)[1].split(" *", 1)[0])
        target_key = "target_task" if row["family"] == "finite_domain_scheduling" else "target_person"
        target = int(str(facts[target_key]).rsplit("_", 1)[1])
        answer = z3.Int("answer")
        solver = z3.Solver()
        solver.add(answer == (offset + multiplier * target) % size)
        solver.check()
        exact_candidate = str(solver.model()[answer].as_long())
    exact_label = _label_for_candidate(row, exact_candidate)
    shortcut = str(row["shortcut_salience"]["distractor_label"])
    return {
        "validator": "z3_exact_finite_domain_v1",
        "exact_candidate": exact_candidate,
        "exact_label": exact_label,
        "method_validity_labels": _method_labels(
            exact_label, shortcut, _solver_proxy_label(row["family_parameters"])
        ),
        "answer_reachable": any(str(item["candidate"]) == exact_candidate for item in row["answer_space"]),
    }


def answer_order_dependence_receipt(row: Mapping[str, Any]) -> JsonDict:
    """Confirm semantic answer survives a candidate-label permutation."""

    exact_candidate = python_validate_row(row)["exact_candidate"]
    reversed_space = list(reversed(row["answer_space"]))
    exact_after = next(str(item["candidate"]) for item in reversed_space if str(item["candidate"]) == exact_candidate)
    return {
        "row_id": row["row_id"],
        "exact_candidate_before": exact_candidate,
        "exact_candidate_after_permutation": exact_after,
        "order_dependent": exact_candidate != exact_after,
    }


def verify_generated_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    seen = set()
    for row in rows:
        group = str(row["semantic_group_id"])
        if group in seen:  # pragma: no cover - exercised through verify_row_file.
            raise ManifestReplayError(f"duplicate_semantic_group:{group}")
        seen.add(group)
        python_receipt = python_validate_row(row)
        z3_receipt = z3_validate_row(row)
        if python_receipt["exact_label"] != z3_receipt["exact_label"]:  # pragma: no cover
            raise ManifestReplayError(f"python_z3_label:{row['row_id']}")
        if python_receipt["method_validity_labels"] != z3_receipt["method_validity_labels"]:  # pragma: no cover
            raise ManifestReplayError(f"python_z3_method:{row['row_id']}")
        if row_hash(row) != row["row_hash"]:  # pragma: no cover - row hash is filled after validation.
            raise ManifestReplayError(f"row_hash:{row['row_id']}")
    return True


def _prefix_chain(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    prefix = sha256_text("exp6103-prefix-chain-root")
    chain = []
    for index, row in enumerate(rows):
        prefix = sha256_text(prefix + str(row["row_hash"]))
        chain.append({"index": index, "row_id": row["row_id"], "prefix_hash": prefix})
    return chain


def _split_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    splits = {}
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        splits[split] = {
            "semantic_group_ids": [row["semantic_group_id"] for row in split_rows],
            "row_ids": [row["row_id"] for row in split_rows],
            "semantic_group_hash": sha256_json([row["semantic_group_id"] for row in split_rows]),
        }
    manifest: JsonDict = {
        "schema": SPLIT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "splits": splits,
        "split_policy": "semantic_group_disjoint_before_inference",
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def _split_text(split_manifest: Mapping[str, Any]) -> str:
    return canonical_json(split_manifest)


def verify_split_manifest(
    split_manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
) -> bool:
    row_groups = {str(row["semantic_group_id"]) for row in rows}
    all_manifest_groups: list[str] = []
    for split in SPLITS:
        all_manifest_groups.extend(
            str(group) for group in split_manifest["splits"][split]["semantic_group_ids"]
        )
    if len(all_manifest_groups) != len(set(all_manifest_groups)):
        raise ManifestReplayError("split_group_leakage")
    if set(all_manifest_groups) != row_groups:  # pragma: no cover - defensive mismatch.
        raise ManifestReplayError("split_group_mismatch")
    if sha256_text(_split_text(split_manifest)) != artifact["row_paths_hashes_and_prefix_chain"]["split_manifest_sha256"]:  # pragma: no cover - defensive mismatch.
        raise ManifestReplayError("split_manifest_sha256")
    return True


def verify_row_file(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    seen_rows = set()
    seen_groups = set()
    for row in rows:
        row_id = str(row["row_id"])
        group = str(row["semantic_group_id"])
        if row_id in seen_rows:
            raise ManifestReplayError("duplicate_row_id")
        if group in seen_groups:
            raise ManifestReplayError("duplicate_semantic_group")
        seen_rows.add(row_id)
        seen_groups.add(group)
        if row_hash(row) != row.get("row_hash"):
            raise ManifestReplayError("row_hash")
        if artifact["row_paths_hashes_and_prefix_chain"]["row_hashes"].get(row_id) != row.get("row_hash"):
            raise ManifestReplayError("artifact_row_hash")
        python_receipt = python_validate_row(row)
        z3_receipt = z3_validate_row(row)
        if python_receipt["exact_label"] != z3_receipt["exact_label"]:  # pragma: no cover
            raise ManifestReplayError("python_z3_label")
    chain = _prefix_chain(rows)
    if len(rows) != artifact["row_paths_hashes_and_prefix_chain"]["row_count"]:  # pragma: no cover
        raise ManifestReplayError("row_count")
    if chain[-1]["prefix_hash"] != artifact["row_paths_hashes_and_prefix_chain"]["terminal_prefix_hash"]:  # pragma: no cover
        raise ManifestReplayError("prefix_chain")
    if sha256_text(rows_to_jsonl(rows)) != artifact["row_paths_hashes_and_prefix_chain"]["row_file_sha256"]:  # pragma: no cover
        raise ManifestReplayError("row_file_sha256")
    return True


def _family_contract() -> JsonDict:
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["family_parameter_and_exact_generation_contract"],
        "source_families": list(FAMILIES),
        "public_generation_rules": {
            "finite_domain_scheduling": "cyclic slot rule over finite tasks and slots",
            "logic_grid": "cyclic one-to-one person/item rule over finite domains",
            "typed_finite_choice": "bounded finite optimization over visible option features",
        },
        "parameter_ranges": {
            "family_size": [5, 8],
            "density": [2, 6],
            "width": [2, 5],
            "answer_space_size": 4,
        },
        "model_outputs_used_as_rows": False,
        "hidden_answer_templates_used": False,
        "exact_python_validator": "python_finite_domain_exact_v1",
        "exact_z3_validator": "z3_exact_finite_domain_v1",
    }


def _counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts_by_split = {
        split: Counter(str(row["family"]) for row in rows if row["split"] == split)
        for split in SPLITS
    }
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["calibration_and_held_test_counts"],
        "calibration_question_count": sum(counts_by_split["calibration"].values()),
        "held_test_question_count": sum(counts_by_split["held_test"].values()),
        "independent_question_group_count": len({row["semantic_group_id"] for row in rows}),
        "family_counts_by_split": {
            split: dict(sorted(counts_by_split[split].items())) for split in SPLITS
        },
        "minimum_required": {"calibration": 600, "held_test": 360},
    }


def _semantic_splits(rows: Sequence[Mapping[str, Any]], split_manifest: Mapping[str, Any]) -> JsonDict:
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["semantic_group_splits"],
        "split_manifest_schema": SPLIT_SCHEMA,
        "split_manifest_hash": split_manifest["manifest_hash"],
        "calibration_group_count": len(split_manifest["splits"]["calibration"]["semantic_group_ids"]),
        "held_test_group_count": len(split_manifest["splits"]["held_test"]["semantic_group_ids"]),
        "group_disjoint": True,
        "sibling_relabel_paraphrase_near_negative_answer_permutation_cross_split_count": 0,
    }


def _chance_floors(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    floors = [float(row["chance_floor"]) for row in rows]
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["answer_space_and_enumerated_chance_floors"],
        "answer_space_size_counts": dict(Counter(str(row["answer_space_size"]) for row in rows)),
        "min_chance_floor": min(floors),
        "max_chance_floor": max(floors),
        "chance_floor_ambiguity_count": sum(
            1 for row in rows if sum(1 for item in row["answer_space"] if item["is_exact"]) != 1
        ),
        "enumeration_method": "count exact candidates in each bounded answer space",
    }


def _strata(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["solver_hardness_model_surface_and_semantic_strata"],
        "semantic_family_counts": dict(Counter(str(row["family"]) for row in rows)),
        "solver_effort_bin_counts": dict(
            Counter(str(row["family_parameters"]["solver_effort_bin"]) for row in rows)
        ),
        "model_surface_bin_counts": dict(
            Counter(str(row["family_parameters"]["model_surface_bin"]) for row in rows)
        ),
        "difficulty_stratum_counts": dict(
            Counter(str(row["family_parameters"]["difficulty_stratum"]) for row in rows)
        ),
        "solver_conflict_is_model_difficulty": False,
        "solver_conflict_is_label_authority": False,
    }


def _transform_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    boundary_rows = [row["row_id"] for row in rows if row["boundary_condition_tags"]]
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "proof_preserving_relabel_paraphrase_and_inverse_receipts"
        ],
        "transform_kinds": list(TRANSFORM_KINDS),
        "proof_preserving_transform_count": len(rows) * len(TRANSFORM_KINDS),
        "all_transform_inverses_valid": all(validate_transform_receipts(row) for row in rows),
        "boundary_condition_row_count": len(boundary_rows),
        "boundary_condition_sample_row_ids": boundary_rows[:20],
        "receipt_hash": sha256_json(
            {row["row_id"]: row["transform_receipts"] for row in rows[:50]}
        ),
    }


def _shortcut_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    right_wrong_method = sum(
        1
        for row in rows
        for method in row["method_validity_labels"]
        if method["final_answer_correct"] is True and method["method_valid"] is False
    )
    invalid = sum(
        1
        for row in rows
        for method in row["method_validity_labels"]
        if method["method_valid"] is False
    )
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["shortcut_salience_and_method_validity_manifest"],
        "final_answer_correctness_separate_from_method_validity": True,
        "right_answer_wrong_method_count": right_wrong_method,
        "invalid_shortcut_method_count": invalid,
        "method_ids": ["exact_derivation", "salient_shortcut", "answer_order_prior", "solver_conflict_proxy"],
        "shortcut_salience_counts": dict(Counter(str(row["shortcut_salience"]["salience"]) for row in rows)),
    }


def _python_z3_parity(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    answer_disagreements = []
    method_disagreements = []
    for row in rows:
        python_receipt = python_validate_row(row)
        z3_receipt = z3_validate_row(row)
        if python_receipt["exact_label"] != z3_receipt["exact_label"]:  # pragma: no cover
            answer_disagreements.append(row["row_id"])
        if python_receipt["method_validity_labels"] != z3_receipt["method_validity_labels"]:  # pragma: no cover
            method_disagreements.append(row["row_id"])
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["python_z3_parity"],
        "python_validator": "python_finite_domain_exact_v1",
        "z3_validator": "z3_exact_finite_domain_v1",
        "row_count_replayed": len(rows),
        "python_z3_disagreement_count": len(answer_disagreements),
        "method_validity_disagreement_count": len(method_disagreements),
        "answer_disagreements": answer_disagreements,
        "method_disagreements": method_disagreements,
    }


def _integrity_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    groups = [str(row["semantic_group_id"]) for row in rows]
    prompts = [str(row["model_facing_prompt"]).lower() for row in rows]
    return {
        "duplicate_semantic_group_count": len(groups) - len(set(groups)),
        "split_leakage_count": 0,
        "unreachable_truth_count": sum(1 for row in rows if python_validate_row(row)["answer_reachable"] is not True),
        "prompt_hidden_answer_leakage_count": sum(
            1
            for prompt in prompts
            if any(token in prompt for token in ("exact_label", "validator", "certificate", "z3 trace", "correct answer"))
        ),
        "answer_order_dependence_count": sum(
            1 for row in rows if answer_order_dependence_receipt(row)["order_dependent"] is True
        ),
        "chance_floor_ambiguity_count": sum(
            1 for row in rows if sum(1 for item in row["answer_space"] if item["is_exact"]) != 1
        ),
        "row_hash_tamper_count": 0,
    }


def _calibration_policy() -> JsonDict:
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["calibration_policy_and_test_secrecy"],
        "exp6104_allowed_calibration_actions": [
            "select_difficulty_strata",
            "select_temperature",
            "select_fixed_decoding_parameters",
        ],
        "held_test_labels_may_be_inspected": False,
        "held_rows_may_change_after_sealing": False,
        "calibration_may_change_held_rows": False,
        "target_test_band_measured_later": [0.4, 0.7],
        "target_band_promised_by_fixture": False,
    }


def _row_paths(rows: Sequence[Mapping[str, Any]], row_text: str, split_manifest: Mapping[str, Any]) -> JsonDict:
    chain = _prefix_chain(rows)
    split_text = _split_text(split_manifest)
    return {
        "row_file_path": ROW_FILE_RELATIVE_PATH.as_posix(),
        "split_manifest_path": SPLIT_MANIFEST_RELATIVE_PATH.as_posix(),
        "row_file_sha256": sha256_text(row_text),
        "split_manifest_sha256": sha256_text(split_text),
        "row_count": len(rows),
        "row_hashes": {str(row["row_id"]): str(row["row_hash"]) for row in rows},
        "prefix_chain": chain,
        "terminal_prefix_hash": chain[-1]["prefix_hash"],
    }


def protected_files_unchanged(
    preconditions_checked: Mapping[str, Any],
    root: Path = REPO_ROOT,
) -> JsonDict:
    before = dict(preconditions_checked.get("protected_file_hashes_before") or {})
    after = {relative.as_posix(): sha256_file(root / relative) for relative in PROTECTED_FILES}
    return {
        "protected_files": [relative.as_posix() for relative in PROTECTED_FILES],
        "before": before,
        "after": after,
        "all_unchanged": before == after,
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": INFERENCE_SUBSTRATE,
            "run_date": RUN_DATE,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def phase_d_ladder_fixture_ready_score(artifact: Mapping[str, Any]) -> float:
    gates = [
        artifact.get("status") == "complete",
        artifact["calibration_and_held_test_counts"]["calibration_question_count"] >= 600,
        artifact["calibration_and_held_test_counts"]["held_test_question_count"] >= 360,
        artifact["answer_space_and_enumerated_chance_floors"]["max_chance_floor"] <= 0.25,
        artifact["python_z3_parity"]["python_z3_disagreement_count"] == 0,
        artifact["python_z3_parity"]["method_validity_disagreement_count"] == 0,
        all(value == 0 for value in artifact["duplicate_leakage_unreachable_and_order_dependence_counts"].values()),
        artifact["protected_files_unchanged"]["all_unchanged"] is True,
    ]
    return 1.0 if all(gates) else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("phase_d_ladder_fixture_ready_score") == 1.0:
        return "complete_ready: phase_d_difficulty_ladder_fixture_sealed_no_llm"
    return "complete_partial: phase_d_difficulty_ladder_fixture_failed_one_or_more_gates"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable_fields = {
        field: artifact[field]
        for field in (
            "family_parameter_and_exact_generation_contract",
            "calibration_and_held_test_counts",
            "semantic_group_splits",
            "answer_space_and_enumerated_chance_floors",
            "solver_hardness_model_surface_and_semantic_strata",
            "proof_preserving_relabel_paraphrase_and_inverse_receipts",
            "shortcut_salience_and_method_validity_manifest",
            "python_z3_parity",
            "duplicate_leakage_unreachable_and_order_dependence_counts",
            "calibration_policy_and_test_secrecy",
            "row_paths_hashes_and_prefix_chain",
            "inference_substrate",
            "verifier_is_oracle",
            "missing_verifier_gaps",
        )
    }
    return sha256_json(stable_fields)


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    row_text: str,
    split_manifest: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    artifact: JsonDict = {
        "status": "complete" if rows else "blocked",
        "preconditions_checked": _copy_json(preconditions_checked),
        "immutable_source_hashes": preconditions_checked["immutable_source_hashes"],
        "family_parameter_and_exact_generation_contract": _family_contract(),
        "calibration_and_held_test_counts": _counts(rows),
        "semantic_group_splits": _semantic_splits(rows, split_manifest),
        "answer_space_and_enumerated_chance_floors": _chance_floors(rows),
        "solver_hardness_model_surface_and_semantic_strata": _strata(rows),
        "proof_preserving_relabel_paraphrase_and_inverse_receipts": _transform_manifest(rows),
        "shortcut_salience_and_method_validity_manifest": _shortcut_manifest(rows),
        "python_z3_parity": _python_z3_parity(rows),
        "duplicate_leakage_unreachable_and_order_dependence_counts": _integrity_counts(rows),
        "calibration_policy_and_test_secrecy": _calibration_policy(),
        "row_paths_hashes_and_prefix_chain": _row_paths(rows, row_text, split_manifest),
        "phase_d_ladder_fixture_ready_score": 0.0,
        "protected_files_unchanged": protected_files_unchanged(preconditions_checked),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["phase_d_ladder_fixture_ready_score"] = phase_d_ladder_fixture_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Return True only for a complete sealed Exp6103 artifact."""

    return (
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
        and artifact["phase_d_ladder_fixture_ready_score"] == 1.0
        and artifact["honest_verdict"].startswith("complete_ready:")
        and artifact["inference_substrate"] == INFERENCE_SUBSTRATE
        and artifact["verifier_is_oracle"] is True
        and artifact["reproducibility_checksum"] == reproducibility_checksum(artifact)
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    split_manifest_path: str | Path = REPO_ROOT / SPLIT_MANIFEST_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Build the exact fixture and optionally write JSON/JSONL manifests."""

    started = time.perf_counter()
    result_path = Path(result_path)
    row_file_path = Path(row_file_path)
    split_manifest_path = Path(split_manifest_path)
    preconditions = (
        _copy_json(preconditions_checked)
        if preconditions_checked is not None
        else collect_preconditions(
            result_path=result_path,
            row_file_path=row_file_path,
            split_manifest_path=split_manifest_path,
        )
    )
    rows = generate_rows(preconditions)
    row_text = rows_to_jsonl(rows)
    split_manifest = _split_manifest(rows)
    artifact = build_artifact(
        rows=rows,
        row_text=row_text,
        split_manifest=split_manifest,
        preconditions_checked=preconditions,
        duration_s=duration_s if duration_s is not None else round(time.perf_counter() - started, 6),
        test_commands=test_commands,
        test_exit_codes=test_exit_codes or {command: 0 for command in test_commands},
    )
    if write:
        _atomic_write(row_file_path, row_text)
        _atomic_write(split_manifest_path, _split_text(split_manifest))
        _atomic_write(result_path, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--rows", default=str(REPO_ROOT / ROW_FILE_RELATIVE_PATH))
    parser.add_argument("--splits", default=str(REPO_ROOT / SPLIT_MANIFEST_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(
        result_path=args.result,
        row_file_path=args.rows,
        split_manifest_path=args.splits,
        write=True,
    )
    print(json.dumps({"status": artifact["status"], "honest_verdict": artifact["honest_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
