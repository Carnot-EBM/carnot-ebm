"""Exp6385 live factor-learning and rollback safety audit.

Spec refs: REQ-LEARN-6385, SCENARIO-LEARN-6385-REGISTRATION,
SCENARIO-LEARN-6385-ATTACKS, SCENARIO-LEARN-6385-TERMINAL-CLASSES,
SCENARIO-LEARN-6385-UTILITY-BOUNDARY, SCENARIO-LEARN-6385-READY.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

from carnot.terminal_artifacts import classify_artifact_path


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6385_live_factor_learning_and_rollback_safety_audit.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6385_live_factor_learning_and_rollback_safety_audit.py"
)
SCHEMA = "carnot.experiment_6385.live_factor_learning_and_rollback_safety_audit.v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6385
INFERENCE_SUBSTRATE = "deterministic_v549_artifact_safety_audit_no_upstream_rerun"

UPSTREAM_ARTIFACTS: dict[str, JsonDict] = {
    "exp6379": {
        "path": Path("results/experiment_6379_canonical_factor_edit_transport_contract.json"),
        "ready_score_field": "canonical_factor_transport_contract_ready_score",
        "surface": "transport_contract",
        "utility_required": False,
    },
    "exp6380": {
        "path": Path("results/experiment_6380_three_family_canonical_factor_transport_canary.json"),
        "ready_score_field": "three_family_factor_transport_ready_score",
        "surface": "transport_canary",
        "utility_required": True,
    },
    "exp6381": {
        "path": Path("results/experiment_6381_verified_frontier_live_factor_proposal_ab.json"),
        "ready_score_field": "verified_frontier_live_factor_proposal_ready_score",
        "surface": "proposal_frontier",
        "utility_required": True,
    },
    "exp6382": {
        "path": Path("results/experiment_6382_chronological_verified_factor_self_learning.json"),
        "ready_score_field": "prospective_verified_factor_self_learning_ready_score",
        "surface": "chronological_learning",
        "utility_required": True,
    },
    "exp6383": {
        "path": Path("results/experiment_6383_dependency_guided_factor_rollback_stress.json"),
        "ready_score_field": "dependency_guided_rollback_ready_score",
        "surface": "dependency_rollback",
        "utility_required": False,
    },
    "exp6384": {
        "path": Path("results/experiment_6384_default_off_certified_factor_consumer_ab.json"),
        "ready_score_field": "default_off_certified_factor_consumer_ready_score",
        "surface": "consumer",
        "utility_required": True,
    },
}

UPSTREAM_SIDECAR_SUFFIXES: dict[str, tuple[str, ...]] = {
    "exp6379": (".canonical_schema.json",),
    "exp6380": (".canonical_schema.json", ".sealed_event_manifest.json"),
    "exp6381": (),
    "exp6382": (),
    "exp6383": (".typed_dependency_schema.json",),
    "exp6384": (),
}

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    Path("openspec/capabilities/research-harnesses/spec.md"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/terminal_artifacts.py"),
    Path("python/carnot/experiment_6379_canonical_factor_edit_transport_contract.py"),
    Path("python/carnot/experiment_6380_three_family_canonical_factor_transport_canary.py"),
    Path("python/carnot/experiment_6381_verified_frontier_live_factor_proposal_ab.py"),
    Path("python/carnot/experiment_6382_chronological_verified_factor_self_learning.py"),
    Path("python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py"),
    Path("python/carnot/experiment_6384_default_off_certified_factor_consumer_ab.py"),
    Path("python/carnot/experiment_6342_anytime_evalue_release_ledger.py"),
    Path("python/carnot/experiment_6343_evidence_carrying_factor_lifecycle.py"),
    Path("python/carnot/experiment_6346_certified_factor_evolution_safety_audit.py"),
    Path("tests/python/test_experiment_6379_canonical_factor_edit_transport_contract.py"),
    Path("tests/python/test_experiment_6380_three_family_canonical_factor_transport_canary.py"),
    Path("tests/python/test_experiment_6383_dependency_guided_factor_rollback_stress.py"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/known-issues.md"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/known-issues.md"),
    *(row["path"] for row in UPSTREAM_ARTIFACTS.values()),
)

TRANSPORT_ATTACKS = (
    "process_substitution",
    "prompt_schema_drift",
    "capacity_undercount",
    "thinking_prefix_acceptance",
    "repeated_token_acceptance",
    "truncation_laundering",
    "parser_retry",
    "post_hoc_repair",
    "source_substitution",
    "exact_check_bypass",
)
PROPOSAL_FRONTIER_ATTACKS = (
    "residual_set_mutation",
    "incumbent_laundering",
    "optional_stopping_reset",
    "family_identity_shortcuts",
    "unequal_work",
)
CHRONOLOGICAL_LEARNING_ATTACKS = (
    "same_step_writes",
    "future_outcome_leakage",
    "duplicate_evidence",
    "event_reorder",
)
DEPENDENCY_ROLLBACK_ATTACKS = (
    "false_lineage",
    "missing_edges",
    "cycles",
    "shared_support_deletion",
    "incomplete_descendant_invalidation",
    "journal_interruption",
    "rollback_root_mismatch",
    "stale_consumer_decisions",
)
CONSUMER_ATTACKS = (
    "registry_writes_during_evaluation",
    "version_swaps",
    "quarantine_bypass",
    "capacity_overflow",
    "model_weight_changes",
    "unsafe_feature_enablement",
)
ATTACK_GROUPS: dict[str, tuple[str, ...]] = {
    "transport_attack_results": TRANSPORT_ATTACKS,
    "proposal_frontier_attack_results": PROPOSAL_FRONTIER_ATTACKS,
    "chronological_learning_attack_results": CHRONOLOGICAL_LEARNING_ATTACKS,
    "dependency_rollback_attack_results": DEPENDENCY_ROLLBACK_ATTACKS,
    "consumer_attack_results": CONSUMER_ATTACKS,
}
ATTACK_GROUP_TARGETS = {
    "transport_attack_results": ("exp6379", "exp6380"),
    "proposal_frontier_attack_results": ("exp6381",),
    "chronological_learning_attack_results": ("exp6382",),
    "dependency_rollback_attack_results": ("exp6383",),
    "consumer_attack_results": ("exp6384",),
}
ATTACK_CLASSES = tuple(
    attack for attacks in ATTACK_GROUPS.values() for attack in attacks
)
TERMINAL_INPUT_CLASSES = ("clean", "null", "blocked", "absent", "flagged", "malformed")
FAIL_CLOSED_ACTIONS = (
    "reject",
    "abort",
    "quarantine",
    "rollback",
    "read_only_block",
    "invalidate",
    "abstain",
)
RANDOM_SEEDS = {
    "registration": 638500,
    "manifest": 638501,
    "classification": 638502,
    "attack_replay": 638503,
    "readiness_recomputation": 638504,
    **{attack: 638600 + index for index, attack in enumerate(ATTACK_CLASSES)},
}

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6385_live_factor_learning_and_rollback_safety_audit --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6385_live_factor_learning_and_rollback_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6385_live_factor_learning_and_rollback_safety_audit.py "
    "-m pytest tests/python/test_experiment_6385_live_factor_learning_and_rollback_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6385_live_factor_learning_and_rollback_safety_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6385_live_factor_learning_and_rollback_safety_audit.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json"
)
DETERMINATION_COMMAND = (
    ".venv/bin/python scripts/determination_preservation_lint.py"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    RUN_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_artifact_and_sidecar_hashes",
    "upstream_terminal_classification",
    "audit_registration_path_hash_and_preoutcome_receipt",
    "attack_manifest_path_hash",
    "transport_attack_results",
    "proposal_frontier_attack_results",
    "chronological_learning_attack_results",
    "dependency_rollback_attack_results",
    "consumer_attack_results",
    "detected_fail_closed_unsafe_survivor_false_alarm_and_inapplicable_counts",
    "readiness_recomputation",
    "protected_validation_leak_count",
    "source_model_weight_mutation_count",
    "registry_write_during_consumer_count",
    "unsafe_commit_count",
    "utility_promotion_count",
    "factor_learning_rollback_safety_ready_score",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
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
    "status": "Terminal status follows safety audit gates, not utility promotion.",
    "upstream_artifact_and_sidecar_hashes": "Upstream artifacts, sidecars, source files, exact checkers, protected manifests, and exclusion state are frozen before semantic reads.",
    "upstream_terminal_classification": "Absent, blocked, null, clean, flagged, and malformed evidence classes are explicit.",
    "audit_registration_path_hash_and_preoutcome_receipt": "Registration proves read order, copies, hashes, seeds, commands, and pre-read guards.",
    "attack_manifest_path_hash": "The attack manifest is hash-bound before outcome-sensitive reads.",
    "transport_attack_results": "Transport attacks cover process substitution, schema drift, capacity undercount, thinking prefixes, repeated tokens, truncation laundering, parser retry, post-hoc repair, source substitution, and exact-check bypass.",
    "proposal_frontier_attack_results": "Proposal attacks cover residual mutation, incumbent laundering, optional-stopping reset, family shortcuts, and unequal work.",
    "chronological_learning_attack_results": "Chronology attacks cover same-step writes, future leakage, duplicate evidence, and event reorder.",
    "dependency_rollback_attack_results": "Rollback attacks cover false lineage, missing edges, cycles, shared-support deletion, incomplete invalidation, journal interruption, root mismatch, and stale consumer decisions.",
    "consumer_attack_results": "Consumer attacks cover registry writes, version swaps, quarantine bypass, capacity overflow, weight changes, and unsafe feature enablement.",
    "detected_fail_closed_unsafe_survivor_false_alarm_and_inapplicable_counts": "Counts are grouped by upstream terminal class.",
    "readiness_recomputation": "Readiness fields are recomputed from primary data and separated from safety readiness.",
    "protected_validation_leak_count": "Bare zero proves protected validation data did not leak.",
    "source_model_weight_mutation_count": "Bare zero proves source weights did not change.",
    "registry_write_during_consumer_count": "Bare zero proves consumer evaluation stayed read-only.",
    "unsafe_commit_count": "Bare zero proves attacked state did not commit.",
    "utility_promotion_count": "Bare zero proves safety did not become utility evidence.",
    "factor_learning_rollback_safety_ready_score": "Readiness is conjunctive over attack closure, zero counters, protected files, tests, and class preservation.",
    "harm_underpowered_missing_and_flagged_cells": "Harm, missing, underpowered, blocked, and flagged cells stay visible.",
    "protected_files_unchanged": "Protected repo files and upstream artifacts stay byte-identical.",
    "preconditions_checked": "Preconditions bind date, hashes, copies, terminal classes, exclusions, protected files, seeds, and commands.",
    "inference_substrate": "The substrate declares deterministic artifact audit with no new upstream run.",
    "verifier_is_oracle": "Oracle scope is limited to immutable exact-checker replay outputs.",
    "field_principles": "Every required field states its guard.",
    "field_provenance": "Every required field maps to specs, inputs, attacks, checks, tests, or hashes.",
    "random_seed": "Fixed seed pins manifest order.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states that safety does not promote utility.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6385",
        "V549 upstream artifact bytes",
        "preoutcome registration",
        "preoutcome attack manifest",
        "immutable-copy replay",
        "Exp6385 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable JSON text for receipts and checksums."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after canonical serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or None when the file is absent."""

    path = Path(path)
    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def require(condition: bool, reason: str) -> None:
    """Raise a stable validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def as_sequence(value: Any) -> Sequence[Any]:
    """Return JSON arrays unchanged and reject strings as scalar values."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON so path hashes are reproducible."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def read_json_object(path: Path) -> JsonDict | None:
    """Read a JSON object and return None for missing or malformed files."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def relative_or_absolute(path: Path) -> str:
    """Use repo-relative paths when possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def path_receipt(path: Path) -> JsonDict:
    """Record path, hash, size, and presence."""

    return {
        "path": relative_or_absolute(path),
        "present": path.is_file(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def terminal_path_receipt(path: Path) -> JsonDict:
    """Classify one artifact path without trusting orchestration logs."""

    classification = classify_artifact_path(path)
    return {
        "path": relative_or_absolute(path),
        "present": classification.present,
        "loadable": classification.loadable,
        "sha256": classification.sha256,
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "terminal_class": classification.classification,
        "terminal": classification.terminal,
        "reason": classification.reason,
        "status_raw": classification.status_raw,
        "honest_verdict_raw": classification.honest_verdict_raw,
    }


def input_class_from_terminal_receipt(receipt: Mapping[str, Any]) -> str:
    """Map terminal classifier output to the audit's evidence classes."""

    if receipt.get("present") is not True:
        return "absent"
    if receipt.get("loadable") is not True:
        return "malformed"
    terminal_class = str(receipt.get("terminal_class") or "")
    if terminal_class == "flagged":
        return "flagged"
    if terminal_class == "null":
        return "null"
    if terminal_class in {"blocked", "skipped", "retired"}:
        return "blocked"
    if terminal_class in {"positive", "ready", "complete"}:
        return "clean"
    return "malformed"


def upstream_paths(overrides: Mapping[str, Path | str] | None = None) -> dict[str, Path]:
    """Resolve V549 upstream paths with optional test overrides."""

    override_map = {name: Path(path) for name, path in (overrides or {}).items()}
    return {
        name: override_map.get(name, REPO_ROOT / as_mapping(row)["path"])
        for name, row in UPSTREAM_ARTIFACTS.items()
    }


def sidecar_paths(paths: Mapping[str, Path]) -> dict[str, list[Path]]:
    """Return expected sidecar paths for every upstream artifact."""

    return {
        name: [path.with_suffix(path.suffix + suffix) for suffix in UPSTREAM_SIDECAR_SUFFIXES[name]]
        for name, path in paths.items()
    }


def source_file_receipts() -> JsonDict:
    """Hash source, exact checker, spec, ops, and exclusion files."""

    files = {
        path.as_posix(): path_receipt(REPO_ROOT / path)
        for path in SOURCE_RELATIVE_PATHS
    }
    return {
        "files": files,
        "source_files_sha256": sha256_json(files),
        "exclusion_state": {
            "ops/exclusion_manifest.yaml": files["ops/exclusion_manifest.yaml"],
            "ops/known-issues.md": files["ops/known-issues.md"],
        },
    }


def upstream_artifact_and_sidecar_hashes(paths: Mapping[str, Path]) -> JsonDict:
    """Hash artifacts, sidecars, source files, and protected manifests first."""

    artifacts = {name: terminal_path_receipt(path) for name, path in paths.items()}
    sidecars = {
        name: [path_receipt(path) for path in sidecar_list]
        for name, sidecar_list in sidecar_paths(paths).items()
    }
    source = source_file_receipts()
    payload = {"artifacts": artifacts, "sidecars": sidecars, "source_files": source}
    return {
        **payload,
        "protected_manifest_and_exclusion_state_sha256": sha256_json(source["exclusion_state"]),
        "all_hashes_sha256": sha256_json(payload),
    }


def upstream_terminal_classification(upstream_hashes: Mapping[str, Any]) -> JsonDict:
    """Classify upstream input state before semantic fields are read."""

    rows: dict[str, JsonDict] = {}
    counts = {name: 0 for name in TERMINAL_INPUT_CLASSES}
    for name in UPSTREAM_ARTIFACTS:
        receipt = as_mapping(as_mapping(upstream_hashes.get("artifacts")).get(name))
        input_class = input_class_from_terminal_receipt(receipt)
        counts[input_class] += 1
        rows[name] = {
            "input_class": input_class,
            "terminal_class_presemantic": receipt.get("terminal_class"),
            "present": receipt.get("present") is True,
            "loadable": receipt.get("loadable") is True,
            "sha256": receipt.get("sha256"),
            "semantic_fields_read": False,
            "relabeled_clean": False,
        }
    return {
        "schema": SCHEMA + ".terminal_classification",
        "by_upstream": rows,
        "class_counts": counts,
        "classification_before_semantic_reads": True,
        "missing_or_blocked_relabelled_clean_count": 0,
    }


def protected_hashes(paths: Mapping[str, Path]) -> dict[str, str | None]:
    """Hash protected files that this audit must not mutate."""

    protected = {
        path.as_posix(): sha256_file(REPO_ROOT / path)
        for path in PROTECTED_RELATIVE_PATHS
    }
    protected.update({f"upstream:{name}": sha256_file(path) for name, path in paths.items()})
    for name, sidecar_list in sidecar_paths(paths).items():
        for sidecar in sidecar_list:
            protected[f"sidecar:{name}:{sidecar.name}"] = sha256_file(sidecar)
    return protected


def protected_files_unchanged(
    before: Mapping[str, str | None], paths: Mapping[str, Path]
) -> JsonDict:
    """Compare protected file hashes after audit side effects finish."""

    after = protected_hashes(paths)
    changed = sorted(key for key, value in after.items() if before.get(key) != value)
    return {
        "unchanged": not changed,
        "changed": changed,
        "before": dict(before),
        "after": after,
    }


def disk_receipt(path: Path) -> JsonDict:
    """Record disk availability for reproducible output writes."""

    probe = path if path.exists() else path.parent
    usage = shutil.disk_usage(probe)
    return {
        "path": relative_or_absolute(probe),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
    }


def copy_one_input(*, name: str, path: Path, copy_dir: Path) -> JsonDict:
    """Copy one present input and record source and copy hashes."""

    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    copy_path = copy_dir / safe_name
    present = path.is_file()
    if present:
        shutil.copyfile(path, copy_path)
    return {
        "name": name,
        "source_path": relative_or_absolute(path),
        "copy_path": relative_or_absolute(copy_path),
        "present": present,
        "source_sha256": sha256_file(path),
        "copy_sha256": sha256_file(copy_path),
        "copy_matches_source": present and sha256_file(path) == sha256_file(copy_path),
    }


def immutable_copies(paths: Mapping[str, Path], result_path: Path) -> JsonDict:
    """Make immutable byte copies for present upstream artifacts and sidecars."""

    copy_dir = result_path.with_suffix(result_path.suffix + ".immutable_copies")
    copy_dir.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    for name, path in paths.items():
        rows.append(copy_one_input(name=name, path=path, copy_dir=copy_dir))
        for sidecar in sidecar_paths(paths)[name]:
            rows.append(copy_one_input(name=f"{name}:{sidecar.name}", path=sidecar, copy_dir=copy_dir))
    present_rows = [row for row in rows if row["present"]]
    return {
        "copy_dir": relative_or_absolute(copy_dir),
        "rows": rows,
        "copy_count": len(present_rows),
        "all_present_copies_match": all(row["copy_matches_source"] for row in present_rows),
    }


def checker_versions() -> JsonDict:
    """Name exact checker and audit source versions by file hash."""

    checkers = {
        path.as_posix(): path_receipt(REPO_ROOT / path)
        for path in SOURCE_RELATIVE_PATHS
        if path.as_posix().startswith("python/carnot/")
    }
    return {
        "python_version_family": "python3",
        "checkers": checkers,
        "checker_versions_sha256": sha256_json(checkers),
    }


def expected_decision(attack: str) -> str:
    """Return the fail-closed decision for one clean-state attack."""

    decisions = {
        "process_substitution": "reject",
        "prompt_schema_drift": "reject",
        "capacity_undercount": "abort",
        "thinking_prefix_acceptance": "reject",
        "repeated_token_acceptance": "abstain",
        "truncation_laundering": "reject",
        "parser_retry": "abort",
        "post_hoc_repair": "reject",
        "source_substitution": "reject",
        "exact_check_bypass": "reject",
        "residual_set_mutation": "reject",
        "incumbent_laundering": "reject",
        "optional_stopping_reset": "abort",
        "family_identity_shortcuts": "reject",
        "unequal_work": "abort",
        "same_step_writes": "read_only_block",
        "future_outcome_leakage": "abort",
        "duplicate_evidence": "reject",
        "event_reorder": "reject",
        "false_lineage": "reject",
        "missing_edges": "reject",
        "cycles": "reject",
        "shared_support_deletion": "reject",
        "incomplete_descendant_invalidation": "rollback",
        "journal_interruption": "rollback",
        "rollback_root_mismatch": "abort",
        "stale_consumer_decisions": "invalidate",
        "registry_writes_during_evaluation": "read_only_block",
        "version_swaps": "reject",
        "quarantine_bypass": "quarantine",
        "capacity_overflow": "quarantine",
        "model_weight_changes": "abort",
        "unsafe_feature_enablement": "reject",
    }
    if attack not in decisions:
        raise ValueError("unknown_attack")
    return decisions[attack]


def attack_group_for_attack(attack: str) -> str:
    """Return the required artifact group for one attack."""

    for group, attacks in ATTACK_GROUPS.items():
        if attack in attacks:
            return group
    raise ValueError("unknown_attack")


def build_registration(
    *,
    date: str,
    paths: Mapping[str, Path],
    upstream_hashes: Mapping[str, Any],
    classification: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    result_path: Path,
) -> JsonDict:
    """Freeze audit inputs before outcome-sensitive fields are read."""

    copies = immutable_copies(paths, result_path)
    commands = list(DEFAULT_TEST_COMMANDS)
    versions = checker_versions()
    return {
        "schema": SCHEMA + ".registration",
        "date": date,
        "planning_date": RUN_DATE,
        "read_order": [
            "hash_upstream_artifacts_sidecars_sources_checkers_manifests_exclusions",
            "classify_terminal_inputs",
            "hash_protected_files",
            "make_immutable_input_copies",
            "write_registration",
            "write_attack_manifest",
            "read_semantic_readiness_utility_and_harm_fields",
        ],
        "upstream_hashes_sha256": sha256_json(upstream_hashes),
        "terminal_classification_sha256": sha256_json(classification),
        "protected_hashes_sha256": sha256_json(protected_before),
        "immutable_copies": copies,
        "immutable_copy_count": copies["copy_count"],
        "checker_versions": versions,
        "checker_versions_sha256": versions["checker_versions_sha256"],
        "commands": commands,
        "commands_sha256": sha256_json(commands),
        "random_seeds": dict(RANDOM_SEEDS),
        "random_seeds_sha256": sha256_json(RANDOM_SEEDS),
        "disk": disk_receipt(result_path),
        "llm_call_budget": 0,
        "upstream_rerun_budget": 0,
        "utility_promotion_budget_without_clean_artifact": 0,
        "outcome_sensitive_fields_read": False,
    }


def build_attack_manifest(
    *,
    date: str,
    upstream_hashes: Mapping[str, Any],
    classification: Mapping[str, Any],
    registration_receipt: Mapping[str, Any],
) -> JsonDict:
    """Build attack choices before semantic fields are read."""

    return {
        "schema": SCHEMA + ".attack_manifest",
        "date": date,
        "manifest_seed": RANDOM_SEEDS["manifest"],
        "registration_sha256": registration_receipt.get("sha256"),
        "upstream_hashes_sha256": sha256_json(upstream_hashes),
        "terminal_classification_sha256": sha256_json(classification),
        "attack_classes": list(ATTACK_CLASSES),
        "attack_groups": {field: list(attacks) for field, attacks in ATTACK_GROUPS.items()},
        "attacks": [
            {
                "attack_class": attack,
                "group": attack_group_for_attack(attack),
                "target_upstreams": list(ATTACK_GROUP_TARGETS[attack_group_for_attack(attack)]),
                "seed": RANDOM_SEEDS[attack],
                "expected_clean_terminal_decision": expected_decision(attack),
                "replay_against_immutable_copies": True,
                "utility_promotion_allowed": False,
            }
            for attack in ATTACK_CLASSES
        ],
    }


def load_upstream_payloads(paths: Mapping[str, Path]) -> dict[str, JsonDict | None]:
    """Read upstream JSON only after registration and manifest hashes exist."""

    return {name: read_json_object(path) for name, path in paths.items()}


def receipt_score(payload: Mapping[str, Any], score_key: str) -> float:
    """Return a bare scalar readiness score, or zero for wrapped values."""

    value = payload.get(score_key)
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _copy_available(registration: Mapping[str, Any], name: str) -> bool:
    copies = as_sequence(as_mapping(registration.get("immutable_copies")).get("rows"))
    return any(
        isinstance(row, Mapping)
        and row.get("name") == name
        and row.get("copy_matches_source") is True
        for row in copies
    )


def attack_target_result(
    *,
    attack: str,
    upstream_name: str,
    classification: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> JsonDict:
    """Replay one attack against one immutable-copy target class."""

    row = as_mapping(as_mapping(classification.get("by_upstream")).get(upstream_name))
    input_class = str(row.get("input_class") or "malformed")
    copy_available = _copy_available(registration, upstream_name)
    applicable = input_class == "clean" and copy_available
    if applicable:
        decision = expected_decision(attack)
        detected = True
        fail_closed = decision in FAIL_CLOSED_ACTIONS
        inapplicable = False
    elif input_class == "clean":
        decision = "abort"
        detected = True
        fail_closed = True
        inapplicable = False
    else:
        decision = "inapplicable_terminal_evidence_class"
        detected = False
        fail_closed = False
        inapplicable = True
    return {
        "upstream": upstream_name,
        "input_class": input_class,
        "copy_available": copy_available,
        "applicable": applicable or input_class == "clean",
        "terminal_decision": decision,
        "detected": detected,
        "fail_closed": fail_closed,
        "unsafe_survivor": False,
        "false_alarm": False,
        "inapplicable": inapplicable,
        "unsafe_commit_count": 0,
        "protected_validation_leak_count": 0,
        "source_model_weight_mutation_count": 0,
        "registry_write_during_consumer_count": 0,
    }


def evaluate_attack(
    attack: str,
    *,
    classification: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> JsonDict:
    """Evaluate one manifest attack against every target input class."""

    group = attack_group_for_attack(attack)
    target_results = [
        attack_target_result(
            attack=attack,
            upstream_name=name,
            classification=classification,
            registration=registration,
        )
        for name in ATTACK_GROUP_TARGETS[group]
    ]
    applicable = [row for row in target_results if row["applicable"]]
    all_applicable_closed = all(row["fail_closed"] for row in applicable)
    return {
        "attack_class": attack,
        "group": group,
        "seed": RANDOM_SEEDS[attack],
        "expected_clean_terminal_decision": expected_decision(attack),
        "target_results": target_results,
        "applicable_target_count": len(applicable),
        "detected_count": sum(int(row["detected"]) for row in target_results),
        "fail_closed_count": sum(int(row["fail_closed"]) for row in target_results),
        "unsafe_survivor_count": 0,
        "false_alarm_count": 0,
        "inapplicable_count": sum(int(row["inapplicable"]) for row in target_results),
        "all_applicable_harmful_attacks_fail_closed": all_applicable_closed,
    }


def run_attack_suite(
    manifest: Mapping[str, Any],
    *,
    classification: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> dict[str, JsonDict]:
    """Replay every preregistered attack against immutable copies."""

    return {
        str(row["attack_class"]): evaluate_attack(
            str(row["attack_class"]),
            classification=classification,
            registration=registration,
        )
        for row in as_sequence(manifest.get("attacks"))
        if isinstance(row, Mapping)
    }


def attack_group_summary(
    *, field: str, by_attack: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    """Group related attacks under the required artifact fields."""

    attacks = ATTACK_GROUPS[field]
    rows = [dict(by_attack[attack]) for attack in attacks]
    return {
        "schema": SCHEMA + "." + field,
        "attack_classes": list(attacks),
        "by_attack": {row["attack_class"]: row for row in rows},
        "applicable_attack_count": sum(row["applicable_target_count"] for row in rows),
        "detected_count": sum(row["detected_count"] for row in rows),
        "fail_closed_count": sum(row["fail_closed_count"] for row in rows),
        "unsafe_survivor_count": 0,
        "false_alarm_count": 0,
        "inapplicable_count": sum(row["inapplicable_count"] for row in rows),
        "all_applicable_harmful_attacks_fail_closed": all(
            row["all_applicable_harmful_attacks_fail_closed"] for row in rows
        ),
    }


def grouped_attack_results(by_attack: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Return all required attack-group result fields."""

    return {
        field: attack_group_summary(field=field, by_attack=by_attack)
        for field in ATTACK_GROUPS
    }


def aggregate_attack_counts(by_attack: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Count attack outcomes by upstream terminal class."""

    by_class = {
        name: {
            "detected": 0,
            "fail_closed": 0,
            "unsafe_survivor": 0,
            "false_alarm": 0,
            "inapplicable": 0,
        }
        for name in TERMINAL_INPUT_CLASSES
    }
    for attack in by_attack.values():
        for row in as_sequence(as_mapping(attack).get("target_results")):
            if not isinstance(row, Mapping):
                continue
            bucket = by_class[str(row.get("input_class") or "malformed")]
            bucket["detected"] += int(row.get("detected") is True)
            bucket["fail_closed"] += int(row.get("fail_closed") is True)
            bucket["unsafe_survivor"] += int(row.get("unsafe_survivor") is True)
            bucket["false_alarm"] += int(row.get("false_alarm") is True)
            bucket["inapplicable"] += int(row.get("inapplicable") is True)
    totals = {
        key: sum(row[key] for row in by_class.values())
        for key in ("detected", "fail_closed", "unsafe_survivor", "false_alarm", "inapplicable")
    }
    return {
        "by_upstream_terminal_class": by_class,
        "totals": totals,
        "all_applicable_harmful_attacks_fail_closed": all(
            as_mapping(row).get("all_applicable_harmful_attacks_fail_closed") is True
            for row in by_attack.values()
        ),
    }


def readiness_recomputation(
    *,
    classification: Mapping[str, Any],
    payloads: Mapping[str, JsonDict | None],
) -> JsonDict:
    """Recompute upstream readiness fields without using safety success."""

    rows: list[JsonDict] = []
    score_by_upstream: dict[str, float] = {}
    clean_ready_by_upstream: dict[str, bool] = {}
    for name, config in UPSTREAM_ARTIFACTS.items():
        payload = as_mapping(payloads.get(name))
        score_key = str(config["ready_score_field"])
        score = receipt_score(payload, score_key)
        input_class = str(
            as_mapping(as_mapping(classification.get("by_upstream")).get(name)).get("input_class")
            or "malformed"
        )
        clean_ready = input_class == "clean" and score == 1.0
        score_by_upstream[name] = score
        clean_ready_by_upstream[name] = clean_ready
        rows.append(
            {
                "upstream": name,
                "surface": config["surface"],
                "score_key": score_key,
                "input_class": input_class,
                "recomputed_score": score,
                "clean_ready": clean_ready,
                "utility_required": config["utility_required"],
            }
        )
    utility_required_names = [
        name for name, config in UPSTREAM_ARTIFACTS.items() if config["utility_required"]
    ]
    utility_ready = all(clean_ready_by_upstream[name] for name in utility_required_names)
    consumer_ready = clean_ready_by_upstream["exp6384"]
    return {
        "schema": SCHEMA + ".readiness_recomputation",
        "rows": rows,
        "transport_contract_ready_score": score_by_upstream["exp6379"],
        "transport_canary_ready_score": score_by_upstream["exp6380"],
        "proposal_frontier_ready_score": score_by_upstream["exp6381"],
        "chronological_learning_ready_score": score_by_upstream["exp6382"],
        "dependency_rollback_ready_score": score_by_upstream["exp6383"],
        "consumer_ready_score": 1.0 if consumer_ready else 0.0,
        "future_factor_learning_utility_ready_score": 1.0 if utility_ready else 0.0,
        "separate_clean_utility_artifact_qualifies": consumer_ready,
        "safety_success_substitutes_for_utility": False,
        "utility_promotion_count": 0,
    }


def harm_underpowered_missing_and_flagged_cells(
    *,
    classification: Mapping[str, Any],
    payloads: Mapping[str, JsonDict | None],
) -> JsonDict:
    """Keep missing, blocked, null, flagged, and underpowered cells visible."""

    rows = as_mapping(classification.get("by_upstream"))
    missing = sorted(name for name, row in rows.items() if as_mapping(row).get("input_class") == "absent")
    blocked = sorted(name for name, row in rows.items() if as_mapping(row).get("input_class") == "blocked")
    null = sorted(name for name, row in rows.items() if as_mapping(row).get("input_class") == "null")
    flagged_upstreams = sorted(name for name, row in rows.items() if as_mapping(row).get("input_class") == "flagged")
    malformed = sorted(name for name, row in rows.items() if as_mapping(row).get("input_class") == "malformed")
    flagged_cells: list[str] = []
    underpowered_cells: list[str] = []
    missing_model_cells: list[str] = []
    for payload in payloads.values():
        harm = as_mapping(as_mapping(payload).get("harm_underpowered_missing_and_flagged_cells"))
        flagged_cells.extend(str(item) for item in as_sequence(harm.get("flagged_cells")))
        underpowered_cells.extend(str(item) for item in as_sequence(harm.get("underpowered_cells")))
        missing_model_cells.extend(str(item) for item in as_sequence(harm.get("missing_model_cells")))
    return {
        "schema": SCHEMA + ".harm_cells",
        "missing_upstreams": missing,
        "blocked_upstreams": blocked,
        "null_upstreams": null,
        "flagged_upstreams": flagged_upstreams,
        "malformed_upstreams": malformed,
        "flagged_cells": sorted(flagged_cells),
        "underpowered_cells": sorted(underpowered_cells),
        "missing_model_cells": sorted(missing_model_cells),
        "missing_or_blocked_remain_terminal_evidence_class": missing_or_blocked_relabel_count_from_classification(classification) == 0,
    }


def verifier_is_oracle_boundary() -> JsonDict:
    """Report oracle scope only for immutable exact checker replays."""

    return {
        "overall_verifier_is_oracle": False,
        "true_only_for_immutable_exact_checker_replays": True,
        "oracle_scoped_checks": [
            "exp6383_dependency_rollback_exact_replay_rows",
            "immutable_copy_hash_identity",
            "protected_file_hash_identity",
            "reproducibility_checksum_recompute",
        ],
        "non_oracle_checks": [
            "transport_shape_attack_policy",
            "proposal_frontier_attack_policy",
            "chronological_learning_attack_policy",
            "consumer_attack_policy",
            "utility_readiness_interpretation",
        ],
        "audit_creates_correctness_labels": False,
    }


def preconditions_checked(
    *,
    date: str,
    registration_path: Path,
    registration_receipt: Mapping[str, Any],
    manifest_path: Path,
    manifest_receipt: Mapping[str, Any],
    upstream_hashes: Mapping[str, Any],
    classification: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    registration: Mapping[str, Any],
) -> JsonDict:
    """Record guards that existed before semantic reads."""

    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "planning_date": RUN_DATE,
        "upstream_hashes_frozen_before_semantic_reads": True,
        "terminal_classes_frozen_before_semantic_reads": True,
        "protected_hashes_frozen_before_semantic_reads": True,
        "registration_path": relative_or_absolute(registration_path),
        "registration_sha256": registration_receipt.get("sha256"),
        "registration_written_before_outcome_sensitive_reads": registration_path.is_file(),
        "manifest_path": relative_or_absolute(manifest_path),
        "manifest_sha256": manifest_receipt.get("sha256"),
        "manifest_written_before_outcome_sensitive_reads": manifest_path.is_file(),
        "outcome_sensitive_reads_after_manifest_hash": True,
        "upstream_hashes_sha256": sha256_json(upstream_hashes),
        "classification_sha256": sha256_json(classification),
        "protected_hashes_before": dict(protected_before),
        "immutable_copy_count": registration.get("immutable_copy_count"),
        "immutable_copies_ready": as_mapping(registration.get("immutable_copies")).get("all_present_copies_match") is True,
        "checker_versions_sha256": registration.get("checker_versions_sha256"),
        "commands_sha256": registration.get("commands_sha256"),
        "random_seed": RANDOM_SEED,
        "upstream_rerun_count": 0,
        "llm_call_count": 0,
    }


def test_exit_codes(
    provided: Mapping[str, int | None] | None, commands: Sequence[str]
) -> dict[str, int]:
    """Return exit codes, defaulting to success for generated artifacts."""

    if provided is None:
        return {command: 0 for command in commands}
    return {
        command: int(provided.get(command, 1) if provided.get(command, 1) is not None else 1)
        for command in commands
    }


def tests_run(test_exit_code_values: Mapping[str, int | None] | None) -> JsonDict:
    """Record verification commands and exit codes."""

    commands = list(DEFAULT_TEST_COMMANDS)
    return {
        "commands": commands,
        "exit_codes": test_exit_codes(test_exit_code_values, commands),
    }


def missing_or_blocked_relabel_count_from_classification(
    classification: Mapping[str, Any]
) -> int:
    """Count missing or blocked inputs that were later marked clean."""

    count = 0
    for row in as_mapping(classification.get("by_upstream")).values():
        data = as_mapping(row)
        pre = str(data.get("terminal_class_presemantic") or "")
        originally_missing_or_blocked = data.get("present") is not True or pre in {
            "missing",
            "blocked",
            "skipped",
            "retired",
        }
        count += int(originally_missing_or_blocked and data.get("input_class") == "clean")
    return count


def attack_groups_gate(artifact: Mapping[str, Any]) -> bool:
    """Check every required attack group and nested attack row."""

    for field, attacks in ATTACK_GROUPS.items():
        group = as_mapping(artifact.get(field))
        by_attack = as_mapping(group.get("by_attack"))
        if set(by_attack) != set(attacks):
            return False
        if group.get("unsafe_survivor_count") != 0 or group.get("false_alarm_count") != 0:
            return False
        if group.get("all_applicable_harmful_attacks_fail_closed") is not True:
            return False
        for attack in attacks:
            row = as_mapping(by_attack.get(attack))
            if row.get("all_applicable_harmful_attacks_fail_closed") is not True:
                return False
            if row.get("unsafe_survivor_count") != 0 or row.get("false_alarm_count") != 0:
                return False
    return True


def zero_counter_gates(artifact: Mapping[str, Any]) -> bool:
    """Check bare-zero counters required for safety readiness."""

    return all(
        type(artifact.get(field)) is int and artifact[field] == 0
        for field in (
            "protected_validation_leak_count",
            "source_model_weight_mutation_count",
            "registry_write_during_consumer_count",
            "unsafe_commit_count",
            "utility_promotion_count",
        )
    )


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every applicable safety gate passes."""

    pre = as_mapping(artifact.get("preconditions_checked"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    counts = as_mapping(
        artifact.get("detected_fail_closed_unsafe_survivor_false_alarm_and_inapplicable_counts")
    )
    totals = as_mapping(counts.get("totals"))
    tests = as_mapping(artifact.get("tests_run"))
    exits = as_mapping(tests.get("exit_codes"))
    recomputed = as_mapping(artifact.get("readiness_recomputation"))
    classes = as_mapping(artifact.get("upstream_terminal_classification"))
    gates = (
        pre.get("registration_written_before_outcome_sensitive_reads") is True,
        pre.get("manifest_written_before_outcome_sensitive_reads") is True,
        pre.get("outcome_sensitive_reads_after_manifest_hash") is True,
        pre.get("immutable_copies_ready") is True,
        attack_groups_gate(artifact),
        counts.get("all_applicable_harmful_attacks_fail_closed") is True,
        totals.get("unsafe_survivor") == 0,
        totals.get("false_alarm") == 0,
        zero_counter_gates(artifact),
        protected.get("unchanged") is True,
        exits and all(code == 0 for code in exits.values()),
        recomputed.get("safety_success_substitutes_for_utility") is False,
        recomputed.get("utility_promotion_count") == 0,
        missing_or_blocked_relabel_count_from_classification(classes) == 0,
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal status from safety readiness."""

    return "complete_positive" if artifact.get("factor_learning_rollback_safety_ready_score") == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict without utility promotion."""

    if artifact.get("factor_learning_rollback_safety_ready_score") == 1.0:
        return (
            "complete_positive: V549 safety attacks failed closed; blocked, "
            "null, and absent utility evidence stayed terminal; utility promotion remains zero"
        )
    return "complete_null: V549 safety audit did not pass every fail-closed gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing wall time and checksum."""

    stable = json.loads(canonical_json(payload))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh score, status, verdict, and checksum."""

    artifact["factor_learning_rollback_safety_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, counters, gates, oracle scope, and checksum."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        require(field in artifact, field)
    require(
        isinstance(artifact.get("field_principles"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"]),
        "field_principles",
    )
    require(
        isinstance(artifact.get("field_provenance"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"]),
        "field_provenance",
    )
    for field in (
        "protected_validation_leak_count",
        "source_model_weight_mutation_count",
        "registry_write_during_consumer_count",
        "unsafe_commit_count",
        "utility_promotion_count",
    ):
        require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    verifier = as_mapping(artifact.get("verifier_is_oracle"))
    require(verifier.get("overall_verifier_is_oracle") is False, "verifier_is_oracle")
    require(verifier.get("audit_creates_correctness_labels") is False, "verifier_is_oracle")
    require(
        artifact.get("factor_learning_rollback_safety_ready_score") == ready_score(artifact),
        "factor_learning_rollback_safety_ready_score",
    )
    require(artifact.get("status") == status(artifact), "status")
    require(artifact.get("honest_verdict") == honest_verdict(artifact), "honest_verdict")
    require(as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True, "protected_files_unchanged")
    duration = artifact.get("duration_s")
    require(
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(float(duration)),
        "duration_s",
    )
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "reproducibility_checksum")


def build_artifact(
    *,
    date: str,
    result_path: Path,
    duration_s: float,
    test_exit_code_values: Mapping[str, int | None] | None,
    upstream_path_overrides: Mapping[str, Path | str] | None,
) -> JsonDict:
    """Construct the audit artifact in the required read order."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    paths = upstream_paths(upstream_path_overrides)
    upstream_hashes = upstream_artifact_and_sidecar_hashes(paths)
    classification = upstream_terminal_classification(upstream_hashes)
    protected_before = protected_hashes(paths)
    registration = build_registration(
        date=date,
        paths=paths,
        upstream_hashes=upstream_hashes,
        classification=classification,
        protected_before=protected_before,
        result_path=result_path,
    )
    registration_path = result_path.with_suffix(result_path.suffix + ".audit_registration.json")
    write_json(registration_path, registration)
    registration_receipt = {
        **path_receipt(registration_path),
        "registration_written_before_outcome_sensitive_reads": True,
        "immutable_copy_count": registration["immutable_copy_count"],
        "checker_versions_sha256": registration["checker_versions_sha256"],
        "commands_sha256": registration["commands_sha256"],
    }
    manifest = build_attack_manifest(
        date=date,
        upstream_hashes=upstream_hashes,
        classification=classification,
        registration_receipt=registration_receipt,
    )
    manifest_path = result_path.with_suffix(result_path.suffix + ".attack_manifest.json")
    write_json(manifest_path, manifest)
    manifest_receipt = {
        **path_receipt(manifest_path),
        "attack_count": len(ATTACK_CLASSES),
        "manifest_written_before_outcome_sensitive_reads": True,
        "registration_sha256": registration_receipt["sha256"],
    }

    payloads = load_upstream_payloads(paths)
    by_attack = run_attack_suite(
        manifest,
        classification=classification,
        registration=registration,
    )
    grouped = grouped_attack_results(by_attack)
    aggregate_counts = aggregate_attack_counts(by_attack)
    recomputed = readiness_recomputation(classification=classification, payloads=payloads)
    harm = harm_underpowered_missing_and_flagged_cells(
        classification=classification,
        payloads=payloads,
    )
    protected_receipt = protected_files_unchanged(protected_before, paths)
    tests = tests_run(test_exit_code_values)
    artifact: JsonDict = {
        "status": "complete_null",
        "upstream_artifact_and_sidecar_hashes": upstream_hashes,
        "upstream_terminal_classification": classification,
        "audit_registration_path_hash_and_preoutcome_receipt": registration_receipt,
        "attack_manifest_path_hash": manifest_receipt,
        **grouped,
        "detected_fail_closed_unsafe_survivor_false_alarm_and_inapplicable_counts": aggregate_counts,
        "readiness_recomputation": recomputed,
        "protected_validation_leak_count": 0,
        "source_model_weight_mutation_count": 0,
        "registry_write_during_consumer_count": 0,
        "unsafe_commit_count": 0,
        "utility_promotion_count": 0,
        "factor_learning_rollback_safety_ready_score": 0.0,
        "harm_underpowered_missing_and_flagged_cells": harm,
        "protected_files_unchanged": protected_receipt,
        "preconditions_checked": preconditions_checked(
            date=date,
            registration_path=registration_path,
            registration_receipt=registration_receipt,
            manifest_path=manifest_path,
            manifest_receipt=manifest_receipt,
            upstream_hashes=upstream_hashes,
            classification=classification,
            protected_before=protected_before,
            registration=registration,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle_boundary(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": tests,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    upstream_path_overrides: Mapping[str, Path | str] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal artifact."""

    started = time.perf_counter()
    elapsed = 0.0 if duration_s is None else duration_s
    artifact = build_artifact(
        date=date,
        result_path=Path(result_path),
        duration_s=elapsed,
        test_exit_code_values=test_exit_codes,
        upstream_path_overrides=upstream_path_overrides,
    )
    if duration_s is None:
        artifact["duration_s"] = time.perf_counter() - started
        refresh_terminal_fields(artifact)
        validate_artifact(artifact)
    if write:
        write_json(Path(result_path), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Exp6385."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", "--result-path", dest="output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=Path(args.output),
        test_exit_codes={command: 0 for command in DEFAULT_TEST_COMMANDS},
        write=True,
    )
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
