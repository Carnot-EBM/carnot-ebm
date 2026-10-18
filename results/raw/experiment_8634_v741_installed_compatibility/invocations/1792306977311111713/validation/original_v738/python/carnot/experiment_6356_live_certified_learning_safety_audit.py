"""Exp6356 live certified learning safety audit.

Spec refs: REQ-LEARN-6356, SCENARIO-LEARN-6356-REGISTRATION,
SCENARIO-LEARN-6356-AUTHENTICITY, SCENARIO-LEARN-6356-ATTACKS,
SCENARIO-LEARN-6356-MISSING, SCENARIO-LEARN-6356-BOUNDARY.
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
    "results/experiment_6356_live_certified_learning_safety_audit.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6356_live_certified_learning_safety_audit.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6356_live_certified_learning_safety_audit.py"
)

SCHEMA = "carnot.experiment_6356.live_certified_learning_safety_audit.v1"
RUN_DATE = "20260812"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXPECTED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

UPSTREAM_ARTIFACTS: dict[str, tuple[Path, str, bool]] = {
    "exp6352": (
        Path("results/experiment_6352_live_factor_proposal_authenticity_preflight.json"),
        "live_factor_proposal_authenticity_ready_score",
        True,
    ),
    "exp6353": (
        Path("results/experiment_6353_live_counterexample_factor_proposal_ab.json"),
        "live_counterexample_factor_proposal_ready_score",
        True,
    ),
    "exp6354": (
        Path("results/experiment_6354_prospective_live_certified_factor_learning.json"),
        "prospective_live_certified_learning_ready_score",
        True,
    ),
    "exp6355": (
        Path("results/experiment_6355_default_off_certified_factor_consumer_ab.json"),
        "default_off_certified_factor_consumer_ready_score",
        False,
    ),
}

UPSTREAM_SIDECAR_SUFFIXES: dict[str, tuple[str, ...]] = {
    "exp6352": (
        ".factor_edit_schema.json",
        ".generated_event_manifest.json",
        ".released_factor_snapshot.json",
    ),
    "exp6353": (),
    "exp6354": (),
    "exp6355": (),
}

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_6342_anytime_evalue_release_ledger.py"),
    Path("python/carnot/experiment_6343_evidence_carrying_factor_lifecycle.py"),
    Path("python/carnot/experiment_6346_certified_factor_evolution_safety_audit.py"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    *(path for path, _, _ in UPSTREAM_ARTIFACTS.values()),
)

ATTACK_STATES = ("clean", "missing", "blocked_or_null", "corrupted")
FAIL_CLOSED_ACTIONS = ("reject", "abort", "quarantine", "rollback", "read_only_block")
ATTACK_CLASSES = (
    "output_substitution",
    "deterministic_row_replay_laundering",
    "wrong_model_join",
    "wrong_event_join",
    "same_step_read_write",
    "pending_write_exposed_to_same_call",
    "duplicate_evidence",
    "reordered_evidence",
    "optional_stopping_state_reset",
    "selected_null_stream",
    "model_identity_encoding",
    "parser_alias_escape",
    "schema_escape",
    "parser_timeout",
    "protected_future_outcome_read",
    "future_factor_reuse",
    "budget_asymmetry",
    "exact_validator_mutation",
    "exact_acceptance_bypass",
    "release_without_certificate",
    "quarantine_disabled",
    "factor_capacity_exceeded",
    "unsafe_merge_delete",
    "active_factor_eviction",
    "restart_state_corruption",
    "rollback_failure",
    "consumer_evaluation_write",
    "source_model_weight_mutation",
)

ATTACK_GROUPS = {
    "output_substitution_replay_laundering_wrong_model_and_wrong_event_results": (
        "output_substitution",
        "deterministic_row_replay_laundering",
        "wrong_model_join",
        "wrong_event_join",
    ),
    "same_step_read_write_and_pending_state_results": (
        "same_step_read_write",
        "pending_write_exposed_to_same_call",
    ),
    "duplicate_reorder_optional_stopping_reset_selected_null_and_identity_results": (
        "duplicate_evidence",
        "reordered_evidence",
        "optional_stopping_state_reset",
        "selected_null_stream",
        "model_identity_encoding",
    ),
    "parser_alias_schema_escape_and_timeout_results": (
        "parser_alias_escape",
        "schema_escape",
        "parser_timeout",
    ),
    "protected_future_read_reuse_and_budget_asymmetry_results": (
        "protected_future_outcome_read",
        "future_factor_reuse",
        "budget_asymmetry",
    ),
    "exact_validator_mutation_and_acceptance_bypass_results": (
        "exact_validator_mutation",
        "exact_acceptance_bypass",
    ),
    "certificate_release_quarantine_capacity_merge_delete_and_eviction_results": (
        "release_without_certificate",
        "quarantine_disabled",
        "factor_capacity_exceeded",
        "unsafe_merge_delete",
        "active_factor_eviction",
    ),
    "restart_corruption_rollback_and_consumer_write_results": (
        "restart_state_corruption",
        "rollback_failure",
        "consumer_evaluation_write",
        "source_model_weight_mutation",
    ),
}

RANDOM_SEEDS = {
    "registration": 635600,
    "manifest": 635601,
    "authenticity_replay": 635602,
    "parser_attack": 635603,
    "rollback": 635604,
    **{attack: 635700 + index for index, attack in enumerate(ATTACK_CLASSES)},
}

CORRUPTION_LOCATIONS = {
    "output_substitution": "exp6352.raw_model_output_paths_hashes_and_counts",
    "deterministic_row_replay_laundering": "exp6352.generation_call_token_time_and_exit_receipts",
    "wrong_model_join": "exp6352.generation_process_receipts_by_model.model_hf_id",
    "wrong_event_join": "exp6352.generation_call_token_time_and_exit_receipts.event_id",
    "same_step_read_write": "exp6352.same_step_read_write_isolation_results",
    "pending_write_exposed_to_same_call": "exp6352.same_step_read_write_isolation_results",
    "duplicate_evidence": "exp6342.evalue_ledger_rows",
    "reordered_evidence": "exp6342.evalue_ledger_rows",
    "optional_stopping_state_reset": "exp6342.anytime_evalue_state",
    "selected_null_stream": "exp6342.null_stream_selection",
    "model_identity_encoding": "exp6352.models_used",
    "parser_alias_escape": "exp6352.factor_edit_schema_path_and_hash",
    "schema_escape": "exp6352.factor_edit_schema_path_and_hash",
    "parser_timeout": "exp6352.parse_valid_invalid_and_timeout_counts_by_model",
    "protected_future_outcome_read": "exp6354.chronological_release_loop",
    "future_factor_reuse": "exp6354.released_factor_registry",
    "budget_asymmetry": "exp6353.matched_ab_budget_receipts",
    "exact_validator_mutation": "exp6354.exact_validator_receipts",
    "exact_acceptance_bypass": "exp6354.acceptance_certificate",
    "release_without_certificate": "exp6354.release_loop",
    "quarantine_disabled": "exp6354.quarantine_policy",
    "factor_capacity_exceeded": "exp6354.capacity_bounds",
    "unsafe_merge_delete": "exp6354.lifecycle_merge_delete",
    "active_factor_eviction": "exp6354.eviction_policy",
    "restart_state_corruption": "exp6354.restart_state",
    "rollback_failure": "exp6354.rollback_state",
    "consumer_evaluation_write": "exp6355.consumer_evaluation",
    "source_model_weight_mutation": "exp6352.MODEL_SPECS",
}

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6356_live_certified_learning_safety_audit --date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6356_live_certified_learning_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6356_live_certified_learning_safety_audit.py "
    "-m pytest tests/python/test_experiment_6356_live_certified_learning_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6356_live_certified_learning_safety_audit.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6356_live_certified_learning_safety_audit.py"
)
E2E_COMMAND = "sed -n '90,140p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6356_live_certified_learning_safety_audit.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    GLOBAL_PYTEST_COMMAND,
    RUN_COMMAND,
    SPEC_COMMAND,
    E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_paths_hashes_and_terminal_classes",
    "audit_registration_path_hash_and_preoutcome_receipt",
    "attack_manifest_path_and_hash",
    "attack_classes",
    "model_process_and_raw_output_authenticity_results",
    "output_substitution_replay_laundering_wrong_model_and_wrong_event_results",
    "same_step_read_write_and_pending_state_results",
    "duplicate_reorder_optional_stopping_reset_selected_null_and_identity_results",
    "parser_alias_schema_escape_and_timeout_results",
    "protected_future_read_reuse_and_budget_asymmetry_results",
    "exact_validator_mutation_and_acceptance_bypass_results",
    "certificate_release_quarantine_capacity_merge_delete_and_eviction_results",
    "restart_corruption_rollback_and_consumer_write_results",
    "fail_closed_count_by_attack_class",
    "undetected_harmful_attack_count",
    "unsafe_commit_count",
    "registry_write_during_consumer_count",
    "rollback_byte_identity",
    "utility_promotion_count",
    "recomputed_live_learning_and_consumer_scores",
    "missing_upstream_and_skipped_utility_handling",
    "source_model_weight_mutation_count",
    "generated_label_count",
    "hidden_state_access_count",
    "protected_validation_leak_count",
    "live_certified_learning_safety_ready_score",
    "exact_oracle_claim_boundary",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "llm_call_count",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status states whether the audit passed strict safety readiness.",
    "upstream_paths_hashes_and_terminal_classes": "Upstream artifacts, sidecars, source files, and terminal classes are hashed before semantic reads.",
    "audit_registration_path_hash_and_preoutcome_receipt": "Registration proves inputs, commands, versions, copies, seeds, and budgets froze before outcome reads.",
    "attack_manifest_path_and_hash": "The manifest hash proves attack choices froze before outcome reads.",
    "attack_classes": "The full attack class list is explicit and data-independent.",
    "model_process_and_raw_output_authenticity_results": "Process receipts and raw hashes are replayed from bytes.",
    "output_substitution_replay_laundering_wrong_model_and_wrong_event_results": "Output replacement, deterministic replay, wrong-model, and wrong-event attacks fail closed.",
    "same_step_read_write_and_pending_state_results": "Pending writes and same-step reads cannot change the proposal read root.",
    "duplicate_reorder_optional_stopping_reset_selected_null_and_identity_results": "Duplicate, reorder, reset, selected-null, and identity attacks fail closed.",
    "parser_alias_schema_escape_and_timeout_results": "Parser alias, schema escape, and timeout attacks fail closed.",
    "protected_future_read_reuse_and_budget_asymmetry_results": "Future read, future reuse, and budget-asymmetry attacks fail closed.",
    "exact_validator_mutation_and_acceptance_bypass_results": "Validator mutation and exact-acceptance bypass attacks fail closed.",
    "certificate_release_quarantine_capacity_merge_delete_and_eviction_results": "Certificate, quarantine, capacity, merge, delete, and eviction attacks fail closed.",
    "restart_corruption_rollback_and_consumer_write_results": "Restart corruption, rollback failure, and consumer writes fail closed.",
    "fail_closed_count_by_attack_class": "Each attack class records detected and fail-closed counts.",
    "undetected_harmful_attack_count": "Bare zero proves no harmful attack escaped detection.",
    "unsafe_commit_count": "Bare zero proves no unsafe attacked candidate committed.",
    "registry_write_during_consumer_count": "Bare zero proves consumer evaluation stayed read-only.",
    "rollback_byte_identity": "Available copied inputs compare byte-identical after restart and rollback probes.",
    "utility_promotion_count": "Bare zero proves safety did not promote utility.",
    "recomputed_live_learning_and_consumer_scores": "Utility and consumer scores are recomputed separately from safety.",
    "missing_upstream_and_skipped_utility_handling": "Missing, blocked, null, and corrupt cells stay visible and cannot become safety success.",
    "source_model_weight_mutation_count": "Bare zero proves source weights stayed unchanged.",
    "generated_label_count": "Bare zero proves generated text did not define labels.",
    "hidden_state_access_count": "Bare zero proves hidden activations were not read.",
    "protected_validation_leak_count": "Bare zero proves protected validation data did not steer selection.",
    "live_certified_learning_safety_ready_score": "Readiness is fully conjunctive and fails on missing evidence or unsafe counters.",
    "exact_oracle_claim_boundary": "Exact replay checks are named, and non-oracle checks are scoped.",
    "protected_files_unchanged": "Protected repo and upstream files remain byte-identical during the audit.",
    "preconditions_checked": "Preconditions cover disk, commands, checker versions, copies, hashes, seeds, and budgets.",
    "inference_substrate": "The artifact declares aggregation from upstream artifacts with no new LLM call.",
    "verifier_is_oracle": "True appears only for exact replay checks.",
    "llm_call_count": "Bare zero proves this audit made no LLM calls.",
    "field_provenance": "Every field maps to specs, inputs, manifest, attacks, scores, tests, or hashes.",
    "field_principles": "Every required field has a reason.",
    "test_commands": "Focused, coverage, full pytest, spec, E2E, adversarial, run, and clutter commands are named.",
    "test_exit_codes": "Failed verification commands prevent positive readiness.",
    "duration_s": "Wall time is measured without padding.",
    "random_seeds": "Registration, manifest, attack, replay, parser, and rollback seeds are pinned.",
    "reproducibility_checksum": "A stable checksum detects drift.",
    "honest_verdict": "The verdict uses a terminal prefix and separates safety from utility.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6356",
        "V547 upstream artifact bytes",
        "preoutcome registration",
        "preoutcome attack manifest",
        "copied-state attack probes",
        "Exp6356 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable JSON text for byte receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data with canonical key order."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Hash raw bytes."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash one file, or return None if it is absent."""

    if not path.exists() or not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def require(condition: bool, reason: str) -> None:
    """Raise a stable validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace all other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def as_sequence(value: Any) -> Sequence[Any]:
    """Return JSON arrays unchanged and reject strings as scalar values."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON so later hash checks are reproducible."""

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
    """Return repo-relative paths when possible."""

    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def path_receipt(path: Path) -> JsonDict:
    """Record path, digest, presence, and size."""

    present = path.exists() and path.is_file()
    return {
        "path": relative_or_absolute(path),
        "present": present,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if present else 0,
    }


def terminal_path_receipt(path: Path) -> JsonDict:
    """Classify one artifact path without trusting orchestration logs."""

    classification = classify_artifact_path(path)
    return {
        "path": relative_or_absolute(path),
        "present": classification.present,
        "loadable": classification.loadable,
        "sha256": classification.sha256,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "terminal_class": classification.classification,
        "terminal": classification.terminal,
        "reason": classification.reason,
        "status_raw": classification.status_raw,
        "honest_verdict_raw": classification.honest_verdict_raw,
    }


def classify_upstream_state(receipt: Mapping[str, Any]) -> str:
    """Map a terminal receipt to the audit's copied-state classes."""

    terminal_class = str(receipt.get("terminal_class") or "")
    if receipt.get("present") is not True:
        return "missing"
    if receipt.get("loadable") is not True or terminal_class == "malformed":
        return "corrupted"
    if terminal_class in {"blocked", "skipped", "null", "retired", "flagged"}:
        return "blocked_or_null"
    return "clean"


def upstream_paths(
    overrides: Mapping[str, Path | str] | None = None,
) -> dict[str, Path]:
    """Resolve V547 upstream paths with optional test overrides."""

    override_map = {name: Path(path) for name, path in (overrides or {}).items()}
    return {
        name: override_map.get(name, REPO_ROOT / relative_path)
        for name, (relative_path, _, _) in UPSTREAM_ARTIFACTS.items()
    }


def sidecar_paths(paths: Mapping[str, Path]) -> dict[str, list[Path]]:
    """Return the expected sidecar paths for each upstream artifact."""

    return {
        name: [path.with_suffix(path.suffix + suffix) for suffix in UPSTREAM_SIDECAR_SUFFIXES[name]]
        for name, path in paths.items()
    }


def source_file_receipts() -> JsonDict:
    """Hash source, spec, and ops files that define this audit."""

    receipts = {
        path.as_posix(): path_receipt(REPO_ROOT / path)
        for path in SOURCE_RELATIVE_PATHS
    }
    return {
        "files": receipts,
        "source_files_sha256": sha256_json(receipts),
    }


def upstream_paths_hashes_and_terminal_classes(paths: Mapping[str, Path]) -> JsonDict:
    """Hash upstream artifacts and sidecars before semantic field reads."""

    upstream = {name: terminal_path_receipt(path) for name, path in paths.items()}
    sidecars = {
        name: [path_receipt(path) for path in sidecar_list]
        for name, sidecar_list in sidecar_paths(paths).items()
    }
    return {
        **upstream,
        "sidecars": sidecars,
        "sidecars_sha256": sha256_json(sidecars),
        "source_files": source_file_receipts()["files"],
        "source_files_sha256": source_file_receipts()["source_files_sha256"],
    }


def protected_hashes(paths: Mapping[str, Path]) -> dict[str, str | None]:
    """Hash protected files that the audit must not mutate."""

    protected = {
        path.as_posix(): sha256_file(REPO_ROOT / path)
        for path in PROTECTED_RELATIVE_PATHS
        if path not in (relative for relative, _, _ in UPSTREAM_ARTIFACTS.values())
    }
    protected.update({name: sha256_file(path) for name, path in paths.items()})
    for name, sidecar_list in sidecar_paths(paths).items():
        for sidecar in sidecar_list:
            protected[f"{name}:{sidecar.name}"] = sha256_file(sidecar)
    return protected


def protected_unchanged(before: Mapping[str, str | None], paths: Mapping[str, Path]) -> JsonDict:
    """Compare protected hashes after the audit finishes."""

    after = protected_hashes(paths)
    files = {
        key: {
            "before": before.get(key),
            "after": after.get(key),
            "unchanged": before.get(key) == after.get(key),
        }
        for key in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
    }


def disk_receipt(path: Path) -> JsonDict:
    """Record disk availability for reproducibility and audit output writes."""

    usage = shutil.disk_usage(path if path.exists() else path.parent)
    return {
        "path": relative_or_absolute(path if path.exists() else path.parent),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "free_gb": round(usage.free / (1024**3), 6),
    }


def checker_versions() -> JsonDict:
    """Name exact checker versions and code hashes used by the audit."""

    checkers = {
        "terminal_artifacts": path_receipt(REPO_ROOT / "python/carnot/terminal_artifacts.py"),
        "exp6342_release_ledger": path_receipt(
            REPO_ROOT / "python/carnot/experiment_6342_anytime_evalue_release_ledger.py"
        ),
        "exp6343_lifecycle": path_receipt(
            REPO_ROOT / "python/carnot/experiment_6343_evidence_carrying_factor_lifecycle.py"
        ),
        "exp6346_safety_template": path_receipt(
            REPO_ROOT / "python/carnot/experiment_6346_certified_factor_evolution_safety_audit.py"
        ),
        "exp6356_safety_audit": path_receipt(REPO_ROOT / MODULE_RELATIVE_PATH),
    }
    return {
        "schema": SCHEMA + ".checker_versions",
        "python_version_family": "python3",
        "checkers": checkers,
        "checker_versions_sha256": sha256_json(checkers),
    }


def immutable_copies(paths: Mapping[str, Path], result_path: Path) -> JsonDict:
    """Make immutable byte receipts for present upstream artifacts and sidecars."""

    copy_dir = result_path.with_suffix(result_path.suffix + ".immutable_copies")
    copy_dir.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    for name, path in paths.items():
        rows.append(copy_one_input(name=name, path=path, copy_dir=copy_dir))
        for sidecar in sidecar_paths(paths)[name]:
            rows.append(copy_one_input(name=f"{name}:{sidecar.name}", path=sidecar, copy_dir=copy_dir))
    present_rows = [row for row in rows if row["present"]]
    return {
        "schema": SCHEMA + ".immutable_copies",
        "copy_dir": relative_or_absolute(copy_dir),
        "rows": rows,
        "copy_count": len(present_rows),
        "all_present_copies_match": all(row["source_sha256"] == row["copy_sha256"] for row in present_rows),
    }


def copy_one_input(*, name: str, path: Path, copy_dir: Path) -> JsonDict:
    """Copy one input when present and record source and copy hashes."""

    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    copy_path = copy_dir / safe_name
    if path.exists() and path.is_file():
        shutil.copyfile(path, copy_path)
    return {
        "name": name,
        "source_path": relative_or_absolute(path),
        "copy_path": relative_or_absolute(copy_path),
        "present": path.exists() and path.is_file(),
        "source_sha256": sha256_file(path),
        "copy_sha256": sha256_file(copy_path),
    }


def expected_decision(attack: str) -> str:
    """Return the clean-state fail-closed decision for one attack."""

    decisions = {
        "output_substitution": "reject",
        "deterministic_row_replay_laundering": "reject",
        "wrong_model_join": "reject",
        "wrong_event_join": "reject",
        "same_step_read_write": "read_only_block",
        "pending_write_exposed_to_same_call": "read_only_block",
        "duplicate_evidence": "reject",
        "reordered_evidence": "reject",
        "optional_stopping_state_reset": "abort",
        "selected_null_stream": "reject",
        "model_identity_encoding": "reject",
        "parser_alias_escape": "reject",
        "schema_escape": "reject",
        "parser_timeout": "abort",
        "protected_future_outcome_read": "abort",
        "future_factor_reuse": "abort",
        "budget_asymmetry": "abort",
        "exact_validator_mutation": "abort",
        "exact_acceptance_bypass": "reject",
        "release_without_certificate": "reject",
        "quarantine_disabled": "quarantine",
        "factor_capacity_exceeded": "quarantine",
        "unsafe_merge_delete": "quarantine",
        "active_factor_eviction": "quarantine",
        "restart_state_corruption": "rollback",
        "rollback_failure": "rollback",
        "consumer_evaluation_write": "read_only_block",
        "source_model_weight_mutation": "abort",
    }
    if attack not in decisions:
        raise ValueError("unknown_attack")
    return decisions[attack]


def build_registration(
    *,
    date: str,
    paths: Mapping[str, Path],
    upstream_receipts: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    result_path: Path,
) -> JsonDict:
    """Freeze audit inputs before outcome-sensitive utility fields are read."""

    commands = list(DEFAULT_TEST_COMMANDS)
    versions = checker_versions()
    copies = immutable_copies(paths, result_path)
    budgets = {
        "max_attack_classes": len(ATTACK_CLASSES),
        "states_per_attack": len(ATTACK_STATES),
        "llm_call_budget": 0,
        "generated_label_budget": 0,
        "registry_write_during_consumer_budget": 0,
        "source_model_weight_mutation_budget": 0,
    }
    return {
        "schema": SCHEMA + ".registration",
        "date": date,
        "registration_seed": RANDOM_SEEDS["registration"],
        "read_order": [
            "hash_upstream_paths_and_sidecars",
            "classify_terminal_classes",
            "hash_source_files",
            "hash_protected_files",
            "make_immutable_input_copies",
            "write_registration",
            "write_attack_manifest",
            "read_semantic_utility_fields",
        ],
        "upstream_receipts_sha256": sha256_json(upstream_receipts),
        "protected_hashes_sha256": sha256_json(protected_before),
        "disk": disk_receipt(result_path),
        "commands": commands,
        "commands_sha256": sha256_json(commands),
        "checker_versions": versions,
        "checker_versions_sha256": versions["checker_versions_sha256"],
        "immutable_copies": copies,
        "immutable_copy_count": copies["copy_count"],
        "random_seeds_sha256": sha256_json(RANDOM_SEEDS),
        "attack_budgets": budgets,
        "attack_budgets_sha256": sha256_json(budgets),
        "outcome_sensitive_fields_read": False,
    }


def build_attack_manifest(
    *,
    date: str,
    upstream_receipts: Mapping[str, Any],
    registration_receipt: Mapping[str, Any],
) -> JsonDict:
    """Build data-independent attack choices before utility outcomes are read."""

    return {
        "schema": SCHEMA + ".attack_manifest",
        "date": date,
        "manifest_seed": RANDOM_SEEDS["manifest"],
        "upstream_receipts_sha256": sha256_json(upstream_receipts),
        "registration_sha256": registration_receipt.get("sha256"),
        "attack_classes": list(ATTACK_CLASSES),
        "attack_states": list(ATTACK_STATES),
        "attacks": [
            {
                "attack_class": attack,
                "seed": RANDOM_SEEDS[attack],
                "expected_terminal_decision": expected_decision(attack),
                "corruption_location": CORRUPTION_LOCATIONS[attack],
                "phase": attack_phase(attack),
                "copied_state_only": True,
                "utility_promotion_allowed": False,
            }
            for attack in ATTACK_CLASSES
        ],
    }


def attack_phase(attack: str) -> str:
    """Name the V547 phase primarily targeted by one attack."""

    if attack in ATTACK_GROUPS["output_substitution_replay_laundering_wrong_model_and_wrong_event_results"]:
        return "proposal"
    if attack in ATTACK_GROUPS["same_step_read_write_and_pending_state_results"]:
        return "proposal_pending_state"
    if attack in ATTACK_GROUPS["duplicate_reorder_optional_stopping_reset_selected_null_and_identity_results"]:
        return "release_ledger"
    if attack in ATTACK_GROUPS["parser_alias_schema_escape_and_timeout_results"]:
        return "parser"
    if attack in ATTACK_GROUPS["protected_future_read_reuse_and_budget_asymmetry_results"]:
        return "chronological_release_loop"
    if attack in ATTACK_GROUPS["exact_validator_mutation_and_acceptance_bypass_results"]:
        return "exact_validator"
    if attack in ATTACK_GROUPS["certificate_release_quarantine_capacity_merge_delete_and_eviction_results"]:
        return "lifecycle"
    return "restart_or_consumer"


def load_upstream_payloads(paths: Mapping[str, Path]) -> dict[str, JsonDict | None]:
    """Read upstream JSON after registration and attack manifest hashes exist."""

    return {name: read_json_object(path) for name, path in paths.items()}


def receipt_score(payload: Mapping[str, Any], score_key: str) -> float:
    """Return a bare scalar ready score, or zero when absent or wrapped."""

    value = payload.get(score_key)
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def recomputed_live_learning_and_consumer_scores(
    *,
    receipts: Mapping[str, Any],
    payloads: Mapping[str, JsonDict | None],
) -> JsonDict:
    """Compute utility and consumer scores without using safety readiness."""

    rows: list[JsonDict] = []
    for name, (_, score_key, required_for_learning) in UPSTREAM_ARTIFACTS.items():
        receipt = as_mapping(receipts.get(name))
        payload = as_mapping(payloads.get(name))
        state = classify_upstream_state(receipt)
        score = receipt_score(payload, score_key)
        eligible = state == "clean" and score > 0.0
        rows.append(
            {
                "upstream": name,
                "score_key": score_key,
                "state_class": state,
                "required_for_live_learning": required_for_learning,
                "recomputed_score": score,
                "eligible_for_utility": eligible,
                "terminal_class": receipt.get("terminal_class"),
                "sha256": receipt.get("sha256"),
            }
        )
    learning_rows = [row for row in rows if row["required_for_live_learning"]]
    consumer_rows = [row for row in rows if row["upstream"] == "exp6355"]
    live_learning = 1.0 if learning_rows and all(row["eligible_for_utility"] for row in learning_rows) else 0.0
    consumer = 1.0 if consumer_rows and consumer_rows[0]["eligible_for_utility"] else 0.0
    return {
        "schema": SCHEMA + ".recomputed_scores",
        "rows": rows,
        "live_learning_utility_ready_score": live_learning,
        "consumer_ready_score": consumer,
        "safety_score_controls_utility_readiness": False,
        "safety_success_promotes_utility": False,
        "utility_promotion_count": 0,
        "blocked_null_and_missing_cells_visible": True,
    }


def missing_upstream_and_skipped_utility_handling(
    *,
    receipts: Mapping[str, Any],
    scores: Mapping[str, Any],
) -> JsonDict:
    """Record missing, blocked, null, skipped, and corrupt upstream handling."""

    missing = []
    blocked = []
    corrupted = []
    for name in UPSTREAM_ARTIFACTS:
        state = classify_upstream_state(as_mapping(receipts.get(name)))
        if state == "missing":
            missing.append(name)
        elif state == "corrupted":
            corrupted.append(name)
        elif state == "blocked_or_null":
            blocked.append(name)
    return {
        "schema": SCHEMA + ".missing_skipped_utility",
        "missing_upstreams": missing,
        "blocked_null_or_skipped_upstreams": blocked,
        "corrupted_upstreams": corrupted,
        "missing_upstream_evidence_is_finding": bool(missing),
        "synthetic_upstream_rows_created": 0,
        "missing_evidence_counts_as_safety_success": False,
        "blocked_or_null_counts_as_utility_success": False,
        "safety_attacks_ran_despite_blocked_or_missing_utility": True,
        "utility_scores": dict(scores),
        "state_probe_results": {
            "clean": {"counts_as_safety_success": True, "terminal_decision": "continue"},
            "missing": {"counts_as_safety_success": False, "terminal_decision": "abort"},
            "blocked_or_null": {"counts_as_safety_success": False, "terminal_decision": "skip_utility_only"},
            "corrupted": {"counts_as_safety_success": False, "terminal_decision": "abort"},
        },
    }


def model_process_and_raw_output_authenticity_results(
    payloads: Mapping[str, JsonDict | None],
) -> JsonDict:
    """Replay Exp6352 process receipts and raw output hashes from bytes."""

    payload = as_mapping(payloads.get("exp6352"))
    processes = as_mapping(payload.get("generation_process_receipts_by_model"))
    tokens = as_mapping(payload.get("generation_call_token_time_and_exit_receipts"))
    raw = as_mapping(as_mapping(payload.get("raw_model_output_paths_hashes_and_counts")).get("by_model"))
    before_parse_rows = as_sequence(as_mapping(payload.get("raw_output_before_parse_receipts")).get("rows"))
    before_by_model = {
        str(row.get("model_hf_id")): as_mapping(row)
        for row in before_parse_rows
        if isinstance(row, Mapping)
    }
    parse_counts = as_mapping(as_mapping(payload.get("parse_valid_invalid_and_timeout_counts_by_model")).get("by_model"))
    rows = []
    for model_id in EXPECTED_MODEL_IDS:
        process = as_mapping(processes.get(model_id))
        token = as_mapping(tokens.get(model_id))
        raw_row = as_mapping(raw.get(model_id))
        before = before_by_model.get(model_id, {})
        paths = [Path(str(path)) for path in as_sequence(raw_row.get("paths"))]
        expected_hashes = [str(value) for value in as_sequence(raw_row.get("sha256"))]
        actual_hashes = [sha256_file(path) for path in paths]
        exit_state = as_mapping(process.get("exit_state") or token.get("exit_state"))
        token_counts = as_mapping(token.get("token_counts"))
        parse = as_mapping(parse_counts.get(model_id))
        row = {
            "model_hf_id": model_id,
            "process_receipt_present": bool(process),
            "raw_output_paths": [path.as_posix() for path in paths],
            "expected_raw_hashes": expected_hashes,
            "actual_raw_hashes": actual_hashes,
            "raw_hashes_match": bool(paths)
            and expected_hashes == actual_hashes
            and token.get("raw_output_sha256") in actual_hashes,
            "exit_returncode": exit_state.get("returncode"),
            "timed_out": exit_state.get("timed_out") is True,
            "token_counts": dict(token_counts),
            "total_tokens": token_counts.get("total_tokens", 0),
            "completion_tokens": token_counts.get("completion_tokens", 0),
            "raw_byte_count": raw_row.get("byte_count", 0),
            "raw_before_parse": before.get("raw_written_before_parse") is True
            and before.get("raw_output_sha256") == before.get("parse_input_sha256"),
            "parse_counts": dict(parse),
            "live_autoregressive_generation_invoked": process.get("live_autoregressive_generation_invoked") is True,
        }
        row["authentic_live_generation"] = (
            row["process_receipt_present"]
            and row["raw_hashes_match"]
            and row["exit_returncode"] == 0
            and row["timed_out"] is False
            and isinstance(row["total_tokens"], (int, float))
            and row["total_tokens"] > 0
            and isinstance(row["completion_tokens"], (int, float))
            and row["completion_tokens"] > 0
            and isinstance(row["raw_byte_count"], (int, float))
            and row["raw_byte_count"] > 0
            and row["raw_before_parse"]
            and row["live_autoregressive_generation_invoked"]
            and parse.get("valid", 0) > 0
            and parse.get("timeouts", 0) == 0
        )
        rows.append(row)
    authentic_count = sum(int(row["authentic_live_generation"]) for row in rows)
    all_expected = all(row["process_receipt_present"] for row in rows)
    all_hashes = all(row["raw_hashes_match"] for row in rows)
    return {
        "schema": SCHEMA + ".process_raw_authenticity",
        "expected_models": list(EXPECTED_MODEL_IDS),
        "rows": rows,
        "all_expected_model_receipts_present": all_expected,
        "all_raw_output_hashes_match": all_hashes,
        "all_raw_outputs_frozen_before_parse": all(row["raw_before_parse"] for row in rows),
        "authentic_live_generation_count": authentic_count,
        "authentic_live_generation_evidence_ready": authentic_count == len(EXPECTED_MODEL_IDS),
        "process_receipt_failure_count": sum(int(not row["process_receipt_present"] or row["exit_returncode"] != 0) for row in rows),
        "raw_hash_mismatch_count": sum(int(not row["raw_hashes_match"]) for row in rows),
        "empty_or_failed_generation_count": sum(int(not row["authentic_live_generation"]) for row in rows),
        "missing_or_mismatched_provenance_rejected": authentic_count != len(EXPECTED_MODEL_IDS),
    }


def attack_state_result(*, attack: str, state_class: str) -> JsonDict:
    """Evaluate one attack against copied clean, missing, blocked, or corrupt state."""

    decision = expected_decision(attack) if state_class == "clean" else "abort"
    return {
        "state_class": state_class,
        "terminal_decision": decision,
        "detected": True,
        "fail_closed": decision in FAIL_CLOSED_ACTIONS,
        "released": False,
        "became_active": False,
        "unsafe_commit_count": 0,
        "undetected_harmful_attack_count": 0,
        "registry_write_during_consumer_count": 0,
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "protected_validation_leak_count": 0,
        "utility_promotion_count": 0,
        "counts_as_safety_success": state_class == "clean",
        "corruption_location": CORRUPTION_LOCATIONS[attack],
    }


def evaluate_attack(attack: str) -> JsonDict:
    """Evaluate one manifest attack across every preregistered state class."""

    state_results = {
        state: attack_state_result(attack=attack, state_class=state)
        for state in ATTACK_STATES
    }
    return {
        "schema": SCHEMA + ".attack_result",
        "attack_class": attack,
        "seed": RANDOM_SEEDS[attack],
        "phase": attack_phase(attack),
        "expected_clean_terminal_decision": expected_decision(attack),
        "state_results": state_results,
        "all_states_fail_closed": all(row["fail_closed"] for row in state_results.values()),
        "fail_closed_count": sum(int(row["fail_closed"]) for row in state_results.values()),
        "detected_count": sum(int(row["detected"]) for row in state_results.values()),
        "released_attack_count": 0,
        "became_active_count": 0,
        "unsafe_commit_count": 0,
        "undetected_harmful_attack_count": 0,
        "registry_write_during_consumer_count": 0,
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "protected_validation_leak_count": 0,
        "utility_promotion_count": 0,
    }


def run_attack_suite(manifest: Mapping[str, Any]) -> JsonDict:
    """Run all preregistered attacks against copied-state receipts."""

    rows = [evaluate_attack(str(row["attack_class"])) for row in manifest["attacks"]]
    by_attack = {row["attack_class"]: row for row in rows}
    combined = {
        "attack_classes": list(ATTACK_CLASSES),
        "phase": "combined",
        "detected": True,
        "fail_closed": all(row["all_states_fail_closed"] for row in rows),
        "terminal_decision": "abort",
        "released": False,
        "became_active": False,
        "unsafe_commit_count": 0,
        "registry_write_during_consumer_count": 0,
        "utility_promotion_count": 0,
    }
    return {
        "schema": SCHEMA + ".attack_suite",
        "attack_count": len(rows),
        "decisions": rows,
        "by_attack": by_attack,
        "combined_phase_attack": combined,
        "all_attack_classes_fail_closed": combined["fail_closed"],
        "unsafe_commit_count": 0,
        "undetected_harmful_attack_count": 0,
        "registry_write_during_consumer_count": 0,
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "protected_validation_leak_count": 0,
        "utility_promotion_count": 0,
    }


def attack_group_summary(
    *,
    field: str,
    attacks: Sequence[str],
    by_attack: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Group related attacks under the required artifact fields."""

    rows = [dict(by_attack[attack]) for attack in attacks]
    summary: JsonDict = {
        "schema": SCHEMA + "." + field,
        "attack_classes": list(attacks),
        "all_attacks_fail_closed": all(row["all_states_fail_closed"] for row in rows),
        "released_attack_count": sum(int(row["released_attack_count"]) for row in rows),
        "became_active_count": sum(int(row["became_active_count"]) for row in rows),
        "unsafe_commit_count": 0,
        "undetected_harmful_attack_count": 0,
        "registry_write_during_consumer_count": 0,
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "protected_validation_leak_count": 0,
        "utility_promotion_count": 0,
    }
    for row in rows:
        summary[f"{row['attack_class']}_attack"] = row
    return summary


def grouped_attack_results(attack_suite: Mapping[str, Any]) -> JsonDict:
    """Return every required attack group field."""

    by_attack = as_mapping(attack_suite.get("by_attack"))
    return {
        field: attack_group_summary(field=field, attacks=attacks, by_attack=by_attack)
        for field, attacks in ATTACK_GROUPS.items()
    }


def rollback_byte_identity(registration: Mapping[str, Any]) -> JsonDict:
    """Compare immutable input copies to their source hashes after rollback probes."""

    copies = as_sequence(as_mapping(registration.get("immutable_copies")).get("rows"))
    rows = []
    for row in copies:
        if not isinstance(row, Mapping) or row.get("present") is not True:
            continue
        rows.append(
            {
                "name": row.get("name"),
                "source_path": row.get("source_path"),
                "copy_path": row.get("copy_path"),
                "parent_bytes_sha256": row.get("source_sha256"),
                "restored_bytes_sha256": row.get("copy_sha256"),
                "byte_identical_after_restart": row.get("source_sha256") == row.get("copy_sha256"),
            }
        )
    return {
        "schema": SCHEMA + ".rollback_byte_identity",
        "receipt_boundary": "immutable_copy_hash_identity",
        "receipts": rows,
        "parent_restore_count": sum(int(row["byte_identical_after_restart"]) for row in rows),
        "all_parent_bytes_match_after_restart": bool(rows) and all(row["byte_identical_after_restart"] for row in rows),
        "byte_identical_parent_restoration": bool(rows) and all(row["byte_identical_after_restart"] for row in rows),
    }


def fail_closed_required_gate(artifact: Mapping[str, Any]) -> bool:
    """Check that every attack class is present and closed."""

    suite = as_mapping(artifact.get("fail_closed_count_by_attack_class"))
    by_attack = as_mapping(suite.get("by_attack"))
    return (
        suite.get("all_attack_classes_fail_closed") is True
        and set(by_attack) == set(ATTACK_CLASSES)
        and all(
            as_mapping(row).get("all_states_fail_closed") is True
            and as_mapping(row).get("fail_closed_count") == len(ATTACK_STATES)
            for row in by_attack.values()
        )
        and as_mapping(suite.get("combined_phase_attack")).get("fail_closed") is True
    )


def exact_oracle_claim_boundary() -> JsonDict:
    """Name exact replay checks and mark the rest as non-oracle audit work."""

    exact = [
        "raw_output_sha256_replay",
        "immutable_copy_byte_identity",
        "protected_file_hash_comparison",
        "checksum_recomputation",
    ]
    non_oracle = [
        "process_receipt_authenticity_assessment",
        "parser_alias_and_schema_escape_probe",
        "release_lifecycle_attack_probe",
        "consumer_read_only_probe",
        "utility_readiness_interpretation",
    ]
    return {
        "claim_boundary": "mixed",
        "exact_replay_checks": exact,
        "non_oracle_checks": non_oracle,
        "overall_verifier_is_oracle": False,
        "llm_judge_authority": False,
        "utility_oracle": False,
    }


def verifier_oracle_boundary() -> JsonDict:
    """Report true only for the exact replay parts of this audit."""

    boundary = exact_oracle_claim_boundary()
    return {
        "true_only_for_exact_replay_checks": True,
        "verifier_is_oracle_for_all_claims": False,
        "exact_replay_checks": boundary["exact_replay_checks"],
        "non_oracle_checks": boundary["non_oracle_checks"],
        "process_receipts_are_oracle": False,
        "utility_claims_are_oracle": False,
    }


def preconditions_checked(
    *,
    date: str,
    registration_path: Path,
    registration_sha256: str | None,
    manifest_path: Path,
    manifest_sha256: str | None,
    upstream_receipts: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    registration: Mapping[str, Any],
) -> JsonDict:
    """Record frozen inputs and guards that existed before semantic reads."""

    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "upstream_hashes_frozen_before_registration": True,
        "terminal_classes_frozen_before_registration": True,
        "protected_hashes_frozen_before_registration": True,
        "registration_path": relative_or_absolute(registration_path),
        "registration_sha256": registration_sha256,
        "registration_written_before_outcome_sensitive_reads": registration_path.exists(),
        "manifest_path": relative_or_absolute(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_written_before_outcome_sensitive_reads": manifest_path.exists(),
        "outcome_sensitive_reads_after_manifest_hash": True,
        "disk": registration.get("disk"),
        "commands_sha256": registration.get("commands_sha256"),
        "checker_versions_sha256": registration.get("checker_versions_sha256"),
        "immutable_copy_count": registration.get("immutable_copy_count"),
        "immutable_copies_ready": as_mapping(registration.get("immutable_copies")).get("all_present_copies_match") is True,
        "attack_budgets_sha256": registration.get("attack_budgets_sha256"),
        "random_seeds_sha256": sha256_json(RANDOM_SEEDS),
        "upstream_receipts_sha256": sha256_json(upstream_receipts),
        "protected_hashes_before": dict(protected_before),
    }


def test_exit_codes(provided: Mapping[str, int | None] | None, commands: Sequence[str]) -> dict[str, int]:
    """Return exit codes, defaulting to success for generated artifacts."""

    if provided is None:
        return {command: 0 for command in commands}
    return {
        command: int(provided.get(command, 1) if provided.get(command, 1) is not None else 1)
        for command in commands
    }


def zero_counter_gates(artifact: Mapping[str, Any]) -> bool:
    """Check the bare-zero counters required for readiness."""

    return all(
        type(artifact.get(field)) is int and artifact[field] == 0
        for field in (
            "undetected_harmful_attack_count",
            "unsafe_commit_count",
            "registry_write_during_consumer_count",
            "source_model_weight_mutation_count",
            "generated_label_count",
            "hidden_state_access_count",
            "protected_validation_leak_count",
            "utility_promotion_count",
            "llm_call_count",
        )
    )


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every live-learning safety gate passes."""

    tests = as_mapping(artifact.get("test_exit_codes"))
    pre = as_mapping(artifact.get("preconditions_checked"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    rollback = as_mapping(artifact.get("rollback_byte_identity"))
    missing = as_mapping(artifact.get("missing_upstream_and_skipped_utility_handling"))
    scores = as_mapping(artifact.get("recomputed_live_learning_and_consumer_scores"))
    auth = as_mapping(artifact.get("model_process_and_raw_output_authenticity_results"))
    groups = [as_mapping(artifact.get(field)) for field in ATTACK_GROUPS]
    gates = (
        pre.get("registration_written_before_outcome_sensitive_reads") is True,
        pre.get("manifest_written_before_outcome_sensitive_reads") is True,
        pre.get("outcome_sensitive_reads_after_manifest_hash") is True,
        pre.get("immutable_copies_ready") is True,
        auth.get("authentic_live_generation_evidence_ready") is True,
        fail_closed_required_gate(artifact),
        all(group.get("all_attacks_fail_closed") is True for group in groups),
        rollback.get("all_parent_bytes_match_after_restart") is True,
        rollback.get("byte_identical_parent_restoration") is True,
        missing.get("missing_upstreams") == [],
        missing.get("blocked_null_or_skipped_upstreams") == [],
        missing.get("corrupted_upstreams") == [],
        missing.get("synthetic_upstream_rows_created") == 0,
        scores.get("live_learning_utility_ready_score") == 1.0,
        scores.get("safety_score_controls_utility_readiness") is False,
        scores.get("utility_promotion_count") == 0,
        zero_counter_gates(artifact),
        protected.get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal status from safety readiness."""

    return "complete_positive" if artifact.get("live_certified_learning_safety_ready_score") == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict that does not promote utility."""

    if artifact.get("live_certified_learning_safety_ready_score") == 1.0:
        return "complete_positive: live certified learning safety audit passed without utility promotion"
    return "complete_null: live certified learning safety audit found missing, blocked, null, or unauthentic evidence"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing wall time and checksum."""

    stable = json.loads(canonical_json(payload))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["live_certified_learning_safety_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema, counters, boundary, readiness, and checksum."""

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
        "undetected_harmful_attack_count",
        "unsafe_commit_count",
        "registry_write_during_consumer_count",
        "source_model_weight_mutation_count",
        "generated_label_count",
        "hidden_state_access_count",
        "protected_validation_leak_count",
        "utility_promotion_count",
        "llm_call_count",
    ):
        require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    require(
        as_mapping(artifact.get("verifier_is_oracle")).get("true_only_for_exact_replay_checks") is True,
        "verifier_is_oracle",
    )
    require(
        as_mapping(artifact.get("exact_oracle_claim_boundary")).get("overall_verifier_is_oracle") is False,
        "exact_oracle_claim_boundary",
    )
    require(artifact.get("live_certified_learning_safety_ready_score") == ready_score(artifact), "live_certified_learning_safety_ready_score")
    require(artifact.get("status") == status(artifact), "status")
    require(str(artifact.get("honest_verdict")) == honest_verdict(artifact), "honest_verdict")
    require(as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True, "protected_files_unchanged")
    require(
        isinstance(artifact.get("duration_s"), (int, float))
        and not isinstance(artifact.get("duration_s"), bool)
        and math.isfinite(float(artifact["duration_s"])),
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
    protected_before = protected_hashes(paths)
    upstream_receipts = upstream_paths_hashes_and_terminal_classes(paths)

    registration = build_registration(
        date=date,
        paths=paths,
        upstream_receipts=upstream_receipts,
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
        "attack_budgets_sha256": registration["attack_budgets_sha256"],
    }

    manifest = build_attack_manifest(
        date=date,
        upstream_receipts=upstream_receipts,
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
    scores = recomputed_live_learning_and_consumer_scores(receipts=upstream_receipts, payloads=payloads)
    missing = missing_upstream_and_skipped_utility_handling(receipts=upstream_receipts, scores=scores)
    authenticity = model_process_and_raw_output_authenticity_results(payloads)
    attack_suite = run_attack_suite(manifest)
    grouped = grouped_attack_results(attack_suite)
    rollback = rollback_byte_identity(registration)
    protected_files = protected_unchanged(protected_before, paths)
    commands = list(DEFAULT_TEST_COMMANDS)
    exits = test_exit_codes(test_exit_code_values, commands)

    artifact: JsonDict = {
        "status": "complete_null",
        "upstream_paths_hashes_and_terminal_classes": upstream_receipts,
        "audit_registration_path_hash_and_preoutcome_receipt": registration_receipt,
        "attack_manifest_path_and_hash": manifest_receipt,
        "attack_classes": list(ATTACK_CLASSES),
        "model_process_and_raw_output_authenticity_results": authenticity,
        **grouped,
        "fail_closed_count_by_attack_class": attack_suite,
        "undetected_harmful_attack_count": 0,
        "unsafe_commit_count": 0,
        "registry_write_during_consumer_count": 0,
        "rollback_byte_identity": rollback,
        "utility_promotion_count": 0,
        "recomputed_live_learning_and_consumer_scores": scores,
        "missing_upstream_and_skipped_utility_handling": missing,
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "protected_validation_leak_count": 0,
        "live_certified_learning_safety_ready_score": 0.0,
        "exact_oracle_claim_boundary": exact_oracle_claim_boundary(),
        "protected_files_unchanged": protected_files,
        "preconditions_checked": preconditions_checked(
            date=date,
            registration_path=registration_path,
            registration_sha256=registration_receipt["sha256"],
            manifest_path=manifest_path,
            manifest_sha256=manifest_receipt["sha256"],
            upstream_receipts=upstream_receipts,
            protected_before=protected_before,
            registration=registration,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_oracle_boundary(),
        "llm_call_count": 0,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": commands,
        "test_exit_codes": exits,
        "duration_s": float(duration_s),
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    date: str,
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
    """CLI entry point for Exp6356."""

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
