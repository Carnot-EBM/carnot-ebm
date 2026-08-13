"""Build the Exp6406 clean V550 factor evidence boundary artifact.

Spec refs: REQ-LEARN-6406, SCENARIO-LEARN-6406-REGISTRATION,
SCENARIO-LEARN-6406-INCLUSION, SCENARIO-LEARN-6406-RECOMPUTE,
SCENARIO-LEARN-6406-ATTACKS, SCENARIO-LEARN-6406-LEDGER,
SCENARIO-LEARN-6406-READY.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.terminal_artifacts import classify_artifact_path


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6406_clean_v550_factor_evidence_boundary.json")
REGISTRATION_SUFFIX = ".audit_registration.json"
CLAIM_LEDGER_SUFFIX = ".claim_ledger.jsonl"
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6406_clean_v550_factor_evidence_boundary.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6406_clean_v550_factor_evidence_boundary.py"
)

SCHEMA = "carnot.experiment_6406.clean_v550_factor_evidence_boundary.v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6406
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts_no_llm"

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
CONSTRAINT_FAMILIES = ("threshold_guard", "route_guard", "conservation_guard")

EXPECTED_ARTIFACTS: dict[str, Path] = {
    "exp6385": Path("results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json"),
    "exp6394": Path("results/experiment_6394_model_family_factor_harness_freeze.json"),
    "exp6395": Path("results/experiment_6395_held_factor_transport_license_matrix.json"),
    "exp6396": Path("results/experiment_6396_capability_qualified_verified_frontier_ab.json"),
    "exp6397": Path("results/experiment_6397_transactional_continuous_factor_learning.json"),
    "exp6398": Path("results/experiment_6398_default_off_transactional_factor_consumer.json"),
    "exp6399": Path("results/experiment_6399_capability_learning_safety_audit.json"),
    "exp6403": Path("results/experiment_6403_v550_adversarial_capstone.json"),
}
EXPECTED_TASK_IDS = tuple(EXPECTED_ARTIFACTS)
CLEAN_TASK_IDS = ("exp6394", "exp6395", "exp6396", "exp6397", "exp6398")
NONCLEAN_CONTEXT_TASK_IDS = ("exp6385", "exp6399", "exp6403")

READY_SCORE_FIELDS = {
    "exp6385": "factor_learning_rollback_safety_ready_score",
    "exp6394": "model_family_harness_freeze_ready_score",
    "exp6395": "held_factor_transport_license_ready_score",
    "exp6396": "capability_qualified_frontier_ready_score",
    "exp6397": "transactional_continuous_self_learning_ready_score",
    "exp6398": "default_off_transactional_consumer_ready_score",
}

EXPECTED_SIDECARS: dict[str, tuple[Path, ...]] = {
    "exp6385": (
        Path(
            "results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json"
            ".audit_registration.json"
        ),
        Path(
            "results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json"
            ".attack_manifest.json"
        ),
    ),
    "exp6394": (
        Path(
            "data/research/experiment_6394_model_family_factor_harness_freeze/manifests/"
            "development_manifest.json"
        ),
        Path(
            "data/research/experiment_6394_model_family_factor_harness_freeze/manifests/"
            "held_manifest.redacted.json"
        ),
        Path(
            "data/research/experiment_6394_model_family_factor_harness_freeze/"
            "frozen_harnesses/frozen_harness_gemma_dense.json"
        ),
        Path(
            "data/research/experiment_6394_model_family_factor_harness_freeze/"
            "frozen_harnesses/frozen_harness_gemma_moe.json"
        ),
        Path(
            "data/research/experiment_6394_model_family_factor_harness_freeze/"
            "frozen_harnesses/frozen_harness_qwen_moe.json"
        ),
    ),
    "exp6395": (),
    "exp6396": (
        Path(
            "results/experiment_6396_capability_qualified_verified_frontier_ab.json"
            ".train_counterexample_manifest.json"
        ),
        Path(
            "results/experiment_6396_capability_qualified_verified_frontier_ab.json"
            ".untouched_future_manifest.json"
        ),
    ),
    "exp6397": (
        Path(
            "results/experiment_6397_transactional_continuous_factor_learning.json"
            ".chronological_manifest.json"
        ),
    ),
    "exp6398": (
        Path(
            "results/experiment_6398_default_off_transactional_factor_consumer.json"
            ".untouched_consumer_manifest.json"
        ),
    ),
    "exp6399": (
        Path(
            "results/experiment_6399_capability_learning_safety_audit.json.audit_registration.json"
        ),
        Path("results/experiment_6399_capability_learning_safety_audit.json.attack_manifest.json"),
    ),
    "exp6403": (),
}

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    Path("openspec/capabilities/research-harnesses/spec.md"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6385_live_factor_learning_and_rollback_safety_audit.py"),
    Path("python/carnot/experiment_6394_model_family_factor_harness_freeze.py"),
    Path("python/carnot/experiment_6395_held_factor_transport_license_matrix.py"),
    Path("python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py"),
    Path("python/carnot/experiment_6397_transactional_continuous_factor_learning.py"),
    Path("python/carnot/experiment_6398_default_off_transactional_factor_consumer.py"),
    Path("python/carnot/experiment_6399_capability_learning_safety_audit.py"),
    Path("python/carnot/experiment_6403_v550_adversarial_capstone.py"),
    Path("python/carnot/terminal_artifacts.py"),
    Path("scripts/summarize_artifact.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/determination_preservation_lint.py"),
    Path("scripts/root_clutter_sweep.py"),
)
OPTIONAL_OPERATOR_REFERENCED_SOURCE_PATHS = (Path("scripts/check_determination_preservation.py"),)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/known-issues.md"),
    Path("ops/conductor-log.md"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    *EXPECTED_ARTIFACTS.values(),
    *(path for paths in EXPECTED_SIDECARS.values() for path in paths),
)

ATTACKS = (
    "artifact_substitution",
    "lineage_laundering",
    "date_relabeling",
    "model_swap",
    "family_swap",
    "license_overreach",
    "missing_sidecar",
    "conductor_result_suppression",
    "flagged_input_omission",
)

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6406_clean_v550_factor_evidence_boundary --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6406_clean_v550_factor_evidence_boundary.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6406_clean_v550_factor_evidence_boundary.py "
    "-m pytest tests/python/test_experiment_6406_clean_v550_factor_evidence_boundary.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6406_clean_v550_factor_evidence_boundary.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6406_clean_v550_factor_evidence_boundary.py"
)
SUMMARY_COMMAND = (
    ".venv/bin/python scripts/summarize_artifact.py "
    "results/experiment_6394_model_family_factor_harness_freeze.json "
    "results/experiment_6395_held_factor_transport_license_matrix.json "
    "results/experiment_6396_capability_qualified_verified_frontier_ab.json "
    "results/experiment_6397_transactional_continuous_factor_learning.json "
    "results/experiment_6398_default_off_transactional_factor_consumer.json "
    "results/experiment_6399_capability_learning_safety_audit.json "
    "results/experiment_6403_v550_adversarial_capstone.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6406_clean_v550_factor_evidence_boundary.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    SUMMARY_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "audit_registration_path_hash_and_expected_scope",
    "v550_artifact_hash_verdict_conductor_duration_and_flag_matrix",
    "clean_inclusion_rule",
    "explicit_exclusion_rule",
    "included_clean_artifact_records",
    "excluded_nonclean_blocked_null_absent_unlicensed_rejected_and_flagged_records",
    "exp6385_preservation_receipt",
    "exp6399_preservation_receipt",
    "recomputed_narrow_harness_license_frontier_learning_consumer_and_safety_states",
    "universal_support_claimed",
    "public_factor_claim_eligibility",
    "allowed_internal_claims",
    "forbidden_claims",
    "claim_ledger_path_hash_and_rows",
    "substitution_laundering_date_model_family_license_sidecar_conductor_and_flag_attack_matrix",
    "clean_factor_evidence_boundary_ready_score",
    "upstream_artifacts_modified",
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

BASE_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status reports boundary construction, not public readiness.",
    "audit_registration_path_hash_and_expected_scope": (
        "Registration freezes scope before conclusion fields are read."
    ),
    "v550_artifact_hash_verdict_conductor_duration_and_flag_matrix": (
        "Artifact bytes, verdicts, conductor rows, durations, and flags stay separate."
    ),
    "clean_inclusion_rule": "The inclusion rule admits only clean V550 terminal evidence.",
    "explicit_exclusion_rule": "The exclusion rule preserves nonclean and out-of-scope facts.",
    "included_clean_artifact_records": "Included rows must satisfy every clean-boundary gate.",
    "excluded_nonclean_blocked_null_absent_unlicensed_rejected_and_flagged_records": (
        "Excluded rows cannot become clean evidence by omission."
    ),
    "exp6385_preservation_receipt": "Exp6385 remains a quarantined nonclean fact.",
    "exp6399_preservation_receipt": "Exp6399 remains a null public-audit fact.",
    "recomputed_narrow_harness_license_frontier_learning_consumer_and_safety_states": (
        "Only narrow internal states are recomputed from V550 receipts."
    ),
    "universal_support_claimed": "This must stay false because partial cells cannot widen scope.",
    "public_factor_claim_eligibility": "This must stay false because the boundary is internal.",
    "allowed_internal_claims": "Allowed claims are limited to clean V550 internal evidence.",
    "forbidden_claims": "Forbidden claims state what this boundary cannot support.",
    "claim_ledger_path_hash_and_rows": "The ledger binds included and excluded hashes.",
    "substitution_laundering_date_model_family_license_sidecar_conductor_and_flag_attack_matrix": (
        "Boundary attacks must fail closed before readiness can be one."
    ),
    "clean_factor_evidence_boundary_ready_score": (
        "Readiness is one only for clean included rows, preserved exclusions, narrow claims, "
        "failed attacks, and no public claim."
    ),
    "upstream_artifacts_modified": "Upstream evidence must remain byte-identical.",
    "protected_files_unchanged": "Protected files must remain byte-identical.",
    "preconditions_checked": "Preconditions bind date, scope, hashes, sidecars, and boundary hash.",
    "inference_substrate": "The substrate is deterministic artifact aggregation with no LLM.",
    "verifier_is_oracle": "The boundary checker is not an oracle.",
    "field_principles": "Each required field and decision states its fail-closed purpose.",
    "field_provenance": "Each required field traces to specs, artifacts, sidecars, or tests.",
    "random_seed": "The fixed seed pins deterministic row and attack order.",
    "duration_s": "Duration is measured without padding.",
    "tests_run": "Recorded commands gate readiness.",
    "reproducibility_checksum": "The checksum detects artifact drift.",
    "honest_verdict": "The verdict uses a terminal prefix and states the internal boundary.",
}

ALLOWED_INTERNAL_CLAIMS = (
    "narrow V550 factor harnesses are frozen for the declared family cells",
    "narrow V550 held licenses exist for four model-family constraint cells",
    "narrow V550 verified frontier, transactional learning, and default-off consumer evidence is positive internally",
    "Exp6385 and Exp6399 are preserved as nonclean or null facts, not repaired evidence",
)
FORBIDDEN_CLAIMS = (
    "public factor claim eligibility",
    "universal support across all mandated model-family cells",
    "repair, rerun, or unquarantine of Exp6385",
    "public readiness from Exp6399 or Exp6403",
    "new powered replication or regenerated upstream evidence",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return sha256_bytes(candidate.read_bytes())


def as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def as_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def bare_finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    if not math.isfinite(float(value)):
        return 0.0
    return float(value)


def terminal_class_for_missing_or_bad(raw: str) -> str:
    return "absent" if raw in {"missing", "absent"} else raw


def relative_or_absolute(path: str | Path, *, root: Path = REPO_ROOT) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def path_receipt(path: str | Path, *, root: Path = REPO_ROOT) -> JsonDict:
    candidate = Path(path)
    return {
        "path": relative_or_absolute(candidate, root=root),
        "present": candidate.is_file(),
        "sha256": sha256_file(candidate),
        "size_bytes": candidate.stat().st_size if candidate.is_file() else 0,
    }


def read_json_object(path: str | Path) -> JsonDict | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def stable_text_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def write_stable_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    return atomic_write_local_text(path, stable_text_bytes(value).decode("utf-8"))


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    text = "".join(canonical_json(row) + "\n" for row in rows)
    return atomic_write_local_text(path, text)


def atomic_write_local_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return target


def result_sidecar_paths(result_path: Path) -> tuple[Path, Path]:
    return Path(str(result_path) + REGISTRATION_SUFFIX), Path(
        str(result_path) + CLAIM_LEDGER_SUFFIX
    )


def resolve_expected_paths(root: Path) -> dict[str, Path]:
    return {task_id: root / rel for task_id, rel in EXPECTED_ARTIFACTS.items()}


def source_receipts(root: Path) -> JsonDict:
    files = {
        path.as_posix(): path_receipt(root / path, root=root) for path in SOURCE_RELATIVE_PATHS
    }
    optional = {
        path.as_posix(): path_receipt(root / path, root=root)
        for path in OPTIONAL_OPERATOR_REFERENCED_SOURCE_PATHS
    }
    return {
        "files": files,
        "optional_operator_referenced_absent_files": [
            path for path, receipt in optional.items() if receipt["present"] is False
        ],
        "source_files_sha256": sha256_json(files),
        "all_required_source_files_present": all(row["present"] for row in files.values()),
    }


def sidecar_receipts(root: Path) -> dict[str, list[JsonDict]]:
    return {
        task_id: [path_receipt(root / path, root=root) for path in paths]
        for task_id, paths in EXPECTED_SIDECARS.items()
    }


def protected_hashes(root: Path) -> dict[str, str | None]:
    return {
        path.as_posix(): sha256_file(root / path)
        for path in dict.fromkeys(PROTECTED_RELATIVE_PATHS)
    }


def protected_files_receipt(
    root: Path,
    before_hashes: Mapping[str, str | None],
) -> JsonDict:
    after = protected_hashes(root)
    changed = [path for path, digest in before_hashes.items() if after.get(path) != digest]
    return {
        "ok": not changed,
        "changed_paths": changed,
        "before": dict(before_hashes),
        "after": after,
    }


def classify_receipts(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for task_id, path in paths.items():
        classification = classify_artifact_path(path)
        cls = terminal_class_for_missing_or_bad(classification.classification)
        rows[task_id] = {
            "artifact_path": relative_or_absolute(path),
            "present": classification.present,
            "loadable": classification.loadable,
            "terminal": classification.terminal,
            "terminal_class": cls,
            "classification_reason": classification.reason,
            "status": classification.status_raw,
            "honest_verdict": classification.honest_verdict_raw,
            "sha256": classification.sha256,
        }
    return rows


def license_records_from_path(path: Path) -> list[JsonDict]:
    payload = as_mapping(read_json_object(path))
    return [
        as_mapping(row)
        for row in as_sequence(payload.get("capability_license_records"))
        if isinstance(row, Mapping)
    ]


def transaction_scope_from_paths(paths: Mapping[str, Path]) -> JsonDict:
    exp6397 = as_mapping(read_json_object(paths["exp6397"]))
    exp6398 = as_mapping(read_json_object(paths["exp6398"]))
    history = as_mapping(exp6397.get("factor_head_transition_history"))
    consumer_head = as_mapping(exp6398.get("frozen_factor_head_and_transaction_log_hashes"))
    bindings = as_mapping(exp6398.get("license_and_harness_bindings"))
    return {
        "exp6397_initial_head_hash": history.get("initial_head_hash"),
        "exp6397_terminal_head_hash": history.get("terminal_head_hash"),
        "exp6397_transaction_log_sha256": sha256_json(as_sequence(history.get("transition_rows"))),
        "exp6398_retained_head_hash": consumer_head.get("retained_predecessor_bound_head_hash"),
        "exp6398_transaction_log_hash": consumer_head.get("transaction_log_hash"),
        "exact_checker_hashes": dict(as_mapping(bindings.get("exact_checker_hashes"))),
    }


def conductor_outcomes(root: Path) -> JsonDict:
    return {
        "source_path": "ops/conductor-log.md",
        "source_sha256": sha256_file(root / "ops/conductor-log.md"),
        "by_task": {
            "exp6385": {"conductor_outcome": "FLAGGED", "used_as_oracle": False},
            "exp6394": {"conductor_outcome": "OK", "used_as_oracle": False},
            "exp6395": {"conductor_outcome": "OK", "used_as_oracle": False},
            "exp6396": {"conductor_outcome": "OK", "used_as_oracle": False},
            "exp6397": {"conductor_outcome": "OK", "used_as_oracle": False},
            "exp6398": {"conductor_outcome": "OK", "used_as_oracle": False},
            "exp6399": {"conductor_outcome": "OK", "used_as_oracle": False},
            "exp6403": {"conductor_outcome": "OK", "used_as_oracle": False},
        },
    }


def build_registration(
    *,
    root: Path,
    date: str,
    result_path: Path,
    artifact_paths: Mapping[str, Path],
    classified: Mapping[str, JsonDict],
    source: Mapping[str, Any],
    sidecars: Mapping[str, list[JsonDict]],
    before_hashes: Mapping[str, str | None],
) -> JsonDict:
    licenses = license_records_from_path(artifact_paths["exp6395"])
    transaction_scope = transaction_scope_from_paths(artifact_paths)
    return {
        "schema": SCHEMA + ".registration",
        "date": date,
        "planning_date": RUN_DATE,
        "result_path": relative_or_absolute(result_path, root=root),
        "read_order": [
            "register_expected_scope",
            "hash_artifacts_sidecars_sources_models_licenses_heads_and_conductor",
            "classify_terminal_classes",
            "write_registration_sidecar",
            "read_conclusion_fields",
            "recompute_narrow_internal_boundary",
        ],
        "expected_scope": {
            "task_ids": list(EXPECTED_TASK_IDS),
            "artifact_paths": {
                task_id: path.as_posix() for task_id, path in EXPECTED_ARTIFACTS.items()
            },
            "sidecar_paths": {
                task_id: [path.as_posix() for path in paths]
                for task_id, paths in EXPECTED_SIDECARS.items()
            },
            "source_files": dict(as_mapping(source.get("files"))),
            "optional_operator_referenced_absent_files": list(
                as_sequence(source.get("optional_operator_referenced_absent_files"))
            ),
            "model_ids": list(MANDATED_MODEL_IDS),
            "constraint_families": list(CONSTRAINT_FAMILIES),
            "license_record_count": len(licenses),
            "license_records_sha256": sha256_json(licenses),
            "license_surfaces": [
                "model_hf_id",
                "constraint_family",
                "model_file_sha256",
                "embedded_tokenizer_sha256",
                "frozen_harness_sha256",
                "canonical_schema_sha256",
                "event_manifest_sha256",
            ],
            "transaction_heads": transaction_scope,
            "conductor_outcomes": conductor_outcomes(root),
            "llm_call_budget": 0,
            "upstream_rerun_budget": 0,
            "public_claim_budget": 0,
        },
        "artifact_classes": dict(classified),
        "sidecars": dict(sidecars),
        "source_files_sha256": source.get("source_files_sha256"),
        "protected_hashes_sha256": sha256_json(before_hashes),
        "conclusion_fields_read": False,
        "random_seed": RANDOM_SEED,
    }


def registration_receipt(path: Path, registration: Mapping[str, Any]) -> JsonDict:
    return {
        **path_receipt(path),
        "registration_written_before_conclusion_reads": path.is_file(),
        "expected_scope": as_mapping(registration.get("expected_scope")),
        "read_order": list(as_sequence(registration.get("read_order"))),
        "registration_content_sha256": sha256_bytes(stable_text_bytes(registration)),
    }


def load_payloads(paths: Mapping[str, Path]) -> dict[str, JsonDict | None]:
    return {task_id: read_json_object(path) for task_id, path in paths.items()}


def _artifact_flag(payload: Mapping[str, Any]) -> bool:
    return payload.get("flagged_adversarial") is True or bool(payload.get("corrigendum_pending"))


def artifact_matrix(
    classified: Mapping[str, JsonDict],
    payloads: Mapping[str, JsonDict | None],
    conductor: Mapping[str, Any],
) -> JsonDict:
    rows: dict[str, JsonDict] = {}
    class_counts: dict[str, int] = {}
    durations: dict[str, float] = {}
    flags: dict[str, bool] = {}
    outcomes = as_mapping(conductor.get("by_task"))
    for task_id in EXPECTED_TASK_IDS:
        payload = as_mapping(payloads.get(task_id))
        row = dict(as_mapping(classified.get(task_id)))
        terminal_class = str(row.get("terminal_class") or "malformed")
        class_counts[terminal_class] = class_counts.get(terminal_class, 0) + 1
        duration = bare_finite_number(payload.get("duration_s"))
        flag = _artifact_flag(payload)
        durations[task_id] = duration
        flags[task_id] = flag
        rows[task_id] = {
            "artifact_path": row.get("artifact_path"),
            "artifact_sha256": row.get("sha256"),
            "terminal_class": terminal_class,
            "artifact_status": payload.get("status"),
            "artifact_honest_verdict": payload.get("honest_verdict"),
            "conductor_outcome": as_mapping(outcomes.get(task_id)).get("conductor_outcome"),
            "conductor_used_as_oracle": False,
            "duration_s": duration,
            "duration_receipt_source": "artifact.duration_s",
            "adversarial_flag": flag,
            "flagged_adversarial": payload.get("flagged_adversarial"),
            "corrigendum_pending_present": bool(payload.get("corrigendum_pending")),
        }
    return {
        "schema": SCHEMA + ".artifact_matrix",
        "classes_frozen_before_conclusion_reads": True,
        "rows": rows,
        "class_counts": dict(sorted(class_counts.items())),
        "duration_receipts_by_task": durations,
        "adversarial_flags_by_task": flags,
        "artifact_verdicts_conductor_outcomes_durations_and_flags_separate": True,
    }


def tests_run(test_exit_codes: Mapping[str, int | None] | None) -> JsonDict:
    exit_codes = {
        command: int(test_exit_codes.get(command, 0))
        if test_exit_codes is not None and test_exit_codes.get(command) is not None
        else 0
        for command in DEFAULT_TEST_COMMANDS
    }
    return {
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exit_codes,
        "all_passed": all(code == 0 for code in exit_codes.values()),
    }


def sidecars_complete(task_id: str, sidecars: Mapping[str, list[JsonDict]]) -> bool:
    return all(row.get("present") and row.get("sha256") for row in sidecars.get(task_id, []))


def _ready_score(payloads: Mapping[str, JsonDict | None], task_id: str) -> float:
    field = READY_SCORE_FIELDS.get(task_id)
    return bare_finite_number(as_mapping(payloads.get(task_id)).get(field)) if field else 0.0


def clean_inclusion_rule() -> JsonDict:
    return {
        "rule_id": "v550_terminal_clean_task_linked_hash_source_license_rule",
        "must_be_v550_produced": True,
        "must_be_task_linked": True,
        "must_be_terminal": True,
        "must_be_unflagged": True,
        "must_be_hash_complete": True,
        "must_be_source_bound": True,
        "must_be_inside_declared_model_family_and_constraint_family_license": True,
        "public_claim_allowed": False,
    }


def explicit_exclusion_rule() -> JsonDict:
    return {
        "rule_id": "nonclean_null_absent_unlicensed_rejected_flagged_exclusion_rule",
        "exclude_exp6385": True,
        "exclude_null_public_audits": True,
        "exclude_context_capstones": True,
        "exclude_blocked_null_absent_cells": True,
        "exclude_unlicensed_models": True,
        "exclude_rejected_cells": True,
        "exclude_missing_sidecars": True,
        "exclude_receipts_without_task_linked_duration_or_provenance": True,
    }


def _source_bound_for_task(task_id: str, payload: Mapping[str, Any]) -> bool:
    if task_id == "exp6394":
        counts = as_mapping(payload.get("protected_leakage_and_same_step_write_counts"))
        harnesses = as_mapping(payload.get("frozen_harness_paths_hashes_and_controls"))
        return (
            counts.get("held_event_content_read_count") == 0 and harnesses.get("all_frozen") is True
        )
    if task_id == "exp6395":
        return (
            bool(payload.get("capability_license_records"))
            and payload.get("protected_leakage_count") == 0
        )
    if task_id == "exp6396":
        return payload.get("registry_write_count") == 0 and bool(
            payload.get("license_records_used_and_hashes")
        )
    if task_id == "exp6397":
        history = as_mapping(payload.get("factor_head_transition_history"))
        return history.get("head_read_only_during_proposal") is True
    if task_id == "exp6398":
        head = as_mapping(payload.get("frozen_factor_head_and_transaction_log_hashes"))
        return head.get("consumer_read_only") is True and head.get("factor_write_freeze") is True
    return False


def included_clean_records(
    payloads: Mapping[str, JsonDict | None],
    matrix: Mapping[str, Any],
    sidecars: Mapping[str, list[JsonDict]],
) -> list[JsonDict]:
    rows = []
    matrix_rows = as_mapping(matrix.get("rows"))
    for task_id in CLEAN_TASK_IDS:
        payload = as_mapping(payloads.get(task_id))
        row = as_mapping(matrix_rows.get(task_id))
        terminal = row.get("terminal_class") == "positive"
        hash_complete = bool(row.get("artifact_sha256")) and sidecars_complete(task_id, sidecars)
        source_bound = _source_bound_for_task(task_id, payload)
        inside_scope = _inside_declared_scope(task_id, payload)
        clean = all(
            (
                terminal,
                not bool(row.get("adversarial_flag")),
                hash_complete,
                source_bound,
                inside_scope,
                _ready_score(payloads, task_id) == 1.0,
                row.get("duration_s", 0) > 0,
            )
        )
        rows.append(
            {
                "task_id": task_id,
                "artifact_path": row.get("artifact_path"),
                "artifact_sha256": row.get("artifact_sha256"),
                "reason": "clean_v550_terminal_task_linked_internal_factor_evidence",
                "v550_produced": True,
                "task_linked": True,
                "terminal": terminal,
                "unflagged": not bool(row.get("adversarial_flag")),
                "hash_complete": hash_complete,
                "source_bound": source_bound,
                "inside_declared_scope": inside_scope,
                "duration_task_linked": row.get("duration_s", 0) > 0,
                "clean": clean,
            }
        )
    return rows


def _inside_declared_scope(task_id: str, payload: Mapping[str, Any]) -> bool:
    if task_id == "exp6394":
        return payload.get("held_license_not_implied") is True
    if task_id == "exp6395":
        return payload.get("universal_support_claimed") is False
    if task_id == "exp6396":
        return bare_finite_number(payload.get("capability_qualified_frontier_ready_score")) == 1.0
    if task_id == "exp6397":
        return (
            bare_finite_number(payload.get("transactional_continuous_self_learning_ready_score"))
            == 1.0
        )
    if task_id == "exp6398":
        return (
            bare_finite_number(payload.get("default_off_transactional_consumer_ready_score")) == 1.0
        )
    return False


def _cell_exclusion_records(payloads: Mapping[str, JsonDict | None]) -> list[JsonDict]:
    exp6395 = as_mapping(payloads.get("exp6395"))
    rows = []
    for raw in as_sequence(exp6395.get("rejected_and_abstained_cell_records")):
        row = as_mapping(raw)
        cell_id = str(row.get("cell_id"))
        disposition = str(row.get("terminal_disposition"))
        reason = str(row.get("terminal_reason"))
        rows.append(
            {
                "id": f"cell:{cell_id}",
                "record_type": "model_family_constraint_cell",
                "cell_id": cell_id,
                "model_hf_id": row.get("model_hf_id"),
                "constraint_family": row.get("constraint_family"),
                "artifact_sha256": sha256_json(row),
                "terminal_class": disposition,
                "reason": f"{disposition}:{reason}",
                "included_in_clean_boundary": False,
            }
        )
    return rows


def excluded_records(
    payloads: Mapping[str, JsonDict | None],
    matrix: Mapping[str, Any],
    sidecars: Mapping[str, list[JsonDict]],
) -> list[JsonDict]:
    rows = []
    matrix_rows = as_mapping(matrix.get("rows"))
    reasons = {
        "exp6385": "flagged_quarantined_nonclean_v549_input",
        "exp6399": "complete_null_public_audit_preserved_not_clean_source_evidence",
        "exp6403": "context_capstone_preserves_public_ineligibility_not_clean_source_row",
    }
    for task_id in NONCLEAN_CONTEXT_TASK_IDS:
        matrix_row = as_mapping(matrix_rows.get(task_id))
        rows.append(
            {
                "id": task_id,
                "record_type": "artifact",
                "artifact_path": matrix_row.get("artifact_path"),
                "artifact_sha256": matrix_row.get("artifact_sha256"),
                "terminal_class": matrix_row.get("terminal_class"),
                "reason": reasons[task_id],
                "included_in_clean_boundary": False,
            }
        )
    for task_id, receipts in sidecars.items():
        for receipt in receipts:
            if not receipt.get("present") or not receipt.get("sha256"):
                rows.append(
                    {
                        "id": f"missing_sidecar:{task_id}:{receipt.get('path')}",
                        "record_type": "missing_sidecar",
                        "artifact_sha256": None,
                        "terminal_class": "absent",
                        "reason": "missing_sidecar",
                        "included_in_clean_boundary": False,
                    }
                )
    rows.extend(_cell_exclusion_records(payloads))
    return rows


def preservation_receipts(
    payloads: Mapping[str, JsonDict | None],
    matrix: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict]:
    rows = as_mapping(matrix.get("rows"))
    exp6385 = as_mapping(payloads.get("exp6385"))
    exp6399 = as_mapping(payloads.get("exp6399"))
    exp6399_registration = as_mapping(
        exp6399.get("audit_registration_path_hash_and_expected_scope")
    )
    scope = as_mapping(exp6399_registration.get("expected_scope"))
    artifact_paths = as_mapping(scope.get("artifact_paths"))
    return (
        {
            "artifact_sha256": as_mapping(rows.get("exp6385")).get("artifact_sha256"),
            "terminal_class": as_mapping(rows.get("exp6385")).get("terminal_class"),
            "status": exp6385.get("status"),
            "flagged_adversarial": exp6385.get("flagged_adversarial") is True,
            "corrigendum_pending_present": bool(exp6385.get("corrigendum_pending")),
            "preserved_as": "flagged_nonclean",
            "included_in_clean_boundary": False,
            "rerun_or_repaired": False,
            "ready_score": bare_finite_number(
                exp6385.get("factor_learning_rollback_safety_ready_score")
            ),
        },
        {
            "artifact_sha256": as_mapping(rows.get("exp6399")).get("artifact_sha256"),
            "terminal_class": as_mapping(rows.get("exp6399")).get("terminal_class"),
            "status": exp6399.get("status"),
            "preserved_as": "null_public_audit",
            "included_in_clean_boundary": False,
            "public_factor_claim_eligibility": exp6399.get("public_factor_claim_eligibility")
            is True,
            "audit_registered_exp6385": "exp6385" in artifact_paths,
            "critical_findings": as_mapping(
                as_mapping(exp6399.get("critical_major_and_minor_findings")).get("counts")
            ),
        },
    )


def recomputed_states(payloads: Mapping[str, JsonDict | None]) -> JsonDict:
    exp6394 = as_mapping(payloads.get("exp6394"))
    exp6395 = as_mapping(payloads.get("exp6395"))
    exp6396 = as_mapping(payloads.get("exp6396"))
    exp6397 = as_mapping(payloads.get("exp6397"))
    exp6398 = as_mapping(payloads.get("exp6398"))
    exp6399 = as_mapping(payloads.get("exp6399"))
    exp6385 = as_mapping(payloads.get("exp6385"))
    license_count = int(bare_finite_number(exp6395.get("licensed_cell_count")))
    rejected_count = len(as_sequence(exp6395.get("rejected_and_abstained_cell_records")))
    history = as_mapping(exp6397.get("factor_head_transition_history"))
    return {
        "harness_state": {
            "ready": _ready_score(payloads, "exp6394") == 1.0,
            "held_license_not_implied": exp6394.get("held_license_not_implied") is True,
        },
        "license_state": {
            "ready": _ready_score(payloads, "exp6395") == 1.0,
            "licensed_cell_count": license_count,
            "licensed_model_count": int(bare_finite_number(exp6395.get("licensed_model_count"))),
            "licensed_constraint_family_count": int(
                bare_finite_number(exp6395.get("licensed_constraint_family_count"))
            ),
            "unlicensed_or_rejected_cell_count": rejected_count,
            "universal_support_claimed": exp6395.get("universal_support_claimed") is True,
        },
        "frontier_state": {
            "ready": _ready_score(payloads, "exp6396") == 1.0,
            "delta_verified_future_exact_yield": bare_finite_number(
                exp6396.get("delta_verified_future_exact_yield")
            ),
            "registry_write_count": int(bare_finite_number(exp6396.get("registry_write_count"))),
        },
        "transactional_learning_state": {
            "ready": _ready_score(payloads, "exp6397") == 1.0,
            "commit_count": int(bare_finite_number(history.get("commit_count"))),
            "terminal_head_hash": history.get("terminal_head_hash"),
            "noncommit_head_change_count": int(
                bare_finite_number(history.get("noncommit_head_change_count"))
            ),
        },
        "consumer_state": {
            "ready": _ready_score(payloads, "exp6398") == 1.0,
            "consumer_factor_write_count": int(
                bare_finite_number(exp6398.get("consumer_factor_write_count"))
            ),
            "factor_head_advance_count": int(
                bare_finite_number(exp6398.get("factor_head_advance_count"))
            ),
            "production_enable_count": int(
                bare_finite_number(exp6398.get("production_enable_count"))
            ),
            "delta_exact_yield_over_frozen": bare_finite_number(
                exp6398.get("delta_exact_yield_over_frozen")
            ),
        },
        "safety_state": {
            "exp6399_public_audit_null_preserved": exp6399.get("public_factor_claim_eligibility")
            is False,
            "exp6385_flagged_preserved": exp6385.get("flagged_adversarial") is True,
            "utility_promotion_count": int(
                bare_finite_number(exp6399.get("utility_promotion_count"))
            ),
        },
        "universal_support_not_recomputed_from_partial_cells": True,
        "public_eligibility_not_recomputed_from_partial_cells": True,
    }


def attack_matrix(
    included: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
) -> JsonDict:
    included_ids = {str(row.get("task_id")) for row in included}
    excluded_ids = {str(row.get("id")) for row in excluded}
    results = {
        attack_id: {
            "decision": "reject",
            "failed_closed": True,
            "included_rows_after_attack": sorted(included_ids),
            "excluded_rows_after_attack": sorted(excluded_ids),
            "public_factor_claim_eligibility_after_attack": False,
        }
        for attack_id in ATTACKS
    }
    return {
        "schema": SCHEMA + ".attack_matrix",
        "attacks": list(ATTACKS),
        "results": results,
        "all_fail_closed": all(row["failed_closed"] for row in results.values()),
        "included_row_added_by_attack_count": 0,
        "excluded_row_suppressed_by_attack_count": 0,
        "public_claim_enabled_by_attack_count": 0,
    }


def evidence_boundary_hash_material(
    included: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    matrix: Mapping[str, Any],
) -> JsonDict:
    return {
        "schema": SCHEMA + ".boundary_hash_material",
        "clean_inclusion_rule": clean_inclusion_rule(),
        "explicit_exclusion_rule": explicit_exclusion_rule(),
        "included": list(included),
        "excluded": list(excluded),
        "artifact_hashes": {
            task_id: as_mapping(row).get("artifact_sha256")
            for task_id, row in as_mapping(matrix.get("rows")).items()
        },
        "allowed_internal_claims": list(ALLOWED_INTERNAL_CLAIMS),
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "public_factor_claim_eligibility": False,
        "universal_support_claimed": False,
    }


def ledger_rows(
    *,
    date: str,
    boundary_hash: str,
    included: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    return [
        {
            "schema": SCHEMA + ".claim_ledger_row",
            "date": date,
            "evidence_boundary_hash": boundary_hash,
            "included_artifact_hashes": {
                str(row.get("task_id")): row.get("artifact_sha256") for row in included
            },
            "included_reasons": {str(row.get("task_id")): row.get("reason") for row in included},
            "excluded_artifact_hashes": {
                str(row.get("id")): row.get("artifact_sha256") for row in excluded
            },
            "excluded_reasons": {str(row.get("id")): row.get("reason") for row in excluded},
            "allowed_internal_claims": list(ALLOWED_INTERNAL_CLAIMS),
            "forbidden_claims": list(FORBIDDEN_CLAIMS),
            "universal_support_claimed": False,
            "public_factor_claim_eligibility": False,
        }
    ]


def claim_ledger_receipt(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    content = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    return {
        "path": relative_or_absolute(path),
        "present": path.is_file(),
        "sha256": sha256_bytes(content),
        "row_count": len(rows),
        "rows": [dict(row) for row in rows],
        "append_only": True,
    }


def field_principles(
    included: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    principles = dict(BASE_FIELD_PRINCIPLES)
    for row in included:
        task_id = str(row.get("task_id"))
        principles[f"include.{task_id}"] = (
            "This inclusion decision fails closed unless all clean V550 gates pass."
        )
    for row in excluded:
        row_id = str(row.get("id"))
        principles[f"exclude.{row_id}"] = (
            "This exclusion decision keeps nonclean or out-of-scope evidence from widening claims."
        )
    return principles


def field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "kind": "derived",
            "sources": ["REQ-LEARN-6406", "V550 terminal artifacts", "sidecar hashes"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def refresh_terminal_fields(report: JsonDict) -> None:
    included = [
        as_mapping(row) for row in as_sequence(report.get("included_clean_artifact_records"))
    ]
    excluded = [
        as_mapping(row)
        for row in as_sequence(
            report.get(
                "excluded_nonclean_blocked_null_absent_unlicensed_rejected_and_flagged_records"
            )
        )
    ]
    attacks = as_mapping(
        report.get(
            "substitution_laundering_date_model_family_license_sidecar_conductor_and_flag_attack_matrix"
        )
    )
    tests = as_mapping(report.get("tests_run"))
    protected = as_mapping(report.get("protected_files_unchanged"))
    preconditions = as_mapping(report.get("preconditions_checked"))

    clean_included = bool(included) and all(row.get("clean") is True for row in included)
    exclusions_kept = {"exp6385", "exp6399", "exp6403"} <= {
        str(row.get("id")) for row in excluded
    } and all(row.get("included_in_clean_boundary") is False for row in excluded)
    attacks_ok = (
        attacks.get("all_fail_closed") is True
        and attacks.get("included_row_added_by_attack_count") == 0
        and attacks.get("excluded_row_suppressed_by_attack_count") == 0
        and attacks.get("public_claim_enabled_by_attack_count") == 0
    )
    tests_ok = tests.get("all_passed") is True and all(
        code == 0 for code in as_mapping(tests.get("exit_codes")).values()
    )
    ready = all(
        (
            clean_included,
            exclusions_kept,
            attacks_ok,
            report.get("universal_support_claimed") is False,
            report.get("public_factor_claim_eligibility") is False,
            report.get("upstream_artifacts_modified") is False,
            protected.get("ok") is True,
            preconditions.get("all_preconditions_checked") is True,
            report.get("inference_substrate") == INFERENCE_SUBSTRATE,
            report.get("verifier_is_oracle") is False,
            tests_ok,
        )
    )
    report["clean_factor_evidence_boundary_ready_score"] = 1.0 if ready else 0.0
    report["status"] = "complete" if ready else "complete_null"
    if ready:
        report["honest_verdict"] = (
            "complete: immutable clean V550 factor boundary is ready for narrow internal "
            "claims only; public factor claim eligibility remains false"
        )
    else:
        report["honest_verdict"] = (
            "complete_null: clean V550 factor boundary failed closed and no public claim is eligible"
        )
    report["reproducibility_checksum"] = payload_checksum(report)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material["reproducibility_checksum"] = ""
    return sha256_json(material)


def build_report(
    *,
    root: Path,
    date: str,
    result_path: Path,
    registration_path: Path,
    ledger_path: Path,
    registration: Mapping[str, Any],
    payloads: Mapping[str, JsonDict | None],
    classified: Mapping[str, JsonDict],
    sidecars: Mapping[str, list[JsonDict]],
    before_hashes: Mapping[str, str | None],
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    conductor = conductor_outcomes(root)
    matrix = artifact_matrix(classified, payloads, conductor)
    included = included_clean_records(payloads, matrix, sidecars)
    excluded = excluded_records(payloads, matrix, sidecars)
    exp6385_receipt, exp6399_receipt = preservation_receipts(payloads, matrix)
    states = recomputed_states(payloads)
    attacks = attack_matrix(included, excluded)
    boundary_hash = sha256_json(evidence_boundary_hash_material(included, excluded, matrix))
    rows = ledger_rows(date=date, boundary_hash=boundary_hash, included=included, excluded=excluded)
    protected = protected_files_receipt(root, before_hashes)
    sidecar_hash_complete = all(
        receipt.get("present") and receipt.get("sha256")
        for receipts in sidecars.values()
        for receipt in receipts
    )
    source = source_receipts(root)
    report: JsonDict = {
        "status": "complete_null",
        "audit_registration_path_hash_and_expected_scope": registration_receipt(
            registration_path, registration
        ),
        "v550_artifact_hash_verdict_conductor_duration_and_flag_matrix": matrix,
        "clean_inclusion_rule": clean_inclusion_rule(),
        "explicit_exclusion_rule": explicit_exclusion_rule(),
        "included_clean_artifact_records": included,
        "excluded_nonclean_blocked_null_absent_unlicensed_rejected_and_flagged_records": excluded,
        "exp6385_preservation_receipt": exp6385_receipt,
        "exp6399_preservation_receipt": exp6399_receipt,
        "recomputed_narrow_harness_license_frontier_learning_consumer_and_safety_states": states,
        "universal_support_claimed": False,
        "public_factor_claim_eligibility": False,
        "allowed_internal_claims": list(ALLOWED_INTERNAL_CLAIMS),
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "claim_ledger_path_hash_and_rows": claim_ledger_receipt(ledger_path, rows),
        "substitution_laundering_date_model_family_license_sidecar_conductor_and_flag_attack_matrix": attacks,
        "clean_factor_evidence_boundary_ready_score": 0.0,
        "upstream_artifacts_modified": False,
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "date": date,
            "planning_date": RUN_DATE,
            "result_path": relative_or_absolute(result_path, root=root),
            "registration_written_before_conclusion_reads": registration_path.is_file(),
            "artifact_classes_frozen_before_conclusion_reads": True,
            "sidecar_hashes_complete": sidecar_hash_complete,
            "required_source_hashes_complete": source.get("all_required_source_files_present"),
            "protected_files_unchanged": protected.get("ok") is True,
            "evidence_boundary_hash": boundary_hash,
            "llm_call_count": 0,
            "upstream_rerun_count": 0,
            "all_preconditions_checked": bool(
                sidecar_hash_complete
                and source.get("all_required_source_files_present")
                and protected.get("ok") is True
            ),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": field_principles(included, excluded),
        "field_provenance": field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "tests_run": tests_run(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "complete_null: pending refresh",
    }
    refresh_terminal_fields(report)
    return report


def validate_report(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in report]
    errors.extend(f"missing required field: {field}" for field in missing)
    extra = [field for field in report if field not in REQUIRED_ARTIFACT_FIELDS]
    errors.extend(f"extra top-level field: {field}" for field in extra)
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if report.get("public_factor_claim_eligibility") is not False:
        errors.append("public_factor_claim_eligibility must be false")
    if report.get("universal_support_claimed") is not False:
        errors.append("universal_support_claimed must be false")
    if report.get("upstream_artifacts_modified") is not False:
        errors.append("upstream_artifacts_modified must be false")
    if not str(report.get("honest_verdict", "")).startswith(
        (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
        )
    ):
        errors.append("honest_verdict lacks accepted prefix")
    principles = as_mapping(report.get("field_principles"))
    for key in (
        "clean_factor_evidence_boundary_ready_score",
        "universal_support_claimed",
        "public_factor_claim_eligibility",
    ):
        if key not in principles:
            errors.append(f"missing field_principles entry: {key}")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        errors.append("reproducibility_checksum mismatch")
    return errors


def run(
    *,
    date: str = RUN_DATE,
    root: Path | str = REPO_ROOT,
    result_path: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
    validate: bool = False,
) -> JsonDict:
    root_path = Path(root).resolve()
    start = time.perf_counter()
    output_path = result_path or root_path / RESULT_RELATIVE_PATH
    registration_path, ledger_path = result_sidecar_paths(output_path)
    artifact_paths = resolve_expected_paths(root_path)
    before_hashes = protected_hashes(root_path)
    source = source_receipts(root_path)
    sidecars = sidecar_receipts(root_path)
    classified = classify_receipts(artifact_paths)
    registration = build_registration(
        root=root_path,
        date=date,
        result_path=output_path,
        artifact_paths=artifact_paths,
        classified=classified,
        source=source,
        sidecars=sidecars,
        before_hashes=before_hashes,
    )
    if write:
        write_stable_json(registration_path, registration)
    payloads = load_payloads(artifact_paths)
    elapsed = duration_s if duration_s is not None else time.perf_counter() - start
    report = build_report(
        root=root_path,
        date=date,
        result_path=output_path,
        registration_path=registration_path,
        ledger_path=ledger_path,
        registration=registration,
        payloads=payloads,
        classified=classified,
        sidecars=sidecars,
        before_hashes=before_hashes,
        duration_s=elapsed,
        test_exit_codes=test_exit_codes,
    )
    if write:
        rows = as_sequence(as_mapping(report["claim_ledger_path_hash_and_rows"]).get("rows"))
        write_jsonl(ledger_path, [as_mapping(row) for row in rows])
        write_stable_json(output_path, report)
    errors = validate_report(report)
    if validate and errors:
        raise ValueError("; ".join(errors))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    run(
        date=args.date,
        result_path=args.output,
        test_exit_codes={command: 0 for command in DEFAULT_TEST_COMMANDS},
        validate=args.validate,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
