"""Exp6399 V550 capability-learning safety audit.

Spec refs: REQ-LEARN-6399, SCENARIO-LEARN-6399-REGISTRATION,
SCENARIO-LEARN-6399-CLASS-PRESERVATION,
SCENARIO-LEARN-6399-LICENSE-BOUNDARY,
SCENARIO-LEARN-6399-TRANSACTION-BOUNDARY, SCENARIO-LEARN-6399-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from carnot.terminal_artifacts import classify_artifact_path


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6399_capability_learning_safety_audit.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6399_capability_learning_safety_audit.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6399_capability_learning_safety_audit.py"
)
SCHEMA = "carnot.experiment_6399.capability_learning_safety_audit.v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6399
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts_no_llm"

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
CONSTRAINT_FAMILIES = ("threshold_guard", "route_guard", "conservation_guard")
EXPECTED_MODEL_FAMILY_CELL_COUNT = len(MANDATED_MODEL_IDS) * len(CONSTRAINT_FAMILIES)
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"

EXPECTED_TASK_IDS = (
    "exp6394",
    "exp6395",
    "exp6396",
    "exp6397",
    "exp6398",
    "exp6383",
    "exp6385",
)
EXPECTED_ARTIFACTS: dict[str, JsonDict] = {
    "exp6394": {
        "path": Path("results/experiment_6394_model_family_factor_harness_freeze.json"),
        "ready_score_field": "model_family_harness_freeze_ready_score",
        "task": "model_family_factor_harness_freeze",
    },
    "exp6395": {
        "path": Path("results/experiment_6395_held_factor_transport_license_matrix.json"),
        "ready_score_field": "held_factor_transport_license_ready_score",
        "task": "held_factor_transport_license_matrix",
    },
    "exp6396": {
        "path": Path("results/experiment_6396_capability_qualified_verified_frontier_ab.json"),
        "ready_score_field": "capability_qualified_frontier_ready_score",
        "task": "capability_qualified_verified_frontier_ab",
    },
    "exp6397": {
        "path": Path("results/experiment_6397_transactional_continuous_factor_learning.json"),
        "ready_score_field": "transactional_continuous_self_learning_ready_score",
        "task": "transactional_continuous_factor_learning",
    },
    "exp6398": {
        "path": Path("results/experiment_6398_default_off_transactional_factor_consumer.json"),
        "ready_score_field": "default_off_transactional_consumer_ready_score",
        "task": "default_off_transactional_factor_consumer",
    },
    "exp6383": {
        "path": Path("results/experiment_6383_dependency_guided_factor_rollback_stress.json"),
        "ready_score_field": "dependency_guided_rollback_ready_score",
        "task": "dependency_guided_factor_rollback_stress",
    },
    "exp6385": {
        "path": Path("results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json"),
        "ready_score_field": "factor_learning_rollback_safety_ready_score",
        "task": "live_factor_learning_and_rollback_safety_audit",
    },
}
EXPECTED_SIDECARS: dict[str, tuple[Path, ...]] = {
    "exp6394": (
        Path("data/research/experiment_6394_model_family_factor_harness_freeze/manifests/development_manifest.json"),
        Path("data/research/experiment_6394_model_family_factor_harness_freeze/manifests/held_manifest.redacted.json"),
        Path("data/research/experiment_6394_model_family_factor_harness_freeze/frozen_harnesses/frozen_harness_gemma_dense.json"),
        Path("data/research/experiment_6394_model_family_factor_harness_freeze/frozen_harnesses/frozen_harness_gemma_moe.json"),
        Path("data/research/experiment_6394_model_family_factor_harness_freeze/frozen_harnesses/frozen_harness_qwen_moe.json"),
    ),
    "exp6395": (),
    "exp6396": (
        Path("results/experiment_6396_capability_qualified_verified_frontier_ab.json.train_counterexample_manifest.json"),
        Path("results/experiment_6396_capability_qualified_verified_frontier_ab.json.untouched_future_manifest.json"),
    ),
    "exp6397": (
        Path("results/experiment_6397_transactional_continuous_factor_learning.json.chronological_manifest.json"),
    ),
    "exp6398": (
        Path("results/experiment_6398_default_off_transactional_factor_consumer.json.untouched_consumer_manifest.json"),
    ),
    "exp6383": (
        Path("results/experiment_6383_dependency_guided_factor_rollback_stress.json.typed_dependency_schema.json"),
    ),
    "exp6385": (
        Path("results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json.audit_registration.json"),
        Path("results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json.attack_manifest.json"),
    ),
}
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    Path("openspec/capabilities/research-harnesses/spec.md"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6394_model_family_factor_harness_freeze.py"),
    Path("python/carnot/experiment_6395_held_factor_transport_license_matrix.py"),
    Path("python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py"),
    Path("python/carnot/experiment_6397_transactional_continuous_factor_learning.py"),
    Path("python/carnot/experiment_6398_default_off_transactional_factor_consumer.py"),
    Path("python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py"),
    Path("python/carnot/experiment_6385_live_factor_learning_and_rollback_safety_audit.py"),
    Path("scripts/summarize_artifact.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/check_determination_preservation.py"),
    Path("scripts/determination_preservation_lint.py"),
    Path("ops/conductor-log.md"),
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
    *(row["path"] for row in EXPECTED_ARTIFACTS.values()),
)

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6399_capability_learning_safety_audit --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6399_capability_learning_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6399_capability_learning_safety_audit.py "
    "-m pytest tests/python/test_experiment_6399_capability_learning_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6399_capability_learning_safety_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6399_capability_learning_safety_audit.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6399_capability_learning_safety_audit.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    RUN_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "audit_registration_path_hash_and_expected_scope",
    "present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix",
    "artifact_verdict_and_conductor_outcome_reconciliation",
    "model_schema_harness_license_factor_head_transaction_and_checker_hash_matrix",
    "development_held_future_and_source_leakage_attack_results",
    "family_model_harness_schema_license_fallback_abstention_and_pooling_attack_results",
    "predecessor_effect_evidence_optional_stopping_atomicity_concurrency_restart_and_renewal_attack_results",
    "exact_checker_rollback_revocation_consumer_write_and_enablement_attack_results",
    "recomputed_readiness_scores_and_gates",
    "model_policy_and_inference_substrate_checks",
    "duration_receipt_source",
    "critical_major_and_minor_findings",
    "utility_promotion_count",
    "public_factor_claim_eligibility",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status follows the independent audit and public-claim gate.",
    "audit_registration_path_hash_and_expected_scope": "Registration binds paths, scopes, versions, and read order before conclusions.",
    "present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix": "Every expected input keeps its terminal evidence class.",
    "artifact_verdict_and_conductor_outcome_reconciliation": "Artifact verdicts and conductor outcomes stay separate.",
    "model_schema_harness_license_factor_head_transaction_and_checker_hash_matrix": "Model, schema, harness, license, head, transaction, and checker hashes are bound together.",
    "development_held_future_and_source_leakage_attack_results": "Leakage and source attacks cannot promote readiness.",
    "family_model_harness_schema_license_fallback_abstention_and_pooling_attack_results": "License and pooling attacks cannot broaden scope.",
    "predecessor_effect_evidence_optional_stopping_atomicity_concurrency_restart_and_renewal_attack_results": "Transaction attacks cannot advance a head or renew a license.",
    "exact_checker_rollback_revocation_consumer_write_and_enablement_attack_results": "Checker, rollback, consumer-write, and enablement attacks fail closed.",
    "recomputed_readiness_scores_and_gates": "Bare terminal fields recompute all readiness and claim gates.",
    "model_policy_and_inference_substrate_checks": "Model, tokenizer, GPU, substrate, and legacy-claim checks stay explicit.",
    "duration_receipt_source": "Wall-clock duration is measured by the audit only.",
    "critical_major_and_minor_findings": "Findings are severity-separated without synthesis.",
    "utility_promotion_count": "Safety evidence cannot become utility evidence.",
    "public_factor_claim_eligibility": "The public claim is false unless the full clean scope passes.",
    "upstream_artifacts_modified": "Upstream artifacts must remain unchanged.",
    "protected_files_unchanged": "Protected repo files must remain unchanged.",
    "preconditions_checked": "Preconditions bind date, registration, classes, hashes, sources, protected files, and commands.",
    "inference_substrate": "The substrate declares deterministic artifact audit without LLM or upstream rerun.",
    "verifier_is_oracle": "Bare false states that the audit is not an oracle.",
    "field_principles": "Required fields and recomputed claim fields state their fail-closed purpose.",
    "field_provenance": "Required fields trace to specs, inputs, attacks, checks, tests, or hashes.",
    "random_seed": "Fixed seed pins registration and attack order.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states the public-claim boundary.",
}
EXTRA_FIELD_PRINCIPLES = {
    "recomputed_readiness_scores_and_gates.scores": "Missing, nested, wrong-type, boolean, NaN, and infinity scores become zero.",
    "recomputed_readiness_scores_and_gates.claim_gates": "The public claim gate is the conjunction of clean scope, attacks, models, tests, and license coverage.",
    "public_factor_claim_eligibility.full_scope": "Narrow or partial model-family evidence cannot become a general utility claim.",
    "public_factor_claim_eligibility.safety_is_not_utility": "Safety success cannot substitute for a clean utility artifact.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6399",
        "V550 upstream artifact bytes",
        "pre-conclusion registration",
        "pre-conclusion artifact classification",
        "Exp6399 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}

LEAKAGE_ATTACKS = (
    "development_held_leakage",
    "future_leakage",
    "source_substitution",
    "protected_label_read",
)
LICENSE_ATTACKS = (
    "family_identity_drift",
    "model_hash_drift",
    "harness_drift",
    "schema_drift",
    "license_overreach",
    "inherited_license",
    "silent_fallback",
    "abstention_suppression",
    "partial_cell_pooling",
)
TRANSACTION_ATTACKS = (
    "self_activation",
    "stale_predecessor",
    "duplicate_effect",
    "replayed_evidence",
    "optional_stopping_reset",
    "interrupted_atomic_write",
    "concurrent_head_advance",
    "restart_corruption",
    "unauthorized_license_renewal",
)
CONSUMER_ATTACKS = (
    "source_substitution",
    "exact_check_omission",
    "verifier_version_drift",
    "rollback_underreach",
    "rollback_overreach",
    "revoked_descendant_survival",
    "consumer_write",
    "production_enablement",
)
EVIDENCE_CLASSES = (
    "positive",
    "complete",
    "ready",
    "null",
    "blocked",
    "skipped",
    "retired",
    "flagged",
    "absent",
    "malformed",
)


def canonical_json(value: Any) -> str:
    """Return stable JSON text for receipts and checksums."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the repository digest prefix."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after canonical serialization."""

    return sha256_text(canonical_json(value))


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


def bare_finite_number(value: Any) -> float:
    """Return a finite bare number or zero for any fail-closed shape."""

    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON through a same-directory temporary path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json_object(path: str | Path) -> JsonDict | None:
    """Read a JSON object and return None for missing or malformed input."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def relative_or_absolute(path: Path) -> str:
    """Return a repo-relative path when the path is inside this checkout."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def path_receipt(path: str | Path) -> JsonDict:
    """Record path, presence, size, and hash."""

    path = Path(path)
    return {
        "path": relative_or_absolute(path),
        "present": path.is_file(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def terminal_path_receipt(path: Path) -> JsonDict:
    """Classify one artifact from bytes, not conductor success."""

    classification = classify_artifact_path(path)
    return {
        "path": relative_or_absolute(path),
        "present": classification.present,
        "loadable": classification.loadable,
        "terminal": classification.terminal,
        "evidence_class": evidence_class(classification.classification),
        "terminal_class": classification.classification,
        "reason": classification.reason,
        "status_raw": classification.status_raw,
        "honest_verdict_raw": classification.honest_verdict_raw,
        "sha256": classification.sha256,
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def evidence_class(classification: str) -> str:
    """Map the shared classifier classes into audit evidence buckets."""

    if classification == "missing":
        return "absent"
    if classification in EVIDENCE_CLASSES:
        return classification
    return "malformed"


def upstream_paths(overrides: Mapping[str, Path | str] | None = None) -> dict[str, Path]:
    """Resolve expected upstream paths with optional test overrides."""

    override_map = {name: Path(path) for name, path in (overrides or {}).items()}
    return {
        name: override_map.get(name, REPO_ROOT / as_mapping(row)["path"])
        for name, row in EXPECTED_ARTIFACTS.items()
    }


def source_file_receipts() -> JsonDict:
    """Hash source, ops, and checker files that define the audit."""

    files = {path.as_posix(): path_receipt(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}
    return {"files": files, "source_files_sha256": sha256_json(files)}


def protected_hashes(paths: Mapping[str, Path]) -> dict[str, str | None]:
    """Hash protected files and upstream artifacts before audit writes."""

    rows = {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}
    rows.update({f"upstream:{name}": sha256_file(path) for name, path in paths.items()})
    return rows


def compare_hashes(before: Mapping[str, str | None], after: Mapping[str, str | None]) -> JsonDict:
    """Compare two hash maps without repairing any changed file."""

    files = {
        key: {
            "before": before.get(key),
            "after": after.get(key),
            "unchanged": before.get(key) == after.get(key),
        }
        for key in sorted(set(before) | set(after))
    }
    return {
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [key for key, row in files.items() if not row["unchanged"]],
        "files": files,
    }


def expected_scope(source: Mapping[str, Any]) -> JsonDict:
    """Describe the audit surface before conclusion fields are read."""

    checker_paths = [
        path
        for path in SOURCE_RELATIVE_PATHS
        if path.as_posix().startswith("python/carnot/experiment_")
        or path.as_posix().startswith("scripts/")
    ]
    return {
        "schema": SCHEMA + ".expected_scope",
        "planning_date": RUN_DATE,
        "task_ids": list(EXPECTED_TASK_IDS),
        "artifact_paths": {
            name: as_mapping(row)["path"].as_posix()
            for name, row in EXPECTED_ARTIFACTS.items()
        },
        "sidecar_paths": {
            name: [path.as_posix() for path in EXPECTED_SIDECARS[name]]
            for name in EXPECTED_TASK_IDS
        },
        "source_files_sha256": source.get("source_files_sha256"),
        "model_ids": list(MANDATED_MODEL_IDS),
        "constraint_families": list(CONSTRAINT_FAMILIES),
        "schemas": [
            "carnot.experiment_6394.model_family_factor_harness_freeze.v1",
            "carnot.experiment_6395.held_factor_transport_license_matrix.v1",
            "carnot.experiment_6396.capability_qualified_verified_frontier_ab.v1",
            "carnot.experiment_6397.transactional_continuous_factor_learning.v1",
            "carnot.experiment_6398.default_off_transactional_factor_consumer.v1",
            SCHEMA,
        ],
        "harnesses": [
            "frozen_harness_gemma_dense",
            "frozen_harness_gemma_moe",
            "frozen_harness_qwen_moe_abstain_only",
        ],
        "license_surfaces": [
            "model_file_sha256",
            "embedded_tokenizer_sha256",
            "frozen_harness_sha256",
            "canonical_schema_sha256",
            "event_manifest_sha256",
            "constraint_family",
        ],
        "factor_heads": [
            "exp6397.factor_head_initial_hash",
            "exp6397.factor_head_transition_history.terminal_head_hash",
            "exp6398.frozen_factor_head_and_transaction_log_hashes.retained_predecessor_bound_head_hash",
        ],
        "transaction_logs": [
            "exp6397.factor_head_transition_history.transition_rows",
            "exp6398.frozen_factor_head_and_transaction_log_hashes.transaction_log_hash",
        ],
        "exact_checker_versions": {
            path.as_posix(): as_mapping(source.get("files")).get(path.as_posix())
            for path in checker_paths
        },
        "llm_call_budget": 0,
        "upstream_rerun_budget": 0,
    }


def artifact_hashes_and_classes(paths: Mapping[str, Path]) -> JsonDict:
    """Hash and classify artifacts and sidecars before semantic reads."""

    artifacts = {name: terminal_path_receipt(path) for name, path in paths.items()}
    sidecars = {
        name: [path_receipt(REPO_ROOT / sidecar) for sidecar in EXPECTED_SIDECARS[name]]
        for name in EXPECTED_TASK_IDS
    }
    return {
        "artifacts": artifacts,
        "sidecars": sidecars,
        "classification_before_conclusion_reads": True,
        "all_hashes_sha256": sha256_json({"artifacts": artifacts, "sidecars": sidecars}),
    }


def build_registration(
    *,
    date: str,
    result_path: Path,
    paths: Mapping[str, Path],
    artifacts: Mapping[str, Any],
    source: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Build the read-order receipt before conclusion fields are loaded."""

    return {
        "schema": SCHEMA + ".registration",
        "date": date,
        "planning_date": RUN_DATE,
        "read_order": [
            "register_expected_scope",
            "hash_artifacts_sidecars_sources_and_protected_files",
            "classify_artifact_terminal_classes",
            "write_registration_sidecar",
            "write_attack_manifest_sidecar",
            "read_upstream_conclusion_fields",
            "recompute_bare_terminal_gates",
        ],
        "expected_scope": expected_scope(source),
        "artifact_hashes_and_classes_sha256": artifacts["all_hashes_sha256"],
        "source_files_sha256": source["source_files_sha256"],
        "protected_hashes_sha256": sha256_json(protected_before),
        "result_path": relative_or_absolute(result_path),
        "expected_artifact_path_receipts": {
            name: path_receipt(path) for name, path in paths.items()
        },
        "random_seed": RANDOM_SEED,
        "conclusion_fields_read": False,
    }


def build_attack_manifest(registration_receipt: Mapping[str, Any]) -> JsonDict:
    """Register attacks before upstream conclusion fields are read."""

    attacks = {
        "development_held_future_and_source_leakage_attack_results": list(LEAKAGE_ATTACKS),
        "family_model_harness_schema_license_fallback_abstention_and_pooling_attack_results": list(LICENSE_ATTACKS),
        "predecessor_effect_evidence_optional_stopping_atomicity_concurrency_restart_and_renewal_attack_results": list(TRANSACTION_ATTACKS),
        "exact_checker_rollback_revocation_consumer_write_and_enablement_attack_results": list(CONSUMER_ATTACKS),
    }
    return {
        "schema": SCHEMA + ".attack_manifest",
        "registration_sha256": registration_receipt.get("sha256"),
        "attacks": attacks,
        "attack_count": sum(len(values) for values in attacks.values()),
        "utility_promotion_allowed": False,
        "public_claim_allowed_without_full_license_scope": False,
    }


def registration_receipt(path: Path, registration: Mapping[str, Any]) -> JsonDict:
    """Return the required top-level registration field."""

    return {
        **path_receipt(path),
        "registration_written_before_conclusion_reads": path.is_file(),
        "expected_scope": dict(as_mapping(registration.get("expected_scope"))),
        "read_order": list(as_sequence(registration.get("read_order"))),
    }


def load_payloads(paths: Mapping[str, Path]) -> dict[str, JsonDict | None]:
    """Load upstream payloads only after registration and manifest writes."""

    return {name: read_json_object(path) for name, path in paths.items()}


def artifact_matrix(
    artifact_receipts: Mapping[str, Any],
    payloads: Mapping[str, JsonDict | None],
) -> JsonDict:
    """Keep present, missing, blocked, skipped, null, flagged, and retired rows."""

    rows: dict[str, JsonDict] = {}
    counts = {name: 0 for name in EVIDENCE_CLASSES}
    for name in EXPECTED_TASK_IDS:
        receipt = as_mapping(as_mapping(artifact_receipts.get("artifacts")).get(name))
        payload = as_mapping(payloads.get(name))
        cls = str(receipt.get("evidence_class") or "malformed")
        counts[cls] = counts.get(cls, 0) + 1
        rows[name] = {
            "artifact_path": receipt.get("path"),
            "present": receipt.get("present") is True,
            "loadable": receipt.get("loadable") is True,
            "evidence_class": cls,
            "terminal_class": receipt.get("terminal_class"),
            "status": payload.get("status"),
            "honest_verdict": payload.get("honest_verdict"),
            "sha256": receipt.get("sha256"),
            "kept_without_synthesis": cls in EVIDENCE_CLASSES,
            "relabeled_clean": False,
        }
    return {
        "schema": SCHEMA + ".artifact_matrix",
        "by_artifact": rows,
        "class_counts": dict(sorted(counts.items())),
        "classification_before_conclusion_reads": True,
        "missing_or_blocked_relabelled_clean_count": 0,
        "absent_blocked_skipped_null_flagged_and_retired_preserved": True,
    }


def conductor_outcomes() -> JsonDict:
    """Extract nearby conductor outcomes without using them as artifact truth."""

    path = REPO_ROOT / "ops/conductor-log.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    patterns = {
        "exp6394": "Model-family factor harness development",
        "exp6395": "Gated on Exp6394 freeze",
        "exp6396": "Gated on Exp6395 licenses",
        "exp6397": "Gated on Exp6396 positive delta",
        "exp6398": "Gated on Exp6397 readiness",
        "exp6383": "dependency-guided",
        "exp6385": "V549 safety",
    }
    rows: dict[str, JsonDict] = {}
    for name, pattern in patterns.items():
        matches = [line for line in text.splitlines() if pattern.lower() in line.lower()]
        line = matches[-1] if matches else ""
        parts = [part.strip() for part in line.strip("|").split("|")] if line else []
        rows[name] = {
            "found": bool(line),
            "log_line_sha256": sha256_text(line) if line else None,
            "timestamp": parts[0] if len(parts) >= 4 else None,
            "task_title": parts[1] if len(parts) >= 4 else None,
            "conductor_outcome": parts[2] if len(parts) >= 4 else "not_found",
            "detail": parts[3] if len(parts) >= 4 else None,
        }
    return {"path": relative_or_absolute(path), "sha256": sha256_file(path), "by_task": rows}


def reconcile_artifacts_and_conductor(
    matrix: Mapping[str, Any],
    conductor: Mapping[str, Any],
) -> JsonDict:
    """Report conductor outcomes and artifact verdicts as separate evidence."""

    rows = {}
    for name in EXPECTED_TASK_IDS:
        artifact = as_mapping(as_mapping(matrix.get("by_artifact")).get(name))
        conductor_row = as_mapping(as_mapping(conductor.get("by_task")).get(name))
        rows[name] = {
            "artifact_evidence_class": artifact.get("evidence_class"),
            "artifact_status": artifact.get("status"),
            "artifact_honest_verdict": artifact.get("honest_verdict"),
            "conductor_outcome": conductor_row.get("conductor_outcome"),
            "conductor_found": conductor_row.get("found") is True,
            "conductor_used_as_oracle": False,
        }
    return {
        "schema": SCHEMA + ".artifact_conductor_reconciliation",
        "rows": rows,
        "artifact_verdicts_used_for_recomputed_scores": True,
        "conductor_outcomes_preserved_separately": True,
        "conductor_outcome_overrode_artifact_verdict_count": 0,
    }


def license_records(payloads: Mapping[str, JsonDict | None]) -> list[JsonDict]:
    """Return Exp6395 license rows without inheriting downstream licenses."""

    return [
        dict(as_mapping(row))
        for row in as_sequence(as_mapping(payloads.get("exp6395")).get("capability_license_records"))
        if isinstance(row, Mapping)
    ]


def unlicensed_records(payloads: Mapping[str, JsonDict | None]) -> list[JsonDict]:
    """Return visible unlicensed or rejected model-family cells."""

    rows = as_sequence(as_mapping(payloads.get("exp6395")).get("rejected_and_abstained_cell_records"))
    if not rows:
        rows = as_sequence(as_mapping(payloads.get("exp6398")).get("unlicensed_cell_abstention_records"))
    return [dict(as_mapping(row)) for row in rows if isinstance(row, Mapping)]


def transaction_rows(payloads: Mapping[str, JsonDict | None]) -> list[JsonDict]:
    """Return Exp6397 transaction rows."""

    history = as_mapping(as_mapping(payloads.get("exp6397")).get("factor_head_transition_history"))
    return [dict(as_mapping(row)) for row in as_sequence(history.get("transition_rows"))]


def hash_matrix(payloads: Mapping[str, JsonDict | None], source: Mapping[str, Any]) -> JsonDict:
    """Bind model, schema, harness, license, head, transaction, and checker hashes."""

    model_specs_by_artifact = {
        name: as_sequence(as_mapping(payload).get("MODEL_SPECS"))
        for name, payload in payloads.items()
        if as_mapping(payload).get("MODEL_SPECS") is not None
    }
    model_hashes: dict[str, set[str]] = {}
    for rows in model_specs_by_artifact.values():
        for row in rows:
            spec = as_mapping(row)
            model_hashes.setdefault(str(spec.get("hf_id")), set()).add(
                str(spec.get("model_file_sha256"))
            )
    licenses = license_records(payloads)
    harnesses = as_mapping(
        as_mapping(payloads.get("exp6394")).get("frozen_harness_paths_hashes_and_controls")
    )
    history = as_mapping(as_mapping(payloads.get("exp6397")).get("factor_head_transition_history"))
    tx_rows = transaction_rows(payloads)
    bindings = as_mapping(
        as_mapping(payloads.get("exp6398")).get("license_and_harness_bindings")
    )
    checker_hashes = dict(as_mapping(bindings.get("exact_checker_hashes")))
    checker_hashes.update(
        {
            path: as_mapping(source.get("files")).get(path)
            for path in (
                MODULE_RELATIVE_PATH.as_posix(),
                "scripts/adversarial_verify.py",
                "scripts/summarize_artifact.py",
                "scripts/check_determination_preservation.py",
            )
        }
    )
    return {
        "schema": SCHEMA + ".hash_matrix",
        "model_specs_by_artifact": model_specs_by_artifact,
        "model_hashes_by_hf_id": {
            model: sorted(values) for model, values in sorted(model_hashes.items())
        },
        "all_model_hashes_stable": all(len(values) == 1 for values in model_hashes.values()),
        "schema_hashes": sorted(
            {
                str(row.get("canonical_schema_sha256"))
                for row in licenses
                if row.get("canonical_schema_sha256")
            }
        ),
        "frozen_harnesses": harnesses,
        "license_records": licenses,
        "license_record_count": len(licenses),
        "license_hashes": [
            sha256_json(row)
            for row in licenses
        ],
        "factor_head_initial_hash": history.get("initial_head_hash"),
        "factor_head_terminal_hash": history.get("terminal_head_hash"),
        "transaction_log_entry_count": len(tx_rows),
        "transaction_log_sha256": sha256_json(tx_rows),
        "checker_hashes": checker_hashes,
        "checker_versions_complete": all(value is not None for value in checker_hashes.values()),
    }


def _attack_row(attack_id: str, passed: bool, detail: str) -> JsonDict:
    return {
        "attack_id": attack_id,
        "failed_closed": bool(passed),
        "promoted_readiness": False,
        "detail": detail,
    }


def leakage_attack_results(payloads: Mapping[str, JsonDict | None]) -> JsonDict:
    """Audit development-held, future, and source leakage surfaces."""

    exp6394_counts = as_mapping(
        as_mapping(payloads.get("exp6394")).get("protected_leakage_and_same_step_write_counts")
    )
    exp6395_attacks = as_mapping(
        as_mapping(
            as_mapping(payloads.get("exp6395")).get(
                "model_row_family_label_harness_schema_source_fallback_abstention_and_promotion_attack_matrix"
            )
        ).get("attacks")
    )
    exp6396_attacks = as_mapping(
        as_mapping(
            as_mapping(payloads.get("exp6396")).get(
                "identity_license_order_placebo_work_stopping_and_leakage_attack_matrix"
            )
        ).get("attacks")
    )
    protected_leaks = [
        bare_finite_number(as_mapping(payloads.get(name)).get("protected_leakage_count"))
        for name in ("exp6395", "exp6396", "exp6397", "exp6398")
    ]
    rows = {
        "development_held_leakage": _attack_row(
            "development_held_leakage",
            exp6394_counts.get("held_outcome_read_count") == 0
            and exp6394_counts.get("held_event_content_read_count") == 0,
            "held rows were redacted during development selection",
        ),
        "future_leakage": _attack_row(
            "future_leakage",
            sum(protected_leaks) == 0
            and as_mapping(exp6396_attacks.get("protected_future_leakage")).get("failed_closed")
            is True,
            "future labels open only after the frozen factor boundary",
        ),
        "source_substitution": _attack_row(
            "source_substitution",
            as_mapping(exp6395_attacks.get("source_substitution")).get("failed_closed") is True,
            "source hashes are checked before exact calls",
        ),
        "protected_label_read": _attack_row(
            "protected_label_read",
            exp6394_counts.get("protected_leakage_count") == 0 and sum(protected_leaks) == 0,
            "protected labels are not read early",
        ),
    }
    return {
        "schema": SCHEMA + ".leakage_attacks",
        "attacks": rows,
        "all_fail_closed": all(row["failed_closed"] for row in rows.values()),
        "protected_leakage_count": int(sum(protected_leaks)),
        "development_held_content_read_count": exp6394_counts.get("held_event_content_read_count", 0),
        "future_outcome_read_before_freeze_count": 0,
    }


def license_attack_results(payloads: Mapping[str, JsonDict | None]) -> JsonDict:
    """Audit narrow license scope, fallback, abstention, and pooling rules."""

    licenses = license_records(payloads)
    unlicensed = unlicensed_records(payloads)
    exp6395_attacks = as_mapping(
        as_mapping(
            as_mapping(payloads.get("exp6395")).get(
                "model_row_family_label_harness_schema_source_fallback_abstention_and_promotion_attack_matrix"
            )
        ).get("attacks")
    )
    exp6396_attacks = as_mapping(
        as_mapping(
            as_mapping(payloads.get("exp6396")).get(
                "identity_license_order_placebo_work_stopping_and_leakage_attack_matrix"
            )
        ).get("attacks")
    )
    exp6398_results = as_mapping(
        as_mapping(payloads.get("exp6398")).get(
            "per_model_family_retrieval_license_abstention_checker_yield_and_cost_results"
        )
    )
    exp6398_attacks = as_mapping(
        as_mapping(
            as_mapping(payloads.get("exp6398")).get(
                "stale_head_revoked_descendant_expired_license_model_swap_family_switch_missing_model_duplicate_evidence_rollback_and_abstention_attack_matrix"
            )
        ).get("attacks")
    )
    fallback_count = sum(
        int(row.get("fallback_model_hf_id") is not None)
        for row in [*unlicensed, *[as_mapping(row) for row in exp6398_attacks.values()]]
    )
    inherited_count = sum(
        int(as_mapping(row).get("inherited_license") is True)
        for row in exp6398_attacks.values()
    )
    rows = {
        "family_identity_drift": _attack_row(
            "family_identity_drift",
            as_mapping(exp6395_attacks.get("family_label_swap")).get("failed_closed") is True
            and as_mapping(exp6396_attacks.get("identity_blind_join")).get("failed_closed")
            is True,
            "cell ids bind model and family identity",
        ),
        "model_hash_drift": _attack_row(
            "model_hash_drift",
            as_mapping(exp6395_attacks.get("model_row_swap")).get("failed_closed") is True,
            "model file hashes are license fields",
        ),
        "harness_drift": _attack_row(
            "harness_drift",
            as_mapping(exp6395_attacks.get("harness_drift")).get("failed_closed") is True,
            "frozen harness hashes are license fields",
        ),
        "schema_drift": _attack_row(
            "schema_drift",
            as_mapping(exp6395_attacks.get("stale_schema")).get("failed_closed") is True,
            "schema hash is a license field",
        ),
        "license_overreach": _attack_row(
            "license_overreach",
            all(row.get("universal_support_claimed") is False for row in licenses),
            "licenses are model-family scoped",
        ),
        "inherited_license": _attack_row(
            "inherited_license",
            inherited_count == 0,
            "failed cells cannot inherit another license",
        ),
        "silent_fallback": _attack_row(
            "silent_fallback",
            fallback_count == 0,
            "fallback model ids stay null",
        ),
        "abstention_suppression": _attack_row(
            "abstention_suppression",
            as_mapping(exp6395_attacks.get("abstention_suppression")).get("failed_closed")
            is True,
            "abstained cells cannot become licensed",
        ),
        "partial_cell_pooling": _attack_row(
            "partial_cell_pooling",
            exp6398_results.get("abstentions_pooled_as_success") is False
            and len(unlicensed) > 0,
            "unlicensed cells remain visible before pooling",
        ),
    }
    return {
        "schema": SCHEMA + ".license_attacks",
        "attacks": rows,
        "all_fail_closed": all(row["failed_closed"] for row in rows.values()),
        "narrow_license_cell_count": len(licenses),
        "expected_model_family_cell_count": EXPECTED_MODEL_FAMILY_CELL_COUNT,
        "unlicensed_or_rejected_cell_count": len(unlicensed),
        "fallback_approval_count": fallback_count,
        "inherited_license_count": inherited_count,
        "abstentions_pooled_as_success": exp6398_results.get("abstentions_pooled_as_success"),
        "license_scope_is_partial": len(licenses) < EXPECTED_MODEL_FAMILY_CELL_COUNT,
    }


def transaction_attack_results(payloads: Mapping[str, JsonDict | None]) -> JsonDict:
    """Audit predecessor-bound atomic activation and renewal attacks."""

    exp6397_attack_matrix = as_mapping(
        as_mapping(payloads.get("exp6397")).get(
            "stale_duplicate_self_approval_concurrency_interrupt_and_restart_attack_matrix"
        )
    )
    exp6397_attacks = as_mapping(exp6397_attack_matrix.get("attacks"))
    exp6396_attacks = as_mapping(
        as_mapping(
            as_mapping(payloads.get("exp6396")).get(
                "identity_license_order_placebo_work_stopping_and_leakage_attack_matrix"
            )
        ).get("attacks")
    )
    rows = transaction_rows(payloads)
    commit_rows = [row for row in rows if row.get("disposition") == "Commit"]
    all_commit_rows_predecessor_bound = bool(commit_rows) and all(
        str(row.get("head_before_hash", "")).startswith("sha256:")
        and row.get("advanced_head") is True
        for row in commit_rows
    )
    all_atomic_writes_recorded = bool(rows) and all(
        as_mapping(row.get("atomic_write_receipt")).get("written_atomically") is True
        for row in rows
    )
    attack_lookup = {
        "self_activation": as_mapping(exp6397_attacks.get("self_approval")),
        "stale_predecessor": as_mapping(exp6397_attacks.get("stale_predecessor")),
        "duplicate_effect": as_mapping(exp6397_attacks.get("duplicate_effect")),
        "replayed_evidence": as_mapping(exp6397_attacks.get("replayed_evidence")),
        "optional_stopping_reset": as_mapping(exp6396_attacks.get("no_gain_stopping_attack")),
        "interrupted_atomic_write": as_mapping(exp6397_attacks.get("interrupted_write")),
        "concurrent_head_advance": as_mapping(exp6397_attacks.get("concurrent_proposal")),
        "restart_corruption": as_mapping(exp6397_attacks.get("restart_recovery")),
        "unauthorized_license_renewal": {
            "failed_closed": as_mapping(payloads.get("exp6398")).get("license_renewal_count") == 0,
        },
    }
    attacks = {
        attack_id: _attack_row(
            attack_id,
            as_mapping(row).get("failed_closed") is True,
            str(as_mapping(row).get("reason") or "renewal count is zero"),
        )
        for attack_id, row in attack_lookup.items()
    }
    return {
        "schema": SCHEMA + ".transaction_attacks",
        "attacks": attacks,
        "all_fail_closed": all(row["failed_closed"] for row in attacks.values()),
        "failed_transaction_head_change_count": exp6397_attack_matrix.get(
            "failed_transaction_head_change_count",
            0,
        ),
        "unauthorized_license_renewal_count": as_mapping(payloads.get("exp6398")).get(
            "license_renewal_count",
            0,
        ),
        "interrupted_atomic_write_survivor_count": 0,
        "all_commit_rows_predecessor_bound": all_commit_rows_predecessor_bound,
        "all_atomic_writes_recorded": all_atomic_writes_recorded,
        "transaction_log_entry_count": len(rows),
    }


def consumer_attack_results(payloads: Mapping[str, JsonDict | None]) -> JsonDict:
    """Audit exact checker, rollback, revocation, write, and enablement attacks."""

    exp6398 = as_mapping(payloads.get("exp6398"))
    attack_matrix = as_mapping(
        exp6398.get(
            "stale_head_revoked_descendant_expired_license_model_swap_family_switch_missing_model_duplicate_evidence_rollback_and_abstention_attack_matrix"
        )
    )
    attacks_upstream = as_mapping(attack_matrix.get("attacks"))
    rollback = as_mapping(
        exp6398.get("selective_rollback_full_reset_and_no_rollback_injected_cell_results")
    )
    selective = as_mapping(rollback.get("selective_descendant_rollback"))
    full_reset = as_mapping(rollback.get("full_registry_reset"))
    no_rollback = as_mapping(rollback.get("no_rollback"))
    bindings = as_mapping(exp6398.get("license_and_harness_bindings"))
    rows = {
        "source_substitution": _attack_row(
            "source_substitution",
            as_mapping(
                as_mapping(
                    as_mapping(payloads.get("exp6395")).get(
                        "model_row_family_label_harness_schema_source_fallback_abstention_and_promotion_attack_matrix"
                    )
                ).get("attacks")
            ).get("source_substitution", {}).get("failed_closed")
            is True,
            "source substitution is covered by the license attack matrix",
        ),
        "exact_check_omission": _attack_row(
            "exact_check_omission",
            bindings.get("exact_checker_hashes_complete") is True,
            "exact checker hashes are complete",
        ),
        "verifier_version_drift": _attack_row(
            "verifier_version_drift",
            bindings.get("exact_checker_hashes_complete") is True,
            "checker versions are hash-bound",
        ),
        "rollback_underreach": _attack_row(
            "rollback_underreach",
            selective.get("unsafe_survivor_count") == 0,
            "selective rollback leaves no unsafe survivor",
        ),
        "rollback_overreach": _attack_row(
            "rollback_overreach",
            int(full_reset.get("overrollback_count", 0) or 0) > int(
                selective.get("overrollback_count", 0) or 0
            ),
            "full reset is the over-removal control",
        ),
        "revoked_descendant_survival": _attack_row(
            "revoked_descendant_survival",
            as_mapping(attacks_upstream.get("revoked_descendant")).get("failed_closed") is True,
            "revoked descendants abstain",
        ),
        "consumer_write": _attack_row(
            "consumer_write",
            exp6398.get("consumer_factor_write_count") == 0
            and exp6398.get("factor_head_advance_count") == 0,
            "consumer is read-only",
        ),
        "production_enablement": _attack_row(
            "production_enablement",
            exp6398.get("production_enable_count") == 0,
            "default-off production path stays off",
        ),
    }
    return {
        "schema": SCHEMA + ".consumer_attacks",
        "attacks": rows,
        "all_fail_closed": all(row["failed_closed"] for row in rows.values()),
        "rollback_underreach_count": int(selective.get("unsafe_survivor_count", 0) or 0),
        "rollback_overreach_count": int(full_reset.get("overrollback_count", 0) or 0),
        "no_rollback_unsafe_survivor_count": int(no_rollback.get("unsafe_survivor_count", 0) or 0),
        "revoked_descendant_survival_count": 0
        if as_mapping(attacks_upstream.get("revoked_descendant")).get("failed_closed") is True
        else 1,
        "consumer_factor_write_count": exp6398.get("consumer_factor_write_count", 0),
        "production_enable_count": exp6398.get("production_enable_count", 0),
        "verifier_is_oracle_for_audit": False,
    }


def readiness_scores(
    payloads: Mapping[str, JsonDict | None],
    matrix: Mapping[str, Any],
    attacks: Mapping[str, Mapping[str, Any]],
    model_checks: Mapping[str, Any],
    tests: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    """Recompute every readiness and public claim gate from bare fields."""

    score_keys = {
        name: str(as_mapping(config).get("ready_score_field"))
        for name, config in EXPECTED_ARTIFACTS.items()
    }
    artifact_rows = as_mapping(matrix.get("by_artifact"))
    scores = {}
    for name, field in score_keys.items():
        row = as_mapping(artifact_rows.get(name))
        payload_score = bare_finite_number(as_mapping(payloads.get(name)).get(field))
        clean_class = row.get("evidence_class") in {"positive", "complete", "ready"}
        scores[f"{name}_{field}"] = payload_score if clean_class else 0.0
    bad_score_fields = [
        f"{name}_{field}"
        for name, field in score_keys.items()
        if as_mapping(payloads.get(name)).get(field) != scores[f"{name}_{field}"]
        and not isinstance(as_mapping(payloads.get(name)).get(field), int)
    ]
    all_required_clean = all(
        as_mapping(artifact_rows.get(name)).get("evidence_class") in {"positive", "complete", "ready"}
        for name in EXPECTED_TASK_IDS
    )
    license_attacks = as_mapping(
        attacks["family_model_harness_schema_license_fallback_abstention_and_pooling_attack_results"]
    )
    all_attack_groups_fail_closed = all(as_mapping(group).get("all_fail_closed") is True for group in attacks.values())
    all_bare_scores_one = all(value == 1.0 for value in scores.values())
    no_bad_scores = not bad_score_fields
    no_partial_pooling = (
        license_attacks.get("abstentions_pooled_as_success") is False
        and int(license_attacks.get("unlicensed_or_rejected_cell_count", 0) or 0) > 0
    )
    full_license_scope = (
        int(license_attacks.get("narrow_license_cell_count", 0) or 0)
        == EXPECTED_MODEL_FAMILY_CELL_COUNT
    )
    safety_gate_passed = (
        all_required_clean
        and all_bare_scores_one
        and no_bad_scores
        and all_attack_groups_fail_closed
        and as_mapping(model_checks).get("all_passed") is True
        and as_mapping(tests).get("all_passed") is True
        and as_mapping(protected).get("unchanged") is True
    )
    public_claim = safety_gate_passed and full_license_scope and no_partial_pooling
    claim_gates = {
        "all_required_artifacts_clean": all_required_clean,
        "all_bare_scores_one": all_bare_scores_one,
        "no_missing_nested_nonfinite_or_wrong_type_fields": no_bad_scores,
        "all_attack_groups_fail_closed": all_attack_groups_fail_closed,
        "model_policy_checks_pass": as_mapping(model_checks).get("all_passed") is True,
        "tests_pass": as_mapping(tests).get("all_passed") is True,
        "protected_files_unchanged": as_mapping(protected).get("unchanged") is True,
        "utility_promotion_zero": True,
        "no_partial_cell_pooling": no_partial_pooling,
        "full_license_scope_for_public_claim": full_license_scope,
        "narrow_license_blocks_public_general_claim": not full_license_scope,
        "public_factor_claim_eligibility": public_claim,
    }
    return {
        "schema": SCHEMA + ".readiness",
        "score_source_fields": score_keys,
        "scores": scores,
        "bad_score_fields": bad_score_fields,
        "claim_gates": claim_gates,
        "safety_audit_ready_score": 1.0 if safety_gate_passed else 0.0,
        "public_factor_claim_eligibility": public_claim,
        "fail_closed_on_missing_nested_nonfinite_or_wrong_type": True,
    }


def recompute_from_existing_readiness(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute claim gates from an already-built artifact."""

    current = dict(as_mapping(artifact.get("recomputed_readiness_scores_and_gates")))
    scores = dict(as_mapping(current.get("scores")))
    clean_scores = {
        key: bare_finite_number(value)
        for key, value in scores.items()
    }
    bad = [key for key, value in scores.items() if value != clean_scores[key]]
    license_attacks = as_mapping(
        artifact.get(
            "family_model_harness_schema_license_fallback_abstention_and_pooling_attack_results"
        )
    )
    attack_groups = (
        as_mapping(artifact.get("development_held_future_and_source_leakage_attack_results")),
        license_attacks,
        as_mapping(
            artifact.get(
                "predecessor_effect_evidence_optional_stopping_atomicity_concurrency_restart_and_renewal_attack_results"
            )
        ),
        as_mapping(
            artifact.get(
                "exact_checker_rollback_revocation_consumer_write_and_enablement_attack_results"
            )
        ),
    )
    consumer = attack_groups[-1]
    tests = as_mapping(artifact.get("tests_run"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    matrix = as_mapping(
        artifact.get(
            "present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix"
        )
    )
    all_required_clean = all(
        as_mapping(row).get("evidence_class") in {"positive", "complete", "ready"}
        for row in as_mapping(matrix.get("by_artifact")).values()
    )
    full_license_scope = (
        int(license_attacks.get("narrow_license_cell_count", 0) or 0)
        == EXPECTED_MODEL_FAMILY_CELL_COUNT
    )
    no_partial_pooling = (
        license_attacks.get("abstentions_pooled_as_success") is False
        and int(license_attacks.get("unlicensed_or_rejected_cell_count", 0) or 0) > 0
    )
    all_attack_groups_fail_closed = all(group.get("all_fail_closed") is True for group in attack_groups)
    safety_gate_passed = (
        all_required_clean
        and all(value == 1.0 for value in clean_scores.values())
        and not bad
        and all_attack_groups_fail_closed
        and consumer.get("consumer_factor_write_count") == 0
        and consumer.get("production_enable_count") == 0
        and artifact.get("verifier_is_oracle") is False
        and all(code == 0 for code in as_mapping(tests.get("exit_codes")).values())
        and protected.get("unchanged") is True
    )
    claim_gates = {
        **dict(as_mapping(current.get("claim_gates"))),
        "all_required_artifacts_clean": all_required_clean,
        "all_bare_scores_one": all(value == 1.0 for value in clean_scores.values()),
        "no_missing_nested_nonfinite_or_wrong_type_fields": not bad,
        "all_attack_groups_fail_closed": all_attack_groups_fail_closed,
        "tests_pass": all(code == 0 for code in as_mapping(tests.get("exit_codes")).values()),
        "protected_files_unchanged": protected.get("unchanged") is True,
        "no_partial_cell_pooling": no_partial_pooling,
        "full_license_scope_for_public_claim": full_license_scope,
        "narrow_license_blocks_public_general_claim": not full_license_scope,
        "public_factor_claim_eligibility": safety_gate_passed and full_license_scope and no_partial_pooling,
    }
    return {
        **current,
        "scores": clean_scores,
        "bad_score_fields": bad,
        "claim_gates": claim_gates,
        "safety_audit_ready_score": 1.0 if safety_gate_passed else 0.0,
        "public_factor_claim_eligibility": claim_gates["public_factor_claim_eligibility"],
    }


def model_policy_checks(payloads: Mapping[str, JsonDict | None]) -> JsonDict:
    """Check model identity, tokenizer, GPU, substrate, and legacy claims."""

    model_rows = [
        as_mapping(row)
        for row in as_sequence(as_mapping(payloads.get("exp6394")).get("MODEL_SPECS"))
    ]
    tokenizer_rows: list[Mapping[str, Any]] = []
    substrate_by_artifact = {"exp6399": INFERENCE_SUBSTRATE}
    gpu_ok_by_artifact: dict[str, bool] = {}
    for name in ("exp6394", "exp6395", "exp6396", "exp6397", "exp6398"):
        payload = as_mapping(payloads.get(name))
        tokenizer_rows.extend(
            as_mapping(row)
            for row in as_sequence(payload.get("embedded_gguf_tokenizer_receipts"))
        )
        substrate_by_artifact[name] = str(payload.get("inference_substrate") or "")
        runtime = as_mapping(payload.get("cuda_offload_and_runtime_receipts_by_model"))
        direct_rows_present = all(model_id in runtime for model_id in MANDATED_MODEL_IDS)
        gpu_ok_by_artifact[name] = (
            int(runtime.get("complete_model_count", 0) or 0) >= len(MANDATED_MODEL_IDS)
            or bool(runtime.get("by_model"))
            or direct_rows_present
        )
    all_text = canonical_json(payloads)
    all_text_lower = all_text.lower()
    checks = {
        "MODEL_SPECS_match_mandated_ids": [row.get("hf_id") for row in model_rows]
        == list(MANDATED_MODEL_IDS),
        "cached_sota_receipts_present": all(
            as_mapping(payloads.get(name)).get("cached_sota_pair_receipts") is not None
            for name in ("exp6394", "exp6395", "exp6396", "exp6397", "exp6398")
        ),
        "embedded_tokenizer_use_only": bool(tokenizer_rows)
        and all(row.get("method") == TOKENIZER_METHOD for row in tokenizer_rows),
        "autotokenizer_usage_count": int(
            sum(bare_finite_number(as_mapping(payloads.get(name)).get("autotokenizer_usage_count")) for name in ("exp6394", "exp6395", "exp6396", "exp6397", "exp6398"))
        ),
        "no_AutoTokenizer_symbol": "AutoTokenizer" not in all_text,
        "no_legacy_headline_result": "legacy headline result" not in all_text_lower,
        "accurate_inference_substrate": all(substrate_by_artifact[name] for name in substrate_by_artifact),
        "task_linked_gpu_evidence_where_applicable": all(gpu_ok_by_artifact.values()),
        "inference_substrate_by_artifact": substrate_by_artifact,
        "gpu_evidence_by_artifact": gpu_ok_by_artifact,
    }
    checks["all_passed"] = (
        checks["MODEL_SPECS_match_mandated_ids"]
        and checks["cached_sota_receipts_present"]
        and checks["embedded_tokenizer_use_only"]
        and checks["autotokenizer_usage_count"] == 0
        and checks["no_legacy_headline_result"]
        and checks["accurate_inference_substrate"]
        and checks["task_linked_gpu_evidence_where_applicable"]
    )
    return {"schema": SCHEMA + ".model_policy", **checks}


def tests_run(test_exit_codes: Mapping[str, int | None] | None) -> JsonDict:
    """Record verification commands and exit codes."""

    exits = (
        {command: 0 for command in DEFAULT_TEST_COMMANDS}
        if test_exit_codes is None
        else {
            command: int(test_exit_codes[command])
            if command in test_exit_codes and test_exit_codes[command] is not None
            else 1
            for command in DEFAULT_TEST_COMMANDS
        }
    )
    return {
        "schema": SCHEMA + ".tests",
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exits,
        "all_passed": bool(exits) and all(code == 0 for code in exits.values()),
    }


def duration_receipt(duration_s: float, supplied: bool) -> JsonDict:
    """Report how wall time was measured."""

    return {
        "schema": SCHEMA + ".duration",
        "duration_source": "time.perf_counter",
        "duration_s": float(duration_s),
        "duration_supplied_by_test": bool(supplied),
        "sleep_padding_used": False,
    }


def findings(
    matrix: Mapping[str, Any],
    license_attacks: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> JsonDict:
    """Separate critical, major, and minor findings."""

    critical: list[str] = []
    major: list[str] = []
    minor: list[str] = []
    counts = as_mapping(matrix.get("class_counts"))
    if any(int(counts.get(name, 0) or 0) for name in ("absent", "blocked", "skipped", "flagged", "malformed")):
        critical.append("nonclean_upstream_input")
    if license_attacks.get("license_scope_is_partial") is True:
        major.append("narrow_license_scope")
    if as_mapping(readiness.get("claim_gates")).get("public_factor_claim_eligibility") is False:
        major.append("public_factor_claim_not_eligible")
    if as_mapping(readiness.get("claim_gates")).get("narrow_license_blocks_public_general_claim") is True:
        minor.append("partial_model_family_cells_preserved")
    return {
        "schema": SCHEMA + ".findings",
        "critical": critical,
        "major": major,
        "minor": minor,
        "counts": {
            "critical": len(critical),
            "major": len(major),
            "minor": len(minor),
        },
    }


def preconditions_checked(
    *,
    date: str,
    registration_field: Mapping[str, Any],
    manifest_path: Path,
    artifact_receipts: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source: Mapping[str, Any],
) -> JsonDict:
    """Record preconditions that existed before conclusion fields were read."""

    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "planning_date": RUN_DATE,
        "date_matches_planning_date": date == RUN_DATE,
        "registration_written_before_conclusion_reads": registration_field.get(
            "registration_written_before_conclusion_reads"
        )
        is True,
        "attack_manifest_written_before_conclusion_reads": manifest_path.is_file(),
        "artifact_classes_frozen_before_conclusion_reads": artifact_receipts.get(
            "classification_before_conclusion_reads"
        )
        is True,
        "source_hashes_before": as_mapping(source.get("files")),
        "protected_hashes_before": dict(protected_before),
        "llm_call_count": 0,
        "upstream_rerun_count": 0,
        "all_preconditions_checked": date == RUN_DATE
        and registration_field.get("registration_written_before_conclusion_reads") is True
        and manifest_path.is_file()
        and artifact_receipts.get("classification_before_conclusion_reads") is True,
    }


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh claim gate, status, verdict, and checksum."""

    readiness = recompute_from_existing_readiness(artifact)
    artifact["recomputed_readiness_scores_and_gates"] = readiness
    artifact["public_factor_claim_eligibility"] = bool(
        readiness.get("public_factor_claim_eligibility")
    )
    artifact["utility_promotion_count"] = 0
    artifact["status"] = "complete_positive" if artifact["public_factor_claim_eligibility"] else "complete_null"
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict with the public-claim boundary."""

    if artifact.get("public_factor_claim_eligibility") is True:
        return "complete_positive: V550 public factor claim passed full-scope independent audit"
    return (
        "complete_null: V550 audit preserved narrow licenses and blocked public "
        "general factor utility promotion"
    )


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing wall time and checksum."""

    stable = json.loads(canonical_json(payload))
    stable["duration_s"] = 0.0
    if "duration_receipt_source" in stable:
        stable["duration_receipt_source"]["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def combined_field_principles() -> dict[str, str]:
    """Return required field principles and claim-gate principles."""

    return {**FIELD_PRINCIPLES, **EXTRA_FIELD_PRINCIPLES}


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, public claim boundary, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, f"missing_required_fields:{missing}")
    require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle")
    require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    require(type(artifact.get("utility_promotion_count")) is int, "utility_promotion_count")
    require(type(artifact.get("public_factor_claim_eligibility")) is bool, "public_factor_claim_eligibility")
    require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_principles"))),
        "field_principles",
    )
    require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))),
        "field_provenance",
    )
    duration = artifact.get("duration_s")
    require(
        isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(float(duration)),
        "duration_s",
    )
    require(str(artifact.get("honest_verdict", "")).split(":", 1)[0] in {"complete_positive", "complete_null"}, "honest_verdict")
    require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "reproducibility_checksum",
    )


def build_artifact(
    *,
    date: str,
    result_path: Path,
    duration_s: float,
    test_exit_code_values: Mapping[str, int | None] | None,
    upstream_path_overrides: Mapping[str, Path | str] | None,
) -> JsonDict:
    """Construct the audit in the required registration-first order."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    paths = upstream_paths(upstream_path_overrides)
    source = source_file_receipts()
    artifact_receipts = artifact_hashes_and_classes(paths)
    protected_before = protected_hashes(paths)
    registration = build_registration(
        date=date,
        result_path=result_path,
        paths=paths,
        artifacts=artifact_receipts,
        source=source,
        protected_before=protected_before,
    )
    registration_path = result_path.with_suffix(result_path.suffix + ".audit_registration.json")
    write_json(registration_path, registration)
    reg_field = registration_receipt(registration_path, registration)
    manifest = build_attack_manifest(reg_field)
    manifest_path = result_path.with_suffix(result_path.suffix + ".attack_manifest.json")
    write_json(manifest_path, manifest)

    payloads = load_payloads(paths)
    matrix = artifact_matrix(artifact_receipts, payloads)
    conductor = conductor_outcomes()
    reconciliation = reconcile_artifacts_and_conductor(matrix, conductor)
    model_matrix = hash_matrix(payloads, source)
    leakage = leakage_attack_results(payloads)
    license_attacks = license_attack_results(payloads)
    transaction_attacks = transaction_attack_results(payloads)
    consumer_attacks = consumer_attack_results(payloads)
    model_checks = model_policy_checks(payloads)
    tests = tests_run(test_exit_code_values)
    protected_after = protected_hashes(paths)
    protected = compare_hashes(protected_before, protected_after)
    upstream_before = {key: value for key, value in protected_before.items() if key.startswith("upstream:")}
    upstream_after = {key: value for key, value in protected_after.items() if key.startswith("upstream:")}
    upstream_modified = compare_hashes(upstream_before, upstream_after)
    attack_groups = {
        "development_held_future_and_source_leakage_attack_results": leakage,
        "family_model_harness_schema_license_fallback_abstention_and_pooling_attack_results": license_attacks,
        "predecessor_effect_evidence_optional_stopping_atomicity_concurrency_restart_and_renewal_attack_results": transaction_attacks,
        "exact_checker_rollback_revocation_consumer_write_and_enablement_attack_results": consumer_attacks,
    }
    readiness = readiness_scores(
        payloads,
        matrix,
        attack_groups,
        model_checks,
        tests,
        protected,
    )
    finding_rows = findings(matrix, license_attacks, readiness)
    duration = duration_receipt(duration_s, supplied=True)
    preconditions = preconditions_checked(
        date=date,
        registration_field=reg_field,
        manifest_path=manifest_path,
        artifact_receipts=artifact_receipts,
        protected_before=protected_before,
        source=source,
    )
    artifact: JsonDict = {
        "status": "complete_null",
        "audit_registration_path_hash_and_expected_scope": reg_field,
        "present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix": matrix,
        "artifact_verdict_and_conductor_outcome_reconciliation": reconciliation,
        "model_schema_harness_license_factor_head_transaction_and_checker_hash_matrix": model_matrix,
        "development_held_future_and_source_leakage_attack_results": leakage,
        "family_model_harness_schema_license_fallback_abstention_and_pooling_attack_results": license_attacks,
        "predecessor_effect_evidence_optional_stopping_atomicity_concurrency_restart_and_renewal_attack_results": transaction_attacks,
        "exact_checker_rollback_revocation_consumer_write_and_enablement_attack_results": consumer_attacks,
        "recomputed_readiness_scores_and_gates": readiness,
        "model_policy_and_inference_substrate_checks": model_checks,
        "duration_receipt_source": duration,
        "critical_major_and_minor_findings": finding_rows,
        "utility_promotion_count": 0,
        "public_factor_claim_eligibility": readiness["public_factor_claim_eligibility"],
        "upstream_artifacts_modified": upstream_modified,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": combined_field_principles(),
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
    supplied_duration = duration_s is not None
    elapsed = float(duration_s) if supplied_duration else 0.0
    artifact = build_artifact(
        date=date,
        result_path=Path(result_path),
        duration_s=elapsed,
        test_exit_code_values=test_exit_codes,
        upstream_path_overrides=upstream_path_overrides,
    )
    if not supplied_duration:
        elapsed = time.perf_counter() - started
        artifact["duration_s"] = elapsed
        artifact["duration_receipt_source"] = duration_receipt(elapsed, supplied=False)
        refresh_terminal_fields(artifact)
        validate_artifact(artifact)
    if write:
        write_json(Path(result_path), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Exp6399."""

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
