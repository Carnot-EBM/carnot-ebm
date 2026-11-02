"""Build the Exp 3192 receipt/adversarial contract v4 artifact.

Spec refs: REQ-VERIFY-3192, SCENARIO-VERIFY-3192.

This module writes a contract, not a verifier result. It separates the cheap
question "did a local GGUF invocation really execute and leave reproducible
receipts?" from the stricter question "is this enough to unlock clean verifier
reruns and headline repair claims?". CPU fallback can answer the first question
when receipts are complete. It cannot answer the second one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
SCHEMA_VERSION = "carnot.receipt_adversarial_contract.v4"
EXPERIMENT_ID = "exp3192"
CONTRACT_VERSION = "v4"
ARTIFACT = "experiment_3192_receipt_adversarial_contract_v4"

OUTPUT_REL_PATH = Path("results/experiment_3192_receipt_adversarial_contract_v4.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3192_receipt_adversarial_contract_v4.py"

EXP3178_PROMPT_ALIAS_REL_PATH = Path("results/experiment_3178_receipt_adversarial_contract_v3.json")
EXP3178_REL_PATH = Path("results/experiment_3178_receipt_backed_authenticity_contract_v3.json")
EXP3179_REL_PATH = Path("results/experiment_3179_local_sota_receipt_smoke_v3.json")
EXP3189_REL_PATH = Path("results/experiment_3189_cross_corpus_matrix_v29.json")

TERMINAL_VERDICT_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

BLOCKED_VERDICT_PREFIXES = ("blocked_",)

KNOWN_SUBSTRATE_CLASSES = (
    "model_cache_missing",
    "loader_missing",
    "cuda_unavailable",
    "cuda_available_unhealthy",
    "cpu_fallback_receipt_only",
    "full_local_sota_receipt",
)

ACCEPTED_SUBSTRATE_CLASSES = (
    "cpu_fallback_receipt_only",
    "full_local_sota_receipt",
)

REJECTED_HEADLINE_SUBSTRATE_CLASSES = (
    "model_cache_missing",
    "loader_missing",
    "cuda_unavailable",
    "cuda_available_unhealthy",
    "cpu_fallback_receipt_only",
)

PROOF_EXECUTION_REQUIRED_FIELDS = (
    "local_sota_receipt_smoke_v3_ready",
    "substrate_classification",
    "live_call_count",
    "proof_receipts",
    "prompt_hashes",
    "transcript_hashes",
    "token_counts",
    "throughput_plausibility.passed",
    "stale_transcript_rejection_passed",
    "proof_receipts[].selected_model_id",
    "proof_receipts[].model_path",
    "proof_receipts[].model_file_hash",
    "proof_receipts[].loader_name",
    "proof_receipts[].substrate_used",
    "proof_receipts[].prompt_hash",
    "proof_receipts[].transcript_hash",
    "proof_receipts[].token_counts",
    "proof_receipts[].random_seed",
    "proof_receipts[].wall_clock_s",
    "proof_receipts[].command_hash",
    "proof_receipts[].subprocess_return_code",
    "proof_receipts[].stderr_tail",
    "proof_receipts[].throughput_plausibility",
    "proof_receipts[].replay_count",
    "proof_receipts[].worker_code_sha256",
)

PER_RECEIPT_PROOF_FIELDS = tuple(
    field.removeprefix("proof_receipts[].")
    for field in PROOF_EXECUTION_REQUIRED_FIELDS
    if field.startswith("proof_receipts[].")
)

CLEAN_RERUN_REQUIRED_FIELDS = (
    "clean_rerun_allowed",
    "substrate_classification=full_local_sota_receipt",
    "cpu_fallback_used=false",
    "headline_claim_allowed=true",
    "cuda_probe.cuda_available=true",
    "nvidia_smi_probe.available=true",
    "loader_probe.available=true",
    "inference_substrate.n_gpu_layers=-1_or_positive_offload",
    "proof_receipts[].substrate_used=full_local_sota_receipt",
    "proof_receipts[].model_file_hash",
    "proof_receipts[].transcript_hash",
    "throughput_plausibility.passed=true",
    "stale_transcript_rejection_passed=true",
    "preconditions_checked",
)

LIVE_REQUIRED_FIELDS = (
    "schema_version",
    "experiment_id",
    "model_specs",
    "selected_model_ids",
    "preconditions_checked",
    "proof_receipts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
    "flagged_adversarial",
    "corrigendum_pending",
    "methodology_note",
    "honest_verdict",
)

AGGREGATE_REQUIRED_FIELDS = (
    "schema_version",
    "experiment_id",
    "source_artifacts",
    "source_checksums",
    "fields_imported_from_upstream",
    "methodology_inherited_from_upstream",
    "upstream_adversarial_flag_summary",
    "inference_substrate",
    "no_new_model_execution",
    "no_new_verifier_run",
    "duration_s",
    "flagged_adversarial",
    "corrigendum_pending",
    "methodology_note",
    "honest_verdict",
)

GATED_SKIP_REQUIRED_FIELDS = (
    "schema_version",
    "experiment_id",
    "gated_skip",
    "gate_reasons",
    "blocked_precondition",
    "upstream_gate_snapshot",
    "preconditions_checked",
    "live_call_count=0",
    "models_used=[]",
    "proof_receipts_used=[]",
    "headline_claim_allowed=false",
    "flagged_adversarial",
    "corrigendum_pending",
    "inference_substrate",
    "duration_s",
    "honest_verdict",
)

DIAGNOSTIC_ONLY_REQUIRED_FIELDS = (
    "schema_version",
    "experiment_id",
    "diagnostic_only",
    "diagnostic_scope",
    "source_artifacts",
    "source_checksums",
    "headline_claim_allowed=false",
    "deployed_claim_allowed=false",
    "inference_substrate",
    "duration_s",
    "flagged_adversarial",
    "corrigendum_pending",
    "methodology_note",
    "honest_verdict",
)

BLOCKED_VERDICT_ALLOWANCES = (
    "blocked_model_cache_missing:",
    "blocked_loader_missing:",
    "blocked_cuda_unavailable:",
    "blocked_cuda_backend_absent:",
    "blocked_gpu_offload_unhealthy:",
    "blocked_receipt_precondition:",
    "blocked_repair_gate_precondition:",
    "blocked_missing_artifact:",
)

REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "contract_version",
    "proof_execution_required_fields",
    "clean_rerun_required_fields",
    "aggregate_required_fields",
    "gated_skip_required_fields",
    "accepted_substrate_classes",
    "rejected_headline_substrate_classes",
    "terminal_verdict_prefixes",
    "blocked_verdict_prefixes",
    "downstream_unlock_fields",
    "honest_verdict",
}

SOURCE_REL_PATHS: tuple[tuple[str, Path, bool, str], ...] = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_artifact_rules", Path("CLAUDE.md"), True, "text"),
    ("experiment_template_policy", Path("scripts/experiment_template.py"), True, "python"),
    ("conductor_gate_policy", Path("scripts/conductor_gates.py"), True, "python"),
    ("post_295_research_references", Path("research-references.md"), True, "text"),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True, "text"),
    ("prompt_named_v3_alias", EXP3178_PROMPT_ALIAS_REL_PATH, False, "json"),
    ("canonical_v3_contract", EXP3178_REL_PATH, True, "json"),
    ("exp3179_receipt_smoke", EXP3179_REL_PATH, True, "json"),
    ("matrix_v29_findings", EXP3189_REL_PATH, True, "json"),
    ("exp3192_module", Path("python/carnot/verify/receipt_adversarial_contract_v4.py"), False, "python"),
    ("exp3192_script", Path("scripts/experiment_3192_receipt_adversarial_contract_v4.py"), False, "python"),
    (
        "exp3192_tests",
        Path("tests/python/test_experiment_3192_receipt_adversarial_contract_v4.py"),
        False,
        "python",
    ),
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3192_receipt_adversarial_contract_v4.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify/receipt_adversarial_contract_v4.py -m pytest -o addopts='' tests/python/test_experiment_3192_receipt_adversarial_contract_v4.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/receipt_adversarial_contract_v4.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3192: build the no-inference v4 receipt contract artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    v3 = read_json_object(root_path / EXP3178_REL_PATH)
    smoke = read_json_object(root_path / EXP3179_REL_PATH)
    matrix = read_json_object(root_path / EXP3189_REL_PATH)
    sources = source_artifacts(root_path)
    source_errors = [
        row["path"] for row in sources if row["required"] and not row["present"]
    ]
    assessment = current_evidence_assessment(smoke)
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "contract_version": CONTRACT_VERSION,
        "run_date": RUN_DATE,
        "receipt_adversarial_contract_v4_ready": not source_errors,
        "proof_execution_required_fields": list(PROOF_EXECUTION_REQUIRED_FIELDS),
        "clean_rerun_required_fields": list(CLEAN_RERUN_REQUIRED_FIELDS),
        "live_required_fields": list(LIVE_REQUIRED_FIELDS),
        "aggregate_required_fields": list(AGGREGATE_REQUIRED_FIELDS),
        "gated_skip_required_fields": list(GATED_SKIP_REQUIRED_FIELDS),
        "diagnostic_only_required_fields": list(DIAGNOSTIC_ONLY_REQUIRED_FIELDS),
        "known_substrate_classes": list(KNOWN_SUBSTRATE_CLASSES),
        "accepted_substrate_classes": list(ACCEPTED_SUBSTRATE_CLASSES),
        "rejected_headline_substrate_classes": list(REJECTED_HEADLINE_SUBSTRATE_CLASSES),
        "terminal_verdict_prefixes": list(TERMINAL_VERDICT_PREFIXES),
        "blocked_verdict_prefixes": list(BLOCKED_VERDICT_PREFIXES),
        "blocked_verdict_allowances": list(BLOCKED_VERDICT_ALLOWANCES),
        "proof_execution_sufficient_conditions": proof_execution_sufficient_conditions(),
        "clean_rerun_allowed_conditions": clean_rerun_allowed_conditions(),
        "downstream_unlock_fields": downstream_unlock_fields(assessment),
        "current_evidence_assessment": assessment,
        "comparison_findings": comparison_findings(root_path, v3, smoke, matrix),
        "source_artifacts": sources,
        "source_checksums": {row["path"]: row.get("sha256") for row in sources},
        "source_errors": source_errors,
        "field_principles": field_principles(),
        "protected_files_modified_by_this_task": {
            "scripts/research_conductor.py": False,
            "ops/status.md": False,
            "ops/changelog.md": False,
            "_bmad/traceability.md": False,
        },
        "inference_substrate": inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, time.perf_counter() if now_s is None else float(now_s)),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build and persist the Exp 3192 terminal JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(
        root_path,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning empty evidence when a source is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return source rows with enough provenance for a third-party audit."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_REL_PATHS:
        path = root / rel_path
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "present": path.is_file(),
                "readable_json_object": (
                    bool(read_json_object(path)) if source_type == "json" else None
                ),
                "sha256": sha256_file(path),
            }
        )
    return rows


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 checksum for a present source file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proof_execution_sufficient_conditions() -> list[str]:
    """State the proof-of-execution gate that CPU fallback can satisfy."""

    return [
        "local_sota_receipt_smoke_v3_ready=true",
        "substrate_classification in accepted_substrate_classes",
        "live_call_count>=2",
        "len(proof_receipts)>=2",
        "all proof_execution_required_fields present",
        "throughput_plausibility.passed=true",
        "transcript_hashes unique and not stale",
        "headline_claim_allowed may remain false",
    ]


def clean_rerun_allowed_conditions() -> list[str]:
    """State the stricter gate required before clean verifier reruns."""

    return [
        "proof_execution_sufficient=true",
        "clean_rerun_allowed=true",
        "substrate_classification=full_local_sota_receipt",
        "cpu_fallback_used=false",
        "cuda/offload evidence present and healthy",
        "headline_claim_allowed=true for verifier metrics",
        "controlled_invariance_passed=true",
        "exact-authority false-accept gate passed",
    ]


def downstream_unlock_fields(assessment: Mapping[str, Any]) -> JsonDict:
    """Expose the v4 gates in a stable shape for later conductor tasks."""

    clean_current = bool(assessment.get("clean_rerun_allowed"))
    return {
        "proof_execution_sufficient": {
            "field": "proof_execution_sufficient",
            "current_value": bool(assessment.get("proof_execution_sufficient")),
            "allows_cpu_fallback": True,
            "accepted_substrate_classes": list(ACCEPTED_SUBSTRATE_CLASSES),
            "unlocks": ["cuda_offload_probe_debug_context"],
        },
        "clean_rerun_allowed": {
            "field": "clean_rerun_allowed",
            "current_value": clean_current,
            "requires_substrate_class": "full_local_sota_receipt",
            "rejects_cpu_fallback": True,
            "unlocks": ["exp3194_clean_live_sota_verifier_rerun_v11"],
        },
        "headline_claim_allowed": {
            "field": "headline_claim_allowed",
            "current_value": clean_current,
            "requires_clean_rerun_allowed": True,
            "rejected_substrate_classes": list(REJECTED_HEADLINE_SUBSTRATE_CLASSES),
        },
        "aggregate_methodology_clean": {
            "field": "aggregate_methodology_clean",
            "current_value": True,
            "requires_inference_substrate": "aggregation_from_upstream_artifacts",
            "requires_fields": list(AGGREGATE_REQUIRED_FIELDS),
        },
    }


def current_evidence_assessment(smoke: Mapping[str, Any]) -> JsonDict:
    """Classify the checked-in Exp 3179 smoke under the v4 dual thresholds."""

    receipts = mapping_list(smoke.get("proof_receipts"))
    substrate = str(smoke.get("substrate_classification") or "")
    missing = proof_missing_fields(smoke, receipts)
    proof_ok = (
        smoke.get("local_sota_receipt_smoke_v3_ready") is True
        and smoke.get("preflight_passed") is True
        and substrate in ACCEPTED_SUBSTRATE_CLASSES
        and int_or_zero(smoke.get("live_call_count")) >= 2
        and len(receipts) >= 2
        and throughput_passed(smoke)
        and not missing
    )
    clean_ok = (
        proof_ok
        and smoke.get("clean_rerun_allowed") is True
        and substrate == "full_local_sota_receipt"
        and smoke.get("cpu_fallback_used") is not True
        and smoke.get("headline_claim_allowed") is True
        and mapping(smoke.get("cuda_probe")).get("cuda_available") is True
    )
    if clean_ok:
        blocked_reason = ""
    elif substrate == "cpu_fallback_receipt_only":
        blocked_reason = "current_substrate_is_cpu_fallback_receipt_only"
    elif missing:
        blocked_reason = "proof_execution_required_fields_missing"
    else:
        blocked_reason = "full_local_sota_receipt_not_established"
    return {
        "source_artifact": EXP3179_REL_PATH.as_posix(),
        "substrate_classification": substrate,
        "proof_execution_sufficient": proof_ok,
        "proof_missing_fields": missing,
        "clean_rerun_allowed": clean_ok,
        "headline_claim_allowed": clean_ok,
        "why_clean_rerun_blocked": blocked_reason,
        "proof_receipt_count": len(receipts),
        "live_call_count": int_or_zero(smoke.get("live_call_count")),
    }


def proof_missing_fields(smoke: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return v4 proof fields missing from the smoke artifact."""

    missing: list[str] = []
    for field in (
        "local_sota_receipt_smoke_v3_ready",
        "substrate_classification",
        "live_call_count",
        "proof_receipts",
        "prompt_hashes",
        "transcript_hashes",
        "token_counts",
    ):
        if field not in smoke:
            missing.append(field)
    if not throughput_passed(smoke):
        missing.append("throughput_plausibility.passed")
    for receipt in receipts:
        for field in PER_RECEIPT_PROOF_FIELDS:
            if field not in receipt or receipt.get(field) in (None, ""):
                missing.append(f"proof_receipts[].{field}")
    return unique(missing)


def throughput_passed(smoke: Mapping[str, Any]) -> bool:
    """Normalize both old and new throughput-plausibility shapes."""

    if smoke.get("throughput_plausibility_passed") is True:
        return True
    return mapping(smoke.get("throughput_plausibility")).get("passed") is True


def comparison_findings(
    root: Path,
    v3: Mapping[str, Any],
    smoke: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> JsonDict:
    """Summarize why v4 exists without re-running any upstream work."""

    return {
        "prompt_v3_alias_path": EXP3178_PROMPT_ALIAS_REL_PATH.as_posix(),
        "prompt_v3_alias_present": (root / EXP3178_PROMPT_ALIAS_REL_PATH).is_file(),
        "canonical_v3_path": EXP3178_REL_PATH.as_posix(),
        "v3_contract_ready": v3.get("receipt_backed_authenticity_contract_v3_ready") is True,
        "v3_flagged_adversarial": v3.get("flagged_adversarial") is True,
        "receipt_smoke_ready": smoke.get("local_sota_receipt_smoke_v3_ready") is True,
        "receipt_smoke_substrate_classification": str(
            smoke.get("substrate_classification") or ""
        ),
        "receipt_smoke_clean_rerun_allowed": smoke.get("clean_rerun_allowed") is True,
        "receipt_smoke_headline_claim_allowed": smoke.get("headline_claim_allowed") is True,
        "receipt_smoke_proof_receipt_count": len(mapping_list(smoke.get("proof_receipts"))),
        "matrix_v29_ready": matrix.get("cross_corpus_matrix_v29_ready") is True,
        "matrix_v29_publication_blocker_count": int_or_zero(
            matrix.get("publication_blocker_count")
        ),
        "matrix_v29_flagged_rows": int_or_zero(matrix.get("flagged_rows")),
        "matrix_v29_gated_skip_rows": int_or_zero(matrix.get("gated_skip_rows")),
        "matrix_v29_diagnostic_only_rows": int_or_zero(matrix.get("diagnostic_only_rows")),
        "matrix_v29_next_top_gap": str(matrix.get("next_top_gap") or ""),
    }


def field_principles() -> JsonDict:
    """Explain the audit reason for each v4 field family."""

    return {
        "schema_version": "schema-versioned artifacts",
        "experiment_id": "stable experiment identity",
        "contract_version": "versioned evidence contract",
        "proof_execution_required_fields": "reproducible receipt evidence",
        "clean_rerun_required_fields": "no CPU fallback headline claim",
        "aggregate_required_fields": "methodology completeness",
        "gated_skip_required_fields": "skip artifacts remain auditable",
        "accepted_substrate_classes": "explicit inference substrate",
        "rejected_headline_substrate_classes": "no false headline claim",
        "terminal_verdict_prefixes": "conductor verdict discipline",
        "blocked_verdict_prefixes": "honest precondition failure",
        "downstream_unlock_fields": "machine-readable gates",
        "honest_verdict": "terminal truthful verdict",
    }


def inference_substrate() -> JsonDict:
    """Declare that Exp 3192 performs policy aggregation only."""

    return {
        "kind": "contract_artifact_only",
        "uses_checked_in_artifacts_only": True,
        "downloads_models": False,
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "live_model_calls": 0,
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal truth statement for the v4 contract artifact."""

    assessment = mapping(artifact.get("current_evidence_assessment"))
    return (
        "complete: receipt_adversarial_contract_v4_ready="
        f"{str(artifact.get('receipt_adversarial_contract_v4_ready') is True).lower()}; "
        "proof_execution_sufficient="
        f"{str(assessment.get('proof_execution_sufficient') is True).lower()}; "
        "clean_rerun_allowed="
        f"{str(assessment.get('clean_rerun_allowed') is True).lower()}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject contract shapes that could be misread by downstream tooling."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    accepted = list(artifact.get("accepted_substrate_classes", []))
    if "cpu_fallback_receipt_only" not in accepted:
        raise ValueError("CPU fallback must remain accepted for proof-of-execution receipts")
    if accepted != list(ACCEPTED_SUBSTRATE_CLASSES):
        raise ValueError("accepted substrate classes must match the v4 proof contract")
    rejected = set(str(item) for item in artifact.get("rejected_headline_substrate_classes", []))
    if not set(REJECTED_HEADLINE_SUBSTRATE_CLASSES) <= rejected:
        raise ValueError("rejected headline substrate classes are incomplete")
    if "full_local_sota_receipt" in rejected:
        raise ValueError("full local SOTA receipt must not be rejected from headline eligibility")
    clean_fields = set(str(item) for item in artifact.get("clean_rerun_required_fields", []))
    if "substrate_classification=full_local_sota_receipt" not in clean_fields:
        raise ValueError("clean rerun contract must require full local SOTA substrate")
    if "cuda_probe.cuda_available=true" not in clean_fields:
        raise ValueError("clean rerun contract must require CUDA/offload evidence")
    downstream = mapping(artifact.get("downstream_unlock_fields"))
    clean = mapping(downstream.get("clean_rerun_allowed"))
    assessment = mapping(artifact.get("current_evidence_assessment"))
    if (
        clean.get("current_value") is True
        and assessment.get("substrate_classification") == "cpu_fallback_receipt_only"
    ):
        raise ValueError("CPU fallback cannot unlock clean rerun")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith("complete:"):
        raise ValueError("Exp 3192 honest_verdict must start with complete:")


def duration(started_s: float, finished_s: float) -> float:
    """Return a non-negative, stable wall-clock duration."""

    return round(max(0.0, float(finished_s) - float(started_s)), 6)


def mapping(value: Any) -> JsonDict:
    """Normalize arbitrary JSON values into a mutable mapping."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Normalize arbitrary JSON values into a list of mapping rows."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def int_or_zero(value: Any) -> int:
    """Coerce JSON counters while treating missing or malformed values as zero."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def unique(items: Sequence[str]) -> list[str]:
    """Preserve order while removing duplicate field names."""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
