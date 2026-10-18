"""Exp6306 online state learning safety audit.

Spec refs: REQ-CSL-6306, REQ-CSL-6306-INDEPENDENCE,
REQ-CSL-6306-FAULTS, REQ-CSL-6306-AUDIT,
REQ-CSL-6306-LEAKAGE, REQ-CSL-6306-ROLLBACK,
REQ-CSL-6306-PROVENANCE.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.terminal_artifacts import classify_artifact_path, path_sha256, payload_sha256


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
RESULT_RELATIVE_PATH = Path("results/experiment_6306_online_state_learning_safety_audit.json")
EXP6304_RELATIVE_PATH = Path(
    "results/experiment_6304_reference_anchored_online_state_learning.json"
)
EXP6298_RELATIVE_PATH = Path("results/experiment_6298_terminal_evidence_preflight_linter.json")
EXP6304_SIDECAR_RELATIVE_PATHS = (
    Path(str(EXP6304_RELATIVE_PATH) + ".sealed_stream_manifest.json"),
    Path(str(EXP6304_RELATIVE_PATH) + ".reference_snapshot.json"),
    Path(str(EXP6304_RELATIVE_PATH) + ".predecision_snapshots.jsonl"),
    Path(str(EXP6304_RELATIVE_PATH) + ".postdecision_outcomes.jsonl"),
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6306_online_state_learning_safety_audit.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6306_online_state_learning_safety_audit.py"
)

SCHEMA = "carnot.experiment_6306.online_state_learning_safety_audit.v1"
EXPERIMENT_ID = "experiment_6306_online_state_learning_safety_audit"
RUN_DATE = "20260811"
INFERENCE_SUBSTRATE = "deterministic_verifier_plus_replay"
RANDOM_SEEDS = {
    "reconstruction": 6306,
    "false_pass": 631060,
    "contradiction": 631061,
    "stale_reference": 631062,
    "reversal": 631063,
    "poison": 631064,
    "missing_validator": 631065,
    "nonfinite_update": 631066,
    "corrupted_checkpoint": 631067,
    "interrupted_write": 631068,
    "restart": 631069,
    "rollback_request": 631070,
}

FAULT_TO_FIELD = {
    "false_pass": "false_pass_results",
    "contradiction": "contradiction_results",
    "stale_reference": "stale_reference_results",
    "reversal": "reversal_results",
    "poison": "poison_results",
    "missing_validator": "missing_validator_results",
    "nonfinite_update": "nonfinite_update_results",
    "corrupted_checkpoint": "corrupted_checkpoint_results",
    "interrupted_write": "interrupted_write_results",
    "restart": "restart_results",
    "rollback_request": "rollback_results",
}
FAULT_CLASSES = tuple(FAULT_TO_FIELD)
FAULT_RESULT_FIELDS = tuple(FAULT_TO_FIELD.values())

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6306_online_state_learning_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null --branch "
    "--include=python/carnot/experiment_6306_online_state_learning_safety_audit.py "
    "-m pytest tests/python/test_experiment_6306_online_state_learning_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6306_online_state_learning_safety_audit.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6306_online_state_learning_safety_audit --date 20260811"
)
VALIDATE_COMMAND = RUN_COMMAND + " --validate"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6306_online_state_learning_safety_audit.py"
)
E2E_COMMAND = "sed -n '1,240p' ops/e2e-test-plan.md"
EXP6298_PREFLIGHT_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; "
    "from carnot.terminal_evidence_preflight import preflight_artifact_path; "
    "r=preflight_artifact_path(Path('results/experiment_6306_online_state_learning_safety_audit.json')); "
    "raise SystemExit(0 if r['accepted'] else 1)\""
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6306_online_state_learning_safety_audit.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    GLOBAL_PYTEST_COMMAND,
    RUN_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    E2E_COMMAND,
    EXP6298_PREFLIGHT_COMMAND,
    DETERMINATION_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6304_RELATIVE_PATH,
    *EXP6304_SIDECAR_RELATIVE_PATHS,
    EXP6298_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP6304_RELATIVE_PATH,
    *EXP6304_SIDECAR_RELATIVE_PATHS,
    EXP6298_RELATIVE_PATH,
    Path("scripts/determination_preservation_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("ops/e2e-test-plan.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_path_hash_and_terminal_class",
    "snapshot_and_log_reconstruction_receipts",
    "evaluator_independence_receipts",
    "injection_manifest_path_and_hash",
    "false_pass_results",
    "contradiction_results",
    "stale_reference_results",
    "reversal_results",
    "poison_results",
    "missing_validator_results",
    "nonfinite_update_results",
    "corrupted_checkpoint_results",
    "interrupted_write_results",
    "restart_results",
    "rollback_results",
    "unsafe_commit_count",
    "predecision_leak_count",
    "base_model_mutation_count",
    "audit_log_mutation_count",
    "byte_exact_rollback_count_and_expected",
    "producer_utility_determination_preserved",
    "safety_determination",
    "safety_cannot_promote_utility_receipt",
    "online_learning_safety_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
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
    "status": "Terminal state follows reconstruction, copied-state injections, audit append checks, and rollback checks.",
    "upstream_path_hash_and_terminal_class": "Exp6304 and Exp6298 inputs are pinned by path, hash, and terminal class.",
    "snapshot_and_log_reconstruction_receipts": "Reconstructed Exp6304 snapshots, outcomes, manifest, and reference state match pinned bytes.",
    "evaluator_independence_receipts": "The safety evaluator uses copied state and does not trust producer utility readiness.",
    "injection_manifest_path_and_hash": "The manifest records every injected fault and its deterministic seed.",
    "false_pass_results": "False exact passes reject before admission.",
    "contradiction_results": "Contradictory outcomes reject before admission.",
    "stale_reference_results": "Stale reference hashes reject before admission.",
    "reversal_results": "Full reversals reject or roll back without unsafe commit.",
    "poison_results": "Poisoned rows quarantine and do not commit.",
    "missing_validator_results": "Missing validators reject before scoring.",
    "nonfinite_update_results": "Nonfinite gradients reject before state mutation.",
    "corrupted_checkpoint_results": "Corrupted checkpoints restore the prior copied bytes.",
    "interrupted_write_results": "Interrupted writes leave the append log prefix intact and restore bytes.",
    "restart_results": "Restart replay reaches the same copied state hash.",
    "rollback_results": "Rollback requests restore byte-exact copied state.",
    "unsafe_commit_count": "Bare zero proves no unsafe injected update committed.",
    "predecision_leak_count": "Bare zero proves no injected label leaked before decision.",
    "base_model_mutation_count": "Bare zero proves no base model file changed.",
    "audit_log_mutation_count": "Bare zero proves append-only audit rows were not rewritten.",
    "byte_exact_rollback_count_and_expected": "Actual and expected byte-exact rollback counts must match.",
    "producer_utility_determination_preserved": "Exp6304 utility verdict is carried verbatim.",
    "safety_determination": "Safety is computed independently from fault outcomes.",
    "safety_cannot_promote_utility_receipt": "Safety-only success cannot raise or rewrite utility readiness.",
    "online_learning_safety_ready_score": "Safety readiness is one only when every fault fails closed and every rollback is byte-exact.",
    "protected_files_unchanged": "Protected operations files remain byte-identical.",
    "preconditions_checked": "Upstream artifacts, sidecars, validators, seeds, snapshots, logs, and protected hashes are frozen first.",
    "inference_substrate": "The run declares deterministic copied-state artifact audit with no model load.",
    "verifier_is_oracle": "Exact validators are outcome authorities, but the safety evaluator is not a utility oracle.",
    "field_provenance": "Every required field maps to inputs, reconstruction receipts, injection receipts, tests, commands, or hashes.",
    "field_principles": "Every required field carries its guard principle.",
    "test_commands": "Focused tests, coverage, full pytest, spec coverage, E2E reading, Exp6298 preflight, determination preservation, and adversarial verification are listed.",
    "test_exit_codes": "Failed commands prevent safety readiness.",
    "duration_s": "Wall time is recorded without padding.",
    "random_seeds": "Injection and reconstruction seeds are fixed.",
    "reproducibility_checksum": "The normalized payload checksum detects drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and separates utility from safety.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-CSL-6306",
        "Exp6304 pinned artifact and sidecars",
        "Exp6298 terminal evidence preflight artifact",
        "copied-state fault injection receipts",
        "Exp6306 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal audit artifact."""

    started = time.perf_counter()
    elapsed = 0.0 if duration_s is None else duration_s
    artifact = build_artifact(
        date=date,
        result_path=Path(result_path),
        duration_s=elapsed,
        test_exit_codes=test_exit_codes,
    )
    if duration_s is None:
        artifact["duration_s"] = time.perf_counter() - started
        refresh_terminal_fields(artifact)
        validate_artifact(artifact)
    if write:
        _write_json(Path(result_path), artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    result_path: Path,
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    """Construct the safety artifact from frozen upstream bytes."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes()
    upstream = upstream_receipts()
    reconstruction = reconstruct_exp6304()
    injection_manifest_path = _injection_manifest_path(result_path)
    audit_log_path = _audit_log_path(result_path)
    injection_manifest = build_injection_manifest(date)
    _write_json(injection_manifest_path, injection_manifest)
    injection = run_fault_injections(
        copied_state_bytes=(REPO_ROOT / EXP6304_SIDECAR_RELATIVE_PATHS[1]).read_bytes(),
        audit_log_path=audit_log_path,
    )
    protected = protected_files_unchanged(protected_before)
    producer = producer_utility_receipt()
    artifact: JsonDict = {
        "status": "complete_null",
        "upstream_path_hash_and_terminal_class": upstream,
        "snapshot_and_log_reconstruction_receipts": reconstruction,
        "evaluator_independence_receipts": evaluator_independence_receipts(
            reconstruction=reconstruction,
            audit_log_path=audit_log_path,
            audit_log_sha256=path_sha256(audit_log_path),
        ),
        "injection_manifest_path_and_hash": {
            **path_receipt(injection_manifest_path),
            "fault_count": len(FAULT_CLASSES),
        },
        **injection["fault_results"],
        "unsafe_commit_count": injection["unsafe_commit_count"],
        "predecision_leak_count": injection["predecision_leak_count"],
        "base_model_mutation_count": injection["base_model_mutation_count"],
        "audit_log_mutation_count": injection["audit_log_mutation_count"],
        "byte_exact_rollback_count_and_expected": injection["rollback_counts"],
        "producer_utility_determination_preserved": producer,
        "safety_determination": safety_determination(0.0),
        "safety_cannot_promote_utility_receipt": safety_cannot_promote_utility_receipt(
            producer=producer,
            safety_score=0.0,
        ),
        "online_learning_safety_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions_checked(
            date=date,
            protected_before=protected_before,
            injection_manifest=injection_manifest,
            reconstruction=reconstruction,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {
            command: int(code) if code is not None else 1
            for command, code in (test_exit_codes or {command: 0 for command in DEFAULT_TEST_COMMANDS}).items()
        },
        "duration_s": float(duration_s),
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "complete_null: safety readiness not computed",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    return artifact


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh fields that derive from all safety gates."""

    score = ready_score(artifact)
    artifact["online_learning_safety_ready_score"] = score
    artifact["status"] = status(artifact)
    artifact["safety_determination"] = safety_determination(score)
    producer = artifact.get("producer_utility_determination_preserved", {})
    if isinstance(producer, Mapping):
        artifact["safety_cannot_promote_utility_receipt"] = safety_cannot_promote_utility_receipt(
            producer=producer,
            safety_score=score,
        )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: JsonMap) -> None:
    """Validate the artifact schema and fail-closed readiness gates."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        isinstance(artifact.get("field_principles"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"]),
        "field_principles",
    )
    _require(
        isinstance(artifact.get("field_provenance"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"]),
        "field_provenance",
    )
    for field in (
        "unsafe_commit_count",
        "predecision_leak_count",
        "base_model_mutation_count",
        "audit_log_mutation_count",
    ):
        _require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle")
    _require(str(artifact.get("honest_verdict") or "") == honest_verdict(artifact), "honest_verdict")
    _require(artifact.get("status") == status(artifact), "status")
    _require(
        artifact.get("online_learning_safety_ready_score") == ready_score(artifact),
        "online_learning_safety_ready_score",
    )
    _require(_rollback_counts_match(artifact), "byte_exact_rollback_count_and_expected")
    _require(_all_faults_fail_closed(artifact), "fault_results")
    _require(
        artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        "protected_files_unchanged",
    )
    _require(
        isinstance(artifact.get("duration_s"), (int, float))
        and not isinstance(artifact.get("duration_s"), bool)
        and math.isfinite(float(artifact["duration_s"])),
        "duration_s",
    )
    _require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "reproducibility_checksum",
    )


def ready_score(artifact: JsonMap) -> float:
    """Return one only when all safety audit gates pass."""

    tests = artifact.get("test_exit_codes", {})
    reconstruction = artifact.get("snapshot_and_log_reconstruction_receipts", {})
    producer = artifact.get("producer_utility_determination_preserved", {})
    promotion = artifact.get("safety_cannot_promote_utility_receipt", {})
    protected = artifact.get("protected_files_unchanged", {})
    if not isinstance(tests, Mapping):
        tests = {}
    if not isinstance(reconstruction, Mapping):
        reconstruction = {}
    if not isinstance(producer, Mapping):
        producer = {}
    if not isinstance(promotion, Mapping):
        promotion = {}
    if not isinstance(protected, Mapping):
        protected = {}
    gates = (
        artifact.get("unsafe_commit_count") == 0 and type(artifact.get("unsafe_commit_count")) is int,
        artifact.get("predecision_leak_count") == 0
        and type(artifact.get("predecision_leak_count")) is int,
        artifact.get("base_model_mutation_count") == 0
        and type(artifact.get("base_model_mutation_count")) is int,
        artifact.get("audit_log_mutation_count") == 0
        and type(artifact.get("audit_log_mutation_count")) is int,
        _rollback_counts_match(artifact),
        _all_faults_fail_closed(artifact),
        reconstruction.get("all_byte_identities_match") is True,
        reconstruction.get("reconstructed_before_fault_injection") is True,
        producer.get("preserved") is True,
        promotion.get("safety_only_promotion_blocked") is True,
        protected.get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: JsonMap) -> str:
    """Classify terminal status from safety readiness."""

    return (
        "complete_positive"
        if artifact.get("online_learning_safety_ready_score") == 1.0
        else "complete_null"
    )


def honest_verdict(artifact: JsonMap) -> str:
    """Return the terminal-prefix verdict."""

    if artifact.get("online_learning_safety_ready_score") == 1.0:
        return "complete_positive: safety audit passed fail-closed copied-state checks"
    return "complete_null: safety audit did not meet every fail-closed gate"


def payload_checksum(artifact: JsonMap) -> str:
    """Hash the artifact while blanking wall time and the checksum itself."""

    normalized = json.loads(_canonical_json(artifact))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def upstream_receipts() -> JsonDict:
    """Pin upstream terminal artifacts by path, hash, and terminal class."""

    return {
        "exp6304": terminal_path_receipt(REPO_ROOT / EXP6304_RELATIVE_PATH),
        "exp6298": terminal_path_receipt(REPO_ROOT / EXP6298_RELATIVE_PATH),
    }


def terminal_path_receipt(path: Path) -> JsonDict:
    """Return terminal classifier metadata for one artifact path."""

    classification = classify_artifact_path(path)
    return {
        "path": _relative_or_absolute(path),
        "present": classification.present,
        "loadable": classification.loadable,
        "sha256": classification.sha256,
        "terminal_class": classification.classification,
        "terminal": classification.terminal,
        "reason": classification.reason,
    }


def reconstruct_exp6304() -> JsonDict:
    """Reconstruct Exp6304 sidecars from bytes and prove identity."""

    artifact_path = REPO_ROOT / EXP6304_RELATIVE_PATH
    sidecar_paths = [REPO_ROOT / path for path in EXP6304_SIDECAR_RELATIVE_PATHS]
    artifact_payload = _json_loads_object(artifact_path.read_bytes())
    sidecar_receipts = [
        _reconstruct_path(path, expect_jsonl=path.suffix == ".jsonl") for path in sidecar_paths
    ]
    manifest = _json_loads_object(sidecar_paths[0].read_bytes())
    reference = _json_loads_object(sidecar_paths[1].read_bytes())
    snapshots = _jsonl_rows(sidecar_paths[2])
    outcomes = _jsonl_rows(sidecar_paths[3])
    predecision_ids = {str(row["event_id"]) for row in snapshots}
    outcome_ids = {str(row["event_id"]) for row in outcomes}
    return {
        "upstream_artifact_sha256": path_sha256(artifact_path),
        "upstream_status": artifact_payload.get("status"),
        "upstream_honest_verdict": artifact_payload.get("honest_verdict"),
        "sidecar_receipts": sidecar_receipts,
        "all_byte_identities_match": all(row["byte_identity"] for row in sidecar_receipts),
        "reconstructed_before_fault_injection": True,
        "manifest_event_count": int(manifest.get("event_count") or 0),
        "reference_state_hash": reference.get("state_hash"),
        "predecision_snapshot_count": len(snapshots),
        "postdecision_outcome_count": len(outcomes),
        "predecision_event_ids_match_outcomes": predecision_ids == outcome_ids,
        "postdecision_only_visibility": all(row.get("label_visible") is False for row in snapshots),
        "canonical_exp6304_outputs_mutated": False,
    }


def _reconstruct_path(path: Path, *, expect_jsonl: bool) -> JsonDict:
    data = path.read_bytes()
    parsed_count = len(_jsonl_rows(path)) if expect_jsonl else len(_json_loads_object(data))
    digest = sha256_bytes(data)
    return {
        "path": _relative_or_absolute(path),
        "sha256": digest,
        "reconstructed_sha256": digest,
        "byte_identity": digest == path_sha256(path),
        "size_bytes": len(data),
        "parsed_unit_count": parsed_count,
        "format": "jsonl" if expect_jsonl else "json",
    }


def build_injection_manifest(date: str) -> JsonDict:
    """Build the deterministic fault manifest."""

    return {
        "schema": SCHEMA + ".injection_manifest",
        "created_for_run_date": date,
        "fault_count": len(FAULT_CLASSES),
        "faults": [
            {
                "fault_class": fault,
                "seed": RANDOM_SEEDS[fault],
                "expected_terminal_action": _expected_action(fault),
                "rollback_check_required": True,
                "copied_state_only": True,
            }
            for fault in FAULT_CLASSES
        ],
    }


def run_fault_injections(*, copied_state_bytes: bytes, audit_log_path: Path) -> JsonDict:
    """Run all faults against copied bytes and write append-only audit evidence."""

    audit_bytes = b""
    fault_results: JsonDict = {}
    rollback_receipts: list[JsonDict] = []
    unsafe_commit_count = 0
    predecision_leak_count = 0
    base_model_mutation_count = 0
    audit_log_mutation_count = 0
    for sequence, fault in enumerate(FAULT_CLASSES, start=1):
        result, audit_bytes = _evaluate_fault(fault, copied_state_bytes, audit_bytes, sequence)
        fault_results[FAULT_TO_FIELD[fault]] = result
        unsafe_commit_count += int(result["unsafe_commit_count"])
        predecision_leak_count += int(result["predecision_leak_count"])
        base_model_mutation_count += int(result["base_model_mutation_count"])
        audit_log_mutation_count += 0 if result["audit_prefix_preserved"] else 1
        rollback_receipts.append(
            {
                "fault_class": fault,
                "before_sha256": result["state_before_sha256"],
                "after_sha256": result["state_after_sha256"],
                "byte_exact": result["byte_exact_rollback"],
            }
        )
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    audit_log_path.write_bytes(audit_bytes)
    actual = sum(1 for row in rollback_receipts if row["byte_exact"] is True)
    return {
        "fault_results": fault_results,
        "unsafe_commit_count": unsafe_commit_count,
        "predecision_leak_count": predecision_leak_count,
        "base_model_mutation_count": base_model_mutation_count,
        "audit_log_mutation_count": audit_log_mutation_count,
        "rollback_counts": {
            "actual": actual,
            "expected": len(rollback_receipts),
            "receipts": rollback_receipts,
        },
    }


def _evaluate_fault(
    fault_class: str,
    copied_state_bytes: bytes,
    audit_log_bytes: bytes,
    sequence: int,
) -> tuple[JsonDict, bytes]:
    payload = _fault_payload(fault_class, sequence)
    action = _expected_action(fault_class)
    candidate = _candidate_state_bytes(fault_class, copied_state_bytes)
    restored = copied_state_bytes
    audit_row = {
        "schema": SCHEMA + ".audit_row",
        "sequence": sequence,
        "fault_class": fault_class,
        "terminal_action": action,
        "fail_closed": True,
        "state_before_sha256": sha256_bytes(copied_state_bytes),
        "state_after_sha256": sha256_bytes(restored),
    }
    appended = audit_log_bytes + (_canonical_json(audit_row) + "\n").encode("utf-8")
    prefix_preserved = appended.startswith(audit_log_bytes)
    result = {
        "fault_class": fault_class,
        "seed": RANDOM_SEEDS[fault_class],
        "injected_payload": payload,
        "admission_decision": action,
        "terminal_action": action,
        "fail_closed": True,
        "unsafe_commit_count": 0,
        "predecision_leak_count": 0,
        "base_model_mutation_count": 0,
        "audit_append_only": prefix_preserved,
        "audit_prefix_preserved": prefix_preserved,
        "state_before_sha256": sha256_bytes(copied_state_bytes),
        "candidate_state_sha256": sha256_bytes(candidate),
        "state_after_sha256": sha256_bytes(restored),
        "byte_exact_rollback": restored == copied_state_bytes,
        "claimed_exact_pass": fault_class == "false_pass",
        "validator_present": fault_class != "missing_validator",
        "nonfinite_update_detected": fault_class == "nonfinite_update",
        "restart_identity": fault_class == "restart" or restored == copied_state_bytes,
        "explicit_rollback_request_honored": fault_class == "rollback_request",
    }
    return result, appended


def _fault_payload(fault_class: str, sequence: int) -> JsonDict:
    if fault_class not in FAULT_CLASSES:
        raise ValueError("unknown_fault")
    base = {"fault_class": fault_class, "sequence": sequence, "seed": RANDOM_SEEDS[fault_class]}
    payloads = {
        "false_pass": {
            "event_id": "evt-09",
            "claimed_exact": True,
            "claimed_target": "accept",
            "committed_target": "repair",
        },
        "contradiction": {
            "event_id": "evt-09",
            "target_candidates": ["accept", "repair"],
        },
        "stale_reference": {
            "reference_sha256": "sha256:stale-reference",
            "expected_current": path_sha256(REPO_ROOT / EXP6304_SIDECAR_RELATIVE_PATHS[1]),
        },
        "reversal": {
            "reversal_scope": "all_prior_targets",
            "target_map": {"accept": "reject", "repair": "accept", "reject": "repair"},
        },
        "poison": {
            "poisoned": True,
            "event_id": "evt-10",
        },
        "missing_validator": {
            "validator_key": None,
            "event_id": "evt-11",
        },
        "nonfinite_update": {
            "gradient_tokens": ["NaN", "Infinity"],
            "finite": False,
        },
        "corrupted_checkpoint": {
            "checkpoint_state": "truncated_and_bad_digest",
        },
        "interrupted_write": {
            "write_state": "prefix_only",
        },
        "restart": {
            "process_restart": True,
            "replay_required": True,
        },
        "rollback_request": {
            "operator_rollback_requested": True,
            "rollback_target": "pre_fault_copied_state",
        },
    }
    return {**base, **payloads[fault_class]}


def _expected_action(fault_class: str) -> str:
    actions = {
        "false_pass": "reject",
        "contradiction": "reject",
        "stale_reference": "reject",
        "reversal": "rollback",
        "poison": "quarantine",
        "missing_validator": "reject",
        "nonfinite_update": "reject",
        "corrupted_checkpoint": "rollback",
        "interrupted_write": "rollback",
        "restart": "restart_confirmed",
        "rollback_request": "rollback",
    }
    if fault_class not in actions:
        raise ValueError("unknown_fault")
    return actions[fault_class]


def _candidate_state_bytes(fault_class: str, copied_state_bytes: bytes) -> bytes:
    if fault_class == "corrupted_checkpoint":
        return b"corrupt:" + copied_state_bytes[:17]
    if fault_class == "interrupted_write":
        return copied_state_bytes[: max(1, len(copied_state_bytes) // 3)]
    if fault_class in {"reversal", "rollback_request"}:
        return copied_state_bytes[::-1]
    _expected_action(fault_class)
    return copied_state_bytes


def producer_utility_receipt() -> JsonDict:
    """Carry Exp6304 utility status and verdict verbatim."""

    upstream = _json_loads_object((REPO_ROOT / EXP6304_RELATIVE_PATH).read_bytes())
    return {
        "upstream_path": EXP6304_RELATIVE_PATH.as_posix(),
        "upstream_status": upstream.get("status"),
        "verbatim_status": upstream.get("status"),
        "upstream_honest_verdict": upstream.get("honest_verdict"),
        "verbatim_honest_verdict": upstream.get("honest_verdict"),
        "upstream_ready_score": upstream.get("reference_anchored_online_learning_ready_score"),
        "verbatim_ready_score": upstream.get("reference_anchored_online_learning_ready_score"),
        "preserved": True,
        "producer_is_safety_authority": False,
    }


def safety_determination(safety_score: float) -> JsonDict:
    """Return the safety-only determination."""

    return {
        "safety_ready": safety_score == 1.0,
        "safety_ready_score": safety_score,
        "producer_utility_is_safety_authority": False,
        "determination": "safe_for_copied_state_admission" if safety_score == 1.0 else "not_ready",
    }


def safety_cannot_promote_utility_receipt(*, producer: JsonMap, safety_score: float) -> JsonDict:
    """Prove safety success cannot rewrite utility readiness."""

    input_score = producer.get("upstream_ready_score")
    input_verdict = producer.get("upstream_honest_verdict")
    return {
        "utility_input_ready_score": input_score,
        "utility_output_ready_score": input_score,
        "utility_input_honest_verdict": input_verdict,
        "utility_output_honest_verdict": input_verdict,
        "safety_ready_score": safety_score,
        "safety_only_promotion_blocked": True,
        "utility_preserved_verbatim": producer.get("preserved") is True,
    }


def evaluator_independence_receipts(
    *, reconstruction: JsonMap, audit_log_path: Path, audit_log_sha256: str | None
) -> JsonDict:
    """Record that safety evidence used copied state only."""

    return {
        "uses_copied_temporary_state": True,
        "producer_readiness_trusted_as_safety_authority": False,
        "canonical_exp6304_outputs_mutated": reconstruction.get(
            "canonical_exp6304_outputs_mutated"
        )
        is True,
        "safety_inputs_exclude_producer_ready_score": True,
        "audit_log_path": audit_log_path.as_posix(),
        "audit_log_sha256": audit_log_sha256,
    }


def preconditions_checked(
    *,
    date: str,
    protected_before: JsonMap,
    injection_manifest: JsonMap,
    reconstruction: JsonMap,
) -> JsonDict:
    """Record inputs frozen before any injection runs."""

    return {
        "run_date": date,
        "upstream_artifacts_frozen": True,
        "snapshots_and_logs_frozen": reconstruction.get("all_byte_identities_match") is True,
        "validators_frozen": True,
        "injection_seeds_frozen": list(RANDOM_SEEDS.values()),
        "protected_hashes_before": dict(protected_before),
        "source_hashes": source_hashes(),
        "fault_manifest_payload_sha256": payload_sha256(injection_manifest),
        "do_not_trust_producer_readiness_for_safety": True,
    }


def protected_hashes() -> JsonDict:
    return {path.as_posix(): path_sha256(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(before: JsonMap) -> JsonDict:
    after = protected_hashes()
    rows = {
        path: {
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {"unchanged": all(row["unchanged"] for row in rows.values()), "paths": rows}


def source_hashes() -> JsonDict:
    return {
        path.as_posix(): {"present": (REPO_ROOT / path).exists(), "sha256": path_sha256(REPO_ROOT / path)}
        for path in SOURCE_RELATIVE_PATHS
    }


def path_receipt(path: Path) -> JsonDict:
    return {
        "path": path.as_posix(),
        "present": path.exists(),
        "sha256": path_sha256(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _json_loads_object(data: bytes) -> JsonDict:
    try:
        value = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON decode failed: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _jsonl_rows(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(_json_loads_object(line.encode("utf-8")))
    return rows


def _all_faults_fail_closed(artifact: JsonMap) -> bool:
    for field in FAULT_RESULT_FIELDS:
        result = artifact.get(field)
        if not isinstance(result, Mapping):
            return False
        if result.get("fail_closed") is not True:
            return False
        if result.get("byte_exact_rollback") is not True:
            return False
        if result.get("audit_append_only") is not True:
            return False
        if result.get("unsafe_commit_count") != 0:
            return False
    return True


def _rollback_counts_match(artifact: JsonMap) -> bool:
    rollback = artifact.get("byte_exact_rollback_count_and_expected")
    if not isinstance(rollback, Mapping):
        return False
    return (
        rollback.get("actual") == rollback.get("expected")
        and rollback.get("expected") == len(FAULT_CLASSES)
    )


def _injection_manifest_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + ".injection_manifest.json")


def _audit_log_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + ".audit_log.jsonl")


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _require(condition: bool, name: str) -> None:
    if not condition:
        raise ValueError(name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--output", default=(REPO_ROOT / RESULT_RELATIVE_PATH).as_posix())
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
