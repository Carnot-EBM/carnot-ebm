"""Exp5865 requalification artifact builder for the adaptive-state kernel.

Spec refs: REQ-LEARN-5865, SCENARIO-LEARN-5865-ATTRIBUTION,
SCENARIO-LEARN-5865-PARITY-PRESERVED.

This module does not change the Exp5859 state machine. It records whether the
already-qualified adaptive-state semantics still pass focused parity while the
repository-wide pytest blocker is attributed separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot import adaptive_state as base


JsonDict = dict[str, Any]

RESULT_RELATIVE_PATH = Path("results/experiment_5865_adaptive_state_kernel_requalification.json")
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
INFERENCE_SUBSTRATE = base.INFERENCE_SUBSTRATE

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "exp5859_input_receipt",
    "original_nonzero_exit_reproduction",
    "root_cause_classification",
    "failing_collection_and_node_receipts",
    "semantic_change_scope",
    "qualified_operation_mapping",
    "abi_schema_and_bounds",
    "python_rust_binding_receipts",
    "cross_language_operation_parity",
    "canonical_state_and_hash_parity",
    "serialization_restart_and_rollback_parity",
    "invalid_input_and_capacity_controls",
    "before_after_test_matrix",
    "global_suite_debt_classification",
    "applicable_e2e_receipts",
    "protected_files_unchanged",
    "adaptive_state_microkernel_requalified_score",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes requalified parity from a documented remaining block.",
    "preconditions_checked": (
        "Hashes, toolchains, plugins, resources, outputs, and protected files prevent blind repair."
    ),
    "exp5859_input_receipt": (
        "The new result must consume the exact blocked artifact without rewriting history."
    ),
    "original_nonzero_exit_reproduction": (
        "A failing seam must be observed before it can be attributed or repaired."
    ),
    "root_cause_classification": (
        "Owned, unrelated, environment, and protected-work failures require different actions."
    ),
    "failing_collection_and_node_receipts": (
        "Exact logs and node IDs replace a generic exit-code narrative."
    ),
    "semantic_change_scope": (
        "Only a proven owned defect may alter already-passing kernel semantics."
    ),
    "qualified_operation_mapping": "Only Exp5858-qualified operations remain in the ABI.",
    "abi_schema_and_bounds": (
        "Version, types, capacities, and ordering remain finite and hardware-mappable."
    ),
    "python_rust_binding_receipts": (
        "The deployable call surface must build and import on the tested toolchain."
    ),
    "cross_language_operation_parity": (
        "Every accepted and rejected operation must match across implementations."
    ),
    "canonical_state_and_hash_parity": (
        "Equivalent states must produce identical canonical hashes."
    ),
    "serialization_restart_and_rollback_parity": (
        "Durable state must restore and roll back identically."
    ),
    "invalid_input_and_capacity_controls": (
        "Malformed, duplicate, out-of-order, overflow, and unbounded input fail closed."
    ),
    "before_after_test_matrix": "No pre-existing passing receipt may regress during seam repair.",
    "global_suite_debt_classification": (
        "Unrelated failures stay visible and cannot be silently waived."
    ),
    "applicable_e2e_receipts": (
        "Promotion requires the exact end-to-end checks relevant to the kernel path."
    ),
    "protected_files_unchanged": "User work and operator-curated files remain untouched.",
    "adaptive_state_microkernel_requalified_score": (
        "EMIT BARE scalar; only 1.0 permits Exp5866 and Exp5875."
    ),
    "duration_s": "Measured wall time exposes bootstrap-only requalification.",
    "inference_substrate": (
        "`deterministic_cross_language_state_execution_no_llm` declares the actual path."
    ),
    "field_provenance": (
        "Every decision traces to logs, fixtures, source hashes, toolchains, and outputs."
    ),
    "test_commands": "Commands document reproduction, parity, binding, regression, and E2E checks.",
    "test_exit_codes": "Every claimed readiness command must have exit code zero.",
    "reproducibility_checksum": ("A checksum detects ABI, fixture, toolchain, or test drift."),
    "honest_verdict": (
        "A `complete:`, `parity:`, `retired:`, or `blocked:` prefix states the terminal outcome."
    ),
}


def read_json(path: str | Path) -> JsonDict:
    """Read JSON through the same canonical helper family as Exp5859."""

    return base.read_json(path)


def exp5859_input_receipt(path: str | Path | None = None) -> JsonDict:
    """Summarize the immutable blocked Exp5859 artifact consumed by Exp5865."""

    artifact_path = Path(path) if path is not None else base.REPO_ROOT / base.RESULT_RELATIVE_PATH
    artifact = base.read_json(artifact_path)
    return {
        "adaptive_state_microkernel_ready_score": artifact.get(
            "adaptive_state_microkernel_ready_score"
        ),
        "checksum_valid": base.validate_artifact(artifact),
        "honest_verdict": artifact.get("honest_verdict"),
        "path": artifact_path.relative_to(base.REPO_ROOT).as_posix(),
        "recorded_full_suite_exit_code": artifact.get("test_exit_codes", {}).get(
            FULL_PYTEST_COMMAND
        ),
        "sha256": base.sha256_file(artifact_path),
        "status": artifact.get("status"),
    }


def _protected_files_unchanged(exp5859_artifact: JsonDict | None = None) -> JsonDict:
    if exp5859_artifact is None:
        exp5859_artifact = base.read_json(base.REPO_ROOT / base.RESULT_RELATIVE_PATH)
    source_hashes = exp5859_artifact.get("field_provenance", {}).get("source_hashes", {})
    relative_path = base.PROTECTED_FILE_RELATIVE_PATH.as_posix()
    baseline_hash = source_hashes.get(relative_path)
    current_hash = base.sha256_file(base.REPO_ROOT / base.PROTECTED_FILE_RELATIVE_PATH)
    return {
        "baseline_sha256": baseline_hash,
        "current_sha256": current_hash,
        "path": relative_path,
        "unchanged": bool(baseline_hash and baseline_hash == current_hash),
    }


def _source_hashes(result_path: Path) -> JsonDict:
    paths = (
        base.SELF_LEARNING_SPEC_RELATIVE_PATH,
        base.PY_MODULE_RELATIVE_PATH,
        Path("python/carnot/adaptive_state_requalification.py"),
        base.PY_TEST_RELATIVE_PATH,
        Path("tests/python/test_experiment_5865_adaptive_state_kernel_requalification.py"),
        base.RUST_CORE_RELATIVE_PATH,
        base.RUST_BINDING_RELATIVE_PATH,
        base.RESULT_RELATIVE_PATH,
        RESULT_RELATIVE_PATH,
        base.PROTECTED_FILE_RELATIVE_PATH,
    )
    hashes: JsonDict = {}
    for relative_path in paths:
        absolute = base.REPO_ROOT / relative_path
        if absolute.exists():
            hashes[relative_path.as_posix()] = base.sha256_file(absolute)
    hashes["output_path"] = str(result_path)
    return hashes


def _root_cause(
    failing_receipts: list[JsonDict],
    global_suite_debt_classification: JsonDict,
) -> JsonDict:
    classifications = [str(item.get("classification", "unclassified")) for item in failing_receipts]
    owned = "adaptive-state-owned" in classifications
    protected = "protected_user_work_conflict" in classifications or bool(
        global_suite_debt_classification.get("protected_user_work_conflict")
    )
    primary = (
        "adaptive-state-owned"
        if owned
        else str(global_suite_debt_classification.get("classification", "none"))
    )
    if primary == "none" and protected:
        primary = "protected_user_work_conflict"
    return {
        "adaptive_state_owned_failure_count": classifications.count("adaptive-state-owned"),
        "classification_counts": {
            name: classifications.count(name) for name in sorted(set(classifications))
        },
        "owned_repair_performed": False,
        "primary": primary,
        "protected_user_work_conflict": protected,
    }


def _parity_ready(parity: JsonDict) -> bool:
    serialization = parity["serialization_restart_and_rollback_parity"]
    return (
        parity["binding_receipt"]["binding_available"]
        and parity["cross_language_operation_parity"]["accept_reject_parity"]
        and parity["canonical_state_and_hash_parity"]["hash_parity"]
        and serialization["round_trip_parity"]
        and serialization["rollback_parity"]
        and serialization["restart_parity"]
        and parity["invalid_input_and_capacity_controls"]["fail_closed"]
    )


def _all_applicable_e2e_zero(applicable_e2e_receipts: JsonDict) -> bool:
    for receipt in applicable_e2e_receipts.values():
        if isinstance(receipt, dict) and receipt.get("exit_code") not in (0, None):
            return False
    return True


def build_artifact(
    *,
    result_path: str | Path = base.REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float,
    test_commands: list[str],
    test_exit_codes: dict[str, int | None],
    original_nonzero_exit_reproduction: JsonDict,
    failing_collection_and_node_receipts: list[JsonDict],
    global_suite_debt_classification: JsonDict,
    applicable_e2e_receipts: JsonDict,
    before_after_test_matrix: JsonDict,
    parity_receipt: JsonDict | None = None,
    preconditions_checked: JsonDict | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the Exp5865 terminal artifact without mutating Exp5859."""

    result_path = Path(result_path)
    exp5859_artifact = base.read_json(base.REPO_ROOT / base.RESULT_RELATIVE_PATH)
    parity = parity_receipt if parity_receipt is not None else base._run_parity_receipts()
    preconditions = (
        preconditions_checked
        if preconditions_checked is not None
        else base.collect_preconditions(result_path)
    )
    protected = _protected_files_unchanged(exp5859_artifact)
    root_cause = _root_cause(
        list(failing_collection_and_node_receipts),
        dict(global_suite_debt_classification),
    )
    py_receipt, rust_receipt = base.implementation_receipts()
    exits_ok = bool(test_exit_codes) and all(code == 0 for code in test_exit_codes.values())
    global_blocking = bool(global_suite_debt_classification.get("blocking"))
    ready = (
        exits_ok
        and preconditions["preconditions_ready"]
        and _parity_ready(parity)
        and not global_blocking
        and protected["unchanged"]
        and _all_applicable_e2e_zero(applicable_e2e_receipts)
    )
    artifact: JsonDict = {
        "abi_schema_and_bounds": base.abi_schema_and_bounds(),
        "adaptive_state_microkernel_requalified_score": 1.0 if ready else 0.0,
        "applicable_e2e_receipts": applicable_e2e_receipts,
        "before_after_test_matrix": before_after_test_matrix,
        "canonical_state_and_hash_parity": parity["canonical_state_and_hash_parity"],
        "cross_language_operation_parity": parity["cross_language_operation_parity"],
        "duration_s": round(duration_s, 6),
        "exp5859_input_receipt": exp5859_input_receipt(),
        "failing_collection_and_node_receipts": failing_collection_and_node_receipts,
        "field_provenance": {
            "exp5859_checksum": exp5859_artifact.get("reproducibility_checksum"),
            "field_principles": REQUIRED_FIELD_PRINCIPLES,
            "operation_trace_hash": base.sha256_json(
                [base.deterministic_fixture_trace()]
                + base.randomized_operation_traces(seed=5859, trace_count=4, events_per_trace=6)
            ),
            "result_path": RESULT_RELATIVE_PATH.as_posix(),
            "source_hashes": _source_hashes(result_path),
        },
        "global_suite_debt_classification": global_suite_debt_classification,
        "honest_verdict": (
            "complete: adaptive_state_microkernel_requalified"
            if ready
            else "retired: adaptive_state_requalification_blocked_by_unrelated_global_suite_debt"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "invalid_input_and_capacity_controls": parity["invalid_input_and_capacity_controls"],
        "original_nonzero_exit_reproduction": original_nonzero_exit_reproduction,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "python_rust_binding_receipts": {
            "binding": parity["binding_receipt"],
            "python_implementation": py_receipt,
            "rust_implementation": rust_receipt,
        },
        "qualified_operation_mapping": base.qualified_operation_mapping(),
        "root_cause_classification": root_cause,
        "semantic_change_scope": {
            "abi_changed": False,
            "capacity_or_bound_changed": False,
            "code_semantics_changed": False,
            "failure_codes_changed": False,
            "operation_set_changed": False,
            "ordering_changed": False,
            "schema_version_changed": False,
            "summary": "No adaptive-state kernel semantic changes were made.",
        },
        "serialization_restart_and_rollback_parity": parity[
            "serialization_restart_and_rollback_parity"
        ],
        "status": "requalified" if ready else "retired",
        "test_commands": test_commands,
        "test_exit_codes": test_exit_codes,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        base._write_json_atomic(result_path, artifact)
    return artifact


def reproducibility_checksum(artifact: JsonDict) -> str:
    """Hash every Exp5865 field except the checksum slot itself."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return base.sha256_json(payload)


def validate_artifact(artifact: JsonDict) -> bool:
    """Validate Exp5865 without trusting prose verdicts."""

    required = set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    checksum_ok = artifact.get("reproducibility_checksum") == reproducibility_checksum(artifact)
    score = artifact.get("adaptive_state_microkernel_requalified_score")
    exits_ok = all(code == 0 for code in artifact.get("test_exit_codes", {}).values())
    blocked_score_ok = score == 0.0 and str(artifact.get("honest_verdict", "")).startswith(
        ("retired:", "blocked:")
    )
    ready_score_ok = (
        score == 1.0
        and artifact.get("status") == "requalified"
        and exits_ok
        and not artifact.get("global_suite_debt_classification", {}).get("blocking")
        and artifact.get("protected_files_unchanged", {}).get("unchanged")
        and artifact.get("cross_language_operation_parity", {}).get("accept_reject_parity")
        and artifact.get("canonical_state_and_hash_parity", {}).get("hash_parity")
        and artifact.get("serialization_restart_and_rollback_parity", {}).get("round_trip_parity")
        and artifact.get("invalid_input_and_capacity_controls", {}).get("fail_closed")
    )
    return (
        required
        and checksum_ok
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and (blocked_score_ok or ready_score_ok)
    )
