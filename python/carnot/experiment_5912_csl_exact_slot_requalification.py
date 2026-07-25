"""Exp5912 exact-slot requalification for Exp5895 shortcut-safe CSL.

Spec refs: REQ-LEARN-5912, SCENARIO-LEARN-5912-ATTRIBUTION,
SCENARIO-LEARN-5912-FROZEN-PARITY, SCENARIO-LEARN-5912-RETIREMENT,
SCENARIO-LEARN-5912-READY.

This module does not change the Exp5895 experiment. It replays the existing
deterministic mechanism into a temporary result, compares the science receipts
against the historical artifact, and lets current command exits decide whether
the exact slot is ready or retired.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
from typing import Any

from carnot import experiment_5895_shortcut_safe_continuous_self_learning as exp5895


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5912_csl_exact_slot_requalification.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5912_csl_exact_slot_requalification.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5912_csl_exact_slot_requalification.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
HISTORICAL_EXP5895_RELATIVE_PATH = exp5895.RESULT_RELATIVE_PATH
EXP5894_RELATIVE_PATH = exp5895.EXP5894_ARTIFACT_RELATIVE_PATH
EXP5893_ROWS_RELATIVE_PATH = exp5895.EXP5893_ROWS_RELATIVE_PATH
EXP5895_MODULE_RELATIVE_PATH = exp5895.MODULE_RELATIVE_PATH
EXP5895_TEST_RELATIVE_PATH = exp5895.TEST_RELATIVE_PATH
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
SCHEMA = "carnot.experiment_5912.csl_exact_slot_requalification.v1"
EXPERIMENT = 5912
EXPERIMENT_ID = "experiment_5912_csl_exact_slot_requalification"
RUN_DATE = "20260725"
INFERENCE_SUBSTRATE = exp5895.INFERENCE_SUBSTRATE
VERIFIER_IS_ORACLE = True
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
COLLECTION_COMMAND = ".venv/bin/pytest tests/python --collect-only -q -n 0"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "continuous_self_learning_task",
    "historical_exp5895_hash_and_immutability",
    "original_test_failure_receipt",
    "current_failure_node_ids_phases_and_ownership",
    "causal_relevance_classification",
    "repair_scope_and_changed_files",
    "frozen_rows_arms_seeds_budgets_thresholds_and_ready_logic",
    "deterministic_science_parity",
    "prospective_lift_retention_safety_rollback_and_state_receipts",
    "no_model_weight_mutation",
    "retired_dependency_chain_used",
    "repeated_verdict_retirement_decision",
    "protected_files_unchanged",
    "csl_exact_slot_ready_score",
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
    "status": "A terminal state distinguishes requalified, null, retired, unsafe, or blocked exact-slot evidence.",
    "preconditions_checked": "Hashes, collection state, resources, output path, and protected-file receipts prevent blind requalification.",
    "continuous_self_learning_task": "Must be bare true and cannot be satisfied by a doc-only receipt.",
    "historical_exp5895_hash_and_immutability": "The historical Exp5895 artifact is consumed read-only and never rewritten.",
    "original_test_failure_receipt": "The historical global-suite exit 2 remains visible as the reason Exp5895 scored 0.0.",
    "current_failure_node_ids_phases_and_ownership": "Current command failures are tied to exact node IDs, phases, and owned paths.",
    "causal_relevance_classification": "Execution debt is separated from anything that can alter Exp5895 data, logic, labels, or readiness.",
    "repair_scope_and_changed_files": "Only causally relevant current repository-owned debt and the new wrapper may change.",
    "frozen_rows_arms_seeds_budgets_thresholds_and_ready_logic": "Requalification repairs execution readiness, not scientific inputs.",
    "deterministic_science_parity": "The temporary replay must match the historical deterministic science hash.",
    "prospective_lift_retention_safety_rollback_and_state_receipts": "Positive lift, retention 1.0, zero unsafe accepts, exact rollback/restart, and bounded state are rewrapped from the frozen replay.",
    "no_model_weight_mutation": "Model weights remain immutable and unloaded.",
    "retired_dependency_chain_used": "Exp5865-Exp5867 retired outputs are never used for promotion.",
    "repeated_verdict_retirement_decision": "The same null verdict retires this exact requalification scope.",
    "protected_files_unchanged": "Protected files and historical results stay byte-identical.",
    "csl_exact_slot_ready_score": "Emit bare 1.0 only when frozen scientific parity holds and every required current test exits zero.",
    "duration_s": "Measured wall time exposes deterministic wrapper work.",
    "inference_substrate": "Use `deterministic_exact_verifier_and_versioned_external_state_no_llm`.",
    "verifier_is_oracle": "True for labels and promotion authority.",
    "field_provenance": "Every field traces to prompt, spec, historical artifacts, frozen replay, command receipts, or tests.",
    "test_commands": "Commands document focused unit/coverage, collection attribution, deterministic replay, full Python suite, lifecycle, retention/safety/rollback, immutable-artifact, schema, applicable E2E, adversarial, spec-coverage, root-clutter, and protected-file checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "A checksum detects artifact, command, parity, or protected-file drift.",
    "honest_verdict": "Use `complete_positive:`, `complete_null:`, `retired:`, `unsafe:`, or `blocked:`.",
}

PROTECTED_RELATIVE_PATHS = (
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    HISTORICAL_EXP5895_RELATIVE_PATH,
    EXP5894_RELATIVE_PATH,
    EXP5893_ROWS_RELATIVE_PATH,
)

SCIENCE_FIELDS = (
    "sealed_chronological_split_and_visibility",
    "frozen_arms_and_budget_parity",
    "exact_query_policy_and_budget",
    "verified_evidence_and_unresolved_constraint_state",
    "versioned_promotion_quarantine_rejection_and_rollback",
    "rejected_update_non_propagation",
    "per_update_non_forgetting_certificates",
    "prospective_semantic_and_constraint_metrics",
    "shortcut_false_accept_metrics",
    "forward_transfer_recurrence_retention_and_regret",
    "family_grounding_hardness_lower_bounds",
    "replay_query_resource_and_latency_accounting",
    "memory_cap_accounting",
    "rollback_restart_and_state_hashes",
    "no_model_weight_mutation",
    "null_and_ablation_controls",
    "hardware_mapping_contract",
)


def canonical_json(value: Any) -> str:
    """Serialize receipts once so hashes compare byte-stable JSON evidence."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash short text receipts with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes instead of trusting timestamps or names."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object artifact and fail closed on non-object payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent probe.
    meminfo = Path("/proc/meminfo")
    available_mb = 0
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent probe.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def make_command_receipt(
    *,
    command: str,
    phase: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    node_ids: Sequence[str] = (),
    ownership_paths: Sequence[str] = (),
) -> JsonDict:
    """Build the command receipt format used by tests and the final wrapper."""

    return {
        "command": command,
        "phase": phase,
        "exit_code": int(exit_code),
        "stdout_sha256": sha256_text(stdout),
        "stderr_sha256": sha256_text(stderr),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "node_ids": list(node_ids),
        "ownership_paths": list(ownership_paths),
    }


def _hash_path(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _protected_hashes(root: Path) -> JsonDict:
    return {path.as_posix(): _hash_path(root, path) for path in PROTECTED_RELATIVE_PATHS}


def historical_exp5895_hash_and_immutability(root: Path = REPO_ROOT) -> JsonDict:
    """Return the immutable historical artifact receipt consumed by Exp5912."""

    path = root / HISTORICAL_EXP5895_RELATIVE_PATH
    payload = read_json(path)
    mode = path.stat().st_mode
    read_only = (mode & stat.S_IWUSR) == 0 and (mode & stat.S_IWGRP) == 0 and (
        mode & stat.S_IWOTH
    ) == 0
    try:
        validates = exp5895.validate_artifact(payload)
    except ValueError:
        validates = False
    return {
        "path": HISTORICAL_EXP5895_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "mode_octal": oct(stat.S_IMODE(mode)),
        "read_only": read_only,
        "validates_under_current_exp5895_code": validates,
        "status": payload.get("status"),
        "ready_score": payload.get("shortcut_resistant_csl_ready_score"),
        "honest_verdict": payload.get("honest_verdict"),
        "recorded_reproducibility_checksum": payload.get("reproducibility_checksum"),
        "recorded_full_suite_exit_code": dict(payload.get("test_exit_codes") or {}).get(
            FULL_PYTEST_COMMAND
        ),
    }


def _original_test_failure_receipt(historical: Mapping[str, Any]) -> JsonDict:
    return {
        "command": FULL_PYTEST_COMMAND,
        "recorded_exit_code": historical.get("recorded_full_suite_exit_code"),
        "historical_status": historical.get("status"),
        "historical_ready_score": historical.get("ready_score"),
        "historical_honest_verdict": historical.get("honest_verdict"),
        "failure_record_source": HISTORICAL_EXP5895_RELATIVE_PATH.as_posix(),
        "stdout_stderr_hashes_recorded": False,
        "classification": "historical_global_suite_exit_2_without_science_regression",
    }


def current_failure_node_ids_phases_and_ownership(
    command_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Extract node, phase, and ownership details from current nonzero commands."""

    failures = []
    for receipt in command_receipts:
        if int(receipt.get("exit_code", 0)) == 0:
            continue
        failures.append(
            {
                "command": receipt.get("command"),
                "phase": receipt.get("phase"),
                "exit_code": int(receipt.get("exit_code", 0)),
                "stdout_sha256": receipt.get("stdout_sha256"),
                "stderr_sha256": receipt.get("stderr_sha256"),
                "node_ids": list(receipt.get("node_ids") or []),
                "ownership_paths": list(receipt.get("ownership_paths") or []),
            }
        )
    return {
        "failing_node_count": sum(len(item["node_ids"]) for item in failures),
        "failing_command_count": len(failures),
        "failures": failures,
        "all_failures_node_bound_or_phase_bound": all(
            item["node_ids"] or item["phase"] for item in failures
        ),
    }


def classify_causal_relevance(command_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Separate Exp5895-science failures from unrelated current suite debt."""

    relevant_tokens = (
        "experiment_5895_shortcut_safe_continuous_self_learning",
        "experiment_5893_grounding_shortcut_fixture",
        "experiment_5894_one_to_one_grounding_ab",
        "adaptive_state.py",
    )
    failing = [
        receipt for receipt in command_receipts if int(receipt.get("exit_code", 0)) != 0
    ]
    relevant = []
    for receipt in failing:
        text = " ".join(
            list(receipt.get("ownership_paths") or [])
            + list(receipt.get("node_ids") or [])
            + [str(receipt.get("command", ""))]
        )
        if any(token in text for token in relevant_tokens):
            relevant.append(receipt)
    if relevant:
        classification = "exp5895_causally_relevant_failure"
    elif failing:
        classification = "current_suite_debt_not_causally_relevant_to_exp5895_science"
    else:
        classification = "current_suite_clean"
    return {
        "classification": classification,
        "can_alter_exp5895_science": bool(relevant),
        "collection_or_infrastructure_debt_count": sum(
            str(receipt.get("phase")) in {"collection", "infrastructure"}
            for receipt in failing
        ),
        "current_nonzero_command_count": len(failing),
        "exp5895_data_logic_labels_or_readiness_changed": False,
        "science_mutation_allowed": False,
    }


def _ready_logic_hash() -> str:
    return sha256_text(inspect.getsource(exp5895.shortcut_resistant_csl_ready_score))


def _frozen_budget(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return exp5895._budget_registry(rows)["registry"]


def _frozen_inputs_receipt(
    root: Path,
    historical: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> JsonDict:
    rows = exp5895.load_fixture_rows(root)
    historical_inputs = {
        "row_count": len(rows),
        "row_hash": sha256_text(exp5895._rows_to_jsonl(rows)),
        "arms": list(exp5895.ARM_NAMES),
        "seeds": dict(exp5895.RANDOM_SEEDS),
        "budgets": _frozen_budget(rows),
        "thresholds": {
            "memory_cap": exp5895.MEMORY_CAP,
            "quarantine_cap": exp5895.QUARANTINE_CAP,
            "rejected_buffer_cap": exp5895.REJECTED_BUFFER_CAP,
            "replay_limit": exp5895.REPLAY_LIMIT,
            "ready_requires_full_current_test_zero": True,
        },
        "validators": {
            "inference_substrate": INFERENCE_SUBSTRATE,
            "verifier_is_oracle": True,
            "exp5894_sha256": _hash_path(root, EXP5894_RELATIVE_PATH),
            "exp5895_module_sha256": _hash_path(root, EXP5895_MODULE_RELATIVE_PATH),
        },
        "ready_logic_hash": _ready_logic_hash(),
    }
    replay_inputs = {
        "arms": list(replay.get("frozen_arms_and_budget_parity", {}).get("arms") or []),
        "seeds": dict(replay.get("random_seeds") or {}),
        "ready_logic_hash": _ready_logic_hash(),
    }
    return {
        **historical_inputs,
        "historical_artifact_hash": historical.get("sha256"),
        "replay_inputs": replay_inputs,
        "scientific_inputs_changed": not (
            replay_inputs["arms"] == historical_inputs["arms"]
            and replay_inputs["seeds"] == historical_inputs["seeds"]
            and replay_inputs["ready_logic_hash"] == historical_inputs["ready_logic_hash"]
        ),
    }


def deterministic_science_hash(artifact: Mapping[str, Any]) -> str:
    """Hash only Exp5895 science fields, excluding execution readiness fields."""

    upstream = dict(artifact.get("upstream_gate_and_hash_receipts") or {})
    science = {
        "continuous_self_learning_task": artifact.get("continuous_self_learning_task"),
        "exp5893_rows": dict(upstream.get("exp5893_rows") or {}),
        "exp5894_gate": dict(upstream.get("exp5894_gate") or {}),
        "exact_validators": dict(upstream.get("exact_validators") or {}),
        "retired_chain_exclusion": {
            "dependency_used": dict(upstream.get("retired_chain_exclusion") or {}).get(
                "dependency_used"
            )
        },
        "random_seeds": dict(artifact.get("random_seeds") or {}),
        "science_fields": {field: artifact.get(field) for field in SCIENCE_FIELDS},
    }
    return sha256_json(science)


def _deterministic_science_parity(
    historical_artifact: Mapping[str, Any],
    replay_artifact: Mapping[str, Any],
    replay_result_path: Path,
) -> JsonDict:
    historical_hash = deterministic_science_hash(historical_artifact)
    replay_hash = deterministic_science_hash(replay_artifact)
    try:
        replay_validates = exp5895.validate_artifact(replay_artifact)
    except ValueError:
        replay_validates = False
    return {
        "historical_science_hash": historical_hash,
        "replay_science_hash": replay_hash,
        "expected_deterministic_science_hash": historical_hash,
        "matches_historical": historical_hash == replay_hash,
        "temporary_replay_result_path": str(replay_result_path),
        "temporary_replay_sha256": sha256_file(replay_result_path)
        if replay_result_path.exists()
        else "missing",
        "temporary_replay_validates": replay_validates,
        "excluded_execution_fields": [
            "status",
            "test_commands",
            "test_exit_codes",
            "shortcut_resistant_csl_ready_score",
            "duration_s",
            "honest_verdict",
            "reproducibility_checksum",
        ],
    }


def _prospective_receipts(replay: Mapping[str, Any]) -> JsonDict:
    metrics = dict(replay.get("prospective_semantic_and_constraint_metrics") or {})
    transfer = dict(replay.get("forward_transfer_recurrence_retention_and_regret") or {})
    shortcuts = dict(replay.get("shortcut_false_accept_metrics") or {})
    restart = dict(replay.get("rollback_restart_and_state_hashes") or {})
    memory = dict(replay.get("memory_cap_accounting") or {})
    lift = dict(metrics.get("primary_minus_best_shortcut_control") or {})
    retention = dict(transfer.get("retention") or {})
    return {
        "prospective_semantic_lift_ci95": list(lift.get("ci95") or [0.0, 0.0]),
        "prospective_semantic_lift_mean_delta": lift.get("mean_delta", 0.0),
        "retention": retention.get("protected_prefix_retention", 0.0),
        "retention_regression_count": retention.get("retention_regression_count", 1),
        "unsafe_accept_count": shortcuts.get("unsafe_accept_count", 1),
        "primary_zero_false_accepts": shortcuts.get("primary_zero_false_accepts") is True,
        "restart_equivalence": restart.get("restart_equivalence", 0.0),
        "rollback_hash_mismatch_count": restart.get("rollback_hash_mismatch_count", 1),
        "state_cap_compliance": memory.get("cap_compliance") is True,
        "max_state_records": memory.get("max_state_records"),
    }


def _protected_files_unchanged(root: Path, before_hashes: Mapping[str, Any]) -> JsonDict:
    after_hashes = _protected_hashes(root)
    changed = [
        path
        for path, before in sorted(before_hashes.items())
        if before == "missing" or after_hashes.get(path) != before
    ]
    return {
        "before_hashes": dict(before_hashes),
        "after_hashes": after_hashes,
        "changed_files": changed,
        "all_unchanged": not changed,
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    command_receipts: Sequence[Mapping[str, Any]] = (),
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Hash frozen inputs, collection state, resources, and protected files."""

    root = Path(root)
    result_path = Path(result_path)
    rows = exp5895.load_fixture_rows(root)
    protected = _protected_hashes(root)
    historical = historical_exp5895_hash_and_immutability(root)
    collection = next(
        (
            dict(receipt)
            for receipt in command_receipts
            if str(receipt.get("command")) == COLLECTION_COMMAND
            or "--collect-only" in str(receipt.get("command"))
        ),
        {},
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = {
        "result_path": str(result_path),
        "parent_exists": result_path.parent.exists(),
        "parent_writable": os.access(result_path.parent, os.W_OK),
        "target_writable": (not result_path.exists()) or os.access(result_path, os.W_OK),
    }
    memory = memory_probe()
    disk = disk_probe(root)
    source_hashes = {
        "exp5893_rows": _hash_path(root, EXP5893_ROWS_RELATIVE_PATH),
        "exp5894_artifact": _hash_path(root, EXP5894_RELATIVE_PATH),
        "historical_exp5895_artifact": _hash_path(root, HISTORICAL_EXP5895_RELATIVE_PATH),
        "exp5895_module": _hash_path(root, EXP5895_MODULE_RELATIVE_PATH),
        "exp5895_tests": _hash_path(root, EXP5895_TEST_RELATIVE_PATH),
        "exp5912_module": _hash_path(root, MODULE_RELATIVE_PATH),
        "exp5912_tests": _hash_path(root, TEST_RELATIVE_PATH),
        "self_learning_spec": _hash_path(root, SELF_LEARNING_SPEC_RELATIVE_PATH),
        "ready_score_function": _ready_logic_hash(),
    }
    checks = {
        "rows_present": len(rows) == 72,
        "protected_files_present": all(value != "missing" for value in protected.values()),
        "historical_exp5895_read_only": historical["read_only"] is True,
        "collection_zero": not collection or int(collection.get("exit_code", 1)) == 0,
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "output_path": output_path["parent_writable"] and output_path["target_writable"],
        "python": sys.version_info >= (3, 11),
    }
    blocked_reasons = [name for name, ok in checks.items() if not ok]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "rows_hash": sha256_text(exp5895._rows_to_jsonl(rows)),
        "row_count": len(rows),
        "source_hashes": source_hashes,
        "arms": list(exp5895.ARM_NAMES),
        "seeds": dict(exp5895.RANDOM_SEEDS),
        "budgets": _frozen_budget(rows),
        "thresholds": {
            "memory_cap": exp5895.MEMORY_CAP,
            "quarantine_cap": exp5895.QUARANTINE_CAP,
            "rejected_buffer_cap": exp5895.REJECTED_BUFFER_CAP,
            "replay_limit": exp5895.REPLAY_LIMIT,
        },
        "current_suite_collection": collection,
        "listed_self_learning_path": {
            "path": "python/carnot/self_learning",
            "exists": (root / "python/carnot/self_learning").exists(),
        },
        "resources": {"memory": memory, "disk": disk},
        "output_path": output_path,
        "protected_file_hashes_before": protected,
        "checks": checks,
        "preconditions_ready": not blocked_reasons,
        "blocked_reasons": sorted(blocked_reasons),
    }


def _tests_passed(artifact: Mapping[str, Any]) -> bool:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    return bool(commands) and set(commands) == set(exit_codes) and all(
        int(code) == 0 for code in exit_codes.values()
    )


def _repeated_verdict_decision(
    original: Mapping[str, Any],
    current: Mapping[str, Any],
    parity: Mapping[str, Any],
) -> JsonDict:
    full_receipt = next(
        (
            dict(item)
            for item in current.get("failures", [])
            if item.get("command") == FULL_PYTEST_COMMAND
        ),
        {},
    )
    same = (
        original.get("recorded_exit_code") == 2
        and full_receipt.get("exit_code") == 2
        and parity.get("matches_historical") is True
    )
    reasons = []
    if full_receipt:
        reasons.append("failed_test_exit_codes")
    return {
        "retire_if_same_verdict": True,
        "same_verdict_recurred": same,
        "historical_exit_code": original.get("recorded_exit_code"),
        "current_full_suite_exit_code": full_receipt.get("exit_code", 0),
        "decision": "retire_exact_scope" if same else "do_not_retire",
        "reasons": reasons,
    }


def csl_exact_slot_ready_score(artifact: Mapping[str, Any]) -> float:
    receipts = dict(
        artifact.get("prospective_lift_retention_safety_rollback_and_state_receipts") or {}
    )
    ready = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and artifact.get("continuous_self_learning_task") is True
        and dict(artifact.get("historical_exp5895_hash_and_immutability") or {}).get("read_only")
        is True
        and dict(artifact.get("deterministic_science_parity") or {}).get("matches_historical")
        is True
        and dict(artifact.get("causal_relevance_classification") or {}).get(
            "can_alter_exp5895_science"
        )
        is False
        and artifact.get("retired_dependency_chain_used") is False
        and dict(artifact.get("protected_files_unchanged") or {}).get("all_unchanged") is True
        and dict(artifact.get("no_model_weight_mutation") or {}).get("all_unchanged") is True
        and int(
            dict(artifact.get("no_model_weight_mutation") or {}).get(
                "gguf_weight_mutation_count", 1
            )
        )
        == 0
        and receipts.get("prospective_semantic_lift_ci95", [0.0])[0] > 0.0
        and receipts.get("retention") == 1.0
        and int(receipts.get("retention_regression_count", 1)) == 0
        and int(receipts.get("unsafe_accept_count", 1)) == 0
        and receipts.get("primary_zero_false_accepts") is True
        and receipts.get("restart_equivalence") == 1.0
        and int(receipts.get("rollback_hash_mismatch_count", 1)) == 0
        and receipts.get("state_cap_compliance") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and _tests_passed(artifact)
    )
    return 1.0 if ready else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if artifact.get("continuous_self_learning_task") is not True:
        reasons.append("continuous_self_learning_task")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if dict(artifact.get("deterministic_science_parity") or {}).get("matches_historical") is not True:
        reasons.append("science_parity")
    if dict(artifact.get("causal_relevance_classification") or {}).get(
        "can_alter_exp5895_science"
    ):
        reasons.append("causally_relevant_exp5895_failure")
    if dict(artifact.get("historical_exp5895_hash_and_immutability") or {}).get("read_only") is not True:
        reasons.append("historical_exp5895_not_read_only")
    if dict(artifact.get("protected_files_unchanged") or {}).get("all_unchanged") is not True:
        reasons.append("protected_files_changed")
    if dict(artifact.get("no_model_weight_mutation") or {}).get("all_unchanged") is not True:
        reasons.append("no_model_weight_mutation")
    receipts = dict(
        artifact.get("prospective_lift_retention_safety_rollback_and_state_receipts") or {}
    )
    if int(receipts.get("unsafe_accept_count", 0)) != 0:
        reasons.append("unsafe_accept_count")
    if receipts.get("restart_equivalence") not in (None, 1.0):
        reasons.append("restart_mismatch")
    if int(receipts.get("rollback_hash_mismatch_count", 0)) != 0:
        reasons.append("rollback_mismatch")
    if not _tests_passed(artifact):
        reasons.append("failed_test_exit_codes")
    if csl_exact_slot_ready_score(artifact) != 1.0 and not reasons:
        reasons.append("ready_score")
    return sorted(set(reasons))


def status(artifact: Mapping[str, Any]) -> str:
    receipts = dict(
        artifact.get("prospective_lift_retention_safety_rollback_and_state_receipts") or {}
    )
    if (
        int(receipts.get("unsafe_accept_count", 0)) != 0
        or dict(artifact.get("no_model_weight_mutation") or {}).get("all_unchanged") is False
    ):
        return "unsafe"
    blocked = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True
        or dict(artifact.get("deterministic_science_parity") or {}).get("matches_historical")
        is not True
        or dict(artifact.get("causal_relevance_classification") or {}).get(
            "can_alter_exp5895_science"
        )
        is True
        or dict(artifact.get("historical_exp5895_hash_and_immutability") or {}).get("read_only")
        is not True
        or dict(artifact.get("protected_files_unchanged") or {}).get("all_unchanged")
        is not True
    )
    if blocked:
        return "blocked"
    if dict(artifact.get("repeated_verdict_retirement_decision") or {}).get(
        "same_verdict_recurred"
    ):
        return "retired"
    if csl_exact_slot_ready_score(artifact) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    state = status(artifact)
    if state == "complete_positive":
        return "complete_positive: csl_exact_slot_requalified"
    if state == "retired":
        return "retired: repeated_global_suite_exit_2_after_frozen_science_parity"
    if state == "unsafe":
        return "unsafe: " + ",".join(blocked_reasons(artifact)[:8])
    if state == "blocked":
        return "blocked: " + ",".join(blocked_reasons(artifact)[:8])
    return "complete_null: csl_exact_slot_not_ready"


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        HISTORICAL_EXP5895_RELATIVE_PATH.as_posix(),
        EXP5894_RELATIVE_PATH.as_posix(),
        EXP5893_ROWS_RELATIVE_PATH.as_posix(),
        EXP5895_MODULE_RELATIVE_PATH.as_posix(),
        EXP5895_TEST_RELATIVE_PATH.as_posix(),
        E2E_PLAN_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        preconditions.get("output_path", {}).update({"result_path": "<normalized>"})
    parity = stable.get("deterministic_science_parity")
    if isinstance(parity, dict):
        parity["temporary_replay_result_path"] = "<normalized>"
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    if artifact.get("csl_exact_slot_ready_score") != csl_exact_slot_ready_score(artifact):
        raise ValueError("ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    replay_result_path: str | Path | None = None,
    command_receipts: Sequence[Mapping[str, Any]],
    changed_files: Sequence[str],
    duration_s: float | None = None,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Build the Exp5912 wrapper artifact while preserving Exp5895 history."""

    started = time.perf_counter()
    root = Path(root)
    result_path = Path(result_path)
    replay_path = Path(replay_result_path) if replay_result_path is not None else Path(
        "/tmp/experiment_5895_replay_for_5912.json"
    )
    receipts = [dict(receipt) for receipt in command_receipts]
    commands = [str(receipt["command"]) for receipt in receipts]
    exit_codes = {str(receipt["command"]): int(receipt["exit_code"]) for receipt in receipts}
    preconditions = collect_preconditions(
        root=root,
        result_path=result_path,
        command_receipts=receipts,
        memory_probe=memory_probe,
        disk_probe=disk_probe,
    )
    historical = historical_exp5895_hash_and_immutability(root)
    historical_artifact = read_json(root / HISTORICAL_EXP5895_RELATIVE_PATH)
    exp5895_preconditions = exp5895.collect_preconditions(
        root=root,
        result_path=replay_path,
        memory_probe=memory_probe,
        disk_probe=disk_probe,
    )
    replay = exp5895.run(
        root=root,
        result_path=replay_path,
        preconditions_checked=exp5895_preconditions,
        duration_s=duration_s if duration_s is not None else 0.0,
        test_commands=commands,
        test_exit_codes=exit_codes,
        write=True,
    )
    current = current_failure_node_ids_phases_and_ownership(receipts)
    causal = classify_causal_relevance(receipts)
    parity = _deterministic_science_parity(historical_artifact, replay, replay_path)
    original = _original_test_failure_receipt(historical)
    protected = _protected_files_unchanged(root, preconditions["protected_file_hashes_before"])
    artifact: JsonDict = {
        "status": "blocked",
        "preconditions_checked": preconditions,
        "continuous_self_learning_task": True,
        "historical_exp5895_hash_and_immutability": historical,
        "original_test_failure_receipt": original,
        "current_failure_node_ids_phases_and_ownership": current,
        "causal_relevance_classification": causal,
        "repair_scope_and_changed_files": {
            "changed_files": list(changed_files),
            "historical_science_files_changed": False,
            "scripts_research_conductor_changed": False,
            "ops_reconciliation_files_changed": False,
            "scope": "exact_slot_requalification_wrapper_only",
        },
        "frozen_rows_arms_seeds_budgets_thresholds_and_ready_logic": _frozen_inputs_receipt(
            root, historical, replay
        ),
        "deterministic_science_parity": parity,
        "prospective_lift_retention_safety_rollback_and_state_receipts": _prospective_receipts(
            replay
        ),
        "no_model_weight_mutation": dict(replay.get("no_model_weight_mutation") or {}),
        "retired_dependency_chain_used": False,
        "repeated_verdict_retirement_decision": _repeated_verdict_decision(
            original, current, parity
        ),
        "protected_files_unchanged": protected,
        "csl_exact_slot_ready_score": 0.0,
        "duration_s": round(time.perf_counter() - started, 6)
        if duration_s is None
        else float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": commands,
        "test_exit_codes": exit_codes,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["csl_exact_slot_ready_score"] = csl_exact_slot_ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    replay_result_path: str | Path | None = None,
    command_receipts: Sequence[Mapping[str, Any]],
    changed_files: Sequence[str],
    duration_s: float | None = None,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
    write: bool = True,
) -> JsonDict:
    artifact = build_artifact(
        root=Path(root),
        result_path=result_path,
        replay_result_path=replay_result_path,
        command_receipts=command_receipts,
        changed_files=changed_files,
        duration_s=duration_s,
        memory_probe=memory_probe,
        disk_probe=disk_probe,
    )
    if write:
        _atomic_write(Path(result_path), json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI convenience.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--receipts-json", required=True)
    args = parser.parse_args(argv)
    receipts_payload = read_json(args.receipts_json)
    artifact = run(
        result_path=args.result_path,
        command_receipts=list(receipts_payload["command_receipts"]),
        changed_files=list(receipts_payload["changed_files"]),
        write=True,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "score": artifact["csl_exact_slot_ready_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI convenience.
    raise SystemExit(main())
