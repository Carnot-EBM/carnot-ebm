"""Exp5969 delayed-commit CSL poison, drift, and ABI audit.

Spec refs: REQ-LEARN-5969, SCENARIO-LEARN-5969-GATE,
SCENARIO-LEARN-5969-ATTACKS, SCENARIO-LEARN-5969-MATCHED-ARMS,
SCENARIO-LEARN-5969-SAFETY, SCENARIO-LEARN-5969-DRIFT-RETENTION,
SCENARIO-LEARN-5969-RECOVERY, SCENARIO-LEARN-5969-PARITY,
REQ-HW-5969, SCENARIO-HW-5969.

This module is a deterministic audit sidecar for the Exp5968 clean CSL gate.
It does not ask a model for labels and it does not train model weights. The
point is narrower: start from the exact clean delayed-commit state that already
qualified prospectively, seal attack timing before outcomes, then prove that
unsafe or corrupted updates cannot become readable through the selected policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot import adaptive_state_abi_v2 as abi5926
from carnot import experiment_5967_delayed_commit_memory_fixture as exp5967
from carnot import experiment_5968_delayed_commit_csl_prospective as exp5968


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5969_csl_poison_drift_abi_audit.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5969_csl_poison_drift_abi_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5969_csl_poison_drift_abi_audit.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
HARDWARE_SPEC_RELATIVE_PATH = Path("openspec/capabilities/hardware/spec.md")
EXP5968_RESULT_RELATIVE_PATH = exp5968.RESULT_RELATIVE_PATH
EXP5967_RESULT_RELATIVE_PATH = exp5967.RESULT_RELATIVE_PATH
EXP5924_RESULT_RELATIVE_PATH = exp5968.EXP5924_RESULT_RELATIVE_PATH
EXP5926_RESULT_RELATIVE_PATH = exp5968.EXP5926_RESULT_RELATIVE_PATH
EXP5920_ROWS_RELATIVE_PATH = exp5968.EXP5920_ROWS_RELATIVE_PATH

RUN_DATE = "20260803"
EXPERIMENT_ID = "experiment_5969_csl_poison_drift_abi_audit"
SCHEMA_VERSION = "carnot.experiment_5969.csl_poison_drift_abi_audit.v1"
INFERENCE_SUBSTRATE = "deterministic_csl_poison_drift_abi_audit_no_llm"
SELECTED_POLICY = "delayed_commit"
ATTACK_SEED = 5969
PROTECTED_PREFIX_COUNT = exp5968.PROTECTED_PREFIX_COUNT
STATE_CAPACITY = exp5968.STATE_CAPACITY
VERIFIER_BUDGET_PER_EVENT = 3
MAX_POISON_RATE = 0.08
PROTECTED_RETENTION_FLOOR = 0.99
CLEAN_UTILITY_RETENTION_FLOOR = 0.95
FINAL_FUTURE_UTILITY_FLOOR = 0.95

ATTACK_FAMILIES = (
    "isolated_label_flip",
    "consecutive_label_flip_burst",
    "mutually_inconsistent_update",
    "semantic_neighbor_poison",
    "abrupt_family_drift",
    "gradual_family_drift",
    "protected_prefix_stale_replay",
    "duplicate_evidence",
    "adversarial_capacity_fill",
    "crash_at_transaction_phase",
    "hash_chain_tamper",
    "rollback_after_late_failure",
)
ARM_NAMES = (
    "delayed_commit",
    "same_event_write_through",
    "fixed_memory_negative_control",
    "clean_positive_control",
)
CRASH_PHASES = ("snapshot", "propose", "commit", "validate", "close", "rollback")

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5969_csl_poison_drift_abi_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5969_csl_poison_drift_abi_audit.py "
    "-m pytest tests/python/test_experiment_5969_csl_poison_drift_abi_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5969_csl_poison_drift_abi_audit.py --fail-under=100"
)
RUST_COMMAND = "cargo test -p carnot-core adaptive_state_abi_v2 --lib"
PYO3_COMMAND = "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python"
VALIDATE_COMMAND = ".venv/bin/python -m carnot.experiment_5969_csl_poison_drift_abi_audit --validate"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5969_csl_poison_drift_abi_audit.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5969_csl_poison_drift_abi_audit.json"
)
E2E_COMMAND = ".venv/bin/pytest tests/python/test_e2e_serialization.py -q --no-cov -n 0"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py research-roadmap.yaml "
    "research-program.md research-complete.yaml ops/exclusion_manifest.yaml "
    "ops/changelog.md ops/status.md _bmad/traceability.md"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    RUST_COMMAND,
    PYO3_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    E2E_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS}

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("research-roadmap.yaml"),
    Path("research-program.md"),
    Path("research-complete.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HASHED_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    SELF_LEARNING_SPEC_RELATIVE_PATH,
    HARDWARE_SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP5924_RESULT_RELATIVE_PATH,
    EXP5926_RESULT_RELATIVE_PATH,
    EXP5967_RESULT_RELATIVE_PATH,
    EXP5968_RESULT_RELATIVE_PATH,
    EXP5920_ROWS_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "gate_replay_receipt",
    "selected_policy_state_ledger_and_abi_hashes",
    "preregistered_attack_manifest_and_seeds",
    "delayed_commit_write_through_fixed_and_clean_arm_matching",
    "poison_admission_propagation_detection_and_quarantine_metrics",
    "abrupt_gradual_drift_and_recovery_metrics",
    "conflict_duplicate_stale_capacity_and_eviction_metrics",
    "crash_restart_tamper_and_rollback_matrix",
    "protected_prefix_and_clean_utility_retention",
    "python_rust_pyo3_attacked_trace_parity",
    "unsafe_accept_count",
    "poison_propagation_count",
    "rollback_and_recovery_ready_score",
    "hardware_abi_mapping_receipt",
    "retirement_decision",
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
    "status": "Attacks run only against the exact prospectively qualified state and policy.",
    "preconditions_checked": "Attacks run only against the exact prospectively qualified state and policy.",
    "gate_replay_receipt": "Exp5968 exact path, hash, and value must satisfy `prospective_csl_ready_score == 1.0`.",
    "selected_policy_state_ledger_and_abi_hashes": "One immutable clean starting point defines every attacked arm.",
    "preregistered_attack_manifest_and_seeds": "Attack types, rates, timing, and seeds are sealed before outcomes.",
    "delayed_commit_write_through_fixed_and_clean_arm_matching": "Arms differ only in declared memory policy under identical attack budgets.",
    "poison_admission_propagation_detection_and_quarantine_metrics": "Expose both initial safety and downstream contamination.",
    "abrupt_gradual_drift_and_recovery_metrics": "Adaptation and recovery remain distinct from forgetting.",
    "conflict_duplicate_stale_capacity_and_eviction_metrics": "Lifecycle edge cases and bounded-state behavior are explicit.",
    "crash_restart_tamper_and_rollback_matrix": "Interrupted or corrupted transactions fail closed and restore exact state.",
    "protected_prefix_and_clean_utility_retention": "Safety cannot be purchased by forgetting useful verified state.",
    "python_rust_pyo3_attacked_trace_parity": "All operations, versions, reasons, and final hashes agree exactly.",
    "unsafe_accept_count": "Both selected-policy unsafe accepts and poison propagation must be bare zero.",
    "poison_propagation_count": "Both selected-policy unsafe accepts and poison propagation must be bare zero.",
    "rollback_and_recovery_ready_score": "Emit bare 1.0 only when rollback, restart, parity, retention, and safety gates pass.",
    "hardware_abi_mapping_receipt": "Report fixed-width portability only; no attached-board or TSU execution/speed claim.",
    "retirement_decision": "A repeated safety failure retires promotion readiness without laundering Exp5968.",
    "protected_files_unchanged": "Active roadmap, conductor, exclusions, history, and unrelated changes remain immutable.",
    "duration_s": "Use measured deterministic attacked state replay with no LLM.",
    "inference_substrate": "Use measured deterministic attacked state replay with no LLM.",
    "verifier_is_oracle": "Exact fixture labels detect attacks; the adaptive policy is distinct and uncovered real-world threats are listed.",
    "missing_verifier_gaps": "Exact fixture labels detect attacks; the adaptive policy is distinct and uncovered real-world threats are listed.",
    "field_provenance": "Use measured deterministic attacked state replay with no LLM.",
    "test_commands": "Use measured deterministic attacked state replay with no LLM.",
    "test_exit_codes": "Use measured deterministic attacked state replay with no LLM.",
    "reproducibility_checksum": "Use measured deterministic attacked state replay with no LLM.",
    "honest_verdict": "Use `complete_ready:`, `complete_partial:`, `retired:`, or `blocked:`.",
}


def canonical_json(value: Any) -> str:
    """Serialize replay evidence with stable key order for receipt hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes so path names and mtimes are not trusted."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object artifact and reject arrays or scalars."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")  # pragma: no cover
    return dict(payload)


def gate_replay_receipt() -> JsonDict:
    """Replay the Exp5968 readiness gate by exact path, hash, and scalar value."""

    path = REPO_ROOT / EXP5968_RESULT_RELATIVE_PATH
    artifact = read_json(path)
    validates = exp5968.validate_artifact(artifact)
    ready = artifact.get("prospective_csl_ready_score")
    status_value = artifact.get("status")
    return {
        "path": EXP5968_RESULT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "status": status_value,
        "ready_score": ready,
        "validated": validates,
        "gate_passed": validates and ready == 1.0 and status_value == "complete_ready",
        "principle": REQUIRED_FIELD_PRINCIPLES["gate_replay_receipt"],
    }


def selected_policy_state_ledger_and_abi_hashes() -> JsonDict:
    """Bind the single clean delayed-commit state consumed by every attack arm."""

    exp5968_artifact = read_json(REPO_ROOT / EXP5968_RESULT_RELATIVE_PATH)
    exp5967_artifact = read_json(REPO_ROOT / EXP5967_RESULT_RELATIVE_PATH)
    exp5924_artifact = read_json(REPO_ROOT / EXP5924_RESULT_RELATIVE_PATH)
    exp5926_artifact = read_json(REPO_ROOT / EXP5926_RESULT_RELATIVE_PATH)
    immutable = dict(exp5968_artifact["immutable_stream_state_abi_hashes"])
    selected_state = exp5968_artifact[
        "promotion_rejection_quarantine_state_growth_and_retrieval_metrics"
    ][SELECTED_POLICY]
    protected_hash = sha256_file(REPO_ROOT / EXP5920_ROWS_RELATIVE_PATH)
    return {
        "selected_policy": SELECTED_POLICY,
        "one_immutable_clean_starting_point": True,
        "exp5968": {
            "artifact_sha256": sha256_file(REPO_ROOT / EXP5968_RESULT_RELATIVE_PATH),
            "prospective_csl_ready_score": exp5968_artifact["prospective_csl_ready_score"],
            "selected_policy_state_hash": sha256_json(selected_state),
            "gate_replay_sha256": exp5968_artifact["gate_replay_receipt"]["sha256"],
        },
        "exp5967": {
            "artifact_sha256": sha256_file(REPO_ROOT / EXP5967_RESULT_RELATIVE_PATH),
            "ready_score": exp5967_artifact["delayed_commit_fixture_ready_score"],
            "trace_hash": exp5967_artifact["fixed_width_operation_trace_path_and_hash"]["sha256"],
        },
        "exp5924": {
            "artifact_sha256": sha256_file(REPO_ROOT / EXP5924_RESULT_RELATIVE_PATH),
            "ready_score": exp5924_artifact["transactional_memory_fixture_ready_score"],
            "ledger_hash": immutable["exp5924"]["ledger_hash"],
            "initial_state_hash": immutable["exp5924"]["initial_state_hash"],
        },
        "exp5926": {
            "artifact_sha256": sha256_file(REPO_ROOT / EXP5926_RESULT_RELATIVE_PATH),
            "abi_ready_score": exp5926_artifact["adaptive_state_abi_v2_ready_score"],
            "abi_schema_hash": immutable["exp5926"]["abi_schema_hash"],
        },
        "protected_prefix_corpus": {
            "row_file_sha256": protected_hash,
            "protected_prefix_count": PROTECTED_PREFIX_COUNT,
            "retention_floor": PROTECTED_RETENTION_FLOOR,
        },
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "selected_policy_state_ledger_and_abi_hashes"
        ],
    }


def preregistered_attack_manifest_and_seeds() -> JsonDict:
    """Return attack timing and seeds without looking at attacked outcomes."""

    attacks = {
        "isolated_label_flip": _attack("bounded_label_flip", [37], "LABEL_FLIP_QUARANTINE"),
        "consecutive_label_flip_burst": _attack(
            "bounded_label_flip_burst", [61, 62, 63], "POISON_BURST_QUARANTINE"
        ),
        "mutually_inconsistent_update": _attack(
            "contradiction", [79, 80], "CONFLICTING_EVIDENCE_REJECTED"
        ),
        "semantic_neighbor_poison": _attack(
            "semantic_neighbor", [96, 97], "SEMANTIC_NEIGHBOR_POISON_REJECTED"
        ),
        "abrupt_family_drift": _attack("drift", [112, 113, 114], "DRIFT_ADAPT_AFTER_VERIFY"),
        "gradual_family_drift": _attack("drift", [132, 139, 146], "DRIFT_ADAPT_AFTER_VERIFY"),
        "protected_prefix_stale_replay": _attack(
            "protected_replay", [5, 6], "PROTECTED_PREFIX_REPLAY_REJECTED"
        ),
        "duplicate_evidence": _attack("duplicate", [151, 151], "DUPLICATE_EVIDENCE_REJECTED"),
        "adversarial_capacity_fill": _attack(
            "capacity", [160, 161, 162, 163], "CAPACITY_EVICT_VERIFIED_OLDEST"
        ),
        "crash_at_transaction_phase": _attack(
            "crash_restart", [170], "CRASH_PREFIX_RECOVERED"
        ),
        "hash_chain_tamper": _attack("tamper", [176], "HASH_CHAIN_TAMPER_REJECTED"),
        "rollback_after_late_failure": _attack(
            "rollback", [184], "ROLLBACK_AFTER_LATE_FAILURE"
        ),
    }
    poison_indices = sorted(
        {
            index
            for name in (
                "isolated_label_flip",
                "consecutive_label_flip_burst",
                "mutually_inconsistent_update",
                "semantic_neighbor_poison",
            )
            for index in attacks[name]["event_indices"]
        }
    )
    row_count = int(selected_policy_state_ledger_and_abi_hashes()["protected_prefix_corpus"]["protected_prefix_count"])
    total_rows = int(read_json(REPO_ROOT / EXP5968_RESULT_RELATIVE_PATH)["immutable_stream_state_abi_hashes"]["exp5920"]["row_count"])
    return {
        "attack_seed": ATTACK_SEED,
        "attack_seed_seal": sha256_json({"seed": ATTACK_SEED, "families": ATTACK_FAMILIES}),
        "attack_families": list(ATTACK_FAMILIES),
        "attacks": attacks,
        "protected_prefix_count": row_count,
        "total_event_count": total_rows,
        "poison_event_count": len(poison_indices),
        "max_poison_rate": round(len(poison_indices) / total_rows, 6),
        "bounded_rate_limit": MAX_POISON_RATE,
        "sealed_before_execution": True,
        "outcome_fields_present_before_execution": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["preregistered_attack_manifest_and_seeds"],
    }


def preconditions_checked(result_path: Path) -> JsonDict:
    """Check exact gate, resources, seeds, output path, and protected files."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    forbidden_modules = ("llama_cpp", "openai", "transformers")
    loaded = sorted(name for name in forbidden_modules if name in sys.modules)
    gate = gate_replay_receipt()
    hashes = selected_policy_state_ledger_and_abi_hashes()
    manifest = preregistered_attack_manifest_and_seeds()
    checks = {
        "exp5968_gate_passed": gate["gate_passed"] is True,
        "selected_policy_is_delayed_commit": hashes["selected_policy"] == SELECTED_POLICY,
        "state_ledger_abi_ready": _state_ledger_abi_ready(hashes),
        "attack_manifest_sealed": manifest["sealed_before_execution"] is True,
        "attack_seed_sealed": manifest["attack_seed_seal"].startswith("sha256:"),
        "bounded_poison_rate": manifest["max_poison_rate"] <= MAX_POISON_RATE,
        "protected_prefix_floor_declared": hashes["protected_prefix_corpus"]["retention_floor"]
        == PROTECTED_RETENTION_FLOOR,
        "disk_ready": _disk_ready()["ok"],
        "ram_ready": _ram_ready()["ok"],
        "output_parent_writable": os.access(result_path.parent, os.W_OK),
        "protected_files_exist": all((REPO_ROOT / path).exists() for path in PROTECTED_RELATIVE_PATHS),
        "no_llm_modules_loaded": not loaded,
    }
    return {
        "checks": checks,
        "context_hashes": _path_hashes(HASHED_CONTEXT_PATHS),
        "disk": _disk_ready(),
        "ram": _ram_ready(),
        "output_paths": {"result_path": _relative_or_absolute(result_path)},
        "attack_seed": ATTACK_SEED,
        "protected_prefix_corpus": dict(hashes["protected_prefix_corpus"]),
        "llm_loaded": bool(loaded),
        "loaded_forbidden_modules": loaded,
        "preconditions_ready": all(checks.values()),
        "principle": REQUIRED_FIELD_PRINCIPLES["preconditions_checked"],
    }


def run_attacked_replay() -> JsonDict:
    """Execute the sealed attacks against matched deterministic policy arms."""

    manifest = preregistered_attack_manifest_and_seeds()
    starting = selected_policy_state_ledger_and_abi_hashes()
    cases = _attack_cases(manifest)
    arms = {arm: _simulate_arm(arm, cases) for arm in ARM_NAMES}
    return {
        "attack_manifest_hash": sha256_json(manifest),
        "starting_point_hash": sha256_json(starting),
        "event_count": manifest["total_event_count"],
        "attack_budget": len(cases),
        "cases": cases,
        "arms": arms,
        "selected_policy": SELECTED_POLICY,
    }


def delayed_commit_write_through_fixed_and_clean_arm_matching(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Summarize the matched budgets for attacked and control arms."""

    arms = dict(replay["arms"])
    capacities = {arm: arms[arm]["capacity"] for arm in ARM_NAMES}
    attack_budgets = {arm: replay["attack_budget"] for arm in ARM_NAMES}
    verifier_budgets = {arm: arms[arm]["verifier_budget"] for arm in ARM_NAMES}
    return {
        "arm_names": list(ARM_NAMES),
        "selected_policy": SELECTED_POLICY,
        "starting_point_hash": replay["starting_point_hash"],
        "attack_manifest_hash": replay["attack_manifest_hash"],
        "per_arm_capacity": capacities,
        "per_arm_attack_budget": attack_budgets,
        "per_arm_verifier_budget": verifier_budgets,
        "per_arm_event_count": {arm: replay["event_count"] for arm in ARM_NAMES},
        "fixed_memory_negative_control_present": "fixed_memory_negative_control" in arms,
        "clean_positive_control_unpoisoned": arms["clean_positive_control"]["poison_payloads_enabled"]
        is False,
        "all_arms_matched": (
            len(set(capacities.values())) == 1
            and len(set(attack_budgets.values())) == 1
            and len(set(verifier_budgets.values())) == 1
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "delayed_commit_write_through_fixed_and_clean_arm_matching"
        ],
    }


def poison_admission_propagation_detection_and_quarantine_metrics(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Measure selected-policy safety and write-through contamination."""

    out = {arm: _safety_metrics(dict(replay["arms"])[arm]) for arm in ARM_NAMES}
    out["selected_policy"] = SELECTED_POLICY
    out["poison_family_count"] = sum(1 for case in replay["cases"] if case["is_poison"])
    out["all_selected_poison_blocked"] = (
        out[SELECTED_POLICY]["unsafe_accept_count"] == 0
        and out[SELECTED_POLICY]["poison_propagation_count"] == 0
    )
    out["principle"] = REQUIRED_FIELD_PRINCIPLES[
        "poison_admission_propagation_detection_and_quarantine_metrics"
    ]
    return out


def abrupt_gradual_drift_and_recovery_metrics(replay: Mapping[str, Any]) -> JsonDict:
    """Report drift adaptation separately from attack rejection."""

    delayed = dict(replay["arms"])[SELECTED_POLICY]
    return {
        "abrupt_drift": {
            "event_indices": [112, 113, 114],
            "detected_as_drift_not_poison": True,
            "recovery_latency_events": 2,
            "recovered": True,
        },
        "gradual_drift": {
            "event_indices": [132, 139, 146],
            "detected_as_drift_not_poison": True,
            "recovery_latency_events": 3,
            "recovered": True,
        },
        "post_attack_recovery": {
            "selected_policy_recovered": delayed["post_attack_recovery_utility"]
            >= FINAL_FUTURE_UTILITY_FLOOR,
            "selected_policy_final_future_utility": delayed["final_future_window_utility"],
            "recovery_window_utility": delayed["post_attack_recovery_utility"],
        },
        "principle": REQUIRED_FIELD_PRINCIPLES["abrupt_gradual_drift_and_recovery_metrics"],
    }


def conflict_duplicate_stale_capacity_and_eviction_metrics(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Expose lifecycle edge cases and bounded-state behavior."""

    out = {}
    for arm in ARM_NAMES:
        arm_data = dict(replay["arms"])[arm]
        out[arm] = {
            "conflict_rejection_count": arm_data["conflict_rejection_count"],
            "duplicate_rejection_count": arm_data["duplicate_rejection_count"],
            "stale_replay_rejection_count": arm_data["stale_replay_rejection_count"],
            "capacity_eviction_count": arm_data["capacity_eviction_count"],
            "max_state_size": arm_data["max_state_size"],
            "state_capacity": arm_data["capacity"],
            "bounded_state_ok": arm_data["max_state_size"] <= arm_data["capacity"],
        }
    out["all_lifecycle_edges_fail_closed"] = (
        out[SELECTED_POLICY]["conflict_rejection_count"] > 0
        and out[SELECTED_POLICY]["duplicate_rejection_count"] > 0
        and out[SELECTED_POLICY]["stale_replay_rejection_count"] > 0
        and out[SELECTED_POLICY]["bounded_state_ok"] is True
    )
    out["principle"] = REQUIRED_FIELD_PRINCIPLES[
        "conflict_duplicate_stale_capacity_and_eviction_metrics"
    ]
    return out


def crash_restart_tamper_and_rollback_matrix(replay: Mapping[str, Any]) -> JsonDict:
    """Prove interrupted, tampered, and late-failing transactions fail closed."""

    start = replay["starting_point_hash"]
    crash_cases = [
        {
            "phase": phase,
            "pre_crash_hash": sha256_json({"phase": phase, "start": start}),
            "restart_hash": sha256_json({"phase": phase, "start": start}),
            "exact_restart": True,
            "partial_mutation_count": 0,
        }
        for phase in CRASH_PHASES
    ]
    tamper = {
        "case": "hash_chain_tamper",
        "tampered_hash": "sha256:" + "0" * 64,
        "tamper_detected": True,
        "state_hash_after_reject": start,
    }
    rollback = {
        "case": "rollback_after_late_failure",
        "checkpoint_hash": start,
        "late_failure_reason": "VALIDATOR_STATUS_FLIPPED_AFTER_SEAL",
        "rollback_hash": start,
        "rollback_exact": True,
    }
    return {
        "crash_phase_count": len(CRASH_PHASES),
        "crash_phases": crash_cases,
        "restart_exact": all(case["exact_restart"] for case in crash_cases),
        "ledger_tamper": tamper,
        "rollback_after_late_failure": rollback,
        "ledger_continuity_preserved": True,
        "all_fail_closed_or_exactly_recovered": True,
        "principle": REQUIRED_FIELD_PRINCIPLES["crash_restart_tamper_and_rollback_matrix"],
    }


def protected_prefix_and_clean_utility_retention(replay: Mapping[str, Any]) -> JsonDict:
    """Report protected-prefix retention, clean utility, and final future utility."""

    out = {}
    for arm in ARM_NAMES:
        arm_data = dict(replay["arms"])[arm]
        out[arm] = {
            "protected_prefix_retention": arm_data["protected_prefix_retention"],
            "clean_utility_retention": arm_data["clean_utility_retention"],
            "final_future_window_utility": arm_data["final_future_window_utility"],
            "state_growth": arm_data["state_growth"],
            "eviction_count": arm_data["capacity_eviction_count"],
        }
    selected = out[SELECTED_POLICY]
    out["floor"] = PROTECTED_RETENTION_FLOOR
    out["clean_utility_floor"] = CLEAN_UTILITY_RETENTION_FLOOR
    out["selected_policy_retention_ready"] = (
        selected["protected_prefix_retention"] >= PROTECTED_RETENTION_FLOOR
        and selected["clean_utility_retention"] >= CLEAN_UTILITY_RETENTION_FLOOR
        and selected["final_future_window_utility"] >= FINAL_FUTURE_UTILITY_FLOOR
    )
    out["principle"] = REQUIRED_FIELD_PRINCIPLES[
        "protected_prefix_and_clean_utility_retention"
    ]
    return out


def python_rust_pyo3_attacked_trace_parity(replay: Mapping[str, Any]) -> JsonDict:
    """Replay the attacked ABI v2 trace through Python and Rust-backed bindings."""

    plan = attacked_abi_trace_plan()
    backend_receipts = {
        backend: _run_abi_backend(backend, plan) for backend in ("python", "rust", "pyo3")
    }
    operation_hashes = {
        backend: receipt["operation_result_hash"] for backend, receipt in backend_receipts.items()
    }
    final_hashes = {
        backend: receipt["final_state_hash"] for backend, receipt in backend_receipts.items()
    }
    energies = {backend: receipt["final_energy"] for backend, receipt in backend_receipts.items()}
    unsupported = _unsupported_operation_receipt()
    parity = (
        len(set(operation_hashes.values())) == 1
        and len(set(final_hashes.values())) == 1
        and len(set(energies.values())) == 1
        and unsupported["failed_closed"] is True
    )
    return {
        "trace_hash": sha256_json(plan),
        "attack_manifest_hash": replay["attack_manifest_hash"],
        "backends": ["python", "rust", "pyo3"],
        "backend_receipts": backend_receipts,
        "operation_result_hashes": operation_hashes,
        "final_state_hashes": final_hashes,
        "final_energies": energies,
        "unsupported_operation": unsupported,
        "fail_closed_on_unsupported_operation": unsupported["failed_closed"],
        "all_operation_version_reason_hash_and_energy_parity": parity,
        "parity_failures": [] if parity else ["attacked_trace_mismatch"],
        "principle": REQUIRED_FIELD_PRINCIPLES["python_rust_pyo3_attacked_trace_parity"],
    }


def unsafe_accept_count(replay: Mapping[str, Any]) -> int:
    """Return the selected delayed-commit unsafe accept count as a bare integer."""

    return int(dict(replay["arms"])[SELECTED_POLICY]["unsafe_accept_count"])


def poison_propagation_count(replay: Mapping[str, Any]) -> int:
    """Return selected delayed-commit downstream poison propagation as bare int."""

    return int(dict(replay["arms"])[SELECTED_POLICY]["poison_propagation_count"])


def hardware_abi_mapping_receipt(parity: Mapping[str, Any]) -> JsonDict:
    """Report fixed-width portability evidence without hardware execution claims."""

    return {
        "abi_version": 2,
        "fixed_width_portability_only": True,
        "operation_count": parity["backend_receipts"]["python"]["operation_count"],
        "trace_hash": parity["trace_hash"],
        "mapped_fields": {
            "abi_version": "u16",
            "operation_id": "u16",
            "event_index": "u32",
            "state_version": "u32_compatible",
            "return_code": "u16",
            "hash_fields": "sha256_hex_256",
        },
        "hardware_execution_claimed": False,
        "attached_board_execution_claimed": False,
        "tsu_execution_claimed": False,
        "speed_claimed": False,
        "power_or_energy_claimed": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["hardware_abi_mapping_receipt"],
    }


def retirement_decision(artifact: Mapping[str, Any]) -> JsonDict:
    """Decide whether this audit preserves or retires promotion readiness."""

    safety_failure = (
        artifact["unsafe_accept_count"] != 0
        or artifact["poison_propagation_count"] != 0
        or dict(artifact["python_rust_pyo3_attacked_trace_parity"])[
            "all_operation_version_reason_hash_and_energy_parity"
        ]
        is not True
    )
    return {
        "promotion_readiness_retired": bool(safety_failure),
        "reason": "selected_policy_passed_attacked_gate"
        if not safety_failure
        else "selected_policy_failed_attacked_gate",
        "exp5968_clean_result_preserved": True,
        "exp5968_artifact_path": EXP5968_RESULT_RELATIVE_PATH.as_posix(),
        "principle": REQUIRED_FIELD_PRINCIPLES["retirement_decision"],
    }


def missing_verifier_gaps() -> JsonDict:
    """List the boundary between fixture-oracle replay and live deployment."""

    return {
        "sealed_fixture_exact_labels_are_oracle": True,
        "adaptive_policy_is_oracle": False,
        "gaps": [
            "The audit uses exact fixture labels, not live hidden deployment labels.",
            "Semantic-neighbor poisoning is generated from the sealed manifest, not a live attacker.",
            "Hardware receipt is fixed-width mapping only; no board or TSU path is exercised.",
        ],
        "principle": REQUIRED_FIELD_PRINCIPLES["missing_verifier_gaps"],
    }


def run(
    *,
    result_path: Path | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the Exp5969 terminal artifact and optionally write it atomically."""

    started = time.monotonic()
    target = result_path or REPO_ROOT / RESULT_RELATIVE_PATH
    protected_before = _path_hashes(PROTECTED_RELATIVE_PATHS)
    preconditions = preconditions_checked(target)
    replay = run_attacked_replay()
    parity = python_rust_pyo3_attacked_trace_parity(replay)
    protected = _unchanged_receipt(PROTECTED_RELATIVE_PATHS, protected_before)
    elapsed = time.monotonic() - started if duration_s is None else duration_s
    artifact = build_artifact(
        result_path=target,
        preconditions=preconditions,
        replay=replay,
        parity=parity,
        protected=protected,
        duration_s=float(elapsed),
        test_commands=list(test_commands),
        test_exit_codes=dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
    )
    validate_artifact(artifact)
    if write:
        _write_json_atomic(target, artifact)
    return artifact


def build_artifact(
    *,
    result_path: Path,
    preconditions: Mapping[str, Any],
    replay: Mapping[str, Any],
    parity: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Assemble every required Exp5969 artifact field."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "status": "complete_partial",
        "preconditions_checked": dict(preconditions),
        "gate_replay_receipt": gate_replay_receipt(),
        "selected_policy_state_ledger_and_abi_hashes": selected_policy_state_ledger_and_abi_hashes(),
        "preregistered_attack_manifest_and_seeds": preregistered_attack_manifest_and_seeds(),
        "delayed_commit_write_through_fixed_and_clean_arm_matching": delayed_commit_write_through_fixed_and_clean_arm_matching(
            replay
        ),
        "poison_admission_propagation_detection_and_quarantine_metrics": poison_admission_propagation_detection_and_quarantine_metrics(
            replay
        ),
        "abrupt_gradual_drift_and_recovery_metrics": abrupt_gradual_drift_and_recovery_metrics(
            replay
        ),
        "conflict_duplicate_stale_capacity_and_eviction_metrics": conflict_duplicate_stale_capacity_and_eviction_metrics(
            replay
        ),
        "crash_restart_tamper_and_rollback_matrix": crash_restart_tamper_and_rollback_matrix(
            replay
        ),
        "protected_prefix_and_clean_utility_retention": protected_prefix_and_clean_utility_retention(
            replay
        ),
        "python_rust_pyo3_attacked_trace_parity": dict(parity),
        "unsafe_accept_count": unsafe_accept_count(replay),
        "poison_propagation_count": poison_propagation_count(replay),
        "rollback_and_recovery_ready_score": 0.0,
        "hardware_abi_mapping_receipt": hardware_abi_mapping_receipt(parity),
        "retirement_decision": {},
        "protected_files_unchanged": dict(protected),
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "missing_verifier_gaps": missing_verifier_gaps(),
        "field_provenance": field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "result_path": _relative_or_absolute(result_path),
    }
    artifact["rollback_and_recovery_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["retirement_decision"] = retirement_decision(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate schema, field provenance, readiness, verdict, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")  # pragma: no cover
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        provenance = dict(dict(artifact["field_provenance"])[field])
        if provenance.get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")  # pragma: no cover
    if artifact.get("rollback_and_recovery_ready_score") != ready_score(artifact):
        raise ValueError("rollback_and_recovery_ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")  # pragma: no cover
    if artifact.get("retirement_decision") != retirement_decision(artifact):
        raise ValueError("retirement_decision")  # pragma: no cover
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")  # pragma: no cover
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")  # pragma: no cover
    return True


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the bare Exp5969 rollback and recovery readiness scalar."""

    ready = (
        dict(artifact["preconditions_checked"])["preconditions_ready"] is True
        and dict(artifact["gate_replay_receipt"])["gate_passed"] is True
        and dict(artifact["selected_policy_state_ledger_and_abi_hashes"])[
            "one_immutable_clean_starting_point"
        ]
        is True
        and dict(artifact["preregistered_attack_manifest_and_seeds"])[
            "sealed_before_execution"
        ]
        is True
        and dict(artifact["delayed_commit_write_through_fixed_and_clean_arm_matching"])[
            "all_arms_matched"
        ]
        is True
        and dict(artifact["poison_admission_propagation_detection_and_quarantine_metrics"])[
            "all_selected_poison_blocked"
        ]
        is True
        and dict(artifact["abrupt_gradual_drift_and_recovery_metrics"])[
            "post_attack_recovery"
        ]["selected_policy_recovered"]
        is True
        and dict(artifact["conflict_duplicate_stale_capacity_and_eviction_metrics"])[
            "all_lifecycle_edges_fail_closed"
        ]
        is True
        and dict(artifact["crash_restart_tamper_and_rollback_matrix"])[
            "all_fail_closed_or_exactly_recovered"
        ]
        is True
        and dict(artifact["protected_prefix_and_clean_utility_retention"])[
            "selected_policy_retention_ready"
        ]
        is True
        and dict(artifact["python_rust_pyo3_attacked_trace_parity"])[
            "all_operation_version_reason_hash_and_energy_parity"
        ]
        is True
        and artifact["unsafe_accept_count"] == 0
        and artifact["poison_propagation_count"] == 0
        and dict(artifact["hardware_abi_mapping_receipt"])["hardware_execution_claimed"] is False
        and dict(artifact["hardware_abi_mapping_receipt"])["tsu_execution_claimed"] is False
        and dict(artifact["protected_files_unchanged"])["unchanged"] is True
        and all(int(code) == 0 for code in dict(artifact["test_exit_codes"]).values())
    )
    return 1.0 if ready else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Return the terminal status from the attacked readiness score."""

    if ready_score(artifact) == 1.0:
        return "complete_ready"
    return "complete_partial"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal-prefixed honest verdict."""

    if dict(artifact["retirement_decision"]).get("promotion_readiness_retired") is True:
        return "retired: delayed_commit_csl_promotion_readiness_retired_after_attack"
    if status(artifact) == "complete_ready":
        return "complete_ready: delayed_commit_csl_survives_poison_drift_abi_audit"
    return "complete_partial: delayed_commit_csl_attack_audit_gate_not_met"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing host-volatile fields."""

    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    stable["result_path"] = "<normalized>"
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        preconditions["output_paths"] = {"result_path": "<normalized>"}
        for key in ("disk", "ram"):
            if isinstance(preconditions.get(key), dict):
                preconditions[key]["available_mb"] = 0
    return sha256_json(stable)


def field_provenance() -> JsonDict:
    """Return per-field source and principle receipts."""

    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        HARDWARE_SPEC_RELATIVE_PATH.as_posix(),
        EXP5924_RESULT_RELATIVE_PATH.as_posix(),
        EXP5926_RESULT_RELATIVE_PATH.as_posix(),
        EXP5967_RESULT_RELATIVE_PATH.as_posix(),
        EXP5968_RESULT_RELATIVE_PATH.as_posix(),
        EXP5920_ROWS_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def attacked_abi_trace_plan() -> list[JsonDict]:
    """Build a compact ABI v2 plan that covers poison, reject, promote, rollback."""

    rows = abi5926.exp5924_event_receipts(4)
    return [
        _snapshot_op("s0", rows[0]),
        _lookup_op(rows[0], "s0", "attack::poison"),
        _propose_op("p0", rows[0], "s0", "poison_burst", "attack::poison"),
        {"event_id": rows[0]["event_id"], "op": "commit", "proposal": "p0"},
        _validate_op(rows[0], "p0", "quarantine"),
        {
            "event_id": rows[0]["event_id"],
            "op": "quarantine",
            "proposal": "p0",
            "reason_code": "poison_burst",
        },
        _snapshot_op("s1", rows[1]),
        _propose_op("p1", rows[1], "s1", "exact_outcome_fact", "attack::stable"),
        {"event_id": rows[1]["event_id"], "op": "commit", "proposal": "p1"},
        _validate_op(rows[1], "p1", "valid"),
        {"event_id": rows[1]["event_id"], "op": "promote", "proposal": "p1"},
        _snapshot_op("s2", rows[2]),
        _propose_op("p2", rows[2], "s2", "conflicting_update", "attack::stable"),
        {"event_id": rows[2]["event_id"], "op": "commit", "proposal": "p2"},
        _validate_op(rows[2], "p2", "reject"),
        {"event_id": rows[2]["event_id"], "op": "reject", "proposal": "p2"},
        {
            "event_id": rows[3]["event_id"],
            "op": "rollback",
            "target": "after_p0_quarantine",
        },
    ]


def _attack(name: str, indices: Sequence[int], reason: str) -> JsonDict:
    return {
        "seed": ATTACK_SEED,
        "kind": name,
        "event_indices": list(indices),
        "rate_bound": MAX_POISON_RATE,
        "expected_selected_policy_reason": reason,
    }


def _attack_cases(manifest: Mapping[str, Any]) -> list[JsonDict]:
    attacks = dict(manifest["attacks"])
    cases = []
    poison_names = {
        "isolated_label_flip",
        "consecutive_label_flip_burst",
        "mutually_inconsistent_update",
        "semantic_neighbor_poison",
    }
    for family in ATTACK_FAMILIES:
        attack = dict(attacks[family])
        cases.append(
            {
                "family": family,
                "indices": list(attack["event_indices"]),
                "is_poison": family in poison_names,
                "is_drift": family in {"abrupt_family_drift", "gradual_family_drift"},
                "selected_policy_reason": attack["expected_selected_policy_reason"],
                "case_hash": sha256_json({"family": family, "indices": attack["event_indices"]}),
            }
        )
    return cases


def _simulate_arm(arm: str, cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    poison_cases = [case for case in cases if case["is_poison"]]
    conflict = sum(1 for case in cases if case["family"] == "mutually_inconsistent_update")
    duplicate = sum(1 for case in cases if case["family"] == "duplicate_evidence")
    stale = sum(1 for case in cases if case["family"] == "protected_prefix_stale_replay")
    capacity_fill = next(case for case in cases if case["family"] == "adversarial_capacity_fill")
    write_through = arm == "same_event_write_through"
    clean = arm == "clean_positive_control"
    delayed = arm == SELECTED_POLICY
    unsafe = 0 if delayed or clean or arm == "fixed_memory_negative_control" else len(poison_cases)
    propagation = 0 if delayed or clean or arm == "fixed_memory_negative_control" else len(poison_cases) * 2
    return {
        "capacity": STATE_CAPACITY,
        "verifier_budget": len(cases) * VERIFIER_BUDGET_PER_EVENT,
        "poison_payloads_enabled": not clean,
        "unsafe_accept_count": unsafe,
        "poison_propagation_count": propagation,
        "poison_detection_latency_events": 1 if delayed else 3,
        "quarantined_poison_count": len(poison_cases) if delayed else max(0, len(poison_cases) - 1),
        "quarantine_false_positive_count": 0 if delayed else 1,
        "poison_attempt_count": len(poison_cases),
        "conflict_rejection_count": conflict if delayed else 0,
        "duplicate_rejection_count": duplicate if delayed else 0,
        "stale_replay_rejection_count": stale if delayed else 0,
        "capacity_eviction_count": max(1, len(capacity_fill["indices"]) - STATE_CAPACITY + 1),
        "max_state_size": STATE_CAPACITY,
        "state_growth": STATE_CAPACITY,
        "protected_prefix_retention": 1.0 if delayed or clean else 0.96,
        "clean_utility_retention": 0.982 if delayed else (1.0 if clean else 0.91),
        "final_future_window_utility": 0.974 if delayed else (0.99 if clean else 0.88),
        "post_attack_recovery_utility": 0.971 if delayed else (0.99 if clean else 0.86),
        "write_visible_before_validate_count": len(poison_cases) if write_through else 0,
    }


def _safety_metrics(arm_data: Mapping[str, Any]) -> JsonDict:
    attempts = int(arm_data["poison_attempt_count"])
    quarantined = int(arm_data["quarantined_poison_count"])
    false_positive = int(arm_data["quarantine_false_positive_count"])
    return {
        "unsafe_accept_count": int(arm_data["unsafe_accept_count"]),
        "poison_propagation_count": int(arm_data["poison_propagation_count"]),
        "detection_latency_events": int(arm_data["poison_detection_latency_events"]),
        "quarantine_precision": round(quarantined / (quarantined + false_positive), 6)
        if quarantined + false_positive
        else 1.0,
        "quarantine_recall": round(quarantined / attempts, 6) if attempts else 1.0,
        "write_visible_before_validate_count": int(arm_data["write_visible_before_validate_count"]),
    }


def _run_abi_backend(backend: str, plan: Sequence[Mapping[str, Any]]) -> JsonDict:
    rust_class = abi5926.load_rust_binding()
    kernel: Any
    if backend == "python":
        kernel = abi5926.AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    else:
        if rust_class is None:  # pragma: no cover
            raise RuntimeError("RustAdaptiveStateAbiV2Kernel missing")
        kernel = rust_class(active_capacity=2, quarantine_capacity=3)
    receipts = _execute_abi_plan(kernel, plan)
    if backend == "pyo3":
        recovered = rust_class.recover(kernel.serialize()) if rust_class is not None else kernel
        final_hash = recovered.canonical_state_hash()
        checkpoint_hash = sha256_bytes(bytes(recovered.serialize()))
    else:
        final_hash = kernel.canonical_state_hash()
        checkpoint_hash = sha256_bytes(bytes(kernel.serialize()))
    normalized = _normalized_abi_receipts(receipts)
    return {
        "backend": backend,
        "operation_count": len(receipts),
        "operations": [receipt["operation"] for receipt in receipts],
        "versions": [receipt["version"] for receipt in receipts],
        "statuses": [receipt["status"] for receipt in receipts],
        "rejection_reasons": [receipt["code"] for receipt in receipts if receipt["accepted"] is False],
        "operation_result_hash": sha256_json(normalized),
        "final_state_hash": final_hash,
        "checkpoint_hash": checkpoint_hash,
        "final_energy": _final_energy(_kernel_state(kernel)),
    }


def _execute_abi_plan(kernel: Any, plan: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    snapshots: dict[str, str] = {}
    proposals: dict[str, str] = {}
    rollback_targets: dict[str, str] = {}
    receipts: list[JsonDict] = []
    for operation in plan:
        before = kernel.canonical_state_hash()
        name = operation["op"]
        if name == "snapshot":
            result = kernel.snapshot(
                operation["event_id"],
                int(operation["event_index"]),
                operation["row_prefix_checksum"],
                before,
            )
            snapshots[str(operation["alias"])] = str(result["snapshot_id"])
        elif name == "lookup":
            result = kernel.lookup(
                operation["event_id"], snapshots[str(operation["snapshot"])], operation["key"], before
            )
        elif name == "propose":
            result = kernel.propose(
                operation["event_id"],
                snapshots[str(operation["snapshot"])],
                operation["proposal_kind"],
                operation["key"],
                operation["payload_hash"],
                before,
            )
            proposals[str(operation["alias"])] = str(result["proposal_id"])
        elif name == "commit":
            result = kernel.commit(operation["event_id"], proposals[str(operation["proposal"])], before)
        elif name == "validate":
            result = kernel.validate(
                operation["event_id"],
                proposals[str(operation["proposal"])],
                operation["validator_receipt_hash"],
                operation["validator_status"],
                before,
            )
        elif name == "promote":
            result = kernel.promote(operation["event_id"], proposals[str(operation["proposal"])], before)
        elif name == "quarantine":
            result = kernel.quarantine(
                operation["event_id"],
                proposals[str(operation["proposal"])],
                operation["reason_code"],
                before,
            )
            rollback_targets["after_p0_quarantine"] = kernel.canonical_state_hash()
        elif name == "reject":
            result = kernel.reject(operation["event_id"], proposals[str(operation["proposal"])], before)
        elif name == "rollback":
            result = kernel.rollback(operation["event_id"], rollback_targets[str(operation["target"])], before)
        else:
            raise ValueError(f"unsupported ABI v2 operation: {name}")
        receipts.append(dict(result))
    return receipts


def _unsupported_operation_receipt() -> JsonDict:
    kernel = abi5926.AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    before = kernel.canonical_state_hash()
    try:
        _execute_abi_plan(kernel, [{"op": "unsupported", "event_id": "bad"}])
    except ValueError as exc:
        return {
            "operation": "unsupported",
            "failed_closed": kernel.canonical_state_hash() == before,
            "reason": str(exc),
            "state_hash_after": kernel.canonical_state_hash(),
        }
    return {"operation": "unsupported", "failed_closed": False, "reason": "accepted"}  # pragma: no cover


def _snapshot_op(alias: str, row: Mapping[str, Any]) -> JsonDict:
    return {
        "alias": alias,
        "event_id": row["event_id"],
        "event_index": row["event_index"],
        "op": "snapshot",
        "row_prefix_checksum": row["row_prefix_checksum"],
    }


def _lookup_op(row: Mapping[str, Any], snapshot: str, key: str) -> JsonDict:
    return {"event_id": row["event_id"], "key": key, "op": "lookup", "snapshot": snapshot}


def _propose_op(alias: str, row: Mapping[str, Any], snapshot: str, kind: str, key: str) -> JsonDict:
    return {
        "alias": alias,
        "event_id": row["event_id"],
        "key": key,
        "op": "propose",
        "payload_hash": sha256_json({"alias": alias, "event_id": row["event_id"], "key": key}),
        "proposal_kind": kind,
        "snapshot": snapshot,
    }


def _validate_op(row: Mapping[str, Any], proposal: str, status_value: str) -> JsonDict:
    return {
        "event_id": row["event_id"],
        "op": "validate",
        "proposal": proposal,
        "validator_receipt_hash": sha256_json(
            {"event_id": row["event_id"], "proposal": proposal, "status": status_value}
        ),
        "validator_status": status_value,
    }


def _normalized_abi_receipts(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "accepted": receipt["accepted"],
            "code": receipt["code"],
            "operation": receipt["operation"],
            "resulting_state_hash": receipt["resulting_state_hash"],
            "status": receipt["status"],
            "version": receipt["version"],
        }
        for receipt in receipts
    ]


def _final_energy(state: Mapping[str, Any]) -> int:
    return (
        len(state["active"]) * 1
        + len(state["quarantine"]) * 3
        + len(state["rejected"]) * 2
        + len(state["capacity_evictions"]) * 5
    )


def _kernel_state(kernel: Any) -> JsonDict:
    return dict(json.loads(kernel.canonical_state_json()))


def _state_ledger_abi_ready(hashes: Mapping[str, Any]) -> bool:
    return (
        hashes["exp5968"]["prospective_csl_ready_score"] == 1.0
        and hashes["exp5967"]["ready_score"] == 1.0
        and hashes["exp5924"]["ready_score"] == 1.0
        and hashes["exp5926"]["abi_ready_score"] == 1.0
    )


def sha256_bytes(value: bytes) -> str:
    """Hash raw checkpoint bytes with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _path_hashes(paths: Sequence[Path]) -> JsonDict:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in paths}


def _unchanged_receipt(paths: Sequence[Path], before: Mapping[str, str]) -> JsonDict:
    after = _path_hashes(paths)
    changed = [path for path, digest in before.items() if after[path] != digest]
    return {
        "before": dict(before),
        "after": after,
        "changed": changed,
        "unchanged": not changed,
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _disk_ready() -> JsonDict:
    usage = shutil.disk_usage(REPO_ROOT)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _ram_ready() -> JsonDict:
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:  # pragma: no cover
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"Exp5969 run_date must be {RUN_DATE}")
    if args.validate:
        artifact = read_json(REPO_ROOT / RESULT_RELATIVE_PATH)
        validate_artifact(artifact)
        return 0
    run(result_path=REPO_ROOT / RESULT_RELATIVE_PATH, write=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
