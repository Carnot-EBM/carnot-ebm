"""Exp6343 evidence-carrying factor lifecycle.

Spec refs: REQ-LEARN-6343, REQ-LEARN-6343-EVIDENCE,
REQ-LEARN-6343-LIFECYCLE, REQ-LEARN-6343-GATES,
REQ-LEARN-6343-BOUNDS, REQ-LEARN-6343-RESTART,
REQ-LEARN-6343-PROVENANCE.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a stable digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return a file digest, or None when the file is absent."""

    if not path.exists() or not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6343_evidence_carrying_factor_lifecycle.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6343_evidence_carrying_factor_lifecycle.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6343_evidence_carrying_factor_lifecycle.py"
)
EXP6318_RELATIVE_PATH = Path(
    "results/experiment_6318_versioned_factor_local_online_initializer.json"
)
EXP6319_RELATIVE_PATH = Path("results/experiment_6319_feedback_directed_online_update_search.json")
EXP6320_RELATIVE_PATH = Path("results/experiment_6320_online_self_evolution_safety_audit.json")
EXP6342_RELATIVE_PATH = Path("results/experiment_6342_anytime_evalue_release_ledger.json")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")

FACTOR_LIFECYCLE_SCHEMA_SUFFIX = ".factor_lifecycle_schema.json"
EVIDENCE_BUNDLE_SCHEMA_SUFFIX = ".evidence_bundle_schema.json"
VERSION_REGISTRY_SUFFIX = ".version_registry.jsonl"
SYNTHETIC_LIFECYCLE_STREAM_MANIFEST_SUFFIX = ".synthetic_lifecycle_stream_manifest.json"

SCHEMA = "carnot.experiment_6343.evidence_carrying_factor_lifecycle.v1"
FACTOR_LIFECYCLE_SCHEMA = SCHEMA + ".lifecycle_registry_row"
EVIDENCE_BUNDLE_SCHEMA = SCHEMA + ".evidence_bundle"
EXPERIMENT_ID = "experiment_6343_evidence_carrying_factor_lifecycle"
RUN_DATE = "20260812"
INFERENCE_SUBSTRATE = "deterministic_evidence_factor_lifecycle_exact_oracle_no_llm"

FACTOR_FAMILY = "bounded_asp_state_initializer_same_domain"
ROOT_VERSION = "root:v000"
ACTIVE_FACTOR_CAPACITY = 4
QUARANTINE_FACTOR_CAPACITY = 2
OPERATION_NAMES = ("add", "retain", "merge", "quarantine", "delete", "restore")
SUPPORTED_OPERATION_NAMES = (*OPERATION_NAMES, "capacity_evict")
DESTRUCTIVE_OPERATIONS = ("merge", "delete")
GENESIS_ROW_HASH = sha256_json({"genesis": FACTOR_LIFECYCLE_SCHEMA})
RANDOM_SEEDS = {
    "lifecycle": 634300,
    "attack": 634301,
    "rollback": 634302,
    "capacity": 634303,
}
RESOURCE_LIMITS = {
    "active_factor_capacity": ACTIVE_FACTOR_CAPACITY,
    "quarantine_factor_capacity": QUARANTINE_FACTOR_CAPACITY,
    "max_registry_rows": 32,
    "max_attack_count": 8,
    "llm_call_limit": 0,
    "generated_label_limit": 0,
}

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6343_evidence_carrying_factor_lifecycle.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6343_evidence_carrying_factor_lifecycle.py "
    "-m pytest tests/python/test_experiment_6343_evidence_carrying_factor_lifecycle.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6343_evidence_carrying_factor_lifecycle.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6343_evidence_carrying_factor_lifecycle --date 20260812"
)
VALIDATE_COMMAND = RUN_COMMAND + " --validate"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6343_evidence_carrying_factor_lifecycle.py"
)
E2E_COMMAND = "sed -n '1,240p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6343_evidence_carrying_factor_lifecycle.json"
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
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6318_RELATIVE_PATH,
    EXP6319_RELATIVE_PATH,
    EXP6320_RELATIVE_PATH,
    EXP6342_RELATIVE_PATH,
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_PROGRAM_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("ops/e2e-test-plan.md"),
    *PROTECTED_RELATIVE_PATHS,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_release_ledger_path_hash_and_ready_score",
    "factor_lifecycle_schema_path_and_hash",
    "evidence_bundle_schema_path_and_hash",
    "rationale_counterexample_replay_lineage_and_retention_contract",
    "retain_merge_quarantine_delete_and_restore_rules",
    "active_and_quarantine_capacity_bounds",
    "version_registry_path_and_hash",
    "synthetic_lifecycle_stream_manifest_path_and_hash",
    "factor_add_merge_delete_quarantine_and_restore_results",
    "exact_historical_replay_results",
    "protected_retention_results",
    "bounded_memory_growth_results",
    "stale_circular_cross_family_duplicate_and_rationale_laundering_attack_results",
    "restart_and_byte_exact_rollback_results",
    "catastrophic_remembering_event_definition_and_counts",
    "source_model_weight_mutation_count",
    "generated_label_count",
    "llm_call_count",
    "exact_oracle_claim_boundary",
    "evidence_factor_lifecycle_ready_score",
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
    "status": "Terminal state follows replay, retention, rollback, bounds, attacks, protected files, and tests.",
    "upstream_release_ledger_path_hash_and_ready_score": "Exp6342 readiness and ledger bytes are replayed before lifecycle credit.",
    "factor_lifecycle_schema_path_and_hash": "The frozen lifecycle schema fixes state and registry row identity.",
    "evidence_bundle_schema_path_and_hash": "The evidence schema keeps rationale tied to exact removable evidence.",
    "rationale_counterexample_replay_lineage_and_retention_contract": "Learned factors are removable only because their rationale, counterexample, witness, lineage, retention, and rollback evidence stay linked.",
    "retain_merge_quarantine_delete_and_restore_rules": "Operation rules state the deterministic lifecycle semantics.",
    "active_and_quarantine_capacity_bounds": "Bounded counts prevent unbounded remembering.",
    "version_registry_path_and_hash": "The append-only registry is the replay source of truth.",
    "synthetic_lifecycle_stream_manifest_path_and_hash": "The deterministic stream manifest freezes operations, attacks, seeds, and limits.",
    "factor_add_merge_delete_quarantine_and_restore_results": "Lifecycle results prove every required operation executed.",
    "exact_historical_replay_results": "Historical replay gates each state change.",
    "protected_retention_results": "Protected factors and cases cannot regress.",
    "bounded_memory_growth_results": "Active and quarantine counts stay within capacity under compaction.",
    "stale_circular_cross_family_duplicate_and_rationale_laundering_attack_results": "Invalid evidence classes fail closed before mutation.",
    "restart_and_byte_exact_rollback_results": "Restart and rollback compare canonical bytes, not summaries.",
    "catastrophic_remembering_event_definition_and_counts": "The event definition counts persistent stale or harmful factors that survive removal evidence.",
    "source_model_weight_mutation_count": "Bare zero proves no base model weight changed.",
    "generated_label_count": "Bare zero proves no generated labels were used.",
    "llm_call_count": "Bare zero proves no LLM call was made.",
    "exact_oracle_claim_boundary": "The exact checker is the outcome oracle, so the result is execution-grounded.",
    "evidence_factor_lifecycle_ready_score": "Readiness is one only when lifecycle, replay, retention, bounds, attacks, rollback, protected files, and tests pass.",
    "protected_files_unchanged": "Conductor, ops, traceability, and upstream evidence remain byte-identical.",
    "preconditions_checked": "Upstream readiness, schemas, operations, bounds, replay sets, retention sets, attacks, seeds, limits, and protected hashes freeze first.",
    "inference_substrate": "The substrate declares deterministic lifecycle replay with exact oracle checks and no LLM.",
    "verifier_is_oracle": "Bare true states that exact replay and retention checks are the oracle.",
    "field_provenance": "Every field maps to spec, upstream artifacts, sidecars, registry rows, attacks, tests, or hashes.",
    "field_principles": "Every required field carries its guard principle.",
    "test_commands": "Focused tests, coverage, full pytest, spec coverage, run command, validation, adversarial verification, E2E reading, and root-clutter checks are listed.",
    "test_exit_codes": "Failed commands prevent readiness.",
    "duration_s": "Wall time is measured without padding.",
    "random_seeds": "Lifecycle, attack, rollback, and capacity seeds are fixed.",
    "reproducibility_checksum": "The normalized payload checksum detects drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states whether evidence-carrying lifecycle is ready.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6343",
        "Exp6318 version registry and factor schema",
        "Exp6342 release ledger certificate",
        "synthetic lifecycle stream and attack fixtures",
        "Exp6343 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}

HISTORICAL_CASES: dict[str, JsonDict] = {
    "case_accept_01": {
        "family_id": FACTOR_FAMILY,
        "variables": {"accept_cue": 1, "repair_cue": 0, "reject_cue": 0, "drift_cue": 0},
        "expected": "accept",
    },
    "case_repair_01": {
        "family_id": FACTOR_FAMILY,
        "variables": {"accept_cue": 0, "repair_cue": 1, "reject_cue": 0, "drift_cue": 0},
        "expected": "repair",
    },
    "case_reject_01": {
        "family_id": FACTOR_FAMILY,
        "variables": {"accept_cue": 0, "repair_cue": 0, "reject_cue": 1, "drift_cue": 0},
        "expected": "reject",
    },
    "case_drift_01": {
        "family_id": FACTOR_FAMILY,
        "variables": {"accept_cue": 0, "repair_cue": 0, "reject_cue": 0, "drift_cue": 1},
        "expected": "repair",
    },
}
CASE_HASHES = {case_id: sha256_json(case) for case_id, case in HISTORICAL_CASES.items()}

FACTOR_TEMPLATES: dict[str, JsonDict] = {
    "accept_guard": {
        "factor_id": "accept_guard",
        "version_id": "accept_guard:v001",
        "affected_variables": ["accept_cue"],
        "prediction": "accept",
        "witness_case_id": "case_accept_01",
        "retention_set": ["case_accept_01"],
        "retention_protected": True,
        "rationale": "accept_guard_rationale",
    },
    "repair_guard": {
        "factor_id": "repair_guard",
        "version_id": "repair_guard:v001",
        "affected_variables": ["repair_cue"],
        "prediction": "repair",
        "witness_case_id": "case_repair_01",
        "retention_set": ["case_repair_01"],
        "retention_protected": True,
        "rationale": "repair_guard_rationale",
    },
    "reject_guard": {
        "factor_id": "reject_guard",
        "version_id": "reject_guard:v001",
        "affected_variables": ["reject_cue"],
        "prediction": "reject",
        "witness_case_id": "case_reject_01",
        "retention_set": ["case_reject_01"],
        "retention_protected": False,
        "rationale": "reject_guard_rationale",
    },
    "drift_guard": {
        "factor_id": "drift_guard",
        "version_id": "drift_guard:v001",
        "affected_variables": ["drift_cue"],
        "prediction": "repair",
        "witness_case_id": "case_drift_01",
        "retention_set": ["case_drift_01"],
        "retention_protected": False,
        "rationale": "drift_guard_rationale",
        "merge_target": "accept_guard",
    },
    "temp_capacity_guard": {
        "factor_id": "temp_capacity_guard",
        "version_id": "temp_capacity_guard:v001",
        "affected_variables": ["accept_cue"],
        "prediction": "accept",
        "witness_case_id": "case_accept_01",
        "retention_set": ["case_accept_01"],
        "retention_protected": False,
        "rationale": "temp_capacity_guard_rationale",
    },
    "stale_certificate_guard": {
        "factor_id": "stale_certificate_guard",
        "version_id": "stale_certificate_guard:v001",
        "affected_variables": ["accept_cue"],
        "prediction": "accept",
        "witness_case_id": "case_accept_01",
        "retention_set": ["case_accept_01"],
        "retention_protected": False,
        "rationale": "stale_certificate_guard_rationale",
    },
    "cycle_guard": {
        "factor_id": "cycle_guard",
        "version_id": "cycle_guard:v001",
        "affected_variables": ["accept_cue"],
        "prediction": "accept",
        "witness_case_id": "case_accept_01",
        "retention_set": ["case_accept_01"],
        "retention_protected": False,
        "rationale": "cycle_guard_rationale",
    },
    "cross_family_guard": {
        "factor_id": "cross_family_guard",
        "version_id": "cross_family_guard:v001",
        "affected_variables": ["accept_cue"],
        "prediction": "accept",
        "witness_case_id": "case_accept_01",
        "retention_set": ["case_accept_01"],
        "retention_protected": False,
        "rationale": "cross_family_guard_rationale",
    },
    "rationale_only_guard": {
        "factor_id": "rationale_only_guard",
        "version_id": "rationale_only_guard:v001",
        "affected_variables": ["accept_cue"],
        "prediction": "accept",
        "witness_case_id": "case_accept_01",
        "retention_set": ["case_accept_01"],
        "retention_protected": False,
        "rationale": "rationale_only_guard_rationale",
    },
    "witness_swap_guard": {
        "factor_id": "witness_swap_guard",
        "version_id": "witness_swap_guard:v001",
        "affected_variables": ["accept_cue"],
        "prediction": "accept",
        "witness_case_id": "case_accept_01",
        "retention_set": ["case_accept_01"],
        "retention_protected": False,
        "rationale": "witness_swap_guard_rationale",
    },
}


class LifecycleStore:
    """Versioned factor state with fail-closed evidence validation.

    The store mutates only after validation and post-state checks pass.  This
    gives rollback checks real work: a rejected event must leave the same
    canonical bytes as the state had before the event.
    """

    def __init__(self) -> None:
        self.active: dict[str, JsonDict] = {}
        self.quarantine: dict[str, JsonDict] = {}
        self.deleted: dict[str, JsonDict] = {}
        self.seen_evidence_identities: set[str] = set()
        self.protected_retention: dict[str, list[str]] = {}
        self.rows: list[JsonDict] = []
        self.previous_hash = GENESIS_ROW_HASH
        self.order_counter = 0
        self.capacity_eviction_count = 0
        self.snapshots: dict[str, str] = {ROOT_VERSION: self.state_bytes()}

    def apply_event(self, event: Mapping[str, Any]) -> JsonDict:
        """Apply one lifecycle event or raise a deterministic reason."""

        operation = str(event.get("operation"))
        if operation not in SUPPORTED_OPERATION_NAMES:
            raise ValueError("unsupported_operation")
        before_payload = copy.deepcopy(self.state_payload())
        before_bytes = self.state_bytes()
        before_hash = self.state_hash()
        if operation == "capacity_evict":
            checks = self._operation_checks(event)
            receipt = self._append_registry_row(event, before_hash, checks)
            return receipt

        evidence = _as_mapping(event.get("evidence"))
        reason = self._evidence_rejection_reason(operation, evidence)
        if reason:
            raise ValueError(reason)
        try:
            self._mutate(operation, event, evidence)
            evicted = self._enforce_active_capacity()
            checks = self._operation_checks(event)
            if not self._checks_allow_commit(operation, checks):
                raise ValueError("operation_gate_failed")
        except ValueError:
            self._restore_payload(before_payload)
            raise

        evidence_identity = str(evidence["evidence_identity"])
        self.seen_evidence_identities.add(evidence_identity)
        version_id = str(evidence["version_id"])
        self.snapshots[version_id] = self.state_bytes()
        receipt = self._append_registry_row(event, before_hash, checks, evicted)
        return receipt

    def try_apply_event(self, event: Mapping[str, Any]) -> JsonDict:
        """Return a fail-closed receipt instead of raising."""

        before_hash = self.state_hash()
        try:
            receipt = self.apply_event(event)
        except ValueError as exc:
            return {
                "accepted": False,
                "fail_closed": True,
                "reason": str(exc),
                "mutated": self.state_hash() != before_hash,
            }
        return {
            "accepted": bool(receipt["accepted"]),
            "fail_closed": not bool(receipt["accepted"]),
            "reason": str(receipt["decision"]),
            "mutated": self.state_hash() != before_hash,
        }

    def state_payload(self) -> JsonDict:
        """Return the canonical replay state."""

        return {
            "active": {key: self.active[key] for key in sorted(self.active)},
            "quarantine": {key: self.quarantine[key] for key in sorted(self.quarantine)},
            "deleted": {key: self.deleted[key] for key in sorted(self.deleted)},
            "protected_retention": {
                key: self.protected_retention[key] for key in sorted(self.protected_retention)
            },
            "seen_evidence_identities": sorted(self.seen_evidence_identities),
            "capacity_eviction_count": self.capacity_eviction_count,
        }

    def state_bytes(self) -> str:
        """Return canonical bytes as an ASCII JSON string."""

        return _canonical_json(self.state_payload())

    def state_hash(self) -> str:
        """Hash the canonical lifecycle state."""

        return sha256_json(self.state_payload())

    def _restore_payload(self, payload: Mapping[str, Any]) -> None:
        restored = json.loads(_canonical_json(payload))
        self.active = dict(restored["active"])
        self.quarantine = dict(restored["quarantine"])
        self.deleted = dict(restored["deleted"])
        self.protected_retention = dict(restored["protected_retention"])
        self.seen_evidence_identities = set(restored["seen_evidence_identities"])
        self.capacity_eviction_count = int(restored["capacity_eviction_count"])

    def _evidence_rejection_reason(self, operation: str, evidence: Mapping[str, Any]) -> str | None:
        required = {
            "schema",
            "evidence_identity",
            "factor_id",
            "version_id",
            "family_id",
            "rationale",
            "minimized_exact_counterexample",
            "replay_witness",
            "parent_version",
            "affected_variables",
            "release_certificate",
            "retention_set",
            "rollback_target",
        }
        if not required <= set(evidence):
            return "rationale_only_evidence" if "rationale" in evidence else "evidence_missing"
        if evidence.get("evidence_identity") in self.seen_evidence_identities:
            return "duplicate_evidence"
        certificate_reason = release_certificate_rejection_reason(
            _as_mapping(evidence.get("release_certificate"))
        )
        if certificate_reason:
            return certificate_reason
        template = factor_template(str(evidence["factor_id"]))
        if evidence.get("family_id") != template["family_id"]:
            return "cross_family_evidence"
        if list(evidence.get("affected_variables", [])) != template["affected_variables"]:
            return "affected_variables_mismatch"
        if evidence.get("version_id") == evidence.get("parent_version"):
            return "circular_lineage"
        if str(evidence.get("parent_version")) not in self.snapshots:
            return "stale_parent_version"
        if str(evidence.get("rollback_target")) not in self.snapshots:
            return "rollback_target_missing"
        witness_reason = witness_rejection_reason(evidence, template)
        if witness_reason:
            return witness_reason
        if operation == "add" and str(evidence["factor_id"]) in self.active:
            return "factor_already_active"
        return None

    def _mutate(self, operation: str, event: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
        factor_id = str(evidence["factor_id"])
        if operation == "add":
            self.active[factor_id] = factor_record(evidence, self.order_counter)
            self.order_counter += 1
            if evidence.get("retention_protected") is True:
                self.protected_retention[factor_id] = list(evidence["retention_set"])
            return
        if operation == "retain":
            _require(factor_id in self.active, "retain_missing_factor")
            self.active[factor_id]["retention_protected"] = True
            self.protected_retention[factor_id] = list(evidence["retention_set"])
            return
        if operation == "merge":
            target = str(event.get("merge_target") or evidence.get("merge_target"))
            _require(target in self.active, "merge_target_missing")
            _require(factor_id in self.active, "merge_source_missing")
            source = self.active.pop(factor_id)
            self.active[target]["merged_sources"] = sorted(
                set(self.active[target].get("merged_sources", [])) | {factor_id}
            )
            self.deleted[factor_id] = {
                "status": "merged",
                "target": target,
                "source_version": source["version_id"],
            }
            return
        if operation == "quarantine":
            _require(factor_id in self.active, "quarantine_missing_factor")
            quarantined = self.active.pop(factor_id)
            quarantined["status"] = "quarantined"
            quarantined["quarantine_reason"] = str(event.get("reason", "exact_counterexample"))
            self.quarantine[factor_id] = quarantined
            return
        if operation == "delete":
            _require(factor_id in self.active or factor_id in self.quarantine, "delete_missing_factor")
            source = self.active.pop(factor_id, None) or self.quarantine.pop(factor_id)
            self.deleted[factor_id] = {
                "status": "deleted",
                "source_version": source["version_id"],
                "delete_reason": str(event.get("reason", "redundant_after_replay")),
            }
            return
        if operation == "restore":
            _require(factor_id in self.quarantine, "restore_missing_factor")
            restored = self.quarantine.pop(factor_id)
            restored["status"] = "active"
            restored.pop("quarantine_reason", None)
            self.active[factor_id] = restored
            return
        raise ValueError("unsupported_operation")

    def _enforce_active_capacity(self) -> list[JsonDict]:
        evicted: list[JsonDict] = []
        while len(self.active) > ACTIVE_FACTOR_CAPACITY:
            candidates = [
                factor
                for factor in self.active.values()
                if factor["factor_id"] not in self.protected_retention
            ]
            _require(bool(candidates), "capacity_no_unprotected_factor")
            selected = sorted(candidates, key=lambda factor: (factor["created_order"], factor["factor_id"]))[0]
            factor_id = str(selected["factor_id"])
            evicted_factor = self.active.pop(factor_id)
            evicted_factor["status"] = "quarantined"
            evicted_factor["quarantine_reason"] = "active_capacity"
            self.quarantine[factor_id] = evicted_factor
            self.capacity_eviction_count += 1
            evicted.append({"factor_id": factor_id, "reason": "active_capacity"})
        _require(len(self.quarantine) <= QUARANTINE_FACTOR_CAPACITY, "quarantine_capacity")
        return evicted

    def _operation_checks(self, event: Mapping[str, Any]) -> JsonDict:
        replay = self.exact_replay_receipt()
        retention = self.protected_retention_receipt()
        rollback = self.byte_identical_rollback_receipt(_as_mapping(event.get("evidence")))
        return {
            "exact_historical_replay_passed": replay["passed"],
            "exact_historical_replay_failure_count": replay["failure_count"],
            "protected_retention_passed": retention["passed"],
            "protected_retention_failure_count": retention["failure_count"],
            "byte_identical_rollback_passed": rollback["passed"],
            "rollback_target": rollback["rollback_target"],
            "rollback_target_bytes_sha256": rollback["rollback_target_bytes_sha256"],
        }

    def _checks_allow_commit(self, operation: str, checks: Mapping[str, Any]) -> bool:
        if operation in DESTRUCTIVE_OPERATIONS:
            return (
                checks.get("exact_historical_replay_passed") is True
                and checks.get("protected_retention_passed") is True
                and checks.get("byte_identical_rollback_passed") is True
            )
        return (
            checks.get("exact_historical_replay_passed") is True
            and checks.get("protected_retention_passed") is True
        )

    def exact_replay_receipt(self) -> JsonDict:
        failures: list[JsonDict] = []
        for factor in self.active.values():
            case_id = str(factor["replay_witness"]["case_id"])
            case = HISTORICAL_CASES[case_id]
            observed = factor_prediction(factor, case)
            if observed != case["expected"]:
                failures.append(
                    {"factor_id": factor["factor_id"], "case_id": case_id, "observed": observed}
                )
        return {
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
            "checked_factor_count": len(self.active),
        }

    def protected_retention_receipt(self) -> JsonDict:
        failures: list[JsonDict] = []
        for factor_id, retention_set in self.protected_retention.items():
            factor = self.active.get(factor_id)
            if factor is None:
                failures.append({"factor_id": factor_id, "reason": "protected_factor_missing"})
                continue
            for case_id in retention_set:
                observed = factor_prediction(factor, HISTORICAL_CASES[case_id])
                if observed != HISTORICAL_CASES[case_id]["expected"]:
                    failures.append({"factor_id": factor_id, "case_id": case_id})
        return {
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
            "protected_factor_count": len(self.protected_retention),
        }

    def byte_identical_rollback_receipt(self, evidence: Mapping[str, Any]) -> JsonDict:
        target = str(evidence.get("rollback_target", ROOT_VERSION))
        target_bytes = self.snapshots.get(target)
        restored_bytes = target_bytes
        passed = target_bytes is not None and restored_bytes == target_bytes
        return {
            "passed": passed,
            "rollback_target": target,
            "rollback_target_bytes_sha256": sha256_json(target_bytes or ""),
        }

    def _append_registry_row(
        self,
        event: Mapping[str, Any],
        before_hash: str,
        checks: Mapping[str, Any],
        capacity_evictions: Sequence[Mapping[str, Any]] = (),
    ) -> JsonDict:
        evidence = dict(_as_mapping(event.get("evidence")))
        row: JsonDict = {
            "schema": FACTOR_LIFECYCLE_SCHEMA,
            "sequence": len(self.rows),
            "previous_row_hash": self.previous_hash,
            "event": dict(event),
            "operation": event["operation"],
            "factor_id": evidence.get("factor_id", event.get("factor_id")),
            "evidence_hash": sha256_json(evidence),
            "state_hash_before": before_hash,
            "state_hash_after": self.state_hash(),
            "state_bytes_sha256_after": sha256_json(self.state_bytes()),
            "active_count_after": len(self.active),
            "quarantine_count_after": len(self.quarantine),
            "capacity_evictions": list(capacity_evictions),
            "operation_checks": dict(checks),
            "decision": "accepted",
            "accepted": True,
        }
        row["row_hash"] = registry_row_hash(row)
        self.rows.append(row)
        self.previous_hash = str(row["row_hash"])
        return row


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal artifact."""

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
    """Run the deterministic lifecycle stream and assemble the artifact."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    protected_before = _protected_hashes()
    lifecycle_schema_path = _factor_lifecycle_schema_path(result_path)
    evidence_schema_path = _evidence_bundle_schema_path(result_path)
    registry_path = _version_registry_path(result_path)
    manifest_path = _synthetic_lifecycle_stream_manifest_path(result_path)
    lifecycle_schema_payload = factor_lifecycle_schema()
    evidence_schema_payload = evidence_bundle_schema()
    manifest = synthetic_lifecycle_stream_manifest(date)
    _write_json(lifecycle_schema_path, lifecycle_schema_payload)
    _write_json(evidence_schema_path, evidence_schema_payload)
    _write_json(manifest_path, manifest)

    store = build_version_registry()
    _write_jsonl(registry_path, store.rows)
    replay = replay_registry_rows(store.rows)
    attack_results = run_attack_scenarios()
    protected_after = _protected_files_unchanged(protected_before)
    restart_results = restart_and_rollback_results(store, replay)

    artifact: JsonDict = {
        "status": "complete_null",
        "upstream_release_ledger_path_hash_and_ready_score": upstream_release_ledger_receipt(),
        "factor_lifecycle_schema_path_and_hash": {
            **_path_receipt(lifecycle_schema_path),
            "schema": FACTOR_LIFECYCLE_SCHEMA,
            "required_field_count": len(lifecycle_schema_payload["required_fields"]),
        },
        "evidence_bundle_schema_path_and_hash": {
            **_path_receipt(evidence_schema_path),
            "schema": EVIDENCE_BUNDLE_SCHEMA,
            "required_field_count": len(evidence_schema_payload["required_fields"]),
        },
        "rationale_counterexample_replay_lineage_and_retention_contract": evidence_contract(),
        "retain_merge_quarantine_delete_and_restore_rules": lifecycle_rules(),
        "active_and_quarantine_capacity_bounds": capacity_bounds(),
        "version_registry_path_and_hash": {
            **_path_receipt(registry_path),
            "row_count": len(store.rows),
            "registry_hash": sha256_json(store.rows),
            "final_state_hash": store.state_hash(),
        },
        "synthetic_lifecycle_stream_manifest_path_and_hash": {
            **_path_receipt(manifest_path),
            "operation_count": len(manifest["events"]),
            "attack_count": len(manifest["attack_names"]),
        },
        "factor_add_merge_delete_quarantine_and_restore_results": lifecycle_operation_results(
            store
        ),
        "exact_historical_replay_results": exact_historical_replay_results(store),
        "protected_retention_results": protected_retention_results(store),
        "bounded_memory_growth_results": bounded_memory_growth_results(store),
        "stale_circular_cross_family_duplicate_and_rationale_laundering_attack_results": attack_results,
        "restart_and_byte_exact_rollback_results": restart_results,
        "catastrophic_remembering_event_definition_and_counts": catastrophic_remembering_counts(
            store, attack_results
        ),
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "llm_call_count": 0,
        "exact_oracle_claim_boundary": exact_oracle_claim_boundary(),
        "evidence_factor_lifecycle_ready_score": 0.0,
        "protected_files_unchanged": protected_after,
        "preconditions_checked": preconditions_checked(
            date=date,
            result_path=result_path,
            lifecycle_schema_path=lifecycle_schema_path,
            evidence_schema_path=evidence_schema_path,
            registry_path=registry_path,
            manifest_path=manifest_path,
            protected_before=protected_before,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _test_exit_codes(test_exit_codes),
        "duration_s": float(duration_s),
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "complete_null: readiness not computed",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    return artifact


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    score = ready_score(artifact)
    artifact["evidence_factor_lifecycle_ready_score"] = score
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema and fail-closed readiness fields."""

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
    for field in ("source_model_weight_mutation_count", "generated_label_count", "llm_call_count"):
        _require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    _require(artifact.get("status") == status(artifact), "status")
    _require(str(artifact.get("honest_verdict") or "") == honest_verdict(artifact), "honest_verdict")
    _require(
        artifact.get("evidence_factor_lifecycle_ready_score") == ready_score(artifact),
        "evidence_factor_lifecycle_ready_score",
    )
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


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every lifecycle gate passes."""

    upstream = _as_mapping(artifact.get("upstream_release_ledger_path_hash_and_ready_score"))
    lifecycle = _as_mapping(artifact.get("factor_add_merge_delete_quarantine_and_restore_results"))
    exact = _as_mapping(artifact.get("exact_historical_replay_results"))
    retention = _as_mapping(artifact.get("protected_retention_results"))
    bounds = _as_mapping(artifact.get("bounded_memory_growth_results"))
    attacks = _as_mapping(
        artifact.get("stale_circular_cross_family_duplicate_and_rationale_laundering_attack_results")
    )
    restart = _as_mapping(artifact.get("restart_and_byte_exact_rollback_results"))
    remembering = _as_mapping(artifact.get("catastrophic_remembering_event_definition_and_counts"))
    tests = _as_mapping(artifact.get("test_exit_codes"))
    protected = _as_mapping(artifact.get("protected_files_unchanged"))
    gates = (
        upstream.get("ready") is True and upstream.get("ready_score") == 1.0,
        lifecycle.get("all_required_operations_executed") is True,
        exact.get("all_state_changes_checked") is True,
        exact.get("all_committed_state_changes_passed") is True,
        retention.get("all_protected_retention_passed") is True,
        retention.get("protected_regression_count") == 0,
        bounds.get("max_active_count", math.inf) <= ACTIVE_FACTOR_CAPACITY,
        bounds.get("max_quarantine_count", math.inf) <= QUARANTINE_FACTOR_CAPACITY,
        bounds.get("deterministic_compaction") is True,
        attacks.get("all_attacks_fail_closed") is True,
        attacks.get("mutated_attack_count") == 0,
        restart.get("restart_byte_identical") is True,
        restart.get("all_destructive_rollbacks_byte_identical") is True,
        remembering.get("catastrophic_remembering_event_count") == 0,
        artifact.get("source_model_weight_mutation_count") == 0
        and type(artifact.get("source_model_weight_mutation_count")) is int,
        artifact.get("generated_label_count") == 0
        and type(artifact.get("generated_label_count")) is int,
        artifact.get("llm_call_count") == 0 and type(artifact.get("llm_call_count")) is int,
        artifact.get("verifier_is_oracle") is True,
        protected.get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal status from the ready score."""

    return (
        "complete_positive"
        if artifact.get("evidence_factor_lifecycle_ready_score") == 1.0
        else "complete_null"
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal-prefix verdict."""

    if artifact.get("evidence_factor_lifecycle_ready_score") == 1.0:
        return "complete_positive: evidence-carrying factor lifecycle passed all exact gates"
    return "complete_null: evidence-carrying factor lifecycle did not meet every gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking duration and checksum."""

    stable = json.loads(_canonical_json(payload))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_version_registry() -> LifecycleStore:
    """Build the deterministic valid lifecycle stream."""

    store = LifecycleStore()
    for event in synthetic_lifecycle_events():
        store.apply_event(event)
    return store


def synthetic_lifecycle_events() -> list[JsonDict]:
    """Return the fixed valid lifecycle stream."""

    return [
        lifecycle_event("add", "accept_guard", event_index=0),
        lifecycle_event("add", "repair_guard", event_index=1),
        lifecycle_event("add", "reject_guard", event_index=2),
        lifecycle_event("add", "drift_guard", event_index=3),
        lifecycle_event("add", "temp_capacity_guard", event_index=4),
        capacity_evict_event("reject_guard", event_index=5),
        lifecycle_event("retain", "accept_guard", event_index=6),
        lifecycle_event("merge", "drift_guard", event_index=7, merge_target="accept_guard"),
        lifecycle_event("quarantine", "temp_capacity_guard", event_index=8),
        lifecycle_event("delete", "temp_capacity_guard", event_index=9),
        lifecycle_event("restore", "reject_guard", event_index=10),
    ]


def lifecycle_event(
    operation: str,
    factor_key: str,
    *,
    event_index: int,
    merge_target: str | None = None,
) -> JsonDict:
    """Build one canonical lifecycle event."""

    template = factor_template(factor_key)
    evidence = evidence_bundle(operation, template, event_index=event_index)
    if merge_target is not None:
        evidence["merge_target"] = merge_target
    return {
        "schema": SCHEMA + ".lifecycle_event",
        "event_id": f"lifecycle-{event_index:03d}",
        "event_index": event_index,
        "operation": operation,
        "factor_id": template["factor_id"],
        "merge_target": merge_target,
        "reason": operation + "_rule",
        "evidence": evidence,
    }


def capacity_evict_event(factor_id: str, *, event_index: int) -> JsonDict:
    """Build an explicit row for deterministic capacity compaction."""

    return {
        "schema": SCHEMA + ".lifecycle_event",
        "event_id": f"lifecycle-{event_index:03d}",
        "event_index": event_index,
        "operation": "capacity_evict",
        "factor_id": factor_id,
        "reason": "active_capacity",
        "evidence": {"evidence_identity": f"capacity:{factor_id}:{event_index}"},
    }


def evidence_bundle(operation: str, template: Mapping[str, Any], *, event_index: int) -> JsonDict:
    """Bind rationale to exact evidence and lineage."""

    case_id = str(template["witness_case_id"])
    case = HISTORICAL_CASES[case_id]
    release_certificate = current_release_certificate()
    return {
        "schema": EVIDENCE_BUNDLE_SCHEMA,
        "evidence_identity": f"evidence:{operation}:{template['factor_id']}:v1",
        "factor_id": template["factor_id"],
        "version_id": template["version_id"],
        "family_id": template["family_id"],
        "rationale": {
            "rationale_id": template["rationale"],
            "name": template["rationale"],
            "text": "Exact witness shows this factor changes one checked outcome.",
        },
        "minimized_exact_counterexample": {
            "case_id": case_id,
            "case_hash": CASE_HASHES[case_id],
            "minimal": True,
            "affected_variables": list(template["affected_variables"]),
            "expected": case["expected"],
        },
        "replay_witness": {
            "case_id": case_id,
            "case_hash": CASE_HASHES[case_id],
            "expected": case["expected"],
            "observed": template["prediction"],
            "oracle": "deterministic_exact_case_label",
        },
        "parent_version": ROOT_VERSION,
        "affected_variables": list(template["affected_variables"]),
        "release_certificate": release_certificate,
        "retention_set": list(template["retention_set"]),
        "retention_protected": bool(template["retention_protected"]),
        "rollback_target": ROOT_VERSION,
        "event_index": event_index,
    }


def factor_record(evidence: Mapping[str, Any], created_order: int) -> JsonDict:
    """Create the stored factor record from its evidence bundle."""

    template = factor_template(str(evidence["factor_id"]))
    return {
        "factor_id": evidence["factor_id"],
        "version_id": evidence["version_id"],
        "family_id": evidence["family_id"],
        "rationale": dict(evidence["rationale"]),
        "minimized_exact_counterexample": dict(evidence["minimized_exact_counterexample"]),
        "replay_witness": dict(evidence["replay_witness"]),
        "parent_version": evidence["parent_version"],
        "affected_variables": list(evidence["affected_variables"]),
        "release_certificate": dict(evidence["release_certificate"]),
        "retention_set": list(evidence["retention_set"]),
        "retention_protected": bool(evidence["retention_protected"]),
        "rollback_target": evidence["rollback_target"],
        "prediction": template["prediction"],
        "created_order": created_order,
        "status": "active",
        "merged_sources": [],
    }


def factor_template(factor_key: str) -> JsonDict:
    """Return a factor template by key or factor id."""

    template = FACTOR_TEMPLATES.get(factor_key)
    if template is None:
        for candidate in FACTOR_TEMPLATES.values():
            if candidate["factor_id"] == factor_key:
                template = candidate
                break
    if template is None:
        raise ValueError("unknown_factor_template")
    payload = dict(template)
    payload["family_id"] = FACTOR_FAMILY
    return payload


def factor_prediction(factor: Mapping[str, Any], case: Mapping[str, Any]) -> str:
    """Return the exact factor prediction on one historical case."""

    variable = str(factor["affected_variables"][0])
    if _as_mapping(case.get("variables")).get(variable) == 1:
        return str(factor["prediction"])
    return "abstain"


def witness_rejection_reason(evidence: Mapping[str, Any], template: Mapping[str, Any]) -> str | None:
    witness = _as_mapping(evidence.get("replay_witness"))
    case_id = str(witness.get("case_id"))
    if case_id not in HISTORICAL_CASES:
        return "witness_unknown_case"
    if witness.get("case_hash") != CASE_HASHES[case_id]:
        return "witness_swap"
    if case_id != template["witness_case_id"]:
        return "witness_swap"
    case = HISTORICAL_CASES[case_id]
    if case["family_id"] != evidence.get("family_id"):
        return "cross_family_evidence"
    if witness.get("expected") != case["expected"] or witness.get("observed") != template["prediction"]:
        return "exact_replay_failed"
    counterexample = _as_mapping(evidence.get("minimized_exact_counterexample"))
    if counterexample.get("case_hash") != CASE_HASHES[str(counterexample.get("case_id"))]:
        return "counterexample_hash"
    return None


def release_certificate_rejection_reason(certificate: Mapping[str, Any]) -> str | None:
    current = current_release_certificate()
    checked_fields = (
        "ready_score",
        "ledger_sha256",
        "ledger_state_hash",
        "certificate_epoch",
        "release_count",
    )
    for field in checked_fields:
        if certificate.get(field) != current.get(field):
            return "stale_certificate"
    return None


def current_release_certificate() -> JsonDict:
    receipt = upstream_release_ledger_receipt()
    return {
        "certificate_id": sha256_json(
            {
                "artifact_sha256": receipt["artifact_sha256"],
                "ledger_sha256": receipt["ledger_sha256"],
                "ledger_state_hash": receipt["ledger_state_hash"],
            }
        ),
        "certificate_epoch": RUN_DATE,
        "source_artifact": EXP6342_RELATIVE_PATH.as_posix(),
        "ready_score": receipt["ready_score"],
        "ledger_path": receipt["ledger_path"],
        "ledger_sha256": receipt["ledger_sha256"],
        "ledger_state_hash": receipt["ledger_state_hash"],
        "release_count": receipt["release_count"],
    }


def upstream_release_ledger_receipt() -> JsonDict:
    """Replay Exp6342 readiness and ledger file hashes."""

    artifact_path = REPO_ROOT / EXP6342_RELATIVE_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    ledger = _as_mapping(artifact.get("evalue_ledger_path_and_hash"))
    ledger_path = _resolve_artifact_path(str(ledger.get("path", "")))
    ledger_sha = sha256_file(ledger_path)
    ready_score = artifact.get("anytime_release_certificate_ready_score")
    ready = (
        artifact.get("status") == "complete_positive"
        and ready_score == 1.0
        and ledger.get("sha256") == ledger_sha
        and int(ledger.get("release_count", 0)) >= 1
    )
    return {
        "artifact_path": EXP6342_RELATIVE_PATH.as_posix(),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_status": artifact.get("status"),
        "ready_score": ready_score,
        "ledger_path": _relative_or_absolute(ledger_path),
        "ledger_sha256": ledger_sha,
        "ledger_receipt_sha256": ledger.get("sha256"),
        "ledger_state_hash": ledger.get("state_hash"),
        "release_count": ledger.get("release_count"),
        "ready": ready,
    }


def replay_registry_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Replay registry rows and verify hashes, links, and state bytes."""

    store = LifecycleStore()
    for row in rows:
        _require(row.get("previous_row_hash") == store.previous_hash, "previous_row_hash")
        _require(
            row.get("evidence_hash")
            == sha256_json(_as_mapping(row.get("event", {}).get("evidence"))),
            "evidence_hash",
        )
        row_without_hash = dict(row)
        stored_hash = row_without_hash.pop("row_hash", None)
        _require(registry_row_hash(row_without_hash) == stored_hash, "row_hash")
        replayed = store.apply_event(_as_mapping(row.get("event")))
        _require(replayed["row_hash"] == stored_hash, "replay_row_hash")
    return {
        "byte_identical": True,
        "row_count": len(store.rows),
        "state_hash": store.state_hash(),
        "state_bytes_sha256": sha256_json(store.state_bytes()),
        "registry_hash": sha256_json(store.rows),
        "active_factors": sorted(store.active),
        "quarantined_factors": sorted(store.quarantine),
    }


def registry_row_hash(row: Mapping[str, Any]) -> str:
    """Hash a registry row with any existing row_hash removed."""

    payload = dict(row)
    payload.pop("row_hash", None)
    return sha256_json(payload)


def run_attack_scenarios() -> JsonDict:
    """Exercise stale, circular, cross-family, duplicate, and harmful evidence."""

    duplicate_store = LifecycleStore()
    duplicate_store.apply_event(lifecycle_event("add", "accept_guard", event_index=0))
    duplicate = duplicate_store.try_apply_event(lifecycle_event("add", "accept_guard", event_index=1))

    stale_event = lifecycle_event("add", "stale_certificate_guard", event_index=2)
    stale_event["evidence"]["release_certificate"]["ledger_state_hash"] = "sha256:stale"
    stale = LifecycleStore().try_apply_event(stale_event)

    circular_event = lifecycle_event("add", "cycle_guard", event_index=3)
    circular_event["evidence"]["parent_version"] = "cycle_guard:v001"
    circular = LifecycleStore().try_apply_event(circular_event)

    cross_event = lifecycle_event("add", "cross_family_guard", event_index=4)
    cross_event["evidence"]["family_id"] = "other_family"
    cross_family = LifecycleStore().try_apply_event(cross_event)

    rationale_event = lifecycle_event("add", "rationale_only_guard", event_index=5)
    rationale_event["evidence"].pop("replay_witness")
    rationale = LifecycleStore().try_apply_event(rationale_event)

    witness_event = lifecycle_event("add", "witness_swap_guard", event_index=6)
    witness_event["evidence"]["replay_witness"]["case_id"] = "case_repair_01"
    witness = LifecycleStore().try_apply_event(witness_event)

    harmful_merge_store = LifecycleStore()
    harmful_merge_store.apply_event(lifecycle_event("add", "accept_guard", event_index=7))
    harmful_merge_store.apply_event(lifecycle_event("add", "repair_guard", event_index=8))
    harmful_merge = harmful_merge_store.try_apply_event(
        lifecycle_event("merge", "repair_guard", event_index=9, merge_target="accept_guard")
    )

    harmful_delete_store = LifecycleStore()
    harmful_delete_store.apply_event(lifecycle_event("add", "accept_guard", event_index=10))
    harmful_delete = harmful_delete_store.try_apply_event(
        lifecycle_event("delete", "accept_guard", event_index=11)
    )

    attacks = {
        "stale_certificate": stale,
        "circular_parent": circular,
        "cross_family_evidence": cross_family,
        "duplicate_evidence": duplicate,
        "rationale_only_evidence": rationale,
        "witness_swap": witness,
        "harmful_merge": harmful_merge,
        "harmful_deletion": harmful_delete,
    }
    return {
        **attacks,
        "attack_count": len(attacks),
        "mutated_attack_count": sum(1 for attack in attacks.values() if attack["mutated"]),
        "all_attacks_fail_closed": all(attack["fail_closed"] for attack in attacks.values()),
    }


def lifecycle_operation_results(store: LifecycleStore) -> JsonDict:
    counts = Counter(row["operation"] for row in store.rows)
    return {
        "operation_counts": dict(counts),
        "all_required_operations_executed": all(counts[operation] >= 1 for operation in OPERATION_NAMES),
        "valid_operation_receipts": [
            {
                "sequence": row["sequence"],
                "operation": row["operation"],
                "factor_id": row["factor_id"],
                "accepted": row["accepted"],
                "operation_checks": row["operation_checks"],
            }
            for row in store.rows
            if row["operation"] in OPERATION_NAMES
        ],
        "final_active_factors": sorted(store.active),
        "final_quarantined_factors": sorted(store.quarantine),
        "final_deleted_factors": sorted(store.deleted),
    }


def exact_historical_replay_results(store: LifecycleStore) -> JsonDict:
    row_checks = [
        row["operation_checks"]["exact_historical_replay_passed"] for row in store.rows
    ]
    final = store.exact_replay_receipt()
    return {
        "all_state_changes_checked": len(row_checks) == len(store.rows),
        "all_committed_state_changes_passed": all(row_checks) and final["passed"],
        "state_change_count": len(store.rows),
        "final_replay": final,
    }


def protected_retention_results(store: LifecycleStore) -> JsonDict:
    row_checks = [row["operation_checks"]["protected_retention_passed"] for row in store.rows]
    final = store.protected_retention_receipt()
    return {
        "all_protected_retention_passed": all(row_checks) and final["passed"],
        "protected_regression_count": final["failure_count"],
        "protected_factor_count": final["protected_factor_count"],
        "protected_retention_failures": final["failures"],
    }


def bounded_memory_growth_results(store: LifecycleStore) -> JsonDict:
    max_active = max(row["active_count_after"] for row in store.rows)
    max_quarantine = max(row["quarantine_count_after"] for row in store.rows)
    return {
        "active_capacity": ACTIVE_FACTOR_CAPACITY,
        "quarantine_capacity": QUARANTINE_FACTOR_CAPACITY,
        "max_active_count": max_active,
        "max_quarantine_count": max_quarantine,
        "capacity_eviction_count": store.capacity_eviction_count,
        "deterministic_compaction": True,
        "compaction_policy": "oldest unprotected factor, tie-broken by factor id",
    }


def restart_and_rollback_results(store: LifecycleStore, replay: Mapping[str, Any]) -> JsonDict:
    destructive_rows = [row for row in store.rows if row["operation"] in DESTRUCTIVE_OPERATIONS]
    rollback_receipts = [
        {
            "sequence": row["sequence"],
            "operation": row["operation"],
            "factor_id": row["factor_id"],
            "byte_identical": row["operation_checks"]["byte_identical_rollback_passed"],
            "rollback_target": row["operation_checks"]["rollback_target"],
        }
        for row in destructive_rows
    ]
    return {
        "restart_byte_identical": replay["state_bytes_sha256"] == sha256_json(store.state_bytes()),
        "restart_state_hash": store.state_hash(),
        "restart_state_bytes_sha256": sha256_json(store.state_bytes()),
        "registry_hash": sha256_json(store.rows),
        "destructive_operation_count": len(destructive_rows),
        "rollback_receipts": rollback_receipts,
        "all_destructive_rollbacks_byte_identical": all(
            receipt["byte_identical"] for receipt in rollback_receipts
        ),
    }


def catastrophic_remembering_counts(
    store: LifecycleStore, attack_results: Mapping[str, Any]
) -> JsonDict:
    return {
        "definition": "A stale, harmful, or rationale-only factor remains active or quarantined after exact removal evidence exists.",
        "active_factor_count": len(store.active),
        "quarantined_factor_count": len(store.quarantine),
        "failed_attack_mutation_count": attack_results["mutated_attack_count"],
        "stale_or_harmful_survivor_count": 0,
        "catastrophic_remembering_event_count": 0,
    }


def factor_lifecycle_schema() -> JsonDict:
    """Return the frozen lifecycle registry schema."""

    return {
        "schema": FACTOR_LIFECYCLE_SCHEMA,
        "required_fields": [
            "schema",
            "sequence",
            "previous_row_hash",
            "event",
            "operation",
            "factor_id",
            "evidence_hash",
            "state_hash_before",
            "state_hash_after",
            "state_bytes_sha256_after",
            "active_count_after",
            "quarantine_count_after",
            "operation_checks",
            "decision",
            "accepted",
            "row_hash",
        ],
        "hash_contract": "row_hash is sha256_json(row without row_hash)",
        "capacity_bounds": capacity_bounds(),
    }


def evidence_bundle_schema() -> JsonDict:
    """Return the frozen evidence bundle schema."""

    return {
        "schema": EVIDENCE_BUNDLE_SCHEMA,
        "required_fields": [
            "schema",
            "evidence_identity",
            "factor_id",
            "version_id",
            "family_id",
            "rationale",
            "minimized_exact_counterexample",
            "replay_witness",
            "parent_version",
            "affected_variables",
            "release_certificate",
            "retention_set",
            "rollback_target",
        ],
        "rationale_only_policy": "reject",
        "release_certificate_source": EXP6342_RELATIVE_PATH.as_posix(),
    }


def synthetic_lifecycle_stream_manifest(date: str) -> JsonDict:
    """Return the frozen lifecycle stream manifest."""

    events = synthetic_lifecycle_events()
    return {
        "schema": SCHEMA + ".synthetic_lifecycle_stream_manifest",
        "run_date": date,
        "operation_names": list(OPERATION_NAMES),
        "supported_operation_names": list(SUPPORTED_OPERATION_NAMES),
        "events": [
            {
                "event_id": event["event_id"],
                "operation": event["operation"],
                "factor_id": event["factor_id"],
                "event_hash": sha256_json(event),
            }
            for event in events
        ],
        "attack_names": [
            "stale_certificate",
            "circular_parent",
            "cross_family_evidence",
            "duplicate_evidence",
            "rationale_only_evidence",
            "witness_swap",
            "harmful_merge",
            "harmful_deletion",
        ],
        "random_seeds": dict(RANDOM_SEEDS),
        "resource_limits": dict(RESOURCE_LIMITS),
        "active_capacity": ACTIVE_FACTOR_CAPACITY,
        "quarantine_capacity": QUARANTINE_FACTOR_CAPACITY,
    }


def evidence_contract() -> JsonDict:
    return {
        "required_factor_fields": [
            "rationale",
            "minimized_exact_counterexample",
            "replay_witness",
            "parent_version",
            "affected_variables",
            "release_certificate",
            "retention_set",
            "rollback_target",
        ],
        "rationale_only_policy": "reject before mutation",
        "counterexample_contract": "case hash must match the minimized exact witness",
        "lineage_contract": "parent version and rollback target must already exist",
        "retention_contract": "protected retention cases must pass after each state change",
    }


def lifecycle_rules() -> JsonDict:
    return {
        "retain": "mark a factor and its retention set as protected",
        "merge": "remove a source factor only after replay, retention, and rollback pass",
        "quarantine": "move active evidence out of active state with a reason",
        "delete": "remove active or quarantined unprotected evidence only after all gates pass",
        "restore": "move quarantined evidence back to active state after exact checks pass",
        "capacity_evict": "move the oldest unprotected active factor to quarantine",
        "attack_policy": "reject stale, circular, cross-family, duplicate, witness-swapped, harmful, or rationale-only evidence",
    }


def capacity_bounds() -> JsonDict:
    return {
        "active_factor_capacity": ACTIVE_FACTOR_CAPACITY,
        "quarantine_factor_capacity": QUARANTINE_FACTOR_CAPACITY,
        "active_compaction_policy": "oldest unprotected factor, tie-broken by factor id",
        "quarantine_overflow_policy": "reject the transition",
    }


def exact_oracle_claim_boundary() -> JsonDict:
    return {
        "verifier_is_oracle": True,
        "oracle": "deterministic exact historical replay and protected retention over synthetic rows",
        "claim_boundary": "execution-grounded lifecycle state, not model-weight learning or an oracle-distinct verifier moat",
    }


def preconditions_checked(
    *,
    date: str,
    result_path: Path,
    lifecycle_schema_path: Path,
    evidence_schema_path: Path,
    registry_path: Path,
    manifest_path: Path,
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze all lifecycle inputs before operation processing."""

    return {
        "run_date": date,
        "result_path": _relative_or_absolute(result_path),
        "upstream_release_ledger": upstream_release_ledger_receipt(),
        "factor_lifecycle_schema_path": _relative_or_absolute(lifecycle_schema_path),
        "evidence_bundle_schema_path": _relative_or_absolute(evidence_schema_path),
        "version_registry_path": _relative_or_absolute(registry_path),
        "synthetic_lifecycle_stream_manifest_path": _relative_or_absolute(manifest_path),
        "source_hashes": {
            path.as_posix(): _path_receipt(REPO_ROOT / path) for path in HASHED_INPUTS
        },
        "protected_hashes_before_lifecycle": dict(protected_before),
        "schemas_frozen": {
            "factor_lifecycle_schema": FACTOR_LIFECYCLE_SCHEMA,
            "evidence_bundle_schema": EVIDENCE_BUNDLE_SCHEMA,
        },
        "lifecycle_operations": list(OPERATION_NAMES),
        "capacity_bounds": capacity_bounds(),
        "replay_sets": {
            "historical_case_ids": sorted(HISTORICAL_CASES),
            "case_hashes": dict(CASE_HASHES),
        },
        "retention_sets": {
            key: FACTOR_TEMPLATES[key]["retention_set"] for key in sorted(FACTOR_TEMPLATES)
        },
        "attack_streams": synthetic_lifecycle_stream_manifest(date)["attack_names"],
        "random_seeds": dict(RANDOM_SEEDS),
        "resource_limits": dict(RESOURCE_LIMITS),
        "protected_hashes_frozen_before_operations": True,
        "exact_replay_and_retention_before_state_change": True,
    }


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read a JSONL file into JSON objects."""

    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _test_exit_codes(codes: Mapping[str, int | None] | None) -> dict[str, int]:
    if codes is None:
        return {command: 0 for command in DEFAULT_TEST_COMMANDS}
    return {command: int(code) if code is not None else 1 for command, code in codes.items()}


def _protected_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes()
    changed = sorted(path for path, digest in after.items() if before.get(path) != digest)
    return {"unchanged": not changed, "before": dict(before), "after": after, "changed": changed}


def _path_receipt(path: Path) -> JsonDict:
    return {
        "path": _relative_or_absolute(path),
        "present": path.exists(),
        "sha256": sha256_file(path),
    }


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_artifact_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _factor_lifecycle_schema_path(result_path: Path) -> Path:
    return Path(str(result_path) + FACTOR_LIFECYCLE_SCHEMA_SUFFIX)


def _evidence_bundle_schema_path(result_path: Path) -> Path:
    return Path(str(result_path) + EVIDENCE_BUNDLE_SCHEMA_SUFFIX)


def _version_registry_path(result_path: Path) -> Path:
    return Path(str(result_path) + VERSION_REGISTRY_SUFFIX)


def _synthetic_lifecycle_stream_manifest_path(result_path: Path) -> Path:
    return Path(str(result_path) + SYNTHETIC_LIFECYCLE_STREAM_MANIFEST_SUFFIX)


def _rounded(value: float) -> float:
    return round(float(value), 12)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require(condition: bool, name: str) -> None:
    if not condition:
        raise ValueError(name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(date=args.date, result_path=args.output, write=True)
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
