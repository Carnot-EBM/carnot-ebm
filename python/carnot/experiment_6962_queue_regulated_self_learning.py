"""Run prospective exact-certificate memory with a virtual debt queue.

Spec refs: REQ-LEARN-6962 and SCENARIO-LEARN-6962-*.

The language models only propose relations. An external exact outcome decides
whether a proposal can enter memory. Model confidence and rationale are stored
for audit only. They never control admission or retrieval.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.inference.sota_models import gguf_tokenizer_loadable, resolve_cached_gguf
from carnot.task_runtime_receipts import write_json_atomic


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = Path("results/experiment_6961_certified_event_sequence.json")
RESULT_PATH = Path("results/experiment_6962_queue_regulated_self_learning.json")
CHECKPOINT_ROOT = Path("results/checkpoints/experiment_6962_queue_regulated_self_learning")
INFERENCE_SUBSTRATE = "prospective_local_gguf_exact_certificate_debt_queue_memory"
SCHEMA_VERSION = "carnot.exp6962.queue_regulated_self_learning.v1"
STORE_SCHEMA_VERSION = "carnot.exp6962.transactional_certificate_store.v1"
CHECKPOINT_SCHEMA_VERSION = "carnot.exp6962.event_checkpoint.v1"
EXACT_AUTHORITY = "external_exact_outcome_certifier"
RANDOM_SEED = 696220260904
ARMS = ("no_memory", "fifo", "debt_queue")
MODEL_SPECS = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "gpu": 0,
        "headline_eligible": True,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "gpu": 1,
        "headline_eligible": True,
    },
)


@dataclass(frozen=True)
class FrozenPolicy:
    """Keep all arm controls equal and immutable for the full stream."""

    capacity: int = 6
    retrieval_count: int = 3
    debt_threshold: int = 3
    debt_arrival_weight: int = 1
    service_per_clean_success: int = 1
    drift_plus_penalty_coefficient: float = 0.5
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = RANDOM_SEED


FROZEN_POLICY = FrozenPolicy()

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "model_specs",
    "model_rows",
    "arm_rows",
    "event_rows",
    "chronology_rows",
    "prompt_rows",
    "raw_output_rows",
    "exact_outcome_rows",
    "retrieval_rows",
    "write_rows",
    "admission_rows",
    "fifo_rows",
    "debt_arrival_rows",
    "debt_service_rows",
    "debt_balance_rows",
    "memory_hash_rows",
    "token_budget_rows",
    "latency_rows",
    "poison_rows",
    "contradiction_rows",
    "retention_rows",
    "restart_rows",
    "tombstone_rows",
    "rollback_rows",
    "future_label_isolation_rows",
    "paired_metric_rows",
    "confidence_interval_rows",
    "checkpoint_rows",
    "model_lifecycle_rows",
    "task_runtime_receipt",
    "continuous_self_learning_task",
    "learning_tier",
    "no_model_weight_mutation",
    "model_hashes_before",
    "model_hashes_after",
    "random_seed",
    "reproducibility_checksum",
    "queue_learning_run_complete_score",
    "queue_learning_positive_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason for each field makes the evidence contract auditable.",
    "preconditions_checked": "Fail-closed checks prevent fabricated inference evidence.",
    "inference_substrate": "The substrate declaration separates live generation from reduction.",
    "duration_s": "Measured wall time makes implausibly short model runs visible.",
    "source_artifact_hashes": "Source hashes bind this result to the sealed event sequence.",
    "rows": "Authoritative rows let an auditor recompute every headline.",
    "model_specs": "Exact local model identities prevent silent legacy-model substitution.",
    "model_rows": "Per-model evidence prevents pooled results from hiding one weak model.",
    "arm_rows": "Per-arm summaries preserve matched comparator outcomes.",
    "event_rows": "One row per call preserves the chronological unit of evidence.",
    "chronology_rows": "Ordinals prove that retrieval used only past certificates.",
    "prompt_rows": "Prompt hashes expose any arm-specific information advantage.",
    "raw_output_rows": "Durable raw bytes preserve model behavior before certification.",
    "exact_outcome_rows": "External outcomes, not self-reports, authorize writes.",
    "retrieval_rows": "Retrieved identifiers expose causal memory use.",
    "write_rows": "Write timing proves that no outcome entered memory early.",
    "admission_rows": "Admission reasons expose the exact authority used.",
    "fifo_rows": "FIFO rows make fixed-age eviction reproducible.",
    "debt_arrival_rows": "Causal arrivals expose each harmful memory use.",
    "debt_service_rows": "Service rows prevent hidden debt forgiveness.",
    "debt_balance_rows": "Balances prove the nonnegative virtual-queue recurrence.",
    "memory_hash_rows": "Store hashes bind each transaction to exact bytes.",
    "token_budget_rows": "Token rows prove equal generation budgets across arms.",
    "latency_rows": "Latency rows preserve the cost of each policy.",
    "poison_rows": "Poison tests prove delayed copies cannot become authority.",
    "contradiction_rows": "Contradiction tests connect harmful retrieval to debt.",
    "retention_rows": "Retention tests stop utility gains from hiding forgetting.",
    "restart_rows": "Fresh restore evidence detects volatile or altered state.",
    "tombstone_rows": "Tombstones prevent removed certificates from returning.",
    "rollback_rows": "Byte-exact rollback protects the last valid parent state.",
    "future_label_isolation_rows": "Leakage checks keep future outcomes out of decisions.",
    "paired_metric_rows": "Paired effects compare the same events under each policy.",
    "confidence_interval_rows": "Intervals distinguish repeatable gains from noise.",
    "checkpoint_rows": "Per-event checkpoints permit idempotent recovery.",
    "model_lifecycle_rows": "Lifecycle evidence binds processes, GPUs, and frozen weights.",
    "task_runtime_receipt": "Runtime receipts connect outputs to the process that made them.",
    "continuous_self_learning_task": "The flag classifies this as persistent online learning.",
    "learning_tier": "Tier 2 identifies certificate memory without weight training.",
    "no_model_weight_mutation": "Frozen weights isolate memory as the only learning mechanism.",
    "model_hashes_before": "Before hashes establish the immutable model baseline.",
    "model_hashes_after": "After hashes detect any model-file mutation.",
    "random_seed": "A fixed seed makes decoding and ordering reproducible.",
    "reproducibility_checksum": "A content digest detects later scientific-data drift.",
    "queue_learning_run_complete_score": "Completeness requires every call and safety case.",
    "queue_learning_positive_score": "The positive gate requires utility and safety together.",
    "gate_check_summary": "Expected and observed values make failures actionable.",
    "verifier_is_oracle": "False prevents exact outcome checks from becoming a circular moat claim.",
    "verdict_class": "A closed verdict class prevents ambiguous automation states.",
    "honest_verdict": "A matching terminal prefix preserves the scientific conclusion.",
}

_DEBT_CAUSES = {"exact_failure", "contradiction", "stale_use", "retention_regression"}
_VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
_FORBIDDEN_VISIBLE_KEYS = {
    "exact_outcome",
    "exact_success",
    "sealed_answer_mapping",
    "answer_mapping_hash",
    "answer_mapping_signature",
}


class TransactionError(RuntimeError):
    """Report a transaction that did not replace its valid parent bytes."""


def canonical_json(value: Any) -> str:
    """Return stable JSON so every hash has one spelling."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash text with the repository's explicit digest prefix."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value independently of source formatting."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Stream a file hash without loading a multi-gigabyte model into RAM."""

    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record both sides of one fail-closed comparison."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": expected == observed,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all checks and surface the first failed comparison."""

    copied = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in copied if row.get("passed") is not True), None)
    return {
        "checks": copied,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def empty_store_state(capacity: int = FROZEN_POLICY.capacity) -> JsonDict:
    """Build the complete empty state used at every arm boundary."""

    return {
        "schema_version": STORE_SCHEMA_VERSION,
        "capacity": int(capacity),
        "active": [],
        "tombstones": {},
        "debt_balance": 0,
        "charged_pairs": [],
        "revision": 0,
    }


def certificate_hash(certificate: Mapping[str, Any]) -> str:
    """Bind exact evidence while excluding its stored self-hash."""

    payload = dict(certificate)
    payload.pop("certificate_hash", None)
    return sha256_json(payload)


def certificate_is_valid(certificate: Mapping[str, Any]) -> bool:
    """Accept only intact external exact-success certificates."""

    return bool(
        certificate.get("authority") == EXACT_AUTHORITY
        and certificate.get("exact_success") is True
        and certificate.get("certificate_hash") == certificate_hash(certificate)
        and str(certificate.get("raw_output_hash", "")).startswith("sha256:")
        and str(certificate.get("prompt_hash", "")).startswith("sha256:")
    )


class TransactionalMemoryStore:
    """Persist private certificate state through atomic file replacement."""

    def __init__(self, path: str | Path, *, capacity: int = FROZEN_POLICY.capacity) -> None:
        self.path = Path(path)
        self.capacity = int(capacity)
        if self.path.is_file():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = empty_store_state(self.capacity)
            self._write(self.state)

    def _write(self, state: Mapping[str, Any]) -> None:
        """Publish one complete state so interruption cannot expose partial JSON."""

        write_json_atomic(self.path, dict(state))

    def apply_transaction(
        self,
        mutation: Callable[[JsonDict], Any],
        *,
        fail_before_commit: bool = False,
    ) -> Any:
        """Apply one mutation or retain the exact parent bytes on failure."""

        parent_bytes = self.path.read_bytes()
        candidate = deepcopy(self.state)
        try:
            result = mutation(candidate)
            if fail_before_commit:
                raise TransactionError("injected failure before atomic replacement")
            candidate["revision"] = int(candidate.get("revision", 0)) + 1
            self._write(candidate)
        except Exception:
            self.path.write_bytes(parent_bytes)
            self.state = json.loads(parent_bytes)
            raise
        self.state = candidate
        return result

    def admit(self, certificate: Mapping[str, Any]) -> str | None:
        """Admit one exact certificate and tombstone any fixed-age eviction."""

        if not certificate_is_valid(certificate):
            raise ValueError("certificate is not externally exact and intact")

        def mutation(state: JsonDict) -> str | None:
            active = state["active"]
            certificate_id = str(certificate["certificate_id"])
            if certificate_id in state["tombstones"] or any(
                row["certificate_id"] == certificate_id for row in active
            ):
                return None
            active.append(deepcopy(dict(certificate)))
            evicted = None
            if len(active) > self.capacity:
                removed = active.pop(0)
                evicted = str(removed["certificate_id"])
                state["tombstones"][evicted] = {
                    "reason": "fifo_capacity_eviction",
                    "certificate_hash": removed["certificate_hash"],
                }
            return evicted

        return self.apply_transaction(mutation)

    def tombstone(self, certificate_id: str, reason: str) -> bool:
        """Remove one active record and retain a durable non-resurrection marker."""

        def mutation(state: JsonDict) -> bool:
            found = next(
                (row for row in state["active"] if row["certificate_id"] == certificate_id),
                None,
            )
            state["active"] = [
                row for row in state["active"] if row["certificate_id"] != certificate_id
            ]
            if found is not None:
                state["tombstones"][certificate_id] = {
                    "reason": reason,
                    "certificate_hash": found["certificate_hash"],
                }
            return found is not None

        return bool(self.apply_transaction(mutation))

    def set_debt(self, transition: Mapping[str, Any]) -> None:
        """Commit debt and idempotency keys in the same state transaction."""

        def mutation(state: JsonDict) -> None:
            state["debt_balance"] = int(transition["balance"])
            state["charged_pairs"] = sorted(set(transition["charged_pairs_after"]))

        self.apply_transaction(mutation)

    def active_ids(self) -> list[str]:
        """Return active identifiers in their fixed admission order."""

        return [str(row["certificate_id"]) for row in self.state["active"]]

    def store_hash(self) -> str:
        """Hash the exact durable bytes rather than an in-memory approximation."""

        return "sha256:" + hashlib.sha256(self.path.read_bytes()).hexdigest()

    def rollback(self, parent_bytes: bytes) -> None:
        """Restore a caller-provided valid parent byte-for-byte."""

        self.path.write_bytes(parent_bytes)
        self.state = json.loads(parent_bytes)

    def hard_reset(self) -> None:
        """Create a new empty arm state without retaining prior process data."""

        self.state = empty_store_state(self.capacity)
        self._write(self.state)


class EventCheckpoint:
    """Store completed model-arm-event rows for idempotent recovery."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_file():
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.payload = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "completed": {}}

    def record(self, key: str, row: Mapping[str, Any]) -> bool:
        """Record a missing key once so resume never repeats an external call."""

        if key in self.payload["completed"]:
            return False
        self.payload["completed"][key] = deepcopy(dict(row))
        write_json_atomic(self.path, self.payload)
        return True

    def completed_keys(self) -> set[str]:
        """Return all keys that already have durable terminal rows."""

        return set(self.payload["completed"])

    def rows(self) -> list[JsonDict]:
        """Return completed rows in chronological key insertion order."""

        return [deepcopy(row) for row in self.payload["completed"].values()]


def next_missing_key(planned: Sequence[str], completed: set[str]) -> str | None:
    """Resume at the first planned key that lacks a durable result."""

    return next((key for key in planned if key not in completed), None)


def arm_isolation_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject process or store reuse across two arms of one model."""

    errors: list[str] = []
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row.get("model_id"))].append(row)
    for model_rows in by_model.values():
        process_ids = [row.get("process_id") for row in model_rows]
        store_paths = [row.get("store_path") for row in model_rows]
        if len(process_ids) != len(set(process_ids)) and "process_reused_across_arms" not in errors:
            errors.append("process_reused_across_arms")
        if len(store_paths) != len(set(store_paths)) and "store_reused_across_arms" not in errors:
            errors.append("store_reused_across_arms")
    return errors


def _walk_mapping_keys(value: Any) -> set[str]:
    """Collect nested mapping keys for information-leak checks."""

    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.add(str(key))
            found.update(_walk_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_walk_mapping_keys(child))
    return found


def future_label_isolation_errors(event: Mapping[str, Any]) -> list[str]:
    """Reject prompt payloads that expose sealed answers before generation."""

    visible = event.get("arm_prompt_payloads", {})
    if _walk_mapping_keys(visible) & _FORBIDDEN_VISIBLE_KEYS:
        return ["exact_outcome_visible_before_certification"]
    return []


def retrieval_eligibility(
    certificate: Mapping[str, Any], event: Mapping[str, Any]
) -> tuple[bool, str]:
    """Allow only earlier exact certificates that the sequence marks eligible."""

    certificate_id = str(certificate.get("certificate_id"))
    ordinal = int(event.get("ordinal", -1))
    if int(certificate.get("available_after_ordinal", ordinal)) >= ordinal:
        return False, "future_certificate"
    if certificate_id in set(event.get("prohibited_future_certificate_ids", [])):
        return False, "future_certificate"
    if certificate_id == event.get("prohibited_current_certificate_id"):
        return False, "current_certificate"
    if not certificate_is_valid(certificate):
        return False, "invalid_exact_certificate"
    eligible_ids = set(event.get("eligible_prior_certificate_ids", []))
    if eligible_ids and certificate_id not in eligible_ids:
        return False, "not_in_sealed_eligibility_set"
    return True, "eligible_prior_exact_certificate"


def select_retrieval(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
    limit: int,
    *,
    arm: str = "debt_queue",
) -> list[JsonDict]:
    """Select earlier active records without labels from the current event."""

    if arm == "debt_queue" and int(state.get("debt_balance", 0)) >= FROZEN_POLICY.debt_threshold:
        return []
    candidates = []
    tombstones = set(state.get("tombstones", {}))
    for row in state.get("active", []):
        certificate_id = str(row.get("certificate_id"))
        eligible, _ = retrieval_eligibility(row, event)
        if eligible and certificate_id not in tombstones:
            candidates.append(row)
    if arm == "debt_queue":
        candidates.sort(
            key=lambda row: (
                row.get("problem_family") == event.get("problem_family"),
                int(row.get("event_ordinal", -1)),
            ),
            reverse=True,
        )
        candidates = [
            row for row in candidates if row.get("problem_family") == event.get("problem_family")
        ]
    else:
        candidates.sort(key=lambda row: int(row.get("event_ordinal", -1)), reverse=True)
    return [deepcopy(dict(row)) for row in candidates[: max(0, int(limit))]]


def apply_debt_transition(
    *,
    previous_balance: int,
    event_id: str,
    causal_source_ids: Sequence[str],
    cause: str,
    requested_service: int,
    charged_pairs: set[str],
) -> JsonDict:
    """Apply the frozen nonnegative queue recurrence with idempotent arrivals."""

    new_pairs: list[str] = []
    if cause in _DEBT_CAUSES:
        for certificate_id in dict.fromkeys(str(value) for value in causal_source_ids):
            pair = f"{event_id}|{certificate_id}"
            if pair not in charged_pairs:
                new_pairs.append(pair)
    arrival = len(new_pairs) * FROZEN_POLICY.debt_arrival_weight
    available = max(0, int(previous_balance)) + arrival
    service = 0
    if arrival == 0 and cause == "exact_success":
        service = min(max(0, int(requested_service)), available)
    balance = max(0, available - service)
    charged_after = set(charged_pairs) | set(new_pairs)
    return {
        "event_id": event_id,
        "cause": cause,
        "previous_balance": max(0, int(previous_balance)),
        "arrival": arrival,
        "service": service,
        "balance": balance,
        "causal_source_ids": [pair.split("|", 1)[1] for pair in new_pairs],
        "charged_pairs_after": sorted(charged_after),
    }


def admission_decision(
    certificate: Mapping[str, Any],
    arm: str,
    state: Mapping[str, Any],
    policy: FrozenPolicy,
) -> JsonDict:
    """Use exact metadata and past debt, never confidence or rationale."""

    if arm == "no_memory":
        return {"admit": False, "reason": "no_memory_policy", "objective": None}
    if not certificate_is_valid(certificate):
        return {"admit": False, "reason": "invalid_exact_certificate", "objective": None}
    if arm == "fifo":
        return {"admit": True, "reason": "fifo_exact_success", "objective": None}
    if arm != "debt_queue":
        return {"admit": False, "reason": "unknown_arm", "objective": None}
    debt = int(state.get("debt_balance", 0))
    occupancy = len(state.get("active", [])) / max(1, policy.capacity)
    risk = len(certificate.get("certified_risk_flags", []))
    admission_cost = debt * risk + policy.drift_plus_penalty_coefficient * occupancy
    rejection_cost = policy.drift_plus_penalty_coefficient
    objective = {
        "debt": debt,
        "certified_risk": risk,
        "admission_cost": admission_cost,
        "rejection_cost": rejection_cost,
        "coefficient": policy.drift_plus_penalty_coefficient,
    }
    admit = debt < policy.debt_threshold and admission_cost <= rejection_cost
    return {
        "admit": admit,
        "reason": "drift_plus_penalty_accept" if admit else "drift_plus_penalty_reject",
        "objective": objective,
    }


def _proposal_relation(raw_output: str) -> str | None:
    """Extract the model proposal while leaving exact correctness external."""

    lowered = raw_output.lower()
    if re.search(r"non[_ -]?equivalent|not\s+equivalent|reject", lowered):
        return "non_equivalent"
    if "equivalent" in lowered:
        return "equivalent"
    return None


def _make_certificate(
    *,
    event: Mapping[str, Any],
    model_id: str,
    prompt_hash: str,
    raw_output: str,
    exact_outcome: str,
    exact_success: bool,
    parent_store_hash: str,
) -> JsonDict:
    """Bind one external result to its event, model, prompt, output, and parent."""

    certificate: JsonDict = {
        "certificate_id": str(event["certificate_id"]),
        "event_id": str(event["event_id"]),
        "event_ordinal": int(event["ordinal"]),
        "available_after_ordinal": int(event["ordinal"]),
        "problem_family": str(event["problem_family"]),
        "model_id": model_id,
        "prompt_hash": prompt_hash,
        "raw_output_hash": sha256_text(raw_output),
        "exact_outcome": exact_outcome,
        "exact_success": bool(exact_success),
        "authority": EXACT_AUTHORITY,
        "parent_store_hash": parent_store_hash,
        "reusable_factor_ids": list(event.get("reusable_factor_ids", [])),
        "certified_risk_flags": [
            value
            for value in (event.get("conflict_class"), event.get("control_class"))
            if value not in (None, "standard")
        ],
    }
    certificate["certificate_hash"] = certificate_hash(certificate)
    return certificate


def run_event_transaction(
    *,
    event: Mapping[str, Any],
    model_id: str,
    arm: str,
    store: TransactionalMemoryStore,
    raw_output: str,
    exact_outcome: str,
    exact_success: bool,
    retrieved_ids: Sequence[str],
    raw_path: str | Path | None = None,
) -> JsonDict:
    """Persist raw bytes before revealing the exact outcome and deciding a write."""

    started = time.perf_counter_ns()
    parent_hash = store.store_hash()
    output_path = (
        Path(raw_path) if raw_path else store.path.parent / "raw" / f"{event['event_id']}.json"
    )
    step = 1
    write_json_atomic(
        output_path,
        {
            "event_id": event["event_id"],
            "model_id": model_id,
            "arm": arm,
            "raw_output": raw_output,
            "raw_output_hash": sha256_text(raw_output),
        },
    )
    raw_step = step
    step += 1
    outcome_step = step
    certificate = _make_certificate(
        event=event,
        model_id=model_id,
        prompt_hash=str(event.get("dynamic_prompt_hash", event["prompt_hash"])),
        raw_output=raw_output,
        exact_outcome=exact_outcome,
        exact_success=exact_success,
        parent_store_hash=parent_hash,
    )
    cause = "exact_success" if exact_success else "exact_failure"
    if not exact_success and event.get("conflict_class"):
        cause = "contradiction"
    transition = apply_debt_transition(
        previous_balance=int(store.state.get("debt_balance", 0)),
        event_id=str(event["event_id"]),
        causal_source_ids=retrieved_ids,
        cause=cause,
        requested_service=FROZEN_POLICY.service_per_clean_success,
        charged_pairs=set(store.state.get("charged_pairs", [])),
    )
    if arm == "debt_queue":
        store.set_debt(transition)
    decision = admission_decision(certificate, arm, store.state, FROZEN_POLICY)
    evicted_id = None
    write_step = None
    if decision["admit"]:
        step += 1
        write_step = step
        evicted_id = store.admit(certificate)
    ended = time.perf_counter_ns()
    return {
        "model_id": model_id,
        "arm": arm,
        "event_id": event["event_id"],
        "ordinal": int(event["ordinal"]),
        "split": event["split"],
        "problem_family": event["problem_family"],
        "process_id": os.getpid(),
        "store_path": str(store.path),
        "prompt_hash": certificate["prompt_hash"],
        "raw_output": raw_output,
        "raw_output_hash": certificate["raw_output_hash"],
        "proposal_relation": _proposal_relation(raw_output),
        "exact_outcome": exact_outcome,
        "exact_success": bool(exact_success),
        "retrieved_ids": list(retrieved_ids),
        "write_admitted": bool(decision["admit"]),
        "write_reason": decision["reason"],
        "admission_objective": decision["objective"],
        "certificate_id": certificate["certificate_id"],
        "certificate_hash": certificate["certificate_hash"],
        "evicted_certificate_id": evicted_id,
        "debt_transition": transition,
        "debt_balance": int(store.state.get("debt_balance", 0)),
        "store_hash_before": parent_hash,
        "store_hash_after": store.store_hash(),
        "raw_output_durable_step": raw_step,
        "exact_outcome_visible_step": outcome_step,
        "write_step": write_step,
        "latency_ms": (ended - started) / 1_000_000,
        "retention_success": True,
    }


def recompute_arm_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce exact accuracy and retention from authoritative event rows."""

    result: list[JsonDict] = []
    for arm in ARMS:
        selected = [row for row in rows if row.get("arm") == arm]
        exact = [row for row in selected if row.get("split") == "evaluation"]
        result.append(
            {
                "arm": arm,
                "attempted_calls": len(selected),
                "later_event_count": len(exact),
                "exact_success_count": sum(row.get("exact_success") is True for row in exact),
                "exact_accuracy": (
                    sum(row.get("exact_success") is True for row in exact) / len(exact)
                    if exact
                    else 0.0
                ),
                "retention_rate": (
                    sum(row.get("retention_success") is True for row in selected) / len(selected)
                    if selected
                    else 0.0
                ),
            }
        )
    return result


def compute_paired_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Compute paired queue-minus-comparator effects and normal CI95 bounds."""

    by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("split") == "evaluation":
            key = (str(row.get("model_id")), str(row.get("event_id")))
            by_key[key][str(row.get("arm"))] = float(row.get("exact_success") is True)
    paired_rows: list[JsonDict] = []
    confidence_rows: list[JsonDict] = []
    for comparator in ("no_memory", "fifo"):
        differences = [
            values["debt_queue"] - values[comparator]
            for values in by_key.values()
            if "debt_queue" in values and comparator in values
        ]
        count = len(differences)
        mean = sum(differences) / count if count else 0.0
        if count > 1:
            variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
            half_width = 1.96 * math.sqrt(variance / count)
        else:
            half_width = 0.0 if count else 1.0
        paired_rows.append(
            {
                "queue_arm": "debt_queue",
                "comparator": comparator,
                "paired_event_count": count,
                "mean_exact_accuracy_effect": mean,
            }
        )
        confidence_rows.append(
            {
                "queue_arm": "debt_queue",
                "comparator": comparator,
                "ci95_lower": mean - half_width,
                "ci95_upper": mean + half_width,
                "method": "paired_normal_interval",
            }
        )
    return paired_rows, confidence_rows


def evaluate_positive_gate(
    *,
    paired_rows: Sequence[Mapping[str, Any]],
    confidence_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    future_rows: Sequence[Mapping[str, Any]],
    restart_rows: Sequence[Mapping[str, Any]],
    rollback_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Require utility and every safety gate before a positive conclusion."""

    comparators = {"no_memory", "fifo"}
    positive_effects = {
        row.get("comparator")
        for row in paired_rows
        if float(row.get("mean_exact_accuracy_effect", 0.0)) > 0
    }
    positive_intervals = {
        row.get("comparator") for row in confidence_rows if float(row.get("ci95_lower", 0.0)) > 0
    }
    retention_passed = bool(retention_rows) and all(
        row.get("retained", row.get("passed")) is True for row in retention_rows
    )
    safety_groups = (future_rows, restart_rows, rollback_rows)
    safety_passed = all(
        group and all(row.get("passed") is True for row in group) for group in safety_groups
    )
    checks = {
        "paired_accuracy_gain": comparators <= positive_effects,
        "ci95_above_zero": comparators <= positive_intervals,
        "retention_not_reduced": retention_passed,
        "future_certificate_isolation": safety_passed,
    }
    return {"positive": all(checks.values()), "checks": checks}


def empty_artifact(run_date: str) -> JsonDict:
    """Build every required field so blocked runs remain machine-readable."""

    row_fields = [
        field for field in REQUIRED_ARTIFACT_FIELDS if field == "rows" or field.endswith("_rows")
    ]
    artifact: JsonDict = {field: [] for field in row_fields}
    artifact.update(
        {
            "schema": SCHEMA_VERSION,
            "experiment_id": 6962,
            "run_date": run_date,
            "field_principles": deepcopy(FIELD_PRINCIPLES),
            "preconditions_checked": [],
            "inference_substrate": INFERENCE_SUBSTRATE,
            "duration_s": 0.0,
            "source_artifact_hashes": {},
            "model_specs": [dict(row) for row in MODEL_SPECS],
            "task_runtime_receipt": {"rows": []},
            "continuous_self_learning_task": True,
            "learning_tier": 2,
            "no_model_weight_mutation": True,
            "model_hashes_before": {},
            "model_hashes_after": {},
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "queue_learning_run_complete_score": 0,
            "queue_learning_positive_score": 0,
            "gate_check_summary": gate_summary([]),
            "verifier_is_oracle": False,
            "verdict_class": "partial",
            "honest_verdict": "complete_partial_queue_regulated_self_learning",
        }
    )
    return artifact


def _checksum_projection(artifact: Mapping[str, Any]) -> JsonDict:
    """Exclude wall-clock and process-local values from the reproducibility digest."""

    projected = deepcopy(dict(artifact))
    projected.pop("reproducibility_checksum", None)
    projected.pop("duration_s", None)
    for field in ("latency_rows", "task_runtime_receipt", "model_lifecycle_rows"):
        projected.pop(field, None)
    return projected


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding timing and process identities."""

    return sha256_json(_checksum_projection(artifact))


def build_blocked_artifact(
    *,
    run_date: str,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Return a complete blocked result without invented model rows."""

    artifact = empty_artifact(run_date)
    artifact.update(
        {
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "duration_s": float(duration_s),
            "source_artifact_hashes": deepcopy(dict(source_hashes)),
            "gate_check_summary": gate_summary(checks),
            "verdict_class": "blocked",
            "honest_verdict": "blocked_queue_regulated_self_learning",
        }
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _float_equal(left: Any, right: Any) -> bool:
    """Compare metric floats without hiding material aggregate changes."""

    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return left == right


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute authority, isolation, lifecycle, and headline consistency."""

    errors: list[str] = []

    def add(error: str) -> None:
        if error not in errors:
            errors.append(error)

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        add("required_fields_missing")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        add("field_principles_incomplete")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        add("inference_substrate_mismatch")
    if artifact.get("continuous_self_learning_task") is not True:
        add("continuous_self_learning_task_invalid")
    if artifact.get("learning_tier") != 2:
        add("learning_tier_invalid")
    if artifact.get("no_model_weight_mutation") is not True:
        add("model_weight_mutation_declared")
    if artifact.get("verifier_is_oracle") is not False:
        add("verifier_is_oracle_invalid")
    verdict_class = artifact.get("verdict_class")
    if verdict_class not in _VERDICT_CLASSES:
        add("verdict_class_invalid")
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict_class == "blocked":
        if verdict != "blocked_queue_regulated_self_learning":
            add("blocked_verdict_invalid")
        if artifact.get("queue_learning_run_complete_score") != 0:
            add("blocked_run_complete_invalid")
        if artifact.get("queue_learning_positive_score") != 0:
            add("blocked_positive_invalid")
    elif not verdict.startswith(f"complete_{verdict_class}"):
        add("honest_verdict_prefix_mismatch")
    before = artifact.get("model_hashes_before", {})
    after = artifact.get("model_hashes_after", {})
    if before != after:
        add("model_hash_drift")
    for error in arm_isolation_errors(artifact.get("model_rows", [])):
        add(error)
    stored_arms = {row.get("arm"): row for row in artifact.get("arm_rows", [])}
    recomputed_arms = {
        row["arm"]: row for row in recompute_arm_rows(artifact.get("event_rows", []))
    }
    for arm, recomputed in recomputed_arms.items():
        stored = stored_arms.get(arm)
        if stored is None:
            if artifact.get("verdict_class") != "blocked":
                add("aggregate_mismatch")
            continue
        for key in (
            "attempted_calls",
            "later_event_count",
            "exact_success_count",
            "exact_accuracy",
        ):
            if not _float_equal(stored.get(key), recomputed.get(key)):
                add("aggregate_mismatch")
    if artifact.get("queue_learning_positive_score") == 1 and errors:
        add("positive_score_invalid")
    if artifact.get("reproducibility_checksum") and artifact.get(
        "reproducibility_checksum"
    ) != payload_checksum(artifact):
        add("reproducibility_checksum_mismatch")
    return errors


def resolve_model_specs() -> list[JsonDict]:
    """Resolve only the two required GGUF models to concrete local files."""

    resolved = []
    for spec in MODEL_SPECS:
        row = dict(spec)
        row["model_path"] = resolve_cached_gguf(row["hf_id"])
        resolved.append(row)
    return resolved


def _read_json(path: Path) -> JsonDict:
    """Read one JSON object or raise a clear precondition error."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def collect_preconditions(
    *,
    source: Mapping[str, Any],
    source_path: Path,
    workspace: Path,
    model_specs: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], JsonDict]:
    """Check every external resource before starting a prospective call."""

    source_hashes = {"event_sequence": sha256_file(source_path)}
    checks = [
        gate_check(
            "certified_event_sequence_ready_score",
            1,
            source.get("certified_event_sequence_ready_score"),
        )
    ]
    checkpoint_text = source.get("sealed_checkpoint_path")
    checkpoint = Path(str(checkpoint_text)) if checkpoint_text else workspace / "missing-checkpoint"
    checkpoint_hash = sha256_file(checkpoint)
    source_hashes["sealed_event_checkpoint"] = checkpoint_hash
    checks.append(
        gate_check(
            "sealed_event_checkpoint", source.get("sealed_checkpoint_sha256"), checkpoint_hash
        )
    )
    for spec in model_specs:
        model_id = str(spec["hf_id"])
        model_path = spec.get("model_path")
        checks.append(
            gate_check(
                f"cached_gguf:{model_id}", True, bool(model_path and Path(model_path).is_file())
            )
        )
        probe_ok, probe_detail = gguf_tokenizer_loadable(str(model_path) if model_path else None)
        checks.append(gate_check(f"vocab_only_probe:{model_id}", True, probe_ok))
        source_hashes[f"model:{model_id}"] = {
            "path": model_path,
            "sha256": sha256_file(model_path) if model_path else None,
            "vocab_probe": probe_detail,
        }
    try:
        from llama_cpp import llama_cpp

        gpu_offload = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        gpu_offload = False
    try:
        gpu_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        cuda_devices = [line for line in gpu_query.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        cuda_devices = []
    checks.append(
        gate_check(
            "authenticated_cuda_offload",
            {"llama_cpp_support": True, "cuda_device_count": len(model_specs)},
            {"llama_cpp_support": gpu_offload, "cuda_device_count": len(cuda_devices)},
        )
    )
    try:
        with tempfile.TemporaryDirectory(dir=workspace) as temporary:
            root = Path(temporary)
            store = TransactionalMemoryStore(root / "store.json", capacity=2)
            before = store.store_hash()
            store.hard_reset()
            transactional_ok = before == store.store_hash()
            checkpoint_probe = EventCheckpoint(root / "events.json")
            recovery_ok = checkpoint_probe.record(
                "probe", {"ok": True}
            ) and not checkpoint_probe.record("probe", {"ok": False})
    except (OSError, ValueError, TransactionError):
        transactional_ok = False
        recovery_ok = False
    checks.append(gate_check("transactional_stores", True, transactional_ok))
    checks.append(gate_check("hard_reset_support", True, transactional_ok))
    exact_access = False
    try:
        from carnot.experiment_6961_certified_event_sequence import sequence_conformance_errors

        exact_access = sequence_conformance_errors(source) == []
    except (ImportError, KeyError, TypeError, ValueError):
        exact_access = False
    checks.append(gate_check("exact_certifier_access", True, exact_access))
    checks.append(gate_check("per_event_recovery", True, recovery_ok))
    return checks, source_hashes


def _memory_snippets(certificates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose bounded prior factors without copying a full answer mapping."""

    return [
        {
            "certificate_id": row["certificate_id"],
            "prior_exact_relation": row["exact_outcome"],
            "problem_family": row["problem_family"],
            "reusable_factor_ids": list(row.get("reusable_factor_ids", [])),
        }
        for row in certificates
    ]


def render_prompt(
    event: Mapping[str, Any], retrieved: Sequence[Mapping[str, Any]]
) -> tuple[str, str]:
    """Build one answer-free prompt from the sealed visible payload and prior memory."""

    payload = deepcopy(dict(event["prompt_payload"]))
    payload["retrieved_memory"] = _memory_snippets(retrieved)
    prompt = (
        "Return JSON with one key named relation. Its value must be equivalent or "
        "non_equivalent. Judge only the two formulations and the bounded prior exact "
        "certificates.\n" + canonical_json(payload)
    )
    return prompt, sha256_text(prompt)


def _gpu_process_memory(pid: int) -> int:
    """Read CUDA memory tied to the current worker process."""

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    for line in query.stdout.splitlines():
        columns = [value.strip() for value in line.split(",")]
        if len(columns) == 2 and columns[0] == str(pid):
            try:
                return int(columns[1])
            except ValueError:
                return 0
    return 0


def _load_llama(model_path: str, gpu: int, policy: FrozenPolicy) -> Any:
    """Load one model on its private CUDA device with frozen decoding controls."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from llama_cpp import Llama

    return Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=8192,
        n_batch=512,
        seed=policy.seed,
        verbose=False,
    )


def _generate(llm: Any, prompt: str, max_tokens: int, policy: FrozenPolicy) -> tuple[str, int, int]:
    """Generate one frozen-budget proposal through the GGUF chat template."""

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=int(max_tokens),
        temperature=policy.temperature,
        top_p=policy.top_p,
        seed=policy.seed,
    )
    text = str(response["choices"][0]["message"]["content"])
    usage = response.get("usage", {})
    return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def _runtime_receipt(row: Mapping[str, Any], gpu_memory_mb: int) -> JsonDict:
    """Bind one output hash to its worker process and CUDA evidence."""

    return {
        "task_id": "experiment_6962_queue_regulated_self_learning",
        "control_id": f"{row['model_id']}|{row['arm']}|{row['event_id']}",
        "process_id": row["process_id"],
        "parent_process_id": os.getppid(),
        "raw_output_hash": row["raw_output_hash"],
        "model_id": row["model_id"],
        "gpu_memory_mb": gpu_memory_mb,
        "latency_ms": row["latency_ms"],
        "attribution_confidence": 1.0,
    }


def run_worker(
    *,
    source_path: Path,
    model_spec: Mapping[str, Any],
    arm: str,
    workspace: Path,
    generator_factory: Callable[[str, int, FrozenPolicy], Any] = _load_llama,
    generate_fn: Callable[[Any, str, int, FrozenPolicy], tuple[str, int, int]] = _generate,
) -> JsonDict:
    """Run one model-arm stream in a fresh process with private durable state."""

    source = _read_json(source_path)
    model_id = str(model_spec["hf_id"])
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
    control_root = workspace / safe_name / arm
    control_root.mkdir(parents=True, exist_ok=True)
    store = TransactionalMemoryStore(control_root / "memory.json", capacity=FROZEN_POLICY.capacity)
    checkpoint = EventCheckpoint(control_root / "events.json")
    model_path = str(model_spec["model_path"])
    model_hash_before = sha256_file(model_path)
    load_started = time.perf_counter()
    llm = generator_factory(model_path, int(model_spec["gpu"]), FROZEN_POLICY)
    load_s = time.perf_counter() - load_started
    gpu_memory_mb = _gpu_process_memory(os.getpid())
    if generator_factory is _load_llama and gpu_memory_mb <= 0:
        raise RuntimeError("authenticated CUDA offload has no PID-linked memory evidence")
    events = sorted(source["event_rows"], key=lambda row: int(row["ordinal"]))
    for event in events:
        key = f"{model_id}|{arm}|{event['event_id']}"
        if key in checkpoint.completed_keys():
            continue
        visible_event = {
            key: value for key, value in event.items() if key not in _FORBIDDEN_VISIBLE_KEYS
        }
        retrieval = (
            []
            if arm == "no_memory"
            else select_retrieval(
                store.state,
                event,
                FROZEN_POLICY.retrieval_count,
                arm=arm,
            )
        )
        prompt, dynamic_prompt_hash = render_prompt(visible_event, retrieval)
        raw_output, prompt_tokens, completion_tokens = generate_fn(
            llm,
            prompt,
            int(event["token_budget"]),
            FROZEN_POLICY,
        )
        raw_path = control_root / "raw" / f"{event['event_id']}.json"
        transaction_event = dict(event)
        transaction_event["dynamic_prompt_hash"] = dynamic_prompt_hash
        proposal = _proposal_relation(raw_output)
        exact_outcome = str(event["exact_outcome"])
        exact_success = proposal == exact_outcome
        row = run_event_transaction(
            event=transaction_event,
            model_id=model_id,
            arm=arm,
            store=store,
            raw_output=raw_output,
            exact_outcome=exact_outcome,
            exact_success=exact_success,
            retrieved_ids=[str(value["certificate_id"]) for value in retrieval],
            raw_path=raw_path,
        )
        row.update(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "token_budget": int(event["token_budget"]),
                "retrieval_count": len(retrieval),
                "gpu_memory_mb": gpu_memory_mb,
            }
        )
        row["runtime_receipt"] = _runtime_receipt(row, gpu_memory_mb)
        checkpoint.record(key, row)
    model_hash_after = sha256_file(model_path)
    return {
        "model_id": model_id,
        "arm": arm,
        "process_id": os.getpid(),
        "store_path": str(store.path),
        "checkpoint_path": str(checkpoint.path),
        "model_hash_before": model_hash_before,
        "model_hash_after": model_hash_after,
        "model_load_s": load_s,
        "gpu_memory_mb": gpu_memory_mb,
        "event_rows": checkpoint.rows(),
        "final_store_hash": store.store_hash(),
        "final_debt_balance": int(store.state.get("debt_balance", 0)),
    }


def _run_worker_process(
    *,
    source_path: Path,
    model_spec: Mapping[str, Any],
    arm: str,
    workspace: Path,
) -> JsonDict:
    """Launch one isolated worker and read its atomic summary."""

    summary_path = (
        workspace
        / "summaries"
        / (re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_spec["hf_id"])) + f"-{arm}.json")
    )
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6962_queue_regulated_self_learning",
        "--worker",
        "--event-sequence",
        str(source_path),
        "--workspace",
        str(workspace),
        "--model-spec",
        canonical_json(model_spec),
        "--arm",
        arm,
        "--worker-output",
        str(summary_path),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed for {model_spec['hf_id']} {arm}: {completed.stderr[-2000:]}"
        )
    return _read_json(summary_path)


def _fresh_restore_probe(store_path: Path) -> JsonDict:
    """Restore one store in another process and return its exact state hash."""

    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6962_queue_regulated_self_learning",
        "--restore-probe",
        str(store_path),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        return {"passed": False, "stderr": completed.stderr[-1000:]}
    return json.loads(completed.stdout.strip().splitlines()[-1])


def run_safety_cases(
    model_rows: Sequence[Mapping[str, Any]], source: Mapping[str, Any], workspace: Path
) -> JsonDict:
    """Run frozen post-stream poison, conflict, retention, and recovery cases."""

    output = {
        "poison_rows": [],
        "contradiction_rows": [],
        "retention_rows": [],
        "restart_rows": [],
        "tombstone_rows": [],
        "rollback_rows": [],
        "future_label_isolation_rows": [],
    }
    event = sorted(source["event_rows"], key=lambda row: int(row["ordinal"]))[-1]
    for model_row in model_rows:
        store_path = Path(str(model_row["store_path"]))
        store = TransactionalMemoryStore(store_path, capacity=FROZEN_POLICY.capacity)
        active = deepcopy(store.state.get("active", []))
        model_id = str(model_row["model_id"])
        arm = str(model_row["arm"])
        forged = (
            deepcopy(active[-1])
            if active
            else _make_certificate(
                event=event,
                model_id=model_id,
                prompt_hash=str(event["prompt_hash"]),
                raw_output="equivalent",
                exact_outcome="equivalent",
                exact_success=True,
                parent_store_hash=store.store_hash(),
            )
        )
        forged["authority"] = "model_self_report"
        poison_decision = admission_decision(forged, arm, store.state, FROZEN_POLICY)
        output["poison_rows"].append(
            {
                "model_id": model_id,
                "arm": arm,
                "admitted": poison_decision["admit"],
                "passed": not poison_decision["admit"],
            }
        )
        causal_ids = [str(row["certificate_id"]) for row in active[-1:]]
        transition = apply_debt_transition(
            previous_balance=int(store.state.get("debt_balance", 0)),
            event_id="SAFETY-CONTRADICTION",
            causal_source_ids=causal_ids,
            cause="contradiction",
            requested_service=0,
            charged_pairs=set(store.state.get("charged_pairs", [])),
        )
        output["contradiction_rows"].append(
            {
                "model_id": model_id,
                "arm": arm,
                "arrival": transition["arrival"],
                "causal_source_ids": causal_ids,
                "passed": transition["arrival"] == len(causal_ids),
            }
        )
        retained = bool(active) or arm == "no_memory"
        output["retention_rows"].append(
            {"model_id": model_id, "arm": arm, "retained": retained, "passed": retained}
        )
        restore = _fresh_restore_probe(store_path)
        output["restart_rows"].append(
            {"model_id": model_id, "arm": arm, **restore, "expected_store_hash": store.store_hash()}
        )
        with tempfile.TemporaryDirectory(dir=workspace) as temporary:
            copy_path = Path(temporary) / "safety-store.json"
            copy_path.write_bytes(store_path.read_bytes())
            safety_store = TransactionalMemoryStore(copy_path, capacity=FROZEN_POLICY.capacity)
            parent_bytes = copy_path.read_bytes()
            tombstone_passed = True
            if safety_store.active_ids():
                target = safety_store.active_ids()[0]
                safety_store.tombstone(target, "safety_tombstone")
                tombstone_passed = (
                    target not in safety_store.active_ids()
                    and target in safety_store.state["tombstones"]
                )
            output["tombstone_rows"].append(
                {"model_id": model_id, "arm": arm, "passed": tombstone_passed}
            )
            safety_store.rollback(parent_bytes)
            output["rollback_rows"].append(
                {"model_id": model_id, "arm": arm, "passed": copy_path.read_bytes() == parent_bytes}
            )
        future = deepcopy(forged)
        future["certificate_id"] = str(event["prohibited_current_certificate_id"])
        future["event_ordinal"] = int(event["ordinal"]) + 1
        future["available_after_ordinal"] = int(event["ordinal"]) + 1
        future["authority"] = EXACT_AUTHORITY
        future["certificate_hash"] = certificate_hash(future)
        eligible, reason = retrieval_eligibility(future, event)
        output["future_label_isolation_rows"].append(
            {"model_id": model_id, "arm": arm, "passed": not eligible, "reason": reason}
        )
    return output


def _project_event_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build the normalized row families required by the artifact contract."""

    return {
        "chronology_rows": [
            {
                "model_id": row["model_id"],
                "arm": row["arm"],
                "event_id": row["event_id"],
                "ordinal": row["ordinal"],
            }
            for row in rows
        ],
        "prompt_rows": [
            {
                "model_id": row["model_id"],
                "arm": row["arm"],
                "event_id": row["event_id"],
                "prompt_hash": row["prompt_hash"],
            }
            for row in rows
        ],
        "raw_output_rows": [
            {
                "model_id": row["model_id"],
                "arm": row["arm"],
                "event_id": row["event_id"],
                "raw_output": row["raw_output"],
                "raw_output_hash": row["raw_output_hash"],
            }
            for row in rows
        ],
        "exact_outcome_rows": [
            {
                "model_id": row["model_id"],
                "arm": row["arm"],
                "event_id": row["event_id"],
                "exact_outcome": row["exact_outcome"],
                "exact_success": row["exact_success"],
            }
            for row in rows
        ],
        "retrieval_rows": [
            {
                "model_id": row["model_id"],
                "arm": row["arm"],
                "event_id": row["event_id"],
                "retrieved_ids": row["retrieved_ids"],
            }
            for row in rows
        ],
        "write_rows": [
            {
                "model_id": row["model_id"],
                "arm": row["arm"],
                "event_id": row["event_id"],
                "write_admitted": row["write_admitted"],
                "raw_output_durable_step": row["raw_output_durable_step"],
                "exact_outcome_visible_step": row["exact_outcome_visible_step"],
                "write_step": row["write_step"],
            }
            for row in rows
        ],
        "admission_rows": [
            {
                "model_id": row["model_id"],
                "arm": row["arm"],
                "event_id": row["event_id"],
                "admit": row["write_admitted"],
                "reason": row["write_reason"],
                "objective": row["admission_objective"],
            }
            for row in rows
        ],
        "fifo_rows": [
            {
                "model_id": row["model_id"],
                "event_id": row["event_id"],
                "evicted_certificate_id": row["evicted_certificate_id"],
            }
            for row in rows
            if row["arm"] == "fifo"
        ],
        "debt_arrival_rows": [
            {
                "model_id": row["model_id"],
                "event_id": row["event_id"],
                "arrival": row["debt_transition"]["arrival"],
                "causal_source_ids": row["debt_transition"]["causal_source_ids"],
            }
            for row in rows
            if row["arm"] == "debt_queue"
        ],
        "debt_service_rows": [
            {
                "model_id": row["model_id"],
                "event_id": row["event_id"],
                "service": row["debt_transition"]["service"],
            }
            for row in rows
            if row["arm"] == "debt_queue"
        ],
        "debt_balance_rows": [
            {
                "model_id": row["model_id"],
                "event_id": row["event_id"],
                "balance": row["debt_balance"],
            }
            for row in rows
            if row["arm"] == "debt_queue"
        ],
        "memory_hash_rows": [
            {
                "model_id": row["model_id"],
                "arm": row["arm"],
                "event_id": row["event_id"],
                "before": row["store_hash_before"],
                "after": row["store_hash_after"],
            }
            for row in rows
        ],
        "token_budget_rows": [
            {
                "model_id": row["model_id"],
                "arm": row["arm"],
                "event_id": row["event_id"],
                "budget": row["token_budget"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
            }
            for row in rows
        ],
        "latency_rows": [
            {
                "model_id": row["model_id"],
                "arm": row["arm"],
                "event_id": row["event_id"],
                "latency_ms": row["latency_ms"],
            }
            for row in rows
        ],
        "checkpoint_rows": [
            {
                "model_id": row["model_id"],
                "arm": row["arm"],
                "event_id": row["event_id"],
                "completed_key": f"{row['model_id']}|{row['arm']}|{row['event_id']}",
                "store_hash": row["store_hash_after"],
            }
            for row in rows
        ],
    }


def build_complete_artifact(
    *,
    run_date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    model_rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    workspace: Path,
) -> JsonDict:
    """Reduce isolated workers and post-stream safety cases into one artifact."""

    event_rows = [deepcopy(row) for model_row in model_rows for row in model_row["event_rows"]]
    paired_rows, confidence_rows = compute_paired_metrics(event_rows)
    safety = run_safety_cases(model_rows, source, workspace)
    positive = evaluate_positive_gate(
        paired_rows=paired_rows,
        confidence_rows=confidence_rows,
        retention_rows=[row for row in safety["retention_rows"] if row["arm"] == "debt_queue"],
        future_rows=safety["future_label_isolation_rows"],
        restart_rows=safety["restart_rows"],
        rollback_rows=safety["rollback_rows"],
    )
    expected_calls = len(source["event_rows"]) * len(MODEL_SPECS) * len(ARMS)
    isolation_errors = arm_isolation_errors(model_rows)
    all_safety = [row for values in safety.values() for row in values]
    run_complete = int(
        len(event_rows) == expected_calls
        and not isolation_errors
        and all(row.get("passed") is True for row in all_safety)
    )
    positive_score = int(run_complete == 1 and positive["positive"])
    verdict_class = "positive" if positive_score else "null"
    verdict = (
        "complete_positive_queue_regulated_self_learning"
        if positive_score
        else "complete_null_queue_accuracy_gain_not_certified"
    )
    artifact = empty_artifact(run_date)
    artifact.update(
        {
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "duration_s": duration_s,
            "source_artifact_hashes": deepcopy(dict(source_hashes)),
            "rows": event_rows,
            "model_specs": [deepcopy(dict(row)) for row in model_specs],
            "model_rows": [deepcopy(dict(row)) for row in model_rows],
            "arm_rows": recompute_arm_rows(event_rows),
            "event_rows": event_rows,
            **_project_event_rows(event_rows),
            **safety,
            "paired_metric_rows": paired_rows,
            "confidence_interval_rows": confidence_rows,
            "model_lifecycle_rows": [
                {
                    "model_id": row["model_id"],
                    "arm": row["arm"],
                    "process_id": row["process_id"],
                    "store_path": row["store_path"],
                    "model_hash_before": row["model_hash_before"],
                    "model_hash_after": row["model_hash_after"],
                    "gpu_memory_mb": row["gpu_memory_mb"],
                    "model_load_s": row["model_load_s"],
                }
                for row in model_rows
            ],
            "task_runtime_receipt": {
                "rows": [row["runtime_receipt"] for row in event_rows],
                "isolated_process_count": len({row["process_id"] for row in model_rows}),
            },
            "model_hashes_before": {
                str(row["hf_id"]): source_hashes[f"model:{row['hf_id']}"]["sha256"]
                for row in model_specs
            },
            "model_hashes_after": {
                str(row["hf_id"]): next(
                    model_row["model_hash_after"]
                    for model_row in model_rows
                    if model_row["model_id"] == row["hf_id"]
                )
                for row in model_specs
            },
            "queue_learning_run_complete_score": run_complete,
            "queue_learning_positive_score": positive_score,
            "gate_check_summary": gate_summary(
                list(checks)
                + [
                    gate_check("all_model_arm_event_rows", expected_calls, len(event_rows)),
                    gate_check("arm_isolation", [], isolation_errors),
                    gate_check(
                        "all_safety_rows_pass",
                        True,
                        all(row.get("passed") is True for row in all_safety),
                    ),
                    gate_check("paired_positive_gate", True, positive["positive"]),
                ]
            ),
            "verdict_class": verdict_class,
            "honest_verdict": verdict,
        }
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    *,
    run_date: str,
    source_path: Path,
    output_path: Path,
    workspace: Path,
    worker_runner: Callable[..., JsonDict] = _run_worker_process,
) -> JsonDict:
    """Check gates, run isolated streams, and atomically publish one result."""

    started = time.perf_counter()
    if output_path.is_file():
        existing = _read_json(output_path)
        if existing.get("queue_learning_run_complete_score") == 1 and not validate_artifact(
            existing
        ):
            return existing
    try:
        source = _read_json(source_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        checks = [
            gate_check(
                "certified_event_sequence_readable", True, f"{type(error).__name__}: {error}"
            )
        ]
        artifact = build_blocked_artifact(
            run_date=run_date,
            checks=checks,
            source_hashes={"event_sequence": sha256_file(source_path)},
            duration_s=time.perf_counter() - started,
        )
        write_json_atomic(output_path, artifact)
        return artifact
    workspace.mkdir(parents=True, exist_ok=True)
    model_specs = resolve_model_specs()
    checks, source_hashes = collect_preconditions(
        source=source,
        source_path=source_path,
        workspace=workspace,
        model_specs=model_specs,
    )
    if not gate_summary(checks)["passed"]:
        artifact = build_blocked_artifact(
            run_date=run_date,
            checks=checks,
            source_hashes=source_hashes,
            duration_s=time.perf_counter() - started,
        )
        write_json_atomic(output_path, artifact)
        return artifact
    model_rows = []
    try:
        for model_spec in model_specs:
            for arm in ARMS:
                model_rows.append(
                    worker_runner(
                        source_path=source_path,
                        model_spec=model_spec,
                        arm=arm,
                        workspace=workspace,
                    )
                )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        failure = gate_check(
            "all_model_arm_workers", "complete", f"{type(error).__name__}: {error}"
        )
        artifact = build_blocked_artifact(
            run_date=run_date,
            checks=list(checks) + [failure],
            source_hashes=source_hashes,
            duration_s=time.perf_counter() - started,
        )
        artifact["model_rows"] = model_rows
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
        write_json_atomic(output_path, artifact)
        return artifact
    artifact = build_complete_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        checks=checks,
        source_hashes=source_hashes,
        model_specs=model_specs,
        model_rows=model_rows,
        source=source,
        workspace=workspace,
    )
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        artifact["queue_learning_positive_score"] = 0
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_queue_regulated_self_learning"
        artifact["gate_check_summary"] = gate_summary(
            list(artifact["gate_check_summary"]["checks"])
            + [gate_check("artifact_validation", [], validation_errors)]
        )
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
    write_json_atomic(output_path, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse public orchestration and private worker modes in one command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--event-sequence", type=Path, default=REPO_ROOT / SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_PATH)
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT / CHECKPOINT_ROOT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-spec")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--restore-probe", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and print a compact terminal summary."""

    args = _parse_args(argv)
    if args.restore_probe:
        store = TransactionalMemoryStore(args.restore_probe)
        print(
            canonical_json(
                {
                    "passed": True,
                    "store_hash": store.store_hash(),
                    "state_hash": sha256_json(store.state),
                }
            )
        )
        return 0
    if args.worker:
        if not args.model_spec or not args.arm or not args.worker_output:
            raise SystemExit("worker mode requires --model-spec, --arm, and --worker-output")
        summary = run_worker(
            source_path=args.event_sequence,
            model_spec=json.loads(args.model_spec),
            arm=args.arm,
            workspace=args.workspace,
        )
        write_json_atomic(args.worker_output, summary)
        return 0
    artifact = run(
        run_date=args.date,
        source_path=args.event_sequence,
        output_path=args.output,
        workspace=args.workspace,
    )
    print(
        canonical_json(
            {
                "output": str(args.output),
                "queue_learning_run_complete_score": artifact["queue_learning_run_complete_score"],
                "queue_learning_positive_score": artifact["queue_learning_positive_score"],
                "verdict_class": artifact["verdict_class"],
                "honest_verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the public wrapper.
    raise SystemExit(main())
