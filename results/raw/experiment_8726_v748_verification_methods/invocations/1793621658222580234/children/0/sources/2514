"""Run a prospective four-arm online constraint-routing comparison.

Spec refs: REQ-CL-6791 and SCENARIO-CL-6791-*.

Each arm freezes one route before the exact receipt becomes visible. The CPU
learners can update only after that receipt. No LLM or model weight is used.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

from carnot import experiment_6790_chronological_constraint_routing_stream as stream


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_6791_compositional_online_constraint_routing_ab"
SCHEMA = "carnot.experiment_6791.compositional_online_constraint_routing_ab.v1"
STATE_SCHEMA = "carnot.experiment_6791.isolated_transaction_store.v1"
RUN_DATE = "20260830"
RANDOM_SEED = 6_791_030
INFERENCE_SUBSTRATE = "CPU prospective Tier-2 constraint-memory controller, no LLM"
SOURCE_RELATIVE_PATH = Path("results/experiment_6790_chronological_constraint_routing_stream.json")
RESULT_RELATIVE_PATH = Path("results") / f"{EXPERIMENT_ID}.json"
EXPECTED_SOURCE_ARTIFACT_SHA256 = (
    "sha256:2f2cf984ac9d4dcf4be0fc211329022bc773d3cfd88bb861aaec474e9c53aeb4"
)
EXPECTED_ORDER_HASHES: JsonDict = {
    "order_1": "sha256:eb1b5e209eb223964c68e0b479ed5f0c339a41c40fdbb7bf743c831722b55183",
    "order_2": "sha256:58c7ecf2ce649ad058c96d192c8667945f11ebe185516855f504417e67f08ddd",
    "order_3": "sha256:259b36180d68a5c7214078f3659c42b55bb663ee4ab82ec1b81226c7c815ccf4",
    "order_4": "sha256:b3a2752bf52c4ca72ed084b21ada328a154b4504979f369f7cc6a9682704cf39",
    "order_5": "sha256:ac41cb8ac831556cca9e2fb2a0ebebd966cb567be7e294ca8201d0f81f32d14e",
}

ARMS = (
    "frozen_controller",
    "compositional_online",
    "random_update_placebo",
    "retrieval_disabled_online",
)
ONLINE_ARM = "compositional_online"
PLACEBO_ARM = "random_update_placebo"
RETRIEVAL_DISABLED_ARM = "retrieval_disabled_online"
UPDATING_ARMS = ARMS[1:]
EVENT_COUNT = 240
ORDER_COUNT = 5
PLANNED_ROW_COUNT = EVENT_COUNT * ORDER_COUNT * len(ARMS)
MAX_RECORDS = 256
RECORD_SLOT_BYTES = 1_024
STORAGE_BYTES = MAX_RECORDS * RECORD_SLOT_BYTES
TOP_K = 3
ROUTE_COST_WEIGHT = 0.05
MISSED_DEPENDENCY_PENALTY = 0.25
BOOTSTRAP_SAMPLES = 20_000
LIVE_ROUTE_IDS = stream.LIVE_ROUTE_IDS
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

ARM_DEFINITIONS: JsonDict = {
    "frozen_controller": {
        "updates": False,
        "retrieval": False,
        "route_selection": "Exp6790 frozen controller",
    },
    "compositional_online": {
        "updates": True,
        "retrieval": True,
        "route_selection": "broad route state plus motif retrieval",
    },
    "random_update_placebo": {
        "updates": True,
        "retrieval": True,
        "route_selection": "matched activity with past-only shuffled targets",
    },
    "retrieval_disabled_online": {
        "updates": True,
        "retrieval": False,
        "route_selection": "exact online factors with retrieval disabled",
    },
}
COMPONENT_DEFINITIONS: JsonDict = {
    "factor_admission": {
        "algorithm": "exact typed factor after receipt",
        "threshold": "non-poison receipt with one earlier same-difficulty target",
        "cadence": "after every exact receipt",
    },
    "retrieval": {
        "algorithm": "stable top-k motif match",
        "top_k": TOP_K,
        "cadence": "once per pre-action snapshot",
    },
    "route_selection": {
        "algorithm": "frozen prior plus broad counts plus retrieved-factor votes",
        "broad_vote_weight": 0.25,
        "retrieval_vote_weight": 2.0,
        "frozen_prior_weight": 0.5,
        "cadence": "once per event",
    },
}
TRANSACTION_CAPACITY: JsonDict = {
    "max_records": MAX_RECORDS,
    "record_slot_bytes": RECORD_SLOT_BYTES,
    "storage_bytes": STORAGE_BYTES,
    "eviction": "none before the frozen stop",
}

ROW_DERIVED_FIELDS = (
    "writes_by_arm_order",
    "later_reads_by_arm_order",
    "action_changes_by_arm_order",
    "component_action_attribution",
    "held_future_utility_by_arm_order",
    "online_minus_frozen_order_effects",
    "online_minus_frozen_lcb",
    "online_minus_placebo_order_effects",
    "hard_case_harm_by_arm_order",
    "retention_by_arm_order",
    "action_support_by_arm_order",
)
REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hash",
    "frozen_manifest",
    "arm_definitions",
    "component_definitions",
    "transaction_capacity",
    "rows",
    "transaction_receipts",
    *ROW_DERIVED_FIELDS,
    "future_feature_violations",
    "active_event_write_violations",
    "compositional_csl_completed",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned schema makes incompatible readers fail closed.",
    "experiment_id": "A stable ID binds the result to its owned producer.",
    "run_date": "The fixed date prevents silent protocol substitution.",
    "status": "Status separates a complete run from a complete precondition block.",
    "field_principles": "Each required field names the evidence boundary it protects.",
    "inference_substrate": "The CPU declaration prevents this result from becoming an LLM claim.",
    "duration_s": "Measured wall time shows that the prospective replay executed.",
    "random_seed": "One seed fixes placebo targets and bootstrap samples.",
    "reproducibility_checksum": "A stable hash detects row or protocol drift.",
    "source_artifact_hash": "The exact Exp6790 bytes bind every action opportunity.",
    "frozen_manifest": "The manifest fixes algorithms, order, budget, and stopping before actions.",
    "arm_definitions": "Named controls prevent an unplanned substitute arm.",
    "component_definitions": "Separate component contracts make action attribution possible.",
    "transaction_capacity": "Equal fixed slots make update activity comparable.",
    "rows": "Every order-event-arm cell preserves action and receipt chronology.",
    "transaction_receipts": "Receipts bind writes, restarts, rollback, and later use.",
    "writes_by_arm_order": "Write counts prove online activity and placebo matching.",
    "later_reads_by_arm_order": "Later reads distinguish stored factors from inert writes.",
    "action_changes_by_arm_order": "Changed routes prove behavioral influence.",
    "component_action_attribution": "Counterfactual actions assign influence to one component.",
    "held_future_utility_by_arm_order": "Held-family utility measures future value net of route cost.",
    "online_minus_frozen_order_effects": "Paired order effects retain the replicate unit.",
    "online_minus_frozen_lcb": "A seeded bootstrap limits positive credit from unstable effects.",
    "online_minus_placebo_order_effects": "The matched placebo tests update content, not activity.",
    "hard_case_harm_by_arm_order": "Hard-case checks block aggregate gains that damage difficult rows.",
    "retention_by_arm_order": "Late old-family rows detect forgetting before held-family transfer.",
    "action_support_by_arm_order": "Unique routes detect policy support contraction.",
    "future_feature_violations": "An empty list proves every snapshot used past-only evidence.",
    "active_event_write_violations": "An empty list proves no action episode mutated state.",
    "compositional_csl_completed": "Completion depends on attributable evidence, not effect sign.",
    "gate_check_summary": "Every gate keeps its expected and observed value.",
    "verifier_is_oracle": "False records that exact receipts arrive after frozen actions.",
    "verdict_class": "A closed class separates gain, no gain, block, and disqualification.",
    "honest_verdict": "A terminal prefix gives the conductor an unambiguous final state.",
}


class ReadOnlyEventError(RuntimeError):
    """Raised when an arm tries to write during its current action."""


def canonical_json_bytes(value: Any) -> bytes:
    """Use one byte form for durable state and evidence hashes."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a named SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one value through the canonical JSON byte form."""

    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str | None:
    """Hash one source file, or report absence without inventing evidence."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace one file only after its complete bytes reach the temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream_handle:
            stream_handle.write(data)
            stream_handle.flush()
            os.fsync(stream_handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class IsolatedTransactionStore:
    """Keep one arm-order state isolated and immutable during each action."""

    def __init__(self, state_dir: Path | str, arm: str, order_id: str) -> None:
        self.state_dir = Path(state_dir)
        self.state_path = self.state_dir / "state.json"
        self.arm = arm
        self.order_id = order_id
        self._active: JsonDict | None = None
        self.active_event_write_violations: list[JsonDict] = []
        if not self.state_path.exists():
            state = {
                "schema": STATE_SCHEMA,
                "arm": arm,
                "order_id": order_id,
                "version": 0,
                "records": [],
            }
            _atomic_write(self.state_path, canonical_json_bytes(state))
        self._state = self._read_state()
        if self._state["arm"] != arm or self._state["order_id"] != order_id:
            raise ValueError("transaction store ownership mismatch")

    def _read_state(self) -> JsonDict:
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
            raise ValueError("invalid transaction state")
        return value

    def state_bytes(self) -> bytes:
        return self.state_path.read_bytes()

    def state_hash(self) -> str:
        return sha256_bytes(self.state_bytes())

    def records(self) -> list[JsonDict]:
        return deepcopy(list(self._state["records"]))

    def begin_event(self, event_id: str) -> JsonDict:
        state_bytes = self.state_bytes()
        self._active = {
            "event_id": event_id,
            "state_bytes": state_bytes,
            "state_hash": sha256_bytes(state_bytes),
            "version": int(self._state["version"]),
            "records": self.records(),
        }
        return deepcopy(self._active)

    def end_event(self) -> None:
        self._active = None

    def commit_factor(self, factor: Mapping[str, Any], *, transaction_id: str) -> JsonDict:
        if self._active is not None:
            violation = {
                "transaction_id": transaction_id,
                "event_id": self._active["event_id"],
                "attempted": True,
                "rejected": True,
                "state_hash": self._active["state_hash"],
            }
            self.active_event_write_violations.append(violation)
            raise ReadOnlyEventError("active event is read-only")
        if len(self._state["records"]) >= MAX_RECORDS:
            raise ValueError("transaction capacity exhausted")
        parent_bytes = self.state_bytes()
        new_state = deepcopy(self._state)
        new_state["version"] = int(new_state["version"]) + 1
        new_state["records"].append(deepcopy(dict(factor)))
        new_bytes = canonical_json_bytes(new_state)
        _atomic_write(self.state_path, new_bytes)
        self._state = new_state
        return {
            "transaction_id": transaction_id,
            "committed": True,
            "parent_hash": sha256_bytes(parent_bytes),
            "new_state_hash": sha256_bytes(new_bytes),
            "state_version_after": new_state["version"],
            "inverse_patch": {
                "operation": "remove_last_factor",
                "factor_id": factor["factor_id"],
                "parent_version": new_state["version"] - 1,
            },
            "parent_bytes_b64": base64.b64encode(parent_bytes).decode("ascii"),
            "new_state_bytes_b64": base64.b64encode(new_bytes).decode("ascii"),
        }

    def restart_receipt(self) -> JsonDict:
        expected = self.state_bytes()
        restarted = type(self)(self.state_dir, self.arm, self.order_id)
        actual = restarted.state_bytes()
        return {
            "expected_hash": sha256_bytes(expected),
            "actual_hash": sha256_bytes(actual),
            "bytes_match": actual == expected,
        }

    def rollback(self, receipt: Mapping[str, Any]) -> JsonDict:
        parent = base64.b64decode(str(receipt["parent_bytes_b64"]).encode("ascii"))
        current = self._read_state()
        patch = receipt["inverse_patch"]
        reverted = deepcopy(current)
        removed = reverted["records"].pop()
        reverted["version"] = patch["parent_version"]
        inverse_matches = (
            removed["factor_id"] == patch["factor_id"] and canonical_json_bytes(reverted) == parent
        )
        _atomic_write(self.state_path, parent)
        self._state = self._read_state()
        return {
            "inverse_patch_applied": inverse_matches,
            "byte_identical": inverse_matches and self.state_bytes() == parent,
            "restored_hash": self.state_hash(),
        }

    def reapply(self, receipt: Mapping[str, Any]) -> None:
        new_bytes = base64.b64decode(str(receipt["new_state_bytes_b64"]).encode("ascii"))
        _atomic_write(self.state_path, new_bytes)
        self._state = self._read_state()


def _gate(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Keep the exact value that decides one precondition or completion gate."""

    return {
        "check": check,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": observed == expected if passed is None else passed,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Collect failures without losing their expected and observed values."""

    copied = [deepcopy(dict(row)) for row in checks]
    failures = [row for row in copied if row["passed"] is not True]
    return {
        "all_passed": not failures,
        "checks": copied,
        "failed_checks": [row["check"] for row in failures],
        "failures": failures,
    }


def _load_source(path: Path) -> JsonDict:
    """Load the frozen source object and fail closed on a non-object root."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source artifact root must be an object")
    return value


def _frozen_manifest(source: Mapping[str, Any]) -> JsonDict:
    """Freeze all algorithms, resource limits, chronology, and stopping rules."""

    source_orders = source.get("order_definitions", [])
    manifest: JsonDict = {
        "schema": "carnot.experiment_6791.frozen_manifest.v1",
        "source_experiment_id": source.get("experiment_id"),
        "source_reproducibility_checksum": source.get("reproducibility_checksum"),
        "source_event_manifest_sha256": source.get("frozen_manifest", {}).get("manifest_sha256"),
        "arms": list(ARMS),
        "component_algorithms": deepcopy(COMPONENT_DEFINITIONS),
        "transaction_capacity": deepcopy(TRANSACTION_CAPACITY),
        "thresholds": {
            "route_cost_weight": ROUTE_COST_WEIGHT,
            "missed_dependency_penalty": MISSED_DEPENDENCY_PENALTY,
            "bootstrap_alpha": 0.05,
        },
        "update_cadence": "after the current exact receipt and before the next event",
        "seeds": {
            "controller": RANDOM_SEED,
            "placebo": RANDOM_SEED + 1,
            "bootstrap": RANDOM_SEED + 2,
        },
        "route_budget": stream.LIVE_ROUTE_BUDGET,
        "order_hashes": {
            str(row["order_id"]): row["order_hash"]
            for row in source_orders
            if isinstance(row, Mapping)
        },
        "held_future_first_position": {
            str(row["order_id"]): row["held_future_first_position"]
            for row in source_orders
            if isinstance(row, Mapping)
        },
        "stopping_rule": {
            "orders": ORDER_COUNT,
            "events_per_order": EVENT_COUNT,
            "arms": len(ARMS),
            "planned_rows": PLANNED_ROW_COUNT,
            "reduced_order_fallback": False,
        },
        "frozen_before_first_action": True,
        "unplanned_substitute": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def evaluate_preconditions(
    source: Mapping[str, Any],
    *,
    source_path: Path,
    state_root: Path,
    overrides: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Check source authority, chronology, legal actions, headroom, and stores."""

    orders = source.get("order_definitions", [])
    order_hashes = {
        str(row.get("order_id")): row.get("order_hash")
        for row in orders
        if isinstance(row, Mapping)
    }
    headroom = source.get("diagnostic_headroom_by_order", {})
    rows = source.get("rows", [])
    legal_actions = bool(rows) and all(
        set(row.get("available_actions", [])) == set(LIVE_ROUTE_IDS)
        and stream.EXHAUSTIVE_ROUTE_ID not in row.get("available_actions", [])
        and all(
            action in LIVE_ROUTE_IDS for action in row.get("chosen_baseline_actions", {}).values()
        )
        for row in rows
        if isinstance(row, Mapping)
    )
    hidden_separation = bool(rows) and all(
        row.get("chronology", {}).get("receipt_visible_at_action") is False
        and row.get("chronology", {}).get("actions_fixed_before_receipt") is True
        and "revealed_post_action_receipt" not in row.get("pre_action", {})
        for row in rows
        if isinstance(row, Mapping)
    )
    isolated_store_paths: list[str] = []
    stores_writable = True
    try:
        for order_id in EXPECTED_ORDER_HASHES:
            for arm in ARMS:
                store = IsolatedTransactionStore(state_root / order_id / arm, arm, order_id)
                isolated_store_paths.append(str(store.state_path.resolve()))
                stores_writable = stores_writable and store.state_path.is_file()
    except (OSError, ValueError):
        stores_writable = False
    observed: JsonDict = {
        "source_artifact_sha256": sha256_file(source_path),
        "constraint_routing_stream_ready": source.get("constraint_routing_stream_ready") is True,
        "all_five_order_hashes": order_hashes,
        "positive_headroom_every_order": set(headroom) == set(EXPECTED_ORDER_HASHES)
        and all(float(headroom[order_id].get("accuracy_gap", 0.0)) > 0 for order_id in headroom),
        "legal_route_actions": legal_actions,
        "hidden_receipt_separation": hidden_separation
        and source.get("future_feature_violations") == [],
        "writable_isolated_transaction_stores": stores_writable
        and len(isolated_store_paths) == len(set(isolated_store_paths)) == ORDER_COUNT * len(ARMS),
    }
    observed.update(dict(overrides or {}))
    checks = [
        _gate(
            "source_artifact_sha256",
            EXPECTED_SOURCE_ARTIFACT_SHA256,
            observed["source_artifact_sha256"],
        ),
        _gate(
            "constraint_routing_stream_ready",
            True,
            observed["constraint_routing_stream_ready"],
        ),
        _gate("all_five_order_hashes", EXPECTED_ORDER_HASHES, observed["all_five_order_hashes"]),
        _gate(
            "positive_headroom_every_order",
            True,
            observed["positive_headroom_every_order"],
        ),
        _gate("legal_route_actions", True, observed["legal_route_actions"]),
        _gate(
            "hidden_receipt_separation",
            True,
            observed["hidden_receipt_separation"],
        ),
        _gate(
            "writable_isolated_transaction_stores",
            True,
            observed["writable_isolated_transaction_stores"],
        ),
    ]
    return _gate_summary(checks)


def _best_target_route(event: Mapping[str, Any]) -> str:
    """Choose the best live route only after the exact receipt is available."""

    candidates = []
    for index, route_id in enumerate(LIVE_ROUTE_IDS):
        result = stream.evaluate_route(event, route_id)
        utility = _route_utility(result)
        candidates.append((utility, -index, route_id))
    return str(max(candidates)[2])


def _route_utility(result: Mapping[str, Any]) -> float:
    """Credit exact detection after route cost and missed hard dependencies."""

    return round(
        float(result["credited_reward"])
        - ROUTE_COST_WEIGHT * float(result["route_cost"])
        - MISSED_DEPENDENCY_PENALTY * len(result["missed_dependencies"]),
        6,
    )


def _retrieve(
    records: Sequence[Mapping[str, Any]], event: Mapping[str, Any], *, enabled: bool
) -> list[JsonDict]:
    """Return stable top-k motif factors from earlier committed receipts."""

    if not enabled:
        return []
    ranked = []
    for record in records:
        score = 4.0 * int(record["motif_id"] == event["reusable_motif_id"])
        score += 1.0 * int(record["stratum"] == event["difficulty"])
        score += 0.5 * int(record["source_topology_family"] == event["topology_family"])
        if score >= 4.0:
            ranked.append(
                {
                    "factor_id": record["factor_id"],
                    "target_route": record["target_route"],
                    "score": score,
                    "source_position": record["source_position"],
                }
            )
    return sorted(ranked, key=lambda row: (-row["score"], row["factor_id"]))[:TOP_K]


def _select_route(
    observation: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    retrieved: Sequence[Mapping[str, Any]],
    *,
    route_selection_enabled: bool = True,
    disabled_factor_ids: Sequence[str] = (),
) -> tuple[str, JsonDict]:
    """Select from one snapshot while allowing exact component counterfactuals."""

    disabled = set(disabled_factor_ids)
    frozen_action = stream.choose_frozen_action(observation)
    scores = {route_id: 0.0 for route_id in LIVE_ROUTE_IDS}
    scores[frozen_action] += 0.5
    broad_counts = Counter()
    if route_selection_enabled:
        for record in records:
            if record["factor_id"] in disabled:
                continue
            if record["stratum"] == observation["difficulty"]:
                broad_counts[str(record["target_route"])] += 1
                scores[str(record["target_route"])] += 0.25
    retrieval_counts = Counter()
    for item in retrieved:
        if item["factor_id"] in disabled:
            continue
        retrieval_counts[str(item["target_route"])] += 1
        scores[str(item["target_route"])] += 2.0
    selected = max(
        LIVE_ROUTE_IDS, key=lambda route_id: (scores[route_id], -LIVE_ROUTE_IDS.index(route_id))
    )
    return selected, {
        "scores": {route_id: round(scores[route_id], 6) for route_id in LIVE_ROUTE_IDS},
        "broad_route_counts": dict(sorted(broad_counts.items())),
        "retrieval_route_counts": dict(sorted(retrieval_counts.items())),
        "selected_action": selected,
        "tie_break_order": list(LIVE_ROUTE_IDS),
    }


def _factor_type(event: Mapping[str, Any]) -> str:
    planted = list(event["planted_failure_factor_ids"])
    if planted and str(planted[0]).startswith("local:"):
        return "local_route_factor"
    if planted:
        return "dependency_route_factor"
    return "clean_route_anchor"


def _build_factor(
    *,
    event: Mapping[str, Any],
    order_id: str,
    position: int,
    target_route: str,
    evidence_hash: str,
    update_target: Mapping[str, Any],
    placebo_shuffled: bool,
) -> JsonDict:
    """Bind one typed factor to its exact receipt and selected update target."""

    return {
        "factor_id": f"factor:{order_id}:{event['event_id']}",
        "factor_type": _factor_type(event),
        "source_event_id": event["event_id"],
        "source_position": position,
        "source_topology_family": event["topology_family"],
        "stratum": event["difficulty"],
        "motif_id": event["reusable_motif_id"],
        "target_route": target_route,
        "update_target_event_id": update_target["event_id"],
        "update_target_position": update_target["position"],
        "evidence_hash": evidence_hash,
        "exact_provenance": True,
        "placebo_shuffled": placebo_shuffled,
    }


def _snapshot_metadata(snapshot: Mapping[str, Any], *, arm: str, order_id: str) -> JsonDict:
    records = snapshot["records"]
    positions = [int(row["source_position"]) for row in records]
    return {
        "state_hash": snapshot["state_hash"],
        "state_bytes_sha256": sha256_bytes(snapshot["state_bytes"]),
        "state_path": f"{order_id}/{arm}/state.json",
        "owner_arm": arm,
        "owner_order": order_id,
        "version": snapshot["version"],
        "record_count": len(records),
        "max_source_position": max(positions) if positions else None,
        "held_future_factor_count": sum(
            row["source_topology_family"] == stream.HELD_FUTURE_FAMILY for row in records
        ),
    }


def _compact_transaction(
    receipt: Mapping[str, Any],
    *,
    factor: Mapping[str, Any] | None,
    order_id: str,
    event_id: str,
    position: int,
    arm: str,
    stratum: str,
    admission_reason: str,
    restart_bytes_match: bool,
    rollback_byte_identical: bool,
) -> JsonDict:
    committed = receipt.get("committed") is True
    return {
        "transaction_id": receipt["transaction_id"],
        "order_id": order_id,
        "event_id": event_id,
        "position": position,
        "arm": arm,
        "stratum": stratum,
        "committed": committed,
        "admission_reason": admission_reason,
        "factor_id": factor["factor_id"] if factor else None,
        "factor_type": factor["factor_type"] if factor else None,
        "source_position": factor["source_position"] if factor else None,
        "evidence_hash": factor["evidence_hash"] if factor else None,
        "target_route": factor["target_route"] if factor else None,
        "update_target_event_id": factor["update_target_event_id"] if factor else None,
        "update_target_position": factor["update_target_position"] if factor else None,
        "update_target_stratum": factor["stratum"] if factor else stratum,
        "exact_provenance": factor["exact_provenance"] if factor else None,
        "placebo_shuffled": factor["placebo_shuffled"] if factor else False,
        "parent_hash": receipt.get("parent_hash"),
        "new_state_hash": receipt.get("new_state_hash"),
        "state_version_after": receipt.get("state_version_after"),
        "inverse_patch": deepcopy(receipt.get("inverse_patch")),
        "logical_transaction_bytes": RECORD_SLOT_BYTES if committed else 0,
        "restart_bytes_match": restart_bytes_match,
        "rollback_byte_identical": rollback_byte_identical,
        "later_read_count": 0,
        "later_action_influence_count": 0,
    }


def _rejected_transaction(
    *, arm: str, order_id: str, event_id: str, position: int, stratum: str, reason: str
) -> JsonDict:
    receipt = {"transaction_id": f"tx:{order_id}:{event_id}:{arm}", "committed": False}
    return _compact_transaction(
        receipt,
        factor=None,
        order_id=order_id,
        event_id=event_id,
        position=position,
        arm=arm,
        stratum=stratum,
        admission_reason=reason,
        restart_bytes_match=True,
        rollback_byte_identical=True,
    )


def _placebo_target(
    past_targets: Sequence[Mapping[str, Any]], *, order_id: str, event_id: str
) -> JsonDict:
    material = f"{RANDOM_SEED + 1}:{order_id}:{event_id}".encode()
    index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(past_targets)
    return deepcopy(dict(past_targets[index]))


def _counterfactuals(
    observation: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    retrieved: Sequence[Mapping[str, Any]],
    selected_action: str,
    snapshot_hash: str,
) -> tuple[JsonDict, list[JsonDict]]:
    frozen_action, _ = _select_route(observation, [], [])
    without_retrieval, _ = _select_route(observation, records, [])
    without_route_selection, _ = _select_route(
        observation, records, retrieved, route_selection_enabled=False
    )
    factors = []
    for item in retrieved:
        action, _ = _select_route(
            observation,
            records,
            retrieved,
            disabled_factor_ids=[str(item["factor_id"])],
        )
        factors.append(
            {
                "factor_id": item["factor_id"],
                "snapshot_state_hash": snapshot_hash,
                "same_snapshot_bytes": True,
                "action_without_factor": action,
                "action_changed": action != selected_action,
            }
        )
    return {
        "without_admission": frozen_action,
        "without_retrieval": without_retrieval,
        "without_route_selection": without_route_selection,
    }, factors


def _run_order(
    *,
    order: Mapping[str, Any],
    event_by_id: Mapping[str, Mapping[str, Any]],
    source_row_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    state_root: Path,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    order_id = str(order["order_id"])
    stores = {
        arm: IsolatedTransactionStore(state_root / order_id / arm, arm, order_id) for arm in ARMS
    }
    rows: list[JsonDict] = []
    transactions: list[JsonDict] = []
    active_violations: list[JsonDict] = []
    past_targets: dict[str, list[JsonDict]] = defaultdict(list)
    for position, event_id_value in enumerate(order["event_ids"]):
        event_id = str(event_id_value)
        event = event_by_id[event_id]
        source_row = source_row_by_key[(order_id, event_id)]
        snapshots = {arm: stores[arm].begin_event(event_id) for arm in ARMS}
        event_rows: dict[str, JsonDict] = {}
        observation = deepcopy(event["legal_observation"])
        baseline_action = stream.choose_frozen_action(observation)
        for arm in ARMS:
            records = snapshots[arm]["records"]
            retrieval_enabled = bool(ARM_DEFINITIONS[arm]["retrieval"])
            retrieved = _retrieve(records, event, enabled=retrieval_enabled)
            selected_action, route_state = (
                (
                    baseline_action,
                    {
                        "scores": {
                            route_id: float(route_id == baseline_action)
                            for route_id in LIVE_ROUTE_IDS
                        },
                        "broad_route_counts": {},
                        "retrieval_route_counts": {},
                        "selected_action": baseline_action,
                        "tie_break_order": list(LIVE_ROUTE_IDS),
                    },
                )
                if arm == "frozen_controller"
                else _select_route(observation, records, retrieved)
            )
            component_actions, factor_counterfactuals = _counterfactuals(
                observation,
                records,
                retrieved,
                selected_action,
                snapshots[arm]["state_hash"],
            )
            if stores[arm].state_hash() != snapshots[arm]["state_hash"]:
                active_violations.append(
                    {
                        "order_id": order_id,
                        "event_id": event_id,
                        "arm": arm,
                        "reason": "state_changed_during_action",
                    }
                )
            event_rows[arm] = {
                "schema": "carnot.experiment_6791.event_arm_row.v1",
                "row_key": f"{order_id}:{event_id}:{arm}",
                "pair_key": f"{order_id}:{event_id}",
                "order_id": order_id,
                "order_hash": order["order_hash"],
                "event_id": event_id,
                "position": position,
                "arm": arm,
                "snapshot": _snapshot_metadata(snapshots[arm], arm=arm, order_id=order_id),
                "chronology": {
                    "receipt_visible_at_action": False,
                    "actions_fixed_before_receipt": True,
                    "write_allowed_during_action": False,
                },
                "available_actions": list(event["available_actions"]),
                "baseline_action": baseline_action,
                "selected_action": selected_action,
                "retrieved_factor_ids": [row["factor_id"] for row in retrieved],
                "memory_read_count": len(retrieved),
                "memory_write_count": 0,
                "controller_state": {
                    "factor_admission": {
                        "algorithm": COMPONENT_DEFINITIONS["factor_admission"]["algorithm"],
                        "admitted_factor_count": len(records),
                        "capacity_remaining": MAX_RECORDS - len(records),
                    },
                    "retrieval": {
                        "enabled": retrieval_enabled,
                        "retrieved_factor_ids": [row["factor_id"] for row in retrieved],
                        "scores": {row["factor_id"]: row["score"] for row in retrieved},
                    },
                    "route_selection": route_state,
                },
                "component_counterfactual_actions": component_actions,
                "factor_counterfactuals": factor_counterfactuals,
                "hidden_receipt_hash": source_row["hidden_receipt_hash"],
                "topology_family": event["topology_family"],
                "difficulty": event["difficulty"],
                "held_future": event["held_future"],
                "poison_status": event["poison_status"],
                "retention_partition": not event["held_future"] and position >= 80,
            }
        for store in stores.values():
            store.end_event()

        exact_target = _best_target_route(event)
        exact_target_row = {
            "event_id": event_id,
            "position": position,
            "target_route": exact_target,
        }
        eligible = event["poison_status"] == "none" and bool(past_targets[event["difficulty"]])
        for arm in ARMS:
            row = event_rows[arm]
            route_result = stream.evaluate_route(event, row["selected_action"])
            row["revealed_post_action_receipt"] = {
                "source_receipt_hash": source_row["hidden_receipt_hash"],
                "exact_event_valid": event["exact_receipt"]["exact_valid"],
                "route_result": route_result,
                "exact_target_route": exact_target,
                "poison_status": event["poison_status"],
            }
            row["route_utility"] = _route_utility(route_result)
            row["route_success"] = bool(route_result["correct_decision"])
            row["route_cost"] = route_result["route_cost"]
            row["missed_hard_dependency_count"] = len(route_result["missed_dependencies"])
            if arm == "frozen_controller":
                transaction = _rejected_transaction(
                    arm=arm,
                    order_id=order_id,
                    event_id=event_id,
                    position=position,
                    stratum=str(event["difficulty"]),
                    reason="frozen_control_no_update",
                )
            elif not eligible:
                reason = (
                    "poison_or_retention_rejected"
                    if event["poison_status"] != "none"
                    else "no_earlier_target_in_stratum"
                )
                transaction = _rejected_transaction(
                    arm=arm,
                    order_id=order_id,
                    event_id=event_id,
                    position=position,
                    stratum=str(event["difficulty"]),
                    reason=reason,
                )
            else:
                update_target = exact_target_row
                target_route = exact_target
                shuffled = False
                if arm == PLACEBO_ARM:
                    update_target = _placebo_target(
                        past_targets[event["difficulty"]], order_id=order_id, event_id=event_id
                    )
                    target_route = str(update_target["target_route"])
                    shuffled = True
                factor = _build_factor(
                    event=event,
                    order_id=order_id,
                    position=position,
                    target_route=target_route,
                    evidence_hash=str(source_row["hidden_receipt_hash"]),
                    update_target=update_target,
                    placebo_shuffled=shuffled,
                )
                raw = stores[arm].commit_factor(
                    factor, transaction_id=f"tx:{order_id}:{event_id}:{arm}"
                )
                restart = stores[arm].restart_receipt()
                rollback = stores[arm].rollback(raw)
                stores[arm].reapply(raw)
                transaction = _compact_transaction(
                    raw,
                    factor=factor,
                    order_id=order_id,
                    event_id=event_id,
                    position=position,
                    arm=arm,
                    stratum=str(event["difficulty"]),
                    admission_reason="exact_factor_admitted_after_receipt",
                    restart_bytes_match=restart["bytes_match"],
                    rollback_byte_identical=rollback["byte_identical"],
                )
                row["memory_write_count"] = 1
            row["transaction"] = {
                "transaction_id": transaction["transaction_id"],
                "phase": "after_exact_receipt",
                "committed": transaction["committed"],
                "admission_reason": transaction["admission_reason"],
                "state_version_after": transaction["state_version_after"],
            }
            transactions.append(transaction)
            rows.append(row)
        past_targets[event["difficulty"]].append(exact_target_row)
    for store in stores.values():
        active_violations.extend(store.active_event_write_violations)
    return rows, transactions, active_violations


def _annotate_later_use(rows: Sequence[Mapping[str, Any]], transactions: list[JsonDict]) -> None:
    """Attach later read and causal-action counts to their owning write receipt."""

    receipt_by_factor = {
        (row["order_id"], row["arm"], row["factor_id"]): row
        for row in transactions
        if row["committed"] is True
    }
    for row in rows:
        changed = {
            item["factor_id"]
            for item in row["factor_counterfactuals"]
            if item["action_changed"] is True
        }
        for factor_id in row["retrieved_factor_ids"]:
            receipt = receipt_by_factor[(row["order_id"], row["arm"], factor_id)]
            receipt["later_read_count"] += 1
            receipt["later_action_influence_count"] += int(factor_id in changed)


def bootstrap_lcb(effects: Sequence[float]) -> float:
    """Return a seeded 95 percent lower bound over order-level paired effects."""

    if not effects:
        return 0.0
    generator = random.Random(RANDOM_SEED + 2)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [effects[generator.randrange(len(effects))] for _ in effects]
        means.append(sum(sample) / len(sample))
    means.sort()
    return round(means[int(0.025 * BOOTSTRAP_SAMPLES)], 6)


def _nested_counts() -> JsonDict:
    return {arm: {order_id: 0 for order_id in EXPECTED_ORDER_HASHES} for arm in ARMS}


def _selected(
    rows: Sequence[Mapping[str, Any]], *, arm: str, order_id: str
) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["arm"] == arm and row["order_id"] == order_id]


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return round(sum(float(row[field]) for row in rows) / len(rows), 6) if rows else 0.0


def reduce_evidence(
    rows: Sequence[Mapping[str, Any]], transactions: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Derive every effect, safety metric, and activity count from stored evidence."""

    writes = _nested_counts()
    later_reads = _nested_counts()
    action_changes = _nested_counts()
    for receipt in transactions:
        if receipt["committed"] is True:
            writes[receipt["arm"]][receipt["order_id"]] += 1
    for row in rows:
        later_reads[row["arm"]][row["order_id"]] += len(row["retrieved_factor_ids"])
        action_changes[row["arm"]][row["order_id"]] += int(
            row["selected_action"] != row["baseline_action"]
        )
    attribution: JsonDict = {arm: {} for arm in ARMS}
    held: JsonDict = {arm: {} for arm in ARMS}
    hard: JsonDict = {arm: {} for arm in ARMS}
    retention: JsonDict = {arm: {} for arm in ARMS}
    support: JsonDict = {arm: {} for arm in ARMS}
    for order_id in EXPECTED_ORDER_HASHES:
        frozen_hard_rows = [
            row
            for row in _selected(rows, arm="frozen_controller", order_id=order_id)
            if row["difficulty"] in {"hard", "challenge"}
        ]
        frozen_retention_rows = [
            row
            for row in _selected(rows, arm="frozen_controller", order_id=order_id)
            if row["retention_partition"] is True
        ]
        frozen_hard_success = _mean(frozen_hard_rows, "route_success")
        frozen_retention_success = _mean(frozen_retention_rows, "route_success")
        frozen_support = len(
            {
                row["selected_action"]
                for row in _selected(rows, arm="frozen_controller", order_id=order_id)
            }
        )
        for arm in ARMS:
            arm_rows = _selected(rows, arm=arm, order_id=order_id)
            attribution[arm][order_id] = {
                "factor_admission_action_count": sum(
                    row["component_counterfactual_actions"]["without_admission"]
                    != row["selected_action"]
                    for row in arm_rows
                ),
                "retrieval_action_count": sum(
                    row["component_counterfactual_actions"]["without_retrieval"]
                    != row["selected_action"]
                    for row in arm_rows
                ),
                "route_selection_action_count": sum(
                    row["component_counterfactual_actions"]["without_route_selection"]
                    != row["selected_action"]
                    for row in arm_rows
                ),
                "factor_counterfactual_action_count": sum(
                    item["action_changed"] is True
                    for row in arm_rows
                    for item in row["factor_counterfactuals"]
                ),
            }
            held_rows = [row for row in arm_rows if row["held_future"] is True]
            held[arm][order_id] = {
                "event_count": len(held_rows),
                "total_utility": round(sum(float(row["route_utility"]) for row in held_rows), 6),
                "mean_utility": _mean(held_rows, "route_utility"),
                "total_route_cost": sum(int(row["route_cost"]) for row in held_rows),
                "route_success_rate": _mean(held_rows, "route_success"),
                "missed_hard_dependencies": sum(
                    int(row["missed_hard_dependency_count"]) for row in held_rows
                ),
            }
            hard_rows = [row for row in arm_rows if row["difficulty"] in {"hard", "challenge"}]
            hard_success = _mean(hard_rows, "route_success")
            hard[arm][order_id] = {
                "event_count": len(hard_rows),
                "route_success_rate": hard_success,
                "utility_mean": _mean(hard_rows, "route_utility"),
                "success_delta_vs_frozen": round(hard_success - frozen_hard_success, 6),
                "harm": hard_success < frozen_hard_success,
            }
            retention_rows = [row for row in arm_rows if row["retention_partition"] is True]
            retention_success = _mean(retention_rows, "route_success")
            retention[arm][order_id] = {
                "event_count": len(retention_rows),
                "route_success_rate": retention_success,
                "utility_mean": _mean(retention_rows, "route_utility"),
                "success_delta_vs_frozen": round(retention_success - frozen_retention_success, 6),
                "harm": retention_success < frozen_retention_success,
            }
            unique_actions = sorted({str(row["selected_action"]) for row in arm_rows})
            support[arm][order_id] = {
                "unique_actions": unique_actions,
                "unique_action_count": len(unique_actions),
                "support_rate": round(len(unique_actions) / len(LIVE_ROUTE_IDS), 6),
                "harm": len(unique_actions) < frozen_support,
            }
    frozen_effects = {
        order_id: round(
            held[ONLINE_ARM][order_id]["mean_utility"]
            - held["frozen_controller"][order_id]["mean_utility"],
            6,
        )
        for order_id in EXPECTED_ORDER_HASHES
    }
    placebo_effects = {
        order_id: round(
            held[ONLINE_ARM][order_id]["mean_utility"]
            - held[PLACEBO_ARM][order_id]["mean_utility"],
            6,
        )
        for order_id in EXPECTED_ORDER_HASHES
    }
    return {
        "writes_by_arm_order": writes,
        "later_reads_by_arm_order": later_reads,
        "action_changes_by_arm_order": action_changes,
        "component_action_attribution": attribution,
        "held_future_utility_by_arm_order": held,
        "online_minus_frozen_order_effects": frozen_effects,
        "online_minus_frozen_lcb": bootstrap_lcb(list(frozen_effects.values())),
        "online_minus_placebo_order_effects": placebo_effects,
        "hard_case_harm_by_arm_order": hard,
        "retention_by_arm_order": retention,
        "action_support_by_arm_order": support,
    }


def audit_future_features(
    rows: Sequence[Mapping[str, Any]], transactions: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Find a current, future, or early held-family factor in any action snapshot."""

    violations = []
    for row in rows:
        maximum = row["snapshot"]["max_source_position"]
        if maximum is not None and maximum >= row["position"]:
            violations.append(f"future_snapshot:{row['row_key']}")
        if row["position"] <= 160 and row["snapshot"]["held_future_factor_count"] != 0:
            violations.append(f"early_held_factor:{row['row_key']}")
        for item in row["factor_counterfactuals"]:
            if item["snapshot_state_hash"] != row["snapshot"]["state_hash"]:
                violations.append(f"counterfactual_snapshot:{row['row_key']}")
    for receipt in transactions:
        if receipt["committed"] and receipt["source_position"] != receipt["position"]:
            violations.append(f"source_position:{receipt['transaction_id']}")
        if receipt["committed"] and receipt["arm"] == PLACEBO_ARM:
            if receipt["update_target_position"] >= receipt["position"]:
                violations.append(f"placebo_future_target:{receipt['transaction_id']}")
    return sorted(violations)


def audit_cross_arm_state(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Find shared paths, owner mismatches, or unpaired receipt hashes."""

    violations = []
    by_order: dict[str, dict[str, str]] = defaultdict(dict)
    receipt_hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["snapshot"]["owner_arm"] != row["arm"]:
            violations.append(f"owner_arm:{row['row_key']}")
        if row["snapshot"]["owner_order"] != row["order_id"]:
            violations.append(f"owner_order:{row['row_key']}")
        by_order[row["order_id"]][row["arm"]] = row["snapshot"]["state_path"]
        receipt_hashes[row["pair_key"]].add(row["hidden_receipt_hash"])
    for order_id, paths in by_order.items():
        if len(paths) != len(ARMS) or len(set(paths.values())) != len(ARMS):
            violations.append(f"shared_state_path:{order_id}")
    violations.extend(
        f"unpaired_receipt:{pair_key}"
        for pair_key, hashes in receipt_hashes.items()
        if len(hashes) != 1
    )
    return sorted(violations)


def completion_checks(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute completion without consulting the scientific effect direction."""

    rows = artifact["rows"]
    transactions = artifact["transaction_receipts"]
    pairs: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        pairs[row["pair_key"]].add(row["arm"])
    reduced = reduce_evidence(rows, transactions) if rows else {}
    checks = {
        "preconditions_pass": artifact["gate_check_summary"].get("preconditions_pass") is True,
        "all_planned_rows_present": len(rows) == PLANNED_ROW_COUNT,
        "row_keys_unique": len({row["row_key"] for row in rows}) == len(rows),
        "paired_keys_complete": len(pairs) == EVENT_COUNT * ORDER_COUNT
        and all(arms == set(ARMS) for arms in pairs.values()),
        "transactions_attributable": len(transactions) == PLANNED_ROW_COUNT
        and len({row["transaction_id"] for row in transactions}) == len(transactions),
        "aggregates_row_derived": bool(reduced)
        and all(artifact[field] == reduced[field] for field in ROW_DERIVED_FIELDS),
        "component_states_complete": all(
            set(row["controller_state"]) == {"factor_admission", "retrieval", "route_selection"}
            for row in rows
        ),
        "future_features_absent": artifact["future_feature_violations"] == [],
        "active_event_writes_absent": artifact["active_event_write_violations"] == [],
        "cross_arm_state_absent": audit_cross_arm_state(rows) == [],
        "restart_complete": all(
            row["restart_bytes_match"] is True for row in transactions if row["committed"]
        ),
        "rollback_complete": all(
            row["rollback_byte_identical"] is True for row in transactions if row["committed"]
        ),
        "placebo_activity_matched": all(
            reduced.get("writes_by_arm_order", {}).get(ONLINE_ARM, {}).get(order_id)
            == reduced.get("writes_by_arm_order", {}).get(PLACEBO_ARM, {}).get(order_id)
            for order_id in EXPECTED_ORDER_HASHES
        ),
        "retrieval_ablation_exact": all(
            row["retrieved_factor_ids"] == []
            for row in rows
            if row["arm"] == RETRIEVAL_DISABLED_ARM
        ),
        "no_unplanned_substitute": artifact["frozen_manifest"]["unplanned_substitute"] is False,
    }
    return checks


def positive_credit_checks(artifact: Mapping[str, Any]) -> JsonDict:
    """Apply the preregistered gain, activity, harm, retention, and support gates."""

    return {
        "online_writes_every_order": all(
            artifact["writes_by_arm_order"][ONLINE_ARM][order_id] > 0
            for order_id in EXPECTED_ORDER_HASHES
        ),
        "online_later_reads_every_order": all(
            artifact["later_reads_by_arm_order"][ONLINE_ARM][order_id] > 0
            for order_id in EXPECTED_ORDER_HASHES
        ),
        "online_action_changes_every_order": all(
            artifact["action_changes_by_arm_order"][ONLINE_ARM][order_id] > 0
            for order_id in EXPECTED_ORDER_HASHES
        ),
        "online_minus_frozen_lcb_positive": artifact["online_minus_frozen_lcb"] > 0.0,
        "online_beats_placebo_every_order": all(
            effect > 0.0 for effect in artifact["online_minus_placebo_order_effects"].values()
        ),
        "no_hard_case_harm": all(
            artifact["hard_case_harm_by_arm_order"][ONLINE_ARM][order_id]["harm"] is False
            for order_id in EXPECTED_ORDER_HASHES
        ),
        "no_retention_harm": all(
            artifact["retention_by_arm_order"][ONLINE_ARM][order_id]["harm"] is False
            for order_id in EXPECTED_ORDER_HASHES
        ),
        "no_support_harm": all(
            artifact["action_support_by_arm_order"][ONLINE_ARM][order_id]["harm"] is False
            for order_id in EXPECTED_ORDER_HASHES
        ),
    }


def terminal_verdict(
    *, completed: bool, failed_checks: set[str], positive: bool
) -> tuple[str, str]:
    """Map row-derived completion and gain checks to the closed terminal class."""

    disqualifying = {
        "all_planned_rows_present",
        "paired_keys_complete",
        "future_features_absent",
        "cross_arm_state_absent",
        "no_unplanned_substitute",
    }
    if positive:
        return (
            "positive",
            "complete_positive: compositional online routing beat frozen and matched placebo "
            "with no preregistered harm",
        )
    if completed:
        return (
            "null",
            "complete_null: the prospective comparison completed without all positive-credit gates",
        )
    if failed_checks & disqualifying:
        return (
            "disqualified",
            "complete_disqualified: leakage, cross-arm state, missing rows, or substitute detected",
        )
    return (
        "partial",
        "complete_partial: lifecycle or attribution evidence is incomplete",
    )


def _empty_metrics() -> JsonDict:
    empty = _nested_counts()
    return {
        "writes_by_arm_order": deepcopy(empty),
        "later_reads_by_arm_order": deepcopy(empty),
        "action_changes_by_arm_order": deepcopy(empty),
        "component_action_attribution": {},
        "held_future_utility_by_arm_order": {},
        "online_minus_frozen_order_effects": {},
        "online_minus_frozen_lcb": 0.0,
        "online_minus_placebo_order_effects": {},
        "hard_case_harm_by_arm_order": {},
        "retention_by_arm_order": {},
        "action_support_by_arm_order": {},
    }


def _base_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_hash: str | None,
    manifest: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_blocked_online_constraint_routing_ab",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hash": source_hash,
        "frozen_manifest": deepcopy(dict(manifest)),
        "arm_definitions": deepcopy(ARM_DEFINITIONS),
        "component_definitions": deepcopy(COMPONENT_DEFINITIONS),
        "transaction_capacity": deepcopy(TRANSACTION_CAPACITY),
        "rows": [],
        "transaction_receipts": [],
        **_empty_metrics(),
        "future_feature_violations": [],
        "active_event_write_violations": [],
        "compositional_csl_completed": False,
        "gate_check_summary": deepcopy(dict(gate_summary)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_online_constraint_routing_ab: owned precondition failed",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(
    *,
    source_path: Path | str = REPO_ROOT / SOURCE_RELATIVE_PATH,
    state_root: Path | str | None = None,
    run_date: str = RUN_DATE,
    precondition_overrides: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Run all frozen order-event-arm cells or return one complete block."""

    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("run date must use YYYYMMDD")
    started = time.monotonic()
    if state_root is None:
        with tempfile.TemporaryDirectory(prefix="carnot-exp6791-") as directory:
            return run_experiment(
                source_path=source_path,
                state_root=directory,
                run_date=run_date,
                precondition_overrides=precondition_overrides,
                duration_s=duration_s,
            )
    source_path = Path(source_path)
    source: JsonDict = {}
    if source_path.is_file():
        try:
            source = _load_source(source_path)
        except (OSError, ValueError, json.JSONDecodeError):
            source = {}
    manifest = _frozen_manifest(source)
    preconditions = evaluate_preconditions(
        source,
        source_path=source_path,
        state_root=Path(state_root),
        overrides=precondition_overrides,
    )
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact = _base_artifact(
        run_date=run_date,
        duration_s=elapsed,
        source_hash=sha256_file(source_path),
        manifest=manifest,
        gate_summary=preconditions,
    )
    if preconditions["all_passed"] is not True:
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
        return artifact

    event_by_id = {str(row["event_id"]): row for row in source["frozen_manifest"]["events"]}
    source_row_by_key = {
        (str(row["order_id"]), str(row["event_id"])): row for row in source["rows"]
    }
    all_rows: list[JsonDict] = []
    all_transactions: list[JsonDict] = []
    active_violations: list[JsonDict] = []
    for order in source["order_definitions"]:
        order_rows, transactions, violations = _run_order(
            order=order,
            event_by_id=event_by_id,
            source_row_by_key=source_row_by_key,
            state_root=Path(state_root),
        )
        all_rows.extend(order_rows)
        all_transactions.extend(transactions)
        active_violations.extend(violations)
    _annotate_later_use(all_rows, all_transactions)
    metrics = reduce_evidence(all_rows, all_transactions)
    future_violations = audit_future_features(all_rows, all_transactions)
    artifact.update(
        {
            "status": "complete_online_constraint_routing_ab",
            "rows": all_rows,
            "transaction_receipts": all_transactions,
            **metrics,
            "future_feature_violations": future_violations,
            "active_event_write_violations": active_violations,
        }
    )
    artifact["gate_check_summary"] = {**preconditions, "preconditions_pass": True}
    checks = completion_checks(artifact)
    artifact["gate_check_summary"]["completion_checks"] = checks
    artifact["gate_check_summary"]["completion_failures"] = [
        _gate(name, True, value) for name, value in checks.items() if value is not True
    ]
    completed = all(checks.values())
    artifact["compositional_csl_completed"] = completed
    failed = {name for name, value in checks.items() if value is not True}
    positive = completed and all(positive_credit_checks(artifact).values())
    artifact["verdict_class"], artifact["honest_verdict"] = terminal_verdict(
        completed=completed,
        failed_checks=failed,
        positive=positive,
    )
    artifact["duration_s"] = round(
        float(duration_s) if duration_s is not None else time.monotonic() - started, 6
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - generated evidence is validated before this point.
        raise ValueError("; ".join(errors))
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the frozen protocol, rows, receipts, aggregates, and terminal result."""

    material = {
        key: artifact.get(key)
        for key in REQUIRED_ARTIFACT_FIELDS
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(material)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return all closed schema and row-derived consistency errors."""

    errors = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field principle coverage mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random seed mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict class is outside the closed enum")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest verdict lacks a terminal prefix")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    rows = artifact.get("rows", [])
    transactions = artifact.get("transaction_receipts", [])
    if artifact.get("verdict_class") == "blocked":
        if rows != [] or transactions != []:
            errors.append("blocked artifact contains prospective evidence")
        if artifact.get("status") != "complete_blocked_online_constraint_routing_ab":
            errors.append("blocked artifact status mismatch")
    elif rows:
        reduced = reduce_evidence(rows, transactions)
        if any(artifact.get(field) != reduced[field] for field in ROW_DERIVED_FIELDS):
            errors.append("row-derived metrics mismatch")
        checks = completion_checks(artifact)
        if artifact.get("compositional_csl_completed") is not all(checks.values()):
            errors.append("completion checks mismatch")
        if all(checks.values()):
            expected_class = (
                "positive" if all(positive_credit_checks(artifact).values()) else "null"
            )
            if artifact.get("verdict_class") != expected_class:
                errors.append("row-derived verdict mismatch")
    else:
        errors.append("non-blocked artifact has no rows")
    return errors


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and publish one terminal artifact through an atomic rename."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    target = Path(path)
    data = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    _atomic_write(target, data)
    return {
        "path": str(target),
        "atomic_rename": True,
        "sha256": sha256_file(target),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the fixed date, run the comparison, and publish its result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run_experiment(run_date=args.date)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    write_artifact(output, artifact)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper calls main.
    raise SystemExit(main())
