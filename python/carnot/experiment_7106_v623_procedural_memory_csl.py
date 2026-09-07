"""Compare five external-memory treatments on the sealed Exp7105 stream.

The candidate policy is deterministic and has no trainable weights. Exact
feedback opens only after each decision is sealed. This makes any later change
come from bounded external memory instead of hidden model state.

Spec refs: REQ-CL-7106 and SCENARIO-CL-7106-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_7105_v623_exact_constraint_stream as exp7105


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7106
SCHEMA = "carnot.exp7106.v623_procedural_memory_csl.v1"
RUN_DATE = "20260907"
RANDOM_SEED = 7_106_202_609_07
EVENT_COUNT = 144
ROW_COUNT = EVENT_COUNT * 5
RECORD_SLOT_BYTES = 16_384
CONTEXT_BUDGET_BYTES = 4_096
RETRIEVAL_SLOT_COUNT = 1
DECISION_BUDGET = 1
VALIDATION_BUDGET = 1
MODEL_WEIGHTS_CHANGED = False
INFERENCE_SUBSTRATE = (
    "deterministic prospective candidate policy with verifier-signed bounded external memory"
)
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
EVICTION_POLICY = "fifo_oldest_write"
EMPTY_MEMORY_HASH = "sha256:" + hashlib.sha256(b"[]").hexdigest()

ARMS = (
    "delayed_procedural",
    "raw_trace",
    "equal_context_replay",
    "write_while_deciding",
    "no_memory",
)
PERSISTENT_ARMS = {"delayed_procedural", "raw_trace", "write_while_deciding"}
STATELESS_ARMS = {"equal_context_replay", "no_memory"}
FROZEN_PRIMARY_COMPARISONS = (
    "delayed_procedural_vs_raw_trace",
    "delayed_procedural_vs_equal_context_replay",
    "delayed_procedural_vs_write_while_deciding",
    "delayed_procedural_vs_no_memory",
)
FROZEN_ARM_DEFINITIONS = (
    {
        "arm": "delayed_procedural",
        "representation": "abstract_procedure",
        "persistent_state": True,
        "write_timing": "atomic_after_exact_feedback_between_events",
    },
    {
        "arm": "raw_trace",
        "representation": "bounded_raw_trajectory",
        "persistent_state": True,
        "write_timing": "atomic_after_exact_feedback_between_events",
    },
    {
        "arm": "equal_context_replay",
        "representation": "one_prior_feedback_replay",
        "persistent_state": False,
        "write_timing": "none",
    },
    {
        "arm": "write_while_deciding",
        "representation": "provisional_self_generated_procedure",
        "persistent_state": True,
        "write_timing": "atomic_provisional_before_decision_then_exact_resolution",
    },
    {
        "arm": "no_memory",
        "representation": "neutral_context",
        "persistent_state": False,
        "write_timing": "none",
    },
)

DEFAULT_UPSTREAM_ARTIFACT_PATH = Path("results/experiment_7105_v623_exact_constraint_stream.json")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7106_v623_procedural_memory_csl.json")
SOURCE_PATHS = (
    Path("python/carnot/experiment_7105_v623_exact_constraint_stream.py"),
    Path("python/carnot/experiment_7106_v623_procedural_memory_csl.py"),
    Path("scripts/experiments/experiment_7105_v623_exact_constraint_stream.py"),
    Path("scripts/experiments/experiment_7106_v623_procedural_memory_csl.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("python/carnot/learning/constraint_policy_store.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "upstream_gate_receipt",
    "stream_hash",
    "decision_view_hash",
    "label_view_hash",
    "frozen_arm_definitions",
    "frozen_capacity_schedule",
    "frozen_primary_comparisons",
    "equal_information_receipts",
    "rows",
    "per_event_results",
    "event_rows",
    "group_rows",
    "family_rows",
    "hardness_rows",
    "reuse_decoy_rows",
    "slice_rows",
    "capacity_rows",
    "retrieval_rows",
    "decision_rows",
    "feedback_rows",
    "update_rows",
    "transaction_rows",
    "memory_hash_rows",
    "eviction_rows",
    "protected_retention_rows",
    "paired_delta_rows",
    "confidence_interval_rows",
    "negative_transfer_rows",
    "model_weights_changed",
    "procedural_memory_comparison_complete_score",
    "procedural_memory_value_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

_PRINCIPLES = {
    "schema": "A versioned schema makes incompatible comparison readers fail closed.",
    "experiment_id": "A stable experiment ID prevents another run from supplying these claims.",
    "run_date": "The fixed date binds the prospective plan to its declared execution window.",
    "field_principles": "Field-level reasons keep every evidence requirement scientifically reviewable.",
    "preconditions_checked": "Explicit preflight evidence prevents a broken stream from becoming a measurement.",
    "inference_substrate": "The substrate identifies deterministic policy work and excludes an undeclared model call.",
    "inference_substrate_class": "The closed compute class separates no-model work from a blocked no-run.",
    "execution_venue": "The venue prevents local host evidence from becoming a remote or hardware claim.",
    "duration_s": "Measured wall time shows that the full comparison actually executed.",
    "source_artifact_hashes": "Source hashes expose code or evidence drift between run and audit.",
    "upstream_gate_receipt": "The upstream receipt proves that the sealed stream authorized this comparison.",
    "stream_hash": "The stream hash binds every arm to one immutable chronology.",
    "decision_view_hash": "The decision seal binds the information visible before outcomes.",
    "label_view_hash": "The label seal binds exact feedback without leaking it into decisions.",
    "frozen_arm_definitions": "Frozen treatments prevent outcome-driven changes to memory behavior.",
    "frozen_capacity_schedule": "A prior capacity schedule prevents a favorable post-run memory limit.",
    "frozen_primary_comparisons": "Prior contrasts prevent selective reporting against weak controls.",
    "equal_information_receipts": "Matched receipts isolate external memory from resource and chronology changes.",
    "rows": "Complete arm-event rows preserve the denominator for independent reduction.",
    "per_event_results": "Paired event summaries keep wins and losses aligned to the same decisions.",
    "event_rows": "An explicit event table lets generic row linters inspect measured outcomes.",
    "group_rows": "Every-group estimates reveal local failures hidden by a pooled mean.",
    "family_rows": "Family estimates test transfer across distinct constraint mechanisms.",
    "hardness_rows": "Hard-versus-ordinary rows expose concentrated negative transfer.",
    "reuse_decoy_rows": "Reuse-versus-decoy rows separate useful transfer from indiscriminate copying.",
    "slice_rows": "Early, middle, and late rows expose learning speed and later value.",
    "capacity_rows": "Capacity slices prove value was not purchased with extra state.",
    "retrieval_rows": "Retrieval evidence shows what external information reached each decision.",
    "decision_rows": "Sealed decision evidence establishes the causal boundary before feedback.",
    "feedback_rows": "Signed outcomes show exactly what became available after each decision.",
    "update_rows": "Proposals and validation results distinguish learning attempts from silent state.",
    "transaction_rows": "Parent-child receipts make atomic update timing auditable.",
    "memory_hash_rows": "Pre and post hashes reveal hidden persistence and state changes.",
    "eviction_rows": "Eviction evidence proves the common bounded-capacity policy executed.",
    "protected_retention_rows": "Protected probes make forgetting a falsifiable gate.",
    "paired_delta_rows": "Paired deltas retain adverse events instead of comparing unpaired averages.",
    "confidence_interval_rows": "Predeclared intervals quantify whether later gains clear sampling uncertainty.",
    "negative_transfer_rows": "Per-group hard deltas prevent hard-case regressions from being pooled away.",
    "model_weights_changed": "A false value isolates external memory from online gradient updates.",
    "procedural_memory_comparison_complete_score": "Completion requires every arm, event, aggregate, and receipt.",
    "procedural_memory_value_ready_score": "Value requires later gains, hard safety, capacity, and retention together.",
    "random_seed": "A fixed seed binds every deterministic ordering and policy tie break.",
    "reproducibility_checksum": "A content checksum detects silent changes to timing-free evidence.",
    "gate_check_summary": "Exact diagnostics distinguish a blocked prerequisite from a scientific null.",
    "verifier_is_oracle": "False states that exact feedback signs updates but does not select current candidates.",
    "verdict_class": "A closed terminal class prevents a null result from being described as positive.",
    "honest_verdict": "A terminal-prefix headline lets automation classify the measured result safely.",
}
FIELD_PRINCIPLES = {field: _PRINCIPLES[field] for field in REQUIRED_ARTIFACT_FIELDS}


def canonical_json(value: Any) -> bytes:
    """Return stable JSON bytes so equal memory states have equal identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Prefix SHA-256 identities so hashes cannot be mistaken for raw labels."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one structured value with the experiment's canonical encoding."""

    return sha256_bytes(canonical_json(value))


def sha256_path(path: Path) -> str | None:
    """Hash a readable file and return no identity when the file is absent."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read the immutable views without changing their storage order."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def gate_check(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Keep expected and observed values so a blocked run names its exact cause."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report all checks and copy the first failure into stable top-level keys."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "checks": [dict(row) for row in checks],
        "failed_check": None if failed is None else failed.get("check"),
        "expected_value": None if failed is None else failed.get("expected_value"),
        "observed_value": None if failed is None else failed.get("observed_value"),
    }


def _resolve(repo_root: Path, value: str | Path) -> Path:
    """Resolve artifact-relative evidence paths against the selected checkout."""

    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_object(path: Path) -> JsonDict:
    """Return one object or an empty value so preflight can fail with evidence."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_preconditions(
    repo_root: Path, upstream_artifact_path: Path
) -> tuple[list[JsonDict], JsonDict]:
    """Check every sealed-stream dependency before the first policy decision."""

    upstream = _load_object(upstream_artifact_path)
    decision_path = _resolve(repo_root, str(upstream.get("decision_view_path", "missing")))
    label_path = _resolve(repo_root, str(upstream.get("label_view_path", "missing")))
    stream_path = _resolve(repo_root, str(upstream.get("stream_path", "missing")))
    decisions = read_jsonl(decision_path) if decision_path.is_file() else []
    labels = read_jsonl(label_path) if label_path.is_file() else []
    decision_ids = {row.get("event_id") for row in decisions}
    label_ids = {row.get("event_id") for row in labels}
    exact_witnesses = len(labels) == EVENT_COUNT and all(
        row.get("exact_label") and isinstance(row.get("witness"), dict) for row in labels
    )
    expected_capacity = [dict(row) for row in exp7105.FROZEN_CAPACITY_SCHEDULE]
    expected_slices = [dict(row) for row in exp7105.FROZEN_SLICE_DEFINITIONS]
    upstream_validation = (
        exp7105.validate_artifact(upstream, repo_root=repo_root, check_files=True)
        if upstream
        else ["missing_upstream_artifact"]
    )
    checks = [
        gate_check(
            "exact_constraint_stream_ready_score",
            1,
            upstream.get("exact_constraint_stream_ready_score"),
        ),
        gate_check("event_count", EVENT_COUNT, upstream.get("event_count")),
        gate_check("decision_view_row_count", EVENT_COUNT, len(decisions)),
        gate_check(
            "decision_view_hash",
            upstream.get("decision_view_hash"),
            sha256_path(decision_path),
        ),
        gate_check(
            "group_count_at_least_12",
            ">=12",
            len({row.get("group_id") for row in decisions}),
            len({row.get("group_id") for row in decisions}) >= 12,
        ),
        gate_check(
            "separate_label_view", True, label_path != decision_path and label_path.is_file()
        ),
        gate_check("label_view_row_count", EVENT_COUNT, len(labels)),
        gate_check("label_view_hash", upstream.get("label_view_hash"), sha256_path(label_path)),
        gate_check("exact_witnesses", True, exact_witnesses),
        gate_check("decision_label_identity", sorted(decision_ids), sorted(label_ids)),
        gate_check(
            "frozen_capacity_schedule",
            expected_capacity,
            upstream.get("frozen_capacity_schedule"),
        ),
        gate_check(
            "frozen_slice_definitions",
            expected_slices,
            upstream.get("frozen_slice_definitions"),
        ),
        gate_check("stream_hash", upstream.get("stream_hash"), sha256_path(stream_path)),
        gate_check("upstream_artifact_validation", [], upstream_validation),
        gate_check("fixed_policy_hash", FIXED_POLICY_HASH, FIXED_POLICY_HASH),
    ]
    return checks, upstream


def _capacity_for(index: int, schedule: Sequence[Mapping[str, Any]]) -> int:
    """Read the item limit from the frozen slice that contains this event."""

    slice_id = "early" if index < 48 else "middle" if index < 96 else "late"
    return int(next(row["memory_capacity"] for row in schedule if row["slice_id"] == slice_id))


def _slice_for(index: int) -> str:
    """Map an event to the predeclared early, middle, or late interval."""

    return "early" if index < 48 else "middle" if index < 96 else "late"


class BoundedMemory:
    """Keep one arm's state private and replace it through complete transitions."""

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.records: list[JsonDict] = []
        self.capacity_items = 0

    @property
    def state_hash(self) -> str:
        """Identify the complete ordered state, including its stored content."""

        return sha256_json(self.records)

    def snapshot(self) -> JsonDict:
        """Return a detached copy so an active decision cannot mutate the store."""

        return {"records": deepcopy(self.records), "state_hash": self.state_hash}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a provisional write when exact feedback rejects its content."""

        self.records = deepcopy(list(snapshot["records"]))

    def commit(self, record: Mapping[str, Any], capacity_items: int) -> JsonDict:
        """Build the next state fully, then publish it as one in-memory swap."""

        parent_hash = self.state_hash
        self.capacity_items = capacity_items
        next_records = [
            deepcopy(row)
            for row in self.records
            if row.get("memory_key") != record.get("memory_key")
        ]
        next_records.append(deepcopy(dict(record)))
        evicted: list[str] = []
        while len(next_records) > capacity_items:
            evicted.append(str(next_records.pop(0)["memory_id"]))
        self.records = next_records
        actual_bytes = len(canonical_json(self.records))
        if actual_bytes > capacity_items * RECORD_SLOT_BYTES:
            raise ValueError("memory_byte_capacity_exceeded")
        return {
            "committed": True,
            "atomic": True,
            "parent_hash": parent_hash,
            "child_hash": self.state_hash,
            "evicted_ids": evicted,
            "item_count": len(self.records),
            "charged_bytes": len(self.records) * RECORD_SLOT_BYTES,
            "actual_bytes": actual_bytes,
        }

    def retrieve(self, structure_key: str) -> JsonDict | None:
        """Return the newest matching record without changing recency or state."""

        return next(
            (
                deepcopy(row)
                for row in reversed(self.records)
                if row.get("scope_key") == structure_key
            ),
            None,
        )


def _neutral_record() -> JsonDict:
    """Fill the common retrieval slot without supplying historical information."""

    return {
        "memory_id": "neutral",
        "representation": "neutral",
        "source_event_id": None,
        "source_chronology_index": None,
    }


def _candidate_index(label: str) -> int:
    """Read the public candidate ordinal from one exact signed label."""

    return int(label.rsplit(":", 1)[1])


def _abstract_record(event: Mapping[str, Any], label: str) -> JsonDict:
    """Convert one outcome into a reusable operation, not a copied trajectory."""

    visible = event["decision_visible_input"]
    return {
        "memory_key": f"procedure:{event['structure_key']}",
        "memory_id": f"procedure:{event['event_id']}",
        "representation": "abstract_procedure",
        "scope_key": event["structure_key"],
        "constraint_family": event["constraint_family"],
        "template": visible["template"],
        "operation": "evaluate_declared_visible_constraints",
        "source_event_id": event["event_id"],
        "source_chronology_index": event["chronology_index"],
        "source_label_hash": sha256_json(label),
        "provenance_complete": True,
    }


def _raw_record(event: Mapping[str, Any], label: str, witness: Mapping[str, Any]) -> JsonDict:
    """Store the detailed source episode so transfer requires a payload match."""

    valid_payload = witness["valid_candidate_payload"]
    return {
        "memory_key": f"trace:{event['event_id']}",
        "memory_id": f"trace:{event['event_id']}",
        "representation": "raw_trace",
        "scope_key": event["structure_key"],
        "source_event_id": event["event_id"],
        "source_chronology_index": event["chronology_index"],
        "decision_visible_input": deepcopy(event["decision_visible_input"]),
        "candidate_set": deepcopy(event["candidate_set"]),
        "exact_label": label,
        "valid_payload_hash": sha256_json(valid_payload),
        "witness": deepcopy(dict(witness)),
        "provenance_complete": True,
    }


def _provisional_record(event: Mapping[str, Any], chosen_index: int) -> JsonDict:
    """Represent unverified self-writing as the policy's own current guess."""

    return {
        "memory_key": f"procedure:{event['structure_key']}",
        "memory_id": f"provisional:{event['event_id']}",
        "representation": "provisional_procedure",
        "scope_key": event["structure_key"],
        "chosen_index": chosen_index,
        "source_event_id": event["event_id"],
        "source_chronology_index": event["chronology_index"],
        "provenance_complete": False,
    }


def _replay_record(previous: Mapping[str, Any] | None) -> JsonDict:
    """Build one transient prior-outcome replay without retaining mutable state."""

    if previous is None:
        return _neutral_record()
    return {
        "memory_id": f"replay:{previous['event_id']}",
        "representation": "equal_context_replay",
        "source_event_id": previous["event_id"],
        "source_chronology_index": previous["chronology_index"],
        "chosen_index": _candidate_index(str(previous["exact_label"])),
    }


def _procedure_accepts(event: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    """Execute the learned visible operation without consulting a hidden label."""

    family = str(event["constraint_family"])
    problem = event["decision_visible_input"]["problem"]
    if family == "sat":
        values = list(payload["assignment"])
        return all(
            any(values[abs(literal) - 1] == (literal > 0) for literal in clause)
            for clause in problem["clauses"]
        )
    if family == "graph_coloring":
        colors = list(payload["colors"])
        return all(0 <= color < int(problem["n_colors"]) for color in colors) and all(
            colors[left] != colors[right] for left, right in problem["edges"]
        )
    if family == "arithmetic":
        left, right = int(problem["a"]), int(problem["b"])
        targets = {
            "triangle": left * right + left,
            "delta": left * right - right,
            "box": (left + right) * 2,
        }
        return int(payload["value"]) == targets[str(problem["rule_name"])]
    trace = list(payload["trace"])
    operator = str(problem["temporal_operator"])
    signal = str(problem["signal"])
    if operator == "always":
        return all(bool(step.get(signal, False)) for step in trace)
    if operator == "eventually":
        return any(bool(step.get(signal, False)) for step in trace)
    goal_positions = [i for i, step in enumerate(trace) if bool(step.get(signal, False))]
    return bool(goal_positions) and all(
        bool(trace[i].get(str(problem["guard_signal"]), False)) for i in range(goal_positions[0])
    )


def fixed_candidate_policy(event: Mapping[str, Any], retrieved: Mapping[str, Any]) -> JsonDict:
    """Select once from the supplied candidates; only the external record can vary."""

    candidates = list(event["candidate_set"])
    representation = retrieved.get("representation")
    chosen_index = 0
    score = 0.5
    if representation == "abstract_procedure":
        matches = [
            index
            for index, candidate in enumerate(candidates)
            if _procedure_accepts(event, candidate["payload"])
        ]
        if len(matches) == 1:
            chosen_index, score = matches[0], 0.9
    elif representation == "raw_trace":
        target_hash = retrieved.get("valid_payload_hash")
        matches = [
            index
            for index, candidate in enumerate(candidates)
            if sha256_json(candidate["payload"]) == target_hash
        ]
        if len(matches) == 1:
            chosen_index, score = matches[0], 0.75
    elif representation in {"equal_context_replay", "provisional_procedure"}:
        chosen_index = int(retrieved.get("chosen_index", 0)) % len(candidates)
        score = 0.6
    return {
        "candidate_id": candidates[chosen_index]["candidate_id"],
        "candidate_index": chosen_index,
        "score": score,
        "policy_hash": FIXED_POLICY_HASH,
    }


FIXED_POLICY_HASH = sha256_bytes(
    b"exp7106-fixed-policy-v1:neutral0:procedure-visible-constraints:raw-exact-payload:replay-index"
)


def _render_context(record: Mapping[str, Any]) -> tuple[str, int]:
    """Pad every arm to one exact byte budget without changing record content."""

    compact = {
        key: record.get(key)
        for key in (
            "memory_id",
            "representation",
            "source_event_id",
            "source_chronology_index",
            "operation",
            "chosen_index",
            "valid_payload_hash",
        )
        if key in record
    }
    data = canonical_json(compact)
    if len(data) > CONTEXT_BUDGET_BYTES:
        raise ValueError("context_budget_exceeded")
    rendered = data + (b" " * (CONTEXT_BUDGET_BYTES - len(data)))
    return rendered.decode("ascii"), len(rendered)


def _source_hashes(repo_root: Path, upstream_artifact_path: Path) -> JsonDict:
    """Bind the run to its implementation, transaction support, and sealed input."""

    rows = {str(path): sha256_path(repo_root / path) for path in SOURCE_PATHS}
    rows[str(upstream_artifact_path)] = sha256_path(upstream_artifact_path)
    return rows


def _label_map(repo_root: Path, upstream: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Index the sealed sidecar by ID while preserving release through loop order."""

    path = _resolve(repo_root, str(upstream["label_view_path"]))
    return {str(row["event_id"]): row for row in read_jsonl(path)}


def _decision_rows(repo_root: Path, upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Load only the label-free view used to make prospective decisions."""

    return read_jsonl(_resolve(repo_root, str(upstream["decision_view_path"])))


def _projection_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Project complete event rows into the required causal evidence tables."""

    return {
        "retrieval_rows": [
            {
                "row_key": row["row_key"],
                "retrieved_items": deepcopy(row["retrieved_items"]),
                "retrieval_slot_count": row["retrieval_slot_count"],
                "context_bytes": row["context_bytes"],
            }
            for row in rows
        ],
        "decision_rows": [
            {
                "row_key": row["row_key"],
                "decision": deepcopy(row["decision"]),
                "decision_sequence": row["decision_sequence"],
                "decision_seal": row["decision_seal"],
                "exact_feedback_visible_at_decision": row["exact_feedback_visible_at_decision"],
            }
            for row in rows
        ],
        "feedback_rows": [
            {
                "row_key": row["row_key"],
                "feedback_sequence": row["feedback_sequence"],
                "exact_post_decision_label": row["exact_post_decision_label"],
                "witness": deepcopy(row["witness"]),
            }
            for row in rows
        ],
        "update_rows": [
            {
                "row_key": row["row_key"],
                "proposed_update": deepcopy(row["proposed_update"]),
                "validation_result": deepcopy(row["validation_result"]),
            }
            for row in rows
        ],
        "transaction_rows": [
            {"row_key": row["row_key"], **deepcopy(row["commit_record"])} for row in rows
        ],
        "memory_hash_rows": [
            {
                "row_key": row["row_key"],
                "pre_event_memory_hash": row["pre_event_memory_hash"],
                "pre_decision_memory_hash": row["pre_decision_memory_hash"],
                "post_commit_memory_hash": row["post_commit_memory_hash"],
            }
            for row in rows
        ],
        "eviction_rows": [
            {
                "row_key": row["row_key"],
                "evicted_ids": list(row["evicted_ids"]),
                "eviction_policy": row["eviction_policy"],
                "capacity_items": row["capacity_items"],
                "capacity_bytes": row["capacity_bytes"],
            }
            for row in rows
        ],
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]], field: str, values: Sequence[str]
) -> list[JsonDict]:
    """Compute one complete arm-by-stratum table, including adverse cells."""

    result: list[JsonDict] = []
    for value in values:
        for arm in ARMS:
            selected = [row for row in rows if row[field] == value and row["arm"] == arm]
            correct = sum(bool(row["correct"]) for row in selected)
            result.append(
                {
                    field: value,
                    "arm": arm,
                    "event_count": len(selected),
                    "correct_count": correct,
                    "accuracy": round(correct / len(selected), 6) if selected else None,
                }
            )
    return result


def _binomial_sign_p(wins: int, losses: int) -> float:
    """Return the exact one-sided sign-test tail on discordant paired events."""

    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(wins, discordant + 1))
    return round(tail / (2**discordant), 12)


def _paired_tables(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Measure the predeclared later-event contrast with a fixed Hoeffding interval."""

    paired: list[JsonDict] = []
    intervals: list[JsonDict] = []
    later = [row for row in rows if int(row["chronology_index"]) >= 48]
    by_identity = {
        (int(row["chronology_index"]), str(row["arm"])): bool(row["correct"]) for row in later
    }
    for comparison in FROZEN_PRIMARY_COMPARISONS:
        control = comparison.removeprefix("delayed_procedural_vs_")
        deltas = [
            int(by_identity[(index, "delayed_procedural")]) - int(by_identity[(index, control)])
            for index in range(48, EVENT_COUNT)
            if (index, "delayed_procedural") in by_identity and (index, control) in by_identity
        ]
        wins = sum(delta > 0 for delta in deltas)
        losses = sum(delta < 0 for delta in deltas)
        mean = sum(deltas) / len(deltas) if deltas else 0.0
        radius = math.sqrt(math.log(40.0) / (2 * len(deltas))) if deltas else 1.0
        lower, upper = max(-1.0, mean - radius), min(1.0, mean + radius)
        paired.append(
            {
                "comparison": comparison,
                "slice": "middle_and_late",
                "event_count": len(deltas),
                "wins": wins,
                "losses": losses,
                "ties": len(deltas) - wins - losses,
                "mean_delta": round(mean, 6),
                "exact_one_sided_sign_p": _binomial_sign_p(wins, losses),
            }
        )
        intervals.append(
            {
                "comparison": comparison,
                "method": "predeclared_hoeffding_95_percent",
                "confidence_level": 0.95,
                "event_count": len(deltas),
                "lower": round(lower, 6),
                "upper": round(upper, 6),
            }
        )
    return paired, intervals


def reduce_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute every scientific table from the complete paired event rows."""

    group_ids = list(exp7105.FROZEN_GROUP_IDS)
    family_ids = list(exp7105.FROZEN_FAMILY_IDS)
    group_rows = _aggregate(rows, "group_id", group_ids)
    family_rows = _aggregate(rows, "constraint_family", family_ids)
    normalized = [
        {**dict(row), "hardness_class": "hard" if row["hardness_stratum"] == "hard" else "ordinary"}
        for row in rows
    ]
    hardness_rows = _aggregate(normalized, "hardness_class", ("ordinary", "hard"))
    reuse_decoy_rows = _aggregate(rows, "reuse_or_decoy_status", ("reusable", "decoy"))
    slice_rows = _aggregate(rows, "slice_id", ("early", "middle", "late"))
    capacity_values = [int(row["memory_capacity"]) for row in exp7105.FROZEN_CAPACITY_SCHEDULE]
    capacity_rows = _aggregate(rows, "capacity_items", capacity_values)  # type: ignore[arg-type]
    protected_retention_rows: list[JsonDict] = []
    for group_id in exp7105.FROZEN_PROTECTED_GROUP_IDS:
        for arm in ARMS:
            selected = [
                row
                for row in rows
                if row["group_id"] == group_id
                and row["arm"] == arm
                and row["protected_retention_probe"] is True
            ]
            correct = sum(bool(row["correct"]) for row in selected)
            protected_retention_rows.append(
                {
                    "group_id": group_id,
                    "arm": arm,
                    "probe_count": len(selected),
                    "correct_count": correct,
                    "retention_passed": bool(selected) and correct == len(selected),
                }
            )
    hard_by_group_arm: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["hardness_stratum"] == "hard":
            hard_by_group_arm[(str(row["group_id"]), str(row["arm"]))].append(row)
    negative_transfer_rows: list[JsonDict] = []
    for group_id in group_ids:
        delayed = hard_by_group_arm[(group_id, "delayed_procedural")]
        delayed_rate = sum(bool(row["correct"]) for row in delayed) / len(delayed) if delayed else 0
        for comparison in FROZEN_PRIMARY_COMPARISONS:
            control = comparison.removeprefix("delayed_procedural_vs_")
            control_rows = hard_by_group_arm[(group_id, control)]
            control_rate = (
                sum(bool(row["correct"]) for row in control_rows) / len(control_rows)
                if control_rows
                else 0
            )
            delta = delayed_rate - control_rate
            negative_transfer_rows.append(
                {
                    "group_id": group_id,
                    "comparison": comparison,
                    "hard_event_count": len(delayed),
                    "delayed_accuracy": round(delayed_rate, 6),
                    "control_accuracy": round(control_rate, 6),
                    "delta": round(delta, 6),
                    "regression": delta < 0,
                }
            )
    paired, intervals = _paired_tables(rows)
    per_event_results: list[JsonDict] = []
    for index in range(EVENT_COUNT):
        selected = [row for row in rows if row["chronology_index"] == index]
        per_event_results.append(
            {
                "chronology_index": index,
                "event_id": selected[0]["event_id"] if selected else None,
                "arm_results": {
                    str(row["arm"]): {
                        "candidate_id": row["decision"]["candidate_id"],
                        "correct": row["correct"],
                    }
                    for row in selected
                },
            }
        )
    return {
        "per_event_results": per_event_results,
        "group_rows": group_rows,
        "family_rows": family_rows,
        "hardness_rows": hardness_rows,
        "reuse_decoy_rows": reuse_decoy_rows,
        "slice_rows": slice_rows,
        "capacity_rows": capacity_rows,
        "protected_retention_rows": protected_retention_rows,
        "paired_delta_rows": paired,
        "confidence_interval_rows": intervals,
        "negative_transfer_rows": negative_transfer_rows,
    }


def _empty_tables() -> JsonDict:
    """Keep blocked artifacts schema-complete without inventing measurements."""

    names = (
        "equal_information_receipts",
        "rows",
        "per_event_results",
        "event_rows",
        "group_rows",
        "family_rows",
        "hardness_rows",
        "reuse_decoy_rows",
        "slice_rows",
        "capacity_rows",
        "retrieval_rows",
        "decision_rows",
        "feedback_rows",
        "update_rows",
        "transaction_rows",
        "memory_hash_rows",
        "eviction_rows",
        "protected_retention_rows",
        "paired_delta_rows",
        "confidence_interval_rows",
        "negative_transfer_rows",
    )
    return {name: [] for name in names}


def _base_artifact(
    upstream: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    upstream_artifact_path: Path,
    run_date: str,
    duration_s: float,
) -> JsonDict:
    """Build common provenance without claiming that a comparison completed."""

    passed = all(row.get("passed") is True for row in checks)
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS if passed else "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "source_artifact_hashes": _source_hashes(repo_root, upstream_artifact_path),
        "upstream_gate_receipt": {
            "experiment_id": upstream.get("experiment_id", 7105),
            "artifact_path": str(upstream_artifact_path),
            "artifact_hash": sha256_path(upstream_artifact_path),
            "gate_field": "exact_constraint_stream_ready_score",
            "expected_value": 1,
            "observed_value": upstream.get("exact_constraint_stream_ready_score"),
            "passed": upstream.get("exact_constraint_stream_ready_score") == 1,
        },
        "stream_hash": upstream.get("stream_hash"),
        "decision_view_hash": upstream.get("decision_view_hash"),
        "label_view_hash": upstream.get("label_view_hash"),
        "frozen_arm_definitions": deepcopy(list(FROZEN_ARM_DEFINITIONS)),
        "frozen_capacity_schedule": deepcopy(
            upstream.get("frozen_capacity_schedule", list(exp7105.FROZEN_CAPACITY_SCHEDULE))
        ),
        "frozen_primary_comparisons": list(FROZEN_PRIMARY_COMPARISONS),
        "model_weights_changed": MODEL_WEIGHTS_CHANGED,
        "procedural_memory_comparison_complete_score": 0,
        "procedural_memory_value_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_procedural_memory_precondition",
        **_empty_tables(),
    }


def _stable_value(value: Any) -> Any:
    """Remove wall-clock fields so reproducibility tests compare scientific evidence."""

    if isinstance(value, dict):
        return {
            key: _stable_value(item)
            for key, item in value.items()
            if key not in {"duration_s", "latency_ms", "reproducibility_checksum"}
        }
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash timing-free evidence so another host can reproduce the causal result."""

    return sha256_json(_stable_value(dict(artifact)))


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
    *,
    run_date: str,
    duration_s: float,
    upstream_artifact_path: Path,
    repo_root: Path | None = None,
) -> JsonDict:
    """Publish a complete terminal no-run record for any failed precondition."""

    root = repo_root or Path(__file__).resolve().parents[2]
    artifact = _base_artifact(
        upstream,
        checks,
        repo_root=root,
        upstream_artifact_path=upstream_artifact_path,
        run_date=run_date,
        duration_s=duration_s,
    )
    failed = artifact["gate_check_summary"]["failed_check"] or "unknown_precondition"
    artifact["honest_verdict"] = f"complete_blocked_procedural_memory:{failed}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _run_rows(
    decisions: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Execute each sealed event in order with private state for every treatment."""

    stores = {arm: BoundedMemory(arm) for arm in PERSISTENT_ARMS}
    released_feedback: list[JsonDict] = []
    rows: list[JsonDict] = []
    for event in decisions:
        index = int(event["chronology_index"])
        capacity_items = _capacity_for(index, schedule)
        for arm_index, arm in enumerate(ARMS):
            started = time.perf_counter()
            base_sequence = index * 100 + arm_index * 10
            store = stores.get(arm)
            pre_event_hash = store.state_hash if store else EMPTY_MEMORY_HASH
            provisional_parent: JsonDict | None = None
            provisional_receipt: JsonDict | None = None
            if arm == "write_while_deciding":
                assert store is not None
                provisional_parent = store.snapshot()
                provisional_receipt = store.commit(_provisional_record(event, 0), capacity_items)
                provisional_receipt["commit_sequence"] = base_sequence + 2
                provisional_receipt["timing"] = "during_decision_provisional"
            pre_decision_hash = store.state_hash if store else EMPTY_MEMORY_HASH
            if arm == "equal_context_replay":
                retrieved = _replay_record(released_feedback[-1] if released_feedback else None)
            elif store is not None:
                retrieved = store.retrieve(str(event["structure_key"])) or _neutral_record()
            else:
                retrieved = _neutral_record()
            _, context_bytes = _render_context(retrieved)
            decision = fixed_candidate_policy(event, retrieved)
            decision_sequence = base_sequence + 3
            decision_seal = sha256_json(
                {
                    "event_id": event["event_id"],
                    "arm": arm,
                    "candidate_id": decision["candidate_id"],
                    "memory_hash": pre_decision_hash,
                }
            )
            feedback = deepcopy(dict(labels[str(event["event_id"])]))
            feedback_sequence = base_sequence + 4
            exact_label = str(feedback["exact_label"])
            witness = deepcopy(dict(feedback["witness"]))
            proposed_update: JsonDict | None = None
            validation_result: JsonDict = {
                "validated": True,
                "exact_label_match": decision["candidate_id"] == exact_label,
                "validation_sequence": base_sequence + 6,
                "budget_used": 1,
                "authority": witness.get("authority"),
            }
            commit_record: JsonDict = {
                "committed": False,
                "atomic": True,
                "timing": "not_applicable",
                "parent_hash": pre_decision_hash,
                "child_hash": pre_decision_hash,
                "commit_sequence": None,
                "after_feedback": None,
                "evicted_ids": [],
            }
            if arm in {"delayed_procedural", "raw_trace", "write_while_deciding"}:
                assert store is not None
                proposed_update = (
                    _raw_record(event, exact_label, witness)
                    if arm == "raw_trace"
                    else _abstract_record(event, exact_label)
                )
                proposed_update["proposal_sequence"] = base_sequence + 5
                reusable = event["reuse_or_decoy_status"] == "reusable"
                validation_result["update_accepted"] = reusable or arm == "raw_trace"
                validation_result["update_reason"] = (
                    "verifier_signed_reusable"
                    if reusable
                    else "raw_trace_retained_as_observation"
                    if arm == "raw_trace"
                    else "decoy_not_reusable"
                )
                if arm == "write_while_deciding" and provisional_parent is not None:
                    store.restore(provisional_parent)
                if validation_result["update_accepted"]:
                    committed = store.commit(proposed_update, capacity_items)
                    commit_record = {
                        **committed,
                        "commit_sequence": base_sequence + 7,
                        "timing": "after_exact_feedback_between_events",
                        "after_feedback": True,
                        "provisional_commit": provisional_receipt,
                    }
                else:
                    commit_record = {
                        **commit_record,
                        "parent_hash": store.state_hash,
                        "child_hash": store.state_hash,
                        "timing": "rejected_after_exact_feedback",
                        "after_feedback": True,
                        "provisional_commit": provisional_receipt,
                    }
            post_hash = store.state_hash if store else EMPTY_MEMORY_HASH
            evicted_ids = list(commit_record.get("evicted_ids", []))
            state_items = len(store.records) if store else 0
            state_actual_bytes = len(canonical_json(store.records)) if store else 2
            row = {
                "row_key": f"{index:03d}:{arm}",
                "event_id": event["event_id"],
                "chronology_index": index,
                "arm": arm,
                "constraint_family": event["constraint_family"],
                "group_id": event["group_id"],
                "hardness_stratum": event["hardness_stratum"],
                "reuse_or_decoy_status": event["reuse_or_decoy_status"],
                "protected_group": event["protected_group"],
                "protected_retention_probe": event["protected_retention_probe"],
                "slice_id": _slice_for(index),
                "visible_input_hash": sha256_json(event["decision_visible_input"]),
                "candidate_set_hash": sha256_json(event["candidate_set"]),
                "candidate_ids": [
                    candidate["candidate_id"] for candidate in event["candidate_set"]
                ],
                "pre_event_memory_hash": pre_event_hash,
                "pre_decision_memory_hash": pre_decision_hash,
                "retrieved_items": [deepcopy(retrieved)],
                "retrieval_slot_count": RETRIEVAL_SLOT_COUNT,
                "context_budget_bytes": CONTEXT_BUDGET_BYTES,
                "context_bytes": context_bytes,
                "capacity_items": capacity_items,
                "capacity_bytes": capacity_items * RECORD_SLOT_BYTES,
                "record_slot_bytes": RECORD_SLOT_BYTES,
                "state_item_count": state_items,
                "state_charged_bytes": state_items * RECORD_SLOT_BYTES,
                "state_actual_bytes": state_actual_bytes,
                "decision_budget": DECISION_BUDGET,
                "validation_budget": VALIDATION_BUDGET,
                "eviction_policy": EVICTION_POLICY,
                "persistent_state_used": arm in PERSISTENT_ARMS,
                "future_label_accessed": False,
                "exact_feedback_visible_at_decision": False,
                "decision": decision,
                "decision_sequence": decision_sequence,
                "decision_seal": decision_seal,
                "exact_post_decision_label": exact_label,
                "witness": witness,
                "feedback_sequence": feedback_sequence,
                "correct": decision["candidate_id"] == exact_label,
                "proposed_update": proposed_update,
                "validation_result": validation_result,
                "commit_record": commit_record,
                "post_commit_memory_hash": post_hash,
                "evicted_ids": evicted_ids,
                "latency_ms": round((time.perf_counter() - started) * 1_000, 6),
            }
            rows.append(row)
        released_feedback.append(
            {
                "event_id": event["event_id"],
                "chronology_index": index,
                "exact_label": labels[str(event["event_id"])]["exact_label"],
            }
        )
    return rows


def _information_receipts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Prove the fixed causal inputs and resource budgets match within each event."""

    receipts: list[JsonDict] = []
    for index in range(EVENT_COUNT):
        selected = [row for row in rows if row["chronology_index"] == index]
        fields = (
            "visible_input_hash",
            "candidate_set_hash",
            "context_budget_bytes",
            "context_bytes",
            "capacity_items",
            "capacity_bytes",
            "retrieval_slot_count",
            "decision_budget",
            "validation_budget",
            "eviction_policy",
        )
        matched = len(selected) == len(ARMS) and all(
            len({json.dumps(row[field], sort_keys=True) for row in selected}) == 1
            for field in fields
        )
        receipts.append(
            {
                "chronology_index": index,
                "event_id": selected[0]["event_id"] if selected else None,
                "arm_count": len(selected),
                "matched_fields": list(fields),
                "matched": matched,
            }
        )
    return receipts


def _completion_score(artifact: Mapping[str, Any]) -> int:
    """Require the full panel, causal receipts, transactions, and all aggregate cells."""

    rows = artifact["rows"]
    identities = {(row["chronology_index"], row["arm"]) for row in rows}
    committed = [row for row in rows if row["commit_record"]["committed"]]
    expected_cells = {
        "group_rows": len(exp7105.FROZEN_GROUP_IDS) * len(ARMS),
        "family_rows": len(exp7105.FROZEN_FAMILY_IDS) * len(ARMS),
        "hardness_rows": 2 * len(ARMS),
        "reuse_decoy_rows": 2 * len(ARMS),
        "slice_rows": 3 * len(ARMS),
        "capacity_rows": 3 * len(ARMS),
    }
    return int(
        len(rows) == ROW_COUNT
        and len(identities) == ROW_COUNT
        and all(receipt["matched"] for receipt in artifact["equal_information_receipts"])
        and len(artifact["equal_information_receipts"]) == EVENT_COUNT
        and all(row["decision_sequence"] < row["feedback_sequence"] for row in rows)
        and all(row["commit_record"]["atomic"] for row in committed)
        and all(row["state_item_count"] <= row["capacity_items"] for row in rows)
        and all(row["state_charged_bytes"] <= row["capacity_bytes"] for row in rows)
        and all(len(artifact[field]) == count for field, count in expected_cells.items())
    )


def _value_score(artifact: Mapping[str, Any]) -> int:
    """Require every predeclared later contrast plus hard, capacity, and retention safety."""

    intervals = artifact["confidence_interval_rows"]
    protected = [
        row for row in artifact["protected_retention_rows"] if row["arm"] == "delayed_procedural"
    ]
    return int(
        artifact["procedural_memory_comparison_complete_score"] == 1
        and len(intervals) == len(FROZEN_PRIMARY_COMPARISONS)
        and all(row["lower"] > 0 for row in intervals)
        and not any(row["regression"] for row in artifact["negative_transfer_rows"])
        and all(row["retention_passed"] for row in protected)
        and all(row["state_charged_bytes"] <= row["capacity_bytes"] for row in artifact["rows"])
    )


def run_experiment(
    *,
    repo_root: Path,
    upstream_artifact_path: Path,
    run_date: str,
    duration_s: float | None = None,
) -> JsonDict:
    """Run preflight, the complete prospective panel, and cold row reduction."""

    started = time.perf_counter()
    checks, upstream = collect_preconditions(repo_root, upstream_artifact_path)
    if not all(row["passed"] for row in checks):
        return build_blocked_artifact(
            checks,
            upstream,
            run_date=run_date,
            duration_s=duration_s if duration_s is not None else time.perf_counter() - started,
            upstream_artifact_path=upstream_artifact_path,
            repo_root=repo_root,
        )
    decisions = _decision_rows(repo_root, upstream)
    labels = _label_map(repo_root, upstream)
    schedule = list(upstream["frozen_capacity_schedule"])
    rows = _run_rows(decisions, labels, schedule)
    artifact = _base_artifact(
        upstream,
        checks,
        repo_root=repo_root,
        upstream_artifact_path=upstream_artifact_path,
        run_date=run_date,
        duration_s=duration_s if duration_s is not None else time.perf_counter() - started,
    )
    artifact["rows"] = rows
    artifact["event_rows"] = deepcopy(rows)
    artifact["equal_information_receipts"] = _information_receipts(rows)
    artifact.update(reduce_rows(rows))
    artifact.update(_projection_rows(rows))
    artifact["procedural_memory_comparison_complete_score"] = _completion_score(artifact)
    artifact["procedural_memory_value_ready_score"] = _value_score(artifact)
    value_ready = artifact["procedural_memory_value_ready_score"] == 1
    artifact["verdict_class"] = "positive" if value_ready else "null"
    artifact["honest_verdict"] = (
        "complete: delayed procedural memory has positive later value with protected retention"
        if value_ready
        else "complete_null_procedural_memory_value"
    )
    completion_checks = [
        gate_check(
            "procedural_memory_comparison_complete_score",
            1,
            artifact["procedural_memory_comparison_complete_score"],
        ),
        gate_check(
            "procedural_memory_value_ready_score",
            1,
            artifact["procedural_memory_value_ready_score"],
        ),
    ]
    artifact["gate_check_summary"] = gate_summary(completion_checks)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, repo_root=repo_root, check_source_files=True)
    if errors:
        raise ValueError("invalid_exp7106_artifact:" + ",".join(errors))
    return artifact


def _matched_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Find unequal resources and hidden state within the five paired event rows."""

    errors: list[str] = []
    names = {
        "context_bytes": "unequal_context",
        "context_budget_bytes": "unequal_context",
        "capacity_items": "unequal_capacity",
        "capacity_bytes": "unequal_capacity",
        "retrieval_slot_count": "unequal_retrieval_count",
        "decision_budget": "unequal_decision_budget",
        "validation_budget": "unequal_validation_budget",
        "visible_input_hash": "unequal_visible_information",
        "candidate_set_hash": "unequal_candidate_set",
        "eviction_policy": "unequal_eviction_policy",
    }
    for index in range(EVENT_COUNT):
        selected = [row for row in rows if row.get("chronology_index") == index]
        for field, error in names.items():
            if len({json.dumps(row.get(field), sort_keys=True) for row in selected}) > 1:
                errors.append(error)
    if any(
        row.get("arm") in STATELESS_ARMS
        and (
            row.get("persistent_state_used") is not False
            or row.get("pre_event_memory_hash") != EMPTY_MEMORY_HASH
            or row.get("pre_decision_memory_hash") != EMPTY_MEMORY_HASH
            or row.get("post_commit_memory_hash") != EMPTY_MEMORY_HASH
        )
        for row in rows
    ):
        errors.append("hidden_persistent_state")
    return errors


def _causal_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check label isolation, delayed timing, atomicity, chronology, and bounds."""

    errors: list[str] = []
    if any(row.get("future_label_accessed") is not False for row in rows):
        errors.append("future_label_access")
    if any(row.get("exact_feedback_visible_at_decision") is not False for row in rows):
        errors.append("decision_time_exact_feedback")
    if any(
        int(row.get("decision_sequence", 0)) >= int(row.get("feedback_sequence", 0)) for row in rows
    ):
        errors.append("feedback_not_after_decision")
    delayed_commits = [
        row
        for row in rows
        if row.get("arm") == "delayed_procedural" and row["commit_record"].get("committed")
    ]
    if any(
        int(row["commit_record"].get("commit_sequence") or 0) <= int(row["feedback_sequence"])
        for row in delayed_commits
    ):
        errors.append("early_delayed_commit")
    if any(
        row["commit_record"].get("committed") and row["commit_record"].get("atomic") is not True
        for row in rows
    ):
        errors.append("non_atomic_commit")
    if any(
        int(row.get("state_item_count", 0)) > int(row.get("capacity_items", 0))
        or int(row.get("state_charged_bytes", 0)) > int(row.get("capacity_bytes", 0))
        or int(row.get("state_actual_bytes", 0)) > int(row.get("capacity_bytes", 0))
        for row in rows
    ):
        errors.append("capacity_exceeded")
    expected_order = [f"{index:03d}:{arm}" for index in range(EVENT_COUNT) for arm in ARMS]
    observed_order = [row.get("row_key") for row in rows]
    if len(rows) != ROW_COUNT or len(set(observed_order)) != ROW_COUNT:
        errors.append("dropped_or_duplicate_rows")
    elif observed_order != expected_order:
        errors.append("event_reorder")
    return errors


def _expected_verdict(artifact: Mapping[str, Any], value_score: int) -> tuple[str, str]:
    """Return the only legal terminal class and headline for row-derived scores."""

    if artifact.get("inference_substrate_class") == "blocked_no_run":
        failed = (
            artifact.get("gate_check_summary", {}).get("failed_check") or "unknown_precondition"
        )
        return "blocked", f"complete_blocked_procedural_memory:{failed}"
    if value_score == 1:
        return (
            "positive",
            "complete: delayed procedural memory has positive later value with protected retention",
        )
    return "null", "complete_null_procedural_memory_value"


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    check_source_files: bool = False,
) -> list[str]:
    """Cold-check schema, causal rows, aggregates, scores, hashes, and verdict."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing_fields:{','.join(missing)}"]
    if not set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"]):
        errors.append("field_principles_incomplete")
    blocked = artifact["inference_substrate_class"] == "blocked_no_run"
    rows = list(artifact["rows"])
    if artifact["event_rows"] != rows:
        errors.append("event_rows_mismatch")
    if blocked:
        if rows or artifact["procedural_memory_comparison_complete_score"] != 0:
            errors.append("blocked_artifact_has_measurements")
        if artifact["gate_check_summary"].get("failed_check") is None:
            errors.append("blocked_gate_diagnostic_missing")
        completion, value = 0, 0
    else:
        errors.extend(_matched_errors(rows))
        errors.extend(_causal_errors(rows))
        reduced = reduce_rows(rows)
        for field, expected in reduced.items():
            if artifact[field] != expected:
                errors.append(f"{field}_mismatch")
        projections = _projection_rows(rows)
        for field, expected in projections.items():
            if artifact[field] != expected:
                errors.append(f"{field}_mismatch")
        expected_receipts = _information_receipts(rows)
        if artifact["equal_information_receipts"] != expected_receipts:
            errors.append("equal_information_receipts_mismatch")
        shadow = dict(artifact)
        shadow.update(reduced)
        shadow.update(projections)
        shadow["equal_information_receipts"] = expected_receipts
        completion = _completion_score(shadow)
        shadow["procedural_memory_comparison_complete_score"] = completion
        value = _value_score(shadow)
        if artifact["procedural_memory_comparison_complete_score"] != completion:
            errors.append("completion_score_mismatch")
        if artifact["procedural_memory_value_ready_score"] != value:
            errors.append("value_score_mismatch")
    expected_class, expected_headline = _expected_verdict(artifact, value)
    if artifact["verdict_class"] != expected_class:
        errors.append("verdict_class_mismatch")
    if artifact["honest_verdict"] != expected_headline:
        errors.append("honest_verdict_mismatch")
    if artifact["model_weights_changed"] is not False:
        errors.append("model_weights_changed")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_oracle_mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact["execution_venue"] != EXECUTION_VENUE:
        errors.append("execution_venue_mismatch")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if check_source_files:
        root = repo_root or Path(__file__).resolve().parents[2]
        declared = artifact["source_artifact_hashes"]
        for source in SOURCE_PATHS:
            if declared.get(str(source)) != sha256_path(root / source):
                errors.append(f"source_hash_mismatch:{source}")
    return list(dict.fromkeys(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Publish complete JSON through one rename so readers never see partial rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse only explicit input and output paths so tests remain isolated."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument(
        "--upstream-artifact-path", type=Path, default=DEFAULT_UPSTREAM_ARTIFACT_PATH
    )
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp7106 from the checkout and write its one requested terminal artifact."""

    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    upstream_path = _resolve(repo_root, args.upstream_artifact_path)
    artifact_path = _resolve(repo_root, args.artifact_path)
    artifact = run_experiment(
        repo_root=repo_root,
        upstream_artifact_path=upstream_path,
        run_date=args.date,
    )
    write_artifact(artifact_path, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - the command wrapper owns this path.
    raise SystemExit(main())
