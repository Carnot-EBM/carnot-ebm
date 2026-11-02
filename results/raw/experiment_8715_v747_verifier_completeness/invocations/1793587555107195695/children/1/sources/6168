"""REQ-VERIFY-8492: reconstruct clocks and freeze append semantics without timing a new store.

The frame reader is a reference fixture, not a durable service. Independent
sealed transactions prevent repaired checksums from changing logical work.
"""

from __future__ import annotations

import json
import struct
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v730_contract_methods import protocol as methods_protocol
from carnot.verify.persistence_localization_8464 import STAGES
from carnot.verify.journal_equivalence_8465 import GENESIS as GENESIS, encode, record

Json = dict[str, Any]
REMOVABLE = {
    "pointer_serialization",
    "pointer_write",
    "pointer_file_fsync",
    "pointer_publication",
    "pointer_directory_fsync",
}


def protocol() -> Json:
    """Freeze already sealed methods so this task cannot quietly change recovery rules."""
    return dict(methods_protocol()["append_protocol"])


def frame(tape: Json, sequence: int, previous: str) -> bytes:
    """Length-prefixed canonical bytes retain full inputs and one stable response identity."""
    payload = dict(
        record(tape["transactions"][sequence - 1], sequence, previous, tape), framing_version=1
    )
    payload["digest"] = canonical_hash({k: v for k, v in payload.items() if k != "digest"})
    encoded = bytes(encode(payload))
    return struct.pack(">I", len(encoded)) + encoded


def recover(data: bytes, tape: Json) -> Json:
    """Replay complete unacknowledged records; only a truncated final frame may disappear.

    A complete malformed frame is corruption even at the tail. A stable ID is
    accepted once in the sealed order; callers reuse its previously saved reply.
    """
    offset, sequence, previous = 0, 0, GENESIS
    ids = []
    while offset < len(data):
        if len(data) - offset < 4:
            break
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        if not 0 < length <= 2 * 1024 * 1024 or len(data) > 16 * 1024 * 1024:
            raise ValueError("frame_budget")
        end = offset + 4 + length
        if sequence < len(tape["transactions"]):
            expected_length = len(frame(tape, sequence + 1, previous)) - 4
            if length != expected_length:
                raise ValueError("interior_length_corruption")
        if end > len(data):
            break
        if sequence >= len(tape["transactions"]):
            raise ValueError("conflicting_id_or_sequence")
        expected = frame(tape, sequence + 1, previous)
        if data[offset:end] != expected:
            raise ValueError("independent_transition_frame")
        parsed = json.loads(data[offset + 4 : end])
        identity = parsed["acknowledgment_id"]
        if identity in ids:
            raise ValueError("conflicting_id")
        ids.append(identity)
        sequence += 1
        previous, offset = parsed["digest"], end
    return dict(
        sequence=sequence,
        committed_ids=ids,
        prefix_digest=previous,
        durable_prefix_bytes=offset,
        discarded_tail_bytes=len(data) - offset,
    )


def ordering(events: list[str]) -> bool:
    """Namespace obligations survive append; a checkpoint must be durable before reclamation."""
    namespace_dirty = data_dirty = checkpoint_pending = durable_checkpoint = False
    for event in events:
        if event in {"create", "checkpoint", "reclaim"}:
            if event == "reclaim" and not durable_checkpoint:
                raise ValueError("reclamation_before_checkpoint")
            namespace_dirty = True
            data_dirty = event != "reclaim"
            checkpoint_pending = event == "checkpoint"
        elif event == "append":
            if namespace_dirty:
                raise ValueError("namespace_sync_missing")
            data_dirty = True
        elif event == "file_sync":
            data_dirty = False
        elif event == "directory_sync":
            if data_dirty:
                raise ValueError("file_sync_missing")
            namespace_dirty = False
            durable_checkpoint |= checkpoint_pending
            checkpoint_pending = False
        elif event == "ack":
            if data_dirty or namespace_dirty:
                raise ValueError("ack_before_durability")
        else:
            raise ValueError("unknown_ordering_event")
    if data_dirty or namespace_dirty:
        raise ValueError("undrained_obligations")
    return True


def namespace(arm: str, transactions: int) -> Json:
    """Count shipped atomic replacements: every directory sync follows a real namespace change."""
    n = transactions
    if arm == "snapshot":
        operations = dict(snapshot_create_replace=n + 1, pointer_replace=n + 1)
        file_syncs, directory_syncs = 2 * (n + 1), 2 * (n + 1)
    elif arm == "journal":
        checkpoints = n // 32 + 2
        operations = dict(
            precreate_journal=1,
            certificate_create_replace=n,
            checkpoint_create_replace=checkpoints,
            checkpoint_pointer_replace=checkpoints,
            checkpoint_marker_create_replace=checkpoints,
        )
        file_syncs, directory_syncs = 2 + 2 * n + 3 * checkpoints, 1 + n + 3 * checkpoints
    else:
        raise ValueError("registered_namespace_arm")
    return dict(
        arm=arm,
        transactions=n,
        namespace_operations=operations,
        file_syncs=file_syncs,
        directory_syncs=directory_syncs,
        total_fsyncs=file_syncs + directory_syncs,
        directory_syncs_without_namespace_change=0,
        transaction_pointer_replacements=n if arm == "snapshot" else 0,
        transaction_certificate_creations=n if arm == "journal" else 0,
        unlink_operations=0,
        sync_call_sources={
            "snapshot": "direct_atomic_state_8376.Store.commit + atomic_json + sync_directory",
            "journal": "journal_equivalence_8465.Journal.initialize/apply/_checkpoint/drain + atomic_json + sync_directory",
        }[arm],
    )


def partition(primitive: Json) -> Json:
    """Count each leaf exactly once; all gaps stay charged to the complete service clock."""
    costs = dict.fromkeys(STAGES, 0)
    removable, previous_end = 0, primitive["start_ns"]
    for span in primitive["persistence_spans"]:
        clock = span["start_ns"]
        if not previous_end <= clock <= span["end_ns"] <= primitive["end_ns"]:
            raise ValueError("complete_span_nesting")
        for stage in span["stages"]:
            name, end = stage["name"], stage["end_ns"]
            if name not in STAGES or stage["start_ns"] != clock or end < clock:
                raise ValueError("disjoint_leaf_clock")
            cost = end - clock
            costs[name] += cost
            if span["start_ns"] >= primitive["initialized_ns"] and name in REMOVABLE:
                removable += cost
            clock = end
        if clock != span["end_ns"]:
            raise ValueError("incomplete_leaf_roster")
        previous_end = clock
    wall = primitive["end_ns"] - primitive["start_ns"]
    residual = wall - sum(costs.values())
    if residual < 0 or wall <= 0:
        raise ValueError("complete_clock_residual")
    categories = dict(
        serialization=costs["serialization_hash"] + costs["pointer_serialization"],
        file_sync=costs["file_fsync"] + costs["pointer_file_fsync"],
        directory_sync=costs["directory_fsync"] + costs["pointer_directory_fsync"],
        pointer_publication=costs["pointer_publication"],
    )
    categories["residual"] = wall - sum(categories.values())
    return dict(
        complete_wall_ns=wall,
        stage_ns=costs,
        category_ns=categories,
        fractions={k: v / wall for k, v in categories.items()},
        outer_residual_ns=residual,
        removable_transaction_ns=removable,
    )


def ceiling(wall: int, removable: int) -> Json:
    """A zero-new-cost bound assumes unchanged retained work and equal transferred marginal cost.

    Journal rows have no measured disjoint stages. A bound using snapshot
    pointer stages is valid only under the declared transfer assumption, and
    a nonpositive denominator invalidates that assumption rather than clipping it.
    """
    remaining = wall - removable
    return dict(
        complete_wall_ns=wall,
        removable_ns=removable,
        retained_complete_cost_ns=remaining,
        conditional_speed_ceiling=wall / remaining if remaining > 0 else None,
        measured_speedup=None,
        assumptions=[
            "retained work unchanged",
            "new append overhead zero",
            "snapshot pointer marginal cost equals journal certificate marginal cost",
        ],
        applicability="conditional engineering cost bound; requires Exp8494 measurement",
    )
