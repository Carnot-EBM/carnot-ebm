"""Replay Exp6791 while retaining exact canonical state bytes at each commit.

Spec refs: REQ-CL-6797 and SCENARIO-CL-6797-*.

The scientific policy stays inside the checked-in Exp6791 implementation. This
module adds evidence capture at its transaction boundary. A parent process
owns the durable order checkpoint. Worker processes only compute complete
orders and send them to the parent.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6791_compositional_online_constraint_routing_ab as source_code
from carnot.durable_row_checkpoint import (
    DurableRowCheckpoint,
    InvalidEnvelopeError,
    ManifestMismatchError,
    complete_row_envelope,
    sha256_json,
)
from carnot.memory.transactional_constraint_memory import canonical_json_bytes


JsonDict = dict[str, Any]
WorkerLauncher = Callable[[list[str], int, int | None], tuple[list[JsonDict], JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_6797_canonical_transaction_byte_replay"
SCHEMA = "carnot.experiment_6797.canonical_transaction_byte_replay.v1"
RUN_DATE = "20260831"
RANDOM_SEED = source_code.RANDOM_SEED
INFERENCE_SUBSTRATE = "deterministic CPU transactional replay, no LLM"
EXP6790_PATH = REPO_ROOT / source_code.SOURCE_RELATIVE_PATH
EXP6791_PATH = REPO_ROOT / source_code.RESULT_RELATIVE_PATH
RESULT_PATH = REPO_ROOT / "results" / f"{EXPERIMENT_ID}.json"
CHECKPOINT_PATH = REPO_ROOT / "results/.checkpoints" / EXPERIMENT_ID / "orders.json"
STATE_ROOT = REPO_ROOT / "results/.checkpoints" / EXPERIMENT_ID / "stores"
EXPECTED_SOURCE_HASHES = {
    "experiment_6790": source_code.EXPECTED_SOURCE_ARTIFACT_SHA256,
    "experiment_6791": "sha256:bf07395629a10ec9ec434c2f8bac809ac9729e921ebf1e55f10268ea10e27d99",
}
EXPECTED_COUNTS = {
    "compositional_online_writes": 1_063,
    "compositional_online_later_reads": 3_132,
    "compositional_online_action_changes": 721,
    "committed_transactions": 3_189,
}
MIN_FREE_DISK_BYTES = 4 * 1024**3
MIN_AVAILABLE_MEMORY_BYTES = 1024**3
ATTACK_IDS = (
    "byte_flip",
    "receipt_reorder",
    "stale_parent",
    "valid_bytes_wrong_arm",
    "duplicate_commit",
    "cross_arm_store_access",
    "interrupted_write",
    "manifest_mismatch",
)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
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
    "source_artifact_hashes",
    "frozen_manifest",
    "arm_definitions",
    "order_hashes",
    "checkpoint_receipts",
    "transaction_schema",
    "transaction_receipts",
    "committed_transaction_count",
    "parent_byte_snapshot_count",
    "new_state_byte_snapshot_count",
    "byte_hash_match_count",
    "replay_identity_checks",
    "attack_results",
    "rows",
    "transaction_byte_snapshot_fixture_ready",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "schema": "A version rejects readers that do not understand byte snapshots.",
    "experiment_id": "A stable ID binds this replay to its owned output.",
    "run_date": "The fixed execution date prevents a silent rerun substitution.",
    "status": "Status separates complete evidence from a complete precondition block.",
    "field_principles": "Each required field states the evidence boundary it protects.",
    "inference_substrate": "The CPU declaration prevents this replay from becoming an LLM claim.",
    "duration_s": "Measured wall time shows that the deterministic replay executed.",
    "random_seed": "The source seed proves that no learner or policy was refit.",
    "reproducibility_checksum": "A stable hash detects protocol or receipt drift.",
    "source_artifact_hashes": "Exact source bytes bind the replay to Exp6790 and Exp6791.",
    "frozen_manifest": "The source manifest keeps every scientific mechanism fixed.",
    "arm_definitions": "Exact arm definitions prevent a substitute control.",
    "order_hashes": "Five hashes fix every event chronology.",
    "checkpoint_receipts": "Parent receipts prove interruption and fresh-process resume.",
    "transaction_schema": "The schema states how exact bytes and chain links are encoded.",
    "transaction_receipts": "Each commit retains both state snapshots and immediate checks.",
    "committed_transaction_count": "The source and replay must contain 3,189 commits.",
    "parent_byte_snapshot_count": "Every commit must retain one exact parent snapshot.",
    "new_state_byte_snapshot_count": "Every commit must retain one exact new snapshot.",
    "byte_hash_match_count": "Every stored snapshot pair must match both stored hashes.",
    "replay_identity_checks": "Paired source checks prevent a changed action or outcome.",
    "attack_results": "Eight corruptions must fail without changing accepted bytes.",
    "rows": "All 4,800 source cells remain available for paired inspection.",
    "transaction_byte_snapshot_fixture_ready": "Readiness needs complete bytes and identity.",
    "gate_check_summary": "Each gate keeps its expected and observed value.",
    "verifier_is_oracle": "False records that this verifier does not choose route outcomes.",
    "verdict_class": "A closed class separates reproduction evidence from a new claim.",
    "honest_verdict": "A terminal prefix gives automation one final state.",
}
TRANSACTION_SCHEMA = {
    "schema": "carnot.experiment_6797.transaction_receipt.v1",
    "byte_encoding": "base64",
    "canonical_serializer": "UTF-8 JSON, sorted keys, compact separators, ASCII escapes, newline",
    "committed_required_fields": [
        "transaction_id",
        "order_id",
        "arm",
        "event_id",
        "position",
        "parent_state_bytes",
        "parent_hash",
        "new_state_bytes",
        "new_state_hash",
        "chain_index",
        "chain_predecessor",
        "receipt_hash",
    ],
    "hash_algorithm": "SHA-256",
    "chain_scope": "one isolated arm-order store",
}
CHAIN_RECEIPT_FIELDS = frozenset(
    {
        "transaction_id",
        "committed",
        "order_id",
        "arm",
        "event_id",
        "position",
        "parent_state_bytes",
        "parent_hash",
        "new_state_bytes",
        "new_state_hash",
        "byte_encoding",
        "parent_hash_verified_at_commit",
        "new_state_hash_verified_at_commit",
        "chain_index",
        "chain_predecessor",
    }
)


def sha256_bytes(value: bytes) -> str:
    """Return the project hash form for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash a file as a stream so large byte artifacts do not duplicate in RAM."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def encode_snapshot(value: bytes) -> str:
    """Encode exact state bytes without changing or interpreting them."""

    return base64.b64encode(value).decode("ascii")


def decode_snapshot(value: str) -> bytes:
    """Decode one portable snapshot and reject non-base64 input."""

    return base64.b64decode(value.encode("ascii"), validate=True)


def genesis_predecessor(*, order_id: str, arm: str, parent_hash: str) -> str:
    """Name the immutable first link for one isolated arm-order chain."""

    return sha256_json(
        {
            "schema": "carnot.experiment_6797.transaction_genesis.v1",
            "order_id": order_id,
            "arm": arm,
            "parent_hash": parent_hash,
        }
    )


def _receipt_hash(receipt: Mapping[str, Any]) -> str:
    # Later-use counters are added after the transaction boundary. They cannot
    # change the receipt hash that already sealed the commit and its snapshots.
    material = {key: receipt.get(key) for key in CHAIN_RECEIPT_FIELDS}
    return sha256_json(material)


def capture_commit_receipt(
    raw: Mapping[str, Any],
    *,
    order_id: str,
    arm: str,
    event_id: str,
    position: int,
    chain_index: int,
    chain_predecessor: str,
) -> JsonDict:
    """Capture and verify both canonical states at the commit boundary."""

    parent = decode_snapshot(str(raw["parent_bytes_b64"]))
    new = decode_snapshot(str(raw["new_state_bytes_b64"]))
    parent_state = json.loads(parent)
    new_state = json.loads(new)
    if canonical_json_bytes(parent_state) != parent or canonical_json_bytes(new_state) != new:
        raise ValueError("transaction state is not canonical JSON")
    if parent_state.get("arm") != arm or new_state.get("arm") != arm:
        raise ValueError("transaction state arm ownership mismatch")
    if parent_state.get("order_id") != order_id or new_state.get("order_id") != order_id:
        raise ValueError("transaction state order ownership mismatch")
    parent_match = sha256_bytes(parent) == raw.get("parent_hash")
    new_match = sha256_bytes(new) == raw.get("new_state_hash")
    if not parent_match or not new_match:
        raise ValueError("transaction snapshot hash mismatch at commit")
    receipt: JsonDict = {
        "transaction_id": raw["transaction_id"],
        "committed": True,
        "order_id": order_id,
        "arm": arm,
        "event_id": event_id,
        "position": position,
        "parent_state_bytes": encode_snapshot(parent),
        "parent_hash": raw["parent_hash"],
        "new_state_bytes": encode_snapshot(new),
        "new_state_hash": raw["new_state_hash"],
        "byte_encoding": "base64",
        "parent_hash_verified_at_commit": parent_match,
        "new_state_hash_verified_at_commit": new_match,
        "chain_index": chain_index,
        "chain_predecessor": chain_predecessor,
    }
    receipt["receipt_hash"] = _receipt_hash(receipt)
    return receipt


def verify_commit_receipts(receipts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Verify exact bytes, ownership, order, and predecessor links without writes."""

    errors: list[str] = []
    committed = [row for row in receipts if row.get("committed") is True]
    transaction_ids = [str(row.get("transaction_id")) for row in committed]
    if len(transaction_ids) != len(set(transaction_ids)):
        errors.append("duplicate_transaction_id")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in committed:
        groups[(str(row.get("order_id")), str(row.get("arm")))].append(row)
    for (order_id, arm), chain in groups.items():
        prior_new: bytes | None = None
        prior_receipt: str | None = None
        prior_position = -1
        for expected_index, row in enumerate(chain, start=1):
            try:
                parent = decode_snapshot(str(row.get("parent_state_bytes")))
                new = decode_snapshot(str(row.get("new_state_bytes")))
                parent_state = json.loads(parent)
                new_state = json.loads(new)
            except (UnicodeEncodeError, binascii.Error, json.JSONDecodeError, TypeError):
                errors.append("snapshot_decode_failed")
                continue
            if sha256_bytes(parent) != row.get("parent_hash"):
                errors.append("parent_hash_mismatch")
            if sha256_bytes(new) != row.get("new_state_hash"):
                errors.append("new_state_hash_mismatch")
            if (
                canonical_json_bytes(parent_state) != parent
                or canonical_json_bytes(new_state) != new
            ):
                errors.append("snapshot_not_canonical")
            if parent_state.get("arm") != arm or new_state.get("arm") != arm:
                errors.append("state_arm_ownership_mismatch")
            if parent_state.get("order_id") != order_id or new_state.get("order_id") != order_id:
                errors.append("state_order_ownership_mismatch")
            if row.get("receipt_hash") != _receipt_hash(row):
                errors.append("receipt_hash_mismatch")
            position = int(row.get("position", -1))
            if position <= prior_position:
                errors.append("chain_position_not_increasing")
            if row.get("chain_index") != expected_index:
                errors.append("chain_index_mismatch")
            expected_predecessor = (
                genesis_predecessor(order_id=order_id, arm=arm, parent_hash=sha256_bytes(parent))
                if prior_receipt is None
                else prior_receipt
            )
            if row.get("chain_predecessor") != expected_predecessor:
                errors.append("chain_predecessor_mismatch")
            if prior_new is not None and parent != prior_new:
                errors.append("parent_bytes_do_not_extend_chain")
            prior_new = new
            prior_receipt = str(row.get("receipt_hash"))
            prior_position = position
    return list(dict.fromkeys(errors))


def _gate(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    return {
        "check": check,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": observed == expected if passed is None else passed,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    copied = [deepcopy(dict(row)) for row in checks]
    failures = [row for row in copied if row.get("passed") is not True]
    return {
        "all_passed": not failures,
        "checks": copied,
        "failed_checks": [row["check"] for row in failures],
        "failures": failures,
    }


def _load_object(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} root must be an object")
    return value


def _available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def evaluate_preconditions(
    exp6790: Mapping[str, Any],
    exp6791: Mapping[str, Any],
    *,
    exp6790_path: Path,
    exp6791_path: Path,
    resource_root: Path,
    overrides: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Check all authority, identity, serializer, and resource gates before replay."""

    source_rows = exp6791.get("rows", [])
    source_transactions = exp6791.get("transaction_receipts", [])
    unique_cells = {
        (row.get("order_id"), row.get("event_id"), row.get("arm"))
        for row in source_rows
        if isinstance(row, Mapping)
    }
    order_hashes = exp6791.get("frozen_manifest", {}).get("order_hashes", {})
    serializer_probe = {
        "arm": "serializer_probe",
        "records": [{"unicode": "canonical", "value": 1}],
        "version": 0,
    }
    serializer_bytes = canonical_json_bytes(serializer_probe)
    resource_root.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(resource_root).free
    available_memory = _available_memory_bytes()
    observed: JsonDict = {
        "experiment_6790_hash": sha256_file(exp6790_path),
        "experiment_6791_hash": sha256_file(exp6791_path),
        "unique_frozen_cells": len(unique_cells),
        "five_frozen_order_hashes": order_hashes,
        "four_arm_definitions": exp6791.get("arm_definitions"),
        "source_committed_transactions": sum(
            row.get("committed") is True for row in source_transactions if isinstance(row, Mapping)
        ),
        "canonical_serializer": serializer_bytes
        == source_code.canonical_json_bytes(serializer_probe),
        "sufficient_disk_bytes": free_disk >= MIN_FREE_DISK_BYTES,
        "sufficient_memory_bytes": available_memory >= MIN_AVAILABLE_MEMORY_BYTES,
        "source_stream_binding": exp6791.get("source_artifact_hash")
        == EXPECTED_SOURCE_HASHES["experiment_6790"]
        and exp6790.get("reproducibility_checksum")
        == exp6791.get("frozen_manifest", {}).get("source_reproducibility_checksum"),
    }
    observed.update(dict(overrides or {}))
    checks = [
        _gate(
            "experiment_6790_hash",
            EXPECTED_SOURCE_HASHES["experiment_6790"],
            observed["experiment_6790_hash"],
        ),
        _gate(
            "experiment_6791_hash",
            EXPECTED_SOURCE_HASHES["experiment_6791"],
            observed["experiment_6791_hash"],
        ),
        _gate("unique_frozen_cells", 4_800, observed["unique_frozen_cells"]),
        _gate(
            "five_frozen_order_hashes",
            source_code.EXPECTED_ORDER_HASHES,
            observed["five_frozen_order_hashes"],
        ),
        _gate(
            "four_arm_definitions", source_code.ARM_DEFINITIONS, observed["four_arm_definitions"]
        ),
        _gate("source_committed_transactions", 3_189, observed["source_committed_transactions"]),
        _gate("canonical_serializer", True, observed["canonical_serializer"]),
        _gate("sufficient_disk_bytes", True, observed["sufficient_disk_bytes"])
        | {"available_bytes": free_disk, "minimum_bytes": MIN_FREE_DISK_BYTES},
        _gate("sufficient_memory_bytes", True, observed["sufficient_memory_bytes"])
        | {"available_bytes": available_memory, "minimum_bytes": MIN_AVAILABLE_MEMORY_BYTES},
        _gate("source_stream_binding", True, observed["source_stream_binding"]),
    ]
    return _gate_summary(checks)


def checkpoint_manifest(exp6791: Mapping[str, Any], source_hashes: Mapping[str, Any]) -> JsonDict:
    """Freeze only replay inputs so a scientific or byte-contract change refuses resume."""

    return {
        "schema": "carnot.experiment_6797.checkpoint_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "source_reproducibility_checksum": exp6791.get("reproducibility_checksum"),
        "source_manifest_sha256": exp6791.get("frozen_manifest", {}).get("manifest_sha256"),
        "order_ids": list(source_code.EXPECTED_ORDER_HASHES),
        "order_hashes": deepcopy(source_code.EXPECTED_ORDER_HASHES),
        "arm_definitions": deepcopy(source_code.ARM_DEFINITIONS),
        "interruption_after_complete_orders": 1,
        "cells_per_order": 960,
        "transaction_schema": deepcopy(TRANSACTION_SCHEMA),
    }


def _event_id_from_transaction(transaction_id: str, order_id: str, arm: str) -> str:
    prefix = f"tx:{order_id}:"
    suffix = f":{arm}"
    if not transaction_id.startswith(prefix) or not transaction_id.endswith(suffix):
        raise ValueError("transaction identity does not match store owner")
    return transaction_id[len(prefix) : -len(suffix)]


def replay_one_order(exp6790: Mapping[str, Any], *, order_id: str, state_root: Path) -> JsonDict:
    """Run one untouched Exp6791 order while intercepting its raw commit boundary."""

    order = next(row for row in exp6790["order_definitions"] if row["order_id"] == order_id)
    event_by_id = {str(row["event_id"]): row for row in exp6790["frozen_manifest"]["events"]}
    source_row_by_key = {
        (str(row["order_id"]), str(row["event_id"])): row for row in exp6790["rows"]
    }
    captured: dict[str, JsonDict] = {}
    predecessor: dict[tuple[str, str], str] = {}
    chain_indexes: dict[tuple[str, str], int] = defaultdict(int)
    base_store = source_code.IsolatedTransactionStore

    class CapturingStore(base_store):
        """Add byte evidence after the source store creates an exact receipt."""

        def commit_factor(self, factor: Mapping[str, Any], *, transaction_id: str) -> JsonDict:
            raw = super().commit_factor(factor, transaction_id=transaction_id)
            key = (self.order_id, self.arm)
            chain_indexes[key] += 1
            event_id = _event_id_from_transaction(transaction_id, self.order_id, self.arm)
            prior = predecessor.get(
                key,
                genesis_predecessor(
                    order_id=self.order_id,
                    arm=self.arm,
                    parent_hash=str(raw["parent_hash"]),
                ),
            )
            receipt = capture_commit_receipt(
                raw,
                order_id=self.order_id,
                arm=self.arm,
                event_id=event_id,
                position=int(factor["source_position"]),
                chain_index=chain_indexes[key],
                chain_predecessor=prior,
            )
            predecessor[key] = receipt["receipt_hash"]
            captured[transaction_id] = receipt
            return raw

    original_store = source_code.IsolatedTransactionStore
    source_code.IsolatedTransactionStore = CapturingStore
    try:
        rows, transactions, violations = source_code._run_order(
            order=order,
            event_by_id=event_by_id,
            source_row_by_key=source_row_by_key,
            state_root=state_root,
        )
    finally:
        source_code.IsolatedTransactionStore = original_store
    source_code._annotate_later_use(rows, transactions)
    for transaction in transactions:
        if transaction["committed"] is True:
            transaction.update(captured[str(transaction["transaction_id"])])
    if violations:
        raise RuntimeError(f"active event state changed: {violations}")
    return {
        "order_id": order_id,
        "rows": rows,
        "transaction_receipts": transactions,
    }


def execute_checkpointed_orders(
    *,
    checkpoint_path: Path,
    checkpoint_manifest: Mapping[str, Any],
    ordered_ids: Sequence[str],
    launcher: WorkerLauncher,
    cells_per_order: int,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    """Persist one interrupted prefix, then accept only pending fresh-process orders."""

    checkpoint = DurableRowCheckpoint(checkpoint_path, checkpoint_manifest)
    atomic_receipts: list[JsonDict] = []
    process_receipts: list[JsonDict] = []
    if not checkpoint.rows:
        payloads, process = launcher(list(ordered_ids), 1, 1)
        process_receipts.append(process)
        if len(payloads) != 1 or payloads[0].get("order_id") != ordered_ids[0]:
            raise RuntimeError("interrupted worker did not return the fixed complete prefix")
        envelope = complete_row_envelope(
            row_id=str(payloads[0]["order_id"]),
            manifest_hash=checkpoint.manifest_hash,
            payload=payloads[0],
            attempt=1,
            start_receipt={"attempt": 1, "fresh_process": True},
            end_receipt={"complete": True, "interrupted_after_publish": True},
        )
        atomic_receipts.append(checkpoint.append(envelope))
    reloaded = DurableRowCheckpoint(checkpoint_path, checkpoint_manifest)
    prefix_rows = [row for row in reloaded.rows if row.get("attempt") == 1]
    prefix_cell_count = sum(len(row["payload"].get("rows", [])) for row in prefix_rows)
    pending = reloaded.pending(ordered_ids)
    skipped_once = prefix_cell_count == cells_per_order and len(prefix_rows) == 1
    if pending:
        payloads, process = launcher(pending, 2, None)
        process_receipts.append(process)
        if [row.get("order_id") for row in payloads] != pending:
            raise RuntimeError("fresh worker returned a changed pending order sequence")
        for payload in payloads:
            envelope = complete_row_envelope(
                row_id=str(payload["order_id"]),
                manifest_hash=reloaded.manifest_hash,
                payload=payload,
                attempt=2,
                start_receipt={"attempt": 2, "fresh_process": True},
                end_receipt={"complete": True},
            )
            atomic_receipts.append(reloaded.append(envelope))
    final_checkpoint = DurableRowCheckpoint(checkpoint_path, checkpoint_manifest)
    by_id = {str(row["row_id"]): row["payload"] for row in final_checkpoint.rows}
    if set(by_id) != set(ordered_ids):
        raise RuntimeError("checkpoint does not contain every frozen order")
    rows = [row for order_id in ordered_ids for row in by_id[str(order_id)]["rows"]]
    transactions = [
        row for order_id in ordered_ids for row in by_id[str(order_id)]["transaction_receipts"]
    ]
    return (
        rows,
        transactions,
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "manifest_hash": final_checkpoint.manifest_hash,
            "interruption_after_complete_orders": 1,
            "interrupted_prefix_cell_count": prefix_cell_count,
            "skipped_complete_cells": prefix_cell_count,
            "skipped_complete_cells_exactly_once": skipped_once,
            "fresh_process_resume": bool(
                any(row.get("attempt") == 2 for row in final_checkpoint.rows)
                and all(
                    row.get("start_receipt", {}).get("fresh_process") is True
                    for row in final_checkpoint.rows
                )
            ),
            "complete_order_ids": list(ordered_ids),
            "atomic_append_receipts": atomic_receipts,
            "process_receipts": process_receipts,
        },
    )


def _subprocess_launcher(
    *,
    exp6790_path: Path,
    state_root: Path,
) -> WorkerLauncher:
    def launch(
        order_ids: list[str], attempt: int, stop_after: int | None
    ) -> tuple[list[JsonDict], JsonDict]:
        command = [
            sys.executable,
            "-m",
            __name__,
            "--worker",
            "--source-exp6790",
            str(exp6790_path),
            "--state-root",
            str(state_root),
            "--worker-attempt",
            str(attempt),
            "--worker-order-ids",
            *order_ids,
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("worker pipes were not created")
        payloads: list[JsonDict] = []
        interrupted = False
        for order_id in order_ids:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"worker stopped before {order_id}: {process.stderr.read()}")
            payload = json.loads(line)
            if payload.get("order_id") != order_id:
                raise RuntimeError("worker order sequence changed")
            payloads.append(payload)
            if stop_after is not None and len(payloads) == stop_after:
                process.terminate()
                interrupted = True
                break
            process.stdin.write("ack\n")
            process.stdin.flush()
        if not interrupted:
            process.stdin.close()
        return_code = process.wait(timeout=120)
        stderr = process.stderr.read()
        if not interrupted and return_code != 0:
            raise RuntimeError(f"worker failed with {return_code}: {stderr}")
        return payloads, {
            "attempt": attempt,
            "pid": process.pid,
            "fresh_process": True,
            "requested_order_ids": order_ids,
            "emitted_order_ids": [row["order_id"] for row in payloads],
            "interrupted": interrupted,
            "return_code": return_code,
            "stderr": stderr,
        }

    return launch


def replay_identity_checks(
    source: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    transactions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Pair replay evidence with Exp6791 and recompute every stored summary."""

    source_transactions = source.get("transaction_receipts", [])
    source_transaction_fields = set(source_transactions[0]) if source_transactions else set()
    compact_transactions = [
        {key: row.get(key) for key in source_transaction_fields} for row in transactions
    ]
    reduced = source_code.reduce_evidence(rows, compact_transactions) if rows else {}
    observed_counts = {
        "compositional_online_writes": sum(
            reduced.get("writes_by_arm_order", {}).get(source_code.ONLINE_ARM, {}).values()
        ),
        "compositional_online_later_reads": sum(
            reduced.get("later_reads_by_arm_order", {}).get(source_code.ONLINE_ARM, {}).values()
        ),
        "compositional_online_action_changes": sum(
            reduced.get("action_changes_by_arm_order", {}).get(source_code.ONLINE_ARM, {}).values()
        ),
        "committed_transactions": sum(row.get("committed") is True for row in transactions),
    }
    checks = [
        _gate("row_count", 4_800, len(rows)),
        _gate("unique_row_keys", 4_800, len({row.get("row_key") for row in rows})),
        _gate("paired_rows_identical", source.get("rows"), list(rows)),
        _gate("transaction_decisions_identical", source_transactions, compact_transactions),
        _gate("frozen_activity_counts", EXPECTED_COUNTS, observed_counts),
        _gate(
            "all_order_summaries_identical",
            True,
            bool(reduced)
            and all(
                source.get(field) == reduced.get(field) for field in source_code.ROW_DERIVED_FIELDS
            ),
        ),
    ]
    summary = _gate_summary(checks)
    summary["observed_counts"] = observed_counts
    return summary


def _snapshot_digest(receipts: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in receipts:
        if row.get("committed") is True:
            for field in ("transaction_id", "parent_state_bytes", "new_state_bytes"):
                digest.update(str(row.get(field)).encode("ascii"))
                digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _first_chain_pair(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    groups: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for row in receipts:
        if row.get("committed") is True:
            groups[(str(row["order_id"]), str(row["arm"]))].append(deepcopy(dict(row)))
    return next(chain[:2] for chain in groups.values() if len(chain) >= 2)


def run_attack_suite(
    receipts: Sequence[Mapping[str, Any]],
    *,
    checkpoint_path: Path,
    checkpoint_manifest: Mapping[str, Any],
    state_root: Path,
) -> list[JsonDict]:
    """Run eight corruptions against copies and prove accepted bytes did not move."""

    before_snapshots = _snapshot_digest(receipts)
    before_checkpoint = sha256_file(checkpoint_path)
    pair = _first_chain_pair(receipts)
    outcomes: dict[str, bool] = {}

    flipped = deepcopy(pair)
    encoded = str(flipped[0]["new_state_bytes"])
    flipped[0]["new_state_bytes"] = ("A" if encoded[0] != "A" else "B") + encoded[1:]
    outcomes["byte_flip"] = bool(verify_commit_receipts(flipped))
    outcomes["receipt_reorder"] = "chain_position_not_increasing" in verify_commit_receipts(
        list(reversed(deepcopy(pair)))
    )
    stale = deepcopy(pair)
    stale[1]["parent_state_bytes"] = stale[0]["parent_state_bytes"]
    stale[1]["parent_hash"] = stale[0]["parent_hash"]
    outcomes["stale_parent"] = "parent_bytes_do_not_extend_chain" in verify_commit_receipts(stale)
    wrong_arm = deepcopy(pair[:1])
    wrong_arm[0]["arm"] = next(arm for arm in source_code.ARMS if arm != wrong_arm[0]["arm"])
    outcomes["valid_bytes_wrong_arm"] = "state_arm_ownership_mismatch" in verify_commit_receipts(
        wrong_arm
    )
    duplicate = [*deepcopy(pair), deepcopy(pair[0])]
    outcomes["duplicate_commit"] = "duplicate_transaction_id" in verify_commit_receipts(duplicate)

    first = pair[0]
    store_path = state_root / str(first["order_id"]) / str(first["arm"])
    wrong_owner = next(arm for arm in source_code.ARMS if arm != first["arm"])
    store_bytes = (store_path / "state.json").read_bytes()
    try:
        source_code.IsolatedTransactionStore(store_path, wrong_owner, str(first["order_id"]))
    except ValueError:
        outcomes["cross_arm_store_access"] = (store_path / "state.json").read_bytes() == store_bytes
    else:  # pragma: no cover - a passing invalid access is a terminal implementation defect.
        outcomes["cross_arm_store_access"] = False

    with tempfile.TemporaryDirectory(prefix="exp6797-attack-", dir=checkpoint_path.parent) as name:
        attack_checkpoint = DurableRowCheckpoint(Path(name) / "interrupted.json", {"attack": 1})
        good_bytes = attack_checkpoint.path.read_bytes()
        envelope = complete_row_envelope(
            row_id="interrupted",
            manifest_hash=attack_checkpoint.manifest_hash,
            payload={"partial": True},
            attempt=1,
            start_receipt={},
            end_receipt={},
        )
        envelope["status"] = "partial"
        try:
            attack_checkpoint.append(envelope)
        except InvalidEnvelopeError:
            outcomes["interrupted_write"] = attack_checkpoint.path.read_bytes() == good_bytes
        else:  # pragma: no cover - accepting an incomplete envelope is a contract defect.
            outcomes["interrupted_write"] = False

    try:
        DurableRowCheckpoint(checkpoint_path, {**dict(checkpoint_manifest), "attack_change": True})
    except ManifestMismatchError:
        outcomes["manifest_mismatch"] = True
    else:  # pragma: no cover - accepting a changed manifest is a contract defect.
        outcomes["manifest_mismatch"] = False

    unchanged = before_snapshots == _snapshot_digest(receipts) and before_checkpoint == sha256_file(
        checkpoint_path
    )
    return [
        {
            "attack_id": attack_id,
            "failed_closed": outcomes.get(attack_id) is True,
            "committed_bytes_unchanged": unchanged,
        }
        for attack_id in ATTACK_IDS
    ]


def _empty_checkpoint_receipts() -> JsonDict:
    return {
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "manifest_hash": None,
        "interruption_after_complete_orders": 1,
        "interrupted_prefix_cell_count": 0,
        "skipped_complete_cells": 0,
        "skipped_complete_cells_exactly_once": False,
        "fresh_process_resume": False,
        "complete_order_ids": [],
        "atomic_append_receipts": [],
        "process_receipts": [],
    }


def _base_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_hashes: Mapping[str, Any],
    source: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_blocked_transaction_byte_replay",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "frozen_manifest": deepcopy(source.get("frozen_manifest", {})),
        "arm_definitions": deepcopy(source.get("arm_definitions", source_code.ARM_DEFINITIONS)),
        "order_hashes": deepcopy(
            source.get("frozen_manifest", {}).get("order_hashes", source_code.EXPECTED_ORDER_HASHES)
        ),
        "checkpoint_receipts": _empty_checkpoint_receipts(),
        "transaction_schema": deepcopy(TRANSACTION_SCHEMA),
        "transaction_receipts": [],
        "committed_transaction_count": 0,
        "parent_byte_snapshot_count": 0,
        "new_state_byte_snapshot_count": 0,
        "byte_hash_match_count": 0,
        "replay_identity_checks": {"all_passed": False, "checks": [], "failed_checks": []},
        "attack_results": [],
        "rows": [],
        "transaction_byte_snapshot_fixture_ready": False,
        "gate_check_summary": deepcopy(dict(preconditions)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_transaction_byte_replay: one owned precondition failed",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(
    *,
    run_date: str = RUN_DATE,
    exp6790_path: Path = EXP6790_PATH,
    exp6791_path: Path = EXP6791_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    state_root: Path = STATE_ROOT,
    precondition_overrides: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    launcher: WorkerLauncher | None = None,
) -> JsonDict:
    """Run all five frozen orders or return one complete precondition block."""

    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("run date must use YYYYMMDD")
    started = time.monotonic()
    exp6790 = _load_object(exp6790_path)
    exp6791 = _load_object(exp6791_path)
    source_hashes = {
        "experiment_6790": sha256_file(exp6790_path),
        "experiment_6791": sha256_file(exp6791_path),
    }
    preconditions = evaluate_preconditions(
        exp6790,
        exp6791,
        exp6790_path=exp6790_path,
        exp6791_path=exp6791_path,
        resource_root=checkpoint_path.parent,
        overrides=precondition_overrides,
    )
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact = _base_artifact(
        run_date=run_date,
        duration_s=elapsed,
        source_hashes=source_hashes,
        source=exp6791,
        preconditions=preconditions,
    )
    if preconditions["all_passed"] is not True:
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
        return artifact

    manifest = checkpoint_manifest(exp6791, source_hashes)
    worker = launcher or _subprocess_launcher(exp6790_path=exp6790_path, state_root=state_root)
    try:
        rows, transactions, checkpoint_receipts = execute_checkpointed_orders(
            checkpoint_path=checkpoint_path,
            checkpoint_manifest=manifest,
            ordered_ids=list(source_code.EXPECTED_ORDER_HASHES),
            launcher=worker,
            cells_per_order=960,
        )
    except ManifestMismatchError:
        failed = _gate("checkpoint_manifest_matches", True, False)
        blocked = _gate_summary([*preconditions["checks"], failed])
        artifact["gate_check_summary"] = blocked
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact
    identity = replay_identity_checks(exp6791, rows, transactions)
    committed = [row for row in transactions if row.get("committed") is True]
    receipt_errors = verify_commit_receipts(committed)
    byte_matches = sum(
        row.get("parent_hash_verified_at_commit") is True
        and row.get("new_state_hash_verified_at_commit") is True
        for row in committed
    )
    attacks = run_attack_suite(
        committed,
        checkpoint_path=checkpoint_path,
        checkpoint_manifest=manifest,
        state_root=state_root,
    )
    completion_checks = [
        _gate("committed_transaction_count", 3_189, len(committed)),
        _gate(
            "parent_byte_snapshot_count",
            3_189,
            sum(bool(row.get("parent_state_bytes")) for row in committed),
        ),
        _gate(
            "new_state_byte_snapshot_count",
            3_189,
            sum(bool(row.get("new_state_bytes")) for row in committed),
        ),
        _gate("byte_hash_match_count", 3_189, byte_matches),
        _gate("byte_chain_verification", [], receipt_errors),
        _gate("replay_identity", True, identity["all_passed"]),
        _gate(
            "attack_suite",
            True,
            all(row["failed_closed"] and row["committed_bytes_unchanged"] for row in attacks),
        ),
        _gate(
            "checkpoint_resume",
            True,
            checkpoint_receipts["skipped_complete_cells_exactly_once"]
            and checkpoint_receipts["fresh_process_resume"],
        ),
    ]
    completion = _gate_summary([*preconditions["checks"], *completion_checks])
    ready = completion["all_passed"] is True
    artifact.update(
        {
            "status": "complete_transaction_byte_snapshot_fixture_ready"
            if ready
            else "complete_partial_transaction_byte_replay",
            "checkpoint_receipts": checkpoint_receipts,
            "transaction_receipts": transactions,
            "committed_transaction_count": len(committed),
            "parent_byte_snapshot_count": sum(
                bool(row.get("parent_state_bytes")) for row in committed
            ),
            "new_state_byte_snapshot_count": sum(
                bool(row.get("new_state_bytes")) for row in committed
            ),
            "byte_hash_match_count": byte_matches,
            "replay_identity_checks": identity,
            "attack_results": attacks,
            "rows": rows,
            "transaction_byte_snapshot_fixture_ready": ready,
            "gate_check_summary": completion,
            "verdict_class": "circular_positive" if ready else "partial",
            "honest_verdict": (
                "complete_transaction_byte_snapshot_fixture_ready: 3,189 commits retain verified canonical parent and new state bytes"
                if ready
                else "complete_partial_transaction_byte_replay: one byte, identity, checkpoint, or attack gate failed"
            ),
        }
    )
    artifact["duration_s"] = round(
        float(duration_s) if duration_s is not None else time.monotonic() - started, 6
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable protocol, row, receipt-chain, attack, and terminal evidence."""

    transaction_evidence = [
        {
            "transaction_id": row.get("transaction_id"),
            "committed": row.get("committed"),
            "receipt_hash": row.get("receipt_hash"),
            "later_read_count": row.get("later_read_count"),
            "later_action_influence_count": row.get("later_action_influence_count"),
        }
        for row in artifact.get("transaction_receipts", [])
    ]
    checkpoint = artifact.get("checkpoint_receipts", {})
    material = {
        "schema": artifact.get("schema"),
        "run_date": artifact.get("run_date"),
        "random_seed": artifact.get("random_seed"),
        "source_artifact_hashes": artifact.get("source_artifact_hashes"),
        "frozen_manifest": artifact.get("frozen_manifest"),
        "arm_definitions": artifact.get("arm_definitions"),
        "order_hashes": artifact.get("order_hashes"),
        "transaction_schema": artifact.get("transaction_schema"),
        "rows": artifact.get("rows"),
        "transaction_evidence": transaction_evidence,
        "counts": {
            key: artifact.get(key)
            for key in (
                "committed_transaction_count",
                "parent_byte_snapshot_count",
                "new_state_byte_snapshot_count",
                "byte_hash_match_count",
            )
        },
        "checkpoint": {
            key: checkpoint.get(key)
            for key in (
                "manifest_hash",
                "interrupted_prefix_cell_count",
                "skipped_complete_cells",
                "skipped_complete_cells_exactly_once",
                "fresh_process_resume",
                "complete_order_ids",
            )
        },
        "replay_identity_checks": artifact.get("replay_identity_checks"),
        "attack_results": artifact.get("attack_results"),
        "ready": artifact.get("transaction_byte_snapshot_fixture_ready"),
        "verdict_class": artifact.get("verdict_class"),
        "honest_verdict": artifact.get("honest_verdict"),
    }
    return sha256_json(material)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return closed schema, byte, identity, and terminal errors."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field principle coverage mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict class is outside the closed enum")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest verdict lacks a terminal prefix")
    ready = artifact.get("transaction_byte_snapshot_fixture_ready") is True
    counts = [
        artifact.get("committed_transaction_count"),
        artifact.get("parent_byte_snapshot_count"),
        artifact.get("new_state_byte_snapshot_count"),
        artifact.get("byte_hash_match_count"),
    ]
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("rows") != [] or artifact.get("transaction_receipts") != []:
            errors.append("blocked artifact contains replay evidence")
        if any(count != 0 for count in counts):
            errors.append("blocked artifact has nonzero snapshot counts")
        if artifact.get("status") != "complete_blocked_transaction_byte_replay":
            errors.append("blocked artifact status mismatch")
    elif ready:
        if counts != [3_189, 3_189, 3_189, 3_189]:
            errors.append("ready artifact snapshot counts mismatch")
        if len(artifact.get("rows", [])) != 4_800:
            errors.append("ready artifact row count mismatch")
        if verify_commit_receipts(artifact.get("transaction_receipts", [])):
            errors.append("ready artifact byte chain mismatch")
        if artifact.get("replay_identity_checks", {}).get("all_passed") is not True:
            errors.append("ready artifact identity mismatch")
        if not all(
            row.get("failed_closed") is True and row.get("committed_bytes_unchanged") is True
            for row in artifact.get("attack_results", [])
        ) or len(artifact.get("attack_results", [])) != len(ATTACK_IDS):
            errors.append("ready artifact attack evidence mismatch")
        if artifact.get("verdict_class") != "circular_positive":
            errors.append("ready replay must use circular_positive")
    elif artifact.get("verdict_class") != "partial":
        errors.append("non-ready replay must use partial or blocked")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and atomically publish the large artifact without one giant byte copy."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if (
            temporary.exists()
        ):  # pragma: no cover - only an operating-system write failure leaves it.
            temporary.unlink()
    return {"path": str(path), "sha256": sha256_file(path), "atomic_replace": True}


def _worker_main(args: argparse.Namespace) -> int:
    """Compute complete orders and wait for a parent acknowledgement between orders."""

    source = _load_object(args.source_exp6790)
    for order_id in args.worker_order_ids:
        payload = replay_one_order(source, order_id=order_id, state_root=args.state_root)
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)
        if sys.stdin.readline() != "ack\n":
            return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parent replay or one explicitly requested deterministic worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--state-root", type=Path, default=STATE_ROOT)
    parser.add_argument("--source-exp6790", type=Path, default=EXP6790_PATH)
    parser.add_argument("--source-exp6791", type=Path, default=EXP6791_PATH)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-attempt", type=int, default=1)
    parser.add_argument("--worker-order-ids", nargs="*", default=[])
    args = parser.parse_args(argv)
    if args.worker:
        return _worker_main(args)
    artifact = run_experiment(
        run_date=args.date,
        exp6790_path=args.source_exp6790,
        exp6791_path=args.source_exp6791,
        checkpoint_path=args.checkpoint_path,
        state_root=args.state_root,
    )
    write_artifact(args.output, artifact)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper calls main.
    raise SystemExit(main())
