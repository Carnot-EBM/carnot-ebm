"""Replay Exp6978 without model access or learned-store mutation.

The live run made a scientific claim from durable rows and store bytes. This
module treats those bytes as evidence, not authority. It rebuilds chronology,
transactions, metrics, and gates in a fresh process so a stored headline
cannot make its own audit pass.

Spec refs: REQ-LEARN-6979 and SCENARIO-LEARN-6979-*.
"""

from __future__ import annotations

import argparse
from base64 import b64decode
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6979_self_learning_cold_audit.json"
SOURCE_ARTIFACT = REPO_ROOT / "results/experiment_6978_transactional_constraint_self_learning.json"
FIXTURE_ARTIFACT = REPO_ROOT / "results/experiment_6967_certified_error_headroom_fixture.json"
DEFAULT_STORE_ROOT = (
    REPO_ROOT / "results/checkpoints/experiment_6978_transactional_constraint_self_learning/run"
)
WRAPPER_PATH = REPO_ROOT / "scripts/experiments/experiment_6979_self_learning_cold_audit.py"

EXPERIMENT_ID = 6979
RUN_DATE = "20260904"
RANDOM_SEED = 6_979_202_609_04
SCHEMA_VERSION = "carnot.exp6979.self_learning_cold_audit.v1"
INFERENCE_SUBSTRATE = "fresh_process_readonly_transaction_replay"
EXPECTED_SOURCE_SHA256 = "sha256:7850cfd4b5c3f5cc453086498808ed62d16f58f89379c3f93d91c45851c2abce"
EXPECTED_FIXTURE_SHA256 = "sha256:1685ad1bff1b82aae3a17f80d341e0593d99879809bb9afb3268060100e54fee"
EXPECTED_STREAM_SHA256 = "sha256:a15d6bf945d9c3a943884f666894a7a0d3b2cab3b5be57341457cf98e1fb7b6a"
ARMS = ("frozen", "read_only", "transactional_write")
MAX_STATE_BYTES = 8_192
HELD_FUTURE_START = 6
MEMORY_LOOKUP_LIMIT = 4
EXPECTED_JOURNAL_COUNTS = {
    "transactional_write": 36,
    "forced_interruption": 2,
    "harmful_update": 3,
}
STORE_FILES = {
    "frozen_state": "arms/frozen/state.json",
    "frozen_journal": "arms/frozen/journal.jsonl",
    "read_only_state": "arms/read_only/state.json",
    "read_only_journal": "arms/read_only/journal.jsonl",
    "final_state": "arms/transactional_write/state.json",
    "main_journal": "arms/transactional_write/journal.jsonl",
    "interruption_state": "fixtures/interruption/state.json",
    "interruption_journal": "fixtures/interruption/journal.jsonl",
    "rollback_state": "fixtures/rollback/state.json",
    "rollback_journal": "fixtures/rollback/journal.jsonl",
}
JOURNAL_INPUTS = {
    "transactional_write": "main_journal",
    "forced_interruption": "interruption_journal",
    "harmful_update": "rollback_journal",
}
STATE_INPUTS = {
    "transactional_write": "final_state",
    "forced_interruption": "interruption_state",
    "harmful_update": "rollback_state",
}
FORBIDDEN_FIELDS = {
    "confidence",
    "model_confidence",
    "rationale",
    "future_label",
    "future_labels",
    "later_outcome",
    "later_outcomes",
    "held_future_label",
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "per_event_results",
    "visibility_replay_rows",
    "journal_replay_rows",
    "state_hash_rows",
    "budget_recomputation_rows",
    "metric_recomputation_rows",
    "positive_gate_recomputation",
    "source_disagreement_rows",
    "leakage_audit_rows",
    "rollback_audit_rows",
    "read_only_enforcement_receipt",
    "self_learning_audit_complete_score",
    "learning_safety_confirmed_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned schema lets another checker reject incompatible evidence.",
    "experiment_id": "A fixed identity prevents evidence from another audit entering this result.",
    "run_date": "The date binds the audit to its requested execution window.",
    "field_principles": "A reason for each field makes the audit contract reviewable.",
    "preconditions_checked": "Fail-closed checks prevent incomplete upstream evidence from becoming a partial claim.",
    "inference_substrate": "The substrate states that no model or hardware produced audit evidence.",
    "duration_s": "Measured wall time distinguishes execution from a hand-written result.",
    "source_artifact_hashes": "Input hashes bind every conclusion to exact source and store bytes.",
    "rows": "The 72 source units remain the authority for all outcome arithmetic.",
    "per_event_results": "Paired event rows expose gains and losses without pooled masking.",
    "visibility_replay_rows": "Per-call frontiers expose future evidence before it can receive learning credit.",
    "journal_replay_rows": "Phase replay proves state transitions without trusting stored summaries.",
    "state_hash_rows": "Per-event hashes connect transaction replay to the state each arm used.",
    "budget_recomputation_rows": "Independent budgets prevent compute differences from posing as learning.",
    "metric_recomputation_rows": "Row-only estimates and intervals prevent stored headlines from becoming authority.",
    "positive_gate_recomputation": "Explicit terms keep write volume from substituting for useful later gains.",
    "source_disagreement_rows": "Named disagreements preserve corrections instead of silently repairing evidence.",
    "leakage_audit_rows": "Leakage checks keep future labels and confidence outside causal authority.",
    "rollback_audit_rows": "Recovery checks prove harmful and interrupted updates restore allowed bytes.",
    "read_only_enforcement_receipt": "Runtime denials and unchanged hashes prove the audit did not train the store.",
    "self_learning_audit_complete_score": "One means every source row and state completed independent replay.",
    "learning_safety_confirmed_score": "One means hashes, timing, storage, budgets, and arithmetic all agree.",
    "random_seed": "A fixed bootstrap seed makes confidence intervals reproducible.",
    "reproducibility_checksum": "A timing-free digest detects later scientific-content drift.",
    "gate_check_summary": "Expected and observed values make a blocked input actionable.",
    "verifier_is_oracle": "False separates this replay checker from the source exact evaluator.",
    "verdict_class": "A closed class prevents a reproduced null from becoming a positive result.",
    "honest_verdict": "A class-consistent prefix gives automation one stable terminal state.",
}


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    """Serialize evidence with stable keys so hashes have one meaning."""

    suffix = "\n" if newline else ""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + suffix
    ).encode()


def _sha256(value: bytes) -> str:
    """Return the repository's explicit spelling for a SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _check(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Keep both sides of a gate so failure is diagnosable."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed if passed is None else passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed gate while retaining the complete check list."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "checks": rows,
        "passed": failed is None,
    }


def _snapshot_file(path: Path) -> JsonDict:
    """Read bytes once so hashing always happens before metric decoding."""

    try:
        payload = path.read_bytes()
    except OSError:
        return {"path": str(path), "bytes": None, "sha256": None, "size": None}
    return {
        "path": str(path),
        "bytes": payload,
        "sha256": _sha256(payload),
        "size": len(payload),
    }


def capture_input_snapshot(
    *,
    source_artifact: Path = SOURCE_ARTIFACT,
    fixture_artifact: Path = FIXTURE_ARTIFACT,
    store_root: Path = DEFAULT_STORE_ROOT,
) -> JsonDict:
    """Capture every audit input before JSON fields can influence control flow."""

    files = {
        "source_artifact": _snapshot_file(Path(source_artifact)),
        "fixture_artifact": _snapshot_file(Path(fixture_artifact)),
    }
    files.update(
        {
            name: _snapshot_file(Path(store_root) / relative)
            for name, relative in STORE_FILES.items()
        }
    )
    durable_root = Path(store_root) / "durable"
    durable_files = {
        str(path.relative_to(durable_root)): _snapshot_file(path)
        for path in sorted(durable_root.glob("*/*/*.json"))
        if path.is_file()
    }
    return {
        "files": files,
        "durable_files": durable_files,
        "store_root": str(store_root),
    }


def decode_json_input(snapshot: Mapping[str, Any], name: str) -> JsonDict:
    """Decode one object only after its captured bytes already have a hash."""

    payload = snapshot.get("files", {}).get(name, {}).get("bytes")
    if not isinstance(payload, bytes):
        raise ValueError(f"input is not readable:{name}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"input is not a JSON object:{name}")
    return value


def hash_protected_inputs(store_root: Path) -> JsonDict:
    """Hash the learned store tree so a caller can prove byte identity."""

    root = Path(store_root)
    return {
        str(path.relative_to(root)): _sha256(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def make_runtime_guard(protected_roots: Sequence[Path]):
    """Build an audit hook that denies network use and learned-store mutation."""

    roots = tuple(Path(root).resolve() for root in protected_roots)
    mutation_events = {
        "os.remove",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.mkdir",
        "os.chmod",
        "os.truncate",
    }

    def protected(value: Any) -> bool:
        try:
            path = Path(os.fspath(value)).resolve()
        except (TypeError, ValueError, OSError):
            return False
        return any(path == root or root in path.parents for root in roots)

    def guard(event: str, args: tuple[Any, ...]) -> None:
        if event.startswith("socket."):
            raise PermissionError("network disabled for self-learning cold audit")
        if event == "open" and args and protected(args[0]):
            mode = args[1] if len(args) > 1 else "r"
            flags = args[2] if len(args) > 2 else 0
            writes = isinstance(mode, str) and any(mark in mode for mark in "wax+")
            writes = writes or (
                isinstance(flags, int)
                and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
            )
            if writes:
                raise PermissionError("learned store write denied by cold audit")
        if event in mutation_events and args and protected(args[0]):
            raise PermissionError("learned store mutation denied by cold audit")

    return guard


def _find_forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    """Find denied keys without treating harmless text values as fields."""

    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_FIELDS:
                paths.append(path)
            paths.extend(_find_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_find_forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


def _json_lines(payload: bytes | None) -> list[JsonDict] | None:
    """Decode a journal as objects while preserving parse failure as null."""

    if not isinstance(payload, bytes):
        return None
    try:
        rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return rows if all(isinstance(row, dict) for row in rows) else None


def _stream_observation(fixture: Mapping[str, Any]) -> JsonDict:
    """Rebuild the fixture event chain and its chronological split digest."""

    events = list(fixture.get("chronological_event_rows", []))
    seed = fixture.get("random_seed")
    previous_id: str | None = None
    previous_hash = _sha256(_canonical_bytes({"chronological_genesis": seed}))
    chain_ok = len(events) == 24
    for ordinal, row in enumerate(events):
        payload = {key: value for key, value in row.items() if key != "event_hash"}
        dependency = {
            "dependency_ids": row.get("dependency_ids"),
            "predecessor_event_id": previous_id,
            "predecessor_record_hash": previous_hash,
        }
        chain_ok = chain_ok and all(
            (
                row.get("event_ordinal") == ordinal,
                row.get("predecessor_event_id") == previous_id,
                row.get("predecessor_record_hash") == previous_hash,
                row.get("dependency_record_hash") == _sha256(_canonical_bytes(dependency)),
                row.get("event_hash") == _sha256(_canonical_bytes(payload)),
                row.get("later_outcome_exists") is False,
                "outcome" not in row,
            )
        )
        previous_id = str(row.get("event_id"))
        previous_hash = str(row.get("event_hash"))
    witnesses = [
        row
        for row in fixture.get("exact_witness_rows", [])
        if row.get("subject_kind") == "slice_pair" and row.get("split") == "chronological"
    ]
    sealed = _sha256(_canonical_bytes(witnesses))
    prompts = [
        row for row in fixture.get("prompt_visible_rows", []) if row.get("split") == "chronological"
    ]
    stream = _sha256(
        _canonical_bytes(
            {
                "split": "chronological",
                "prompt_record_hashes": [row.get("prompt_record_hash") for row in prompts],
                "sealed_label_hash": sealed,
                "event_hashes": [row.get("event_hash") for row in events],
            }
        )
    )
    event_order = _sha256(
        _canonical_bytes(
            [
                {
                    "event_id": row.get("event_id"),
                    "event_hash": row.get("event_hash"),
                    "prompt_record_hash": row.get("prompt_record_hash"),
                }
                for row in events
            ],
            newline=True,
        )
    )
    return {
        "event_count": len(events),
        "prompt_count": len(prompts),
        "witness_count": len(witnesses),
        "chain_ok": chain_ok,
        "stream_hash": stream,
        "event_order_hash": event_order,
    }


def _input_hashes(snapshot: Mapping[str, Any]) -> JsonDict:
    """Reduce captured hashes without placing raw bytes in the result artifact."""

    files = {name: row.get("sha256") for name, row in snapshot.get("files", {}).items()}
    durable = [
        {
            "path": path,
            "sha256": row.get("sha256"),
            "size": row.get("size"),
        }
        for path, row in sorted(snapshot.get("durable_files", {}).items())
    ]
    files["durable_checkpoint_tree"] = _sha256(_canonical_bytes(durable))
    return files


def _collect_preconditions(
    snapshot: Mapping[str, Any],
) -> tuple[list[JsonDict], JsonDict | None, JsonDict | None]:
    """Reject missing or changed upstream bytes before using source metrics."""

    files = snapshot.get("files", {})
    source_row = files.get("source_artifact", {})
    fixture_row = files.get("fixture_artifact", {})
    checks = [
        _check("source_artifact_readable", True, isinstance(source_row.get("bytes"), bytes)),
        _check("source_artifact_hash", EXPECTED_SOURCE_SHA256, source_row.get("sha256")),
        _check("fixture_artifact_readable", True, isinstance(fixture_row.get("bytes"), bytes)),
        _check("fixture_artifact_hash", EXPECTED_FIXTURE_SHA256, fixture_row.get("sha256")),
        _check(
            "all_store_inputs_readable",
            True,
            all(isinstance(files.get(name, {}).get("bytes"), bytes) for name in STORE_FILES),
        ),
        _check("all_durable_checkpoints_present", 144, len(snapshot.get("durable_files", {}))),
    ]
    if not all(row["passed"] for row in checks):
        return checks, None, None
    try:
        source = decode_json_input(snapshot, "source_artifact")
        fixture = decode_json_input(snapshot, "fixture_artifact")
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        checks.append(_check("input_json_objects", "valid", type(exc).__name__))
        return checks, None, None
    observation = _stream_observation(fixture)
    rows = list(source.get("rows", []))
    identities = {(row.get("event_ordinal"), row.get("arm")) for row in rows}
    embedded = list(source.get("transaction_journal_rows", []))
    journal_matches = True
    for store, input_name in JOURNAL_INPUTS.items():
        disk_rows = _json_lines(files[input_name]["bytes"])
        source_rows = [
            {key: value for key, value in row.items() if key != "store"}
            for row in embedded
            if row.get("store") == store
        ]
        journal_matches = journal_matches and disk_rows == source_rows
        journal_matches = journal_matches and len(source_rows) == EXPECTED_JOURNAL_COUNTS[store]
    source_fixture_hash = source.get("source_artifact_hashes", {}).get("fixture")
    checks.extend(
        [
            _check(
                "self_learning_run_complete_score",
                1,
                source.get("self_learning_run_complete_score"),
            ),
            _check(
                "exact_72_terminal_rows",
                True,
                len(rows) == 72
                and len(identities) == 72
                and all(row.get("terminal") is True for row in rows),
            ),
            _check("exact_72_outcome_rows", 72, len(source.get("exact_outcome_rows", []))),
            _check("complete_transaction_journals", True, journal_matches),
            _check("source_fixture_hash", EXPECTED_FIXTURE_SHA256, source_fixture_hash),
            _check("source_stream_hash", EXPECTED_STREAM_SHA256, source.get("stream_hash")),
            _check(
                "fixture_stream_hash",
                EXPECTED_STREAM_SHA256,
                fixture.get("split_hashes", {}).get("chronological"),
            ),
            _check("recomputed_stream_hash", EXPECTED_STREAM_SHA256, observation["stream_hash"]),
            _check("event_chain", True, observation["chain_ok"]),
            _check(
                "event_order_hash", source.get("event_order_hash"), observation["event_order_hash"]
            ),
            _check(
                "final_store_snapshot_readable",
                True,
                isinstance(files["final_state"]["bytes"], bytes),
            ),
        ]
    )
    return checks, source, fixture


def _candidate_bytes(parent: bytes, proposal: Mapping[str, Any]) -> bytes:
    """Rebuild the bounded store update without creating a filesystem store."""

    state = json.loads(parent)
    candidate = deepcopy(state)
    candidate["version"] = int(candidate["version"]) + 1
    key = str(proposal.get("policy_key"))
    candidate["records"] = [
        row for row in candidate["records"] if str(row.get("policy_key")) != key
    ]
    candidate["records"].append(deepcopy(dict(proposal)))
    payload = _canonical_bytes(candidate, newline=True)
    while len(payload) > MAX_STATE_BYTES:
        removable = next(
            (
                index
                for index, row in enumerate(candidate["records"])
                if not row.get("protected") and str(row.get("policy_key")) != key
            ),
            None,
        )
        if removable is None:
            raise ValueError("candidate exceeds state budget")
        candidate["records"].pop(removable)
        payload = _canonical_bytes(candidate, newline=True)
    return payload


def replay_all_journals(
    source: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    compare_disk: bool = True,
) -> JsonDict:
    """Replay all phase rows into private byte strings and compare final states."""

    embedded = list(source.get("transaction_journal_rows", []))
    result_rows: list[JsonDict] = []
    summaries: list[JsonDict] = []
    catalog: dict[str, bytes] = {}
    for store in EXPECTED_JOURNAL_COUNTS:
        rows = [row for row in embedded if row.get("store") == store]
        disk_rows = _json_lines(
            snapshot.get("files", {}).get(JOURNAL_INPUTS[store], {}).get("bytes")
        )
        disk_match = disk_rows == [
            {key: value for key, value in row.items() if key != "store"} for row in rows
        ]
        previous: str | None = None
        logical: bytes | None = None
        prepares: dict[str, JsonDict] = {}
        store_passed = len(rows) == EXPECTED_JOURNAL_COUNTS[store]
        for expected_sequence, source_row in enumerate(rows):
            row = dict(source_row)
            transaction_id = str(row.get("transaction_id"))
            phase = str(row.get("phase"))
            errors: list[str] = []
            payload = {key: value for key, value in row.items() if key not in {"store", "row_hash"}}
            if row.get("row_hash") != _sha256(_canonical_bytes(payload, newline=True)):
                errors.append("row_hash_mismatch")
            if row.get("sequence") != expected_sequence or row.get("previous_row_hash") != previous:
                errors.append("row_chain_mismatch")
            before = _sha256(logical) if logical is not None else None
            try:
                if phase == "prepare":
                    parent = b64decode(str(row.get("parent_state_b64")), validate=True)
                    proposed = b64decode(str(row.get("new_state_b64")), validate=True)
                    if _sha256(parent) != row.get("parent_state_hash"):
                        errors.append("prepare_parent_hash_mismatch")
                    if _sha256(proposed) != row.get("new_state_hash"):
                        errors.append("prepare_new_hash_mismatch")
                    if logical is None:
                        logical = parent
                    elif logical != parent:
                        errors.append("prepare_parent_state_mismatch")
                    if _candidate_bytes(parent, row.get("proposal", {})) != proposed:
                        errors.append("candidate_state_mismatch")
                    prepares[transaction_id] = {"parent": parent, "new": proposed, "row": row}
                    catalog[_sha256(parent)] = parent
                    catalog[_sha256(proposed)] = proposed
                elif phase in {"commit", "commit_recovered"}:
                    prepare = prepares.get(transaction_id)
                    if prepare is None:
                        errors.append("commit_without_prepare")
                    elif row.get("new_state_hash") != _sha256(prepare["new"]):
                        errors.append("commit_new_hash_mismatch")
                    else:
                        logical = prepare["new"]
                elif phase == "abort_recovered":
                    prepare = prepares.get(transaction_id)
                    if prepare is None:
                        errors.append("abort_without_prepare")
                    elif logical != prepare["parent"]:
                        errors.append("abort_parent_state_mismatch")
                elif phase == "rollback":
                    restored = b64decode(str(row.get("restored_state_b64")), validate=True)
                    if _sha256(restored) != row.get("restored_state_hash"):
                        errors.append("rollback_bytes_mismatch")
                    logical = restored
                    catalog[_sha256(restored)] = restored
                else:
                    errors.append("unknown_phase")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                errors.append("invalid_state_bytes")
            after = _sha256(logical) if logical is not None else None
            result_rows.append(
                {
                    "store": store,
                    "sequence": row.get("sequence"),
                    "phase": phase,
                    "transaction_id": transaction_id,
                    "state_hash_before": before,
                    "state_hash_after": after,
                    "errors": errors,
                    "passed": not errors,
                }
            )
            store_passed = store_passed and not errors
            previous = str(row.get("row_hash"))
        final_bytes = snapshot.get("files", {}).get(STATE_INPUTS[store], {}).get("bytes")
        final_hash = _sha256(final_bytes) if isinstance(final_bytes, bytes) else None
        replay_hash = _sha256(logical) if logical is not None else final_hash
        if isinstance(final_bytes, bytes):
            catalog[final_hash] = final_bytes
        passed = store_passed and replay_hash == final_hash and (disk_match or not compare_disk)
        summaries.append(
            {
                "store": store,
                "expected_row_count": EXPECTED_JOURNAL_COUNTS[store],
                "observed_row_count": len(rows),
                "disk_rows_match": disk_match,
                "replayed_final_state_hash": replay_hash,
                "observed_final_state_hash": final_hash,
                "passed": passed,
            }
        )
    return {
        "journal_replay_rows": result_rows,
        "store_summary_rows": summaries,
        "state_catalog": catalog,
    }


def _prompt_surfaces(prompt: str) -> JsonDict:
    """Parse only the structured prompt lines that can carry causal evidence."""

    surfaces: JsonDict = {}
    for line in prompt.splitlines():
        for prefix, key in (
            ("VISIBLE_PREDECESSOR_IDS=", "predecessors"),
            ("POLICY_MEMORY=", "memory"),
            ("PUBLIC_PAIR=", "public_pair"),
        ):
            if line.startswith(prefix):
                surfaces[key] = json.loads(line[len(prefix) :])
    return surfaces


def _lookup_records(records: Sequence[Mapping[str, Any]], family: str) -> list[JsonDict]:
    """Reproduce the source lookup order from state records only."""

    eligible = [deepcopy(dict(row)) for row in records if row.get("scope") in {"global", family}]
    eligible.sort(
        key=lambda row: (int(row.get("source_event_ordinal", -1)), str(row.get("policy_key", ""))),
        reverse=True,
    )
    return eligible[:MEMORY_LOOKUP_LIMIT]


def _classify_row(row: Mapping[str, Any]) -> str:
    """Recompute the atomic error class from exact outcome fields."""

    if row.get("exact_success") is True:
        return "none"
    if row.get("parse_success") is False:
        return f"parse:{row.get('parse_reason') or 'malformed_json'}"
    if row.get("schema_outcome") == "rejected":
        return "schema:rejected"
    if row.get("domain_correspondence_outcome") == "failed":
        return "domain_correspondence"
    if row.get("objective_direction_outcome") == "failed":
        return "objective:direction"
    if row.get("objective_order_outcome") == "failed":
        return "objective:order"
    return "exact:relation"


def replay_visibility(
    source: Mapping[str, Any],
    fixture: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    state_catalog: Mapping[str, bytes],
) -> JsonDict:
    """Rebuild every prompt frontier, memory lookup, and causal sequence."""

    events = sorted(
        fixture.get("chronological_event_rows", []), key=lambda row: row.get("event_ordinal", -1)
    )
    event_ordinals = {str(row.get("event_id")): int(row.get("event_ordinal", -1)) for row in events}
    pairs = {
        str(row.get("pair_id")): row
        for row in fixture.get("prompt_visible_rows", [])
        if row.get("split") == "chronological"
    }
    visibility = {
        (row.get("event_id"), row.get("arm")): row
        for row in source.get("prompt_visibility_rows", [])
    }
    outcomes = {
        (row.get("event_id"), row.get("arm")): row for row in source.get("exact_outcome_rows", [])
    }
    lookups = {
        (row.get("event_id"), row.get("arm")): row for row in source.get("memory_lookup_rows", [])
    }
    proposals = {
        (row.get("event_id"), row.get("arm")): row for row in source.get("update_proposal_rows", [])
    }
    commits = {(row.get("event_id"), row.get("arm")): row for row in source.get("commit_rows", [])}
    checkpoints = {
        (row.get("event_id"), row.get("arm"), row.get("stage")): row
        for row in source.get("checkpoint_rows", [])
    }
    replay_rows: list[JsonDict] = []
    leakage_rows: list[JsonDict] = []
    for row in source.get("rows", []):
        event_id = str(row.get("event_id"))
        arm = str(row.get("arm"))
        ordinal = int(row.get("event_ordinal", -1))
        event = events[ordinal] if 0 <= ordinal < len(events) else {}
        expected_ids = [str(item.get("event_id")) for item in events[: max(ordinal, 0)]]
        source_visibility = visibility.get((event_id, arm), {})
        observed_ids = list(source_visibility.get("visible_predecessor_ids", []))
        future = any(event_ordinals.get(str(item), ordinal) >= ordinal for item in observed_ids)
        errors: list[str] = []
        prompt_entry = snapshot.get("durable_files", {}).get(f"{arm}/{ordinal:02d}/prompt.json", {})
        raw_entry = snapshot.get("durable_files", {}).get(
            f"{arm}/{ordinal:02d}/raw_completion.json", {}
        )
        try:
            prompt_payload = json.loads(prompt_entry.get("bytes"))
            raw_payload = json.loads(raw_entry.get("bytes"))
            surfaces = _prompt_surfaces(str(prompt_payload.get("prompt", "")))
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
            prompt_payload, raw_payload, surfaces = {}, {}, {}
            errors.append("durable_payload_invalid")
        if event.get("event_id") != event_id:
            errors.append("event_order_mismatch")
        if observed_ids != expected_ids or surfaces.get("predecessors") != expected_ids:
            errors.append("predecessor_frontier_mismatch")
        if source_visibility.get("visible_predecessor_ordinals") != list(range(max(ordinal, 0))):
            errors.append("predecessor_ordinal_mismatch")
        state_bytes = state_catalog.get(str(row.get("state_hash_before")))
        try:
            records = json.loads(state_bytes)["records"] if isinstance(state_bytes, bytes) else []
        except (TypeError, KeyError, json.JSONDecodeError):
            records = []
            errors.append("state_records_invalid")
        expected_records = (
            []
            if arm == "frozen"
            else _lookup_records(records, str(event.get("formulation_family")))
        )
        expected_keys = [str(record.get("policy_key")) for record in expected_records]
        prompt_memory = [
            {
                "policy_key": record.get("policy_key"),
                "scope": record.get("scope"),
                "policy_text": record.get("policy_text"),
            }
            for record in expected_records
        ]
        lookup = lookups.get((event_id, arm), {})
        if (
            row.get("memory_record_keys") != expected_keys
            or lookup.get("record_keys") != expected_keys
        ):
            errors.append("memory_lookup_mismatch")
        if surfaces.get("memory") != prompt_memory:
            errors.append("prompt_memory_mismatch")
        pair = pairs.get(str(event.get("pair_id")), {})
        public_pair = {
            "pair_id": pair.get("pair_id"),
            "source_formulation": pair.get("source_formulation"),
            "target_formulation": pair.get("target_formulation"),
        }
        if surfaces.get("public_pair") != public_pair:
            errors.append("public_pair_mismatch")
        forbidden = _find_forbidden_paths(surfaces)
        if forbidden:
            errors.append("forbidden_prompt_field")
        prompt_hash = _sha256(str(prompt_payload.get("prompt", "")).encode())
        raw_text = str(raw_payload.get("raw_completion", ""))
        raw_hash = _sha256(raw_text.encode())
        if prompt_hash != row.get("prompt_hash") or raw_hash != row.get("raw_completion_hash"):
            errors.append("durable_content_hash_mismatch")
        outcome = outcomes.get((event_id, arm), {})
        sequence_ok = (
            row.get("prompt_sequence") == prompt_payload.get("sequence")
            and row.get("raw_sequence") == raw_payload.get("sequence")
            and int(row.get("prompt_sequence", -1)) < int(row.get("raw_sequence", -1))
            and int(row.get("raw_sequence", -1)) < int(row.get("outcome_sequence", -1))
            and outcome.get("raw_sequence") == row.get("raw_sequence")
            and outcome.get("outcome_sequence") == row.get("outcome_sequence")
            and outcome.get("raw_was_durable") is True
        )
        proposal = proposals.get((event_id, arm))
        commit = commits.get((event_id, arm))
        if proposal is not None:
            sequence_ok = sequence_ok and int(proposal.get("proposal_sequence", -1)) > int(
                row.get("outcome_sequence", -1)
            )
        if commit is not None:
            # The commit receipt stores journal-local sequence numbers. The
            # arm row retains the global causal sequence used for this check.
            sequence_ok = sequence_ok and commit.get("after_outcome") is True
            sequence_ok = sequence_ok and int(row.get("commit_sequence", -1)) > int(
                row.get("outcome_sequence", -1)
            )
        if not sequence_ok:
            errors.append("causal_sequence_mismatch")
        if outcome.get("exact_success") is not row.get("exact_success") or _classify_row(
            outcome
        ) != row.get("error_class"):
            errors.append("exact_outcome_mismatch")
        for stage, sequence, content_hash in (
            ("prompt_durable", row.get("prompt_sequence"), row.get("prompt_hash")),
            ("raw_completion_durable", row.get("raw_sequence"), row.get("raw_completion_hash")),
            ("exact_outcome", row.get("outcome_sequence"), outcome.get("exact_certificate_digest")),
        ):
            checkpoint = checkpoints.get((event_id, arm, stage), {})
            if (
                checkpoint.get("sequence") != sequence
                or checkpoint.get("content_hash") != content_hash
            ):
                errors.append(f"checkpoint_mismatch:{stage}")
        replay_rows.append(
            {
                "event_id": event_id,
                "event_ordinal": ordinal,
                "arm": arm,
                "expected_predecessor_ids": expected_ids,
                "observed_predecessor_ids": observed_ids,
                "memory_record_keys": expected_keys,
                "future_visibility_detected": future,
                "prompt_sequence": row.get("prompt_sequence"),
                "raw_sequence": row.get("raw_sequence"),
                "outcome_sequence": row.get("outcome_sequence"),
                "errors": errors,
                "passed": not errors and not future,
            }
        )
        leakage_rows.append(
            {
                "check": f"event_visibility:{event_id}:{arm}",
                "observed_forbidden_paths": forbidden,
                "future_visibility_detected": future,
                "passed": not forbidden
                and not future
                and not any("frontier" in error or "memory" in error for error in errors),
            }
        )
    for proposal in source.get("update_proposal_rows", []):
        forbidden = _find_forbidden_paths(proposal.get("writer_input", {}))
        source_ordinal = proposal.get("proposal", {}).get("source_event_ordinal")
        passed = (
            not forbidden
            and source_ordinal == proposal.get("event_ordinal")
            and proposal.get("after_outcome") is True
        )
        leakage_rows.append(
            {
                "check": f"writer_input:{proposal.get('event_id')}",
                "observed_forbidden_paths": forbidden,
                "future_visibility_detected": False,
                "passed": passed,
            }
        )
    fixture_forbidden = _find_forbidden_paths(fixture.get("prompt_visible_rows", []))
    leakage_rows.append(
        {
            "check": "fixture_prompt_surfaces",
            "observed_forbidden_paths": fixture_forbidden,
            "future_visibility_detected": False,
            "passed": not fixture_forbidden,
        }
    )
    return {"visibility_replay_rows": replay_rows, "leakage_audit_rows": leakage_rows}


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    """Return a two-sided Wilson 95 percent interval for one rate."""

    if trials <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _bootstrap_ci(values: Sequence[int]) -> tuple[float, float]:
    """Return a seeded paired-bootstrap 95 percent interval for a mean."""

    if not values:
        return 0.0, 0.0
    rng = random.Random(RANDOM_SEED)
    estimates = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(10_000))
    return estimates[249], estimates[9749]


def recompute_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Derive budgets, outcomes, utility, forgetting, and intervals from rows."""

    copied = [deepcopy(dict(row)) for row in rows]
    per_event: list[JsonDict] = []
    exact_rows: list[JsonDict] = []
    for ordinal in range(24):
        members = [row for row in copied if row.get("event_ordinal") == ordinal]
        by_arm = {str(row.get("arm")): row for row in members}
        per_event.append(
            {
                "event_ordinal": ordinal,
                "event_id": members[0].get("event_id") if members else None,
                "arm_exact_success": {
                    arm: by_arm.get(arm, {}).get("exact_success") for arm in ARMS
                },
                "arm_parse_success": {
                    arm: by_arm.get(arm, {}).get("parse_success") for arm in ARMS
                },
                "terminal": len(members) == 3
                and all(row.get("terminal") is True for row in members),
            }
        )
    for row in copied:
        agrees = (row.get("exact_success") is True) == (row.get("error_class") == "none")
        exact_rows.append(
            {
                "event_id": row.get("event_id"),
                "event_ordinal": row.get("event_ordinal"),
                "arm": row.get("arm"),
                "exact_success": row.get("exact_success") is True,
                "parse_success": row.get("parse_success") is True,
                "source_agrees": agrees,
            }
        )
    budgets: list[JsonDict] = []
    event_parity = all(
        len([row for row in copied if row.get("event_ordinal") == ordinal]) == 3
        and {row.get("arm") for row in copied if row.get("event_ordinal") == ordinal} == set(ARMS)
        and len({row.get("model_id") for row in copied if row.get("event_ordinal") == ordinal}) == 1
        and len({row.get("random_seed") for row in copied if row.get("event_ordinal") == ordinal})
        == 1
        and {row.get("token_cap") for row in copied if row.get("event_ordinal") == ordinal} == {128}
        for ordinal in range(24)
    )
    for arm in ARMS:
        arm_rows = [row for row in copied if row.get("arm") == arm]
        budgets.append(
            {
                "arm": arm,
                "attempted_call_count": sum(row.get("attempted") is True for row in arm_rows),
                "terminal_row_count": sum(row.get("terminal") is True for row in arm_rows),
                "assigned_token_cap_per_event": 128,
                "assigned_token_budget_total": sum(
                    int(row.get("token_cap", 0)) for row in arm_rows
                ),
                "prompt_tokens_used": sum(int(row.get("prompt_tokens", 0)) for row in arm_rows),
                "completion_tokens_used": sum(
                    int(row.get("completion_tokens", 0)) for row in arm_rows
                ),
                "event_budget_parity": event_parity,
                "passed": len(arm_rows) == 24
                and all(row.get("terminal") is True for row in arm_rows)
                and event_parity,
            }
        )
    held: list[JsonDict] = []
    seen_write_errors: set[str] = set()
    plastic_opportunities = 0
    plastic_successes = 0
    read_successes = 0
    retained_successes = 0
    lost_count = 0
    max_forgetting = 0
    for ordinal in range(24):
        members = {
            str(row.get("arm")): row for row in copied if row.get("event_ordinal") == ordinal
        }
        read = members.get("read_only", {})
        write = members.get("transactional_write", {})
        read_error = str(read.get("error_class", "none"))
        recurring = read_error != "none" and read_error in seen_write_errors
        if ordinal >= HELD_FUTURE_START:
            read_success = read.get("exact_success") is True
            write_success = write.get("exact_success") is True
            if recurring:
                plastic_opportunities += 1
                plastic_successes += int(write_success)
            if read_success:
                read_successes += 1
                retained_successes += int(write_success)
                if not write_success:
                    lost_count += 1
            max_forgetting = max(max_forgetting, lost_count)
            held.append(
                {
                    "event_id": read.get("event_id"),
                    "event_ordinal": ordinal,
                    "read_only_exact_success": read_success,
                    "transactional_write_exact_success": write_success,
                    "paired_delta": int(write_success) - int(read_success),
                    "read_only_error_class": read_error,
                    "newly_recurring_error_class": recurring,
                }
            )
        write_error = str(write.get("error_class", "none"))
        if write_error != "none":
            seen_write_errors.add(write_error)
    gain = sum(row["paired_delta"] for row in held)
    plasticity = plastic_successes / plastic_opportunities if plastic_opportunities else 0.0
    stability = retained_successes / read_successes if read_successes else 1.0
    metrics: list[JsonDict] = []
    for arm in ARMS:
        arm_rows = [row for row in copied if row.get("arm") == arm]
        successes = sum(row.get("exact_success") is True for row in arm_rows)
        low, high = _wilson(successes, len(arm_rows))
        metrics.append(
            {
                "metric": f"{arm}_exact_success_rate",
                "numerator": successes,
                "denominator": len(arm_rows),
                "value": successes / len(arm_rows) if arm_rows else 0.0,
                "ci_method": "wilson_95",
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    plastic_low, plastic_high = _wilson(plastic_successes, plastic_opportunities)
    stability_low, stability_high = _wilson(retained_successes, read_successes)
    delta_low, delta_high = _bootstrap_ci([row["paired_delta"] for row in held])
    metrics.extend(
        [
            {
                "metric": "plasticity_rate",
                "numerator": plastic_successes,
                "denominator": plastic_opportunities,
                "value": plasticity,
                "ci_method": "wilson_95",
                "ci95_low": plastic_low,
                "ci95_high": plastic_high,
            },
            {
                "metric": "stability_rate",
                "numerator": retained_successes,
                "denominator": read_successes,
                "value": stability,
                "ci_method": "wilson_95",
                "ci95_low": stability_low,
                "ci95_high": stability_high,
            },
            {
                "metric": "held_future_mean_paired_delta",
                "numerator": gain,
                "denominator": len(held),
                "value": gain / len(held) if held else 0.0,
                "ci_method": "seeded_paired_bootstrap_95_10000",
                "ci95_low": delta_low,
                "ci95_high": delta_high,
            },
            {"metric": "chronological_gain_over_readonly", "value": gain, "ci_method": None},
            {"metric": "maximum_forgetting", "value": max_forgetting, "ci_method": None},
            {
                "metric": "final_memory_state_bytes",
                "value": max(
                    (
                        int(row.get("state_bytes", 0))
                        for row in copied
                        if row.get("arm") == "transactional_write"
                    ),
                    default=0,
                ),
                "ci_method": None,
            },
        ]
    )
    return {
        "per_event_results": per_event,
        "exact_outcome_rows": exact_rows,
        "budget_recomputation_rows": budgets,
        "metric_recomputation_rows": metrics,
        "held_future_rows": held,
        "chronological_gain_over_readonly": gain,
        "plasticity_score": plasticity,
        "stability_score": stability,
        "max_forgetting": max_forgetting,
        "memory_state_bytes": max(
            (
                int(row.get("state_bytes", 0))
                for row in copied
                if row.get("arm") == "transactional_write"
            ),
            default=0,
        ),
        "lost_read_only_successes": lost_count,
        "held_future_gain_count": sum(row["paired_delta"] > 0 for row in held),
    }


def _state_hash_rows(
    source: Mapping[str, Any], replay: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> list[JsonDict]:
    """Connect each arm-event state claim to replayed transaction bytes."""

    configs = {row.get("arm"): row for row in source.get("arm_config_rows", [])}
    commits = {(row.get("event_id"), row.get("arm")): row for row in source.get("commit_rows", [])}
    catalog = replay.get("state_catalog", {})
    current = {arm: configs.get(arm, {}).get("initial_state_hash") for arm in ARMS}
    rows: list[JsonDict] = []
    for source_row in sorted(
        source.get("rows", []),
        key=lambda row: (
            row.get("event_ordinal", -1),
            ARMS.index(str(row.get("arm"))) if row.get("arm") in ARMS else 99,
        ),
    ):
        arm = str(source_row.get("arm"))
        commit = commits.get((source_row.get("event_id"), arm))
        expected_before = current.get(arm)
        expected_after = commit.get("new_state_hash") if commit else expected_before
        if commit and commit.get("rolled_back") is True:
            expected_after = commit.get("parent_state_hash")
        state_bytes = catalog.get(str(expected_after))
        errors: list[str] = []
        if source_row.get("state_hash_before") != expected_before:
            errors.append("state_hash_before_mismatch")
        if source_row.get("state_hash_after") != expected_after:
            errors.append("state_hash_after_mismatch")
        if source_row.get("commit") is not (commit is not None):
            errors.append("commit_presence_mismatch")
        if isinstance(state_bytes, bytes) and source_row.get("state_bytes") != len(state_bytes):
            errors.append("state_byte_count_mismatch")
        rows.append(
            {
                "event_id": source_row.get("event_id"),
                "event_ordinal": source_row.get("event_ordinal"),
                "arm": arm,
                "expected_state_hash_before": expected_before,
                "observed_state_hash_before": source_row.get("state_hash_before"),
                "expected_state_hash_after": expected_after,
                "observed_state_hash_after": source_row.get("state_hash_after"),
                "state_bytes": source_row.get("state_bytes"),
                "errors": errors,
                "passed": not errors,
            }
        )
        current[arm] = expected_after
    final_bytes = snapshot.get("files", {}).get("final_state", {}).get("bytes")
    final_hash = _sha256(final_bytes) if isinstance(final_bytes, bytes) else None
    rows.append(
        {
            "event_id": None,
            "event_ordinal": None,
            "arm": "transactional_write",
            "check": "final_store_snapshot",
            "expected_state_hash_after": current.get("transactional_write"),
            "observed_state_hash_after": final_hash,
            "state_bytes": len(final_bytes) if isinstance(final_bytes, bytes) else None,
            "errors": []
            if current.get("transactional_write") == final_hash
            else ["final_state_hash_mismatch"],
            "passed": current.get("transactional_write") == final_hash,
        }
    )
    return rows


def _rollback_rows(source: Mapping[str, Any], replay: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute harmful rollback and restart recovery from replay summaries."""

    summaries = {row.get("store"): row for row in replay.get("store_summary_rows", [])}
    source_rollback = next(
        (row for row in source.get("rollback_rows", []) if row.get("fixture") == "harmful_update"),
        {},
    )
    source_restart = {row.get("fixture"): row for row in source.get("restart_recovery_rows", [])}
    harmful_phases = [
        row.get("phase")
        for row in replay.get("journal_replay_rows", [])
        if row.get("store") == "harmful_update"
    ]
    interruption_phases = [
        row.get("phase")
        for row in replay.get("journal_replay_rows", [])
        if row.get("store") == "forced_interruption"
    ]
    return [
        {
            "fixture": "harmful_update",
            "observed_phases": harmful_phases,
            "expected_phases": ["prepare", "commit", "rollback"],
            "parent_bytes_restored": source_rollback.get("parent_bytes_restored") is True,
            "passed": harmful_phases == ["prepare", "commit", "rollback"]
            and summaries.get("harmful_update", {}).get("passed") is True
            and source_rollback.get("passed") is True,
        },
        {
            "fixture": "forced_interruption",
            "observed_phases": interruption_phases,
            "expected_phases": ["prepare", "abort_recovered"],
            "parent_bytes_restored": source_restart.get("forced_interruption", {}).get(
                "parent_bytes_preserved"
            )
            is True,
            "passed": interruption_phases == ["prepare", "abort_recovered"]
            and summaries.get("forced_interruption", {}).get("passed") is True
            and source_restart.get("forced_interruption", {}).get("passed") is True,
        },
        {
            "fixture": "final_write_store_restart",
            "observed_hash": summaries.get("transactional_write", {}).get(
                "observed_final_state_hash"
            ),
            "expected_hash": source_restart.get("final_write_store_restart", {}).get(
                "expected_hash"
            ),
            "passed": summaries.get("transactional_write", {}).get("passed") is True
            and source_restart.get("final_write_store_restart", {}).get("passed") is True,
        },
    ]


def _source_disagreements(
    source: Mapping[str, Any], recomputed: Mapping[str, Any], raw_positive_score: int
) -> list[JsonDict]:
    """Name each stored headline that differs from row-only arithmetic."""

    rows: list[JsonDict] = []
    comparisons = {
        "per_event_results": recomputed["per_event_results"],
        "chronological_gain_over_readonly": recomputed["chronological_gain_over_readonly"],
        "plasticity_score": recomputed["plasticity_score"],
        "stability_score": recomputed["stability_score"],
        "max_forgetting": recomputed["max_forgetting"],
        "memory_state_bytes": recomputed["memory_state_bytes"],
        "transactional_learning_positive_score": raw_positive_score,
    }
    stored_budgets = [
        {key: value for key, value in row.items() if key not in {"event_budget_parity", "passed"}}
        for row in recomputed["budget_recomputation_rows"]
    ]
    comparisons["budget_rows"] = stored_budgets
    for field, expected in comparisons.items():
        observed = source.get(field)
        if observed != expected:
            rows.append(
                {
                    "field": field,
                    "expected_from_rows": deepcopy(expected),
                    "observed_source_value": deepcopy(observed),
                    "disagreement": True,
                }
            )
    return rows


def evaluate_loaded_inputs(
    source: Mapping[str, Any], fixture: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> JsonDict:
    """Run every independent replay after immutable preconditions have passed."""

    preconditions, _, _ = _collect_preconditions(snapshot)
    journal = replay_all_journals(source, snapshot)
    visibility = replay_visibility(source, fixture, snapshot, journal["state_catalog"])
    recomputed = recompute_rows(source.get("rows", []))
    state_rows = _state_hash_rows(source, journal, snapshot)
    rollback_rows = _rollback_rows(source, journal)
    leakage_ok = bool(visibility["leakage_audit_rows"]) and all(
        row.get("passed") is True for row in visibility["leakage_audit_rows"]
    )
    rollback_ok = bool(rollback_rows) and all(row.get("passed") is True for row in rollback_rows)
    budget_ok = bool(recomputed["budget_recomputation_rows"]) and all(
        row.get("passed") is True for row in recomputed["budget_recomputation_rows"]
    )
    state_budget_ok = recomputed["memory_state_bytes"] <= MAX_STATE_BYTES
    raw_positive = int(
        recomputed["chronological_gain_over_readonly"] >= 2
        and recomputed["lost_read_only_successes"] == 0
        and rollback_ok
        and leakage_ok
        and state_budget_ok
    )
    disagreements = _source_disagreements(source, recomputed, raw_positive)
    positive_terms = [
        _check(
            "held_future_gain_over_read_only",
            ">=2",
            recomputed["chronological_gain_over_readonly"],
            recomputed["chronological_gain_over_readonly"] >= 2,
        ),
        _check("lost_read_only_successes", 0, recomputed["lost_read_only_successes"]),
        _check("rollback_replay", True, rollback_ok),
        _check("no_future_leakage", True, leakage_ok),
        _check(
            "state_budget",
            f"<={MAX_STATE_BYTES}",
            recomputed["memory_state_bytes"],
            state_budget_ok,
        ),
    ]
    positive = {
        "terms": positive_terms,
        "held_future_gain_count": recomputed["held_future_gain_count"],
        "chronological_gain_over_readonly": recomputed["chronological_gain_over_readonly"],
        "lost_read_only_successes": recomputed["lost_read_only_successes"],
        "raw_recomputed_score": raw_positive,
        "source_score": source.get("transactional_learning_positive_score"),
        "source_agrees": source.get("transactional_learning_positive_score") == raw_positive,
        "score": int(raw_positive == 1 and not disagreements),
    }
    all_rows_replayed = (
        len(visibility["visibility_replay_rows"]) == 72
        and all(row.get("passed") is True for row in visibility["visibility_replay_rows"])
        and len(recomputed["exact_outcome_rows"]) == 72
        and all(row.get("source_agrees") is True for row in recomputed["exact_outcome_rows"])
    )
    all_states_replayed = (
        len(journal["journal_replay_rows"]) == 41
        and all(row.get("passed") is True for row in journal["journal_replay_rows"])
        and all(row.get("passed") is True for row in journal["store_summary_rows"])
        and len(state_rows) == 73
        and all(row.get("passed") is True for row in state_rows)
    )
    return {
        "source": deepcopy(dict(source)),
        "preconditions_checked": preconditions,
        "source_artifact_hashes": _input_hashes(snapshot),
        "per_event_results": recomputed["per_event_results"],
        "visibility_replay_rows": visibility["visibility_replay_rows"],
        "journal_replay_rows": journal["journal_replay_rows"],
        "state_hash_rows": state_rows,
        "budget_recomputation_rows": recomputed["budget_recomputation_rows"],
        "metric_recomputation_rows": recomputed["metric_recomputation_rows"],
        "positive_gate_recomputation": positive,
        "source_disagreement_rows": disagreements,
        "leakage_audit_rows": visibility["leakage_audit_rows"],
        "rollback_audit_rows": rollback_rows,
        "all_rows_replayed": all_rows_replayed,
        "all_states_replayed": all_states_replayed,
        "leakage_ok": leakage_ok,
        "rollback_ok": rollback_ok,
        "budget_ok": budget_ok,
        "state_budget_ok": state_budget_ok,
    }


def _enforcement_ok(receipt: Mapping[str, Any]) -> bool:
    """Require each isolation property instead of trusting one summary flag."""

    return all(
        (
            receipt.get("fresh_process") is True,
            receipt.get("network_disabled") is True,
            receipt.get("gpu_disabled") is True,
            receipt.get("llm_disabled") is True,
            receipt.get("learned_store_write_probe_denied") is True,
            receipt.get("protected_inputs_unchanged") is True,
            receipt.get("replay_store_backend") == "temporary_in_memory",
        )
    )


def _checksum_projection(artifact: Mapping[str, Any]) -> JsonDict:
    """Exclude wall time and process IDs while retaining scientific evidence."""

    value = {
        key: deepcopy(item)
        for key, item in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    receipt = dict(value.get("read_only_enforcement_receipt", {}))
    receipt.pop("worker_pid", None)
    receipt.pop("parent_pid", None)
    value["read_only_enforcement_receipt"] = receipt
    return value


def _empty_evidence(checks: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any]) -> JsonDict:
    """Return complete empty surfaces for a fail-closed upstream block."""

    return {
        "source": {},
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "source_artifact_hashes": _input_hashes(snapshot),
        "per_event_results": [],
        "visibility_replay_rows": [],
        "journal_replay_rows": [],
        "state_hash_rows": [],
        "budget_recomputation_rows": [],
        "metric_recomputation_rows": [],
        "positive_gate_recomputation": {"terms": [], "score": 0},
        "source_disagreement_rows": [],
        "leakage_audit_rows": [],
        "rollback_audit_rows": [],
        "all_rows_replayed": False,
        "all_states_replayed": False,
        "leakage_ok": False,
        "rollback_ok": False,
        "budget_ok": False,
        "state_budget_ok": False,
        "blocked": True,
    }


def build_audit_artifact(
    evidence: Mapping[str, Any],
    *,
    run_date: str,
    duration_s: float,
    enforcement_receipt: Mapping[str, Any],
) -> JsonDict:
    """Build one terminal artifact without promoting a null or disagreement."""

    checks = list(evidence.get("preconditions_checked", []))
    blocked = evidence.get("blocked") is True or not all(
        row.get("passed") is True for row in checks
    )
    audit_complete = int(
        not blocked
        and evidence.get("all_rows_replayed") is True
        and evidence.get("all_states_replayed") is True
    )
    safety = int(
        audit_complete == 1
        and evidence.get("leakage_ok") is True
        and evidence.get("rollback_ok") is True
        and evidence.get("budget_ok") is True
        and evidence.get("state_budget_ok") is True
        and not evidence.get("source_disagreement_rows")
        and _enforcement_ok(enforcement_receipt)
    )
    positive = deepcopy(
        dict(evidence.get("positive_gate_recomputation", {"terms": [], "score": 0}))
    )
    if safety != 1:
        positive["score"] = 0
    if blocked:
        verdict_class = "blocked"
        verdict = "blocked_self_learning_cold_audit"
    elif audit_complete != 1 or safety != 1:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_self_learning_cold_audit"
    elif positive.get("score") == 1:
        verdict_class = "positive"
        verdict = "complete_positive_self_learning_cold_audit"
    else:
        verdict_class = "null"
        verdict = "complete_null_self_learning_cold_audit"
    source = evidence.get("source", {})
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(evidence.get("source_artifact_hashes", {}))),
        "rows": deepcopy(list(source.get("rows", []))),
        "per_event_results": deepcopy(list(evidence.get("per_event_results", []))),
        "visibility_replay_rows": deepcopy(list(evidence.get("visibility_replay_rows", []))),
        "journal_replay_rows": deepcopy(list(evidence.get("journal_replay_rows", []))),
        "state_hash_rows": deepcopy(list(evidence.get("state_hash_rows", []))),
        "budget_recomputation_rows": deepcopy(list(evidence.get("budget_recomputation_rows", []))),
        "metric_recomputation_rows": deepcopy(list(evidence.get("metric_recomputation_rows", []))),
        "positive_gate_recomputation": positive,
        "source_disagreement_rows": deepcopy(list(evidence.get("source_disagreement_rows", []))),
        "leakage_audit_rows": deepcopy(list(evidence.get("leakage_audit_rows", []))),
        "rollback_audit_rows": deepcopy(list(evidence.get("rollback_audit_rows", []))),
        "read_only_enforcement_receipt": deepcopy(dict(enforcement_receipt)),
        "self_learning_audit_complete_score": audit_complete,
        "learning_safety_confirmed_score": safety,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = _sha256(_canonical_bytes(_checksum_projection(artifact)))
    return artifact


def audit_snapshot(
    snapshot: Mapping[str, Any],
    *,
    run_date: str,
    enforcement_receipt: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Audit one pre-hashed snapshot and preserve blocked evidence honestly."""

    checks, source, fixture = _collect_preconditions(snapshot)
    evidence = (
        evaluate_loaded_inputs(source, fixture, snapshot)
        if source is not None and fixture is not None and all(row["passed"] for row in checks)
        else _empty_evidence(checks, snapshot)
    )
    return build_audit_artifact(
        evidence,
        run_date=run_date,
        duration_s=duration_s,
        enforcement_receipt=enforcement_receipt,
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required structure, bare scores, verdict, and checksum."""

    errors = [
        f"missing required field:{field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in artifact
    ]
    if errors:
        return errors
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover every required field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    for field in ("self_learning_audit_complete_score", "learning_safety_confirmed_score"):
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
            errors.append(f"{field} must be a bare integer")
    verdict_class = artifact.get("verdict_class")
    prefixes = {
        "positive": "complete_positive",
        "circular_positive": "complete_circular",
        "null": "complete_null",
        "blocked": "blocked_",
        "disqualified": "complete_disqualified",
        "partial": "complete_partial",
    }
    if verdict_class not in prefixes:
        errors.append("verdict_class is invalid")
    elif not str(artifact.get("honest_verdict", "")).startswith(prefixes[str(verdict_class)]):
        errors.append("honest_verdict prefix disagrees with verdict_class")
    if verdict_class == "blocked":
        if artifact.get("rows"):
            errors.append("blocked artifact must not contain source rows")
        if artifact.get("gate_check_summary", {}).get("failed_check") is None:
            errors.append("blocked artifact must name a failed check")
    if (
        artifact.get("source_disagreement_rows")
        and artifact.get("learning_safety_confirmed_score") != 0
    ):
        errors.append("source disagreement must clear learning safety")
    if (
        artifact.get("positive_gate_recomputation", {}).get("score") == 1
        and verdict_class != "positive"
    ):
        errors.append("positive gate and verdict disagree")
    expected = _sha256(_canonical_bytes(_checksum_projection(artifact)))
    if artifact.get("reproducibility_checksum") != expected:
        errors.append("reproducibility_checksum mismatch")
    return errors


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish only the audit artifact through one complete rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _store_write_is_denied(path: Path) -> bool:
    """Probe one protected path without changing it when the guard works."""

    try:
        with path.open("ab"):
            pass
    except PermissionError:
        return True
    return False


def _worker_artifact(args: argparse.Namespace) -> JsonDict:
    """Run the read-only replay after installing process-wide denials."""

    started = time.perf_counter()
    store_root = Path(args.store_root)
    sys.addaudithook(make_runtime_guard((store_root,)))
    before = hash_protected_inputs(store_root)
    snapshot = capture_input_snapshot(
        source_artifact=Path(args.source_artifact),
        fixture_artifact=Path(args.fixture_artifact),
        store_root=store_root,
    )
    network_denied = False
    try:
        import socket

        socket.socket()
    except PermissionError:
        network_denied = True
    write_denied = _store_write_is_denied(store_root / STORE_FILES["final_state"])
    after = hash_protected_inputs(store_root)
    receipt = {
        "fresh_process": os.getpid() != os.getppid(),
        "worker_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "network_disabled": network_denied,
        "gpu_disabled": os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("NVIDIA_VISIBLE_DEVICES") == "none",
        "llm_disabled": not any(
            name in sys.modules for name in ("llama_cpp", "transformers", "torch")
        ),
        "learned_store_write_probe_denied": write_denied,
        "protected_inputs_unchanged": before == after,
        "replay_store_backend": "temporary_in_memory",
    }
    return audit_snapshot(
        snapshot,
        run_date=args.date,
        enforcement_receipt=receipt,
        duration_s=time.perf_counter() - started,
    )


def _spawn_worker(args: argparse.Namespace) -> JsonDict:
    """Start an isolated interpreter with model and accelerator access hidden."""

    command = [
        sys.executable,
        "-I",
        str(WRAPPER_PATH),
        "--worker",
        "--date",
        args.date,
        "--source-artifact",
        str(args.source_artifact),
        "--fixture-artifact",
        str(args.fixture_artifact),
        "--store-root",
        str(args.store_root),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cold audit worker failed:{completed.stderr.strip()}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("cold audit worker did not return a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fresh-process audit or validate its checked-in artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--source-artifact", type=Path, default=SOURCE_ARTIFACT)
    parser.add_argument("--fixture-artifact", type=Path, default=FIXTURE_ARTIFACT)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        artifact = _worker_artifact(args)
        print(json.dumps(artifact, sort_keys=True, separators=(",", ":")))
        return 0
    if args.validate:
        try:
            artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            artifact = {}
        errors = validate_artifact(artifact)
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True))
        return int(bool(errors))
    artifact = _spawn_worker(args)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"audit artifact validation failed:{errors}")
    _write_json_atomic(args.result_path, artifact)
    print(
        json.dumps(
            {"honest_verdict": artifact["honest_verdict"], "result": str(args.result_path)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the command wrapper is the tested boundary.
    raise SystemExit(main())
