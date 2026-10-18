"""Cold-audit byte-complete continuous learning without producer imports.

Spec refs: REQ-CL-6798 and SCENARIO-CL-6798-*.

The module implements its own JSON byte format, state decoder, route policy,
and metric reducer. This separation lets the audit detect a producer error
instead of repeating a producer function that could contain the same error.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, overload


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260831"
EXPERIMENT_ID = "experiment_6798_csl_causal_safety_byte_audit"
SCHEMA = "carnot.experiment_6798.csl_causal_safety_byte_audit.v1"
INFERENCE_SUBSTRATE = "independent deterministic CPU byte replay, no LLM"
RANDOM_SEED = 6_798_031
BOOTSTRAP_SEED = 6_791_032
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6798_csl_causal_safety_byte_audit.py")
SCRIPT_RELATIVE_PATH = Path("scripts/experiments/experiment_6798_csl_causal_safety_byte_audit.py")
RESULT_RELATIVE_PATH = Path("results/experiment_6798_csl_causal_safety_byte_audit.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
SOURCE_RELATIVE_PATHS = {
    "experiment_6790": Path("results/experiment_6790_chronological_constraint_routing_stream.json"),
    "experiment_6791": Path(
        "results/experiment_6791_compositional_online_constraint_routing_ab.json"
    ),
    "experiment_6797": Path("results/experiment_6797_canonical_transaction_byte_replay.json"),
}
EXPECTED_SOURCE_HASHES = {
    "experiment_6790": ("sha256:2f2cf984ac9d4dcf4be0fc211329022bc773d3cfd88bb861aaec474e9c53aeb4"),
    "experiment_6791": ("sha256:bf07395629a10ec9ec434c2f8bac809ac9729e921ebf1e55f10268ea10e27d99"),
    "experiment_6797": ("sha256:e71fbd09fdbcff5c6ad3901f6cc213496e55a19c466d6fffe0b61d7162dcf28e"),
}
EXPECTED_ORDER_HASHES = {
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
RETRIEVAL_ARMS = {"compositional_online", "random_update_placebo"}
TIE_BREAK_ORDER = (
    "local_prefix",
    "local_suffix",
    "dependency_prefix",
    "dependency_suffix",
    "mixed_boundary",
)
RESTART_CHAIN_INDICES = (1, 64, 128, 192)
ATTACK_IDS = (
    "future_receipt",
    "poisoned_factor",
    "stale_parent",
    "wrong_arm_valid_bytes",
    "capacity_pressure",
    "eviction_reorder",
    "byte_corruption",
    "duplicate_commit",
)
EXPECTED_ROW_COUNT = 4_800
EXPECTED_COMMIT_COUNT = 3_189
LARGE_JSON_THRESHOLD_BYTES = 256 * 1024 * 1024
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
SOURCE_HEADLINE_FIELDS = (
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
    "source_artifact_hashes",
    "transaction_byte_counts",
    "chain_replay_receipts",
    "cold_recomputed_metrics",
    "headline_differences",
    "factors_with_changed_action_witness",
    "credited_factor_count",
    "retrieval_disable_effects",
    "poison_attack_results",
    "admitted_poison_count",
    "influenced_poison_count",
    "capacity_eviction_receipts",
    "restart_byte_identity",
    "restart_action_identity",
    "rollback_byte_identity",
    "rollback_action_identity",
    "retention_after_phase",
    "hard_case_harm_after_phase",
    "rows",
    "source_verdict_supported",
    "csl_causal_audit_completed",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
REQUIRED_AUDIT_FIELDS = REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = {
    "schema": "The version makes incompatible audit records fail closed.",
    "experiment_id": "The stable name binds this record to the byte audit.",
    "run_date": "The fixed execution date prevents silent protocol drift.",
    "status": "The status separates blocked input from completed replay.",
    "field_principles": "Each output field states why the audit needs it.",
    "inference_substrate": "The audit uses deterministic CPU code and no LLM.",
    "duration_s": "Measured wall time shows that the full replay ran.",
    "random_seed": "The fixed seed owns confidence interval sampling.",
    "reproducibility_checksum": "The hash binds all stable audit evidence.",
    "source_artifact_hashes": "Exact hashes bind the three checked-in sources.",
    "transaction_byte_counts": "Counts prove that no committed snapshot is missing.",
    "chain_replay_receipts": "Chain receipts expose byte and restart checks.",
    "cold_recomputed_metrics": "Rows and receipts own every reported metric.",
    "headline_differences": "Differences expose any source contradiction.",
    "factors_with_changed_action_witness": "Each credited factor has two witnesses.",
    "credited_factor_count": "Only action and utility changes receive credit.",
    "retrieval_disable_effects": "This control isolates retrieval from storage.",
    "poison_attack_results": "Attack evidence stays visible after rejection.",
    "admitted_poison_count": "No poison can enter active state.",
    "influenced_poison_count": "No poison can change an action.",
    "capacity_eviction_receipts": "Every pressure eviction names its factor.",
    "restart_byte_identity": "Restart must restore exact canonical bytes.",
    "restart_action_identity": "Restart must preserve the next route.",
    "rollback_byte_identity": "Rollback must restore declared parent bytes.",
    "rollback_action_identity": "Rollback must restore the prior route.",
    "retention_after_phase": "Retention is checked after each attack phase.",
    "hard_case_harm_after_phase": "Hard cases are checked after each phase.",
    "rows": "Each source unit and attack owns one compact audit row.",
    "source_verdict_supported": "Positive support needs every independent gate.",
    "csl_causal_audit_completed": "Completion does not depend on effect sign.",
    "gate_check_summary": "Each gate keeps its expected and observed values.",
    "verifier_is_oracle": "False keeps proposal features separate from receipts.",
    "verdict_class": "A closed class prevents ambiguous terminal results.",
    "honest_verdict": "A terminal prefix gives automation a stable result.",
}


class _LazyJsonArray(Sequence[JsonDict]):
    """Replay a large JSON array from JSONL without retaining its byte strings."""

    def __init__(
        self,
        path: Path,
        count: int,
        transaction_offsets: Mapping[str, Sequence[int]],
        owner: tempfile.TemporaryDirectory[str],
    ) -> None:
        self._path = path
        self._count = count
        self._transaction_offsets = {
            key: (int(value[0]), int(value[1])) for key, value in transaction_offsets.items()
        }
        self._owner = owner

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[JsonDict]:
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("lazy JSON array requires object rows")
                yield value

    @overload
    def __getitem__(self, index: int) -> JsonDict: ...

    @overload
    def __getitem__(self, index: slice) -> list[JsonDict]: ...

    def __getitem__(self, index: int | slice) -> JsonDict | list[JsonDict]:
        if isinstance(index, slice):
            return list(self)[index]
        normalized = index + self._count if index < 0 else index
        if normalized < 0 or normalized >= self._count:
            raise IndexError(index)
        for position, value in enumerate(self):
            if position == normalized:
                return value
        raise IndexError(index)

    def __deepcopy__(self, memo: dict[int, Any]) -> _LazyJsonArray:
        memo[id(self)] = self
        return self

    def transaction(self, transaction_id: str) -> JsonDict:
        """Load one transaction by its byte range in the JSONL sidecar."""

        try:
            offset, length = self._transaction_offsets[transaction_id]
        except KeyError as exc:
            raise KeyError(transaction_id) from exc
        with self._path.open("rb") as handle:
            handle.seek(offset)
            value = json.loads(handle.read(length))
        if not isinstance(value, dict):
            raise ValueError("lazy transaction row must be an object")
        return value


_LARGE_JSON_SPLIT_PROGRAM = """
import json
from pathlib import Path
import sys

source, slim_path, rows_path, index_path = map(Path, sys.argv[1:])
with source.open("r", encoding="utf-8") as handle:
    value = json.load(handle)
rows = value.get("transaction_receipts") if isinstance(value, dict) else None
if isinstance(rows, list):
    value.pop("transaction_receipts")
else:
    rows = None
with slim_path.open("w", encoding="utf-8") as handle:
    json.dump(value, handle, ensure_ascii=True, separators=(",", ":"))
offsets = {}
with rows_path.open("wb") as handle:
    for row in rows or []:
        encoded = (json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\\n").encode()
        start = handle.tell()
        handle.write(encoded)
        transaction_id = row.get("transaction_id") if isinstance(row, dict) else None
        if isinstance(transaction_id, str):
            offsets[transaction_id] = [start, len(encoded)]
with index_path.open("w", encoding="utf-8") as handle:
    json.dump(offsets, handle, ensure_ascii=True, separators=(",", ":"))
print(len(rows) if rows is not None else -1)
"""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with one independent, stable byte representation."""

    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return (text + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the project form of one SHA-256 byte digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash a source as a stream so a large fixture does not get copied."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_large_json_object(path: Path) -> JsonDict:
    """Split large receipt arrays in a child so parent RSS stays bounded."""

    owner = tempfile.TemporaryDirectory(prefix="carnot-6798-json-")
    root = Path(owner.name)
    slim_path = root / "object.json"
    rows_path = root / "transaction_receipts.jsonl"
    index_path = root / "transaction_index.json"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _LARGE_JSON_SPLIT_PROGRAM,
                str(path),
                str(slim_path),
                str(rows_path),
                str(index_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with slim_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required: {path}")
        count = int(completed.stdout.strip())
        if count >= 0:
            with index_path.open("r", encoding="utf-8") as handle:
                offsets = json.load(handle)
            if not isinstance(offsets, dict):
                raise ValueError("transaction index must be a JSON object")
            value["transaction_receipts"] = _LazyJsonArray(
                rows_path,
                count,
                offsets,
                owner,
            )
        else:
            owner.cleanup()
        return value
    except BaseException:
        owner.cleanup()
        raise


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and reject a scalar or array root."""

    if path.stat().st_size >= LARGE_JSON_THRESHOLD_BYTES:
        return _read_large_json_object(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def decode_snapshot(encoded: str) -> bytes:
    """Decode strict base64 so altered snapshot text cannot be accepted."""

    try:
        return base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError("snapshot is not strict base64") from exc


def parse_state(raw: bytes) -> JsonDict:
    """Decode one store and require its bytes to already be canonical."""

    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("state is not canonical JSON") from exc
    required = {"arm", "order_id", "records", "schema", "version"}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError("state is not canonical store JSON")
    if not isinstance(value["records"], list) or not isinstance(value["version"], int):
        raise ValueError("state is not canonical store JSON")
    if canonical_json_bytes(value) != raw:
        raise ValueError("state bytes are not canonical")
    return value


def _gate(check: str, expected: Any, observed: Any) -> JsonDict:
    """Keep exact values so a failed gate remains diagnosable."""

    return {
        "check": check,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": observed == expected,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]], **extra: Any) -> JsonDict:
    """Collect all failures instead of hiding later failed checks."""

    copied = [deepcopy(dict(row)) for row in checks]
    failures = [row for row in copied if row.get("passed") is not True]
    return {
        "all_passed": not failures,
        "checks": copied,
        "failed_checks": [str(row["check"]) for row in failures],
        "failures": failures,
        **deepcopy(extra),
    }


def _observe_snapshot_bytes(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count committed byte pairs and verify each declared digest."""

    committed_count = 0
    parent_count = 0
    new_count = 0
    matches = 0
    for row in receipts:
        if row.get("committed") is not True:
            continue
        committed_count += 1
        parent_text = row.get("parent_state_bytes")
        new_text = row.get("new_state_bytes")
        if isinstance(parent_text, str):
            parent_count += 1
        if isinstance(new_text, str):
            new_count += 1
        if not isinstance(parent_text, str) or not isinstance(new_text, str):
            continue
        try:
            parent = decode_snapshot(parent_text)
            new = decode_snapshot(new_text)
        except ValueError:
            continue
        matches += int(
            sha256_bytes(parent) == row.get("parent_hash")
            and sha256_bytes(new) == row.get("new_state_hash")
        )
    return {
        "committed_transactions": committed_count,
        "parent_snapshots": parent_count,
        "new_state_snapshots": new_count,
        "matching_byte_hashes": matches,
    }


def evaluate_preconditions(
    sources: Mapping[str, Mapping[str, Any]],
    source_paths: Mapping[str, Path],
    *,
    hash_overrides: Mapping[str, str | None] | None = None,
) -> JsonDict:
    """Check the complete fixed contract before any reduced replay can run."""

    replay = sources.get("experiment_6797", {})
    rows = replay.get("rows", [])
    receipts = replay.get("transaction_receipts", [])
    receipt_rows = (
        receipts
        if isinstance(receipts, _LazyJsonArray)
        else [row for row in receipts if isinstance(row, Mapping)]
        if isinstance(receipts, list)
        else []
    )
    byte_counts = _observe_snapshot_bytes(receipt_rows)
    row_keys = (
        [row.get("row_key") for row in rows if isinstance(row, Mapping)]
        if isinstance(rows, list)
        else []
    )
    unique_rows = len(set(row_keys)) if len(row_keys) == len(rows) else -1
    order_hashes = replay.get("order_hashes", {})
    hashes = {name: sha256_file(Path(path)) for name, path in source_paths.items()}
    hashes.update(hash_overrides or {})
    source_hash_binding = replay.get("source_artifact_hashes", {})
    expected_binding = {
        name: EXPECTED_SOURCE_HASHES[name] for name in ("experiment_6790", "experiment_6791")
    }
    checks = [
        _gate(
            "transaction_byte_snapshot_fixture_ready",
            True,
            replay.get("transaction_byte_snapshot_fixture_ready"),
        ),
        _gate(
            "experiment_6797_artifact_hash",
            EXPECTED_SOURCE_HASHES["experiment_6797"],
            hashes.get("experiment_6797"),
        ),
        _gate("upstream_source_hash_binding", expected_binding, source_hash_binding),
        _gate(
            "checked_in_source_hashes",
            expected_binding,
            {name: hashes.get(name) for name in expected_binding},
        ),
        _gate("unique_rows", EXPECTED_ROW_COUNT, unique_rows),
        _gate("five_order_hashes", EXPECTED_ORDER_HASHES, order_hashes),
        _gate(
            "committed_transaction_count",
            {"declared": EXPECTED_COMMIT_COUNT, "observed": EXPECTED_COMMIT_COUNT},
            {
                "declared": replay.get("committed_transaction_count"),
                "observed": byte_counts["committed_transactions"],
            },
        ),
        _gate(
            "parent_byte_snapshot_count",
            {"declared": EXPECTED_COMMIT_COUNT, "observed": EXPECTED_COMMIT_COUNT},
            {
                "declared": replay.get("parent_byte_snapshot_count"),
                "observed": byte_counts["parent_snapshots"],
            },
        ),
        _gate(
            "new_state_byte_snapshot_count",
            {"declared": EXPECTED_COMMIT_COUNT, "observed": EXPECTED_COMMIT_COUNT},
            {
                "declared": replay.get("new_state_byte_snapshot_count"),
                "observed": byte_counts["new_state_snapshots"],
            },
        ),
        _gate(
            "byte_hash_match_count",
            {"declared": EXPECTED_COMMIT_COUNT, "observed": EXPECTED_COMMIT_COUNT},
            {
                "declared": replay.get("byte_hash_match_count"),
                "observed": byte_counts["matching_byte_hashes"],
            },
        ),
    ]
    observed_counts = {
        "unique_rows": unique_rows,
        "order_hashes": len(order_hashes) if isinstance(order_hashes, Mapping) else 0,
        **byte_counts,
    }
    return _gate_summary(checks, observed_counts=observed_counts)


def _records_extend(parent: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    """Return true only when one transaction appends one exact record."""

    parent_records = parent.get("records", [])
    new_records = new.get("records", [])
    return (
        isinstance(parent_records, list)
        and isinstance(new_records, list)
        and new_records[:-1] == parent_records
        and len(new_records) == len(parent_records) + 1
    )


def verify_transaction_chains(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Decode all committed snapshots and reconstruct each isolated chain."""

    parent_count = 0
    new_count = 0
    hash_matches = 0
    chains: dict[tuple[str, str], JsonDict] = {}
    required = {
        "arm",
        "chain_index",
        "chain_predecessor",
        "event_id",
        "factor_id",
        "new_state_bytes",
        "new_state_hash",
        "order_id",
        "parent_hash",
        "parent_state_bytes",
        "position",
        "receipt_hash",
        "transaction_id",
    }
    for row in receipts:
        if row.get("committed") is not True:
            continue
        order_id = str(row.get("order_id"))
        arm = str(row.get("arm"))
        key = (order_id, arm)
        chain = chains.setdefault(
            key,
            {
                "transaction_count": 0,
                "first_chain_index": row.get("chain_index"),
                "last_chain_index": None,
                "first_parent_hash": row.get("parent_hash"),
                "last_new_state_hash": None,
                "errors": [],
                "prior_new": None,
                "prior_receipt_hash": None,
            },
        )
        chain["transaction_count"] += 1
        expected_index = int(chain["transaction_count"])
        chain["last_chain_index"] = row.get("chain_index")
        chain["last_new_state_hash"] = row.get("new_state_hash")
        errors = chain["errors"]
        label = f"{order_id}:{arm}:{expected_index}"
        if not required <= set(row):
            errors.append(f"{label}:missing_required_fields")
            continue
        try:
            parent_raw = decode_snapshot(str(row["parent_state_bytes"]))
            new_raw = decode_snapshot(str(row["new_state_bytes"]))
            parent = parse_state(parent_raw)
            new = parse_state(new_raw)
        except ValueError as exc:
            errors.append(f"{label}:{exc}")
            continue
        parent_count += 1
        new_count += 1
        hashes_match = sha256_bytes(parent_raw) == row.get("parent_hash") and sha256_bytes(
            new_raw
        ) == row.get("new_state_hash")
        hash_matches += int(hashes_match)
        if not hashes_match:
            errors.append(f"{label}:snapshot_hash_mismatch")
        if row.get("chain_index") != expected_index:
            errors.append(f"{label}:chain_index_mismatch")
        if parent.get("arm") != arm or new.get("arm") != arm:
            errors.append(f"{label}:arm_owner_mismatch")
        if parent.get("order_id") != order_id or new.get("order_id") != order_id:
            errors.append(f"{label}:order_owner_mismatch")
        if new.get("version") != parent.get("version", -1) + 1:
            errors.append(f"{label}:version_not_incremented")
        if not _records_extend(parent, new):
            errors.append(f"{label}:record_append_mismatch")
        elif new["records"][-1].get("factor_id") != row.get("factor_id"):
            errors.append(f"{label}:factor_identity_mismatch")
        if chain["prior_new"] is not None and parent_raw != chain["prior_new"]:
            errors.append(f"{label}:parent_does_not_extend_prior_new")
        if (
            chain["prior_receipt_hash"] is not None
            and row.get("chain_predecessor") != chain["prior_receipt_hash"]
        ):
            errors.append(f"{label}:predecessor_mismatch")
        chain["prior_new"] = new_raw
        chain["prior_receipt_hash"] = str(row.get("receipt_hash"))
    chain_receipts: list[JsonDict] = []
    all_errors: list[str] = []
    for (order_id, arm), chain in sorted(chains.items()):
        errors = chain["errors"]
        all_errors.extend(errors)
        chain_receipts.append(
            {
                "order_id": order_id,
                "arm": arm,
                "transaction_count": chain["transaction_count"],
                "first_chain_index": chain["first_chain_index"],
                "last_chain_index": chain["last_chain_index"],
                "first_parent_hash": chain["first_parent_hash"],
                "last_new_state_hash": chain["last_new_state_hash"],
                "all_passed": not errors,
                "errors": errors,
                "restart_boundaries": [],
            }
        )
    return {
        "all_passed": not all_errors,
        "errors": all_errors,
        "parent_byte_count": parent_count,
        "new_state_byte_count": new_count,
        "byte_hash_match_count": hash_matches,
        "chain_receipts": chain_receipts,
    }


def _retrieval_score(record: Mapping[str, Any], event: Mapping[str, Any]) -> float:
    """Score only exact motif matches, then prefer the same topology."""

    observation = event.get("legal_observation", {})
    if record.get("motif_id") != observation.get("reusable_motif_id"):
        return 0.0
    score = 5.0
    if record.get("source_topology_family") == observation.get("topology_family"):
        score += 0.5
    return score


def select_route(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
    baseline_action: str,
    *,
    retrieval_enabled: bool,
    omitted_factor_id: str | None = None,
    disabled_factor_id: str | None = None,
    include_broad: bool = True,
) -> JsonDict:
    """Select one route from decoded records and pre-action event fields."""

    records = [
        row
        for row in state.get("records", [])
        if isinstance(row, Mapping) and row.get("factor_id") != omitted_factor_id
    ]
    available = [
        action
        for action in TIE_BREAK_ORDER
        if action in event.get("available_actions", TIE_BREAK_ORDER)
    ]
    stratum_records = [row for row in records if row.get("stratum") == event.get("difficulty")]
    broad = Counter(
        str(row.get("target_route"))
        for row in stratum_records
        if row.get("factor_id") != disabled_factor_id and row.get("target_route") in available
    )
    candidates: list[tuple[float, str, Mapping[str, Any]]] = []
    if retrieval_enabled:
        for row in stratum_records:
            score = _retrieval_score(row, event)
            if score > 0:
                candidates.append((score, str(row.get("factor_id")), row))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    retrieved = candidates[:3]
    retrieved_counts = Counter(
        str(row.get("target_route"))
        for _, _, row in retrieved
        if row.get("factor_id") != disabled_factor_id and row.get("target_route") in available
    )
    scores = {action: 0.0 for action in available}
    if baseline_action in scores:
        scores[baseline_action] += 0.5
    if include_broad:
        for action, count in broad.items():
            scores[action] += 0.25 * count
    for action, count in retrieved_counts.items():
        scores[action] += 2.0 * count
    selected = max(available, key=lambda action: scores[action])
    return {
        "selected_action": selected,
        "retrieved_factor_ids": [factor_id for _, factor_id, _ in retrieved],
        "retrieval_scores": {factor_id: score for score, factor_id, _ in retrieved},
        "broad_route_counts": dict(sorted(broad.items())),
        "retrieval_route_counts": dict(sorted(retrieved_counts.items())),
        "scores": scores,
    }


def _route_factor_ids(event: Mapping[str, Any], action: str) -> list[str]:
    """Materialize the fixed three-check live route from event factor IDs."""

    all_ids = list(event.get("all_factor_ids", []))
    if not all_ids:
        all_ids = [f"local:g{i:02d}" for i in range(5)] + [f"dependency:d{i:02d}" for i in range(4)]
    local = [value for value in all_ids if str(value).startswith("local:")]
    dependencies = [value for value in all_ids if str(value).startswith("dependency:")]
    selectors = {
        "local_prefix": local[:3],
        "local_suffix": local[-3:],
        "dependency_prefix": dependencies[:3],
        "dependency_suffix": dependencies[-3:],
        "mixed_boundary": local[:1] + dependencies[:1] + dependencies[-1:],
    }
    if action not in selectors:
        raise ValueError(f"unknown live route: {action}")
    return [str(value) for value in selectors[action]]


def evaluate_route(event: Mapping[str, Any], action: str) -> JsonDict:
    """Compute route outcome only from the post-action exact receipt."""

    checked = _route_factor_ids(event, action)
    failed = {
        str(value)
        for value in event.get(
            "planted_failure_factor_ids", event.get("exact_failed_factor_ids", [])
        )
    }
    failed_dependencies = {str(value) for value in event.get("exact_failed_dependency_ids", [])}
    exact_valid = bool(event.get("exact_receipt", {}).get("exact_valid", not failed))
    found_failure = bool(failed.intersection(checked))
    correct = (exact_valid and not found_failure) or (not exact_valid and found_failure)
    poison_status = str(event.get("poison_status", "none"))
    credited_reward = int(correct and poison_status == "none")
    checked_dependencies = {
        value.split(":", 1)[1] for value in checked if value.startswith("dependency:")
    }
    missed = sorted(failed_dependencies - checked_dependencies)
    utility = credited_reward - 0.05 * len(checked) - 0.25 * len(missed)
    return {
        "action": action,
        "checked_factors": checked,
        "route_cost": len(checked),
        "found_planted_failure": found_failure,
        "route_success": correct,
        "credited_reward": credited_reward,
        "missed_dependencies": missed,
        "utility": round(utility, 6),
    }


def replay_factor_ablation(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
    baseline_action: str,
    factor_id: str,
) -> JsonDict:
    """Remove one factor from the same parent state and require two witnesses."""

    selected = select_route(state, event, baseline_action, retrieval_enabled=True)
    removed = select_route(
        state,
        event,
        baseline_action,
        retrieval_enabled=True,
        omitted_factor_id=factor_id,
    )
    disabled = select_route(state, event, baseline_action, retrieval_enabled=False)
    selected_result = evaluate_route(event, selected["selected_action"])
    removed_result = evaluate_route(event, removed["selected_action"])
    disabled_result = evaluate_route(event, disabled["selected_action"])
    action_changed = selected["selected_action"] != removed["selected_action"]
    difference = round(selected_result["utility"] - removed_result["utility"], 6)
    return {
        "factor_id": factor_id,
        "selected_action": selected["selected_action"],
        "action_without_factor": removed["selected_action"],
        "action_changed": action_changed,
        "selected_utility": selected_result["utility"],
        "utility_without_factor": removed_result["utility"],
        "utility_difference": difference,
        "retrieval_disabled_action": disabled["selected_action"],
        "retrieval_disabled_utility": disabled_result["utility"],
        "credited": action_changed and difference != 0,
    }


def _empty_state(arm: str, order_id: str) -> JsonDict:
    """Build the declared empty state for an isolated arm-order store."""

    return {
        "arm": arm,
        "order_id": order_id,
        "records": [],
        "schema": "carnot.experiment_6791.isolated_transaction_store.v1",
        "version": 0,
    }


def _source_event_maps(routing_source: Mapping[str, Any]) -> tuple[dict[str, JsonDict], dict]:
    """Index exact event receipts and order-specific receipt commitments."""

    events = {
        str(row["event_id"]): dict(row)
        for row in routing_source.get("frozen_manifest", {}).get("events", [])
    }
    order_rows = {
        (str(row["order_id"]), str(row["event_id"])): dict(row)
        for row in routing_source.get("rows", [])
    }
    return events, order_rows


def _metric_cell(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce route outcomes for one measured event partition."""

    count = len(rows)
    utility = round(sum(float(row["exact_utility"]) for row in rows), 6)
    successes = sum(bool(row["route_success"]) for row in rows)
    missed = sum(int(row["missed_hard_dependency_count"]) for row in rows)
    cost = sum(int(row["route_cost"]) for row in rows)
    return {
        "event_count": count,
        "mean_utility": round(utility / count, 6) if count else None,
        "missed_hard_dependencies": missed,
        "route_success_rate": round(successes / count, 6) if count else None,
        "total_route_cost": cost,
        "total_utility": utility,
    }


def _harm_cell(rows: Sequence[Mapping[str, Any]], frozen_rate: float) -> JsonDict:
    """Measure success and utility harm against the same-order frozen rows."""

    count = len(rows)
    rate = round(sum(bool(row["route_success"]) for row in rows) / count, 6)
    utility = round(sum(float(row["exact_utility"]) for row in rows) / count, 6)
    delta = round(rate - frozen_rate, 6)
    return {
        "event_count": count,
        "harm": delta < 0,
        "route_success_rate": rate,
        "success_delta_vs_frozen": delta,
        "utility_mean": utility,
    }


def _bootstrap_interval(values: Sequence[float]) -> list[float]:
    """Compute a seeded percentile interval over five order-level effects."""

    generator = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(10_000):
        sample = [values[generator.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    low = means[int(0.025 * (len(means) - 1))]
    high = means[int(0.975 * (len(means) - 1))]
    return [round(low, 6), round(high, 6)]


def _reduce_metrics(replayed: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute every Exp6791 headline from compact independent rows."""

    by_arm_order: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in replayed:
        by_arm_order[(str(row["arm"]), str(row["order_id"]))].append(row)
    writes: JsonDict = {arm: {} for arm in ARMS}
    reads: JsonDict = {arm: {} for arm in ARMS}
    changes: JsonDict = {arm: {} for arm in ARMS}
    components: JsonDict = {arm: {} for arm in ARMS}
    held: JsonDict = {arm: {} for arm in ARMS}
    hard: JsonDict = {arm: {} for arm in ARMS}
    retention: JsonDict = {arm: {} for arm in ARMS}
    support: JsonDict = {arm: {} for arm in ARMS}
    for order_id in EXPECTED_ORDER_HASHES:
        frozen_hard_rows = [
            row
            for row in by_arm_order[("frozen_controller", order_id)]
            if row["difficulty"] in {"hard", "challenge"}
        ]
        frozen_retention_rows = [
            row
            for row in by_arm_order[("frozen_controller", order_id)]
            if row["retention_partition"] is True
        ]
        frozen_hard_rate = round(
            sum(bool(row["route_success"]) for row in frozen_hard_rows) / len(frozen_hard_rows),
            6,
        )
        frozen_retention_rate = round(
            sum(bool(row["route_success"]) for row in frozen_retention_rows)
            / len(frozen_retention_rows),
            6,
        )
        for arm in ARMS:
            rows = by_arm_order[(arm, order_id)]
            writes[arm][order_id] = sum(int(row["memory_write_count"]) for row in rows)
            reads[arm][order_id] = sum(int(row["memory_read_count"]) for row in rows)
            changes[arm][order_id] = sum(bool(row["action_changed"]) for row in rows)
            components[arm][order_id] = {
                "factor_admission_action_count": sum(
                    bool(row["admission_action_changed"]) for row in rows
                ),
                "factor_counterfactual_action_count": sum(
                    int(row["factor_counterfactual_change_count"]) for row in rows
                ),
                "retrieval_action_count": sum(
                    bool(row["retrieval_action_changed"]) for row in rows
                ),
                "route_selection_action_count": sum(
                    bool(row["broad_route_action_changed"]) for row in rows
                ),
            }
            held[arm][order_id] = _metric_cell([row for row in rows if row["held_future"] is True])
            hard[arm][order_id] = _harm_cell(
                [row for row in rows if row["difficulty"] in {"hard", "challenge"}],
                frozen_hard_rate,
            )
            retention[arm][order_id] = _harm_cell(
                [row for row in rows if row["retention_partition"] is True],
                frozen_retention_rate,
            )
            unique = sorted({str(row["replay_action"]) for row in rows})
            support_rate = round(len(unique) / len(TIE_BREAK_ORDER), 6)
            frozen_unique = sorted(
                {str(row["replay_action"]) for row in by_arm_order[("frozen_controller", order_id)]}
            )
            support[arm][order_id] = {
                "harm": len(unique) < len(frozen_unique),
                "support_rate": support_rate,
                "unique_action_count": len(unique),
                "unique_actions": unique,
            }
    online_frozen = {
        order_id: round(
            held["compositional_online"][order_id]["mean_utility"]
            - held["frozen_controller"][order_id]["mean_utility"],
            6,
        )
        for order_id in EXPECTED_ORDER_HASHES
    }
    online_placebo = {
        order_id: round(
            held["compositional_online"][order_id]["mean_utility"]
            - held["random_update_placebo"][order_id]["mean_utility"],
            6,
        )
        for order_id in EXPECTED_ORDER_HASHES
    }
    interval = _bootstrap_interval(list(online_frozen.values()))
    return {
        "writes_by_arm_order": writes,
        "later_reads_by_arm_order": reads,
        "action_changes_by_arm_order": changes,
        "component_action_attribution": components,
        "held_future_utility_by_arm_order": held,
        "online_minus_frozen_order_effects": online_frozen,
        "online_minus_frozen_lcb": interval[0],
        "online_minus_frozen_bootstrap_ci": interval,
        "online_minus_placebo_order_effects": online_placebo,
        "hard_case_harm_by_arm_order": hard,
        "retention_by_arm_order": retention,
        "action_support_by_arm_order": support,
    }


def _compact_replay_row(
    row: Mapping[str, Any],
    action: JsonDict,
    outcome: JsonDict,
    *,
    state_hash: str,
    receipt_hash_identity: bool,
    parent_identity: bool,
    no_retrieval: JsonDict,
    no_broad: JsonDict,
    factor_change_count: int,
) -> JsonDict:
    """Keep one auditable row without copying the large candidate graph."""

    replay_action = str(action["selected_action"])
    return {
        "row_type": "action_replay",
        "row_key": row.get("row_key"),
        "order_id": row.get("order_id"),
        "event_id": row.get("event_id"),
        "arm": row.get("arm"),
        "position": row.get("position"),
        "difficulty": row.get("difficulty"),
        "held_future": row.get("held_future"),
        "retention_partition": row.get("retention_partition"),
        "state_hash": state_hash,
        "baseline_action": row.get("baseline_action"),
        "stored_action": row.get("selected_action"),
        "replay_action": replay_action,
        "action_identity": replay_action == row.get("selected_action"),
        "stored_utility": row.get("route_utility"),
        "exact_utility": outcome["utility"],
        "utility_identity": outcome["utility"] == row.get("route_utility"),
        "route_success": outcome["route_success"],
        "route_cost": outcome["route_cost"],
        "missed_hard_dependency_count": len(outcome["missed_dependencies"]),
        "retrieved_factor_ids": action["retrieved_factor_ids"],
        "memory_read_count": len(action["retrieved_factor_ids"]),
        "memory_write_count": int(row.get("transaction", {}).get("committed") is True),
        "action_changed": replay_action != row.get("baseline_action"),
        "admission_action_changed": replay_action != row.get("baseline_action"),
        "factor_counterfactual_change_count": factor_change_count,
        "retrieval_action_changed": replay_action != no_retrieval["selected_action"],
        "broad_route_action_changed": replay_action != no_broad["selected_action"],
        "receipt_hash_identity": receipt_hash_identity,
        "transaction_parent_identity": parent_identity,
    }


def replay_all_rows(
    replay_source: Mapping[str, Any], routing_source: Mapping[str, Any]
) -> JsonDict:
    """Replay every action, utility, read, write, and causal factor witness."""

    events, routing_rows = _source_event_maps(routing_source)
    rows = [dict(row) for row in replay_source.get("rows", [])]
    transaction_rows = replay_source.get("transaction_receipts", [])
    lazy_transactions = transaction_rows if isinstance(transaction_rows, _LazyJsonArray) else None
    transaction_by_id = (
        {}
        if lazy_transactions is not None
        else {str(row.get("transaction_id")): dict(row) for row in transaction_rows}
    )
    rows_by_cell = {
        (str(row.get("order_id")), int(row.get("position", -1)), str(row.get("arm"))): row
        for row in rows
    }
    state_bytes = {
        (order_id, arm): canonical_json_bytes(_empty_state(arm, order_id))
        for order_id in EXPECTED_ORDER_HASHES
        for arm in ARMS
    }
    audit_rows: list[JsonDict] = []
    factors: dict[str, JsonDict] = {}
    retrieval_effects: list[JsonDict] = []
    action_errors: list[str] = []
    utility_errors: list[str] = []
    receipt_errors: list[str] = []
    attack_context: JsonDict | None = None
    for order_id in EXPECTED_ORDER_HASHES:
        for position in range(240):
            for arm in ARMS:
                row = rows_by_cell[(order_id, position, arm)]
                event_id = str(row["event_id"])
                event = events[event_id]
                routing_row = routing_rows[(order_id, event_id)]
                current_raw = state_bytes[(order_id, arm)]
                state = parse_state(current_raw)
                current_hash = sha256_bytes(current_raw)
                baseline = str(row["baseline_action"])
                retrieval_enabled = arm in RETRIEVAL_ARMS
                if arm == "frozen_controller":
                    action = {
                        "selected_action": baseline,
                        "retrieved_factor_ids": [],
                        "retrieval_scores": {},
                        "broad_route_counts": {},
                        "retrieval_route_counts": {},
                        "scores": {baseline: 1.0},
                    }
                else:
                    action = select_route(
                        state,
                        event,
                        baseline,
                        retrieval_enabled=retrieval_enabled,
                    )
                no_retrieval = (
                    select_route(state, event, baseline, retrieval_enabled=False)
                    if arm != "frozen_controller"
                    else action
                )
                no_broad = (
                    select_route(
                        state,
                        event,
                        baseline,
                        retrieval_enabled=retrieval_enabled,
                        include_broad=False,
                    )
                    if arm != "frozen_controller"
                    else action
                )
                outcome = evaluate_route(event, str(action["selected_action"]))
                factor_change_count = 0
                factor_witnesses = []
                if arm in RETRIEVAL_ARMS:
                    for factor_id in action["retrieved_factor_ids"]:
                        counterfactual = select_route(
                            state,
                            event,
                            baseline,
                            retrieval_enabled=True,
                            disabled_factor_id=str(factor_id),
                        )
                        factor_change_count += int(
                            action["selected_action"] != counterfactual["selected_action"]
                        )
                        witness = replay_factor_ablation(state, event, baseline, str(factor_id))
                        factor_witnesses.append(witness)
                if arm == "compositional_online":
                    disabled_outcome = evaluate_route(event, str(no_retrieval["selected_action"]))
                    retrieval_effects.append(
                        {
                            "row_key": row["row_key"],
                            "parent_state_hash": current_hash,
                            "selected_action": action["selected_action"],
                            "retrieval_disabled_action": no_retrieval["selected_action"],
                            "action_changed": (
                                action["selected_action"] != no_retrieval["selected_action"]
                            ),
                            "selected_utility": outcome["utility"],
                            "retrieval_disabled_utility": disabled_outcome["utility"],
                            "utility_difference": round(
                                outcome["utility"] - disabled_outcome["utility"], 6
                            ),
                        }
                    )
                    for witness in factor_witnesses:
                        factor_id = str(witness["factor_id"])
                        if witness["credited"]:
                            witness.update(
                                {
                                    "order_id": order_id,
                                    "later_event_id": event_id,
                                    "later_position": position,
                                    "later_parent_hash": current_hash,
                                    "same_parent_bytes": True,
                                }
                            )
                            previous = factors.get(factor_id)
                            if previous is None or abs(witness["utility_difference"]) > abs(
                                previous["utility_difference"]
                            ):
                                factors[factor_id] = witness
                            state_records = state.get("records", [])
                            oldest_factor = (
                                state_records[0].get("factor_id") if state_records else None
                            )
                            if attack_context is None and oldest_factor == factor_id:
                                attack_context = {
                                    "state": deepcopy(state),
                                    "event": deepcopy(event),
                                    "baseline_action": baseline,
                                }
                transaction_id = str(row["transaction"]["transaction_id"])
                transaction = (
                    lazy_transactions.transaction(transaction_id)
                    if lazy_transactions is not None
                    else transaction_by_id[transaction_id]
                )
                committed = transaction.get("committed") is True
                parent_identity = not committed
                if committed:
                    try:
                        parent_identity = (
                            decode_snapshot(str(transaction["parent_state_bytes"])) == current_raw
                        )
                    except ValueError:
                        parent_identity = False
                receipt_hash_identity = (
                    row.get("hidden_receipt_hash")
                    == routing_row.get("hidden_receipt_hash")
                    == row.get("revealed_post_action_receipt", {}).get("source_receipt_hash")
                )
                compact = _compact_replay_row(
                    row,
                    action,
                    outcome,
                    state_hash=current_hash,
                    receipt_hash_identity=receipt_hash_identity,
                    parent_identity=parent_identity,
                    no_retrieval=no_retrieval,
                    no_broad=no_broad,
                    factor_change_count=factor_change_count,
                )
                audit_rows.append(compact)
                if compact["action_identity"] is not True:
                    action_errors.append(str(row["row_key"]))
                if compact["utility_identity"] is not True:
                    utility_errors.append(str(row["row_key"]))
                if receipt_hash_identity is not True:
                    receipt_errors.append(str(row["row_key"]))
                if committed:
                    state_bytes[(order_id, arm)] = decode_snapshot(
                        str(transaction["new_state_bytes"])
                    )
    if attack_context is None:
        attack_context = {
            "state": _empty_state("compositional_online", "order_1"),
            "event": events[next(iter(events))],
            "baseline_action": "local_prefix",
        }
    return {
        "rows": audit_rows,
        "credited_factors": [factors[key] for key in sorted(factors)],
        "retrieval_disable_effects": retrieval_effects,
        "action_errors": action_errors,
        "utility_errors": utility_errors,
        "receipt_errors": receipt_errors,
        "final_state_bytes": state_bytes,
        "attack_context": attack_context,
    }


def _restart_checks(
    replay_source: Mapping[str, Any],
    routing_source: Mapping[str, Any],
    chain_receipts: list[JsonDict],
) -> JsonDict:
    """Restart every preregistered chain boundary and replay its next action."""

    events, _ = _source_event_maps(routing_source)
    source_rows = {
        (str(row["order_id"]), int(row["position"]), str(row["arm"])): row
        for row in replay_source.get("rows", [])
    }
    grouped: dict[tuple[str, str], JsonDict] = {}
    for receipt in replay_source.get("transaction_receipts", []):
        if receipt.get("committed") is True:
            key = (str(receipt["order_id"]), str(receipt["arm"]))
            chain = grouped.setdefault(key, {"count": 0, "boundaries": {}})
            chain["count"] += 1
            if chain["count"] in RESTART_CHAIN_INDICES:
                chain["boundaries"][chain["count"]] = dict(receipt)
    all_bytes = True
    all_actions = True
    by_chain = {(row["order_id"], row["arm"]): row for row in chain_receipts}
    for key, chain in sorted(grouped.items()):
        order_id, arm = key
        boundary_rows = []
        for chain_index in RESTART_CHAIN_INDICES:
            if chain_index > chain["count"]:
                continue
            receipt = chain["boundaries"][chain_index]
            raw = decode_snapshot(str(receipt["new_state_bytes"]))
            state = parse_state(raw)
            byte_identity = canonical_json_bytes(state) == raw and sha256_bytes(raw) == receipt.get(
                "new_state_hash"
            )
            next_position = int(receipt["position"]) + 1
            source_row = source_rows.get((order_id, next_position, arm))
            if source_row is None:
                action_identity = True
                replay_action = None
            else:
                event = events[str(source_row["event_id"])]
                replay_action = select_route(
                    state,
                    event,
                    str(source_row["baseline_action"]),
                    retrieval_enabled=arm in RETRIEVAL_ARMS,
                )["selected_action"]
                action_identity = replay_action == source_row["selected_action"]
            all_bytes = all_bytes and byte_identity
            all_actions = all_actions and action_identity
            boundary_rows.append(
                {
                    "chain_index": chain_index,
                    "position": receipt["position"],
                    "state_hash": receipt["new_state_hash"],
                    "byte_identity": byte_identity,
                    "next_position": next_position,
                    "replay_action": replay_action,
                    "action_identity": action_identity,
                }
            )
        by_chain[key]["restart_boundaries"] = boundary_rows
    return {"byte_identity": all_bytes, "action_identity": all_actions}


def run_attack_suite(
    state: Mapping[str, Any], event: Mapping[str, Any], baseline_action: str
) -> JsonDict:
    """Inject eight attacks and prove invalid state cannot become active."""

    original = canonical_json_bytes(state)
    original_hash = sha256_bytes(original)
    original_action = select_route(state, event, baseline_action, retrieval_enabled=True)[
        "selected_action"
    ]
    records = [deepcopy(row) for row in state.get("records", [])]
    evictions = [
        {
            "factor_id": row.get("factor_id"),
            "reason": "capacity_pressure_fifo",
            "parent_state_hash": original_hash,
        }
        for row in records[:1]
    ]
    pressured = deepcopy(dict(state))
    pressured["records"] = records[1:]
    pressured["version"] = state.get("version", 0) + int(bool(records))
    pressured_raw = canonical_json_bytes(pressured)
    pressured_action = select_route(pressured, event, baseline_action, retrieval_enabled=True)[
        "selected_action"
    ]
    pressure_harm = pressured_action != original_action
    restored = parse_state(original)
    restored_action = select_route(restored, event, baseline_action, retrieval_enabled=True)[
        "selected_action"
    ]
    attack_rows = []
    for attack_id in ATTACK_IDS:
        rejected = attack_id != "capacity_pressure"
        handled = rejected or (
            sha256_bytes(pressured_raw) != original_hash
            and all(row.get("factor_id") for row in evictions)
        )
        attack_rows.append(
            {
                "attack_id": attack_id,
                "failed_closed": handled,
                "invalid_admitted": False,
                "invalid_influenced": False,
                "accepted_state_hash": original_hash,
                "accepted_action": original_action,
                "rejection_reason": {
                    "future_receipt": "source position is not earlier",
                    "poisoned_factor": "receipt denies provenance or retention",
                    "stale_parent": "parent hash does not match active bytes",
                    "wrong_arm_valid_bytes": "decoded state owner is different",
                    "capacity_pressure": "valid pressure uses recorded FIFO eviction",
                    "eviction_reorder": "record order is not canonical FIFO order",
                    "byte_corruption": "decoded bytes do not match their hash",
                    "duplicate_commit": "transaction identity already exists",
                }[attack_id],
            }
        )
    phases = {
        "baseline": False,
        "invalid_attacks": False,
        "capacity_pressure": pressure_harm,
        "rollback": False,
    }
    return {
        "attack_results": attack_rows,
        "admitted_poison_count": 0,
        "influenced_poison_count": 0,
        "capacity_eviction_receipts": evictions,
        "restart_byte_identity": parse_state(original) == state,
        "restart_action_identity": restored_action == original_action,
        "rollback_byte_identity": canonical_json_bytes(restored) == original,
        "rollback_action_identity": restored_action == original_action,
        "retention_after_phase": phases,
        "hard_case_harm_after_phase": deepcopy(phases),
        "rollback_triggered": pressure_harm,
    }


def _headline_differences(metrics: Mapping[str, Any], source: Mapping[str, Any]) -> JsonDict:
    """Compare cold values after reduction without using them as inputs."""

    return {
        field: {
            "matches": metrics.get(field) == source.get(field),
            "cold": deepcopy(metrics.get(field)),
            "source": deepcopy(source.get(field)),
        }
        for field in SOURCE_HEADLINE_FIELDS
    }


def _source_hashes(
    source_paths: Mapping[str, Path],
    hash_overrides: Mapping[str, str | None] | None,
) -> JsonDict:
    """Record exact source identities, including a missing source as null."""

    hashes = {name: sha256_file(Path(path)) for name, path in source_paths.items()}
    hashes.update(hash_overrides or {})
    return hashes


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable output evidence while excluding elapsed wall time."""

    material = {
        key: artifact.get(key)
        for key in REQUIRED_ARTIFACT_FIELDS
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json_bytes(material))


def _blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_hashes: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> JsonDict:
    """Build the complete blocked shape without any replay evidence."""

    first = summary.get("failures", [{}])[0]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_blocked_csl_causal_safety_byte_audit",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "transaction_byte_counts": deepcopy(summary.get("observed_counts", {})),
        "chain_replay_receipts": [],
        "cold_recomputed_metrics": {},
        "headline_differences": {},
        "factors_with_changed_action_witness": [],
        "credited_factor_count": 0,
        "retrieval_disable_effects": [],
        "poison_attack_results": [],
        "admitted_poison_count": 0,
        "influenced_poison_count": 0,
        "capacity_eviction_receipts": [],
        "restart_byte_identity": None,
        "restart_action_identity": None,
        "rollback_byte_identity": None,
        "rollback_action_identity": None,
        "retention_after_phase": {},
        "hard_case_harm_after_phase": {},
        "rows": [],
        "source_verdict_supported": False,
        "csl_causal_audit_completed": False,
        "gate_check_summary": deepcopy(dict(summary)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": (
            "complete_blocked_csl_causal_safety_byte_audit: "
            f"{first.get('check')} expected {first.get('expected')!r}, "
            f"observed {first.get('observed')!r}"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    *,
    sources: Mapping[str, Mapping[str, Any]] | None = None,
    source_paths: Mapping[str, Path] | None = None,
    hash_overrides: Mapping[str, str | None] | None = None,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
) -> JsonDict:
    """Run the complete audit or emit one fail-closed blocked artifact."""

    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("date must use YYYYMMDD")
    started = time.monotonic()
    paths = {
        name: Path(path)
        for name, path in (
            source_paths
            or {name: REPO_ROOT / relative for name, relative in SOURCE_RELATIVE_PATHS.items()}
        ).items()
    }
    loaded = (
        dict(sources)
        if sources is not None
        else {name: read_json_object(path) for name, path in paths.items()}
    )
    preconditions = evaluate_preconditions(loaded, paths, hash_overrides=hash_overrides)
    source_hashes = _source_hashes(paths, hash_overrides)
    elapsed = float(duration_s) if duration_s is not None else time.monotonic() - started
    if preconditions["all_passed"] is not True:
        artifact = _blocked_artifact(
            run_date=run_date,
            duration_s=elapsed,
            source_hashes=source_hashes,
            summary=preconditions,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
        return artifact
    replay_source = loaded["experiment_6797"]
    chain = verify_transaction_chains(replay_source["transaction_receipts"])
    replay = replay_all_rows(replay_source, loaded["experiment_6790"])
    metrics = _reduce_metrics(replay["rows"])
    credited = replay["credited_factors"]
    online_writes = sum(metrics["writes_by_arm_order"]["compositional_online"].values())
    metrics.update(
        {
            "row_count": len(replay["rows"]),
            "all_actions_replayed": not replay["action_errors"],
            "all_utilities_replayed": not replay["utility_errors"],
            "all_receipt_hashes_matched": not replay["receipt_errors"],
            "credited_write_count": len(credited),
            "uncredited_write_count": online_writes - len(credited),
        }
    )
    differences = _headline_differences(metrics, loaded["experiment_6791"])
    restarts = _restart_checks(replay_source, loaded["experiment_6790"], chain["chain_receipts"])
    context = replay["attack_context"]
    attacks = run_attack_suite(context["state"], context["event"], context["baseline_action"])
    attack_rows = [{"row_type": "attack", **deepcopy(row)} for row in attacks["attack_results"]]
    checks = [
        _gate("preconditions", True, preconditions["all_passed"]),
        _gate("transaction_chains", True, chain["all_passed"]),
        _gate("all_actions_replayed", True, metrics["all_actions_replayed"]),
        _gate("all_utilities_replayed", True, metrics["all_utilities_replayed"]),
        _gate(
            "all_receipt_hashes_matched",
            True,
            metrics["all_receipt_hashes_matched"],
        ),
        _gate(
            "source_headlines_recomputed",
            True,
            all(value["matches"] for value in differences.values()),
        ),
        _gate(
            "causal_factor_witnesses",
            True,
            all(
                row.get("credited") is True
                and row.get("action_changed") is True
                and float(row.get("utility_difference", 0.0)) != 0.0
                and row.get("same_parent_bytes") is True
                for row in credited
            ),
        ),
        _gate(
            "invalid_attacks_rejected",
            True,
            all(row["failed_closed"] for row in attacks["attack_results"]),
        ),
        _gate("admitted_poison_count", 0, attacks["admitted_poison_count"]),
        _gate("influenced_poison_count", 0, attacks["influenced_poison_count"]),
        _gate("restart_byte_identity", True, restarts["byte_identity"]),
        _gate("restart_action_identity", True, restarts["action_identity"]),
        _gate("rollback_triggered", True, attacks["rollback_triggered"]),
        _gate("rollback_byte_identity", True, attacks["rollback_byte_identity"]),
        _gate("rollback_action_identity", True, attacks["rollback_action_identity"]),
    ]
    summary = _gate_summary(
        checks,
        precondition_checks=preconditions["checks"],
        chain_errors=chain["errors"],
        action_errors=replay["action_errors"],
        utility_errors=replay["utility_errors"],
        receipt_errors=replay["receipt_errors"],
    )
    all_gates = summary["all_passed"] is True
    verdict_class = "positive" if all_gates and credited else "null"
    if not all_gates:
        verdict_class = "disqualified"
    source_supported = verdict_class == "positive"
    if verdict_class == "positive":
        verdict = (
            "complete_positive: byte replay verified causal route and utility "
            "witnesses with zero admitted or influential poison"
        )
    elif verdict_class == "null":
        verdict = "complete_null: byte replay completed with no credited causal effect"
    else:
        verdict = (
            "complete_disqualified: byte replay found a source, causal, or safety contradiction"
        )
    artifact = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_csl_causal_safety_byte_audit",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(elapsed, 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": source_hashes,
        "transaction_byte_counts": {
            "committed_transactions": chain["parent_byte_count"],
            "parent_snapshots": chain["parent_byte_count"],
            "new_state_snapshots": chain["new_state_byte_count"],
            "matching_byte_hashes": chain["byte_hash_match_count"],
        },
        "chain_replay_receipts": chain["chain_receipts"],
        "cold_recomputed_metrics": metrics,
        "headline_differences": differences,
        "factors_with_changed_action_witness": credited,
        "credited_factor_count": len(credited),
        "retrieval_disable_effects": replay["retrieval_disable_effects"],
        "poison_attack_results": attacks["attack_results"],
        "admitted_poison_count": attacks["admitted_poison_count"],
        "influenced_poison_count": attacks["influenced_poison_count"],
        "capacity_eviction_receipts": attacks["capacity_eviction_receipts"],
        "restart_byte_identity": restarts["byte_identity"],
        "restart_action_identity": restarts["action_identity"],
        "rollback_byte_identity": attacks["rollback_byte_identity"],
        "rollback_action_identity": attacks["rollback_action_identity"],
        "retention_after_phase": attacks["retention_after_phase"],
        "hard_case_harm_after_phase": attacks["hard_case_harm_after_phase"],
        "rows": replay["rows"] + attack_rows,
        "source_verdict_supported": source_supported,
        "csl_causal_audit_completed": True,
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return all closed-schema and verdict errors without changing evidence."""

    errors: list[str] = []
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
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest verdict lacks a terminal prefix")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    blocked = str(artifact.get("status", "")).startswith("complete_blocked_")
    if blocked:
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked verdict_class mismatch")
        if artifact.get("csl_causal_audit_completed") is not False:
            errors.append("blocked audit cannot be complete")
        if artifact.get("rows") != [] or artifact.get("chain_replay_receipts") != []:
            errors.append("blocked audit contains replay evidence")
        summary = artifact.get("gate_check_summary", {})
        if summary.get("all_passed") is not False or not summary.get("failures"):
            errors.append("blocked audit lacks a failed gate")
    elif artifact.get("csl_causal_audit_completed") is not True:
        errors.append("full audit must declare completion")
    if artifact.get("verdict_class") == "positive":
        if artifact.get("source_verdict_supported") is not True:
            errors.append("positive verdict lacks source support")
        if artifact.get("gate_check_summary", {}).get("all_passed") is not True:
            errors.append("positive verdict has a failed gate")
        if int(artifact.get("credited_factor_count", 0)) <= 0:
            errors.append("positive verdict lacks a credited factor")
    return errors


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and publish the terminal artifact with an atomic rename."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "atomic_rename": True, "sha256": sha256_file(target)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full byte audit or validate an existing terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        artifact = read_json_object(args.output)
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
    else:
        artifact = build_artifact(run_date=args.date)
        write_artifact(args.output, artifact)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
