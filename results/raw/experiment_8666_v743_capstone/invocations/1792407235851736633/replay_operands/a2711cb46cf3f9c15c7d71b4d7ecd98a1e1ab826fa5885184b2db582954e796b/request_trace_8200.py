"""REQ-VERIFY-8200: preserve observed requests without inventing chronology.

A response creation time is not a request issue time. Records without an
authenticated issue timestamp remain evidence, but cannot enter the sequence.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from carnot.verify.exact_request_8188 import key as exact_identity_key
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8200_v708_request_trace_census"
CLI = f"scripts/experiments/{NAME}.py"
SCHEMA = "carnot.exact_request.v1"
IDENTITY_FIELDS = (
    "source_bytes",
    "answer_bytes",
    "model_gguf",
    "model_revision",
    "tokenizer",
    "runtime",
    "rendered_prompt",
    "grammar",
    "seed",
    "generation_parameters",
    "request_schema",
    "chat_template_sha256",
    "quantization",
    "runtime_version",
)
MAPPED_FIELDS = (
    "model_gguf",
    "model_revision",
    "tokenizer",
    "runtime",
    "chat_template_sha256",
    "quantization",
    "runtime_version",
)
DESIGNED = {
    "exact_repeat",
    "forced_refresh",
    "changed_source",
    "restart",
    "retry",
    "warmup",
    "designed_hit",
    "invalidation",
    "duplicate",
    "unconstrained",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts keep the operator informed about actual work."""
    print(f"[8200] {phase} completed={completed} pending={pending}", flush=True)


def seal(path: Path, value: Json) -> Json:
    """New immutable paths protect earlier evidence from accidental replacement."""
    if path.exists():
        raise FileExistsError(path)
    atomic_json(path, value)
    path.chmod(0o444)
    return reference(path)


def reference(path: Path) -> Json:
    """Bind exact bytes so a later reader can reject changes."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def exact_key(row: Json) -> str:
    """Reuse the qualified service's canonical identity encoding."""
    return exact_identity_key(row["identity"])


def unavailable(row: Json) -> list[str]:
    """Missing inputs and constructed conditions are explicit exclusions."""
    reasons = []
    if row.get("operation", "generation") != "generation":
        reasons.append("not_generation")
    if row.get("status") != "completed":
        reasons.append("not_completed")
    if row.get("attempt", 1) != 1 or row.get("historical"):
        reasons.append("retry_restart_or_imported_call")
    if (
        row.get("condition") in DESIGNED
        or row.get("arm") in DESIGNED
        or "warmup" in str(row.get("call_id", ""))
    ):
        reasons.append("designed_call")
    if row.get("condition") not in {"original", "original_source", "full_source", "cold_miss"}:
        reasons.append("unavailable_original_source_condition")
    for field in ("issued_at", "call_id", "session_id", "source_cluster_id", "request", "identity"):
        if not row.get(field):
            reasons.append("unavailable_" + field)
    try:
        stamp = datetime.fromisoformat(str(row.get("issued_at", "")).replace("Z", "+00:00"))
        if stamp.utcoffset() is None:
            reasons.append("unavailable_issue_timezone")
    except ValueError:
        reasons.append("unavailable_issue_timestamp")
    identity = row.get("identity", {})
    if not isinstance(identity, dict) or any(k not in identity for k in IDENTITY_FIELDS):
        reasons.append("unavailable_exact_identity")
    return reasons


def census(records: list[Json], mapping: Json) -> Json:
    """Select by time alone; never replace seeds to obtain a cache hit."""
    eligible, exclusions = [], []
    for index, original in enumerate(records):
        row = deepcopy(original)
        reasons = unavailable(row)
        if reasons:
            exclusions.append(
                dict(
                    inventory_index=index,
                    call_id=row.get("call_id"),
                    exclusion_reason=reasons[0],
                    reasons=reasons,
                )
            )
        else:
            eligible.append(row)
    eligible.sort(
        key=lambda r: (datetime.fromisoformat(r["issued_at"].replace("Z", "+00:00")), r["call_id"])
    )
    selected = eligible[-96:]
    original_seen: dict[str, int] = {}
    replay_seen: dict[str, int] = {}
    shapes: set[str] = set()
    reuse, identities = [], []
    original_hits = replay_hits = shape_hits = 0
    for index, row in enumerate(selected):
        original_key = exact_key(row)
        identity = dict(row["identity"], **{k: v for k, v in mapping.items() if k in MAPPED_FIELDS})
        replay_key = exact_identity_key(identity)
        shape_key = exact_identity_key(
            {k: v for k, v in row["identity"].items() if k not in (*MAPPED_FIELDS, "seed")}
        )
        original_hits += int(original_key in original_seen)
        replay_hits += int(replay_key in replay_seen)
        shape_hits += int(shape_key in shapes)
        reuse.append(
            dict(
                call_id=row["call_id"],
                original_distance=index - original_seen[original_key]
                if original_key in original_seen
                else None,
                replay_distance=index - replay_seen[replay_key]
                if replay_key in replay_seen
                else None,
            )
        )
        identities.append(
            dict(
                call_id=row["call_id"],
                original_key=original_key,
                replay_key=replay_key,
                changed_fields=[
                    k for k in MAPPED_FIELDS if row["identity"].get(k) != identity.get(k)
                ],
            )
        )
        original_seen[original_key] = replay_seen[replay_key] = index
        shapes.add(shape_key)
    count = len(selected)
    independent = len({r["source_cluster_id"] for r in selected})
    schema = {r["identity"]["request_schema"] for r in selected}
    return dict(
        requests=selected,
        eligible_count=len(eligible),
        independent_count=independent,
        completed_count=count,
        excluded_count=len(exclusions),
        exclusion_rows=exclusions,
        ready=count == 96 and independent >= 24 and schema == {SCHEMA},
        original_exact_repeat_frequency=original_hits / count if count else None,
        replay_exact_repeat_frequency=replay_hits / count if count else None,
        original_exact_hit_count=original_hits,
        replay_exact_hit_count=replay_hits,
        request_shape_duplicate_count=shape_hits,
        reuse_distance_rows=reuse,
        identity_mapping_rows=identities,
        supported_schema=schema == {SCHEMA},
    )


def replay(path: Path) -> bool:
    """Independent reopening catches changed evidence and forged headline counts."""
    try:
        value = json.loads(path.read_text())
        for ref in value["raw_shard_hashes"]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for ref in value["source_artifact_hashes"]:
            if sha256_file(Path(ref.get("frozen_path", ref["path"]))) != ref["sha256"]:
                return False
        frozen = json.loads(Path(value["primitive_inventory_path"]).read_text())
        result = census(frozen["records"], frozen["replay_identity"])
        trace = json.loads(Path(value["trace_path"]).read_text())
        if trace != result or sha256_file(Path(value["trace_path"])) != value["trace_sha256"]:
            return False
        fields = (
            "eligible_count",
            "independent_count",
            "completed_count",
            "excluded_count",
            "original_exact_repeat_frequency",
            "replay_exact_repeat_frequency",
            "reuse_distance_rows",
            "identity_mapping_rows",
            "exclusion_rows",
            "original_exact_hit_count",
            "replay_exact_hit_count",
            "request_shape_duplicate_count",
        )
        if any(value[k] != result[k] for k in fields):
            return False
        expected = int(
            result["ready"]
            and all(c["passed"] for c in value["gate_check_summary"])
            and value["required_checks_passed"]
        )
        return value["request_trace_ready_score"] == expected
    except (OSError, ValueError, KeyError, TypeError):
        return False
