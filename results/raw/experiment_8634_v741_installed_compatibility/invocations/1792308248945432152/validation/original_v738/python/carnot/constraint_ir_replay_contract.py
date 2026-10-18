"""Shared ConstraintIR artifact replay projection.

Spec refs: REQ-BENCH-5907, SCENARIO-BENCH-5907-CANONICAL,
SCENARIO-BENCH-5907-FRESH-PROCESS, SCENARIO-BENCH-5907-TAMPER.

The Exp5896 producer and Exp5897 consumer both need to decide whether a
ConstraintIR fixture artifact is the same replayable evidence. This module is
the single public checksum contract so that decision does not drift between
entrypoints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any


JsonDict = dict[str, Any]

PROJECTION_SCHEMA_VERSION = "carnot.constraint_ir.replay_contract_projection.v1"
NORMALIZATION_VERSION = "json_sort_keys_ascii_finite_numbers_v1"

EXCLUDED_TOP_LEVEL_FIELDS = (
    "duration_s",
    "protected_files_unchanged",
    "reproducibility_checksum",
    "test_exit_codes",
)
EXCLUDED_NESTED_PATHS = (
    ("preconditions_checked", "disk", "available_mb"),
    ("preconditions_checked", "ram", "available_mb"),
)
BOUND_FIELD_NAMES = (
    "artifact_schema",
    "constraint_ir_schema_version",
    "artifact_schema_version",
    "row_schema_version",
    "row_file_sha256",
)


class ConstraintIRReplayContractError(ValueError):
    """Raised when a ConstraintIR replay projection cannot be trusted."""


def canonical_json(value: Any) -> str:
    """Serialize normalized JSON evidence into stable UTF-8 text."""

    return json.dumps(
        _normalize_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for UTF-8 text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed SHA-256 digest for already canonical bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def attach_projection_receipt(
    artifact: Mapping[str, Any],
    *,
    row_file_sha256: str | None,
) -> JsonDict:
    """Return an artifact copy carrying the public projection metadata."""

    updated = _copy_json(artifact)
    updated["canonical_projection_schema_and_version"] = projection_metadata(
        updated, row_file_sha256=row_file_sha256
    )
    return updated


def projection_metadata(
    artifact: Mapping[str, Any],
    *,
    row_file_sha256: str | None,
) -> JsonDict:
    """Describe the projection version, exclusions, and bound replay fields."""

    return {
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "checksum_field": "reproducibility_checksum",
        "excluded_top_level_fields": list(EXCLUDED_TOP_LEVEL_FIELDS),
        "excluded_nested_paths": [".".join(path) for path in EXCLUDED_NESTED_PATHS],
        "bound_fields": _bound_fields(artifact, row_file_sha256=row_file_sha256),
        "principle": "one explicit public projection owns every producer-consumer checksum",
    }


def projection_receipt(
    artifact: Mapping[str, Any],
    *,
    row_file_sha256: str | None,
    projection_version: str = PROJECTION_SCHEMA_VERSION,
) -> JsonDict:
    """Return the canonical projection checksum and audit metadata."""

    payload = canonical_projection_payload(
        artifact,
        row_file_sha256=row_file_sha256,
        projection_version=projection_version,
    )
    byte_payload = canonical_json(payload).encode("utf-8")
    return {
        "projection_schema_version": projection_version,
        "normalization_version": NORMALIZATION_VERSION,
        "checksum": sha256_bytes(byte_payload),
        "byte_length": len(byte_payload),
        "bound_fields": payload["bound_fields"],
        "row_file_sha256": payload["bound_fields"]["row_file_sha256"],
        "excluded_top_level_fields": list(EXCLUDED_TOP_LEVEL_FIELDS),
        "excluded_nested_paths": [".".join(path) for path in EXCLUDED_NESTED_PATHS],
    }


def canonical_projection_bytes(
    artifact: Mapping[str, Any],
    *,
    row_file_sha256: str | None,
    projection_version: str = PROJECTION_SCHEMA_VERSION,
) -> bytes:
    """Return the exact UTF-8 bytes owned by the public replay projection."""

    payload = canonical_projection_payload(
        artifact,
        row_file_sha256=row_file_sha256,
        projection_version=projection_version,
    )
    return canonical_json(payload).encode("utf-8")


def canonical_projection_payload(
    artifact: Mapping[str, Any],
    *,
    row_file_sha256: str | None,
    projection_version: str = PROJECTION_SCHEMA_VERSION,
) -> JsonDict:
    """Build the versioned projection used for every replay checksum."""

    _require_known_projection_version(projection_version)
    explicit_version = _explicit_projection_version(artifact)
    if explicit_version is not None and explicit_version != projection_version:
        raise ConstraintIRReplayContractError(f"unknown projection version: {explicit_version}")
    _validate_row_file_binding(artifact, row_file_sha256=row_file_sha256)

    projected = _copy_json(artifact)
    for field in EXCLUDED_TOP_LEVEL_FIELDS:
        projected.pop(field, None)
    for path in EXCLUDED_NESTED_PATHS:
        _remove_nested_path(projected, path)
    if isinstance(projected.get("row_file_receipt"), dict):
        projected["row_file_receipt"]["sha256"] = row_file_sha256
    projected["canonical_projection_schema_and_version"] = projection_metadata(
        artifact, row_file_sha256=row_file_sha256
    )

    return {
        "projection_schema_version": projection_version,
        "normalization_version": NORMALIZATION_VERSION,
        "bound_fields": _bound_fields(artifact, row_file_sha256=row_file_sha256),
        "artifact": projected,
    }


def verify_reproducibility_checksum(
    artifact: Mapping[str, Any],
    *,
    row_file_sha256: str | None,
    allow_legacy_without_projection: bool = False,
) -> JsonDict:
    """Check the stored checksum against the public projection when applicable."""

    receipt = projection_receipt(artifact, row_file_sha256=row_file_sha256)
    stored = str(artifact.get("reproducibility_checksum") or "")
    legacy_mode = _explicit_projection_version(artifact) is None
    if legacy_mode and allow_legacy_without_projection:
        receipt.update(
            {
                "stored_checksum": stored,
                "stored_checksum_matched": stored == receipt["checksum"],
                "legacy_mode_without_projection_field": True,
            }
        )
        return receipt
    if stored != receipt["checksum"]:
        raise ConstraintIRReplayContractError("artifact reproducibility checksum mismatch")
    receipt.update(
        {
            "stored_checksum": stored,
            "stored_checksum_matched": True,
            "legacy_mode_without_projection_field": False,
        }
    )
    return receipt


def _require_known_projection_version(version: str) -> None:
    if version != PROJECTION_SCHEMA_VERSION:
        raise ConstraintIRReplayContractError(f"unknown projection version: {version}")


def _explicit_projection_version(artifact: Mapping[str, Any]) -> str | None:
    raw = artifact.get("canonical_projection_schema_and_version")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ConstraintIRReplayContractError("canonical projection metadata must be an object")
    version = raw.get("projection_schema_version")
    if not isinstance(version, str) or not version:
        raise ConstraintIRReplayContractError(
            "canonical projection metadata lacks projection version"
        )
    return version


def _bound_fields(artifact: Mapping[str, Any], *, row_file_sha256: str | None) -> JsonDict:
    schema = artifact.get("constraint_ir_schema_and_version")
    schema_map = schema if isinstance(schema, Mapping) else {}
    return {
        "artifact_schema": artifact.get("schema"),
        "constraint_ir_schema_version": schema_map.get("schema_version"),
        "artifact_schema_version": schema_map.get("artifact_schema_version"),
        "row_schema_version": schema_map.get("row_schema_version"),
        "row_file_sha256": row_file_sha256,
    }


def _validate_row_file_binding(
    artifact: Mapping[str, Any],
    *,
    row_file_sha256: str | None,
) -> None:
    if row_file_sha256 is None:
        return
    if not row_file_sha256.startswith("sha256:"):
        raise ConstraintIRReplayContractError("row file SHA-256 must be prefixed")
    receipt = artifact.get("row_file_receipt")
    if not isinstance(receipt, Mapping):
        raise ConstraintIRReplayContractError("row_file_receipt must be an object")
    recorded = receipt.get("sha256")
    if recorded is not None and recorded != row_file_sha256:
        raise ConstraintIRReplayContractError("row file hash does not match artifact receipt")


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConstraintIRReplayContractError("canonical projection numbers must be finite")
        return value
    raise ConstraintIRReplayContractError(f"unsupported JSON value: {type(value).__name__}")


def _remove_nested_path(target: JsonDict, path: Sequence[str]) -> None:
    cursor: Any = target
    for key in path[:-1]:
        if not isinstance(cursor, dict):
            return
        cursor = cursor.get(key)
    if isinstance(cursor, dict):
        cursor.pop(path[-1], None)
