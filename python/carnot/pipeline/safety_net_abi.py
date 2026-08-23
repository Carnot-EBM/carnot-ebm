"""Deterministic Safety-Net router ABI shared with Rust.

Spec refs: REQ-RUSTPY-6550, REQ-RUSTPY-6550-SCHEMA,
REQ-RUSTPY-6550-NUMERIC, REQ-RUSTPY-6550-SERIALIZATION,
REQ-RUSTPY-6550-ERRORS, REQ-RUSTPY-6550-PARITY,
REQ-RUSTPY-6550-FALLBACK, SCENARIO-RUSTPY-6550-BOUNDARY-PARITY.

This module only replays the compact routing decision. It does not inspect
natural language or decide whether an answer can be released.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

from carnot.pipeline.production_safety_net_adapter import (
    DEFAULT_ABSTENTION_THRESHOLD,
    FORBIDDEN_POLICY_FEATURES,
    FROZEN_V566_FEATURE_NAMES,
    FROZEN_V566_ROUTER_CONTRACT_HASH,
)
from carnot.task_runtime_receipts import sha256_bytes, sha256_json


JsonDict = dict[str, Any]

ABI_SCHEMA_VERSION = "carnot.safety_net.router_abi.v1"
MAX_STRUCTURAL_FEATURE_ABS = 1_000_000_000
ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "candidate_ids",
        "feature_values",
        "split_name",
        "seed",
        "router_contract_hash",
        "exception_table",
        "forced_abstain",
        "forced_fallback_reason",
    }
)


def canonical_json(value: Any) -> str:
    """Return stable JSON bytes shared by Python artifacts and Rust tests."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_request_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a request after normalizing integer-like structural features."""

    return canonical_json(_normalized_request_payload(payload)).encode("utf-8")


def canonical_decision_bytes(decision: Mapping[str, Any]) -> bytes:
    """Serialize a decision with the same byte contract as the Rust binding."""

    return canonical_json(dict(decision)).encode("utf-8")


def request_payload(
    *,
    request_id: str,
    candidate_ids: Sequence[str],
    feature_values: Mapping[str, int | float] | None = None,
    split_name: str = "live",
    seed: int = 6550,
    schema_version: str = ABI_SCHEMA_VERSION,
    router_contract_hash: str = FROZEN_V566_ROUTER_CONTRACT_HASH,
    exception_table: Mapping[str, str] | None = None,
    forced_abstain: bool = False,
    forced_fallback_reason: str = "",
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a complete JSON-safe request for tests and artifact rows."""

    payload: JsonDict = {
        "schema_version": schema_version,
        "request_id": request_id,
        "candidate_ids": [str(item) for item in candidate_ids],
        "feature_values": dict(feature_values or {}),
        "split_name": split_name,
        "seed": int(seed),
        "router_contract_hash": router_contract_hash,
        "exception_table": dict(exception_table or {}),
        "forced_abstain": bool(forced_abstain),
        "forced_fallback_reason": forced_fallback_reason,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def exception_key(*, candidate_ids: Sequence[Any], split_name: str) -> str:
    """Hash the train-only exception-table key used by the production adapter."""

    ids = [str(candidate_id) for candidate_id in candidate_ids]
    return sha256_json(
        {
            "candidate_hashes": ids,
            "candidate_count": len(ids),
            "split_name": split_name,
        }
    )


def route_request_bytes(request_bytes: bytes) -> JsonDict:
    """Route one raw request byte string and fail closed on invalid JSON."""

    request_hash = sha256_bytes(bytes(request_bytes))
    try:
        payload = json.loads(
            request_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return _error_decision(
            request_hash=request_hash,
            reason="malformed_input:invalid_json",
            error_type="JsonDecodeError",
        )
    if not isinstance(payload, Mapping):
        return _error_decision(
            request_hash=request_hash,
            reason="malformed_input:not_object",
            error_type="SafetyNetAbiError",
        )
    return route_request(payload, request_hash=request_hash)


def route_request(payload: Mapping[str, Any], *, request_hash: str | None = None) -> JsonDict:
    """Return the compact router decision for a parsed request mapping."""

    if request_hash is not None:
        input_hash = request_hash
    else:
        try:
            input_hash = sha256_bytes(canonical_request_bytes(payload))
        except (TypeError, ValueError):
            input_hash = sha256_bytes(repr(sorted(payload.items())).encode("utf-8"))
    parsed, reason, error_type = _parse_request(payload)
    if parsed is None:
        return _error_decision(
            request_hash=input_hash,
            reason=reason,
            error_type=error_type,
            router_contract_hash=str(payload.get("router_contract_hash", "")),
        )

    original = tuple(parsed["candidate_ids"])
    router_hash = str(parsed["router_contract_hash"])
    if router_hash != FROZEN_V566_ROUTER_CONTRACT_HASH:
        return _fallback_decision(
            request_hash=input_hash,
            original_order=original,
            reason="stale_configuration",
            error_type="SchemaVersionError",
            router_contract_hash=router_hash,
            uncertainty_bucket="unsupported",
        )

    key_hash = exception_key(candidate_ids=original, split_name=str(parsed["split_name"]))
    exception_value = str(parsed["exception_table"].get(key_hash, ""))
    exception_hit = exception_value == "native_exact_fallback"
    if parsed["forced_fallback_reason"]:
        return _fallback_decision(
            request_hash=input_hash,
            original_order=original,
            reason=str(parsed["forced_fallback_reason"]),
            router_contract_hash=router_hash,
            exception_hit=exception_hit,
            uncertainty_bucket=_uncertainty_bucket(len(original)),
        )
    if exception_hit:
        return _fallback_decision(
            request_hash=input_hash,
            original_order=original,
            reason="exception_table_hit",
            router_contract_hash=router_hash,
            exception_hit=True,
            uncertainty_bucket=_uncertainty_bucket(len(original)),
        )

    abstain = bool(parsed["forced_abstain"]) or (
        1.0 / max(len(original) + 1, 1) >= DEFAULT_ABSTENTION_THRESHOLD
    )
    if abstain:
        return _fallback_decision(
            request_hash=input_hash,
            original_order=original,
            reason="abstention",
            router_contract_hash=router_hash,
            abstain=True,
            uncertainty_bucket=_uncertainty_bucket(len(original)),
        )

    return {
        "schema_version": ABI_SCHEMA_VERSION,
        "route": "compact_router",
        "abstain": False,
        "uncertainty_bucket": _uncertainty_bucket(len(original)),
        "exception_hit": False,
        "fallback_reason": "",
        "original_order": list(original),
        "chosen_order": list(reversed(original)),
        "error_type": "",
        "exact_fallback_reachable": True,
        "request_hash": input_hash,
        "router_contract_hash": router_hash,
    }


def exact_downstream_result(decision: Mapping[str, Any]) -> JsonDict:
    """Model the unchanged native exact release result used for parity rows."""

    original = list(decision.get("original_order", []))
    return {
        "release_authority": "native_exact_verifier",
        "verified": bool(original),
        "accepted_candidate_hash": str(original[0]) if original else "",
        "error_type": "" if original else "NoCandidateError",
    }


def nan_attack_request_bytes() -> bytes:
    """Return invalid JSON that exercises the NaN fail-closed branch."""

    return (
        b'{"candidate_ids":["sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],'
        b'"exception_table":{},"feature_values":{"candidate_count":NaN},'
        b'"forced_abstain":false,"forced_fallback_reason":"",'
        b'"request_id":"nan","router_contract_hash":"'
        + FROZEN_V566_ROUTER_CONTRACT_HASH.encode("ascii")
        + b'","schema_version":"'
        + ABI_SCHEMA_VERSION.encode("ascii")
        + b'","seed":6550,"split_name":"live"}'
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _normalized_request_payload(payload: Mapping[str, Any]) -> JsonDict:
    normalized = dict(payload)
    if isinstance(normalized.get("feature_values"), Mapping):
        normalized["feature_values"] = {
            str(key): _normalize_feature_for_serialization(value)
            for key, value in normalized["feature_values"].items()
        }
    if isinstance(normalized.get("exception_table"), Mapping):
        normalized["exception_table"] = {
            str(key): str(value) for key, value in normalized["exception_table"].items()
        }
    if isinstance(normalized.get("candidate_ids"), Sequence) and not isinstance(
        normalized.get("candidate_ids"), str
    ):
        normalized["candidate_ids"] = [str(item) for item in normalized["candidate_ids"]]
    return normalized


def _parse_request(payload: Mapping[str, Any]) -> tuple[JsonDict | None, str, str]:
    extras = set(payload) - ALLOWED_TOP_LEVEL_FIELDS
    if extras:
        return None, "malformed_input:extra_keys", "SafetyNetAbiError"
    schema = payload.get("schema_version")
    if schema is None:
        return None, "schema_version_missing", "SchemaVersionError"
    if schema != ABI_SCHEMA_VERSION:
        return None, "stale_schema_version", "SchemaVersionError"
    candidate_ids_raw = payload.get("candidate_ids")
    if not isinstance(candidate_ids_raw, Sequence) or isinstance(candidate_ids_raw, str):
        return None, "malformed_input:missing_candidate_ids", "SafetyNetAbiError"
    candidate_ids = tuple(str(item) for item in candidate_ids_raw)
    reject_reason = _candidate_reject_reason(candidate_ids)
    if reject_reason:
        return None, reject_reason, "SafetyNetAbiError"

    features_raw = payload.get("feature_values", {})
    if not isinstance(features_raw, Mapping):
        return None, "malformed_input:feature_values_not_object", "SafetyNetAbiError"
    features: dict[str, int] = {}
    for key, value in features_raw.items():
        feature = str(key)
        if feature in FORBIDDEN_POLICY_FEATURES:
            return None, "malformed_input:forbidden_feature", "SafetyNetAbiError"
        if feature not in FROZEN_V566_FEATURE_NAMES:
            return None, "malformed_input:unsupported_feature", "SafetyNetAbiError"
        try:
            features[feature] = _normalize_feature_number(value)
        except ValueError as exc:
            return None, str(exc), "SafetyNetAbiError"

    exception_table_raw = payload.get("exception_table", {})
    if not isinstance(exception_table_raw, Mapping):
        return None, "malformed_input:exception_table_not_object", "SafetyNetAbiError"
    exception_table = {str(key): str(value) for key, value in exception_table_raw.items()}
    forced_abstain = payload.get("forced_abstain", False)
    if not isinstance(forced_abstain, bool):
        return None, "malformed_input:forced_abstain_not_bool", "SafetyNetAbiError"
    forced_fallback_reason = payload.get("forced_fallback_reason", "")
    if not isinstance(forced_fallback_reason, str):
        return None, "malformed_input:forced_fallback_not_string", "SafetyNetAbiError"

    return (
        {
            "candidate_ids": candidate_ids,
            "feature_values": features,
            "split_name": str(payload.get("split_name", "live")),
            "router_contract_hash": str(
                payload.get("router_contract_hash", FROZEN_V566_ROUTER_CONTRACT_HASH)
            ),
            "exception_table": exception_table,
            "forced_abstain": forced_abstain,
            "forced_fallback_reason": forced_fallback_reason,
        },
        "",
        "",
    )


def _candidate_reject_reason(candidate_ids: Sequence[str]) -> str:
    if not candidate_ids:
        return "malformed_input:no_candidates"
    if len(set(candidate_ids)) != len(candidate_ids):
        return "malformed_input:duplicate_candidate_ids"
    if any(not candidate_id.strip() for candidate_id in candidate_ids):
        return "malformed_input:blank_candidate_id"
    if any(not candidate_id.isascii() for candidate_id in candidate_ids):
        return "malformed_input:non_ascii_candidate_id"
    return ""


def _normalize_feature_number(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("malformed_input:non_numeric_feature")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("malformed_input:non_finite_feature")
        if not value.is_integer():
            raise ValueError("malformed_input:non_integer_feature")
    normalized = int(value)
    if abs(normalized) > MAX_STRUCTURAL_FEATURE_ABS:
        raise ValueError("malformed_input:numeric_out_of_range")
    return normalized


def _normalize_feature_for_serialization(value: Any) -> Any:
    """Normalize safe integer floats without hiding invalid JSON feature values."""

    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _uncertainty_bucket(candidate_count: int) -> str:
    if candidate_count <= 1:
        return "high"
    if candidate_count == 2:
        return "medium"
    return "low"


def _fallback_decision(
    *,
    request_hash: str,
    original_order: Sequence[str],
    reason: str,
    router_contract_hash: str = FROZEN_V566_ROUTER_CONTRACT_HASH,
    error_type: str = "",
    abstain: bool = False,
    exception_hit: bool = False,
    uncertainty_bucket: str = "unsupported",
) -> JsonDict:
    return {
        "schema_version": ABI_SCHEMA_VERSION,
        "route": "native_exact_fallback",
        "abstain": bool(abstain or reason == "abstention"),
        "uncertainty_bucket": uncertainty_bucket,
        "exception_hit": bool(exception_hit),
        "fallback_reason": reason,
        "original_order": list(original_order),
        "chosen_order": list(original_order),
        "error_type": error_type,
        "exact_fallback_reachable": True,
        "request_hash": request_hash,
        "router_contract_hash": router_contract_hash,
    }


def _error_decision(
    *,
    request_hash: str,
    reason: str,
    error_type: str,
    router_contract_hash: str = FROZEN_V566_ROUTER_CONTRACT_HASH,
) -> JsonDict:
    return _fallback_decision(
        request_hash=request_hash,
        original_order=(),
        reason=reason,
        error_type=error_type,
        router_contract_hash=router_contract_hash or FROZEN_V566_ROUTER_CONTRACT_HASH,
        uncertainty_bucket="unsupported",
    )
