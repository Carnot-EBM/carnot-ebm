"""Reusable generation-to-verdict path receipt helpers.

Spec refs: REQ-VERIFY-6449, SCENARIO-VERIFY-6449-CHAIN,
SCENARIO-VERIFY-6449-CONTROLS, SCENARIO-VERIFY-6449-ATTACKS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any


JsonDict = dict[str, Any]

SCHEMA_VERSION = "carnot.path_receipt.v1"
GENESIS_HASH = "sha256:" + "0" * 64
REQUIRED_STAGE_NAMES = (
    "raw_generation_bytes",
    "parse_output",
    "typed_facts",
    "energy_input",
    "checker_request",
    "checker_transport",
    "checker_response",
    "final_verdict",
)
REQUIRED_STAGE_FIELDS = (
    "schema_version",
    "unit_id",
    "stage_id",
    "stage_index",
    "stage_name",
    "parent_hash",
    "input_hash",
    "output_hash",
    "code_hash",
    "configuration_hash",
    "monotonic_start_ns",
    "monotonic_end_ns",
    "terminal_exact_outcome",
    "output_payload",
    "stage_hash",
)


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for receipt hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest with the project prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after stable serialization."""

    return sha256_text(canonical_json(value))


def json_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes for a stage payload."""

    return canonical_json(value).encode("utf-8")


def _stage_hash_payload(stage: Mapping[str, Any]) -> JsonDict:
    """Return the stage content that is covered by ``stage_hash``."""

    return {key: value for key, value in stage.items() if key != "stage_hash"}


def refresh_stage_hash(stage: Mapping[str, Any]) -> JsonDict:
    """Recompute and return a copy of one stage with a fresh stage hash."""

    out = dict(stage)
    out["stage_hash"] = sha256_json(_stage_hash_payload(out))
    return out


def build_stage(
    *,
    unit_id: str,
    stage_index: int,
    stage_name: str,
    parent_hash: str,
    input_bytes: bytes,
    output_bytes: bytes,
    code_hash: str,
    configuration_hash: str,
    monotonic_start_ns: int,
    monotonic_end_ns: int,
    terminal_exact_outcome: bool | None,
    output_payload: Mapping[str, Any],
) -> JsonDict:
    """Build one immutable stage receipt."""

    stage = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": unit_id,
        "stage_id": f"{stage_index:02d}-{stage_name}",
        "stage_index": int(stage_index),
        "stage_name": stage_name,
        "parent_hash": parent_hash,
        "input_hash": sha256_bytes(input_bytes),
        "output_hash": sha256_bytes(output_bytes),
        "code_hash": code_hash,
        "configuration_hash": configuration_hash,
        "monotonic_start_ns": int(monotonic_start_ns),
        "monotonic_end_ns": int(monotonic_end_ns),
        "terminal_exact_outcome": terminal_exact_outcome,
        "output_payload": dict(output_payload),
    }
    return refresh_stage_hash(stage)


def verdict_from_checker_response(response: Mapping[str, Any]) -> str:
    """Return the terminal verdict represented by a checker response."""

    return "exact_pass" if response.get("exact_outcome") is True else "exact_fail"


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def validate_stage_chain(
    stages: Sequence[Mapping[str, Any]],
    *,
    allowed_code_hashes: set[str],
) -> JsonDict:
    """Validate the stage order, hash chain, code hashes, and terminal verdict."""

    reasons: list[str] = []
    stage_list = [dict(stage) for stage in stages]
    names = [str(stage.get("stage_name")) for stage in stage_list]
    ids = [str(stage.get("stage_id")) for stage in stage_list]
    if len(set(ids)) != len(ids):
        reasons.append("duplicate_stage_id")
    if len(set(names)) != len(names):
        reasons.append("duplicate_stage_name")
    for expected in REQUIRED_STAGE_NAMES:
        if expected not in names:
            reasons.append(f"missing_stage:{expected}")
    if names != list(REQUIRED_STAGE_NAMES):
        reasons.append("stage_reordering")

    previous_stage: Mapping[str, Any] | None = None
    for index, stage in enumerate(stage_list):
        stage_name = str(stage.get("stage_name", f"stage_{index}"))
        missing = [field for field in REQUIRED_STAGE_FIELDS if field not in stage]
        if missing:
            reasons.append(f"missing_stage_fields:{stage_name}")
            continue
        if stage.get("stage_hash") != sha256_json(_stage_hash_payload(stage)):
            reasons.append(f"stage_hash_mismatch:{stage_name}")
        if stage.get("code_hash") not in allowed_code_hashes:
            reasons.append(f"unknown_code_hash:{stage_name}")
        if int(stage.get("monotonic_end_ns", 0)) < int(stage.get("monotonic_start_ns", 0)):
            reasons.append(f"negative_stage_interval:{stage_name}")
        expected_parent = GENESIS_HASH if previous_stage is None else previous_stage.get("stage_hash")
        if stage.get("parent_hash") != expected_parent:
            reasons.append(f"parent_hash_break:{stage_name}")
        if previous_stage is not None and stage.get("input_hash") != previous_stage.get(
            "output_hash"
        ):
            reasons.append(f"silent_input_mutation:{stage_name}")
        if index < len(REQUIRED_STAGE_NAMES):
            expected_stage_id = f"{index:02d}-{REQUIRED_STAGE_NAMES[index]}"
            if stage.get("stage_id") != expected_stage_id:
                reasons.append(f"stage_id_mismatch:{stage_name}")
        previous_stage = stage

    by_name = {str(stage.get("stage_name")): stage for stage in stage_list}
    raw_payload = _mapping(_mapping(by_name.get("raw_generation_bytes")).get("output_payload"))
    if raw_payload and raw_payload.get("raw_event_id") != by_name.get(
        "raw_generation_bytes", {}
    ).get("unit_id"):
        reasons.append("unit_id_binding_mismatch")

    checker_response = _mapping(
        _mapping(by_name.get("checker_response")).get("output_payload")
    )
    final_verdict = _mapping(_mapping(by_name.get("final_verdict")).get("output_payload"))
    if checker_response and final_verdict:
        expected_verdict = verdict_from_checker_response(checker_response)
        expected_outcome = checker_response.get("exact_outcome") is True
        if (
            final_verdict.get("observed_verdict") != expected_verdict
            or final_verdict.get("terminal_exact_outcome") is not expected_outcome
        ):
            reasons.append("final_verdict_recompute_mismatch")

    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "stage_count": len(stage_list),
        "stage_names": names,
    }
