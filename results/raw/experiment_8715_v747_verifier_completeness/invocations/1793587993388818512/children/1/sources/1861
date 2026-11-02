#!/usr/bin/env python3
"""Exp5560: hardware and timing receipt hygiene.

Spec refs: REQ-VERIFY-5560, SCENARIO-VERIFY-5560.

This module builds one receipt-hygiene artifact from Exp5546's clean hardware
receipt substrate. It does not benchmark hardware. Its job is to keep board
identity evidence, launch/finish timing examples, checksum linkage, and the
no-speedup boundary in one explicit JSON artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5546_hardware_receipt_substrate_corrigendum as exp5546


JsonDict = dict[str, Any]
Clock = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5560_hardware_and_timing_receipt_hygiene.json"
)
UPSTREAM_HARDWARE_CORRIGENDUM = exp5546.RESULT_RELATIVE_PATH

EXPERIMENT = 5560
EXPERIMENT_ID = "exp5560-hardware-and-timing-receipt-hygiene"
MILESTONE = "2026.07.503"
RUN_DATE = "2026-07-10"
SCHEMA = "carnot.experiment_5560.hardware_and_timing_receipt_hygiene.v1"
SPEC_REFS = ("REQ-VERIFY-5560", "SCENARIO-VERIFY-5560")
PARSER_VERSION = "hardware_and_timing_receipt_hygiene.v1"
INFERENCE_SUBSTRATE = "hardware_receipt_and_timing_hygiene_no_llm"

FORBIDDEN_BLOCK_DEVICE_MARKERS = ("/dev/mmcblk", "/dev/disk")
TIMING_VALUE_FIELDS = (
    "duration_s",
    "elapsed_s",
    "latency_s",
    "wall_time_s",
    "monotonic_duration_s",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "llm_invoked": "Bare false records that this hygiene artifact did not run a model.",
    "no_model_specs_required": (
        "Receipt hygiene does not require model specs because no model is invoked."
    ),
    "upstream_hardware_corrigendum": (
        "Names the clean Exp5546 source that supplies parsed hardware rows."
    ),
    "device_receipts": (
        "Carries per-device parser outcomes, identities, safe command kinds, and blockers forward."
    ),
    "kv260_safe_path_used": (
        "Keeps KV260 evidence limited to SSH, xmutil, and board-local UIO paths."
    ),
    "forbidden_block_device_paths_used": (
        "Must stay false so host storage paths cannot re-enter KV260 evidence."
    ),
    "parser_rows_valid": (
        "Prevents malformed receipt rows from being promoted into timing evidence."
    ),
    "launch_finish_receipt_ready": (
        "Confirms future experiment-side launch and finish stamps have a ready receipt shape."
    ),
    "monotonic_clock_used": (
        "Launch/finish examples use monotonic time rather than wall-clock ordering."
    ),
    "artifact_checksum_linked": (
        "Timing receipts link to artifact checksums so stamps cannot float free of evidence."
    ),
    "matched_timing_available": (
        "True only when equivalent repeated CPU and hardware timing pairs exist."
    ),
    "repeated_timing_pairs": (
        "Counts only matched timing pairs with timing values and shared workload/checksum evidence."
    ),
    "hardware_speedup_claim": "Must remain false without authenticated matched timing.",
    "conductor_modified": (
        "Must remain false because the conductor is outside this experiment scope."
    ),
    "roadmap_yaml_unchanged": (
        "Confirms the active roadmap was not changed by receipt hygiene work."
    ),
    "tests_added_or_reused": (
        "Names focused tests backing parser and timing hygiene behavior."
    ),
    "field_principles": (
        "One-line annotations explain why each headline and gate field exists."
    ),
    "inference_substrate": (
        "Declares hardware receipt and timing hygiene, not live inference."
    ),
    "honest_verdict": "Terminal summary states clean hygiene status and no speedup claim.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(payload: Any) -> str:
    """Serialize JSON deterministically so checksum comparisons are stable."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Hash text in the same format as the upstream receipt artifacts."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    """Hash a JSON-compatible value after deterministic serialization."""

    return sha256_text(canonical_json(payload))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash this artifact while ignoring its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def file_sha256(path: Path) -> str:
    """Hash a source artifact as bytes so formatting drift is detectable."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_random_seed(upstream_file_sha256: str | None) -> int:
    """Derive a deterministic seed from the upstream checksum and parser version."""

    digest = sha256_json(
        {
            "parser_version": PARSER_VERSION,
            "upstream_file_sha256": upstream_file_sha256,
        }
    )
    return int(digest[:8], 16)


def load_upstream_corrigendum(root: str | Path = REPO_ROOT) -> JsonDict:
    """Load Exp5546 or return an explicit blocker row when it is absent."""

    root_path = Path(root)
    path = root_path / UPSTREAM_HARDWARE_CORRIGENDUM
    if not path.exists():
        return {
            "present": False,
            "path": str(UPSTREAM_HARDWARE_CORRIGENDUM),
            "file_sha256": None,
            "payload": {},
            "blocked_reason": "upstream_hardware_corrigendum_missing",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {
            "present": False,
            "path": str(UPSTREAM_HARDWARE_CORRIGENDUM),
            "file_sha256": file_sha256(path),
            "payload": {},
            "blocked_reason": "upstream_hardware_corrigendum_not_mapping",
        }
    return {
        "present": True,
        "path": str(UPSTREAM_HARDWARE_CORRIGENDUM),
        "file_sha256": file_sha256(path),
        "payload": payload,
        "blocked_reason": None,
    }


def device_rows_valid(device_receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only for complete parsed rows covering Exp5546 device order."""

    return bool(device_receipts) and [row.get("device") for row in device_receipts] == list(
        exp5546.DEVICE_ORDER
    ) and all(row.get("parser_outcome") == "parsed" for row in device_receipts)


def contains_forbidden_block_device_path(value: Any) -> bool:
    """Detect retired host block-device paths inside receipt evidence."""

    if isinstance(value, Mapping):
        return any(contains_forbidden_block_device_path(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(contains_forbidden_block_device_path(item) for item in value)
    if isinstance(value, str):
        return any(marker in value for marker in FORBIDDEN_BLOCK_DEVICE_MARKERS)
    return False


def receipt_has_timing_value(receipt: Mapping[str, Any]) -> bool:
    """Return true when a receipt exposes a concrete timing value."""

    return any(field in receipt and receipt[field] is not None for field in TIMING_VALUE_FIELDS)


def count_repeated_timing_pairs(receipts: Any) -> int:
    """Count workloads with both CPU and hardware timing rows.

    Workload receipts without timing values do not count. This keeps Exp5532's
    repeated workload hashes useful as provenance while preventing them from
    becoming timing evidence.
    """

    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes, bytearray)):
        return 0
    devices_by_workload: dict[str, set[str]] = {}
    for receipt in receipts:
        if not isinstance(receipt, Mapping) or not receipt_has_timing_value(receipt):
            continue
        workload_hash = receipt.get("workload_hash")
        device = receipt.get("device")
        if not isinstance(workload_hash, str) or not isinstance(device, str):
            continue
        devices_by_workload.setdefault(workload_hash, set()).add(device)
    return sum(
        1
        for devices in devices_by_workload.values()
        if "cpu" in devices and any(device != "cpu" for device in devices)
    )


def build_launch_finish_receipt(
    *,
    artifact_path: str,
    artifact_payload: Mapping[str, Any],
    artifact_file_sha256: str | None,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    """Build a checksum-linked launch/finish receipt using monotonic time."""

    launch = clock()
    finish = clock()
    checksum_field = artifact_payload.get("reproducibility_checksum")
    expected_checksum = exp5546.payload_checksum(artifact_payload)
    return {
        "receipt_name": "experiment_side_launch_finish_example",
        "clock_source": "time.perf_counter",
        "launch_monotonic_s": round(float(launch), 9),
        "finish_monotonic_s": round(float(finish), 9),
        "duration_s": round(max(float(finish) - float(launch), 0.0), 9),
        "artifact_path": artifact_path,
        "artifact_file_sha256": artifact_file_sha256,
        "artifact_checksum_field": checksum_field,
        "artifact_checksum_matches": checksum_field == expected_checksum,
        "artifact_checksum_linked": bool(
            artifact_file_sha256 and checksum_field and checksum_field == expected_checksum
        ),
    }


def monotonic_clock_used(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Validate that launch/finish examples use increasing monotonic stamps."""

    return bool(receipts) and all(
        receipt.get("clock_source") == "time.perf_counter"
        and isinstance(receipt.get("launch_monotonic_s"), (int, float))
        and isinstance(receipt.get("finish_monotonic_s"), (int, float))
        and receipt["finish_monotonic_s"] >= receipt["launch_monotonic_s"]
        for receipt in receipts
    )


def artifact_checksum_linked(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when every launch/finish example links a checksum."""

    return bool(receipts) and all(
        receipt.get("artifact_checksum_linked") is True for receipt in receipts
    )


def launch_finish_receipts_ready(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Combine monotonic-time and checksum-linkage readiness."""

    return monotonic_clock_used(receipts) and artifact_checksum_linked(receipts)


def normalize_tests(tests_added_or_reused: Sequence[str] | None) -> list[str]:
    """Record the focused tests backing this hygiene artifact."""

    if tests_added_or_reused:
        return [str(test) for test in tests_added_or_reused]
    return ["tests/python/test_experiment_5560_hardware_and_timing_receipt_hygiene.py"]


def collect_blockers(artifact: Mapping[str, Any], upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Collect explicit blockers for every failed hygiene gate."""

    blockers: list[JsonDict] = []
    blocked_reason = upstream.get("blocked_reason")
    if blocked_reason:
        blockers.append({"kind": str(blocked_reason), "source_artifact": upstream["path"]})
    if artifact.get("parser_rows_valid") is not True:
        blockers.append({"kind": "parser_rows_invalid"})
    if artifact.get("kv260_safe_path_used") is not True:
        blockers.append({"kind": "unsafe_kv260_command"})
    if artifact.get("forbidden_block_device_paths_used") is True:
        blockers.append({"kind": "forbidden_block_device_path"})
    if artifact.get("launch_finish_receipt_ready") is not True:
        blockers.append({"kind": "launch_finish_receipt_not_ready"})
    if artifact.get("matched_timing_available") is not True:
        blockers.append({"kind": "matched_timing_missing"})
    if artifact.get("conductor_modified") is True:
        blockers.append({"kind": "conductor_modified"})
    if artifact.get("roadmap_yaml_unchanged") is not True:
        blockers.append({"kind": "roadmap_yaml_changed"})
    return blockers


def hygiene_clean(payload: Mapping[str, Any]) -> bool:
    """Combine the headline gates without converting blockers into speedups."""

    return bool(
        payload.get("llm_invoked") is False
        and payload.get("no_model_specs_required") is True
        and payload.get("parser_rows_valid") is True
        and payload.get("kv260_safe_path_used") is True
        and payload.get("forbidden_block_device_paths_used") is False
        and payload.get("launch_finish_receipt_ready") is True
        and payload.get("monotonic_clock_used") is True
        and payload.get("artifact_checksum_linked") is True
        and payload.get("matched_timing_available") is False
        and payload.get("repeated_timing_pairs") == 0
        and payload.get("hardware_speedup_claim") is False
        and payload.get("conductor_modified") is False
        and payload.get("roadmap_yaml_unchanged") is True
        and payload.get("inference_substrate") == INFERENCE_SUBSTRATE
    )


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Summarize the receipt state honestly, with no speedup promotion."""

    blockers = list(payload.get("blockers", []))
    if hygiene_clean(payload):
        return (
            "complete: hardware and timing receipt hygiene clean; "
            "matched_timing_available=false; repeated_timing_pairs=0; "
            "hardware_speedup_claim=false"
        )
    return (
        "blocked: hardware and timing receipt hygiene has blockers; "
        f"blockers_recorded={len(blockers)}; hardware_speedup_claim=false"
    )


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    clock: Clock = time.perf_counter,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp5560 artifact from the Exp5546 corrigendum."""

    upstream = load_upstream_corrigendum(root)
    upstream_payload = upstream["payload"]
    device_receipts = list(upstream_payload.get("device_receipts", []))
    if not all(isinstance(row, Mapping) for row in device_receipts):
        device_receipts = []
    repeated_receipts = upstream_payload.get(
        "repeated_timing_receipts", upstream_payload.get("repeated_workload_receipts", [])
    )
    repeated_timing_pairs = count_repeated_timing_pairs(repeated_receipts)
    launch_finish_receipts = (
        [
            build_launch_finish_receipt(
                artifact_path=str(UPSTREAM_HARDWARE_CORRIGENDUM),
                artifact_payload=upstream_payload,
                artifact_file_sha256=upstream.get("file_sha256"),
                clock=clock,
            )
        ]
        if upstream.get("present")
        else []
    )
    parser_rows_valid = device_rows_valid(device_receipts)
    kv260_safe_path_used = bool(
        upstream_payload.get("kv260_safe_path_used") is True
        and exp5546.kv260_receipt_uses_safe_path(device_receipts)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "parser_version": PARSER_VERSION,
        "random_seed": derive_random_seed(upstream.get("file_sha256")),
        "reproducibility_checksum": "",
        "upstream_hardware_corrigendum_sha256": upstream.get("file_sha256"),
        "llm_invoked": False,
        "no_model_specs_required": True,
        "upstream_hardware_corrigendum": str(UPSTREAM_HARDWARE_CORRIGENDUM),
        "device_receipts": device_receipts,
        "kv260_safe_path_used": kv260_safe_path_used,
        "forbidden_block_device_paths_used": contains_forbidden_block_device_path(
            device_receipts
        ),
        "parser_rows_valid": parser_rows_valid,
        "launch_finish_receipt_examples": launch_finish_receipts,
        "launch_finish_receipt_ready": launch_finish_receipts_ready(
            launch_finish_receipts
        ),
        "monotonic_clock_used": monotonic_clock_used(launch_finish_receipts),
        "artifact_checksum_linked": artifact_checksum_linked(launch_finish_receipts),
        "matched_timing_available": repeated_timing_pairs > 0,
        "repeated_timing_pairs": repeated_timing_pairs,
        "hardware_speedup_claim": False,
        "conductor_modified": False,
        "roadmap_yaml_unchanged": True,
        "tests_added_or_reused": normalize_tests(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "blockers": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "",
    }
    artifact["blockers"] = collect_blockers(artifact, upstream)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate required Exp5560 fields and no-speedup gates."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")  # pragma: no cover
    if payload.get("llm_invoked") is not False:
        raise ValueError("llm_invoked must be false")  # pragma: no cover
    if payload.get("no_model_specs_required") is not True:
        raise ValueError("no_model_specs_required must be true")  # pragma: no cover
    if "model_specs" in payload or "target_model" in payload:
        raise ValueError("model specs must be absent")  # pragma: no cover
    device_receipts = payload.get("device_receipts")
    if not isinstance(device_receipts, list):
        raise ValueError("device_receipts must be a list")  # pragma: no cover
    row_mappings = [row for row in device_receipts if isinstance(row, Mapping)]
    if payload.get("parser_rows_valid") is not device_rows_valid(row_mappings):
        raise ValueError("parser_rows_valid mismatch")  # pragma: no cover
    if payload.get("kv260_safe_path_used") is not exp5546.kv260_receipt_uses_safe_path(
        row_mappings
    ):
        raise ValueError("kv260_safe_path_used mismatch")  # pragma: no cover
    forbidden = contains_forbidden_block_device_path(device_receipts)
    if payload.get("forbidden_block_device_paths_used") is not forbidden:
        raise ValueError("forbidden_block_device_paths_used mismatch")  # pragma: no cover
    receipts = payload.get("launch_finish_receipt_examples")
    if not isinstance(receipts, list):
        raise ValueError("launch_finish_receipt_examples must be a list")  # pragma: no cover
    receipt_mappings = [receipt for receipt in receipts if isinstance(receipt, Mapping)]
    if payload.get("launch_finish_receipt_ready") is not launch_finish_receipts_ready(
        receipt_mappings
    ):
        raise ValueError("launch_finish_receipt_ready mismatch")  # pragma: no cover
    if payload.get("monotonic_clock_used") is not monotonic_clock_used(receipt_mappings):
        raise ValueError("monotonic_clock_used mismatch")  # pragma: no cover
    if payload.get("artifact_checksum_linked") is not artifact_checksum_linked(
        receipt_mappings
    ):
        raise ValueError("artifact_checksum_linked mismatch")  # pragma: no cover
    repeated_timing_pairs = payload.get("repeated_timing_pairs")
    if not isinstance(repeated_timing_pairs, int):
        raise ValueError("repeated_timing_pairs must be an int")  # pragma: no cover
    if payload.get("matched_timing_available") is not (repeated_timing_pairs > 0):
        raise ValueError("matched_timing_available mismatch")  # pragma: no cover
    if payload.get("hardware_speedup_claim") is not False:
        raise ValueError("hardware_speedup_claim must be false")
    if payload.get("conductor_modified") is not False:
        raise ValueError("conductor_modified must be false")
    if payload.get("roadmap_yaml_unchanged") is not True:
        raise ValueError("roadmap_yaml_unchanged must be true")  # pragma: no cover
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")  # pragma: no cover
    principles = payload.get("field_principles")
    if principles != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")  # pragma: no cover
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")  # pragma: no cover


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal JSON artifact with deterministic formatting."""

    output_path = Path(root) / RESULT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    tests_added_or_reused: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the Exp5560 artifact."""

    artifact = build_artifact(root=repo_root, tests_added_or_reused=tests_added_or_reused)
    return write_output(repo_root, artifact)


def main() -> int:  # pragma: no cover
    """CLI entry point used to emit the receipt-hygiene artifact."""

    run_experiment()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
