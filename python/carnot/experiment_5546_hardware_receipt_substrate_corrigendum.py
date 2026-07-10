#!/usr/bin/env python3
"""Exp5546: hardware receipt substrate corrigendum.

Spec refs: REQ-VERIFY-5546, SCENARIO-VERIFY-5546.

This module reads prior hardware receipt artifacts and rewrites their evidence
into a narrower no-LLM methodology artifact. The purpose is not to benchmark
hardware. The purpose is to preserve reachability, parser outcomes, blockers,
and the no-speedup boundary without live-model fields that make receipt-only
work look like compute-bound inference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5546_hardware_receipt_substrate_corrigendum.json"
)
SOURCE_RELATIVE_PATHS = (
    Path("results/experiment_5532_hardware_receipt_parser_repeatability.json"),
    Path("results/experiment_5519_hardware_continuity_methodology_receipts.json"),
)

EXPERIMENT = 5546
EXPERIMENT_ID = "exp5546-hardware-receipt-substrate-corrigendum"
MILESTONE = "2026.07.502"
RUN_DATE = "2026-07-10"
SCHEMA = "carnot.experiment_5546.hardware_receipt_substrate_corrigendum.v1"
SPEC_REFS = ("REQ-VERIFY-5546", "SCENARIO-VERIFY-5546")
INFERENCE_SUBSTRATE = "hardware_receipt_methodology_no_llm"
PARSER_VERSION = "hardware_receipt_substrate_corrigendum.v1"
DEVICE_ORDER = ("cpu", "cuda", "polarfire", "kv260", "gatemate")
KV260_SAFE_COMMAND_KINDS = {
    "kv260_ssh_identity",
    "kv260_xmutil_listapps",
    "kv260_remote_uio_list",
}
UNSAFE_KV260_MARKERS = ("mmcblk", "/dev/disk", "host_sdcard", "sdcard", "block")
LIVE_MODEL_MARKERS = (
    '"model_specs":',
    '"target_model":',
    "live_llm_inference",
    "llama_cpp",
    "transformers",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "llm_invoked": (
        "Bare false records that this corrigendum reads receipts only and did not run an LLM."
    ),
    "no_model_specs_required": (
        "Bare true prevents receipt-only hardware metadata from being mistaken for a model invocation."
    ),
    "random_seed": (
        "Deterministic seed derived from receipt inputs and parser version anchors reproducibility without random execution."
    ),
    "reproducibility_checksum": (
        "Content hash over receipt inputs and parser version catches future drift in the corrigendum source evidence."
    ),
    "compute_bound_markers_absent": (
        "True only when live-model, target-model, and model-spec fields are absent from the emitted artifact."
    ),
    "device_receipts": (
        "Flat per-device rows preserve identity, parser outcome, source artifact, and blocker evidence for CPU, CUDA, PolarFire, KV260, and GateMate."
    ),
    "parser_rows_valid": (
        "Bare boolean proving every required device row was parsed into the corrigendum schema."
    ),
    "kv260_safe_path_used": (
        "KV260 evidence must come only from SSH, xmutil, or board-local UIO command kinds."
    ),
    "blockers": (
        "Explicit blockers prevent unreachable devices or missing matched timing from being laundered into speedup evidence."
    ),
    "matched_timing_available": (
        "True only when authenticated matched CPU/device timing exists for the same workload."
    ),
    "hardware_speedup_claim": "Must remain false without matched authenticated timing.",
    "hardware_receipt_corrigendum_clean": (
        "Headline gate combining no-LLM, no-model-spec, parser-valid, safe-KV260, and no-speedup conditions."
    ),
    "tests_added_or_reused": (
        "Names focused tests that assert the corrigendum schema and parser behavior."
    ),
    "field_principles": (
        "One-line annotations explain why each headline and gate field exists."
    ),
    "inference_substrate": (
        "Declares hardware_receipt_methodology_no_llm so receipt parsing is not treated as live inference."
    ),
    "honest_verdict": (
        "Terminal summary states clean corrigendum status, blockers, and no speedup claim."
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(payload: Any) -> str:
    """Serialize JSON deterministically so hashes track content, not formatting."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Hash text with the same SHA-256 shape used by prior receipt artifacts."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    """Hash a JSON-compatible value after canonical serialization."""

    return sha256_text(canonical_json(payload))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while ignoring its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def derive_random_seed(source_input_checksums: Sequence[Mapping[str, Any]], parser_version: str) -> int:
    """Derive a deterministic integer seed from receipt inputs and parser version."""

    digest = sha256_json(
        {
            "parser_version": parser_version,
            "source_input_checksums": list(source_input_checksums),
        }
    )
    return int(digest[:8], 16)


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp5546 artifact from local JSON receipts only."""

    root_path = Path(root)
    sources = load_source_inputs(root_path)
    source_input_checksums = [
        {
            "source_artifact": source["source_artifact"],
            "present": source["present"],
            "sha256": source["sha256"],
        }
        for source in sources
    ]
    device_receipts = [
        normalize_device_receipt(device=device, sources=sources) for device in DEVICE_ORDER
    ]
    blockers = collect_blockers(sources=sources, device_receipts=device_receipts)
    parser_rows_valid = all(row["parser_outcome"] == "parsed" for row in device_receipts)
    kv260_safe_path_used = kv260_receipt_uses_safe_path(device_receipts)
    matched_timing_available = False
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "parser_version": PARSER_VERSION,
        "source_input_checksums": source_input_checksums,
        "llm_invoked": False,
        "no_model_specs_required": True,
        "random_seed": derive_random_seed(source_input_checksums, PARSER_VERSION),
        "reproducibility_checksum": "",
        "compute_bound_markers_absent": True,
        "device_receipts": device_receipts,
        "parser_rows_valid": parser_rows_valid,
        "kv260_safe_path_used": kv260_safe_path_used,
        "blockers": blockers,
        "matched_timing_available": matched_timing_available,
        "hardware_speedup_claim": False,
        "hardware_receipt_corrigendum_clean": False,
        "tests_added_or_reused": normalize_tests(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "",
    }
    artifact["compute_bound_markers_absent"] = compute_bound_markers_absent(artifact)
    artifact["hardware_receipt_corrigendum_clean"] = hardware_receipt_corrigendum_clean(
        artifact
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def load_source_inputs(root: Path) -> list[JsonDict]:
    """Read prior receipt artifacts and preserve missing inputs as evidence."""

    sources: list[JsonDict] = []
    for relative_path in SOURCE_RELATIVE_PATHS:
        path = root / relative_path
        if not path.exists():
            sources.append(
                {
                    "source_artifact": str(relative_path),
                    "present": False,
                    "sha256": None,
                    "payload": None,
                    "error": "source_missing",
                }
            )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            sources.append(
                {
                    "source_artifact": str(relative_path),
                    "present": False,
                    "sha256": None,
                    "payload": None,
                    "error": "source_not_mapping",
                }
            )
            continue
        sources.append(
            {
                "source_artifact": str(relative_path),
                "present": True,
                "sha256": sha256_json(payload),
                "payload": payload,
                "error": None,
            }
        )
    return sources


def normalize_device_receipt(*, device: str, sources: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Convert one prior device receipt into the flat Exp5546 row shape."""

    source, raw = find_source_receipt(device=device, sources=sources)
    if source is None:
        return missing_device_receipt(device)
    if not isinstance(raw, Mapping):
        return malformed_device_receipt(
            device=device,
            source=source,
            raw={},
            blocked_reason="receipt_not_mapping",
        )
    missing_fields = [
        field for field in ("status", "classification", "command_kinds") if field not in raw
    ]
    if missing_fields:
        return malformed_device_receipt(
            device=device,
            source=source,
            raw=raw,
            blocked_reason="missing_" + "_".join(missing_fields),
        )
    return parsed_device_receipt(device=device, source=source, raw=raw)


def find_source_receipt(
    *, device: str, sources: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any] | None, Any]:
    """Find the preferred prior receipt row, falling back only when absent."""

    for source in sources:
        payload = source.get("payload")
        if not isinstance(payload, Mapping):
            continue
        raw = receipt_from_payload(device=device, payload=payload)
        if raw is not None:
            return source, raw
    return None, None


def receipt_from_payload(*, device: str, payload: Mapping[str, Any]) -> Any:
    """Return a device row from Exp5532 or an Exp5519-style individual field."""

    device_receipts = payload.get("device_receipts")
    if isinstance(device_receipts, Mapping) and device in device_receipts:
        return device_receipts[device]
    fallback_keys = {
        "cpu": "cpu_receipt",
        "cuda": "cuda_receipt",
        "polarfire": "polar_fire_receipt",
        "kv260": "kv260_receipt",
        "gatemate": "gatemate_receipt",
    }
    return payload.get(fallback_keys[device])


def missing_device_receipt(device: str) -> JsonDict:
    """Create a blocked row when no prior source contains the required device."""

    return {
        "device": device,
        "source_artifact": None,
        "source_experiment_id": None,
        "parser_outcome": "missing",
        "status": "missing",
        "classification": "parser_blocked",
        "blocked_reason": "receipt_missing",
        "source_parser_version": None,
        "device_identities": [],
        "driver_versions": {},
        "memory": {},
        "metadata": {},
        "safe_command_kinds": [],
    }


def malformed_device_receipt(
    *,
    device: str,
    source: Mapping[str, Any],
    raw: Mapping[str, Any],
    blocked_reason: str,
) -> JsonDict:
    """Create a blocked row when a prior receipt exists but lacks schema fields."""

    return {
        "device": device,
        "source_artifact": source["source_artifact"],
        "source_experiment_id": source_experiment_id(source),
        "parser_outcome": "malformed",
        "status": str(raw.get("status", "malformed")),
        "classification": "parser_blocked",
        "blocked_reason": blocked_reason,
        "source_parser_version": raw.get("parser_version"),
        "device_identities": list(raw.get("device_names", [])),
        "driver_versions": dict(raw.get("driver_versions", {})),
        "memory": dict(raw.get("memory", {})),
        "metadata": dict(raw.get("metadata", {})),
        "safe_command_kinds": list(raw.get("command_kinds", [])),
    }


def parsed_device_receipt(
    *,
    device: str,
    source: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> JsonDict:
    """Create a valid Exp5546 row from a prior parser receipt."""

    status = str(raw["status"])
    blocked_reason = raw.get("blocked_reason")
    return {
        "device": device,
        "source_artifact": source["source_artifact"],
        "source_experiment_id": source_experiment_id(source),
        "parser_outcome": "parsed",
        "status": status,
        "classification": str(raw["classification"]),
        "blocked_reason": str(blocked_reason) if blocked_reason else None,
        "source_parser_version": raw.get("parser_version"),
        "device_identities": list(raw.get("device_names", [])),
        "driver_versions": dict(raw.get("driver_versions", {})),
        "memory": dict(raw.get("memory", {})),
        "metadata": dict(raw.get("metadata", {})),
        "safe_command_kinds": list(raw.get("command_kinds", [])),
    }


def source_experiment_id(source: Mapping[str, Any]) -> str | None:
    """Extract the upstream experiment id from a loaded source payload."""

    payload = source.get("payload")
    if isinstance(payload, Mapping):
        experiment_id = payload.get("experiment_id")
        if experiment_id:
            return str(experiment_id)
    return None


def collect_blockers(
    *, sources: Sequence[Mapping[str, Any]], device_receipts: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Collect source, parser, device, KV260, and timing blockers."""

    blockers: list[JsonDict] = []
    for source in sources:
        if not source.get("present"):
            blockers.append(
                {
                    "kind": str(source.get("error") or "source_missing"),
                    "source_artifact": source["source_artifact"],
                }
            )
    for row in device_receipts:
        if row.get("parser_outcome") != "parsed":
            blockers.append(
                {
                    "kind": "parser_blocker",
                    "device": row["device"],
                    "blocked_reason": row["blocked_reason"],
                    "source_artifact": row["source_artifact"],
                }
            )
        if row.get("blocked_reason") and row.get("parser_outcome") == "parsed":
            blockers.append(
                {
                    "kind": "device_blocker",
                    "device": row["device"],
                    "blocked_reason": row["blocked_reason"],
                    "source_artifact": row["source_artifact"],
                }
            )
    kv260_row = next(row for row in device_receipts if row["device"] == "kv260")
    if not kv260_command_kinds_safe(kv260_row.get("safe_command_kinds", [])):
        blockers.append(
            {
                "kind": "unsafe_kv260_command",
                "device": "kv260",
                "blocked_reason": "kv260_command_kinds_not_ssh_xmutil_or_uio",
                "command_kinds": list(kv260_row.get("safe_command_kinds", [])),
            }
        )
    blockers.append(
        {
            "kind": "matched_timing_missing",
            "blocked_reason": (
                "receipt-only inputs do not provide authenticated matched CPU/device timing"
            ),
        }
    )
    return blockers


def kv260_receipt_uses_safe_path(device_receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when the KV260 row uses SSH, xmutil, or UIO evidence."""

    kv260_rows = [row for row in device_receipts if row.get("device") == "kv260"]
    return bool(kv260_rows) and kv260_command_kinds_safe(
        kv260_rows[0].get("safe_command_kinds", [])
    )


def kv260_command_kinds_safe(command_kinds: Any) -> bool:
    """Check the KV260 command-kind whitelist without using host storage probes."""

    if not isinstance(command_kinds, Sequence) or isinstance(command_kinds, (str, bytes)):
        return False
    kinds = [str(kind) for kind in command_kinds]
    if not kinds:
        return False
    if any(any(marker in kind.lower() for marker in UNSAFE_KV260_MARKERS) for kind in kinds):
        return False
    return all(kind in KV260_SAFE_COMMAND_KINDS for kind in kinds)


def compute_bound_markers_absent(payload: Mapping[str, Any]) -> bool:
    """Detect live-model markers that should never appear in this receipt artifact."""

    if "model_specs" in payload or "target_model" in payload:
        return False
    text = canonical_json(payload)
    return not any(marker in text for marker in LIVE_MODEL_MARKERS)


def hardware_receipt_corrigendum_clean(payload: Mapping[str, Any]) -> bool:
    """Combine the no-LLM, parser-valid, safe-KV260, and no-speedup gates."""

    return bool(
        payload.get("llm_invoked") is False
        and payload.get("no_model_specs_required") is True
        and payload.get("compute_bound_markers_absent") is True
        and payload.get("parser_rows_valid") is True
        and payload.get("kv260_safe_path_used") is True
        and payload.get("hardware_speedup_claim") is False
        and payload.get("inference_substrate") == INFERENCE_SUBSTRATE
    )


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Summarize the methodology gate without converting blockers into speedups."""

    blockers = list(payload.get("blockers", []))
    if payload.get("hardware_receipt_corrigendum_clean") is True:
        return (
            "complete: no-LLM hardware receipt corrigendum clean; "
            f"blockers_recorded={len(blockers)}; matched_timing_available=false; "
            "hardware_speedup_claim=false"
        )
    return (
        "blocked: hardware receipt corrigendum is not clean; "
        f"blockers_recorded={len(blockers)}; matched_timing_available=false; "
        "hardware_speedup_claim=false"
    )


def normalize_tests(tests_added_or_reused: Sequence[str] | None) -> list[str]:
    """Record focused tests backing this parser."""

    if tests_added_or_reused:
        return [str(test) for test in tests_added_or_reused]
    return ["tests/python/test_experiment_5546_hardware_receipt_substrate_corrigendum.py"]


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5546 gates and checksum consistency."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")  # pragma: no cover
    if payload.get("llm_invoked") is not False:
        raise ValueError("llm_invoked must be false")
    if payload.get("no_model_specs_required") is not True:
        raise ValueError("no_model_specs_required must be true")  # pragma: no cover
    if "model_specs" in payload:
        raise ValueError("model_specs must be absent")
    if "target_model" in payload:
        raise ValueError("target_model must be absent")  # pragma: no cover
    if payload.get("hardware_speedup_claim") is not False:
        raise ValueError("hardware_speedup_claim must be false")
    if payload.get("matched_timing_available") is not False:
        raise ValueError("matched_timing_available must be false")  # pragma: no cover
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")  # pragma: no cover
    device_receipts = payload.get("device_receipts")
    if not isinstance(device_receipts, list):
        raise ValueError("device_receipts must be a list")  # pragma: no cover
    if [row.get("device") for row in device_receipts] != list(DEVICE_ORDER):
        raise ValueError("device_receipts must cover required devices in order")  # pragma: no cover
    parser_rows_valid = all(row.get("parser_outcome") == "parsed" for row in device_receipts)
    if payload.get("parser_rows_valid") is not parser_rows_valid:
        raise ValueError("parser_rows_valid mismatch")  # pragma: no cover
    if payload.get("kv260_safe_path_used") is not kv260_receipt_uses_safe_path(device_receipts):
        raise ValueError("kv260_safe_path_used mismatch")  # pragma: no cover
    if payload.get("compute_bound_markers_absent") is not compute_bound_markers_absent(payload):
        raise ValueError("compute_bound_markers_absent mismatch")
    clean = hardware_receipt_corrigendum_clean(payload)
    if payload.get("hardware_receipt_corrigendum_clean") is not clean:
        raise ValueError("hardware_receipt_corrigendum_clean mismatch")  # pragma: no cover
    source_checksums = payload.get("source_input_checksums")
    if not isinstance(source_checksums, list):
        raise ValueError("source_input_checksums must be a list")  # pragma: no cover
    expected_seed = derive_random_seed(source_checksums, PARSER_VERSION)
    if payload.get("random_seed") != expected_seed:
        raise ValueError("random_seed mismatch")  # pragma: no cover
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


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
    """Build, validate, and write the Exp5546 artifact."""

    artifact = build_artifact(root=repo_root, tests_added_or_reused=tests_added_or_reused)
    return write_output(repo_root, artifact)


def main() -> int:  # pragma: no cover
    """CLI entry point used by operators when emitting the artifact manually."""

    run_experiment()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
