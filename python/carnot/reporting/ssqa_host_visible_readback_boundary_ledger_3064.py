"""Build the Exp 3064 SSQA host-visible readback boundary ledger.

Spec refs: REQ-HW-090, SCENARIO-HW-090.

This module records the SSQA readback boundary without touching hardware. It
keeps the current claim narrow while GateMate host-visible smoke is absent and
predeclares the concrete transcript fields needed before later work may claim
readback, acceleration, or sampler behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.286"
SCHEMA = "carnot.ssqa.host_visible_readback_boundary_ledger.v1"
ARTIFACT = "experiment_3064_ssqa_host_visible_readback_boundary_ledger_v1"
OUTPUT_REL_PATH = Path(
    "results/experiment_3064_ssqa_host_visible_readback_boundary_ledger_v1.json"
)

EXP3051_REQUESTED_REL_PATH = Path("results/experiment_3051_ssqa_readback_eligibility_gate_v3.json")
EXP3051_BOUNDED_REL_PATH = Path(
    "results/experiment_3051_ssqa_readback_eligibility_bounded_gate_v3.json"
)
EXP3053_REL_PATH = Path("results/experiment_3053_capstone_v285.json")
EXP3063_REL_PATH = Path("results/experiment_3063_gatemate_no_rerun_operator_action_ledger_v1.json")
EXP3050_REL_PATH = Path("results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json")
HARDWARE_WISHLIST_REL_PATH = Path("research-hardware-wishlist.md")
CHANGELOG_REL_PATH = Path("ops/changelog.md")

EXP3051_SOURCE: JsonDict = {
    "experiment_id": "exp3051",
    "paths": (EXP3051_REQUESTED_REL_PATH, EXP3051_BOUNDED_REL_PATH),
    "role": "ssqa_readback_gate_status",
    "required": False,
    "source_type": "json",
}
SOURCE_SPECS: tuple[JsonDict, ...] = (
    EXP3051_SOURCE,
    {
        "experiment_id": "exp3053",
        "paths": (EXP3053_REL_PATH,),
        "role": "capstone_v285_ssqa_status",
        "required": True,
        "source_type": "json",
    },
    {
        "experiment_id": "exp3063",
        "paths": (EXP3063_REL_PATH,),
        "role": "gatemate_no_rerun_ssqa_gate",
        "required": True,
        "source_type": "json",
    },
    {
        "experiment_id": "exp3050",
        "paths": (EXP3050_REL_PATH,),
        "role": "host_visible_gatemate_smoke_transcript",
        "required": False,
        "source_type": "json",
    },
    {
        "experiment_id": "hardware_wishlist",
        "paths": (HARDWARE_WISHLIST_REL_PATH,),
        "role": "hardware_claim_boundary",
        "required": False,
        "source_type": "text",
    },
    {
        "experiment_id": "changelog",
        "paths": (CHANGELOG_REL_PATH,),
        "role": "milestone_gate_context",
        "required": False,
        "source_type": "text",
    },
)

HOST_VISIBLE_FIELD_SPECS: tuple[tuple[str, str, str], ...] = (
    ("gatemate_host_visible_smoke_passed", "readback", "GateMate host-visible smoke gate passed."),
    ("host_visible_transcript_path", "readback", "Path to the raw host-visible smoke transcript."),
    ("transcript_sha256", "readback", "Checksum for the raw transcript."),
    ("host_reader_command", "readback", "Concrete command that observed the selected output."),
    ("expected_transcript", "readback", "Expected pass/fail transcript for the reader command."),
    ("observed_transcript", "readback", "Observed host transcript lines."),
    ("transcript_matched", "readback", "Observed transcript matched the expected output."),
    ("selected_output_signal", "readback", "Deterministic RTL output signal under observation."),
    ("ccf_binding", "readback", "Authoritative CCF Pin_out binding for the selected signal."),
    ("flash_succeeded", "readback", "Flash completed before the host-visible observation."),
    ("flash_command", "readback", "Exact openFPGALoader flash command used."),
    ("readback_supported", "readback", "Installed tool path exposes a readback or equivalent host IO path."),
    ("readback_attempted", "readback", "Readback or equivalent host-visible IO was attempted."),
    ("readback_hash", "readback", "Hash of the readback or equivalent observed payload."),
    ("sample_count", "acceleration", "Number of observed hardware samples."),
    ("wall_clock_duration_s", "acceleration", "Wall-clock timing for the observed samples."),
    ("per_sample_latency_s", "acceleration", "Per-sample timing derived from the observed transcript."),
    ("sampler_configuration", "sampler_behavior", "Sampler configuration tied to the observed transcript."),
)

INFERENCE_SUBSTRATE = {
    "kind": "ssqa_host_visible_readback_boundary_ledger",
    "source": "checked_in_local_artifacts",
    "model_inference": False,
    "executes_models": False,
    "executes_hardware": False,
    "executes_conductor": False,
    "hardware_readback_attempted": False,
    "flash_attempted": False,
    "rtl_run": False,
    "local_repo_only": True,
    "timing_or_speedup_claim": False,
}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, treating missing or malformed files as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def build_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-HW-090: build the SSQA boundary ledger from local artifacts only."""

    root_path = Path(root)
    source_artifacts = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    source_by_id = {str(row["experiment_id"]): row for row in source_artifacts}
    missing_sources = [
        str(row["path"])
        for row in source_artifacts
        if row.get("required") is True and row.get("readable") is not True
    ]
    smoke_source = _as_mapping(source_by_id.get("exp3050"))
    smoke_payload = _as_mapping(smoke_source.get("payload"))
    required_fields = _required_host_visible_fields(smoke_payload)
    smoke_evidence = _host_visible_smoke_evidence(smoke_source, required_fields)
    ledger_ready = not missing_sources
    readback_allowed = ledger_ready and smoke_evidence["readback_unlocks_ssqa"] is True
    ssqa_status = (
        "clean_host_visible_smoke_transcript_present"
        if readback_allowed
        else "gated_skipped_host_visible_smoke_missing"
    )
    current_claim = (
        "host_visible_smoke_transcript_present_ssqa_readback_may_run_no_performance_claim"
        if readback_allowed
        else "gated_skipped_host_visible_smoke_missing"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "ssqa_boundary_ledger_ready": ledger_ready,
        "ssqa_readback_allowed": readback_allowed,
        "ssqa_status": ssqa_status,
        "required_host_visible_fields": required_fields,
        "current_allowed_claim": current_claim,
        "host_visible_smoke_evidence": smoke_evidence,
        "hardware_execution_claim_made": False,
        "speedup_claim_made": False,
        "hardware_performance_claim_made": False,
        "acceleration_claim_made": False,
        "sampler_behavior_claim_made": False,
        "source_artifacts": _public_sources(source_artifacts),
        "source_summary": _source_summary(source_by_id),
        "missing_source_artifacts": missing_sources,
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "no_new_model_execution": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_hardware_readback_attempt": True,
        "no_new_rtl_or_pnr_run": True,
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
) -> Path:
    """Build and persist the Exp 3064 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _existing_path(root: Path, candidates: tuple[Path, ...]) -> Path:
    for rel_path in candidates:
        if (root / rel_path).is_file():
            return rel_path
    return candidates[0]


def _source_payload(root: Path, spec: Mapping[str, Any]) -> JsonDict:
    rel_path = _existing_path(root, tuple(spec["paths"]))
    path = root / rel_path
    source_type = str(spec.get("source_type") or "json")
    payload = read_json_object(path) if source_type == "json" else {}
    readable = path.is_file() and (source_type != "json" or bool(payload))
    return {
        "experiment_id": str(spec["experiment_id"]),
        "path": rel_path.as_posix(),
        "role": str(spec["role"]),
        "required": spec.get("required") is True,
        "present": path.is_file(),
        "readable": readable,
        "readable_json_object": bool(payload) if source_type == "json" else None,
        "payload": payload,
    }


def _required_host_visible_fields(payload: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "field_id": field_id,
            "claim_scope": claim_scope,
            "required": True,
            "present": _field_present(payload, field_id),
            "description": description,
        }
        for field_id, claim_scope, description in HOST_VISIBLE_FIELD_SPECS
    ]


def _host_visible_smoke_evidence(
    smoke_source: Mapping[str, Any],
    required_fields: list[Mapping[str, Any]],
) -> JsonDict:
    missing = [str(row["field_id"]) for row in required_fields if row.get("present") is not True]
    payload = _as_mapping(smoke_source.get("payload"))
    return {
        "path": str(smoke_source.get("path") or EXP3050_REL_PATH.as_posix()),
        "present": smoke_source.get("present") is True,
        "readable": smoke_source.get("readable") is True,
        "gatemate_host_visible_smoke_passed": payload.get("gatemate_host_visible_smoke_passed")
        is True,
        "transcript_matched": payload.get("transcript_matched") is True,
        "missing_required_fields": missing,
        "readback_unlocks_ssqa": smoke_source.get("readable") is True and not missing,
        "source_honest_verdict": str(payload.get("honest_verdict") or ""),
    }


def _field_present(payload: Mapping[str, Any], field_id: str) -> bool:
    value = payload.get(field_id)
    if isinstance(value, bool):
        return value is True
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return bool(str(value or "").strip())


def _source_summary(source_by_id: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp3051 = _as_mapping(_as_mapping(source_by_id.get("exp3051")).get("payload"))
    exp3053 = _as_mapping(_as_mapping(source_by_id.get("exp3053")).get("payload"))
    exp3063 = _as_mapping(_as_mapping(source_by_id.get("exp3063")).get("payload"))
    exp3050 = _as_mapping(source_by_id.get("exp3050"))
    return {
        "exp3051_status": str(exp3051.get("status") or exp3051.get("ssqa_status") or ""),
        "exp3051_gate_check_summary": str(exp3051.get("gate_check_summary") or ""),
        "capstone_ssqa_status": str(exp3053.get("ssqa_status") or ""),
        "capstone_gatemate_status": str(exp3053.get("gatemate_status") or ""),
        "gatemate_rerun_allowed": exp3063.get("gatemate_rerun_allowed") is True,
        "missing_host_visible_smoke_artifact": exp3050.get("present") is not True,
    }


def _public_sources(source_artifacts: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "experiment_id": str(row["experiment_id"]),
            "path": str(row["path"]),
            "role": str(row["role"]),
            "required": row.get("required") is True,
            "present": row.get("present") is True,
            "readable": row.get("readable") is True,
        }
        for row in source_artifacts
    ]


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("ssqa_boundary_ledger_ready") is not True:
        missing = ", ".join(_as_list(artifact.get("missing_source_artifacts")))
        return f"blocked_precondition: missing SSQA boundary source artifact(s): {missing}"
    allowed = str(artifact.get("ssqa_readback_allowed")).lower()
    return (
        "complete: "
        f"ssqa_boundary_ledger_ready=true; ssqa_readback_allowed={allowed}; "
        f"ssqa_status={artifact.get('ssqa_status')}; current_allowed_claim={artifact.get('current_allowed_claim')}"
    )


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    return [str(item) for item in value if str(item).strip()] if isinstance(value, list) else ([text] if text else [])
