"""Build the Exp 3119 GateMate/SSQA operator evidence ingestion v4 ledger.

Spec refs: REQ-HW-094, SCENARIO-HW-094.

This module is intentionally only an evidence-ingestion ledger. It reads the
Exp 3106 v3 artifact, rechecks the local files that v3 already documented, and
fails closed unless operator-owned evidence is complete. It does not flash a
board, synthesize RTL, perform readback, run a model, or turn evidence review
into a speedup claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.290"
SCHEMA = "carnot.gatemate_ssqa_operator_evidence_ingestion.v4"
ARTIFACT = "experiment_3119_gatemate_ssqa_operator_evidence_ingestion_v4"
OUTPUT_REL_PATH = Path("results/experiment_3119_gatemate_ssqa_operator_evidence_ingestion_v4.json")

EXP3106_REL_PATH = Path("results/experiment_3106_gatemate_ssqa_operator_evidence_ingestion_v3.json")
EXP3048_REL_PATH = Path("results/experiment_3048_gatemate_output_contract_operator_package_v1.json")
EXP3050_REL_PATH = Path("results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json")
GATEMATE_CCF_REL_PATH = Path("hardware/gatemate/ising_n16_gatemate.ccf")

HOST_VISIBLE_REQUIRED_FIELDS: tuple[tuple[str, str], ...] = (
    ("gatemate_host_visible_smoke_passed", "readback"),
    ("host_visible_transcript_path", "readback"),
    ("transcript_sha256", "readback"),
    ("host_reader_command", "readback"),
    ("expected_transcript", "readback"),
    ("observed_transcript", "readback"),
    ("transcript_matched", "readback"),
    ("selected_output_signal", "readback"),
    ("ccf_binding", "readback"),
    ("flash_succeeded", "readback"),
    ("flash_command", "readback"),
    ("readback_supported", "readback"),
    ("readback_attempted", "readback"),
    ("readback_hash", "readback"),
    ("sample_count", "acceleration"),
    ("wall_clock_duration_s", "acceleration"),
    ("per_sample_latency_s", "acceleration"),
    ("sampler_configuration", "sampler_behavior"),
)

DEFAULT_OPERATOR_ACTIONS: dict[str, str] = {
    "authoritative_pinout_ccf_binding": (
        "Provide an authoritative GateMate A1-EVB-2M output pinout and commit "
        "a CCF Pin_out binding for done."
    ),
    "host_reader_command": "Commit a concrete host reader command for done.",
    "expected_transcript": "Record the expected pass/fail transcript for the done host reader command.",
    "safety_limits": "Open the downstream flash safety gate only after the output contract is authoritative.",
    "host_visible_smoke_evidence": (
        "Commit a passing GateMate host-visible smoke transcript with checksum, "
        "expected and observed output, CCF binding, flash/readback fields, "
        "timing fields, sampler configuration, and a local transcript file."
    ),
}

INFERENCE_SUBSTRATE = {
    "kind": "operator_evidence_ingestion_v4",
    "source": "documented_local_artifact_paths",
    "local_repo_only": True,
    "no_model_inference": True,
    "no_live_model_inference": True,
    "model_inference": False,
    "executes_models": False,
    "executes_hardware": False,
    "executes_conductor": False,
    "flash_attempted": False,
    "synthesis_or_pnr_run": False,
    "hardware_readback_attempted": False,
    "timing_or_speedup_claim": False,
}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON evidence file, returning empty when evidence is unusable."""

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def build_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-HW-094: build the v4 ledger from documented v3 evidence paths only."""

    root_path = Path(root)
    v3 = read_json_object(root_path / EXP3106_REL_PATH)
    exp3048 = read_json_object(root_path / EXP3048_REL_PATH)
    exp3050 = read_json_object(root_path / EXP3050_REL_PATH)
    ccf_text = _read_text(root_path / GATEMATE_CCF_REL_PATH)

    source_catalog = _documented_locations(v3)
    transcript_path = _coerce_rel_path(exp3050.get("host_visible_transcript_path"))
    if _has_path(transcript_path):
        _add_source(
            source_catalog,
            transcript_path,
            "host_visible_transcript_file",
            False,
        )
    source_rows = [_source_status(root_path, path, role, required) for path, role, required in source_catalog.values()]
    missing_sources = [
        str(row["path"])
        for row in source_rows
        if row.get("required") is True and row.get("readable") is not True
    ]
    v4_ready = bool(v3) and not missing_sources
    actions = _operator_action_map(v3)
    gatemate_checks = _gatemate_evidence_checks(exp3048, ccf_text, actions)
    host_evidence = _host_visible_evidence(root_path, exp3050)
    gatemate_allowed = v4_ready and all(row["present"] for row in gatemate_checks)
    ssqa_allowed = v4_ready and gatemate_allowed and host_evidence["ready"] is True

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "operator_evidence_ingestion_v4_ready": v4_ready,
        "gatemate_rerun_allowed": gatemate_allowed,
        "ssqa_readback_allowed": ssqa_allowed,
        "missing_operator_actions": _missing_operator_actions(
            gatemate_checks,
            host_evidence,
            missing_sources,
        ),
        "evidence_files_seen": [row for row in source_rows if row["present"] and row["readable"]],
        "documented_evidence_locations": source_rows,
        "hardware_commands_run": [],
        "hardware_execution_claim_made": False,
        "hardware_execution_performed": False,
        "hardware_readback_attempted": False,
        "speedup_claim_made": False,
        "timing_claim_made": False,
        "source_artifacts": source_rows,
        "source_artifact_count": len(source_rows),
        "missing_source_artifacts": missing_sources,
        "gatemate_evidence_checks": gatemate_checks,
        "host_visible_smoke_evidence": host_evidence,
        "allowed_next_experiment_scope": _allowed_next_scope(
            v4_ready,
            gatemate_allowed,
            ssqa_allowed,
        ),
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
) -> Path:
    """Persist the Exp 3119 JSON deliverable."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _documented_locations(v3: Mapping[str, Any]) -> dict[str, tuple[Path, str, bool]]:
    catalog: dict[str, tuple[Path, str, bool]] = {}
    _add_source(catalog, EXP3106_REL_PATH, "prior_v3_operator_evidence_ingestion", True)
    for row in _as_list(v3.get("source_artifacts")):
        mapped = _as_mapping(row)
        _add_source(
            catalog,
            _coerce_rel_path(mapped.get("path")),
            str(mapped.get("role") or "v3_source_artifact"),
            mapped.get("required") is True,
        )
    for row in _as_list(v3.get("checked_paths")):
        mapped = _as_mapping(row)
        _add_source(
            catalog,
            _coerce_rel_path(mapped.get("path")),
            str(mapped.get("evidence_class") or "v3_checked_path"),
            False,
        )
    for row in _as_list(v3.get("missing_operator_actions")):
        mapped = _as_mapping(row)
        _add_source(
            catalog,
            _coerce_rel_path(mapped.get("source_artifact")),
            "v3_missing_action_source",
            False,
        )
        for checked_path in _as_text_list(mapped.get("checked_paths")):
            _add_source(
                catalog,
                _coerce_rel_path(checked_path),
                "v3_missing_action_checked_path",
                False,
            )
    return dict(sorted(catalog.items()))


def _add_source(
    catalog: dict[str, tuple[Path, str, bool]],
    path: Path,
    role: str,
    required: bool,
) -> None:
    if not path.as_posix() or path.as_posix() == ".":
        return
    key = path.as_posix()
    existing = catalog.get(key)
    if existing is None or (required and existing[2] is not True):
        catalog[key] = (path, role, required)


def _source_status(root: Path, rel_path: Path, role: str, required: bool) -> JsonDict:
    path = _resolve_path(root, rel_path)
    present = path.is_file()
    source_type = "json" if path.suffix.lower() == ".json" else "text"
    readable = present and (source_type != "json" or bool(read_json_object(path)))
    row: JsonDict = {
        "path": rel_path.as_posix(),
        "role": role,
        "required": required,
        "present": present,
        "readable": readable,
        "source_type": source_type,
        "sha256": _sha256_file(path) if present else None,
    }
    return row


def _gatemate_evidence_checks(
    exp3048: Mapping[str, Any],
    ccf_text: str,
    actions: Mapping[str, str],
) -> list[JsonDict]:
    selected_signal = str(exp3048.get("selected_output_signal") or "done")
    output_ready = exp3048.get("gatemate_output_contract_ready") is True
    plan_ready = exp3048.get("host_visible_io_plan_ready") is True
    ccf_binding = _as_mapping(exp3048.get("ccf_binding")) or _ccf_binding_from_text(
        ccf_text,
        selected_signal,
    )
    host_reader_command = str(exp3048.get("host_reader_command") or "")
    expected_transcript = _as_text_list(exp3048.get("expected_transcript"))
    safety_limits = _as_mapping(exp3048.get("safety_limits"))
    safety_open = safety_limits.get("downstream_flash_gate_open") is True
    safety_attempts = int(safety_limits.get("max_flash_attempts_without_operator_review") or 0)
    return [
        _gate_check(
            "authoritative_pinout_ccf_binding",
            actions,
            "ccf_binding",
            ccf_binding,
            [
                ("gatemate_output_contract_ready", output_ready),
                ("host_visible_io_plan_ready", plan_ready),
                (f"ccf_binding_for_{selected_signal}", bool(ccf_binding)),
            ],
            [EXP3048_REL_PATH, GATEMATE_CCF_REL_PATH],
        ),
        _gate_check(
            "host_reader_command",
            actions,
            "host_reader_command",
            host_reader_command,
            [
                ("gatemate_output_contract_ready", output_ready),
                ("host_visible_io_plan_ready", plan_ready),
                ("host_reader_command", _concrete(host_reader_command)),
            ],
            [EXP3048_REL_PATH],
        ),
        _gate_check(
            "expected_transcript",
            actions,
            "expected_transcript",
            expected_transcript,
            [
                ("gatemate_output_contract_ready", output_ready),
                ("host_visible_io_plan_ready", plan_ready),
                ("expected_transcript", _concrete(expected_transcript)),
            ],
            [EXP3048_REL_PATH],
        ),
        _gate_check(
            "safety_limits",
            actions,
            "safety_limits",
            safety_limits,
            [
                ("safety_limits.downstream_flash_gate_open", safety_open),
                ("safety_limits.max_flash_attempts_without_operator_review", safety_attempts > 0),
            ],
            [EXP3048_REL_PATH],
        ),
    ]


def _gate_check(
    evidence_id: str,
    actions: Mapping[str, str],
    source_field: str,
    evidence: Any,
    checks: list[tuple[str, bool]],
    checked_paths: list[Path],
) -> JsonDict:
    missing = [field for field, present in checks if present is not True]
    return {
        "evidence_id": evidence_id,
        "present": not missing,
        "operator_action": str(actions.get(evidence_id) or DEFAULT_OPERATOR_ACTIONS[evidence_id]),
        "path": EXP3048_REL_PATH.as_posix(),
        "source_field": source_field,
        "checked_paths": [path.as_posix() for path in checked_paths],
        "missing_required_fields": missing,
        "evidence": evidence if not missing else None,
    }


def _host_visible_evidence(root: Path, exp3050: Mapping[str, Any]) -> JsonDict:
    required_fields = [
        {
            "field_id": field_id,
            "claim_scope": claim_scope,
            "present": _field_present(exp3050, field_id),
            "path": EXP3050_REL_PATH.as_posix(),
        }
        for field_id, claim_scope in HOST_VISIBLE_REQUIRED_FIELDS
    ]
    missing = [str(row["field_id"]) for row in required_fields if row["present"] is not True]
    transcript_rel_path = _coerce_rel_path(exp3050.get("host_visible_transcript_path"))
    transcript_path = _resolve_path(root, transcript_rel_path)
    transcript_file_present = _has_path(transcript_rel_path) and transcript_path.is_file()
    expected_sha = str(exp3050.get("transcript_sha256") or "")
    observed_sha = _sha256_file(transcript_path) if transcript_file_present else ""
    sha_matched = bool(expected_sha and observed_sha and expected_sha == observed_sha)
    if _has_path(transcript_rel_path):
        if not transcript_file_present:
            missing.append("host_visible_transcript_file")
        elif not sha_matched:
            missing.append("transcript_sha256_match")

    smoke_present = bool(exp3050)
    smoke_passed = exp3050.get("gatemate_host_visible_smoke_passed") is True
    transcript_matched = exp3050.get("transcript_matched") is True
    checked_paths = [EXP3050_REL_PATH.as_posix()]
    if _has_path(transcript_rel_path):
        checked_paths.append(transcript_rel_path.as_posix())
    return {
        "path": EXP3050_REL_PATH.as_posix(),
        "present": smoke_present,
        "gatemate_host_visible_smoke_passed": smoke_passed,
        "transcript_matched": transcript_matched,
        "host_visible_transcript_path": transcript_rel_path.as_posix(),
        "transcript_file_present": transcript_file_present,
        "transcript_file_sha256": observed_sha or None,
        "transcript_file_sha256_matched": sha_matched,
        "missing_required_fields": missing,
        "required_fields": required_fields,
        "checked_paths": checked_paths,
        "ready": smoke_present and smoke_passed and transcript_matched and not missing,
    }


def _missing_operator_actions(
    gatemate_checks: list[Mapping[str, Any]],
    host_evidence: Mapping[str, Any],
    missing_sources: list[str],
) -> list[JsonDict]:
    missing = [
        {
            "missing_item": str(row["evidence_id"]),
            "present": False,
            "operator_action": str(row["operator_action"]),
            "source_artifact": str(row["path"]),
            "checked_paths": _as_text_list(row.get("checked_paths")),
            "missing_required_fields": _as_text_list(row.get("missing_required_fields")),
        }
        for row in gatemate_checks
        if row.get("present") is not True
    ]
    if host_evidence.get("ready") is not True:
        missing.append(
            {
                "missing_item": "host_visible_smoke_evidence",
                "present": host_evidence.get("present") is True,
                "operator_action": DEFAULT_OPERATOR_ACTIONS["host_visible_smoke_evidence"],
                "source_artifact": EXP3050_REL_PATH.as_posix(),
                "checked_paths": _as_text_list(host_evidence.get("checked_paths")),
                "missing_required_fields": _as_text_list(
                    host_evidence.get("missing_required_fields")
                ),
            }
        )
    for source in missing_sources:
        missing.append(
            {
                "missing_item": f"source_artifact:{source}",
                "present": False,
                "operator_action": "Commit the required Exp 3106 v3 operator-evidence artifact.",
                "source_artifact": source,
                "checked_paths": [source],
                "missing_required_fields": [source],
            }
        )
    return missing


def _operator_action_map(v3: Mapping[str, Any]) -> dict[str, str]:
    actions = dict(DEFAULT_OPERATOR_ACTIONS)
    for row in _as_list(v3.get("missing_operator_actions")):
        mapped = _as_mapping(row)
        item = str(mapped.get("missing_item") or "")
        action = str(mapped.get("operator_action") or "")
        if item and action:
            actions[item] = action
    return actions


def _allowed_next_scope(v4_ready: bool, gatemate_allowed: bool, ssqa_allowed: bool) -> str:
    if not v4_ready:
        return "blocked: required Exp 3106 v3 operator-evidence artifact is missing or unreadable"
    if not gatemate_allowed:
        return (
            "blocked: operator must commit authoritative GateMate Pin_out/CCF binding, "
            "host reader command, expected transcript, opened safety limits, and "
            "host-visible smoke evidence"
        )
    if not ssqa_allowed:
        return (
            "operator_allowed_future_task: documented GateMate rerun evidence is "
            "complete, but SSQA readback remains blocked until host-visible smoke "
            "transcript evidence is complete"
        )
    return (
        "operator_allowed_future_task: documented evidence supports an "
        "operator-owned SSQA readback task only; Exp 3119 ran no hardware "
        "command and makes no timing or speedup claim"
    )


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("operator_evidence_ingestion_v4_ready") is not True:
        missing = ", ".join(_as_text_list(artifact.get("missing_source_artifacts")))
        return f"blocked_precondition: missing GateMate/SSQA source artifact(s): {missing}"
    gate = str(artifact.get("gatemate_rerun_allowed")).lower()
    ssqa = str(artifact.get("ssqa_readback_allowed")).lower()
    return (
        "complete: "
        "operator_evidence_ingestion_v4_ready=true; "
        f"gatemate_rerun_allowed={gate}; ssqa_readback_allowed={ssqa}; "
        "hardware_commands_run=0; speedup_claim_made=false"
    )


def _ccf_binding_from_text(text: str, signal: str) -> JsonDict:
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.search(
            rf"\bPin_out\b\s+{re.escape(signal)}\s+Loc\s*=\s*([A-Za-z0-9_]+)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            return {
                "signal_name": signal,
                "pin": match.group(1),
                "line": line.strip(),
                "line_number": line_number,
                "source_path": GATEMATE_CCF_REL_PATH.as_posix(),
            }
    return {}


def _field_present(payload: Mapping[str, Any], field_id: str) -> bool:
    value = payload.get(field_id)
    if isinstance(value, bool):
        return value is True
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, list):
        return any(_concrete(item) for item in value)
    return _concrete(value)


def _concrete(value: Any) -> bool:
    if isinstance(value, list):
        return any(_concrete(item) for item in value)
    text = str(value or "").strip()
    lowered = text.lower()
    return (
        bool(text)
        and not lowered.startswith("blocked")
        and "explicit_no_ready_contract" not in lowered
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _coerce_rel_path(value: Any) -> Path:
    text = str(value or "").strip()
    return Path(text) if text else Path("")


def _has_path(path: Path) -> bool:
    return path.as_posix() not in {"", "."}


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []
