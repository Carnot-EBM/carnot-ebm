"""Build the Exp 3078 GateMate/SSQA no-rerun operator refresh.

Spec refs: REQ-HW-091, SCENARIO-HW-091.

This refresh is a matrix v21 boundary artifact. It reads the prior GateMate and
SSQA ledgers plus any committed operator evidence, then states what hardware
work is allowed next. It deliberately does not touch the board, rerun RTL, or
turn old flash/contact artifacts into sampler or speedup claims.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.287"
SCHEMA = "carnot.gatemate_ssqa_no_rerun_operator_refresh.v1"
ARTIFACT = "experiment_3078_gatemate_ssqa_no_rerun_operator_refresh_v1"
OUTPUT_REL_PATH = Path("results/experiment_3078_gatemate_ssqa_no_rerun_operator_refresh_v1.json")

EXP3048_REL_PATH = Path("results/experiment_3048_gatemate_output_contract_operator_package_v1.json")
EXP3063_REL_PATH = Path("results/experiment_3063_gatemate_no_rerun_operator_action_ledger_v1.json")
EXP3064_REL_PATH = Path(
    "results/experiment_3064_ssqa_host_visible_readback_boundary_ledger_v1.json"
)
EXP3050_REL_PATH = Path("results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json")
HARDWARE_WISHLIST_REL_PATH = Path("research-hardware-wishlist.md")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
STATUS_REL_PATH = Path("ops/status.md")
GATEMATE_CCF_REL_PATH = Path("hardware/gatemate/ising_n16_gatemate.ccf")
GATEMATE_RTL_REL_PATH = Path("hardware/gatemate/ising_n16_gatemate.v")

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

SOURCE_SPECS: tuple[JsonDict, ...] = (
    {
        "experiment_id": "exp3048",
        "path": EXP3048_REL_PATH,
        "role": "gatemate_output_contract_operator_package",
        "required": True,
        "source_type": "json",
    },
    {
        "experiment_id": "exp3063",
        "path": EXP3063_REL_PATH,
        "role": "gatemate_no_rerun_operator_action_ledger",
        "required": True,
        "source_type": "json",
    },
    {
        "experiment_id": "exp3064",
        "path": EXP3064_REL_PATH,
        "role": "ssqa_host_visible_readback_boundary_ledger",
        "required": True,
        "source_type": "json",
    },
    {
        "experiment_id": "exp3050",
        "path": EXP3050_REL_PATH,
        "role": "host_visible_gatemate_smoke_transcript",
        "required": False,
        "source_type": "json",
    },
    {
        "experiment_id": "hardware_wishlist",
        "path": HARDWARE_WISHLIST_REL_PATH,
        "role": "hardware_claim_boundary",
        "required": True,
        "source_type": "text",
    },
    {
        "experiment_id": "conductor_log",
        "path": CONDUCTOR_LOG_REL_PATH,
        "role": "hardware_gate_history",
        "required": False,
        "source_type": "text",
    },
    {
        "experiment_id": "status",
        "path": STATUS_REL_PATH,
        "role": "current_ops_status",
        "required": False,
        "source_type": "text",
    },
    {
        "experiment_id": "gatemate_ccf",
        "path": GATEMATE_CCF_REL_PATH,
        "role": "checked_in_gatemate_constraints",
        "required": False,
        "source_type": "text",
    },
    {
        "experiment_id": "gatemate_rtl",
        "path": GATEMATE_RTL_REL_PATH,
        "role": "checked_in_gatemate_rtl",
        "required": False,
        "source_type": "text",
    },
)

INFERENCE_SUBSTRATE = {
    "kind": "gatemate_ssqa_no_rerun_operator_refresh",
    "source": "checked_in_local_artifacts",
    "model_inference": False,
    "executes_models": False,
    "executes_hardware": False,
    "executes_conductor": False,
    "flash_attempted": False,
    "rtl_or_pnr_run": False,
    "hardware_readback_attempted": False,
    "local_repo_only": True,
    "timing_or_speedup_claim": False,
}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON artifact while treating missing or malformed files as no evidence.

    Hardware boundary ledgers must fail closed. Returning an empty object keeps
    the caller from accidentally turning a parse failure into permission.
    """

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def build_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-HW-091: build the matrix v21 refresh from checked-in evidence only."""

    root_path = Path(root)
    source_rows = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    source_by_id = {str(row["experiment_id"]): row for row in source_rows}
    missing_sources = [
        str(row["path"])
        for row in source_rows
        if row.get("required") is True and row.get("readable") is not True
    ]
    exp3048 = _payload(source_by_id, "exp3048")
    exp3063 = _payload(source_by_id, "exp3063")
    exp3064 = _payload(source_by_id, "exp3064")
    exp3050 = _payload(source_by_id, "exp3050")
    ccf_text = str(_as_mapping(source_by_id.get("gatemate_ccf")).get("text") or "")

    gate = _gatemate_evidence(exp3048, exp3063, ccf_text)
    host = _host_visible_evidence(exp3050, exp3064)
    refresh_ready = not missing_sources
    gatemate_allowed = refresh_ready and gate["ready"] is True
    ssqa_allowed = refresh_ready and host["ready"] is True

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "gatemate_ssqa_refresh_ready": refresh_ready,
        "gatemate_rerun_allowed": gatemate_allowed,
        "ssqa_readback_allowed": ssqa_allowed,
        "missing_operator_actions": _missing_operator_actions(gate, host),
        "operator_ready_artifacts": _operator_ready_artifacts(
            gatemate_allowed,
            ssqa_allowed,
            gate,
            host,
        ),
        "next_allowed_hardware_task": _next_allowed_task(
            refresh_ready,
            gatemate_allowed,
            ssqa_allowed,
        ),
        "gatemate_evidence": gate,
        "host_visible_smoke_evidence": host,
        "hardware_execution_claim_made": False,
        "speedup_claim_made": False,
        "hardware_execution_performed": False,
        "hardware_readback_attempted": False,
        "flash_command_executed": "",
        "rtl_or_pnr_commands_run": [],
        "source_artifacts": _public_sources(source_rows),
        "missing_source_artifacts": missing_sources,
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
    """Persist the Exp 3078 JSON deliverable."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_payload(root: Path, spec: Mapping[str, Any]) -> JsonDict:
    rel_path = Path(str(spec["path"]))
    path = root / rel_path
    source_type = str(spec.get("source_type") or "json")
    payload = read_json_object(path) if source_type == "json" else {}
    text = path.read_text(encoding="utf-8") if source_type == "text" and path.is_file() else ""
    readable = path.is_file() and (source_type != "json" or bool(payload))
    return {
        "experiment_id": str(spec["experiment_id"]),
        "path": rel_path.as_posix(),
        "role": str(spec["role"]),
        "required": spec.get("required") is True,
        "present": path.is_file(),
        "readable": readable,
        "source_type": source_type,
        "payload": payload,
        "text": text,
    }


def _payload(source_by_id: Mapping[str, Mapping[str, Any]], experiment_id: str) -> JsonDict:
    return _as_mapping(_as_mapping(source_by_id.get(experiment_id)).get("payload"))


def _gatemate_evidence(
    exp3048: Mapping[str, Any],
    exp3063: Mapping[str, Any],
    ccf_text: str,
) -> JsonDict:
    selected_signal = str(exp3048.get("selected_output_signal") or "done")
    ccf_binding = _as_mapping(exp3048.get("ccf_binding")) or _ccf_binding_from_text(
        ccf_text,
        selected_signal,
    )
    host_reader_command = str(exp3048.get("host_reader_command") or "")
    expected_transcript = _as_text_list(exp3048.get("expected_transcript"))
    safety_limits = _as_mapping(exp3048.get("safety_limits"))
    safety_open = safety_limits.get("downstream_flash_gate_open") is True
    upstream_ready = (
        exp3048.get("gatemate_output_contract_ready") is True
        and exp3048.get("host_visible_io_plan_ready") is True
    )
    prior_ledger_allowed = exp3063.get("gatemate_rerun_allowed") is True
    evidence_rows = [
        _gate_evidence("authoritative_pinout_ccf_binding", bool(ccf_binding)),
        _gate_evidence("host_reader_command", _concrete(host_reader_command)),
        _gate_evidence("expected_transcript", _concrete(expected_transcript)),
        _gate_evidence("safety_limits", safety_open),
    ]
    ready = upstream_ready and prior_ledger_allowed and all(row["present"] for row in evidence_rows)
    return {
        "selected_output_signal": selected_signal,
        "ccf_binding": ccf_binding,
        "host_reader_command": host_reader_command if _concrete(host_reader_command) else "",
        "expected_transcript": expected_transcript if _concrete(expected_transcript) else [],
        "safety_limits": safety_limits,
        "upstream_output_contract_ready": upstream_ready,
        "prior_gatemate_rerun_allowed": prior_ledger_allowed,
        "required_evidence": evidence_rows,
        "ready": ready,
    }


def _gate_evidence(evidence_id: str, present: bool) -> JsonDict:
    return {
        "evidence_id": evidence_id,
        "present": present,
        "path": EXP3048_REL_PATH.as_posix(),
    }


def _host_visible_evidence(exp3050: Mapping[str, Any], exp3064: Mapping[str, Any]) -> JsonDict:
    required_fields = [
        {
            "field_id": field_id,
            "claim_scope": claim_scope,
            "present": _field_present(exp3050, field_id),
            "path": EXP3050_REL_PATH.as_posix(),
        }
        for field_id, claim_scope in HOST_VISIBLE_REQUIRED_FIELDS
    ]
    missing = [row["field_id"] for row in required_fields if row["present"] is not True]
    smoke_present = bool(exp3050)
    smoke_passed = exp3050.get("gatemate_host_visible_smoke_passed") is True
    transcript_matched = exp3050.get("transcript_matched") is True
    prior_ssqa_allowed = exp3064.get("ssqa_readback_allowed") is True
    ready = (
        smoke_present and smoke_passed and transcript_matched and prior_ssqa_allowed and not missing
    )
    return {
        "path": EXP3050_REL_PATH.as_posix(),
        "present": smoke_present,
        "gatemate_host_visible_smoke_passed": smoke_passed,
        "transcript_matched": transcript_matched,
        "prior_ssqa_readback_allowed": prior_ssqa_allowed,
        "missing_required_fields": missing,
        "required_fields": required_fields,
        "ready": ready,
    }


def _missing_operator_actions(gate: Mapping[str, Any], host: Mapping[str, Any]) -> list[JsonDict]:
    actions: list[JsonDict] = []
    gate_rows = {str(row["evidence_id"]): row for row in _as_list(gate.get("required_evidence"))}
    if _as_mapping(gate_rows.get("authoritative_pinout_ccf_binding")).get("present") is not True:
        actions.append(
            _action(
                "authoritative_pinout_ccf_binding",
                False,
                "Provide an authoritative GateMate A1-EVB-2M output pinout and commit a CCF Pin_out binding for done.",
                EXP3048_REL_PATH,
            )
        )
    if _as_mapping(gate_rows.get("host_reader_command")).get("present") is not True:
        actions.append(
            _action(
                "host_reader_command",
                False,
                "Commit a concrete host reader command for done.",
                EXP3048_REL_PATH,
            )
        )
    if _as_mapping(gate_rows.get("expected_transcript")).get("present") is not True:
        actions.append(
            _action(
                "expected_transcript",
                False,
                "Record the expected pass/fail transcript for the done host reader command.",
                EXP3048_REL_PATH,
            )
        )
    if _as_mapping(gate_rows.get("safety_limits")).get("present") is not True:
        actions.append(
            _action(
                "safety_limits",
                False,
                "Open the downstream flash safety gate only after the output contract is authoritative.",
                EXP3048_REL_PATH,
            )
        )
    if host.get("ready") is not True:
        actions.append(
            _action(
                "host_visible_smoke_evidence",
                host.get("present") is True,
                "Commit a passing GateMate host-visible smoke transcript with checksum, expected and observed output, CCF binding, flash/readback fields, timing fields, and sampler configuration.",
                EXP3050_REL_PATH,
            )
        )
    return actions


def _action(missing_item: str, present: bool, operator_action: str, source_path: Path) -> JsonDict:
    return {
        "missing_item": missing_item,
        "present": present,
        "operator_action": operator_action,
        "source_artifact": source_path.as_posix(),
    }


def _operator_ready_artifacts(
    gatemate_allowed: bool,
    ssqa_allowed: bool,
    gate: Mapping[str, Any],
    host: Mapping[str, Any],
) -> list[JsonDict]:
    ready: list[JsonDict] = []
    if gatemate_allowed:
        ready.extend(
            [
                _ready_artifact(
                    "authoritative_pinout_ccf_binding",
                    EXP3048_REL_PATH,
                    "ccf_binding",
                    _as_mapping(gate.get("ccf_binding")),
                ),
                _ready_artifact(
                    "host_reader_command",
                    EXP3048_REL_PATH,
                    "host_reader_command",
                    str(gate.get("host_reader_command") or ""),
                ),
                _ready_artifact(
                    "expected_transcript",
                    EXP3048_REL_PATH,
                    "expected_transcript",
                    _as_text_list(gate.get("expected_transcript")),
                ),
            ]
        )
    if ssqa_allowed:
        ready.append(
            _ready_artifact(
                "host_visible_smoke_evidence",
                EXP3050_REL_PATH,
                "required_host_visible_fields",
                {"missing_required_fields": _as_list(host.get("missing_required_fields"))},
            )
        )
    return ready


def _ready_artifact(evidence_id: str, path: Path, source_field: str, evidence: Any) -> JsonDict:
    return {
        "evidence_id": evidence_id,
        "path": path.as_posix(),
        "source_field": source_field,
        "evidence": evidence,
    }


def _next_allowed_task(
    refresh_ready: bool,
    gatemate_allowed: bool,
    ssqa_allowed: bool,
) -> str:
    if not refresh_ready:
        return "blocked: required prior GateMate/SSQA ledger source artifacts are missing"
    if not gatemate_allowed:
        return (
            "blocked: operator must commit authoritative GateMate A1-EVB-2M Pin_out/CCF "
            "binding, host reader command, expected transcript, and open safety gate "
            "before any GateMate rerun"
        )
    if not ssqa_allowed:
        return (
            "operator_allowed: run the gated GateMate output shim RTL/CCF simulation and "
            "host-visible flash smoke using the committed output contract; do not claim "
            "speedup"
        )
    return (
        "operator_allowed: run the gated SSQA readback task using "
        "results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json; "
        "do not claim speedup without a new timing transcript"
    )


def _public_sources(source_rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "experiment_id": str(row["experiment_id"]),
            "path": str(row["path"]),
            "role": str(row["role"]),
            "required": row.get("required") is True,
            "present": row.get("present") is True,
            "readable": row.get("readable") is True,
        }
        for row in source_rows
    ]


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("gatemate_ssqa_refresh_ready") is not True:
        missing = ", ".join(_as_text_list(artifact.get("missing_source_artifacts")))
        return f"blocked_precondition: missing GateMate/SSQA refresh source artifact(s): {missing}"
    gate = str(artifact.get("gatemate_rerun_allowed")).lower()
    ssqa = str(artifact.get("ssqa_readback_allowed")).lower()
    return (
        "complete: "
        f"gatemate_ssqa_refresh_ready=true; gatemate_rerun_allowed={gate}; "
        f"ssqa_readback_allowed={ssqa}; hardware_execution_claim_made=false; "
        "speedup_claim_made=false"
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
        return any(str(item).strip() for item in value)
    return bool(str(value or "").strip())


def _concrete(value: Any) -> bool:
    if isinstance(value, list):
        return any(_concrete(item) for item in value)
    text = str(value).strip()
    lowered = text.lower()
    return (
        bool(text)
        and not lowered.startswith("blocked")
        and "explicit_no_ready_contract" not in lowered
    )


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []
