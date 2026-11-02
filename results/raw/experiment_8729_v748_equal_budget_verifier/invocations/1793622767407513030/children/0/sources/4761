"""Build the Exp 3092 GateMate/SSQA operator evidence ingestion ledger.

Spec refs: REQ-HW-092, SCENARIO-HW-092.

This module performs a repository-only evidence comparison. It reads the prior
Exp 3078 missing-action ledger, checks whether those operator-owned artifacts
now exist, and writes the next hardware boundary without flashing boards,
rerunning timing, or treating old build artifacts as performance evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.288"
SCHEMA = "carnot.gatemate_ssqa_operator_evidence_ingestion.v2"
ARTIFACT = "experiment_3092_gatemate_ssqa_operator_evidence_ingestion_v2"
OUTPUT_REL_PATH = Path("results/experiment_3092_gatemate_ssqa_operator_evidence_ingestion_v2.json")

EXP3078_REL_PATH = Path("results/experiment_3078_gatemate_ssqa_no_rerun_operator_refresh_v1.json")
EXP3048_REL_PATH = Path("results/experiment_3048_gatemate_output_contract_operator_package_v1.json")
EXP3063_REL_PATH = Path("results/experiment_3063_gatemate_no_rerun_operator_action_ledger_v1.json")
EXP3064_REL_PATH = Path(
    "results/experiment_3064_ssqa_host_visible_readback_boundary_ledger_v1.json"
)
EXP3050_REL_PATH = Path("results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json")
EXP3051_REL_PATH = Path("results/experiment_3051_ssqa_readback_eligibility_bounded_gate_v3.json")
EXP3049_REL_PATH = Path("results/experiment_3049_gatemate_output_shim_rtl_ccf_sim_v2.json")
HARDWARE_WISHLIST_REL_PATH = Path("research-hardware-wishlist.md")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
STATUS_REL_PATH = Path("ops/status.md")
CHANGELOG_REL_PATH = Path("ops/changelog.md")
GATEMATE_CCF_REL_PATH = Path("hardware/gatemate/ising_n16_gatemate.ccf")
GATEMATE_RTL_REL_PATH = Path("hardware/gatemate/ising_n16_gatemate.v")
GATEMATE_TEST_VECTOR_REL_PATH = Path("hardware/gatemate/ising_n16_gatemate_test_vector.json")
GATEMATE_JTAG_DOC_REL_PATH = Path("docs/jtag-wiring-gatemate-dirtyjtag.md")

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
        "experiment_id": "exp3078",
        "path": EXP3078_REL_PATH,
        "role": "prior_missing_action_ledger",
        "required": True,
        "source_type": "json",
    },
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
        "experiment_id": "exp3051",
        "path": EXP3051_REL_PATH,
        "role": "ssqa_readback_gate_status",
        "required": False,
        "source_type": "json",
    },
    {
        "experiment_id": "exp3049",
        "path": EXP3049_REL_PATH,
        "role": "gatemate_output_shim_gate_status",
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
        "experiment_id": "changelog",
        "path": CHANGELOG_REL_PATH,
        "role": "milestone_gate_context",
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
    {
        "experiment_id": "gatemate_test_vector",
        "path": GATEMATE_TEST_VECTOR_REL_PATH,
        "role": "checked_in_gatemate_test_vector",
        "required": False,
        "source_type": "json",
    },
    {
        "experiment_id": "gatemate_jtag_doc",
        "path": GATEMATE_JTAG_DOC_REL_PATH,
        "role": "operator_wiring_context",
        "required": False,
        "source_type": "text",
    },
)

SEARCH_DIRS: tuple[Path, ...] = (
    Path("results"),
    Path("hardware/gatemate"),
    Path("docs"),
    Path("ops"),
    Path("logs"),
)
SEARCH_PATTERNS: tuple[str, ...] = (
    "*gatemate*",
    "*ssqa*",
    "*transcript*",
    "*smoke*",
    "*pinout*",
    "*ccf*",
    "*safety*",
    "*readback*",
)
SEARCH_RELEVANCE_TOKENS: tuple[str, ...] = ("gatemate", "ssqa")

DEFAULT_PRIOR_ACTIONS: dict[str, str] = {
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
        "timing fields, and sampler configuration."
    ),
}

INFERENCE_SUBSTRATE = {
    "kind": "operator_evidence_ledger",
    "source": "checked_in_local_artifacts",
    "model_inference": False,
    "no_live_model_inference": True,
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
    """Read a JSON file as evidence, returning empty when evidence is unusable.

    Hardware gates should fail closed. A missing or malformed artifact therefore
    becomes no evidence instead of partial permission to run board work.
    """

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def build_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-HW-092: build the v2 ledger from checked-in operator evidence only."""

    root_path = Path(root)
    source_rows = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    source_by_id = {str(row["experiment_id"]): row for row in source_rows}
    missing_sources = [
        str(row["path"])
        for row in source_rows
        if row.get("required") is True and row.get("readable") is not True
    ]
    checked_paths = _checked_paths(root_path, source_rows)
    exp3078 = _payload(source_by_id, "exp3078")
    exp3048 = _payload(source_by_id, "exp3048")
    exp3050 = _payload(source_by_id, "exp3050")
    ccf_text = str(_as_mapping(source_by_id.get("gatemate_ccf")).get("text") or "")
    prior_actions = _prior_missing_action_map(exp3078)
    gatemate_checks = _gatemate_evidence_checks(exp3048, ccf_text, prior_actions)
    host_evidence = _host_visible_evidence(exp3050)
    ingestion_ready = not missing_sources
    gatemate_allowed = ingestion_ready and all(row["present"] for row in gatemate_checks)
    ssqa_allowed = ingestion_ready and gatemate_allowed and host_evidence["ready"] is True

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "operator_evidence_ingestion_ready": ingestion_ready,
        "gatemate_rerun_allowed": gatemate_allowed,
        "ssqa_readback_allowed": ssqa_allowed,
        "operator_ready_artifacts": _operator_ready_artifacts(
            gatemate_checks,
            host_evidence,
            ssqa_allowed,
        ),
        "missing_operator_actions": _missing_operator_actions(
            gatemate_checks,
            host_evidence,
        ),
        "allowed_next_experiment_scope": _allowed_next_scope(
            ingestion_ready,
            gatemate_allowed,
            ssqa_allowed,
        ),
        "checked_paths": checked_paths,
        "evidence_comparison": {
            "baseline_artifact": EXP3078_REL_PATH.as_posix(),
            "prior_missing_items": list(prior_actions),
            "current_missing_items": [],
        },
        "gatemate_evidence_checks": gatemate_checks,
        "host_visible_smoke_evidence": host_evidence,
        "speedup_claim_made": False,
        "hardware_commands_run": [],
        "hardware_execution_claim_made": False,
        "hardware_execution_performed": False,
        "hardware_readback_attempted": False,
        "source_artifacts": _public_sources(source_rows),
        "missing_source_artifacts": missing_sources,
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "honest_verdict": "",
    }
    artifact["evidence_comparison"]["current_missing_items"] = [
        str(row["missing_item"]) for row in artifact["missing_operator_actions"]
    ]
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
) -> Path:
    """Persist the Exp 3092 JSON deliverable."""

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
        "source_type": source_type,
        "present": path.is_file(),
        "readable": readable,
        "payload": payload,
        "text": text,
    }


def _payload(source_by_id: Mapping[str, Mapping[str, Any]], experiment_id: str) -> JsonDict:
    return _as_mapping(_as_mapping(source_by_id.get(experiment_id)).get("payload"))


def _checked_paths(root: Path, source_rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    rows_by_path = {
        str(row["path"]): {
            "path": str(row["path"]),
            "present": row.get("present") is True,
            "readable": row.get("readable") is True,
            "evidence_class": str(row["role"]),
        }
        for row in source_rows
    }
    for rel_path in _discover_evidence_candidates(root):
        rows_by_path.setdefault(
            rel_path.as_posix(),
            {
                "path": rel_path.as_posix(),
                "present": True,
                "readable": True,
                "evidence_class": "discovered_operator_evidence_candidate",
            },
        )
    return [rows_by_path[path] for path in sorted(rows_by_path)]


def _discover_evidence_candidates(root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for rel_dir in SEARCH_DIRS:
        base = root / rel_dir
        if not base.is_dir():
            continue
        for pattern in SEARCH_PATTERNS:
            for path in base.rglob(pattern):
                rel_path = path.relative_to(root)
                rel_text = rel_path.as_posix().lower()
                if path.is_file() and any(token in rel_text for token in SEARCH_RELEVANCE_TOKENS):
                    candidates.add(rel_path)
    return sorted(candidates)


def _prior_missing_action_map(exp3078: Mapping[str, Any]) -> dict[str, str]:
    actions = dict(DEFAULT_PRIOR_ACTIONS)
    for row in _as_list(exp3078.get("missing_operator_actions")):
        item = str(_as_mapping(row).get("missing_item") or "")
        action = str(_as_mapping(row).get("operator_action") or "")
        if item and action:
            actions[item] = action
    return actions


def _gatemate_evidence_checks(
    exp3048: Mapping[str, Any],
    ccf_text: str,
    prior_actions: Mapping[str, str],
) -> list[JsonDict]:
    selected_signal = str(exp3048.get("selected_output_signal") or "done")
    output_contract_ready = exp3048.get("gatemate_output_contract_ready") is True
    host_plan_ready = exp3048.get("host_visible_io_plan_ready") is True
    ccf_binding = _as_mapping(exp3048.get("ccf_binding")) or _ccf_binding_from_text(
        ccf_text,
        selected_signal,
    )
    host_reader_command = str(exp3048.get("host_reader_command") or "")
    expected_transcript = _as_text_list(exp3048.get("expected_transcript"))
    safety_limits = _as_mapping(exp3048.get("safety_limits"))
    safety_open = (
        safety_limits.get("downstream_flash_gate_open") is True
        and int(safety_limits.get("max_flash_attempts_without_operator_review") or 0) > 0
    )
    return [
        _evidence_check(
            "authoritative_pinout_ccf_binding",
            output_contract_ready and host_plan_ready and bool(ccf_binding),
            prior_actions,
            EXP3048_REL_PATH,
            "ccf_binding",
            ccf_binding,
            [EXP3048_REL_PATH, GATEMATE_CCF_REL_PATH],
        ),
        _evidence_check(
            "host_reader_command",
            output_contract_ready and host_plan_ready and _concrete(host_reader_command),
            prior_actions,
            EXP3048_REL_PATH,
            "host_reader_command",
            host_reader_command,
            [EXP3048_REL_PATH],
        ),
        _evidence_check(
            "expected_transcript",
            output_contract_ready and host_plan_ready and _concrete(expected_transcript),
            prior_actions,
            EXP3048_REL_PATH,
            "expected_transcript",
            expected_transcript,
            [EXP3048_REL_PATH],
        ),
        _evidence_check(
            "safety_limits",
            output_contract_ready and host_plan_ready and safety_open,
            prior_actions,
            EXP3048_REL_PATH,
            "safety_limits",
            safety_limits,
            [EXP3048_REL_PATH],
        ),
    ]


def _evidence_check(
    evidence_id: str,
    present: bool,
    prior_actions: Mapping[str, str],
    path: Path,
    source_field: str,
    evidence: Any,
    checked_paths: list[Path],
) -> JsonDict:
    return {
        "evidence_id": evidence_id,
        "present": bool(present),
        "operator_action": str(prior_actions.get(evidence_id) or DEFAULT_PRIOR_ACTIONS[evidence_id]),
        "path": path.as_posix(),
        "source_field": source_field,
        "checked_paths": [item.as_posix() for item in checked_paths],
        "evidence": evidence if present else None,
    }


def _host_visible_evidence(exp3050: Mapping[str, Any]) -> JsonDict:
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
    smoke_present = bool(exp3050)
    smoke_passed = exp3050.get("gatemate_host_visible_smoke_passed") is True
    transcript_matched = exp3050.get("transcript_matched") is True
    return {
        "path": EXP3050_REL_PATH.as_posix(),
        "present": smoke_present,
        "gatemate_host_visible_smoke_passed": smoke_passed,
        "transcript_matched": transcript_matched,
        "missing_required_fields": missing,
        "required_fields": required_fields,
        "ready": smoke_present and smoke_passed and transcript_matched and not missing,
    }


def _operator_ready_artifacts(
    gatemate_checks: list[Mapping[str, Any]],
    host_evidence: Mapping[str, Any],
    ssqa_allowed: bool,
) -> list[JsonDict]:
    ready = [
        {
            "evidence_id": str(row["evidence_id"]),
            "path": str(row["path"]),
            "source_field": str(row["source_field"]),
            "checked_paths": _as_text_list(row.get("checked_paths")),
            "evidence": row.get("evidence"),
        }
        for row in gatemate_checks
        if row.get("present") is True
    ]
    if ssqa_allowed:
        ready.append(
            {
                "evidence_id": "host_visible_smoke_evidence",
                "path": EXP3050_REL_PATH.as_posix(),
                "source_field": "required_host_visible_fields",
                "checked_paths": [EXP3050_REL_PATH.as_posix()],
                "evidence": {"missing_required_fields": []},
            }
        )
    return ready


def _missing_operator_actions(
    gatemate_checks: list[Mapping[str, Any]],
    host_evidence: Mapping[str, Any],
) -> list[JsonDict]:
    missing = [
        {
            "missing_item": str(row["evidence_id"]),
            "present": False,
            "operator_action": str(row["operator_action"]),
            "source_artifact": str(row["path"]),
            "checked_paths": _as_text_list(row.get("checked_paths")),
        }
        for row in gatemate_checks
        if row.get("present") is not True
    ]
    if host_evidence.get("ready") is not True:
        missing.append(
            {
                "missing_item": "host_visible_smoke_evidence",
                "present": host_evidence.get("present") is True,
                "operator_action": DEFAULT_PRIOR_ACTIONS["host_visible_smoke_evidence"],
                "source_artifact": EXP3050_REL_PATH.as_posix(),
                "checked_paths": [EXP3050_REL_PATH.as_posix()],
                "missing_required_fields": _as_text_list(
                    host_evidence.get("missing_required_fields")
                ),
            }
        )
    return missing


def _allowed_next_scope(
    ingestion_ready: bool,
    gatemate_allowed: bool,
    ssqa_allowed: bool,
) -> str:
    if not ingestion_ready:
        return "blocked: required prior GateMate/SSQA source artifact is missing"
    if not gatemate_allowed:
        return (
            "blocked: operator must commit authoritative GateMate Pin_out/CCF binding, "
            "host reader command, expected transcript, and opened safety limits"
        )
    if not ssqa_allowed:
        return (
            "operator_allowed: run the gated GateMate host-visible flash smoke only; "
            "do not claim timing or speedup"
        )
    return (
        "operator_allowed: run the gated SSQA readback experiment using committed "
        "host-visible smoke evidence; do not make speedup claims without a new "
        "operator timing transcript"
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
    if artifact.get("operator_evidence_ingestion_ready") is not True:
        missing = ", ".join(_as_text_list(artifact.get("missing_source_artifacts")))
        return f"blocked_precondition: missing GateMate/SSQA source artifact(s): {missing}"
    gate = str(artifact.get("gatemate_rerun_allowed")).lower()
    ssqa = str(artifact.get("ssqa_readback_allowed")).lower()
    return (
        "complete: "
        "operator_evidence_ingestion_ready=true; "
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
