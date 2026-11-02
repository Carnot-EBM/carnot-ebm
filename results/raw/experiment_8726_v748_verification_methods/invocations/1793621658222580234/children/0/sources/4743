"""Build the Exp 3063 GateMate no-rerun operator-action ledger.

Spec refs: REQ-HW-089, SCENARIO-HW-089.

The ledger is a stop sign for blocked GateMate branches. It does not try to
repair the hardware path. Instead it reads checked-in artifacts, names the
operator-owned evidence still missing, and keeps RTL, flash, and SSQA reruns
mechanically closed until a host-visible output contract exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.286"
SCHEMA = "carnot.gatemate.no_rerun_operator_action_ledger.v1"
ARTIFACT = "experiment_3063_gatemate_no_rerun_operator_action_ledger_v1"
OUTPUT_REL_PATH = Path("results/experiment_3063_gatemate_no_rerun_operator_action_ledger_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3063_gatemate_no_rerun_operator_action_ledger_v1.py"

EXP3048_REL_PATH = Path("results/experiment_3048_gatemate_output_contract_operator_package_v1.json")
EXP3049_REL_PATH = Path("results/experiment_3049_gatemate_output_shim_rtl_ccf_sim_v2.json")
EXP3051_REQUESTED_REL_PATH = Path("results/experiment_3051_ssqa_readback_eligibility_gate_v3.json")
EXP3051_BOUNDED_REL_PATH = Path(
    "results/experiment_3051_ssqa_readback_eligibility_bounded_gate_v3.json"
)
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
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
    {
        "experiment_id": "exp3048",
        "paths": (EXP3048_REL_PATH,),
        "role": "output_contract_operator_package",
        "required": True,
        "source_type": "json",
    },
    {
        "experiment_id": "exp3049",
        "paths": (EXP3049_REL_PATH,),
        "role": "rtl_ccf_sim_gate_status",
        "required": False,
        "source_type": "json",
    },
    EXP3051_SOURCE,
    {
        "experiment_id": "conductor_log",
        "paths": (CONDUCTOR_LOG_REL_PATH,),
        "role": "downstream_gate_history",
        "required": False,
        "source_type": "text",
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
INFERENCE_SUBSTRATE = {
    "kind": "hardware_contract_no_rerun_ledger",
    "source": "checked_in_local_artifacts",
    "model_inference": False,
    "executes_models": False,
    "executes_hardware": False,
    "executes_conductor": False,
    "flash_attempted": False,
    "rtl_run": False,
    "local_repo_only": True,
    "timing_or_speedup_claim": False,
}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning empty evidence for missing or malformed files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def build_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-HW-089: build the no-rerun ledger from local checked-in artifacts."""

    root_path = Path(root)
    source_artifacts = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    source_by_id = {str(row["experiment_id"]): row for row in source_artifacts}
    exp3048 = _as_mapping(source_by_id["exp3048"].get("payload"))
    missing_sources = [
        str(row["path"])
        for row in source_artifacts
        if row.get("required") is True and row.get("readable") is not True
    ]
    operator_contract = _operator_contract(exp3048)
    evidence = _required_evidence(operator_contract)
    core_evidence_ready = all(row["rerun_satisfied"] is True for row in evidence)
    upstream_ready = exp3048.get("gatemate_output_contract_ready") is True and exp3048.get(
        "host_visible_io_plan_ready"
    ) is True
    ledger_ready = not missing_sources
    rerun_allowed = bool(ledger_ready and upstream_ready and core_evidence_ready)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "gatemate_no_rerun_ledger_ready": ledger_ready,
        "gatemate_rerun_allowed": rerun_allowed,
        "operator_contract": operator_contract,
        "missing_operator_actions": _missing_operator_actions(operator_contract, evidence),
        "required_evidence_before_rerun": evidence,
        "downstream_tasks_blocked": _downstream_rows(rerun_allowed, source_by_id, operator_contract),
        "rerun_permission_basis": _rerun_permission_basis(rerun_allowed, evidence),
        "hardware_execution_claim_made": False,
        "speedup_claim_made": False,
        "source_artifacts": _public_sources(source_artifacts),
        "missing_source_artifacts": missing_sources,
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "hardware_execution_performed": False,
        "flash_command_executed": "",
        "rtl_commands_run": [],
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
) -> Path:
    """Build and persist the Exp 3063 deliverable JSON."""

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


def _operator_contract(exp3048: Mapping[str, Any]) -> JsonDict:
    selected_signal = str(exp3048.get("selected_output_signal") or "done")
    binding = _as_mapping(exp3048.get("ccf_binding"))
    reader_command = str(exp3048.get("host_reader_command") or "")
    transcript = _as_text_list(exp3048.get("expected_transcript"))
    safety_limits = _as_mapping(exp3048.get("safety_limits"))
    return {
        "selected_output_signal": selected_signal,
        "ccf_binding": binding,
        "host_reader_command": reader_command if _concrete(reader_command) else "",
        "expected_transcript": transcript if _concrete(transcript) else [],
        "safety_limits": safety_limits,
        "gatemate_output_contract_ready": exp3048.get("gatemate_output_contract_ready") is True,
        "host_visible_io_plan_ready": exp3048.get("host_visible_io_plan_ready") is True,
        "upstream_honest_verdict": str(exp3048.get("honest_verdict") or ""),
    }


def _required_evidence(contract: Mapping[str, Any]) -> list[JsonDict]:
    signal = str(contract.get("selected_output_signal") or "done")
    safety_limits = _as_mapping(contract.get("safety_limits"))
    return [
        _evidence_row(
            "selected_output_signal",
            "Selected deterministic RTL output signal.",
            bool(signal),
            bool(signal),
            f"selected_output_signal={signal}",
        ),
        _evidence_row(
            "authoritative_pinout_ccf_binding",
            f"Authoritative GateMate A1-EVB-2M CCF Pin_out binding for {signal}.",
            bool(_as_mapping(contract.get("ccf_binding"))),
            bool(_as_mapping(contract.get("ccf_binding"))),
            "operator_contract.ccf_binding",
        ),
        _evidence_row(
            "host_reader_command",
            f"Concrete host reader command for {signal}.",
            _concrete(str(contract.get("host_reader_command") or "")),
            _concrete(str(contract.get("host_reader_command") or "")),
            "operator_contract.host_reader_command",
        ),
        _evidence_row(
            "expected_transcript",
            f"Expected pass/fail transcript for the {signal} reader.",
            _concrete(_as_text_list(contract.get("expected_transcript"))),
            _concrete(_as_text_list(contract.get("expected_transcript"))),
            "operator_contract.expected_transcript",
        ),
        _evidence_row(
            "safety_limits",
            "Safety limits that mechanically keep flash and speedup claims gated.",
            bool(safety_limits),
            safety_limits.get("downstream_flash_gate_open") is True,
            "operator_contract.safety_limits",
        ),
    ]


def _evidence_row(
    evidence_id: str,
    description: str,
    present: bool,
    rerun_satisfied: bool,
    citation: str,
) -> JsonDict:
    return {
        "evidence_id": evidence_id,
        "required": True,
        "present": present,
        "rerun_satisfied": rerun_satisfied,
        "description": description,
        "source_artifact": EXP3048_REL_PATH.as_posix(),
        "source_field": citation,
    }


def _missing_operator_actions(
    contract: Mapping[str, Any],
    evidence: list[Mapping[str, Any]],
) -> list[JsonDict]:
    signal = str(contract.get("selected_output_signal") or "done")
    by_id = {str(row["evidence_id"]): row for row in evidence}
    actions: list[JsonDict] = []
    if by_id["selected_output_signal"].get("present") is not True:
        actions.append(
            _action(
                "selected_output_signal",
                False,
                "Select one deterministic GateMate RTL output signal before any downstream rerun.",
            )
        )
    if by_id["authoritative_pinout_ccf_binding"].get("present") is not True:
        actions.append(
            _action(
                "authoritative_pinout_ccf_binding",
                False,
                f"Provide an authoritative GateMate A1-EVB-2M output pinout and commit a CCF Pin_out binding for {signal}.",
            )
        )
    if by_id["host_reader_command"].get("present") is not True:
        actions.append(
            _action(
                "host_reader_command",
                False,
                f"Commit a concrete host reader command for {signal}: GPIO/LED read, UART serial decode, or JTAG-readable status command.",
            )
        )
    if by_id["expected_transcript"].get("present") is not True:
        actions.append(
            _action(
                "expected_transcript",
                False,
                f"Record the expected pass/fail transcript for the {signal} host reader command.",
            )
        )
    if by_id["safety_limits"].get("present") is not True:
        actions.append(
            _action(
                "safety_limits",
                False,
                "Commit safety limits that keep flash attempts, hardware claims, and speedup claims gated.",
            )
        )
    return actions


def _action(missing_item: str, present: bool, operator_action: str) -> JsonDict:
    return {
        "missing_item": missing_item,
        "present": present,
        "operator_action": operator_action,
        "source_artifact": EXP3048_REL_PATH.as_posix(),
    }


def _downstream_rows(
    rerun_allowed: bool,
    source_by_id: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[JsonDict]:
    source3049 = _as_mapping(source_by_id.get("exp3049"))
    source3051 = _as_mapping(source_by_id.get("exp3051"))
    blocker = _blocker(contract, source3049)
    return [
        {
            "task_id": "exp3049-gatemate-output-shim-rtl-ccf-sim-v2",
            "branch_type": "rtl_ccf_sim",
            "allowed_to_rerun": rerun_allowed,
            "matrix_status": "ready_to_rerun" if rerun_allowed else "blocked",
            "upstream_blocker": blocker,
            "source_artifact": str(source3049.get("path") or EXP3049_REL_PATH.as_posix()),
        },
        {
            "task_id": "exp3050-gatemate-host-visible-flash-smoke-v5",
            "branch_type": "flash_smoke",
            "allowed_to_rerun": rerun_allowed,
            "matrix_status": "ready_to_rerun" if rerun_allowed else "gate_skipped",
            "upstream_blocker": "exp3049 output shim/CCF simulation has not passed with a host-visible contract",
            "source_artifact": CONDUCTOR_LOG_REL_PATH.as_posix(),
        },
        {
            "task_id": "exp3051-ssqa-readback-eligibility-bounded-gate-v3",
            "branch_type": "ssqa_readback",
            "allowed_to_rerun": rerun_allowed,
            "matrix_status": "ready_to_rerun" if rerun_allowed else "gate_skipped",
            "upstream_blocker": "exp3050.gatemate_host_visible_smoke_passed is missing or false",
            "source_artifact": str(source3051.get("path") or EXP3051_BOUNDED_REL_PATH.as_posix()),
        },
    ]


def _blocker(contract: Mapping[str, Any], exp3049_source: Mapping[str, Any]) -> str:
    payload = _as_mapping(exp3049_source.get("payload"))
    if payload.get("gate_check_summary"):
        return "exp3048.gatemate_output_contract_ready=false; " + str(payload["gate_check_summary"])
    if contract.get("gatemate_output_contract_ready") is not True:
        return "exp3048.gatemate_output_contract_ready=false"
    return "host-visible GateMate output evidence incomplete"


def _rerun_permission_basis(rerun_allowed: bool, evidence: list[Mapping[str, Any]]) -> list[JsonDict]:
    if not rerun_allowed:
        return []
    return [
        {
            "basis_id": "exp3048_output_contract_ready",
            "cite": EXP3048_REL_PATH.as_posix(),
            "satisfied_evidence": [str(row["evidence_id"]) for row in evidence],
        }
    ]


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
    if artifact.get("gatemate_no_rerun_ledger_ready") is not True:
        return "blocked_precondition: exp3048 GateMate output contract package missing or malformed"
    allowed = str(artifact.get("gatemate_rerun_allowed")).lower()
    blocked_count = sum(
        1
        for row in _as_list(artifact.get("downstream_tasks_blocked"))
        if _as_mapping(row).get("allowed_to_rerun") is not True
    )
    return (
        "complete: "
        f"gatemate_no_rerun_ledger_ready=true; gatemate_rerun_allowed={allowed}; "
        f"downstream_blocked={blocked_count}"
    )


def _concrete(value: Any) -> bool:
    if isinstance(value, list):
        return any(_concrete(item) for item in value)
    text = str(value).strip()
    lowered = text.lower()
    return bool(text) and not lowered.startswith("blocked") and "explicit_no_ready_contract" not in lowered


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []
