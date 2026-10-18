"""Build the Exp 3106 GateMate/SSQA operator evidence ingestion v3 ledger.

Spec refs: REQ-HW-093, SCENARIO-HW-093.

This module is deliberately a ledger builder, not a hardware runner. It reads
the .288 capstone hardware boundary and the Exp 3092 v2 operator-evidence
ledger, rechecks only local files, and fails closed unless operator-owned
GateMate/SSQA evidence is complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from carnot.reporting import gatemate_ssqa_operator_evidence_ingestion_3092 as v2


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.289"
SCHEMA = "carnot.gatemate_ssqa_operator_evidence_ingestion.v3"
ARTIFACT = "experiment_3106_gatemate_ssqa_operator_evidence_ingestion_v3"
OUTPUT_REL_PATH = Path("results/experiment_3106_gatemate_ssqa_operator_evidence_ingestion_v3.json")

EXP3094_REL_PATH = Path("results/experiment_3094_capstone_v288.json")
EXP3092_REL_PATH = Path("results/experiment_3092_gatemate_ssqa_operator_evidence_ingestion_v2.json")
EXP3034_REL_PATH = Path("results/experiment_3034_gatemate_output_contract_pinout_decision_v1.json")
EXP3035_REL_PATH = Path("results/experiment_3035_gatemate_output_shim_rtl_ccf_sim.json")
EXP3036_REL_PATH = Path("results/experiment_3036_gatemate_host_visible_flash_smoke_v4.json")
EXP3037_REL_PATH = Path("results/experiment_3037_ssqa_bounded_rtl_pnr_gate_artifact_v2.json")

BASE_SOURCE_ROLES: tuple[tuple[Path, str, bool], ...] = (
    (EXP3094_REL_PATH, "dot288_capstone_hardware_ledger", True),
    (EXP3092_REL_PATH, "prior_operator_evidence_ingestion_v2", True),
)

HARDWARE_LEDGER_ROW_PATHS: tuple[tuple[str, Path, str], ...] = (
    ("exp3034", EXP3034_REL_PATH, "dot288_implied_output_contract_decision"),
    ("gatemate:output_contract", v2.EXP3048_REL_PATH, "dot288_implied_output_contract_package"),
    ("gatemate:no_rerun_ledger", v2.EXP3063_REL_PATH, "dot288_implied_no_rerun_ledger"),
    ("exp3035", EXP3035_REL_PATH, "dot288_implied_output_shim_status"),
    ("exp3036", EXP3036_REL_PATH, "dot288_implied_host_visible_flash_smoke"),
    ("exp3037", EXP3037_REL_PATH, "dot288_implied_ssqa_gate_artifact"),
    ("ssqa:readback_gate", v2.EXP3051_REL_PATH, "dot288_implied_ssqa_readback_gate"),
    (
        "ssqa:host_visible_readback_boundary",
        v2.EXP3064_REL_PATH,
        "dot288_implied_ssqa_readback_boundary",
    ),
    ("gatemate:host_visible_smoke", v2.EXP3050_REL_PATH, "dot288_implied_host_visible_smoke"),
    ("exp3078", v2.EXP3078_REL_PATH, "dot288_implied_refresh_ledger"),
    ("exp3092", EXP3092_REL_PATH, "dot288_implied_operator_evidence_v2"),
)

INFERENCE_SUBSTRATE = {
    "kind": "operator_evidence_only",
    "source": "checked_in_local_artifacts",
    "local_repo_only": True,
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
    """Read a JSON evidence file, returning no evidence on missing/bad input."""

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def build_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-HW-093: build the v3 boundary from .288 and v2 evidence only."""

    root_path = Path(root)
    capstone = read_json_object(root_path / EXP3094_REL_PATH)
    prior_v2 = read_json_object(root_path / EXP3092_REL_PATH)
    current_v2 = v2.build_artifact(root_path)
    source_catalog = _source_catalog(root_path, capstone, prior_v2, current_v2)
    checked_paths = [_path_status(root_path, path, role) for path, role, _required in source_catalog]
    source_artifacts = [
        {
            "path": row["path"],
            "role": role,
            "required": required,
            "present": row["present"],
            "readable": row["readable"],
        }
        for row, (_path, role, required) in zip(checked_paths, source_catalog)
    ]
    missing_sources = [
        row["path"] for row in source_artifacts if row["required"] is True and row["readable"] is not True
    ]
    v3_ready = not missing_sources and current_v2.get("operator_evidence_ingestion_ready") is True
    gatemate_allowed = v3_ready and current_v2.get("gatemate_rerun_allowed") is True
    ssqa_allowed = gatemate_allowed and current_v2.get("ssqa_readback_allowed") is True
    missing_actions = _missing_actions(current_v2, missing_sources)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "operator_evidence_ingestion_v3_ready": v3_ready,
        "gatemate_rerun_allowed": gatemate_allowed,
        "ssqa_readback_allowed": ssqa_allowed,
        "operator_ready_artifacts": _operator_ready_artifacts(current_v2, v3_ready),
        "missing_operator_actions": missing_actions,
        "allowed_next_experiment_scope": _allowed_next_scope(v3_ready, gatemate_allowed, ssqa_allowed),
        "checked_paths": checked_paths,
        "missing_checked_paths": [
            row["path"] for row in checked_paths if row["present"] is not True
        ],
        "missing_source_artifacts": missing_sources,
        "speedup_claim_made": False,
        "timing_claim_made": False,
        "hardware_commands_run": [],
        "hardware_execution_claim_made": False,
        "hardware_execution_performed": False,
        "hardware_readback_attempted": False,
        "source_artifacts": source_artifacts,
        "source_artifact_count": len(source_artifacts),
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
    """Persist the Exp 3106 JSON deliverable."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_catalog(
    root: Path,
    capstone: Mapping[str, Any],
    prior_v2: Mapping[str, Any],
    current_v2: Mapping[str, Any],
) -> list[tuple[Path, str, bool]]:
    catalog: dict[str, tuple[Path, str, bool]] = {}
    for path, role, required in BASE_SOURCE_ROLES:
        _add_source(catalog, path, role, required)
    for path, role in _paths_from_dot288_hardware_ledger(capstone):
        _add_source(catalog, path, role, False)
    for path, role in _paths_from_v2_ledger(prior_v2, "exp3092_v2_ledger"):
        _add_source(catalog, path, role, False)
    for path, role in _paths_from_v2_ledger(current_v2, "current_v2_recheck"):
        _add_source(catalog, path, role, False)
    for row in current_v2.get("checked_paths", []):
        mapped = _as_mapping(row)
        _add_source(
            catalog,
            _coerce_rel_path(mapped.get("path")),
            str(mapped.get("evidence_class") or "current_v2_checked_path"),
            False,
        )
    return [catalog[key] for key in sorted(catalog)]


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


def _paths_from_dot288_hardware_ledger(capstone: Mapping[str, Any]) -> list[tuple[Path, str]]:
    hardware = _as_mapping(_as_mapping(capstone.get("prd_gap_summary")).get("hardware_evidence"))
    rows = _as_text_list(hardware.get("row_ids")) + _as_text_list(
        hardware.get("publication_blocker_row_ids")
    )
    paths: list[tuple[Path, str]] = []
    for row_id in rows:
        for needle, path, role in HARDWARE_LEDGER_ROW_PATHS:
            if needle in row_id:
                paths.append((path, role))
    return paths


def _paths_from_v2_ledger(payload: Mapping[str, Any], role_prefix: str) -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = []
    for row in _as_list(payload.get("source_artifacts")):
        mapped = _as_mapping(row)
        paths.append((_coerce_rel_path(mapped.get("path")), f"{role_prefix}:source_artifact"))
    for row in _as_list(payload.get("checked_paths")):
        mapped = _as_mapping(row)
        paths.append((_coerce_rel_path(mapped.get("path")), f"{role_prefix}:checked_path"))
    for row in _as_list(payload.get("missing_operator_actions")):
        mapped = _as_mapping(row)
        paths.append((_coerce_rel_path(mapped.get("source_artifact")), f"{role_prefix}:missing_source"))
        for checked in _as_text_list(mapped.get("checked_paths")):
            paths.append((_coerce_rel_path(checked), f"{role_prefix}:missing_checked_path"))
    return paths


def _path_status(root: Path, rel_path: Path, role: str) -> JsonDict:
    path = root / rel_path
    return {
        "path": rel_path.as_posix(),
        "present": path.is_file(),
        "readable": path.is_file() and _readable(path),
        "evidence_class": role,
    }


def _readable(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return True
    return bool(read_json_object(path))


def _missing_actions(current_v2: Mapping[str, Any], missing_sources: list[str]) -> list[JsonDict]:
    rows = [_copy_missing_action(row) for row in _as_list(current_v2.get("missing_operator_actions"))]
    for source in missing_sources:
        rows.append(
            {
                "missing_item": f"source_artifact:{source}",
                "present": False,
                "operator_action": "Commit the required prior GateMate/SSQA ledger before v3 ingestion.",
                "source_artifact": source,
                "checked_paths": [source],
            }
        )
    return rows


def _copy_missing_action(row: Any) -> JsonDict:
    mapped = _as_mapping(row)
    copied: JsonDict = {
        "missing_item": str(mapped.get("missing_item") or ""),
        "present": mapped.get("present") is True,
        "operator_action": str(mapped.get("operator_action") or ""),
        "source_artifact": str(mapped.get("source_artifact") or ""),
        "checked_paths": _as_text_list(mapped.get("checked_paths")),
    }
    missing_fields = _as_text_list(mapped.get("missing_required_fields"))
    if missing_fields:
        copied["missing_required_fields"] = missing_fields
    return copied


def _operator_ready_artifacts(current_v2: Mapping[str, Any], v3_ready: bool) -> list[JsonDict]:
    if not v3_ready:
        return []
    return [dict(row) for row in _as_list(current_v2.get("operator_ready_artifacts"))]


def _allowed_next_scope(v3_ready: bool, gatemate_allowed: bool, ssqa_allowed: bool) -> str:
    if not v3_ready:
        return "blocked: required .288 or v2 GateMate/SSQA evidence ledger is missing"
    if not gatemate_allowed:
        return (
            "blocked: operator must commit authoritative GateMate Pin_out/CCF binding, "
            "host reader command, expected transcript, opened safety limits, and "
            "host-visible smoke evidence"
        )
    if not ssqa_allowed:
        return (
            "operator_allowed: run only the gated GateMate host-visible flash-smoke scope; "
            "no timing or speedup claim is authorized by this ingestion"
        )
    return (
        "operator_allowed: run only the gated SSQA readback-scope experiment using "
        "committed host-visible smoke evidence; no timing or speedup claim is "
        "authorized by this ingestion"
    )


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("operator_evidence_ingestion_v3_ready") is not True:
        missing = ", ".join(_as_text_list(artifact.get("missing_source_artifacts")))
        return f"blocked_precondition: missing GateMate/SSQA source artifact(s): {missing}"
    gate = str(artifact.get("gatemate_rerun_allowed")).lower()
    ssqa = str(artifact.get("ssqa_readback_allowed")).lower()
    return (
        "complete: "
        "operator_evidence_ingestion_v3_ready=true; "
        f"gatemate_rerun_allowed={gate}; ssqa_readback_allowed={ssqa}; "
        "hardware_commands_run=0; speedup_claim_made=false"
    )


def _coerce_rel_path(value: Any) -> Path:
    text = str(value or "").strip()
    return Path(text) if text else Path("")


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []
