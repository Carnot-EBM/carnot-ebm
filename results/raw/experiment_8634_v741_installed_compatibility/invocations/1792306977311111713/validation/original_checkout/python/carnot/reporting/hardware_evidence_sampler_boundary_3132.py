"""Build the Exp 3132 hardware evidence and sampler boundary ledger.

Spec refs: REQ-HW-095, SCENARIO-HW-095.

This module is an evidence-ingestion pass, not a hardware run. It inventories
checked-in artifacts and transcripts so downstream reports can distinguish
CPU-only sampler evidence, authenticated historical board evidence, and missing
operator-visible proof without probing boards or promoting speedup claims.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.291"
SCHEMA = "carnot.hardware_evidence_sampler_boundary.v5"
ARTIFACT = "experiment_3132_hardware_evidence_sampler_boundary_v5"
OUTPUT_REL_PATH = Path("results/experiment_3132_hardware_evidence_sampler_boundary_v5.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3132_hardware_evidence_sampler_boundary_v5.py"

EXP3118_CLUT_REL_PATH = Path("results/experiment_3118_clut_sampler_backend_integration_boundary_v2.json")
EXP3119_GATEMATE_SSQA_REL_PATH = Path(
    "results/experiment_3119_gatemate_ssqa_operator_evidence_ingestion_v4.json"
)
HARDWARE_WISHLIST_REL_PATH = Path("research-hardware-wishlist.md")
OPS_STATUS_REL_PATH = Path("ops/status.md")
OPS_CHANGELOG_REL_PATH = Path("ops/changelog.md")
GATEMATE_CCF_REL_PATH = Path("hardware/gatemate/ising_n16_gatemate.ccf")

KV260_LATENCY_REL_PATH = Path("results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json")
KV260_LATENCY_TRANSCRIPT_REL_PATH = Path("results/experiment_2898_kv260_transcript.log")
KV260_CLAIM_BOUNDARY_REL_PATH = Path("results/experiment_2913_kv260_hardware_cpu_claim_boundary_v1.json")
KV260_MMD_REL_PATH = Path("results/experiment_2938_kv260_mmd_vs_cpu_sequential_gibbs_v1.json")
KV260_MMD_TRANSCRIPT_REL_PATH = Path("results/experiment_2938_kv260_mmd_transcript.log")
KV260_N_SCALING_REL_PATH = Path("results/experiment_2942_kv260_continuation_n_scaling_v1.json")
KV260_N_SCALING_TRANSCRIPT_REL_PATH = Path("results/experiment_2942_kv260_n_scaling_transcript.log")

POLARFIRE_DISPATCH_REL_PATH = Path("results/experiment_2900_polarfire_carnot_dispatch_smoke_v1.json")
POLARFIRE_DISPATCH_TRANSCRIPT_REL_PATH = Path("results/experiment_2900_polarfire_transcript_v1.json")
POLARFIRE_500_REL_PATH = Path("results/experiment_2941_polarfire_continuation_v1.json")
POLARFIRE_500_TRANSCRIPT_REL_PATH = Path("results/experiment_2941_polarfire_transcript_v1.json")
POLARFIRE_1000_REL_PATH = Path("results/experiment_2958_polarfire_1000_clause_scorer_v2.json")
POLARFIRE_1000_TRANSCRIPT_REL_PATH = Path("results/experiment_2958_polarfire_1000_clause_transcript_v2.json")

THRML_PORTABILITY_REL_PATH = Path("results/experiment_2883_thrml_sampler_portability_smoke_v2.json")
THRML_IMPORT_REL_PATH = Path("results/experiment_2901_thrml_local_import_repair_v1.json")
THRML_PARITY_REL_PATH = Path("results/experiment_2916_thrml_kv260_sampler_parity_v1.json")

POST_EXP3119_SCAN_PATHS = (
    Path("results/experiment_3122_archive_v290_activate_v291.json"),
    Path("results/experiment_3123_sota_cache_preconditions_manifest_v2.json"),
    Path("results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json"),
    Path("results/experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.json"),
    Path("results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"),
    Path("results/experiment_3127_multi_turn_monitored_repair_ladder_v1.json"),
    Path("results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json"),
    Path("results/experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.json"),
    Path("results/experiment_3130_arm_ebt_energy_budget_sidecar_diagnostic_v2.json"),
    Path("results/experiment_3131_kan_pwa_milp_verifier_abstraction_audit_v1.json"),
)
POST_EXP3119_EVIDENCE_TOKENS = (
    "host_visible_transcript_path",
    "gatemate_host_visible_smoke_passed",
    "readback_hash",
    "Pin_out",
    "per_sample_latency_s",
    "flash_command",
)

SOURCE_SPECS: tuple[tuple[str, Path, bool], ...] = (
    ("exp3118_clut_sampler_boundary", EXP3118_CLUT_REL_PATH, True),
    ("exp3119_gatemate_ssqa_operator_evidence", EXP3119_GATEMATE_SSQA_REL_PATH, True),
    ("hardware_wishlist", HARDWARE_WISHLIST_REL_PATH, False),
    ("ops_status", OPS_STATUS_REL_PATH, False),
    ("ops_changelog", OPS_CHANGELOG_REL_PATH, False),
    ("gatemate_ccf", GATEMATE_CCF_REL_PATH, False),
    ("kv260_latency_artifact", KV260_LATENCY_REL_PATH, False),
    ("kv260_latency_transcript", KV260_LATENCY_TRANSCRIPT_REL_PATH, False),
    ("kv260_claim_boundary", KV260_CLAIM_BOUNDARY_REL_PATH, False),
    ("kv260_mmd_artifact", KV260_MMD_REL_PATH, False),
    ("kv260_mmd_transcript", KV260_MMD_TRANSCRIPT_REL_PATH, False),
    ("kv260_n_scaling_artifact", KV260_N_SCALING_REL_PATH, False),
    ("kv260_n_scaling_transcript", KV260_N_SCALING_TRANSCRIPT_REL_PATH, False),
    ("polarfire_dispatch_artifact", POLARFIRE_DISPATCH_REL_PATH, False),
    ("polarfire_dispatch_transcript", POLARFIRE_DISPATCH_TRANSCRIPT_REL_PATH, False),
    ("polarfire_500_clause_artifact", POLARFIRE_500_REL_PATH, False),
    ("polarfire_500_clause_transcript", POLARFIRE_500_TRANSCRIPT_REL_PATH, False),
    ("polarfire_1000_clause_artifact", POLARFIRE_1000_REL_PATH, False),
    ("polarfire_1000_clause_transcript", POLARFIRE_1000_TRANSCRIPT_REL_PATH, False),
    ("thrml_portability_smoke", THRML_PORTABILITY_REL_PATH, False),
    ("thrml_import_repair", THRML_IMPORT_REL_PATH, False),
    ("thrml_kv260_simulator_parity", THRML_PARITY_REL_PATH, False),
)

INFERENCE_SUBSTRATE = {
    "kind": "hardware_evidence_sampler_boundary_v5",
    "source": "checked_in_local_artifacts",
    "local_repo_only": True,
    "executes_hardware": False,
    "hardware_readback_attempted": False,
    "board_flash_attempted": False,
    "synthesis_or_pnr_run": False,
    "executes_models": False,
    "no_live_model_inference": True,
    "hardware_commands_run": [],
}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and return empty evidence when it is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash a local evidence file when it exists."""

    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-HW-095: build the evidence-only hardware/sampler ledger."""

    root_path = Path(root)
    clut = read_json_object(root_path / EXP3118_CLUT_REL_PATH)
    gatemate = read_json_object(root_path / EXP3119_GATEMATE_SSQA_REL_PATH)
    kv260_latency = read_json_object(root_path / KV260_LATENCY_REL_PATH)
    kv260_claim = read_json_object(root_path / KV260_CLAIM_BOUNDARY_REL_PATH)
    polarfire_1000 = read_json_object(root_path / POLARFIRE_1000_REL_PATH)
    thrml_parity = read_json_object(root_path / THRML_PARITY_REL_PATH)
    source_artifacts = [_source_artifact(root_path, role, path, required) for role, path, required in SOURCE_SPECS]
    missing_required = [
        str(row["path"])
        for row in source_artifacts
        if row["required"] is True and row["readable"] is not True
    ]
    clut_boundary = _clut_sampler_boundary(clut)
    gatemate_complete = _gatemate_complete(gatemate)
    ssqa_ready = gatemate.get("ssqa_readback_allowed") is True
    kv260_status, kv260_details = _kv260_status(root_path, kv260_latency, kv260_claim)
    polarfire_status, polarfire_details = _polarfire_status(root_path, polarfire_1000)
    thrml_tsu_claim_allowed = _thrml_tsu_claim_allowed(thrml_parity)
    decisions = {
        "gatemate": "authenticated hardware evidence" if gatemate_complete else "blocked",
        "ssqa": "authenticated hardware evidence" if ssqa_ready else "blocked",
        "kv260": "authenticated hardware evidence" if kv260_details["authenticated"] else "blocked",
        "polarfire": "authenticated hardware evidence" if polarfire_details["authenticated"] else "blocked",
        "thrml_tsu": "authenticated hardware evidence" if thrml_tsu_claim_allowed else "out-of-scope",
        "clut": str(clut_boundary["decision"]),
    }
    ready = not missing_required
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "hardware_evidence_sampler_boundary_v5_ready": ready,
        "hardware_commands_run": [],
        "gatemate_evidence_complete": gatemate_complete,
        "ssqa_readback_ready": ssqa_ready,
        "kv260_evidence_status": kv260_status,
        "polarfire_evidence_status": polarfire_status,
        "thrml_tsu_claim_allowed": thrml_tsu_claim_allowed,
        "clut_sampler_boundary": clut_boundary,
        "missing_operator_evidence": _missing_operator_evidence(
            gatemate,
            clut_boundary,
            kv260_details,
            polarfire_details,
            thrml_tsu_claim_allowed,
        ),
        "speedup_claim_allowed": False,
        "source_artifacts": source_artifacts,
        "missing_required_source_artifacts": missing_required,
        "evidence_inventory": {
            "gatemate": _gatemate_inventory(gatemate),
            "ssqa": _ssqa_inventory(gatemate),
            "kv260": kv260_details,
            "polarfire": polarfire_details,
            "thrml_tsu": _thrml_inventory(thrml_parity, thrml_tsu_claim_allowed),
            "clut": clut_boundary,
        },
        "sampler_boundary_decisions": decisions,
        "post_exp3119_evidence_scan": _post_exp3119_evidence_scan(root_path),
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
    """Persist the Exp 3132 result JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_artifact(root: Path, role: str, rel_path: Path, required: bool) -> JsonDict:
    path = root / rel_path
    present = path.is_file()
    source_type = "json" if rel_path.suffix == ".json" else "text"
    readable = present and (source_type != "json" or bool(read_json_object(path)))
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "required": required,
        "present": present,
        "readable": readable,
        "source_type": source_type,
        "sha256": sha256_file(path),
    }


def _clut_sampler_boundary(clut: Mapping[str, Any]) -> JsonDict:
    substrate = dict(clut.get("inference_substrate", {})) if isinstance(clut.get("inference_substrate"), Mapping) else {}
    ready = clut.get("clut_backend_integration_boundary_v2_ready") is True
    hardware_claim_allowed = clut.get("hardware_claim_made") is True and substrate.get("executes_hardware") is True
    decision = "authenticated hardware evidence" if hardware_claim_allowed else "CPU simulation" if ready else "blocked"
    return {
        "decision": decision,
        "cpu_only": substrate.get("cpu_only") is True,
        "ready": ready,
        "hardware_claim_allowed": hardware_claim_allowed,
        "hardware_commands_run": list(clut.get("hardware_commands_run", [])),
        "source_artifact": EXP3118_CLUT_REL_PATH.as_posix(),
        "flagged_adversarial": clut.get("flagged_adversarial") is True,
    }


def _gatemate_complete(gatemate: Mapping[str, Any]) -> bool:
    return gatemate.get("operator_evidence_ingestion_v4_ready") is True and gatemate.get("gatemate_rerun_allowed") is True


def _kv260_status(root: Path, latency: Mapping[str, Any], claim: Mapping[str, Any]) -> tuple[str, JsonDict]:
    transcript = _artifact_path(latency, "board_transcript_path", KV260_LATENCY_TRANSCRIPT_REL_PATH)
    transcript_present = (root / transcript).is_file()
    authenticated = latency.get("inference_substrate") == "hardware_smoke" and transcript_present
    status = (
        "authenticated_historical_kv260_hardware_evidence_present_no_new_execution"
        if authenticated
        else "blocked_missing_authenticated_kv260_hardware_transcript"
    )
    return status, {
        "authenticated": authenticated,
        "decision": "authenticated hardware evidence" if authenticated else "blocked",
        "present_evidence": [KV260_LATENCY_REL_PATH.as_posix(), transcript.as_posix()] if authenticated else [],
        "missing_fields": [] if authenticated else ["hardware_smoke_artifact", "board_transcript_path"],
        "historical_speedup_claim_seen": claim.get("speedup_claim_made") is True,
        "historical_speedup_not_promoted_by_exp3132": True,
    }


def _polarfire_status(root: Path, polarfire: Mapping[str, Any]) -> tuple[str, JsonDict]:
    transcript_paths = [Path(str(path)) for path in polarfire.get("transcript_paths", [])]
    if not transcript_paths:
        transcript_paths = [POLARFIRE_1000_TRANSCRIPT_REL_PATH]
    transcripts_present = all((root / path).is_file() for path in transcript_paths)
    authenticated = (
        polarfire.get("polarfire_1000_clause_hash_verified") is True
        and polarfire.get("board_reachable") is True
        and transcripts_present
    )
    status = (
        "authenticated_polarfire_dispatch_hash_evidence_present_no_speedup_claim"
        if authenticated
        else "blocked_missing_polarfire_dispatch_or_readback_transcript"
    )
    return status, {
        "authenticated": authenticated,
        "decision": "authenticated hardware evidence" if authenticated else "blocked",
        "present_evidence": [POLARFIRE_1000_REL_PATH.as_posix(), *[path.as_posix() for path in transcript_paths]]
        if authenticated
        else [],
        "missing_fields": []
        if authenticated
        else ["polarfire_1000_clause_hash_verified", "board_reachable", "transcript_paths"],
        "no_speedup_claim": polarfire.get("no_speedup_claim") is True,
    }


def _thrml_tsu_claim_allowed(thrml: Mapping[str, Any]) -> bool:
    return thrml.get("authenticated_tsu_hardware_evidence") is True or thrml.get("tsu_hardware_claim_allowed") is True


def _missing_operator_evidence(
    gatemate: Mapping[str, Any],
    clut_boundary: Mapping[str, Any],
    kv260: Mapping[str, Any],
    polarfire: Mapping[str, Any],
    thrml_tsu_claim_allowed: bool,
) -> list[JsonDict]:
    missing = [_operator_missing_row(row) for row in gatemate.get("missing_operator_actions", [])]
    if not kv260["authenticated"]:
        missing.append(_simple_missing("kv260:authenticated_board_transcript", "Commit a KV260 board transcript and matching hardware-smoke artifact.", kv260["missing_fields"]))
    if not polarfire["authenticated"]:
        missing.append(_simple_missing("polarfire:dispatch_readback_transcript", "Commit PolarFire dispatch/readback transcript evidence with verified hashes.", polarfire["missing_fields"]))
    if not thrml_tsu_claim_allowed:
        missing.append(_simple_missing("thrml_tsu:authenticated_tsu_hardware_evidence", "Commit authenticated TSU hardware execution evidence before any TSU claim.", ["authenticated_tsu_hardware_evidence"]))
    if clut_boundary.get("hardware_claim_allowed") is not True:
        missing.append(_simple_missing("clut:authenticated_hardware_execution_evidence", "Keep cLUT bounded to CPU until authenticated hardware execution evidence exists.", ["hardware_execution_evidence"]))
    return missing


def _operator_missing_row(row: Any) -> JsonDict:
    mapped = dict(row) if isinstance(row, Mapping) else {}
    item = str(mapped.get("missing_item") or "unknown")
    prefix = "ssqa" if item == "host_visible_smoke_evidence" else "gatemate"
    return {
        "missing_item": f"{prefix}:{item}",
        "operator_action": str(mapped.get("operator_action") or ""),
        "source_artifact": str(mapped.get("source_artifact") or ""),
        "checked_paths": [str(path) for path in mapped.get("checked_paths", [])],
        "missing_required_fields": [str(field) for field in mapped.get("missing_required_fields", [])],
    }


def _simple_missing(item: str, action: str, fields: Any) -> JsonDict:
    return {
        "missing_item": item,
        "operator_action": action,
        "source_artifact": "",
        "checked_paths": [],
        "missing_required_fields": [str(field) for field in fields],
    }


def _gatemate_inventory(gatemate: Mapping[str, Any]) -> JsonDict:
    return {
        "decision": "authenticated hardware evidence" if _gatemate_complete(gatemate) else "blocked",
        "operator_evidence_ingestion_v4_ready": gatemate.get("operator_evidence_ingestion_v4_ready") is True,
        "gatemate_rerun_allowed": gatemate.get("gatemate_rerun_allowed") is True,
        "missing_fields": [
            field
            for row in gatemate.get("missing_operator_actions", [])
            if isinstance(row, Mapping) and row.get("missing_item") != "host_visible_smoke_evidence"
            for field in row.get("missing_required_fields", [])
        ],
    }


def _ssqa_inventory(gatemate: Mapping[str, Any]) -> JsonDict:
    host_visible = dict(gatemate.get("host_visible_smoke_evidence", {})) if isinstance(gatemate.get("host_visible_smoke_evidence"), Mapping) else {}
    return {
        "decision": "authenticated hardware evidence" if gatemate.get("ssqa_readback_allowed") is True else "blocked",
        "ssqa_readback_allowed": gatemate.get("ssqa_readback_allowed") is True,
        "host_visible_smoke_ready": host_visible.get("ready") is True,
        "missing_fields": [
            field
            for row in gatemate.get("missing_operator_actions", [])
            if isinstance(row, Mapping) and row.get("missing_item") == "host_visible_smoke_evidence"
            for field in row.get("missing_required_fields", [])
        ],
    }


def _thrml_inventory(thrml: Mapping[str, Any], allowed: bool) -> JsonDict:
    return {
        "decision": "authenticated hardware evidence" if allowed else "out-of-scope",
        "present_evidence": [THRML_PARITY_REL_PATH.as_posix()] if thrml else [],
        "missing_fields": [] if allowed else ["authenticated_tsu_hardware_evidence"],
        "no_tsu_hardware_claim": thrml.get("no_tsu_hardware_claim") is True,
    }


def _post_exp3119_evidence_scan(root: Path) -> JsonDict:
    scanned = []
    matched_paths = []
    for rel_path in POST_EXP3119_SCAN_PATHS:
        path = root / rel_path
        if path.is_file():
            scanned.append(rel_path.as_posix())
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(token in text for token in POST_EXP3119_EVIDENCE_TOKENS):
                matched_paths.append(rel_path.as_posix())
    return {
        "scanned_paths": scanned,
        "matched_paths": matched_paths,
        "new_operator_evidence_found": bool(matched_paths),
        "evidence_tokens": list(POST_EXP3119_EVIDENCE_TOKENS),
    }


def _artifact_path(payload: Mapping[str, Any], field: str, fallback: Path) -> Path:
    value = str(payload.get(field) or "").strip()
    return Path(value) if value else fallback


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("hardware_evidence_sampler_boundary_v5_ready") is not True:
        missing = ", ".join(str(path) for path in artifact.get("missing_required_source_artifacts", []))
        return f"blocked_precondition: missing required hardware boundary sources: {missing}"
    gate = str(artifact.get("gatemate_evidence_complete")).lower()
    ssqa = str(artifact.get("ssqa_readback_ready")).lower()
    speedup = str(artifact.get("speedup_claim_allowed")).lower()
    return (
        "complete: hardware_evidence_sampler_boundary_v5_ready=true; "
        f"gatemate_evidence_complete={gate}; ssqa_readback_ready={ssqa}; "
        f"speedup_claim_allowed={speedup}; hardware_commands_run=0"
    )
