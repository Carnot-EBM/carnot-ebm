"""Build the Exp 3160 hardware sampler evidence boundary ledger.

Spec refs: REQ-HW-097, SCENARIO-HW-097.

This module is deliberately an evidence-ingestion pass. It reads checked-in
JSON/text artifacts, classifies whether evidence is local, operator supplied,
wishlist intent, ops documentation, or public architecture context, and then
writes a no-claim boundary. It does not probe GPUs, run board commands, flash
devices, synthesize designs, read hardware, or run live inference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.293"
SCHEMA = "carnot.hardware_sampler_evidence_boundary.v7"
ARTIFACT = "experiment_3160_hardware_sampler_evidence_boundary_v7"
OUTPUT_REL_PATH = Path("results/experiment_3160_hardware_sampler_evidence_boundary_v7.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3160_hardware_sampler_evidence_boundary_v7.py"

EXP3146_BOUNDARY_REL_PATH = Path(
    "results/experiment_3146_hardware_sampler_evidence_boundary_v6.json"
)
CUDA_RUNTIME_REL_PATH = Path("results/experiment_2862_sota_runtime_cache_offload_resolver_v3.json")
CUDA_CAPSTONE_REL_PATH = Path("results/experiment_2872_capstone_v271.json")
GATEMATE_EVIDENCE_REL_PATH = Path(
    "results/experiment_3119_gatemate_ssqa_operator_evidence_ingestion_v4.json"
)
HARDWARE_WISHLIST_REL_PATH = Path("research-hardware-wishlist.md")
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")
OPS_STATUS_REL_PATH = Path("ops/status.md")
OPS_CHANGELOG_REL_PATH = Path("ops/changelog.md")

KV260_LATENCY_REL_PATH = Path(
    "results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
)
KV260_LATENCY_TRANSCRIPT_REL_PATH = Path("results/experiment_2898_kv260_transcript.log")
KV260_CLAIM_BOUNDARY_REL_PATH = Path(
    "results/experiment_2913_kv260_hardware_cpu_claim_boundary_v1.json"
)
POLARFIRE_1000_REL_PATH = Path("results/experiment_2958_polarfire_1000_clause_scorer_v2.json")
POLARFIRE_1000_TRANSCRIPT_REL_PATH = Path(
    "results/experiment_2958_polarfire_1000_clause_transcript_v2.json"
)
THRML_PARITY_REL_PATH = Path("results/experiment_2916_thrml_kv260_sampler_parity_v1.json")
KONA_BOUNDARY_REL_PATH = Path(
    "results/experiment_1362_publication_hold_ebt_arm_kona_claim_boundary.json"
)

SPEEDUP_REQUIRED_FIELDS = (
    "command_transcript",
    "board_or_device_identity",
    "baseline",
    "artifact_checksum",
    "workload",
    "reproducibility_notes",
)

SOURCE_SPECS: tuple[tuple[str, Path, bool, str, str], ...] = (
    (
        "exp3146_hardware_sampler_boundary_v6",
        EXP3146_BOUNDARY_REL_PATH,
        True,
        "checked_in_local_artifact",
        "prior boundary; not fresh hardware evidence",
    ),
    (
        "cuda_runtime_exp2862",
        CUDA_RUNTIME_REL_PATH,
        True,
        "checked_in_local_artifact",
        "runtime readiness only; not sampler speedup evidence",
    ),
    (
        "cuda_capstone_flag_boundary",
        CUDA_CAPSTONE_REL_PATH,
        False,
        "checked_in_local_artifact",
        "adversarial flag carry-forward for CUDA runtime evidence",
    ),
    (
        "gatemate_operator_evidence_v4",
        GATEMATE_EVIDENCE_REL_PATH,
        True,
        "local_operator_evidence",
        "operator action ledger; missing fields remain blocking",
    ),
    (
        "kv260_latency_artifact",
        KV260_LATENCY_REL_PATH,
        False,
        "local_operator_evidence",
        "historical board timing evidence scoped to recorded workload",
    ),
    (
        "kv260_latency_transcript",
        KV260_LATENCY_TRANSCRIPT_REL_PATH,
        False,
        "local_operator_evidence",
        "historical command transcript; not a fresh run",
    ),
    (
        "kv260_claim_boundary",
        KV260_CLAIM_BOUNDARY_REL_PATH,
        False,
        "checked_in_local_artifact",
        "historical claim boundary; speedup not promoted by v7",
    ),
    (
        "polarfire_1000_clause_artifact",
        POLARFIRE_1000_REL_PATH,
        False,
        "local_operator_evidence",
        "historical dispatch/readback evidence scoped to hash workload",
    ),
    (
        "polarfire_1000_clause_transcript",
        POLARFIRE_1000_TRANSCRIPT_REL_PATH,
        False,
        "local_operator_evidence",
        "historical transcript; not a fresh run",
    ),
    (
        "thrml_kv260_simulator_parity",
        THRML_PARITY_REL_PATH,
        False,
        "checked_in_local_artifact",
        "simulator parity only; not TSU/XTR/Z1 execution",
    ),
    (
        "kona_claim_boundary",
        KONA_BOUNDARY_REL_PATH,
        False,
        "checked_in_local_artifact",
        "local claim boundary for external Kona context",
    ),
    (
        "hardware_wishlist",
        HARDWARE_WISHLIST_REL_PATH,
        True,
        "wishlist_intent",
        "planning intent; never execution evidence",
    ),
    (
        "research_references_public_architecture_pages",
        RESEARCH_REFERENCES_REL_PATH,
        True,
        "public_architecture_reference",
        "vendor/project pages and related work; never local execution evidence",
    ),
    (
        "ops_status",
        OPS_STATUS_REL_PATH,
        False,
        "ops_documentation",
        "operational status context",
    ),
    (
        "ops_changelog",
        OPS_CHANGELOG_REL_PATH,
        False,
        "ops_documentation",
        "operational changelog context",
    ),
)

INFERENCE_SUBSTRATE = {
    "kind": "hardware_sampler_evidence_boundary_v7",
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
    """Read a JSON object and fail closed when evidence is malformed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash a checked-in evidence source when the file exists."""

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def build_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-HW-097: build the evidence-only v7 hardware sampler boundary."""

    root_path = Path(root)
    cuda_runtime = read_json_object(root_path / CUDA_RUNTIME_REL_PATH)
    cuda_capstone = read_json_object(root_path / CUDA_CAPSTONE_REL_PATH)
    gatemate = read_json_object(root_path / GATEMATE_EVIDENCE_REL_PATH)
    kv260_latency = read_json_object(root_path / KV260_LATENCY_REL_PATH)
    polarfire_1000 = read_json_object(root_path / POLARFIRE_1000_REL_PATH)
    thrml = read_json_object(root_path / THRML_PARITY_REL_PATH)
    kona = read_json_object(root_path / KONA_BOUNDARY_REL_PATH)

    source_artifacts = [_source_artifact(root_path, spec) for spec in SOURCE_SPECS]
    missing_required_sources = [
        row["path"]
        for row in source_artifacts
        if row["required"] is True and row["readable"] is not True
    ]
    cuda_status = _cuda_status(cuda_runtime, cuda_capstone)
    kv260_status = _kv260_status(root_path, kv260_latency)
    gatemate_status = _gatemate_status(gatemate)
    polarfire_status = _polarfire_status(root_path, polarfire_1000)
    extropic_thrml_status = _extropic_thrml_status(thrml)
    kona_status = _kona_status(kona)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "hardware_sampler_evidence_boundary_v7_ready": not missing_required_sources,
        "authenticated_speedup_claim_allowed": False,
        "no_hardware_commands_run": True,
        "hardware_commands_run": [],
        "evidence_sources": _evidence_sources(source_artifacts),
        "missing_operator_evidence": _missing_operator_evidence(
            gatemate,
            cuda_status,
            kv260_status,
            polarfire_status,
            extropic_thrml_status,
            kona_status,
        ),
        "cuda_status": cuda_status,
        "kv260_status": kv260_status,
        "gatemate_status": gatemate_status,
        "polarfire_status": polarfire_status,
        "extropic_thrml_status": extropic_thrml_status,
        "kona_status": kona_status,
        "source_artifacts": source_artifacts,
        "missing_required_source_artifacts": missing_required_sources,
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "evidence_inventory": {
            "cuda": {
                "runtime_ready": cuda_status.startswith("runtime_ready"),
                "flagged_adversarial": _cuda_flagged(cuda_runtime, cuda_capstone),
                "source_artifacts": [
                    CUDA_RUNTIME_REL_PATH.as_posix(),
                    CUDA_CAPSTONE_REL_PATH.as_posix(),
                ],
            },
            "kv260": {
                "status": kv260_status,
                "source_artifacts": [
                    KV260_LATENCY_REL_PATH.as_posix(),
                    KV260_LATENCY_TRANSCRIPT_REL_PATH.as_posix(),
                ],
            },
            "gatemate": {
                "status": gatemate_status,
                "missing_operator_actions": len(gatemate.get("missing_operator_actions", [])),
            },
            "polarfire": {
                "status": polarfire_status,
                "source_artifacts": [
                    POLARFIRE_1000_REL_PATH.as_posix(),
                    POLARFIRE_1000_TRANSCRIPT_REL_PATH.as_posix(),
                ],
            },
            "extropic_thrml": {
                "status": extropic_thrml_status,
                "architecture_reference_only": "architecture_reference_only"
                in extropic_thrml_status,
            },
            "kona_aleph": {
                "status": kona_status,
                "architecture_reference_only": "architecture_reference_only" in kona_status,
            },
        },
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT, *, output_path: Path | str = OUTPUT_REL_PATH
) -> Path:
    """Persist the Exp 3160 result JSON."""

    root_path = Path(root)
    out_path = root_path / Path(output_path)
    artifact = build_artifact(root_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_artifact(root: Path, spec: tuple[str, Path, bool, str, str]) -> JsonDict:
    role, rel_path, required, evidence_class, claim_use = spec
    path = root / rel_path
    source_type = "json" if rel_path.suffix == ".json" else "text"
    present = path.is_file()
    readable = present and (source_type != "json" or bool(read_json_object(path)))
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "required": required,
        "present": present,
        "readable": readable,
        "source_type": source_type,
        "evidence_class": evidence_class,
        "claim_use": claim_use,
        "sha256": sha256_file(path),
    }


def _evidence_sources(source_artifacts: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "role": row["role"],
            "path": row["path"],
            "evidence_class": row["evidence_class"],
            "claim_use": row["claim_use"],
            "present": row["present"],
            "readable": row["readable"],
        }
        for row in source_artifacts
    ]


def _cuda_status(cuda_runtime: Mapping[str, Any], cuda_capstone: Mapping[str, Any]) -> str:
    ready = (
        cuda_runtime.get("sota_runtime_ready_v3") is True
        and cuda_runtime.get("llama_cpp_gpu_offload_verified") is True
        and int(cuda_runtime.get("usable_response_count") or 0) > 0
    )
    if not ready:
        return "blocked_cuda_runtime_evidence_missing"
    suffix = "_flagged_adversarial" if _cuda_flagged(cuda_runtime, cuda_capstone) else ""
    return f"runtime_ready_no_sampler_speedup_claim{suffix}"


def _cuda_flagged(cuda_runtime: Mapping[str, Any], cuda_capstone: Mapping[str, Any]) -> bool:
    flagged = cuda_capstone.get("adversarially_flagged_artifacts", [])
    return cuda_runtime.get("flagged_adversarial") is True or "exp2862" in flagged


def _kv260_status(root: Path, kv260_latency: Mapping[str, Any]) -> str:
    transcript = Path(
        str(kv260_latency.get("board_transcript_path") or KV260_LATENCY_TRANSCRIPT_REL_PATH)
    )
    authenticated = (
        kv260_latency.get("inference_substrate") == "hardware_smoke"
        and (root / transcript).is_file()
    )
    return (
        "authenticated_historical_board_evidence_scoped_no_fresh_speedup_claim"
        if authenticated
        else "blocked_missing_authenticated_kv260_transcript"
    )


def _gatemate_status(gatemate: Mapping[str, Any]) -> str:
    complete = (
        gatemate.get("operator_evidence_ingestion_v4_ready") is True
        and gatemate.get("gatemate_rerun_allowed") is True
    )
    return (
        "operator_evidence_complete_no_speedup_claim"
        if complete
        else "blocked_operator_evidence_incomplete_no_speedup_claim"
    )


def _polarfire_status(root: Path, polarfire: Mapping[str, Any]) -> str:
    transcript_paths = [Path(str(path)) for path in polarfire.get("transcript_paths", [])]
    transcripts_present = bool(transcript_paths) and all(
        (root / path).is_file() for path in transcript_paths
    )
    authenticated = (
        polarfire.get("polarfire_1000_clause_hash_verified") is True
        and polarfire.get("board_reachable") is True
        and transcripts_present
    )
    return (
        "authenticated_historical_dispatch_evidence_no_speedup_claim"
        if authenticated
        else "blocked_missing_polarfire_dispatch_or_readback_transcript"
    )


def _extropic_thrml_status(thrml: Mapping[str, Any]) -> str:
    authenticated = (
        thrml.get("authenticated_tsu_hardware_evidence") is True
        or thrml.get("tsu_hardware_claim_allowed") is True
    )
    return (
        "authenticated_extropic_tsu_evidence_no_speedup_claim"
        if authenticated
        else "architecture_reference_only_no_local_tsu_or_xtr_execution"
    )


def _kona_status(kona: Mapping[str, Any]) -> str:
    hardware = (
        dict(kona.get("hardware_evidence_summary", {}))
        if isinstance(kona.get("hardware_evidence_summary"), Mapping)
        else {}
    )
    authenticated = (
        kona.get("authenticated_local_kona_access_or_execution_evidence") is True
        or kona.get("external_dependency_claim_allowed") is True
        or hardware.get("hardware_execution_claim_allowed") is True
    )
    return (
        "authenticated_local_kona_or_aleph_evidence_no_speedup_claim"
        if authenticated
        else "architecture_reference_only_no_local_kona_or_aleph_execution"
    )


def _missing_operator_evidence(
    gatemate: Mapping[str, Any],
    cuda_status: str,
    kv260_status: str,
    polarfire_status: str,
    extropic_thrml_status: str,
    kona_status: str,
) -> list[JsonDict]:
    missing = [
        _simple_missing(
            "authenticated_speedup_claim:complete_local_evidence_bundle",
            "Before any speedup claim, commit a local evidence bundle with transcript, identity, baseline, checksum, workload, and reproducibility notes.",
            SPEEDUP_REQUIRED_FIELDS,
        )
    ]
    missing.extend(
        _operator_missing_row(row) for row in gatemate.get("missing_operator_actions", [])
    )
    if cuda_status.startswith("blocked"):
        missing.append(
            _simple_missing(
                "cuda:usable_runtime_and_speedup_evidence",
                "Commit clean CUDA runtime evidence plus a benchmark transcript before any CUDA speedup claim.",
                ("usable_response", *SPEEDUP_REQUIRED_FIELDS),
            )
        )
    if kv260_status.startswith("blocked"):
        missing.append(
            _simple_missing(
                "kv260:authenticated_board_transcript",
                "Commit a KV260 hardware-smoke artifact and matching board transcript before any KV260 claim.",
                ("hardware_smoke_artifact", "board_transcript_path"),
            )
        )
    if polarfire_status.startswith("blocked"):
        missing.append(
            _simple_missing(
                "polarfire:dispatch_readback_transcript",
                "Commit PolarFire dispatch/readback transcript evidence with verified hashes.",
                ("polarfire_1000_clause_hash_verified", "board_reachable", "transcript_paths"),
            )
        )
    if extropic_thrml_status.startswith("architecture_reference_only"):
        missing.append(
            _simple_missing(
                "extropic_thrml:authenticated_tsu_xtr_z1_execution_evidence",
                "Keep THRML/Extropic bounded to simulator and architecture context until local TSU/XTR/Z1 evidence exists.",
                ("authenticated_tsu_hardware_evidence",),
            )
        )
    if kona_status.startswith("architecture_reference_only"):
        missing.append(
            _simple_missing(
                "kona_aleph:authenticated_local_kona_or_aleph_execution_evidence",
                "Keep Kona/Aleph bounded to architecture context until authenticated local execution evidence exists.",
                ("authenticated_local_kona_or_aleph_execution_evidence",),
            )
        )
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
        "missing_required_fields": [
            str(field) for field in mapped.get("missing_required_fields", [])
        ],
    }


def _simple_missing(item: str, action: str, fields: Any) -> JsonDict:
    return {
        "missing_item": item,
        "operator_action": action,
        "source_artifact": "",
        "checked_paths": [],
        "missing_required_fields": [str(field) for field in fields],
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("hardware_sampler_evidence_boundary_v7_ready") is not True:
        missing = ", ".join(
            str(path) for path in artifact.get("missing_required_source_artifacts", [])
        )
        return f"blocked_precondition: missing required v7 hardware sampler boundary sources: {missing}"
    return (
        "complete: hardware_sampler_evidence_boundary_v7_ready=true; "
        "authenticated_speedup_claim_allowed=false; no_hardware_commands_run=true; "
        f"cuda_status={artifact.get('cuda_status')}; gatemate_status={artifact.get('gatemate_status')}; "
        "extropic_thrml_status=architecture_bounded; kona_status=architecture_bounded"
    )
