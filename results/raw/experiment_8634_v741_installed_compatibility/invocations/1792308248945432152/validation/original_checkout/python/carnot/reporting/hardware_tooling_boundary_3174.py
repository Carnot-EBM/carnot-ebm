"""Build the Exp 3174 hardware/tooling boundary ledger.

Spec refs: REQ-HW-098, SCENARIO-HW-098.

This module is a claim-boundary ledger, not a benchmark. It reads checked-in
evidence and local Python package metadata so the project can distinguish three
different facts that are easy to blur together: public ecosystem context, local
tool availability, and authenticated performance evidence. Import availability
means only "this Python environment can see a package"; it does not mean TSU,
Kona, FPGA, CUDA, or sampler performance exists.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
MILESTONE = "2026.05.294"
SCHEMA = "carnot.hardware_tooling_boundary.v8"
ARTIFACT = "experiment_3174_hardware_tooling_boundary_v8"
OUTPUT_REL_PATH = Path("results/experiment_3174_hardware_tooling_boundary_v8.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3174_hardware_tooling_boundary_v8.py"

EXP3160_BOUNDARY_REL_PATH = Path(
    "results/experiment_3160_hardware_sampler_evidence_boundary_v7.json"
)
EXP3146_BOUNDARY_REL_PATH = Path(
    "results/experiment_3146_hardware_sampler_evidence_boundary_v6.json"
)
CUDA_RUNTIME_REL_PATH = Path("results/experiment_2862_sota_runtime_cache_offload_resolver_v3.json")
GATEMATE_EVIDENCE_REL_PATH = Path(
    "results/experiment_3119_gatemate_ssqa_operator_evidence_ingestion_v4.json"
)
KV260_LATENCY_REL_PATH = Path(
    "results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
)
KV260_LATENCY_TRANSCRIPT_REL_PATH = Path("results/experiment_2898_kv260_transcript.log")
POLARFIRE_1000_REL_PATH = Path("results/experiment_2958_polarfire_1000_clause_scorer_v2.json")
POLARFIRE_1000_TRANSCRIPT_REL_PATH = Path(
    "results/experiment_2958_polarfire_1000_clause_transcript_v2.json"
)
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")
HARDWARE_WISHLIST_REL_PATH = Path("research-hardware-wishlist.md")
OPS_STATUS_REL_PATH = Path("ops/status.md")

SPEEDUP_REQUIRED_FIELDS = (
    "command_transcript",
    "board_or_device_identity",
    "baseline",
    "artifact_checksum",
    "workload",
    "reproducibility_notes",
)
TOOLING_PROBES = (
    ("thrml", "thrml"),
    ("xgrammar", "xgrammar"),
    ("llguidance", "llguidance"),
)
PUBLIC_ECOSYSTEM_REFERENCES: tuple[tuple[str, str, str], ...] = (
    (
        "extropic_software",
        "https://extropic.ai/software",
        "public THRML/TSU ecosystem context; not local hardware evidence",
    ),
    (
        "extropic_thrml_github",
        "https://github.com/extropic-ai/thrml",
        "public THRML source context; not TSU/XTR execution evidence",
    ),
    (
        "thrml_docs",
        "https://docs.thrml.ai/en/latest/examples/00_probabilistic_computing/",
        "public simulator documentation; not local hardware access",
    ),
    (
        "xgrammar_github",
        "https://github.com/mlc-ai/xgrammar",
        "public structured-output tooling context; not sampler speedup evidence",
    ),
    (
        "llguidance_github",
        "https://github.com/guidance-ai/llguidance",
        "public structured-output tooling context; not sampler speedup evidence",
    ),
    (
        "kona_ebms",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "public Kona/Aleph architecture context; not local interoperability evidence",
    ),
    (
        "kona_reasoning_blog",
        "https://logicalintelligence.com/blog/energy-based-models-for-reasoning",
        "public EBRM reasoning context; not local Kona execution evidence",
    ),
)
LOCAL_SOURCE_SPECS: tuple[tuple[str, Path, bool, str, str], ...] = (
    (
        "exp3160_hardware_sampler_boundary_v7",
        EXP3160_BOUNDARY_REL_PATH,
        True,
        "checked_in_local_artifact",
        "prior boundary; not fresh hardware evidence",
    ),
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
        False,
        "checked_in_local_artifact",
        "local GGUF/CUDA runtime readiness only; not sampler speedup evidence",
    ),
    (
        "gatemate_operator_evidence_v4",
        GATEMATE_EVIDENCE_REL_PATH,
        False,
        "local_operator_evidence",
        "operator-visible GateMate evidence ledger; incomplete fields remain blocking",
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
        "historical command transcript; not a fresh Exp 3174 run",
    ),
    (
        "polarfire_1000_clause_artifact",
        POLARFIRE_1000_REL_PATH,
        False,
        "local_operator_evidence",
        "historical dispatch evidence scoped to hash-verified workload",
    ),
    (
        "polarfire_1000_clause_transcript",
        POLARFIRE_1000_TRANSCRIPT_REL_PATH,
        False,
        "local_operator_evidence",
        "historical workload transcript; not a fresh Exp 3174 run",
    ),
    (
        "research_references_public_pages",
        RESEARCH_REFERENCES_REL_PATH,
        True,
        "public_ecosystem_reference",
        "public pages are context only; never local execution evidence",
    ),
    (
        "hardware_wishlist",
        HARDWARE_WISHLIST_REL_PATH,
        True,
        "wishlist_intent",
        "planning intent and boundary text; never performance evidence",
    ),
    (
        "ops_status",
        OPS_STATUS_REL_PATH,
        False,
        "ops_documentation",
        "operational status context",
    ),
)
INFERENCE_SUBSTRATE = {
    "kind": "hardware_tooling_boundary_v8",
    "source": "checked_in_local_artifacts_and_local_import_metadata",
    "local_repo_only": True,
    "executes_hardware": False,
    "hardware_readback_attempted": False,
    "board_flash_attempted": False,
    "synthesis_or_pnr_run": False,
    "executes_models": False,
    "no_live_model_inference": True,
    "remote_hardware_called": False,
    "installs_packages": False,
    "hardware_commands_run": [],
}


def read_json_object(path: Path) -> JsonDict:
    """Read trusted JSON evidence and fail closed on missing or malformed data."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a file checksum when a checked-in evidence source exists."""

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def probe_local_tooling(
    probes: tuple[tuple[str, str], ...] = TOOLING_PROBES,
    *,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
    version: Callable[[str], str] = metadata.version,
) -> JsonDict:
    """REQ-HW-098: check import metadata without installing or benchmarking."""

    checks: JsonDict = {}
    for module_name, distribution_name in probes:
        available = find_spec(module_name) is not None
        package_version = None
        if available:
            try:
                package_version = version(distribution_name)
            except metadata.PackageNotFoundError:
                package_version = None
        checks[module_name] = {
            "module": module_name,
            "distribution": distribution_name,
            "available": available,
            "version": package_version,
            "check_method": "importlib.util.find_spec + importlib.metadata.version",
            "installs_packages": False,
            "hardware_commands_run": [],
            "hardware_performance_evidence": False,
        }
    return checks


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    local_tooling_checks: Mapping[str, Mapping[str, Any]] | None = None,
) -> JsonDict:
    """REQ-HW-098: build the evidence-only hardware/tooling boundary."""

    root_path = Path(root)
    v7 = read_json_object(root_path / EXP3160_BOUNDARY_REL_PATH)
    v6 = read_json_object(root_path / EXP3146_BOUNDARY_REL_PATH)
    cuda_runtime = read_json_object(root_path / CUDA_RUNTIME_REL_PATH)
    gatemate = read_json_object(root_path / GATEMATE_EVIDENCE_REL_PATH)
    kv260_latency = read_json_object(root_path / KV260_LATENCY_REL_PATH)
    polarfire = read_json_object(root_path / POLARFIRE_1000_REL_PATH)
    research_text = _read_text(root_path / RESEARCH_REFERENCES_REL_PATH)
    tooling_checks = (
        {key: dict(value) for key, value in local_tooling_checks.items()}
        if local_tooling_checks is not None
        else probe_local_tooling()
    )

    source_artifacts = [
        _local_source_artifact(root_path, spec) for spec in LOCAL_SOURCE_SPECS
    ]
    source_artifacts.extend(_public_source_rows(research_text))
    source_artifacts.extend(_tooling_source_rows(tooling_checks))
    missing_required_sources = [
        row["path"]
        for row in source_artifacts
        if row.get("required") is True and row.get("readable") is not True
    ]

    cuda_status = _cuda_status(v7, cuda_runtime)
    kv260_status = _kv260_status(root_path, kv260_latency)
    gatemate_status = _gatemate_status(v7, gatemate)
    polarfire_status = _polarfire_status(root_path, polarfire)
    extropic_thrml_status = _prior_or_default(
        v7,
        "extropic_thrml_status",
        "architecture_reference_only_no_local_tsu_or_xtr_execution",
    )
    kona_status = _prior_or_default(
        v7,
        "kona_status",
        "architecture_reference_only_no_local_kona_or_aleph_execution",
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "hardware_tooling_boundary_v8_ready": not missing_required_sources,
        "authenticated_speedup_claim_allowed": False,
        "hardware_commands_run": [],
        "local_tooling_checks": tooling_checks,
        "cuda_status": cuda_status,
        "kv260_status": kv260_status,
        "gatemate_status": gatemate_status,
        "polarfire_status": polarfire_status,
        "extropic_thrml_status": extropic_thrml_status,
        "kona_status": kona_status,
        "speedup_claim_made": False,
        "claim_boundaries": _claim_boundaries(
            cuda_status,
            kv260_status,
            gatemate_status,
            polarfire_status,
            extropic_thrml_status,
            kona_status,
        ),
        "evidence_partitions": {
            "public_ecosystem_references": _public_reference_partition(research_text),
            "local_tooling_checks": tooling_checks,
            "authenticated_performance_evidence": _authenticated_performance_evidence(
                kv260_status, polarfire_status
            ),
        },
        "source_artifacts": source_artifacts,
        "missing_required_source_artifacts": missing_required_sources,
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "prior_boundaries": {
            "v7_ready": v7.get("hardware_sampler_evidence_boundary_v7_ready") is True,
            "v6_ready": v6.get("hardware_sampler_evidence_boundary_v6_ready") is True,
            "v7_speedup_allowed": v7.get("authenticated_speedup_claim_allowed") is True,
            "v6_speedup_allowed": v6.get("speedup_claim_allowed") is True,
        },
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    local_tooling_checks: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Persist the Exp 3174 result JSON."""

    root_path = Path(root)
    out_path = root_path / Path(output_path)
    artifact = build_artifact(root_path, local_tooling_checks=local_tooling_checks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _local_source_artifact(root: Path, spec: tuple[str, Path, bool, str, str]) -> JsonDict:
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


def _public_source_rows(research_text: str) -> list[JsonDict]:
    return [
        {
            "role": role,
            "url": url,
            "required": False,
            "present": url in research_text,
            "readable": True,
            "source_type": "public_url_reference",
            "evidence_class": "public_ecosystem_reference",
            "claim_use": claim_use,
        }
        for role, url, claim_use in PUBLIC_ECOSYSTEM_REFERENCES
    ]


def _tooling_source_rows(tooling_checks: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "role": f"local_tooling_{tool_name}",
            "tool": tool_name,
            "required": False,
            "present": True,
            "readable": True,
            "source_type": "local_import_metadata",
            "evidence_class": "local_tooling_check",
            "claim_use": "local import/package availability only; not hardware performance evidence",
        }
        for tool_name in sorted(tooling_checks)
    ]


def _public_reference_partition(research_text: str) -> list[JsonDict]:
    return [
        {
            "role": role,
            "url": url,
            "present_in_research_references": url in research_text,
            "claim_use": claim_use,
            "local_execution_evidence": False,
            "speedup_evidence": False,
        }
        for role, url, claim_use in PUBLIC_ECOSYSTEM_REFERENCES
    ]


def _cuda_status(v7: Mapping[str, Any], cuda_runtime: Mapping[str, Any]) -> str:
    prior = v7.get("cuda_status")
    if isinstance(prior, str) and prior:
        return prior
    ready = (
        cuda_runtime.get("sota_runtime_ready_v3") is True
        and cuda_runtime.get("llama_cpp_gpu_offload_verified") is True
        and int(cuda_runtime.get("usable_response_count") or 0) > 0
    )
    if not ready:
        return "blocked_cuda_runtime_evidence_missing"
    suffix = "_flagged_adversarial" if cuda_runtime.get("flagged_adversarial") is True else ""
    return f"runtime_ready_no_sampler_speedup_claim{suffix}"


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


def _gatemate_status(v7: Mapping[str, Any], gatemate: Mapping[str, Any]) -> str:
    prior = v7.get("gatemate_status")
    if isinstance(prior, str) and prior:
        return prior
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


def _prior_or_default(v7: Mapping[str, Any], key: str, default: str) -> str:
    prior = v7.get(key)
    return prior if isinstance(prior, str) and prior else default


def _claim_boundaries(
    cuda_status: str,
    kv260_status: str,
    gatemate_status: str,
    polarfire_status: str,
    extropic_thrml_status: str,
    kona_status: str,
) -> JsonDict:
    return {
        "cuda_local_gguf": _claim_boundary_row(
            cuda_status,
            "local runtime/tooling readiness only; no sampler or speedup claim",
        ),
        "kv260": _claim_boundary_row(
            kv260_status,
            "historical board transcript may scope a workload; fresh speedup requires full bundle",
        ),
        "gatemate": _claim_boundary_row(
            gatemate_status,
            "operator-visible output/readback evidence required before board claims",
        ),
        "polarfire": _claim_boundary_row(
            polarfire_status,
            "historical workload transcript may scope dispatch; no general speedup claim",
        ),
        "extropic_thrml_tsu": _claim_boundary_row(
            extropic_thrml_status,
            "public TSU/THRML context and import availability are not local TSU access",
        ),
        "kona_aleph": _claim_boundary_row(
            kona_status,
            "public Kona/Aleph context is not local interoperability evidence",
        ),
    }


def _claim_boundary_row(status: str, allowed_claim: str) -> JsonDict:
    return {
        "status": status,
        "allowed_claim": allowed_claim,
        "speedup_claim_allowed": False,
        "speedup_required_fields": list(SPEEDUP_REQUIRED_FIELDS),
    }


def _authenticated_performance_evidence(kv260_status: str, polarfire_status: str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if kv260_status.startswith("authenticated"):
        rows.append(
            {
                "substrate": "kv260",
                "status": kv260_status,
                "source_artifacts": [
                    KV260_LATENCY_REL_PATH.as_posix(),
                    KV260_LATENCY_TRANSCRIPT_REL_PATH.as_posix(),
                ],
                "fresh_speedup_claim_allowed": False,
            }
        )
    if polarfire_status.startswith("authenticated"):
        rows.append(
            {
                "substrate": "polarfire",
                "status": polarfire_status,
                "source_artifacts": [
                    POLARFIRE_1000_REL_PATH.as_posix(),
                    POLARFIRE_1000_TRANSCRIPT_REL_PATH.as_posix(),
                ],
                "fresh_speedup_claim_allowed": False,
            }
        )
    return rows


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("hardware_tooling_boundary_v8_ready") is not True:
        missing = ", ".join(
            str(path) for path in artifact.get("missing_required_source_artifacts", [])
        )
        return f"blocked_precondition: missing required v8 hardware/tooling sources: {missing}"
    return (
        "complete: hardware_tooling_boundary_v8_ready=true; "
        "authenticated_speedup_claim_allowed=false; hardware_commands_run=0; "
        "speedup_claim_made=false; local_tooling_checks_recorded=true"
    )
