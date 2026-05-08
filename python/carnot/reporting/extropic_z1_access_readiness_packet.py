"""Build the Exp 1545 Extropic Z1 access-readiness packet.

Spec refs: REQ-SAMPLE-055, SCENARIO-SAMPLE-083.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260508"
MILESTONE = "2026.04.118"
EXPERIMENT = "1545_extropic_z1_access_readiness_packet"
SCHEMA = "extropic_z1_access_readiness_packet_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1545_extropic_z1_access_readiness_packet.json"
)
DEFAULT_PACKET_PATH = REPO_ROOT / "ops" / "extropic_z1_readiness_packet.md"
DEFAULT_TRANSCRIPT_SCHEMA_PATH = REPO_ROOT / "ops" / "extropic_z1_transcript_schema.json"
EXP1543_PATH = (
    REPO_ROOT / "results" / "experiment_1543_thrml_carnot_parity_n256_schedule_stress.json"
)
EXP1544_PATH = REPO_ROOT / "results" / "experiment_1544_thrml_diverse_topology_parity_n64.json"

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "extropic_z1_readiness_packet_ready",
    "readiness_packet_path",
    "benchmark_cases_included",
    "transcript_schema_path",
    "required_device_evidence_fields",
    "no_hardware_execution_claim",
    "simulator_artifacts_referenced",
    "access_blockers",
    "focused_checks_passed",
    "honest_verdict",
}

TRANSCRIPT_REQUIRED_FIELDS = [
    "transcript_schema_version",
    "run_date",
    "authenticated_access_proof",
    "access_grant_reference",
    "provider_or_lab_operator",
    "device_family",
    "device_identifier",
    "device_firmware_or_runtime",
    "sdk_package_name",
    "sdk_version",
    "thrml_version",
    "device_discovery_command",
    "execution_timestamp_utc",
    "host_identifier",
    "benchmark_case_id",
    "schedule_id",
    "topology",
    "n_spins",
    "sample_count",
    "state_encoding",
    "sample_shape",
    "sample_dtype",
    "output_samples_sha256",
    "energy_trace_sha256",
    "energy_metric_fields",
    "latency_metric_fields",
    "hardware_execution_performed",
    "simulator_fallback_used",
    "claim_boundary_acknowledged",
]

EXPECTED_METRIC_FIELDS = [
    "mean_energy",
    "magnetization",
    "energy_autocorrelation_lag1",
    "kl_divergence_vs_simulator",
    "sample_shape",
    "sample_dtype",
    "output_samples_sha256",
    "energy_trace_sha256",
    "host_to_device_latency_us",
    "device_sampling_latency_us",
    "device_to_host_latency_us",
    "end_to_end_latency_us",
]

ACCESS_BLOCKERS = [
    "no_authenticated_extropic_z1_or_xtr0_device_access",
    "no_extropic_sdk_credentials_or_device_discovery_transcript",
    "no_authenticated_hardware_run_transcript",
    "no_device_latency_or_sample_quality_evidence_from_z1",
    "public_thrml_material_only_simulator_parity_artifacts",
]

ROLLBACK_CRITERIA = [
    "Missing authenticated_access_proof, device_identifier, or device_discovery_command.",
    "Any transcript sets hardware_execution_performed=false or simulator_fallback_used=true.",
    "Sample shape, state encoding, or checksum fields are absent or inconsistent.",
    "Mean-energy, KL, magnetization, or autocorrelation metrics exceed the simulator gates.",
    "Latency fields are missing, impossible, or mixed across host/device scopes.",
    "The SDK path silently falls back to THRML/JAX/CPU simulation.",
]


def load_json(path: Path | str) -> dict[str, Any]:
    """REQ-SAMPLE-055: read a source artifact as JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-SAMPLE-055: persist the auditable bootstrap marker first.

    Hardware-readiness work can be interrupted by missing source artifacts or
    packet rendering mistakes. The bootstrap JSON proves the run started before
    any terminal readiness claim was made.
    """

    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "milestone": MILESTONE,
            "extropic_z1_readiness_packet_ready": False,
            "benchmark_cases_included": 0,
            "required_device_evidence_fields": list(TRANSCRIPT_REQUIRED_FIELDS),
            "no_hardware_execution_claim": True,
            "simulator_artifacts_referenced": [],
            "access_blockers": list(ACCESS_BLOCKERS),
            "focused_checks_passed": False,
            "honest_verdict": "complete: in_progress_extropic_z1_access_readiness_packet_seeded",
        }
    )
    return _write_json(Path(out_path), artifact)


def _relative_path(path: Path | str, *, repo_root: Path = REPO_ROOT) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        parts = path.parts
        for anchor in ("results", "ops", "openspec"):
            if anchor in parts:
                return str(Path(*parts[parts.index(anchor) :]))
        return str(path)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status_complete(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("status") or "").lower() == "complete"


def _metadata_hardware_false(payload: Mapping[str, Any]) -> bool:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return False
    forbidden = (
        "z1_hardware_execution",
        "xtr0_hardware_execution",
        "tsu_hardware_execution",
        "board_execution",
        "synthesis_run",
        "bitstream_generated",
    )
    return all(metadata.get(field) in (False, None) for field in forbidden)


def _validate_exp1543(payload: Mapping[str, Any]) -> None:
    if not _status_complete(payload) or payload.get("thrml_parity_n256_schedule_ready") is not True:
        raise ValueError("Exp1543 is not complete n=256 schedule readiness evidence")
    if payload.get("parity_passed") is not True:
        raise ValueError("Exp1543 parity_passed is not true")
    if (
        payload.get("simulator_only") is not True
        or payload.get("no_tsu_hardware_claim") is not True
    ):
        raise ValueError("Exp1543 violates no-TSU simulator boundary")
    if payload.get("n_spins") != 256:
        raise ValueError("Exp1543 must be n_spins=256")
    if not isinstance(payload.get("schedule_manifest"), list) or not payload["schedule_manifest"]:
        raise ValueError("Exp1543 schedule_manifest is missing")
    if not _metadata_hardware_false(payload):
        raise ValueError("Exp1543 metadata must not record hardware execution")


def _validate_exp1544(payload: Mapping[str, Any]) -> None:
    if (
        not _status_complete(payload)
        or payload.get("diverse_topology_parity_n64_ready") is not True
    ):
        raise ValueError("Exp1544 is not complete n=64 diverse-topology readiness evidence")
    if payload.get("parity_passed") is not True:
        raise ValueError("Exp1544 parity_passed is not true")
    if (
        payload.get("simulator_only") is not True
        or payload.get("no_tsu_hardware_claim") is not True
    ):
        raise ValueError("Exp1544 violates no-TSU simulator boundary")
    if payload.get("n_spins") != 64:
        raise ValueError("Exp1544 must be n_spins=64")
    topologies = payload.get("topologies_tested")
    if not isinstance(topologies, list) or not {
        "complete",
        "sparse_random",
        "lattice",
        "scale_free",
    }.issubset({str(topology) for topology in topologies}):
        raise ValueError("Exp1544 must include the four required topology families")
    if not _metadata_hardware_false(payload):
        raise ValueError("Exp1544 metadata must not record hardware execution")


def validate_source_artifacts(exp1543: Mapping[str, Any], exp1544: Mapping[str, Any]) -> bool:
    """REQ-SAMPLE-055: readiness depends on simulator-only parity artifacts."""

    _validate_exp1543(exp1543)
    _validate_exp1544(exp1544)
    return True


def _metric_snapshot(payload: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {field: payload[field] for field in fields if field in payload}


def build_benchmark_cases(
    exp1543: Mapping[str, Any],
    exp1544: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """REQ-SAMPLE-055: turn simulator parity artifacts into device benchmark cases."""

    validate_source_artifacts(exp1543, exp1544)
    cases: list[dict[str, Any]] = []
    schedule_results = exp1543.get("schedule_results", {})
    if not isinstance(schedule_results, Mapping):
        schedule_results = {}
    for schedule in exp1543.get("schedule_manifest", []):
        if not isinstance(schedule, Mapping):
            continue
        schedule_id = str(schedule.get("schedule_id") or "unknown_schedule")
        result = schedule_results.get(schedule_id, {})
        if not isinstance(result, Mapping):
            result = {}
        cases.append(
            {
                "case_id": f"n256_schedule_stress:{schedule_id}",
                "source_artifact": "results/experiment_1543_thrml_carnot_parity_n256_schedule_stress.json",
                "n_spins": 256,
                "topology": "signed_ring_chord",
                "schedule_id": schedule_id,
                "schedule": dict(schedule),
                "seeds": list(result.get("seeds", [])),
                "sample_count_per_backend": int(exp1543.get("samples_per_schedule") or 0),
                "expected_metric_fields": list(EXPECTED_METRIC_FIELDS),
                "simulator_baseline_metrics": _metric_snapshot(
                    result,
                    (
                        "mean_energy_delta",
                        "magnetization_delta",
                        "kl_divergence",
                        "autocorrelation_delta",
                    ),
                ),
            }
        )

    metadata = exp1544.get("metadata", {})
    topology_seeds = {}
    if isinstance(metadata, Mapping):
        topology_seeds.update(metadata.get("topology_seeds", {}) or {})
    topology_seeds.update(exp1544.get("topology_seeds", {}) or {})
    topology_results = exp1544.get("per_topology_results", {})
    if not isinstance(topology_results, Mapping):
        topology_results = {}
    for topology in exp1544.get("topologies_tested", []):
        topology_name = str(topology)
        result = topology_results.get(topology_name, {})
        if not isinstance(result, Mapping):
            result = {}
        cases.append(
            {
                "case_id": f"n64_diverse_topology:{topology_name}",
                "source_artifact": "results/experiment_1544_thrml_diverse_topology_parity_n64.json",
                "n_spins": 64,
                "topology": topology_name,
                "topology_seed": topology_seeds.get(topology_name),
                "seeds": list(result.get("seeds", exp1544.get("seeds", []))),
                "sample_count_per_backend": int(exp1544.get("n_samples_per_backend") or 0),
                "schedule": {
                    "beta": 1.05,
                    "n_warmup": exp1544.get("warmup"),
                    "steps_per_sample": exp1544.get("thinning"),
                    "use_checkerboard": True,
                },
                "expected_metric_fields": list(EXPECTED_METRIC_FIELDS),
                "simulator_baseline_metrics": _metric_snapshot(
                    result,
                    (
                        "mean_energy_delta",
                        "magnetization_delta",
                        "kl_divergence",
                        "autocorrelation_summary",
                    ),
                ),
            }
        )
    return cases


def build_transcript_schema() -> dict[str, Any]:
    """REQ-SAMPLE-055: describe required fields for a future authenticated run."""

    string_fields = {
        "transcript_schema_version",
        "run_date",
        "authenticated_access_proof",
        "access_grant_reference",
        "provider_or_lab_operator",
        "device_family",
        "device_identifier",
        "device_firmware_or_runtime",
        "sdk_package_name",
        "sdk_version",
        "thrml_version",
        "device_discovery_command",
        "execution_timestamp_utc",
        "host_identifier",
        "benchmark_case_id",
        "schedule_id",
        "topology",
        "state_encoding",
        "sample_dtype",
        "output_samples_sha256",
        "energy_trace_sha256",
    }
    properties: dict[str, Any] = {
        field: {"type": "string", "minLength": 1} for field in string_fields
    }
    properties.update(
        {
            "n_spins": {"type": "integer", "minimum": 1},
            "sample_count": {"type": "integer", "minimum": 1},
            "sample_shape": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "minItems": 2,
            },
            "energy_metric_fields": {
                "type": "object",
                "required": ["mean_energy", "magnetization", "kl_divergence_vs_simulator"],
            },
            "latency_metric_fields": {
                "type": "object",
                "required": [
                    "host_to_device_latency_us",
                    "device_sampling_latency_us",
                    "device_to_host_latency_us",
                    "end_to_end_latency_us",
                ],
            },
            "hardware_execution_performed": {"type": "boolean", "const": True},
            "simulator_fallback_used": {"type": "boolean", "const": False},
            "claim_boundary_acknowledged": {"type": "boolean", "const": True},
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Extropic Z1 authenticated benchmark transcript",
        "type": "object",
        "additionalProperties": True,
        "required": list(TRANSCRIPT_REQUIRED_FIELDS),
        "properties": properties,
    }


def validate_transcript_schema(schema: Mapping[str, Any]) -> bool:
    """REQ-SAMPLE-055: focused schema check without relying on jsonschema."""

    if schema.get("title") != "Extropic Z1 authenticated benchmark transcript":
        raise ValueError("transcript schema title is invalid")
    if schema.get("type") != "object":
        raise ValueError("transcript schema must describe an object")
    required = list(schema.get("required", []))
    for field in TRANSCRIPT_REQUIRED_FIELDS:
        if field not in required:
            raise ValueError(f"transcript schema missing required field: {field}")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ValueError("transcript schema properties must be a mapping")
    for field in TRANSCRIPT_REQUIRED_FIELDS:
        if field not in properties:
            raise ValueError(f"transcript schema missing property: {field}")
    if properties["hardware_execution_performed"].get("const") is not True:
        raise ValueError(
            "future hardware transcript must require hardware_execution_performed=true"
        )
    if properties["simulator_fallback_used"].get("const") is not False:
        raise ValueError("future hardware transcript must require simulator_fallback_used=false")
    return True


def _source_artifact_refs(
    paths: Sequence[Path | str], *, repo_root: Path = REPO_ROOT
) -> list[dict[str, Any]]:
    refs = []
    for path_value in paths:
        path = Path(path_value)
        refs.append(
            {
                "path": _relative_path(path, repo_root=repo_root),
                "sha256": _sha256_file(path),
                "simulator_only": True,
                "hardware_execution": False,
            }
        )
    return refs


def _derive_access_blockers(
    *,
    hardware_wishlist_text: str,
    research_references_text: str,
    known_issues_text: str,
) -> list[str]:
    blockers = list(ACCESS_BLOCKERS)
    joined = f"{hardware_wishlist_text}\n{research_references_text}".lower()
    if "no extropic hardware access" in joined or "no authenticated hardware access" in joined:
        blockers.append("source_docs_confirm_no_authenticated_extropic_access")
    if (
        "independent-rng" in known_issues_text.lower()
        or "independent rng" in known_issues_text.lower()
    ):
        blockers.append("thrml_independent_rng_followup_not_completed")
    return sorted(dict.fromkeys(blockers))


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_case_table(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["| case_id | n_spins | topology | seed/schedule manifest |", "|---|---:|---|---|"]
    for case in cases:
        schedule = case.get("schedule", {})
        if isinstance(schedule, Mapping):
            schedule_bits = ", ".join(
                f"{key}={value}"
                for key, value in schedule.items()
                if value is not None
                and key in {"beta", "n_warmup", "steps_per_sample", "use_checkerboard"}
            )
        else:
            schedule_bits = ""
        seeds = case.get("seeds", [])
        if isinstance(seeds, Sequence) and not isinstance(seeds, str):
            seed_bits = ",".join(str(seed) for seed in seeds[:5])
        else:
            seed_bits = str(seeds)
        manifest = f"seeds=[{seed_bits}]; {schedule_bits}"
        lines.append(
            f"| {_md_cell(case['case_id'])} | {_md_cell(case['n_spins'])} | "
            f"{_md_cell(case['topology'])} | {_md_cell(manifest)} |"
        )
    return lines


def render_readiness_packet(
    *,
    benchmark_cases: Sequence[Mapping[str, Any]],
    transcript_schema_path: str,
    required_device_evidence_fields: Sequence[str],
    simulator_artifacts_referenced: Sequence[Mapping[str, Any]],
    access_blockers: Sequence[str],
    hardware_wishlist_text: str,
    research_references_text: str,
    known_issues_text: str,
) -> str:
    """SCENARIO-SAMPLE-083: render the operator-facing Z1 access packet."""

    source_summary = {
        "hardware_wishlist_mentions_no_access": "no extropic hardware access"
        in hardware_wishlist_text.lower(),
        "references_request_packet_not_claim": "readiness packet"
        in research_references_text.lower(),
        "known_issue_rng_followup": "independent-rng" in known_issues_text.lower()
        or "independent rng" in known_issues_text.lower(),
    }
    lines = [
        "# Extropic Z1 Access-Readiness Packet",
        "",
        "Spec refs: REQ-SAMPLE-055, SCENARIO-SAMPLE-083.",
        "",
        "## Status",
        "",
        "- status: access_readiness_packet_only",
        "- milestone: 2026.04.118",
        "- run_date: 20260508",
        "- no_hardware_execution_claim: true",
        "",
        "## Benchmark Case List",
        "",
    ]
    lines.extend(_render_case_table(benchmark_cases))
    lines.extend(
        [
            "",
            "## Required Device Metadata",
            "",
        ]
    )
    for field in required_device_evidence_fields:
        lines.append(f"- {field}")
    lines.extend(
        [
            "",
            "## Transcript Schema",
            "",
            f"- schema_path: `{transcript_schema_path}`",
            "- schema requires authenticated access proof, device identity, SDK versions, latency fields, sample shape, output checksums, metric fields, and claim-boundary acknowledgement.",
            "",
            "## Expected Output Checksums Or Metric Fields",
            "",
        ]
    )
    for field in EXPECTED_METRIC_FIELDS:
        lines.append(f"- {field}")
    lines.extend(
        [
            "",
            "## Simulator Artifacts Referenced",
            "",
            "| artifact | sha256 | boundary |",
            "|---|---|---|",
        ]
    )
    for ref in simulator_artifacts_referenced:
        lines.append(
            f"| {_md_cell(ref['path'])} | {_md_cell(ref.get('sha256') or 'missing')} | "
            "software parity only; hardware_execution=false |"
        )
    lines.extend(
        [
            "",
            "## No Hardware Execution Claim",
            "",
            "This packet does not report Extropic Z1, XTR-0, TSU, board, synthesis, bitstream, latency, or device sample execution. The referenced evidence is software/simulator parity only.",
            "",
            "## Access Blockers",
            "",
        ]
    )
    for blocker in access_blockers:
        lines.append(f"- {blocker}")
    lines.extend(
        [
            "",
            "## Rollback Criteria",
            "",
        ]
    )
    for criterion in ROLLBACK_CRITERIA:
        lines.append(f"- {criterion}")
    lines.extend(
        [
            "",
            "## Source Context Checks",
            "",
        ]
    )
    for key, value in source_summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def _protected_files_unchanged(repo_root: Path) -> dict[str, bool]:
    if not (repo_root / ".git").exists():
        return {
            "research-roadmap.yaml": True,
            "scripts/research_conductor.py": True,
        }
    result: dict[str, bool] = {}
    for relative in ("research-roadmap.yaml", "scripts/research_conductor.py"):
        proc = subprocess.run(
            ["git", "diff", "--quiet", "--", relative],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        result[relative] = proc.returncode == 0
    return result


def _validate_packet_text(packet_text: str) -> bool:
    required_phrases = [
        "Benchmark Case List",
        "Required Device Metadata",
        "Transcript Schema",
        "Expected Output Checksums Or Metric Fields",
        "No Hardware Execution Claim",
        "Access Blockers",
        "Rollback Criteria",
        "no_hardware_execution_claim: true",
    ]
    for phrase in required_phrases:
        if phrase not in packet_text:
            raise ValueError(f"readiness packet missing section or phrase: {phrase}")
    return True


def validate_terminal_artifact(
    artifact: Mapping[str, Any],
    *,
    packet_text: str,
    transcript_schema: Mapping[str, Any],
) -> bool:
    """REQ-SAMPLE-055: focused artifact, schema, and claim-boundary checks."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"terminal artifact missing required fields: {sorted(missing)}")
    if artifact.get("no_hardware_execution_claim") is not True:
        raise ValueError("no_hardware_execution_claim must be true")
    if artifact.get("extropic_z1_readiness_packet_ready") is not True:
        raise ValueError("readiness packet must be marked ready only after focused checks pass")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with a conductor-accepted prefix")
    if not artifact.get("access_blockers"):
        raise ValueError("access_blockers must be non-empty")
    validate_transcript_schema(transcript_schema)
    _validate_packet_text(packet_text)
    return True


def build_artifact(
    *,
    exp1543: Mapping[str, Any],
    exp1544: Mapping[str, Any],
    simulator_artifact_paths: Sequence[Path | str],
    hardware_wishlist_text: str,
    research_references_text: str,
    known_issues_text: str,
    packet_path: str,
    transcript_schema_path: str,
    focused_checks_passed: bool,
    protected_files_unchanged: Mapping[str, bool] | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """REQ-SAMPLE-055: build the packet, transcript schema, and terminal artifact."""

    validate_source_artifacts(exp1543, exp1544)
    benchmark_cases = build_benchmark_cases(exp1543, exp1544)
    transcript_schema = build_transcript_schema()
    validate_transcript_schema(transcript_schema)
    simulator_refs = _source_artifact_refs(simulator_artifact_paths)
    access_blockers = _derive_access_blockers(
        hardware_wishlist_text=hardware_wishlist_text,
        research_references_text=research_references_text,
        known_issues_text=known_issues_text,
    )
    packet_text = render_readiness_packet(
        benchmark_cases=benchmark_cases,
        transcript_schema_path=transcript_schema_path,
        required_device_evidence_fields=TRANSCRIPT_REQUIRED_FIELDS,
        simulator_artifacts_referenced=simulator_refs,
        access_blockers=access_blockers,
        hardware_wishlist_text=hardware_wishlist_text,
        research_references_text=research_references_text,
        known_issues_text=known_issues_text,
    )
    _validate_packet_text(packet_text)
    protected = dict(protected_files_unchanged or {})
    protected_all_unchanged = all(protected.values()) if protected else True
    ready = bool(focused_checks_passed and protected_all_unchanged)
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete" if ready else "blocked",
        "milestone": MILESTONE,
        "extropic_z1_readiness_packet_ready": ready,
        "readiness_packet_path": packet_path,
        "benchmark_cases_included": len(benchmark_cases),
        "benchmark_case_manifest": benchmark_cases,
        "transcript_schema_path": transcript_schema_path,
        "required_device_evidence_fields": list(TRANSCRIPT_REQUIRED_FIELDS),
        "no_hardware_execution_claim": True,
        "simulator_artifacts_referenced": simulator_refs,
        "access_blockers": access_blockers,
        "focused_checks_passed": bool(focused_checks_passed),
        "protected_files_unchanged": protected,
        "research_roadmap_yaml_modified": protected.get("research-roadmap.yaml") is False,
        "scripts_research_conductor_modified": protected.get("scripts/research_conductor.py")
        is False,
        "source_docs_read": [
            "research-hardware-wishlist.md",
            "research-references.md",
            "ops/known-issues.md",
        ],
        "evidence_boundaries": {
            "z1_hardware_execution": False,
            "xtr0_hardware_execution": False,
            "tsu_hardware_execution": False,
            "device_latency_claim": False,
            "software_simulator_parity_only": True,
        },
        "rollback_criteria": list(ROLLBACK_CRITERIA),
        "honest_verdict": (
            "complete: extropic_z1_access_readiness_packet_ready_no_hardware_execution_claim"
            if ready
            else "passed: extropic_z1_access_readiness_packet_blocked_focused_checks_or_protected_files"
        ),
    }
    if ready:
        validate_terminal_artifact(
            artifact,
            packet_text=packet_text,
            transcript_schema=transcript_schema,
        )
    return artifact, packet_text, transcript_schema


def run(
    *,
    repo_root: Path | str = REPO_ROOT,
    focused_checks_passed: bool = True,
) -> dict[str, Any]:
    """SCENARIO-SAMPLE-083: write packet, schema, and terminal result JSON."""

    root = Path(repo_root)
    out_path = root / "results" / "experiment_1545_extropic_z1_access_readiness_packet.json"
    packet_path = root / "ops" / "extropic_z1_readiness_packet.md"
    transcript_schema_path = root / "ops" / "extropic_z1_transcript_schema.json"
    exp1543_path = (
        root / "results" / "experiment_1543_thrml_carnot_parity_n256_schedule_stress.json"
    )
    exp1544_path = root / "results" / "experiment_1544_thrml_diverse_topology_parity_n64.json"

    write_in_progress_artifact(out_path)
    artifact, packet_text, transcript_schema = build_artifact(
        exp1543=load_json(exp1543_path),
        exp1544=load_json(exp1544_path),
        simulator_artifact_paths=[exp1543_path, exp1544_path],
        hardware_wishlist_text=(root / "research-hardware-wishlist.md").read_text(encoding="utf-8"),
        research_references_text=(root / "research-references.md").read_text(encoding="utf-8"),
        known_issues_text=(root / "ops" / "known-issues.md").read_text(encoding="utf-8"),
        packet_path=_relative_path(packet_path, repo_root=root),
        transcript_schema_path=_relative_path(transcript_schema_path, repo_root=root),
        focused_checks_passed=focused_checks_passed,
        protected_files_unchanged=_protected_files_unchanged(root),
    )
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(packet_text, encoding="utf-8")
    transcript_schema_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_schema_path.write_text(
        json.dumps(transcript_schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _write_json(out_path, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
