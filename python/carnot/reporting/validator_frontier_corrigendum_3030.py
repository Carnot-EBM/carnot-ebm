"""Build the Exp 3030 validator-frontier corrigendum artifact.

Spec refs: REQ-REPORT-3030, SCENARIO-REPORT-3030.

This module is an accounting ledger, not a verifier rerun. It reads the
validator-tree corpus, the BEAVER-style frontier certificate, and the prior
methodology corrigendum, then turns their existing rows into explicit claim
regions. That separation matters because cached exact checks can support a
candidate-level statement, while unresolved probability bounds, semantic-only
space, and fallback provenance must remain visible instead of being promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.284"
SCHEMA = "carnot.validator_frontier_corrigendum.v2"
ARTIFACT = "experiment_3030_validator_frontier_corrigendum_v2"
INFERENCE_SUBSTRATE_KIND = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3030_validator_frontier_corrigendum_v2.json")

EXP3017_REL_PATH = Path("results/experiment_3017_nsvif_instruction_validator_tree_expansion_v1.json")
EXP3017_MANIFEST_REL_PATH = Path(
    "results/nsvif_instruction_validator_tree_expansion_3017/validator_manifest.jsonl"
)
EXP3018_REL_PATH = Path("results/experiment_3018_beaver_style_validator_frontier_certificate_v1.json")
EXP3018_MANIFEST_REL_PATH = Path(
    "results/beaver_style_validator_frontier_certificate_3018/certificate_manifest.jsonl"
)
EXP3027_REL_PATH = Path("results/experiment_3027_adversarial_flag_methodology_corrigendum_v1.json")

CLASSIFICATIONS = (
    "verified",
    "irrelevant",
    "unresolved",
    "fallback_only",
    "missing_authority",
)


@dataclass(frozen=True)
class SourceSpec:
    """A JSON or JSONL source that must be present for complete accounting."""

    experiment_id: str
    path: Path
    source_format: str
    required: bool = True


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3017", EXP3017_REL_PATH, "json"),
    SourceSpec("exp3017_manifest", EXP3017_MANIFEST_REL_PATH, "jsonl"),
    SourceSpec("exp3018", EXP3018_REL_PATH, "json"),
    SourceSpec("exp3018_manifest", EXP3018_MANIFEST_REL_PATH, "jsonl"),
    SourceSpec("exp3027", EXP3027_REL_PATH, "json"),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object while treating absence or malformed JSON as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl_objects(path: Path) -> list[JsonDict]:
    """Read a JSONL manifest, dropping non-object rows and failing closed on errors."""

    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 digest for an existing file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3030: classify validator-frontier regions from upstream artifacts."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_json_sources(root_path)
    manifests = _load_manifest_sources(root_path)
    source_artifacts = _source_artifacts(root_path, payloads, manifests)
    required_errors = _required_source_errors(source_artifacts)
    duration_s = _duration(start, now_s)
    if required_errors:
        return _blocked_artifact(source_artifacts, payloads, manifests, required_errors, duration_s)

    validator_rows = {
        str(row.get("item_id")): row
        for row in manifests["exp3017_manifest"]
        if row.get("item_id")
    }
    frontier_rows = [
        classify_certificate_row(row, validator_rows) for row in manifests["exp3018_manifest"]
    ]
    frontier_rows.extend(_fallback_only_rows(root_path, payloads["exp3018"]))
    counts = _counts_by_class(frontier_rows)
    source_unresolved = int(payloads["exp3018"].get("unresolved_count") or 0)
    ready = (
        bool(frontier_rows)
        and all(row.get("classification") in CLASSIFICATIONS for row in frontier_rows)
        and counts["unresolved"] >= source_unresolved
    )

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "validator_frontier_corrigendum_ready": ready,
        "verified_region_count": counts["verified"],
        "irrelevant_region_count": counts["irrelevant"],
        "unresolved_region_count": counts["unresolved"],
        "fallback_only_count": counts["fallback_only"],
        "missing_authority_count": counts["missing_authority"],
        "frontier_rows": frontier_rows,
        "cited_upstream_artifacts": _cited_upstream_artifacts(source_artifacts, payloads, manifests),
        "source_checksums": _source_checksums(source_artifacts),
        "source_unresolved_count": source_unresolved,
        "exp3027_unresolved_bound_rows": _mapping_list(
            payloads["exp3027"].get("unresolved_bound_rows")
        ),
        "inference_substrate": _inference_substrate(),
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "status_updates_written": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration_s,
        "honest_verdict": _honest_verdict(ready, counts),
    }


def classify_certificate_row(
    row: Mapping[str, Any],
    validator_rows_by_item: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Classify one Exp 3018 certificate manifest row for claim wording."""

    status = str(row.get("certificate_status") or "unknown")
    source_reason = str(row.get("source_rejection_reason") or "")
    exact_authorities = _exact_authorities(validator_rows_by_item.get(str(row.get("item_id")), {}))
    base = _frontier_base(row, exact_authorities)
    if row.get("enumerator_fallback_used") is True:
        return base | _classification(
            "fallback_only",
            "deterministic_fallback",
            "fallback_only_not_promotable",
            "Fallback evidence is visible but cannot be promoted as exact validator authority.",
        )
    outcome = _mapping(row.get("deterministic_validator_outcome"))
    if row.get("live_llm_evidence_used") is True or outcome.get("llm_judge_used") is True:
        return base | _classification(
            "missing_authority",
            "live_llm_dependency",
            "missing_exact_authority_or_provenance",
            "Do not promote this row; it depends on live or judge-style LLM authority.",
        )
    if status in {"certified_safe", "certified_violating"}:
        if outcome and exact_authorities:
            return base | _classification(
                "verified",
                "exact_validator_tree",
                "exact_candidate_check_only_probability_bound_unresolved",
                "May claim this cached candidate was exact-checked. Do not claim full BEAVER probability bounds.",
            )
        return base | _classification(
            "missing_authority",
            "missing_exact_validator_provenance",
            "missing_exact_authority_or_provenance",
            "Do not promote this row; exact validator outcome or provenance is missing.",
        )
    if status == "non_prefix_closed":
        return base | _classification(
            "irrelevant",
            _irrelevant_authority_type(row),
            "clipped_irrelevant_to_exact_authority",
            "May report this as clipped or irrelevant to exact authority, not as verified.",
        )
    if status == "unresolved":
        return base | _classification(
            "unresolved",
            _unresolved_authority_type(source_reason),
            "unresolved_probability_or_authority_bound",
            "Do not promote this region; the unresolved bound remains visible.",
        )
    return base | _classification(
        "missing_authority",
        "unknown_certificate_status",
        "missing_exact_authority_or_provenance",
        "Do not promote this row; the certificate status is not recognized.",
    )


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3030 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def main(root: Path | str = REPO_ROOT) -> int:
    """Write the corrigendum artifact and return process-style success."""

    output = write_artifact(root)
    artifact = read_json_object(output)
    return 0 if artifact.get("validator_frontier_corrigendum_ready") is True else 1


def _blocked_artifact(
    source_artifacts: list[JsonDict],
    payloads: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, list[JsonDict]],
    required_errors: list[JsonDict],
    duration_s: float,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "validator_frontier_corrigendum_ready": False,
        "verified_region_count": 0,
        "irrelevant_region_count": 0,
        "unresolved_region_count": 0,
        "fallback_only_count": 0,
        "missing_authority_count": 0,
        "frontier_rows": [],
        "required_source_errors": required_errors,
        "cited_upstream_artifacts": _cited_upstream_artifacts(source_artifacts, payloads, manifests),
        "source_checksums": _source_checksums(source_artifacts),
        "inference_substrate": _inference_substrate(),
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "status_updates_written": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration_s,
        "honest_verdict": "blocked_required_upstream_missing",
    }


def _load_json_sources(root: Path) -> dict[str, JsonDict]:
    return {
        spec.experiment_id: read_json_object(root / spec.path)
        for spec in SOURCE_SPECS
        if spec.source_format == "json"
    }


def _load_manifest_sources(root: Path) -> dict[str, list[JsonDict]]:
    return {
        spec.experiment_id: read_jsonl_objects(root / spec.path)
        for spec in SOURCE_SPECS
        if spec.source_format == "jsonl"
    }


def _source_artifacts(
    root: Path,
    payloads: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, list[JsonDict]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        readable = bool(
            payloads.get(spec.experiment_id)
            if spec.source_format == "json"
            else manifests.get(spec.experiment_id)
        )
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "format": spec.source_format,
                "present": path.is_file(),
                "required": spec.required,
                "readable": readable,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _required_source_errors(source_artifacts: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "experiment_id": row["experiment_id"],
            "path": row["path"],
            "reason": "missing_or_malformed_artifact",
        }
        for row in source_artifacts
        if row["required"] and not row["readable"]
    ]


def _frontier_base(row: Mapping[str, Any], exact_authorities: list[str]) -> JsonDict:
    return {
        "row_id": str(row.get("row_id") or "unknown"),
        "source_artifact_path": EXP3018_REL_PATH.as_posix(),
        "source_manifest_path": EXP3018_MANIFEST_REL_PATH.as_posix(),
        "source_row_type": str(row.get("row_type") or ""),
        "certificate_status": str(row.get("certificate_status") or "unknown"),
        "item_id": str(row.get("item_id") or ""),
        "exact_authorities": exact_authorities,
        "probability_bound_exact": _probability_bound_exact(row),
    }


def _classification(
    classification: str,
    authority_type: str,
    bound_status: str,
    allowed_claim_wording: str,
) -> JsonDict:
    return {
        "classification": classification,
        "authority_type": authority_type,
        "bound_status": bound_status,
        "allowed_claim_wording": allowed_claim_wording,
    }


def _exact_authorities(validator_row: Mapping[str, Any]) -> list[str]:
    nodes = _mapping_list(_mapping(validator_row.get("validator_tree")).get("nodes"))
    return _unique(
        str(node.get("authority") or "")
        for node in nodes
        if node.get("authoritative") is True
        and node.get("exact_checked") is not False
        and node.get("authority") != "semantic_boundary_non_authoritative"
    )


def _probability_bound_exact(row: Mapping[str, Any]) -> bool:
    placeholder = _mapping(row.get("probability_bound_placeholder"))
    return placeholder.get("exact_probability_computed") is True


def _irrelevant_authority_type(row: Mapping[str, Any]) -> str:
    if row.get("row_type") == "non_prefix_closed_node":
        return "non_authoritative_semantic_boundary"
    return "ambiguous_instruction_no_deterministic_boundary"


def _unresolved_authority_type(source_reason: str) -> str:
    if source_reason == "llm_only_label":
        return "live_llm_dependency"
    if source_reason == "nondeterministic_validator":
        return "nondeterministic_validator"
    return "unresolved_validator_authority"


def _fallback_only_rows(root: Path, exp3018: Mapping[str, Any]) -> list[JsonDict]:
    provenance = _mapping(exp3018.get("enumerator_fallback_provenance"))
    return [
        {
            "row_id": f"exp3004_enumerator_fallback:{index}",
            "source_artifact_path": _path_for_row(root, path),
            "source_manifest_path": "",
            "source_row_type": "enumerator_fallback_provenance",
            "certificate_status": "fallback_only",
            "item_id": "",
            "exact_authorities": [],
            "probability_bound_exact": False,
            "classification": "fallback_only",
            "authority_type": "deterministic_fallback",
            "bound_status": "fallback_only_not_promotable",
            "allowed_claim_wording": (
                "Enumerator fallback is accounting evidence only and cannot be promoted as exact authority."
            ),
        }
        for index, path in enumerate(_string_list(provenance.get("paths")))
    ]


def _path_for_row(root: Path, path: str) -> str:
    raw = Path(path)
    if not raw.is_absolute():
        return path
    try:
        return raw.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path


def _counts_by_class(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {classification: 0 for classification in CLASSIFICATIONS}
    for row in rows:
        classification = str(row.get("classification") or "")
        if classification in counts:
            counts[classification] += 1
    return counts


def _cited_upstream_artifacts(
    source_artifacts: list[JsonDict],
    payloads: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, list[JsonDict]],
) -> list[JsonDict]:
    return [
        {
            "experiment_id": str(source["experiment_id"]),
            "path": str(source["path"]),
            "present": bool(source["present"]),
            "required": bool(source["required"]),
            "readable": bool(source["readable"]),
            "sha256": source["sha256"],
            "honest_verdict": str(payloads.get(str(source["experiment_id"]), {}).get("honest_verdict") or ""),
            "inference_substrate": payloads.get(str(source["experiment_id"]), {}).get("inference_substrate"),
            "source_field_summary": _source_field_summary(
                str(source["experiment_id"]), payloads, manifests
            ),
        }
        for source in source_artifacts
    ]


def _source_field_summary(
    experiment_id: str,
    payloads: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, list[JsonDict]],
) -> JsonDict:
    if experiment_id == "exp3017":
        payload = payloads.get("exp3017", {})
        return {
            "instruction_validator_tree_ready": payload.get("instruction_validator_tree_ready") is True,
            "all_authoritative_nodes_exact_checked": (
                payload.get("all_authoritative_nodes_exact_checked") is True
            ),
            "llm_judge_used": payload.get("llm_judge_used") is True,
        }
    if experiment_id == "exp3017_manifest":
        return {"row_count": len(manifests.get("exp3017_manifest", []))}
    if experiment_id == "exp3018":
        payload = payloads.get("exp3018", {})
        return {
            "frontier_certificate_ready": payload.get("frontier_certificate_ready") is True,
            "unresolved_count": payload.get("unresolved_count"),
            "non_prefix_closed_count": payload.get("non_prefix_closed_count"),
            "live_llm_evidence_used": payload.get("live_llm_evidence_used") is True,
            "enumerator_fallback_separated": payload.get("enumerator_fallback_separated") is True,
            "probability_bound_policy": payload.get("probability_bound_policy"),
        }
    if experiment_id == "exp3018_manifest":
        rows = manifests.get("exp3018_manifest", [])
        return {"row_count": len(rows), "certificate_status_counts": _status_counts(rows)}
    payload = payloads.get("exp3027", {})
    return {
        "methodology_corrigendum_ready": payload.get("methodology_corrigendum_ready") is True,
        "unresolved_bound_row_count": len(_mapping_list(payload.get("unresolved_bound_rows"))),
    }


def _source_checksums(source_artifacts: list[JsonDict]) -> dict[str, str | None]:
    return {str(row["path"]): row["sha256"] for row in source_artifacts}


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("certificate_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _inference_substrate() -> JsonDict:
    return {
        "kind": INFERENCE_SUBSTRATE_KIND,
        "no_live_llm_inference": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_top_level_live_model_metadata": True,
        "source_metadata_location": "cited_upstream_artifacts[].source_field_summary",
    }


def _honest_verdict(ready: bool, counts: Mapping[str, int]) -> str:
    if ready:
        return (
            "complete: validator_frontier_corrigendum_ready=true; "
            f"verified={counts['verified']}; irrelevant={counts['irrelevant']}; "
            f"unresolved={counts['unresolved']}; fallback_only={counts['fallback_only']}; "
            f"missing_authority={counts['missing_authority']}"
        )
    return "blocked_required_upstream_missing"


def _duration(start: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - start), 6)


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _unique(values: Any) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


__all__ = [
    "EXP3017_MANIFEST_REL_PATH",
    "EXP3017_REL_PATH",
    "EXP3018_MANIFEST_REL_PATH",
    "EXP3018_REL_PATH",
    "EXP3027_REL_PATH",
    "INFERENCE_SUBSTRATE_KIND",
    "OUTPUT_REL_PATH",
    "build_artifact",
    "classify_certificate_row",
    "main",
    "read_json_object",
    "read_jsonl_objects",
    "sha256_file",
    "write_artifact",
]
