"""Build the Exp 3202 sparse Potts/PAOA/THRML factor boundary artifact.

Spec refs: REQ-HW-100, SCENARIO-HW-100.

This module prepares a hardware-boundary representation, not a hardware run.
It turns exact rows and invariant certificates into sparse q-state Potts factor
records so a later adapter can see the categorical states, sparse couplings,
and PAOA update metadata without inferring any KV260, GateMate, PolarFire,
TSU, Z1, XTR-0, Kona, or speedup evidence that is not in authenticated
transcripts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
MILESTONE = "2026.05.296"
SCHEMA_VERSION = "carnot.sparse_potts_paoa_thrml_factor_boundary.v1"
EXPERIMENT_ID = "exp3202"
ARTIFACT = "experiment_3202_sparse_potts_paoa_thrml_factor_boundary_v1"

OUTPUT_REL_PATH = Path("results/experiment_3202_sparse_potts_paoa_thrml_factor_boundary_v1.json")
EXP3188_REL_PATH = Path("results/experiment_3188_thrml_factor_graph_api_boundary_v1.json")
EXP3197_REL_PATH = Path(
    "results/experiment_3197_exverus_inductive_certificate_expansion_v1.json"
)
SPEC_REL_PATH = Path("openspec/capabilities/fpga/spec.md")

DENIED_HARDWARE_CLAIMS = (
    "KV260",
    "GateMate",
    "PolarFire",
    "TSU",
    "Z1",
    "XTR-0",
    "Kona",
    "speedup",
)
REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "source_artifacts",
    "factor_record_schema",
    "factor_record_count",
    "q_state_count_summary",
    "graph_density_summary",
    "paoa_metadata_schema",
    "thrml_local_api_checked",
    "authenticated_hardware_transcript_present",
    "speedup_claim_allowed",
    "hardware_claims_denied",
    "honest_verdict",
}
SOURCE_REL_PATHS: tuple[tuple[str, Path, bool, str], ...] = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "text"),
    ("post_295_research_references", Path("research-references.md"), True, "text"),
    ("hardware_wishlist", Path("research-hardware-wishlist.md"), True, "text"),
    ("hardware_exclusion_manifest", Path("ops/exclusion_manifest.yaml"), True, "text"),
    ("fpga_openspec", SPEC_REL_PATH, True, "text"),
    ("exp3188_thrml_factor_graph_boundary", EXP3188_REL_PATH, True, "json"),
    ("exp3197_exverus_invariant_certificates", EXP3197_REL_PATH, True, "json"),
    ("kv260_potts_rtl_reference", Path("hardware/kv260/potts_sampler_v1.v"), False, "verilog"),
    (
        "exp3202_module",
        Path("python/carnot/reporting/sparse_potts_paoa_thrml_factor_boundary_3202.py"),
        False,
        "python",
    ),
    (
        "exp3202_script",
        Path("scripts/experiment_3202_sparse_potts_paoa_thrml_factor_boundary_v1.py"),
        False,
        "python",
    ),
    (
        "exp3202_tests",
        Path("tests/python/test_experiment_3202_sparse_potts_paoa_thrml_factor_boundary_v1.py"),
        False,
        "python",
    ),
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3202_sparse_potts_paoa_thrml_factor_boundary_v1.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/reporting/sparse_potts_paoa_thrml_factor_boundary_3202.py -m pytest -o addopts='' tests/python/test_experiment_3202_sparse_potts_paoa_thrml_factor_boundary_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/reporting/sparse_potts_paoa_thrml_factor_boundary_3202.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3202_sparse_potts_paoa_thrml_factor_boundary_v1.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-HW-100: build sparse q-state factor records from local artifacts only."""

    root_path = Path(root)
    exp3188 = read_json_object(root_path / EXP3188_REL_PATH)
    exp3197 = read_json_object(root_path / EXP3197_REL_PATH)
    sources = source_artifacts(root_path)
    errors = source_errors(sources)
    records = [] if errors else build_factor_records(exp3188, exp3197)
    transcript_present = authenticated_hardware_transcript_present(exp3188, exp3197)
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "spec_refs": ["REQ-HW-100", "SCENARIO-HW-100"],
        "source_artifacts": sources,
        "source_errors": errors,
        "factor_record_schema": factor_record_schema(),
        "factor_records": records,
        "factor_record_count": len(records),
        "q_state_count_summary": q_state_count_summary(records),
        "graph_density_summary": graph_density_summary(records),
        "paoa_metadata_schema": paoa_metadata_schema(),
        "thrml_local_api_checked": thrml_local_api_checked(exp3188),
        "thrml_local_api_evidence": thrml_local_api_evidence(exp3188),
        "authenticated_hardware_transcript_present": transcript_present,
        "speedup_claim_allowed": False,
        "hardware_claims_denied": hardware_claim_denials(transcript_present),
        "inference_substrate": inference_substrate(exp3188),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(errors, records, transcript_present)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the schema-versioned Exp 3202 artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_factor_records(exp3188: Mapping[str, Any], exp3197: Mapping[str, Any]) -> list[JsonDict]:
    """Create one sparse Potts factor record per exact row and invariant certificate."""

    translation_by_row = {
        str(row.get("row_id") or ""): dict(row)
        for row in mapping_rows(exp3188.get("factor_graph_translation_records"))
    }
    exact_records = [
        build_exact_row_factor_record(row, translation_by_row.get(str(row.get("row_id") or ""), {}))
        for row in mapping_rows(exp3188.get("selected_exact_rows"))
    ]
    invariant_records = [
        build_invariant_factor_record(row)
        for row in mapping_rows(exp3197.get("invariant_records"))
    ]
    return exact_records + invariant_records


def build_exact_row_factor_record(
    row: Mapping[str, Any],
    translation: Mapping[str, Any],
) -> JsonDict:
    """Represent an exact-authority row as sparse categorical Potts couplings."""

    row_id = str(row.get("row_id") or "")
    labels = ordered_labels([row.get("exact_label"), *list_values(row.get("candidate_answers"))])
    variables = categorical_variables(labels, ("candidate_label", "exact_label"))
    coupling_entries = alignment_couplings(labels) + false_accept_couplings(row, labels)
    return {
        "record_id": f"potts-exact:{row_id}",
        "source_kind": "exact_row",
        "row_id": row_id,
        "lineage": {
            "source_artifact": row.get("source_artifact") or EXP3188_REL_PATH.as_posix(),
            "thrml_boundary_artifact": EXP3188_REL_PATH.as_posix(),
            "thrml_translation_present": bool(translation),
        },
        "q_state_count": len(labels),
        "state_labels": labels,
        "variables": variables,
        "sparse_scope": [variable["name"] for variable in variables],
        "factor_kind": "potts_exact_alignment_and_false_accept_penalty",
        "coupling_entries": coupling_entries,
        "dense_energy_slot_count": len(labels) ** 2 + len(labels),
        "sparse_nonzero_energy_entry_count": len(coupling_entries),
        "paoa_metadata": paoa_metadata("exact_row", row_id, "exact_authority"),
    }


def build_invariant_factor_record(row: Mapping[str, Any]) -> JsonDict:
    """Represent an ExVerus-style invariant certificate as a sparse Potts guard."""

    row_id = str(row.get("row_id") or "")
    record_id = str(row.get("record_id") or row_id)
    guard = dict(row.get("exact_guard") or {})
    observed = dict(row.get("observed_counterexample") or {})
    labels = ordered_labels(
        [
            row.get("exact_label"),
            observed.get("canonical_answer"),
            guard.get("canonical_answer"),
            guard.get("required_exact_label"),
            *list_values(observed.get("candidate_answers")),
            *list_values(guard.get("preview_candidate_domain")),
        ]
    )
    variables = categorical_variables(labels, ("candidate_label", "exact_guard_label"))
    coupling_entries = invariant_guard_couplings(labels, str(guard.get("required_exact_label") or ""))
    return {
        "record_id": f"potts-invariant:{record_id}",
        "source_kind": "invariant_certificate",
        "row_id": row_id,
        "lineage": {
            "source_artifact": row.get("source_artifact") or EXP3197_REL_PATH.as_posix(),
            "invariant_artifact": EXP3197_REL_PATH.as_posix(),
            "invariant_record_id": record_id,
        },
        "q_state_count": len(labels),
        "state_labels": labels,
        "variables": variables,
        "sparse_scope": [variable["name"] for variable in variables],
        "factor_kind": "potts_invariant_guard_penalty",
        "coupling_entries": coupling_entries,
        "dense_energy_slot_count": len(labels) ** 2,
        "sparse_nonzero_energy_entry_count": len(coupling_entries),
        "invariant_certificate": {
            "guard_id": guard.get("guard_id"),
            "required_exact_label": guard.get("required_exact_label"),
            "certificate_type": observed.get("certificate_type"),
            "statement": dict(row.get("generalized_invariant") or {}).get("statement"),
            "anti_overfit_test_id": dict(row.get("anti_overfit_test") or {}).get("test_id"),
        },
        "paoa_metadata": paoa_metadata(
            "invariant_certificate",
            row_id,
            str(row.get("row_family") or "invariant_certificate"),
        ),
    }


def ordered_labels(values: Sequence[Any]) -> list[str]:
    """Return stable non-empty categorical state labels for q-state records."""

    labels = sorted({str(value) for value in values if str(value or "").strip()})
    return labels if labels else ["INVALID", "VALID"]


def list_values(value: Any) -> list[Any]:
    """Normalize optional JSON arrays into value lists."""

    return list(value) if isinstance(value, list) else []


def categorical_variables(labels: Sequence[str], names: Sequence[str]) -> list[JsonDict]:
    """Build the variable descriptors shared by Potts and PAOA metadata."""

    return [
        {
            "name": name,
            "kind": "categorical_potts_state",
            "q": len(labels),
            "state_labels": list(labels),
        }
        for name in names
    ]


def alignment_couplings(labels: Sequence[str]) -> list[JsonDict]:
    """Encode dense equality as sparse non-zero penalties for mismatched states."""

    return [
        {
            "source_variable": "candidate_label",
            "source_state": candidate,
            "target_variable": "exact_label",
            "target_state": exact,
            "energy": 1.0,
            "reason": "candidate label differs from exact authority",
        }
        for candidate in labels
        for exact in labels
        if candidate != exact
    ]


def false_accept_couplings(row: Mapping[str, Any], labels: Sequence[str]) -> list[JsonDict]:
    """Add sparse unary penalties for known false-accept candidate labels."""

    if row.get("known_false_accept") is not True:
        return []
    exact_label = str(row.get("exact_label") or "")
    candidates = {str(value) for value in list_values(row.get("candidate_answers"))}
    return [
        {
            "source_variable": "candidate_label",
            "source_state": label,
            "target_variable": None,
            "target_state": None,
            "energy": 2.0,
            "reason": "known false-accept candidate conflicts with exact authority",
        }
        for label in labels
        if label in candidates and label != exact_label
    ]


def invariant_guard_couplings(labels: Sequence[str], required_label: str) -> list[JsonDict]:
    """Encode an invariant guard as sparse penalties away from the required label."""

    return [
        {
            "source_variable": "candidate_label",
            "source_state": label,
            "target_variable": "exact_guard_label",
            "target_state": required_label,
            "energy": 1.5,
            "reason": "candidate violates invariant guard required label",
        }
        for label in labels
        if required_label and label != required_label
    ]


def paoa_metadata(source_kind: str, row_id: str, constraint_family: str) -> JsonDict:
    """Attach derivative-free PAOA adapter metadata without running a sampler."""

    return {
        "boundary_only": True,
        "source_kind": source_kind,
        "row_id": row_id,
        "constraint_family": constraint_family,
        "coupling_format": "sparse_categorical_triplets",
        "update_rule": "categorical_single_site_resample",
        "update_order": "row_id_then_variable_index",
        "schedule": {
            "kind": "placeholder_beta_schedule",
            "executed": False,
            "values": [],
        },
        "sampler_execution": "not_run",
        "hardware_execution": "not_run",
    }


def q_state_count_summary(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize q-state accounting without collapsing Potts states to binary."""

    counts = [int(record.get("q_state_count") or 0) for record in records]
    labels = sorted(
        {
            str(label)
            for record in records
            for label in list_values(record.get("state_labels"))
        }
    )
    return {
        "record_count": len(records),
        "min_q_state_count": min(counts) if counts else 0,
        "max_q_state_count": max(counts) if counts else 0,
        "total_record_state_count": sum(counts),
        "unique_state_labels": labels,
        "records_by_q_state_count": {
            str(count): counts.count(count) for count in sorted(set(counts))
        },
    }


def graph_density_summary(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Estimate sparse non-zero entries against dense q-state energy tables."""

    dense_slots = sum(int(record.get("dense_energy_slot_count") or 0) for record in records)
    sparse_entries = sum(
        int(record.get("sparse_nonzero_energy_entry_count") or 0) for record in records
    )
    return {
        "factor_record_count": len(records),
        "total_variable_count": sum(len(list_values(record.get("variables"))) for record in records),
        "dense_energy_slot_count": dense_slots,
        "sparse_nonzero_energy_entry_count": sparse_entries,
        "sparse_density_ratio": round(sparse_entries / dense_slots, 6) if dense_slots else 0.0,
        "sparse_vs_dense_slot_delta": dense_slots - sparse_entries,
        "dense_float64_bytes_estimate": dense_slots * 8,
        "sparse_triplet_bytes_estimate": sparse_entries * 24,
        "per_record": [
            {
                "record_id": record.get("record_id"),
                "q_state_count": record.get("q_state_count"),
                "dense_energy_slot_count": record.get("dense_energy_slot_count"),
                "sparse_nonzero_energy_entry_count": record.get(
                    "sparse_nonzero_energy_entry_count"
                ),
            }
            for record in records
        ],
    }


def factor_record_schema() -> JsonDict:
    """Document the structured q-state factor-record boundary."""

    return {
        "type": "object",
        "required": [
            "record_id",
            "source_kind",
            "row_id",
            "q_state_count",
            "state_labels",
            "variables",
            "sparse_scope",
            "coupling_entries",
            "paoa_metadata",
        ],
        "properties": {
            "record_id": "stable factor record id",
            "source_kind": "exact_row or invariant_certificate",
            "row_id": "exact row or certificate row id",
            "q_state_count": "number of categorical Potts states",
            "state_labels": "ordered categorical state vocabulary",
            "variables": "categorical Potts variables with q and labels",
            "sparse_scope": "variables touched by this sparse factor",
            "coupling_entries": "non-zero sparse categorical energy entries",
            "paoa_metadata": "adapter metadata for future PAOA update construction",
        },
    }


def paoa_metadata_schema() -> JsonDict:
    """Document PAOA-ready metadata without implying execution."""

    return {
        "type": "object",
        "required": [
            "boundary_only",
            "coupling_format",
            "update_rule",
            "update_order",
            "schedule",
            "sampler_execution",
            "hardware_execution",
        ],
        "properties": {
            "coupling_format": "sparse categorical energy triplets",
            "update_rule": "categorical single-site update metadata",
            "update_order": "deterministic adapter ordering",
            "schedule": "non-executed beta or annealing schedule placeholder",
            "constraint_family": "exact row or invariant family provenance",
            "sampler_execution": "must remain not_run in this artifact",
            "hardware_execution": "must remain not_run in this artifact",
        },
    }


def thrml_local_api_checked(exp3188: Mapping[str, Any]) -> bool:
    """Treat Exp 3188 as the local THRML API transparency source."""

    return "thrml_import_available" in exp3188 or "local_api_smoke_passed" in exp3188


def thrml_local_api_evidence(exp3188: Mapping[str, Any]) -> JsonDict:
    """Carry forward local API evidence while separating it from hardware claims."""

    return {
        "source_artifact": EXP3188_REL_PATH.as_posix(),
        "thrml_import_available": exp3188.get("thrml_import_available"),
        "local_api_smoke_passed": exp3188.get("local_api_smoke_passed"),
        "selected_exact_row_count": len(mapping_rows(exp3188.get("selected_exact_rows"))),
        "claim_boundary": "local software API construction only; no TSU, Kona, latency, or speedup",
    }


def authenticated_hardware_transcript_present(*payloads: Mapping[str, Any]) -> bool:
    """Return true only when checked-in payloads expose authenticated transcript hashes."""

    return any(
        valid_authenticated_transcript(payload.get("authenticated_hardware_transcript"))
        or payload.get("authenticated_hardware_transcript_present") is True
        for payload in payloads
    )


def valid_authenticated_transcript(value: Any) -> bool:
    """Validate the minimal shape needed before hardware claims can even be considered."""

    return (
        isinstance(value, Mapping)
        and value.get("present") is True
        and isinstance(value.get("sha256"), str)
        and len(str(value.get("sha256"))) == 64
    )


def hardware_claim_denials(transcript_present: bool) -> list[JsonDict]:
    """Explicitly deny hardware/speedup claims when transcript authority is absent."""

    return [
        {
            "claim": claim,
            "denied": not transcript_present,
            "reason": "no authenticated hardware transcript present"
            if not transcript_present
            else "authenticated transcript present, but this artifact still performs no execution",
        }
        for claim in DENIED_HARDWARE_CLAIMS
    ]


def inference_substrate(exp3188: Mapping[str, Any]) -> JsonDict:
    """Expose that Exp 3202 is a representation task, not a run."""

    return {
        "kind": "local_factor_record_only_no_hardware_speedup",
        "local_repo_only": True,
        "executes_hardware": False,
        "hardware_commands_run": [],
        "board_commands_run": [],
        "executes_models": False,
        "no_live_model_inference": True,
        "installs_packages": False,
        "sampler_benchmark_run": False,
        "sampler_speedup_reported": False,
        "thrml_boundary_source_local_api_smoke_passed": exp3188.get("local_api_smoke_passed"),
        "kv260_execution_claimed": False,
        "gatemate_execution_claimed": False,
        "polarfire_execution_claimed": False,
        "tsu_z1_xtr0_kona_execution_claimed": False,
    }


def honest_verdict(
    errors: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    transcript_present: bool,
) -> str:
    """Return a terminal verdict that preserves the no-speedup boundary."""

    if errors:
        return "blocked_missing_source: required local source artifacts are missing or malformed"
    if not records:
        return "blocked_empty_scope: no exact rows or invariant certificates available"
    transcript_note = "authenticated_transcript_present" if transcript_present else "no_transcript"
    return (
        "complete: sparse Potts/PAOA/THRML factor boundary materialized; "
        f"factor_record_count={len(records)}; {transcript_note}; speedup_claim_allowed=false"
    )


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return auditable lineage for every local source used by the artifact."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_REL_PATHS:
        path = root / rel_path
        readable = bool(read_json_object(path)) if source_type == "json" else None
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "present": path.is_file(),
                "readable_structured_source": readable,
                "sha256": sha256_file(path),
            }
        )
    return rows


def source_errors(sources: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Fail closed on missing or malformed required local evidence."""

    errors: list[JsonDict] = []
    for row in sources:
        if row.get("required") is True and row.get("present") is not True:
            errors.append({"path": str(row.get("path")), "reason": "missing_required_source"})
        elif (
            row.get("required") is True
            and row.get("source_type") == "json"
            and row.get("readable_structured_source") is not True
        ):
            errors.append({"path": str(row.get("path")), "reason": "malformed_required_source"})
    return errors


def mapping_rows(value: Any) -> list[JsonDict]:
    """Normalize a JSON array into object rows only."""

    return [dict(row) for row in value] if isinstance(value, list) else []


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and treat absent or non-object evidence as unavailable."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash available source files so downstream matrix rows can verify lineage."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail fast if a required Exp 3202 field is absent."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required Exp 3202 artifact fields: {sorted(missing)}")
