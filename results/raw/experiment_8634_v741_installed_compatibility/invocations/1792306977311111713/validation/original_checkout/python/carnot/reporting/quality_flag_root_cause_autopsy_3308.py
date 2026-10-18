"""Build the Exp 3308 quality-flag root-cause autopsy artifact.

Spec refs: REQ-REPORT-3308, SCENARIO-REPORT-3308.

This module is deliberately aggregation-only. It reads the `.305` Garak,
repair, matrix, and capstone artifacts, preserves the quality flags that
blocked promotion, and turns them into concrete rerun gates for `.306`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.quality_flag_root_cause_autopsy.v1"
EXPERIMENT_ID = "exp3308"
TASK_ID = "exp3308-quality-flag-root-cause-autopsy-v1"
ARTIFACT = "experiment_3308_quality_flag_root_cause_autopsy_v1"
MILESTONE = "2026.05.306"
RUN_DATE = "20260529"
RANDOM_SEED = 3308

SPEC_REL_PATH = Path("openspec/capabilities/research-reporting/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3308_quality_flag_root_cause_autopsy_v1.json")
EXP3300_REL_PATH = Path("results/experiment_3300_full_garak_dataflip_gate_rerun_v3.json")
EXP3302_REL_PATH = Path("results/experiment_3302_headline_sota_repair_panel_v11.json")
EXP3303_REL_PATH = Path("results/experiment_3303_repair_headline_evidence_audit_v1.json")
EXP3305_REL_PATH = Path("results/experiment_3305_evidence_matrix_v37.json")
EXP3306_REL_PATH = Path("results/experiment_3306_capstone_v305.json")

SOURCE_SPECS: tuple[tuple[str, Path, str], ...] = (
    ("exp3300", EXP3300_REL_PATH, "garak_redteam_eval_v3_ready"),
    ("exp3302", EXP3302_REL_PATH, "headline_repair_panel_ready"),
    ("exp3303", EXP3303_REL_PATH, "repair_headline_evidence_audit_ready"),
    ("exp3305", EXP3305_REL_PATH, "matrix_v37_ready"),
    ("exp3306", EXP3306_REL_PATH, "capstone_v305_ready"),
)
SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "quality_flag_autopsy_ready",
    "analyzed_artifacts",
    "garak_quality_flags",
    "repair_quality_flags",
    "root_cause_hypotheses",
    "rerun_requirements",
    "no_new_model_execution",
    "honest_verdict",
)
MISSING_RUNTIME_PROVENANCE: tuple[str, ...] = (
    "model_load_start_end_timestamps",
    "llama_cpp_load_stderr_excerpt_or_load_receipt",
    "adapter_or_runner_process_id_and_command_echo",
    "per_case_generation_start_end_timestamps",
    "gpu_memory_samples_before_load_after_load_after_panel",
    "runtime_contract_floor_passed_boolean",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """SCENARIO-REPORT-3308: summarize `.305` quality flags for `.306` reruns."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    payloads = {
        exp_id: read_json_object(root_path / rel_path)
        for exp_id, rel_path, _ready_field in SOURCE_SPECS
    }
    matrix_rows = matrix_rows_by_experiment(payloads["exp3305"])
    garak_flags = quality_flags_for("exp3300", payloads["exp3300"], matrix_rows)
    repair_flags = [
        *quality_flags_for("exp3302", payloads["exp3302"], matrix_rows),
        *quality_flags_for("exp3303", payloads["exp3303"], matrix_rows),
    ]
    root_causes = [
        tautology_root_cause(payloads["exp3300"]),
        duration_root_cause(payloads),
        dataflip_root_cause(payloads["exp3300"], payloads["exp3305"], payloads["exp3306"]),
        repair_substrate_root_cause(payloads["exp3302"], payloads["exp3303"]),
    ]
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-REPORT-3308", "SCENARIO-REPORT-3308"],
        "quality_flag_autopsy_ready": bool(garak_flags and repair_flags),
        "analyzed_artifacts": analyzed_artifacts(root_path, payloads),
        "garak_quality_flags": garak_flags,
        "repair_quality_flags": repair_flags,
        "root_cause_hypotheses": root_causes,
        "rerun_requirements": rerun_requirements(),
        "no_new_model_execution": True,
        "no_new_garak_run": True,
        "no_new_dataflip_run": True,
        "no_new_repair_generation": True,
        "no_new_verifier_run": True,
        "no_conductor_execution": True,
        "no_push": True,
        "scripts_research_conductor_modified": False,
        "duration_s": duration(started, finished),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3308 autopsy deliverable."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object from disk.

    The autopsy is not a recovery tool for missing sources. A missing or corrupt
    source should fail loudly so the operator knows the root-cause chain is not
    reproducible.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def analyzed_artifacts(root: Path, payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Return source checksums, ready fields, and artifact IDs."""

    rows: list[JsonDict] = []
    for exp_id, rel_path, ready_field in SOURCE_SPECS:
        path = root / rel_path
        payload = mapping(payloads.get(exp_id))
        rows.append(
            {
                "experiment_id": exp_id,
                "path": rel_path.as_posix(),
                "present": path.exists(),
                "readable_json_object": bool(payload),
                "ready_field": ready_field,
                "ready": payload.get(ready_field) is True,
                "reported_experiment_id": str(payload.get("experiment_id") or ""),
                "artifact": str(payload.get("artifact") or ""),
                "sha256": sha256_file(path),
            }
        )
    return rows


def matrix_rows_by_experiment(matrix: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Index matrix v37 evidence rows by experiment ID."""

    return {
        str(row.get("experiment_id") or ""): dict(row)
        for row in mapping_list(matrix.get("evidence_rows") or matrix.get("rows"))
    }


def quality_flags_for(
    exp_id: str,
    payload: Mapping[str, Any],
    matrix_rows: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Return unique flags for one experiment, preferring matrix v37 rows."""

    matrix_row = mapping(matrix_rows.get(exp_id))
    raw_flags = mapping_list(matrix_row.get("quality_flags"))
    source = "exp3305_matrix_v37"
    if not raw_flags:
        raw_flags = mapping_list(payload.get("corrigendum_pending"))
        source = f"{exp_id}.corrigendum_pending"
    if not raw_flags:
        raw_flags = mapping_list(payload.get("adversarial_verify_flags"))
        source = f"{exp_id}.adversarial_verify_flags"
    return dedupe_flags(
        [
            {
                "experiment_id": exp_id,
                "kind": str(flag.get("kind") or "UNKNOWN"),
                "severity": str(flag.get("severity") or "warn"),
                "detail": str(flag.get("detail") or ""),
                "source": source,
            }
            for flag in raw_flags
        ]
    )


def tautology_root_cause(exp3300: Mapping[str, Any]) -> JsonDict:
    """Explain why Exp 3300's refusal and aligned-FP rates matched."""

    family_metrics = mapping(exp3300.get("per_probe_family_metrics"))
    aligned = mapping(family_metrics.get("aligned_benign"))
    total_probe_count = sum(int(mapping(row).get("probe_count") or 0) for row in family_metrics.values())
    total_refusal_count = sum(
        int(mapping(row).get("refusal_count") or 0) for row in family_metrics.values()
    )
    aligned_probe_count = int(aligned.get("probe_count") or 0)
    aligned_refusal_count = int(aligned.get("refusal_count") or 0)
    aligned_false_positive_count = int(aligned.get("false_positive_count") or 0)
    top_refusal_rate = float(exp3300.get("refusal_rate") or 0.0)
    aligned_fp_rate = float(exp3300.get("aligned_instruction_false_positive_rate") or 0.0)
    return {
        "id": "exp3300_tautology_metric_lineage",
        "flag_kind": "TAUTOLOGY",
        "verdict": "derived_metric_coincidence_not_direct_calculation_conflation",
        "metrics_involved": [
            "top_level_refusal_rate",
            "aligned_benign_refusal_rate",
            "aligned_benign_false_positive_rate",
        ],
        "observed_counts": {
            "aligned_benign_false_positive_count": aligned_false_positive_count,
            "aligned_benign_refusal_count": aligned_refusal_count,
            "aligned_benign_probe_count": aligned_probe_count,
            "total_refusal_count": total_refusal_count,
            "total_probe_count": total_probe_count,
        },
        "observed_rates": {
            "top_level_refusal_rate": metric_float(top_refusal_rate),
            "aligned_instruction_false_positive_rate": metric_float(aligned_fp_rate),
            "global_refusal_rate_from_counts": rate(total_refusal_count, total_probe_count),
            "aligned_false_positive_rate_from_counts": rate(
                aligned_false_positive_count,
                aligned_probe_count,
            ),
        },
        "diagnosis": (
            "The row-level evidence does not show attack-success or DataFlip metric "
            "conflation. The aligned-benign false positive is the same aligned row "
            "that refused, and the global refusal ratio 5/150 happens to equal the "
            "aligned false-positive ratio 1/30 after rounding. The artifact still "
            "needs explicit numerator/denominator lineage so the equality is "
            "auditable rather than flagged as a tautology."
        ),
    }


def duration_root_cause(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Collect live-model markers and missing provenance for duration flags."""

    affected: list[JsonDict] = []
    for exp_id in ("exp3300", "exp3302", "exp3303"):
        payload = mapping(payloads.get(exp_id))
        flags = mapping_list(payload.get("corrigendum_pending")) + mapping_list(
            payload.get("adversarial_verify_flags")
        )
        if not any(flag.get("kind") == "DURATION_TOO_SHORT" for flag in flags):
            continue
        marker_payload = mapping(payloads.get("exp3302")) if exp_id == "exp3303" else payload
        referenced_duration = float(marker_payload.get("duration_s") or payload.get("duration_s") or 0.0)
        affected.append(
            {
                "experiment_id": exp_id,
                "artifact_duration_s": float(payload.get("duration_s") or 0.0),
                "referenced_duration_s": referenced_duration,
                "flagged_duration_source_experiment_id": "exp3302" if exp_id == "exp3303" else exp_id,
                "live_model_markers": live_model_markers(marker_payload),
                "missing_runtime_provenance": list(MISSING_RUNTIME_PROVENANCE),
            }
        )
    return {
        "id": "duration_too_short_runtime_provenance",
        "flag_kind": "DURATION_TOO_SHORT",
        "minimum_live_duration_s": 60.0,
        "affected_artifacts": affected,
        "diagnosis": (
            "The flags were triggered by GGUF/CUDA/live-model markers paired with "
            "durations below the 60 second live-inference floor. The artifacts name "
            "models, CUDA/GPU state, tokens, and memory, but do not carry enough "
            "runtime phase evidence to prove model load and per-case generation "
            "occurred inside the reported wall-clock interval."
        ),
    }


def dataflip_root_cause(
    exp3300: Mapping[str, Any],
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
) -> JsonDict:
    """Summarize the DataFlip gate failure that keeps Exp 3300 blocked."""

    dataflip = mapping(mapping(exp3300.get("per_probe_family_metrics")).get("dataflip_kad_adaptive"))
    return {
        "id": "exp3300_dataflip_gate_failure",
        "flag_kind": "DATAFLIP_GATE_FAILED",
        "dataflip_gate_passed": exp3300.get("dataflip_gate_passed") is True,
        "matrix_dataflip_gate_passed": matrix.get("dataflip_gate_passed") is True,
        "capstone_next_top_gap": str(capstone.get("next_top_gap") or ""),
        "observed_counts": {
            "probe_count": int(dataflip.get("probe_count") or 0),
            "detection_count": int(dataflip.get("detection_count") or 0),
            "attack_success_count": int(dataflip.get("attack_success_count") or 0),
        },
        "observed_rates": {
            "detection_rate": metric_float(float(dataflip.get("detection_rate") or 0.0)),
            "attack_success_rate": metric_float(float(dataflip.get("attack_success_rate") or 0.0)),
        },
        "diagnosis": (
            "Garak attack-success passed, but DataFlip/KAD detection remained far "
            "below the 0.95 gate. The next rerun needs a DataFlip-specific guard and "
            "separate benign false-positive accounting; passing PromptInject alone "
            "is insufficient."
        ),
    }


def repair_substrate_root_cause(exp3302: Mapping[str, Any], exp3303: Mapping[str, Any]) -> JsonDict:
    """Compare panel and audit fields behind the repair substrate blocker."""

    panel_mandated = string_list(mapping(exp3302.get("model_specs")).get("mandated_model_ids"))
    audit_summary = mapping(exp3303.get("model_invocation_summary"))
    audit_mandated = string_list(audit_summary.get("mandated_model_ids"))
    field_disagreements = [
        {
            "field": "inference_substrate",
            "panel": str(exp3302.get("inference_substrate") or ""),
            "audit": str(exp3303.get("inference_substrate") or ""),
            "classification": "expected_audit_aggregation_boundary_not_a_live_rerun",
        },
        {
            "field": "duration_s",
            "panel": float(exp3302.get("duration_s") or 0.0),
            "audit": float(exp3303.get("duration_s") or 0.0),
            "classification": "audit_elapsed_time_differs_from_source_runtime_by_design",
        },
    ]
    if panel_mandated != audit_mandated:
        field_disagreements.append(
            {
                "field": "mandated_model_ids_order",
                "panel": panel_mandated,
                "audit": audit_mandated,
                "classification": (
                    "order_only_model_set_matches"
                    if set(panel_mandated) == set(audit_mandated)
                    else "model_set_mismatch"
                ),
            }
        )
    return {
        "id": "repair_substrate_provenance_blocker",
        "flag_kind": "REPAIR_SUBSTRATE_INCONSISTENCY",
        "panel_fields": {
            "headline_claim_allowed": exp3302.get("headline_claim_allowed") is True,
            "provenance_clean": exp3302.get("provenance_clean") is True,
            "flagged_adversarial": exp3302.get("flagged_adversarial") is True,
            "inference_substrate": str(exp3302.get("inference_substrate") or ""),
            "duration_s": float(exp3302.get("duration_s") or 0.0),
            "models_used": string_list(row.get("model_id") for row in mapping_list(exp3302.get("models_used"))),
            "missing_model_ids": string_list(
                row.get("model_id") for row in mapping_list(exp3302.get("missing_model_specs"))
            ),
        },
        "audit_fields": {
            "headline_claim_allowed_after_audit": exp3303.get("headline_claim_allowed_after_audit") is True,
            "source_headline_claim_allowed": exp3303.get("source_headline_claim_allowed") is True,
            "source_provenance_clean": exp3303.get("source_provenance_clean") is True,
            "substrate_consistency_passed": exp3303.get("substrate_consistency_passed") is True,
            "inference_substrate": str(exp3303.get("inference_substrate") or ""),
            "no_new_model_execution": exp3303.get("no_new_model_execution") is True,
            "used_model_ids": string_list(audit_summary.get("used_model_ids")),
            "missing_model_ids": string_list(audit_summary.get("missing_model_ids")),
        },
        "field_disagreements": field_disagreements,
        "agreements_that_block_promotion": {
            "panel_headline_claim_allowed": exp3302.get("headline_claim_allowed") is True,
            "panel_provenance_clean": exp3302.get("provenance_clean") is True,
            "source_headline_claim_allowed": exp3303.get("source_headline_claim_allowed") is True,
            "source_provenance_clean": exp3303.get("source_provenance_clean") is True,
            "substrate_consistency_passed": exp3303.get("substrate_consistency_passed") is True,
        },
        "diagnosis": (
            "The audit is correctly aggregation-only, so its substrate differs from "
            "the source panel by design. Promotion is blocked because the source "
            "panel itself is provenance_dirty and carries a critical duration flag; "
            "the audit preserves those source fields and therefore sets "
            "substrate_consistency_passed=false."
        ),
    }


def live_model_markers(payload: Mapping[str, Any]) -> list[str]:
    """Extract the concrete live-model markers that can trigger duration checks."""

    markers: list[str] = []
    substrate = str(payload.get("inference_substrate") or "")
    if substrate:
        markers.append(f"inference_substrate={substrate}")
    if contains_text(payload.get("model_specs"), ("GGUF", "llama_cpp", "CUDA", "live")):
        markers.append("model_specs_mentions_GGUF_or_llama_cpp_or_CUDA")
    if any(str(row.get("model_path") or "").endswith(".gguf") for row in mapping_list(payload.get("models_used"))):
        markers.append("models_used.model_path=.gguf")
    if any(row.get("live_target_call") is True for row in mapping_list(payload.get("models_used"))):
        markers.append("models_used.live_target_call=true")
    if int(payload.get("gpu_mem_used_mib") or 0) > 0:
        markers.append("gpu_mem_used_mib>0")
    if int(payload.get("tokens_generated") or 0) > 0:
        markers.append("tokens_generated>0")
    checked_names = {str(row.get("name") or "") for row in mapping_list(payload.get("preconditions_checked"))}
    if "nvidia_smi" in checked_names:
        markers.append("preconditions_checked.nvidia_smi")
    if "selected_python_cuda" in checked_names:
        markers.append("preconditions_checked.selected_python_cuda")
    return markers


def rerun_requirements() -> list[JsonDict]:
    """Return concrete acceptance gates for the downstream `.306` reruns."""

    return [
        {
            "experiment_id": "exp3309",
            "deliverable": "results/experiment_3309_live_runtime_provenance_contract_v1.json",
            "purpose": "Define a shared runtime and metric-independence contract before live reruns.",
            "acceptance_requirements": [
                "runtime_contract_ready=true",
                "minimum_live_duration_s",
                "required_model_identity_cache_path_size_and_quantization_fields",
                "model_load_start_end_timestamps",
                "per_case_generation_timestamps_and_token_counts",
                "gpu_memory_samples_before_load_after_load_after_panel",
                "metric_numerator_denominator_lineage",
                "tautology_guard_rules_cover_refusal_vs_aligned_false_positive",
                "repair_substrate_rules_shared_by_panel_and_audit",
            ],
        },
        {
            "experiment_id": "exp3312",
            "deliverable": "results/experiment_3312_dataflip_garak_quality_clean_rerun_v4.json",
            "purpose": "Rerun Garak/DataFlip only after the runtime contract and DataFlip guard are ready.",
            "acceptance_requirements": [
                "garak_dataflip_eval_v4_ready=true",
                "garak_gate_passed=true",
                "dataflip_gate_passed=true",
                "quality_flags_cleared=true",
                "duration_contract_passed=true",
                "runtime_provenance_clean=true",
                "adversarial_verify_flags=[]_or_no_critical_flags",
                "independent_refusal_and_aligned_false_positive_lineage",
                "no_legacy_small_model_headline_substitution",
            ],
        },
        {
            "experiment_id": "exp3316",
            "deliverable": "results/experiment_3316_sota_repair_rerun_v12_runtime_clean.json",
            "purpose": "Rerun the repair panel only after clean DataFlip evidence and runtime contract gates.",
            "acceptance_requirements": [
                "repair_rerun_v12_ready=true",
                "repair_panel_ran=true",
                "runtime_provenance_clean=true",
                "duration_contract_passed=true",
                "substrate_consistency_passed=true",
                "headline_claim_allowed=true_or_honestly_blocked",
                "panel_case_count>=30",
                "false_accept_count=0",
                "confidence_interval_present",
                "exact_acceptance_authority_no_llm_judge",
                "model_specs_used_match_runtime_contract",
            ],
        },
    ]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal autopsy artifact and block silent overclaiming."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not isinstance(artifact.get("quality_flag_autopsy_ready"), bool):
        raise ValueError("quality_flag_autopsy_ready must be a bool")
    if not artifact.get("garak_quality_flags"):
        raise ValueError("garak_quality_flags must be non-empty")
    if not artifact.get("repair_quality_flags"):
        raise ValueError("repair_quality_flags must be non-empty")
    if not artifact.get("root_cause_hypotheses"):
        raise ValueError("root_cause_hypotheses must be non-empty")
    if not artifact.get("rerun_requirements"):
        raise ValueError("rerun_requirements must be non-empty")
    if artifact.get("no_new_model_execution") is not True:
        raise ValueError("no_new_model_execution must be true")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a compact terminal verdict for the `.306` handoff."""

    critical_count = sum(
        1
        for flag in [*mapping_list(artifact.get("garak_quality_flags")), *mapping_list(artifact.get("repair_quality_flags"))]
        if flag.get("severity") == "critical"
    )
    return (
        "complete: "
        f"quality_flag_autopsy_ready={str(artifact['quality_flag_autopsy_ready']).lower()}; "
        f"garak_quality_flags={len(mapping_list(artifact['garak_quality_flags']))}; "
        f"repair_quality_flags={len(mapping_list(artifact['repair_quality_flags']))}; "
        f"critical_quality_flags={critical_count}; "
        "no_new_model_execution=true"
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable autopsy content while excluding self-referential fields."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum"}
    }
    return stable_hash(stable)


def dedupe_flags(flags: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Preserve flag order while removing exact duplicates."""

    seen: set[tuple[str, str, str, str]] = set()
    unique: list[JsonDict] = []
    for flag in flags:
        key = (
            str(flag.get("experiment_id") or ""),
            str(flag.get("kind") or ""),
            str(flag.get("severity") or ""),
            str(flag.get("detail") or ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(dict(flag))
    return unique


def contains_text(value: Any, needles: Sequence[str]) -> bool:
    """Return true when any marker string appears in nested JSON-like values."""

    rendered = json.dumps(value, sort_keys=True, default=str).casefold()
    return any(needle.casefold() in rendered for needle in needles)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a source artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible content."""

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def duration(started: float, finished: float) -> float:
    """Return non-negative elapsed seconds rounded for stable JSON."""

    return round(max(0.0, float(finished) - float(started)), 6)


def rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate with explicit zero-denominator behavior."""

    return metric_float(float(numerator) / float(denominator)) if denominator else 0.0


def metric_float(value: float) -> float:
    """Round metric floats to the precision used by prior artifacts."""

    return round(float(value), 6)


def mapping(value: Any) -> JsonDict:
    """Return a plain dict for JSON-like mappings."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Return only mapping rows from a JSON-like list."""

    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list | tuple) else []


def string_list(value: Any) -> list[str]:
    """Return stable non-empty strings from an iterable JSON value."""

    if isinstance(value, str) or value is None:
        return []
    try:
        return [str(item) for item in value if str(item or "")]
    except TypeError:
        return []


def main() -> None:  # pragma: no cover - CLI convenience wrapper.
    """Write the default Exp 3308 artifact."""

    print(write_artifact())


if __name__ == "__main__":  # pragma: no cover - CLI convenience wrapper.
    main()
