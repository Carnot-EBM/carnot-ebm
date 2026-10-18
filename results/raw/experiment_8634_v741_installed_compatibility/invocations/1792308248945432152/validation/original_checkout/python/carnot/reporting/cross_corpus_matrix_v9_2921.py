"""Build the Exp 2921 cross-corpus matrix v9 and paper boundary artifact.

Spec refs: REQ-REPORT-2921, SCENARIO-REPORT-2921.

This module is deliberately an aggregation layer. It does not rerun models,
hardware, or verifiers; it reads the artifacts that already exist and records
which rows can support bounded paper-v6 claims without letting flagged,
blocked, pilot-only, or simulator-only evidence drift into headline rows.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
SCHEMA = "carnot.cross_corpus_matrix.v9_paper_boundary.v1"
ARTIFACT = "experiment_2921_cross_corpus_matrix_v9_paper_boundary_v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2921_cross_corpus_matrix_v9_paper_boundary_v1.json")

MATRIX_V8_REL_PATH = Path("results/experiment_2902_cross_corpus_matrix_v8.json")
EXP2910_REL_PATH = Path("results/experiment_2910_sota_code_generation_corrigendum_v2.json")
EXP2911_REL_PATH = Path("results/experiment_2911_code_hallucination_taxonomy_verifier_v1.json")
EXP2912_REL_PATH = Path("results/experiment_2912_kv260_same_basis_cpu_gibbs_baseline_v1.json")
EXP2913_REL_PATH = Path("results/experiment_2913_kv260_hardware_cpu_claim_boundary_v1.json")
EXP2914_REL_PATH = Path("results/experiment_2914_gatemate_toolchain_preflight_v2.json")
EXP2915_REL_PATH = Path("results/experiment_2915_gatemate_n16_ising_tile_bitstream_build_v2.json")
EXP2916_REL_PATH = Path("results/experiment_2916_thrml_kv260_sampler_parity_v1.json")
EXP2917_REL_PATH = Path("results/experiment_2917_spilled_energy_logit_detector_micro_panel_v1.json")
EXP2918_REL_PATH = Path(
    "results/experiment_2918_fr11_verifiable_process_rewards_self_learning_v1.json"
)
EXP2919_REL_PATH = Path("results/experiment_2919_constraintbench_mini_direct_optimization_v1.json")
EXP2920_REL_PATH = Path("results/experiment_2920_opencomputer_style_state_verifier_harness_v1.json")


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    row_id: str
    label: str
    path: Path
    fields_imported: tuple[str, ...]


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "exp2902",
        "exp2902_matrix_v8",
        "Cross-corpus matrix v8",
        MATRIX_V8_REL_PATH,
        (
            "honest_verdict",
            "inference_substrate",
            "matrix_rows",
            "rows_clean",
            "rows_flagged",
            "rows_blocked",
            "rows_pilot_only",
        ),
    ),
    SourceSpec(
        "exp2910",
        "exp2910_sota_codegen",
        "SOTA code-generation corrigendum",
        EXP2910_REL_PATH,
        (
            "honest_verdict",
            "inference_substrate",
            "codegen_corrigendum_ready",
            "candidate_generation_clean",
            "aggregate_pass_at_1",
            "aggregate_pass_at_k",
            "pass_at_k_exceeds_pass_at_1",
            "model_specs",
            "random_seed",
            "reproducibility_checksum",
        ),
    ),
    SourceSpec(
        "exp2911",
        "exp2911_code_hallucination_verifier",
        "Code hallucination taxonomy verifier",
        EXP2911_REL_PATH,
        (
            "honest_verdict",
            "inference_substrate",
            "code_hallucination_verifier_ready",
            "pass_rate_after_taxonomy_filter",
            "syntax_error_rate",
            "runtime_error_rate",
            "true_test_failure_rate",
            "undefined_name_rate",
            "flagged_adversarial",
            "corrigendum_pending",
        ),
    ),
    SourceSpec(
        "exp2912",
        "exp2912_kv260_cpu_baseline",
        "KV260 same-basis CPU Gibbs baseline",
        EXP2912_REL_PATH,
        (
            "honest_verdict",
            "inference_substrate",
            "same_basis_cpu_baseline_ready",
            "speedup_claim_made",
            "n_spins",
            "sample_count_sweep",
            "random_seeds_used",
        ),
    ),
    SourceSpec(
        "exp2913",
        "exp2913_kv260_claim_boundary",
        "KV260 CPU claim boundary",
        EXP2913_REL_PATH,
        (
            "honest_verdict",
            "inference_substrate",
            "kv260_claim_boundary_ready",
            "same_basis_verified",
            "hardware_speedup_claim_eligible",
            "speedup_claim_made",
            "matrix_row_candidate",
            "paper_claim_boundary",
        ),
    ),
    SourceSpec(
        "exp2914",
        "exp2914_gatemate_toolchain",
        "GateMate toolchain preflight",
        EXP2914_REL_PATH,
        (
            "honest_verdict",
            "inference_substrate",
            "gatemate_toolchain_ready",
            "missing_toolchain",
            "constraints_present",
            "rtl_sources_present",
            "no_flash_attempted",
        ),
    ),
    SourceSpec(
        "exp2915",
        "exp2915_gatemate_bitstream",
        "GateMate n16 bitstream build",
        EXP2915_REL_PATH,
        (
            "honest_verdict",
            "inference_substrate",
            "gatemate_bitstream_built",
            "bitstream_path",
            "hardware_flash_attempted",
        ),
    ),
    SourceSpec(
        "exp2916",
        "exp2916_thrml_parity",
        "THRML KV260 simulator parity",
        EXP2916_REL_PATH,
        (
            "honest_verdict",
            "inference_substrate",
            "thrml_kv260_parity_ready",
            "no_tsu_hardware_claim",
            "matched_full_n64_basis",
            "cpu_vs_thrml_distance",
            "kv260_vs_thrml_summary",
        ),
    ),
    SourceSpec(
        "exp2917",
        "exp2917_spilled_energy_micro_panel",
        "Spilled energy micro-panel",
        EXP2917_REL_PATH,
        (
            "honest_verdict",
            "inference_substrate",
            "spilled_energy_micro_panel_ready",
            "benchmark_claim_made",
            "claim_boundary",
            "separation_summary",
        ),
    ),
    SourceSpec(
        "exp2918",
        "exp2918_fr11_process_rewards",
        "FR-11 process rewards self-learning",
        EXP2918_REL_PATH,
        (
            "honest_verdict",
            "inference_substrate",
            "online_self_learning_ready",
            "online_update_performed",
            "replay_scheduler_updated",
            "model_weights_mutated",
            "forgetting_rate",
            "delta_overall",
            "hardware_replay_used",
        ),
    ),
    SourceSpec(
        "exp2919",
        "exp2919_constraintbench_mini",
        "ConstraintBench mini direct optimization",
        EXP2919_REL_PATH,
        (
            "honest_verdict",
            "inference_substrate",
            "constraintbench_mini_ready",
            "syntax_valid_rate",
            "feasibility_rate",
            "optimality_rate",
            "flagged_adversarial",
            "corrigendum_pending",
        ),
    ),
    SourceSpec(
        "exp2920",
        "exp2920_state_verifier_harness",
        "OpenComputer-style state verifier harness",
        EXP2920_REL_PATH,
        (
            "honest_verdict",
            "inference_substrate",
            "state_verifier_harness_ready",
            "n_state_tasks",
            "llm_judge_used",
            "golden_state_pass_rate",
            "negative_state_reject_rate",
        ),
    ),
)

SOURCE_BY_EXP = {spec.experiment_id: spec for spec in SOURCE_SPECS}
SOURCE_BY_ROW = {spec.row_id: spec for spec in SOURCE_SPECS}

GATED_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("exp2911", "code_hallucination_verifier_ready"),
    ("exp2913", "kv260_claim_boundary_ready"),
    ("exp2918", "online_self_learning_ready"),
    ("exp2919", "constraintbench_mini_ready"),
    ("exp2920", "state_verifier_harness_ready"),
)

HEADLINE_IF_CLEAN = {
    "exp2910_sota_codegen",
    "exp2913_kv260_claim_boundary",
    "exp2918_fr11_process_rewards",
    "exp2920_state_verifier_harness",
}


def read_json(path: Path) -> dict[str, Any]:
    """Read one artifact object and degrade to `{}` for missing or bad inputs.

    The conductor may leave partial files during failed runs. Returning an empty
    object lets the caller classify that source as missing/blocked instead of
    raising and accidentally hiding the boundary in a traceback.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2921: build matrix v9 from consistent upstream artifacts."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    payloads = _load_payloads(root_path)
    source_shas = _source_shas(root_path)
    end = time.perf_counter() if now_s is None else now_s
    duration_s = end - started

    gate_errors = _gate_errors(payloads)
    if gate_errors:
        return _blocked_artifact(
            honest_verdict="blocked_gate_inconsistent",
            blocked_rows=[
                error["row_id"] for error in gate_errors if error["actual_value"] is not None
            ],
            missing_rows=[
                error["row_id"] for error in gate_errors if error["actual_value"] is None
            ],
            gate_errors=gate_errors,
            source_shas=source_shas,
            duration_s=duration_s,
        )

    matrix_v8 = payloads["exp2902"]
    if not matrix_v8:
        return _blocked_artifact(
            honest_verdict="blocked_matrix_v8_missing",
            blocked_rows=["exp2902_matrix_v8"],
            missing_rows=[],
            gate_errors=[],
            source_shas=source_shas,
            duration_s=duration_s,
        )

    rows = _build_matrix_rows(matrix_v8, payloads, source_shas)
    buckets = _bucket_rows(rows)
    headline_eligible = [
        row["row_id"]
        for row in rows
        if row["headline_eligible"] is True and row["row_status"] == "clean"
    ]

    return _base_artifact(
        honest_verdict=_complete_verdict(buckets, headline_eligible),
        cross_corpus_matrix_v9_built=True,
        paper_claim_boundary_ready=True,
        headline_eligible_rows=headline_eligible,
        buckets=buckets,
        rows=rows,
        source_shas=source_shas,
        gate_errors=[],
        duration_s=duration_s,
    )


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2921 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_payloads(root: Path) -> dict[str, dict[str, Any]]:
    return {spec.experiment_id: read_json(root / spec.path) for spec in SOURCE_SPECS}


def _source_shas(root: Path) -> dict[str, str | None]:
    shas: dict[str, str | None] = {}
    for spec in SOURCE_SPECS:
        path = root / spec.path
        shas[spec.experiment_id] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    return shas


def _gate_errors(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for exp_id, field in GATED_REQUIREMENTS:
        spec = SOURCE_BY_EXP[exp_id]
        payload = payloads[exp_id]
        actual = payload.get(field) if payload else None
        if actual is not True:
            errors.append(
                {
                    "experiment_id": exp_id,
                    "row_id": spec.row_id,
                    "artifact_path": str(spec.path),
                    "required_field": field,
                    "actual_value": actual,
                }
            )
    return errors


def _build_matrix_rows(
    matrix_v8: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    source_shas: dict[str, str | None],
) -> list[dict[str, Any]]:
    rows = [_v8_row(row, source_shas["exp2902"]) for row in _v8_source_rows(matrix_v8)]
    for exp_id in (
        "exp2910",
        "exp2911",
        "exp2913",
        "exp2914",
        "exp2915",
        "exp2916",
        "exp2917",
        "exp2918",
        "exp2919",
        "exp2920",
    ):
        rows.append(_candidate_row(SOURCE_BY_EXP[exp_id], payloads[exp_id], source_shas[exp_id]))
    return rows


def _v8_source_rows(matrix_v8: dict[str, Any]) -> list[dict[str, Any]]:
    rows = matrix_v8.get("matrix_rows")
    return [dict(row) for row in rows] if isinstance(rows, list) else []


def _v8_row(source_row: dict[str, Any], source_sha: str | None) -> dict[str, Any]:
    row_id = str(source_row.get("row_id") or "v8_row_missing_id")
    original_status = str(source_row.get("row_status") or "blocked")
    flag_reasons = _as_string_list(source_row.get("flag_reasons"))
    status = _classify_v8_status(original_status, flag_reasons)
    summary = source_row.get("summary") if isinstance(source_row.get("summary"), dict) else {}
    headline_eligible = status == "clean" and summary.get("headline_eligible") is True
    return {
        "row_id": row_id,
        "row_label": str(source_row.get("row_label") or row_id),
        "row_kind": "v8_carry_forward",
        "row_status": status,
        "original_row_status": original_status,
        "headline_eligible": headline_eligible,
        "claim_boundary": _v8_claim_boundary(row_id, source_row, headline_eligible),
        "non_headline_reason": "" if headline_eligible else _non_headline_reason(status, row_id),
        "flag_reasons": flag_reasons,
        "summary": summary,
        "source_artifact": str(MATRIX_V8_REL_PATH),
        "source_experiment_id": "exp2902",
        "source_sha256": source_sha,
    }


def _classify_v8_status(original_status: str, flag_reasons: list[str]) -> str:
    if flag_reasons or original_status == "flagged" or "flagged" in original_status:
        return "flagged"
    if original_status == "pilot_only":
        return "pilot_only"
    if original_status == "blocked":
        return "blocked"
    if original_status == "diagnostic_only":
        return "diagnostic_only"
    return "clean" if original_status == "clean" else "blocked"


def _v8_claim_boundary(
    row_id: str,
    source_row: dict[str, Any],
    headline_eligible: bool,
) -> str:
    if headline_eligible:
        label = str(source_row.get("row_label") or row_id)
        return f"Carry-forward clean matrix-v8 headline row for {label}; no new metric inferred."
    return ""


def _candidate_row(
    spec: SourceSpec, payload: dict[str, Any], source_sha: str | None
) -> dict[str, Any]:
    status = _candidate_status(spec.experiment_id, payload)
    summary = _selected_fields(payload, spec.fields_imported)
    headline_eligible = status == "clean" and spec.row_id in HEADLINE_IF_CLEAN
    return {
        "row_id": spec.row_id,
        "row_label": spec.label,
        "row_kind": "dot275_artifact_row",
        "row_status": status,
        "headline_eligible": headline_eligible,
        "claim_boundary": _candidate_claim_boundary(spec.row_id, payload, headline_eligible),
        "non_headline_reason": ""
        if headline_eligible
        else _non_headline_reason(status, spec.row_id),
        "flag_reasons": _flag_reasons(payload) if status == "flagged" else [],
        "summary": summary,
        "source_artifact": str(spec.path),
        "source_experiment_id": spec.experiment_id,
        "source_sha256": source_sha,
    }


def _candidate_status(exp_id: str, payload: dict[str, Any]) -> str:
    if not payload:
        return "missing"
    if _has_flags(payload):
        return "flagged"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if exp_id == "exp2910":
        return (
            "clean"
            if payload.get("codegen_corrigendum_ready") is True
            and payload.get("candidate_generation_clean") is True
            and payload.get("legacy_smoke_only") is not True
            and _is_number(payload.get("aggregate_pass_at_1"))
            and _is_number(payload.get("aggregate_pass_at_k"))
            else "blocked"
        )
    if exp_id == "exp2911":
        return "clean" if payload.get("code_hallucination_verifier_ready") is True else "blocked"
    if exp_id == "exp2913":
        candidate = payload.get("matrix_row_candidate")
        eligible = isinstance(candidate, dict) and candidate.get("eligible_for_matrix_v9") is True
        return (
            "clean"
            if payload.get("kv260_claim_boundary_ready") is True
            and payload.get("same_basis_verified") is True
            and payload.get("hardware_speedup_claim_eligible") is True
            and eligible
            else "blocked"
        )
    if exp_id == "exp2914":
        return "clean" if payload.get("gatemate_toolchain_ready") is True else "blocked"
    if exp_id == "exp2915":
        return "clean" if payload.get("gatemate_bitstream_built") is True else "blocked"
    if exp_id == "exp2916":
        return (
            "diagnostic_only"
            if payload.get("thrml_kv260_parity_ready") is True
            and payload.get("inference_substrate") == "simulator_parity"
            and payload.get("no_tsu_hardware_claim") is True
            else "blocked"
        )
    if exp_id == "exp2917":
        return (
            "diagnostic_only"
            if payload.get("spilled_energy_micro_panel_ready") is True
            and payload.get("benchmark_claim_made") is False
            else "blocked"
        )
    if exp_id == "exp2918":
        return (
            "clean"
            if payload.get("online_self_learning_ready") is True
            and payload.get("online_update_performed") is True
            and payload.get("replay_scheduler_updated") is True
            and payload.get("model_weights_mutated") is False
            else "blocked"
        )
    if exp_id == "exp2919":
        return "clean" if payload.get("constraintbench_mini_ready") is True else "blocked"
    if exp_id == "exp2920":
        return (
            "clean"
            if payload.get("state_verifier_harness_ready") is True
            and payload.get("llm_judge_used") is False
            else "blocked"
        )
    return "blocked"


def _candidate_claim_boundary(
    row_id: str,
    payload: dict[str, Any],
    headline_eligible: bool,
) -> str:
    if not headline_eligible:
        return ""
    if row_id == "exp2910_sota_codegen":
        return (
            "Bounded SOTA code-generation claim: "
            f"aggregate pass@1={payload.get('aggregate_pass_at_1')}, "
            f"pass@k={payload.get('aggregate_pass_at_k')} over the recorded local GGUF run."
        )
    if row_id == "exp2913_kv260_claim_boundary":
        boundary = payload.get("paper_claim_boundary")
        return str(boundary) if isinstance(boundary, str) else "Bounded KV260/CPU claim boundary."
    if row_id == "exp2918_fr11_process_rewards":
        return (
            "Bounded FR-11 self-learning claim: replay scheduler/process rewards updated "
            f"with forgetting_rate={payload.get('forgetting_rate')} and no model-weight mutation."
        )
    if row_id == "exp2920_state_verifier_harness":
        return (
            "Bounded state-verifier harness claim: deterministic golden and negative-state "
            "checks pass without an LLM judge; no broad benchmark generalization is implied."
        )
    return ""


def _selected_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: payload[field] for field in fields if field in payload}


def _has_flags(payload: dict[str, Any]) -> bool:
    if payload.get("flagged_adversarial") is True:
        return True
    if payload.get("adversarial_verify_passed") is False:
        return True
    for key in ("corrigendum_pending", "adversarial_verify_flags"):
        value = payload.get(key)
        if isinstance(value, list) and bool(value):
            return True
    summary = payload.get("adversarial_verify_summary")
    return isinstance(summary, dict) and int(summary.get("flag_count") or 0) > 0


def _flag_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("flagged_adversarial") is True:
        reasons.append("flagged_adversarial=true")
    if payload.get("adversarial_verify_passed") is False:
        reasons.append("adversarial_verify_passed=false")
    for item in payload.get("corrigendum_pending") or []:
        if isinstance(item, dict):
            kind = item.get("kind", "corrigendum_pending")
            severity = item.get("severity", "unknown")
            reasons.append(f"{kind}:{severity}")
    if payload.get("adversarial_verify_flags"):
        reasons.append("adversarial_verify_flags_present")
    summary = payload.get("adversarial_verify_summary")
    if isinstance(summary, dict) and int(summary.get("flag_count") or 0) > 0:
        reasons.append("adversarial_verify_summary_flag_count")
    return reasons


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("blocked", "gate_blocked")
    )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _bucket_rows(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    buckets = {
        "clean_rows": [],
        "flagged_rows": [],
        "blocked_rows": [],
        "pilot_only_rows": [],
        "diagnostic_only_rows": [],
        "missing_rows": [],
    }
    for row in rows:
        row_id = row["row_id"]
        status = row["row_status"]
        if status == "clean":
            buckets["clean_rows"].append(row_id)
        elif status == "flagged":
            buckets["flagged_rows"].append(row_id)
        elif status == "blocked":
            buckets["blocked_rows"].append(row_id)
        elif status == "pilot_only":
            buckets["pilot_only_rows"].append(row_id)
        elif status == "diagnostic_only":
            buckets["diagnostic_only_rows"].append(row_id)
        elif status == "missing":
            buckets["missing_rows"].append(row_id)
        if row.get("original_row_status") == "pilot_only_flagged_support":
            buckets["pilot_only_rows"].append(row_id)
    return buckets


def _non_headline_reason(status: str, row_id: str) -> str:
    if status == "flagged":
        return "excluded_from_headline_due_to_unresolved_flags_or_corrigendum"
    if status == "blocked":
        return "excluded_from_headline_because_source_is_blocked"
    if status == "pilot_only":
        return "excluded_from_headline_because_row_is_pilot_only"
    if status == "diagnostic_only":
        return "excluded_from_headline_because_row_is_diagnostic_or_simulator_only"
    if status == "missing":
        return "excluded_from_headline_because_source_artifact_is_missing"
    return f"{row_id} is clean context/support but not a bounded paper-v6 headline row"


def _blocked_artifact(
    *,
    honest_verdict: str,
    blocked_rows: list[str],
    missing_rows: list[str],
    gate_errors: list[dict[str, Any]],
    source_shas: dict[str, str | None],
    duration_s: float,
) -> dict[str, Any]:
    buckets = {
        "clean_rows": [],
        "flagged_rows": [],
        "blocked_rows": blocked_rows,
        "pilot_only_rows": [],
        "diagnostic_only_rows": [],
        "missing_rows": missing_rows,
    }
    return _base_artifact(
        honest_verdict=honest_verdict,
        cross_corpus_matrix_v9_built=False,
        paper_claim_boundary_ready=False,
        headline_eligible_rows=[],
        buckets=buckets,
        rows=[],
        source_shas=source_shas,
        gate_errors=gate_errors,
        duration_s=duration_s,
    )


def _base_artifact(
    *,
    honest_verdict: str,
    cross_corpus_matrix_v9_built: bool,
    paper_claim_boundary_ready: bool,
    headline_eligible_rows: list[str],
    buckets: dict[str, list[str]],
    rows: list[dict[str, Any]],
    source_shas: dict[str, str | None],
    gate_errors: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "honest_verdict": honest_verdict,
        "cross_corpus_matrix_v9_built": cross_corpus_matrix_v9_built,
        "paper_claim_boundary_ready": paper_claim_boundary_ready,
        "headline_eligible_rows": headline_eligible_rows,
        "clean_rows": buckets["clean_rows"],
        "flagged_rows": buckets["flagged_rows"],
        "blocked_rows": buckets["blocked_rows"],
        "pilot_only_rows": buckets["pilot_only_rows"],
        "diagnostic_only_rows": buckets["diagnostic_only_rows"],
        "missing_rows": buckets["missing_rows"],
        "matrix_v9_path": str(OUTPUT_REL_PATH),
        "matrix_rows": rows,
        "paper_v6_claim_boundary": _paper_claim_boundary(
            paper_claim_boundary_ready,
            headline_eligible_rows,
            rows,
        ),
        "cited_upstream_artifacts": _cited_upstream_artifacts(source_shas),
        "gate_errors": gate_errors,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "run_date": RUN_DATE,
        "duration_s": round(max(0.0, duration_s), 6),
    }


def _paper_claim_boundary(
    ready: bool,
    headline_eligible_rows: list[str],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    headline_claims = {
        row["row_id"]: row["claim_boundary"]
        for row in rows
        if row["row_id"] in headline_eligible_rows
    }
    non_headline = {
        row["row_id"]: {
            "row_label": row["row_label"],
            "status": row["row_status"],
            "reason": row["non_headline_reason"],
        }
        for row in rows
        if row["row_id"] not in headline_eligible_rows
    }
    return {
        "ready": ready,
        "headline_eligible_rows": headline_eligible_rows,
        "headline_claims": headline_claims,
        "non_headline_rows": non_headline,
        "boundary_rules": [
            "Only clean rows with direct bounded claim support are headline eligible.",
            "Flagged, blocked, pilot-only, missing, simulator-only, and diagnostic-only rows are excluded.",
            "No row is promoted from a success boolean when unresolved flags or claim-boundary caveats exist.",
        ],
    }


def _cited_upstream_artifacts(source_shas: dict[str, str | None]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": spec.experiment_id,
            "row_id": spec.row_id,
            "artifact_path": str(spec.path),
            "fields_imported": list(spec.fields_imported),
            "sha256": source_shas.get(spec.experiment_id),
            "present": source_shas.get(spec.experiment_id) is not None,
        }
        for spec in SOURCE_SPECS
    ]


def _complete_verdict(
    buckets: dict[str, list[str]],
    headline_eligible_rows: list[str],
) -> str:
    return (
        "complete: cross-corpus matrix v9 and paper-v6 claim boundary built; "
        f"headline_eligible={len(headline_eligible_rows)}; "
        f"clean={len(buckets['clean_rows'])}; flagged={len(buckets['flagged_rows'])}; "
        f"blocked={len(buckets['blocked_rows'])}; pilot_only={len(buckets['pilot_only_rows'])}; "
        f"diagnostic_only={len(buckets['diagnostic_only_rows'])}; "
        f"missing={len(buckets['missing_rows'])}"
    )
