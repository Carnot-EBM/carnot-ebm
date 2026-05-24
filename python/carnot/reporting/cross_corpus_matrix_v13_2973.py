"""Build the Exp 2973 cross-corpus matrix v13 artifact.

Spec refs: REQ-REPORT-2973, SCENARIO-REPORT-2973.

This module is only an aggregation layer. It reads the v12 matrix, the .278
capstone, and completed .279 artifacts, then emits a compact v13 matrix. It
does not rerun model inference, verifier scoring, solver execution, synthesis,
board flashing, or hardware smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
SCHEMA = "carnot.cross_corpus_matrix.v13_279_aggregation.v1"
ARTIFACT = "experiment_2973_cross_corpus_matrix_v13"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2973_cross_corpus_matrix_v13.json")

MATRIX_V12_REL_PATH = Path("results/experiment_2960_cross_corpus_matrix_v12.json")
CAPSTONE_V278_REL_PATH = Path("results/experiment_2961_capstone_v278.json")
EXP2962_REL_PATH = Path("results/experiment_2962_archive_v278_activate_v279.json")
EXP2963_REL_PATH = Path("results/experiment_2963_dccd_repair_protocol_manifest_v1.json")
EXP2964_REL_PATH = Path("results/experiment_2964_sota_dccd_repair_replication_v1.json")
EXP2965_REL_PATH = Path("results/experiment_2965_beaver_style_repair_certificate_v1.json")
EXP2966_REL_PATH = Path("results/experiment_2966_logic_frontier_materializer_v1.json")
EXP2967_REL_PATH = Path("results/experiment_2967_sota_nl_to_z3_dccd_formalization_v1.json")
EXP2968_REL_PATH = Path("results/experiment_2968_interwhen_partial_monitor_harness_v1.json")
EXP2969_REL_PATH = Path("results/experiment_2969_fr11_non_tautological_utility_gate_v3.json")
EXP2970_REL_PATH = Path("results/experiment_2970_kan_forgetting_guard_memory_audit_v1.json")
EXP2971_REL_PATH = Path("results/experiment_2971_gatemate_board_detection_flash_harness_v3.json")
EXP2972_REL_PATH = Path("results/experiment_2972_gatemate_post_flash_output_hash_v3.json")

FORBIDDEN_PHRASES = (
    "KV260 hardware speedup",
    "KV260 speedup",
    "FPGA acceleration over CPU",
    "runs faster on KV260",
    "Boltzmann",
    "thermalization",
    "TSU performance",
    "Kona performance",
    "native EBT training",
)


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    path: Path
    required: bool = False


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp2960", MATRIX_V12_REL_PATH, required=True),
    SourceSpec("exp2961", CAPSTONE_V278_REL_PATH, required=True),
    SourceSpec("exp2962", EXP2962_REL_PATH),
    SourceSpec("exp2963", EXP2963_REL_PATH),
    SourceSpec("exp2964", EXP2964_REL_PATH),
    SourceSpec("exp2965", EXP2965_REL_PATH),
    SourceSpec("exp2966", EXP2966_REL_PATH),
    SourceSpec("exp2967", EXP2967_REL_PATH),
    SourceSpec("exp2968", EXP2968_REL_PATH),
    SourceSpec("exp2969", EXP2969_REL_PATH, required=True),
    SourceSpec("exp2970", EXP2970_REL_PATH),
    SourceSpec("exp2971", EXP2971_REL_PATH),
    SourceSpec("exp2972", EXP2972_REL_PATH),
)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one upstream JSON object, returning an empty mapping when unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    """Return the SHA256 digest for a source file, or None when absent."""

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
) -> dict[str, Any]:
    """REQ-REPORT-2973: build matrix v13 from upstream artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)
    payloads = _load_sources(root_path)
    source_rows = _upstream_artifacts_read(root_path, payloads)
    checksums = _upstream_checksums(source_rows)

    artifact = _base_artifact(payloads, source_rows, checksums, duration_s)
    required_errors = _required_source_errors(payloads)
    if required_errors:
        artifact.update(
            {
                "honest_verdict": "blocked_required_upstream_missing",
                "matrix_v13_ready": False,
                "required_upstream_errors": required_errors,
            }
        )
        artifact["forbidden_claims_absent"] = _forbidden_claims_absent(artifact)
        return artifact

    if payloads["exp2969"].get("non_tautological_self_learning_ready") is not True:
        artifact["blocked_rows"] = _unique_strings(
            [*artifact["blocked_rows"], "exp2969_non_tautological_fr11"]
        )
        artifact.update(
            {
                "honest_verdict": "blocked_non_tautological_self_learning_not_ready",
                "matrix_v13_ready": False,
            }
        )
        artifact["forbidden_claims_absent"] = _forbidden_claims_absent(artifact)
        return artifact

    artifact["matrix_v13_ready"] = True
    artifact["honest_verdict"] = _complete_verdict(artifact)
    artifact["forbidden_claims_absent"] = _forbidden_claims_absent(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2973 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_sources(root: Path) -> dict[str, dict[str, Any]]:
    return {spec.experiment_id: read_json_object(root / spec.path) for spec in SOURCE_SPECS}


def _upstream_artifacts_read(
    root: Path,
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "present": path.is_file(),
                "required": spec.required,
                "readable_json_object": bool(payloads.get(spec.experiment_id)),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _upstream_checksums(source_rows: list[dict[str, Any]]) -> dict[str, str | None]:
    return {str(row["path"]): row["sha256"] for row in source_rows}


def _required_source_errors(
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        if spec.required and not payloads.get(spec.experiment_id):
            errors.append(
                {
                    "experiment_id": spec.experiment_id,
                    "path": spec.path.as_posix(),
                    "reason": "missing_or_malformed_artifact",
                }
            )
    return errors


def _base_artifact(
    payloads: Mapping[str, dict[str, Any]],
    source_rows: list[dict[str, Any]],
    checksums: dict[str, str | None],
    duration_s: float,
) -> dict[str, Any]:
    v12 = payloads.get("exp2960", {})
    v13_rows = _v13_rows(payloads)
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": "blocked_required_upstream_missing",
        "matrix_v13_ready": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "upstream_artifacts_read": source_rows,
        "upstream_checksums": checksums,
        "clean_rows": _unique_strings(
            [*_v12_bucket(v12, "clean"), *_ids_by_class(v13_rows, "clean")]
        ),
        "flagged_rows": _unique_strings(
            [*_v12_bucket(v12, "flagged"), *_ids_by_class(v13_rows, "flagged")]
        ),
        "blocked_rows": _unique_strings(
            [*_v12_bucket(v12, "blocked"), *_ids_by_class(v13_rows, "blocked")]
        ),
        "gated_skipped_rows": _unique_strings(
            [*_v12_bucket(v12, "gated-skipped"), *_ids_by_class(v13_rows, "gated-skipped")]
        ),
        "pilot_only_rows": _unique_strings(
            [*_v12_bucket(v12, "pilot-only"), *_ids_by_class(v13_rows, "pilot-only")]
        ),
        "aggregation_only_rows": _unique_strings(
            [
                "exp2960_matrix_v12_carry_forward",
                "exp2961_capstone_v278_carry_forward",
                *_v12_bucket(v12, "aggregation-only"),
                *_ids_by_class(v13_rows, "aggregation-only"),
            ]
        ),
        "forbidden_claims_absent": False,
        "repair_replication_summary": _repair_replication_summary(payloads),
        "solver_frontier_summary": _solver_frontier_summary(payloads),
        "self_learning_summary": _self_learning_summary(payloads.get("exp2969", {})),
        "kan_memory_summary": _kan_memory_summary(payloads.get("exp2970", {})),
        "hardware_state_summary": _hardware_state_summary(payloads),
        "matrix_rows": [
            {
                "row_id": "exp2960_matrix_v12_carry_forward",
                "row_class": "aggregation-only",
                "source_experiment_id": "exp2960",
                "claim_boundary": "v12 row buckets are copied without metric recomputation.",
            },
            {
                "row_id": "exp2961_capstone_v278_carry_forward",
                "row_class": "aggregation-only",
                "source_experiment_id": "exp2961",
                "claim_boundary": "The .278 capstone boundary is carried forward.",
                "paper_ready": bool(payloads.get("exp2961", {}).get("paper_ready")),
            },
            *v13_rows,
        ],
        "delta_relative_to_278": _delta_relative_to_278(payloads),
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "duration_s": duration_s,
    }


def _v13_rows(payloads: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _row(
            "exp2962_archive_activation",
            _class_from_ready(payloads.get("exp2962", {}), "archive_ready", ready_class="aggregation-only"),
            "exp2962",
            "Archive/activation state is copied without reclassifying .278 evidence.",
            {"archive_ready": bool(payloads.get("exp2962", {}).get("archive_ready"))},
            payloads.get("exp2962", {}),
        ),
        _row(
            "exp2963_dccd_repair_protocol",
            _class_from_ready(payloads.get("exp2963", {}), "dccd_repair_protocol_ready", ready_class="aggregation-only"),
            "exp2963",
            "DCCD protocol manifest prepares replication and makes no pass-rate claim.",
            {
                "protocol_ready": bool(payloads.get("exp2963", {}).get("dccd_repair_protocol_ready")),
                "n_tasks_planned_min": _coerce_int(payloads.get("exp2963", {}).get("n_tasks_planned_min")),
            },
            payloads.get("exp2963", {}),
        ),
        _row(
            "exp2964_dccd_repair_replication",
            _dccd_replication_class(payloads.get("exp2964", {})),
            "exp2964",
            "Live DCCD repair replication did not promote unless explicit gates passed.",
            _repair_replication_summary(payloads),
            payloads.get("exp2964", {}),
        ),
        _row(
            "exp2965_beaver_style_certificates",
            _class_from_ready(payloads.get("exp2965", {}), "beaver_style_certificate_ready", ready_class="clean"),
            "exp2965",
            "Bounded deterministic repair certificates are recorded without probability-bound claims.",
            _beaver_certificate_summary(payloads.get("exp2965", {})),
            payloads.get("exp2965", {}),
        ),
        _row(
            "exp2966_logic_frontier_materializer",
            _class_from_ready(payloads.get("exp2966", {}), "logic_frontier_materialized", ready_class="clean"),
            "exp2966",
            "Exact skill-labeled logic frontier is copied as deterministic source evidence.",
            _logic_frontier_summary(payloads.get("exp2966", {})),
            payloads.get("exp2966", {}),
        ),
        _row(
            "exp2967_solver_frontier_formalization",
            _formalization_class(payloads.get("exp2967", {})),
            "exp2967",
            "Live NL-to-Z3 frontier formalization is recorded with parseability and solver gates.",
            _solver_formalization_summary(payloads.get("exp2967", {})),
            payloads.get("exp2967", {}),
        ),
        _row(
            "exp2968_partial_monitor_harness",
            _partial_monitor_class(payloads.get("exp2968", {})),
            "exp2968",
            "Partial monitor fixture harness remains pilot-only, not a full streaming verifier claim.",
            _partial_monitor_summary(payloads.get("exp2968", {})),
            payloads.get("exp2968", {}),
        ),
        _row(
            "exp2969_non_tautological_fr11",
            _class_from_ready(
                payloads.get("exp2969", {}),
                "non_tautological_self_learning_ready",
                ready_class="clean",
            ),
            "exp2969",
            "FR-11 utility gate uses disjoint checked-in slices and reports flags separately.",
            _self_learning_summary(payloads.get("exp2969", {})),
            payloads.get("exp2969", {}),
        ),
        _row(
            "exp2970_kan_forgetting_guard",
            _class_from_ready(payloads.get("exp2970", {}), "kan_forgetting_guard_ready", ready_class="clean"),
            "exp2970",
            "KAN constraint memory is bounded to forgetting guards and local cost fields.",
            _kan_memory_summary(payloads.get("exp2970", {})),
            payloads.get("exp2970", {}),
        ),
        _row(
            "exp2971_gatemate_flash_preflight",
            _class_from_ready(payloads.get("exp2971", {}), "gatemate_flash_preconditions_ready", ready_class="clean"),
            "exp2971",
            "GateMate board contact preflight prepared the command but did not flash.",
            _gatemate_preflight_summary(payloads.get("exp2971", {})),
            payloads.get("exp2971", {}),
        ),
        _row(
            "exp2972_gatemate_flash_contact_hash",
            _gatemate_flash_class(payloads.get("exp2972", {})),
            "exp2972",
            "GateMate flash/contact hash evidence is scoped to board contact only.",
            _gatemate_flash_summary(payloads.get("exp2972", {})),
            payloads.get("exp2972", {}),
        ),
    ]


def _row(
    row_id: str,
    row_class: str,
    source_experiment_id: str,
    claim_boundary: str,
    summary: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "row_class": row_class,
        "source_experiment_id": source_experiment_id,
        "headline_eligible": row_class == "clean",
        "paper_claim_eligible": row_class == "clean",
        "claim_boundary": claim_boundary,
        "source_honest_verdict": payload.get("honest_verdict", ""),
        "upstream_flags": _flag_kinds(payload),
        "summary": dict(summary),
    }


def _class_from_flags(payload: Mapping[str, Any], *, default: str) -> str:
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if _has_flags(payload):
        return "flagged"
    return default


def _class_from_ready(
    payload: Mapping[str, Any],
    ready_field: str,
    *,
    ready_class: str,
) -> str:
    if not payload:
        return "gated-skipped"
    if payload.get(ready_field) is not True:
        return _class_from_flags(payload, default="blocked")
    return _class_from_flags(payload, default=ready_class)


def _dccd_replication_class(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "gated-skipped"
    if payload.get("dccd_repair_replication_clean") is True:
        return _class_from_flags(payload, default="clean")
    return _class_from_flags(payload, default="flagged")


def _formalization_class(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "gated-skipped"
    if payload.get("formalization_delta_clean") is True:
        return _class_from_flags(payload, default="clean")
    return _class_from_flags(payload, default="flagged")


def _partial_monitor_class(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "gated-skipped"
    if payload.get("partial_monitor_harness_ready") is not True:
        return _class_from_flags(payload, default="blocked")
    return _class_from_flags(payload, default="pilot-only")


def _gatemate_flash_class(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "gated-skipped"
    if payload.get("flash_succeeded") is True:
        return _class_from_flags(payload, default="clean")
    return _class_from_flags(payload, default="blocked")


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("blocked", "gate_blocked")
    )


def _has_flags(payload: Mapping[str, Any]) -> bool:
    if payload.get("flagged_adversarial") is True:
        return True
    flags = payload.get("corrigendum_pending")
    return isinstance(flags, list) and bool(flags)


def _flag_kinds(payload: Mapping[str, Any]) -> list[str]:
    kinds: list[str] = []
    if payload.get("flagged_adversarial") is True:
        kinds.append("flagged_adversarial=true")
    flags = payload.get("corrigendum_pending")
    if isinstance(flags, list):
        for item in flags:
            if isinstance(item, Mapping):
                kind = str(item.get("kind") or "unknown")
                severity = str(item.get("severity") or "unknown")
                kinds.append(f"{kind}:{severity}")
    return _unique_strings(kinds)


def _repair_replication_summary(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    v12 = payloads.get("exp2960", {})
    exp2963 = payloads.get("exp2963", {})
    exp2964 = payloads.get("exp2964", {})
    prior = v12.get("code_repair_delta_summary")
    prior_summary = prior if isinstance(prior, Mapping) else {}
    return {
        "prior_source_experiment_id": "exp2952",
        "prior_278_pass_at_1_delta": _coerce_float(prior_summary.get("pass_at_1_delta")),
        "prior_278_pass_at_k_delta": _coerce_float(prior_summary.get("pass_at_k_delta")),
        "prior_278_syntax_failure_rate_delta": _coerce_float(
            prior_summary.get("syntax_failure_rate_delta")
        ),
        "prior_278_false_accept_delta": _coerce_float(prior_summary.get("false_accept_delta")),
        "protocol_ready": bool(exp2963.get("dccd_repair_protocol_ready")),
        "source_experiment_id": "exp2964",
        "n_tasks": _coerce_int(exp2964.get("n_tasks")),
        "baseline_pass_at_1": _coerce_float(exp2964.get("baseline_pass_at_1")),
        "taxonomy_repair_pass_at_1": _coerce_float(exp2964.get("taxonomy_repair_pass_at_1")),
        "dccd_repair_pass_at_1": _coerce_float(exp2964.get("dccd_repair_pass_at_1")),
        "pass_at_1_delta": _coerce_float(exp2964.get("pass_at_1_delta")),
        "baseline_pass_at_k": _coerce_float(exp2964.get("baseline_pass_at_k")),
        "dccd_repair_pass_at_k": _coerce_float(exp2964.get("dccd_repair_pass_at_k")),
        "pass_at_k_delta": _coerce_float(exp2964.get("pass_at_k_delta")),
        "syntax_failure_rate_delta": _coerce_float(exp2964.get("syntax_failure_rate_delta")),
        "schema_failure_rate_delta": _coerce_float(exp2964.get("schema_failure_rate_delta")),
        "false_accept_delta": _coerce_float(exp2964.get("false_accept_delta")),
        "dccd_repair_replication_clean": bool(exp2964.get("dccd_repair_replication_clean")),
        "headline_models_used": exp2964.get("headline_models_used")
        if isinstance(exp2964.get("headline_models_used"), list)
        else [],
        "artifact_flagged": _has_flags(exp2964),
    }


def _beaver_certificate_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "certificate_ready": bool(payload.get("beaver_style_certificate_ready")),
        "available_candidate_count": _coerce_int(payload.get("available_repair_candidate_count")),
        "audited_candidate_count": _coerce_int(
            payload.get("available_repair_candidate_audited_count")
        ),
        "probability_bound_claimed": bool(payload.get("full_beaver_claim")),
        "validation_fixture_passed": bool(payload.get("validation_fixture_passed")),
        "validation_fixture_count": _coerce_int(payload.get("validation_fixture_count")),
    }


def _logic_frontier_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frontier_materialized": bool(payload.get("logic_frontier_materialized")),
        "frontier_n_items": _coerce_int(payload.get("n_items")),
        "reference_z3_execution_rate": _coerce_float(payload.get("reference_z3_execution_rate")),
        "reference_solver_accuracy": _coerce_float(payload.get("reference_solver_accuracy")),
        "skill_label_counts": payload.get("skill_label_counts")
        if isinstance(payload.get("skill_label_counts"), Mapping)
        else {},
        "manifest_sha256": payload.get("manifest_sha256"),
        "artifact_flagged": _has_flags(payload),
    }


def _solver_formalization_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    baseline_parse = _coerce_float(payload.get("baseline_parseability_rate"))
    parse = _coerce_float(payload.get("parseability_rate"))
    baseline_solver = _coerce_float(payload.get("baseline_solver_verified_accuracy"))
    solver = _coerce_float(payload.get("solver_verified_accuracy"))
    return {
        "formalization_n_items": _coerce_int(payload.get("n_items")),
        "baseline_parseability_rate": baseline_parse,
        "parseability_rate": parse,
        "parseability_delta_vs_278": _delta(parse, baseline_parse),
        "baseline_solver_verified_accuracy": baseline_solver,
        "solver_verified_accuracy": solver,
        "solver_verified_accuracy_delta_vs_278": _delta(solver, baseline_solver),
        "answer_accuracy": _coerce_float(payload.get("answer_accuracy")),
        "z3_execution_rate": _coerce_float(payload.get("z3_execution_rate")),
        "formalization_delta_clean": bool(payload.get("formalization_delta_clean")),
        "failure_categories": payload.get("failure_categories")
        if isinstance(payload.get("failure_categories"), Mapping)
        else {},
        "skill_wise_metrics": payload.get("skill_wise_metrics")
        if isinstance(payload.get("skill_wise_metrics"), Mapping)
        else {},
        "artifact_flagged": _has_flags(payload),
    }


def _solver_frontier_summary(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        **_logic_frontier_summary(payloads.get("exp2966", {})),
        **_solver_formalization_summary(payloads.get("exp2967", {})),
    }


def _partial_monitor_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "partial_monitor_harness_ready": bool(payload.get("partial_monitor_harness_ready")),
        "fixture_trace_count": _coerce_int(payload.get("fixture_trace_count")),
        "fixture_checks_passed": bool(payload.get("fixture_checks_passed")),
        "full_streaming_verification_claim": bool(payload.get("full_streaming_verification_claim")),
        "latency_estimate_ms": _coerce_float(payload.get("latency_estimate_ms")),
        "coverage_by_event": payload.get("coverage_by_event")
        if isinstance(payload.get("coverage_by_event"), Mapping)
        else {},
    }


def _self_learning_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_experiment_id": "exp2969",
        "continuous_self_learning_task": bool(payload.get("continuous_self_learning_task")),
        "non_tautological_self_learning_ready": bool(
            payload.get("non_tautological_self_learning_ready")
        ),
        "leakage_check_passed": bool(payload.get("leakage_check_passed")),
        "frozen_heldout_utility": _coerce_float(payload.get("frozen_heldout_utility")),
        "random_replay_heldout_utility": _coerce_float(
            payload.get("random_replay_heldout_utility")
        ),
        "prior_utility_gated_heldout_utility": _coerce_float(
            payload.get("prior_utility_gated_heldout_utility")
        ),
        "new_heldout_utility": _coerce_float(payload.get("new_heldout_utility")),
        "heldout_utility_delta_vs_random": _coerce_float(
            payload.get("heldout_utility_delta_vs_random")
        ),
        "negative_control_delta": _coerce_float(payload.get("negative_control_delta")),
        "forgetting_guard_passed": bool(payload.get("forgetting_guard_passed")),
        "rollback_triggered": bool(payload.get("rollback_triggered")),
        "artifact_flagged": _has_flags(payload),
    }


def _kan_memory_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_experiment_id": "exp2970",
        "kan_forgetting_guard_ready": bool(payload.get("kan_forgetting_guard_ready")),
        "selected_policy": payload.get("selected_policy"),
        "forgetting_threshold": _coerce_float(payload.get("forgetting_threshold")),
        "forgetting_delta_by_policy": payload.get("forgetting_delta_by_policy")
        if isinstance(payload.get("forgetting_delta_by_policy"), Mapping)
        else {},
        "current_domain_utility": payload.get("current_domain_utility")
        if isinstance(payload.get("current_domain_utility"), Mapping)
        else {},
        "old_domain_utility": payload.get("old_domain_utility")
        if isinstance(payload.get("old_domain_utility"), Mapping)
        else {},
        "high_dimensional_claim_allowed": bool(payload.get("high_dimensional_claim_allowed")),
        "no_synthesis_claim": bool(payload.get("no_synthesis_claim")),
        "no_analog_claim": bool(payload.get("no_analog_claim")),
        "claim_boundary": "bounded_fixture_no_synthesis_or_analog_claim",
    }


def _gatemate_preflight_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "board_detected": bool(payload.get("gatemate_board_detected")),
        "bitstream_sha256_verified": bool(payload.get("bitstream_sha256_verified")),
        "flash_preconditions_ready": bool(payload.get("gatemate_flash_preconditions_ready")),
        "bitstream_sha256": payload.get("bitstream_sha256"),
        "flash_command_prepared": bool(payload.get("flash_command")),
    }


def _gatemate_flash_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "board_detected": bool(payload.get("board_detected")),
        "bitstream_sha256_verified": bool(payload.get("bitstream_sha256_verified")),
        "flash_attempted": bool(payload.get("flash_attempted")),
        "flash_succeeded": bool(payload.get("flash_succeeded")),
        "smoke_vector_passed": bool(payload.get("smoke_vector_passed")),
        "observed_output_sha256": payload.get("observed_output_sha256"),
        "post_flash_contact_detected": _get_path(
            payload, "timing_observation.post_flash_contact_detected"
        ),
    }


def _hardware_state_summary(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    v12_gatemate = _get_path(payloads.get("exp2960", {}), "hardware_state_summary.gatemate")
    prior = v12_gatemate if isinstance(v12_gatemate, Mapping) else {}
    preflight = _gatemate_preflight_summary(payloads.get("exp2971", {}))
    flash = _gatemate_flash_summary(payloads.get("exp2972", {}))
    return {
        "gatemate": {
            "prior_278_flash_state": prior.get("flash_state"),
            "prior_278_board_detected": bool(prior.get("board_detected")),
            **preflight,
            **flash,
        },
        "claim_boundary": (
            "GateMate rows record constraints, bitstream, board contact, flash, "
            "and transcript hashes only; no sampler or comparative claim is made."
        ),
    }


def _delta_relative_to_278(payloads: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    capstone = payloads.get("exp2961", {})
    return {
        "capstone_source_experiment_id": "exp2961",
        "capstone_paper_ready": bool(capstone.get("paper_ready")),
        "repair_replication": _repair_replication_summary(payloads),
        "solver_frontier": _solver_frontier_summary(payloads),
        "self_learning": _self_learning_summary(payloads.get("exp2969", {})),
        "kan_memory": _kan_memory_summary(payloads.get("exp2970", {})),
        "hardware_state": _hardware_state_summary(payloads),
    }


def _complete_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: matrix_v13_ready=true; "
        f"clean={len(artifact['clean_rows'])}; "
        f"flagged={len(artifact['flagged_rows'])}; "
        f"blocked={len(artifact['blocked_rows'])}; "
        f"gated_skipped={len(artifact['gated_skipped_rows'])}; "
        f"pilot_only={len(artifact['pilot_only_rows'])}"
    )


def _ids_by_class(rows: list[dict[str, Any]], row_class: str) -> list[str]:
    return [str(row["row_id"]) for row in rows if row.get("row_class") == row_class]


def _v12_bucket(payload: Mapping[str, Any], bucket: str) -> list[str]:
    current_key = f"{bucket.replace('-', '_')}_rows"
    legacy_key = f"rows_{bucket.replace('-', '_')}"
    rows = payload.get(current_key, payload.get(legacy_key, []))
    return [str(item) for item in rows] if isinstance(rows, list) else []


def _forbidden_claims_absent(artifact: Mapping[str, Any]) -> bool:
    rendered = json.dumps(artifact, sort_keys=True)
    return not any(phrase in rendered for phrase in FORBIDDEN_PHRASES)


def _get_path(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return round(current - baseline, 6)


def _unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
