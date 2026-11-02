"""Build the Exp 2974 milestone .279 capstone artifact.

Spec refs: REQ-REPORT-2974, SCENARIO-REPORT-2974.

This module is an aggregation-only closeout layer. It reads the active
milestone roadmap, the .278 capstone, the available .279 result artifacts, and
the v13 matrix, then writes one compact closeout JSON. It does not rerun model
inference, verifier scoring, solver execution, synthesis, board flashing, or
tests for upstream experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
MILESTONE = "2026.05.279"
SCHEMA = "carnot.milestone_capstone.v279_aggregation.v1"
ARTIFACT = "experiment_2974_capstone_v279"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2974_capstone_v279.json")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")

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
MATRIX_V13_REL_PATH = Path("results/experiment_2973_cross_corpus_matrix_v13.json")

CLASSIFICATIONS = (
    "clean",
    "flagged",
    "blocked",
    "gated-skipped",
    "missing",
    "pilot-only",
    "aggregation-only",
)

TASK_PATH_OVERRIDES = {
    "exp2962": EXP2962_REL_PATH,
    "exp2963": EXP2963_REL_PATH,
    "exp2964": EXP2964_REL_PATH,
    "exp2965": EXP2965_REL_PATH,
    "exp2966": EXP2966_REL_PATH,
    "exp2967": EXP2967_REL_PATH,
    "exp2968": EXP2968_REL_PATH,
    "exp2969": EXP2969_REL_PATH,
    "exp2970": EXP2970_REL_PATH,
    "exp2971": EXP2971_REL_PATH,
    "exp2972": EXP2972_REL_PATH,
    "exp2973": MATRIX_V13_REL_PATH,
    "exp2974": OUTPUT_REL_PATH,
}

AGGREGATION_TASK_IDS = {"exp2962", "exp2973", "exp2974"}
PILOT_TASK_IDS = {"exp2968"}
PAPER_RELEVANT_FLAGGED_IDS = {"exp2963", "exp2964", "exp2966", "exp2967", "exp2969", "exp2973"}

FORBIDDEN_CLAIMS_REAFFIRMED = [
    "KV260 speedup claims remain forbidden.",
    "KV260 Boltzmann, thermalization, or equilibrium-sampling claims remain forbidden.",
    "Extropic TSU, Kona, and Aleph-equivalent performance claims remain forbidden.",
    "Native EBT training claims remain forbidden.",
    "Broad verifier-generalization claims beyond measured rows remain forbidden.",
    "Full BEAVER probability-bound claims remain forbidden.",
    "Full streaming partial-monitor verification claims remain forbidden.",
]

FORBIDDEN_SAFE_CLAIM_TOKENS = (
    "kv260",
    "boltzmann",
    "thermalization",
    "tsu",
    "kona",
    "native ebt",
)


@dataclass(frozen=True)
class TaskSpec:
    """One planned roadmap task and the local artifact path it should emit."""

    task_id: str
    title: str
    deliverable: Path
    inference_substrate: str
    gated_on: tuple[Mapping[str, Any], ...]


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object and fail closed when a source is absent or malformed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    """Return a file digest so the capstone can prove which sources it read."""

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
    """REQ-REPORT-2974: synthesize the terminal .279 capstone."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)
    if started_s is None and now_s is None and duration_s == 0.0:
        duration_s = 0.000001

    tasks = _load_tasks(root_path)
    payloads = _load_task_payloads(root_path, tasks)
    capstone_v278 = read_json_object(root_path / CAPSTONE_V278_REL_PATH)
    matrix_v13 = payloads.get("exp2973") or read_json_object(root_path / MATRIX_V13_REL_PATH)
    classification_details = _classification_details(tasks, payloads, root_path)
    buckets = _bucket_task_ids(classification_details)
    outcome_summaries = _outcome_summaries(payloads, capstone_v278, matrix_v13)
    gap_assessment = _three_biggest_gap_assessment(buckets, outcome_summaries)
    safe_claims = _paper_v6_safe_claims(outcome_summaries, buckets)
    forbidden_claims_absent = _forbidden_claims_absent(matrix_v13, safe_claims)
    gaps_closed = _gaps_closed(buckets, outcome_summaries)
    gaps_remaining = _gaps_remaining(buckets, outcome_summaries, matrix_v13, forbidden_claims_absent)
    paper_ready = _paper_ready(matrix_v13, buckets, forbidden_claims_absent)
    source_paths = _source_paths(root_path, tasks)
    source_rows = _source_rows(root_path, source_paths)

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": _honest_verdict(paper_ready, buckets),
        "milestone": MILESTONE,
        "paper_ready": paper_ready,
        "headline_outcome": _headline_outcome(paper_ready, gaps_closed, gaps_remaining),
        "clean_artifacts": buckets["clean"],
        "flagged_artifacts": buckets["flagged"],
        "blocked_artifacts": buckets["blocked"],
        "gated_skipped_artifacts": buckets["gated-skipped"],
        "missing_artifacts": buckets["missing"],
        "pilot_only_artifacts": buckets["pilot-only"],
        "aggregation_only_artifacts": buckets["aggregation-only"],
        "artifact_classification_counts": {name: len(buckets[name]) for name in CLASSIFICATIONS},
        "classification_details": classification_details,
        "outcome_summaries": outcome_summaries,
        "three_biggest_gap_assessment": gap_assessment,
        "gaps_closed": gaps_closed,
        "gaps_remaining": gaps_remaining,
        "forbidden_claims_absent": forbidden_claims_absent,
        "forbidden_claims_reaffirmed": FORBIDDEN_CLAIMS_REAFFIRMED,
        "paper_v6_safe_claims": safe_claims,
        "next_milestone_recommendations": _next_milestone_recommendations(),
        "source_artifacts_read": source_rows,
        "source_checksums": {row["path"]: row["sha256"] for row in source_rows},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration_s,
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2974 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_tasks(root: Path) -> list[TaskSpec]:
    roadmap = yaml.safe_load((root / ROADMAP_REL_PATH).read_text(encoding="utf-8"))
    raw_tasks = roadmap.get("tasks", []) if isinstance(roadmap, Mapping) else []
    tasks: list[TaskSpec] = []
    for item in raw_tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or "")
        if task_id not in TASK_PATH_OVERRIDES:
            continue
        gated_on = item.get("gated_on")
        tasks.append(
            TaskSpec(
                task_id=task_id,
                title=str(item.get("title") or ""),
                deliverable=TASK_PATH_OVERRIDES[task_id],
                inference_substrate=str(item.get("inference_substrate") or ""),
                gated_on=tuple(gate for gate in gated_on if isinstance(gate, Mapping))
                if isinstance(gated_on, list)
                else (),
            )
        )
    return tasks


def _load_task_payloads(root: Path, tasks: list[TaskSpec]) -> dict[str, dict[str, Any]]:
    return {task.task_id: read_json_object(root / task.deliverable) for task in tasks}


def _classification_details(
    tasks: list[TaskSpec],
    payloads: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        payload = payloads.get(task.task_id, {})
        gate_failures = _gate_failures(task, payloads)
        present = (root / task.deliverable).is_file()
        rows.append(
            {
                "task_id": task.task_id,
                "title": task.title,
                "path": task.deliverable.as_posix(),
                "classification": _classify_task(task, payload, present, gate_failures),
                "present": present,
                "source_honest_verdict": str(payload.get("honest_verdict") or ""),
                "flag_kinds": _flag_kinds(payload),
                "gate_blocked_by": gate_failures,
            }
        )
    return rows


def _classify_task(
    task: TaskSpec,
    payload: Mapping[str, Any],
    present: bool,
    gate_failures: list[str],
) -> str:
    if task.task_id == "exp2974":
        return "aggregation-only"
    if not present or not payload:
        return "gated-skipped" if gate_failures else "missing"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if _has_flags(payload):
        return "flagged"
    if task.task_id in AGGREGATION_TASK_IDS:
        return "aggregation-only"
    if task.task_id in PILOT_TASK_IDS or payload.get("pilot_only") is True:
        return "pilot-only"
    return "clean"


def _gate_failures(task: TaskSpec, payloads: Mapping[str, Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for gate in task.gated_on:
        upstream = str(gate.get("upstream") or "")
        field = str(gate.get("artifact_field") or "")
        expected = gate.get("value")
        actual = _get_path(payloads.get(upstream, {}), field)
        if gate.get("op", "==") == "==" and actual != expected:
            failures.append(f"{upstream}.{field}")
    return failures


def _bucket_task_ids(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    buckets = {name: [] for name in CLASSIFICATIONS}
    for row in rows:
        classification = str(row["classification"])
        if classification in buckets:
            buckets[classification].append(str(row["task_id"]))
    return buckets


def _source_paths(root: Path, tasks: list[TaskSpec]) -> list[Path]:
    paths = {CAPSTONE_V278_REL_PATH}
    paths.update(task.deliverable for task in tasks)
    results_dir = root / "results"
    if results_dir.is_dir():
        paths.update(path.relative_to(root) for path in results_dir.glob("experiment_296*.json"))
        paths.update(path.relative_to(root) for path in results_dir.glob("experiment_297*.json"))
    return sorted(paths, key=lambda path: path.as_posix())


def _source_rows(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.as_posix(),
            "present": (root / path).is_file(),
            "sha256": sha256_file(root / path),
        }
        for path in paths
    ]


def _outcome_summaries(
    payloads: Mapping[str, Mapping[str, Any]],
    capstone_v278: Mapping[str, Any],
    matrix_v13: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dccd_code_repair": _dccd_code_repair_summary(payloads, matrix_v13),
        "beaver_style_certificates": _beaver_certificate_summary(payloads.get("exp2965", {})),
        "solver_frontier_formalization": _solver_frontier_summary(payloads, matrix_v13),
        "partial_monitors": _partial_monitor_summary(payloads.get("exp2968", {})),
        "fr11_non_tautology": _fr11_summary(payloads.get("exp2969", {}), matrix_v13),
        "kan_memory": _kan_memory_summary(payloads.get("exp2970", {}), matrix_v13),
        "gatemate": _gatemate_summary(payloads, capstone_v278, matrix_v13),
        "matrix_v13": _matrix_summary(matrix_v13),
    }


def _dccd_code_repair_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    matrix_v13: Mapping[str, Any],
) -> dict[str, Any]:
    matrix_summary = _mapping_or_empty(matrix_v13.get("repair_replication_summary"))
    protocol = payloads.get("exp2963", {})
    replication = payloads.get("exp2964", {})
    summary = {
        "protocol_ready": bool(
            matrix_summary.get("protocol_ready") or protocol.get("dccd_repair_protocol_ready")
        ),
        "protocol_artifact_flagged": _has_flags(protocol),
        "n_tasks": _coerce_int(_pick(matrix_summary, replication, "n_tasks")),
        "baseline_pass_at_1": _coerce_float(_pick(matrix_summary, replication, "baseline_pass_at_1")),
        "taxonomy_repair_pass_at_1": _coerce_float(
            _pick(matrix_summary, replication, "taxonomy_repair_pass_at_1")
        ),
        "dccd_repair_pass_at_1": _coerce_float(
            _pick(matrix_summary, replication, "dccd_repair_pass_at_1")
        ),
        "pass_at_1_delta": _coerce_float(_pick(matrix_summary, replication, "pass_at_1_delta")),
        "pass_at_k_delta": _coerce_float(_pick(matrix_summary, replication, "pass_at_k_delta")),
        "syntax_failure_rate_delta": _coerce_float(
            _pick(matrix_summary, replication, "syntax_failure_rate_delta")
        ),
        "schema_failure_rate_delta": _coerce_float(
            _pick(matrix_summary, replication, "schema_failure_rate_delta")
        ),
        "false_accept_delta": _coerce_float(_pick(matrix_summary, replication, "false_accept_delta")),
        "dccd_repair_replication_clean": bool(
            _pick(matrix_summary, replication, "dccd_repair_replication_clean")
        ),
        "headline_model_count": len(replication.get("headline_models_used", []))
        if isinstance(replication.get("headline_models_used"), list)
        else len(matrix_summary.get("headline_models_used", []))
        if isinstance(matrix_summary.get("headline_models_used"), list)
        else 0,
        "artifact_flagged": _has_flags(replication) or bool(matrix_summary.get("artifact_flagged")),
    }
    summary["status"] = (
        "clean_repair_uplift"
        if summary["dccd_repair_replication_clean"] and not summary["artifact_flagged"]
        else "open_flagged_regression"
    )
    return summary


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
        "status": "bounded_certificate_audit_only",
    }


def _solver_frontier_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    matrix_v13: Mapping[str, Any],
) -> dict[str, Any]:
    matrix_summary = _mapping_or_empty(matrix_v13.get("solver_frontier_summary"))
    frontier = payloads.get("exp2966", {})
    formalization = payloads.get("exp2967", {})
    parseability = _coerce_float(_pick(matrix_summary, formalization, "parseability_rate"))
    baseline_parseability = _coerce_float(
        _pick(matrix_summary, formalization, "baseline_parseability_rate")
    )
    solver_accuracy = _coerce_float(
        _pick(matrix_summary, formalization, "solver_verified_accuracy")
    )
    baseline_solver = _coerce_float(
        _pick(matrix_summary, formalization, "baseline_solver_verified_accuracy")
    )
    artifact_flagged = _has_flags(frontier) or _has_flags(formalization) or bool(
        matrix_summary.get("artifact_flagged")
    )
    return {
        "frontier_materialized": bool(
            matrix_summary.get("frontier_materialized")
            or frontier.get("logic_frontier_materialized")
        ),
        "frontier_n_items": _coerce_int(matrix_summary.get("frontier_n_items") or frontier.get("n_items")),
        "reference_solver_accuracy": _coerce_float(
            matrix_summary.get("reference_solver_accuracy")
            or frontier.get("reference_solver_accuracy")
        ),
        "formalization_n_items": _coerce_int(
            matrix_summary.get("formalization_n_items") or formalization.get("n_items")
        ),
        "baseline_parseability_rate": baseline_parseability,
        "parseability_rate": parseability,
        "parseability_delta_vs_278": _coerce_float(matrix_summary.get("parseability_delta_vs_278"))
        if "parseability_delta_vs_278" in matrix_summary
        else _delta(parseability, baseline_parseability),
        "baseline_solver_verified_accuracy": baseline_solver,
        "solver_verified_accuracy": solver_accuracy,
        "solver_verified_accuracy_delta_vs_278": _coerce_float(
            matrix_summary.get("solver_verified_accuracy_delta_vs_278")
        )
        if "solver_verified_accuracy_delta_vs_278" in matrix_summary
        else _delta(solver_accuracy, baseline_solver),
        "formalization_delta_clean": bool(
            _pick(matrix_summary, formalization, "formalization_delta_clean")
        ),
        "artifact_flagged": artifact_flagged,
        "status": "open_flagged_partial_improvement"
        if artifact_flagged
        else "clean_formalization_delta",
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
        "status": "pilot_only",
    }


def _fr11_summary(payload: Mapping[str, Any], matrix_v13: Mapping[str, Any]) -> dict[str, Any]:
    matrix_summary = _mapping_or_empty(matrix_v13.get("self_learning_summary"))
    summary = {
        "continuous_self_learning_task": bool(
            matrix_summary.get("continuous_self_learning_task")
            or payload.get("continuous_self_learning_task")
        ),
        "non_tautological_self_learning_ready": bool(
            matrix_summary.get("non_tautological_self_learning_ready")
            or payload.get("non_tautological_self_learning_ready")
        ),
        "leakage_check_passed": bool(
            matrix_summary.get("leakage_check_passed") or payload.get("leakage_check_passed")
        ),
        "heldout_utility_delta_vs_random": _coerce_float(
            _pick(matrix_summary, payload, "heldout_utility_delta_vs_random")
        ),
        "negative_control_delta": _coerce_float(
            _pick(matrix_summary, payload, "negative_control_delta")
        ),
        "forgetting_guard_passed": bool(
            matrix_summary.get("forgetting_guard_passed") or payload.get("forgetting_guard_passed")
        ),
        "rollback_triggered": bool(
            matrix_summary.get("rollback_triggered") or payload.get("rollback_triggered")
        ),
        "artifact_flagged": _has_flags(payload) or bool(matrix_summary.get("artifact_flagged")),
    }
    summary["status"] = (
        "clean_non_tautological_ready"
        if summary["non_tautological_self_learning_ready"] and not summary["artifact_flagged"]
        else "flagged_non_tautological_ready"
    )
    return summary


def _kan_memory_summary(payload: Mapping[str, Any], matrix_v13: Mapping[str, Any]) -> dict[str, Any]:
    matrix_summary = _mapping_or_empty(matrix_v13.get("kan_memory_summary"))
    return {
        "kan_forgetting_guard_ready": bool(
            matrix_summary.get("kan_forgetting_guard_ready")
            or payload.get("kan_forgetting_guard_ready")
        ),
        "selected_policy": matrix_summary.get("selected_policy") or payload.get("selected_policy"),
        "forgetting_threshold": _coerce_float(
            matrix_summary.get("forgetting_threshold") or payload.get("forgetting_threshold")
        ),
        "forgetting_delta_by_policy": _mapping_or_empty(
            matrix_summary.get("forgetting_delta_by_policy")
            or payload.get("forgetting_delta_by_policy")
        ),
        "current_domain_utility": _mapping_or_empty(
            matrix_summary.get("current_domain_utility") or payload.get("current_domain_utility")
        ),
        "old_domain_utility": _mapping_or_empty(
            matrix_summary.get("old_domain_utility") or payload.get("old_domain_utility")
        ),
        "high_dimensional_claim_allowed": bool(
            matrix_summary.get("high_dimensional_claim_allowed")
            or payload.get("high_dimensional_claim_allowed")
        ),
        "no_synthesis_claim": bool(
            matrix_summary.get("no_synthesis_claim") or payload.get("no_synthesis_claim")
        ),
        "no_analog_claim": bool(matrix_summary.get("no_analog_claim") or payload.get("no_analog_claim")),
        "status": "clean_bounded_fixture",
    }


def _gatemate_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    capstone_v278: Mapping[str, Any],
    matrix_v13: Mapping[str, Any],
) -> dict[str, Any]:
    matrix_hardware = _mapping_or_empty(matrix_v13.get("hardware_state_summary"))
    matrix_gatemate = _mapping_or_empty(matrix_hardware.get("gatemate"))
    prior = _mapping_or_empty(_get_path(capstone_v278, "outcome_summaries.hardware.gatemate"))
    exp2971 = payloads.get("exp2971", {})
    exp2972 = payloads.get("exp2972", {})
    return {
        "prior_278_flash_state": matrix_gatemate.get("prior_278_flash_state")
        or prior.get("flash_state"),
        "prior_278_board_detected": bool(
            matrix_gatemate.get("prior_278_board_detected") or prior.get("board_detected")
        ),
        "board_detected": bool(
            matrix_gatemate.get("board_detected") or exp2971.get("gatemate_board_detected")
        ),
        "bitstream_sha256_verified": bool(
            matrix_gatemate.get("bitstream_sha256_verified")
            or exp2971.get("bitstream_sha256_verified")
            or exp2972.get("bitstream_sha256_verified")
        ),
        "flash_preconditions_ready": bool(
            matrix_gatemate.get("flash_preconditions_ready")
            or exp2971.get("gatemate_flash_preconditions_ready")
        ),
        "flash_attempted": bool(matrix_gatemate.get("flash_attempted") or exp2972.get("flash_attempted")),
        "flash_succeeded": bool(matrix_gatemate.get("flash_succeeded") or exp2972.get("flash_succeeded")),
        "smoke_vector_passed": bool(
            matrix_gatemate.get("smoke_vector_passed") or exp2972.get("smoke_vector_passed")
        ),
        "observed_output_sha256": matrix_gatemate.get("observed_output_sha256")
        or exp2972.get("observed_output_sha256"),
        "post_flash_contact_detected": matrix_gatemate.get("post_flash_contact_detected")
        or _get_path(exp2972, "timing_observation.post_flash_contact_detected"),
        "status": "contact_and_flash_hash_only",
    }


def _matrix_summary(matrix_v13: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "matrix_v13_ready": bool(matrix_v13.get("matrix_v13_ready")),
        "artifact_flagged": _has_flags(matrix_v13),
        "forbidden_claims_absent": bool(matrix_v13.get("forbidden_claims_absent")),
        "clean_rows": _coerce_int(len(matrix_v13.get("clean_rows", [])))
        if isinstance(matrix_v13.get("clean_rows"), list)
        else 0,
        "flagged_rows": _coerce_int(len(matrix_v13.get("flagged_rows", [])))
        if isinstance(matrix_v13.get("flagged_rows"), list)
        else 0,
        "blocked_rows": _coerce_int(len(matrix_v13.get("blocked_rows", [])))
        if isinstance(matrix_v13.get("blocked_rows"), list)
        else 0,
        "pilot_only_rows": _coerce_int(len(matrix_v13.get("pilot_only_rows", [])))
        if isinstance(matrix_v13.get("pilot_only_rows"), list)
        else 0,
    }


def _three_biggest_gap_assessment(
    buckets: Mapping[str, list[str]],
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    dccd = summaries["dccd_code_repair"]
    solver = summaries["solver_frontier_formalization"]
    fr11 = summaries["fr11_non_tautology"]
    gatemate = summaries["gatemate"]
    return {
        "code_repair_replication": "closed_clean"
        if dccd.get("dccd_repair_replication_clean") and "exp2964" in buckets["clean"]
        else "open_flagged_regression",
        "solver_frontier_formalization": "closed_clean"
        if solver.get("formalization_delta_clean") and "exp2967" in buckets["clean"]
        else "open_flagged_partial_improvement",
        "fr11_and_hardware": "closed_clean"
        if "exp2969" in buckets["clean"] and gatemate.get("smoke_vector_passed")
        else (
            "partial_gate_mate_contact_closed_fr11_flagged"
            if fr11.get("non_tautological_self_learning_ready")
            and gatemate.get("flash_succeeded")
            else "open"
        ),
    }


def _gaps_closed(
    buckets: Mapping[str, list[str]],
    summaries: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    closed: list[str] = []
    if "exp2965" in buckets["clean"] and summaries["beaver_style_certificates"].get(
        "certificate_ready"
    ):
        closed.append("BEAVER-style bounded repair certificate audit landed without probability-bound claims.")
    if "exp2970" in buckets["clean"] and summaries["kan_memory"].get("kan_forgetting_guard_ready"):
        closed.append("KAN per-knot forgetting guard landed as a bounded fixture result.")
    if "exp2972" in buckets["clean"] and summaries["gatemate"].get("flash_succeeded"):
        closed.append("GateMate board contact and flash hash path closed the .278 board-detection blocker.")
    return closed


def _gaps_remaining(
    buckets: Mapping[str, list[str]],
    summaries: Mapping[str, Mapping[str, Any]],
    matrix_v13: Mapping[str, Any],
    forbidden_claims_absent: bool,
) -> list[str]:
    gaps: list[str] = []
    dccd = summaries["dccd_code_repair"]
    solver = summaries["solver_frontier_formalization"]
    fr11 = summaries["fr11_non_tautology"]
    gatemate = summaries["gatemate"]
    matrix = summaries["matrix_v13"]
    if "exp2964" in buckets["flagged"] or not dccd.get("dccd_repair_replication_clean"):
        gaps.append("DCCD repair replication remains open: pass deltas regressed and the artifact is flagged.")
    if "exp2967" in buckets["flagged"] or not solver.get("formalization_delta_clean"):
        gaps.append("Solver-frontier formalization remains flagged despite parseability improvement.")
    if "exp2969" in buckets["flagged"]:
        gaps.append("FR-11 non-tautology remains flagged and cannot headline self-learning.")
    if "exp2973" in buckets["flagged"] or matrix.get("artifact_flagged"):
        gaps.append("Matrix v13 is aggregation-complete but adversarially flagged.")
    if "exp2968" in buckets["pilot-only"]:
        gaps.append("Partial monitors remain pilot-only; no full streaming verification claim is available.")
    if not gatemate.get("smoke_vector_passed"):
        gaps.append("GateMate still lacks a passed smoke vector or readback-backed sampler claim.")
    if buckets["blocked"]:
        gaps.append(f"Blocked planned .279 artifacts: {', '.join(buckets['blocked'])}.")
    if buckets["gated-skipped"]:
        gaps.append(f"Gated-skipped planned .279 artifacts: {', '.join(buckets['gated-skipped'])}.")
    if buckets["missing"]:
        gaps.append(f"Missing planned .279 artifacts: {', '.join(buckets['missing'])}.")
    if not matrix_v13.get("matrix_v13_ready"):
        gaps.append("Cross-corpus matrix v13 is not ready.")
    if not forbidden_claims_absent:
        gaps.append("Forbidden claim boundary failed in matrix v13.")
    return _unique_strings(gaps)


def _paper_ready(
    matrix_v13: Mapping[str, Any],
    buckets: Mapping[str, list[str]],
    forbidden_claims_absent: bool,
) -> bool:
    unresolved_hard_stop = buckets["blocked"] + buckets["gated-skipped"] + buckets["missing"]
    paper_relevant_flags = [task_id for task_id in buckets["flagged"] if task_id in PAPER_RELEVANT_FLAGGED_IDS]
    return bool(matrix_v13.get("matrix_v13_ready")) and forbidden_claims_absent and not (
        unresolved_hard_stop or paper_relevant_flags
    )


def _forbidden_claims_absent(matrix_v13: Mapping[str, Any], safe_claims: list[str]) -> bool:
    safe_text = json.dumps(safe_claims, sort_keys=True).lower()
    return bool(matrix_v13.get("forbidden_claims_absent")) and not any(
        token in safe_text for token in FORBIDDEN_SAFE_CLAIM_TOKENS
    )


def _paper_v6_safe_claims(
    summaries: Mapping[str, Mapping[str, Any]],
    buckets: Mapping[str, list[str]],
) -> list[str]:
    claims = [
        "BEAVER-style evidence is limited to bounded deterministic candidate-certificate auditing.",
        "KAN memory evidence is limited to a bounded per-knot forgetting-guard fixture.",
        "GateMate evidence is limited to board contact, bitstream integrity, flash, and output hash.",
    ]
    if "exp2964" not in buckets.get("clean", []):
        claims.append("DCCD repair replication remains non-headline because current repair deltas regressed.")
    if "exp2967" not in buckets.get("clean", []):
        claims.append("Solver formalization remains non-headline despite higher parseability.")
    if summaries["fr11_non_tautology"].get("artifact_flagged"):
        claims.append("FR-11 non-tautology controls are recorded as flagged, non-headline evidence.")
    return claims


def _headline_outcome(paper_ready: bool, gaps_closed: list[str], gaps_remaining: list[str]) -> str:
    if paper_ready:
        return "paper_ready: .279 cleared the .278 flagged code, FR-11, and solver rows"
    return (
        "partial: bounded KAN, BEAVER, and GateMate-contact evidence landed, "
        f"but {len(gaps_remaining)} unresolved gaps keep paper readiness false"
        if gaps_closed
        else "blocked: no .279 gap closed cleanly"
    )


def _honest_verdict(paper_ready: bool, buckets: Mapping[str, list[str]]) -> str:
    return (
        "complete: milestone_279_capstone; "
        f"paper_ready={str(paper_ready).lower()}; "
        f"clean={len(buckets['clean'])}; "
        f"flagged={len(buckets['flagged'])}; "
        f"blocked={len(buckets['blocked'])}; "
        f"missing={len(buckets['missing'])}"
    )


def _next_milestone_recommendations() -> list[str]:
    return [
        "DCCD .280: repair schema failures and rerun n>=20 with positive pass delta and no new flags.",
        "Solver .280: target parseability >=0.50 and solver-verified accuracy >=0.40 with tautology flags removed.",
        "FR-11 .280: rerun the non-tautological utility gate with independent metric slices and no identical-metric flags.",
        "GateMate .280: add readback or a passed smoke vector before any sampler-facing hardware claim.",
    ]


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


def _get_path(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _mapping_or_empty(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _pick(primary: Mapping[str, Any], fallback: Mapping[str, Any], key: str) -> Any:
    return primary[key] if key in primary else fallback.get(key)


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


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact())
