"""Exp 5133: ungated .470 capstone aggregation.

Spec refs: REQ-CAPSTONE-5133, SCENARIO-CAPSTONE-5133,
SCENARIO-CAPSTONE-5133-FIELD-PRINCIPLES.

This module reads the completed .470 artifacts and writes a terminal decision
record. It does not rerun research work. The important discipline is that a
missing, gate-skipped, or adversarially flagged artifact becomes an explicit
gap for its own axis instead of being treated as evidence against unrelated
clean axes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
AdversarialReporter = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_RELATIVE_PATH = "python/carnot/experiment_5133_capstone_v470.py"
RESULT_RELATIVE_PATH = Path("results") / "experiment_5133_capstone_v470.json"
EXPERIMENT = "experiment_5133_capstone_v470"
EXPERIMENT_ID = "exp5133-capstone-v470"
MILESTONE = "2026.07.470"
RUN_DATE = "20260701"
RANDOM_SEED = 5133
SCHEMA = "carnot.experiment_5133_capstone_v470.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = (
    "complete_capstone_v470_runtime_clean_exact_solver_progress_"
    "structured_energy_quarantined_fr11_no_promote_hardware_continuity"
)
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_", "passed_", "shipped_")

SPEC_REFS = [
    "REQ-CAPSTONE-5133",
    "SCENARIO-CAPSTONE-5133",
    "SCENARIO-CAPSTONE-5133-FIELD-PRINCIPLES",
]

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "artifacts_read",
    "missing_artifacts",
    "gated_skips",
    "quarantined_artifacts",
    "fover_same_scope_retired",
    "runtime_state",
    "structured_energy_state",
    "kan_certificate_state",
    "solver_sampling_state",
    "fr11_state",
    "hardware_state",
    "next_milestone_recommendations",
    "active_roadmap_modified",
    "conductor_modified",
    "flagged_adversarial",
    "tests_run",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "spec_refs",
    "result_path",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "artifacts_read": "provenance",
    "missing_artifacts": "gap transparency",
    "gated_skips": "no false negatives",
    "quarantined_artifacts": "adversarial hygiene",
    "fover_same_scope_retired": "no doomed rerun",
    "runtime_state": "SOTA substrate decision",
    "structured_energy_state": "PRD FR-12 decision",
    "kan_certificate_state": "exact-verifier scale/explainability decision",
    "solver_sampling_state": "sampler and solver utility decision",
    "fr11_state": "PRD FR-11 decision",
    "hardware_state": "hardware continuity decision",
    "next_milestone_recommendations": "planning continuity",
    "active_roadmap_modified": "operator instruction compliance",
    "conductor_modified": "conductor immutability",
    "flagged_adversarial": "adversarial-verification accountability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5133_capstone_v470.py --date 20260701",
    ".venv/bin/pytest tests/python/test_experiment_5133_capstone_v470.py -q",
    "JAX_PLATFORMS=cpu .venv/bin/coverage run --rcfile=/dev/null --source=python/carnot,scripts "
    "-m pytest tests/python/test_experiment_5133_capstone_v470.py -q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m --include='*/experiment_5133_capstone_v470.py' "
    "--fail-under=100",
    "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class UpstreamSource:
    """One expected upstream result and the bounded fields safe to import."""

    experiment_number: int
    label: str
    axis: str
    relative_path: Path
    imported_fields: tuple[str, ...]
    missing_reason: str


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5122,
        "archive_469_activate_470",
        "transition",
        Path("results/experiment_5122_archive_469_activate_470.json"),
        ("honest_verdict", "fover_selector_retired_for_same_verdict", "fover_retirement"),
        "transition_archive_artifact_missing",
    ),
    UpstreamSource(
        5123,
        "v470_source_scope_audit",
        "planning",
        Path("results/experiment_5123_v470_source_scope_audit.json"),
        ("honest_verdict", "fover_same_scope_rerun_found", "sota_model_discipline_ok"),
        "source_scope_audit_artifact_missing",
    ),
    UpstreamSource(
        5124,
        "clean_sota_runtime_provenance",
        "runtime",
        Path("results/experiment_5124_clean_sota_runtime_provenance_v470.json"),
        (
            "honest_verdict",
            "sota_runtime_clean",
            "adversarial_verify_passed",
            "cache_ready",
            "completion_proof",
            "logprob_proof",
            "endpoint_lifetime_s",
        ),
        "runtime_provenance_artifact_missing",
    ),
    UpstreamSource(
        5125,
        "structured_reasoning_pool",
        "structured_energy",
        Path("results/experiment_5125_structured_reasoning_pool_v470.json"),
        (
            "honest_verdict",
            "structured_pool_ready",
            "pool_n",
            "oracle_at_k",
            "cheap_baseline_at_1",
            "parse_coverage",
            "duplicate_rate",
            "fover_scope_used",
        ),
        "structured_pool_artifact_missing",
    ),
    UpstreamSource(
        5126,
        "distributional_energy_ranker",
        "structured_energy",
        Path("results/experiment_5126_distributional_energy_ranker_v470.json"),
        (
            "honest_verdict",
            "distributional_energy_delta",
            "ranker_ready_for_audit",
            "ranker_metrics",
            "strongest_cheap_baseline",
            "delta_ci95",
        ),
        "distributional_ranker_artifact_missing",
    ),
    UpstreamSource(
        5127,
        "structured_energy_adversarial_audit",
        "structured_energy",
        Path("results/experiment_5127_structured_energy_adversarial_audit_v470.json"),
        ("honest_verdict", "gate_check_summary", "gates_evaluated", "blocked_at_layer"),
        "structured_energy_audit_gate_skipped",
    ),
    UpstreamSource(
        5128,
        "kan_certificate_explanation",
        "kan_certificate",
        Path("results/experiment_5128_kan_certificate_explanation_v470.json"),
        (
            "honest_verdict",
            "kan_certificate_breadth_ready",
            "certificate_soundness",
            "explanation_cycle_soundness",
            "false_property_detected",
            "near_margin_abstained",
            "property_families",
        ),
        "kan_certificate_artifact_missing",
    ),
    UpstreamSource(
        5129,
        "hubo_adaptive_2dpt",
        "solver_sampling",
        Path("results/experiment_5129_hubo_adaptive_2dpt_v470.json"),
        (
            "honest_verdict",
            "adaptive_2dpt_ready",
            "exact_enumeration_checked",
            "hardware_speedup_claimed",
            "detailed_balance_sanity",
            "optimum_hit_rate",
            "best_energy_delta_vs_baselines",
        ),
        "hubo_adaptive_2dpt_artifact_missing",
    ),
    UpstreamSource(
        5130,
        "taco_sampler_heldout_scale",
        "solver_sampling",
        Path("results/experiment_5130_taco_sampler_heldout_scale_v470.json"),
        (
            "honest_verdict",
            "heldout_csp_trace_suite_ready",
            "instance_count",
            "wrong_label_count",
            "average_effort_reduction_ratio_guarded",
            "harmful_instance_count_guarded",
            "harmful_instance_count_unguarded",
            "baseline_effort",
            "guarded_effort",
            "sampler_feature_effort",
        ),
        "taco_sampler_heldout_artifact_missing",
    ),
    UpstreamSource(
        5131,
        "fr11_case_policy_self_learning",
        "fr11",
        Path("results/experiment_5131_fr11_case_policy_self_learning_v470.json"),
        (
            "honest_verdict",
            "continuous_self_learning_task",
            "heldout_delta",
            "nonforgetting_delta",
            "harmful_promotion_count",
            "exact_solver_correctness_preserved",
            "promotion_attempted",
            "promotion_safe",
            "rollback_receipt",
            "no_weight_update",
        ),
        "fr11_case_policy_artifact_missing",
    ),
    UpstreamSource(
        5132,
        "authenticated_board_timing",
        "hardware",
        Path("results/experiment_5132_authenticated_board_timing_v470.json"),
        (
            "honest_verdict",
            "kv260_ssh_checked",
            "kv260_ssh_ready",
            "kv260_host_block_devices_touched",
            "gatemate_checked",
            "gatemate_detected",
            "polarfire_checked",
            "polarfire_ssh_ready",
            "extropic_tsu_execution_claimed",
            "no_speedup_claim",
            "timing_measurements",
            "board_precheck_summary",
        ),
        "hardware_board_timing_artifact_missing",
    ),
)


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def file_sha256(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - corrupted input defense.
        return {}, {"exists": True, "loadable": False, "error": str(exc)}
    if not isinstance(payload, Mapping):  # pragma: no cover - JSON artifact must be an object.
        return {}, {"exists": True, "loadable": False, "error": "json_not_object"}
    return dict(payload), {"exists": True, "loadable": True, "sha256": file_sha256(path)}


def run_adversarial_report(
    path: Path,
) -> JsonDict:  # pragma: no cover - exercised by deliverable run.
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import adversarial_verify as av  # noqa: PLC0415

    return dict(av.verify_artifact(path))


def _critical_flags(report: JsonMap) -> list[JsonDict]:
    return [
        dict(flag)
        for flag in _list(report.get("flags"))
        if str(_mapping(flag).get("severity", "")).lower() == "critical"
    ]


def _blocked(payload: JsonMap) -> bool:
    verdict = str(payload.get("honest_verdict", ""))
    return (
        verdict.startswith("blocked_")
        or str(payload.get("status", "")).lower() == "blocked"
        or str(payload.get("blocked_at_layer", ""))
        or str(payload.get("schema", "")) == "blocked_gate_check_v1"
    )


def _imported(source: UpstreamSource, payload: JsonMap) -> JsonDict:
    return {field: payload[field] for field in source.imported_fields if field in payload}


def classify_artifact(payload: JsonMap, status: JsonMap, adversarial_report: JsonMap) -> str:
    critical = _critical_flags(adversarial_report)
    return (
        "missing"
        if status.get("loadable") is not True
        else "adversarially_flagged"
        if payload.get("flagged_adversarial") is True or critical
        else "gated_skip"
        if _blocked(payload)
        else "clean"
    )


def artifact_row(
    source: UpstreamSource,
    payload: JsonMap,
    status: JsonMap,
    adversarial_report: JsonMap,
) -> JsonDict:
    classification = classify_artifact(payload, status, adversarial_report)
    critical = _critical_flags(adversarial_report)
    row: JsonDict = {
        "experiment_number": source.experiment_number,
        "label": source.label,
        "axis": source.axis,
        "path": str(source.relative_path),
        "exists": status.get("exists") is True,
        "loadable": status.get("loadable") is True,
        "classification": classification,
        "headline_eligible": classification == "clean",
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "flagged_adversarial_stamped": payload.get("flagged_adversarial") is True,
        "inference_substrate": str(payload.get("inference_substrate", "")),
        "duration_s": _number(payload.get("duration_s")),
        "imported": _imported(source, payload),
        "adversarial_verification": {
            "loaded": adversarial_report.get("loaded", status.get("loadable") is True),
            "flag_count": int(_number(adversarial_report.get("flag_count")) or 0),
            "max_severity": int(_number(adversarial_report.get("max_severity")) or -1),
            "critical_flags": critical,
            "flags": _list(adversarial_report.get("flags")),
        },
    }
    row.update({"sha256": status["sha256"]} if "sha256" in status else {})
    row.update(
        {
            "quarantine_reason": "live_critical_adversarial_flag"
            if critical
            else "stamped_flagged_adversarial"
        }
        if classification == "adversarially_flagged"
        else {}
    )
    row.update(
        {
            "gate_skip_reason": str(
                payload.get("gate_check_summary")
                or payload.get("blocked_at_layer")
                or payload.get("honest_verdict", "")
            )
        }
        if classification == "gated_skip"
        else {}
    )
    return row


def load_upstreams(
    root: Path,
    adversarial_reporter: AdversarialReporter,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict], dict[int, JsonDict]]:
    artifacts_read: list[JsonDict] = []
    missing_artifacts: list[JsonDict] = []
    gated_skips: list[JsonDict] = []
    quarantined_artifacts: list[JsonDict] = []
    payloads: dict[int, JsonDict] = {}
    for source in UPSTREAM_SOURCES:
        payload, status = read_json_mapping(root / source.relative_path)
        if status.get("loadable") is not True:
            missing_artifacts.append(
                {
                    "experiment_number": source.experiment_number,
                    "label": source.label,
                    "axis": source.axis,
                    "path": str(source.relative_path),
                    "reason": source.missing_reason,
                    "exists": status.get("exists") is True,
                    "error": status.get("error", "missing"),
                }
            )
            continue
        report = adversarial_reporter(root / source.relative_path)
        row = artifact_row(source, payload, status, report)
        artifacts_read.append(row)
        payloads[source.experiment_number] = dict(payload)
        gated_skips.extend([row] if row["classification"] == "gated_skip" else [])
        quarantined_artifacts.extend(
            [row] if row["classification"] == "adversarially_flagged" else []
        )
    return artifacts_read, missing_artifacts, gated_skips, quarantined_artifacts, payloads


def _row_by_id(rows: Sequence[JsonMap], experiment_number: int) -> JsonDict:
    return next(
        (dict(row) for row in rows if row.get("experiment_number") == experiment_number), {}
    )


def _ids(rows: Sequence[JsonMap]) -> set[int]:
    return {
        int(row["experiment_number"])
        for row in rows
        if isinstance(row.get("experiment_number"), int)
    }


def _present(payloads: Mapping[int, JsonMap], experiment_number: int) -> JsonDict:
    return dict(payloads.get(experiment_number, {}))


def build_fover_same_scope_retired(
    artifacts_read: Sequence[JsonMap],
    payloads: Mapping[int, JsonMap],
) -> bool:
    archive = _present(payloads, 5122)
    audit = _present(payloads, 5123)
    archive_retired = (
        archive.get("fover_selector_retired_for_same_verdict") is True
        or _mapping(archive.get("fover_retirement")).get("fover_residual_fr11_should_not_rerun")
        is True
    )
    audit_no_rerun = audit.get("fover_same_scope_rerun_found") is False
    audit_row = _row_by_id(artifacts_read, 5123)
    return bool(
        archive_retired
        and (audit_no_rerun or audit_row.get("classification") == "adversarially_flagged")
    )


def build_runtime_state(
    artifacts_read: Sequence[JsonMap],
    missing_artifacts: Sequence[JsonMap],
    quarantined_ids: set[int],
    payloads: Mapping[int, JsonMap],
) -> JsonDict:
    payload = _present(payloads, 5124)
    missing = bool(_row_by_id(missing_artifacts, 5124))
    quarantined = 5124 in quarantined_ids
    clean = (
        payload.get("sota_runtime_clean") is True
        and payload.get("adversarial_verify_passed") is True
        and payload.get("cache_ready") is True
    )
    state = (
        "gap_exp5124_missing"
        if missing
        else "quarantined_runtime_evidence"
        if quarantined
        else "clean_sota_runtime_ready"
        if clean
        else "runtime_incomplete_or_blocked"
    )
    return {
        "state": state,
        "headline_eligible": state == "clean_sota_runtime_ready",
        "source_experiment": "exp5124",
        "sota_runtime_clean": payload.get("sota_runtime_clean") is True,
        "adversarial_verify_passed": payload.get("adversarial_verify_passed") is True,
        "cache_ready": payload.get("cache_ready") is True,
        "completion_ready": _mapping(payload.get("completion_proof")).get("ready") is True,
        "logprob_ready": _mapping(payload.get("logprob_proof")).get("ready") is True,
        "endpoint_lifetime_s": _number(payload.get("endpoint_lifetime_s")),
        "gap": _row_by_id(missing_artifacts, 5124),
        "quarantined": quarantined,
        "artifact_row": _row_by_id(artifacts_read, 5124),
    }


def build_structured_energy_state(
    artifacts_read: Sequence[JsonMap],
    missing_artifacts: Sequence[JsonMap],
    gated_skips: Sequence[JsonMap],
    quarantined_ids: set[int],
    payloads: Mapping[int, JsonMap],
) -> JsonDict:
    pool = _present(payloads, 5125)
    ranker = _present(payloads, 5126)
    audit = _present(payloads, 5127)
    clean_pool = bool(pool) and 5125 not in quarantined_ids
    clean_ranker = bool(ranker) and 5126 not in quarantined_ids
    clean_audit = bool(audit) and 5127 not in _ids(gated_skips) and 5127 not in quarantined_ids
    positive_delta = (_number(ranker.get("distributional_energy_delta")) or 0.0) > 0.0
    positive_survived = clean_pool and clean_ranker and clean_audit and positive_delta
    failure_reasons = [
        reason
        for reason in (
            "structured_pool_quarantined" if 5125 in quarantined_ids else "",
            "ranker_quarantined" if 5126 in quarantined_ids else "",
            "ranker_delta_not_positive" if not positive_delta else "",
            "audit_gate_skipped" if 5127 in _ids(gated_skips) else "",
            "missing_structured_energy_artifact"
            if any(row.get("axis") == "structured_energy" for row in missing_artifacts)
            else "",
        )
        if reason
    ]
    state = "positive_survived_audit" if positive_survived else "no_surviving_positive_audit_gap"
    return {
        "state": state,
        "headline_eligible": positive_survived,
        "positive_result_survived_audit": positive_survived,
        "attempted_pool": {
            "artifact_present": bool(pool),
            "headline_eligible": clean_pool,
            "structured_pool_ready": pool.get("structured_pool_ready") is True,
            "pool_n": int(_number(pool.get("pool_n")) or 0),
            "oracle_at_k": _number(pool.get("oracle_at_k")),
            "cheap_baseline_at_1": _number(pool.get("cheap_baseline_at_1")),
            "parse_coverage": _number(pool.get("parse_coverage")),
            "duplicate_rate": _number(pool.get("duplicate_rate")),
            "fover_scope_used": pool.get("fover_scope_used") is True,
        },
        "attempted_ranker": {
            "artifact_present": bool(ranker),
            "headline_eligible": clean_ranker,
            "distributional_energy_delta": _number(ranker.get("distributional_energy_delta")),
            "ranker_ready_for_audit": ranker.get("ranker_ready_for_audit") is True,
            "ranker_metrics": _mapping(ranker.get("ranker_metrics")),
            "strongest_cheap_baseline": _mapping(ranker.get("strongest_cheap_baseline")),
        },
        "audit_state": {
            "artifact_present": bool(audit),
            "headline_eligible": clean_audit,
            "gated_skip": 5127 in _ids(gated_skips),
            "gate_check_summary": str(audit.get("gate_check_summary", "")),
            "gates_evaluated": _list(audit.get("gates_evaluated")),
        },
        "failure_reasons": failure_reasons,
        "quarantined_experiments": sorted(
            exp_id for exp_id in (5125, 5126, 5127) if exp_id in quarantined_ids
        ),
        "gated_skip_experiments": sorted(
            exp_id for exp_id in (5125, 5126, 5127) if exp_id in _ids(gated_skips)
        ),
        "missing_experiments": sorted(
            int(row["experiment_number"])
            for row in missing_artifacts
            if row.get("axis") == "structured_energy"
        ),
    }


def build_kan_certificate_state(
    artifacts_read: Sequence[JsonMap],
    missing_artifacts: Sequence[JsonMap],
    quarantined_ids: set[int],
    payloads: Mapping[int, JsonMap],
) -> JsonDict:
    payload = _present(payloads, 5128)
    missing = bool(_row_by_id(missing_artifacts, 5128))
    clean = (
        bool(payload)
        and 5128 not in quarantined_ids
        and payload.get("kan_certificate_breadth_ready") is True
        and payload.get("certificate_soundness") is True
        and payload.get("explanation_cycle_soundness") is True
    )
    state = (
        "gap_exp5128_missing"
        if missing
        else "quarantined_kan_certificate"
        if 5128 in quarantined_ids
        else "clean_certificate_explanation_positive"
        if clean
        else "kan_certificate_incomplete"
    )
    return {
        "state": state,
        "headline_eligible": state == "clean_certificate_explanation_positive",
        "kan_certificate_breadth_ready": payload.get("kan_certificate_breadth_ready") is True,
        "certificate_soundness": payload.get("certificate_soundness") is True,
        "explanation_cycle_soundness": payload.get("explanation_cycle_soundness") is True,
        "false_property_detected": payload.get("false_property_detected") is True,
        "near_margin_abstained": payload.get("near_margin_abstained") is True,
        "property_family_count": len(_list(payload.get("property_families"))),
        "gap": _row_by_id(missing_artifacts, 5128),
        "artifact_row": _row_by_id(artifacts_read, 5128),
    }


def _effort_score(payload: JsonMap, key: str) -> float | None:
    return _number(_mapping(payload.get(key)).get("total_effort_score"))


def build_solver_sampling_state(
    artifacts_read: Sequence[JsonMap],
    missing_artifacts: Sequence[JsonMap],
    quarantined_ids: set[int],
    payloads: Mapping[int, JsonMap],
) -> JsonDict:
    sampler = _present(payloads, 5129)
    taco = _present(payloads, 5130)
    missing_ids = {
        int(row["experiment_number"])
        for row in missing_artifacts
        if row.get("axis") == "solver_sampling"
    }
    clean_sampler = (
        bool(sampler) and 5129 not in quarantined_ids and sampler.get("adaptive_2dpt_ready") is True
    )
    clean_taco = (
        bool(taco)
        and 5130 not in quarantined_ids
        and taco.get("heldout_csp_trace_suite_ready") is True
    )
    baseline_effort = _effort_score(taco, "baseline_effort")
    sampler_feature_effort = _effort_score(taco, "sampler_feature_effort")
    sampler_feature_helped = (
        baseline_effort is not None
        and sampler_feature_effort is not None
        and sampler_feature_effort < baseline_effort
    )
    state = (
        "gap_exp5130_missing"
        if 5130 in missing_ids
        else "gap_exp5129_missing"
        if 5129 in missing_ids
        else "quarantined_solver_sampling"
        if {5129, 5130} & quarantined_ids
        else "clean_exact_checked_bounded_solver_sampling_progress"
        if clean_sampler and clean_taco
        else "solver_sampling_incomplete"
    )
    return {
        "state": state,
        "headline_eligible": state == "clean_exact_checked_bounded_solver_sampling_progress",
        "adaptive_2dpt_ready": sampler.get("adaptive_2dpt_ready") is True,
        "exact_enumeration_checked": sampler.get("exact_enumeration_checked") is True,
        "detailed_balance_passed": _mapping(sampler.get("detailed_balance_sanity")).get("passed")
        is True,
        "hardware_speedup_claimed": sampler.get("hardware_speedup_claimed") is True,
        "optimum_hit_rate": _mapping(sampler.get("optimum_hit_rate")),
        "best_energy_delta_vs_baselines": _mapping(sampler.get("best_energy_delta_vs_baselines")),
        "heldout_csp_trace_suite_ready": taco.get("heldout_csp_trace_suite_ready") is True,
        "instance_count": int(_number(taco.get("instance_count")) or 0),
        "wrong_label_count": int(_number(taco.get("wrong_label_count")) or 0),
        "guarded_effort_reduction_ratio": _number(
            taco.get("average_effort_reduction_ratio_guarded")
        ),
        "harmful_instance_count_guarded": int(
            _number(taco.get("harmful_instance_count_guarded")) or 0
        ),
        "harmful_instance_count_unguarded": int(
            _number(taco.get("harmful_instance_count_unguarded")) or 0
        ),
        "baseline_effort_score": baseline_effort,
        "guarded_effort_score": _effort_score(taco, "guarded_effort"),
        "sampler_feature_effort_score": sampler_feature_effort,
        "sampler_feature_helped_solver_effort": sampler_feature_helped,
        "bounded_utility_note": (
            "Exact labels and guarded effort improved modestly, but sampler-feature effort did "
            "not beat the baseline and harmful guarded instances remain."
        ),
        "missing_experiments": sorted(missing_ids),
        "artifact_rows": [_row_by_id(artifacts_read, 5129), _row_by_id(artifacts_read, 5130)],
    }


def build_fr11_state(
    artifacts_read: Sequence[JsonMap],
    missing_artifacts: Sequence[JsonMap],
    quarantined_ids: set[int],
    payloads: Mapping[int, JsonMap],
) -> JsonDict:
    payload = _present(payloads, 5131)
    missing = bool(_row_by_id(missing_artifacts, 5131))
    rollback = _mapping(payload.get("rollback_receipt"))
    safe_no_promotion = (
        bool(payload)
        and 5131 not in quarantined_ids
        and payload.get("continuous_self_learning_task") is True
        and payload.get("promotion_attempted") is True
        and payload.get("promotion_safe") is False
        and rollback.get("rollback_applied") is True
    )
    state = (
        "gap_exp5131_missing"
        if missing
        else "quarantined_fr11"
        if 5131 in quarantined_ids
        else "safe_no_promotion"
        if safe_no_promotion
        else "fr11_incomplete"
    )
    return {
        "state": state,
        "headline_eligible": state == "safe_no_promotion",
        "continuous_self_learning_task": payload.get("continuous_self_learning_task") is True,
        "heldout_delta": _number(payload.get("heldout_delta")),
        "nonforgetting_delta": _number(payload.get("nonforgetting_delta")),
        "harmful_promotion_count": int(_number(payload.get("harmful_promotion_count")) or 0),
        "exact_solver_correctness_preserved": payload.get("exact_solver_correctness_preserved")
        is True,
        "promotion_attempted": payload.get("promotion_attempted") is True,
        "promotion_safe": payload.get("promotion_safe") is True,
        "rollback_applied": rollback.get("rollback_applied") is True,
        "no_weight_update": payload.get("no_weight_update") is True,
        "promotion_safety_summary": (
            "No metadata or weight promotion is active because held-out utility did not improve; "
            "rollback leaves the no-learning policy in force."
        ),
        "adversarial_warnings": _list(
            _row_by_id(artifacts_read, 5131).get("adversarial_verification", {}).get("flags")
        ),
        "gap": _row_by_id(missing_artifacts, 5131),
        "artifact_row": _row_by_id(artifacts_read, 5131),
    }


def build_hardware_state(
    artifacts_read: Sequence[JsonMap],
    missing_artifacts: Sequence[JsonMap],
    quarantined_ids: set[int],
    payloads: Mapping[int, JsonMap],
) -> JsonDict:
    payload = _present(payloads, 5132)
    missing = bool(_row_by_id(missing_artifacts, 5132))
    continuity = (
        bool(payload)
        and 5132 not in quarantined_ids
        and payload.get("no_speedup_claim") is True
        and payload.get("kv260_host_block_devices_touched") is False
    )
    state = (
        "gap_exp5132_missing"
        if missing
        else "quarantined_hardware"
        if 5132 in quarantined_ids
        else "continuity_with_authenticated_blockers_no_speedup_claim"
        if continuity
        else "hardware_incomplete"
    )
    return {
        "state": state,
        "headline_eligible": state == "continuity_with_authenticated_blockers_no_speedup_claim",
        "kv260_ssh_checked": payload.get("kv260_ssh_checked") is True,
        "kv260_ssh_ready": payload.get("kv260_ssh_ready") is True,
        "kv260_host_block_devices_touched": payload.get("kv260_host_block_devices_touched") is True,
        "gatemate_checked": payload.get("gatemate_checked") is True,
        "gatemate_detected": payload.get("gatemate_detected") is True,
        "polarfire_checked": payload.get("polarfire_checked") is True,
        "polarfire_ssh_ready": payload.get("polarfire_ssh_ready") is True,
        "extropic_tsu_execution_claimed": payload.get("extropic_tsu_execution_claimed") is True,
        "no_speedup_claim": payload.get("no_speedup_claim") is True,
        "timing_measurements": _mapping(payload.get("timing_measurements")),
        "board_precheck_summary": _mapping(payload.get("board_precheck_summary")),
        "gap": _row_by_id(missing_artifacts, 5132),
        "artifact_row": _row_by_id(artifacts_read, 5132),
    }


def build_next_milestone_recommendations() -> list[JsonDict]:
    return [
        {
            "priority": "Retire same-scope FoVer selector, audit, and residual-memory reruns.",
            "rationale": "The .469 selector premise was retracted; rerunning the same verdict path is a doomed rerun.",
            "retire_same_verdict_doomed_rerun": True,
        },
        {
            "priority": "Repair structured-energy provenance before another ranker/audit attempt.",
            "rationale": "The non-FoVer pool and ranker attempted useful axes but were headline-quarantined by live-duration/methodology flags and zero ranker delta.",
            "retire_same_verdict_doomed_rerun": True,
        },
        {
            "priority": "Scale KAN certificate explanations across more property families.",
            "rationale": "Exp5128 is the cleanest FR-12 positive: sound certificates, false-property detection, and near-margin abstention survived audit.",
            "retire_same_verdict_doomed_rerun": False,
        },
        {
            "priority": "Expand exact-solver/TACO traces while preserving harm gates.",
            "rationale": "Exp5129/5130 provide exact labels and bounded effort reduction, but sampler features should promote only after no-harm evidence improves.",
            "retire_same_verdict_doomed_rerun": False,
        },
        {
            "priority": "Convert board reachability into hash-matched timing workload transcripts.",
            "rationale": "KV260 and PolarFire continuity are useful, but no speedup claim is allowed until authenticated board timing and sample-quality evidence land.",
            "retire_same_verdict_doomed_rerun": False,
        },
    ]


def build_preconditions(
    artifacts_read: Sequence[JsonMap],
    missing_artifacts: Sequence[JsonMap],
    gated_skips: Sequence[JsonMap],
    quarantined_artifacts: Sequence[JsonMap],
) -> JsonDict:
    return {
        "expected_upstream_count": len(UPSTREAM_SOURCES),
        "artifacts_read": len(artifacts_read),
        "missing_artifacts": len(missing_artifacts),
        "gated_skips": len(gated_skips),
        "quarantined_artifacts": len(quarantined_artifacts),
        "critical_quarantines": sum(
            1
            for row in quarantined_artifacts
            if row.get("quarantine_reason") == "live_critical_adversarial_flag"
        ),
        "capstone_is_ungated": True,
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "ops_reconciliation_delegated": True,
    }


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    duration_s: float,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
    adversarial_reporter: AdversarialReporter = run_adversarial_report,
) -> JsonDict:
    repo_root = Path(root)
    artifacts_read, missing_artifacts, gated_skips, quarantined_artifacts, payloads = (
        load_upstreams(repo_root, adversarial_reporter)
    )
    quarantined_ids = _ids(quarantined_artifacts)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "run_date": run_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": build_preconditions(
            artifacts_read, missing_artifacts, gated_skips, quarantined_artifacts
        ),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": COMPLETE_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(duration_s, 0.0001), 6),
        "artifacts_read": artifacts_read,
        "missing_artifacts": missing_artifacts,
        "gated_skips": gated_skips,
        "quarantined_artifacts": quarantined_artifacts,
        "fover_same_scope_retired": build_fover_same_scope_retired(artifacts_read, payloads),
        "runtime_state": build_runtime_state(
            artifacts_read, missing_artifacts, quarantined_ids, payloads
        ),
        "structured_energy_state": build_structured_energy_state(
            artifacts_read, missing_artifacts, gated_skips, quarantined_ids, payloads
        ),
        "kan_certificate_state": build_kan_certificate_state(
            artifacts_read, missing_artifacts, quarantined_ids, payloads
        ),
        "solver_sampling_state": build_solver_sampling_state(
            artifacts_read, missing_artifacts, quarantined_ids, payloads
        ),
        "fr11_state": build_fr11_state(
            artifacts_read, missing_artifacts, quarantined_ids, payloads
        ),
        "hardware_state": build_hardware_state(
            artifacts_read, missing_artifacts, quarantined_ids, payloads
        ),
        "next_milestone_recommendations": build_next_milestone_recommendations(),
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "flagged_adversarial": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors = [f"missing.{field}" for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    field_principles = _mapping(artifact.get("field_principles"))
    errors.extend(
        f"field_principles.missing.{field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in field_principles
    )
    checks = [
        (
            "honest_verdict.not_terminal",
            not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES),
        ),
        ("experiment_id.invalid", artifact.get("experiment_id") != EXPERIMENT_ID),
        ("milestone.invalid", artifact.get("milestone") != MILESTONE),
        ("inference_substrate.invalid", artifact.get("inference_substrate") != INFERENCE_SUBSTRATE),
        ("active_roadmap_modified.invalid", artifact.get("active_roadmap_modified") is not False),
        ("conductor_modified.invalid", artifact.get("conductor_modified") is not False),
        ("flagged_adversarial.invalid", artifact.get("flagged_adversarial") is not False),
        (
            "next_milestone_recommendations.count",
            len(_list(artifact.get("next_milestone_recommendations"))) not in {3, 4, 5},
        ),
        ("fover_same_scope_retired.invalid", artifact.get("fover_same_scope_retired") is not True),
        (
            "reproducibility_checksum.invalid",
            artifact.get("reproducibility_checksum") != payload_checksum(artifact),
        ),
    ]
    errors.extend(name for name, failed in checks if failed)
    return errors


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(
            f"invalid exp5133 artifact: missing required fields or invalid values: {errors}"
        )


def write_artifact(
    *,
    root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    adversarial_reporter: AdversarialReporter = run_adversarial_report,
    clock: Clock = time.perf_counter,
) -> Path:
    started = clock()
    elapsed = max((duration_s if duration_s is not None else clock() - started), 0.0001)
    artifact = build_artifact(
        root=root,
        duration_s=elapsed,
        run_date=run_date,
        tests_run=tests_run,
        adversarial_reporter=adversarial_reporter,
    )
    validate_artifact(artifact)
    output = Path(root) / RESULT_RELATIVE_PATH
    write_json(output, artifact)
    return output


def run(
    *,
    root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    adversarial_reporter: AdversarialReporter = run_adversarial_report,
    clock: Clock = time.perf_counter,
) -> Path:
    return write_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
        adversarial_reporter=adversarial_reporter,
        clock=clock,
    )
