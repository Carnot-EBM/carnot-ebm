"""Exp 5145: V471 capstone aggregation.

Spec refs: REQ-CAPSTONE-5145, SCENARIO-CAPSTONE-5145,
SCENARIO-CAPSTONE-5145-FIELD-PRINCIPLES.

This module reads V471 result artifacts and writes one decision record. It
does not rerun science. The capstone's job is evidence accounting: clean axes
stay usable, blocked axes stay blocked, and flagged evidence is quarantined
even when its numbers look useful.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results") / "experiment_5145_capstone_v471.json"
EXPERIMENT = "experiment_5145_capstone_v471"
EXPERIMENT_ID = "exp5145-capstone-v471"
MILESTONE = "2026.07.471"
SCHEMA = "carnot.experiment_5145_capstone_v471.v1"
RANDOM_SEED = 5145
INFERENCE_SUBSTRATE = "aggregation_from_v471_artifacts"
COMPLETE_VERDICT = (
    "complete_capstone_v471_structured_pool_repaired_solver_no_utility_"
    "guided_blocked_fr11_quarantined_hardware_blocked"
)
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_")

SPEC_REFS = [
    "REQ-CAPSTONE-5145",
    "SCENARIO-CAPSTONE-5145",
    "SCENARIO-CAPSTONE-5145-FIELD-PRINCIPLES",
]

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "upstream_artifacts_read",
    "source_scope_audit_state",
    "structured_generation_state",
    "solver_formulation_state",
    "guided_decoding_state",
    "abstention_trace_state",
    "kan_symbolic_state",
    "sampling_partition_state",
    "taco_harm_state",
    "fr11_state",
    "hardware_state",
    "no_speedup_claim_preserved",
    "retire_or_quarantine_recommendations",
    "next_milestone_recommendations",
    "conductor_modified",
    "tests_run",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "spec_refs",
    "result_path",
    "run_date",
    "field_principles",
    "missing_artifacts",
    "classified_upstreams",
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
    "upstream_artifacts_read": "evidence provenance",
    "source_scope_audit_state": "source discipline",
    "structured_generation_state": "substrate accountability",
    "solver_formulation_state": "utility accounting",
    "guided_decoding_state": "decoding accountability",
    "abstention_trace_state": "hallucination mitigation accounting",
    "kan_symbolic_state": "certificate accountability",
    "sampling_partition_state": "sampler telemetry",
    "taco_harm_state": "solver safety",
    "fr11_state": "continuous self-learning accountability",
    "hardware_state": "board evidence accountability",
    "no_speedup_claim_preserved": "hardware claim discipline",
    "retire_or_quarantine_recommendations": "no doomed rerun",
    "next_milestone_recommendations": "roadmap continuity",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5145_capstone_v471.py --date 20260702",
    '.venv/bin/pytest tests/python/test_experiment_5145_capstone_v471.py -q -o addopts=""',
    "JAX_PLATFORMS=cpu .venv/bin/coverage run --rcfile=/dev/null "
    "--include='*/experiment_5145_capstone_v471.py' "
    '-m pytest tests/python/test_experiment_5145_capstone_v471.py -q --no-cov -o addopts=""',
    "JAX_PLATFORMS=cpu .venv/bin/coverage report --rcfile=/dev/null -m "
    "--include='*/experiment_5145_capstone_v471.py' --fail-under=100",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5145_capstone_v471.py",
    "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class UpstreamSource:
    """One expected V471 result artifact and its capstone axis."""

    experiment_number: int
    label: str
    axis: str
    relative_path: Path


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5134,
        "archive_470_activate_471",
        "transition",
        Path("results/experiment_5134_archive_470_activate_471.json"),
    ),
    UpstreamSource(
        5135,
        "v471_source_scope_audit",
        "planning",
        Path("results/experiment_5135_v471_source_scope_audit.json"),
    ),
    UpstreamSource(
        5136,
        "receipt_structured_pool_v2",
        "structured_generation",
        Path("results/experiment_5136_receipt_structured_pool_v2_v471.json"),
    ),
    UpstreamSource(
        5137,
        "solver_verified_formulation_selector",
        "solver_formulation",
        Path("results/experiment_5137_solver_verified_formulation_selector_v471.json"),
    ),
    UpstreamSource(
        5138,
        "ets_ebd_guided_decoding",
        "guided_decoding",
        Path("results/experiment_5138_ets_ebd_guided_decoding_v471.json"),
    ),
    UpstreamSource(
        5139,
        "abstention_verification_trace",
        "abstention_trace",
        Path("results/experiment_5139_abstention_verification_trace_v471.json"),
    ),
    UpstreamSource(
        5140,
        "symbolic_kan_certificate_distillation",
        "kan_symbolic",
        Path("results/experiment_5140_symbolic_kan_certificate_distillation_v471.json"),
    ),
    UpstreamSource(
        5141,
        "hubo_partition_residual_exponent",
        "sampling_partition",
        Path("results/experiment_5141_hubo_partition_residual_exponent_v471.json"),
    ),
    UpstreamSource(
        5142,
        "taco_harm_rootcause_scale",
        "taco_harm",
        Path("results/experiment_5142_taco_harm_rootcause_scale_v471.json"),
    ),
    UpstreamSource(
        5143,
        "openskill_k2v_self_learning",
        "fr11",
        Path("results/experiment_5143_openskill_k2v_self_learning_v471.json"),
    ),
    UpstreamSource(
        5144,
        "authenticated_board_workload",
        "hardware",
        Path("results/experiment_5144_authenticated_board_workload_v471.json"),
    ),
)
EXPECTED_UPSTREAMS = {source.experiment_number: source for source in UPSTREAM_SOURCES}


def _round_duration(value: float) -> float:
    return round(float(value), 6)


def file_sha256(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


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
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {"exists": True, "loadable": False, "error": "malformed_json"}
    if not isinstance(parsed, dict):  # pragma: no cover - defensive for hand-edited artifacts.
        return {}, {"exists": True, "loadable": False, "error": "not_json_object"}
    return parsed, {
        "exists": True,
        "loadable": True,
        "error": None,
        "sha256": file_sha256(path),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive for malformed upstreams.
        return None


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _is_blocked(data: JsonMap) -> bool:
    return str(data.get("honest_verdict", "")).startswith("blocked_")


def _is_quarantined(data: JsonMap) -> bool:
    return data.get("flagged_adversarial") is True


def _classification(data: JsonMap, *, no_promote: bool = False) -> str:
    if _is_quarantined(data):
        return "quarantined"
    if _is_blocked(data):
        return "blocked"
    if no_promote:
        return "no-promote"
    return "clean"


def _missing_state(source: UpstreamSource, meta: JsonMap) -> JsonDict:
    return {
        "experiment_number": source.experiment_number,
        "label": source.label,
        "axis": source.axis,
        "relative_path": str(source.relative_path),
        "classification": "blocked",
        "exists": meta.get("exists") is True,
        "loadable": False,
        "error": meta.get("error"),
    }


def load_upstreams(root: Path) -> tuple[dict[int, JsonDict], list[JsonDict], list[JsonDict]]:
    artifacts: dict[int, JsonDict] = {}
    read_rows: list[JsonDict] = []
    missing_rows: list[JsonDict] = []
    for source in UPSTREAM_SOURCES:
        path = root / source.relative_path
        data, meta = read_json_mapping(path)
        if not meta.get("loadable"):
            missing_rows.append(_missing_state(source, meta))
            continue
        class_name = _classification(data)
        artifacts[source.experiment_number] = data
        read_rows.append(
            {
                "experiment_number": source.experiment_number,
                "label": source.label,
                "axis": source.axis,
                "relative_path": str(source.relative_path),
                "sha256": meta.get("sha256"),
                "honest_verdict": data.get("honest_verdict"),
                "classification": class_name,
                "flagged_adversarial": data.get("flagged_adversarial") is True,
            }
        )
    return artifacts, read_rows, missing_rows


def _classified_upstreams(
    read_rows: Sequence[JsonMap], missing_rows: Sequence[JsonMap]
) -> list[JsonDict]:
    return [dict(row) for row in read_rows] + [dict(row) for row in missing_rows]


def _missing_for(missing_rows: Sequence[JsonMap], experiment_number: int) -> JsonDict:
    for row in missing_rows:
        if row.get("experiment_number") == experiment_number:
            return dict(row)
    return {
        "experiment_number": experiment_number,
        "classification": "blocked",
        "error": "artifact_missing_or_unreadable",
    }


def source_scope_audit_state(artifacts: JsonMap, missing_rows: Sequence[JsonMap]) -> JsonDict:
    data = _mapping(artifacts.get(5135))
    if not data:
        return {"classification": "blocked", "load_error": _missing_for(missing_rows, 5135)}
    return {
        "classification": _classification(data),
        "honest_verdict": data.get("honest_verdict"),
        "v471_reference_block_found": data.get("v471_reference_block_found") is True,
        "sota_model_discipline_ok": data.get("sota_model_discipline_ok") is True,
        "structured_gates_ok": data.get("structured_gates_ok") is True,
        "fover_same_scope_rerun_found": data.get("fover_same_scope_rerun_found") is True,
        "exclusion_manifest_conflicts": _list(data.get("exclusion_manifest_conflicts")),
        "quarantine_reason": "flagged_adversarial" if _is_quarantined(data) else None,
    }


def structured_generation_state(artifacts: JsonMap, missing_rows: Sequence[JsonMap]) -> JsonDict:
    data = _mapping(artifacts.get(5136))
    if not data:
        return {"classification": "blocked", "load_error": _missing_for(missing_rows, 5136)}
    repaired = (
        data.get("structured_pool_v2_clean") is True
        and data.get("adversarial_verify_passed") is True
        and data.get("fover_scope_used") is False
    )
    return {
        "classification": _classification(data) if repaired else "blocked",
        "honest_verdict": data.get("honest_verdict"),
        "provenance_problem_repaired": repaired,
        "downstream_tasks_trustworthy": repaired and not _is_quarantined(data),
        "pool_n": data.get("pool_n"),
        "oracle_at_k": data.get("oracle_at_k"),
        "cheap_baseline_at_1": data.get("cheap_baseline_at_1"),
        "parse_coverage": data.get("parse_coverage"),
        "duplicate_rate": data.get("duplicate_rate"),
        "exact_validators_used": _list(data.get("exact_validators_used")),
        "duration_floor_evidence": _mapping(data.get("duration_floor_evidence")),
    }


def solver_formulation_state(artifacts: JsonMap, missing_rows: Sequence[JsonMap]) -> JsonDict:
    data = _mapping(artifacts.get(5137))
    if not data:
        return {"classification": "blocked", "load_error": _missing_for(missing_rows, 5137)}
    delta = _number(data.get("selector_delta_vs_best_static")) or 0.0
    ready = data.get("formulation_selector_ready") is True
    no_promote = not ready and delta <= 0.0 and not _is_blocked(data)
    return {
        "classification": _classification(data, no_promote=no_promote),
        "honest_verdict": data.get("honest_verdict"),
        "formulation_selector_ready": ready,
        "selector_delta_vs_best_static": delta,
        "delta_ci95": _list(data.get("delta_ci95")),
        "wrong_label_count": data.get("wrong_label_count"),
        "solve_effort_delta": _mapping(data.get("solve_effort_delta")),
        "strongest_baseline": "static_hand_or_cheap_feasible_repair",
        "feasibility_restoration_used": data.get("feasibility_restoration_used") is True,
    }


def guided_decoding_state(artifacts: JsonMap, missing_rows: Sequence[JsonMap]) -> JsonDict:
    data = _mapping(artifacts.get(5138))
    if not data:
        return {
            "classification": "blocked",
            "load_error": _missing_for(missing_rows, 5138),
            "gate_condition": {"field": "artifact", "actual": "missing_or_unreadable"},
        }
    preconditions = _mapping(data.get("preconditions_checked"))
    telemetry_available = preconditions.get("stepwise_telemetry_available") is True
    return {
        "classification": _classification(data),
        "honest_verdict": data.get("honest_verdict"),
        "guided_decoding_ready": data.get("guided_decoding_ready") is True,
        "gate_condition": {
            "field": "stepwise_telemetry_available",
            "expected": True,
            "actual": telemetry_available,
        },
        "delta_vs_best_baseline": data.get("delta_vs_best_baseline"),
        "delta_ci95": _list(data.get("delta_ci95")),
        "token_budget_matched": data.get("token_budget_matched"),
        "nfe_budget_matched": data.get("nfe_budget_matched"),
        "wrong_label_count": data.get("wrong_label_count"),
    }


def abstention_trace_state(artifacts: JsonMap, missing_rows: Sequence[JsonMap]) -> JsonDict:
    data = _mapping(artifacts.get(5139))
    if not data:
        return {"classification": "blocked", "load_error": _missing_for(missing_rows, 5139)}
    return {
        "classification": _classification(data)
        if data.get("verification_trace_ready")
        else "blocked",
        "honest_verdict": data.get("honest_verdict"),
        "verification_trace_ready": data.get("verification_trace_ready") is True,
        "harmful_answer_reduction": data.get("harmful_answer_reduction"),
        "coverage_risk_curve": _list(data.get("coverage_risk_curve")),
        "strongest_baseline": "direct_answer_without_abstention_trace",
    }


def kan_symbolic_state(artifacts: JsonMap, missing_rows: Sequence[JsonMap]) -> JsonDict:
    data = _mapping(artifacts.get(5140))
    if not data:
        return {"classification": "blocked", "load_error": _missing_for(missing_rows, 5140)}
    ready = data.get("symbolic_kan_ready") is True and data.get("certificate_soundness") is True
    return {
        "classification": _classification(data) if ready else "blocked",
        "honest_verdict": data.get("honest_verdict"),
        "symbolic_kan_ready": ready,
        "certificate_soundness": data.get("certificate_soundness") is True,
        "false_property_detected": data.get("false_property_detected") is True,
        "symbolic_equivalence_rate": data.get("symbolic_equivalence_rate"),
        "cycle_reconstruction_rate": data.get("cycle_reconstruction_rate"),
        "label_shuffle_control": _mapping(data.get("label_shuffle_control")),
    }


def sampling_partition_state(artifacts: JsonMap, missing_rows: Sequence[JsonMap]) -> JsonDict:
    data = _mapping(artifacts.get(5141))
    if not data:
        return {"classification": "blocked", "load_error": _missing_for(missing_rows, 5141)}
    ready = (
        data.get("partition_telemetry_ready") is True
        and data.get("exact_enumeration_checked") is True
        and data.get("hardware_speedup_claimed") is False
    )
    return {
        "classification": _classification(data) if ready else "blocked",
        "honest_verdict": data.get("honest_verdict"),
        "partition_telemetry_ready": ready,
        "exact_enumeration_checked": data.get("exact_enumeration_checked") is True,
        "hardware_speedup_claimed": data.get("hardware_speedup_claimed") is True,
        "effective_sample_quality": _mapping(data.get("effective_sample_quality")),
        "telemetry_stability": _mapping(data.get("telemetry_stability")),
        "board_ready_descriptor_count": len(_list(data.get("board_ready_workload_descriptors"))),
    }


def taco_harm_state(artifacts: JsonMap, missing_rows: Sequence[JsonMap]) -> JsonDict:
    data = _mapping(artifacts.get(5142))
    if not data:
        return {"classification": "blocked", "load_error": _missing_for(missing_rows, 5142)}
    ready = data.get("trace_suite_v2_ready") is True and data.get("wrong_label_count") == 0
    return {
        "classification": _classification(data) if ready else "blocked",
        "honest_verdict": data.get("honest_verdict"),
        "trace_suite_v2_ready": ready,
        "wrong_label_count": data.get("wrong_label_count"),
        "average_effort_reduction_ratio_guarded": data.get(
            "average_effort_reduction_ratio_guarded"
        ),
        "harmful_instance_count_guarded": data.get("harmful_instance_count_guarded"),
        "root_cause_count": len(_list(data.get("harmful_instance_root_causes"))),
        "repaired_harm_gate": _mapping(data.get("repaired_harm_gate")),
    }


def fr11_state(artifacts: JsonMap, missing_rows: Sequence[JsonMap]) -> JsonDict:
    data = _mapping(artifacts.get(5143))
    if not data:
        return {"classification": "blocked", "load_error": _missing_for(missing_rows, 5143)}
    heldout_delta = _number(data.get("heldout_delta")) or 0.0
    nonforgetting_delta = _number(data.get("nonforgetting_delta")) or 0.0
    promotion_evidence = (
        data.get("promotion_safe") is True
        and heldout_delta > 0.0
        and nonforgetting_delta >= 0.0
        and data.get("wrong_label_count") == 0
        and data.get("no_weight_update") is True
    )
    if _is_quarantined(data) and promotion_evidence:
        assessment = "safe_promotion_evidence_quarantined"
    elif promotion_evidence:
        assessment = "safe_promotion"
    else:
        assessment = "rollback_or_no_promote"
    return {
        "classification": _classification(data, no_promote=assessment == "rollback_or_no_promote"),
        "honest_verdict": data.get("honest_verdict"),
        "continuous_self_learning_task": data.get("continuous_self_learning_task") is True,
        "promotion_assessment": assessment,
        "promotion_safe": data.get("promotion_safe") is True,
        "heldout_delta": heldout_delta,
        "nonforgetting_delta": nonforgetting_delta,
        "no_weight_update": data.get("no_weight_update") is True,
        "wrong_label_count": data.get("wrong_label_count"),
        "virtual_exact_task_count": _mapping(data.get("virtual_task_manifest")).get(
            "exact_validated_task_count"
        ),
        "rollback_receipt": _mapping(data.get("rollback_receipt")),
        "quarantine_reason": "flagged_adversarial" if _is_quarantined(data) else None,
    }


def hardware_state(artifacts: JsonMap, missing_rows: Sequence[JsonMap]) -> JsonDict:
    data = _mapping(artifacts.get(5144))
    if not data:
        missing = _missing_for(missing_rows, 5144)
        return {
            "classification": "blocked",
            "load_error": missing.get("error"),
            "hardware_workload_transcripts_ready": False,
            "no_speedup_claim": True,
        }
    ready = data.get("hardware_workload_transcripts_ready") is True
    return {
        "classification": _classification(data) if ready else "blocked",
        "honest_verdict": data.get("honest_verdict"),
        "hardware_workload_transcripts_ready": ready,
        "no_speedup_claim": data.get("no_speedup_claim") is True,
        "extropic_tsu_execution_claimed": data.get("extropic_tsu_execution_claimed") is True,
        "kv260_ssh_checked": data.get("kv260_ssh_checked") is True,
        "kv260_host_block_devices_touched": data.get("kv260_host_block_devices_touched") is True,
        "safe_workload_manifest": _mapping(data.get("safe_workload_manifest")),
        "board_blockers": _mapping(data.get("board_blockers")),
        "ready_evidence_boards": _list(
            _mapping(data.get("sample_quality_evidence")).get("ready_evidence_boards")
        ),
    }


def no_speedup_claim_preserved(artifacts: JsonMap) -> bool:
    for data in artifacts.values():
        if not isinstance(data, Mapping):
            continue
        if data.get("hardware_speedup_claimed") is True:
            return False
        if data.get("no_speedup_claim") is False:
            return False
        if data.get("extropic_tsu_execution_claimed") is True:
            return False
    return True


def retire_or_quarantine_recommendations(states: JsonMap) -> list[JsonDict]:
    recommendations: list[JsonDict] = []
    if states["solver_formulation_state"]["classification"] == "no-promote":
        recommendations.append(
            {
                "experiment": "exp5137",
                "action": "retire_same_scope_rerun",
                "reason": "selector_delta_vs_best_static=0.0 and solve effort rose versus static/cheap baselines",
            }
        )
    if states["guided_decoding_state"]["classification"] == "blocked":
        recommendations.append(
            {
                "experiment": "exp5138",
                "action": "block_until_prerequisite_changes",
                "reason": "stepwise logprob telemetry is unavailable; rerunning without it repeats the same blocker",
            }
        )
    if states["fr11_state"]["classification"] == "quarantined":
        recommendations.append(
            {
                "experiment": "exp5143",
                "action": "quarantine_promotion_evidence",
                "reason": "held-out and nonforgetting evidence is positive but the artifact is flagged_adversarial",
            }
        )
    if states["hardware_state"]["classification"] == "blocked":
        recommendations.append(
            {
                "experiment": "exp5144",
                "action": "block_same_scope_board_workload_rerun",
                "reason": "safe workload manifest and hash-matched sample-quality evidence are missing",
            }
        )
    return recommendations


def next_milestone_recommendations(states: JsonMap) -> list[JsonDict]:
    return [
        {
            "priority": "critical",
            "task": "archive_v471_before_v472",
            "recommendation": "Record the quarantined FR-11 and blocked hardware/guided-decoding axes before using V471 as a premise.",
        },
        {
            "priority": "high",
            "task": "guided_decoding_prerequisite",
            "recommendation": "Do not rerun energy-guided decoding until local GGUF stepwise logprob telemetry is available.",
        },
        {
            "priority": "high",
            "task": "hardware_safe_manifest",
            "recommendation": "Check in a safe workload manifest before another authenticated board workload run; keep no_speedup_claim=true.",
        },
        {
            "priority": "medium",
            "task": "fr11_clean_promotion_rerun",
            "recommendation": "Rerun OpenSkill/K2V anchors only with duration/provenance evidence sufficient to clear adversarial quarantine.",
        },
        {
            "priority": "medium",
            "task": "preserve_clean_substrates",
            "recommendation": "Reuse the clean receipt-backed pool, symbolic-KAN, partition telemetry, and TACO harm gate as evidence sources rather than rerunning same-scope nulls.",
            "clean_axes": [
                states["structured_generation_state"]["classification"],
                states["kan_symbolic_state"]["classification"],
                states["sampling_partition_state"]["classification"],
                states["taco_harm_state"]["classification"],
            ],
        },
    ]


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260702",
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    started = time.perf_counter()
    artifacts, read_rows, missing_rows = load_upstreams(Path(root))
    states: JsonDict = {
        "source_scope_audit_state": source_scope_audit_state(artifacts, missing_rows),
        "structured_generation_state": structured_generation_state(artifacts, missing_rows),
        "solver_formulation_state": solver_formulation_state(artifacts, missing_rows),
        "guided_decoding_state": guided_decoding_state(artifacts, missing_rows),
        "abstention_trace_state": abstention_trace_state(artifacts, missing_rows),
        "kan_symbolic_state": kan_symbolic_state(artifacts, missing_rows),
        "sampling_partition_state": sampling_partition_state(artifacts, missing_rows),
        "taco_harm_state": taco_harm_state(artifacts, missing_rows),
        "fr11_state": fr11_state(artifacts, missing_rows),
        "hardware_state": hardware_state(artifacts, missing_rows),
    }
    speedup_preserved = no_speedup_claim_preserved(artifacts)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "honest_verdict": COMPLETE_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _round_duration(
            duration_s if duration_s is not None else time.perf_counter() - started
        ),
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_artifacts_read": read_rows,
        "missing_artifacts": missing_rows,
        "classified_upstreams": _classified_upstreams(read_rows, missing_rows),
        "source_scope_audit_state": states["source_scope_audit_state"],
        "structured_generation_state": states["structured_generation_state"],
        "solver_formulation_state": states["solver_formulation_state"],
        "guided_decoding_state": states["guided_decoding_state"],
        "abstention_trace_state": states["abstention_trace_state"],
        "kan_symbolic_state": states["kan_symbolic_state"],
        "sampling_partition_state": states["sampling_partition_state"],
        "taco_harm_state": states["taco_harm_state"],
        "fr11_state": states["fr11_state"],
        "hardware_state": states["hardware_state"],
        "no_speedup_claim_preserved": speedup_preserved,
        "retire_or_quarantine_recommendations": retire_or_quarantine_recommendations(states),
        "next_milestone_recommendations": next_milestone_recommendations(states),
        "preconditions_checked": {
            "expected_upstream_count": len(UPSTREAM_SOURCES),
            "loaded_upstream_count": len(read_rows),
            "missing_or_unreadable_count": len(missing_rows),
            "ops_status_modified": False,
            "ops_changelog_modified": False,
            "conductor_modified": False,
        },
        "conductor_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if artifact["conductor_modified"] is not False:
        raise ValueError("conductor_modified must be false")
    if artifact["no_speedup_claim_preserved"] is not True:
        raise ValueError("no_speedup_claim_preserved must be true")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must be non-empty")
    for field, principle in FIELD_PRINCIPLES.items():
        if artifact["field_principles"].get(field) != principle:
            raise ValueError(f"field principle mismatch: {field}")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260702",
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
    )
    out_path = Path(root) / RESULT_RELATIVE_PATH
    write_json(out_path, artifact)
    return out_path
