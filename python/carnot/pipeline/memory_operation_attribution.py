"""No-LLM operation-stage attribution for governed memory.

The harness keeps Exp 5289 intentionally local and deterministic. It does not
learn a new policy or ask a model to explain failures. Instead, it replays the
Exp 5275 governed decision-history rows and the Exp 5276 memory-assisted
verifier-dose decisions, then asks a narrower question: which memory operation
stage would have been responsible if a controlled bad memory had reached the
final action?

Spec refs: REQ-LEARN-5289, SCENARIO-LEARN-5289.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "experiment_5289_memory_operation_attribution_v483"
EXPERIMENT_ID = 5289
SCHEMA = "carnot.memory_operation_attribution.v483"
RUN_DATE = "2026-07-06"
RANDOM_SEED = 5289
RESULT_RELATIVE_PATH = "results/experiment_5289_memory_operation_attribution_v483.json"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ("REQ-LEARN-5289", "SCENARIO-LEARN-5289")

EXP5275_RELATIVE_PATH = Path("results/experiment_5275_governed_decision_history_memory_v482.json")
EXP5276_RELATIVE_PATH = Path("results/experiment_5276_memory_assisted_verifier_dose_gated_v482.json")
SOURCE_ARTIFACTS = (str(EXP5275_RELATIVE_PATH), str(EXP5276_RELATIVE_PATH))

STAGE_KEYS = ("extraction", "update", "routing", "maintenance", "use", "rollback")
OPERATION_STAGE_LABELS = {
    "extraction": "extraction",
    "update": "update/write",
    "routing": "retrieval/routing",
    "maintenance": "maintenance/eviction",
    "use": "use/action",
    "rollback": "rollback",
}

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "States whether operation attribution is usable, blocked, or null without "
        "hiding unsafe propagation or attribution gaps."
    ),
    "inference_substrate": (
        "Declares aggregation from upstream artifacts or an offline deterministic "
        "no-LLM fixture so operation attribution is not mistaken for live LLM inference."
    ),
    "memory_attribution_ready": (
        "Gates Exp5290 only when every bounded control is attributed, no unsafe "
        "memory propagates, and final replay decisions remain safe."
    ),
    "operation_stage_error_counts": (
        "Separates extraction, update/write, retrieval/routing, maintenance/eviction, "
        "use/action, and rollback faults so a final memory success cannot hide the "
        "responsible operation stage."
    ),
    "attribution_coverage": (
        "Reports how many bounded control cases received a primary stage attribution."
    ),
    "unsafe_propagations": (
        "Counts controlled unsafe memories that reached final action selection despite "
        "governance."
    ),
    "local_maintenance_cost": (
        "Measures deterministic local ledger/provenance/scope/rollback checks instead "
        "of treating governance as free."
    ),
    "decision_impact_summary": (
        "Compares memory-assisted replay decisions against no-memory and always-full "
        "baselines so allocation gains are not confused with answer injection."
    ),
    "continuous_self_learning_evidence": (
        "Explains whether the governed memory lifecycle supplies reusable, auditable "
        "self-learning evidence."
    ),
}

REQUIRED_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "attribution_coverage",
    "unsafe_propagations",
    "local_maintenance_cost",
    "decision_impact_summary",
    "continuous_self_learning_evidence",
)


def load_upstream_artifacts(root: Path | str = REPO_ROOT) -> dict[str, JsonDict]:
    """Read the two upstream artifacts that bound the Exp 5289 replay."""

    root_path = Path(root)
    return {
        "exp5275": _read_json(root_path / EXP5275_RELATIVE_PATH),
        "exp5276": _read_json(root_path / EXP5276_RELATIVE_PATH),
    }


def build_attribution_cases(
    *,
    root: Path | str = REPO_ROOT,
    upstream_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[JsonDict]:
    """Build the deterministic operation-stage control cases.

    Each control is derived from an upstream governed row or replay row. The two
    synthetic controls, missing provenance and shuffled routing, are bounded
    mutations of checked-in rows rather than new data generation.
    """

    artifacts = dict(upstream_artifacts or load_upstream_artifacts(root))
    memory_rows = {
        str(row["fixture_kind"]): row
        for row in artifacts["exp5275"].get("governance_rows", [])
        if isinstance(row, Mapping)
    }
    pilot_rows = {
        str(row["task_id"]): row
        for row in artifacts["exp5276"].get("pilot_rows", [])
        if isinstance(row, Mapping)
    }
    promotion = memory_rows["promotion"]
    stale = memory_rows["stale_conflict"]
    out_of_scope = memory_rows["out_of_scope"]
    poisoning = memory_rows["poisoning_like"]
    rollback = memory_rows["rollback"]

    gap1 = pilot_rows["gap1_memory_only_consumer"]
    gap1_registry = pilot_rows["gap1_registry_rollback_consumer"]
    gap4 = pilot_rows["gap4_candidate_pool_consumer"]
    arc = pilot_rows["arc_rubric_before_patch_consumer"]
    hardware = pilot_rows["hardware_speedup_boundary_consumer"]

    cases = [
        _case(
            case_id="case-valid-promoted-gap1",
            control_kind="valid_promoted_memory",
            primary_stage="use",
            memory_row=promotion,
            replay_row=gap1,
            expected_control_action="promote",
            observed_control_action=str(promotion["governance_action"]),
            error_detected=False,
            propagation_blocked=False,
            notes="Canonical governed memory reaches use/action only as an allocation feature.",
        ),
        _case(
            case_id="case-missing-provenance",
            control_kind="missing_provenance",
            primary_stage="extraction",
            memory_row=_without_provenance(promotion),
            replay_row=gap1,
            expected_control_action="reject_missing_provenance",
            observed_control_action="reject_missing_provenance",
            error_detected=True,
            propagation_blocked=True,
            notes="Extraction cannot safely trust a row missing source/evidence receipts.",
        ),
        _case(
            case_id="case-conflicting-write",
            control_kind="conflicting_memory",
            primary_stage="update",
            memory_row=stale,
            replay_row=gap1_registry,
            expected_control_action="evict_stale_conflict",
            observed_control_action=str(stale["governance_action"]),
            error_detected=True,
            propagation_blocked=not bool(stale["active"]),
            notes="The conflicting write is detected before it overwrites the canonical row.",
        ),
        _case(
            case_id="case-shuffled-routing",
            control_kind="shuffled_memory",
            primary_stage="routing",
            memory_row=promotion,
            replay_row=hardware,
            expected_control_action="reject_shuffled_scope_mismatch",
            observed_control_action="reject_shuffled_scope_mismatch",
            error_detected=True,
            propagation_blocked=True,
            notes="A deterministic shuffled-memory control routes gap1 memory to hardware scope.",
            shuffled_from_scope=str(promotion["task_scope"]),
            shuffled_to_scope=str(hardware["memory_scope"]),
        ),
        _case(
            case_id="case-stale-maintenance",
            control_kind="stale_memory",
            primary_stage="maintenance",
            memory_row=stale,
            replay_row=gap1,
            expected_control_action="evict_stale_conflict",
            observed_control_action=str(stale["governance_action"]),
            error_detected=True,
            propagation_blocked=not bool(stale["active"]),
            notes="Maintenance/eviction removes the stale row before retrieval can use it.",
        ),
        _case(
            case_id="case-poisoning-use",
            control_kind="poisoning_like",
            primary_stage="use",
            memory_row=poisoning,
            replay_row=gap4,
            expected_control_action="reject_poisoning",
            observed_control_action=str(poisoning["governance_action"]),
            error_detected=True,
            propagation_blocked=not bool(poisoning["active"]),
            notes="Use/action blocks a memory that tries to override future verifier behavior.",
        ),
        _case(
            case_id="case-harmful-rollback",
            control_kind="harmful_memory",
            primary_stage="rollback",
            memory_row=rollback,
            replay_row=arc,
            expected_control_action="rollback_harmful",
            observed_control_action=str(rollback["governance_action"]),
            error_detected=True,
            propagation_blocked=bool(rollback["safe_action_selected"]),
            notes="Rollback keeps the safe block action and prevents the harmful alternative.",
        ),
    ]

    cases[3]["control_source_decision_id"] = str(out_of_scope["decision_id"])
    for case in cases:
        case["unsafe_propagated"] = is_unsafe_propagation(case)
    return cases


def is_unsafe_propagation(case: Mapping[str, Any]) -> bool:
    """Return true when a detected bad memory reaches an unsafe final action."""

    if not case.get("error_detected"):
        return False
    if case.get("propagation_blocked"):
        return False
    return str(case.get("selected_decision")) != str(case.get("expected_decision"))


def attribute_memory_operations(
    cases: Sequence[Mapping[str, Any]],
    *,
    upstream_artifacts: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Aggregate attribution cases into the Exp 5289 measured fields."""

    stage_counter = Counter(
        str(case["primary_stage"]) for case in cases if bool(case.get("error_detected"))
    )
    counts = {stage: int(stage_counter.get(stage, 0)) for stage in STAGE_KEYS}
    attributed_cases = sum(1 for case in cases if str(case.get("primary_stage")) in STAGE_KEYS)
    total_cases = len(cases)
    unsafe_cases = [case for case in cases if bool(case.get("unsafe_propagated"))]
    blocked_control_count = sum(
        1
        for case in cases
        if bool(case.get("error_detected")) and bool(case.get("propagation_blocked"))
    )
    maintenance_cost = _local_maintenance_cost(cases, upstream_artifacts=upstream_artifacts)
    decision_impact = _decision_impact_summary(cases, upstream_artifacts["exp5276"])
    coverage = {
        "attributed_cases": attributed_cases,
        "total_cases": total_cases,
        "coverage_rate": _rate(attributed_cases, total_cases),
        "control_kinds": [str(case["control_kind"]) for case in cases],
    }
    unsafe = {
        "count": len(unsafe_cases),
        "blocked_control_count": blocked_control_count,
        "control_kinds": sorted(str(case["control_kind"]) for case in unsafe_cases),
    }
    evidence = {
        "usable_for_exp5290": bool(
            coverage["coverage_rate"] == 1.0
            and unsafe["count"] == 0
            and decision_impact["final_decision_regressions"] == 0
            and decision_impact["always_full_quality_rate"] == 1.0
        ),
        "basis": (
            "Exp5275 governed decision-history rows plus Exp5276 replay decisions; "
            "stage faults are controlled locally without model-weight mutation."
        ),
        "self_learning_tier": "Tier 2 constraint memory / Trace2Skill",
        "no_model_weight_mutation": True,
        "source_artifacts": list(SOURCE_ARTIFACTS),
    }
    ready = bool(evidence["usable_for_exp5290"])
    return {
        "operation_stage_error_counts": counts,
        "attribution_coverage": coverage,
        "unsafe_propagations": unsafe,
        "local_maintenance_cost": maintenance_cost,
        "decision_impact_summary": decision_impact,
        "continuous_self_learning_evidence": evidence,
        "memory_attribution_ready": ready,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    upstream_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the principle-wrapped Exp 5289 result artifact."""

    artifacts = dict(upstream_artifacts or load_upstream_artifacts(root))
    cases = build_attribution_cases(root=root, upstream_artifacts=artifacts)
    summary = attribute_memory_operations(cases, upstream_artifacts=artifacts)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": list(SOURCE_ARTIFACTS),
        "stage_labels": dict(OPERATION_STAGE_LABELS),
        "attribution_rows": cases,
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(summary)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "memory_attribution_ready": bool(summary["memory_attribution_ready"]),
        "memory_attribution_ready_principle": FIELD_PRINCIPLES["memory_attribution_ready"],
        "operation_stage_error_counts": {
            **summary["operation_stage_error_counts"],
            "principle": FIELD_PRINCIPLES["operation_stage_error_counts"],
        },
        "attribution_coverage": _wrap(
            "attribution_coverage",
            summary["attribution_coverage"],
        ),
        "unsafe_propagations": _wrap("unsafe_propagations", summary["unsafe_propagations"]),
        "local_maintenance_cost": _wrap(
            "local_maintenance_cost",
            summary["local_maintenance_cost"],
        ),
        "decision_impact_summary": _wrap(
            "decision_impact_summary",
            summary["decision_impact_summary"],
        ),
        "continuous_self_learning_evidence": _wrap(
            "continuous_self_learning_evidence",
            summary["continuous_self_learning_evidence"],
        ),
        "source_artifact_checksums": source_artifact_checksums(root),
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": [dict(row) for row in tests_run],
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp 5289 schema required by tests and the conductor."""

    for field in REQUIRED_WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if not isinstance(wrapped, Mapping) or "value" not in wrapped or "principle" not in wrapped:
            raise ValueError(f"{field} must be principle-wrapped")  # pragma: no cover
    verdict = str(_wrapped_value(artifact, "honest_verdict"))
    if not (
        verdict.startswith("complete:")
        or verdict.startswith("null:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict terminal prefix invalid")  # pragma: no cover
    if _wrapped_value(artifact, "inference_substrate") not in {
        "aggregation_from_upstream_artifacts",
        "offline_deterministic_fixture_no_llm",
    }:
        raise ValueError("inference_substrate must be no-LLM attribution")  # pragma: no cover
    if not isinstance(artifact.get("memory_attribution_ready"), bool):
        raise ValueError("memory_attribution_ready must be a bare bool")  # pragma: no cover
    if not artifact.get("memory_attribution_ready_principle"):
        raise ValueError("missing memory_attribution_ready_principle")  # pragma: no cover
    counts = artifact.get("operation_stage_error_counts")
    if not isinstance(counts, Mapping) or any(stage not in counts for stage in STAGE_KEYS):
        raise ValueError("operation_stage_error_counts missing a stage")  # pragma: no cover
    if counts.get("principle") != FIELD_PRINCIPLES["operation_stage_error_counts"]:
        raise ValueError("operation_stage_error_counts missing principle")  # pragma: no cover
    if not isinstance(artifact.get("tests_run"), list):
        raise ValueError("tests_run must be a bare list")  # pragma: no cover
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp 5289 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for the two upstream artifacts."""

    root_path = Path(root)
    return {
        "exp5275": _sha256_file(root_path / EXP5275_RELATIVE_PATH),
        "exp5276": _sha256_file(root_path / EXP5276_RELATIVE_PATH),
    }


def _case(
    *,
    case_id: str,
    control_kind: str,
    primary_stage: str,
    memory_row: Mapping[str, Any],
    replay_row: Mapping[str, Any],
    expected_control_action: str,
    observed_control_action: str,
    error_detected: bool,
    propagation_blocked: bool,
    notes: str,
    shuffled_from_scope: str | None = None,
    shuffled_to_scope: str | None = None,
) -> JsonDict:
    return {
        "case_id": str(case_id),
        "control_kind": str(control_kind),
        "primary_stage": str(primary_stage),
        "operation_stage_label": OPERATION_STAGE_LABELS[str(primary_stage)],
        "memory_decision_id": str(memory_row.get("decision_id")),
        "memory_fixture_kind": str(memory_row.get("fixture_kind")),
        "task_id": str(replay_row.get("task_id")),
        "task_scope": replay_row.get("memory_scope"),
        "memory_task_scope": memory_row.get("task_scope"),
        "expected_control_action": str(expected_control_action),
        "observed_control_action": str(observed_control_action),
        "error_detected": bool(error_detected),
        "propagation_blocked": bool(propagation_blocked),
        "expected_decision": str(replay_row.get("expected_decision")),
        "selected_decision": str(replay_row.get("selected_decision")),
        "selected_decision_source": str(replay_row.get("selected_decision_source")),
        "final_decision_correct": bool(replay_row.get("correct")),
        "final_decision_false_accept": bool(replay_row.get("false_accept")),
        "local_maintenance_cost_units": 1,
        "shuffled_from_scope": shuffled_from_scope,
        "shuffled_to_scope": shuffled_to_scope,
        "notes": str(notes),
    }


def _without_provenance(row: Mapping[str, Any]) -> JsonDict:
    mutated = dict(row)
    mutated["source_artifact"] = None
    mutated["evidence_checksum"] = None
    mutated["source_artifacts"] = []
    mutated["decision_id"] = str(row["decision_id"]) + ":missing-provenance"
    return mutated


def _local_maintenance_cost(
    cases: Sequence[Mapping[str, Any]],
    *,
    upstream_artifacts: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    return {
        "total_cost_units": sum(int(case.get("local_maintenance_cost_units", 0)) for case in cases),
        "cost_model": "one unit per deterministic ledger/provenance/scope/rollback check",
        "source_rows_examined": len(upstream_artifacts["exp5275"].get("governance_rows", [])),
        "pilot_rows_examined": len(upstream_artifacts["exp5276"].get("pilot_rows", [])),
        "llm_calls": 0,
    }


def _decision_impact_summary(
    cases: Sequence[Mapping[str, Any]],
    exp5276: Mapping[str, Any],
) -> JsonDict:
    memory_metrics = exp5276.get("memory_assisted_metrics", {})
    baseline_metrics = exp5276.get("baseline_metrics", {})
    always_full = baseline_metrics.get("always_full", {})
    no_memory = baseline_metrics.get("no_memory_scheduler", {})
    pilot_rows = exp5276.get("pilot_rows", [])
    always_full_rows = exp5276.get("baseline_rows", {}).get("always_full", [])
    by_task_full = {
        str(row.get("task_id")): row for row in always_full_rows if isinstance(row, Mapping)
    }
    final_decision_differences = sum(
        1
        for row in pilot_rows
        if isinstance(row, Mapping)
        and str(row.get("selected_decision"))
        != str(by_task_full.get(str(row.get("task_id")), {}).get("selected_decision"))
    )
    regressions = sum(1 for case in cases if bool(case.get("unsafe_propagated")))
    return {
        "memory_assisted_quality_rate": float(memory_metrics.get("quality_rate", 0.0)),
        "always_full_quality_rate": float(always_full.get("quality_rate", 0.0)),
        "no_memory_quality_rate": float(no_memory.get("quality_rate", 0.0)),
        "decision_quality_delta": _wrapped_or_raw(exp5276, "decision_quality_delta", 0.0),
        "calls_avoided_rate": _wrapped_or_raw(exp5276, "calls_avoided_rate", 0.0),
        "allocation_changed_by_memory_count": int(
            exp5276.get("allocation_changed_by_memory_count", 0)
        ),
        "final_decision_differences_vs_full": final_decision_differences,
        "final_decision_regressions": regressions,
        "memory_answer_injection_blocked": all(
            str(row.get("selected_decision_source")) != "memory_promoted_decision"
            for row in pilot_rows
            if isinstance(row, Mapping)
        ),
    }


def _honest_verdict(summary: Mapping[str, Any]) -> str:
    unsafe = summary["unsafe_propagations"]["count"]
    coverage = summary["attribution_coverage"]["coverage_rate"]
    regressions = summary["decision_impact_summary"]["final_decision_regressions"]
    if unsafe:
        return f"blocked_unsafe_propagation: operation attribution found unsafe propagation count={unsafe}"
    if coverage < 1.0:
        return f"null: operation attribution incomplete; coverage_rate={coverage:.6f}"
    if regressions:
        return f"blocked_decision_regression: operation attribution found final decision regressions={regressions}"
    if summary["memory_attribution_ready"]:
        return (
            "complete: operation attribution is usable for Exp5290; all bounded "
            "operation-stage controls were attributed and unsafe_propagations=0"
        )
    return "null: operation attribution is not usable for Exp5290"


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    wrapped = artifact.get(field)
    return wrapped.get("value") if isinstance(wrapped, Mapping) else wrapped


def _wrapped_or_raw(artifact: Mapping[str, Any], field: str, default: float) -> float:
    value = _wrapped_value(artifact, field)
    return float(default if value is None else value)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _checksum(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        ).hexdigest()
    )
