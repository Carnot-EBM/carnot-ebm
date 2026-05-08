"""Build the Exp 1546 milestone .118 retrospective artifact.

Spec: REQ-REPORT-059, SCENARIO-REPORT-059.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260508"
MILESTONE = "2026.04.118"
NEXT_MILESTONE = "2026.04.119"
EXPERIMENT = "1546_milestone_118_retro"
SCHEMA = "milestone_118_retro_v1"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_1546_milestone_118_retro.json"
ROADMAP_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"

MET = "MET"
NOT_MET = "NOT_MET"
HONESTLY_TERMINAL = "HONESTLY_TERMINAL"
SATISFIES_CRITERION = {MET, HONESTLY_TERMINAL}
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
MANDATED_SOTA_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "criteria_met",
    "criteria_total",
    "completed_tasks",
    "honestly_terminal_tasks",
    "failed_or_blocked_tasks",
    "automata_contract_gate",
    "satquest_verifier_gate",
    "residual_drift_gate",
    "fr11_positive_utility_gate",
    "product_line_carry_forward_gate",
    "claim_isolation_router_gate",
    "arm_ebm_diagnostic_boundary",
    "thrml_next_scaling_gate",
    "extropic_access_readiness_gate",
    "recommended_119_focus",
    "ops_reconciliation_needed",
    "active_roadmap_modified",
    "conductor_modified",
    "honest_verdict",
}

SOURCE_FILES = {
    "exp1533": "experiment_1533_117_completion_archive_118_activation.json",
    "exp1534": "experiment_1534_planner_orphan_test_guard.json",
    "exp1535": "experiment_1535_xgrammar_abs_contract_decoder_adapter.json",
    "exp1536": "experiment_1536_satquest_cnf_verifier_benchmark.json",
    "exp1537": "experiment_1537_beaver_prefix_bound_contracts_v3.json",
    "exp1538": "experiment_1538_residual_drift_commitment_ledger.json",
    "exp1539": "experiment_1539_fr11_external_feedback_skill_promotion_v13.json",
    "exp1540": "experiment_1540_product_line_staged_benchmark_scale_v3.json",
    "exp1541": "experiment_1541_claim_isolation_uncertainty_router_v2.json",
    "exp1542": "experiment_1542_arm_ebm_soft_value_diagnostic.json",
    "exp1543": "experiment_1543_thrml_carnot_parity_n256_schedule_stress.json",
    "exp1544": "experiment_1544_thrml_diverse_topology_parity_n64.json",
    "exp1545": "experiment_1545_extropic_z1_access_readiness_packet.json",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-059: persist a started marker before mutable evidence reads.

    This makes an interrupted conductor run auditable. The marker intentionally
    carries the required top-level fields so downstream tooling can distinguish
    a started retrospective from a missing deliverable without treating it as a
    terminal success.
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
            "criteria_met": 0,
            "criteria_total": 0,
            "completed_tasks": [],
            "honestly_terminal_tasks": [],
            "failed_or_blocked_tasks": [],
            "recommended_119_focus": [],
            "ops_reconciliation_needed": {"needed": True, "deferred_to_conductor": True},
            "active_roadmap_modified": False,
            "conductor_modified": False,
            "honest_verdict": "complete: in_progress_milestone_118_retro_seeded",
        }
    )
    return _write_json(Path(out_path), artifact)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _load_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    loaded: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for exp_id, filename in SOURCE_FILES.items():
        payload = _read_json(results_dir / filename)
        if payload is None:
            missing.append(exp_id)
        else:
            loaded[exp_id] = payload
    return loaded, missing


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or "").lower()


def _verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict") or "")


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return _status(payload) == "complete"


def _is_honestly_terminal(payload: Mapping[str, Any]) -> bool:
    return _status(payload) in {"blocked", "gate_blocked", "skipped"} or _verdict(
        payload
    ).lower().startswith(TERMINAL_PREFIXES)


def _zero(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0


def _positive_int(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _number(payload: Mapping[str, Any], field: str, default: float = 0.0) -> float:
    value = payload.get(field)
    return (
        float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
    )


def _has_mandated_sota(payload: Mapping[str, Any]) -> bool:
    model_names = {str(model) for model in payload.get("models_used", []) if model}
    for item in payload.get("model_specs", []):
        if isinstance(item, Mapping) and item.get("hf_id"):
            model_names.add(str(item["hf_id"]))
        elif item:
            model_names.add(str(item))
    return bool(model_names.intersection(MANDATED_SOTA_MODELS))


def _source_path(exp_id: str, field: str | None = None) -> str:
    path = f"results/{SOURCE_FILES[exp_id]}"
    return f"{path}:{field}" if field else path


def _blocker_reason(payload: Mapping[str, Any]) -> str:
    blockers = payload.get("blockers")
    if isinstance(blockers, list) and blockers:
        return ", ".join(str(blocker) for blocker in blockers)
    return str(blockers or _verdict(payload) or _status(payload) or "criterion not satisfied")


def _criterion(
    *,
    key: str,
    exp_id: str,
    status: str,
    target: str,
    fields: tuple[str, ...],
    payload: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    source_values = {field: payload.get(field) for field in fields}
    source_values.update({"status": payload.get("status"), "honest_verdict": _verdict(payload)})
    return {
        "criterion": key,
        "experiment_id": exp_id,
        "status": status,
        "target": target,
        "evidence_paths": [_source_path(exp_id, field) for field in fields],
        "source_values": source_values,
        "reason": reason,
    }


def _missing_criterion(key: str, exp_id: str, target: str) -> dict[str, Any]:
    return {
        "criterion": key,
        "experiment_id": exp_id,
        "status": NOT_MET,
        "target": target,
        "evidence_paths": [_source_path(exp_id)],
        "source_values": {"status": "missing", "honest_verdict": "missing_artifact"},
        "reason": f"{exp_id} source artifact is missing.",
    }


def _score_source_criterion(
    *,
    key: str,
    exp_id: str,
    target: str,
    fields: tuple[str, ...],
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
    passed: Callable[[Mapping[str, Any]], bool],
    honest_terminal: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    if exp_id in missing_source_ids or exp_id not in sources:
        return _missing_criterion(key, exp_id, target)
    payload = sources[exp_id]
    status = (
        MET
        if passed(payload)
        else HONESTLY_TERMINAL
        if honest_terminal and honest_terminal(payload)
        else NOT_MET
    )
    reason = (
        "criterion satisfied"
        if status == MET
        else "honestly terminal within the criterion boundary"
        if status == HONESTLY_TERMINAL
        else _blocker_reason(payload)
    )
    return _criterion(
        key=key,
        exp_id=exp_id,
        status=status,
        target=target,
        fields=fields,
        payload=payload,
        reason=reason,
    )


def _topologies_all_passed(payload: Mapping[str, Any]) -> bool:
    tested = {str(item) for item in payload.get("topologies_tested", [])}
    passed = {str(item) for item in payload.get("topologies_passed", [])}
    return bool(tested) and tested.issubset(passed)


def _honest_thrml_blocker(payload: Mapping[str, Any]) -> bool:
    return _status(payload) in {"blocked", "gate_blocked", "skipped"} and (
        payload.get("simulator_only") is True and payload.get("no_tsu_hardware_claim") is True
    )


def _criteria_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "key": "activation",
            "exp_id": "exp1533",
            "target": "activation_manifest_complete=true with .117 criteria and explicit .118 gates",
            "fields": (
                "activation_manifest_complete",
                "predecessor_criteria_met",
                "predecessor_criteria_total",
                "research_roadmap_yaml_modified",
                "scripts_research_conductor_modified",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and p.get("activation_manifest_complete") is True
                and p.get("predecessor_criteria_met") == 14
                and p.get("predecessor_criteria_total") == 14
                and p.get("research_roadmap_yaml_modified") is not True
                and p.get("scripts_research_conductor_modified") is not True
            ),
        },
        {
            "key": "planner_guard",
            "exp_id": "exp1534",
            "target": "orphan_test_guard_ready=true and generated import targets audited",
            "fields": (
                "orphan_test_guard_ready",
                "import_targets_checked",
                "orphan_imports_detected",
                "active_roadmap_modified",
                "conductor_modified",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and p.get("orphan_test_guard_ready") is True
                and _positive_int(p, "import_targets_checked")
                and _zero(p, "orphan_imports_detected")
                and p.get("active_roadmap_modified") is not True
                and p.get("conductor_modified") is not True
            ),
        },
        {
            "key": "automata_decoder",
            "exp_id": "exp1535",
            "target": "contract decoder adapter ready with local SOTA, deltas, and zero false accepts",
            "fields": (
                "contract_decoder_adapter_ready",
                "model_specs",
                "models_used",
                "baseline_parse_rate",
                "automata_parse_rate",
                "baseline_contract_accept_rate",
                "automata_contract_accept_rate",
                "latency_delta_seconds",
                "false_accept_count",
                "false_accept_rate",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and p.get("contract_decoder_adapter_ready") is True
                and _has_mandated_sota(p)
                and p.get("latency_delta_seconds") is not None
                and _number(p, "automata_parse_rate") > _number(p, "baseline_parse_rate")
                and _number(p, "automata_contract_accept_rate")
                >= _number(p, "baseline_contract_accept_rate")
                and _zero(p, "false_accept_count")
                and _zero(p, "false_accept_rate")
            ),
        },
        {
            "key": "satquest_benchmark",
            "exp_id": "exp1536",
            "target": "SATQuest benchmark ready with solver oracle authority and zero false accepts",
            "fields": (
                "satquest_benchmark_ready",
                "solver_oracle_used",
                "solver_oracle_false_accepts",
                "false_accept_rate",
                "cnf_instances",
                "formats_tested",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and p.get("satquest_benchmark_ready") is True
                and bool(p.get("solver_oracle_used"))
                and _zero(p, "solver_oracle_false_accepts")
                and _zero(p, "false_accept_rate")
                and _positive_int(p, "cnf_instances")
            ),
        },
        {
            "key": "prefix_risk_bounds",
            "exp_id": "exp1537",
            "target": "BEAVER prefix-risk metrics ready and auxiliary to deterministic validators",
            "fields": (
                "beaver_bound_ready",
                "bounded_prefixes",
                "bound_violations",
                "deterministic_validator_final_authority",
                "false_accept_count",
                "false_accept_rate",
                "blockers",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and p.get("beaver_bound_ready") is True
                and _positive_int(p, "bounded_prefixes")
                and isinstance(p.get("bound_violations"), list)
                and p.get("deterministic_validator_final_authority") is True
                and _zero(p, "false_accept_count")
                and _zero(p, "false_accept_rate")
            ),
        },
        {
            "key": "residual_drift",
            "exp_id": "exp1538",
            "target": "residual drift ledger separates contradiction and satisfiable drift counts",
            "fields": (
                "residual_drift_ledger_ready",
                "multi_turn_cases",
                "contradiction_cases",
                "satisfiable_drift_cases",
                "drift_rate",
                "false_accept_count",
                "false_accept_rate",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and p.get("residual_drift_ledger_ready") is True
                and _positive_int(p, "multi_turn_cases")
                and p.get("contradiction_cases") is not None
                and p.get("satisfiable_drift_cases") is not None
                and _zero(p, "false_accept_count")
                and _zero(p, "false_accept_rate")
            ),
        },
        {
            "key": "continuous_self_learning",
            "exp_id": "exp1539",
            "target": "FR-11 is query-time safe; positive utility only if utility_delta > 0",
            "fields": (
                "fr11_external_feedback_ready",
                "continuous_self_learning_task",
                "no_model_weight_mutation",
                "soundness_mistakes",
                "utility_delta",
                "positive_utility_promotion_ready",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and p.get("continuous_self_learning_task") is True
                and p.get("no_model_weight_mutation") is True
                and _zero(p, "soundness_mistakes")
                and (
                    p.get("positive_utility_promotion_ready") is not True
                    or _number(p, "utility_delta") > 0.0
                )
            ),
        },
        {
            "key": "product_line_scale",
            "exp_id": "exp1540",
            "target": "product-line scale is ready or retired with parser/oracle/false-accept metrics",
            "fields": (
                "product_line_scale_ready",
                "branch_retired",
                "cases_total",
                "automata_constraints_used",
                "syntax_stage_pass_rate",
                "feasibility_stage_pass_rate",
                "oracle_agreement_rate",
                "false_accept_count",
                "false_accept_rate",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and (p.get("product_line_scale_ready") is True or p.get("branch_retired") is True)
                and _positive_int(p, "cases_total")
                and p.get("syntax_stage_pass_rate") is not None
                and p.get("feasibility_stage_pass_rate") is not None
                and p.get("oracle_agreement_rate") is not None
                and _zero(p, "false_accept_count")
                and _zero(p, "false_accept_rate")
            ),
        },
        {
            "key": "claim_isolation_routing",
            "exp_id": "exp1541",
            "target": "uncertainty router reports routed budget metrics and zero false accepts",
            "fields": (
                "uncertainty_router_ready",
                "cases_loaded",
                "claims_extracted",
                "routed_cases",
                "budget_delta",
                "budget_improvement_claimed",
                "false_accept_count",
                "false_accept_rate",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and p.get("uncertainty_router_ready") is True
                and _positive_int(p, "cases_loaded")
                and _positive_int(p, "claims_extracted")
                and p.get("routed_cases") is not None
                and p.get("budget_delta") is not None
                and _zero(p, "false_accept_count")
                and _zero(p, "false_accept_rate")
            ),
        },
        {
            "key": "arm_ebm_diagnostic",
            "exp_id": "exp1542",
            "target": "ARM/EBT diagnostic ready with deterministic validators as final authority",
            "fields": (
                "arm_ebm_diagnostic_ready",
                "deterministic_validators_final_authority",
                "no_model_weight_mutation",
                "diagnostic_cases",
                "routing_auc",
                "energy_label_correlation",
                "soft_value_label_correlation",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and p.get("arm_ebm_diagnostic_ready") is True
                and p.get("deterministic_validators_final_authority") is True
                and p.get("no_model_weight_mutation") is True
                and _positive_int(p, "diagnostic_cases")
            ),
        },
        {
            "key": "thrml_n256",
            "exp_id": "exp1543",
            "target": "THRML n=256 schedule parity passes or honestly blocks simulator-only",
            "fields": (
                "thrml_parity_n256_schedule_ready",
                "parity_passed",
                "n_spins",
                "schedules_tested",
                "kl_divergence",
                "simulator_only",
                "no_tsu_hardware_claim",
                "blockers",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and p.get("thrml_parity_n256_schedule_ready") is True
                and p.get("parity_passed") is True
                and p.get("n_spins") == 256
                and p.get("simulator_only") is True
                and p.get("no_tsu_hardware_claim") is True
            ),
            "honest_terminal": _honest_thrml_blocker,
        },
        {
            "key": "thrml_diverse_n64",
            "exp_id": "exp1544",
            "target": "THRML diverse topology n=64 passes or honestly blocks simulator-only",
            "fields": (
                "diverse_topology_parity_n64_ready",
                "parity_passed",
                "n_spins",
                "topologies_tested",
                "topologies_passed",
                "kl_divergence",
                "simulator_only",
                "no_tsu_hardware_claim",
                "blockers",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and p.get("diverse_topology_parity_n64_ready") is True
                and p.get("parity_passed") is True
                and p.get("n_spins") == 64
                and _topologies_all_passed(p)
                and p.get("simulator_only") is True
                and p.get("no_tsu_hardware_claim") is True
            ),
            "honest_terminal": _honest_thrml_blocker,
        },
        {
            "key": "hardware_readiness",
            "exp_id": "exp1545",
            "target": "Extropic readiness packet exists with explicit no-hardware boundary",
            "fields": (
                "extropic_z1_readiness_packet_ready",
                "no_hardware_execution_claim",
                "benchmark_cases_included",
                "access_blockers",
                "required_device_evidence_fields",
                "research_roadmap_yaml_modified",
                "scripts_research_conductor_modified",
            ),
            "passed": lambda p: (
                _is_complete(p)
                and p.get("extropic_z1_readiness_packet_ready") is True
                and p.get("no_hardware_execution_claim") is True
                and _positive_int(p, "benchmark_cases_included")
                and p.get("research_roadmap_yaml_modified") is not True
                and p.get("scripts_research_conductor_modified") is not True
            ),
        },
    )


def _source_success_criteria(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
) -> dict[str, dict[str, Any]]:
    return {
        spec["key"]: _score_source_criterion(
            key=spec["key"],
            exp_id=spec["exp_id"],
            target=spec["target"],
            fields=spec["fields"],
            sources=sources,
            missing_source_ids=missing_source_ids,
            passed=spec["passed"],
            honest_terminal=spec.get("honest_terminal"),
        )
        for spec in _criteria_specs()
    }


def _retrospective_criterion(
    *,
    missing_source_ids: set[str],
    active_roadmap_modified: bool,
    conductor_modified: bool,
) -> dict[str, Any]:
    status = MET if not active_roadmap_modified and not conductor_modified else NOT_MET
    return {
        "criterion": "retrospective",
        "experiment_id": "exp1546",
        "status": status,
        "target": "criteria_met and criteria_total summarize .118 with .119 carry-forward gates",
        "evidence_paths": [
            "results/experiment_1546_milestone_118_retro.json",
            "research-roadmap.yaml",
            "scripts/research_conductor.py",
        ],
        "source_values": {
            "missing_source_ids": sorted(missing_source_ids),
            "active_roadmap_modified": active_roadmap_modified,
            "conductor_modified": conductor_modified,
        },
        "reason": "retrospective artifact can close without protected-file edits"
        if status == MET
        else "protected roadmap or conductor file changed",
    }


def _task_record(result: Mapping[str, Any]) -> dict[str, str]:
    return {
        "experiment_id": str(result["experiment_id"]),
        "criterion": str(result["criterion"]),
        "status": str(result["status"]),
        "reason": str(result["reason"]),
    }


def _automata_contract_gate(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1535", {})
    parse_delta = _number(payload, "automata_parse_rate") - _number(payload, "baseline_parse_rate")
    accept_delta = _number(payload, "automata_contract_accept_rate") - _number(
        payload, "baseline_contract_accept_rate"
    )
    return {
        "status": "advance",
        "source_artifact": _source_path("exp1535"),
        "adapter_ready": payload.get("contract_decoder_adapter_ready") is True,
        "automata_constraints_improved_contract_generation": parse_delta > 0 and accept_delta >= 0,
        "baseline_parse_rate": payload.get("baseline_parse_rate"),
        "automata_parse_rate": payload.get("automata_parse_rate"),
        "baseline_contract_accept_rate": payload.get("baseline_contract_accept_rate"),
        "automata_contract_accept_rate": payload.get("automata_contract_accept_rate"),
        "latency_delta_seconds": payload.get("latency_delta_seconds"),
        "false_accept_rate": payload.get("false_accept_rate"),
        "carry_forward_to_119": (
            "Use automata masks as contract-generation support while keeping runtime "
            "contracts and deterministic validators as authority."
        ),
    }


def _satquest_verifier_gate(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1536", {})
    zero_false_accepts = _zero(payload, "solver_oracle_false_accepts") and _zero(
        payload, "false_accept_rate"
    )
    return {
        "status": "advance" if zero_false_accepts else "carry_forward_blocked",
        "source_artifact": _source_path("exp1536"),
        "benchmark_ready": payload.get("satquest_benchmark_ready") is True,
        "solver_oracle_used": payload.get("solver_oracle_used"),
        "solver_oracle_false_accepts": payload.get("solver_oracle_false_accepts"),
        "false_accept_rate": payload.get("false_accept_rate"),
        "zero_solver_oracle_false_accepts": zero_false_accepts,
        "carry_forward_to_119": (
            "repair_oracle_false_accepts_before_acceptance_use"
            if not zero_false_accepts
            else "keep_as_solver_oracle_regression_gate"
        ),
    }


def _residual_drift_gate(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1538", {})
    return {
        "status": "advance_bounded_replay",
        "source_artifact": _source_path("exp1538"),
        "ledger_ready": payload.get("residual_drift_ledger_ready") is True,
        "multi_turn_cases": payload.get("multi_turn_cases"),
        "contradiction_cases": payload.get("contradiction_cases"),
        "satisfiable_drift_cases": payload.get("satisfiable_drift_cases"),
        "repaired_drift_cases": payload.get("repaired_drift_cases"),
        "drift_rate": payload.get("drift_rate"),
        "false_accept_rate": payload.get("false_accept_rate"),
        "carry_forward_to_119": "use ledger to separate repairable drift from contradictions",
    }


def _fr11_positive_utility_gate(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1539", {})
    positive = (
        payload.get("positive_utility_promotion_ready") is True
        and _number(payload, "utility_delta") > 0.0
    )
    return {
        "status": "positive_utility_ready" if positive else "safety_only",
        "source_artifact": _source_path("exp1539"),
        "continuous_self_learning_task": payload.get("continuous_self_learning_task") is True,
        "no_model_weight_mutation": payload.get("no_model_weight_mutation") is True,
        "soundness_mistakes": payload.get("soundness_mistakes"),
        "utility_delta": payload.get("utility_delta"),
        "positive_utility_achieved": positive,
        "positive_utility_promotion_ready": payload.get("positive_utility_promotion_ready"),
        "carry_forward_to_119": (
            "headline positive utility is allowed"
            if positive
            else "repeat only with measurable external utility or retire positive-utility claim"
        ),
    }


def _product_line_carry_forward_gate(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1540", {})
    decision = "retire" if payload.get("branch_retired") is True else "continue"
    return {
        "decision": decision,
        "source_artifact": _source_path("exp1540"),
        "product_line_scale_ready": payload.get("product_line_scale_ready") is True,
        "branch_retired": payload.get("branch_retired") is True,
        "cases_total": payload.get("cases_total"),
        "automata_constraints_used": payload.get("automata_constraints_used") is True,
        "syntax_stage_pass_rate": payload.get("syntax_stage_pass_rate"),
        "feasibility_stage_pass_rate": payload.get("feasibility_stage_pass_rate"),
        "oracle_agreement_rate": payload.get("oracle_agreement_rate"),
        "false_accept_rate": payload.get("false_accept_rate"),
        "carry_forward_to_119": (
            "continue staged scaling under parser, feasibility, oracle, and false-accept gates"
            if decision == "continue"
            else "retire until a materially different product-line benchmark exists"
        ),
    }


def _claim_isolation_router_gate(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1541", {})
    budget_delta = payload.get("budget_delta")
    budget_reduced = isinstance(budget_delta, (int, float)) and budget_delta < 0
    return {
        "status": "advance_with_budget_validation",
        "source_artifact": _source_path("exp1541"),
        "uncertainty_router_ready": payload.get("uncertainty_router_ready") is True,
        "routed_cases": payload.get("routed_cases"),
        "budget_delta": budget_delta,
        "budget_reduced": budget_reduced,
        "budget_improvement_claimed": payload.get("budget_improvement_claimed") is True,
        "verifier_calls_claim_isolated": payload.get("verifier_calls_claim_isolated"),
        "verifier_calls_full_context": payload.get("verifier_calls_full_context"),
        "false_accept_rate": payload.get("false_accept_rate"),
        "carry_forward_to_119": "scale routed corpus and keep deterministic validators final",
    }


def _arm_ebm_diagnostic_boundary(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1542", {})
    return {
        "status": "diagnostic_only",
        "source_artifact": _source_path("exp1542"),
        "arm_ebm_diagnostic_ready": payload.get("arm_ebm_diagnostic_ready") is True,
        "acceptance_authority": "deterministic_validators_only",
        "deterministic_validators_final_authority": (
            payload.get("deterministic_validators_final_authority") is True
        ),
        "diagnostic_cases": payload.get("diagnostic_cases"),
        "routing_auc": payload.get("routing_auc"),
        "energy_label_correlation": payload.get("energy_label_correlation"),
        "soft_value_label_correlation": payload.get("soft_value_label_correlation"),
        "logprob_available": payload.get("logprob_available"),
        "carry_forward_to_119": "do not promote ARM/EBT soft values to acceptance authority",
    }


def _thrml_next_scaling_gate(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    n256 = sources.get("exp1543", {})
    diverse = sources.get("exp1544", {})
    n256_ready = n256.get("thrml_parity_n256_schedule_ready") is True
    diverse_ready = diverse.get("diverse_topology_parity_n64_ready") is True
    return {
        "status": "advance_software_only" if n256_ready and diverse_ready else "repeat_failed_gate",
        "can_scale_further_in_software": n256_ready and diverse_ready,
        "hardware_execution_claimed": False,
        "n256": {
            "source_artifact": _source_path("exp1543"),
            "ready": n256_ready,
            "kl_divergence": n256.get("kl_divergence"),
            "max_energy_delta": n256.get("max_energy_delta"),
            "simulator_only": n256.get("simulator_only"),
            "no_tsu_hardware_claim": n256.get("no_tsu_hardware_claim"),
        },
        "diverse_n64": {
            "source_artifact": _source_path("exp1544"),
            "ready": diverse_ready,
            "kl_divergence": diverse.get("kl_divergence"),
            "topologies_tested": diverse.get("topologies_tested", []),
            "topologies_passed": diverse.get("topologies_passed", []),
            "simulator_only": diverse.get("simulator_only"),
            "no_tsu_hardware_claim": diverse.get("no_tsu_hardware_claim"),
        },
        "carry_forward_to_119": (
            "scale simulator parity only after independent-RNG validation; require authenticated "
            "hardware transcript before any Z1/TSU claim"
        ),
    }


def _extropic_access_readiness_gate(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1545", {})
    blockers = payload.get("access_blockers", [])
    return {
        "status": "packet_ready_no_access"
        if payload.get("extropic_z1_readiness_packet_ready") is True
        else "packet_missing_or_blocked",
        "source_artifact": _source_path("exp1545"),
        "readiness_packet_ready": payload.get("extropic_z1_readiness_packet_ready") is True,
        "hardware_execution_claimed": payload.get("no_hardware_execution_claim") is not True,
        "no_hardware_execution_claim": payload.get("no_hardware_execution_claim") is True,
        "benchmark_cases_included": payload.get("benchmark_cases_included"),
        "access_blockers": blockers,
        "required_device_evidence_fields": payload.get("required_device_evidence_fields", []),
        "carry_forward_to_119": "request authenticated access and collect transcript schema fields",
    }


def _honestly_terminal_tasks(
    *,
    criteria: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    records: dict[str, dict[str, str]] = {
        str(result["experiment_id"]): _task_record(result)
        for result in criteria.values()
        if result["status"] == HONESTLY_TERMINAL
    }
    satquest = sources.get("exp1536", {})
    fr11 = sources.get("exp1539", {})
    beaver = sources.get("exp1537", {})
    extropic = sources.get("exp1545", {})
    extras = (
        (
            "exp1536",
            "satquest_benchmark",
            "completed_with_solver_false_accepts",
            _is_honestly_terminal(satquest) and not _zero(satquest, "solver_oracle_false_accepts"),
        ),
        (
            "exp1539",
            "continuous_self_learning",
            "safe_query_time_promotion_without_positive_utility",
            _is_honestly_terminal(fr11)
            and not (
                fr11.get("positive_utility_promotion_ready") is True
                and _number(fr11, "utility_delta") > 0.0
            ),
        ),
        (
            "exp1537",
            "prefix_risk_bounds",
            "prefix_metrics_auxiliary_with_structural_logprob_blockers",
            _is_honestly_terminal(beaver) and bool(beaver.get("blockers")),
        ),
        (
            "exp1545",
            "hardware_readiness",
            "readiness_packet_only_without_authenticated_hardware_access",
            _is_honestly_terminal(extropic) and bool(extropic.get("access_blockers")),
        ),
    )
    for exp_id, criterion, reason, include in extras:
        if include:
            records[exp_id] = {
                "experiment_id": exp_id,
                "criterion": criterion,
                "status": HONESTLY_TERMINAL,
                "reason": reason,
            }
    return [records[exp_id] for exp_id in sorted(records)]


def _source_inputs_read(
    *,
    missing_source_ids: set[str],
    roadmap_doc_text: str,
    research_roadmap_yaml_text: str,
    research_roadmap_next_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    ops_known_issues_text: str,
    conductor_log_text: str,
) -> dict[str, dict[str, bool]]:
    inputs = {
        f"results/{filename}": {"exists": exp_id not in missing_source_ids}
        for exp_id, filename in SOURCE_FILES.items()
    }
    inputs.update(
        {
            ROADMAP_DOC: {"exists": bool(roadmap_doc_text)},
            "research-roadmap.yaml": {"exists": bool(research_roadmap_yaml_text)},
            "research-roadmap-next.yaml": {"exists": bool(research_roadmap_next_text)},
            "research-complete.yaml": {"exists": bool(research_complete_text)},
            "ops/conductor-log.md": {"exists": bool(conductor_log_text)},
            "ops/status.md": {"exists": bool(ops_status_text)},
            "ops/changelog.md": {"exists": bool(ops_changelog_text)},
            "ops/known-issues.md": {"exists": bool(ops_known_issues_text)},
        }
    )
    return inputs


def _ops_reconciliation_needed() -> dict[str, Any]:
    return {
        "needed": True,
        "deferred_to_conductor_reconciler": True,
        "reason": (
            "stop-when-done retro run writes the machine-readable artifact only; "
            "the conductor's reconciliation step owns ops and traceability docs"
        ),
        "files": {
            "research-complete.yaml": "append a 2026.04.118 archive entry for exp1533-exp1546",
            "ops/status.md": "summarize .118 score, SATQuest false accepts, FR-11 utility=0, and THRML/Extropic boundaries",
            "ops/changelog.md": "record the .118 retrospective artifact and .119 gates",
            "ops/known-issues.md": "track SATQuest false accepts, FR-11 positive-utility gap, and Extropic access blockers",
            "_bmad/traceability.md": "link REQ-REPORT-059 to module, tests, and deliverable",
        },
    }


def _recommended_119_focus() -> list[str]:
    return [
        "Repair SATQuest solver-oracle false accepts before using CNF verification for acceptance.",
        "Use automata/ABS masks for contract generation, then cross-check with runtime contracts and solver oracles.",
        "Repeat FR-11 external feedback only with a measurable positive-utility gate or retire the headline claim.",
        "Continue product-line staged scaling under parser, feasibility, oracle, and false-accept gates.",
        "Scale claim-isolation routing while validating that routed budget savings do not hide deterministic failures.",
        "Keep ARM/EBT soft-value signals diagnostic-only until deterministic acceptance proof exists.",
        "Advance THRML simulator parity only with independent-RNG validation and no hardware claim.",
        "Use the Extropic readiness packet to seek authenticated Z1/XTR-0 access and collect transcript evidence.",
    ]


def build_artifact(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: list[str],
    roadmap_doc_text: str,
    research_roadmap_yaml_text: str,
    research_roadmap_next_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    ops_known_issues_text: str,
    conductor_log_text: str,
    protected_files_unchanged: bool,
) -> dict[str, Any]:
    """REQ-REPORT-059: score `.118` and emit carry-forward gates for `.119`.

    The artifact is intentionally conservative: source tasks may be complete
    while still failing a stricter criterion. Those cases remain visible in the
    task lists so the roadmap can carry forward the real blocker instead of a
    fabricated success.
    """

    missing = set(missing_source_ids)
    source_active_modified = any(
        payload.get("active_roadmap_modified") is True
        or payload.get("research_roadmap_yaml_modified") is True
        for payload in sources.values()
    )
    source_conductor_modified = any(
        payload.get("conductor_modified") is True
        or payload.get("scripts_research_conductor_modified") is True
        for payload in sources.values()
    )
    active_roadmap_modified = source_active_modified or not protected_files_unchanged
    conductor_modified = source_conductor_modified or not protected_files_unchanged
    criteria = _source_success_criteria(sources=sources, missing_source_ids=missing)
    criteria["retrospective"] = _retrospective_criterion(
        missing_source_ids=missing,
        active_roadmap_modified=active_roadmap_modified,
        conductor_modified=conductor_modified,
    )
    criteria_met = sum(1 for result in criteria.values() if result["status"] in SATISFIES_CRITERION)
    criteria_total = len(criteria)
    completed_tasks = [
        _task_record(result) for result in criteria.values() if result["status"] == MET
    ]
    failed_or_blocked_tasks = [
        _task_record(result) for result in criteria.values() if result["status"] == NOT_MET
    ]
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete",
        "milestone": MILESTONE,
        "next_milestone": NEXT_MILESTONE,
        "criteria_results": criteria,
        "criteria_met": criteria_met,
        "criteria_total": criteria_total,
        "completed_tasks": completed_tasks,
        "honestly_terminal_tasks": _honestly_terminal_tasks(criteria=criteria, sources=sources),
        "failed_or_blocked_tasks": failed_or_blocked_tasks,
        "automata_contract_gate": _automata_contract_gate(sources),
        "satquest_verifier_gate": _satquest_verifier_gate(sources),
        "residual_drift_gate": _residual_drift_gate(sources),
        "fr11_positive_utility_gate": _fr11_positive_utility_gate(sources),
        "product_line_carry_forward_gate": _product_line_carry_forward_gate(sources),
        "claim_isolation_router_gate": _claim_isolation_router_gate(sources),
        "arm_ebm_diagnostic_boundary": _arm_ebm_diagnostic_boundary(sources),
        "thrml_next_scaling_gate": _thrml_next_scaling_gate(sources),
        "extropic_access_readiness_gate": _extropic_access_readiness_gate(sources),
        "recommended_119_focus": _recommended_119_focus(),
        "ops_reconciliation_needed": _ops_reconciliation_needed(),
        "active_roadmap_modified": active_roadmap_modified,
        "conductor_modified": conductor_modified,
        "source_inputs_read": _source_inputs_read(
            missing_source_ids=missing,
            roadmap_doc_text=roadmap_doc_text,
            research_roadmap_yaml_text=research_roadmap_yaml_text,
            research_roadmap_next_text=research_roadmap_next_text,
            research_complete_text=research_complete_text,
            ops_status_text=ops_status_text,
            ops_changelog_text=ops_changelog_text,
            ops_known_issues_text=ops_known_issues_text,
            conductor_log_text=conductor_log_text,
        ),
        "protected_files_unchanged": protected_files_unchanged,
        "required_artifact_fields_present": sorted(REQUIRED_ARTIFACT_FIELDS),
        "honest_verdict": (
            f"complete: milestone_118_{criteria_met}_of_{criteria_total}_criteria_met_"
            "satquest_fr11_limits_carried_to_119"
        ),
    }


def _protected_files_clean(root: Path) -> bool:  # pragma: no cover - environment guard
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "--",
                "research-roadmap.yaml",
                "scripts/research_conductor.py",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return result.returncode == 0


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    protected_files_unchanged: bool | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-059: write bootstrap and terminal retrospective JSON.

    The function deliberately does not edit ops, roadmap, conductor, or BMAD
    files. The conductor runs a separate reconciliation step after this focused
    artifact is stable.
    """

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing = _load_sources(root_path / "results")
    protected = (
        _protected_files_clean(root_path)
        if protected_files_unchanged is None
        else protected_files_unchanged
    )
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=missing,
        roadmap_doc_text=_read_text(root_path / ROADMAP_DOC),
        research_roadmap_yaml_text=_read_text(root_path / "research-roadmap.yaml"),
        research_roadmap_next_text=_read_text(root_path / "research-roadmap-next.yaml"),
        research_complete_text=_read_text(root_path / "research-complete.yaml"),
        ops_status_text=_read_text(root_path / "ops" / "status.md"),
        ops_changelog_text=_read_text(root_path / "ops" / "changelog.md"),
        ops_known_issues_text=_read_text(root_path / "ops" / "known-issues.md"),
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        protected_files_unchanged=protected,
    )
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
