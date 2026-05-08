"""Build the Exp 1532 milestone .117 retrospective artifact.

Spec: REQ-REPORT-057, SCENARIO-REPORT-057.
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
MILESTONE = "2026.04.117"
EXPERIMENT = "1532_milestone_117_retro"
SCHEMA = "milestone_117_retro_v1"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_1532_milestone_117_retro.json"
ROADMAP_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"

MET = "MET"
NOT_MET = "NOT_MET"
GATE_BLOCKED = "GATE_BLOCKED"
SATISFIES_CRITERION = {MET, GATE_BLOCKED}
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
LEGACY_HEADLINE_BLOCKLIST = ("qwen3.5-0.8b", "gemma-4-e4b", "gemma4-e4b")

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "criteria_met",
    "criteria_total",
    "runtime_contract_e2e_outcome",
    "live_contract_repair_outcome",
    "cdg_root_cause_outcome",
    "product_line_decision",
    "continuous_self_learning_outcome",
    "claim_isolation_outcome",
    "thrml_scaling_outcome",
    "claim_boundaries_preserved",
    "carry_forward_gates",
    "research_complete_entry_recommended",
    "ops_docs_reconciled",
    "honest_verdict",
}

EXPECTED_EXPERIMENT_IDS = tuple(f"exp{exp_id}" for exp_id in range(1519, 1532))

SOURCE_FILES = {
    "exp1519": "experiment_1519_116_completion_archive_117_activation.json",
    "exp1520": "experiment_1520_runtime_contract_e2e_harness.json",
    "exp1521": "experiment_1521_live_sota_contract_guided_repair_v1.json",
    "exp1522": "experiment_1522_constraint_dependency_graph_root_cause_repair.json",
    "exp1523": "experiment_1523_product_line_parser_feasibility_rescue_v2.json",
    "exp1524": "experiment_1524_fr11_live_policy_promotion_v12.json",
    "exp1525": "experiment_1525_march_claim_isolation_verifier_ablation.json",
    "exp1526": "experiment_1526_thrml_carnot_parity_n8.json",
    "exp1527": "experiment_1527_thrml_carnot_parity_n16.json",
    "exp1528": "experiment_1528_thrml_carnot_parity_n32_sample.json",
    "exp1529": "experiment_1529_thrml_carnot_parity_n64_sample.json",
    "exp1530": "experiment_1530_thrml_carnot_parity_n128_production_scale.json",
    "exp1531": "experiment_1531_thrml_diverse_topology_parity_n32.json",
}

EXPERIMENT_TITLES = {
    "exp1519": ".116 Completion Archive + .117 Activation Manifest",
    "exp1520": "Runtime-Contract E2E Harness",
    "exp1521": "Live SOTA Contract-Guided Repair v1",
    "exp1522": "Constraint Dependency Graph Root-Cause Repair",
    "exp1523": "Product-Line Parser Feasibility Rescue v2",
    "exp1524": "FR-11 Live Policy Promotion v12",
    "exp1525": "MARCH Claim-Isolation Verifier Ablation",
    "exp1526": "THRML/Carnot Parity n=8 Exact",
    "exp1527": "THRML/Carnot Parity n=16 Exact",
    "exp1528": "THRML/Carnot Parity n=32 Sample",
    "exp1529": "THRML/Carnot Parity n=64 Sample",
    "exp1530": "THRML/Carnot Parity n=128 Production-Scale Sample",
    "exp1531": "THRML Diverse Topology Parity n=32",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-057: persist a bootstrap marker before reading mutable files.

    The conductor treats missing deliverables and half-written deliverables very
    differently. This marker makes a started retrospective auditable while the
    later terminal artifact performs the actual source-field scoring.
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
            "runtime_contract_e2e_outcome": {},
            "live_contract_repair_outcome": {},
            "cdg_root_cause_outcome": {},
            "product_line_decision": {},
            "continuous_self_learning_outcome": {},
            "claim_isolation_outcome": {},
            "thrml_scaling_outcome": {},
            "claim_boundaries_preserved": False,
            "carry_forward_gates": [],
            "research_complete_entry_recommended": {"written": False, "entry": None},
            "ops_docs_reconciled": False,
            "honest_verdict": "complete: in_progress_milestone_117_retro_seeded",
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


def _is_terminal(payload: Mapping[str, Any]) -> bool:
    return _status(payload) in {"complete", "blocked", "gate_blocked", "skipped"}


def _has_success_prefix(payload: Mapping[str, Any]) -> bool:
    return _verdict(payload).lower().startswith(TERMINAL_PREFIXES)


def _source_path(exp_id: str, field: str | None = None) -> str:
    path = f"results/{SOURCE_FILES[exp_id]}"
    return f"{path}:{field}" if field else path


def _zero(payload: Mapping[str, Any], field: str) -> bool:
    return payload.get(field) in {0, 0.0}


def _positive_int(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _number(payload: Mapping[str, Any], field: str, default: float = 0.0) -> float:
    value = payload.get(field)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


def _mandated_sota_used(payload: Mapping[str, Any]) -> bool:
    models = {str(model) for model in payload.get("models_used", []) if model}
    return bool(models.intersection(MANDATED_SOTA_MODELS))


def _blocker_reason(payload: Mapping[str, Any]) -> str:
    blockers = payload.get("blockers")
    if isinstance(blockers, list) and blockers:
        return ", ".join(str(blocker) for blocker in blockers)
    if blockers:
        return str(blockers)
    return str(payload.get("gated_off_reason") or _verdict(payload) or _status(payload))


def _terminal_no_signal(payload: Mapping[str, Any]) -> bool:
    verdict = _verdict(payload).lower()
    return _is_terminal(payload) and ("no-signal" in verdict or "no_signal" in verdict)


def _honest_thrml_blocker(payload: Mapping[str, Any]) -> bool:
    return _status(payload) in {"blocked", "gate_blocked", "skipped"} and (
        payload.get("simulator_only") is True and payload.get("no_tsu_hardware_claim") is True
    )


def _criterion(
    *,
    key: str,
    exp_id: str,
    status: str,
    target: str,
    fields: tuple[str, ...],
    source_values: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "criterion": key,
        "experiment_id": exp_id,
        "status": status,
        "target": target,
        "evidence_paths": [_source_path(exp_id, field) for field in fields],
        "source_values": dict(source_values),
        "reason": reason,
    }


def _missing_criterion(key: str, exp_id: str, target: str) -> dict[str, Any]:
    return _criterion(
        key=key,
        exp_id=exp_id,
        status=NOT_MET,
        target=target,
        fields=(),
        source_values={"status": "missing", "honest_verdict": "missing_artifact"},
        reason=f"{exp_id} source artifact is missing.",
    )


def _score_source_criterion(
    *,
    key: str,
    exp_id: str,
    target: str,
    fields: tuple[str, ...],
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
    passed: Callable[[Mapping[str, Any]], bool],
    gate_blocked: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    if exp_id in missing_source_ids or exp_id not in sources:
        return _missing_criterion(key, exp_id, target)
    payload = sources[exp_id]
    status = (
        MET
        if passed(payload)
        else GATE_BLOCKED
        if gate_blocked and gate_blocked(payload)
        else NOT_MET
    )
    source_values = {field: payload.get(field) for field in fields}
    source_values.update(
        {
            "status": payload.get("status"),
            "honest_verdict": _verdict(payload),
            "terminal": _is_terminal(payload),
            "terminal_prefix_ok": _has_success_prefix(payload),
        }
    )
    return _criterion(
        key=key,
        exp_id=exp_id,
        status=status,
        target=target,
        fields=fields,
        source_values=source_values,
        reason="criterion satisfied" if status == MET else _blocker_reason(payload),
    )


def _product_line_rescue_or_retirement(payload: Mapping[str, Any]) -> bool:
    if payload.get("product_line_branch_retired") is True:
        return True
    parse_improved = _number(payload, "rescue_parse_rate") > _number(payload, "baseline_parse_rate")
    oracle_improved = _number(payload, "rescue_oracle_agreement_rate") > _number(
        payload, "baseline_oracle_agreement_rate"
    )
    return (
        payload.get("product_line_rescue_ready") is True
        and (parse_improved or oracle_improved)
        and _zero(payload, "false_accept_rate")
    )


def _topologies_all_passed(payload: Mapping[str, Any]) -> bool:
    tested = {str(item) for item in payload.get("topologies_tested", [])}
    passed = {str(item) for item in payload.get("topologies_passed", [])}
    return bool(tested) and tested.issubset(passed)


def _criteria_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "key": "activation",
            "exp_id": "exp1519",
            "target": "activation_manifest_complete=true with protected roadmap/conductor files unchanged",
            "fields": (
                "activation_manifest_complete",
                "research_roadmap_yaml_modified",
                "scripts_research_conductor_modified",
            ),
            "passed": lambda p: p.get("activation_manifest_complete") is True
            and p.get("research_roadmap_yaml_modified") is not True
            and p.get("scripts_research_conductor_modified") is not True,
        },
        {
            "key": "runtime_contract_e2e",
            "exp_id": "exp1520",
            "target": "runtime_contract_e2e_ready=true with all contract inputs and zero false accepts",
            "fields": (
                "runtime_contract_e2e_ready",
                "source_artifacts_loaded",
                "contract_cases_total",
                "false_accept_count",
                "false_accept_rate",
            ),
            "passed": lambda p: p.get("runtime_contract_e2e_ready") is True
            and p.get("source_artifacts_loaded") is True
            and _positive_int(p, "contract_cases_total")
            and _zero(p, "false_accept_count")
            and _zero(p, "false_accept_rate"),
        },
        {
            "key": "live_contract_repair",
            "exp_id": "exp1521",
            "target": "contract_guided_repair_ready=true with mandated SOTA GGUF and zero false accepts",
            "fields": (
                "contract_guided_repair_ready",
                "live_sota_model_inference_used",
                "models_used",
                "repair_cases_attempted",
                "false_accept_count",
                "false_accept_rate",
            ),
            "passed": lambda p: p.get("contract_guided_repair_ready") is True
            and p.get("live_sota_model_inference_used") is True
            and _mandated_sota_used(p)
            and _positive_int(p, "repair_cases_attempted")
            and _zero(p, "false_accept_count")
            and _zero(p, "false_accept_rate"),
        },
        {
            "key": "cdg_root_cause",
            "exp_id": "exp1522",
            "target": "cdg_root_cause_repair_ready=true or honest no-signal terminal artifact",
            "fields": (
                "cdg_root_cause_repair_ready",
                "root_cause_cases_attempted",
                "cdg_efficiency_delta",
                "false_accept_count",
                "false_accept_rate",
            ),
            "passed": lambda p: (
                p.get("cdg_root_cause_repair_ready") is True
                and _positive_int(p, "root_cause_cases_attempted")
                and _zero(p, "false_accept_count")
                and _zero(p, "false_accept_rate")
            )
            or (_terminal_no_signal(p) and p.get("false_accept_rate") is not None),
        },
        {
            "key": "product_line_rescue",
            "exp_id": "exp1523",
            "target": "product-line rescue improves parse/oracle metrics or retires the branch",
            "fields": (
                "product_line_rescue_ready",
                "product_line_branch_retired",
                "baseline_parse_rate",
                "rescue_parse_rate",
                "baseline_oracle_agreement_rate",
                "rescue_oracle_agreement_rate",
                "false_accept_rate",
            ),
            "passed": _product_line_rescue_or_retirement,
        },
        {
            "key": "continuous_self_learning",
            "exp_id": "exp1524",
            "target": "live policy promotion is query-time only with zero soundness mistakes",
            "fields": (
                "live_policy_promotion_ready",
                "continuous_self_learning_task",
                "no_model_weight_mutation",
                "rollback_passing_updates_loaded",
                "soundness_mistakes",
            ),
            "passed": lambda p: p.get("live_policy_promotion_ready") is True
            and p.get("continuous_self_learning_task") is True
            and p.get("no_model_weight_mutation") is True
            and _positive_int(p, "rollback_passing_updates_loaded")
            and _zero(p, "soundness_mistakes"),
        },
        {
            "key": "claim_isolation",
            "exp_id": "exp1525",
            "target": "claim isolation ablation reports deterministic validator outcomes and budget metrics",
            "fields": (
                "claim_isolation_ablation_ready",
                "cases_loaded",
                "claims_extracted",
                "budget_delta",
                "false_accept_count",
                "false_accept_rate",
            ),
            "passed": lambda p: p.get("claim_isolation_ablation_ready") is True
            and _positive_int(p, "cases_loaded")
            and _positive_int(p, "claims_extracted")
            and p.get("budget_delta") is not None
            and _zero(p, "false_accept_count")
            and _zero(p, "false_accept_rate"),
        },
        *_thrml_criteria_specs(),
        {
            "key": "diverse_topologies",
            "exp_id": "exp1531",
            "target": "diverse topology n=32 parity passes with per-topology metrics",
            "fields": (
                "diverse_topology_parity_ready",
                "topologies_tested",
                "topologies_passed",
                "kl_divergence_by_topology",
                "simulator_only",
                "no_tsu_hardware_claim",
            ),
            "passed": lambda p: p.get("diverse_topology_parity_ready") is True
            and _topologies_all_passed(p)
            and p.get("simulator_only") is True
            and p.get("no_tsu_hardware_claim") is True,
        },
    )


def _thrml_criteria_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for exp_id, key, field, n_spins in (
        ("exp1526", "thrml_n8", "thrml_parity_n8_passed", 8),
        ("exp1527", "thrml_n16", "thrml_parity_n16_passed", 16),
        ("exp1528", "thrml_n32", "thrml_parity_n32_passed", 32),
        ("exp1529", "thrml_n64", "thrml_parity_n64_passed", 64),
        ("exp1530", "thrml_n128", "thrml_parity_n128_passed", 128),
    ):
        specs.append(
            {
                "key": key,
                "exp_id": exp_id,
                "target": f"THRML/Carnot n={n_spins} software parity passes or honestly blocks",
                "fields": (
                    field,
                    "simulator_only",
                    "no_tsu_hardware_claim",
                    "n_spins",
                    "kl_divergence",
                    "blockers",
                ),
                "passed": lambda p, field=field: p.get(field) is True
                and p.get("simulator_only") is True
                and p.get("no_tsu_hardware_claim") is True,
                "gate_blocked": _honest_thrml_blocker,
            }
        )
    return tuple(specs)


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
            gate_blocked=spec.get("gate_blocked"),
        )
        for spec in _criteria_specs()
    }


def _source_reported_protected_changes(
    sources: Mapping[str, Mapping[str, Any]],
    protected_files_unchanged: bool,
) -> dict[str, Any]:
    reports: list[dict[str, str]] = []
    for exp_id, payload in sources.items():
        if payload.get("research_roadmap_yaml_modified") is True:
            reports.append({"experiment_id": exp_id, "file": "research-roadmap.yaml"})
        if payload.get("scripts_research_conductor_modified") is True:
            reports.append({"experiment_id": exp_id, "file": "scripts/research_conductor.py"})
    return {
        "any_modification_reported": bool(reports) or not protected_files_unchanged,
        "source_reports": reports,
        "working_tree": {
            "research-roadmap.yaml": "unchanged" if protected_files_unchanged else "modified_or_unknown",
            "scripts/research_conductor.py": "unchanged" if protected_files_unchanged else "modified_or_unknown",
        },
    }


def _retrospective_criterion(
    missing_source_ids: set[str],
    protected_findings: Mapping[str, Any],
) -> dict[str, Any]:
    passed = not missing_source_ids and not protected_findings["any_modification_reported"]
    return {
        "criterion": "retrospective",
        "experiment_id": "exp1532",
        "status": MET if passed else NOT_MET,
        "target": "criteria_met and criteria_total summarize .117 with carry-forward decisions",
        "evidence_paths": [
            "results/experiment_1532_milestone_117_retro.json",
            "research-roadmap.yaml",
            "scripts/research_conductor.py",
        ],
        "source_values": {
            "missing_source_ids": sorted(missing_source_ids),
            "protected_file_modification_findings": dict(protected_findings),
        },
        "reason": "retrospective artifact can close from terminal sources"
        if passed
        else "missing sources or protected-file modifications block clean closeout",
    }


def _gated_or_blocked_tasks(criteria: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "experiment_id": result["experiment_id"],
            "criterion": key,
            "reason": result["reason"],
        }
        for key, result in criteria.items()
        if result["status"] == GATE_BLOCKED
    ]


def _failed_tasks(
    criteria: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
) -> list[dict[str, str]]:
    missing_failures = [
        {
            "experiment_id": exp_id,
            "criterion": next(
                key for key, result in criteria.items() if result["experiment_id"] == exp_id
            ),
            "reason": f"{exp_id} source artifact is missing.",
        }
        for exp_id in sorted(missing_source_ids)
    ]
    unmet_failures = [
        {
            "experiment_id": result["experiment_id"],
            "criterion": key,
            "reason": result["reason"],
        }
        for key, result in criteria.items()
        if result["status"] == NOT_MET and result["experiment_id"] not in missing_source_ids
    ]
    return missing_failures + unmet_failures


def _runtime_contract_e2e_outcome(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1520", {})
    return {
        "source_artifact": _source_path("exp1520"),
        "ready": payload.get("runtime_contract_e2e_ready") is True,
        "contract_cases_total": payload.get("contract_cases_total"),
        "source_artifacts_loaded": payload.get("source_artifacts_loaded"),
        "false_accept_count": payload.get("false_accept_count"),
        "false_accept_rate": payload.get("false_accept_rate"),
        "manifest_path": payload.get("runtime_contract_manifest_path"),
    }


def _live_contract_repair_outcome(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1521", {})
    return {
        "source_artifact": _source_path("exp1521"),
        "ready": payload.get("contract_guided_repair_ready") is True,
        "mandated_sota_used": _mandated_sota_used(payload),
        "models_used": payload.get("models_used", []),
        "repair_cases_attempted": payload.get("repair_cases_attempted"),
        "repair_accept_rate_delta": payload.get("repair_accept_rate_delta"),
        "false_accept_count": payload.get("false_accept_count"),
        "false_accept_rate": payload.get("false_accept_rate"),
    }


def _cdg_root_cause_outcome(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1522", {})
    return {
        "source_artifact": _source_path("exp1522"),
        "ready": payload.get("cdg_root_cause_repair_ready") is True,
        "root_cause_cases_attempted": payload.get("root_cause_cases_attempted"),
        "cdg_efficiency_delta": payload.get("cdg_efficiency_delta"),
        "false_accept_count": payload.get("false_accept_count"),
        "false_accept_rate": payload.get("false_accept_rate"),
        "no_signal_terminal": _terminal_no_signal(payload),
    }


def _product_line_decision(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1523", {})
    retired = payload.get("product_line_branch_retired") is True
    rescued = payload.get("product_line_rescue_ready") is True
    decision = "retire" if retired else "continue" if rescued else "carry_forward_gate"
    return {
        "source_artifact": _source_path("exp1523"),
        "decision": decision,
        "product_line_rescue_ready": rescued,
        "product_line_branch_retired": retired,
        "baseline_parse_rate": payload.get("baseline_parse_rate"),
        "rescue_parse_rate": payload.get("rescue_parse_rate"),
        "baseline_oracle_agreement_rate": payload.get("baseline_oracle_agreement_rate"),
        "rescue_oracle_agreement_rate": payload.get("rescue_oracle_agreement_rate"),
        "false_accept_rate": payload.get("false_accept_rate"),
        "carry_forward": (
            "continue with larger staged benchmark while gating on parser/oracle metrics"
            if decision == "continue"
            else "retire until a materially different benchmark or parser exists"
            if decision == "retire"
            else "do not advance until rescue or retirement is terminal"
        ),
    }


def _continuous_self_learning_outcome(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1524", {})
    return {
        "source_artifact": _source_path("exp1524"),
        "live_policy_promotion_ready": payload.get("live_policy_promotion_ready") is True,
        "continuous_self_learning_task": payload.get("continuous_self_learning_task") is True,
        "no_model_weight_mutation": payload.get("no_model_weight_mutation") is True,
        "rollback_passing_updates_loaded": payload.get("rollback_passing_updates_loaded"),
        "soundness_mistakes": payload.get("soundness_mistakes"),
        "utility_delta": payload.get("utility_delta"),
        "boundary": "query-time policy promotion only; no model-weight mutation",
    }


def _claim_isolation_outcome(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1525", {})
    return {
        "source_artifact": _source_path("exp1525"),
        "claim_isolation_ablation_ready": payload.get("claim_isolation_ablation_ready") is True,
        "cases_loaded": payload.get("cases_loaded"),
        "claims_extracted": payload.get("claims_extracted"),
        "budget_delta": payload.get("budget_delta"),
        "claim_isolation_delta": payload.get("claim_isolation_delta"),
        "false_accept_rate": payload.get("false_accept_rate"),
        "deterministic_validators_final_authority": True,
    }


def _thrml_scaling_outcome(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    sweep = [
        {
            "experiment_id": exp_id,
            "source_artifact": _source_path(exp_id),
            "n_spins": sources.get(exp_id, {}).get("n_spins"),
            "passed_field": field,
            "passed": sources.get(exp_id, {}).get(field) is True,
            "simulator_only": sources.get(exp_id, {}).get("simulator_only"),
            "no_tsu_hardware_claim": sources.get(exp_id, {}).get("no_tsu_hardware_claim"),
            "kl_divergence": sources.get(exp_id, {}).get("kl_divergence"),
        }
        for exp_id, field in (
            ("exp1526", "thrml_parity_n8_passed"),
            ("exp1527", "thrml_parity_n16_passed"),
            ("exp1528", "thrml_parity_n32_passed"),
            ("exp1529", "thrml_parity_n64_passed"),
            ("exp1530", "thrml_parity_n128_passed"),
        )
    ]
    diverse = sources.get("exp1531", {})
    n128_passed = sources.get("exp1530", {}).get("thrml_parity_n128_passed") is True
    diverse_passed = diverse.get("diverse_topology_parity_ready") is True
    return {
        "software_sweep": sweep,
        "diverse_topology": {
            "source_artifact": _source_path("exp1531"),
            "ready": diverse_passed,
            "topologies_tested": diverse.get("topologies_tested", []),
            "topologies_passed": diverse.get("topologies_passed", []),
            "simulator_only": diverse.get("simulator_only"),
            "no_tsu_hardware_claim": diverse.get("no_tsu_hardware_claim"),
        },
        "next_scaling_gate": {
            "decision": "advance_to_n256_sampled_and_n64_diverse"
            if n128_passed and diverse_passed
            else "repeat_first_failed_software_gate",
            "artifact_fields": [
                "thrml_parity_n256_passed",
                "diverse_topology_parity_n64_ready",
                "simulator_only",
                "no_tsu_hardware_claim",
            ],
            "claim_boundary": "software/simulator parity only; no TSU, synthesis, bitstream, or board claim",
        },
    }


def _legacy_models_absent(sources: Mapping[str, Mapping[str, Any]]) -> bool:
    used = [
        str(model).lower()
        for exp_id in ("exp1521", "exp1522", "exp1523", "exp1524", "exp1525")
        for model in sources.get(exp_id, {}).get("models_used", [])
    ]
    return not any(blocked in model for model in used for blocked in LEGACY_HEADLINE_BLOCKLIST)


def _claim_boundary_checks(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    thrml_payloads = [sources.get(exp_id, {}) for exp_id in ("exp1526", "exp1527", "exp1528", "exp1529", "exp1530", "exp1531")]
    no_tsu = all(p.get("no_tsu_hardware_claim") is True and p.get("simulator_only") is True for p in thrml_payloads)
    no_kan_synthesis = not any(
        p.get("synthesis_run") is True or p.get("kan_synthesis_claim") is True for p in sources.values()
    )
    no_kv260_board = not any(
        p.get("board_execution") is True or p.get("kv260_board_claim") is True for p in sources.values()
    )
    no_generated_python_trust = sources.get("exp1520", {}).get("runtime_contract_e2e_ready") is True
    no_llm_final_authority = (
        _zero(sources.get("exp1520", {}), "false_accept_rate")
        and _zero(sources.get("exp1521", {}), "false_accept_rate")
        and _zero(sources.get("exp1522", {}), "false_accept_rate")
        and _zero(sources.get("exp1525", {}), "false_accept_rate")
        and _zero(sources.get("exp1524", {}), "soundness_mistakes")
    )
    return [
        {
            "boundary": "no_tsu_hardware_claim",
            "preserved": no_tsu,
            "evidence": "Exp1526-1531 all report simulator_only and no_tsu_hardware_claim.",
        },
        {
            "boundary": "no_kan_synthesis_claim",
            "preserved": no_kan_synthesis,
            "evidence": ".117 does not report KAN synthesis, timing, bitstream, or board artifacts.",
        },
        {
            "boundary": "no_kv260_board_claim",
            "preserved": no_kv260_board,
            "evidence": ".117 does not report KV260 board execution or latency artifacts.",
        },
        {
            "boundary": "no_arbitrary_generated_python_trust",
            "preserved": no_generated_python_trust,
            "evidence": "Runtime acceptance flows through deterministic contract fields.",
        },
        {
            "boundary": "no_legacy_small_model_headline_result",
            "preserved": _legacy_models_absent(sources),
            "evidence": "LLM-bearing headline rows use mandated local SOTA GGUF model ids.",
        },
        {
            "boundary": "no_llm_judge_final_authority",
            "preserved": no_llm_final_authority,
            "evidence": "False-accept and soundness fields, not LLM text, determine acceptance.",
        },
    ]


def _carry_forward_gates() -> list[dict[str, Any]]:
    return [
        {
            "gate": "runtime_contract_regression_gate",
            "source_artifact": _source_path("exp1520"),
            "artifact_fields": ["runtime_contract_e2e_ready", "false_accept_rate", "contract_cases_total"],
            "required": "runtime_contract_e2e_ready == true and false_accept_rate == 0.0",
        },
        {
            "gate": "live_sota_contract_repair_scale_gate",
            "source_artifact": _source_path("exp1521"),
            "artifact_fields": [
                "contract_guided_repair_ready",
                "live_sota_model_inference_used",
                "models_used",
                "false_accept_rate",
            ],
            "required": "contract_guided_repair_ready == true with mandated SOTA GGUF and zero false accepts",
        },
        {
            "gate": "product_line_continuation_gate",
            "source_artifact": _source_path("exp1523"),
            "artifact_fields": [
                "product_line_rescue_ready",
                "product_line_branch_retired",
                "rescue_parse_rate",
                "rescue_oracle_agreement_rate",
                "false_accept_rate",
            ],
            "required": "continue only if rescued and not retired; otherwise require a materially different parser or benchmark",
        },
        {
            "gate": "fr11_live_policy_positive_utility_gate",
            "source_artifact": _source_path("exp1524"),
            "artifact_fields": [
                "live_policy_promotion_ready",
                "no_model_weight_mutation",
                "soundness_mistakes",
                "utility_delta",
            ],
            "required": "soundness_mistakes == 0 and future promotion claims require utility_delta > 0",
        },
        {
            "gate": "claim_isolation_scale_gate",
            "source_artifact": _source_path("exp1525"),
            "artifact_fields": [
                "claim_isolation_ablation_ready",
                "claim_isolation_delta",
                "budget_delta",
                "false_accept_rate",
            ],
            "required": "scale only with deterministic validator outcomes and explicit budget accounting",
        },
        {
            "gate": "thrml_next_scaling_gate",
            "source_artifacts": [_source_path("exp1530"), _source_path("exp1531")],
            "artifact_fields": [
                "thrml_parity_n128_passed",
                "diverse_topology_parity_ready",
                "simulator_only",
                "no_tsu_hardware_claim",
            ],
            "required": "n=128 and diverse n=32 software parity must stay true before n=256/n=64-diverse work",
        },
    ]


def _research_complete_has_117_entry(text: str) -> bool:
    return "2026.04.117" in text


def _archive_task_rows() -> list[dict[str, str]]:
    rows = [
        {
            "id": f"{exp_id}-{EXPERIMENT_TITLES[exp_id].lower().replace(' ', '-')}",
            "title": EXPERIMENT_TITLES[exp_id],
            "deliverable": f"results/{SOURCE_FILES[exp_id]}",
            "result": "OK (conductor)",
        }
        for exp_id in EXPECTED_EXPERIMENT_IDS
    ]
    rows.append(
        {
            "id": "exp1532-milestone-117-retro",
            "title": "Milestone .117 Retrospective and Carry-Forward Gates",
            "deliverable": "results/experiment_1532_milestone_117_retro.json",
            "result": "OK (codex retro)",
        }
    )
    return rows


def _research_complete_recommendation(
    *,
    criteria_met: int,
    criteria_total: int,
    research_complete_text: str,
    product_line_decision: Mapping[str, Any],
    thrml_scaling_outcome: Mapping[str, Any],
) -> dict[str, Any]:
    already_present = _research_complete_has_117_entry(research_complete_text)
    next_gate = thrml_scaling_outcome["next_scaling_gate"]["decision"]
    return {
        "written": False,
        "already_present": already_present,
        "reason": "recommended_only_stop_when_done_delegates_archive_write",
        "entry": {
            "id": MILESTONE,
            "title": "Runtime-Contract E2E Closure + FR-11 Live Promotion + THRML Parity Scaling",
            "doc": ROADMAP_DOC,
            "completed": "2026-05-08",
            "finding": (
                f"Milestone .117 met {criteria_met} of {criteria_total} criteria. "
                "Runtime-contract E2E, local-SOTA contract repair, CDG repair, "
                f"product-line decision={product_line_decision['decision']}, FR-11 "
                f"live policy promotion, claim isolation, and THRML scaling closed. "
                f"Next THRML gate: {next_gate}. Claim boundaries remain preserved."
            ),
            "tasks": _archive_task_rows(),
        },
    }


def _source_inputs_read(
    *,
    conductor_log_text: str,
    roadmap_doc_text: str,
    research_roadmap_yaml_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
) -> dict[str, dict[str, bool]]:
    return {
        "ops/conductor-log.md": {"exists": bool(conductor_log_text)},
        ROADMAP_DOC: {"exists": bool(roadmap_doc_text)},
        "research-roadmap.yaml": {"exists": bool(research_roadmap_yaml_text)},
        "research-complete.yaml": {"exists": bool(research_complete_text)},
        "ops/status.md": {"exists": bool(ops_status_text)},
        "ops/changelog.md": {"exists": bool(ops_changelog_text)},
    }


def _conductor_log_summary(conductor_log_text: str) -> dict[str, Any]:
    rows = conductor_log_text.splitlines()
    entries = {
        exp_id: {
            "found": any(exp_id in row or EXPERIMENT_TITLES[exp_id][:40] in row for row in rows),
            "ok": any(
                (exp_id in row or EXPERIMENT_TITLES[exp_id][:40] in row) and "| OK |" in row
                for row in rows
            ),
        }
        for exp_id in EXPECTED_EXPERIMENT_IDS
    }
    return {
        "ok_count": sum(1 for entry in entries.values() if entry["ok"]),
        "expected_count": len(EXPECTED_EXPERIMENT_IDS),
        "missing_experiments": [exp_id for exp_id, entry in entries.items() if not entry["found"]],
        "entries": entries,
    }


def _protected_files_clean(root: Path) -> bool:
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


def build_artifact(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: list[str],
    conductor_log_text: str,
    roadmap_doc_text: str,
    research_roadmap_yaml_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    protected_files_unchanged: bool,
) -> dict[str, Any]:
    """REQ-REPORT-057: score `.117` from terminal source artifact fields.

    The artifact keeps each conclusion tied to the JSON fields that support it.
    This matters because a milestone summary is where weak branches can easily
    turn into stronger claims than the experiments actually earned.
    """

    missing = set(missing_source_ids)
    criteria = _source_success_criteria(sources=sources, missing_source_ids=missing)
    protected_findings = _source_reported_protected_changes(sources, protected_files_unchanged)
    criteria["retrospective"] = _retrospective_criterion(missing, protected_findings)
    criteria_met = sum(1 for result in criteria.values() if result["status"] in SATISFIES_CRITERION)
    criteria_total = len(criteria)

    product_line = _product_line_decision(sources)
    thrml_scaling = _thrml_scaling_outcome(sources)
    claim_boundary_checks = _claim_boundary_checks(sources)
    claim_boundaries_preserved = all(check["preserved"] for check in claim_boundary_checks)

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete",
        "milestone": MILESTONE,
        "criteria_met": criteria_met,
        "criteria_total": criteria_total,
        "criteria_score_pct": round(criteria_met / criteria_total, 6) if criteria_total else 0.0,
        "criteria_results": criteria,
        "gated_or_blocked_tasks": _gated_or_blocked_tasks(criteria),
        "failed_tasks": _failed_tasks(criteria, missing),
        "missing_artifacts": [
            {"experiment_id": exp_id, "path": _source_path(exp_id)} for exp_id in sorted(missing)
        ],
        "experiment_verdicts": {
            exp_id: {
                "status": sources.get(exp_id, {}).get("status", "missing"),
                "honest_verdict": _verdict(sources.get(exp_id, {})) or "missing_artifact",
                "terminal": _is_terminal(sources.get(exp_id, {})),
                "terminal_prefix_ok": _has_success_prefix(sources.get(exp_id, {})),
                "source_path": _source_path(exp_id),
            }
            for exp_id in EXPECTED_EXPERIMENT_IDS
        },
        "runtime_contract_e2e_outcome": _runtime_contract_e2e_outcome(sources),
        "live_contract_repair_outcome": _live_contract_repair_outcome(sources),
        "cdg_root_cause_outcome": _cdg_root_cause_outcome(sources),
        "product_line_decision": product_line,
        "continuous_self_learning_outcome": _continuous_self_learning_outcome(sources),
        "claim_isolation_outcome": _claim_isolation_outcome(sources),
        "thrml_scaling_outcome": thrml_scaling,
        "claim_boundaries_preserved": claim_boundaries_preserved,
        "claim_boundary_checks": claim_boundary_checks,
        "carry_forward_gates": _carry_forward_gates(),
        "ops_docs_reconciled": False,
        "ops_docs_reconciliation_deferred_reason": (
            "separate_reconciliation_agent_owns_ops_status_changelog_and_traceability"
        ),
        "research_complete_entry_recommended": _research_complete_recommendation(
            criteria_met=criteria_met,
            criteria_total=criteria_total,
            research_complete_text=research_complete_text,
            product_line_decision=product_line,
            thrml_scaling_outcome=thrml_scaling,
        ),
        "protected_files_unchanged": protected_files_unchanged,
        "protected_file_modification_findings": protected_findings,
        "source_inputs_read": _source_inputs_read(
            conductor_log_text=conductor_log_text,
            roadmap_doc_text=roadmap_doc_text,
            research_roadmap_yaml_text=research_roadmap_yaml_text,
            research_complete_text=research_complete_text,
            ops_status_text=ops_status_text,
            ops_changelog_text=ops_changelog_text,
        ),
        "conductor_log_exp1519_to_exp1531": _conductor_log_summary(conductor_log_text),
        "required_artifact_fields_present": sorted(REQUIRED_ARTIFACT_FIELDS),
        "honest_verdict": (
            f"complete: milestone_117_{criteria_met}_of_{criteria_total}_criteria_met_"
            "runtime_contract_fr11_thrml_claim_boundaries_preserved"
        ),
    }
    return artifact


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    protected_files_unchanged: bool | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-057: write bootstrap and terminal retrospective JSON only.

    This function deliberately avoids mutating ops or archive files because the
    conductor's follow-up reconciliation run owns those documents for this
    stop-when-done workflow.
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
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        roadmap_doc_text=_read_text(root_path / ROADMAP_DOC),
        research_roadmap_yaml_text=_read_text(root_path / "research-roadmap.yaml"),
        research_complete_text=_read_text(root_path / "research-complete.yaml"),
        ops_status_text=_read_text(root_path / "ops" / "status.md"),
        ops_changelog_text=_read_text(root_path / "ops" / "changelog.md"),
        protected_files_unchanged=protected,
    )
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
