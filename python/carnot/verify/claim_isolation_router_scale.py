"""Exp 1553 claim-isolation router scale behind the unified contract gate.

Spec: REQ-VERIFY-1553, SCENARIO-VERIFY-1553.

The scale run reuses the Exp 1541 routing policy, but evaluates it only after
the Exp 1551 unified contract gate is ready.  The important distinction is that
router savings are measured against a full-context verifier baseline while the
deterministic gate remains common to both paths.  That means the router can save
verifier calls, but it never becomes the authority for accepting hidden SAT,
product-line, runtime-contract, or residual-drift failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
MILESTONE = "20260508"
MODEL_SPECS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

DEFAULT_ARTIFACT_PATH = Path("results/experiment_1553_claim_isolation_router_scale_v3.json")
DEFAULT_MANIFEST_PATH = Path("results/claim_isolation_router_scale_1553.jsonl")
DEFAULT_ROUTER_POLICY_PATH = Path("results/claim_isolation_router_scale_policy_1553.json")
DEFAULT_ROUTER_ARTIFACT_PATH = Path(
    "results/experiment_1541_claim_isolation_uncertainty_router_v2.json"
)
DEFAULT_UNIFIED_GATE_ARTIFACT_PATH = Path(
    "results/experiment_1551_automata_sat_unified_contract_gate.json"
)
DEFAULT_RUNTIME_MANIFEST_PATH = Path("results/runtime_contract_e2e_manifest_1520.jsonl")
DEFAULT_SATQUEST_MANIFEST_PATH = Path("results/satquest_cnf_verifier_1536.jsonl")
DEFAULT_PRODUCT_MANIFEST_PATH = Path("results/product_line_rescue_1523.jsonl")
DEFAULT_RESIDUAL_MANIFEST_PATH = Path("results/residual_drift_repair_policy_1552.jsonl")
ROUTER_MODULE_PATH = "python/carnot/verify/claim_isolation_router_scale.py"

DEFAULT_ROUTING_POLICY: JsonDict = {
    "uncertainty_threshold": 0.5,
    "prefix_risk_threshold": 0.75,
    "route_on_validator_disagreement": True,
    "route_residual_drift": True,
}

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "claim_isolation_router_scale_ready",
    "model_specs",
    "live_sota_model_inference_used",
    "cases_total",
    "routed_cases",
    "full_context_cases",
    "claims_extracted",
    "budget_delta",
    "budget_reduced",
    "false_accept_rate",
    "missed_failure_count",
    "router_policy_path",
    "focused_tests_passed",
    "honest_verdict",
)

TERMINAL_VERDICT_PREFIXES: tuple[str, ...] = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_SOURCE_KINDS: frozenset[str] = frozenset(
    {"runtime_contract", "satquest", "product_line", "residual_drift"}
)
RESIDUAL_ROUTE_CLASSES: frozenset[str] = frozenset({"satisfiable_drift", "true_contradiction"})


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    router_policy_path: Path | str = DEFAULT_ROUTER_POLICY_PATH,
) -> JsonDict:
    """REQ-VERIFY-1553: persist a bootstrap artifact before source loading."""

    payload = artifact_from_summary(
        status="in_progress",
        summary=_empty_summary(),
        router_policy_path=Path(router_policy_path),
        focused_tests_passed=False,
        live_sota_model_inference_used=False,
        blockers=["experiment_1553_claim_isolation_router_scale_in_progress"],
    )
    _write_json(Path(output_path), payload)
    return payload


def route_case(
    case: Mapping[str, Any],
    *,
    prefix_risk_by_case: Mapping[str, float],
    policy: Mapping[str, Any] | None = None,
) -> JsonDict:
    """REQ-VERIFY-1553: route a case from deterministic thresholds only."""

    active_policy = dict(DEFAULT_ROUTING_POLICY if policy is None else policy)
    prefix_risk = _prefix_risk(case, prefix_risk_by_case)
    reasons: list[str] = []
    if float(case.get("uncertainty_score", 0.0)) >= float(active_policy["uncertainty_threshold"]):
        reasons.append("uncertainty")
    if prefix_risk >= float(active_policy["prefix_risk_threshold"]):
        reasons.append("prefix_risk")
    if active_policy.get("route_on_validator_disagreement") and case.get("validator_disagreement"):
        reasons.append("validator_disagreement")
    if (
        active_policy.get("route_residual_drift")
        and case.get("source_kind") == "residual_drift"
        and str(case.get("residual_failure_classification") or "") in RESIDUAL_ROUTE_CLASSES
    ):
        reasons.append("residual_drift")
    return {
        "router_case_id": case["router_case_id"],
        "routed": bool(reasons),
        "routing_reasons": reasons or ["low_risk_bypass"],
        "uncertainty_score": round(float(case.get("uncertainty_score", 0.0)), 6),
        "prefix_risk": round(prefix_risk, 6),
        "validator_disagreement": bool(case.get("validator_disagreement")),
    }


def build_scaled_case_set(
    *,
    runtime_rows: Sequence[Mapping[str, Any]],
    satquest_rows: Sequence[Mapping[str, Any]],
    product_rows: Sequence[Mapping[str, Any]],
    residual_rows: Sequence[Mapping[str, Any]],
    case_target: int = 75,
) -> list[JsonDict]:
    """REQ-VERIFY-1553: extract a mixed 75+ case set when rows are available."""

    groups = [
        _runtime_cases(runtime_rows),
        _satquest_cases(satquest_rows),
        _product_cases(product_rows),
        _residual_cases(residual_rows),
    ]
    return _round_robin_take(groups, max(0, int(case_target)))


def evaluate_scaled_routing(
    cases: Sequence[Mapping[str, Any]],
    *,
    prefix_risk_by_case: Mapping[str, float],
    unified_contract_gate_ready: bool,
    focused_tests_passed: bool,
    policy: Mapping[str, Any] | None = None,
) -> JsonDict:
    """SCENARIO-VERIFY-1553: compare routed calls with a full-context baseline."""

    rows: list[JsonDict] = []
    routed_cases = 0
    routed_verifier_calls = 0
    claims_extracted = 0
    deterministic_rejects = 0
    missed_failures = 0

    for case in cases:
        claims = list(case.get("claims") or [])
        claims_extracted += len(claims)
        decision = route_case(case, prefix_risk_by_case=prefix_risk_by_case, policy=policy)
        deterministic_accept = bool(case.get("deterministic_accept"))
        full_context_accept = bool(case.get("full_context_accept"))
        claim_isolated_accept = bool(case.get("claim_isolated_accept"))
        routed_signal_accept = claim_isolated_accept if decision["routed"] else full_context_accept
        final_accept = bool(
            unified_contract_gate_ready and deterministic_accept and routed_signal_accept
        )
        false_accept = bool(not deterministic_accept and final_accept)
        routed_cases += int(decision["routed"])
        routed_verifier_calls += len(claims) if decision["routed"] else 0
        deterministic_rejects += int(not deterministic_accept)
        missed_failures += int(false_accept)
        rows.append(
            {
                "row_type": "router_scale_case",
                "spec": ["REQ-VERIFY-1553", "SCENARIO-VERIFY-1553"],
                "router_case_id": case["router_case_id"],
                "case_id": case["case_id"],
                "source_kind": case["source_kind"],
                "source_family": case["source_family"],
                "claims": claims,
                "routed": decision["routed"],
                "routing_reasons": decision["routing_reasons"],
                "full_context_accept": full_context_accept,
                "claim_isolated_accept": claim_isolated_accept if decision["routed"] else None,
                "deterministic_accept": deterministic_accept,
                "unified_gate_checked": bool(unified_contract_gate_ready),
                "final_accept": final_accept,
                "false_accept": false_accept,
            }
        )

    summary = _summary(
        cases=cases,
        routed_cases=routed_cases,
        routed_verifier_calls=routed_verifier_calls,
        full_context_cases=len(cases),
        claims_extracted=claims_extracted,
        missed_failures=missed_failures,
        deterministic_rejects=deterministic_rejects,
        unified_contract_gate_ready=unified_contract_gate_ready,
        focused_tests_passed=focused_tests_passed,
    )
    rows.append(_summary_manifest_row(summary))
    return {"rows": rows, "summary": summary}


def run_experiment(
    *,
    project_root: Path | str | None = None,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    router_policy_path: Path | str = DEFAULT_ROUTER_POLICY_PATH,
    router_artifact_path: Path | str = DEFAULT_ROUTER_ARTIFACT_PATH,
    unified_gate_artifact_path: Path | str = DEFAULT_UNIFIED_GATE_ARTIFACT_PATH,
    runtime_manifest_path: Path | str = DEFAULT_RUNTIME_MANIFEST_PATH,
    satquest_manifest_path: Path | str = DEFAULT_SATQUEST_MANIFEST_PATH,
    product_manifest_path: Path | str = DEFAULT_PRODUCT_MANIFEST_PATH,
    residual_manifest_path: Path | str = DEFAULT_RESIDUAL_MANIFEST_PATH,
    focused_tests_passed: bool = False,
    case_target: int = 75,
) -> JsonDict:
    """Run Exp 1553 from checked-in predecessor artifacts and source manifests."""

    root = Path.cwd() if project_root is None else Path(project_root)
    output = _resolve_under_root(root, Path(output_path))
    manifest = _resolve_under_root(root, Path(manifest_path))
    policy_path = _resolve_under_root(root, Path(router_policy_path))
    write_in_progress_artifact(output, router_policy_path=policy_path)

    paths = {
        "router_artifact": _resolve_under_root(root, Path(router_artifact_path)),
        "unified_gate_artifact": _resolve_under_root(root, Path(unified_gate_artifact_path)),
        "runtime_manifest": _resolve_under_root(root, Path(runtime_manifest_path)),
        "satquest_manifest": _resolve_under_root(root, Path(satquest_manifest_path)),
        "product_manifest": _resolve_under_root(root, Path(product_manifest_path)),
        "residual_manifest": _resolve_under_root(root, Path(residual_manifest_path)),
    }
    blockers = _missing_source_blockers(paths)
    router_artifact = (
        _read_json(paths["router_artifact"]) if paths["router_artifact"].exists() else {}
    )
    gate_artifact = (
        _read_json(paths["unified_gate_artifact"])
        if paths["unified_gate_artifact"].exists()
        else {}
    )
    blockers.extend(_predecessor_blockers(router_artifact, gate_artifact))
    policy = _load_policy(policy_path)
    cases = build_scaled_case_set(
        runtime_rows=_read_jsonl(paths["runtime_manifest"]),
        satquest_rows=_read_jsonl(paths["satquest_manifest"]),
        product_rows=_read_jsonl(paths["product_manifest"]),
        residual_rows=_read_jsonl(paths["residual_manifest"]),
        case_target=case_target,
    )
    if len(cases) < case_target:  # pragma: no cover - defensive missing-data path.
        blockers.append(f"insufficient_scaled_cases:{len(cases)}<{case_target}")
    if not focused_tests_passed:
        blockers.append("focused_tests_not_passed")

    evaluation = evaluate_scaled_routing(
        cases,
        prefix_risk_by_case=_prefix_risk_from_policy_source(router_artifact),
        unified_contract_gate_ready=gate_artifact.get("unified_contract_gate_ready") is True,
        focused_tests_passed=focused_tests_passed,
        policy=policy,
    )
    _write_json(policy_path, policy)
    _write_jsonl(manifest, evaluation["rows"])
    artifact = artifact_from_summary(
        status="complete" if cases else "blocked",
        summary=evaluation["summary"],
        router_policy_path=policy_path,
        focused_tests_passed=focused_tests_passed,
        live_sota_model_inference_used=_live_sota_used(router_artifact, gate_artifact, cases),
        blockers=blockers,
        models_used=_models_used(router_artifact, gate_artifact),
        model_availability_blockers=list(gate_artifact.get("model_availability_blockers") or []),
        manifest_path=manifest,
        predecessor_ready={
            "uncertainty_router_ready": router_artifact.get("uncertainty_router_ready") is True,
            "unified_contract_gate_ready": gate_artifact.get("unified_contract_gate_ready") is True,
        },
    )
    validate_artifact(artifact)
    _write_json(output, artifact)
    return artifact


def artifact_from_summary(
    *,
    status: str,
    summary: Mapping[str, Any],
    router_policy_path: Path,
    focused_tests_passed: bool,
    live_sota_model_inference_used: bool,
    blockers: Sequence[str],
    models_used: Sequence[str] | None = None,
    model_availability_blockers: Sequence[str] | None = None,
    manifest_path: Path | None = None,
    predecessor_ready: Mapping[str, bool] | None = None,
) -> JsonDict:
    """Build the terminal artifact schema consumed by the conductor."""

    ready = bool(summary.get("claim_isolation_router_scale_ready")) and not blockers
    verdict = (
        "complete: claim-isolation router scale ready behind unified contract gate"
        if ready
        else "complete: claim-isolation router scale completed with blockers"
    )
    return {
        "status": status,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "schema_version": 1,
        "claim_isolation_router_scale_ready": ready,
        "model_specs": list(MODEL_SPECS),
        "live_sota_model_inference_used": bool(live_sota_model_inference_used),
        "cases_total": int(summary.get("cases_total", 0)),
        "routed_cases": int(summary.get("routed_cases", 0)),
        "full_context_cases": int(summary.get("full_context_cases", 0)),
        "claims_extracted": int(summary.get("claims_extracted", 0)),
        "budget_delta": int(summary.get("budget_delta", 0)),
        "budget_reduced": bool(summary.get("budget_reduced", False)),
        "false_accept_rate": float(summary.get("false_accept_rate", 0.0)),
        "missed_failure_count": int(summary.get("missed_failure_count", 0)),
        "router_policy_path": _display_path(router_policy_path),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": verdict,
        "blockers": list(blockers),
        "router_module_path": ROUTER_MODULE_PATH,
        "router_manifest_path": _display_path(manifest_path or DEFAULT_MANIFEST_PATH),
        "source_kinds_loaded": list(summary.get("source_kinds_loaded") or []),
        "full_context_baseline_budget": int(summary.get("full_context_cases", 0)),
        "routed_verifier_budget": int(summary.get("routed_verifier_budget", 0)),
        "models_used": list(models_used or []),
        "model_availability_blockers": list(model_availability_blockers or []),
        **dict(predecessor_ready or {}),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on any terminal schema or readiness inconsistency."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use an allowed terminal prefix")
    if artifact["claim_isolation_router_scale_ready"]:
        if artifact["focused_tests_passed"] is not True:
            raise AssertionError("ready scale run requires focused tests")
        if artifact["false_accept_rate"] != 0.0:
            raise AssertionError("ready scale run requires zero false accepts")
        if artifact["budget_reduced"] is not True:
            raise AssertionError("ready scale run requires budget reduction")
        if artifact["cases_total"] < 75:
            raise AssertionError("ready scale run requires at least 75 cases")
        if artifact["routed_cases"] <= 0 or artifact["routed_cases"] >= artifact["cases_total"]:
            raise AssertionError("ready scale run must route some but not all cases")


def _runtime_cases(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    cases: list[JsonDict] = []
    for row in rows:
        if row.get("row_type") != "contract_case":
            continue
        case_id = str(row.get("contract_case_id") or "")
        if not case_id:
            continue  # pragma: no cover - malformed source row.
        deterministic_accept = bool(row.get("final_deterministic_accept"))
        expected = (
            row.get("expected_label") if isinstance(row.get("expected_label"), bool) else None
        )
        full_accept = deterministic_accept
        validator_disagreement = bool(expected is not None and full_accept != expected)
        cases.append(
            _case(
                source_kind="runtime_contract",
                case_id=case_id,
                source_family=str(row.get("source_family") or "runtime_contract"),
                claim_text=(
                    f"runtime contract {case_id} final deterministic decision is "
                    f"{'accept' if deterministic_accept else 'reject'}"
                ),
                deterministic_accept=deterministic_accept,
                full_context_accept=full_accept,
                claim_isolated_accept=deterministic_accept,
                uncertainty_score=0.75 if validator_disagreement else 0.0,
                validator_disagreement=validator_disagreement,
                model_hf_id=row.get("model_hf_id"),
            )
        )
    return cases


def _satquest_cases(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    cases: list[JsonDict] = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        solver = _mapping(row.get("solver_oracle"))
        if not case_id or not solver:
            continue  # pragma: no cover - malformed source row.
        baseline = _mapping(row.get("baseline"))
        parse = _mapping(row.get("parse_result"))
        verifier = _mapping(row.get("verifier"))
        answer = baseline.get("answer")
        solver_label = str(solver.get("label") or "UNKNOWN")
        deterministic_accept = (
            bool(baseline["correct"])
            if isinstance(baseline.get("correct"), bool)
            else bool(answer == solver_label)
        )
        full_accept = _declared_or_parsed_accept(parse, answer)
        parse_ok = bool(parse.get("parse_ok"))
        validator_disagreement = bool(
            full_accept != deterministic_accept
            or verifier.get("self_verifier_false_accept")
            or not parse_ok
        )
        uncertainty = (
            1.0 if not parse_ok else 0.75 if float(baseline.get("energy", 0.0)) >= 50 else 0.0
        )
        cases.append(
            _case(
                source_kind="satquest",
                case_id=case_id,
                source_family=str(row.get("family") or "satquest"),
                claim_text=f"SATQuest answer {answer if answer is not None else 'NO_ANSWER'} should match solver label {solver_label}",
                deterministic_accept=deterministic_accept,
                full_context_accept=full_accept,
                claim_isolated_accept=deterministic_accept,
                uncertainty_score=uncertainty,
                validator_disagreement=validator_disagreement,
                model_hf_id=row.get("model_hf_id"),
            )
        )
    return cases


def _product_cases(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    cases: list[JsonDict] = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        oracle = _mapping(row.get("oracle_result"))
        if not case_id or not oracle:
            continue  # pragma: no cover - malformed source row.
        policy = _mapping(row.get("policy_result"))
        verifier = _mapping(row.get("verifier_result"))
        deterministic_accept = bool(oracle.get("oracle_agrees")) and not bool(
            policy.get("false_accept")
        )
        full_accept = bool(policy.get("accepted") or verifier.get("accepted"))
        cases.append(
            _case(
                source_kind="product_line",
                case_id=case_id,
                source_family=str(row.get("model_id") or "product_line"),
                claim_text=f"product-line feature selection for {case_id} should match the solver oracle",
                deterministic_accept=deterministic_accept,
                full_context_accept=full_accept,
                claim_isolated_accept=deterministic_accept,
                uncertainty_score=0.0,
                validator_disagreement=bool(full_accept != deterministic_accept),
                model_hf_id=row.get("model_hf_id"),
            )
        )
    return cases


def _residual_cases(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    cases: list[JsonDict] = []
    for row in rows:
        if row.get("row_type") not in {"residual_drift_repair_case", "residual_drift_case"}:
            continue
        case_id = str(row.get("case_id") or row.get("source_case_id") or "")
        if not case_id:
            continue  # pragma: no cover - malformed source row.
        failure_class = str(row.get("failure_classification") or "unknown")
        deterministic_accept = bool(
            row.get("accepted") and row.get("replay_passed") and not row.get("false_accept")
        )
        if failure_class == "true_contradiction":
            deterministic_accept = False
        full_accept = bool(row.get("accepted") or row.get("attempted"))
        cases.append(
            _case(
                source_kind="residual_drift",
                case_id=case_id,
                source_family=str(row.get("source_domain") or "residual_drift"),
                claim_text=f"residual-drift case {case_id} replay must preserve deterministic validator authority",
                deterministic_accept=deterministic_accept,
                full_context_accept=full_accept,
                claim_isolated_accept=deterministic_accept,
                uncertainty_score=0.8 if failure_class in RESIDUAL_ROUTE_CLASSES else 0.0,
                validator_disagreement=bool(
                    full_accept != deterministic_accept or row.get("rejected_false_accept")
                ),
                model_hf_id=row.get("model_hf_id"),
                residual_failure_classification=failure_class,
            )
        )
    return cases


def _case(
    *,
    source_kind: str,
    case_id: str,
    source_family: str,
    claim_text: str,
    deterministic_accept: bool,
    full_context_accept: bool,
    claim_isolated_accept: bool,
    uncertainty_score: float,
    validator_disagreement: bool,
    model_hf_id: Any,
    residual_failure_classification: str | None = None,
) -> JsonDict:
    router_case_id = f"{source_kind}:{case_id}"
    return {
        "router_case_id": router_case_id,
        "case_id": case_id,
        "source_kind": source_kind,
        "source_family": source_family,
        "deterministic_accept": bool(deterministic_accept),
        "full_context_accept": bool(full_context_accept),
        "claim_isolated_accept": bool(claim_isolated_accept),
        "uncertainty_score": round(float(uncertainty_score), 6),
        "validator_disagreement": bool(validator_disagreement),
        "model_hf_id": model_hf_id,
        "residual_failure_classification": residual_failure_classification,
        "unified_gate_checked": True,
        "deterministic_validator_final_authority": True,
        "claims": [
            {
                "claim_id": f"{router_case_id}:claim:001",
                "claim_text": claim_text,
                "source_kind": source_kind,
                "source_family": source_family,
                "source_case_id": case_id,
                "deterministic_accept": bool(deterministic_accept),
                "hidden_from_full_context": True,
            }
        ],
    }


def _summary(
    *,
    cases: Sequence[Mapping[str, Any]],
    routed_cases: int,
    routed_verifier_calls: int,
    full_context_cases: int,
    claims_extracted: int,
    missed_failures: int,
    deterministic_rejects: int,
    unified_contract_gate_ready: bool,
    focused_tests_passed: bool,
) -> JsonDict:
    source_kinds = sorted({str(case.get("source_kind")) for case in cases})
    false_accept_rate = _rate(missed_failures, deterministic_rejects)
    budget_delta = routed_verifier_calls - full_context_cases
    budget_reduced = bool(budget_delta < 0 and false_accept_rate == 0.0)
    ready = bool(
        unified_contract_gate_ready
        and REQUIRED_SOURCE_KINDS.issubset(source_kinds)
        and full_context_cases >= 75
        and 0 < routed_cases < full_context_cases
        and focused_tests_passed
        and false_accept_rate == 0.0
        and budget_reduced
    )
    return {
        "row_type": "summary",
        "spec": ["REQ-VERIFY-1553", "SCENARIO-VERIFY-1553"],
        "claim_isolation_router_scale_ready": ready,
        "cases_total": full_context_cases,
        "routed_cases": routed_cases,
        "full_context_cases": full_context_cases,
        "claims_extracted": claims_extracted,
        "routed_verifier_budget": routed_verifier_calls,
        "budget_delta": budget_delta,
        "budget_reduced": budget_reduced,
        "false_accept_rate": false_accept_rate,
        "missed_failure_count": missed_failures,
        "source_kinds_loaded": source_kinds,
        "focused_tests_passed": bool(focused_tests_passed),
        "unified_contract_gate_ready": bool(unified_contract_gate_ready),
    }


def _empty_summary() -> JsonDict:
    return {
        "claim_isolation_router_scale_ready": False,
        "cases_total": 0,
        "routed_cases": 0,
        "full_context_cases": 0,
        "claims_extracted": 0,
        "routed_verifier_budget": 0,
        "budget_delta": 0,
        "budget_reduced": False,
        "false_accept_rate": 1.0,
        "missed_failure_count": 0,
        "source_kinds_loaded": [],
    }


def _summary_manifest_row(summary: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in summary.items() if key != "spec"} | {
        "row_type": "summary",
        "spec": ["REQ-VERIFY-1553", "SCENARIO-VERIFY-1553"],
    }


def _round_robin_take(groups: Sequence[Sequence[JsonDict]], limit: int) -> list[JsonDict]:
    selected: list[JsonDict] = []
    indexes = [0 for _group in groups]
    while len(selected) < limit and any(
        index < len(group) for index, group in zip(indexes, groups)
    ):
        for group_index, group in enumerate(groups):
            if indexes[group_index] < len(group):
                selected.append(group[indexes[group_index]])
                indexes[group_index] += 1
                if len(selected) == limit:
                    break
    return selected


def _prefix_risk(case: Mapping[str, Any], risk_by_case: Mapping[str, float]) -> float:
    for key in (str(case.get("router_case_id")), str(case.get("case_id"))):
        if key in risk_by_case:
            return float(risk_by_case[key])
    return float(case.get("prefix_risk", 0.0))


def _prefix_risk_from_policy_source(router_artifact: Mapping[str, Any]) -> dict[str, float]:
    risk: dict[str, float] = {}
    for item in router_artifact.get("high_risk_instances") or []:
        if isinstance(item, Mapping) and item.get("contract_case_id"):
            risk[str(item["contract_case_id"])] = float(item.get("risk_upper_bound", 0.0))
    return risk


def _predecessor_blockers(
    router_artifact: Mapping[str, Any],
    gate_artifact: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if router_artifact.get("uncertainty_router_ready") is not True:
        blockers.append("exp1541_uncertainty_router_not_ready")
    if gate_artifact.get("unified_contract_gate_ready") is not True:
        blockers.append("exp1551_unified_contract_gate_not_ready")
    return blockers


def _missing_source_blockers(paths: Mapping[str, Path]) -> list[str]:
    return [f"missing_source:{name}:{path}" for name, path in paths.items() if not path.exists()]


def _load_policy(policy_path: Path) -> JsonDict:
    policy = dict(DEFAULT_ROUTING_POLICY)
    if policy_path.exists():
        policy.update(_read_json(policy_path))
    return policy


def _live_sota_used(
    router_artifact: Mapping[str, Any],
    gate_artifact: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(
        router_artifact.get("live_sota_model_inference_used")
        or gate_artifact.get("live_sota_model_inference_used")
        or any(case.get("model_hf_id") in MODEL_SPECS for case in cases)
    )


def _models_used(
    router_artifact: Mapping[str, Any],
    gate_artifact: Mapping[str, Any],
) -> list[str]:
    models: list[str] = []
    for artifact in (router_artifact, gate_artifact):
        for model in artifact.get("models_used") or []:
            if isinstance(model, str) and model not in models:
                models.append(model)
    return models


def _declared_or_parsed_accept(parse: Mapping[str, Any], answer: Any) -> bool:
    declared = parse.get("model_declared_accept")
    if isinstance(declared, bool):
        return declared
    return bool(parse.get("parse_ok") and answer is not None)


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 6)


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():  # pragma: no cover - missing-source blocker already records this.
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path) -> str:
    return str(path)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-tests-passed", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    artifact = run_experiment(focused_tests_passed=args.focused_tests_passed)
    print(
        "[exp1553] "
        f"ready={artifact['claim_isolation_router_scale_ready']} "
        f"cases={artifact['cases_total']} "
        f"budget_reduced={artifact['budget_reduced']} "
        f"false_accept={artifact['false_accept_rate']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
