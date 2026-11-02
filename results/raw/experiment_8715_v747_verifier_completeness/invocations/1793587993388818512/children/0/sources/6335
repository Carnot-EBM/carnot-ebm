"""Exp 1557 Weaver-style verification-compute router.

Spec: REQ-VERIFY-1557, SCENARIO-VERIFY-1557.

The router spends cheap verification checks first, then decides whether a case
can use a single deterministic source validator or must fall back to every
available deterministic validator.  The distinction matters because soft
signals are useful triage evidence, but they are not proof.  A candidate is
accepted only after at least one deterministic validator accepts it, and the
reported missed-failure metric compares the routed decision with an
always-run-all-deterministic baseline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
MILESTONE = "20260508"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1557_weaver_verification_compute_router.json")
DEFAULT_MANIFEST_PATH = Path("results/weaver_verification_compute_router_1557.jsonl")
DEFAULT_ROUTER_POLICY_PATH = Path("results/weaver_verification_compute_router_policy_1557.json")
DEFAULT_SATQUEST_ARTIFACT_PATH = Path(
    "results/experiment_1550_satquest_sota_reeval_zero_false_accepts.json"
)
DEFAULT_UNIFIED_GATE_ARTIFACT_PATH = Path(
    "results/experiment_1551_automata_sat_unified_contract_gate.json"
)
DEFAULT_CLAIM_ROUTER_ARTIFACT_PATH = Path(
    "results/experiment_1553_claim_isolation_router_scale_v3.json"
)
DEFAULT_TELEMETRY_ARTIFACT_PATH = Path(
    "results/experiment_1556_arm_ebm_logprob_telemetry_repair.json"
)
DEFAULT_BEAVER_ARTIFACT_PATH = Path("results/experiment_1537_beaver_prefix_bound_contracts_v3.json")
DEFAULT_SATQUEST_MANIFEST_PATH = Path("results/satquest_sota_reeval_zero_false_accepts_1550.jsonl")
DEFAULT_UNIFIED_GATE_MANIFEST_PATH = Path("results/automata_sat_unified_contract_gate_1551.jsonl")
DEFAULT_CLAIM_ROUTER_MANIFEST_PATH = Path("results/claim_isolation_router_scale_1553.jsonl")
DEFAULT_TELEMETRY_DIAGNOSTIC_PATH = Path("results/arm_ebm_logprob_telemetry_diagnostic_1556.jsonl")
DEFAULT_BEAVER_MANIFEST_PATH = Path("results/beaver_prefix_bound_contracts_1537.jsonl")
ROUTER_MODULE_PATH = "python/carnot/verify/verification_compute_router.py"

WEAK_VERIFIER_COSTS: JsonDict = {
    "automata_format_validity": 1,
    "beaver_prefix_risk": 1,
    "claim_router_uncertainty": 1,
    "energy_diagnostic": 1,
    "telemetry_logprob_diagnostic": 1,
}
DETERMINISTIC_VALIDATOR_COSTS: JsonDict = {
    "runtime_contract_replay": 5,
    "sat_solver": 8,
    "product_line_oracle": 7,
    "residual_drift_replay": 6,
    "unified_contract_gate": 6,
}
PRIMARY_VALIDATOR_BY_SOURCE: JsonDict = {
    "runtime_contract": "runtime_contract_replay",
    "satquest": "sat_solver",
    "product_line": "product_line_oracle",
    "residual_drift": "residual_drift_replay",
}
DEFAULT_ROUTER_POLICY: JsonDict = {
    "claim_uncertainty_threshold": 0.5,
    "beaver_prefix_risk_threshold": 0.75,
    "telemetry_routing_score_threshold": 10.0,
    "energy_high_risk_threshold": 50.0,
    "route_on_validator_disagreement": True,
    "route_on_parse_failure": True,
    "route_on_automata_reject": True,
}
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "verification_compute_router_ready",
    "candidate_selection_cases",
    "weak_verifiers_used",
    "deterministic_validators_used",
    "soft_signals_used_for_routing_only",
    "verification_cost_baseline",
    "verification_cost_router",
    "verification_cost_delta",
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


def write_in_progress_artifact(output_path: Path | str = DEFAULT_ARTIFACT_PATH) -> JsonDict:
    """REQ-VERIFY-1557: write the durable bootstrap result before source loading."""

    artifact = _artifact_from_summary(
        status="in_progress",
        summary=_empty_summary(),
        router_policy_path=Path(DEFAULT_ROUTER_POLICY_PATH),
        focused_tests_passed=False,
        blockers=["experiment_1557_weaver_verification_compute_router_in_progress"],
        manifest_path=Path(DEFAULT_MANIFEST_PATH),
        predecessor_ready={},
        optional_artifacts_loaded={},
    )
    _write_json(Path(output_path), artifact)
    return artifact


def route_candidate(
    candidate: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None = None,
) -> JsonDict:
    """REQ-VERIFY-1557: choose cheap validation or deterministic fallback."""

    active_policy = dict(DEFAULT_ROUTER_POLICY if policy is None else policy)
    weak_signals = _mapping(candidate.get("weak_signals"))
    deterministic_outcomes = _bool_mapping(candidate.get("deterministic_outcomes"))
    routing_reasons, risk_score = _routing_reasons(weak_signals, active_policy)
    selected_validators = (
        list(deterministic_outcomes)
        if routing_reasons
        else [_primary_validator(candidate, deterministic_outcomes)]
    )
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "route": "deterministic_fallback" if routing_reasons else "cheap_primary_validator",
        "routing_reasons": routing_reasons or ["low_risk_primary_validator"],
        "risk_score": round(risk_score, 6),
        "weak_verifiers": _weak_verifiers_for_signals(weak_signals),
        "selected_deterministic_validators": selected_validators,
    }


def evaluate_routing(
    candidates: Sequence[Mapping[str, Any]],
    *,
    focused_tests_passed: bool,
    policy: Mapping[str, Any] | None = None,
) -> JsonDict:
    """SCENARIO-VERIFY-1557: compare routed cost with the all-validator baseline."""

    rows: list[JsonDict] = []
    weak_used: set[str] = set()
    deterministic_used: set[str] = set()
    soft_signals_used: set[str] = set()
    verification_cost_baseline = 0
    verification_cost_router = 0
    baseline_rejects = 0
    missed_failures = 0

    for candidate in candidates:
        deterministic_outcomes = _bool_mapping(candidate.get("deterministic_outcomes"))
        soft_signals = _mapping(candidate.get("soft_signals"))
        decision = route_candidate(candidate, policy=policy)
        selected_validators = list(decision["selected_deterministic_validators"])
        weak_verifiers = list(decision["weak_verifiers"])
        baseline_accept = bool(deterministic_outcomes) and all(deterministic_outcomes.values())
        final_accept = bool(selected_validators) and all(
            deterministic_outcomes.get(validator, False) for validator in selected_validators
        )
        soft_signal_accept = _soft_accept(soft_signals)
        missed_failure = bool(final_accept and not baseline_accept)
        verification_cost_baseline += sum(
            _deterministic_cost(validator) for validator in deterministic_outcomes
        )
        verification_cost_router += sum(_weak_cost(verifier) for verifier in weak_verifiers)
        verification_cost_router += sum(
            _deterministic_cost(validator) for validator in selected_validators
        )
        baseline_rejects += int(not baseline_accept)
        missed_failures += int(missed_failure)
        weak_used.update(weak_verifiers)
        deterministic_used.update(selected_validators)
        soft_signals_used.update(str(key) for key in soft_signals)
        rows.append(
            {
                "row_type": "verification_compute_router_case",
                "spec": ["REQ-VERIFY-1557", "SCENARIO-VERIFY-1557"],
                "candidate_id": str(candidate["candidate_id"]),
                "case_id": str(candidate.get("case_id") or candidate["candidate_id"]),
                "source_kind": str(candidate.get("source_kind") or "unknown"),
                "source_family": str(
                    candidate.get("source_family") or candidate.get("source_kind") or "unknown"
                ),
                "route": decision["route"],
                "routing_reasons": list(decision["routing_reasons"]),
                "weak_verifiers": weak_verifiers,
                "selected_deterministic_validators": selected_validators,
                "baseline_accept": baseline_accept,
                "final_accept": final_accept,
                "soft_signal_accept": soft_signal_accept,
                "soft_signal_overrode_validator": False,
                "missed_failure": missed_failure,
                "deterministic_outcomes": deterministic_outcomes,
            }
        )

    false_accept_rate = round(missed_failures / baseline_rejects, 6) if baseline_rejects else 0.0
    summary = {
        "candidate_count": len(candidates),
        "candidate_selection_cases": [_selection_case(row) for row in rows],
        "weak_verifiers_used": sorted(weak_used),
        "deterministic_validators_used": sorted(deterministic_used),
        "soft_signals_used_for_routing_only": sorted(soft_signals_used),
        "verification_cost_baseline": verification_cost_baseline,
        "verification_cost_router": verification_cost_router,
        "verification_cost_delta": verification_cost_router - verification_cost_baseline,
        "false_accept_rate": false_accept_rate,
        "missed_failure_count": missed_failures,
        "focused_tests_passed": bool(focused_tests_passed),
    }
    rows.append({"row_type": "summary", **summary})
    return {"rows": rows, "summary": summary}


def build_candidate_set(
    *,
    satquest_rows: Sequence[Mapping[str, Any]],
    unified_gate_rows: Sequence[Mapping[str, Any]],
    claim_router_rows: Sequence[Mapping[str, Any]] = (),
    telemetry_rows: Sequence[Mapping[str, Any]] = (),
    beaver_rows: Sequence[Mapping[str, Any]] = (),
    limit: int = 32,
) -> list[JsonDict]:
    """REQ-VERIFY-1557: build a bounded mixed set from available manifests."""

    candidates: list[JsonDict] = []
    seen: set[str] = set()
    telemetry_by_case = _rows_by_case(telemetry_rows, "case_id")
    gate_by_case = _rows_by_case(unified_gate_rows, "case_id")
    beaver_risk_by_case = _beaver_risk_by_case(beaver_rows)

    for row in claim_router_rows:
        if row.get("row_type") != "router_scale_case":
            continue
        candidate = _candidate_from_claim_router_row(
            row,
            telemetry_by_case=telemetry_by_case,
            gate_by_case=gate_by_case,
            beaver_risk_by_case=beaver_risk_by_case,
        )
        _append_candidate(candidates, seen, candidate, limit)

    for row in satquest_rows:
        if not row.get("case_id"):
            continue
        candidate = _candidate_from_satquest_row(
            row,
            telemetry_by_case=telemetry_by_case,
            gate_by_case=gate_by_case,
        )
        _append_candidate(candidates, seen, candidate, limit)

    for row in unified_gate_rows:
        if row.get("row_type") != "gate_case":
            continue
        candidate = _candidate_from_gate_row(row, beaver_risk_by_case=beaver_risk_by_case)
        _append_candidate(candidates, seen, candidate, limit)

    return candidates[: max(0, int(limit))]


def run_experiment(
    *,
    project_root: Path | str = ".",
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    router_policy_path: Path | str = DEFAULT_ROUTER_POLICY_PATH,
    satquest_artifact_path: Path | str = DEFAULT_SATQUEST_ARTIFACT_PATH,
    unified_gate_artifact_path: Path | str = DEFAULT_UNIFIED_GATE_ARTIFACT_PATH,
    claim_router_artifact_path: Path | str = DEFAULT_CLAIM_ROUTER_ARTIFACT_PATH,
    telemetry_artifact_path: Path | str = DEFAULT_TELEMETRY_ARTIFACT_PATH,
    beaver_artifact_path: Path | str = DEFAULT_BEAVER_ARTIFACT_PATH,
    satquest_manifest_path: Path | str = DEFAULT_SATQUEST_MANIFEST_PATH,
    unified_gate_manifest_path: Path | str = DEFAULT_UNIFIED_GATE_MANIFEST_PATH,
    claim_router_manifest_path: Path | str = DEFAULT_CLAIM_ROUTER_MANIFEST_PATH,
    telemetry_diagnostic_path: Path | str = DEFAULT_TELEMETRY_DIAGNOSTIC_PATH,
    beaver_manifest_path: Path | str = DEFAULT_BEAVER_MANIFEST_PATH,
    focused_tests_passed: bool = False,
    case_limit: int = 32,
) -> JsonDict:
    """Run Exp 1557 from checked-in artifacts and write the terminal report."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    manifest = _resolve(root, manifest_path)
    policy_path = _resolve(root, router_policy_path)
    write_in_progress_artifact(output)

    satquest_artifact_file = _resolve(root, satquest_artifact_path)
    unified_gate_artifact_file = _resolve(root, unified_gate_artifact_path)
    claim_router_artifact_file = _resolve(root, claim_router_artifact_path)
    telemetry_artifact_file = _resolve(root, telemetry_artifact_path)
    beaver_artifact_file = _resolve(root, beaver_artifact_path)
    satquest_artifact = _read_json(satquest_artifact_file)
    unified_gate_artifact = _read_json(unified_gate_artifact_file)
    claim_router_artifact = _read_json(claim_router_artifact_file)
    telemetry_artifact = _read_json(telemetry_artifact_file)
    beaver_artifact = _read_json(beaver_artifact_file)
    blockers = _required_source_blockers(
        satquest_artifact_file, satquest_artifact, unified_gate_artifact_file, unified_gate_artifact
    )
    if not focused_tests_passed:
        blockers.append("focused_tests_not_passed")

    policy = _load_policy(policy_path)
    candidates = build_candidate_set(
        satquest_rows=_read_jsonl(_resolve(root, satquest_manifest_path)),
        unified_gate_rows=_read_jsonl(_resolve(root, unified_gate_manifest_path)),
        claim_router_rows=_read_jsonl(_resolve(root, claim_router_manifest_path)),
        telemetry_rows=_read_jsonl(_resolve(root, telemetry_diagnostic_path)),
        beaver_rows=_read_jsonl(_resolve(root, beaver_manifest_path)),
        limit=case_limit,
    )
    if not candidates:
        blockers.append("no_candidate_selection_cases")

    evaluation = evaluate_routing(
        candidates, focused_tests_passed=focused_tests_passed, policy=policy
    )
    _write_json(policy_path, policy)
    _write_jsonl(manifest, evaluation["rows"])
    artifact = _artifact_from_summary(
        status="complete" if candidates else "blocked",
        summary=evaluation["summary"],
        router_policy_path=policy_path,
        focused_tests_passed=focused_tests_passed,
        blockers=blockers,
        manifest_path=manifest,
        predecessor_ready={
            "satquest_sota_reeval_ready": satquest_artifact.get("satquest_sota_reeval_ready")
            is True,
            "unified_contract_gate_ready": unified_gate_artifact.get("unified_contract_gate_ready")
            is True,
        },
        optional_artifacts_loaded={
            "claim_router_loaded": bool(claim_router_artifact),
            "telemetry_loaded": bool(telemetry_artifact),
            "beaver_loaded": bool(beaver_artifact),
        },
    )
    validate_artifact(artifact)
    _write_json(output, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Enforce the Exp1557 terminal schema and authority invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use an allowed terminal prefix")
    if artifact["verification_compute_router_ready"]:
        if artifact["focused_tests_passed"] is not True:
            raise AssertionError("ready router requires focused tests")
        if artifact["false_accept_rate"] != 0.0:
            raise AssertionError("ready router requires zero false accepts")
        if artifact["missed_failure_count"] != 0:
            raise AssertionError("ready router requires zero missed failures")
        if artifact["verification_cost_delta"] >= 0:
            raise AssertionError("ready router requires lower routed cost")
        if not artifact["candidate_selection_cases"]:
            raise AssertionError("ready router requires candidate cases")
        if not artifact["deterministic_validators_used"]:
            raise AssertionError("ready router requires deterministic validators")


def _artifact_from_summary(
    *,
    status: str,
    summary: Mapping[str, Any],
    router_policy_path: Path,
    focused_tests_passed: bool,
    blockers: Sequence[str],
    manifest_path: Path,
    predecessor_ready: Mapping[str, bool],
    optional_artifacts_loaded: Mapping[str, bool],
) -> JsonDict:
    ready = bool(
        summary.get("candidate_count", 0) > 0
        and focused_tests_passed
        and not blockers
        and predecessor_ready.get("satquest_sota_reeval_ready") is True
        and predecessor_ready.get("unified_contract_gate_ready") is True
        and summary.get("verification_cost_delta", 0) < 0
        and summary.get("false_accept_rate", 0.0) == 0.0
        and summary.get("missed_failure_count", 0) == 0
        and summary.get("deterministic_validators_used")
    )
    verdict = (
        "complete: weaver_verification_compute_router_ready"
        if ready
        else "complete: weaver_verification_compute_router_completed_with_blockers"
    )
    return {
        "status": status,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "schema_version": 1,
        "verification_compute_router_ready": ready,
        "candidate_selection_cases": list(summary.get("candidate_selection_cases") or []),
        "weak_verifiers_used": list(summary.get("weak_verifiers_used") or []),
        "deterministic_validators_used": list(summary.get("deterministic_validators_used") or []),
        "soft_signals_used_for_routing_only": list(
            summary.get("soft_signals_used_for_routing_only") or []
        ),
        "verification_cost_baseline": int(summary.get("verification_cost_baseline", 0)),
        "verification_cost_router": int(summary.get("verification_cost_router", 0)),
        "verification_cost_delta": int(summary.get("verification_cost_delta", 0)),
        "false_accept_rate": float(summary.get("false_accept_rate", 0.0)),
        "missed_failure_count": int(summary.get("missed_failure_count", 0)),
        "router_policy_path": _display_path(router_policy_path),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": verdict,
        "blockers": list(blockers),
        "router_module_path": ROUTER_MODULE_PATH,
        "router_manifest_path": _display_path(manifest_path),
        **dict(predecessor_ready),
        **dict(optional_artifacts_loaded),
    }


def _empty_summary() -> JsonDict:
    return {
        "candidate_count": 0,
        "candidate_selection_cases": [],
        "weak_verifiers_used": [],
        "deterministic_validators_used": [],
        "soft_signals_used_for_routing_only": [],
        "verification_cost_baseline": 0,
        "verification_cost_router": 0,
        "verification_cost_delta": 0,
        "false_accept_rate": 0.0,
        "missed_failure_count": 0,
    }


def _routing_reasons(
    weak_signals: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[list[str], float]:
    scores = [
        _number(weak_signals.get("claim_uncertainty_score")) or 0.0,
        _number(weak_signals.get("beaver_prefix_risk")) or 0.0,
        (_number(weak_signals.get("telemetry_routing_score")) or 0.0)
        / max(float(policy["telemetry_routing_score_threshold"]), 1.0),
        (_number(weak_signals.get("carnot_energy_score")) or 0.0)
        / max(float(policy["energy_high_risk_threshold"]), 1.0),
    ]
    reasons: list[str] = []
    if scores[0] >= float(policy["claim_uncertainty_threshold"]):
        reasons.append("claim_uncertainty")
    if scores[1] >= float(policy["beaver_prefix_risk_threshold"]):
        reasons.append("beaver_prefix_risk")
    if _number(weak_signals.get("telemetry_routing_score")) is not None and (
        float(weak_signals["telemetry_routing_score"])
        >= float(policy["telemetry_routing_score_threshold"])
    ):
        reasons.append("telemetry_routing_score")
    if _number(weak_signals.get("carnot_energy_score")) is not None and (
        float(weak_signals["carnot_energy_score"]) >= float(policy["energy_high_risk_threshold"])
    ):
        reasons.append("energy_high_risk")
    if policy.get("route_on_validator_disagreement") and weak_signals.get("validator_disagreement"):
        reasons.append("validator_disagreement")
    if policy.get("route_on_parse_failure") and weak_signals.get("parse_failure"):
        reasons.append("parse_failure")
    if policy.get("route_on_automata_reject") and weak_signals.get("automata_reject"):
        reasons.append("automata_reject")
    return _unique(reasons), max(scores)


def _weak_verifiers_for_signals(weak_signals: Mapping[str, Any]) -> list[str]:
    verifiers = ["automata_format_validity", "claim_router_uncertainty"]
    if "beaver_prefix_risk" in weak_signals:
        verifiers.append("beaver_prefix_risk")
    if "carnot_energy_score" in weak_signals:
        verifiers.append("energy_diagnostic")
    if "telemetry_routing_score" in weak_signals:
        verifiers.append("telemetry_logprob_diagnostic")
    return _unique(verifiers)


def _primary_validator(candidate: Mapping[str, Any], outcomes: Mapping[str, bool]) -> str:
    preferred = PRIMARY_VALIDATOR_BY_SOURCE.get(str(candidate.get("source_kind") or ""))
    if preferred in outcomes:
        return preferred
    return next(iter(outcomes), "unified_contract_gate")


def _candidate_from_claim_router_row(
    row: Mapping[str, Any],
    *,
    telemetry_by_case: Mapping[str, Mapping[str, Any]],
    gate_by_case: Mapping[str, Mapping[str, Any]],
    beaver_risk_by_case: Mapping[str, float],
) -> JsonDict:
    case_id = str(row["case_id"])
    source_kind = str(row.get("source_kind") or "runtime_contract")
    deterministic_accept = bool(row.get("deterministic_accept"))
    telemetry = _mapping(telemetry_by_case.get(case_id))
    gate = _mapping(gate_by_case.get(case_id))
    primary = PRIMARY_VALIDATOR_BY_SOURCE[source_kind]
    deterministic_outcomes = {
        primary: deterministic_accept,
        "unified_contract_gate": bool(row.get("unified_gate_checked", True))
        and deterministic_accept,
    }
    reasons = [str(reason) for reason in row.get("routing_reasons") or []]
    weak_signals = _drop_none(
        {
            "claim_uncertainty_score": 1.0 if "uncertainty" in reasons else 0.0,
            "validator_disagreement": bool(
                "validator_disagreement" in reasons
                or row.get("full_context_accept") != row.get("deterministic_accept")
            ),
            "parse_failure": bool("parse_failure" in reasons),
            "automata_reject": _gate_has_automata_reject(gate),
            "beaver_prefix_risk": beaver_risk_by_case.get(case_id),
            "telemetry_routing_score": _number(telemetry.get("routing_score")),
            "carnot_energy_score": _number(telemetry.get("carnot_energy_score")),
        }
    )
    return {
        "candidate_id": f"{source_kind}:{case_id}",
        "case_id": case_id,
        "source_kind": source_kind,
        "source_family": str(row.get("source_family") or source_kind),
        "weak_signals": weak_signals,
        "soft_signals": _drop_none(
            {
                "full_context_accept": row.get("full_context_accept"),
                "claim_isolated_accept": row.get("claim_isolated_accept"),
                "model_declared_accept": telemetry.get("model_declared_accept"),
                "mean_logprob": telemetry.get("mean_logprob"),
            }
        ),
        "deterministic_outcomes": deterministic_outcomes,
    }


def _candidate_from_satquest_row(
    row: Mapping[str, Any],
    *,
    telemetry_by_case: Mapping[str, Mapping[str, Any]],
    gate_by_case: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    case_id = str(row["case_id"])
    baseline = _mapping(row.get("baseline"))
    parse = _mapping(row.get("parse_result"))
    telemetry = _mapping(telemetry_by_case.get(case_id))
    gate = _mapping(gate_by_case.get(case_id))
    deterministic_accept = bool(baseline.get("correct"))
    return {
        "candidate_id": f"satquest:{case_id}",
        "case_id": case_id,
        "source_kind": "satquest",
        "source_family": str(row.get("family") or "satquest"),
        "weak_signals": _drop_none(
            {
                "claim_uncertainty_score": 0.0,
                "validator_disagreement": bool(
                    _mapping(row.get("verifier")).get("self_verifier_false_accept")
                ),
                "parse_failure": not bool(parse.get("parse_ok")),
                "automata_reject": _gate_has_automata_reject(gate),
                "telemetry_routing_score": _number(telemetry.get("routing_score")),
                "carnot_energy_score": _number(telemetry.get("carnot_energy_score"))
                if "carnot_energy_score" in telemetry
                else _number(baseline.get("energy")),
            }
        ),
        "soft_signals": _drop_none(
            {
                "model_declared_accept": parse.get("model_declared_accept"),
                "mean_logprob": telemetry.get("mean_logprob"),
            }
        ),
        "deterministic_outcomes": {
            "sat_solver": deterministic_accept,
            "unified_contract_gate": bool(gate.get("final_accept", deterministic_accept))
            and deterministic_accept,
        },
    }


def _candidate_from_gate_row(
    row: Mapping[str, Any],
    *,
    beaver_risk_by_case: Mapping[str, float],
) -> JsonDict:
    case_id = str(row["case_id"])
    source_kind = _source_kind_from_gate_family(str(row.get("source_family") or "runtime_contract"))
    primary = PRIMARY_VALIDATOR_BY_SOURCE[source_kind]
    final_accept = bool(row.get("final_accept"))
    return {
        "candidate_id": f"{source_kind}:{case_id}",
        "case_id": case_id,
        "source_kind": source_kind,
        "source_family": str(row.get("source_family") or source_kind),
        "weak_signals": _drop_none(
            {
                "claim_uncertainty_score": 0.0,
                "automata_reject": _gate_has_automata_reject(row),
                "beaver_prefix_risk": beaver_risk_by_case.get(case_id),
            }
        ),
        "soft_signals": _drop_none({"soft_accept": row.get("soft_accept")}),
        "deterministic_outcomes": {primary: final_accept, "unified_contract_gate": final_accept},
    }


def _source_kind_from_gate_family(source_family: str) -> str:
    if source_family in PRIMARY_VALIDATOR_BY_SOURCE:
        return source_family
    return "runtime_contract"


def _gate_has_automata_reject(row: Mapping[str, Any]) -> bool:
    return any(
        stage.get("stage") == "automata_mask" and stage.get("passed") is False
        for stage in row.get("stages") or []
        if isinstance(stage, Mapping)
    )


def _append_candidate(
    candidates: list[JsonDict],
    seen: set[str],
    candidate: Mapping[str, Any],
    limit: int,
) -> None:
    candidate_id = str(candidate["candidate_id"])
    if len(candidates) >= int(limit) or candidate_id in seen:
        return
    seen.add(candidate_id)
    candidates.append(dict(candidate))


def _selection_case(row: Mapping[str, Any]) -> JsonDict:
    return {
        "candidate_id": row["candidate_id"],
        "source_kind": row["source_kind"],
        "route": row["route"],
        "routing_reasons": row["routing_reasons"],
        "selected_deterministic_validators": row["selected_deterministic_validators"],
        "baseline_accept": row["baseline_accept"],
        "final_accept": row["final_accept"],
        "missed_failure": row["missed_failure"],
    }


def _required_source_blockers(
    satquest_path: Path,
    satquest_artifact: Mapping[str, Any],
    gate_path: Path,
    gate_artifact: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not satquest_artifact:
        blockers.append(f"missing_required_artifact:{_display_path(satquest_path)}")
    elif satquest_artifact.get("satquest_sota_reeval_ready") is not True:
        blockers.append("satquest_sota_reeval_not_ready")
    if not gate_artifact:
        blockers.append(f"missing_required_artifact:{_display_path(gate_path)}")
    elif gate_artifact.get("unified_contract_gate_ready") is not True:
        blockers.append("unified_contract_gate_not_ready")
    return blockers


def _rows_by_case(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, JsonDict]:
    return {str(row[key]): dict(row) for row in rows if row.get(key)}


def _beaver_risk_by_case(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    risk: dict[str, float] = {}
    for row in rows:
        case_id = row.get("contract_case_id")
        value = _number(row.get("unsafe_upper_bound"))
        if case_id and value is not None:
            risk[str(case_id)] = max(risk.get(str(case_id), 0.0), value)
    return risk


def _soft_accept(soft_signals: Mapping[str, Any]) -> bool:
    return any(value is True for value in soft_signals.values())


def _deterministic_cost(validator: str) -> int:
    return int(DETERMINISTIC_VALIDATOR_COSTS.get(validator, 6))


def _weak_cost(verifier: str) -> int:
    return int(WEAK_VERIFIER_COSTS.get(verifier, 1))


def _bool_mapping(value: Any) -> dict[str, bool]:
    return {str(key): bool(item) for key, item in _mapping(value).items()}


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _drop_none(values: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in values.items() if value is not None}


def _unique(values: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _load_policy(path: Path) -> JsonDict:
    existing = _read_json(path)
    return dict(existing if existing else DEFAULT_ROUTER_POLICY)


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    return [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _display_path(path: Path) -> str:
    return path.as_posix()


__all__ = [
    "DEFAULT_ROUTER_POLICY",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_candidate_set",
    "evaluate_routing",
    "route_candidate",
    "run_experiment",
    "validate_artifact",
    "write_in_progress_artifact",
]
