"""Build the Exp 1315 CerCE non-forgetting audit artifact.

Spec: REQ-LEARN-1315, SCENARIO-LEARN-1315.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_FILE = "experiment_1315_continuous_self_learning_cerce_nonforgetting_audit.json"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE

EXPERIMENT = "1315_continuous_self_learning_cerce_nonforgetting_audit"
SCHEMA = "continuous_self_learning_cerce_nonforgetting_audit_v1"
RUN_DATE = "20260505"
EXP1302_FILE = "experiment_1302_skill_graph_promotion_demotion_v2.json"
EXP1303_FILE = "experiment_1303_querybandits_ngc_online_memory_policy.json"
EXP1288_FILE = "experiment_1288_interwhen_dvi_verifier_feedback_replay.json"
SOURCE_FILES = (EXP1302_FILE, EXP1303_FILE, EXP1288_FILE)
SOURCE_ARTIFACTS = [f"results/{name}" for name in SOURCE_FILES]

AUDIT_DECISIONS = ("promote", "demote", "rewrite", "abstain", "expire")
PROMOTE_MIN_SUPPORT = 5
SUPPORTED_VERDICTS = {
    "in_progress",
    "blocked_missing_inputs",
    "cerce_nonforgetting_preserved_improved_non_headline",
    "cerce_nonforgetting_preserved_neutral_non_headline",
    "cerce_nonforgetting_preserved_regressed_non_headline",
    "cerce_nonforgetting_regression_non_headline",
}
REQUIRED_FIELDS = {
    "status",
    "nonforgetting_certificate_rate",
    "memory_regression_count",
    "self_learning_delta_overall",
    "lagrangian_violation_penalty",
    "accepted_violation_delta",
    "promoted_memory_count",
    "demoted_memory_count",
    "headline_result_allowed",
    "honest_verdict",
}


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def _zero_decision_counts() -> dict[str, int]:
    return {decision: 0 for decision in AUDIT_DECISIONS}


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1315-1: write the bootstrap artifact before loading inputs."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "source_artifacts": SOURCE_ARTIFACTS,
            "status": "in_progress",
            "nonforgetting_certificate_rate": 0.0,
            "memory_regression_count": 0,
            "self_learning_delta_overall": 0.0,
            "lagrangian_violation_penalty": 0.0,
            "accepted_violation_delta": 0.0,
            "promoted_memory_count": 0,
            "demoted_memory_count": 0,
            "headline_result_allowed": False,
            "honest_verdict": "in_progress",
            "audit_decision_counts": _zero_decision_counts(),
        },
    )


def write_terminal_blocker(
    out_path: Path | str,
    missing_inputs: Sequence[str],
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1315-2: write an honest terminal blocker when inputs are absent."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "source_artifacts": SOURCE_ARTIFACTS,
            "status": "blocked",
            "missing_inputs": list(missing_inputs),
            "nonforgetting_certificate_rate": 0.0,
            "memory_regression_count": 0,
            "self_learning_delta_overall": 0.0,
            "lagrangian_violation_penalty": 0.0,
            "accepted_violation_delta": 0.0,
            "promoted_memory_count": 0,
            "demoted_memory_count": 0,
            "headline_result_allowed": False,
            "honest_verdict": "blocked_missing_inputs",
            "audit_decision_counts": _zero_decision_counts(),
        },
    )


def load_source_artifacts(results_dir: Path | str = DEFAULT_RESULTS_DIR) -> tuple[dict[str, Any], list[str]]:
    """Load the three required source artifacts and report absent inputs."""

    results_path = Path(results_dir)
    payloads: dict[str, Any] = {}
    missing: list[str] = []
    for filename in SOURCE_FILES:
        path = results_path / filename
        if path.exists():
            payloads[filename] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(f"results/{filename}")
    return payloads, missing


def _support(candidate: Mapping[str, Any] | None) -> int:
    if not candidate:
        return 0
    evidence = candidate.get("replay_evidence")
    evidence_support = int(evidence.get("support") or 0) if isinstance(evidence, Mapping) else 0
    return evidence_support or int(candidate.get("support") or 0)


def _candidate_lookup(candidates: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for candidate in candidates:
        key = (
            str(candidate.get("constraint_pattern") or ""),
            str(candidate.get("selected_decision") or ""),
            str(candidate.get("verifier_result") or ""),
        )
        if key not in lookup or _support(candidate) > _support(lookup[key]):
            lookup[key] = candidate
    return lookup


def _fallback_pattern(
    records: Sequence[Mapping[str, Any]],
    target_decision: str,
    verifier_result: str,
    position: int,
) -> str:
    matching = [
        record
        for record in records
        if str(record.get("selected_decision") or "") == target_decision
        and str(record.get("verifier_result") or "") == verifier_result
    ]
    if not matching:
        return "unknown"
    return str(matching[position % len(matching)].get("constraint_pattern") or "unknown")


def _case_candidate(
    lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
    pattern: str,
    target_decision: str,
    verifier_result: str,
) -> Mapping[str, Any] | None:
    return lookup.get((pattern, target_decision, verifier_result))


def _cohort(target_decision: str, verifier_result: str) -> str:
    if target_decision == "accept" and verifier_result == "passed":
        return "old_verified"
    if target_decision == "repair" and verifier_result == "failed":
        return "101_improved"
    return "adversarial_unknown"


def _replay_case(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    *,
    pattern: str,
    target_decision: str,
    verifier_result: str,
    position: int,
) -> dict[str, Any]:
    return {
        "case_id": str(row.get("case_id") or f"case-{position}"),
        "chronological_index": int(row.get("chronological_index") or position),
        "cohort": _cohort(target_decision, verifier_result),
        "constraint_pattern": pattern,
        "target_decision": target_decision,
        "verifier_result": verifier_result,
        "memory_routing_decision": str(candidate.get("memory_routing_decision") if candidate else "missing"),
        "memory_selected_decision": str(candidate.get("selected_decision") if candidate else "missing"),
        "memory_support": _support(candidate),
    }


def build_replay_set(
    exp1302_payload: Mapping[str, Any],
    exp1303_payload: Mapping[str, Any],
    exp1288_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """REQ-LEARN-1315-3: build old, improved, and adversarial replay cases."""

    candidates = [row for row in exp1302_payload.get("skill_graph_candidates", []) if isinstance(row, Mapping)]
    lookup = _candidate_lookup(candidates)
    records = [row for row in exp1288_payload.get("clause_prediction_records", []) if isinstance(row, Mapping)]
    replay_cases: list[dict[str, Any]] = []
    for position, row in enumerate(exp1288_payload.get("replay_slices", [])):
        if not isinstance(row, Mapping):
            continue
        verifier_result = str(row.get("verifier_result") or "unknown")
        target_decision = str(
            row.get("target_decision") or ("repair" if verifier_result == "failed" else "accept")
        )
        pattern = str(
            row.get("constraint_pattern")
            or _fallback_pattern(records, target_decision, verifier_result, position)
        )
        candidate = _case_candidate(lookup, pattern, target_decision, verifier_result)
        replay_cases.append(
            _replay_case(
                row,
                candidate,
                pattern=pattern,
                target_decision=target_decision,
                verifier_result=verifier_result,
                position=position,
            )
        )

    policy_counts = exp1303_payload.get("selected_policy_counts")
    rewrite_count = int(policy_counts.get("rewrite_repair_prompt") or 0) if isinstance(policy_counts, Mapping) else 0
    has_rewrite_case = any(
        case["cohort"] == "101_improved" and case["memory_routing_decision"] == "missing"
        for case in replay_cases
    )
    if rewrite_count > 0 and not has_rewrite_case:
        replay_cases.append(
            {
                "case_id": "exp1303-rewrite-probe",
                "chronological_index": len(replay_cases),
                "cohort": "101_improved",
                "constraint_pattern": "policy:rewrite_repair_prompt",
                "target_decision": "repair",
                "verifier_result": "failed",
                "memory_routing_decision": "missing",
                "memory_selected_decision": "missing",
                "memory_support": 0,
            }
        )

    for suffix, pattern in (
        ("unknown", "unknown:out_of_distribution"),
        ("adversarial", "unknown:adversarial_violation"),
    ):
        replay_cases.append(
            {
                "case_id": f"exp1315-{suffix}-abstain-probe",
                "chronological_index": len(replay_cases),
                "cohort": "adversarial_unknown",
                "constraint_pattern": pattern,
                "target_decision": "abstain",
                "verifier_result": "unknown",
                "memory_routing_decision": "missing",
                "memory_selected_decision": "missing",
                "memory_support": 0,
            }
        )
    return replay_cases


def _policy_decision(case: Mapping[str, Any]) -> tuple[str, str]:
    cohort = str(case["cohort"])
    target = str(case["target_decision"])
    routing = str(case["memory_routing_decision"])
    support = int(case["memory_support"])
    if cohort == "adversarial_unknown":
        return "abstain", "abstain"
    if target == "accept":
        if routing in {"promote", "demote", "expire"}:
            return routing, "accept"
        return "abstain", "accept"
    if target == "repair" and routing == "promote" and support >= PROMOTE_MIN_SUPPORT:
        return "promote", "repair"
    return "rewrite", "repair"


def audit_policy(
    replay_cases: Sequence[Mapping[str, Any]],
    exp1302_payload: Mapping[str, Any],
    exp1303_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """REQ-LEARN-1315-4/5: measure non-forgetting and violation penalties."""

    decision_counts: Counter[str] = Counter({decision: 0 for decision in AUDIT_DECISIONS})
    cohort_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    old_total = 0
    old_preserved = 0
    memory_regressions = 0
    policy_accepted_violations = 0
    baseline_accepted_violations = 0
    adversarial_promotions = 0
    for case in replay_cases:
        action, final_decision = _policy_decision(case)
        target = str(case["target_decision"])
        cohort = str(case["cohort"])
        accepted_violation = int(final_decision == "accept" and target == "repair")
        baseline_accepted_violation = int(target == "repair")
        old_case = cohort == "old_verified"
        preserved_old_verified = old_case and final_decision == "accept" and not accepted_violation
        adversarial_promotion = int(cohort == "adversarial_unknown" and action == "promote")
        memory_regression = int(old_case and not preserved_old_verified)
        lagrangian_violation = int(bool(accepted_violation or adversarial_promotion or memory_regression))

        decision_counts[action] += 1
        cohort_counts[cohort] += 1
        old_total += int(old_case)
        old_preserved += int(preserved_old_verified)
        memory_regressions += memory_regression
        policy_accepted_violations += accepted_violation
        baseline_accepted_violations += baseline_accepted_violation
        adversarial_promotions += adversarial_promotion
        records.append(
            {
                "case_id": str(case["case_id"]),
                "cohort": cohort,
                "constraint_pattern": str(case["constraint_pattern"]),
                "target_decision": target,
                "verifier_result": str(case["verifier_result"]),
                "audit_decision": action,
                "final_decision": final_decision,
                "preserved_old_verified": bool(preserved_old_verified),
                "accepted_violation": bool(accepted_violation),
                "lagrangian_violation": bool(lagrangian_violation),
            }
        )

    denominator = len(replay_cases) or 1
    raw_self_learning_delta = float(exp1303_payload.get("self_learning_delta_overall") or 0.0)
    accepted_violation_delta = round(
        (policy_accepted_violations - baseline_accepted_violations) / denominator,
        6,
    )
    lagrangian_violation_penalty = round(
        (memory_regressions + policy_accepted_violations + adversarial_promotions) / denominator,
        6,
    )
    nonforgetting_rate = round(old_preserved / old_total, 6) if old_total else 0.0
    demoted_count = int(exp1302_payload.get("demoted_memory_count") or 0) + int(
        exp1302_payload.get("expired_memory_count") or 0
    )
    return {
        "replay_case_count": len(replay_cases),
        "replay_cohort_counts": {key: cohort_counts[key] for key in sorted(cohort_counts)},
        "audit_decision_counts": {decision: decision_counts[decision] for decision in AUDIT_DECISIONS},
        "policy_audit_records": records,
        "nonforgetting_certificate_rate": nonforgetting_rate,
        "old_verified_case_count": old_total,
        "old_verified_preserved_count": old_preserved,
        "memory_regression_count": memory_regressions,
        "policy_accepted_violation_count": policy_accepted_violations,
        "baseline_accepted_violation_count": baseline_accepted_violations,
        "accepted_violation_delta": accepted_violation_delta,
        "adversarial_promotion_count": adversarial_promotions,
        "lagrangian_violation_penalty": lagrangian_violation_penalty,
        "self_learning_delta_overall": round(raw_self_learning_delta - lagrangian_violation_penalty, 6),
        "source_self_learning_delta_overall": raw_self_learning_delta,
        "promoted_memory_count": int(exp1302_payload.get("promoted_memory_count") or 0),
        "demoted_memory_count": demoted_count,
    }


def derive_honest_verdict(
    *,
    nonforgetting_certificate_rate: float,
    memory_regression_count: int,
    self_learning_delta_overall: float,
    lagrangian_violation_penalty: float,
) -> str:
    """REQ-LEARN-1315-7: classify the audit without headline overclaiming."""

    if (
        nonforgetting_certificate_rate < 1.0
        or memory_regression_count > 0
        or lagrangian_violation_penalty > 0.0
    ):
        return "cerce_nonforgetting_regression_non_headline"
    if self_learning_delta_overall > 0.0:
        outcome = "improved"
    elif self_learning_delta_overall < 0.0:
        outcome = "regressed"
    else:
        outcome = "neutral"
    return f"cerce_nonforgetting_preserved_{outcome}_non_headline"


def build_artifact(
    exp1302_payload: Mapping[str, Any],
    exp1303_payload: Mapping[str, Any],
    exp1288_payload: Mapping[str, Any],
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1315-6: build the final CerCE non-forgetting artifact."""

    replay_cases = build_replay_set(exp1302_payload, exp1303_payload, exp1288_payload)
    metrics = audit_policy(replay_cases, exp1302_payload, exp1303_payload)
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "source_artifacts": SOURCE_ARTIFACTS,
        "status": "complete",
        "action_space": list(AUDIT_DECISIONS),
        "headline_result_allowed": False,
        "source_honest_verdicts": {
            "exp1302": str(exp1302_payload.get("honest_verdict") or ""),
            "exp1303": str(exp1303_payload.get("honest_verdict") or ""),
            "exp1288": str(exp1288_payload.get("honest_verdict") or ""),
        },
        **metrics,
    }
    artifact["honest_verdict"] = derive_honest_verdict(
        nonforgetting_certificate_rate=float(artifact["nonforgetting_certificate_rate"]),
        memory_regression_count=int(artifact["memory_regression_count"]),
        self_learning_delta_overall=float(artifact["self_learning_delta_overall"]),
        lagrangian_violation_penalty=float(artifact["lagrangian_violation_penalty"]),
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 1315 schema fields."""

    missing = sorted(REQUIRED_FIELDS.difference(artifact))
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise AssertionError("status must be complete or blocked")
    rate = float(artifact["nonforgetting_certificate_rate"])
    if rate < 0.0 or rate > 1.0:
        raise AssertionError("nonforgetting_certificate_rate must be between 0 and 1")
    if int(artifact["memory_regression_count"]) < 0:
        raise AssertionError("memory_regression_count must be non-negative")
    if float(artifact["lagrangian_violation_penalty"]) < 0.0:
        raise AssertionError("lagrangian_violation_penalty must be non-negative")
    if int(artifact["promoted_memory_count"]) < 0:
        raise AssertionError("promoted_memory_count must be non-negative")
    if int(artifact["demoted_memory_count"]) < 0:
        raise AssertionError("demoted_memory_count must be non-negative")
    if not isinstance(artifact["headline_result_allowed"], bool):
        raise AssertionError("headline_result_allowed must be boolean")
    if artifact["honest_verdict"] not in SUPPORTED_VERDICTS:
        raise AssertionError("honest_verdict is unsupported")


def run(
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    out_path: Path | str = DEFAULT_OUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1315-1/2: load inputs, audit policy, and write the result."""

    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    payloads, missing = load_source_artifacts(results_dir)
    if missing:
        artifact = write_terminal_blocker(
            out_path,
            missing,
            project_root=project_root,
            run_date=run_date,
        )
        validate_artifact(artifact)
        return artifact
    artifact = build_artifact(
        payloads[EXP1302_FILE],
        payloads[EXP1303_FILE],
        payloads[EXP1288_FILE],
        project_root=project_root,
        run_date=run_date,
    )
    validate_artifact(artifact)
    return _write_json(out_path, artifact)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        artifact["nonforgetting_certificate_rate"],
        artifact["self_learning_delta_overall"],
        artifact["honest_verdict"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXP1288_FILE",
    "EXP1302_FILE",
    "EXP1303_FILE",
    "OUTPUT_FILE",
    "audit_policy",
    "build_artifact",
    "build_replay_set",
    "derive_honest_verdict",
    "load_source_artifacts",
    "run",
    "validate_artifact",
    "write_in_progress_artifact",
    "write_terminal_blocker",
]
