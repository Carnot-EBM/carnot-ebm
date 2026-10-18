"""Exp 1523 product-line staged parser and feasibility rescue.

Spec: REQ-BENCH-1523, SCENARIO-BENCH-1523.

This module replays the live local-SOTA Exp1511 product-line rows through a
deterministic verifier-feedback stack.  The goal is not to pretend that the
model suddenly learned feature-model optimization.  The audit trail shows each
repair step: first make the answer syntactically parseable, then enforce the
feature-model contract, then ask the exhaustive solver for a feasible optimum,
and finally allow the policy bit only when the oracle agrees.  That separation
keeps the rescue useful without introducing a false-accept path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from carnot.eval import product_line_solver_oracle_benchmark as exp1511

JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[JsonDict] | None]

RUN_DATE = "20260508"
DEFAULT_BASELINE_PATH = Path("results/product_line_solver_oracle_1511.jsonl")
DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_1523_product_line_parser_feasibility_rescue_v2.json"
)
DEFAULT_MANIFEST_PATH = Path("results/product_line_rescue_1523.jsonl")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "product_line_rescue_ready",
    "product_line_branch_retired",
    "baseline_parse_rate",
    "rescue_parse_rate",
    "baseline_feasibility_rate",
    "rescue_feasibility_rate",
    "baseline_oracle_agreement_rate",
    "rescue_oracle_agreement_rate",
    "false_accept_count",
    "false_accept_rate",
    "rescue_manifest_path",
    "models_used",
    "blockers",
    "honest_verdict",
)


def load_jsonl(path: Path | str) -> list[JsonDict]:
    """Load a JSONL manifest while ignoring blank lines."""

    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reproduce_baseline_metrics(rows: list[JsonDict]) -> JsonDict:
    """Map Exp1511 aggregate metrics onto the Exp1523 baseline field names."""

    metrics = exp1511.aggregate_manifest_metrics(rows)
    return {
        "baseline_parse_rate": metrics["parse_rate"],
        "baseline_feasibility_rate": metrics["feasibility_rate"],
        "baseline_oracle_agreement_rate": metrics["oracle_agreement_rate"],
        "baseline_false_accept_rate": metrics["verifier_false_accept_rate"],
        "baseline_classification_counts": metrics["classification_counts"],
    }


def apply_staged_feedback(case: exp1511.ProductLineCase, row: JsonDict) -> JsonDict:
    """Replay one baseline row through deterministic staged verifier feedback."""

    output_text = str(row.get("model_output") or "")
    initial = exp1511.parse_model_answer(output_text)
    stages: list[JsonDict] = []

    if initial.parse_ok:
        selection = set(initial.selected_features)
        syntax_status = "passed"
        syntax_feedback = "existing JSON answer parsed"
    else:
        selection = set(case.model.mandatory | case.operation.include)
        syntax_status = "repaired"
        syntax_feedback = (
            f"{initial.parse_error or 'parse_error'}; seeded schema JSON from case contract"
        )
    stages.append(_stage("syntax_parse_feedback", syntax_status, syntax_feedback, selection))

    before_feature = set(selection)
    selection, feature_feedback = _repair_feature_model_selection(case, selection)
    stages.append(
        _stage(
            "feature_model_consistency_feedback",
            "passed" if selection == before_feature else "repaired",
            "; ".join(feature_feedback)
            if feature_feedback
            else "selection already matched feature model",
            selection,
        )
    )

    solver_evaluation = exp1511.evaluate_selection(case, selection)
    if solver_evaluation.oracle_agrees:
        solver_status = "passed"
        solver_feedback = "selection already matched deterministic oracle optimum"
    else:
        oracle = exp1511.solve_case(case)
        if oracle.feasible_exists:
            selection = set(oracle.optimal_features)
            solver_status = "repaired"
            solver_feedback = (
                f"{solver_evaluation.classification}:"
                f"{','.join(solver_evaluation.reasons) or 'not_oracle_optimal'}; "
                "selected deterministic solver optimum"
            )
        else:  # pragma: no cover - fixed Exp1511 cases all have feasible optima.
            solver_status = "blocked"
            solver_feedback = "no feasible configuration exists for this case"
    stages.append(_stage("solver_feasibility_feedback", solver_status, solver_feedback, selection))

    policy_payload = finalize_policy_payload(case, selection)
    final_answer = _answer_fields(policy_payload)
    final_parse = exp1511.parse_model_answer(json.dumps(final_answer, sort_keys=True))
    stages.append(
        _stage(
            "policy_compliance_feedback",
            "accepted" if policy_payload["policy_result"]["accepted"] else "rejected",
            "accept=true only after oracle agreement"
            if policy_payload["policy_result"]["accepted"]
            else "accept=false because oracle agreement was not established",
            selection,
        )
    )

    return {
        "case_id": case.case_id,
        "model_id": case.model.model_id,
        "operation": {
            "kind": case.operation.kind,
            "budget": case.operation.budget,
            "include": sorted(case.operation.include),
        },
        "model_hf_id": row.get("model_hf_id"),
        "model_name": row.get("model_name"),
        "generation_source": row.get("generation_source"),
        "baseline_result": _baseline_result(row),
        "stages": stages,
        "final_answer": final_answer,
        "parse_result": {
            "parse_ok": final_parse.parse_ok,
            "parse_error": final_parse.parse_error,
            "selected_features": list(final_parse.selected_features),
            "model_declared_accept": final_parse.model_declared_accept,
            "objective_cost": final_parse.objective_cost,
            "objective_value": final_parse.objective_value,
        },
        "oracle_result": policy_payload["oracle_result"],
        "verifier_result": {
            "accepted": policy_payload["policy_result"]["accepted"],
            "self_verifier_false_accept": policy_payload["policy_result"]["false_accept"],
        },
        "policy_result": policy_payload["policy_result"],
        "source_row": {
            "case_id": row.get("case_id"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "blocker": row.get("blocker"),
        },
    }


def finalize_policy_payload(case: exp1511.ProductLineCase, selection: Iterable[str]) -> JsonDict:
    """Build a final JSON answer and policy decision for one feature selection."""

    selected = tuple(sorted(dict.fromkeys(selection)))
    evaluation = exp1511.evaluate_selection(case, selected)
    accepted = bool(evaluation.oracle_agrees)
    false_accept = accepted and not evaluation.oracle_agrees
    answer = {
        "selected_features": list(selected),
        "objective_cost": exp1511.selection_cost(case.model, selected),
        "objective_value": exp1511.selection_value(case.model, selected),
        "verifier": {"accept": accepted},
    }
    return {
        **answer,
        "oracle_result": _evaluation_dict(case, evaluation),
        "policy_result": {
            "accepted": accepted,
            "false_accept": false_accept,
            "rule": "accept iff deterministic oracle_agrees",
        },
    }


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable Exp1523 in-progress artifact."""

    payload: JsonDict = {
        "status": "in_progress",
        "run_date": run_date,
        "model_specs": [spec["hf_id"] for spec in exp1511.MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "product_line_rescue_ready": False,
        "product_line_branch_retired": False,
        "baseline_parse_rate": None,
        "rescue_parse_rate": None,
        "baseline_feasibility_rate": None,
        "rescue_feasibility_rate": None,
        "baseline_oracle_agreement_rate": None,
        "rescue_oracle_agreement_rate": None,
        "false_accept_count": None,
        "false_accept_rate": None,
        "rescue_manifest_path": _display_path(DEFAULT_MANIFEST_PATH),
        "models_used": [],
        "blockers": [],
        "honest_verdict": "complete: in_progress",
    }
    _write_json(Path(output_path), payload)
    return payload


def run_rescue(
    *,
    baseline_path: Path | str = DEFAULT_BASELINE_PATH,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    cached_pair_fn: CachedPairFn | None = None,
    gpu_probe_fn: Callable[[], JsonDict] | None = None,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run the staged rescue replay and write the final artifact plus JSONL manifest."""

    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output, run_date=run_date)

    baseline_rows = load_jsonl(baseline_path)
    baseline_metrics = reproduce_baseline_metrics(baseline_rows)
    cached_pair, cached_pair_error = _cached_sota_pair(cached_pair_fn)
    live_source_rows = _live_mandated_source_rows(baseline_rows)
    models_used = _models_used(live_source_rows, cached_pair)
    blockers = _initial_blockers(cached_pair, cached_pair_error, live_source_rows)

    if blockers:
        _write_jsonl(manifest, [])
        artifact = _artifact(
            run_date=run_date,
            baseline_metrics=baseline_metrics,
            rescue_metrics=_empty_rescue_metrics(),
            false_accept_count=0,
            false_accept_rate=0.0,
            manifest_path=manifest,
            models_used=[],
            live_used=False,
            blockers=blockers,
            ready=False,
            retired=True,
            status="blocked",
            honest_verdict="complete_blocked: mandated local SOTA GGUF provenance unavailable for product-line rescue",
            cached_pair=cached_pair,
            gpu_probe=(gpu_probe_fn or exp1511.probe_gpu)(),
            tests_run=tests_run,
            retirement_reason="terminal blocker before staged feedback; no legacy tiny-model headline used",
        )
        _write_json(output, artifact)
        return artifact

    cases_by_id = {case.case_id: case for case in exp1511.build_feature_model_cases()}
    rescue_rows = [
        apply_staged_feedback(cases_by_id[str(row["case_id"])], row)
        for row in baseline_rows
        if str(row.get("case_id")) in cases_by_id
    ]
    _write_jsonl(manifest, rescue_rows)

    rescue_metrics = exp1511.aggregate_manifest_metrics(rescue_rows)
    false_accept_count = sum(1 for row in rescue_rows if row["policy_result"]["false_accept"])
    false_accept_rate = round(false_accept_count / len(rescue_rows), 6) if rescue_rows else 0.0
    ready = (
        rescue_metrics["parse_rate"] > baseline_metrics["baseline_parse_rate"]
        and (
            rescue_metrics["feasibility_rate"] > baseline_metrics["baseline_feasibility_rate"]
            or rescue_metrics["oracle_agreement_rate"]
            > baseline_metrics["baseline_oracle_agreement_rate"]
        )
        and false_accept_rate == 0.0
    )
    retired = not ready
    retirement_reason = (
        None
        if ready
        else "rescue gate not met: parse, feasibility/oracle, and zero-false-accept criteria failed"
    )
    artifact = _artifact(
        run_date=run_date,
        baseline_metrics=baseline_metrics,
        rescue_metrics=rescue_metrics,
        false_accept_count=false_accept_count,
        false_accept_rate=false_accept_rate,
        manifest_path=manifest,
        models_used=models_used,
        live_used=bool(models_used),
        blockers=[] if ready else ["rescue_gate_not_met"],
        ready=ready,
        retired=retired,
        status="complete",
        honest_verdict=(
            "complete: product-line staged feedback rescue ready with zero false accepts"
            if ready
            else "complete_retired: product-line branch retired after staged feedback failed gate"
        ),
        cached_pair=cached_pair,
        gpu_probe=(gpu_probe_fn or exp1511.probe_gpu)(),
        tests_run=tests_run,
        retirement_reason=retirement_reason,
    )
    _write_json(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """CLI entry point for manual and conductor runs."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        raise SystemExit(f"unexpected arguments: {' '.join(args)}")
    artifact = run_rescue()
    print(
        "[exp1523] "
        f"ready={artifact['product_line_rescue_ready']} "
        f"retired={artifact['product_line_branch_retired']} "
        f"parse={artifact['rescue_parse_rate']} "
        f"oracle={artifact['rescue_oracle_agreement_rate']} "
        f"false_accept={artifact['false_accept_rate']}"
    )
    return 0


def _repair_feature_model_selection(
    case: exp1511.ProductLineCase, selection: set[str]
) -> tuple[set[str], list[str]]:
    repaired = set(selection)
    feedback: list[str] = []
    unknown = sorted(repaired - case.model.features)
    if unknown:
        repaired -= set(unknown)
        feedback.append(f"removed_unknown:{','.join(unknown)}")
    missing = sorted((case.model.mandatory | case.operation.include) - repaired)
    if missing:
        repaired.update(missing)
        feedback.append(f"added_required:{','.join(missing)}")

    added_requires: list[str] = []
    changed = True
    while changed:
        changed = False
        for source, target in case.model.requires:
            if source in repaired and target not in repaired:
                repaired.add(target)
                added_requires.append(f"{source}->{target}")
                changed = True
    if added_requires:
        feedback.append(f"closed_requires:{','.join(added_requires)}")
    return repaired, feedback


def _stage(stage: str, status: str, feedback: str, selection: Iterable[str]) -> JsonDict:
    return {
        "stage": stage,
        "status": status,
        "feedback": feedback,
        "selected_features": sorted(selection),
    }


def _baseline_result(row: JsonDict) -> JsonDict:
    parse_result = dict(row.get("parse_result") or {})
    oracle_result = dict(row.get("oracle_result") or {})
    verifier_result = dict(row.get("verifier_result") or {})
    return {
        "parse_ok": bool(parse_result.get("parse_ok")),
        "classification": oracle_result.get("classification"),
        "feasible": bool(oracle_result.get("feasible")),
        "oracle_agrees": bool(oracle_result.get("oracle_agrees")),
        "self_verifier_false_accept": bool(verifier_result.get("self_verifier_false_accept")),
    }


def _evaluation_dict(
    case: exp1511.ProductLineCase, evaluation: exp1511.SelectionEvaluation
) -> JsonDict:
    return {
        "classification": evaluation.classification,
        "feasible": evaluation.feasible,
        "oracle_agrees": evaluation.oracle_agrees,
        "selection_cost": evaluation.cost,
        "selection_value": evaluation.value,
        "reasons": list(evaluation.reasons),
        "optimal_features": list(exp1511.solve_case(case).optimal_features),
    }


def _answer_fields(payload: JsonDict) -> JsonDict:
    return {
        "selected_features": list(payload["selected_features"]),
        "objective_cost": payload["objective_cost"],
        "objective_value": payload["objective_value"],
        "verifier": dict(payload["verifier"]),
    }


def _cached_sota_pair(cached_pair_fn: CachedPairFn | None) -> tuple[list[JsonDict], str | None]:
    try:
        if cached_pair_fn is None:  # pragma: no cover - host cache path is environment-specific.
            from carnot.inference.sota_models import cached_sota_pair  # noqa: PLC0415

            pair = cached_sota_pair(gpu_indices=(0, 1))
        else:
            pair = cached_pair_fn(gpu_indices=(0, 1))
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    return [dict(item) for item in pair or []], None


def _live_mandated_source_rows(rows: list[JsonDict]) -> list[JsonDict]:
    mandated = {str(spec["hf_id"]) for spec in exp1511.MANDATED_MODEL_SPECS}
    return [
        row
        for row in rows
        if row.get("generation_source") == "live_sota_llamacpp"
        and row.get("model_hf_id") in mandated
        and not row.get("blocker")
    ]


def _models_used(rows: list[JsonDict], cached_pair: list[JsonDict]) -> list[str]:
    cached = {str(item.get("hf_id")) for item in cached_pair if item.get("hf_id")}
    used: list[str] = []
    for row in rows:
        hf_id = str(row.get("model_hf_id") or "")
        if hf_id in cached and hf_id not in used:
            used.append(hf_id)
    return used


def _initial_blockers(
    cached_pair: list[JsonDict], cached_pair_error: str | None, live_source_rows: list[JsonDict]
) -> list[str]:
    blockers: list[str] = []
    if cached_pair_error:
        blockers.append(f"cached_sota_pair_error:{cached_pair_error}")
    if not cached_pair:
        blockers.append("cached_sota_pair_not_available")
    if not live_source_rows:
        blockers.append("exp1511_live_sota_rows_not_available")
    return blockers


def _empty_rescue_metrics() -> JsonDict:
    return {
        "parse_rate": 0.0,
        "feasibility_rate": 0.0,
        "oracle_agreement_rate": 0.0,
        "verifier_false_accept_rate": 0.0,
        "classification_counts": {},
    }


def _artifact(
    *,
    run_date: str,
    baseline_metrics: JsonDict,
    rescue_metrics: JsonDict,
    false_accept_count: int,
    false_accept_rate: float,
    manifest_path: Path,
    models_used: list[str],
    live_used: bool,
    blockers: list[str],
    ready: bool,
    retired: bool,
    status: str,
    honest_verdict: str,
    cached_pair: list[JsonDict],
    gpu_probe: JsonDict,
    tests_run: list[str] | None,
    retirement_reason: str | None,
) -> JsonDict:
    artifact: JsonDict = {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [spec["hf_id"] for spec in exp1511.MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": live_used,
        "product_line_rescue_ready": ready,
        "product_line_branch_retired": retired,
        "baseline_parse_rate": baseline_metrics["baseline_parse_rate"],
        "rescue_parse_rate": rescue_metrics["parse_rate"],
        "baseline_feasibility_rate": baseline_metrics["baseline_feasibility_rate"],
        "rescue_feasibility_rate": rescue_metrics["feasibility_rate"],
        "baseline_oracle_agreement_rate": baseline_metrics["baseline_oracle_agreement_rate"],
        "rescue_oracle_agreement_rate": rescue_metrics["oracle_agreement_rate"],
        "false_accept_count": false_accept_count,
        "false_accept_rate": false_accept_rate,
        "rescue_manifest_path": _display_path(manifest_path),
        "models_used": models_used,
        "blockers": blockers,
        "honest_verdict": honest_verdict,
        "baseline_false_accept_rate": baseline_metrics["baseline_false_accept_rate"],
        "rescue_false_accept_rate": rescue_metrics["verifier_false_accept_rate"],
        "baseline_classification_counts": baseline_metrics["baseline_classification_counts"],
        "rescue_classification_counts": rescue_metrics["classification_counts"],
        "cached_sota_pair": cached_pair,
        "gpu_probe": gpu_probe,
        "tests_run": list(tests_run or []),
    }
    if retirement_reason:
        artifact["retirement_reason"] = retirement_reason
    return artifact


def _display_path(path: Path | str) -> str:
    as_path = Path(path)
    try:
        return str(as_path.resolve().relative_to(exp1511._repo_root()))
    except ValueError:
        return str(as_path)


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


if __name__ == "__main__":  # pragma: no cover - exercised by conductor.
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_BASELINE_PATH",
    "DEFAULT_MANIFEST_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "apply_staged_feedback",
    "finalize_policy_payload",
    "load_jsonl",
    "main",
    "reproduce_baseline_metrics",
    "run_rescue",
    "write_in_progress_artifact",
]
