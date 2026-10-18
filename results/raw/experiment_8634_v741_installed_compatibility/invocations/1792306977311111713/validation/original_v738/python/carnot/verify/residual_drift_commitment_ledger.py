"""Residual-drift commitment ledger for Exp 1538.

Spec: REQ-VERIFY-1538, SCENARIO-VERIFY-1538.

This module treats prior experiment manifests as recorded multi-turn traces:
the prompt introduces hard commitments, the model or staged repair introduces
an attempted final state, and the deterministic oracle decides whether the
state remembered every prior commitment.  The important distinction is that a
bad final state is not automatically a contradiction.  If a solver can still
show a satisfying completion for the earlier commitments, the failure is
ledgered as satisfiable drift instead of an impossible constraint set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1538_residual_drift_commitment_ledger.json")
DEFAULT_LEDGER_PATH = Path("results/residual_drift_commitment_ledger_1538.jsonl")
DEFAULT_SATQUEST_MANIFEST_PATH = Path("results/satquest_cnf_verifier_1536.jsonl")
DEFAULT_PRODUCT_LINE_MANIFEST_PATH = Path("results/product_line_rescue_1523.jsonl")
DEFAULT_RUNTIME_CDG_MANIFEST_PATH = Path("results/cdg_root_cause_repair_1522.jsonl")

MANDATED_MODEL_SPECS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

CLASS_TRUE_CONTRADICTION = "true_contradiction"
CLASS_SATISFIABLE_DRIFT = "satisfiable_drift"
CLASS_OTHER_BLOCKER = "other_blocker"
CLASS_ACCEPTED = "accepted"

DEFAULT_SATQUEST_LIMIT = 18
DEFAULT_PRODUCT_LINE_LIMIT = 6
DEFAULT_RUNTIME_CONTRACT_LIMIT = 120

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "residual_drift_ledger_ready",
    "model_specs",
    "live_sota_model_inference_used",
    "multi_turn_cases",
    "contradiction_cases",
    "satisfiable_drift_cases",
    "drift_rate",
    "repaired_drift_cases",
    "solver_oracle_used",
    "false_accept_rate",
    "ledger_path",
    "focused_tests_passed",
    "honest_verdict",
)


def classify_satquest_case(row: Mapping[str, Any]) -> JsonDict:
    """Replay one SATQuest row against the SAT solver commitment."""

    baseline = _mapping(row.get("baseline"))
    parse_result = _mapping(row.get("parse_result"))
    verifier = _mapping(row.get("verifier"))
    oracle = _sat_solver_oracle(row)
    parse_ok = bool(parse_result.get("parse_ok")) and baseline.get("answer") is not None
    baseline_correct = bool(baseline.get("correct"))
    baseline_answer = _optional_upper(baseline.get("answer"))
    classification = str(baseline.get("classification") or "")

    if baseline_correct:
        failure_class = CLASS_ACCEPTED
        other_blocker = None
    elif not parse_ok:
        failure_class = CLASS_OTHER_BLOCKER
        other_blocker = classification or str(parse_result.get("parse_error") or "parse_failure")
    elif oracle["satisfiable"]:
        failure_class = CLASS_SATISFIABLE_DRIFT
        other_blocker = None
    elif baseline_answer == "SAT":
        failure_class = CLASS_TRUE_CONTRADICTION
        other_blocker = None
    else:
        failure_class = CLASS_OTHER_BLOCKER
        other_blocker = classification or "solver_oracle_mismatch"

    repaired = failure_class == CLASS_SATISFIABLE_DRIFT and (
        bool(_mapping(row.get("repair_hint")).get("correct"))
        or bool(_mapping(row.get("energy_ranked")).get("correct"))
    )
    return {
        "row_type": "residual_drift_case",
        "source_domain": "satquest",
        "source_case_id": row.get("case_id"),
        "source_instance_id": row.get("instance_id"),
        "model_hf_id": row.get("model_hf_id"),
        "live_sota_model_inference_used": _is_live_sota_row(row),
        "commitments": [
            _commitment(1, "cnf_constraints", {"n_vars": row.get("n_vars"), "clauses": row.get("clauses")}),
            _commitment(2, "model_answer", {"answer": baseline_answer, "parse_ok": parse_ok}),
            _commitment(3, "solver_oracle_validation", oracle),
            _commitment(4, "final_decision", {"baseline_correct": baseline_correct}),
        ],
        "solver_oracle": oracle,
        "deterministic_validator": {
            "baseline_correct": baseline_correct,
            "classification": classification,
            "model_self_false_accept": bool(verifier.get("self_verifier_false_accept")),
        },
        "failure_classification": failure_class,
        "other_blocker": other_blocker,
        "repaired_drift": repaired,
        "false_accept": False,
        "bounded_claim": "classification applies only to this replayed SATQuest row",
    }


def classify_product_line_case(row: Mapping[str, Any]) -> JsonDict:
    """Replay one product-line rescue row against the exhaustive oracle result."""

    baseline = _mapping(row.get("baseline_result"))
    oracle = _mapping(row.get("oracle_result"))
    verifier = _mapping(row.get("verifier_result"))
    policy = _mapping(row.get("policy_result"))
    parse_ok = bool(baseline.get("parse_ok"))
    baseline_agrees = bool(baseline.get("oracle_agrees"))
    oracle_agrees_after = bool(oracle.get("oracle_agrees")) or oracle.get("classification") == "oracle_agreement"
    has_completion = oracle_agrees_after or bool(oracle.get("optimal_features"))

    if baseline_agrees:
        failure_class = CLASS_ACCEPTED
        other_blocker = None
    elif not parse_ok:
        failure_class = CLASS_OTHER_BLOCKER
        other_blocker = "parse_failure"
    elif has_completion:
        failure_class = CLASS_SATISFIABLE_DRIFT
        other_blocker = None
    else:
        failure_class = CLASS_TRUE_CONTRADICTION
        other_blocker = None

    repaired = failure_class == CLASS_SATISFIABLE_DRIFT and bool(
        verifier.get("accepted") and oracle_agrees_after
    )
    return {
        "row_type": "residual_drift_case",
        "source_domain": "product_line",
        "source_case_id": row.get("case_id"),
        "model_hf_id": row.get("model_hf_id"),
        "live_sota_model_inference_used": _is_live_sota_row(row),
        "commitments": [
            _commitment(1, "feature_model_contract", {"model_id": row.get("model_id"), "operation": row.get("operation")}),
            _commitment(2, "baseline_selection", baseline),
            _commitment(3, "staged_feedback", {"stages": row.get("stages") or []}),
            _commitment(4, "solver_oracle_validation", oracle),
        ],
        "solver_oracle": {
            "used": True,
            "satisfiable": has_completion,
            "oracle_agrees_after_repair": oracle_agrees_after,
        },
        "deterministic_validator": {
            "baseline_oracle_agrees": baseline_agrees,
            "oracle_agrees_after_repair": oracle_agrees_after,
            "policy_false_accept": bool(policy.get("false_accept")),
        },
        "failure_classification": failure_class,
        "other_blocker": other_blocker,
        "repaired_drift": repaired,
        "false_accept": bool(policy.get("false_accept")),
        "bounded_claim": "classification applies only to this replayed product-line row",
    }


def classify_runtime_contract_case(row: Mapping[str, Any]) -> JsonDict:
    """Replay one CDG runtime-contract failure row against contract validation evidence."""

    validation = _mapping(row.get("contract_validation_row"))
    structural = _mapping(validation.get("structural_contract_result"))
    root_cause = str(row.get("root_cause_category") or "")
    has_structural_completion = root_cause == "structural_dependency" and bool(
        structural.get("detected_violation")
    )
    false_accept = bool(row.get("false_accept"))

    if false_accept or root_cause == "solver_oracle":
        failure_class = CLASS_TRUE_CONTRADICTION
        other_blocker = None
    elif has_structural_completion:
        failure_class = CLASS_SATISFIABLE_DRIFT
        other_blocker = None
    else:
        failure_class = CLASS_OTHER_BLOCKER
        other_blocker = root_cause or "runtime_contract_blocker"

    return {
        "row_type": "residual_drift_case",
        "source_domain": "runtime_contract",
        "source_case_id": row.get("contract_case_id"),
        "prompt_or_case_id": row.get("prompt_or_case_id"),
        "model_hf_id": None,
        "live_sota_model_inference_used": False,
        "commitments": [
            _commitment(1, "runtime_contract_prompt", {"prompt_or_case_id": row.get("prompt_or_case_id")}),
            _commitment(2, "cdg_failure_localization", {"root_cause_category": root_cause, "failure_categories": row.get("failure_categories") or []}),
            _commitment(3, "deterministic_contract_validation", validation),
        ],
        "solver_oracle": {
            "used": True,
            "satisfiable": has_structural_completion,
            "root_cause_category": root_cause,
        },
        "deterministic_validator": {
            "deterministic_validator_accept": bool(row.get("deterministic_validator_accept")),
            "final_deterministic_accept": validation.get("final_deterministic_accept"),
            "expected_label": validation.get("expected_label"),
        },
        "failure_classification": failure_class,
        "other_blocker": other_blocker,
        "repaired_drift": bool(
            failure_class == CLASS_SATISFIABLE_DRIFT
            and row.get("candidate_repair_final_deterministic_accept") is True
        ),
        "false_accept": false_accept,
        "bounded_claim": "classification applies only to this replayed runtime-contract row",
    }


def build_ledger_rows(
    *,
    satquest_rows: Sequence[Mapping[str, Any]],
    product_line_rows: Sequence[Mapping[str, Any]],
    runtime_cdg_rows: Sequence[Mapping[str, Any]],
    satquest_limit: int = DEFAULT_SATQUEST_LIMIT,
    product_line_limit: int = DEFAULT_PRODUCT_LINE_LIMIT,
    runtime_contract_limit: int = DEFAULT_RUNTIME_CONTRACT_LIMIT,
) -> list[JsonDict]:
    """Build bounded ledger rows from the three source domains."""

    rows: list[JsonDict] = []
    rows.extend(
        classify_satquest_case(row)
        for row in _bounded((row for row in satquest_rows if _satquest_failure(row)), satquest_limit)
    )
    rows.extend(
        classify_product_line_case(row)
        for row in _bounded(
            (row for row in product_line_rows if _product_line_failure(row)),
            product_line_limit,
        )
    )
    rows.extend(
        classify_runtime_contract_case(row)
        for row in _bounded(
            (row for row in runtime_cdg_rows if row.get("row_type") == "cdg_root_cause_case"),
            runtime_contract_limit,
        )
    )
    return rows


def summarize_ledger_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute artifact metrics from already-classified ledger rows."""

    total = len(rows)
    contradiction = sum(1 for row in rows if row.get("failure_classification") == CLASS_TRUE_CONTRADICTION)
    drift = sum(1 for row in rows if row.get("failure_classification") == CLASS_SATISFIABLE_DRIFT)
    false_accepts = sum(1 for row in rows if row.get("false_accept") is True)
    models_used = sorted(
        {
            str(row.get("model_hf_id"))
            for row in rows
            if row.get("model_hf_id") in MANDATED_MODEL_SPECS
        }
    )
    source_counts = {
        domain: sum(1 for row in rows if row.get("source_domain") == domain)
        for domain in ("satquest", "product_line", "runtime_contract")
    }
    return {
        "multi_turn_cases": total,
        "contradiction_cases": contradiction,
        "satisfiable_drift_cases": drift,
        "other_blocker_cases": sum(1 for row in rows if row.get("failure_classification") == CLASS_OTHER_BLOCKER),
        "drift_rate": _rate(drift, total),
        "repaired_drift_cases": sum(1 for row in rows if row.get("repaired_drift") is True),
        "solver_oracle_used": any(_mapping(row.get("solver_oracle")).get("used") for row in rows),
        "false_accept_count": false_accepts,
        "false_accept_rate": _rate(false_accepts, total),
        "live_sota_model_inference_used": any(
            row.get("live_sota_model_inference_used") is True for row in rows
        ),
        "models_used": models_used,
        "source_case_counts": source_counts,
        "source_model_self_false_accept_count": sum(
            1
            for row in rows
            if _mapping(row.get("deterministic_validator")).get("model_self_false_accept") is True
        ),
    }


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable bootstrap artifact before source manifests are loaded."""

    payload = _artifact(
        status="in_progress",
        run_date=run_date,
        ledger_path=Path(ledger_path),
        metrics=summarize_ledger_rows([]),
        ready=False,
        focused_tests_passed=False,
        blockers=["experiment_1538_residual_drift_in_progress"],
        honest_verdict="in_progress: residual drift commitment ledger initialized",
    )
    _write_json(Path(output_path), payload)
    return payload


def run_experiment(
    *,
    project_root: Path | str | None = None,
    run_date: str = RUN_DATE,
    satquest_manifest_path: Path | str = DEFAULT_SATQUEST_MANIFEST_PATH,
    product_line_manifest_path: Path | str = DEFAULT_PRODUCT_LINE_MANIFEST_PATH,
    runtime_cdg_manifest_path: Path | str = DEFAULT_RUNTIME_CDG_MANIFEST_PATH,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
    focused_tests_passed: bool = False,
) -> JsonDict:
    """Run Exp 1538 and persist the terminal artifact plus JSONL ledger."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    ledger = _resolve_under_root(root, Path(ledger_path))
    satquest_manifest = _resolve_under_root(root, Path(satquest_manifest_path))
    product_line_manifest = _resolve_under_root(root, Path(product_line_manifest_path))
    runtime_cdg_manifest = _resolve_under_root(root, Path(runtime_cdg_manifest_path))
    write_in_progress_artifact(output, ledger_path=ledger, run_date=run_date)

    blockers = _missing_source_blockers(
        satquest_manifest=satquest_manifest,
        product_line_manifest=product_line_manifest,
        runtime_cdg_manifest=runtime_cdg_manifest,
    )
    if blockers:
        _write_jsonl(ledger, [])
        artifact = _artifact(
            status="blocked",
            run_date=run_date,
            ledger_path=ledger,
            metrics=summarize_ledger_rows([]),
            ready=False,
            focused_tests_passed=focused_tests_passed,
            blockers=blockers,
            honest_verdict="complete: blocked before residual-drift source loading",
        )
        _write_json(output, artifact)
        return artifact

    ledger_rows = build_ledger_rows(
        satquest_rows=_read_jsonl(satquest_manifest),
        product_line_rows=_read_jsonl(product_line_manifest),
        runtime_cdg_rows=_read_jsonl(runtime_cdg_manifest),
    )
    metrics = summarize_ledger_rows(ledger_rows)
    blockers.extend(_case_domain_blockers(metrics["source_case_counts"]))
    summary = {
        "row_type": "residual_drift_summary",
        "run_date": run_date,
        **metrics,
        "bounded_claim": "metrics cover only replayed Exp 1538 source manifests",
    }
    _write_jsonl(ledger, [*ledger_rows, summary] if ledger_rows else [])
    ready = (
        bool(ledger_rows)
        and not blockers
        and focused_tests_passed
        and metrics["false_accept_rate"] == 0.0
        and ledger.exists()
    )
    artifact = _artifact(
        status="complete" if ready else "blocked",
        run_date=run_date,
        ledger_path=ledger,
        metrics=metrics,
        ready=ready,
        focused_tests_passed=focused_tests_passed,
        blockers=blockers,
        honest_verdict=(
            "complete: residual-drift commitment ledger ready for bounded replay only"
            if ready
            else "complete: blocked before residual-drift ledger readiness"
        ),
    )
    _write_json(output, artifact)
    return artifact


def _artifact(
    *,
    status: str,
    run_date: str,
    ledger_path: Path,
    metrics: Mapping[str, Any],
    ready: bool,
    focused_tests_passed: bool,
    blockers: Sequence[str],
    honest_verdict: str,
) -> JsonDict:
    return {
        "status": status,
        "milestone": run_date,
        "run_date": run_date,
        "schema_version": 1,
        "residual_drift_ledger_ready": bool(ready),
        "model_specs": list(MANDATED_MODEL_SPECS),
        "live_sota_model_inference_used": bool(metrics["live_sota_model_inference_used"]),
        "multi_turn_cases": int(metrics["multi_turn_cases"]),
        "contradiction_cases": int(metrics["contradiction_cases"]),
        "satisfiable_drift_cases": int(metrics["satisfiable_drift_cases"]),
        "other_blocker_cases": int(metrics["other_blocker_cases"]),
        "drift_rate": metrics["drift_rate"],
        "repaired_drift_cases": int(metrics["repaired_drift_cases"]),
        "solver_oracle_used": bool(metrics["solver_oracle_used"]),
        "false_accept_count": int(metrics["false_accept_count"]),
        "false_accept_rate": metrics["false_accept_rate"],
        "ledger_path": _display_path(ledger_path),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": honest_verdict,
        "models_used": list(metrics["models_used"]),
        "source_case_counts": dict(metrics["source_case_counts"]),
        "source_model_self_false_accept_count": int(metrics["source_model_self_false_accept_count"]),
        "blockers": list(dict.fromkeys(blockers)),
        "claim_scope": "bounded replay of checked-in SATQuest, product-line, and runtime-contract rows only",
    }


def _sat_solver_oracle(row: Mapping[str, Any]) -> JsonDict:
    source = _mapping(row.get("solver_oracle"))
    label = str(source.get("label") or "").upper()
    return {
        "used": bool(label),
        "backend": source.get("backend"),
        "label": label or None,
        "satisfiable": label == "SAT",
        "checked_assignments": source.get("checked_assignments"),
        "satisfying_assignment": source.get("satisfying_assignment"),
    }


def _satquest_failure(row: Mapping[str, Any]) -> bool:
    return not bool(_mapping(row.get("baseline")).get("correct"))


def _product_line_failure(row: Mapping[str, Any]) -> bool:
    return not bool(_mapping(row.get("baseline_result")).get("oracle_agrees"))


def _missing_source_blockers(
    *,
    satquest_manifest: Path,
    product_line_manifest: Path,
    runtime_cdg_manifest: Path,
) -> list[str]:
    blockers: list[str] = []
    if not satquest_manifest.exists():
        blockers.append(f"missing_satquest_manifest:{satquest_manifest}")
    if not product_line_manifest.exists():
        blockers.append(f"missing_product_line_manifest:{product_line_manifest}")
    if not runtime_cdg_manifest.exists():
        blockers.append(f"missing_runtime_cdg_manifest:{runtime_cdg_manifest}")
    return blockers


def _case_domain_blockers(source_counts: Mapping[str, int]) -> list[str]:
    return [
        f"no_{domain}_cases"
        for domain, count in source_counts.items()
        if int(count) <= 0
    ]


def _commitment(turn: int, name: str, evidence: Mapping[str, Any]) -> JsonDict:
    return {"turn": turn, "name": name, "evidence": dict(evidence)}


def _is_live_sota_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("generation_source") == "live_sota_llamacpp"
        and row.get("model_hf_id") in MANDATED_MODEL_SPECS
        and not row.get("blocker")
    )


def _bounded(rows: Iterable[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    return list(rows)[: max(0, int(limit))]


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_upper(value: Any) -> str | None:
    return str(value).upper() if isinstance(value, str) else None


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 6)


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path | str) -> str:
    as_path = Path(path)
    try:
        return str(as_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(as_path)


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused-tests-passed", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(focused_tests_passed=args.focused_tests_passed)
    print(
        "[exp1538] "
        f"ready={artifact['residual_drift_ledger_ready']} "
        f"cases={artifact['multi_turn_cases']} "
        f"drift={artifact['satisfiable_drift_cases']} "
        f"false_accept={artifact['false_accept_rate']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())


__all__ = [
    "CLASS_ACCEPTED",
    "CLASS_OTHER_BLOCKER",
    "CLASS_SATISFIABLE_DRIFT",
    "CLASS_TRUE_CONTRADICTION",
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_LEDGER_PATH",
    "MANDATED_MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_ledger_rows",
    "classify_product_line_case",
    "classify_runtime_contract_case",
    "classify_satquest_case",
    "run_experiment",
    "summarize_ledger_rows",
    "write_in_progress_artifact",
]
