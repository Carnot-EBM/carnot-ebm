"""Exp1554 product-line staged scale v4 behind the unified contract gate.

Spec: REQ-VERIFY-1554, SCENARIO-VERIFY-1554.

Exp1540 already proved the bounded product-line branch can reach zero false
accepts on a 40-case staged pack.  This module scales the same deterministic
authority boundary while requiring Exp1551's unified contract gate to be ready
first.  The model output can be malformed, natural-language wrapped, or
automata-shaped, but the reported metrics come only from structured parser and
solver-oracle fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from carnot.eval import product_line_solver_oracle_benchmark as product_line1511
from carnot.eval import product_line_staged_benchmark_scale as product_line1540
from carnot.verify import unified_contract_gate

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
MILESTONE = "20260508"
TARGET_CASE_COUNT = 120
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1554_product_line_staged_scale_v4.json")
DEFAULT_MANIFEST_PATH = Path("results/product_line_staged_scale_v4_1554.jsonl")

MODEL_SPECS: tuple[str, ...] = unified_contract_gate.MODEL_SPECS
STAGE_VARIANTS: tuple[str, ...] = (
    "syntax_only",
    "feasibility",
    "objective",
    "natural_language",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "product_line_scale_v4_ready",
    "branch_retired",
    "model_specs",
    "live_sota_model_inference_used",
    "cases_total",
    "stages_tested",
    "parse_rate",
    "feasibility_rate",
    "objective_gap_mean",
    "oracle_agreement_rate",
    "entity_hallucination_rate",
    "false_accept_rate",
    "automata_constraints_used",
    "product_line_manifest_path",
    "focused_tests_passed",
    "honest_verdict",
)
REQUIRED_ROW_FIELDS: tuple[str, ...] = (
    "case_id",
    "v4_stage",
    "parse_result",
    "oracle_result",
    "oracle_label",
    "policy_result",
    "stages",
)


@dataclass(frozen=True)
class PredecessorPaths:
    """Artifacts whose deterministic readiness gates the v4 scale report."""

    exp1540_artifact: Path = product_line1540.DEFAULT_ARTIFACT_PATH
    exp1551_artifact: Path = unified_contract_gate.DEFAULT_ARTIFACT_PATH


RowBuilderFn = Callable[[int], list[JsonDict]]
ModelProbeFn = Callable[[Path], JsonDict]


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable bootstrap artifact before loading predecessor state."""

    payload: JsonDict = {
        "status": "in_progress",
        "milestone": run_date,
        "product_line_scale_v4_ready": False,
        "branch_retired": False,
        "model_specs": list(MODEL_SPECS),
        "live_sota_model_inference_used": False,
        "cases_total": 0,
        "stages_tested": [],
        "parse_rate": 0.0,
        "feasibility_rate": 0.0,
        "objective_gap_mean": None,
        "oracle_agreement_rate": 0.0,
        "entity_hallucination_rate": 0.0,
        "false_accept_rate": 0.0,
        "automata_constraints_used": False,
        "product_line_manifest_path": _display_path(Path(manifest_path)),
        "focused_tests_passed": False,
        "honest_verdict": "complete: in_progress",
    }
    _write_json(Path(output_path), payload)
    return payload


def build_stage_manifest(target_count: int = TARGET_CASE_COUNT) -> list[JsonDict]:
    """Build v4 rows that cover syntax, feasibility, objective, and NL variants."""

    cases = product_line1540.build_staged_product_line_cases(target_count=target_count)
    rows: list[JsonDict] = []
    for index, case in enumerate(cases):
        stage = STAGE_VARIANTS[index % len(STAGE_VARIANTS)]
        source_row = _source_row_for_stage(case, stage, use_automata=index % 8 == 4)
        row = product_line1540.evaluate_staged_case(case, source_row)
        row["v4_stage"] = stage
        row["structured_solver_fields_present"] = True
        rows.append(row)
    return rows


def validate_stage_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Validate that every row still has parser, solver, and policy evidence."""

    errors: list[str] = []
    stages_tested: list[str] = []
    for index, row in enumerate(rows):
        for field in REQUIRED_ROW_FIELDS:
            if field not in row:
                errors.append(f"row:{index}:missing_required_field:{field}")
        stage = str(row.get("v4_stage", ""))
        if stage in STAGE_VARIANTS and stage not in stages_tested:
            stages_tested.append(stage)
        _validate_nested_bool(errors, row, index, "parse_result", "parse_ok")
        _validate_nested_bool(errors, row, index, "oracle_result", "feasible")
        _validate_nested_bool(errors, row, index, "oracle_result", "oracle_agrees")
        _validate_nested_bool(errors, row, index, "policy_result", "false_accept")
        _validate_nested_number(errors, row, index, "oracle_label", "optimal_cost")
        _validate_nested_number(errors, row, index, "oracle_label", "optimal_value")

    missing_stages = [stage for stage in STAGE_VARIANTS if stage not in stages_tested]
    errors.extend(f"missing_stage_variant:{stage}" for stage in missing_stages)
    deterministic = not any("missing_required_field" in error for error in errors)
    deterministic = deterministic and not any("missing_deterministic_field" in error for error in errors)
    return {
        "valid": not errors,
        "cases_total": len(rows),
        "stages_tested": stages_tested,
        "deterministic_checks_available": deterministic,
        "errors": errors,
    }


def aggregate_manifest_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate only structured parser and solver-oracle row fields."""

    total = len(rows)
    if total == 0:
        return {
            "parse_rate": 0.0,
            "feasibility_rate": 0.0,
            "objective_gap_mean": None,
            "oracle_agreement_rate": 0.0,
            "entity_hallucination_rate": 0.0,
            "false_accept_rate": 0.0,
            "false_accept_count": 0,
            "entity_hallucination_count": 0,
            "classification_counts": {},
        }

    parse_count = sum(1 for row in rows if _nested_bool(row, "parse_result", "parse_ok"))
    feasible_count = sum(1 for row in rows if _nested_bool(row, "oracle_result", "feasible"))
    oracle_count = sum(1 for row in rows if _nested_bool(row, "oracle_result", "oracle_agrees"))
    false_accept_count = sum(1 for row in rows if _nested_bool(row, "policy_result", "false_accept"))
    hallucination_count = sum(1 for row in rows if _entity_hallucination_detected(row))
    objective_gaps = [_objective_gap(row) for row in rows]
    counts: dict[str, int] = {}
    for row in rows:
        classification = str(_nested_mapping(row, "oracle_result").get("classification"))
        counts[classification] = counts.get(classification, 0) + 1
    return {
        "parse_rate": _rate(parse_count, total),
        "feasibility_rate": _rate(feasible_count, total),
        "objective_gap_mean": round(sum(objective_gaps) / total, 6),
        "oracle_agreement_rate": _rate(oracle_count, total),
        "entity_hallucination_rate": _rate(hallucination_count, total),
        "false_accept_rate": _rate(false_accept_count, total),
        "false_accept_count": false_accept_count,
        "entity_hallucination_count": hallucination_count,
        "classification_counts": counts,
    }


def decide_scale_v4_readiness(
    *,
    cases_total: int,
    validation: Mapping[str, Any],
    metrics: Mapping[str, Any],
    unified_contract_gate_ready: bool,
    focused_tests_passed: bool,
    blockers: Sequence[str],
) -> JsonDict:
    """Apply the REQ-VERIFY-1554 ready-or-retire gate."""

    false_accept_rate = float(metrics.get("false_accept_rate", 0.0))
    retirement_reason = ""
    if false_accept_rate > 0.0:
        retirement_reason = f"false_accept_rate exceeded zero: {false_accept_rate}"
    elif not bool(validation.get("deterministic_checks_available", False)):
        retirement_reason = "deterministic_manifest_fields_missing"

    blocking_errors = [str(blocker) for blocker in blockers]
    ready = (
        unified_contract_gate_ready
        and cases_total > 0
        and bool(validation.get("valid", False))
        and set(validation.get("stages_tested", [])) == set(STAGE_VARIANTS)
        and false_accept_rate == 0.0
        and focused_tests_passed
        and not blocking_errors
        and not retirement_reason
    )
    return {
        "product_line_scale_v4_ready": bool(ready),
        "branch_retired": bool(retirement_reason),
        "retirement_reason": retirement_reason,
    }


def load_predecessor_artifacts(
    predecessor_paths: PredecessorPaths | None = None,
    *,
    project_root: Path | str | None = None,
) -> tuple[JsonDict, list[str]]:
    """Load Exp1540 and Exp1551, returning concrete blockers for closed gates."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    paths = predecessor_paths or PredecessorPaths()
    loaded: JsonDict = {}
    blockers: list[str] = []
    for key, source in (
        ("exp1540", paths.exp1540_artifact),
        ("exp1551", paths.exp1551_artifact),
    ):
        path = _resolve_under_root(root, Path(source))
        if not path.exists():
            blockers.append(f"missing_{key}_artifact:{_display_path(path, root)}")
            continue
        loaded[key] = _read_json(path)

    exp1540 = _mapping(loaded.get("exp1540"))
    exp1551 = _mapping(loaded.get("exp1551"))
    if exp1540 and exp1540.get("product_line_scale_ready") is not True:
        blockers.append("exp1540_product_line_scale_not_ready")
    if exp1540 and float(exp1540.get("false_accept_rate", 1.0)) != 0.0:
        blockers.append(f"exp1540_false_accept_rate_nonzero:{exp1540.get('false_accept_rate')}")
    if exp1540 and float(exp1540.get("oracle_agreement_rate", 0.0)) < 1.0:
        blockers.append(f"exp1540_oracle_agreement_below_one:{exp1540.get('oracle_agreement_rate')}")
    if exp1551 and exp1551.get("unified_contract_gate_ready") is not True:
        blockers.append("exp1551_unified_contract_gate_not_ready")
    if exp1551 and float(exp1551.get("false_accept_rate", 1.0)) != 0.0:
        blockers.append(f"exp1551_false_accept_rate_nonzero:{exp1551.get('false_accept_rate')}")
    if exp1551 and exp1551.get("product_line_oracle_used") is not True:
        blockers.append("exp1551_product_line_oracle_not_used")
    return loaded, blockers


def run_experiment(
    *,
    project_root: Path | str | None = None,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    predecessor_paths: PredecessorPaths | None = None,
    target_count: int = TARGET_CASE_COUNT,
    focused_tests_passed: bool = False,
    row_builder_fn: RowBuilderFn | None = None,
    model_probe_fn: ModelProbeFn | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Run Exp1554 and write the terminal JSON plus JSONL manifest."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    manifest = _resolve_under_root(root, Path(manifest_path))
    write_in_progress_artifact(output, manifest_path=manifest, run_date=run_date)

    predecessors, blockers = load_predecessor_artifacts(predecessor_paths, project_root=root)
    model_probe = (model_probe_fn or unified_contract_gate.probe_headline_model_availability)(root)
    unified_ready = _mapping(predecessors.get("exp1551")).get("unified_contract_gate_ready") is True
    rows = [] if blockers else (row_builder_fn or build_stage_manifest)(target_count)
    if not rows and not blockers:
        blockers.append("no_product_line_v4_cases")
    if not focused_tests_passed:
        blockers.append("focused_tests_not_passed")

    validation = validate_stage_manifest_rows(rows)
    metrics = aggregate_manifest_metrics(rows)
    if rows and not validation["valid"] and "manifest_validation_failed" not in blockers:
        blockers.append("manifest_validation_failed")
    decision = decide_scale_v4_readiness(
        cases_total=len(rows),
        validation=validation,
        metrics=metrics,
        unified_contract_gate_ready=unified_ready,
        focused_tests_passed=focused_tests_passed,
        blockers=blockers,
    )
    _write_jsonl(manifest, rows)
    artifact = _terminal_artifact(
        status="complete" if rows else "blocked",
        run_date=run_date,
        rows=rows,
        manifest_path=manifest,
        validation=validation,
        metrics=metrics,
        decision=decision,
        focused_tests_passed=focused_tests_passed,
        model_probe=model_probe,
        predecessors=predecessors,
        blockers=blockers,
    )
    _write_json(output, artifact)
    return artifact


def _terminal_artifact(
    *,
    status: str,
    run_date: str,
    rows: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    validation: Mapping[str, Any],
    metrics: Mapping[str, Any],
    decision: Mapping[str, Any],
    focused_tests_passed: bool,
    model_probe: Mapping[str, Any],
    predecessors: Mapping[str, Any],
    blockers: Sequence[str],
) -> JsonDict:
    ready = bool(decision["product_line_scale_v4_ready"])
    retired = bool(decision["branch_retired"])
    return {
        "status": status,
        "milestone": MILESTONE,
        "run_date": run_date,
        "schema_version": 1,
        "product_line_scale_v4_ready": ready,
        "branch_retired": retired,
        "model_specs": list(MODEL_SPECS),
        "live_sota_model_inference_used": bool(
            model_probe.get("live_sota_model_inference_used", False)
        ),
        "cases_total": len(rows),
        "stages_tested": list(validation.get("stages_tested", [])),
        "parse_rate": metrics["parse_rate"],
        "feasibility_rate": metrics["feasibility_rate"],
        "objective_gap_mean": metrics["objective_gap_mean"],
        "oracle_agreement_rate": metrics["oracle_agreement_rate"],
        "entity_hallucination_rate": metrics["entity_hallucination_rate"],
        "false_accept_rate": metrics["false_accept_rate"],
        "automata_constraints_used": any(
            bool(row.get("automata_constraints_used")) for row in rows
        ),
        "product_line_manifest_path": _display_path(manifest_path),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": _honest_verdict(ready, retired, str(decision["retirement_reason"])),
        "false_accept_count": metrics["false_accept_count"],
        "entity_hallucination_count": metrics["entity_hallucination_count"],
        "classification_counts": metrics["classification_counts"],
        "deterministic_checks_available": bool(
            validation.get("deterministic_checks_available", False)
        ),
        "stage_manifest_validation_errors": list(validation.get("errors", [])),
        "retirement_reason": str(decision["retirement_reason"]),
        "target_cases_requested": TARGET_CASE_COUNT,
        "evaluated_target_count": len(rows),
        "model_availability_blockers": list(model_probe.get("availability_blockers", [])),
        "models_used": list(model_probe.get("models_used", [])),
        "model_probe": dict(model_probe),
        "legacy_small_models_excluded_from_headline_metrics": bool(
            model_probe.get("legacy_small_models_excluded_from_headline_metrics", True)
        ),
        "predecessor_artifacts_loaded": sorted(predecessors),
        "predecessor_summary": _predecessor_summary(predecessors),
        "blockers": list(dict.fromkeys(blockers)),
    }


def _source_row_for_stage(
    case: product_line1511.ProductLineCase,
    stage: str,
    *,
    use_automata: bool,
) -> JsonDict:
    if stage == "syntax_only" and use_automata:
        output = product_line1540.compile_product_line_answer_dfa(case).generate()
        return _source_row(case, "automata_guided_oracle", "automata_guided_abs_dfa", output)
    if stage == "syntax_only":
        return _source_row(
            case,
            "syntax_failure",
            "deterministic_syntax_failure_seed",
            f"not-json answer for {case.case_id}",
        )
    if stage == "feasibility":
        payload = _feature_repair_payload(case)
        return _source_row(
            case,
            "feature_model_repair",
            "deterministic_feature_model_seed",
            json.dumps(payload, sort_keys=True),
        )
    if stage == "objective":
        payload = product_line1540._solver_repair_payload(case)  # noqa: SLF001
        return _source_row(
            case,
            "solver_repair",
            "deterministic_solver_repair_seed",
            json.dumps(payload, sort_keys=True),
        )
    answer = product_line1511.compliant_answer_for_case(case)
    return _source_row(
        case,
        "natural_language",
        "deterministic_natural_language_seed",
        f"The solver-backed product-line selection is:\n{answer}\nNo extra feature is selected.",
    )


def _feature_repair_payload(case: product_line1511.ProductLineCase) -> JsonDict:
    selected = set(case.model.mandatory)
    selected.update(sorted(case.operation.include)[:1])
    selected.add("BogusFeature")
    return {
        "selected_features": sorted(selected),
        "objective_cost": 0,
        "objective_value": 0,
        "verifier": {"accept": False},
    }


def _source_row(
    case: product_line1511.ProductLineCase,
    seed_mode: str,
    generation_source: str,
    output: str,
) -> JsonDict:
    return {
        "case_id": case.case_id,
        "seed_mode": seed_mode,
        "model_hf_id": MODEL_SPECS[0],
        "model_name": "Qwen3.6-35B-A3B",
        "generation_source": generation_source,
        "model_output": output,
        "elapsed_seconds": 0.0,
        "blocker": None,
    }


def _objective_gap(row: Mapping[str, Any]) -> float:
    oracle = _nested_mapping(row, "oracle_result")
    label = _nested_mapping(row, "oracle_label")
    operation = _nested_mapping(row, "operation")
    if bool(oracle.get("oracle_agrees")):
        return 0.0
    if operation.get("kind") == "min_cost":
        return max(0.0, float(oracle.get("selection_cost") or 0) - float(label.get("optimal_cost") or 0))
    return max(0.0, float(label.get("optimal_value") or 0) - float(oracle.get("selection_value") or 0))


def _entity_hallucination_detected(row: Mapping[str, Any]) -> bool:
    for stage in row.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        feedback = str(stage.get("feedback", ""))
        if "removed_unknown:" in feedback or "unknown:" in feedback:
            return True
    return False


def _validate_nested_bool(
    errors: list[str],
    row: Mapping[str, Any],
    index: int,
    container: str,
    field: str,
) -> None:
    value = _nested_mapping(row, container).get(field)
    if not isinstance(value, bool):
        errors.append(f"row:{index}:missing_deterministic_field:{container}.{field}")


def _validate_nested_number(
    errors: list[str],
    row: Mapping[str, Any],
    index: int,
    container: str,
    field: str,
) -> None:
    value = _nested_mapping(row, container).get(field)
    if not isinstance(value, int | float):
        errors.append(f"row:{index}:missing_deterministic_field:{container}.{field}")


def _nested_bool(row: Mapping[str, Any], container: str, field: str) -> bool:
    return _nested_mapping(row, container).get(field) is True


def _nested_mapping(row: Mapping[str, Any], container: str) -> Mapping[str, Any]:
    value = row.get(container)
    return value if isinstance(value, Mapping) else {}


def _predecessor_summary(predecessors: Mapping[str, Any]) -> JsonDict:
    exp1540 = _mapping(predecessors.get("exp1540"))
    exp1551 = _mapping(predecessors.get("exp1551"))
    return {
        "exp1540_status": exp1540.get("status"),
        "exp1540_product_line_scale_ready": exp1540.get("product_line_scale_ready"),
        "exp1540_false_accept_rate": exp1540.get("false_accept_rate"),
        "exp1540_oracle_agreement_rate": exp1540.get("oracle_agreement_rate"),
        "exp1551_status": exp1551.get("status"),
        "exp1551_unified_contract_gate_ready": exp1551.get("unified_contract_gate_ready"),
        "exp1551_false_accept_rate": exp1551.get("false_accept_rate"),
        "exp1551_product_line_oracle_used": exp1551.get("product_line_oracle_used"),
    }


def _honest_verdict(ready: bool, retired: bool, reason: str) -> str:
    if ready:
        return "complete: product-line staged scale v4 ready with zero false accepts"
    if retired:
        return f"complete_retired: product-line scale v4 branch retired: {reason}"
    return "complete_blocked: product-line staged scale v4 not ready"


def _rate(count: int, total: int) -> float:
    return round(count / total, 6)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path | None = None) -> str:
    base = root or Path.cwd()
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """CLI entry point for conductor and manual experiment runs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-tests-passed", action="store_true")
    parser.add_argument("--target-count", type=int, default=TARGET_CASE_COUNT)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    artifact = run_experiment(
        target_count=args.target_count,
        focused_tests_passed=args.focused_tests_passed,
    )
    print(
        "[exp1554] "
        f"ready={artifact['product_line_scale_v4_ready']} "
        f"retired={artifact['branch_retired']} "
        f"cases={artifact['cases_total']} "
        f"false_accept={artifact['false_accept_rate']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "MODEL_SPECS",
    "PredecessorPaths",
    "REQUIRED_ARTIFACT_FIELDS",
    "STAGE_VARIANTS",
    "aggregate_manifest_metrics",
    "build_stage_manifest",
    "decide_scale_v4_readiness",
    "load_predecessor_artifacts",
    "run_experiment",
    "validate_stage_manifest_rows",
    "write_in_progress_artifact",
]
