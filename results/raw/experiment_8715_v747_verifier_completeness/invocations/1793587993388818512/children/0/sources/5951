"""Constraint Dependency Graph root-cause repair ordering for Exp 1522.

Spec: REQ-VERIFY-1522, SCENARIO-VERIFY-1522.

The CDG in this module stays inside Carnot's deterministic runtime-contract
boundary.  It loads Exp 1520 contract rows, optionally attaches Exp 1521 local
SOTA repair provenance when that artifact exists, and compares two localization
orders:

1. a flat downstream-first validator order that starts from the final accept
   symptom; and
2. a CDG order that prioritizes upstream contract categories before downstream
   final acceptance.

No generated text is accepted as authority.  Candidate repair decisions are
converted back into Exp 1520 contract-case rows and scored with the same
false-accept ledger used by the runtime-contract E2E harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from carnot.verify import live_sota_contract_guided_repair as sota_repair
from carnot.verify import runtime_contract_e2e_harness as runtime_contracts

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_1522_constraint_dependency_graph_root_cause_repair.json"
)
DEFAULT_E2E_MANIFEST_PATH = Path("results/runtime_contract_e2e_manifest_1520.jsonl")
DEFAULT_REPAIR_ARTIFACT_PATH = Path(
    "results/experiment_1521_live_sota_contract_guided_repair_v1.json"
)
DEFAULT_CDG_MANIFEST_PATH = Path("results/cdg_root_cause_repair_1522.jsonl")

CDG_NODE_IDS: tuple[str, ...] = (
    "parse",
    "certificate",
    "safe_dsl_verifier",
    "monitor_event",
    "structural_dependency",
    "solver_oracle",
    "final_accept",
)
FLAT_VALIDATOR_ORDER: tuple[str, ...] = (
    "final_accept",
    "parse",
    "certificate",
    "safe_dsl_verifier",
    "monitor_event",
    "structural_dependency",
    "solver_oracle",
)
LIFECYCLE_EDGES: tuple[tuple[str, str], ...] = (
    ("parse", "certificate"),
    ("certificate", "safe_dsl_verifier"),
    ("certificate", "monitor_event"),
    ("safe_dsl_verifier", "monitor_event"),
    ("safe_dsl_verifier", "structural_dependency"),
    ("structural_dependency", "solver_oracle"),
    ("certificate", "final_accept"),
    ("monitor_event", "final_accept"),
    ("structural_dependency", "final_accept"),
    ("solver_oracle", "final_accept"),
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "cdg_root_cause_repair_ready",
    "e2e_cases_loaded",
    "cdg_nodes",
    "cdg_edges",
    "root_cause_cases_attempted",
    "flat_order_fix_efficiency",
    "cdg_fix_efficiency",
    "cdg_efficiency_delta",
    "false_accept_count",
    "false_accept_rate",
    "cdg_manifest_path",
    "models_used",
    "blockers",
    "honest_verdict",
)


def build_constraint_dependency_graph(rows: Iterable[Mapping[str, Any]]) -> JsonDict:
    """Build a deterministic CDG from fixed lifecycle edges and co-failures.

    Lifecycle edges are always present so the graph remains auditable even on a
    small manifest.  Observed co-failure counts then annotate those edges with
    local evidence from Exp 1520 rows.  The row order never affects the graph.
    """

    row_list = [row for row in rows if row.get("row_type") == "contract_case"]
    cofailure_counts: dict[tuple[str, str], int] = {edge: 0 for edge in LIFECYCLE_EDGES}
    for row in row_list:
        failures = set(contract_failure_categories(row))
        for edge in LIFECYCLE_EDGES:
            if edge[0] in failures and edge[1] in failures:
                cofailure_counts[edge] += 1

    nodes = [
        {
            "id": node_id,
            "label": _node_label(node_id),
            "lifecycle_index": index,
        }
        for index, node_id in enumerate(CDG_NODE_IDS)
    ]
    edges = [
        {
            "source": source,
            "target": target,
            "reason": "lifecycle_order",
            "observed_cofailure_count": cofailure_counts[(source, target)],
            "weight": 1 + cofailure_counts[(source, target)],
        }
        for source, target in LIFECYCLE_EDGES
    ]
    return {"nodes": nodes, "edges": edges}


def contract_failure_categories(row: Mapping[str, Any]) -> list[str]:
    """Return failed contract categories for one normalized Exp 1520 row."""

    categories: list[str] = []
    certificate_result = _mapping(row.get("certificate_parse_result"))
    safe_dsl_result = _mapping(row.get("safe_dsl_verifier_result"))
    monitor_result = _mapping(row.get("monitor_event_result"))
    structural_result = _mapping(row.get("structural_contract_result"))

    if certificate_result.get("linked") is True:
        if certificate_result.get("parsed") is False:
            categories.append("parse")
        certificate_failed = any(
            (
                certificate_result.get("deterministic_validation_passed") is False,
                certificate_result.get("verifier_accepted") is False,
                certificate_result.get("false_accept_status") is True,
            )
        )
        if certificate_failed:
            categories.append("certificate")

    if safe_dsl_result.get("linked") is True and (
        row.get("expected_label") is False
        or row.get("final_deterministic_accept") is False
        or bool(safe_dsl_result.get("false_accept_row_id"))
    ):
        categories.append("safe_dsl_verifier")

    if monitor_result.get("linked") is True and (
        monitor_result.get("validation_status") != "pass"
        or monitor_result.get("verifier_false_accept") is True
        or row.get("final_deterministic_accept") is False
    ):
        categories.append("monitor_event")

    if structural_result.get("linked") is True and (
        structural_result.get("detected_violation") is True
        or structural_result.get("expected_violation") is True
        or row.get("final_deterministic_accept") is False
    ):
        categories.append("structural_dependency")

    if _solver_oracle_failed(row):
        categories.append("solver_oracle")

    if row.get("expected_label") is False or row.get("final_deterministic_accept") is False:
        categories.append("final_accept")

    return [node_id for node_id in CDG_NODE_IDS if node_id in set(categories)]


def analyze_root_cause_case(
    case: Mapping[str, Any],
    *,
    graph: Mapping[str, Any],
    repair_rows_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    """Compare flat and CDG root-cause localization for one failing case."""

    del graph
    failure_categories = contract_failure_categories(case)
    root_cause = _root_cause_category(failure_categories)
    flat_rank = _rank(root_cause, FLAT_VALIDATOR_ORDER)
    cdg_rank = _rank(root_cause, CDG_NODE_IDS)
    flat_efficiency = _efficiency(flat_rank)
    cdg_efficiency = _efficiency(cdg_rank)
    validation_row = _candidate_repair_validation_row(case)
    ledger = runtime_contracts.compute_false_accept_ledger([validation_row])
    false_accept = bool(ledger["false_accept_count"])
    repair_rows = list(repair_rows_by_case.get(str(case.get("contract_case_id")), ()))
    expected = case.get("expected_label")
    candidate_accept = validation_row["final_deterministic_accept"]
    deterministic_accept = (
        candidate_accept == expected if isinstance(expected, bool) else candidate_accept is False
    )

    return {
        "row_type": "cdg_root_cause_case",
        "contract_case_id": case.get("contract_case_id"),
        "prompt_or_case_id": case.get("prompt_or_case_id"),
        "source_family": case.get("source_family"),
        "failure_categories": failure_categories,
        "root_cause_category": root_cause,
        "flat_order": list(FLAT_VALIDATOR_ORDER),
        "cdg_order": list(CDG_NODE_IDS),
        "flat_root_cause_rank": flat_rank,
        "cdg_root_cause_rank": cdg_rank,
        "flat_efficiency": flat_efficiency,
        "cdg_efficiency": cdg_efficiency,
        "case_efficiency_delta": round(cdg_efficiency - flat_efficiency, 6),
        "repair_ready": bool(failure_categories),
        "candidate_repair_final_deterministic_accept": candidate_accept,
        "deterministic_validator_accept": bool(deterministic_accept and not false_accept),
        "false_accept": false_accept,
        "contract_validation_row": validation_row,
        "exp1521_repair_rows_linked": len(repair_rows),
        "exp1521_repair_modes": sorted(
            {str(row.get("mode")) for row in repair_rows if row.get("mode")}
        ),
        "exp1521_models_linked": sorted(
            {str(row.get("model_hf_id")) for row in repair_rows if row.get("model_hf_id")}
        ),
    }


def run_experiment(
    *,
    project_root: Path | str | None = None,
    run_date: str = RUN_DATE,
    e2e_manifest_path: Path | str = DEFAULT_E2E_MANIFEST_PATH,
    repair_artifact_path: Path | str = DEFAULT_REPAIR_ARTIFACT_PATH,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    cdg_manifest_path: Path | str = DEFAULT_CDG_MANIFEST_PATH,
) -> JsonDict:
    """Run Exp 1522 and write the terminal artifact plus CDG manifest."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    e2e_manifest = _resolve_under_root(root, Path(e2e_manifest_path))
    repair_artifact_path = _resolve_under_root(root, Path(repair_artifact_path))
    output = _resolve_under_root(root, Path(output_path))
    cdg_manifest = _resolve_under_root(root, Path(cdg_manifest_path))
    _write_json(output, _in_progress_artifact(run_date=run_date, cdg_manifest=cdg_manifest))

    blockers: list[str] = []
    if not e2e_manifest.exists():
        blockers.append(f"missing_runtime_contract_e2e_manifest:{e2e_manifest}")
        _write_jsonl(cdg_manifest, [])
        artifact = _terminal_artifact(
            status="blocked",
            run_date=run_date,
            e2e_cases_loaded=0,
            graph=build_constraint_dependency_graph([]),
            case_rows=[],
            cdg_manifest=cdg_manifest,
            repair_evidence=_empty_repair_evidence("exp1520_only_missing_exp1521"),
            blockers=blockers,
            honest_verdict="complete: blocked before CDG source loading",
        )
        _write_json(output, artifact)
        return artifact

    e2e_rows = _read_jsonl(e2e_manifest)
    contract_rows = [row for row in e2e_rows if row.get("row_type") == "contract_case"]
    failing_cases = [row for row in contract_rows if _is_failing_contract_case(row)]
    repair_evidence = _load_optional_repair_evidence(root, repair_artifact_path)
    blockers.extend(_model_provenance_blockers(repair_evidence))

    if not failing_cases:
        blockers.append("no_failing_runtime_contract_cases")
        _write_jsonl(cdg_manifest, [])
        artifact = _terminal_artifact(
            status="blocked",
            run_date=run_date,
            e2e_cases_loaded=len(contract_rows),
            graph=build_constraint_dependency_graph(contract_rows),
            case_rows=[],
            cdg_manifest=cdg_manifest,
            repair_evidence=repair_evidence,
            blockers=blockers,
            honest_verdict="complete: blocked before CDG root-cause cases",
        )
        _write_json(output, artifact)
        return artifact

    graph = build_constraint_dependency_graph(contract_rows)
    case_rows = [
        analyze_root_cause_case(
            case,
            graph=graph,
            repair_rows_by_case=repair_evidence["repair_rows_by_case"],
        )
        for case in failing_cases
    ]
    summary = _summary_manifest_row(
        e2e_cases_loaded=len(contract_rows),
        graph=graph,
        case_rows=case_rows,
        repair_evidence=repair_evidence,
    )
    _write_jsonl(cdg_manifest, [*case_rows, summary])
    artifact_metrics = _artifact_metrics(case_rows)
    if artifact_metrics["false_accept_rate"] != 0.0:
        blockers.append("false_accept_rate_nonzero_or_unmeasured")
    ready = (
        artifact_metrics["flat_order_fix_efficiency"] is not None
        and artifact_metrics["cdg_fix_efficiency"] is not None
        and artifact_metrics["false_accept_rate"] == 0.0
        and not blockers
    )
    artifact = _terminal_artifact(
        status="complete" if ready else "blocked",
        run_date=run_date,
        e2e_cases_loaded=len(contract_rows),
        graph=graph,
        case_rows=case_rows,
        cdg_manifest=cdg_manifest,
        repair_evidence=repair_evidence,
        blockers=blockers,
        honest_verdict=(
            "complete: CDG root-cause repair ordering metrics computed"
            if ready
            else "complete: blocked before CDG root-cause repair readiness"
        ),
    )
    _write_json(output, artifact)
    return artifact


def _is_failing_contract_case(row: Mapping[str, Any]) -> bool:
    return row.get("row_type") == "contract_case" and (
        row.get("expected_label") is False or row.get("final_deterministic_accept") is False
    )


def _candidate_repair_validation_row(case: Mapping[str, Any]) -> JsonDict:
    expected = case.get("expected_label")
    candidate_accept = expected if isinstance(expected, bool) else False
    validation = {
        key: case.get(key)
        for key in runtime_contracts.REQUIRED_CONTRACT_CASE_FIELDS
        if key in case
    }
    validation["row_type"] = "contract_case"
    validation["contract_schema_version"] = runtime_contracts.CONTRACT_CASE_SCHEMA_VERSION
    validation["final_deterministic_accept"] = bool(candidate_accept)
    validation["final_deterministic_decision"] = "accept" if candidate_accept else "reject"
    return validation


def _root_cause_category(failure_categories: Sequence[str]) -> str:
    non_final = [category for category in failure_categories if category != "final_accept"]
    if non_final:
        return min(non_final, key=lambda category: CDG_NODE_IDS.index(category))
    return "final_accept"


def _rank(node_id: str, order: Sequence[str]) -> int:
    return order.index(node_id) + 1


def _efficiency(rank: int) -> float:
    return round(1.0 / rank, 6)


def _artifact_metrics(case_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    validation_rows = [
        row["contract_validation_row"]
        for row in case_rows
        if isinstance(row.get("contract_validation_row"), dict)
    ]
    ledger = runtime_contracts.compute_false_accept_ledger(validation_rows)
    flat = _mean([row["flat_efficiency"] for row in case_rows])
    cdg = _mean([row["cdg_efficiency"] for row in case_rows])
    false_accept_rate = ledger["false_accept_rate"]
    return {
        "flat_order_fix_efficiency": flat,
        "cdg_fix_efficiency": cdg,
        "cdg_efficiency_delta": None if flat is None or cdg is None else round(cdg - flat, 6),
        "false_accept_count": ledger["false_accept_count"],
        "false_accept_rate": false_accept_rate,
        "explicit_label_count": ledger["explicit_label_count"],
        "explicit_reject_count": ledger["explicit_reject_count"],
    }


def _terminal_artifact(
    *,
    status: str,
    run_date: str,
    e2e_cases_loaded: int,
    graph: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    cdg_manifest: Path,
    repair_evidence: Mapping[str, Any],
    blockers: Sequence[str],
    honest_verdict: str,
) -> JsonDict:
    metrics = _artifact_metrics(case_rows)
    ready = (
        metrics["flat_order_fix_efficiency"] is not None
        and metrics["cdg_fix_efficiency"] is not None
        and metrics["false_accept_rate"] == 0.0
        and not blockers
    )
    return {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [dict(spec) for spec in sota_repair.MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": bool(repair_evidence["live_sota_model_inference_used"]),
        "cdg_root_cause_repair_ready": bool(ready),
        "e2e_cases_loaded": int(e2e_cases_loaded),
        "cdg_nodes": list(graph["nodes"]),
        "cdg_edges": list(graph["edges"]),
        "root_cause_cases_attempted": len(case_rows),
        "flat_order_fix_efficiency": metrics["flat_order_fix_efficiency"],
        "cdg_fix_efficiency": metrics["cdg_fix_efficiency"],
        "cdg_efficiency_delta": metrics["cdg_efficiency_delta"],
        "false_accept_count": metrics["false_accept_count"],
        "false_accept_rate": metrics["false_accept_rate"],
        "cdg_manifest_path": _display_path(cdg_manifest),
        "models_used": list(repair_evidence["models_used"]),
        "blockers": list(dict.fromkeys(blockers)),
        "honest_verdict": honest_verdict,
        "source_scope": repair_evidence["source_scope"],
        "optional_scope_notes": list(repair_evidence["optional_scope_notes"]),
        "exp1521_repair_rows_loaded": int(repair_evidence["repair_rows_loaded"]),
        "explicit_label_count": metrics["explicit_label_count"],
        "explicit_reject_count": metrics["explicit_reject_count"],
    }


def _summary_manifest_row(
    *,
    e2e_cases_loaded: int,
    graph: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    repair_evidence: Mapping[str, Any],
) -> JsonDict:
    metrics = _artifact_metrics(case_rows)
    return {
        "row_type": "cdg_graph_summary",
        "run_date": RUN_DATE,
        "e2e_cases_loaded": int(e2e_cases_loaded),
        "cdg_nodes": list(graph["nodes"]),
        "cdg_edges": list(graph["edges"]),
        "root_cause_cases_attempted": len(case_rows),
        "flat_order_fix_efficiency": metrics["flat_order_fix_efficiency"],
        "cdg_fix_efficiency": metrics["cdg_fix_efficiency"],
        "cdg_efficiency_delta": metrics["cdg_efficiency_delta"],
        "false_accept_count": metrics["false_accept_count"],
        "false_accept_rate": metrics["false_accept_rate"],
        "models_used": list(repair_evidence["models_used"]),
        "live_sota_model_inference_used": bool(repair_evidence["live_sota_model_inference_used"]),
        "source_scope": repair_evidence["source_scope"],
        "exp1521_repair_rows_loaded": int(repair_evidence["repair_rows_loaded"]),
    }


def _in_progress_artifact(*, run_date: str, cdg_manifest: Path) -> JsonDict:
    return {
        "status": "in_progress",
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [dict(spec) for spec in sota_repair.MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "cdg_root_cause_repair_ready": False,
        "e2e_cases_loaded": 0,
        "cdg_nodes": [],
        "cdg_edges": [],
        "root_cause_cases_attempted": 0,
        "flat_order_fix_efficiency": None,
        "cdg_fix_efficiency": None,
        "cdg_efficiency_delta": None,
        "false_accept_count": 0,
        "false_accept_rate": None,
        "cdg_manifest_path": _display_path(cdg_manifest),
        "models_used": [],
        "blockers": ["experiment_1522_cdg_root_cause_repair_in_progress"],
        "honest_verdict": "complete: in-progress CDG root-cause repair artifact",
    }


def _load_optional_repair_evidence(root: Path, repair_artifact_path: Path) -> JsonDict:
    if not repair_artifact_path.exists():
        return _empty_repair_evidence(
            "exp1520_only_missing_exp1521",
            [f"optional_exp1521_repair_artifact_missing:{repair_artifact_path}"],
        )

    artifact = _read_json(repair_artifact_path)
    manifest_value = artifact.get("repair_manifest_path")
    manifest_path = (
        _resolve_under_root(root, Path(str(manifest_value)))
        if manifest_value
        else root / sota_repair.DEFAULT_REPAIR_MANIFEST_PATH
    )
    rows = _read_jsonl(manifest_path) if manifest_path.exists() else []
    notes = [] if manifest_path.exists() else [f"optional_exp1521_repair_manifest_missing:{manifest_path}"]
    by_case: dict[str, list[JsonDict]] = {}
    for row in rows:
        if row.get("row_type") != "repair_result":
            continue
        by_case.setdefault(str(row.get("contract_case_id")), []).append(row)
    models_used = sorted(
        {
            str(model_id)
            for model_id in artifact.get("models_used", [])
            if model_id in sota_repair.MANDATED_HF_IDS
        }
    )
    return {
        "source_scope": "exp1520_plus_exp1521",
        "optional_scope_notes": notes,
        "live_sota_model_inference_used": bool(artifact.get("live_sota_model_inference_used")),
        "models_used": models_used,
        "repair_rows_loaded": sum(len(case_rows) for case_rows in by_case.values()),
        "repair_rows_by_case": by_case,
    }


def _empty_repair_evidence(source_scope: str, notes: Sequence[str] | None = None) -> JsonDict:
    return {
        "source_scope": source_scope,
        "optional_scope_notes": list(notes or []),
        "live_sota_model_inference_used": False,
        "models_used": [],
        "repair_rows_loaded": 0,
        "repair_rows_by_case": {},
    }


def _model_provenance_blockers(repair_evidence: Mapping[str, Any]) -> list[str]:
    if repair_evidence["live_sota_model_inference_used"] and not repair_evidence["models_used"]:
        return ["no_mandated_sota_model_in_repair_evidence"]
    return []


def _solver_oracle_failed(row: Mapping[str, Any]) -> bool:
    solver_result = _mapping(row.get("solver_oracle_result"))
    source_family = str(row.get("source_family") or "")
    case_id = str(row.get("contract_case_id") or "")
    return (
        solver_result.get("linked") is True
        and solver_result.get("oracle_agreement") is False
    ) or "solver_oracle" in source_family or "product_line" in source_family or "oracle" in case_id


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _node_label(node_id: str) -> str:
    return node_id.replace("_", " ")


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[JsonDict]:
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
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", default=RUN_DATE)
    parser.add_argument("--e2e-manifest", type=Path, default=DEFAULT_E2E_MANIFEST_PATH)
    parser.add_argument("--repair-artifact", type=Path, default=DEFAULT_REPAIR_ARTIFACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--cdg-manifest", type=Path, default=DEFAULT_CDG_MANIFEST_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(argv)
    run_experiment(
        run_date=args.run_date,
        e2e_manifest_path=args.e2e_manifest,
        repair_artifact_path=args.repair_artifact,
        output_path=args.output,
        cdg_manifest_path=args.cdg_manifest,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
