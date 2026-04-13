"""Build a deterministic explicit code-spec corpus from benchmark traces.

Spec: REQ-CODE-023, REQ-CODE-024,
      SCENARIO-CODE-020, SCENARIO-CODE-021
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any

from carnot.pipeline.property_code_verifier import (
    _REVERSE_PATTERN,
    _SORT_ASCENDING_PATTERN,
    _SORT_DESCENDING_PATTERN,
    _allows_input_mutation,
    _annotation_kind,
    extract_official_test_examples,
    extract_prompt_examples,
)

RUN_DATE = "20260413"
SCHEMA_VERSION = "carnot.code_spec_corpus.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = REPO_ROOT / "data" / "research" / "code_spec_corpus_236.jsonl"
RESULTS_PATH = REPO_ROOT / "results" / "experiment_236_results.json"
SOURCE_ARTIFACTS = (
    Path("results/experiment_226_results.json"),
    Path("results/experiment_227_results.json"),
)
SPEC_FAMILIES = (
    "preconditions",
    "postconditions",
    "invariants",
    "mutation_constraints",
    "oracle_hints",
)
_MUTABLE_KINDS = {"list", "dict", "set"}


def get_repo_root() -> Path:
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return REPO_ROOT


def resolve_path(repo_root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else repo_root / candidate


def _cli_default_path(candidate: Path) -> Path:
    try:
        return candidate.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return candidate


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(content, encoding="utf-8")


def _display_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _cohort_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cohort = payload.get("cohort")
    cohort_dict = cohort if isinstance(cohort, dict) else {}
    cases = cohort_dict.get("cases")
    if not isinstance(cases, list):
        return {}
    return {
        str(case["case_id"]): case
        for case in cases
        if isinstance(case, dict) and case.get("case_id")
    }


def _find_function_node(source: str, entry_point: str) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return node
    return None


def _extract_signature_data(prompt: str, entry_point: str) -> tuple[str, list[tuple[str, str, str]], str | None]:
    node = _find_function_node(prompt, entry_point)
    if node is None:
        return (f"{entry_point}(...)",
                [],
                None)

    params: list[tuple[str, str, str]] = []
    rendered_args: list[str] = []
    for arg in node.args.args:
        if arg.arg == "self":
            continue
        annotation_text = ast.unparse(arg.annotation) if arg.annotation is not None else "Any"
        rendered_args.append(f"{arg.arg}: {annotation_text}")
        params.append((arg.arg, _annotation_kind(arg.annotation) or "unknown", annotation_text))

    signature = f"{entry_point}({', '.join(rendered_args)})"
    return_annotation = ast.unparse(node.returns) if node.returns is not None else None
    if return_annotation is not None:
        signature = f"{signature} -> {return_annotation}"
    return (signature, params, return_annotation)


def _trace_observation(
    *,
    artifact_label: str,
    payload: dict[str, Any],
    case_meta: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    baseline = result.get("baseline")
    baseline_dict = baseline if isinstance(baseline, dict) else {}
    verify_repair = result.get("verify_repair")
    verify_repair_dict = verify_repair if isinstance(verify_repair, dict) else {}
    failure_records = baseline_dict.get("pbt_failure_records")
    failure_names: set[str] = set()
    if isinstance(failure_records, list):
        for record in failure_records:
            if isinstance(record, dict) and record.get("property_name"):
                failure_names.add(str(record["property_name"]))

    experiment = int(payload.get("experiment") or 0)
    case_id = str(result.get("case_id") or case_meta.get("case_id") or "")
    return {
        "artifact": artifact_label,
        "case_id": case_id,
        "experiment": experiment,
        "failure_properties": sorted(failure_names),
        "model_name": str(payload.get("metadata", {}).get("model_name") or ""),
        "official_test_miss": bool(baseline_dict.get("official_test_miss_caught_by_pbt")),
        "repair_iterations": int(verify_repair_dict.get("n_repairs") or 0),
        "repaired": bool(verify_repair_dict.get("repaired")),
        "source_ref": f"exp{experiment}:{case_id}",
    }


def _empty_family_map() -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    return {family: {} for family in SPEC_FAMILIES}


def _add_clause(
    families: dict[str, dict[tuple[str, str], dict[str, Any]]],
    family: str,
    *,
    kind: str,
    text: str,
    sources: list[str] | None = None,
    trace_refs: list[str] | None = None,
) -> None:
    item = families[family].setdefault(
        (kind, text),
        {"kind": kind, "text": text, "sources": [], "trace_refs": []},
    )
    for source in sources or []:
        if source and source not in item["sources"]:
            item["sources"].append(source)
    for trace_ref in trace_refs or []:
        if trace_ref and trace_ref not in item["trace_refs"]:
            item["trace_refs"].append(trace_ref)


def _base_family_map(
    *,
    prompt: str,
    official_tests: str,
    entry_point: str,
) -> tuple[dict[str, dict[tuple[str, str], dict[str, Any]]], str]:
    families = _empty_family_map()
    signature, params, return_annotation = _extract_signature_data(prompt, entry_point)
    prompt_examples = extract_prompt_examples(prompt, entry_point)
    official_examples = extract_official_test_examples(official_tests)

    if params:
        _add_clause(
            families,
            "preconditions",
            kind="declared_arity",
            text=f"call uses the declared {len(params)} positional input(s)",
            sources=["signature"],
        )
        for name, kind, annotation_text in params:
            _add_clause(
                families,
                "preconditions",
                kind="typed_input",
                text=f"{name} satisfies the declared {annotation_text} input contract",
                sources=["signature"],
            )
            if kind in _MUTABLE_KINDS:
                mutation_kind = "mutation_allowed" if _allows_input_mutation(prompt) else "input_immutability"
                mutation_text = (
                    f"{name} may be updated in place when required by the prompt"
                    if mutation_kind == "mutation_allowed"
                    else f"{name} is not mutated in caller-owned state"
                )
                _add_clause(
                    families,
                    "mutation_constraints",
                    kind=mutation_kind,
                    text=mutation_text,
                    sources=["prompt_intent" if mutation_kind == "mutation_allowed" else "signature"],
                )
    else:
        _add_clause(
            families,
            "preconditions",
            kind="declared_arity",
            text="call uses the declared zero-argument input contract",
            sources=["signature"],
        )

    _add_clause(
        families,
        "invariants",
        kind="no_exception",
        text="admitted inputs execute without raising exceptions",
        sources=["verifier_default"],
    )
    _add_clause(
        families,
        "invariants",
        kind="deterministic",
        text="repeated calls on the same input stay stable",
        sources=["verifier_default"],
    )

    if return_annotation is not None:
        _add_clause(
            families,
            "postconditions",
            kind="typed_output",
            text=f"return value satisfies the declared {return_annotation} contract",
            sources=["signature"],
        )

    if prompt_examples or official_examples:
        _add_clause(
            families,
            "postconditions",
            kind="example_consistency",
            text="behavior remains consistent with the extracted prompt and official examples",
            sources=[
                "docstring_example" if prompt_examples else "",
                "official_test" if official_examples else "",
            ],
        )

    if _SORT_ASCENDING_PATTERN.search(prompt):
        _add_clause(
            families,
            "postconditions",
            kind="sorted_output",
            text=(
                "output is an ordered permutation of the primary sequence input"
                if not _SORT_DESCENDING_PATTERN.search(prompt)
                else "output is a descending ordered permutation of the primary sequence input"
            ),
            sources=["prompt_intent"],
        )

    if _REVERSE_PATTERN.search(prompt):
        _add_clause(
            families,
            "postconditions",
            kind="reverse_output",
            text="output mirrors the primary sequence or string input in reverse order",
            sources=["prompt_intent"],
        )

    for args, expected in prompt_examples:
        _add_clause(
            families,
            "oracle_hints",
            kind="prompt_example",
            text=f"{entry_point}{args!r} -> {expected!r}",
            sources=["docstring_example"],
        )

    for args, expected in official_examples:
        _add_clause(
            families,
            "oracle_hints",
            kind="official_test_example",
            text=f"{entry_point}{args!r} -> {expected!r}",
            sources=["official_test"],
        )

    return (families, signature)


def _trace_clause_mapping(property_name: str) -> tuple[str, str, str]:
    mapping = {
        "annotated_return_type": (
            "postconditions",
            "typed_output",
            "return value satisfies the declared annotation-backed output contract",
        ),
        "deterministic": (
            "invariants",
            "deterministic",
            "repeated calls on the same input stay stable",
        ),
        "input_immutability": (
            "mutation_constraints",
            "input_immutability",
            "caller-owned mutable inputs are not mutated",
        ),
        "no_exception": (
            "invariants",
            "no_exception",
            "admitted inputs execute without raising exceptions",
        ),
        "reverse_output": (
            "postconditions",
            "reverse_output",
            "output mirrors the primary sequence or string input in reverse order",
        ),
        "sorted_output": (
            "postconditions",
            "sorted_output",
            "output is an ordered permutation of the primary sequence input",
        ),
    }
    return mapping[property_name]


def _apply_trace_support(
    families: dict[str, dict[tuple[str, str], dict[str, Any]]],
    traces: list[dict[str, Any]],
) -> None:
    for trace in traces:
        trace_ref = trace["source_ref"]
        for property_name in trace["failure_properties"]:
            if property_name not in {
                "annotated_return_type",
                "deterministic",
                "input_immutability",
                "no_exception",
                "reverse_output",
                "sorted_output",
            }:
                continue
            family, kind, text = _trace_clause_mapping(property_name)
            _add_clause(
                families,
                family,
                kind=kind,
                text=text,
                sources=["trace_property"],
                trace_refs=[trace_ref],
            )

        if trace["official_test_miss"]:
            _add_clause(
                families,
                "oracle_hints",
                kind="official_test_miss_trace",
                text="checked-in additive verification surfaced a harness-passing bug",
                sources=["trace_outcome"],
                trace_refs=[trace_ref],
            )

        if trace["repaired"]:
            _add_clause(
                families,
                "oracle_hints",
                kind="repair_trace",
                text=(
                    "checked-in verify-repair recovered this task after "
                    f"{trace['repair_iterations']} repair iteration(s)"
                ),
                sources=["trace_outcome"],
                trace_refs=[trace_ref],
            )


def _finalize_family_items(
    families: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    finalized: dict[str, list[dict[str, Any]]] = {}
    for family in SPEC_FAMILIES:
        items = list(families[family].values())
        for item in items:
            item["sources"] = sorted(item["sources"])
            item["trace_refs"] = sorted(item["trace_refs"])
        finalized[family] = sorted(
            items,
            key=lambda item: (
                item["kind"],
                item["text"],
                tuple(item["sources"]),
                tuple(item["trace_refs"]),
            ),
        )
    return finalized


def _sorted_trace_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    source_refs = [trace["source_ref"] for trace in traces]
    artifacts = sorted({trace["artifact"] for trace in traces})
    failure_properties = sorted(
        {property_name for trace in traces for property_name in trace["failure_properties"]}
    )
    return {
        "artifacts": artifacts,
        "failure_properties": failure_properties,
        "official_test_miss_trace_count": sum(1 for trace in traces if trace["official_test_miss"]),
        "repaired_trace_count": sum(1 for trace in traces if trace["repaired"]),
        "source_refs": source_refs,
        "source_trace_count": len(traces),
    }


def build_corpus(repo_root: Path | None = None) -> list[dict[str, Any]]:
    root = repo_root or get_repo_root()
    rows_by_task: dict[str, dict[str, Any]] = {}

    for source_artifact in SOURCE_ARTIFACTS:
        artifact_path = resolve_path(root, source_artifact)
        artifact_label = _display_path(root, artifact_path)
        payload = load_json(artifact_path)
        case_lookup = _cohort_lookup(payload)
        per_problem = payload.get("per_problem_results")
        if not isinstance(per_problem, list):
            continue

        for result in per_problem:
            if not isinstance(result, dict):
                continue
            case_id = str(result.get("case_id") or "")
            case_meta = case_lookup.get(case_id)
            if case_meta is None:
                continue
            task_id = str(result.get("task_id") or case_meta.get("task_id") or "")
            if not task_id:
                continue

            trace = _trace_observation(
                artifact_label=artifact_label,
                payload=payload,
                case_meta=case_meta,
                result=result,
            )
            row = rows_by_task.get(task_id)
            if row is None:
                prompt = str(case_meta.get("prompt") or "")
                official_tests = str(case_meta.get("test") or "")
                entry_point = str(case_meta.get("entry_point") or result.get("entry_point") or "")
                families, signature = _base_family_map(
                    prompt=prompt,
                    official_tests=official_tests,
                    entry_point=entry_point,
                )
                row = {
                    "_dataset_idx": int(case_meta.get("dataset_idx") or result.get("dataset_idx") or 0),
                    "_families": families,
                    "case_id": str(case_meta.get("case_id") or case_id),
                    "entry_point": entry_point,
                    "row_id": f"exp236-{str(case_meta.get('case_id') or case_id)}",
                    "run_date": RUN_DATE,
                    "schema_version": SCHEMA_VERSION,
                    "signature": signature,
                    "source_traces": [],
                    "task_id": task_id,
                }
                rows_by_task[task_id] = row

            row["source_traces"].append(trace)

    finalized_rows: list[dict[str, Any]] = []
    for task_id in sorted(rows_by_task, key=lambda item: rows_by_task[item]["_dataset_idx"]):
        row = rows_by_task[task_id]
        traces = sorted(row["source_traces"], key=lambda item: item["source_ref"])
        _apply_trace_support(row["_families"], traces)
        finalized = {
            "case_id": row["case_id"],
            "entry_point": row["entry_point"],
            "row_id": row["row_id"],
            "run_date": row["run_date"],
            "schema_version": row["schema_version"],
            "signature": row["signature"],
            "source_traces": traces,
            "task_id": row["task_id"],
            "trace_summary": _sorted_trace_summary(traces),
        }
        finalized.update(_finalize_family_items(row["_families"]))
        finalized_rows.append(finalized)
    return finalized_rows


def build_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source_trace: dict[str, int] = {}
    by_spec_family = {
        family: sum(len(row[family]) for row in rows)
        for family in SPEC_FAMILIES
    }

    for row in rows:
        for trace in row["source_traces"]:
            by_source_trace[trace["artifact"]] = by_source_trace.get(trace["artifact"], 0) + 1

    return {
        "experiment": "Exp 236",
        "run_date": RUN_DATE,
        "schema_version": SCHEMA_VERSION,
        "source_artifacts": [str(path) for path in SOURCE_ARTIFACTS],
        "summary": {
            "by_source_trace": dict(sorted(by_source_trace.items())),
            "by_spec_family": by_spec_family,
            "n_official_test_miss_traces": sum(
                row["trace_summary"]["official_test_miss_trace_count"] for row in rows
            ),
            "n_repaired_traces": sum(row["trace_summary"]["repaired_trace_count"] for row in rows),
            "n_rows": len(rows),
            "n_rows_with_multi_trace_provenance": sum(
                1 for row in rows if row["trace_summary"]["source_trace_count"] > 1
            ),
            "n_rows_with_official_test_miss": sum(
                1 for row in rows if row["trace_summary"]["official_test_miss_trace_count"] > 0
            ),
            "n_rows_with_repairs": sum(
                1 for row in rows if row["trace_summary"]["repaired_trace_count"] > 0
            ),
            "n_source_traces": sum(row["trace_summary"]["source_trace_count"] for row in rows),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Exp 236 explicit code-spec corpus from checked-in traces.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_cli_default_path(CORPUS_PATH),
        help="Relative or absolute output path for the JSONL corpus.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=_cli_default_path(RESULTS_PATH),
        help="Relative or absolute output path for the summary artifact.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = get_repo_root()
    rows = build_corpus(repo_root)
    results = build_results(rows)
    write_jsonl(resolve_path(repo_root, Path(args.output)), rows)
    write_json(resolve_path(repo_root, Path(args.results)), results)
    return 0


__all__ = [
    "CORPUS_PATH",
    "RESULTS_PATH",
    "REPO_ROOT",
    "RUN_DATE",
    "SCHEMA_VERSION",
    "SOURCE_ARTIFACTS",
    "SPEC_FAMILIES",
    "build_corpus",
    "build_parser",
    "build_results",
    "get_repo_root",
    "load_json",
    "main",
    "resolve_path",
    "write_json",
    "write_jsonl",
]
