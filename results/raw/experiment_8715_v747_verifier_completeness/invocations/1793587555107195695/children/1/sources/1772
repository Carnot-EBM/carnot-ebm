#!/usr/bin/env python3
"""Exp5456 guided-decoding tautology corrigendum.

Spec refs: REQ-SAFE-5456, SCENARIO-SAFE-5456.

Exp5444 is useful evidence because its rows include exact final verifier
labels, but its headline artifact was quarantined by two top-level TAUTOLOGY
findings.  This module does not rerun generation or score a model.  It audits
the checked-in row JSONL, rebuilds the metric dependencies, and makes the
readiness decision from exact row labels instead of the prior scalar fields that
were supposed to validate themselves.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
import re
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIOR_FLAGGED_ARTIFACT = Path(
    "results/experiment_5444_gated_sota_energy_guided_decoding_v495.json"
)
PRIOR_ROW_RESULTS = Path(
    "results/experiment_5444_gated_sota_energy_guided_decoding_v495_rows.jsonl"
)
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5456_guided_decoding_tautology_corrigendum_v496.json"
)
GRAPH_RELATIVE_PATH = Path(
    "results/experiment_5456_guided_decoding_tautology_corrigendum_v496_metric_dependency_graph.json"
)
EXPERIMENT_ID = "experiment_5456_guided_decoding_tautology_corrigendum_v496"
TASK_ID = "exp5456-guided-decoding-tautology-corrigendum-v496"
MILESTONE = "2026.07.496"
RUN_DATE = "2026-07-09"
SCHEMA = "carnot.experiment_5456.guided_decoding_tautology_corrigendum.v496"
SPEC_REFS = ("REQ-SAFE-5456", "SCENARIO-SAFE-5456")
INFERENCE_SUBSTRATE = "posthoc_row_metric_audit_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked:")

CONDITIONS = ("unconstrained", "grammar_only", "verifier_potential_guided")
GUIDED_CONDITION = "verifier_potential_guided"
BASELINE_CONDITIONS = ("unconstrained", "grammar_only")
INVALID_READINESS_SOURCES = frozenset(
    {
        "metric_independence_checks_passed",
        "verifier_guided_decoding_ready",
    }
)

FIELD_PRINCIPLES: dict[str, str] = {
    "prior_flagged_artifact": "preserves the quarantined Exp5444 source.",
    "adversarial_flags_found": "records prior TAUTOLOGY findings before repair.",
    "metric_dependency_graph_path": "inspectable dependency graph written beside the artifact.",
    "invalid_tautological_fields": "exact prior scalar fields invalidated by TAUTOLOGY.",
    "recomputed_row_count": "row-level audit denominator.",
    "exact_final_labels_used": "final verifier authority only.",
    "independent_metric_fields": "clean row-evidence and derived audit fields.",
    "guided_decoding_corrigendum_clean": "clean corrigendum receipt, not a new SOTA headline.",
    "rerun_gate_reason": "why a fresh non-tautological guided-decoding run is required.",
    "inference_substrate": "posthoc row audit with no live LLM call.",
    "honest_verdict": 'terminal status; start with "complete:" or "blocked:".',
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str | None = None,
    graph_path: Path | str | None = None,
    prior_artifact: Mapping[str, Any] | None = None,
    row_records: Sequence[Mapping[str, Any]] | None = None,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5456 corrigendum artifacts."""

    root_path = Path(root)
    destination = _destination(root_path, result_path, RESULT_RELATIVE_PATH)
    graph_destination = _destination(root_path, graph_path, GRAPH_RELATIVE_PATH)
    prior = dict(prior_artifact) if prior_artifact is not None else _read_json(
        root_path / PRIOR_FLAGGED_ARTIFACT
    )
    rows = (
        [copy.deepcopy(dict(row)) for row in row_records]
        if row_records is not None
        else _read_jsonl(root_path / PRIOR_ROW_RESULTS)
    )
    artifact, graph = _assemble_artifact_and_graph(
        prior_artifact=prior,
        rows=rows,
        metric_dependency_graph_path=_artifact_path_text(root_path, graph_destination),
        tests_run=tests_run,
    )
    if write:
        _write_json(graph_destination, graph)
        _write_json(destination, artifact)
        validate_artifact(artifact, root=root_path, require_graph_file=True)
    else:
        validate_artifact(artifact, root=root_path, require_graph_file=False)
    return artifact


def build_artifact(
    *,
    prior_artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    metric_dependency_graph_path: str,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Return the terminal JSON payload without writing files."""

    artifact, _graph = _assemble_artifact_and_graph(
        prior_artifact=prior_artifact,
        rows=rows,
        metric_dependency_graph_path=metric_dependency_graph_path,
        tests_run=tests_run,
    )
    return artifact


def recompute_row_metric_audit(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute validity, false-accept, abstention, and guided deltas from rows.

    The prior artifact's scalar fields are intentionally ignored.  The only
    acceptance label used here is `exact_final_verdict.accepted` on rows whose
    authority is the deterministic exact final verifier.
    """

    row_list = [copy.deepcopy(dict(row)) for row in rows if isinstance(row, Mapping)]
    condition_rows: dict[str, list[JsonDict]] = {condition: [] for condition in CONDITIONS}
    for row in row_list:
        condition = str(row.get("condition"))
        if condition in condition_rows:
            condition_rows[condition].append(row)

    exact_labels_used = bool(row_list) and all(_exact_final_label_ok(row) for row in row_list)
    condition_validity: dict[str, JsonDict] = {}
    false_accepts_by_condition: dict[str, list[str]] = {}
    abstentions_by_condition: dict[str, list[str]] = {}
    for condition, entries in condition_rows.items():
        valid_ids = [_row_id(row) for row in entries if _exact_accepted(row)]
        false_accept_ids = [_row_id(row) for row in entries if _false_accept(row)]
        abstention_ids = [_row_id(row) for row in entries if row.get("parse_status") == "abstained"]
        condition_validity[condition] = {
            "total": len(entries),
            "valid_count": len(valid_ids),
            "invalid_count": len(entries) - len(valid_ids),
            "rate": _rate(len(valid_ids), len(entries)),
            "exact_valid_row_ids": valid_ids,
        }
        false_accepts_by_condition[condition] = false_accept_ids
        abstentions_by_condition[condition] = abstention_ids

    guided = condition_validity[GUIDED_CONDITION]
    unconstrained = condition_validity["unconstrained"]
    grammar = condition_validity["grammar_only"]
    guided_vs_unconstrained = guided["rate"] - unconstrained["rate"]
    guided_vs_grammar = guided["rate"] - grammar["rate"]
    guided_false_accept_count = len(false_accepts_by_condition[GUIDED_CONDITION])
    guided_ready = bool(
        exact_labels_used
        and guided["total"] > 0
        and guided_vs_unconstrained > 0.0
        and guided_vs_grammar > 0.0
        and guided_false_accept_count == 0
    )
    family_counts = Counter(str(row.get("constraint_family")) for row in row_list)
    return {
        "row_count": len(row_list),
        "exact_final_labels_used": exact_labels_used,
        "condition_validity": condition_validity,
        "false_accepts_by_condition": false_accepts_by_condition,
        "abstentions_by_condition": abstentions_by_condition,
        "false_accept_rate_by_condition": {
            condition: _rate(len(ids), len(condition_rows[condition]))
            for condition, ids in false_accepts_by_condition.items()
        },
        "abstention_rate_by_condition": {
            condition: _rate(len(ids), len(condition_rows[condition]))
            for condition, ids in abstentions_by_condition.items()
        },
        "guided_delta_audit": {
            "vs_unconstrained": {
                "baseline_condition": "unconstrained",
                "valid_count_delta": guided["valid_count"] - unconstrained["valid_count"],
                "rate_delta": guided_vs_unconstrained,
            },
            "vs_grammar_only": {
                "baseline_condition": "grammar_only",
                "valid_count_delta": guided["valid_count"] - grammar["valid_count"],
                "rate_delta": guided_vs_grammar,
            },
        },
        "guided_condition_false_accept_count": guided_false_accept_count,
        "guided_ready_from_rows": guided_ready,
        "constraint_family_counts": dict(sorted(family_counts.items())),
        "row_authority_failures": [
            _row_id(row) for row in row_list if not _exact_final_label_ok(row)
        ],
        "prior_scalar_fields_used": [],
    }


def extract_tautology_flags(prior_artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Return prior TAUTOLOGY flags from the Exp5444 adversarial stamp."""

    flags: list[JsonDict] = []
    for item in prior_artifact.get("corrigendum_pending", []):
        if not isinstance(item, Mapping) or item.get("kind") != "TAUTOLOGY":
            continue
        flags.append(
            {
                "kind": str(item.get("kind")),
                "severity": str(item.get("severity", "")),
                "detail": str(item.get("detail", "")),
            }
        )
    return flags


def invalid_fields_from_tautology_flags(flags: Sequence[Mapping[str, Any]]) -> list[str]:
    """Extract both field names from each TAUTOLOGY detail string."""

    fields: set[str] = set()
    for flag in flags:
        detail = str(flag.get("detail", ""))
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)=.* and ([A-Za-z_][A-Za-z0-9_]*)=", detail)
        if match:
            fields.update(match.groups())
    return sorted(fields)


def build_metric_dependency_graph(
    *,
    prior_artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    row_audit: Mapping[str, Any],
    adversarial_flags: Sequence[Mapping[str, Any]],
    invalid_tautological_fields: Sequence[str],
) -> JsonDict:
    """Build the metric dependency graph used by the corrigendum receipt."""

    invalid = set(invalid_tautological_fields)
    nodes: dict[str, JsonDict] = {
        "row.condition": _node("independent", [], ["row_counts"], "condition label from row JSONL"),
        "row.constraint_family": _node(
            "independent",
            [],
            ["row_counts"],
            "fixture family from row JSONL",
        ),
        "row.condition_advisory_accept": _node(
            "independent",
            [],
            ["derived_aggregates"],
            "parse/advisory accept recorded per row, never final authority",
        ),
        "row.parse_status": _node(
            "independent",
            [],
            ["row_counts"],
            "row parse status used only for abstention accounting",
        ),
        "row.exact_final_verdict.accepted": _node(
            "independent",
            [],
            ["exact_final_verifier_labels"],
            "deterministic exact final verifier label",
        ),
        "row.exact_final_verdict.verified": _node(
            "independent",
            [],
            ["exact_final_verifier_labels"],
            "row confirms exact verifier ran",
        ),
        "row.final_authority_bypassed": _node(
            "independent",
            [],
            ["exact_final_verifier_labels"],
            "must be false for every row",
        ),
        "row.reward_evaluation_count": _node(
            "independent",
            [],
            ["guided_rewards"],
            "cost accounting row field, not a readiness label",
        ),
        "corrected.condition_validity": _node(
            "derived-from-independent",
            ["row.condition", "row.exact_final_verdict.accepted"],
            ["baselines", "exact_final_verifier_labels", "row_counts"],
            "validity counts and rates by condition",
        ),
        "corrected.false_accepts_by_condition": _node(
            "derived-from-independent",
            [
                "row.condition",
                "row.condition_advisory_accept",
                "row.exact_final_verdict.accepted",
            ],
            ["exact_final_verifier_labels", "derived_aggregates", "row_counts"],
            "advisory accepts rejected by exact final verifier",
        ),
        "corrected.abstentions_by_condition": _node(
            "derived-from-independent",
            ["row.condition", "row.parse_status", "row.exact_final_verdict.verified"],
            ["exact_final_verifier_labels", "row_counts"],
            "abstention counts by condition",
        ),
        "corrected.guided_delta_vs_unconstrained": _node(
            "derived-from-independent",
            ["corrected.condition_validity"],
            ["baselines", "exact_final_verifier_labels", "row_counts"],
            "guided accepted-validity rate minus unconstrained accepted-validity rate",
        ),
        "corrected.guided_delta_vs_grammar_only": _node(
            "derived-from-independent",
            ["corrected.condition_validity"],
            ["baselines", "exact_final_verifier_labels", "row_counts"],
            "guided accepted-validity rate minus grammar-only accepted-validity rate",
        ),
        "corrected.guided_decoding_ready_from_rows": _node(
            "derived-from-independent",
            [
                "corrected.condition_validity",
                "corrected.false_accepts_by_condition",
                "corrected.guided_delta_vs_unconstrained",
                "corrected.guided_delta_vs_grammar_only",
            ],
            ["baselines", "exact_final_verifier_labels", "row_counts", "derived_aggregates"],
            "clean readiness recomputed from row-derived metrics only",
        ),
    }
    for field in _prior_metric_fields(prior_artifact):
        node_name = f"prior.{field}"
        if field in invalid:
            classification = "invalid-tautological"
            reason = "prior field appears in a TAUTOLOGY adversarial finding"
        elif field in INVALID_READINESS_SOURCES:
            classification = "invalid-tautological"
            reason = "prior readiness validator failed to catch the TAUTOLOGY metric surface"
        else:
            classification = "derived-from-independent"
            reason = "prior aggregate preserved as context only, not readiness evidence"
        nodes[node_name] = _node(
            classification,
            _prior_dependencies_for(field),
            _prior_dependency_kinds_for(field),
            reason,
        )

    readiness_dependencies = list(nodes["corrected.guided_decoding_ready_from_rows"]["depends_on"])
    graph: JsonDict = {
        "schema": "carnot.experiment_5456.metric_dependency_graph.v1",
        "prior_flagged_artifact": str(PRIOR_FLAGGED_ARTIFACT),
        "prior_row_results": str(PRIOR_ROW_RESULTS),
        "prior_flagged_adversarial": bool(prior_artifact.get("flagged_adversarial")),
        "prior_tautology_flags": [dict(flag) for flag in adversarial_flags],
        "invalid_tautological_fields": sorted(invalid),
        "row_count": int(row_audit.get("row_count", len(rows))),
        "nodes": nodes,
        "readiness_field": "corrected.guided_decoding_ready_from_rows",
        "readiness_dependencies": readiness_dependencies,
        "forbidden_readiness_dependencies": sorted(
            {f"prior.{field}" for field in invalid}.union(
                {f"prior.{field}" for field in INVALID_READINESS_SOURCES}
            )
        ),
    }
    graph["readiness_dependency_errors"] = audit_readiness_dependencies(graph)
    graph["readiness_dependencies_clean"] = not graph["readiness_dependency_errors"]
    graph["independent_metric_fields"] = [
        name
        for name, node in sorted(nodes.items())
        if not name.startswith("prior.")
        and node.get("classification") in {"independent", "derived-from-independent"}
    ]
    return graph


def audit_readiness_dependencies(
    graph: Mapping[str, Any],
    readiness_field: str = "corrected.guided_decoding_ready_from_rows",
) -> list[str]:
    """Return errors when readiness depends on prior or invalid scalar fields."""

    nodes = graph.get("nodes")
    if not isinstance(nodes, Mapping) or readiness_field not in nodes:
        return [f"readiness field missing from dependency graph: {readiness_field}"]
    dependencies = _transitive_dependencies(nodes, readiness_field)
    errors: list[str] = []
    if readiness_field in dependencies:
        errors.append(f"self-validating readiness dependency: {readiness_field}")
    invalid_bare = set(str(field) for field in graph.get("invalid_tautological_fields", []))
    forbidden_bare = invalid_bare.union(INVALID_READINESS_SOURCES)
    forbidden_full = {f"prior.{field}" for field in forbidden_bare}
    forbidden_hits = sorted(
        dep for dep in dependencies if dep in forbidden_full or _bare_field(dep) in forbidden_bare
    )
    if forbidden_hits:
        errors.append(f"forbidden readiness dependency: {forbidden_hits}")
    invalid_nodes = sorted(
        dep
        for dep in dependencies
        if isinstance(nodes.get(dep), Mapping)
        and nodes[dep].get("classification") == "invalid-tautological"
    )
    if invalid_nodes:
        errors.append(f"readiness depends on invalid-tautological nodes: {invalid_nodes}")
    return errors


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    root: Path | str = REPO_ROOT,
    require_graph_file: bool = True,
) -> None:
    """Raise when the Exp5456 corrigendum cannot support downstream use."""

    errors = artifact_schema_errors(
        artifact,
        root=root,
        require_graph_file=require_graph_file,
    )
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(
    artifact: Mapping[str, Any],
    *,
    root: Path | str = REPO_ROOT,
    require_graph_file: bool = True,
) -> list[str]:
    """Return schema and dependency-validation errors for an Exp5456 artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("prior_flagged_artifact") != str(PRIOR_FLAGGED_ARTIFACT):
        errors.append("prior_flagged_artifact must point to Exp5444")
    if not isinstance(artifact.get("adversarial_flags_found"), list):
        errors.append("adversarial_flags_found must be a list")
    elif artifact.get("guided_decoding_corrigendum_clean") is True and not artifact[
        "adversarial_flags_found"
    ]:
        errors.append("guided_decoding_corrigendum_clean requires prior TAUTOLOGY flags")
    if not isinstance(artifact.get("invalid_tautological_fields"), list):
        errors.append("invalid_tautological_fields must be a list")
    elif artifact.get("guided_decoding_corrigendum_clean") is True and not artifact[
        "invalid_tautological_fields"
    ]:
        errors.append("guided_decoding_corrigendum_clean requires invalid fields")
    if not _bare_non_negative_int(artifact.get("recomputed_row_count")):
        errors.append("recomputed_row_count must be a non-negative integer")
    for field in ("exact_final_labels_used", "guided_decoding_corrigendum_clean"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be boolean")
    if not isinstance(artifact.get("independent_metric_fields"), list):
        errors.append("independent_metric_fields must be a list")
    if not isinstance(artifact.get("rerun_gate_reason"), str) or not artifact.get(
        "rerun_gate_reason"
    ):
        errors.append("rerun_gate_reason must be a non-empty string")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be posthoc_row_metric_audit_no_llm")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or "\n" in verdict or not verdict.startswith(
        TERMINAL_PREFIXES
    ):
        errors.append("honest_verdict must start with complete: or blocked:")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-SAFE-5456")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py must not be modified")
    if artifact.get("guided_decoding_corrigendum_clean") is True:
        if artifact.get("status") != "complete":
            errors.append("guided_decoding_corrigendum_clean requires complete status")
        if artifact.get("exact_final_labels_used") is not True:
            errors.append("guided_decoding_corrigendum_clean requires exact final labels")
        if int(artifact.get("recomputed_row_count") or 0) <= 0:
            errors.append("guided_decoding_corrigendum_clean requires recomputed rows")
    errors.extend(_graph_file_errors(artifact, Path(root), require_graph_file))
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for writing the Exp5456 deliverable JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=None)
    parser.add_argument("--graph-path", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run(
        root=args.root,
        result_path=args.result_path,
        graph_path=args.graph_path,
        write=True,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["guided_decoding_corrigendum_clean"] else 1


def _assemble_artifact_and_graph(
    *,
    prior_artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    metric_dependency_graph_path: str,
    tests_run: Sequence[str | Mapping[str, Any]],
) -> tuple[JsonDict, JsonDict]:
    flags = extract_tautology_flags(prior_artifact)
    invalid_fields = invalid_fields_from_tautology_flags(flags)
    row_audit = recompute_row_metric_audit(rows)
    graph = build_metric_dependency_graph(
        prior_artifact=prior_artifact,
        rows=rows,
        row_audit=row_audit,
        adversarial_flags=flags,
        invalid_tautological_fields=invalid_fields,
    )
    readiness_errors = audit_readiness_dependencies(graph)
    blockers = _blockers(prior_artifact, rows, flags, row_audit, graph, readiness_errors)
    clean = not blockers
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "status": "complete" if clean else "blocked",
        "prior_flagged_artifact": str(PRIOR_FLAGGED_ARTIFACT),
        "prior_flagged_adversarial": bool(prior_artifact.get("flagged_adversarial")),
        "prior_row_results": str(PRIOR_ROW_RESULTS),
        "adversarial_flags_found": flags,
        "metric_dependency_graph_path": metric_dependency_graph_path,
        "invalid_tautological_fields": invalid_fields,
        "recomputed_row_count": int(row_audit["row_count"]),
        "exact_final_labels_used": bool(row_audit["exact_final_labels_used"]),
        "independent_metric_fields": graph["independent_metric_fields"],
        "guided_decoding_corrigendum_clean": clean,
        "guided_decoding_ready_from_independent_rows": bool(row_audit["guided_ready_from_rows"]),
        "rerun_gate_reason": _rerun_gate_reason(row_audit, invalid_fields),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(clean, blockers, row_audit),
        "row_metric_audit": row_audit,
        "metric_classification_summary": {
            "independent": [
                name
                for name, node in graph["nodes"].items()
                if node["classification"] == "independent"
            ],
            "derived_from_independent": [
                name
                for name, node in graph["nodes"].items()
                if node["classification"] == "derived-from-independent"
            ],
            "invalid_tautological": [
                name
                for name, node in graph["nodes"].items()
                if node["classification"] == "invalid-tautological"
            ],
        },
        "blockers": blockers,
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalise_tests_run(tests_run),
        "research_conductor_modified": False,
    }
    return artifact, graph


def _blockers(
    prior_artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    flags: Sequence[Mapping[str, Any]],
    row_audit: Mapping[str, Any],
    graph: Mapping[str, Any],
    readiness_errors: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    if prior_artifact.get("flagged_adversarial") is not True:
        blockers.append("prior_artifact_not_flagged_adversarial")
    if not flags:
        blockers.append("prior_tautology_flags_missing")
    if not rows:
        blockers.append("row_evidence_missing")
    if row_audit.get("exact_final_labels_used") is not True:
        blockers.append("exact_final_labels_missing_or_bypassed")
    if graph.get("readiness_dependencies_clean") is not True or readiness_errors:
        blockers.append("readiness_dependency_not_independent")
    return sorted(set(blockers))


def _graph_file_errors(
    artifact: Mapping[str, Any],
    root: Path,
    require_graph_file: bool,
) -> list[str]:
    graph_text = artifact.get("metric_dependency_graph_path")
    if not isinstance(graph_text, str) or not graph_text:
        return ["metric_dependency_graph_path must be a non-empty string"]
    if not require_graph_file:
        return []
    graph_path = Path(graph_text)
    if not graph_path.is_absolute():
        graph_path = root / graph_path
    if not graph_path.is_file():
        return ["metric_dependency_graph_path must point to a written graph"]
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"metric_dependency_graph_path is unreadable: {type(exc).__name__}: {exc}"]
    errors = audit_readiness_dependencies(graph)
    if graph.get("invalid_tautological_fields") != artifact.get("invalid_tautological_fields"):
        errors.append("metric dependency graph invalid fields must match artifact")
    if graph.get("readiness_dependencies_clean") is not True:
        errors.append("metric dependency graph readiness dependencies must be clean")
    return errors


def _prior_metric_fields(prior_artifact: Mapping[str, Any]) -> list[str]:
    candidates = {
        "accepted_validity_rate",
        "abstention_rate",
        "action_unreachability_rate",
        "condition_metrics",
        "guided_validity_delta_vs_grammar_only",
        "guided_validity_delta_vs_unconstrained",
        "metric_independence_checks_passed",
        "semantic_false_accept_rate",
        "unsafe_false_accept_rate",
        "verifier_guided_decoding_ready",
    }
    return sorted(field for field in candidates if field in prior_artifact)


def _prior_dependencies_for(field: str) -> list[str]:
    if field == "metric_independence_checks_passed":
        return ["prior.metric_details", "prior.row_checksums"]
    if field == "verifier_guided_decoding_ready":
        return [
            "prior.metric_independence_checks_passed",
            "prior.exact_final_authority",
            "prior.gpu_offload_verified",
            "prior.row_results",
        ]
    if "guided_validity_delta" in field:
        return ["prior.condition_metrics", "prior.row_results"]
    if field == "condition_metrics":
        return ["prior.row_results"]
    return ["prior.row_results"]


def _prior_dependency_kinds_for(field: str) -> list[str]:
    kinds = {"derived_aggregates", "row_counts"}
    if field in {"guided_validity_delta_vs_unconstrained", "guided_validity_delta_vs_grammar_only"}:
        kinds.update({"baselines", "exact_final_verifier_labels"})
    if "false_accept" in field or field in {"accepted_validity_rate", "condition_metrics"}:
        kinds.add("exact_final_verifier_labels")
    if field == "verifier_guided_decoding_ready":
        kinds.update({"baselines", "exact_final_verifier_labels", "guided_rewards"})
    if field == "metric_independence_checks_passed":
        kinds.add("derived_aggregates")
    return sorted(kinds)


def _node(
    classification: str,
    depends_on: Sequence[str],
    dependency_kinds: Sequence[str],
    reason: str,
) -> JsonDict:
    return {
        "classification": classification,
        "depends_on": list(depends_on),
        "dependency_kinds": sorted(set(dependency_kinds)),
        "reason": reason,
    }


def _transitive_dependencies(
    nodes: Mapping[str, Any],
    field: str,
    seen: set[str] | None = None,
) -> set[str]:
    seen = set() if seen is None else seen
    node = nodes.get(field)
    if not isinstance(node, Mapping):
        return seen
    for dependency in node.get("depends_on", []):
        dep = str(dependency)
        if dep in seen:
            continue
        seen.add(dep)
        _transitive_dependencies(nodes, dep, seen)
    return seen


def _false_accept(row: Mapping[str, Any]) -> bool:
    return row.get("condition_advisory_accept") is True and not _exact_accepted(row)


def _exact_accepted(row: Mapping[str, Any]) -> bool:
    return _mapping(row.get("exact_final_verdict")).get("accepted") is True


def _exact_final_label_ok(row: Mapping[str, Any]) -> bool:
    verdict = _mapping(row.get("exact_final_verdict"))
    return bool(
        verdict.get("verified") is True
        and verdict.get("authority") == "exact_final_verifier"
        and row.get("final_authority_bypassed") is False
        and row.get("accepted_by_final_authority") is verdict.get("accepted")
    )


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id") or row.get("fixture_row_id") or "unknown")


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _rerun_gate_reason(row_audit: Mapping[str, Any], invalid_fields: Sequence[str]) -> str:
    if row_audit.get("guided_ready_from_rows") is True and not invalid_fields:
        return "no rerun required by this audit"
    return (
        "prior Exp5444 has invalid tautological top-level metrics; row recompute shows "
        "guided readiness is not headline-clean, so require a fresh non-tautological "
        "guided-decoding rerun before any SOTA guided-decoding result headlines"
    )


def _honest_verdict(
    clean: bool,
    blockers: Sequence[str],
    row_audit: Mapping[str, Any],
) -> str:
    if clean:
        if row_audit.get("guided_ready_from_rows") is True:
            return "complete: guided-decoding corrigendum clean with row-derived readiness"
        return "complete: guided-decoding corrigendum clean; Exp5444 headline readiness blocked"
    return "blocked: " + ",".join(blockers or ["guided_decoding_corrigendum_clean_false"])


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bare_field(field: str) -> str:
    return field.rsplit(".", 1)[-1]


def _bare_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _destination(root: Path, path: Path | str | None, default: Path) -> Path:
    if path is None:
        return root / default
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _artifact_path_text(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [dict(json.loads(line)) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def _normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    if not tests_run:
        return [{"command": "not_recorded", "outcome": "not_recorded"}]
    return [_normalise_test_run(item) for item in tests_run]


def _normalise_test_run(item: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(item, str):
        return {"command": item, "outcome": "not_recorded"}
    return {
        "command": str(item.get("command", "not_recorded")),
        "outcome": str(item.get("outcome", "not_recorded")),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
