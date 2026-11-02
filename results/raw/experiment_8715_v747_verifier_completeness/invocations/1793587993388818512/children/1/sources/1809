"""Exp5502 CSL metric-independence tautology corrigendum.

Spec refs: REQ-LEARN-5502,
SCENARIO-LEARN-5502-METRIC-GRAPH,
SCENARIO-LEARN-5502-CROSS-CHECK,
SCENARIO-LEARN-5502-ARTIFACT.

This module does not rerun local SOTA GGUF inference. It reads the prior CSL
artifacts, reconstructs which fields were policy inputs versus final
validator outcomes, and emits the downstream decision that Exp5474 cannot be a
clean CSL scale headline until a same-scope rerun separates those surfaces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5502_csl_tautology_static_corrigendum_v499.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5502_csl_tautology_static_corrigendum_v499.py"
)
EXP5474_RELATIVE_PATH = Path("results/experiment_5474_sota_csl_scale_v497.json")
EXP5473_RELATIVE_PATH = Path("results/experiment_5473_csl_kan_surrogate_assurance_v497.json")
EXP5475_RELATIVE_PATH = Path("results/experiment_5475_csl_behavioral_memory_ladder_v497.json")
EXP5461_RELATIVE_PATH = Path("results/experiment_5461_gated_sota_csl_memory_routing_v496.json")

EXPERIMENT_ID = "experiment_5502_csl_tautology_static_corrigendum_v499"
TASK_ID = "exp5502-csl-tautology-static-corrigendum-v499"
MILESTONE = "2026.07.499"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5502
SCHEMA = "carnot.experiment_5502.csl_tautology_static_corrigendum.v499"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")
SPEC_REFS = (
    "REQ-LEARN-5502",
    "SCENARIO-LEARN-5502-METRIC-GRAPH",
    "SCENARIO-LEARN-5502-CROSS-CHECK",
    "SCENARIO-LEARN-5502-ARTIFACT",
)

CLASSIFICATIONS = {
    "upstream feature",
    "policy decision",
    "evaluator outcome",
    "baseline outcome",
    "derived summary",
}
RECOMMENDATIONS = {
    "clean",
    "bounded_requires_rerun",
    "retire_same_scope_if_repeated",
}

PRIOR_SUCCESS_FIELD = "experiment_5473.surrogate_rows.features.prior_success"
QUALITY_SCORE_SURFACE = "condition_metrics.*.quality_score / row.accepted_by_final_authority"
POLICY_ACCEPT_FIELD = "experiment_5473.surrogate_rows.surrogate_accept"
HEADLINE_OUTCOME_FIELDS = (
    "experiment_5474.kan_assured_csl_score",
    "experiment_5474.exact_validator_pass_rate",
    "experiment_5474.csl_scale_ready",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "audited_artifacts": "Exact upstream artifacts read by the static corrigendum.",
    "metric_graph_nodes": "Classified policy, outcome, baseline, and summary dependency graph.",
    "policy_score_fields": "Fields that can affect action or memory-route acceptance.",
    "outcome_metric_fields": "Fields that report exact-validator, baseline, or derived outcomes.",
    "independence_violations": "Reasons Exp5474 cannot be clean headline evidence.",
    "metric_independence_clean": "False when policy-score inputs overlap headline outcomes.",
    "tautology_flag_resolved": "True only when the prior TAUTOLOGY is adjudicated downstream.",
    "csl_scale_headline_allowed": "False unless Exp5474 can be cited as clean scale evidence.",
    "downstream_recommendation": "clean, bounded_requires_rerun, or retire_same_scope_if_repeated.",
    "retire_same_scope_if_repeated": "Whether a repeated same-scope coupling should retire the lane.",
    "inference_substrate": "Aggregation-only audit; no local SOTA inference is run.",
    "honest_verdict": "Terminal summary starting with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp5502 corrigendum JSON."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    validate_artifact(artifact)
    if write:
        destination = Path(result_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Return the deterministic static corrigendum from upstream artifacts."""

    root_path = Path(root)
    exp5474 = _read_json(root_path / EXP5474_RELATIVE_PATH)
    exp5473 = _read_json(root_path / EXP5473_RELATIVE_PATH)
    exp5475 = _read_json(root_path / EXP5475_RELATIVE_PATH)
    exp5461 = _read_json(root_path / EXP5461_RELATIVE_PATH)

    metric_graph_nodes = build_metric_graph_nodes()
    policy_score_fields = [
        node["field"]
        for node in metric_graph_nodes
        if node["classification"] in {"upstream feature", "policy decision"}
    ]
    outcome_metric_fields = [
        node["field"]
        for node in metric_graph_nodes
        if node["classification"] in {"evaluator outcome", "baseline outcome", "derived summary"}
    ]
    tautology_fields = adversarial_tautology_fields(exp5474)
    violations = independence_violations(
        exp5474=exp5474,
        exp5473=exp5473,
        exp5461=exp5461,
        tautology_fields=tautology_fields,
    )
    cross_check = cross_check_summary(
        exp5474=exp5474,
        exp5473=exp5473,
        exp5475=exp5475,
        tautology_fields=tautology_fields,
    )
    clean = not violations
    recommendation = downstream_recommendation(
        metric_independence_clean=clean,
        bounded_behavioral_memory_support=bool(
            cross_check["exp5475"]["bounded_behavioral_memory_support"]
        ),
    )
    headline_allowed = clean and recommendation == "clean"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "status": "complete",
        "audited_artifacts": audited_artifacts(),
        "metric_graph_nodes": metric_graph_nodes,
        "policy_score_fields": policy_score_fields,
        "outcome_metric_fields": outcome_metric_fields,
        "independence_violations": violations,
        "metric_independence_clean": clean,
        "tautology_flag_resolved": bool(tautology_fields and not headline_allowed),
        "csl_scale_headline_allowed": headline_allowed,
        "downstream_recommendation": recommendation,
        "retire_same_scope_if_repeated": recommendation != "clean",
        "cross_check_summary": cross_check,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(
            metric_independence_clean=clean,
            recommendation=recommendation,
        ),
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalise_tests_run(tests_run),
        "source_files": {
            "module": str(MODULE_RELATIVE_PATH),
            "spec": str(SPEC_RELATIVE_PATH),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return _json_ready(artifact)


def audited_artifacts() -> list[str]:
    """Return the upstream artifacts read by this static audit."""

    return [
        str(EXP5474_RELATIVE_PATH),
        str(EXP5473_RELATIVE_PATH),
        str(EXP5475_RELATIVE_PATH),
        str(EXP5461_RELATIVE_PATH),
    ]


def build_metric_graph_nodes() -> list[JsonDict]:
    """Classify the Exp5474 policy/outcome metric graph."""

    nodes = [
        _node(
            "experiment_5473.surrogate_rows.features.context_cost",
            "upstream feature",
            str(EXP5473_RELATIVE_PATH),
            ["experiment_5461.row_results.context_cost"],
            "KAN surrogate cost input.",
        ),
        _node(
            "experiment_5473.surrogate_rows.features.verifier_cost",
            "upstream feature",
            str(EXP5473_RELATIVE_PATH),
            ["experiment_5461.row_results.verifier_cost"],
            "KAN surrogate verifier-cost input.",
        ),
        _node(
            PRIOR_SUCCESS_FIELD,
            "upstream feature",
            str(EXP5473_RELATIVE_PATH),
            ["experiment_5461.condition_metrics.*.quality_score"],
            "Policy-score input copied from exact-validator quality score.",
        ),
        _node(
            "experiment_5473.surrogate_rows.features.conflict_risk",
            "upstream feature",
            str(EXP5473_RELATIVE_PATH),
            ["experiment_5461.memory_receipt.negative_transfer_candidate"],
            "Conflict-risk input for memory-route scoring.",
        ),
        _node(
            "experiment_5473.surrogate_rows.features.memory_age",
            "upstream feature",
            str(EXP5473_RELATIVE_PATH),
            ["experiment_5461.memory_receipt.stale_memory_candidate"],
            "Stale-memory input for memory-route scoring.",
        ),
        _node(
            "experiment_5473.surrogate_rows.features.constraint_violation_history",
            "upstream feature",
            str(EXP5473_RELATIVE_PATH),
            ["experiment_5461.row_results.negative_transfer_detected"],
            "Risk input derived from prior unsafe memory behavior.",
        ),
        _node(
            "experiment_5473.surrogate_rows.surrogate_score",
            "policy decision",
            str(EXP5473_RELATIVE_PATH),
            [
                "experiment_5473.surrogate_rows.features.context_cost",
                "experiment_5473.surrogate_rows.features.verifier_cost",
                PRIOR_SUCCESS_FIELD,
                "experiment_5473.surrogate_rows.features.conflict_risk",
                "experiment_5473.surrogate_rows.features.memory_age",
                "experiment_5473.surrogate_rows.features.constraint_violation_history",
            ],
            "Additive KAN-style score used for acceptance.",
        ),
        _node(
            "experiment_5473.surrogate_rows.acceptance_threshold",
            "policy decision",
            str(EXP5473_RELATIVE_PATH),
            [
                "experiment_5473.surrogate_rows.features.conflict_risk",
                "experiment_5473.surrogate_rows.features.memory_age",
                "experiment_5473.surrogate_rows.features.constraint_violation_history",
            ],
            "Risk-adjusted policy threshold.",
        ),
        _node(
            "experiment_5473.surrogate_rows.acceptance_margin",
            "policy decision",
            str(EXP5473_RELATIVE_PATH),
            [
                "experiment_5473.surrogate_rows.surrogate_score",
                "experiment_5473.surrogate_rows.acceptance_threshold",
            ],
            "Score minus threshold.",
        ),
        _node(
            POLICY_ACCEPT_FIELD,
            "policy decision",
            str(EXP5473_RELATIVE_PATH),
            ["experiment_5473.surrogate_rows.acceptance_margin"],
            "Boolean route acceptance from surrogate margin.",
        ),
        _node(
            "experiment_5474.panel_rows.threshold_offset",
            "policy decision",
            str(EXP5474_RELATIVE_PATH),
            ["experiment_5473.surrogate_rows.acceptance_threshold"],
            "Conservative KAN threshold offset copied into Exp5474 rows.",
        ),
        _node(
            "experiment_5474.panel_rows.surrogate_acceptance_margin",
            "policy decision",
            str(EXP5474_RELATIVE_PATH),
            ["experiment_5473.surrogate_rows.acceptance_margin"],
            "KAN margin copied into Exp5474 rows.",
        ),
        _node(
            "experiment_5474.panel_rows.memory_decision",
            "policy decision",
            str(EXP5474_RELATIVE_PATH),
            [POLICY_ACCEPT_FIELD, "experiment_5461.row_results.memory_receipt"],
            "Effective no-memory, naive, or governed-memory route.",
        ),
        _node(
            "experiment_5474.panel_rows.action_decision.selected_answer",
            "policy decision",
            str(EXP5474_RELATIVE_PATH),
            ["experiment_5474.panel_rows.memory_decision"],
            "Selected action/answer before final validator aggregation.",
        ),
        _node(
            "experiment_5474.panel_rows.accepted_by_final_authority",
            "evaluator outcome",
            str(EXP5474_RELATIVE_PATH),
            ["experiment_5461.row_results.exact_verifier_witness.accepted"],
            "Exact task verifier outcome copied per row.",
        ),
        _node(
            "experiment_5474.panel_rows.action_decision.downstream_action_passed",
            "evaluator outcome",
            str(EXP5474_RELATIVE_PATH),
            ["experiment_5474.panel_rows.accepted_by_final_authority"],
            "Downstream action pass derived from exact row acceptance.",
        ),
        _node(
            "experiment_5474.no_memory_score",
            "baseline outcome",
            str(EXP5474_RELATIVE_PATH),
            ["experiment_5474.panel_rows.accepted_by_final_authority"],
            "Exact-validator no-memory baseline.",
        ),
        _node(
            "experiment_5474.naive_icl_score",
            "baseline outcome",
            str(EXP5474_RELATIVE_PATH),
            ["experiment_5474.panel_rows.accepted_by_final_authority"],
            "Exact-validator naive-ICL baseline.",
        ),
        _node(
            "experiment_5474.kan_assured_csl_score",
            "evaluator outcome",
            str(EXP5474_RELATIVE_PATH),
            ["experiment_5474.panel_rows.accepted_by_final_authority"],
            "Exact-validator KAN-assured CSL score.",
        ),
        _node(
            "experiment_5474.exact_validator_pass_rate",
            "evaluator outcome",
            str(EXP5474_RELATIVE_PATH),
            ["experiment_5474.kan_assured_csl_score"],
            "Alias of the KAN condition exact-validator score.",
        ),
        _node(
            "experiment_5474.negative_transfer_deflection_rate",
            "evaluator outcome",
            str(EXP5474_RELATIVE_PATH),
            ["experiment_5473.negative_transfer_deflection_rate"],
            "Risky memory rows deflected according to exact-validator outcomes.",
        ),
        _node(
            "experiment_5474.delta_vs_no_memory",
            "derived summary",
            str(EXP5474_RELATIVE_PATH),
            ["experiment_5474.kan_assured_csl_score", "experiment_5474.no_memory_score"],
            "Reported KAN minus no-memory delta.",
        ),
        _node(
            "experiment_5474.delta_vs_naive_icl",
            "derived summary",
            str(EXP5474_RELATIVE_PATH),
            ["experiment_5474.kan_assured_csl_score", "experiment_5474.naive_icl_score"],
            "Reported KAN minus naive-ICL delta.",
        ),
        _node(
            "experiment_5474.csl_scale_ready",
            "derived summary",
            str(EXP5474_RELATIVE_PATH),
            [
                "experiment_5474.kan_assured_csl_score",
                "experiment_5474.exact_validator_pass_rate",
                "experiment_5474.negative_transfer_deflection_rate",
            ],
            "Headline readiness gate.",
        ),
        _node(
            "experiment_5475.axis_details.support_removal.accepted_by_exact_validator",
            "evaluator outcome",
            str(EXP5475_RELATIVE_PATH),
            ["experiment_5475.row_results.exact_validator_results.accepted"],
            "Independent deterministic replay support-removal validator.",
        ),
        _node(
            "experiment_5475.axis_details.conflict_handling.accepted_by_exact_validator",
            "evaluator outcome",
            str(EXP5475_RELATIVE_PATH),
            ["experiment_5475.row_results.exact_validator_results.accepted"],
            "Independent deterministic replay conflict validator.",
        ),
        _node(
            "experiment_5475.axis_details.downstream_action_use.accepted_by_exact_validator",
            "evaluator outcome",
            str(EXP5475_RELATIVE_PATH),
            ["experiment_5475.row_results.exact_validator_results.accepted"],
            "Independent deterministic replay downstream-action validator.",
        ),
    ]
    return sorted(nodes, key=lambda item: str(item["field"]))


def independence_violations(
    *,
    exp5474: Mapping[str, Any],
    exp5473: Mapping[str, Any],
    exp5461: Mapping[str, Any],
    tautology_fields: Sequence[str],
) -> list[JsonDict]:
    """Return metric-independence violations found in the audited artifacts."""

    violations: list[JsonDict] = []
    prior_success_values = {
        float(_mapping(row.get("features")).get("prior_success", -1.0))
        for row in _rows(exp5473.get("surrogate_rows"))
        if row.get("condition") == "policy_selected"
    }
    policy_quality = _quality_score(exp5461, "policy_selected")
    headline_values = {
        float(exp5474.get("kan_assured_csl_score", -2.0)),
        float(exp5474.get("exact_validator_pass_rate", -3.0)),
    }
    if policy_quality in prior_success_values and policy_quality in headline_values:
        violations.append(
            {
                "kind": "policy_outcome_scalar_overlap",
                "policy_score_field": PRIOR_SUCCESS_FIELD,
                "policy_decision_field": POLICY_ACCEPT_FIELD,
                "headline_outcome_fields": list(HEADLINE_OUTCOME_FIELDS),
                "shared_scalar_family": QUALITY_SCORE_SURFACE,
                "source_artifacts": [str(EXP5473_RELATIVE_PATH), str(EXP5461_RELATIVE_PATH)],
                "detail": (
                    "Exp5473 prior_success is populated from Exp5461 condition quality "
                    "scores, and Exp5474 headline quality/pass-rate fields are computed "
                    "from the same exact-validator acceptance surface."
                ),
            }
        )
    if list(tautology_fields) == ["delta_vs_naive_icl", "naive_icl_score"]:
        violations.append(
            {
                "kind": "top_level_summary_collision",
                "policy_score_field": None,
                "headline_outcome_fields": [
                    "experiment_5474.delta_vs_naive_icl",
                    "experiment_5474.naive_icl_score",
                ],
                "shared_scalar_family": "reported_top_level_scalar_equality",
                "source_artifacts": [str(EXP5474_RELATIVE_PATH)],
                "detail": (
                    "The adversarial backfill observed delta_vs_naive_icl and "
                    "naive_icl_score at the same scalar value, so the summary must "
                    "not be treated as clean headline evidence."
                ),
            }
        )
    return violations


def cross_check_summary(
    *,
    exp5474: Mapping[str, Any],
    exp5473: Mapping[str, Any],
    exp5475: Mapping[str, Any],
    tautology_fields: Sequence[str],
) -> JsonDict:
    """Summarize what the adjacent CSL artifacts can and cannot support."""

    axis_details = _mapping(exp5475.get("axis_details"))
    independent_axes = sorted(
        axis
        for axis, rows in axis_details.items()
        if isinstance(rows, list)
        and any(
            isinstance(row, Mapping)
            and row.get("axis_pass") is True
            and row.get("accepted_by_exact_validator") is True
            for row in rows
        )
    )
    exp5475_rows = _rows(exp5475.get("row_results"))
    exact_validator_ok = bool(exp5475_rows) and all(
        _mapping(row.get("exact_validator_results")).get("authority")
        == "deterministic_replay_validator"
        and row.get("final_authority_bypassed") is False
        for row in exp5475_rows
    )
    if exact_validator_ok:
        independent_axes.append("exact_validator")
    required_axes = {
        "support_removal",
        "conflict_handling",
        "downstream_action_use",
        "stale_memory_rejection",
        "exact_validator",
    }
    policy_prior_reused = any(
        _mapping(row.get("features")).get("prior_success") == exp5473.get("governed_policy_score")
        for row in _rows(exp5473.get("surrogate_rows"))
        if row.get("condition") == "policy_selected"
    )
    exp5474_rows = _rows(exp5474.get("panel_rows"))
    return {
        "exp5474": {
            "prior_flagged_adversarial": exp5474.get("flagged_adversarial") is True,
            "adversarial_tautology_fields": list(tautology_fields),
            "exact_validator_authority_ok": bool(exp5474_rows)
            and all(row.get("exact_validator_authority") == "exact_task_verifier" for row in exp5474_rows),
            "final_authority_bypassed_count": sum(
                1 for row in exp5474_rows if row.get("final_authority_bypassed") is True
            ),
        },
        "exp5473": {
            "csl_kan_surrogate_ready": exp5473.get("csl_kan_surrogate_ready") is True,
            "policy_quality_score_reused_as_prior_success": policy_prior_reused,
            "independent_for_scale_headline": not policy_prior_reused,
            "bounded_validator_support": exp5473.get("constraint_violation_count") == 0,
            "negative_transfer_deflection_rate": exp5473.get("negative_transfer_deflection_rate"),
        },
        "exp5475": {
            "csl_behavioral_memory_ready": exp5475.get("csl_behavioral_memory_ready") is True,
            "independent_axes_present": sorted(set(independent_axes)),
            "bounded_behavioral_memory_support": required_axes.issubset(set(independent_axes)),
            "headline_permission": False,
        },
    }


def adversarial_tautology_fields(exp5474: Mapping[str, Any]) -> list[str]:
    """Extract field names from the prior Exp5474 TAUTOLOGY detail."""

    fields: set[str] = set()
    for flag in exp5474.get("corrigendum_pending", []):
        flag_map = _mapping(flag)
        if flag_map.get("kind") == "TAUTOLOGY":
            fields.update(
                re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=", str(flag_map.get("detail", "")))
            )
    return sorted(fields)


def downstream_recommendation(
    *,
    metric_independence_clean: bool,
    bounded_behavioral_memory_support: bool,
) -> str:
    """Choose the downstream state requested by REQ-LEARN-5502."""

    if metric_independence_clean:
        return "clean"
    if bounded_behavioral_memory_support:
        return "bounded_requires_rerun"
    return "retire_same_scope_if_repeated"


def honest_verdict(*, metric_independence_clean: bool, recommendation: str) -> str:
    """Return a terminal verdict that does not launder the Exp5474 headline."""

    if metric_independence_clean:
        return "complete: Exp5474 CSL metric graph is clean for prior headline use"
    return (
        "complete: Exp5474 CSL scale headline is bounded, not clean; "
        f"recommendation={recommendation} after policy-score/outcome overlap audit"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp5502 corrigendum is malformed or laundering evidence."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, recommendation, and dependency-classification errors."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py must not be modified")
    if not isinstance(artifact.get("audited_artifacts"), list):
        errors.append("audited_artifacts must be a list")
    graph_nodes = artifact.get("metric_graph_nodes")
    if not isinstance(graph_nodes, list):
        errors.append("metric_graph_nodes must be a list")
        graph_nodes = []
    if any(_mapping(node).get("classification") not in CLASSIFICATIONS for node in graph_nodes):
        errors.append("metric graph classifications must be known")
    policy_score_fields = artifact.get("policy_score_fields")
    if not isinstance(policy_score_fields, list) or PRIOR_SUCCESS_FIELD not in policy_score_fields:
        errors.append("policy_score_fields must include prior_success")
    outcome_metric_fields = artifact.get("outcome_metric_fields")
    if not isinstance(outcome_metric_fields, list) or "experiment_5474.kan_assured_csl_score" not in outcome_metric_fields:
        errors.append("outcome_metric_fields must include Exp5474 headline score")
    violations = artifact.get("independence_violations")
    if not isinstance(violations, list):
        errors.append("independence_violations must be a list")
        violations = []
    if artifact.get("metric_independence_clean") != (not violations):
        errors.append("metric_independence_clean must match independence_violations")
    if artifact.get("tautology_flag_resolved") is not True:
        errors.append("tautology_flag_resolved must be true after bounded adjudication")
    if violations and artifact.get("csl_scale_headline_allowed") is not False:
        errors.append("csl_scale_headline_allowed must be false when violations exist")
    recommendation = artifact.get("downstream_recommendation")
    if recommendation not in RECOMMENDATIONS:
        errors.append("downstream_recommendation must be recognized")
    if violations and recommendation != "bounded_requires_rerun":
        errors.append("downstream_recommendation must be bounded_requires_rerun")
    if recommendation != "clean" and artifact.get("retire_same_scope_if_repeated") is not True:
        errors.append("retire_same_scope_if_repeated must be true for non-clean recommendations")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or "\n" in verdict or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked:")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def source_file_checksums(root: Path) -> JsonDict:
    """Hash the module, spec, and upstream artifacts used by the audit."""

    paths = {
        "module": root / MODULE_RELATIVE_PATH,
        "spec": root / SPEC_RELATIVE_PATH,
        "exp5474": root / EXP5474_RELATIVE_PATH,
        "exp5473": root / EXP5473_RELATIVE_PATH,
        "exp5475": root / EXP5475_RELATIVE_PATH,
        "exp5461": root / EXP5461_RELATIVE_PATH,
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while excluding the self-referential checksum field."""

    return _sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(root=args.root, result_path=args.result_path, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["tautology_flag_resolved"] else 1


def _node(
    field: str,
    classification: str,
    source_artifact: str,
    depends_on: Sequence[str],
    note: str,
) -> JsonDict:
    return {
        "field": field,
        "classification": classification,
        "source_artifact": source_artifact,
        "depends_on": list(depends_on),
        "note": note,
    }


def _quality_score(artifact: Mapping[str, Any], condition: str) -> float:
    return float(
        _mapping(_mapping(artifact.get("condition_metrics")).get(condition)).get(
            "quality_score", -1.0
        )
    )


def _normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    return [
        dict(item) if isinstance(item, Mapping) else {"command": str(item), "outcome": "reported"}
        for item in tests_run
    ]


def _read_json(path: Path) -> JsonDict:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _rows(value: Any) -> list[JsonDict]:
    return [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(_json_ready(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
