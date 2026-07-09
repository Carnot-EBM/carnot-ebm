"""Tests for Exp5502 CSL tautology static corrigendum.

Spec refs: REQ-LEARN-5502,
SCENARIO-LEARN-5502-METRIC-GRAPH,
SCENARIO-LEARN-5502-CROSS-CHECK,
SCENARIO-LEARN-5502-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5502_csl_tautology_static_corrigendum_v499 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5502_csl_tautology_static_corrigendum_v499.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5502_csl_tautology_static_corrigendum_v499.py "
    "-m pytest tests/python/test_experiment_5502_csl_tautology_static_corrigendum_v499.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5502_csl_tautology_static_corrigendum_v499.py "
    "--fail-under=100"
)


def _artifact() -> dict[str, object]:
    return mod.build_artifact(root=REPO, tests_run=[TEST_COMMAND, COVERAGE_COMMAND])


def test_req_learn_5502_spec_declares_static_corrigendum_contract() -> None:
    """REQ-LEARN-5502: OpenSpec anchors the metric-independence audit."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5502") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5502",
        "SCENARIO-LEARN-5502-METRIC-GRAPH",
        "SCENARIO-LEARN-5502-CROSS-CHECK",
        "SCENARIO-LEARN-5502-ARTIFACT",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "surrogate_rows.features.prior_success",
        "condition_metrics.*.quality_score",
        "delta_vs_naive_icl",
        "bounded_requires_rerun",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5502_metric_graph_fails_quality_score_overlap() -> None:
    """SCENARIO-LEARN-5502-METRIC-GRAPH: policy/outcome reuse blocks headline."""

    artifact = _artifact()

    mod.validate_artifact(artifact)
    assert artifact["metric_independence_clean"] is False
    assert artifact["tautology_flag_resolved"] is True
    assert artifact["csl_scale_headline_allowed"] is False
    assert artifact["downstream_recommendation"] == "bounded_requires_rerun"
    assert artifact["retire_same_scope_if_repeated"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")

    policy_fields = set(artifact["policy_score_fields"])
    outcome_fields = set(artifact["outcome_metric_fields"])
    assert "experiment_5473.surrogate_rows.features.prior_success" in policy_fields
    assert "experiment_5474.kan_assured_csl_score" in outcome_fields
    assert "experiment_5474.delta_vs_naive_icl" in outcome_fields

    violations = artifact["independence_violations"]
    assert {violation["kind"] for violation in violations} == {
        "policy_outcome_scalar_overlap",
        "top_level_summary_collision",
    }
    overlap = next(v for v in violations if v["kind"] == "policy_outcome_scalar_overlap")
    assert overlap["policy_score_field"] == "experiment_5473.surrogate_rows.features.prior_success"
    assert "condition_metrics.*.quality_score" in overlap["shared_scalar_family"]
    assert "experiment_5474.kan_assured_csl_score" in overlap["headline_outcome_fields"]

    graph = {node["field"]: node for node in artifact["metric_graph_nodes"]}
    assert graph["experiment_5473.surrogate_rows.features.prior_success"][
        "classification"
    ] == "upstream feature"
    assert graph["experiment_5473.surrogate_rows.surrogate_accept"][
        "classification"
    ] == "policy decision"
    assert graph["experiment_5474.panel_rows.accepted_by_final_authority"][
        "classification"
    ] == "evaluator outcome"
    assert graph["experiment_5474.naive_icl_score"]["classification"] == "baseline outcome"
    assert graph["experiment_5474.delta_vs_naive_icl"]["classification"] == "derived summary"


def test_scenario_learn_5502_cross_check_bounds_independent_behavioral_support() -> None:
    """SCENARIO-LEARN-5502-CROSS-CHECK: Exp5475 supports behavior, not headline."""

    artifact = _artifact()
    cross_check = artifact["cross_check_summary"]

    assert cross_check["exp5473"]["independent_for_scale_headline"] is False
    assert cross_check["exp5473"]["policy_quality_score_reused_as_prior_success"] is True
    assert cross_check["exp5475"]["bounded_behavioral_memory_support"] is True
    assert cross_check["exp5475"]["headline_permission"] is False
    assert set(cross_check["exp5475"]["independent_axes_present"]) >= {
        "support_removal",
        "conflict_handling",
        "downstream_action_use",
        "stale_memory_rejection",
        "exact_validator",
    }
    assert cross_check["exp5474"]["prior_flagged_adversarial"] is True
    assert cross_check["exp5474"]["adversarial_tautology_fields"] == [
        "delta_vs_naive_icl",
        "naive_icl_score",
    ]


def test_scenario_learn_5502_run_writes_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5502-ARTIFACT: run() writes deterministic JSON."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=REPO,
        result_path=destination,
        tests_run=[TEST_COMMAND, COVERAGE_COMMAND],
        write=True,
    )
    no_write_path = tmp_path / "no-write.json"
    no_write = mod.run(
        root=REPO,
        result_path=no_write_path,
        tests_run=[TEST_COMMAND, COVERAGE_COMMAND],
        write=False,
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == artifact
    assert no_write == artifact
    assert not no_write_path.exists()
    mod.validate_artifact(artifact)
    assert artifact["source_file_checksums"]["module"].startswith("sha256:")
    assert artifact["source_file_checksums"]["spec"].startswith("sha256:")


def test_req_learn_5502_validation_fails_closed_on_schema_or_claim_drift() -> None:
    """REQ-LEARN-5502-3/5: malformed or laundering artifacts fail validation."""

    artifact = _artifact()

    assert mod.downstream_recommendation(
        metric_independence_clean=True,
        bounded_behavioral_memory_support=False,
    ) == "clean"
    assert mod.downstream_recommendation(
        metric_independence_clean=False,
        bounded_behavioral_memory_support=False,
    ) == "retire_same_scope_if_repeated"
    assert mod.honest_verdict(
        metric_independence_clean=True,
        recommendation="clean",
    ).startswith("complete:")

    mutations = []
    missing = deepcopy(artifact)
    missing.pop("audited_artifacts")
    mutations.append((missing, "missing required fields"))

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    mutations.append((bad_principles, "field_principles"))

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    mutations.append((bad_substrate, "inference_substrate"))

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    mutations.append((bad_conductor, "scripts/research_conductor.py"))

    bad_audited = deepcopy(artifact)
    bad_audited["audited_artifacts"] = {}
    mutations.append((bad_audited, "audited_artifacts"))

    bad_graph_type = deepcopy(artifact)
    bad_graph_type["metric_graph_nodes"] = {}
    mutations.append((bad_graph_type, "metric_graph_nodes"))

    bad_clean = deepcopy(artifact)
    bad_clean["metric_independence_clean"] = True
    mutations.append((bad_clean, "metric_independence_clean"))

    bad_headline = deepcopy(artifact)
    bad_headline["csl_scale_headline_allowed"] = True
    mutations.append((bad_headline, "csl_scale_headline_allowed"))

    bad_recommendation = deepcopy(artifact)
    bad_recommendation["downstream_recommendation"] = "clean"
    mutations.append((bad_recommendation, "downstream_recommendation"))

    unknown_recommendation = deepcopy(artifact)
    unknown_recommendation["downstream_recommendation"] = "unknown"
    mutations.append((unknown_recommendation, "downstream_recommendation must be recognized"))

    bad_retire = deepcopy(artifact)
    bad_retire["retire_same_scope_if_repeated"] = False
    mutations.append((bad_retire, "retire_same_scope_if_repeated"))

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    mutations.append((bad_verdict, "honest_verdict"))

    bad_graph = deepcopy(artifact)
    bad_graph["metric_graph_nodes"][0]["classification"] = "mystery"
    mutations.append((bad_graph, "metric graph classifications"))

    bad_policy = deepcopy(artifact)
    bad_policy["policy_score_fields"] = []
    mutations.append((bad_policy, "policy_score_fields"))

    bad_outcomes = deepcopy(artifact)
    bad_outcomes["outcome_metric_fields"] = []
    mutations.append((bad_outcomes, "outcome_metric_fields"))

    bad_violations = deepcopy(artifact)
    bad_violations["independence_violations"] = {}
    mutations.append((bad_violations, "independence_violations"))

    bad_tautology = deepcopy(artifact)
    bad_tautology["tautology_flag_resolved"] = False
    mutations.append((bad_tautology, "tautology_flag_resolved"))

    bad_seed = deepcopy(artifact)
    bad_seed["random_seed"] = 1
    mutations.append((bad_seed, "random_seed"))

    for payload, expected in mutations:
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(payload)


def test_req_learn_5502_repository_artifact_is_valid() -> None:
    """REQ-LEARN-5502-5: committed deliverable keeps Exp5474 bounded."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(result)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(result)
    assert result["downstream_recommendation"] == "bounded_requires_rerun"
    assert result["csl_scale_headline_allowed"] is False
