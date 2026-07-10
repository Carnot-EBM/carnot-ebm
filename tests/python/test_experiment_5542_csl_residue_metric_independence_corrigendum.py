"""Tests for Exp5542 CSL residue metric independence corrigendum.

Spec refs: REQ-LEARN-5542,
SCENARIO-LEARN-5542-DISTINCT-FAMILIES,
SCENARIO-LEARN-5542-CONTROLS,
SCENARIO-LEARN-5542-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5542_csl_residue_metric_independence_corrigendum as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5542_csl_residue_metric_independence_corrigendum.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5542_csl_residue_metric_independence_corrigendum.py "
    "-m pytest tests/python/test_experiment_5542_csl_residue_metric_independence_corrigendum.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5542_csl_residue_metric_independence_corrigendum.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TESTS_ADDED_OR_REUSED = [TEST_COMMAND, COVERAGE_COMMAND, FULL_TEST_COMMAND]


def test_req_learn_5542_spec_declares_corrigendum_contract() -> None:
    """REQ-LEARN-5542: OpenSpec anchors the residue-metric corrigendum."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5542") :]

    for marker in (
        "REQ-LEARN-5542",
        "SCENARIO-LEARN-5542-DISTINCT-FAMILIES",
        "SCENARIO-LEARN-5542-CONTROLS",
        "SCENARIO-LEARN-5542-ARTIFACT",
        str(exp.RESULT_RELATIVE_PATH),
        str(exp.CANONICAL_GATE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "event-only memory and topic-only memory",
        "distinct held-out labels or distinct query families",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_5542_distinct_families_score_nonidentically() -> None:
    """SCENARIO-LEARN-5542-DISTINCT-FAMILIES: score arms use disjoint labels."""

    fixture = exp.build_fixture()
    evaluation = exp.evaluate_fixture(fixture)
    family = evaluation["metric_family_evidence"]

    assert family["event_only_query_family"] == exp.EVENT_QUERY_FAMILY
    assert family["topic_only_query_family"] == exp.TOPIC_QUERY_FAMILY
    assert family["overlap_count"] == 0
    assert set(family["event_only_label_ids"]).isdisjoint(family["topic_only_label_ids"])
    assert len(family["event_only_label_ids"]) == 7
    assert len(family["topic_only_label_ids"]) == 5
    assert evaluation["scores"]["event_only"] == pytest.approx(0.7142857143)
    assert evaluation["scores"]["topic_only"] == pytest.approx(0.4)
    assert evaluation["score_difference_abs"] == pytest.approx(0.3142857143)
    assert evaluation["nonidentical_metric_evidence"] is True
    assert {
        row["query_family"] for row in evaluation["condition_results"]["event_only"]
    } == {exp.EVENT_QUERY_FAMILY}
    assert {
        row["query_family"] for row in evaluation["condition_results"]["topic_only"]
    } == {exp.TOPIC_QUERY_FAMILY}


def test_scenario_learn_5542_controls_and_artifact_fields(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5542-CONTROLS: controls are separate from headlines."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH.name
    artifact = exp.run(
        root=REPO,
        result_path=result_path,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )
    relative_artifact = exp.run(
        root=REPO,
        result_path=exp.RESULT_RELATIVE_PATH,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=False,
    )
    written = json.loads(result_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert relative_artifact["event_only_score"] == artifact["event_only_score"]
    assert exp.validate_artifact(artifact) is True
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field]

    assert artifact["canonical_gate_path"] == exp.CANONICAL_GATE_PATH.as_posix()
    assert artifact["canonical_gate_fields"]["csl_gate_fields_conductor_visible"] is True
    assert artifact["canonical_gate_fields"]["metric_independence_clean"] is True
    assert artifact["canonical_gate_fields"]["continuous_self_learning_evidence"] is True
    assert artifact["prior_exp5529_tautology"]["event_topic_scores_identical"] is True
    assert artifact["event_only_score"] == pytest.approx(0.7142857143)
    assert artifact["topic_only_score"] == pytest.approx(0.4)
    assert artifact["score_difference_abs"] == pytest.approx(0.3142857143)
    assert artifact["event_topic_score"] == pytest.approx(0.8333333333)
    assert artifact["no_memory_score"] == pytest.approx(0.1666666667)
    assert artifact["shuffled_memory_score"] == pytest.approx(0.25)
    assert artifact["stale_evidence_rejection_rate"] == pytest.approx(1.0)
    assert artifact["negative_transfer_rate"] == pytest.approx(0.0)
    assert artifact["independent_outcome_labels"] is True
    assert artifact["nonidentical_metric_evidence"] is True
    assert artifact["csl_residue_tautology_resolved"] is True
    assert artifact["csl_residue_stress_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")

    assert artifact["control_counts"]["stale_candidates_seen"] == 3
    assert artifact["control_counts"]["stale_candidates_rejected"] == 3
    assert artifact["control_counts"]["negative_transfer_candidates_seen"] == 2
    assert artifact["control_counts"]["negative_transfer_candidates_accepted"] == 0
    assert artifact["metric_family_evidence"]["overlap_count"] == 0
    assert set(artifact["condition_results"]) == set(exp.CONDITIONS)


def test_scenario_learn_5542_artifact_fails_closed_on_tautology_and_drift(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5542-ARTIFACT: equal scores block the corrigendum."""

    artifact = exp.run(
        root=REPO,
        result_path=tmp_path / exp.RESULT_RELATIVE_PATH.name,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=False,
    )
    assert exp.validate_artifact(artifact) is True

    blocked = deepcopy(artifact)
    blocked["topic_only_score"] = blocked["event_only_score"]
    blocked["score_difference_abs"] = 0.0
    blocked["nonidentical_metric_evidence"] = False
    blocked["csl_residue_tautology_resolved"] = False
    blocked["csl_residue_stress_ready"] = False
    blocked["honest_verdict"] = exp.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    assert exp.validate_artifact(blocked) is True
    assert blocked["honest_verdict"].startswith("blocked:")

    inconsistent_tautology = deepcopy(blocked)
    inconsistent_tautology["csl_residue_tautology_resolved"] = True
    inconsistent_tautology["csl_residue_stress_ready"] = True
    inconsistent_tautology["honest_verdict"] = exp.honest_verdict(inconsistent_tautology)
    inconsistent_tautology["reproducibility_checksum"] = exp.reproducibility_checksum(
        inconsistent_tautology
    )
    with pytest.raises(ValueError, match="csl_residue_tautology_resolved"):
        exp.validate_artifact(inconsistent_tautology)

    drift_cases = [
        ("canonical_gate_path", "results/wrong.json", "canonical_gate_path"),
        ("score_difference_abs", 0.1, "score_difference_abs"),
        ("event_topic_score", 0.0, "event_topic_score"),
        ("shuffled_memory_score", artifact["event_topic_score"], "shuffled_memory_score"),
        ("stale_evidence_rejection_rate", 0.5, "stale_evidence_rejection_rate"),
        ("negative_transfer_rate", 0.5, "negative_transfer_rate"),
        ("independent_outcome_labels", False, "independent_outcome_labels"),
        ("nonidentical_metric_evidence", False, "nonidentical_metric_evidence"),
        ("csl_residue_stress_ready", False, "csl_residue_stress_ready"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("honest_verdict", "ready", "honest_verdict"),
    ]
    for field, value, expected in drift_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    missing = deepcopy(artifact)
    missing.pop("event_only_score")
    missing["reproducibility_checksum"] = exp.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("event_only_score")
    missing_principle["reproducibility_checksum"] = exp.reproducibility_checksum(
        missing_principle
    )
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(missing_principle)

    no_tests = deepcopy(artifact)
    no_tests["tests_added_or_reused"] = []
    no_tests["reproducibility_checksum"] = exp.reproducibility_checksum(no_tests)
    with pytest.raises(ValueError, match="tests_added_or_reused"):
        exp.validate_artifact(no_tests)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)
