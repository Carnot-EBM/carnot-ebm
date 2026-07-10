"""Tests for Exp5543 retrieval-warmed CSL five-arm ablation.

Spec refs: REQ-LEARN-5543,
SCENARIO-LEARN-5543-FIVE-ARMS,
SCENARIO-LEARN-5543-CONTROLS,
SCENARIO-LEARN-5543-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5543_retrieval_warmed_csl_five_arm_ablation as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5543_retrieval_warmed_csl_five_arm_ablation.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5543_retrieval_warmed_csl_five_arm_ablation.py "
    "-m pytest tests/python/test_experiment_5543_retrieval_warmed_csl_five_arm_ablation.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5543_retrieval_warmed_csl_five_arm_ablation.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TESTS_ADDED_OR_REUSED = [TEST_COMMAND, COVERAGE_COMMAND, FULL_TEST_COMMAND]


def test_req_learn_5543_spec_declares_five_arm_contract() -> None:
    """REQ-LEARN-5543: OpenSpec anchors the five-arm ablation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5543") :]

    for marker in (
        "REQ-LEARN-5543",
        "SCENARIO-LEARN-5543-FIVE-ARMS",
        "SCENARIO-LEARN-5543-CONTROLS",
        "SCENARIO-LEARN-5543-ARTIFACT",
        str(exp.RESULT_RELATIVE_PATH),
        str(exp.UPSTREAM_RESIDUE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "oracle memory, best constant answer, per-query random memory",
        "aligned retrieval beats both shuffled and per-query random controls",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_5543_five_arms_share_queries_and_hashes() -> None:
    """SCENARIO-LEARN-5543-FIVE-ARMS: every arm scores the same queries."""

    fixture = exp.build_fixture()
    evaluation = exp.evaluate_five_arms(fixture)
    query_ids = evaluation["shared_query_ids"]

    assert list(evaluation["arm_results"]) == list(exp.ARM_NAMES)
    assert len(query_ids) == 12
    assert len(evaluation["query_hashes"]) == len(query_ids)
    assert len(set(evaluation["query_hashes"])) == len(query_ids)
    for arm in exp.ARM_NAMES:
        assert [row["query_id"] for row in evaluation["arm_results"][arm]] == query_ids
    assert evaluation["same_heldout_query_set"] is True
    assert evaluation["scores"]["oracle_memory"] == pytest.approx(1.0)
    assert evaluation["scores"]["best_constant_answer"] == pytest.approx(0.0833333333)
    assert evaluation["scores"]["per_query_random_memory"] == pytest.approx(0.0833333333)
    assert evaluation["scores"]["shuffled_memory"] == pytest.approx(0.25)
    assert evaluation["scores"]["aligned_retrieval_memory"] == pytest.approx(0.8333333333)
    assert evaluation["aligned_minus_shuffled_delta"] == pytest.approx(0.5833333333)
    assert evaluation["aligned_minus_random_delta"] == pytest.approx(0.75)
    assert evaluation["stale_evidence_rejection_rate"] == pytest.approx(1.0)
    assert evaluation["negative_transfer_rate"] == pytest.approx(0.0)
    assert len(evaluation["memory_hashes"]) == len(exp.ARM_NAMES)
    assert len(set(evaluation["memory_hashes"])) == len(exp.ARM_NAMES)


def test_scenario_learn_5543_artifact_fields_and_ready_gate(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5543-CONTROLS: aligned retrieval beats controls."""

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
    assert relative_artifact["query_hashes"] == artifact["query_hashes"]
    assert exp.validate_artifact(artifact) is True
    assert exp._resolve_path(REPO, exp.RESULT_RELATIVE_PATH) == REPO / exp.RESULT_RELATIVE_PATH
    assert exp._resolve_path(REPO, result_path) == result_path

    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field]

    assert artifact["upstream_csl_residue_tautology_resolved"] is True
    assert artifact["upstream_residue_path"] == exp.UPSTREAM_RESIDUE_PATH.as_posix()
    assert artifact["oracle_score"] == pytest.approx(1.0)
    assert artifact["best_constant_score"] == pytest.approx(0.0833333333)
    assert artifact["per_query_random_score"] == pytest.approx(0.0833333333)
    assert artifact["shuffled_memory_score"] == pytest.approx(0.25)
    assert artifact["aligned_memory_score"] == pytest.approx(0.8333333333)
    assert artifact["aligned_minus_shuffled_delta"] == pytest.approx(0.5833333333)
    assert artifact["aligned_minus_random_delta"] == pytest.approx(0.75)
    assert artifact["stale_evidence_rejection_rate"] == pytest.approx(1.0)
    assert artifact["negative_transfer_rate"] == pytest.approx(0.0)
    assert artifact["no_weight_mutation"] is True
    assert artifact["weight_mutation_evidence"]["before_hash"] == artifact[
        "weight_mutation_evidence"
    ]["after_hash"]
    assert artifact["csl_five_arm_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED
    assert set(artifact["memory_hash_evidence"]) == set(exp.ARM_NAMES)
    assert artifact["memory_hashes"] == [
        artifact["memory_hash_evidence"][arm] for arm in exp.ARM_NAMES
    ]
    assert artifact["control_counts"] == {
        "stale_candidates_seen": 3,
        "stale_candidates_rejected": 3,
        "negative_transfer_candidates_seen": 2,
        "negative_transfer_candidates_accepted": 0,
    }
    assert exp.upstream_residue_status(REPO)["csl_residue_tautology_resolved"] is True
    assert exp.upstream_residue_status(tmp_path)["loadable"] is False


def test_scenario_learn_5543_artifact_fails_closed_on_gate_drift(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5543-ARTIFACT: invalid positive gates are rejected."""

    artifact = exp.run(
        root=REPO,
        result_path=tmp_path / exp.RESULT_RELATIVE_PATH.name,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=False,
    )
    assert exp.validate_artifact(artifact) is True

    blocked = deepcopy(artifact)
    blocked["upstream_csl_residue_tautology_resolved"] = False
    blocked["upstream_residue_status"]["csl_residue_tautology_resolved"] = False
    blocked["csl_five_arm_ready"] = False
    blocked["honest_verdict"] = exp.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    assert exp.validate_artifact(blocked) is True
    assert blocked["honest_verdict"].startswith("blocked:")

    drift_cases = [
        ("upstream_residue_path", "results/wrong.json", "upstream_residue_path"),
        ("oracle_score", 0.0, "oracle_score"),
        ("best_constant_score", 0.0, "best_constant_score"),
        ("per_query_random_score", artifact["aligned_memory_score"], "per_query_random_score"),
        ("shuffled_memory_score", artifact["aligned_memory_score"], "shuffled_memory_score"),
        ("aligned_minus_shuffled_delta", 0.0, "aligned_minus_shuffled_delta"),
        ("aligned_minus_random_delta", 0.0, "aligned_minus_random_delta"),
        ("stale_evidence_rejection_rate", 0.5, "stale_evidence_rejection_rate"),
        ("negative_transfer_rate", 0.5, "negative_transfer_rate"),
        ("no_weight_mutation", False, "no_weight_mutation"),
        ("csl_five_arm_ready", False, "csl_five_arm_ready"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("honest_verdict", "ready", "honest_verdict"),
    ]
    for field, value, expected in drift_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    divergent_queries = deepcopy(artifact)
    divergent_queries["arm_results"]["shuffled_memory"] = divergent_queries["arm_results"][
        "shuffled_memory"
    ][1:]
    divergent_queries["reproducibility_checksum"] = exp.reproducibility_checksum(
        divergent_queries
    )
    with pytest.raises(ValueError, match="same_heldout_query_set"):
        exp.validate_artifact(divergent_queries)

    bad_query_hashes = deepcopy(artifact)
    bad_query_hashes["query_hashes"] = bad_query_hashes["query_hashes"][1:]
    bad_query_hashes["reproducibility_checksum"] = exp.reproducibility_checksum(
        bad_query_hashes
    )
    with pytest.raises(ValueError, match="query_hashes"):
        exp.validate_artifact(bad_query_hashes)

    bad_memory_hashes = deepcopy(artifact)
    bad_memory_hashes["memory_hashes"] = bad_memory_hashes["memory_hashes"][1:]
    bad_memory_hashes["reproducibility_checksum"] = exp.reproducibility_checksum(
        bad_memory_hashes
    )
    with pytest.raises(ValueError, match="memory_hashes"):
        exp.validate_artifact(bad_memory_hashes)

    missing = deepcopy(artifact)
    missing.pop("aligned_memory_score")
    missing["reproducibility_checksum"] = exp.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("aligned_memory_score")
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

    assert exp.arm_scores_from_artifact(None) == {}
    assert exp.rate_from_counts({}, "accepted", "seen") == 0.0
