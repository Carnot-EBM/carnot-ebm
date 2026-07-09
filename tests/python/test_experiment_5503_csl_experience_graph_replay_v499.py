"""Tests for Exp5503 executor-frozen CSL experience-graph replay.

Spec refs: REQ-LEARN-5503, SCENARIO-LEARN-5503-GRAPH-UPDATE,
SCENARIO-LEARN-5503-RETRIEVAL-CONTROLS, SCENARIO-LEARN-5503-BASELINE,
SCENARIO-LEARN-5503-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5503_csl_experience_graph_replay_v499 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5503_csl_experience_graph_replay_v499.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5503_csl_experience_graph_replay_v499.py "
    "-m pytest tests/python/test_experiment_5503_csl_experience_graph_replay_v499.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5503_csl_experience_graph_replay_v499.py "
    "--fail-under=100"
)
TESTS_RUN = [TEST_COMMAND, COVERAGE_COMMAND]


def _artifact() -> exp.JsonDict:
    return exp.build_artifact(root=REPO, tests_run=TESTS_RUN)


def test_req_learn_5503_spec_declares_experience_graph_contract() -> None:
    """REQ-LEARN-5503: OpenSpec anchors fixture, controls, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5503") :]

    for marker in (
        "REQ-LEARN-5503",
        "SCENARIO-LEARN-5503-GRAPH-UPDATE",
        "SCENARIO-LEARN-5503-RETRIEVAL-CONTROLS",
        "SCENARIO-LEARN-5503-BASELINE",
        "SCENARIO-LEARN-5503-ARTIFACT",
        str(exp.RESULT_RELATIVE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "stale evidence",
        "negative-transfer",
        "utility score",
        "exact verifier pass rates",
    ):
        assert marker in section

    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_5503_graph_update_records_episode_hash_and_next_retrieval() -> None:
    """SCENARIO-LEARN-5503-GRAPH-UPDATE: every episode writes auditable memory."""

    replay = exp.run_graph_replay(exp.build_replay_fixture(), exp.initial_memory_graph())
    first_episode = replay["episode_records"][0]
    first_next = first_episode["next_task_retrieval_decision"]

    assert replay["num_stream_tasks"] == 7
    assert len(replay["memory_state_hashes"]) == replay["num_stream_tasks"]
    assert len(set(replay["memory_state_hashes"])) == replay["num_stream_tasks"]
    assert first_episode["task_id"] == "5503-train-dock-crate"
    assert first_episode["selected_action"] == "crate-red"
    assert first_episode["verifier_outcome"]["accepted"] is False
    assert first_episode["learned_node"]["node_type"] == "failure"
    assert first_episode["memory_update_hash"] == replay["memory_state_hashes"][0]
    assert first_next["task_id"] == "5503-heldout-dock-crate"
    assert first_next["selected_node_id"] == "node5503-failure-dock-crate-red"
    assert first_next["selected_action"] == "crate-blue"


def test_scenario_learn_5503_retrieval_rejects_stale_conflict_and_transfer() -> None:
    """SCENARIO-LEARN-5503-RETRIEVAL-CONTROLS: unsafe memories do not steer."""

    replay = exp.run_graph_replay(exp.build_replay_fixture(), exp.initial_memory_graph())
    episodes = {episode["task_id"]: episode for episode in replay["episode_records"]}

    rx4_decision = episodes["5503-heldout-rx4-handoff"]["retrieval_decision"]
    python_decision = episodes["5503-heldout-python-loop"]["retrieval_decision"]
    gate_decision = episodes["5503-heldout-dock-gate"]["retrieval_decision"]

    assert rx4_decision["selected_action"] == "queue-beta"
    assert rx4_decision["rejected_node_ids_by_reason"]["stale_evidence"] == [
        "node5503-stale-rx4-alpha"
    ]
    assert python_decision["selected_action"] == "use-range-len"
    assert python_decision["rejected_node_ids_by_reason"]["negative_transfer"] == [
        "node5503-transfer-sql-offset"
    ]
    assert gate_decision["selected_node_id"] == "node5503-conflict-gate-new"
    assert gate_decision["rejected_node_ids_by_reason"]["conflict_lower_utility"] == [
        "node5503-conflict-gate-old"
    ]
    assert replay["negative_transfer_rate"] == pytest.approx(0.0)
    assert replay["stale_evidence_rejection_rate"] == pytest.approx(1.0)


def test_scenario_learn_5503_baseline_delta_uses_exact_heldout_outcomes() -> None:
    """SCENARIO-LEARN-5503-BASELINE: held-out scores are exact action matches."""

    artifact = _artifact()

    exp.validate_artifact(artifact)
    assert artifact["num_stream_tasks"] == 7
    assert artifact["no_memory_baseline_score"] == pytest.approx(0.0)
    assert artifact["graph_memory_score"] == pytest.approx(1.0)
    assert artifact["heldout_delta"] == pytest.approx(1.0)
    assert artifact["negative_transfer_rate"] == pytest.approx(0.0)
    assert artifact["stale_evidence_rejection_rate"] == pytest.approx(1.0)
    assert artifact["csl_experience_graph_ready"] is True
    assert artifact["model_weights_mutated"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert "not the retrieval utility" in artifact["metric_independence_notes"]
    assert {
        row["task_id"] for row in artifact["no_memory_baseline_results"]
    } == set(artifact["heldout_task_ids"])


def test_scenario_learn_5503_artifact_write_and_repository_replay_match(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5503-ARTIFACT: run() writes stable deliverable JSON."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    fixture_path = tmp_path / exp.REPLAY_FIXTURE_RELATIVE_PATH
    graph_path = tmp_path / exp.MEMORY_GRAPH_RELATIVE_PATH
    artifact = exp.run(
        root=REPO,
        result_path=result_path,
        replay_fixture_path=fixture_path,
        memory_graph_path=graph_path,
        tests_run=TESTS_RUN,
        write=True,
    )
    dry_run = exp.run(
        root=REPO,
        result_path=tmp_path / "dry-run.json",
        replay_fixture_path=fixture_path,
        memory_graph_path=graph_path,
        tests_run=TESTS_RUN,
        write=False,
    )
    repo_result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    repo_replay = exp.build_artifact(root=REPO, tests_run=repo_result["tests_run"])

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert json.loads(fixture_path.read_text(encoding="utf-8")) == artifact["replay_fixture"]
    assert json.loads(graph_path.read_text(encoding="utf-8")) == artifact["memory_graph"]
    assert dry_run == artifact
    assert not (tmp_path / "dry-run.json").exists()
    assert repo_result == repo_replay
    assert repo_result["csl_experience_graph_ready"] is True
    assert repo_result["model_weights_mutated"] is False


def test_req_learn_5503_validation_rejects_schema_and_gate_drift() -> None:
    """REQ-LEARN-5503-6: validation fails closed on artifact drift."""

    artifact = _artifact()
    exp.validate_artifact(artifact)

    drift_cases = [
        ("model_weights_mutated", True, "model_weights_mutated"),
        ("inference_substrate", "deterministic_replay_no_llm", "inference_substrate"),
        ("csl_experience_graph_ready", False, "csl_experience_graph_ready"),
        ("graph_memory_score", 0.0, "heldout_delta"),
        ("negative_transfer_rate", 0.5, "negative_transfer_rate"),
        ("stale_evidence_rejection_rate", 0.5, "stale_evidence_rejection_rate"),
        ("honest_verdict", "done", "honest_verdict"),
        ("research_conductor_modified", True, "scripts/research_conductor.py"),
    ]
    for field, value, expected in drift_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    missing = deepcopy(artifact)
    missing.pop("memory_graph_path")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    bad_hashes = deepcopy(artifact)
    bad_hashes["memory_state_hashes"] = bad_hashes["memory_state_hashes"][:-1]
    with pytest.raises(ValueError, match="memory_state_hashes"):
        exp.validate_artifact(bad_hashes)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    bad_tests["csl_experience_graph_ready"] = False
    bad_tests["honest_verdict"] = "blocked: experience_graph_replay_not_ready"
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    bad_repro = deepcopy(artifact)
    bad_repro["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_repro)


def test_req_learn_5503_defensive_paths_for_cached_verifier_and_helpers() -> None:
    """REQ-LEARN-5503-3/4: defensive helpers keep exact metrics explicit."""

    task = exp.build_replay_fixture()["stream_tasks"][0]
    verifier = exp.exact_verifier(task, "not-a-cached-action")
    controls = exp._control_rates(
        [
            {
                "rejected_node_ids_by_reason": {},
                "accepted_nodes": [
                    {
                        "node_id": "accepted-transfer",
                        "negative_transfer_domains": ["other-domain"],
                    }
                ],
            }
        ]
    )

    assert verifier["accepted"] is False
    assert verifier["cached_candidate"] is False
    assert verifier["failure_reasons"] == ["selected_action_not_cached"]
    assert controls["negative_transfer_rate"] == pytest.approx(1.0)
    assert controls["stale_evidence_rejection_rate"] == pytest.approx(0.0)
    assert exp._resolve_output_path(REPO, exp.RESULT_RELATIVE_PATH) == REPO / exp.RESULT_RELATIVE_PATH
    assert exp._list_of_mappings(None) == []
    assert exp._honest_verdict(False, 0.0).startswith("blocked:")
