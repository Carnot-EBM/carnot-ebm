"""Tests for Exp 1302 skill-graph promotion/demotion.

Spec: REQ-LEARN-1302, SCENARIO-LEARN-1302.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.skill_graph_promotion_demotion import (
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _exp1288_fixture(*, memory_update_written: bool = True) -> dict[str, object]:
    replay_slices = [
        {
            "case_id": f"case-{idx}",
            "chronological_index": idx,
            "online_decision": "repair" if idx % 2 == 0 else "accept",
            "posthoc_decision": "repair" if idx % 2 == 0 else "accept",
            "target_decision": "repair" if idx % 2 == 0 else "accept",
            "verifier_result": "failed" if idx % 2 == 0 else "passed",
        }
        for idx in range(6)
    ]
    return {
        "experiment": "1288_interwhen_dvi_verifier_feedback_replay",
        "status": "complete",
        "memory_update_written": memory_update_written,
        "clause_prediction_records": [
            {
                "constraint_pattern": "arithmetic:addition",
                "repair_hint": "recompute_arithmetic_result",
                "selected_decision": "repair",
                "support": 115,
                "verifier_result": "failed",
            },
            {
                "constraint_pattern": "arithmetic:addition",
                "repair_hint": "accept_verified_answer",
                "selected_decision": "accept",
                "support": 18,
                "verifier_result": "passed",
            },
            {
                "constraint_pattern": "arithmetic:balance",
                "repair_hint": "recompute_arithmetic_result",
                "selected_decision": "repair",
                "support": 14,
                "verifier_result": "failed",
            },
            {
                "constraint_pattern": "arithmetic:general",
                "repair_hint": "recompute_arithmetic_result",
                "selected_decision": "repair",
                "support": 115,
                "verifier_result": "failed",
            },
            {
                "constraint_pattern": "arithmetic:general",
                "repair_hint": "accept_verified_answer",
                "selected_decision": "accept",
                "support": 3,
                "verifier_result": "passed",
            },
            {
                "constraint_pattern": "arithmetic:ratio",
                "repair_hint": "recompute_arithmetic_result",
                "selected_decision": "repair",
                "support": 84,
                "verifier_result": "failed",
            },
            {
                "constraint_pattern": "arithmetic:ratio",
                "repair_hint": "accept_verified_answer",
                "selected_decision": "accept",
                "support": 1,
                "verifier_result": "passed",
            },
        ],
        "replay_slices": replay_slices,
    }


def test_req_learn_1302_in_progress_artifact_is_durable(tmp_path: Path) -> None:
    """REQ-LEARN-1302-1: the workflow writes an in-progress artifact first."""

    out_path = tmp_path / "results" / "experiment_1302_skill_graph_promotion_demotion_v2.json"

    artifact = write_in_progress_artifact(
        out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["artifact_metadata"]["run_date"] == "20260505"
    assert written["skill_graph_candidate_count"] == 0
    assert written["memory_update_written"] is False


def test_scenario_learn_1302_builds_promotion_demotion_and_expiry_counts() -> None:
    """SCENARIO-LEARN-1302: Exp1288 memory becomes a sandboxed skill graph."""

    artifact = build_artifact(
        _exp1288_fixture(),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert artifact["status"] == "complete"
    assert artifact["skill_graph_candidate_count"] == 7
    assert artifact["promoted_memory_count"] == 5
    assert artifact["demoted_memory_count"] == 1
    assert artifact["expired_memory_count"] == 1
    assert artifact["replay_evidence_count"] == 6
    assert artifact["memory_update_written"] is True
    assert artifact["honest_verdict"] == "skill_graph_candidates_written_sandboxed"
    assert artifact["candidate_artifact_path"].startswith("results/")

    by_pattern = {
        (entry["constraint_pattern"], entry["repair_hint"]): entry
        for entry in artifact["skill_graph_candidates"]
    }
    promoted = by_pattern[("arithmetic:addition", "recompute_arithmetic_result")]
    demoted = by_pattern[("arithmetic:general", "accept_verified_answer")]
    expired = by_pattern[("arithmetic:ratio", "accept_verified_answer")]

    assert promoted["memory_routing_decision"] == "promote"
    assert promoted["memory_type_tags"] == ["procedural", "verifier_feedback", "repair_policy"]
    assert promoted["replay_evidence"]["support"] == 115
    assert promoted["promotion_criteria"]["min_support"] == 5
    assert promoted["demotion_criteria"]["harmful_if_verifier_conflict"] is True
    assert promoted["expiry_criteria"]["expire_when_support_at_or_below"] == 1

    assert demoted["memory_routing_decision"] == "demote"
    assert demoted["memory_type_tags"] == ["episodic", "verifier_feedback", "accept_policy"]
    assert expired["memory_routing_decision"] == "expire"


def test_req_learn_1302_run_loads_exp1288_and_writes_results_only(tmp_path: Path) -> None:
    """REQ-LEARN-1302-2/5: run reads Exp1288 and writes under results."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    source_path = results_dir / "experiment_1288_interwhen_dvi_verifier_feedback_replay.json"
    source_path.write_text(json.dumps(_exp1288_fixture()), encoding="utf-8")
    out_path = results_dir / "experiment_1302_skill_graph_promotion_demotion_v2.json"

    artifact = run(
        results_dir=results_dir,
        out_path=out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["source_artifact"] == "results/experiment_1288_interwhen_dvi_verifier_feedback_replay.json"
    assert artifact["candidate_artifact_path"] == "results/experiment_1302_skill_graph_promotion_demotion_v2.json"
    assert not (tmp_path / "skills").exists()
    assert not (tmp_path / "skill_directory").exists()


def test_req_learn_1302_missing_memory_update_reports_no_write() -> None:
    """REQ-LEARN-1302-4: artifacts stay honest when Exp1288 lacks a memory write."""

    artifact = build_artifact(
        _exp1288_fixture(memory_update_written=False),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert artifact["status"] == "complete"
    assert artifact["skill_graph_candidate_count"] == 0
    assert artifact["promoted_memory_count"] == 0
    assert artifact["demoted_memory_count"] == 0
    assert artifact["expired_memory_count"] == 0
    assert artifact["replay_evidence_count"] == 0
    assert artifact["memory_update_written"] is False
    assert artifact["honest_verdict"] == "blocked_no_exp1288_memory_update"
