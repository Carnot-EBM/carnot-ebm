"""Tests for Exp 1555 FR-11 positive-utility-or-retire gate.

Spec: REQ-LEARN-1555, SCENARIO-LEARN-1555, SCENARIO-LEARN-1556,
SCENARIO-LEARN-1557.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fr11_positive_utility_or_retire as exp


def test_req_learn_1555_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1555-1/7: bootstrap artifact exposes every required field."""

    output = tmp_path / exp.OUTPUT_FILE
    skill_graph = tmp_path / exp.SKILL_GRAPH_FILE

    artifact = exp.write_in_progress_artifact(
        output,
        skill_graph_path=skill_graph,
        project_root=tmp_path,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["milestone"] == ".119"
    assert artifact["continuous_self_learning_task"] == "fr11_positive_utility_or_retire_v14"
    assert artifact["fr11_positive_utility_gate_ready"] is False
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["skill_graph_path"] == exp.SKILL_GRAPH_FILE
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    exp.validate_artifact(artifact)
    with pytest.raises(AssertionError, match="honest_verdict"):
        exp.validate_artifact(dict(artifact, honest_verdict="not-terminal"))


def test_scenario_learn_1555_residual_repair_promotion_demonstrates_utility(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1555: verifier-backed repairs can make utility positive."""

    candidates = exp.select_candidate_skill_updates(
        exp1539_artifact=_exp1539_zero_utility_artifact(),
        repair_artifact=_repair_artifact(),
        repair_rows=[
            _repair_row("heldout-1", accepted=True),
            _repair_row("heldout-2", accepted=True),
            _repair_row("heldout-3", accepted=False),
        ],
    )
    graph = exp.build_skill_graph(candidates, skill_graph_path=tmp_path / exp.SKILL_GRAPH_FILE)
    artifact = exp.build_artifact(
        candidates=candidates,
        graph=graph,
        skill_graph_path=tmp_path / exp.SKILL_GRAPH_FILE,
        focused_tests_passed=True,
        project_root=tmp_path,
    )

    promoted_ids = {update["update_id"] for update in artifact["skill_updates_promoted"]}

    assert "policy:residual_drift_repair:1552" in promoted_ids
    assert artifact["status"] == "complete"
    assert artifact["fr11_positive_utility_gate_ready"] is True
    assert artifact["external_feedback_used"] is True
    assert artifact["self_feedback_only_rejected"] is False
    assert artifact["baseline_utility"] == pytest.approx(0.0)
    assert artifact["post_promotion_utility"] == pytest.approx(1.0)
    assert artifact["utility_delta"] == pytest.approx(1.0)
    assert artifact["replay_pass_rate"] == pytest.approx(1.0)
    assert artifact["soundness_mistakes"] == 0
    assert artifact["positive_utility_achieved"] is True
    assert artifact["positive_utility_claim_retired"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    exp.validate_artifact(artifact, skill_graph_path=tmp_path / exp.SKILL_GRAPH_FILE)


def test_scenario_learn_1556_zero_utility_retires_positive_headline(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1556: zero utility completes but retires the headline."""

    candidates = exp.select_candidate_skill_updates(
        exp1539_artifact=_exp1539_zero_utility_artifact(),
        repair_artifact={"status": "blocked"},
        repair_rows=[],
    )
    graph = exp.build_skill_graph(candidates, skill_graph_path=tmp_path / exp.SKILL_GRAPH_FILE)
    artifact = exp.build_artifact(
        candidates=candidates,
        graph=graph,
        skill_graph_path=tmp_path / exp.SKILL_GRAPH_FILE,
        focused_tests_passed=True,
        project_root=tmp_path,
    )

    assert artifact["status"] == "complete"
    assert artifact["fr11_positive_utility_gate_ready"] is False
    assert artifact["skill_updates_promoted"] == [
        {
            "update_id": "daily_eval:zero",
            "source": "exp1539_external_feedback",
            "utility_delta": 0.0,
            "replay_pass_rate": 1.0,
        }
    ]
    assert artifact["baseline_utility"] == pytest.approx(0.0)
    assert artifact["post_promotion_utility"] == pytest.approx(0.0)
    assert artifact["utility_delta"] == pytest.approx(0.0)
    assert artifact["positive_utility_achieved"] is False
    assert artifact["positive_utility_claim_retired"] is True
    assert "retired" in artifact["honest_verdict"]


def test_scenario_learn_1557_rejects_self_feedback_unsafe_replay_and_mutation() -> None:
    """SCENARIO-LEARN-1557: unsafe or self-only candidates cannot be promoted."""

    candidates = exp.select_candidate_skill_updates(
        exp1539_artifact={"status": "blocked", "candidate_updates": []},
        repair_artifact={"status": "blocked"},
        repair_rows=[],
        extra_candidates=[
            _manual_candidate("self-only", external_feedback=False),
            _manual_candidate("mutates", no_model_weight_mutation=False),
            _manual_candidate("replay-fail", post_replay_passed=False),
            _manual_candidate("false-accept", false_accepts=1),
            _manual_candidate("soundness", soundness_mistakes=1),
        ],
    )
    graph = exp.build_skill_graph(candidates)
    reasons = {
        candidate["update_id"]: candidate["rejection_reasons"]
        for candidate in candidates
    }

    assert graph["nodes"] == []
    assert "self_feedback_only" in reasons["self-only"]
    assert "model_weight_mutation" in reasons["mutates"]
    assert "post_replay_failed" in reasons["replay-fail"]
    assert "false_accepts_positive" in reasons["false-accept"]
    assert "soundness_mistakes_positive" in reasons["soundness"]
    assert all(candidate["promotion_decision"] == "reject" for candidate in candidates)


def test_req_learn_1555_runner_writes_terminal_artifact_and_skill_graph(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1555-2/3/4/5/6/7: runner persists the gate artifacts."""

    paths = _write_sources(tmp_path)
    artifact = exp.run_experiment(
        project_root=tmp_path,
        output_path=paths["output"],
        skill_graph_path=paths["skill_graph"],
        exp1539_artifact_path=paths["exp1539"],
        repair_artifact_path=paths["repair_artifact"],
        repair_manifest_path=paths["repair_manifest"],
        product_line_artifact_paths=[paths["product_line"]],
        focused_tests_passed=True,
    )
    graph = json.loads(paths["skill_graph"].read_text(encoding="utf-8"))

    assert json.loads(paths["output"].read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["focused_tests_passed"] is True
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["positive_utility_achieved"] is True
    assert graph["summary"]["promoted_update_count"] == len(artifact["skill_updates_promoted"])
    assert paths["skill_graph"].exists()


def test_req_learn_1555_missing_sources_and_defensive_guards(tmp_path: Path) -> None:
    """REQ-LEARN-1555-3/6/7: blockers and schema guards fail closed."""

    missing_output = tmp_path / "missing" / exp.OUTPUT_FILE
    missing_graph = tmp_path / "missing" / exp.SKILL_GRAPH_FILE
    blocked = exp.run_experiment(
        project_root=tmp_path,
        output_path=missing_output,
        skill_graph_path=missing_graph,
        exp1539_artifact_path=tmp_path / "missing1539.json",
        repair_artifact_path=tmp_path / "missing1552.json",
        repair_manifest_path=tmp_path / "missing1552.jsonl",
        product_line_artifact_paths=[tmp_path / "missing1554.json"],
        focused_tests_passed=True,
    )

    implicit = exp.select_candidate_skill_updates(
        exp1539_artifact={
            "promoted_updates": ["daily_eval:implicit"],
            "live_sota_model_inference_used": True,
            "no_model_weight_mutation": True,
        },
        repair_artifact={"status": "blocked"},
        repair_rows=[],
    )[0]
    missing_replay = _manual_candidate("missing-replay")
    missing_replay["replay_case_ids"] = []
    rejected = exp.select_candidate_skill_updates(
        exp1539_artifact={"promoted_updates": []},
        repair_artifact={"status": "blocked"},
        repair_rows=[],
        extra_candidates=[
            _manual_candidate("pre-replay", pre_replay_passed=False),
            missing_replay,
        ],
    )
    zero_candidates = exp.select_candidate_skill_updates(
        exp1539_artifact=_exp1539_zero_utility_artifact(),
        repair_artifact={"status": "blocked"},
        repair_rows=[],
    )
    zero_graph = exp.build_skill_graph(
        zero_candidates,
        skill_graph_path=tmp_path / "zero_graph.json",
    )
    zero_artifact = exp.build_artifact(
        candidates=zero_candidates,
        graph=zero_graph,
        skill_graph_path=tmp_path / "zero_graph.json",
        focused_tests_passed=True,
        source_limitations=["bounded_replay_only"],
    )
    bad_no_mutation = dict(zero_artifact, no_model_weight_mutation=False)

    assert blocked["status"] == "blocked"
    assert len(blocked["source_limitations"]) == 4
    assert all(item.startswith("missing:") for item in blocked["source_limitations"])
    assert {Path(item.removeprefix("missing:")).name for item in blocked["source_limitations"]} == {
        "missing1539.json",
        "missing1552.json",
        "missing1552.jsonl",
        "missing1554.json",
    }
    assert blocked["honest_verdict"] == "complete: fr11 positive utility gate blocked"
    assert implicit["update_id"] == "daily_eval:implicit"
    reasons = {candidate["update_id"]: candidate["rejection_reasons"] for candidate in rejected}
    assert "pre_replay_failed" in reasons["pre-replay"]
    assert "missing_replay_cases" in reasons["missing-replay"]
    assert zero_artifact["source_limitations"] == ["bounded_replay_only"]
    with pytest.raises(AssertionError, match="model weights"):
        exp.validate_artifact(bad_no_mutation)
    with pytest.raises(AssertionError, match="skill graph"):
        exp.validate_artifact(zero_artifact, skill_graph_path=tmp_path / "absent_graph.json")


def _exp1539_zero_utility_artifact() -> dict[str, Any]:
    return {
        "status": "complete",
        "continuous_self_learning_task": True,
        "live_sota_model_inference_used": True,
        "no_model_weight_mutation": True,
        "candidate_updates": [
            {
                "policy_update_id": "daily_eval:zero",
                "external_deterministic_feedback": True,
                "promotion_decision": "promote_external_feedback",
                "replay_evidence": {
                    "rollback_decision": "keep",
                    "rollback_soundness_mistakes": 0,
                    "rollback_false_accept_delta": 0,
                    "deterministic_validator_supported": True,
                },
                "verifier_reward": 0.0,
            }
        ],
        "promoted_updates": ["daily_eval:zero"],
        "baseline_task_success_rate": 0.0,
        "promoted_task_success_rate": 0.0,
        "utility_delta": 0.0,
        "soundness_mistakes": 0,
        "skill_graph_path": "results/fr11_external_feedback_skill_graph_1539.json",
    }


def _repair_artifact() -> dict[str, Any]:
    return {
        "status": "complete",
        "residual_drift_repair_ready": True,
        "live_sota_model_inference_used": True,
        "no_model_weight_mutation": True,
        "drift_cases_before": 2,
        "repaired_drift_cases": 2,
        "replay_pass_rate": 1.0,
        "false_accept_rate": 0.0,
        "rejected_false_accept_repairs": 0,
        "focused_tests_passed": True,
        "repair_policy_path": "python/carnot/verify/residual_drift_repair_policy.py",
    }


def _repair_row(case_id: str, *, accepted: bool) -> dict[str, Any]:
    return {
        "row_type": "residual_drift_repair_case",
        "case_id": case_id,
        "source_domain": "runtime_contract",
        "failure_classification": "satisfiable_drift",
        "attempted": True,
        "localized": True,
        "accepted": accepted,
        "replay_passed": accepted,
        "false_accept": False,
        "rejected_false_accept": False,
        "contradiction_untouched": False,
        "replay": {"validator": "runtime_contract", "passed": accepted},
    }


def _manual_candidate(
    update_id: str,
    *,
    external_feedback: bool = True,
    no_model_weight_mutation: bool = True,
    pre_replay_passed: bool = True,
    post_replay_passed: bool = True,
    false_accepts: int = 0,
    soundness_mistakes: int = 0,
) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "source": "fixture",
        "external_feedback": external_feedback,
        "self_feedback_only": not external_feedback,
        "no_model_weight_mutation": no_model_weight_mutation,
        "pre_replay_passed": pre_replay_passed,
        "post_replay_passed": post_replay_passed,
        "false_accepts": false_accepts,
        "soundness_mistakes": soundness_mistakes,
        "baseline_utility": 0.0,
        "post_promotion_utility": 1.0,
        "replay_case_ids": [f"{update_id}-case"],
        "lineage": {"source_artifacts": ["fixture"]},
    }


def _write_sources(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "output": tmp_path / exp.OUTPUT_FILE,
        "skill_graph": tmp_path / exp.SKILL_GRAPH_FILE,
        "exp1539": tmp_path / "experiment_1539.json",
        "repair_artifact": tmp_path / "experiment_1552.json",
        "repair_manifest": tmp_path / "repair.jsonl",
        "product_line": tmp_path / "experiment_1554.json",
    }
    _write_json(paths["exp1539"], _exp1539_zero_utility_artifact())
    _write_json(paths["repair_artifact"], _repair_artifact())
    _write_jsonl(
        paths["repair_manifest"],
        [_repair_row("heldout-1", accepted=True), _repair_row("heldout-2", accepted=True)],
    )
    _write_json(
        paths["product_line"],
        {
            "status": "complete",
            "live_sota_model_inference_used": True,
            "false_accept_rate": 0.0,
            "oracle_agreement_rate": 1.0,
        },
    )
    return paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
