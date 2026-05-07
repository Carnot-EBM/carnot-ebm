"""Tests for Exp 1485 FR-11 completeness-reduction audit.

Spec: REQ-LEARN-1485, SCENARIO-LEARN-1486, SCENARIO-LEARN-1487.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fr11_completeness_reduction_audit as mod


def _decision(
    case_id: str,
    *,
    completeness_mistake: bool,
    soundness_mistake: bool = False,
    memory_hit: bool = False,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "memory_enabled": memory_hit,
        "memory_hit": memory_hit,
        "verifier_signal": "verified_memory_repair_hint" if memory_hit else "baseline_verifier_only",
        "task_success": not completeness_mistake and not soundness_mistake,
        "soundness_mistake": soundness_mistake,
        "completeness_mistake": completeness_mistake,
    }


def _eval(decisions: list[dict[str, Any]], *, memory_enabled: bool) -> dict[str, Any]:
    return {
        "memory_enabled": memory_enabled,
        "task_success_rate": sum(1 for item in decisions if item["task_success"])
        / len(decisions),
        "soundness_mistakes": sum(1 for item in decisions if item["soundness_mistake"]),
        "completeness_mistakes": sum(1 for item in decisions if item["completeness_mistake"]),
        "decisions": decisions,
    }


def _exp1484(*, memory_soundness: int = 0) -> dict[str, Any]:
    baseline_decisions = [
        _decision("positive-a", completeness_mistake=True),
        _decision("positive-b", completeness_mistake=True),
        _decision("negative-a", completeness_mistake=False),
        _decision("negative-b", completeness_mistake=False),
    ]
    memory_decisions = [
        _decision("positive-a", completeness_mistake=False, memory_hit=True),
        _decision("positive-b", completeness_mistake=False, memory_hit=True),
        _decision("negative-a", completeness_mistake=False),
        _decision(
            "negative-b",
            completeness_mistake=False,
            soundness_mistake=memory_soundness > 0,
            memory_hit=memory_soundness > 0,
        ),
    ]
    return {
        "experiment": "1484_fr11_v9_query_time_memory_policy",
        "status": "complete",
        "memory_policy_replay": {
            "baseline_memory_disabled": _eval(baseline_decisions, memory_enabled=False),
            "memory_enabled": _eval(memory_decisions, memory_enabled=True),
            "bounded_replay_pairs": 2,
        },
    }


def test_req_learn_1485_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1485-1/4: bootstrap artifact exposes required fields first."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "in_progress"
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["completeness_reduction_audit_complete"] is False
    assert artifact["honest_verdict"] == "in_progress"


def test_scenario_learn_1486_selects_verified_memory_candidate() -> None:
    """SCENARIO-LEARN-1486: verified-memory routing reduces false rejects."""

    artifact = mod.build_artifact(
        exp1484_artifact=_exp1484(),
        project_root="/repo",
        commands_run=["pytest targeted"],
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["source_experiment"] == mod.SOURCE_EXPERIMENT
    assert artifact["baseline_completeness_mistakes"] == 2
    assert artifact["candidate_completeness_mistakes"] == 0
    assert artifact["completeness_mistake_delta"] == -2
    assert artifact["baseline_soundness_mistakes"] == 0
    assert artifact["candidate_soundness_mistakes"] == 0
    assert artifact["candidate_policy"]["name"] == "exp1484_opt_in_verified_memory_enabled"
    assert artifact["policy_change_allowed"] is True
    assert artifact["tests_run"] == ["pytest targeted"]
    assert artifact["honest_verdict"] == mod.ALLOWED_VERDICT


def test_scenario_learn_1487_rejects_unsafe_broad_routing() -> None:
    """SCENARIO-LEARN-1487: unsafe negative-control accepts cannot win selection."""

    artifact = mod.build_artifact(exp1484_artifact=_exp1484(), project_root="/repo")

    unsafe = next(
        item
        for item in artifact["candidate_variants"]
        if item["name"] == "unsafe_accept_all_replay_ids"
    )

    assert unsafe["allowed"] is False
    assert unsafe["candidate_soundness_mistakes"] == 2
    assert artifact["candidate_policy"]["name"] != unsafe["name"]
    assert artifact["candidate_policy"]["allowed"] is True
    assert artifact["policy_change_allowed"] is True


def test_req_learn_1485_unsafe_memory_candidate_falls_back_to_baseline() -> None:
    """REQ-LEARN-1485-3/5: a candidate above the soundness gate is rejected."""

    artifact = mod.build_artifact(exp1484_artifact=_exp1484(memory_soundness=1), project_root="/repo")

    mod.validate_artifact(artifact)
    assert artifact["candidate_policy"]["name"] == "baseline_memory_disabled"
    assert artifact["candidate_completeness_mistakes"] == 2
    assert artifact["candidate_soundness_mistakes"] == 0
    assert artifact["completeness_mistake_delta"] == 0
    assert artifact["policy_change_allowed"] is False
    assert artifact["honest_verdict"] == mod.NO_ALLOWED_REDUCTION_VERDICT


def test_req_learn_1485_run_writes_terminal_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-1485-1/2/4: run loads Exp 1484 and writes the final audit."""

    source_path = tmp_path / "results" / "experiment_1484.json"
    out_path = tmp_path / "results" / mod.OUTPUT_FILE
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(_exp1484(), sort_keys=True), encoding="utf-8")

    artifact = mod.run(
        exp1484_path=source_path,
        out_path=out_path,
        project_root=tmp_path,
        commands_run=["pytest targeted"],
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["tests_run"] == ["pytest targeted"]


def test_req_learn_1485_validation_rejects_bad_contract() -> None:
    """REQ-LEARN-1485-4/5: validation enforces soundness and delta invariants."""

    artifact = mod.build_artifact(exp1484_artifact=_exp1484(), project_root="/repo")

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({key: value for key, value in artifact.items() if key != "status"})

    bad_delta = dict(artifact, completeness_mistake_delta=999)
    with pytest.raises(AssertionError, match="completeness_mistake_delta"):
        mod.validate_artifact(bad_delta)

    bad_soundness = dict(artifact, candidate_soundness_mistakes=1)
    with pytest.raises(AssertionError, match="candidate soundness"):
        mod.validate_artifact(bad_soundness)

    bad_policy_gate = dict(artifact, policy_change_allowed=False)
    with pytest.raises(AssertionError, match="policy_change_allowed"):
        mod.validate_artifact(bad_policy_gate)

    malformed_replay = dict(_exp1484())
    malformed_replay["memory_policy_replay"] = {}
    with pytest.raises(AssertionError, match="baseline_memory_disabled"):
        mod.build_artifact(exp1484_artifact=malformed_replay, project_root="/repo")

    with pytest.raises(AssertionError, match="memory_policy_replay"):
        mod.build_artifact(exp1484_artifact={}, project_root="/repo")

    assert mod._allowed_gate(
        candidate_soundness_mistakes=0,
        baseline_soundness_mistakes=-1,
    ) == (False, "candidate_soundness_mistakes_exceeds_baseline")
