"""Tests for Exp 1568 FR-11 v14 retained-policy mode-collapse audit.

Spec: REQ-LEARN-1568, SCENARIO-LEARN-1568, SCENARIO-LEARN-1569.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fr11_v14_retained_mode_collapse_audit as exp


def test_scenario_learn_1568_multiple_predictors_drive_reversal() -> None:
    """SCENARIO-LEARN-1568: two or more confirmed predictors flag a retention."""

    collapsed = exp.audit_retained_policy(
        policy_id="policy:collapsed",
        source="fixture",
        generated_repairs=[
            "return cached_patch\n" * 16,
            "return cached_patch\n" * 15 + "return cached_patch_final\n",
        ],
        baseline_repairs=[
            "def solve(values):\n    total = sum(v * i for i, v in enumerate(values))\n"
            "    return {'answer': total, 'trace': list(values)}\n",
        ],
        training_corpus=[
            "return cached_patch\n" * 8,
            "cached_patch cached_patch cached_patch cached_patch",
        ],
        reward_groups=[[1.0] * 8, [1.0] * 8],
        baseline_ood_accuracy=0.75,
        post_ood_accuracy=0.50,
    )
    retained = exp.audit_retained_policy(
        policy_id="policy:healthy",
        source="fixture",
        generated_repairs=[
            "def repair_alpha(x):\n    return x + 1\n",
            "def repair_beta(items):\n    return [item.strip() for item in items]\n",
        ],
        baseline_repairs=[
            "def old_alpha(x):\n    return x\n",
            "def old_beta(items):\n    return list(items)\n",
        ],
        training_corpus=["while True:\n    break\n"],
        reward_groups=[[0.0, 1.0] * 4],
        baseline_ood_accuracy=0.60,
        post_ood_accuracy=0.70,
    )

    artifact = exp.build_artifact(
        retained_policy_audits=[collapsed, retained],
        source_limitations=[],
    )

    assert collapsed["confirmed_predictor_count"] >= 2
    assert collapsed["mode_collapse_confirmed"] is True
    assert retained["mode_collapse_confirmed"] is False
    assert artifact["status"] == "complete"
    assert artifact["mode_collapse_audit_complete"] is True
    assert artifact["retained_policies_audited_count"] == 2
    assert artifact["mode_collapse_confirmed_count"] == 1
    assert artifact["mode_collapse_confirmed_percent"] == pytest.approx(0.5)
    assert artifact["reversal_recommended_count"] == 1
    assert artifact["retention_reversal_recommended_policy_ids"] == ["policy:collapsed"]
    assert artifact["honest_verdict"].startswith("complete:")
    exp.validate_artifact(artifact)


def test_scenario_learn_1569_unavailable_evidence_is_not_confirmed() -> None:
    """SCENARIO-LEARN-1569: missing evidence remains unavailable, not positive."""

    audit = exp.audit_retained_policy(
        policy_id="policy:limited",
        source="fixture",
        generated_repairs=["return answer"],
    )
    artifact = exp.build_artifact(
        retained_policy_audits=[audit],
        source_limitations=["pre_rl_baseline_unavailable:policy:limited"],
    )

    predictors = audit["predictors"]

    assert predictors["token_entropy_drop"]["available"] is False
    assert predictors["boilerplate_fraction"]["available"] is False
    assert predictors["reward_variance_collapse"]["available"] is False
    assert predictors["ood_adversarial_accuracy_regression"]["available"] is False
    assert audit["confirmed_predictor_count"] == 0
    assert audit["mode_collapse_confirmed"] is False
    assert artifact["retained_policy_target_count"] == 5
    assert artifact["retained_policy_target_met"] is False
    assert artifact["mode_collapse_confirmed_count"] == 0
    assert artifact["reversal_recommended_count"] == 0
    assert artifact["source_limitations"] == ["pre_rl_baseline_unavailable:policy:limited"]


def test_req_learn_1568_runner_snapshots_exp1555_retained_policies(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1568-1/2/3/7: runner persists the complete audit artifact."""

    paths = _write_sources(tmp_path)

    artifact = exp.run_experiment(
        project_root=tmp_path,
        output_path=paths["output"],
        exp1555_artifact_path=paths["exp1555"],
        skill_graph_path=paths["skill_graph"],
        exp1539_artifact_path=paths["exp1539"],
        repair_artifact_path=paths["repair_artifact"],
        repair_manifest_path=paths["repair_manifest"],
    )
    saved = json.loads(paths["output"].read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["mode_collapse_audit_complete"] is True
    assert artifact["retained_policies_audited_count"] == 2
    assert artifact["retained_policy_target_met"] is False
    assert any(
        item == "retained_policy_target_not_met:2_of_5"
        for item in artifact["source_limitations"]
    )
    assert all(
        set(exp.REQUIRED_POLICY_AUDIT_FIELDS) <= set(policy)
        for policy in artifact["retained_policy_audits"]
    )
    assert artifact["honest_verdict"].startswith("complete:")
    exp.validate_artifact(artifact)

    with pytest.raises(AssertionError, match="required fields"):
        exp.validate_artifact({field: artifact[field] for field in exp.REQUIRED_ARTIFACT_FIELDS[1:]})


def test_req_learn_1568_defensive_missing_source_and_schema_guards(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1568-2/3/7: missing sources and schema drift fail honestly."""

    retained = exp.snapshot_retained_policies(
        {
            "skill_updates_promoted": [
                {"update_id": "policy:fallback", "source": "exp1555"},
                {"update_id": "", "source": "exp1555"},
            ]
        },
        {
            "nodes": [
                {"promotion_decision": {"update_id": ""}},
                {
                    "update_id": "policy:string-promoted",
                    "source": "graph",
                    "promotion_decision": "promote_external_feedback",
                },
                {
                    "update_id": "policy:string-promoted",
                    "source": "graph",
                    "promotion_decision": "promote_external_feedback",
                },
                {
                    "update_id": "policy:rejected",
                    "source": "graph",
                    "promotion_decision": "reject",
                },
            ]
        },
    )
    tiny_text = exp.audit_retained_policy(
        policy_id="policy:tiny",
        source="fixture",
        generated_repairs=["x"],
        baseline_repairs=[""],
        training_corpus=["x"],
        reward_groups=[[0.0] * 8],
    )
    missing = exp.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / "missing_sources.json",
        exp1555_artifact_path=tmp_path / "absent1555.json",
        skill_graph_path=tmp_path / "absent_graph.json",
        exp1539_artifact_path=tmp_path / "absent1539.json",
        repair_artifact_path=tmp_path / "absent1552.json",
        repair_manifest_path=tmp_path / "absent1552.jsonl",
    )
    valid = exp.build_artifact(retained_policy_audits=[tiny_text], source_limitations=[])

    assert retained == [
        {"policy_id": "policy:string-promoted", "source": "graph", "node_id": ""},
        {"policy_id": "policy:fallback", "source": "exp1555", "node_id": ""},
    ]
    assert tiny_text["predictors"]["token_entropy_drop"]["generated_mean_nats_per_token"] == 0.0
    assert tiny_text["predictors"]["boilerplate_fraction"]["boilerplate_fraction"] == 0.0
    assert exp._variance([]) == 0.0
    assert exp._find_exp1539_candidate({}, "policy:absent") == {}
    assert missing["retained_policies_audited_count"] == 0
    assert missing["mode_collapse_confirmed_percent"] == 0.0
    assert "no_exp1555_retained_policies_found" in missing["source_limitations"]
    assert all(item.startswith("missing:") for item in missing["source_limitations"][:5])

    with pytest.raises(AssertionError, match="complete"):
        exp.validate_artifact(dict(valid, status="blocked"))
    with pytest.raises(AssertionError, match="honest_verdict"):
        exp.validate_artifact(dict(valid, honest_verdict="blocked"))
    with pytest.raises(AssertionError, match="audited_count"):
        exp.validate_artifact(dict(valid, retained_policies_audited_count=3))
    bad_policy = dict(valid)
    bad_policy["retained_policy_audits"] = [{"policy_id": "incomplete"}]
    with pytest.raises(AssertionError, match="policy audit fields"):
        exp.validate_artifact(bad_policy)
    with pytest.raises(AssertionError, match="confirmed_count"):
        exp.validate_artifact(dict(valid, mode_collapse_confirmed_count=9))
    with pytest.raises(AssertionError, match="reversal_recommended_count"):
        exp.validate_artifact(dict(valid, reversal_recommended_count=9))


def _write_sources(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "output": tmp_path / exp.OUTPUT_FILE,
        "exp1555": tmp_path / "experiment_1555.json",
        "skill_graph": tmp_path / "skill_graph.json",
        "exp1539": tmp_path / "experiment_1539.json",
        "repair_artifact": tmp_path / "experiment_1552.json",
        "repair_manifest": tmp_path / "repair.jsonl",
    }
    _write_json(
        paths["exp1555"],
        {
            "status": "complete",
            "skill_updates_promoted": [
                {"update_id": "daily_eval:zero", "source": "exp1539_external_feedback"},
                {"update_id": "policy:residual_drift_repair:1552", "source": "exp1552_residual_drift_repair"},
            ],
            "candidate_skill_updates": [],
        },
    )
    _write_json(
        paths["skill_graph"],
        {
            "nodes": [
                {
                    "node_id": "skill:fr11_v14/daily_eval-zero",
                    "update_id": "daily_eval:zero",
                    "source": "exp1539_external_feedback",
                    "promotion_decision": {"update_id": "daily_eval:zero"},
                },
                {
                    "node_id": "skill:fr11_v14/policy-residual",
                    "update_id": "policy:residual_drift_repair:1552",
                    "source": "exp1552_residual_drift_repair",
                    "promotion_decision": {"update_id": "policy:residual_drift_repair:1552"},
                },
            ]
        },
    )
    _write_json(
        paths["exp1539"],
        {
            "candidate_updates": [
                {
                    "policy_update_id": "daily_eval:zero",
                    "model_outputs": {
                        "baseline_excerpt": "def baseline(x):\n    return x + 1\n",
                        "promoted_excerpt": "return cached cached cached cached cached\n",
                    },
                    "verifier_reward": 0.0,
                    "baseline_task_success_rate": 0.0,
                    "promoted_task_success_rate": 0.0,
                }
            ],
            "baseline_task_success_rate": 0.0,
            "promoted_task_success_rate": 0.0,
        },
    )
    _write_json(
        paths["repair_artifact"],
        {
            "model_probe": {
                "proposal_output_excerpt": "return cached_patch\n" * 8,
            },
            "repair_manifest_path": paths["repair_manifest"].as_posix(),
        },
    )
    _write_jsonl(
        paths["repair_manifest"],
        [
            _repair_row(f"heldout-{idx}", "return cached_patch\n" * 8)
            for idx in range(10)
        ],
    )
    return paths


def _repair_row(case_id: str, proposal: str) -> dict[str, Any]:
    return {
        "row_type": "residual_drift_repair_case",
        "case_id": case_id,
        "accepted": True,
        "replay_passed": True,
        "false_accept": False,
        "proposal": {"model_proposal_excerpt": proposal},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
