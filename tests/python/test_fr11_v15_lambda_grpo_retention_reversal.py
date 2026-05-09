"""Tests for Exp 1581 FR-11 v15 lambda-GRPO retention reversal.

Spec: REQ-LEARN-1581, SCENARIO-LEARN-1581, SCENARIO-LEARN-1582.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fr11_v15_lambda_grpo_retention_reversal as exp


def test_scenario_learn_1581_replay_confirmed_collapse_reverses_retention() -> None:
    """SCENARIO-LEARN-1581: replay-confirmed collapse reverses the v14 retention."""

    rows = [_repair_row(index, "return cached_patch\n" * 8) for index in range(18)]
    replay = exp.replay_flagged_policy(
        exp1568_artifact=_exp1568_recommended(),
        repair_artifact=_repair_artifact(),
        repair_rows=rows,
        lambda_grpo_patch_available=False,
    )
    artifact = exp.build_artifact(replay=replay)

    assert artifact["status"] == "complete"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["flagged_policy_replayed"] is True
    assert artifact["retention_reversal_applied"] is True
    assert artifact["lambda_grpo_patch_implemented"] is False
    assert artifact["lambda_grpo_simulated_only"] is True
    assert artifact["implementation_deferred"] is True
    assert artifact["soundness_mistakes"] == 0
    assert artifact["entropy_preservation_rate"] == 0.0
    assert artifact["replay_boilerplate_fraction"] >= 0.30
    assert artifact["boilerplate_fraction_delta"] < 0.0
    assert artifact["ood_accuracy_proxy"] == 1.0
    assert artifact["fr11_v15_decision_ready"] is True
    assert "boilerplate_fraction" in artifact["replay_confirmed_predictors"]
    assert "reward_variance_collapse" in artifact["replay_confirmed_predictors"]
    assert artifact["retention_reversal_audit_note"]["action"] == "v14_retention_reversed"
    assert artifact["honest_verdict"].startswith("complete:")
    exp.validate_artifact(artifact)


def test_scenario_learn_1582_unconfirmed_replay_blocks_reversal() -> None:
    """SCENARIO-LEARN-1582: fresh replay must still confirm collapse."""

    rows = [
        _repair_row(
            index,
            (
                "train_anchor seed_only corpus_only\n"
                if index < 3
                else f"unique{index} branch{index} answer{index}\n"
            ),
        )
        for index in range(18)
    ]
    replay = exp.replay_flagged_policy(
        exp1568_artifact=_exp1568_recommended(),
        repair_artifact={
            "model_probe": {"proposal_output_excerpt": "def baseline(value):\n    return value\n"}
        },
        repair_rows=rows,
        lambda_grpo_patch_available=True,
    )
    artifact = exp.build_artifact(replay=replay)

    assert artifact["status"] == "complete"
    assert artifact["flagged_policy_replayed"] is True
    assert artifact["lambda_grpo_patch_implemented"] is True
    assert artifact["lambda_grpo_simulated_only"] is False
    assert artifact["implementation_deferred"] is False
    assert artifact["retention_reversal_applied"] is False
    assert artifact["replay_confirmed_predictor_count"] < 2
    assert artifact["fr11_v15_decision_ready"] is True
    assert artifact["retention_reversal_audit_note"]["action"] == "v14_retention_preserved"
    assert "blocked by replay" in artifact["honest_verdict"]


def test_req_learn_1581_lambda_grpo_normalization_prefers_diverse_repairs() -> None:
    """REQ-LEARN-1581-4: corrected weights preserve exploration under equal rewards."""

    normalized = exp.simulate_lambda_grpo_weights(
        [
            {
                "case_id": "collapsed",
                "reward": 1.0,
                "entropy_preservation": 0.0,
                "boilerplate_fraction": 1.0,
                "ood_accuracy_proxy": 1.0,
                "soundness_mistake": 0,
            },
            {
                "case_id": "diverse",
                "reward": 1.0,
                "entropy_preservation": 1.0,
                "boilerplate_fraction": 0.0,
                "ood_accuracy_proxy": 1.0,
                "soundness_mistake": 0,
            },
        ]
    )

    weights = {entry["case_id"]: entry["normalized_weight"] for entry in normalized}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["diverse"] > weights["collapsed"]
    assert normalized[0]["corrected_score"] < normalized[1]["corrected_score"]


def test_req_learn_1581_runner_persists_required_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-1581-1/2/3/5/6: runner writes and validates the artifact."""

    paths = _write_sources(tmp_path, collapsed=True)
    in_progress = exp.write_in_progress_artifact(paths["output"])

    assert in_progress["status"] == "in_progress"

    artifact = exp.run_experiment(
        project_root=tmp_path,
        output_path=paths["output"],
        exp1568_artifact_path=paths["exp1568"],
        repair_artifact_path=paths["repair_artifact"],
        repair_manifest_path=paths["repair_manifest"],
        lambda_grpo_patch_available=False,
    )
    saved = json.loads(paths["output"].read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["flagged_policy_replayed"] is True
    assert artifact["retention_reversal_applied"] is True
    assert artifact["fr11_v15_decision_ready"] is True
    exp.validate_artifact(artifact)

    missing = {field: artifact[field] for field in exp.REQUIRED_ARTIFACT_FIELDS[1:]}
    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact(missing)
    with pytest.raises(AssertionError, match="simulated_only"):
        exp.validate_artifact(
            dict(artifact, lambda_grpo_patch_implemented=True, lambda_grpo_simulated_only=True)
        )


def test_req_learn_1581_missing_recommendation_fails_honestly(tmp_path: Path) -> None:
    """REQ-LEARN-1581-2: absent Exp 1568 reversal evidence blocks the decision."""

    paths = _write_sources(tmp_path, collapsed=True)
    _write_json(paths["exp1568"], {"retention_reversal_recommended_policy_ids": []})

    artifact = exp.run_experiment(
        project_root=tmp_path,
        output_path=paths["output"],
        exp1568_artifact_path=paths["exp1568"],
        repair_artifact_path=paths["repair_artifact"],
        repair_manifest_path=paths["repair_manifest"],
        lambda_grpo_patch_available=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["flagged_policy_replayed"] is False
    assert artifact["retention_reversal_applied"] is False
    assert artifact["fr11_v15_decision_ready"] is False
    assert "not recommended" in artifact["honest_verdict"]


def test_req_learn_1581_defensive_guards_and_edge_metrics() -> None:
    """REQ-LEARN-1581-5/6: schema guards and metric edges fail closed."""

    rows = [_repair_row(index, "return cached_patch\n" * 8) for index in range(18)]
    base = exp.build_artifact(
        replay=exp.replay_flagged_policy(
            exp1568_artifact=_exp1568_recommended(),
            repair_artifact=_repair_artifact(),
            repair_rows=rows,
            lambda_grpo_patch_available=False,
        )
    )

    with pytest.raises(AssertionError, match="status"):
        exp.validate_artifact(dict(base, status="done"))
    with pytest.raises(AssertionError, match="continuous_self_learning_task"):
        exp.validate_artifact(dict(base, continuous_self_learning_task=False))
    with pytest.raises(AssertionError, match="flagged_policy_replayed"):
        exp.validate_artifact(dict(base, flagged_policy_replayed=False))
    with pytest.raises(AssertionError, match="zero soundness"):
        exp.validate_artifact(dict(base, soundness_mistakes=1))
    with pytest.raises(AssertionError, match="model weight"):
        exp.validate_artifact(dict(base, no_model_weight_mutation=False))
    with pytest.raises(AssertionError, match="at least two"):
        exp.validate_artifact(dict(base, replay_confirmed_predictor_count=1))

    safety_blocked = exp.build_artifact(
        replay={
            "flagged_policy_id": exp.FLAGGED_POLICY_ID,
            "flagged_policy_replayed": True,
            "lambda_grpo_patch_implemented": True,
            "lambda_grpo_simulated_only": False,
            "no_model_weight_mutation": True,
            "soundness_mistakes": 1,
            "replay_mode_collapse_confirmed": True,
            "replay_confirmed_predictors": ["boilerplate_fraction", "reward_variance_collapse"],
            "replay_confirmed_predictor_count": 2,
        }
    )

    assert "safety gates" in safety_blocked["honest_verdict"]
    assert exp._text_distribution_entropy_rate([]) == 0.0
    assert exp._text_distribution_entropy_rate(["only one"]) == 1.0
    assert exp._boilerplate_fraction(["x"], ["x"]) == 0.0
    assert exp._token_entropy("") == 0.0


def _exp1568_recommended() -> dict[str, Any]:
    return {"retention_reversal_recommended_policy_ids": [exp.FLAGGED_POLICY_ID]}


def _repair_artifact() -> dict[str, Any]:
    return {"model_probe": {"proposal_output_excerpt": "return cached_patch\n" * 8}}


def _repair_row(index: int, proposal_text: str) -> dict[str, Any]:
    return {
        "row_type": "residual_drift_repair_case",
        "case_id": f"heldout-{index:03d}",
        "source_domain": "runtime_contract" if index >= 9 else "satquest",
        "accepted": True,
        "false_accept": False,
        "replay_passed": True,
        "proposal": {
            "model_proposal_excerpt": proposal_text,
            "localized_span": f"commitments[{index}].evidence.answer",
            "replacement": {"answer": f"repair-{index}", "index": index},
        },
        "replay": {"passed": True, "false_accept": False},
    }


def _write_sources(tmp_path: Path, *, collapsed: bool) -> dict[str, Path]:
    paths = {
        "output": tmp_path / exp.OUTPUT_FILE,
        "exp1568": tmp_path / "experiment_1568.json",
        "repair_artifact": tmp_path / "experiment_1552.json",
        "repair_manifest": tmp_path / "repair.jsonl",
    }
    proposal = "return cached_patch\n" * 8
    rows = [
        _repair_row(
            index,
            proposal if collapsed else f"def repair_{index}(value):\n    return value + {index}\n",
        )
        for index in range(18)
    ]
    _write_json(paths["exp1568"], _exp1568_recommended())
    _write_json(paths["repair_artifact"], _repair_artifact())
    _write_jsonl(paths["repair_manifest"], rows)
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
