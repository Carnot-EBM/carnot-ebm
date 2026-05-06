"""Tests for Exp 1398 NGRPO Advantage Calibration theory probe.

Spec: REQ-LEARN-1398, SCENARIO-LEARN-1398.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import experiment_1398_ngrpo_theory_probe as mod


def _exp1383_payload() -> dict[str, object]:
    return {
        "experiment": "1383_grpo_v7_jury_rl_formal_verifier_rewards",
        "run_date": "20260505",
        "status": "complete",
        "formal_reward_pass_rate": 0.0,
        "grpo_v7_improvement_pp": 0.0,
        "honest_verdict": "grpo_v7_jury_rl_no_improvement",
        "training_reward_rows": [
            {
                "case_id": f"train_{index}",
                "rollout_answers": ["UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"],
                "rewards": [0.0, 0.0, 0.0, 0.0],
                "reward_mean": 0.0,
                "reszero_applied": True,
            }
            for index in range(4)
        ],
        "heldout_evaluation_rows": [
            {
                "case_id": f"heldout_{index}",
                "rollout_answers": ["UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"],
            }
            for index in range(4)
        ],
    }


def test_req_learn_1398_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1398-1: the probe starts with a visible in-progress artifact."""

    output_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(
        output_path,
        project_root=tmp_path,
        run_date="20260506",
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["honest_verdict"] == "in_progress"


def test_req_learn_1398_exp1383_summary_confirms_all_unknown_rewards() -> None:
    """REQ-LEARN-1398-2: exp1383 data summary records all-UNKNOWN zero rewards."""

    summary = mod.summarize_exp1383_rollouts(_exp1383_payload())

    assert summary["all_rollouts_unknown"] is True
    assert summary["formal_reward_pass_rate"] == 0.0
    assert summary["training_group_count"] == 4
    assert summary["rollouts_per_training_group"] == 4
    assert summary["training_reward_distribution"] == {"0.0": 16}
    assert summary["training_rollout_answer_distribution"] == {"UNKNOWN": 16}
    assert summary["heldout_rollout_answer_distribution"] == {"UNKNOWN": 16}


def test_scenario_learn_1398_virtual_sample_breaks_zero_reward_symmetry() -> None:
    """SCENARIO-LEARN-1398: virtual reward creates non-zero advantages."""

    reszero = mod.simulate_reszero_advantages([0.0, 0.0, 0.0, 0.0])
    ngrpo = mod.simulate_ngrpo_advantage_calibration(
        [0.0, 0.0, 0.0, 0.0],
        virtual_reward=1.0,
    )

    assert reszero["mean_reward"] == 0.0
    assert reszero["advantages"] == [0.0, 0.0, 0.0, 0.0]
    assert reszero["advantage_variance"] == 0.0
    assert ngrpo["augmented_mean_reward"] == pytest.approx(0.2)
    assert ngrpo["real_advantages"] == [-0.2, -0.2, -0.2, -0.2]
    assert ngrpo["virtual_advantage"] == pytest.approx(0.8)
    assert ngrpo["augmented_advantages"] == [-0.2, -0.2, -0.2, -0.2, 0.8]
    assert ngrpo["augmented_advantage_variance"] == pytest.approx(0.16)


def test_req_learn_1398_run_writes_terminal_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-1398-5: runner emits every required artifact field."""

    exp1383_path = tmp_path / "experiment_1383.json"
    output_path = tmp_path / mod.OUTPUT_FILE
    exp1383_path.write_text(json.dumps(_exp1383_payload()), encoding="utf-8")
    writes: list[dict[str, object]] = []

    artifact = mod.run_experiment(
        exp1383_path=exp1383_path,
        output_path=output_path,
        project_root=tmp_path,
        run_date="20260506",
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "complete"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["exp1383_rollout_data_used"]["all_rollouts_unknown"] is True
    assert artifact["original_resZero_advantage_variance"] == 0.0
    assert artifact["ngrpo_virtual_sample_reward"] == 1.0
    assert artifact["ngrpo_augmented_advantage_variance"] == pytest.approx(0.16)
    assert artifact["ngrpo_advantage_calibration_verified"] is True
    assert artifact["ngrpo_expected_gradient_magnitude"] == pytest.approx(0.16)
    assert artifact["theory_supports_exp1393"] is True
