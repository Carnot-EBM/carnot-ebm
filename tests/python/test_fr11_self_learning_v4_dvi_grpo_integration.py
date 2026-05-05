"""Tests for Exp 1388 FR-11 self-learning v4 DVI/GRPO integration.

Spec: REQ-LEARN-1388, SCENARIO-LEARN-1388.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import fr11_self_learning_v4_dvi_grpo_integration as mod


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            metric=np.ones(8, dtype=np.float32),
            bias=np.array([0.0], dtype=np.float32),
        )


def _exp1374() -> dict[str, Any]:
    return {
        "status": "complete",
        "path_used": "primary_semantic_verified",
        "replay_cases_used": 10,
        "fresh_verified_sample_count": 4,
        "self_learning_delta_overall": 1.0,
        "nonforgetting_certificate_rate": 1.0,
        "memory_regression_count": 0,
        "accepted_violation_delta": -0.5,
        "promoted_memory_count": 4,
        "demoted_memory_count": 0,
        "honest_verdict": "continuous_self_learning_v3_primary_semantic_verified",
        "variant_questions": [
            {
                "variant_id": f"semantic:exp1374:{index}",
                "case_id": f"base_{index}",
                "verifier_accepted": True,
                "semantic_rejected": False,
                "memory_action": "promote",
                "support": 1,
            }
            for index in range(4)
        ],
    }


def _exp1381(checkpoint_path: Path, *, dvi_auroc_delta: float = 0.2) -> dict[str, Any]:
    return {
        "status": "complete",
        "dvi_deployed": True,
        "dvi_checkpoint_path": str(checkpoint_path),
        "dvi_baseline_auroc": 0.4,
        "dvi_trained_auroc": 0.4 + dvi_auroc_delta,
        "dvi_auroc_delta": dvi_auroc_delta,
        "honest_verdict": "dvi_discriminative_improvement_measured_positive_delta",
    }


def _semantic_row(case_id: str, *, passed: bool) -> dict[str, Any]:
    expected = "SAT" if passed else "REPAIR_HINT"
    return {
        "case_id": case_id,
        "claim_route": "dvi_updated_fover_semantic_validator",
        "expected_state": expected,
        "certificate_state": expected,
        "semantic_result": expected if passed else "SAT",
        "constraint_evaluated": True,
        "constraint_passed": passed,
        "dvi_incorrect_probability": 0.1 if passed else 0.8,
        "dvi_incorrect_threshold": 0.72,
        "fover_label": "correct" if passed else "incorrect",
        "failure_reason": None if passed else "dvi_disagrees_with_fover_label",
    }


def _exp1382(*, passed: int = 6, failed: int = 2) -> dict[str, Any]:
    rows = [_semantic_row(f"pass_{index}", passed=True) for index in range(passed)]
    rows.extend(_semantic_row(f"fail_{index}", passed=False) for index in range(failed))
    return {
        "status": "complete",
        "total_fover_cases": len(rows),
        "semantic_validation_rows": rows,
        "scheduler_false_acceptance_rate": 0.0,
        "honest_verdict": "fullscale_pipeline_headline_allowed_parse_rate_1_0",
    }


def _exp1383(*, improvement: float = 0.0) -> dict[str, Any]:
    return {
        "status": "complete",
        "grpo_v7_improvement_pp": improvement,
        "training_reward_rows": [
            {
                "case_id": "grpo_verified",
                "candidate_answer": "REPAIR_HINT",
                "expected_answer": "REPAIR_HINT",
                "verifier_result": "VERIFIED",
            }
        ],
        "heldout_evaluation_rows": [
            {
                "case_id": "grpo_heldout",
                "expected_answer": "SAT",
                "post_grpo_verifier_result": "SAT",
            }
        ],
        "honest_verdict": "grpo_v7_jury_rl_positive_improvement",
    }


def test_req_learn_1388_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1388-1: bootstrap output exists before input loading."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["path_used"] is None
    assert written["dvi_checkpoint_active"] is False
    assert written["fresh_verified_sample_count"] == 0
    assert written["headline_result_allowed"] is False


def test_req_learn_1388_dvi_only_replay_promotes_exp1382_cases(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1388: no positive GRPO gate uses DVI-only replay."""

    checkpoint_path = tmp_path / "models" / "dvi_checkpoint_v1.pt"
    _write_checkpoint(checkpoint_path)

    artifact = mod.build_artifact(
        exp1374_artifact=_exp1374(),
        exp1381_artifact=_exp1381(checkpoint_path),
        exp1382_artifact=_exp1382(passed=6, failed=2),
        exp1383_artifact=_exp1383(improvement=0.0),
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["path_used"] == mod.PATH_DVI_ONLY
    assert artifact["dvi_checkpoint_active"] is True
    assert artifact["replay_cases_used"] == 10
    assert artifact["fresh_verified_sample_count"] == 6
    assert artifact["fresh_verified_delta_vs_exp1374"] == 2
    assert artifact["grpo_cases_integrated"] == 0
    assert artifact["promoted_memory_count"] == 10
    assert artifact["demoted_memory_count"] == 2
    assert artifact["self_learning_delta_overall"] == 1.2
    assert artifact["nonforgetting_certificate_rate"] == 1.0
    assert artifact["memory_regression_count"] == 0
    assert artifact["accepted_violation_delta"] == -0.5
    assert artifact["dvi_auroc_delta_effect"]["quality_improved_vs_exp1374_baseline"] is True
    assert artifact["headline_result_allowed"] is True
    assert (
        artifact["honest_verdict"]
        == "fr11_self_learning_v4_dvi_only_exp1382_headline_allowed_fresh_6"
    )


def test_scenario_learn_1388_positive_grpo_gate_integrates_verified_cases(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1388: positive GRPO improvement adds verified cases."""

    checkpoint_path = tmp_path / "models" / "dvi_checkpoint_v1.pt"
    _write_checkpoint(checkpoint_path)

    artifact = mod.build_artifact(
        exp1374_artifact=_exp1374(),
        exp1381_artifact=_exp1381(checkpoint_path),
        exp1382_artifact=_exp1382(passed=5, failed=0),
        exp1383_artifact=_exp1383(improvement=25.0),
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["path_used"] == mod.PATH_DVI_GRPO
    assert artifact["fresh_verified_sample_count"] == 7
    assert artifact["grpo_cases_integrated"] == 2
    assert artifact["promoted_memory_count"] == 11
    assert artifact["headline_result_allowed"] is True
    assert artifact["honest_verdict"] == (
        "fr11_self_learning_v4_dvi_grpo_integrated_headline_allowed_fresh_7_grpo_2"
    )


def test_req_learn_1388_inactive_dvi_checkpoint_blocks_headline(tmp_path: Path) -> None:
    """REQ-LEARN-1388-2/6: missing DVI checkpoint prevents headline claims."""

    artifact = mod.build_artifact(
        exp1374_artifact=_exp1374(),
        exp1381_artifact=_exp1381(tmp_path / "missing.pt"),
        exp1382_artifact=_exp1382(passed=6, failed=0),
        exp1383_artifact=_exp1383(improvement=0.0),
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["dvi_checkpoint_active"] is False
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == "fr11_self_learning_v4_blocked_dvi_checkpoint_inactive"


def test_req_learn_1388_run_loads_sources_and_writes_complete_artifact(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1388-1/5: runner writes bootstrap then terminal artifact."""

    results = tmp_path / "results"
    results.mkdir()
    checkpoint_path = tmp_path / "models" / "dvi_checkpoint_v1.pt"
    _write_checkpoint(checkpoint_path)
    (results / mod.EXP1374_FILE).write_text(json.dumps(_exp1374()), encoding="utf-8")
    (results / mod.EXP1381_FILE).write_text(
        json.dumps(_exp1381(checkpoint_path)),
        encoding="utf-8",
    )
    (results / mod.EXP1382_FILE).write_text(json.dumps(_exp1382()), encoding="utf-8")
    (results / mod.EXP1383_FILE).write_text(json.dumps(_exp1383(improvement=0.0)), encoding="utf-8")
    out_path = results / mod.OUTPUT_FILE

    artifact = mod.run(results_dir=results, out_path=out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["status"] == "complete"
    assert artifact["path_used"] == mod.PATH_DVI_ONLY
    assert artifact["source_artifacts"] == [
        f"results/{mod.EXP1374_FILE}",
        f"results/{mod.EXP1381_FILE}",
        f"results/{mod.EXP1382_FILE}",
        f"results/{mod.EXP1383_FILE}",
    ]
