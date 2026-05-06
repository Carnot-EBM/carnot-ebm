"""Tests for Exp 1395 FR-11 self-learning v5.

Spec: REQ-LEARN-1395, SCENARIO-LEARN-1395.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import fr11_self_learning_v5 as mod


def _write_checkpoint(path: Path, *, bias: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            metric=np.zeros(128, dtype=np.float32),
            bias=np.asarray([bias], dtype=np.float32),
            secl_bin_values=np.full(10, 0.5, dtype=np.float32),
            secl_global_value=np.asarray([0.5], dtype=np.float32),
            secl_n_bins=np.asarray([10], dtype=np.int32),
            fresh_cases_used=np.asarray([59], dtype=np.int32),
        )


def _exp1388() -> dict[str, Any]:
    return {
        "status": "complete",
        "fresh_verified_sample_count": 59,
        "replay_cases_used": 282,
        "memory_updates": {"promoted": ["dvi:exp1382:old_promoted"]},
        "honest_verdict": "fr11_self_learning_v4_dvi_only_exp1382_headline_allowed_fresh_59",
    }


def _exp1394(checkpoint_path: Path) -> dict[str, Any]:
    return {
        "status": "complete",
        "dvi_v2_deployed": True,
        "checkpoint_path": str(checkpoint_path),
        "fresh_cases_used": 59,
        "honest_verdict": "dvi_v2_secl_combined_deployed_positive_auroc_delta_ece_reduced",
    }


def _exp1393(*, improvement: float = 0.0) -> dict[str, Any]:
    return {
        "status": "complete",
        "grpo_v8_improvement_pp": improvement,
        "training_reward_rows": [
            {
                "case_id": "train_verified",
                "candidate_answer": "SAT",
                "expected_answer": "SAT",
                "verifier_result": "SAT",
            }
        ],
        "heldout_evaluation_rows": [
            {
                "case_id": "heldout_verified",
                "expected_answer": "REPAIR_HINT",
                "post_grpo_answer": "REPAIR_HINT",
                "post_grpo_verifier_result": "REPAIR_HINT",
            }
        ],
        "honest_verdict": "grpo_v8_ngrpo_positive_improvement_1pp",
    }


def _fover_rows(*, new_incorrect: int = 60) -> list[dict[str, Any]]:
    rows = [
        {
            "question_id": "old_promoted",
            "step_text": "Old incorrect step already promoted by Exp 1388.",
            "label": "incorrect",
        }
    ]
    rows.extend(
        {
            "question_id": f"new_{index}",
            "step_text": f"Fresh incorrect FoVer step {index}.",
            "label": "incorrect",
        }
        for index in range(new_incorrect)
    )
    rows.append(
        {
            "question_id": "correct_rejected",
            "step_text": "A correct row is rejected by this all-incorrect test checkpoint.",
            "label": "correct",
        }
    )
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_req_learn_1395_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1395-1: bootstrap output exists before source loading."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["path_used"] is None
    assert written["dvi_v2_checkpoint_active"] is False
    assert written["fresh_verified_sample_count"] == 0
    assert written["headline_result_allowed"] is False


def test_scenario_learn_1395_dvi_v2_secl_promotes_fresh_fover_cases(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1395: DVI v2 + SECL promotes fresh FoVer agreements."""

    checkpoint_path = tmp_path / "verify" / "dvi_v2_secl_combined_checkpoint.pt"
    _write_checkpoint(checkpoint_path)

    artifact = mod.build_artifact(
        exp1388_artifact=_exp1388(),
        exp1393_artifact=_exp1393(improvement=0.0),
        exp1394_artifact=_exp1394(checkpoint_path),
        fover_rows=_fover_rows(new_incorrect=60),
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["path_used"] == mod.PATH_DVI_V2_ONLY
    assert artifact["dvi_v2_checkpoint_active"] is True
    assert artifact["replay_cases_used"] == 282
    assert artifact["fresh_verified_sample_count"] == 60
    assert artifact["grpo_v8_cases_integrated"] == 0
    assert artifact["self_learning_delta_overall"] == 1
    assert artifact["headline_result_allowed"] is True
    assert artifact["memory_updates"]["promoted"][0] == "dvi_v2:fover:new_0"


def test_scenario_learn_1395_positive_grpo_gate_integrates_verified_cases(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1395: positive GRPO v8 improvement adds verified rows."""

    checkpoint_path = tmp_path / "verify" / "dvi_v2_secl_combined_checkpoint.pt"
    _write_checkpoint(checkpoint_path)

    artifact = mod.build_artifact(
        exp1388_artifact=_exp1388(),
        exp1393_artifact=_exp1393(improvement=1.0),
        exp1394_artifact=_exp1394(checkpoint_path),
        fover_rows=_fover_rows(new_incorrect=60),
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["path_used"] == mod.PATH_DVI_V2_GRPO
    assert artifact["fresh_verified_sample_count"] == 62
    assert artifact["grpo_v8_cases_integrated"] == 2
    assert artifact["self_learning_delta_overall"] == 3
    assert artifact["headline_result_allowed"] is True


def test_req_learn_1395_inactive_checkpoint_blocks_headline(tmp_path: Path) -> None:
    """REQ-LEARN-1395-2/7: missing DVI v2 checkpoint prevents headline claims."""

    artifact = mod.build_artifact(
        exp1388_artifact=_exp1388(),
        exp1393_artifact=_exp1393(improvement=0.0),
        exp1394_artifact=_exp1394(tmp_path / "missing.pt"),
        fover_rows=_fover_rows(new_incorrect=60),
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["dvi_v2_checkpoint_active"] is False
    assert artifact["fresh_verified_sample_count"] == 0
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == "fr11_self_learning_v5_blocked_dvi_v2_checkpoint_inactive"


def test_req_learn_1395_run_loads_sources_and_writes_complete_artifact(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1395-1/6: runner writes bootstrap then terminal artifact."""

    results = tmp_path / "results"
    results.mkdir()
    checkpoint_path = tmp_path / "verify" / "dvi_v2_secl_combined_checkpoint.pt"
    _write_checkpoint(checkpoint_path)
    exp1388_path = results / mod.EXP1388_FILE
    exp1393_path = results / mod.EXP1393_FILE
    exp1394_path = results / mod.EXP1394_FILE
    fover_path = tmp_path / "fover_corpus.jsonl"
    out_path = results / mod.OUTPUT_FILE
    _write_json(exp1388_path, _exp1388())
    _write_json(exp1393_path, _exp1393(improvement=0.0))
    _write_json(exp1394_path, _exp1394(checkpoint_path))
    _write_jsonl(fover_path, _fover_rows(new_incorrect=60))

    artifact = mod.run(
        exp1388_path=exp1388_path,
        exp1393_path=exp1393_path,
        exp1394_path=exp1394_path,
        fover_path=fover_path,
        out_path=out_path,
        project_root=tmp_path,
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["status"] == "complete"
    assert artifact["fresh_verified_sample_count"] == 60
    assert artifact["headline_result_allowed"] is True
