"""Tests for Exp 1433 FR-11 self-learning v6 gated on deployed DVI v3.

Spec: REQ-LEARN-1433, SCENARIO-LEARN-1433, SCENARIO-LEARN-1434.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.reporting import fr11_self_learning_v6_dvi_v3_gated as mod


def _write_checkpoint(path: Path, *, bias: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            metric=np.zeros(128, dtype=np.float32),
            bias=np.asarray([bias], dtype=np.float32),
            secl_bin_values=np.ones(10, dtype=np.float32),
            secl_global_value=np.asarray([1.0], dtype=np.float32),
            secl_n_bins=np.asarray([10], dtype=np.int32),
            dvi_incorrect_threshold=np.asarray([0.72], dtype=np.float32),
            secl_confidence_threshold=np.asarray([0.5], dtype=np.float32),
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _exp1395(promoted_ids: list[str], *, baseline_count: int | None = None) -> dict[str, Any]:
    return {
        "status": "complete",
        "fresh_verified_sample_count": baseline_count
        if baseline_count is not None
        else len(promoted_ids),
        "memory_updates": {
            "promoted": [f"dvi_v2:fover:{case_id}" for case_id in promoted_ids],
            "demoted": [],
        },
    }


def _exp1432(checkpoint_path: Path, *, deployed: bool = True) -> dict[str, Any]:
    return {
        "status": "complete" if deployed else "blocked",
        "dvi_v3_deployed": deployed,
        "dvi_v3_checkpoint_path": str(checkpoint_path),
        "nonforgetting_rate": 1.0,
        "honest_verdict": "dvi_v3_deployed_replay_heldout_threshold_calibrated",
    }


def _fover_row(case_id: str, label: str) -> dict[str, str]:
    return {
        "question_id": case_id,
        "step_text": f"{label} FoVer trace for {case_id}",
        "label": label,
        "source": "unit_fover",
    }


def test_req_learn_1433_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1433-1: bootstrap output exists before source loading."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["dvi_v3_artifact_used"] is None
    assert written["fresh_verified_sample_count"] is None
    assert written["baseline_fresh_verified_sample_count"] == mod.EXP1395_BASELINE_COUNT
    assert written["headline_result_allowed"] is False
    assert written["honest_verdict"] == "in_progress"


def test_scenario_learn_1434_blocks_without_dvi_v3_deployment(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1434: missing DVI v3 deployment blocks v6."""

    artifact = mod.build_artifact(
        exp1395_artifact=_exp1395(["baseline"], baseline_count=1),
        exp1432_artifact=_exp1432(tmp_path / "missing.pt", deployed=False),
        fover_rows=[_fover_row("candidate", "incorrect")],
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert (
        artifact["dvi_v3_artifact_used"]
        == "results/experiment_1432_dvi_v3_nonforgetting_replay_balanced.json"
    )
    assert artifact["fresh_verified_sample_count"] == 1
    assert artifact["self_learning_delta_overall"] == 0
    assert artifact["session_memory_updated"] is False
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == "fr11_self_learning_v6_blocked_exp1432_dvi_v3_not_deployed"


def test_scenario_learn_1433_dvi_v3_promotes_new_heldout_case(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1433: deployed DVI v3 can add held-out fresh cases."""

    results = tmp_path / "results"
    results.mkdir()
    checkpoint_path = tmp_path / "verify" / "dvi_v3_checkpoint.pt"
    exp1395_path = results / mod.EXP1395_FILE
    exp1432_path = results / mod.EXP1432_FILE
    fover_path = tmp_path / "fover.jsonl"
    out_path = results / mod.OUTPUT_FILE
    _write_checkpoint(checkpoint_path, bias=10.0)
    _write_json(exp1395_path, _exp1395(["baseline_correct", "baseline_bad"]))
    _write_json(exp1432_path, _exp1432(checkpoint_path))
    _write_jsonl(
        fover_path,
        [
            _fover_row("baseline_correct", "correct"),
            _fover_row("baseline_bad", "incorrect"),
            _fover_row("new_bad", "incorrect"),
            _fover_row("new_good", "correct"),
        ],
    )

    artifact = mod.run(
        exp1395_path=exp1395_path,
        exp1432_path=exp1432_path,
        fover_path=fover_path,
        out_path=out_path,
        project_root=tmp_path,
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["fresh_verified_sample_count"] == 3
    assert artifact["baseline_fresh_verified_sample_count"] == 2
    assert artifact["self_learning_delta_overall"] == 1
    assert artifact["nonforgetting_rate"] == 1.0
    assert artifact["session_memory_updated"] is True
    assert artifact["headline_result_allowed"] is True
    assert artifact["memory_updates"]["promoted"] == ["dvi_v3:fover:new_bad"]


def test_req_learn_1433_no_growth_disables_headline_and_memory_update(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1433-6: headline claims require positive growth."""

    checkpoint_path = tmp_path / "dvi_v3_checkpoint.pt"
    _write_checkpoint(checkpoint_path, bias=10.0)

    artifact = mod.build_artifact(
        exp1395_artifact=_exp1395(["baseline"], baseline_count=1),
        exp1432_artifact=_exp1432(checkpoint_path),
        fover_rows=[_fover_row("baseline", "incorrect"), _fover_row("new_good", "correct")],
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["fresh_verified_sample_count"] == 1
    assert artifact["self_learning_delta_overall"] == 0
    assert artifact["session_memory_updated"] is False
    assert artifact["headline_result_allowed"] is False
    assert (
        artifact["honest_verdict"] == "fr11_self_learning_v6_dvi_v3_no_positive_growth_non_headline"
    )


def test_req_learn_1433_validation_rejects_bad_headline_invariants() -> None:
    """REQ-LEARN-1433-5/6: required fields and headline invariants are enforced."""

    mod.validate_artifact(
        {
            "status": "in_progress",
            "dvi_v3_artifact_used": None,
            "fresh_verified_sample_count": None,
            "baseline_fresh_verified_sample_count": mod.EXP1395_BASELINE_COUNT,
            "self_learning_delta_overall": None,
            "nonforgetting_rate": None,
            "session_memory_updated": None,
            "headline_result_allowed": False,
            "honest_verdict": "in_progress",
        }
    )

    artifact = {
        "status": "complete",
        "dvi_v3_artifact_used": "results/experiment_1432_dvi_v3_nonforgetting_replay_balanced.json",
        "fresh_verified_sample_count": 1,
        "baseline_fresh_verified_sample_count": 1,
        "self_learning_delta_overall": 0,
        "nonforgetting_rate": 1.0,
        "session_memory_updated": False,
        "headline_result_allowed": True,
        "honest_verdict": "bad",
    }

    with pytest.raises(AssertionError, match="positive self-learning delta"):
        mod.validate_artifact(artifact)

    mismatched_delta = dict(artifact)
    mismatched_delta["headline_result_allowed"] = False
    mismatched_delta["self_learning_delta_overall"] = 99
    with pytest.raises(AssertionError, match="fresh minus baseline"):
        mod.validate_artifact(mismatched_delta)

    bad_nonforgetting = dict(artifact)
    bad_nonforgetting["fresh_verified_sample_count"] = 2
    bad_nonforgetting["self_learning_delta_overall"] = 1
    bad_nonforgetting["nonforgetting_rate"] = 0.5
    with pytest.raises(AssertionError, match="preserved nonforgetting"):
        mod.validate_artifact(bad_nonforgetting)

    stale_memory_flag = dict(artifact)
    stale_memory_flag["headline_result_allowed"] = False
    stale_memory_flag["session_memory_updated"] = True
    with pytest.raises(AssertionError, match="session_memory_updated"):
        mod.validate_artifact(stale_memory_flag)

    missing = dict(artifact)
    del missing["honest_verdict"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)
