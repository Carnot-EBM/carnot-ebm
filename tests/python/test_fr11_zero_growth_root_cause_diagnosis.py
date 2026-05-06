"""Tests for Exp 1446 FR-11 zero-growth root-cause diagnosis.

Spec: REQ-LEARN-1446, SCENARIO-LEARN-1446.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.reporting import fr11_zero_growth_root_cause_diagnosis as mod


def _write_checkpoint(path: Path, *, secl_threshold: float = 0.500001) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            metric=np.zeros(128, dtype=np.float32),
            bias=np.asarray([10.0], dtype=np.float32),
            secl_bin_values=np.full(10, 0.5, dtype=np.float32),
            secl_global_value=np.asarray([0.5], dtype=np.float32),
            secl_n_bins=np.asarray([10], dtype=np.int32),
            dvi_incorrect_threshold=np.asarray([0.72], dtype=np.float32),
            secl_confidence_threshold=np.asarray([secl_threshold], dtype=np.float32),
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _exp1395() -> dict[str, Any]:
    return {
        "status": "complete",
        "fresh_verified_sample_count": 1,
        "memory_updates": {"promoted": ["dvi_v2:fover:baseline"], "demoted": []},
    }


def _exp1432(checkpoint_path: Path, *, deployed: bool = True) -> dict[str, Any]:
    return {
        "status": "complete" if deployed else "blocked",
        "dvi_v3_deployed": deployed,
        "dvi_v3_checkpoint_path": str(checkpoint_path),
        "nonforgetting_rate": 1.0,
    }


def _exp1433() -> dict[str, Any]:
    return {
        "status": "complete",
        "dvi_v3_checkpoint_active": True,
        "v6_candidate_cases_evaluated": 2,
        "v6_new_promoted_count": 0,
        "self_learning_delta_overall": 0,
        "session_memory_updated": False,
        "honest_verdict": "fr11_self_learning_v6_dvi_v3_no_positive_growth_non_headline",
    }


def _fover_rows() -> list[dict[str, str]]:
    return [
        {"question_id": "baseline", "step_text": "already promoted", "label": "incorrect"},
        {"question_id": "new_bad", "step_text": "fresh incorrect", "label": "incorrect"},
        {"question_id": "new_good", "step_text": "fresh correct", "label": "correct"},
    ]


def _fover_correct_only() -> list[dict[str, str]]:
    return [{"question_id": "new_good", "step_text": "fresh correct", "label": "correct"}]


def test_req_learn_1446_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1446-1: bootstrap artifact is persisted before diagnosis."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["fr11_zero_growth_root_cause_identified"] is False
    assert written["candidate_supply_count"] == 0
    assert written["candidate_rejection_reason_counts"]["no_candidates"] == 0
    assert written["exact_rerun_forbidden"] is False


def test_scenario_learn_1446_counts_losses_and_recommends_v7_policy(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1446: active DVI v3 with zero growth forbids exact rerun."""

    checkpoint_path = tmp_path / "verify" / "dvi_v3.pt"
    _write_checkpoint(checkpoint_path)

    artifact = mod.build_artifact(
        exp1395_artifact=_exp1395(),
        exp1432_artifact=_exp1432(checkpoint_path),
        exp1433_artifact=_exp1433(),
        fover_rows=_fover_rows(),
        project_root="/repo",
        commands_run=["unit command"],
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["fr11_zero_growth_root_cause_identified"] is True
    assert artifact["candidate_supply_count"] == 2
    assert artifact["candidate_rejection_reason_counts"] == {
        "no_candidates": 0,
        "verifier_rejection": 1,
        "dvi_threshold": 1,
        "novelty_threshold": 1,
        "duplicate_memory": 0,
        "persistence_blocker": 0,
    }
    assert artifact["promotion_thresholds"]["exp1433_secl_confidence_threshold"] == pytest.approx(
        0.500001
    )
    assert artifact["recommended_v7_policy"]["fresh_secl_confidence_threshold"] == 0.5
    assert artifact["recommended_v7_policy"]["expected_promotions_under_v7_policy"] == 1
    assert artifact["memory_update_policy"]["exp1433_promoted_memory_count"] == 0
    assert artifact["exact_rerun_forbidden"] is True
    assert "asymmetric_fresh_threshold" in artifact["honest_verdict"]


def test_req_learn_1446_run_loads_sources_and_writes_complete_artifact(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1446-4: run writes all required terminal artifact fields."""

    results = tmp_path / "results"
    checkpoint_path = tmp_path / "verify" / "dvi_v3.pt"
    fover_path = tmp_path / "fover.jsonl"
    out_path = results / mod.OUTPUT_FILE
    _write_checkpoint(checkpoint_path)
    _write_json(results / mod.EXP1395_FILE, _exp1395())
    _write_json(results / mod.EXP1432_FILE, _exp1432(checkpoint_path))
    _write_json(results / mod.EXP1433_FILE, _exp1433())
    _write_jsonl(fover_path, _fover_rows())

    artifact = mod.run(
        exp1395_path=results / mod.EXP1395_FILE,
        exp1432_path=results / mod.EXP1432_FILE,
        exp1433_path=results / mod.EXP1433_FILE,
        fover_path=fover_path,
        out_path=out_path,
        project_root=tmp_path,
        commands_run=["pytest test"],
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["commands_run"] == ["pytest test"]


def test_req_learn_1446_validation_rejects_incomplete_terminal_artifact() -> None:
    """REQ-LEARN-1446-4: required terminal fields are enforced."""

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({"status": "complete"})

    artifact = mod.write_in_progress_artifact(Path("/tmp/nonpersistent_exp1446.json"))
    mod.validate_artifact(artifact)
    artifact["status"] = "complete"
    artifact["fr11_zero_growth_root_cause_identified"] = True
    artifact["exact_rerun_forbidden"] = False
    with pytest.raises(AssertionError, match="exact_rerun_forbidden"):
        mod.validate_artifact(artifact)

    missing_count = mod.write_in_progress_artifact(Path("/tmp/nonpersistent_exp1446_counts.json"))
    missing_count["status"] = "complete"
    del missing_count["candidate_rejection_reason_counts"]["dvi_threshold"]
    with pytest.raises(AssertionError, match="missing rejection count"):
        mod.validate_artifact(missing_count)

    unchanged_policy = mod.write_in_progress_artifact(
        Path("/tmp/nonpersistent_exp1446_policy.json")
    )
    unchanged_policy["status"] = "complete"
    unchanged_policy["exact_rerun_forbidden"] = True
    with pytest.raises(AssertionError, match="changed v7 policy"):
        mod.validate_artifact(unchanged_policy)


def test_req_learn_1446_handles_no_candidate_and_blocked_dvi_paths(tmp_path: Path) -> None:
    """REQ-LEARN-1446-2: no-candidate and persistence blockers are counted."""

    checkpoint_path = tmp_path / "verify" / "dvi_v3.pt"
    _write_checkpoint(checkpoint_path)
    no_candidate_artifact = mod.build_artifact(
        exp1395_artifact=_exp1395(),
        exp1432_artifact=_exp1432(checkpoint_path),
        exp1433_artifact={**_exp1433(), "dvi_v3_checkpoint_active": True},
        fover_rows=[
            {"question_id": "baseline", "step_text": "already promoted", "label": "incorrect"},
            {"step_text": "missing id", "label": "incorrect"},
            {"question_id": "no_text", "label": "incorrect"},
        ],
        project_root="/repo",
    )

    assert no_candidate_artifact["candidate_supply_count"] == 0
    assert no_candidate_artifact["candidate_rejection_reason_counts"]["no_candidates"] == 1
    assert no_candidate_artifact["candidate_generation_counts"]["missing_case_id"] == 1
    assert no_candidate_artifact["candidate_generation_counts"]["unusable_candidate"] == 1
    assert (
        no_candidate_artifact["honest_verdict"] == "fr11_v6_zero_growth_root_cause_not_identified"
    )

    blocked_artifact = mod.build_artifact(
        exp1395_artifact=_exp1395(),
        exp1432_artifact=_exp1432(checkpoint_path, deployed=False),
        exp1433_artifact={**_exp1433(), "dvi_v3_checkpoint_active": False},
        fover_rows=_fover_rows(),
        project_root="/repo",
    )

    assert blocked_artifact["candidate_rejection_reason_counts"]["persistence_blocker"] == 1
    assert blocked_artifact["promotion_thresholds"]["exp1433_dvi_incorrect_threshold"] is None
    assert blocked_artifact["recommended_v7_policy"]["dvi_incorrect_threshold"] is None
    assert (
        blocked_artifact["honest_verdict"]
        == "fr11_v6_zero_growth_diagnosis_blocked_dvi_v3_inactive"
    )


def test_req_learn_1446_reports_existing_promotable_and_zero_expected_policy(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1446-3: diagnosis distinguishes existing and changed promotions."""

    promotable_checkpoint = tmp_path / "verify" / "promotable.pt"
    _write_checkpoint(promotable_checkpoint, secl_threshold=0.5)
    promotable = mod.build_artifact(
        exp1395_artifact=_exp1395(),
        exp1432_artifact=_exp1432(promotable_checkpoint),
        exp1433_artifact=_exp1433(),
        fover_rows=_fover_rows(),
        project_root="/repo",
    )
    assert promotable["candidate_rejection_detail"]["exp1433_promotable"] == 1

    mismatch_only_checkpoint = tmp_path / "verify" / "mismatch_only.pt"
    _write_checkpoint(mismatch_only_checkpoint)
    mismatch_only = mod.build_artifact(
        exp1395_artifact={"memory_updates": {"promoted": []}, "fresh_verified_sample_count": 0},
        exp1432_artifact=_exp1432(mismatch_only_checkpoint),
        exp1433_artifact=_exp1433(),
        fover_rows=_fover_correct_only(),
        project_root="/repo",
    )
    assert mismatch_only["recommended_v7_policy"]["expected_promotions_under_v7_policy"] == 0
    assert mismatch_only["honest_verdict"] == (
        "fr11_v6_zero_growth_root_cause_identified_policy_change_required"
    )
