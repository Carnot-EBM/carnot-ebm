"""Tests for Exp 1430 PRM-guided repair selector artifact.

Spec: REQ-VERIFY-1430, SCENARIO-VERIFY-1430
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.reporting import prm_guided_repair_selector as mod
from carnot.reporting import process_reward_model_v1_fover_1508 as prmv1


def _candidate_result(index: int, *, accepted: bool, final_state: str) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "accepted": accepted,
        "energy": 0.5 if accepted else 9.0,
        "validation_result": {
            "semantic_result": "SAT" if accepted else "REPAIR_HINT",
            "constraint_passed": accepted,
        },
        "candidate": {
            "draft_certificate": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT: repair step.",
            "draft_state": "REPAIR_HINT",
            "final_certificate": f"<CARNOT_CERT_STATE:{final_state}>\n{final_state}",
            "final_state": final_state,
            "repair_action_type": "STEP_REWRITE",
            "repair_rationale": "Repair the localized reasoning step.",
            "repair_target": "localized FoVer reasoning step",
            "validator_metadata": {"prototype_accept": accepted},
        },
    }


def _exp1429_with_pool() -> dict[str, Any]:
    return {
        "status": "complete",
        "repair_success_rate_best_of_n": 1.0,
        "executor_runtime_mode": "unit_test_candidate_pool",
        "candidate_search_results": [
            {
                "case_id": "case_1",
                "best_of_n_success": True,
                "candidate_results": [
                    _candidate_result(0, accepted=False, final_state="REPAIR_HINT"),
                    _candidate_result(1, accepted=True, final_state="SAT"),
                ],
            }
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_checkpoint(path: Path) -> None:
    weights = np.zeros(prmv1.FEATURE_DIM, dtype=np.float32)
    weights[prmv1.HASH_FEATURES + 4] = 8.0
    weights[prmv1.HASH_FEATURES + 5] = -8.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            weights=weights,
            bias=np.asarray([0.0], dtype=np.float32),
            threshold=np.asarray([0.5], dtype=np.float32),
            feature_dim=np.asarray([prmv1.FEATURE_DIM], dtype=np.int32),
        )


def test_req1430_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-VERIFY-1430: bootstrap artifact is written before candidate loading."""

    output = tmp_path / "exp1430.json"

    artifact = mod.write_in_progress_artifact(output, project_root=tmp_path)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["prm_guided_selection_ready"] is False
    assert artifact["honest_verdict"] == "in_progress"


def test_req1430_run_blocks_without_exp1429_candidate_pool(tmp_path: Path) -> None:
    """REQ-VERIFY-1430: missing candidate pool produces a blocked artifact."""

    exp1429 = tmp_path / "exp1429.json"
    prmv1_artifact = tmp_path / "prmv1.json"
    output = tmp_path / "exp1430.json"
    _write_json(exp1429, {"status": "complete", "candidate_search_results": []})
    writes: list[dict[str, Any]] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1429_path=exp1429,
        prmv1_artifact_path=prmv1_artifact,
        output_path=output,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "blocked"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["prm_guided_selection_ready"] is False
    assert artifact["cases_evaluated"] == 0
    assert artifact["selected_repair_success_rate"] is None
    assert artifact["honest_verdict"] == "blocked_exp1429_candidate_pool_unavailable"


def test_scenario1430_run_uses_prmv1_checkpoint_and_writes_complete_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1430: runner scores candidates with PRM v1 checkpoint."""

    exp1429 = tmp_path / "exp1429.json"
    prmv1_artifact = tmp_path / "prmv1.json"
    checkpoint = tmp_path / "models" / "prmv1_checkpoint.pt"
    output = tmp_path / "exp1430.json"
    _write_json(exp1429, _exp1429_with_pool())
    _write_checkpoint(checkpoint)
    _write_json(
        prmv1_artifact,
        {
            "status": "complete",
            "prmv1_trained": True,
            "prmv1_auroc": 0.832874,
            "checkpoint_path": str(checkpoint),
        },
    )

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1429_path=exp1429,
        prmv1_artifact_path=prmv1_artifact,
        output_path=output,
        tests_run=[".venv/bin/pytest tests/python -q"],
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["prm_guided_selection_ready"] is True
    assert artifact["cases_evaluated"] == 1
    assert artifact["selector_auroc"] == pytest.approx(1.0)
    assert artifact["selected_repair_success_rate"] == pytest.approx(1.0)
    assert artifact["selection_improvement_pp"] == pytest.approx(0.0)
    assert artifact["prmv1_artifact_used"] is True
    assert artifact["selector_scoring_mode"] == "prmv1_checkpoint"
    assert artifact["case_selections"][0]["selected_candidate_index"] == 1


def test_req1430_proxy_fallback_is_recorded_as_non_headline(tmp_path: Path) -> None:
    """REQ-VERIFY-1430: deterministic proxy fallback is explicit and non-headline."""

    exp1429 = tmp_path / "exp1429.json"
    output = tmp_path / "exp1430.json"
    _write_json(exp1429, _exp1429_with_pool())

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1429_path=exp1429,
        prmv1_artifact_path=tmp_path / "missing_prmv1.json",
        output_path=output,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["prmv1_artifact_used"] is False
    assert artifact["selector_scoring_mode"] == "deterministic_proxy_non_headline"
    assert "proxy_non_headline" in artifact["honest_verdict"]


def test_req1430_prmv1_loader_failures_fall_back_to_proxy(tmp_path: Path) -> None:
    """REQ-VERIFY-1430: unusable PRM artifacts do not become headline selectors."""

    bad_json = tmp_path / "bad_json.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact must be a JSON object"):
        mod.load_json(bad_json)

    incomplete = tmp_path / "incomplete_prmv1.json"
    _write_json(incomplete, {"status": "blocked", "prmv1_trained": False})
    _scorer, metadata = mod._scorer_from_prmv1_or_proxy(incomplete)
    assert metadata["selector_scoring_mode"] == "deterministic_proxy_non_headline"

    missing_checkpoint = tmp_path / "missing_checkpoint_prmv1.json"
    _write_json(
        missing_checkpoint,
        {
            "status": "complete",
            "prmv1_trained": True,
            "checkpoint_path": str(tmp_path / "missing.pt"),
        },
    )
    _scorer, metadata = mod._scorer_from_prmv1_or_proxy(missing_checkpoint)
    assert metadata["prmv1_artifact_used"] is False

    wrong_shape_checkpoint = tmp_path / "wrong_shape.pt"
    with wrong_shape_checkpoint.open("wb") as handle:
        np.savez(
            handle,
            weights=np.zeros(2, dtype=np.float32),
            bias=np.asarray([0.0], dtype=np.float32),
            feature_dim=np.asarray([2], dtype=np.int32),
        )
    wrong_shape = tmp_path / "wrong_shape_prmv1.json"
    _write_json(
        wrong_shape,
        {
            "status": "complete",
            "prmv1_trained": True,
            "checkpoint_path": str(wrong_shape_checkpoint),
        },
    )
    _scorer, metadata = mod._scorer_from_prmv1_or_proxy(wrong_shape)
    assert metadata["deterministic_proxy_used"] is True


def test_req1430_validate_artifact_rejects_bad_shapes(tmp_path: Path) -> None:
    """REQ-VERIFY-1430: artifact schema catches missing and inconsistent fields."""

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({})

    bad_status = dict.fromkeys(mod.REQUIRED_ARTIFACT_FIELDS, None)
    bad_status["status"] = "weird"
    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(bad_status)

    complete = dict.fromkeys(mod.REQUIRED_ARTIFACT_FIELDS, None)
    complete.update(
        {
            "status": "complete",
            "prm_guided_selection_ready": False,
            "cases_evaluated": 1,
            "selector_auroc": 0.5,
            "raw_best_of_n_repair_success_rate": 0.0,
            "selected_repair_success_rate": 0.0,
            "selection_improvement_pp": 0.0,
            "prmv1_artifact_used": False,
            "honest_verdict": "bad",
        }
    )
    with pytest.raises(AssertionError, match="requires prm_guided_selection_ready=true"):
        mod.validate_artifact(complete)

    blocked = dict(complete, status="blocked", prm_guided_selection_ready=True)
    with pytest.raises(AssertionError, match="must not be ready"):
        mod.validate_artifact(blocked)
