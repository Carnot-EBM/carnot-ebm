"""Tests for the Exp 1448 PRM v3 online process-reward agent artifact.

Spec: REQ-VERIFY-1448, SCENARIO-VERIFY-1448.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.reporting import prm_v3_online_process_reward_agent as mod
from carnot.reporting import process_reward_model_v1_fover_1508 as prmv1


def _candidate_result(
    index: int,
    *,
    accepted: bool,
    final_state: str,
    target: str,
    false_acceptance: bool = False,
) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "accepted": accepted,
        "validation_result": {
            "semantic_result": final_state,
            "constraint_passed": accepted,
            "false_acceptance": false_acceptance,
        },
        "candidate": {
            "draft_certificate": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT: repair step.",
            "draft_state": "REPAIR_HINT",
            "final_certificate": f"<CARNOT_CERT_STATE:{final_state}>\n{final_state}",
            "final_state": final_state,
            "repair_action_type": "STEP_REWRITE",
            "repair_rationale": "Repair the localized reasoning step.",
            "repair_target": target,
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
                "best_candidate_index": 2,
                "candidate_results": [
                    _candidate_result(
                        0,
                        accepted=False,
                        final_state="REPAIR_HINT",
                        target="reject_first_candidate localized FoVer reasoning step",
                    ),
                    _candidate_result(
                        1,
                        accepted=True,
                        final_state="SAT",
                        target="accept_high_energy localized FoVer reasoning step",
                        false_acceptance=True,
                    ),
                    _candidate_result(
                        2,
                        accepted=True,
                        final_state="SAT",
                        target="accept_low_energy localized FoVer reasoning step",
                    ),
                ],
            }
        ],
    }


def _exp1430_selection() -> dict[str, Any]:
    return {
        "status": "complete",
        "prm_guided_selection_ready": True,
        "selected_repair_success_rate": 1.0,
        "case_selections": [
            {
                "case_id": "case_1",
                "selected_candidate_index": 1,
                "selected_accepted": True,
            }
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            weights=np.zeros(prmv1.FEATURE_DIM, dtype=np.float32),
            bias=np.asarray([0.0], dtype=np.float32),
            threshold=np.asarray([0.5], dtype=np.float32),
            feature_dim=np.asarray([prmv1.FEATURE_DIM], dtype=np.int32),
        )


def _complete_prmv2_artifact(checkpoint: Path) -> dict[str, Any]:
    return {
        "status": "complete",
        "prmv2_trained": True,
        "headline_label_coverage_ready": True,
        "training_traces_used": 1508,
        "step_labels_available": 2302,
        "prmv2_auroc": 0.851789,
        "checkpoint_path": str(checkpoint),
    }


def test_req1448_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-VERIFY-1448: bootstrap artifact is written before source loading."""

    output = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(output, project_root=tmp_path)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["pra_selector_ready"] is False
    assert artifact["prm_v2_labels_used"] is False
    assert artifact["honest_verdict"] == "in_progress"


def test_req1448_run_blocks_without_prmv2_labels(tmp_path: Path) -> None:
    """REQ-VERIFY-1448: missing PRM v2 labels produce a blocked artifact."""

    exp1429 = tmp_path / "exp1429.json"
    exp1430 = tmp_path / "exp1430.json"
    exp1434 = tmp_path / "exp1434.json"
    output = tmp_path / mod.OUTPUT_FILE
    _write_json(exp1429, _exp1429_with_pool())
    _write_json(exp1430, _exp1430_selection())
    _write_json(exp1434, {"status": "blocked", "prmv2_trained": False})
    writes: list[str] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1429_path=exp1429,
        exp1430_path=exp1430,
        exp1434_path=exp1434,
        output_path=output,
        write_observer=lambda _path, payload: writes.append(str(payload["status"])),
    )

    assert writes == ["in_progress", "blocked"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["pra_selector_ready"] is False
    assert artifact["prm_v2_labels_used"] is False
    assert artifact["honest_verdict"] == "blocked_prm_v2_labels_or_checkpoint_unavailable"


def test_req1448_run_blocks_without_candidate_pool_or_prmv1(tmp_path: Path) -> None:
    """REQ-VERIFY-1448: PRM v1 comparison and candidate pool are required."""

    exp1429 = tmp_path / "exp1429.json"
    exp1430 = tmp_path / "exp1430.json"
    exp1434 = tmp_path / "exp1434.json"
    checkpoint = tmp_path / "models" / "prmv2_checkpoint.pt"
    output = tmp_path / mod.OUTPUT_FILE
    _write_json(exp1429, {"status": "complete", "candidate_search_results": []})
    _write_json(exp1430, {"status": "complete", "prm_guided_selection_ready": False})
    _write_checkpoint(checkpoint)
    _write_json(exp1434, _complete_prmv2_artifact(checkpoint))

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1429_path=exp1429,
        exp1430_path=exp1430,
        exp1434_path=exp1434,
        output_path=output,
    )

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_candidate_pool_or_prm_v1_comparison_unavailable"


def test_scenario1448_run_uses_prmv2_checkpoint_and_writes_complete_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1448: runner compares PRM v3 with raw and PRM v1."""

    exp1429 = tmp_path / "exp1429.json"
    exp1430 = tmp_path / "exp1430.json"
    exp1434 = tmp_path / "exp1434.json"
    checkpoint = tmp_path / "models" / "prmv2_checkpoint.pt"
    output = tmp_path / mod.OUTPUT_FILE
    _write_json(exp1429, _exp1429_with_pool())
    _write_json(exp1430, _exp1430_selection())
    _write_checkpoint(checkpoint)
    _write_json(exp1434, _complete_prmv2_artifact(checkpoint))

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1429_path=exp1429,
        exp1430_path=exp1430,
        exp1434_path=exp1434,
        output_path=output,
        commands_run=[".venv/bin/pytest tests/python -q"],
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["pra_selector_ready"] is True
    assert artifact["prm_v2_labels_used"] is True
    assert artifact["traces_evaluated"] == 3
    assert artifact["step_scores_generated"] == 12
    assert artifact["selection_improvement_pp"] == pytest.approx(0.0)
    assert artifact["false_acceptance_rate_delta"] == pytest.approx(-1.0)
    assert artifact["regression_against_prm_v1"] is False
    assert artifact["case_selections"][0]["selected_candidate_index"] == 2
    assert "no_headline_improvement" in artifact["honest_verdict"]


def test_req1448_validate_artifact_and_loader_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-1448: schema validation and loader failures are explicit."""

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
            "pra_selector_ready": False,
            "prm_v2_labels_used": True,
            "traces_evaluated": 1,
            "step_scores_generated": 1,
            "selection_improvement_pp": 0.0,
            "false_acceptance_rate_delta": 0.0,
            "regression_against_prm_v1": False,
            "commands_run": [],
            "honest_verdict": "bad",
        }
    )
    with pytest.raises(AssertionError, match="requires pra_selector_ready=true"):
        mod.validate_artifact(complete)

    complete["pra_selector_ready"] = True
    complete["prm_v2_labels_used"] = False
    with pytest.raises(AssertionError, match="requires prm_v2_labels_used=true"):
        mod.validate_artifact(complete)

    blocked = dict(complete, status="blocked", pra_selector_ready=True)
    with pytest.raises(AssertionError, match="must not be ready"):
        mod.validate_artifact(blocked)

    bad_jsonl = tmp_path / "rows.jsonl"
    bad_jsonl.write_text('\n{"ok": true}\nnot-json\n[]\n', encoding="utf-8")
    assert mod.load_jsonl_rows(bad_jsonl) == [{"ok": True}]

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact must be a JSON object"):
        mod.load_json(malformed)


def test_req1448_prmv2_loader_and_verdict_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-1448: checkpoint loader and verdict branches stay explicit."""

    with pytest.raises(ValueError, match="not a complete trained"):
        mod._scorer_from_prmv2({"status": "blocked", "prmv2_trained": False})
    with pytest.raises(ValueError, match="coverage is not headline-ready"):
        mod._scorer_from_prmv2(
            {"status": "complete", "prmv2_trained": True, "headline_label_coverage_ready": False}
        )
    with pytest.raises(ValueError, match="checkpoint path does not exist"):
        mod._scorer_from_prmv2(
            {
                "status": "complete",
                "prmv2_trained": True,
                "headline_label_coverage_ready": True,
                "checkpoint_path": str(tmp_path / "missing.pt"),
            }
        )

    wrong_shape_checkpoint = tmp_path / "wrong_shape.pt"
    with wrong_shape_checkpoint.open("wb") as handle:
        np.savez(
            handle,
            weights=np.zeros(2, dtype=np.float32),
            bias=np.asarray([0.0], dtype=np.float32),
            feature_dim=np.asarray([2], dtype=np.int32),
        )
    with pytest.raises(ValueError, match="feature_dim"):
        mod._scorer_from_prmv2(
            {
                "status": "complete",
                "prmv2_trained": True,
                "headline_label_coverage_ready": True,
                "checkpoint_path": str(wrong_shape_checkpoint),
            }
        )

    checkpoint = tmp_path / "good.pt"
    _write_checkpoint(checkpoint)
    scorer, metadata = mod._scorer_from_prmv2(
        {
            "status": "complete",
            "prmv2_trained": True,
            "headline_label_coverage_ready": True,
            "checkpoint_path": str(checkpoint),
            "prmv2_auroc": "bad",
        }
    )
    assert 0.0 <= scorer("SAT accept_low_energy") <= 1.0
    assert metadata["prm_v2_reported_auroc"] is None

    exp1429 = {"executor_runtime_mode": "live_local_sota_gguf"}
    assert (
        mod._complete_verdict(
            {
                "regression_against_prm_v1": True,
                "selection_improvement_pp": 1.0,
                "false_acceptance_rate_delta": 0.0,
            },
            exp1429=exp1429,
        )
        == "complete_prmv3_regression_against_prm_v1_no_improvement_claim"
    )
    assert (
        mod._complete_verdict(
            {
                "regression_against_prm_v1": False,
                "selection_improvement_pp": 1.0,
                "false_acceptance_rate_delta": 0.1,
            },
            exp1429=exp1429,
        )
        == "complete_prmv3_selection_improved_but_false_acceptance_worsened"
    )
    assert (
        mod._complete_verdict(
            {
                "regression_against_prm_v1": False,
                "selection_improvement_pp": 1.0,
                "false_acceptance_rate_delta": 0.0,
            },
            exp1429=exp1429,
        )
        == "complete_prmv3_selection_improved_non_regressing"
    )
