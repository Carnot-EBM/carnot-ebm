"""Tests for Exp 4026 verifier-vs-judge efficiency corrigendum.

Spec refs: REQ-PHASE4-034, SCENARIO-PHASE4-034.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from carnot.agentic.arc_verifier_vs_judge_efficiency_corrigendum import (
    REQUIRED_ARTIFACT_FIELDS,
    artifact_schema_errors,
    build_corrected_artifact,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_4026_verifier_vs_judge_efficiency as exp  # noqa: E402


def _exp4013_payload() -> dict[str, object]:
    return {
        "experiment": "experiment_4013_verifier_vs_judge_efficiency",
        "honest_verdict": "success: verifier_parity_at_120x_cheaper_than_judge",
        "flagged_adversarial": True,
        "selection_accuracy_parity": True,
        "verifier_gold_rate": 0.5,
        "judge_gold_rate": 0.5,
        "selection_agreement_rate": 0.5,
        "accuracy_gap": 0.0,
        "cost_verifier_seconds": 0.1,
        "cost_judge_seconds": 12.0,
        "cost_ratio_judge_over_verifier": 120.0,
        "token_ratio_judge_over_verifier": 120.0,
        "judge_tokens_total": 0,
        "verifier_tokens": 0,
        "n_tasks": 2,
        "n_judge_calls": 1,
        "random_seed": 42,
        "candidate_set_summary": [
            {
                "task_key": "arc1:0:T1",
                "n_candidates": 2,
                "verifier_choice_id": "C0",
                "judge_choice_id": "C0",
                "verifier_choice_gold": True,
                "judge_choice_gold": True,
            },
            {
                "task_key": "arc2:0:T2",
                "n_candidates": 3,
                "verifier_choice_id": "C0",
                "judge_choice_id": "C1",
                "verifier_choice_gold": False,
                "judge_choice_gold": False,
            },
        ],
    }


def test_req_phase4_034_spec_declares_exp4026_contract() -> None:
    """REQ-PHASE4-034: OpenSpec declares Exp 4026 and its required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-034" in spec
    assert "SCENARIO-PHASE4-034" in spec
    assert "experiment_4026_verifier_vs_judge_efficiency.json" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_phase4_034_corrects_tautological_cost_axis() -> None:
    """SCENARIO-PHASE4-034: wall-clock ratio is independent from token ratio."""

    artifact = build_corrected_artifact(exp4013=_exp4013_payload(), duration_s=0.25)

    assert artifact["accuracy_parity"] is True
    assert artifact["wallclock_seconds_ratio_judge_over_verifier"] == 120.0
    assert artifact["token_ratio_judge_over_verifier"] == 225.0
    assert (
        artifact["wallclock_seconds_ratio_judge_over_verifier"]
        != artifact["token_ratio_judge_over_verifier"]
    )
    assert artifact["honest_verdict"] == "success: verifier_parity_wallclock_120x_judge_over_verifier"
    assert artifact["prior_failure_addressed"]["experiment_id"] == "exp4013-verifier-vs-judge-efficiency"
    assert artifact["flagged_adversarial"] is False
    assert artifact_schema_errors(artifact) == []


def test_scenario_phase4_034_blocks_without_cached_candidate_summary() -> None:
    """SCENARIO-PHASE4-034: missing cached Exp 4013 selections block honestly."""

    artifact = build_corrected_artifact(exp4013=None, duration_s=0.1)
    empty_summary = build_corrected_artifact(exp4013={}, duration_s=0.1)

    assert artifact["honest_verdict"] == "blocked_exp4013_candidate_summary_missing"
    assert artifact["accuracy_parity"] is False
    assert artifact["wallclock_seconds_ratio_judge_over_verifier"] == 0.0
    assert artifact["token_ratio_judge_over_verifier"] == 0.0
    assert artifact_schema_errors(artifact) == []
    assert empty_summary["honest_verdict"] == "blocked_exp4013_candidate_summary_missing"


def test_scenario_phase4_034_uses_reported_tokens_when_available() -> None:
    """SCENARIO-PHASE4-034: token ratio can come from reported token counts."""

    payload = _exp4013_payload()
    payload["judge_tokens_total"] = 1000
    payload["verifier_tokens"] = 5

    artifact = build_corrected_artifact(exp4013=payload, duration_s=0.2)

    assert artifact["token_ratio_source"] == "reported_exp4013_token_usage"
    assert artifact["token_ratio_judge_over_verifier"] == 200.0


def test_scenario_phase4_034_nonparity_gets_complete_verdict() -> None:
    """SCENARIO-PHASE4-034: cheaper verifier is not promoted when parity fails."""

    payload = _exp4013_payload()
    payload["candidate_set_summary"] = [
        {
            "task_key": f"T{i}",
            "n_candidates": 2,
            "verifier_choice_id": "C0",
            "judge_choice_id": "C1",
            "verifier_choice_gold": False,
            "judge_choice_gold": True,
        }
        for i in range(20)
    ]

    artifact = build_corrected_artifact(exp4013=payload, duration_s=0.2)

    assert artifact["accuracy_parity"] is False
    assert artifact["honest_verdict"] == "complete: verifier_wallclock_120x_but_accuracy_gap_1"


def test_req_phase4_034_schema_rejects_bad_required_fields() -> None:
    """REQ-PHASE4-034: required artifact fields stay bare JSON scalars."""

    artifact = build_corrected_artifact(exp4013=_exp4013_payload(), duration_s=0.25)
    bad = dict(artifact)
    bad["honest_verdict"] = "finished"
    bad["accuracy_parity"] = 1
    bad["wallclock_seconds_ratio_judge_over_verifier"] = "120"
    bad["token_ratio_judge_over_verifier"] = 225
    bad["inference_substrate"] = None

    errors = artifact_schema_errors(bad)
    missing = artifact_schema_errors({})

    assert any("honest_verdict" in error for error in errors)
    assert any("accuracy_parity" in error for error in errors)
    assert any("wallclock_seconds_ratio_judge_over_verifier" in error for error in errors)
    assert any("token_ratio_judge_over_verifier" in error for error in errors)
    assert any("inference_substrate" in error for error in errors)
    assert any("missing required field honest_verdict" in error for error in missing)


def test_req_phase4_034_build_rejects_identical_positive_ratios() -> None:
    """REQ-PHASE4-034: Exp 4026 cannot repeat the Exp 4013 ratio tautology."""

    payload = _exp4013_payload()
    payload["judge_tokens_total"] = 120
    payload["verifier_tokens"] = 1

    try:
        build_corrected_artifact(exp4013=payload, duration_s=0.2)
    except ValueError as exc:
        assert "non-identical measurements" in str(exc)
    else:  # pragma: no cover - this branch would mean the regression guard failed.
        raise AssertionError("expected identical-ratio guard to reject the artifact")


def test_runner_writes_exp4026_result_json(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-034: runner writes the stable exp4026 deliverable."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "experiment_4013_verifier_vs_judge_efficiency.json").write_text(
        json.dumps(_exp4013_payload()) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True)

    written = results_dir / "experiment_4026_verifier_vs_judge_efficiency.json"
    assert artifact["honest_verdict"] == "success: verifier_parity_wallclock_120x_judge_over_verifier"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8")) == artifact
