"""Tests for Exp 5033 D3 EBRM uncertainty verifier v3.

Spec refs: REQ-VERIFY-5033, SCENARIO-VERIFY-5033.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5033_ebrm_uncertainty_verifier_v3 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _audit_clean(_path: Path) -> dict[str, Any]:
    return {"flag_count": 0, "flags": []}


def _write_d1_artifact(
    root: Path,
    *,
    trained: bool = True,
    predictions: list[str] | None = None,
) -> None:
    _write_json(
        root / mod.D1_BASE_RELATIVE_PATH,
        {
            "experiment": "experiment_5031_lora_ebm_scorer_musr_v3",
            "scorer_trained": trained,
            "train_loss": 0.25 if trained else None,
            "n_pairs": 12 if trained else 0,
            "base_used": "Qwen/Qwen3.5-2B",
            "checkpoint_path": "/tmp/d1/epoch_1" if trained else None,
            "verifier_is_oracle": False,
            "trained_scorer_accuracy": 0.75 if trained else None,
            "genuine_tuned_sc_accuracy": 0.5 if trained else None,
            "model_specs": {"base_model": "Qwen/Qwen3.5-2B", "adapter": "LoRA"},
            "evaluation": {"verifier": {"predictions": predictions or []}},
        },
    )


def _write_checkpoint(root: Path, index: int, *, gold: str, answers: list[str]) -> None:
    _write_json(
        root / mod.MUSR_CHECKPOINT_RELATIVE_DIR / f"q{index:04d}.json",
        {"q": index, "gold": gold, "answers": answers, "energy_pure_answer": answers[0]},
    )


def _fixture_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "q0",
            "gold": "A",
            "d1_prediction": "A",
            "candidates": [
                {"candidate_id": "q0/c0", "answer": "A", "cache_index": 0, "d1_base_reward": 1.0},
                {"candidate_id": "q0/c1", "answer": "B", "cache_index": 1, "d1_base_reward": 0.0},
            ],
        },
        {
            "row_id": "q1",
            "gold": "B",
            "d1_prediction": "B",
            "candidates": [
                {"candidate_id": "q1/c0", "answer": "A", "cache_index": 0, "d1_base_reward": 0.05},
                {"candidate_id": "q1/c1", "answer": "B", "cache_index": 1, "d1_base_reward": 0.9},
            ],
        },
        {
            "row_id": "q2",
            "gold": "A",
            "d1_prediction": "B",
            "candidates": [
                {"candidate_id": "q2/c0", "answer": "B", "cache_index": 0, "d1_base_reward": 0.51},
                {"candidate_id": "q2/c1", "answer": "A", "cache_index": 1, "d1_base_reward": 0.5},
            ],
        },
    ]


def test_req_verify_5033_spec_declares_contract() -> None:
    """REQ-VERIFY-5033: OpenSpec anchors the D1-gated EBRM v3 contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-5033",
        "SCENARIO-VERIFY-5033",
        "experiment_5033_ebrm_uncertainty_verifier_v3.py",
        "results/experiment_5033_ebrm_uncertainty_verifier_v3.json",
        "blocked_d1_base_scorer_not_trained",
        "success_ebrm_beats_sc_musr_",
        "complete_ebrm_no_win_musr_",
        "CoT-entropy",
        "CROP-style conformal abstention",
        "abstention_rate",
        "genuine_tuned_sc_accuracy",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_verify_5033_distribution_and_calibration_are_capped() -> None:
    """REQ-VERIFY-5033: reward distributions include uncertainty and capped CROP calibration."""

    rows = mod.prepare_rows_with_ebrm_distributions(_fixture_rows())
    pairs = mod.conflict_aware_training_rows(rows)
    calibration = mod.build_uncertainty_calibration(
        rows,
        calibration_indices=[0, 1, 2],
        thresholds=[0.0, 0.25, 0.5, 1.0],
    )
    best = mod._best_candidate(rows[0])
    uncertain = mod._best_candidate(rows[2])

    assert best is not None
    assert best["ebrm_reward_distribution"]["mean_reward"] > 0.0
    assert "cot_entropy" in best["uncertainty_head"]
    assert "uarm_heteroscedastic_variance" in best["uncertainty_head"]
    assert "distributional_pessimistic_reward" in best["uncertainty_head"]
    assert uncertain is not None
    assert uncertain["ebrm_uncertainty"] > best["ebrm_uncertainty"]
    assert pairs and all(pair["conflict_aware_filtered"] for pair in pairs)
    assert calibration["selected_threshold"] in [0.0, 0.25, 0.5, 1.0]
    assert calibration["selected_abstention_rate"] <= mod.ABSTENTION_CAP
    assert calibration["ece"] >= 0.0
    assert 0.0 <= calibration["auroc_correct_vs_incorrect"] <= 1.0
    assert "selection_delta_after_abstention" in calibration
    assert mod._number(True) is None
    assert mod._number("nan") is None


def test_scenario_verify_5033_complete_run_uses_trained_d1_and_no_registry_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5033: complete run refines D1 and beats a bad genuine-SC fixture."""

    _write_d1_artifact(tmp_path, predictions=["A"] * 6)
    for index in range(6):
        _write_checkpoint(tmp_path, index, gold="A", answers=["B", "B", "A"])
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        min_questions=6,
        limit=6,
        bootstrap_samples=32,
        threshold_grid=[0.0, 0.5, 1.0],
        calibration_indices=[0, 1],
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        now=lambda: 10.0,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("success_ebrm_beats_sc_musr_")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["oracle_distinctness_enforced"] is True
    assert artifact["headroom_present"] is True
    assert artifact["abstention_rate"] <= mod.ABSTENTION_CAP
    assert artifact["ebrm_selection_accuracy"] == 1.0
    assert artifact["genuine_tuned_sc_accuracy"] == 0.0
    assert artifact["delta_vs_tuned_sc"] == 1.0
    assert artifact["base_scorer_refined"] == "d1_lora_ebm_trained"
    assert artifact["model_specs"]["registry_fallback_used"] is False
    assert artifact["degeneracy_guard"]["degeneracy_flag"] is False
    assert (
        artifact["uncertainty_calibration"]["threshold_source"] == "held_out_crop_conformal_split"
    )
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["summarize_artifact_exit_code"] == 0
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5033_missing_or_untrained_d1_blocks_honestly(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5033: missing or untrained D1 never falls back to registry."""

    blocked = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        min_questions=1,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )
    _write_d1_artifact(tmp_path, trained=False)
    _write_checkpoint(tmp_path, 0, gold="A", answers=["A", "B"])
    untrained = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "untrained.json",
        min_questions=1,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )

    for artifact in (blocked, untrained):
        assert artifact["honest_verdict"] == "blocked_d1_base_scorer_not_trained"
        assert artifact["verifier_is_oracle"] is False
        assert artifact["headroom_present"] is False
        assert artifact["ebrm_selection_accuracy"] is None
        assert artifact["base_scorer_refined"] != "registry_quality_ensemble"
        assert artifact["model_specs"]["registry_fallback_used"] is False
        assert artifact["preconditions_checked"][0]["resource"] == "d1_base_scorer_trained"
        assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5033_schema_verdict_and_error_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5033: schema, verdict, and fail-closed branches are explicit."""

    base = mod.BaseScorer(
        name="d1_lora_ebm_trained",
        detail="fixture D1",
        artifact_path=tmp_path / "d1.json",
        predictions=[],
        model_specs={"base_model": "Qwen/Qwen3.5-2B"},
    )
    evaluation = {
        "n_rows": 200,
        "ebrm_selection_accuracy": 0.7,
        "point_estimate_accuracy": 0.65,
        "tuned_self_consistency": {"accuracy": 0.6, "config": {"k": 5}, "predictions": []},
        "oracle_at_k": 0.85,
        "headroom_present": True,
        "delta_vs_tuned_sc": 0.1,
        "paired_ci95": [0.01, 0.19],
        "mcnemar_p": 0.01,
        "abstention_rate": 0.25,
        "abstention_degeneracy_guard": mod.harness.abstention_degeneracy_guard(0.25),
        "paired_correct": {"ebrm": [], "point_estimate": [], "tuned_self_consistency": []},
        "predictions": {"ebrm": [], "point_estimate": [], "tuned_self_consistency": []},
    }
    calibration = {
        "selected_threshold": 0.5,
        "selected_abstention_rate": 0.25,
        "threshold_source": "held_out_crop_conformal_split",
        "calibration_curve": [],
        "point_estimate_accuracy": 0.65,
        "best_abstaining_accuracy": 0.7,
        "selection_delta_after_abstention": 0.05,
        "ece": 0.1,
        "auroc_correct_vs_incorrect": 0.8,
        "degeneracy_flag": False,
        "abstention_degeneracy_guard": mod.harness.abstention_degeneracy_guard(0.25),
    }
    success = mod.build_complete_artifact(
        evaluation=evaluation,
        uncertainty_calibration=calibration,
        base_scorer=base,
        preconditions_checked=[],
        duration_s=2.0,
    )
    null = mod.build_complete_artifact(
        evaluation={**evaluation, "paired_ci95": [-0.01, 0.19]},
        uncertainty_calibration=calibration,
        base_scorer=base,
        preconditions_checked=[],
        duration_s=2.0,
    )
    degenerate = mod.build_complete_artifact(
        evaluation={
            **evaluation,
            "abstention_rate": 0.75,
            "abstention_degeneracy_guard": mod.harness.abstention_degeneracy_guard(0.75),
        },
        uncertainty_calibration={
            **calibration,
            "selected_abstention_rate": 0.75,
            "degeneracy_flag": True,
            "abstention_degeneracy_guard": mod.harness.abstention_degeneracy_guard(0.75),
        },
        base_scorer=base,
        preconditions_checked=[],
        duration_s=2.0,
    )
    blocked = mod.build_blocked_artifact(
        preconditions_checked=[],
        duration_s=0.1,
        base_status="missing",
    )

    assert success["honest_verdict"].startswith("success_ebrm_beats_sc_musr_")
    assert null["honest_verdict"].endswith("ci_incl_0")
    assert degenerate["honest_verdict"].endswith("degenerate_abstention")
    assert blocked["honest_verdict"] == "blocked_d1_base_scorer_not_trained"
    assert mod._ci_includes_zero([-0.1, 0.0]) is True
    assert mod._ci_includes_zero([0.1]) is False
    assert mod._format_delta(0.125) == "plus_0p125"
    assert mod._audit_is_clean({"flagged_count": 0}) is True
    assert mod._audit_is_clean({"flags": [{"kind": "WARN"}]}) is False
    assert mod._compact_adversarial_flags({"reports": [{"flags": [{"kind": "WARN"}, "bad"]}]}) == [
        {"kind": "WARN"}
    ]
    assert mod._read_json(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod._read_json(bad) is None
    assert mod.artifact_schema_errors(blocked) == []
    assert "verifier_is_oracle" in mod.artifact_schema_errors(
        {**blocked, "verifier_is_oracle": True}
    )
    assert "paired_ci95" in mod.artifact_schema_errors({**blocked, "paired_ci95": [0.0]})
    assert "spec_refs" in mod.artifact_schema_errors({**blocked, "spec_refs": []})
    assert "abstention_rate" in mod.artifact_schema_errors({**blocked, "abstention_rate": 2.0})
    assert "genuine_tuned_sc_accuracy" in mod.artifact_schema_errors(
        {**blocked, "genuine_tuned_sc_accuracy": 2.0}
    )
    assert "uncertainty_calibration" in mod.artifact_schema_errors(
        {**blocked, "uncertainty_calibration": []}
    )
    assert "field_principles" in mod.artifact_schema_errors({**blocked, "field_principles": {}})
    assert "honest_verdict" in mod.artifact_schema_errors({**blocked, "honest_verdict": "bad"})

    _write_d1_artifact(tmp_path, predictions=["A"])
    _write_checkpoint(tmp_path, 0, gold="A", answers=["A", "B"])
    monkeypatch.setattr(mod, "_oracle_distinctness_enforced", lambda _rows: False)
    oracle_blocked = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "oracle.json",
        min_questions=1,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )
    assert oracle_blocked["honest_verdict"] == "blocked_oracle_distinctness_violation"
    assert mod.artifact_schema_errors(oracle_blocked) == []

    monkeypatch.setattr(mod, "_oracle_distinctness_enforced", lambda _rows: True)
    monkeypatch.setattr(mod, "prepare_rows_with_ebrm_distributions", lambda _rows: [])
    scoring_blocked = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "scoring.json",
        min_questions=1,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )
    assert scoring_blocked["honest_verdict"] == "blocked_ebrm_scoring_error"
    assert "no conflict-aware" in scoring_blocked["blocked_error"]


def test_req_verify_5033_defensive_helpers_and_candidate_missing_branch(tmp_path: Path) -> None:
    """REQ-VERIFY-5033: defensive parsing and validation branches stay fail-closed."""

    base = mod.BaseScorer(
        name="d1_lora_ebm_trained",
        detail="fixture D1",
        artifact_path=tmp_path / "d1.json",
        predictions=[],
        model_specs={"base_model": "Qwen/Qwen3.5-2B"},
    )
    ckdir = tmp_path / mod.MUSR_CHECKPOINT_RELATIVE_DIR
    ckdir.mkdir(parents=True)
    (ckdir / "q0000.json").write_text("{bad", encoding="utf-8")
    _write_json(ckdir / "q0001.json", {"gold": "A", "answers": "A"})
    _write_json(ckdir / "q0002.json", {"gold": "A", "answers": [None, ""]})
    _write_json(
        ckdir / "q0003.json",
        {"gold": "A", "answers": ["A", "B"], "energy_pure_answer": "B"},
    )
    rows = mod.load_cached_musr_rows(ckdir, base_scorer=base, min_questions=1, limit=None)
    with pytest.raises(mod.EbrmScoringError, match="only 1 cached"):
        mod.load_cached_musr_rows(ckdir, base_scorer=base, min_questions=2, limit=None)

    assert rows[0]["d1_prediction"] == "B"
    assert rows[0]["candidates"][0]["d1_base_reward"] == 0.0
    assert mod._candidate_base_reward({"cached_energy_selected": True}) == 1.0
    assert mod._normalized_entropy(mod.Counter(), 0) == 0.0
    assert mod.prepare_rows_with_ebrm_distributions([{"candidates": []}]) == []
    same_answer_rows = mod.prepare_rows_with_ebrm_distributions(
        [{"row_id": "same", "gold": "A", "candidates": [{"answer": "A"}, {"answer": "A"}]}]
    )
    assert mod.conflict_aware_training_rows(same_answer_rows) == []
    duplicate_best_rows = mod.prepare_rows_with_ebrm_distributions(
        [
            {
                "row_id": "dup",
                "gold": "A",
                "candidates": [
                    {"candidate_id": "dup/c0", "answer": "A", "d1_base_reward": 1.0},
                    {"candidate_id": "dup/c1", "answer": "A", "d1_base_reward": 0.9},
                    {"candidate_id": "dup/c2", "answer": "B", "d1_base_reward": 0.1},
                ],
            }
        ]
    )
    duplicate_pairs = mod.conflict_aware_training_rows(duplicate_best_rows)
    assert len(duplicate_pairs) == 1
    assert duplicate_pairs[0]["negative_candidate_id"] == "dup/c2"
    assert mod._best_candidate({"candidates": []}) is None
    assert mod.point_estimate_answer({"candidates": []}) is None
    with pytest.raises(mod.EbrmScoringError, match="no candidates"):
        mod.select_ebrm_answer({"candidates": []}, tuned_sc_answer="A", threshold=0.0)
    fallback_uncertainty = {
        "gold": "A",
        "candidates": [
            {
                "answer": "A",
                "ebrm_selection_reward": 1.0,
                "ebrm_uncertainty": 0.8,
                "cache_index": 0,
            }
        ],
    }
    assert mod.select_ebrm_answer(fallback_uncertainty, tuned_sc_answer="B", threshold=0.9) == "A"
    assert mod._ece([], []) == 0.0
    assert mod._auroc([0.5, 0.5], [1, 0]) == 0.5
    assert mod.calibrate_uncertainty_threshold(rows, [99], [])[1]["degenerate"] is False
    assert mod._default_calibration_indices(6) == [0]
    assert mod._audit_is_clean({"max_severity": 0}) is True
    assert mod._audit_is_clean({"max_severity": -1}) is True

    _write_d1_artifact(tmp_path, predictions=["A"])
    for path in ckdir.glob("q*.json"):
        path.unlink()
    missing_candidates = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "missing_candidates.json",
        min_questions=1,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )
    assert missing_candidates["honest_verdict"] == "blocked_cached_musr_candidates"
    assert mod.artifact_schema_errors(missing_candidates) == []

    errored = mod.build_blocked_artifact(
        preconditions_checked=[],
        duration_s=0.1,
        base_status="d1_lora_ebm_trained_unavailable",
        error="bad d1",
    )
    assert errored["blocked_error"] == "bad d1"

    mcnemar_gated = mod.build_complete_artifact(
        evaluation={
            "n_rows": 200,
            "ebrm_selection_accuracy": 0.7,
            "point_estimate_accuracy": 0.65,
            "tuned_self_consistency": {"accuracy": 0.6, "config": {"k": 5}, "predictions": []},
            "oracle_at_k": 0.85,
            "headroom_present": True,
            "delta_vs_tuned_sc": 0.1,
            "paired_ci95": [0.01, 0.19],
            "mcnemar_p": 0.5,
            "abstention_rate": 0.0,
            "abstention_degeneracy_guard": mod.harness.abstention_degeneracy_guard(0.0),
        },
        uncertainty_calibration={"selected_threshold": 1.0},
        base_scorer=base,
        preconditions_checked=[],
        duration_s=1.0,
    )
    assert mcnemar_gated["honest_verdict"].endswith("mcnemar_or_headroom_gate")

    schema_base = mod.build_blocked_artifact(preconditions_checked=[], duration_s=0.1)
    missing_field = dict(schema_base)
    missing_field.pop("duration_s")
    assert "duration_s" in mod.artifact_schema_errors(missing_field)
    assert "oracle_distinctness_enforced" in mod.artifact_schema_errors(
        {**schema_base, "oracle_distinctness_enforced": "yes"}
    )
    assert "delta_vs_tuned_sc" in mod.artifact_schema_errors(
        {**schema_base, "delta_vs_tuned_sc": "bad"}
    )
    assert "mcnemar_p" in mod.artifact_schema_errors({**schema_base, "mcnemar_p": 2.0})
    assert "preconditions_checked" in mod.artifact_schema_errors(
        {**schema_base, "preconditions_checked": {}}
    )
    assert "model_specs" in mod.artifact_schema_errors({**schema_base, "model_specs": []})
    assert "degeneracy_guard" in mod.artifact_schema_errors({**schema_base, "degeneracy_guard": []})
