"""Tests for Exp 5005 EBRM uncertainty-aware MuSR selector.

Spec refs: REQ-VERIFY-5005, SCENARIO-VERIFY-5005.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5005_ebrm_uncertainty_verifier as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_cheap_base(root: Path) -> None:
    _write_json(
        root / mod.CHEAP_BASE_RELATIVE_PATH,
        {
            "experiment": "distributional_energy_verifier_musr",
            "verifier_is_oracle": False,
            "distributional_energy_accuracy": 0.52,
            "self_consistency_accuracy": 0.50,
        },
    )


def _write_checkpoint(
    path: Path,
    *,
    gold: str,
    answers: list[str | None],
    energy_answer: str | None = None,
) -> None:
    first_answer = next((answer for answer in answers if answer is not None), "")
    _write_json(
        path,
        {
            "q": int(path.stem.removeprefix("q")),
            "gold": gold,
            "answers": answers,
            "temperature": "cached",
            "energy_answer": energy_answer or str(first_answer),
            "energy_pure_answer": energy_answer or str(first_answer),
        },
    )


def _audit_clean(_path: Path) -> dict[str, Any]:
    return {"flag_count": 0, "flags": []}


def _fixture_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "r0",
            "gold": "A",
            "candidates": [
                {"candidate_id": "r0/c0", "answer": "A", "cache_index": 0, "base_reward": 1.0},
                {"candidate_id": "r0/c1", "answer": "B", "cache_index": 1, "base_reward": 0.0},
            ],
        },
        {
            "row_id": "r1",
            "gold": "B",
            "candidates": [
                {"candidate_id": "r1/c0", "answer": "A", "cache_index": 0, "base_reward": 0.1},
                {"candidate_id": "r1/c1", "answer": "B", "cache_index": 1, "base_reward": 0.9},
            ],
        },
        {
            "row_id": "r2",
            "gold": "A",
            "candidates": [
                {"candidate_id": "r2/c0", "answer": "B", "cache_index": 0, "base_reward": 0.55},
                {"candidate_id": "r2/c1", "answer": "A", "cache_index": 1, "base_reward": 0.54},
            ],
        },
    ]


def test_req_verify_5005_spec_declares_ebrm_contract() -> None:
    """REQ-VERIFY-5005: OpenSpec anchors the EBRM artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5005",
        "SCENARIO-VERIFY-5005",
        "experiment_5005_ebrm_uncertainty_verifier.py",
        "results/experiment_5005_ebrm_uncertainty_verifier.json",
        "arXiv:2504.13134",
        "blocked_<resource>",
        "success_ebrm_beats_sc_musr_",
        "complete_ebrm_no_win_musr_",
        "reward-distribution",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_verify_5005_reward_distribution_and_training_rows_are_bounded() -> None:
    """REQ-VERIFY-5005: conflict filtering and noise weights shape distributions."""

    rows = mod.prepare_rows_with_ebrm_distributions(_fixture_rows())
    pairs = mod.conflict_aware_training_rows(rows)
    curve = mod.calibration_curve(rows, [0.0, 0.4, 1.0])
    threshold = mod.calibrate_uncertainty_threshold(rows, [0, 2], [0.0, 0.4, 1.0])

    confident = rows[0]["candidates"][0]["ebrm_reward_distribution"]
    ambiguous = rows[2]["candidates"][0]["ebrm_reward_distribution"]

    assert confident["mean_reward"] > rows[0]["candidates"][1]["ebrm_reward_distribution"]["mean_reward"]
    assert ambiguous["spread"] > confident["spread"]
    assert pairs[0]["label_noise_weight"] > 0.0
    assert all(pair["conflict_aware_filtered"] for pair in pairs)
    assert threshold in [0.0, 0.4, 1.0]
    assert {row["threshold"] for row in curve} == {0.0, 0.4, 1.0}
    assert curve[-1]["coverage"] == 1.0
    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod._number("nan") is None


def test_scenario_verify_5005_blocked_artifact_names_missing_resource(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5005: missing base scorer writes an honest blocked artifact."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        cuda_available=lambda: False,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_base_scorer"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["headroom_present"] is False
    assert artifact["ebrm_selection_accuracy"] is None
    assert artifact["preconditions_checked"][0]["available"] is False
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5005_complete_run_scores_and_abstains_oracle_distinct(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5005: complete run evaluates guarded cached MuSR rows."""

    _write_cheap_base(tmp_path)
    ckpt_dir = tmp_path / mod.CHECKPOINT_RELATIVE_DIR
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    _write_checkpoint(ckpt_dir / "q0000.json", gold="A", answers=["A", "B"], energy_answer="A")
    _write_checkpoint(ckpt_dir / "q0001.json", gold="B", answers=["A", "B"], energy_answer="B")
    _write_checkpoint(ckpt_dir / "q0002.json", gold="A", answers=["B", "A"], energy_answer="B")

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        cuda_available=lambda: False,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=3,
        limit=3,
        bootstrap_samples=32,
        threshold_grid=[0.0, 0.5, 1.0],
        calibration_indices=[0],
        write=True,
    )

    assert artifact["honest_verdict"].startswith("complete_ebrm_no_win_musr_")
    assert artifact["ebrm_selection_accuracy"] == pytest.approx(2 / 3, abs=1e-6)
    assert artifact["tuned_sc_accuracy"] == pytest.approx(1 / 3, abs=1e-6)
    assert artifact["delta_vs_tuned_sc"] == pytest.approx(1 / 3, abs=1e-6)
    assert artifact["headroom_present"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["oracle_distinctness_enforced"] is True
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["summarize_artifact_exit_code"] == 0
    assert artifact["base_scorer_refined"] == "registry_quality_ensemble"
    assert artifact["uncertainty_calibration"]["selected_threshold"] == 0.0
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5005_verdict_schema_and_oracle_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5005: schema and guarded candidate branches fail closed."""

    evaluation = {
        "n_rows": 200,
        "ebrm_selection_accuracy": 0.7,
        "tuned_self_consistency": {"accuracy": 0.6, "config": {"k": 1}, "predictions": []},
        "oracle_at_k": 0.8,
        "headroom_present": True,
        "delta_vs_tuned_sc": 0.1,
        "paired_ci95": [0.01, 0.19],
        "mcnemar_p": 0.01,
        "paired_correct": {"ebrm": [], "tuned_self_consistency": []},
        "predictions": {"ebrm": [], "point_estimate": [], "tuned_self_consistency": []},
    }
    success = mod.build_complete_artifact(
        evaluation=evaluation,
        uncertainty_calibration={
            "selected_threshold": 0.5,
            "calibration_curve": [],
            "point_estimate_accuracy": 0.65,
            "best_abstaining_accuracy": 0.7,
            "calibration_improvement_vs_point": 0.05,
        },
        base_scorer=mod.BaseScorer("registry_quality_ensemble", "fixture", None, {}),
        preconditions_checked=[],
        duration_s=1.0,
    )
    null = mod.build_complete_artifact(
        evaluation={**evaluation, "mcnemar_p": 0.5},
        uncertainty_calibration=success["uncertainty_calibration"],
        base_scorer=mod.BaseScorer("registry_quality_ensemble", "fixture", None, {}),
        preconditions_checked=[],
        duration_s=1.0,
    )
    blocked = mod.build_blocked_artifact(
        missing_resource="cached_musr_candidates",
        preconditions_checked=[],
        duration_s=0.1,
    )

    assert success["honest_verdict"].startswith("success_ebrm_beats_sc_musr_")
    assert null["honest_verdict"].endswith("mcnemar_or_headroom_gate")
    assert mod._ci_includes_zero([-0.1, 0.1]) is True
    assert mod._ci_includes_zero([0.01, 0.1]) is False
    assert mod._audit_is_clean({"max_severity": 0, "flag_count": 1}) is True
    assert mod._audit_is_clean({"flagged_count": 0}) is True
    assert mod._audit_is_clean({"reports": [{"flags": [{"kind": "WARN"}, "bad"]}]}) is False
    assert mod._compact_adversarial_flags({"reports": [{"flags": [{"kind": "WARN"}, "bad"]}]}) == [
        {"kind": "WARN"}
    ]
    assert mod.artifact_schema_errors(blocked) == []
    assert "verifier_is_oracle" in mod.artifact_schema_errors({**blocked, "verifier_is_oracle": True})
    assert "paired_ci95" in mod.artifact_schema_errors({**blocked, "paired_ci95": [0.0]})
    assert "spec_refs" in mod.artifact_schema_errors({**blocked, "spec_refs": ["REQ-VERIFY-5005"]})
    assert "ebrm_selection_accuracy" in mod.artifact_schema_errors(
        {**blocked, "ebrm_selection_accuracy": 2.0}
    )
    assert "uncertainty_calibration" in mod.artifact_schema_errors(
        {**blocked, "uncertainty_calibration": []}
    )
    with pytest.raises(mod.EbrmScoringError, match="no candidates"):
        mod.select_ebrm_answer({"candidates": []}, tuned_sc_answer="A", threshold=0.5)


def test_req_verify_5005_defensive_edges_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5005: fallback branches stay deterministic and covered."""

    d1_path = tmp_path / mod.D1_BASE_RELATIVE_PATH
    _write_json(
        d1_path,
        {
            "checkpoint_path": "/tmp/adapter",
            "trained_scorer_accuracy": 0.6,
            "adversarial_verify_clean": True,
            "model_specs": {"base_model": "Qwen/Qwen3.5-4B"},
        },
    )
    assert mod.resolve_base_scorer(tmp_path).name == "d1_lora_ebm"
    assert mod._candidate_base_reward({"trivial_energy": "0.25"}) == -0.25
    assert mod._candidate_base_reward({"cached_judge_selected": True}) == 0.25
    assert mod._default_calibration_indices(3) == [0]

    ckpt_dir = tmp_path / "edge_ckpts"
    _write_json(ckpt_dir / "q0000.json", ["bad"])
    _write_checkpoint(ckpt_dir / "q0001.json", gold="A", answers=[None, ""], energy_answer="A")
    _write_checkpoint(ckpt_dir / "q0002.json", gold="A", answers=["A", "A"], energy_answer="A")
    rows = mod.load_cached_musr_rows(ckpt_dir, limit=None, min_questions=1)
    assert rows[0]["row_id"] == "q0002"
    with pytest.raises(mod.EbrmScoringError, match="only 1 cached MuSR rows"):
        mod.load_cached_musr_rows(ckpt_dir, limit=None, min_questions=2)

    prepared = mod.prepare_rows_with_ebrm_distributions(
        [{"row_id": "empty", "candidates": []}, {"row_id": "same", "candidates": rows[0]["candidates"]}]
    )
    assert len(prepared) == 1
    assert mod.conflict_aware_training_rows(prepared) == []
    same_answer_row = {
        "row_id": "same-answer",
        "candidates": [
            {
                "candidate_id": "a",
                "answer": "A",
                "ebrm_reward_distribution": {"mean_reward": 1.0, "spread": 0.0},
            },
            {
                "candidate_id": "b",
                "answer": "A",
                "ebrm_reward_distribution": {"mean_reward": 0.5, "spread": 0.0},
            },
        ],
    }
    assert mod.conflict_aware_training_rows([same_answer_row]) == []
    mixed_same_answer_row = {
        "row_id": "mixed-same-answer",
        "candidates": [
            {
                "candidate_id": "a0",
                "answer": "A",
                "ebrm_reward_distribution": {"mean_reward": 1.0, "spread": 0.0},
            },
            {
                "candidate_id": "a1",
                "answer": "A",
                "ebrm_reward_distribution": {"mean_reward": 0.9, "spread": 0.0},
            },
            {
                "candidate_id": "b0",
                "answer": "B",
                "ebrm_reward_distribution": {"mean_reward": 0.4, "spread": 0.2},
            },
        ],
    }
    assert len(mod.conflict_aware_training_rows([mixed_same_answer_row])) == 1
    row_without_uncertainty = {
        "candidates": [
            {
                "candidate_id": "a",
                "answer": "A",
                "ebrm_reward_distribution": {"mean_reward": 1.0, "spread": 0.0},
                "ebrm_uncertainty": 1.0,
            }
        ]
    }
    assert mod.select_ebrm_answer(row_without_uncertainty, tuned_sc_answer="B", threshold=0.5) == "B"
    assert mod.calibrate_uncertainty_threshold(prepared, [99], [0.0]) == 0.0
    assert mod.build_blocked_artifact(
        missing_resource="x",
        preconditions_checked=[],
        duration_s=0.0,
        error="boom",
    )["blocked_error"] == "boom"

    blocked = mod.build_blocked_artifact(
        missing_resource="base_scorer",
        preconditions_checked=[],
        duration_s=0.0,
    )
    for mutated, field in (
        ({key: value for key, value in blocked.items() if key != "duration_s"}, "duration_s"),
        ({**blocked, "headroom_present": "no"}, "headroom_present"),
        ({**blocked, "delta_vs_tuned_sc": "0.1"}, "delta_vs_tuned_sc"),
        ({**blocked, "mcnemar_p": 2.0}, "mcnemar_p"),
        ({**blocked, "preconditions_checked": {}}, "preconditions_checked"),
        ({**blocked, "model_specs": []}, "model_specs"),
        ({**blocked, "field_principles": {}}, "field_principles"),
        ({**blocked, "honest_verdict": "maybe"}, "honest_verdict"),
    ):
        assert field in mod.artifact_schema_errors(mutated)

    _write_cheap_base(tmp_path)
    run_ckpt_dir = tmp_path / mod.CHECKPOINT_RELATIVE_DIR
    _write_checkpoint(run_ckpt_dir / "q0000.json", gold="A", answers=["A"], energy_answer="A")
    scoring_error = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "scoring_error.json",
        cuda_available=lambda: False,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=1,
        limit=1,
        write=True,
    )
    assert scoring_error["honest_verdict"] == "blocked_scoring_error"
    assert "no conflict-aware pseudo-pairs" in scoring_error["blocked_error"]

    _write_checkpoint(run_ckpt_dir / "q0000.json", gold="A", answers=["A", "B"], energy_answer="A")
    sleeps: list[float] = []

    def fake_default_audit(_path: Path) -> dict[str, Any]:
        return {"max_severity": 0, "flags": []}

    monkeypatch.setattr(mod, "run_adversarial_verify", fake_default_audit)
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: sleeps.append(seconds))
    sleepy_complete = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "sleepy_complete.json",
        cuda_available=lambda: False,
        audit_runner=fake_default_audit,
        summary_runner=lambda _path: 0,
        min_questions=1,
        limit=1,
        now=lambda: 0.0,
        write=True,
    )
    assert sleepy_complete["honest_verdict"].startswith("complete_ebrm_no_win_musr_")
    assert sleeps == [pytest.approx(1.05)]

    monkeypatch.setattr(mod, "_oracle_distinctness_enforced", lambda _rows: False)
    oracle_error = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "oracle_error.json",
        cuda_available=lambda: False,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=1,
        limit=1,
        write=True,
    )
    assert oracle_error["honest_verdict"] == "blocked_oracle_distinctness_violation"
