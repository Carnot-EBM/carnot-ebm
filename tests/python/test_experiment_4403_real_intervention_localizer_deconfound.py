"""Tests for Exp 4403 real-intervention localizer deconfounding.

Spec refs: REQ-VERIFY-4403, SCENARIO-VERIFY-4403.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4403_real_intervention_localizer_deconfound as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _fover_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, n_bad in (("train_family", 4), ("heldout_family", 5)):
        rows.append(
            {
                "question_id": f"{family}_correct",
                "source": family,
                "label": "correct",
                "confidence": 0.9,
                "step_text": f"{family} reference correction",
            }
        )
        for idx in range(n_bad):
            rows.append(
                {
                    "question_id": f"{family}_bad_{idx}",
                    "source": family,
                    "label": "incorrect",
                    "confidence": 0.8,
                    "step_text": f"{family} failed step {idx}",
                }
            )
    rows.append(
        {
            "question_id": "orphan_bad",
            "source": "orphan_family",
            "label": "incorrect",
            "confidence": 0.7,
            "step_text": "unverified failed step",
        }
    )
    return rows


def _arc_pool() -> dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": "arc-a",
                "candidates": [
                    {"candidate_id": "arc-a-0", "candidate_index": 0, "is_correct": True, "q_mean": 0.9},
                    {"candidate_id": "arc-a-1", "candidate_index": 1, "is_correct": False, "q_mean": 0.1},
                ],
            },
            {
                "task_id": "arc-b",
                "candidates": [
                    {"candidate_id": "arc-b-0", "candidate_index": 0, "is_correct": False, "q_mean": 0.1},
                    {"candidate_id": "arc-b-1", "candidate_index": 1, "is_correct": True, "q_mean": 0.9},
                ],
            },
        ]
    }


def _fixture_config(
    tmp_path: Path,
    *,
    artifact_name: str = "experiment_4403.json",
    min_eval_traces: int = 5,
) -> mod.ExperimentConfig:
    fover_rows_path = tmp_path / "data" / f"{artifact_name}.fover.jsonl"
    step_path = tmp_path / "data" / f"{artifact_name}.steps.jsonl"
    arc_summary_path = tmp_path / "results" / f"{artifact_name}.arc_summary.json"
    arc_pool_path = tmp_path / "results" / f"{artifact_name}.arc_pool.json"
    exp2850_path = tmp_path / "results" / f"{artifact_name}.2850.json"
    exp4381_path = tmp_path / "results" / f"{artifact_name}.4381.json"
    registry_path = tmp_path / "ops" / f"{artifact_name}.registry.yaml"
    _write_jsonl(fover_rows_path, _fover_rows())
    _write_jsonl(step_path, [{"question_id": "s", "step_label": "wrong", "partial_cot": "bad"}])
    _write_json(arc_summary_path, {"n_tasks": 2, "per_task": [{"task": "arc-a"}, {"task": "arc-b"}]})
    _write_json(arc_pool_path, _arc_pool())
    _write_json(exp2850_path, {"n_examples": 1000, "honest_verdict": "complete: fixture"})
    _write_json(
        exp4381_path,
        {"localization_f1_by_direction": {"bidirectional_fusion": {"f1": 0.096491}}},
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")
    return mod.ExperimentConfig(
        repo_root=tmp_path,
        fover_row_corpus_path=fover_rows_path,
        fover_step_corpus_path=step_path,
        arc_summary_path=arc_summary_path,
        arc_candidate_pool_path=arc_pool_path,
        exp2850_artifact_path=exp2850_path,
        exp4381_artifact_path=exp4381_path,
        verifier_registry_path=registry_path,
        verifier_gaps_path=tmp_path / "ops" / f"{artifact_name}.gaps.md",
        artifact_path=tmp_path / "results" / artifact_name,
        heldout_family="heldout_family",
        min_real_intervention_labels=4,
        min_eval_traces=min_eval_traces,
        bootstrap_resamples=80,
        started_at=10.0,
        clock=lambda: 12.0,
    )


def test_req_verify_4403_spec_declares_real_intervention_contract() -> None:
    """REQ-VERIFY-4403: OpenSpec declares the deconfounded localizer contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4403",
        "SCENARIO-VERIFY-4403",
        "experiment_4403_real_intervention_localizer_deconfound.json",
        "localizer_genuinely_beats_position_only",
        "intervention_label_receipts",
        "position_only_baseline_f1",
        "template_family_holdout_drop",
        "blocked_no_intervention_verification_path",
    ):
        assert marker in spec


def test_req_verify_4403_real_intervention_receipts_require_reference_correction() -> None:
    """REQ-VERIFY-4403: failed rows are labels only with verifier-checked corrections."""

    labels = mod.build_fover_intervention_labels_from_rows(_fover_rows())
    failed = [label for label in labels if label.trace.first_error_index is not None]
    receipts = mod.intervention_label_receipts(labels)

    assert len(labels) == 11
    assert len(failed) == 9
    assert all(label.intervention_verified for label in failed)
    assert {label.family for label in failed} == {"train_family", "heldout_family"}
    assert receipts["n_real_traces"] == 11
    assert receipts["n_failed_real_traces"] == 9
    assert receipts["n_intervention_verified"] == 9
    assert receipts["position_distribution"] == {"0": 9}
    assert receipts["family_count"] == {"heldout_family": 5, "train_family": 4}


def test_req_verify_4403_position_only_baseline_ties_one_step_real_labels() -> None:
    """REQ-VERIFY-4403: position-only control blocks a one-step localization win."""

    labels = mod.build_fover_intervention_labels_from_rows(_fover_rows())
    train, heldout = mod.split_by_heldout_family(labels, heldout_family="heldout_family")
    localizer = mod.train_real_contrastive_localizer(train)
    baseline = mod.PositionOnlyBaseline.fit([label.trace for label in train])
    report = mod.evaluate_label_split(
        heldout,
        localizer,
        baseline,
        seed=4403,
        bootstrap_resamples=80,
    )

    assert report["real_intervention_localizer"] == pytest.approx(1.0)
    assert report["position_only_baseline"] == pytest.approx(1.0)
    assert report["delta_vs_position_only"] == pytest.approx(0.0)
    assert report["delta_ci95"] == [0.0, 0.0]
    assert report["beats_position_only_baseline"] is False


def test_req_verify_4403_defensive_helpers_and_schema_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-4403: helper branches stay deterministic and explicit."""

    assert mod._round_float(None) is None
    assert mod._round_float(math.nan) is None
    assert mod.template_family_for_row({"question_id": "gsm8k_1_0"}) == "gsm8k"
    assert mod.template_family_for_row({"question_id": "math_v3_1"}) == "math_v3"
    assert mod.template_family_for_row({"question_id": "math_1"}) == "math"
    assert mod.template_family_for_row({"question_id": "156"}) == "fover_v4_numeric"
    assert mod.reference_intervention_path_available() is True
    assert mod._fover_features({"confidence": "not-a-number"})["detector_score"] == pytest.approx(0.5)
    assert mod._paired_delta_ci95([], [], seed=1, resamples=10) == [None, None]

    gz_path = tmp_path / "payload.json.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        json.dump({"ok": True}, handle)
    assert mod._read_json_any(gz_path) == {"ok": True}

    list_path = tmp_path / "list.json"
    _write_json(list_path, [])
    assert mod._load_exp2850_available(list_path) == (False, "unreadable")
    assert mod._arc_summary_available(list_path) == (False, "unreadable")
    assert (
        mod._blocked_reason([mod.PreconditionCheck("some_resource", False, "missing")])
        == "blocked_cached_corpus_or_ensemble_unavailable"
    )
    clean_trace = mod.exp4392.ProcessTrace(
        trace_id="clean",
        source_domain="fixture",
        steps=(mod.exp4392.ProcessStep(0, "ok", False, {}),),
        first_error_index=None,
    )
    assert mod._arc_successes([clean_trace], lambda _trace: 0) == []

    gap_path = tmp_path / "ops" / "gaps.md"
    mod.append_missing_verifier_gaps(gap_path, [])
    assert not gap_path.exists()
    gap = mod._missing_gap("fixture_confound", 3)
    mod.append_missing_verifier_gaps(gap_path, [gap])
    first = gap_path.read_text(encoding="utf-8")
    mod.append_missing_verifier_gaps(gap_path, [gap])
    assert gap_path.read_text(encoding="utf-8") == first

    errors = mod.artifact_schema_errors(
        {
            "honest_verdict": "bad",
            "localizer_genuinely_beats_position_only": "false",
            "position_only_baseline_f1": 1,
            "template_family_holdout_drop": 0,
            "n_traces": "6",
            "verifier_is_oracle": True,
        }
    )
    assert "missing preconditions_checked" in errors
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "localizer_genuinely_beats_position_only must be bare bool" in errors
    assert "position_only_baseline_f1 must be bare float" in errors
    assert "template_family_holdout_drop must be bare float" in errors
    assert "n_traces must be bare int" in errors
    assert "verifier_is_oracle must be false" in errors


def test_scenario_verify_4403_run_experiment_writes_clean_powered_null(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4403: complete run writes a terminal clean null artifact."""

    fover_rows_path = tmp_path / "data" / "fover_corpus.jsonl"
    step_path = tmp_path / "data" / "step_level_prm_training.jsonl"
    arc_summary_path = tmp_path / "results" / "arc3_trm_verifier_rerank.json"
    arc_pool_path = tmp_path / "results" / "arc_pool.json"
    exp2850_path = tmp_path / "results" / "experiment_2850.json"
    exp4381_path = tmp_path / "results" / "experiment_4381.json"
    registry_path = tmp_path / "ops" / "verifier_registry.yaml"
    artifact_path = tmp_path / "results" / "experiment_4403.json"
    _write_jsonl(fover_rows_path, _fover_rows())
    _write_jsonl(step_path, [{"question_id": "s", "step_label": "wrong", "partial_cot": "bad"}])
    _write_json(arc_summary_path, {"n_tasks": 2, "per_task": [{"task": "arc-a"}, {"task": "arc-b"}]})
    _write_json(arc_pool_path, _arc_pool())
    _write_json(exp2850_path, {"n_examples": 1000, "honest_verdict": "complete: fixture"})
    _write_json(
        exp4381_path,
        {"localization_f1_by_direction": {"bidirectional_fusion": {"f1": 0.096491}}},
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            fover_row_corpus_path=fover_rows_path,
            fover_step_corpus_path=step_path,
            arc_summary_path=arc_summary_path,
            arc_candidate_pool_path=arc_pool_path,
            exp2850_artifact_path=exp2850_path,
            exp4381_artifact_path=exp4381_path,
            verifier_registry_path=registry_path,
            verifier_gaps_path=tmp_path / "ops" / "verifier_gaps.md",
            artifact_path=artifact_path,
            heldout_family="heldout_family",
            min_real_intervention_labels=4,
            min_eval_traces=5,
            bootstrap_resamples=80,
            started_at=10.0,
            clock=lambda: 12.0,
        ),
        ensemble_loader=lambda: True,
        intervention_verifier_checker=lambda: True,
        adversarial_verify_runner=lambda _path: {"returncode": 0, "stdout_tail": "clean"},
        write=True,
    )

    assert artifact_path.is_file()
    assert artifact["honest_verdict"] == "complete: clean_powered_null_position_only_not_beaten"
    assert artifact["localizer_genuinely_beats_position_only"] is False
    assert artifact["position_only_baseline_f1"] == pytest.approx(1.0)
    assert artifact["template_family_holdout_drop"] == pytest.approx(0.0)
    assert artifact["n_traces"] == 6
    assert artifact["verifier_is_oracle"] is False
    assert artifact["intervention_label_receipts"]["n_intervention_verified"] == 9
    assert artifact["localization_f1_by_domain"]["FoVer"]["delta_ci95"] == [0.0, 0.0]
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4403_short_heldout_dry_run_and_adversarial_flag(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4403: edge run modes are explicit and terminal."""

    blocked = mod.run_experiment(
        _fixture_config(tmp_path, artifact_name="blocked_short.json", min_eval_traces=99),
        ensemble_loader=lambda: True,
        intervention_verifier_checker=lambda: True,
        write=True,
    )
    assert blocked["honest_verdict"] == "blocked_cached_corpus_or_ensemble_unavailable"
    assert blocked["preconditions_checked"][-1]["resource"] == "heldout_family_eval_split"

    dry_run = mod.run_experiment(
        _fixture_config(tmp_path, artifact_name="dry_run.json"),
        ensemble_loader=lambda: True,
        intervention_verifier_checker=lambda: True,
        write=False,
    )
    assert dry_run["adversarial_verify"] == {"returncode": None, "skipped": True}

    flagged = mod.run_experiment(
        _fixture_config(tmp_path, artifact_name="flagged.json"),
        ensemble_loader=lambda: True,
        intervention_verifier_checker=lambda: True,
        adversarial_verify_runner=lambda _path: {"returncode": 1, "stdout_tail": "flag"},
        write=True,
    )
    assert flagged["flagged_adversarial"] is True
    assert mod.artifact_schema_errors(flagged) == []


def test_scenario_verify_4403_blocks_without_intervention_path(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4403: missing intervention verification path stops early."""

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            fover_row_corpus_path=tmp_path / "missing_fover.jsonl",
            fover_step_corpus_path=tmp_path / "missing_steps.jsonl",
            arc_summary_path=tmp_path / "missing_arc.json",
            arc_candidate_pool_path=tmp_path / "missing_pool.json",
            exp2850_artifact_path=tmp_path / "missing_2850.json",
            exp4381_artifact_path=tmp_path / "missing_4381.json",
            verifier_registry_path=tmp_path / "missing_registry.yaml",
            artifact_path=tmp_path / "results" / "blocked.json",
            started_at=1.0,
            clock=lambda: 1.5,
        ),
        ensemble_loader=lambda: False,
        intervention_verifier_checker=lambda: False,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_no_intervention_verification_path"
    assert artifact["localizer_genuinely_beats_position_only"] is False
    assert artifact["intervention_label_receipts"]["n_intervention_verified"] == 0
    assert artifact["adversarial_verify"]["skipped"] == "blocked"
    assert mod.artifact_schema_errors(artifact) == []
