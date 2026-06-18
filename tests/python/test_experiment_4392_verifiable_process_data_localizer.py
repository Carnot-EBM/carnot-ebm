"""Tests for Exp 4392 verifiable process-data first-error localizer.

Spec refs: REQ-VERIFY-4392, SCENARIO-VERIFY-4392.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4392_verifiable_process_data_localizer as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _fover_rows() -> list[dict[str, Any]]:
    return [
        {
            "trace_id": "bad",
            "step_index": 0,
            "partial_cot": "Claim 2 + 2 = 5.",
            "step_label": "wrong",
            "cascade_score": 0.91,
            "prefix_fraction": 0.25,
        },
        {
            "trace_id": "bad",
            "step_index": 1,
            "partial_cot": "Then 5 + 1 = 6.",
            "step_label": "wrong",
            "cascade_score": 0.65,
            "prefix_fraction": 0.50,
        },
        {
            "trace_id": "clean",
            "step_index": 0,
            "partial_cot": "Claim 2 + 2 = 4.",
            "step_label": "correct",
            "cascade_score": 0.04,
            "prefix_fraction": 0.25,
        },
        {
            "trace_id": "clean",
            "step_index": 1,
            "partial_cot": "Then 4 + 1 = 5.",
            "step_label": "correct",
            "cascade_score": 0.03,
            "prefix_fraction": 0.50,
        },
    ]


def _arc_pool() -> dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": "arc-a",
                "candidates": [
                    {
                        "candidate_id": "arc-a-0",
                        "candidate_index": 0,
                        "is_correct": True,
                        "q_mean": 0.95,
                        "features": {"cell_confidence_mean": 0.95},
                    },
                    {
                        "candidate_id": "arc-a-1",
                        "candidate_index": 1,
                        "is_correct": False,
                        "q_mean": 0.10,
                        "features": {"cell_confidence_mean": 0.10},
                    },
                ],
            },
            {
                "task_id": "arc-b",
                "candidates": [
                    {
                        "candidate_id": "arc-b-0",
                        "candidate_index": 0,
                        "is_correct": False,
                        "q_mean": 0.20,
                        "features": {"cell_confidence_mean": 0.20},
                    },
                    {
                        "candidate_id": "arc-b-1",
                        "candidate_index": 1,
                        "is_correct": True,
                        "q_mean": 0.90,
                        "features": {"cell_confidence_mean": 0.90},
                    },
                ],
            },
        ]
    }


def test_req_verify_4392_spec_declares_process_data_localizer_contract() -> None:
    """REQ-VERIFY-4392: OpenSpec declares fields, blocked modes, and gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4392",
        "SCENARIO-VERIFY-4392",
        "experiment_4392_verifiable_process_data_localizer.json",
        "localizer_beats_ensemble_baseline",
        "synthesis_verification",
        "structured_abstention",
        "blocked_cached_corpus_or_ensemble_unavailable",
        "blocked_no_prefix_verification_path",
    ):
        assert marker in spec


def test_req_verify_4392_synthesis_verifies_prefix_invalidity_and_suffix() -> None:
    """REQ-VERIFY-4392: synthetic first-error traces are executable and verified."""

    corpus = mod.synthesize_verifiable_first_error_corpus(n_traces=40, seed=4392)
    summary = mod.synthesis_verification_summary(corpus)

    assert len(corpus) == 40
    assert summary["n_synthetic_traces"] == 40
    assert summary["prefix_invalidity_verified_fraction"] == pytest.approx(1.0)
    assert summary["trajectory_consistency_fraction"] == pytest.approx(1.0)
    assert all(trace.first_error_index is not None for trace in corpus)
    assert {trace.source_domain for trace in corpus} == {"fover_math_symbolic"}


def test_req_verify_4392_contrastive_localizer_selects_earliest_break() -> None:
    """REQ-VERIFY-4392: fitted localizer separates first break from inheritors."""

    train = mod.synthesize_verifiable_first_error_corpus(n_traces=80, seed=4392)
    model = mod.train_contrastive_localizer(train)
    holdout = mod.synthesize_verifiable_first_error_corpus(n_traces=16, seed=4400)
    metrics = mod.evaluate_domain_localization(
        "synthetic_check",
        holdout,
        model,
        baseline_f1=0.096,
        seed=4392,
        bootstrap_resamples=120,
    )

    assert model.weights["score_onset"] > 0.0
    assert metrics["synthetic_trained_localizer"] > 0.80
    assert metrics["delta"] > 0.70
    assert metrics["delta_ci95"][0] > 0.0


def test_scenario_verify_4392_artifact_has_required_bare_fields(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4392: complete artifacts expose decision fields bare."""

    synthetic = mod.synthesize_verifiable_first_error_corpus(n_traces=24, seed=4392)
    model = mod.train_contrastive_localizer(synthetic)
    fover = mod.load_fover_real_traces_from_rows(_fover_rows())
    arc = mod.load_arc_process_proxy_traces(_arc_pool())
    domain_metrics = {
        "FoVer": mod.evaluate_domain_localization(
            "FoVer", fover, model, baseline_f1=0.096, seed=4392, bootstrap_resamples=120
        ),
        "GAP-4 ARC": mod.evaluate_domain_localization(
            "GAP-4 ARC", arc, model, baseline_f1=0.096, seed=4393, bootstrap_resamples=120
        ),
    }
    artifact = mod.build_complete_artifact(
        synthetic_corpus=synthetic,
        localizer=model,
        domain_traces={"FoVer": fover, "GAP-4 ARC": arc},
        localization_f1_by_domain=domain_metrics,
        preconditions_checked=[
            mod.PreconditionCheck("cached_step_labeled_fover_corpus", True, "fixture").as_dict()
        ],
        source_paths=[tmp_path / "source.json"],
        duration_s=1.25,
        random_seed=4392,
        bootstrap_resamples=120,
    )

    assert artifact["localizer_beats_ensemble_baseline"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["n_traces"] == len(fover)
    assert artifact["synthesis_verification"]["n_synthetic_traces"] == 24
    assert artifact["structured_abstention"]["precision_at_recall_0_9"] is not None
    assert artifact["model_specs"]["trm_training"] == "stood_down_not_invoked"
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4392_blocked_artifact_is_terminal() -> None:
    """SCENARIO-VERIFY-4392: missing prefix verification blocks without metrics."""

    artifact = mod.build_blocked_artifact(
        honest_verdict="blocked_no_prefix_verification_path",
        preconditions_checked=[
            mod.PreconditionCheck("prefix_verification_path", False, "disabled").as_dict()
        ],
        source_paths=[],
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "blocked_no_prefix_verification_path"
    assert artifact["localizer_beats_ensemble_baseline"] is False
    assert artifact["localization_f1_by_domain"] == {}
    assert artifact["synthesis_verification"]["n_synthetic_traces"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4392_run_experiment_writes_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4392: run path fits CPU localizer and writes verifier report."""

    fover_path = tmp_path / "data" / "steps.jsonl"
    arc_path = tmp_path / "results" / "arc_pool.json"
    registry_path = tmp_path / "ops" / "verifier_registry.yaml"
    exp4381_path = tmp_path / "results" / "experiment_4381.json"
    artifact_path = tmp_path / "results" / "experiment_4392.json"
    gaps_path = tmp_path / "ops" / "verifier_gaps.md"
    _write_jsonl(fover_path, _fover_rows())
    _write_json(arc_path, _arc_pool())
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        "verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8"
    )
    _write_json(
        exp4381_path,
        {
            "localization_f1_by_direction": {
                "bidirectional_fusion": {"f1": 0.096491},
            },
            "model_specs": {"fusion_method": "mean_l2r_r2l"},
        },
    )

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            fover_step_corpus_path=fover_path,
            arc_candidate_pool_path=arc_path,
            verifier_registry_path=registry_path,
            exp4381_artifact_path=exp4381_path,
            artifact_path=artifact_path,
            verifier_gaps_path=gaps_path,
            min_synthetic_traces=32,
            min_eval_traces=2,
            bootstrap_resamples=120,
            started_at=2.0,
            clock=lambda: 4.0,
        ),
        ensemble_loader=lambda: True,
        prefix_verifier_checker=lambda: True,
        adversarial_verify_runner=lambda _path: {"returncode": 0, "flags": []},
        write=True,
    )

    assert artifact_path.is_file()
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["localizer_beats_ensemble_baseline"] is True
    assert artifact["localization_f1_by_domain"]["FoVer"]["ensemble_baseline_0096"] == 0.096
    assert artifact["adversarial_verify"]["returncode"] == 0
