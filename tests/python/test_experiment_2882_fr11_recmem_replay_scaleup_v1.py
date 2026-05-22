"""Tests for Exp 2882 FR-11 RecMem replay scale-up.

Spec: REQ-LEARN-2882,
      SCENARIO-LEARN-2882,
      SCENARIO-LEARN-2882-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.eval.fr11_recmem_replay_scaleup_v1 as scaleup
from carnot.eval.fr11_recmem_replay_scaleup_v1 import (
    OUTPUT_FILENAME,
    REQUIRED_ARTIFACT_FIELDS,
    ExperimentConfig,
    run_experiment,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_ready_sources(repo_root: Path) -> None:
    _write_json(
        repo_root / "results" / "experiment_2869_fr11_continuous_self_learning_replay_v3.json",
        {
            "continuous_self_learning_task": True,
            "fr11_self_learning_ready": True,
            "honest_verdict": "complete: offline verifier-feedback replay lowered energy",
            "live_model_invoked": False,
            "no_model_weight_mutation": True,
        },
    )
    _write_json(
        repo_root / "results" / "experiment_2881_fr11_recmem_recurrence_trigger_v1.json",
        {
            "continuous_self_learning_task": True,
            "honest_verdict": "complete: recurrence-triggered consolidation ready",
            "live_llm_called": False,
            "recmem_trigger_ready": True,
        },
    )


def _write_clean_corpus(repo_root: Path, n_per_source: int = 24) -> None:
    _write_json(
        repo_root / "results" / "experiment_2865_cross_corpus_matrix_v5.json",
        {
            "paper_eligible_rows": ["FoVer", "HaluEval/FEVER"],
            "row_status_by_corpus": {"FoVer": "clean", "HaluEval/FEVER": "clean"},
        },
    )
    _write_jsonl(
        repo_root / "data" / "fover_corpus.jsonl",
        [
            {
                "confidence": 1.0,
                "label": "incorrect" if index % 3 == 0 else "correct",
                "question_id": f"fover-{index:03d}",
                "source": "fover_v4",
                "verifier": "heuristic",
            }
            for index in range(n_per_source)
        ],
    )
    _write_jsonl(
        repo_root / "data" / "eval_manifests" / "halueval_20260522.jsonl",
        [
            {
                "candidate": f"candidate-{index}",
                "dataset": "HaluEval",
                "label": 1 if index % 2 == 0 else 0,
                "prompt": f"prompt-{index}",
                "stable_id": f"halueval-{index:03d}",
            }
            for index in range(n_per_source)
        ],
    )
    _write_jsonl(
        repo_root / "data" / "eval_manifests" / "fever_20260522.jsonl",
        [
            {
                "claim": f"claim-{index}",
                "dataset": "FEVER",
                "label": "1" if index % 2 == 0 else "0",
                "prompt": f"prompt-{index}",
                "stable_id": f"fever-{index:03d}",
            }
            for index in range(n_per_source)
        ],
    )


def test_scenario_learn_2882_scaleup_compares_identical_examples(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2882: RecMem trigger matches eager replay with lower token cost."""

    _write_ready_sources(tmp_path)
    _write_clean_corpus(tmp_path)

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            target_examples=50,
            started_at=10.0,
            clock=lambda: 12.0,
        )
    )

    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["recmem_replay_scaleup_ready"] is True
    assert artifact["source_artifacts"] == [
        "results/experiment_2869_fr11_continuous_self_learning_replay_v3.json",
        "results/experiment_2881_fr11_recmem_recurrence_trigger_v1.json",
        "results/experiment_2865_cross_corpus_matrix_v5.json",
    ]
    assert artifact["n_examples"] == 50
    assert artifact["target_examples_met"] is True
    assert artifact["energy_delta_mean"] > 0.0
    assert artifact["eager_energy_delta_mean"] == pytest.approx(artifact["energy_delta_mean"])
    assert artifact["energy_delta_vs_eager"] == pytest.approx(0.0)
    assert artifact["correctness_delta"] == pytest.approx(0.0)
    assert artifact["auroc_valid"] is True
    assert artifact["auroc_delta"] == pytest.approx(0.0)
    assert artifact["token_reduction_pct"] > 90.0
    assert artifact["memory_drift_score"] == pytest.approx(0.0)
    assert artifact["forgetting_regression_count"] == 0
    assert artifact["model_weights_mutated"] is False
    assert artifact["live_llm_called"] is False
    assert artifact["random_seed"] == 2882
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["selected_example_ids"] == artifact["eager_example_ids"]
    assert artifact["recmem_triggered_example_count"] > 0
    assert len(artifact["field_principles"]) >= len(REQUIRED_ARTIFACT_FIELDS) - 1

    saved = json.loads((tmp_path / "results" / OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact

    guard_blocked = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            target_examples=50,
            min_support=99,
        ),
        write=False,
    )
    assert guard_blocked["honest_verdict"] == "blocked_recmem_scaleup_guard_failed"


def test_scenario_learn_2882_blocked_sources_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2882-BLOCKED: missing/not-ready sources do not infer metrics."""

    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_missing_exp2869_artifact"
    assert artifact["recmem_replay_scaleup_ready"] is False
    assert artifact["n_examples"] == 0
    assert artifact["target_examples_met"] is False
    assert artifact["energy_delta_mean"] == 0.0
    assert artifact["correctness_delta"] == 0.0
    assert artifact["auroc_delta"] == 0.0
    assert artifact["token_reduction_pct"] == 0.0
    assert artifact["memory_drift_score"] == 0.0
    assert artifact["forgetting_regression_count"] == 0
    assert artifact["model_weights_mutated"] is False
    assert artifact["live_llm_called"] is False

    written_blocked = run_experiment(
        ExperimentConfig(repo_root=tmp_path / "written", results_dir=tmp_path / "written"),
    )
    assert written_blocked["honest_verdict"] == "blocked_missing_exp2869_artifact"
    assert (tmp_path / "written" / OUTPUT_FILENAME).is_file()

    _write_json(
        tmp_path / "results" / "experiment_2869_fr11_continuous_self_learning_replay_v3.json",
        {"fr11_self_learning_ready": False, "honest_verdict": "blocked_prior"},
    )
    not_ready = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert not_ready["honest_verdict"] == "blocked_exp2869_not_ready"

    _write_json(
        tmp_path / "results" / "experiment_2869_fr11_continuous_self_learning_replay_v3.json",
        {"fr11_self_learning_ready": True},
    )
    missing_2881 = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert missing_2881["honest_verdict"] == "blocked_missing_exp2881_artifact"

    _write_json(
        tmp_path / "results" / "experiment_2881_fr11_recmem_recurrence_trigger_v1.json",
        {"recmem_trigger_ready": False, "honest_verdict": "blocked_prior"},
    )
    not_ready_2881 = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert not_ready_2881["honest_verdict"] == "blocked_exp2881_not_ready"


def test_req_learn_2882_guards_auc_and_small_corpus_accounting(tmp_path: Path) -> None:
    """REQ-LEARN-2882-2/4: guards, AUROC, and target accounting are explicit."""

    with pytest.raises(ValueError, match="target_examples"):
        ExperimentConfig(target_examples=0)
    with pytest.raises(ValueError, match="max_loops"):
        ExperimentConfig(max_loops=0)
    with pytest.raises(ValueError, match="recurrence_threshold"):
        ExperimentConfig(recurrence_threshold=1.1)
    with pytest.raises(ValueError, match="min_support"):
        ExperimentConfig(min_support=1)

    assert scaleup._roc_auc([1, 0], [0.8, 0.2]) == pytest.approx(1.0)
    assert scaleup._roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert scaleup._roc_auc([1, 1], [0.8, 0.7]) is None

    _write_ready_sources(tmp_path)
    _write_clean_corpus(tmp_path, n_per_source=2)
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            target_examples=50,
        ),
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_target_examples_not_met"
    assert artifact["n_examples"] == 6
    assert artifact["target_examples_met"] is False
    assert artifact["auroc_valid"] is True
    assert artifact["model_weights_mutated"] is False
    assert artifact["live_llm_called"] is False
    assert scaleup._final_energy({"energy_before": 0.7, "energy_after_each_loop": []}) == 0.7
    assert scaleup._memory_drift_score(0.5, 1, 0) == 0.0
