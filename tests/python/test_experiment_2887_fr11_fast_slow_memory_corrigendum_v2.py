"""Tests for Exp 2887 FR-11 fast/slow memory corrigendum.

Spec: REQ-LEARN-2887,
      SCENARIO-LEARN-2887,
      SCENARIO-LEARN-2887-GUARD.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.eval.fr11_fast_slow_memory_corrigendum_v2 as fs_corrigendum
from carnot.eval.fr11_fast_slow_memory_corrigendum_v2 import (
    OUTPUT_FILENAME,
    POLICIES,
    REQUIRED_ARTIFACT_FIELDS,
    CausalRecMemPolicy,
    ExperimentConfig,
    FastSlowMemoryPolicy,
    PolicyRow,
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
    _write_json(
        repo_root / "results" / "experiment_2882_fr11_recmem_replay_scaleup_v1.json",
        {
            "continuous_self_learning_task": True,
            "honest_verdict": (
                "complete: RecMem-triggered replay matched eager replay with lower token cost"
            ),
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {
                    "kind": "TAUTOLOGY",
                    "severity": "critical",
                    "detail": "eager_energy_delta_mean=0.1 and energy_delta_mean=0.1",
                }
            ],
            "recmem_replay_scaleup_ready": True,
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


def _row(row_id: str, motif: str, *, after: float = 0.35) -> PolicyRow:
    return PolicyRow(
        event_id=row_id,
        source="FEVER",
        motif_key=motif,
        initial_energy=0.85,
        replay_final_energy=after,
        initial_correct=False,
        replay_correct=True,
        localized_violations=(motif,),
    )


def test_scenario_learn_2887_fast_slow_update_rule_consolidates_and_forgets() -> None:
    """SCENARIO-LEARN-2887-GUARD: fast/slow memory weakens unsafe edges."""

    policy = FastSlowMemoryPolicy()
    first = _row("first", "factuality_mismatch")
    second = _row("second", "factuality_mismatch")
    third = _row("third", "factuality_mismatch")

    assert policy.should_apply(first) is False
    policy.observe(first)
    assert policy.fast_strength_by_motif["factuality_mismatch"] > 0.0
    assert policy.should_apply(second) is True
    policy.observe(second)
    policy.observe(third)
    assert policy.slow_strength_by_motif["factuality_mismatch"] > 0.0

    slow_before = policy.slow_strength_by_motif["factuality_mismatch"]
    policy.observe(
        PolicyRow(
            event_id="contradiction",
            source="FEVER",
            motif_key="factuality_mismatch",
            initial_energy=0.2,
            replay_final_energy=0.6,
            initial_correct=True,
            replay_correct=False,
            localized_violations=("factuality_mismatch",),
        )
    )

    assert policy.slow_strength_by_motif["factuality_mismatch"] < slow_before
    assert policy.forgetting_regression_count == 1
    assert policy.contradiction_rate() == pytest.approx(1.0)


def test_scenario_learn_2887_corrigendum_separates_policies(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2887: the corrigendum uses causal, non-retroactive policies."""

    _write_ready_sources(tmp_path)
    _write_clean_corpus(tmp_path)

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            target_examples=50,
            started_at=10.0,
            clock=lambda: 12.25,
        )
    )

    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["fr11_scaleup_clean"] is True
    assert artifact["source_artifacts"] == [
        "results/experiment_2869_fr11_continuous_self_learning_replay_v3.json",
        "results/experiment_2881_fr11_recmem_recurrence_trigger_v1.json",
        "results/experiment_2882_fr11_recmem_replay_scaleup_v1.json",
    ]
    assert artifact["n_examples"] == 50
    assert artifact["target_examples_met"] is True
    assert artifact["policies_compared"] == list(POLICIES)
    assert artifact["model_weights_mutated"] is False
    assert artifact["live_llm_called"] is False
    assert artifact["random_seed"] == 2887
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == pytest.approx(2.25)

    energy = artifact["energy_delta_by_policy"]
    assert energy["eager_replay"] > energy["fast_slow_memory"]
    assert energy["fast_slow_memory"] > energy["recmem_causal_triggered"]
    assert len(set(energy.values())) == 3

    assert artifact["token_reduction_by_policy"]["eager_replay"] == pytest.approx(0.0)
    assert artifact["token_reduction_by_policy"]["recmem_causal_triggered"] > 0.0
    assert artifact["token_reduction_by_policy"]["fast_slow_memory"] > 0.0
    assert artifact["duplicate_rate_by_policy"]["eager_replay"] > 0.0
    assert artifact["duplicate_rate_by_policy"]["recmem_causal_triggered"] == pytest.approx(0.0)
    assert artifact["memory_drift_by_policy"]["fast_slow_memory"] == pytest.approx(0.0)
    assert artifact["forgetting_regression_count_by_policy"]["fast_slow_memory"] == 0
    assert artifact["selected_examples_checksum"]
    assert artifact["source_file_checksums"]
    assert artifact["exp2882_flag_diagnosis"]["root_cause"] == "retroactive_cluster_application"
    assert artifact["adversarial_clean_checks"]["non_tautological_policy_energy"] is True

    saved = json.loads((tmp_path / "results" / OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact


def test_req_learn_2887_blocks_missing_sources_and_small_corpus(tmp_path: Path) -> None:
    """REQ-LEARN-2887-1/2: source and target-example gates fail closed."""

    with pytest.raises(ValueError, match="target_examples"):
        ExperimentConfig(target_examples=0)
    with pytest.raises(ValueError, match="max_loops"):
        ExperimentConfig(max_loops=0)
    with pytest.raises(ValueError, match="min_support"):
        ExperimentConfig(min_support=1)
    with pytest.raises(ValueError, match="min_support"):
        CausalRecMemPolicy(min_support=1)

    missing = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
    )
    assert missing["honest_verdict"] == "blocked_missing_exp2869_artifact"
    assert missing["fr11_scaleup_clean"] is False
    assert missing["n_examples"] == 0
    assert missing["energy_delta_by_policy"] == {}
    assert (tmp_path / "results" / OUTPUT_FILENAME).is_file()

    _write_json(
        tmp_path / "results" / "experiment_2869_fr11_continuous_self_learning_replay_v3.json",
        {"fr11_self_learning_ready": False, "honest_verdict": "blocked_prior"},
    )
    not_ready_2869 = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert not_ready_2869["honest_verdict"] == "blocked_exp2869_not_ready"

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

    _write_json(
        tmp_path / "results" / "experiment_2881_fr11_recmem_recurrence_trigger_v1.json",
        {"recmem_trigger_ready": True},
    )
    missing_2882 = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert missing_2882["honest_verdict"] == "blocked_missing_exp2882_artifact"

    tmp_path = tmp_path / "small"
    _write_ready_sources(tmp_path)
    _write_clean_corpus(tmp_path, n_per_source=2)
    small = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert small["honest_verdict"] == "blocked_target_examples_not_met"
    assert small["target_examples_met"] is False
    assert small["n_examples"] == 6
    assert "only 6 local labeled examples" in small["target_examples_note"]

    guarded = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results", min_support=99),
        write=False,
    )
    assert guarded["honest_verdict"] == "blocked_target_examples_not_met"


def test_req_learn_2887_helper_edges_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-2887-3/5: edge-case metric helpers stay deterministic."""

    assert fs_corrigendum._contradiction_rate_from_labels({}) == 0.0
    assert fs_corrigendum._duplicate_rate([]) == 0.0
    assert fs_corrigendum._memory_drift_score(0.5, 1, 0) == 0.0
    assert fs_corrigendum._token_reduction(0, 10) == 0.0
    assert fs_corrigendum._roc_auc([1, 1], [0.9, 0.8]) is None
    assert fs_corrigendum._roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert fs_corrigendum._final_energy({"energy_before": 0.7, "energy_after_each_loop": []}) == 0.7

    _write_ready_sources(tmp_path)
    _write_clean_corpus(tmp_path)
    monkeypatch.setattr(
        fs_corrigendum,
        "_adversarial_clean_checks",
        lambda *, metrics, target_examples_met: {
            "target_examples_met": target_examples_met,
            "forced_clean_gate_failure": False,
        },
    )
    clean_false = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            target_examples=50,
        ),
        write=False,
    )
    assert clean_false["honest_verdict"].startswith("complete_with_clean_gate_false")
    assert clean_false["fr11_scaleup_clean"] is False
    assert clean_false["adversarial_clean_checks"]["forced_clean_gate_failure"] is False
