"""Tests for Exp 2869 FR-11 continuous self-learning offline replay.

Spec: REQ-LEARN-2869,
      SCENARIO-LEARN-2869,
      SCENARIO-LEARN-2869-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval.fr11_continuous_self_learning_replay_v3 import (
    BACKEND_MODULE_PATH,
    MEMORY_STATE_FILENAME,
    OUTPUT_FILENAME,
    REQUIRED_ARTIFACT_FIELDS,
    ExperimentConfig,
    ReplayExample,
    build_backend_rows,
    run_experiment,
    summarize_trace_metrics,
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


def _write_success_inputs(repo_root: Path) -> None:
    _write_json(
        repo_root / "results" / "experiment_2868_offline_recurrence_backend_adapter_v2.json",
        {
            "backend_module_path": BACKEND_MODULE_PATH,
            "live_model_invoked": False,
            "no_model_weight_mutation": True,
            "offline_recurrence_backend_ready": True,
        },
    )
    _write_json(
        repo_root / "results" / "experiment_2865_cross_corpus_matrix_v5.json",
        {
            "paper_eligible_rows": ["FoVer", "HaluEval/FEVER"],
            "row_status_by_corpus": {
                "FoVer": "clean",
                "HaluEval/FEVER": "clean",
                "HumanEval": "missing",
            },
        },
    )
    _write_json(
        repo_root / "results" / "experiment_2850_fover_dual_condition_integrity_v4.json",
        {
            "fr11_state_files": [
                {"path": "data/constraint_memory.db"},
                {"path": "data/constraint_memory.db"},
            ]
        },
    )
    (repo_root / "data").mkdir(parents=True)
    (repo_root / "data" / "constraint_memory.db").write_bytes(b"baseline-memory")
    _write_jsonl(
        repo_root / "data" / "fover_corpus.jsonl",
        [
            {
                "confidence": 1.0,
                "label": "incorrect",
                "question_id": "fover-bad",
                "source": "fover_v4",
                "verifier": "heuristic",
            },
            {
                "confidence": 1.0,
                "label": "correct",
                "question_id": "fover-good",
                "source": "fover_v4",
                "verifier": "heuristic",
            },
            {
                "confidence": 1.0,
                "label": "not-a-binary-label",
                "question_id": "fover-ignored",
                "source": "fover_v4",
                "verifier": "heuristic",
            },
        ],
    )
    _write_jsonl(
        repo_root / "data" / "eval_manifests" / "halueval_20260522.jsonl",
        [
            {
                "candidate": "right",
                "dataset": "HaluEval",
                "label": 0,
                "prompt": "p",
                "stable_id": "halueval-good",
            },
            {
                "candidate": "wrong",
                "dataset": "HaluEval",
                "label": 1,
                "prompt": "p",
                "stable_id": "halueval-bad",
            },
            {
                "candidate": "ignored",
                "dataset": "HaluEval",
                "label": True,
                "prompt": "p",
                "stable_id": "halueval-ignored",
            },
            {
                "candidate": "ignored",
                "dataset": "HaluEval",
                "label": "not-binary",
                "prompt": "p",
                "stable_id": "halueval-ignored-text",
            },
        ],
    )
    _write_jsonl(
        repo_root / "data" / "eval_manifests" / "fever_20260522.jsonl",
        [
            {
                "claim": "supported",
                "dataset": "FEVER",
                "label": 0,
                "prompt": "p",
                "stable_id": "fever-good",
            },
            {
                "claim": "refuted",
                "dataset": "FEVER",
                "label": "1",
                "prompt": "p",
                "stable_id": "fever-bad",
            },
        ],
    )


def test_scenario_learn_2869_replay_updates_permitted_memory_only(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2869: offline verifier feedback lowers energy and updates memory."""

    _write_success_inputs(tmp_path)

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            replay_n_examples=9,
            started_at=1.0,
            clock=lambda: 2.25,
        )
    )

    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["fr11_self_learning_ready"] is True
    assert artifact["offline_recurrence_backend_used"] == BACKEND_MODULE_PATH
    assert artifact["live_model_invoked"] is False
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["n_examples"] == 6
    assert artifact["max_loops"] == 3
    assert artifact["source_counts"] == {"FEVER": 2, "FoVer": 2, "HaluEval": 2}
    assert artifact["recurrence_success_rate"] == pytest.approx(0.5)
    assert artifact["energy_delta_mean"] > 0.0
    assert artifact["correctness_delta"] == 0.0
    assert artifact["forgetting_regression_count"] == 0
    assert artifact["memory_hash_before"] != artifact["memory_hash_after"]
    assert artifact["per_loop_energy_summary"]["loop_1"]["n"] == 6
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == pytest.approx(1.25)

    checks = {str(check["check"]): check for check in artifact["preconditions_checked"]}
    assert checks["offline_backend_imported"]["observed"] == BACKEND_MODULE_PATH
    assert checks["clean_replay_corpus"]["observed"] == 6
    assert checks["permitted_memory_changes"]["passed"] is True
    assert checks["permitted_memory_changes"]["observed"] == [f"results/{MEMORY_STATE_FILENAME}"]

    memory_state = json.loads(
        (tmp_path / "results" / MEMORY_STATE_FILENAME).read_text(encoding="utf-8")
    )
    assert memory_state["n_examples"] == 6
    assert memory_state["source_counts"] == artifact["source_counts"]
    assert memory_state["no_model_weight_mutation"] is True

    saved = json.loads((tmp_path / "results" / OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact

    one_row_artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            replay_n_examples=1,
            random_seed=2868,
            memory_state_path=tmp_path / "results" / "one-row-memory.json",
        ),
        write=False,
    )
    assert one_row_artifact["n_examples"] == 1
    assert one_row_artifact["source_counts"] == {"FEVER": 1}


def test_scenario_learn_2869_blocked_backend_preconditions_do_not_write_memory(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2869-BLOCKED: missing or unimportable backend blocks metrics."""

    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
    )

    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"] == "blocked_missing_exp2868_artifact"
    assert artifact["fr11_self_learning_ready"] is False
    assert artifact["offline_recurrence_backend_used"] == "unavailable"
    assert artifact["n_examples"] == 0
    assert artifact["recurrence_success_rate"] == 0.0
    assert artifact["energy_delta_mean"] == 0.0
    assert artifact["correctness_delta"] == 0.0
    assert artifact["forgetting_regression_count"] == 0
    assert artifact["memory_hash_before"] == "not_checked_precondition_failed"
    assert artifact["memory_hash_after"] == "not_checked_precondition_failed"
    assert artifact["live_model_invoked"] is False
    assert artifact["no_model_weight_mutation"] is True
    assert not (tmp_path / "results" / MEMORY_STATE_FILENAME).exists()
    assert (tmp_path / "results" / OUTPUT_FILENAME).is_file()

    _write_json(
        tmp_path / "results" / "experiment_2868_offline_recurrence_backend_adapter_v2.json",
        {"backend_module_path": "carnot.eval.not_a_real_backend"},
    )
    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_offline_backend_import"

    _write_json(
        tmp_path / "results" / "experiment_2868_offline_recurrence_backend_adapter_v2.json",
        {"backend_module_path": ""},
    )
    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_offline_backend_module_path"


def test_req_learn_2869_blocked_clean_corpus_preconditions(tmp_path: Path) -> None:
    """REQ-LEARN-2869-2: missing, non-clean, and empty clean corpora fail closed."""

    _write_json(
        tmp_path / "results" / "experiment_2868_offline_recurrence_backend_adapter_v2.json",
        {"backend_module_path": BACKEND_MODULE_PATH},
    )

    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_missing_exp2865_artifact"

    _write_json(
        tmp_path / "results" / "experiment_2865_cross_corpus_matrix_v5.json",
        {"row_status_by_corpus": {"FoVer": "missing"}},
    )
    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_no_clean_corpus_rows"

    _write_json(
        tmp_path / "results" / "experiment_2865_cross_corpus_matrix_v5.json",
        {"paper_eligible_rows": ["FoVer"]},
    )
    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_empty_replay_corpus"


def test_req_learn_2869_trace_metrics_count_forgetting_regression() -> None:
    """REQ-LEARN-2869-4: trace metrics expose energy, correctness, and regression."""

    metrics = summarize_trace_metrics(
        [
            {
                "energy_before": 1.0,
                "energy_after_each_loop": [1.2],
                "correctness_before": False,
                "correctness_after": False,
            },
            {
                "energy_before": 0.6,
                "energy_after_each_loop": [0.2],
                "correctness_before": False,
                "correctness_after": True,
            },
        ]
    )

    assert metrics["energy_delta_mean"] == pytest.approx(0.1)
    assert metrics["correctness_delta"] == pytest.approx(0.5)
    assert metrics["recurrence_success_rate"] == pytest.approx(0.5)
    assert metrics["forgetting_regression_count"] == 1


def test_req_learn_2869_config_guards_and_early_exit_rows() -> None:
    """REQ-LEARN-2869-4: config guards and low-energy convergence are explicit."""

    with pytest.raises(ValueError, match="max_loops"):
        ExperimentConfig(max_loops=0)
    with pytest.raises(ValueError, match="replay_n_examples"):
        ExperimentConfig(replay_n_examples=0)

    rows = build_backend_rows(
        [
            ReplayExample(
                example_id="low-energy",
                source="FoVer",
                energy_before=0.2,
                correctness_before=False,
                localized_violations=("heuristic",),
            )
        ],
        max_loops=3,
    )

    assert rows[0]["energy_after_each_loop"] == [0.0]
    assert rows[0]["early_exit_reason"] == "offline_energy_converged"
