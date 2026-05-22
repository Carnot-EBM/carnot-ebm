"""Tests for Exp 2881 FR-11 RecMem recurrence-triggered consolidation.

Spec: REQ-LEARN-2881,
      SCENARIO-LEARN-2881,
      SCENARIO-LEARN-2881-GUARD.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.eval.fr11_recmem_recurrence_trigger_v1 as recmem
from carnot.eval.fr11_continuous_self_learning_replay_v3 import (
    BACKEND_MODULE_PATH,
)
from carnot.eval.fr11_recmem_recurrence_trigger_v1 import (
    MEMORY_STATE_FILENAME,
    OUTPUT_FILENAME,
    REQUIRED_ARTIFACT_FIELDS,
    ExperimentConfig,
    RecurrenceCluster,
    RecurrenceDetector,
    ReplayEvent,
    SubconsciousRecurrenceBuffer,
    evaluate_recmem_events,
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


def _event(
    event_id: str,
    source: str,
    violation: str,
    *,
    before: float = 0.9,
    after: float = 0.45,
    correctness_before: bool = False,
    correctness_after: bool = False,
) -> ReplayEvent:
    return ReplayEvent(
        event_id=event_id,
        source=source,
        energy_before=before,
        energy_after=after,
        correctness_before=correctness_before,
        correctness_after=correctness_after,
        localized_violations=(violation,) if violation else (),
        early_exit_reason="offline_energy_replayed"
        if violation
        else "offline_no_energy_improvement",
    )


def _write_success_inputs(repo_root: Path) -> None:
    _write_json(
        repo_root / "results" / "experiment_2868_offline_recurrence_backend_adapter_v2.json",
        {
            "backend_module_path": BACKEND_MODULE_PATH,
            "offline_recurrence_backend_ready": True,
            "live_model_invoked": False,
            "per_example_trace": [
                {
                    "correctness_after": False,
                    "correctness_before": False,
                    "early_exit_reason": "offline_energy_replayed",
                    "energy_after_each_loop": [0.55],
                    "energy_before": 0.9,
                    "example_id": "offline-a",
                    "localized_violations": ["factuality_mismatch"],
                    "source": "HaluEval",
                },
                {
                    "correctness_after": False,
                    "correctness_before": False,
                    "early_exit_reason": "offline_energy_replayed",
                    "energy_after_each_loop": [0.5],
                    "energy_before": 0.85,
                    "example_id": "offline-b",
                    "localized_violations": ["factuality_mismatch"],
                    "source": "FEVER",
                },
            ],
        },
    )
    _write_json(
        repo_root / "results" / "experiment_2869_fr11_continuous_self_learning_replay_v3.json",
        {
            "continuous_self_learning_task": True,
            "fr11_self_learning_ready": True,
            "live_model_invoked": False,
            "memory_hash_after": "prior-hash",
            "offline_recurrence_backend_used": BACKEND_MODULE_PATH,
        },
    )
    _write_json(
        repo_root / "results" / "experiment_2865_cross_corpus_matrix_v5.json",
        {
            "paper_eligible_rows": ["FoVer", "HaluEval/FEVER"],
            "row_status_by_corpus": {"FoVer": "clean", "HaluEval/FEVER": "clean"},
        },
    )
    _write_json(
        repo_root / "results" / "fr11_continuous_self_learning_replay_2869_memory.json",
        {
            "artifact": "fr11_continuous_self_learning_replay_2869_memory",
            "constraint_summaries": {
                "FEVER::factuality_mismatch": {"observations": 1},
                "HaluEval::factuality_mismatch": {"observations": 1},
            },
        },
    )
    _write_jsonl(
        repo_root / "data" / "fover_corpus.jsonl",
        [
            {
                "confidence": 1.0,
                "label": "correct",
                "question_id": "fover-good",
                "source": "fover_v4",
                "verifier": "heuristic",
            }
        ],
    )
    _write_jsonl(
        repo_root / "data" / "eval_manifests" / "halueval_20260522.jsonl",
        [
            {
                "candidate": "wrong",
                "dataset": "HaluEval",
                "label": 1,
                "prompt": "p",
                "stable_id": "halueval-bad",
            }
        ],
    )
    _write_jsonl(
        repo_root / "data" / "eval_manifests" / "fever_20260522.jsonl",
        [
            {
                "claim": "refuted",
                "dataset": "FEVER",
                "label": "1",
                "prompt": "p",
                "stable_id": "fever-bad",
            }
        ],
    )


def test_scenario_learn_2881_recurring_motifs_trigger_consolidation() -> None:
    """SCENARIO-LEARN-2881: repeated failure motifs consolidate after recurrence."""

    events = [
        _event("halueval-1", "HaluEval", "factuality_mismatch"),
        _event("fever-1", "FEVER", "factuality_mismatch", before=0.85, after=0.5),
        _event(
            "fover-clean",
            "FoVer",
            "",
            before=0.0,
            after=0.0,
            correctness_before=True,
            correctness_after=True,
        ),
    ]
    buffer = SubconsciousRecurrenceBuffer()
    buffer.ingest_many(events)

    detector = RecurrenceDetector(recurrence_threshold=0.6, min_support=2)
    clusters = detector.detect(buffer.events)
    ready_clusters = [cluster for cluster in clusters if cluster.consolidation_ready]
    report = evaluate_recmem_events(buffer.events, recurrence_threshold=0.6, min_support=2)

    assert buffer.n_events == 3
    assert len(ready_clusters) == 1
    assert ready_clusters[0].motif_key == "factuality_mismatch"
    assert sorted(ready_clusters[0].event_ids) == ["fever-1", "halueval-1"]
    assert report["recmem_trigger_ready"] is True
    assert report["n_recurrence_clusters"] == 1
    assert report["n_consolidations_triggered"] == 1
    assert report["eager_consolidations_avoided"] == 2
    assert report["token_reduction_proxy_pct"] > 0.0
    assert report["live_llm_called"] is False


def test_scenario_learn_2881_guard_blocks_contradiction_and_forgetting() -> None:
    """SCENARIO-LEARN-2881-GUARD: unsafe recurring motifs fail closed."""

    events = [
        _event("same-1", "HaluEval", "factuality_mismatch", correctness_after=False),
        _event("same-2", "FEVER", "factuality_mismatch", correctness_after=True),
        _event("same-3", "FEVER", "factuality_mismatch", before=0.3, after=0.6),
    ]

    report = evaluate_recmem_events(events, recurrence_threshold=0.6, min_support=2)

    assert report["recmem_trigger_ready"] is False
    assert report["n_recurrence_clusters"] == 1
    assert report["contradiction_rate"] > 0.0
    assert report["forgetting_regression_count"] == 1


def test_req_learn_2881_run_experiment_writes_artifact_and_memory(tmp_path: Path) -> None:
    """REQ-LEARN-2881-3/5: experiment writes schema and memory hash changes."""

    _write_success_inputs(tmp_path)

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=1.0,
            clock=lambda: 2.5,
        ),
        tests_run=["unit-test-placeholder"],
    )

    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["recmem_trigger_ready"] is True
    assert artifact["source_artifacts"] == [
        "results/experiment_2868_offline_recurrence_backend_adapter_v2.json",
        "results/experiment_2869_fr11_continuous_self_learning_replay_v3.json",
        "results/fr11_continuous_self_learning_replay_2869_memory.json",
    ]
    assert artifact["recurrence_threshold"] == pytest.approx(0.6)
    assert artifact["n_events_ingested"] >= 4
    assert artifact["n_consolidations_triggered"] >= 1
    assert artifact["eager_consolidations_avoided"] > 0
    assert artifact["token_reduction_proxy_pct"] > 0.0
    assert artifact["memory_hash_before"] != artifact["memory_hash_after"]
    assert artifact["contradiction_rate"] == 0.0
    assert artifact["forgetting_regression_count"] == 0
    assert artifact["live_llm_called"] is False
    assert artifact["tests_run"] == ["unit-test-placeholder"]
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == pytest.approx(1.5)

    memory_state = json.loads(
        (tmp_path / "results" / MEMORY_STATE_FILENAME).read_text(encoding="utf-8")
    )
    assert memory_state["schema"] == "carnot.fr11.recmem_recurrence_memory.v1"
    assert memory_state["consolidations"]

    saved = json.loads((tmp_path / "results" / OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact


def test_req_learn_2881_guard_branches_and_empty_metrics(tmp_path: Path) -> None:
    """REQ-LEARN-2881-1/2/4: local guard branches are deterministic."""

    with pytest.raises(ValueError, match="recurrence_threshold"):
        ExperimentConfig(recurrence_threshold=1.5)
    with pytest.raises(ValueError, match="min_support"):
        ExperimentConfig(min_support=1)
    with pytest.raises(ValueError, match="max_loops"):
        ExperimentConfig(max_loops=0)
    with pytest.raises(ValueError, match="replay_n_examples"):
        ExperimentConfig(replay_n_examples=0)
    with pytest.raises(ValueError, match="recurrence_threshold"):
        RecurrenceDetector(recurrence_threshold=-0.1)
    with pytest.raises(ValueError, match="min_support"):
        RecurrenceDetector(recurrence_threshold=0.6, min_support=1)

    unlocalized = _event("unlocalized", "FoVer", "", correctness_before=False)
    clean = _event(
        "clean",
        "FoVer",
        "",
        before=0.0,
        after=0.0,
        correctness_before=True,
        correctness_after=True,
    )
    single_cluster = RecurrenceCluster(
        cluster_id="single",
        events=(clean,),
        recurrence_threshold=0.6,
        min_support=2,
    )
    empty_cluster = RecurrenceCluster(
        cluster_id="empty",
        events=(),
        recurrence_threshold=0.6,
        min_support=2,
    )

    assert unlocalized.motif_key == "unlocalized_failure"
    assert "unlocalized_failure" in unlocalized.feature_tokens
    assert "clean" in clean.feature_tokens
    assert single_cluster.motif_key == "clean"
    assert single_cluster.similarity_floor == 1.0
    assert empty_cluster.motif_key == "empty"
    assert recmem._jaccard(frozenset(), frozenset()) == 1.0

    empty_report = evaluate_recmem_events([], recurrence_threshold=0.6, min_support=2)
    assert empty_report["token_reduction_proxy_pct"] == 0.0
    assert empty_report["duplicate_rate"] == 0.0
    assert empty_report["contradiction_rate"] == 0.0

    duplicate_path = tmp_path / "memory.json"
    memory_hash, manifest = recmem._hash_memory_state(
        [duplicate_path, duplicate_path],
        tmp_path,
    )
    assert len(memory_hash) == 64
    assert len(manifest) == 1


def test_req_learn_2881_blocked_paths_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-2881-5: each source/precondition failure has a stable verdict."""

    write_blocked = run_experiment(ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path))
    assert write_blocked["honest_verdict"] == "blocked_missing_exp2868_artifact"
    assert (tmp_path / OUTPUT_FILENAME).is_file()

    repo_missing_path = tmp_path / "missing-path"
    _write_json(
        repo_missing_path
        / "results"
        / "experiment_2868_offline_recurrence_backend_adapter_v2.json",
        {"backend_module_path": "carnot.eval.not_the_backend"},
    )
    path_blocked = run_experiment(
        ExperimentConfig(repo_root=repo_missing_path, results_dir=repo_missing_path / "results"),
        write=False,
    )
    assert path_blocked["honest_verdict"] == "blocked_offline_backend_module_path"

    repo_import = tmp_path / "import"
    _write_success_inputs(repo_import)

    def _raise_import_error(_module_path: str) -> object:
        raise ModuleNotFoundError("forced")

    monkeypatch.setattr(recmem.importlib, "import_module", _raise_import_error)
    import_blocked = run_experiment(
        ExperimentConfig(repo_root=repo_import, results_dir=repo_import / "results"),
        write=False,
    )
    assert import_blocked["honest_verdict"] == "blocked_offline_backend_import"
    monkeypatch.undo()

    repo_no_2869 = tmp_path / "no-2869"
    _write_json(
        repo_no_2869 / "results" / "experiment_2868_offline_recurrence_backend_adapter_v2.json",
        {"backend_module_path": BACKEND_MODULE_PATH},
    )
    missing_2869 = run_experiment(
        ExperimentConfig(repo_root=repo_no_2869, results_dir=repo_no_2869 / "results"),
        write=False,
    )
    assert missing_2869["honest_verdict"] == "blocked_missing_exp2869_artifact"

    repo_not_ready = tmp_path / "not-ready"
    _write_success_inputs(repo_not_ready)
    _write_json(
        repo_not_ready / "results" / "experiment_2869_fr11_continuous_self_learning_replay_v3.json",
        {"fr11_self_learning_ready": False},
    )
    not_ready = run_experiment(
        ExperimentConfig(repo_root=repo_not_ready, results_dir=repo_not_ready / "results"),
        write=False,
    )
    assert not_ready["honest_verdict"] == "blocked_exp2869_replay_not_ready"

    repo_no_memory = tmp_path / "no-memory"
    _write_success_inputs(repo_no_memory)
    (repo_no_memory / "results" / "fr11_continuous_self_learning_replay_2869_memory.json").unlink()
    missing_memory = run_experiment(
        ExperimentConfig(repo_root=repo_no_memory, results_dir=repo_no_memory / "results"),
        write=False,
    )
    assert missing_memory["honest_verdict"] == "blocked_missing_exp2869_memory"

    repo_no_events = tmp_path / "no-events"
    _write_json(
        repo_no_events / "results" / "experiment_2868_offline_recurrence_backend_adapter_v2.json",
        {"backend_module_path": BACKEND_MODULE_PATH, "per_example_trace": "not-a-list"},
    )
    _write_json(
        repo_no_events / "results" / "experiment_2869_fr11_continuous_self_learning_replay_v3.json",
        {"fr11_self_learning_ready": True},
    )
    _write_json(
        repo_no_events / "results" / "fr11_continuous_self_learning_replay_2869_memory.json",
        {"artifact": "prior"},
    )
    no_events = run_experiment(
        ExperimentConfig(repo_root=repo_no_events, results_dir=repo_no_events / "results"),
        write=False,
    )
    assert no_events["honest_verdict"] == "blocked_no_offline_events"


def test_req_learn_2881_missing_sources_block_without_live_llm(tmp_path: Path) -> None:
    """REQ-LEARN-2881-5: missing source artifacts block without fake metrics."""

    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_missing_exp2868_artifact"
    assert artifact["recmem_trigger_ready"] is False
    assert artifact["n_events_ingested"] == 0
    assert artifact["n_consolidations_triggered"] == 0
    assert artifact["eager_consolidations_avoided"] == 0
    assert artifact["token_reduction_proxy_pct"] == 0.0
    assert artifact["memory_hash_before"] == "not_checked_precondition_failed"
    assert artifact["memory_hash_after"] == "not_checked_precondition_failed"
    assert artifact["live_llm_called"] is False
