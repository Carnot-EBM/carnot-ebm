"""Tests for Exp 4244 grown-pool ARC set-encoder aggregator.

Spec refs: REQ-VERIFY-4244, SCENARIO-VERIFY-4244,
SCENARIO-VERIFY-4244-NO-GAIN, SCENARIO-VERIFY-4244-DEFERRED.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from carnot.reporting import arc_set_encoder_aggregator_4244 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _features(
    *,
    vote: float,
    confidence: float,
    parity_signal: float,
    candidate_count: float = 3.0,
) -> dict[str, float]:
    return {
        name: 0.0
        for name in mod.FEATURE_NAMES
    } | {
        "vote_weight": vote,
        "self_consistency_margin": vote - 0.5,
        "vote_weight_rank_fraction": vote,
        "cell_confidence_mean": confidence,
        "cell_confidence_margin": confidence - 0.5,
        "cell_confidence_rank_fraction": confidence,
        "grid_height": 2.0,
        "grid_width": 2.0,
        "grid_cells": 4.0,
        "grid_color_count": 2.0,
        "grid_nonzero_frac": parity_signal,
        "grid_entropy": 1.0 + parity_signal,
        "program_length": 20.0 + parity_signal,
        "program_digit_fraction": 0.1,
        "program_demo_fit": confidence,
        "program_n_calls": 1.0,
        "set_candidate_count": candidate_count,
        "set_vote_mean": 1.0 / candidate_count,
        "set_vote_max": max(vote, 0.6),
        "set_vote_std": 0.2,
        "set_confidence_mean": 0.5,
        "set_confidence_max": max(confidence, 0.6),
        "set_confidence_std": 0.2,
        "set_entropy_mean": 1.0,
        "set_entropy_max": 2.0,
        "set_entropy_std": 0.2,
        "set_cells_mean": 4.0,
        "set_cells_max": 4.0,
        "set_cells_std": 0.0,
        "vote_weight_zscore": (vote - 0.33) / 0.2,
        "cell_confidence_zscore": (confidence - 0.5) / 0.2,
        "grid_entropy_zscore": parity_signal,
        "grid_cells_zscore": 0.0,
        "modal_cell_agreement_frac": parity_signal,
        "grid_duplicate_count": 1.0,
        "grid_duplicate_frac": 1.0 / candidate_count,
        "shape_family_count": candidate_count,
        "shape_family_frac": 1.0,
        "shape_vote_frac": 1.0,
        "is_modal_shape": 1.0,
        "palette_family_count": 1.0 + parity_signal,
        "palette_family_frac": (1.0 + parity_signal) / candidate_count,
        "palette_vote_frac": vote,
        "is_modal_palette": parity_signal,
        "same_shape_as_input": 1.0,
        "area_delta_from_input_frac": 0.0,
    }


def _write_grown_pool(root: Path, *, task_count: int = 12, arc_pool_grown: bool = True) -> None:
    pool_rel = Path("results/mini_4244_pool.json.gz")
    pool_path = root / pool_rel
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    for task_index in range(task_count):
        right_slot = task_index % 3
        candidates = []
        for candidate_index in range(3):
            is_correct = candidate_index == right_slot
            # The majority vote is intentionally wrong for every task. Correctness
            # alternates across slots so task-held-out leakage is easy to catch.
            vote = 0.75 if candidate_index == (right_slot + 1) % 3 else 0.15
            confidence = 0.9 if is_correct else 0.2 + 0.1 * candidate_index
            parity_signal = 1.0 if is_correct else 0.0
            candidates.append(
                {
                    "candidate_id": f"task-{task_index}::candidate{candidate_index}",
                    "candidate_index": candidate_index,
                    "features": _features(
                        vote=vote,
                        confidence=confidence,
                        parity_signal=parity_signal,
                    ),
                    "is_correct": is_correct,
                    "votes": vote * 100.0,
                }
            )
        tasks.append(
            {
                "candidate_count": 3,
                "candidates": candidates,
                "oracle_present": True,
                "raw_task_id": f"task-{task_index}",
                "source_id": "mini",
                "task_id": f"mini:task-{task_index}",
                "vote_top_candidate_id": candidates[(right_slot + 1) % 3]["candidate_id"],
                "wrong_majority": True,
            }
        )
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "candidate_n": task_count * 3,
                "positive_candidate_n": task_count,
                "random_seed": 4243,
                "reproducibility_checksum": "sha256:" + "1" * 64,
                "schema": "carnot.arc_candidate_pool_grow.v1",
                "source_paths": [],
                "source_sha256": {},
                "spec_refs": ["REQ-CAPSTONE-4243"],
                "task_n": task_count,
                "tasks": tasks,
                "wrong_majority_n": task_count,
            },
            handle,
        )
    (root / "results" / "experiment_4243_arc_candidate_pool_grow.json").write_text(
        json.dumps(
            {
                "arc_pool_grown": arc_pool_grown,
                "held_out_task_n": task_count,
                "pool_artifact_path": str(pool_rel),
                "positive_candidate_n": task_count,
                "random_seed": 4243,
                "reproducibility_checksum": "sha256:" + "2" * 64,
                "verifier_is_oracle": False,
                "wrong_majority_n": task_count,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_custom_build_and_pool(root: Path, tasks: list[dict]) -> Path:
    pool_rel = Path("results/custom_4244_pool.json.gz")
    pool_path = root / pool_rel
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    positive_n = sum(
        1
        for task in tasks
        if isinstance(task, dict)
        for candidate in task.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("is_correct") is True
    )
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "candidate_n": sum(
                    len(task.get("candidates", [])) for task in tasks if isinstance(task, dict)
                ),
                "positive_candidate_n": positive_n,
                "reproducibility_checksum": "sha256:" + "3" * 64,
                "schema": "carnot.arc_candidate_pool_grow.v1",
                "task_n": len(tasks),
                "tasks": tasks,
                "wrong_majority_n": 0,
            },
            handle,
        )
    (root / "results" / "experiment_4243_arc_candidate_pool_grow.json").write_text(
        json.dumps(
            {
                "arc_pool_grown": True,
                "held_out_task_n": len(tasks),
                "pool_artifact_path": str(pool_rel),
                "positive_candidate_n": positive_n,
                "reproducibility_checksum": "sha256:" + "4" * 64,
                "verifier_is_oracle": False,
                "wrong_majority_n": 0,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return pool_path


def test_req_4244_spec_declares_set_encoder_contract() -> None:
    """REQ-VERIFY-4244: OpenSpec declares the grown-pool set-encoder build."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4244",
        "SCENARIO-VERIFY-4244",
        "SCENARIO-VERIFY-4244-NO-GAIN",
        "SCENARIO-VERIFY-4244-DEFERRED",
        "python/carnot/reporting/arc_set_encoder_aggregator_4244.py",
        "results/experiment_4244_arc_set_encoder_aggregator_build.py",
        "complete_arc_set_encoder_deferred_no_grown_pool",
        "set_encoder_vs_logistic_auroc_delta",
        "verifier_is_oracle=false",
        "permutation-invariant set encoder",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4244_loads_grown_pool_candidate_sets(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4244: grown-pool tasks become labeled candidate sets."""

    _write_grown_pool(tmp_path, task_count=4)

    corpus = mod.load_grown_pool(tmp_path)

    assert corpus.held_out_task_n == 4
    assert corpus.wrong_majority_n == 4
    assert mod.accepted_rejected_counts(corpus.rows) == {
        "accepted": 4,
        "rejected": 8,
        "total": 12,
    }
    assert len({row.candidate_id for row in corpus.rows}) == len(corpus.rows)
    row = next(item for item in corpus.rows if item.correct)
    assert row.features["vote_weight"] == pytest.approx(0.15)
    assert row.features["cell_confidence_mean"] == pytest.approx(0.9)


def test_scenario_4244_trains_oof_set_encoder_and_logistic_ablation(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4244: set encoder and logistic ablation use same folds."""

    _write_grown_pool(tmp_path, task_count=12)

    artifact = mod.run(
        tmp_path,
        random_seed=mod.RANDOM_SEED,
        n_folds=3,
        bootstrap_n=32,
        training_epochs=18,
        hidden_dim=8,
    )

    mod.validate_artifact(artifact)
    assert artifact["aggregator_trained"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["oracle_distinct_auroc"] > 0.5
    assert artifact["held_out_task_n"] == 12
    assert artifact["wrong_majority_n"] == 12
    assert isinstance(artifact["set_encoder_vs_logistic_auroc_delta"], float)
    assert artifact["model_specs"]["architecture"] == "deepsets_pooled_context_set_encoder"
    assert artifact["model_specs"]["logistic_ablation"] == "same_fold_augmented_logistic_392"
    assert "class_weighted_bce_with_positive_task_minibatches" in artifact["model_specs"][
        "imbalance_loss"
    ]

    model_path = Path(artifact["learned_verifier_path"])
    assert model_path.exists()
    model = mod.load_set_encoder(model_path)
    assert model["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert model["verifier_is_oracle"] is False
    assert model["set_encoder_oof"]["fold_task_ids"] == model["logistic_ablation"]["fold_task_ids"]
    for row in model["set_encoder_oof"]["rows"]:
        assert row["task_id"] not in row["train_task_ids"]

    corpus = mod.load_grown_pool(tmp_path)
    task_rows = [row for row in corpus.rows if row.task_id == "mini:task-0"]
    correct = next(row for row in task_rows if row.correct)
    wrong = next(row for row in task_rows if not row.correct)
    assert mod.score_with_set_encoder(model, correct, task_rows) > mod.score_with_set_encoder(
        model, wrong, task_rows
    )
    assert mod.score_with_set_encoder(model, correct, task_rows) == pytest.approx(
        mod.score_with_set_encoder(model, correct, list(reversed(task_rows)))
    )


def test_scenario_4244_no_gain_still_persists_model(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4244-NO-GAIN: same-fold logistic null still feeds A3."""

    _write_grown_pool(tmp_path, task_count=9)

    artifact = mod.run(
        tmp_path,
        random_seed=mod.RANDOM_SEED,
        n_folds=3,
        bootstrap_n=16,
        training_epochs=8,
        hidden_dim=8,
        baseline_392_high=1.1,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith(
        "complete_arc_set_encoder_no_gain_over_logistic_auroc"
    )
    assert artifact["aggregator_trained"] is True
    assert Path(artifact["learned_verifier_path"]).exists()


def test_scenario_4244_deferred_without_grown_pool(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4244-DEFERRED: missing or false A1 precondition stops training."""

    artifact = mod.run(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete_arc_set_encoder_deferred_no_grown_pool"
    assert artifact["aggregator_trained"] is False
    assert artifact["oracle_distinct_auroc"] == 0.0
    assert artifact["set_encoder_vs_logistic_auroc_delta"] == 0.0
    assert artifact["learned_verifier_path"] == ""
    assert artifact["verifier_is_oracle"] is False

    _write_grown_pool(tmp_path, task_count=3, arc_pool_grown=False)
    false_artifact = mod.run(tmp_path)
    assert false_artifact["honest_verdict"] == "complete_arc_set_encoder_deferred_no_grown_pool"


def test_req_4244_validation_rejects_wrapped_gate_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4244: A3 gate fields must stay bare."""

    _write_grown_pool(tmp_path, task_count=9)
    artifact = mod.run(
        tmp_path,
        n_folds=3,
        bootstrap_n=8,
        training_epochs=6,
        hidden_dim=8,
    )

    invalid_cases = [
        ({key: value for key, value in artifact.items() if key != "aggregator_trained"}, "missing"),
        ({**artifact, "honest_verdict": "done"}, "terminal-prefixed"),
        ({**artifact, "aggregator_trained": {"value": True}}, "bare bool"),
        ({**artifact, "oracle_distinct_auroc": {"value": 0.5}}, "bare float"),
        ({**artifact, "set_encoder_vs_logistic_auroc_delta": None}, "bare float"),
        ({**artifact, "wrong_majority_n": {"value": 9}}, "bare int"),
        ({**artifact, "held_out_task_n": {"value": 9}}, "bare int"),
        ({**artifact, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**artifact, "learned_verifier_path": "results/missing.json"}, "persisted"),
        ({**artifact, "field_principles": {}}, "field_principles"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)


def test_scenario_4244_deferred_and_constant_edge_paths(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4244-DEFERRED: malformed pools and sparse labels stay honest."""

    bad_json_root = tmp_path / "bad-json"
    (bad_json_root / "results").mkdir(parents=True)
    (bad_json_root / "results" / "experiment_4243_arc_candidate_pool_grow.json").write_text(
        "{bad",
        encoding="utf-8",
    )
    assert mod.run(bad_json_root)["honest_verdict"] == (
        "complete_arc_set_encoder_deferred_no_grown_pool"
    )

    list_json_root = tmp_path / "list-json"
    (list_json_root / "results").mkdir(parents=True)
    (list_json_root / "results" / "experiment_4243_arc_candidate_pool_grow.json").write_text(
        "[]",
        encoding="utf-8",
    )
    assert mod.run(list_json_root)["honest_verdict"] == (
        "complete_arc_set_encoder_deferred_no_grown_pool"
    )

    missing_path_root = tmp_path / "missing-path"
    (missing_path_root / "results").mkdir(parents=True)
    (missing_path_root / "results" / "experiment_4243_arc_candidate_pool_grow.json").write_text(
        json.dumps({"arc_pool_grown": True}),
        encoding="utf-8",
    )
    assert mod.run(missing_path_root)["honest_verdict"] == (
        "complete_arc_set_encoder_deferred_no_grown_pool"
    )
    (missing_path_root / "results" / "experiment_4243_arc_candidate_pool_grow.json").write_text(
        json.dumps({"arc_pool_grown": True, "pool_artifact_path": "results/missing.json.gz"}),
        encoding="utf-8",
    )
    assert mod.run(missing_path_root)["honest_verdict"] == (
        "complete_arc_set_encoder_deferred_no_grown_pool"
    )

    bad_gzip_root = tmp_path / "bad-gzip"
    bad_pool = bad_gzip_root / "results" / "pool.json.gz"
    bad_pool.parent.mkdir(parents=True)
    bad_pool.write_text("not gzip", encoding="utf-8")
    (bad_gzip_root / "results" / "experiment_4243_arc_candidate_pool_grow.json").write_text(
        json.dumps({"arc_pool_grown": True, "pool_artifact_path": "results/pool.json.gz"}),
        encoding="utf-8",
    )
    assert mod.run(bad_gzip_root)["honest_verdict"] == (
        "complete_arc_set_encoder_deferred_no_grown_pool"
    )

    bad_schema_root = tmp_path / "bad-schema"
    schema_pool = bad_schema_root / "results" / "pool.json.gz"
    schema_pool.parent.mkdir(parents=True)
    with gzip.open(schema_pool, "wt", encoding="utf-8") as handle:
        json.dump({"tasks": {}}, handle)
    (bad_schema_root / "results" / "experiment_4243_arc_candidate_pool_grow.json").write_text(
        json.dumps({"arc_pool_grown": True, "pool_artifact_path": "results/pool.json.gz"}),
        encoding="utf-8",
    )
    assert mod.run(bad_schema_root)["honest_verdict"] == (
        "complete_arc_set_encoder_deferred_no_grown_pool"
    )

    malformed_root = tmp_path / "malformed"
    _write_custom_build_and_pool(
        malformed_root,
        [
            None,
            {"task_id": "", "candidates": []},
            {
                "task_id": "mini:edge",
                "candidates": [
                    None,
                    {"candidate_index": 0, "features": "bad", "is_correct": True},
                    {
                        "candidate_id": "mini:edge::wrong",
                        "candidate_index": 1,
                        "features": _features(vote=0.8, confidence=0.1, parity_signal=0.0),
                        "is_correct": False,
                    },
                ],
            },
        ],
    )
    corpus = mod.load_grown_pool(malformed_root)
    assert len(corpus.rows) == 2
    assert corpus.rows[0].candidate_id == "mini:edge::candidate1"
    assert corpus.rows[0].features["vote_weight"] == 0.0

    empty_root = tmp_path / "empty"
    _write_custom_build_and_pool(empty_root, [])
    assert mod.run(empty_root)["honest_verdict"] == (
        "complete_arc_set_encoder_deferred_no_grown_pool"
    )

    no_positive_root = tmp_path / "no-positive"
    negative_tasks = []
    for task_index in range(4):
        negative_tasks.append(
            {
                "task_id": f"mini:negative-{task_index}",
                "candidates": [
                    {
                        "candidate_id": f"mini:negative-{task_index}::candidate0",
                        "candidate_index": 0,
                        "features": _features(vote=0.7, confidence=0.4, parity_signal=0.0),
                        "is_correct": False,
                    },
                    {
                        "candidate_id": f"mini:negative-{task_index}::candidate1",
                        "candidate_index": 1,
                        "features": _features(vote=0.3, confidence=0.2, parity_signal=0.0),
                        "is_correct": False,
                    },
                ],
            }
        )
    _write_custom_build_and_pool(no_positive_root, negative_tasks)
    no_positive = mod.run(
        no_positive_root,
        n_folds=2,
        bootstrap_n=0,
        training_epochs=1,
        hidden_dim=4,
    )
    assert no_positive["aggregator_trained"] is True
    assert no_positive["oracle_distinct_auroc"] == 0.0
    no_positive_model = mod.load_set_encoder(no_positive["learned_verifier_path"])
    no_positive_corpus = mod.load_grown_pool(no_positive_root)
    assert mod.score_with_set_encoder(
        no_positive_model,
        no_positive_corpus.rows[0],
        no_positive_corpus.rows[:2],
    ) == 0.0

    assert mod._as_float(True) == 0.0
    assert mod._as_float("x") == 0.0
    assert mod._standardizer([])[1][0] == 1.0
    assert mod._fit_temperature([0.1], [True]) == 1.0
    assert mod._fit_isotonic([0.1], [True]) == {"x": [0.0, 1.0], "y": [1.0, 1.0]}
    assert mod._apply_isotonic(0.4, {}) == 0.4
    assert mod._apply_isotonic(0.5, {"x": [0.0, 1.0], "y": [0.0, 1.0]}) == 0.5
    assert mod._sigmoid(-1.0) < 0.5
    assert mod._bootstrap_auroc_ci95([True, False], [0.2, 0.1], 1, bootstrap_n=0) == (1.0, 1.0)
    assert mod._bootstrap_auroc_ci95([True, False], [0.2, 0.1], 1, bootstrap_n=4)[0] >= 0.0
    assert mod._no_gain_reason(0.83, 0.5) == "no_gain_over_392_augmented_logistic_range"
    assert mod._no_gain_reason(0.9, 0.5) is None

    bad_model_path = tmp_path / "bad_model.json"
    bad_model_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod.load_set_encoder(bad_model_path)

    success_artifact = mod._complete_artifact(
        corpus,
        mod.OOFModelReport(0.91, (0.8, 0.95), [["mini:edge"]], [], {}),
        mod.OOFModelReport(0.5, (0.4, 0.6), [["mini:edge"]], [], {}),
        checksum="sha256:" + "5" * 64,
        counts=mod.accepted_rejected_counts(corpus.rows),
        model_path=tmp_path / "placeholder.json",
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
        no_gain_reason=None,
        hidden_dim=4,
        training_epochs=1,
    )
    assert success_artifact["honest_verdict"].startswith("complete: arc_set_encoder")
