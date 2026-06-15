"""Tests for Exp 4258 ARC oracle-distinct cross-game transfer.

Spec refs: REQ-VERIFY-4258, SCENARIO-VERIFY-4258.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_oracle_distinct_cross_game_transfer_4258 as mod
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {"returncode": 0, "reports": [{"flag_count": 0, "flags": [], "max_severity": 0}]}


def _adversarial_flagged(_path: Path) -> dict[str, Any]:
    return {
        "returncode": 1,
        "reports": [
            {
                "flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}],
                "max_severity": 2,
            }
        ],
    }


def _features(vote: float, confidence: float) -> dict[str, float]:
    values = {name: 0.0 for name in exp4244.FEATURE_NAMES}
    values.update(
        {
            "vote_weight": float(vote),
            "self_consistency_margin": float(vote) - 0.5,
            "vote_weight_rank_fraction": float(vote),
            "cell_confidence_mean": float(confidence),
            "cell_confidence_margin": float(confidence) - 0.5,
            "cell_confidence_rank_fraction": float(confidence),
            "grid_height": 2.0,
            "grid_width": 2.0,
            "grid_cells": 4.0,
            "set_candidate_count": 2.0,
            "set_vote_mean": 0.5,
            "set_vote_max": float(vote),
        }
    )
    return values


def _write_cross_game_fixture(
    root: Path,
    *,
    include_game_ids: bool = True,
    vote_all_correct: bool = False,
) -> dict[str, float]:
    task_specs = [
        ("game-a", "task-0", 1, [0.9, 0.1], [0.1, 0.9]),
        ("game-a", "task-1", 1, [0.9, 0.1], [0.1, 0.9]),
        ("game-b", "task-2", 1, [0.9, 0.1], [0.1, 0.9]),
        ("game-b", "task-3", 0, [0.9, 0.1], [0.9, 0.1]),
    ]
    if vote_all_correct:
        task_specs = [
            (game_id, raw_task_id, 0, votes, [0.9, 0.1])
            for game_id, raw_task_id, _correct_index, votes, _scores in task_specs
        ]
    score_by_candidate: dict[str, float] = {}
    tasks: list[dict[str, Any]] = []
    for game_id, raw_task_id, correct_index, votes, scores in task_specs:
        task_id = f"{game_id}:{raw_task_id}" if include_game_ids else f"fixture:{raw_task_id}"
        candidates = []
        for candidate_index, (vote, score) in enumerate(zip(votes, scores, strict=True)):
            candidate_id = f"{task_id}::candidate{candidate_index}"
            is_correct = candidate_index == correct_index
            score_by_candidate[candidate_id] = score
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "features": _features(vote, confidence=score),
                    "grid": [[candidate_index]],
                    "is_correct": is_correct,
                    "source_kinds": ["gold_flag"] if is_correct else ["pool_candidate"],
                    "votes": vote,
                }
            )
        task = {
            "candidate_count": len(candidates),
            "candidates": candidates,
            "oracle_present": True,
            "raw_task_id": raw_task_id,
            "source_id": "fixture",
            "task_id": task_id,
            "vote_top_candidate_id": f"{task_id}::candidate0",
            "wrong_majority": correct_index != 0,
        }
        if include_game_ids:
            task["game_id"] = game_id
        tasks.append(task)

    pool_rel = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
    pool_path = root / pool_rel
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "candidate_n": sum(len(task["candidates"]) for task in tasks),
                "positive_candidate_n": len(tasks),
                "random_seed": 4243,
                "reproducibility_checksum": "sha256:" + "1" * 64,
                "schema": "carnot.arc_candidate_pool_grow.v1",
                "source_paths": [],
                "source_sha256": {},
                "spec_refs": ["REQ-CAPSTONE-4243"],
                "task_n": len(tasks),
                "tasks": tasks,
                "wrong_majority_n": sum(int(task["wrong_majority"]) for task in tasks),
            },
            handle,
        )
    _write_json(
        root / "results" / "experiment_4243_arc_candidate_pool_grow.json",
        {
            "arc_pool_grown": True,
            "held_out_task_n": len(tasks),
            "pool_artifact_path": str(pool_rel),
            "positive_candidate_n": len(tasks),
            "random_seed": 4243,
            "reproducibility_checksum": "sha256:" + "2" * 64,
            "verifier_is_oracle": False,
            "wrong_majority_n": sum(int(task["wrong_majority"]) for task in tasks),
        },
    )
    _write_json(
        root / "results" / "arc3_win_condition_survey.json",
        {"n_games": 2, "per_game_surveys": [{"game": "game-a"}, {"game": "game-b"}]},
    )
    model_path = root / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
    _write_json(
        model_path,
        {
            "feature_names": list(exp4244.FEATURE_NAMES),
            "model": {"model_type": "fixture"},
            "model_specs": {"architecture": "fixture_set_encoder", "training_epochs": 0},
            "random_seed": 4244,
            "reproducibility_checksum": "sha256:" + "3" * 64,
            "set_encoder_oof": {"fold_task_ids": [], "rows": []},
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / "results" / "experiment_4244_arc_set_encoder_aggregator_build.json",
        {
            "aggregator_trained": True,
            "honest_verdict": "complete: fixture",
            "learned_verifier_path": str(model_path),
            "model_specs": {"architecture": "fixture_set_encoder", "training_epochs": 0},
            "random_seed": 4244,
            "reproducibility_checksum": "sha256:" + "4" * 64,
            "verifier_is_oracle": False,
        },
    )
    return score_by_candidate


def _fake_training_report(
    corpus: mod.GameAnnotatedCorpus,
    folds: list[mod.GameFold],
    score_by_candidate: dict[str, float],
) -> mod.CrossGameTrainingReport:
    rows = []
    for fold_index, fold in enumerate(folds):
        train_task_ids = tuple(sorted(fold.train_task_ids))
        for row in corpus.rows:
            if row.task_id in fold.held_out_task_ids:
                rows.append(
                    exp4244.OOFRow(
                        task_id=row.task_id,
                        candidate_id=row.candidate_id,
                        correct=row.correct,
                        score=score_by_candidate[row.candidate_id],
                        fold=fold_index,
                        train_task_ids=train_task_ids,
                    )
                )
    return mod.CrossGameTrainingReport(
        rows=rows,
        fold_summaries=[
            {
                "fold": index,
                "held_out_games": sorted(fold.held_out_games),
                "train_games": sorted(fold.train_games),
                "held_out_task_n": len(fold.held_out_task_ids),
            }
            for index, fold in enumerate(folds)
        ],
        training_config={"fixture": True},
    )


def test_req_4258_spec_declares_cross_game_transfer_contract() -> None:
    """REQ-VERIFY-4258: OpenSpec declares the cross-game transfer gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4258",
        "SCENARIO-VERIFY-4258",
        "python/carnot/reporting/arc_oracle_distinct_cross_game_transfer_4258.py",
        "results/experiment_4258_arc_oracle_distinct_cross_game_transfer.py",
        "blocked_arc_game_ids_unrecoverable",
        "cross_game_delta",
        "cross_game_ci95",
        "within_game_minus_cross_game_gap",
        "held_out_game_n",
        "oracle_at_k",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4258_recovers_game_ids_and_builds_game_disjoint_folds(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4258: recovered folds never share games between train and test."""

    _write_cross_game_fixture(tmp_path)
    corpus = mod.load_game_annotated_corpus(tmp_path)
    folds = mod.build_game_disjoint_folds(corpus.task_game_ids, random_seed=4258, n_folds=2)

    assert corpus.held_out_game_n == 2
    assert corpus.held_out_task_n == 4
    assert {row.game_id for row in corpus.rows} == {"game-a", "game-b"}
    assert len(folds) == 2
    assert set().union(*(fold.held_out_games for fold in folds)) == {"game-a", "game-b"}
    for fold in folds:
        assert fold.train_games.isdisjoint(fold.held_out_games)
        assert fold.train_task_ids.isdisjoint(fold.held_out_task_ids)


def test_scenario_4258_measures_held_out_game_gate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4258: held-out-game metrics use task-level deltas."""

    scores = _write_cross_game_fixture(tmp_path)
    corpus = mod.load_game_annotated_corpus(tmp_path)
    folds = mod.build_game_disjoint_folds(corpus.task_game_ids, random_seed=4258, n_folds=2)
    report = _fake_training_report(corpus, folds, scores)

    metrics = mod.measure_cross_game_gate(
        corpus,
        report.rows,
        random_seed=4258,
        bootstrap_resamples=200,
    )

    assert metrics["cross_game_delta"] == pytest.approx(0.75)
    assert metrics["within_game_minus_cross_game_gap"] == pytest.approx(
        mod.WITHIN_GAME_DELTA_393 - 0.75
    )
    assert metrics["pass_rates"]["vote_at_1"] == pytest.approx(0.25)
    assert metrics["pass_rates"]["set_encoder_at_1"] == pytest.approx(1.0)
    assert metrics["oracle_at_k"] == pytest.approx(1.0)
    assert metrics["matched_control_delta"] == pytest.approx(0.75)
    assert metrics["held_out_game_n"] == 2
    assert metrics["held_out_task_n"] == 4
    assert metrics["task_rows"][0]["game_id"] in {"game-a", "game-b"}


def test_scenario_4258_run_writes_cross_game_transfer_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-4258: run emits required fields and keeps verifier oracle-distinct."""

    scores = _write_cross_game_fixture(tmp_path)

    def fake_train(
        corpus: mod.GameAnnotatedCorpus,
        folds: list[mod.GameFold],
        **_kwargs: object,
    ) -> mod.CrossGameTrainingReport:
        return _fake_training_report(corpus, folds, scores)

    monkeypatch.setattr(mod, "train_cross_game_oof", fake_train)
    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean, bootstrap_resamples=200)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: cross_game_transfers"
    assert artifact["cross_game_delta"] == pytest.approx(0.75)
    assert artifact["cross_game_ci95"][0] > 0.0
    assert artifact["held_out_game_n"] == 2
    assert artifact["held_out_task_n"] == 4
    assert artifact["oracle_at_k"] == pytest.approx(1.0)
    assert artifact["verifier_is_oracle"] is False
    assert artifact["adversarial_verify"]["circular_moat_overclaim_clean"] is True
    assert (tmp_path / mod.OUTPUT_REL).exists()


def test_scenario_4258_no_headroom_and_unrecoverable_game_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-4258: no-headroom and missing-game cases are terminal."""

    scores = _write_cross_game_fixture(tmp_path / "no-headroom", vote_all_correct=True)

    def fake_train(
        corpus: mod.GameAnnotatedCorpus,
        folds: list[mod.GameFold],
        **_kwargs: object,
    ) -> mod.CrossGameTrainingReport:
        return _fake_training_report(corpus, folds, scores)

    monkeypatch.setattr(mod, "train_cross_game_oof", fake_train)
    no_headroom = mod.run(
        tmp_path / "no-headroom",
        adversarial_runner=_adversarial_flagged,
        bootstrap_resamples=200,
    )
    mod.validate_artifact(no_headroom)
    assert no_headroom["honest_verdict"] == "complete: no_headroom"
    assert no_headroom["honest_read"] == "no_headroom"
    assert no_headroom["oracle_at_k"] == no_headroom["pass_rates"]["vote_at_1"]
    assert no_headroom["false_negative_risk"] is True
    assert no_headroom["adversarial_verify"]["circular_moat_overclaim_clean"] is False

    _write_cross_game_fixture(tmp_path / "blocked", include_game_ids=False)
    blocked = mod.run(tmp_path / "blocked", adversarial_runner=_adversarial_clean)
    mod.validate_artifact(blocked)
    assert blocked["honest_verdict"] == mod.BLOCKED_GAME_IDS_VERDICT
    assert blocked["cross_game_delta"] is None
    assert blocked["held_out_game_n"] == 0
    assert blocked["verifier_is_oracle"] is False


def test_req_4258_validation_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4258: schema validation, checksums, and defensive helpers are deterministic."""

    _write_cross_game_fixture(tmp_path)
    corpus = mod.load_game_annotated_corpus(tmp_path)
    folds = mod.build_game_disjoint_folds(corpus.task_game_ids, random_seed=1, n_folds=99)
    assert len(folds) == 2
    trained = mod.train_cross_game_oof(
        corpus,
        folds,
        bootstrap_n=0,
        training_epochs=0,
        hidden_dim=4,
        lr=0.01,
    )
    assert len(trained.rows) == len(corpus.rows)
    assert trained.training_config["training_epochs"] == 0

    assert mod._round_metric(1.23456789123) == 1.2345678912
    assert mod._safe_float(True) == 0.0
    assert mod._safe_float("bad") == 0.0
    assert mod._safe_float(math.inf) == 0.0
    assert mod._safe_int(False) == 0
    assert mod._safe_int("bad") == 0
    assert mod._bootstrap_ci95([], random_seed=1, resamples=10) == [0.0, 0.0]
    assert mod._bootstrap_ci95([0.25], random_seed=1, resamples=10) == [0.25, 0.25]
    assert mod._bootstrap_ci95([0.0, 1.0], random_seed=1, resamples=0) == [0.5, 0.5]
    assert mod._ci_excludes_zero([0.1, 0.2]) is True
    assert mod._ci_excludes_zero([-0.1, 0.1]) is False
    assert mod._clean_adversarial_report({"reports": [{"flags": []}]})["status"] == "clean"
    assert mod._sha256_file(tmp_path / mod.POOL_REL)

    checksum = mod.reproducibility_checksum(
        corpus=corpus,
        folds=folds,
        metrics={"cross_game_delta": 0.5},
        random_seed=4258,
    )
    assert checksum.startswith("sha256:")

    blocked = mod._blocked_artifact(
        mod.BLOCKED_GAME_IDS_VERDICT,
        random_seed=4258,
        checksum="sha256:" + "0" * 64,
        duration_s=0.01,
        missing_task_ids=["fixture:missing"],
    )
    invalid_cases = [
        ({key: value for key, value in blocked.items() if key != "cross_game_delta"}, "missing required"),
        ({**blocked, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**blocked, "cross_game_delta": True}, "cross_game_delta"),
        ({**blocked, "cross_game_ci95": [0.0]}, "cross_game_ci95"),
        ({**blocked, "within_game_minus_cross_game_gap": True}, "within_game_minus_cross_game_gap"),
        ({**blocked, "held_out_game_n": 1.2}, "held_out_game_n"),
        ({**blocked, "held_out_task_n": 1.2}, "held_out_task_n"),
        ({**blocked, "oracle_at_k": True}, "oracle_at_k"),
        ({**blocked, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**blocked, "random_seed": "4258"}, "random_seed"),
        ({**blocked, "field_principles": {}}, "field_principles"),
        ({**blocked, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)


def test_req_4258_defensive_precondition_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4258: malformed inputs and non-winning transfer reads are explicit."""

    scores = _write_cross_game_fixture(tmp_path)
    corpus = mod.load_game_annotated_corpus(tmp_path)
    folds = mod.build_game_disjoint_folds(corpus.task_game_ids, random_seed=4258, n_folds=2)
    vote_scores = {
        row.candidate_id: row.vote_weight
        for row in corpus.rows
    }
    within_metrics = mod.measure_cross_game_gate(
        corpus,
        _fake_training_report(corpus, folds, vote_scores).rows,
        random_seed=4258,
        bootstrap_resamples=100,
    )
    assert within_metrics["honest_read"] == "within_game_only"
    dropped_metrics = mod.measure_cross_game_gate(
        corpus,
        _fake_training_report(corpus, folds, scores).rows[:-1],
        random_seed=4258,
        bootstrap_resamples=10,
    )
    assert dropped_metrics["dropped_task_n"] == 1

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json_object(list_json)
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_ARTIFACT_VERDICT):
        mod._resolve_required_path(tmp_path, None)
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_ARTIFACT_VERDICT):
        mod._resolve_required_path(tmp_path, "missing.json")

    assert mod._survey_games(tmp_path / "no-survey") == []
    bad_survey = tmp_path / "bad-survey"
    _write_json(bad_survey / mod.SURVEY_REL, {"per_game_surveys": {}, "ranked_targets": []})
    assert mod._survey_games(bad_survey) == []
    (bad_survey / mod.SURVEY_REL).write_text("[]", encoding="utf-8")
    assert mod._survey_games(bad_survey) == []

    assert mod._candidate_game_ids({}, {"game-a"}) == set()
    assert mod._candidate_game_ids([None, {"game_id": "game-a"}], {"game-a"}) == {"game-a"}
    assert mod._recover_game_id({"candidates": [{"game_id": "game-b"}]}, {"game-b"}) == "game-b"
    assert mod._recover_game_id({"task_id": "game-a:task-x"}, {"game-a"}) == "game-a"
    mapping, missing = mod._task_game_map_from_pool(
        {"tasks": [None, {"task_id": ""}, {"task_id": "x"}]},
        ["game-a"],
    )
    assert mapping == {}
    assert missing == ["x"]

    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_GAME_IDS_VERDICT):
        mod.build_game_disjoint_folds({"task": "only-game"})

    malformed_pool_root = tmp_path / "malformed-pool"
    malformed_pool = malformed_pool_root / mod.POOL_REL
    malformed_pool.parent.mkdir(parents=True)
    malformed_pool.write_text("not gzip", encoding="utf-8")
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_GAME_IDS_VERDICT):
        mod._load_pool_payload(malformed_pool)
    with gzip.open(malformed_pool, "wt", encoding="utf-8") as handle:
        json.dump({"tasks": {}}, handle)
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_GAME_IDS_VERDICT):
        mod._load_pool_payload(malformed_pool)

    missing_rows_root = tmp_path / "missing-rows"
    _write_cross_game_fixture(missing_rows_root)
    with gzip.open(missing_rows_root / mod.POOL_REL, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["tasks"].append(
        {
            "candidate_count": 0,
            "candidates": [],
            "game_id": "game-a",
            "task_id": "game-a:empty-task",
        }
    )
    with gzip.open(missing_rows_root / mod.POOL_REL, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_GAME_IDS_VERDICT):
        mod.load_game_annotated_corpus(missing_rows_root)

    for index, edit in enumerate(
        (
            lambda payloads: payloads["build"].update({"aggregator_trained": False}),
            lambda payloads: payloads["build"].update({"verifier_is_oracle": True}),
            lambda payloads: payloads["model"].update({"verifier_is_oracle": True}),
        )
    ):
        bad_root = tmp_path / f"bad-required-{index}"
        _write_cross_game_fixture(bad_root)
        paths = {
            "build": bad_root / mod.SET_ENCODER_BUILD_REL,
            "model": bad_root / mod.SET_ENCODER_MODEL_REL,
        }
        payloads = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
        edit(payloads)
        for key, path in paths.items():
            _write_json(path, payloads[key])
        with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_ARTIFACT_VERDICT):
            mod._load_required_artifacts(bad_root)

    broken_root = tmp_path / "broken-required"
    _write_cross_game_fixture(broken_root)
    (broken_root / mod.SET_ENCODER_BUILD_REL).write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_ARTIFACT_VERDICT):
        mod._load_required_artifacts(broken_root)
