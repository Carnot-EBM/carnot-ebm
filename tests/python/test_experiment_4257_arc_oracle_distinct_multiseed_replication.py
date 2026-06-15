"""Tests for Exp 4257 ARC oracle-distinct multi-seed replication.

Spec refs: REQ-VERIFY-4257, SCENARIO-VERIFY-4257.
"""

from __future__ import annotations

import gzip
import inspect
import json
from pathlib import Path

import pytest

from carnot.reporting import arc_oracle_distinct_multiseed_replication_4257 as mod
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adversarial_clean(_path: Path) -> dict:
    return {"returncode": 0, "reports": [{"flag_count": 0, "flags": [], "max_severity": 0}]}


def _adversarial_flagged(_path: Path) -> dict:
    return {
        "returncode": 1,
        "reports": [
            {
                "flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}],
                "max_severity": 2,
            }
        ],
    }


def _features(vote: float, confidence: float = 0.5) -> dict[str, float]:
    return {
        name: 0.0
        for name in exp4244.FEATURE_NAMES
    } | {
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


def _write_replication_fixture(
    root: Path,
    *,
    correct_indices: list[int],
    vote_weights: list[list[float]],
    set_encoder_scores: list[list[float]],
    ci95: list[float] | None = None,
) -> Path:
    task_ids = [f"mini:task-{index}" for index in range(len(correct_indices))]
    tasks = []
    oof_rows = []
    for task_index, task_id in enumerate(task_ids):
        candidates = []
        for candidate_index, vote in enumerate(vote_weights[task_index]):
            candidate_id = f"{task_id}::candidate{candidate_index}"
            is_correct = candidate_index == correct_indices[task_index]
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "features": _features(vote, 0.9 if is_correct else 0.2),
                    "grid": [[candidate_index]],
                    "is_correct": is_correct,
                    "votes": vote,
                }
            )
            oof_rows.append(
                {
                    "candidate_id": candidate_id,
                    "correct": is_correct,
                    "fold": task_index % 2,
                    "score": float(set_encoder_scores[task_index][candidate_index]),
                    "task_id": task_id,
                    "train_task_ids": [other for other in task_ids if other != task_id],
                }
            )
        vote_top = max(candidates, key=lambda item: (item["features"]["vote_weight"], -item["candidate_index"]))
        tasks.append(
            {
                "candidate_count": len(candidates),
                "candidates": candidates,
                "oracle_present": True,
                "raw_task_id": f"task-{task_index}",
                "source_id": "mini",
                "task_id": task_id,
                "vote_top_candidate_id": vote_top["candidate_id"],
                "wrong_majority": vote_top["candidate_index"] != correct_indices[task_index],
            }
        )

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
    model_path = root / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
    _write_json(
        model_path,
        {
            "feature_names": list(exp4244.FEATURE_NAMES),
            "held_out_task_n": len(tasks),
            "model": {"model_type": "fixture_oof_only"},
            "model_specs": {"architecture": "fixture_set_encoder", "training_epochs": 32},
            "pool_artifact_path": str(pool_path),
            "pool_artifact_sha256": "fixture",
            "random_seed": 4244,
            "reproducibility_checksum": "sha256:" + "3" * 64,
            "set_encoder_oof": {
                "auroc": 1.0,
                "ci95": [1.0, 1.0],
                "fold_task_ids": [task_ids],
                "rows": oof_rows,
            },
            "spec_refs": ["REQ-VERIFY-4244"],
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / "results" / "experiment_4244_arc_set_encoder_aggregator_build.json",
        {
            "aggregator_trained": True,
            "held_out_task_n": len(tasks),
            "honest_verdict": "complete: fixture",
            "learned_verifier_path": str(model_path),
            "model_specs": {"architecture": "fixture_set_encoder", "training_epochs": 32},
            "oracle_distinct_auroc": 1.0,
            "random_seed": 4244,
            "reproducibility_checksum": "sha256:" + "4" * 64,
            "set_encoder_vs_logistic_auroc_delta": 0.1,
            "verifier_is_oracle": False,
            "wrong_majority_n": sum(int(task["wrong_majority"]) for task in tasks),
        },
    )
    _write_json(
        root / "results" / "experiment_4245_arc_set_encoder_beats_vote.json",
        {
            "honest_verdict": "complete: arc_oracle_distinct_set_encoder_beats_vote",
            "held_out_task_n": len(tasks),
            "learned_verifier_path": str(model_path),
            "model_specs": {"architecture": "fixture_set_encoder"},
            "oracle_at_k": 1.0,
            "set_encoder_minus_vote_ci95": ci95 or [0.5, 0.9],
            "set_encoder_minus_vote_delta": 0.75,
            "verifier_is_oracle": False,
        },
    )
    return root


def test_req_4257_spec_declares_multiseed_replication_contract() -> None:
    """REQ-VERIFY-4257: OpenSpec declares the multi-seed replication gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4257",
        "SCENARIO-VERIFY-4257",
        "python/carnot/reporting/arc_oracle_distinct_multiseed_replication_4257.py",
        "results/experiment_4257_arc_oracle_distinct_multiseed_replication.py",
        "blocked_arc_grown_pool_missing",
        "oracle_distinct_win_replicates",
        "per_seed_deltas",
        "mean_delta",
        "cross_seed_ci95",
        "independent_rescore_delta",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4257_independent_rescore_recomputes_persisted_gate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4257: clean-room re-score reads pool and model rows directly."""

    _write_replication_fixture(
        tmp_path,
        correct_indices=[1, 0, 1, 1],
        vote_weights=[[9, 1], [9, 1], [9, 1], [9, 1]],
        set_encoder_scores=[[0.1, 0.9], [0.8, 0.2], [0.1, 0.9], [0.1, 0.9]],
    )

    report = mod.independent_rescore_persisted_artifact(tmp_path)

    assert report["independent_rescore_delta"] == pytest.approx(0.75)
    assert report["pass_rates"]["vote_at_1"] == pytest.approx(0.25)
    assert report["pass_rates"]["set_encoder_at_1"] == pytest.approx(1.0)
    assert report["oracle_at_k"] == pytest.approx(1.0)
    assert report["held_out_task_n"] == 4
    source = inspect.getsource(mod.independent_rescore_persisted_artifact)
    assert "arc_set_encoder_beats_vote_4245" not in inspect.getsource(mod)
    assert "exp4245" not in source
    assert "exp4244." not in source


def test_scenario_4257_run_writes_replicated_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-4257: positive multi-seed lift plus in-CI rescore gates true."""

    _write_replication_fixture(
        tmp_path,
        correct_indices=[1, 0, 1, 1],
        vote_weights=[[9, 1], [9, 1], [9, 1], [9, 1]],
        set_encoder_scores=[[0.1, 0.9], [0.8, 0.2], [0.1, 0.9], [0.1, 0.9]],
    )
    deltas = [0.72, 0.74, 0.75, 0.76, 0.78]

    def fake_seed(*_args: object, random_seed: int, **_kwargs: object) -> mod.SeedReplicationResult:
        index = mod.DEFAULT_REPLICATION_SEEDS.index(random_seed)
        return mod.SeedReplicationResult(
            random_seed=random_seed,
            delta=deltas[index],
            held_out_task_n=4,
            vote_at_1=0.25,
            set_encoder_at_1=0.25 + deltas[index],
            oracle_at_k=1.0,
            fold_task_ids=[[f"mini:task-{index}"]],
        )

    monkeypatch.setattr(mod, "_train_seed_replication", fake_seed)
    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: arc_oracle_distinct_win_replicates_multiseed"
    assert artifact["oracle_distinct_win_replicates"] is True
    assert artifact["per_seed_deltas"] == deltas
    assert artifact["mean_delta"] == pytest.approx(sum(deltas) / len(deltas))
    assert artifact["cross_seed_ci95"][0] > 0.0
    assert artifact["independent_rescore_delta"] == pytest.approx(0.75)
    assert artifact["independent_rescore_within_4245_ci"] is True
    assert artifact["sign_flip_seeds"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["random_seeds_used"] == mod.DEFAULT_REPLICATION_SEEDS
    assert artifact["n_seeds"] == 5
    assert artifact["adversarial_verify"]["status"] == "clean"
    assert (tmp_path / mod.OUTPUT_REL).exists()


def test_scenario_4257_fragility_and_blocked_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-4257: sign flips or missing pool stop the positive gate."""

    blocked = mod.run(tmp_path, adversarial_runner=_adversarial_clean)
    mod.validate_artifact(blocked)
    assert blocked["honest_verdict"] == mod.BLOCKED_GROWN_POOL_VERDICT
    assert blocked["oracle_distinct_win_replicates"] is False
    assert blocked["per_seed_deltas"] == []
    assert blocked["verifier_is_oracle"] is False

    _write_replication_fixture(
        tmp_path / "fragile",
        correct_indices=[1, 0, 1, 1],
        vote_weights=[[9, 1], [9, 1], [9, 1], [9, 1]],
        set_encoder_scores=[[0.1, 0.9], [0.8, 0.2], [0.1, 0.9], [0.1, 0.9]],
        ci95=[0.8, 0.9],
    )
    deltas = [0.3, 0.2, -0.1, 0.25, 0.2]

    def fragile_seed(*_args: object, random_seed: int, **_kwargs: object) -> mod.SeedReplicationResult:
        index = mod.DEFAULT_REPLICATION_SEEDS.index(random_seed)
        return mod.SeedReplicationResult(
            random_seed=random_seed,
            delta=deltas[index],
            held_out_task_n=4,
            vote_at_1=0.25,
            set_encoder_at_1=0.25 + deltas[index],
            oracle_at_k=1.0,
            fold_task_ids=[],
        )

    monkeypatch.setattr(mod, "_train_seed_replication", fragile_seed)
    fragile = mod.run(tmp_path / "fragile", adversarial_runner=_adversarial_flagged)

    mod.validate_artifact(fragile)
    assert fragile["honest_verdict"] == "complete: arc_oracle_distinct_win_fragile_multiseed"
    assert fragile["oracle_distinct_win_replicates"] is False
    assert fragile["sign_flip_seeds"] == [mod.DEFAULT_REPLICATION_SEEDS[2]]
    assert fragile["independent_rescore_within_4245_ci"] is False
    assert fragile["adversarial_verify"]["circular_moat_overclaim_clean"] is False


def test_training_wrapper_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4257: schema, CI, checksum, and helper edge cases are deterministic."""

    root = _write_replication_fixture(
        tmp_path,
        correct_indices=[1, 0, 1, 0],
        vote_weights=[[9, 1], [9, 1], [9, 1], [9, 1]],
        set_encoder_scores=[[0.1, 0.9], [0.8, 0.2], [0.1, 0.9], [0.8, 0.2]],
    )
    corpus = exp4244.load_grown_pool(root)
    result = mod._train_seed_replication(
        corpus,
        random_seed=4257,
        n_folds=2,
        bootstrap_n=0,
        training_epochs=0,
        hidden_dim=4,
        lr=0.01,
    )
    assert result.random_seed == 4257
    assert result.held_out_task_n == 4
    assert len(result.fold_task_ids) == 2

    assert mod._cross_seed_ci95([]) == [0.0, 0.0]
    assert mod._cross_seed_ci95([0.4]) == [0.4, 0.4]
    assert mod._ci_excludes_zero([0.1, 0.2]) is True
    assert mod._ci_excludes_zero([-0.2, -0.1]) is True
    assert mod._ci_excludes_zero([-0.1, 0.1]) is False
    assert mod._mean([]) == 0.0
    assert mod._round_metric(1.23456789123) == 1.2345678912
    assert mod._clean_adversarial_report({"reports": [{"flags": []}]})["status"] == "clean"
    assert mod._safe_float(True) == 0.0
    assert mod._safe_float("bad") == 0.0
    assert mod._safe_int(False) == 0
    assert mod._safe_int("bad") == 0

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json_object(bad_json)
    with pytest.raises(mod.BlockedRun, match="missing"):
        mod._resolve_required_path(tmp_path, "")
    with pytest.raises(mod.BlockedRun, match="missing"):
        mod._resolve_required_path(tmp_path, "missing.json")
    assert mod._load_oof_scores({"set_encoder_oof": {"rows": {}}}) == {}

    malformed_root = tmp_path / "malformed"
    _write_replication_fixture(
        malformed_root,
        correct_indices=[0],
        vote_weights=[[1, 0]],
        set_encoder_scores=[[0.8, 0.1]],
    )
    model_path = malformed_root / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
    model_payload = json.loads(model_path.read_text(encoding="utf-8"))
    model_payload["set_encoder_oof"]["rows"] = [None, {"candidate_id": 7}, {"candidate_id": "x", "task_id": "t"}]
    _write_json(model_path, model_payload)
    assert mod.independent_rescore_persisted_artifact(malformed_root)["held_out_task_n"] == 0

    too_few = mod.run(malformed_root, random_seeds=[1, 2, 3, 4], adversarial_runner=_adversarial_clean)
    assert too_few["honest_verdict"] == "blocked_arc_multiseed_requires_at_least_5_seeds"

    for index, edit in enumerate(
        (
            lambda payloads: payloads["build"].update({"aggregator_trained": False}),
            lambda payloads: payloads["build"].update({"verifier_is_oracle": True}),
            lambda payloads: payloads["model"].update({"verifier_is_oracle": True}),
            lambda payloads: payloads["single_seed"].update({"verifier_is_oracle": True}),
        )
    ):
        bad_root = tmp_path / f"bad-required-{index}"
        _write_replication_fixture(
            bad_root,
            correct_indices=[0],
            vote_weights=[[1, 0]],
            set_encoder_scores=[[0.8, 0.1]],
        )
        paths = {
            "build": bad_root / mod.SET_ENCODER_BUILD_REL,
            "model": bad_root / mod.SET_ENCODER_MODEL_REL,
            "single_seed": bad_root / mod.SINGLE_SEED_WIN_REL,
        }
        payloads = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
        edit(payloads)
        for key, path in paths.items():
            _write_json(path, payloads[key])
        with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_ARTIFACT_VERDICT):
            mod._load_required_artifacts(bad_root)

    list_build_root = tmp_path / "list-build"
    _write_replication_fixture(
        list_build_root,
        correct_indices=[0],
        vote_weights=[[1, 0]],
        set_encoder_scores=[[0.8, 0.1]],
    )
    (list_build_root / mod.SET_ENCODER_BUILD_REL).write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_ARTIFACT_VERDICT):
        mod._load_required_artifacts(list_build_root)

    missing_pool_root = tmp_path / "missing-pool"
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_GROWN_POOL_VERDICT):
        mod._load_clean_room_pool(missing_pool_root)

    bad_gzip_root = tmp_path / "bad-gzip"
    bad_gzip_pool = bad_gzip_root / mod.POOL_REL
    bad_gzip_pool.parent.mkdir(parents=True)
    bad_gzip_pool.write_text("not gzip", encoding="utf-8")
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_GROWN_POOL_VERDICT):
        mod._load_clean_room_pool(bad_gzip_root)

    bad_schema_root = tmp_path / "bad-schema"
    bad_schema_pool = bad_schema_root / mod.POOL_REL
    bad_schema_pool.parent.mkdir(parents=True)
    with gzip.open(bad_schema_pool, "wt", encoding="utf-8") as handle:
        json.dump({"tasks": {}}, handle)
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_GROWN_POOL_VERDICT):
        mod._load_clean_room_pool(bad_schema_root)

    odd_pool_root = tmp_path / "odd-pool"
    odd_pool = odd_pool_root / mod.POOL_REL
    odd_pool.parent.mkdir(parents=True)
    with gzip.open(odd_pool, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "reproducibility_checksum": "sha256:" + "9" * 64,
                "tasks": [
                    None,
                    {"task_id": "", "candidates": []},
                    {
                        "task_id": "mini:odd",
                        "candidates": [
                            None,
                            {
                                "candidate_index": "bad",
                                "features": "bad",
                                "is_correct": True,
                                "votes": "bad",
                            },
                        ],
                    },
                ],
            },
            handle,
        )
    odd = mod._load_clean_room_pool(odd_pool_root)
    assert len(odd.candidates) == 1
    assert odd.candidates[0].candidate_id == "mini:odd::candidate1"
    assert odd.candidates[0].vote_weight == 0.0

    blocked = mod._blocked_artifact(
        mod.BLOCKED_GROWN_POOL_VERDICT,
        random_seed=4257,
        checksum="sha256:" + "0" * 64,
        duration_s=0.01,
    )
    invalid_cases = [
        ({key: value for key, value in blocked.items() if key != "mean_delta"}, "missing required"),
        ({**blocked, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**blocked, "oracle_distinct_win_replicates": {"value": False}}, "bare bool"),
        ({**blocked, "per_seed_deltas": [True]}, "per_seed_deltas"),
        ({**blocked, "mean_delta": True}, "bare float"),
        ({**blocked, "cross_seed_ci95": [0.0]}, "ci95"),
        ({**blocked, "independent_rescore_delta": None}, "bare float"),
        ({**blocked, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**blocked, "random_seeds_used": [1, "2"]}, "random_seeds_used"),
        ({**blocked, "field_principles": {}}, "field_principles"),
        ({**blocked, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    specs = mod._model_specs(
        build_artifact={"model_specs": {"architecture": "fixture"}},
        seed_results=[result],
        independent_report={"held_out_task_n": 4},
        random_seeds=[4257],
        n_folds=2,
        training_epochs=0,
        hidden_dim=4,
    )
    checksum = mod.reproducibility_checksum(
        corpus=corpus,
        seed_results=[result],
        independent_report={"independent_rescore_delta": 0.5},
        random_seeds=[4257],
    )
    assert specs["status"] == "complete"
    assert checksum.startswith("sha256:")
