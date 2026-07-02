"""Tests for Exp 5160 oracle-distinct cross-corpus closure.

Spec refs: REQ-VERIFY-5160, SCENARIO-VERIFY-5160,
SCENARIO-VERIFY-5160-UPSTREAM-BLOCKED.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carnot.reporting import oracle_distinct_cross_corpus_closure_5160 as mod
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {"returncode": 0, "reports": [{"flags": [], "flag_count": 0}]}


def _features(vote: float, confidence: float) -> dict[str, float]:
    values = {name: 0.0 for name in exp4244.FEATURE_NAMES}
    values.update(
        {
            "vote_weight": vote,
            "self_consistency_margin": vote - 0.5,
            "vote_weight_rank_fraction": vote,
            "cell_confidence_mean": confidence,
            "cell_confidence_margin": confidence - 0.5,
            "cell_confidence_rank_fraction": confidence,
            "grid_height": 2.0,
            "grid_width": 2.0,
            "grid_cells": 4.0,
            "set_candidate_count": 2.0,
            "set_vote_mean": 0.5,
            "set_vote_max": vote,
        }
    )
    return values


def _candidate(
    task_id: str,
    index: int,
    *,
    correct: bool,
    grid: list[list[int]],
    vote: float,
    score_hint: float,
) -> dict[str, Any]:
    return {
        "candidate_id": f"{task_id}::candidate{index}",
        "candidate_index": index,
        "features": _features(vote, score_hint),
        "grid": grid,
        "is_correct": correct,
        "votes": vote * 10.0,
    }


def _task(task_id: str, raw: str, source_id: str, *, correct_index: int) -> dict[str, Any]:
    candidates = [
        _candidate(
            task_id,
            0,
            correct=correct_index == 0,
            grid=[[int(raw[-1], 16), 0]],
            vote=0.8,
            score_hint=0.2 if correct_index else 0.9,
        ),
        _candidate(
            task_id,
            1,
            correct=correct_index == 1,
            grid=[[int(raw[-1], 16), 1]],
            vote=0.2,
            score_hint=0.9 if correct_index else 0.2,
        ),
    ]
    return {
        "candidate_count": 2,
        "candidates": candidates,
        "oracle_present": True,
        "raw_task_id": raw,
        "source_id": source_id,
        "task_id": task_id,
        "vote_top_candidate_id": candidates[0]["candidate_id"],
        "wrong_majority": correct_index != 0,
    }


def _write_5160_fixture(root: Path) -> None:
    original_tasks = [
        _task("gap3_stage2:50a16a69", "50a16a69", "gap3_stage2", correct_index=1),
        _task("gap3_stage2:66e6c45b", "66e6c45b", "gap3_stage2", correct_index=0),
        _task("gap4_arc2:13e47133", "13e47133", "gap4_arc2", correct_index=1),
    ]
    original = {
        "candidate_n": 6,
        "positive_candidate_n": 3,
        "schema": "carnot.arc_candidate_pool_grow.v1",
        "source_paths": [],
        "task_n": 3,
        "tasks": original_tasks,
        "wrong_majority_n": 2,
    }
    _write_gzip_json(root / mod.ORIGINAL_POOL_REL, original)
    _write_json(
        root / mod.EXP5151_REL,
        {
            "experiment": "experiment_5151_arc_oracle_distinct_hardening_v472",
            "honest_verdict": "complete_arc_set_encoder_win_not_hardened: unresolved_axes=cross_game",
            "leak_audit_passed": True,
            "multiseed_delta_ci95": [0.42, 0.49],
            "cross_game_blocked_reason": "blocked_arc_game_ids_unrecoverable",
            "verifier_is_oracle": False,
        },
    )
    _write_gzip_json(
        root / mod.PREFERRED_SECOND_POOL_REL,
        {
            "experiment": "arc3_gap4_arc2_eval_pool",
            "entries": [
                {
                    "task": "13e47133",
                    "test_input": [[1]],
                    "demos": [],
                    "candidates": [{"grid": [[1]], "votes": 1, "q_mean": 0.1, "correct": True}],
                }
            ],
            "n_candidates": 1,
            "n_entries": 1,
        },
    )
    fallback_tasks: list[dict[str, Any]] = []
    overlap_grid = original_tasks[0]["candidates"][0]["grid"]
    for task_index in range(8):
        task_id = f"arcgen:task{task_index:02d}:000"
        correct_index = 1 if task_index < 6 else 0
        candidates = [
            _candidate(
                task_id,
                0,
                correct=correct_index == 0,
                grid=overlap_grid if task_index == 0 else [[task_index, 0]],
                vote=0.8,
                score_hint=0.1 if correct_index else 0.9,
            ),
            _candidate(
                task_id,
                1,
                correct=correct_index == 1,
                grid=[[task_index, 1]],
                vote=0.2,
                score_hint=0.9 if correct_index else 0.1,
            ),
        ]
        fallback_tasks.append(
            {
                "candidate_count": len(candidates),
                "candidates": candidates,
                "oracle_present": True,
                "raw_task_id": f"task{task_index:02d}:000",
                "source_id": "arcgen",
                "task_id": task_id,
                "vote_top_candidate_id": candidates[0]["candidate_id"],
                "wrong_majority": correct_index != 0,
            }
        )
    _write_gzip_json(
        root / mod.FALLBACK_SECOND_POOL_REL,
        {
            "schema": "carnot.arcgen_cross_generator_pool_4291.v1",
            "source_kind": "arcgen",
            "tasks": fallback_tasks,
        },
    )


def _patch_seed_results(monkeypatch: pytest.MonkeyPatch, deltas: list[float] | None = None) -> None:
    seed_deltas = deltas or [0.42, 0.44, 0.46, 0.48, 0.50]

    def fake_seed(
        corpus: exp4244.GrownPoolCorpus,
        *,
        random_seed: int,
        **_kwargs: object,
    ) -> mod.SeedReplicationResult:
        index = mod.DEFAULT_RANDOM_SEEDS.index(random_seed)
        delta = seed_deltas[index]
        return mod.SeedReplicationResult(
            random_seed=random_seed,
            auroc=0.9 + index * 0.01,
            delta=delta,
            held_out_task_n=corpus.held_out_task_n,
            candidate_count=corpus.positive_candidate_n + index,
            vote_at_1=0.25,
            set_encoder_at_1=0.25 + delta,
            oracle_at_k=0.75,
            task_deltas=[1.0, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0],
            oof_rows=[
                exp4244.OOFRow(
                    task_id="arcgen:fixture",
                    candidate_id="arcgen:fixture::candidate0",
                    correct=True,
                    score=0.9,
                    fold=0,
                    train_task_ids=("arcgen:other",),
                )
            ],
        )

    monkeypatch.setattr(mod, "_train_seed_replication", fake_seed)


def test_req_5160_spec_declares_cross_corpus_contract() -> None:
    """REQ-VERIFY-5160: OpenSpec declares the corrected closure contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-5160",
        "SCENARIO-VERIFY-5160",
        "SCENARIO-VERIFY-5160-UPSTREAM-BLOCKED",
        "python/carnot/reporting/oracle_distinct_cross_corpus_closure_5160.py",
        "results/experiment_5160_oracle_distinct_cross_corpus_closure_v473.json",
        "game_id_misnomer_confirmed",
        "cross_corpus_delta_ci95",
        "diffusiongemma_gate_updated_recommendation",
        "ungate_now",
        "blocked_upstream_artifact_missing",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_5160_cross_corpus_replication_passes_on_disjoint_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-5160: misnomer confirmed, overlap filtered, five seeds pass."""

    _write_5160_fixture(tmp_path)
    _patch_seed_results(monkeypatch)

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("success_")
    assert artifact["game_id_misnomer_confirmed"] is True
    assert artifact["schema_inspection"]["game_id_fields_present"] == []
    assert artifact["preferred_second_pool_audit"]["disjoint"] is False
    assert artifact["preferred_second_pool_audit"]["raw_task_id_collision_count"] == 1
    assert artifact["second_pool_source"] == str(mod.FALLBACK_SECOND_POOL_REL)
    assert artifact["second_pool_adapter"]["dropped_overlap_candidate_n"] == 3
    assert artifact["second_pool_leak_audit_passed"] is True
    assert artifact["second_pool_leak_audit"]["candidate_grid_hash_collision_count"] == 0
    assert artifact["cross_corpus_delta"] == pytest.approx(0.46)
    assert artifact["cross_corpus_delta_ci95"][0] > 0.0
    assert artifact["cross_corpus_replication_passed"] is True
    assert artifact["diffusiongemma_gate_updated_recommendation"] == "ungate_now"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["adversarial_verify"]["status"] == "clean"
    assert (tmp_path / mod.OUTPUT_REL).exists()


def test_scenario_5160_upstream_missing_blocks_honestly(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5160-UPSTREAM-BLOCKED: missing inputs keep gate closed."""

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert artifact["game_id_misnomer_confirmed"] is False
    assert artifact["second_pool_leak_audit_passed"] is False
    assert artifact["cross_corpus_delta"] == 0.0
    assert artifact["cross_corpus_delta_ci95"] == [0.0, 0.0]
    assert artifact["cross_corpus_replication_passed"] is False
    assert artifact["diffusiongemma_gate_updated_recommendation"] == "keep_gated"
    assert artifact["random_seeds_used"] == []


def test_req_5160_schema_and_overlap_helpers(tmp_path: Path) -> None:
    """REQ-VERIFY-5160: schema inspection and overlap audit are explicit."""

    original = {"tasks": [_task("gap3_stage2:50a16a69", "50a16a69", "gap3_stage2", correct_index=1)]}
    inspection = mod.inspect_original_pool_schema(original)
    assert inspection["game_id_misnomer_confirmed"] is True
    assert inspection["raw_task_id_format"] == "8_hex_static_arc_puzzle_id"
    refuted = mod.inspect_original_pool_schema(
        {"tasks": [{**original["tasks"][0], "game_id": "bp35"}]}
    )
    assert refuted["game_id_misnomer_confirmed"] is False
    assert refuted["game_id_fields_present"] == ["game_id"]

    second = {
        "tasks": [
            {
                "task_id": "second:50a16a69",
                "raw_task_id": "50a16a69",
                "candidates": [
                    {
                        "candidate_id": "different",
                        "candidate_index": 0,
                        "features": _features(1.0, 1.0),
                        "grid": original["tasks"][0]["candidates"][1]["grid"],
                        "is_correct": True,
                    }
                ],
            }
        ]
    }
    audit = mod.audit_pool_overlap(original, second)
    assert audit["disjoint"] is False
    assert audit["raw_task_id_collision_count"] == 1
    assert audit["candidate_grid_hash_collision_count"] == 1
    assert audit["gold_grid_hash_collision_count"] == 1

    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BlockedRun):
        mod._read_json_object(path)
    with pytest.raises(mod.BlockedRun):
        mod._read_json_object(tmp_path / "missing.json")
    with pytest.raises(mod.BlockedRun):
        mod._read_gzip_json_object(tmp_path / "missing.json.gz")
    bad_gzip = tmp_path / "bad.json.gz"
    with gzip.open(bad_gzip, "wt", encoding="utf-8") as handle:
        json.dump([], handle)
    with pytest.raises(mod.BlockedRun):
        mod._read_gzip_json_object(bad_gzip)

    normalized = mod._normalized_tasks(
        {
            "entries": [
                "not-a-dict",
                {"task": "abcdef12", "candidates": ["bad", {"grid": None, "votes": "bad"}]},
            ]
        }
    )
    assert len(normalized) == 1
    assert normalized[0]["raw_task_id"] == "abcdef12"
    assert normalized[0]["candidates"][0]["grid"] is None
    assert mod._normalized_tasks({"entries": "not-a-list"}) == []
    assert mod._grid_hash(None) == ""

    empty_signature = mod._pool_signature(
        {"tasks": [{"task_id": "", "raw_task_id": "", "candidates": ["bad", {"grid": None}]}]}
    )
    assert empty_signature["task_ids"] == set()
    assert empty_signature["candidate_ids"] == set()

    fallback_features = mod._candidate_features({"features": "not-a-dict", "votes": "2.5"})
    assert fallback_features["vote_weight"] == 2.5

    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_SECOND_POOL_VERDICT):
        mod._build_second_corpus(
            tmp_path,
            original_payload=original,
            second_payload={
                "tasks": [
                    {"task_id": "second:50a16a69", "raw_task_id": "50a16a69", "candidates": []},
                    {"task_id": "second:unique", "raw_task_id": "unique", "candidates": ["bad"]},
                ]
            },
            source_rel=Path("second.json.gz"),
            source_kind="fixture",
            classic_arc_static_puzzle_pool=False,
            preferred_audit={"disjoint": False},
        )


def test_req_5160_oof_measurement_and_training_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-5160: OOF scoring keeps held-out scoring separate from training."""

    rows = [
        exp4244.GrownPoolRow(
            task_id="task0",
            candidate_id="task0::candidate0",
            candidate_index=0,
            correct=False,
            features=_features(0.8, 0.1),
            vote_weight=0.8,
        ),
        exp4244.GrownPoolRow(
            task_id="task0",
            candidate_id="task0::candidate1",
            candidate_index=1,
            correct=True,
            features=_features(0.2, 0.9),
            vote_weight=0.2,
        ),
        exp4244.GrownPoolRow(
            task_id="task1",
            candidate_id="task1::candidate0",
            candidate_index=0,
            correct=True,
            features=_features(1.0, 1.0),
            vote_weight=1.0,
        ),
    ]
    oof_rows = [
        exp4244.OOFRow(
            task_id="task0",
            candidate_id="task0::candidate0",
            correct=False,
            score=0.1,
            fold=0,
            train_task_ids=("task1",),
        ),
        exp4244.OOFRow(
            task_id="task0",
            candidate_id="task0::candidate1",
            correct=True,
            score=0.9,
            fold=0,
            train_task_ids=("task1",),
        ),
    ]
    metrics = mod._measure_oof(rows, oof_rows)
    assert metrics["delta"] == 1.0
    assert metrics["held_out_task_n"] == 1
    assert metrics["candidate_count"] == 2
    assert metrics["vote_at_1"] == 0.0
    assert metrics["set_encoder_at_1"] == 1.0

    def fake_split(
        split_rows: list[exp4244.GrownPoolRow], *, random_seed: int, n_folds: int
    ) -> list[tuple[str, ...]]:
        assert split_rows == rows
        assert random_seed == 5160
        assert n_folds == 2
        return [("task0",), ("task1",)]

    def fake_train(
        train_rows: list[exp4244.GrownPoolRow],
        *,
        folds: list[tuple[str, ...]],
        random_seed: int,
        bootstrap_n: int,
        hidden_dim: int,
        training_epochs: int,
        lr: float,
    ) -> SimpleNamespace:
        assert train_rows == rows
        assert folds == [("task0",), ("task1",)]
        assert (random_seed, bootstrap_n, hidden_dim, training_epochs, lr) == (5160, 0, 8, 3, 0.01)
        return SimpleNamespace(auroc=0.75, rows=oof_rows)

    monkeypatch.setattr(exp4244, "split_task_folds", fake_split)
    monkeypatch.setattr(exp4244, "train_oof_set_encoder", fake_train)
    corpus = exp4244.GrownPoolCorpus(
        rows=rows,
        pool_artifact_path=Path("fixture.json.gz"),
        pool_artifact_sha256="fixture",
        upstream_checksum="",
        held_out_task_n=2,
        wrong_majority_n=1,
        positive_candidate_n=2,
    )
    result = mod._train_seed_replication(
        corpus,
        random_seed=5160,
        n_folds=2,
        bootstrap_n=0,
        training_epochs=3,
        hidden_dim=8,
        lr=0.01,
    )
    assert result.auroc == 0.75
    assert result.delta == 1.0
    assert result.oof_rows == oof_rows


def test_req_5160_leak_audit_and_run_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-5160-UPSTREAM-BLOCKED: invalid inputs block before scoring."""

    original = {"tasks": [_task("gap3_stage2:50a16a69", "50a16a69", "gap3_stage2", correct_index=1)]}
    collision_row = exp4244.GrownPoolRow(
        task_id="gap3_stage2:50a16a69",
        candidate_id="gap3_stage2:50a16a69::candidate0",
        candidate_index=0,
        correct=True,
        features=_features(1.0, 1.0),
        vote_weight=1.0,
    )
    selection = mod.SecondPoolSelection(
        corpus=exp4244.GrownPoolCorpus(
            rows=[collision_row],
            pool_artifact_path=Path("fixture.json.gz"),
            pool_artifact_sha256="fixture",
            upstream_checksum="",
            held_out_task_n=1,
            wrong_majority_n=0,
            positive_candidate_n=1,
        ),
        source_rel=Path("fixture.json.gz"),
        source_sha256="fixture",
        source_kind="fixture",
        classic_arc_static_puzzle_pool=False,
        preferred_audit={},
        adapter={"dropped_overlap_candidate_n": 0},
    )
    leak = mod.second_pool_leak_audit(
        original,
        selection,
        [
            mod.SeedReplicationResult(
                random_seed=5160,
                auroc=0.5,
                delta=0.0,
                held_out_task_n=1,
                candidate_count=1,
                vote_at_1=1.0,
                set_encoder_at_1=1.0,
                oracle_at_k=1.0,
                task_deltas=[0.0],
                oof_rows=[
                    exp4244.OOFRow(
                        task_id="gap3_stage2:50a16a69",
                        candidate_id="gap3_stage2:50a16a69::candidate0",
                        correct=True,
                        score=1.0,
                        fold=0,
                        train_task_ids=("gap3_stage2:50a16a69",),
                    )
                ],
            )
        ],
    )
    assert leak["passed"] is False
    assert leak["heldout_training_task_collision_count"] == 1

    _write_5160_fixture(tmp_path)
    _write_json(
        tmp_path / mod.EXP5151_REL,
        {
            "experiment": "experiment_5151_arc_oracle_distinct_hardening_v472",
            "verifier_is_oracle": True,
        },
    )
    blocked = mod.run(tmp_path, adversarial_runner=_adversarial_clean)
    assert blocked["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT

    _write_5160_fixture(tmp_path)
    _patch_seed_results(monkeypatch)
    too_few_seeds = mod.run(
        tmp_path,
        random_seeds=[5160, 5161, 5162, 5163],
        adversarial_runner=_adversarial_clean,
    )
    assert too_few_seeds["honest_verdict"] == mod.BLOCKED_SECOND_POOL_VERDICT


def test_req_5160_validation_and_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-5160: artifact validation rejects non-actionable fields."""

    _write_5160_fixture(tmp_path)
    _patch_seed_results(monkeypatch, deltas=[0.0, 0.02, -0.01, 0.01, 0.0])
    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)
    assert artifact["cross_corpus_replication_passed"] is False
    assert artifact["diffusiongemma_gate_updated_recommendation"] == "keep_gated"

    invalid_cases = [
        ({key: value for key, value in artifact.items() if key != "cross_corpus_delta"}, "missing"),
        ({**artifact, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**artifact, "game_id_misnomer_confirmed": {"value": True}}, "bare bool"),
        ({**artifact, "second_pool_leak_audit_passed": "true"}, "bare bool"),
        ({**artifact, "cross_corpus_delta": True}, "bare float"),
        ({**artifact, "cross_corpus_delta_ci95": [0.0]}, "ci95"),
        ({**artifact, "cross_corpus_replication_passed": 1}, "bare bool"),
        ({**artifact, "diffusiongemma_gate_updated_recommendation": "maybe"}, "recommendation"),
        ({**artifact, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**artifact, "solve_provenance": "live"}, "solve_provenance"),
        ({**artifact, "random_seeds_used": [5160.0]}, "bare ints"),
        ({**artifact, "reproducibility_checksum": "nope"}, "sha256"),
        ({**artifact, "field_principles": {}}, "field_principles"),
        ({**artifact, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    assert mod._multiseed_ci95([]) == [0.0, 0.0]
    assert mod._multiseed_ci95([0.5]) == [0.5, 0.5]
    assert mod._mean([]) == 0.0
    assert mod._safe_float(True) == 0.0
    assert mod._safe_float("bad") == 0.0
    assert mod._clean_adversarial_report({"returncode": 1, "reports": [{"flags": [{"kind": "X"}]}]})[
        "status"
    ] == "flagged"
    assert mod._clean_adversarial_report({"reports": [None, {"flags": [None]}]})["status"] == "clean"
