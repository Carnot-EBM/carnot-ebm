"""Tests for Exp 5151 ARC Set-Encoder oracle-distinct hardening.

Spec refs: REQ-VERIFY-5151, SCENARIO-VERIFY-5151,
SCENARIO-VERIFY-5151-CROSS-GAME-BLOCKED,
SCENARIO-VERIFY-5151-UPSTREAM-BLOCKED.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_oracle_distinct_hardening_5151 as mod
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {"returncode": 0, "reports": [{"flag_count": 0, "flags": [], "max_severity": 0}]}


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


def _write_hardening_fixture(
    root: Path,
    *,
    include_game_ids: bool = False,
    leak_candidate_id: bool = False,
) -> Path:
    task_ids = [f"mini:task-{index}" for index in range(8)]
    tasks: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    task_rows_4245: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(task_ids):
        correct_index = 1 if task_index < 6 else 0
        game_id = "game-a" if task_index < 4 else "game-b"
        candidates: list[dict[str, Any]] = []
        for candidate_index in range(2):
            candidate_id = f"{task_id}::candidate{candidate_index}"
            is_correct = candidate_index == correct_index
            score = 0.9 if is_correct else 0.1
            candidate = {
                "candidate_grid_hash": f"gridhash-{task_index}-{candidate_index}",
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "features": _features(0.9 if candidate_index == 0 else 0.1, confidence=score),
                "grid": [[task_index, candidate_index]],
                "is_correct": is_correct,
                "source_kinds": ["gold_flag"] if is_correct else ["pool_candidate"],
                "votes": 9.0 if candidate_index == 0 else 1.0,
            }
            candidates.append(candidate)
            row: dict[str, Any] = {
                "candidate_id": candidate_id,
                "correct": is_correct,
                "fold": task_index % 2,
                "score": score,
                "task_id": task_id,
                "train_task_ids": [other for other in task_ids if other != task_id],
            }
            if leak_candidate_id and task_index == 0 and candidate_index == 0:
                row["train_candidate_ids"] = [candidate_id]
            oof_rows.append(row)
        task: dict[str, Any] = {
            "candidate_count": len(candidates),
            "candidates": candidates,
            "oracle_present": True,
            "raw_task_id": f"task-{task_index}",
            "source_id": "mini",
            "task_id": task_id,
            "vote_top_candidate_id": f"{task_id}::candidate0",
            "wrong_majority": correct_index != 0,
        }
        if include_game_ids:
            task["game_id"] = game_id
        tasks.append(task)
        vote_correct = correct_index == 0
        task_rows_4245.append(
            {
                "task_id": task_id,
                "oracle_hit": True,
                "vote_candidate_id": f"{task_id}::candidate0",
                "vote_correct": vote_correct,
                "set_encoder_candidate_id": f"{task_id}::candidate{correct_index}",
                "set_encoder_correct": True,
                "matched_control_candidate_id": f"{task_id}::candidate0",
                "matched_control_correct": vote_correct,
            }
        )

    pool_rel = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
    pool_path = root / pool_rel
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "candidate_n": 16,
                "positive_candidate_n": 8,
                "random_seed": 4243,
                "reproducibility_checksum": "sha256:" + "1" * 64,
                "schema": "carnot.arc_candidate_pool_grow.v1",
                "source_paths": [],
                "source_sha256": {},
                "spec_refs": ["REQ-CAPSTONE-4243"],
                "task_n": 8,
                "tasks": tasks,
                "wrong_majority_n": 6,
            },
            handle,
        )
    _write_json(
        root / "results" / "experiment_4243_arc_candidate_pool_grow.json",
        {
            "arc_pool_grown": True,
            "held_out_task_n": 8,
            "pool_artifact_path": str(pool_rel),
            "positive_candidate_n": 8,
            "random_seed": 4243,
            "reproducibility_checksum": "sha256:" + "2" * 64,
            "verifier_is_oracle": False,
            "wrong_majority_n": 6,
        },
    )
    model_path = root / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
    _write_json(
        model_path,
        {
            "feature_names": list(exp4244.FEATURE_NAMES),
            "held_out_task_n": 8,
            "model": {"model_type": "fixture_oof_only"},
            "model_specs": {"architecture": "fixture_set_encoder", "training_epochs": 0},
            "pool_artifact_path": str(pool_path),
            "pool_artifact_sha256": "fixture",
            "random_seed": 4244,
            "reproducibility_checksum": "sha256:" + "3" * 64,
            "set_encoder_oof": {
                "auroc": 1.0,
                "ci95": [1.0, 1.0],
                "fold_task_ids": [task_ids[:4], task_ids[4:]],
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
            "held_out_task_n": 8,
            "honest_verdict": "complete: fixture",
            "learned_verifier_path": str(model_path),
            "model_specs": {"architecture": "fixture_set_encoder", "training_epochs": 0},
            "oracle_distinct_auroc": 1.0,
            "random_seed": 4244,
            "reproducibility_checksum": "sha256:" + "4" * 64,
            "set_encoder_vs_logistic_auroc_delta": 0.1,
            "verifier_is_oracle": False,
            "wrong_majority_n": 6,
        },
    )
    _write_json(
        root / "results" / "experiment_4245_arc_set_encoder_beats_vote.json",
        {
            "honest_verdict": "complete: arc_oracle_distinct_set_encoder_beats_vote",
            "held_out_task_n": 8,
            "learned_verifier_path": str(model_path),
            "model_specs": {"architecture": "fixture_set_encoder"},
            "oracle_at_k": 1.0,
            "pass_rates": {"vote_at_1": 0.25, "set_encoder_at_1": 1.0},
            "set_encoder_minus_vote_ci95": [0.5, 1.0],
            "set_encoder_minus_vote_delta": 0.75,
            "task_rows": task_rows_4245,
            "verifier_is_oracle": False,
        },
    )
    return root


def _patch_seed_results(monkeypatch: pytest.MonkeyPatch, deltas: list[float] | None = None) -> None:
    seed_deltas = deltas or [0.72, 0.74, 0.76, 0.78, 0.80]

    def fake_seed(*_args: object, random_seed: int, **_kwargs: object) -> mod.SeedHardeningResult:
        index = mod.DEFAULT_RANDOM_SEEDS.index(random_seed)
        delta = seed_deltas[index]
        return mod.SeedHardeningResult(
            random_seed=random_seed,
            auroc=0.8 + index * 0.01,
            delta=delta,
            held_out_task_n=8,
            vote_at_1=0.25,
            set_encoder_at_1=0.25 + delta,
            oracle_at_k=1.0,
            fold_task_ids=[[f"mini:fold-{index}"]],
        )

    monkeypatch.setattr(mod, "_train_seed_hardening", fake_seed)


def test_req_5151_spec_declares_hardening_contract() -> None:
    """REQ-VERIFY-5151: OpenSpec declares the hardening artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-5151",
        "SCENARIO-VERIFY-5151",
        "SCENARIO-VERIFY-5151-CROSS-GAME-BLOCKED",
        "SCENARIO-VERIFY-5151-UPSTREAM-BLOCKED",
        "python/carnot/reporting/arc_oracle_distinct_hardening_5151.py",
        "results/experiment_5151_arc_oracle_distinct_hardening_v472.json",
        "blocked_upstream_artifact_missing",
        "blocked_arc_game_ids_unrecoverable",
        "multiseed_delta_ci95",
        "leak_audit_passed",
        "cross_game_replication_delta",
        "solve_provenance=development_proxy",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_5151_hardening_survives_all_available_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-5151: multiseed, leak audit, exact test, and transfer pass."""

    _write_hardening_fixture(tmp_path, include_game_ids=True)
    _patch_seed_results(monkeypatch)
    monkeypatch.setattr(
        mod,
        "_run_cross_game_check",
        lambda *_args, **_kwargs: {
            "cross_game_replication_delta": 0.5,
            "cross_game_replication_ci95": [0.25, 0.75],
            "cross_game_blocked_reason": None,
            "cross_game_honest_read": "cross_game_transfers",
            "held_out_game_n": 2,
        },
    )

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean, bootstrap_resamples=200)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("success_arc_set_encoder_win_survives_hardening")
    assert artifact["multiseed_delta_ci95"][0] > 0.0
    assert artifact["per_seed_results"][0]["auroc"] == pytest.approx(0.8)
    assert artifact["leak_audit_passed"] is True
    assert artifact["leak_audit"]["task_id_collision_count"] == 0
    assert artifact["cross_game_replication_delta"] == pytest.approx(0.5)
    assert artifact["exact_test_discordant_wins"] == 6
    assert artifact["exact_test_discordant_losses"] == 0
    assert artifact["exact_test_p_value"] < 0.05
    assert artifact["exact_test_passes_min6_rule"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["adversarial_verify"]["status"] == "clean"
    assert (tmp_path / mod.OUTPUT_REL).exists()


def test_scenario_5151_cross_game_blocked_does_not_skip_multiseed_or_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-5151-CROSS-GAME-BLOCKED: transfer blocks only that sub-claim."""

    _write_hardening_fixture(tmp_path, include_game_ids=False)
    _patch_seed_results(monkeypatch)

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean, bootstrap_resamples=200)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete_arc_set_encoder_win_not_hardened")
    assert artifact["multiseed_delta_ci95"][0] > 0.0
    assert artifact["leak_audit_passed"] is True
    assert artifact["exact_test_passes_min6_rule"] is True
    assert artifact["cross_game_replication_delta"] is None
    assert artifact["cross_game_blocked_reason"] == mod.BLOCKED_GAME_IDS_VERDICT
    assert artifact["hardening_axes"]["cross_game"] == "blocked"


def test_scenario_5151_upstream_missing_blocks_honestly(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5151-UPSTREAM-BLOCKED: missing Exp 4245 or pool stops."""

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert artifact["multiseed_delta_ci95"] == [0.0, 0.0]
    assert artifact["leak_audit_passed"] is False
    assert artifact["cross_game_replication_delta"] is None
    assert artifact["random_seeds_used"] == []
    assert artifact["verifier_is_oracle"] is False


def test_scenario_5151_row_level_leak_collision_fails_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-5151: held-out candidate surrogates cannot appear in training signal."""

    _write_hardening_fixture(tmp_path, include_game_ids=False, leak_candidate_id=True)
    _patch_seed_results(monkeypatch)

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean, bootstrap_resamples=200)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete_arc_set_encoder_win_not_hardened")
    assert artifact["leak_audit_passed"] is False
    assert artifact["leak_audit"]["candidate_signal_collision_count"] == 1
    assert artifact["hardening_axes"]["leak_audit"] == "failed"


def test_req_5151_exact_test_and_validation_edges() -> None:
    """REQ-VERIFY-5151: exact-test helpers and schema validation are deterministic."""

    assert mod._two_sided_binomial_p(6, 0) == pytest.approx(0.03125)
    assert mod._two_sided_binomial_p(3, 3) == pytest.approx(1.0)
    assert mod._cluster_bootstrap_ci95([1.0, 1.0], random_seed=1, resamples=10) == [1.0, 1.0]
    blocked = mod._blocked_artifact(
        mod.BLOCKED_UPSTREAM_VERDICT,
        random_seed=5151,
        checksum="sha256:" + "0" * 64,
        duration_s=0.01,
    )
    invalid_cases = [
        (
            {key: value for key, value in blocked.items() if key != "multiseed_delta_ci95"},
            "missing",
        ),
        ({**blocked, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**blocked, "leak_audit_passed": {"value": False}}, "bare bool"),
        ({**blocked, "cross_game_replication_delta": "blocked"}, "bare float or null"),
        ({**blocked, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**blocked, "random_seeds_used": [5151.0]}, "bare ints"),
        ({**blocked, "multiseed_delta_ci95": [0.0]}, "ci95"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)


def test_req_5151_helper_edges_and_tiny_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-5151: helpers cover defensive paths without touching the real pool."""

    assert mod._safe_float(True) == 0.0
    assert mod._safe_float("nan-ish") == 0.0
    assert mod._safe_float(float("nan")) == 0.0
    assert mod._safe_int(True) == 0
    assert mod._safe_int("not-int") == 0
    assert mod._t_critical_975(99) == pytest.approx(1.96)
    assert mod._multiseed_ci95([]) == [0.0, 0.0]
    assert mod._multiseed_ci95([0.3]) == [0.3, 0.3]
    assert mod._cluster_bootstrap_ci95([], random_seed=1, resamples=10) == [0.0, 0.0]
    assert mod._cluster_bootstrap_ci95([0.4], random_seed=1, resamples=10) == [0.4, 0.4]
    assert mod._cluster_bootstrap_ci95([1.0, -1.0], random_seed=1, resamples=0) == [0.0, 0.0]
    assert mod._two_sided_binomial_p(0, 0) == 1.0
    assert mod._strings_from({"a": ["x", {"b": ("y",)}]}) == {"x", "y"}

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BlockedRun):
        mod._read_json_object(bad_json)
    with pytest.raises(mod.BlockedRun):
        mod._read_json_object(tmp_path / "missing.json")
    bad_gzip = tmp_path / "bad.json.gz"
    with gzip.open(bad_gzip, "wt", encoding="utf-8") as handle:
        json.dump([], handle)
    with pytest.raises(mod.BlockedRun):
        mod._read_gzip_json_object(bad_gzip)
    with pytest.raises(mod.BlockedRun):
        mod._read_gzip_json_object(tmp_path / "missing.json.gz")
    with pytest.raises(mod.BlockedRun):
        mod._resolve_required_path(tmp_path, None)
    with pytest.raises(mod.BlockedRun):
        mod._resolve_required_path(tmp_path, "missing")

    invalid_root = _write_hardening_fixture(tmp_path / "invalid-inputs")
    build_path = invalid_root / mod.SET_ENCODER_BUILD_REL
    build = json.loads(build_path.read_text(encoding="utf-8"))
    _write_json(build_path, {**build, "aggregator_trained": False})
    with pytest.raises(mod.BlockedRun):
        mod._load_required_inputs(invalid_root)
    _write_json(build_path, build)
    model_path = Path(build["learned_verifier_path"])
    model = json.loads(model_path.read_text(encoding="utf-8"))
    _write_json(model_path, {**model, "verifier_is_oracle": True})
    with pytest.raises(mod.BlockedRun):
        mod._load_required_inputs(invalid_root)
    _write_json(model_path, model)
    exp4245_path = invalid_root / mod.EXP4245_REL
    exp4245 = json.loads(exp4245_path.read_text(encoding="utf-8"))
    _write_json(exp4245_path, {**exp4245, "verifier_is_oracle": True})
    with pytest.raises(mod.BlockedRun):
        mod._load_required_inputs(invalid_root)
    _write_json(exp4245_path, exp4245)
    with gzip.open(invalid_root / mod.POOL_REL, "wt", encoding="utf-8") as handle:
        json.dump({"tasks": "bad"}, handle)
    with pytest.raises(mod.BlockedRun):
        mod._load_required_inputs(invalid_root)

    payload = {
        "tasks": [
            None,
            {},
            {"task_id": "empty", "candidates": "bad"},
            {
                "task_id": "mini:t0",
                "raw_task_id": "t0",
                "candidates": [
                    None,
                    {
                        "candidate_id": "mini:t0::candidate0",
                        "candidate_grid_hash": "gold-hash",
                        "grid": [[1]],
                        "is_correct": True,
                    },
                ],
            },
        ]
    }
    surrogate = mod._task_surrogates(payload)["mini:t0"]
    assert "t0" in surrogate["task"]
    assert "gold-hash" in surrogate["gold"]

    audit = mod.row_level_leak_audit(
        payload,
        {
            "set_encoder_oof": {
                "rows": [
                    "bad-row",
                    {"task_id": "unknown", "candidate_id": "x", "train_task_ids": []},
                    {
                        "task_id": "mini:t0",
                        "candidate_id": "mini:t0::candidate0",
                        "train_task_ids": ["mini:t0"],
                        "train_gold_grid_hashes": ["gold-hash"],
                        "training_signal": {"nested": ["t0"]},
                    },
                ]
            }
        },
    )
    assert audit["passed"] is False
    assert audit["task_excluded_row_count"] == 0
    assert audit["task_id_collision_count"] == 2
    assert audit["gold_signal_collision_count"] == 1
    assert (
        mod.row_level_leak_audit(payload, {"set_encoder_oof": {"rows": "bad"}})["passed"] is False
    )

    assert mod._task_deltas_from_exp4245({"task_rows": "bad"}) == ([], 0, 0)
    assert mod._task_deltas_from_exp4245(
        {"task_rows": ["bad", {"vote_correct": True, "set_encoder_correct": False}]}
    ) == ([-1.0], 0, 1)
    assert (
        mod._task_metrics_from_scores(
            [mod.CleanCandidate("t", "c0", 0, 1.0, True)],
            {},
        )["held_out_task_n"]
        == 0
    )

    rows: list[exp4244.GrownPoolRow] = []
    for task_index in range(4):
        task_id = f"tiny:{task_index}"
        correct_index = task_index % 2
        for candidate_index in range(2):
            rows.append(
                exp4244.GrownPoolRow(
                    task_id=task_id,
                    candidate_id=f"{task_id}::candidate{candidate_index}",
                    candidate_index=candidate_index,
                    correct=candidate_index == correct_index,
                    features=_features(
                        0.8 if candidate_index == 0 else 0.2,
                        0.9 if candidate_index == correct_index else 0.1,
                    ),
                    vote_weight=0.8 if candidate_index == 0 else 0.2,
                )
            )
    corpus = exp4244.GrownPoolCorpus(
        rows=rows,
        pool_artifact_path=tmp_path / "pool.json.gz",
        pool_artifact_sha256="sha256:" + "1" * 64,
        upstream_checksum="sha256:" + "2" * 64,
        held_out_task_n=4,
        wrong_majority_n=2,
        positive_candidate_n=4,
    )
    seed_result = mod._train_seed_hardening(
        corpus,
        random_seed=9,
        n_folds=2,
        bootstrap_n=0,
        training_epochs=0,
        hidden_dim=4,
        lr=0.01,
    )
    assert seed_result.random_seed == 9
    assert seed_result.held_out_task_n == 4

    _write_hardening_fixture(tmp_path / "game", include_game_ids=True)

    def fake_cross_train(
        cross_corpus: mod.exp4258.GameAnnotatedCorpus,
        folds: list[mod.exp4258.GameFold],
        **_kwargs: object,
    ) -> mod.exp4258.CrossGameTrainingReport:
        oof_rows: list[exp4244.OOFRow] = []
        for fold_index, fold in enumerate(folds):
            for row in cross_corpus.rows:
                if row.task_id in fold.held_out_task_ids:
                    oof_rows.append(
                        exp4244.OOFRow(
                            task_id=row.task_id,
                            candidate_id=row.candidate_id,
                            correct=row.correct,
                            score=0.9 if row.correct else 0.1,
                            fold=fold_index,
                            train_task_ids=tuple(sorted(fold.train_task_ids)),
                        )
                    )
        return mod.exp4258.CrossGameTrainingReport(
            rows=oof_rows,
            fold_summaries=[],
            training_config={"fixture": True},
        )

    monkeypatch.setattr(mod.exp4258, "train_cross_game_oof", fake_cross_train)
    cross = mod._run_cross_game_check(
        tmp_path / "game",
        random_seed=5151,
        n_folds=2,
        bootstrap_resamples=100,
        training_epochs=0,
        hidden_dim=4,
        lr=0.01,
    )
    assert cross["cross_game_replication_delta"] > 0.0
    assert cross["cross_game_blocked_reason"] is None
    assert mod._hardening_axes(
        multiseed_passed=False,
        leak_audit_passed=False,
        exact_test_passed=False,
        cross_game={
            "cross_game_replication_delta": 0.0,
            "cross_game_replication_ci95": [-0.1, 0.1],
            "cross_game_blocked_reason": None,
        },
    ) == {
        "multiseed": "failed",
        "leak_audit": "failed",
        "exact_test": "failed",
        "cross_game": "failed",
    }

    short_seed = mod.run(
        tmp_path / "short-seed", random_seeds=[1], adversarial_runner=_adversarial_clean
    )
    assert short_seed["honest_verdict"] == "blocked_arc_hardening_requires_at_least_5_seeds"

    blocked = mod._blocked_artifact(
        mod.BLOCKED_UPSTREAM_VERDICT,
        random_seed=5151,
        checksum="sha256:" + "0" * 64,
        duration_s=0.01,
    )
    more_invalid_cases = [
        ({**blocked, "solve_provenance": "live"}, "solve_provenance"),
        ({**blocked, "inference_substrate": ""}, "inference_substrate"),
        ({**blocked, "per_seed_results": {}}, "per_seed_results"),
        ({**blocked, "exact_test_discordant_wins": 0.0}, "exact_test_discordant_wins"),
        ({**blocked, "exact_test_p_value": True}, "exact_test_p_value"),
        ({**blocked, "cross_game_blocked_reason": 123}, "cross_game_blocked_reason"),
        ({**blocked, "field_principles": {}}, "field_principles"),
        ({**blocked, "spec_refs": []}, "spec_refs"),
        ({**blocked, "cross_game_replication_ci95": [0.0]}, "cross_game_replication_ci95"),
    ]
    for payload, message in more_invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)
