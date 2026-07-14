"""Tests for Exp 5600 PTRM Stage-1 multi-seed, pre-registered leave-one-game-out gate.

Spec refs: REQ-ARC-PTRM-5600-1, REQ-ARC-PTRM-5600-2,
SCENARIO-ARC-PTRM-5600-WIRING-FIX, SCENARIO-ARC-PTRM-5600-LOO-GATE.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5600_ptrm_loo_gate as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-trm-generator" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_ptrm_5600_spec_declares_wiring_fix_and_loo_gate_contract() -> None:
    """REQ-ARC-PTRM-5600-1/2: OpenSpec declares both the wiring-fix regression
    contract and the multi-seed pre-registered LOO gate contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-PTRM-5600-1") :]

    for marker in (
        "REQ-ARC-PTRM-5600-1",
        "REQ-ARC-PTRM-5600-2",
        "SCENARIO-ARC-PTRM-5600-WIRING-FIX",
        "SCENARIO-ARC-PTRM-5600-LOO-GATE",
        "ft09",
        "m0r0",
        "vc33",
        "sk48",
        "cd82",
        "Wilcoxon",
        "retire_if_same_verdict",
    ):
        assert marker in section


def test_scenario_arc_ptrm_5600_blocked_precondition_never_trains(monkeypatch) -> None:
    """A missing precondition fails closed without running any combination."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda corpus_dir=mod.DEFAULT_CORPUS_DIR, require_cuda=True: {
            "corpus_manifest_present": False,
            "cuda_available": True,
            "cuda_detail": "n/a",
            "ok": False,
        },
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("_run_one_combination must not run when blocked")

    monkeypatch.setattr(mod, "_run_one_combination", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["loo_verdict_reached"] is False
    assert artifact["per_game_results"] == {}
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_ptrm_5600_majority_action_and_accuracy_helpers() -> None:
    """Deterministic unit checks for the small pure-Python scoring helpers."""

    example_a = mod.Stage1Example(
        game="g",
        guid="a",
        start_step=1,
        frame_features=[0.0] * 4,
        history_actions=[0, 0, 0],
        history_coords=[(-1, -1)] * 3,
        history_intent_vector=[0.0] * 4,
        target_actions=[1, 1, 2, 2],
        target_coords=[(-1, -1)] * 4,
    )
    example_b = mod.Stage1Example(
        game="g",
        guid="b",
        start_step=1,
        frame_features=[0.0] * 4,
        history_actions=[0, 0, 0],
        history_coords=[(-1, -1)] * 3,
        history_intent_vector=[0.0] * 4,
        target_actions=[1, 1, 1, 2],
        target_coords=[(-1, -1)] * 4,
    )

    assert mod._majority_action([example_a, example_b]) == 1
    assert mod._majority_action([]) == 0

    predictions = [[1, 1, 2, 2], [1, 1, 1, 1]]
    accuracy, exact = mod._accuracy_and_exact_match(predictions, [example_a, example_b])
    # example_a: 4/4 correct, exact match; example_b: 3/4 correct, not exact
    assert accuracy == round(7 / 8, 6)
    assert exact == 0.5

    baseline_preds = mod._majority_baseline_predictions(1, [example_a, example_b])
    assert baseline_preds == [[1, 1, 1, 1], [1, 1, 1, 1]]


def test_scenario_arc_ptrm_5600_wilcoxon_and_gate_majority_logic(monkeypatch) -> None:
    """SCENARIO-ARC-PTRM-5600-LOO-GATE: the majority-of-5-games gate is computed
    from the per-game significance + baseline comparisons, not post-hoc."""

    monkeypatch.setattr(
        mod, "preconditions", lambda **_kwargs: {"ok": True, "cuda_available": True}
    )
    monkeypatch.setattr(mod, "_load_corpus", lambda corpus_dir: ([], {}))
    monkeypatch.setattr(mod, "build_stage1_dataset", lambda rows, config, heldout_games: object())

    # 3 of 5 games: PTRM clearly and significantly beats both non-recursive and
    # majority baseline (alternating small deltas so Wilcoxon has variance).
    winning_games = {"ft09", "m0r0", "vc33"}

    def _fake_run_one_combination(*, bundle, game, seed_index, require_cuda):
        del bundle, require_cuda
        if game in winning_games:
            ptrm = 0.60 + 0.01 * (seed_index % 2)
            non_recursive = 0.40 + 0.01 * (seed_index % 3)
            majority = 0.30
        else:
            ptrm = 0.30 + 0.01 * (seed_index % 2)
            non_recursive = 0.30 + 0.01 * (seed_index % 3)
            majority = 0.35
        return {
            "game": game,
            "seed_index": seed_index,
            "n_train_examples": 10,
            "n_heldout_examples": 5,
            "leakage_count": 0,
            "ptrm_per_action_accuracy": round(ptrm, 6),
            "ptrm_exact_window_accuracy": 0.0,
            "non_recursive_per_action_accuracy": round(non_recursive, 6),
            "non_recursive_exact_window_accuracy": 0.0,
            "majority_baseline_per_action_accuracy": majority,
            "majority_baseline_exact_window_accuracy": 0.0,
        }

    monkeypatch.setattr(mod, "_run_one_combination", _fake_run_one_combination)

    artifact = mod.build_artifact(n_seeds=10)

    assert set(artifact["games_ptrm_beats_non_recursive_significantly"]) >= winning_games
    assert set(artifact["games_ptrm_beats_majority_baseline"]) >= winning_games
    assert artifact["loo_verdict_reached"] is True
    assert artifact["honest_verdict"] == "complete: ptrm_loo_gate_passed_majority_of_heldout_games"
    assert artifact["heldout_generalization_signal"] == "loo_gate_passed_majority_5_games"
    assert artifact["retire_trm_generator_line"] is False
    assert "exp5574_artifact_mismatch_corrigendum" in artifact
    assert "70c857a69" in artifact["exp5574_artifact_mismatch_corrigendum"]
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_ptrm_5600_gate_fails_honestly_when_no_majority(monkeypatch) -> None:
    """When fewer than 3 of 5 games clear the gate, the verdict is an honest
    failure and `retire_trm_generator_line` is set, per known-issues.md task 8's
    `retire_if_same_verdict: true` condition."""

    monkeypatch.setattr(
        mod, "preconditions", lambda **_kwargs: {"ok": True, "cuda_available": True}
    )
    monkeypatch.setattr(mod, "_load_corpus", lambda corpus_dir: ([], {}))
    monkeypatch.setattr(mod, "build_stage1_dataset", lambda rows, config, heldout_games: object())

    def _fake_run_one_combination(*, bundle, game, seed_index, require_cuda):
        del bundle, require_cuda
        return {
            "game": game,
            "seed_index": seed_index,
            "n_train_examples": 10,
            "n_heldout_examples": 5,
            "leakage_count": 0,
            "ptrm_per_action_accuracy": 0.3,
            "ptrm_exact_window_accuracy": 0.0,
            "non_recursive_per_action_accuracy": 0.3,
            "non_recursive_exact_window_accuracy": 0.0,
            "majority_baseline_per_action_accuracy": 0.5,
            "majority_baseline_exact_window_accuracy": 0.0,
        }

    monkeypatch.setattr(mod, "_run_one_combination", _fake_run_one_combination)

    artifact = mod.build_artifact(n_seeds=4)

    assert artifact["games_ptrm_beats_non_recursive_significantly"] == []
    assert artifact["loo_verdict_reached"] is True
    assert artifact["honest_verdict"] == (
        "complete: ptrm_loo_gate_failed_no_majority_significant_and_above_baseline"
    )
    assert artifact["retire_trm_generator_line"] is True


def test_req_arc_ptrm_5600_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-PTRM-5600-2: the checked-in real run measured all 5 held-out games
    across 10 seeds each, using the wiring-fixed generation path. The gate
    failed (only ft09 clears both bars, below the required majority of 3), so
    task 8's `retire_if_same_verdict: true` condition fires."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["heldout_games"] == list(mod.HELDOUT_GAMES)
    assert result["n_seeds"] == mod.N_SEEDS
    assert result["loo_verdict_reached"] is True
    assert result["solve_provenance"] == "development_proxy"
    assert result["inference_substrate"] == "trained_ptrm_offline_development_proxy"
    assert result["verifier_is_oracle"] is False
    assert result["no_level_solve_claim"] is True
    assert set(result["per_game_results"].keys()) == set(mod.HELDOUT_GAMES)
    for game in mod.HELDOUT_GAMES:
        game_result = result["per_game_results"][game]
        assert game_result["n_seeds"] == mod.N_SEEDS
        assert len(game_result["per_seed_rows"]) == mod.N_SEEDS
        assert game_result["per_seed_rows"][0]["leakage_count"] == 0
    assert result["honest_verdict"] == (
        "complete: ptrm_loo_gate_failed_no_majority_significant_and_above_baseline"
    )
    assert result["games_ptrm_beats_non_recursive_significantly"] == ["ft09"]
    assert result["retire_trm_generator_line"] is True
    assert result["duration_s"] > 0.0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
