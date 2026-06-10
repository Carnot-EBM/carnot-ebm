import json
import sys
from pathlib import Path

import numpy as np

from carnot.agentic.arc_execution_guided_world_model import (
    ExecutionGuidedProgram,
    exact_replay_report,
    induce_execution_guided,
    select_consistent_transitions,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_3979_world_model_gen_execution_guided as exp


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _toggle(pos, h=5, w=5):
    s = np.zeros((h, w), dtype=np.int16)
    s[pos] = 2
    s2 = s.copy()
    s2[pos] = 5
    return s, s2


def _predict_toggle(grid, action):
    out = np.asarray(grid, dtype=np.int16).copy()
    if len(action) == 3 and action[0] == 6:
        out[action[2], action[1]] = 5
    return out


def test_req_phase4_017_spec_declares_execution_guided_sweep():
    """REQ-PHASE4-017: OpenSpec declares Exp 3979 before implementation."""
    spec = SPEC_PATH.read_text("utf-8")

    assert "REQ-PHASE4-017" in spec
    assert "SCENARIO-PHASE4-017" in spec
    assert "positive_control_passed" in spec
    assert "per_game_best_energy" in spec


def test_req_phase4_017_rejects_conflicting_hidden_state_observations():
    """REQ-PHASE4-017: accepted programs exactly replay observed transitions."""
    s, s2 = _toggle((1, 1))
    s3 = s.copy()
    s3[1, 1] = 7
    accepted, rejected = select_consistent_transitions(
        [(s, (6, 1, 1), s2), (s, (6, 1, 1), s2), (s, (6, 1, 1), s3)]
    )
    program = ExecutionGuidedProgram("toy", "toggle", accepted, _predict_toggle)

    assert len(accepted) == 2
    assert len(rejected) == 1
    assert exact_replay_report(program.predict, accepted)["all_exact"] is True
    assert exact_replay_report(program.predict, rejected)["all_exact"] is False


def test_req_phase4_017_inducer_selects_heldout_consistent_program():
    """REQ-PHASE4-017: synthesis keeps exact-replay programs and grades held-out prediction energy."""
    train = []
    for pos in [(0, 0), (1, 1), (2, 2)]:
        s, s2 = _toggle(pos)
        train.append((s, (6, pos[1], pos[0]), s2))
    held_s, held_s2 = _toggle((3, 3))

    def noop(grid, action):
        return np.asarray(grid, dtype=np.int16).copy()

    result = induce_execution_guided(
        "toy",
        train,
        [(held_s, (6, 3, 3), held_s2)],
        max_synthesis_iters=2,
        extra_predictors=[("noop", noop), ("toggle", _predict_toggle)],
    )

    assert result["best_program"] == "toggle"
    assert result["best_energy"] == 0.0
    assert result["total_synthesis_calls"] == 2
    assert result["accepted_train_count"] == 3
    assert result["history"][0]["train_replay_exact"] is True


def test_req_phase4_017_program_falls_back_on_bad_predictor_outputs():
    """REQ-PHASE4-017: unsafe candidate execution cannot bypass exact replay checks."""
    s, s2 = _toggle((1, 1))

    def crashes(grid, action):
        raise RuntimeError("bad candidate")

    def wrong_shape(grid, action):
        return np.zeros((1, 1), dtype=np.int16)

    crash_program = ExecutionGuidedProgram("toy", "crash", [], crashes)
    shape_program = ExecutionGuidedProgram("toy", "shape", [], wrong_shape)

    assert np.array_equal(crash_program.predict(s, (6, 1, 1)), s)
    assert np.array_equal(shape_program.predict(s, (6, 1, 1)), s)
    assert exact_replay_report(crash_program.predict, [(s, (6, 1, 1), s2)])["all_exact"] is False


def test_scenario_phase4_017_blocks_when_positive_control_fails(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-017: failing vc33 positive control writes a blocked artifact."""
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_select_game_ids", lambda arc, games: {"vc33": "vc33-full", "r11l": "r11l-full"})
    monkeypatch.setattr(exp, "_load_cached_vc33_predictor", lambda: ("bad_vc33", lambda grid, action: grid))

    s, s2 = _toggle((1, 1))
    held_s, held_s2 = _toggle((2, 2))
    monkeypatch.setattr(exp, "_collect_train_and_heldout", lambda *args: ([(s, (6, 1, 1), s2)], [(held_s, (6, 2, 2), held_s2)]))

    artifact = exp.run(games=["r11l"], iters=1, write=True, _arc_client=object())

    assert artifact["honest_verdict"] == "blocked_positive_control_failed"
    assert artifact["positive_control_passed"] is False
    assert artifact["n_trustworthy_at_0.15"] == 0
    assert artifact["per_game_best_energy"] == {}
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text("utf-8"))["honest_verdict"] == "blocked_positive_control_failed"


def test_scenario_phase4_017_success_artifact_compares_exp3968(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-017: completed sweep reports trustworthiness, baseline delta, and hidden split."""
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_select_game_ids", lambda arc, games: {"vc33": "vc33-full", "r11l": "r11l-full", "sc25": "sc25-full"})
    monkeypatch.setattr(exp, "_load_cached_vc33_predictor", lambda: ("good_vc33", _predict_toggle))
    monkeypatch.setattr(exp, "_load_hidden_state_games", lambda: {"sc25"})
    monkeypatch.setattr(
        exp,
        "_load_exp3968_baseline",
        lambda: (0, {"r11l": 0.7752, "sc25": 0.6155}),
    )

    s, s2 = _toggle((1, 1))
    held_s, held_s2 = _toggle((2, 2))
    monkeypatch.setattr(exp, "_collect_train_and_heldout", lambda *args: ([(s, (6, 1, 1), s2)], [(held_s, (6, 2, 2), held_s2)]))

    calls = []

    def fake_induce(game, train, held, **kwargs):
        calls.append(game)
        energies = {"vc33": 0.04, "r11l": 0.10, "sc25": 0.30}
        return {
            "best_energy": energies[game],
            "best_program": f"{game}_program",
            "history": [{"program": f"{game}_program", "train_replay_exact": True}],
            "total_synthesis_calls": 2,
            "total_synthesis_seconds": 1.25,
            "accepted_train_count": len(train),
            "rejected_conflict_count": 0,
        }

    monkeypatch.setattr(exp, "induce_execution_guided", fake_induce)

    artifact = exp.run(games=["r11l", "sc25"], write=True, _arc_client=object())

    assert calls == ["vc33", "r11l", "sc25"]
    assert artifact["positive_control_passed"] is True
    assert artifact["honest_verdict"] == "success: exec_guided_trustworthy_1of2"
    assert artifact["n_trustworthy_at_0.15"] == 1
    assert artifact["beats_exp3968"] is True
    assert artifact["per_game_best_energy"] == {"r11l": 0.10, "sc25": 0.30}
    assert artifact["total_synthesis_calls"] == 6
    assert artifact["total_synthesis_seconds"] == 3.75
    assert artifact["markov_vs_hidden_split"]["determinism_probe_hidden_state"] == ["sc25"]
    assert artifact["per_game"][0]["exp3968_best_energy"] == 0.7752
    assert "trustworthiness" in artifact["caveat"]
