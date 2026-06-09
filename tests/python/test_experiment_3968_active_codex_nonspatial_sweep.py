import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_3968_active_codex_nonspatial_sweep as exp


class MockEnv:
    def __init__(self, game_id):
        self.game_id = game_id


class MockArcade:
    def __init__(self, game_ids):
        self._game_ids = game_ids

    def get_environments(self):
        return [MockEnv(game_id) for game_id in self._game_ids]


def test_blocked_codex_unavailable_writes_required_schema(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-008-3968: absent codex blocks honestly but keeps artifact schema."""
    monkeypatch.setattr(exp, "REPO", tmp_path)

    art = exp.run(games=["r11l"], write=True, _codex_available=False)

    assert art["honest_verdict"] == "blocked_codex_unavailable"
    assert art["n_trustworthy_at_0.15"] == 0
    assert art["per_game_best_energy"] == {}
    assert art["total_codex_calls"] == 0
    assert art["total_codex_seconds"] == 0.0
    assert art["markov_vs_hidden_split"]["determinism_probe_markov"] == []
    written = tmp_path / "results" / "experiment_3968_active_codex_nonspatial_sweep.json"
    assert json.loads(written.read_text())["honest_verdict"] == "blocked_codex_unavailable"


def test_blocked_arc_offline_env_unavailable(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-008-3968: unavailable offline ARC env blocks without synthesis."""
    monkeypatch.setattr(exp, "REPO", tmp_path)

    def fail_load():
        raise RuntimeError("offline env missing")

    monkeypatch.setattr(exp, "_load_offline_arcade", fail_load)

    art = exp.run(games=["r11l"], write=True, _codex_available=True)

    assert art["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert art["total_codex_calls"] == 0
    written = tmp_path / "results" / "experiment_3968_active_codex_nonspatial_sweep.json"
    assert json.loads(written.read_text())["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_empty_offline_env_list_blocks(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-008-3968: an empty offline env listing is not treated as success."""
    monkeypatch.setattr(exp, "REPO", tmp_path)

    art = exp.run(
        games=["r11l"],
        write=False,
        _codex_available=True,
        _arc_client=MockArcade([]),
    )

    assert art["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_success_contract_bounded_iters_and_hidden_split(monkeypatch, tmp_path):
    """REQ-PHASE4-008: active codex sweep reports per-game energy and hidden-state context."""
    monkeypatch.setattr(exp, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_determinism_probe.json").write_text(
        json.dumps({"hidden_state_games": ["sc25", "dc22"]}) + "\n",
        "utf-8",
    )
    client = MockArcade(["r11l-495a7899", "sc25-64edb92b", "other-0000"])
    grid = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    transition = (grid, (6, 1, 0), grid.copy())

    monkeypatch.setattr(exp, "_collect", lambda *args: [transition, transition])
    monkeypatch.setattr(exp, "active_collect", lambda *args: [transition])
    monkeypatch.setattr(exp, "_common_test", lambda *args: [transition])
    monkeypatch.setattr(exp, "_keys", lambda transitions: {("seen", (6, 1, 0))})

    seen_iters = []

    def fake_codex_best_energy(train, test, iters, rng):
        seen_iters.append(iters)
        if len(seen_iters) == 1:
            return 0.10, [{"status": "graded", "codex_s": 1.2}], 1.2
        return 0.25, [{"status": "graded", "codex_s": 2.0}, {"status": "no_code", "codex_s": 0.4}], 2.4

    monkeypatch.setattr(exp, "codex_best_energy", fake_codex_best_energy)

    art = exp.run(
        games=["r11l", "sc25"],
        iters=7,
        write=True,
        _codex_available=True,
        _arc_client=client,
    )

    assert seen_iters == [3, 3]
    assert art["honest_verdict"] == "complete: exp3968_active_codex_nonspatial_sweep_trustworthy_1of2"
    assert art["n_trustworthy_at_0.15"] == 1
    assert art["per_game_best_energy"] == {"r11l": 0.10, "sc25": 0.25}
    assert art["total_codex_calls"] == 3
    assert art["total_codex_seconds"] == 3.6
    assert art["markov_vs_hidden_split"]["determinism_probe_markov"] == ["r11l"]
    assert art["markov_vs_hidden_split"]["determinism_probe_hidden_state"] == ["sc25"]
    assert art["markov_vs_hidden_split"]["energy_trustworthy_low"] == ["r11l"]
    assert art["markov_vs_hidden_split"]["energy_high_or_missing"] == ["sc25"]
    assert "predicts transitions" in art["caveat"]
    assert art["per_game"][1]["diff_from_vc33_0.005"] == 0.245
    written = tmp_path / "results" / "experiment_3968_active_codex_nonspatial_sweep.json"
    assert json.loads(written.read_text())["per_game_best_energy"]["sc25"] == 0.25


def test_default_games_are_the_six_nonspatial_targets():
    """REQ-PHASE4-008: Exp 3968 defaults to the six surveyed non-spatial games."""
    assert exp.DEFAULT_GAMES == ("r11l", "sc25", "lp85", "tn36", "dc22", "su15")


def test_missing_determinism_probe_defaults_to_no_hidden_games(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-008-3968: hidden-state cross-reference tolerates missing prior probe."""
    monkeypatch.setattr(exp, "REPO", tmp_path)

    assert exp._load_hidden_state_games() == set()
