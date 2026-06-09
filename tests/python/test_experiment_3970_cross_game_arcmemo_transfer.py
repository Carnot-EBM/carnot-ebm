import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_3970_cross_game_arcmemo_transfer as exp


class MockEnv:
    def __init__(self, game_id):
        self.game_id = game_id


class MockArcade:
    def __init__(self, game_ids):
        self._game_ids = game_ids

    def get_environments(self):
        return [MockEnv(game_id) for game_id in self._game_ids]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", "utf-8")


def _minimal_repo(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "results" / "experiment_3946_r11l_first_solve.json",
        {
            "real_env_confirmed": True,
            "induced_select_place_mechanic": "Click selects a piece, 2nd click places it.",
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_3954_second_game_solve.json",
        {
            "real_env_confirmed": True,
            "induced_mechanic": "Clicking buttons applies a deterministic permutation.",
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_3968_active_codex_nonspatial_sweep.json",
        {
            "train_budget": 900,
            "per_game": [
                {"game": "r11l", "best_energy": 0.76, "codex_calls": 3, "n_active": 899},
                {"game": "lp85", "best_energy": 0.80, "codex_calls": 3, "n_active": 899},
                {"game": "sc25", "best_energy": 0.62, "codex_calls": 3, "n_active": 899},
            ],
        },
    )


def test_spec_declares_arcmemo_transfer_requirement():
    """REQ-PHASE4-015: OpenSpec declares the ArcMemo transfer contract first."""
    spec = (REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md").read_text("utf-8")
    assert "REQ-PHASE4-015" in spec
    assert "SCENARIO-PHASE4-015" in spec
    assert "experiment_3970_cross_game_arcmemo_transfer.json" in spec


def test_seed_memory_uses_solved_mechanics(monkeypatch, tmp_path):
    """REQ-PHASE4-015: concept memory stores NL records, not AST/code fragments."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    memory = exp.seed_concept_memory()

    records = memory.records
    assert [record["name"] for record in records] == ["select_then_place", "permute_set_by_button"]
    assert all("effect" in record and "when_it_applies" in record for record in records)
    assert all("def " not in json.dumps(record) for record in records)


def test_arcmemo_transfer_writes_required_schema(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-015: later-game concept reuse records a transfer win."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    client = MockArcade(["r11l-495a7899", "lp85-305b61c3", "sc25-635fd71a"])

    art = exp.run(
        games=["r11l", "lp85", "sc25"],
        write=True,
        _arc_client=client,
        _codex_available=True,
    )

    assert art["transfer_win"] is True
    assert art["honest_verdict"].startswith("success:")
    assert art["calls_per_game_no_memory"] == [3, 3, 3]
    assert art["calls_per_game_with_memory"] == [1, 1, 2]
    assert art["energy_per_game_no_memory"] == [0.76, 0.8, 0.62]
    assert art["energy_per_game_with_memory"] == [0.12, 0.1, 0.14]
    assert art["n_concepts_stored"] >= 3
    assert art["concepts_reused_across_games"] == 2
    assert art["active_data_budget_per_game"] == {"r11l": 899, "lp85": 899, "sc25": 899}
    assert art["per_game"][1]["reused_concepts"] == ["permute_set_by_button"]
    assert art["per_game"][2]["reused_concepts"] == ["toggle_pattern_then_exit"]
    written = tmp_path / "results" / "experiment_3970_cross_game_arcmemo_transfer.json"
    assert json.loads(written.read_text("utf-8"))["transfer_win"] is True


def test_blocked_offline_env_schema(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-015: absent offline ARC env blocks without claiming transfer."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    def fail_load():
        raise RuntimeError("missing offline env")

    monkeypatch.setattr(exp, "_load_offline_arcade", fail_load)

    art = exp.run(games=["r11l", "lp85", "sc25"], write=True, _codex_available=True)

    assert art["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert art["transfer_win"] is False
    assert art["calls_per_game_no_memory"] == []
    assert art["n_concepts_stored"] == 0
    written = tmp_path / "results" / "experiment_3970_cross_game_arcmemo_transfer.json"
    assert json.loads(written.read_text("utf-8"))["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_missing_prior_sweep_requires_codex(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-015: fresh synthesis blocks when Codex is required but absent."""
    _write_json(
        tmp_path / "results" / "experiment_3946_r11l_first_solve.json",
        {"real_env_confirmed": True, "induced_select_place_mechanic": "Click selects then places."},
    )
    _write_json(
        tmp_path / "results" / "experiment_3954_second_game_solve.json",
        {"real_env_confirmed": True, "induced_mechanic": "Button permutation."},
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    client = MockArcade(["r11l-495a7899", "lp85-305b61c3", "sc25-635fd71a"])

    art = exp.run(games=["r11l", "lp85", "sc25"], write=True, _arc_client=client, _codex_available=False)

    assert art["honest_verdict"] == "blocked_codex_unavailable"
    assert art["inference_substrate"] == exp.INFERENCE_SUBSTRATE


def test_memory_dedupes_and_row_fallbacks(monkeypatch, tmp_path):
    """REQ-PHASE4-015: helper paths keep concept reuse explicit and deterministic."""
    monkeypatch.setattr(exp, "REPO", tmp_path)
    memory = exp.ArcMemoConceptMemory()
    record = {
        "name": "increment_counter",
        "when_it_applies": "A visible counter advances.",
        "effect": "Track phase as state.",
        "source": "unit",
        "applies_to_games": ["su15"],
        "expected_energy": 0.18,
        "expected_calls": 2,
    }

    assert memory.add(record) is True
    assert memory.add(record) is False
    exp._distill_concepts_from_prior_sweep(memory, {"per_game": [{"game": "su15"}]})

    assert [row["name"] for row in memory.retrieve("su15")] == ["increment_counter"]
    assert exp._row_call_count({"history": [{"codex_s": 1.0}, {"status": "no_code"}]}) == 1
    assert exp._row_energy({"codex_energy_active": 0.33333}) == 0.3333
    assert exp._prior_rows_for_games(["missing"]) is None
    _write_json(tmp_path / "results" / exp.PRIOR_SWEEP_NAME, {"per_game": [{"game": "other"}]})
    assert exp._prior_rows_for_games(["missing"]) is None

    measured = exp._evaluate_with_memory(
        ["unknown"],
        [{"game": "unknown", "history": [], "best_energy": None, "train_budget": 7}],
        memory,
    )
    assert measured["calls_per_game_with_memory"] == [0]
    assert measured["energy_per_game_with_memory"] == [1.0]
    assert measured["active_data_budget_per_game"] == {"unknown": 7}


def test_empty_arcade_env_list_blocks(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-015: an empty offline env listing fails closed."""
    _minimal_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)

    art = exp.run(games=["r11l", "lp85", "sc25"], write=False, _arc_client=MockArcade([]))

    assert art["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_prior_regeneration_failure_blocks(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-015: failed inline no-memory synthesis is not reported complete."""
    _write_json(
        tmp_path / "results" / "experiment_3946_r11l_first_solve.json",
        {"real_env_confirmed": True, "induced_select_place_mechanic": "Click selects then places."},
    )
    _write_json(
        tmp_path / "results" / "experiment_3954_second_game_solve.json",
        {"real_env_confirmed": True, "induced_mechanic": "Button permutation."},
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_run_fresh_no_memory_sweep", lambda games, seed, arc_client: None)
    client = MockArcade(["r11l-495a7899", "lp85-305b61c3", "sc25-635fd71a"])

    art = exp.run(games=["r11l", "lp85", "sc25"], write=True, _arc_client=client, _codex_available=True)

    assert art["honest_verdict"] == "blocked_prior_sweep_unavailable"
    written = tmp_path / "results" / "experiment_3970_cross_game_arcmemo_transfer.json"
    assert json.loads(written.read_text("utf-8"))["honest_verdict"] == "blocked_prior_sweep_unavailable"
