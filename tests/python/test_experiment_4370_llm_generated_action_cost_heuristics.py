"""Tests for Exp 4370 LLM-generated ARC action-cost heuristics.

Spec refs: REQ-LEARN-4370, SCENARIO-LEARN-4370,
SCENARIO-LEARN-4370-BLOCKED.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot import experiment_4370_llm_generated_action_cost_heuristics as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


SAFE_SRC = """
def h(state):
    grid = state.get("grid", [])
    cells = [cell for row in grid for cell in row]
    nonzero = sum(1 for cell in cells if cell != 0)
    colors = len(set(cells))
    return float(nonzero + colors)
"""


def _candidate(game: str, name: str, source: str = SAFE_SRC) -> dict[str, Any]:
    return {"game": game, "name": name, "source": source}


def _row(
    game: str,
    level_id: str,
    *,
    split: str,
    linear: int = 10,
    llm: int = 8,
    bfs: int = 12,
    reproduced: bool = True,
) -> dict[str, Any]:
    return {
        "game": game,
        "level_id": level_id,
        "split": split,
        "linear_actions": linear,
        "bfs_baseline_actions": bfs,
        "candidate_metrics": {
            "safe_a": {"actions": llm + 1, "expansions": 20, "reproduced": reproduced},
            "safe_b": {"actions": llm, "expansions": 30, "reproduced": reproduced},
            "leaky": {"actions": 1, "expansions": 1, "reproduced": True},
        },
        "reproduce_result": {
            "game": game,
            "claimed_level": int(level_id.rsplit("L", 1)[-1]),
            "reached_level": int(level_id.rsplit("L", 1)[-1]),
            "reproduced": reproduced,
        },
    }


def test_req_learn_4370_spec_declares_llm_heuristic_contract() -> None:
    """REQ-LEARN-4370: OpenSpec declares the 4370 artifact and gate fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-LEARN-4370",
        "SCENARIO-LEARN-4370",
        "SCENARIO-LEARN-4370-BLOCKED",
        "experiment_4370_llm_generated_action_cost_heuristics.json",
        "llm_heuristic_beats_linear",
        "held_out_actions_by_heuristic",
        "static_leakage_clean",
        "reproduction_gated",
        "blocked_solver_kit_unavailable",
    ):
        assert marker in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4370_static_leakage_analysis_rejects_cheating_programs() -> None:
    """REQ-LEARN-4370-3: leaky generated programs are dropped before scoring."""

    assert exp.static_leakage_report(SAFE_SRC)["clean"] is True

    leaky_sources = [
        "def h(state):\n    return state.env._game.secret_answer\n",
        "from carnot.agentic.arc_solver_kit import reproduce\n\ndef h(state):\n    return reproduce\n",
        "def h(state):\n    answer_cells = [(1, 2), (3, 4)]\n    return len(answer_cells)\n",
        "def h(state):\n    layout = [[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,1]]\n    return layout[0][0]\n",
    ]

    reports = [exp.static_leakage_report(source) for source in leaky_sources]

    assert all(report["clean"] is False for report in reports)
    assert any("env internal" in reason for report in reports for reason in report["reasons"])
    assert any("solver/reproduce" in reason for report in reports for reason in report["reasons"])
    assert any("answer/target" in reason for report in reports for reason in report["reasons"])
    assert any("hard-coded layout" in reason for report in reports for reason in report["reasons"])


def test_req_learn_4370_generation_writes_three_clean_programs_per_eligible_game() -> None:
    """REQ-LEARN-4370-2/3: Codex-generated per-game programs are clean."""

    generated = exp.generate_candidate_programs(["lp85", "tr87"])

    assert sorted(generated) == ["lp85", "tr87"]
    for game, candidates in generated.items():
        assert len(candidates) >= 3
        assert all(candidate["game"] == game for candidate in candidates)
        assert all(candidate["source"].count("def h(state):") == 1 for candidate in candidates)
        assert all(exp.static_leakage_report(candidate["source"])["clean"] for candidate in candidates)


def test_req_learn_4370_selection_scores_only_clean_training_candidates() -> None:
    """REQ-LEARN-4370-4: GBFS selection ignores leaky programs and keeps best clean."""

    leaky_source = "def h(state):\n    return state.env._game.answer_grid\n"
    candidates = [
        _candidate("alpha", "safe_a"),
        _candidate("alpha", "safe_b"),
        _candidate("alpha", "leaky", leaky_source),
    ]
    training_rows = [
        _row("alpha", "alpha:L1", split="train", llm=7),
        _row("alpha", "alpha:L2", split="train", llm=6),
    ]

    selected = exp.select_by_training_gbfs(candidates, training_rows)

    assert selected["alpha"]["name"] == "safe_b"
    assert selected["alpha"]["training_actions"] == 13
    assert selected["alpha"]["training_expansions"] == 60
    assert selected["alpha"]["static_leakage_clean"] is True
    assert selected["alpha"]["dropped_candidates"] == ["leaky"]


def test_scenario_learn_4370_complete_artifact_schema_and_gap_logging(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4370: complete artifacts preserve bare gates and gaps."""

    held_out = [_row("alpha", f"alpha:L{i}", split="held_out") for i in range(1, 5)]
    held_out.extend(_row("beta", f"beta:L{i}", split="held_out", linear=9, llm=7) for i in range(1, 5))
    selected = {
        "alpha": {"name": "safe_b", "source": SAFE_SRC, "static_leakage_clean": True},
        "beta": {"name": "safe_b", "source": SAFE_SRC, "static_leakage_clean": True},
    }

    artifact = exp.build_complete_artifact(
        training_rows=[_row("alpha", "alpha:L0", split="train")],
        held_out_rows=held_out,
        selected_by_game=selected,
        preconditions_checked={"usable_reproduced_level_count": 8},
        duration_s=0.25,
        adversarial_verify={"status": "clean", "returncode": 0, "flagged_count": 0},
    )

    assert artifact["honest_verdict"] == "success: llm_generated_heuristic_beats_linear_76_to_60"
    assert artifact["llm_heuristic_beats_linear"] is True
    assert artifact["held_out_actions_by_heuristic"] == {
        "linear": 76,
        "llm_generated": 60,
        "bfs_baseline": 96,
    }
    assert artifact["n_held_out_levels"] == 8
    assert artifact["static_leakage_clean"] is True
    assert artifact["reproduction_gated"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert exp.artifact_schema_errors(artifact) == []

    null_rows = [_row("alpha", f"alpha:L{i}", split="held_out", llm=10) for i in range(1, 9)]
    null_artifact = exp.build_complete_artifact(
        training_rows=[],
        held_out_rows=null_rows,
        selected_by_game={"alpha": {"name": "safe_b", "source": SAFE_SRC, "static_leakage_clean": True}},
        preconditions_checked={"usable_reproduced_level_count": 8},
        duration_s=0.1,
    )
    assert null_artifact["honest_verdict"] == "complete: clean_powered_null_linear_not_beaten"
    assert null_artifact["llm_heuristic_beats_linear"] is False
    assert null_artifact["missing_verifier_gaps"]

    exp.ensure_gap_logged(tmp_path, null_artifact)
    gap_text = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert exp.GAP_ID in gap_text
    assert "alpha:L1" in gap_text
    exp.ensure_gap_logged(tmp_path, null_artifact)
    assert (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8") == gap_text


def test_scenario_learn_4370_blocked_artifact_and_schema_fail_closed() -> None:
    """SCENARIO-LEARN-4370-BLOCKED: missing resources emit bare false gates."""

    artifact = exp.build_blocked_artifact(
        verdict="blocked_solver_kit_unavailable",
        usable_levels=["alpha:L1"],
        preconditions_checked={"solver_kit_importable": False},
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "blocked_solver_kit_unavailable"
    assert artifact["llm_heuristic_beats_linear"] is False
    assert artifact["held_out_actions_by_heuristic"] == {
        "linear": 0,
        "llm_generated": 0,
        "bfs_baseline": 0,
    }
    assert artifact["static_leakage_clean"] is False
    assert artifact["reproduction_gated"] is False
    assert artifact["n_held_out_levels"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert exp.artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad["llm_heuristic_beats_linear"] = 1
    bad["held_out_actions_by_heuristic"] = {"linear": "0"}
    bad["per_game_scorecard"] = {}
    bad["static_leakage_clean"] = "false"
    bad["reproduction_gated"] = None
    bad["n_held_out_levels"] = 8.0
    bad["verifier_is_oracle"] = True
    bad["field_principles"] = {**exp.FIELD_PRINCIPLES, "honest_verdict": "wrong"}

    errors = exp.artifact_schema_errors(bad)

    for field in (
        "llm_heuristic_beats_linear",
        "held_out_actions_by_heuristic",
        "per_game_scorecard",
        "static_leakage_clean",
        "reproduction_gated",
        "n_held_out_levels",
        "verifier_is_oracle",
        "field_principles mismatch for honest_verdict",
    ):
        assert any(field in error for error in errors)
