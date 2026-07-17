"""Tests for Exp 5714 rescoped /think vs /no_think induction-quality A/B.

Spec refs: REQ-ARC-WMTE-5714, SCENARIO-ARC-WMTE-5714-CODEONLY-TOGGLE-INERT,
SCENARIO-ARC-WMTE-5714-GENUINE-THINK-ENGAGES, SCENARIO-ARC-WMTE-5714-BLOCKS-CLEANLY.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5714_think_mode_rescoped_ab as mod

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5714_spec_declares_rescoped_contract() -> None:
    """REQ-ARC-WMTE-5714: OpenSpec declares the rescoped A/B contract, names the prior it
    re-tests, and its three scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5714") :]
    for marker in (
        "REQ-ARC-WMTE-5714",
        "REQ-ARC-WMTE-5594",  # names the prior it re-tests
        "SCENARIO-ARC-WMTE-5714-CODEONLY-TOGGLE-INERT",
        "SCENARIO-ARC-WMTE-5714-GENUINE-THINK-ENGAGES",
        "SCENARIO-ARC-WMTE-5714-BLOCKS-CLEANLY",
        "_L2_CODEONLY_DIRECTIVE",
        "levelup_positive_recall",
        "codeonly_toggle_inert",
    ):
        assert marker in section, marker


def test_scenario_5714_blocked_precondition_never_runs(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5714-BLOCKS-CLEANLY: a missing precondition fails closed without
    building any window or attempting any induction call."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "adapters_available": True,
            "e3_import": True,
            "gguf_cached": False,  # the miss
            "cuda_llama_server_present": True,
            "cuda_server_up_or_gpu1_headroom": True,
            "ok": False,
        },
    )

    def _fail(*_a, **_k):
        raise AssertionError("no window/induction may run when a precondition is missing")

    monkeypatch.setattr(mod, "build_levelup_window", _fail)
    monkeypatch.setattr(mod, "run_arm", _fail)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["per_game_results"] == []
    assert artifact["codeonly_toggle_inert"] is None
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_verdict_inert_toggle_still_reports_genuine_reasoning_signal() -> None:
    """SCENARIO-ARC-WMTE-5714-CODEONLY-TOGGLE-INERT: when the frozen toggle is inert, the
    verdict leads with the inertness (operator's literal decision = no-op) but STILL encodes
    the genuine-reasoning (A1-vs-B2) win-recognition signal -- the compound headline."""

    comp_a = {"per_game_headtohead_think_tie_nothink": {"levelup_positive_recall": [0, 5, 0]}}
    comp_gen_win = {"per_game_headtohead_think_tie_nothink": {"levelup_positive_recall": [4, 1, 1]}}
    v = mod._verdict(comp_a, comp_gen_win, inert=True)
    assert v.startswith("complete: think_toggle_inert_under_codeonly_but_")
    assert "genuine_reasoning_improves_winrecognition_1_to_4" in v

    comp_gen_flat = {
        "per_game_headtohead_think_tie_nothink": {"levelup_positive_recall": [0, 6, 0]}
    }
    v2 = mod._verdict(comp_a, comp_gen_flat, inert=True)
    assert (
        v2
        == "complete: think_toggle_inert_under_codeonly_but_genuine_reasoning_no_winrecognition_delta"
    )


def test_verdict_reports_toggle_direction_when_not_inert() -> None:
    """When the toggle is NOT inert (it produced reasoning), the verdict reports the
    Comparison-A win-recognition direction AND the genuine-reasoning signal, terminal-prefixed."""

    comp_gen = {"per_game_headtohead_think_tie_nothink": {"levelup_positive_recall": [0, 3, 0]}}
    win_think = {"per_game_headtohead_think_tie_nothink": {"levelup_positive_recall": [4, 1, 1]}}
    assert mod._verdict(win_think, comp_gen, inert=False).startswith(
        "complete: think_toggle_higher_winrecognition_"
    )
    win_nothink = {"per_game_headtohead_think_tie_nothink": {"levelup_positive_recall": [1, 1, 4]}}
    assert mod._verdict(win_nothink, comp_gen, inert=False).startswith(
        "complete: no_think_higher_winrecognition_"
    )
    tie = {"per_game_headtohead_think_tie_nothink": {"levelup_positive_recall": [2, 3, 2]}}
    assert mod._verdict(tie, comp_gen, inert=False).startswith("complete: think_toggle_null_but_")


def test_summarize_pair_counts_headtohead_and_reason_engaged() -> None:
    """_summarize_pair counts per-game think/tie/no_think wins on the decision-relevant
    metrics and reports each arm's reasoning-engagement fraction."""

    rows = [
        {
            "game": "g1",
            "arm": "A1",
            "induction_ok": True,
            "reason_engaged": False,
            "goal_predicate_accuracy": 0.9,
            "levelup_positive_recall": 0.0,
            "heldout_accuracy": 0.0,
        },
        {
            "game": "g1",
            "arm": "A2",
            "induction_ok": True,
            "reason_engaged": False,
            "goal_predicate_accuracy": 0.9,
            "levelup_positive_recall": 1.0,
            "heldout_accuracy": 0.0,
        },
        {
            "game": "g2",
            "arm": "A1",
            "induction_ok": True,
            "reason_engaged": False,
            "goal_predicate_accuracy": 0.8,
            "levelup_positive_recall": 1.0,
            "heldout_accuracy": 0.0,
        },
        {
            "game": "g2",
            "arm": "A2",
            "induction_ok": True,
            "reason_engaged": False,
            "goal_predicate_accuracy": 0.8,
            "levelup_positive_recall": 1.0,
            "heldout_accuracy": 0.0,
        },
    ]
    s = mod._summarize_pair(rows, "A1", "A2")
    assert s["n_games_both_induced"] == 2
    # levelup_positive_recall: g1 think>no_think (1.0>0.0 -> think_win), g2 tie -> [1,1,0]
    assert s["per_game_headtohead_think_tie_nothink"]["levelup_positive_recall"] == [1, 1, 0]
    assert s["think_reason_engaged_frac"] == 0.0
    assert s["no_think_induction_success"] == 2


def test_select_levelup_window_ends_at_last_levelup() -> None:
    """_select_levelup_window returns a window that ENDS at (and includes) the LAST real
    level-up transition, satisfying score_goal_predicate_consistency's single-boundary caller
    contract. Pure logic -- no live env, no LLM."""

    import numpy as np
    from carnot.agentic.arc_executable_world_model import Transition

    g = np.zeros((2, 2), dtype=int)

    def t(level_before, level_after):
        return Transition(g, 1, None, g, level_before, level_after)

    # 6 no-ops at L0, then a real L0->L1 level-up at index 6, then 2 more at L1
    trans = [t(0, 0)] * 6 + [t(0, 1)] + [t(1, 1)] * 2
    window = mod._select_levelup_window(trans, k=4)
    assert window is not None
    assert len(window) == 4  # k transitions ending at the level-up
    assert window[-1].level_after > window[-1].level_before  # ends at the level-up
    assert all(w.level_after <= 1 for w in window)  # single boundary (no L1->L2 mixed in)

    # no level-up anywhere -> None (that game is skipped, goal-predicate can't fire)
    assert mod._select_levelup_window([t(0, 0)] * 5, k=4) is None


def test_req_arc_wmte_5714_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-WMTE-5714: the checked-in run is a real, live, non-blocked measurement over a
    materially-larger-than-2 roster, with the goal-predicate metric actually exercised and the
    mechanistic codeonly_toggle_inert finding recorded -- not a fabricated or blocked stub."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert result["honest_verdict"].startswith("complete:")
    assert not result["honest_verdict"].startswith("complete: blocked_")
    assert result["inference_substrate"] == "live_llm_inference"
    assert result["solve_provenance"] == "development_proxy"
    assert result["mtp_enabled"] is True
    assert result["duration_s"] > 60.0
    assert isinstance(result["codeonly_toggle_inert"], bool)
    # materially larger than the prior's N=2: count games with at least one successful A-arm induce
    a_games = {
        r["game"]
        for r in result["per_game_results"]
        if r.get("arm") in ("A1", "A2") and r.get("induction_ok")
    }
    assert len(a_games) >= 6
    # the goal-predicate half (the prior's biggest gap) actually fired at least once
    assert any(
        isinstance(r.get("levelup_positive_recall"), (int, float))
        for r in result["per_game_results"]
    )
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
