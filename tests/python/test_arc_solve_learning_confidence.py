"""Tests for the FinAcumen-inspired (arXiv:2606.17642) routing refinements in arc_solve_learning:
a confidence threshold (skip-below -> don't blind-transfer a misleading recipe) + aggregated
failure CAUTIONS. Every test asserts (Tests-Must-Run-and-Assert). Backward-compat: existing keys
remain present.
"""

from __future__ import annotations

from carnot.agentic import arc_solve_learning as L


def test_confident_transfer_threshold_constant() -> None:
    # the bar is at least an action-type match (+3 on the _similarity scale)
    assert L._CONFIDENT_TRANSFER_MIN_SIM == 3.0


def test_similarity_action_match_clears_bar() -> None:
    a = {"action_type": "click", "spatial": True, "difficulty": "med", "win_kw": {"drag"}}
    b = {"action_type": "click", "spatial": True, "difficulty": "med", "win_kw": {"drag"}}
    assert L._similarity(a, b) >= L._CONFIDENT_TRANSFER_MIN_SIM
    # no shared action model + nothing else -> below the bar
    c = {"action_type": "keyboard", "spatial": False, "difficulty": "", "win_kw": set()}
    assert L._similarity(a, c) < L._CONFIDENT_TRANSFER_MIN_SIM


def test_cautions_aggregates_dedups_and_caps() -> None:
    reg = {
        "games": [
            {"game": "g1", "dead_ends": ["avoid X", "avoid X"]},  # dup within
            {"game": "g2", "dead_ends_recorded": [{"note": "avoid Y"}]},
        ],
        "general_gotchas": ["general Z", "avoid X"],  # dup across sources
    }
    ranked = [{"game": "g1"}, {"game": "g2"}]
    cautions = L._cautions_from(ranked, reg)
    assert "avoid X" in cautions and "avoid Y" in cautions and "general Z" in cautions
    assert cautions.count("avoid X") == 1  # deduped
    assert len(cautions) <= 8  # capped


def test_recommend_approach_emits_new_fields_and_keeps_old(monkeypatch) -> None:
    # known game present in the survey -> full routing path
    rec = L.recommend_approach("tr87")
    if "error" in rec:  # unseen-branch shape still carries the new fields
        for k in ("confident_transfer", "routing_confidence", "cautions"):
            assert k in rec
        return
    # backward-compat: original keys still there
    for k in ("recommended", "strategy", "general_gotchas", "guidance"):
        assert k in rec
    # new FinAcumen fields
    assert isinstance(rec["confident_transfer"], bool)
    assert 0.0 <= rec["routing_confidence"] <= 1.0
    assert isinstance(rec["cautions"], list)
    # guidance reflects the confidence decision
    if rec["confident_transfer"]:
        assert "CONFIDENT transfer" in rec["guidance"]
    else:
        assert "LOW-confidence" in rec["guidance"]


def test_unseen_live_game_is_not_a_confident_transfer() -> None:
    # a game NOT in the public survey (the live held-out case) must never claim a confident transfer
    rec = L.recommend_approach("zz99_definitely_unseen")
    assert rec.get("confident_transfer") is False
    assert rec.get("routing_confidence") == 0.0
    assert "cautions" in rec
