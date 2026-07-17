"""Unit tests for the retrieval-based playbook injection (REQ-ARC-WMTE-5718):
the pattern records, the retrieval scoring (cosine + mechanic-tag relevance), the offline
index load/round-trip, the AST kit-reference builder, the agent-side retrieval gate, and the
3-arm A/B's pure aggregation/verdict logic. No GPU/LLM is exercised here.

Spec: REQ-ARC-WMTE-5718, SCENARIO-ARC-WMTE-5718.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic import arc_playbook_retrieval as rag
from carnot.agentic.arc_playbook_patterns import MECHANIC_TAGS, playbook_patterns


# --------------------------------------------------------------------------
# patterns + tag taxonomy
# --------------------------------------------------------------------------
def test_patterns_are_nonempty_and_tags_valid():
    patterns = [p.as_dict() for p in playbook_patterns()]
    assert len(patterns) >= 20
    assert rag.validate_mechanic_tags(patterns) == []  # all tags in the taxonomy
    # every statement is game-agnostic: no per-game id / raw color-N token
    for p in patterns:
        low = p["statement"].lower()
        for banned in ("bp35", "lp85", "wa30", "color-15", "color 15"):
            assert banned not in low


def test_validate_mechanic_tags_flags_bad():
    assert rag.validate_mechanic_tags([{"mechanic_tags": ["navigation", "made_up_tag"]}]) == [
        "made_up_tag"
    ]


def test_all_taxonomy_tags_are_strings():
    assert all(isinstance(t, str) for t in MECHANIC_TAGS)


# --------------------------------------------------------------------------
# infer_query_mechanic_tags
# --------------------------------------------------------------------------
def test_infer_tags_from_public_game_id():
    tags = rag.infer_query_mechanic_tags(game="sk48")
    assert "chain_sort" in tags and "universal" in tags


def test_infer_tags_from_mechanic_class_keywords():
    tags = rag.infer_query_mechanic_tags(mechanic_class="goal_directed_navigation_obstacle")
    assert "navigation" in tags and "universal" in tags


def test_infer_tags_hidden_game_defaults_to_universal_only():
    tags = rag.infer_query_mechanic_tags(game="zz99_unknown_hidden")
    assert tags == ("universal",)


# --------------------------------------------------------------------------
# tag relevance + cosine
# --------------------------------------------------------------------------
def test_tag_relevance_on_mechanic_universal_off():
    assert rag._tag_relevance(("navigation", "camera_scroll"), ("navigation", "universal")) == 1.0
    assert (
        rag._tag_relevance(("universal",), ("navigation", "universal"))
        == rag._UNIVERSAL_TAG_RELEVANCE
    )
    assert rag._tag_relevance(("chain_sort",), ("navigation", "universal")) == 0.0
    assert rag._tag_relevance((), ("navigation",)) == 0.0


def test_cosine_scores_zero_query_is_zeros():
    m = np.eye(3, dtype=np.float32)
    assert np.array_equal(rag._cosine_scores(np.zeros(3), m), np.zeros(3, dtype=np.float32))


# --------------------------------------------------------------------------
# retrieve (in-memory index)
# --------------------------------------------------------------------------
def _toy_index():
    patterns = (
        {"pattern_id": "nav1", "statement": "nav a", "mechanic_tags": ["navigation"]},
        {"pattern_id": "uni1", "statement": "uni a", "mechanic_tags": ["universal"]},
        {"pattern_id": "chain1", "statement": "chain a", "mechanic_tags": ["chain_sort"]},
    )
    emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    return rag.PlaybookIndex(patterns=patterns, embeddings=emb, model="toy", dim=2)


def test_retrieve_ranks_by_cosine_plus_tag_boost():
    idx = _toy_index()
    # query aligned with [1,0]; nav1 and uni1 both near, but nav-tag boost lifts nav1 to #1.
    top = rag.retrieve(idx, np.array([1.0, 0.0]), top_k=2, query_tags=("navigation", "universal"))
    assert [r["pattern_id"] for r in top] == ["nav1", "uni1"]
    assert top[0]["tag_relevance"] == 1.0
    assert "score" in top[0] and "cosine" in top[0]


def test_retrieve_top_k_and_zero_k():
    idx = _toy_index()
    assert len(rag.retrieve(idx, np.array([1.0, 0.0]), top_k=1)) == 1
    assert rag.retrieve(idx, np.array([1.0, 0.0]), top_k=0) == []


def test_retrieve_off_mechanic_query_leaves_specific_pattern_low():
    idx = _toy_index()
    # a query aligned to chain, with navigation tag -> chain1 has high cosine, nav-boost helps nav1.
    top = rag.retrieve(idx, np.array([0.0, 1.0]), top_k=3, query_tags=("navigation", "universal"))
    ids = [r["pattern_id"] for r in top]
    assert "chain1" in ids  # cosine still surfaces it


# --------------------------------------------------------------------------
# format_injection
# --------------------------------------------------------------------------
def test_format_injection_empty_and_nonempty():
    assert rag.format_injection([]) == ""
    block = rag.format_injection([{"statement": "do X"}, {"statement": "do Y"}])
    assert block.startswith("RELEVANT EXPLORATION PRINCIPLES")
    assert "- do X" in block and "- do Y" in block
    assert block.endswith("\n\n")


# --------------------------------------------------------------------------
# load_index round-trip + errors
# --------------------------------------------------------------------------
def test_load_index_round_trip(tmp_path):
    patterns = [{"pattern_id": "a", "statement": "s", "mechanic_tags": ["universal"]}]
    (tmp_path / "index.json").write_text(json.dumps({"model": "m", "dim": 2, "patterns": patterns}))
    np.save(tmp_path / "embeddings.npy", np.array([[1.0, 2.0]], dtype=np.float32))
    idx = rag.load_index(tmp_path)
    assert len(idx) == 1 and idx.dim == 2 and idx.model == "m"


def test_load_index_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rag.load_index(tmp_path)


def test_load_index_length_mismatch_raises(tmp_path):
    (tmp_path / "index.json").write_text(
        json.dumps({"model": "m", "dim": 2, "patterns": [{"pattern_id": "a"}, {"pattern_id": "b"}]})
    )
    np.save(tmp_path / "embeddings.npy", np.array([[1.0, 2.0]], dtype=np.float32))
    with pytest.raises(ValueError):
        rag.load_index(tmp_path)


# --------------------------------------------------------------------------
# AST kit reference (exp5718)
# --------------------------------------------------------------------------
def test_build_kit_reference_extracts_signatures():
    import carnot.experiment_5718_playbook_index_build as b

    ref = b.build_kit_reference()
    names = {r["name"] for r in ref}
    assert "bounded_reachability_search" in names
    row = next(r for r in ref if r["name"] == "bounded_reachability_search")
    assert row["signature"].startswith("bounded_reachability_search(")
    assert row["doc"]  # non-empty one-line docstring, no body


# --------------------------------------------------------------------------
# agent-side retrieval gate + 3-mode prompt
# --------------------------------------------------------------------------
def test_agent_retrieval_gate_off_by_default(monkeypatch):
    monkeypatch.setattr(agent, "SUBMITTED_PLAYBOOK_RETRIEVAL_ENABLED", False)
    monkeypatch.delenv("CARNOT_ARC_PLAYBOOK_RETRIEVAL", raising=False)
    assert agent._playbook_retrieval_gate_on() is False


def test_agent_retrieval_gate_on_via_env(monkeypatch):
    monkeypatch.setattr(agent, "SUBMITTED_PLAYBOOK_RETRIEVAL_ENABLED", False)
    monkeypatch.setenv("CARNOT_ARC_PLAYBOOK_RETRIEVAL", "1")
    assert agent._playbook_retrieval_gate_on() is True


def test_induce_prompt_string_injection_mode():
    from carnot.agentic.arc_executable_world_model import Transition, induce_prompt

    g0 = np.zeros((3, 3), dtype=int)
    g1 = g0.copy()
    g1[0, 0] = 5
    t = Transition(g0, 1, None, g1, 0, 0)
    off = induce_prompt("gx", [t], 5)
    retrieved = induce_prompt("gx", [t], 5, include_playbook_exemplars="RETRIEVED BLOCK: - do X")
    assert retrieved.startswith("RETRIEVED BLOCK: - do X")
    assert retrieved == "RETRIEVED BLOCK: - do X\n\n" + off
    # empty string -> no injection (byte-identical)
    assert induce_prompt("gx", [t], 5, include_playbook_exemplars="   ") == off


# --------------------------------------------------------------------------
# 3-arm A/B pure logic (exp5719)
# --------------------------------------------------------------------------
def _ab():
    import carnot.experiment_5719_playbook_retrieval_ab as e

    return e


def _rows(none, static, retrieval):
    rows = []
    for arm, recalls in (("none", none), ("static", static), ("retrieval", retrieval)):
        for i, r in enumerate(recalls):
            rows.append(
                {"arm": arm, "trial": i, "induction_ok": True, "cell_recall": r, "induce_s": 1.0}
            )
    return rows


def test_ab_arm_summary_and_pair_delta():
    e = _ab()
    rows = _rows([0.10, 0.12], [0.10, 0.11], [0.20, 0.22])
    s = e._arm_summary(rows, "retrieval")
    assert s["runs"] == 2 and s["mean_cell_recall"] == 0.21
    pair = e._pair_delta(rows, "retrieval", "none")
    assert pair["delta"] == 0.10


def test_ab_verdict_floored_when_all_near_zero():
    e = _ab()
    rows = _rows([0.0, 0.01], [0.0, 0.0], [0.0, 0.01])
    summaries = {a: e._arm_summary(rows, a) for a in ("none", "static", "retrieval")}
    pairs = {
        "retrieval_vs_none": e._pair_delta(rows, "retrieval", "none"),
        "retrieval_vs_static": e._pair_delta(rows, "retrieval", "static"),
        "static_vs_none": e._pair_delta(rows, "static", "none"),
    }
    verdict, floored = e._verdict(summaries, pairs)
    assert floored is True
    assert "metric_floored" in verdict


def test_ab_verdict_reports_pairwise_directions_when_not_floored():
    e = _ab()
    # retrieval robustly beats none and static (tight, above floor, LOO-robust).
    rows = _rows([0.10, 0.11, 0.09, 0.10], [0.10, 0.11, 0.10, 0.09], [0.20, 0.22, 0.21, 0.19])
    summaries = {a: e._arm_summary(rows, a) for a in ("none", "static", "retrieval")}
    pairs = {
        "retrieval_vs_none": e._pair_delta(rows, "retrieval", "none"),
        "retrieval_vs_static": e._pair_delta(rows, "retrieval", "static"),
        "static_vs_none": e._pair_delta(rows, "static", "none"),
    }
    verdict, floored = e._verdict(summaries, pairs)
    assert floored is False
    assert "retrieval_vs_none_improved" in verdict
    assert "retrieval_vs_static_improved" in verdict


def test_ab_direction_high_variance_guard():
    e = _ab()
    # a single 0.7 outlier in retrieval; leave-one-out flips the sign -> not reliable.
    rows = _rows([0.02, 0.0, 0.0, 0.0], [0.02, 0.0, 0.0, 0.0], [0.7, 0.0, 0.0, 0.0])
    pair = e._pair_delta(rows, "retrieval", "none")
    assert e._direction(pair) == "no_reliable_signal_high_variance"


def test_ab_checksum_deterministic():
    e = _ab()
    assert e._checksum({"a": [1, 2]}) == e._checksum({"a": [1, 2]})
