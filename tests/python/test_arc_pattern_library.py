"""Tests for the verified-pattern in-context library (operator-directed 2026-06-28).

Contract: build_pattern_library mines WORKED patterns (solved trajectories + registry
win-condition/action-model/gotchas + public SOURCE CODE win-conditions) and FAILED patterns (registry
dead_ends); supports leave-one-out (exclude_game) for honest transfer testing; retrieve() returns the
top-K worked + top-M failed most similar to a query; format_incontext_block renders a few-shot reasoning
block. Verified-not-done-before: this is the in-context-exemplar lever (distinct from the nulled
weight-transfer exp4318/4342, the single-recipe router exp4556, and the efficiency-retrieval exp4933).
"""

from __future__ import annotations

import json

import yaml

from carnot.agentic.arc_pattern_library import (
    VerifiedPattern,
    _source_win_condition,
    build_pattern_library,
    format_incontext_block,
    retrieve,
)


def _write_corpus(root):
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    registry = {
        "general_gotchas": ["reset is not idempotent on some games"],
        "games": [
            {
                "game": "alpha",
                "mechanic_class": "click_fill",
                "win_condition": "fill the region to match the target palette",
                "action_model": "ACTION6 click only",
                "gotchas": ["clicks snap to centroids"],
                "dead_ends": ["value-head transfer was null on alpha"],
            },
            {
                "game": "beta",
                "mechanic_class": "chain_reorder",
                "win_condition": "reorder the chain segments by colour",
                "dead_ends": ["random exploration never found beta L2"],
            },
        ],
    }
    (root / "ops" / "arc_solve_registry.yaml").write_text(yaml.safe_dump(registry))
    (root / "results" / "arc_loop_solve_alpha.json").write_text(json.dumps({
        "game": "alpha", "offline_reproduced": True, "reproduced_levels": 2,
        "solution_labels": [json.dumps({"action": 6, "data": {"x": 1, "y": 2}}),
                            json.dumps({"action": 6, "data": {"x": 3, "y": 4}})],
        "verifier_src": "learned",
    }))


def test_build_extracts_worked_and_failed(tmp_path):
    _write_corpus(tmp_path)
    lib = build_pattern_library(root=tmp_path, include_source_code=False)
    worked = [p for p in lib if p.kind == "worked"]
    failed = [p for p in lib if p.kind == "failed"]
    assert worked and failed, "must extract both worked and failed patterns"
    # alpha's solved trajectory -> a worked solve_trajectory pattern
    assert any(p.game == "alpha" and p.source == "solve_trajectory" for p in worked)
    # alpha's win_condition -> a worked registry pattern; alpha's dead_end -> a failed pattern
    assert any(p.game == "alpha" and "fill the region" in p.text for p in worked)
    assert any(p.game == "alpha" and p.kind == "failed" for p in failed)
    # general_gotchas -> a worked 'general' pattern
    assert any(p.game == "general" for p in worked)


def test_leave_one_out_excludes_target_game(tmp_path):
    _write_corpus(tmp_path)
    lib = build_pattern_library(root=tmp_path, exclude_game="alpha", include_source_code=False)
    assert not any(p.game == "alpha" for p in lib), "LOO must drop the held-out game's own patterns"
    assert any(p.game == "beta" for p in lib), "other games remain"


def test_retrieve_ranks_by_similarity_and_caps(tmp_path):
    _write_corpus(tmp_path)
    lib = build_pattern_library(root=tmp_path, include_source_code=False)
    # query matching beta's chain-reorder mechanic should surface beta worked patterns first
    out = retrieve(lib, {"mechanic": "chain_reorder", "text": "reorder chain segments colour"},
                   k_pos=2, k_neg=1)
    assert len([p for p in out if p.kind == "worked"]) <= 2
    assert len([p for p in out if p.kind == "failed"]) <= 1
    worked = [p for p in out if p.kind == "worked"]
    assert worked and worked[0].game == "beta", "most-similar worked pattern ranks first"


def test_format_block_has_worked_and_failed_sections(tmp_path):
    _write_corpus(tmp_path)
    lib = build_pattern_library(root=tmp_path, include_source_code=False)
    block = format_incontext_block(retrieve(lib, "fill region palette target", k_pos=2, k_neg=2))
    assert "WORKED on" in block and "FAILED" in block
    assert "reason by analogy" in block.lower()


def test_source_code_win_condition_extraction(tmp_path):
    # craft a public game source with an is_solved() and confirm it is extracted (the genuinely-new input)
    gdir = tmp_path / "environment_files" / "gamez" / "v1"
    gdir.mkdir(parents=True)
    (gdir / "gamez.py").write_text(
        "import numpy as np\n"
        "def step(self, a):\n    return self.frame\n"
        "def is_solved(self, grid):\n    # win when all cells equal the target colour\n"
        "    return bool((grid == self.target).all())\n"
    )
    wc = _source_win_condition("gamez", root=tmp_path)
    assert wc is not None and "is_solved" in wc and "target" in wc


def test_real_corpus_builds_nonempty_with_source_and_deadends():
    """Integration smoke against the real repo corpus: builds many patterns incl. source_code + dead_end."""
    lib = build_pattern_library()
    assert len(lib) > 20
    sources = {p.source for p in lib}
    assert "dead_end" in sources, "failed patterns from registry dead_ends"
    assert "source_code" in sources, "worked patterns from public game SOURCE CODE (the new input)"
    assert any(p.kind == "worked" for p in lib) and any(p.kind == "failed" for p in lib)


def test_pattern_to_prompt_line():
    p = VerifiedPattern("alpha", "worked", "click_fill", "fill region", "source_code", frozenset({"fill"}))
    assert "WORKED on alpha" in p.to_prompt_line() and "fill region" in p.to_prompt_line()
