"""Tests for the ARC-AGI-3 registry genre-prior distillation module.

Exercises python/carnot/agentic/arc_genre_prior_distillation.py per
docs/research-notes/arc-agi3-registry-genre-prior-distillation-scope-2026-07-12.md
section 2.1 (the offline, dev-only mining pass) and section 2.5 (the mechanical
half of the memorization-leak-through adversarial check: independent sourcing
across >= min_distinct_games non-near-duplicate games). This module extends the
routing infrastructure introduced under REQ-CAPSTONE-4582 (mechanic-class-based
live routing); the coarse taxonomy it indexes priors by is the same one
REQ-CAPSTONE-4582's live behavioral classifier already produces.

Spec: REQ-CAPSTONE-4582, SCENARIO-CAPSTONE-4582.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.agentic import arc_genre_prior_distillation as mod


def _write_registry(root: Path, games: list[dict[str, Any]]) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump({"games": games}, sort_keys=False), encoding="utf-8"
    )


def test_load_registry_missing_file_returns_empty(tmp_path: Path) -> None:
    assert mod.load_registry(tmp_path) == {"games": []}


def test_group_registry_by_coarse_class_buckets_via_shared_classifier(tmp_path: Path) -> None:
    """Games route to the SAME coarse taxonomy arc_solve_learning already uses,
    not a second drifting classifier, and entries with no minable text are
    dropped."""

    _write_registry(
        tmp_path,
        [
            {
                "game": "dc22",
                "mechanic_class": "config_toggle_navigation",
                "win_condition": "toggle the gate twice to hold it open",
                "gotchas": ["a parked ghost holds a plate open forever"],
            },
            {
                "game": "sk48",
                "mechanic_class": "chain_color_reorder",
                "win_condition": "chain endpoint must pass fully beyond the block",
                "gotchas": ["ordering contamination can make a nearer position wrong"],
            },
            {
                "game": "empty_entry",
                "mechanic_class": "config_toggle_navigation",
                # No win_condition/gotchas/novel_mechanics_found at all -> nothing to mine.
            },
        ],
    )

    grouped = mod.group_registry_by_coarse_class(tmp_path)

    config_toggle_games = {g["game"] for g in grouped.get("config_toggle", [])}
    assert config_toggle_games == {"dc22"}
    assert "empty_entry" not in {g["game"] for games in grouped.values() for g in games}
    # sk48's mechanic_class contains no config/toggle/connect/navigation keyword
    # tokens, so _coarse_mechanic_class falls through toward "unknown" (no
    # survey features available in this fixture to route it via action_type).
    all_games = {g["game"] for games in grouped.values() for g in games}
    assert "sk48" in all_games


def test_heuristic_shared_phrase_propose_finds_overlap_across_two_games() -> None:
    games = [
        {
            "game": "alpha",
            "text": "a parked ghost holds the elevator plate open forever after commit",
        },
        {
            "game": "beta",
            "text": "in this level a parked ghost holds the elevator plate open forever too",
        },
        {
            "game": "gamma",
            "text": "completely unrelated text about a spinning gear mechanism",
        },
    ]

    candidates = mod.heuristic_shared_phrase_propose("config_toggle", games)

    assert candidates, "expected at least one shared-phrase candidate"
    best = max(candidates, key=lambda c: len(c["text"].split()))
    assert set(best["sourced_from"]) == {"alpha", "beta"}
    assert "parked ghost holds" in best["text"]


def test_heuristic_shared_phrase_propose_finds_nothing_for_disjoint_text() -> None:
    games = [
        {"game": "alpha", "text": "zebra quantum lattice bicycle"},
        {"game": "beta", "text": "wombat trombone satellite umbrella"},
    ]

    assert mod.heuristic_shared_phrase_propose("unknown", games) == []


def test_mine_priors_rejects_candidate_below_min_distinct_games() -> None:
    grouped = {
        "config_toggle": [
            {"game": "alpha", "text": "shared repeated phrase across games here"},
            {"game": "beta", "text": "unrelated content entirely different words"},
        ],
    }

    def propose(_mechanic_class: str, _games: Any) -> list[dict[str, Any]]:
        return [{"text": "only one source", "sourced_from": ["alpha"]}]

    index = mod.mine_priors(grouped, propose, survey_features={})

    assert index == {}


def test_mine_priors_rejects_near_duplicate_only_sourcing() -> None:
    """The mechanical half of the memorization-leak-through adversarial check
    (scope doc section 2.5): a candidate sourced ONLY from two near-duplicate
    games (per arc_solve_learning._similarity) must not survive, even though
    it clears the bare distinct-game-count bar."""

    grouped = {
        "config_toggle": [
            {"game": "alpha", "text": "text a"},
            {"game": "beta", "text": "text b"},
        ],
    }
    survey_features = {
        "alpha": {
            "action_type": "click",
            "spatial": True,
            "difficulty": "hard",
            "win_kw": {"align", "goal"},
        },
        "beta": {
            "action_type": "click",
            "spatial": True,
            "difficulty": "hard",
            "win_kw": {"align", "goal"},
        },
    }

    def propose(_mechanic_class: str, _games: Any) -> list[dict[str, Any]]:
        return [{"text": "near duplicate sourced prior", "sourced_from": ["alpha", "beta"]}]

    index = mod.mine_priors(grouped, propose, survey_features=survey_features)

    assert index == {}


def test_mine_priors_accepts_genuinely_distinct_sourcing() -> None:
    grouped = {
        "config_toggle": [
            {"game": "alpha", "text": "text a"},
            {"game": "gamma", "text": "text c"},
        ],
    }
    survey_features = {
        "alpha": {
            "action_type": "click",
            "spatial": True,
            "difficulty": "hard",
            "win_kw": {"align"},
        },
        "gamma": {
            "action_type": "keyboard",
            "spatial": False,
            "difficulty": "easy",
            "win_kw": set(),
        },
    }

    def propose(_mechanic_class: str, _games: Any) -> list[dict[str, Any]]:
        return [{"text": "genuinely distinct sourced prior", "sourced_from": ["alpha", "gamma"]}]

    index = mod.mine_priors(grouped, propose, survey_features=survey_features)

    assert index == {
        "config_toggle": [
            {"text": "genuinely distinct sourced prior", "sourced_from": ["alpha", "gamma"]}
        ]
    }


def test_mine_priors_skips_mechanic_classes_below_min_group_size() -> None:
    grouped = {"config_toggle": [{"game": "alpha", "text": "solo game only"}]}

    def propose(_mechanic_class: str, _games: Any) -> list[dict[str, Any]]:
        raise AssertionError("propose_fn must not be called for an under-sized group")

    index = mod.mine_priors(grouped, propose, survey_features={})

    assert index == {}


def test_build_genre_prior_index_end_to_end_is_deterministic(tmp_path: Path) -> None:
    _write_registry(
        tmp_path,
        [
            {
                "game": "alpha",
                "mechanic_class": "config_toggle_target_offset",
                "win_condition": "the toggle must be hit twice to open a temporary shelf",
            },
            {
                "game": "beta",
                "mechanic_class": "config_toggle_navigation",
                "win_condition": "here the toggle must be hit twice to open a temporary shelf as well",
            },
        ],
    )

    index_a = mod.build_genre_prior_index(tmp_path)
    index_b = mod.build_genre_prior_index(tmp_path)

    assert index_a["reproducibility_checksum"] == index_b["reproducibility_checksum"]
    assert index_a["schema"] == "arc_genre_prior_index_v1"
    assert index_a["mining_mechanism"] == "heuristic_shared_phrase"
    assert "config_toggle" in index_a["games_considered_by_class"]
    assert set(index_a["games_considered_by_class"]["config_toggle"]) == {"alpha", "beta"}


def test_write_genre_prior_index_round_trips_yaml(tmp_path: Path) -> None:
    index = {"schema": "arc_genre_prior_index_v1", "priors": {"config_toggle": []}}

    out_path = mod.write_genre_prior_index(index, tmp_path / "arc_genre_priors.yaml")

    assert out_path.exists()
    loaded = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert loaded == index


def test_is_near_duplicate_pair_permissive_when_features_missing() -> None:
    assert mod._is_near_duplicate_pair("unknown_a", "unknown_b", {}) is False
