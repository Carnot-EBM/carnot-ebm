"""ARC-AGI-3 registry genre-prior distillation — OFFLINE, dev-only mining pass.

Why this exists
----------------
2026-07-12: an operator question ("why was the live mechanism unable to find
an efficient search path within budget when we have a pretty complete
registry?") led to a code-grounded finding: even when the live agent's
`arc_strategy_router.detect_mechanic()` reads a target game's own registry
`mechanic_class` field, all that buys is picking one of ~5 generic search
algorithms (`route_strategy`). None of the actual hard-won structural
knowledge banked in `ops/arc_solve_registry.yaml`'s free-text `gotchas` /
`win_condition` fields (e.g. "a toggle sometimes needs to fire TWICE" or
"a nearer position can still be wrong because a wrong-color block gets
seen first") ever reaches the search as a usable prior. See
`docs/research-notes/arc-agi3-registry-genre-prior-distillation-scope-2026-07-12.md`
for the full scope this module implements the FIRST step of (its own
rollout-order §4 step 2: "run once by hand as a dev-only script producing
ops/arc_genre_priors.yaml; inspect the output... BEFORE wiring any
consumer").

What this module deliberately does NOT do (yet)
-------------------------------------------------
- It does NOT wire into `E3AgentPolicy` / any live-scored path. The scope
  doc's §2.2 (confidence-gated live consultation) and §2.3 (kill switch) are
  separate, later work gated on this module's output being inspected first
  and on the §2.4 offline A/B validation clearing a real bar. Building the
  consumer before the miner has anything worth consuming would be premature.
- It does NOT replay, memorize, or surface any single game's own action
  sequence. Every candidate prior REQUIRES independent textual support from
  at least `min_distinct_games` (default 2) registry entries, and rejects
  candidates whose only sources are near-duplicate games (see
  `_is_near_duplicate_pair`) — the mechanical half of the scope doc's §2.5
  "memorization leak-through" adversarial check. This keeps the module on
  the PERMITTED side of the operator's memorization-vs-genre-learning ruling
  (quoted in full in the scope doc's §2.0): genre/class-level generalization
  is fine, a single game's specific solution laundered as a "genre prior" is
  not.

Mining mechanism (honest disclosure, per CLAUDE.md's Verifier Authenticity
Discipline's naming-honesty principle even though this module is not itself
a verifier): the DEFAULT `propose_fn` (`heuristic_shared_phrase_propose`) is
a plain text-statistical heuristic — it looks for multi-word phrases that
appear, near-verbatim, in more than one game's gotchas text within the same
coarse mechanic class. It is NOT an LLM-based generalizer and should not be
mistaken for one.

PILOT RESULT (2026-07-12,
`results/experiment_5581_genre_prior_distillation_heuristic_pilot.json`): run
once by hand against the live registry, this heuristic produced 604 raw
candidates, of which ZERO — after manually reading all 140 that survived an
18-token boilerplate filter — expressed genuine game-mechanic content. Every
surviving candidate was drawn from the outer loop's own repetitive
verification-narration template ("frontier stays at level", "round N 2026 07
12 gpt 5 6 sol via", "all 5 confirmed level"), not game mechanics — because
mechanic-specific wording (colors, coordinates, object names) differs per game
by construction, while the outer loop's own authorial narration template does
not. This is a CONFIRMED, DECISIVE negative for the lexical mechanism on this
corpus, not merely a weak-but-worth-trying first pass. The scope doc's §2.1
anticipates an LLM-assisted (semantic) mining pass as the mechanism that
would actually work; `propose_fn` is a plain callable
(`(mechanic_class, games) -> list[dict]`) specifically so a semantic/LLM-backed
proposer can be swapped in without touching the harness (loading, grouping,
independence filtering, writing) built here — the harness is retained as
correct, reusable infrastructure regardless of which propose_fn is used, but
`heuristic_shared_phrase_propose` itself should not be relied on to produce
anything useful as shipped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

from . import arc_solve_learning as learning

REPO = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"
DEFAULT_OUTPUT_PATH = REPO / "ops" / "arc_genre_priors.yaml"

# The same coarse 7-bucket taxonomy `arc_solve_learning.classify_early_play_mechanic`
# produces from a hidden game's own live transitions. Indexing priors by this set
# (rather than the registry's own much more specific per-game `mechanic_class`
# strings, e.g. "two_phase_cast_grid_then_tank_exit") is what lets a genuinely
# hidden game — one with zero registry presence — still match a prior the instant
# its mechanic class is classified from its own play.
MECHANIC_CLASSES: tuple[str, ...] = (
    "avatar_navigation",
    "click_connect",
    "config_toggle",
    "hidden_carry_state",
    "keyboard_graph",
    "click_graph",
    "unknown",
)

MIN_DISTINCT_GAMES_DEFAULT = 2
# Conservative first-pass bar on arc_solve_learning._similarity()'s 0..~7.5 scale
# (action_type match=3.0, spatial match=1.5, difficulty match=0.5, +1.0 per shared
# win-condition keyword). Two games already share a coarse mechanic class by
# construction (that's how they were grouped) so some baseline similarity is
# expected; this threshold additionally requires near-total survey-feature
# identity before treating a sourcing pair as "not independent enough" to count
# toward the >= min_distinct_games bar. Tunable; documented here rather than
# buried so a future pass can justify moving it with evidence.
NEAR_DUPLICATE_SIMILARITY_THRESHOLD = 7.0

_GOTCHA_TEXT_FIELDS: tuple[str, ...] = ("win_condition", "gotchas", "novel_mechanics_found")

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "and",
        "or",
        "but",
        "if",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "by",
        "from",
        "into",
        "onto",
        "not",
        "no",
        "do",
        "does",
        "did",
        "done",
    ]
)

_PHRASE_MIN_WORDS = 4
_PHRASE_MAX_WORDS = 9


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _entry_text(entry: Mapping[str, Any]) -> str:
    """Join a registry entry's free-text fields into one blob for mining."""

    parts: list[str] = []
    for field in _GOTCHA_TEXT_FIELDS:
        value = entry.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Sequence):
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, Mapping):
                    # dead_ends-style structured rows and principle-wrapped fields;
                    # only pull the human-readable prose, not identifiers.
                    for key in ("text", "summary", "value", "filled_summary"):
                        nested = item.get(key)
                        if isinstance(nested, str):
                            parts.append(nested)
    return "\n".join(parts)


def load_registry(root: Path | str = REPO) -> dict[str, Any]:
    path = Path(root) / "ops" / "arc_solve_registry.yaml"
    if not path.exists():
        return {"games": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {"games": []}


def coarse_class_for_entry(
    entry: Mapping[str, Any], survey_features: Mapping[str, dict] | None = None
) -> str:
    """Map a registry entry to the coarse 7-bucket taxonomy.

    Reuses `arc_solve_learning._coarse_mechanic_class`, the SAME mapping the
    offline feature-router-policy trainer already applies to registry rows —
    deliberately not a second, drifting classifier.
    """

    game = str(entry.get("game") or "")
    features = None
    if survey_features is not None:
        features = survey_features.get(game)
    return learning._coarse_mechanic_class(entry.get("mechanic_class"), features)


def group_registry_by_coarse_class(root: Path | str = REPO) -> dict[str, list[dict[str, Any]]]:
    """Group every registry game into the coarse mechanic-class taxonomy.

    Returns {mechanic_class: [{"game": ..., "text": ..., "raw_mechanic_class": ...}]}.
    Games with empty mined text (no gotchas/win_condition prose at all) are
    dropped — there is nothing to mine from them.
    """

    registry = load_registry(root)
    try:
        survey_features = learning._survey_features()
    except (OSError, FileNotFoundError, json.JSONDecodeError, KeyError):
        survey_features = {}

    grouped: dict[str, list[dict[str, Any]]] = {cls: [] for cls in MECHANIC_CLASSES}
    for entry in registry.get("games", []) or []:
        if not isinstance(entry, Mapping):
            continue
        text = _entry_text(entry)
        if not text.strip():
            continue
        coarse = coarse_class_for_entry(entry, survey_features)
        if coarse not in grouped:
            grouped[coarse] = []
        grouped[coarse].append(
            {
                "game": str(entry.get("game") or ""),
                "text": text,
                "raw_mechanic_class": str(entry.get("mechanic_class") or ""),
            }
        )
    return grouped


def _is_near_duplicate_pair(
    game_a: str,
    game_b: str,
    survey_features: Mapping[str, dict],
    threshold: float = NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
) -> bool:
    """Conservative "these two sources aren't independent enough" check.

    Permissive by default: if either game is missing survey features (true
    for any game outside the 25-game public survey), we cannot assess
    similarity and treat the pair as NOT a near-duplicate rather than
    silently dropping otherwise-valid sourcing. This mirrors
    `arc_solve_learning.recommend_approach`'s own cold-route-on-missing-data
    behavior rather than inventing a new failure mode.
    """

    features_a = survey_features.get(game_a)
    features_b = survey_features.get(game_b)
    if not features_a or not features_b:
        return False
    return learning._similarity(features_a, features_b) >= threshold


def _sourcing_is_independent(
    sourced_from: Sequence[str],
    survey_features: Mapping[str, dict],
    min_distinct_games: int,
    near_duplicate_threshold: float,
) -> bool:
    distinct = sorted(set(sourced_from))
    if len(distinct) < min_distinct_games:
        return False
    # Reject only if EVERY pair is a near-duplicate — i.e. the sourcing set
    # has no genuinely distinct pair backing it. A candidate sourced from 3
    # games where 2 are near-duplicates but the 3rd is genuinely different
    # still has independent support and should survive.
    pairs = [
        (distinct[i], distinct[j])
        for i in range(len(distinct))
        for j in range(i + 1, len(distinct))
    ]
    if not pairs:
        return False
    return any(
        not _is_near_duplicate_pair(a, b, survey_features, near_duplicate_threshold)
        for a, b in pairs
    )


def _normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _candidate_phrases(words: Sequence[str]) -> set[str]:
    phrases: set[str] = set()
    for length in range(_PHRASE_MIN_WORDS, _PHRASE_MAX_WORDS + 1):
        for start in range(0, len(words) - length + 1):
            window = words[start : start + length]
            # Require at least one non-stopword content word so we don't
            # match on filler phrases like "of the game and the".
            if all(w in _STOPWORDS for w in window):
                continue
            phrases.add(" ".join(window))
    return phrases


def heuristic_shared_phrase_propose(
    mechanic_class: str, games: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Default, honest-heuristic propose_fn: shared near-verbatim phrase mining.

    Finds `_PHRASE_MIN_WORDS`..`_PHRASE_MAX_WORDS`-word phrases that appear in
    more than one game's mined text within `games` (all already the same
    coarse mechanic class). This is plain text-statistical overlap detection,
    not semantic generalization — see the module docstring's "Mining
    mechanism" section. Expected to find FEW or ZERO candidates on registry
    prose that is mostly per-game-specific (coordinates, colors, action
    counts differ per game even when the underlying mechanic class matches);
    a sparse or empty result here is an honest, informative signal that the
    generalizable structure exists at the SEMANTIC level, not the lexical
    level, and motivates the LLM-assisted upgrade path the scope doc
    anticipates rather than being a bug in this pass.
    """

    phrase_sources: dict[str, set[str]] = {}
    for game in games:
        words = _normalize_words(str(game.get("text") or ""))
        for phrase in _candidate_phrases(words):
            phrase_sources.setdefault(phrase, set()).add(str(game.get("game") or ""))

    # Drop phrases subsumed by a longer phrase with the identical source set
    # (keep the more specific/informative phrase, not every sub-window of it).
    phrases_sorted = sorted(phrase_sources, key=len, reverse=True)
    kept: list[str] = []
    for phrase in phrases_sorted:
        sources = phrase_sources[phrase]
        if len(sources) < 2:
            continue
        if any(phrase in longer and phrase_sources[longer] == sources for longer in kept):
            continue
        kept.append(phrase)

    return [
        {
            "text": phrase,
            "sourced_from": sorted(phrase_sources[phrase]),
            "mechanic_class": mechanic_class,
        }
        for phrase in kept
    ]


PriorProposer = Callable[[str, Sequence[Mapping[str, Any]]], list[dict[str, Any]]]


def mine_priors(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    propose_fn: PriorProposer = heuristic_shared_phrase_propose,
    *,
    min_distinct_games: int = MIN_DISTINCT_GAMES_DEFAULT,
    near_duplicate_threshold: float = NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
    survey_features: Mapping[str, dict] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Apply propose_fn per mechanic class, then the independence filters.

    This function owns the mechanical, non-negotiable filters (distinct-game
    count + near-duplicate rejection) regardless of which propose_fn was
    used — so even an LLM-backed proposer swapped in later cannot ship a
    memorization-leak-through prior without also passing this gate.
    """

    if survey_features is None:
        try:
            survey_features = learning._survey_features()
        except (OSError, FileNotFoundError, json.JSONDecodeError, KeyError):
            survey_features = {}

    index: dict[str, list[dict[str, Any]]] = {}
    for mechanic_class, games in grouped.items():
        if len(games) < min_distinct_games:
            continue
        candidates = propose_fn(mechanic_class, games)
        accepted = [
            candidate
            for candidate in candidates
            if _sourcing_is_independent(
                candidate.get("sourced_from") or [],
                survey_features,
                min_distinct_games,
                near_duplicate_threshold,
            )
        ]
        if accepted:
            index[mechanic_class] = accepted
    return index


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def build_genre_prior_index(
    root: Path | str = REPO,
    propose_fn: PriorProposer = heuristic_shared_phrase_propose,
    *,
    min_distinct_games: int = MIN_DISTINCT_GAMES_DEFAULT,
    near_duplicate_threshold: float = NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    grouped = group_registry_by_coarse_class(root)
    try:
        survey_features = learning._survey_features()
    except (OSError, FileNotFoundError, json.JSONDecodeError, KeyError):
        survey_features = {}
    priors = mine_priors(
        grouped,
        propose_fn,
        min_distinct_games=min_distinct_games,
        near_duplicate_threshold=near_duplicate_threshold,
        survey_features=survey_features,
    )
    games_considered = {
        mechanic_class: sorted({g["game"] for g in games})
        for mechanic_class, games in grouped.items()
        if games
    }
    index = {
        "schema": "arc_genre_prior_index_v1",
        "mining_mechanism": "heuristic_shared_phrase"
        if propose_fn is heuristic_shared_phrase_propose
        else "custom",
        "min_distinct_games": min_distinct_games,
        "near_duplicate_similarity_threshold": near_duplicate_threshold,
        "games_considered_by_class": games_considered,
        "priors": priors,
    }
    index["reproducibility_checksum"] = hashlib.sha256(
        _stable_json(priors).encode("utf-8")
    ).hexdigest()
    return index


def write_genre_prior_index(
    index: Mapping[str, Any], path: Path | str = DEFAULT_OUTPUT_PATH
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(dict(index), sort_keys=False), encoding="utf-8")
    return out_path


def main() -> None:  # pragma: no cover — thin CLI wrapper, exercised manually
    index = build_genre_prior_index()
    out_path = write_genre_prior_index(index)
    total_priors = sum(len(v) for v in index["priors"].values())
    print(
        f"wrote {out_path} — {total_priors} candidate prior(s) across {len(index['priors'])} class(es)"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
