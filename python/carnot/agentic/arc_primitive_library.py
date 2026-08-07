"""Documented ARC primitive library induced from reproduced solve artifacts.

The goal is LILO-style reuse without live inference: compress repeated registry
mechanics and consolidated operators into named entries that a first-contact ARC
solver can retrieve from a digest before asking a generator to invent a plan.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from . import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[3]
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
SOLVED_CORPUS_SIZE = 18


@dataclass(frozen=True)
class DocumentedPrimitive:
    """AutoDoc-style row for a retrievable ARC mechanic or primitive."""

    name: str
    mechanic_class: str
    operator: str
    description: str
    derived_from_games: tuple[str, ...]
    retrieval_cues: tuple[str, ...]
    supported_mechanic_classes: tuple[str, ...]
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mechanic_class": self.mechanic_class,
            "operator": self.operator,
            "description": self.description,
            "derived_from_games": list(self.derived_from_games),
            "retrieval_cues": list(self.retrieval_cues),
            "supported_mechanic_classes": list(self.supported_mechanic_classes),
            "source": self.source,
        }


_MECHANIC_TEMPLATES: dict[str, dict[str, Any]] = {
    "action_sequence_replay": {
        "operator": "graph_astar_action_cost",
        "description": "Replays a discovered deterministic action-kind sequence through the offline gate.",
        "cues": ("first solve", "action kind sequence", "replays offline", "trajectory"),
        "supports": ("action_sequence_replay", "graph_explore_navigation"),
    },
    "cast_grid_world_model": {
        "operator": "active_data_collection",
        "description": "Models a cast-grid interaction whose spell changes object state before exit planning.",
        "cues": ("cast grid", "spell", "shrink", "exit", "world model", "tank controls"),
        "supports": ("cast_grid_world_model", "world_model", "object_motion_world_model"),
    },
    "click_connect_line": {
        "operator": "graph_astar_action_cost",
        "description": "Searches click-only line-connection mechanics using object click candidates.",
        "cues": ("click connect", "line", "diagonal", "click only", "rich action candidates"),
        "supports": ("click_connect_line", "graph_explore_navigation"),
    },
    "config_rule": {
        "operator": "config_rule_verifier",
        "description": "Grounds visible configuration predicates such as coverage, local constraints, and toggles.",
        "cues": (
            "config",
            "toggle",
            "coverage",
            "local constraint",
            "target offset",
            "rule family",
        ),
        "supports": (
            "config_rule",
            "config_toggle_marker_coverage",
            "config_support_clearance",
            "config_toggle_target_offset",
            "local_constraint_color_cycle",
            "config_substitution",
        ),
    },
    "color_match_slot_sequence": {
        "operator": "color_match_slot_sequence_verifier",
        "description": "Places colored items into matching colored slots in left-to-right order with undo recovery.",
        "cues": (
            "color match",
            "slot sequence",
            "item slot",
            "left to right",
            "undo",
            "validation action",
        ),
        "supports": ("color_match_slot_sequence", "config_rule", "ordered_item_slot_color_match"),
    },
    "config_substitution": {
        "operator": "glyph_rewrite_matcher",
        "description": "Matches editable glyph or sequence substitutions through reusable rewrite rules.",
        "cues": ("glyph", "rewrite", "substitution", "editable sequence", "lhs", "rhs"),
        "supports": ("config_substitution", "config_substitution_glyph_rewrite", "config_rule"),
    },
    "config_toggle_marker_coverage": {
        "operator": "config_rule_verifier",
        "description": "Moves controlled markers or supports until all visible target slots are covered.",
        "cues": (
            "marker coverage",
            "controlled markers",
            "target markers",
            "support clearance",
            "coverage",
        ),
        "supports": ("config_toggle_marker_coverage", "config_support_clearance", "config_rule"),
    },
    "config_toggle_target_offset": {
        "operator": "config_rule_verifier",
        "description": "Grounds a relative player-target offset and commits the visible toggle.",
        "cues": ("target offset", "player", "target", "commit", "toggle"),
        "supports": ("config_toggle_target_offset", "config_rule"),
    },
    "graph_explore_navigation": {
        "operator": "graph_astar_action_cost",
        "description": "Plans keyboard or click graph exploration with additive path cost and verifier heuristic.",
        "cues": (
            "graph explore",
            "keyboard directional",
            "goal distance",
            "adapter free",
            "confirm commit",
        ),
        "supports": ("graph_explore_navigation", "graph_explore"),
    },
    "local_constraint_color_cycle": {
        "operator": "config_rule_verifier",
        "description": "Cycles visible cell colors until local equality or inequality constraints hold.",
        "cues": ("local constraint", "color cycle", "neighbor", "constraints", "cells"),
        "supports": ("local_constraint_color_cycle", "config_rule"),
    },
    "object_motion_push": {
        "operator": "object_motion_world_model",
        "description": "Models player or selector movement that pushes a block through target geometry.",
        "cues": ("push", "block", "dynamic selection", "object motion", "world model"),
        "supports": ("object_motion_push", "object_motion_world_model", "world_model"),
    },
    "object_motion_reflect": {
        "operator": "object_motion_world_model",
        "description": "Models paired object motion where a selected object and reflected object move together.",
        "cues": (
            "reflect",
            "reflection",
            "selected object",
            "reflected object",
            "object slots",
            "motion family",
        ),
        "supports": ("object_motion_reflect", "object_motion_world_model", "world_model"),
    },
    "object_motion_world_model": {
        "operator": "object_motion_world_model",
        "description": "Uses object slots to synthesize translate, reflect, push, and selection transitions.",
        "cues": ("object motion", "world model", "motion family", "translate", "reflect", "push"),
        "supports": (
            "object_motion_world_model",
            "object_motion_reflect",
            "object_motion_push",
            "world_model",
        ),
    },
    "object_template_alignment": {
        "operator": "object_centric_digest",
        "description": "Aligns moveable colored objects with visible goal templates using object geometry.",
        "cues": ("align", "template", "goal sprite", "moveable piece", "drag", "object"),
        "supports": ("object_template_alignment", "program_editor_object_attribute_match"),
    },
    "program_editor_object_attribute_match": {
        "operator": "object_centric_digest",
        "description": "Edits object command slots until object attributes match the visible target.",
        "cues": (
            "program editor",
            "move program",
            "slot",
            "attribute",
            "object target",
            "bit code",
        ),
        "supports": ("program_editor_object_attribute_match", "program_editor"),
    },
}

_OPERATOR_MECHANICS: dict[str, dict[str, Any]] = {
    "active_data_collection": {
        "mechanic": "world_model",
        "description": "Collects balanced action and object cases for offline transition grounding.",
        "cues": ("active data", "transition", "coverage", "object signature", "world model"),
        "supports": ("world_model", "object_motion_world_model", "cast_grid_world_model"),
    },
    "config_rule_grounding": {
        "mechanic": "config_rule",
        "description": "Converts grounded visible configuration rules into executable action labels.",
        "cues": ("config", "rule", "marker coverage", "local constraint", "toggle"),
        "supports": _MECHANIC_TEMPLATES["config_rule"]["supports"],
    },
    "config_rule_verifier": {
        "mechanic": "config_rule",
        "description": "Retrieves and execution-grounds coverage, local-constraint, and target-toggle predicates.",
        "cues": (
            "config",
            "verifier",
            "marker coverage",
            "local constraint",
            "target offset",
            "toggle",
        ),
        "supports": _MECHANIC_TEMPLATES["config_rule"]["supports"],
    },
    "color_match_slot_sequence_verifier": {
        "mechanic": "color_match_slot_sequence",
        "description": "Retrieves and execution-grounds ordered color item-slot matching with undo-aware CEGIS.",
        "cues": _MECHANIC_TEMPLATES["color_match_slot_sequence"]["cues"],
        "supports": _MECHANIC_TEMPLATES["color_match_slot_sequence"]["supports"],
    },
    "glyph_rewrite_matcher": {
        "mechanic": "config_substitution",
        "description": "Applies reusable glyph and sequence rewrite rules before per-game reverse engineering.",
        "cues": ("glyph", "rewrite", "substitution", "sequence", "editable"),
        "supports": _MECHANIC_TEMPLATES["config_substitution"]["supports"],
    },
    "graph_astar_action_cost": {
        "mechanic": "graph_explore_navigation",
        "description": "Ranks graph-search frontiers by path cost plus verifier or distance heuristic.",
        "cues": ("graph", "astar", "keyboard", "click", "goal distance", "action cost"),
        "supports": (
            "graph_explore_navigation",
            "graph_explore",
            "click_connect_line",
            "action_sequence_replay",
        ),
    },
    "object_centric_digest": {
        "mechanic": "object_geometry_digest",
        "description": "Summarizes connected objects so routing and grounding match mechanic structure.",
        "cues": ("object", "digest", "component", "bbox", "template", "program editor"),
        "supports": (
            "object_geometry_digest",
            "object_template_alignment",
            "program_editor_object_attribute_match",
            "object_motion_world_model",
        ),
    },
    "object_motion_world_model": {
        "mechanic": "object_motion_world_model",
        "description": "Retrieves translate, reflect, push, and dynamic-selection transition models.",
        "cues": _MECHANIC_TEMPLATES["object_motion_world_model"]["cues"],
        "supports": _MECHANIC_TEMPLATES["object_motion_world_model"]["supports"],
    },
    "persistent_action_effect_memory_operator": {
        "mechanic": "action_effect_memory",
        "description": (
            "Ranks candidate actions with leave-one-game cross-game memory of cached "
            "frame/action effects before per-game reverse engineering."
        ),
        "cues": (
            "action effect",
            "clickability",
            "candidate ranking",
            "actions to first levelup",
            "cross game memory",
        ),
        "supports": (
            "action_effect_memory",
            "graph_explore_navigation",
            "click_connect_line",
            "program_editor_object_attribute_match",
        ),
    },
}

_CONSTANT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("coordinate_literal", re.compile(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)|\b-?\d+\s*,\s*-?\d+\b")),
    ("level_id_literal", re.compile(r"\bL\d+\b|\blevel\s*\d+\b", re.IGNORECASE)),
    (
        "sprite_tag_literal",
        re.compile(r"\b(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d)[a-z0-9]{10,}\b", re.IGNORECASE),
    ),
    ("action_sequence_literal", re.compile(r"\b[1-6]{5,}\b|\bcell\d+\s*,\s*\d+\b", re.IGNORECASE)),
)


_load_registry_cache_lock = threading.Lock()
_load_registry_cache: dict[Path, tuple[float, dict[str, Any]]] = {}


def _load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    """mtime-gated cache, per-``path`` (2026-08-06): profiling a live sp80 run found this
    re-parsing the 452KB registry from scratch on every call (3 calls, ~2.1s total in a
    2000-action run) via its two call sites below. An UNCONDITIONAL cache first shipped
    here (and on the sibling `arc_solve_learning._registry`) broke
    tests/python/test_experiment_4447_lilo_documented_primitive_library.py: the research
    conductor runs concurrently with the test suite and genuinely appends to this file
    mid-session, so a cache with no invalidation served a stale snapshot for the rest of
    the process. Gating on `st_mtime` keeps the fast path (a cheap `stat()`, not a 452KB
    reparse) for a short-lived process where the file cannot change mid-run -- the actual
    target, a single game-eval subprocess -- while still re-reading correctly inside a
    long-lived process (the test suite) if the file's mtime genuinely moves. Keyed by
    `path` so distinct test fixtures (each its own `tmp_path`) never share a cache slot.

    LOCKED (2026-08-06, adversarial review): a real competition submission runs every
    game's `E3AgentPolicy` on separate THREADS in one process (`Swarm.main()`, see
    `arc_competition_agent.py`), each indirectly reaching this loader via
    `recommend_approach()` up to 3x -- an unlocked check-then-write let N threads' near-
    simultaneous first calls each miss the cache and redundantly reparse. Matches the
    `RunLocalMechanicLedger` convention already used in this codebase for the same
    concurrency model."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {"games": []}
    with _load_registry_cache_lock:
        cached = _load_registry_cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return {"games": []}
        value = loaded if isinstance(loaded, dict) else {"games": []}
        _load_registry_cache[path] = (mtime, value)
        return value


def _clean_mechanic(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return text or "unknown_arc_mechanic"


def _entry_text(entry: Mapping[str, Any]) -> str:
    dead_ends = entry.get("dead_ends") or entry.get("dead_ends_recorded") or []
    pieces = [
        entry.get("mechanic_class", ""),
        entry.get("win_condition", ""),
        entry.get("action_model", ""),
        entry.get("solver", ""),
        entry.get("world_model", ""),
        entry.get("reproduce", ""),
        " ".join(str(item) for item in entry.get("gotchas", []) or []),
        " ".join(str(item) for item in dead_ends if item),
    ]
    return " ".join(str(piece).lower() for piece in pieces)


def infer_mechanic_class(entry: Mapping[str, Any]) -> str:
    """Infer a reusable mechanic class from registry metadata without copying constants."""

    explicit = _clean_mechanic(entry.get("mechanic_class"))
    if explicit != "unknown_arc_mechanic":
        if explicit == "graph_explore":
            return "graph_explore_navigation"
        if explicit == "program_editor":
            return "program_editor_object_attribute_match"
        return explicit

    text = _entry_text(entry)
    if "local constraint" in text or "color-cycle" in text or "color cycle" in text:
        return "local_constraint_color_cycle"
    if (
        "color_match_slot_sequence" in text
        or "color-match" in text
        or "color match" in text
        or "item-slot" in text
        or "item slot" in text
    ) and ("slot" in text or "sequence" in text):
        return "color_match_slot_sequence"
    if "marker-coverage" in text or "marker coverage" in text:
        return "config_toggle_marker_coverage"
    if "support-clearance" in text or "support clearance" in text:
        return "config_support_clearance"
    if "target-offset" in text or "target offset" in text:
        return "config_toggle_target_offset"
    if "editable glyph" in text or "glyph" in text or "rewrite" in text or "substitution" in text:
        return "config_substitution"
    if "reflect" in text or "reflection" in text:
        return "object_motion_reflect"
    if "push" in text or "block" in text and "world_model" in text:
        return "object_motion_push"
    if "cast-grid" in text or "cast grid" in text or "spell" in text or "shrink" in text:
        return "cast_grid_world_model"
    if "program-editor" in text or "move-program" in text or "slot" in text and "bit" in text:
        return "program_editor_object_attribute_match"
    if "click-to-connect" in text or "diagonal line" in text:
        return "click_connect_line"
    if "template" in text or "goal sprite" in text or "moveable piece" in text:
        return "object_template_alignment"
    if (
        "adapter-free" in text
        or "keyboard directional" in text
        or "goal-distance" in text
        or "move-right" in text
    ):
        return "graph_explore_navigation"
    if "action + kind sequence" in text:
        return "action_sequence_replay"
    if "world_model" in text or "e3 executable" in text:
        return "object_motion_world_model"
    return "unknown_arc_mechanic"


def solved_game_entries(
    registry: Mapping[str, Any] | None = None,
    *,
    solved_game_limit: int | None = SOLVED_CORPUS_SIZE,
) -> list[dict[str, Any]]:
    reg = dict(registry or _load_registry())
    rows: list[dict[str, Any]] = []
    for entry in reg.get("games", []) if isinstance(reg.get("games"), list) else []:
        if not isinstance(entry, Mapping):
            continue
        try:
            levels = int(entry.get("levels_reproduced") or 0)
        except (TypeError, ValueError):
            levels = 0
        if entry.get("reproducibility") == "reproduced" and levels > 0 and entry.get("solver"):
            rows.append(dict(entry))
        if solved_game_limit is not None and len(rows) >= int(solved_game_limit):
            break
    return rows


def game_digest(entry: Mapping[str, Any]) -> dict[str, Any]:
    mechanic = infer_mechanic_class(entry)
    return {
        "game": str(entry.get("game") or ""),
        "mechanic_class": mechanic,
        "win_condition": str(entry.get("win_condition") or ""),
        "action_model": str(entry.get("action_model") or ""),
        "solver": str(entry.get("solver") or ""),
        "world_model": str(entry.get("world_model") or ""),
        "registry_text": _entry_text(entry),
    }


def _filtered_games(games: Sequence[str], exclude_games: set[str]) -> tuple[str, ...]:
    return tuple(str(game) for game in games if str(game) not in exclude_games)


def _primitive_entries(exclude_games: set[str]) -> list[DocumentedPrimitive]:
    entries: list[DocumentedPrimitive] = []
    for primitive in kit.primitive_operator_registry():
        meta = _OPERATOR_MECHANICS.get(primitive.operator)
        if meta is None:
            continue
        derived = _filtered_games(primitive.derived_from_games, exclude_games)
        if not derived:
            continue
        entries.append(
            DocumentedPrimitive(
                name=primitive.operator,
                mechanic_class=str(meta["mechanic"]),
                operator=primitive.operator,
                description=str(meta["description"]),
                derived_from_games=derived,
                retrieval_cues=tuple(str(cue) for cue in meta["cues"]),
                supported_mechanic_classes=tuple(str(item) for item in meta["supports"]),
                source="consolidated_primitive",
            )
        )
    return entries


def _registry_mechanic_entries(
    registry: Mapping[str, Any] | None,
    *,
    solved_game_limit: int | None,
    exclude_games: set[str],
) -> list[DocumentedPrimitive]:
    grouped: dict[str, list[str]] = {}
    for entry in solved_game_entries(registry, solved_game_limit=solved_game_limit):
        game = str(entry.get("game") or "")
        if not game or game in exclude_games:
            continue
        mechanic = infer_mechanic_class(entry)
        if mechanic == "unknown_arc_mechanic":
            continue
        grouped.setdefault(mechanic, []).append(game)

    entries: list[DocumentedPrimitive] = []
    for mechanic, games in sorted(grouped.items()):
        template = _MECHANIC_TEMPLATES.get(mechanic) or _MECHANIC_TEMPLATES.get("config_rule")
        if template is None:
            continue
        entries.append(
            DocumentedPrimitive(
                name=f"mechanic_{mechanic}",
                mechanic_class=mechanic,
                operator=str(template["operator"]),
                description=str(template["description"]),
                derived_from_games=tuple(games),
                retrieval_cues=tuple(str(cue) for cue in template["cues"]),
                supported_mechanic_classes=tuple(str(item) for item in template["supports"]),
                source="registry_mechanic",
            )
        )
    return entries


def documented_primitive_library(
    registry: Mapping[str, Any] | None = None,
    *,
    solved_game_limit: int | None = SOLVED_CORPUS_SIZE,
    exclude_games: Sequence[str] = (),
) -> tuple[DocumentedPrimitive, ...]:
    """Build the AutoDoc primitive library from existing artifacts only."""

    excluded = {str(game) for game in exclude_games}
    rows = _primitive_entries(excluded)
    rows.extend(
        _registry_mechanic_entries(
            registry,
            solved_game_limit=solved_game_limit,
            exclude_games=excluded,
        )
    )
    rows.sort(key=lambda row: (row.source != "consolidated_primitive", row.name))
    return tuple(rows)


def _flatten_digest(value: Any, *, key: str = "") -> list[str]:
    skip_keys = {"game", "derived_from_games", "source"}
    chunks: list[str] = []
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            child_key_text = str(child_key)
            if child_key_text in skip_keys:
                continue
            chunks.append(child_key_text)
            chunks.extend(_flatten_digest(child_value, key=child_key_text))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            chunks.extend(_flatten_digest(item, key=key))
    else:
        chunks.append(str(value))
    return chunks


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", " ".join(_flatten_digest(value)).lower()).strip()


def _cue_score(cue: str, text: str, tokens: set[str]) -> float:
    cue_text = _normalize_text(cue)
    if not cue_text:
        return 0.0
    if cue_text in text:
        return 2.0
    cue_tokens = set(cue_text.split())
    if cue_tokens and cue_tokens <= tokens:
        return 1.0
    overlap = cue_tokens & tokens
    if len(overlap) >= 2:
        return 0.5
    return 0.0


def _score_entry(digest: Mapping[str, Any], entry: DocumentedPrimitive) -> tuple[float, list[str]]:
    text = _normalize_text(digest)
    tokens = set(text.split())
    target_mechanic = _clean_mechanic(digest.get("mechanic_class") or digest.get("rule_family"))
    score = 0.0
    matched: list[str] = []
    if (
        target_mechanic in entry.supported_mechanic_classes
        or target_mechanic == entry.mechanic_class
    ):
        score += 5.0
        matched.append(f"mechanic:{target_mechanic}")
    template = _MECHANIC_TEMPLATES.get(target_mechanic)
    if template is not None and entry.operator == template.get("operator"):
        score += 2.5
        matched.append(f"template_operator:{entry.operator}")
    for cue in entry.retrieval_cues:
        delta = _cue_score(cue, text, tokens)
        if delta:
            score += delta
            matched.append(cue)
    if entry.operator and _normalize_text(entry.operator) in text:
        score += 1.0
        matched.append(f"operator:{entry.operator}")
    return score, matched


def retrieve_primitives(
    digest: Mapping[str, Any],
    *,
    library: Sequence[DocumentedPrimitive] | None = None,
    top_k: int = 5,
    exclude_games: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Return ranked documented primitives for a target digest."""

    rows = (
        tuple(library)
        if library is not None
        else documented_primitive_library(exclude_games=exclude_games)
    )
    ranked: list[dict[str, Any]] = []
    for entry in rows:
        score, matched = _score_entry(digest, entry)
        if score <= 0.0:
            continue
        item = entry.as_dict()
        item["score"] = round(float(score), 3)
        item["matched_cues"] = matched
        ranked.append(item)
    ranked.sort(
        key=lambda row: (
            -float(row["score"]),
            row["source"] != "registry_mechanic",
            row["name"],
        )
    )
    return ranked[: max(0, int(top_k))]


def _supported_set(row: Mapping[str, Any]) -> set[str]:
    return {_clean_mechanic(item) for item in row.get("supported_mechanic_classes", [])}


def retrieval_identifies_mechanic(
    digest: Mapping[str, Any],
    retrieved: Sequence[Mapping[str, Any]],
) -> bool:
    target = _clean_mechanic(digest.get("mechanic_class") or digest.get("rule_family"))
    if target == "unknown_arc_mechanic":
        return False
    for row in retrieved:
        supported = _supported_set(row)
        supported.add(_clean_mechanic(row.get("mechanic_class")))
        if target in supported:
            return True
    return False


def constant_leak_violations(
    entries: Sequence[DocumentedPrimitive | Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Literal-scan AutoDoc fields for constants that would make a primitive non-generic."""

    violations: list[dict[str, str]] = []
    for entry in entries:
        row = entry.as_dict() if hasattr(entry, "as_dict") else dict(entry)
        name = str(row.get("name") or row.get("operator") or "unknown")
        fields = {
            "name": row.get("name", ""),
            "mechanic_class": row.get("mechanic_class", ""),
            "description": row.get("description", ""),
            "operator": row.get("operator", ""),
            "retrieval_cues": " ".join(str(cue) for cue in row.get("retrieval_cues", []) or []),
        }
        for field, value in fields.items():
            text = str(value)
            for kind, pattern in _CONSTANT_PATTERNS:
                match = pattern.search(text)
                if match:
                    violations.append(
                        {
                            "entry": name,
                            "field": field,
                            "kind": kind,
                            "excerpt": match.group(0),
                        }
                    )
    return violations


def documented_primitives_summary(entries: Sequence[DocumentedPrimitive]) -> list[dict[str, Any]]:
    return [
        {
            "name": entry.name,
            "mechanic_class": entry.mechanic_class,
            "derived_from_games": list(entry.derived_from_games),
        }
        for entry in entries
    ]


def measure_leave_one_out(
    registry: Mapping[str, Any] | None = None,
    *,
    solved_game_limit: int = SOLVED_CORPUS_SIZE,
) -> dict[str, Any]:
    """Measure whether documented primitives retrieve held-out mechanics."""

    reg = dict(registry or _load_registry())
    targets = solved_game_entries(reg, solved_game_limit=solved_game_limit)
    per_game: list[dict[str, Any]] = []
    identified_count = 0
    top1_count = 0
    for entry in targets:
        game = str(entry.get("game") or "")
        digest = game_digest(entry)
        loo_library = documented_primitive_library(
            reg,
            solved_game_limit=solved_game_limit,
            exclude_games=(game,),
        )
        retrieved = retrieve_primitives(digest, library=loo_library, top_k=5)
        identified = retrieval_identifies_mechanic(digest, retrieved)
        top_identified = retrieval_identifies_mechanic(digest, retrieved[:1])
        identified_count += int(identified)
        top1_count += int(top_identified)
        per_game.append(
            {
                "game": game,
                "mechanic_class": digest["mechanic_class"],
                "top_primitive": retrieved[0]["name"] if retrieved else "",
                "top_operator": retrieved[0]["operator"] if retrieved else "",
                "identified": bool(identified),
                "precision_at_1_hit": bool(top_identified),
            }
        )
    target_count = len(targets)
    library = documented_primitive_library(reg, solved_game_limit=solved_game_limit)
    return {
        "target_count": target_count,
        "library_coverage": round(float(identified_count / target_count), 6)
        if target_count
        else 0.0,
        "retrieval_precision_at_1": round(float(top1_count / target_count), 6)
        if target_count
        else 0.0,
        "per_game": per_game,
        "constant_leak_violations": constant_leak_violations(library),
    }


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_digest(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()
