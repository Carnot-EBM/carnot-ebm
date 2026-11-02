"""Structured, game-AGNOSTIC ARC exploration-playbook patterns (REQ-ARC-WMTE-5718).

This is the machine-readable source of truth behind the prose playbook
(docs/research-notes/arc-exploration-playbook-20260717.md). Each record is one
transferable exploration PATTERN: a concise, injectable one-line `statement`
(never a per-game color/coordinate/mechanic), plus the `mechanic_tags` it applies
to (the "graph" edges: pattern -> mechanic-type), the `source_games` its rounds
were distilled from, and a short `citation`. The offline index builder
(experiment_5718) embeds the `statement` text with the live GGUF so that, at stall
time, the live agent can RETRIEVE only the patterns relevant to the current stuck
situation instead of injecting a fixed block regardless of relevance.

`mechanic_tags` taxonomy (coarse, derived from the registry's per-game
`mechanic_class`): navigation, config_toggle, chain_sort, peg_rail, drag_merge,
pattern_align, fill_flow, slot_match, program_editor, multi_agent, camera_scroll,
and `universal` (applies regardless of a game's mechanic). A pattern may carry
several tags; `universal` patterns are always at least weakly relevant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlaybookPattern:
    pattern_id: str
    theme: str
    statement: str
    mechanic_tags: tuple[str, ...]
    source_games: tuple[str, ...]
    citation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "theme": self.theme,
            "statement": self.statement,
            "mechanic_tags": list(self.mechanic_tags),
            "source_games": list(self.source_games),
            "citation": self.citation,
        }


# The coarse mechanic taxonomy, for validation + the query-side tag inference.
MECHANIC_TAGS: tuple[str, ...] = (
    "navigation",
    "config_toggle",
    "chain_sort",
    "peg_rail",
    "drag_merge",
    "pattern_align",
    "fill_flow",
    "slot_match",
    "program_editor",
    "multi_agent",
    "camera_scroll",
    "universal",
)

# Map each public game's registry mechanic_class to coarse tags -- used to infer a
# hidden game's rough mechanic class at query time from whatever class label the
# live agent has, and to tag patterns by the games they came from.
GAME_MECHANIC_TAGS: dict[str, tuple[str, ...]] = {
    "bp35": ("navigation", "camera_scroll"),
    "tu93": ("navigation", "multi_agent"),
    "m0r0": ("navigation", "camera_scroll", "multi_agent"),
    "ls20": ("navigation",),
    "sc25": ("navigation", "config_toggle"),
    "lp85": ("pattern_align", "navigation"),
    "ka59": ("navigation",),
    "wa30": ("multi_agent", "navigation"),
    "s5i5": ("config_toggle",),
    "ft09": ("config_toggle",),
    "tr87": ("config_toggle",),
    "vc33": ("config_toggle", "fill_flow"),
    "g50t": ("config_toggle", "multi_agent"),
    "dc22": ("config_toggle", "navigation"),
    "sk48": ("chain_sort",),
    "lf52": ("peg_rail", "camera_scroll"),
    "su15": ("drag_merge", "multi_agent"),
    "cn04": ("pattern_align",),
    "ar25": ("pattern_align",),
    "re86": ("pattern_align",),
    "r11l": ("pattern_align",),
    "cd82": ("fill_flow",),
    "sp80": ("fill_flow",),
    "sb26": ("slot_match",),
    "tn36": ("program_editor",),
}


def _p(
    pattern_id: str,
    theme: str,
    statement: str,
    mechanic_tags: tuple[str, ...],
    source_games: tuple[str, ...],
    citation: str,
) -> PlaybookPattern:
    return PlaybookPattern(pattern_id, theme, statement, mechanic_tags, source_games, citation)


def playbook_patterns() -> tuple[PlaybookPattern, ...]:
    """The ~22 transferable patterns, mirroring the 5 prose themes. `statement` is
    the concise, game-agnostic text that gets embedded and (when retrieved) injected."""
    return (
        # ---- Theme 1: verification habits ----
        _p(
            "verify_action_semantics_per_level",
            "verification",
            "Re-verify what each action does at THIS level from the observed transitions; do "
            "not assume an action mapping carried over from a prior level or a similar game.",
            ("universal",),
            ("bp35", "m0r0", "sc25", "sk48"),
            "bp35 L9 r13 / m0r0 L5 reversed controls",
        ),
        _p(
            "level_counter_is_ground_truth",
            "verification",
            "Only the level counter advancing is a real win; a glyph recoloring, a piece "
            "overlapping a target, or an all-green board is not proof the level completed.",
            ("universal",),
            ("sp80", "wa30", "cn04"),
            "sp80 L6 glyph-color!=entry / wa30 overlap!=placement",
        ),
        _p(
            "measure_geometry_programmatically",
            "verification",
            "Read exact positions from the frame array and reason in lattice units; off-by-one "
            "eyeballing of a rendered image wastes scarce commit/action budget.",
            ("pattern_align", "fill_flow", "config_toggle"),
            ("sp80", "wa30", "cd82"),
            "sp80 L6 one-cell elbow offset",
        ),
        _p(
            "win_frame_is_next_level",
            "verification",
            "Many games fire the level transition on the same action that completes the "
            "arrangement, so the solved board never renders; ground the win predicate on the "
            "state just before the transition and count only the reproduced level advance.",
            ("config_toggle", "pattern_align", "slot_match"),
            ("lp85", "s5i5", "cd82"),
            "lp85 L6-L7 search frame-after-next",
        ),
        # ---- Theme 2: perception / object interrogation ----
        _p(
            "interrogate_unexplained_objects",
            "perception",
            "Click or otherwise interrogate every visually-unexplained object before committing "
            "a route; hazards and helper/utility objects are frequently camouflaged as "
            "background or decoration and are often the actual unlock.",
            ("universal",),
            ("wa30", "bp35"),
            "wa30 v9 helper robot / bp35 growable pillar",
        ),
        _p(
            "naive_color_detectors_fooled_by_highlight",
            "perception",
            "Objects recolor on contact or selection; a detector keyed on one resting color "
            "misses the same object once it is adjacent, selected, or carrying something.",
            ("multi_agent", "pattern_align", "navigation"),
            ("wa30", "r11l", "sp80"),
            "wa30 contact-highlight / r11l selection diamond",
        ),
        _p(
            "read_multilayer_animation_for_absolute_motion",
            "perception",
            "Read the multi-layer animation array (all sub-frames of one action), not just the "
            "settled grid, to recover absolute motion; the camera-relative settled frame hides "
            "up/down/left/right displacement when the camera recenters.",
            ("camera_scroll", "navigation"),
            ("bp35", "wa30", "ka59"),
            "bp35 L9 r19 animation-frame trajectory reader",
        ),
        _p(
            "camera_is_view_relative_reverify_after_scroll",
            "perception",
            "In any game with a scrolling or recentering camera, click coordinates are "
            "view-relative and the same grid cell maps to different screen coordinates after "
            "the camera moves; re-verify positions after any scroll.",
            ("camera_scroll", "navigation", "peg_rail"),
            ("lf52", "bp35", "m0r0"),
            "lf52 L3 view-relative clicks",
        ),
        _p(
            "use_games_own_move_enumeration",
            "perception",
            "Many games render their own legal-move affordances (highlighted landing squares, a "
            "required-color preview); read that as a free move enumerator, but a preview of a "
            "target is not proof the target is satisfied.",
            ("peg_rail", "drag_merge", "pattern_align"),
            ("lf52", "su15", "re86"),
            "lf52 L3 ACTION6 paints legal landings",
        ),
        _p(
            "target_is_not_the_goal_display",
            "perception",
            "Distinguish the interactive delivery target from an inert goal DISPLAY/legend; "
            "clicking the picture of the goal or a HUD legend does nothing and wastes actions.",
            ("drag_merge", "fill_flow", "slot_match"),
            ("su15", "cd82", "sb26"),
            "su15 deliver to blobs not top boxes",
        ),
        # ---- Theme 3: search / reachability ----
        _p(
            "proven_exhausted_vs_search_capped",
            "search",
            "A search that finds no solution is a PROOF of unreachability only if the frontier "
            "emptied with no cut branches; if it hit a node/time/depth budget it proves nothing "
            "-- never treat a capped search as a settled dead end.",
            ("universal",),
            ("wa30", "bp35", "lf52"),
            "wa30 settled-dead-end overturned; bp35 L9 partial-not-proven",
        ),
        _p(
            "hash_semantic_state_ignore_cosmetics",
            "search",
            "Deduplicate search states on the load-bearing subset of state (position, facing, "
            "the flags that matter) and explicitly ignore cosmetic variation (animation phase, "
            "click counters, decorative growth), or the frontier explodes -- but verify what is "
            "truly cosmetic before discarding it.",
            ("navigation", "config_toggle", "peg_rail"),
            ("lp85", "bp35"),
            "lp85 goal-key dedup; bp35 pillar-growth not cosmetic",
        ),
        _p(
            "question_win_model_when_navigation_exhausted",
            "search",
            "If every reachable route and interaction has been tried and nothing wins, the bug "
            "may be your model of what winning MEANS (spatial-touch vs a score/collection/"
            "activation condition), not your search over how to get there.",
            ("navigation", "config_toggle", "fill_flow"),
            ("bp35", "sp80", "vc33"),
            "bp35 L9 tested non-spatial win; sp80 L5 hidden win condition",
        ),
        _p(
            "source_grounded_validate_and_enumerate_all_pieces",
            "search",
            "For a PUBLIC development game only, a source-grounded offline simulator can break "
            "walls blind interaction cannot -- but validate it move-for-move against the real "
            "engine before trusting it, and any impossibility proof must enumerate EVERY "
            "selectable piece type, not just the one that satisfies the win.",
            ("peg_rail", "chain_sort"),
            ("lf52", "sk48"),
            "lf52 r14 confinement-proof omitted a piece type",
        ),
        _p(
            "verifier_routes_gate_decides",
            "search",
            "Use a learned or computed verifier to route best-first search (a dense "
            "cyclic-distance heuristic beats a sparse mismatch count), but the executable "
            "reproduction gate is the only authority that counts a level.",
            ("config_toggle", "universal"),
            ("tr87", "sk48", "ar25"),
            "tr87 summed-cyclic-distance routes L4",
        ),
        # ---- Theme 4: hazard / budget / reset ----
        _p(
            "fresh_env_branches_for_risky_tests",
            "hazard",
            "Test a dangerous hypothesis on a fresh throwaway environment branch, not on a real "
            "in-progress attempt; a fresh env per candidate is also the correct branch mode for "
            "a game whose reset is not idempotent.",
            ("navigation", "config_toggle", "universal"),
            ("tu93", "sp80", "cn04"),
            "tu93 gotcha#7 fresh_env; sp80 fresh-env branching",
        ),
        _p(
            "env_reset_is_often_not_clean",
            "hazard",
            "After a game-over or mid-level, env.reset() often leaves poisoned hidden state "
            "(parity flags, leaked ghosts, depleted timers); build a fresh env, or use the "
            "game's own in-game RESET action, rather than chaining resets.",
            ("navigation", "multi_agent"),
            ("tu93", "g50t", "r11l"),
            "tu93 parity-poisoned reset; g50t ghost leak",
        ),
        _p(
            "count_every_action_including_failed",
            "hazard",
            "Games meter actions; blocked moves, failed clicks, selects, no-ops, and undo often "
            "consume the same budget, and failed clicks sometimes cost more (escalating "
            "lockouts). Plan routes as speedruns when a hard action ceiling exists.",
            ("drag_merge", "navigation", "pattern_align"),
            ("su15", "bp35", "r11l"),
            "su15 failed-click escalating meter; bp35 L6 hard action clock",
        ),
        _p(
            "reset_and_timer_scope_varies",
            "hazard",
            "Check whether a reset restarts the current level or the whole game, and whether the "
            "level's internal action/life timer is separate from your session budget; the same "
            "action can even have different scope depending on when it fires.",
            ("navigation", "drag_merge", "config_toggle"),
            ("r11l", "su15", "ka59"),
            "r11l ACTION0 scope depends on timing; su15 L9 full-game reset",
        ),
        _p(
            "bisect_death_to_minimal_prefix",
            "hazard",
            "When a sequence ends in death, binary-search the prefix to the exact action that "
            "kills, and separate a real hazard from an unrelated budget/timer/harness cause "
            "before concluding a move is lethal.",
            ("navigation",),
            ("bp35",),
            "bp35 L6 'lethal step' was the action clock",
        ),
        # ---- Theme 5: when stuck / reframe ----
        _p(
            "distinguish_harness_artifact_from_real_state",
            "reframe",
            "An empty/degenerate frame or a wide game-over OVERVIEW render is frequently a "
            "harness/engine artifact, not a real signal; a death-overview shows the whole level "
            "regardless of where death occurred, so it is not a location-specific reveal.",
            ("universal",),
            ("g50t", "bp35", "cd82"),
            "g50t L7 post-win extra-step sentinel; bp35 L9 fixed overview render",
        ),
        _p(
            "name_the_one_missing_step",
            "reframe",
            "When a plan fails, diagnose the single missing action/ordering from its exact "
            "failure point and insert it, rather than restarting; a failing candidate is data "
            "that localizes the fix.",
            ("navigation", "config_toggle"),
            ("ls20", "s5i5", "sp80"),
            "ls20 L5 'insert the ring reset'; s5i5 L4 exchange-order fix",
        ),
        _p(
            "minimize_tooling_reason_from_raw_arrays",
            "reframe",
            "Do not burn a session building visualization harnesses or over-investing in "
            "offline simulation; reason from raw grid arrays, handle multi-layer frames "
            "defensively, and budget translating any offline plan into live actions before the "
            "session ends.",
            ("universal",),
            ("lf52", "ka59", "dc22"),
            "lf52 r3 PNG-harness timeouts; ka59 tooling crash",
        ),
        _p(
            "beware_ordering_and_prefix_contamination",
            "reframe",
            "Reaching the right positions is not enough when order matters: an intended sequence "
            "can be contaminated by a nearer element seen first, by leftover prefix actions "
            "bleeding into a level entry, or by residual win-animation state -- make candidate "
            "scripts phase-robust.",
            ("chain_sort", "config_toggle", "pattern_align"),
            ("sk48", "wa30", "tr87"),
            "sk48 L6 ordering contamination; tr87 fade-animation residue",
        ),
    )
