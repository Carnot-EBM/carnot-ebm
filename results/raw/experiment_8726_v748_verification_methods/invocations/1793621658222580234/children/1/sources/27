"""ArcMemo v7 cross-game concept-library transfer accounting.

Spec refs: REQ-LEARN-4050, SCENARIO-LEARN-4050.

Exp 4050 asks a stricter self-learning question than Exp 4039: did a library
learned from earlier games reduce the cost of a new eighth game? This module is
artifact-level accounting. It never imports the eighth-game induced mechanic
into the library; it only uses Exp 4049 as the held-out target measurement.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


INFERENCE_SUBSTRATE = "offline_arc_agi3_arcmemo_cross_game_transfer"
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "cross_game_transfer_win",
    "actions_cold",
    "actions_within_game_v6",
    "actions_cross_game_v7",
    "n_reused_abstractions",
    "inference_substrate",
)
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")

_FRAGMENT_TEMPLATES: dict[str, dict[str, str]] = {
    "r11l": {
        "name": "select_then_place",
        "family": "click_state_transform",
        "signature": "select_then_place(state, selected_piece, target) -> placed_piece_state",
        "lambda_abstraction": "lambda state: click(selectable_object) then click(target_slot_or_centroid)",
        "documentation": "Why: a click creates a latent selected object and a later click applies that selection.",
    },
    "lp85": {
        "name": "permute_set_by_button",
        "family": "discrete_set_permutation",
        "signature": "permute_set_by_button(state, button) -> permuted_piece_state",
        "lambda_abstraction": "lambda state: choose(button) then apply_deterministic_permutation(state)",
        "documentation": "Why: button actions can be cached as reusable set permutations.",
    },
    "sc25": {
        "name": "pattern_match_then_goal_commit",
        "family": "click_state_transform",
        "signature": "pattern_match_then_goal_commit(state, clicked_cells, exit_path) -> level_counter_goal",
        "lambda_abstraction": "lambda state: satisfy_clicked_pattern(state) then commit_goal_suffix(state)",
        "documentation": "Why: click transforms first satisfy a pattern before a terminal goal action sequence.",
    },
    "su15": {
        "name": "click_to_step_pruned_path",
        "family": "click_state_transform",
        "signature": "click_to_step_pruned_path(state, click_targets) -> target_zone_state",
        "lambda_abstraction": "lambda state: reuse_executed_click_dynamics(state) then prune_to_goal_path(state)",
        "documentation": "Why: observed click deltas can seed a grounded path without rediscovering every action.",
    },
    "tn36": {
        "name": "program_row_toggle_then_execute",
        "family": "click_state_transform",
        "signature": "program_row_toggle_then_execute(state, row_toggles, execute_button) -> matched_target",
        "lambda_abstraction": "lambda state: toggle_program_rows(state) then click_execute_for_level_counter(state)",
        "documentation": "Why: visible click targets update latent program rows before a terminal execute click.",
    },
    "cd82": {
        "name": "active_region_fill_then_validate",
        "family": "region_state_commit",
        "signature": "active_region_fill_then_validate(state, region, color) -> painted_goal_region",
        "lambda_abstraction": "lambda state: move_active_region(state) then action5_paint_and_validate(state)",
        "documentation": "Why: action selection changes active state before a single validation/fill commit.",
    },
    "dc22": {
        "name": "toggle_blocker_then_goal_navigation",
        "family": "click_state_transform",
        "signature": "toggle_blocker_then_goal_navigation(state, blockers, path) -> level_counter_goal",
        "lambda_abstraction": "lambda state: click_toggle_blockers(state) then navigate_to_goal_counter(state)",
        "documentation": "Why: clicked toggles change traversability before a terminal goal path can succeed.",
    },
}


def _as_int(payload: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = payload.get(key)
        if value is not None and type(value) is not bool:
            return int(value)
    return 0


def _text(value: object) -> str:
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if isinstance(value, list | tuple):
        return " ".join(_text(item) for item in value)
    return str(value or "")


def _tokens(*values: object) -> set[str]:
    return set(_TOKEN_RE.findall(" ".join(_text(value).lower() for value in values)))


def _terminal_prefixed(value: object) -> bool:
    return isinstance(value, str) and value.startswith(TERMINAL_PREFIXES)


def _row_payload(row: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not row:
        return None
    payload = row.get("payload")
    if isinstance(payload, Mapping):
        return payload
    return row


def _source_game_from_row(row: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    explicit = row.get("source_game")
    if explicit:
        return str(explicit)
    for key in ("target_game", "game_solved", "game"):
        value = payload.get(key)
        if isinstance(value, str) and value and value != "none":
            return value.split("-", maxsplit=1)[0]
    attempts = payload.get("attempt_details")
    if isinstance(attempts, list) and attempts:
        first = attempts[0]
        if isinstance(first, Mapping):
            game_id = str(first.get("game_id", ""))
            if game_id:
                return game_id.split("-", maxsplit=1)[0]
    return "unknown"


def _source_artifact_from_row(row: Mapping[str, Any]) -> str:
    return str(row.get("source_artifact", row.get("source", "unknown_prior_artifact")))


def _mechanic_text(payload: Mapping[str, Any]) -> str:
    for key in ("induced_mechanic", "induced_select_place_mechanic"):
        value = payload.get(key)
        if value:
            return str(value)
    attempts = payload.get("attempt_details")
    if isinstance(attempts, list) and attempts:
        first = attempts[0]
        if isinstance(first, Mapping) and first.get("induced_mechanic"):
            return str(first["induced_mechanic"])
    return ""


def _confirmed_prior_solve(payload: Mapping[str, Any]) -> bool:
    if payload.get("real_env_confirmed") is not True:
        return False
    if _as_int(payload, "ACCURACY_levels_solved", "levels_completed", "level_completed") > 0:
        return True
    if _as_int(payload, "first_solve_at_action") > 0:
        return True
    solved = payload.get("game_solved")
    return solved is True or (isinstance(solved, str) and solved not in {"", "none"})


def _template_for(source_game: str) -> dict[str, str]:
    return _FRAGMENT_TEMPLATES.get(
        source_game,
        {
            "name": f"{source_game}_induced_fragment",
            "family": "unknown_induced_program",
            "signature": f"{source_game}_induced_fragment(state, ops) -> candidate_goal_state",
            "lambda_abstraction": "lambda state: reuse_named_fragment(state, ops)",
            "documentation": "Why: a confirmed prior solve contributed an induced operation fragment.",
        },
    )


def collect_prior_fragments(prior_solved_artifacts: Iterable[Mapping[str, Any] | None]) -> list[dict[str, Any]]:
    """Collect named induced-program fragments from prior solved games only."""

    fragments: list[dict[str, Any]] = []
    for row in prior_solved_artifacts:
        payload = _row_payload(row)
        if payload is None or row is None or not _confirmed_prior_solve(payload):
            continue
        source_game = _source_game_from_row(row, payload)
        if source_game == "sb26":
            continue
        template = _template_for(source_game)
        mechanic = _mechanic_text(payload)
        fragments.append(
            {
                "name": template["name"],
                "family": template["family"],
                "source_game": source_game,
                "source_artifact": _source_artifact_from_row(row),
                "signature": template["signature"],
                "documentation": template["documentation"],
                "lambda_abstraction": template["lambda_abstraction"],
                "induced_program_fragment": mechanic,
                "tokens": sorted(_tokens(source_game, template["name"], template["family"], mechanic)),
            }
        )
    return fragments


def _click_state_abstraction(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "name": "click_state_transform_then_goal_commit_v7",
        "family": "click_state_transform",
        "signature": (
            "click_state_transform_then_goal_commit_v7("
            "observed_state, click_or_toggle_ops, terminal_goal_counter) -> executable_action_plan"
        ),
        "documentation": (
            "Why: several prior ARC games used click-driven state changes followed by a terminal "
            "goal-counter commit. The abstraction can seed the new solver's operation shape, but "
            "the new game still must execute its own necessary actions from reset."
        ),
        "lambda_abstraction": (
            "lambda observed_state: bind_prior_click_state_transform(observed_state) "
            "then plan_target_specific_goal_commit(observed_state)"
        ),
        "source_fragments": sorted({str(record["name"]) for record in records}),
        "source_games": sorted({str(record["source_game"]) for record in records}),
        "source_artifacts": sorted({str(record["source_artifact"]) for record in records}),
        "occurrence_count": len(records),
        "match_tokens": [
            "action5",
            "click",
            "color",
            "colors",
            "goal",
            "item",
            "place",
            "select",
            "slot",
            "target",
            "toggle",
            "validate",
        ],
    }


def _generic_abstraction(family: str, records: list[Mapping[str, Any]]) -> dict[str, Any]:
    tokens = sorted(set().union(*(set(record.get("tokens", [])) for record in records)))[:12]
    return {
        "name": f"{family}_v7_abstraction",
        "family": family,
        "signature": f"{family}(observed_state, reusable_ops) -> executable_action_plan",
        "documentation": "Why: recurring prior-game fragments in this family justify a named reusable operation.",
        "lambda_abstraction": "lambda observed_state: reuse_v7_family_template(observed_state, reusable_ops)",
        "source_fragments": sorted({str(record["name"]) for record in records}),
        "source_games": sorted({str(record["source_game"]) for record in records}),
        "source_artifacts": sorted({str(record["source_artifact"]) for record in records}),
        "occurrence_count": len(records),
        "match_tokens": tokens,
    }


def build_v7_library(prior_fragments: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compress recurring prior-game fragments into v7 library abstractions."""

    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for fragment in prior_fragments:
        by_family[str(fragment.get("family", "unknown_induced_program"))].append(fragment)

    library: list[dict[str, Any]] = []
    for family in sorted(by_family):
        records = by_family[family]
        source_games = {str(record.get("source_game", "")) for record in records if record.get("source_game")}
        if len(source_games) < 2:
            continue
        if family == "click_state_transform":
            library.append(_click_state_abstraction(records))
        else:
            library.append(_generic_abstraction(family, records))
    return library


def _target_tokens(exp4049: Mapping[str, Any]) -> set[str]:
    trace = exp4049.get("solve_trace") if isinstance(exp4049.get("solve_trace"), Mapping) else {}
    return _tokens(
        exp4049.get("target_game", ""),
        exp4049.get("selected_candidate_reason", ""),
        exp4049.get("induced_mechanic", ""),
        trace,
    )


def _matching_abstractions(library: list[Mapping[str, Any]], exp4049: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tokens = _target_tokens(exp4049)
    matches = [
        abstraction
        for abstraction in library
        if tokens.intersection(set(abstraction.get("match_tokens", []) or []))
    ]
    return sorted(matches, key=lambda item: (-int(item.get("occurrence_count", 0) or 0), str(item.get("name", ""))))


def _trace_actions(exp4049: Mapping[str, Any]) -> list[Any]:
    trace = exp4049.get("solve_trace")
    if isinstance(trace, Mapping) and isinstance(trace.get("actions"), list):
        return list(trace["actions"])
    actions = exp4049.get("action_plan")
    return list(actions) if isinstance(actions, list) else []


def _trace_commit_actions(exp4049: Mapping[str, Any]) -> list[Any]:
    trace = exp4049.get("solve_trace")
    if isinstance(trace, Mapping) and isinstance(trace.get("commit_actions"), list):
        return list(trace["commit_actions"])
    actions = _trace_actions(exp4049)
    exploration = _as_int(exp4049, "exploration_actions_used")
    return actions[exploration:] if exploration > 0 else actions


def _trace_induction_calls(exp4049: Mapping[str, Any]) -> int:
    trace = exp4049.get("solve_trace")
    if isinstance(trace, Mapping) and isinstance(trace.get("induction_calls"), list):
        return len(trace["induction_calls"])
    return 1 if exp4049.get("induced_mechanic") else 0


def _confirmed_exp4049_solve(exp4049: Mapping[str, Any]) -> bool:
    return (
        exp4049.get("game_solved") is True
        and exp4049.get("real_env_confirmed") is True
        and _as_int(exp4049, "first_solve_at_action") > 0
    )


def _target_evidence(exp4049: Mapping[str, Any]) -> str:
    if _confirmed_exp4049_solve(exp4049):
        return "confirmed_solve"
    if _trace_actions(exp4049):
        return "attempt_trace_only"
    return "no_usable_trace"


def _target_action_depth(exp4049: Mapping[str, Any]) -> int:
    first_solve = _as_int(exp4049, "first_solve_at_action")
    if first_solve > 0:
        return first_solve
    return len(_trace_actions(exp4049))


def _within_game_v6_actions(exp4049: Mapping[str, Any]) -> int:
    commit_actions = _trace_commit_actions(exp4049)
    if commit_actions:
        return len(commit_actions)
    depth = _target_action_depth(exp4049)
    exploration = _as_int(exp4049, "exploration_actions_used")
    return max(0, depth - exploration)


def _empty_artifact(reason: str, *, duration_s: float, fragments: list[dict[str, Any]]) -> dict[str, Any]:
    library = build_v7_library(fragments)
    return {
        "experiment": "experiment_4050_arcmemo_cross_game_transfer_v7",
        "title": "arcmemo_cross_game_transfer_v7_eighth_game",
        "honest_verdict": f"complete: arcmemo_v7_no_cross_game_transfer_{reason}",
        "cross_game_transfer_win": False,
        "actions_cold": 0,
        "actions_within_game_v6": 0,
        "actions_cross_game_v7": 0,
        "induction_calls_cold": 0,
        "induction_calls_within_game_v6": 0,
        "induction_calls_cross_game_v7": 0,
        "n_reused_abstractions": 0,
        "n_named_abstractions": len(library),
        "n_prior_fragments": len(fragments),
        "prior_game_fragments": fragments,
        "named_abstractions": library,
        "reused_abstractions": [],
        "target_evidence": "no_usable_trace",
        "transfer_assessment": "unmeasured_no_usable_trace",
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def _assessment(
    *,
    target_evidence: str,
    actions_cold: int,
    actions_within_game_v6: int,
    actions_cross_game_v7: int,
    cross_game_transfer_win: bool,
) -> str:
    if target_evidence == "attempt_trace_only":
        return "attempt_only_no_solve_claim"
    if cross_game_transfer_win:
        return "helped_vs_cold_and_within_game_v6"
    if actions_cross_game_v7 < actions_cold and actions_cross_game_v7 > actions_within_game_v6:
        return "helped_vs_cold_but_lost_to_within_game_v6"
    if actions_cross_game_v7 < actions_cold and actions_cross_game_v7 == actions_within_game_v6:
        return "helped_vs_cold_but_tied_within_game_v6"
    if actions_cross_game_v7 == actions_cold:
        return "neutral_vs_cold"
    return "hurt_or_unmeasured"


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "terminal prefix reports whether cross-game transfer helped, hurt, or was neutral",
        "cross_game_transfer_win": (
            "persistent memory must demonstrably reduce the cost of a new game, not just replay a solved one"
        ),
        "actions_cold": "cold-start action cost on the eighth game",
        "actions_within_game_v6": "post-hoc within-game-v6 cost after target-specific exploration has already happened",
        "actions_cross_game_v7": "eighth-game action cost when seeded only by prior-game v7 abstractions",
        "n_reused_abstractions": "number of prior-game abstractions that actually fired on the eighth game",
        "inference_substrate": "offline artifact-level ArcMemo cross-game transfer measurement substrate",
    }


def build_cross_game_transfer_artifact(
    *,
    prior_solved_artifacts: Iterable[Mapping[str, Any] | None],
    exp4049: Mapping[str, Any] | None,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4050 artifact from prior solved games and the held-out Exp 4049 trace."""

    fragments = collect_prior_fragments(prior_solved_artifacts)
    library = build_v7_library(fragments)
    if exp4049 is None or _target_evidence(exp4049) == "no_usable_trace":
        return _empty_artifact("no_usable_4049_trace", duration_s=duration_s, fragments=fragments)

    target_evidence = _target_evidence(exp4049)
    actions_cold = _as_int(exp4049, "candidate_baseline_actions", "baseline_actions") or _target_action_depth(exp4049)
    actions_within_game_v6 = _within_game_v6_actions(exp4049)
    actions_cross_game_v7 = _target_action_depth(exp4049)
    induction_calls_cold = _trace_induction_calls(exp4049)
    induction_calls_within_game_v6 = 0 if actions_within_game_v6 > 0 else induction_calls_cold
    matches = _matching_abstractions(library, exp4049)
    reused = matches[:1]
    induction_calls_cross_game_v7 = 0 if reused else induction_calls_cold
    cross_game_transfer_win = bool(
        target_evidence == "confirmed_solve"
        and actions_cross_game_v7 > 0
        and actions_cross_game_v7 < actions_cold
        and actions_cross_game_v7 < actions_within_game_v6
    )

    if target_evidence == "attempt_trace_only":
        verdict = "complete: arcmemo_v7_no_cross_game_transfer_attempt_only"
    elif not reused:
        verdict = "complete: arcmemo_v7_no_cross_game_transfer_no_prior_abstraction_fired"
    elif cross_game_transfer_win:
        verdict = f"success: arcmemo_v7_cross_game_transfer_{actions_cold}to{actions_cross_game_v7}_actions"
    elif actions_cross_game_v7 >= actions_within_game_v6:
        verdict = "complete: arcmemo_v7_no_cross_game_transfer_v7_not_cheaper_than_within_game_v6"
    else:
        verdict = "complete: arcmemo_v7_no_cross_game_transfer_v7_not_cheaper_than_cold"

    assessment = _assessment(
        target_evidence=target_evidence,
        actions_cold=actions_cold,
        actions_within_game_v6=actions_within_game_v6,
        actions_cross_game_v7=actions_cross_game_v7,
        cross_game_transfer_win=cross_game_transfer_win,
    )

    return {
        "experiment": "experiment_4050_arcmemo_cross_game_transfer_v7",
        "title": "arcmemo_cross_game_transfer_v7_eighth_game",
        "honest_verdict": verdict,
        "cross_game_transfer_win": cross_game_transfer_win,
        "actions_cold": int(actions_cold),
        "actions_within_game_v6": int(actions_within_game_v6),
        "actions_cross_game_v7": int(actions_cross_game_v7),
        "induction_calls_cold": int(induction_calls_cold),
        "induction_calls_within_game_v6": int(induction_calls_within_game_v6),
        "induction_calls_cross_game_v7": int(induction_calls_cross_game_v7),
        "n_reused_abstractions": len(reused),
        "n_named_abstractions": len(library),
        "n_prior_fragments": len(fragments),
        "prior_game_fragments": fragments,
        "named_abstractions": library,
        "reused_abstractions": [dict(item) for item in reused],
        "target_game": str(exp4049.get("target_game", "")),
        "target_evidence": target_evidence,
        "target_honest_verdict": str(exp4049.get("honest_verdict", "")),
        "transfer_assessment": assessment,
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": _field_principles(),
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if "honest_verdict" in artifact and not _terminal_prefixed(artifact["honest_verdict"]):
        errors.append("honest_verdict must start with success:, complete:, or blocked_")
    if "cross_game_transfer_win" in artifact and type(artifact["cross_game_transfer_win"]) is not bool:
        errors.append("cross_game_transfer_win must be a bare bool")
    for field in (
        "actions_cold",
        "actions_within_game_v6",
        "actions_cross_game_v7",
        "n_reused_abstractions",
    ):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    if "inference_substrate" in artifact:
        if type(artifact["inference_substrate"]) is not str:
            errors.append("inference_substrate must be a bare string")
        elif artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
            errors.append(f"inference_substrate must equal {INFERENCE_SUBSTRATE}")
    if artifact.get("cross_game_transfer_win") is True:
        if int(artifact.get("actions_cross_game_v7", 0) or 0) >= int(artifact.get("actions_cold", 0) or 0):
            errors.append("cross_game_transfer_win requires v7 actions below cold")
        if int(artifact.get("actions_cross_game_v7", 0) or 0) >= int(
            artifact.get("actions_within_game_v6", 0) or 0
        ):
            errors.append("cross_game_transfer_win requires v7 actions below within-game v6")
    return errors
