"""ArcMemo v8 richer cross-game concept-library transfer accounting.

Spec refs: REQ-LEARN-4062, SCENARIO-LEARN-4062.

Exp 4062 measures whether a library built from eight prior solved ARC-AGI-3
games can reduce the cost of a ninth game. The module works at artifact level:
it reads prior solve traces, names reusable fragments, and compares the ninth
game against cold and within-game memory without inventing a solve when the
ninth-game trace is missing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import carnot.agentic.arc_arcmemo_cross_game_transfer_v7 as v7


INFERENCE_SUBSTRATE = "offline_arc_agi3_arcmemo_cross_game_transfer"
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "cross_game_transfer_win",
    "actions_cold",
    "actions_within_game",
    "actions_cross_game_v8",
    "n_reused_abstractions",
    "inference_substrate",
)

_SB26_TEMPLATE = {
    "name": "select_place_color_sequence",
    "family": "click_state_transform",
    "signature": "select_place_color_sequence(state, target_colors, slots) -> validated_slot_sequence",
    "lambda_abstraction": "lambda state: select_items_by_color(state) then place_slots_left_to_right(state)",
    "documentation": (
        "Why: the eighth solved game added a concrete select/place/color-sequence fragment that can seed "
        "slot-filling games without relearning the same item-to-slot mechanic."
    ),
}


def _template_for(source_game: str) -> dict[str, str]:
    if source_game == "sb26":
        return _SB26_TEMPLATE
    return v7._template_for(source_game)


def collect_prior_fragments(prior_solved_artifacts: Iterable[Mapping[str, Any] | None]) -> list[dict[str, Any]]:
    """Collect named induced-program fragments from all eight prior solved games."""

    fragments: list[dict[str, Any]] = []
    for row in prior_solved_artifacts:
        payload = v7._row_payload(row)
        if payload is None or row is None or not v7._confirmed_prior_solve(payload):
            continue
        source_game = v7._source_game_from_row(row, payload)
        template = _template_for(source_game)
        mechanic = v7._mechanic_text(payload)
        fragments.append(
            {
                "name": template["name"],
                "family": template["family"],
                "source_game": source_game,
                "source_artifact": v7._source_artifact_from_row(row),
                "signature": template["signature"],
                "documentation": template["documentation"],
                "lambda_abstraction": template["lambda_abstraction"],
                "induced_program_fragment": mechanic,
                "tokens": sorted(v7._tokens(source_game, template["name"], template["family"], mechanic)),
            }
        )
    return fragments


def _click_state_abstraction(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "name": "click_state_transform_then_goal_commit_v8",
        "family": "click_state_transform",
        "signature": (
            "click_state_transform_then_goal_commit_v8("
            "observed_state, click_or_toggle_ops, terminal_goal_counter) -> executable_action_plan"
        ),
        "documentation": (
            "Why: the richer eight-game library preserves the recurring click-driven state-change pattern "
            "from prior games while still requiring the ninth game to provide its own trace evidence."
        ),
        "lambda_abstraction": (
            "lambda observed_state: bind_v8_click_state_transform(observed_state) "
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


def _select_place_color_sequence(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "name": "select_place_color_sequence_v8",
        "family": "click_state_transform",
        "signature": "select_place_color_sequence_v8(state, target_colors, slots) -> seeded_action_plan",
        "documentation": (
            "Why: sb26 turned the broad click-state family into a specific reusable slot-color sequence "
            "program fragment, which is the extra v8 library content absent from the seven-game v7 library."
        ),
        "lambda_abstraction": (
            "lambda observed_state: order_items_by_target_colors(observed_state) "
            "then emit_select_place_validate_plan(observed_state)"
        ),
        "source_fragments": sorted({str(record["name"]) for record in records}),
        "source_games": sorted({str(record["source_game"]) for record in records}),
        "source_artifacts": sorted({str(record["source_artifact"]) for record in records}),
        "occurrence_count": len(records),
        "match_tokens": ["color", "colors", "item", "place", "select", "sequence", "slot", "target", "validate"],
    }


def build_v8_library(prior_fragments: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compress eight-game fragments into named v8 library abstractions."""

    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    sb26_records: list[Mapping[str, Any]] = []
    for fragment in prior_fragments:
        by_family[str(fragment.get("family", "unknown_induced_program"))].append(fragment)
        if fragment.get("source_game") == "sb26":
            sb26_records.append(fragment)

    library: list[dict[str, Any]] = []
    click_records = by_family.get("click_state_transform", [])
    source_games = {str(record.get("source_game", "")) for record in click_records if record.get("source_game")}
    if len(source_games) >= 2:
        library.append(_click_state_abstraction(click_records))
    if sb26_records:
        library.append(_select_place_color_sequence(sb26_records))
    return library


def _target_tokens(exp4060: Mapping[str, Any]) -> set[str]:
    trace = exp4060.get("solve_trace") if isinstance(exp4060.get("solve_trace"), Mapping) else {}
    return v7._tokens(
        exp4060.get("target_game", ""),
        exp4060.get("selected_candidate_reason", ""),
        exp4060.get("induced_mechanic", ""),
        exp4060.get("v8_seeded_action_plan", ""),
        trace,
    )


def _matching_abstractions(library: list[Mapping[str, Any]], exp4060: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tokens = _target_tokens(exp4060)
    matches = [
        abstraction
        for abstraction in library
        if tokens.intersection(set(abstraction.get("match_tokens", []) or []))
    ]
    return sorted(matches, key=lambda item: (-int(item.get("occurrence_count", 0) or 0), str(item.get("name", ""))))


def _trace_actions(exp4060: Mapping[str, Any]) -> list[Any]:
    trace = exp4060.get("solve_trace")
    if isinstance(trace, Mapping) and isinstance(trace.get("actions"), list):
        return list(trace["actions"])
    actions = exp4060.get("action_plan")
    return list(actions) if isinstance(actions, list) else []


def _trace_commit_actions(exp4060: Mapping[str, Any]) -> list[Any]:
    trace = exp4060.get("solve_trace")
    if isinstance(trace, Mapping) and isinstance(trace.get("commit_actions"), list):
        return list(trace["commit_actions"])
    actions = _trace_actions(exp4060)
    exploration = v7._as_int(exp4060, "exploration_actions_used")
    return actions[exploration:] if exploration > 0 else actions


def _trace_induction_calls(exp4060: Mapping[str, Any]) -> int:
    trace = exp4060.get("solve_trace")
    if isinstance(trace, Mapping) and isinstance(trace.get("induction_calls"), list):
        return len(trace["induction_calls"])
    return 1 if exp4060.get("induced_mechanic") else 0


def _confirmed_exp4060_solve(exp4060: Mapping[str, Any]) -> bool:
    return (
        exp4060.get("game_solved") is True
        and exp4060.get("real_env_confirmed") is True
        and v7._as_int(exp4060, "first_solve_at_action") > 0
    )


def _target_evidence(exp4060: Mapping[str, Any]) -> str:
    if _confirmed_exp4060_solve(exp4060):
        return "confirmed_solve"
    if _trace_actions(exp4060):
        return "attempt_trace_only"
    return "no_usable_trace"


def _target_action_depth(exp4060: Mapping[str, Any]) -> int:
    first_solve = v7._as_int(exp4060, "first_solve_at_action")
    if first_solve > 0:
        return first_solve
    return len(_trace_actions(exp4060))


def _within_game_actions(exp4060: Mapping[str, Any]) -> int:
    commit_actions = _trace_commit_actions(exp4060)
    if commit_actions:
        return len(commit_actions)
    depth = _target_action_depth(exp4060)
    exploration = v7._as_int(exp4060, "exploration_actions_used")
    return max(0, depth - exploration)


def _seeded_actions(exp4060: Mapping[str, Any]) -> list[Any]:
    for key in ("v8_seeded_action_plan", "cross_game_seeded_action_plan", "seeded_action_plan"):
        actions = exp4060.get(key)
        if isinstance(actions, list):
            return list(actions)
    return []


def _cross_game_action_count(exp4060: Mapping[str, Any], reused: list[Mapping[str, Any]]) -> int:
    seeded_actions = _seeded_actions(exp4060)
    if reused and seeded_actions:
        return len(seeded_actions)
    return _target_action_depth(exp4060)


def _empty_artifact(reason: str, *, duration_s: float, fragments: list[dict[str, Any]]) -> dict[str, Any]:
    library = build_v8_library(fragments)
    return {
        "experiment": "experiment_4062_arcmemo_cross_game_transfer_v8",
        "title": "arcmemo_cross_game_transfer_v8_ninth_game",
        "honest_verdict": f"complete: arcmemo_v8_no_cross_game_transfer_{reason}",
        "cross_game_transfer_win": False,
        "actions_cold": 0,
        "actions_within_game": 0,
        "actions_cross_game_v8": 0,
        "induction_calls_cold": 0,
        "induction_calls_within_game": 0,
        "induction_calls_cross_game_v8": 0,
        "n_reused_abstractions": 0,
        "n_named_abstractions": len(library),
        "n_prior_fragments": len(fragments),
        "prior_game_fragments": fragments,
        "named_abstractions": library,
        "reused_abstractions": [],
        "target_evidence": "no_usable_trace",
        "target_honest_verdict": "",
        "transfer_assessment": "unmeasured_no_usable_trace",
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": _field_principles(),
    }


def _assessment(
    *,
    target_evidence: str,
    actions_cold: int,
    actions_within_game: int,
    actions_cross_game_v8: int,
    cross_game_transfer_win: bool,
) -> str:
    if target_evidence == "attempt_trace_only":
        return "attempt_only_no_solve_claim"
    if cross_game_transfer_win:
        return "helped_vs_cold_and_within_game"
    if actions_cross_game_v8 < actions_cold and actions_cross_game_v8 > actions_within_game:
        return "helped_vs_cold_but_lost_to_within_game"
    if actions_cross_game_v8 < actions_cold and actions_cross_game_v8 == actions_within_game:
        return "helped_vs_cold_but_tied_within_game"
    if actions_cross_game_v8 == actions_cold:
        return "neutral_vs_cold"
    return "hurt_or_unmeasured"


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "terminal prefix reports whether cross-game transfer helped, hurt, or was neutral",
        "cross_game_transfer_win": (
            "persistent memory must demonstrably reduce the cost of a new game below both cold and within-game memory"
        ),
        "actions_cold": "cold-start action cost on the ninth game or zero if no usable ninth-game trace exists",
        "actions_within_game": "within-game-only memory cost after target-specific exploration has already happened",
        "actions_cross_game_v8": "ninth-game action cost when seeded by the eight-game v8 abstraction library",
        "n_reused_abstractions": "number of v8 abstractions that actually fired on the ninth game",
        "inference_substrate": "offline artifact-level ArcMemo cross-game transfer measurement substrate",
    }


def build_cross_game_transfer_artifact(
    *,
    prior_solved_artifacts: Iterable[Mapping[str, Any] | None],
    exp4060: Mapping[str, Any] | None,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4062 artifact from eight prior solves and the held-out Exp 4060 trace."""

    fragments = collect_prior_fragments(prior_solved_artifacts)
    library = build_v8_library(fragments)
    if exp4060 is None or _target_evidence(exp4060) == "no_usable_trace":
        return _empty_artifact("no_usable_4060_trace", duration_s=duration_s, fragments=fragments)

    target_evidence = _target_evidence(exp4060)
    matches = _matching_abstractions(library, exp4060)
    reused = matches
    actions_cold = v7._as_int(exp4060, "candidate_baseline_actions", "baseline_actions") or _target_action_depth(
        exp4060
    )
    actions_within_game = _within_game_actions(exp4060)
    actions_cross_game_v8 = _cross_game_action_count(exp4060, reused)
    induction_calls_cold = _trace_induction_calls(exp4060)
    induction_calls_within_game = 0 if actions_within_game > 0 else induction_calls_cold
    induction_calls_cross_game_v8 = 0 if reused else induction_calls_cold
    cross_game_transfer_win = bool(
        target_evidence == "confirmed_solve"
        and actions_cross_game_v8 > 0
        and actions_cross_game_v8 < actions_cold
        and actions_cross_game_v8 < actions_within_game
    )

    if target_evidence == "attempt_trace_only":
        verdict = "complete: arcmemo_v8_no_cross_game_transfer_attempt_only"
    elif not reused:
        verdict = "complete: arcmemo_v8_no_cross_game_transfer_no_prior_abstraction_fired"
    elif cross_game_transfer_win:
        verdict = f"success: arcmemo_v8_cross_game_transfer_{actions_cold}to{actions_cross_game_v8}_actions"
    elif actions_cross_game_v8 >= actions_within_game:
        verdict = "complete: arcmemo_v8_no_cross_game_transfer_v8_not_cheaper_than_within_game"
    else:
        verdict = "complete: arcmemo_v8_no_cross_game_transfer_v8_not_cheaper_than_cold"

    return {
        "experiment": "experiment_4062_arcmemo_cross_game_transfer_v8",
        "title": "arcmemo_cross_game_transfer_v8_ninth_game",
        "honest_verdict": verdict,
        "cross_game_transfer_win": cross_game_transfer_win,
        "actions_cold": int(actions_cold),
        "actions_within_game": int(actions_within_game),
        "actions_cross_game_v8": int(actions_cross_game_v8),
        "induction_calls_cold": int(induction_calls_cold),
        "induction_calls_within_game": int(induction_calls_within_game),
        "induction_calls_cross_game_v8": int(induction_calls_cross_game_v8),
        "n_reused_abstractions": len(reused),
        "n_named_abstractions": len(library),
        "n_prior_fragments": len(fragments),
        "prior_game_fragments": fragments,
        "named_abstractions": library,
        "reused_abstractions": [dict(item) for item in reused],
        "target_game": str(exp4060.get("target_game", "")),
        "target_evidence": target_evidence,
        "target_honest_verdict": str(exp4060.get("honest_verdict", "")),
        "transfer_assessment": _assessment(
            target_evidence=target_evidence,
            actions_cold=actions_cold,
            actions_within_game=actions_within_game,
            actions_cross_game_v8=actions_cross_game_v8,
            cross_game_transfer_win=cross_game_transfer_win,
        ),
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": _field_principles(),
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if "honest_verdict" in artifact and not v7._terminal_prefixed(artifact["honest_verdict"]):
        errors.append("honest_verdict must start with success:, complete:, or blocked_")
    if "cross_game_transfer_win" in artifact and type(artifact["cross_game_transfer_win"]) is not bool:
        errors.append("cross_game_transfer_win must be a bare bool")
    for field in ("actions_cold", "actions_within_game", "actions_cross_game_v8", "n_reused_abstractions"):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    if "inference_substrate" in artifact:
        if type(artifact["inference_substrate"]) is not str:
            errors.append("inference_substrate must be a bare string")
        elif artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
            errors.append(f"inference_substrate must equal {INFERENCE_SUBSTRATE}")
    if artifact.get("cross_game_transfer_win") is True:
        if int(artifact.get("actions_cross_game_v8", 0) or 0) >= int(artifact.get("actions_cold", 0) or 0):
            errors.append("cross_game_transfer_win requires v8 actions below cold")
        if int(artifact.get("actions_cross_game_v8", 0) or 0) >= int(
            artifact.get("actions_within_game", 0) or 0
        ):
            errors.append("cross_game_transfer_win requires v8 actions below within-game")
    return errors
