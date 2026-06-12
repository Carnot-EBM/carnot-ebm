"""ArcMemo v9 richer cross-game concept-library transfer accounting.

Spec refs: REQ-LEARN-4072, SCENARIO-LEARN-4072.

Exp 4072 reuses the v8 artifact-level accounting pattern against the fresh
Exp 4070 ninth-game trace. The target trace is measured honestly: if it solved,
the v9 library must still beat both cold start and within-game memory before
claiming cross-game transfer; if it did not solve, only the bounded attempt
depth is measured.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import carnot.agentic.arc_arcmemo_cross_game_transfer_v7 as v7
import carnot.agentic.arc_arcmemo_cross_game_transfer_v8 as v8


INFERENCE_SUBSTRATE = "offline_arc_agi3_arcmemo_cross_game_transfer"
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "cross_game_transfer_win",
    "actions_cold",
    "actions_within_game",
    "actions_cross_game_v9",
    "n_reused_abstractions",
    "inference_substrate",
)
_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "before",
    "by",
    "can",
    "from",
    "is",
    "it",
    "of",
    "or",
    "s",
    "than",
    "the",
    "then",
    "to",
    "with",
}


def collect_prior_fragments(prior_solved_artifacts: Iterable[Mapping[str, Any] | None]) -> list[dict[str, Any]]:
    """Collect named induced-program fragments from all eight prior solved games."""

    return v8.collect_prior_fragments(prior_solved_artifacts)


def _source_names(records: Iterable[Mapping[str, Any]], key: str) -> list[str]:
    return sorted({str(record.get(key, "")) for record in records if record.get(key)})


def _source_fragments(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return _source_names(records, "name")


def _source_games(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return _source_names(records, "source_game")


def _source_artifacts(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return _source_names(records, "source_artifact")


def _abstraction(
    *,
    name: str,
    family: str,
    signature: str,
    documentation: str,
    lambda_abstraction: str,
    records: list[Mapping[str, Any]],
    match_tokens: Iterable[str],
    min_match_tokens: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "signature": signature,
        "documentation": documentation,
        "lambda_abstraction": lambda_abstraction,
        "source_fragments": _source_fragments(records),
        "source_games": _source_games(records),
        "source_artifacts": _source_artifacts(records),
        "occurrence_count": len(records),
        "match_tokens": sorted(set(match_tokens)),
        "min_match_tokens": int(min_match_tokens),
    }


def _click_state_abstraction(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    return _abstraction(
        name="click_state_transform_then_goal_commit_v9",
        family="click_state_transform",
        signature=(
            "click_state_transform_then_goal_commit_v9("
            "observed_state, click_or_toggle_ops, terminal_goal_counter) -> executable_action_plan"
        ),
        documentation=(
            "Why: many prior solves use clicks or toggles to change state before a terminal goal check; "
            "v9 keeps that repeated operation shape as a named reusable fragment."
        ),
        lambda_abstraction=(
            "lambda observed_state: bind_v9_click_state_transform(observed_state) "
            "then plan_target_specific_goal_commit(observed_state)"
        ),
        records=records,
        match_tokens=[
            "cell",
            "cells",
            "click",
            "clicking",
            "color",
            "goal",
            "grid",
            "state",
            "target",
            "toggle",
            "toggles",
        ],
        min_match_tokens=1,
    )


def _local_constraint_click_state_update(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    return _abstraction(
        name="local_constraint_click_state_update_v9",
        family="click_state_transform",
        signature=(
            "local_constraint_click_state_update_v9("
            "observed_cells, click_updates, local_constraints) -> satisfied_constraint_plan"
        ),
        documentation=(
            "Why: prior click-state games repeatedly show local state edits before a constraint or pattern "
            "is satisfied; v9 makes that local edit/search loop explicit instead of only preserving a "
            "generic click suffix."
        ),
        lambda_abstraction=(
            "lambda observed_state: infer_local_click_update_rule(observed_state) "
            "then click_unsatisfied_local_constraints(observed_state)"
        ),
        records=records,
        match_tokens=[
            "cell",
            "cells",
            "click",
            "clicking",
            "constraint",
            "cycle",
            "grid",
            "local",
            "pattern",
            "state",
            "target",
            "toggle",
            "violation",
        ],
        min_match_tokens=2,
    )


def _select_place_color_sequence(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    return _abstraction(
        name="select_place_color_sequence_v9",
        family="click_state_transform",
        signature="select_place_color_sequence_v9(state, target_colors, slots) -> seeded_action_plan",
        documentation=(
            "Why: the eighth solve contributed a specific select/place/color sequence that remains useful "
            "for slot-filling games, but v9 requires slot-placement evidence before it fires."
        ),
        lambda_abstraction=(
            "lambda observed_state: order_items_by_target_colors(observed_state) "
            "then emit_select_place_validate_plan(observed_state)"
        ),
        records=records,
        match_tokens=["item", "place", "select", "sequence", "slot", "slots", "validate"],
        min_match_tokens=2,
    )


def _color_state_commit(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    return _abstraction(
        name="color_state_update_then_constraint_commit_v9",
        family="color_state_update",
        signature=(
            "color_state_update_then_constraint_commit_v9("
            "observed_state, color_updates, verifier_constraints) -> color_goal_plan"
        ),
        documentation=(
            "Why: prior solved games include color-sensitive placement or paint commits; v9 exposes this "
            "as a color-state update abstraction that can seed color-cycle or color-fill targets."
        ),
        lambda_abstraction=(
            "lambda observed_state: bind_color_state_update(observed_state) "
            "then commit_when_color_constraints_pass(observed_state)"
        ),
        records=records,
        match_tokens=["action5", "cell", "color", "colors", "constraint", "cycle", "paint", "target"],
        min_match_tokens=2,
    )


def _family_abstraction(family: str, records: list[Mapping[str, Any]]) -> dict[str, Any]:
    tokens = sorted(set().union(*(set(record.get("tokens", [])) for record in records)) - _STOP_TOKENS)[:16]
    return _abstraction(
        name=f"{family}_v9_named_fragment",
        family=family,
        signature=f"{family}_v9_named_fragment(observed_state, reusable_ops) -> candidate_action_plan",
        documentation=(
            "Why: v9 keeps every prior solved-game family as an auditable named fragment, including "
            "singleton families that are not strong enough by themselves to prove transfer."
        ),
        lambda_abstraction="lambda observed_state: reuse_v9_named_family_fragment(observed_state, reusable_ops)",
        records=records,
        match_tokens=tokens,
        min_match_tokens=4,
    )


def build_v9_library(prior_fragments: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compress eight-game fragments into richer named v9 library abstractions."""

    records = list(prior_fragments)
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for fragment in records:
        by_family[str(fragment.get("family", "unknown_induced_program"))].append(fragment)

    library: list[dict[str, Any]] = []
    click_records = by_family.get("click_state_transform", [])
    if len(_source_games(click_records)) >= 2:
        library.append(_click_state_abstraction(click_records))
        library.append(_local_constraint_click_state_update(click_records))

    color_records = [
        record
        for record in records
        if {"action5", "color", "colors", "paint", "slot"}.intersection(set(record.get("tokens", []) or []))
    ]
    if len(_source_games(color_records)) >= 2:
        library.append(_color_state_commit(color_records))

    sb26_records = [record for record in records if record.get("source_game") == "sb26"]
    if sb26_records:
        library.append(_select_place_color_sequence(sb26_records))

    for family in sorted(by_family):
        if family == "click_state_transform":
            continue
        library.append(_family_abstraction(family, by_family[family]))
    return library


def _target_tokens(exp4070: Mapping[str, Any]) -> set[str]:
    trace = exp4070.get("solve_trace") if isinstance(exp4070.get("solve_trace"), Mapping) else {}
    return v7._tokens(
        exp4070.get("target_game", ""),
        exp4070.get("selected_candidate_reason", ""),
        exp4070.get("induced_mechanic", ""),
        exp4070.get("v9_seeded_action_plan", ""),
        exp4070.get("phase_trace", ""),
        trace,
    )


def _matching_abstractions(library: list[Mapping[str, Any]], exp4070: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tokens = _target_tokens(exp4070)
    matches: list[dict[str, Any]] = []
    for abstraction in library:
        matched_tokens = sorted(tokens.intersection(set(abstraction.get("match_tokens", []) or [])))
        min_match_tokens = int(abstraction.get("min_match_tokens", 1) or 1)
        if len(matched_tokens) >= min_match_tokens:
            match = dict(abstraction)
            match["matched_tokens"] = matched_tokens
            matches.append(match)
    return sorted(matches, key=lambda item: (-int(item.get("occurrence_count", 0) or 0), str(item.get("name", ""))))


def _trace_actions(exp4070: Mapping[str, Any]) -> list[Any]:
    return v8._trace_actions(exp4070)


def _trace_induction_calls(exp4070: Mapping[str, Any]) -> int:
    return v8._trace_induction_calls(exp4070)


def _confirmed_exp4070_solve(exp4070: Mapping[str, Any]) -> bool:
    return (
        exp4070.get("game_solved") is True
        and exp4070.get("real_env_confirmed") is True
        and v7._as_int(exp4070, "first_solve_at_action") > 0
    )


def _target_evidence(exp4070: Mapping[str, Any]) -> str:
    if _confirmed_exp4070_solve(exp4070):
        return "confirmed_solve"
    if _trace_actions(exp4070):
        return "attempt_trace_only"
    return "no_usable_trace"


def _target_action_depth(exp4070: Mapping[str, Any]) -> int:
    return v8._target_action_depth(exp4070)


def _within_game_actions(exp4070: Mapping[str, Any]) -> int:
    return v8._within_game_actions(exp4070)


def _seeded_actions(exp4070: Mapping[str, Any]) -> list[Any]:
    for key in ("v9_seeded_action_plan", "cross_game_v9_seeded_action_plan", "cross_game_seeded_action_plan"):
        actions = exp4070.get(key)
        if isinstance(actions, list):
            return list(actions)
    return []


def _cross_game_action_count(exp4070: Mapping[str, Any], reused: list[Mapping[str, Any]]) -> int:
    seeded_actions = _seeded_actions(exp4070)
    if reused and seeded_actions:
        return len(seeded_actions)
    return _target_action_depth(exp4070)


def _target_future_fragments(exp4070: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if exp4070 is None or not _confirmed_exp4070_solve(exp4070):
        return []
    target_game = str(exp4070.get("target_game", "ft09"))
    source_game = target_game.split("-", maxsplit=1)[0]
    mechanic = str(exp4070.get("induced_mechanic", ""))
    return [
        {
            "name": f"{source_game}_local_constraint_color_cycle",
            "family": "target_derived_future_fragment",
            "source_game": source_game,
            "source_artifact": "results/experiment_4070_ninth_game_explore_first.json",
            "signature": (
                f"{source_game}_local_constraint_color_cycle(observed_cells, color_cycle, constraints) "
                "-> satisfied_constraint_plan"
            ),
            "documentation": (
                "Why: Exp 4070 solved a local color-cycle constraint game; this fragment is preserved "
                "for future games but excluded from the held-out Exp 4072 transfer claim."
            ),
            "lambda_abstraction": (
                "lambda observed_state: cycle_cells_until_local_equality_inequality_constraints_pass("
                "observed_state)"
            ),
            "induced_program_fragment": mechanic,
            "excluded_from_exp4072_transfer_claim": True,
            "tokens": sorted(v7._tokens(source_game, target_game, mechanic, exp4070.get("solve_trace", ""))),
        }
    ]


def _empty_artifact(reason: str, *, duration_s: float, fragments: list[dict[str, Any]]) -> dict[str, Any]:
    library = build_v9_library(fragments)
    return {
        "experiment": "experiment_4072_arcmemo_cross_game_transfer_v9",
        "title": "arcmemo_cross_game_transfer_v9_ninth_game",
        "honest_verdict": f"complete: arcmemo_v9_no_cross_game_transfer_{reason}",
        "cross_game_transfer_win": False,
        "actions_cold": 0,
        "actions_within_game": 0,
        "actions_cross_game_v9": 0,
        "induction_calls_cold": 0,
        "induction_calls_within_game": 0,
        "induction_calls_cross_game_v9": 0,
        "n_reused_abstractions": 0,
        "n_named_abstractions": len(library),
        "n_prior_fragments": len(fragments),
        "prior_game_fragments": fragments,
        "named_abstractions": library,
        "reused_abstractions": [],
        "target_future_fragments": [],
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
    actions_cross_game_v9: int,
    cross_game_transfer_win: bool,
) -> str:
    if target_evidence == "attempt_trace_only":
        return "attempt_only_no_solve_claim"
    if cross_game_transfer_win:
        return "helped_vs_cold_and_within_game"
    if actions_cross_game_v9 < actions_cold and actions_cross_game_v9 > actions_within_game:
        return "helped_vs_cold_but_lost_to_within_game"
    if actions_cross_game_v9 < actions_cold and actions_cross_game_v9 == actions_within_game:
        return "helped_vs_cold_but_tied_within_game"
    if actions_cross_game_v9 == actions_cold:
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
        "actions_cross_game_v9": "ninth-game action cost when seeded by the eight-game v9 abstraction library",
        "n_reused_abstractions": "number of prior-game v9 abstractions that actually fired on the ninth game",
        "inference_substrate": "offline artifact-level ArcMemo cross-game transfer measurement substrate",
    }


def build_cross_game_transfer_artifact(
    *,
    prior_solved_artifacts: Iterable[Mapping[str, Any] | None],
    exp4070: Mapping[str, Any] | None,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4072 artifact from eight prior solves and the Exp 4070 trace."""

    fragments = collect_prior_fragments(prior_solved_artifacts)
    library = build_v9_library(fragments)
    if exp4070 is None or _target_evidence(exp4070) == "no_usable_trace":
        return _empty_artifact("no_usable_4070_trace", duration_s=duration_s, fragments=fragments)

    target_evidence = _target_evidence(exp4070)
    reused = _matching_abstractions(library, exp4070)
    actions_cold = v7._as_int(exp4070, "candidate_baseline_actions", "baseline_actions") or _target_action_depth(
        exp4070
    )
    actions_within_game = _within_game_actions(exp4070)
    actions_cross_game_v9 = _cross_game_action_count(exp4070, reused)
    induction_calls_cold = _trace_induction_calls(exp4070)
    induction_calls_within_game = 0 if actions_within_game > 0 else induction_calls_cold
    induction_calls_cross_game_v9 = 0 if reused else induction_calls_cold
    cross_game_transfer_win = bool(
        target_evidence == "confirmed_solve"
        and actions_cross_game_v9 > 0
        and actions_cross_game_v9 < actions_cold
        and actions_cross_game_v9 < actions_within_game
    )

    if target_evidence == "attempt_trace_only":
        verdict = "complete: arcmemo_v9_no_cross_game_transfer_attempt_only"
    elif not reused:
        verdict = "complete: arcmemo_v9_no_cross_game_transfer_no_prior_abstraction_fired"
    elif cross_game_transfer_win:
        verdict = f"success: arcmemo_v9_cross_game_transfer_{actions_cold}to{actions_cross_game_v9}_actions"
    elif actions_cross_game_v9 >= actions_within_game:
        verdict = "complete: arcmemo_v9_no_cross_game_transfer_v9_not_cheaper_than_within_game"
    else:
        verdict = "complete: arcmemo_v9_no_cross_game_transfer_v9_not_cheaper_than_cold"

    return {
        "experiment": "experiment_4072_arcmemo_cross_game_transfer_v9",
        "title": "arcmemo_cross_game_transfer_v9_ninth_game",
        "honest_verdict": verdict,
        "cross_game_transfer_win": cross_game_transfer_win,
        "actions_cold": int(actions_cold),
        "actions_within_game": int(actions_within_game),
        "actions_cross_game_v9": int(actions_cross_game_v9),
        "induction_calls_cold": int(induction_calls_cold),
        "induction_calls_within_game": int(induction_calls_within_game),
        "induction_calls_cross_game_v9": int(induction_calls_cross_game_v9),
        "n_reused_abstractions": len(reused),
        "n_named_abstractions": len(library),
        "n_prior_fragments": len(fragments),
        "prior_game_fragments": fragments,
        "named_abstractions": library,
        "reused_abstractions": [dict(item) for item in reused],
        "target_future_fragments": _target_future_fragments(exp4070),
        "target_game": str(exp4070.get("target_game", "")),
        "target_evidence": target_evidence,
        "target_honest_verdict": str(exp4070.get("honest_verdict", "")),
        "transfer_assessment": _assessment(
            target_evidence=target_evidence,
            actions_cold=actions_cold,
            actions_within_game=actions_within_game,
            actions_cross_game_v9=actions_cross_game_v9,
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
    for field in ("actions_cold", "actions_within_game", "actions_cross_game_v9", "n_reused_abstractions"):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    if "inference_substrate" in artifact:
        if type(artifact["inference_substrate"]) is not str:
            errors.append("inference_substrate must be a bare string")
        elif artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
            errors.append(f"inference_substrate must equal {INFERENCE_SUBSTRATE}")
    if artifact.get("cross_game_transfer_win") is True:
        if int(artifact.get("actions_cross_game_v9", 0) or 0) >= int(artifact.get("actions_cold", 0) or 0):
            errors.append("cross_game_transfer_win requires v9 actions below cold")
        if int(artifact.get("actions_cross_game_v9", 0) or 0) >= int(
            artifact.get("actions_within_game", 0) or 0
        ):
            errors.append("cross_game_transfer_win requires v9 actions below within-game")
    return errors
