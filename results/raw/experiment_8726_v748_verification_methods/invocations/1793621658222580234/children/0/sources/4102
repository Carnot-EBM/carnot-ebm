"""REQ-REPORT-8691: completed outcomes and censored exposure answer different questions."""

from __future__ import annotations

from typing import Any

from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER

Json = dict[str, Any]


def reduce(episodes: list[Json]) -> Json:
    """Join observations by action identity; equal pixels never establish duplicate actions."""
    rows: list[Json] = []
    games = sorted({ep["unit"]["game"] for ep in episodes})
    for ep in episodes:
        unit = ep["unit"]
        context = {k: unit[k] for k in ("game", "seed", "episode_id")}
        context["recorder"] = unit["reset_provenance"]
        canonical = [r for r in ep["delivery_rows"] if r["status"] == "canonical"]
        callbacks = {r["action_index"]: r for r in canonical}
        if len(callbacks) != len(canonical):
            raise ValueError("duplicate_action_owner")
        horizons = ep["observer_rows"]
        if not horizons:
            rows.append(dict(context, outcome="missing", arm=None, observation_count=None))
        for r in horizons:
            status, count, sequence = r["horizon_status"], r["observed_horizon"], r["sequence"]
            if r["horizon"] != 120 or not 0 <= count <= 120:
                raise ValueError("registered_horizon")
            if status not in {"censored", "no_progress_within_horizon", "level_progress_observed"}:
                raise ValueError("missing_or_unknown_outcome")
            # Reset observations can close a window without adding an observation count.
            if len(sequence) not in {count, count + int(status == "censored")}:
                raise ValueError("observation_count")
            joined = []
            for offset, observation in enumerate(sequence):
                index = observation["action_index"]
                callback = callbacks.get(index)
                if (
                    index != r["application_index"] + offset
                    or callback is None
                    or callback["action"] != observation["action"]
                    or ep["actions"][index - 1] != observation["action"]
                ):
                    raise ValueError("callback_action_join")
                joined.append(dict(action_index=index, owner=callback["owner"]))
            positive = status == "level_progress_observed"
            if (
                (status == "no_progress_within_horizon" and count != 120)
                or (positive != (r["level_progress"] > 0))
                or (positive and not r["applied"])
                or (positive and sequence[-1]["observation"]["level"] <= r["level_before"])
            ):
                raise ValueError("contradictory_outcome_false_promotion")
            last = sequence[-1]["action_index"] if sequence else r["application_index"]
            boundary = last + int(r["end_reason"] == "reset" and len(sequence) == count)
            requests = [
                x
                for x in ep["reset_provenance"]["reset_request_rows"]
                if x["action_index"] == boundary
            ]
            terminals = [
                x
                for x in ep["reset_provenance"]["terminal_reason_rows"]
                if x["action_index"] == boundary
            ]
            rows.append(
                dict(
                    context,
                    redirect_id=r["redirect_id"],
                    arm=r["arm"],
                    selected=r["selected"],
                    applied=r["applied"],
                    application_action_index=r["application_index"],
                    observation_count=count,
                    registered_horizon=120,
                    outcome="positive"
                    if positive
                    else "no_progress"
                    if status == "no_progress_within_horizon"
                    else "censored",
                    complete_outcome=positive or status == "no_progress_within_horizon",
                    censoring_boundary=dict(action_index=boundary, end_reason=r["end_reason"]),
                    reset_owner_cause=requests,
                    environment_termination=terminals,
                    boundary_class="policy_triggered_reset"
                    if requests
                    else "environment_termination"
                    if r["end_reason"] == "terminal"
                    else "unresolved_or_cutoff",
                    callback_action_join=joined,
                    reset_callback_join=[
                        x
                        for x in ep["reset_provenance"]["callback_sequence_rows"]
                        if x["action_index"] == boundary
                    ],
                    descriptive_actions_to_censoring=count if status == "censored" else None,
                    efficacy_estimate=None,
                )
            )
    complete = [r for r in rows if r.get("complete_outcome")]
    return dict(
        censoring_rows=rows,
        complete_horizon_count=len(complete),
        censored_count=sum(r["outcome"] == "censored" for r in rows),
        independent_count=len(games),
        per_game_results=[
            dict(
                game=g,
                episodes=sum(ep["unit"]["game"] == g for ep in episodes),
                complete_outcomes=sum(r["game"] == g for r in complete),
                censored_count=sum(r["game"] == g and r["outcome"] == "censored" for r in rows),
                descriptive_actions_to_censoring=sorted(
                    {
                        r["observation_count"]
                        for r in rows
                        if r["game"] == g and r["outcome"] == "censored"
                    }
                ),
            )
            for g in games
        ],
        arm_support_rows=[
            dict(
                arm=a,
                independent_games_with_uncensored_support=len(
                    {r["game"] for r in complete if r["arm"] == a}
                ),
                positive_outcomes=sum(
                    r["arm"] == a and r["outcome"] == "positive" for r in complete
                ),
                no_progress_outcomes=sum(
                    r["arm"] == a and r["outcome"] == "no_progress" for r in complete
                ),
                censored_outcomes=sum(r["arm"] == a and r["outcome"] == "censored" for r in rows),
                decision="retain_unchanged",
                efficacy_estimate=None,
            )
            for a in ARM_ORDER
        ],
        policy_decision="no_supported_policy_change",
        policy_frozen_unchanged=True,
        new_mechanism_prerequisites=[
            "Distinguish policy-triggered resets from environment termination before a new curated intervention; require uncensored cross-game support before retiring or reprioritizing any arm."
        ],
    )
