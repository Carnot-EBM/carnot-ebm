"""REQ-REPORT-8760: retain unknown binary outcomes over the intended roster."""

from itertools import combinations, product
from typing import Any

from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER
from carnot.reporting.arc_censoring_reduce_8691 import reduce as frozen_reduce

Json = dict[str, Any]
ARMS = list(ARM_ORDER)


def bounds(rows: list[Json]) -> Json:
    """Use [0,1] for every unavailable potential outcome, including absent arms.

    The same intended episode roster defines each arm's denominator. These
    logical bounds do not assume censoring is random. Games are the independent
    groups; recorder siblings and seeds remain descriptive exposure counts.
    """
    games = sorted({r["game"] for r in rows})
    cells, arms, overlaps = [], [], []
    for game, arm in product(games, ARMS):
        roster = [r for r in rows if r["game"] == game]
        observed = [r for r in roster if r["arm"] == arm and not r["masked"]]
        known = [r["binary_outcome"] for r in observed if r["binary_outcome"] is not None]
        positive = sum(known)
        cells.append(
            dict(
                game=game,
                arm=arm,
                intended_count=len(roster),
                observed_firings=sum(r["applied"] is True for r in observed),
                selected_count=sum(r["selected"] is True for r in observed),
                completed_count=len(known),
                positive_count=positive,
                observed_zero_count=len(known) - positive,
                missing_count=len(roster) - len(known),
                identified_bounds=[
                    positive / len(roster),
                    (positive + len(roster) - len(known)) / len(roster),
                ],
            )
        )
    for arm in ARMS:
        members = [c for c in cells if c["arm"] == arm]
        n = sum(c["intended_count"] for c in members)
        k = sum(c["completed_count"] for c in members)
        positive = sum(c["positive_count"] for c in members)
        arms.append(
            dict(
                arm=arm,
                intended_count=n,
                completed_count=k,
                missing_count=n - k,
                identified_bounds=[positive / n, (positive + n - k) / n] if n else [0.0, 1.0],
                complete_case_mean=positive / k if k else None,
            )
        )
    for left, right in combinations(arms, 2):
        shared = [
            g
            for g in games
            if all(
                next(c for c in cells if c["game"] == g and c["arm"] == a)["observed_firings"] > 0
                for a in (left["arm"], right["arm"])
            )
        ]
        supported = len(shared) >= 3 and all(
            c["observed_firings"] >= 5
            for c in cells
            if c["game"] in shared and c["arm"] in (left["arm"], right["arm"])
        )
        effect = [
            left["identified_bounds"][0] - right["identified_bounds"][1],
            left["identified_bounds"][1] - right["identified_bounds"][0],
        ]
        overlaps.append(
            dict(
                left_arm=left["arm"],
                right_arm=right["arm"],
                shared_games=shared,
                admission_passed=supported,
                full_roster_effect_bounds=effect,
                identifiable_benefit=effect[0] > 0,
                generalization_claim=False,
                reason="exposed_observational_diagnostic_no_causal_or_transfer_claim",
            )
        )
    return dict(arm_rows=arms, per_game_per_arm_rows=cells, overlap_rows=overlaps)


def assert_widens(before: Json, after: Json) -> None:
    """Deleting observations cannot create a tighter identified effect interval."""
    for old, new in zip(before["overlap_rows"], after["overlap_rows"], strict=True):
        lo, hi = old["full_roster_effect_bounds"], new["full_roster_effect_bounds"]
        if hi[0] > lo[0] + 1e-12 or hi[1] < lo[1] - 1e-12:
            raise ValueError("manufactured_support")


def check_intervals(rows: list[Json], result: Json) -> bool:
    """Sum per-episode extreme assignments independently of aggregate counts.

    Endpoints express possibilities, never additional successful observations.
    This reconstruction uses the frozen reducer's outcome rows, so disagreement
    in either the status join or the interval arithmetic rejects the analysis.
    """
    endpoints = {}
    for arm in ARMS:
        vector = [
            [r["binary_outcome"], r["binary_outcome"]]
            if r["arm"] == arm and not r["masked"] and r["binary_outcome"] is not None
            else [0, 1]
            for r in rows
        ]
        endpoints[arm] = (
            [sum(v[i] for v in vector) / len(rows) for i in (0, 1)] if rows else [0.0, 1.0]
        )
    for cell in result["overlap_rows"]:
        a, b = endpoints[cell["left_arm"]], endpoints[cell["right_arm"]]
        if cell["full_roster_effect_bounds"] != [a[0] - b[1], a[1] - b[0]]:
            raise ValueError("independent_interval_disagreement")
    return True


def perturb(episodes: list[Json], game: str | None, horizon: int, arm: str | None) -> list[Json]:
    """Mask private observer copies while retaining episode identity and action history.

    A positive reached within available follow-up remains observed. A later
    success and an incomplete negative become unknown; no success is invented.
    Reset provenance remains separate from this artificial observation cutoff.
    """
    # Only observer dictionaries change; action and reset evidence stays read-only.
    copied = [dict(ep, observer_rows=[dict(row) for row in ep["observer_rows"]]) for ep in episodes]
    for ep in copied:
        if ep["unit"]["game"] == game:
            ep["observer_rows"] = []
        kept = []
        for row in ep["observer_rows"]:
            if row["arm"] == arm:
                continue
            if row["observed_horizon"] > horizon:
                row.update(
                    observed_horizon=horizon,
                    sequence=row["sequence"][:horizon],
                    horizon_status="censored",
                    level_progress=0,
                    end_reason="cutoff",
                )
            kept.append(row)
        ep["observer_rows"] = kept
    return copied


def reconstruct(episodes: list[Json], original: list[Json]) -> list[Json]:
    """Read primitive status/count fields directly, without using the frozen reducer."""
    rows = []
    for ep, old in zip(episodes, original, strict=True):
        events = ep["observer_rows"]
        event = events[0] if events else None
        status = event["horizon_status"] if event else "missing"
        binary = (
            1
            if status == "level_progress_observed"
            else 0
            if status == "no_progress_within_horizon"
            else None
        )
        rows.append(
            dict(
                old,
                binary_outcome=binary,
                masked=event is None,
                observation_count=event["observed_horizon"] if event else None,
                outcome="positive"
                if binary == 1
                else "no_progress"
                if binary == 0
                else "censored"
                if event
                else "missing",
                complete_outcome=binary is not None,
                status="completed" if binary is not None else "censored",
            )
        )
    return rows


def grid(episodes: list[Json], progress: Any) -> Json:
    """Cross all registered masks and compare independent and frozen reductions."""
    original = frozen_reduce(episodes)["censoring_rows"]
    baseline = reconstruct(episodes, original)
    reference = bounds(baseline)
    cases = list(
        product([None, *sorted({r["game"] for r in original})], [120, 30, 60, 90], [None, *ARMS])
    )
    sensitivity, overlap, independent = [], [], []
    for index, (game, horizon, arm) in enumerate(cases):
        progress("sensitivity_case", index, len(cases) - index)
        private = perturb(episodes, game, horizon, arm)
        reduced = frozen_reduce(private)["censoring_rows"]
        direct = reconstruct(private, original)
        for actual, expected in zip(reduced, direct, strict=True):
            if any(
                actual.get(k) != expected[k]
                for k in ("episode_id", "game", "outcome", "observation_count")
            ):
                raise ValueError("independent_reduction_disagreement")
        identified = bounds(direct)
        from_frozen = [
            dict(
                old,
                binary_outcome=1
                if r["outcome"] == "positive"
                else 0
                if r["outcome"] == "no_progress"
                else None,
                masked=r["outcome"] == "missing",
            )
            for old, r in zip(original, reduced, strict=True)
        ]
        check_intervals(from_frozen, identified)
        assert_widens(reference, identified)
        case = dict(
            case_id=index,
            removed_game=game,
            completion_horizon=horizon,
            missing_arm=arm,
            intended_count=len(original),
        )
        sensitivity.append(dict(case, rows=direct, **identified))
        overlap.extend(dict(case, **r) for r in identified["overlap_rows"])
        independent.append(
            dict(
                case,
                passed=True,
                frozen_count=len(reduced),
                independent_count=len(direct),
                intervals_reconstructed_from_frozen_rows=True,
                observed_zero_count=sum(r["binary_outcome"] == 0 for r in direct),
                unknown_count=sum(r["binary_outcome"] is None for r in direct),
            )
        )
    return dict(
        rows=baseline,
        censoring_sensitivity_rows=sensitivity,
        overlap_rows=overlap,
        independent_reduction_rows=independent,
        baseline_bounds=reference,
    )
