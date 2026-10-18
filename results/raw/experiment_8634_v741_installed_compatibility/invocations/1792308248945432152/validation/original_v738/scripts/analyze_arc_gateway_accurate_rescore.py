#!/usr/bin/env python3
"""GATEWAY-ACCURATE re-score of our ARC corpora, with the OPENING RESET charged CORRECTLY.

WHY THIS EXISTS, in one paragraph a stranger can follow.
------------------------------------------------------
Our offline harness charged a RESET zero actions. The live ARC gateway charges it one
(`arc_agi/scorecard.py:701-704` `inc_reset_count` -> `resets[i] += 1` AND `actions[i] += 1`).
Because the competition score is a SQUARED efficiency ratio over a DIFFERENCE of cumulative
charged counts, every per-level efficiency figure this project holds was optimistic. That much was
established by `results/outer_loop_arc_gateway_rescore_20260726.json` (commit fd23b8b1f), which
measured the optimism at "4.62% median / 16.80% max" over 44 live cells.

That measurement charged EVERY reset, including the run's FIRST one. The installed source says the
first one is FREE. `update_scorecard` (`arc_agi/scorecard.py:834-843`) routes a RESET frame to
`new_play()` when `full_reset` is true and to `reset()` otherwise; only `reset()` reaches
`inc_reset_count`. `new_play` -> `inc_play_count` (:692-699) APPENDS a fresh zeroed row and charges
nothing. And `full_reset` is true exactly when the environment is being CREATED
(`arc_agi/api.py:405-437`: a cached guid returns `(game, False)`; a cache miss reaches
`arcade.make` and returns `(game, True)`). The reference agent opens with `self.guid = ""` and a
RESET, so the run's opening RESET is always the env-creating one. One free reset per play.

So the correction needs its own correction, and it moves in the direction that makes our numbers
BETTER, not worse. This analyser re-scores three ways and reports all three side by side:

  M0  offline_recorded      resets charged 0            -- what our rows recorded
  M1  all_resets_charged    every reset charged 1       -- the prior artifact's model
  M2  bootstrap_free        every reset EXCEPT the       -- what the installed gateway does
                            opening full_reset charged 1

M2 is the gateway-accurate model. M1 is retained, not deleted, because it is the published claim
being corrected and a reader must be able to see both numbers next to each other.

A STRUCTURAL FACT WORTH STATING PLAINLY: M1 is not merely pessimistic, it is UNREACHABLE through
the real chain. `update_scorecard` cannot charge the opening RESET, because that RESET is the call
that CREATES the play; driving the real `Scorecard` with no `full_reset` at all raises `KeyError`
(there is no card yet). Simulating M1 requires injecting a phantom `new_play()` the gateway never
performs. That is computed here as a witness, not asserted.

THREE UNITS, NEVER CONFLATED (they differ enough to flip a conclusion, and have):
  offline actions   our harness `actions`; EXCLUDES resets. The unit every recorded `efficiency` is in.
  frames            loop iterations; INCLUDES resets.
  gateway-charged   non-RESET moves PLUS charged resets. The ONLY unit the score is a function of.
                    Under M1 that is `actions + resets`; under M2 it is `actions + resets - 1`.

WHAT THIS ANALYSER IS NOT. It is an aggregation over already-persisted rows plus a re-drive of the
INSTALLED scorer. It runs no ARC agent, claims no solve, and submits nothing. Its `duration_s` is
its own runtime; the clock of the measurements being re-scored is `measurement_wall_s`, summed from
each upstream row FILE's own `elapsed_s`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import accumulate
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arc_gateway_rescore import (  # noqa: E402
    _baselines_from_row,
    gateway_score_via_calculator,
    greedy_worst_case,
    load_rows,
    worst_case_allocation,
)

# --------------------------------------------------------------------------- inputs

PRIOR_ARTIFACT = "results/outer_loop_arc_gateway_rescore_20260726.json"
PERLEVEL_ARTIFACT = "results/arc_per_level_reset_attribution_20260726.json"
PEER_ARTIFACT = "results/outer_loop_arc_reset_charge_attribution_20260726.json"

# The MEASURED gateway charge, read off the arcade's own scorecard Card (2026-07-27). Added because
# every "gateway-charged" number in this artifact is a MODEL of the charge, and the model was found
# wrong on 17 of 44 cells. G7 below joins on (game, seed, budget) and asserts agreement with THIS
# file, which is the only authority a fidelity gate can appeal to.
CARD_GROUND_TRUTH_ROWS = "results/card_ground_truth_rows_20260727/rows_card_ground_truth.json"
CARD_GROUND_TRUTH_ARTIFACT = "results/arc_gateway_card_ground_truth_20260727.json"

EXACT_ROW_FILES = [
    "results/early_stop_sweep_20260726/rows_exact_attribution.json",
    "results/early_stop_sweep_20260726/rows_exact_attribution_b2000.json",
]
RECORDED_ROW_FILES = [
    "results/early_stop_sweep_20260726/rows_b2000.json",
    "results/early_stop_sweep_20260726/rows_b400.json",
    "results/early_stop_sweep_20260726/rows_b4000_stress.json",
    "results/early_stop_sweep_20260726/rows_b400_g350.json",
    "results/early_stop_sweep_20260726/rows_contention_conc_1.json",
    "results/early_stop_sweep_20260726/rows_contention_conc_2.json",
    "results/early_stop_sweep_20260726/rows_contention_conc_3.json",
    "results/early_stop_sweep_20260726/rows_contention_serial.json",
    "results/early_stop_sweep_20260726/rows_reproduction.json",
]

SOURCE_SPANS = {
    "gateway_charges_a_reset": (
        "arc_agi/scorecard.py:701-704 `inc_reset_count` -> `self.resets[i] += 1` AND "
        "`self.actions[i] += 1`"
    ),
    "but_only_when_full_reset_is_FALSE": (
        "arc_agi/scorecard.py:834-843 `update_scorecard` -- a RESET frame routes to `new_play()` "
        "when `full_reset` else to `reset()`; only `reset()` reaches `inc_reset_count`"
    ),
    "new_play_charges_nothing": (
        "arc_agi/scorecard.py:692-699 `inc_play_count` APPENDS actions=0/resets=0 for a new play; "
        "it never increments an existing counter"
    ),
    # SPAN CORRECTED 2026-07-27. The original citation here was
    # `arc_agi/api.py:405-437 _get_or_create_environment`, which is the wrong code path for the
    # LOCAL chain these measurements run on. On that chain the scorecard sees `resp.full_reset`
    # (`arc_agi/wrapper.py:194`), which is set by `arcengine/base_game.py:305-316 handle_reset`:
    # `full_reset` holds whenever `_action_count == 0` OR `state == WIN`. So the free-reset condition
    # is a GAME-STATE predicate, not "the first reset of the run", and it can fire more than once
    # (env construction, a RESET with no intervening action, a RESET immediately after a win) --
    # each appending a new zeroed play row and charging nothing.
    "full_reset_predicate_is_action_count_0_or_state_WIN": (
        "arcengine/base_game.py:305-316 `handle_reset` sets `_full_reset` when `_action_count == 0` "
        "or `state == WIN`; `arc_agi/wrapper.py:187-195` gates the scorecard update on a non-empty "
        "frame and passes `resp.full_reset` through to `update_scorecard`"
    ),
    "the_api_path_is_a_DIFFERENT_mechanism_not_this_one": (
        "arc_agi/api.py:405-437 `_get_or_create_environment` returns True only on a fresh "
        "`arcade.make`; that is the REMOTE/HTTP guid-cache path and is NOT what sets `full_reset` on "
        "the local chain. Citing it for the local measurement was an error, corrected 2026-07-27."
    ),
    "one_free_reset_per_play_is_a_TRAJECTORY_PROPERTY_not_a_law": (
        "the reference agent harness opens with `self.guid = ''` (ARC-AGI-3-Agents/agents/"
        "agent.py:55) and a RESET, so a run's first RESET is free. But that is a property of THESE "
        "trajectories, not a guarantee: a trajectory that RESETs twice in a row, or RESETs after a "
        "win, gets more than one free reset. `arc_leaderboard_eval.run_game` now OBSERVES the count "
        "(`observed_full_resets`, `consecutive_reset_pairs`) instead of assuming 1, and "
        "`arc_gateway_card_ground_truth.py` removes the need for the assumption entirely by reading "
        "the Card. Measured on this corpus: 0 consecutive-RESET pairs, exactly 1 full reset per run."
    ),
    "per_level_charge_is_a_DIFFERENCE": (
        "arc_agi/scorecard.py:479 `level_actions = actions_at_level - prev_actions`, so a charge "
        "added before a level-up inflates that level's denominator only"
    ),
    "per_level_score_is_squared_and_capped": (
        "arc_agi/scorecard.py:166-173 `min((baseline/actions_taken)**2 * 100, 115)`"
    ),
    "incomplete_level_scores_zero_so_the_tail_is_free": "arc_agi/scorecard.py:178-183",
    "game_aggregation_has_a_second_clamp": (
        "arc_agi/scorecard.py:192-206 index-weighted mean with a further "
        "`min(., max_weights/total_weights*100)` clamp -- a per-game score can never exceed 100"
    ),
}


# --------------------------------------------------------------------------- helpers


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(rel: str) -> dict | None:
    p = REPO / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _dist(vals: Sequence[float]) -> dict[str, Any]:
    """Distribution summary that reports n first, so an n=0 case cannot read as a clean zero."""
    v = sorted(float(x) for x in vals)
    if not v:
        return {"n": 0, "EMPTY_do_not_read_as_zero": True}
    return {
        "n": len(v),
        "min": round(v[0], 6),
        "p25": round(v[max(0, int(0.25 * (len(v) - 1)))], 6),
        "median": round(statistics.median(v), 6),
        "p75": round(v[min(len(v) - 1, int(0.75 * (len(v) - 1)))], 6),
        "max": round(v[-1], 6),
        "mean": round(statistics.fmean(v), 6),
    }


def measurement_wall_clock(rels: Sequence[str]) -> dict[str, Any]:
    """TRUE wall clock of the measurements being re-scored, from each row FILE's own clock.

    This analyser runs in seconds over rows that took hours. Publishing only `duration_s` beside a
    live-substrate declaration was a real incident on 2026-07-26; this is the corrected basis.
    Prefers the file's own `elapsed_s` / `measurement_wall_s` (the driving process's clock) over the
    sum of per-cell `wall_s`, because the per-cell timer omits setup and env construction and
    undercounts by roughly a quarter. Says per file which basis was used so the two are never
    silently mixed, and stamps an unreadable file rather than crashing mid-sweep.
    """
    per_file: list[dict[str, Any]] = []
    total = 0.0
    fallbacks: list[str] = []
    for rel in sorted(set(rels)):
        d = _read_json(rel)
        if d is None:
            per_file.append({"file": rel, "elapsed_s": None, "basis": "unreadable"})
            continue
        rows = d.get("rows") if isinstance(d.get("rows"), list) else d.get("cells") or []
        elapsed = d.get("elapsed_s")
        basis = "file_elapsed_s"
        if elapsed is None:
            elapsed = d.get("measurement_wall_s")
            basis = "file_measurement_wall_s"
        if elapsed is None:
            elapsed = sum(float(r.get("wall_s") or 0.0) for r in rows)
            basis = "summed_cell_wall_s_fallback_UNDERCOUNT"
            fallbacks.append(rel)
        per_file.append(
            {
                "file": rel,
                "n_rows": len(rows),
                "elapsed_s": round(float(elapsed), 1),
                "basis": basis,
            }
        )
        total += float(elapsed)
    return {
        "total_s": round(total, 1),
        "total_h": round(total / 3600.0, 3),
        "n_rows": sum(int(f.get("n_rows") or 0) for f in per_file),
        "per_file": per_file,
        "files_using_fallback_basis": fallbacks,
        "all_files_report_their_own_clock": not fallbacks,
        "principle": (
            "the clock of the LIVE runs being re-scored, NOT this analyser's runtime. Conflating "
            "them published an 8-second duration for a 2.54-hour measurement once already."
        ),
    }


# ------------------------------------------------------- path 2: the REAL update_scorecard chain


def score_via_update_scorecard(
    game_id: str,
    baselines: Sequence[int],
    seg_offline: Sequence[int],
    seg_resets: Sequence[int],
    tail_offline: int,
    tail_resets: int,
    *,
    charge_the_opening_reset: bool,
) -> dict[str, Any]:
    """Drive the INSTALLED `Scorecard.update_scorecard` with a real frame sequence.

    This is strictly closer to the gateway than driving the `Card` mutators directly, because
    `update_scorecard` is the function that decides `new_play` vs `reset` from the `full_reset`
    flag -- i.e. it is where the "opening reset is free" behaviour actually lives. The prior
    artifact's path 2 bypassed it and called `sc.new_play()` + `sc.reset()` by hand, which is why
    the free opening reset was invisible to it.

    Frame order within a span does not affect the charge (a RESET and a move each cost one), so a
    canonical order is used: the span's resets first, then its moves, with the level increment on
    the span's final frame. Putting resets first also guarantees the run's FIRST frame is a RESET,
    which is what the real agent does.

    `charge_the_opening_reset=False` is the gateway-accurate model M2. `True` is M1, and reaching
    it at all requires a phantom `new_play()` the gateway never performs -- returned as
    `required_phantom_new_play` so the reader can see that M1 is a model, not a behaviour.
    """
    from arc_agi.models import EnvironmentInfo
    from arc_agi.scorecard import EnvironmentScorecard, Scorecard
    from arcengine.enums import ActionInput, FrameDataRaw, GameAction, GameState

    guid = "rescore-guid"
    sc = Scorecard(card_id="rescore")
    phantom = False
    if charge_the_opening_reset:
        # The gateway never does this. Without it there is no card for `inc_reset_count` to find,
        # and the real chain raises. Recorded, not hidden.
        sc.new_play(game_id, guid)
        phantom = True

    # EVERY frame must report the levels_completed the gateway would observe AT THAT FRAME. An
    # earlier draft of this function hardcoded 0 on the non-final frames of each span, which made
    # `levels_completed` DROP from 1 back to 0 at the start of span 2. `Card.set_levels_completed`
    # appends an entry whenever the value CHANGES in EITHER direction, so that inserted a spurious
    # `(0, 16)` into `actions_by_level`, and the scorer consumes that list POSITIONALLY -- shifting
    # every later level's charge by one slot and turning a 2.09 score into 21.31. Caught only
    # because the real chain reports `actions_by_level`; a calculator-only path would have hidden it.
    frames: list[tuple[str, int]] = []
    level = 0
    pending_jump = 0
    for i, moves in enumerate(seg_offline):
        n_r = int(seg_resets[i]) if i < len(seg_resets) else 0
        span_len = n_r + int(moves)
        if span_len == 0:
            # A span with no frames at all would mean two level-ups on one frame. None exist in
            # the measured corpus (asserted upstream); if one ever does, the increment is carried
            # to the next emitted frame so the real chain shows its own single-entry behaviour
            # rather than us modelling it away.
            pending_jump += 1
            continue
        kinds = ["RESET"] * n_r + ["ACTION"] * int(moves)
        for k in kinds[:-1]:
            frames.append((k, level))
        level += 1 + pending_jump
        pending_jump = 0
        frames.append((kinds[-1], level))
    frames.extend([("RESET", level)] * int(tail_resets))
    frames.extend([("ACTION", level)] * int(tail_offline))

    seen_full = False
    for kind, lvl in frames:
        if kind == "RESET":
            full = (not seen_full) and (not charge_the_opening_reset)
            seen_full = True
            ai = ActionInput(id=GameAction.RESET)
        else:
            full = False
            ai = ActionInput(id=GameAction.ACTION1)
        data = FrameDataRaw(
            game_id=game_id,
            state=GameState.NOT_FINISHED,
            levels_completed=int(lvl),
            action_input=ai,
            guid=guid,
        )
        sc.update_scorecard(guid, data, full)

    card = sc.cards.get(game_id)
    if card is None:
        return {"score": None, "error": "no_card_created", "required_phantom_new_play": phantom}
    info = EnvironmentInfo(game_id=game_id, baseline_actions=[int(b) for b in baselines])
    env_sc = EnvironmentScorecard.from_scorecard(sc, [info])
    score = None
    for env in env_sc.environments:
        for run in env.runs:
            score = float(run.score)
    return {
        "score": score,
        "card_actions": list(card.actions),
        "card_resets": list(card.resets),
        "actions_by_level": [list(map(list, x)) for x in card.actions_by_level],
        "n_frames_driven": len(frames),
        "required_phantom_new_play": phantom,
    }


def witness_opening_reset_is_free() -> dict[str, Any]:
    """COMPUTED witness for the whole correction, driven through the installed chain.

    Three facts, each a number rather than a citation:
      1. with the opening RESET flagged `full_reset=True`, the card's charged `actions` equals the
         offline move count and its `resets` counter equals `n_resets - 1`;
      2. with it charged, both are one higher and the score is strictly lower;
      3. charging it AT ALL requires a phantom `new_play`, and omitting that phantom makes the real
         chain raise -- so M1 is not a behaviour the gateway can exhibit.
    """
    baselines = [7, 18, 44, 61, 131, 34, 152]
    seg_offline, seg_resets = [15, 42], [1, 0]
    tail_offline, tail_resets = 330, 12

    free = score_via_update_scorecard(
        "wit00",
        baselines,
        seg_offline,
        seg_resets,
        tail_offline,
        tail_resets,
        charge_the_opening_reset=False,
    )
    charged = score_via_update_scorecard(
        "wit01",
        baselines,
        seg_offline,
        seg_resets,
        tail_offline,
        tail_resets,
        charge_the_opening_reset=True,
    )

    unreachable_error = None
    try:
        from arc_agi.scorecard import Scorecard
        from arcengine.enums import ActionInput, FrameDataRaw, GameAction, GameState

        sc = Scorecard(card_id="wit")
        data = FrameDataRaw(
            game_id="wit02",
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            action_input=ActionInput(id=GameAction.RESET),
            guid="g",
        )
        sc.update_scorecard("g", data, False)  # no full_reset ever -> no card exists
        sc.cards["wit02"].actions  # noqa: B018 - the access is the probe
        unreachable_error = "NONE_the_chain_accepted_it"
    except Exception as exc:  # noqa: BLE001 - the exception IS the witness
        unreachable_error = f"{type(exc).__name__}: {str(exc)[:80]}"

    total_offline = sum(seg_offline) + tail_offline
    total_resets = sum(seg_resets) + tail_resets
    return {
        "fixture": {
            "baselines": baselines,
            "seg_offline": seg_offline,
            "seg_resets": seg_resets,
            "tail_offline": tail_offline,
            "tail_resets": tail_resets,
            "note": "the vc33/seed-20260724/b400 shape, chosen because its ONLY pre-level-up reset IS the opening one",
        },
        "M2_bootstrap_free": {
            "score": free["score"],
            "card_actions": free["card_actions"],
            "card_resets": free["card_resets"],
            "actions_by_level": free["actions_by_level"],
        },
        "M1_all_resets_charged": {
            "score": charged["score"],
            "card_actions": charged["card_actions"],
            "card_resets": charged["card_resets"],
            "actions_by_level": charged["actions_by_level"],
            "required_phantom_new_play": charged["required_phantom_new_play"],
        },
        "charged_actions_M2_equals_offline_plus_resets_minus_one": (
            free["card_actions"] == [total_offline + total_resets - 1]
        ),
        "charged_actions_M1_equals_offline_plus_resets": (
            charged["card_actions"] == [total_offline + total_resets]
        ),
        "resets_counter_M2_is_one_lower": (
            free["card_resets"] == [total_resets - 1] and charged["card_resets"] == [total_resets]
        ),
        "M2_score_is_strictly_higher_than_M1": bool(
            free["score"] is not None
            and charged["score"] is not None
            and free["score"] > charged["score"]
        ),
        "M1_is_unreachable_without_a_phantom_new_play": {
            "what_was_tried": "update_scorecard(RESET, full_reset=False) as the run's FIRST call",
            "result": unreachable_error,
            "verdict": (
                "M1 CANNOT be produced by the installed chain. The opening RESET is the call that "
                "CREATES the play, so there is no counter for it to increment. Charging it is a "
                "modelling choice, not gateway behaviour."
            ),
        },
    }


# --------------------------------------------------------------------------- exact cells


def _exact_cell(cell: dict, baselines_by_game: dict[str, list[int]]) -> dict[str, Any] | None:
    """Re-score ONE exact-attribution cell under M0 / M1 / M2, two independent scorer paths.

    FIELD-CONVENTION HAZARD, named because it is a live misreading trap in the row schema: within
    the same cell, `level_up_actions_offline` is PER-SPAN, while `resets_before_levelups` and
    `level_up_charged` are CUMULATIVE. Mixing them silently produces a plausible wrong number. The
    cumulative identity `level_up_charged[i] == cumsum(offline)[i] + resets_before[i]` is asserted
    per cell here so a convention drift upstream fails loudly.
    """
    levels = int(cell.get("levels") or 0)
    if levels <= 0:
        return None
    game = str(cell.get("game") or "")
    seg_offline = [int(x) for x in (cell.get("level_up_actions_offline") or [])]
    cum_resets = [int(x) for x in (cell.get("resets_before_levelups") or [])]
    cum_charged = [int(x) for x in (cell.get("level_up_charged") or [])]
    if not (len(seg_offline) == len(cum_resets) == len(cum_charged) == levels):
        return {"game": game, "usable": False, "reason": "attribution_length_mismatch"}

    baselines = [int(p.get("human_actions") or 0) for p in (cell.get("per_level") or [])]
    if not baselines or not all(baselines):
        baselines = baselines_by_game.get(game) or []
    if not baselines or not all(baselines):
        # The dead-channel guard. A zero baseline makes every charge model agree at score 0 and
        # reads as a clean "no optimism" null.
        return {"game": game, "usable": False, "reason": "baseline_channel_dead_or_zero"}

    cum_offline: list[int] = []
    run = 0
    for x in seg_offline:
        run += x
        cum_offline.append(run)
    identity_ok = all(cum_charged[i] == cum_offline[i] + cum_resets[i] for i in range(levels))

    seg_resets = [cum_resets[0]] + [cum_resets[i] - cum_resets[i - 1] for i in range(1, levels)]
    opening_reset_in_first_span = seg_resets[0] >= 1

    offline_actions = int(cell.get("offline_actions") or cell.get("actions") or 0)
    n_resets = int(cell.get("n_resets") or 0)
    tail_offline = offline_actions - cum_offline[-1]
    tail_resets = n_resets - cum_resets[-1]

    # Per-span CHARGED costs under each model. M2 differs from M1 only in the FIRST span, because
    # every later span is a difference of two cumulative counts and the -1 cancels.
    seg_charged_m1 = [seg_offline[i] + seg_resets[i] for i in range(levels)]
    seg_charged_m2 = list(seg_charged_m1)
    if opening_reset_in_first_span:
        seg_charged_m2[0] -= 1
    tail_charged = tail_offline + tail_resets
    game_won = levels >= len(baselines)

    s0, _ = gateway_score_via_calculator(baselines, seg_offline, tail_offline, game_won=game_won)
    s1, _ = gateway_score_via_calculator(baselines, seg_charged_m1, tail_charged, game_won=game_won)
    s2, _ = gateway_score_via_calculator(baselines, seg_charged_m2, tail_charged, game_won=game_won)

    chain2 = score_via_update_scorecard(
        f"x2{game}",
        baselines,
        seg_offline,
        seg_resets,
        tail_offline,
        tail_resets,
        charge_the_opening_reset=False,
    )
    chain1 = score_via_update_scorecard(
        f"x1{game}",
        baselines,
        seg_offline,
        seg_resets,
        tail_offline,
        tail_resets,
        charge_the_opening_reset=True,
    )

    def _rel(base: float, corrected: float) -> float | None:
        return None if base <= 0 else round((base - corrected) / base, 6)

    return {
        "game": game,
        "seed": cell.get("seed"),
        "budget": cell.get("budget"),
        "usable": True,
        "levels": levels,
        "offline_actions": offline_actions,
        "n_resets": n_resets,
        "resets_before_first_span": seg_resets[0],
        "resets_in_completed_spans": int(cum_resets[-1]),
        "resets_in_free_tail": int(tail_resets),
        "cumulative_identity_holds": identity_ok,
        "opening_reset_in_first_span": opening_reset_in_first_span,
        "seg_offline": seg_offline,
        "seg_charged_M1": seg_charged_m1,
        "seg_charged_M2": seg_charged_m2,
        "score_M0_offline_recorded": round(s0, 6),
        "score_M1_all_resets_charged": round(s1, 6),
        "score_M2_bootstrap_free_MODELLED": round(s2, 6),
        "recorded_efficiency_in_row": cell.get("efficiency_offline"),
        "recorded_gateway_charged_in_row_M1": cell.get("efficiency_gateway_charged"),
        "rel_optimism_M1": _rel(s0, s1),
        "rel_optimism_M2_MODELLED": _rel(s0, s2),
        "chain_M1_score": chain1["score"],
        "chain_M2_score": chain2["score"],
        "chain_agrees_M1": (chain1["score"] is not None and abs(chain1["score"] - s1) < 1e-9),
        "chain_agrees_M2": (chain2["score"] is not None and abs(chain2["score"] - s2) < 1e-9),
        "baselines_nonzero": True,
    }


def _perlevel_cell(cell: dict, baselines_by_game: dict[str, list[int]]) -> dict[str, Any] | None:
    """Re-score ONE cell from the per-span attribution capture (a SECOND, independent channel).

    That capture records per-SPAN offline / resets / gateway-charged directly (it never publishes a
    cumulative), which is a different code path from the exact-attribution rows' cumulative
    `level_up_charged`. Agreement between the two on overlapping cells is therefore worth something.
    """
    levels = int(cell.get("levels") or 0)
    if levels <= 0:
        return None
    game = str(cell.get("game") or "")
    seg_offline = [int(x) for x in (cell.get("segment_offline_actions") or [])]
    seg_resets = [int(x) for x in (cell.get("segment_resets") or [])]
    if len(seg_offline) != levels or len(seg_resets) != levels:
        return {"game": game, "usable": False, "reason": "segment_length_mismatch"}
    baselines = baselines_by_game.get(game) or []
    if not baselines or not all(baselines):
        return {"game": game, "usable": False, "reason": "baseline_channel_dead_or_zero"}

    tail_offline = int(cell.get("tail_offline_actions") or 0)
    tail_resets = int(cell.get("tail_resets") or 0)
    game_won = levels >= len(baselines)
    seg_charged_m1 = [seg_offline[i] + seg_resets[i] for i in range(levels)]
    seg_charged_m2 = list(seg_charged_m1)
    opening = seg_resets[0] >= 1
    if opening:
        seg_charged_m2[0] -= 1
    tail_charged = tail_offline + tail_resets

    s0, _ = gateway_score_via_calculator(baselines, seg_offline, tail_offline, game_won=game_won)
    s1, _ = gateway_score_via_calculator(baselines, seg_charged_m1, tail_charged, game_won=game_won)
    s2, _ = gateway_score_via_calculator(baselines, seg_charged_m2, tail_charged, game_won=game_won)

    def _rel(base: float, corrected: float) -> float | None:
        return None if base <= 0 else round((base - corrected) / base, 6)

    return {
        "game": game,
        "seed": cell.get("seed"),
        "budget": cell.get("budget"),
        "usable": True,
        "levels": levels,
        "opening_reset_in_first_span": opening,
        "resets_that_cost_score_M1": int(sum(seg_resets)),
        "resets_that_cost_score_M2": int(max(0, sum(seg_resets) - (1 if opening else 0))),
        "resets_in_free_tail": tail_resets,
        "score_M0_offline_recorded": round(s0, 6),
        "score_M1_all_resets_charged": round(s1, 6),
        "score_M2_bootstrap_free_MODELLED": round(s2, 6),
        "rel_optimism_M1": _rel(s0, s1),
        "rel_optimism_M2_MODELLED": _rel(s0, s2),
        "recorded_rel_loss_in_that_artifact_M1": None,
    }


# --------------------------------------------------------------------------- Part A bounds


def _bound_row(row: dict) -> dict[str, Any] | None:
    """Two-sided BOUND for one recorded row, under BOTH charge models.

    A recorded row carries a WHOLE-RUN `n_resets` only, so the correction is bounded, not
    determined: the worst case puts every chargeable reset before a level-up, the best case puts
    them all in the free tail. Under M2 one reset -- the opening one -- is removed from the
    chargeable pool, so M2's bound is strictly inside M1's.

    The best case is 0 BY CONSTRUCTION. That is stated everywhere this bound is reported, because a
    bound whose lower end is structurally zero can never establish that the correction is small.
    """
    if row.get("n_resets") is None:
        return None
    lua = [int(x) for x in (row.get("level_up_actions") or [])]
    if not lua:
        return None
    baselines = _baselines_from_row(row)
    if not baselines or not all(baselines):
        return None
    seg_offline, prev = [], 0
    for at in lua:
        seg_offline.append(at - prev)
        prev = at
    offline_actions = int(row.get("actions") or 0)
    tail = offline_actions - prev
    n_resets = int(row.get("n_resets") or 0)
    game_won = len(seg_offline) >= len(baselines)

    s0, _ = gateway_score_via_calculator(baselines, seg_offline, tail, game_won=game_won)
    if s0 <= 0:
        return None
    # `tail` is passed UNMODIFIED: in the worst case every chargeable reset is allocated to a
    # COMPLETED level, so none of them land in the tail. (The tail charge is score-neutral anyway --
    # an incomplete level scores 0 regardless -- but passing a tail that also counted those resets
    # would be double-counting in a field a reader can see.)
    worst1, alloc1 = worst_case_allocation(
        baselines, seg_offline, n_resets, tail, game_won=game_won
    )
    worst2, alloc2 = worst_case_allocation(
        baselines, seg_offline, max(0, n_resets - 1), tail, game_won=game_won
    )
    greedy1, _ = greedy_worst_case(baselines, seg_offline, n_resets, tail, game_won=game_won)
    return {
        "game": row.get("game"),
        "seed": row.get("seed"),
        "budget": row.get("budget"),
        "arm": row.get("arm"),
        "n_resets": n_resets,
        "score_M0_offline_recorded": round(s0, 6),
        "worst_case_M1": round(worst1, 6),
        "worst_case_M2": round(worst2, 6),
        "rel_worst_M1": round((s0 - worst1) / s0, 6),
        "rel_worst_M2": round((s0 - worst2) / s0, 6),
        "best_case_is_zero_by_construction": True,
        "dp_alloc_M1": alloc1,
        "dp_alloc_M2": alloc2,
        "greedy_matches_dp": abs(greedy1 - worst1) < 1e-9,
    }


# --------------------------------------------------------------------------- build


def build(out_path: Path | None = None) -> dict[str, Any]:  # noqa: C901 - one artifact, one builder
    t0 = time.time()
    art: dict[str, Any] = {
        "experiment": "outer_loop_arc_gateway_accurate_rescore_20260726",
        "title": (
            "GATEWAY-ACCURATE re-score: the opening RESET is FREE, so the published "
            "uncharged-RESET optimism was itself overstated"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "carnot.arc_gateway_accurate_rescore.v1",
    }

    # ---------------------------------------------------------------- charge model
    art["charge_model_settled_from_installed_source"] = {
        "source_spans": SOURCE_SPANS,
        "three_models_scored_side_by_side": {
            "M0_offline_recorded": "resets charged 0 -- what our rows recorded",
            "M1_all_resets_charged": (
                "every reset charged 1 -- the model behind the published 4.62% figure"
            ),
            "M2_bootstrap_free_MODELLED": (
                "every reset EXCEPT the opening full_reset charged 1 -- what the installed "
                "gateway does"
            ),
        },
        "charged_total_formula": {
            "M1": "actions + resets",
            "M2": "actions + resets - n_full_resets, and n_full_resets == 1 per play",
        },
        "three_units_never_conflated": {
            "offline_actions": "our harness `actions`; EXCLUDES resets",
            "frames": "loop iterations; INCLUDES resets",
            "gateway_charged": "non-RESET moves PLUS charged resets -- the only unit the score sees",
        },
        "why_only_the_first_span_moves": (
            "the scorer's per-level cost is a DIFFERENCE of cumulative charged counts, so removing "
            "one charge from every cumulative total cancels in every span except the first"
        ),
        "assumption_stated_not_verified": (
            "the `full_reset` semantics were read from the INSTALLED LOCAL `arc_agi`/`arcengine`. "
            "The hidden competition gateway is a remote service whose implementation was not "
            "inspected. If the remote scorer differs, M2 is wrong in the same direction M1 is -- "
            "this is an assumption, not a measurement."
        ),
        "residual_divergence_not_quantified": (
            "under `competition_mode`, a RESET issued at `_action_count == 0` is CHARGED but the "
            "game is NOT stepped (arc_agi/api.py:316-334), so a back-to-back RESET diverges "
            "behaviourally between the gateway and our offline env, which really does reset. Only "
            "reachable when the agent resets with no moves since its last reset. Named, not measured."
        ),
    }

    # ---------------------------------------------------------------- witness
    art["witness_opening_reset_is_free_driven_through_the_installed_chain"] = (
        witness_opening_reset_is_free()
    )

    # ---------------------------------------------------------------- inputs / provenance of claims
    prior = _read_json(PRIOR_ARTIFACT)
    perlevel = _read_json(PERLEVEL_ARTIFACT)
    peer = _read_json(PEER_ARTIFACT)

    cited: list[dict[str, Any]] = []
    for rel, what in (
        (
            PRIOR_ARTIFACT,
            "published the uncharged-RESET optimism as median 4.62% / max 16.80% over 44 live "
            "cells, computed under M1",
        ),
        (
            PERLEVEL_ARTIFACT,
            "published relative score loss from charged resets as median 4.5% / max 12.0% over 7 "
            "won cells, computed under M1",
        ),
        (
            PEER_ARTIFACT,
            "independent bound over the same rows (median 11.4% / max 95.7% worst case) plus the "
            "first statement that the opening reset is free",
        ),
    ):
        p = REPO / rel
        tracked = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", rel],
                cwd=REPO,
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )
        cited.append(
            {
                "path": rel,
                "exists": p.exists(),
                "sha256": _sha(p) if p.exists() else None,
                "bytes": p.stat().st_size if p.exists() else None,
                "committed_at_time_of_writing": tracked,
                "what_it_reported": what,
                "NOT_MODIFIED": True,
            }
        )
    art["cited_originals_never_rewritten"] = {
        "artifacts": cited,
        "principle": (
            "CLAUDE.md never-prune: a re-score emits a NEW artifact that states the correction and "
            "cites the originals by path + sha256. No historical artifact's recorded numbers were "
            "edited, and the numbers here are published BESIDE the M1 numbers, not on top of them."
        ),
    }

    # ---------------------------------------------------------------- baselines map (+ dead-channel guard)
    baselines_by_game: dict[str, list[int]] = {}
    baseline_conflicts: list[dict[str, Any]] = []
    all_row_files = EXACT_ROW_FILES + RECORDED_ROW_FILES
    for rel in all_row_files:
        for r in load_rows(str(REPO / rel)):
            g = str(r.get("game") or "")
            pl = r.get("per_level") or []
            if not g or not pl:
                continue
            b = [int(x.get("human_actions") or 0) for x in pl]
            if not all(b):
                continue
            if g not in baselines_by_game:
                baselines_by_game[g] = b
            elif baselines_by_game[g] != b:
                baseline_conflicts.append({"game": g, "a": baselines_by_game[g], "b": b})

    # ---------------------------------------------------------------- Part B: exact cells
    exact_cells_raw: list[dict] = []
    for rel in EXACT_ROW_FILES:
        exact_cells_raw.extend(load_rows(str(REPO / rel)))
    part_b = [_exact_cell(c, baselines_by_game) for c in exact_cells_raw]
    part_b = [c for c in part_b if c is not None]
    usable_b = [c for c in part_b if c.get("usable")]

    relm1 = [c["rel_optimism_M1"] for c in usable_b if c["rel_optimism_M1"] is not None]
    relm2 = [
        c["rel_optimism_M2_MODELLED"] for c in usable_b if c["rel_optimism_M2_MODELLED"] is not None
    ]
    per_game_m2: dict[str, float] = {}
    per_game_m1: dict[str, float] = {}
    for g in sorted({c["game"] for c in usable_b}):
        vs2 = [
            c["rel_optimism_M2_MODELLED"]
            for c in usable_b
            if c["game"] == g and c["rel_optimism_M2_MODELLED"] is not None
        ]
        vs1 = [
            c["rel_optimism_M1"]
            for c in usable_b
            if c["game"] == g and c["rel_optimism_M1"] is not None
        ]
        if vs2:
            per_game_m2[g] = round(statistics.median(vs2), 6)
        if vs1:
            per_game_m1[g] = round(statistics.median(vs1), 6)

    n_zero_m2 = sum(1 for v in relm2 if v <= 1e-12)
    n_zero_m1 = sum(1 for v in relm1 if v <= 1e-12)
    m2_le_m1 = all(
        c["rel_optimism_M2_MODELLED"] <= c["rel_optimism_M1"] + 1e-12
        for c in usable_b
        if c["rel_optimism_M1"] is not None and c["rel_optimism_M2_MODELLED"] is not None
    )
    n_strictly_lower = sum(
        1
        for c in usable_b
        if c["rel_optimism_M1"] is not None
        and c["rel_optimism_M2_MODELLED"] is not None
        and c["rel_optimism_M2_MODELLED"] < c["rel_optimism_M1"] - 1e-12
    )

    # agreement of my recomputed M1 with the numbers the ROWS already carry (a check on this
    # analyser, not on the gateway)
    reprod: list[dict[str, Any]] = []
    for c in usable_b:
        rec = c.get("recorded_gateway_charged_in_row_M1")
        if rec is None:
            continue
        reprod.append(
            {
                "cell": f"{c['game']}/{c['seed']}/b{c['budget']}",
                "recorded": rec,
                "recomputed_M1": round(c["score_M1_all_resets_charged"], 4),
                "agrees": abs(float(rec) - c["score_M1_all_resets_charged"]) < 5e-4,
            }
        )
    reprod_offline: list[dict[str, Any]] = []
    for c in usable_b:
        rec = c.get("recorded_efficiency_in_row")
        if rec is None:
            continue
        reprod_offline.append(
            {
                "cell": f"{c['game']}/{c['seed']}/b{c['budget']}",
                "recorded": rec,
                "recomputed_M0": round(c["score_M0_offline_recorded"], 4),
                "agrees": abs(float(rec) - c["score_M0_offline_recorded"]) < 5e-4,
            }
        )

    # Which cells become EXACTLY zero under M2, and why -- a zero is the most misreadable value in
    # this artifact, so each one is classified rather than counted.
    zeros_m2: list[dict[str, Any]] = []
    for c in usable_b:
        v = c["rel_optimism_M2_MODELLED"]
        if v is None or v > 1e-12:
            continue
        chargeable_after_bootstrap = c["resets_in_completed_spans"] - 1
        already_zero = (c["rel_optimism_M1"] or 0.0) <= 1e-12
        if already_zero:
            why = (
                "ALREADY zero under M1 -- M2 adds nothing here. The prior artifact identified these "
                "as cap-absorbed: a superhuman level sitting at the 115 cap swallows the extra "
                "charge with no score change. (This cell ALSO has only the opening reset before its "
                "level-up, so the zero is now doubly determined.)"
            )
        elif chargeable_after_bootstrap <= 0:
            why = (
                "NEWLY zero under M2: the opening RESET was the ONLY reset before any level-up, so "
                "with it free there is nothing left to charge -- a TRUE zero under the gateway model"
            )
        else:
            why = (
                "resets remained chargeable yet the score did not move, so the affected level must "
                "sit at the 115 cap -- a STRUCTURAL zero, not a consequence of the free opening reset"
            )
        zeros_m2.append(
            {
                "cell": f"{c['game']}/{c['seed']}/b{c['budget']}",
                "resets_in_completed_spans": c["resets_in_completed_spans"],
                "chargeable_after_the_free_opening_reset": chargeable_after_bootstrap,
                "rel_optimism_M1": c["rel_optimism_M1"],
                "why_zero_under_M2": why,
            }
        )

    prior_b = (prior or {}).get("part_b_exact_attribution_live", {})
    prior_m1_median = (prior_b.get("rel_optimism_fraction_of_offline_score") or {}).get("median")

    art["part_b_exact_reset_ATTRIBUTION_44_cells"] = {
        "independent_reproduction_of_the_prior_artifacts_M1_number": {
            "prior_artifact": PRIOR_ARTIFACT,
            "prior_unrounded_M1_median": prior_m1_median,
            "recomputed_here_M1_median": (round(statistics.median(relm1), 6) if relm1 else None),
            "agrees": (
                prior_m1_median is not None
                and relm1
                and abs(float(prior_m1_median) - statistics.median(relm1)) < 1e-6
            ),
            "why_this_matters": (
                "two independently written analysers, driving the installed scorer through "
                "different code, land on the same M1 figure. That is what licenses reading the "
                "M2-vs-M1 difference as a CHARGE-MODEL change rather than as one of the two "
                "pipelines being wrong."
            ),
        },
        "cells_that_become_exactly_zero_under_M2": {
            "n": len(zeros_m2),
            "n_under_M1": n_zero_m1,
            "detail": zeros_m2,
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_note": (
            "this part re-scores rows that a LIVE offline-arcade agent produced; the live substrate "
            "belongs to those row files, whose own clock is reported below. Nothing here steps an env."
        ),
        "measurement_wall_s": measurement_wall_clock(EXACT_ROW_FILES),
        "n_cells_read": len(exact_cells_raw),
        "n_cells_with_a_levelup": len(part_b),
        "n_cells_usable": len(usable_b),
        "n_cells_unusable": [c for c in part_b if not c.get("usable")],
        "cumulative_identity_holds_on_every_cell": all(
            c["cumulative_identity_holds"] for c in usable_b
        ),
        "opening_reset_in_first_span_on_every_cell": all(
            c["opening_reset_in_first_span"] for c in usable_b
        ),
        "rel_optimism_M1_the_PUBLISHED_model": _dist(relm1),
        "rel_optimism_M2_MODELLED": _dist(relm2),
        "abs_optimism_score_points_M1": _dist(
            [c["score_M0_offline_recorded"] - c["score_M1_all_resets_charged"] for c in usable_b]
        ),
        "abs_optimism_score_points_M2_MODELLED": _dist(
            [
                c["score_M0_offline_recorded"] - c["score_M2_bootstrap_free_MODELLED"]
                for c in usable_b
            ]
        ),
        "abs_vs_rel_reading_note": (
            "the ABSOLUTE correction is tiny mostly because these games score near zero to begin "
            "with; the RELATIVE figure is the one that says how wrong a per-level efficiency claim "
            "was. Neither substitutes for the other, and the score-heavy game in this corpus (vc33, "
            "~2.09 offline against <0.02 for most others) is precisely the game whose gateway "
            "optimism nearly vanishes under M2 -- so an absolute-score-weighted reading of the "
            "correction is smaller still than the per-cell median suggests."
        ),
        "n_cells_with_zero_optimism_M1": n_zero_m1,
        "n_cells_with_zero_optimism_M2": n_zero_m2,
        "M2_never_exceeds_M1": m2_le_m1,
        "n_cells_where_M2_is_strictly_lower": n_strictly_lower,
        "per_game_median_M1": per_game_m1,
        "per_game_median_M2_MODELLED": per_game_m2,
        "per_game_median_of_medians_M2": (
            round(statistics.median(list(per_game_m2.values())), 6) if per_game_m2 else None
        ),
        "recomputation_reproduces_the_rows_own_M1_numbers": {
            "n_compared": len(reprod),
            "n_agree": sum(1 for r in reprod if r["agrees"]),
            "disagreements": [r for r in reprod if not r["agrees"]],
        },
        "recomputation_reproduces_the_rows_own_OFFLINE_numbers": {
            "n_compared": len(reprod_offline),
            "n_agree": sum(1 for r in reprod_offline if r["agrees"]),
            "disagreements": [r for r in reprod_offline if not r["agrees"]],
        },
        "cells": usable_b,
    }

    # ---------------------------------------------------------------- Part C: per-span capture
    part_c: list[dict[str, Any]] = []
    if perlevel:
        for c in perlevel.get("cells") or []:
            got = _perlevel_cell(c, baselines_by_game)
            if got is not None:
                part_c.append(got)
    usable_c = [c for c in part_c if c.get("usable")]
    relc2 = [
        c["rel_optimism_M2_MODELLED"] for c in usable_c if c["rel_optimism_M2_MODELLED"] is not None
    ]
    relc1 = [c["rel_optimism_M1"] for c in usable_c if c["rel_optimism_M1"] is not None]

    # The per-span artifact computed its published `rel_loss` from the row's 4-DECIMAL-ROUNDED
    # efficiency fields. On a cell whose score is ~0.007 that rounding is a large fraction of the
    # quantity being differenced, so its published numbers are not a clean M1 baseline. Both are
    # reported per cell so the reader can see which part of the movement is the charge model and
    # which part is rounding.
    # HISTORICAL BASELINE, READ FROM GIT (2026-07-27). This block's entire purpose is to compare the
    # per-level lane's PUBLISHED (4-dp-rounded) rel_loss against an unrounded recomputation. The
    # per-level lane has since been FIXED to divide unrounded scores, so reading its CURRENT
    # working-tree values would compare unrounded against unrounded and report a rounding error of
    # exactly 0.0 -- silently erasing the hazard this block documents. The committed version at git
    # HEAD is the historical record, so that is what is read.
    def _perlevel_published_rel() -> tuple[dict, str]:
        raw = os.popen(f"git show HEAD:{PERLEVEL_ARTIFACT} 2>/dev/null").read()
        if raw.strip():
            try:
                d = json.loads(raw)
                return (
                    {
                        str(x.get("cell") or ""): x.get("rel_loss")
                        for x in (
                            (d.get("score_loss_from_charged_resets") or {}).get("per_cell") or []
                        )
                    },
                    f"git show HEAD:{PERLEVEL_ARTIFACT} (the PUBLISHED, pre-fix values)",
                )
            except Exception:
                pass
        return (
            {
                str(x.get("cell") or ""): x.get("rel_loss")
                for x in (
                    (perlevel or {}).get("score_loss_from_charged_resets", {}).get("per_cell") or []
                )
            },
            f"working tree {PERLEVEL_ARTIFACT} (git HEAD unavailable -- may already be FIXED, in "
            "which case the reported rounding error is 0 by construction, not by measurement)",
        )

    published_rel, published_rel_source = _perlevel_published_rel()

    # The whole PUBLISHED per-level artifact at git HEAD. Same reason as above: three more fields
    # below quote "as published" figures from that lane, and the working-tree copy has since been
    # FIXED to divide unrounded scores -- so reading it would relabel the corrected numbers as the
    # rounded ones they replaced.
    def _perlevel_at_head() -> tuple[dict, str]:
        raw = os.popen(f"git show HEAD:{PERLEVEL_ARTIFACT} 2>/dev/null").read()
        if raw.strip():
            try:
                return json.loads(raw), f"git show HEAD:{PERLEVEL_ARTIFACT}"
            except Exception:
                pass
        return (perlevel or {}), f"working tree {PERLEVEL_ARTIFACT} (git HEAD unavailable)"

    perlevel_published, perlevel_published_source = _perlevel_at_head()
    part_c_precision: list[dict[str, Any]] = []
    for c in usable_c:
        key = f"{c['game']}@{c['seed']}"
        pub = published_rel.get(key)
        if pub is None:
            continue
        part_c_precision.append(
            {
                "cell": key,
                "published_rel_loss_from_4dp_rounded_fields_M1": pub,
                "recomputed_unrounded_M1": c["rel_optimism_M1"],
                "rounding_error_in_the_published_M1": (
                    None if pub is None else round(float(c["rel_optimism_M1"]) - float(pub), 6)
                ),
                "MODELLED_M2": c["rel_optimism_M2_MODELLED"],
            }
        )

    art["part_c_exact_per_span_capture_second_channel"] = {
        "precision_hazard_in_the_published_M1_baseline": {
            "per_cell": part_c_precision,
            "published_baseline_source": published_rel_source,
            "STATUS_2026_07_27": (
                "CLOSED UPSTREAM. `arc_per_level_reset_attribution_capture.py` now divides the "
                "UNROUNDED scorer outputs (`efficiency_*_precise`, added to run_game as pure "
                "addition), so its per-cell median moved 0.045078 -> 0.046236449 and tu93's cell "
                "moved 0.041667 (= exactly 1/24) -> 0.046236449. The per-cell values compared here "
                "are read from git HEAD so this block keeps documenting the historical hazard "
                "instead of comparing the fixed values against themselves."
            ),
            "what_happened": (
                "`run_game` rounds both efficiency fields to 4 decimals before they reach a row. "
                "tu93's published loss of 0.041667 is literally 0.0003/0.0072 -- a ratio of two "
                "rounded numbers. Its unrounded M1 loss is 0.046236, so roughly a tenth of that "
                "published figure was rounding, not charge model."
            ),
            "how_it_is_handled_here": (
                "every statistic in this artifact is recomputed UNROUNDED from each cell's own "
                "per-span charge vectors. The rounded row fields are retained for reference and "
                "never differenced."
            ),
            "note_this_is_not_a_new_finding": (
                "the prior gateway-rescore artifact independently found and corrected the same "
                "hazard for its own Part B (2 cells whose optimism the rounding erased to exactly "
                "0.0). It is re-stated here because the per-span capture did NOT apply that "
                "correction, so its published headline carries the hazard."
            ),
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "source": PERLEVEL_ARTIFACT,
        "available": bool(perlevel),
        "measurement_wall_s_of_that_capture": (
            (perlevel or {}).get("duration_s") if perlevel else None
        ),
        "measurement_wall_s_basis_note": (
            "that capture is a LIVE run whose analyser clock IS its measurement clock (it declares "
            "`measurement_clock_is_the_analyser_clock_because_this_is_a_live_capture: true`), so its "
            "`duration_s` is the right number to carry forward here."
        ),
        "n_cells_with_a_levelup": len(part_c),
        "n_cells_usable": len(usable_c),
        "rel_optimism_M1": _dist(relc1),
        "rel_optimism_M2_MODELLED": _dist(relc2),
        "published_in_that_artifact_M1_median_source": perlevel_published_source,
        "published_in_that_artifact_M1_median": (
            (perlevel_published or {}).get("score_loss_from_charged_resets", {}).get("median")
        ),
        "published_in_that_artifact_M1_max": (
            (perlevel_published or {}).get("score_loss_from_charged_resets", {}).get("max")
        ),
        "cells": usable_c,
        "why_this_is_a_second_channel_not_a_duplicate": (
            "that capture records per-SPAN offline / resets directly and never publishes a "
            "cumulative; Part B derives spans from a CUMULATIVE `level_up_charged`. Two different "
            "code paths reaching the same per-span costs is worth more than one path run twice."
        ),
    }

    # ---------------------------------------------------------------- Part D: B vs C agreement
    keyed_b = {(c["game"], c["seed"], c["budget"]): c for c in usable_b}
    overlap: list[dict[str, Any]] = []
    for c in usable_c:
        k = (c["game"], c["seed"], c["budget"])
        b = keyed_b.get(k)
        if b is None:
            continue
        overlap.append(
            {
                "cell": f"{k[0]}/{k[1]}/b{k[2]}",
                "M2_from_cumulative_channel": b["score_M2_bootstrap_free_MODELLED"],
                "M2_from_per_span_channel": c["score_M2_bootstrap_free_MODELLED"],
                "agrees": abs(
                    b["score_M2_bootstrap_free_MODELLED"] - c["score_M2_bootstrap_free_MODELLED"]
                )
                < 1e-6,
            }
        )
    art["part_d_two_channels_agree"] = {
        "n_overlapping_cells": len(overlap),
        "n_agree": sum(1 for o in overlap if o["agrees"]),
        "disagreements": [o for o in overlap if not o["agrees"]],
        "comparison": overlap,
        "UNINTERPRETABLE_IF_EMPTY": (
            "with zero overlapping cells this section proves nothing and must not be read as "
            "agreement"
            if not overlap
            else None
        ),
    }

    # ---------------------------------------------------------------- Part A: bounds
    bounds: list[dict[str, Any]] = []
    n_rows_total = 0
    for rel in RECORDED_ROW_FILES:
        for r in load_rows(str(REPO / rel)):
            n_rows_total += 1
            got = _bound_row(r)
            if got is not None:
                bounds.append(got)
    art["part_a_bounds_over_recorded_rows_NOT_POINT_ESTIMATES"] = {
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "measurement_wall_s": measurement_wall_clock(RECORDED_ROW_FILES),
        "n_rows_total": n_rows_total,
        "n_rows_bounded": len(bounds),
        "n_rows_no_levelup_or_zero_score_either_way": n_rows_total - len(bounds),
        "why_only_a_bound": (
            "a recorded row carries a WHOLE-RUN `n_resets` only. The correction needs resets "
            "attributed PER SPAN, because a reset before a level-up is charged into that level's "
            "squared denominator while a reset in the post-solve tail costs nothing."
        ),
        "rel_worst_case_M1": _dist([b["rel_worst_M1"] for b in bounds]),
        "rel_worst_case_M2_MODELLED": _dist([b["rel_worst_M2"] for b in bounds]),
        "best_case_is_zero_by_construction_on_every_row": all(
            b["best_case_is_zero_by_construction"] for b in bounds
        ),
        "UNINFORMATIVE_STAMP": (
            "THIS BOUND CANNOT ANSWER THE HEADLINE QUESTION IN EITHER DIRECTION. Its lower end is "
            "0 by construction, so it can never establish that the correction is material; its "
            "upper end reaches 95%+, so it can never establish that the correction is small. It is "
            "reported because it is the honest width available from a whole-run reset count, and "
            "because it is exactly what justified instrumenting per span. Do NOT quote either end "
            "as an estimate."
        ),
        "greedy_vs_exact_dp": {
            "n_rows": len(bounds),
            "n_rows_where_greedy_disagrees_with_the_DP": sum(
                1 for b in bounds if not b["greedy_matches_dp"]
            ),
            "greedy_never_beat_the_DP": all(
                b["worst_case_M1"] <= b["score_M0_offline_recorded"] + 1e-12 for b in bounds
            ),
            "how_to_read_a_disagreement": (
                "a greedy-on-marginals allocation is computed independently as a cross-check on the "
                "exact DP. Disagreements are EXPECTED and harmless: greedy is an upper bound on the "
                "worst case (it can fail to find the minimising allocation), so the DP is the number "
                "reported. A greedy result BELOW the DP would indicate a DP bug, and none occur."
            ),
        },
        "independent_reproduction_of_the_prior_artifacts_M1_bound": {
            "prior_artifact": PRIOR_ARTIFACT,
            "prior": (
                (prior or {})
                .get("part_a_bounds_over_recorded_rows", {})
                .get("rel_delta_fraction_of_offline_score_erased")
            ),
            "recomputed_here_M1": _dist([b["rel_worst_M1"] for b in bounds]),
            "note": (
                "reproduced independently, which is why the M2 bound beside it can be read as a "
                "charge-model change rather than a pipeline difference"
            ),
        },
        "M2_bound_is_strictly_inside_M1_bound": all(
            b["rel_worst_M2"] <= b["rel_worst_M1"] + 1e-12 for b in bounds
        ),
        "worst_case_bound_rows_sample": bounds[:12],
    }

    # ---------------------------------------------------------------- scorer fidelity
    chain_ok_m1 = sum(1 for c in usable_b if c["chain_agrees_M1"])
    chain_ok_m2 = sum(1 for c in usable_b if c["chain_agrees_M2"])
    models_differ = sum(
        1
        for c in usable_b
        if abs(c["score_M1_all_resets_charged"] - c["score_M2_bootstrap_free_MODELLED"]) > 1e-9
    )
    art["scorer_fidelity_two_independent_paths"] = {
        "path_1": (
            "arc_agi.scorecard.EnvironmentScoreCalculator, driven exactly as "
            "EnvironmentScorecard._calculate_score drives it"
        ),
        "path_2": (
            "the REAL Scorecard.update_scorecard chain -- synthetic FrameDataRaw per frame with the "
            "real `full_reset` flag, then EnvironmentScorecard.from_scorecard. This is the function "
            "where the free-opening-reset behaviour lives; the prior artifact's path 2 bypassed it "
            "by calling the Card mutators directly, which is why the free reset was invisible to it."
        ),
        "n_cells_compared": len(usable_b),
        "n_agree_M1": chain_ok_m1,
        "n_agree_M2": chain_ok_m2,
        "disagreements": [
            {
                "cell": f"{c['game']}/{c['seed']}/b{c['budget']}",
                "path1_M1": c["score_M1_all_resets_charged"],
                "path2_M1": c["chain_M1_score"],
                "path1_M2": c["score_M2_bootstrap_free_MODELLED"],
                "path2_M2": c["chain_M2_score"],
            }
            for c in usable_b
            if not (c["chain_agrees_M1"] and c["chain_agrees_M2"])
        ],
        "non_vacuity_witness_the_two_models_actually_differ": {
            "n_cells_where_M1_and_M2_scores_differ": models_differ,
            "why_this_matters": (
                "if M1 and M2 scored identically everywhere, a path1-vs-path2 agreement would be "
                "consistent with the analyser ignoring the charge model entirely. A non-zero count "
                "here shows the comparison has content."
            ),
        },
        "baselines_nonzero_on_every_cell": all(c["baselines_nonzero"] for c in usable_b),
        "baseline_channel_note": (
            "baselines come from each row's `per_level[*].human_actions`, produced upstream by "
            "`arc_leaderboard_eval._baseline_actions` reading through `env.info`. A prior agent read "
            "`getattr(env, 'baseline_actions')` directly -- a DEAD CHANNEL that summed to 0.0 and "
            "made every charge model agree at score 0, reading as a clean 'no optimism' null. "
            "Asserted non-zero per cell, and cross-source consistency asserted below."
        ),
        "baseline_cross_source_conflicts": baseline_conflicts,
        "n_games_with_baselines": len(baselines_by_game),
    }

    # ---------------------------------------------------------------- peer cross-check
    art["peer_bound_crosscheck"] = {
        "peer_artifact": PEER_ARTIFACT,
        "available": bool(peer),
        "peer_sha256": _sha(REPO / PEER_ARTIFACT) if (REPO / PEER_ARTIFACT).exists() else None,
        "peer_is_uncommitted_at_time_of_writing": True,
        "peer_first_stated_the_opening_reset_is_free": True,
        "what_agrees": (
            "the peer's independent reading of the same installed source reached the same "
            "conclusion -- gateway-charged == actions + resets - n_full_resets, n_full_resets >= 1 "
            "-- and its three hand-recomputed cells (vc33 4.51%->0.00%, r11l 16.66%->16.46%, tu93 "
            "4.62%->4.06%) are reproduced independently here over 44 cells."
        ),
        "how_to_read_the_dependency": (
            "the peer artifact is NOT a provenance dependency of this one: it is a corroborating "
            "read, present-or-absent flagged, so this artifact does not go stale if the peer's "
            "commit lands or does not."
        ),
    }

    # ---------------------------------------------------------------- correction owed (PROMINENT)
    m1_med = art["part_b_exact_reset_ATTRIBUTION_44_cells"][
        "rel_optimism_M1_the_PUBLISHED_model"
    ].get("median")
    m1_max = art["part_b_exact_reset_ATTRIBUTION_44_cells"][
        "rel_optimism_M1_the_PUBLISHED_model"
    ].get("max")
    m2_med = art["part_b_exact_reset_ATTRIBUTION_44_cells"]["rel_optimism_M2_MODELLED"].get(
        "median"
    )
    m2_max = art["part_b_exact_reset_ATTRIBUTION_44_cells"]["rel_optimism_M2_MODELLED"].get("max")
    art["CORRECTION_OWED"] = {
        "read_this_first": (
            "A number this session reported to the operator MOVES. It moves in the direction that "
            "makes our estimates better, and it is named here rather than buried."
        ),
        "claim_1": {
            "where_it_was_stated": (
                "commit fd23b8b1f subject line -- 'the uncharged-RESET optimism is 4.6% "
                "(measured)' -- and `headline.answer_from_exact_measurement` in "
                f"{PRIOR_ARTIFACT}"
            ),
            "as_published_M1": {
                "median": m1_med,
                "max": m1_max,
                "n_cells_nonzero": len(relm1) - n_zero_m1,
                "n_cells": len(relm1),
            },
            "MODELLED_M2": {
                "median": m2_med,
                "max": m2_max,
                "n_cells_nonzero": len(relm2) - n_zero_m2,
                "n_cells": len(relm2),
            },
            "why_it_moved": (
                "M1 charged the run's opening RESET. The installed gateway routes that RESET to "
                "`new_play()`, which charges nothing. One charge was removed from the first span of "
                "every cell, and the first span is the only span a uniform -1 can move."
            ),
            "direction": "the published figure OVERSTATED the optimism",
        },
        "claim_2": {
            "where_it_was_stated": (
                f"`score_loss_from_charged_resets` in {PERLEVEL_ARTIFACT} (median 0.045078, max "
                "0.119524) and commit a1d8360eb"
            ),
            "as_published_M1_from_4dp_rounded_fields": {
                "median": (perlevel_published or {})
                .get("score_loss_from_charged_resets", {})
                .get("median"),
                "max": (perlevel_published or {})
                .get("score_loss_from_charged_resets", {})
                .get("max"),
                "source": perlevel_published_source,
                "note": (
                    "read from git HEAD on purpose. The working-tree per-level artifact has since "
                    "been FIXED to divide UNROUNDED scores, so reading it here would label the "
                    "corrected figures as the rounded ones they replaced."
                ),
            },
            "the_same_cells_unrounded_M1": {
                "median": _dist(relc1).get("median"),
                "max": _dist(relc1).get("max"),
            },
            "MODELLED_M2": {
                "median": _dist(relc2).get("median"),
                "max": _dist(relc2).get("max"),
            },
            "TWO_separate_errors_in_that_published_figure": (
                "(1) it charged the opening reset, and (2) it differenced two 4-decimal-ROUNDED "
                "scores, which on a ~0.007 score is a large fraction of the difference. Both are "
                "shown separately above so neither is credited with the other's movement."
            ),
            "direction": "the published figure OVERSTATED the optimism",
        },
        "claim_3_a_CODE_correction_not_just_a_number": {
            "what_is_wrong": (
                "`scripts/arc_leaderboard_eval.py` computes `level_up_charged.append(actions + "
                "resets)` and `charged_actions = actions + resets`, charging the opening reset. So "
                "EVERY row this harness writes from now on carries an `efficiency_gateway_charged` "
                "that is too low and an `efficiency_optimism_vs_gateway` that is too high, by one "
                "charged action in the first span."
            ),
            "the_minimal_fix_AS_ORIGINALLY_SPECIFIED_and_why_it_was_WRONG": (
                "the original spec here was `charged_actions = actions + resets - n_full_resets` "
                "with `n_full_resets` HARD-CODED to 1. That is wrong twice over. (a) The free-reset "
                "predicate is `_action_count == 0 or state == WIN` "
                "(`arcengine/base_game.py:305-316`), so a trajectory that RESETs twice in a row, or "
                "RESETs after a win, earns more than one free reset and the formula would "
                "under-charge it silently. (b) More importantly the whole approach is a MODEL, and "
                "the model is wrong for an unrelated reason: a non-RESET action taken while the game "
                "is GAME_OVER/WIN returns `frame=[]` (`arcengine/base_game.py:204-216`) and the "
                "scorecard update is gated on `len(resp.frame) > 0` (`arc_agi/wrapper.py:187`), so "
                "POST-DEATH actions are FREE at the gateway while this harness counts them."
            ),
            "the_fix_ACTUALLY_APPLIED_2026_07_27": (
                "stop modelling the charge and READ it. `arc_leaderboard_eval._read_gateway_card` "
                "reads `card.actions[idx]`, `card.resets[idx]` and `card.actions_by_level[idx]` off "
                "the arcade's own scorecard Card -- the exact object the leaderboard scorer consumes "
                "-- and every row now carries them plus `efficiency_gateway_card`, "
                "`empty_frame_actions`, `observed_full_resets` and `consecutive_reset_pairs`. The "
                "modelled fields are RETAINED unchanged so no historical comparison shifts unit, and "
                "`gateway_card_vs_model_charged_delta` records the disagreement per row. Measured "
                "consequence: the M2 numbers in THIS artifact overstate the real optimism on 17 of "
                "44 cells and have the WRONG SIGN on 6 -- see "
                "results/arc_gateway_card_ground_truth_20260727.json."
            ),
            "why_it_was_deferred_at_the_time": (
                "that file is a freshness-tracked provenance dependency of at least five committed "
                "artifacts plus one uncommitted peer artifact, so editing it marks all of them stale "
                "in the same commit. That was the right call for a measure-and-report brief; the "
                "edit has since been made as a PURE ADDITION (no pre-existing field's value moves), "
                "with the dependent artifacts rebuilt in the same change."
            ),
            "severity": (
                "HIGHER than the number correction. A wrong number in one artifact is citable and "
                "correctable; a wrong measurement channel silently mis-stamps every future row."
            ),
        },
        "what_does_NOT_change": (
            "no settled structural conclusion is reordered. DEPTH still dominates (1/2/4/8 of 8 "
            "levels -> 2.78/8.33/27.78/100), a per-game score still cannot exceed 100, the "
            "post-solve tail is still free, and the correction still tracks resets-BEFORE-a-level-up "
            "rather than the whole-run reset count. The optimism is still real and still non-zero on "
            "most cells -- it is smaller than published."
        ),
    }

    # ---------------------------------------------------------------- superseded-by + residual
    art["THE_M2_NUMBERS_HERE_ARE_A_MODEL_SUPERSEDED_BY"] = {
        "artifact": CARD_GROUND_TRUTH_ARTIFACT,
        "what_it_measures": (
            "the same 48 cells re-run with the gateway's own scorecard Card READ rather than the "
            "charge modelled as offline_actions + resets - 1"
        ),
        "why_the_model_here_is_wrong": (
            "a non-RESET action taken while the game is GAME_OVER/WIN returns frame=[] "
            "(arcengine/base_game.py:204-216) and the scorecard update is gated on "
            "len(resp.frame) > 0 (arc_agi/wrapper.py:187), so post-death actions are FREE at the "
            "gateway while this harness counts them. The model has no term for that."
        ),
        "measured_consequence": (
            "the M2 figures below reproduce the real per-level charged vector on only 27 of 44 "
            "cells; on 6 (tu93, 3 seeds x 2 budgets) the TRUE sign is NEGATIVE -- the recorded "
            "offline number was PESSIMISTIC, not optimistic. The model never understates."
        ),
        "which_number_to_cite": (
            "cite the REAL median from the superseding artifact. The M2 median here is retained "
            "verbatim per never-prune and must be labelled MODELLED wherever it is quoted."
        ),
        "nothing_here_was_rewritten": True,
    }
    art["COMPETITION_MODE_RESIDUAL"] = {
        "claim": (
            "every figure in this artifact is gateway-accurate-at-best for the OFFLINE/LOCAL charge "
            "path, and a LOWER BOUND under competition_mode"
        ),
        "source": "arc_agi/api.py:316-334",
        "mechanism": (
            "when a RESET arrives with full_reset=False on a LocalEnvironmentWrapper and "
            "scorecard.competition_mode and g._game._action_count == 0, the code calls "
            "update_scorecard WITHOUT stepping the game -- which routes to reset() -> "
            "inc_reset_count -> resets += 1 AND actions += 1 (scorecard.py:701-704). A "
            "competition-mode RESET at action-count 0 is therefore BILLED while doing nothing."
        ),
        "status": "UNMEASURED -- no run here sets competition_mode",
        "the_free_opening_reset_itself_is_confirmed": (
            "_get_or_create_environment returns True only on a fresh arcade.make; a cached guid "
            "returns False, and inc_play_count increments no counter"
        ),
        "also_unconfirmed": (
            "that the REMOTE hidden gateway matches the INSTALLED local arc_agi package. Read from "
            "source, never measured against the live service. No figure here may be carried into a "
            "scored-path claim without that confirmation."
        ),
    }

    # ---------------------------------------------------------------- headline
    art["headline"] = {
        "question": (
            "What is our ARC score under the gateway's own charge model, and by how much were the "
            "published corrections wrong?"
        ),
        "answer": (
            f"[MODELLED CHARGE -- superseded, see THE_M2_NUMBERS_HERE_ARE_A_MODEL_SUPERSEDED_BY] "
            f"The modelled optimism over {len(relm2)} matched live cells is "
            f"{m2_med} at the median and {m2_max} at most, against the published "
            f"{m1_med} / {m1_max}. The published figure was too PESSIMISTIC because it charged the "
            "run's opening RESET, which the installed gateway gives away free. The correction is "
            f"still non-zero on {len(relm2) - n_zero_m2} of {len(relm2)} cells, so the defect is "
            "real; it is simply smaller than reported."
        ),
        "from_bounds_alone": (
            "STILL UNANSWERABLE, and stamped as such. Over the recorded rows the worst case erases "
            f"a median {art['part_a_bounds_over_recorded_rows_NOT_POINT_ESTIMATES']['rel_worst_case_M2_MODELLED'].get('median')} "
            "of a cell's score and the best case is 0 BY CONSTRUCTION. A bound with a structural "
            "zero at one end cannot establish either that the correction matters or that it does not."
        ),
        "the_structural_finding": (
            "M1 -- charging every reset -- is not a conservative reading of the gateway, it is an "
            "UNREACHABLE one. Producing it through the installed chain requires injecting a phantom "
            "`new_play()` the gateway never performs; without it the chain raises because the "
            "opening RESET is the call that creates the play."
        ),
        "what_this_does_not_touch": (
            "the actual hidden-set score (0.08) is unexplained by anything here. This correction "
            "moves a per-game efficiency term by single-digit percent on public games; it is not a "
            "candidate explanation for the hidden-set result, and is not offered as one."
        ),
    }

    # ---------------------------------------------------------------- gates with witnesses
    def gate(name: str, passed: bool, witness: dict[str, Any], interpretable: bool = True) -> dict:
        out = {"name": name, "passed": bool(passed), "witness": witness}
        if not interpretable:
            out["UNINTERPRETABLE"] = (
                "the pass region is empty or forced; do not read this as a pass"
            )
        return out

    gates = [
        gate(
            "G1_two_scorer_paths_agree_under_both_charge_models",
            chain_ok_m1 == len(usable_b) and chain_ok_m2 == len(usable_b) and len(usable_b) > 0,
            {
                "n_cells": len(usable_b),
                "n_agree_M1": chain_ok_m1,
                "n_agree_M2": chain_ok_m2,
                "non_vacuity_n_cells_where_the_models_differ": models_differ,
                "pass_region_nonempty_because": (
                    "the two models produce DIFFERENT scores on "
                    f"{models_differ} of {len(usable_b)} cells, so agreement between paths is not "
                    "obtainable by ignoring the model"
                ),
            },
            interpretable=len(usable_b) > 0 and models_differ > 0,
        ),
        gate(
            "G2_recomputation_reproduces_the_rows_own_numbers",
            len(reprod) > 0
            and all(r["agrees"] for r in reprod)
            and len(reprod_offline) > 0
            and all(r["agrees"] for r in reprod_offline),
            {
                "n_M1_compared": len(reprod),
                "n_M1_agree": sum(1 for r in reprod if r["agrees"]),
                "n_M0_compared": len(reprod_offline),
                "n_M0_agree": sum(1 for r in reprod_offline if r["agrees"]),
                "why_this_is_a_real_check": (
                    "these are numbers the LIVE harness wrote independently of this analyser; "
                    "reproducing both the offline and the M1 figure means the pipeline is wired to "
                    "the same quantities, so the M2 figure is a model change and not a plumbing bug"
                ),
            },
            interpretable=len(reprod) > 0 and len(reprod_offline) > 0,
        ),
        gate(
            "G3_baseline_channel_is_alive_and_consistent",
            all(c["baselines_nonzero"] for c in usable_b)
            and not baseline_conflicts
            and len(baselines_by_game) > 0,
            {
                "n_games_with_baselines": len(baselines_by_game),
                "n_cross_source_conflicts": len(baseline_conflicts),
                "min_baseline_seen": min(
                    (min(v) for v in baselines_by_game.values() if v), default=None
                ),
                "why": (
                    "a zero baseline makes every charge model agree at score 0, so a dead baseline "
                    "channel reads as a clean 'no optimism' null -- the exact failure a prior agent hit"
                ),
            },
            interpretable=len(baselines_by_game) > 0,
        ),
        gate(
            "G4_the_correction_is_directional_and_non_vacuous",
            m2_le_m1 and n_strictly_lower > 0,
            {
                "M2_never_exceeds_M1": m2_le_m1,
                "n_cells_strictly_lower_under_M2": n_strictly_lower,
                "n_cells": len(usable_b),
                "why_both_conjuncts": (
                    "monotonicity alone would be satisfied by M2 == M1 everywhere (a no-op "
                    "correction); the strict count proves the correction actually bites"
                ),
            },
            interpretable=len(usable_b) > 0,
        ),
        gate(
            "G5_the_opening_reset_is_inside_the_first_span_on_every_cell",
            all(c["opening_reset_in_first_span"] for c in usable_b) and len(usable_b) > 0,
            {
                "n_cells": len(usable_b),
                "n_with_a_reset_in_the_first_span": sum(
                    1 for c in usable_b if c["opening_reset_in_first_span"]
                ),
                "min_resets_in_first_span": min(
                    (c["resets_before_first_span"] for c in usable_b), default=None
                ),
                "why_this_must_be_checked_not_assumed": (
                    "the -1 is applied to the FIRST span. If a cell's first span carried zero "
                    "resets the opening reset would sit elsewhere and the subtraction would be "
                    "wrong; the code stamps rather than subtracts in that case"
                ),
            },
            interpretable=len(usable_b) > 0,
        ),
        gate(
            "G6_both_attribution_channels_agree_where_they_overlap",
            len(overlap) > 0 and all(o["agrees"] for o in overlap),
            {
                "n_overlapping_cells": len(overlap),
                "n_agree": sum(1 for o in overlap if o["agrees"]),
                "channels": "cumulative level_up_charged vs per-span segment_offline/segment_resets",
            },
            interpretable=len(overlap) > 0,
        ),
    ]
    # ---- G7: AGREEMENT WITH THE LIVE CARD, not agreement between two reconstructions ----------
    # ADDED 2026-07-27. G1 ("both scorer paths agree under both models, 44/44") and part_d ("two
    # channels agree 7/7") compare two implementations of the SAME unverified assumption: both
    # reconstruct the charge as `offline_actions + resets - k` and neither consults the gateway's own
    # bookkeeping. So the fidelity gate could not detect the error it missed. The witness_opening_
    # reset_is_free section has the same structural blind spot: it SYNTHESISES a FrameDataRaw
    # sequence and calls Card mutators directly, a path that cannot exhibit the empty-frame
    # short-circuit that turned out to be the defect. This gate's pass condition is that the
    # RECORDED charge equals the LIVE Card's `actions` / `actions_by_level`.
    # The Card numbers live in the ground-truth capture's rows file (the rows THIS artifact reads
    # were captured before `_read_gateway_card` existed). Joined on (game, seed, budget). If that
    # file is absent the gate is stamped UNINTERPRETABLE rather than passed vacuously.
    card_by_cell: dict[tuple, dict] = {}
    try:
        _cg = json.loads((REPO / CARD_GROUND_TRUTH_ROWS).read_text())
        for _c in _cg.get("cells") or []:
            card_by_cell[
                (str(_c.get("game")), int(_c.get("seed") or 0), int(_c.get("budget") or 0))
            ] = _c
    except Exception:
        card_by_cell = {}
    card_rows = []
    for c in usable_b:
        src = card_by_cell.get(
            (str(c.get("game")), int(c.get("seed") or 0), int(c.get("budget") or 0))
        )
        if not src:
            continue
        ca, abl = src.get("card_actions"), src.get("card_actions_by_level")
        if ca is None:
            continue
        # The MODELLED charge this artifact published, in the Card's own shape: a CUMULATIVE
        # per-level vector plus a total.
        seg_m2 = [int(x) for x in (c.get("seg_charged_M2") or [])]
        model_vec = list(accumulate(seg_m2)) if seg_m2 else None
        model_total = int(c.get("offline_actions") or 0) + max(int(c.get("n_resets") or 0) - 1, 0)
        card_vec = [int(b) for _lv, b in (abl or [])]
        card_rows.append(
            {
                "cell": f"{c['game']}@{c.get('seed')}@b{c.get('budget')}",
                "card_actions": int(ca),
                "recorded_model_total": model_total,
                "card_actions_by_level": card_vec,
                "recorded_model_by_level": model_vec,
                "totals_agree": (model_total is not None and int(model_total) == int(ca)),
                "vectors_agree": (model_vec is not None and list(model_vec) == card_vec),
            }
        )
    gates.append(
        gate(
            "G7_the_recorded_charge_agrees_with_the_LIVE_CARD_not_with_a_second_model",
            bool(card_rows) and all(r["vectors_agree"] and r["totals_agree"] for r in card_rows),
            {
                "n_cells_with_a_card_read": len(card_rows),
                "n_cells_totals_agree": sum(1 for r in card_rows if r["totals_agree"]),
                "n_cells_vectors_agree": sum(1 for r in card_rows if r["vectors_agree"]),
                "n_cells_DISAGREE": sum(
                    1 for r in card_rows if not (r["vectors_agree"] and r["totals_agree"])
                ),
                "disagreeing_cells": [
                    r for r in card_rows if not (r["vectors_agree"] and r["totals_agree"])
                ][:20],
                "why_this_gate_exists": (
                    "two reconstructions of the same assumption agreeing is not independence. The "
                    "Card is the object the leaderboard scorer consumes, so it is the only "
                    "authority a fidelity gate can appeal to."
                ),
                "if_this_gate_FAILS_the_M2_numbers_in_this_artifact_are_the_model_not_the_charge": True,
                "measured_elsewhere": "results/arc_gateway_card_ground_truth_20260727.json",
                "card_rows_source": CARD_GROUND_TRUTH_ROWS,
                "note_on_the_extra_play_row": (
                    "the arcade's construction reset creates play 0 (zero actions) and the agent's "
                    "opening RESET creates play 1, so the Card row read is the LAST play. The scorer "
                    "takes a max over plays, so a zero-action play 0 is harmless."
                ),
            },
            # UNINTERPRETABLE when the rows predate the Card instrumentation: an empty pass region
            # must never be counted as a pass. The rows this artifact reads were captured BEFORE
            # `_read_gateway_card` existed, so this is the expected state until they are re-captured.
            interpretable=bool(card_rows),
        )
    )
    # G7 is EXPECTED TO FAIL on this artifact, and that failure IS the artifact's headline
    # correction (2026-07-27). The M2 numbers here are a MODEL of the charge; read against the live
    # Card they are wrong on 17 of 44 cells. Marking the failure "expected" is NOT excusing it -- the
    # gate stays FAILED, the count stays honest, and the reason is named so a reader does not conclude
    # the artifact is malformed or fabricated. An UNEXPECTED failure of any other gate is still a real
    # problem.
    EXPECTED_FAILURES = {
        "G7_the_recorded_charge_agrees_with_the_LIVE_CARD_not_with_a_second_model": (
            "EXPECTED. This artifact's M2 charge is a MODEL; the Card disagrees with it on the cells "
            "carrying post-death (empty-frame) actions. The failure is the correction this artifact "
            "now documents, measured in results/arc_gateway_card_ground_truth_20260727.json. It is "
            "NOT a defect in this analyser's arithmetic and NOT a fabrication signal."
        )
    }
    _unexpected = [
        g["name"] for g in gates if not g["passed"] and g["name"] not in EXPECTED_FAILURES
    ]
    art["acceptance_gates"] = {
        "gates": gates,
        "n_gates": len(gates),
        "n_passed": sum(1 for g in gates if g["passed"]),
        "all_passed": all(g["passed"] for g in gates),
        "EXPECTED_FAILURES": EXPECTED_FAILURES,
        "n_failed_and_EXPECTED": sum(
            1 for g in gates if not g["passed"] and g["name"] in EXPECTED_FAILURES
        ),
        "n_failed_and_UNEXPECTED": len(_unexpected),
        "unexpected_failures": _unexpected,
        "all_passed_EXCLUDING_expected_failures": not _unexpected,
        "n_uninterpretable": sum(1 for g in gates if "UNINTERPRETABLE" in g),
        "principle": (
            "every gate carries a COMPUTED witness that its pass region is non-empty at the gate's "
            "own aggregation level. A gate whose region is empty or forced is stamped "
            "UNINTERPRETABLE rather than counted as a pass."
        ),
    }

    # ---------------------------------------------------------------- scope and power
    art["scope_and_power_READ_BESIDE_THE_VERDICT"] = {
        "part_b": {
            "n_cells": len(usable_b),
            "games": sorted({c["game"] for c in usable_b}),
            "seeds": sorted({c["seed"] for c in usable_b if c.get("seed")}),
            "budgets": sorted({c["budget"] for c in usable_b if c.get("budget")}),
            "llm": "OFF on every cell",
        },
        "part_c": {
            "n_cells": len(usable_c),
            "games": sorted({c["game"] for c in usable_c}),
            "seeds": sorted({c["seed"] for c in usable_c if c.get("seed")}),
        },
        "part_a": {"n_rows_bounded": len(bounds), "n_rows_total": n_rows_total},
        "single_game_concentration_warning": (
            "the per-game SCORES are dominated by vc33, which is the only game in this corpus with "
            "a large offline score (~2.09 vs <0.02 for most others). The RELATIVE optimism is "
            "reported per game and as a median-of-medians precisely so the corpus answer does not "
            "rest on that one game's magnitude."
        ),
        "what_this_measurement_CANNOT_support": [
            "any claim about the HIDDEN set. Every cell is a PUBLIC game played offline.",
            "any claim about the LLM-on scored path. The LLM is off on every cell.",
            "any claim that the remote gateway matches the installed local scorer. That is an "
            "assumption read from source, not a measurement against the live service.",
            "a per-cell optimism estimate for any cell NOT in Parts B or C -- for those, only the "
            "structurally-uninformative bound exists.",
        ],
        "why_no_p_value_is_reported": (
            "this is an ARITHMETIC re-score of a deterministic scoring function, not a sample from "
            "a distribution. The per-cell correction is exact given the cell's recorded attribution; "
            "there is nothing to test. The only sampling question -- how the corpus median would "
            "move on other games -- is answered by reporting the per-game spread, not by a test."
        ),
    }

    # ---------------------------------------------------------------- standard fields
    _g7_disagree = 0
    for _g in gates:
        if _g["name"].startswith("G7_") and isinstance(_g.get("witness"), dict):
            _g7_disagree = int(_g["witness"].get("n_cells_DISAGREE") or 0)
    art["honest_verdict"] = (
        "complete_gateway_accurate_rescore_opening_reset_is_free_published_optimism_overstated_"
        f"median_{str(m1_med).replace('.', 'p')}_to_{str(m2_med).replace('.', 'p')}_correction_owed_named"
        f"_AND_this_M2_is_itself_a_MODEL_that_disagrees_with_the_live_card_on_{_g7_disagree}_cells"
        "_see_arc_gateway_card_ground_truth_20260727"
    )
    art["honest_verdict_principle"] = (
        "terminal-prefixed so the conductor's reconciler classifies it as terminal; the underscore "
        "form is used deliberately over `complete:` because a ': ' inside a verdict has poisoned "
        "research-complete.yaml before."
    )
    art["verifier_is_oracle"] = True
    art["verifier_is_oracle_principle"] = (
        "the 'verifier' here IS the installed competition scorer -- the executable oracle that "
        "defines the score. Execution-grounded measurement, NOT an oracle-distinct verifier-moat "
        "claim, and it must never be headlined as one."
    )
    art["solve_provenance"] = "development_proxy"
    art["solve_provenance_principle"] = (
        "no ARC level is claimed, no agent was run, no registry entry added. This is an accounting "
        "re-score over already-recorded PUBLIC-game rows."
    )
    art["arc_solve_claim"] = False
    art["random_seed"] = 20260726
    art["random_seeds_used"] = sorted({c["seed"] for c in usable_b if c.get("seed")})
    art["preconditions_checked"] = [
        {"resource": "installed arc_agi scorer importable", "available": True},
        {
            "resource": "installed arcengine FrameDataRaw/ActionInput importable",
            "available": True,
        },
        {"resource": "exact-attribution row files present", "available": bool(exact_cells_raw)},
        {"resource": "recorded early-stop row corpora present", "available": bool(bounds)},
        {"resource": "per-level human baselines non-zero", "available": not baseline_conflicts},
        {"resource": "prior artifact readable for the correction", "available": bool(prior)},
        {"resource": "per-span capture readable (second channel)", "available": bool(perlevel)},
    ]
    art["tests_and_mutation_proofs"] = {
        "test_file": "tests/python/test_arc_gateway_accurate_rescore.py",
        "n_tests": 19,
        "spec": "REQ-ARC-WMTE-5986 + 5 SCENARIOs",
        "mutations_applied_and_proved_caught": [
            "the level-drop defect reintroduced (non-final span frames report level 0) -> 4 tests fail",
            "M2 never subtracts the free opening reset (a no-op correction) -> 2 tests fail",
            "the -1 applied to EVERY span instead of only the first -> 2 tests fail",
            "the opening-reset-in-first-span stamp forced True -> 1 test fails",
            "the cumulative-identity check forced True -> 1 test fails",
            "the dead-baseline guard removed so a zero-baseline cell scores instead of stamping -> 1 test fails",
            "`_dist([])` returns median 0.0 instead of stamping EMPTY -> 1 test fails",
            "`_bound_row` bounds a row that never recorded a reset count -> 1 test fails",
            "the phantom-new_play disclosure suppressed -> 1 test fails",
            "path 2 driven with the wrong charge model -> 1 test fails",
            "the chain-agreement FLAG hardcoded True -> 1 test fails (only after the test was "
            "strengthened; the first two versions of that test SURVIVED this mutation, which is "
            "recorded here because a flag asserted True is not evidence the flag is derived)",
        ],
        "a_mutation_that_initially_ESCAPED": (
            "hardcoding `chain_agrees_M1: True` survived a test asserting the flag was True and that "
            "the two path scores matched -- the scores genuinely did match, so nothing failed. Caught "
            "only by a third test that monkeypatches path 2 to return a wrong score and requires the "
            "flag to notice. This is the forced-gate failure mode arriving inside the test suite "
            "rather than inside the artifact."
        ),
        "a_defect_this_analyser_shipped_and_caught_in_itself": (
            "the path-2 frame builder hardcoded `levels_completed=0` on the non-final frames of each "
            "span. Because `Card.set_levels_completed` appends an entry on any CHANGE and the scorer "
            "consumes `actions_by_level` POSITIONALLY, that inserted a spurious `(0, 16)` entry and "
            "turned a 2.09 score into 21.31. Visible only because the real chain reports "
            "`actions_by_level`; a calculator-only path would have hidden it, and the wrong number "
            "was large enough to look like a finding."
        ),
    }
    art["what_was_NOT_changed"] = [
        "nothing was submitted to ARC or Kaggle -- no code path here can submit",
        "MAX_ACTIONS and every SUBMITTED_* flag untouched",
        "no historical artifact's recorded numbers rewritten; the M1 figures are published beside "
        "the M2 figures, not over them",
        "no pre-existing field of scripts/arc_leaderboard_eval.py's row shape changed value; the "
        "2026-07-27 Card-read instrumentation is PURE ADDITION and the fix history is recorded in "
        "CORRECTION_OWED.claim_3_a_CODE_correction_not_just_a_number",
    ]

    # ---------------------------------------------------------------- provenance
    try:
        import analyze_scored_path_lever_ab as sibling

        code = []
        for rel in (
            "scripts/analyze_arc_gateway_accurate_rescore.py",
            "scripts/arc_gateway_rescore.py",
            "scripts/arc_leaderboard_eval.py",
        ):
            p = REPO / rel
            if p.exists():
                code.append(
                    {
                        "path": rel,
                        "sha256": _sha(p),
                        "bytes": p.stat().st_size,
                        "mtime_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime)
                        ),
                    }
                )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
        art["git_head"] = head
        art["provenance"] = {
            "git_head": head,
            "code": code,
            "rows_sources": {
                "exact_attribution_rows": [
                    {"path": rel, "sha256": _sha(REPO / rel)}
                    for rel in EXACT_ROW_FILES
                    if (REPO / rel).exists()
                ],
                "recorded_rows": [
                    {"path": rel, "sha256": _sha(REPO / rel)}
                    for rel in RECORDED_ROW_FILES
                    if (REPO / rel).exists()
                ],
                "cited_artifacts": [
                    {"path": rel, "sha256": _sha(REPO / rel)}
                    for rel in (PRIOR_ARTIFACT, PERLEVEL_ARTIFACT)
                    if (REPO / rel).exists()
                ],
            },
            "rebuild_command": (
                "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python "
                "scripts/analyze_arc_gateway_accurate_rescore.py"
            ),
            "peer_artifact_deliberately_NOT_a_dependency": PEER_ARTIFACT,
            "submitted_nothing": True,
            "max_actions_untouched": True,
        }
        if out_path is not None:
            sibling.preserve_freshness_acknowledgements(art, out_path)
            sibling.register_analyzed_artifact(out_path, analyzer=Path(__file__).resolve())
    except Exception as exc:  # noqa: BLE001 - a provenance failure is recorded, never swallowed
        art["provenance"] = {"error": f"{type(exc).__name__}:{exc}"}

    art["duration_s"] = round(time.time() - t0, 3)

    # TOP-LEVEL MEASUREMENT CLOCK (2026-07-27). The per-part clocks were correct and correctly based
    # on each row FILE's own `elapsed_s`, but there was no top-level field -- so
    # `scripts/summarize_artifact.py` reported only `duration_s: 10.086` for a 2.6-hour measurement.
    # The parts remain the authority; this is their sum, with the decomposition named.
    # `measurement_wall_s` on each PART is a BLOCK, not a float ({total_s, total_h, n_rows,
    # per_file:[...]}). Assuming a bare float here raised TypeError -- the same field-shape assumption
    # this session fixed twice elsewhere. Read `total_s` out of the block, and tolerate a bare number
    # in case a future part emits one.
    def _clock_total_s(block) -> float:
        if isinstance(block, dict):
            v = block.get("total_s")
            return float(v) if isinstance(v, (int, float)) else 0.0
        return float(block) if isinstance(block, (int, float)) else 0.0

    _part_blocks = {
        "part_a": (art.get("part_a_bounds_over_recorded_rows_NOT_POINT_ESTIMATES") or {}).get(
            "measurement_wall_s"
        ),
        "part_b": (art.get("part_b_exact_reset_ATTRIBUTION_44_cells") or {}).get(
            "measurement_wall_s"
        ),
    }
    _part_clocks = {k: _clock_total_s(v) for k, v in _part_blocks.items()}
    art["measurement_wall_s"] = round(sum(_part_clocks.values()), 1)
    art["measurement_wall_s_basis"] = {
        "per_part": _part_clocks,
        "basis": (
            "sum of each PART's `measurement_wall_s`, each of which is summed from the upstream row "
            "FILES' own `elapsed_s` / `measurement_wall_s` -- NOT from summed per-cell `wall_s`, "
            "which undercounts ~25% because per-cell timing excludes policy construction."
        ),
        "why_a_top_level_field": (
            "summarize_artifact.py reads the top level; without this field an aggregation artifact "
            "reports its analyser's seconds as the only visible cost of a multi-hour measurement."
        ),
    }
    art["duration_s_principle"] = (
        "THIS ANALYSER'S runtime only. It is NOT the measurement clock: each part publishes its own "
        "`measurement_wall_s`, summed from the upstream row FILES' own elapsed clocks. Conflating "
        "the two published an 8-second duration for a 2.54-hour measurement once already."
    )
    art["inference_substrate"] = "aggregation_from_upstream_artifacts"
    art["inference_substrate_principle"] = (
        "this artifact steps no environment and loads no model. It re-reads persisted rows and "
        "re-drives the INSTALLED scorer. Declaring a live substrate here would borrow credibility "
        "the pass does not have."
    )
    payload = json.dumps(
        {k: art[k] for k in art if k not in ("run_date", "duration_s")},
        sort_keys=True,
        default=str,
    ).encode()
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    return art


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(REPO / "results/outer_loop_arc_gateway_accurate_rescore_20260726.json"),
    )
    ap.add_argument("--witness-only", action="store_true", help="print the free-reset witness only")
    a = ap.parse_args(argv)
    if a.witness_only:
        print(json.dumps(witness_opening_reset_is_free(), indent=1))
        return 0
    out = Path(a.out).resolve()
    art = build(out_path=out)
    # Carry hand-authored keys through the rebuild (REQ-OPS-REBUILD-PRESERVE-1).
    import sys as _sys

    if str(Path(__file__).resolve().parent) not in _sys.path:
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from artifact_merge_preserve import merge_preserve_with_file

    art = merge_preserve_with_file(out, art)
    out.write_text(json.dumps(art, indent=1, default=str) + "\n")
    print(json.dumps(art["CORRECTION_OWED"]["claim_1"], indent=1))
    print(json.dumps(art["acceptance_gates"]["gates"], indent=1)[:1500])
    print("verdict:", art["honest_verdict"])
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
