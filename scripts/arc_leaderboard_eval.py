"""The FOCUSED ARC-AGI-3 LOOP — the measurement engine for rapid leaderboard progress.

Scores the CarnotAgent on the public games FROM SCRATCH (force-explore: the unseen-game
proxy, since the real eval is hidden games) using the AUTHORITATIVE LEADERBOARD METRIC: per
game, levels_completed plus `efficiency`, which is the score returned by the INSTALLED
`arc_agi.scorecard.EnvironmentScoreCalculator` when driven exactly as the gateway drives it
(see the long comment at run_game, ~line 285). Concretely: each solved level scores
`min((baseline_actions / agent_actions_on_that_level)**2 * 100, 115)`, unsolved levels score
0, and the per-game score is the INDEX-WEIGHTED mean over the game's FULL level list, clamped
by `max_score` = the index-weighted fraction of levels solved x 100.
NOTE (docstring corrected 2026-07-26): this paragraph previously read
"efficiency = sum over solved levels of min(baseline/agent_actions,1)^2", which is the
paraphrased formula a 2026-06-20 adversarial review had ALREADY retracted in the code below
while this header kept advertising it. That stale line was then read as the definition by a
downstream analyser, which built a "pessimistic total-action charge" model on it and inverted
its own recommendation. Do not reintroduce a paraphrase here; the installed scorer is the
definition.
Writes a leaderboard-style scorecard AND a per-game GAP LOG (which games fail + the
failure signature) so the next iteration targets the worst gap, not a random change.

The loop cadence (one turn each, gated):
  1. RUN this harness  -> levels + efficiency + gap log (no quota, offline).
  2. READ the worst gap (a game stuck at L0, or a solved game with terrible efficiency).
  3. IMPROVE one ingredient (salience tiers / frontier-distance nav / status-masking /
     E3 world-model induction for a deep game) -- a single, attributable change.
  4. RE-RUN; keep the change only if levels or efficiency strictly improved (regression
     gate: the previously-solved games must not regress).
  5. Append the closed/!closed gap to the log (never-prune) and repeat.

Usage: arc_leaderboard_eval.py [--budget N] [--mode explore|replay]
"""

from __future__ import annotations

import json
import hashlib
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _write_json_atomic(out: Path, payload: str) -> None:
    """Write a result file so a crash or a concurrent writer cannot leave it half-written.

    The submission gate runs this script for 8 games at once, so the old bare `write_text`
    (truncate, then write) could interleave two runs into spliced JSON. Writing a temp file in
    the same directory and renaming is atomic on POSIX, so a reader sees the old file or the new
    one, never a mixture.
    """
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.write_text(payload)
    tmp.replace(out)


sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_eval_provenance import (
    completion_counters,
    generator_provenance,
)
from carnot.agentic.arc_competition_agent import (
    CLAIMED,
    CarnotAgentPolicy,
    E3AgentPolicy,
    load_cross_game_value_head,
    load_solutions,
    _level_of,
)

_VALUE_HEAD = None  # BRIDGE: the cross-game value head, loaded once for the explorer_vh policy
_DISC_ROUTER = None  # Exp4556: cross-game DiscriminativeVerifier candidate router, loaded once
_RANDOM_ROUTER = None  # Exp4556 positive-control router
_PROPOSER = None  # E3: a stronger world-model proposer (a bigger GGUF), loaded once + reused
_PROPOSER_REPO = ""  # repo substr for the E3 proposer (e.g. "Qwen3.6-35B-A3B"); "" = E3 default 12B
# _VALUE_WEIGHT: A* blend weight for explorer_vh; 0 = depth-primary tiebreak (neutral, safe).
# weight 5 REGRESSED the live explorer (6/32 vs 8/32) -- the value head misroutes
# the depth-first-ride structure. Set via --value-weight for sweeps.
# (This was an aligned trailing comment until 2026-07-26; `ruff format` dedents such continuations
# to column 0, where they read as annotating the NEXT statement instead. Written as a leading block
# so the note stays attached to the name it describes.)
_VALUE_WEIGHT = 0.0


def _oracle_levels() -> dict:
    """The per-game levels our OFFLINE oracle (GameAdapter/OfflineSolver path) reproduces, read from
    ops/arc_solve_registry.yaml. This is the upper bound the FRAME-ONLY LIVE path is measured against:
    the honest 'live gap' = oracle_levels - live_frame_only_levels, per game and in total."""
    import yaml

    d = yaml.safe_load((REPO / "ops" / "arc_solve_registry.yaml").read_text())
    return {
        g["game"]: int(g.get("levels_reproduced", 0))
        for g in d.get("games", [])
        if g.get("reproducibility") == "reproduced" and int(g.get("levels_reproduced", 0)) > 0
    }


def _build_policy(kind: str, game: str):
    """The LIVE agent policy under test -- NO banked solution, NO GameAdapter, NO internal-state reads
    (the unseen-game simulation). 'explorer' = tier-1 graph_explore only; 'e3' = the FULL competition
    cascade (graph_explore -> E3 executable-world-model induction on stall) that make_carnot_agent runs."""
    if kind == "e3":
        # INDUCTION ARM: optionally use a STRONGER world-model proposer (a bigger GGUF) -- the lever
        # where gemma-4-12B closed 0/6. One proposer instance is loaded + reused across games.
        global _PROPOSER
        if _PROPOSER_REPO and _PROPOSER is None:
            from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

            # port 8920 (NOT the conductor's E3 default 8919): LocalGGUFProposer reuses any server on its
            # port WITHOUT a model check, so a shared port silently runs the wrong model (a stale gemma
            # server made an earlier "Qwen" run actually use gemma-12B). A distinct port isolates this
            # test from the conductor's E3 work and guarantees the requested model loads.
            _PROPOSER = LocalGGUFProposer(repo_substr=_PROPOSER_REPO, port=8920)
        return E3AgentPolicy(game, proposer=_PROPOSER)
    if kind in ("explorer_vh", "explorer_bf"):  # BRIDGE: value-head-routed explorer
        global _VALUE_HEAD
        if _VALUE_HEAD is None:
            _VALUE_HEAD = load_cross_game_value_head()
        # explorer_bf = BEST-FIRST search (the graph_explore form where the value head's routing helped,
        # unlocking cn04); explorer_vh = the default depth-first-ride with an A*-frontier nudge.
        mode = "best_first" if kind == "explorer_bf" else "depth_first_ride"
        return CarnotAgentPolicy(
            game,
            {},
            force_explore=True,
            value_head=_VALUE_HEAD,
            value_weight=_VALUE_WEIGHT,
            search_mode=mode,
        )
    if kind in ("explorer_goalbias", "goalbias"):  # zero-shot order-prior goal-biased explorer
        from carnot.agentic.arc_goal_bias import GoalBiasValueHead

        # best_first A* frontier nudged toward more-ordered (plausibly-winning) states by a hand-crafted
        # distance-to-win prior (NO trained weights). value_weight defaults to 3.0 here (the global
        # --value-weight default is 0, which would disable the nudge); override with --value-weight.
        # REFUTED 2026-06-21 (fixed direction misroutes); use explorer_confirm instead.
        _w = _VALUE_WEIGHT if _VALUE_WEIGHT > 0 else 3.0
        return CarnotAgentPolicy(
            game,
            {},
            force_explore=True,
            value_head=GoalBiasValueHead(),
            value_weight=_w,
            search_mode="best_first",
        )
    if kind in ("explorer_confirm", "confirm"):  # direction-AGNOSTIC, online-confirming goal-bias
        from carnot.agentic.arc_goal_bias import ConfirmingGoalBiasValueHead

        # depth_first_ride (NOT best_first): best_first's priority is PURE value (no depth term -> value_weight
        # ignored), which ballooned paths (lp85 20->437) and tanked the efficiency score. depth_first_ride is
        # depth-PRIMARY (priority = depth + value_weight*value) -- it preserves the action-efficient ride that
        # earns the deep wins, while the value head nudges the frontier toward extremal (plausibly-winning)
        # states. Stateful head -> fresh PER GAME (correct: _build_policy is per game). --value-weight overrides.
        _w = _VALUE_WEIGHT if _VALUE_WEIGHT > 0 else 3.0
        return CarnotAgentPolicy(
            game,
            {},
            force_explore=True,
            value_head=ConfirmingGoalBiasValueHead(),
            value_weight=_w,
            search_mode="depth_first_ride",
        )
    if kind in ("explorer_dv", "verifier_router"):
        global _DISC_ROUTER
        if _DISC_ROUTER is None:
            from carnot.agentic.arc_discriminative_router import (
                load_cross_game_discriminative_router,
            )

            _DISC_ROUTER = load_cross_game_discriminative_router(root=REPO)
        return CarnotAgentPolicy(game, {}, force_explore=True, candidate_router=_DISC_ROUTER)
    if kind in ("explorer_random_router", "random_router"):
        global _RANDOM_ROUTER
        if _RANDOM_ROUTER is None:
            from carnot.agentic.arc_discriminative_router import RandomCandidateRouter

            _RANDOM_ROUTER = RandomCandidateRouter(seed=4556)
        return CarnotAgentPolicy(game, {}, force_explore=True, candidate_router=_RANDOM_ROUTER)
    return CarnotAgentPolicy(
        game, {}, force_explore=True
    )  # force_explore -> ignores any banked plan


def _baseline_actions(env, game: str) -> dict:
    """Per-level human/reference action counts if the env exposes them (efficiency
    denominator). Best-effort; returns {} if unavailable offline."""
    for attr in ("baseline_actions", "human_actions", "reference_actions"):
        v = getattr(getattr(env, "info", env), attr, None)
        if v:
            return {i: int(x) for i, x in enumerate(v)} if isinstance(v, (list, tuple)) else dict(v)
    return {}


def _nav_uninstrumented(reason: str) -> dict:
    """The nav-diagnostics channel is UNAVAILABLE -- say so, don't imply a measured zero.

    The two legacy keys keep their zero values so no existing consumer breaks, but
    `instrumented: False` + `uninstrumented_reason` let a reader distinguish "measured,
    the agent never reset-replayed" from "this channel was never wired up". Emitting a
    bare 0.0 for an unmeasured quantity is how a DEAD CHANNEL reads as a clean null --
    the same defect this project already shipped once as a
    `getattr(env, "baseline_actions")` against a field that lives on `env.info`.
    """

    return {
        "instrumented": False,
        "uninstrumented_reason": str(reason),
        "reset_replay_steps": 0,
        "forward_walk_hit_rate": 0.0,
    }


def _navigation_diagnostics(policy) -> dict:
    """Expose the live explorer replay tax without making it a score metric.

    2026-07-26 (per-level reset attribution, Change 3): this function used to narrow the
    explorer's 24-key diagnostics dict (arc_competition_agent.py:1634-1662) down to TWO
    keys, discarding precisely the fields that CLASSIFY a reset -- `navigation_attempts`,
    `reset_replay_fallbacks`, and the exact / partial-forward-walk / similarity hit split.
    That projection loss is why the reset-composition analysis had to re-run the agent LIVE
    instead of reading persisted rows: the numbers were computed at :394 and then thrown
    away at :406-407. The full dict is already flat ints/floats/bools, so it is now passed
    through whole, with the two legacy keys retained (and coerced) for compatibility.
    """

    explorer = getattr(policy, "explorer", None)
    if explorer is None:
        return _nav_uninstrumented("policy_has_no_explorer_attribute")
    if not hasattr(explorer, "navigation_diagnostics"):
        return _nav_uninstrumented("explorer_lacks_navigation_diagnostics")
    try:
        diagnostics = explorer.navigation_diagnostics()
    except Exception as exc:  # a diagnostics bug must not take the measurement down
        return _nav_uninstrumented(f"navigation_diagnostics_raised:{type(exc).__name__}")
    if not isinstance(diagnostics, dict):
        return _nav_uninstrumented(f"navigation_diagnostics_returned:{type(diagnostics).__name__}")
    out = {str(k): _json_safe(v) for k, v in diagnostics.items()}
    # The two legacy keys are read positionally by run_game's return dict; keep their types.
    out["reset_replay_steps"] = int(out.get("reset_replay_steps") or 0)
    out["forward_walk_hit_rate"] = float(out.get("forward_walk_hit_rate") or 0.0)
    out["instrumented"] = True
    out["uninstrumented_reason"] = None
    return out


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _frame_public_summary(frame, *, frame_index: int, action_count: int) -> dict:
    """Small public-frame receipt for post-run gap characterization."""

    row = {
        "frame_index": int(frame_index),
        "action_count": int(action_count),
        "levels_completed": _level_of(frame),
    }
    try:
        import numpy as np
        from carnot.agentic.arc_agi3_world_model import grid_of

        grid = np.asarray(grid_of(frame))
        row["grid_shape"] = [int(x) for x in grid.shape]
        row["grid_hash"] = "sha256:" + hashlib.sha256(grid.tobytes()).hexdigest()
        row["colors"] = [int(x) for x in np.unique(grid).tolist()]
    except Exception:
        row["grid_shape"] = []
        row["grid_hash"] = ""
        row["colors"] = []
    try:
        actions = getattr(frame, "available_actions", []) or []
        row["available_actions"] = [str(getattr(a, "name", a)) for a in actions]
    except Exception:
        row["available_actions"] = []
    return row


def _policy_diagnostics(policy) -> dict:
    proposer = getattr(policy, "proposer", None)
    explorer = getattr(policy, "explorer", None)
    diagnostics = {
        "phase": getattr(policy, "phase", None),
        "explore_budget": getattr(policy, "explore_budget", None),
        "target_levels": getattr(policy, "target_levels", None),
        "strategy_route": getattr(policy, "strategy_route", None),
        "feature_router": getattr(policy, "feature_router", None),
        "level_induction_events": getattr(policy, "level_induction_events", []),
        "induction_attempts": getattr(policy, "induction_attempts", []),
        "proposer": {
            "instantiated": proposer is not None,
            "repo_substr": getattr(proposer, "repo_substr", None),
            "port": getattr(proposer, "port", None),
            "mtp": getattr(proposer, "mtp", None),
            "kv_quant": getattr(proposer, "kv_quant", None),
            "last_stop_type": getattr(proposer, "last_stop_type", None),
            "last_prompt_truncated": getattr(proposer, "last_prompt_truncated", None),
            "last_raw_completion_len": len(getattr(proposer, "last_raw_completion", "") or ""),
        },
    }
    if explorer is not None:
        for name in (
            "adaptive_budget_diagnostics",
            "lazy_value_diagnostics",
            "goal_bias_diagnostics",
            "goal_candidate_guidance_diagnostics",
            "action_effect_expansion_prior_diagnostics",
            "action_salience_diagnostics",
            "curiosity_diagnostics",
            "qd_generation_diagnostics",
            "controllable_novelty_diagnostics",
            "object_centric_proposal_diagnostics",
            "program_synthesis_filter_diagnostics",
            "transition_cycle_diagnostics",
        ):
            fn = getattr(explorer, name, None)
            if callable(fn):
                try:
                    diagnostics[name] = fn()
                except Exception as exc:
                    diagnostics[name] = {"error": repr(exc)[:160]}
    return _json_safe(diagnostics)


# =========================================================================================
# PER-LEVEL (per-SEGMENT) RESET ATTRIBUTION -- three units, two independent accountings.
#
# WHY SEGMENTS AND NOT TOTALS. The authoritative scorer charges each COMPLETED level
# `actions_at_level - prev_actions` (arc_agi/scorecard.py:479) and scores it
# `min((baseline / charged)**2 * 100, 115)` (:166-173). It therefore DIFFERENCES a vector of
# cumulative checkpoints -- so the quantity that sets a level's score is the SPAN between two
# consecutive level-ups, and a whole-run `n_resets` cannot be apportioned across those spans
# after the fact. Attribution has to be recorded per span, while the run is happening.
#
# WHY THREE UNITS, NAMED. This project has already flipped a conclusion by conflating them:
#   offline_actions  -- this harness's `actions`. EXCLUDES resets (:355-361 increments it only
#                       in the non-RESET branch). What `level_up_actions` has always been in.
#   frames           -- loop iterations, resets INCLUDED. The unit the early-stop grace window
#                       counts in, so a grace of G frames buys fewer than G actions.
#   gateway_charged  -- offline_actions + resets. The ONLY unit the competition score is a
#                       function of, because `inc_reset_count` (scorecard.py:701-704, reached
#                       from update_scorecard:839-843) does `resets += 1` AND `actions += 1`.
# A single number in an unnamed unit is the defect, so every segment carries all three.
# =========================================================================================

_SEGMENT_UNIT_KEYS = ("offline_actions", "resets", "frames")


def _new_segment() -> dict:
    """A fresh open span. Zeros here are MEASURED zeros -- the span has just begun."""

    return {"offline_actions": 0, "resets": 0, "frames": 0}


def _close_segment(seg: dict, level_completed) -> dict:
    """Seal an open span and derive its gateway-charged length."""

    out = {k: int(seg.get(k) or 0) for k in _SEGMENT_UNIT_KEYS}
    out["gateway_charged"] = out["offline_actions"] + out["resets"]
    out["level_completed"] = None if level_completed is None else int(level_completed)
    return out


def segment_attribution_from_frame_sequence(frame_sequence) -> dict:
    """INDEPENDENT (channel 2) derivation of the same spans, from the recorded frames alone.

    This deliberately re-derives what run_game's in-loop accumulators already produce, from a
    different source (the persisted `frame_sequence` rows rather than the live counters), so the
    two can be cross-checked against each other. Two accountings that must agree is the only
    way an off-by-one in a counting loop announces itself; one accounting just looks plausible.

    Level JUMPS: a jump of k levels in one frame closes k spans, the FIRST carrying the whole
    cost and the rest zero -- because that is what the gateway's `actions_by_level` list does
    (`set_levels_completed` appends ONE entry per observed change, so a jump appends one entry
    and the scorer charges the remaining levels off the tail).
    """

    segments: list[dict] = []
    cur = _new_segment()
    prev_level = None
    for fr in frame_sequence or []:
        kind = ((fr or {}).get("move") or {}).get("kind")
        cur["frames"] += 1
        if kind == "RESET":
            cur["resets"] += 1
        elif kind is not None:
            cur["offline_actions"] += 1
        lvl = (fr or {}).get("levels_completed")
        if lvl is None:
            continue
        lvl = int(lvl)
        if prev_level is None:  # first observed level only SEEDS -- matches run_game's `start`
            prev_level = lvl
            continue
        if lvl > prev_level:
            for j in range(lvl - prev_level):
                closed = cur if j == 0 else _new_segment()
                segments.append(_close_segment(closed, prev_level + j + 1))
            cur = _new_segment()
        prev_level = lvl
    return {
        "segments": segments,
        "tail": _close_segment(cur, None),
        "n_segments": len(segments),
        "resets_in_completed_segments": sum(int(s["resets"]) for s in segments),
        "resets_in_tail": int(cur["resets"]),
    }


def _build_level_reset_attribution(
    *,
    segments: list[dict],
    open_tail: dict,
    frame_sequence,
    total_offline_actions: int,
    total_resets: int,
) -> dict:
    """Assemble the attribution + PROVE it against the whole-run counters and channel 2.

    Every field here POPULATES unconditionally, including on a crash/early-break path: the
    accumulators are plain ints that exist from the first line of run_game, so a run that dies
    on frame 1 still emits real (small) spans rather than a null that a reader will silently
    read as zero. `discrepancies` is an empty LIST when clean, never None.
    """

    segs = [_close_segment(s, s.get("level_completed")) for s in segments]
    tail = _close_segment(open_tail, None)
    charged_total = int(total_offline_actions) + int(total_resets)

    def _tot(unit: str) -> int:
        return sum(int(s[unit]) for s in segs) + int(tail[unit])

    discrepancies: list[dict] = []
    for unit, expected in (
        ("offline_actions", int(total_offline_actions)),
        ("resets", int(total_resets)),
        ("gateway_charged", charged_total),
    ):
        got = _tot(unit)
        if got != expected:
            discrepancies.append(
                {"check": f"segments_plus_tail_eq_run_total[{unit}]", "got": got, "want": expected}
            )

    # Channel 2: the frame_sequence-derived spans. NOTE a KNOWN, benign divergence class --
    # `frame_sequence` only receives a row when `latest is not None` (run_game skips the append
    # and breaks on a None frame), while the in-loop accumulators count that terminal move. So a
    # run that ends on a None frame legitimately shows channel 2 short by exactly one move. That
    # is recorded as a discrepancy rather than smoothed over, because "which channel is short,
    # and by how much" is information; silently reconciling it is how a real off-by-one hides.
    derived = segment_attribution_from_frame_sequence(frame_sequence)
    if derived["n_segments"] != len(segs):
        discrepancies.append(
            {
                "check": "channel2_n_segments",
                "got": derived["n_segments"],
                "want": len(segs),
            }
        )
    else:
        for i, (a, b) in enumerate(zip(segs, derived["segments"])):
            for unit in (*_SEGMENT_UNIT_KEYS, "gateway_charged"):
                if int(a[unit]) != int(b[unit]):
                    discrepancies.append(
                        {
                            "check": f"channel2_segment[{i}][{unit}]",
                            "got": int(b[unit]),
                            "want": int(a[unit]),
                        }
                    )

    return {
        "unit_definitions": {
            "offline_actions": "this harness's `actions`; EXCLUDES resets",
            "frames": "loop iterations; INCLUDES resets",
            "gateway_charged": "offline_actions + resets; the unit the scorer bills",
        },
        "segments": segs,
        "tail": tail,
        "n_segments": len(segs),
        # Flat per-unit projections -- the shape a downstream analyser wants without walking dicts.
        "segment_offline_actions": [int(s["offline_actions"]) for s in segs],
        "segment_resets": [int(s["resets"]) for s in segs],
        "segment_frames": [int(s["frames"]) for s in segs],
        "segment_gateway_charged": [int(s["gateway_charged"]) for s in segs],
        "tail_offline_actions": int(tail["offline_actions"]),
        "tail_resets": int(tail["resets"]),
        "tail_frames": int(tail["frames"]),
        "tail_gateway_charged": int(tail["gateway_charged"]),
        "resets_in_completed_segments": sum(int(s["resets"]) for s in segs),
        "resets_in_tail": int(tail["resets"]),
        "run_total_offline_actions": int(total_offline_actions),
        "run_total_resets": int(total_resets),
        "run_total_gateway_charged": charged_total,
        "reconciles": not discrepancies,
        "discrepancies": discrepancies,
        "channel2_frame_sequence_derived": derived,
    }


def _read_gateway_card(arc, game: str) -> dict:
    """Read the GATEWAY'S OWN bookkeeping for `game` off the live `Card` object.

    WHY THIS EXISTS (2026-07-27). Everything this project previously called a
    "gateway-charged" number was a MODEL: it reconstructed the charge as
    `offline_actions + resets` (optionally minus a free opening reset). An
    adversarial review of that model found it wrong on 17 of 44 measured cells,
    and SIGN-FLIPPED on 6 -- because a non-RESET action taken while the game is
    already GAME_OVER or WIN returns `frame=[]`
    (`arcengine/base_game.py:204-216`), and the scorecard update is gated on
    `len(resp.frame) > 0` (`arc_agi/wrapper.py:187`). Those post-death actions
    are FREE at the gateway and were being counted as charged by the model. The
    same gate is on the HTTP path (`arc_agi/api.py:336` -> `g.step` ->
    `_set_last_response`), so this is server behaviour, not an offline artifact.

    The fix is to stop modelling and READ. `run_game` already constructs its
    arcade with `scorecard_id=arc.open_scorecard()`, and
    `LocalEnvironmentWrapper` is handed the arcade's `scorecard_manager`
    (`arc_agi/base.py:789`), so `card.actions[idx]`, `card.resets[idx]` and
    `card.actions_by_level[idx]` ARE the gateway's own numbers for this run --
    the exact values the leaderboard scorer consumes.

    READING THE RIGHT PLAY ROW. `inc_play_count` appends a fresh zeroed row per
    play. The arcade's own construction reset creates play 0 (zero actions) and
    the agent's opening RESET creates play 1, so the row we want is the LAST
    one. (The scorer takes a max over plays, so a zero-action play 0 is
    harmless -- but reading `idx = total_plays - 1` keeps the recorded numbers
    aligned with the play the agent actually drove.)

    Returns an empty dict when no card can be found, so a consumer can
    distinguish "read, nothing charged" from "never read".
    """
    try:
        mgr = getattr(arc, "scorecard_manager", None)
        if mgr is None:
            return {}
        best = None
        for scorecard in getattr(mgr, "scorecards", {}).values():
            for gid, card in getattr(scorecard, "cards", {}).items():
                if str(gid).startswith(game):
                    best = card
        if best is None:
            return {}
        dumped = best.model_dump() if hasattr(best, "model_dump") else dict(best)
        plays = int(dumped.get("total_plays") or 0)
        if plays <= 0:
            return {"gateway_card_total_plays": 0}
        idx = plays - 1
        abl = dumped.get("actions_by_level") or []
        row_abl = abl[idx] if idx < len(abl) else []
        return {
            "gateway_card_total_plays": plays,
            "gateway_card_play_index_read": idx,
            "gateway_card_actions": int((dumped.get("actions") or [0])[idx]),
            "gateway_card_resets": int((dumped.get("resets") or [0])[idx]),
            # [(levels_completed, charged_actions_at_that_moment), ...] -- the
            # gateway's per-level cumulative charge vector, i.e. the exact input
            # to `scorecard.py:479 level_actions = actions_at_level - prev`.
            "gateway_card_actions_by_level": [[int(a), int(b)] for a, b in (row_abl or [])],
            "gateway_card_actions_all_plays": [int(x) for x in (dumped.get("actions") or [])],
            "gateway_card_resets_all_plays": [int(x) for x in (dumped.get("resets") or [])],
        }
    except Exception as exc:  # pragma: no cover - defensive; never fail a run on bookkeeping
        return {"gateway_card_read_error": f"{type(exc).__name__}: {str(exc)[:120]}"}


def supervisor_row_field(policy) -> dict[str, Any]:
    """The row's `trajectory_supervisor` value. Never absent, never None.

    WHY THIS EXISTS HERE (2026-09-01). This eval is the live-path harness, and it collected NO
    supervisor evidence at all: `trajectory_supervisor_diagnostics()` had exactly three callers,
    and every one of them was unusable for a live applied run. The lever harness is frozen on a
    RETIRED generator and refuses to start; exp6776 is hardcoded to shadow mode, and a shadow
    receipt is not redirect evidence; exp6558 only reduces receipts others produce. So a 13.8-hour
    live run on 2026-09-01 produced two rows with no supervisor field, and the refinement ledger
    could not move however long the loop ran.

    The shape is copied deliberately from `arc_scored_path_lever_harness.supervisor_row_field`,
    including its failure handling: a raising diagnostics call becomes an error marker rather than
    taking the row down, because an absent field reads as zero to a flat consumer and that
    ambiguity cost a full day of wrong A/B reporting on 2026-08-21.
    """
    try:
        d = policy.trajectory_supervisor_diagnostics()
    except Exception as exc:  # instrumentation must never mask the run itself
        return {"error": f"{type(exc).__name__}:{exc}"}
    if not isinstance(d, dict):
        return {"error": f"non_dict_diagnostics:{type(d).__name__}"}
    return d


def generator_channels_row_field(policy) -> dict[str, Any]:
    """The row's `generator_channels` value. Never absent, never None.

    WHY THIS EXISTS (2026-09-04). The proposer counts what the generator actually returned --
    `chat_completions`, `chars_final`, `chars_reasoning` -- and the ONLY code that ever read those
    into an artifact is `arc_scored_path_lever_harness.py`, which is frozen on the RETIRED
    Qwen3.5-9B-MTP pin and refuses to start against the live generator. So all ELEVEN eval
    artifacts in `results/arc_leaderboard_eval_runs/` carry zero rows with these counters, including
    runs whose own `llm_reached` is true.

    That is the same shape as the supervisor gap fixed three days earlier: a diagnostic that exists
    on the object, has one caller, and that caller cannot run. The practical cost is that "did the
    induce tier emit anything after the n_ctx fix" is not answerable from any artifact, so a
    multi-hour live run would have produced another row that cannot answer it.

    Failure handling is copied from `supervisor_row_field` for the same reason: an absent field
    reads as zero to a flat consumer, so every path returns a dict and a broken instrument says so.
    """
    try:
        proposer = getattr(policy, "proposer", None)
        if proposer is None:
            # Not an error: the LLM tier may legitimately never have fired.
            return {"proposer": "absent"}
        totals = getattr(proposer, "channel_totals", None)
    except Exception as exc:  # instrumentation must never mask the run itself
        return {"error": f"{type(exc).__name__}:{exc}"}
    if totals is None:
        return {"error": "no_channel_totals_attr"}
    if not isinstance(totals, dict):
        return {"error": f"non_dict_channel_totals:{type(totals).__name__}"}
    return dict(totals)


PROGRESS_SCHEMA = "carnot.arc.eval_progress.v1"
# Loop steps between periodic heartbeats. Measured 2026-09-04: the classical path runs 272
# actions per second with the LLM off, so 100 steps is well under a second of search. Only a
# generator call can leave the file stale, and the induction hook marks that case.
PROGRESS_EVERY_STEPS = 100


def _iso(ts: float) -> str:
    """UTC ISO-8601 with an explicit offset, the same shape the policy stamps on attempts."""
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds")


class ProgressWriter:
    """In-run heartbeat for the game in flight (REQ-ARC-WMTE-7010).

    WHY. A single-game eval wrote nothing until it ended, so a reader could not tell a
    level-up from a hang for hours (see the 2026-09-04 four-axis baseline note). This file
    is rewritten atomically on every event that matters (level-up, induce start and end, a
    new induction attempt) and every PROGRESS_EVERY_STEPS loop steps, so a reader judges
    progress from the file rather than from CPU percent. A step that is both a level-up
    and a period boundary writes twice; that is harmless and only inflates `writes`.

    Instrumentation never takes the run down: every public method swallows its own
    errors and counts them in `write_errors`, which the final row carries. No write
    happens at construction; the first write is the first loop step inside `run_game`.
    """

    def __init__(
        self, path, *, game: str, game_index: int, games_planned: int, policy=None
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.game = str(game)
        self.game_index = int(game_index)
        self.games_planned = int(games_planned)
        self.policy = policy
        self.started_at = time.time()
        self.last_event = "start"
        self.writes = 0
        self.write_errors = 0
        self.game_complete = False
        self.induction_in_flight: dict[str, Any] | None = None
        self.last_induction: dict[str, Any] | None = None
        self._attempts_seen = 0
        self._counters: dict[str, Any] = {
            "loop_index": 0,
            "actions": 0,
            "resets": 0,
            "start_level": 0,
            "level": 0,
            "level_up_actions": [],
        }
        # Register with the policy so a 30-minute generator call is visible while it runs
        # (REQ-ARC-WMTE-7011). A policy without the seam simply gets no in-flight marker.
        if policy is not None and hasattr(policy, "induction_progress_hook"):
            try:
                policy.induction_progress_hook = self.on_induction_event
            except Exception:
                self.write_errors += 1

    def elapsed_s(self) -> float:
        return round(time.time() - self.started_at, 3)

    def on_induction_event(self, kind: str, payload: Any) -> None:
        """The policy's hook: an induce started, or one finished (with its wall time)."""
        try:
            payload = dict(payload) if isinstance(payload, dict) else {"raw": payload}
            if kind == "induction_started":
                self.induction_in_flight = payload
            else:
                self.induction_in_flight = None
                self.last_induction = payload
                print(
                    f"  {self.game} induce#{payload.get('attempt_index')} "
                    f"reason={payload.get('reason')} planned={payload.get('planned')} "
                    f"skipped={payload.get('skipped') or '-'} "
                    f"transitions={payload.get('transition_count')} "
                    f"wall={float(payload.get('wall_s') or 0.0):.0f}s [{self.elapsed_s():.0f}s]",
                    flush=True,
                )
            self.write(str(kind))
        except Exception:
            self.write_errors += 1

    def step(
        self, *, loop_index: int, actions: int, resets: int, level: int, start_level: int
    ) -> None:
        """Once per loop step. Writes on a period boundary or a new induction attempt."""
        try:
            c = self._counters
            c["loop_index"] = int(loop_index)
            c["actions"] = int(actions)
            c["resets"] = int(resets)
            c["level"] = int(level)
            c["start_level"] = int(start_level)
            attempts = getattr(self.policy, "induction_attempts", None)
            n_attempts = len(attempts) if isinstance(attempts, list) else 0
            if n_attempts != self._attempts_seen:
                self._attempts_seen = n_attempts
                self.write("induction_attempt_recorded")
            elif int(loop_index) % PROGRESS_EVERY_STEPS == 0:
                self.write("periodic")
        except Exception:
            self.write_errors += 1

    def level_up(self, *, from_level: int, to_level: int, actions: int) -> None:
        try:
            c = self._counters
            c["level"] = int(to_level)
            c["actions"] = int(actions)
            c["level_up_actions"] = list(c["level_up_actions"]) + [int(actions)]
            print(
                f"  {self.game} level {from_level}->{to_level} at action {actions} "
                f"[{self.elapsed_s():.0f}s]",
                flush=True,
            )
            self.write("level_up")
        except Exception:
            self.write_errors += 1

    def finish(self) -> None:
        try:
            self.game_complete = True
            self.write("end")
        except Exception:
            self.write_errors += 1

    def snapshot(self) -> dict[str, Any]:
        """The heartbeat record. `game_complete`, not `complete`: a reader of the partial
        files keys on `complete`, and this file is never a result record."""
        policy = self.policy
        attempts = getattr(policy, "induction_attempts", None)
        attempts = (
            [a for a in attempts if isinstance(a, dict)] if isinstance(attempts, list) else []
        )
        last = attempts[-1] if attempts else None
        generator_wall_s = round(sum(float(a.get("wall_s") or 0.0) for a in attempts), 3)
        sup: Any = (
            supervisor_row_field(policy)
            if hasattr(policy, "trajectory_supervisor_diagnostics")
            else {"enabled": False}
        )
        if not isinstance(sup, dict):
            sup = {"error": "non_dict_receipt"}
        redirects = sup.get("redirects")
        if not isinstance(redirects, list):
            redirects = sup.get("would_have_redirects")
        redirects = redirects if isinstance(redirects, list) else []
        last_redirect = redirects[-1] if redirects and isinstance(redirects[-1], dict) else None
        c = self._counters
        now = time.time()
        return {
            "schema": PROGRESS_SCHEMA,
            "pid": os.getpid(),
            "game": self.game,
            "game_index": self.game_index,
            "games_planned": self.games_planned,
            "started_at": _iso(self.started_at),
            "updated_at": _iso(now),
            "elapsed_s": round(now - self.started_at, 3),
            "last_event": self.last_event,
            "game_complete": self.game_complete,
            "loop_index": c["loop_index"],
            "actions": c["actions"],
            "resets": c["resets"],
            "start_level": c["start_level"],
            "level": c["level"],
            "levels": max(0, int(c["level"]) - int(c["start_level"])),
            "level_up_actions": list(c["level_up_actions"]),
            "induction_attempts_n": len(attempts),
            "induction_in_flight": self.induction_in_flight,
            "last_induction": self.last_induction
            or (
                {
                    k: last.get(k)
                    for k in ("reason", "planned", "skipped", "transition_count", "wall_s")
                }
                if last is not None
                else None
            ),
            "generator_wall_s": generator_wall_s,
            "supervisor": {
                "enabled": sup.get("enabled"),
                "mode": sup.get("mode"),
                "redirects_n": len(redirects),
                "last_redirect": (
                    {k: last_redirect.get(k) for k in ("arm", "action_index", "level")}
                    if last_redirect is not None
                    else None
                ),
            },
            "generator_channels": generator_channels_row_field(policy),
            "writes": self.writes,
            "write_errors": self.write_errors,
        }

    def write(self, event: str) -> None:
        self.last_event = str(event)
        if self.path is None:
            return
        try:
            _write_json_atomic(self.path, json.dumps(self.snapshot(), indent=1, default=str))
            self.writes += 1
        except Exception:
            self.write_errors += 1


def run_game(
    game: str, policy, *, budget: int, variant: int = 0, reflect=None, progress=None
) -> dict:
    # REQ-ARC-WMTE-7010: the game's own wall clock, so the row can say how long it took.
    _t_start = time.time()
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    if variant:
        # MANUFACTURED held-out layout variant (operator 2026-06-19): the agent observes a mechanic-
        # preserving recolor/reflection it has never seen, while the REAL env keeps the win logic -> a
        # solve here is a real solve and a genuine generic-transfer test (a bigger benchmark than the
        # 2/7 LOO on 25 games). See arc_variant_generator.VariantEnv.
        from carnot.agentic.arc_variant_generator import VariantEnv

        env = VariantEnv(env, game, variant, reflect=reflect)
    base = _baseline_actions(env, game)
    _counters_before = completion_counters(policy)
    frames, latest, actions = [], None, 0
    start = None
    best = None
    level_up_actions: list[
        int
    ] = []  # cumulative `actions` count at each level-up (for per-level cost)
    # GATEWAY-CHARGE INSTRUMENTATION (2026-07-26). `actions` above EXCLUDES
    # RESETs, but the LIVE gateway CHARGES a reset an action
    # (arc_agi/scorecard.py:701-704 `inc_reset_count` does `resets += 1` AND
    # `actions += 1`, reached from `update_scorecard`:839-843). Because the
    # scorer's per-level cost is a DIFFERENCE of cumulative CHARGED actions
    # (:479 `level_actions = actions_at_level - prev_actions`) and the per-level
    # score is min((baseline/level_actions)**2 * 100, 115), every per-level
    # efficiency number recorded before this instrumentation is OPTIMISTIC in
    # the SQUARED term by exactly the resets charged before that level-up.
    # Whole-run `n_resets` alone cannot recover that -- attribution has to be
    # PER LEVEL, which is what these two lists record. `resets_before_levelups`
    # is the cumulative reset count at each level-up; `level_up_charged` is a
    # MODEL of the gateway-charged count at each level-up (actions + resets so far).
    #
    # WARNING -- `level_up_charged` IS A MODEL AND IT IS MEASURABLY WRONG (2026-07-27).
    # This comment previously claimed it was "exactly what the gateway's actions_by_level
    # would hold". That claim was refuted by reading the arcade's OWN Card:
    # results/arc_gateway_card_ground_truth_20260727.json finds the model disagrees with the
    # real per-level charge vector on 17 of 44 cells, with 6 SIGN-FLIPPED, and it NEVER
    # understates (max signed error 0.058843).
    #
    # THE CAUSE IS NOT THE RESET PREDICATE -- it is POST-DEATH ACTIONS. A non-RESET action
    # taken while the game is already GAME_OVER or WIN returns `frame=[]`
    # (arcengine/base_game.py:204-216), and the scorecard update is gated on
    # `len(resp.frame) > 0` (arc_agi/wrapper.py:187), so those actions are FREE at the
    # gateway while this model counts them as charged. Measured: the uncharged count equals
    # the run's empty-frame action count on 48/48 cells. The same gate is on the HTTP path
    # (arc_agi/api.py:336), so it is server behaviour, not an offline artifact. The free
    # OPENING reset is a real but separate and much smaller effect; attributing the whole
    # discrepancy to it was itself a wrong intermediate answer on the way here.
    #
    # THE GROUND TRUTH IS `gateway_card_actions` / `gateway_card_actions_by_level`, read off
    # the live Card by `_read_gateway_card` below. Prefer those for any score claim. This
    # modelled pair is retained (never-prune) because five committed artifacts cite it and
    # because a model that never understates is a usable UPPER BOUND -- but it is a bound,
    # not the charge. Do not reintroduce the "exactly what the gateway would hold" framing:
    # two independent reimplementations of this same `actions + resets - k` shape agreed
    # 44/44 with each other and were BOTH wrong, which is why agreement between
    # reconstructions is not evidence about the gateway.
    resets = 0
    resets_before_levelups: list[int] = []
    level_up_charged: list[int] = []
    # PER-SEGMENT accumulators (2026-07-26). `resets_before_levelups` / `level_up_charged` above
    # are CUMULATIVE checkpoints; the scorer's per-level denominator is the DIFFERENCE between
    # consecutive checkpoints, and it needs each span in all three units (see the block comment
    # above `_new_segment`). `_seg` is the currently-open span; `level_segments` holds the sealed
    # ones. Pure accumulation: nothing here is ever read by a branch predicate.
    _seg = _new_segment()
    level_segments: list[dict] = []
    frame_sequence = []
    # CHARGE-MECHANISM OBSERVATION (2026-07-27). Three counters that make the
    # gateway's charge behaviour OBSERVED rather than assumed. Each was a bug in
    # the prior modelled accounting:
    #   * `observed_full_resets` -- a reset is FREE exactly when the game takes
    #     the `full_reset()` branch, because `arc_agi/wrapper.py:187-195` gates
    #     the scorecard update on a non-empty frame and passes `resp.full_reset`
    #     through. The predicate is `_action_count == 0 or state == WIN`
    #     (`arcengine/base_game.py:305-316 handle_reset`) -- NOT "the first
    #     reset of the run". It can therefore fire more than once: on env
    #     construction, on a RESET with no intervening action, and on a RESET
    #     immediately after a win. The old model hard-coded 1.
    #   * `empty_frame_actions` -- a non-RESET action taken while the state is
    #     GAME_OVER/WIN returns `frame=[]` (`arcengine/base_game.py:204-216`),
    #     so the scorecard update is skipped and the action is FREE. Our harness
    #     counts it. This is the defect that made the modelled charge overstate.
    #   * `consecutive_reset_pairs` -- the trajectory property the old model's
    #     "one free reset" assumption silently depended on. Recording it means a
    #     future violation announces itself instead of under-charging quietly.
    # All three are pure accumulation: nothing here is read by a branch predicate.
    observed_full_resets = 0
    empty_frame_actions = 0
    consecutive_reset_pairs = 0
    prev_kind = None
    for step_index in range(budget):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
            resets += 1
            _seg["resets"] += 1
            if bool(getattr(latest, "full_reset", False)):
                observed_full_resets += 1
            if prev_kind == "RESET":
                consecutive_reset_pairs += 1
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            actions += 1
            _seg["offline_actions"] += 1
            if not (getattr(latest, "frame", None) or []):
                empty_frame_actions += 1
        prev_kind = kind if kind == "RESET" else "ACTION"
        _seg["frames"] += 1
        if start is None:
            start = _level_of(latest)
            best = start
        lvl = _level_of(latest)
        if best is not None and lvl > best:
            if progress is not None:
                progress.level_up(from_level=int(best), to_level=int(lvl), actions=actions)
            for _lv in range(
                best, lvl
            ):  # record the action count for each new level (handles jumps)
                level_up_actions.append(actions)
                resets_before_levelups.append(resets)
                level_up_charged.append(actions + resets)
                # SEAL the span. A multi-level JUMP closes the first level with the whole
                # span's cost and the rest at zero, mirroring the gateway's actions_by_level
                # (one appended entry per observed change, remaining levels charged off the
                # tail). `_lv == best` is true only on the first iteration -- `best` is not
                # reassigned until after this loop -- so the open span is consumed once.
                level_segments.append(
                    _close_segment(_seg if _lv == best else _new_segment(), _lv + 1)
                )
            _seg = _new_segment()
            best = lvl
        if progress is not None:
            progress.step(
                loop_index=step_index,
                actions=actions,
                resets=resets,
                level=int(lvl or 0),
                start_level=int(start or 0),
            )
        frames.append(latest)
        if latest is not None:
            frame_row = _frame_public_summary(
                latest,
                frame_index=len(frames) - 1,
                action_count=actions,
            )
            frame_row["move"] = _json_safe({"kind": kind, "data": data})
            frame_row["loop_index"] = int(step_index)
            frame_sequence.append(frame_row)
        if latest is None:
            break
    reached = _level_of(latest)
    levels = max(0, reached - (start or 0))
    # REAL per-game score via the AUTHORITATIVE scorer (arc_agi.scorecard.EnvironmentScoreCalculator,
    # package arc-agi 0.9.8). A 2026-06-20 adversarial review caught that reimplementing the formula from a
    # doc paraphrase was WRONG on three counts: the real per-level score is min((human/agent)^2 * 100, 115)
    # -- a 115 cap that REWARDS superhuman solves, NOT min(human/agent,1)^2 capped at 1; the per-game
    # aggregation is an INDEX-WEIGHTED MEAN over ALL the game's levels (deeper levels weigh more, and
    # UNSOLVED levels score 0 and drag the mean down -> solving MORE/DEEPER levels is the real lever); and a
    # missing/zero baseline scores 0.0, NOT 1.0 (the old code's inverse was a gameable hole). We now DRIVE
    # the installed scorer exactly as arc_agi/scorecard.py:474-491 does, so the gate cannot drift from the
    # leaderboard. lp85 (1 of 8 levels solved) -> ~2.007. Missing baselines -> 0.0 (matches the real scorer).
    eff = 0.0
    eff_precise = None
    per_level = []
    try:
        from arc_agi.scorecard import EnvironmentScoreCalculator

        baseline_list = [base[i] for i in sorted(base)] if base else []
        if baseline_list:
            calc = EnvironmentScoreCalculator()
            prev = 0
            for li in range(len(baseline_list)):
                if li < len(level_up_actions):
                    at = level_up_actions[li]
                    lvl_actions = at - prev
                    done = True
                    prev = at
                else:
                    done = False
                    lvl_actions = actions - prev
                    prev = actions
                calc.add_level(
                    level_index=li + 1,
                    completed=done,
                    actions_taken=lvl_actions,
                    baseline_actions=baseline_list[li],
                )
                per_level.append(
                    {
                        "level": li,
                        "agent_actions": lvl_actions,
                        "human_actions": baseline_list[li],
                        "completed": done,
                    }
                )
            _eff_raw = float(calc.to_score(include_levels=False).score)
            eff = round(_eff_raw, 4)
            # FULL-PRECISION COMPANION (2026-07-27). `eff` keeps its 4-dp rounding because a long
            # history of recorded numbers is in that unit and changing it would move published
            # values on rebuild. But a RATIO of two 4-dp values is badly quantised: a downstream
            # lane computed `rel_loss = optimism / offline` and got 0.0003/0.0072 = exactly 1/24 --
            # a six-decimal number carrying ONE significant figure. Consumers that divide must use
            # the precise field.
            eff_precise = _eff_raw
    except Exception:
        eff = 0.0  # no baselines / scorer unavailable -> 0.0 (matches the real scorer; NEVER 1.0)
    gap = None
    if reached <= (start or 0):
        gap = {
            "game": game,
            "stuck_at_level": reached,
            "actions_spent": actions,
            "signature": "no_level_up_within_budget",
            "needs": "richer exploration (salience tiers / frontier-dist nav) OR E3 world-model induction",
        }
    nav = _navigation_diagnostics(policy)
    # GATEWAY-ACCURATE efficiency (2026-07-26). Identical driving of the SAME
    # installed scorer, but fed the CHARGED per-level counts (which include the
    # resets the gateway bills) instead of the reset-free `level_up_actions`.
    # `efficiency` above is retained UNCHANGED so no historical comparison
    # silently shifts unit; this is an ADDITIONAL field.
    #
    # DEAD-CHANNEL FIX (2026-07-26). `eff_gateway` used to default to 0.0 both at init and on
    # the exception path, while `efficiency_optimism_vs_gateway` is computed as
    # `eff - eff_gateway`. So a scorer import failure -- or merely an empty baseline dict
    # (`base == {}` -> `baseline_list == []`, which is the NORMAL case for a game whose env
    # exposes no human baselines) -- silently reported the FULL value of `eff` as optimism,
    # i.e. "offline accounting is 100% optimistic": the most alarming finding the field can
    # express, emitted precisely when nothing was measured. It now defaults to None with an
    # explicit error/reason string, so "measured, no optimism" and "never measured" are
    # distinguishable by a consumer.
    eff_gateway = None
    eff_gateway_precise = None
    eff_gateway_error = None
    per_level_gateway = []
    try:
        from arc_agi.scorecard import EnvironmentScoreCalculator

        baseline_list = [base[i] for i in sorted(base)] if base else []
        if not baseline_list:
            eff_gateway_error = "no_baseline_actions_exposed_by_env"
        if baseline_list:
            calc_g = EnvironmentScoreCalculator()
            prev_g = 0
            charged_total = actions + resets
            for li in range(len(baseline_list)):
                if li < len(level_up_charged):
                    at_g = level_up_charged[li]
                    lvl_charged = at_g - prev_g
                    done_g = True
                    prev_g = at_g
                else:
                    done_g = False
                    lvl_charged = charged_total - prev_g
                    prev_g = charged_total
                calc_g.add_level(
                    level_index=li + 1,
                    completed=done_g,
                    actions_taken=lvl_charged,
                    baseline_actions=baseline_list[li],
                )
                per_level_gateway.append(
                    {
                        "level": li,
                        "agent_charged_actions": lvl_charged,
                        "human_actions": baseline_list[li],
                        "completed": done_g,
                    }
                )
            _effg_raw = float(calc_g.to_score(include_levels=False).score)
            eff_gateway = round(_effg_raw, 4)
            eff_gateway_precise = _effg_raw
    except Exception as exc:
        eff_gateway = None
        eff_gateway_precise = None
        eff_gateway_error = f"{type(exc).__name__}: {str(exc)[:120]}"
    level_reset_attribution = _build_level_reset_attribution(
        segments=level_segments,
        open_tail=_seg,
        frame_sequence=frame_sequence,
        total_offline_actions=actions,
        total_resets=resets,
    )
    # GATEWAY-CARD READ (2026-07-27) -- the MEASURED charge, not a model of it.
    # `eff_gateway` above reconstructs the charge as `actions + resets`; this
    # block instead drives the SAME installed scorer with the gateway's own
    # `actions_by_level` / `actions` numbers. Where the two disagree, the CARD is
    # authoritative by construction: it is literally the object the leaderboard
    # scorer consumes. Both are retained so no historical comparison silently
    # shifts unit, and so the disagreement itself is a first-class recorded
    # number (`gateway_card_vs_model_charged_delta`).
    card = _read_gateway_card(arc, game)
    eff_card = None
    eff_card_error = None
    per_level_card = []
    if not card or card.get("gateway_card_actions") is None:
        eff_card_error = card.get("gateway_card_read_error") or "no_card_found"
    else:
        try:
            from arc_agi.scorecard import EnvironmentScoreCalculator

            baseline_list = [base[i] for i in sorted(base)] if base else []
            if not baseline_list:
                eff_card_error = "no_baseline_actions_exposed_by_env"
            else:
                card_total = int(card["gateway_card_actions"])
                # actions_by_level is [(levels_completed, cumulative_charged), ...]
                card_cum = [int(b) for _lv, b in card["gateway_card_actions_by_level"]]
                calc_c = EnvironmentScoreCalculator()
                prev_c = 0
                for li in range(len(baseline_list)):
                    if li < len(card_cum):
                        at_c = card_cum[li]
                        lvl_c = at_c - prev_c
                        done_c = True
                        prev_c = at_c
                    else:
                        done_c = False
                        lvl_c = card_total - prev_c
                        prev_c = card_total
                    calc_c.add_level(
                        level_index=li + 1,
                        completed=done_c,
                        actions_taken=lvl_c,
                        baseline_actions=baseline_list[li],
                    )
                    per_level_card.append(
                        {
                            "level": li,
                            "agent_card_charged_actions": lvl_c,
                            "human_actions": baseline_list[li],
                            "completed": done_c,
                        }
                    )
                eff_card = round(float(calc_c.to_score(include_levels=False).score), 6)
        except Exception as exc:
            eff_card = None
            eff_card_error = f"{type(exc).__name__}: {str(exc)[:120]}"
    _counters_after = completion_counters(policy)
    # Snapshot provenance AFTER the loop, never before it. E3AgentPolicy builds its proposer
    # LAZILY on first use inside the loop, so a before-loop snapshot reads "no proposer" on
    # every default-arm row even when the LLM was reached (adversarial review, 2026-08-31).
    _gen_prov = generator_provenance(policy)
    _consumed = {
        k: _counters_after.get(k, 0) - _counters_before.get(k, 0)
        for k in set(_counters_before) | set(_counters_after)
    }
    # REQ-ARC-WMTE-7010/7011: wall clock for the game and for its generator calls. Before
    # this the only record of a 7-hour game's duration was a line on stdout.
    _t_end = time.time()
    _attempts = getattr(policy, "induction_attempts", None)
    _attempts = [a for a in _attempts if isinstance(a, dict)] if isinstance(_attempts, list) else []
    _attempt_walls = [a.get("wall_s") for a in _attempts]
    _generator_wall_s = round(sum(float(w) for w in _attempt_walls if w is not None), 3)
    if progress is not None:
        progress.finish()
    return {
        "game": game,
        "started_at": _iso(_t_start),
        "finished_at": _iso(_t_end),
        "wall_s": round(_t_end - _t_start, 3),
        "generator_wall_s": _generator_wall_s,
        "induction_attempt_wall_s": _attempt_walls,
        "progress_write_errors": (None if progress is None else int(progress.write_errors)),
        # GENERATOR PROVENANCE (2026-08-31). Without these two fields a row cannot be read as an
        # e3 result at all -- see `generator_provenance`. `completions_consumed == 0` means no
        # model was reached for this game, whatever the levels say.
        "generator_provenance": _gen_prov,
        "completions_consumed": _consumed,
        "llm_reached": bool(_consumed.get("completions", 0) > 0),
        # REQ-ARC-WMTE-6640 rule 5: emitted on EVERY row, both paths, so a consumer can tell
        # "supervisor off" from "supervisor absent". See supervisor_row_field above.
        "trajectory_supervisor": supervisor_row_field(policy),
        # 2026-09-04: the induce-channel counters, for the same reason and by the same
        # rule -- on EVERY row, so "the generator returned nothing" is distinguishable from
        # "nobody looked". See generator_channels_row_field above.
        "generator_channels": generator_channels_row_field(policy),
        "levels": levels,
        "reached": reached,
        "actions": actions,
        "efficiency": eff,
        "per_level_efficiency": eff,
        "per_level": per_level,
        # --- gateway-charge accounting (resets charged, as the live gateway does)
        "action_count_convention": "resets_excluded_run_game_native",
        "n_resets_run_game": resets,
        "charged_actions": actions + resets,
        "resets_before_levelups": resets_before_levelups,
        "level_up_charged": level_up_charged,
        "efficiency_gateway_charged": eff_gateway,
        "efficiency_gateway_charged_error": eff_gateway_error,
        # UNROUNDED companions. Any consumer taking a RATIO of these two must use these, not the
        # 4-dp fields above (see the comment at `eff_precise`).
        "efficiency_precise": eff_precise,
        "efficiency_gateway_charged_precise": eff_gateway_precise,
        "efficiency_optimism_vs_gateway_precise": (
            None
            if (eff_precise is None or eff_gateway_precise is None)
            else float(eff_precise) - float(eff_gateway_precise)
        ),
        "per_level_gateway": per_level_gateway,
        "efficiency_optimism_vs_gateway": (
            None if eff_gateway is None else round(float(eff) - float(eff_gateway), 6)
        ),
        # PER-LEVEL RESET ATTRIBUTION: every inter-level-up span in all three units, plus the
        # post-solve tail and a reconciliation against both the run totals and an independent
        # frame_sequence-derived re-derivation. This is the field that makes a per-level
        # efficiency claim auditable from a persisted row without re-running the agent.
        "level_reset_attribution": level_reset_attribution,
        # --- MEASURED gateway charge (read off the live Card), plus the three
        #     mechanism counters that explain where it differs from the model.
        **card,
        "efficiency_gateway_card": eff_card,
        "efficiency_gateway_card_error": eff_card_error,
        "per_level_gateway_card": per_level_card,
        "gateway_card_vs_model_charged_delta": (
            None
            if card.get("gateway_card_actions") is None
            else int(actions + resets) - int(card["gateway_card_actions"])
        ),
        "observed_full_resets": observed_full_resets,
        "empty_frame_actions": empty_frame_actions,
        "consecutive_reset_pairs": consecutive_reset_pairs,
        "deepest_level_reached": reached,
        "navigation_diagnostics": nav,
        "frame_sequence": frame_sequence,
        "policy_diagnostics": _policy_diagnostics(policy),
        "reset_replay_steps": nav["reset_replay_steps"],
        "forward_walk_hit_rate": nav["forward_walk_hit_rate"],
        "actions_to_first_levelup": (level_up_actions[0] if level_up_actions else None),
        "gap": gap,
    }


def _arg(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


def main() -> int:
    argv = sys.argv[1:]
    seed = int(_arg(argv, "--seed", os.environ.get("CARNOT_ARC_RANDOM_SEED", "20260719")))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    # --games oracle: measure the LIVE frame-only agent against the 16 games our OFFLINE oracle solved,
    #   reporting the honest GAP (oracle 32 levels - what the live path reaches with NO per-game knowledge).
    #   This is the north-star metric: live capability, not the offline reproducibility scorecard.
    # --policy e3: the FULL competition cascade (graph_explore + E3 world-model induction). default
    #   'explorer' = the fast tier-1 graph_explore floor (no LLM); 'e3' needs the local GGUF + is slower.
    games_mode = _arg(argv, "--games", "claimed")
    policy_kind = _arg(argv, "--policy", "explorer")
    budget = int(_arg(argv, "--budget", "20000"))
    global _VALUE_WEIGHT, _PROPOSER_REPO
    _VALUE_WEIGHT = float(
        _arg(argv, "--value-weight", "0")
    )  # explorer_vh A* blend weight (0 = neutral)
    _PROPOSER_REPO = _arg(argv, "--proposer", "")  # e3 stronger proposer repo substr ("" = 12B)
    oracle = _oracle_levels()
    games = sorted(oracle) if games_mode == "oracle" else list(CLAIMED)
    only = _arg(argv, "--only", "")  # --only g1,g2 : target a subset (e.g. the worst gaps)
    variant = int(
        _arg(argv, "--variant", "0")
    )  # 0 = real game; N>0 = manufactured held-out variant N
    reflect_arg = _arg(
        argv, "--reflect", ""
    )  # "" none; "0"/"1" = reflect axis (vertical/horizontal)
    reflect = int(reflect_arg) if reflect_arg != "" else None
    if only:
        keep = set(only.split(","))
        games = [g for g in games if g in keep]
    print(
        f"== ARC LIVE-loop eval — games={games_mode} policy={policy_kind} budget={budget} "
        f"(frame-only, no banked plan, no GameAdapter) ==",
        flush=True,
    )
    rows, total_levels, total_eff, gaps = [], 0, 0.0, []
    live_levels_sum, oracle_sum, gap_sum = 0, 0, 0

    def _payload(verdict: str, *, complete: bool) -> dict[str, Any]:
        """Build the result record. One builder so the partial and the final cannot drift.

        Two copies of this dict is how a partial ends up missing the field a reader needs.
        """
        return {
            "experiment": "arc_live_oracle_gap"
            if games_mode == "oracle"
            else "arc_leaderboard_eval",
            "games_mode": games_mode,
            "policy": policy_kind,
            "budget": budget,
            "random_seed": seed,
            "live_levels": live_levels_sum if games_mode == "oracle" else total_levels,
            "oracle_levels": oracle_sum,
            "gap": gap_sum,
            "efficiency_sum": round(total_eff, 4),
            "open_gaps": gaps,
            "per_game": rows,
            "complete": complete,
            "games_completed": len(rows),
            "games_planned": len(games),
            "inference_substrate": "offline_sim_no_quota_frame_only_live_agent",
            "honest_verdict": verdict,
        }

    # BANK EACH GAME AS IT FINISHES (2026-09-01).
    #
    # The final write below runs after the whole loop. A two-game run at roughly 5.4 hours per
    # game therefore produced nothing for about 11 hours, and a crash at hour 10 lost both
    # games -- including every trajectory-supervisor receipt they had accrued, which is the
    # evidence such runs exist to gather.
    #
    # The partial ALWAYS goes to the run-scoped, gitignored directory, never to the tracked
    # sweep file, even on a full sweep. Writing a partial to the tracked path would replace a
    # complete sweep with an in-progress one -- the exact record loss that commit f2b82c89a6's
    # 25-game sweep suffered in 2026-08 and that the block below this loop exists to prevent.
    _sweep_name = (
        "arc_live_oracle_gap.json" if games_mode == "oracle" else "arc_leaderboard_eval.json"
    )
    _partial_dir = REPO / "results" / (_sweep_name[: -len(".json")] + "_runs")
    _partial_dir.mkdir(parents=True, exist_ok=True)
    _partial = (
        _partial_dir
        / f"{'-'.join(sorted(only.split(','))) if only else 'sweep'}-{os.getpid()}.partial.json"
    )

    # REQ-ARC-WMTE-7010: one heartbeat file per run, rewritten for the game in flight. The
    # games already finished live in the partial; this file is the one that moves within a game.
    _progress_path = _partial.with_name(_partial.name[: -len(".partial.json")] + ".progress.json")
    for _game_index, game in enumerate(games, start=1):
        t0 = time.time()
        _policy = _build_policy(policy_kind, game)
        _progress = ProgressWriter(
            _progress_path,
            game=game,
            game_index=_game_index,
            games_planned=len(games),
            policy=_policy,
        )
        r = run_game(
            game, _policy, budget=budget, variant=variant, reflect=reflect, progress=_progress
        )
        if games_mode == "oracle":
            r["oracle_levels"] = oracle.get(game, 0)
            r["gap_vs_oracle"] = max(0, oracle.get(game, 0) - r["levels"])
            live_levels_sum += r["levels"]
            oracle_sum += r["oracle_levels"]
            gap_sum += r["gap_vs_oracle"]
        rows.append(r)
        total_levels += r["levels"]
        total_eff += r["efficiency"]
        if r["gap"]:
            gaps.append(r["gap"])
        extra = (
            f" vs oracle L{r['oracle_levels']} (gap {r['gap_vs_oracle']})"
            if games_mode == "oracle"
            else ""
        )
        print(
            f"  {game:5} live=L{r['reached']} (+{r['levels']}) actions={r['actions']:5} "
            f"eff={r['efficiency']:.4f} nav_reset={r['reset_replay_steps']} "
            f"nav_fwhr={r['forward_walk_hit_rate']:.4f}{extra}  [{time.time() - t0:.0f}s]",
            flush=True,
        )
        _write_json_atomic(
            _partial,
            json.dumps(
                _payload(
                    f"partial_{len(rows)}_of_{len(games)}_games_run_in_progress",
                    complete=False,
                ),
                indent=2,
            ),
        )
    if games_mode == "oracle":
        print(
            f"\n  LIVE-vs-ORACLE GAP: frame-only live path reaches {live_levels_sum}/{oracle_sum} "
            f"oracle levels (gap {gap_sum}). Closed: "
            f"{sorted(g for g in games if oracle.get(g, 0) and rows[games.index(g)]['levels'] >= oracle[g])}",
            flush=True,
        )
    else:
        print(
            f"\n  LEADERBOARD SCORE: {total_levels} levels, efficiency-sum {total_eff:.3f}; "
            f"{len(gaps)} open gaps",
            flush=True,
        )
    verdict = (
        f"complete_live_oracle_gap_{live_levels_sum}_of_{oracle_sum}_levels_gap_{gap_sum}"
        if games_mode == "oracle"
        else f"complete_leaderboard_eval_{total_levels}_levels_{len(gaps)}_gaps"
    )
    # WHERE THIS RUN'S RECORD GOES, AND WHY IT IS NOT ALWAYS THE TRACKED FILE (2026-08-17).
    #
    # These two tracked files hold a FULL-SWEEP record. This script used to overwrite them on
    # every completed run, with no flag and no opt-out, so a `--only <game>` run replaced a
    # 25-game sweep with a single row. That is not hypothetical: the real 25-game sweep from
    # 2026-07-19 (f2b82c89a6) was destroyed that way and the 1-row remnant was committed on
    # 2026-08-07 by adada4146e, losing the record for ten days. No guard could see it, because
    # the writer is the sanctioned measurement harness rather than a test.
    #
    # Worse, the submission gate runs this script for 8 games through a thread pool, so every
    # completing subprocess was a concurrent writer to one path, non-atomically.
    #
    # So: a subset run writes a RUN-SCOPED file and leaves the sweep alone. Only a genuine full
    # sweep may claim the tracked path, and it does so atomically.
    _is_subset = bool(only)
    if _is_subset:
        _runs = REPO / "results" / (_sweep_name[: -len(".json")] + "_runs")
        _runs.mkdir(parents=True, exist_ok=True)
        _tag = "-".join(sorted(only.split(","))) if only else "subset"
        out = _runs / f"{_tag}-{os.getpid()}.json"
    else:
        out = REPO / "results" / _sweep_name
    _write_json_atomic(out, json.dumps(_payload(verdict, complete=True), indent=2))
    print(f"  wrote {out.relative_to(REPO)}", flush=True)
    # The crash-insurance copy has served its purpose; the real record now exists. Removing it
    # stops a later reader globbing this directory from finding a stale partial beside a
    # finished run.
    _partial.unlink(missing_ok=True)
    # The heartbeat has the same lifetime: every game's final state is in the record now.
    _progress_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
