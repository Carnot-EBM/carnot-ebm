#!/usr/bin/env python3
"""Where do the NON-EXPANDING actions of a live ARC exploration run actually go?

**The measured fact this instrument decomposes.** The 2026-08-01 live action census on
tn36 (`results/outer_loop_arc_action_provenance_tn36_*.json`) found that of 240 actions
the SCORED policy spent, 44 (18.3%) expanded anything new and 177 (73.8%) were navigation
or replay back to already-visited states. The LLM-OFF arm has the same shape (45 / 194),
so the overhead is a property of the SEARCH, not of the generator. This script answers the
obvious follow-up the census could not: **which named class does each of those overhead
actions belong to, and what would have to change for it not to be spent?**

**Why it matters in the benchmark's own units.** ARC-AGI-3 scores a level as
`min((baseline_actions / agent_actions)**2, 115)`. The cost is QUADRATIC in actions, so a
mechanism that halves the overhead is worth ~4x on per-level score. That is a larger effect
size than anything else measured in this session, and it is LLM-independent.

**The corpus is NOT solve-conditioned, and that is the whole design constraint.** Every
window built through `arc_actions_to_progress.build_progress_window` requires the game to
be ALREADY SOLVED to L1 through a registered `GameAdapter`, and then shows a prefix of a
banked WINNING route -- a re-navigation measurement built on that inherits its own answer.
This driver instead runs `E3AgentPolicy` (the scored policy) from a FRESH RESET on the
offline arcade with a bounded action budget, exactly as `run_bounded_progress` does. The
agent does not know the answer, does not get an adapter, and in these runs does not solve
the game; the trace is what the live agent really produces on a game it has not cracked.

**Induction is DISABLED and no generator is constructed.** `CARNOT_ARC_DISABLE_INDUCTION=1`
short-circuits the induce tier before any proposer method is called. Two reasons: (1) the
census already showed the overhead shape is identical with the LLM on and off, so the
generator is not the object of study; (2) with no sampling in the loop the run is
deterministic, which is what makes a per-action accounting auditable rather than a
one-sample anecdote. No GPU is touched and no llama-server is started.

**Nothing under `python/carnot/agentic/` is modified.** Every quantity here is read from
attributes the explorer already maintains (`cur`, `graph`, `adj`, `pending`, the `_nav_*`
counters, `navigation_diagnostics()`) and from provenance labels (`_prov_branch`,
`_prov_serve_kind`) that `next_move` assigns unconditionally as bare constants. The
instrument is pure observation between decisions: it cannot change what the agent chooses.

**Evidence guard.** `results/arc_e3/<game>/world_model.py` is TRACKED, READ-ONLY evidence,
and the module guard that would catch a write to it is pytest-scoped -- so a measurement
driver is precisely the caller nothing protects. This process refuses to start unless
`CARNOT_ARC_E3_DIR` has already been redirected to scratch, checked AFTER import because
`E3_DIR` is resolved at import time.

Never plays a scored or online game: `arc_solver_kit.offline_arcade()` runs
`OperationMode.OFFLINE` over the local `environment_files/` tree -- no API key, no network,
no scorecard submission.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Any, Optional

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)


# --------------------------------------------------------------------------------------
# Action classes. Each is a bucket in the final accounting, and each carries the mechanism
# that would have to change for those actions not to be spent. The classes are derived from
# the code paths in `StepwiseExplorer.next_move` / `_serve` and from the traces, not from a
# guessed taxonomy: every one of them corresponds to an identifiable return site or to a
# structural feature of the RESET-replay plan the explorer enqueues.
# --------------------------------------------------------------------------------------
CLASS_BOOTSTRAP_RESET = "bootstrap.reset"
CLASS_EXPANSION_NEW_NODE = "expansion.probe_discovered_new_state"
CLASS_EXPANSION_INERT = "expansion.probe_was_inert_frame_unchanged"
CLASS_EXPANSION_KNOWN_STATE = "expansion.probe_revisited_known_state"
CLASS_NAV_RESET = "renavigation.reset_action"
CLASS_NAV_REPLAY_SHARED = "renavigation.replay_prefix_shared_with_current_path"
CLASS_NAV_REPLAY_DIVERGENT = "renavigation.replay_suffix_past_divergence"
CLASS_NAV_FORWARD_WALK = "renavigation.forward_walk_no_reset"
CLASS_PLAN_STEP = "plan.execute_step"
CLASS_OTHER = "other"


def _path_common_prefix(a, b) -> int:
    """Length of the longest common prefix of two action paths (list of {action,data})."""

    n = 0
    for x, y in zip(a, b):
        if int(x.get("action")) != int(y.get("action")):
            break
        if (x.get("data") or None) != (y.get("data") or None):
            break
        n += 1
    return n


def _forward_distances(adj, src: Optional[str]) -> dict[str, int]:
    """BFS hop-distance over the explorer's OWN recorded forward edges, from `src`.

    Mirrors `_exact_shortest_path`'s reachability exactly (same `adj`, same forward-only
    edges), so a node this reports as reachable at distance d is one the agent's own
    `_shortest_path` could have walked to in d actions with no RESET.
    """

    if src is None:
        return {}
    dist = {src: 0}
    q = deque([src])
    while q:
        node = q.popleft()
        for _act, nxt in adj.get(node, []) or []:
            if nxt in dist:
                continue
            dist[nxt] = dist[node] + 1
            q.append(nxt)
    return dist


def _snapshot_frontier(explorer) -> dict[str, Any]:
    """Cheap pre-decision snapshot: every node the frontier would admit, and its depth.

    Used for the LOCALITY counterfactual -- what the cheapest eligible frontier node would
    have cost, versus what the depth-primary ordering actually chose. Deliberately does NOT
    call `_frontier()`: that method MUTATES (it advances the tier barrier and records
    discriminator training samples), and an instrument that mutates the thing it measures is
    not an instrument. Eligibility is instead tested with `_node_has_open_tier`, which is
    `_frontier`'s own per-node filter and is pure. The one filter not replicated is the
    online-discriminator prune, which is inactive whenever `online_discriminator is None`
    (its state at every decision in these runs, recorded per row so the claim is checkable).

    `deferred_nodes` records the OTHER half of that test: nodes that still hold untested
    work which the GLOBAL tier barrier is currently refusing. That distinction is the
    difference between "the agent had nowhere near to go" and "the agent had work exactly
    where it was standing and a barrier sent it away", which are different findings with
    different fixes.
    """

    open_nodes: dict[str, int] = {}
    deferred_nodes: dict[str, int] = {}
    eligible_n: dict[str, int] = {}
    tier_on = bool(explorer._tier_active())
    for h, node in (explorer.graph or {}).items():
        rows = node.get("untested") or []
        if not rows:
            continue
        depth = len(node.get("path") or [])
        if explorer._node_has_open_tier(node):
            open_nodes[h] = depth
            # How many probes ONE navigation to this node could have bought, if the
            # frontier batch were not pinned at 1. Same eligibility rule
            # `_pop_frontier_batch` applies, so this is the real ceiling, not a guess.
            if tier_on:
                try:
                    eligible_n[h] = len(
                        explorer.tier_policy.eligible_indices(rows, explorer._active_tier)
                    )
                except Exception:
                    eligible_n[h] = len(rows)
            else:
                eligible_n[h] = len(rows)
        else:
            deferred_nodes[h] = depth
    return {
        "tier_eligible_n": eligible_n,
        "cur": explorer.cur,
        "cur_path": list((explorer.graph.get(explorer.cur) or {}).get("path") or []),
        "cur_untested_n": len((explorer.graph.get(explorer.cur) or {}).get("untested") or []),
        "cur_open_at_tier": bool(explorer._node_has_open_tier(explorer.graph.get(explorer.cur))),
        "open_nodes": open_nodes,
        "deferred_nodes": deferred_nodes,
        "dist": _forward_distances(explorer.adj, explorer.cur),
        "n_graph": len(explorer.graph or {}),
        "tier_active": bool(explorer._tier_active()),
        "active_tier": int(getattr(explorer, "_active_tier", -1)),
        "online_discriminator_active": explorer.online_discriminator is not None,
    }


def _nav_cost(target_hash: str, depth: int, snap: dict[str, Any]) -> tuple[int, str]:
    """Actions to reach `target_hash` from the snapshot's current state.

    Same two options the explorer itself has: walk forward over known edges (cost = hops,
    no RESET) or RESET and replay the target's root path (cost = 1 + depth). Returns the
    cheaper, with a label.
    """

    d = snap["dist"].get(target_hash)
    reset_cost = 1 + int(depth)
    if d is None:
        return reset_cost, "reset_replay"
    return (int(d), "forward_walk") if int(d) <= reset_cost else (reset_cost, "reset_replay")


def _counterfactual_cheapest_frontier(snap: dict[str, Any]) -> dict[str, Any]:
    """The LOCALITY counterfactual: cost of navigating to the CHEAPEST open frontier node.

    The explorer orders the frontier depth-primary (`_frontier`: `depth` is the first key;
    navigation cost enters only as a late tiebreak within equal depth). After a deep
    depth-first ride, the shallowest open node is by construction far from where the agent
    is -- so depth-primary ordering does not merely fail to optimize navigation, it
    actively maximizes it. This measures what a navigation-aware ordering could have paid
    instead, over the same eligible set.
    """

    best = None
    for h, depth in snap["open_nodes"].items():
        if h == snap["cur"]:
            cost, kind = 0, "already_here"
        else:
            cost, kind = _nav_cost(h, depth, snap)
        if best is None or cost < best[0]:
            best = (cost, h, kind, depth)
    if best is None:
        return {"available": False}
    return {
        "available": True,
        "cost": int(best[0]),
        "node": best[1],
        "kind": best[2],
        "depth": int(best[3]),
        "n_open_nodes": len(snap["open_nodes"]),
    }


def run_game(
    game: str,
    *,
    seed: int,
    budget: int,
    explore_budget: int,
    wall_s: float,
) -> dict[str, Any]:
    """Drive the scored policy from a fresh reset and record every action's provenance."""

    import random

    import numpy as np
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    random.seed(seed)
    np.random.seed(seed)

    class _NoGeneratorStandIn:
        """Absorbs the config attribute writes `apply_arm` makes; constructs nothing."""

        include_playbook_exemplars = False
        no_think_prefix = ""
        max_tokens = 0
        tries = 0

    t0 = time.time()
    pol = E3AgentPolicy(game, proposer=_NoGeneratorStandIn(), explore_budget=explore_budget)
    explorer = pol.explorer
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())

    frames: list[Any] = []
    latest: Any = None
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    open_episode: Optional[dict[str, Any]] = None
    start_level: Optional[int] = None
    best_level = 0
    timed_out = False

    def _counters() -> dict[str, int]:
        return {
            k: int(getattr(explorer, k, 0))
            for k in (
                "_nav_attempts",
                "_nav_exact_hits",
                "_nav_partial_hits",
                "_nav_similarity_hits",
                "_nav_reset_fallbacks",
                "_nav_forward_steps",
                "_nav_reset_replay_steps",
            )
        }

    for i in range(budget):
        if time.time() - t0 > wall_s:
            timed_out = True
            break
        if pol.is_done(frames, latest):
            break

        snap = _snapshot_frontier(explorer)
        pre = _counters()
        pre_nodes = len(explorer.graph or {})

        kind, data = pol.next_move(frames, latest)
        if kind is None:
            rows.append(
                {
                    "i": i,
                    "action": None,
                    "top_branch": getattr(pol, "_prov_top", None),
                    "explorer_branch": getattr(explorer, "_prov_branch", None),
                    "serve_kind": getattr(explorer, "_prov_serve_kind", None),
                    "terminal": True,
                }
            )
            break

        post = _counters()
        branch = getattr(explorer, "_prov_branch", None)
        serve_kind = getattr(explorer, "_prov_serve_kind", None)
        top = getattr(pol, "_prov_top", None)

        # ---- episode bookkeeping -------------------------------------------------------
        # A `frontier.navigate` return is the START of a navigation episode: it enqueues the
        # whole plan (RESET + replay, or a forward walk) followed by the probe batch, and
        # serves the first item. The remaining items drain through `pending_drain` on
        # subsequent turns. Reading `explorer.pending` right after the navigate call is
        # therefore a complete, exact statement of what the episode will cost -- no
        # inference required, which is the whole point.
        if branch == "frontier.navigate":
            pend = [dict(it) for it in (explorer.pending or [])]
            plan = [{"kind": kind, "data": data, "probe": serve_kind == "probe"}] + pend
            target = None
            for it in pend:
                if it.get("origin"):
                    target = it["origin"]
                    break
            tgt_path = list((explorer.graph.get(target) or {}).get("path") or [])
            used_reset = bool(plan and plan[0].get("kind") == "RESET")
            n_probe = sum(1 for it in plan if it.get("probe"))
            n_nav = sum(1 for it in plan if not it.get("probe") and it.get("kind") != "RESET")
            shared = _path_common_prefix(snap["cur_path"], tgt_path)
            mech = (
                "exact_shortest_path"
                if post["_nav_exact_hits"] > pre["_nav_exact_hits"]
                else (
                    "similarity_forward_walk"
                    if post["_nav_similarity_hits"] > pre["_nav_similarity_hits"]
                    else (
                        "partial_forward_walk"
                        if post["_nav_partial_hits"] > pre["_nav_partial_hits"]
                        else (
                            "reset_replay_fallback"
                            if post["_nav_reset_fallbacks"] > pre["_nav_reset_fallbacks"]
                            else "unknown"
                        )
                    )
                )
            )
            cheapest = _counterfactual_cheapest_frontier(snap)
            open_episode = {
                "start_index": i,
                "mechanism": mech,
                "used_reset": used_reset,
                "n_reset_actions": 1 if used_reset else 0,
                "n_nav_actions": n_nav,
                "n_probe_actions": n_probe,
                "episode_actions": len(plan),
                "src_depth": len(snap["cur_path"]),
                "target_depth": len(tgt_path),
                "shared_prefix_with_current_path": shared,
                # The target's whole path is a prefix of the path the agent is standing on:
                # i.e. the agent already WALKED THROUGH this node earlier in this very run,
                # and the entire replay is retracing its own past route back to it.
                "target_is_ancestor_of_current": (
                    shared == len(tgt_path) and len(tgt_path) < len(snap["cur_path"])
                ),
                "n_open_frontier_nodes": len(snap["open_nodes"]),
                "n_tier_deferred_nodes": len(snap["deferred_nodes"]),
                # The agent LEFT a node that still had untested work, because the global
                # tier barrier was refusing that work -- a departure the search chose, not
                # one the state space forced.
                "departed_node_with_deferred_work": bool(
                    snap["cur_untested_n"] > 0 and not snap["cur_open_at_tier"]
                ),
                "cur_untested_n": int(snap["cur_untested_n"]),
                # The ceiling on probes this ONE navigation could have bought if the batch
                # were not pinned at 1 (`SUBMITTED_FRONTIER_BATCH_SIZE`). See the artifact's
                # amortization note for why raising it is not a free lunch.
                "tier_eligible_rows_at_target": int(snap["tier_eligible_n"].get(target, 0)),
                "tier_active": snap["tier_active"],
                "active_tier": snap["active_tier"],
                "online_discriminator_active": snap["online_discriminator_active"],
                "cheapest_eligible": cheapest,
                "graph_nodes_at_decision": snap["n_graph"],
            }
            episodes.append(open_episode)

        # ---- per-action class ----------------------------------------------------------
        if kind == "RESET":
            latest = env.reset()
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
        frames.append(latest)
        # RAW frame identity, computed WITHOUT the explorer's HUD mask. The masked node id
        # answers "is this a new node to the search"; this answers the strictly cheaper
        # question "did anything on the screen change at all", which is what separates a
        # genuinely inert action (a click on dead pixels) from a real move that happened to
        # land back on a known state.
        try:
            raw_hash = frame_hash(grid_of(latest)) if latest is not None else None
        except Exception:
            raw_hash = None
        lvl = _level_of(latest)
        if start_level is None:
            start_level = lvl
        best_level = max(best_level, lvl)
        post_nodes = len(explorer.graph or {})

        # Graph growth is only observable on the NEXT turn (the explorer ingests `latest` at
        # the top of `next_move`), so `discovered_new_state` is attributed one turn late and
        # then shifted back below. Recorded as the raw delta here rather than guessed.
        rows.append(
            {
                "i": i,
                "action": "RESET" if kind == "RESET" else int(kind),
                "data": data,
                "top_branch": top,
                "explorer_branch": branch,
                "serve_kind": serve_kind,
                "episode_start": branch == "frontier.navigate",
                "episode_index": (len(episodes) - 1) if open_episode is not None else None,
                "graph_nodes_before_decision": pre_nodes,
                "graph_nodes_after_decision": post_nodes,
                "level": lvl,
                "raw_frame_hash": raw_hash,
                "nav_counter_delta": {k: post[k] - pre[k] for k in pre if post[k] != pre[k]},
            }
        )

        if not explorer.pending:
            open_episode = None

    # Attribute node discovery to the action that CAUSED it. The turn order is:
    #   snapshot pre_nodes -> next_move() [which ingests the frame produced by the PREVIOUS
    #   action, and is the only place the graph grows] -> post_nodes -> env.step().
    # So the node created by action i first becomes visible as growth ACROSS decision i+1,
    # i.e. post_nodes(i+1) > pre_nodes(i+1). Comparing across the gap between turns instead
    # (pre(i+1) vs post(i)) measures a window in which nothing can happen and is therefore
    # always False -- the first version of this line did exactly that and reported zero
    # discoveries on a run that plainly built a 17-node graph.
    for idx in range(len(rows) - 1):
        nxt = rows[idx + 1]
        rows[idx]["discovered_new_state"] = bool(
            nxt.get("graph_nodes_after_decision", 0) > nxt.get("graph_nodes_before_decision", 0)
        )
    if rows:
        rows[-1].setdefault("discovered_new_state", False)

    # Did the SCREEN change at all? Compared against the previous action's raw frame hash,
    # so a probe whose only effect was to leave the board exactly as it found it is
    # separable from one that moved something and happened to arrive at a known state.
    prev_raw = None
    for row in rows:
        raw = row.get("raw_frame_hash")
        row["raw_frame_unchanged"] = bool(
            prev_raw is not None and raw is not None and raw == prev_raw
        )
        prev_raw = raw if raw is not None else prev_raw

    # ---- classify -----------------------------------------------------------------------
    # Within a RESET-replay episode the navigation steps split at the point where the
    # target's path diverges from the path the agent was already standing on: the steps
    # before the split re-walk ground the agent's own current path already covers, the ones
    # after are genuinely new ground toward the target.
    ep_by_index = {e["start_index"]: e for e in episodes}
    active: Optional[dict[str, Any]] = None
    nav_seen = 0
    for row in rows:
        if row.get("action") is None:
            row["action_class"] = CLASS_OTHER
            continue
        if row.get("episode_start"):
            active = ep_by_index.get(row["i"])
            nav_seen = 0
        branch = row.get("explorer_branch")
        serve = row.get("serve_kind")
        top = row.get("top_branch") or ""
        if branch == "bootstrap_reset":
            row["action_class"] = CLASS_BOOTSTRAP_RESET
        elif top.startswith("execute."):
            row["action_class"] = CLASS_PLAN_STEP
        elif serve == "reset" or row.get("action") == "RESET":
            row["action_class"] = CLASS_NAV_RESET
        elif serve == "probe" or branch in (
            "depth_ride.pop_untested",
            "frontier.pop_untested",
        ):
            if row.get("discovered_new_state"):
                row["action_class"] = CLASS_EXPANSION_NEW_NODE
            elif row.get("raw_frame_unchanged"):
                row["action_class"] = CLASS_EXPANSION_INERT
            else:
                row["action_class"] = CLASS_EXPANSION_KNOWN_STATE
        elif serve == "navigation":
            if active is not None and active.get("used_reset"):
                shared = int(active.get("shared_prefix_with_current_path") or 0)
                row["action_class"] = (
                    CLASS_NAV_REPLAY_SHARED if nav_seen < shared else CLASS_NAV_REPLAY_DIVERGENT
                )
            else:
                row["action_class"] = CLASS_NAV_FORWARD_WALK
            nav_seen += 1
        else:
            row["action_class"] = CLASS_OTHER

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("action_class", CLASS_OTHER)] = (
            counts.get(row.get("action_class", CLASS_OTHER), 0) + 1
        )

    nav_diag = {}
    try:
        nav_diag = explorer.navigation_diagnostics()
    except Exception as exc:  # pragma: no cover - diagnostics must never fail the run
        nav_diag = {"error": f"{type(exc).__name__}: {exc}"}

    # LOCALITY prize: actions the depth-primary frontier ordering spent above what the
    # cheapest eligible open node would have cost, summed over episodes.
    locality_actual = 0
    locality_cheapest = 0
    locality_episodes = 0
    for ep in episodes:
        cheap = ep.get("cheapest_eligible") or {}
        if not cheap.get("available"):
            continue
        actual = int(ep["n_reset_actions"]) + int(ep["n_nav_actions"])
        locality_actual += actual
        locality_cheapest += int(cheap["cost"])
        locality_episodes += 1

    nav_actions_total = sum(
        int(ep["n_reset_actions"]) + int(ep["n_nav_actions"]) for ep in episodes
    )
    probes_after_nav = sum(int(ep["n_probe_actions"]) for ep in episodes)
    retrace_nav_actions = sum(
        int(ep["n_reset_actions"]) + int(ep["n_nav_actions"])
        for ep in episodes
        if ep.get("target_is_ancestor_of_current")
    )
    tier_departure_nav_actions = sum(
        int(ep["n_reset_actions"]) + int(ep["n_nav_actions"])
        for ep in episodes
        if ep.get("departed_node_with_deferred_work")
    )
    episode_summary = {
        "navigation_actions_total": nav_actions_total,
        "probe_actions_bought_by_navigation": probes_after_nav,
        "navigation_actions_per_probe": (
            round(nav_actions_total / probes_after_nav, 3) if probes_after_nav else None
        ),
        "episodes_target_is_ancestor_of_current": sum(
            1 for ep in episodes if ep.get("target_is_ancestor_of_current")
        ),
        "navigation_actions_retracing_own_path": retrace_nav_actions,
        "episodes_departed_node_with_tier_deferred_work": sum(
            1 for ep in episodes if ep.get("departed_node_with_deferred_work")
        ),
        "navigation_actions_after_tier_deferred_departure": tier_departure_nav_actions,
        "tier_eligible_rows_at_target_total": sum(
            int(ep.get("tier_eligible_rows_at_target") or 0) for ep in episodes
        ),
        "episodes_by_mechanism": {
            m: sum(1 for ep in episodes if ep.get("mechanism") == m)
            for m in sorted({ep.get("mechanism") for ep in episodes})
        },
        "frontier_batch_size": getattr(explorer, "frontier_batch_size", None),
    }

    return {
        "game": game,
        "seed": seed,
        "budget": budget,
        "explore_budget": explore_budget,
        "actions_recorded": len(rows),
        "start_level": start_level,
        "best_level": best_level,
        "solved": bool(start_level is not None and best_level > start_level),
        "timed_out": timed_out,
        "wall_s": round(time.time() - t0, 3),
        "class_counts": counts,
        "navigation_diagnostics": nav_diag,
        "n_navigation_episodes": len(episodes),
        "episode_summary": episode_summary,
        "episodes": episodes,
        "locality_counterfactual": {
            "episodes_scored": locality_episodes,
            "actual_navigation_actions": locality_actual,
            "cheapest_eligible_actions": locality_cheapest,
            "actions_above_cheapest": locality_actual - locality_cheapest,
        },
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--explore-budget", default="routed")
    ap.add_argument("--wall-s", type=float, default=600.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from carnot.agentic.arc_executable_world_model import E3_DIR, _TRACKED_E3_EVIDENCE_DIR

    if E3_DIR.resolve() == _TRACKED_E3_EVIDENCE_DIR.resolve():
        print(
            "[probe] REFUSING: E3_DIR resolves to the tracked evidence store "
            f"({_TRACKED_E3_EVIDENCE_DIR}). Set CARNOT_ARC_E3_DIR to a scratch directory "
            "BEFORE the interpreter starts.",
            file=sys.stderr,
        )
        return 2
    if os.environ.get("CARNOT_ARC_DISABLE_INDUCTION") != "1":
        print("[probe] REFUSING: CARNOT_ARC_DISABLE_INDUCTION must be 1.", file=sys.stderr)
        return 2

    if isinstance(args.explore_budget, str) and args.explore_budget.strip().lower() == "routed":
        import carnot.agentic.arc_strategy_router as arc_strategy_router
        from carnot.agentic.arc_competition_agent import (
            _recommend_live_approach,
            _route_explore_budget,
        )

        rec = _recommend_live_approach(args.game)
        strategy = dict(rec.get("strategy") or arc_strategy_router.route_for_game(args.game))
        explore_budget = int(_route_explore_budget(strategy))
        provenance = (
            f"routed_as_scored:{strategy.get('name')}:"
            f"uses_goal_distance_heuristic={strategy.get('uses_goal_distance_heuristic')}"
        )
    else:
        explore_budget = int(args.explore_budget)
        provenance = "pinned_by_caller"

    out = run_game(
        args.game,
        seed=args.seed,
        budget=args.budget,
        explore_budget=explore_budget,
        wall_s=args.wall_s,
    )
    out["explore_budget_provenance"] = provenance
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"[probe] {args.game}: {out['actions_recorded']} actions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
