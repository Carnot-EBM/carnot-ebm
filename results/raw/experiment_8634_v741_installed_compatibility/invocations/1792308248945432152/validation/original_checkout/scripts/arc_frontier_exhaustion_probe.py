#!/usr/bin/env python3
"""Diagnose whether ARC frontier exhaustion is PREMATURE or GENUINE.

REQ-ARC-WMTE-6082 (diagnostic; no shipped default is changed by this file).

WHY THIS EXISTS. In ``results/arc_inert_label_defer_20260802`` game g50t stopped at 1,022 of a
2,000-action budget with ``timed_out: false`` -- the largest single "action saving" in the run,
and it was scored as a win. An agent that stops early because it ran out of PLACES TO GO has not
saved actions; it has stopped working. The two cases are indistinguishable from the cell artifact,
because the artifact records only how many actions were spent, never whether reachable unexplored
work remained at the moment the run ended.

WHAT IT MEASURES, and why counting beats reasoning about the code. ``StepwiseExplorer.next_move``
sets ``explored_out = True`` (and the worker loop then breaks on ``kind is None``) exactly when
``_frontier()`` returns ``None``, i.e. when its ``eligible`` list is empty. ``eligible`` can be
empty for FOUR structurally different reasons, and only the first is genuine exhaustion:

  1. every node really has an empty ``untested`` list  -- GENUINE;
  2. nodes hold untested rows the GLOBAL TIER BARRIER (REQ-ARC-WMTE-5836) refuses at the active
     tier -- PREMATURE unless the barrier truly cannot advance;
  3. the online discriminator pruned every eligible node (``on_path <
     discriminative_prune_threshold``) -- PREMATURE, and note this branch has NO never-empty guard,
     unlike the hazard pruner's explicit one at ``_pop_untested``'s row build;
  4. the graph is empty / some other degenerate state.

So this probe monkey-patches ``_frontier`` to snapshot the graph the moment it is about to return
``None``, and counts, directly:

  * nodes and untested ROWS remaining (a node with rows is a place the agent could still go);
  * how many of those nodes are TIER-DEFERRED (``_node_is_tier_deferred``) vs genuinely empty;
  * how many were DISCRIMINATIVE-PRUNED;
  * the active tier and whether the barrier could still advance;
  * of the nodes that still hold rows, how many are REACHABLE from the current node over the
    edges the agent itself recorded -- because an untested row on an unreachable node is not a
    place the agent could actually have gone, and counting it would overstate the bug.

Reachability is computed over the explorer's OWN forward edges plus the RESET-replay route every
node carries (``node["path"]`` is replayable from a reset by construction), which is exactly the
navigation the agent has available at ``_frontier``'s call site.

NEVER plays a scored or online game: ``kit.offline_arcade()`` runs OFFLINE over the local
``environment_files/`` tree. No generator, no GPU, no network. Reuses the shipped worker's
``run_cell`` configuration so the run under the microscope is the SAME run the A/B measured --
this file only observes.

EVIDENCE GUARD: refuses to start unless ``CARNOT_ARC_E3_DIR`` points away from the tracked
``results/arc_e3`` evidence store, same contract as the A/B worker.
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


class _NoGeneratorStandIn:
    """Absorbs the config attribute writes a proposer would receive; constructs nothing."""

    include_playbook_exemplars = False
    no_think_prefix = ""
    max_tokens = 0
    tries = 0


def _reachable_from(explorer: Any, start: Optional[str]) -> set[str]:
    """Nodes reachable from ``start`` over edges the explorer itself recorded.

    Every node also carries ``path`` -- an action sequence replayable from a RESET -- and the
    agent's own navigation falls back to exactly that (``_nav_reset_fallbacks`` in next_move).
    So a node with a path is reachable whether or not a forward edge chain leads to it. We
    therefore report BOTH the strict forward-edge closure and the practical (forward | has-path)
    set, and let the artifact state which is used for the headline count.
    """
    graph = explorer.graph or {}
    # Forward edges live in `explorer.adj` (origin -> [(act, next_hash), ...]), written by the
    # single `_record_forward_edge` path. Only state-CHANGING edges are recorded there, which is
    # exactly the edge set the agent's own `_shortest_path` navigation can use.
    adj: dict[str, set[str]] = {h: set() for h in graph}
    for origin, edges in (getattr(explorer, "adj", None) or {}).items():
        bucket = adj.setdefault(origin, set())
        for _act, nxt in edges:
            if isinstance(nxt, str):
                bucket.add(nxt)
    seen: set[str] = set()
    if start is not None and start in graph:
        dq = deque([start])
        seen.add(start)
        while dq:
            cur = dq.popleft()
            for nxt in adj.get(cur, ()):  # pragma: no branch
                if nxt not in seen:
                    seen.add(nxt)
                    dq.append(nxt)
    return seen


def probe(
    game: str,
    *,
    seed: int,
    budget: int,
    wall_s: float,
    defer: bool,
    audit_nodes: bool = False,
) -> dict[str, Any]:
    import random

    import numpy as np
    from arcengine import GameAction

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of  # noqa: F401
    from carnot.agentic.arc_competition_agent import (
        E3AgentPolicy,
        _level_of,
        _recommend_live_approach,
        _route_explore_budget,
    )
    import carnot.agentic.arc_strategy_router as arc_strategy_router

    random.seed(seed)
    np.random.seed(seed)

    rec = _recommend_live_approach(game)
    strategy = dict(rec.get("strategy") or arc_strategy_router.route_for_game(game))
    explore_budget = int(_route_explore_budget(strategy))

    t0 = time.time()
    pol = E3AgentPolicy(game, proposer=_NoGeneratorStandIn(), explore_budget=explore_budget)
    explorer = pol.explorer

    # Count the candidate rows the search MINTS, independently of how many it later consumes.
    # "The frontier was empty" is only evidence of genuine exhaustion if the rows were spent;
    # if far fewer were ever created, the graph closed for a supply reason, not a search one.
    candidate_mint: dict[str, int] = {}
    orig_candidates = explorer._candidates

    def _counting_candidates(frame: Any, *a: Any, **kw: Any) -> Any:
        rows = orig_candidates(frame, *a, **kw)
        try:
            candidate_mint["rows_minted"] = candidate_mint.get("rows_minted", 0) + len(rows)
            candidate_mint["nodes_minted"] = candidate_mint.get("nodes_minted", 0) + 1
        except Exception:
            pass
        return rows

    explorer._candidates = _counting_candidates  # type: ignore[method-assign]

    # NAVIGATION ARRIVAL. A row is removed from node X's `untested` at the moment the search
    # DECIDES to probe X, but the action only executes after a navigation walk/replay reaches X.
    # If the walk lands somewhere else, the probe fires from the wrong state while X's row is
    # already spent -- X then reads as exhausted having never actually been probed. `_serve`
    # carries the intended origin, so comparing it to `self.cur` at serve time measures exactly
    # this, with no assumption about why a walk might miss.
    nav = {"probes": 0, "probes_with_origin": 0, "arrived": 0, "missed": 0}
    missed_origins: dict[str, int] = {}
    nav_samples: list[dict[str, Any]] = []
    orig_serve = explorer._serve

    def _instrumented_serve() -> tuple:
        peek = explorer.pending[0] if explorer.pending else None
        cur_before = explorer.cur
        out = orig_serve()
        if isinstance(peek, dict) and peek.get("probe"):
            nav["probes"] += 1
            if "origin" in peek:
                nav["probes_with_origin"] += 1
                intended = peek.get("origin")
                hit = intended == cur_before
                nav["arrived" if hit else "missed"] += 1
                if not hit and isinstance(intended, str):
                    missed_origins[intended] = missed_origins.get(intended, 0) + 1
                if len(nav_samples) < 40:
                    g = explorer.graph or {}
                    nav_samples.append(
                        {
                            "intended_origin": intended,
                            "actual_cur": cur_before,
                            "arrived": hit,
                            "intended_in_graph": intended in g,
                            "actual_in_graph": cur_before in g,
                            "intended_depth": len((g.get(intended) or {}).get("path") or [])
                            if intended in g
                            else None,
                            "actual_depth": len((g.get(cur_before) or {}).get("path") or [])
                            if cur_before in g
                            else None,
                            "intended_untested_left": len(
                                (g.get(intended) or {}).get("untested") or []
                            )
                            if intended in g
                            else None,
                        }
                    )
        return out

    explorer._serve = _instrumented_serve  # type: ignore[method-assign]

    snapshots: list[dict[str, Any]] = []
    orig_frontier = explorer._frontier

    def _instrumented_frontier() -> Optional[str]:
        out = orig_frontier()
        if out is not None:
            return out
        # About to report "nothing left". Snapshot the graph BEFORE the caller flips
        # explored_out, and classify every node by WHY it is not eligible.
        graph = explorer.graph or {}
        nodes_with_rows = 0
        rows_remaining = 0
        tier_deferred_nodes = 0
        tier_deferred_rows = 0
        disc_pruned_nodes = 0
        disc_pruned_rows = 0
        open_tier_nodes = 0
        open_tier_rows = 0
        for h, node in graph.items():
            rows = node.get("untested") or []
            if rows:
                nodes_with_rows += 1
                rows_remaining += len(rows)
            if node.get("discriminative_pruned") is True:
                disc_pruned_nodes += 1
                disc_pruned_rows += len(rows)
            try:
                deferred = bool(explorer._node_is_tier_deferred(node))
            except Exception:
                deferred = False
            if deferred:
                tier_deferred_nodes += 1
                tier_deferred_rows += len(rows)
            try:
                if explorer._node_has_open_tier(node):
                    open_tier_nodes += 1
                    open_tier_rows += len(rows)
            except Exception:
                pass

        fwd_reach = _reachable_from(explorer, explorer.cur)
        # A node with a recorded `path` is replayable from RESET, which is the agent's own
        # navigation fallback -- so it is practically reachable even with no forward-edge chain.
        practical_reach = {
            h for h, n in graph.items() if h in fwd_reach or (n.get("path") is not None)
        }
        reachable_rows = sum(
            len(graph[h].get("untested") or []) for h in practical_reach if graph[h].get("untested")
        )
        reachable_nodes_with_rows = sum(1 for h in practical_reach if graph[h].get("untested"))
        fwd_reachable_rows = sum(
            len(graph[h].get("untested") or []) for h in fwd_reach if graph[h].get("untested")
        )

        try:
            tier_active = bool(explorer._tier_active())
        except Exception:
            tier_active = False
        active_tier = getattr(explorer, "_active_tier", None)
        # Could the barrier still advance? Ask the policy directly with the current graph.
        next_tier = None
        try:
            if tier_active:
                next_tier = explorer.tier_policy.next_active_tier(
                    (n.get("untested") or [] for n in graph.values()), explorer._active_tier
                )
        except Exception:
            next_tier = None

        snapshots.append(
            {
                "at_action": len(snapshots),
                "n_nodes": len(graph),
                "nodes_with_untested_rows": nodes_with_rows,
                "untested_rows_remaining": rows_remaining,
                "nodes_open_at_active_tier": open_tier_nodes,
                "rows_open_at_active_tier": open_tier_rows,
                "tier_deferred_nodes": tier_deferred_nodes,
                "tier_deferred_rows": tier_deferred_rows,
                "discriminative_pruned_nodes": disc_pruned_nodes,
                "discriminative_pruned_rows": disc_pruned_rows,
                "tier_barrier_active": tier_active,
                "active_tier": active_tier,
                "tier_policy_next_tier": next_tier,
                "online_discriminator_trained": explorer.online_discriminator is not None,
                "discriminative_prune_threshold": float(explorer.discriminative_prune_threshold),
                "cur_node": explorer.cur,
                "forward_reachable_nodes": len(fwd_reach),
                "forward_reachable_untested_rows": fwd_reachable_rows,
                "practically_reachable_nodes_with_rows": reachable_nodes_with_rows,
                "practically_reachable_untested_rows": reachable_rows,
            }
        )
        return out

    explorer._frontier = _instrumented_frontier  # type: ignore[method-assign]

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())

    frames: list[Any] = []
    observed_grids: list[Any] = []
    latest: Any = None
    n_actions = 0
    best_level = 0
    start_level: Optional[int] = None
    stop_reason = "budget_exhausted"

    for _ in range(budget):
        if time.time() - t0 > wall_s:
            stop_reason = "wall_clock"
            break
        if pol.is_done(frames, latest):
            stop_reason = (
                "explored_out" if getattr(explorer, "explored_out", False) else "is_done_other"
            )
            break
        kind, data = pol.next_move(frames, latest)
        if kind is None:
            stop_reason = "next_move_returned_none"
            break
        if kind == "RESET":
            latest = env.reset()
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
        frames.append(latest)
        try:
            from carnot.agentic.arc_agi3_world_model import grid_of as _go0

            observed_grids.append(_go0(latest).copy())
        except Exception:
            pass
        n_actions += 1
        lvl = _level_of(latest)
        if start_level is None:
            start_level = lvl
        best_level = max(best_level, lvl)

    # NODE AUDIT (the direct count the diagnosis turns on). For every node the search believes
    # it explored, replay that node's OWN recorded path from a fresh reset in a SEPARATE env and
    # ask two questions the graph cannot answer about itself:
    #   (a) does the path actually land on the hash it is filed under?  (bookkeeping soundness)
    #   (b) from the state it lands on, does any action in the game's OWN advertised vocabulary
    #       lead to a state the search never recorded?  (were there reachable untested pairs)
    # This is ONE PLY from already-visited states, using only paths the agent itself wrote down
    # and the action list the frame itself advertises. It is not a ground-truth BFS, not a
    # solve, and banks no level -- it is an audit of the search's own bookkeeping.
    audit: dict[str, Any] = {}
    if audit_nodes:
        from carnot.agentic.arc_agi3_live_adapter import _available_action_ids

        # MASK-CONSISTENCY, and why this audit is built the way it is. `explorer._hash` is NOT a
        # fixed function of a frame: it masks with `self.hud_mask`, which on this game starts
        # UNRESOLVED (0 cells) and is only inferred partway through the run (measured: 0 -> 64
        # cells, source `status_bar_classifier_req5583...`). So a node keyed early carries an
        # UNMASKED hash while a node keyed late carries a MASKED one, and re-hashing anything
        # after the run compares against a function that did not exist when most keys were
        # written. A first version of this audit did exactly that and reported "130 of 142 node
        # paths do not replay to their own hash" -- an artifact of the moving mask, NOT a defect,
        # and it is recorded here rather than quietly deleted because it is the shape of mistake
        # this whole diagnosis is about.
        #
        # The fix is to compare like with like: re-key EVERY node by replaying its own recorded
        # path and hashing the landed frame with the FINAL mask, then expand one ply and hash
        # successors with that same final mask. Both sides of the comparison then use one
        # identity function, so mask evolution cannot manufacture a difference. The collapse
        # guard is confirmed inert here (`split_node_count: 0`), so it cannot skew this either.
        env2 = arc.make(game, scorecard_id=arc.open_scorecard())
        final_mask = explorer.hud_mask

        def canon(frame: Any) -> str:
            from carnot.agentic.arc_agi3_world_model import frame_hash as _fh, grid_of as _go

            g = _go(frame)
            if final_mask is not None and getattr(final_mask, "shape", None) == g.shape:
                m = g.copy()
                m[final_mask] = 0
                return _fh(m)
            return _fh(g)

        def canon_grid(g: Any) -> str:
            from carnot.agentic.arc_agi3_world_model import frame_hash as _fh

            if final_mask is not None and getattr(final_mask, "shape", None) == g.shape:
                m = g.copy()
                m[final_mask] = 0
                return _fh(m)
            return _fh(g)

        known_now_extra: set[str] = set()

        def replay_to(p: list) -> Any:
            f = env2.reset()
            for step in p:
                f = env2.step(
                    getattr(GameAction, f"ACTION{int(step['action'])}"), data=step.get("data")
                )
            return f

        paths = [(h, list(node.get("path") or [])) for h, node in (explorer.graph or {}).items()]
        landed: dict[str, str] = {}
        for h, p in paths:
            try:
                landed[h] = canon(replay_to(p))
            except Exception:
                pass
        # `known_now` must be every state the agent ACTUALLY OBSERVED during the run, not merely
        # the endpoints of node paths. A successor reached transiently mid-walk was seen by the
        # agent even if it never became a node, and counting it as "unrecorded" would inflate the
        # defect. `observed_grids` holds the raw grid of every frame the run produced, re-hashed
        # here under the same final mask -- so the comparison is strictly conservative.
        for _g in observed_grids:
            try:
                known_now_extra.add(canon_grid(_g))
            except Exception:
                pass
        known_now = set(landed.values()) | known_now_extra

        nodes_with_escape = 0
        escape_pairs = 0
        escape_targets: set[str] = set()
        for h, p in paths:
            if h not in landed:
                continue
            try:
                f = replay_to(p)
            except Exception:
                continue
            found = 0
            for a in sorted(_available_action_ids(f)):
                if int(a) == 6:
                    continue  # click needs coordinates; g50t advertises no click action
                try:
                    f2 = env2.step(getattr(GameAction, f"ACTION{int(a)}"), data=None)
                except Exception:
                    continue
                if canon(f2) not in known_now:
                    found += 1
                    escape_targets.add(canon(f2))
                f = replay_to(p)  # restore the antecedent state for the next action
            if found:
                nodes_with_escape += 1
                escape_pairs += found
        audit = {
            "identity_function": "final_hud_mask_applied_uniformly_to_both_sides",
            "final_hud_mask_digest": (explorer.hud_mask_diagnostics().get("hud_mask_digest")),
            "final_hud_mask_cell_count": (
                explorer.hud_mask_diagnostics().get("hud_mask_cell_count")
            ),
            "collapse_guard_split_node_count": (
                (explorer.hud_mask_diagnostics().get("collapse_guard") or {}).get(
                    "split_node_count"
                )
            ),
            "distinct_states_actually_observed_during_run": len(known_now_extra),
            "nodes_replayed": len(landed),
            "distinct_states_after_recanonicalisation": len(known_now),
            "nodes_with_an_unrecorded_successor": nodes_with_escape,
            "reachable_untested_state_action_pairs": escape_pairs,
            "distinct_unrecorded_successor_states": len(escape_targets),
        }

    return {
        "game": game,
        "seed": seed,
        "arm": "defer" if defer else "control",
        "node_audit": audit,
        "flag_env": os.environ.get("CARNOT_ARC_INERT_LABEL_DEFER"),
        "budget": budget,
        "actions_spent": n_actions,
        "best_level": best_level,
        "start_level": start_level,
        "states_discovered": len(explorer.graph or {}),
        "explored_out": bool(getattr(explorer, "explored_out", False)),
        "stop_reason": stop_reason,
        "n_exhaustion_snapshots": len(snapshots),
        "exhaustion_snapshots": snapshots,
        "candidate_rows_minted": candidate_mint.get("rows_minted", 0),
        "candidate_nodes_minted": candidate_mint.get("nodes_minted", 0),
        "nav_arrival": dict(nav),
        "nav_missed_origin_nodes": len(missed_origins),
        "nav_samples": nav_samples,
        "nav_diagnostics": (
            explorer.navigation_diagnostics() if hasattr(explorer, "navigation_diagnostics") else {}
        ),
        # The agent's OWN view of the graph -- node hashes, the depth of each, and the forward
        # edges it recorded. Comparing these across two arms answers "was the smaller graph a
        # closed subgraph, or a truncated one" using only what the two runs themselves observed:
        # no ground-truth BFS, no game-source reading, no per-game calibration.
        "graph_nodes": sorted(explorer.graph or {}),
        "graph_node_depth": {
            h: len(n.get("path") or []) for h, n in (explorer.graph or {}).items()
        },
        "forward_edges": {
            origin: sorted({nxt for _a, nxt in edges})
            for origin, edges in (getattr(explorer, "adj", None) or {}).items()
        },
        "wall_s": round(time.time() - t0, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--wall-s", type=float, default=900.0)
    ap.add_argument("--audit-nodes", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from carnot.agentic.arc_executable_world_model import E3_DIR, _TRACKED_E3_EVIDENCE_DIR

    if E3_DIR.resolve() == _TRACKED_E3_EVIDENCE_DIR.resolve():
        print(
            "[probe] REFUSING: E3_DIR resolves to the tracked evidence store "
            f"({_TRACKED_E3_EVIDENCE_DIR}). Set CARNOT_ARC_E3_DIR to scratch first.",
            file=sys.stderr,
        )
        return 2
    if os.environ.get("CARNOT_ARC_DISABLE_INDUCTION") != "1":
        print("[probe] REFUSING: CARNOT_ARC_DISABLE_INDUCTION must be 1.", file=sys.stderr)
        return 2

    defer = os.environ.get("CARNOT_ARC_INERT_LABEL_DEFER") == "1"
    out = probe(
        args.game,
        seed=args.seed,
        budget=args.budget,
        wall_s=args.wall_s,
        defer=defer,
        audit_nodes=bool(args.audit_nodes),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    s = out["exhaustion_snapshots"][-1] if out["exhaustion_snapshots"] else {}
    print(
        f"[probe] {out['arm']}/{out['game']}/{out['seed']}: {out['actions_spent']} actions, "
        f"stop={out['stop_reason']}, states={out['states_discovered']}, "
        f"rows_left={s.get('untested_rows_remaining')}, "
        f"reachable_rows_left={s.get('practically_reachable_untested_rows')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
